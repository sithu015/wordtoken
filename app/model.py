"""Model service for Myanmar NLP inference."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import Settings
from app.utils import (
    collapse_syllable_predictions,
    infer_pos_tag,
    simple_segment,
    split_text_units,
)


logger = logging.getLogger("wordtoken")
MYANMAR_CODEPOINT_RANGE = ("\u1000", "\u109f")
MYANBERTA_BASE_MODEL = "UCSYNLP/MyanBERTa"
MYANBERTA_RECOMMENDED_MAX_LENGTH = 300

try:
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torchcrf import CRF
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - exercised only when runtime deps are absent.
    torch = None
    hf_hub_download = None
    snapshot_download = None
    nn = None
    pack_padded_sequence = None
    pad_packed_sequence = None
    CRF = None
    AutoConfig = None
    AutoModel = None
    AutoTokenizer = None


def _reverse_padded_sequence(tensor, lengths):
    """Reverse each sequence in a padded batch independently."""
    max_length = tensor.size(1)
    indices = torch.arange(max_length, device=tensor.device).unsqueeze(0)
    indices = indices.expand(lengths.size(0), -1)
    reversed_indices = torch.where(
        indices < lengths.unsqueeze(1),
        lengths.unsqueeze(1) - indices - 1,
        indices,
    )
    gather_index = reversed_indices.unsqueeze(-1).expand_as(tensor)
    return tensor.gather(1, gather_index)


def _build_distance_positions(attention_mask, embedding_size: int):
    """Compute distance-from-end positions for the custom positional embedding."""
    lengths = attention_mask.long().sum(dim=1)
    seq_length = attention_mask.size(1)
    indices = torch.arange(seq_length, device=attention_mask.device).unsqueeze(0)
    distances = lengths.unsqueeze(1) - indices - 1
    distances = distances.clamp(min=0, max=embedding_size - 1)
    return torch.where(attention_mask.bool(), distances, torch.zeros_like(distances))


if nn is not None:

    class JointSegPosModel(nn.Module):
        """Reconstructed transformer + asymmetric BiLSTM + dual CRF network."""

        def __init__(
            self,
            base_config,
            *,
            num_ws_labels: int,
            num_pos_labels: int,
            position_embedding_size: int = 512,
            position_embedding_dim: int = 64,
            forward_hidden_size: int = 256,
            backward_hidden_size: int = 512,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.bert = AutoModel.from_config(base_config)
            if hasattr(self.bert, "pooler"):
                self.bert.pooler = None
            hidden_size = int(base_config.hidden_size)
            lstm_input_size = hidden_size + position_embedding_dim

            self.pos_embedding = nn.Embedding(
                position_embedding_size,
                position_embedding_dim,
            )
            self.lstm_fwd = nn.LSTM(
                lstm_input_size,
                forward_hidden_size,
                batch_first=True,
            )
            self.lstm_bwd = nn.LSTM(
                lstm_input_size,
                backward_hidden_size,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            joint_hidden_size = forward_hidden_size + backward_hidden_size
            self.ws_classifier = nn.Linear(joint_hidden_size, num_ws_labels)
            self.pos_classifier = nn.Linear(joint_hidden_size, num_pos_labels)
            self.ws_ce = nn.CrossEntropyLoss(weight=torch.ones(num_ws_labels))
            self.pos_ce = nn.CrossEntropyLoss(weight=torch.ones(num_pos_labels))
            self.ws_crf = CRF(num_ws_labels, batch_first=True)
            self.pos_crf = CRF(num_pos_labels, batch_first=True)

        def forward(self, input_ids, attention_mask):
            """Produce WS and POS emissions."""
            encoder_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            distance_positions = _build_distance_positions(
                attention_mask,
                self.pos_embedding.num_embeddings,
            )
            distance_features = self.pos_embedding(distance_positions)
            sequence_features = torch.cat([encoder_output, distance_features], dim=-1)

            lengths = attention_mask.long().sum(dim=1)

            packed_forward = pack_padded_sequence(
                sequence_features,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            forward_output, _ = self.lstm_fwd(packed_forward)
            forward_output, _ = pad_packed_sequence(
                forward_output,
                batch_first=True,
                total_length=sequence_features.size(1),
            )

            reversed_features = _reverse_padded_sequence(sequence_features, lengths)
            packed_backward = pack_padded_sequence(
                reversed_features,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            backward_output, _ = self.lstm_bwd(packed_backward)
            backward_output, _ = pad_packed_sequence(
                backward_output,
                batch_first=True,
                total_length=sequence_features.size(1),
            )
            backward_output = _reverse_padded_sequence(backward_output, lengths)

            combined_output = self.dropout(
                torch.cat([forward_output, backward_output], dim=-1)
            )
            return (
                self.ws_classifier(combined_output),
                self.pos_classifier(combined_output),
            )

        def decode(self, input_ids, attention_mask):
            """Run CRF decoding for WS and POS heads."""
            ws_emissions, pos_emissions = self.forward(input_ids, attention_mask)
            mask = attention_mask.bool()
            return (
                self.ws_crf.decode(ws_emissions, mask=mask),
                self.pos_crf.decode(pos_emissions, mask=mask),
            )

else:  # pragma: no cover - this branch is only used when runtime deps are missing.
    JointSegPosModel = None


class InferenceUnavailableError(RuntimeError):
    """Raised when inference is requested before a working model is available."""


class MyanmarNLPModel:
    """Model service with real inference plus an explicit opt-in fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = "uninitialized"
        self.model_loaded = False
        self.fallback_enabled = settings.enable_fallback_model
        self.error_message: Optional[str] = None
        self._device = None
        self._model = None
        self._tokenizer = None
        self._ws_id2label: Dict[int, str] = {}
        self._pos_id2label: Dict[int, str] = {}
        self._max_length = settings.max_length
        self._tokenizer_workspace: Optional[tempfile.TemporaryDirectory[str]] = None

    def load(self) -> None:
        """Load model artifacts into memory."""
        if self.model_loaded:
            return

        if None in {
            torch,
            hf_hub_download,
            JointSegPosModel,
            AutoConfig,
            AutoTokenizer,
        }:
            self._mark_unavailable("Model runtime dependencies are not installed.")
            return

        try:
            model_config_path = self._download_artifact("config.json")
            state_path = self._download_artifact("best_model.pt")

            with open(model_config_path, "r", encoding="utf-8") as config_file:
                model_config = json.load(config_file)

            base_model_name = model_config.get("base_model", "xlm-roberta-base")
            base_config = AutoConfig.from_pretrained(
                base_model_name,
                token=self.settings.hf_token,
            )

            network = JointSegPosModel(
                base_config,
                num_ws_labels=int(model_config["num_ws_labels"]),
                num_pos_labels=int(model_config["num_pos_labels"]),
            )
            state_dict = torch.load(state_path, map_location="cpu")
            network.load_state_dict(state_dict)

            self._device = self._resolve_device()
            network.to(self._device)
            network.eval()

            self._tokenizer = self._load_tokenizer(base_model_name)
            self._model = network
            self._ws_id2label = {
                int(value): key for key, value in model_config["ws_label2id"].items()
            }
            self._pos_id2label = {
                int(value): key for key, value in model_config["pos_label2id"].items()
            }
            self._max_length = min(
                self._resolve_runtime_max_length(base_model_name),
                int(network.pos_embedding.num_embeddings),
            )
            self.backend = "huggingface"
            self.model_loaded = True
            self.error_message = None
            logger.info(
                "Loaded model %s on %s with max_length=%s",
                self.settings.model_name,
                self._device,
                self._max_length,
            )
        except Exception as exc:  # pragma: no cover - depends on external artifacts.
            logger.exception("Failed to load model artifacts")
            self._mark_unavailable(str(exc))

    def predict(self, text: str) -> List[Dict[str, str]]:
        """Run inference over a single text input."""
        return self.batch_predict([text])[0]

    def batch_predict(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        """Run inference over a batch of texts."""
        if not texts:
            return []

        predictions: List[List[Dict[str, str]]] = [[] for _ in texts]
        model_indices: List[int] = []
        model_texts: List[str] = []
        for index, text in enumerate(texts):
            if self._should_use_heuristic_path(text):
                predictions[index] = self._fallback_predict(text)
                continue
            model_indices.append(index)
            model_texts.append(text)

        if not model_texts:
            return predictions

        if self.backend == "uninitialized":
            self.load()

        if not self.model_loaded:
            if self.fallback_enabled:
                for index in model_indices:
                    predictions[index] = self._fallback_predict(texts[index])
                return predictions
            detail = self.error_message or "Model is not available."
            raise InferenceUnavailableError(detail)

        units_batch = [split_text_units(text) for text in model_texts]
        encoded = self._tokenizer(
            units_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            ws_paths, pos_paths = self._model.decode(input_ids, attention_mask)

        attention_mask_cpu = attention_mask.cpu()
        for batch_index, units in enumerate(units_batch):
            valid_word_ids = self._valid_word_ids(
                encoded.word_ids(batch_index=batch_index),
                int(attention_mask_cpu[batch_index].sum().item()),
            )
            ws_labels, pos_labels = self._align_predictions(
                valid_word_ids,
                ws_paths[batch_index],
                pos_paths[batch_index],
            )

            aligned_units = units[: len(ws_labels)]
            if len(aligned_units) != len(units):
                logger.warning(
                    "Input text truncated from %s units to %s units due to max_length=%s",
                    len(units),
                    len(aligned_units),
                    self._max_length,
                )

            tagged_words = collapse_syllable_predictions(
                aligned_units,
                ws_labels,
                pos_labels,
            )
            predictions[model_indices[batch_index]] = (
                [{"word": word, "pos": pos} for word, pos in tagged_words]
            )

        return predictions

    def segment_text(self, text: str) -> List[str]:
        """Return segmented words for a single text input."""
        return [token["word"] for token in self.predict(text)]

    def tag_text(self, text: str) -> List[Tuple[str, str]]:
        """Return `(word, pos)` tuples for a single text input."""
        return [(token["word"], token["pos"]) for token in self.predict(text)]

    def batch_tag_text(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """Return tagged tuples for a batch of texts."""
        return [
            [(token["word"], token["pos"]) for token in tagged_words]
            for tagged_words in self.batch_predict(texts)
        ]

    def _download_artifact(self, filename: str) -> str:
        """Download a model artifact from the Hugging Face Hub."""
        return hf_hub_download(
            repo_id=self.settings.model_name,
            filename=filename,
            revision=self.settings.model_revision,
            token=self.settings.hf_token,
        )

    def _load_tokenizer(self, base_model_name: str):
        """Load the tokenizer, retrying with the notebook compatibility patch if needed."""
        try:
            return AutoTokenizer.from_pretrained(
                base_model_name,
                token=self.settings.hf_token,
            )
        except Exception as exc:
            logger.warning(
                "Tokenizer load for %s failed (%s). Retrying with patched tokenizer_config.json.",
                base_model_name,
                exc,
            )
            return self._load_patched_tokenizer(base_model_name)

    def _load_patched_tokenizer(self, base_model_name: str):
        """Patch tokenizer assets locally before loading them with transformers."""
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub snapshot_download is unavailable.")

        snapshot_dir = snapshot_download(
            repo_id=base_model_name,
            allow_patterns=[
                "tokenizer*",
                "vocab*",
                "merges*",
                "special_tokens*",
            ],
            token=self.settings.hf_token,
        )

        workspace = tempfile.TemporaryDirectory(prefix="wordtoken-tokenizer-")
        shutil.copytree(snapshot_dir, workspace.name, dirs_exist_ok=True)
        config_path = os.path.join(workspace.name, "tokenizer_config.json")
        if self._patch_tokenizer_config(config_path):
            logger.info(
                "Patched tokenizer_config.json for %s compatibility before loading.",
                base_model_name,
            )

        tokenizer = AutoTokenizer.from_pretrained(workspace.name)
        self._tokenizer_workspace = workspace
        return tokenizer

    @staticmethod
    def _patch_tokenizer_config(tokenizer_config_path: str) -> bool:
        """Remove incompatible tokenizer metadata emitted by the training notebook."""
        if not os.path.exists(tokenizer_config_path):
            return False

        with open(tokenizer_config_path, "r", encoding="utf-8") as config_file:
            tokenizer_config = json.load(config_file)

        original = dict(tokenizer_config)
        tokenizer_config.pop("post_processor", None)
        tokenizer_config.pop("model", None)

        if tokenizer_config == original:
            return False

        with open(tokenizer_config_path, "w", encoding="utf-8") as config_file:
            json.dump(tokenizer_config, config_file, ensure_ascii=False, indent=2)

        return True

    def _resolve_runtime_max_length(self, base_model_name: str) -> int:
        """Clamp max_length to the range the selected checkpoint was trained with."""
        if base_model_name == MYANBERTA_BASE_MODEL:
            return min(int(self.settings.max_length), MYANBERTA_RECOMMENDED_MAX_LENGTH)
        return int(self.settings.max_length)

    def _resolve_device(self):
        """Resolve the target inference device."""
        requested = self.settings.device.lower()
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(self.settings.device)
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cpu")

    def _mark_unavailable(self, detail: str) -> None:
        """Record a model-loading failure."""
        self.model_loaded = False
        self.error_message = detail
        self.backend = "bootstrap" if self.fallback_enabled else "unavailable"
        logger.warning("Model unavailable: %s", detail)

    def _fallback_predict(self, text: str) -> List[Dict[str, str]]:
        """Use the deterministic bootstrap fallback."""
        return [
            {"word": word, "pos": infer_pos_tag(word)}
            for word in simple_segment(text)
        ]

    def _should_use_heuristic_path(self, text: str) -> bool:
        """Route non-Myanmar text through the deterministic fallback."""
        return not any(
            MYANMAR_CODEPOINT_RANGE[0] <= character <= MYANMAR_CODEPOINT_RANGE[1]
            for character in text
        )

    def _valid_word_ids(
        self,
        word_ids: Sequence[Optional[int]],
        valid_length: int,
    ) -> List[Optional[int]]:
        """Trim `word_ids` to the valid decoded span."""
        return list(word_ids[:valid_length])

    def _align_predictions(
        self,
        word_ids: Sequence[Optional[int]],
        ws_path: Sequence[int],
        pos_path: Sequence[int],
    ) -> Tuple[List[str], List[str]]:
        """Collapse subword-level CRF predictions back to the original units."""
        ws_labels: List[str] = []
        pos_labels: List[str] = []
        seen_word_ids = set()

        for word_id, ws_label_id, pos_label_id in zip(word_ids, ws_path, pos_path):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            ws_labels.append(self._ws_id2label[int(ws_label_id)])
            pos_labels.append(self._pos_id2label[int(pos_label_id)])

        return ws_labels, pos_labels
