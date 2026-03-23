"""Starter model service for Myanmar NLP inference."""

from __future__ import annotations

from typing import List, Tuple

from app.config import Settings
from app.utils import infer_pos_tag, simple_segment


class MyanmarNLPModel:
    """Bootstrap inference service used until the HuggingFace model is wired in."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = "bootstrap"
        self.model_loaded = False

    def load(self) -> None:
        """Prepare the service state for requests."""
        self.backend = "bootstrap"
        self.model_loaded = False

    def segment_text(self, text: str) -> List[str]:
        """Return a deterministic starter segmentation."""
        return simple_segment(text)

    def tag_text(self, text: str) -> List[Tuple[str, str]]:
        """Return a starter POS tagging output."""
        return [(word, infer_pos_tag(word)) for word in self.segment_text(text)]

    def batch_tag_text(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """Run starter tagging over a batch of text strings."""
        return [self.tag_text(text) for text in texts]
