"""Utility helpers for starter inference behavior."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple


PUNCTUATION_PATTERN = re.compile(r"[၊။!?.,:;()\[\]{}\"']")
ASCII_WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
SPECIAL_TOKENS = {"<s>", "</s>", "<pad>"}
BOUNDARY_TAGS = {"B", "I", "E", "S"}
MIXED_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[A-Za-z0-9_-]*[A-Za-z0-9])?|\d+(?:[.,]\d+)*|\s+|[၊။!?.,:;()\[\]{}\"']|[^A-Za-z0-9\s၊။!?.,:;()\[\]{}\"']+"
)
SYLLABLE_BREAK_PATTERN = re.compile(
    r"((?<!္)[က-အ](?![်္])|[ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@[-`{-~\s])"
)


def normalize_text(text: str) -> str:
    """Collapse extra whitespace while preserving meaningful characters."""
    return re.sub(r"\s+", " ", text.strip())


def simple_segment(text: str) -> List[str]:
    """Provide a deterministic bootstrap tokenizer for early project stages."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens: List[str] = []
    current: List[str] = []

    for char in normalized:
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current.clear()
            continue

        if PUNCTUATION_PATTERN.fullmatch(char):
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(char)
            continue

        current.append(char)

    if current:
        tokens.append("".join(current))

    return tokens


def split_text_units(text: str) -> List[str]:
    """Split text into Myanmar syllables plus ASCII tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    units: List[str] = []
    for chunk in MIXED_TOKEN_PATTERN.findall(normalized):
        if not chunk or chunk.isspace():
            continue
        if ASCII_WORD_PATTERN.fullmatch(chunk) or chunk.isdigit():
            units.append(chunk)
            continue
        if PUNCTUATION_PATTERN.fullmatch(chunk):
            units.append(chunk)
            continue
        units.extend(_split_myanmar_chunk(chunk))
    return units


def _split_myanmar_chunk(chunk: str) -> List[str]:
    """Break a Myanmar-only text chunk into syllables."""
    segmented = SYLLABLE_BREAK_PATTERN.sub(r"|\1", chunk)
    if segmented.startswith("|"):
        segmented = segmented[1:]
    return [piece for piece in segmented.split("|") if piece and not piece.isspace()]


def infer_pos_tag(token: str) -> str:
    """Assign a lightweight placeholder POS tag."""
    if PUNCTUATION_PATTERN.fullmatch(token):
        return "PUNC"
    if token.isdigit():
        return "NUM"
    if ASCII_WORD_PATTERN.fullmatch(token):
        return "X"
    return "UNK"


def merge_subword_tokens(
    tokens: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Merge SentencePiece-style tokens into a whole-token sequence."""
    merged: List[Tuple[str, Optional[str]]] = []
    current_token = ""
    current_label: Optional[str] = None

    for index, token in enumerate(tokens):
        if token in SPECIAL_TOKENS:
            continue

        piece = token.removeprefix("▁").removeprefix("##")
        starts_new = token.startswith("▁") or not current_token

        if starts_new and current_token:
            merged.append((current_token, current_label))
            current_token = ""
            current_label = None

        current_token += piece

        if current_label is None and labels is not None:
            current_label = labels[index]

    if current_token:
        merged.append((current_token, current_label))

    return merged


def decode_bio_tags(tokens: Sequence[str], labels: Sequence[str]) -> List[Tuple[str, str]]:
    """Decode BIO/BIOES tags into `(word, pos)` pairs."""
    decoded: List[Tuple[str, str]] = []
    current_word = ""
    current_pos = "UNK"

    for token, label in zip(tokens, labels):
        if token in SPECIAL_TOKENS:
            continue

        piece = token.removeprefix("▁").removeprefix("##")
        if not label or label == "O":
            if current_word:
                decoded.append((current_word, current_pos))
                current_word = ""
                current_pos = "UNK"
            if piece and PUNCTUATION_PATTERN.fullmatch(piece):
                decoded.append((piece, "PUNC"))
            continue

        prefix, _, pos = label.partition("-")
        if prefix in {"B", "S"} or not current_word:
            if current_word:
                decoded.append((current_word, current_pos))
            current_word = piece
            current_pos = pos or "UNK"
            if prefix == "S":
                decoded.append((current_word, current_pos))
                current_word = ""
                current_pos = "UNK"
            continue

        if prefix in {"I", "E"}:
            current_word += piece
            current_pos = pos or current_pos
            if prefix == "E":
                decoded.append((current_word, current_pos))
                current_word = ""
                current_pos = "UNK"

    if current_word:
        decoded.append((current_word, current_pos))

    return decoded


def collapse_syllable_predictions(
    units: Sequence[str],
    ws_labels: Sequence[str],
    pos_labels: Sequence[str],
) -> List[Tuple[str, str]]:
    """Collapse syllable-level predictions into `(word, pos)` pairs."""
    decoded: List[Tuple[str, str]] = []
    current_units: List[str] = []
    current_pos = "X"

    def flush() -> None:
        nonlocal current_pos
        if current_units:
            decoded.append(("".join(current_units), current_pos or "X"))
            current_units.clear()
            current_pos = "X"

    for unit, ws_label, pos_label in zip(units, ws_labels, pos_labels):
        pos_prefix, _, pos_tag = pos_label.partition("-")
        boundary = ws_label if ws_label in BOUNDARY_TAGS else pos_prefix
        if boundary not in BOUNDARY_TAGS:
            boundary = "S"

        if boundary in {"B", "S"}:
            flush()
            current_units.append(unit)
            current_pos = pos_tag or "X"
            if boundary == "S":
                flush()
            continue

        if not current_units:
            current_units.append(unit)
            current_pos = pos_tag or "X"
        else:
            current_units.append(unit)
            if current_pos == "X" and pos_tag:
                current_pos = pos_tag

        if boundary == "E":
            flush()

    flush()
    return decoded
