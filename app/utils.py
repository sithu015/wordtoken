"""Utility helpers for starter inference behavior."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple


PUNCTUATION_PATTERN = re.compile(r"[၊။!?.,:;()\[\]{}\"']")
ASCII_WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
SPECIAL_TOKENS = {"<s>", "</s>", "<pad>"}


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
