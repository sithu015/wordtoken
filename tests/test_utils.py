"""Utility tests."""

from __future__ import annotations

from app.utils import decode_bio_tags


def test_decode_bio_tags_preserves_initial_pos_label():
    decoded = decode_bio_tags(
        tokens=["▁ကျွန်", "တော်"],
        labels=["B-PRON", "E-VERB"],
    )

    assert decoded == [("ကျွန်တော်", "PRON")]
