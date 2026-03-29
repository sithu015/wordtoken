"""Model helper tests."""

from __future__ import annotations

import json

from app.config import get_settings
from app.model import (
    MYANBERTA_BASE_MODEL,
    MYANBERTA_RECOMMENDED_MAX_LENGTH,
    MyanmarNLPModel,
)


def test_patch_tokenizer_config_removes_incompatible_fields(tmp_path):
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(
        json.dumps(
            {
                "post_processor": {"type": "ByteLevel"},
                "model": {"type": "BPE"},
                "keep": "value",
            }
        ),
        encoding="utf-8",
    )

    changed = MyanmarNLPModel._patch_tokenizer_config(str(config_path))

    assert changed is True
    patched = json.loads(config_path.read_text(encoding="utf-8"))
    assert patched == {"keep": "value"}


def test_patch_tokenizer_config_noops_when_fields_absent(tmp_path):
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(json.dumps({"keep": "value"}), encoding="utf-8")

    changed = MyanmarNLPModel._patch_tokenizer_config(str(config_path))

    assert changed is False
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"keep": "value"}


def test_runtime_max_length_is_clamped_for_myanberta():
    model = MyanmarNLPModel(get_settings())

    assert (
        model._resolve_runtime_max_length(MYANBERTA_BASE_MODEL)
        == MYANBERTA_RECOMMENDED_MAX_LENGTH
    )
    assert model._resolve_runtime_max_length("xlm-roberta-base") == get_settings().max_length
