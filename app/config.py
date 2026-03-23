"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


DEFAULT_MODEL_NAME = "sithu015/XLM-RoBERTa-BiLSTM-CRF-Joint"


def _as_bool(value: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_csv(value: str, *, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Environment-backed application settings."""

    app_name: str
    app_version: str
    app_description: str
    model_name: str
    device: str
    max_length: int
    max_batch_size: int
    host: str
    port: int
    debug: bool
    cors_allowed_origins: Tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        return cls(
            app_name="Myanmar Word Segmentation & POS Tagging API",
            app_version="0.1.0",
            app_description=(
                "Starter FastAPI service for Myanmar word segmentation and POS "
                "tagging."
            ),
            model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
            device=os.getenv("DEVICE", "cpu"),
            max_length=int(os.getenv("MAX_LENGTH", "512")),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "32")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=_as_bool(os.getenv("DEBUG"), default=False),
            cors_allowed_origins=_as_csv(
                os.getenv("CORS_ALLOWED_ORIGINS"),
                default=("*",),
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings.from_env()
