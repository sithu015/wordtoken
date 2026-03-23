"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Annotated, List, Optional

from pydantic import BaseModel, ConfigDict, Field


TextField = Annotated[str, Field(min_length=1, max_length=2000)]


class SegmentRequest(BaseModel):
    """Input payload for word segmentation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: TextField


class POSToken(BaseModel):
    """Single tagged token."""

    word: str
    pos: str


class SegmentResponse(BaseModel):
    """Response payload for word segmentation."""

    input: str
    words: List[str]
    processing_time_ms: float


class POSResponse(BaseModel):
    """Response payload for joint POS tagging."""

    input: str
    tokens: List[POSToken]
    processing_time_ms: float


class BatchRequest(BaseModel):
    """Input payload for batch inference."""

    texts: List[TextField] = Field(min_length=1)


class BatchItemResponse(BaseModel):
    """Tagged output for a single text item in a batch request."""

    input: str
    tokens: List[POSToken]


class BatchResponse(BaseModel):
    """Response payload for batch inference."""

    results: List[BatchItemResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    device: str
    backend: str
    model_loaded: bool
    fallback_enabled: bool
    detail: Optional[str] = None
