"""NLP routes for starter inference endpoints."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, HTTPException, Request

from app.model import InferenceUnavailableError
from app.schemas import (
    BatchItemResponse,
    BatchRequest,
    BatchResponse,
    POSToken,
    POSResponse,
    SegmentRequest,
    SegmentResponse,
)


router = APIRouter(prefix="/api/v1", tags=["nlp"])


@router.post("/segment", response_model=SegmentResponse, summary="Segment text")
async def segment_text(payload: SegmentRequest, request: Request) -> SegmentResponse:
    """Segment Myanmar text into token strings."""
    started_at = perf_counter()
    try:
        words = request.app.state.model.segment_text(payload.text)
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SegmentResponse(
        input=payload.text,
        words=words,
        processing_time_ms=(perf_counter() - started_at) * 1000,
    )


@router.post("/pos", response_model=POSResponse, summary="Tag text")
async def tag_text(payload: SegmentRequest, request: Request) -> POSResponse:
    """Produce starter POS tags for the provided text."""
    started_at = perf_counter()
    try:
        tagged_tokens = request.app.state.model.tag_text(payload.text)
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return POSResponse(
        input=payload.text,
        tokens=[POSToken(word=word, pos=pos) for word, pos in tagged_tokens],
        processing_time_ms=(perf_counter() - started_at) * 1000,
    )


@router.post("/batch", response_model=BatchResponse, summary="Batch tag text")
async def batch_tag_text(payload: BatchRequest, request: Request) -> BatchResponse:
    """Process a batch of text strings with the starter tagger."""
    max_batch_size = request.app.state.settings.max_batch_size
    if len(payload.texts) > max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size exceeds MAX_BATCH_SIZE={max_batch_size}.",
        )

    started_at = perf_counter()
    try:
        results = request.app.state.model.batch_tag_text(payload.texts)
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    items = [
        BatchItemResponse(
            input=text,
            tokens=[POSToken(word=word, pos=pos) for word, pos in tagged_words],
        )
        for text, tagged_words in zip(payload.texts, results)
    ]
    return BatchResponse(
        results=items,
        processing_time_ms=(perf_counter() - started_at) * 1000,
    )
