"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(request: Request) -> HealthResponse:
    """Report service health and model bootstrap state."""
    settings = request.app.state.settings
    model = request.app.state.model
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        device=settings.device,
        backend=model.backend,
        model_loaded=model.model_loaded,
    )
