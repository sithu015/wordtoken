"""FastAPI application entry point."""

from __future__ import annotations

import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.model import MyanmarNLPModel
from app.routes.health import router as health_router
from app.routes.nlp import router as nlp_router


logger = logging.getLogger("wordtoken")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
    )

    model = MyanmarNLPModel(settings)
    model.load()
    app.state.settings = settings
    app.state.model = model

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        started_at = perf_counter()
        response = await call_next(request)
        duration_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "%s %s -> %s (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Process-Time-MS"] = f"{duration_ms:.2f}"
        return response

    app.include_router(health_router)
    app.include_router(nlp_router)

    return app


app = create_app()
