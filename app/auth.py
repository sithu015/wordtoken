"""API key authentication helpers."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description="Send a valid API key in the X-API-Key header.",
    auto_error=False,
)


async def require_api_key(
    request: Request,
    provided_key: Annotated[Optional[str], Security(api_key_header)] = None,
) -> None:
    """Validate the configured API key header when auth is enabled."""
    expected_keys = request.app.state.settings.api_keys
    if not expected_keys:
        return

    if provided_key is None or provided_key not in expected_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required.",
        )
