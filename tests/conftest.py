"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


class StubModel:
    """Simple in-memory model used to isolate API tests from heavy runtime setup."""

    backend = "stub"
    model_loaded = True
    fallback_enabled = False
    error_message = None

    def load(self) -> None:
        """Keep a compatible startup hook for the app."""

    def segment_text(self, text: str):
        if text == "မြန်မာ ဘာသာ။":
            return ["မြန်မာ", "ဘာသာ", "။"]
        return text.split()

    def tag_text(self, text: str):
        if text == "API test 123":
            return [("API", "X"), ("test", "X"), ("123", "NUM")]
        return [(word, "X") for word in self.segment_text(text)]

    def batch_tag_text(self, texts):
        return [self.tag_text(text) for text in texts]


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Create an async test client for the FastAPI app."""
    app = create_app(model=StubModel())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
