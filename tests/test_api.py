"""API tests for the FastAPI application."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.model import InferenceUnavailableError


class UnavailableModel:
    """Model double that simulates an unavailable runtime."""

    backend = "unavailable"
    model_loaded = False
    fallback_enabled = False
    error_message = "Model artifacts are unavailable."

    def load(self) -> None:
        """Keep a compatible startup hook for the app."""

    def segment_text(self, text: str):
        raise InferenceUnavailableError(self.error_message)

    def tag_text(self, text: str):
        raise InferenceUnavailableError(self.error_message)

    def batch_tag_text(self, texts):
        raise InferenceUnavailableError(self.error_message)


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "sithu015/XLM-RoBERTa-BiLSTM-CRF-Joint"
    assert payload["device"] == "cpu"
    assert payload["backend"] == "stub"
    assert payload["model_loaded"] is True
    assert payload["fallback_enabled"] is False
    assert payload["detail"] is None


@pytest.mark.asyncio
async def test_segment_endpoint_returns_tokens(client):
    response = await client.post(
        "/api/v1/segment",
        json={"text": "မြန်မာ ဘာသာ။"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["words"] == ["မြန်မာ", "ဘာသာ", "။"]


@pytest.mark.asyncio
async def test_pos_endpoint_returns_tagged_tokens(client):
    response = await client.post(
        "/api/v1/pos",
        json={"text": "API test 123"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tokens"] == [
        {"word": "API", "pos": "X"},
        {"word": "test", "pos": "X"},
        {"word": "123", "pos": "NUM"},
    ]


@pytest.mark.asyncio
async def test_batch_endpoint_returns_results(client):
    response = await client.post(
        "/api/v1/batch",
        json={"texts": ["မင်္ဂလာပါ", "hello world"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["results"][0]["input"] == "မင်္ဂလာပါ"
    assert payload["results"][1]["tokens"] == [
        {"word": "hello", "pos": "X"},
        {"word": "world", "pos": "X"},
    ]


@pytest.mark.asyncio
async def test_empty_text_validation_error(client):
    response = await client.post("/api/v1/segment", json={"text": ""})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_size_validation_error(client):
    response = await client.post(
        "/api/v1/batch",
        json={"texts": ["sample"] * 33},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_unavailable_model_returns_503():
    app = create_app(model=UnavailableModel())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/segment",
            json={"text": "ကျွန်တော်သည်ကျောင်းသွားသည်"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model artifacts are unavailable."


@pytest.mark.asyncio
async def test_segment_endpoint_requires_api_key_when_enabled(auth_client):
    response = await auth_client.post(
        "/api/v1/segment",
        json={"text": "မြန်မာ ဘာသာ။"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "A valid X-API-Key header is required."


@pytest.mark.asyncio
async def test_segment_endpoint_accepts_valid_api_key(auth_client):
    response = await auth_client.post(
        "/api/v1/segment",
        headers={"X-API-Key": "test-api-key"},
        json={"text": "မြန်မာ ဘာသာ။"},
    )

    assert response.status_code == 200
    assert response.json()["words"] == ["မြန်မာ", "ဘာသာ", "။"]


@pytest.mark.asyncio
async def test_openapi_schema_exposes_api_key_security(auth_client):
    response = await auth_client.get("/openapi.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["securitySchemes"]["ApiKeyAuth"]["type"] == "apiKey"
    assert payload["components"]["securitySchemes"]["ApiKeyAuth"]["name"] == "X-API-Key"
    assert payload["paths"]["/api/v1/segment"]["post"]["security"] == [{"ApiKeyAuth": []}]
