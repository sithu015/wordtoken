"""API tests for the starter FastAPI application."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "sithu015/XLM-RoBERTa-BiLSTM-CRF-Joint"
    assert payload["device"] == "cpu"
    assert payload["backend"] == "bootstrap"
    assert payload["model_loaded"] is False


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
