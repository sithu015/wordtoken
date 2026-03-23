"""Tests for the public documentation pages."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_root_page_exposes_documentation(client):
    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "Production docs for segmentation and POS tagging." in body
    assert "Interactive API docs" in body
    assert "/wiki" in body
    assert "wordtoken.ygn.app" in body


@pytest.mark.asyncio
async def test_wiki_page_exposes_operations_content(client):
    response = await client.get("/wiki")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "GitHub Actions deployment" in body
    assert "DEPLOY_SSH_KEY" in body
    assert "docker logs --tail=200 wordtoken" in body
