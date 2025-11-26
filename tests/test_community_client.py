"""Tests for the Plex community GraphQL client."""

from __future__ import annotations

import asyncio
from typing import cast

import aiohttp
import pytest

from anibridge_plex_provider.community import PlexCommunityClient


class StubResponse:
    """Minimal aiohttp-like response wrapper for tests."""

    def __init__(self, *, status: int, payload: dict, retry_after: str = "0") -> None:
        """Initialize response metadata and optional retry header."""
        self.status = status
        self._payload = payload
        self.headers = {"Retry-After": retry_after}

    async def __aenter__(self):
        """Support async context usage."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Allow context exit without suppressing errors."""
        return False

    async def json(self):
        """Return the JSON payload."""
        return self._payload

    async def text(self):
        """Return payload as a string when error text is needed."""
        return "error"

    def raise_for_status(self) -> None:
        """Raise for HTTP statuses that indicate failure."""
        if self.status >= 400:
            raise RuntimeError("boom")


class StubSession:
    """Coroutine-friendly session stub feeding queued responses."""

    def __init__(self, responses: list[StubResponse]) -> None:
        """Store responses that will be returned sequentially."""
        self.responses = responses
        self.closed = False
        self.calls: list[dict] = []

    def post(self, url: str, json: dict):
        """Record the request and return the next response."""
        self.calls.append({"url": url, "json": json})
        return self.responses.pop(0)

    async def close(self) -> None:
        """Mirror aiohttp close behavior for cleanup assertions."""
        self.closed = True


@pytest.mark.asyncio
async def test_get_session_initializes_headers(monkeypatch: pytest.MonkeyPatch):
    """Test that the session initializes with the correct headers."""
    created = {}

    class DummySession:
        def __init__(self, *, headers: dict):
            created["headers"] = headers
            self.closed = False

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        "anibridge_plex_provider.community.importlib.metadata.version",
        lambda _: "1.2.3",
    )
    monkeypatch.setattr(
        "anibridge_plex_provider.community.aiohttp.ClientSession",
        DummySession,
    )

    client = PlexCommunityClient("token")
    session = await client._get_session()
    assert created["headers"]["X-Plex-Token"] == "token"
    await client.close()
    assert session.closed is True


@pytest.mark.asyncio
async def test_make_request_handles_rate_limit(monkeypatch: pytest.MonkeyPatch):
    """Test that rate-limited requests are retried after the specified delay."""
    responses = [
        StubResponse(status=429, payload={}),
        StubResponse(status=200, payload={"data": {"ok": True}}),
    ]
    stub_session = StubSession(responses)
    client = PlexCommunityClient("token")
    client._session = cast(aiohttp.ClientSession, stub_session)

    async def fast_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    result = await client._make_request("query", {}, "Op")
    assert result == {"data": {"ok": True}}
    assert len(stub_session.calls) == 2


@pytest.mark.asyncio
async def test_get_watch_activity_accumulates_pages(monkeypatch: pytest.MonkeyPatch):
    """Test that paginated watch activity is fully accumulated."""
    client = PlexCommunityClient("token")
    pages = [
        {
            "data": {
                "activityFeed": {
                    "nodes": ["first"],
                    "pageInfo": {"hasNextPage": True, "endCursor": "abc"},
                }
            }
        },
        {
            "data": {
                "activityFeed": {
                    "nodes": ["second"],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        },
    ]
    calls = {"count": 0}

    async def fake_request(*_args, **_kwargs):
        payload = pages[calls["count"]]
        calls["count"] += 1
        return payload

    monkeypatch.setattr(client, "_make_request", fake_request)
    result = await client.get_watch_activity("1")
    assert result == ["first", "second"]


@pytest.mark.asyncio
async def test_get_reviews_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch):
    """Test that get_reviews returns None when no review is found."""
    client = PlexCommunityClient("token")

    async def fake_request(*_args, **_kwargs):
        return {"data": {"metadataReviewV2": {}}}

    monkeypatch.setattr(client, "_make_request", fake_request)
    assert await client.get_reviews("1") is None

    async def fake_message(*_args, **_kwargs):
        return {"data": {"metadataReviewV2": {"message": "hi"}}}

    monkeypatch.setattr(client, "_make_request", fake_message)
    assert await client.get_reviews("1") == "hi"
