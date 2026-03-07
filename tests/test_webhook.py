"""Tests for the Plex webhook parsing helpers."""

from dataclasses import dataclass
from typing import Any, cast

import pytest

from anibridge.providers.library.plex.webhook import (
    Account,
    Metadata,
    PlexWebhook,
    PlexWebhookEventType,
    TautulliWebhook,
    WebhookParser,
)


@dataclass
class _StubRequest:
    headers: dict[str, str]
    form_payload: str | None = None
    json_payload: dict | None = None
    json_error: Exception | None = None
    query_params: dict[str, str] | None = None

    def __post_init__(self):
        if self.query_params is None:
            self.query_params = {}

    async def form(self):
        if self.form_payload is None:
            return {}
        return {"payload": self.form_payload}

    async def json(self):
        if self.json_error is not None:
            raise self.json_error
        if self.json_payload is None:
            raise ValueError("No JSON provided")
        return self.json_payload


def test_webhook_event_helpers_resolve_expected_fields():
    """event_type, account_id, and rating key helpers work."""
    payload = PlexWebhook(
        event=PlexWebhookEventType.PLAY.value,
        user=True,
        owner=True,
        Account=Account(id=4),
        Server=None,
        Player=None,
        Metadata=Metadata(
            ratingKey="episode",
            parentRatingKey="season",
            grandparentRatingKey="show",
        ),
    )

    assert payload.event_type is PlexWebhookEventType.PLAY
    assert payload.account_id == 4
    assert payload.top_level_rating_key == "show"


@pytest.mark.asyncio
async def test_from_request_handles_multipart_form_payload():
    """Multipart form requests are decoded via payload field."""
    stub_request = _StubRequest(
        headers={"content-type": "multipart/form-data"},
        query_params={"format": "plex"},
        form_payload='{"event": "media.stop", "user": true, "owner": false}',
    )

    payload = await WebhookParser.from_request(cast(Any, stub_request))
    assert isinstance(payload, PlexWebhook)
    assert payload.event_type is PlexWebhookEventType.STOP
    assert payload.owner is False


@pytest.mark.asyncio
async def test_from_request_falls_back_to_json_body():
    """Non-multipart bodies are parsed as JSON."""
    stub_request = _StubRequest(
        headers={"content-type": "application/json"},
        query_params={"format": "plex"},
        json_payload={"event": "library.on.deck", "user": True, "owner": True},
    )

    payload = await WebhookParser.from_request(cast(Any, stub_request))
    assert isinstance(payload, PlexWebhook)
    assert payload.event_type is PlexWebhookEventType.ON_DECK


@pytest.mark.asyncio
async def test_from_request_requires_explicit_tautulli_format_hint():
    """Requests without a recognized format hint raise ValueError."""
    stub_request = _StubRequest(
        headers={"content-type": "application/json"},
        json_payload={
            "action": "created",
            "user_id": "7",
            "rating_key": "episode",
            "parent_rating_key": "season",
            "grandparent_rating_key": "show",
        },
    )

    with pytest.raises(ValueError, match="Unsupported format"):
        await WebhookParser.from_request(cast(Any, stub_request))


@pytest.mark.asyncio
async def test_from_request_supports_explicit_tautulli_format_hint():
    """format=tautulli forces Tautulli payload parsing semantics."""
    stub_request = _StubRequest(
        headers={"content-type": "application/json"},
        query_params={"format": "tautulli"},
        json_payload={
            "action": "scrobble",
            "user_id": 9,
            "rating_key": "movie",
        },
    )

    payload = await WebhookParser.from_request(cast(Any, stub_request))
    assert isinstance(payload, TautulliWebhook)
    assert payload.event_type is PlexWebhookEventType.SCROBBLE
    assert payload.account_id == 9
    assert payload.top_level_rating_key == "movie"


@pytest.mark.asyncio
async def test_from_request_raises_for_invalid_payload():
    """Invalid payload inputs raise ValueError for visibility."""
    stub_request = _StubRequest(
        headers={"content-type": "multipart/form-data"},
        query_params={"format": "plex"},
        form_payload="not-json",
    )

    with pytest.raises(ValueError):
        await WebhookParser.from_request(cast(Any, stub_request))
