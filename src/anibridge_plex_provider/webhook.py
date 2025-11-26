"""Plex webhook implementation."""

from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from starlette.requests import Request


class PlexWebhookEventType(StrEnum):
    """Enumeration of Plex webhook event types."""

    MEDIA_ADDED = "library.new"
    ON_DECK = "library.on.deck"
    PLAY = "media.play"
    PAUSE = "media.pause"
    STOP = "media.stop"
    RESUME = "media.resume"
    SCROBBLE = "media.scrobble"
    RATE = "media.rate"
    DATABASE_BACKUP = "admin.database.backup"
    DATABASE_CORRUPTED = "admin.database.corrupted"
    NEW_ADMIN_DEVICE = "device.new"
    SHARED_PLAYBACK_STARTED = "playback.started"


class Account(BaseModel):
    """Represents a Plex account involved in a webhook event."""

    id: int | None = None
    thumb: str | None = None
    title: str | None = None


class Server(BaseModel):
    """Represents a Plex server involved in a webhook event."""

    title: str | None = None
    uuid: str | None = None


class Player(BaseModel):
    """Represents a Plex player involved in a webhook event."""

    local: bool
    publicAddress: str | None = None
    title: str | None = None
    uuid: str | None = None


class Metadata(BaseModel):
    """Represents metadata information received from a Plex webhook event."""

    librarySectionType: str | None = None
    ratingKey: str | None = None
    key: str | None = None
    parentRatingKey: str | None = None
    grandparentRatingKey: str | None = None
    guid: str | None = None
    librarySectionID: int | None = None
    type: str | None = None
    title: str | None = None
    year: int | None = None
    grandparentKey: str | None = None
    parentKey: str | None = None
    grandparentTitle: str | None = None
    parentTitle: str | None = None
    summary: str | None = None
    index: int | None = None
    parentIndex: int | None = None
    ratingCount: int | None = None
    thumb: str | None = None
    art: str | None = None
    parentThumb: str | None = None
    grandparentThumb: str | None = None
    grandparentArt: str | None = None
    addedAt: int | None = None
    updatedAt: int | None = None


class PlexWebhook(BaseModel):
    """Represents a Plex webhook event."""

    event: str | None = None
    user: bool
    owner: bool
    account: Account | None = Field(None, alias="Account")
    server: Server | None = Field(None, alias="Server")
    player: Player | None = Field(None, alias="Player")
    metadata: Metadata | None = Field(None, alias="Metadata")

    @cached_property
    def event_type(self) -> PlexWebhookEventType | None:
        """The webhook event type."""
        if self.event is None:
            return None
        try:
            return PlexWebhookEventType(self.event)
        except ValueError:
            return None

    @cached_property
    def account_id(self) -> int | None:
        """The webhook owner's Plex account ID."""
        return self.account.id if self.account and self.account.id is not None else None

    @cached_property
    def top_level_rating_key(self) -> str | None:
        """The top-level rating key for the media item."""
        if not self.metadata:
            return None
        return (
            self.metadata.grandparentRatingKey
            or self.metadata.parentRatingKey
            or self.metadata.ratingKey
        )

    @classmethod
    async def from_request(cls, request: Request) -> PlexWebhook:
        """Create a PlexWebhook instance from an incoming HTTP request."""
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            payload_raw = form.get("payload")
            if not payload_raw:
                raise ValueError("Missing 'payload' form field")
            try:
                return PlexWebhook.model_validate_json(str(payload_raw))
            except Exception as e:
                raise ValueError(f"Invalid payload JSON: {e}") from e
        # Fallback to JSON body
        try:
            data = await request.json()
        except Exception as e:
            raise ValueError(f"Invalid JSON body: {e}") from e
        try:
            return PlexWebhook.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid payload structure: {e}") from e
