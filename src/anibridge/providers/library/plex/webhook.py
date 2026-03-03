"""Plex webhook implementation."""

from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

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
    async def from_request(cls, request: Request) -> PlexWebhook | TautulliWebhook:
        """Create a webhook instance from an incoming HTTP request."""
        content_type = request.headers.get("content-type", "")
        format = request.query_params.get("format", "plex").lower()

        if format == "plex":
            if content_type.startswith("multipart/form-data"):
                form = await request.form()
                payload_raw = form.get("payload")
                if not payload_raw:
                    raise ValueError("Missing 'payload' form field")
                try:
                    return PlexWebhook.model_validate_json(str(payload_raw))
                except Exception as e:
                    raise ValueError(f"Invalid payload JSON: {e}") from e

            elif content_type.startswith("application/json"):
                try:
                    data = await request.json()
                    if not isinstance(data, dict):
                        raise ValueError(
                            "Invalid payload structure: expected JSON object"
                        )
                    return PlexWebhook.model_validate(data)
                except Exception as e:
                    raise ValueError(f"Invalid Plex payload structure: {e}") from e

            else:
                raise ValueError(
                    f"Unsupported content type '{content_type}' for Plex webhook"
                )

        elif format == "tautulli":
            try:
                data = await request.json()
                if not isinstance(data, dict):
                    raise ValueError("Invalid payload structure: expected JSON object")
                return TautulliWebhook.model_validate(data)
            except Exception as e:
                raise ValueError(f"Invalid Tautulli payload structure: {e}") from e

        else:
            raise ValueError(
                f"Unsupported format '{format}' specified in query parameters"
            )


class TautulliWebhook(BaseModel):
    """Represents a normalized Tautulli webhook payload."""

    _TAUTULLI_EVENT_MAP: ClassVar[dict[str, PlexWebhookEventType]] = {
        "created": PlexWebhookEventType.MEDIA_ADDED,
        "on_created": PlexWebhookEventType.MEDIA_ADDED,
        "recently_added": PlexWebhookEventType.MEDIA_ADDED,
        "library.new": PlexWebhookEventType.MEDIA_ADDED,
        "rated": PlexWebhookEventType.RATE,
        "rate": PlexWebhookEventType.RATE,
        "on_rate": PlexWebhookEventType.RATE,
        "watched": PlexWebhookEventType.SCROBBLE,
        "scrobble": PlexWebhookEventType.SCROBBLE,
        "on_watched": PlexWebhookEventType.SCROBBLE,
        "on_scrobble": PlexWebhookEventType.SCROBBLE,
        "media.scrobble": PlexWebhookEventType.SCROBBLE,
    }

    event: str | None = None
    action: str | None = None
    notify_action: str | None = None
    user_id: int | str | None = None
    account_id_raw: int | str | None = None
    rating_key: str | None = None
    parent_rating_key: str | None = None
    grandparent_rating_key: str | None = None
    parentRatingKey: str | None = None
    grandparentRatingKey: str | None = None

    @cached_property
    def event_type(self) -> PlexWebhookEventType | None:
        """The webhook event type normalized to Plex event enum values."""
        candidates = (
            self.event,
            self.action,
            self.notify_action,
        )
        for candidate in candidates:
            if not candidate:
                continue
            normalized = str(candidate).strip().lower()
            try:
                return PlexWebhookEventType(normalized)
            except ValueError:
                mapped = TautulliWebhook._TAUTULLI_EVENT_MAP.get(normalized)
                if mapped is not None:
                    return mapped
        return None

    @cached_property
    def account_id(self) -> int | None:
        """The webhook owner's Plex account ID if present."""
        candidate = self.user_id if self.user_id is not None else self.account_id_raw
        if candidate is None:
            return None
        try:
            return int(candidate)
        except TypeError, ValueError:
            return None

    @cached_property
    def top_level_rating_key(self) -> str | None:
        """The top-level rating key for the media item."""
        return (
            self.grandparent_rating_key
            or self.grandparentRatingKey
            or self.parent_rating_key
            or self.parentRatingKey
            or self.rating_key
        )
