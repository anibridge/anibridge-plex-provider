"""Plex webhook implementation."""

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import msgspec

if TYPE_CHECKING:
    from litestar.connection.request import Request

type WebhookPayload = PlexWebhook | TautulliWebhook


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


class Account(msgspec.Struct):
    """Represents a Plex account involved in a webhook event."""

    id: int | None = None
    # thumb: str | None = None
    # title: str | None = None


class Server(msgspec.Struct):
    """Represents a Plex server involved in a webhook event."""

    # title: str | None = None
    # uuid: str | None = None


class Player(msgspec.Struct):
    """Represents a Plex player involved in a webhook event."""

    # local: bool | None = None
    # publicAddress: str | None = None
    # title: str | None = None
    # uuid: str | None = None


class Metadata(
    msgspec.Struct,
    rename={
        "rating_key": "ratingKey",
        "parent_rating_key": "parentRatingKey",
        "grandparent_rating_key": "grandparentRatingKey",
        "library_section_id": "librarySectionID",
        "grandparent_key": "grandparentKey",
        "parent_key": "parentKey",
    },
):
    """Represents metadata information received from a Plex webhook event."""

    # librarySectionType: str | None = None
    rating_key: str | None = None
    key: str | None = None
    parent_rating_key: str | None = None
    grandparent_rating_key: str | None = None
    # guid: str | None = None
    library_section_id: int | None = None
    # type: str | None = None
    title: str | None = None
    # year: int | None = None
    grandparent_key: str | None = None
    parent_key: str | None = None
    # grandparentTitle: str | None = None
    # parentTitle: str | None = None
    # summary: str | None = None
    # index: int | None = None
    # parentIndex: int | None = None
    # ratingCount: int | None = None
    # thumb: str | None = None
    # art: str | None = None
    # parentThumb: str | None = None
    # grandparentThumb: str | None = None
    # grandparentArt: str | None = None
    # addedAt: int | None = None
    # updatedAt: int | None = None


class PlexWebhook(
    msgspec.Struct,
    rename={"account": "Account", "metadata": "Metadata"},
):
    """Represents a Plex webhook event."""

    event: str | None = None
    # user: bool | None = None
    # owner: bool | None = None
    account: Account | None = None
    # server: Server | None = Field(None, alias="Server")
    # player: Player | None = Field(None, alias="Player")
    metadata: Metadata | None = None

    @property
    def event_type(self) -> PlexWebhookEventType | None:
        """The webhook event type."""
        if not self.event:
            return None
        try:
            return PlexWebhookEventType(self.event)
        except ValueError:
            return None

    @property
    def account_id(self) -> int | None:
        """The webhook owner's Plex account ID."""
        return self.account.id if self.account and self.account.id is not None else None

    @property
    def top_level_rating_key(self) -> str | None:
        """The top-level rating key for the media item."""
        if not self.metadata:
            return None
        return (
            self.metadata.grandparent_rating_key
            or self.metadata.parent_rating_key
            or self.metadata.rating_key
        )

    @property
    def section_key(self) -> str | None:
        """The library section key if present."""
        if not self.metadata or self.metadata.library_section_id is None:
            return None
        return str(self.metadata.library_section_id)


class TautulliWebhook(msgspec.Struct):
    """Represents a normalized Tautulli webhook payload."""

    _TAUTULLI_ACTION_MAP: ClassVar[dict[str, PlexWebhookEventType]] = {
        "play": PlexWebhookEventType.PLAY,
        "pause": PlexWebhookEventType.PAUSE,
        "stop": PlexWebhookEventType.STOP,
        "resume": PlexWebhookEventType.RESUME,
        "scrobble": PlexWebhookEventType.SCROBBLE,
        "rate": PlexWebhookEventType.RATE,
        "rated": PlexWebhookEventType.RATE,
        "created": PlexWebhookEventType.MEDIA_ADDED,
        "recently_added": PlexWebhookEventType.MEDIA_ADDED,
        "on_deck": PlexWebhookEventType.ON_DECK,
    }

    action: str | None = None
    user_id: int | str | None = None
    rating_key: str | None = None
    parent_rating_key: str | None = None
    grandparent_rating_key: str | None = None
    library_section_key: int | str | None = None

    @property
    def event_type(self) -> PlexWebhookEventType | None:
        """The webhook event type normalized to Plex event enum values."""
        if not self.action:
            return None
        normalized = str(self.action).strip().lower()
        return self._TAUTULLI_ACTION_MAP.get(normalized)

    @property
    def account_id(self) -> int | None:
        """The webhook owner's Plex account ID if present."""
        if self.user_id is None:
            return None
        try:
            return int(self.user_id)
        except TypeError, ValueError:
            return None

    @property
    def top_level_rating_key(self) -> str | None:
        """The top-level rating key for the media item."""
        return self.grandparent_rating_key or self.parent_rating_key or self.rating_key

    @property
    def section_key(self) -> str | None:
        """The library section key if present."""
        if self.library_section_key is None:
            return None
        return str(self.library_section_key)


class WebhookParser:
    """Parser for incoming Plex (multipart) or Tautulli webhooks."""

    @staticmethod
    def media_type(content_type: str | None) -> str:
        """Read the media type portion of a Content-Type header.

        Args:
            content_type (str): The full Content-Type header value, e.g.
                "multipart/form-data; boundary=abc".

        Returns:
            str: The media type portion of the Content-Type header.
        """
        if not content_type:
            return ""
        return content_type.split(";", 1)[0].strip().lower()

    @classmethod
    async def from_request(cls, request: Request) -> WebhookPayload:
        """Create a webhook instance from an incoming HTTP request.

        Args:
            request (Request): The incoming HTTP request containing the webhook payload.

        Returns:
            WebhookPayload: An instance of PlexWebhook or TautulliWebhook parsed from
                the request.

        Raises:
            ValueError: If the 'format' query parameter is missing or not one of
                'plex' or 'tautulli'.
        """
        payload_format = (
            request.query_params.get("format", "plex").strip().lower() or "plex"
        )
        content_type = WebhookParser.media_type(request.headers.get("content-type"))

        if payload_format == "plex":
            if content_type in (
                "multipart/form-data",
                "application/x-www-form-urlencoded",
            ):
                form = await request.form()
                payload_raw = form.get("payload")
                if not payload_raw:
                    raise ValueError(
                        "Missing 'payload' field in multipart/form-data request"
                    )

                if isinstance(payload_raw, bytes):
                    payload_raw = payload_raw.decode("utf-8", "replace")
                try:
                    return msgspec.json.decode(str(payload_raw), type=PlexWebhook)
                except Exception as e:
                    raise ValueError(
                        f"Invalid Plex payload JSON in 'payload' field: {e}"
                    ) from e
            elif content_type == "application/json":
                try:
                    data = await request.json()
                    return msgspec.convert(data, type=PlexWebhook)
                except Exception as e:
                    raise ValueError(f"Invalid Plex JSON payload: {e}") from e
            else:
                raise ValueError(
                    f"Unsupported content type '{content_type}' for Plex webhook "
                    "(expected multipart/form-data, application/x-www-form-urlencoded, "
                    "or application/json)"
                )

        elif payload_format == "tautulli":
            if content_type != "application/json":
                raise ValueError(
                    f"Unsupported content type '{content_type}' for Tautulli webhook "
                    "(expected application/json)"
                )
            try:
                data = await request.json()
            except Exception as e:
                raise ValueError(f"Invalid JSON body: {e}") from e
            if not isinstance(data, dict):
                raise ValueError("Invalid payload structure: expected JSON object")
            return msgspec.convert(data, type=TautulliWebhook)

        else:
            raise ValueError(
                f"Unsupported format '{payload_format}' specified in query parameters"
            )
