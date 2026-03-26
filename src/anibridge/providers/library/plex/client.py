"""Plex client abstractions consumed by the Plex library provider."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import Logger
from time import monotonic
from typing import Literal, cast
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from anibridge.utils.datetime import normalize_local_datetime
from anibridge.utils.types import ProviderLogger
from plexapi.library import LibrarySection, MovieSection, ShowSection
from plexapi.myplex import MyPlexAccount
from plexapi.server import PlexServer
from plexapi.video import Movie, Show, Video

from anibridge.providers.library.plex.utils import SelectiveVerifySession

__all__ = ["Ordering", "PlexClient"]

type Ordering = Literal["tmdb", "tvdb", ""]


@dataclass(slots=True)
class _FrozenCacheEntry:
    """Immutable cache entry for storing Plex item keys with expiration."""

    keys: frozenset
    expires_at: float


class PlexClient:
    """High-level Plex client wrapper used by the library provider."""

    def __init__(
        self,
        *,
        logger: ProviderLogger,
        url: str,
        token: str,
        home_user: str | None = None,
        section_filter: Sequence[str] | None = None,
        genre_filter: Sequence[str] | None = None,
    ) -> None:
        """Initialize client wrapper with optional section and genre filters.

        Args:
            logger (ProviderLogger): Injected logger.
            url (str): The base URL of the Plex server.
            token (str): The Plex authentication token.
            home_user (str | None): Optional Plex Home user to switch to.
            section_filter (Sequence[str] | None): If provided, only include sections
                whose titles are in this list (case-insensitive).
            genre_filter (Sequence[str] | None): If provided, only include items that
                have at least one genre in this list.
        """
        self.log = logger

        self._url = url
        self._token = token
        self._home_user = home_user
        self._section_filter = {value.lower() for value in section_filter or ()}
        self._genre_filter = tuple(genre_filter or ())

        self._user_client: PlexServer | None = None
        self._account: MyPlexAccount | None = None
        self._user_id: int | None = None
        self._display_name: str | None = None

        self._sections: list[MovieSection | ShowSection] = []
        self._continue_cache: dict[str, _FrozenCacheEntry] = {}
        self._ordering_cache: dict[int, Literal["tmdb", "tvdb", ""]] = {}
        self._watchlist_cache: _FrozenCacheEntry | None = None
        self._on_deck_window: timedelta | None = None

    @property
    def on_deck_window(self) -> timedelta | None:
        """Return the configured on-deck time window if available."""
        return self._on_deck_window

    async def initialize(self) -> None:
        """Establish the Plex session and prime provider caches."""
        (
            self._user_client,
            self._account,
            self._user_id,
            self._display_name,
        ) = await asyncio.to_thread(self._initialize_clients)

        self._sections = await asyncio.to_thread(
            lambda: [
                raw
                for raw in self._ensure_user_client().library.sections()
                if isinstance(raw, (MovieSection, ShowSection))
                and (
                    not self._section_filter
                    or raw.title.lower() in self._section_filter
                )
            ]
        )

        def _on_deck_window_sync() -> timedelta | None:
            user_client = self._user_client
            if user_client is None:
                return None
            try:
                window_value = user_client.settings.get("onDeckWindow").value
            except Exception:
                return None
            try:
                return timedelta(weeks=float(window_value))
            except TypeError, ValueError:
                return None

        self._on_deck_window = await asyncio.to_thread(_on_deck_window_sync)
        self.clear_cache()

    def _initialize_clients(
        self,
    ) -> tuple[PlexServer, MyPlexAccount, int, str]:
        session = requests.Session()
        parsed = urlparse(self._url)
        if parsed.scheme == "https":
            session = SelectiveVerifySession(
                whitelist=[parsed.hostname], logger=cast(Logger, self.log)
            )

        # Extract machineIdentifier from the identity endpoint (it's unauthenticated)
        identity = session.get(f"{self._url}/identity", timeout=10)
        try:
            machine_id = ElementTree.fromstring(identity.content).attrib.get(
                "machineIdentifier"
            )
            self.log.debug(f"Parsed Plex machineIdentifier '{machine_id}'")
        except Exception:
            self.log.error(f"Failed to parse Plex identity from {self._url}/identity")
            raise

        account = MyPlexAccount(token=self._token, session=session)
        if self._home_user:
            self.log.debug(
                f"Attempting to switch to Plex Home user '{self._home_user}'"
            )
            account = cast(
                MyPlexAccount,
                account.switchHomeUser(self._home_user),
            )
            if account.restricted:  # Supposedly means this is a managed home user
                self.log.debug(
                    f"Switched to managed Plex Home user '{account.username}' "
                    f"({account.id})"
                )
            else:
                self.log.debug(
                    f"Switched to Plex Home user '{account.username}' ({account.id})"
                )

        user_token = account.resource(machine_id).accessToken
        user_client = PlexServer(self._url, token=user_token, session=session)

        user_id = int(account.id)
        if user_id is None:
            raise ValueError("Unable to resolve Plex account id for the active user")

        display_name = next(
            (
                candidate
                for candidate in (account.username, account.email, account.title)
                if candidate
            ),
            f"Plex User ({user_id})",
        )

        return (user_client, account, user_id, display_name)

    async def close(self) -> None:
        """Release any held resources."""
        self._user_client = None
        self._account = None
        self._user_id = None
        self._display_name = None
        self._sections.clear()
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear cached continue-watching and ordering metadata."""
        self._continue_cache.clear()
        self._ordering_cache.clear()
        self._watchlist_cache = None

    @property
    def user_id(self) -> int:
        """Return the numeric Plex user id for the connected user."""
        if self._user_id is None:
            raise RuntimeError("Plex client has not been initialized")
        return self._user_id

    @property
    def display_name(self) -> str:
        """Return the display name for the connected user."""
        if self._display_name is None:
            raise RuntimeError("Plex client has not been initialized")
        return self._display_name

    @property
    def account(self) -> MyPlexAccount:
        """Return the active Plex account."""
        if self._account is None:
            raise RuntimeError("Plex client has not been initialized")
        return self._account

    @property
    def is_managed_user(self) -> bool:
        """Return whether the connected user is a managed Plex Home user."""
        return bool(self.account.restricted)

    @property
    def user_client(self) -> PlexServer:
        """Return the active Plex user client."""
        return self._ensure_user_client()

    def sections(self) -> Sequence[MovieSection | ShowSection]:
        """Return the cached list of Plex library sections."""
        return tuple(self._sections)

    async def list_section_items(
        self,
        section: LibrarySection,
        *,
        min_last_modified: datetime | None = None,
        require_watched: bool = False,
        keys: Sequence[str] | None = None,
        **kwargs,
    ) -> Sequence[Movie | Show]:
        """Return Plex media items that match the provided filters."""

        def _search_sync() -> tuple[Movie | Show, ...]:
            filters: list[dict] = []

            if min_last_modified is not None:
                reference_dt = min_last_modified.astimezone()
                if reference_dt is None:
                    reference_dt = datetime.now(tz=UTC)

                if section.type == "movie":
                    filters.append(
                        {
                            "or": [
                                {"lastViewedAt>>=": reference_dt},
                                {"lastRatedAt>>=": reference_dt},
                                {"addedAt>>=": reference_dt},
                                {"updatedAt>>=": reference_dt},
                            ]
                        }
                    )
                else:
                    filters.append(
                        {
                            "or": [
                                {"show.lastViewedAt>>=": reference_dt},
                                {"show.lastRatedAt>>=": reference_dt},
                                {"show.addedAt>>=": reference_dt},
                                {"show.updatedAt>>=": reference_dt},
                                {"season.lastViewedAt>>=": reference_dt},
                                {"season.lastRatedAt>>=": reference_dt},
                                {"season.addedAt>>=": reference_dt},
                                {"season.updatedAt>>=": reference_dt},
                                {"episode.lastViewedAt>>=": reference_dt},
                                {"episode.lastRatedAt>>=": reference_dt},
                                {"episode.addedAt>>=": reference_dt},
                                {"episode.updatedAt>>=": reference_dt},
                            ]
                        }
                    )

            if require_watched:
                epoch = datetime.fromtimestamp(0, tz=UTC)
                if section.type == "movie":
                    filters.append(
                        {
                            "or": [
                                {"viewCount>>": 0},
                                {"lastViewedAt>>": epoch},
                                {"lastRatedAt>>": epoch},
                            ]
                        }
                    )
                else:
                    filters.append(
                        {
                            "or": [
                                {"show.viewCount>>": 0},
                                {"show.lastViewedAt>>": epoch},
                                {"show.lastRatedAt>>": epoch},
                                {"season.viewCount>>": 0},
                                {"season.lastViewedAt>>": epoch},
                                {"season.lastRatedAt>>": epoch},
                                {"episode.viewCount>>": 0},
                                {"episode.lastViewedAt>>": epoch},
                                {"episode.lastRatedAt>>": epoch},
                            ]
                        }
                    )

            if self._genre_filter:
                filters.append({"genre": self._genre_filter})

            search_kwargs = dict(kwargs)
            if filters:
                search_kwargs["filters"] = {"and": filters}

            try:
                results = section.search(**search_kwargs)
            except Exception:
                return ()

            key_filter: frozenset[str] | None = (
                frozenset(str(k) for k in keys) if keys else None
            )

            items: list[Movie | Show] = []
            for item in results:
                if not isinstance(item, (Movie, Show)):
                    continue

                if key_filter is not None and str(item.ratingKey) not in key_filter:
                    continue

                items.append(item)

            return tuple(items)

        return await asyncio.to_thread(_search_sync)

    def is_on_continue_watching(
        self,
        section: LibrarySection,
        item: Video,
    ) -> bool:
        """Determine whether the given item appears in the Continue Watching hub."""
        self._ensure_user_client()

        cache_entry = self._continue_cache.get(str(section.key))
        now = monotonic()
        if cache_entry is None or cache_entry.expires_at <= now:
            rating_keys: set[str] = set()
            try:
                for continue_item in section.continueWatching():
                    if continue_item.ratingKey is not None:
                        rating_keys.add(str(continue_item.ratingKey))
            except Exception:
                rating_keys.clear()

            cache_entry = _FrozenCacheEntry(
                keys=frozenset(rating_keys),
                expires_at=monotonic() + 300,
            )
            self._continue_cache[str(section.key)] = cache_entry
        return str(item.ratingKey) in cache_entry.keys

    async def fetch_history(self, item: Video) -> Sequence[tuple[str, datetime]]:
        """Return the watch history for the given Plex item."""
        user_client = self._ensure_user_client()
        try:
            history_objects = await asyncio.to_thread(
                user_client.history,
                ratingKey=item.ratingKey,
                accountID=self.user_id,
                librarySectionID=item.librarySectionID,
            )
        except Exception:
            return []

        entries = [
            (
                str(record.ratingKey),
                normalize_local_datetime(record.viewedAt),
            )
            for record in history_objects
            if record.viewedAt is not None
        ]
        entries = [
            (key, viewed_at) for key, viewed_at in entries if viewed_at is not None
        ]
        return entries

    def is_on_watchlist(self, item: Video) -> bool:
        """Determine whether the given item appears in the user's watchlist."""
        now = monotonic()
        cache_entry = self._watchlist_cache
        if cache_entry is None or cache_entry.expires_at <= now:
            try:
                # Rating keys won't work here because watchlist items can exist outside
                # of the user's server. We'll use GUIDs as as substitute.
                keys = {
                    str(watch_item.guid)
                    for watch_item in self.account.watchlist()
                    if watch_item.guid is not None
                }
            except Exception:
                keys = set()

            cache_entry = _FrozenCacheEntry(
                keys=frozenset(keys),
                expires_at=monotonic() + 300,
            )
            self._watchlist_cache = cache_entry

        return item.guid is not None and item.guid in cache_entry.keys

    def get_ordering(self, show: Show) -> Ordering:
        """Return the preferred episode ordering for the provided show."""
        if show.showOrdering:
            if show.showOrdering == "tmdbAiring":
                return "tmdb"
            if show.showOrdering in {"tvdbAiring", "aired"}:
                return "tvdb"
            return ""

        cached = self._ordering_cache.get(show.librarySectionID)
        if cached is not None:
            return cached

        ordering_setting = next(
            (
                setting
                for setting in show.section().settings()
                if setting.id == "showOrdering"
            ),
            None,
        )
        if not ordering_setting:
            resolved = ""
        else:
            value = ordering_setting.value
            if value == "tmdbAiring":
                resolved = "tmdb"
            elif value in {"aired", "tvdbAiring"}:
                resolved = "tvdb"
            else:
                resolved = ""

        self._ordering_cache[show.librarySectionID] = resolved
        return resolved

    def _ensure_user_client(self) -> PlexServer:
        """Ensure the user Plex client is available."""
        if self._user_client is None:
            raise RuntimeError("Plex client has not been initialized")
        return self._user_client
