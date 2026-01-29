"""Plex library provider implementation."""

import base64
import itertools
from collections.abc import Sequence
from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, cast

import plexapi.library as plexapi_library
import plexapi.video as plexapi_video
import requests
from anibridge.library import (
    HistoryEntry,
    LibraryEntry,
    LibraryEpisode,
    LibraryMedia,
    LibraryMovie,
    LibraryProvider,
    LibrarySeason,
    LibrarySection,
    LibraryShow,
    LibraryUser,
    MediaKind,
    library_provider,
)
from anibridge.library.base import MappingDescriptor

from anibridge_plex_provider.client import PlexClient, PlexClientConfig
from anibridge_plex_provider.community import PlexCommunityClient
from anibridge_plex_provider.webhook import PlexWebhook, PlexWebhookEventType

if TYPE_CHECKING:
    from starlette.requests import Request

_LOG = getLogger(__name__)

_GUID_NAMESPACE_MAP = {
    "movie": {
        "imdb": "imdb_movie",
        "tmdb": "tmdb_movie",
        "tvdb": "tvdb_movie",
        "com.plexapp.agents.imdb": "imdb_movie",
        "com.plexapp.agents.themoviedb": "tmdb_movie",
        "com.plexapp.agents.tmdb": "tmdb_movie",
        "com.plexapp.agents.thetvdb": "tvdb_movie",
    },
    "show": {
        "imdb": "imdb_show",
        "tmdb": "tmdb_show",
        "tvdb": "tvdb_show",
        "com.plexapp.agents.imdb": "imdb_show",
        "com.plexapp.agents.themoviedb": "tmdb_show",
        "com.plexapp.agents.tmdb": "tmdb_show",
        "com.plexapp.agents.thetvdb": "tvdb_show",
    },
}


class PlexLibrarySection(LibrarySection):
    """Concrete `LibrarySection` backed by a python-plexapi library section."""

    def __init__(
        self, provider: PlexLibraryProvider, item: plexapi_library.LibrarySection
    ) -> None:
        """Represent a Plex library section.

        Args:
            provider (PlexLibraryProvider): The owning Plex library provider.
            item (plexapi_library.LibrarySection): The underlying Plex section.
        """
        self._provider = provider
        self._section = item

        self._key = str(item.key)
        self._title = item.title
        self._media_kind = MediaKind.SHOW if item.type == "show" else MediaKind.MOVIE


class PlexLibraryMedia(LibraryMedia):
    """The base class for Plex media objects (metdata focused)."""

    def __init__(
        self,
        provider: PlexLibraryProvider,
        section: PlexLibrarySection,
        item: plexapi_video.Video,
        kind: MediaKind,
    ) -> None:
        """Initialize the media wrapper.

        Args:
            provider (PlexLibraryProvider): The owning Plex library provider.
            section (PlexLibrarySection): The parent Plex library section.
            item (plexapi_video.Video): The underlying Plex media item.
            kind (MediaKind): The kind of media represented.
        """
        self._provider = provider
        self._section = section
        self._item = item
        self._media_kind = kind

    @property
    def external_url(self) -> str | None:
        """URL to the Plex online page, if available."""
        if not self._item.guid:
            return None
        key = f"/library/metadata/{self._item.guid.rsplit('/', 1)[-1]}"
        return f"https://app.plex.tv/desktop/#!/provider/tv.plex.provider.discover/details?key={key}"

    @property
    def poster_image(self) -> str | None:
        """Return a base64 data URL for the item's poster artwork if available.

        We need to encode the image as a data URL because Plex requires authentication,
        so direct linking would expose the token in client image URLs.
        """
        if not self._item.thumb:
            return None

        try:
            bundle = self._provider._client.bundle()
            url = bundle.user_client.url(self._item.thumb, includeToken=True)

            # Low timeout because this is low priority
            response = requests.get(url, timeout=3)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "image/jpeg")
            encoded = base64.b64encode(response.content).decode("utf-8")

            return f"data:{content_type};base64,{encoded}"

        except Exception:
            return None


class PlexLibraryEntry(LibraryEntry):
    """Common behaviour for Plex-backed library objects."""

    def __init__(
        self,
        provider: PlexLibraryProvider,
        section: PlexLibrarySection,
        item: plexapi_video.Video,
        kind: MediaKind,
    ) -> None:
        """Initialize the media wrapper.

        Args:
            provider (PlexLibraryProvider): The owning Plex library provider.
            section (PlexLibrarySection): The parent Plex library section.
            item (plexapi_video.Video): The underlying Plex media item.
            kind (MediaKind): The kind of media represented.
        """
        self._provider = provider
        self._section = section
        self._item = item
        self._media_kind = kind
        self._media = PlexLibraryMedia(provider, section, item, kind)
        self._key = str(item.ratingKey)
        self._title = item.title

    def mapping_descriptors(self) -> Sequence[MappingDescriptor]:
        """Return mapping descriptors for this media item.

        Returns:
            Sequence[MappingDescriptor]: The mapping descriptors.
        """
        descriptors: list[MappingDescriptor] = []
        for guid in self._item.guids:
            if not guid.id or "://" not in guid.id:
                continue
            prefix, suffix = guid.id.split("://", 1)
            guid_namespace = _GUID_NAMESPACE_MAP.get(self._media_kind.value, {}).get(
                prefix
            )
            if guid_namespace is None:
                continue
            descriptors.append((guid_namespace, suffix.split("?", 1)[0], None))
        return descriptors

    @property
    def on_watching(self) -> bool:
        """Check if the media item is on the user's current watching list."""
        return self._provider.is_on_continue_watching(self._section, self._item)

    @property
    def on_watchlist(self) -> bool:
        """Check if the media item is on the user's watchlist."""
        return self._provider.is_on_watchlist(self._item)

    @property
    def user_rating(self) -> int | None:
        """Return the user rating for this media item on a 0-100 scale."""
        if self._item.userRating is None:
            return None
        try:
            # Normalize to a 0-100 scale
            return round(float(self._item.userRating) * 10)
        except (TypeError, ValueError):
            return None

    @property
    def view_count(self) -> int:
        """Return the number of times this media item has been viewed."""
        return self._item.viewCount or 0

    async def history(self) -> Sequence[HistoryEntry]:
        """Fetch the viewing history for this media item.

        Returns:
            Sequence[HistoryEntry]: A sequence of history entries for this media item.
        """
        return await self._provider.get_history(self._item)

    def media(self) -> LibraryMedia:
        """Return the media metadata for this item.

        Returns:
            LibraryMedia: The media metadata.
        """
        return self._media

    async def review(self) -> str | None:
        """Fetch the user's review for this media item, if available.

        Returns:
            str | None: The user's review text, or None if not reviewed.
        """
        return await self._provider.get_review(self._item)

    def section(self) -> PlexLibrarySection:
        """Return the library section this media item belongs to.

        Returns:
            PlexLibrarySection: The parent library section.
        """
        if self._section is not None:
            return self._section

        raw_section = self._item.section()
        self._section = PlexLibrarySection(self._provider, raw_section)
        return self._section


class PlexLibraryMovie(PlexLibraryEntry, LibraryMovie):
    """Concrete `LibraryMovie` wrapper for python-plexapi `Movie` objects."""

    __slots__ = ()

    def __init__(
        self,
        provider: PlexLibraryProvider,
        section: PlexLibrarySection,
        item: plexapi_video.Movie,
    ) -> None:
        """Initialize the movie wrapper.

        Args:
            provider (PlexLibraryProvider): The owning Plex library provider.
            section (PlexLibrarySection): The parent Plex library section.
            item (plexapi_video.Movie): The underlying Plex movie item.
        """
        super().__init__(provider, section, item, MediaKind.MOVIE)
        self._item = cast(plexapi_video.Movie, self._item)


class PlexLibraryShow(PlexLibraryEntry, LibraryShow):
    """Concrete `LibraryShow` wrapper for Plex `Show` objects."""

    __slots__ = ()

    def __init__(
        self,
        provider: PlexLibraryProvider,
        section: PlexLibrarySection,
        item: plexapi_video.Show,
    ) -> None:
        """Initialize the show wrapper.

        Args:
            provider (PlexLibraryProvider): The owning Plex library provider.
            section (PlexLibrarySection): The parent Plex library section.
            item (plexapi_video.Show): The underlying Plex show item.
        """
        super().__init__(provider, section, item, MediaKind.SHOW)
        self._item = cast(plexapi_video.Show, self._item)

    def episodes(self) -> Sequence[PlexLibraryEpisode]:
        """Return all episodes belonging to the show.

        Returns:
            Sequence[PlexLibraryEpisode]: All episodes in the show.
        """
        return [
            cast(PlexLibraryEpisode, self._provider._wrap_entry(self._section, episode))
            for episode in self._item.episodes()
        ]

    def seasons(self) -> Sequence[PlexLibrarySeason]:
        """Return all seasons belonging to the show.

        Returns:
            Sequence[PlexLibrarySeason]: All seasons in the show.
        """
        return tuple(
            PlexLibrarySeason(self._provider, self._section, season, show=self)
            for season in self._item.seasons()
        )

    def mapping_descriptors(self) -> Sequence[MappingDescriptor]:
        """Return mapping descriptors for this show.

        Includes additional logic to prefer show ordering.

        Returns:
            Sequence[MappingDescriptor]: The mapping descriptors.
        """
        descriptors = super().mapping_descriptors()
        ordering = self._provider._client.get_ordering(
            cast(plexapi_video.Show, self._item)
        )
        # If strict matching is enabled, filter to only the preferred ordering
        if self._provider._strict:
            if ordering == "tmdb":
                descriptors = tuple(
                    d for d in descriptors if d[0] in ("tmdb_show", "tmdb_movie")
                )
            elif ordering == "tvdb":
                descriptors = tuple(
                    d for d in descriptors if d[0] in ("tvdb_show", "tvdb_movie")
                )
            return descriptors

        # Otherwise, sort to prefer the ordering
        def sort_key(descriptor: MappingDescriptor) -> int:
            if ordering == "tmdb" and descriptor[0] in ("tmdb_show", "tmdb_movie"):
                return 0
            if ordering == "tvdb" and descriptor[0] in ("tvdb_show", "tvdb_movie"):
                return 0
            return 1

        return tuple(sorted(descriptors, key=sort_key))


class PlexLibrarySeason(PlexLibraryEntry, LibrarySeason):
    """Concrete `LibrarySeason` wrapper for Plex `Season` objects."""

    def __init__(
        self,
        provider: PlexLibraryProvider,
        section: PlexLibrarySection,
        item: plexapi_video.Season,
        *,
        show: PlexLibraryShow | None = None,
    ) -> None:
        """Initialize the season wrapper.

        Args:
            provider (PlexLibraryProvider): The owning Plex library provider.
            section (PlexLibrarySection): The parent Plex library section.
            item (plexapi_video.Season): The underlying Plex season item.
            show (PlexLibraryShow | None): The parent show, if known.
        """
        super().__init__(provider, section, item, MediaKind.SEASON)
        self._item = cast(plexapi_video.Season, self._item)
        self._show = show
        self.index = self._item.index

    def episodes(self) -> Sequence[LibraryEpisode]:
        """Return the episodes belonging to this season.

        Returns:
            Sequence[LibraryEpisode]: All episodes in the season.
        """
        return tuple(
            PlexLibraryEpisode(
                self._provider, self._section, episode, season=self, show=self._show
            )
            for episode in self._item.episodes()
        )

    def show(self) -> LibraryShow:
        """Return the parent show.

        Returns:
            LibraryShow: The parent show.
        """
        if self._show is not None:
            return self._show

        raw_parent = self._item._parent() if self._item._parent else None
        raw_show = (
            cast(plexapi_video.Show, raw_parent)
            if isinstance(raw_parent, plexapi_video.Show)
            else self._item.show()
        )
        self._show = PlexLibraryShow(self._provider, self._section, raw_show)
        return self._show

    def mapping_descriptors(self) -> Sequence[MappingDescriptor]:
        """Return mapping descriptors with season scopes applied."""
        descriptors: list[MappingDescriptor] = []
        for descriptor in self.show().mapping_descriptors():
            descriptors.append((descriptor[0], descriptor[1], f"s{self.index}"))
        return tuple(descriptors)


class PlexLibraryEpisode(PlexLibraryEntry, LibraryEpisode):
    """Concrete `LibraryEpisode` wrapper for Plex `Episode` objects."""

    def __init__(
        self,
        provider: PlexLibraryProvider,
        section: PlexLibrarySection,
        item: plexapi_video.Episode,
        *,
        season: PlexLibrarySeason | None = None,
        show: PlexLibraryShow | None = None,
    ) -> None:
        """Initialize the episode wrapper.

        Args:
            provider (PlexLibraryProvider): The owning Plex library provider.
            section (PlexLibrarySection): The parent Plex library section.
            item (plexapi_video.Episode): The underlying Plex episode item.
            season (PlexLibrarySeason | None): The parent season, if known.
            show (PlexLibraryShow | None): The parent show, if known.
        """
        super().__init__(provider, section, item, MediaKind.EPISODE)
        self._item = cast(plexapi_video.Episode, self._item)
        self._show = show
        self._season = season
        self.index = self._item.index
        self.season_index = self._item.parentIndex

    def season(self) -> LibrarySeason:
        """Return the parent season.

        Returns:
            LibrarySeason: The parent season.
        """
        if self._season is not None:
            return self._season

        raw_parent = self._item._parent
        if isinstance(raw_parent, plexapi_video.Season):
            raw_season = cast(plexapi_video.Season, raw_parent)
        else:
            raw_season = self._item.season()

        self._season = PlexLibrarySeason(
            self._provider,
            self._section,
            raw_season,
            show=cast(PlexLibraryShow, self.show()),
        )
        return self._season

    def show(self) -> LibraryShow:
        """Return the parent show.

        Returns:
            LibraryShow: The parent show.
        """
        if self._show is not None:
            return self._show

        raw_parent = self._item._parent() if self._item._parent else None
        raw_grandparent = (
            raw_parent._parent() if raw_parent and raw_parent._parent else None
        )

        if isinstance(raw_parent, plexapi_video.Show):
            raw_show = raw_parent
        elif isinstance(raw_grandparent, plexapi_video.Show):
            raw_show = raw_grandparent
        else:
            raw_show = self._item.show()

        self._show = PlexLibraryShow(self._provider, self._section, raw_show)
        return self._show

    def mapping_descriptors(self) -> Sequence[MappingDescriptor]:
        """Return mapping descriptors with season scopes applied."""
        return self.season().mapping_descriptors()


@library_provider
class PlexLibraryProvider(LibraryProvider):
    """Default Plex `LibraryProvider` backed by the local Plex Media Server."""

    NAMESPACE = "plex"

    def __init__(self, *, config: dict | None = None) -> None:
        """Parse configuration and prepare provider defaults.

        Args:
            config (dict | None): Optional configuration options for the provider.
        """
        self.config = config or {}

        url = self.config.get("url") or ""
        token = self.config.get("token") or ""
        user = self.config.get("user") or ""
        if not url or not token or not user:
            raise ValueError(
                "The Plex provider requires 'url', 'token', and 'user' configuration "
                "values"
            )

        self._plex_url = str(url)
        self._plex_token = str(token)
        self._plex_user = str(user)

        self._client_config = PlexClientConfig(url=url, token=token, user=user)

        self._strict = bool(self.config.get("strict", True))
        self._section_filter = list(self.config.get("sections") or [])
        self._genre_filter = list(self.config.get("genres") or [])

        self._client = self._create_client()
        self._community_client: PlexCommunityClient | None = None

        self._is_admin_user = False
        self._user: LibraryUser | None = None

        self._sections: list[PlexLibrarySection] = []
        self._section_map: dict[str, PlexLibrarySection] = {}

    async def initialize(self) -> None:
        """Connect to Plex and prepare provider state."""
        await self._client.initialize()
        bundle = self._client.bundle()
        self._is_admin_user = bundle.is_admin
        self._user = LibraryUser(key=str(bundle.user_id), title=bundle.display_name)

        self._sections = self._build_sections()
        self._community_client = PlexCommunityClient(self._client_config.token)

        await self.clear_cache()

    async def close(self) -> None:
        """Release any resources held by the provider."""
        await self._client.close()
        if self._community_client is not None:
            await self._community_client.close()
            self._community_client = None
        self._sections.clear()
        self._section_map.clear()

    def user(self) -> LibraryUser | None:
        """Return the Plex account represented by this provider.

        Returns:
            LibraryUser | None: The user information, or None if not available.
        """
        return self._user

    async def get_sections(self) -> Sequence[LibrarySection]:
        """Enumerate Plex library sections visible to the provider user.

        Returns:
            Sequence[LibrarySection]: Available library sections.
        """
        return tuple(self._sections)

    async def list_items(
        self,
        section: LibrarySection,
        *,
        min_last_modified: datetime | None = None,
        require_watched: bool = False,
        keys: Sequence[str] | None = None,
    ) -> Sequence[LibraryEntry]:
        """List items in a Plex library section matching the provided criteria.

        Each item returned must belong to the specified section and meet the provided
        filtering criteria.

        Args:
            section (LibrarySection): The library section to list items from.
            min_last_modified (datetime | None): If provided, only items modified after
                this timestamp will be included.
            require_watched (bool): If True, only include items that have been marked as
                watched/viewed.
            keys (Sequence[str] | None): If provided, only include items whose
                media keys are in this list.

        Returns:
            Sequence[LibraryEntry]: The entries matching the criteria.
        """
        if not isinstance(section, PlexLibrarySection):
            raise TypeError(
                "Plex providers expect section objects created by the provider"
            )

        raw_items = await self._client.list_section_items(
            section._section,
            min_last_modified=min_last_modified,
            require_watched=require_watched,
            keys=keys,
        )
        return tuple(self._wrap_entry(section, item) for item in raw_items)

    async def parse_webhook(self, request: Request) -> tuple[bool, Sequence[str]]:
        """Parse a Plex webhook request and determine affected media items."""
        payload = await PlexWebhook.from_request(request)

        if not payload.account_id:
            _LOG.debug("Webhook: No account ID found in payload")
            raise ValueError("No account ID found in webhook payload")
        if not payload.top_level_rating_key:
            _LOG.debug("Webhook: No rating key found in payload")
            raise ValueError("No rating key found in webhook payload")

        if (
            payload.event
            in (
                PlexWebhookEventType.MEDIA_ADDED,
                PlexWebhookEventType.RATE,
                PlexWebhookEventType.SCROBBLE,
            )
            and self._user
            and self._user.key == str(payload.account_id)
        ):
            _LOG.info(
                f"Webhook: Matched webhook event {payload.event_type} to provider user "
                f"ID {self._user.key} for sync"
            )
            return (True, (payload.top_level_rating_key,))

        _LOG.debug(
            f"Webhook: Ignoring event {payload.event_type} for account ID "
            f"{payload.account_id}"
        )
        return (False, tuple())

    async def clear_cache(self) -> None:
        """Reset any cached Plex responses maintained by the provider."""
        self._client.clear_cache()

    def is_on_continue_watching(
        self, section: PlexLibrarySection, item: plexapi_video.Video
    ) -> bool:
        """Determine whether the given item appears in the Continue Watching hub.

        Args:
            section (PlexLibrarySection): The library section the item belongs to.
            item (plexapi_video.Video): The Plex media item to check.

        Returns:
            bool: True if the item is on the Continue Watching list, False otherwise.
        """
        return self._client.is_on_continue_watching(section._section, item)

    def is_on_watchlist(self, item: plexapi_video.Video) -> bool:
        """Determine whether the given item appears in the user's watchlist.

        Args:
            item (plexapi_video.Video): The Plex media item to check.

        Returns:
            bool: True if the item is on the watchlist, False otherwise.
        """
        return self._client.is_on_watchlist(item)

    async def get_review(self, item: plexapi_video.Video) -> str | None:
        """Fetch the user's review for the provided Plex item, if available.

        Args:
            item (plexapi_video.Video): The Plex media item to fetch the review for.

        Returns:
            str | None: The user's review text, or None if not reviewed.
        """
        if not self._community_client or not self._is_admin_user:
            return None
        if item.userRating is None and item.lastRatedAt is None:  # Prereq for reviews
            return None
        if not item.guid:
            return None
        metadata_id = item.guid.rsplit("/", 1)[-1]
        try:
            return await self._community_client.get_reviews(metadata_id)
        except Exception:
            _LOG.debug("Failed to fetch Plex review", exc_info=True)
            return None

    async def get_history(self, item: plexapi_video.Video) -> Sequence[HistoryEntry]:
        """Return the watch history for the given Plex item.

        Args:
            item (plexapi_video.Video): The Plex media item to fetch history for.

        Returns:
            Sequence[HistoryEntry]: A sequence of history entries for the media item.
        """
        plex_history = await self._client.fetch_history(item)

        if isinstance(item, (plexapi_video.Show, plexapi_video.Season)):
            children_iter = item.episodes()
        else:
            children_iter = (item,)

        children = list(children_iter)

        if not children:
            # python-plexapi returns datetimes in the local timezone (without tzinfo)
            return tuple(
                HistoryEntry(library_key=rating_key, viewed_at=viewed_at.astimezone())
                for rating_key, viewed_at in plex_history
            )

        base_entries: list[HistoryEntry] = []
        base_keys = set()

        for rating_key, viewed_at in plex_history:
            base_keys.add(rating_key)
            base_entries.append(
                HistoryEntry(library_key=rating_key, viewed_at=viewed_at)
            )

        derived_children: list[HistoryEntry] = []

        for child in children:
            last_viewed = (
                child.lastViewedAt.astimezone() if child.lastViewedAt else None
            )
            if last_viewed is None:
                continue

            rating_key_str = str(child.ratingKey)
            if rating_key_str in base_keys:
                continue

            derived_children.append(
                HistoryEntry(
                    library_key=rating_key_str,
                    viewed_at=last_viewed,
                )
            )

        return tuple(itertools.chain(derived_children, base_entries))

    def _build_sections(self) -> list[PlexLibrarySection]:
        """Construct the list of Plex library sections available to the user."""
        sections: list[PlexLibrarySection] = []
        self._section_map.clear()

        for raw in self._client.sections():
            wrapper = PlexLibrarySection(self, raw)
            self._section_map[wrapper.key] = wrapper
            sections.append(wrapper)
        return sections

    def _wrap_entry(
        self, section: PlexLibrarySection, item: plexapi_video.Video
    ) -> LibraryEntry:
        """Wrap a Plex entry in the appropriate library entry class."""
        if isinstance(item, plexapi_video.Episode):
            return PlexLibraryEpisode(self, section, item)
        if isinstance(item, plexapi_video.Season):
            return PlexLibrarySeason(self, section, item)
        if isinstance(item, plexapi_video.Show):
            return PlexLibraryShow(self, section, item)
        if isinstance(item, plexapi_video.Movie):
            return PlexLibraryMovie(self, section, item)
        raise TypeError(f"Unsupported Plex media type: {type(item)!r}")

    def _create_client(self) -> PlexClient:
        """Construct and return a PlexClient for this provider."""
        return PlexClient(
            config=self._client_config,
            section_filter=self._section_filter,
            genre_filter=self._genre_filter,
        )
