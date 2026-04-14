"""Plex library provider implementation."""

import contextlib
import itertools
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, cast

import plexapi.library as plexapi_library
import plexapi.video as plexapi_video
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
)
from anibridge.library.base import MappingDescriptor
from anibridge.utils.cache import cache, ttl_cache
from anibridge.utils.datetime import normalize_local_datetime
from anibridge.utils.types import ProviderLogger

from anibridge.providers.library.plex.client import PlexClient
from anibridge.providers.library.plex.community import PlexCommunityClient
from anibridge.providers.library.plex.config import PlexProviderConfig
from anibridge.providers.library.plex.webhook import PlexWebhookEventType, WebhookParser

if TYPE_CHECKING:
    from starlette.requests import Request

_GUID_NAMESPACE_MAP: dict[MediaKind, dict[str, str]] = {
    MediaKind.MOVIE: {
        "imdb": "imdb_movie",
        "tmdb": "tmdb_movie",
        "tvdb": "tvdb_movie",
        "com.plexapp.agents.imdb": "imdb_movie",
        "com.plexapp.agents.themoviedb": "tmdb_movie",
        "com.plexapp.agents.tmdb": "tmdb_movie",
        "com.plexapp.agents.thetvdb": "tvdb_movie",
    },
    MediaKind.SHOW: {
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
        """Represent a Plex library section."""
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
        """Initialize the media wrapper."""
        self._provider = provider
        self._section = section
        self._item = item
        self._media_kind = kind

        self._key = str(item.guid) if item.guid else str(item.ratingKey)
        self._title = item.title

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
        with contextlib.suppress(Exception):
            return self._provider._client.get_thumb_url(self._item)


class PlexLibraryEntry(LibraryEntry):
    """Common behaviour for Plex-backed library objects."""

    def __init__(
        self,
        provider: PlexLibraryProvider,
        section: PlexLibrarySection,
        item: plexapi_video.Video,
        kind: MediaKind,
    ) -> None:
        """Initialize the media wrapper."""
        self._provider = provider
        self._section = section
        self._item = item
        self._media_kind = kind

        self._key = str(item.ratingKey)
        self._title = item.title
        self._media = PlexLibraryMedia(provider, section, item, kind)

    def mapping_descriptors(self) -> Sequence[MappingDescriptor]:
        """Return mapping descriptors for this media item."""
        raw_guids = [self._item.guid or ""] + [g.id for g in self._item.guids if g.id]
        descriptors: list[MappingDescriptor] = []
        for guid in raw_guids:
            if not guid or "://" not in guid:
                continue
            prefix, suffix = guid.split("://", 1)
            guid_namespace = _GUID_NAMESPACE_MAP.get(self._media_kind, {}).get(prefix)
            if guid_namespace is None:
                continue
            # Strip query params
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
        except TypeError, ValueError:
            return None

    @property
    def view_count(self) -> int:
        """Return the number of times this media item has been viewed."""
        return self._item.viewCount or 0

    async def history(self) -> Sequence[HistoryEntry]:
        """Fetch the viewing history for this media item."""
        return await self._provider.get_history(self._item)

    def media(self) -> PlexLibraryMedia:
        """Return the media metadata for this item."""
        return self._media

    @property
    async def review(self) -> str | None:
        """Fetch the user's review for this media item, if available."""
        return await self._provider.get_review(self._item)

    def section(self) -> PlexLibrarySection:
        """Return the library section this media item belongs to."""
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
        """Initialize the movie wrapper."""
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
        """Initialize the show wrapper."""
        super().__init__(provider, section, item, MediaKind.SHOW)
        self._item = cast(plexapi_video.Show, self._item)

    def episodes(self) -> Sequence[PlexLibraryEpisode]:
        """Return all episodes belonging to the show."""
        seasons = self.seasons()
        return tuple(episode for season in seasons for episode in season.episodes())

    @ttl_cache(ttl=15)
    def seasons(self) -> Sequence[PlexLibrarySeason]:
        """Return all seasons belonging to the show."""
        return tuple(
            PlexLibrarySeason(self._provider, self._section, season, show=self)
            for season in self._item.seasons()
        )

    def mapping_descriptors(self) -> Sequence[MappingDescriptor]:
        """Return mapping descriptors for this show."""
        descriptors = super().mapping_descriptors()
        ordering = self._provider._client.get_ordering(
            cast(plexapi_video.Show, self._item)
        )
        # If strict matching is enabled, filter to only the preferred ordering
        if self._provider.parsed_config.strict:
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
        """Initialize the season wrapper."""
        super().__init__(provider, section, item, MediaKind.SEASON)
        self._item = cast(plexapi_video.Season, self._item)
        self._show = show
        self.index = self._item.index

    @ttl_cache(15)
    def episodes(self) -> Sequence[PlexLibraryEpisode]:
        """Return the episodes belonging to this season."""
        return tuple(
            PlexLibraryEpisode(
                self._provider, self._section, episode, season=self, show=self._show
            )
            for episode in self._item.episodes()
        )

    @cache
    def show(self) -> PlexLibraryShow:
        """Return the parent show."""
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
        """Initialize the episode wrapper."""
        super().__init__(provider, section, item, MediaKind.EPISODE)
        self._item = cast(plexapi_video.Episode, self._item)
        self._show = show
        self._season = season
        self.index = self._item.index
        self.season_index = self._item.parentIndex

    @cache
    def season(self) -> PlexLibrarySeason:
        """Return the parent season."""
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
            show=self.show(),
        )
        return self._season

    @cache
    def show(self) -> PlexLibraryShow:
        """Return the parent show."""
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


class PlexLibraryProvider(LibraryProvider):
    """Default Plex `LibraryProvider` backed by the local Plex Media Server."""

    NAMESPACE = "plex"

    def __init__(self, *, logger: ProviderLogger, config: dict | None = None) -> None:
        """Parse configuration and prepare provider defaults."""
        super().__init__(logger=logger, config=config)
        self.parsed_config = PlexProviderConfig.model_validate(config or {})

        self._client = PlexClient(
            logger=self.log,
            url=self.parsed_config.url,
            token=self.parsed_config.token,
            home_user=self.parsed_config.home_user,
            section_filter=self.parsed_config.sections,
            genre_filter=self.parsed_config.genres,
        )
        self._community_client: PlexCommunityClient | None = None

        self._user: LibraryUser | None = None

        self._sections: list[PlexLibrarySection] = []
        self._section_map: dict[str, PlexLibrarySection] = {}

    async def initialize(self) -> None:
        """Connect to Plex and prepare provider state."""
        self.log.debug("Initializing Plex provider client")
        await self._client.initialize()
        self._user = LibraryUser(
            key=str(self._client.user_id),
            title=self._client.display_name,
        )

        self._sections = self._build_sections()

        # Managed users don't have access to the Plex Community API
        if not self._client.is_managed_user:
            self._community_client = PlexCommunityClient(
                plex_token=self._client.account.authToken,
                logger=self.log.getChild("community_client"),
            )

        self.log.debug(
            "Plex provider initialized for user id=%s with %s sections",
            self._user.key,
            len(self._sections),
        )

    async def close(self) -> None:
        """Release any resources held by the provider."""
        self.log.debug("Closing Plex provider")
        await self._client.close()
        if self._community_client is not None:
            await self._community_client.close()
            self._community_client = None
        self._sections.clear()
        self._section_map.clear()
        self.log.debug("Closed Plex provider")

    def user(self) -> LibraryUser | None:
        """Return the Plex account represented by this provider."""
        return self._user

    async def get_sections(self) -> Sequence[LibrarySection]:
        """Enumerate Plex library sections visible to the provider user."""
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
        """
        if not isinstance(section, PlexLibrarySection):
            self.log.warning(
                "Plex list_items received an incompatible section instance"
            )
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
        if not self._user:
            self.log.error(
                "Webhook: Received webhook request before provider initialization"
            )
            raise RuntimeError("Provider must be initialized before parsing webhooks")

        payload = await WebhookParser.from_request(request)

        if not payload.account_id:
            self.log.warning("Webhook: No account ID found in payload")
            raise ValueError("No account ID found in webhook payload")
        if not payload.top_level_rating_key:
            self.log.warning("Webhook: No rating key found in payload")
            raise ValueError("No rating key found in webhook payload")

        webhook_account_id = payload.account_id
        user_matches = (
            self._client.account_id == webhook_account_id
            or self._client.user_id == webhook_account_id
        )

        if (
            payload.event_type
            in (
                PlexWebhookEventType.MEDIA_ADDED,
                PlexWebhookEventType.RATE,
                PlexWebhookEventType.SCROBBLE,
                PlexWebhookEventType.STOP,
            )
            and user_matches
        ):
            if (
                payload.section_key is not None
                and payload.section_key not in self._section_map
            ):
                self.log.debug(
                    "Webhook: Ignoring webhook for rating key %s because section key "
                    "%s is not in the provider's enabled sections",
                    payload.top_level_rating_key,
                    payload.section_key,
                )
                return (False, tuple())

            self.log.info(
                "Matched webhook event %s to provider user ID %s for sync",
                payload.event_type,
                self._user.key,
            )
            return (True, (payload.top_level_rating_key,))

        self.log.debug(
            "Ignoring webhook event %s for account ID %s",
            payload.event_type,
            payload.account_id,
        )
        return (False, tuple())

    async def clear_cache(self) -> None:
        """Reset any cached Plex responses maintained by the provider."""
        self._client.clear_cache()
        # Note this clears the class level caches, which will be a no-op since the
        # caches used here are instance level. However, I'm leaving this in place
        # in case anibridge-utils caches support instance level cache clearing from
        # the class call in the future.
        PlexLibraryShow.seasons.cache_clear()
        PlexLibrarySeason.episodes.cache_clear()

    def is_on_continue_watching(
        self, section: PlexLibrarySection, item: plexapi_video.Video
    ) -> bool:
        """Determine whether the given item appears in the Continue Watching hub."""
        return self._client.is_on_continue_watching(section._section, item)

    def is_on_watchlist(self, item: plexapi_video.Video) -> bool:
        """Determine whether the given item appears in the user's watchlist."""
        return self._client.is_on_watchlist(item)

    async def get_review(self, item: plexapi_video.Video) -> str | None:
        """Fetch the user's review for the provided Plex item, if available."""
        if item.userRating is None and item.lastRatedAt is None:  # Prereq for reviews
            return None
        if self._community_client is None or not item.guid:
            return None
        metadata_id = item.guid.rsplit("/", 1)[-1]
        try:
            return await self._community_client.get_reviews(metadata_id)
        except Exception:
            self.log.exception("Failed to fetch Plex review")
            return None

    async def get_history(self, item: plexapi_video.Video) -> Sequence[HistoryEntry]:
        """Return the watch history for the given Plex item."""
        plex_history = await self._client.fetch_history(item)

        if isinstance(item, (plexapi_video.Show, plexapi_video.Season)):
            children_iter = item.episodes()
        else:
            children_iter = (item,)

        children = list(children_iter)

        if not children:
            # python-plexapi returns a naive datetime in the local timezone (no tzinfo)
            return tuple(
                HistoryEntry(
                    library_key=rating_key,
                    viewed_at=cast(datetime, normalize_local_datetime(viewed_at)),
                )
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
                normalize_local_datetime(child.lastViewedAt)
                if child.lastViewedAt
                else None
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
    ) -> PlexLibraryEntry:
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
