"""Tests for the Plex library provider integration."""

from datetime import UTC, datetime
from logging import getLogger
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import pytest_asyncio
from anibridge.utils.types import ProviderLogger
from starlette.requests import Request

import anibridge.providers.library.plex.client as client_module
import anibridge.providers.library.plex.library as library_module


def _server_stub(**kwargs: Any) -> client_module.PlexServer:
    return cast(client_module.PlexServer, SimpleNamespace(**kwargs))


def _account_stub(**kwargs: Any) -> client_module.MyPlexAccount:
    return cast(client_module.MyPlexAccount, SimpleNamespace(**kwargs))


class StubBaseVideo:
    """Stub Plex base video object."""

    def __init__(self, rating_key: str, *, guid: str | None = None) -> None:
        """Initialize the stub video item."""
        self.ratingKey = rating_key
        self.title = f"Title-{rating_key}"
        self.guids = [
            SimpleNamespace(id="imdb://tt12345"),
            SimpleNamespace(id="com.plexapp.agents.thetvdb://42"),
        ]
        self.guid = guid or "plex://movie/1"
        self.thumb = "/thumb"
        self.userRating = 7.5
        self.lastRatedAt: datetime | None = None
        self.viewCount = 2
        self.librarySectionID = 1
        self.lastViewedAt = datetime.now(tz=UTC)
        self.parentIndex = 1
        self.index = 1
        self.on_deck = True
        self.watchlisted = True
        self._section = SimpleNamespace(settings=lambda: [])
        self._episodes: list[StubEpisode] = []
        self._seasons: list[StubSeason] = []
        self._show: StubShow | None = None
        self._season: StubSeason | None = None

    def section(self):
        """Return the section the item belongs to."""
        return self._section

    def episodes(self):
        """Return the episodes in the item."""
        return tuple(self._episodes)

    def seasons(self):
        """Return the seasons in the item."""
        return tuple(self._seasons)

    def _parent(self):
        return self._season or self._show

    def show(self):
        """Return the show the item belongs to."""
        return self._show or self

    def season(self):
        """Return the season the item belongs to."""
        return self._season


class StubMovie(StubBaseVideo):
    """Stub Plex Movie object."""

    pass


class StubShow(StubBaseVideo):
    """Stub Plex Show object."""

    pass


class StubSeason(StubBaseVideo):
    """Stub Plex Season object."""

    def __init__(self, rating_key: str, show: StubShow) -> None:
        """Initialize the stub season item."""
        super().__init__(rating_key)
        self._show = show

    def episodes(self):
        """Return the episodes in the season."""
        return tuple(self._episodes)

    def show(self):
        """Return the show the season belongs to."""
        if self._show is None:
            raise ValueError("Season has no show assigned")
        return self._show


class StubEpisode(StubBaseVideo):
    """Stub Plex Episode object."""

    def __init__(self, rating_key: str, season: StubSeason, show: StubShow) -> None:
        """Initialize the stub episode item."""
        super().__init__(rating_key)
        self._season = season
        self._show = show


class FakeRawSection:
    """Stub for a raw Plex library section."""

    def __init__(self, title: str, media_type: str) -> None:
        """Initialize the fake raw section."""
        self.title = title
        self.type = media_type
        self.key = f"section-{title}"


class FakePlexClient:
    """Stub for a Plex client."""

    def __init__(
        self,
        *,
        sections: list[FakeRawSection],
        items: list[StubBaseVideo],
    ) -> None:
        """Initialize the fake Plex client."""
        self._sections = sections
        self._items = items
        self._history = [("derived", datetime.now(tz=UTC))]
        self._user_client = _server_stub(
            url=lambda path, includeToken=True: f"https://plex{path}",
        )
        self._account = _account_stub(id=1, authToken="token", watchlist=lambda: [])
        self._user_id = 1
        self._display_name = "Demo"
        self._is_managed_user = False
        self._helper = client_module.PlexClient(
            logger=cast(ProviderLogger, getLogger("test.library.client")),
            url="https://plex.example",
            token="token",
        )
        self.initialized = False
        self.closed = False
        self.cleared = False

    async def initialize(self) -> None:
        """Simulate client initialization."""
        self.initialized = True

    async def close(self) -> None:
        """Simulate client closure."""
        self.closed = True

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def user_client(self):
        return self._user_client

    @property
    def account(self):
        return self._account

    @property
    def is_managed_user(self) -> bool:
        return self._is_managed_user

    def sections(self):
        """Return the library sections."""
        return list(self._sections)

    async def list_section_items(self, *_, **__):
        """Return the items in the section."""
        return tuple(self._items)

    def clear_cache(self) -> None:
        """Simulate clearing the client's cache."""
        self.cleared = True

    def is_on_continue_watching(self, section, item) -> bool:
        """Check if the item is on the continue watching list."""
        return getattr(item, "on_deck", False)

    def is_on_watchlist(self, item) -> bool:
        """Check if the item is on the watchlist."""
        return getattr(item, "watchlisted", False)

    async def fetch_history(self, item):
        """Return the watch history for the item."""
        return list(self._history)

    def get_ordering(self, _show) -> str:
        return "airdate"


class StubCommunityClient:
    """Stub for the PlexCommunityClient."""

    instances: ClassVar[list[StubCommunityClient]] = []

    def __init__(self, plex_token: str, *, logger=None) -> None:
        """Initialize the stub community client."""
        self.token = plex_token
        self.logger = logger
        self.calls: list[str] = []
        self.closed = False
        StubCommunityClient.instances.append(self)

    async def close(self) -> None:
        """Simulate closing the community client."""
        self.closed = True

    async def get_reviews(self, metadata_id: str) -> str:
        """Simulate fetching reviews for a metadata ID."""
        self.calls.append(metadata_id)
        return f"review-{metadata_id}"


@pytest.fixture()
def library_setup(monkeypatch: pytest.MonkeyPatch):
    """Set up a PlexLibraryProvider with stubbed dependencies."""
    show = StubShow("show-key")
    season = StubSeason("season-key", show)
    episode = StubEpisode("episode-key", season, show)
    season._episodes = [episode]
    show._seasons = [season]
    show._episodes = [episode]

    movie = StubMovie("movie-key")
    items = cast(list[StubBaseVideo], [movie, show])
    sections = [FakeRawSection("Movies", "movie")]
    fake_client = FakePlexClient(sections=sections, items=items)

    monkeypatch.setattr(library_module.plexapi_video, "Movie", StubMovie)
    monkeypatch.setattr(library_module.plexapi_video, "Show", StubShow)
    monkeypatch.setattr(library_module.plexapi_video, "Season", StubSeason)
    monkeypatch.setattr(library_module.plexapi_video, "Episode", StubEpisode)
    StubCommunityClient.instances.clear()
    monkeypatch.setattr(library_module, "PlexCommunityClient", StubCommunityClient)
    monkeypatch.setattr(library_module, "PlexClient", lambda **_: fake_client)

    provider = library_module.PlexLibraryProvider(
        logger=cast(ProviderLogger, getLogger("test.library.provider")),
        config={"url": "https://plex.example", "token": "token"},
    )
    return provider, fake_client, movie, show, episode


@pytest_asyncio.fixture()
async def initialized_provider(
    library_setup: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Provide an initialized PlexLibraryProvider for tests."""
    provider, fake_client, movie, show, episode = library_setup
    await provider.initialize()
    yield provider, fake_client, movie, show, episode
    await provider.close()


@pytest.mark.asyncio
async def test_initialize_sets_sections_and_user(
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Initialization sets sections and user correctly."""
    provider, fake_client, *_ = initialized_provider
    assert fake_client.initialized is True
    assert fake_client.cleared is True
    sections = await provider.get_sections()
    assert len(sections) == 1 and sections[0].title == "Movies"
    user = provider.user()
    assert user is not None and user.title == "Demo"


@pytest.mark.asyncio
async def test_list_items_wraps_media_and_exposes_metadata(
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Listing items wraps them and exposes metadata correctly."""
    provider, _fake_client, _movie, _show, _ = initialized_provider
    section = (await provider.get_sections())[0]
    media_items = await provider.list_items(section)
    assert len(media_items) == 2
    movie_item = media_items[0]
    assert movie_item.on_watching is True
    assert movie_item.on_watchlist is True
    history = await movie_item.history()
    assert history and any(entry.library_key == "derived" for entry in history)


@pytest.mark.asyncio
async def test_get_review_checks_admin(
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Get review returns None when rating prerequisites are missing."""
    provider, _fake_client, movie, *_ = initialized_provider
    movie.userRating = None
    movie.lastRatedAt = None
    result = await provider.get_review(cast(library_module.plexapi_video.Video, movie))
    assert result is None


@pytest.mark.asyncio
async def test_get_history_includes_child_entries(
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Get history includes child entries."""
    provider, _fake_client, _movie, show, _episode = initialized_provider
    result = await provider.get_history(cast(library_module.plexapi_video.Video, show))
    assert any(entry.library_key == "derived" for entry in result)


@pytest.mark.asyncio
async def test_parse_webhook_filters_user(
    monkeypatch: pytest.MonkeyPatch,
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Parsing webhooks filters by user and extracts keys."""
    provider, _, *_ = initialized_provider

    class StubWebhook:
        def __init__(
            self, *, account_id: int | None, rating_key: str | None, event: str
        ):
            self.account_id = account_id
            self.top_level_rating_key = rating_key
            self.event = event
            self.event_type = library_module.PlexWebhookEventType.SCROBBLE

    async def fake_from_request(_request):
        return StubWebhook(account_id=1, rating_key="key", event="media.scrobble")

    monkeypatch.setattr(library_module.WebhookParser, "from_request", fake_from_request)
    should_sync, keys = await provider.parse_webhook(cast(Request, SimpleNamespace()))
    assert should_sync is True and keys == ("key",)

    async def missing_account(_request):
        return StubWebhook(account_id=None, rating_key=None, event="media.scrobble")

    monkeypatch.setattr(library_module.WebhookParser, "from_request", missing_account)
    with pytest.raises(ValueError):
        await provider.parse_webhook(cast(Request, SimpleNamespace()))


@pytest.mark.asyncio
async def test_parse_webhook_uses_normalized_event_type(
    monkeypatch: pytest.MonkeyPatch,
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Parsing webhooks uses event_type instead of raw event string."""
    provider, _, *_ = initialized_provider

    class StubWebhook:
        def __init__(self):
            self.account_id = 1
            self.top_level_rating_key = "key"
            self.event = "created"
            self.event_type = library_module.PlexWebhookEventType.MEDIA_ADDED

    async def fake_from_request(_request):
        return StubWebhook()

    monkeypatch.setattr(library_module.WebhookParser, "from_request", fake_from_request)
    should_sync, keys = await provider.parse_webhook(cast(Request, SimpleNamespace()))
    assert should_sync is True and keys == ("key",)


@pytest.mark.asyncio
async def test_list_items_rejects_non_plex_section(
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """list_items should reject section objects from other providers."""
    provider, *_ = initialized_provider
    with pytest.raises(TypeError):
        await provider.list_items(cast(library_module.LibrarySection, object()))


@pytest.mark.asyncio
async def test_parse_webhook_ignores_mismatched_account(
    monkeypatch: pytest.MonkeyPatch,
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Webhook events for other users should not trigger sync."""
    provider, *_ = initialized_provider

    class StubWebhook:
        account_id = 999
        top_level_rating_key = "other"
        event = "media.scrobble"
        event_type = library_module.PlexWebhookEventType.SCROBBLE

    async def fake_from_request(_request):
        return StubWebhook()

    monkeypatch.setattr(library_module.WebhookParser, "from_request", fake_from_request)
    should_sync, keys = await provider.parse_webhook(cast(Request, SimpleNamespace()))
    assert should_sync is False
    assert keys == tuple()


@pytest.mark.asyncio
async def test_get_review_handles_exception(
    monkeypatch: pytest.MonkeyPatch,
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Review retrieval failures should return None rather than raising."""
    provider, _, movie, *_ = initialized_provider
    movie.userRating = 8
    movie.lastRatedAt = datetime.now(tz=UTC)
    movie.guid = "plex://movie/123"

    class FailingCommunityClient:
        async def close(self) -> None:
            return None

        async def get_reviews(self, _metadata_id: str):
            raise RuntimeError("boom")

    provider._community_client = cast(Any, FailingCommunityClient())
    assert (
        await provider.get_review(cast(library_module.plexapi_video.Video, movie))
        is None
    )


@pytest.mark.asyncio
async def test_media_helpers_cover_external_and_poster_paths(
    monkeypatch: pytest.MonkeyPatch,
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Media helpers should handle missing/valid guid and poster fetch errors."""
    provider, _, movie, *_ = initialized_provider
    section = (await provider.get_sections())[0]

    media = library_module.PlexLibraryMedia(
        provider,
        cast(library_module.PlexLibrarySection, section),
        cast(library_module.plexapi_video.Video, movie),
        library_module.MediaKind.MOVIE,
    )
    assert media.external_url is not None

    movie.guid = None
    assert media.external_url is None

    movie.thumb = None
    assert media.poster_image is None

    movie.thumb = "/thumb"
    monkeypatch.setattr(
        library_module,
        "fetch_image_as_data_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    assert media.poster_image is None


@pytest.mark.asyncio
async def test_mapping_descriptors_and_rating_helpers(
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Entry helper properties should handle malformed GUIDs and bad ratings."""
    provider, _, movie, *_ = initialized_provider
    section = (await provider.get_sections())[0]

    movie.guid = "broken-guid"
    movie.guids = [SimpleNamespace(id=""), SimpleNamespace(id="imdb://tt777?lang=en")]
    movie.userRating = "bad"
    movie.viewCount = None

    entry = library_module.PlexLibraryMovie(
        provider,
        cast(library_module.PlexLibrarySection, section),
        cast(library_module.plexapi_video.Movie, movie),
    )
    descriptors = entry.mapping_descriptors()
    assert descriptors == [("imdb_movie", "tt777", None)]
    assert entry.user_rating is None
    assert entry.view_count == 0


@pytest.mark.asyncio
async def test_show_season_episode_mapping_variants(
    initialized_provider: tuple[
        library_module.PlexLibraryProvider,
        FakePlexClient,
        StubMovie,
        StubShow,
        StubEpisode,
    ],
):
    """Show/season/episode wrappers should preserve scoped mapping descriptors."""
    provider, fake_client, _, show, episode = initialized_provider
    section = (await provider.get_sections())[0]

    # Include both tmdb and tvdb descriptors so strict/sort branches are exercised.
    show.guid = "plex://show/88"
    show.guids = [
        SimpleNamespace(id="tmdb://99"),
        SimpleNamespace(id="com.plexapp.agents.thetvdb://42"),
    ]

    fake_client.get_ordering = cast(Any, lambda _show: "tvdb")

    wrapped_show = library_module.PlexLibraryShow(
        provider,
        cast(library_module.PlexLibrarySection, section),
        cast(library_module.plexapi_video.Show, show),
    )
    provider.parsed_config.strict = True
    strict_descriptors = wrapped_show.mapping_descriptors()
    assert all(d[0] in ("tvdb_show", "tvdb_movie") for d in strict_descriptors)

    provider.parsed_config.strict = False
    sorted_descriptors = wrapped_show.mapping_descriptors()
    assert sorted_descriptors[0][0] in ("tvdb_show", "tvdb_movie")

    seasons = wrapped_show.seasons()
    assert seasons
    season_descriptors = seasons[0].mapping_descriptors()
    assert all(d[2] == "s1" for d in season_descriptors)

    wrapped_episode = library_module.PlexLibraryEpisode(
        provider,
        cast(library_module.PlexLibrarySection, section),
        cast(library_module.plexapi_video.Episode, episode),
    )
    assert wrapped_episode.mapping_descriptors() == season_descriptors


def test_wrap_entry_unsupported_type_raises(library_setup):
    """_wrap_entry should reject unsupported media objects."""
    provider, *_ = library_setup
    section = library_module.PlexLibrarySection(
        provider,
        cast(Any, FakeRawSection("Movies", "movie")),
    )

    with pytest.raises(TypeError):
        provider._wrap_entry(
            section, cast(library_module.plexapi_video.Video, object())
        )
