"""Tests for the Plex library provider integration."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import pytest_asyncio
from starlette.requests import Request

import anibridge_plex_provider.client as client_module
import anibridge_plex_provider.library as library_module


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
        self._bundle = client_module.PlexClientBundle(
            admin_client=_server_stub(),
            user_client=_server_stub(
                url=lambda path: f"https://plex{path}",
            ),
            account=_account_stub(id=1, watchlist=lambda: []),
            target_user=None,
            user_id=1,
            display_name="Demo",
            is_admin=True,
        )
        cfg = client_module.PlexClientConfig(
            url="https://plex.example",
            token="t",
            user="u",
        )
        self._helper = client_module.PlexClient(config=cfg)
        self.initialized = False
        self.closed = False
        self.cleared = False

    async def initialize(self) -> None:
        """Simulate client initialization."""
        self.initialized = True

    async def close(self) -> None:
        """Simulate client closure."""
        self.closed = True

    def bundle(self) -> client_module.PlexClientBundle:
        """Return the client bundle."""
        return self._bundle

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


class StubCommunityClient:
    """Stub for the PlexCommunityClient."""

    instances: ClassVar[list[StubCommunityClient]] = []

    def __init__(self, token: str) -> None:
        """Initialize the stub community client."""
        self.token = token
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
    items = [movie, show]
    sections = [FakeRawSection("Movies", "movie")]
    fake_client = FakePlexClient(sections=sections, items=items)

    monkeypatch.setattr(
        library_module.PlexLibraryProvider,
        "_create_client",
        lambda self: fake_client,
    )
    monkeypatch.setattr(library_module.plexapi_video, "Movie", StubMovie)
    monkeypatch.setattr(library_module.plexapi_video, "Show", StubShow)
    monkeypatch.setattr(library_module.plexapi_video, "Season", StubSeason)
    monkeypatch.setattr(library_module.plexapi_video, "Episode", StubEpisode)
    StubCommunityClient.instances.clear()
    monkeypatch.setattr(library_module, "PlexCommunityClient", StubCommunityClient)

    provider = library_module.PlexLibraryProvider(
        config={"url": "https://plex.example", "token": "token", "user": "demo"}
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
    assert movie_item.media().ids()["imdb"] == "tt12345"
    history = await movie_item.history()
    assert history and any(entry.library_key == "derived" for entry in history)
    review = await movie_item.review()
    assert review == "review-1"


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
    """Get review returns None for non-admin users."""
    provider, _fake_client, movie, *_ = initialized_provider
    provider._is_admin_user = False
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

    monkeypatch.setattr(library_module.PlexWebhook, "from_request", fake_from_request)
    should_sync, keys = await provider.parse_webhook(cast(Request, SimpleNamespace()))
    assert should_sync is True and keys == ("key",)

    async def missing_account(_request):
        return StubWebhook(account_id=None, rating_key=None, event="media.scrobble")

    monkeypatch.setattr(library_module.PlexWebhook, "from_request", missing_account)
    with pytest.raises(ValueError):
        await provider.parse_webhook(cast(Request, SimpleNamespace()))
