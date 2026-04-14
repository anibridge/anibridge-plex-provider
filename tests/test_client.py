"""Focused tests for the Plex client helpers."""

from datetime import UTC, datetime, timedelta
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anibridge.utils.types import ProviderLogger

import anibridge.providers.library.plex.client as client_module


def _server_stub(**kwargs: Any) -> client_module.PlexServer:
    return cast(client_module.PlexServer, SimpleNamespace(**kwargs))


def _account_stub(**kwargs: Any) -> client_module.MyPlexAccount:
    kwargs.setdefault("restricted", False)
    return cast(client_module.MyPlexAccount, SimpleNamespace(**kwargs))


def _session_stub(machine_id: str = "machine-1") -> Any:
    class StubResponse:
        def __init__(self) -> None:
            self.content = (
                f'<MediaContainer machineIdentifier="{machine_id}"/>'.encode()
            )

    class StubSession:
        def get(self, _url: str, timeout: int = 10):
            return StubResponse()

    return StubSession()


@pytest.fixture()
def plex_client() -> client_module.PlexClient:
    """Provide a PlexClient instance for tests."""
    return client_module.PlexClient(
        logger=cast(ProviderLogger, getLogger("test.client")),
        url="https://plex.example",
        token="token",
    )


@pytest.mark.asyncio
async def test_initialize_populates_state_and_sections(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that the Plex client initializes with the correct state and sections."""

    class StubSettings:
        def get(self, _):
            return SimpleNamespace(value="2")

    class StubMovieSection:
        def __init__(self, title: str) -> None:
            self.title = title
            self.type = "movie"
            self.key = "m"

    class StubLibrary:
        def sections(self):
            return [StubMovieSection("Movies")]

    account = _account_stub(
        id=1,
        username="demo",
        email="demo@example",
        title="Demo",
        friendlyName="demo",
        users=lambda: [],
        user=lambda _name: SimpleNamespace(id=1),
        resource=lambda _machine_id: SimpleNamespace(accessToken="token"),
    )

    class StubPlexServer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.settings = StubSettings()
            self.library = StubLibrary()
            self.token = _kwargs.get("token") if "token" in _kwargs else _args[1]

    monkeypatch.setattr(client_module, "PlexServer", StubPlexServer)
    monkeypatch.setattr(client_module, "MovieSection", StubMovieSection)
    monkeypatch.setattr(client_module, "ShowSection", StubMovieSection)
    session = _session_stub("machine-1")
    monkeypatch.setattr(client_module.requests, "Session", lambda: session)
    monkeypatch.setattr(client_module, "SelectiveVerifySession", lambda **_: session)
    monkeypatch.setattr(client_module, "MyPlexAccount", lambda **_: account)

    plex_client._continue_cache["stale"] = client_module._FrozenCacheEntry(
        keys=frozenset({"old"}),
        cached_at=datetime.now(tz=UTC),
    )
    plex_client._ordering_cache[1] = "tmdb"

    await plex_client.initialize()

    assert plex_client.user_id == 1
    assert plex_client.display_name == "demo"
    assert plex_client.sections()
    assert not plex_client._continue_cache
    assert not plex_client._ordering_cache


def test_initialize_switches_home_user_when_requested(monkeypatch: pytest.MonkeyPatch):
    """A requested home user should be resolved via MyPlexAccount switching."""

    class StubSettings:
        def get(self, _):
            return SimpleNamespace(value="2")

    class StubLibrary:
        def sections(self):
            return []

    home_user = cast(
        Any,
        SimpleNamespace(
            id=99,
            username="child",
            email="child@example",
            title="Child",
        ),
    )

    switched_account = _account_stub(
        id=2,
        authToken="switched-token",
        username="child",
        email="child@example",
        title="Child",
        friendlyName="child",
        users=lambda: [],
        resource=lambda _machine_id: SimpleNamespace(accessToken="switched-token"),
    )

    def _user_lookup(name):
        if name == "child":
            return home_user
        raise Exception(f"Unknown user {name}")

    account = _account_stub(
        id=1,
        authToken="admin-token",
        username="admin",
        email="admin@example",
        title="Admin",
        users=lambda: [home_user],
        user=_user_lookup,
        switchHomeUser=lambda _user, pin=None: switched_account,
    )

    class StubPlexServer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.settings = StubSettings()
            self.library = StubLibrary()
            self.token = _kwargs.get("token") if "token" in _kwargs else _args[1]

        def myPlexAccount(self):
            if self.token == "switched-token":
                return switched_account
            return account

    monkeypatch.setattr(client_module, "PlexServer", StubPlexServer)
    session = _session_stub("machine-1")
    monkeypatch.setattr(client_module.requests, "Session", lambda: session)
    monkeypatch.setattr(client_module, "SelectiveVerifySession", lambda **_: session)
    monkeypatch.setattr(client_module, "MyPlexAccount", lambda **_: account)

    client = client_module.PlexClient(
        logger=cast(ProviderLogger, getLogger("test.client")),
        url="https://plex.example",
        token="token",
        home_user="child",
    )

    user_client, _, account_id, user, user_id, display_name = (
        client._initialize_clients()
    )
    assert user_client is not None
    assert account_id == 2
    assert user is home_user
    assert user_id == 99
    assert display_name == "child"


@pytest.mark.asyncio
async def test_list_section_items_applies_filters(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that the list_section_items method applies filters correctly."""

    class DummyVideo:
        def __init__(self, rating_key: str) -> None:
            self.ratingKey = rating_key

    class DummyMovie(DummyVideo):
        pass

    class DummyShow(DummyVideo):
        pass

    monkeypatch.setattr(client_module, "Movie", DummyMovie)
    monkeypatch.setattr(client_module, "Show", DummyShow)

    class DummySection:
        key = "sec"
        type = "movie"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def search(self, **kwargs: Any):
            self.calls.append(kwargs)
            return [DummyMovie("1"), DummyShow("2"), object()]

    plex_client._genre_filter = ("Drama",)
    section = DummySection()

    result = await plex_client.list_section_items(
        cast(client_module.LibrarySection, section),
        min_last_modified=datetime.now(UTC),
        require_watched=True,
        keys=("1",),
    )

    assert len(result) == 1 and isinstance(result[0], DummyMovie)
    assert section.calls
    filters = section.calls[0]["filters"]["and"]
    assert any("lastViewedAt" in str(entry) for entry in filters)
    assert any("viewCount" in str(entry) for entry in filters)
    assert filters[-1] == {"genre": ("Drama",)}


def test_is_on_continue_watching_caches_results(plex_client: client_module.PlexClient):
    """Test that is_on_continue_watching caches results correctly."""

    class DummySection:
        key = "sec"

        def __init__(self) -> None:
            self.invocations = 0

        def continueWatching(self):
            self.invocations += 1
            return [SimpleNamespace(ratingKey="5")]

    plex_client._user_client = object()  # type: ignore
    section = cast(client_module.LibrarySection, DummySection())
    video = cast(
        client_module.Video,
        SimpleNamespace(
            ratingKey="5",
            updatedAt=None,
            lastViewedAt=None,
            lastRatedAt=None,
            addedAt=None,
        ),
    )

    assert plex_client.is_on_continue_watching(section, video)
    assert plex_client.is_on_continue_watching(section, video)
    assert section.invocations == 1


def test_is_on_continue_watching_matches_show_and_season_keys(
    plex_client: client_module.PlexClient,
):
    """Continue Watching entries for episodes should also match season/show items."""

    class DummySection:
        key = "sec"

        def continueWatching(self):
            return [
                SimpleNamespace(
                    ratingKey="episode-key",
                    parentRatingKey="season-key",
                    grandparentRatingKey="show-key",
                )
            ]

    plex_client._user_client = object()  # type: ignore
    section = cast(client_module.LibrarySection, DummySection())
    show = cast(
        client_module.Video,
        SimpleNamespace(
            ratingKey="show-key",
            updatedAt=None,
            lastViewedAt=None,
            lastRatedAt=None,
            addedAt=None,
        ),
    )
    season = cast(
        client_module.Video,
        SimpleNamespace(
            ratingKey="season-key",
            updatedAt=None,
            lastViewedAt=None,
            lastRatedAt=None,
            addedAt=None,
        ),
    )
    episode = cast(
        client_module.Video,
        SimpleNamespace(
            ratingKey="episode-key",
            updatedAt=None,
            lastViewedAt=None,
            lastRatedAt=None,
            addedAt=None,
        ),
    )

    assert plex_client.is_on_continue_watching(section, show)
    assert plex_client.is_on_continue_watching(section, season)
    assert plex_client.is_on_continue_watching(section, episode)


@pytest.mark.asyncio
async def test_fetch_history_respects_bundle(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that the fetch_history method respects the client user id."""
    records = [SimpleNamespace(ratingKey=7, viewedAt=datetime.now(tz=UTC))]
    observed: dict[str, Any] = {}

    def fake_history(**kwargs: Any):
        observed.update(kwargs)
        return records

    plex_client._user_client = _server_stub(history=fake_history)
    plex_client._user_id = 99

    video = cast(client_module.Video, SimpleNamespace(ratingKey=5, librarySectionID=9))
    history = await plex_client.fetch_history(video)
    assert history == [("7", records[0].viewedAt)]
    assert observed["accountID"] == 99


def test_is_on_watchlist_caches_results(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Watchlist lookups should cache account watchlist GUIDs."""

    calls = {"count": 0}

    def fake_watchlist():
        calls["count"] += 1
        return [SimpleNamespace(guid="guid"), SimpleNamespace(guid=None)]

    plex_client._account = _account_stub(id=1, watchlist=fake_watchlist)

    assert plex_client.is_on_watchlist(
        cast(client_module.Video, SimpleNamespace(guid="guid"))
    )
    assert plex_client.is_on_watchlist(
        cast(client_module.Video, SimpleNamespace(guid="guid"))
    )
    assert calls["count"] == 1


def test_is_on_watchlist_first_fetch_failure_raises(
    plex_client: client_module.PlexClient,
):
    """If no cache exists yet, watchlist fetch failures should be raised."""

    def failing_watchlist():
        raise RuntimeError("watchlist unavailable")

    plex_client._account = _account_stub(id=1, watchlist=failing_watchlist)

    with pytest.raises(RuntimeError, match="watchlist unavailable"):
        plex_client.is_on_watchlist(
            cast(client_module.Video, SimpleNamespace(guid="guid"))
        )

    assert plex_client._watchlist_cache is None


def test_is_on_watchlist_refresh_failure_uses_stale_cache(
    plex_client: client_module.PlexClient,
):
    """If refresh fails after a successful fetch, stale watchlist should be reused."""
    calls = {"count": 0}

    def failing_watchlist():
        calls["count"] += 1
        raise RuntimeError("transient watchlist failure")

    previous_cached_at = datetime.now(tz=UTC) - timedelta(minutes=10)
    plex_client._watchlist_cache = client_module._FrozenCacheEntry(
        keys=frozenset({"guid"}),
        cached_at=previous_cached_at,
    )
    plex_client._account = _account_stub(id=1, watchlist=failing_watchlist)

    assert plex_client.is_on_watchlist(
        cast(client_module.Video, SimpleNamespace(guid="guid"))
    )
    assert calls["count"] == 1
    assert plex_client._watchlist_cache is not None
    assert plex_client._watchlist_cache.cached_at > previous_cached_at


def test_get_ordering_and_filters(plex_client: client_module.PlexClient):
    """Test that the get_ordering method extracts the correct ordering from shows."""
    show = cast(client_module.Show, SimpleNamespace(showOrdering="tmdbAiring"))
    assert plex_client.get_ordering(show) == "tmdb"

    settings = [SimpleNamespace(id="showOrdering", value="tvdbAiring")]
    section = SimpleNamespace(settings=lambda: settings)
    show = cast(
        client_module.Show,
        SimpleNamespace(showOrdering="", section=lambda: section, librarySectionID=5),
    )
    assert plex_client.get_ordering(show) == "tvdb"

    plex_client._continue_cache = {
        "a": client_module._FrozenCacheEntry(
            keys=frozenset({"1"}), cached_at=datetime.now(tz=UTC)
        )
    }
    plex_client._ordering_cache = {1: "tmdb"}
    plex_client.clear_cache()
    assert not plex_client._continue_cache and not plex_client._ordering_cache


def test_normalize_thumb_rewrites_tmdb_sizes():
    """TMDB poster URLs should be normalized to the requested size."""
    original = "https://image.tmdb.org/t/p/original/abc.jpg"
    resized = "https://image.tmdb.org/t/p/w500/abc.jpg"
    non_tmdb = "https://cdn.example.com/poster.jpg"

    assert client_module.PlexClient._normalize_thumb(original) == (
        "https://image.tmdb.org/t/p/w92/abc.jpg"
    )
    assert client_module.PlexClient._normalize_thumb(resized, size="w185") == (
        "https://image.tmdb.org/t/p/w185/abc.jpg"
    )
    assert client_module.PlexClient._normalize_thumb(non_tmdb) == non_tmdb


def test_get_thumb_url_uses_online_metadata_and_normalizes(
    plex_client: client_module.PlexClient,
):
    """Online Plex metadata should be preferred when available."""
    calls: list[str] = []

    def fetch_item(key: str):
        calls.append(key)
        return SimpleNamespace(thumb="https://image.tmdb.org/t/p/w500/poster.jpg")

    plex_client._account = _account_stub(
        DISCOVER="https://discover.provider.plex.tv",
        fetchItem=fetch_item,
    )
    item = cast(
        client_module.Video,
        SimpleNamespace(guid="plex://show/12345", thumb="/thumb/path"),
    )

    result = plex_client.get_thumb_url(item)

    assert calls == ["https://discover.provider.plex.tv/library/metadata/12345"]
    assert result == "https://image.tmdb.org/t/p/w92/poster.jpg"


def test_get_thumb_url_falls_back_to_transcoded_data_url(
    monkeypatch: pytest.MonkeyPatch,
    plex_client: client_module.PlexClient,
):
    """When online metadata is unavailable, local transcoded artwork is used."""
    transcode_calls: list[tuple[str, int, int]] = []

    def transcode_image(path: str, *, height: int, width: int) -> str:
        transcode_calls.append((path, height, width))
        return "https://plex.example/transcode?img=1"

    fetch_calls: list[tuple[str, int]] = []

    def fake_fetch_image_as_data_url(url: str, timeout: int = 3) -> str:
        fetch_calls.append((url, timeout))
        return "data:image/jpeg;base64,abc123"

    monkeypatch.setattr(
        client_module,
        "fetch_image_as_data_url",
        fake_fetch_image_as_data_url,
    )

    plex_client._account = _account_stub(
        DISCOVER="https://discover.provider.plex.tv",
        fetchItem=lambda _key: SimpleNamespace(thumb=None),
    )
    plex_client._user_client = _server_stub(transcodeImage=transcode_image)
    item = cast(client_module.Video, SimpleNamespace(guid="plex://movie/9", thumb="/t"))

    result = plex_client.get_thumb_url(item)

    assert result == "data:image/jpeg;base64,abc123"
    assert transcode_calls == [("/t", 138, 92)]
    assert fetch_calls == [("https://plex.example/transcode?img=1", 3)]


def test_get_thumb_url_is_stable_across_repeated_calls(
    plex_client: client_module.PlexClient,
):
    """Repeated requests for the same item should return the same value."""

    def fetch_item(_key: str):
        return SimpleNamespace(thumb="https://image.tmdb.org/t/p/original/poster.jpg")

    plex_client._account = _account_stub(
        DISCOVER="https://discover.provider.plex.tv",
        fetchItem=fetch_item,
    )
    item = cast(client_module.Video, SimpleNamespace(guid="plex://movie/7", thumb="/t"))

    first = plex_client.get_thumb_url(item)
    second = plex_client.get_thumb_url(item)

    assert first == second == "https://image.tmdb.org/t/p/w92/poster.jpg"
