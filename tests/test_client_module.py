"""Focused tests for the Plex client helpers."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

import anibridge_plex_provider.client as client_module


def _server_stub(**kwargs: Any) -> client_module.PlexServer:
    return cast(client_module.PlexServer, SimpleNamespace(**kwargs))


def _account_stub(**kwargs: Any) -> client_module.MyPlexAccount:
    return cast(client_module.MyPlexAccount, SimpleNamespace(**kwargs))


@pytest.fixture()
def plex_client() -> client_module.PlexClient:
    """Provide a PlexClient instance for tests."""
    return client_module.PlexClient(
        url="https://plex.example",
        token="token",
        user="demo",
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
        users=lambda: [],
    )

    class StubPlexServer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.settings = StubSettings()
            self.library = StubLibrary()

        def myPlexAccount(self):
            return account

    monkeypatch.setattr(client_module, "PlexServer", StubPlexServer)
    monkeypatch.setattr(client_module, "MovieSection", StubMovieSection)
    monkeypatch.setattr(client_module, "ShowSection", StubMovieSection)
    monkeypatch.setattr(client_module, "SelectiveVerifySession", lambda **_: None)

    plex_client._continue_cache["stale"] = client_module._FrozenCacheEntry(
        keys=frozenset({"old"}),
        expires_at=0,
    )
    plex_client._ordering_cache[1] = "tmdb"

    await plex_client.initialize()

    assert plex_client.user_id == 1
    assert plex_client.display_name == "demo"
    assert plex_client.is_admin is True
    assert plex_client.sections()
    assert plex_client.on_deck_window == timedelta(weeks=2)
    assert not plex_client._continue_cache
    assert not plex_client._ordering_cache


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


def test_is_on_continue_watching_caches_results(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that is_on_continue_watching caches results correctly."""

    class DummySection:
        key = "sec"

        def __init__(self) -> None:
            self.invocations = 0

        def continueWatching(self):
            self.invocations += 1
            return [SimpleNamespace(ratingKey="5")]

    monkeypatch.setattr(client_module, "monotonic", lambda: 10.0)
    plex_client._user_client = object()  # type: ignore
    section = cast(client_module.LibrarySection, DummySection())
    video = cast(client_module.Video, SimpleNamespace(ratingKey="5"))

    assert plex_client.is_on_continue_watching(section, video)
    assert plex_client.is_on_continue_watching(section, video)
    assert section.invocations == 1


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

    plex_client._admin_client = _server_stub(history=fake_history)
    plex_client._is_admin = False
    plex_client._user_id = 99

    video = cast(client_module.Video, SimpleNamespace(ratingKey=5, librarySectionID=9))
    history = await plex_client.fetch_history(video)
    assert history == [("7", records[0].viewedAt)]
    assert observed["accountID"] == 99


def test_is_on_watchlist_only_admin(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that is_on_watchlist only works for admin users."""
    plex_client._is_admin = False
    plex_client._account = _account_stub(id=1, watchlist=lambda: [])
    assert not plex_client.is_on_watchlist(
        cast(client_module.Video, SimpleNamespace(guid="guid"))
    )

    calls = {"count": 0}

    def fake_watchlist():
        calls["count"] += 1
        return [SimpleNamespace(guid="guid"), SimpleNamespace(guid=None)]

    plex_client._is_admin = True
    plex_client._account = _account_stub(id=1, watchlist=fake_watchlist)
    monkeypatch.setattr(client_module, "monotonic", lambda: 50.0)

    assert plex_client.is_on_watchlist(
        cast(client_module.Video, SimpleNamespace(guid="guid"))
    )
    assert plex_client.is_on_watchlist(
        cast(client_module.Video, SimpleNamespace(guid="guid"))
    )
    assert calls["count"] == 1


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
        "a": client_module._FrozenCacheEntry(keys=frozenset({"1"}), expires_at=0)
    }
    plex_client._ordering_cache = {1: "tmdb"}
    plex_client.clear_cache()
    assert not plex_client._continue_cache and not plex_client._ordering_cache
