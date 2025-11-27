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
def plex_client_config() -> client_module.PlexClientConfig:
    """Provide a basic PlexClientConfig for tests."""
    return client_module.PlexClientConfig(
        url="https://plex.example", token="token", user="demo"
    )


@pytest.fixture()
def plex_client(
    plex_client_config: client_module.PlexClientConfig,
) -> client_module.PlexClient:
    """Provide a PlexClient instance for tests."""
    return client_module.PlexClient(config=plex_client_config)


@pytest.mark.asyncio
async def test_initialize_populates_bundle_and_sections(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that the Plex client initializes with the correct bundle and sections."""
    bundle = client_module.PlexClientBundle(
        admin_client=_server_stub(),
        user_client=_server_stub(),
        account=_account_stub(id=1, watchlist=lambda: []),
        target_user=None,
        user_id=1,
        display_name="Demo",
        is_admin=True,
    )
    sections = ["Movies", "Shows"]
    plex_client._continue_cache["stale"] = client_module._FrozenCacheEntry(
        keys=frozenset({"old"}),
        expires_at=0,
    )
    plex_client._ordering_cache[1] = "tmdb"

    monkeypatch.setattr(
        client_module.PlexClient, "_create_client_bundle", lambda self: bundle
    )
    monkeypatch.setattr(
        client_module.PlexClient, "_load_sections", lambda self: sections
    )
    monkeypatch.setattr(
        client_module.PlexClient,
        "_get_on_deck_window",
        lambda self: timedelta(days=7),
    )

    await plex_client.initialize()

    assert plex_client.bundle() is bundle
    assert plex_client.sections() == tuple(sections)
    assert plex_client.on_deck_window == timedelta(days=7)
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

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def search(self, **kwargs: Any):
            self.calls.append(kwargs)
            return [DummyMovie("1"), DummyShow("2"), object()]

    monkeypatch.setattr(
        client_module.PlexClient, "_build_modified_filter", lambda self, s, _: {"m": 1}
    )
    monkeypatch.setattr(
        client_module.PlexClient, "_build_watched_filter", lambda self, s: {"w": 1}
    )

    plex_client._genre_filter = ("Drama",)
    section = DummySection()

    result = await plex_client.list_section_items(
        cast(client_module.LibrarySection, section),
        min_last_modified=datetime.now(UTC),
        require_watched=True,
        keys=("1",),
    )

    assert len(result) == 1 and isinstance(result[0], DummyMovie)
    assert section.calls and section.calls[0]["filters"]["and"][-1] == {
        "genre": ("Drama",)
    }


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
    """Test that the fetch_history method respects the client bundle."""
    records = [SimpleNamespace(ratingKey=7, viewedAt=datetime.now(tz=UTC))]
    observed: dict[str, Any] = {}

    def fake_history(**kwargs: Any):
        observed.update(kwargs)
        return records

    plex_client._admin_client = _server_stub(history=fake_history)
    plex_client._bundle = client_module.PlexClientBundle(
        admin_client=_server_stub(),
        user_client=_server_stub(),
        account=_account_stub(id=1, watchlist=lambda: []),
        target_user=None,
        user_id=99,
        display_name="Demo",
        is_admin=False,
    )

    video = cast(client_module.Video, SimpleNamespace(ratingKey=5, librarySectionID=9))
    history = await plex_client.fetch_history(video)
    assert history == [("7", records[0].viewedAt)]
    assert observed["accountID"] == 99


def test_is_on_watchlist_only_admin(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that is_on_watchlist only works for admin users."""
    plex_client._bundle = client_module.PlexClientBundle(
        admin_client=_server_stub(),
        user_client=_server_stub(),
        account=_account_stub(id=1, watchlist=lambda: []),
        target_user=None,
        user_id=1,
        display_name="Demo",
        is_admin=False,
    )
    assert not plex_client.is_on_watchlist(
        cast(client_module.Video, SimpleNamespace(guid="guid"))
    )

    calls = {"count": 0}

    def fake_watchlist():
        calls["count"] += 1
        return [SimpleNamespace(guid="guid"), SimpleNamespace(guid=None)]

    plex_client._bundle = client_module.PlexClientBundle(
        admin_client=_server_stub(),
        user_client=_server_stub(),
        account=_account_stub(id=1, watchlist=fake_watchlist),
        target_user=None,
        user_id=1,
        display_name="Demo",
        is_admin=True,
    )
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

    movie_section = cast(client_module.MovieSection, SimpleNamespace(type="movie"))
    show_section = cast(client_module.ShowSection, SimpleNamespace(type="show"))
    reference = datetime(2024, 1, 1)

    movie_filter = plex_client._build_modified_filter(movie_section, reference)
    show_filter = plex_client._build_modified_filter(show_section, reference)
    assert "lastViewedAt" in str(movie_filter)
    assert "show.lastViewedAt" in str(show_filter)

    watched_movie = plex_client._build_watched_filter(movie_section)
    watched_show = plex_client._build_watched_filter(show_section)
    assert "viewCount" in str(watched_movie)
    assert "show.viewCount" in str(watched_show)


def test_build_session_and_load_sections(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test session building and section loading helpers."""
    created_session = {}

    class DummySession:
        def __init__(self, whitelist: list[str]) -> None:
            created_session["whitelist"] = whitelist

    monkeypatch.setattr(client_module, "SelectiveVerifySession", DummySession)
    assert client_module._build_session("https://plex.example") is not None
    assert created_session["whitelist"] == ["plex.example"]
    assert client_module._build_session("http://plex.example") is None

    class StubMovieSection:
        def __init__(self, title: str) -> None:
            self.title = title
            self.type = "movie"
            self.key = title

    class StubShowSection(StubMovieSection):
        pass

    class OtherSection:
        title = "Other"

    monkeypatch.setattr(client_module, "MovieSection", StubMovieSection)
    monkeypatch.setattr(client_module, "ShowSection", StubShowSection)

    plex_client._section_filter = {"allowed"}
    plex_client._user_client = cast(
        client_module.PlexServer,
        SimpleNamespace(
            library=SimpleNamespace(
                sections=lambda: [
                    StubMovieSection("Allowed"),
                    StubMovieSection("Other"),
                    OtherSection(),
                ]
            )
        ),
    )

    sections = plex_client._load_sections()
    assert len(sections) == 1 and sections[0].title == "Allowed"


def test_default_bundle_switches_user(
    monkeypatch: pytest.MonkeyPatch, plex_client_config: client_module.PlexClientConfig
):
    """Test that the default bundle switches to the requested user."""
    plex_client_config.user = "friend"
    account = cast(
        client_module.MyPlexAccount,
        SimpleNamespace(
            username="demo",
            email="demo@example",
            title="Demo",
            id=1,
            users=lambda: [SimpleNamespace(username="friend", id=2)],
        ),
    )
    admin_client = cast(
        client_module.PlexServer,
        SimpleNamespace(myPlexAccount=lambda: account),
    )

    monkeypatch.setattr(client_module, "_build_session", lambda url: None)
    monkeypatch.setattr(
        client_module,
        "_create_admin_client",
        lambda config, session=None: admin_client,
    )
    target_user = cast(
        client_module.MyPlexUser,
        SimpleNamespace(username="friend", id=2),
    )
    monkeypatch.setattr(
        client_module,
        "_match_plex_user",
        lambda name, users: target_user,
    )
    monkeypatch.setattr(
        client_module,
        "_create_user_client",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        client_module,
        "_resolve_display_name",
        lambda account, target, requested: "Friend",
    )

    bundle = client_module._default_bundle(plex_client_config)
    assert bundle.target_user is target_user
    assert bundle.is_admin is False
    assert bundle.display_name == "Friend"


def test_refresh_helpers_clear_errors(
    monkeypatch: pytest.MonkeyPatch, plex_client: client_module.PlexClient
):
    """Test that the refresh helpers handle errors and clear caches."""

    def broken_continue():
        raise RuntimeError("boom")

    section = cast(
        client_module.LibrarySection,
        SimpleNamespace(key="sec", continueWatching=broken_continue),
    )
    entry = plex_client._refresh_continue_cache(section)
    assert entry.keys == frozenset()

    plex_client._bundle = client_module.PlexClientBundle(
        admin_client=_server_stub(),
        user_client=_server_stub(),
        account=_account_stub(id=1, watchlist=lambda: []),
        target_user=None,
        user_id=1,
        display_name="Demo",
        is_admin=False,
    )
    assert plex_client._refresh_watchlist_cache().keys == frozenset()

    plex_client._bundle = client_module.PlexClientBundle(
        admin_client=_server_stub(),
        user_client=_server_stub(),
        account=_account_stub(
            id=1,
            watchlist=lambda: [
                SimpleNamespace(guid="guid"),
                SimpleNamespace(guid=None),
            ],
        ),
        target_user=None,
        user_id=1,
        display_name="Demo",
        is_admin=True,
    )
    entry = plex_client._refresh_watchlist_cache()
    assert entry.keys == frozenset({"guid"})

    plex_client._continue_cache = {
        "a": client_module._FrozenCacheEntry(keys=frozenset({"1"}), expires_at=0)
    }
    plex_client._ordering_cache = {1: "tmdb"}
    plex_client.clear_cache()
    assert not plex_client._continue_cache and not plex_client._ordering_cache
