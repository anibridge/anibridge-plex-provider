"""Tests for the SelectiveVerifySession helper."""

from typing import Any

import pytest
import requests

from anibridge_plex_provider.utils import SelectiveVerifySession


class _Recorder:
    """Helper object that records the most recent call arguments."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def request(self, method: str, url: str, **kwargs: Any) -> str:
        self.kwargs = dict(kwargs)
        return "ok"


def test_selective_verify_session_disables_verification(
    monkeypatch: pytest.MonkeyPatch,
):
    """SSL verification is disabled for whitelisted domains."""
    recorder = _Recorder()
    monkeypatch.setattr(requests.Session, "request", recorder.request)

    session = SelectiveVerifySession(whitelist={"plex.tv"})
    result = session.request("GET", "https://plex.tv/library")

    assert result == "ok"
    assert recorder.kwargs is not None
    assert recorder.kwargs.get("verify") is False


def test_selective_verify_session_passes_through_for_other_hosts(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that SSL verification is not disabled for non-whitelisted domains."""
    recorder = _Recorder()
    monkeypatch.setattr(requests.Session, "request", recorder.request)

    session = SelectiveVerifySession(whitelist={"plex.tv"})
    result = session.request("GET", "https://example.com")

    assert result == "ok"
    assert recorder.kwargs is not None
    assert "verify" not in recorder.kwargs
