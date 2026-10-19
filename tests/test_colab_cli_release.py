"""Contracts for the frontend-independent runtime release client."""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "research" / "colab_cli" / "release.py"
SPEC = importlib.util.spec_from_file_location("colab_cli_release", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def _credential(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "access_token": "short-lived-access-token",
        "expires_at": time.time() + 3600,
        "scopes": sorted(release.RELEASE_SCOPES),
    }
    value.update(overrides)
    return value


def test_credential_file_is_private_and_rejects_refresh_token(tmp_path: Path) -> None:
    credential = _credential()
    release.write_release_credential(tmp_path, "endpoint_1", credential, min_valid_seconds=10)
    path = tmp_path / release.RELEASE_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (
        release.load_release_credential(tmp_path, "endpoint_1", min_valid_seconds=10) == credential
    )

    with pytest.raises(release.ReleaseError, match="forbidden"):
        release.write_release_credential(
            tmp_path,
            "endpoint_1",
            _credential(refresh_token="must-not-leave-mac"),
            min_valid_seconds=10,
        )


def test_mint_fails_closed_when_provider_returns_broader_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        stdout = json.dumps(_credential(scopes=sorted(release.RELEASE_SCOPES | {"cloud-platform"})))

    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(release.ReleaseError, match="restricted release scopes"):
        release.mint_release_credential("fake-oauth-python", "", 10)


def test_mint_timeout_is_caller_clamped_to_robust_helper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[float] = []

    class Completed:
        stdout = json.dumps(_credential())

    def fake_run(*_args: object, **kwargs: object) -> Completed:
        seen.append(float(kwargs["timeout"]))
        return Completed()

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    assert release.mint_release_credential("fake-oauth-python", "", 10, timeout=999)
    assert len(seen) == 1
    assert 149 <= seen[0] <= 150
    with pytest.raises(release.ReleaseError, match="timeout must be positive"):
        release.mint_release_credential("fake-oauth-python", "", 10, timeout=0)


def test_mint_retries_transient_transport_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Completed:
        stdout = json.dumps(_credential())

    def fake_run(*_args: object, **_kwargs: object) -> Completed:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise subprocess.TimeoutExpired("oauth", 1)
        return Completed()

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    monkeypatch.setattr(release.time, "sleep", lambda _delay: None)
    assert release.mint_release_credential("fake-oauth-python", "", 10)
    assert calls == 3


def test_mint_scope_refusal_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class Completed:
        stdout = json.dumps(_credential(scopes=sorted(release.RELEASE_SCOPES | {"broader"})))

    def fake_run(*_args: object, **_kwargs: object) -> Completed:
        nonlocal calls
        calls += 1
        return Completed()

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    with pytest.raises(release.ReleaseError, match="restricted release scopes"):
        release.mint_release_credential("fake-oauth-python", "", 10)
    assert calls == 1


def test_mint_refuses_when_the_shared_timeout_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release.time, "monotonic", lambda: 1.0)
    with pytest.raises(release.ReleaseError, match="could not mint"):
        release.mint_release_credential("fake-oauth-python", "", 10, timeout=1e-99)


def test_unassign_replays_cli_get_xsrf_post_without_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    def fake_urlopen(request: object, timeout: int) -> Response:
        calls.append((request, timeout))
        if len(calls) == 1:
            return Response(b')]}\'\n{"token":"xsrf"}')
        return Response(b"{}")

    monkeypatch.setattr(release, "_open", fake_urlopen)
    release.unassign_with_credential(_credential(), "endpoint_1")

    get, post = (item[0] for item in calls)
    assert get.get_method() == "GET"
    assert post.get_method() == "POST"
    assert get.full_url == (
        "https://colab.research.google.com/tun/m/unassign/endpoint_1?authuser=0"
    )
    assert get.get_header("Authorization") == "Bearer short-lived-access-token"
    assert post.get_header("X-goog-colab-token") == "xsrf"


def test_mint_program_treats_naive_google_auth_expiry_as_utc(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Credentials:
        token = "short-lived-access-token"
        expiry = datetime(2030, 1, 2, 3, 4, 5)
        granted_scopes = set(release.RELEASE_SCOPES)

        @classmethod
        def from_authorized_user_file(cls, path: str, scopes: list[str]) -> "Credentials":
            assert set(scopes) == release.RELEASE_SCOPES
            return cls()

        def refresh(self, request: object) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.request = lambda *_args, **_kwargs: None

    raw_requests_module = types.ModuleType("requests")
    raw_requests_module.Session = Session
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = lambda *, session: session
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = Credentials
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "requests", raw_requests_module)
    monkeypatch.setattr(sys, "argv", ["mint", json.dumps(sorted(release.RELEASE_SCOPES))])

    exec(release._MINT_PROGRAM, {"__name__": "__main__"})

    result = json.loads(capsys.readouterr().out)
    assert result["expires_at"] == datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()


def test_release_retry_reverifies_endpoint_before_a_second_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    credential = _credential(expires_at=time.time() + release.RELEASE_MAX_SECONDS + 30)
    release.write_release_credential(
        tmp_path, "endpoint_1", credential, min_valid_seconds=release.RELEASE_MAX_SECONDS
    )
    calls: list[str] = []

    def fake_request(_url: str, _token: str, *, method: str = "GET", **_kwargs: object) -> bytes:
        calls.append(method)
        if method == "GET":
            return b')]}\'\n{"token":"fresh-xsrf"}'
        if calls.count("POST") == 1:
            raise release.ReleaseError("runtime release request failed: timeout")
        return b"{}"

    monkeypatch.setattr(release, "_request", fake_request)
    monkeypatch.setattr(release.time, "sleep", lambda _delay: None)
    release.release_from_file(tmp_path, "endpoint_1")
    assert calls == ["GET", "POST", "GET", "POST"]


def test_release_envelope_remains_fixed_when_mac_oauth_timeout_grows() -> None:
    assert release.REQUEST_TIMEOUT_SECONDS == 20
    assert release.RELEASE_MAX_SECONDS == 130


def test_rejects_nonfinite_expiry_and_redirects() -> None:
    with pytest.raises(release.ReleaseError, match="invalid expiry"):
        release._validate_credential(_credential(expires_at=float("nan")), "endpoint_1")
    with pytest.raises(release.ReleaseError, match="refusing release credential redirect"):
        release._RejectRedirect().redirect_request(
            None, None, 302, "Found", {}, "https://other.example"
        )
