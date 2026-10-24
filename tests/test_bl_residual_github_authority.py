"""Hermetic adversaries for the I2 GitHub publication/time authority."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import socket
import ssl
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from research.bl_ppo.exports import export_json
from research.bl_residual import github_authority as authority


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(root: Path, text: str, author: int, committer: int) -> str:
    (root / "ledger").write_text(text)
    env = os.environ | {
        "GIT_AUTHOR_DATE": datetime.fromtimestamp(author, timezone.utc).isoformat(),
        "GIT_COMMITTER_DATE": datetime.fromtimestamp(committer, timezone.utc).isoformat(),
    }
    _git(root, "add", "ledger", env=env)
    _git(root, "commit", "-qm", "synthetic", env=env)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def synthetic_git(tmp_path: Path) -> Path:
    root = tmp_path / "git"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "config", "user.name", "Synthetic")
    return root


def _event(commit: str, before: str, timestamp: str, *, event_id: str = "event-1") -> dict:
    return {
        "id": event_id,
        "ref": authority.REF,
        "activity_type": "push",
        "before": before,
        "after": commit,
        "timestamp": timestamp,
    }


def _server_objects(root: Path, endpoint: str):
    """Model server-owned raw objects, independently of the client's Git helpers."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_GRAFT_FILE=os.devnull)

    def raw(*args: str) -> bytes:
        return subprocess.run(
            ["git", "--no-replace-objects", *args],
            cwd=root,
            env=env,
            capture_output=True,
            check=True,
        ).stdout

    if "/compare/" in endpoint:
        base, tip = endpoint.split("/compare/", 1)[1].split("...")
        merge = raw("merge-base", base, tip).decode().strip()
        status = "identical" if base == tip else "ahead" if merge == base else "diverged"
        if merge == tip and base != tip:
            status = "behind"
        return {"base_commit": {"sha": base}, "merge_base_commit": {"sha": merge}, "status": status}
    if "/contents/" in endpoint:
        parsed = urlparse(endpoint)
        name = parsed.path.split("/contents/", 1)[1]
        commit = parse_qs(parsed.query)["ref"][0]
        try:
            payload = raw("show", f"{commit}:{name}")
        except subprocess.CalledProcessError as exc:
            raise authority.AuthorityUnavailable("server file absent") from exc
        return {
            "type": "file",
            "path": name,
            "encoding": "base64",
            "content": base64.b64encode(payload).decode(),
            "size": len(payload),
            "sha": hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload
            ).hexdigest(),
        }
    raise AssertionError(endpoint)


def _stub_activity(
    monkeypatch: pytest.MonkeyPatch, head: str, rows: list[dict], *, epoch: float
) -> None:
    def request(endpoint: str):
        if "/compare/" in endpoint or "/contents/" in endpoint:
            return _server_objects(authority.f2.ROOT, endpoint), {}, {"epoch": epoch}
        if endpoint.endswith("git/ref/heads/main"):
            return (
                {"ref": authority.REF, "object": {"type": "commit", "sha": head}},
                {},
                {"epoch": epoch},
            )
        assert endpoint.startswith(f"{authority.API}/activity?")
        return rows, {}, {"epoch": epoch}

    monkeypatch.setattr(authority, "_request", request)


def test_delayed_publication_uses_push_transition_not_old_git_dates(
    synthetic_git: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old ledger commit remains ineligible until the later server push."""
    before = _commit(synthetic_git, "base", 10, 10)
    ledger = _commit(synthetic_git, "ledger", 20, 20)  # deliberately before cutoff
    tip = _commit(synthetic_git, "later", 30, 30)
    monkeypatch.setattr(authority.f2, "ROOT", synthetic_git)
    _stub_activity(
        monkeypatch,
        tip,
        [_event(tip, before, "2026-10-24T03:00:00Z")],  # independently late publication
        epoch=datetime(2026, 10, 24, 4, tzinfo=timezone.utc).timestamp(),
    )
    event, _ = authority.first_publication(ledger)
    assert event["id"] == "event-1"
    assert event["timestamp"] == "2026-10-24T03:00:00Z"
    assert event["epoch"] != 20


def test_backdated_author_and_committer_dates_cannot_backdate_rebased_commit(
    synthetic_git: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --committer-date-is-author-date style rebase needs its own later event."""
    before = _commit(synthetic_git, "base", 10, 10)
    old = _commit(synthetic_git, "ledger", 11, 12)
    _git(synthetic_git, "rebase", "--committer-date-is-author-date", "--root")
    rebased = _git(synthetic_git, "rev-parse", "HEAD")
    assert rebased != old
    monkeypatch.setattr(authority.f2, "ROOT", synthetic_git)
    _stub_activity(
        monkeypatch,
        rebased,
        [_event(rebased, before, "2026-10-24T05:00:00Z", event_id="rebase-push")],
        epoch=datetime(2026, 10, 24, 6, tzinfo=timezone.utc).timestamp(),
    )
    event, _ = authority.first_publication(rebased)
    assert event["id"] == "rebase-push"
    assert event["epoch"] > 11


def test_multi_commit_push_binds_transition_that_first_made_ledger_reachable(
    synthetic_git: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _commit(synthetic_git, "base", 10, 10)
    ledger = _commit(synthetic_git, "ledger", 20, 20)
    after = _commit(synthetic_git, "second commit in same push", 21, 21)
    monkeypatch.setattr(authority.f2, "ROOT", synthetic_git)
    _stub_activity(
        monkeypatch,
        after,
        [_event(after, before, "2026-10-24T07:00:00Z", event_id="two-commits")],
        epoch=datetime(2026, 10, 24, 8, tzinfo=timezone.utc).timestamp(),
    )
    event, _ = authority.first_publication(ledger)
    assert event == {
        "id": "two-commits",
        "timestamp": "2026-10-24T07:00:00Z",
        "epoch": datetime(2026, 10, 24, 7, tzinfo=timezone.utc).timestamp(),
        "before": before,
        "after": after,
        "ref": authority.REF,
        "activity_type": "push",
    }


@pytest.mark.parametrize("bad_rows", [[], [{"id": "bad"}], "not-a-list"])
def test_missing_or_malformed_activity_is_unverifiable(
    synthetic_git: Path, monkeypatch: pytest.MonkeyPatch, bad_rows: object
) -> None:
    _commit(synthetic_git, "base", 10, 10)
    commit = _commit(synthetic_git, "ledger", 20, 20)
    monkeypatch.setattr(authority.f2, "ROOT", synthetic_git)
    _stub_activity(monkeypatch, commit, bad_rows, epoch=100)  # type: ignore[arg-type]
    with pytest.raises(authority.AuthorityUnavailable):
        authority.first_publication(commit)


@pytest.mark.parametrize(
    "bad_head",
    [None, {}, {"ref": authority.REF, "object": None}, {"ref": authority.REF, "object": []}],
)
def test_missing_or_malformed_main_ref_is_unverifiable(
    monkeypatch: pytest.MonkeyPatch, bad_head: object
) -> None:
    monkeypatch.setattr(authority, "_request", lambda _: (bad_head, {}, {"epoch": 100.0}))
    with pytest.raises(authority.AuthorityUnavailable):
        authority.first_publication("a" * 40)


@pytest.mark.parametrize(
    "bad_row",
    [
        None,
        [],
        {},
        {"ref": authority.REF},
        {"ref": authority.REF, "before": None, "after": "a" * 40},
    ],
)
def test_null_or_partial_activity_row_is_unverifiable(
    synthetic_git: Path, monkeypatch: pytest.MonkeyPatch, bad_row: object
) -> None:
    _commit(synthetic_git, "base", 10, 10)
    commit = _commit(synthetic_git, "ledger", 20, 20)
    monkeypatch.setattr(authority.f2, "ROOT", synthetic_git)
    _stub_activity(monkeypatch, commit, [bad_row], epoch=100)  # type: ignore[list-item]
    with pytest.raises(authority.AuthorityUnavailable):
        authority.first_publication(commit)


def test_sealed_evidence_revalidation_refuses_identity_and_event_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = {
        "path": str(tmp_path / "sealed-receipt.json"),
        "publication_ref": "a" * 40,
        "sha256": "b" * 64,
    }
    identity = {
        "schema": authority.TIME_POLICY["publication_schema"],
        "repository": authority.REPOSITORY,
        "ref": authority.REF,
        "commit": "a" * 40,
        "ledger": "research/bl_residual/ledger/f2-result-20261024.json",
        "ledger_sha256": "c" * 64,
        "receipt_sha256": "b" * 64,
    }
    event = {
        "id": "event",
        "timestamp": "2026-10-24T01:00:00Z",
        "epoch": 100.0,
        "before": "d" * 40,
        "after": "a" * 40,
        "ref": authority.REF,
        "activity_type": "push",
    }
    path = authority.evidence_path(binding)
    monkeypatch.setattr(authority, "_identity", lambda _: identity)
    monkeypatch.setattr(authority, "first_publication", lambda _: (event, {"epoch": 101.0}))
    authority.capture_publication(binding)
    assert authority.publication_evidence(binding)["published_epoch"] == 100.0
    for key, value in [
        ("ledger_sha256", "0" * 64),
        ("receipt_sha256", "0" * 64),
        ("ref", "refs/heads/other"),
        ("activity", {**event, "id": "other"}),
    ]:
        receipt = json.loads(path.read_text())
        receipt[key] = value
        # A fresh create-only sealed receipt models a substituted artifact, including sidecars.
        tampered = path.with_name(f"{path.stem}-{key}.json")
        export_json(tampered, receipt)
        monkeypatch.setattr(authority, "evidence_path", lambda _binding, p=tampered: p)
        assert authority.publication_evidence(binding) is None


def test_absent_or_malformed_publication_receipt_returns_strikes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = {
        "path": str(tmp_path / "sealed-receipt.json"),
        "publication_ref": "a" * 40,
        "sha256": "b" * 64,
    }
    path = authority.evidence_path(binding)
    assert authority.publication_evidence(binding) is None
    export_json(path, {"not": "a publication receipt"})
    monkeypatch.setattr(
        authority, "_identity", lambda _: (_ for _ in ()).throw(ValueError("bad identity"))
    )
    assert authority.publication_evidence(binding) is None


def test_receipt_list_is_strikes_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = {
        "path": str(tmp_path / "sealed-receipt.json"),
        "publication_ref": "a" * 40,
        "sha256": "b" * 64,
    }
    export_json(authority.evidence_path(binding), [])
    monkeypatch.setattr(authority, "_identity", lambda _: {"commit": "a" * 40})
    monkeypatch.setattr(
        authority, "first_publication", lambda _: ({"id": "event"}, {"epoch": 101.0})
    )
    assert authority.publication_evidence(binding) is None


def test_publication_revalidation_returns_strikes_when_github_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = {
        "path": str(tmp_path / "sealed-receipt.json"),
        "publication_ref": "a" * 40,
        "sha256": "b" * 64,
    }
    identity = {"commit": "a" * 40}
    event = {"id": "event", "epoch": 100.0}
    monkeypatch.setattr(authority, "_identity", lambda _: identity)
    monkeypatch.setattr(authority, "first_publication", lambda _: (event, {"epoch": 101.0}))
    authority.capture_publication(binding)
    monkeypatch.setattr(
        authority,
        "first_publication",
        lambda _: (_ for _ in ()).throw(authority.AuthorityUnavailable("offline")),
    )
    assert authority.publication_evidence(binding) is None


def test_rebased_commit_refuses_evidence_sealed_for_the_original_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = {
        "path": str(tmp_path / "sealed-receipt.json"),
        "publication_ref": "b" * 40,
        "sha256": "c" * 64,
    }
    identity = {"commit": "b" * 40, "ledger_sha256": "d" * 64, "receipt_sha256": "c" * 64}
    original_event = {"id": "old", "epoch": 10.0, "after": "a" * 40}
    rebased_event = {"id": "rebased", "epoch": 100.0, "after": "b" * 40}
    monkeypatch.setattr(authority, "_identity", lambda _: identity)
    monkeypatch.setattr(authority, "first_publication", lambda _: (rebased_event, {"epoch": 101.0}))
    authority.capture_publication(binding)
    path = authority.evidence_path(binding)
    receipt = json.loads(path.read_text())
    receipt["activity"] = original_event
    substituted = path.with_name("original-commit-evidence.json")
    export_json(substituted, receipt)
    monkeypatch.setattr(authority, "evidence_path", lambda _: substituted)
    assert authority.publication_evidence(binding) is None


def test_activity_pagination_selects_earliest_qualifying_push_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    newest_after, newest_before = "b" * 40, "c" * 40
    earliest_after, earliest_before = "d" * 40, "e" * 40
    next_url = f"https://api.github.com/{authority.API}/activity?ref={authority.REF}&page=2"
    calls = []

    def request(endpoint: str):
        calls.append(endpoint)
        if endpoint.endswith("git/ref/heads/main"):
            return (
                {"ref": authority.REF, "object": {"type": "commit", "sha": newest_after}},
                {},
                {"epoch": 300.0},
            )
        if "page=2" in endpoint:
            return (
                [
                    _event(
                        earliest_after, earliest_before, "1970-01-01T00:01:40Z", event_id="earliest"
                    )
                ],
                {},
                {"epoch": 300.0},
            )
        return (
            [_event(newest_after, newest_before, "1970-01-01T00:03:20Z", event_id="newest")],
            {"link": f'<{next_url}>; rel="next"'},
            {"epoch": 300.0},
        )

    # Each page models a multi-commit push: the ledger is reachable from after,
    # but not its before. A later reintroduction must not hide the earlier event.
    monkeypatch.setattr(authority, "_request", request)
    monkeypatch.setattr(
        authority,
        "_contains",
        lambda checked, tip: checked == commit and tip in {newest_after, earliest_after},
    )
    event, _ = authority.first_publication(commit)
    assert event["id"] == "earliest"
    assert calls[-1].endswith("page=2")


def test_activity_pagination_cycle_is_unverifiable(
    synthetic_git: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _commit(synthetic_git, "base", 10, 10)
    commit = _commit(synthetic_git, "ledger", 20, 20)
    monkeypatch.setattr(authority.f2, "ROOT", synthetic_git)
    endpoint = f"https://api.github.com/{authority.API}/activity?ref={authority.REF}&page=2"

    def request(value: str):
        if "/compare/" in value:
            return _server_objects(synthetic_git, value), {}, {"epoch": 100}
        if value.endswith("git/ref/heads/main"):
            return (
                {"ref": authority.REF, "object": {"type": "commit", "sha": commit}},
                {},
                {"epoch": 100},
            )
        return [], {"link": f'<{endpoint}>; rel="next"'}, {"epoch": 100}

    monkeypatch.setattr(authority, "_request", request)
    with pytest.raises(authority.AuthorityUnavailable, match="pagination"):
        authority.first_publication(commit)


def test_foreign_activity_pagination_is_unverifiable(
    synthetic_git: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _commit(synthetic_git, "base", 10, 10)
    commit = _commit(synthetic_git, "ledger", 20, 20)
    monkeypatch.setattr(authority.f2, "ROOT", synthetic_git)

    def request(endpoint: str):
        if "/compare/" in endpoint:
            return _server_objects(synthetic_git, endpoint), {}, {"epoch": 100}
        if endpoint.endswith("git/ref/heads/main"):
            return (
                {"ref": authority.REF, "object": {"type": "commit", "sha": commit}},
                {},
                {"epoch": 100},
            )
        return (
            [],
            {
                "link": (
                    f"<https://other.invalid/{authority.API}/activity?ref={authority.REF}>; "
                    'rel="next"'
                )
            },
            {"epoch": 100},
        )

    monkeypatch.setattr(authority, "_request", request)
    with pytest.raises(authority.AuthorityUnavailable, match="foreign"):
        authority.first_publication(commit)


class _HTTPSProbe:
    """Fake only the wire, exercising the real policy, credential and Date checks."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = []
        self.headers = [
            ("Date", "Thu, 01 Jan 1970 00:16:40 GMT"),
            ("X-GitHub-Request-Id", "synthetic"),
            ("X-RateLimit-Remaining", "59"),
        ]
        self.payload = {"ref": authority.REF, "object": {"type": "commit", "sha": "a" * 40}}
        self.status = 200
        self.failure = None
        self.bad_peer = False
        self.closed = False
        self.anonymous = False
        self.repo = {"full_name": authority.REPOSITORY, "private": False, "visibility": "public"}
        monkeypatch.setattr(authority.http.client, "HTTPSConnection", self.connection)
        monkeypatch.setattr(authority, "_token", lambda: "secret-test-token")
        monkeypatch.setattr(authority.time, "time", lambda: 1000.0)
        monkeypatch.setattr(authority.time, "monotonic", lambda: 10.0)

    def connection(self, host, port, *, timeout, context):
        assert (host, port, timeout) == ("api.github.com", 443, 30)
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
        self.calls.append((host, port))
        probe = self

        class Connection:
            def __init__(self):
                self.host, self.port = host, port
                self.sock = SimpleNamespace(
                    context=context,
                    server_hostname="foreign.invalid" if probe.bad_peer else host,
                    getpeercert=lambda: {"subjectAltName": (("DNS", host),)},
                    version=lambda: "TLSv1.3",
                    settimeout=lambda seconds: None,
                )

            def connect(self):
                if probe.failure:
                    raise probe.failure

            def request(self, method, path, *, headers):
                assert method == "GET" and (
                    path == "/" + authority.API or path.startswith("/" + authority.API + "/")
                )
                assert headers["Host"] == host
                if probe.anonymous:
                    assert "Authorization" not in headers
                else:
                    assert headers["Authorization"] == "Bearer secret-test-token"
                self.path = path
                probe.calls.append((method, path))

            def getresponse(self):
                payload = probe.repo if self.path == "/" + authority.API else probe.payload
                chunks = iter((json.dumps(payload).encode(), b""))
                return SimpleNamespace(
                    status=probe.status,
                    getheaders=lambda: probe.headers,
                    read1=lambda size: next(chunks),
                    isclosed=lambda: False,
                )

            def close(self):
                probe.closed = True

        return Connection()


@pytest.fixture
def clean_transport(tmp_path, monkeypatch):
    # Hermetic policy tests: do not inherit developer-machine proxies or config.
    for key in list(os.environ):
        if key.upper() in authority._OVERRIDE_ENV or key.upper().endswith("_PROXY"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    token = authority._token
    token.cache_clear()
    yield
    token.cache_clear()


def test_request_accepts_fresh_https_date_and_refuses_clock_or_transport_failures(
    monkeypatch: pytest.MonkeyPatch, clean_transport
) -> None:
    probe = _HTTPSProbe(monkeypatch)
    _, _, observed = authority._request(f"{authority.API}/git/ref/heads/main")
    assert observed["epoch"] == 1000.0
    assert observed["verified_peer_host"] == "api.github.com"
    assert observed["peer_port"] == 443
    assert observed["tls_verified"] is True
    assert observed["tls_version"] == "TLSv1.3"
    assert observed["request_id"] == "synthetic"
    assert "secret-test-token" not in json.dumps(observed)
    assert probe.closed

    for date, wall, duration in [
        (None, 1000.0, 0.2),
        ("not a date", 1000.0, 0.2),
        ("Thu, 01 Jan 1970 00:16:40 GMT", 1121.0, 0.2),
        ("Thu, 01 Jan 1970 00:16:40 GMT", 879.0, 0.2),
        ("Thu, 01 Jan 1970 00:16:40 GMT", 1000.0, 31.0),
    ]:
        probe.headers = [("X-GitHub-Request-Id", "synthetic")]
        if date is not None:
            probe.headers.append(("Date", date))
        monkeypatch.setattr(authority.time, "time", lambda w=wall: w)
        ticks = iter((10.0, 10.0, 10.0, 10.0 + duration))
        monkeypatch.setattr(authority.time, "monotonic", lambda: next(ticks))
        with pytest.raises(authority.AuthorityUnavailable):
            authority._request(f"{authority.API}/git/ref/heads/main")

    probe.failure = OSError("synthetic connection failure")
    monkeypatch.setattr(authority.time, "monotonic", lambda: 50.0)
    with pytest.raises(authority.AuthorityUnavailable):
        authority._request(f"{authority.API}/git/ref/heads/main")


@pytest.mark.parametrize(
    "key",
    [
        "GH_CONFIG_DIR",
        "GH_HOST",
        "GH_API_URL",
        "GITHUB_API_URL",
        "GITHUB_GRAPHQL_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ],
)
def test_transport_environment_override_refuses_time_and_publication(
    tmp_path, monkeypatch, clean_transport, key
):
    probe = _HTTPSProbe(monkeypatch)
    monkeypatch.setenv(key, "sensitive-override-value")
    with pytest.raises(authority.AuthorityUnavailable, match="override") as exc:
        authority.trusted_time()
    assert "sensitive-override-value" not in str(exc.value)
    # Exercise actual publication adapter, without any real F2 paths or payload.
    binding = {"path": str(tmp_path / "synthetic-receipt"), "publication_ref": "a" * 40}
    export_json(authority.evidence_path(binding), {})
    monkeypatch.setattr(authority, "_identity", lambda _: authority.trusted_time())
    assert authority.publication_evidence(binding) is None
    assert probe.calls == []


def test_real_gh_unix_socket_configuration_refuses_before_network(
    tmp_path, monkeypatch, clean_transport
):
    # The installed CLI writes its actual supported config format. No gh API
    # call, network listener thread, or extra probe process is needed.
    gh = shutil.which("gh")
    if not gh:
        pytest.skip("gh CLI not installed")
    probe = _HTTPSProbe(monkeypatch)
    directory = Path(os.environ["XDG_CONFIG_HOME"]) / "gh"
    directory.mkdir(parents=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        # AF_UNIX paths on macOS are limited to 104 bytes.
        unix_path = str(Path("/tmp") / ("i2-" + str(os.getpid()) + ".sock"))
        listener.bind(unix_path)
        listener.listen()
        listener.setblocking(False)
        subprocess.run(
            [gh, "config", "set", "http_unix_socket", unix_path],
            env={
                "PATH": os.environ["PATH"],
                "GH_CONFIG_DIR": str(directory),
                "HOME": str(tmp_path),
                "XDG_STATE_HOME": str(tmp_path / "state"),
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        with pytest.raises(authority.AuthorityUnavailable, match="Unix socket"):
            authority.trusted_time()
        with pytest.raises(BlockingIOError):
            listener.accept()
    Path(unix_path).unlink()
    assert probe.calls == []


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_cross_host_redirect_is_never_followed_or_given_credentials(
    monkeypatch, clean_transport, status
):
    probe = _HTTPSProbe(monkeypatch)
    probe.status = status
    probe.headers.append(("Location", "https://foreign.invalid/stolen"))
    with pytest.raises(authority.AuthorityUnavailable, match="redirects refused"):
        authority.trusted_time()
    assert len(probe.calls) == 2  # one connection and one GET, no redirect request
    assert probe.closed


@pytest.mark.parametrize(
    "failure",
    [
        ssl.SSLCertVerificationError("certificate verify failed"),
        ssl.SSLError("TLS failure"),
    ],
)
def test_tls_failure_is_unavailable(monkeypatch, clean_transport, failure):
    probe = _HTTPSProbe(monkeypatch)
    probe.failure = failure
    with pytest.raises(authority.AuthorityUnavailable):
        authority.trusted_time()
    assert probe.closed
    assert len(probe.calls) == 1  # no Authorization header sent before verified TLS


def test_foreign_peer_is_unavailable_before_credentials_sent(monkeypatch, clean_transport):
    probe = _HTTPSProbe(monkeypatch)
    probe.bad_peer = True
    with pytest.raises(authority.AuthorityUnavailable, match="peer"):
        authority.trusted_time()
    assert len(probe.calls) == 1
    assert probe.closed


def test_system_tls_context_loads_explicit_system_paths(monkeypatch, clean_transport):
    context = authority._system_tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.cert_store_stats()["x509_ca"] > 0
    # An injected env-resolved path is not used by the trust loader.
    paths = ssl.get_default_verify_paths()
    monkeypatch.setattr(
        authority.ssl,
        "get_default_verify_paths",
        lambda: SimpleNamespace(
            cafile="/foreign/cert.pem",
            capath="/foreign/certs",
            openssl_cafile=paths.openssl_cafile,
            openssl_capath=paths.openssl_capath,
        ),
    )
    assert authority._system_tls_context().cert_store_stats() == context.cert_store_stats()


def test_token_uses_sanitized_auth_only_once(monkeypatch, clean_transport, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    for key in ("GH_TOKEN", "GH_DEBUG", "GIT_CONFIG_COUNT", "GH_ENTERPRISE_TOKEN"):
        monkeypatch.setenv(key, "must-not-be-forwarded")
    calls = []

    def run(args, **kwargs):
        assert args == ["gh", "auth", "token", "--hostname", "github.com"]
        assert kwargs["capture_output"] and kwargs["timeout"] == 10
        assert kwargs["env"]["GH_PROMPT_DISABLED"] == "1"
        assert not any(
            key in kwargs["env"]
            for key in (
                "GH_TOKEN",
                "GH_DEBUG",
                "GIT_CONFIG_COUNT",
                "GH_ENTERPRISE_TOKEN",
            )
        )
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="secret-test-token\n")

    monkeypatch.setattr(authority.subprocess, "run", run)
    assert authority._token() == authority._token() == "secret-test-token"
    assert len(calls) == 1
    assert capsys.readouterr() == ("", "")


def test_github_token_preferred_and_never_logged(monkeypatch, clean_transport, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-github-token")
    monkeypatch.setattr(authority.subprocess, "run", lambda *a, **k: pytest.fail("gh invoked"))
    assert authority._token() == "secret-github-token"
    monkeypatch.setenv("GITHUB_TOKEN", "different-token")
    assert authority._token() == "secret-github-token"
    assert capsys.readouterr() == ("", "")


def test_failed_auth_never_logs_token_or_subprocess_output(monkeypatch, clean_transport, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        authority.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1,
            stdout="sensitive-output",
            stderr="sensitive-error",
        ),
    )
    assert authority._token() is None
    assert capsys.readouterr() == ("", "")


def _result_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.test_bl_residual_f2 import git_fixture, seal_fixture

    root, receipt = git_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(authority.f2, "ROOT", root)
    before = _git(root, "rev-parse", "HEAD")
    seal_fixture(root, receipt)
    authority.f2.stage_publication(root=root)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "synthetic result")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", commit)
    binding = authority.published_receipt_binding(receipt, root=root)
    return root, before, commit, binding


def _strikes_verdict(monkeypatch, tmp_path, evidence) -> str:
    """Exercise only the pure funding statistic, never R-A or its entrypoint."""
    from research.bl_residual import funding, ra_guard
    from research.bl_residual.protocol import canonical_sha
    from tests.test_bl_residual_f4_fit import _look

    monkeypatch.setattr(ra_guard, "D4096_CONTROL", tmp_path / "absent-control.json")
    timing = ra_guard.timing_identity(None, evidence)
    eligible = ra_guard.timing_eligible(timing, at=ra_guard.CUTOFF + 120)
    left, right = _look(256, "2" * 64, 0.5), _look(1024, "1" * 64, 0.5)
    result = funding.evaluate(
        left,
        right,
        {
            "schema": authority.f2.SCHEMA,
            "volume_struck": False,
            "published": True,
            "source_sha256": "a" * 64,
            "published_ref": "b" * 40,
        },
        reserve_identity={
            "series_id": left["namespace"],
            "reserve_ancestry_sha256": "z" * 64,
            "reserve_life_ids": left["reserve_life_ids"],
            "reserve_allocation_sha256": canonical_sha(left["reserve_life_ids"]),
        },
        f2_timing_eligible=eligible,
    )
    assert not eligible
    return result["verdict"]


@pytest.mark.parametrize("attack", ["replacement", "graft"])
def test_false_early_ancestry_cannot_backdate_exact_result_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    """Review probe 1: B->A early, C first published by a later force-push."""
    from research.bl_residual import ra_guard

    root, before, commit, binding = _result_fixture(tmp_path, monkeypatch)
    tree = _git(root, "rev-parse", before + "^{tree}")
    early = _git(root, "commit-tree", tree, "-p", before, "-m", "genuine early A")
    if attack == "replacement":
        fake = _git(root, "commit-tree", tree, "-p", commit, "-m", "forged A parent C")
        _git(root, "replace", early, fake)
    else:
        (root / ".git/info/grafts").write_text(f"{early} {commit}\n")
    # Establish the original exploit without using the hardened client helper.
    _git(root, "merge-base", "--is-ancestor", commit, early)
    assert _git(root, "status", "--porcelain") == ""
    with pytest.raises(subprocess.CalledProcessError):
        authority.f2._git(root, "merge-base", "--is-ancestor", commit, early)
    stamp = (
        lambda epoch: datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")
    )
    late = {
        **_event(commit, early, stamp(ra_guard.CUTOFF + 60), event_id="late"),
        "activity_type": "force_push",
    }
    _stub_activity(
        monkeypatch,
        commit,
        [_event(early, before, stamp(ra_guard.CUTOFF - 60), event_id="early"), late],
        epoch=ra_guard.CUTOFF + 120,
    )
    authority.capture_publication(binding)
    evidence = authority.publication_evidence(binding)
    assert evidence["activity"]["id"] == "late"
    assert evidence["published_epoch"] == ra_guard.CUTOFF + 60
    assert _strikes_verdict(monkeypatch, tmp_path, evidence) == "STOP_FUNDING"


@pytest.mark.parametrize("compromised_local_helper", [False, True])
def test_replacement_ledger_tree_cannot_bind_absent_server_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compromised_local_helper: bool
) -> None:
    """Review probe 2: real early A has no result ledger; replacement supplies it."""
    from research.bl_ppo.exports import digest
    from research.bl_residual import ra_guard

    root, before, result, binding = _result_fixture(tmp_path, monkeypatch)
    tree = _git(root, "rev-parse", before + "^{tree}")
    early = _git(root, "commit-tree", tree, "-p", before, "-m", "early with no result")
    result_tree = _git(root, "rev-parse", result + "^{tree}")
    fake = _git(root, "commit-tree", result_tree, "-p", before, "-m", "fake ledger tree")
    _git(root, "replace", early, fake)
    _git(root, "update-ref", "HEAD", early)
    _git(root, "update-ref", "refs/remotes/origin/main", early)
    binding["publication_ref"] = early
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "show", f"{early}:{authority.f2.RESULT_LEDGER}")
    with pytest.raises(subprocess.CalledProcessError):
        authority.f2._git(root, "show", f"{early}:{authority.f2.RESULT_LEDGER}")
    # Actual preparer/consumer binding path must also yield strikes when raw
    # Git exposes the missing ledger, before publication_evidence is called.
    assert ra_guard.receipt_binding(Path(binding["path"])) is None
    if compromised_local_helper:
        # Independently prove the server layer rejects even the old unsafe Git
        # interpretation, rather than succeeding only because local Git refuses.
        monkeypatch.setattr(
            authority.f2,
            "_git",
            lambda root, *args: subprocess.check_output(["git", *args], cwd=root),
        )
    stamp = (
        datetime.fromtimestamp(ra_guard.CUTOFF - 60, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    event = _event(early, before, stamp)
    _stub_activity(monkeypatch, early, [event], epoch=ra_guard.CUTOFF + 120)
    # Model an evidence receipt accepted by the old vulnerable capture path.
    export_json(
        authority.evidence_path(binding),
        {
            "schema": authority.TIME_POLICY["publication_schema"],
            "repository": authority.REPOSITORY,
            "ref": authority.REF,
            "commit": early,
            "ledger": authority.f2.RESULT_LEDGER,
            "ledger_sha256": digest(root / authority.f2.RESULT_LEDGER),
            "receipt_sha256": binding["sha256"],
            "activity": {**event, "epoch": ra_guard.CUTOFF - 60},
        },
    )
    with pytest.raises((authority.AuthorityUnavailable, subprocess.CalledProcessError)):
        authority.capture_publication(binding)
    assert authority.publication_evidence(binding) is None
    assert _strikes_verdict(monkeypatch, tmp_path, None) == "STOP_FUNDING"


@pytest.mark.parametrize("endpoint_kind", ["compare", "contents"])
def test_server_binding_unavailable_never_falls_back_to_valid_local_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint_kind: str
) -> None:
    root, before, commit, binding = _result_fixture(tmp_path, monkeypatch)
    _stub_activity(monkeypatch, commit, [_event(commit, before, "1970-01-01T00:01:00Z")], epoch=100)
    authority.capture_publication(binding)
    request = authority._request

    def unavailable(endpoint):
        if f"/{endpoint_kind}/" in endpoint:
            raise authority.AuthorityUnavailable("synthetic API unavailable")
        return request(endpoint)

    monkeypatch.setattr(authority, "_request", unavailable)
    assert authority.publication_evidence(binding) is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {
            "base_commit": {"sha": "b" * 40},
            "merge_base_commit": {"sha": "a" * 40},
            "status": "ahead",
        },
        {
            "base_commit": {"sha": "a" * 40},
            "merge_base_commit": {"sha": "a" * 40},
            "status": "behind",
        },
        {
            "base_commit": {"sha": "a" * 40},
            "merge_base_commit": {"sha": "invalid"},
            "status": "diverged",
        },
    ],
)
def test_unverifiable_server_comparison_refuses(monkeypatch, value) -> None:
    monkeypatch.setattr(authority, "_request", lambda _: (value, {}, {}))
    with pytest.raises(authority.AuthorityUnavailable):
        authority._contains("a" * 40, "b" * 40)


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "symlink"),
        ("path", "foreign"),
        ("encoding", "none"),
        ("content", "%%%"),
        ("size", True),
        ("size", 2_000_001),
        ("sha", "0" * 40),
        ("content", ""),
    ],
)
def test_unverifiable_server_blob_refuses(tmp_path, monkeypatch, field, value) -> None:
    path = tmp_path / authority.f2.INPUT_LEDGER
    export_json(path, {"synthetic": True})
    payload = path.read_bytes()
    response = {
        "type": "file",
        "path": authority.f2.INPUT_LEDGER,
        "encoding": "base64",
        "content": base64.b64encode(payload).decode(),
        "size": len(payload),
        "sha": hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(),
        field: value,
    }
    monkeypatch.setattr(authority, "_request", lambda _: (response, {}, {}))
    with pytest.raises(authority.AuthorityUnavailable):
        authority._server_export("a" * 40, authority.f2.INPUT_LEDGER, tmp_path)


@pytest.mark.parametrize("relative", [authority.f2.RESULT_LEDGER, authority.f2.INPUT_LEDGER])
@pytest.mark.parametrize("suffix", ["", ".sha256", ".complete"])
def test_server_bytes_rechecked_for_each_ledger_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str, suffix: str
) -> None:
    root, before, commit, binding = _result_fixture(tmp_path, monkeypatch)
    _stub_activity(monkeypatch, commit, [_event(commit, before, "1970-01-01T00:01:00Z")], epoch=100)
    authority.capture_publication(binding)
    assert authority.publication_evidence(binding) is not None
    request = authority._request

    def substituted(endpoint):
        value, headers, observation = request(endpoint)
        if f"/contents/{relative}{suffix}?" in endpoint:
            payload = b"foreign server bytes\n"
            value = {
                **value,
                "content": base64.b64encode(payload).decode(),
                "size": len(payload),
                "sha": hashlib.sha1(
                    b"blob " + str(len(payload)).encode() + b"\0" + payload
                ).hexdigest(),
            }
        return value, headers, observation

    monkeypatch.setattr(authority, "_request", substituted)
    assert authority.publication_evidence(binding) is None


def test_local_identity_git_strips_environment_overrides(tmp_path, monkeypatch) -> None:
    root, before, commit, _ = _result_fixture(tmp_path, monkeypatch)
    fake = _git(root, "commit-tree", _git(root, "rev-parse", before + "^{tree}"), "-m", "foreign")
    _git(root, "replace", commit, fake)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _git(foreign, "init", "-q")
    grafts = tmp_path / "caller-grafts"
    grafts.write_text(f"{commit}\n")
    for key, value in {
        "GIT_DIR": str(foreign / ".git"),
        "GIT_WORK_TREE": str(foreign),
        "GIT_OBJECT_DIRECTORY": str(foreign / ".git/objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(foreign / ".git/objects"),
        "GIT_NO_REPLACE_OBJECTS": "0",
        "GIT_GRAFT_FILE": str(grafts),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.useReplaceRefs",
        "GIT_CONFIG_VALUE_0": "true",
        "GIT_REPLACE_REF_BASE": "refs/replace/",
    }.items():
        monkeypatch.setenv(key, value)
    assert authority.f2._git(root, "rev-parse", "HEAD").decode().strip() == commit
    assert authority.f2._git(root, "merge-base", before, commit).decode().strip() == before
    assert authority.f2._git(root, "show", f"{commit}:{authority.f2.RESULT_LEDGER}")


def test_cyclic_gh_configuration_is_unavailable(tmp_path, monkeypatch, clean_transport):
    directory = Path(os.environ["XDG_CONFIG_HOME"]) / "gh"
    directory.mkdir(parents=True)
    (directory / "config.yml").write_text("loop: &loop {loop: *loop}\n")
    probe = _HTTPSProbe(monkeypatch)
    with pytest.raises(authority.AuthorityUnavailable, match="configuration"):
        authority.trusted_time()
    assert probe.calls == []


def test_deadline_covers_stalled_header_and_restores_signal(monkeypatch, clean_transport):
    import signal

    probe = _HTTPSProbe(monkeypatch)
    original_connection = probe.connection

    def connection(*args, **kwargs):
        conn = original_connection(*args, **kwargs)

        def stalled():
            # Deliver the exact installed timer handler without a 30-second wait.
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)

        conn.getresponse = stalled
        return conn

    previous = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(authority.http.client, "HTTPSConnection", connection)
    with pytest.raises(authority.AuthorityUnavailable, match="deadline"):
        authority.trusted_time()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    assert probe.closed


def test_normal_two_file_gh_configuration_is_not_a_cycle(monkeypatch, clean_transport):
    directory = Path(os.environ["XDG_CONFIG_HOME"]) / "gh"
    directory.mkdir(parents=True)
    (directory / "config.yml").write_text("git_protocol: https\n")
    (directory / "hosts.yml").write_text("github.com: {user: synthetic}\n")
    probe = _HTTPSProbe(monkeypatch)
    assert authority.trusted_time()["tls_verified"] is True
    assert probe.closed


def _anonymous_probe(monkeypatch):
    probe = _HTTPSProbe(monkeypatch)
    probe.anonymous = True
    monkeypatch.setattr(authority, "_token", lambda: None)
    return probe


def test_anonymous_capture_checks_public_repo_once_and_records_all_requests(
    monkeypatch, clean_transport
):
    probe = _anonymous_probe(monkeypatch)
    observed = authority.trusted_time()
    assert observed["auth_mode"] == "anonymous"
    assert observed["repository_public"] is True
    assert observed["request_count"] == 2
    assert observed["request_budget"] == authority.ANONYMOUS_REQUEST_BUDGET
    assert observed["rate_limit_remaining"] == 59
    assert observed["tls_verified"] is True
    assert probe.calls[1] == ("GET", "/" + authority.API)
    assert probe.calls[3] == ("GET", "/" + authority.API + "/git/ref/heads/main")
    # Public status and quota are fresh for every capture, never process-cached.
    assert authority.trusted_time()["request_count"] == 2
    assert len(probe.calls) == 8


@pytest.mark.parametrize(
    "repo",
    [
        {"full_name": authority.REPOSITORY, "private": True, "visibility": "private"},
        {"full_name": authority.REPOSITORY, "private": False},
        {"full_name": "foreign/repo", "private": False, "visibility": "public"},
        {"full_name": authority.REPOSITORY, "private": 0, "visibility": "public"},
    ],
)
def test_anonymous_private_or_unverifiable_repository_is_unavailable(
    monkeypatch, clean_transport, repo
):
    probe = _anonymous_probe(monkeypatch)
    probe.repo = repo
    with pytest.raises(authority.AuthorityUnavailable, match="public repository") as error:
        authority.trusted_time()
    assert len(probe.calls) == 2
    assert error.value.capture["request_count"] == 1
    assert error.value.capture["auth_mode"] == "anonymous"


@pytest.mark.parametrize("anonymous", [False, True])
@pytest.mark.parametrize("status", [200, 403, 429])
def test_rate_limit_exhaustion_is_unavailable_without_further_requests(
    monkeypatch, clean_transport, anonymous, status
):
    probe = _anonymous_probe(monkeypatch) if anonymous else _HTTPSProbe(monkeypatch)
    probe.status = status
    probe.headers[-1] = ("X-RateLimit-Remaining", "0")
    with pytest.raises(authority.AuthorityUnavailable, match="rate limit exhausted") as error:
        authority.trusted_time()
    assert error.value.capture["request_count"] == 1
    assert error.value.capture["rate_limit_remaining"] == 0
    assert len(probe.calls) == 2
    assert probe.closed


@pytest.mark.parametrize("quota", [None, "-1", "nonsense", "1.5", "١"])
def test_anonymous_missing_or_invalid_quota_is_unavailable(monkeypatch, clean_transport, quota):
    probe = _anonymous_probe(monkeypatch)
    probe.headers = probe.headers[:-1]
    if quota is not None:
        probe.headers.append(("X-RateLimit-Remaining", quota))
    with pytest.raises(authority.AuthorityUnavailable, match="rate limit"):
        authority.trusted_time()
    assert len(probe.calls) == 2


def test_anonymous_budget_bounds_entire_nested_capture(monkeypatch, clean_transport):
    probe = _anonymous_probe(monkeypatch)
    monkeypatch.setattr(authority, "ANONYMOUS_REQUEST_BUDGET", 3)
    with pytest.raises(authority.AuthorityUnavailable, match="request budget") as error:
        with authority._capture():
            authority.trusted_time()  # metadata + ref
            authority.trusted_time()  # ref only, public check shared within capture
            authority.trusted_time()  # refused before connection
    assert error.value.capture["request_count"] == 3
    assert error.value.capture["request_budget"] == 3
    assert len(probe.calls) == 6


def test_token_capture_keeps_one_request_and_existing_authorization(monkeypatch, clean_transport):
    probe = _HTTPSProbe(monkeypatch)
    observed = authority.trusted_time()
    assert observed["auth_mode"] == "token"
    assert observed["request_count"] == 1
    assert observed["request_budget"] == authority.TOKEN_REQUEST_BUDGET
    assert observed["repository_public"] is None
    assert len(probe.calls) == 2


@pytest.mark.parametrize(
    "failure", [None, FileNotFoundError(), subprocess.TimeoutExpired("gh", 10)]
)
def test_missing_gh_credential_falls_back_once_without_output(
    monkeypatch, clean_transport, capsys, failure
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        if failure is not None:
            raise failure
        return SimpleNamespace(returncode=0, stdout=" \n", stderr="sensitive")

    monkeypatch.setattr(authority.subprocess, "run", run)
    assert authority._token() is None
    assert authority._token() is None
    assert len(calls) == 1
    assert capsys.readouterr() == ("", "")


def test_anonymous_foreign_endpoint_refused_before_network(monkeypatch, clean_transport):
    probe = _anonymous_probe(monkeypatch)
    with pytest.raises(authority.AuthorityUnavailable, match="foreign"):
        authority._request("https://foreign.invalid/")
    assert not probe.calls


@pytest.mark.parametrize("fault", ["proxy", "tls", "peer", "redirect", "clock"])
def test_anonymous_transport_protections_remain_intact(monkeypatch, clean_transport, fault):
    probe = _anonymous_probe(monkeypatch)
    if fault == "proxy":
        monkeypatch.setenv("HTTPS_PROXY", "https://foreign.invalid")
    elif fault == "tls":
        probe.failure = ssl.SSLCertVerificationError("certificate verify failed")
    elif fault == "peer":
        probe.bad_peer = True
    elif fault == "redirect":
        probe.status = 302
        probe.headers.append(("Location", "https://foreign.invalid"))
    else:
        monkeypatch.setattr(authority.time, "time", lambda: 1121.0)
    with pytest.raises(authority.AuthorityUnavailable):
        authority.trusted_time()
    assert len(probe.calls) <= 2


def test_read_only_publication_check_uses_published_ledger_and_both_sidecars(tmp_path, monkeypatch):
    root, before, commit, _ = _result_fixture(tmp_path, monkeypatch)
    _stub_activity(monkeypatch, commit, [_event(commit, before, "1970-01-01T00:01:00Z")], epoch=100)
    snapshot = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = authority.check_publication(authority.f2.RESULT_LEDGER)
    assert result["status"] == "available"
    assert result["dry_run"] is True
    assert result["commit"] == commit
    assert result["activity"]["after"] == commit
    assert snapshot == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_completed_http_body_does_not_reuse_closed_peer(monkeypatch, clean_transport):
    probe = _anonymous_probe(monkeypatch)
    original = probe.connection

    def connection(*args, **kwargs):
        conn = original(*args, **kwargs)
        getresponse = conn.getresponse
        closed = False

        def settimeout(seconds):
            if closed:
                raise OSError("Bad file descriptor")

        conn.sock.settimeout = settimeout

        def response():
            wire = getresponse()
            read = wire.read1

            def read1(size):
                nonlocal closed
                payload = read(size)
                closed = True
                return payload

            wire.read1 = read1
            wire.isclosed = lambda: closed
            return wire

        conn.getresponse = response
        return conn

    monkeypatch.setattr(authority.http.client, "HTTPSConnection", connection)
    assert authority.trusted_time()["request_count"] == 2


def test_serialized_publication_observation_includes_entire_anonymous_capture(
    tmp_path, monkeypatch, clean_transport
):
    _anonymous_probe(monkeypatch)
    binding = {"path": str(tmp_path / "synthetic-receipt"), "publication_ref": "a" * 40}

    def identity(value):
        authority._request(f"{authority.API}/git/ref/heads/main")
        authority._request(f"{authority.API}/git/ref/heads/main")
        return {"commit": value["publication_ref"]}

    def publication(commit):
        _, _, observation = authority._request(f"{authority.API}/git/ref/heads/main")
        authority._request(f"{authority.API}/git/ref/heads/main")
        return {"id": "synthetic"}, observation

    monkeypatch.setattr(authority, "_identity", identity)
    monkeypatch.setattr(authority, "first_publication", publication)
    authority.capture_publication(binding)
    receipt = json.loads(authority.evidence_path(binding).read_text())
    observation = receipt["captured_time_observation"]
    assert observation["request_count"] == 5  # one public check plus four ordinary GETs
    assert observation["request_budget"] == authority.ANONYMOUS_REQUEST_BUDGET
    assert observation["auth_mode"] == "anonymous"
    assert observation["repository_public"] is True


def test_credential_decode_failure_is_sanitized_before_network(
    monkeypatch, clean_transport, capsys
):
    token = authority._token
    probe = _HTTPSProbe(monkeypatch)
    monkeypatch.setattr(authority, "_token", token)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def invalid(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"sensitive\xff", 9, 10, "invalid")

    monkeypatch.setattr(authority.subprocess, "run", invalid)
    with pytest.raises(authority.AuthorityUnavailable, match="configuration") as error:
        authority.trusted_time()
    assert "sensitive" not in str(error.value)
    assert error.value.capture["request_count"] == 0
    assert not probe.calls
    assert capsys.readouterr() == ("", "")
