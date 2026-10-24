"""Hermetic adversaries for the I2 GitHub publication/time authority."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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


def _stub_activity(
    monkeypatch: pytest.MonkeyPatch, head: str, rows: list[dict], *, epoch: float
) -> None:
    def request(endpoint: str):
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


def _http_response(*, date: str | None, payload: object = None, status: int = 200) -> str:
    headers = [f"HTTP/1.1 {status} OK", "X-GitHub-Request-Id: synthetic"]
    if date is not None:
        headers.append(f"Date: {date}")
    return "\r\n".join(headers) + "\r\n\r\n" + json.dumps({} if payload is None else payload)


def test_request_accepts_fresh_https_date_and_refuses_clock_or_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    date = "Thu, 01 Jan 1970 00:16:40 GMT"

    def run_good(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=_http_response(date=date))

    monkeypatch.setattr(authority, "_gh_environment", lambda: {})
    monkeypatch.setattr(authority.subprocess, "run", run_good)
    monkeypatch.setattr(authority.time, "time", lambda: 1000.0)
    ticks = iter((10.0, 10.2))
    monkeypatch.setattr(authority.time, "monotonic", lambda: next(ticks))
    _, _, observed = authority._request(f"{authority.API}/git/ref/heads/main")
    assert observed["epoch"] == 1000.0

    for response, wall, ticks in [
        (_http_response(date=None), 1000.0, (10.0, 10.2)),
        (_http_response(date="not a date"), 1000.0, (10.0, 10.2)),
        (_http_response(date=date), 1121.0, (10.0, 10.2)),
        (_http_response(date=date), 879.0, (10.0, 10.2)),
        (_http_response(date=date), 1000.0, (10.0, 41.0)),
    ]:
        monkeypatch.setattr(
            authority.subprocess,
            "run",
            lambda *_a, r=response, **_k: SimpleNamespace(returncode=0, stdout=r),
        )
        monkeypatch.setattr(authority.time, "time", lambda w=wall: w)
        monotonic = iter(ticks)
        monkeypatch.setattr(authority.time, "monotonic", lambda m=monotonic: next(m))
        with pytest.raises(authority.AuthorityUnavailable):
            authority._request(f"{authority.API}/git/ref/heads/main")

    monkeypatch.setattr(
        authority.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="")
    )
    monkeypatch.setattr(authority.time, "monotonic", lambda: 50.0)
    with pytest.raises(authority.AuthorityUnavailable):
        authority._request(f"{authority.API}/git/ref/heads/main")
