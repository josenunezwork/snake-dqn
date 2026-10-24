"""Exercise the last local refusal gate without invoking the real Colab CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import small_bench_cli as cli
from tools import small_bench_operator as op


@pytest.fixture
def allocation(tmp_path: Path, monkeypatch):
    account = tmp_path / "account"
    account.mkdir()
    manifest = tmp_path / "main.json"
    output = tmp_path / "main-output"
    manifest.write_text(
        json.dumps(
            {"job_id": "main", "output_dir": str(output), "budget": {"other_assignments": {}}}
        )
    )
    config = {
        "account_root": str(account),
        "manifest": op.binding(manifest),
        "job_id": "main",
        "register": str(tmp_path / "register"),
        "reservation_id": "rid",
        "reservation_lock": str(tmp_path / "ops" / "sprint-reservations.lock"),
    }
    reference = {"path": str(tmp_path / "config"), "sha256": "a" * 64}
    admission = {
        "capture": {"started_utc": datetime.now(timezone.utc).isoformat()},
        "register_row_hash": "r",
        "config": reference,
        "operational_fingerprint": {},
        "live_checks": {"balance": 100},
        "row": {"package": {"operator": "test"}},
    }
    pending = account / "pending-session.json"
    pending.write_text(json.dumps({"session": "session", "output": str(output)}))
    monkeypatch.setenv("SNAKE_COLAB_SESSION", "session")
    monkeypatch.setattr(op, "configuration", lambda p: config)
    monkeypatch.setattr(op, "current_admission", lambda c: admission)
    monkeypatch.setattr(op, "binding", lambda p: reference)
    monkeypatch.setattr(op.register, "read", lambda p: [{"row_hash": "r"}])
    monkeypatch.setattr(op.register, "reserve", lambda *args: {})
    monkeypatch.setattr(op, "fingerprint", lambda c: {str(pending): "p"})
    return config, admission, pending


def test_guard_allows_only_qualified_fresh_owned_request(allocation) -> None:
    cli.guard(Path("/config"), ["--auth", "oauth2", "new", "-s", "session"])


@pytest.mark.parametrize("change", ["stale", "wrong_revision", "wrong_output", "missing_pending"])
def test_guard_refuses_wrong_or_expired_request(allocation, monkeypatch, change: str) -> None:
    config, admission, pending = allocation
    if change == "stale":
        admission["capture"]["started_utc"] = (
            datetime.now(timezone.utc) - timedelta(minutes=11)
        ).isoformat()
    elif change == "wrong_revision":
        monkeypatch.setattr(op.register, "read", lambda p: [{"row_hash": "changed"}])
    elif change == "wrong_output":
        pending.write_text(json.dumps({"session": "session", "output": "/foreign"}))
    else:
        pending.unlink()
    with pytest.raises(ValueError):
        cli.guard(Path("/config"), ["--auth", "oauth2", "new"])


def test_guard_refuses_intervening_liability(allocation, monkeypatch, tmp_path: Path) -> None:
    _, _, pending = allocation
    foreign = tmp_path / "other.json"
    foreign.write_text(json.dumps({"job_id": "other"}))
    monkeypatch.setattr(op, "fingerprint", lambda c: {str(pending): "p", str(foreign): "f"})
    with pytest.raises(ValueError, match="intervening"):
        cli.guard(Path("/config"), ["--auth", "oauth2", "new"])


@pytest.mark.parametrize("valid", [True, False])
def test_guard_exact_existing_allocation_retry(allocation, monkeypatch, valid: bool) -> None:
    config, _, pending = allocation
    original = json.loads(op.bound(config["manifest"]).read_text())
    output = Path(original["output_dir"])
    output.mkdir()
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "job_id": "main",
                "status": "INCOMPLETE",
                "failure_class": "ACCOUNTING",
                "error": "AccountingError('assignment endpoints changed during allocation')",
                "cleanup_error": None,
                "stop_verified": valid,
            }
        )
    )
    logs = Path(config["reservation_lock"]).parent / "logs"
    logs.mkdir(parents=True)
    (logs / "main.log").write_text("EXIT=1\n")
    pending.write_text(json.dumps({"session": "session", "output": str(output) + "-retry-1"}))
    if valid:
        cli.guard(Path("/config"), ["--auth", "oauth2", "new"])
    else:
        with pytest.raises(ValueError, match="retry"):
            cli.guard(Path("/config"), ["--auth", "oauth2", "new"])


@pytest.mark.parametrize("command", ["stop", "sessions", "usage", "upload", "download", "exec"])
def test_cleanup_and_existing_transport_are_not_gated(command: str) -> None:
    cli.guard(Path("/absent"), ["--auth", "oauth2", command])
