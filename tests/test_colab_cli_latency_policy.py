"""Deadline and clock-skew regressions from the live acceptance failures."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.colab_cli import supervisor
from research.colab_cli.acceptance_faults import AcceptanceObserver


def test_watchdog_collects_during_expiry_interval_and_recovers_status_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow final renewal must not skip the short remote checkpoint window."""
    now = [0.0]
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    reads = []
    calls = []

    class CLI:
        session = "owned"
        accounting = SimpleNamespace(endpoint="owned-endpoint")

        def stop_lease(self, *, wait: bool = False) -> None:
            assert wait
            calls.append("pause")

        def poll_rpc(self, code: str) -> bool:
            assert "renew=True" in code
            calls.append("renew")
            now[0] += 30  # Acknowledgement latency; actual expiry is t=210.
            return True

        def call(self, *args: str, timeout: float) -> str:
            assert calls == ["pause", "renew"]
            assert 0 < timeout <= 120
            reads.append(now[0])
            now[0] += 6
            if now[0] < 210:
                raise supervisor.Refused("404 not found")
            if now[0] > 225:
                raise AssertionError("collector skipped the runtime's checkpoint grace")
            Path(args[-1]).write_bytes(
                (digest + "\n").encode() if args[-2].endswith("sha256") else payload
            )
            return "ok"

        def sessions(self, *, timeout: float) -> dict[str, str]:
            assert timeout <= 120
            if "status-reset" not in calls:
                calls.append("status-reset")
                raise supervisor.Refused("HTTP 503 temporarily unavailable")
            return {}

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    observer = AcceptanceObserver("fake")
    monkeypatch.setattr(observer, "_state", lambda context: {})
    observer._watchdog(
        {
            "cli": CLI(),
            "manifest": {"lease_seconds": 180},
            "root": "/content/snake-cli/g",
            "generation": "g",
            "output": tmp_path,
        }
    )
    evidence = observer.records["mac-loss-watchdog-unassign"]
    assert evidence["assignment_disappeared_within_bound"] is True
    assert evidence["lease_expiry_uncertainty_seconds"] == 30
    assert evidence["observed_after_lease_seconds"] < 300
    assert reads[0] == 30  # Poll before expiry, never sleep past the remote grace.
    assert (tmp_path / "watchdog-checkpoint.pkl").read_bytes() == payload
