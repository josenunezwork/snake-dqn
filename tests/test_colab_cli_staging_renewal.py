"""Offline staging-renewal transport regression against the fake Colab CLI."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import acceptance, release, supervisor, transfers
from tests import test_colab_cli_fault_properties as harness

_isolate_account = harness._isolate_account
pinned_repo = harness.pinned_repo


def test_staging_renewal_uses_contents_without_a_kernel_renewal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pinned_repo: Path,
    _isolate_account: harness._TransportClock,
) -> None:
    """A due renewal in staging must fence Contents without queueing a kernel RPC."""
    binary = harness._fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 71,
            "max_latency_seconds": 60,
            "schedule": [
                {
                    "call": "upload",
                    "session_contains": "-l4-",
                    "contains": "bootstrap.sh",
                    "fault": "latency",
                    "latency_seconds": 31,
                }
            ],
        },
    )
    plan = harness._pinned_plan(tmp_path, pinned_repo, profile="final-only")
    monkeypatch.setattr(transfers, "UPLOAD_WORKERS", 1)

    renewal_calls: list[tuple[bool, bool, str]] = []
    put_calls: list[tuple[bool, str]] = []
    kernel_renewals: list[bool] = []
    credential_checks: list[bool] = []
    retained_calls: list[dict[str, Any] | None] = []
    original_renew = supervisor.CLI.renew_runtime_lease
    original_put = supervisor.CLI._put_lease
    original_poll = supervisor.CLI.poll_rpc
    original_due = release.credential_renewal_due
    original_retained = supervisor.Broker.renew_retained

    def has_staging_deadline(cli: supervisor.CLI) -> bool:
        return getattr(cli._control_context, "deadline", math.inf) != math.inf

    def renew_runtime_lease(
        cli: supervisor.CLI, root: str, generation: str, *, force: bool = False
    ) -> Any:
        renewal_calls.append((has_staging_deadline(cli), force, threading.current_thread().name))
        return original_renew(cli, root, generation, force=force)

    def put_lease(cli: supervisor.CLI, root: str, generation: str, timeout: float) -> None:
        put_calls.append((has_staging_deadline(cli), threading.current_thread().name))
        return original_put(cli, root, generation, timeout)

    def poll_rpc(cli: supervisor.CLI, code: str, *args: Any, **kwargs: Any) -> Any:
        is_renewal = "renew_lease(" in code or ("read_state(" in code and "renew=True" in code)
        if is_renewal:
            kernel_renewals.append(has_staging_deadline(cli))
        return original_poll(cli, code, *args, **kwargs)

    def credential_due(*args: Any, **kwargs: Any) -> bool:
        credential_checks.append(True)
        return original_due(*args, **kwargs)

    def renew_retained(cli: supervisor.CLI, retained: dict[str, Any] | None) -> None:
        retained_calls.append(retained)
        return original_retained(cli, retained)

    monkeypatch.setattr(supervisor.CLI, "renew_runtime_lease", renew_runtime_lease)
    monkeypatch.setattr(supervisor.CLI, "_put_lease", put_lease)
    monkeypatch.setattr(supervisor.CLI, "poll_rpc", poll_rpc)
    monkeypatch.setattr(release, "credential_renewal_due", credential_due)
    monkeypatch.setattr(supervisor.Broker, "renew_retained", staticmethod(renew_retained))

    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab=binary,
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python=str(harness.HELPER),
    )

    assert rc == 0
    assert any(
        finite and not force and thread == "MainThread" for finite, force, thread in renewal_calls
    )
    assert any(finite and thread == "MainThread" for finite, thread in put_calls)
    assert not any(kernel_renewals)
    assert credential_checks
    assert retained_calls
    events = [json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()]
    assert any(event.get("fault", {}).get("virtual_latency_seconds") == 31 for event in events)
    harness._assert_safe(tmp_path, plan)
