"""The reviewed permanent-probe normalization is part of transport policy."""

from __future__ import annotations

import subprocess

import pytest

from research.colab_cli import cli_guard, policy


def test_permanent_terminal_probe_does_not_retry_traceback_timeout() -> None:
    error = cli_guard.GuardError(
        "earlier context timed out\nGuardError: OAuth credentials rejected"
    )
    assert not policy.recovery_probe_retryable(error)


def test_only_terminal_helper_exhaustion_extends_transient_policy() -> None:
    assert policy.recovery_probe_retryable(
        cli_guard.GuardError("read helper operation budget exhausted")
    )
    assert policy.recovery_probe_retryable(cli_guard.GuardError("HTTP 503"))
    assert not policy.recovery_probe_retryable(
        cli_guard.GuardError("read helper operation budget exhausted\nGuardError: quota denied")
    )


@pytest.mark.parametrize(
    "error", [ConnectionError(), TimeoutError(), subprocess.TimeoutExpired([], 1)]
)
def test_typed_transport_errors_remain_retryable(error: BaseException) -> None:
    assert policy.recovery_probe_retryable(error)


def test_broker_does_not_swap_transport_policy() -> None:
    from pathlib import Path

    for path in ("research/rpc_qualification/broker.py", "tools/ops/requalify.py"):
        assert "hardened_probe_policy" not in Path(path).read_text()
