"""Lane-local permanent-probe hardening without changing transport bytes."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from research.colab_cli import cli_guard, policy
from research.rpc_qualification.probe_policy import hardened_probe_policy


def _hardened():
    return hardened_probe_policy()


def test_permanent_terminal_probe_does_not_retry_traceback_timeout() -> None:
    error = cli_guard.GuardError(
        "earlier context timed out\nGuardError: OAuth credentials rejected"
    )
    with _hardened():
        assert not policy.recovery_probe_retryable(error)


def test_only_terminal_helper_exhaustion_extends_transient_policy() -> None:
    with _hardened():
        assert policy.recovery_probe_retryable(
            cli_guard.GuardError("read helper operation budget exhausted")
        )
        assert policy.recovery_probe_retryable(cli_guard.GuardError("HTTP 503"))
        assert not policy.recovery_probe_retryable(
            cli_guard.GuardError("read helper operation budget exhausted\nGuardError: quota denied")
        )


@pytest.mark.parametrize("raises", [False, True])
def test_lane_policy_restores_after_normal_or_exceptional_exit(raises: bool) -> None:
    original = policy.recovery_probe_retryable
    with contextlib.suppress(RuntimeError):
        with _hardened():
            assert policy.recovery_probe_retryable is not original
            if raises:
                raise RuntimeError("test exceptional cleanup")
    assert policy.recovery_probe_retryable is original


def test_profile_broker_installs_scoped_policy_at_real_supervisor_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.rpc_qualification import broker

    original = policy.recovery_probe_retryable
    manifest = {"output_dir": str(tmp_path)}
    monkeypatch.setattr(broker, "validate", lambda *_args: manifest)

    def run(_self, _manifest, _retained):
        error = cli_guard.GuardError("context timed out\nGuardError: OAuth credentials rejected")
        assert policy.recovery_probe_retryable is not original
        assert not policy.recovery_probe_retryable(error)
        raise RuntimeError("exercise broker exception restoration")

    monkeypatch.setattr(broker.supervisor.Broker, "run", run)
    runner = broker.QualificationBroker.__new__(broker.QualificationBroker)
    runner.observer = None
    with pytest.raises(RuntimeError, match="exception restoration"):
        runner.run(manifest)
    assert policy.recovery_probe_retryable is original
