"""Short native Darwin maintenance checks, never concurrent factory tests.

Only the test's own children are started or reaped. No factory, gameplay,
checkpoint, held-out outcome or Mac lease is opened by this module.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import pytest

from research.compute.factory_companion import POLICY_SHA256, FactoryCompanionSupervisor
from research.compute.factory_companion_native import usage

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="native Darwin evidence")


def test_native_start_identity_cpu_units_and_rss() -> None:
    before = usage(os.getpid())
    started = time.process_time()
    while time.process_time() - started < 0.015:
        pass
    after = usage(os.getpid())
    consumed = time.process_time() - started
    delta = after.self_cpu_seconds - before.self_cpu_seconds
    assert after.start_tick == before.start_tick > 0
    assert after.pid == before.pid == os.getpid()
    assert 0.01 <= delta <= consumed + 0.015
    assert after.resident_bytes > 0


def test_native_short_reaped_child_is_retained_between_samples() -> None:
    """A child entirely between monitor polls must survive in native counters."""
    before = usage(os.getpid())
    pid = os.fork()
    if pid == 0:
        try:
            started = time.process_time()
            while time.process_time() - started < 0.025:
                pass
        finally:
            os._exit(0)
    waited, status, rusage = os.wait4(pid, 0)
    after = usage(os.getpid())
    assert waited == pid and os.waitstatus_to_exitcode(status) == 0
    child_cpu = rusage.ru_utime + rusage.ru_stime
    assert child_cpu >= 0.02
    delta = after.child_cpu_seconds - before.child_cpu_seconds
    assert abs(delta - child_cpu) < 0.005
    again = usage(os.getpid())
    assert again.child_cpu_seconds == after.child_cpu_seconds
    # The parent self counter and child counter are disjoint, not inclusive.
    assert after.self_cpu_seconds - before.self_cpu_seconds < child_cpu


def test_native_bounded_light_command_has_gate_identity_and_receipt(tmp_path) -> None:
    """The supervisor's own short helper proves gate/nice/wait4 receipt flow."""
    subject = FactoryCompanionSupervisor(tmp_path)
    subject.claim_monitor()
    subject.update_factory_state(
        identity="native-maintenance",
        protected=False,
        streams={"maintenance": {"baseline_ready": True, "fresh_window_ready": True}},
    )
    assert (
        subject.launch(
            "light",
            policy_sha=POLICY_SHA256,
            lease_identity="native-maintenance",
            worktree=str(tmp_path),
            argv=[sys.executable, "-c", "import time; time.sleep(0.2)"],
            approved={
                "argv_sha256": hashlib.sha256(
                    "\0".join([sys.executable, "-c", "import time; time.sleep(0.2)"]).encode()
                ).hexdigest()
            },
        )
        == 0
    )
    state = subject._read()
    record = next(iter(state["registrations"].values()))
    assert record["priority_verified"] is True
    assert record["wait4_reaped"] is True
    assert record["status"] == "completed"
    receipt = subject.finalize_receipt(
        attempt_identity="native-maintenance", policy_sha=POLICY_SHA256
    )
    assert receipt["sha256"] in receipt["path"]
