"""Full-grace real runtime stop/admission proof, explicitly qualified separately.

Keep the new 15-second process-group integration outside the historical nested
180-second preflight suite. No test assertion, timeout or stop grace is reduced.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.sandbox_a_resident.ct_resume import CTResumeRefused, verify_mechanical_stop
from research.sandbox_a_resident.ct_run import SIGUSR1_EXIT_CODE
from research.sandbox_a_resident.ct_task import CTControl
from research.sandbox_a_resident.ct_units import allocation
from research.sandbox_a_resident.test_ct_runner_and_resume import ROOT, _save_tiny_checkpoints


def test_runtime_stop_child_mid_progress_is_admitted_mechanical_138(tmp_path: Path) -> None:
    """Use the unmodified runtime group-stop path during a real C/T callback."""
    from research.colab_cli.runtime import _stop_child

    b8, frp3, champion = _save_tiny_checkpoints(tmp_path)
    control = CTControl(
        source_commit="f" * 40,
        worlds=6,
        collection_ticks=200,
        max_body_capacity=128,
        scoring_chunks=2,
        tiny=True,
    )
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control.to_dict()))
    output = tmp_path / "output"
    ready = tmp_path / "mid-progress.ready"
    # Synchronize the signal inside a real progress callback, after work has
    # completed but before the guard is rearmed. Without the fix, this exact
    # window deterministically writes fault.json and exits 1.
    script = r"""
import sys, time
from pathlib import Path
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
from research.sandbox_a_resident.ct_run import main
ready = Path(sys.argv.pop(1))
original = StallWatchdog.progress
def progress(self, stage, counters=None):
    if stage == 'collection-progress' and not ready.exists():
        ready.write_text('ready\n')
        time.sleep(1)
    return original(self, stage, counters)
StallWatchdog.progress = progress
raise SystemExit(main())
"""
    env = {**os.environ, "SNAKE_DQN_DEVICE": "cpu"}
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    log = tmp_path / "child.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                "nice",
                "-n",
                "15",
                sys.executable,
                "-c",
                script,
                str(ready),
                "--checkpoint",
                str(b8),
                "--frp3",
                str(frp3),
                "--champion",
                str(champion),
                "--control",
                str(control_path),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--progress-interval-seconds",
                "0",
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), log.read_text()
            signaled = time.monotonic()
            _stop_child(process, grace_seconds=15.0)
            process.wait(timeout=5)
            assert process.returncode == SIGUSR1_EXIT_CODE == 138, log.read_text()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    assert not (output / "fault.json").exists()
    diagnostics = output / "stall-watchdog"
    assert (diagnostics / "faulthandler-stacks.log").read_bytes() == b""
    last = json.loads((diagnostics / "stall-watchdog.json").read_text())
    assert last["stage"] == "collection-progress"
    assert last["progress_monotonic_seconds"] > signaled
    marker = json.loads((output / "sigusr1-marker.json").read_text())
    assert marker["within_15s_sla"] is True
    assert marker["allocation"] == allocation(control.to_dict())
    manifest = {"job_id": "ct-runtime-group-stop", "commit": "f" * 40, "output_dir": str(output)}
    receipt = {
        **manifest,
        "manifest_sha256": "e" * 64,
        "generation": "ct-runtime-group-stop-generation",
        "status": "INCOMPLETE",
        "failure_class": "CHILD_NONZERO_EXIT",
        "error": "remote child exited 138: timeout",
        "stop_verified": True,
        "cleanup_error": None,
        "terminal": {
            "job_id": manifest["job_id"],
            "generation": "ct-runtime-group-stop-generation",
            "status": "failed",
            "reason": "timeout",
            "returncode": process.returncode,
        },
    }
    # This is the production continuation admission function, scanning the real
    # child output tree, not just a return-code assertion or a mocked scanner.
    verify_mechanical_stop(receipt, manifest, "e" * 64, export_root=output)
    (output / "fault.json").write_text('{"scientific_fault":true}\n')
    with pytest.raises(CTResumeRefused, match="fault diagnostic present"):
        verify_mechanical_stop(receipt, manifest, "e" * 64, export_root=output)
