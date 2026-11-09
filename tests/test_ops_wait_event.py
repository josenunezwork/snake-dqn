"""Regression tests for tools/ops/wait-event.sh expiry vs active job timeouts."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WAIT_EVENT_SH = REPO_ROOT / "tools" / "ops" / "wait-event.sh"


def _run_wait_event(ops_dir: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OPS_DIR"] = str(ops_dir)
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(WAIT_EVENT_SH)],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        check=False,
    )


def test_wait_event_expiry_outlasts_active_job_timeout(tmp_path: Path) -> None:
    """Watcher expiry must outlast an active job's timeout instead of expiring early."""
    ops = tmp_path / "ops"
    manifests = ops / "manifests"
    logs = ops / "logs"
    manifests.mkdir(parents=True)
    logs.mkdir(parents=True)

    job_name = "bank-v2-screen-main"
    manifest_path = manifests / f"{job_name}.json"
    log_path = logs / f"{job_name}.log"
    manifest_path.write_text(
        json.dumps(
            {
                "job_id": job_name,
                "timeout_minutes": 0.04,  # 2.4s > 1s base watcher window
            }
        )
    )
    log_path.write_text("starting job\n")
    (ops / "watch-logs.txt").write_text(f"{log_path}\n")

    expected_event = (
        f"2026-11-05T19:24:01-0800 JOB_ENDED {job_name} EXIT=0 status=PASS stop_verified=True"
    )

    def emit_after_base_expiry() -> None:
        # Sleep past the 1s base watcher expiry, then emit JOB_ENDED before job timeout + grace.
        time.sleep(1.4)
        log_path.write_text("starting job\nEXIT=0\n")
        with (ops / "events.log").open("a", encoding="utf-8") as fh:
            fh.write(expected_event + "\n")
        state_dir = ops / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"{job_name}.done").write_text("done\n")

    worker = threading.Thread(target=emit_after_base_expiry, daemon=True)
    worker.start()
    res = _run_wait_event(
        ops,
        WAIT_EVENT_SECONDS="1",
        WAIT_EVENT_GRACE_SECONDS="2",
        WAIT_EVENT_POLL_SECONDS="0.2",
    )
    worker.join(timeout=5)

    assert res.returncode == 0
    assert expected_event in res.stdout
    assert "NO EVENT" not in res.stdout
    assert (ops / "events.cursor.operator").read_text().strip() == "1"


def test_wait_event_extends_for_exited_watched_log_awaiting_watchd(tmp_path: Path) -> None:
    """An exited watched log without state/<name>.done keeps the watcher alive through grace."""
    ops = tmp_path / "ops"
    logs = ops / "logs"
    logs.mkdir(parents=True)

    job_name = "just-exited-job"
    log_path = logs / f"{job_name}.log"
    log_path.write_text("EXIT=0\n")
    (ops / "watch-logs.txt").write_text(f"{log_path}\n")

    expected_event = f"2026-11-05T20:00:00-0800 JOB_ENDED {job_name} EXIT=0"

    def watchd_late_emit() -> None:
        time.sleep(1.3)
        with (ops / "events.log").open("a", encoding="utf-8") as fh:
            fh.write(expected_event + "\n")
        state_dir = ops / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"{job_name}.done").write_text("done\n")

    worker = threading.Thread(target=watchd_late_emit, daemon=True)
    worker.start()
    res = _run_wait_event(
        ops,
        WAIT_EVENT_SECONDS="1",
        WAIT_EVENT_GRACE_SECONDS="2",
        WAIT_EVENT_POLL_SECONDS="0.2",
    )
    worker.join(timeout=5)

    assert res.returncode == 0
    assert expected_event in res.stdout
    assert "NO EVENT" not in res.stdout


def test_wait_event_ignores_completed_done_manifests(tmp_path: Path) -> None:
    """Already-completed (.done) jobs must not extend the watcher deadline."""
    ops = tmp_path / "ops"
    manifests = ops / "manifests"
    state = ops / "state"
    manifests.mkdir(parents=True)
    state.mkdir(parents=True)

    job_name = "historical-done-job"
    (manifests / f"{job_name}.json").write_text(
        json.dumps({"job_id": job_name, "timeout_minutes": 932})
    )
    (state / f"{job_name}.done").write_text("done\n")

    t0 = time.monotonic()
    res = _run_wait_event(
        ops,
        WAIT_EVENT_SECONDS="1",
        WAIT_EVENT_GRACE_SECONDS="5",
        WAIT_EVENT_POLL_SECONDS="0.2",
    )
    elapsed = time.monotonic() - t0

    assert res.returncode == 0
    assert "NO EVENT in 3 h" in res.stdout
    assert elapsed < 3.5
