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


def test_wait_event_extends_for_live_json_only_manifest_with_watched_log(tmp_path: Path) -> None:
    """Finding 1.1: .live.json manifest + watched log must not mangle job name via p.stem[:-4]."""
    ops = tmp_path / "ops"
    manifests = ops / "manifests"
    logs = ops / "logs"
    manifests.mkdir(parents=True)
    logs.mkdir(parents=True)

    job_name = "bank-v2-screen-main"
    live_manifest = manifests / f"{job_name}.live.json"
    log_path = logs / f"{job_name}.log"
    live_manifest.write_text(
        json.dumps(
            {
                "job_id": job_name,
                "timeout_minutes": 0.04,  # 2.4s > 1s base watcher window
            }
        )
    )
    log_path.write_text("starting live job\n")
    (ops / "watch-logs.txt").write_text(f"{log_path}\n")

    expected_event = f"2026-11-05T20:10:00-0800 JOB_ENDED {job_name} EXIT=0"

    def emit_after_base_expiry() -> None:
        time.sleep(1.4)
        log_path.write_text("starting live job\nEXIT=0\n")
        with (ops / "events.log").open("a", encoding="utf-8") as fh:
            fh.write(expected_event + "\n")

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


def test_wait_event_completed_watched_log_respects_done_file(tmp_path: Path) -> None:
    """Finding 1.2 & 1.3: An exited watched log with state/<name>.done must not extend grace."""
    ops = tmp_path / "ops"
    manifests = ops / "manifests"
    logs = ops / "logs"
    state = ops / "state"
    manifests.mkdir(parents=True)
    logs.mkdir(parents=True)
    state.mkdir(parents=True)

    job_name = "just-exited-job"
    (manifests / f"{job_name}.json").write_text(
        json.dumps({"job_id": job_name, "timeout_minutes": 10})
    )
    log_path = logs / f"{job_name}.log"
    log_path.write_text("EXIT=0\n")
    (ops / "watch-logs.txt").write_text(f"{log_path}\n")
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
    assert elapsed < 3.0


def test_wait_event_preprepared_manifest_uses_execution_start_and_ignores_undispatched(
    tmp_path: Path,
) -> None:
    """Finding 2.1: Static manifest prepared earlier uses log/.live.json start; idle ignored."""
    ops = tmp_path / "ops"
    manifests = ops / "manifests"
    logs = ops / "logs"
    manifests.mkdir(parents=True)
    logs.mkdir(parents=True)

    # 1) Un-dispatched manifest with timeout > base wait, no log/.live.json, not in watch-logs.txt.
    undispatched = manifests / "undispatched-queued-job.json"
    undispatched.write_text(
        json.dumps({"job_id": "undispatched-queued-job", "timeout_seconds": 3.5})
    )

    t0 = time.monotonic()
    res_idle = _run_wait_event(
        ops,
        WAIT_EVENT_SECONDS="1",
        WAIT_EVENT_GRACE_SECONDS="2",
        WAIT_EVENT_POLL_SECONDS="0.2",
    )
    assert "NO EVENT in 3 h" in res_idle.stdout
    assert time.monotonic() - t0 < 2.2
    undispatched.unlink()

    # 2) Pre-prepared manifest backdated by 1 hour, but dispatched now (log created now).
    job_name = "preprepared-active-job"
    static_mf = manifests / f"{job_name}.json"
    static_mf.write_text(json.dumps({"job_id": job_name, "timeout_minutes": 0.04}))
    old_ts = time.time() - 3600.0
    os.utime(static_mf, (old_ts, old_ts))

    log_path = logs / f"{job_name}.log"
    log_path.write_text("dispatched now\n")
    (ops / "watch-logs.txt").write_text(f"{log_path}\n")

    expected_event = f"2026-11-05T20:20:00-0800 JOB_ENDED {job_name} EXIT=0"

    def emit_after_base_expiry() -> None:
        time.sleep(1.4)
        log_path.write_text("dispatched now\nEXIT=0\n")
        with (ops / "events.log").open("a", encoding="utf-8") as fh:
            fh.write(expected_event + "\n")

    worker = threading.Thread(target=emit_after_base_expiry, daemon=True)
    worker.start()
    res_active = _run_wait_event(
        ops,
        WAIT_EVENT_SECONDS="1",
        WAIT_EVENT_GRACE_SECONDS="2",
        WAIT_EVENT_POLL_SECONDS="0.2",
    )
    worker.join(timeout=5)

    assert res_active.returncode == 0
    assert expected_event in res_active.stdout
    assert "NO EVENT" not in res_active.stdout


def test_wait_event_deadline_does_not_collapse_mid_wait_when_started_after_job_start(
    tmp_path: Path,
) -> None:
    """Finding 2.2: Starting wait-event.sh after job_start must not inflate or collapse deadline."""
    ops = tmp_path / "ops"
    manifests = ops / "manifests"
    logs = ops / "logs"
    manifests.mkdir(parents=True)
    logs.mkdir(parents=True)

    # 1) Job started 3.0s ago (both manifest and log backdated by 3.0s) with timeout=4.2s, grace=0s.
    # True job_end is ~now + 1.2s (<= start_ts + 2).
    # Unfixed code uses max(job_start, start_ts) + 4.2s => start_ts + 5 while now <= job_end,
    # so with POLL_SECONDS=1.0 it sleeps past 2.5s (or inflates compute_deadline by +3s).
    # Also when an event arrives at 1.4s for a job whose static manifest was created 2.1s ago
    # (so static_mf + 1.5s + 1.0s = now + 0.4s is initially >= now at t=0, inflating deadline to
    # start_ts + 2.5s, then collapsing at t=0.5s to start_ts + 1.0s), the watcher must not collapse.
    job_name = "mid-wait-cliff-job"
    static_mf = manifests / f"{job_name}.json"
    static_mf.write_text(json.dumps({"job_id": job_name, "timeout_seconds": 1.5}))
    backdated = time.time() - 2.1
    os.utime(static_mf, (backdated, backdated))

    log_path = logs / f"{job_name}.log"
    log_path.write_text("running\n")

    expected_event = f"2026-11-05T20:30:00-0800 JOB_ENDED {job_name} EXIT=0"

    def emit_after_cliff() -> None:
        time.sleep(1.4)
        log_path.write_text("running\nEXIT=0\n")
        with (ops / "events.log").open("a", encoding="utf-8") as fh:
            fh.write(expected_event + "\n")

    worker = threading.Thread(target=emit_after_cliff, daemon=True)
    worker.start()
    res = _run_wait_event(
        ops,
        WAIT_EVENT_SECONDS="1",
        WAIT_EVENT_GRACE_SECONDS="1",
        WAIT_EVENT_POLL_SECONDS="0.2",
    )
    worker.join(timeout=5)
    assert res.returncode == 0
    assert expected_event in res.stdout
    assert "NO EVENT" not in res.stdout

    # 2) Direct check: when a job started at job_start = now - 500 with timeout=600, grace=0
    # and wait-event starts at start_ts = now with wait_s = 50, the deadline must be anchored
    # at job_start + 600 (~now + 100), never inflated to start_ts + 600 (~now + 600).
    ops2 = tmp_path / "ops2"
    (ops2 / "manifests").mkdir(parents=True)
    (ops2 / "logs").mkdir(parents=True)
    now_ts = time.time()
    job2 = "started-earlier-job"
    mf2 = ops2 / "manifests" / f"{job2}.live.json"
    log2 = ops2 / "logs" / f"{job2}.log"
    mf2.write_text(json.dumps({"job_id": job2, "timeout_seconds": 600}))
    log2.write_text("running\n")
    job2_start = now_ts - 500.0
    os.utime(mf2, (job2_start, job2_start))
    os.utime(log2, (job2_start, job2_start))

    script = WAIT_EVENT_SH.read_text(encoding="utf-8")
    fn_body = script.split("compute_deadline() {\n", 1)[1].split("\n}\n", 1)[0]
    probe = subprocess.run(
        [
            "bash",
            "-c",
            f'OPS="{ops2}"; start_ts={int(now_ts)}; WAIT_SECONDS=50; GRACE_SECONDS=0; {fn_body}',
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    computed_end = int(probe.stdout.strip())
    assert abs(computed_end - (job2_start + 600.0)) <= 5.0
