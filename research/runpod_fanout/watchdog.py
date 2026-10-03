#!/usr/bin/env python3
"""Detached local watchdog for one fan-out run: deletes ``rpf-<job>--*`` pods when needed.

Fires (deletes every pod whose name starts with ``--prefix``, i.e. this run's
``rpf-<job_id>--<run>-``, retrying until GET /pods
shows none) when ANY of:

* the hard time ``--fire-epoch`` (job max wall time + grace) has passed;
* the runner process ``--runner-pid`` is gone and ``receipt.json`` has not appeared for
  ``runner_dead_seconds`` (the runner crashed or was killed -9);
* the runner finished (``receipt.json`` exists): one final sweep, then exit.

Started with ``start_new_session=True`` so it survives the runner's terminal and crashes.
It never creates anything and only deletes pods with this job's prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.runpod_fanout import jobspec  # noqa: E402
from research.runpod_fanout.rp_client import RpClient, RunPodError  # noqa: E402


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep(rp: Any, prefix: str, sleep: Callable[[float], None], attempts: int = 40) -> List[str]:
    """Delete every pod named ``prefix*`` until none is listed; returns leftovers."""
    left: List[str] = []
    for _ in range(attempts):
        try:
            pods = [p for p in rp.list_pods() if str(p.get("name", "")).startswith(prefix)]
        except RunPodError as exc:
            log(f"list failed: {exc}")
            sleep(15)
            continue
        left = [p["id"] for p in pods]
        if not pods:
            return []
        for p in pods:
            try:
                rp.delete_pod(p["id"], confirm=True)
                log(f"deleted {p['id']} {p.get('name')}")
            except RunPodError as exc:
                log(f"delete {p['id']} failed: {exc}")
        sleep(15)
    return left


def watch(
    job_id: str,
    fire_epoch: float,
    run_dir: Path,
    runner_pid: int,
    *,
    rp: Any = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    alive: Callable[[int], bool] = pid_alive,
    poll: float = 30.0,
    prefix: str | None = None,
) -> str:
    policy = jobspec.load_policy()
    job_prefix = f"{policy['pod_name_prefix']}{job_id}--"
    prefix = prefix or job_prefix
    if not prefix.startswith(job_prefix):
        raise SystemExit("watchdog prefix must lie inside the job's rpf-<job>-- namespace")
    rp = rp or RpClient()
    dead_since = None
    log(f"armed prefix={prefix} fire_epoch={fire_epoch:.0f} runner_pid={runner_pid}")
    while True:
        now = clock()
        receipt = Path(run_dir) / "receipt.json"
        if receipt.exists():
            reason = "runner finished"
        elif now >= fire_epoch:
            reason = "hard deadline"
        elif not alive(runner_pid):
            dead_since = dead_since or now
            reason = (
                "runner dead" if now - dead_since >= float(policy["runner_dead_seconds"]) else None
            )
        else:
            dead_since, reason = None, None
        if reason:
            log(f"FIRING: {reason}")
            left = sweep(rp, prefix, sleep)
            # A pod from a create still in flight can surface late: sweep again later.
            sleep(180)
            left = sweep(rp, prefix, sleep)
            log(f"done; leftovers={left}")
            (Path(run_dir) / "watchdog_result.json").write_text(
                json.dumps({"reason": reason, "leftovers": left, "utc": time.time()})
            )
            return reason
        sleep(poll)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--fire-epoch", required=True, type=float)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--runner-pid", required=True, type=int)
    p.add_argument("--prefix", default=None, help="this run's pod-name prefix")
    a = p.parse_args(argv)
    watch(a.job_id, a.fire_epoch, a.run_dir, a.runner_pid, prefix=a.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
