#!/usr/bin/env python3
"""Detached local watchdog for one fan-out run: deletes the run's pods when needed.

Fires when ANY of:

* the hard time ``--fire-epoch`` (job max wall time + grace) has passed;
* the runner process ``--runner-pid`` is gone and ``receipt.json`` has not appeared for
  ``runner_dead_seconds`` (the runner crashed, was killed -9, or lost its terminal);
* the runner finished (``receipt.json`` exists): sweep, re-sweep later, then exit.

On firing it deletes (a) every pod id in the run's on-disk registry
(``<run>/pods/*.response.json``, written right after each create) and (b) every pod whose
name starts with ``--prefix`` (this run's ``rpf-<job>--<run>-``). Registry deletes work even
when ``GET /pods`` fails. It keeps retrying with backoff for up to ``--give-up-seconds``.

Survival: on macOS it is started as a launchd job in the user's GUI domain (not a child of
the runner's terminal/tmux, and keeps Keychain access for rp.py); elsewhere with setsid.
It ignores SIGHUP/SIGINT, writes ``watchdog.pid``, and only ever deletes this run's pods.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.runpod_fanout import jobspec  # noqa: E402
from research.runpod_fanout.rp_client import RpClient, RunPodError  # noqa: E402

FAKE_RP_ENV = "RPF_TEST_FAKE_RP_STATE"  # tests only: a JSON file standing in for RunPod


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


def registry_ids(run_dir: Path, prefix: str) -> Set[str]:
    """Pod ids this run created, from ``pods/*.response.json`` (name must carry the prefix)."""
    ids: Set[str] = set()
    for path in sorted((Path(run_dir) / "pods").glob("*.response.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if row.get("id") and str(row.get("name", path.name)).startswith(prefix):
            ids.add(str(row["id"]))
    return ids


def overdue_registry_ids(run_dir: Path, prefix: str, now: float, grace: float = 120.0) -> Set[str]:
    """Registry pods whose ``rpf_until_epoch`` (+grace) has passed."""
    ids: Set[str] = set()
    for path in sorted((Path(run_dir) / "pods").glob("*.response.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        until = row.get("rpf_until_epoch")
        if (
            row.get("id")
            and str(row.get("name", "")).startswith(prefix)
            and until is not None
            and now > float(until) + grace
        ):
            ids.add(str(row["id"]))
    return ids


def sweep(
    rp: Any,
    prefix: str,
    sleep: Callable[[float], None],
    run_dir: Path,
    clock: Callable[[], float] = time.time,
    give_up_seconds: float = 7200.0,
) -> List[str]:
    """Delete registry ids + prefix-listed pods until all are confirmed gone; leftovers."""
    started = clock()
    gone: Set[str] = set()
    delay = 15.0
    left: List[str] = ["unconfirmed"]
    while clock() - started < give_up_seconds:
        wanted = registry_ids(run_dir, prefix) - gone
        listed: List[str] = []
        list_ok = True
        try:
            listed = [p["id"] for p in rp.list_pods() if str(p.get("name", "")).startswith(prefix)]
        except RunPodError as exc:
            list_ok = False
            log(f"list failed: {exc}")
        targets = sorted(set(listed) | wanted)
        if list_ok:
            gone |= wanted - set(listed)  # not listed any more: deleted
            if not listed:
                return []
        for pod_id in targets:
            try:
                rp.delete_pod(pod_id, confirm=True)
                log(f"deleted {pod_id}")
            except RunPodError as exc:
                log(f"delete {pod_id} failed: {exc}")
                continue
            if not list_ok:
                try:
                    if rp.get_pod(pod_id) is None:
                        gone.add(pod_id)
                except RunPodError:
                    pass
        left = targets
        sleep(delay)
        delay = min(120.0, delay * 1.5)
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
    resweep_seconds: float = 180.0,
    give_up_seconds: float = 7200.0,
) -> str:
    policy = jobspec.load_policy()
    job_prefix = f"{policy['pod_name_prefix']}{job_id}--"
    prefix = prefix or job_prefix
    if not prefix.startswith(job_prefix):
        raise SystemExit("watchdog prefix must lie inside the job's rpf-<job>-- namespace")
    rp = rp or RpClient()
    dead_since = None
    retired: Set[str] = set()
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
        overdue = overdue_registry_ids(run_dir, prefix, now) - retired
        if overdue and not reason:
            # Per-pod max lifetime: the runner should already have retired these pods.
            log(f"pods past their reservation horizon: {sorted(overdue)}")
            for pod_id in sorted(overdue):
                try:
                    rp.delete_pod(pod_id, confirm=True)
                    log(f"deleted overdue {pod_id}")
                    retired.add(pod_id)
                except RunPodError as exc:
                    try:
                        if rp.get_pod(pod_id) is None:
                            retired.add(pod_id)
                            continue
                    except RunPodError:
                        pass
                    log(f"delete overdue {pod_id} failed: {exc}")
        if reason:
            log(f"FIRING: {reason}")
            left = sweep(rp, prefix, sleep, run_dir, clock, give_up_seconds)
            # A pod from a create still in flight can surface late: sweep again later.
            sleep(resweep_seconds)
            left = sweep(rp, prefix, sleep, run_dir, clock, give_up_seconds)
            log(f"done; leftovers={left}")
            (Path(run_dir) / "watchdog_result.json").write_text(
                json.dumps({"reason": reason, "leftovers": left, "utc": time.time()})
            )
            return reason
        sleep(poll)


class FileFakeRp:
    """Tests only: pods live in a JSON file ``{"pods": [...], "deleted": [...]}``."""

    def __init__(self, path: str):
        self.path = Path(path)

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())

    def list_pods(self):
        data = self._load()
        if data.get("list_fails"):
            raise RunPodError("fake list failure")
        return data["pods"]

    def get_pod(self, pod_id):
        return next((p for p in self._load()["pods"] if p["id"] == pod_id), None)

    def delete_pod(self, pod_id, confirm):
        assert confirm
        data = self._load()
        data["pods"] = [p for p in data["pods"] if p["id"] != pod_id]
        data.setdefault("deleted", []).append(pod_id)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.path)
        return {}


def launchd_label(job_id: str, run_name: str) -> str:
    return f"com.snakedqn.rpf.{job_id}.{run_name}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--fire-epoch", required=True, type=float)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--runner-pid", required=True, type=int)
    p.add_argument("--prefix", default=None, help="this run's pod-name prefix")
    p.add_argument("--resweep-seconds", type=float, default=180.0)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--launchd-label", default=None, help="unregister this launchd job at exit")
    a = p.parse_args(argv)
    for sig in (signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)  # survive the terminal / tmux going away
    (a.run_dir / "watchdog.pid").write_text(str(os.getpid()))
    fake = os.environ.get(FAKE_RP_ENV)
    rp = FileFakeRp(fake) if fake else None
    try:
        watch(
            a.job_id,
            a.fire_epoch,
            a.run_dir,
            a.runner_pid,
            rp=rp,
            prefix=a.prefix,
            resweep_seconds=a.resweep_seconds,
            poll=a.poll_seconds,
        )
    finally:
        (a.run_dir / "watchdog.done").write_text(str(time.time()))
        if a.launchd_label:  # last act: remove our own launchd registration
            subprocess.run(
                ["launchctl", "bootout", f"gui/{os.getuid()}/{a.launchd_label}"],
                capture_output=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
