"""CPU-only, four-child watcher for all 70 registered T3 curve probes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.t2_screen_20261015.watch_t2b import (atomic_json,
                                                   training_status, valid_row)
from research.t3_screen_20261015.banks import _sha256
from research.t3_screen_20261015.watch_t3 import environment, owned_workers

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
ARMS = ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
NAMES = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in range(10, 15))


def jobs(train: Path, ctrl: Path) -> list[dict[str, Any]]:
    result = []
    for name in [f"CTRL_s{s}" for s in range(10, 15)] + list(NAMES):
        for length in ((4, 8) if name.startswith("CTRL_") else (2, 4, 6, 8)):
            filename = "final.pth" if length == 8 else f"ckpt_{length * 1000}k.pth"
            result.append(
                dict(
                    name=f"{name}__{length}M",
                    run=name,
                    length=length,
                    checkpoint=str((ctrl if name.startswith("CTRL_") else train) / name / filename),
                    mix="scripted",
                    worlds=32,
                    round_index=64,
                )
            )
    return result


def ready(job: dict[str, Any], statuses: dict[str, int]) -> bool:
    """Only probe completed training runs, with successful terminal summaries."""
    ckpt = Path(job["checkpoint"])
    if job["run"] in NAMES and statuses.get(job["run"]) != 0:
        return False
    summary_path = ckpt.parent / "summary.json"
    if not ckpt.is_file() or not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    if summary.get("halt") or summary.get("transitions", 0) < 8_000_000:
        raise ValueError(f"Incomplete or halted training summary: {summary_path}")
    return True


def command(job: dict[str, Any], out: Path) -> list[str]:
    return [
        "nice",
        "-n",
        "10",
        str(REPO / "venv/bin/python"),
        "-m",
        "research.t2_screen_20261015.probe_t2",
        "--name",
        job["name"],
        "--arm",
        "student",
        "--student",
        job["checkpoint"],
        "--veto",
        "none",
        "--mix",
        "scripted",
        "--worlds",
        "32",
        "--round-index",
        "64",
        "--frames",
        "5000",
        "--ego2s-forward",
        "rowwise",
        "--ego2s-mask",
        "advisory",
        "--v61-backend",
        "numba",
        "--anchors",
        "fast",
        "--out",
        str(out),
    ]


def run(args: argparse.Namespace) -> int:
    from research.redesign_m2_20261008.dev_probe import probe_seeds

    out = args.out
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    lock = (out / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    audit_path = out / "seed_bank_audit.json"
    audit = json.loads(audit_path.read_text())
    if not (
        audit.get("pass")
        and audit.get("round_index") == 64
        and audit.get("worlds") == 32
        and audit.get("seeds") == sorted(probe_seeds(32, 64))
    ):
        raise ValueError("Missing or incompatible round-64 seed audit")
    all_jobs = jobs(args.train, args.ctrl)
    active: dict[str, Any] = {}
    completed: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}

    def event(kind: str, **data: Any) -> None:
        row = dict(utc=datetime.now(timezone.utc).isoformat(), event=kind, **data)
        with (out / "watcher.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    event(
        "started",
        pid=os.getpid(),
        workers=4,
        device="cpu",
        threads=1,
        audit_sha256=_sha256(audit_path),
    )
    with owned_workers(active):
        while True:
            for key, (process, log, job) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                del active[key]
                event("probe_finished" if rc == 0 else "probe_failed", name=key, exit=rc)
                if rc:
                    raise RuntimeError(f"Probe failed; no automatic retry: {key}: {rc}")
                path = out / "rows" / f"{key}.json"
                if not valid_row(path, job, Path(job["checkpoint"])):
                    raise ValueError(f"Successful child did not produce a valid row: {key}")
                if _sha256(Path(job["checkpoint"])) != identities[key]:
                    raise ValueError(f"Checkpoint changed during probe: {key}")
                completed[key] = json.loads(path.read_text())
            statuses = training_status(args.train)
            failed = {
                name: statuses[name] for name in NAMES if name in statuses and statuses[name] != 0
            }
            if failed:
                raise RuntimeError(f"T3 run failed; cannot claim all 15 complete: {failed}")
            for job in all_jobs:
                key = job["name"]
                if key in completed or key in active:
                    continue
                path = out / "rows" / f"{key}.json"
                if valid_row(path, job, Path(job["checkpoint"])):
                    completed[key] = json.loads(path.read_text())
                    continue
                if len(active) >= 4 or not ready(job, statuses):
                    continue
                identities[key] = _sha256(Path(job["checkpoint"]))
                cmd = command(job, path)
                log = (out / "logs" / f"{key}.log").open("a")
                process = subprocess.Popen(
                    cmd,
                    cwd=REPO,
                    env=environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                active[key] = process, log, job
                event(
                    "probe_started",
                    name=key,
                    pid=process.pid,
                    command=cmd,
                    checkpoint_sha256=identities[key],
                )
            runs_done = [
                name for name in NAMES if all(f"{name}__{m}M" in completed for m in (2, 4, 6, 8))
            ]
            atomic_json(
                out / "progress.json",
                dict(
                    completed=len(completed),
                    required=70,
                    runs_probed=runs_done,
                    runs_required=15,
                    active=list(active),
                    training_status={n: statuses.get(n) for n in NAMES},
                ),
            )
            if len(completed) == 70 and not active:
                # Revalidate every saved artifact against its final checkpoint identity.
                for job in all_jobs:
                    assert valid_row(
                        out / "rows" / f"{job['name']}.json", job, Path(job["checkpoint"])
                    )
                rows = [completed[job["name"]] for job in all_jobs]
                atomic_json(out / "probes.json", rows)
                event("complete", rows=70, runs=15, audit_sha256=_sha256(audit_path))
                return 0
            time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ARTIFACTS / "redesign-t3-mac-20261015")
    parser.add_argument("--ctrl", type=Path, default=ARTIFACTS / "redesign-m6-mac-20261014")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "redesign-t3-curve-probes-20261015")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
