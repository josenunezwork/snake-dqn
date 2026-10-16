"""Watch and probe the registered T4a finals without touching training sessions.

The process cap is deliberately fixed at two.  The watcher does not inspect tmux:
the lane status receipts and final/summary files are the only training signals it
consumes, and only children started here are ever terminated or reaped.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.t2_screen_20261015.watch_t2b import (
    atomic_json,
    checkpoint,
    job_key,
    row_path,
    valid_row,
)

from . import banks

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
NAMES = tuple(f"TWO_SIDED_s{seed}" for seed in range(10, 15))
MIX_WORLDS = {"scripted": 96, "frozen": 32, "mixed": 32}
WORKER_LIMIT = 2
LANE_NAMES = (
    ("TWO_SIDED_s10", "TWO_SIDED_s12", "TWO_SIDED_s14"),
    ("TWO_SIDED_s11", "TWO_SIDED_s13"),
)


def base_jobs() -> list[dict[str, Any]]:
    """Return initial round jobs with controls and incumbent before candidates."""
    names = [f"CTRL_s{seed}" for seed in range(10, 15)] + ["baseline"] + list(NAMES)
    return [
        {"name": name, "mix": mix, "worlds": worlds, "round_index": 65}
        for name in names
        for mix, worlds in MIX_WORLDS.items()
    ]


def training_status(train_root: Path) -> dict[str, int]:
    """Read only the two lane receipts; malformed lines never imply success."""
    statuses: dict[str, int] = {}
    for lane in ("lane0-status.txt", "lane1-status.txt"):
        path = train_root / lane
        if not path.is_file():
            continue
        for name, exit_code in re.findall(
            r"^(TWO_SIDED_s1[0-4]) exit=(-?\d+)$", path.read_text(), re.M
        ):
            if name in statuses and statuses[name] != int(exit_code):
                raise ValueError(f"conflicting terminal receipt for {name}")
            statuses[name] = int(exit_code)
    return statuses


def ready(name: str, train_root: Path, ctrl_root: Path) -> bool:
    """Accept a final only after its own terminal receipt and summary exist."""
    candidate = checkpoint(name, train_root, ctrl_root)
    if candidate is None:
        return True
    if name in NAMES and training_status(train_root).get(name) != 0:
        return False
    return candidate.is_file() and (candidate.parent / "summary.json").is_file()


def training_terminal(statuses: dict[str, int]) -> bool:
    """Both registered sequential lanes must finish or stop at a failed seed."""
    return all(
        any(statuses.get(name, 0) != 0 for name in lane) or all(name in statuses for name in lane)
        for lane in LANE_NAMES
    )


def environment() -> dict[str, str]:
    """Constrain each owned child to the registered CPU rowwise path."""
    return dict(
        os.environ,
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        SNAKE_DQN_DEVICE="cpu",
    )


def probe_command(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Build one hermetic T2-path probe command for a registered T4a job."""
    candidate = checkpoint(job["name"], args.train, args.ctrl)
    command = [
        "nice",
        "-n",
        "10",
        str(REPO / "venv/bin/python"),
        "-m",
        "research.t2_screen_20261015.probe_t2",
        "--name",
        job["name"],
        "--arm",
        "baseline" if candidate is None else "student",
        "--veto",
        "v8" if candidate is None else "none",
        "--mix",
        job["mix"],
        "--worlds",
        str(job["worlds"]),
        "--round-index",
        str(job["round_index"]),
        "--out",
        str(row_path(args.out, job)),
    ]
    if candidate is not None:
        command.extend(["--student", str(candidate)])
    return command


@contextmanager
def owned_workers(active: dict[str, Any]):
    """Reap only subprocesses created by this watcher."""
    try:
        yield
    finally:
        for process, log, *_ in active.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        active.clear()


def _controls_complete(completed: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> bool:
    """Require every initial CTRL and incumbent row before a candidate starts."""
    control_jobs = [job for job in jobs if job["round_index"] == 65 and job["name"] not in NAMES]
    keys = {job_key(job) for job in completed}
    return all(job_key(job) in keys for job in control_jobs)


def _reroll_valid(row: dict[str, Any], checkpoint_path: Path, sample_sha: str) -> bool:
    """Validate the immutable reroll bindings without interpreting its metrics."""
    from research.t4a_screen_20261016 import reroll_t4a

    return (
        row.get("schema") == "t4a-reroll-result/v1"
        and row.get("checkpoint_sha256") == reroll_t4a.sha256(checkpoint_path)
        and row.get("sample_sha256") == sample_sha
        and row.get("sample_states") == 200
    )


def run(args: argparse.Namespace) -> int:
    """Run the resumable probe then descriptive reroll phase."""
    from research.t4a_screen_20261016 import analyze_t4a

    out = args.out
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    lock = (out / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    audit_path = out / "seed_bank_audit.json"
    audit = banks.seed_bank_audit(REPO, ARTIFACTS, out)
    if not audit["pass"]:
        raise ValueError("T4a seed-bank audit did not pass")
    if audit_path.exists():
        initial_audit = json.loads(audit_path.read_text())
        if initial_audit.get("schema") != "t4a-seed-bank-audit/v1" or not initial_audit.get("pass"):
            raise ValueError("Existing pre-probe T4a seed-bank audit is invalid")
        atomic_json(out / "seed_bank_audit_refresh.json", audit)
    else:
        atomic_json(audit_path, audit)
    jobs = base_jobs()
    active: dict[str, Any] = {}

    def event(kind: str, **data: Any) -> None:
        record = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **data}
        with (out / "watcher.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    event("started", pid=os.getpid(), audit=str(audit_path), worker_limit=WORKER_LIMIT)
    with owned_workers(active):
        while True:
            for key, (process, log, _job) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                del active[key]
                event("probe_finished" if rc == 0 else "probe_failed", key=key, exit=rc)
                if rc:
                    raise RuntimeError(f"Probe {key} exited {rc}; inspect its log")
            statuses = training_status(args.train)
            eligible = [
                job for job in jobs if job["name"] not in NAMES or statuses.get(job["name"]) == 0
            ]
            completed = [
                job
                for job in eligible
                if valid_row(
                    row_path(out, job), job, checkpoint(job["name"], args.train, args.ctrl)
                )
            ]
            # A lane intentionally exits after its first failed seed.  Missing later
            # receipts are truthful unstarted runs, not synthetic successful exits.
            terminal = training_terminal(statuses)
            if terminal and len(completed) == len(eligible) and not active:
                rows = [json.loads(row_path(out, job).read_text()) for job in completed]
                result = analyze_t4a.analyze(rows, statuses)
                extra = analyze_t4a.extension_requirements(result, rows)
                if extra:
                    known = {job_key(job) for job in jobs}
                    jobs.extend(job for job in extra if job_key(job) not in known)
                    event("extension_required", jobs=extra)
                    continue
                (out / "probes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                atomic_json(
                    out / "progress.json",
                    {
                        "completed": len(completed),
                        "required": len(eligible),
                        "active": [],
                        "training_status": statuses,
                        "controls_complete": True,
                        "worker_limit": WORKER_LIMIT,
                        "phase": "probes_complete",
                    },
                )
                event("probes_complete", rows=len(rows))
                return finish(args, rows, statuses, audit, lock, event)
            controls_complete = _controls_complete(completed, jobs)
            for job in eligible:
                if len(active) >= WORKER_LIMIT:
                    break
                key = job_key(job)
                if key in active or job in completed:
                    continue
                if job["name"] in NAMES and not controls_complete:
                    continue
                if not ready(job["name"], args.train, args.ctrl):
                    continue
                log = (out / "logs" / f"{key}.log").open("a")
                process = subprocess.Popen(
                    probe_command(job, args),
                    cwd=REPO,
                    env=environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                active[key] = process, log, job
                event("probe_started", key=key, pid=process.pid)
            atomic_json(
                out / "progress.json",
                {
                    "completed": len(completed),
                    "required": len(eligible),
                    "active": list(active),
                    "training_status": statuses,
                    "controls_complete": controls_complete,
                    "worker_limit": WORKER_LIMIT,
                },
            )
            time.sleep(args.poll_seconds)


def finish(
    args: argparse.Namespace, rows: list, statuses: dict, audit: dict, lock: Any, event: Any
) -> int:
    """Create fixed descriptive rerolls, then serialize the analysis owner output."""
    from research.t4a_screen_20261016 import analyze_t4a, reroll_t4a

    out = args.out
    (out / "rerolls").mkdir(exist_ok=True)
    sample = out / "reroll_sample.json"
    reroll_t4a.prepare_sample(args.bank, sample)
    sample_sha = reroll_t4a.sha256(sample)
    active: dict[str, Any] = {}
    results: dict[str, Any] = {}
    reroll_names = tuple(name for name in NAMES if statuses.get(name) == 0)
    with owned_workers(active):
        while len(results) < len(reroll_names) or active:
            for name, (process, log) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                del active[name]
                event("reroll_finished" if rc == 0 else "reroll_failed", name=name, exit=rc)
                if rc:
                    raise RuntimeError(f"Re-roll {name} exited {rc}; inspect its log")
            for name in reroll_names:
                path = out / "rerolls" / f"{name}.json"
                if name in results or not path.is_file():
                    continue
                row = json.loads(path.read_text())
                candidate = checkpoint(name, args.train, args.ctrl)
                if not _reroll_valid(row, candidate, sample_sha):
                    raise ValueError(f"Stale re-roll result: {path}")
                results[name] = row
            for name in reroll_names:
                if len(active) >= WORKER_LIMIT:
                    break
                if name in active or name in results:
                    continue
                candidate = checkpoint(name, args.train, args.ctrl)
                command = [
                    "nice",
                    "-n",
                    "10",
                    str(REPO / "venv/bin/python"),
                    "-m",
                    "research.t4a_screen_20261016.reroll_t4a",
                    "--checkpoint",
                    str(candidate),
                    "--bank",
                    str(args.bank),
                    "--sample",
                    str(sample),
                    "--out",
                    str(out / "rerolls" / f"{name}.json"),
                ]
                log = (out / "logs" / f"{name}__reroll.log").open("a")
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                active[name] = process, log
                event("reroll_started", name=name, pid=process.pid)
            atomic_json(
                out / "reroll_progress.json",
                {
                    "completed": list(results),
                    "required": list(reroll_names),
                    "active": list(active),
                },
            )
            if len(results) < len(reroll_names) or active:
                time.sleep(args.poll_seconds)
    run_dirs = {name: checkpoint(name, args.train, args.ctrl).parent for name in reroll_names}
    telemetry = analyze_t4a.read_telemetry(run_dirs)
    result = analyze_t4a.analyze(rows, statuses, telemetry, results)
    result["seed_bank_audit"] = audit
    result["artifacts"] = {"root": str(out), "sample": str(sample), "sample_sha256": sample_sha}
    atomic_json(out / "telemetry.json", telemetry)
    atomic_json(out / "rerolls.json", results)
    analyze_t4a.write_results(result, out / "analysis.json", args.results)
    event("complete", results=str(args.results), analysis=str(out / "analysis.json"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=ARTIFACTS / "redesign-t4a-mac-20261016")
    parser.add_argument("--ctrl", type=Path, default=ARTIFACTS / "redesign-m6-mac-20261014")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "redesign-t4a-probes-20261016")
    parser.add_argument("--bank", type=Path, default=ARTIFACTS / "t4-prep-20261015/bank-v2")
    parser.add_argument(
        "--results", type=Path, default=REPO / "docs/research/t4a_screen_result_2026-10-16.md"
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
