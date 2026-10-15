"""T3 probe orchestration; reuses the T2 evaluator and validated artifact contract."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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
    ready,
    row_path,
    training_status,
    valid_row,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
ARMS = ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
NAMES = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in range(10, 15))
MIX_WORLDS = {"scripted": 96, "frozen": 32, "mixed": 32}


def base_jobs() -> list[dict[str, Any]]:
    """Controls and incumbent must finish before candidate probes start."""
    names = [f"CTRL_s{s}" for s in range(10, 15)] + ["baseline"] + list(NAMES)
    return [
        dict(name=name, mix=mix, worlds=worlds, round_index=62)
        for name in names
        for mix, worlds in MIX_WORLDS.items()
    ]


def training_lanes() -> list[str]:
    """Read session names only; never address or control training sessions."""
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True
    )
    if result.returncode and "no server running" not in result.stderr:
        return ["unknown"]
    return [name for name in result.stdout.splitlines() if name in ("t3-lane0", "t3-lane1")]


def worker_limit(statuses: dict[str, int]) -> int:
    """Four workers only after every training run is terminal and lanes have exited."""
    return 4 if all(name in statuses for name in NAMES) and not training_lanes() else 2


def environment() -> dict[str, str]:
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
    ckpt = checkpoint(job["name"], args.train, args.ctrl)
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
        "baseline" if ckpt is None else "student",
        "--veto",
        "v8" if ckpt is None else "none",
        "--mix",
        job["mix"],
        "--worlds",
        str(job["worlds"]),
        "--round-index",
        str(job["round_index"]),
        "--out",
        str(row_path(args.out, job)),
    ]
    if ckpt is not None:
        command.extend(["--student", str(ckpt)])
    return command


@contextmanager
def owned_workers(active: dict[str, Any]):
    """Reap only children launched by this phase, including on failure."""
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


def run(args: argparse.Namespace) -> int:
    from research.t3_screen_20261015 import analyze_t3

    out = args.out
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    lock = (out / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    audit_path = out / "seed_bank_audit.json"
    if not audit_path.is_file():
        raise ValueError("Create the pre-probe seed bank audit first")
    audit = json.loads(audit_path.read_text())
    if not audit["pass"]:
        raise ValueError("Seed bank audit did not pass")
    jobs = base_jobs()
    active: dict[str, Any] = {}

    def event(kind: str, **data: Any) -> None:
        record = dict(utc=datetime.now(timezone.utc).isoformat(), event=kind, **data)
        with (out / "watcher.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    event("started", pid=os.getpid(), audit=str(audit_path))
    with owned_workers(active):
        while True:
            for key, (process, log, job) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                del active[key]
                event("probe_finished" if rc == 0 else "probe_failed", key=key, exit=rc)
                if rc:
                    # Fail closed; a failed probe needs diagnosis, never automatic retry.
                    raise RuntimeError(f"Probe {key} exited {rc}; inspect its log")
            statuses = {k: v for k, v in training_status(args.train).items() if k in NAMES}
            eligible = [job for job in jobs if statuses.get(job["name"], 0) == 0]
            completed = [
                job
                for job in eligible
                if valid_row(
                    row_path(out, job), job, checkpoint(job["name"], args.train, args.ctrl)
                )
            ]
            terminal = all(name in statuses for name in NAMES)
            if terminal and len(completed) == len(eligible) and not active:
                rows = [json.loads(row_path(out, job).read_text()) for job in completed]
                result = analyze_t3.analyze(rows, statuses)
                extra = analyze_t3.extension_requirements(result, rows)
                if extra:
                    known = {job_key(job) for job in jobs}
                    jobs.extend(job for job in extra if job_key(job) not in known)
                    event("extension_required", jobs=extra)
                    continue
                (out / "probes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                event("probes_complete", rows=len(rows))
                # Re-rolls are a separate, resumable phase, with no overlap with probes.
                return finish(args, rows, statuses, audit, lock, event)
            controls_done = all(job in completed for job in jobs[:18])
            for job in eligible:
                if len(active) >= worker_limit(statuses):
                    break
                key = job_key(job)
                if key in active or job in completed:
                    continue
                if job["name"] in NAMES and not controls_done:
                    continue
                if not ready(job["name"], args.train, args.ctrl):
                    continue
                command = probe_command(job, args)
                log = (out / "logs" / (key + ".log")).open("a")
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                active[key] = process, log, job
                event("probe_started", key=key, pid=process.pid, command=command)
            atomic_json(
                out / "progress.json",
                dict(
                    completed=len(completed),
                    required=len(eligible),
                    active=list(active),
                    training_status=statuses,
                    controls_complete=controls_done,
                ),
            )
            time.sleep(args.poll_seconds)


def finish(
    args: argparse.Namespace, rows: list, statuses: dict, audit: dict, lock: Any, event: Any
) -> int:
    """Run the fixed-sample continuation check, then write the complete result."""
    from research.t3_screen_20261015 import analyze_t3, reroll_t3

    out = args.out
    (out / "rerolls").mkdir(exist_ok=True)
    sample = out / "reroll_sample.json"
    reroll_t3.prepare_sample(args.bank, sample)
    sample_sha = reroll_t3.sha256(sample)
    names = [name for name in NAMES if checkpoint(name, args.train, args.ctrl).is_file()]
    active: dict[str, Any] = {}
    results: dict[str, Any] = {}
    with owned_workers(active):
        while len(results) < len(names) or active:
            for name, (process, log) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                del active[name]
                event("reroll_finished" if rc == 0 else "reroll_failed", name=name, exit=rc)
                if rc:
                    raise RuntimeError(f"Re-roll {name} exited {rc}; inspect its log")
            for name in names:
                path = out / "rerolls" / (name + ".json")
                if name in results or not path.exists():
                    continue
                row = json.loads(path.read_text())
                ckpt = checkpoint(name, args.train, args.ctrl)
                if not (
                    row.get("schema") == "t3-reroll-result/v1"
                    and row.get("checkpoint_sha256") == reroll_t3.sha256(ckpt)
                    and row.get("sample_sha256") == sample_sha
                    and row.get("sample_states") == 200
                    and row.get("depth") == 300
                ):
                    raise ValueError(f"Stale re-roll result: {path}")
                results[name] = row
            for name in names:
                if len(active) >= worker_limit(statuses):
                    break
                if name in active or name in results:
                    continue
                command = [
                    "nice",
                    "-n",
                    "10",
                    str(REPO / "venv/bin/python"),
                    "-m",
                    "research.t3_screen_20261015.reroll_t3",
                    "--checkpoint",
                    str(checkpoint(name, args.train, args.ctrl)),
                    "--bank",
                    str(args.bank),
                    "--sample",
                    str(sample),
                    "--out",
                    str(out / "rerolls" / (name + ".json")),
                ]
                log = (out / "logs" / (name + "__reroll.log")).open("a")
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                active[name] = process, log
                event("reroll_started", name=name, pid=process.pid, command=command)
            atomic_json(
                out / "reroll_progress.json",
                dict(completed=list(results), required=names, active=list(active)),
            )
            if len(results) < len(names) or active:
                time.sleep(args.poll_seconds)
    run_dirs = {
        name: checkpoint(name, args.train, args.ctrl).parent
        for name in [f"CTRL_s{s}" for s in range(10, 15)] + list(NAMES)
    }
    telemetry = analyze_t3.read_telemetry(run_dirs)
    result = analyze_t3.analyze(rows, statuses, telemetry, results)
    result["seed_bank_audit"] = audit
    result["artifacts"] = dict(root=str(out), sample=str(sample), sample_sha256=sample_sha)
    atomic_json(out / "telemetry.json", telemetry)
    atomic_json(out / "rerolls.json", results)
    analyze_t3.write_results(result, out / "analysis.json", args.results)
    event("complete", results=str(args.results), analysis=str(out / "analysis.json"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ARTIFACTS / "redesign-t3-mac-20261015")
    parser.add_argument("--ctrl", type=Path, default=ARTIFACTS / "redesign-m6-mac-20261014")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "redesign-t3-probes-20261015")
    parser.add_argument("--bank", type=Path, default=ARTIFACTS / "t3-rollout-labels-20261015/bank")
    parser.add_argument(
        "--results", type=Path, default=REPO / "docs/research/t3_screen_result_2026-10-15.md"
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
