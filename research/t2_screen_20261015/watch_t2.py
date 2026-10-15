"""Idempotent T2 final-checkpoint watcher; owns only its own nice single-thread probes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.redesign_m2_20261008 import dev_probe, ni_spec

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
TRAIN_ROOT = ARTIFACTS / "redesign-t2-mac-20261015"
CTRL_ROOT = ARTIFACTS / "redesign-m6-mac-20261014"
PROBE_ROOT = ARTIFACTS / "redesign-t2-probes-20261015"
ARMS = ("HUBER", "ANCHOR0", "HUBER_ANCHOR0")
MIX_WORLDS = {"scripted": 96, "frozen": 32, "mixed": 32}


def checkpoint(name: str, train_root: Path, ctrl_root: Path) -> Path | None:
    if name == "baseline":
        return None
    return (ctrl_root if name.startswith("CTRL_") else train_root) / name / "final.pth"


def training_status(train_root: Path) -> dict[str, int]:
    path = train_root / "status.txt"
    if not path.exists():
        return {}
    return {m[0]: int(m[1]) for m in re.findall(r"^(\S+) exit=(-?\d+)$", path.read_text(), re.M)}


def training_lanes() -> list[str]:
    """Read session names only; never send input or modify either training session."""
    completed = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True
    )
    return [name for name in completed.stdout.splitlines() if name in ("t2-lane0", "t2-lane1")]


def ready(name: str, train_root: Path, ctrl_root: Path) -> bool:
    path = checkpoint(name, train_root, ctrl_root)
    if path is None:
        return True
    if not name.startswith("CTRL_") and training_status(train_root).get(name) != 0:
        return False
    return path.is_file() and (path.parent / "summary.json").is_file()


def base_jobs(round_index: int) -> list[dict[str, Any]]:
    names = [f"CTRL_s{s}" for s in range(10, 15)] + ["baseline"]
    names += [f"{arm}_s{s}" for arm in ARMS for s in range(10, 15)]
    return [
        {"name": name, "mix": mix, "worlds": worlds, "round_index": round_index}
        for name in names
        for mix, worlds in MIX_WORLDS.items()
    ]


def job_key(job: dict[str, Any]) -> str:
    return f"{job['name']}__r{job['round_index']}__{job['mix']}"


def row_path(out: Path, job: dict[str, Any]) -> Path:
    return out / "rows" / (job_key(job) + ".json")


def valid_row(path: Path, job: dict[str, Any], ckpt: Path | None) -> bool:
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    for key in ("name", "mix", "round_index", "worlds"):
        if row.get(key) != job[key]:
            raise ValueError(f"stale row: {path}: {key}")
    if not (
        row.get("schema") == "t2-screen-probe/v1"
        and row.get("frames") == 5000
        and row.get("veto") == ("v8" if job["name"] == "baseline" else "none")
        and row.get("ego2s_forward") == "rowwise"
        and row.get("ego2s_mask") == "advisory"
        and row.get("v61_backend") == "numba"
        and row.get("anchors") == "fast"
    ):
        raise ValueError(f"incompatible row: {path}")
    expected = set(map(str, dev_probe.probe_seeds(job["worlds"], job["round_index"])))
    for field in ("mass", "deaths", "alive_frames", "death_frame", "length_at_death", "causes"):
        if set(row[field]) != expected:
            raise ValueError(f"incomplete row: {path}: {field}")
    if job["name"] == "baseline" and row["checkpoint_sha256"] != ni_spec.FRP3_S12[1]:
        raise ValueError(f"wrong incumbent checkpoint: {path}")
    if ckpt and row["checkpoint_sha256"] != hashlib.sha256(ckpt.read_bytes()).hexdigest():
        raise ValueError(f"checkpoint changed since probe: {path}")
    return True


def telemetry(train_root: Path, ctrl_root: Path) -> dict[str, Any]:
    result = {}
    for name in [f"{a}_s{s}" for a in ("CTRL",) + ARMS for s in range(10, 15)]:
        directory = checkpoint(name, train_root, ctrl_root).parent
        values = {"holdout_agreement": None, "greedy_boost_rate": None}
        for line in (directory / "log.jsonl").read_text().splitlines():
            row = json.loads(line)
            for key in ("holdout_agreement", "greedy_boost_rate"):
                if row.get(key) is not None:
                    values[key] = row[key]
                    values[key + "_source"] = "log.jsonl"
        if values["holdout_agreement"] is None:
            summary = json.loads((directory / "summary.json").read_text())
            values["holdout_agreement"] = summary.get("holdout_agreement_final")
            values["holdout_agreement_source"] = "summary.json (log metric unavailable)"
        result[name] = values
    return result


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    from research.t2_screen_20261015 import analyze_t2, banks

    out = args.out
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    lock = (out / "watcher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another T2 watcher owns this output directory")
    lock.write(str(os.getpid()))
    lock.flush()
    audit = banks.seed_bank_audit(REPO)
    if not audit["pass"]:
        raise ValueError("Probe seed banks collide or historical manifests are missing")
    atomic_json(out / "seed_bank_audit.json", audit)
    jobs = base_jobs(60)
    active: dict[str, Any] = {}
    attempts: dict[str, int] = {}
    failures = []
    failures_path = out / "failed_probes.json"
    if failures_path.exists():
        failures = json.loads(failures_path.read_text())

    def event(kind: str, **data: Any) -> None:
        record = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **data}
        with (out / "watcher.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    event("started", pid=os.getpid(), concurrency=2, bank_audit_pass=audit["pass"])
    while True:
        for key, (process, log, job) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            del active[key]
            if rc:
                failures.append({"key": key, "exit": rc, "attempt": attempts[key]})
                atomic_json(failures_path, failures)
                event("probe_failed", key=key, exit=rc)
                if attempts[key] >= 3:
                    raise RuntimeError(f"probe failed 3 times: {key}; see logs")
            else:
                event("probe_finished", key=key)
        completed = [
            job
            for job in jobs
            if valid_row(row_path(out, job), job, checkpoint(job["name"], args.train, args.ctrl))
        ]
        if len(completed) == len(jobs) and not active:
            rows = [json.loads(row_path(out, job).read_text()) for job in completed]
            values = telemetry(args.train, args.ctrl)
            result = analyze_t2.analyze(rows, values)
            extra = analyze_t2.extension_requirements(result, rows)
            if extra:
                existing = {job_key(j) for j in jobs}
                for job in extra:
                    if job_key(job) not in existing:
                        jobs.append(job)
                        existing.add(job_key(job))
                event("extension_required", jobs=extra)
                continue
            (out / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            atomic_json(out / "telemetry.json", values)
            result["seed_bank_audit"] = audit
            result["failed_probes"] = failures
            analyze_t2.write_results(result, out / "analysis.json", args.results)
            event(
                "complete",
                rows=len(rows),
                analysis=str(out / "analysis.json"),
                results=str(args.results),
            )
            return 0
        # Two is deliberately retained after training too: within both resource ceilings.
        for job in jobs:
            key = job_key(job)
            if len(active) >= 2:
                break
            if key in active or job in completed or not ready(job["name"], args.train, args.ctrl):
                continue
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
                str(row_path(out, job)),
            ]
            if ckpt is not None:
                command.extend(["--student", str(ckpt)])
            environment = dict(
                os.environ,
                PYTHONPATH=str(REPO),
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
                SNAKE_DQN_DEVICE="cpu",
            )
            log = (out / "logs" / (key + ".log")).open("a")
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Children retain the singleton flock if this watcher exits unexpectedly.
                pass_fds=(lock.fileno(),),
            )
            active[key] = (process, log, job)
            attempts[key] = attempts.get(key, 0) + 1
            event("probe_started", key=key, pid=process.pid, command=command)
        atomic_json(
            out / "progress.json",
            {
                "completed": len(completed),
                "required": len(jobs),
                "active": list(active),
                "training_status": training_status(args.train),
                "training_lanes_present": training_lanes(),
            },
        )
        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--ctrl", type=Path, default=CTRL_ROOT)
    parser.add_argument("--out", type=Path, default=PROBE_ROOT)
    parser.add_argument(
        "--results", type=Path, default=REPO / "docs/research/t2_screen_result_2026-10-15.md"
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
