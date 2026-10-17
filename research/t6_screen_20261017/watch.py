"""Receipt-only watcher for the T6 RESTART CPU evaluation screen.

This watcher never inspects or controls tmux.  It starts only its own bounded,
single-thread CPU children after the T6 fresh-world audit has passed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.t2_screen_20261015.watch_t2b import checkpoint, job_key, row_path, valid_row

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
ARM = "RESTART"
NAMES = tuple(f"{ARM}_s{seed}" for seed in range(10, 15))
MIX_WORLDS = {"scripted": 96, "frozen": 32, "mixed": 32}
INITIAL_ROUND, EXTENSION_ROUND = 72, 73
WORKER_LIMIT = 2


def base_jobs() -> list[dict[str, Any]]:
    """Return the registered round-72 work, with controls first."""
    names = [*(f"CTRL_s{seed}" for seed in range(10, 15)), "baseline", *NAMES]
    return [
        {"name": name, "mix": mix, "worlds": worlds, "round_index": INITIAL_ROUND}
        for name in names
        for mix, worlds in MIX_WORLDS.items()
    ]


def extension_jobs() -> list[dict[str, Any]]:
    """Return the one registered round-73 scripted-only extension."""
    return [
        {"name": name, "mix": "scripted", "worlds": 96, "round_index": EXTENSION_ROUND}
        for name in [*(f"CTRL_s{seed}" for seed in range(10, 15)), *NAMES]
    ]


def training_status(train_root: Path) -> dict[str, int]:
    """Read terminal RESTART receipts without interacting with their processes."""
    statuses: dict[str, int] = {}
    for lane in range(2):
        path = train_root / f"lane{lane}-status.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            try:
                receipt = json.loads(line)
                output = Path(receipt["out"])
                name = output.name
                code = int(receipt["exit_code"])
                success = receipt["success"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"malformed T6 lane receipt: {path}: {error}") from error
            if name not in NAMES or output != train_root / name or type(success) is not bool:
                raise ValueError(f"incompatible T6 lane receipt: {path}")
            effective = 0 if code == 0 and success else (code if code else 97)
            if name in statuses and statuses[name] != effective:
                raise ValueError(f"conflicting terminal receipt for {name}")
            statuses[name] = effective
    return statuses


def training_terminal(statuses: dict[str, int]) -> bool:
    """Return true after every final receipt, or promptly on a registered failure."""
    return any(code != 0 for code in statuses.values()) or all(name in statuses for name in NAMES)


def summary_ok(path: Path) -> bool:
    """Accept only a clean completed T6 8M training summary."""
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("halt") is None
        and summary.get("actor_exitcodes") == [0, 0]
        and isinstance(summary.get("transitions"), int)
        and summary["transitions"] >= 8_000_000
    )


def ready(name: str, train: Path, ctrl: Path) -> bool:
    """Require receipt, fixed final, and clean summary for a RESTART candidate."""
    candidate = checkpoint(name, train, ctrl)
    if candidate is None:
        return True
    if name in NAMES and (
        training_status(train).get(name) != 0 or not summary_ok(candidate.parent / "summary.json")
    ):
        return False
    return candidate.is_file() and (candidate.parent / "summary.json").is_file()


def environment() -> dict[str, str]:
    """Pin every evaluator child to one CPU numerical thread."""
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
    """Build one registered, bare-network T6 rowwise CPU evaluation command."""
    candidate = checkpoint(job["name"], args.train, args.ctrl)
    command = [
        "nice",
        "-n",
        "10",
        str(REPO / "venv/bin/python"),
        "-m",
        "research.t6_screen_20261017.probe",
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


def valid_t6_row(path: Path, job: dict[str, Any], candidate: Path | None) -> bool:
    """Require the inherited survival row plus T6 terminal-mass observations."""
    if not valid_row(path, job, candidate):
        return False
    row = json.loads(path.read_text())
    values = row.get("length_at_5000")
    if row.get("length_at_5000_convention") != "post-frame 5000 logical mass; dead heroes zero":
        raise ValueError(f"missing T6 final-length convention: {path}")
    if not isinstance(values, dict) or set(values) != set(row["mass"]):
        raise ValueError(f"incomplete T6 final-length map: {path}")
    for seed, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid T6 final length: {path}: {seed}")
        if row["deaths"][seed] and value != 0:
            raise ValueError(f"dead T6 hero has final length: {path}: {seed}")
    return True


def _controls_complete(completed: list[dict[str, Any]]) -> bool:
    controls = [job for job in base_jobs() if job["name"] not in NAMES]
    return {job_key(job) for job in controls}.issubset({job_key(job) for job in completed})


def launchable_jobs(
    jobs: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    active: set[str],
    *,
    controls_complete: bool,
    ready_names: set[str],
) -> list[dict[str, Any]]:
    """Choose at most two work rows while enforcing the control barrier."""
    complete = {job_key(job) for job in completed}
    selected: list[dict[str, Any]] = []
    for job in jobs:
        key = job_key(job)
        if len(selected) == WORKER_LIMIT:
            break
        if (
            key in active
            or key in complete
            or job["name"] not in ready_names
            or (job["name"] in NAMES and not controls_complete)
        ):
            continue
        selected.append(job)
    return selected


def extension_due(result: dict[str, Any], extension_checked: bool) -> bool:
    """Apply T6's one-shot extension branch from the analysis contract."""
    return not extension_checked and result.get("arms", {}).get(ARM, {}).get("decision") == "EXTEND"


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _audit(
    path: Path, *, round_index: int, consumed_round72_evidence: Path | None = None
) -> dict[str, Any]:
    """Refresh the two-round fail-closed audit immediately before launch work."""
    from research.t6_screen_20261017 import audit

    receipt = audit.audit(
        repo=REPO,
        artifacts=ARTIFACTS,
        out=path,
        round_index=round_index,
        consumed_round72_evidence=consumed_round72_evidence,
    )
    if not receipt.get("pass"):
        raise ValueError("T6 round-72/73 seed audit did not pass")
    round_key = str(round_index)
    check = receipt.get("checks", {}).get(round_key, {})
    if check.get("count") != 96 or not check.get("unique"):
        raise ValueError(f"T6 round-{round_key} audit is incomplete")
    _atomic_json(path, receipt)
    return receipt


@contextmanager
def owned_workers(active: dict[str, Any]):
    """Reap only evaluator subprocesses owned by this watcher."""
    try:
        yield
    finally:
        for process, log, _ in active.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        active.clear()


def _event(out: Path, kind: str, **data: Any) -> None:
    record = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **data}
    with (out / "watcher.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def _provisional(rows: list[dict[str, Any]], statuses: dict[str, int]) -> dict[str, Any]:
    from research.t6_screen_20261017 import analyze

    return analyze.analyze(rows, statuses)


def _finish(args: argparse.Namespace, rows: list[dict[str, Any]], statuses: dict[str, int]) -> int:
    """Persist reproducible raw rows and provisional decision evidence for the result owner."""
    result = _provisional(rows, statuses)
    (args.out / "probes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _atomic_json(args.out / "analysis.json", result)
    _event(
        args.out,
        "complete",
        decision=result.get("arms", {}).get(ARM, {}).get("decision"),
        rows=len(rows),
    )
    return 0


def run(args: argparse.Namespace) -> int:
    """Watch receipts and execute T6 rows, including its sole conditional extension."""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows").mkdir(exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    lock = (args.out / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    audit72 = _audit(args.out / "seed_bank_audit_round72.json", round_index=INITIAL_ROUND)
    jobs, active, extension_checked = base_jobs(), {}, False
    _event(args.out, "started", pid=os.getpid(), worker_limit=WORKER_LIMIT, audit=audit72["status"])
    with owned_workers(active):
        while True:
            for key, (process, log, _) in list(active.items()):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                log.close()
                del active[key]
                _event(
                    args.out,
                    "probe_finished" if exit_code == 0 else "probe_failed",
                    key=key,
                    exit=exit_code,
                )
                if exit_code:
                    raise RuntimeError(f"probe {key} exited {exit_code}")
            statuses = training_status(args.train)
            if any(code != 0 for code in statuses.values()):
                _event(args.out, "training_failed", training_status=statuses)
                return _finish(args, [], statuses)
            eligible = [
                job for job in jobs if job["name"] not in NAMES or statuses.get(job["name"]) == 0
            ]
            completed = [
                job
                for job in eligible
                if valid_t6_row(
                    row_path(args.out, job), job, checkpoint(job["name"], args.train, args.ctrl)
                )
            ]
            if training_terminal(statuses) and len(completed) == len(eligible) and not active:
                rows = [json.loads(row_path(args.out, job).read_text()) for job in completed]
                result = _provisional(rows, statuses)
                if extension_due(result, extension_checked):
                    round72 = args.out / "probes_round72.jsonl"
                    round72.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    _audit(
                        args.out / "seed_bank_audit_round73.json",
                        round_index=EXTENSION_ROUND,
                        consumed_round72_evidence=round72,
                    )
                    jobs.extend(extension_jobs())
                    extension_checked = True
                    _event(args.out, "extension_required", jobs=extension_jobs())
                    continue
                return _finish(args, rows, statuses)
            controls_complete = _controls_complete(completed)
            ready_names = {
                job["name"] for job in eligible if ready(job["name"], args.train, args.ctrl)
            }
            capacity = WORKER_LIMIT - len(active)
            for job in launchable_jobs(
                eligible,
                completed,
                set(active),
                controls_complete=controls_complete,
                ready_names=ready_names,
            )[: max(capacity, 0)]:
                key = job_key(job)
                log = (args.out / "logs" / f"{key}.log").open("a")
                process = subprocess.Popen(
                    probe_command(job, args),
                    cwd=REPO,
                    env=environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                active[key] = process, log, job
                _event(args.out, "probe_started", key=key, pid=process.pid)
            _atomic_json(
                args.out / "progress.json",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=ARTIFACTS / "redesign-t6-mac-20261016")
    parser.add_argument("--ctrl", type=Path, default=ARTIFACTS / "redesign-m6-mac-20261014")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "redesign-t6-probes-20261017")
    parser.add_argument("--poll-seconds", type=float, default=30)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
