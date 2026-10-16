"""Watch T4c receipts and own only bounded CPU rowwise probes.

This watcher never queries tmux.  It reads the two lane status receipts and
starts at most two single-thread CPU probe children of its own.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
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

from . import banks_t4c

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
ARM = "TWO_SIDED_T"
NAMES = tuple(f"{ARM}_s{seed}" for seed in range(10, 15))
MIX_WORLDS = {"scripted": 96, "frozen": 32, "mixed": 32}
WORKER_LIMIT = 2
LANES = ((f"{ARM}_s10", f"{ARM}_s12", f"{ARM}_s14"), (f"{ARM}_s11", f"{ARM}_s13"))


def base_jobs() -> list[dict[str, Any]]:
    """Return registered round-66 jobs, ordered controls then candidates."""
    names = [f"CTRL_s{seed}" for seed in range(10, 15)] + ["baseline"] + list(NAMES)
    return [
        {"name": name, "mix": mix, "worlds": worlds, "round_index": 66}
        for name in names
        for mix, worlds in MIX_WORLDS.items()
    ]


def training_status(train_root: Path) -> dict[str, int]:
    """Read only terminal T4c lane receipts; malformed text cannot be success."""
    result: dict[str, int] = {}
    for lane in ("lane0-status.txt", "lane1-status.txt"):
        path = train_root / lane
        if not path.is_file():
            continue
        for name, code in re.findall(
            r"^(TWO_SIDED_T_s1[0-4]) exit=(-?\d+)$", path.read_text(), re.M
        ):
            if name in result and result[name] != int(code):
                raise ValueError(f"conflicting terminal receipt for {name}")
            result[name] = int(code)
    return result


def training_terminal(statuses: dict[str, int]) -> bool:
    """A failed seed stops its lane, while the other lane continues independently."""
    return all(
        any(statuses.get(name, 0) != 0 for name in lane) or all(name in statuses for name in lane)
        for lane in LANES
    )


def ready(name: str, train: Path, ctrl: Path) -> bool:
    """Require receipt, final checkpoint and summary for a trained candidate."""
    path = checkpoint(name, train, ctrl)
    if path is None:
        return True
    return (
        (name not in NAMES or training_status(train).get(name) == 0)
        and path.is_file()
        and (path.parent / "summary.json").is_file()
    )


def valid_t4c_row(path: Path, job: dict[str, Any], candidate: Path | None) -> bool:
    """Validate the preserved T2 contract plus detached final-length values."""
    if not valid_row(path, job, candidate):
        return False
    row = json.loads(path.read_text())
    if not isinstance(row.get("mass"), dict):
        raise ValueError(f"missing T4c final-length source map: {path}")
    expected = set(row["mass"])
    values = row.get("length_at_5000")
    if row.get("length_at_5000_convention") != "post-frame 5000 logical mass; dead heroes zero":
        raise ValueError(f"missing T4c final-length convention: {path}")
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError(f"incomplete T4c final-length map: {path}")
    for seed, value in values.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid T4c final length: {path}: {seed}")
        if row["deaths"][seed] and value != 0:
            raise ValueError(f"dead T4c hero has final length: {path}: {seed}")
    return True


def _controls_complete(done: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> bool:
    control = [job for job in jobs if job["round_index"] == 66 and job["name"] not in NAMES]
    return {job_key(job) for job in control}.issubset({job_key(job) for job in done})


def extension_jobs() -> list[dict[str, Any]]:
    """Return the one registered round-67 scripted extension, if required."""
    return [
        {"name": name, "mix": "scripted", "worlds": 96, "round_index": 67}
        for name in [*(f"CTRL_s{seed}" for seed in range(10, 15)), *NAMES]
    ]


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
    candidate = checkpoint(job["name"], args.train, args.ctrl)
    command = [
        "nice",
        "-n",
        "10",
        str(REPO / "venv/bin/python"),
        "-m",
        "research.t3_screen_20261015.probe_t4c",
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
        command += ["--student", str(candidate)]
    return command


@contextmanager
def owned_workers(active: dict[str, Any]):
    """Terminate only the subprocesses this watcher itself started."""
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


def _event(out: Path, kind: str, **data: Any) -> None:
    record = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **data}
    with (out / "watcher.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def _load_audit(path: Path, *, extension: bool) -> dict[str, Any]:
    """Reuse a passing immutable audit receipt when resuming owned probes."""
    audit = json.loads(path.read_text())
    if audit.get("schema") != "t4c-seed-bank-audit/v1":
        raise ValueError(f"invalid T4c seed-audit schema: {path}")
    round_index = 67 if extension else 66
    check = audit.get("checks", {}).get(str(round_index), {})
    from research.redesign_m2_20261008 import ni_spec
    from research.t3_screen_20261015 import banks

    expected = sorted(ni_spec.distill_seeds(96, round_index=round_index))
    if (
        not check.get("unique")
        or check.get("count") != 96
        or check.get("seeds") != expected
        or check.get("seed_digest") != banks._seed_digest(expected)
    ):
        raise ValueError(f"invalid T4c seed-audit bank: {path}")
    if extension:
        if not audit.get("extension_pass") or audit.get("extension_round") != 67:
            raise ValueError(f"invalid T4c extension audit: {path}")
    elif not audit.get("pass"):
        raise ValueError(f"T4c round-66 audit did not pass: {path}")
    return audit


def _finish(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    statuses: dict[str, int],
    audit: dict[str, Any],
) -> int:
    """Produce the one resumable post-probe analysis and CPU diagnostic receipt."""
    from research.t3_screen_20261015 import analyze_t4c, diagnostics_t4c, drift_t4c

    roots = {
        name: checkpoint(name, args.train, args.ctrl).parent
        for name in [*NAMES, *(f"CTRL_s{s}" for s in range(10, 15))]
    }
    phase = args.out / "diagnostic_phase.json"
    atomic_json(phase, {"phase": "final_bank_and_drift", "status": "running"})
    summary = diagnostics_t4c.load_drift_summary()
    if summary is None:
        raise ValueError("missing verified T4c drift baseline summary")
    drift_root = args.out / "t4c-final-drift"
    receipts = []
    complete_finals = all(
        statuses.get(name) == 0 and (roots[name] / "final.pth").is_file() for name in NAMES
    )
    for seed in range(10, 15) if complete_finals else ():
        checkpoint_path = roots[f"{ARM}_s{seed}"] / "final.pth"
        capture_path = drift_root / "captures" / f"CTRL_s{seed}__r62__scripted.npz"
        receipt_path = capture_path.with_suffix(".t4c-final.receipt.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("schema") != "t4c-final-drift-capture/v1"
                or receipt.get("checkpoint_sha256") != drift_t4c.sha256(checkpoint_path)
                or not capture_path.is_file()
                or receipt.get("capture_sha256") != drift_t4c.sha256(capture_path)
            ):
                raise ValueError(f"invalid existing T4c final drift receipt: {receipt_path}")
        elif capture_path.exists():
            raise ValueError(
                f"T4c drift capture exists without its terminal receipt: {capture_path}"
            )
        else:
            receipt = drift_t4c.capture(seed, checkpoint_path, drift_root)
        receipts.append(receipt)
        atomic_json(
            phase,
            {
                "phase": "final_bank_and_drift",
                "status": "running",
                "completed_seeds": len(receipts),
            },
        )
    candidate_drift = (
        drift_t4c.summarize(drift_root, summary)
        if complete_finals
        else {"candidate_consequential_change_rate": None, "within_ten_percentage_points": None}
    )
    telemetry = diagnostics_t4c.read_telemetry(roots)
    diagnostics = diagnostics_t4c.diagnose(rows, roots, summary, candidate_drift)
    context = {}
    for name, path in (
        ("t4a", REPO / "research/t4a_screen_20261016/results/analysis.json"),
        ("t3", REPO / "research/t3_screen_20261015/results/analysis.json"),
    ):
        if path.is_file():
            context[name] = json.loads(path.read_text())
    result = analyze_t4c.analyze(rows, statuses, telemetry, diagnostics, context)
    diagnostics["declared_fork"] = diagnostics_t4c.declared_fork(result, diagnostics["drift"])
    result = analyze_t4c.analyze(rows, statuses, telemetry, diagnostics, context)
    result["seed_bank_audit"] = audit
    atomic_json(args.out / "telemetry.json", telemetry)
    atomic_json(args.out / "diagnostics.json", diagnostics)
    atomic_json(args.out / "analysis.json", result)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(analyze_t4c.markdown(result))
    atomic_json(phase, {"phase": "complete", "status": "complete", "captures": receipts})
    return 0


def run(args: argparse.Namespace) -> int:
    out = args.out
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    lock = (out / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    audit_path = out / "seed_bank_audit_round66.json"
    if audit_path.exists():
        audit = _load_audit(audit_path, extension=False)
    else:
        audit = banks_t4c.initial_audit(REPO, ARTIFACTS, audit_path)
        atomic_json(audit_path, audit)
    jobs, active, refreshed = base_jobs(), {}, False
    recovered_extension = [job for job in extension_jobs() if row_path(out, job).exists()]
    if recovered_extension:
        # A partial owned extension consumes its bank. Reuse its original passing
        # receipt rather than treating those own rows as a new collision.
        receipts = sorted(out.glob("seed_bank_audit_round67_launch-*.json"))
        if not receipts:
            raise ValueError("T4c round-67 rows exist without a launch audit receipt")
        _load_audit(receipts[-1], extension=True)
        jobs.extend(extension_jobs())
        refreshed = True
    _event(out, "started", pid=os.getpid(), worker_limit=WORKER_LIMIT, audit=str(audit_path))
    with owned_workers(active):
        while True:
            for key, (process, log, _) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[key]
                _event(out, "probe_finished" if code == 0 else "probe_failed", key=key, exit=code)
                if code:
                    raise RuntimeError(f"probe {key} exited {code}")
            statuses = training_status(args.train)
            eligible = [
                job for job in jobs if job["name"] not in NAMES or statuses.get(job["name"]) == 0
            ]
            completed = [
                job
                for job in eligible
                if valid_t4c_row(
                    row_path(out, job), job, checkpoint(job["name"], args.train, args.ctrl)
                )
            ]
            if training_terminal(statuses) and len(completed) == len(eligible) and not active:
                rows = [json.loads(row_path(out, job).read_text()) for job in completed]
                from research.t3_screen_20261015 import analyze_t4c

                result = analyze_t4c.analyze(rows, statuses)
                extra = analyze_t4c.extension_requirements(result, rows)
                if extra:
                    if not refreshed:
                        # No owned r67 row exists yet, so save a new launch-time
                        # receipt rather than trusting an older unconsumed one.
                        refresh_path = out / (
                            "seed_bank_audit_round67_launch-"
                            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                            + ".json"
                        )
                        refresh = banks_t4c.extension_audit(REPO, ARTIFACTS, refresh_path)
                        atomic_json(refresh_path, refresh)
                        refreshed = True
                    known = {job_key(job) for job in jobs}
                    jobs.extend(job for job in extra if job_key(job) not in known)
                    _event(out, "extension_required", jobs=extra)
                    continue
                (out / "probes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                _event(out, "probes_complete", rows=len(rows))
                return _finish(args, rows, statuses, audit)
            controls = _controls_complete(completed, jobs)
            for job in eligible:
                if len(active) >= WORKER_LIMIT:
                    break
                key = job_key(job)
                if (
                    key in active
                    or job in completed
                    or (job["name"] in NAMES and not controls)
                    or not ready(job["name"], args.train, args.ctrl)
                ):
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
                _event(out, "probe_started", key=key, pid=process.pid)
            atomic_json(
                out / "progress.json",
                {
                    "completed": len(completed),
                    "required": len(eligible),
                    "active": list(active),
                    "training_status": statuses,
                    "controls_complete": controls,
                    "worker_limit": WORKER_LIMIT,
                },
            )
            time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=ARTIFACTS / "redesign-t4c-mac-20261016")
    parser.add_argument("--ctrl", type=Path, default=ARTIFACTS / "redesign-m6-mac-20261014")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "redesign-t4c-probes-20261016")
    parser.add_argument(
        "--results", type=Path, default=REPO / "docs/research/t4c_screen_result_2026-10-16.md"
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
