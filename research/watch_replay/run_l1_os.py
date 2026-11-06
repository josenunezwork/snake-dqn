"""Resumable l1-os replay sessions; the frozen engine.py/run.py remain unchanged.

Action-log checkpoints reconstruct every recorded frame and verify its rolling hash
before new search. Reconstructing the prefix also regenerates the gzip stream, so a
resumed session is byte-identical to uninterrupted generation on a matching host.
Only completed replay triples are ledger units; a checkpoint is never a replay.
"""

from __future__ import annotations

import argparse
import faulthandler
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from research.watch_replay import screen_l1_os as screen
from research.watch_replay.run import allocation as base_allocation
from research.watch_replay.run import completed, identity
from research.watch_replay.screen_l1_os import (
    CHECKPOINT_SCHEMA,
    COUNTERS,
    DEFAULT_CHECKPOINT_SECONDS,
    HANG_SECONDS,
    HEARTBEAT_SECONDS,
    MAX_RESUME_TRIES,
    STALL_SECONDS,
    STOP_GRACE_SECONDS,
    CheckpointMismatch,
    UnitStalled,
    _normalize,
    _RootLog,
    hardware_fingerprint,
    reject_checkpoint,
)
from tools.ops.package_spec import (
    PackageRefused,
    publish_immutable,
    publish_json,
    unit_id,
    verify_immutable,
)
from web.backend.watch_replay import (
    B8_SHA,
    FRP3_SHA,
    MAX_DECODED,
    MAX_LINE,
    SCHEMA,
    canonical,
    peek_header,
    verify_payload,
)

CONTROLLER = "exact-stop-l1-os"
MAX_BANK_COMPRESSED = 40 * 1024**2
MAX_CHECKPOINT_BYTES = 128 * 1024**2
DEFAULT_RUNNER = "research.watch_replay.run_l1_os:run_unit"


def allocation(
    namespace: str, count: int, controller: str = CONTROLLER, horizon: int = 6000
) -> dict:
    """Use the established replay contract and seed derivation without screen arms."""
    if controller != CONTROLLER:
        raise ValueError("this bank runner requires exact-stop-l1-os")
    return base_allocation(namespace, count, controller, horizon)


def _unit_path(out: Path, index: int) -> Path:
    return out / f"{index:06d}.watch.gz"


def _verified(path: Path) -> bool:
    try:
        verify_immutable(path)
        return True
    except (PackageRefused, OSError):
        return False


def checkpoint_candidates(
    dirs: list[Path], index: int, contract: dict, *, foreign: str = "refuse"
) -> list[dict]:
    """Exact contract only: no predecessor-commit exception for replay-bank sessions."""
    if foreign not in ("refuse", "skip"):
        raise ValueError("foreign must be refuse or skip")
    rejected = screen._rejected_keys(dirs, index)
    found = {}
    for root in dirs:
        for path in sorted((root / f"{index:06d}").glob("frame-*.json")):
            if not _verified(path):
                continue
            value = json.loads(path.read_text())
            if (
                value.get("schema") != CHECKPOINT_SCHEMA
                or value.get("allocation") != identity(contract)
                or value.get("index") != index
                or value.get("arm") != CONTROLLER
                or value.get("seed") != contract["rows"][index]["seed"]
            ):
                if foreign == "skip":
                    continue
                raise PackageRefused(f"foreign checkpoint: {path}")
            frame = value.get("frame")
            counters = value.get("counters", {})
            overrides = value.get("overrides", {})
            if (
                type(frame) is not int
                or not 1 <= frame < contract["horizon"]
                or set(counters) != set(COUNTERS)
                or any(type(v) is not int or v < 0 for v in counters.values())
                or not isinstance(overrides, dict)
                or any(
                    not str(k).isdigit()
                    or not 1 <= int(k) <= frame
                    or type(v) is not int
                    or not 0 <= v < 6
                    for k, v in overrides.items()
                )
            ):
                raise PackageRefused(f"malformed checkpoint: {path}")
            key = (frame, value["frames_sha256"])
            if key not in rejected:
                found.setdefault(key, value)
    return sorted(found.values(), key=lambda v: v["frame"], reverse=True)


def run_unit(
    contract: dict,
    unit: dict,
    out: Path,
    check: Callable[[], None],
    *,
    checkpoint_dirs: list[Path] | None = None,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
    checkpoint_frames: frozenset[int] = frozenset(),
    stall_seconds: float = STALL_SECONDS,
    heartbeat: Path | None = None,
    hard_watchdog: bool = False,
    foreign: str = "refuse",
) -> dict:
    """Publish one session only after full schema and mirror validation."""
    from research.watch_replay import engine, engine_l1_os

    if not math.isfinite(checkpoint_seconds) or not 0 < checkpoint_seconds <= 1800:
        raise ValueError("checkpoint interval must be in (0, 1800] seconds")
    index = unit["index"]
    path = _unit_path(out, index)
    if path.exists() or Path(str(path) + ".complete").exists():
        raise FileExistsError("never overwrite or recompute a completed session")
    candidates = checkpoint_candidates(
        [out / "checkpoints", *(checkpoint_dirs or [])], index, contract, foreign=foreign
    )[:MAX_RESUME_TRIES]
    rejected = 0
    while True:
        resumed = candidates.pop(0) if candidates else None
        try:
            with tempfile.TemporaryFile() as frames_file:
                return _attempt(
                    engine,
                    engine_l1_os,
                    contract,
                    unit,
                    out,
                    check,
                    resumed,
                    rejected,
                    checkpoint_seconds,
                    checkpoint_frames,
                    stall_seconds,
                    heartbeat,
                    hard_watchdog,
                    frames_file,
                )
        except CheckpointMismatch as exc:
            if resumed is None:
                raise
            reject_checkpoint(out, index, resumed, str(exc))
            rejected += 1


def _attempt(
    engine: Any,
    engine_l1_os: Any,
    contract: dict,
    unit: dict,
    out: Path,
    check: Callable[[], None],
    resumed: dict | None,
    rejected: int,
    checkpoint_seconds: float,
    checkpoint_frames: frozenset[int],
    stall_seconds: float,
    heartbeat: Path | None,
    hard_watchdog: bool,
    frames_file: Any,
) -> dict:
    index, arm, horizon = unit["index"], CONTROLLER, contract["horizon"]
    forced_until = int(resumed["frame"]) if resumed else 0
    log = _RootLog(
        forced_until,
        {int(k): int(v) for k, v in (resumed or {}).get("overrides", {}).items()},
        (resumed or {}).get("counters", {name: 0 for name in COUNTERS}),
    )
    prior_compute = float((resumed or {}).get("compute_seconds", 0.0))
    digest = hashlib.sha256()
    state: dict[str, Any] = {"frame": 0, "decoded": 0, "peak_line": 0}
    started = time.monotonic()
    replay_seconds = 0.0 if resumed else None
    last_ckpt = time.monotonic()
    last_print = time.monotonic()
    state["progress"] = time.monotonic()
    state["beat"] = 0.0
    hardware = hardware_fingerprint()

    def beat() -> None:
        if heartbeat is not None and time.monotonic() - state["beat"] >= HEARTBEAT_SECONDS:
            heartbeat.parent.mkdir(parents=True, exist_ok=True)
            heartbeat.touch()
            state["beat"] = time.monotonic()

    def publish_checkpoint(frame: int, frame_digest: str, counters: dict, overrides: dict) -> None:
        path = out / "checkpoints" / f"{index:06d}" / f"frame-{frame:06d}.json"
        if path.exists():
            verify_immutable(path)
            return
        value = {
            "schema": CHECKPOINT_SCHEMA,
            "allocation": identity(contract),
            "index": index,
            "arm": arm,
            "seed": unit["seed"],
            "frame": frame,
            "frames_sha256": frame_digest,
            "counters": counters,
            "overrides": {str(k): v for k, v in sorted(overrides.items())},
            "compute_seconds": prior_compute + (time.monotonic() - started),
            "hardware": hardware,
        }
        used = sum(p.stat().st_size for p in path.parent.glob("*") if p.is_file())
        if used + len(canonical(value)) + 1024 > MAX_CHECKPOINT_BYTES // contract["count"]:
            raise InterruptedError("checkpoint storage guard; resume in a fresh attempt")
        publish_json(path, value)

    def record(snapshot: dict) -> None:
        nonlocal last_ckpt, last_print, replay_seconds
        frame = state["frame"] + 1
        line = _normalize(snapshot)
        state["decoded"] += len(line)
        state["peak_line"] = max(state["peak_line"], len(line))
        if len(line) > MAX_LINE or state["decoded"] + MAX_LINE > MAX_DECODED:
            raise ValueError("incremental replay size limit")
        frames_file.write(line)
        digest.update(line)
        state["frame"] = frame
        if resumed and frame == forced_until:
            if digest.hexdigest() != resumed["frames_sha256"]:
                raise CheckpointMismatch(f"unit {index}: frames 1..{frame} differ")
            replay_seconds = time.monotonic() - started
        if frame > forced_until:
            overrides = {**log.overrides, **log.new_overrides}
            state["snapshot"] = (frame, digest.hexdigest(), dict(log.counters), overrides)
            now = time.monotonic()
            if frame < horizon and (
                frame in checkpoint_frames or now - last_ckpt >= checkpoint_seconds
            ):
                publish_checkpoint(*state["snapshot"])
                last_ckpt = now
        state["progress"] = time.monotonic()
        beat()
        faulthandler.dump_traceback_later(900, exit=hard_watchdog)
        if time.monotonic() - last_print >= 60:
            print(f"screen unit={index} arm={arm} frame={frame}/{horizon}", flush=True)
            last_print = time.monotonic()

    def guarded_check() -> None:
        nonlocal last_ckpt
        try:
            if "snapshot" in state and time.monotonic() - last_ckpt >= checkpoint_seconds:
                publish_checkpoint(*state["snapshot"])
                last_ckpt = time.monotonic()
            check()
            beat()
            if time.monotonic() - state["progress"] > stall_seconds:
                raise UnitStalled(f"unit {index}: no frame for {stall_seconds:.0f} s")
        except InterruptedError:
            if "snapshot" in state:
                publish_checkpoint(*state["snapshot"])
            raise

    patches = [patch.object(engine, "dense_batch", log.wrap(engine.dense_batch, "exact"))]
    if arm == "exact-stop-l1-os":
        patches += [
            patch.object(
                engine_l1_os, "dense_batch_h", log.wrap(engine_l1_os.dense_batch_h, "scaled")
            ),
            patch.object(
                engine_l1_os,
                "open_stop_batch",
                log.wrap(engine_l1_os.open_stop_batch, "open_stop"),
            ),
        ]
    elif arm != "exact-stop-l1":
        raise ValueError(f"unknown screen arm {arm!r}")
    faulthandler.dump_traceback_later(900, exit=hard_watchdog)
    try:
        for item in patches:
            item.start()
        try:
            raw = engine_l1_os.run_controller(
                unit["seed"], arm, guarded_check, horizon=horizon, record=record
            )
        finally:
            for item in reversed(patches):
                item.stop()
    finally:
        faulthandler.cancel_dump_traceback_later()
    frames = raw["frames"]
    if len(frames) != horizon or state["frame"] != horizon:
        raise RuntimeError("incomplete screen unit")
    deaths = [f for f in frames if f["death"]]
    compute = prior_compute + (time.monotonic() - started)
    stats = {
        "hero_deaths": len(deaths),
        "hero_mass_integral": sum(f["mass"] for f in frames) / horizon,
        "max_hero_length": max(f["mass"] for f in frames),
        "decisions": len(raw["decisions"]),
        "search_roots": log.counters["exact_roots"] + log.counters["open_stop_roots"],
        "interventions": log.counters["exact_overrides"] + log.counters["open_stop_fires"],
        "mirror_identity_frames": raw["mirror_identity_frames"],
    }
    header = {
        "schema": SCHEMA,
        "protocol_version": 2,
        "world": "watch6-mechanics-v2-continuous-respawn",
        "unit_id": unit_id(contract["namespace"], "sessions", index),
        "index": index,
        "seed": unit["seed"],
        "horizon": horizon,
        "controller": CONTROLLER,
        "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
        "source_commit": contract["commit"],
        "allocation": identity(contract),
        "fixture": horizon != 6000,
        "stats": stats,
    }
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0, compresslevel=6) as zipped:
        zipped.write(canonical(header))
        frames_file.seek(0)
        while chunk := frames_file.read(1024**2):
            zipped.write(chunk)
            if buffer.tell() > MAX_BANK_COMPRESSED:
                raise ValueError("bank replay exceeds 40 MiB packet bound")
    payload = buffer.getvalue()
    if len(payload) > MAX_BANK_COMPRESSED:
        raise ValueError("bank replay exceeds 40 MiB packet bound")
    verify_payload(payload, product=not header["fixture"])
    published = publish_immutable(_unit_path(out, index), payload)
    return {
        **published,
        "index": index,
        "stats": stats,
        "frames_sha256": digest.hexdigest(),
        "compute_seconds": compute,
        "replay_seconds": replay_seconds,
        "resumed_from_frame": forced_until or None,
        "checkpoints_rejected": rejected,
        "decoded_bytes": state["decoded"] + len(canonical(header)),
        "peak_line_bytes": max(state["peak_line"], len(canonical(header))),
    }


def _schedule(
    tasks: list[dict],
    workers: int,
    out: Path,
    completed: list[int],
    exclusions: list[dict],
    not_started: list[int],
    session_seconds: float | None,
    started: float,
    hang_seconds: float,
    poll_seconds: float,
) -> str | None:
    """Process-per-unit scheduler with heartbeat hang kill and a session budget."""
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    reason: dict[str, Any] = {"why": None, "at": None}

    def request_stop(why: str) -> None:
        if not stop_event.is_set():
            stop_event.set()
            reason.update(why=why, at=time.monotonic())

    previous = {
        sig: signal.signal(sig, lambda s, f: request_stop("signal"))
        for sig in (signal.SIGUSR1, signal.SIGTERM)
    }
    queue = list(tasks)
    running: dict[int, tuple[Any, float]] = {}
    try:
        while queue or running:
            now = time.monotonic()
            if session_seconds is not None and now - started >= session_seconds:
                request_stop("session budget")
            while queue and len(running) < workers and not stop_event.is_set():
                task = queue.pop(0)
                proc = context.Process(target=screen._child, args=(task, stop_event))
                proc.start()
                running[task["unit"]["index"]] = (proc, time.monotonic())
            for index, (proc, t0) in list(running.items()):
                if not proc.is_alive():
                    proc.join()
                    del running[index]
                    if _verified(_unit_path(out, index)):
                        completed.append(index)
                    else:
                        errs = sorted((out / "errors").glob(f"{index:06d}-*.txt"))
                        exclusions.append(
                            {
                                "index": index,
                                "exitcode": proc.exitcode,
                                "error": errs[-1].read_text()[:400] if errs else None,
                            }
                        )
                    continue
                beat = out / "heartbeats" / f"{index:06d}"
                last = max(t0, beat.stat().st_mtime - time.time() + now if beat.exists() else t0)
                hung = now - last > hang_seconds
                late = stop_event.is_set() and now - reason["at"] > STOP_GRACE_SECONDS
                if hung or late:
                    proc.terminate()
                    proc.join(10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                    del running[index]
                    exclusions.append(
                        {
                            "index": index,
                            "exitcode": proc.exitcode,
                            "error": "hang: heartbeat stale" if hung else "stop grace exceeded",
                        }
                    )
            if stop_event.is_set() and not running:
                break
            if queue or running:
                time.sleep(poll_seconds)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for proc, _ in running.values():
            if proc.is_alive():
                proc.kill()
            proc.join()
    not_started.extend(t["unit"]["index"] for t in queue)
    return reason["why"]


def run(
    contract: dict,
    ledger: Path,
    out: Path,
    workers: int,
    *,
    checkpoint_dirs: list[Path] | None = None,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
    session_seconds: float | None = None,
    foreign: str = "refuse",
    isolate: bool = True,
    runner: str = DEFAULT_RUNNER,
) -> dict:
    """Skip verified sessions, checkpoint interrupted ones, and reap every owned process."""
    expected = allocation(
        contract["namespace"], contract["count"], contract["controller"], contract["horizon"]
    )
    if (
        contract != {**expected, "commit": contract.get("commit")}
        or len(contract.get("commit", "")) != 40
    ):
        raise PackageRefused("malformed frozen replay contract")
    if type(workers) is not int or not 1 <= workers <= min(os.cpu_count() or 1, 48):
        raise ValueError("invalid worker count")
    if not math.isfinite(checkpoint_seconds) or not 0 < checkpoint_seconds <= 1800:
        raise ValueError("checkpoint interval must be in (0, 1800] seconds")
    if session_seconds is not None and (not math.isfinite(session_seconds) or session_seconds <= 0):
        raise ValueError("session seconds must be positive and finite")
    verified = completed(ledger, contract)
    skip = {row["index"] for row in verified}
    out.mkdir(parents=True, exist_ok=True)
    binding = out / "contract.json"
    if binding.exists():
        verify_immutable(binding)
        if json.loads(binding.read_text()) != contract:
            raise PackageRefused("output belongs to a different contract")
    else:
        publish_json(binding, contract)
    for unit in contract["rows"]:
        path = _unit_path(out, unit["index"])
        if _verified(path):
            header = peek_header(path, product=contract["horizon"] == 6000)
            if (
                header.get("allocation") != identity(contract)
                or header.get("unit_id")
                != unit_id(contract["namespace"], "sessions", unit["index"])
                or header.get("seed") != unit["seed"]
                or header.get("controller") != CONTROLLER
            ):
                raise PackageRefused("foreign completed replay")
            skip.add(unit["index"])
        elif path.exists() or Path(str(path) + ".complete").exists():
            raise PackageRefused("partial replay publication; use a fresh attempt output")
    pending = [u for u in contract["rows"] if u["index"] not in skip]
    dirs = [out / "checkpoints", *(checkpoint_dirs or [])]
    resumable = {
        u["index"]
        for u in pending
        if checkpoint_candidates(dirs, u["index"], contract, foreign=foreign)
    }
    pending.sort(key=lambda u: (u["index"] not in resumable, u["index"]))
    tasks = [
        {
            "contract": contract,
            "unit": u,
            "out": str(out),
            "checkpoint_dirs": [str(p) for p in (checkpoint_dirs or [])],
            "checkpoint_seconds": checkpoint_seconds,
            "runner": runner,
            "isolated": isolate,
            "foreign": foreign,
        }
        for u in pending
    ]
    done, exclusions, not_started = [], [], []
    started = time.monotonic()
    if isolate:
        stopped = _schedule(
            tasks,
            workers,
            out,
            done,
            exclusions,
            not_started,
            session_seconds,
            started,
            HANG_SECONDS,
            0.2,
        )
    else:
        # Small foreground fixtures only; operational jobs always use the bounded scheduler.
        if workers != 1 or session_seconds is not None:
            raise ValueError("nonisolated execution is only for unbudgeted single-worker fixtures")
        stopped = None
        for task in tasks:
            done.append(screen.work(task)["index"])
    result = {
        "allocation": identity(contract),
        "completed": len(done),
        "completed_now": sorted(done),
        "skipped": sorted(skip),
        "total": contract["count"],
        "exclusions": sorted(exclusions, key=lambda r: r["index"]),
        "not_started": sorted(not_started),
        "stopped": stopped,
        "elapsed_seconds": time.monotonic() - started,
    }
    number = 0
    while (out / f"operational-{number:03d}.json").exists():
        number += 1
    publish_json(out / f"operational-{number:03d}.json", result)
    if not (out / "operational.json").exists():
        publish_json(out / "operational.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=44)
    parser.add_argument("--checkpoint-in", type=Path, nargs="*", default=[])
    parser.add_argument("--checkpoint-seconds", type=float, default=DEFAULT_CHECKPOINT_SECONDS)
    parser.add_argument("--session-seconds", type=float)
    parser.add_argument("--foreign-checkpoints", choices=("refuse", "skip"), default="refuse")
    args = parser.parse_args()
    result = run(
        json.loads(args.contract.read_text()),
        args.ledger,
        args.out,
        args.workers,
        checkpoint_dirs=args.checkpoint_in,
        checkpoint_seconds=args.checkpoint_seconds,
        session_seconds=args.session_seconds,
        foreign=args.foreign_checkpoints,
    )
    print(json.dumps(result, sort_keys=True))
    if result["completed"] + len(result["skipped"]) != result["total"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
