"""Resumable successor session shards; engineering uses the same publisher/runtime."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import queue
import signal
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from research.bl_served_search.exports import digest, export_json
from research.watch_lead_p_successor.units import (
    ARMS,
    aggregate,
    make_ledger,
    publish,
    read_ledger,
    sha,
    unit_id,
    validate_binding,
)


def host_identity() -> dict:
    """Pair CPU/software inside a session; never assert cross-platform action identity."""
    import numpy as np
    import torch

    from research.watch_lead_p_successor.profile import host_record

    record = host_record()
    models = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in (record.get("cpuinfo") or "").splitlines()
            if line.startswith("model name")
        }
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": models or [platform.processor()],
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(),
    }


def run_session(
    bound: dict, index: int, checkpoints: Path, *, smoke_frames: int | None = None
) -> tuple[dict, dict]:
    """All three natural-start arms execute in one process on one host/software."""
    from research.watch_lead_p import engine
    from research.watch_lead_p.live import LiveWorld
    from research.watch_lead_p.uploaded import uploaded_checkpoints

    bank = bound["bank"]
    unit_id(bound, index)
    if smoke_frames is not None and (not bank["engineering_only"] or not 1 <= smoke_frames <= 60):
        raise ValueError("short smoke is engineering-only and never a resume unit")
    frames = smoke_frames or bank["frames"]
    arms, timing = {}, {}
    with uploaded_checkpoints(checkpoints), patch.object(engine, "NAMESPACE", bank["namespace"]):
        for arm in ARMS:
            construction = finalization = search_seconds = 0.0
            searches = 0
            chunks, lengths, chunk = [], [], time.monotonic()
            original_init, original_digest, original_search = (
                LiveWorld.__init__,
                LiveWorld.digest,
                engine.search,
            )
            original_submit = engine.submit
            observed_host = None

            def init(world: Any, seed: int) -> None:
                nonlocal construction, observed_host
                started = time.monotonic()
                original_init(world, seed)
                construction += time.monotonic() - started
                observed_host = host_identity()

            def state_digest(world: Any) -> str:
                nonlocal finalization
                started = time.monotonic()
                value = original_digest(world)
                if world.frame == frames:
                    finalization += time.monotonic() - started
                return value

            def search(*args: Any, **kwargs: Any) -> Any:
                nonlocal searches, search_seconds
                started = time.monotonic()
                result = original_search(*args, **kwargs)
                search_seconds += time.monotonic() - started
                searches += 1
                return result

            def submit(world: Any, *args: Any, **kwargs: Any) -> Any:
                started = time.monotonic()
                root = {"frame": world.frame, "length": int(world.hero.length)}
                result = original_submit(world, *args, **kwargs)
                lengths.append({**root, "inclusive_seconds": time.monotonic() - started})
                return result

            def pulse(progress: dict) -> None:
                nonlocal chunk
                if progress["frame"] % 1000 == 0:
                    now = time.monotonic()
                    chunks.append({"frame": progress["frame"], "seconds": now - chunk})
                    chunk = now

            # Instrumentation observes work; engine, masks, root spacing and stopping are unchanged.
            started, cpu = time.monotonic(), time.process_time()
            with (
                patch.object(LiveWorld, "__init__", init),
                patch.object(LiveWorld, "digest", state_digest),
                patch.object(engine, "search", search),
                patch.object(engine, "submit", submit),
            ):
                row = engine.run_arm(bank["seeds"][index], arm, frames=frames, pulse=pulse)
            inclusive = time.monotonic() - started
            if (
                observed_host is None
                or observed_host["torch_threads"] != 1
                or observed_host["torch_interop"] != 1
            ):
                raise ValueError("single-thread CPU host identity required")
            # Root length is workload support only; no deaths/mass/branch outcomes are exported.
            arms[arm] = {**row, "host": observed_host}
            timing[arm] = {
                "frames": frames,
                "inclusive_seconds": inclusive,
                "cpu_seconds": time.process_time() - cpu,
                "constructor_seconds": construction,
                "final_digest_seconds": finalization,
                "search_calls": searches,
                "search_seconds": search_seconds,
                "chunks": chunks,
                "root_lengths": lengths,
                **{k: row[k] for k in ("initial_state", "final_state", "trace_sha256")},
            }
    hosts = {sha(r["host"]) for r in arms.values()}
    if len(hosts) != 1 or len({r["initial_state"] for r in arms.values()}) != 1:
        raise ValueError("session host/software/initial pairing differs")
    return (
        {
            "binding": bound,
            "index": index,
            "unit_id": unit_id(bound, index),
            "host": arms["bare"]["host"],
            "arms": arms,
        },
        {
            "index": index,
            "host": arms["bare"]["host"],
            "arms": timing,
            "engineering_only": bank["engineering_only"],
            "smoke": smoke_frames is not None,
        },
    )


def worker(tasks: Any, results: Any, bound: dict, checkpoints: Path) -> None:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGUSR1})
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    while True:
        i = tasks.get()
        try:
            results.put((i, True, run_session(bound, i, checkpoints)))
        except BaseException as exc:
            results.put((i, False, repr(exc)))
            return


def execute(
    bound: dict,
    ledger: dict,
    indices: list[int],
    checkpoints: Path,
    output: Path,
    *,
    workers: int,
    work_seconds: float,
) -> dict:
    """Retain finished sessions at deadline; discard interrupted sessions whole."""
    from research.watch_lead_p_successor.profile import reap

    validate_binding(bound)
    if not bound["bank"]["engineering_only"]:
        raise ValueError("scientific execution awaits prospective ratification/admission")
    verified = read_ledger(ledger, bound)
    if type(workers) is not int or not 1 <= workers <= 11:
        raise ValueError("workers must be frozen in 1..11")
    if not 0 < work_seconds < 24 * 3600 - 70 or len(indices) != len(set(indices)) or not indices:
        raise ValueError("invalid shard/deadline")
    for i in indices:
        unit_id(bound, i)
    pending = [i for i in indices if i not in verified]
    if not pending:
        raise ValueError("no missing units; verified sessions must not be recomputed")
    output.mkdir(parents=True, exist_ok=False)
    export_json(
        output / "entrypoint.json", {"binding": bound, "verified_indices": sorted(verified)}
    )
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    processes, completed = [], []
    started = time.monotonic()
    deadline = started + work_seconds
    stopping, status = False, "INCOMPLETE"

    def stop(_sig: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True  # Never interrupt marker-last publication or worker registration.

    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        for _ in range(min(workers, len(pending))):
            p = ctx.Process(target=worker, args=(tasks, results, bound, checkpoints))
            processes.append(p)
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set(old))
            try:
                p.start()
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        for i in pending:
            tasks.put(i)  # Session FIFO: P0/P48/bare retire together before next task.
        while len(completed) < len(pending) and not stopping and time.monotonic() < deadline:
            try:
                i, ok, value = results.get(timeout=min(1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                if any(p.exitcode is not None for p in processes):
                    raise RuntimeError("session worker exited before completion")
                continue
            if not ok:
                raise RuntimeError(value)
            row, timing = value
            if i not in pending or i in completed:
                raise ValueError("foreign/duplicate worker completion")
            publish(output / "units" / f"session-{i:02d}.json", row, bound, i)
            export_json(output / "timing" / f"session-{i:02d}.json", timing)
            completed.append(i)
            print(
                f"P_SUCCESSOR completed={len(completed)} skipped={len(verified)} "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
        status = "COMPLETE" if len(completed) == len(pending) else "INCOMPLETE"
    finally:
        try:
            reap(processes)  # Only children created above; never external jobs.
            inventory = {
                "binding": bound,
                "status": status,
                "completed": sorted(completed),
                "skipped": sorted(set(indices) & set(verified)),
                "indices": indices,
                "elapsed_seconds": time.monotonic() - started,
                "manual_completion_only": True,
            }
            export_json(output / "inventory.json", inventory)
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
            tasks.close()
            results.close()
    return inventory


def verify_sources(root: Path, sources: dict[str, str], bound: dict) -> None:
    if sha(sources) != bound["source_sha256"]:
        raise ValueError("source inventory binding differs")
    for name, checksum in sources.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or digest(root / path) != checksum:
            raise ValueError(f"runtime source drift: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "ledger", "aggregate"))
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--indices", default="0,1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--work-seconds", type=float, default=3600)
    parser.add_argument("--unit", type=Path, action="append", default=[])
    args = parser.parse_args()
    bound = json.loads(args.binding.read_text())
    validate_binding(bound)
    if args.command == "ledger":
        export_json(args.output, make_ledger(args.unit, bound))
    elif args.command == "aggregate":
        aggregate(args.unit, bound, args.output)
    elif args.command == "smoke":
        verify_sources(
            Path(__file__).resolve().parents[2], json.loads(args.sources.read_text()), bound
        )
        verified = read_ledger(json.loads(args.ledger.read_text()), bound)
        export_json(
            args.output.parent / "entrypoint.json",
            {"binding": bound, "verified_indices": sorted(verified)},
        )
        _, timing = run_session(bound, 0, args.checkpoints, smoke_frames=3)
        export_json(args.output, timing)  # Short smoke NEVER publishes a full-session unit.
    else:
        verify_sources(
            Path(__file__).resolve().parents[2], json.loads(args.sources.read_text()), bound
        )
        inventory = execute(
            bound,
            json.loads(args.ledger.read_text()),
            [int(i) for i in args.indices.split(",")],
            args.checkpoints,
            args.output,
            workers=args.workers,
            work_seconds=args.work_seconds,
        )
        if inventory["status"] != "COMPLETE":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
