"""Finite CPU-heavy L4 runtime. Remote summaries stay sealed; no remote decision look."""

from __future__ import annotations

import argparse
import concurrent.futures
import faulthandler
import json
import multiprocessing
import signal
import time
from pathlib import Path

import psutil

from research.bl_served_search.exports import export_json
from research.watch_lead_p.engine import run_arm, run_pair
from research.watch_lead_p.protocol import ARMS, FIXTURE_SEED, NAMESPACE, SEEDS
from research.watch_lead_p.resume import (
    ALLOCATION,
    aggregate,
    make_ledger,
    read_ledger,
    recipe_fingerprint,
    unit_id,
)
from research.watch_lead_p.uploaded import uploaded_checkpoints

WORKERS = 12
# Longest arm first within each cohort; complete sessions accrue throughout the run.
ARM_ORDER = ("P0", "P48", "bare")


def task_schedule(pending: list[int], workers: int = WORKERS) -> list[tuple[int, str]]:
    """Queue cohort-sized groups of sessions with their three arms kept together."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    return [
        (index, arm)
        for start in range(0, len(pending), workers)
        for arm in ARM_ORDER
        for index in pending[start : start + workers]
    ]


def worker(index: int, arm: str) -> dict:
    """Compute one deterministic arm; only the coordinator publishes full units."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    faulthandler.dump_traceback_later(900, exit=True)

    def pulse(_progress: dict) -> None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(900, exit=True)

    try:
        with uploaded_checkpoints():
            return run_arm(SEEDS[index], arm, pulse=pulse)
    finally:
        faulthandler.cancel_dump_traceback_later()


def publish(index: int, arms: dict, output: Path) -> dict:
    """Atomically publish exactly one complete, matched three-arm resume unit."""
    if set(arms) != set(ARMS) or len({r["initial_state"] for r in arms.values()}) != 1:
        raise ValueError("paired initial world mismatch/incomplete arms")
    row = dict(
        namespace=NAMESPACE,
        seed=SEEDS[index],
        arms=arms,
        allocation=ALLOCATION,
        allocation_index=index,
        unit_id=unit_id(index),
        recipe_sha256=recipe_fingerprint(),
    )
    path = output / "units" / f"session-{index:02d}.json"
    return {"allocation_index": index, **export_json(path, row)}


def execute(ledger: Path, inputs: Path, output: Path) -> None:
    """Skip verified sessions; assemble arm tasks into immutable whole-session units."""
    verified = read_ledger(ledger, inputs)
    pending = [i for i in range(32) if i not in verified]
    if not pending:
        raise ValueError("P completion has no missing units")
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    started = time.monotonic()
    interrupted = False
    defer_interrupt = False

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        # Keep caught dispositions during spawn: SIG_IGN survives exec into children.
        # Repeated signals are inert without changing dispositions until teardown.
        if interrupted:
            return
        interrupted = True
        if not defer_interrupt:
            raise InterruptedError("P completion units retained; manual completion only")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")
    )
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        futures = {}
        for i, arm in task_schedule(pending):
            # Finish registering an in-flight child before unwinding.
            defer_interrupt = True
            try:
                future = pool.submit(worker, i, arm)
            finally:
                defer_interrupt = False
            if interrupted:
                raise InterruptedError("P interrupted during worker submission")
            futures[future] = (i, arm)
        partial = {}
        while futures:
            done, _ = concurrent.futures.wait(futures, timeout=30, return_when="FIRST_COMPLETED")
            for future in done:
                index, arm = futures.pop(future)
                partial.setdefault(index, {})[arm] = future.result()
                if len(partial[index]) == len(ARMS):
                    # Publish payload/sidecars/marker and account for the unit as
                    # one critical section, even when a stop arrives mid-export.
                    defer_interrupt = True
                    try:
                        completed.append(publish(index, partial.pop(index), output))
                    finally:
                        # Once the last missing unit is accounted for, no more
                        # scientific work remains. Protect finalization too.
                        defer_interrupt = len(completed) == len(pending)
                    if interrupted and not defer_interrupt:
                        raise InterruptedError("P interrupted after whole-session publication")
            print(
                f"P_RUNTIME completed={len(completed)} skipped={len(verified)} "
                f"total=32 elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
    finally:
        try:
            # No future spawn is possible. A first stop during cleanup must not
            # abort reaping or mask a worker error; always restore caller handlers.
            defer_interrupt = True
            # Snapshot before shutdown clears _processes, including workers spawned by
            # a submit that was interrupted before its Future was returned.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            processes = list((pool._processes or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            # The executor manager owns waitpid/join. Concurrent Process.join calls
            # race its reaping bookkeeping; inspect OS liveness, then let it join.
            deadline = time.monotonic() + 2
            alive = processes

            def running(process: multiprocessing.Process) -> bool:
                try:
                    return psutil.Process(process.pid).status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    return False

            while alive and time.monotonic() < deadline:
                alive = [p for p in alive if running(p)]
                if alive:
                    time.sleep(0.02)
            for process in alive:
                try:
                    psutil.Process(process.pid).kill()
                except psutil.NoSuchProcess:
                    pass
            pool.shutdown(wait=True, cancel_futures=True)
            for process in processes:
                process.join(timeout=1)
                if psutil.pid_exists(process.pid):
                    raise RuntimeError("P worker reaping failed")
            if interrupted and len(completed) != len(pending):
                export_json(
                    output / "interruption.json",
                    {"completed": completed, "skipped": sorted(verified)},
                )
            if len(completed) == len(pending):
                export_json(
                    output / "inventory.json",
                    {
                        "completed": sorted(completed, key=lambda r: r["allocation_index"]),
                        "skipped": sorted(verified),
                    },
                )
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    for name in ("ledger", "aggregate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--unit", type=Path, action="append", required=True)
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        with uploaded_checkpoints(args.inputs):
            execute(args.ledger, args.inputs, args.output)
    elif args.command == "smoke":
        with uploaded_checkpoints():
            export_json(
                args.output,
                {
                    "engineering_only": True,
                    "recipe_sha256": recipe_fingerprint(),
                    **run_pair(FIXTURE_SEED, frames=60),
                },
            )
    elif args.command == "ledger":
        make_ledger(args.unit, args.output)
    else:
        print(json.dumps(aggregate(args.unit, args.output), indent=2))


if __name__ == "__main__":
    main()
