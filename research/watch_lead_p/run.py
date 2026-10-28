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

from research.bl_served_search.exports import export_json
from research.bl_served_search.isolated_game import _reap_isolated_process_tree
from research.watch_lead_p.engine import run_pair
from research.watch_lead_p.protocol import FIXTURE_SEED, SEEDS
from research.watch_lead_p.resume import (
    ALLOCATION,
    aggregate,
    make_ledger,
    read_ledger,
    recipe_fingerprint,
    unit_id,
)


def worker(index: int, output: str) -> dict:
    """Own one full three-arm unit in a spawned process and atomically export it."""
    faulthandler.dump_traceback_later(900, exit=True)

    def pulse(_progress: dict) -> None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(900, exit=True)

    try:
        row = run_pair(SEEDS[index], pulse=pulse)
        row.update(
            allocation=ALLOCATION,
            allocation_index=index,
            unit_id=unit_id(index),
            recipe_sha256=recipe_fingerprint(),
        )
        path = Path(output) / f"session-{index:02d}.json"
        info = export_json(path, row)
        print(f"P_UNIT_COMPLETE allocation_index={index} path={path.name}", flush=True)
        return {"allocation_index": index, **info}
    finally:
        faulthandler.cancel_dump_traceback_later()


def execute(ledger: Path, inputs: Path, output: Path) -> None:
    """Skip exactly verified units; publish an outcome-free completion inventory."""
    verified = read_ledger(ledger, inputs)
    output.mkdir(parents=True, exist_ok=False)
    pending = [i for i in range(32) if i not in verified]
    completed = []
    started = time.monotonic()

    def interrupt(_signum: int, _frame: object) -> None:
        # Complete units were already exported. No partial trajectory is reused.
        export_json(
            output / "interruption.json", {"completed": completed, "skipped": sorted(verified)}
        )
        raise InterruptedError("P completion units retained; manual completion only")

    signal.signal(signal.SIGUSR1, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=10, mp_context=multiprocessing.get_context("spawn")
    )
    owned = []
    try:
        futures = {pool.submit(worker, i, str(output / "units")): i for i in pending}
        owned = list(pool._processes.values())
        while futures:
            done, _ = concurrent.futures.wait(futures, timeout=30, return_when="FIRST_COMPLETED")
            for future in done:
                completed.append(future.result())
                del futures[future]
            print(
                f"P_RUNTIME completed={len(completed)} skipped={len(verified)} "
                f"total=32 elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for process in owned:
            if process.is_alive():
                process.terminate()
        for process in owned:
            process.join(timeout=0.1)
            _reap_isolated_process_tree(process)
    export_json(
        output / "inventory.json",
        {
            "completed": sorted(completed, key=lambda r: r["allocation_index"]),
            "skipped": sorted(verified),
        },
    )


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
        execute(args.ledger, args.inputs, args.output)
    elif args.command == "smoke":
        export_json(args.output, {"engineering_only": True, **run_pair(FIXTURE_SEED, frames=60)})
    elif args.command == "ledger":
        make_ledger(args.unit, args.output)
    else:
        print(json.dumps(aggregate(args.unit, args.output), indent=2))


if __name__ == "__main__":
    main()
