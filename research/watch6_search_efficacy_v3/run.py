"""Resumable immutable paired-session runner for Watch6 V3 reusing V2 engine & metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import signal
import time
from pathlib import Path

from research.watch6_search_efficacy.run import metrics
from research.watch6_search_efficacy_v3 import protocol as p
from tools.ops.package_spec import publish_json, read_ledger, unit_id


def ledger_identity() -> dict:
    """Bind runtime ledger to the frozen V3 source and primary/reserve allocations."""
    return {"science": p.identity()}


def completed(path: Path) -> list[dict]:
    """Read only identity/size/hash metadata from the Mac-verified V3 ledger."""
    rows = read_ledger(path, p.NAMESPACE, "watch6", ledger_identity())
    if any(row["index"] not in range(p.COUNT) for row in rows):
        raise ValueError("foreign allocation index")
    return rows


def pair(row: dict, fixture: bool = False) -> dict:
    """Run one whole paired session in a private process using V2's unchanged engine."""
    os.environ["SNAKE_WATCH_SOURCE"] = "live"
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.watch6_search_efficacy.engine import run_arm

    configure_cpu()
    import faulthandler

    faulthandler.dump_traceback_later(900, exit=True)
    last_watchdog = time.monotonic()

    def progress_check() -> None:
        nonlocal last_watchdog
        now = time.monotonic()
        if now - last_watchdog >= 60:
            faulthandler.dump_traceback_later(900, exit=True)
            last_watchdog = now

    seed_kind = row.get("seed_kind", "fixture" if fixture else "primary")
    allocation_row = {
        "index": int(row["index"]),
        "seed": int(row["seed"]),
        "horizon": int(row["horizon"]),
    }
    arms = {}
    try:
        for dense in ([False, True] if allocation_row["index"] % 2 == 0 else [True, False]):
            arms["exact_stop" if dense else "bare"] = metrics(
                run_arm(
                    allocation_row["seed"],
                    dense,
                    progress_check,
                    allocation_row["horizon"],
                )
            )
    finally:
        faulthandler.cancel_dump_traceback_later()
    return {
        "schema": "watch6-search-pair/v3",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "seed_kind": seed_kind,
        "allocation": allocation_row,
        "fixture": fixture,
        "unit_id": (
            f"fixture-{allocation_row['index']:02d}"
            if fixture
            else unit_id(p.NAMESPACE, "watch6", allocation_row["index"])
        ),
        "arms": arms,
    }


def run_canary_check(out: Path) -> None:
    """Short L4 canary check (~4s): 1 long-body H300 exact-stop batch with native v8 counters."""
    os.environ["SNAKE_WATCH_SOURCE"] = "live"
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.watch6_search_efficacy.performance import branch_fixture

    configure_cpu()
    branch = branch_fixture(p.FIXTURE_SEEDS[0])
    counters = branch["native_v8"]
    if (
        300 not in branch["frames_executed"]
        or counters["kernel_rows"] <= 0
        or any(counters[k] != 0 for k in ("stock_rows", "fallback_rows", "noncandidate_rows"))
    ):
        raise ValueError("canary long-body H300 native-v8 check failed")
    publish_json(
        out,
        {
            "schema": "watch6-v3-canary-performance/v1",
            "status": "PASS",
            "science": p.identity(),
            "cpu_count": os.cpu_count(),
            "fixture": True,
            "branch_check": branch,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--workers", type=int, default=p.WORKERS)
    parser.add_argument("--shards", type=int, default=p.SHARDS)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-horizon", type=int, default=12)
    parser.add_argument("--canary-performance-out", type=Path, default=None)
    args = parser.parse_args()

    os.environ["SNAKE_WATCH_SOURCE"] = "live"
    if args.workers not in (1, 2, p.WORKERS) or args.shards != p.SHARDS:
        parser.error("requires W1/W2/W4 and registered 24-shard V3 layout")
    if args.shard not in range(args.shards):
        parser.error("foreign shard index")
    verified = completed(args.ledger)
    if args.fixture and not 1 <= args.fixture_horizon <= p.HORIZON:
        parser.error("fixture requires bounded horizon")

    reserve_set: set[int] = set()
    if not args.fixture:
        if args.workers != p.WORKERS:
            parser.error("confirmation shards require W4")
        from research.watch6_search_efficacy_v3.admission import runtime_admit

        contract = runtime_admit(workers=args.workers, shards=args.shards, shard=args.shard)
        reserve_set = set(contract.get("reserve_slots", []))

    args.out.mkdir(parents=True, exist_ok=False)
    skip = {row["index"] for row in verified}
    if args.fixture:
        rows = [
            {"index": i, "seed": seed, "horizon": args.fixture_horizon, "seed_kind": "fixture"}
            for i, seed in enumerate(p.FIXTURE_SEEDS)
        ]
    else:
        primary_rows = p.allocation()
        reserve_rows = p.reserve_allocation()
        rows = []
        for idx in p.shard_indices(args.shard, args.shards):
            if idx in skip:
                continue
            kind = "reserve" if idx in reserve_set else "primary"
            base = reserve_rows[idx] if kind == "reserve" else primary_rows[idx]
            rows.append({**base, "seed_kind": kind})

    started = time.monotonic()
    signal.signal(signal.SIGUSR1, lambda *_: (_ for _ in ()).throw(SystemExit(75)))
    prefix = "fixture-pair" if args.fixture else "pair"

    with concurrent.futures.ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        print(
            f"WATCH6_V3_PROGRESS shard={args.shard} done=0/{len(rows)} "
            f"skipped={len(skip)} elapsed=0.0s",
            flush=True,
        )
        pending = {pool.submit(pair, row, args.fixture): row for row in rows}
        done_count = 0
        try:
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, timeout=60, return_when=concurrent.futures.FIRST_COMPLETED
                )
                print(
                    f"WATCH6_V3_PROGRESS shard={args.shard} done={done_count}/{len(rows)} "
                    f"skipped={len(skip)} elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
                for future in done:
                    row = pending.pop(future)
                    result = future.result()
                    stem = f'{prefix}-{row["index"]:02d}'
                    descriptor = publish_json(args.out / f"{stem}.json", result)
                    publish_json(
                        args.out / f"{stem}.meta.json",
                        {
                            "namespace": p.NAMESPACE,
                            "science": p.identity(),
                            "seed_kind": result["seed_kind"],
                            "allocation": result["allocation"],
                            "fixture": args.fixture,
                            "sha256": descriptor["sha256"],
                            "size": descriptor["size"],
                        },
                    )
                    done_count += 1
        except BaseException:
            for child in pool._processes.values():
                child.terminate()
            raise

    if args.canary_performance_out is not None:
        if not args.fixture:
            raise ValueError("--canary-performance-out is valid only with --fixture")
        run_canary_check(args.canary_performance_out)

    publish_json(
        args.out / "operational.json",
        {
            "schema": "watch6-v3-operational/v1",
            "namespace": p.NAMESPACE,
            "shard": args.shard,
            "shards": args.shards,
            "workers": args.workers,
            "completed": done_count,
            "skipped": sorted(i for i in skip if i % args.shards == args.shard),
            "skipped_total_ledger": len(skip),
            "reserve_slots_executed": sorted(
                r["index"] for r in rows if r["seed_kind"] == "reserve"
            ),
            "fixture": args.fixture,
            "elapsed_seconds": time.monotonic() - started,
            "science": p.identity(),
            "judge": "Astra",
            "implementer": "Argon",
            "independent_implementation_review": True,
        },
    )


if __name__ == "__main__":
    main()
