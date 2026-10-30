"""Resumable immutable paired-session units; outcomes stay sealed until Mac reduction."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import signal
import time
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from tools.ops.package_spec import publish_json, read_ledger, unit_id


def ledger_identity() -> dict:
    """Bind runtime ledger to the frozen allocation and executable science."""
    return {"science": p.identity()}


def completed(path: Path) -> list[dict]:
    """Read only identity/size/hash metadata from the Mac-verified ledger."""
    rows = read_ledger(path, p.NAMESPACE, "watch6", ledger_identity())
    if any(row["index"] not in range(p.COUNT) for row in rows):
        raise ValueError("foreign allocation index")
    return rows


def metrics(raw: dict) -> dict:
    """Retain full-horizon, all-lives session metrics and length diagnostics."""
    import numpy as np

    frames, decisions = raw["frames"], raw["decisions"]
    lengths = [f["mass"] for f in frames]
    actions = [d["chosen"] for d in decisions]
    for start in range(max(0, len(actions) - 99999)):
        window = actions[start : start + 100000]
        if len(window) == 100000 and max(window.count(a) for a in range(6)) > 90000:
            raise ValueError("unconditional action-collapse fault")
    return {
        "frames": len(frames),
        "mass_sum": sum(lengths),
        "deaths": sum(f["death"] for f in frames),
        "decisions": len(decisions),
        "boost_decisions": sum(a >= 3 for a in actions),
        "length_median": float(np.median(lengths)),
        "length_p95": float(np.percentile(lengths, 95)),
        "late_large_body_deaths": sum(f["death"] and f["death_length"] >= 800 for f in frames),
        "death_lengths": [f["death_length"] for f in frames if f["death"]],
        "batches": len(raw["batches"]),
        "interventions": sum(d["bare"] != d["chosen"] for d in decisions),
        "branches_evaluated": sum(b["branches_evaluated"] for b in raw["batches"]),
        "branches_skipped": sum(b["branches_skipped"] for b in raw["batches"]),
        "mirror_identity_frames": raw["mirror_identity_frames"],
    }


def pair(row: dict, fixture: bool = False) -> dict:
    """Run one whole pair in a private process, with the same initial seed."""
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

    arms = {}
    # Fixed alternating order avoids consistently charging one arm cold starts.
    try:
        for dense in ([False, True] if row["index"] % 2 == 0 else [True, False]):
            arms["exact_stop" if dense else "bare"] = metrics(
                run_arm(row["seed"], dense, progress_check, row["horizon"])
            )
    finally:
        # An idle pool worker must not exit900s after finishing its pair while
        # a different worker is still searching. The next pair arms its own timer.
        faulthandler.cancel_dump_traceback_later()
    return {
        "schema": "watch6-search-pair/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "allocation": row,
        "fixture": fixture,
        "unit_id": unit_id(p.NAMESPACE, "watch6", row["index"]),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-horizon", type=int, default=12)
    args = parser.parse_args()
    if args.workers not in (1, 2, *p.WORKERS) or args.shards not in (3, 6, 12, 24, 48, 96):
        parser.error("requires W1/W2/W4/W6/W8 and registered shard count")
    if args.shard not in range(args.shards):
        parser.error("foreign shard index")
    verified = completed(args.ledger)
    if args.fixture and not 1 <= args.fixture_horizon <= 14400:
        parser.error("fixture requires bounded horizon")
    if not args.fixture:
        from research.watch6_search_efficacy.admission import runtime_admit

        runtime_admit(workers=args.workers, shards=args.shards, shard=args.shard)
    args.out.mkdir(parents=True, exist_ok=False)
    skip = {row["index"] for row in verified}
    rows = (
        [
            {"index": i, "seed": seed, "horizon": args.fixture_horizon}
            for i, seed in enumerate(p.FIXTURE_SEEDS)
        ]
        if args.fixture
        else [
            r
            for r in p.allocation()
            if r["index"] % args.shards == args.shard and r["index"] not in skip
        ]
    )
    started = time.monotonic()
    # SIGUSR1 arrives at the supervisor's 15-second checkpoint grace. Completed
    # pairs are already immutable; abandon the active partial pair, never export it.
    signal.signal(signal.SIGUSR1, lambda *_: (_ for _ in ()).throw(SystemExit(75)))
    with concurrent.futures.ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {pool.submit(pair, row, args.fixture): row for row in rows}
        done_count = 0
        try:
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, timeout=60, return_when=concurrent.futures.FIRST_COMPLETED
                )
                print(
                    f"WATCH6_PROGRESS done={done_count}/{len(rows)} "
                    f"elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
                for future in done:
                    row = pending.pop(future)
                    result = future.result()
                    descriptor = publish_json(args.out / f'pair-{row["index"]:02d}.json', result)
                    publish_json(
                        args.out / f'pair-{row["index"]:02d}.meta.json',
                        {
                            "namespace": p.NAMESPACE,
                            "science": p.identity(),
                            "allocation": row,
                            "sha256": descriptor["sha256"],
                            "size": descriptor["size"],
                        },
                    )
                    done_count += 1
        except BaseException:
            for child in pool._processes.values():
                child.terminate()
            raise
    publish_json(
        args.out / "operational.json",
        {
            "completed": done_count,
            "skipped": sorted(skip),
            "fixture": args.fixture,
            "elapsed_seconds": time.monotonic() - started,
            "science": p.identity(),
        },
    )


if __name__ == "__main__":
    main()
