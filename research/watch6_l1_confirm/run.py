"""Paired-world atomic 3-arm execution runner for Watch6-L1-V1 ({bare, l2_prime, l1})."""

from __future__ import annotations

import argparse
import concurrent.futures
import faulthandler
import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.engine import ScientificFault, run_arm
from tools.ops.package_spec import publish_json

WATCHDOG_SECONDS = 900  # Terminates wedged worker processes cleanly without hanging the shard.
DEFAULT_LOCAL_SAVED_DIR = p.ROOT.parent / "snake-dqn/saved_snakes"


def ledger_identity() -> dict[str, str]:
    """Science identity embedded in verified-ledger.json across attempts."""
    return {"science": p.identity()}


def _arm_summary(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k != "roster"}


def branch_fixture(seed: int) -> dict:
    """Time a long-body full-H300 exact-stop batch with the shipped Watch6-v2 roster."""
    os.environ["SNAKE_WATCH_SOURCE"] = "live"
    from research.watch6_search_efficacy.performance import branch_fixture as v2_branch_fixture

    raw = v2_branch_fixture(int(seed))
    counters = raw["native_v8"]
    if (
        p.SEARCH_HORIZON not in raw["frames_executed"]
        or counters["kernel_rows"] <= 0
        or any(counters[k] != 0 for k in ("stock_rows", "fallback_rows", "noncandidate_rows"))
    ):
        raise ValueError("canary long-body H300 native-v8 check failed")
    return {
        "schema": "watch6-l1-canary-performance/v1",
        "science": p.identity(),
        "seed": int(seed),
        "fixture": True,
        "seconds": float(raw["seconds"]),
        "branch_seconds_upper": float(raw["branch_seconds_upper"]),
        "branches": int(raw["branches"]),
        "frames_executed": [int(f) for f in raw["frames_executed"]],
        "native_v8": counters,
    }


def world_unit(row: dict, fixture: bool = False) -> dict:
    """Execute all 3 arms {bare, l2_prime, l1} of one paired world sequentially on one worker."""
    os.environ["SNAKE_WATCH_SOURCE"] = "live"
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    started = time.monotonic()
    index = int(row["index"])
    seed = int(row["seed"])
    horizon = int(row["horizon"])
    arm_order = list(row.get("arm_order") or p.arm_order_for_index(index))
    if sorted(arm_order) != sorted(p.ARMS) or len(arm_order) != len(p.ARMS):
        raise ValueError(f"invalid 3-arm execution order for world {index}: {arm_order!r}")

    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)
    last_watchdog = started

    def progress_check() -> None:
        nonlocal last_watchdog
        now = time.monotonic()
        if now - last_watchdog >= 60:
            faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)
            last_watchdog = now

    try:
        arm_results: dict[str, dict] = {}
        rosters: list[dict] = []
        for arm in arm_order:
            faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)
            last_watchdog = time.monotonic()
            raw = run_arm(
                seed,
                arm,
                progress_check,
                horizon=horizon,
                record=None,
            )
            rosters.append(raw["roster"])
            arm_results[arm] = _arm_summary(raw)
    finally:
        faulthandler.cancel_dump_traceback_later()

    if rosters[0] != rosters[1] or rosters[0] != rosters[2]:
        raise ScientificFault("product Watch6 roster identity differed across paired arms")
    for arm in p.ARMS:
        if arm_results[arm]["mirror_identity_frames"] != horizon:
            raise ScientificFault(f"incomplete mirror frame coverage on arm {arm}")

    return {
        "schema": "watch6-l1-world/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "seed_kind": "fixture" if fixture else "primary",
        "index": index,
        "seed": seed,
        "horizon": horizon,
        "fixture": bool(fixture),
        "record_mode": None,
        "arm_order": arm_order,
        "roster": rosters[0],
        "arms": {arm: arm_results[arm] for arm in p.ARMS},
        "elapsed_seconds": time.monotonic() - started,
    }


def _publish_world_unit(out: Path, row: dict, payload: dict, *, fixture: bool) -> dict:
    prefix = "fixture-world" if fixture else "world"
    target = out / f"{prefix}-{int(row['index']):02d}.json"
    descriptor = publish_json(target, payload)
    publish_json(
        target.with_suffix(".meta.json"),
        {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "seed_kind": "fixture" if fixture else "primary",
            "allocation": row,
            "fixture": bool(fixture),
            "sha256": descriptor["sha256"],
            "size": descriptor["size"],
        },
    )
    return descriptor


def run(
    *,
    out: Path,
    ledger_path: Path | None,
    workers: int,
    shards: int = p.SHARDS,
    shard: int = 0,
    fixture: bool = False,
    fixture_horizon: int = 120,
    fixture_count: int = 2,
    canary_performance_out: Path | None = None,
    local_smoke: bool = False,
) -> dict:
    """Execute one Watch6-L1-V1 shard, Colab canary fixture, or local Mac smoke."""
    if local_smoke:
        if not fixture:
            raise ValueError("--local-smoke only runs excluded operational fixture seeds")
        if "SNAKE_CKPT_DIR" not in os.environ and DEFAULT_LOCAL_SAVED_DIR.is_dir():
            os.environ["SNAKE_CKPT_DIR"] = str(DEFAULT_LOCAL_SAVED_DIR)
        os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")
        skip: set[int] = set()
    else:
        from research.watch6_l1_confirm.admission import completed_ledger, runtime_admit

        runtime_admit(workers=workers, shards=shards, shard=shard)
        if ledger_path is None:
            raise ValueError("admitted runtime execution requires --ledger")
        verified = completed_ledger(ledger_path)
        skip = set() if fixture else {row["index"] for row in verified}

    if fixture:
        if type(fixture_count) is not int or not 1 <= fixture_count <= len(p.FIXTURE_SEEDS):
            raise ValueError(f"fixture_count must be in 1..{len(p.FIXTURE_SEEDS)}")
        if type(fixture_horizon) is not int or fixture_horizon <= 0:
            raise ValueError("fixture_horizon must be a positive integer")
        rows = [
            {
                "index": i,
                "seed": p.FIXTURE_SEEDS[i],
                "horizon": fixture_horizon,
                "arm_order": p.arm_order_for_index(i),
            }
            for i in range(fixture_count)
        ]
    else:
        alloc = p.allocation()
        owned = p.shard_indices(shard, shards)
        rows = [alloc[i] for i in owned if i not in skip]

    started = time.monotonic()
    last_log = started
    completed_now: list[int] = []
    out.mkdir(parents=True, exist_ok=True)
    publish_json(
        out.parent / "workload-started.json",
        {
            "schema": "watch6-l1-workload-started/v1",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "shard": shard,
            "shards": shards,
            "workers": workers,
            "fixture": bool(fixture),
            "local_smoke": bool(local_smoke),
            "pending_units": len(rows),
            "skipped_verified_total": len(skip),
        },
    )

    signal.signal(signal.SIGUSR1, lambda *_: (_ for _ in ()).throw(SystemExit(75)))
    context = multiprocessing.get_context("spawn")
    if rows:
        pool_workers = min(workers, len(rows))
        with concurrent.futures.ProcessPoolExecutor(pool_workers, mp_context=context) as pool:
            pending = {pool.submit(world_unit, row, fixture): row for row in rows}
            try:
                while pending:
                    done, _ = concurrent.futures.wait(
                        pending,
                        timeout=30,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    now = time.monotonic()
                    if not done:
                        if now - last_log >= 120:
                            print(
                                f"WATCH6_L1_PROGRESS shard={shard}/{shards} "
                                f"done={len(completed_now)}/{len(rows)} "
                                f"skipped_total={len(skip)} elapsed_s={now - started:.1f}",
                                flush=True,
                            )
                            last_log = now
                        continue
                    for future in done:
                        row = pending.pop(future)
                        payload = future.result()
                        _publish_world_unit(out, row, payload, fixture=fixture)
                        completed_now.append(int(row["index"]))
                        print(
                            f"WATCH6_L1_UNIT index={row['index']:02d} fixture={fixture} "
                            f"horizon={row['horizon']} order={','.join(row['arm_order'])} "
                            f"unit_s={payload['elapsed_seconds']:.1f} "
                            f"done={len(completed_now)}/{len(rows)}",
                            flush=True,
                        )
                        last_log = time.monotonic()
            except BaseException:
                for child in pool._processes.values():
                    child.terminate()
                raise

    if fixture and canary_performance_out is not None:
        perf = branch_fixture(p.FIXTURE_SEEDS[0])
        publish_json(canary_performance_out, perf)

    summary = {
        "schema": "watch6-l1-operational/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "fixture": bool(fixture),
        "local_smoke": bool(local_smoke),
        "shard": shard,
        "shards": shards,
        "workers": workers,
        "skipped_verified_total": len(skip),
        "completed_indices": sorted(completed_now),
        "elapsed_seconds": time.monotonic() - started,
    }
    publish_json(out / "operational.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shards", type=int, default=p.SHARDS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-horizon", type=int, default=120)
    parser.add_argument("--fixture-count", type=int, default=2)
    parser.add_argument("--canary-performance-out", type=Path, default=None)
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            out=args.out.resolve(),
            ledger_path=args.ledger.resolve() if args.ledger is not None else None,
            workers=args.workers,
            shards=args.shards,
            shard=args.shard,
            fixture=args.fixture or args.local_smoke,
            fixture_horizon=args.fixture_horizon,
            fixture_count=args.fixture_count,
            canary_performance_out=(
                args.canary_performance_out.resolve()
                if args.canary_performance_out is not None
                else None
            ),
            local_smoke=args.local_smoke,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    except ScientificFault as exc:
        print(f"WATCH6_L1_SCIENTIFIC_FAULT: {exc}", file=sys.stderr, flush=True)
        sys.exit(86)


if __name__ == "__main__":
    main()
