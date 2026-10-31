"""Execute resumable 3-arm Watch6-v2 exploration units and publish immutable exports."""

from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from research.watch6_alllen_screen import protocol as p
from tools.ops.package_spec import publish_json, read_ledger, unit_id
from web.backend.watch_replay import B8_SHA, FRP3_SHA

_STOP = None


def initialize_worker(stop_event: object) -> None:
    """Share the stop fence across queued and currently running worker units."""
    global _STOP
    _STOP = stop_event


def completed(ledger: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and validate verified completed unit indices from the uploaded ledger."""
    rows = read_ledger(ledger, contract["namespace"], p.UNIT_KIND, p.contract_identity(contract))
    if any(not 0 <= row["index"] < contract["count"] for row in rows):
        raise ValueError("foreign allocation index in ledger")
    return rows


def validate_contract(contract: dict[str, Any]) -> None:
    """Ensure the runtime contract matches the deterministic protocol constructor."""
    if "commit" not in contract or not isinstance(contract["commit"], str):
        raise ValueError("contract requires commit pin")
    expected = p.allocation(
        contract["namespace"],
        contract["count"],
        contract["horizon"],
        audit_live_mirror=bool(contract.get("audit_live_mirror", False)),
    )
    if contract != {**expected, "commit": contract["commit"]}:
        raise ValueError("malformed frozen exploration contract")


def generate(task: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    """Run all three arms on one world seed and publish one immutable unit-XXXXXX.json."""
    from research.watch6_alllen_screen.engine import run_arm

    contract, row, directory = task
    out = Path(directory)
    index = int(row["index"])
    seed = int(row["seed"])
    path = out / f"unit-{index:06d}.json"
    if path.exists() or Path(str(path) + ".complete").exists():
        raise FileExistsError("never overwrite or recompute a completed unit")

    started = time.monotonic()
    last = started
    faulthandler.dump_traceback_later(900, exit=True)
    stop = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def check() -> None:
        nonlocal last
        if stop or (_STOP is not None and _STOP.is_set()):
            raise InterruptedError("mechanical stop; partial 3-arm unit discarded")
        now = time.monotonic()
        if now - last >= 60:
            print(
                f"watch6-alllen index={index} seed={seed} working elapsed={now - started:.1f}s",
                flush=True,
            )
            last = now

    def on_frame(_frame: int) -> None:
        # Completed main-world frame progress (never heartbeat or inner branch step)
        # renews the 900s watchdog across the 3-arm session.
        faulthandler.dump_traceback_later(900, exit=True)

    arm_order = list(p.arm_order_for_index(index))
    arms_out: dict[str, dict[str, Any]] = {}
    arm_timings: dict[str, float] = {}
    try:
        for arm in arm_order:
            faulthandler.dump_traceback_later(900, exit=True)
            arm_summary = run_arm(
                seed,
                arm,
                check,
                horizon=int(contract["horizon"]),
                audit_live_mirror=bool(contract.get("audit_live_mirror", False)),
                on_frame=on_frame,
            )
            arm_timings[arm] = float(arm_summary.pop("wall_seconds"))
            arms_out[arm] = arm_summary

        rosters = [arms_out[arm].pop("roster") for arm in p.ARMS]
        if rosters[0] != rosters[1] or rosters[0] != rosters[2]:
            raise ValueError("roster identity differed across paired arms")

        unit_payload = {
            "schema": p.UNIT_SCHEMA,
            "lane": p.LANE,
            "world": contract["world"],
            "namespace": contract["namespace"],
            "unit_id": unit_id(contract["namespace"], p.UNIT_KIND, index),
            "allocation": {
                "index": index,
                "seed": seed,
                "horizon": int(contract["horizon"]),
            },
            "contract_identity": p.contract_identity(contract),
            "source_commit": contract["commit"],
            "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
            "fixture": (
                contract["namespace"] != p.NAMESPACE or int(contract["horizon"]) != p.HORIZON
            ),
            "arm_execution_order": arm_order,
            "roster": rosters[0],
            "arms": {arm: arms_out[arm] for arm in p.ARMS},
        }
        published = publish_json(path, unit_payload)
    finally:
        faulthandler.cancel_dump_traceback_later()

    elapsed = time.monotonic() - started
    print(
        f"watch6-alllen index={index} complete seconds={elapsed:.3f} "
        f"mass(B/L600/L1)=({arms_out['bare']['mass_integral']:.1f}/"
        f"{arms_out['l2_prime_600']['mass_integral']:.1f}/"
        f"{arms_out['exact_stop_alllen']['mass_integral']:.1f})",
        flush=True,
    )
    return {
        **published,
        "index": index,
        "seed": seed,
        "seconds": elapsed,
        "arm_seconds": arm_timings,
        "summary": {
            arm: {
                "mass_integral": arms_out[arm]["mass_integral"],
                "deaths_per_1k": arms_out[arm]["deaths_per_1k"],
                "search_roots": arms_out[arm]["search_roots"],
                "interventions": arms_out[arm]["interventions"],
            }
            for arm in p.ARMS
        },
    }


def run(contract: dict[str, Any], ledger: Path, out: Path, workers: int) -> list[dict[str, Any]]:
    """Skip uploaded Mac-verified units; execute remaining units across worker pool."""
    validate_contract(contract)
    if not 1 <= workers <= min(os.cpu_count() or 1, 10):
        raise ValueError("invalid worker count")
    verified = completed(ledger, contract)
    skip = {row["index"] for row in verified}
    out.mkdir(parents=True, exist_ok=False)
    publish_json(out / "contract.json", contract)

    tasks = [(contract, row, str(out)) for row in contract["rows"] if row["index"] not in skip]
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    def collect(
        task: tuple[dict[str, Any], dict[str, Any], str],
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if error is not None:
            exclusions.append(
                {"index": task[1]["index"], "error": str(error), "kind": type(error).__name__}
            )
            print(f"watch6-alllen excluded index={task[1]['index']}: {error}", flush=True)
        elif result is not None:
            rows.append(result)

    try:
        if workers == 1:
            for task in tasks:
                try:
                    collect(task, generate(task))
                except Exception as exc:
                    collect(task, error=exc)
                    if isinstance(exc, InterruptedError):
                        break
        else:
            context = multiprocessing.get_context("spawn")
            stop_event = context.Event()
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=initialize_worker,
                initargs=(stop_event,),
            ) as pool:

                def stop_pool(signum: int, frame: object) -> None:
                    stop_event.set()

                signal.signal(signal.SIGUSR1, stop_pool)
                signal.signal(signal.SIGTERM, stop_pool)
                futures = {pool.submit(generate, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        collect(task, future.result())
                    except Exception as exc:
                        collect(task, error=exc)
    finally:
        rows.sort(key=lambda item: item["index"])
        publish_json(
            out / "operational.json",
            {
                "allocation": p.contract_identity(contract),
                "completed": len(rows),
                "skipped": sorted(skip),
                "total": contract["count"],
                "exclusions": sorted(exclusions, key=lambda item: item["index"]),
                "units": rows,
            },
        )
    if exclusions:
        raise RuntimeError(f"unit exclusions occurred: {exclusions}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    print(json.dumps(run(contract, args.ledger, args.out, args.workers), sort_keys=True))


if __name__ == "__main__":
    main()
