"""Claude-operated Thursday W1 timing, frozen >=5x opponent_v8 decision."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from research.bl_k1.prepare import verify_manifest
from research.bl_mac_latency.host import check_guard, process_screen, snapshot
from research.bl_mac_latency.ruler_run import distribution
from research.bl_served_search.exports import export_json, verify_export
from research.bl_served_search.ruler_operator import ExactT3Operator


def decision(stock: list[float], kernel: list[float], identity_pass: bool) -> dict[str, Any]:
    """The prospective primary bar; partial/cold-only populations cannot pass."""
    if len(stock) != 30 or len(kernel) != 30 or not identity_pass:
        return {"status": "NOT_READY", "speedup": None}
    if not all(__import__("math").isfinite(x) and x > 0 for x in stock + kernel):
        raise ValueError("non-positive/non-finite stage timer")
    ratio = statistics.median(stock[1:]) / statistics.median(kernel[1:])
    return {
        "status": "K1_PASS" if ratio >= 5.0 else "EXACT_SERVED_SEARCH_CLOSED_WEEK2",
        "speedup": ratio,
        "minimum_speedup": 5.0,
        "stage": "opponent_v8",
        "fixture_id": "frozen-1500",
        "mode": "kernel",
        "wp4": "ADMITTED_BUILD_ONLY" if ratio >= 5 else "CANCELLED",
    }


def window() -> None:
    now = dt.datetime.now(ZoneInfo("America/Los_Angeles"))
    if now.date().isoformat() != "2026-10-22" or not 20 <= now.hour < 22:
        raise RuntimeError("K-1 timing is only Thu 2026-10-22 20:00-22:00 PT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_manifest(args.manifest)
    from research.bl_k1.audit import verify_identity

    manifest_hash = verify_export(args.manifest)
    identity_hash = verify_identity(args.identity, manifest, manifest_hash)
    window()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("timing.json", "stopped.json", "idle.json", "batches"):
        if (args.out / name).exists():
            raise FileExistsError(f"fresh timing attempt required: {args.out / name}")
    (args.out / "batches").mkdir()
    batches = []
    started = time.monotonic()
    try:
        before = snapshot()
        export_json(args.out / "idle.json", before)
        if not before["quiet"]:
            raise RuntimeError("initial host not idle; retained, never kill daemons")
        operators = {mode: ExactT3Operator(mode) for mode in ("stock", "replay", "kernel")}
        # Alternation limits drift; each mode still has its own first/cold sample.
        for fixture in manifest["fixtures"]:
            expected = None
            for repeat in range(30):
                for mode in manifest["timing"]["mode_order"]:
                    window()
                    if time.monotonic() - started >= manifest["timing"]["max_seconds"]:
                        raise RuntimeError("timing budget exhausted")
                    guard = {**check_guard(), "process_screen": process_screen()}
                    result = operators[mode].batch(fixture)
                    if expected is not None and result["semantic"] != expected:
                        export_json(
                            args.out / "invalid.json",
                            {
                                "status": "INVALID",
                                "mode": mode,
                                "fixture_id": fixture["fixture_id"],
                                "result": result,
                            },
                        )
                        raise RuntimeError(
                            "action/outcome mismatch: mode INVALID; stop exact search"
                        )
                    expected = result["semantic"]
                    stats = result["native_v8"]
                    if mode != "stock" and fixture["mix"] != "scripted" and stats["rows"] <= 0:
                        raise RuntimeError("native mode executed no native rows")
                    record = {
                        "fixture_id": fixture["fixture_id"],
                        "repeat": repeat,
                        "mode": mode,
                        "guard": guard,
                        **result,
                    }
                    export_json(
                        args.out / "batches" / f"{fixture['fixture_id']}-{repeat:02d}-{mode}.json",
                        record,
                    )
                    batches.append(record)
                    window()
                    if time.monotonic() - started >= manifest["timing"]["max_seconds"]:
                        raise RuntimeError("timing budget exhausted after batch")
                    print(
                        json.dumps(
                            {
                                "completed": len(batches),
                                "fixture": fixture["fixture_id"],
                                "mode": mode,
                            }
                        ),
                        flush=True,
                    )
        samples = {
            mode: [
                row["stage_seconds"]["opponent_v8"]
                for row in batches
                if row["fixture_id"] == "frozen-1500" and row["mode"] == mode
            ]
            for mode in operators
        }
        result = decision(samples["stock"], samples["kernel"], True)
        export_json(
            args.out / "timing.json",
            {
                "schema": "bl-k1-mac-timing/v1",
                "argv": sys.argv,
                "source_closure": manifest["sources"],
                **result,
                "manifest_sha256": manifest_hash,
                "identity_sha256": identity_hash,
                "stage_seconds": {
                    f"{fixture['fixture_id']}/{mode}": {
                        stage: {
                            "all30": distribution(
                                [
                                    row["stage_seconds"][stage]
                                    for row in batches
                                    if row["fixture_id"] == fixture["fixture_id"]
                                    and row["mode"] == mode
                                ]
                            ),
                            "warm29": distribution(
                                [
                                    row["stage_seconds"][stage]
                                    for row in batches
                                    if row["fixture_id"] == fixture["fixture_id"]
                                    and row["mode"] == mode
                                ][1:]
                            ),
                        }
                        for stage in batches[0]["stage_seconds"]
                    }
                    for fixture in manifest["fixtures"]
                    for mode in operators
                },
                "completed_batches": len(batches),
                "workers": 1,
                "threads": 1,
                "elapsed_seconds": time.monotonic() - started,
                "bar": manifest["bar"],
                "foreign_load_observed": any(
                    row["guard"]["process_screen"]["other_heavy_processes"] for row in batches
                ),
                "host_after": snapshot(),
                "serving_or_release": "NOT_GRANTED",
            },
        )
    except Exception as exc:
        export_json(
            args.out / "stopped.json",
            {
                "status": "STOPPED",
                "error": str(exc),
                "completed_batches": len(batches),
                "manifest_sha256": manifest_hash,
                "failed_observation": getattr(exc, "observation", None),
            },
        )
        raise


if __name__ == "__main__":
    main()
