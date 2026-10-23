"""Claude's single Mac-side A3 look: verified whole allocations and accelerator ratios."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from research.a3_p1_ceiling.runtime import UNITS, aggregate, read_ledger
from research.bl_ppo.exports import digest, export_json, verify_export


def analyze(pack: Path, a100_ledger: Path, l4_ledger: Path) -> dict:
    """Require both registered whole allocations before reading timing outcomes."""
    controls = {
        gpu: json.loads((pack / gpu / "control.json").read_text()) for gpu in ("a100", "l4")
    }
    if controls["a100"]["commit"] != controls["l4"]["commit"]:
        raise ValueError("A3 accelerator source pins differ")
    # Establish the complete verified union on both targets without opening outcomes.
    for gpu, path in (("a100", a100_ledger), ("l4", l4_ledger)):
        ledger = read_ledger(path, controls[gpu])
        if set(ledger) != set(range(len(UNITS))):
            raise ValueError("INCOMPLETE A3 paired accelerator allocation")
        for row in ledger.values():
            payload = Path(row["path"])
            verify_export(payload)
            if payload.stat().st_size != row["size"] or digest(payload) != row["sha256"]:
                raise ValueError("A3 payload differs from verified ledger")
    results = {
        gpu: aggregate(controls[gpu], ledger)
        for gpu, ledger in (("a100", a100_ledger), ("l4", l4_ledger))
    }
    ratios = []
    for a100, l4 in zip(results["a100"]["units"], results["l4"]["units"]):
        mode = a100["mode"]
        if mode == "comparison":
            continue
        key = {"cached": "fwd_bwd_adam", "collection": "collection", "cycle": "complete_cycle"}[
            mode
        ]
        seconds = {
            gpu: statistics.median(row[key] for row in unit["result"]["phase_timers_seconds"])
            for gpu, unit in (("a100", a100), ("l4", l4))
        }
        ratios.append(
            {
                "mode": mode,
                "variant": a100["variant"],
                "seconds": seconds,
                "l4_over_a100": seconds["l4"] / seconds["a100"],
            }
        )
    return {
        "schema": "a3-p1-ceiling-comparison/v1",
        "status": "COMPLETE",
        "commit": controls["a100"]["commit"],
        "results": results,
        "ratios": ratios,
        "scope": "Operational WP5 falsifier; no P1 or serving qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--a100-ledger", type=Path, required=True)
    parser.add_argument("--l4-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export_json(args.out, analyze(args.pack, args.a100_ledger, args.l4_ledger))


if __name__ == "__main__":
    main()
