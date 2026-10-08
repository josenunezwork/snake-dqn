"""Build the OC simulation's variance pool from pre-registered Phase R records (read-only).

Sources (merged, pre-registered Phase R studies; their per-world records are read, never
written): FRP-v2 Phase 2 (M@60000), FRP-v3 Phase R (M3@60000), FRP-v4 Phase R (R4@60000), each
paired with its incumbent on the same (seed, mix, world). Only the per-world paired deltas'
*spread* is used: every (study, seed, mix, metric) is centred on its own mean (removing the
observed effects and seed effects, whose size the simulator sets per scenario) and rescaled by
``sqrt(n / (n - 1))``. A block is one (study, seed, world) with its 3 mixes x
(H5000 mass integral, H5000 survival fraction, H10000 mass integral).

Run once on the Mac (seconds): ``python -m research.sequential_phase_r.variance_pool --out
research/sequential_phase_r/variance_pool_20261007.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
MIXES = ("frozen", "mixed", "scripted")
METRICS = ("mi5", "surv5", "mi10")
SOURCES = {
    "frp_v2_phase2": ("frp-v2-20261004/phase2/run-v1/merged/summary.json", "M@60000"),
    "frp_v3_phaseR": ("frp-v3-20261005/phaseR/rp-v1/merged/summary.json", "M3@60000"),
    "frp_v4_phaseR": ("frp-v4-20261007/phaseR/merged-v1/summary.json", "R4@60000"),
}
SCHEMA = "sequential-phase-r-variance-pool/v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _records(dirs: List[str]) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for d in dirs:
        for f in sorted(Path(d, "records").glob("*.json")):
            r = json.loads(f.read_text())
            if r.get("control"):
                continue
            key = (r["hero"], int(r["seed"]), r["mix"], int(r["world_seed"]))
            if key in out:
                raise SystemExit(f"duplicate record {key}")
            out[key] = r
    return out


def _value(record: Dict[str, Any], metric: str) -> float:
    if metric == "mi5":
        return float(record["prefix_h5000"]["mass_integral"])
    if metric == "surv5":
        return float(record["prefix_h5000"]["survival_fraction"])
    return float(record["record"]["mass_integral"])


def build() -> Dict[str, Any]:
    blocks: List[List[List[float]]] = []
    provenance = {}
    for name, (summary_rel, cell) in SOURCES.items():
        summary_path = ARTIFACTS / summary_rel
        summary = json.loads(summary_path.read_text())
        dirs = [d for d in summary["shard_dirs"] if "/b4-shard" not in d]
        rows = _records(dirs)
        deltas: Dict[tuple, float] = {}
        for (hero, seed, mix, world), rec in rows.items():
            if hero != cell:
                continue
            base = rows[("incumbent", seed, mix, world)]
            for metric in METRICS:
                deltas[(seed, world, mix, metric)] = _value(rec, metric) - _value(base, metric)
        seeds = sorted({k[0] for k in deltas})
        centre = {}
        for seed in seeds:
            for mix in MIXES:
                for metric in METRICS:
                    vals = [v for k, v in deltas.items() if k[0] == seed and k[2:] == (mix, metric)]
                    n = len(vals)
                    centre[(seed, mix, metric)] = (sum(vals) / n, math.sqrt(n / (n - 1)))
        worlds = sorted({(k[0], k[1]) for k in deltas})
        for seed, world in worlds:
            blocks.append(
                [
                    [
                        round(
                            (deltas[(seed, world, mix, metric)] - centre[(seed, mix, metric)][0])
                            * centre[(seed, mix, metric)][1],
                            6,
                        )
                        for metric in METRICS
                    ]
                    for mix in MIXES
                ]
            )
        provenance[name] = {
            "summary": str(summary_path),
            "summary_sha256": _sha(summary_path),
            "cell": cell,
            "decision_status": summary["decision"].get("status"),
            "seeds": seeds,
            "worlds": len(worlds),
        }
    return {
        "schema": SCHEMA,
        "mixes": list(MIXES),
        "metrics": list(METRICS),
        "centring": "per (study, seed, mix, metric) mean removed, scaled by sqrt(n/(n-1))",
        "sources": provenance,
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists (create-only)")
    pool = build()
    args.out.write_text(json.dumps(pool, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"{len(pool['blocks'])} blocks -> {args.out}")


if __name__ == "__main__":
    main()
