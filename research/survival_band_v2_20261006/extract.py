#!/usr/bin/env python3
"""Calibration pools of survival band v2 (paired survival, mass delta), from pre-existing runs only.

Writes ``survival_pools_20261006.json`` beside this file, in the ``simulate.py --data`` format of
``research/paired_band_validation_20261003`` (one entry per pool: ``world_seeds`` and, per mix,
``incumbent_survival``, ``candidate_survival`` and ``mass_delta``, index-aligned; the three mixes
of an index are the same world).

Data provenance (governance amendment survival band v2, 2026-10-06, section "Data provenance"):
every pool below is a completed run that closed **before** FRP-v3 Phase R (merged 2026-10-05
20:14 UTC) and does not involve the FRP-v3 candidate.  FRP-v3 Phase R and the LH-1 screen of the
FRP-v3 candidate are NOT read here; they may only be used descriptively, elsewhere, and labelled.

* ``frp2_phase2`` -- **between-checkpoint, primary.**  FRP-v2 Phase 2 (merged 2026-10-05 12:12
  UTC): five FRP-v2 fine-tunes M@60000 (training seeds 0-4) + v8 vs champion_a5 + v8, 32 worlds
  x 3 mixes per seed, exact H5000 prefixes of H10000 episodes (160 world triples).
* ``frp2_phase1`` (and one pool per cell, ``frp2_phase1_cell_<cell>``, 80 world triples
  each) -- **between-checkpoint.**  FRP-v2 Phase 1 (merged 2026-10-05 09:19 UTC): six
  cells (arms C and M at 15k / 30k / 60k updates) x five training seeds, each + v8 vs champion_a5
  + v8 on that seed's 16 worlds x 3 mixes, H5000 (480 world triples; the incumbent episodes of a
  seed are shared by its six cells).
* ``v8_strict`` -- veto change (v7 -> v8, same checkpoint): the v8 strict gate's final records
  (closed STRICT_PASS 2026-10-03), 187 worlds x 3 mixes.
* ``v8_refbank`` -- veto change (v7 -> v8): the v8 reference bank (rp-bank-long, merged
  2026-10-04), 150 worlds x 3 mixes, SIMD engine.
* ``v7_strict``, ``v8_screen`` -- copied unchanged from the 2026-10-03 validation data
  (``paired_survival_20261003.json``, sha256 pinned).

Usage: ``OMP_NUM_THREADS=1 ./venv/bin/python research/survival_band_v2_20261006/extract.py``
(reads JSON records only; no game is played).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
OUT = HERE / "survival_pools_20261006.json"
MIXES = ("frozen", "scripted", "mixed")
CHAMPION = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
V8 = "free-space-veto/v8-space-and-head(lambda=8.0)"
V7 = "free-space-veto/v7-space-preference(lambda=4.0)"
FRP2 = ARTIFACTS / "frp-v2-20261004"
FRP2_SUMMARIES = {
    "phase1": "phase1/run-v1/merged/summary.json",
    "phase2": "phase2/run-v1/merged/summary.json",
}
V8_STRICT = ARTIFACTS / "apex-veto-v8-strict-20261003" / "run-v1" / "output"
V8_REFBANK = (
    ARTIFACTS
    / "runpod-fanout"
    / "rp-bank-long-results"
    / "v8refbank-a-merged"
    / "records"
    / "apex-veto-v8-refbank-v1__h5000__simd"
)
STOCK = REPO / "research" / "paired_band_validation_20261003" / "paired_survival_20261003.json"
FRP3_PHASE_R_MERGED_UTC = "2026-10-05T20:14:21Z"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def files_digest(paths: Sequence[Path]) -> str:
    """sha256 over ``name:sha256`` lines of the files read (sorted by name)."""
    lines = sorted(f"{p.name}:{sha256_file(p)}" for p in paths)
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def triples(
    inc: Mapping[Tuple[str, int], Mapping[str, Any]],
    cand: Mapping[Tuple[str, int], Mapping[str, Any]],
    worlds: Sequence[int],
    label: str,
) -> List[Dict[str, Any]]:
    """World-ordered (index) triples of both arms; every mix of an index must be one world."""
    out = []
    for w in worlds:
        seeds = {inc[(m, w)]["world_seed"] for m in MIXES} | {
            cand[(m, w)]["world_seed"] for m in MIXES
        }
        if len(seeds) != 1:
            raise SystemExit(f"{label} world {w}: mixes/arms on different worlds {seeds}")
        out.append(
            {
                "world_seed": seeds.pop(),
                **{
                    m: (
                        float(inc[(m, w)]["survival_fraction"]),
                        float(cand[(m, w)]["survival_fraction"]),
                        float(cand[(m, w)]["mass_integral"]) - float(inc[(m, w)]["mass_integral"]),
                    )
                    for m in MIXES
                },
            }
        )
    return out


def as_pool(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pool: Dict[str, Any] = {"world_seeds": [int(r["world_seed"]) for r in rows]}
    for m in MIXES:
        pool[m] = {
            "incumbent_survival": [r[m][0] for r in rows],
            "candidate_survival": [r[m][1] for r in rows],
            "mass_delta": [r[m][2] for r in rows],
        }
    return pool


# ---------------------------------------------------------------- FRP-v2


def frp2(phase: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = FRP2 / phase / "run-v1"
    summary_path = FRP2 / FRP2_SUMMARIES[phase]
    summary = json.loads(summary_path.read_text())
    paths = sorted(root.glob("shard-*/records/*.json"))
    by: Dict[Tuple[str, int], Dict[Tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        entry = json.loads(path.read_text())
        if entry["control"]:
            continue
        veto = entry["record"]["probes"]["safety_veto"]["method"]
        if veto != V8:
            raise SystemExit(f"{path.name}: veto {veto}")
        if entry["hero"] == "incumbent" and entry["hero_sha256"] != CHAMPION:
            raise SystemExit(f"{path.name}: incumbent is not champion_a5")
        values = entry.get("prefix_h5000") or entry["record"]
        if entry["horizon"] != 5000 and entry.get("prefix_h5000") is None:
            raise SystemExit(f"{path.name}: H{entry['horizon']} without an H5000 prefix")
        key = (entry["mix"], int(entry["world_index"]))
        cell = (entry["hero"], int(entry["seed"]))
        if key in by[cell]:
            raise SystemExit(f"{path.name}: duplicate record")
        by[cell][key] = {
            "survival_fraction": values["survival_fraction"],
            "mass_integral": values["mass_integral"],
            "world_seed": int(entry["world_seed"]),
        }
    cells = sorted({hero for hero, _ in by if hero != "incumbent"})
    seeds = sorted({seed for _, seed in by})
    rows: List[Dict[str, Any]] = []
    per_cell: Dict[str, List[Dict[str, Any]]] = {}
    for cell in cells:
        per_cell[cell] = []
        for seed in seeds:
            inc, cand = by[("incumbent", seed)], by[(cell, seed)]
            worlds = sorted({w for _, w in inc})
            if sorted({w for _, w in cand}) != worlds or len(inc) != 3 * len(worlds):
                raise SystemExit(f"{phase} {cell} seed {seed}: arms do not pair")
            per_cell[cell].extend(triples(inc, cand, worlds, f"{phase} {cell} s{seed}"))
        rows.extend(per_cell[cell])
    source = {
        "kind": "between-checkpoint (FRP-v2 fine-tune + v8 vs champion_a5 + v8)",
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_created_utc": summary["created_utc"],
        "decision_status": summary["decision"]["status"],
        "cells": cells,
        "training_seeds": seeds,
        "records_read": len(paths),
        "records_digest": files_digest(paths),
        "metric_horizon": "H5000 (exact prefix of H10000 episodes in phase 2)",
        "world_triples": len(rows),
    }
    cell_pools = {cell_pool_name(phase, cell): as_pool(r) for cell, r in per_cell.items()}
    return as_pool(rows), source, cell_pools


def cell_pool_name(phase: str, cell: str) -> str:
    """``frp2_phase1_cell_C15000`` (one recipe cell: five training seeds' checkpoints)."""
    return f"frp2_{phase}_cell_{cell.replace('@', '')}"


# ---------------------------------------------------------------- v8 strict (v7 -> v8)


def v8_strict() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paths = sorted((V8_STRICT / "final" / "records").glob("final-*.json"))
    arms: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {"incumbent": {}, "candidate": {}}
    for path in paths:
        entry = json.loads(path.read_text())
        rec = entry["record"]
        arms[entry["arm"]][(entry["mix"], int(entry["world_index"]))] = {
            "survival_fraction": rec["survival_fraction"],
            "mass_integral": rec["mass_integral"],
            "world_seed": int(entry["world_seed"]),
        }
    worlds = sorted({w for _, w in arms["incumbent"]})
    if sorted({w for _, w in arms["candidate"]}) != worlds:
        raise SystemExit("v8 strict: arms do not pair")
    rows = triples(arms["incumbent"], arms["candidate"], worlds, "v8_strict")
    closeout = V8_STRICT / "closeout.json"
    return as_pool(rows), {
        "kind": "veto change (champion_a5 + v7 -> champion_a5 + v8), same checkpoint",
        "closeout_path": str(closeout),
        "closeout_sha256": sha256_file(closeout),
        "outcome": json.loads(closeout.read_text()).get("outcome"),
        "records_read": len(paths),
        "records_digest": files_digest(paths),
        "world_triples": len(rows),
    }


# ---------------------------------------------------------------- v8 reference bank (v7 -> v8)


def v8_refbank() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paths = sorted(V8_REFBANK.glob("*.json"))
    arms: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {"A": {}, "B": {}}
    for path in paths:
        entry = json.loads(path.read_text())
        expected = V7 if entry["arm"] == "A" else V8
        if entry["safety_veto_method"] != expected or entry["hero_sha256"] != CHAMPION:
            raise SystemExit(f"{path.name}: arm {entry['arm']} is not champion_a5 + {expected}")
        rec = entry["record"]
        arms[entry["arm"]][(entry["mix"], int(entry["world_index"]))] = {
            "survival_fraction": rec["survival_fraction"],
            "mass_integral": rec["mass_integral"],
            "world_seed": int(entry["world_seed"]),
        }
    worlds = sorted({w for _, w in arms["A"]})
    if sorted({w for _, w in arms["B"]}) != worlds:
        raise SystemExit("v8 refbank: arms do not pair")
    rows = triples(arms["A"], arms["B"], worlds, "v8_refbank")
    merge = V8_REFBANK.parents[2] / "v8refbank-a-merge.json"
    return as_pool(rows), {
        "kind": "veto change (champion_a5 + v7 -> champion_a5 + v8), same checkpoint, SIMD",
        "merge_path": str(merge),
        "merge_sha256": sha256_file(merge),
        "records_read": len(paths),
        "records_digest": files_digest(paths),
        "world_triples": len(rows),
    }


def main() -> int:
    stock = json.loads(STOCK.read_text())
    pools: Dict[str, Any] = {}
    sources: Dict[str, Any] = {}
    for phase, name in (("phase2", "frp2_phase2"), ("phase1", "frp2_phase1")):
        pools[name], sources[name], cell_pools = frp2(phase)
        if phase == "phase1":  # one pool per recipe cell (the phase-2 pool is one cell)
            for cell_name, pool in cell_pools.items():
                pools[cell_name] = pool
                sources[cell_name] = {
                    "kind": "between-checkpoint, one FRP-v2 Phase 1 cell (subset of frp2_phase1)",
                    "of": name,
                    "world_triples": len(pool["world_seeds"]),
                }
    pools["v8_strict"], sources["v8_strict"] = v8_strict()
    pools["v8_refbank"], sources["v8_refbank"] = v8_refbank()
    for name in ("v7_strict", "v8_screen"):
        pools[name] = stock[name]
        sources[name] = {
            "kind": "veto change (copied from the 2026-10-03 validation data)",
            "path": str(STOCK.relative_to(REPO)),
            "sha256": sha256_file(STOCK),
            "detail": stock["sources"][name],
        }
    document = {
        "label": "survival band v2 calibration pools: pre-existing paired runs only (no FRP-v3 "
        "Phase R / LH-1 data); H5000 survival_fraction of both arms and the H5000 mass_integral "
        "delta, index-aligned per world",
        "excluded_by_rule": [
            "FRP-v3 Phase R (frp-v3-20261005/phaseR, merged " + FRP3_PHASE_R_MERGED_UTC + ")",
            "LH-1 screen of the FRP-v3 candidate (lh1-screen-frp3-s12)",
            "every other FRP-v3 artifact",
        ],
        "sources": sources,
        **pools,
    }
    OUT.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": sha256_file(OUT), "pools": sorted(pools)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
