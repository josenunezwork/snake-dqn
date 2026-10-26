"""Rebuild the exact sandbox allowlist from pinned, non-F5 metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import ROOT, digest, f4_fold, safe_path

ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
INVENTORY = ART / "big-lever/round-30/bl-r30-postf4-20261025-v1/metadata-v3/corpus-inventory.json"
INVENTORY_SHA = "1dd5b4e42940d9f03cab8d735f8e66645de82fe1a6c5734b0a742540c125528e"
CENSUS = ROOT / "research/bl_residual/f4_postmortem_signal_fix_20261025/development-parents.json"
PINS = {
    str(INVENTORY): INVENTORY_SHA,
    str(CENSUS): "a90e3172c29465ce9bd97bb425fd924fb85b171c9bf48ee328a5b13635c4a512",
    "research/bl_labels/ledger/D256_UNION_FINAL_20261023.json": (
        "30919209b62307f8d4392b4e5925a7c4b0c4a21a2abd55a5e647428453eeb809"
    ),
    "research/bl_labels/ledger/D1024_UNION_FINAL_20261024.json": (
        "f5ec94ccdfae90133362a9072227d83125ae0ce661c44471904ce57333b610b8"
    ),
    "docs/research/governance_a100_sandbox_20261026/ADOPTED.md": (
        "b37094bb0f25e6917078c70964609ac2f53d3427690f97f3ff5c2be81ea97d1c"
    ),
    "docs/research/governance_a100_sandbox_20261026/FABLE_ON_ARGON_VS_ASTRA_20261026.md": (
        "8ab55d631d8655d36a2ca51b5e5418efb95ebb1e5758f70c2f68a4f7bd07f18d"
    ),
    "docs/research/round30_postf4_20261025/RULING.md": None,
    "research/bl_residual/F4_REGISTRATION.md": (
        "66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65"
    ),
    "research/bl_round30/corpus.py": None,
}


def build() -> dict:
    """Never open labels: derive exclusions before selecting payload references."""
    sources = []
    for name, expected in PINS.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError("source metadata pin changed: " + name)
        sources.append({"path": str(path), "sha256": actual})
    inv = json.loads(INVENTORY.read_text())
    census = set(json.loads(CENSUS.read_text())["parent_worlds"])
    r30 = {p["ultimate_parent_id"] for p in inv["parents"] if p["split"] == "validation"}
    if len(census) != 29 or len(r30) != 81 or len(inv["lives"]) != 960:
        raise ValueError("closed corpus or exclusion metadata count differs")
    excluded = census | r30
    lives = [r for r in inv["lives"] if r["ultimate_parent_id"] not in excluded]
    for row in lives:
        if row["original_fold"] == 0 or f4_fold(row["ultimate_parent_id"]) == 0:
            raise ValueError("fold-zero in closed fit selection")
        safe_path(row["payload_path"])
    datasets = {}
    for dose, selected in [("D256", inv["lives"][:192]), ("D1024", inv["lives"])]:
        chosen = [r for r in selected if r["ultimate_parent_id"] not in excluded]
        datasets[dose] = {
            "life_ids": [r["source_life_id"] for r in chosen],
            "life_count": len(chosen),
            "parent_count": len({r["ultimate_parent_id"] for r in chosen}),
            "selected_before_exclusions": len(selected),
        }
    seeds = set()
    fold0 = set()
    allocations = set()
    for name in PINS:
        if "UNION_FINAL" in name:
            ledger = json.loads((ROOT / name).read_text())
            for row in ledger["allocation"]:
                allocations.add(row["allocation_id"])
                seeds.add(row["seed"])
                if f4_fold(row["allocation_id"]) == 0:
                    fold0.add(row["allocation_id"])
    return {
        "schema": "a100-sandbox-input-registry/v1",
        "authority": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        "sources": sources,
        "datasets": datasets,
        "lives": lives,
        "excluded_parents": sorted(excluded | fold0),
        "exclusions": {
            "census_parents": sorted(census),
            "r30_validation_parents": sorted(r30),
            "f4_fold0_parents_in_closed_ledgers": sorted(fold0),
            "all_other_banks": [
                "strict",
                "probe",
                "serving",
                "F5-evaluation",
                "D4096",
                "t34-round0-study-01",
                "natural-archive",
            ],
            "method": (
                "positive closed-fit life+payload SHA allowlist; "
                "all unlisted banks denied without opening them"
            ),
        },
        "excluded_seeds": sorted(seeds),
        "worlds": {
            "formula": (
                'uint64-big-endian-first8(sha256("a100-dev/"' '+study+"/"+canonical_decimal_idx))'
            ),
            "width": 64,
            "normalization": "no truncation or modulo",
            "audit_scope": (
                "incremental closed-ledger intersections; development only; "
                "confirmation requires its own audit"
            ),
        },
        "counts": {
            "closed_ledger_allocation_parents": len(allocations),
            "f4_fold0_parents": len(fold0),
            "census_parents": len(census),
            "r30_validation_parents": len(r30),
            "excluded_parent_union": len(excluded | fold0),
            "selected_fit_lives": 960,
            "allowed_fit_lives": len(lives),
            "excluded_selected_fit_lives": 960 - len(lives),
        },
    }


def main() -> None:
    """Write create-only metadata registry; no payload reads or provider calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with safe_path(args.out).open("x") as stream:
        json.dump(build(), stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
