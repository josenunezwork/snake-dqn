"""Allocate all Round 0 identities before data; never replace a collision."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from research.t34_round0.common import COUNTS, PACKAGE, ROOT, read_json, sha256, write_once

NAMESPACE = "t34-round0-premise-20261020/v1"
PREFLIGHT_NAMESPACE = "t34-round0-preflight-20261020/v1"
_CHUNK_MIX_ORDER = (("scripted", 4), ("frozen", 3), ("mixed", 3))


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the frozen identities in 4:3:3 chunk-dispatch order.

    The bank itself stays in its canonical mix-major order because its bytes are
    hash-bound.  This derived order is the sole execution order: every ten
    worlds contain four scripted, three frozen, and three mixed worlds, while
    preserving each mix's original bank order.
    """
    if all("mix" not in row for row in rows):
        return list(rows)
    groups = {mix: [row for row in rows if row.get("mix") == mix] for mix, _ in _CHUNK_MIX_ORDER}
    if sum(len(group) for group in groups.values()) != len(rows):
        raise ValueError("bank contains an unknown chunk mix")
    cycles = {len(groups[mix]) // quota for mix, quota in _CHUNK_MIX_ORDER}
    if any(len(groups[mix]) % quota for mix, quota in _CHUNK_MIX_ORDER) or len(cycles) != 1:
        raise ValueError("bank mix counts cannot form 4:3:3 chunks")
    ordered: list[dict[str, Any]] = []
    for cycle in range(next(iter(cycles), 0)):
        for mix, quota in _CHUNK_MIX_ORDER:
            start = cycle * quota
            ordered.extend(groups[mix][start : start + quota])
    if len(ordered) != len(rows) or len({row.get("world_id") for row in ordered}) != len(rows):
        raise ValueError("bank chunk order lost or duplicated a world")
    return ordered


def _rows(namespace: str, counts: dict[str, int], horizon: int) -> list[dict[str, Any]]:
    # Roster construction imports NumPy; install one-thread caps first.
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    from research.apex_safety_20260926 import dev_screen
    from research.phase0_search_20261019.bank import _strict_lookup
    from research.t34_pathb.bank import _candidate_seeds

    lookup = _strict_lookup()
    frozen = (
        "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
        dev_screen.CHAMPION[1],
    )
    result = []
    for mix, count in counts.items():
        name = f"{namespace}/{mix}"
        seeds = _candidate_seeds(name, count)
        designs = {
            int(r["world_seed"]): r for r in dev_screen._design_rows(seeds) if r["mix"] == mix
        }
        for index, seed in enumerate(seeds):
            opponents = []
            for slot_index, slot in enumerate(designs[seed]["slots"]):
                kind, reference = lookup[str(slot["member_sha256"])]
                if kind == "checkpoint":
                    reference = frozen[(index + slot_index) % 2]
                opponents.append([kind, reference])
            result.append(
                {
                    "namespace": name,
                    "world_id": f"{name}/{index:04d}",
                    "world_seed": seed,
                    "mix": mix,
                    "opponents": opponents,
                    "opponent_veto": "v8",
                    "opponent_veto_lambda": 8.0,
                    "horizon": horizon,
                    "terminal_hero_slot": 0,
                }
            )
    return result


def validate_bank(bank: dict[str, Any], *, preflight: bool = False) -> None:
    """Check fixed bank shape, mix-disjoint seeds, and portable ordered rosters."""
    if bank.get("schema") != "t34-round0-bank/v1" or bank.get("preflight") is not preflight:
        raise ValueError("wrong Round 0 bank authority")
    counts = {"scripted": 1, "mixed": 1} if preflight else COUNTS
    rows = bank.get("rows", [])
    if len(rows) != sum(counts.values()):
        raise ValueError("bank count differs from registration")
    if len({r["world_seed"] for r in rows}) != len(rows):
        raise ValueError("bank mixes are not seed-disjoint")
    if len({r["world_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate world id")
    for mix, count in counts.items():
        if sum(r["mix"] == mix for r in rows) != count:
            raise ValueError("bank mix balance differs from registration")
    expected = _rows(
        PREFLIGHT_NAMESPACE if preflight else NAMESPACE, counts, 500 if preflight else 5000
    )
    if rows != expected:
        raise ValueError("bank differs from deterministic frozen allocation/rosters")


def allocate(bank_path: Path, audit_path: Path, *, preflight: bool = False) -> None:
    """Audit selection and excluded fixtures together against all visible history."""
    from research.t34_pathb.bank import _audit_candidates

    if bank_path.exists() or audit_path.exists():
        raise FileExistsError("allocation is create-only")
    selection = _rows(NAMESPACE, COUNTS, 5000)
    fixtures = _rows(PREFLIGHT_NAMESPACE, {"scripted": 1, "mixed": 1}, 500)
    all_rows = selection + fixtures
    seeds = [r["world_seed"] for r in all_rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("whole allocation rejected: internal collision")
    audit = _audit_candidates(ROOT, seeds)
    rows = fixtures if preflight else selection
    payload = {
        "schema": "t34-round0-bank/v1",
        "preflight": preflight,
        "counts": {"scripted": 1, "mixed": 1} if preflight else COUNTS,
        "namespace": PREFLIGHT_NAMESPACE if preflight else NAMESPACE,
        "profile": {
            "base": "promotion-v2-watch-rect",
            "horizon": 500 if preflight else 5000,
            "hero": "terminal",
            "hero_slot": 0,
            "snakes": 6,
            "hero_veto": "none",
            "frozen_veto": "v8",
            "mechanics_version": 1,
        },
        "rows": rows,
        "excluded_preflight_rows": [] if preflight else fixtures,
    }
    write_once(bank_path, payload)
    write_once(
        audit_path,
        {
            **audit,
            "schema": "t34-round0-seed-audit/v1",
            "status": "CLEAR",
            "bank_sha256": sha256(bank_path),
            "seeds": seeds,
            "collisions": [],
            "selection_count": 240,
            "excluded_fixture_count": 2,
            "retention_bank": {
                "path": "research/t34_pathb/RETENTION_BANK_V2.json",
                "sha256": sha256(ROOT / "research/t34_pathb/RETENTION_BANK_V2.json"),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=PACKAGE / "BANK_ROUND0.json")
    parser.add_argument("--audit", type=Path, default=PACKAGE / "SEED_AUDIT_ROUND0.json")
    parser.add_argument(
        "--preflight-inputs",
        type=Path,
        help="create excluded fixture inputs from the saved campaign audit",
    )
    args = parser.parse_args()
    if args.preflight_inputs is None:
        allocate(args.bank, args.audit)
    else:
        from research.t34_round0.registration import check_audit

        bank = check_audit(args.bank, args.audit)
        root = args.preflight_inputs
        root.mkdir(parents=True, exist_ok=False)
        fixture = {
            **bank,
            "preflight": True,
            "namespace": PREFLIGHT_NAMESPACE,
            "rows": bank["excluded_preflight_rows"],
            "excluded_preflight_rows": [],
            "counts": {"scripted": 1, "mixed": 1},
            "profile": {**bank["profile"], "horizon": 500},
        }
        write_once(root / "bank.json", fixture)
        audit = read_json(args.audit)
        audit.update(
            bank_sha256=sha256(root / "bank.json"), campaign_audit_sha256=sha256(args.audit)
        )
        write_once(root / "audit.json", audit)


if __name__ == "__main__":
    main()
