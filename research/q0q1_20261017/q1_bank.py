"""Immutable Q1 H5000 shared-world allocation and collision receipt.

This deliberately has a fresh namespace.  It reuses only the released strict
roster materializer, never a historical gate allocation or its worlds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.apex_safety_20260926 import dev_screen
from research.t23_r0_screen_20261017 import bank as t23_bank

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("Q1_WORLD_BANK.json")
AUDIT_OUTPUT = Path(__file__).with_name("Q1_SEED_AUDIT.json")
NAMESPACE = "q1-mature-n1n3-h5000-terminal-hero/20261017/v1"
SCHEMA = "q1-world-bank/v1"
PROFILE_ID = "q1-terminal-h5000/v1"
TRAINING_SEEDS = (261701, 261703, 261704, 261705, 261707)
MIX_COUNTS = {"scripted": 64, "frozen": 32, "mixed": 32}
PARITY_COUNT = 4
TRAINING_NAMESPACES = {
    "n3": "q1-t23-n3-20261017/v1",
    "n1": "q1-task-aligned-n1-20261017/v1",
}


def training_namespace(arm: str) -> str:
    """Return the registered stream namespace shared by its collector arms."""
    if arm in ("A", "B"):
        return TRAINING_NAMESPACES["n3"]
    if arm in ("C", "D"):
        return TRAINING_NAMESPACES["n1"]
    raise ValueError(f"unregistered Q1 arm {arm!r}")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _worlds(seed: int) -> list[dict[str, Any]]:
    worlds: list[dict[str, Any]] = []
    for mix, count in MIX_COUNTS.items():
        numbers = dev_screen.screen_seeds(count, NAMESPACE, f"eval/seed-{seed}/{mix}")
        rows = [row for row in dev_screen._design_rows(numbers) if row["mix"] == mix]
        if len(rows) != count:
            raise RuntimeError(f"strict materializer did not produce {count} {mix} rows")
        for row in rows:
            slots = [
                {
                    "slot": int(x["slot"]),
                    "kind": "member",
                    "member_sha256": str(x["member_sha256"]),
                    "eligible_member_sha256s": [str(v) for v in x["eligible_member_sha256s"]],
                }
                for x in row["slots"]
            ]
            semantic = {
                "namespace": NAMESPACE,
                "training_seed": seed,
                "mix": mix,
                "world_seed": int(row["world_seed"]),
                "profile_id": PROFILE_ID,
                "horizon": 5000,
                "terminal_hero_slot": 0,
                "opponent_slots": slots,
            }
            worlds.append(
                {
                    **semantic,
                    "opponents": [["checkpoint", x["member_sha256"]] for x in slots],
                    "roster_semantic_sha256": canonical_sha({"mix": mix, "slots": slots}),
                    "initial_world_sha256": canonical_sha(
                        {
                            "world_seed": int(row["world_seed"]),
                            "mix": mix,
                            "profile_id": PROFILE_ID,
                            "horizon": 5000,
                            "opponent_slots": slots,
                        }
                    ),
                    "world_semantic_sha256": canonical_sha(semantic),
                }
            )
    if len(worlds) != 128 or len({x["world_seed"] for x in worlds}) != 128:
        raise RuntimeError("Q1 allocation is not internally unique")
    return worlds


def _parity_worlds() -> list[dict[str, Any]]:
    # Allocate using an impossible training id then remove it; parity is reserved.
    worlds = []
    for index, mix in enumerate(("scripted", "scripted", "frozen", "mixed")):
        row = next(
            x
            for x in dev_screen._design_rows(
                dev_screen.screen_seeds(1, NAMESPACE, f"parity/{index}/{mix}")
            )
            if x["mix"] == mix
        )
        slots = [
            {
                "slot": int(x["slot"]),
                "kind": "member",
                "member_sha256": str(x["member_sha256"]),
                "eligible_member_sha256s": [str(v) for v in x["eligible_member_sha256s"]],
            }
            for x in row["slots"]
        ]
        semantic = {
            "namespace": NAMESPACE,
            "training_seed": None,
            "mix": mix,
            "world_seed": int(row["world_seed"]),
            "profile_id": PROFILE_ID,
            "horizon": 5000,
            "terminal_hero_slot": 0,
            "opponent_slots": slots,
        }
        worlds.append(
            {
                **semantic,
                "opponents": [["checkpoint", x["member_sha256"]] for x in slots],
                "roster_semantic_sha256": canonical_sha({"mix": mix, "slots": slots}),
                "initial_world_sha256": canonical_sha(
                    {
                        "world_seed": int(row["world_seed"]),
                        "mix": mix,
                        "profile_id": PROFILE_ID,
                        "horizon": 5000,
                        "opponent_slots": slots,
                    }
                ),
                "world_semantic_sha256": canonical_sha(semantic),
            }
        )
    return worlds


def build_world_bank() -> dict[str, Any]:
    allocation = {str(seed): _worlds(seed) for seed in TRAINING_SEEDS}
    parity = _parity_worlds()
    all_seeds = [x["world_seed"] for rows in allocation.values() for x in rows]
    if len(set(all_seeds + [x["world_seed"] for x in parity])) != len(all_seeds) + len(parity):
        raise RuntimeError("Q1 world/parity collision")
    profile = {
        "profile_id": PROFILE_ID,
        "base_profile_id": dev_screen.PROFILE_NAME,
        "base_profile_digest": dev_screen.PROFILE_DIGEST,
        "horizon": 5000,
        "engine": "grid",
        "hero_mode": "terminal",
        "hero_ego2s": True,
        "vector61_forward": "rowwise",
        "ego2s_forward": "rowwise",
        "ego2s_mask": "advisory",
        "veto": False,
        "vector61_backend": "numba",
        "scripted_anchors": "fast",
        "source": "research.apex_safety_20260926.dev_screen._design_rows",
    }
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "profile": profile,
        "mix_counts": MIX_COUNTS,
        "training_seeds": list(TRAINING_SEEDS),
        "training_namespaces": TRAINING_NAMESPACES,
        "worlds": allocation,
        "parity_worlds": parity,
        "profile_semantic_sha256": canonical_sha(profile),
        "allocation_semantic_sha256": canonical_sha(
            {"worlds": allocation, "parity_worlds": parity}
        ),
    }


def worlds_for_seed(bank: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = bank.get("worlds", {}).get(str(seed))
    if bank.get("schema") != SCHEMA or not isinstance(rows, list) or len(rows) != 128:
        raise ValueError("invalid Q1 world bank or seed allocation")
    if {mix: sum(x.get("mix") == mix for x in rows) for mix in MIX_COUNTS} != MIX_COUNTS:
        raise ValueError("invalid Q1 mix allocation")
    return rows


def load_world_bank(path: Path | None = None) -> dict[str, Any]:
    value = json.loads((path or OUTPUT).read_text())
    if value.get("allocation_semantic_sha256") != canonical_sha(
        {"worlds": value.get("worlds"), "parity_worlds": value.get("parity_worlds")}
    ):
        raise ValueError("Q1 world bank semantic digest mismatch")
    for seed in TRAINING_SEEDS:
        worlds_for_seed(value, seed)
    return value


def audit_world_bank(repo: Path = ROOT, artifacts: Path | None = None) -> dict[str, Any]:
    bank = build_world_bank()
    candidates = {x["world_seed"] for rows in bank["worlds"].values() for x in rows} | {
        x["world_seed"] for x in bank["parity_worlds"]
    }
    # T23's collector supplies the authoritative strict/T7 inventory.  Invoke
    # its broad-text component with a non-Q1 sentinel so it cannot mistake our
    # checked-in allocation for historical evidence; scan actual candidates
    # below with Q1's directory explicitly excluded.  Never subtract values
    # from the authoritative binary/strict inventory.
    training_seeds = set(TRAINING_SEEDS)
    candidates |= training_seeds
    bindings, historical = t23_bank._prior_strict_seed_inventory(
        repo, artifacts or repo.parent / "snake-dqn-artifacts", {4_294_967_291}
    )
    own = repo / "research/q0q1_20261017"
    external = t23_bank._scan_candidate_tokens(
        (
            repo / "research",
            repo / "docs",
            repo / "src",
            artifacts or repo.parent / "snake-dqn-artifacts",
        ),
        own,
        candidates,
    )
    historical |= external
    world_seeds = {x["world_seed"] for rows in bank["worlds"].values() for x in rows} | {
        x["world_seed"] for x in bank["parity_worlds"]
    }
    collisions = sorted(candidates & historical)
    return {
        "schema": "q1-seed-audit/v1",
        "namespace": NAMESPACE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        "candidate_count": len(world_seeds),
        "training_seed_count": len(training_seeds),
        "training_namespaces": TRAINING_NAMESPACES,
        "training_seed_collision_seeds": sorted(training_seeds & historical),
        "internal_unique": len(world_seeds) == 644 and not (world_seeds & training_seeds),
        "registered_collision_seeds": collisions,
        "mandatory_evidence": bindings,
        "pass": len(world_seeds) == 644 and not collisions,
        "limitations": [
            "Visible receipts and bound strict inventories; opaque binary payloads are outside this audit."
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--audit-out", type=Path, default=AUDIT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    bank = build_world_bank()
    audit = audit_world_bank(args.repo, args.artifacts)
    if not audit["pass"]:
        raise SystemExit(f"Q1 seed audit failed: {audit['registered_collision_seeds']}")
    args.out.write_text(json.dumps(bank, sort_keys=True, indent=2) + "\n")
    audit["world_bank_file"] = {
        "path": str(args.out),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.audit_out.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
