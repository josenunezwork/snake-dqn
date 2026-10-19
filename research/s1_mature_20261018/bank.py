"""Immutable S1 shared H5000 world allocation and collision audit.

This module allocates no gameplay and loads no candidate.  It binds the five
fresh training blocks and the one final-look world bank consumed by both S1
arms and B8.  The audit deliberately reuses T23's broad historical collector
instead of treating a local grep as evidence about historical strict banks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.apex_safety_20260926 import dev_screen
from research.t23_r0_screen_20261017 import bank as t23_bank

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("WORLD_BANK.json")
AUDIT_OUTPUT = Path(__file__).with_name("SEED_AUDIT.json")
NAMESPACE = "s1-mature-h5000-terminal-hero/20261018/v5"
SCHEMA = "s1-mature-world-bank/v1"
SEED_BANK_SCHEMA = "s1-seed-bank/v1"
PROFILE_ID = "s1-mature-terminal-h5000/v1"
# Entire replacement block.  The earlier prospective block was rejected as a
# whole when its audit found a collision; no S1 data existed and none of its
# worlds or seeds survive into this registered allocation.
TRAINING_SEED_BLOCK_NAMESPACE = "s1-mature-training-seed-block/20261018/v2"
TRAINING_SEEDS = (223213833, 3055723932, 1783343308, 1125234883, 1809856951)
MIX_COUNTS = {"scripted": 64, "frozen": 32, "mixed": 32}
TIMING_MIX_COUNTS = {"scripted": 4, "frozen": 3, "mixed": 3}
TRAINING_NAMESPACES = {
    "n3": "s1-mature-20261018/train/n3",
    "n1": "q1-task-aligned-n1-20261017/s1-mature-20261018/n1",
}


def canonical_sha(value: Any) -> str:
    """Return a portable canonical JSON SHA-256 binding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def training_namespace(arm: str, seed: int) -> str:
    """Return the registered collection namespace for one S1 run."""
    key = {"N3": "n3", "n3": "n3", "N1": "n1", "n1": "n1"}.get(arm)
    if key is None or seed not in TRAINING_SEEDS:
        raise ValueError(f"unregistered S1 arm/seed: {arm!r}/{seed!r}")
    return f"{TRAINING_NAMESPACES[key]}/seed-{seed}"


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
    if len(worlds) != sum(MIX_COUNTS.values()) or len({x["world_seed"] for x in worlds}) != len(
        worlds
    ):
        raise RuntimeError("S1 allocation is not internally unique")
    return worlds


def _timing_worlds() -> list[dict[str, Any]]:
    """Allocate a disjoint, non-decision 10-world timing fixture."""
    rows: list[dict[str, Any]] = []
    for mix, count in TIMING_MIX_COUNTS.items():
        numbers = dev_screen.screen_seeds(count, NAMESPACE, f"timing/{mix}")
        for row in (item for item in dev_screen._design_rows(numbers) if item["mix"] == mix):
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
                "fixture": "timing",
                "mix": mix,
                "world_seed": int(row["world_seed"]),
                "profile_id": PROFILE_ID,
                "horizon": 5000,
                "terminal_hero_slot": 0,
                "opponent_slots": slots,
            }
            rows.append(
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
    if len(rows) != 10 or len({x["world_seed"] for x in rows}) != 10:
        raise RuntimeError("S1 timing fixture is not internally unique")
    return rows


def build_world_bank() -> dict[str, Any]:
    """Build the 5 x 128 shared final-look allocation."""
    allocation = {str(seed): _worlds(seed) for seed in TRAINING_SEEDS}
    timing = _timing_worlds()
    all_seeds = [x["world_seed"] for rows in allocation.values() for x in rows]
    if len(set(all_seeds + [x["world_seed"] for x in timing])) != len(all_seeds) + len(timing):
        raise RuntimeError("S1 cross-seed world collision")
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
        "seed_bank_schema": SEED_BANK_SCHEMA,
        "namespace": NAMESPACE,
        "training_seed_block_namespace": TRAINING_SEED_BLOCK_NAMESPACE,
        "profile": profile,
        "mix_counts": MIX_COUNTS,
        "training_seeds": list(TRAINING_SEEDS),
        "training_namespaces": TRAINING_NAMESPACES,
        "worlds": allocation,
        "timing_worlds": timing,
        "profile_semantic_sha256": canonical_sha(profile),
        "allocation_semantic_sha256": canonical_sha(
            {"worlds": allocation, "timing_worlds": timing}
        ),
    }


def worlds_for_seed(bank: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Return exactly 128 fixed worlds for a registered training seed."""
    rows = bank.get("worlds", {}).get(str(seed))
    if (
        bank.get("schema") != SCHEMA
        or not isinstance(rows, list)
        or len(rows) != sum(MIX_COUNTS.values())
    ):
        raise ValueError("invalid S1 world bank or seed allocation")
    if {mix: sum(x.get("mix") == mix for x in rows) for mix in MIX_COUNTS} != MIX_COUNTS:
        raise ValueError("invalid S1 mix allocation")
    return rows


def load_world_bank(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the checked-in world bank for evaluator consumers."""
    value = json.loads((path or OUTPUT).read_text())
    if value.get("allocation_semantic_sha256") != canonical_sha(
        {"worlds": value.get("worlds"), "timing_worlds": value.get("timing_worlds")}
    ):
        raise ValueError("S1 world bank semantic digest mismatch")
    for seed in TRAINING_SEEDS:
        worlds_for_seed(value, seed)
    return value


def audit_world_bank(repo: Path = ROOT, artifacts: Path | None = None) -> dict[str, Any]:
    """Fail closed on visible historical seed or namespace collisions."""
    repo, artifacts = Path(repo), Path(artifacts or ROOT.parent / "snake-dqn-artifacts")
    bank = build_world_bank()
    worlds = {x["world_seed"] for rows in bank["worlds"].values() for x in rows}
    timing_worlds = {x["world_seed"] for x in bank["timing_worlds"]}
    candidates = worlds | timing_worlds | set(TRAINING_SEEDS)
    bindings, historical = t23_bank._prior_strict_seed_inventory(repo, artifacts, {4_294_967_291})
    external = t23_bank._scan_candidate_tokens(
        (repo / "research", repo / "docs", repo / "src", artifacts),
        repo / "research/s1_mature_20261018",
        candidates,
    )
    historical |= external
    collisions = sorted(candidates & historical)
    namespace_tokens = (NAMESPACE, *TRAINING_NAMESPACES.values())
    roots = [
        str(root)
        for root in (repo / "research", repo / "docs", repo / "src", artifacts)
        if root.exists()
    ]
    result = subprocess.run(
        [
            "rg",
            "-l",
            "-F",
            "--glob",
            "*.json",
            "--glob",
            "*.jsonl",
            "--glob",
            "*.md",
            *sum((["-e", token] for token in namespace_tokens), []),
            *roots,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"namespace collision scan failed: {result.stderr.strip()}")
    namespace_hits = {token: [] for token in namespace_tokens}
    for filename in result.stdout.splitlines():
        if "s1_mature_20261018" not in filename:
            text = Path(filename).read_text(errors="ignore")
            for token in namespace_tokens:
                if token in text:
                    namespace_hits[token].append(filename)
    return {
        "schema": "s1-mature-seed-audit/v1",
        "seed_bank_schema": SEED_BANK_SCHEMA,
        "namespace": NAMESPACE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        "candidate_count": len(worlds),
        "training_seed_count": len(TRAINING_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "rejected_pre_data_allocations": [
            {
                "training_seed_block": [426823, 426829, 426841, 426863, 426919],
                "reason": (
                    "collision audit rejected the entire provisional block before S1 data existed"
                ),
                "retained_for_execution": False,
            },
            {
                "training_seed_block": list(TRAINING_SEEDS),
                "evaluation_namespace": "s1-mature-h5000-terminal-hero/20261018/v3",
                "reason": (
                    "collision audit rejected the entire provisional world allocation "
                    "before S1 data existed"
                ),
                "retained_for_execution": False,
            },
            {
                "training_seed_block": list(TRAINING_SEEDS),
                "evaluation_namespace": "s1-mature-h5000-terminal-hero/20261018/v4",
                "reason": (
                    "collision audit rejected the entire provisional world allocation "
                    "before S1 data existed"
                ),
                "retained_for_execution": False,
            },
        ],
        "training_namespaces": TRAINING_NAMESPACES,
        "namespace_collision_files": namespace_hits,
        "internal_unique": len(worlds) == 640
        and len(timing_worlds) == 10
        and not (worlds & timing_worlds)
        and not (worlds & set(TRAINING_SEEDS))
        and not (timing_worlds & set(TRAINING_SEEDS)),
        "registered_collision_seeds": collisions,
        "mandatory_evidence": bindings,
        "coverage": [
            "bound T7 strict inventory",
            "visible JSON/JSONL under research, docs, src and artifact root",
            "Q1 package and all other visible repository namespaces are included",
            "exact S1 evaluation and training namespace strings",
        ],
        "limitations": [
            "Opaque binary payloads are outside this audit; a collision discovered later "
            "invalidates the study."
        ],
        "pass": len(worlds) == 640
        and len(timing_worlds) == 10
        and not (worlds & timing_worlds)
        and not (worlds & set(TRAINING_SEEDS))
        and not (timing_worlds & set(TRAINING_SEEDS))
        and not collisions
        and not any(namespace_hits.values()),
    }


def _write(path: Path, value: Any) -> None:
    # The bank is intentionally compact so it remains below repository hook
    # limits while its canonical semantic hash protects whitespace-independent
    # interpretation.
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


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
        raise SystemExit(f"S1 seed audit failed: {audit['registered_collision_seeds']}")
    _write(args.out, bank)
    audit["world_bank_file"] = {
        "path": str(args.out),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    _write(args.audit_out, audit)
    print(json.dumps({"worlds": audit["candidate_count"], "audit": str(args.audit_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
