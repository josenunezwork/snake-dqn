"""Fresh collision-audited Q0 development worlds, disjoint from the Q1 bank."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from research.apex_safety_20260926 import dev_screen
from research.t23_r0_screen_20261017 import bank as t23_bank

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "q0-t23-audit-h5000-development/20261017/v1"
SCHEMA = "q0-audit-world-bank/v1"
SEEDS = (7101, 7102, 7103, 7104, 7105)
MIX_COUNTS = {"scripted": 16, "frozen": 8, "mixed": 8}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_world_bank() -> dict[str, Any]:
    worlds: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        rows: list[dict[str, Any]] = []
        for mix, count in MIX_COUNTS.items():
            source = [
                r
                for r in dev_screen._design_rows(
                    dev_screen.screen_seeds(count, NAMESPACE, f"seed-{seed}/{mix}")
                )
                if r["mix"] == mix
            ]
            if len(source) != count:
                raise RuntimeError(f"unable to materialize {count} {mix} Q0 worlds")
            for row in source:
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
                    "profile_id": "q0-terminal-h5000/v1",
                    "horizon": 5000,
                    "terminal_hero_slot": 0,
                    "opponent_slots": slots,
                }
                rows.append(
                    {
                        **semantic,
                        "opponents": [["checkpoint", x["member_sha256"]] for x in slots],
                        "roster_semantic_sha256": _sha({"mix": mix, "slots": slots}),
                        "world_semantic_sha256": _sha(semantic),
                    }
                )
        worlds[str(seed)] = rows
    bank = {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "training_seeds": list(SEEDS),
        "mix_counts": MIX_COUNTS,
        "worlds": worlds,
        "horizon": 5000,
        "hero_mode": "terminal",
        "allocation_semantic_sha256": _sha(worlds),
    }
    candidate = {r["world_seed"] for rows in worlds.values() for r in rows}
    if len(candidate) != 160:
        raise RuntimeError("Q0 world allocation is not unique")
    return bank


def audit_world_bank(
    repo: Path = ROOT,
    artifacts: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    allow_receipt: bool = True,
) -> dict[str, Any]:
    """Audit locally, or require the Mac receipt for a queued Colab job."""
    bank = build_world_bank()
    environment = os.environ if env is None else env
    if environment.get("Q0Q1_JOB_KIND"):
        if not allow_receipt:
            raise ValueError("Mac inventory receipt producer cannot use a receipt")
        from research.q0q1_20261017.inventory_receipt import KIND_Q0, load_receipt_audit

        return load_receipt_audit(KIND_Q0, bank, env=environment, repo=repo)
    candidate = {r["world_seed"] for rows in bank["worlds"].values() for r in rows}
    bindings, historical = t23_bank._prior_strict_seed_inventory(
        repo, artifacts or repo.parent / "snake-dqn-artifacts", candidate
    )
    historical.update(
        t23_bank._scan_candidate_tokens(
            (repo / "research",), repo / "research/q0q1_20261017", candidate
        )
    )
    collisions = sorted(candidate & historical)
    return {
        "schema": "q0-audit-seed-audit/v1",
        "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        "candidate_count": len(candidate),
        "registered_strict_count": len(historical),
        "registered_collision_seeds": collisions,
        "mandatory_evidence": bindings,
        "pass": not collisions,
    }
