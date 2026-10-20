"""Immutable, fresh development-bank planning for T34.

This module deliberately plans the terminal rosters through the historical grid
authority; it never substitutes the product FRP3 roster for frozen/mixed rows.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from research.t3_rollout_labels_20261015 import bank_engine

NAMESPACE = "t34-pathb-deadline-20261019/v1"
SMOKE_NAMESPACE = "t34-pathb-deadline-20261019/smoke-v1"
MIXES = ("scripted", "frozen", "mixed")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_seeds(namespace: str, count: int) -> list[int]:
    """Derive the complete allocation before inspecting history; never reroll."""
    values = []
    for index in range(count):
        value = int.from_bytes(
            hashlib.sha256(f"{namespace}/world/{index}".encode("utf-8")).digest()[:4], "big"
        )
        values.append(value)
    if len(values) != len(set(values)):
        raise RuntimeError("deterministic T34 allocation contains an internal collision")
    return values


def _audit_candidates(repo: Path, candidates: list[int]) -> dict[str, Any]:
    """One fail-closed scan over source, documents, scripts, and artifact history.

    This deliberately does not call ``allocate_unseen_seeds``: that helper grows
    the candidate prefix and silently replaces collisions, which is forbidden by
    the T34 protocol.
    """
    artifacts = repo.parent / "snake-dqn-artifacts"
    roots = [repo / "docs", repo / "research", repo / "src", repo / "scripts", artifacts]
    missing = [str(path) for path in roots if not path.is_dir()]
    # Some checkouts have no top-level scripts directory.  Record its coverage;
    # the history/artifact roots themselves remain mandatory and fail closed.
    mandatory_missing = [str(path) for path in (repo / "docs", repo / "research", artifacts) if not path.is_dir()]
    if mandatory_missing:
        raise RuntimeError(f"mandatory T34 freshness scan roots unavailable: {mandatory_missing}")
    scanned = [path for path in roots if path.is_dir()]
    hits = {str(path): sorted(bank_engine.concrete_seed_hits(path, candidates)) for path in scanned}
    # Named historical FRP namespaces are deliberately retained as explicit
    # exclusions in the receipt, even though the complete research scan covers
    # them.  This makes the audit reviewable against the protocol.
    frp_exclusions = [
        "research/frp3_checkpoint_serving_20261005",
        "research/frp3_strict_20261005",
        "research/apex_veto_v5_strict_20261001",
        "research/apex_veto_v7_strict_20261002",
        "research/apex_veto_v7_serving_20261002",
        "research/apex_veto_v8_strict_20261003",
        "research/apex_veto_v8_serving_20261003",
    ]
    present = [name for name in frp_exclusions if (repo / name).exists()]
    if len(present) < 7:
        # Names changed over time; evidence is still exhaustive through research,
        # but do not pretend all seven named historical exclusions were located.
        raise RuntimeError(f"historical FRP exclusion coverage incomplete: found {present}")
    collisions = {root: values for root, values in hits.items() if values}
    if collisions:
        raise RuntimeError(f"T34 candidate allocation collides; allocation rejected: {collisions}")
    return {"method": "deterministic-all-at-once-sha256-v1", "candidate_count": len(candidates),
            "scan_roots": [str(path) for path in scanned], "missing_optional_roots": missing,
            "concrete_hits": hits, "historical_frp_exclusions": present,
            "coverage": "complete-source-docs-artifacts-concrete-seed-scan"}


def allocation(*, blocks: int, scripted: int, frozen: int, mixed: int, product: int,
               smoke: bool, repo: Path) -> dict[str, Any]:
    """Allocate once, rejecting any concrete seed already visible in history."""
    if blocks < 1 or min(scripted, frozen, mixed, product) < 0:
        raise ValueError("blocks must be positive and world counts nonnegative")
    per_block = scripted + frozen + mixed + product
    if per_block < 1:
        raise ValueError("at least one world is required")
    namespace = SMOKE_NAMESPACE if smoke else NAMESPACE
    seeds = _candidate_seeds(namespace, blocks * per_block)
    audit = _audit_candidates(Path(repo).resolve(), seeds)
    terminal_mixes = ("scripted",) * scripted + ("frozen",) * frozen + ("mixed",) * mixed
    rows, cursor, plans = [], 0, []
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    context = gid._context(1)
    for block in range(blocks):
        block_rows = []
        for mix in terminal_mixes:
            block_rows.append({"seed": seeds[cursor], "mix": mix, "kind": "terminal"})
            cursor += 1
        # Historical plan materializes exact scripted/frozen/mixed roster specs.
        terminal = bank_engine.planned_rosters(
            context, [r["seed"] for r in block_rows], [r["mix"] for r in block_rows]
        ) if block_rows else []
        for row, plan in zip(block_rows, terminal):
            row["roster"] = plan
        products = []
        for _ in range(product):
            products.append({"seed": seeds[cursor], "mix": "mixed", "kind": "product"})
            cursor += 1
        rows.append({"block": block, "terminal": block_rows, "product": products})
        plans.extend(block_rows + products)
    return {"namespace": namespace, "blocks": rows, "seed_audit": audit, "worlds": plans}
