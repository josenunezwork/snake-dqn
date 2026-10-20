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


def allocation(*, blocks: int, scripted: int, frozen: int, mixed: int, product: int,
               smoke: bool, repo: Path) -> dict[str, Any]:
    """Allocate once, rejecting any concrete seed already visible in history."""
    if blocks < 1 or min(scripted, frozen, mixed, product) < 0:
        raise ValueError("blocks must be positive and world counts nonnegative")
    per_block = scripted + frozen + mixed + product
    if per_block < 1:
        raise ValueError("at least one world is required")
    # The pre-data source declaration contains this namespace, so only concrete
    # seed collisions are meaningful here.  Allocation itself is deterministic.
    namespace = SMOKE_NAMESPACE if smoke else NAMESPACE
    original_namespace, original_domain = bank_engine.NAMESPACE, bank_engine.SEED_DOMAIN
    try:
        bank_engine.NAMESPACE = namespace
        bank_engine.SEED_DOMAIN = namespace
        seeds, audit = bank_engine.allocate_unseen_seeds(blocks * per_block, [repo / "research"])
    finally:
        bank_engine.NAMESPACE, bank_engine.SEED_DOMAIN = original_namespace, original_domain
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
