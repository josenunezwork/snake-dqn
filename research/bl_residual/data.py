"""Outcome-blind ancestry splits and a strict, pickle-free F0 adapter.

Allocation consumes life metadata only. Raw held-out outcomes are loaded only
after the final checkpoint is immutable; no split is chosen from fit metrics.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

DOSES = (256, 1024, 4096)
SPLIT_NAMESPACE = "bl-residual-f3-split-v1"
STATUSES = ("EXACT", "NOT_EVALUATED", "UNKNOWN")


def world_fold(parent_world: str) -> int:
    """Keep every life, audit, extension and DAgger descendant on one side."""
    if not isinstance(parent_world, str) or not parent_world:
        raise ValueError("nonempty ultimate parent_world required")
    raw = f"{SPLIT_NAMESPACE}/{parent_world}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 5


def allocate_lives(
    catalog: list[dict[str, Any]], prior: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Freeze exact nested total doses, including a common 64-life holdout.

    Only identity/stream/rescue-bearing metadata is consumed, never branch
    outcomes. Fold zero is held out. Sixteen Watch6 lives are reserved first;
    all remaining choices use frozen append-only indices. An insufficient pool is refused.
    """
    seen: set[str] = set()
    indices: set[int] = set()
    rows = []
    for row in catalog:
        life = row.get("life_id")
        parent = row.get("parent_world")
        if not isinstance(life, str) or not life or life in seen:
            raise ValueError("duplicate or missing life identity")
        if row.get("stream") not in ("S-Watch6", "S-gate"):
            raise ValueError("foreign natural stream")
        if type(row.get("rescue_bearing")) is not bool:
            raise ValueError("rescue_bearing metadata required")
        index = row.get("allocation_index")
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError("unique frozen allocation_index required")
        indices.add(index)
        seen.add(life)
        fold = world_fold(parent)
        if row["rescue_bearing"]:
            rows.append({**row, "fold": fold})
    # Append-only collection indices, never a fresh ranking of a growing pool.
    rows.sort(key=lambda r: r["allocation_index"])
    eligible = [r for r in rows if r["fold"] == 0]
    watch = [r for r in eligible if r["stream"] == "S-Watch6"][:16]
    selected = {r["life_id"] for r in watch}
    heldout = watch + [r for r in eligible if r["life_id"] not in selected][:48]
    if len(watch) != 16 or len(heldout) != 64:
        raise ValueError("INSUFFICIENT: need 64 fold-zero rescue lives including 16 Watch6")
    train = [r for r in rows if r["fold"] != 0]
    if prior is not None:
        if prior.get("schema") != "bl-residual-life-allocation/v1":
            raise ValueError("foreign prior allocation")
        old_catalog = prior.get("catalog", [])
        by_id = {r["life_id"]: r for r in catalog}
        if any(by_id.get(r["life_id"]) != r for r in old_catalog):
            raise ValueError("prior life metadata changed or disappeared")
        last = max(r["allocation_index"] for r in old_catalog)
        old_ids = {r["life_id"] for r in old_catalog}
        if any(r["allocation_index"] <= last for r in catalog if r["life_id"] not in old_ids):
            raise ValueError("new tranche inserted before frozen allocation prefix")
        heldout_ids = prior["doses"]["256"]["heldout_lives"]
        lookup = {r["life_id"]: r for r in rows}
        heldout = [lookup[life] for life in heldout_ids]
    doses = {}
    for dose in DOSES:
        count = dose - len(heldout)
        # Later tranches may not exist yet; never shrink a ready dose.
        doses[str(dose)] = {
            "ready": len(train) >= count,
            "train_lives": [r["life_id"] for r in train[:count]] if len(train) >= count else [],
            "heldout_lives": [r["life_id"] for r in heldout],
            "train_rescue_lives": count,
            "heldout_rescue_lives": 64,
        }
    return {
        "schema": "bl-residual-life-allocation/v1",
        "split_namespace": SPLIT_NAMESPACE,
        "catalog": sorted(catalog, key=lambda r: r["allocation_index"]),
        "folds": {r["life_id"]: world_fold(r["parent_world"]) for r in catalog},
        "parents": {r["life_id"]: r["parent_world"] for r in catalog},
        "doses": doses,
    }


def validate_rows(table: dict[str, np.ndarray]) -> None:
    """Refuse incomplete lineage, sentinel coercion and invalid observations."""
    required = {
        "local",
        "global",
        "scalars",
        "q",
        "legal",
        "action_mask",
        "bare",
        "died",
        "status",
        "mass",
        "ordinary",
        "natural_stream",
        "sampling_weight",
        "parent_world",
        "life_id",
        "row_id",
    }
    if not required <= table.keys():
        raise ValueError(f"missing F0 columns: {sorted(required - table.keys())}")
    n = len(table["q"])
    shapes = {"local": (n, 7, 31, 31), "global": (n, 4, 37, 37), "scalars": (n, 60)}
    shapes.update({k: (n, 6) for k in ("q", "legal", "action_mask", "died", "status", "mass")})
    shapes.update({k: (n,) for k in required - shapes.keys()})
    if n == 0 or any(table[k].shape != shape for k, shape in shapes.items()):
        raise ValueError("empty or malformed F0 column shapes")
    if any(table[k].dtype != np.uint8 for k in ("local", "global")):
        raise ValueError("raw raster planes must be uint8")
    if any(table[k].dtype != np.float32 for k in ("q", "scalars", "mass", "sampling_weight")):
        raise ValueError("q/scalars/mass/sampling_weight must preserve the fp32 recipe")
    if any(
        table[k].dtype != np.bool_
        for k in ("legal", "action_mask", "died", "ordinary", "natural_stream")
    ):
        raise ValueError("boolean columns must not coerce sentinel values")
    if (table["natural_stream"] & ~table["ordinary"]).any():
        raise ValueError("natural prevalence requires ordinary archive rows")
    if not np.isin(table["status"], STATUSES).all():
        raise ValueError("unknown branch status")
    for key in ("q", "scalars", "sampling_weight"):
        if not np.isfinite(table[key]).all():
            raise ValueError(f"non-finite {key}")
    if (table["sampling_weight"] <= 0).any():
        raise ValueError("positive natural-archive p/q weights required")
    exact = (table["status"] == "EXACT") & table["legal"]
    if not np.isfinite(table["mass"][exact]).all():
        raise ValueError("EXACT branches require finite mass")
    for key in ("parent_world", "life_id", "row_id"):
        if table[key].dtype.kind not in "US" or not (table[key] != "").all():
            raise ValueError("nonempty string lineage required")
    if len(set(table["row_id"].tolist())) != n:
        raise ValueError("duplicate row identity")
    parents: dict[str, str] = {}
    for life, parent in zip(table["life_id"], table["parent_world"]):
        if life in parents and parents[life] != parent:
            raise ValueError("life crosses parent-world ancestry")
        parents[life] = parent
    masks = table["action_mask"].copy()
    masks[~masks.any(axis=1), :3] = True
    if not np.issubdtype(table["bare"].dtype, np.integer) or not np.array_equal(
        np.where(masks, table["q"], -np.inf).argmax(axis=1), table["bare"]
    ):
        raise ValueError("bare action disagrees with served advisory/fallback rule")


def tensor_cache(table: dict[str, np.ndarray], base: Any, *, batch_size: int = 128) -> dict:
    """Cache frozen features once; callers supply a single ancestry partition."""
    from research.a100_a2.model import b8_features

    validate_rows(table)
    base.eval()
    device = next(base.parameters()).device
    parts = []
    with torch.no_grad():
        for start in range(0, len(table["q"]), batch_size):
            inputs = [
                torch.as_tensor(table[k][start : start + batch_size], device=device)
                for k in ("local", "global", "scalars")
            ]
            parts.append(b8_features(base, *inputs).cpu())
    cache = {
        k: torch.as_tensor(table[k].copy())
        for k in (
            "q",
            "legal",
            "action_mask",
            "bare",
            "died",
            "mass",
            "ordinary",
            "natural_stream",
            "sampling_weight",
        )
    }
    cache["features"] = torch.cat(parts)
    cache["status"] = torch.as_tensor(table["status"] == "EXACT")
    return cache
