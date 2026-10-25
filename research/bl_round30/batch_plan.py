"""Freeze scientific sampler lists without constructing features or fitting.

Only G1-approved training rows reach this helper through prepare. Outcomes are
used solely for the unchanged rescue stratum predicate, never scored/summarized.
The L4 worker independently reproduces these lists before numerical fitting.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from research.bl_ppo.exports import export_bytes, export_json
from research.bl_round30.materialize import _npz_bytes
from research.bl_round30.objective import (
    ObjectiveRecipe,
    _life_uniform_indices,
    _stratum_groups,
    resolved_action_mask,
)


def sampler_indices(table: dict, *, seed: int, updates: int = 2000) -> torch.Tensor:
    """Use the original life sampler with no features, coefficients, or loss."""
    lives = {life: index for index, life in enumerate(sorted(set(table["life_id"].tolist())))}
    life = torch.tensor([lives[x] for x in table["life_id"].tolist()], dtype=torch.int64)
    ordinary = torch.as_tensor(table["ordinary"].copy())
    known = (
        torch.as_tensor(table["status"] == "EXACT")
        & torch.as_tensor(table["legal"].copy())
        & resolved_action_mask(torch.as_tensor(table["action_mask"].copy()))
    )
    died = torch.as_tensor(table["died"].copy())
    rescue = (~ordinary) & (known & ~died).any(1) & (known & died).any(1)
    groups = (
        _stratum_groups({"life_id": life}, rescue),
        _stratum_groups({"life_id": life}, ordinary),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack(
        [
            torch.cat(
                (
                    _life_uniform_indices(groups[0], 128, generator),
                    _life_uniform_indices(groups[1], 128, generator),
                )
            )
            for _ in range(updates)
        ]
    )


def freeze_plan(train: Path, output: Path, *, source_commit: str, train_sha256: str) -> dict:
    """Create exact seed3403/3404 lists after approved partition verification."""
    if hashlib.sha256(train.read_bytes()).hexdigest() != train_sha256:
        raise ValueError("training bytes changed before sampler freeze")
    names = (
        "life_id",
        "source_life_id",
        "row_id",
        "ordinary",
        "legal",
        "action_mask",
        "status",
        "died",
    )
    with np.load(train, allow_pickle=False) as archive:
        table = {name: archive[name] for name in names}
    fit = sampler_indices(table, seed=3403).numpy()
    audit = sampler_indices(table, seed=3404).numpy()
    rows = len(table["row_id"])
    payload = _npz_bytes(
        {
            "fit_indices": fit,
            "audit_indices": audit,
            "row_ids": table["row_id"],
            "source_life_ids": table["source_life_id"],
            "fit_multiplicities": np.bincount(fit.reshape(-1), minlength=rows),
            "audit_multiplicities": np.bincount(audit.reshape(-1), minlength=rows),
        }
    )
    export_bytes(output / "sampler-plan.npz", payload)
    contract = {
        "schema": "r30-scientific-sampler-plan/v1",
        "source_commit": source_commit,
        "train_sha256": train_sha256,
        "plan_sha256": hashlib.sha256(payload).hexdigest(),
        "recipe": asdict(ObjectiveRecipe()),
        "fit_seed": 3403,
        "audit_seed": 3404,
        "batches_per_plan": 2000,
        "rescue_per_batch": 128,
        "ordinary_per_batch": 128,
        "row_count": rows,
        "features_constructed": False,
        "fit_performed": False,
        "feature_cache": (
            "L4-only; frozen B8/features; runtime cache SHA exported in indivisible unit"
        ),
    }
    export_json(output / "sampler-plan.json", contract)
    return contract


def verify_plan_bytes(path: Path, table: dict, *, updates: int) -> None:
    """Check the exact frozen production lists before any worker fitting."""
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["row_ids"], table["row_id"]):
            raise ValueError("frozen sampler row identity differs")
        if not np.array_equal(archive["source_life_ids"], table["source_life_id"]):
            raise ValueError("frozen sampler life identity differs")
        for key, seed in (("fit_indices", 3403), ("audit_indices", 3404)):
            indices = archive[key]
            if indices.shape != (2000, 256) or not np.array_equal(
                indices[:updates], sampler_indices(table, seed=seed, updates=updates).numpy()
            ):
                raise ValueError("frozen scientific sampler differs from runtime")
