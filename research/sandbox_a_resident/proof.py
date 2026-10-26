"""Same-platform, same-shape actor/optimizer/world/RNG resume proof before timing."""

from __future__ import annotations

import hashlib
from dataclasses import is_dataclass
from typing import Any

import torch

from research.sandbox_a_resident.core import Recipe, ResidentAssay
from src.model.ego2s_network import Ego2sNet


def signature(value: Any) -> str:
    """Canonical complete-state bytes, independent of torch.save storage IDs."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for nested in item:
                visit(nested)
        elif is_dataclass(item):
            visit(vars(item))
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def prove(b8: Ego2sNet, device: str) -> dict:
    """A complete real update, restore, then identical next reset rows and update."""
    recipe = Recipe(
        worlds=2,
        horizon=3,
        minibatch=3,
        episode_limit=2,
        max_episodes=24,
        tiny=True,
        seed_offset=9_000_000,
    )
    runner = ResidentAssay(recipe, b8, device)
    runner.update(runner.collect())
    frozen = runner.snapshot()
    expected_rows = runner.collect()
    runner.update(expected_rows)
    expected = signature(runner.snapshot())
    runner.restore(frozen)
    actual_rows = runner.collect()
    if signature(expected_rows) != signature(actual_rows):
        raise RuntimeError("INVALID: resume behavior/masks/GAE rows differ")
    runner.update(actual_rows)
    actual = signature(runner.snapshot())
    if actual != expected:
        raise RuntimeError("INVALID: resume actor/optimizer/world/RNG state differs")
    return {
        "status": "PASS",
        "kind": "same-platform bit equality",
        "worlds": 2,
        "horizon": 3,
        "state_sha256": actual,
        "next_rows_sha256": signature(actual_rows),
        "cross_platform_claim": False,
    }
