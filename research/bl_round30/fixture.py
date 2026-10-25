"""Hermetic R30 consumer fixture; it has no label-collection imports."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def synthetic_table(base: Any) -> dict[str, np.ndarray]:
    """Build deterministic synthetic rows for canary plumbing only."""
    rng = np.random.default_rng(3403)
    count = 12
    table = {
        "local": rng.integers(0, 256, (count, 7, 31, 31), dtype=np.uint8),
        "global": rng.integers(0, 256, (count, 4, 37, 37), dtype=np.uint8),
        "scalars": rng.normal(size=(count, 60)).astype(np.float32),
    }
    with torch.no_grad():
        table["q"] = base(
            *[torch.as_tensor(table[key]) for key in ("local", "global", "scalars")]
        ).numpy()
    table.update(
        legal=np.ones((count, 6), bool),
        action_mask=np.ones((count, 6), bool),
        bare=table["q"].argmax(axis=1),
        died=rng.random((count, 6)) < 0.5,
        status=np.full((count, 6), "EXACT"),
        mass=rng.random((count, 6)).astype(np.float32) * 10,
        ordinary=np.arange(count) % 2 == 0,
        sampling_prob=np.ones(count, np.float32),
        sampling_weight=np.ones(count, np.float32),
        natural_stream=np.arange(count) % 2 == 0,
        parent_world=np.asarray([f"synthetic-{index // 3}" for index in range(count)]),
        life_id=np.asarray([f"synthetic-life-{index // 3}" for index in range(count)]),
        row_id=np.asarray([f"synthetic-row-{index}" for index in range(count)]),
    )
    return table
