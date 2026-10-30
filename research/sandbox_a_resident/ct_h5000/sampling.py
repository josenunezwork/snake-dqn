"""Bounded Algorithm R admissions and deterministic, outcome-blind quota sampling."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


def reservoir_admissions(
    proposals: Sequence[tuple[int, tuple[str, str, int]]],
    *,
    reservoirs: Mapping[tuple[str, str, int], list[Any]],
    seen_counts: MutableMapping[tuple[str, str, int], int],
    capacities: Mapping[str, int],
    rng: np.random.Generator,
) -> list[tuple[int, tuple[str, str, int], int | None]]:
    """Reserve slots immediately; materialize only the final writer of each slot.

    This is exactly serial Algorithm R, including multiple replacements in one
    tick. A newly reserved slot is counted before the next proposal. No tail
    outside [0, cap) can be appended or become permanently irreplaceable.
    """
    sizes = {key: len(items) for key, items in reservoirs.items()}
    pending: dict[tuple[tuple[str, str, int], int], int] = {}
    for world, cell in proposals:
        cap = capacities[cell[0]]
        if cap <= 0 or sizes[cell] > cap:
            raise ValueError("positive capacity and bounded input reservoir required")
        seen = seen_counts[cell] + 1
        seen_counts[cell] = seen
        if sizes[cell] < cap:
            slot = sizes[cell]
            sizes[cell] += 1
        else:
            slot = int(rng.integers(0, seen))
            if slot >= cap:
                continue
        pending[(cell, slot)] = world
    # Append new slots in increasing order, then replace existing slots.
    return [
        (world, cell, None if slot >= len(reservoirs[cell]) else slot)
        for (cell, slot), world in sorted(pending.items())
    ]


def quota_sample(items: Sequence[Any], count: int, *, key: tuple[Any, ...]) -> list[Any]:
    """Sample a uniform subset before stable chronological output ordering.

    Identity sorting makes the subset invariant to reservoir storage order. The
    seed includes only frozen cell identity; no future survival enters selection.
    """
    if not 0 <= count <= len(items):
        raise ValueError("quota outside available candidate count")
    ordered = sorted(items, key=lambda c: (c.tick, c.world_id, c.life_id))
    seed = int.from_bytes(
        hashlib.sha256(f"a100-dev/t34-ct-h5000-1029/quota/v1/{key!r}".encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(ordered), size=count, replace=False)
    return sorted((ordered[int(i)] for i in chosen), key=lambda c: (c.tick, c.world_id, c.life_id))
