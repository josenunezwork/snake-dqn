"""Recover the frozen F0 advisory mask from its lossless local collision view.

No labels, branch outcomes, network forwards or simulator steps are consumed.
This inverse applies to reachable living GridBatchSim states, the sole F0 source.
The pixel and scalar conventions are frozen by ego_raster.py; the collision
inequalities mirror grid_sim_nb._action_masks_kernel, not physical legality.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def advisory_from_row(row: Mapping[str, Any], *, boost_cost: int = 3) -> np.ndarray:
    """Return six exact advisory bits, preserving the all-false mask.

    Raster owner/slot selection is the same as the advisory kernel, including
    overlaps. Enemy TTL > 1 blocks; a length-one enemy head also blocks. For
    the hero, n = L - pending growth determines normal/boost tail release.
    Current head and neck cannot occupy any of the six candidate cells: the
    head is at the origin and the neck is behind the current direction.

    Clipping TTL at 255 loses no threshold information. If either own-body
    threshold exceeds 255, the pending-growth invariant implies every occupied
    own cell is above that threshold. Comparing with min(threshold,255) is
    therefore exact. The caller must retain provenance of reachable F0 states;
    this function does not validate arbitrary hand-painted bodies.
    """
    if type(boost_cost) is not int or boost_cost <= 0:
        raise ValueError("positive frozen boost cost required")
    local = np.asarray(row["local"])
    scalars = np.asarray(row["scalars"], dtype=np.float32)
    length = row.get("length")
    if (
        local.shape != (7, 31, 31)
        or not np.issubdtype(local.dtype, np.integer)
        or (local < 0).any()
        or (local > 255).any()
        or scalars.shape != (60,)
        or not np.isfinite(scalars).all()
        or type(length) is not int
        or length <= 0
    ):
        raise ValueError("frozen living F0 observation required")
    pending = int(round(float(scalars[2]) * 16))
    phase = int(round(float(scalars[4]) * boost_cost))
    count = length - pending
    if (
        not 1 <= count <= length
        or scalars[1] != np.float32(length / 1000)
        or scalars[2] != np.float32(pending / 16)
        or not 0 <= phase < boost_cost
        or scalars[4] != np.float32(phase / boost_cost)
        or scalars[3] != np.float32(length >= 5)
    ):
        raise ValueError("F0 length/growth/boost scalar identity differs")
    normal_last = min(count + 1, length) - 2
    final = min(count + 2, length)
    if phase + 1 >= boost_cost:
        final = min(final, max(1, length - 1))
    normal_threshold = min(length - normal_last, 255)
    boost_threshold = min(length - final + 3, 255)

    def blocked(cell: tuple[int, int], threshold: int) -> bool:
        r, c = cell
        return bool(
            local[4, r, c] or local[1, r, c] > 1 or local[2, r, c] or local[0, r, c] >= threshold
        )

    first = ((23, 14), (22, 15), (23, 16))
    second = ((23, 13), (21, 15), (23, 17))
    normal = [not blocked(cell, normal_threshold) for cell in first]
    boost = [
        normal[d]
        and bool(scalars[3])
        and not blocked(first[d], boost_threshold)
        and not blocked(second[d], boost_threshold)
        for d in range(3)
    ]
    return np.asarray(normal + boost, dtype=bool)
