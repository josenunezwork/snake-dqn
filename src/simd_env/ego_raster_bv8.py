"""``ego2s-bv8`` extras: the v8 veto's per-action diagnostics as 30 extra scalars (M4-B).

``ego2s-bv8`` = ``ego2s-b`` (local ``(7, 31, 31)``, global ``(4, 37, 37)``, 30 scalars) with
:func:`src.simd_env.v8_nb.v8_feature_planes` appended to the scalars, flattened action-major
(``scalars[30 + 5 * a + f]`` = feature ``f`` of relative action ``a``; features in
:data:`src.simd_env.v8_nb.FEATURE_NAMES` order: capped landing-area score, head-risky,
landing_ok, v5-eligible, spacious), 60 scalars in all, every extra in ``[0, 1]``.

The features are evaluated eagerly (``features=True``), so they do not depend on Q (zeros
are passed); they do depend on the hard mask, which is the sim's advisory (exact safe-action)
mask, the same mask the v8 shield reads from ``Vector61Runtime.selection``. They need a
:class:`~src.simd_env.grid_sim.GridBatchSim` (the bodies), never an ``EgoGridView``, so a
bv8 student is evaluated on the SIMD grid engine only (no live serving path yet).
Dead rows observe zeros, as every other scalar.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["N_V8_SCALARS", "v8_scalars"]

N_V8_SCALARS = 30


def v8_scalars(sim, rows: Optional[np.ndarray], mask: Optional[np.ndarray] = None, lam=8.0):
    """``((N, 30) float32 features, V8Result or None)`` for ``rows`` (``None``: all ``E*S``).

    Args:
        sim: The decision-time GridBatchSim / BatchSim (read only).
        rows: ``(N, 2)`` ``(env, slot)`` rows, or ``None`` for every row in ``(E, S)`` order.
        mask: ``(N, 6)`` hard mask for the rows (default: the sim's advisory mask). Ignored
            for ``rows=None``.
        lam: v8 ``lambda`` (does not affect the features; kept for the reused decision).

    Returns:
        The features and the ``features=True`` :class:`~src.simd_env.v8_nb.V8Result` for
        ``rows`` (``None`` when ``rows`` is ``None`` or any row is dead), so a caller can take
        the v8 action for any Q with :func:`~src.simd_env.v8_nb.v8_redecide` without another flood.
    """
    from src.simd_env.v8_nb import v8_decide, v8_feature_planes

    if not hasattr(sim, "bodies"):
        raise TypeError("ego2s-bv8 features need a GridBatchSim / BatchSim, not an EgoGridView")
    whole = rows is None
    if whole:
        E, S = sim.alive.shape
        rows = np.stack(np.meshgrid(np.arange(E), np.arange(S), indexing="ij"), -1).reshape(-1, 2)
        mask = None
    rows = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape(-1, 2))
    if mask is None:
        mask = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
    mask = np.asarray(mask, dtype=bool).reshape(-1, 6)
    alive = np.asarray(sim.alive[rows[:, 0], rows[:, 1]], dtype=bool)
    out = np.zeros((len(rows), N_V8_SCALARS), dtype=np.float32)
    live = np.flatnonzero(alive)
    res = None
    if len(live):
        q0 = np.zeros((len(live), 6), dtype=np.float32)
        res = v8_decide(sim, rows[live], q0, mask[live], lam, features=True)
        out[live] = v8_feature_planes(res).reshape(len(live), N_V8_SCALARS)
    if whole or len(live) != len(rows):
        res = None  # only returned when it lines up with ``rows`` (all alive)
    return out, res
