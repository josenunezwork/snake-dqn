"""Minimal served-replay port of open-stop-h32-nb (K=1) for controller ``exact-stop-l1-os``.

Ported (selection path only) from branch ``argon-hero-stage0-1103``:

* ``research/hero_stage0/open_stop.py`` @ fd7d90f6 -- ``evaluate_terminal_openness``,
  ``_eval_open_stop_branch_suffix`` and the selection half of
  ``dense_batch_open_stop_h32_nb`` (the controller that passed the 96-world
  ``t34-openstop-gate-20261104-v1`` gate, readout dc7f1016);
* ``research/hero_stage0/stage1a.py`` @ 3612d369 -- ``evaluation_order_nb``,
  ``branch_suffix_seed`` and ``reseed_branch_suffix``;
* constants from ``research/hero_stage0/protocol.py`` (K=1 arm
  ``open-stop-h32-nb-k1``: H=32, selection suffix (0,), non-boost first,
  alive-every-step + un-enclosed terminal certificate).

Deliberately NOT ported: the 4-suffix override *scoring* (gate diagnostics only;
it never changes the chosen action), K=2, aggregation and runners.

The only intended difference from the gate is the CRN namespace used to reseed
branch suffixes: the gate keyed it on its world-allocation namespace, which a
served replay session does not know, so the replay controller uses the fixed
``OPEN_STOP_CRN_NAMESPACE`` below (still action-independent and deterministic per
world seed / decision / suffix).
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np

from research.bl_k1.native import RowStats, branch_scope, configure
from research.bl_served_search import ceiling, exact_stop
from research.bl_served_search.ruler_operator import _clone_root, _root_bytes
from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared

OPEN_STOP_SOURCE = {
    "branch": "argon-hero-stage0-1103",
    "open_stop_py": "fd7d90f6",
    "stage1a_py": "3612d369",
    "gate_readout": "dc7f1016",
}
OPEN_STOP_CONTROLLER = "open-stop-h32-nb"
OPEN_STOP_BRANCH_HORIZON = 32
OPEN_STOP_SELECTION_SUFFIXES: tuple[int, ...] = (0,)  # K=1
OPEN_STOP_REGION_FRAC_MIN = 0.25
OPEN_STOP_POCKET_MIN_LENGTH = 160
OPEN_STOP_V8_NEED_CAP = 160
OPEN_STOP_V8_NEED_FLOOR = 4
OPEN_STOP_V8_LAMBDA = 8.0
OPEN_STOP_CRN_NAMESPACE = "t34-bank-v2-exact-stop-l1-os-crn-v1"

_CLOSED: dict[str, Any] = {
    "length": 0,
    "advisory_any": False,
    "v8_need": 0,
    "v8_best_safe_area": 0,
    "max_pocket_r": 0,
    "un_enclosed": False,
}


def evaluation_order_nb(legal: Sequence[int], bare: int, q: np.ndarray) -> list[int]:
    """Bare first, then legal non-boost (0..2) in Q-order, then boost (3..5) in Q-order."""
    actions = [int(action) for action in legal]
    if not actions or len(set(actions)) != len(actions) or bare not in actions:
        raise ValueError("exact stop requires unique legal actions containing bare")
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("exact stop requires finite six-action Q values")
    return [
        int(bare),
        *sorted(
            (action for action in actions if action != bare),
            key=lambda a: (int(a) >= 3, -float(q[a]), int(a)),
        ),
    ]


def branch_suffix_seed(namespace: str, world_seed: int, decision: int, suffix_id: int) -> int:
    """Deterministic 64-bit CRN seed for a branch suffix (action-independent)."""
    tag = (
        f"{namespace}:crn-branch:seed-{int(world_seed)}:"
        f"decision-{int(decision)}:suffix-{int(suffix_id)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(tag).digest()[:8], "big")


def reseed_branch_suffix(branch: Any, decision: int, suffix_id: int, *, namespace: str) -> int:
    """Reseed every environment RNG on ``branch.sim`` with the CRN suffix seed."""
    seed64 = branch_suffix_seed(namespace, int(branch.seed), int(decision), int(suffix_id))
    for env_idx, env_rng in enumerate(branch.sim._rngs):
        env_rng._rng.seed(seed64 ^ env_idx)
    return seed64


def evaluate_terminal_openness(sim: Any) -> dict[str, Any]:
    """Terminal un-enclosed certificate on hero slot 0.

    un_enclosed = (v8 best_safe_area >= min(max(4, L), 160))
                  AND (for L >= 160: max over valid actions of pocket R >= L).
    """
    from src.simd_env.ego_raster_b import region_sizes
    from src.simd_env.v8_nb import v8_decide

    if not bool(sim.alive[0, 0]):
        return dict(_CLOSED)
    length = int(sim.length[0, 0])
    v8_need = min(max(OPEN_STOP_V8_NEED_FLOOR, length), OPEN_STOP_V8_NEED_CAP)
    advisory = sim.get_advisory_action_mask()[0, 0].astype(bool)
    if not advisory.any():
        return {**_CLOSED, "length": length, "v8_need": v8_need}
    hero_row = np.array([[0, 0]], dtype=np.int64)
    v8_res = v8_decide(
        sim,
        hero_row,
        np.zeros((1, 6), dtype=np.float32),
        advisory.reshape(1, 6),
        OPEN_STOP_V8_LAMBDA,
        features=True,
    )
    best_safe_area = int(np.where(advisory, v8_res.area[0], 0).max())
    v8_open = bool(best_safe_area >= v8_need)
    reg = region_sizes(sim.ego_view(), hero_row, backend="numba")[0]
    denom = float(max(length, 1))
    reg_frac_dirs = (np.minimum(reg.astype(np.float64) / denom, 4.0) / 4.0).astype(np.float32)
    valid_r = [int(reg[a % 3]) for a in range(6) if bool(advisory[a])]
    valid_frac = [float(reg_frac_dirs[a % 3]) for a in range(6) if bool(advisory[a])]
    max_pocket_r = int(max(valid_r)) if valid_r else 0
    max_region_frac = float(max(valid_frac)) if valid_frac else 0.0
    pocket_r_ge_l = bool(max_pocket_r >= length)
    if pocket_r_ge_l != bool(max_region_frac >= OPEN_STOP_REGION_FRAC_MIN):
        raise RuntimeError(
            f"R >= L ({max_pocket_r} >= {length}) disagreed with region_frac >= 0.25"
        )
    pocket_open = pocket_r_ge_l if length >= OPEN_STOP_POCKET_MIN_LENGTH else True
    return {
        "length": length,
        "advisory_any": True,
        "v8_need": int(v8_need),
        "v8_best_safe_area": best_safe_area,
        "max_pocket_r": max_pocket_r,
        "un_enclosed": bool(v8_open and pocket_open),
    }


def _eval_branch_suffix(
    root: Any, action: int, decision: int, suffix_id: int, *, branch_horizon: int, namespace: str
) -> dict[str, Any]:
    """Roll ``action`` for ``branch_horizon`` steps; alive 1..H and un-enclosed at H."""
    saved_rng_states = [env_rng._rng.getstate() for env_rng in root.sim._rngs]
    try:
        reseed_branch_suffix(root, decision, suffix_id, namespace=namespace)
        with branch_scope(root.mix):
            branch = _clone_root(root)
            branch.sim._policy_phase_prepared = True
            step_prepared(branch, int(action))
            executed = 1
            alive_every_step = bool(branch.sim.alive[0, 0])
            while executed < int(branch_horizon) and alive_every_step:
                choose_and_step(branch, capture_obs=False)
                executed += 1
                if not bool(branch.sim.alive[0, 0]):
                    alive_every_step = False
            raw_survives = bool(alive_every_step and executed == int(branch_horizon))
            open_info = evaluate_terminal_openness(branch.sim) if raw_survives else dict(_CLOSED)
    finally:
        for env_rng, state in zip(root.sim._rngs, saved_rng_states):
            env_rng._rng.setstate(state)
    certified = bool(raw_survives and open_info["un_enclosed"])
    return {
        "suffix_id": int(suffix_id),
        "survives": certified,
        "certified": certified,
        "raw_survives": raw_survives,
        "frames_executed": int(executed),
        "frames_survived": int(branch_horizon) if raw_survives else max(0, int(executed) - 1),
        "terminal_openness": open_info,
    }


def open_stop_batch(
    world: Any,
    decision: int,
    q: np.ndarray,
    legal: np.ndarray,
    *,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    namespace: str = OPEN_STOP_CRN_NAMESPACE,
) -> dict[str, Any]:
    """open-stop-h32-nb K=1 selection on one prepared root; never mutates ``world``."""
    suffix = OPEN_STOP_SELECTION_SUFFIXES[0]
    root = _clone_root(world)
    before = _root_bytes(root)
    bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())

    def evaluate(action: int) -> dict[str, Any]:
        return _eval_branch_suffix(
            root,
            int(action),
            int(decision),
            suffix,
            branch_horizon=branch_horizon,
            namespace=namespace,
        )

    stats = RowStats()
    with (
        configure("kernel", stats),
        patch.object(exact_stop, "evaluation_order", evaluation_order_nb),
    ):
        # K=1: choose_all_fail adds no further suffixes, so it reduces to ceiling._choose.
        result = exact_stop.run(np.flatnonzero(legal).tolist(), bare, q, evaluate, ceiling._choose)
    if _root_bytes(root) != before:
        raise ValueError("open-stop branch mutated its parent RNG/carry")
    counters = stats.as_dict()
    if (
        counters["kernel_rows"] <= 0
        or counters["stock_rows"] != 0
        or counters["fallback_rows"] != 0
        or counters["noncandidate_rows"] != 0
    ):
        raise ValueError("missing native kernel per-decision coverage (open-stop)")
    return {
        **result,
        "bare": bare,
        "native_v8": counters,
        "root_sha256": ceiling.digest_value(before.hex()),
    }
