"""Observation-parity gate: one featurizer, two producers, identical output.

The featurizer (:mod:`src.simd_env.featurizer`) is fed by two independent
producers that must agree cell-for-cell so serve-time observations equal
train-time observations by construction (blueprint §0.10c):

- :func:`~src.simd_env.featurizer.obs_inputs_from_batch_sim` — the batched
  trainer path, reading a :class:`~src.simd_env.batch_sim.BatchSim`.
- :func:`~src.simd_env.live_adapter.game_state_to_obs_inputs` — the web/serve
  path, reading a live ``GameState``-shaped object (``Snake`` objects + a
  ``FoodManager``).

Strategy: reuse the golden-replay parity harness (:mod:`src.simd_env.parity`),
whose :class:`~src.simd_env.parity.PyRefGame` drives real ``Snake`` objects and a
real ``FoodManager`` in bit-exact lockstep with ``BatchSim`` (dynamics parity is
already proven in ``tests/test_simd_parity.py``). ``PyRefGame`` exposes exactly
the duck-typed surface the live adapter reads (``snakes``, ``food_manager``,
``_game_width``/``_game_height``, ``frame``), so it stands in for the live
``GameState``. At each frame we build :class:`ObsInputs` from BOTH producers on
the SAME world state, run the ONE featurizer, and assert the tactical raster,
strategic raster, scalars and mask are elementwise identical (rasters/mask
exactly; scalars within a tight float tolerance).

The configs are multi-snake and long-lived (survivor policy) so the parity
surface covers enemies in view (tactical channels 1/2, strategic enemy density,
nearest-enemy scalars) and at least one boosting snake (tactical channel 3 +
own-head boost bit) — see the coverage assertions.

Boosting-flag parity: the batch producer sources its ``boosting`` flag from
``BatchSim.get_boosted_this_step()`` — a persistent per-agent flag set to
``moved2`` (boost engaged AND eligible AND alive) in ``_move_all`` and, crucially,
NOT reset by the end-of-step ``_rebuild_traversed_from_heads`` mask rebuild. It
therefore equals the live adapter's ``Snake.is_boosting`` cell-for-cell at obs
time, so tactical channel 3 (enemy predicted-next cells), the own-head boost bit
and the enemy ``is_boosting`` scalars agree across producers with no reconcile.
:func:`test_batch_boosting_flag_matches_live` and
:func:`test_boosting_producer_divergence_is_isolated` pin this equivalence.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSim
from src.simd_env.featurizer import (
    CODE_ENEMY_HEAD,
    build_observations,
    obs_inputs_from_batch_sim,
)
from src.simd_env.live_adapter import game_state_to_obs_inputs
from src.simd_env.parity import (
    PyRefGame,
    SurvivorPolicy,
    _install_v2_config,
    build_parity_config,
)

# Featurizer knobs shared by both producers so length/hunger scalars line up.
_MAX_FRAMES = 5000
_STARVATION_MAX = 500
_MAX_LENGTH = 400
# Scalars are float32; every field is an exact rational of integer cell counts,
# so the two producers agree far tighter than this — the tolerance only absorbs
# float32 rounding of identical arithmetic.
_SCALAR_ATOL = 1e-6

# Parity configs: multi-snake, long-lived, enemy-dense. The survivor policy keeps
# snakes alive and boosting so channels 1/2/3 and the enemy scalars are exercised.
_CFG_MID = build_parity_config(
    num_snakes=6, game_width=400, game_height=300, initial_food=40, max_food=50
)
_CFG_TIGHT = build_parity_config(
    num_snakes=8, game_width=250, game_height=180, initial_food=15, max_food=20
)


# ---------------------------------------------------------------------------
# Lockstep driver
# ---------------------------------------------------------------------------
def _both_obs_inputs(sim: BatchSim, ref: PyRefGame):
    """Build :class:`ObsInputs` from the batch sim (env 0) and the live ref.

    The batch producer reads ``sim`` directly; the live producer reads
    ``ref`` (a ``PyRefGame`` that exposes the ``GameState`` duck-typed surface
    the adapter consumes). Both are pointed at the SAME frame's world state.
    """
    batch_inp = obs_inputs_from_batch_sim(sim, max_frames=_MAX_FRAMES)
    live_inp = game_state_to_obs_inputs(
        ref,
        max_frames=_MAX_FRAMES,
        starvation_max=_STARVATION_MAX,
        max_length=_MAX_LENGTH,
    )
    return batch_inp, live_inp


def _first_raster_divergence(a: np.ndarray, b: np.ndarray) -> Tuple[int, ...]:
    """Return the index of the first differing element (for a readable report)."""
    diff = np.argwhere(a != b)
    return tuple(int(x) for x in diff[0]) if diff.size else ()


def _run_lockstep(
    cfg,
    seeds,
    num_frames: int,
    reconcile_boosting: bool = False,
) -> Dict[str, object]:
    """Drive BatchSim + PyRefGame in lockstep and compare per-frame observations.

    At every frame both producers fill an :class:`ObsInputs` from the identical
    world state; the single featurizer renders both; the four outputs are
    asserted elementwise identical (rasters/mask exactly, scalars within
    :data:`_SCALAR_ATOL`).

    Args:
        cfg: Parity config (rectangular, mechanics v2).
        seeds: Env seeds to sweep serially.
        num_frames: Frames per seed.
        reconcile_boosting: When True, force the batch ``boosting`` to equal the
            live ``boosting`` before featurizing. Retained only as an escape hatch;
            the batch producer now sources ``boosting`` from a persistent field
            (see the module docstring), so the raw producers already agree and the
            default (False) compares them directly.

    Returns:
        Coverage counters: total frames, frames with a boosting snake, frames
        with an enemy head in some agent's tactical view.
    """
    saved = get_config()
    _install_v2_config(cfg)
    total = 0
    boosting_frames = 0
    enemy_in_view_frames = 0
    try:
        for seed in seeds:
            random.seed(seed)
            ref = PyRefGame(cfg)
            bat = BatchSim(replace(cfg, num_envs=1), seeds=[seed], train_mode=True)
            policy = SurvivorPolicy(seed, cfg.num_snakes)
            ref_masks = np.array(ref.action_masks(), dtype=bool)

            for f in range(num_frames):
                actions = policy.actions(ref_masks)
                ref.step(actions)
                bat.step(actions.reshape(1, cfg.num_snakes))
                ref_masks = np.array(ref.action_masks(), dtype=bool)

                batch_inp, live_inp = _both_obs_inputs(bat, ref)
                if live_inp.boosting.any():
                    boosting_frames += 1
                if reconcile_boosting:
                    batch_inp = replace(batch_inp, boosting=live_inp.boosting.copy())

                batch_obs = build_observations(batch_inp, mask=bat.get_action_mask())
                live_obs = build_observations(live_inp, mask=ref_masks[None])

                if (batch_obs["tactical_uint8"][0, :, 0] == CODE_ENEMY_HEAD).any():
                    enemy_in_view_frames += 1

                _assert_frame_parity(batch_obs, live_obs, seed, f)
                total += 1
    finally:
        initialize_config(saved)

    return {
        "total_frames": total,
        "boosting_frames": boosting_frames,
        "enemy_in_view_frames": enemy_in_view_frames,
    }


def _assert_frame_parity(batch_obs: dict, live_obs: dict, seed: int, frame: int) -> None:
    """Assert one frame's four observation fields match across producers."""
    for key in ("tactical_uint8", "strategic_uint8", "mask"):
        b = batch_obs[key]
        live_val = live_obs[key]
        if not np.array_equal(b, live_val):
            idx = _first_raster_divergence(b, live_val)
            raise AssertionError(
                f"seed {seed} frame {frame}: {key} diverged at {idx} "
                f"batch={b[idx]!r} live={live_val[idx]!r}"
            )

    bs = np.asarray(batch_obs["scalars"], dtype=np.float64)
    ls = np.asarray(live_obs["scalars"], dtype=np.float64)
    if not np.allclose(bs, ls, atol=_SCALAR_ATOL, rtol=0.0):
        idx = tuple(int(x) for x in np.argwhere(~np.isclose(bs, ls, atol=_SCALAR_ATOL))[0])
        raise AssertionError(
            f"seed {seed} frame {frame}: scalars diverged at {idx} "
            f"batch={bs[idx]!r} live={ls[idx]!r} delta={bs[idx] - ls[idx]:.3e}"
        )


# ---------------------------------------------------------------------------
# Core parity gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [_CFG_MID, _CFG_TIGHT])
def test_obs_parity_batch_vs_live(cfg):
    """Featurizer output is identical whether fed by BatchSim or the live adapter.

    Steps both sims in lockstep on the SAME state and asserts the tactical
    raster, strategic raster, scalars and mask are elementwise identical every
    frame (rasters/mask exactly, scalars within a tight float tolerance). No
    field is reconciled: the batch ``boosting`` flag now matches live by
    construction (see the module docstring), so this gates the full featurizer.
    """
    stats = _run_lockstep(cfg, seeds=range(4), num_frames=300)
    assert stats["total_frames"] >= 4 * 300


def test_obs_parity_covers_enemies_and_boosting():
    """Guard against a vacuous pass: the sweep must exercise enemies + boosting.

    A parity gate that only ever sees a lone surviving snake never populates the
    enemy channels/scalars or the boost-dependent paths. This asserts the sweep
    actually rendered enemy heads in view and had at least one boosting snake, so
    tactical channels 1/2 (enemy body/head), the nearest-enemy scalars and the
    boost paths are genuinely covered.
    """
    stats = _run_lockstep(_CFG_MID, seeds=range(4), num_frames=400)
    assert stats["enemy_in_view_frames"] > 0, "no enemy heads ever entered a tactical view"
    assert stats["boosting_frames"] > 0, "no snake ever boosted — channel 3 path uncovered"


# ---------------------------------------------------------------------------
# Boosting-flag parity: the batch flag survives to obs time and matches live
# ---------------------------------------------------------------------------
def test_batch_boosting_flag_matches_live():
    """The batch ``boosting`` flag is persistent and equals ``Snake.is_boosting``.

    ``BatchSim.step`` ends with ``_rebuild_traversed_from_heads``, which resets
    the internal ``_trav_valid[:, :, 1]`` traversed-head mask. The batch producer
    therefore sources ``boosting`` from the persistent ``get_boosted_this_step()``
    accessor instead, which survives that reset. This test drives snakes into
    confirmed boosts and asserts the batch producer's ``boosting`` equals the
    live ``Snake.is_boosting`` (alive-gated) cell-for-cell on every frame — the
    executable spec that the train/serve boost fields cannot diverge.
    """
    cfg = _CFG_MID
    saved = get_config()
    _install_v2_config(cfg)
    try:
        seed = 0
        random.seed(seed)
        ref = PyRefGame(cfg)
        bat = BatchSim(replace(cfg, num_envs=1), seeds=[seed], train_mode=True)
        policy = SurvivorPolicy(seed, cfg.num_snakes)
        ref_masks = np.array(ref.action_masks(), dtype=bool)

        saw_confirmed_boost = False
        for f in range(500):
            actions = policy.actions(ref_masks)
            ref.step(actions)
            bat.step(actions.reshape(1, cfg.num_snakes))
            ref_masks = np.array(ref.action_masks(), dtype=bool)

            live_boosting = np.array(
                [bool(getattr(s, "is_boosting", False)) and s.is_alive for s in ref.snakes]
            )
            batch_inp = obs_inputs_from_batch_sim(bat, max_frames=_MAX_FRAMES)
            assert np.array_equal(
                batch_inp.boosting[0], live_boosting
            ), f"frame {f}: batch boosting {batch_inp.boosting[0]} != live {live_boosting}"
            if live_boosting.any():
                saw_confirmed_boost = True

        assert saw_confirmed_boost, "test did not exercise a boosting snake"
    finally:
        initialize_config(saved)


def test_boosting_producer_divergence_is_isolated():
    """The raw producers agree even on boosting frames — including boost fields.

    Compares the raw producers (no reconciliation) and asserts ZERO divergence in
    the tactical raster, strategic raster, scalars and mask on every frame,
    specifically including the boosting-derived fields that were previously the
    sole source of divergence: tactical channel 3 (enemy predicted-next) + the
    own-head boost bit, and scalar columns 18/22 (enemy ``is_boosting`` summaries).
    The sweep is asserted to actually contain boosting frames so this is not
    vacuous. This proves the featurizer is fully producer-agnostic once the batch
    ``boosting`` flag is sourced from the persistent accessor (module docstring).
    """
    cfg = _CFG_MID
    saved = get_config()
    _install_v2_config(cfg)
    boosting_frames = 0
    leaked_divs: List[str] = []
    try:
        for seed in range(3):
            random.seed(seed)
            ref = PyRefGame(cfg)
            bat = BatchSim(replace(cfg, num_envs=1), seeds=[seed], train_mode=True)
            policy = SurvivorPolicy(seed, cfg.num_snakes)
            ref_masks = np.array(ref.action_masks(), dtype=bool)

            for f in range(300):
                actions = policy.actions(ref_masks)
                ref.step(actions)
                bat.step(actions.reshape(1, cfg.num_snakes))
                ref_masks = np.array(ref.action_masks(), dtype=bool)

                batch_inp, live_inp = _both_obs_inputs(bat, ref)
                boosting = bool(live_inp.boosting.any())
                if boosting:
                    boosting_frames += 1

                batch_obs = build_observations(batch_inp, mask=bat.get_action_mask())
                live_obs = build_observations(live_inp, mask=ref_masks[None])

                # All four fields must agree elementwise on every frame.
                for key in ("tactical_uint8", "strategic_uint8", "mask"):
                    if not np.array_equal(batch_obs[key], live_obs[key]):
                        leaked_divs.append(f"seed {seed} frame {f}: {key} diverged")

                sdiff = ~np.isclose(
                    batch_obs["scalars"].astype(np.float64),
                    live_obs["scalars"].astype(np.float64),
                    atol=_SCALAR_ATOL,
                    rtol=0.0,
                )
                if sdiff.any():
                    cols = sorted(int(c) for c in np.unique(np.argwhere(sdiff)[:, -1]))
                    leaked_divs.append(f"seed {seed} frame {f}: scalars diverged at cols {cols}")
    finally:
        initialize_config(saved)

    assert not leaked_divs, f"producers diverged (boost fields not in parity): {leaked_divs[:5]}"
    assert boosting_frames > 0, "no boosting frames — parity check is vacuous"
