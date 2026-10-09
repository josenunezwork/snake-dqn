"""M1(iii) parity tier 2: the ego2s observation built from the live game is the sim's, bitwise.

Two producers feed ONE featurizer (:func:`src.simd_env.ego_raster.build_ego_raster`):

- :meth:`GridBatchSim.ego_view` (training / teacher data);
- :func:`src.simd_env.ego_live_adapter.game_state_to_ego_view` (serving), here reading the
  parity harness's :class:`~src.simd_env.parity.PyRefGame` (real ``Snake`` objects and a
  real ``FoodManager`` driven in bit-exact lockstep with ``BatchSim``; the same duck-typed
  surface a live ``GameState`` exposes).

Every frame, for every living snake, the local planes, global planes and scalars must be
byte-identical (no float tolerance), on both featurizer backends. The gate-world version of
this check, on a real ``GameState`` playing frp3-s12+v8 at H5000, is
``research/redesign_scope_20261007/ego_live_identity.py``.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from src.core.game_config import get_config, initialize_config
from src.simd_env.ego_live_adapter import game_state_to_ego_view
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.parity import (
    PyRefGame,
    SurvivorPolicy,
    _install_v2_config,
    build_parity_config,
)

_CFG_MID = build_parity_config(
    num_snakes=6, game_width=400, game_height=300, initial_food=40, max_food=50
)
_CFG_TIGHT = build_parity_config(
    num_snakes=8, game_width=250, game_height=180, initial_food=15, max_food=20
)


def _assert_identical(sim_obs: dict, live_obs: dict, where: str) -> None:
    for key in ("local", "global", "scalars"):
        a, b = sim_obs[key], live_obs[key]
        assert a.dtype == b.dtype and a.shape == b.shape, (where, key)
        if not np.array_equal(a, b):
            idx = tuple(int(x) for x in np.argwhere(a != b)[0])
            raise AssertionError(f"{where}: {key} differs at {idx}: sim={a[idx]} live={b[idx]}")
        if key == "scalars":  # bitwise, not just equal values
            assert a.tobytes() == b.tobytes(), (where, key)


def _lockstep(cfg, seeds, frames, allow_respawn):
    saved = get_config()
    _install_v2_config(cfg)
    stats = {"frames": 0, "rows": 0, "boost": 0, "corpse": 0, "enemy_heads": 0, "deaths": 0}
    stats["enemy_next"] = 0
    try:
        for seed in seeds:
            random.seed(seed)
            ref = PyRefGame(cfg, allow_respawn=allow_respawn)
            sim = GridBatchSim(
                replace(cfg, num_envs=1),
                seeds=[seed],
                train_mode=not allow_respawn,
                allow_respawn=allow_respawn,
            )
            policy = SurvivorPolicy(seed, cfg.num_snakes, boost_prob=0.3)
            masks = np.array(ref.action_masks(), dtype=bool)
            for f in range(frames):
                actions = policy.actions(masks)
                ref.step(actions)
                sim.step(actions.reshape(1, cfg.num_snakes))
                masks = np.array(ref.action_masks(), dtype=bool)
                rows = np.argwhere(sim.get_alive())
                live_alive = np.array([s.is_alive for s in ref.snakes])
                np.testing.assert_array_equal(sim.get_alive()[0], live_alive)
                view = game_state_to_ego_view(ref)
                for backend in ("numpy", "numba"):
                    for version in ("draft", "b"):
                        cfg_v = EgoRasterConfig(version=version)
                        a = build_ego_raster(sim, cfg_v, backend=backend, rows=rows)
                        b = build_ego_raster(view, cfg_v, backend=backend, rows=rows)
                        _assert_identical(a, b, f"seed {seed} frame {f} {backend} {version}")
                stats["enemy_next"] += int((a["local"][:, 6] > 0).any(axis=(1, 2)).sum())
                stats["frames"] += 1
                stats["rows"] += len(rows)
                stats["boost"] += int(sim.get_boosted_this_step().sum())
                stats["corpse"] += int((a["local"][:, 3] == 255).any(axis=(1, 2)).sum())
                stats["enemy_heads"] += int((a["local"][:, 2] > 0).any(axis=(1, 2)).sum())
                stats["deaths"] += int(sim.get_done().sum())
    finally:
        initialize_config(saved)
    return stats


@pytest.mark.parametrize("cfg", [_CFG_MID, _CFG_TIGHT], ids=["mid", "tight"])
@pytest.mark.parametrize("allow_respawn", [False, True], ids=["train", "respawn"])
def test_ego2s_live_equals_sim_every_frame(cfg, allow_respawn):
    stats = _lockstep(cfg, seeds=range(3), frames=250, allow_respawn=allow_respawn)
    assert stats["frames"] == 750
    # Coverage: the planes that differ most between producers were exercised.
    assert stats["corpse"] > 0 and stats["enemy_heads"] > 0 and stats["deaths"] > 0
    assert stats["enemy_next"] > 0
    if cfg is _CFG_MID:  # the tight arena keeps snakes too short to boost
        assert stats["boost"] > 0


def test_live_adapter_rejects_out_of_arena_bodies():
    cfg = _CFG_MID
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(0)
        ref = PyRefGame(cfg)
        ref.snakes[0].segments[0] = (-10, 0)
        with pytest.raises(ValueError, match="outside the arena"):
            game_state_to_ego_view(ref)
    finally:
        initialize_config(saved)
