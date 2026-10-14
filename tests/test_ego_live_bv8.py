"""``ego2s-bv8`` served live: the 30 v8 feature scalars (and the whole observation) are the sim's.

The parity harness's :class:`~src.simd_env.parity.PyRefGame` (real snakes + ``FoodManager``)
runs in bit-exact lockstep with :class:`~src.simd_env.grid_sim.GridBatchSim`; every frame,
for every living snake:

- :class:`~src.simd_env.ego_live_bv8.LiveV8Source` carries the sim's advisory mask and the
  full ``features=True`` :class:`~src.simd_env.v8_nb.V8Result` (every diagnostic), bitwise;
- the ``bv8`` observation built from the live view + live v8 source is byte-identical to the
  one built from the sim.

Bodies are grown long (the tight arena) so the v8 floods are capped and the head layer fires.
"""

from __future__ import annotations

import random
from dataclasses import fields, replace

import numpy as np
import pytest

from src.core.game_config import get_config, initialize_config
from src.simd_env.ego_live_adapter import game_state_to_ego_view
from src.simd_env.ego_live_bv8 import game_state_to_v8_source
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.parity import (
    PyRefGame,
    SurvivorPolicy,
    _install_v2_config,
    build_parity_config,
)
from src.simd_env.v8_nb import V8Result, v8_decide

_CFG_MID = build_parity_config(
    num_snakes=6, game_width=400, game_height=300, initial_food=60, max_food=80
)
_CFG_TIGHT = build_parity_config(
    num_snakes=8, game_width=250, game_height=180, initial_food=40, max_food=50
)


def _lockstep(cfg, seeds, frames, allow_respawn):
    saved = get_config()
    _install_v2_config(cfg)
    st = {"frames": 0, "rows": 0, "risky": 0, "not_spacious": 0, "capped": 0, "landing": 0}
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
            rng = np.random.default_rng(seed)
            for f in range(frames):
                actions = policy.actions(masks)
                ref.step(actions)
                sim.step(actions.reshape(1, cfg.num_snakes))
                masks = np.array(ref.action_masks(), dtype=bool)
                rows = np.argwhere(sim.get_alive())
                if not len(rows):
                    continue
                src = game_state_to_v8_source(ref)
                adv = sim.get_advisory_action_mask()
                np.testing.assert_array_equal(src.get_advisory_action_mask(), adv)
                m = adv[rows[:, 0], rows[:, 1]]
                q = rng.normal(size=(len(rows), 6)).astype(np.float32)
                a = v8_decide(sim, rows, q, m, 8.0, features=True)
                b = v8_decide(src, rows, q, m, 8.0, features=True)
                for fld in fields(V8Result):
                    x, y = getattr(a, fld.name), getattr(b, fld.name)
                    assert np.array_equal(x, y), (seed, f, fld.name)
                cfg_v = EgoRasterConfig(version="bv8")
                so = build_ego_raster(sim, cfg_v, backend="numba", rows=rows)
                lo = build_ego_raster(
                    game_state_to_ego_view(ref), cfg_v, backend="numba", rows=rows, v8_source=src
                )
                for key in ("local", "global", "scalars"):
                    assert so[key].tobytes() == lo[key].tobytes(), (seed, f, key)
                st["frames"] += 1
                st["rows"] += len(rows)
                st["risky"] += int(a.head_risky.any(axis=1).sum())
                st["not_spacious"] += int((a.counts < a.need[:, None]).any(axis=1).sum())
                st["capped"] += int((a.area >= a.area_cap[:, None]).any(axis=1).sum())
                st["landing"] += int((a.landing >= 0).any(axis=1).sum())
    finally:
        initialize_config(saved)
    return st


@pytest.mark.parametrize("cfg", [_CFG_MID, _CFG_TIGHT], ids=["mid", "tight"])
@pytest.mark.parametrize("allow_respawn", [False, True], ids=["train", "respawn"])
def test_bv8_live_v8_features_equal_sim_every_frame(cfg, allow_respawn):
    st = _lockstep(cfg, seeds=range(2), frames=300, allow_respawn=allow_respawn)
    assert st["frames"] > 0 and st["rows"] > 0
    # coverage: the head layer and the boost landing branch were exercised
    assert st["risky"] > 0 and st["landing"] > 0


def test_bv8_needs_a_v8_source_for_a_view():
    cfg = _CFG_MID
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(0)
        ref = PyRefGame(cfg)
        view = game_state_to_ego_view(ref)
        with pytest.raises(TypeError):
            build_ego_raster(view, EgoRasterConfig(version="bv8"), rows=np.zeros((1, 2), int))
    finally:
        initialize_config(saved)


def test_mask_slots_limit_the_live_mask():
    cfg = _CFG_MID
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(0)
        ref = PyRefGame(cfg)
        full = game_state_to_v8_source(ref).get_advisory_action_mask()
        part = game_state_to_v8_source(ref, mask_slots=[1]).get_advisory_action_mask()
        np.testing.assert_array_equal(part[0, 1], full[0, 1])
        assert not part[0, [0, 2, 3, 4, 5]].any() and full[0, 0].any()
    finally:
        initialize_config(saved)
