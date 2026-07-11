"""Golden-replay parity tests: live Python game vs. the batched sim (P2 gate).

Asserts BIT-EXACT dynamics parity between the reference live Python game
(``src.game``, mechanics v2 + reward v2) and the vectorized ``BatchSim`` over a
seed sweep, comparing every load-bearing world quantity per frame: head/body
positions, alive/deaths, kill attribution, the ordered food set, per-snake
rewards, and per-agent action masks.

Parity milestone achieved (see ``src/simd_env/parity.py`` and the module
docstring): rectangular arena, mechanics v2 + reward v2, ``train_mode=True``
(terminal deaths, ``allow_respawn=False`` so the respawn RNG branch never fires).
The scripted-action and mask-following ``survivor`` policies together exercise
growth (with the body fill-in lag), boost 2-step + burn cadence + v2 trail
pellets, corpse food drops, head-on size resolution AND body-kill attribution,
and the RNG-driven food economy (maintain/replacement rejection sampling).

The heavy full-battery sweep (>=20 seeds x >=10k frames) is marked ``slow``; a
smaller multi-config sweep runs by default.
"""

from __future__ import annotations

import pytest

from src.simd_env.parity import (
    SurvivorPolicy,
    build_parity_config,
    run_parity,
    scripted_actions,
)

# ---------------------------------------------------------------------------
# Parity configs: small/fast, mid + survivor, tight many-snake (kill-heavy).
# ---------------------------------------------------------------------------
_CFG_SMALL = build_parity_config(
    num_snakes=4, game_width=300, game_height=200, initial_food=20, max_food=25
)
_CFG_MID = build_parity_config(
    num_snakes=6, game_width=400, game_height=300, initial_food=40, max_food=50
)
_CFG_TIGHT = build_parity_config(
    num_snakes=8, game_width=250, game_height=180, initial_food=15, max_food=20
)


@pytest.mark.parametrize(
    "cfg,policy",
    [
        (_CFG_SMALL, "scripted"),
        (_CFG_MID, "survivor"),
        (_CFG_TIGHT, "scripted"),
    ],
)
def test_parity_bit_exact_smoke(cfg, policy):
    """Bit-exact over a short multi-seed sweep for each config/policy."""
    result = run_parity(list(range(5)), num_frames=1000, cfg=cfg, policy=policy)
    assert result.divergence is None, f"Parity divergence ({policy}): {result.divergence}"
    assert result.frames_tested >= 5000


def test_parity_initial_state_matches():
    """The post-reset world (snakes + food) matches before any step.

    This pins the reset RNG draw order: constructor-time food draw (discarded),
    snake placement (get_random_position first, find_empty_position fallback),
    then reset-time food. A single-seed, single-frame run compares the initial
    snapshot inside ``run_parity``.
    """
    result = run_parity([0, 1, 2], num_frames=1, cfg=_CFG_SMALL, policy="scripted")
    assert result.divergence is None, f"Initial-state divergence: {result.divergence}"


def test_scripted_actions_deterministic():
    """The scripted action log is reproducible from the seed alone."""
    a = scripted_actions(7, 100, 4)
    b = scripted_actions(7, 100, 4)
    assert (a == b).all()
    assert a.shape == (100, 4)
    assert a.min() >= 0 and a.max() <= 5


def test_survivor_policy_deterministic():
    """The survivor policy is a pure function of (seed, mask)."""
    import numpy as np

    mask = np.ones((4, 6), dtype=bool)
    p1 = SurvivorPolicy(3, 4)
    p2 = SurvivorPolicy(3, 4)
    a1 = p1.actions(mask)
    a2 = p2.actions(mask)
    assert (a1 == a2).all()
    assert a1.shape == (4,)


def test_parity_exercises_kills_and_growth():
    """The tight config actually produces kills and growth (not just deaths).

    Guards against a vacuous parity pass where every snake dies at frame 1 and
    kill/growth fields are never exercised. Runs the reference alone and asserts
    it produces at least one kill and grows at least one snake past length 1.
    """
    import random

    from src.core.game_config import get_config, initialize_config
    from src.simd_env.parity import PyRefGame, _install_v2_config

    cfg = _CFG_TIGHT
    saved = get_config()
    _install_v2_config(cfg)
    try:
        total_kills = 0
        max_len = 0
        for seed in range(20):
            actions = scripted_actions(seed, 10000, cfg.num_snakes)
            random.seed(seed)
            ref = PyRefGame(cfg)
            for f in range(10000):
                ref.step(actions[f])
                total_kills += sum(len(v) for v in ref.frame_kills.values())
                max_len = max(max_len, max(s.length for s in ref.snakes))
        assert total_kills > 0, "no kills exercised — kill parity is vacuous"
        assert max_len > 1, "no growth exercised — growth parity is vacuous"
    finally:
        initialize_config(saved)


@pytest.mark.slow
@pytest.mark.parametrize(
    "cfg,policy",
    [
        (_CFG_SMALL, "scripted"),
        (_CFG_MID, "survivor"),
        (_CFG_TIGHT, "scripted"),
    ],
)
def test_parity_bit_exact_full_battery(cfg, policy):
    """P2 exit gate: bit-exact over >=20 seeds x >=10k frames per config."""
    result = run_parity(list(range(20)), num_frames=10000, cfg=cfg, policy=policy)
    assert (
        result.divergence is None
    ), f"Full-battery parity divergence ({policy}): {result.divergence}"
    assert result.seeds_tested == 20
    assert result.frames_tested >= 200000
