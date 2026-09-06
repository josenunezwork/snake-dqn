"""Golden-replay parity tests: live Python game vs. the batched sim (P2 gate).

Asserts BIT-EXACT dynamics parity between the reference live Python game
(``src.game``, mechanics v2 + reward v2) and the vectorized ``BatchSim`` over a
seed sweep, comparing every load-bearing world quantity per frame: head/body
positions, alive/deaths, kill attribution, the ordered food set, per-snake
rewards, and per-agent action masks.

Parity milestone achieved (see ``src/simd_env/parity.py`` and the module
docstring): rectangular arena, mechanics v2 + reward v2, on BOTH respawn arms —
``allow_respawn=False`` (terminal deaths) and the live gate's
``train_mode=True, allow_respawn=True`` pair, where dead snakes serve
``frame_rate`` frames and respawn from the shared RNG stream.
The scripted-action and mask-following ``survivor`` policies together exercise
growth (with the body fill-in lag), boost 2-step + burn cadence + v2 trail
pellets, corpse food drops, head-on size resolution AND body-kill attribution,
and the RNG-driven food economy (maintain/replacement rejection sampling).

The heavy full-battery sweep (>=20 seeds x >=10k frames) is marked ``slow``; a
smaller multi-config sweep runs by default.
"""

from __future__ import annotations

from dataclasses import replace

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


@pytest.mark.parametrize("frame_rate", [1, 3])
def test_parity_bit_exact_with_respawn(frame_rate):
    """Bit-exact on the RESPAWN arm -- the branch ``run_simd_eval`` ships.

    The default arm keeps deaths terminal, so ``_respawn_dead`` never runs and
    the whole respawn surface (its RNG draws, its phase within the frame, and the
    timer cadence) reached the promotion gate uncovered. This runs both sims with
    the live gate's ``train_mode=True, allow_respawn=True`` pair on the tight
    kill-heavy config, so snakes die, serve ``frame_rate`` frames and respawn
    from the shared Mersenne-Twister stream.
    """
    result = run_parity(
        list(range(5)),
        num_frames=1000,
        cfg=replace(_CFG_TIGHT, frame_rate=frame_rate),
        policy="scripted",
        allow_respawn=True,
    )
    assert result.divergence is None, f"Respawn parity divergence: {result.divergence}"
    assert result.frames_tested >= 5000


def test_respawn_arm_actually_respawns():
    """Guard against a vacuous respawn parity pass.

    If no snake ever respawned, ``test_parity_bit_exact_with_respawn`` would be
    green while covering exactly nothing -- which is how the branch reached the
    gate untested in the first place.
    """
    import random

    from src.core.game_config import get_config, initialize_config
    from src.simd_env.parity import PyRefGame, _install_v2_config

    cfg = replace(_CFG_TIGHT, frame_rate=1)
    saved = get_config()
    _install_v2_config(cfg)
    try:
        respawns = 0
        for seed in range(3):
            actions = scripted_actions(seed, 400, cfg.num_snakes)
            random.seed(seed)
            ref = PyRefGame(cfg, allow_respawn=True)
            for f in range(400):
                before = [s.is_alive for s in ref.snakes]
                ref.step(actions[f])
                respawns += sum(1 for b, s in zip(before, ref.snakes) if not b and s.is_alive)
        assert respawns > 0, "no respawns exercised — respawn parity is vacuous"
    finally:
        initialize_config(saved)


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


def test_batch_mask_matches_live_mask_implementation():
    """BatchSim's mask must equal the LIVE mask the serve path actually acts on.

    Calls ``simulate_relative_action_fatality`` -- the exact function
    ``AISnake._get_safe_actions`` builds its mask from -- directly on the
    reference's real ``Snake`` objects, deliberately bypassing
    ``PyRefGame.action_masks``. This is the mask field's independent reference:
    while the harness compared BatchSim against a re-implementation that shared
    BatchSim's own convention, the mask was the one parity field checked only
    against a copy of itself, and a train/serve divergence sat there unnoticed.
    """
    import random
    from dataclasses import replace

    from src.core.game_config import get_config, initialize_config
    from src.game.ai_snake import simulate_relative_action_fatality
    from src.simd_env.batch_sim import BatchSim
    from src.simd_env.parity import PyRefGame, _install_v2_config

    cfg = _CFG_TIGHT
    num_frames = 300
    saved = get_config()
    _install_v2_config(cfg)
    try:
        for seed in range(5):
            actions = scripted_actions(seed, num_frames, cfg.num_snakes)
            random.seed(seed)
            ref = PyRefGame(cfg)
            bat = BatchSim(replace(cfg, num_envs=1), seeds=[seed], train_mode=True)
            for f in range(num_frames):
                ref.step(actions[f])
                bat.step(actions[f].reshape(1, cfg.num_snakes))
                bat_mask = bat.get_action_mask()[0]
                for sidx, snake in enumerate(ref.snakes):
                    if not snake.is_alive:
                        assert not bat_mask[sidx].any()
                        continue
                    normal_fatal, boost_fatal = simulate_relative_action_fatality(snake, ref.snakes)
                    live = [not x for x in normal_fatal] + [not x for x in boost_fatal]
                    assert list(bat_mask[sidx]) == live, (
                        f"batch mask != live mask at seed={seed} frame={f} snake={sidx}: "
                        f"batch={list(bat_mask[sidx])} live={live}"
                    )
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
