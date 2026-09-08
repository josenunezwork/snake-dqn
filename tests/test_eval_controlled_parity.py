"""Independent controlled oracles for the E1 live and SIMD evaluation paths.

The existing runtime smoke test proves that both adapters emit the profile
labels.  These checks drive a fixed action tape through the real update loops,
and use a separately accumulated post-step trace for the headline mass.  The
small hand-built mechanics cases then pin food, boost burn, corpse-food and
frame-rate behavior without reusing the evaluator's metric accumulator.
"""

from __future__ import annotations

import random
import types
from dataclasses import replace

import numpy as np
import pytest

from src.core.game_config import AppConfig, GameSettings, get_config, initialize_config
from src.evaluation.protocol import promotion_v2_watch_rect
from src.game.game_logic import GameLogic
from src.scripts import tournament_eval as te
from src.simd_env import eval_engine as ee
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.parity import PyRefGame, _install_v2_config


def _tiny_config(*, frame_rate: int = 1, num_snakes: int = 3) -> BatchSimConfig:
    """Return the compact rectangular v2 world used by every controlled case."""
    return BatchSimConfig(
        num_envs=1,
        num_snakes=num_snakes,
        game_width=240,
        game_height=180,
        segment_size=10,
        wall_thickness=10,
        initial_food=8,
        max_food=8,
        mechanics_version=2,
        frame_rate=frame_rate,
        max_capacity=32,
    )


def _set_batch_snake(sim: BatchSim, slot: int, cells: list[tuple[int, int]]) -> None:
    """Install an ordered head-to-tail body into a single BatchSim row."""
    assert cells
    for offset, cell in enumerate(cells):
        sim.bodies[0, slot, (-offset) % sim.cap] = cell
    sim.head_ptr[0, slot] = 0
    sim.seg_count[0, slot] = len(cells)
    sim.length[0, slot] = len(cells)
    sim.direction[0, slot] = 1  # right
    sim.alive[0, slot] = True
    sim.respawn_timer[0, slot] = 0
    sim._reward_prev_length[0, slot] = len(cells)


def _set_live_snake(snake: object, cells: list[tuple[int, int]]) -> None:
    """Install the same ordered body into a real live Snake object."""
    snake.segments = [(x * 10, y * 10) for x, y in cells]
    snake.length = len(cells)
    snake.direction = (1, 0)
    snake.is_alive = True
    snake.respawn_timer = 0
    snake._reward_prev_length = len(cells)
    snake.is_boosting = False
    snake.boost_frames = 0
    # Bypass the model policy while retaining the real Snake.move mechanics.
    snake.update = lambda _others, _food, snake=snake, **_kwargs: snake.move()


def _configure_global(cfg: BatchSimConfig) -> object:
    """Apply only the fields required by GameState and return its old config."""
    old = get_config()
    initialize_config(
        AppConfig(
            game=GameSettings(
                width=cfg.game_width,
                height=cfg.game_height,
                num_snakes=cfg.num_snakes,
                segment_size=cfg.segment_size,
                wall_thickness=cfg.wall_thickness,
                initial_food=cfg.initial_food,
                max_food=cfg.max_food,
                mechanics_version=cfg.mechanics_version,
                frame_rate=cfg.frame_rate,
                min_boost_length=cfg.min_boost_length,
                boost_length_cost_frames=cfg.boost_length_cost_frames,
                arena_type=cfg.arena_type,
            )
        )
    )
    return old


def _fixed_tape(seed: int, frames: int, snakes: int) -> np.ndarray:
    """Generate one deterministic action tape, independent of world RNG."""
    gen = np.random.default_rng(seed ^ 0xE100_2026)
    relative = gen.choice(3, size=(frames, snakes), p=[0.25, 0.5, 0.25])
    boost = gen.random((frames, snakes)) < 0.2
    return (relative + boost.astype(np.int64) * 3).astype(np.int64)


def _independent_mass_oracle(
    cfg: BatchSimConfig, seed: int, tape: np.ndarray, *, allow_respawn: bool
) -> dict[str, float]:
    """Accumulate mass directly from a reference world's post-step state."""
    random.seed(seed)
    ref = PyRefGame(cfg, allow_respawn=allow_respawn)
    mass_sum = 0.0
    alive_frames = 0
    deaths = 0
    previous_alive = True
    for frame_actions in tape:
        ref.step(frame_actions)
        hero = ref.snakes[0]
        alive = bool(hero.is_alive)
        if alive:
            alive_frames += 1
            mass_sum += float(hero._logical_length())
        deaths += int(previous_alive and not alive)
        previous_alive = alive
    return {
        "mass_integral": mass_sum / len(tape),
        "survival_fraction": alive_frames / len(tape),
        "deaths": float(deaths),
    }


def test_actual_wrappers_match_independent_poststep_mass_and_profile_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real live/SIMD wrappers agree with a fixed-tape reference mass trace."""
    cfg = _tiny_config(num_snakes=3)
    old = _configure_global(cfg)
    try:
        frames = 3
        seed = 17
        tape = _fixed_tape(seed, frames, cfg.num_snakes)
        world = te._evaluation_world_from_config()
        profile = replace(
            promotion_v2_watch_rect(world),
            scored_horizon=frames,
            observation_progress_horizon=29,
        )

        def attach_fixed(
            game_state: object,
            slot: int,
            spec: tuple[str, str],
            run_seed: int,
            cache: dict[str, object],
            active_profile: object,
        ) -> None:
            del spec, run_seed, cache, active_profile
            snake = game_state.snakes[slot]

            def update(self: object, others: object, food: object, **kwargs: object) -> None:
                del others, food, kwargs
                action = int(tape[int(game_state.frame) - 1, slot])
                self.direction = GameLogic.relative_to_absolute_direction(
                    self.direction, action % 3
                )
                self.is_boosting = bool(action >= 3 and self.length >= 5)
                self.move()
                self.last_action = action

            snake.update = types.MethodType(update, snake)
            snake.policy = None
            snake.ai = None

        class FixedSimdPolicy(ee.SimdPolicy):
            def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
                del masks
                return np.asarray(
                    [tape[int(sim.frame[e]), int(slot)] for e, slot in slots], dtype=np.int64
                )

        monkeypatch.setattr(te, "_attach_agent", attach_fixed)
        monkeypatch.setattr(ee, "build_simd_policy", lambda *args, **kwargs: FixedSimdPolicy())

        live = te.rollout(("scripted", "fixed"), [("scripted", "fixed")] * 2, frames, seed, profile)
        simd = ee.run_simd_eval(
            ("scripted", "fixed"), [("scripted", "fixed")] * 2, frames, [seed], profile=profile
        )[0]
        oracle = _independent_mass_oracle(cfg, seed, tape, allow_respawn=True)

        for record in (live, simd):
            assert record["mass_integral"] == pytest.approx(oracle["mass_integral"])
            assert record["survival_fraction"] == pytest.approx(oracle["survival_fraction"])
            assert record["deaths"] == oracle["deaths"]
            assert record["denominators"] == {
                "scored_frames": frames,
                "decision_frames": frames,
                "alive_frames": frames,
                "food_event_frames": 0,
                "boost_executed_frames": 0,
            }
            assert record["evaluation_profile"]["scored_horizon"] == frames
            assert record["evaluation_profile"]["observation_progress_horizon"] == 29
    finally:
        initialize_config(old)


def test_actual_wrappers_pad_terminal_death_to_the_profile_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal hero may die early while the profile still scores all frames."""
    cfg = replace(_tiny_config(num_snakes=1), initial_food=0, max_food=0)
    old = _configure_global(cfg)
    try:
        frames = 3
        seed = 6  # fixed straight tape reaches the right wall on frame 2
        world = te._evaluation_world_from_config()
        profile = replace(
            promotion_v2_watch_rect(world),
            scored_horizon=frames,
            observation_progress_horizon=29,
        )

        def attach_straight(
            game_state: object,
            slot: int,
            spec: tuple[str, str],
            run_seed: int,
            cache: dict[str, object],
            active_profile: object,
        ) -> None:
            del spec, run_seed, cache, active_profile
            snake = game_state.snakes[slot]

            def update(self: object, others: object, food: object, **kwargs: object) -> None:
                del others, food, kwargs
                self.is_boosting = False
                self.move()

            snake.update = types.MethodType(update, snake)
            snake.policy = None
            snake.ai = None

        class AlwaysStraight(ee.SimdPolicy):
            def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
                del masks, sim
                return np.ones(len(slots), dtype=np.int64)

        monkeypatch.setattr(te, "_attach_agent", attach_straight)
        monkeypatch.setattr(ee, "build_simd_policy", lambda *args, **kwargs: AlwaysStraight())

        live = te.rollout(("scripted", "straight"), [], frames, seed, profile)
        simd = ee.run_simd_eval(("scripted", "straight"), [], frames, [seed], profile=profile)[0]

        for record in (live, simd):
            assert record["mass_integral"] == pytest.approx(2 / 3)
            assert record["survival_fraction"] == pytest.approx(2 / 3)
            assert record["deaths"] == 1
            assert record["denominators"]["scored_frames"] == frames
            assert record["denominators"]["decision_frames"] == frames
            assert record["probes"]["death_cause"] == "wall"
    finally:
        initialize_config(old)


@pytest.mark.parametrize("frame_rate", [1, 3, 100])
def test_actual_game_and_batchsim_match_corpse_food_at_cap(frame_rate: int) -> None:
    """Post-step food and RNG behavior is exact across the three frame rates."""
    from src.game.game_state import GameState

    cfg = replace(_tiny_config(frame_rate=frame_rate, num_snakes=1), max_food=2)
    old = _configure_global(cfg)
    _install_v2_config(cfg)
    try:
        game = GameState(headless=True, num_snakes=1)
        sim = BatchSim(cfg, seeds=[19], train_mode=False)
        _set_live_snake(game.snakes[0], [(2, 2)])
        _set_batch_snake(sim, 0, [(2, 2)])
        # The first pellet is corpse-class; two ambient pellets already meet
        # the cap.  Eating it therefore draws no replacement random position.
        game.food_manager.food = [(30, 20), (50, 50), (60, 50)]
        game.food_manager._corpse_positions = {(30, 20)}
        sim.food_cells[0] = [(3, 2), (5, 5), (6, 5)]
        sim.food_set[0] = set(sim.food_cells[0])
        sim.corpse_cells[0] = {(3, 2)}
        random.seed(917)
        sim._rngs[0]._rng.seed(917)

        game.update(train_mode=False, learn=False, allow_respawn=False)
        sim.step(np.asarray([[1]], dtype=np.int64))

        assert game.frame_ate_food[0] is True
        events = sim.get_step_events()
        assert bool(events["food_ate"][0, 0]) is True
        assert int(game.snakes[0]._logical_length()) == int(sim.get_lengths()[0, 0]) == 2
        assert [(x // 10, y // 10) for x, y in game.food_manager.food] == sim.get_food(0)
        assert random.getstate() == sim._rngs[0]._rng.getstate()
    finally:
        initialize_config(old)


def test_actual_game_and_batchsim_expose_boost_burn_and_death_events() -> None:
    """A controlled v2 tape observes boost execution, burn, and terminal death."""
    from src.game.game_state import GameState

    cfg = replace(_tiny_config(num_snakes=1, frame_rate=3), initial_food=0, max_food=0)
    old = _configure_global(cfg)
    _install_v2_config(cfg)
    try:
        game = GameState(headless=True, num_snakes=1)
        sim = BatchSim(cfg, seeds=[23], train_mode=False, allow_respawn=False)
        body = [(6, 2), (5, 2), (4, 2), (3, 2), (2, 2)]
        _set_live_snake(game.snakes[0], body)
        _set_batch_snake(sim, 0, body)
        game.food_manager.food = []
        game.food_manager._corpse_positions = set()
        sim.food_cells[0] = []
        sim.food_set[0] = set()
        sim.corpse_cells[0] = set()
        game.snakes[0].is_boosting = True
        game.snakes[0].boost_frames = 2
        sim.boost_frames[0, 0] = 2

        game.update(train_mode=False, learn=False, allow_respawn=False)
        sim.step(np.asarray([[4]], dtype=np.int64))

        assert game.snakes[0].is_alive is True
        assert int(game.snakes[0]._logical_length()) == int(sim.get_lengths()[0, 0]) == 4
        assert game.snakes[0].pending_trail_pellets == []
        # Two movement inserts pop the old tail and the cadence burn pops the
        # next tail, so the burned v2 trail pellet is cell (4, 2).
        assert (4, 2) in {(x // 10, y // 10) for x, y in game.food_manager.food}
        assert (4, 2) in set(sim.get_corpse_food(0))
        assert bool(sim.get_boosted_this_step()[0, 0]) is True
        assert bool(sim.get_step_events()["boosted"][0, 0]) is True
        assert game.frame_death_causes == {}

        # A fresh controlled world drives the hero into the right wall.  The
        # independent post-step oracle is total-horizon mass with dead frames 0.
        game = GameState(headless=True, num_snakes=1)
        sim = BatchSim(cfg, seeds=[23], train_mode=False, allow_respawn=False)
        _set_live_snake(game.snakes[0], [(23, 2)])
        _set_batch_snake(sim, 0, [(23, 2)])
        game.food_manager.food = []
        game.food_manager._corpse_positions = set()
        sim.food_cells[0] = []
        sim.food_set[0] = set()
        sim.corpse_cells[0] = set()
        alive_trace = []
        mass_trace = []
        death_cause = 0
        for _ in range(3):
            game.update(train_mode=False, learn=False, allow_respawn=False)
            sim.step(np.asarray([[1]], dtype=np.int64))
            death_cause = death_cause or int(sim.get_step_events()["death_cause"][0, 0])
            alive_trace.append(bool(game.snakes[0].is_alive))
            mass_trace.append(
                float(game.snakes[0]._logical_length()) if game.snakes[0].is_alive else 0.0
            )
            assert bool(game.snakes[0].is_alive) == bool(sim.get_alive()[0, 0])
        assert alive_trace == [False, False, False]
        assert mass_trace == [0.0, 0.0, 0.0]
        assert death_cause != 0
    finally:
        initialize_config(old)


def test_actual_game_and_batchsim_match_body_kill_attribution() -> None:
    """A post-move body hit credits the surviving opponent in both engines."""
    from src.game.game_state import GameState

    cfg = replace(_tiny_config(num_snakes=2), initial_food=0, max_food=0)
    old = _configure_global(cfg)
    _install_v2_config(cfg)
    try:
        game = GameState(headless=True, num_snakes=2)
        sim = BatchSim(cfg, seeds=[4], train_mode=False, allow_respawn=False)
        _set_live_snake(game.snakes[0], [(10, 2)])
        game.snakes[0].direction = (0, 1)
        _set_live_snake(game.snakes[1], [(13, 3), (12, 3), (11, 3), (10, 3), (9, 3)])
        _set_batch_snake(sim, 0, [(10, 2)])
        sim.direction[0, 0] = 2  # down
        _set_batch_snake(sim, 1, [(13, 3), (12, 3), (11, 3), (10, 3), (9, 3)])
        game.food_manager.food = []
        game.food_manager._corpse_positions = set()
        sim.food_cells[0] = []
        sim.food_set[0] = set()
        sim.corpse_cells[0] = set()

        game.update(train_mode=False, learn=False, allow_respawn=False)
        sim.step(np.asarray([[1, 1]], dtype=np.int64))

        assert game.snakes[0].is_alive is False
        assert game.snakes[1].is_alive is True
        assert game.frame_death_causes == {0: "enemy_body"}
        assert game.frame_kills == {1: [0]}
        assert sim.get_alive()[0].tolist() == [False, True]
        assert sim.get_death_cause()[0].tolist() == [4, 0]
        assert sim.get_kill_credit()[0].tolist() == [0, 1]
        assert sim.get_kill_victim_lengths(0, 1) == [1]
        assert [(x // 10, y // 10) for x, y in game.food_manager.food] == sim.get_food(0)
        assert (10, 3) in sim.get_corpse_food(0)
    finally:
        initialize_config(old)
