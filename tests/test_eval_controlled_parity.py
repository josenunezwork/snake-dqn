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

from src.core.game_config import (
    AppConfig,
    GameSettings,
    PQNOverrides,
    get_config,
    initialize_config,
)
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


def _configure_global(cfg: BatchSimConfig, *, max_capacity: int | None = None) -> object:
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
            ),
            pqn=PQNOverrides(max_capacity=max_capacity),
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


def test_actual_wrappers_share_anchor_trace_and_canonical_random_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live and SIMD anchors see the same identity-addressed contexts and actions."""
    from src.evaluation.anchors import ScriptedAnchor

    cfg = _tiny_config(num_snakes=3)
    old = _configure_global(cfg)
    try:
        frames = 3
        seed = 17
        world = te._evaluation_world_from_config()
        profile = replace(
            promotion_v2_watch_rect(world),
            scored_horizon=frames,
            observation_progress_horizon=29,
        )
        original_action = ScriptedAnchor.action
        calls: list[tuple] = []

        def record_action(self: ScriptedAnchor, context: object) -> int:
            action = original_action(self, context)
            calls.append(
                (
                    self.kind,
                    int(context.world_seed),
                    int(context.slot),
                    int(context.frame),
                    tuple(context.head_cell),
                    tuple(context.heading),
                    tuple(context.food_cells),
                    tuple(bool(value) for value in context.allowed_mask),
                    int(action),
                )
            )
            return action

        monkeypatch.setattr(ScriptedAnchor, "action", record_action)
        live = te.rollout(
            ("scripted", "greedy_food"),
            [("scripted", "random_safe")] * 2,
            frames,
            seed,
            profile,
        )
        live_calls = calls.copy()
        calls.clear()
        simd = ee.run_simd_eval(
            ("scripted", "greedy_food"),
            [("scripted", "random_safe")] * 2,
            frames,
            [seed],
            profile=profile,
        )[0]

        assert live_calls == calls
        assert len(live_calls) == frames * cfg.num_snakes
        assert [row[1:4] for row in live_calls] == [
            (seed, slot, frame) for frame in range(frames) for slot in range(cfg.num_snakes)
        ]
        assert live["evaluation_profile_digest"] == simd["evaluation_profile_digest"]
        assert live["mass_integral"] == pytest.approx(simd["mass_integral"])
    finally:
        initialize_config(old)


def test_actual_wrappers_keep_opponent_respawn_in_the_action_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A straight opponent dies, is absent for one step, then acts after respawn."""
    cfg = replace(
        _tiny_config(num_snakes=2), initial_food=0, max_food=0, game_width=120, game_height=100
    )
    old = _configure_global(cfg)
    try:
        frames = 20
        seed = 0
        world = te._evaluation_world_from_config()
        profile = replace(
            promotion_v2_watch_rect(world),
            scored_horizon=frames,
            observation_progress_horizon=29,
        )
        live_respawns: list[int] = []

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
                action = 0 if slot == 0 else 1
                self.direction = GameLogic.relative_to_absolute_direction(self.direction, action)
                self.is_boosting = False
                self.move()

            snake.update = types.MethodType(update, snake)
            snake.policy = None
            snake.ai = None
            if slot == 1:
                original_respawn = snake.respawn

                def respawn(position: tuple[int, int]) -> None:
                    live_respawns.append(int(game_state.frame))
                    original_respawn(position)

                snake.respawn = respawn

        class FixedSimdPolicy(ee.SimdPolicy):
            def __init__(self) -> None:
                self.opponent_calls: list[int] = []

            def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
                del masks
                for env, slot in slots:
                    if int(slot) == 1:
                        self.opponent_calls.append(int(sim.frame[int(env)]))
                return np.asarray([0 if int(slot) == 0 else 1 for _, slot in slots], dtype=np.int64)

        policy = FixedSimdPolicy()
        monkeypatch.setattr(te, "_attach_agent", attach_fixed)
        monkeypatch.setattr(ee, "build_simd_policy", lambda *args, **kwargs: policy)
        live = te.rollout(("scripted", "fixed"), [("scripted", "fixed")], frames, seed, profile)
        simd = ee.run_simd_eval(
            ("scripted", "fixed"), [("scripted", "fixed")], frames, [seed], profile=profile
        )[0]

        # The opponent is absent from action dispatch on frames 11 and 16, then
        # present again on frames 12 and 17 after frame-rate-1 respawns.
        assert live_respawns == [12, 17]
        assert [f for f in range(frames) if f not in policy.opponent_calls] == [11, 16]
        assert live["deaths"] == simd["deaths"] == 0
    finally:
        initialize_config(old)


def test_actual_wrappers_preserve_simultaneous_hero_kill_and_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hero can receive kill credit on the same frame that it dies."""
    from src.game.game_state import GameState

    cfg = replace(_tiny_config(num_snakes=3), initial_food=0, max_food=0)
    old = _configure_global(cfg)
    try:
        frames = 1
        seed = 1
        world = te._evaluation_world_from_config()
        profile = replace(
            promotion_v2_watch_rect(world),
            scored_horizon=frames,
            observation_progress_horizon=29,
        )
        hero_body = [(5, 5), (4, 5)]
        weak_body = [(7, 5)]
        blocker_body = [(6, 8), (6, 7), (6, 6), (6, 5), (6, 4)]

        def controlled_game(**kwargs: object) -> GameState:
            del kwargs
            game = GameState(headless=True, num_snakes=3)
            _set_live_snake(game.snakes[0], hero_body)
            game.snakes[0].direction = (1, 0)
            _set_live_snake(game.snakes[1], weak_body)
            game.snakes[1].direction = (-1, 0)
            _set_live_snake(game.snakes[2], blocker_body)
            game.snakes[2].direction = (0, -1)
            game.food_manager.food = []
            game.food_manager._corpse_positions = set()
            return game

        real_sim_class = ee._TerminalHeroBatchSim

        class ControlledSim(real_sim_class):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                _set_batch_snake(self, 0, hero_body)
                _set_batch_snake(self, 1, weak_body)
                _set_batch_snake(self, 2, blocker_body)
                self.direction[0, 0] = 1
                self.direction[0, 1] = 3
                self.direction[0, 2] = 0
                self.food_cells[0] = []
                self.food_set[0] = set()
                self.corpse_cells[0] = set()
                self._rebuild_traversed_from_heads()
                self._refresh_action_masks()

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

        class FixedSimdPolicy(ee.SimdPolicy):
            def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
                del masks, sim
                return np.ones(len(slots), dtype=np.int64)

        monkeypatch.setattr(te, "create_training_game_state", controlled_game)
        monkeypatch.setattr(te, "_attach_agent", attach_straight)
        monkeypatch.setattr(ee, "_TerminalHeroBatchSim", ControlledSim)
        monkeypatch.setattr(ee, "build_simd_policy", lambda *args, **kwargs: FixedSimdPolicy())

        live = te.rollout(("scripted", "fixed"), [("scripted", "fixed")] * 2, frames, seed, profile)
        simd = ee.run_simd_eval(
            ("scripted", "fixed"), [("scripted", "fixed")] * 2, frames, [seed], profile=profile
        )[0]

        for record in (live, simd):
            assert record["deaths"] == 1
            assert record["kills"] == 1
            assert record["probes"]["death_cause"] in {"body", "enemy_body"}
            assert record["probes"]["food_eaten"] == 0
    finally:
        initialize_config(old)


def test_v3_nondefault_progress_and_scalar_normalizers_are_used() -> None:
    """The v3 producer uses caller supplied progress, hunger, and length caps."""
    from src.model.obs_spec import RASTER31V3
    from src.simd_env.featurizer import build_observations, obs_inputs_from_batch_sim

    cfg = replace(_tiny_config(num_snakes=1), initial_food=0, max_food=0)
    sim = BatchSim(cfg, seeds=[3], train_mode=False, allow_respawn=False)
    _set_batch_snake(sim, 0, [(5, 5), (4, 5), (3, 5)])
    sim.length[0, 0] = 3
    sim.seg_count[0, 0] = 3
    sim.boost_frames[0, 0] = 2
    sim.frames_since_food[0, 0] = 10
    sim.frame[0] = 7

    inputs = obs_inputs_from_batch_sim(
        sim,
        max_frames=29,
        starvation_max=77,
        max_length=66,
    )
    observation = build_observations(inputs, obs_spec=RASTER31V3)
    scalars = observation["scalars"][0, 0]

    assert scalars[0] == pytest.approx(3 / 66)
    assert scalars[1] == pytest.approx(np.log1p(3) / np.log1p(66))
    assert scalars[2] == 0.0  # length below the boost threshold
    assert scalars[3] == pytest.approx(2 / 3)
    assert scalars[4] == pytest.approx(10 / 77)
    assert scalars[5] == pytest.approx(7 / 29)
    assert scalars[6] == 1.0
    assert observation["tactical_uint8"].dtype == np.uint8


def test_actual_wrappers_reject_equality_at_declared_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reaching max_capacity on the final transition is a hard evaluation error."""
    from src.game.game_state import GameState

    cfg = replace(_tiny_config(num_snakes=1), initial_food=1, max_food=1)
    old = _configure_global(cfg, max_capacity=2)
    try:
        frames = 1
        seed = 9
        world = te._evaluation_world_from_config()
        profile = replace(
            promotion_v2_watch_rect(world),
            scored_horizon=frames,
            observation_progress_horizon=29,
        )

        def controlled_game(**kwargs: object) -> GameState:
            del kwargs
            game = GameState(headless=True, num_snakes=1)
            _set_live_snake(game.snakes[0], [(2, 2)])
            game.food_manager.food = [(30, 20)]
            game.food_manager._corpse_positions = set()
            return game

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

        real_sim_class = ee._TerminalHeroBatchSim

        class ControlledSim(real_sim_class):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                _set_batch_snake(self, 0, [(2, 2)])
                self.food_cells[0] = [(3, 2)]
                self.food_set[0] = {(3, 2)}
                self.corpse_cells[0] = set()
                self._rebuild_traversed_from_heads()
                self._refresh_action_masks()

        class FixedSimdPolicy(ee.SimdPolicy):
            def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
                del masks, sim
                return np.ones(len(slots), dtype=np.int64)

        monkeypatch.setattr(te, "create_training_game_state", controlled_game)
        monkeypatch.setattr(te, "_attach_agent", attach_straight)
        monkeypatch.setattr(ee, "_TerminalHeroBatchSim", ControlledSim)
        monkeypatch.setattr(ee, "build_simd_policy", lambda *args, **kwargs: FixedSimdPolicy())

        message = "evaluation world exceeded its declared max_capacity"
        with pytest.raises(RuntimeError, match=message):
            te.rollout(("scripted", "fixed"), [], frames, seed, profile)
        with pytest.raises(RuntimeError, match=message):
            ee.run_simd_eval(("scripted", "fixed"), [], frames, [seed], profile=profile)
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
