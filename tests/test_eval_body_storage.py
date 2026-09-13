"""Independent boundary tests for profile-bound SIMD body storage."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.core.game_config import get_config, initialize_config
from src.core.runtime_contract import EffectiveWorldConfig
from src.core.world_runtime import WorldRuntimeSpec
from src.evaluation.protocol import promotion_v2_watch_rect
from src.model.obs_spec import RASTER31V3
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.featurizer import build_observations, obs_inputs_from_batch_sim
from src.simd_env.live_adapter import game_state_to_obs_inputs
from src.simd_env.parity import _install_v2_config

pytestmark = pytest.mark.usefixtures("setup_config")


def _batch_config(
    *,
    source_capacity: int = 3,
    body_storage_capacity: object = None,
) -> BatchSimConfig:
    return BatchSimConfig(
        num_envs=1,
        num_snakes=1,
        game_width=240,
        game_height=180,
        segment_size=10,
        wall_thickness=10,
        initial_food=0,
        max_food=0,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        max_capacity=source_capacity,
        body_storage_capacity=body_storage_capacity,
        frame_rate=1,
    )


def _profile(*, source_capacity: int = 3, horizon: int = 4):
    world = EffectiveWorldConfig(
        width=240,
        height=180,
        segment_size=10,
        wall_thickness=10,
        arena_type="rectangular",
        mechanics_version=2,
        num_snakes=1,
        max_frames=horizon,
        initial_food=0,
        max_food=0,
        min_boost_length=5,
        boost_length_cost_frames=3,
        frame_rate=1,
        max_length=3,
        starvation_max_frames=20,
        max_capacity=source_capacity,
        kill_scale=0.3,
        death_value=-3.0,
        normalization={
            "max_frames": float(horizon),
            "starvation_max": 20.0,
            "max_length": 3.0,
        },
    )
    return replace(
        promotion_v2_watch_rect(world),
        scored_horizon=horizon,
        observation_progress_horizon=horizon,
    )


def _set_batch_body(sim: BatchSim, cells: list[tuple[int, int]]) -> None:
    """Install an ordered head-to-tail body without depending on ring indices."""
    assert cells and len(cells) <= sim.cap
    sim.bodies[0, 0] = 0
    sim.head_ptr[0, 0] = 0
    for offset, cell in enumerate(cells):
        sim.bodies[0, 0, (-offset) % sim.cap] = cell
    sim.seg_count[0, 0] = len(cells)
    sim.length[0, 0] = len(cells)
    sim.direction[0, 0] = 1
    sim.alive[0, 0] = True
    sim.boost_frames[0, 0] = 0
    sim.frames_since_food[0, 0] = 0
    sim._reward_prev_length[0, 0] = len(cells)
    sim._rebuild_traversed_from_heads()
    sim._refresh_action_masks()


def _set_live_body(snake: object, cells: list[tuple[int, int]]) -> None:
    snake.segments = [(x * 10, y * 10) for x, y in cells]
    snake.length = len(cells)
    snake.direction = (1, 0)
    snake.is_alive = True
    snake.is_boosting = False
    snake.boost_frames = 0
    snake.frames_since_food = 0
    snake.respawn_timer = 0
    snake._reward_prev_length = len(cells)
    snake.update = lambda _others, _food, snake=snake, **_kwargs: snake.move()


def _clear_batch_food(sim: BatchSim) -> None:
    sim.food_cells[0] = []
    sim.food_set[0] = set()
    sim.corpse_cells[0] = set()


def test_legacy_positional_config_keeps_historical_field_bindings() -> None:
    """The additive storage field stays behind every historical positional field."""
    cfg = BatchSimConfig(
        1,
        1,
        240,
        180,
        10,
        10,
        0,
        0,
        5,
        3,
        2,
        0.97,
        7,
        "rectangular",
        13,
        0.7,
        -9.0,
    )

    assert cfg.arena_type == "rectangular"
    assert cfg.frame_rate == 13
    assert cfg.kill_scale == pytest.approx(0.7)
    assert cfg.death_value == pytest.approx(-9.0)
    assert cfg.body_storage_capacity is None
    assert BatchSim(cfg, seeds=[7], train_mode=True).cap == cfg.max_capacity == 7


@pytest.mark.parametrize(
    "override",
    [True, False, 0, -1, 2, 3.5, "8"],
)
def test_invalid_or_downgraded_body_storage_capacity_fails_before_reset(
    override: object,
) -> None:
    """An invalid backing ring cannot create even the initial randomized world."""
    cfg = _batch_config(source_capacity=3, body_storage_capacity=override)

    with pytest.raises(ValueError, match="body_storage_capacity"):
        BatchSim(cfg, seeds=[11], train_mode=True)


@pytest.mark.parametrize(
    ("body_storage_capacity", "expected_capacity"),
    [(None, 3), (5, 5)],
)
def test_ring_overflow_guard_uses_resolved_capacity(
    body_storage_capacity: int | None,
    expected_capacity: int,
) -> None:
    """Both source-exact and expanded rings still refuse a live-tail overwrite."""
    sim = BatchSim(
        _batch_config(body_storage_capacity=body_storage_capacity),
        seeds=[13],
        train_mode=True,
    )
    cells = [(8 - offset, 5) for offset in range(expected_capacity)]
    _set_batch_body(sim, cells)

    assert sim.cfg.max_capacity == 3
    assert sim.cap == expected_capacity
    with pytest.raises(
        RuntimeError,
        match=rf"seg_count {expected_capacity} >= body storage capacity {expected_capacity}",
    ):
        sim.step(np.array([[1]], dtype=np.int64))


def test_default_profile_runtime_is_source_exact_and_retains_post_step_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the runtime spec preserves the historical source-capacity failure."""
    from src.simd_env import eval_engine

    profile = _profile(source_capacity=3, horizon=1)
    original_step = eval_engine._TerminalHeroBatchSim.step
    observed_capacities: list[int] = []

    def reach_source_capacity(self, actions, active_env_mask=None):
        original_step(self, actions, active_env_mask)
        observed_capacities.append(self.cap)
        self.length[:, 0] = self.cfg.max_capacity

    monkeypatch.setattr(eval_engine._TerminalHeroBatchSim, "step", reach_source_capacity)

    with pytest.raises(RuntimeError, match=r"capacity 3 at frame 1") as error:
        eval_engine.run_simd_eval(
            ("scripted", "greedy_food"),
            [],
            frames=1,
            seeds=[17],
            profile=profile,
        )

    assert observed_capacities == [3]
    assert "'env_index': 0" in str(error.value)
    assert "'snake_slot': 0" in str(error.value)
    assert "'logical_length': 3" in str(error.value)


def test_explicit_horizon_runtime_crosses_source_guard_and_reports_both_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expanded storage changes allocation while the profile remains source-exact."""
    from src.simd_env import eval_engine

    profile = _profile(source_capacity=3, horizon=4)
    profile_before = (profile.descriptor(), profile.digest, profile.world.digest)
    runtime = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)
    original_step = eval_engine._TerminalHeroBatchSim.step
    forced_lengths = iter((3, 4, 4, 5))
    observed_capacities: list[int] = []

    def cross_source_capacity(self, actions, active_env_mask=None):
        original_step(self, actions, active_env_mask)
        observed_capacities.append(self.cap)
        self.length[:, 0] = next(forced_lengths)

    monkeypatch.setattr(eval_engine._TerminalHeroBatchSim, "step", cross_source_capacity)

    record = eval_engine.run_simd_eval(
        ("scripted", "greedy_food"),
        [],
        frames=4,
        seeds=[19],
        profile=profile,
        world_runtime_spec=runtime,
    )[0]

    assert observed_capacities == [6, 6, 6, 6]
    assert runtime.source_body_capacity == 3
    assert runtime.body_storage_capacity == 6
    assert record["evaluation_profile"] == profile.descriptor()
    assert record["evaluation_profile_digest"] == profile.digest
    assert record["world_runtime_spec"] == runtime.descriptor()
    assert record["world_runtime_spec_digest"] == runtime.digest
    assert (profile.descriptor(), profile.digest, profile.world.digest) == profile_before


def test_runtime_profile_mismatch_fails_before_simulator_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid spec for a different profile cannot become an allocation override."""
    from src.simd_env import eval_engine

    profile = _profile(source_capacity=3, horizon=2)
    runtime = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)
    changed_profile = replace(profile, observation_progress_horizon=3)

    def allocation_forbidden(*_args, **_kwargs):
        raise AssertionError("simulator allocation occurred before runtime validation")

    monkeypatch.setattr(eval_engine, "_TerminalHeroBatchSim", allocation_forbidden)
    with pytest.raises(ValueError, match="profile digest"):
        eval_engine.run_simd_eval(
            ("scripted", "greedy_food"),
            [],
            frames=2,
            seeds=[23],
            profile=changed_profile,
            world_runtime_spec=runtime,
        )


def test_expanded_ring_matches_live_body_growth_boost_order_and_v3_observation() -> None:
    """Crossing the source capacity preserves list-backed live mechanics and inputs."""
    from src.game.game_state import GameState

    cfg = _batch_config(source_capacity=3, body_storage_capacity=8)
    previous_config = get_config()
    _install_v2_config(cfg)
    try:
        game = GameState(headless=True, num_snakes=1)
        sim = BatchSim(cfg, seeds=[29], train_mode=False, allow_respawn=False)
        initial_body = [(6, 5), (5, 5), (4, 5), (3, 5), (2, 5)]
        _set_live_body(game.snakes[0], initial_body)
        _set_batch_body(sim, initial_body)

        game.food_manager.food = [(80, 50)]
        game.food_manager._corpse_positions = set()
        sim.food_cells[0] = [(8, 5)]
        sim.food_set[0] = {(8, 5)}
        sim.corpse_cells[0] = set()
        game.snakes[0].is_boosting = True

        game.update(train_mode=False, learn=False, allow_respawn=False)
        sim.step(np.array([[4]], dtype=np.int64))

        expected_after_boost = [(8, 5), (7, 5), (6, 5), (5, 5), (4, 5)]
        live_body = [(x // 10, y // 10) for x, y in game.snakes[0].segments]
        assert live_body == sim.get_bodies(0, 0) == expected_after_boost
        assert game.snakes[0].length == int(sim.get_lengths()[0, 0]) == 6
        assert bool(sim.get_step_events()["food_ate"][0, 0]) is True
        assert bool(sim.get_step_events()["boosted"][0, 0]) is True

        # The next normal move materializes the growth beyond source capacity.
        game.snakes[0].is_boosting = False
        game.update(train_mode=False, learn=False, allow_respawn=False)
        sim.step(np.array([[1]], dtype=np.int64))

        expected_after_growth = [(9, 5), (8, 5), (7, 5), (6, 5), (5, 5), (4, 5)]
        live_body = [(x // 10, y // 10) for x, y in game.snakes[0].segments]
        assert live_body == sim.get_bodies(0, 0) == expected_after_growth
        assert game.snakes[0].length == int(sim.get_lengths()[0, 0]) == 6
        assert sim.cfg.max_capacity == 3
        assert sim.cap == 8

        # This harness bypasses the live AI reward hook that normally owns the
        # hunger counter.  Pin that unrelated input to a common value so the
        # observation comparison isolates body ordering and normalization.
        game.snakes[0].frames_since_food = 7
        sim.frames_since_food[0, 0] = 7
        normalization = {"max_frames": 4, "starvation_max": 20, "max_length": 3}
        live_inputs = game_state_to_obs_inputs(game, **normalization)
        sim_inputs = obs_inputs_from_batch_sim(sim, **normalization)
        mask = sim.get_resolved_action_mask()
        live_obs = build_observations(live_inputs, mask=mask, obs_spec=RASTER31V3)
        sim_obs = build_observations(sim_inputs, mask=mask, obs_spec=RASTER31V3)

        assert live_obs.keys() == sim_obs.keys()
        for key in live_obs:
            assert np.array_equal(live_obs[key], sim_obs[key]), key
        # The normalized scalar remains deliberately above one; storage did not
        # leak into the source-bound max_length denominator.
        assert float(sim_obs["scalars"][0, 0, 0]) == pytest.approx(2.0)
    finally:
        initialize_config(previous_config)
