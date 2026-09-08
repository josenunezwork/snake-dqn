"""Independent acceptance checks across the repaired Apex boundaries.

These tests intentionally use the real actor, game, replay, learner, and public
loader paths.  Small deterministic fixtures keep the checks cheap while still
crossing the boundary where the adjacent unit tests stop.
"""

from __future__ import annotations

import copy
import queue
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import GameConfig, get_config, initialize_config
from src.game.game_state import GameState
from src.main import load_checkpoint_into_game_state
from src.model.apex_network import ApexNetwork
from src.scripts.apex_train import attach_runtime_metadata, build_apex_runtime_snapshot
from src.training.apex_actor import ApexActor
from src.training.apex_buffer import BufferProcess, LocalApexBuffer
from src.training.apex_learner import ApexLearner, ApexLearnerConfig
from src.training.apex_policy import ApexPolicy
from src.training.apex_recipe import ApexRecipe
from src.training.td_targets import (
    MASK_MODE_LEGACY_ADVISORY,
    MASK_MODE_RASTER_RESOLVED_V3,
    double_dqn_next_q,
)


@pytest.fixture(autouse=True)
def _cpu_and_restore_config():
    """Keep torch and the compatibility config isolated between acceptance checks."""
    original_config = get_config()
    DeviceManager.override_device(torch.device("cpu"))
    yield
    initialize_config(original_config)
    DeviceManager.reset_for_testing()


class _RasterContextPolicy(ApexPolicy):
    """A real Apex network with the serving context surface AISnake consumes."""

    def __init__(self) -> None:
        super().__init__(58, 8, 6, training=False, device=torch.device("cpu"))
        self.obs_spec = "raster31v3"

    def action_context_for(self, _snake_id: int) -> SimpleNamespace:
        mask = np.array([True, True, True, False, False, False], dtype=bool)
        return SimpleNamespace(q_values=torch.zeros(6), resolved=mask, legal=mask, advisory=mask)


class _RecordingBufferClient:
    """Record actor IPC payloads without replacing actor collection logic."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add_batch(self, **payload: Any) -> None:
        self.rows.append(
            {
                key: (
                    [np.array(value, copy=True) for value in value_list]
                    if isinstance(value_list, list)
                    else value_list
                )
                for key, value_list in payload.items()
            }
        )

    def flush(self) -> None:
        """Match ActorBufferClient's final flush surface."""

    def get_stats(self) -> dict[str, int]:
        return {}


def _actor(
    client: Any,
    shared_network: ApexNetwork,
    *,
    base_seed: int = 17,
    max_episodes: int | None = None,
    n_step: int = 1,
    batch_send_size: int = 1,
) -> ApexActor:
    return ApexActor(
        actor_id=0,
        num_actors=1,
        shared_network=shared_network,
        buffer_client=client,
        weight_queue=queue.Queue(),
        stats_queue=queue.Queue(),
        stop_event=torch.multiprocessing.Event(),
        n_step=n_step,
        batch_send_size=batch_send_size,
        env_num_snakes=1,
        env_board_scale=0.2,
        env_food_multiplier=0.5,
        base_seed=base_seed,
        max_episodes=max_episodes,
    )


def test_real_aisnake_masks_cross_actor_collection_and_buffer_ipc() -> None:
    """Successor and terminal mask semantics survive AISnake -> actor -> IPC replay."""
    buffer_process = BufferProcess(
        capacity=8,
        state_size=58,
        max_queue_size=8,
        base_seed=3,
    )
    original_config = get_config()
    try:
        initialize_config(
            replace(original_config, game=replace(original_config.game, max_frames=2))
        )
        policy = _RasterContextPolicy()
        game = GameState(
            headless=True,
            num_snakes=1,
            shared_policy=policy,
            board_scale=0.2,
            food_multiplier=0.5,
        )
        snake = game.snakes[0]
        game.handle_collisions = lambda: {snake.id} if game.frame == 2 else {}

        buffer_process.start()
        actor = _actor(
            buffer_process.get_actor_client(0),
            ApexNetwork(58, 8, 6),
            base_seed=19,
        )
        actor.env = game
        actor._run_episode(stream_to_buffer=True)

        learner_client = buffer_process.get_learner_client()
        sampled = learner_client.sample(2, device=torch.device("cpu"), timeout=5.0)
        assert sampled is not None
        batch, _handles, _weights = sampled
        assert set(batch["next_action_mask_modes"].tolist()) == {
            MASK_MODE_RASTER_RESOLVED_V3,
            2,
        }
        terminal = batch["dones"].bool()
        assert int(terminal.sum()) == 1
        assert batch["next_action_mask_present"][terminal].item() is False
        assert batch["next_action_mask_present"][~terminal].item() is True
        assert batch["next_action_mask_modes"][terminal].item() == 2
        assert batch["next_action_mask_modes"][~terminal].item() == MASK_MODE_RASTER_RESOLVED_V3
    finally:
        buffer_process.shutdown()
        initialize_config(original_config)


def _make_mask_learner(mode: int, mask: list[bool]) -> ApexLearner:
    """Create one tiny learner whose only replay row carries the requested mode."""
    torch.manual_seed(123)
    replay = LocalApexBuffer(capacity=4, state_size=58)
    state = torch.zeros(58)
    next_state = torch.zeros(58)
    next_state[57] = 1.0
    replay.add(
        state,
        0,
        0.0,
        next_state,
        False,
        priority=1.0,
        bootstrap_steps=2,
        next_action_mask=mask,
        next_action_mask_mode=mode,
    )
    return ApexLearner(
        ApexLearnerConfig(
            input_size=58,
            hidden_size=8,
            output_size=6,
            batch_size=1,
            min_buffer_size=1,
            gamma=0.5,
            n_step=3,
            learning_rate=0.001,
            use_compile=False,
        ),
        buffer_client=replay,
        device=torch.device("cpu"),
    )


def test_learner_train_step_uses_mode_actual_n_and_changes_priority() -> None:
    """Learner train_step consumes row modes and the actual replay horizon."""
    legacy = _make_mask_learner(MASK_MODE_LEGACY_ADVISORY, [False] * 6)
    resolved = _make_mask_learner(
        MASK_MODE_RASTER_RESOLVED_V3, [True, False, False, False, False, False]
    )

    captured: dict[str, Any] = {}
    original_compute = legacy.compute_td_targets

    def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
        captured.update(kwargs)
        return original_compute(*args, **kwargs)

    legacy.compute_td_targets = capture  # type: ignore[method-assign]
    legacy_metrics = legacy.train_step()
    resolved_metrics = resolved.train_step()

    assert captured["bootstrap_steps"].tolist() == [2.0]
    assert captured["next_action_mask_modes"].tolist() == [MASK_MODE_LEGACY_ADVISORY]
    assert legacy_metrics["mean_td_error"] != pytest.approx(resolved_metrics["mean_td_error"])

    next_states = torch.zeros((1, 58))
    next_states[0, 57] = 1.0
    with torch.no_grad():
        online = legacy._online_forward(next_states)
        target = legacy.target_dqn(next_states)
        next_q = double_dqn_next_q(online, target, next_states, None)
        actual_n = legacy.compute_td_targets(
            torch.zeros(1),
            next_states,
            torch.zeros(1),
            bootstrap_steps=torch.tensor([2.0]),
        )
    assert actual_n.item() == pytest.approx((0.5**2) * next_q.item())
    assert actual_n.item() != pytest.approx((0.5**3) * next_q.item())


def test_public_main_loader_accepts_metadata_free_model_weights_as_weights_only(tmp_path) -> None:
    """The public headless loader accepts a legacy model_state_dict safely."""
    source = ApexPolicy(58, 8, 6, device=torch.device("cpu"))
    checkpoint_path = tmp_path / "legacy-model-only.pth"
    torch.save({"model_state_dict": copy.deepcopy(source.dqn.state_dict())}, checkpoint_path)

    receiver = ApexPolicy(58, 8, 6, device=torch.device("cpu"))
    game = GameState(headless=True, num_snakes=1, shared_policy=receiver)
    assert load_checkpoint_into_game_state(
        game,
        str(checkpoint_path),
        strict_training_contract=False,
        resume_mode="weights-only",
    )
    assert receiver.update_counter == 0
    assert receiver.total_reward == 0.0
    assert receiver.target_dqn is not None
    for key, value in receiver.dqn.state_dict().items():
        assert torch.equal(value, source.dqn.state_dict()[key])
        assert torch.equal(value, receiver.target_dqn.state_dict()[key])


def _trained_learner(learning_rate: float = 0.001) -> tuple[ApexLearner, dict[str, Any]]:
    replay = LocalApexBuffer(capacity=4, state_size=8)
    replay.add(torch.zeros(8), 0, 1.0, torch.ones(8), False, priority=1.0, bootstrap_steps=2)
    learner = ApexLearner(
        ApexLearnerConfig(
            input_size=8,
            hidden_size=8,
            output_size=6,
            batch_size=1,
            min_buffer_size=1,
            learning_rate=learning_rate,
            use_compile=False,
        ),
        buffer_client=replay,
        device=torch.device("cpu"),
    )
    learner.train_step()
    active_replay = learner.buffer_client._buffer
    learner.config.apex_recipe = ApexRecipe.distributed(
        optimizer=learner.optimizer,
        input_size=8,
        output_size=6,
        gamma=learner.config.gamma,
        n_step=learner.config.n_step,
        priority_alpha=0.6,
        priority_eps=1e-6,
        world=asdict(get_config().game),
        runtime={"mode": "integration", "training": True},
        actor_scaling={},
        seed_identity={"effective_seed": 9},
        batch_size=learner.config.batch_size,
        replay_capacity=active_replay.capacity,
        min_replay_size=learner.config.min_buffer_size,
        beta_frames=active_replay.beta_frames,
        initial_beta_clock=learner.step_count,
    )
    return learner, learner.get_state_dict()


def _requested_recipe(learner: ApexLearner) -> ApexRecipe:
    active_replay = learner.buffer_client._buffer
    return ApexRecipe.distributed(
        optimizer=learner.optimizer,
        input_size=8,
        output_size=6,
        gamma=learner.config.gamma,
        n_step=learner.config.n_step,
        priority_alpha=0.6,
        priority_eps=1e-6,
        world=asdict(get_config().game),
        runtime={"mode": "integration", "training": True},
        actor_scaling={},
        seed_identity={"effective_seed": 9},
        batch_size=learner.config.batch_size,
        replay_capacity=active_replay.capacity,
        min_replay_size=learner.config.min_buffer_size,
        beta_frames=active_replay.beta_frames,
        initial_beta_clock=1,
    )


def _fresh_continuation_receiver(learning_rate: float) -> ApexLearner:
    """Build a receiver with the same resolved replay geometry as the source."""
    replay = LocalApexBuffer(capacity=4, state_size=8)
    return ApexLearner(
        ApexLearnerConfig(
            input_size=8,
            hidden_size=8,
            output_size=6,
            batch_size=1,
            min_buffer_size=1,
            learning_rate=learning_rate,
            use_compile=False,
        ),
        buffer_client=replay,
        device=torch.device("cpu"),
    )


def test_learner_continuation_rejects_lr_conflict_before_mutating_receiver() -> None:
    """A changed optimizer LR fails the public continuation load preflight."""
    source, checkpoint = _trained_learner(0.001)
    receiver = _fresh_continuation_receiver(0.002)
    before = {key: value.clone() for key, value in receiver.dqn.state_dict().items()}
    with pytest.raises(ValueError, match="conflicts"):
        receiver.load_state_dict(
            checkpoint, resume_mode="continuation", requested_recipe=_requested_recipe(receiver)
        )
    assert receiver.step_count == 0
    for key, value in receiver.dqn.state_dict().items():
        assert torch.equal(value, before[key])
    assert source.step_count == 1


def test_learner_continuation_rejects_forged_optimizer_before_mutating_receiver() -> None:
    """Forged Adam options are rejected before receiver weights or counters change."""
    _source, checkpoint = _trained_learner(0.001)
    forged = copy.deepcopy(checkpoint)
    forged["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.5
    receiver = _fresh_continuation_receiver(0.001)
    before = {key: value.clone() for key, value in receiver.dqn.state_dict().items()}
    with pytest.raises(ValueError, match="state conflicts"):
        receiver.load_state_dict(
            forged, resume_mode="continuation", requested_recipe=_requested_recipe(receiver)
        )
    assert receiver.step_count == 0
    for key, value in receiver.dqn.state_dict().items():
        assert torch.equal(value, before[key])


def _seeded_actor_trace(seed: int, shared_network: ApexNetwork) -> list[dict[str, Any]]:
    client = _RecordingBufferClient()
    actor = _actor(client, shared_network, base_seed=seed, max_episodes=1)
    actor.run()
    return client.rows


def test_explicit_seed_repeats_short_actor_trace_and_changes_with_seed() -> None:
    """The actor seeds before constructing its local network and GameState."""
    original_config = get_config()
    try:
        initialize_config(
            replace(
                original_config,
                game=replace(original_config.game, max_frames=4, num_snakes=1),
                network=replace(original_config.network, hidden_size=16),
            )
        )
        torch.manual_seed(99)
        shared = ApexNetwork(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
        first = _seeded_actor_trace(101, shared)
        repeated = _seeded_actor_trace(101, shared)
        different = _seeded_actor_trace(102, shared)
        assert len(first) == len(repeated) == len(different) == 4
        for first_batch, repeated_batch in zip(first, repeated):
            assert first_batch["actions"] == repeated_batch["actions"]
            for first_state, repeated_state in zip(first_batch["states"], repeated_batch["states"]):
                assert np.array_equal(first_state, repeated_state)
        assert any(
            first_batch["actions"] != different_batch["actions"]
            or any(
                not np.array_equal(first_state, different_state)
                for first_state, different_state in zip(
                    first_batch["states"], different_batch["states"]
                )
            )
            for first_batch, different_batch in zip(first, different)
        )
    finally:
        initialize_config(original_config)


class _VersionQueue:
    def __init__(self, version: int, weights: dict[str, torch.Tensor]) -> None:
        self._items = [(version, weights)]

    def get_nowait(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


def test_runtime_receipt_reconciles_fresh_stale_and_continuation_actor_versions() -> None:
    """Per-actor ages remain visible when versions are mixed at continuation."""
    network = ApexNetwork(8, 4, 6)
    weights = network.state_dict()
    actors = []
    for version in (12, 3, 0):
        actor = ApexActor(
            actor_id=len(actors),
            num_actors=3,
            shared_network=network,
            buffer_client=_RecordingBufferClient(),
            weight_queue=_VersionQueue(version, weights),
            stats_queue=queue.Queue(),
            stop_event=torch.multiprocessing.Event(),
        )
        actor.local_network = ApexNetwork(8, 4, 6)
        actor.target_network = ApexNetwork(8, 4, 6)
        assert actor._check_weight_queue()
        actors.append(actor)

    snapshot = build_apex_runtime_snapshot(
        learner_updates=12,
        actor_stats=[{"policy_version": actor.policy_version} for actor in actors],
        buffer_replay_health={"total_sampled": 3},
        elapsed_seconds=1.0,
        actor_heartbeat_ages={0: 0.1, 1: 1.0, 2: 2.0},
    )
    state: dict[str, Any] = {}
    attach_runtime_metadata(
        state,
        snapshot,
        "budget_exhausted",
        actor_policy_versions={
            0: actors[0].policy_version,
            1: actors[1].policy_version,
            2: actors[2].policy_version,
        },
    )
    runtime = state["apex_runtime"]
    assert runtime["actor_policy_version_age"] == {"0": 0, "1": 9, "2": 12}
    assert runtime["actor_policy_version_age_max"] == 12
    assert runtime["policy_version"] == 12
