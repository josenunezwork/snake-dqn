"""Independent raster31v3 serving identity and pre-move observation oracles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")

from src.core import game_config  # noqa: E402
from src.core.config_loader import load_and_initialize_config  # noqa: E402
from src.core.device_manager import DeviceManager  # noqa: E402
from src.core.game_config import GameConfig  # noqa: E402
from src.core.runtime_contract import ActionMaskSet  # noqa: E402
from src.game.ai_snake import AISnake  # noqa: E402
from src.game.game_state import GameState  # noqa: E402
from src.model.inference_agent import InferenceAgent  # noqa: E402
from src.model.obs_spec import RASTER31V3  # noqa: E402
from src.simd_env.featurizer import build_observations  # noqa: E402
from src.simd_env.live_adapter import game_state_to_obs_inputs  # noqa: E402
from web.backend.raster_policy import RasterServingPolicy  # noqa: E402

NORMALIZATION = {"max_frames": 5000, "starvation_max": 500, "max_length": 150}
CORE_OBSERVATION_KEYS = ("tactical_uint8", "strategic_uint8", "scalars", "mask")


@pytest.fixture(autouse=True)
def _restore_config_and_device():
    """Keep the real GameState fixture isolated from the process singletons."""
    previous = game_config._current_config
    load_and_initialize_config("configs/mechanics_v2.yaml")
    DeviceManager.override_device(torch.device("cpu"))
    yield
    DeviceManager.reset_for_testing()
    game_config._current_config = previous


class _WorldXNetwork(torch.nn.Module):
    """Deterministic raster network whose action is keyed by world-x scalar."""

    output_size = 6

    def forward(self, observations):
        scalars = observations["scalars"]
        rows = int(scalars.shape[0])
        # The v3 scalar at index 23 is world x / (grid width - 1). The explicit
        # 1450x830 mechanics-v2 config has 145 cells, so this recovers the exact
        # source column for the independent action oracle.
        columns = torch.round(scalars[:, 23] * 144.0).to(dtype=torch.long)
        actions = torch.remainder(columns, 3)
        q_values = torch.full((rows, 6), -100.0, dtype=torch.float32, device=scalars.device)
        return q_values.scatter(1, actions.view(-1, 1), 100.0)


class _ForeignPolicy:
    """A separate policy whose action must not consume the raster policy's rows."""

    training = False
    epsilon = 0.0
    memory = None
    device = torch.device("cpu")

    def __init__(self):
        self.prepare_calls = 0

    def prepare_frame(self):
        self.prepare_calls += 1

    @staticmethod
    def dqn(state):
        return torch.tensor(
            [[100.0, -100.0, -100.0, -100.0, -100.0, -100.0]],
            dtype=torch.float32,
        )


def _independent_observation(game: GameState) -> dict[str, np.ndarray]:
    """Build expected v3 bytes without reading RasterServingPolicy's cache."""
    snakes = list(game.snakes)
    inp = game_state_to_obs_inputs(game, **NORMALIZATION)
    count = len(snakes)
    legal = np.zeros((1, count, 6), dtype=bool)
    advisory = np.zeros_like(legal)
    dead = np.ones((1, count), dtype=bool)
    for row, snake in enumerate(snakes):
        if not bool(snake.is_alive):
            continue
        dead[0, row] = False
        legal[0, row, :3] = True
        if int(snake.length) >= GameConfig.MIN_BOOST_LENGTH:
            legal[0, row, 3:] = True
        # The fixture's every-AI safety oracle is deliberately all-actions so
        # each normal/boost bit is legal and action identity is the sole issue.
        advisory[0, row] = legal[0, row]
    resolved = ActionMaskSet(legal=legal, advisory=advisory, dead=dead).resolved()
    return build_observations(inp, mask=resolved, obs_spec=RASTER31V3)


def _row(observation: dict[str, np.ndarray], snake_id: int, game: GameState):
    row = next(index for index, snake in enumerate(game.snakes) if snake.id == snake_id)
    return {key: np.asarray(observation[key])[0, row] for key in CORE_OBSERVATION_KEYS}


def _assert_observation_equal(actual, expected) -> None:
    for key in CORE_OBSERVATION_KEYS:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def _make_game() -> tuple[GameState, RasterServingPolicy, list[AISnake], _ForeignPolicy]:
    """Create one real mixed GameState with sparse IDs and deterministic actions."""
    agent = InferenceAgent(_WorldXNetwork(), device=torch.device("cpu"), obs_spec=RASTER31V3)
    policy = RasterServingPolicy(agent, normalization=NORMALIZATION)
    game = GameState(headless=True, human_mode=True, num_snakes=4, shared_policy=policy)
    policy.attach_game(game)

    # Remove ambient-food randomness and place a collision-free, identity-rich
    # roster. Every snake has length 6, so all six action bits are legal.
    game._effective_initial_food = 0
    game._effective_max_food = 0
    game.food_manager.max_food = 0
    game.food_manager.food.clear()
    game.food_manager._corpse_positions.clear()
    sparse_ids = [10, 40, 70, 90]
    for index, snake in enumerate(game.snakes):
        snake.id = sparse_ids[index]
        x, y = 100 + index * 250, 150 + index * 150
        snake.segments = [(x - 10 * part, y) for part in range(6)]
        snake.length = len(snake.segments)
        snake.direction = (1, 0)
        snake.is_alive = True
        snake.respawn_timer = 0
        snake.last_move_positions = []
        snake.last_action = None
        snake.frames_since_food = 0
        snake.boost_frames = 0
        snake.is_boosting = False
        if isinstance(snake, AISnake):
            # This makes the independent mask construction above exact while
            # still exercising the actual GameState/AISnake update path.
            snake._get_safe_actions = lambda *args, **kwargs: list(range(6))
    game.snake_id_counter = 100
    game.snakes[0].auto_respawn = False
    foreign = _ForeignPolicy()
    game.snakes[3].policy = foreign
    own_ais = [snake for snake in game.snakes[1:] if snake.policy is policy]
    assert len(own_ais) == 2
    original_prepare = policy.prepare_frame
    policy.expected_actions = {}

    def capture_independent_action_oracle():
        # Compute expected actions directly from each ID's live head BEFORE
        # movement, independently of every policy row/context/cache lookup.
        policy.expected_actions = {
            snake.id: (int(snake.segments[0][0]) // GameConfig.SEGMENT_SIZE) % 3
            for snake in game.snakes
            if snake.is_alive and getattr(snake, "policy", None) is policy
        }
        original_prepare()

    policy.prepare_frame = capture_independent_action_oracle
    return game, policy, own_ais, foreign


def _selected_action(context) -> int:
    q = context.q_values.detach().cpu().numpy().copy()
    q[~np.asarray(context.resolved, dtype=bool)] = -np.inf
    return int(np.argmax(q))


def test_real_update_cache_is_pre_human_move_and_matches_independent_bytes():
    """The cached raster is the actual pre-human-move GameState observation."""
    game, policy, own_ais, _ = _make_game()
    human = game.snakes[0]
    pre_head = tuple(human.segments[0])
    captured = {}
    original_prepare = policy.prepare_frame

    def capture_pre_move():
        captured["observation"] = _independent_observation(game)
        captured["ids"] = [snake.id for snake in game.snakes]
        original_prepare()

    policy.prepare_frame = capture_pre_move
    game.update(train_mode=False, learn=False, allow_respawn=True)

    assert tuple(human.segments[0]) != pre_head
    assert captured["ids"] == [snake.id for snake in game.snakes]
    for snake in own_ais:
        _assert_observation_equal(
            policy.hero_observation(snake.id), _row(captured["observation"], snake.id, game)
        )

    post_move = _independent_observation(game)
    pre_human = _row(captured["observation"], human.id, game)
    post_human = _row(post_move, human.id, game)
    assert any(
        not np.array_equal(pre_human[key], post_human[key])
        for key in ("tactical_uint8", "strategic_uint8", "scalars")
    ), "human movement must change at least one rendered pre/post observation plane"


def test_noop_prepare_frame_caches_post_human_bytes_and_fails_pre_move_oracle():
    """A no-op preparation hook demonstrably shifts the cache after human motion."""
    game, policy, own_ais, _ = _make_game()
    captured = {}

    def no_op_but_capture_pre_move():
        captured["observation"] = _independent_observation(game)

    policy.prepare_frame = no_op_but_capture_pre_move
    game.update(train_mode=False, learn=False, allow_respawn=True)

    post_move = _independent_observation(game)
    cached = policy.hero_observation(own_ais[0].id)
    pre = _row(captured["observation"], own_ais[0].id, game)
    post = _row(post_move, own_ais[0].id, game)
    assert any(
        not np.array_equal(pre[key], post[key])
        for key in ("tactical_uint8", "strategic_uint8", "scalars")
    ), "the real update must change at least one observation plane after movement"
    assert any(
        not np.array_equal(cached[key], pre[key])
        for key in ("tactical_uint8", "strategic_uint8", "scalars")
    ), "a no-op prepare hook must not satisfy the pre-human-move oracle"


def test_identity_context_survives_inspector_order_roster_reorder_and_foreign_policy():
    """Repeated out-of-order inspectors and reorder preserve ID-specific actions."""
    game, policy, own_ais, foreign = _make_game()
    game.update(train_mode=False, learn=False, allow_respawn=True)

    first_contexts = {}
    for snake in (own_ais[1], own_ais[0], own_ais[1], own_ais[0]):
        context = policy.action_context_for(snake.id)
        first_contexts.setdefault(snake.id, context)
        _assert_observation_equal(context.observation, policy.hero_observation(snake.id))
        _assert_observation_equal(context.observation, first_contexts[snake.id].observation)
    assert set(first_contexts) == {snake.id for snake in own_ais}

    # Move one owned AI and the foreign AI across roster positions. The same
    # sparse IDs must still address the same pre-move observation rows.
    human = game.snakes[0]
    foreign_snake = game.snakes[3]
    game.snakes = [human, own_ais[1], foreign_snake, own_ais[0]]
    game.update(train_mode=False, learn=False, allow_respawn=True)

    assert foreign.prepare_calls == 2
    for snake in own_ais:
        context = policy.action_context_for(snake.id)
        assert snake.last_action == policy.expected_actions[snake.id]
        assert _selected_action(context) == policy.expected_actions[snake.id]
        _assert_observation_equal(context.observation, policy.hero_observation(snake.id))
    assert foreign_snake.last_action == 0


def test_dead_owned_ai_does_not_act_then_respawns_with_same_sparse_id():
    """A dead owned row is skipped, then its respawned object keeps its ID."""
    game, policy, own_ais, _ = _make_game()
    dead_ai = own_ais[0]
    original_id = dead_ai.id
    dead_ai.is_alive = False
    dead_ai.respawn_timer = 2
    game.update(train_mode=False, learn=False, allow_respawn=True)
    assert not dead_ai.is_alive
    assert dead_ai.id == original_id
    assert dead_ai.last_action is None
    assert policy.hero_observation(original_id) is not None

    dead_ai.respawn_timer = 0
    game.update(train_mode=False, learn=False, allow_respawn=True)
    assert dead_ai.is_alive
    assert dead_ai.id == original_id
    context = policy.action_context_for(original_id)
    assert dead_ai.last_action == policy.expected_actions[original_id]
    assert _selected_action(context) == policy.expected_actions[original_id]
    _assert_observation_equal(context.observation, policy.hero_observation(original_id))


def test_game_state_prepares_each_distinct_policy_once_with_actual_rows():
    """A real update prepares the owned and foreign policy exactly once each."""
    game, policy, own_ais, foreign = _make_game()
    calls = {"owned": 0}
    original_prepare = policy.prepare_frame

    def count_owned_prepare():
        calls["owned"] += 1
        original_prepare()

    policy.prepare_frame = count_owned_prepare
    game.update(train_mode=False, learn=False, allow_respawn=True)
    assert calls["owned"] == 1
    assert foreign.prepare_calls == 1
    for snake in own_ais:
        context = policy.action_context_for(snake.id)
        assert snake.last_action == policy.expected_actions[snake.id]
        assert _selected_action(context) == policy.expected_actions[snake.id]


def test_v3_epsilon_skip_still_populates_id_context_without_row_shift():
    """V3 context remains identity-addressed when action values are not logged."""
    game, policy, own_ais, _ = _make_game()
    for snake in own_ais:
        snake.actor_epsilon = 1.0
        snake.record_q_values = False
        # Sentinel values make this fail if the v3 identity context is skipped
        # on the epsilon/no-record path and stale constructor masks survive.
        snake.current_action_mask = np.zeros(6, dtype=bool)
        snake.current_legal_action_mask = np.zeros(6, dtype=bool)
        snake.current_advisory_action_mask = np.zeros(6, dtype=bool)

    game.update(train_mode=False, learn=False, allow_respawn=True)

    # The selection may explore, but it must still use this snake's resolved
    # context and never leave current masks unset or consume another row.
    for snake in own_ais:
        context = policy.action_context_for(snake.id)
        assert np.array_equal(snake.current_action_mask, context.resolved)
        assert np.array_equal(snake.current_legal_action_mask, context.legal)
        assert np.array_equal(snake.current_advisory_action_mask, context.advisory)
        assert snake.last_action in np.flatnonzero(context.resolved)
