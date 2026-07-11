"""Tests for the actor hot-path de-duplication in AISnake.

Covers the carry-forward selection cache (the next_state/next-mask captured in
compute_reward_and_train is reused as the action-selection state at the next
frame) and the shared candidate-move simulation
(``simulate_relative_action_fatality``) behind ``_get_safe_actions``.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from src.core.game_config import GameConfig
from src.game.ai_snake import AISnake, simulate_relative_action_fatality
from src.game.game_logic import GameLogic
from src.game.game_state import GameState
from src.training.apex_policy import ApexPolicy


def _seed_everything(seed: int) -> None:
    """Seed all RNG sources used by the game/policy path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_env(num_snakes: int = 4, seed: int = 1234, board_scale: float = 0.2) -> GameState:
    """Create a seeded headless multi-snake environment (default: actor board)."""
    _seed_everything(seed)
    return GameState(headless=True, num_snakes=num_snakes, board_scale=board_scale)


def _collect_transitions(env: GameState, frames: int) -> List[dict]:
    """Step ``frames`` frames and collect every completed transition.

    Uses the same freshness contract as ApexActor._run_episode
    (last_transition_frame == env.frame).
    """
    records = []
    for _ in range(frames):
        if not any(snake.is_alive for snake in env.snakes):
            break
        env.update(train_mode=True, learn=False)
        frame = env.frame
        for snake in env.snakes:
            if getattr(snake, "last_transition_frame", None) != frame:
                continue
            records.append(
                {
                    "id": snake.id,
                    "frame": frame,
                    "action": snake.last_action,
                    "reward": snake.last_reward,
                    "done": snake.last_done,
                    "state": snake.last_state.clone(),
                    "next_state": (
                        None if snake.last_next_state is None else snake.last_next_state.clone()
                    ),
                    "mask": (
                        None
                        if snake.last_next_action_mask is None
                        else snake.last_next_action_mask.clone()
                    ),
                }
            )
    return records


def _reference_safe_actions(
    snake: AISnake,
    other_snakes: Optional[List] = None,
    allow_fallback: bool = True,
) -> List[int]:
    """Verbatim pre-refactor _get_safe_actions loop (the old duplicate-scan path)."""
    safe = []
    can_boost = snake.length >= GameConfig.MIN_BOOST_LENGTH

    for relative_action in range(3):
        abs_dir = GameLogic.relative_to_absolute_direction(snake.direction, relative_action)
        normal_segments, normal_heads = snake._simulate_move_after_action(abs_dir, steps=1)

        if not snake._segments_collide_after_move(
            normal_segments,
            other_snakes,
            traversed_heads=normal_heads,
        ):
            safe.append(relative_action)

            if can_boost:
                boost_segments, boost_heads = snake._simulate_move_after_action(
                    abs_dir,
                    steps=2,
                    is_boost=True,
                )
                if not snake._segments_collide_after_move(
                    boost_segments,
                    other_snakes,
                    traversed_heads=boost_heads,
                ):
                    safe.append(relative_action + 3)
    if safe or not allow_fallback:
        return safe
    return list(range(3))


class TestCarryForwardEquivalence:
    """Carry-forward cached path must be transition-for-transition identical."""

    def _run(self, carry_forward: bool, frames: int = 50, seed: int = 424242) -> List[dict]:
        # Full-size board: top-of-frame food maintenance is a no-op there (the
        # in-frame refill already restored the cap), so the world is provably
        # unchanged between next-state capture and the next selection and the
        # legacy/cached paths must agree bit for bit. On the saturated 0.2
        # actor board maintain_count partially places food every frame, which
        # is precisely the train-time capture/selection inconsistency the
        # carry-forward removes (see test_stored_next_state_* below).
        env = _make_env(num_snakes=4, seed=seed, board_scale=1.0)
        for snake in env.snakes:
            snake.carry_forward_selection = carry_forward
        return _collect_transitions(env, frames)

    def test_cached_path_matches_legacy_recompute_path(self):
        """50 seeded frames: old recompute path == new cached path, bit for bit."""
        legacy = self._run(carry_forward=False)
        cached = self._run(carry_forward=True)

        assert len(legacy) == len(cached)
        assert len(cached) > 0
        for old, new in zip(legacy, cached):
            assert old["id"] == new["id"]
            assert old["frame"] == new["frame"]
            assert old["action"] == new["action"]
            assert old["reward"] == pytest.approx(new["reward"])
            assert old["done"] == new["done"]
            assert torch.equal(old["state"], new["state"])
            if old["next_state"] is None:
                assert new["next_state"] is None
            else:
                assert torch.equal(old["next_state"], new["next_state"])
            if old["mask"] is None:
                assert new["mask"] is None
            else:
                assert torch.equal(old["mask"], new["mask"])

    def test_stored_next_state_is_the_next_selection_state(self):
        """A transition's stored next_state must BE the state that selected action t+1."""
        env = _make_env(num_snakes=4, seed=777)
        selection_states: Dict[Tuple[int, int], torch.Tensor] = {}

        for snake in env.snakes:
            original_update = snake.update

            def wrapped(other_snakes, food, _orig=original_update, _snake=snake, **kwargs):
                _orig(other_snakes, food, **kwargs)
                state = getattr(_snake, "_pre_collision_state", None)
                if state is not None:
                    selection_states[(_snake.id, env.frame)] = state

            snake.update = wrapped

        stored_next: Dict[Tuple[int, int], torch.Tensor] = {}
        for _ in range(50):
            if not any(snake.is_alive for snake in env.snakes):
                break
            env.update(train_mode=True, learn=False)
            for snake in env.snakes:
                if getattr(snake, "last_transition_frame", None) != env.frame:
                    continue
                if not snake.last_done and snake.last_next_state is not None:
                    stored_next[(snake.id, env.frame)] = snake.last_next_state

        checked = 0
        for (snake_id, frame), next_state in stored_next.items():
            selection_state = selection_states.get((snake_id, frame + 1))
            if selection_state is None:
                continue  # snake never acted again (death/final frame)
            # Identity, not just equality: the cache reuses the same tensor.
            assert selection_state is next_state
            checked += 1
        assert checked > 0

    def test_stored_next_state_matches_direct_get_state_recompute(self):
        """Old-path reconstruction: stored next_state == a direct get_state call.

        Reconstructs what the legacy path computed as next_state by calling
        get_state directly on the post-frame world (restoring the pre-frame
        enemy-trend baseline, which the carry-forward capture advances) and
        asserts the cached capture stored exactly that tensor.
        """
        env = _make_env(num_snakes=4, seed=31337)
        checked = 0
        post_selection_baseline: Dict[int, Tuple[float, Optional[int]]] = {}

        # Snapshot the enemy-trend baseline right after each snake's action
        # selection: that is exactly the baseline the same-frame next-state
        # capture computes its trend feature against (fresh selections advance
        # the baseline themselves; cached selections leave it as the previous
        # capture set it).
        for snake in env.snakes:
            original_update = snake.update

            def wrapped(other_snakes, food, _orig=original_update, _snake=snake, **kwargs):
                _orig(other_snakes, food, **kwargs)
                post_selection_baseline[_snake.id] = (
                    _snake._prev_nearest_enemy_dist,
                    _snake._prev_nearest_enemy_id,
                )

            snake.update = wrapped

        for _ in range(50):
            if not any(snake.is_alive for snake in env.snakes):
                break
            env.update(train_mode=True, learn=False)
            for snake in env.snakes:
                if getattr(snake, "last_transition_frame", None) != env.frame:
                    continue
                if snake.last_done or snake.last_next_state is None:
                    continue
                saved_baseline = (snake._prev_nearest_enemy_dist, snake._prev_nearest_enemy_id)
                snake._prev_nearest_enemy_dist, snake._prev_nearest_enemy_id = (
                    post_selection_baseline[snake.id]
                )
                recomputed = snake.get_state(env.snakes, env.food, update_enemy_memory=False)
                snake._prev_nearest_enemy_dist, snake._prev_nearest_enemy_id = saved_baseline

                assert torch.equal(snake.last_next_state, recomputed)
                checked += 1
        assert checked > 0

    def test_stale_carry_is_rejected_without_frame_progression(self):
        """A carried cache whose frame does not match is ignored (fresh rebuild)."""
        policy = ApexPolicy(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )
        # No frame callback: _get_frame() is constant 0, so the carried frame
        # (0 + 1) never matches and update() must rebuild from scratch.
        snake._pre_collision_state = snake.get_state([snake], [(200, 200)])
        snake._pre_collision_action = 1
        snake.compute_reward_and_train([snake], [(200, 200)], ate_food=False, collided=False)
        assert snake._carried_selection is not None
        assert snake._carried_selection["frame"] == 1

        stale_state = snake._carried_selection["state"]
        snake.update([snake], [(200, 200)])

        assert snake._carried_selection is None
        assert snake._pre_collision_state is not stale_state

    def test_lifecycle_resets_clear_carry(self):
        policy = ApexPolicy(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )
        for reset in (lambda: snake.respawn((50, 50)), lambda: snake.soft_reset((60, 60))):
            snake._carried_selection = {"frame": 1, "state": torch.zeros(1), "safe_actions": [1]}
            reset()
            assert snake._carried_selection is None

    def test_respawn_invalidates_carried_selection_of_other_snakes(self):
        """An opponent respawn (GameState steps 4-5) drops every snake's carry.

        The carried state/mask were captured at step 9 of frame t, in a world
        without the respawned snake; if the cache survived, frame t+1 selection
        would act on a state and safe-action mask blind to the new opponent.
        """
        env = _make_env(num_snakes=4, seed=31337, board_scale=1.0)
        env.update(train_mode=False, learn=False, allow_respawn=True)

        hero = next(s for s in env.snakes if s.is_alive)
        assert hero._carried_selection is not None

        carried_at_selection = []
        original_update = hero.update

        def spy_update(other_snakes, food, **kwargs):
            carried_at_selection.append(hero._carried_selection is not None)
            return original_update(other_snakes, food, **kwargs)

        hero.update = spy_update

        # Control frame: no respawn, so the carry survives to selection.
        env.update(train_mode=False, learn=False, allow_respawn=True)
        assert carried_at_selection == [True]

        # Kill an opponent and force it to respawn at the top of the next
        # frame (steps 4-5), BEFORE action selection (step 6).
        victim = next(s for s in env.snakes if s is not hero and s.is_alive)
        victim.die()
        victim.respawn_timer = 0
        env.update(train_mode=False, learn=False, allow_respawn=True)

        assert victim.is_alive  # the respawn actually happened
        assert carried_at_selection == [True, False]

    def test_carry_disabled_flag_keeps_legacy_behavior(self):
        policy = ApexPolicy(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )
        snake.carry_forward_selection = False
        snake._pre_collision_state = snake.get_state([snake], [(200, 200)])
        snake._pre_collision_action = 1
        snake.compute_reward_and_train([snake], [(200, 200)], ate_food=False, collided=False)

        assert snake._carried_selection is None


class TestSharedActionSimulation:
    """simulate_relative_action_fatality must match the old per-action loops."""

    def test_safe_actions_match_reference_across_seeded_game(self):
        """Every frame of a seeded game: new mask == verbatim old algorithm."""
        env = _make_env(num_snakes=4, seed=99)
        compared = 0
        for _ in range(50):
            if not any(snake.is_alive for snake in env.snakes):
                break
            env.update(train_mode=True, learn=False)
            for snake in env.snakes:
                if not snake.is_alive:
                    continue
                others = list(env.snakes)
                assert snake._get_safe_actions(others, allow_fallback=False) == (
                    _reference_safe_actions(snake, others, allow_fallback=False)
                )
                assert snake._get_safe_actions(others) == (_reference_safe_actions(snake, others))
                compared += 1
        assert compared > 0

    def test_fatality_shape_and_boost_gating(self):
        policy = ApexPolicy(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )
        # Short snake: boost unavailable regardless of geometry.
        snake.length = 1
        normal_fatal, boost_fatal = simulate_relative_action_fatality(snake, [snake])
        assert len(normal_fatal) == 3 and len(boost_fatal) == 3
        assert normal_fatal == [False, False, False]
        assert boost_fatal == [True, True, True]

        # Long snake in open space: boost becomes available and safe.
        snake.length = GameConfig.MIN_BOOST_LENGTH
        normal_fatal, boost_fatal = simulate_relative_action_fatality(snake, [snake])
        assert normal_fatal == [False, False, False]
        assert boost_fatal == [False, False, False]
