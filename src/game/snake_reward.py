"""Reward computation for Snake.

The reward shaping / contract logic, factored out of snake.py as a mixin so the
game-entity, observation, and reward concerns live in separate modules. Composed
into Snake; `self` resolves at runtime via the MRO.
"""

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from src.core.game_config import GameConfig, StateIndices
from src.core.reward_events import RewardEvents, compute_reward_v2

if TYPE_CHECKING:  # avoid a runtime import cycle (Snake composes this mixin)
    from src.game.snake import Snake

# Per-term reward accounting keys, in accumulation order. Summing a breakdown's
# values in this (insertion) order reproduces the returned reward exactly,
# because it mirrors the float-accumulation order inside calculate_reward.
# "potential" is the reward-v2 PBRS term (always 0.0 at rewards.version 1); it
# leads the tuple so the v2 accumulation order (potential, death, kill) is a
# subsequence of this schema and the exact-sum property holds at both versions.
REWARD_TERM_KEYS: Tuple[str, ...] = (
    "potential",
    "death",
    "food",
    "food_shaping",
    "wall",
    "danger",
    "starvation",
    "survival",
    "kill",
    "boost",
    "clamp_delta",
)


class SnakeRewardMixin:
    """Computes the reward signal for a Snake (see calculate_reward)."""

    # Per-term breakdown of the most recent calculate_reward call (telemetry).
    last_reward_breakdown: Optional[Dict[str, float]] = None

    def calculate_reward(
        self,
        ate_food: bool,
        collided: bool,
        old_state: Optional[torch.Tensor],
        new_state: Optional[torch.Tensor],
        other_snakes: List["Snake"],
        food: List[Tuple[int, int]],
        frame_kills: Optional[Dict[int, List[int]]] = None,
    ) -> float:
        """
        Calculate reward with clean signals and proper incentives.

        Uses centralized reward constants from GameConfig for easy tuning.

        Fixed issues: wall oscillation exploit, food avoidance being positive,
        weak danger penalties, missing starvation penalty.

        Side effect: stores a per-term accounting dict on
        ``self.last_reward_breakdown`` (keys: REWARD_TERM_KEYS) whose values
        sum to the returned reward exactly. ``clamp_delta`` is the correction
        applied by the final clamp (clamped_total - raw_total).

        Args:
            ate_food: Whether food was eaten this step
            collided: Whether snake died this step
            old_state: Previous state tensor (or None)
            new_state: Current state tensor (or None)
            other_snakes: List of all snakes
            food: List of food positions
            frame_kills: Dict mapping killer_snake_id → [victim_snake_ids] (or None)

        Returns:
            Reward value (clamped to [REWARD_MIN, REWARD_MAX] at rewards.version
            1; UNCLAMPED at rewards.version 2, which delegates to
            :func:`src.core.reward_events.compute_reward_v2`)
        """
        breakdown: Dict[str, float] = {key: 0.0 for key in REWARD_TERM_KEYS}
        self.last_reward_breakdown = breakdown

        # Reward v2 (blueprint §3.2): event-based PBRS reward, delegated to the
        # pure shared module. Version 1 below stays bit-identical to legacy.
        if GameConfig.REWARD_VERSION == 2:
            return self._calculate_reward_v2(
                breakdown, ate_food, collided, other_snakes, frame_kills
            )

        # Terminal: death penalty (optionally scaled by mass, so dying while large
        # costs more and the agent learns to protect accumulated length).
        if collided:
            scale = GameConfig.REWARD_DEATH_LENGTH_SCALE
            if scale:
                max_len = max(int(getattr(GameConfig, "MAX_LENGTH", 150)), 1)
                norm_len = min(self.length / max_len, 2.0)
                death_reward = GameConfig.REWARD_DEATH * (1.0 + scale * norm_len)
                # Floor at REWARD_MIN: the length scaling can drive the penalty
                # below the configured clamp, which would otherwise bypass it.
                clamped_death = max(death_reward, GameConfig.REWARD_MIN)
                breakdown["death"] = death_reward
                breakdown["clamp_delta"] = clamped_death - death_reward
                return clamped_death
            breakdown["death"] = GameConfig.REWARD_DEATH
            return GameConfig.REWARD_DEATH

        reward = 0.0
        if ate_food:
            self.frames_since_food = 0
            breakdown["food"] = GameConfig.REWARD_FOOD_BASE
            reward += breakdown["food"]
        else:
            # Track frames since last food (starvation mechanic)
            self.frames_since_food += 1

            # Shaping rewards (when not eating or dying)
            if new_state is not None:
                # 1) Proportional food-distance shaping: reward scales with distance change
                if old_state is not None:
                    fd_idx = StateIndices.FOOD_DISTANCE
                    prev_food_dist = float(old_state[fd_idx])
                    curr_food_dist = float(new_state[fd_idx])
                    delta = prev_food_dist - curr_food_dist  # positive if closer
                    # Scale by TOWARD_FOOD/typical-step-delta, cap at ±0.1
                    alpha = GameConfig.REWARD_TOWARD_FOOD / 0.01
                    food_shaping = max(min(alpha * delta, 0.1), -0.1)
                    breakdown["food_shaping"] = food_shaping
                    reward += food_shaping

                # 2) Wall proximity - penalty ONLY (no escape reward oscillation exploit)
                breakdown["wall"] = self._calculate_wall_awareness_penalty(new_state)
                reward += breakdown["wall"]
                # 3) Immediate danger penalty (per-action danger, vector layout only).
                breakdown["danger"] = self._calculate_action_danger_penalty(new_state)
                reward += breakdown["danger"]

            # 4) Starvation penalty (encourages food hunting, prevents looping)
            if self.frames_since_food > GameConfig.STARVATION_START_FRAME:
                frames_starving = self.frames_since_food - GameConfig.STARVATION_START_FRAME
                starvation_factor = min(frames_starving / GameConfig.STARVATION_MAX_FRAMES, 1.0)
                starvation_penalty = GameConfig.STARVATION_MAX_PENALTY * starvation_factor
                breakdown["starvation"] = -starvation_penalty
                reward -= starvation_penalty

            # 6) Survival reward: small positive signal for staying alive
            breakdown["survival"] = GameConfig.REWARD_SURVIVAL
            reward += breakdown["survival"]

        # 5) Inter-snake interaction rewards (multi-agent competitive/cooperative behavior)
        breakdown["kill"] = self._calculate_interaction_reward(
            other_snakes, frame_kills=frame_kills
        )
        reward += breakdown["kill"]

        # 7) Boost cost: penalize burning body segments via boost so that eating
        # then immediately boosting the mass away is not reward-free. This fixes
        # the boost-abuse exploit (policies otherwise boost ~89% of the time).
        # Triggers only on the frames a boost action actually burns a segment.
        boost_seg_penalty = GameConfig.REWARD_BOOST_SEGMENT
        if boost_seg_penalty:
            action = getattr(self, "_pre_collision_action", None)
            prev_len = getattr(self, "_reward_prev_length", self.length)
            lost = prev_len - self.length
            if action is not None and int(action) >= 3 and lost > 0 and not ate_food:
                boost_penalty = boost_seg_penalty * float(lost)
                breakdown["boost"] = -boost_penalty
                reward -= boost_penalty
        self._reward_prev_length = self.length

        # Clamp to reasonable range (safety net)
        clamped_reward = max(min(reward, GameConfig.REWARD_MAX), GameConfig.REWARD_MIN)
        breakdown["clamp_delta"] = clamped_reward - reward
        return clamped_reward

    def _calculate_reward_v2(
        self,
        breakdown: Dict[str, float],
        ate_food: bool,
        collided: bool,
        other_snakes: List["Snake"],
        frame_kills: Optional[Dict[int, List[int]]],
    ) -> float:
        """Compute the event-based reward v2 (blueprint §3.2).

        Builds a :class:`~src.core.reward_events.RewardEvents` summary from the
        live-sim step (previous length from ``_reward_prev_length``, death from
        ``collided``, kill credit from ``frame_kills`` — including kills whose
        killer died this same frame) and delegates to the pure shared
        :func:`~src.core.reward_events.compute_reward_v2`. Unlike the v1 path
        there is no early return on death: kill + death + potential are always
        computed together, and nothing is clamped.

        Args:
            breakdown: Pre-zeroed per-term dict (REWARD_TERM_KEYS schema) that
                is also ``self.last_reward_breakdown``; mutated in place.
            ate_food: Whether food was eaten this step (drives the
                ``frames_since_food`` counter only; food value flows through
                the length potential).
            collided: Whether the snake died this step.
            other_snakes: List of all snakes in the game (victim lookup).
            frame_kills: Dict mapping killer_snake_id → [victim_snake_ids]
                (or None).

        Returns:
            Unclamped v2 reward: ``gamma * Phi(s') - Phi(s)`` with
            ``Phi(death) = 0``, plus +0.3 x victim_length per kill, plus -3.0
            on death.
        """
        prev_length = float(getattr(self, "_reward_prev_length", self.length))

        kills: List[float] = []
        if frame_kills is not None:
            victim_ids = frame_kills.get(self.id, [])
            if victim_ids:
                snakes_by_id = {snake.id: snake for snake in other_snakes}
                for victim_id in victim_ids:
                    victim = snakes_by_id.get(victim_id)
                    if victim is not None:
                        kills.append(float(victim._logical_length()))

        total, v2_terms = compute_reward_v2(
            RewardEvents(
                prev_length=prev_length,
                new_length=float(self.length),
                died=bool(collided),
                gamma=float(GameConfig.APEX_GAMMA),
                kills=tuple(kills),
            )
        )
        breakdown["potential"] = v2_terms["potential"]
        breakdown["death"] = v2_terms["death"]
        breakdown["kill"] = v2_terms["kill"]

        # Counter/baseline upkeep mirrors v1: on death both are left as-is
        # (respawn resets them); otherwise advance for the next step.
        if not collided:
            if ate_food:
                self.frames_since_food = 0
            else:
                self.frames_since_food += 1
            self._reward_prev_length = self.length

        return total

    def _get_wall_awareness_distance(self, state: torch.Tensor) -> float:
        """Return the normalized distance signal used for wall-awareness reward."""
        boundary_slice = slice(StateIndices.BOUNDARY_LEFT, StateIndices.BOUNDARY_BOTTOM + 1)
        boundary_distances = state[boundary_slice]
        return float(min(boundary_distances))

    def _calculate_wall_awareness_penalty(self, state: torch.Tensor) -> float:
        """Return the smooth wall-proximity penalty for the current arena type."""
        min_boundary_dist = self._get_wall_awareness_distance(state)
        if min_boundary_dist >= GameConfig.WALL_AWARENESS_THRESHOLD:
            return 0.0

        # Linear interpolation: REWARD_WALL_DANGER at distance 0, 0 at threshold.
        return (
            GameConfig.REWARD_WALL_DANGER
            * (GameConfig.WALL_AWARENESS_THRESHOLD - min_boundary_dist)
            / GameConfig.WALL_AWARENESS_THRESHOLD
        )

    def _calculate_action_danger_penalty(self, state: torch.Tensor) -> float:
        """Return danger penalty from immediate action risk."""
        danger_start = StateIndices.PER_ACTION_DANGER_START
        danger_end = StateIndices.PER_ACTION_DANGER_END
        danger_slice = state[danger_start:danger_end]
        if len(danger_slice) == 0:
            return 0.0

        # Average pressure penalizes constrained states without treating a
        # single blocked direction as a critical state when escape moves exist.
        action_danger = float(danger_slice.float().mean())
        danger_thresholds = [
            (GameConfig.DANGER_CRITICAL_THRESHOLD, GameConfig.REWARD_DANGER_CRITICAL),
            (GameConfig.DANGER_HIGH_THRESHOLD, GameConfig.REWARD_DANGER_HIGH),
            (GameConfig.DANGER_MEDIUM_THRESHOLD, GameConfig.REWARD_DANGER_MEDIUM),
        ]
        for threshold, penalty in danger_thresholds:
            if action_danger > threshold:
                return penalty
        return 0.0

    def _calculate_interaction_reward(
        self,
        other_snakes: List["Snake"],
        frame_kills: Optional[Dict[int, List[int]]] = None,
    ) -> float:
        """
        Calculate inter-snake interaction reward.

        Uses collision-pair tracking (frame_kills) for accurate kill attribution
        when available. Falls back to proximity heuristic for backward compatibility.

        Args:
            other_snakes: List of all snakes in the game
            frame_kills: Dict mapping killer_snake_id → [victim_snake_ids] (or None)

        Returns:
            Interaction reward value
        """
        # No interaction rewards in single-snake mode
        if len(other_snakes) <= 1:
            return 0.0

        interaction_reward = 0.0

        # Use accurate kill tracking when available. An empty dict is an
        # authoritative "no kills this frame" signal from GameState.
        if frame_kills is not None:
            if self.id not in frame_kills:
                return 0.0

            victim_ids = frame_kills[self.id]
            for snake in other_snakes:
                if snake.id in victim_ids:
                    victim_length = snake._logical_length()
                    kill_reward = (
                        GameConfig.REWARD_KILL_BASE
                        + GameConfig.REWARD_KILL_LENGTH_SCALE * victim_length
                    )
                    interaction_reward += min(kill_reward, GameConfig.REWARD_KILL_MAX)
            return interaction_reward

        # Fallback: proximity-based heuristic (backward compatibility)
        for snake in other_snakes:
            if snake == self:
                continue

            # Reward only when another snake just died AND was close enough
            # that we likely caused it (within 2 segment sizes of our body)
            if not snake.is_alive and snake.respawn_timer == GameConfig.FRAME_RATE:
                dead_head = snake.head
                for segment in self.segments:
                    dist = math.sqrt(
                        (dead_head[0] - segment[0]) ** 2 + (dead_head[1] - segment[1]) ** 2
                    )
                    if dist < self.segment_size * 2:
                        interaction_reward += 1.0
                        break

        return interaction_reward
