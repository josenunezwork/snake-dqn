"""Exact-event episode metrics for profiled :func:`run_simd_eval` audits.

The evaluator's frame observer deliberately exposes detached pre/post facts.  This
adapter turns those facts into descriptive Q0 measurements without re-inferring
food, boost, or death events from a length delta.  It is usable by any profiled
SIMD evaluation and does not alter the evaluator or training path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np


def length_stratum(length: float) -> str:
    """Return the preregistered coarse length band for a pre-action state."""
    if length < 700:
        return "short_lt700"
    if length >= 800:
        return "long_ge800"
    return "middle_700_799"


def time_stratum(frame: int, horizon: int) -> str:
    """Split a fixed horizon into deterministic early/middle/late thirds."""
    if frame < horizon // 3:
        return "early"
    if frame < (2 * horizon) // 3:
        return "middle"
    return "late"


def space_stratum(resolved_mask: np.ndarray) -> str:
    """Classify the supplied legal-action support, never a geometric proxy."""
    legal = int(np.asarray(resolved_mask, dtype=bool).sum())
    if legal == 0:
        return "no_legal_action"
    if legal <= 2:
        return "constrained_1_2_legal_actions"
    return "open_3plus_legal_actions"


class FrameObserver:
    """Accumulate exact post-event metrics independently for every eval seed.

    Call an instance directly as ``run_simd_eval(..., frame_observer=observer)``
    then read :meth:`result`.  Only ``transition_valid`` rows contribute action,
    event, reward, and strata measurements; terminal-hero padding is counted
    separately.  Mass uses the evaluator's post-step convention: a death frame
    contributes zero mass even when it carries food, kill, or reward events.
    """

    def __init__(self, *, horizon: int = 5000) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self._seeds: tuple[int, ...] | None = None
        self._next_frame = 0
        self._rows: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _array(section: Mapping[str, object], name: str, size: int) -> np.ndarray:
        if name not in section:
            raise ValueError(f"frame observer snapshot is missing {name!r}")
        value = np.asarray(section[name])
        if value.shape[0] != size:
            raise ValueError(f"snapshot {name!r} has {value.shape[0]} rows, expected {size}")
        return value

    @staticmethod
    def _new_row() -> dict[str, Any]:
        return {
            "scored_frames": 0,
            "terminal_padding_frames": 0,
            "decision_frames": 0,
            "alive_frames": 0,
            "mass_sum": 0.0,
            "food_event_frames": 0,
            "food_mass_collected": 0,
            "boost_action_decisions": 0,
            "boost_eligible_decisions": 0,
            "boost_action_when_eligible": 0,
            "boost_executed_frames": 0,
            "reward_total": 0.0,
            "reward_on_surviving_frames": 0.0,
            "reward_on_death_frames": 0.0,
            "death_events": 0,
            "death_causes": Counter(),
            "kills": 0,
            "by_length": defaultdict(FrameObserver._new_bucket),
            "by_time": defaultdict(FrameObserver._new_bucket),
            "by_space": defaultdict(FrameObserver._new_bucket),
        }

    @staticmethod
    def _new_bucket() -> dict[str, Any]:
        return {
            "decision_frames": 0,
            "alive_frames": 0,
            "mass_sum": 0.0,
            "food_event_frames": 0,
            "food_mass_collected": 0,
            "boost_eligible_decisions": 0,
            "boost_executed_frames": 0,
            "reward_total": 0.0,
            "death_events": 0,
        }

    @staticmethod
    def _observe_bucket(bucket: dict[str, Any], values: Mapping[str, Any]) -> None:
        for name in bucket:
            bucket[name] += values[name]

    def __call__(self, snapshot: Mapping[str, object]) -> None:
        """Consume one detached evaluator snapshot; malformed snapshots fail closed."""
        try:
            frame = int(snapshot["frame"])
            seeds = tuple(int(value) for value in snapshot["seeds"])
            pre = snapshot["pre"]
            post = snapshot["post"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid frame observer snapshot") from exc
        if not isinstance(pre, Mapping) or not isinstance(post, Mapping) or not seeds:
            raise ValueError("frame observer snapshot requires non-empty pre/post/seeds")
        if frame != self._next_frame:
            raise ValueError(
                f"non-contiguous frame sequence: expected {self._next_frame}, got {frame}"
            )
        if frame >= self.horizon:
            raise ValueError(f"frame {frame} exceeds observer horizon {self.horizon}")
        if self._seeds is None:
            self._seeds = seeds
            self._rows = {seed: self._new_row() for seed in seeds}
        elif seeds != self._seeds:
            raise ValueError("seed order changed within one frame observer")

        size = len(seeds)
        pre_lengths = self._array(pre, "logical_mass", size)
        masks = self._array(pre, "resolved_masks", size)
        actions = self._array(pre, "actions", size)
        post_alive = self._array(post, "alive", size)
        post_lengths = self._array(post, "logical_mass", size)
        valid = self._array(post, "transition_valid", size)
        food = self._array(post, "food_ate", size)
        boosted = self._array(post, "boosted", size)
        done = self._array(post, "done", size)
        causes = self._array(post, "death_cause", size)
        rewards = self._array(post, "reward", size)
        kills = self._array(post, "kills", size)
        if masks.shape != (size, 6):
            raise ValueError(f"resolved_masks must have shape ({size}, 6), got {masks.shape}")

        for index, seed in enumerate(seeds):
            row = self._rows[seed]
            row["scored_frames"] += 1
            if not bool(valid[index]):
                row["terminal_padding_frames"] += 1
                continue
            action = int(actions[index])
            if action < 0 or action >= 6:
                raise ValueError(f"invalid hero action {action}")
            pre_length = float(pre_lengths[index])
            alive = bool(post_alive[index])
            death = bool(done[index])
            reward = float(rewards[index])
            food_event = int(bool(food[index]))
            boost_eligible = int(np.asarray(masks[index, 3:], dtype=bool).any())
            values = {
                "decision_frames": 1,
                "alive_frames": int(alive),
                "mass_sum": float(post_lengths[index]) if alive else 0.0,
                "food_event_frames": food_event,
                "food_mass_collected": food_event,
                "boost_eligible_decisions": boost_eligible,
                "boost_executed_frames": int(bool(boosted[index])),
                "reward_total": reward,
                "death_events": int(death),
            }
            for name in (
                "decision_frames",
                "alive_frames",
                "mass_sum",
                "food_event_frames",
                "food_mass_collected",
                "boost_eligible_decisions",
                "boost_executed_frames",
                "reward_total",
                "death_events",
            ):
                row[name] += values[name]
            row["boost_action_decisions"] += int(action >= 3)
            row["boost_action_when_eligible"] += int(action >= 3 and boost_eligible)
            row["kills"] += int(kills[index])
            if death:
                row["reward_on_death_frames"] += reward
                row["death_causes"][str(int(causes[index]))] += 1
            else:
                row["reward_on_surviving_frames"] += reward
            for bucket in (
                row["by_length"][length_stratum(pre_length)],
                row["by_time"][time_stratum(frame, self.horizon)],
                row["by_space"][space_stratum(masks[index])],
            ):
                self._observe_bucket(bucket, values)
        self._next_frame += 1

    @staticmethod
    def _render_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
        decisions = int(bucket["decision_frames"])
        return {
            **{name: bucket[name] for name in bucket},
            "food_mass_per_decision": (
                bucket["food_mass_collected"] / decisions if decisions else None
            ),
            "boost_executed_fraction": (
                bucket["boost_executed_frames"] / decisions if decisions else None
            ),
        }

    def result(self) -> dict[str, Any]:
        """Return a deterministic per-seed descriptive receipt suitable for Q0."""
        rows: dict[str, Any] = {}
        for seed in sorted(self._rows):
            row = self._rows[seed]
            decisions = row["decision_frames"]
            rows[str(seed)] = {
                name: value
                for name, value in row.items()
                if name not in {"death_causes", "by_length", "by_time", "by_space"}
            }
            rows[str(seed)].update(
                {
                    "mass_integral": row["mass_sum"] / self.horizon,
                    "food_mass_per_alive_decision": (
                        row["food_mass_collected"] / row["alive_frames"]
                        if row["alive_frames"]
                        else None
                    ),
                    "food_mass_per_whole_horizon": row["food_mass_collected"] / self.horizon,
                    "boost_executed_fraction": (
                        row["boost_executed_frames"] / decisions if decisions else None
                    ),
                    "boost_eligible_fraction": (
                        row["boost_eligible_decisions"] / decisions if decisions else None
                    ),
                    "death_causes": dict(sorted(row["death_causes"].items())),
                    "by_length": {
                        key: self._render_bucket(row["by_length"][key])
                        for key in sorted(row["by_length"])
                    },
                    "by_time": {
                        key: self._render_bucket(row["by_time"][key])
                        for key in sorted(row["by_time"])
                    },
                    "by_space": {
                        key: self._render_bucket(row["by_space"][key])
                        for key in sorted(row["by_space"])
                    },
                }
            )
        return {
            "schema": "q0-frame-episode-metrics/v1",
            "horizon": self.horizon,
            "frames_observed": self._next_frame,
            "complete_horizon": self._next_frame == self.horizon,
            "episodes": rows,
        }
