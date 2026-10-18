"""Exact-event episode metrics for profiled :func:`run_simd_eval` audits.

The evaluator's frame observer deliberately exposes detached pre/post facts.  This
adapter turns those facts into descriptive Q0 measurements without re-inferring
food, boost, or death events from a length delta.  It is usable by any profiled
SIMD evaluation and does not alter the evaluator or training path.
"""

from __future__ import annotations

import time
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

    def __init__(
        self, *, horizon: int = 5000, policy_probe=None, deadline: float | None = None
    ) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self.policy_probe = policy_probe
        self.deadline = deadline
        self._seeds: tuple[int, ...] | None = None
        self._next_frame = 0
        self._rows: dict[int, dict[str, Any]] = {}
        # Compatibility surface for T23's public ``evaluate_world`` observer
        # contract. These are derived from the same exact post-step events.
        self.alive_frames: dict[int, int] = {}
        self.death_frame: dict[int, int | None] = {}

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
            "cumulative_mass": {},
            "length_histogram": Counter(),
            "policy_diagnostics": Counter(),
            "reward_decomposition": Counter(),
            "counterfactual_training_return": {"0.99": 0.0, "0.998": 0.0},
            "scored_frames": 0,
            "terminal_padding_frames": 0,
            "decision_frames": 0,
            "alive_frames": 0,
            "mass_sum": 0.0,
            "food_event_frames": 0,
            "food_mass_collected": 0,
            "boost_length_burned": 0,
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
        for name in values:
            if name in bucket:
                bucket[name] += values[name]

    def __call__(self, snapshot: Mapping[str, object]) -> None:
        """Consume one detached evaluator snapshot; malformed snapshots fail closed."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Q0 CPU runtime cap exhausted")
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
            self.alive_frames = {seed: 0 for seed in seeds}
            self.death_frame = {seed: None for seed in seeds}
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
                if frame + 1 in (500, 1000, 2500, 5000):
                    row["cumulative_mass"][str(frame + 1)] = row["mass_sum"]
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
            row["length_histogram"][str(int(pre_length))] += 1
            diagnostic = self.policy_probe.pending.pop(index, {}) if self.policy_probe else {}
            scalar_diagnostic = {
                key: value for key, value in diagnostic.items() if not key.startswith("reward_")
            }
            row["policy_diagnostics"].update(scalar_diagnostic)
            if diagnostic:
                burn = (
                    bool(boosted[index])
                    and "boost_frame_counter" in pre
                    and int(pre["boost_frame_counter"][index]) + 1
                    >= diagnostic.get("reward_boost_cost_frames", float("inf"))
                )
                row["boost_length_burned"] += int(burn)
                gamma = diagnostic["reward_gamma"]
                potential = gamma * values["mass_sum"] / 10.0 - pre_length / 10.0
                death_term = diagnostic["reward_death_value"] * int(death)
                row["reward_decomposition"].update(
                    {
                        "potential": potential,
                        "death": death_term,
                        "kill_residual": reward - potential - death_term,
                        "mass_term_kappa1": (1.0 - gamma) * values["mass_sum"] / 10.0,
                    }
                )
                # Frozen gamma counterfactuals on identical complete trajectories;
                # these are objective-accounting diagnostics, not actual TD returns.
                for fixed_gamma in (0.99, 0.998):
                    training_reward = values["mass_sum"] / 10.0 - pre_length / 10.0 + death_term
                    row["counterfactual_training_return"][str(fixed_gamma)] += (
                        fixed_gamma**frame * training_reward
                    )
            self.alive_frames[seed] += int(alive)
            if death:
                if self.death_frame[seed] is not None:
                    raise ValueError("terminal hero died twice")
                self.death_frame[seed] = frame + 1
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
            if frame + 1 in (500, 1000, 2500, 5000):
                row["cumulative_mass"][str(frame + 1)] = row["mass_sum"]
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
                bucket.setdefault("policy_diagnostics", Counter()).update(scalar_diagnostic)
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
                        row["food_mass_collected"] / row["decision_frames"]
                        if row["decision_frames"]
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


class PolicyProbe:
    """Read-only same-state B8 comparison around the actual evaluator selector.

    This scoped hook is single-process only. It calls the original selector and
    never changes its returned actions, simulator, optimizer or replay state.
    """

    def __init__(self, b8: str, *, reservoir=None, mix: str = "unknown") -> None:
        self.b8 = b8
        self.reservoir = reservoir
        self.mix = mix
        self.pending: dict[int, dict[str, float]] = {}

    def __enter__(self):
        from unittest.mock import patch

        from src.model.ego2s_network import load_ego2s_checkpoint
        from src.simd_env import ego2s_policy

        self.control = load_ego2s_checkpoint(self.b8, "cpu").eval().requires_grad_(False)
        if self.reservoir is not None:
            from types import SimpleNamespace

            from research.redesign_m3c_20261008 import train_m3c

            self.shadow = (
                train_m3c.load_init(SimpleNamespace(init=self.b8, obs_version="bv8"), "cpu")
                .eval()
                .requires_grad_(False)
            )
        original = ego2s_policy.Ego2sSimdPolicy.actions
        probe = self

        def observed(policy, masks, sim, slots):
            executed = original(policy, masks, sim, slots)
            rows = np.asarray(slots).reshape(-1, 2)
            q = ego2s_policy.student_q(policy.net, sim, rows, policy.forward).numpy()
            bq = ego2s_policy.student_q(probe.control, sim, rows, policy.forward).numpy()
            legal = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            safe = legal.copy()
            safe[~safe.any(1), :3] = True
            greedy = np.where(safe, q.astype(np.float64), -np.inf).argmax(1)
            base = np.where(safe, bq.astype(np.float64), -np.inf).argmax(1)
            if probe.reservoir is not None:
                import torch

                from research.redesign_m3c_20261008 import train_m3c

                shadow_q = ego2s_policy.student_q(probe.shadow, sim, rows, policy.forward)
                trainer_masks = train_m3c.hero_selection_mask(sim, rows)
                trainer_actions = (
                    torch.where(
                        torch.from_numpy(trainer_masks), shadow_q, torch.full_like(shadow_q, -1e9)
                    )
                    .argmax(1)
                    .numpy()
                )
                for i, (env, slot) in enumerate(rows):
                    if slot != 0:
                        continue
                    length = int(sim.get_lengths()[env, slot])
                    probe.reservoir.add(
                        {
                            "mix": probe.mix,
                            "short_lt700": length < 700,
                            "long_ge800": length >= 800,
                            "crowded": int(sim.get_alive()[env].sum()) >= 5,
                            "open": int(sim.get_alive()[env].sum()) <= 3,
                            "exact_q_tie": bool((bq[i, safe[i]] == bq[i, safe[i]].max()).sum() > 1),
                            "boost_eligible": bool(legal[i, 3:].any()),
                            "action_a": int(executed[i]),
                            "action_b": int(trainer_actions[i]),
                            "fallback_used": not bool(legal[i].any()),
                            "mask_disagreement": not np.array_equal(legal[i], trainer_masks[i]),
                            "wrapper_disagreement": not np.array_equal(bq[i], shadow_q[i].numpy()),
                        }
                    )
            for i, (env, slot) in enumerate(rows):
                if slot != 0:
                    continue
                support = q[i, safe[i]]
                ordered = np.sort(support)
                centered = q[i] - q[i].mean()
                bcentered = bq[i] - bq[i].mean()
                probe.pending[int(env)] = {
                    "probe_decisions": 1,
                    "greedy_boost_decisions": int(greedy[i] >= 3),
                    "greedy_boost_when_eligible": int(greedy[i] >= 3 and legal[i, 3:].any()),
                    "policy_boost_eligible_decisions": int(legal[i, 3:].any()),
                    "action_disagreements_b8": int(executed[i] != base[i]),
                    "q_abs_max_sum": float(np.abs(q[i]).max()),
                    "q_legal_gap_sum": (
                        float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
                    ),
                    "q_centered_drift_l1_sum": float(np.abs(centered - bcentered).mean()),
                    "reward_boost_cost_frames": int(sim.cfg.boost_length_cost_frames),
                    "reward_gamma": float(sim.cfg.gamma),
                    "reward_death_value": float(sim.cfg.death_value),
                    "reward_kill_scale": float(sim.cfg.kill_scale),
                }
            if any(p.grad is not None or p.requires_grad for p in probe.control.parameters()):
                raise RuntimeError("B8 inference control accumulated gradients")
            return executed

        self.patch = patch.object(ego2s_policy.Ego2sSimdPolicy, "actions", observed)
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()
