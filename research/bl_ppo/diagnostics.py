"""Bounded, serialisable E8 training diagnostics.

The E2 collector emits world-major transition batches.  This module deliberately
keeps only fixed-size counters and one active-life record per world, so a status
snapshot is safe to publish every million transitions and to carry through a
recovery checkpoint.  It never retains a trajectory or treats a rollout cut as
the end of a life.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

HORIZON = 5000
MIXES = ("scripted", "frozen", "mixed")
LENGTH_BINS = ("lt400", "400_799", "800_1199", "ge1200")
FRAME_BINS = ("lt2000", "2000_3499", "ge3500")
DECILE_LABELS = tuple(f"d{index}" for index in range(10))
SYNTHETIC_DECILE_BOUNDARIES = tuple(range(100, 1000, 100))


def _zero_occupancy() -> dict[str, dict[str, int]]:
    return {length: {frame: 0 for frame in FRAME_BINS} for length in LENGTH_BINS}


def _zero_deciles() -> dict[str, dict[str, int]]:
    return {decile: {"decisions": 0, "departures": 0} for decile in DECILE_LABELS}


def _zero_run_summary() -> dict[str, Any]:
    return {
        "runs_ge2": 0,
        "runs_ge3": 0,
        "lives_with_run_ge2": 0,
        "lives_with_run_ge3": 0,
        "mass_ledger_ge2_sum": 0.0,
        "mass_ledger_ge3_sum": 0.0,
        "terminated_lives": 0,
        "surviving_lives": 0,
        "open_lives": 0,
        "open_runs_ge2": 0,
        "open_runs_ge3": 0,
    }


def _zero_endpoint() -> dict[str, int]:
    return {
        "pairs": 0,
        "both_alive": 0,
        "candidate_only": 0,
        "b8_only": 0,
        "both_dead": 0,
        "rescues": 0,
        "re_deaths": 0,
    }


def _length_bin(length: int) -> str:
    if length < 400:
        return "lt400"
    if length < 800:
        return "400_799"
    if length < 1200:
        return "800_1199"
    return "ge1200"


def _frame_bin(frame: int) -> str:
    if frame < 2000:
        return "lt2000"
    if frame < 3500:
        return "2000_3499"
    return "ge3500"


def _finite_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


class E8Diagnostics:
    """Streaming E8 diagnostics for one arm.

    ``observe`` accepts one collector ``RolloutBatch`` plus actions in the
    batch's world-major order.  Calling ``snapshot`` is side-effect-free, so
    periodic reports cannot double count transitions or completed runs.
    """

    schema = "bl-ppo-e8-diagnostics/v1"

    def __init__(
        self,
        *,
        horizon: int = HORIZON,
        length_decile_boundaries: Sequence[int] | None = None,
        require_fixed_bank: bool = False,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if require_fixed_bank and length_decile_boundaries is not None:
            raise ValueError("scientific E8 diagnostics must configure the frozen bank explicitly")
        self.horizon = int(horizon)
        self.require_fixed_bank = bool(require_fixed_bank)
        self._decile_boundaries: tuple[int, ...] | None = None
        self._decile_source: str | None = None
        if length_decile_boundaries is not None:
            self._set_length_deciles(length_decile_boundaries, source="explicit_synthetic")
        elif not self.require_fixed_bank:
            self._set_length_deciles(SYNTHETIC_DECILE_BOUNDARIES, source="synthetic_default")
        self.transitions = 0
        self.occupancy = {mix: _zero_occupancy() for mix in MIXES}
        self.long_body_exposure = {mix: 0 for mix in MIXES}
        # Natural hero decisions only. Sparse length bins stay bounded by arena
        # capacity; 500-frame bins expose late crowding without storing rows.
        self.length_histogram_by_frame = {
            mix: {str(start): {} for start in range(0, self.horizon, 500)} for mix in MIXES
        }
        self.departures = _zero_deciles()
        self._pending_departures: dict[int, list[int]] = {}
        self.drift = {
            decile: {"observations": 0, "kl_b8_to_policy_sum": 0.0, "greedy_disagreements": 0}
            for decile in DECILE_LABELS
        }
        self.runs = _zero_run_summary()
        self.run_outcomes = {
            outcome: {
                "lives": 0,
                "runs_ge2": 0,
                "runs_ge3": 0,
                "lives_with_run_ge2": 0,
                "lives_with_run_ge3": 0,
                "mass_ledger_ge2_sum": 0.0,
                "mass_ledger_ge3_sum": 0.0,
            }
            for outcome in ("surviving", "terminated")
        }
        self.endpoint_ledger: dict[str, dict[str, dict[str, int]]] = {mix: {} for mix in MIXES}
        # At most one unfinished life per world.  Finished-life quantities are
        # folded into counters immediately, avoiding unbounded episode storage.
        self._lives: dict[int, dict[str, Any]] = {}

    def _set_length_deciles(self, boundaries: Sequence[int], *, source: str) -> None:
        parsed = tuple(_finite_int(value, "length decile boundary") for value in boundaries)
        if len(parsed) != 9 or tuple(sorted(parsed)) != parsed:
            raise ValueError("length decile boundaries must be nine sorted non-negative integers")
        self._decile_boundaries = parsed
        self._decile_source = source

    def configure_length_deciles(self, bank_lengths: Any) -> tuple[int, ...]:
        """Freeze deciles from the immutable 10k ordinary-state bank.

        Cut ``k`` is sorted index ``ceil(k*N/10)-1`` for k=1..9. Membership
        is ``searchsorted(cuts, length, side='right')``: a tied length enters
        the higher decile. Empty tied deciles are retained; training never
        refits their edges.
        """
        if self._decile_boundaries is not None:
            raise RuntimeError("length deciles are immutable once configured")
        values = np.asarray(bank_lengths).reshape(-1)
        if len(values) != 10_000:
            raise ValueError("E8 fixed ordinary-state bank must contain exactly 10,000 lengths")
        ordered = np.sort(
            np.asarray([_finite_int(value, "bank length") for value in values], dtype=np.int64)
        )
        cuts = tuple(int(ordered[int(math.ceil(k * len(ordered) / 10) - 1)]) for k in range(1, 10))
        self._set_length_deciles(cuts, source="fixed_ordinary_bank_10k")
        for length, (decisions, departures) in self._pending_departures.items():
            row = self.departures[self._decile(length)]
            row["decisions"] += decisions
            row["departures"] += departures
        self._pending_departures.clear()
        return cuts

    def _decile(self, length: int) -> str:
        if self._decile_boundaries is None:
            raise RuntimeError("scientific E8 diagnostics require frozen 10k-bank length deciles")
        return DECILE_LABELS[int(np.searchsorted(self._decile_boundaries, length, side="right"))]

    @staticmethod
    def _arrays(raw: Any, actions: Any, b8_actions: Any) -> tuple[np.ndarray, ...]:
        names = (
            "world_id",
            "episode",
            "mix",
            "frame",
            "length",
            "reward",
            "done",
            "survived_horizon",
        )
        values = [np.asarray(getattr(raw, name)).reshape(-1) for name in names]
        size = len(values[0])
        if not size or any(len(value) != size for value in values[1:]):
            raise ValueError("raw batch fields must be non-empty matched vectors")
        candidate = np.asarray(actions).reshape(-1)
        baseline = np.asarray(b8_actions).reshape(-1)
        if len(candidate) != size or len(baseline) != size:
            raise ValueError("actions and b8_actions must match raw batch order")
        if np.any((candidate < 0) | (candidate >= 6)) or np.any((baseline < 0) | (baseline >= 6)):
            raise ValueError("actions must be legal six-action values")
        return (*values, candidate.astype(np.int64), baseline.astype(np.int64))

    def _life(self, world: int, episode: int) -> dict[str, Any]:
        life = self._lives.get(world)
        if life is None:
            life = {
                "episode": episode,
                "mass_ledger": 0.0,
                "current_run": 0,
                "runs_ge2": 0,
                "runs_ge3": 0,
                "has_ge2": False,
                "has_ge3": False,
            }
            self._lives[world] = life
        elif life["episode"] != episode:
            raise ValueError("new episode observed before prior life reached a terminal transition")
        return life

    @staticmethod
    def _close_run(life: dict[str, Any]) -> None:
        if life["current_run"] >= 2:
            life["runs_ge2"] += 1
            life["has_ge2"] = True
        if life["current_run"] >= 3:
            life["runs_ge3"] += 1
            life["has_ge3"] = True
        life["current_run"] = 0

    def _finish_life(self, world: int, *, survived: bool) -> None:
        life = self._lives.pop(world)
        self._close_run(life)
        self.runs["runs_ge2"] += life["runs_ge2"]
        self.runs["runs_ge3"] += life["runs_ge3"]
        for threshold in (2, 3):
            key = f"has_ge{threshold}"
            if life[key]:
                self.runs[f"lives_with_run_ge{threshold}"] += 1
                self.runs[f"mass_ledger_ge{threshold}_sum"] += life["mass_ledger"]
        self.runs["surviving_lives" if survived else "terminated_lives"] += 1
        outcome = self.run_outcomes["surviving" if survived else "terminated"]
        outcome["lives"] += 1
        for threshold in (2, 3):
            outcome[f"runs_ge{threshold}"] += life[f"runs_ge{threshold}"]
            outcome[f"lives_with_run_ge{threshold}"] += int(life[f"has_ge{threshold}"])
            if life[f"has_ge{threshold}"]:
                outcome[f"mass_ledger_ge{threshold}_sum"] += life["mass_ledger"]

    def observe(self, raw: Any, *, actions: Any, b8_actions: Any) -> None:
        """Ingest a complete collector batch once, in world-major order.

        The collector reward is post-transition alive length / 1000.  E8's
        full-H5000 mass ledger is therefore ``reward / 5`` per decision; a
        death contributes zero by construction.
        """
        (
            world_ids,
            episodes,
            mixes,
            frames,
            lengths,
            rewards,
            dones,
            survived_horizon,
            candidate,
            baseline,
        ) = self._arrays(raw, actions, b8_actions)
        for (
            world_raw,
            episode_raw,
            mix_raw,
            frame_raw,
            length_raw,
            reward_raw,
            done_raw,
            survived_raw,
            action,
            b8,
        ) in zip(
            world_ids,
            episodes,
            mixes,
            frames,
            lengths,
            rewards,
            dones,
            survived_horizon,
            candidate,
            baseline,
        ):
            world, episode = _finite_int(world_raw, "world_id"), _finite_int(episode_raw, "episode")
            frame, length = _finite_int(frame_raw, "frame"), _finite_int(length_raw, "length")
            mix = str(mix_raw)
            if mix not in MIXES:
                raise ValueError(f"unknown E8 mix {mix!r}")
            if frame >= self.horizon:
                raise ValueError("collector emitted a transition beyond the finite horizon")
            reward = float(reward_raw)
            if not math.isfinite(reward) or reward < 0.0:
                raise ValueError("reward must be finite and non-negative")
            life = self._life(world, episode)
            self.transitions += 1
            self.occupancy[mix][_length_bin(length)][_frame_bin(frame)] += 1
            histogram = self.length_histogram_by_frame[mix][str((frame // 500) * 500)]
            length_bucket = str(min(length // 100 * 100, 12000))
            histogram[length_bucket] = histogram.get(length_bucket, 0) + 1
            if length >= 800:
                self.long_body_exposure[mix] += 1
            departed = int(action) != int(b8)
            if self._decile_boundaries is None:
                if length > 145 * 83:
                    raise ValueError("pre-bank logical length exceeds arena capacity")
                row = self._pending_departures.setdefault(length, [0, 0])
                row[0] += 1
                row[1] += int(departed)
            else:
                decile = self._decile(length)
                self.departures[decile]["decisions"] += 1
                self.departures[decile]["departures"] += int(departed)
            if length >= 800 and departed:
                life["current_run"] += 1
            elif life["current_run"]:
                self._close_run(life)
            life["mass_ledger"] += reward / float(self.horizon / 1000)
            terminal = bool(done_raw)
            if terminal:
                # The collector exposes this explicit post-step fact. A death
                # on the final frame must not be relabelled as a survivor.
                self._finish_life(world, survived=bool(survived_raw))

    def observe_fixed_bank(self, *, policy_logits: Any, b8_logits: Any, lengths: Any) -> None:
        """Accumulate fixed-ordinary-bank KL(B8||pi) and greedy disagreement.

        The bank owner controls its immutable 10k observations.  This method
        stores only per-fixed-decile sums, never logits or observations.
        """
        policy = np.asarray(policy_logits, dtype=np.float64)
        teacher = np.asarray(b8_logits, dtype=np.float64)
        body = np.asarray(lengths).reshape(-1)
        if policy.ndim != 2 or policy.shape != teacher.shape or policy.shape[1] != 6:
            raise ValueError("policy and B8 logits must have matched [N, 6] shapes")
        if (
            len(body) != len(policy)
            or not np.isfinite(policy).all()
            or not np.isfinite(teacher).all()
        ):
            raise ValueError("fixed-bank inputs must be finite and aligned")
        teacher_log = teacher - np.logaddexp.reduce(teacher, axis=1, keepdims=True)
        policy_log = policy - np.logaddexp.reduce(policy, axis=1, keepdims=True)
        kl = np.exp(teacher_log) * (teacher_log - policy_log)
        disagreement = np.argmax(policy, axis=1) != np.argmax(teacher, axis=1)
        for index, length in enumerate(body):
            decile = self._decile(_finite_int(length, "length"))
            row = self.drift[decile]
            row["observations"] += 1
            row["kl_b8_to_policy_sum"] += float(kl[index].sum())
            row["greedy_disagreements"] += int(disagreement[index])

    def record_paired_endpoint(
        self,
        *,
        mix: str,
        b8_death_frame: int | None,
        b8_length: int,
        candidate_death_frame: int | None,
    ) -> None:
        """Record one final H5000 paired endpoint in the requested B8 stratum.

        ``None`` means alive at H5000.  A rescue is a candidate that remains
        alive past the B8 death; a re-death is a rescued candidate that later
        dies before H5000.  This deliberately belongs at evaluation time: a
        PPO collection rollout has no counterfactual B8 death frame.
        """
        if mix not in MIXES:
            raise ValueError(f"unknown E8 mix {mix!r}")
        b8_length = _finite_int(b8_length, "b8_length")
        if b8_death_frame is not None:
            b8_death_frame = _finite_int(b8_death_frame, "b8_death_frame")
            if b8_death_frame >= self.horizon:
                raise ValueError("B8 death frame must precede the horizon")
        if candidate_death_frame is not None:
            candidate_death_frame = _finite_int(candidate_death_frame, "candidate_death_frame")
            if candidate_death_frame >= self.horizon:
                raise ValueError("candidate death frame must precede the horizon")
        death_label = "survived" if b8_death_frame is None else str(b8_death_frame)
        key = f"b8_frame_{death_label}__length_{_length_bin(b8_length)}"
        row = self.endpoint_ledger[mix].setdefault(key, _zero_endpoint())
        row["pairs"] += 1
        b8_alive, candidate_alive = b8_death_frame is None, candidate_death_frame is None
        if b8_alive and candidate_alive:
            row["both_alive"] += 1
        elif candidate_alive:
            row["candidate_only"] += 1
        elif b8_alive:
            row["b8_only"] += 1
        else:
            row["both_dead"] += 1
        if b8_death_frame is not None and (
            candidate_death_frame is None or candidate_death_frame > b8_death_frame
        ):
            row["rescues"] += 1
            if candidate_death_frame is not None:
                row["re_deaths"] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable, side-effect-free report."""
        runs = copy.deepcopy(self.runs)
        runs["open_lives"] = len(self._lives)
        runs["censored_lives_with_run_ge2"] = 0
        runs["censored_lives_with_run_ge3"] = 0
        runs["censored_mass_ledger_ge2_sum"] = 0.0
        runs["censored_mass_ledger_ge3_sum"] = 0.0
        runs["open_runs_ge2"] = 0
        runs["open_runs_ge3"] = 0
        for life in self._lives.values():
            # A run already closed within an unfinished life is realised and
            # belongs in the headline count. Its life-level outcome is marked
            # censored until terminal, never silently discarded at a 1M cut.
            runs["runs_ge2"] += life["runs_ge2"]
            runs["runs_ge3"] += life["runs_ge3"]
            for threshold in (2, 3):
                runs[f"runs_ge{threshold}"] += int(life["current_run"] >= threshold)
                if life[f"has_ge{threshold}"] or life["current_run"] >= threshold:
                    runs[f"lives_with_run_ge{threshold}"] += 1
                    runs[f"mass_ledger_ge{threshold}_sum"] += life["mass_ledger"]
                    runs[f"censored_lives_with_run_ge{threshold}"] += 1
                    runs[f"censored_mass_ledger_ge{threshold}_sum"] += life["mass_ledger"]
            runs["open_runs_ge2"] += int(life["current_run"] >= 2)
            runs["open_runs_ge3"] += int(life["current_run"] >= 3)
        outcome_rows = copy.deepcopy(self.run_outcomes)
        for outcome in outcome_rows.values():
            for threshold in (2, 3):
                outcome[f"runs_ge{threshold}_per_life"] = (
                    outcome[f"runs_ge{threshold}"] / outcome["lives"] if outcome["lives"] else None
                )
        runs["by_completed_life_outcome"] = outcome_rows
        runs["observed_lives"] = (
            runs["terminated_lives"] + runs["surviving_lives"] + runs["open_lives"]
        )
        drift = copy.deepcopy(self.drift)
        for row in drift.values():
            count = row["observations"]
            row["mean_kl_b8_to_policy"] = row["kl_b8_to_policy_sum"] / count if count else None
            row["greedy_disagreement_rate"] = row["greedy_disagreements"] / count if count else None
        departures = copy.deepcopy(self.departures)
        for row in departures.values():
            row["departure_rate"] = (
                row["departures"] / row["decisions"] if row["decisions"] else None
            )
        return {
            "schema": self.schema,
            "horizon": self.horizon,
            "length_decile_definition": {
                "labels": list(DECILE_LABELS),
                "boundaries": (
                    None if self._decile_boundaries is None else list(self._decile_boundaries)
                ),
                "membership": "searchsorted(boundaries, length, side='right')",
                "source": self._decile_source,
                "requires_fixed_ordinary_bank": self.require_fixed_bank,
            },
            "transitions": self.transitions,
            "departures_pending_fixed_deciles": sum(
                x[0] for x in self._pending_departures.values()
            ),
            "occupancy": copy.deepcopy(self.occupancy),
            "long_body_exposure": dict(self.long_body_exposure),
            "hero_length_histogram_by_frame": {
                "frame_bucket_width": 500,
                "length_bucket_width": 100,
                "length_overflow_bucket": 12000,
                "bucket_intervals": "[start,start+width); length >=12000 in overflow",
                "by_mix": copy.deepcopy(self.length_histogram_by_frame),
            },
            "realised_departures_by_length_decile": departures,
            "fixed_ordinary_bank_drift_by_length_decile": drift,
            "non_greedy_runs_at_l_ge_800": runs,
            "endpoint_ledger_by_b8_death_frame_x_length": copy.deepcopy(self.endpoint_ledger),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return full bounded state for recovery; it is JSON-serialisable."""
        return {
            "report": self.snapshot(),
            "active_lives": copy.deepcopy(self._lives),
            "completed_run_totals": copy.deepcopy(self.runs),
            "run_outcome_totals": copy.deepcopy(self.run_outcomes),
            "pending_departures": copy.deepcopy(self._pending_departures),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a :meth:`state_dict` payload, rejecting schema drift."""
        report = state.get("report")
        active = state.get("active_lives")
        if not isinstance(report, Mapping) or report.get("schema") != self.schema:
            raise ValueError("invalid E8 diagnostics state schema")
        if report.get("horizon") != self.horizon or not isinstance(active, Mapping):
            raise ValueError("invalid E8 diagnostics recovery state")
        definition = report.get("length_decile_definition")
        if not isinstance(definition, Mapping) or definition.get("labels") != list(DECILE_LABELS):
            raise ValueError("invalid E8 length decile definition")
        boundaries = definition.get("boundaries")
        if boundaries is None:
            if self._decile_boundaries is not None:
                raise ValueError("recovery state length deciles differ from this diagnostic")
        elif isinstance(boundaries, list):
            incoming = tuple(_finite_int(value, "length decile boundary") for value in boundaries)
            if self._decile_boundaries is not None and self._decile_boundaries != incoming:
                raise ValueError("recovery state length deciles differ from this diagnostic")
            self._set_length_deciles(incoming, source=str(definition.get("source")))
        else:
            raise ValueError("invalid recovered length deciles")
        self._pending_departures = {
            int(key): list(values) for key, values in state.get("pending_departures", {}).items()
        }
        self.transitions = _finite_int(report.get("transitions"), "transitions")
        self.occupancy = copy.deepcopy(report["occupancy"])
        self.long_body_exposure = copy.deepcopy(report["long_body_exposure"])
        histogram = report.get("hero_length_histogram_by_frame")
        if histogram is not None:
            if (
                histogram.get("frame_bucket_width") != 500
                or histogram.get("length_bucket_width") != 100
            ):
                raise ValueError("invalid recovered histogram buckets")
            self.length_histogram_by_frame = copy.deepcopy(histogram["by_mix"])
        self.departures = copy.deepcopy(report["realised_departures_by_length_decile"])
        for row in self.departures.values():
            row.pop("departure_rate", None)
        self.drift = copy.deepcopy(report["fixed_ordinary_bank_drift_by_length_decile"])
        for row in self.drift.values():
            row.pop("mean_kl_b8_to_policy", None)
            row.pop("greedy_disagreement_rate", None)
        if not isinstance(state.get("completed_run_totals"), Mapping):
            raise ValueError("missing completed run totals in recovery state")
        self.runs = copy.deepcopy(state["completed_run_totals"])
        self.run_outcomes = copy.deepcopy(state["run_outcome_totals"])
        self.runs.pop("open_lives", None)
        self.runs.pop("open_runs_ge2", None)
        self.runs.pop("open_runs_ge3", None)
        self.endpoint_ledger = copy.deepcopy(report["endpoint_ledger_by_b8_death_frame_x_length"])
        self._lives = {int(key): copy.deepcopy(value) for key, value in active.items()}
