"""Small, fail-closed primitives for finite raster-observation diagnostics.

This module deliberately does not estimate values or infer hidden history.  It
only makes reproducible snapshots and finite, fixed-tape counterfactual
rollouts available to a higher-level diagnostic/reporting layer.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import struct
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from src.model.obs_spec import RASTER31V3
from src.simd_env.batch_sim import BatchSim
from src.simd_env.featurizer import build_observations, obs_inputs_from_batch_sim
from src.simd_env.rng import EnvRng


@dataclass(frozen=True)
class ProbeConfig:
    """Feature normalization and finite-rollout limits for a probe."""

    max_frames: int = 5000
    starvation_max: int = 500
    max_length: int = 400
    gamma: float = 0.99


def _validate_config(config: ProbeConfig) -> None:
    if not isinstance(config, ProbeConfig):
        raise TypeError("config must be a ProbeConfig")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (config.max_frames, config.starvation_max, config.max_length)
    ):
        raise ValueError("max_frames, starvation_max, and max_length must be positive integers")
    if not isinstance(config.gamma, (float, int)) or isinstance(config.gamma, bool):
        raise ValueError("gamma must be a finite float in (0, 1]")
    if not math.isfinite(float(config.gamma)) or not 0.0 < float(config.gamma) <= 1.0:
        raise ValueError("gamma must be a finite float in (0, 1]")


def _write_tag(hasher: "hashlib._Hash", tag: bytes) -> None:
    hasher.update(struct.pack(">I", len(tag)))
    hasher.update(tag)


def _write_count(hasher: "hashlib._Hash", count: int) -> None:
    _write_tag(hasher, b"count")
    _write_tag(hasher, str(count).encode("ascii"))


def _canonical_hash(value: Any, hasher: "hashlib._Hash", seen: set[int]) -> None:
    """Hash a complete supported object graph, rejecting unknown/cyclic state."""
    if isinstance(value, type(None)):
        _write_tag(hasher, b"none")
    elif isinstance(value, bool):
        _write_tag(hasher, b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        _write_tag(hasher, b"int")
        _write_tag(hasher, str(value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("state digest rejects non-finite floating-point state")
        _write_tag(hasher, b"float")
        hasher.update(struct.pack(">d", value))
    elif isinstance(value, str):
        _write_tag(hasher, b"str")
        _write_tag(hasher, value.encode("utf-8"))
    elif isinstance(value, bytes):
        _write_tag(hasher, b"bytes")
        _write_tag(hasher, value)
    elif isinstance(value, np.generic):
        _write_tag(hasher, b"numpy-scalar")
        _write_tag(hasher, value.dtype.str.encode("ascii"))
        _canonical_hash(value.item(), hasher, seen)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            identity = id(value)
            if identity in seen:
                raise ValueError("state digest rejects cyclic state")
            seen.add(identity)
            try:
                _write_tag(hasher, b"ndarray-object")
                _write_tag(hasher, value.dtype.str.encode("ascii"))
                _write_tag(hasher, repr(value.shape).encode("ascii"))
                _write_count(hasher, value.size)
                for item in value.flat:
                    _canonical_hash(item, hasher, seen)
                _write_tag(hasher, b"end-ndarray-object")
            finally:
                seen.remove(identity)
        else:
            if not np.isfinite(value).all() if np.issubdtype(value.dtype, np.inexact) else False:
                raise ValueError("state digest rejects non-finite ndarray state")
            _write_tag(hasher, b"ndarray")
            _write_tag(hasher, value.dtype.str.encode("ascii"))
            _write_tag(hasher, repr(value.shape).encode("ascii"))
            _write_tag(hasher, np.ascontiguousarray(value).tobytes())
    else:
        identity = id(value)
        if identity in seen:
            raise ValueError("state digest rejects cyclic state")
        seen.add(identity)
        try:
            if isinstance(value, tuple):
                _write_tag(hasher, b"tuple")
                _write_count(hasher, len(value))
                for item in value:
                    _canonical_hash(item, hasher, seen)
                _write_tag(hasher, b"end-tuple")
            elif isinstance(value, list):
                _write_tag(hasher, b"list")
                _write_count(hasher, len(value))
                for item in value:
                    _canonical_hash(item, hasher, seen)
                _write_tag(hasher, b"end-list")
            elif isinstance(value, set):
                _write_tag(hasher, b"set")
                _write_count(hasher, len(value))
                encoded = []
                for item in value:
                    item_hash = hashlib.sha256()
                    _canonical_hash(item, item_hash, seen.copy())
                    encoded.append(item_hash.digest())
                for item_hash in sorted(encoded):
                    _write_tag(hasher, item_hash)
                _write_tag(hasher, b"end-set")
            elif isinstance(value, dict):
                _write_tag(hasher, b"dict")
                if not all(isinstance(key, str) for key in value):
                    raise ValueError("state digest only supports string-keyed dictionaries")
                _write_count(hasher, len(value))
                for key in sorted(value):
                    _write_tag(hasher, key.encode("utf-8"))
                    _canonical_hash(value[key], hasher, seen)
                _write_tag(hasher, b"end-dict")
            elif is_dataclass(value) and not isinstance(value, type):
                _write_tag(
                    hasher,
                    f"dataclass:{type(value).__module__}.{type(value).__qualname__}".encode(),
                )
                _write_count(hasher, len(fields(value)))
                for field in fields(value):
                    _write_tag(hasher, field.name.encode())
                    _canonical_hash(getattr(value, field.name), hasher, seen)
                _write_tag(hasher, b"end-dataclass")
            elif isinstance(value, random.Random):
                _write_tag(hasher, b"random.Random")
                _canonical_hash(value.getstate(), hasher, seen)
                _write_tag(hasher, b"end-random.Random")
            elif isinstance(value, EnvRng):
                _write_tag(
                    hasher, f"object:{type(value).__module__}.{type(value).__qualname__}".encode()
                )
                _canonical_hash(vars(value), hasher, seen)
                _write_tag(hasher, b"end-EnvRng")
            else:
                raise ValueError(
                    f"unsupported state value: {type(value).__module__}.{type(value).__qualname__}"
                )
        finally:
            seen.remove(identity)


def full_state_digest(sim: BatchSim) -> str:
    """Return a canonical digest over every dynamic attribute of ``sim``.

    The serializer includes object arrays, ordered food lists, sets, and RNG
    wrapper state.  An unfamiliar attribute fails closed instead of being
    silently omitted.
    """
    if not isinstance(sim, BatchSim):
        raise TypeError("sim must be a BatchSim")
    hasher = hashlib.sha256()
    _canonical_hash(vars(sim), hasher, set())
    return hasher.hexdigest()


def clone_sim(sim: BatchSim) -> BatchSim:
    """Deep-copy a simulator and prove the copy has an identical full digest."""
    before = full_state_digest(sim)
    clone = copy.deepcopy(sim)
    if not isinstance(clone, BatchSim) or full_state_digest(clone) != before:
        raise RuntimeError("failed to create a complete independent BatchSim clone")
    return clone


def _validate_single_hero(sim: BatchSim, hero: int, config: ProbeConfig) -> None:
    _validate_config(config)
    if not isinstance(sim, BatchSim) or sim.E != 1:
        raise ValueError("observation probes currently require a BatchSim with num_envs == 1")
    if not isinstance(hero, int) or isinstance(hero, bool) or not 0 <= hero < sim.S:
        raise ValueError(f"hero must be an integer in [0, {sim.S})")


def observe_hero(sim: BatchSim, hero: int, config: ProbeConfig) -> dict[str, np.ndarray]:
    """Copy the canonical v3 observation and resolved mask for one hero."""
    _validate_single_hero(sim, hero, config)
    inputs = obs_inputs_from_batch_sim(
        sim,
        max_frames=config.max_frames,
        starvation_max=config.starvation_max,
        max_length=config.max_length,
    )
    obs = build_observations(inputs, mask=sim.get_resolved_action_mask(), obs_spec=RASTER31V3)
    return {
        "tactical_uint8": np.array(obs["tactical_uint8"][0, hero], copy=True),
        "strategic_uint8": np.array(obs["strategic_uint8"][0, hero], copy=True),
        "scalars": np.array(obs["scalars"][0, hero], copy=True),
        "mask": np.array(obs["mask"][0, hero], dtype=bool, copy=True),
    }


def _obs_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        lhs, rhs = left[key], right[key]
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.tobytes() != rhs.tobytes():
            return False
    return True


def heading_twin(
    sim: BatchSim, hero: int, enemy: int, new_heading: int, config: ProbeConfig
) -> BatchSim | None:
    """Return a geometric heading twin only when the hero observation is equal.

    This is an admissibility helper, not evidence that the altered heading has
    a realizable history.  Callers must report it as geometric susceptibility.
    """
    _validate_single_hero(sim, hero, config)
    if not isinstance(enemy, int) or isinstance(enemy, bool) or not 0 <= enemy < sim.S:
        raise ValueError(f"enemy must be an integer in [0, {sim.S})")
    if (
        not isinstance(new_heading, int)
        or isinstance(new_heading, bool)
        or not 0 <= new_heading < 4
    ):
        raise ValueError("new_heading must be an integer in [0, 4)")
    if (
        enemy == hero
        or not sim.alive[0, hero]
        or not sim.alive[0, enemy]
        or int(sim.seg_count[0, enemy]) != 1
        or int(sim.direction[0, enemy]) == new_heading
    ):
        return None
    original = observe_hero(sim, hero, config)
    twin = clone_sim(sim)
    twin.direction[0, enemy] = new_heading
    active = np.ones(twin.E, dtype=bool)
    twin._rebuild_traversed_from_heads(active)
    twin._refresh_action_masks(active)
    return twin if _obs_equal(original, observe_hero(twin, hero, config)) else None


def _validate_tape(joint_actions: Any, sim: BatchSim, horizons: Sequence[int]) -> np.ndarray:
    tape = np.asarray(joint_actions)
    if tape.ndim != 3 or tape.shape[1:] != (1, sim.S):
        raise ValueError(f"joint_actions must have shape (H, 1, {sim.S})")
    if tape.shape[0] < max(horizons):
        raise ValueError("joint_actions is shorter than the requested horizon")
    if (
        not np.issubdtype(tape.dtype, np.integer)
        or isinstance(joint_actions, np.ndarray)
        and tape.dtype == np.bool_
    ):
        raise ValueError("joint_actions must contain integer relative actions")
    if np.any(tape < 0) or np.any(tape > 2):
        raise ValueError("joint_actions must contain only relative actions 0, 1, or 2")
    return tape.astype(np.int64, copy=True)


def finite_action_returns(
    sim: BatchSim,
    hero: int,
    joint_actions: Any,
    horizons: Iterable[int],
    config: ProbeConfig,
    *,
    action_order: Sequence[int] = (0, 1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Roll each initially resolved action over a common fixed continuation tape.

    Each horizon is the total number of executed rows, including the candidate
    first action.  Tape row zero supplies every opponent's first action while
    the hero action is overwritten by the candidate. Rewards after a branch
    stop are zero by definition.  No branch is
    reset or bootstrapped, and no return is an optimal or learned value claim.
    """
    _validate_single_hero(sim, hero, config)
    horizon_values = tuple(horizons)
    if not horizon_values or any(
        not isinstance(h, int) or isinstance(h, bool) or h <= 0 for h in horizon_values
    ):
        raise ValueError("horizons must be a non-empty sequence of positive integers")
    if len(set(horizon_values)) != len(horizon_values):
        raise ValueError("horizons must not contain duplicates")
    tape = _validate_tape(joint_actions, sim, horizon_values)
    order = tuple(action_order)
    if len(order) != 6 or any(type(action) is not int for action in order):
        raise ValueError("action_order must contain six built-in integer actions")
    if sorted(order) != list(range(6)):
        raise ValueError("action_order must be a permutation of actions 0..5")
    if not (sim.train_mode and not sim.allow_respawn and sim.v2):
        raise ValueError(
            "finite probes require mechanics-v2 train_mode=True and allow_respawn=False"
        )
    if float(config.gamma) != float(sim.cfg.gamma):
        raise ValueError("ProbeConfig.gamma must equal sim.cfg.gamma")
    if not sim.alive[0, hero]:
        raise ValueError("finite probes require a living hero")
    if bool(sim.population_floor_reached()[0]):
        raise ValueError("finite probes reject worlds already at the population floor")
    if int(sim.frame[0]) >= config.max_frames:
        raise ValueError("finite probes reject worlds already at the frame cap")

    source_digest = full_state_digest(sim)
    initial = observe_hero(sim, hero, config)
    initial_mask = initial["mask"].copy()
    maximum = max(horizon_values)
    records: dict[int, dict[str, Any]] = {}
    for action in order:
        if not initial_mask[action]:
            records[action] = None
            continue
        branch = clone_sim(sim)
        actions = tape[0, 0].copy()
        actions[hero] = action
        branch.step(actions.reshape(1, branch.S))
        rewards = [float(branch.get_reward()[0, hero])]
        if not math.isfinite(rewards[0]):
            raise ValueError("finite probes reject non-finite branch rewards")
        actual_steps = 1
        valid_agent_transitions = int(branch.get_transition_valid().sum())
        done = bool(branch.get_done()[0, hero])
        stop_cause: str | None = "hero_death" if done else None
        if stop_cause is None and bool(branch.population_floor_reached()[0]):
            stop_cause = "population_floor"
        if stop_cause is None and int(branch.frame[0]) >= config.max_frames:
            stop_cause = "max_frames"
        first = {
            "reward": rewards[0],
            "done": done,
            "next_observation_digest": _observation_digest(observe_hero(branch, hero, config)),
        }
        for index in range(1, maximum):
            if stop_cause is not None:
                rewards.append(0.0)
                continue
            actions = tape[index, 0].copy()
            branch.step(actions.reshape(1, branch.S))
            actual_steps += 1
            valid_agent_transitions += int(branch.get_transition_valid().sum())
            reward = float(branch.get_reward()[0, hero])
            if not math.isfinite(reward):
                raise ValueError("finite probes reject non-finite branch rewards")
            rewards.append(reward)
            if bool(branch.get_done()[0, hero]):
                stop_cause = "hero_death"
            elif bool(branch.population_floor_reached()[0]):
                stop_cause = "population_floor"
            elif int(branch.frame[0]) >= config.max_frames:
                stop_cause = "max_frames"
        values = {
            horizon: float(
                sum((float(config.gamma) ** step) * rewards[step] for step in range(horizon))
            )
            for horizon in horizon_values
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("finite probes reject non-finite discounted returns")
        records[action] = {
            "discounted_return_by_horizon": values,
            "first_step": first,
            "actual_steps": actual_steps,
            "valid_agent_transitions": valid_agent_transitions,
            "termination_cause": stop_cause,
            "termination_counts": {
                "hero_death": int(stop_cause == "hero_death"),
                "population_floor": int(stop_cause == "population_floor"),
                "max_frames": int(stop_cause == "max_frames"),
            },
        }
    if full_state_digest(sim) != source_digest:
        raise RuntimeError("finite_action_returns mutated its source simulator")
    return {
        "mask": initial_mask,
        "initial_mask": initial_mask,
        "actions": records,
        "horizons_are_total_executed_steps": True,
        "source_fingerprint": source_digest,
    }


def _observation_digest(observation: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    _canonical_hash(observation, hasher, set())
    return hasher.hexdigest()
