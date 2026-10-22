"""Prospective P1-only complete-life late-body critic support.

This module deliberately builds support before critic fitting.  The fixture is
never part of actor data, ordinary-state diagnostics, or an efficacy bank.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Callable, Iterable

import numpy as np
import torch

from research.bl_ppo.collector import CollectorConfig, PersistentCollector, _world_seed

LATE_BODY_MIN_LENGTH = 800
FIXED_LATE_BODY_WORLDS = 12
FIXED_LATE_BODY_SEED_NAMESPACE = 2026102111
FIXED_LATE_BODY_LONG_LENGTH = 900
REFIT_LONG_BODY_START_FRACTION = 0.25
FIXED_HOLDOUT_WORLD_OFFSET = 2_000_000
SUPPLEMENT_WORLD_OFFSET = 1_000_000


def fixed_holdout_config(excluded_seeds: Iterable[int], *, horizon: int) -> CollectorConfig:
    """Return the dedicated, disjoint first-life fixed-support collector config."""
    return CollectorConfig(
        worlds=FIXED_LATE_BODY_WORLDS,
        processes=min(6, FIXED_LATE_BODY_WORLDS),
        rollout_steps=128,
        horizon=horizon,
        seed=FIXED_LATE_BODY_SEED_NAMESPACE,
        excluded_seeds=frozenset(int(seed) for seed in excluded_seeds),
        long_bodies=True,
        world_id_offset=FIXED_HOLDOUT_WORLD_OFFSET,
        preflight=True,
    )


def fixed_holdout_seeds(excluded_seeds: Iterable[int], *, horizon: int) -> tuple[int, ...]:
    """Reserve the actual dedicated first-life seeds before main collection."""
    config = fixed_holdout_config(excluded_seeds, horizon=horizon)
    seeds = tuple(
        _world_seed(config, config.world_id_offset + world, 0) for world in range(config.worlds)
    )
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("fixed late-body roots are not unique")
    return seeds


def support_identity(excluded_seeds: Iterable[int], *, horizon: int) -> dict[str, Any]:
    """Frozen configuration and root identities for snapshot/resume comparison."""
    seeds = fixed_holdout_seeds(excluded_seeds, horizon=horizon)
    encoded = ",".join(str(seed) for seed in seeds).encode()
    return {
        "schema": "bl-ppo-p1-late-body-support/v1",
        "fixture": "collector._inject_long/inject_serpentines",
        "worlds": FIXED_LATE_BODY_WORLDS,
        "seed_namespace": FIXED_LATE_BODY_SEED_NAMESPACE,
        "first_life_seeds": list(seeds),
        "first_life_seeds_sha256": hashlib.sha256(encoded).hexdigest(),
        "minimum_length": LATE_BODY_MIN_LENGTH,
        "hero_initial_length": FIXED_LATE_BODY_LONG_LENGTH,
        "refit_long_body_start_fraction": REFIT_LONG_BODY_START_FRACTION,
        "refit_schedule": "(world_id + episode + 3) % 4 == 0",
        "fallback_trigger": "zero_completed_fit_late_support_before_accuracy",
        "supplement_world_id_offset": SUPPLEMENT_WORLD_OFFSET,
        "fixed_holdout_world_id_offset": FIXED_HOLDOUT_WORLD_OFFSET,
        "supplement_max_steps_per_world": horizon,
        "horizon": horizon,
        "first_life_only": True,
        "actor_data": "excluded",
        "ordinary_bank": "excluded",
    }


def collect_fixed_late_body_holdout(
    adapter: Any,
    bank: Any,
    *,
    excluded_seeds: Iterable[int],
    horizon: int,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Collect one complete injected life per dedicated world into holdout only.

    The adapter is an isolated frozen-initial stochastic actor.  Its caller
    owns its torch RNG fork so this support collection cannot alter the main
    refit/actor action stream.
    """
    config = fixed_holdout_config(excluded_seeds, horizon=horizon)
    complete: set[int] = set()
    transitions = 0
    first_life_transitions = 0
    remaining = config.horizon
    with PersistentCollector(config) as collector:
        while remaining:
            if should_stop is not None and should_stop():
                raise ValueError("fixed late-body support collection interrupted")
            steps = min(config.rollout_steps, remaining)
            raw = collector.collect(adapter, steps=steps)
            transitions += int(raw.active_transitions)
            if on_progress is not None:
                on_progress(transitions)
            adapter.take_rollout(raw, collector.peek())
            first_life = np.asarray(raw.episode, dtype=np.int64) == 0
            if first_life.any():
                features = adapter.last_critic_features.transpose(0, 1).reshape(-1, 257)
                rows = torch.as_tensor(first_life)
                bank.add_batch(
                    world_ids=torch.as_tensor(raw.world_id[first_life]),
                    critic_features=features[rows],
                    lengths=torch.as_tensor(raw.length[first_life]),
                    remaining_horizon=torch.as_tensor(
                        (horizon - raw.frame[first_life].astype(np.float32)) / horizon
                    ),
                    rewards=torch.as_tensor(raw.reward[first_life]),
                    terminals=torch.as_tensor(raw.done[first_life]),
                    split_override="holdout",
                    pin_first_late=True,
                    source="fixed_injected_long_body",
                )
                first_life_transitions += int(first_life.sum())
            complete.update(int(world) for world in raw.world_id[raw.done & first_life])
            remaining -= steps
    if len(complete) != config.worlds:
        raise ValueError(
            "fixed late-body collector did not complete every first life by its horizon"
        )
    support = bank.support_counts()
    fixed_late = bank.late_support_by_source().get("fixed_injected_long_body", 0)
    if fixed_late < config.worlds:
        raise ValueError(
            "fixed late-body collector did not retain one completed support row per world"
        )
    return {
        "worlds_complete": len(complete),
        "transitions": transitions,
        "first_life_target_transitions": first_life_transitions,
        "holdout_support": support["holdout"],
        "fixed_late_support": fixed_late,
    }


def collect_refit_late_supplement(
    adapter: Any,
    bank: Any,
    config: CollectorConfig,
    *,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Bounded, fit-only 25%-root support supplement after a zero-support check."""
    if config.long_body_fraction != 0.25:
        raise ValueError("late-body supplement must use the registered 25% root fraction")
    supplement = replace(config, world_id_offset=SUPPLEMENT_WORLD_OFFSET)
    transitions = 0
    fit_target_transitions = 0
    remaining = supplement.horizon
    with PersistentCollector(supplement) as collector:
        while remaining:
            if should_stop is not None and should_stop():
                raise ValueError("late-body refit support supplement interrupted")
            steps = min(supplement.rollout_steps, remaining)
            raw = collector.collect(adapter, steps=steps)
            transitions += int(raw.active_transitions)
            if on_progress is not None:
                on_progress(transitions)
            adapter.take_rollout(raw, collector.peek())
            features = adapter.last_critic_features.transpose(0, 1).reshape(-1, 257)
            fit_rows = np.asarray(raw.world_id, dtype=np.int64) % 5 != 0
            fit_target_transitions += int(fit_rows.sum())
            bank.add_batch(
                world_ids=torch.as_tensor(raw.world_id[fit_rows]),
                critic_features=features[torch.as_tensor(fit_rows)],
                lengths=torch.as_tensor(raw.length[fit_rows]),
                remaining_horizon=torch.as_tensor(
                    (supplement.horizon - raw.frame[fit_rows].astype(np.float32))
                    / supplement.horizon
                ),
                rewards=torch.as_tensor(raw.reward[fit_rows]),
                terminals=torch.as_tensor(raw.done[fit_rows]),
                split_override="fit",
                pin_first_late=True,
                source="refit_long_body_support_supplement",
            )
            remaining -= steps
    late = bank.late_support_by_source("fit").get("refit_long_body_support_supplement", 0)
    first_life_seeds = [
        _world_seed(supplement, supplement.world_id_offset + world, 0)
        for world in range(supplement.worlds)
    ]
    excluded_encoded = ",".join(str(seed) for seed in sorted(supplement.excluded_seeds)).encode()
    return {
        "transitions": transitions,
        "late_fit_support": late,
        "fit_target_transitions": fit_target_transitions,
        "max_steps_per_world": config.horizon,
        "first_life_seeds": first_life_seeds,
        "excluded_seed_count": len(supplement.excluded_seeds),
        "excluded_seeds_sha256": hashlib.sha256(excluded_encoded).hexdigest(),
    }
