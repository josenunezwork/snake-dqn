"""Pure, pre-dispatch byte accounting for resumable PPO snapshots.

The values in this module are deliberately serialization allowances, rather
than observations from an in-progress run.  They make the export contract
deterministic and reviewable in a manifest before a worker starts.
"""

from __future__ import annotations

from typing import Any, Mapping

MEBIBYTE = 1024**2
SCHEMA = "bl-ppo-snapshot-byte-bounds/v1"
GENEROSITY_MULTIPLIER = 2

# Fixed storage contracts from FixedOrdinaryBank and CompleteLifeRefitBank.
ORDINARY_CAPACITY = 10_000
REFIT_RESERVOIR_PER_SPLIT = 10_000
REFIT_SAMPLES_PER_OPEN_LIFE = 8
REFIT_PINNED_PER_OPEN_LIFE = 1
REFIT_OPEN_WORLD_NAMESPACES = 2  # natural collector plus the disjoint supplement

# ``torch.save`` encodes numpy uint8 payloads conservatively.  This allowance
# includes the largest recurrent raw row (the feed-forward row is smaller),
# pickle/container framing, and the current fixed row metadata.
ORDINARY_ROW_BYTES = 2 * (7 * 31**2 + 4 * 37**2 + 60 * 4 + 6 + 1 + 4 + 4 + 256 * 4) + 2048
# A refit sample has 257 Python float features plus life/provenance metadata.
REFIT_SAMPLE_BYTES = 4096
# The collector snapshot persists its cell-exact fixed simulation arrays.  Keep
# the layout arithmetic here rather than a hand-maintained MiB constant.
SNAKES_PER_WORLD = 6
SIM_MAX_CAPACITY = 10_400
SIM_BODY_COORDINATES = 2
SIM_BODY_INT_BYTES = 8
PADDED_GRID_HEIGHT = 83 + 2 * 24
PADDED_GRID_WIDTH = 200
INTERIOR_GRID_HEIGHT = 83
INTERIOR_GRID_WIDTH = 145
GRID_ARRAYS_PER_WORLD = 7
FROZEN_MODELS_PER_PROCESS = 10  # two frozen refs, five slots each
APEX_FROZEN_MODEL_BYTES = 1_719_324
PPO_POLICY_BYTES = 14_685_128
PPO_ADAM_STATE_BYTES = 16_264_128
ROLLOUT_BUFFER_BYTES_PER_CELL = 257 * 4 + 4 + 8 + 8
PRIMARY_METADATA_BYTES = 32 * MEBIBYTE
# ``refit_seen_seeds`` is a set of packed integer roots.  This deliberately
# allows a distinct serialised root for every raw refit transition.
REFIT_SEEN_SEED_BYTES = 16
TRACE_ROW_BYTES = 128
MONITOR_REPORT_BYTES = 64 * 1024


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _component(base: int) -> dict[str, int]:
    return {"base_bytes": base, "effective_bytes": base * GENEROSITY_MULTIPLIER}


def maximum_refit_raw_transitions(
    worlds: int,
    rollout_steps: int,
    *,
    max_fit_transitions: int = 4_000_000,
) -> int:
    """Bound raw refit collection for the registered 4:1 fit/holdout split.

    A complete collector call can overshoot the fit threshold by one full
    rollout.  The return is the pre-data raw-transition maximum used by both
    the runtime and P1 manifest arithmetic.
    """
    worlds = _positive_int(worlds, "worlds")
    rollout_steps = _positive_int(rollout_steps, "rollout_steps")
    max_fit_transitions = _positive_int(max_fit_transitions, "max_fit_transitions")
    if worlds < 1 or rollout_steps < 1 or max_fit_transitions < 1:
        raise ValueError("refit maxima and collector dimensions must be positive")
    # The split is world_id % 5 != 0, so a 128-world collector has 102 fit
    # worlds (not an idealised 4/5 fraction).  Ceiling preserves an incomplete
    # final split, then a whole call covers the boundary overshoot.
    fit_worlds = worlds - (worlds + 4) // 5
    if fit_worlds < 1:
        raise ValueError("collector shape has no critic-fit worlds")
    return (max_fit_transitions * worlds + fit_worlds - 1) // fit_worlds + worlds * rollout_steps


def derive_snapshot_byte_bounds(
    *,
    worlds: int,
    processes: int,
    rollout_steps: int,
    max_refit_transitions: int,
    max_actor_transitions: int,
    trace_actions: int = 0,
) -> dict[str, Any]:
    """Return a JSON-safe snapshot closure budget from frozen run maxima.

    ``max_refit_transitions`` is the maximum raw collector count, including
    any prospectively registered completion/support collection.  Refit and
    ordinary banks saturate at their registered reservoir capacities; only the
    root identity history is intentionally transition-linear.
    """
    worlds = _positive_int(worlds, "worlds")
    processes = _positive_int(processes, "processes")
    rollout_steps = _positive_int(rollout_steps, "rollout_steps")
    max_refit_transitions = _positive_int(max_refit_transitions, "max_refit_transitions")
    max_actor_transitions = _positive_int(max_actor_transitions, "max_actor_transitions")
    trace_actions = _positive_int(trace_actions, "trace_actions")
    if worlds < 1 or rollout_steps < 1:
        raise ValueError("worlds and rollout_steps must be positive")

    ordinary_rows = min(max_refit_transitions, ORDINARY_CAPACITY)
    refit_rows = min(
        max_refit_transitions,
        2 * REFIT_RESERVOIR_PER_SPLIT
        + REFIT_OPEN_WORLD_NAMESPACES
        * worlds
        * (REFIT_SAMPLES_PER_OPEN_LIFE + REFIT_PINNED_PER_OPEN_LIFE),
    )
    # The fixed reports grow once per 1M hero transitions.  Include refit and
    # actor maxima because both can pass monitor boundaries in a resumed run.
    monitor_reports = (max_refit_transitions + max_actor_transitions + 999_999) // 1_000_000
    body_bytes = (
        worlds * SNAKES_PER_WORLD * SIM_MAX_CAPACITY * SIM_BODY_COORDINATES * SIM_BODY_INT_BYTES
    )
    # Numpy grids are independently serialised even where an interior view
    # aliases a padded array, so budget both representations and framing.
    grid_bytes = (
        worlds
        * GRID_ARRAYS_PER_WORLD
        * (PADDED_GRID_HEIGHT * PADDED_GRID_WIDTH + INTERIOR_GRID_HEIGHT * INTERIOR_GRID_WIDTH)
    )
    frozen_model_bytes = processes * FROZEN_MODELS_PER_PROCESS * APEX_FROZEN_MODEL_BYTES
    rollout_bytes = worlds * rollout_steps * ROLLOUT_BUFFER_BYTES_PER_CELL
    dynamic_primary = (
        max_refit_transitions * REFIT_SEEN_SEED_BYTES
        + trace_actions * TRACE_ROW_BYTES
        + monitor_reports * MONITOR_REPORT_BYTES
    )
    primary_base = (
        # Numpy's protocol-2 representation may expand non-ASCII storage;
        # retain a full second body allowance just as for the aliased grids.
        2 * body_bytes
        + 2 * grid_bytes
        + frozen_model_bytes
        + PPO_POLICY_BYTES
        + PPO_ADAM_STATE_BYTES
        + rollout_bytes
        + PRIMARY_METADATA_BYTES
        + dynamic_primary
    )
    components = {
        "ordinary_bank": _component(ordinary_rows * ORDINARY_ROW_BYTES),
        "refit_bank": _component(refit_rows * REFIT_SAMPLE_BYTES),
        # P1's quarantined fixed late-body bank has the same bounded shape.
        "fixture_late_body_bank": _component(refit_rows * REFIT_SAMPLE_BYTES),
        "primary_snapshot": _component(primary_base),
    }
    closure_base = sum(row["base_bytes"] for row in components.values())
    closure_effective = sum(row["effective_bytes"] for row in components.values())
    return {
        "schema": SCHEMA,
        "generosity_multiplier": GENEROSITY_MULTIPLIER,
        "inputs": {
            "worlds": worlds,
            "processes": processes,
            "rollout_steps": rollout_steps,
            "max_refit_transitions": max_refit_transitions,
            "max_actor_transitions": max_actor_transitions,
            "trace_actions": trace_actions,
        },
        "row_sizes": {
            "ordinary_row_bytes": ORDINARY_ROW_BYTES,
            "refit_sample_bytes": REFIT_SAMPLE_BYTES,
            "refit_open_world_namespaces": REFIT_OPEN_WORLD_NAMESPACES,
            "refit_seen_seed_bytes": REFIT_SEEN_SEED_BYTES,
            "trace_row_bytes": TRACE_ROW_BYTES,
            "body_bytes": body_bytes,
            "grid_bytes": grid_bytes,
            "frozen_model_bytes": frozen_model_bytes,
            "rollout_bytes": rollout_bytes,
            "snakes_per_world": SNAKES_PER_WORLD,
            "sim_max_capacity": SIM_MAX_CAPACITY,
            "body_coordinate_bytes": SIM_BODY_INT_BYTES,
            "grid_arrays_per_world": GRID_ARRAYS_PER_WORLD,
            "frozen_models_per_process": FROZEN_MODELS_PER_PROCESS,
            "apex_frozen_model_bytes": APEX_FROZEN_MODEL_BYTES,
            "ppo_policy_bytes": PPO_POLICY_BYTES,
            "ppo_adam_state_bytes": PPO_ADAM_STATE_BYTES,
            "primary_metadata_bytes": PRIMARY_METADATA_BYTES,
        },
        "retained_rows": {
            "ordinary": ordinary_rows,
            "refit_each": refit_rows,
            "fixture_each": refit_rows,
            "monitor_reports": monitor_reports,
        },
        "components": components,
        # This is one full snapshot closure: primary plus all immutable sidecars.
        # A manifest's job-level export budget accounts for cadence/retention.
        "total_export_bytes": {
            "base_bytes": closure_base,
            "effective_bytes": closure_effective,
        },
    }


def validate_snapshot_byte_bounds(
    value: Mapping[str, Any],
    *,
    worlds: int,
    processes: int,
    rollout_steps: int,
    max_refit_transitions: int,
    max_actor_transitions: int,
    trace_actions: int = 0,
) -> dict[str, Any]:
    """Refuse a supplied P1 control whose byte plan is not the exact formula."""
    expected = derive_snapshot_byte_bounds(
        worlds=worlds,
        processes=processes,
        rollout_steps=rollout_steps,
        max_refit_transitions=max_refit_transitions,
        max_actor_transitions=max_actor_transitions,
        trace_actions=trace_actions,
    )
    if dict(value) != expected:
        raise ValueError("snapshot byte bounds differ from the frozen derived plan")
    return expected


def component_limit(bounds: Mapping[str, Any], name: str) -> int:
    """Return a validated effective payload cap for one exported component."""
    try:
        value = bounds["components"][name]["effective_bytes"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"snapshot byte bounds lack {name}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"invalid snapshot byte bound for {name}")
    return value
