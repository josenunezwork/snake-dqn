"""Draft-only, finite long-body timing supplement for the F5 HOLD route.

This module does not dispatch work, read selected-training data, create a strict
ledger, or make a scientific claim.  It defines the only prospective long-body
workload Astra permitted: for each of the frozen timing fixtures, replay the
ordinary natural trajectory through H5000 and, if it occurs, measure the first
length-700 decision state.  A missing crossing is an enumerated result, never a
reason to substitute a fixture, seed, horizon, or retry.

The actual CUDA/native-v6 admission, parity, immutable export, and independent
acceptance remain a later sealed package.  Records produced by these helpers are
operational timing observations only and deliberately omit outcomes and labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from research.bl_residual import f5_qualification, f5_r10a_annex, f5_timing

SCHEMA = "bl-f5-targeted-long-body-supplement-draft/v1"
COMPONENT_ID = "f5-targeted-long-body-first-crossing-v1"
LONG_BODY_MINIMUM = f5_qualification.LONG_BODY_MINIMUM
NATURAL_HORIZON = f5_qualification.NATURAL_HORIZON
BRANCH_HORIZON = f5_qualification.BRANCH_HORIZON
ACTION_COUNT = f5_qualification.ACTION_COUNT


class SupplementHold(ValueError):
    """Refusal for a proposed timing supplement without scientific meaning."""


def _hold(reason: str) -> None:
    raise SupplementHold("HOLD_TARGETED_TIMING_SUPPLEMENT: " + reason)


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def operational_payload_sha256(record: Mapping[str, Any]) -> str:
    """Hash a redacted record without making it a sealed publication artifact."""
    payload = {key: value for key, value in record.items() if key != "operational_payload_sha256"}
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        _hold("record does not have a canonical operational payload")
        raise AssertionError("unreachable") from exc
    return hashlib.sha256(encoded).hexdigest()


def frozen_fixtures(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and return exactly the original twelve deterministic fixtures."""
    try:
        canonical = f5_timing._timing_parents(rows)
    except f5_timing.TimingHold as exc:
        _hold(str(exc))
    expected = {
        (row["allocation_index"], row["parent_id"]): row["seed"]
        for row in f5_r10a_annex.allocate(0)
        if row["timing_only"] is True
    }
    allowed = {"allocation_index", "group", "group_index", "parent_id", "timing_only", "seed"}
    result = []
    for row in canonical:
        if set(row) != allowed or row.get("timing_only") is not True:
            _hold("fixture identity contains non-frozen fields")
        key = (row["allocation_index"], row["parent_id"])
        if expected.get(key) != row.get("seed"):
            _hold("fixture identity differs from the frozen allocation")
        result.append(dict(row))
    return result


def draft_plan(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create the finite, unsealed workload descriptor without a runtime claim."""
    fixtures = frozen_fixtures(rows)
    return {
        "schema": SCHEMA,
        "status": "DRAFT_UNSEALED_HOLD",
        "component_id": COMPONENT_ID,
        "science_admission": False,
        "dispatch_authorized": False,
        "fixtures": fixtures,
        "construction": {
            "kind": "FIRST_LENGTH_CROSSING_WITHIN_FROZEN_NATURAL_HORIZON",
            "length": LONG_BODY_MINIMUM,
            "horizon": NATURAL_HORIZON,
            "enumerate_absent_crossings": True,
            "no_fixture_substitution": True,
            "no_new_seed": True,
            "no_horizon_extension": True,
            "no_crossing_search_retry": True,
        },
        "primitive_units": {
            "snapshot_restore": "one exact clone/restore digest seam at each present crossing",
            "feature": "one bv8 feature build from the restored crossing state",
            "legal_setup": "one exact decision setup; report root length and legal count",
            "branch": "one H300 rollout_all invocation per legal first action; no divide-by-six",
        },
        "source_seams": {
            "scripted": "existing scripted one-action batched rollout_all path, separately bounded",
            "watch6": "existing nonscripted native rollout_all path",
            "frozen": "existing nonscripted native rollout_all path",
            "mixed": "existing nonscripted native rollout_all path",
        },
        "export_contract": {
            "operational_only": True,
            "forbid_fields": ["loss", "label", "outcome", "science_result"],
            "distinct_component_identity_required": True,
            "strict_ledger_mixing_forbidden": True,
        },
        "acceptance": {
            "requires_genuine_cuda_parity": True,
            "requires_native_v6_environment": True,
            "requires_all_twelve_enumerated": True,
            "requires_each_present_crossing_to_measure_all_legal_branches": True,
            "missing_required_coverage_keeps_hold": True,
            "independent_acceptance_required": True,
        },
    }


def crossing_from_trace(trace: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the first recorded natural decision at length >=700, if any.

    This pure helper makes the selection rule auditable.  Runtime construction
    replays the same fixed natural path and takes a state clone at that decision;
    it never searches beyond the registered H5000 prefix.
    """
    prior = -1
    for item in trace:
        decision, length = item.get("decision"), item.get("hero_length")
        if type(decision) is not int or decision != prior + 1:
            _hold("natural trace is not a contiguous deterministic decision prefix")
        if type(length) is not int or length < 0:
            _hold("natural trace lacks an integer hero length")
        if decision >= NATURAL_HORIZON:
            _hold("natural trace extends beyond the frozen H5000 horizon")
        if length >= LONG_BODY_MINIMUM:
            return decision
        prior = decision
    return None


def _crossing_root(
    row: Mapping[str, Any], checkpoint: Any
) -> tuple[Any | None, int | None, int | None]:
    """Replay a frozen natural prefix and clone its first eligible exact state."""
    from research.bl_residual import gameplay
    from research.t34_shadow.shadow import clone_world_exact, full_digest

    root, _, _ = f5_qualification._source_root_observation(row, checkpoint)
    world = clone_world_exact(root)
    world.hero = copy.copy(root.hero)
    for decision in range(NATURAL_HORIZON):
        captured: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def capture_predecision(prepared: Any) -> None:
            # ``_step_with_trace`` owns the exact natural decision construction.
            # Capture only after its ordinary runtime preparation and before its
            # fixed raster/Q/dispatch sequence, which is the trace's reported
            # hero-length boundary rather than a pre-prepare approximation.
            original_prepare(prepared)
            if (
                "root" not in captured
                and bool(prepared.alive[0, 0])
                and int(prepared.length[0, 0]) >= LONG_BODY_MINIMUM
            ):
                # CapturedWorld.clone is the registered selector-prepared seam:
                # it retains the policy lifecycle marker that a raw exact clone
                # from inside step_with_policy would otherwise normalize away.
                prepared_snapshot = world.clone()
                exact_snapshot = clone_world_exact(prepared_snapshot)
                if full_digest(prepared_snapshot) != full_digest(exact_snapshot):
                    _hold("prepared crossing snapshot/restore digest differs")
                captured["root"] = exact_snapshot
                captured["length"] = int(prepared.length[0, 0])

        world.runtime.prepare = capture_predecision
        try:
            selected = gameplay._step_with_trace(world, decision)
        finally:
            world.runtime.prepare = original_prepare
        if "root" in captured:
            if selected.get("hero_length") != captured["length"]:
                _hold("prepared crossing root disagrees with the natural trace length")
            return captured["root"], decision, captured["length"]
        if not bool(world.sim.alive[0, 0]):
            return None, None, None
    return None, None, None


def _crossing_primitives(root: Any) -> dict[str, Any]:
    """Measure exact crossing seams with one independently attributable branch unit."""
    from research.t3_rollout_labels_20261015 import bank_engine
    from research.t34_shadow.shadow import clone_world_exact, full_digest
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    snapshot_started = time.monotonic()
    root_digest = full_digest(root)
    snapshot = clone_world_exact(root)
    restored = clone_world_exact(snapshot)
    if root_digest != full_digest(snapshot) or root_digest != full_digest(restored):
        _hold("long-body snapshot/restore digest differs")
    snapshot_seconds = time.monotonic() - snapshot_started

    feature_started = time.monotonic()
    observation = build_ego_raster(
        restored.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
    )
    if observation["local"].shape != (1, 7, 31, 31):
        _hold("long-body crossing did not produce the production bv8 feature shape")
    feature_seconds = time.monotonic() - feature_started

    setup_started = time.monotonic()
    branch_root = clone_world_exact(restored)
    root_length_before_setup = int(branch_root.sim.length[0, 0])
    # The crossing clone is captured within ``_step_with_trace`` after its
    # runtime.prepare call and before dispatch.  Read the same advisory mask
    # directly here: calling choose_and_step would consume a natural action and
    # turn the supplement root into a different state before branching.
    advisory = branch_root.sim.get_advisory_action_mask()[0, 0]
    legal = [int(action) for action in np.flatnonzero(np.asarray(advisory, dtype=bool))]
    if not legal or any(action < 0 or action >= ACTION_COUNT for action in legal):
        _hold("long-body crossing has no legal six-action branch setup")
    setup_seconds = time.monotonic() - setup_started

    bundle_started = time.monotonic()
    bundle = bank_engine.rollout_all(
        clone_world_exact(branch_root), legal, depth=BRANCH_HORIZON, record_outcomes=False
    )
    if set(bundle) != set(legal):
        _hold("all-actions branch bundle returned an unexpected action set")
    all_actions_seconds = time.monotonic() - bundle_started
    bundle_units = []
    for action in legal:
        frames = bundle[action].get("frames_executed")
        if type(frames) is not int or frames < 1 or frames > BRANCH_HORIZON:
            _hold("all-actions branch bundle has invalid frame accounting")
        bundle_units.append(
            {"action": action, "frames": frames, "full_h300": frames == BRANCH_HORIZON}
        )

    branches = []
    for action in legal:
        # Clone construction is an independently attributable root/setup cost;
        # measure it separately rather than hiding it in the rollout duration.
        clone_started = time.monotonic()
        individual_root = clone_world_exact(branch_root)
        clone_seconds = time.monotonic() - clone_started
        started = time.monotonic()
        # Passing one action deliberately records a primitive action continuation.
        # The scripted engine's one-action batch and the nonscripted native branch
        # both remain their ordinary frozen rollout_all implementations.
        result = bank_engine.rollout_all(
            individual_root, [action], depth=BRANCH_HORIZON, record_outcomes=False
        )
        if set(result) != {action}:
            _hold("one-action branch primitive returned an unexpected action set")
        frames = result[action].get("frames_executed")
        if type(frames) is not int or frames < 1 or frames > BRANCH_HORIZON:
            _hold("one-action branch primitive has invalid frame accounting")
        branches.append(
            {
                "action": action,
                "seconds": time.monotonic() - started,
                "clone_seconds": clone_seconds,
                "frames": frames,
                "full_h300": frames == BRANCH_HORIZON,
            }
        )
    expected_bundle_units = [
        {"action": unit["action"], "frames": unit["frames"], "full_h300": unit["full_h300"]}
        for unit in branches
    ]
    if bundle_units != expected_bundle_units:
        _hold("all-actions bundle and individual continuation frame mappings differ")
    return {
        "snapshot_restore_seconds": snapshot_seconds,
        "feature_seconds": feature_seconds,
        "feature_shape": [1, 7, 31, 31],
        "setup_seconds": setup_seconds,
        "root_length": int(restored.sim.length[0, 0]),
        "root_length_before_setup": root_length_before_setup,
        "root_length_after_setup": int(branch_root.sim.length[0, 0]),
        "legal_action_count": len(legal),
        "legal_actions": legal,
        "branch_units": branches,
        "all_actions_seconds": all_actions_seconds,
        "all_actions_units": bundle_units,
        "individual_actions_seconds": sum(
            item["clone_seconds"] + item["seconds"] for item in branches
        ),
    }


def measure_fixture(
    row: Mapping[str, Any],
    checkpoint: Any,
    *,
    root_builder: Callable[[Mapping[str, Any], Any], Any] | None = None,
) -> dict[str, Any]:
    """Measure one finite crossing workload, or enumerate its absence.

    The optional root builder exists only for isolated unit tests.  Production
    callers use the common source-root seam above and must add CUDA/v6 admission
    outside this draft module before accepting any record.
    """
    # A single call cannot itself prove that its future package retains all 12.
    # It can prove that this unit is one of the original identities; package-level
    # admission must call ``frozen_fixtures`` over the whole descriptor.
    canonical = {item["allocation_index"]: item for item in f5_timing._canonical_timing_parents()}
    expected = canonical.get(row.get("allocation_index"))
    expected_seeds = {
        item["allocation_index"]: item["seed"]
        for item in f5_r10a_annex.allocate(0)
        if item["timing_only"] is True
    }
    if (
        expected is None
        or any(row.get(key) != value for key, value in expected.items())
        or row.get("seed") != expected_seeds.get(row.get("allocation_index"))
    ):
        _hold("fixture is outside the frozen timing allocation")
    crossing = (
        _crossing_root(row, checkpoint) if root_builder is None else root_builder(row, checkpoint)
    )
    root, decision, length = crossing
    base = {
        "schema": SCHEMA,
        "component_id": COMPONENT_ID,
        "component_key": f"{COMPONENT_ID}/{row['allocation_index']}",
        "allocation_index": row["allocation_index"],
        "parent_id": row["parent_id"],
        "stream": str(row["group"]).split("/", 1)[1],
        "branch_source_seam": (
            "scripted-batched-one-action"
            if row["group"] == "timing/scripted"
            else "common-native-serial-one-action"
        ),
        "horizon": NATURAL_HORIZON,
        "long_body_minimum": LONG_BODY_MINIMUM,
        "fit_usable": False,
        "science_admission": False,
    }
    if root is None:
        return {**base, "crossing": "ABSENT_WITHIN_H5000", "crossing_decision": None}
    if type(decision) is not int or type(length) is not int or length < LONG_BODY_MINIMUM:
        _hold("crossing builder returned an invalid first-long-body root")
    return {
        **base,
        "crossing": "PRESENT_FIRST_WITHIN_H5000",
        "crossing_decision": decision,
        "crossing_length": length,
        **_crossing_primitives(root),
    }


def validate_operational_record(record: Mapping[str, Any]) -> None:
    """Reject records that could silently become a science or strict-ledger input."""
    common = {
        "schema",
        "component_id",
        "component_key",
        "allocation_index",
        "parent_id",
        "stream",
        "branch_source_seam",
        "horizon",
        "long_body_minimum",
        "fit_usable",
        "science_admission",
        "fixture_descriptor_sha256",
        "source_provenance",
        "kernel_provenance",
        "crossing",
        "crossing_decision",
        "operational_payload_sha256",
    }
    present = common | {
        "crossing_length",
        "snapshot_restore_seconds",
        "feature_seconds",
        "feature_shape",
        "setup_seconds",
        "root_length",
        "root_length_before_setup",
        "root_length_after_setup",
        "legal_action_count",
        "legal_actions",
        "branch_units",
        "all_actions_seconds",
        "all_actions_units",
        "individual_actions_seconds",
    }
    if record.get("schema") != SCHEMA or record.get("component_id") != COMPONENT_ID:
        _hold("foreign supplement component")
    forbidden = {"loss", "label", "outcome", "science_result", "strict_ledger"}
    if forbidden & set(record):
        _hold("supplement record contains prohibited science or ledger fields")
    if record.get("fit_usable") is not False or record.get("science_admission") is not False:
        _hold("supplement record claims scientific usability")
    if record.get("operational_payload_sha256") != operational_payload_sha256(record):
        _hold("operational payload digest differs")
    if not _sha256(record.get("fixture_descriptor_sha256")):
        _hold("fixture descriptor digest is absent")
    source = record.get("source_provenance")
    kernel = record.get("kernel_provenance")
    if not isinstance(source, Mapping) or set(source) != {
        "source_sha256",
        "environment_sha256",
        "native_v6",
    }:
        _hold("source provenance is incomplete")
    if not _sha256(source.get("source_sha256")) or not _sha256(source.get("environment_sha256")):
        _hold("source provenance digest is invalid")
    if source.get("native_v6") is not True:
        _hold("source provenance is not native-v6")
    if not isinstance(kernel, Mapping) or set(kernel) != {
        "assignment_raw_sha256",
        "platform",
        "release",
    }:
        _hold("kernel provenance is incomplete")
    if not _sha256(kernel.get("assignment_raw_sha256")) or not all(
        isinstance(kernel.get(key), str) and kernel[key] for key in ("platform", "release")
    ):
        _hold("kernel provenance is invalid")
    if (
        record.get("horizon") != NATURAL_HORIZON
        or record.get("long_body_minimum") != LONG_BODY_MINIMUM
        or record.get("component_key") != f"{COMPONENT_ID}/{record.get('allocation_index')}"
    ):
        _hold("record identity or frozen horizon differs")
    expected = {
        item["allocation_index"]: item for item in f5_r10a_annex.allocate(0) if item["timing_only"]
    }
    if type(record.get("allocation_index")) is not int:
        _hold("record allocation index is not an integer")
    fixture = expected.get(record.get("allocation_index"))
    if fixture is None or record.get("parent_id") != fixture["parent_id"]:
        _hold("record parent is outside the frozen allocation")
    stream = fixture["group"].split("/", 1)[1]
    if record.get("stream") != stream or record.get("branch_source_seam") != (
        "scripted-batched-one-action" if stream == "scripted" else "common-native-serial-one-action"
    ):
        _hold("record source seam differs from its fixture role")
    if record.get("crossing") == "ABSENT_WITHIN_H5000":
        if set(record) != common or record.get("crossing_decision") is not None:
            _hold("absent crossing has a decision")
        return
    if record.get("crossing") != "PRESENT_FIRST_WITHIN_H5000":
        _hold("unknown crossing disposition")
    if set(record) != present:
        _hold("present crossing record has unknown or missing fields")
    if (
        type(record.get("crossing_decision")) is not int
        or not 0 <= record["crossing_decision"] < NATURAL_HORIZON
        or type(record.get("crossing_length")) is not int
        or record["crossing_length"] < LONG_BODY_MINIMUM
        or type(record.get("root_length")) is not int
        or record["root_length"] < LONG_BODY_MINIMUM
    ):
        _hold("present crossing root is invalid")
    if record.get("feature_shape") != [1, 7, 31, 31] or any(
        type(record.get(field)) is not int or record[field] != record["crossing_length"]
        for field in ("root_length", "root_length_before_setup", "root_length_after_setup")
    ):
        _hold("feature shape or prepared root length changed during setup")
    if not isinstance(record.get("legal_actions"), list) or not record["legal_actions"]:
        _hold("present crossing lacks legal action mapping")
    if (
        len(record["legal_actions"]) != len(set(record["legal_actions"]))
        or any(
            type(action) is not int or action < 0 or action >= ACTION_COUNT
            for action in record["legal_actions"]
        )
        or type(record.get("legal_action_count")) is not int
        or record.get("legal_action_count") != len(record["legal_actions"])
    ):
        _hold("legal-action mapping is invalid")
    units = record.get("branch_units")
    bundle_units = record.get("all_actions_units")
    if (
        not isinstance(units, list)
        or [item.get("action") for item in units] != record["legal_actions"]
        or not isinstance(bundle_units, list)
        or [item.get("action") for item in bundle_units] != record["legal_actions"]
    ):
        _hold("branch units do not map one-to-one to legal actions")
    for field in (
        "snapshot_restore_seconds",
        "feature_seconds",
        "setup_seconds",
        "all_actions_seconds",
        "individual_actions_seconds",
    ):
        if not _finite_positive(record.get(field)):
            _hold("primitive timing is not finite")
    for unit in units:
        if set(unit) != {"action", "seconds", "clone_seconds", "frames", "full_h300"}:
            _hold("individual branch has unknown fields")
        if not (
            _finite_positive(unit.get("seconds")) and _finite_positive(unit.get("clone_seconds"))
        ):
            _hold("branch timing is not finite")
        if type(unit.get("frames")) is not int or not 1 <= unit["frames"] <= BRANCH_HORIZON:
            _hold("branch frame accounting is invalid")
        if type(unit.get("full_h300")) is not bool:
            _hold("branch continuation status is absent")
        if unit["full_h300"] != (unit["frames"] == BRANCH_HORIZON):
            _hold("branch continuation status differs from its frame count")
    for unit in bundle_units:
        if set(unit) != {"action", "frames", "full_h300"}:
            _hold("all-actions bundle has unknown fields")
        if type(unit.get("frames")) is not int or not 1 <= unit["frames"] <= BRANCH_HORIZON:
            _hold("all-actions bundle frame accounting is invalid")
        if type(unit.get("full_h300")) is not bool or unit["full_h300"] != (
            unit["frames"] == BRANCH_HORIZON
        ):
            _hold("all-actions bundle continuation status differs from frames")
    expected_bundle_units = [
        {"action": unit["action"], "frames": unit["frames"], "full_h300": unit["full_h300"]}
        for unit in units
    ]
    if bundle_units != expected_bundle_units:
        _hold("all-actions bundle and individual continuation frame mappings differ")
    if not math.isclose(
        record["individual_actions_seconds"],
        sum(unit["clone_seconds"] + unit["seconds"] for unit in units),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        _hold("individual action timing sum differs")
