"""Prospective, annotated reduction for the F5 timing HOLD amendment.

This is deliberately separate from :mod:`f5_timing_resume`.  The original
thirteen immutable units, their maxima, kernel strata, and registered HOLD are
evidence, not input that this amendment is allowed to repair.  This reducer
only attaches a source-bound supplement and an operational forecast to those
bytes.  It never dispatches, reads a payload, or changes scientific recipe or
loss identities.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


class AmendedReductionHold(ValueError):
    """Raised when the prospective engineering amendment is not admissible."""


SHA256 = re.compile(r"[0-9a-f]{64}")
ORIGINAL_FIXTURES = tuple(range(672, 684))
WATCH_FIXTURES = frozenset((672, 673, 674))
NON_WATCH_FIXTURES = frozenset(range(675, 684))
WORKLOAD = {"natural": 128, "root": 14_720, "branch_action": 88_320}
WORKERS = 6
ARCHIVE_BASE_HOURS = 96
ARCHIVE_EFFECTIVE_HOURS = 192
FROZEN_EXECUTION_PLAN_SHA256 = "6f3610824dea181dac6d7dee7fda2cfd69282e25e142dfffebc3b17726d0d32b"
FROZEN_NEW_OVERHEAD_PER_SHARD_SECONDS = 1500.0
FROZEN_CARRIED_OVERHEAD_SECONDS = 2239.787154037055
FROZEN_CARRIED_OVERHEAD_PER_SHARD_SECONDS = FROZEN_CARRIED_OVERHEAD_SECONDS / 16
FROZEN_SHARD_OVERHEAD_FLOOR_SECONDS = (
    FROZEN_NEW_OVERHEAD_PER_SHARD_SECONDS + FROZEN_CARRIED_OVERHEAD_PER_SHARD_SECONDS
)
PUBLISHED_TIMING_REDUCTION_SHA256 = (
    "4d5aa3012ddb0002e2faadfea53a57998eba12cd3fa0dbf1a869c410fc26428e"
)
PUBLISHED_TIMING_REDUCTION_CONTENT_SHA256 = (
    "88bf77c20a4aaee3e5351ea2bda14ed87bcdcc86ebf1227cd5d4673fa2a4f99c"
)
REQUIRED_ROLES = (
    "common_natural_h5000",
    "h300_continuation",
    "watch_respawn_14400",
    "long_body_root",
    "long_body_features",
    "long_body_all_legal_branches",
)
H300_STREAMS = frozenset(("watch6", "scripted", "frozen", "mixed"))


def _hold(reason: str) -> None:
    raise AmendedReductionHold("HOLD_AMENDED_REDUCTION: " + reason)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        _hold(reason)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _canonical_sha256(value: Mapping[str, Any], field: str = "content_sha256") -> str:
    """Digest a typed supplement without allowing a self-referential digest."""
    body = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _immutable_reference(value: Any) -> bool:
    """Validate a create-only evidence reference without opening its payload."""
    return (
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256", "size"}
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and not value["path"].startswith("/")
        and ".." not in Path(value["path"]).parts
        and _sha(value.get("sha256"))
        and isinstance(value.get("size"), int)
        and not isinstance(value["size"], bool)
        and value["size"] > 0
    )


def _published_body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"immutable_units", "published_reduction_sha256"}
    }


def _positive(value: Any, *, allow_zero: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value >= 0 if allow_zero else value > 0)
    )


def primitive_definitions() -> dict[str, Any]:
    """Return the frozen prospective primitive-unit contract.

    ``t_branch_bundle`` is the elapsed time for one ``rollout_all`` call.  The
    registered branch workload counts individual legal action continuations,
    so the default remains a conservative full-bundle charge for every action.
    A divisor or per-action conversion needs explicit source-bound review.
    """
    return {
        "t_natural": {
            "meaning": "one complete natural H5000 continuation",
            "workload_count": WORKLOAD["natural"],
        },
        "t_root": {
            "meaning": "one exact root construction and root observation",
            "workload_count": WORKLOAD["root"],
        },
        "t_branch_bundle": {
            "meaning": "one complete rollout_all call over every legal action at H300",
            "measured_unit": "all_legal_actions_bundle",
            "registered_workload_count": WORKLOAD["branch_action"],
            "default_charge": "one full bundle per individual action continuation",
            "default_divisor": 1,
            "automatic_division_by_legal_action_count_forbidden": True,
            "tighter_mapping_requires": "explicit reviewed source-bound primitive mapping",
        },
        "t_branch_individual": {
            "meaning": "one clone plus one legal-action H300 continuation",
            "measured_unit": "max_actual_individual_action_primitive",
            "registered_workload_count": WORKLOAD["branch_action"],
            "clone_seconds_recorded_separately": True,
        },
        "t_branch_conservative": {
            "meaning": "max of actual bundle and individual primitive bounds",
            "charge": "one conservative branch bound per individual action continuation",
            "legal_count_divisor_forbidden": True,
        },
    }


def load_original_reduction(path: Path) -> dict[str, Any]:
    """Load only the published original reduction and attach a content-digest annex.

    The annex is derived from the already published fixture/calibration metadata;
    it does not rewrite the original file or open labels, checkpoints, or units.
    """
    _require(path.is_file() and not path.is_symlink(), "published reduction is absent or unsafe")
    raw = path.read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == PUBLISHED_TIMING_REDUCTION_SHA256,
        "published original reduction bytes differ",
    )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise AmendedReductionHold(
            "HOLD_AMENDED_REDUCTION: published reduction is malformed"
        ) from exc
    _require(isinstance(value, Mapping), "published reduction is malformed")
    copied = copy.deepcopy(dict(value))
    fixtures = copied.get("fixtures")
    calibration = copied.get("calibration")
    _require(isinstance(fixtures, Sequence) and len(fixtures) == 12, "published fixtures differ")
    _require(isinstance(calibration, Mapping), "published calibration differs")
    copied["immutable_units"] = [
        {"key": row.get("allocation_index"), "sha256": _canonical_sha256(row)}
        for row in fixtures
        if isinstance(row, Mapping)
    ] + [{"key": "coupled-cw-calibration", "sha256": _canonical_sha256(calibration)}]
    _require(len(copied["immutable_units"]) == 13, "published unit digest annex differs")
    copied["published_reduction_sha256"] = PUBLISHED_TIMING_REDUCTION_SHA256
    return copied


def _validate_legacy(legacy: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(legacy, Mapping), "legacy reduction is malformed")
    _require(
        legacy.get("published_reduction_sha256") == PUBLISHED_TIMING_REDUCTION_SHA256
        and _canonical_sha256(_published_body(legacy)) == PUBLISHED_TIMING_REDUCTION_CONTENT_SHA256,
        "original reduction content differs from the published HOLD",
    )
    _require(
        legacy.get("status") == "HOLD_REGISTERED_COVERAGE_FAILED",
        "original registered HOLD must be retained",
    )
    fixtures = legacy.get("fixtures")
    _require(
        isinstance(fixtures, Sequence) and not isinstance(fixtures, (str, bytes)), "fixtures absent"
    )
    indices = [row.get("allocation_index") for row in fixtures if isinstance(row, Mapping)]
    _require(indices == list(ORIGINAL_FIXTURES), "original twelve fixture identities differ")
    maxima = legacy.get("max_seconds")
    _require(isinstance(maxima, Mapping), "legacy maxima absent")
    for name in ("t_nat", "t_root", "t_branch", "watch_seconds"):
        _require(_positive(maxima.get(name)), "legacy primitive maxima differ")
    strata = legacy.get("kernel_strata")
    _require(isinstance(strata, Sequence) and bool(strata), "legacy kernel strata absent")
    immutable_units = legacy.get("immutable_units")
    _require(
        isinstance(immutable_units, Sequence) and len(immutable_units) == 13,
        "thirteen immutable unit hashes are required",
    )
    for row in immutable_units:
        _require(
            isinstance(row, Mapping)
            and isinstance(row.get("key"), (int, str))
            and _sha(row.get("sha256")),
            "immutable unit provenance is malformed",
        )
    _require(
        {row["key"] for row in immutable_units} == {*ORIGINAL_FIXTURES, "coupled-cw-calibration"},
        "immutable unit provenance does not cover the original thirteen units",
    )
    expected_annex = [
        {"key": row["allocation_index"], "sha256": _canonical_sha256(row)} for row in fixtures
    ] + [{"key": "coupled-cw-calibration", "sha256": _canonical_sha256(legacy["calibration"])}]
    _require(list(immutable_units) == expected_annex, "immutable original-unit annex differs")
    return copy.deepcopy(dict(legacy))


def _fixture_set(value: Any, role: str) -> set[int]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        role + " fixtures absent",
    )
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        role + " fixture type",
    )
    result = set(value)
    _require(bool(result) and result <= set(ORIGINAL_FIXTURES), role + " uses substituted fixture")
    return result


def _validate_roles(coverage: Mapping[str, Any], legacy: Mapping[str, Any]) -> dict[str, Any]:
    roles = coverage.get("roles")
    _require(
        isinstance(roles, Mapping) and set(roles) == set(REQUIRED_ROLES),
        "coverage role matrix differs",
    )
    copied: dict[str, Any] = {}
    fixtures_by_index = {row["allocation_index"]: row for row in legacy["fixtures"]}
    for role in REQUIRED_ROLES:
        row = roles[role]
        _require(
            isinstance(row, Mapping) and row.get("complete") is True, role + " coverage missing"
        )
        fixtures = _fixture_set(row.get("fixture_indices"), role)
        _require(_sha(row.get("source_review_sha256")), role + " source review is not bound")
        copied[role] = copy.deepcopy(dict(row))
        if role == "common_natural_h5000":
            _require(fixtures <= NON_WATCH_FIXTURES, "natural role cannot claim Watch substitution")
            _require(
                all(
                    fixtures_by_index[index].get("full_h5000_natural") is True for index in fixtures
                ),
                "natural role has no original complete H5000 evidence",
            )
        elif role == "watch_respawn_14400":
            _require(fixtures <= WATCH_FIXTURES, "Watch must be independently measured")
            _require(row.get("frames") == 14_400, "Watch horizon differs")
            _require(
                all(
                    fixtures_by_index[index].get("watch_fixture_frames") == 14_400
                    for index in fixtures
                ),
                "Watch role has no original independent Watch evidence",
            )
        elif role == "h300_continuation":
            _require(row.get("horizon") == 300, "H300 role horizon differs")
            _require(
                all(fixtures_by_index[index].get("full_h300") is True for index in fixtures),
                "H300 role includes incomplete continuation evidence",
            )
            streams = {
                str(fixtures_by_index[index].get("group", "")).split("/", 1)[-1]
                for index in fixtures
            }
            _require(streams == H300_STREAMS, "H300 role must cover every frozen stream")
    return copied


def _validate_supplement(supplement: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    _require(isinstance(supplement, Mapping), "supplement is malformed")
    _require(
        supplement.get("schema") == "bl-f5-targeted-long-body-supplement-draft/v1",
        "supplement schema",
    )
    crossings = supplement.get("records")
    _require(
        isinstance(crossings, Sequence) and len(crossings) == 12,
        "supplement must enumerate twelve fixtures",
    )
    seen: set[int] = set()
    present: set[int] = set()
    seams: set[str] = set()
    complete_seams: set[str] = set()
    complete_pairs: set[tuple[str, str, str]] = set()
    complete_present: set[int] = set()
    kernel_strata: set[tuple[str, str]] = set()
    maxima = {"t_root": 0.0, "t_branch_bundle": 0.0, "t_branch_individual": 0.0}
    for row in crossings:
        _require(isinstance(row, Mapping), "supplement crossing is malformed")
        from research.bl_residual import f5_timing_supplement

        try:
            f5_timing_supplement.validate_operational_record(row)
        except f5_timing_supplement.SupplementHold as exc:
            raise AmendedReductionHold(
                "HOLD_AMENDED_REDUCTION: supplement record is not operationally admissible: "
                + str(exc)
            ) from exc
        _require(_sha(row.get("fixture_descriptor_sha256")), "fixture descriptor is not bound")
        source = row.get("source_provenance")
        kernel = row.get("kernel_provenance")
        _require(
            isinstance(source, Mapping)
            and set(source) == {"source_sha256", "environment_sha256", "native_v6"}
            and _sha(source.get("source_sha256"))
            and _sha(source.get("environment_sha256"))
            and source.get("native_v6") is True,
            "supplement source provenance differs",
        )
        _require(
            isinstance(kernel, Mapping)
            and set(kernel) == {"assignment_raw_sha256", "platform", "release"}
            and _sha(kernel.get("assignment_raw_sha256"))
            and isinstance(kernel.get("platform"), str)
            and isinstance(kernel.get("release"), str),
            "supplement kernel provenance differs",
        )
        _require(
            row.get("operational_payload_sha256")
            == _canonical_sha256(row, "operational_payload_sha256"),
            "supplement operational payload digest differs",
        )
        index = row.get("allocation_index")
        _require(
            isinstance(index, int) and index in ORIGINAL_FIXTURES and index not in seen,
            "crossing fixture differs",
        )
        seen.add(index)
        _require(
            row.get("component_id") == "f5-targeted-long-body-first-crossing-v1"
            and row.get("horizon") == 5000
            and row.get("long_body_minimum") == 700
            and row.get("fit_usable") is False
            and row.get("science_admission") is False,
            "supplement scientific or horizon contract differs",
        )
        crossing = row.get("crossing")
        kernel_strata.add((str(kernel["platform"]), str(kernel["release"])))
        if crossing == "ABSENT_WITHIN_H5000":
            _require(row.get("crossing_decision") is None, "absent crossing is not explicit")
            continue
        _require(crossing == "PRESENT_FIRST_WITHIN_H5000", "crossing enum differs")
        present.add(index)
        seams.add(str(row.get("branch_source_seam")))
        _require(
            isinstance(row.get("crossing_decision"), int) and row["crossing_decision"] >= 0,
            "crossing index is malformed",
        )
        for name in (
            "snapshot_restore_seconds",
            "feature_seconds",
            "setup_seconds",
            "all_actions_seconds",
        ):
            _require(_positive(row.get(name)), "long-body " + name + " missing")
        legal = row.get("legal_action_count")
        _require(isinstance(legal, int) and legal > 0, "long-body legal actions missing")
        actions, units = row.get("legal_actions"), row.get("branch_units")
        all_action_units = row.get("all_actions_units")
        _require(
            isinstance(actions, Sequence)
            and isinstance(units, Sequence)
            and len(actions) == legal
            and len(units) == legal,
            "long-body legal-action inventory differs",
        )
        _require(
            isinstance(all_action_units, Sequence) and len(all_action_units) == legal,
            "all-actions bundle inventory differs",
        )
        _require(
            [unit.get("action") for unit in units if isinstance(unit, Mapping)] == list(actions)
            and [unit.get("action") for unit in all_action_units if isinstance(unit, Mapping)]
            == list(actions),
            "long-body legal-action order differs",
        )
        seen_actions: set[int] = set()
        for unit in units:
            _require(isinstance(unit, Mapping), "long-body branch unit is malformed")
            action = unit.get("action")
            _require(
                isinstance(action, int)
                and action in actions
                and action not in seen_actions
                and _positive(unit.get("seconds"))
                and _positive(unit.get("clone_seconds"))
                and isinstance(unit.get("frames"), int)
                and 1 <= unit["frames"] <= 300
                and unit.get("full_h300") is (unit["frames"] == 300),
                "long-body all-actions coverage missing",
            )
            seen_actions.add(action)
        _require(seen_actions == set(actions), "long-body action order or coverage differs")
        _require(
            all(
                isinstance(unit, Mapping)
                and unit.get("action") in actions
                and isinstance(unit.get("frames"), int)
                and 1 <= unit["frames"] <= 300
                and unit.get("full_h300") is (unit["frames"] == 300)
                for unit in all_action_units
            )
            and _positive(row.get("individual_actions_seconds")),
            "all-actions bundle coverage missing",
        )
        expected_bundle_units = [
            {"action": unit["action"], "frames": unit["frames"], "full_h300": unit["full_h300"]}
            for unit in units
        ]
        _require(
            list(all_action_units) == expected_bundle_units,
            "all-actions bundle and individual continuation frame mappings differ",
        )
        if any(unit["full_h300"] is True for unit in all_action_units):
            complete_present.add(index)
            complete_seams.add(str(row.get("branch_source_seam")))
            complete_pairs.add(
                (
                    str(row.get("branch_source_seam")),
                    str(kernel["platform"]),
                    str(kernel["release"]),
                )
            )
        maxima["t_root"] = max(
            maxima["t_root"],
            float(row["snapshot_restore_seconds"])
            + float(row["feature_seconds"])
            + float(row["setup_seconds"]),
        )
        # Setup belongs to the root construction charge once. The individual
        # primitive is a maximum, never the sum of separately cloned actions;
        # the real bundle remains a conservative per-action bound without a
        # legal-count divisor.
        maxima["t_branch_bundle"] = max(
            maxima["t_branch_bundle"], float(row["all_actions_seconds"])
        )
        maxima["t_branch_individual"] = max(
            maxima["t_branch_individual"],
            max(float(unit["clone_seconds"]) + float(unit["seconds"]) for unit in units),
        )
    maxima["t_branch"] = max(maxima["t_branch_bundle"], maxima["t_branch_individual"])
    _require(seen == set(ORIGINAL_FIXTURES), "supplement crossing enumeration differs")
    _require(maxima["t_root"] > 0 and maxima["t_branch"] > 0, "no frozen fixture reached long body")
    _require(
        seams == {"scripted-batched-one-action", "common-native-serial-one-action"},
        "both measured branch source seams are required",
    )
    _require(complete_seams == seams, "each branch source seam needs a complete H300 continuation")
    _require(
        all(
            (seam, platform, release) in complete_pairs
            for platform, release in kernel_strata
            for seam in seams
        ),
        "an applicable branch source seam and kernel stratum lacks complete H300 coverage",
    )
    copied = copy.deepcopy(dict(supplement))
    copied["_validated_present_indices"] = sorted(present)
    copied["_validated_complete_present_indices"] = sorted(complete_present)
    copied["_validated_kernel_strata"] = sorted(kernel_strata)
    return copied, maxima


def _validate_long_body_roles(
    roles: Mapping[str, Any], supplement: Mapping[str, Any], legacy: Mapping[str, Any]
) -> None:
    present = set(supplement["_validated_present_indices"])
    complete_present = set(supplement["_validated_complete_present_indices"])
    for role in ("long_body_root", "long_body_features"):
        claimed = _fixture_set(roles[role]["fixture_indices"], role)
        _require(claimed <= present, role + " claims an absent long-body crossing")
    claimed = _fixture_set(
        roles["long_body_all_legal_branches"]["fixture_indices"], "long_body_all_legal_branches"
    )
    _require(
        claimed <= complete_present,
        "long_body_all_legal_branches claims a crossing without complete H300 evidence",
    )
    expected_strata = {
        (str(row.get("platform")), str(row.get("release")))
        for row in legacy["kernel_strata"]
        if isinstance(row, Mapping)
    }
    _require(
        expected_strata <= {tuple(row) for row in supplement["_validated_kernel_strata"]},
        "supplement lacks an applicable observed kernel stratum",
    )


def _forecast(
    legacy: Mapping[str, Any], forecast: Mapping[str, Any], supplement_maxima: Mapping[str, float]
) -> dict[str, Any]:
    _require(isinstance(forecast, Mapping), "forecast is malformed")
    mapping = forecast.get("branch_mapping")
    _require(isinstance(mapping, Mapping), "branch mapping is absent")
    _require(
        mapping.get("kind") == "conservative_full_bundle_per_action",
        "unreviewed branch unit correction",
    )
    _require(mapping.get("divisor") == 1, "automatic legal-action division is forbidden")
    overhead = forecast.get("new_overhead_seconds")
    required_overhead = {"setup", "qualification", "plan_cache", "export_io", "imbalance"}
    _require(
        isinstance(overhead, Mapping) and set(overhead) == required_overhead, "new overhead differs"
    )
    _require(
        all(_positive(value, allow_zero=True) for value in overhead.values()),
        "new overhead malformed",
    )
    historical = forecast.get("forecast_charged_historical_seconds")
    _require(
        historical == {"current_series": 2239.787154037055}, "current series accounting differs"
    )
    exposure = forecast.get("separate_historical_exposure_seconds")
    _require(
        exposure
        == {
            "prior_timing": 6811.840706833173,
            "current_timing": 4090.2527135000564,
            "canary": 3849.601884124102,
            "known_total": 14751.695304457331,
        },
        "historical exposure accounting differs",
    )
    maxima = dict(legacy["max_seconds"])
    maxima["t_root"] = max(float(maxima["t_root"]), float(supplement_maxima["t_root"]))
    maxima["t_branch"] = max(float(maxima["t_branch"]), float(supplement_maxima["t_branch"]))
    compute = (
        WORKLOAD["natural"] * float(maxima["t_nat"])
        + WORKLOAD["root"] * float(maxima["t_root"])
        + WORKLOAD["branch_action"] * float(maxima["t_branch"])
    ) / WORKERS
    total = (
        compute
        + sum(float(value) for value in historical.values())
        + sum(float(value) for value in overhead.values())
    )
    _require(total <= ARCHIVE_BASE_HOURS * 3600, "amended 96-hour base guard exceeded")
    return {
        "equation": (
            "(128*t_nat + 14720*t_root + 88320*max(t_branch_bundle,t_branch_individual))/6 "
            "+ historical_liabilities + new_overhead"
        ),
        "workload": dict(WORKLOAD),
        "workers": WORKERS,
        "compute_seconds": compute,
        "amended_max_seconds": maxima,
        "long_body_primitive_bounds": {
            "t_branch_bundle": float(supplement_maxima["t_branch_bundle"]),
            "t_branch_individual": float(supplement_maxima["t_branch_individual"]),
            "t_branch_conservative": float(supplement_maxima["t_branch"]),
        },
        "forecast_charged_historical_seconds": copy.deepcopy(dict(historical)),
        "separate_historical_exposure_seconds": copy.deepcopy(dict(exposure)),
        "new_overhead_seconds": copy.deepcopy(dict(overhead)),
        "total_seconds": total,
        "base_guard_hours": ARCHIVE_BASE_HOURS,
        "effective_guard_hours": ARCHIVE_EFFECTIVE_HOURS,
        "effective_is_doubled_once": True,
    }


def _validate_schedule(schedule: Mapping[str, Any], maxima: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(schedule, Mapping), "schedule is malformed")
    _require(
        schedule.get("frozen_execution_plan_sha256") == FROZEN_EXECUTION_PLAN_SHA256,
        "schedule does not bind the frozen execution-plan identity",
    )
    _require(schedule.get("l4_assignments_at_once") == 1, "multi-L4 placement is not admitted")
    _require(schedule.get("host_workers") == WORKERS, "host worker count differs")
    _require(schedule.get("math_threads") == 1, "math thread count differs")
    _require(schedule.get("global_max_completion_rounds") == 2, "completion retry scope differs")
    _require(
        schedule.get("runtime_constraints_status") == "PENDING_QUALIFICATION",
        "runtime admission must remain pending",
    )
    shards = schedule.get("shards")
    _require(
        isinstance(shards, Sequence) and len(shards) == 16, "requires sixteen finite planned shards"
    )
    totals = {"natural": 0, "root": 0, "branch_action": 0}
    frozen_shard_compute_seconds = (
        8 * float(maxima["t_nat"])
        + 920 * float(maxima["t_root"])
        + 5520 * float(maxima["t_branch"])
    ) / WORKERS
    for ordinal, shard in enumerate(shards):
        _require(isinstance(shard, Mapping), "shard is malformed")
        _require(shard.get("ordinal") == ordinal, "shard ordering differs")
        _require(
            shard.get("natural") == 8
            and shard.get("root") == 920
            and shard.get("branch_action") == 5520,
            "shard workload differs",
        )
        _require(
            shard.get("natural_range") == [ordinal * 8, ordinal * 8 + 8]
            and shard.get("root_range") == [ordinal * 920, ordinal * 920 + 920]
            and shard.get("branch_action_range") == [ordinal * 5520, ordinal * 5520 + 5520],
            "shard ranges are not contiguous deterministic partitions",
        )
        _require(shard.get("base_hours") == 6, "shard base envelope differs")
        _require(shard.get("effective_hours") == 12, "shard guard not doubled once")
        overhead = shard.get("reserved_setup_export_imbalance_seconds")
        _require(
            _positive(overhead)
            and math.isclose(
                float(overhead), FROZEN_SHARD_OVERHEAD_FLOOR_SECONDS, rel_tol=0, abs_tol=1e-9
            ),
            "shard reserved overhead differs from frozen reservation floor",
        )
        _require(
            frozen_shard_compute_seconds + float(overhead) <= 6 * 3600,
            "shard maximum compute and reserved overhead exceed base envelope",
        )
        for key in totals:
            totals[key] += int(shard[key])
    _require(totals == WORKLOAD, "scheduled shard total differs")
    frozen = copy.deepcopy(dict(schedule))
    frozen["max_shard_compute_seconds"] = frozen_shard_compute_seconds
    frozen["max_shard_base_seconds"] = 6 * 3600
    return frozen


def _validate_attributable_overheads(
    schedule: Mapping[str, Any], forecast: Mapping[str, Any]
) -> list[float]:
    """Require complete per-shard attribution while charging the old series once."""
    _require(
        schedule.get("global_once_overhead_seconds") == FROZEN_CARRIED_OVERHEAD_SECONDS,
        "global once-only carried overhead differs",
    )
    named = forecast["new_overhead_seconds"]
    per_shard: list[float] = []
    totals = {key: 0.0 for key in named}
    for shard in schedule["shards"]:
        attributed = shard.get("attributable_new_overhead_seconds")
        _require(
            isinstance(attributed, Mapping) and set(attributed) == set(named),
            "shard overhead attribution is incomplete or unknown",
        )
        _require(
            all(_positive(value, allow_zero=True) for value in attributed.values()),
            "shard overhead attribution is malformed",
        )
        evidence = shard.get("attributable_overhead_evidence")
        _require(
            isinstance(evidence, Mapping)
            and set(evidence)
            == {
                "status",
                "reference",
                "shard_ordinal",
                "source_identity_sha256",
                "attributed_components_sha256",
            }
            and evidence.get("status") == "QUALIFIED"
            and _immutable_reference(evidence.get("reference"))
            and evidence.get("shard_ordinal") == shard.get("ordinal")
            and evidence.get("source_identity_sha256") == FROZEN_EXECUTION_PLAN_SHA256
            and evidence.get("attributed_components_sha256") == _canonical_sha256(attributed),
            "shard overhead evidence is absent, unqualified, or not bound to its components",
        )
        amount = sum(float(value) for value in attributed.values())
        per_shard.append(amount)
        for key, value in attributed.items():
            totals[key] += float(value)
    _require(
        all(
            math.isclose(totals[key], float(value), rel_tol=0, abs_tol=1e-9)
            for key, value in named.items()
        ),
        "per-shard overhead attribution does not reconcile to the global forecast",
    )
    return per_shard


def reduce_amended(
    legacy: Mapping[str, Any],
    *,
    coverage: Mapping[str, Any],
    supplement: Mapping[str, Any],
    forecast: Mapping[str, Any],
    schedule: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce an annotated prospective reduction without changing original evidence.

    The result remains a HOLD until a separately sealed supplement is supplied;
    callers must not treat a valid amendment as ``ACCEPT_F5_FIT``.
    """
    old = _validate_legacy(legacy)
    _require(isinstance(provenance, Mapping), "amendment provenance is malformed")
    for name in (
        "ruling_sha256",
        "original_reduction_sha256",
        "amendment_sha256",
        "expected_supplement_source_sha256",
        "expected_supplement_environment_sha256",
        "expected_fixture_descriptor_sha256",
    ):
        _require(_sha(provenance.get(name)), "amendment provenance " + name + " is not bound")
    _require(
        provenance["original_reduction_sha256"] == PUBLISHED_TIMING_REDUCTION_SHA256,
        "amendment provenance does not bind the original published reduction",
    )
    roles = _validate_roles(coverage, old)
    _require(
        provenance.get("supplement_content_sha256") == _canonical_sha256(supplement),
        "supplement content provenance differs",
    )
    sealed_supplement, supplement_maxima = _validate_supplement(supplement)
    for record in sealed_supplement["records"]:
        _require(
            record["source_provenance"]["source_sha256"]
            == provenance["expected_supplement_source_sha256"]
            and record["source_provenance"]["environment_sha256"]
            == provenance["expected_supplement_environment_sha256"]
            and record["fixture_descriptor_sha256"]
            == provenance["expected_fixture_descriptor_sha256"],
            "supplement source, environment, or fixture descriptor differs",
        )
    _validate_long_body_roles(roles, sealed_supplement, old)
    amended_forecast = _forecast(old, forecast, supplement_maxima)
    frozen_schedule = _validate_schedule(schedule, amended_forecast["amended_max_seconds"])
    per_shard_new = _validate_attributable_overheads(frozen_schedule, amended_forecast)
    charged_old = float(amended_forecast["forecast_charged_historical_seconds"]["current_series"])
    # A reservation is retained independently for every shard.  Charging a
    # global maximum would let unused floor on one shard hide an overrun on
    # another.  The carried series exposure is global and is charged exactly
    # once; each new component is therefore the per-shard maximum after its
    # carried share is removed.
    reserved_overhead = sum(
        max(
            float(shard["reserved_setup_export_imbalance_seconds"]),
            FROZEN_CARRIED_OVERHEAD_PER_SHARD_SECONDS + attributable_new,
        )
        for shard, attributable_new in zip(frozen_schedule["shards"], per_shard_new)
    )
    charged_new = reserved_overhead - charged_old
    amended_forecast["reserved_shard_overhead_seconds"] = reserved_overhead
    amended_forecast["charged_new_overhead_seconds"] = charged_new
    amended_forecast["total_seconds"] = (
        float(amended_forecast["compute_seconds"]) + charged_old + charged_new
    )
    for shard, attributable_new in zip(frozen_schedule["shards"], per_shard_new):
        _require(
            float(frozen_schedule["max_shard_compute_seconds"])
            + max(
                float(shard["reserved_setup_export_imbalance_seconds"]),
                FROZEN_CARRIED_OVERHEAD_PER_SHARD_SECONDS + attributable_new,
            )
            <= 6 * 3600,
            "shard measured or reserved overhead exceeds base envelope",
        )
    _require(
        amended_forecast["total_seconds"] <= ARCHIVE_BASE_HOURS * 3600,
        "amended 96-hour base guard exceeded after shard reservations",
    )
    return {
        "schema": "bl-f5-amended-reduction/v1",
        "status": "HOLD_PENDING_INDEPENDENT_TARGETED_TIMING_ACCEPTANCE",
        "scientific_recipe_or_loss_changed": False,
        "original_reduction": old,
        "original_hold_retained": True,
        "coverage_matrix": roles,
        "targeted_supplement": {
            key: value
            for key, value in sealed_supplement.items()
            if not key.startswith("_validated_")
        },
        "primitive_definitions": primitive_definitions(),
        "archive_forecast": amended_forecast,
        "scheduled_execution": frozen_schedule,
        "provenance": copy.deepcopy(dict(provenance)),
        "acceptance_authorized": False,
    }
