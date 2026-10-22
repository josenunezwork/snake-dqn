"""Frozen descriptive and rejection-rule summaries for the density screen."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from research.bl_density.statistics import compare, ks_compare, length_summary

ARMS = ("A", "B70m", "B70p", "C-P1")
MIXES = ("scripted", "frozen", "mixed")
_GEOMETRY_FIELDS = (
    "length",
    "total_occupied_cells",
    "hero_occupancy",
    "nearest_opponent_head_manhattan",
    "nearest_opponent_body_manhattan",
    "wall_distance_ahead",
    "snake_length_0",
    "snake_length_1",
    "snake_length_2",
    "snake_length_3",
    "snake_length_4",
    "snake_length_5",
)


def _ratio(numerator: int | float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _sampled(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for record in records for row in record.get("sampled", [])]


def _doom_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**row, "episode_id": record["episode_id"]}
        for record in records
        for row in record.get("sampled", [])
        if row.get("prefilter") and row.get("h300_doom") is True
    ]


def _assessed_avoidable_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Do not recode roots after the four-root cap as non-avoidable."""
    return [row for row in _doom_rows(records) if isinstance(row.get("witnessed_avoidable"), bool)]


def _record_cpu(record: dict[str, Any]) -> float:
    try:
        value = float(record.get("cpu_seconds", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) and value >= 0 else 0.0


def _recall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if isinstance(row.get("natural_doom"), bool)]
    true = [row for row in known if row["natural_doom"]]
    captured = sum(bool(row.get("prefilter")) for row in true)
    return {
        "status": "OK" if true else "INSUFFICIENT",
        "eligible_roots": len(rows),
        "known_labels": len(known),
        "censored_labels": len(rows) - len(known),
        "true_dooms": len(true),
        "captured_true_dooms": captured,
        "rejected_true_dooms": len(true) - captured,
        "recall": _ratio(captured, len(true)),
    }


def _cell(
    records: list[dict[str, Any]],
    attributed_cpu: float,
    attempted_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    attempted_records = records if attempted_records is None else attempted_records
    rows, doom = _sampled(records), _doom_rows(records)
    assessed = _assessed_avoidable_rows(records)
    avoidable = [row for row in assessed if row["witnessed_avoidable"]]
    decisions = sum(int(record.get("decisions", 0)) for record in records)
    process_cpu = sum(_record_cpu(record) for record in records)
    lengths = [value for record in records for value in record.get("lengths", [])]
    causes = {str(row.get("death_cause", row.get("h300_death_cause", "MISSING"))) for row in doom}
    return {
        "episodes": len(records),
        "attempted_episodes": len(attempted_records),
        "hero_decisions": decisions,
        "process_cpu_seconds": process_cpu,
        "partial_episode_cpu_seconds": sum(
            _record_cpu(record) for record in attempted_records if not record.get("complete")
        ),
        "attributed_total_cpu_seconds": attributed_cpu,
        "sum_episode_wall_seconds": sum(
            float(record.get("wall_seconds", 0.0)) for record in records
        ),
        "eligible_sampled_roots": len(rows),
        "prefilter_passes": sum(bool(row.get("prefilter")) for row in rows),
        "prefilter_rejections": sum(not bool(row.get("prefilter")) for row in rows),
        "h300_calls": sum(int(record.get("h300_calls", 0)) for record in records),
        "h300_branch_decisions": sum(
            int(record.get("h300_branch_decisions", 0)) for record in records
        ),
        "h300_cpu_seconds": sum(float(record.get("h300_cpu_seconds", 0.0)) for record in records),
        "doom_moments": len(doom),
        "doom_moments_per_cpu_second": _ratio(len(doom), attributed_cpu),
        "doom_moments_per_hero_decision": _ratio(len(doom), decisions),
        "doom_moments_per_1000_hero_decisions": _ratio(1000 * len(doom), decisions),
        "witnessed_avoidable_doom_roots": len(avoidable),
        "assessed_confirmed_doom_roots": len(assessed),
        "distinct_witnessed_avoidable_episodes": len({row["episode_id"] for row in avoidable}),
        "unassessed_confirmed_doom_roots": len(doom) - len(assessed),
        "witnessed_avoidable_fraction": _ratio(len(avoidable), len(assessed)),
        "witnessed_avoidable_doom_roots_per_total_cpu_second": _ratio(
            len(avoidable), attributed_cpu
        ),
        "distinct_doom_bearing_episodes": len({str(row["episode_id"]) for row in doom}),
        "distinct_doom_bearing_episodes_per_total_cpu_second": _ratio(
            len({str(row["episode_id"]) for row in doom}), attributed_cpu
        ),
        "hero_length_distribution": length_summary(lengths),
        "sampled_length_distribution": length_summary(
            row["length"] for row in rows if "length" in row
        ),
        "doom_length_distribution": length_summary(
            row["length"] for row in doom if "length" in row
        ),
        "death_causes": {
            cause: sum(
                str(row.get("death_cause", row.get("h300_death_cause", "MISSING"))) == cause
                for row in doom
            )
            for cause in causes
        },
        "doom_root_all_snake_lengths": [row.get("all_snake_lengths") for row in doom],
    }


def _map_cells(row: dict[str, Any]) -> float | None:
    try:
        value = float(row["map_cells"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 and np.isfinite(value) else None


def _geometry_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the first six lengths for the common six-snake geometry KS."""
    flattened = []
    for row in rows:
        values = row.get("all_snake_lengths")
        if not isinstance(values, (list, tuple)) or len(values) < 6:
            flattened.append(dict(row))
            continue
        flattened.append(
            {
                **row,
                **{f"snake_length_{index}": values[index] for index in range(6)},
            }
        )
    return flattened


def _opponent_length_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make one KS observation per opponent, retaining B70p's extra opponents."""
    values = []
    for row in rows:
        lengths = row.get("all_snake_lengths")
        if not isinstance(lengths, (list, tuple)) or len(lengths) < 2:
            return []
        values.extend({"opponent_length": length} for length in lengths[1:])
    return values


def _minimum_map_assessments(
    a_records: list[dict[str, Any]], complete: list[dict[str, Any]]
) -> dict[str, Any]:
    """Assess every root using A's natural per-mix rho95 and the 8,349-cell floor."""
    rho95: dict[str, float | None] = {}
    for mix in MIXES:
        fractions = []
        for row in _sampled(record for record in a_records if record["mix"] == mix):
            cells = _map_cells(row)
            try:
                fraction = float(row["total_occupied_cells"]) / cells if cells else None
            except (KeyError, TypeError, ValueError):
                fraction = None
            if fraction is not None and np.isfinite(fraction) and fraction >= 0:
                fractions.append(fraction)
        rho95[mix] = float(np.quantile(fractions, 0.95)) if fractions else None
    by_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm in ARMS:
        by_arm[arm] = {}
        for mix in MIXES:
            values = []
            for row in _sampled(r for r in complete if r.get("arm") == arm and r.get("mix") == mix):
                cells, density = _map_cells(row), rho95[mix]
                try:
                    hero, occupied = float(row["length"]), float(row["total_occupied_cells"])
                except (KeyError, TypeError, ValueError):
                    values.append({"status": "INSUFFICIENT", "reason": "missing_root_geometry"})
                    continue
                crowding = occupied / density if density and density > 0 else None
                if crowding is None:
                    values.append(
                        {
                            "status": "INSUFFICIENT",
                            "reason": "missing_natural_A_rho95",
                            "map_cells": cells,
                            "hero_floor_cells": 8 * hero,
                            "engineering_floor_cells": 8349,
                        }
                    )
                    continue
                required = max(8 * hero, crowding or 0.0, 8349.0)
                values.append(
                    {
                        "status": (
                            "OK"
                            if cells is not None and cells >= required
                            else "OUTSIDE_MINIMUM_MAP"
                        ),
                        "map_cells": cells,
                        "hero_floor_cells": 8 * hero,
                        "crowding_floor_cells": crowding,
                        "engineering_floor_cells": 8349,
                        "required_minimum_cells": required,
                    }
                )
            by_arm[arm][mix] = values
    return {
        "rho95_natural_total_occupied_fraction_by_mix": rho95,
        "formula": "max(8*L_hero, U_total/rho95, 8349)",
        "root_assessments": by_arm,
    }


def _decision(
    arm: str,
    mix: str,
    a: dict[str, Any],
    x: dict[str, Any],
    realism: dict[str, Any],
    map_values: list[dict[str, Any]],
    status: str,
    pooled_a: int,
) -> dict[str, Any]:
    if status == "INVALID":
        return {"status": "INVALID", "reasons": ["invalid_run"]}
    if status != "COMPLETE":
        return {"status": "INCOMPLETE", "reasons": ["incomplete_run"]}
    if pooled_a == 0:
        return {"status": "INSUFFICIENT", "reasons": ["pooled_A_zero_witnessed_avoidable_dooms"]}
    if a["distinct_doom_bearing_episodes"] < 4:
        return {"status": "INSUFFICIENT", "reasons": ["A_fewer_than_4_doom_bearing_episodes"]}
    if any(value.get("status") == "INSUFFICIENT" for value in map_values):
        return {"status": "INSUFFICIENT", "reasons": ["minimum_map_rho95_insufficient"]}
    if any(value.get("status") != "OK" for value in map_values):
        return {"status": "REJECTED", "reasons": ["minimum_map_rule_violation"]}
    a_yield, x_yield = (
        a["witnessed_avoidable_doom_roots_per_total_cpu_second"],
        x["witnessed_avoidable_doom_roots_per_total_cpu_second"],
    )
    a_fraction, x_fraction = a["witnessed_avoidable_fraction"], x["witnessed_avoidable_fraction"]
    yield_ratio = x_yield / a_yield if a_yield and a_yield > 0 and x_yield is not None else None
    fraction_ratio = (
        x_fraction / a_fraction
        if a_fraction and a_fraction > 0 and x_fraction is not None
        else None
    )
    reasons = []
    if a_yield is not None and x_yield is not None and x_yield < 2 * a_yield:
        reasons.append("avoidable_doom_yield_below_2x_A")
    if a_fraction is not None and x_fraction is not None and x_fraction < 0.5 * a_fraction:
        reasons.append("avoidable_fraction_below_half_A")
    classifier = realism.get("classifier")
    if classifier and classifier.get("status") == "OK" and classifier.get("auc") >= 0.80:
        reasons.append("classifier_auc_at_least_0.80")
    if reasons:
        return {
            "status": "REJECTED",
            "reasons": reasons,
            "yield_ratio_to_A": yield_ratio,
            "avoidable_fraction_ratio_to_A": fraction_ratio,
            "classifier_auc": classifier.get("auc") if classifier else None,
            "arm": arm,
            "mix": mix,
        }
    if a_fraction is None or x_fraction is None:
        return {"status": "INSUFFICIENT", "reasons": ["unassessed_avoidable_fraction"]}
    if not classifier or classifier.get("status") != "OK":
        reason = (
            classifier.get("reason", "missing_classifier") if classifier else "missing_classifier"
        )
        return {"status": "INSUFFICIENT", "reasons": ["classifier_" + str(reason)]}
    if realism.get("geometry_ks", {}).get("status") != "OK":
        return {"status": "INSUFFICIENT", "reasons": ["geometry_ks_insufficient"]}
    return {
        "status": "ADMITTED_FOR_A_TRANSFER_TEST",
        "reasons": [],
        "yield_ratio_to_A": yield_ratio,
        "avoidable_fraction_ratio_to_A": fraction_ratio,
        "classifier_auc": classifier["auc"],
        "auc_below_0_80_is_not_similarity_certification": classifier["auc"] < 0.80,
        "arm": arm,
        "mix": mix,
    }


def _set_cpu_rates(cell: dict[str, Any], attributed_cpu: float) -> None:
    """Update only CPU-denominated metrics after measured overhead is known."""
    cell["attributed_total_cpu_seconds"] = attributed_cpu
    cell["doom_moments_per_cpu_second"] = _ratio(cell["doom_moments"], attributed_cpu)
    cell["witnessed_avoidable_doom_roots_per_total_cpu_second"] = _ratio(
        cell["witnessed_avoidable_doom_roots"], attributed_cpu
    )
    cell["distinct_doom_bearing_episodes_per_total_cpu_second"] = _ratio(
        cell["distinct_doom_bearing_episodes"], attributed_cpu
    )


def finalize_cpu(
    analysis: dict[str, Any], records: list[dict[str, Any]], overhead_cpu_seconds: float
) -> dict[str, Any]:
    """Charge the final measured overhead without repeating frozen realism work.

    ``overhead_cpu_seconds`` is the absolute non-episode CPU total, including
    the first analysis pass itself.  The function intentionally reuses the
    existing raw counts, feature KS/AUC outputs, length summaries, and map
    assessments; it changes only CPU denominators and their decisions.
    """
    overhead = max(0.0, float(overhead_cpu_seconds))
    attempted_count = len(records)
    total_cpu = sum(_record_cpu(record) for record in records) + overhead
    cells = analysis["cells"]
    for arm in ARMS:
        for mix in MIXES:
            attempted = [
                record
                for record in records
                if record.get("arm") == arm and record.get("mix") == mix
            ]
            attributed = sum(_record_cpu(record) for record in attempted)
            if attempted_count:
                attributed += overhead * len(attempted) / attempted_count
            _set_cpu_rates(cells[arm][mix], attributed)
            cells[arm][mix]["attempted_episodes"] = len(attempted)
            cells[arm][mix]["partial_episode_cpu_seconds"] = sum(
                _record_cpu(record) for record in attempted if not record.get("complete")
            )
    _set_cpu_rates(analysis["aggregate_complete_episode_metrics"], total_cpu)
    analysis["overhead_cpu_seconds"] = overhead
    analysis["partial_episode_cpu_seconds"] = sum(
        _record_cpu(record) for record in records if not record.get("complete")
    )
    analysis["total_cpu_seconds"] = total_cpu
    pooled_a = sum(cells["A"][mix]["witnessed_avoidable_doom_roots"] for mix in MIXES)
    analysis["decisions"] = {
        arm: {
            mix: _decision(
                arm,
                mix,
                cells["A"][mix],
                cells[arm][mix],
                analysis["realism"][mix][arm],
                analysis["minimum_map"]["root_assessments"][arm][mix],
                analysis["run_status"],
                pooled_a,
            )
            for mix in MIXES
        }
        for arm in ARMS[1:]
    }
    return analysis


def summarize(
    records: list[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    overhead_cpu_seconds: float = 0.0,
    run_status: str | None = None,
) -> dict[str, Any]:
    """Apply amended endpoints and precedence, charging all CPU to rates before decisions."""
    invalid = [record["episode_id"] for record in records if record.get("invalid_reason")]
    complete = [record for record in records if record.get("complete")]
    status = "INVALID" if invalid or run_status == "INVALID" else (run_status or "COMPLETE")
    if status == "COMPLETE" and len(complete) != len(records):
        status = "INCOMPLETE"
    overhead, partial_cpu = max(0.0, float(overhead_cpu_seconds)), sum(
        _record_cpu(r) for r in records if not r.get("complete")
    )
    complete_cpu = sum(_record_cpu(r) for r in complete)
    attempted_count = len(records)
    cells: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        cells[arm] = {}
        for mix in MIXES:
            attempted = [r for r in records if r.get("arm") == arm and r.get("mix") == mix]
            subset = [r for r in attempted if r.get("complete")]
            base = sum(_record_cpu(r) for r in attempted)
            cells[arm][mix] = _cell(
                subset,
                base + overhead * len(attempted) / attempted_count if attempted_count else base,
                attempted,
            )
    a_records = [r for r in complete if r.get("arm") == "A"]
    realism: dict[str, Any] = {}
    for mix in MIXES:
        reference = _doom_rows(r for r in a_records if r.get("mix") == mix)
        realism[mix] = {}
        for arm in ARMS[1:]:
            candidate = _doom_rows(
                r for r in complete if r.get("arm") == arm and r.get("mix") == mix
            )
            realism[mix][arm] = {
                **compare(reference, candidate, feature_names),
                "geometry_ks": ks_compare(
                    _geometry_rows(reference), _geometry_rows(candidate), _GEOMETRY_FIELDS
                ),
                "all_opponent_length_ks": ks_compare(
                    _opponent_length_rows(reference),
                    _opponent_length_rows(candidate),
                    ["opponent_length"],
                ),
            }
            if arm == "C-P1":
                candidate_records = [
                    record
                    for record in complete
                    if record.get("arm") == arm and record.get("mix") == mix
                ]
                by_length = {}
                for target in (900, 1200, 1500):
                    subset = [
                        record
                        for record in candidate_records
                        if record.get("construction", {})
                        .get("fixture", {})
                        .get("body_lengths", [None])[0]
                        == target
                    ]
                    rows = _doom_rows(subset)
                    by_length[str(target)] = {
                        **compare(reference, rows, feature_names),
                        "geometry_ks": ks_compare(
                            _geometry_rows(reference), _geometry_rows(rows), _GEOMETRY_FIELDS
                        ),
                        "all_opponent_length_ks": ks_compare(
                            _opponent_length_rows(reference),
                            _opponent_length_rows(rows),
                            ["opponent_length"],
                        ),
                    }
                realism[mix][arm]["by_injected_start_length"] = by_length
    minimum_map = _minimum_map_assessments(a_records, complete)
    pooled_a = sum(cells["A"][mix]["witnessed_avoidable_doom_roots"] for mix in MIXES)
    decisions = {
        arm: {
            mix: _decision(
                arm,
                mix,
                cells["A"][mix],
                cells[arm][mix],
                realism[mix][arm],
                minimum_map["root_assessments"][arm][mix],
                status,
                pooled_a,
            )
            for mix in MIXES
        }
        for arm in ARMS[1:]
    }
    a_rows = _sampled(a_records)
    aggregate_cpu = complete_cpu + partial_cpu + overhead
    return {
        "scope": "development measurement; no science or release claims",
        "run_status": status,
        "denominator_scope": "all CPU work: complete episodes, partial episodes, and overhead",
        "invalid_episode_ids": invalid,
        "partial_episode_ids": [r["episode_id"] for r in records if not r.get("complete")],
        "partial_episode_cpu_seconds": partial_cpu,
        "overhead_cpu_seconds": overhead,
        "total_cpu_seconds": aggregate_cpu,
        "cells": cells,
        "arm_a_prefilter_recall": {
            "overall": _recall(a_rows),
            "by_mix": {
                mix: _recall(_sampled(r for r in a_records if r.get("mix") == mix)) for mix in MIXES
            },
        },
        "realism": realism,
        "minimum_map": minimum_map,
        "decisions": decisions,
        "feature_names": list(feature_names),
        "aggregate_complete_episode_metrics": _cell(complete, aggregate_cpu, records),
    }
