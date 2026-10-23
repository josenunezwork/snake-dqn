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


# The functions below deliberately live after the frozen density endpoints above.
# They are the Mac-only, one-look aggregation path introduced by the resume ruling.

import json  # noqa: E402
from pathlib import Path  # noqa: E402


def _merge_episode_dirs(tree: "Path", replay: bool) -> list["Path"]:
    """Find one unambiguous exported episode directory in a recovered output tree."""
    leaf = "episodes-replay" if replay else "episodes"
    candidates = [
        tree / leaf,
        tree / "density" / leaf,
        tree / "exports" / "density" / leaf,
    ]
    present = [path for path in candidates if path.is_dir()]
    if len(present) > 1:
        raise ValueError(f"ambiguous {leaf} directories under {tree}")
    return sorted(present[0].glob("*.json")) if present else []


def _merge_receipt(tree: "Path") -> dict[str, str]:
    """Bind a source attempt receipt by path and digest without interpreting it."""
    import hashlib

    candidates = [tree / "receipt.json", tree / "exports" / "receipt.json"]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise ValueError(f"exactly one immutable receipt is required for {tree}")
    receipt = present[0]
    return {
        "path": str(receipt.resolve()),
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }


def _merge_manifest(tree: Path) -> dict[str, Any]:
    """Validate the immutable attempt identity before opening any episode payload."""
    from research.bl_density.completion import SCIENCE_COMMIT
    from research.bl_ppo.exports import digest

    receipt = json.loads((tree / "receipt.json").read_text())
    manifest_path = tree / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        receipt.get("status") not in {"PASS", "INCOMPLETE"}
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error")
        or receipt.get("manifest_sha256") != digest(manifest_path)
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("job_id") != manifest.get("job_id")
    ):
        raise ValueError("receipt/manifest identity or verified release differs")
    density = manifest.get("density", {})
    if (
        density.get("namespace") != "bl-density-round14b-20261022"
        or density.get("total_episodes") != 492
        or density.get("horizon") != 5000
        or density.get("sample_stride") != 200
    ):
        raise ValueError("manifest frozen allocation contract differs")
    if density.get("phase") == "production-completion":
        if density.get("science_commit") != SCIENCE_COMMIT:
            raise ValueError("completion science baseline differs")
    elif density.get("phase") != "production" or manifest.get("commit") != SCIENCE_COMMIT:
        raise ValueError("original production science baseline differs")
    return manifest


def _receipt_inventory(tree: Path) -> set[tuple[str, str, int]]:
    """Read only verified receipt/poll metadata; refuse conflicting inventories."""
    manifest = _merge_manifest(tree)
    receipt = json.loads((tree / "receipt.json").read_text())
    values = list(receipt.get("exports", []))
    values.extend((receipt.get("terminal") or {}).get("exports", []))
    # The original transport lost its receipt inventory. Only its frozen saved
    # terminal poll may establish those exports; do not trust arbitrary polls.
    if manifest["density"]["phase"] == "production":
        poll = json.loads((tree / "poll-000015.json").read_text())
        terminal = poll.get("terminal") or {}
        if (
            terminal.get("generation") != receipt.get("generation")
            or terminal.get("job_id") != receipt.get("job_id")
            or terminal.get("returncode") != 0
        ):
            raise ValueError("original saved terminal identity differs")
        values.extend(poll.get("exports", []))
    result: dict[str, tuple[str, int]] = {}
    for item in values:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or type(item.get("size")) is not int
            or item["size"] < 0
        ):
            raise ValueError("malformed immutable export inventory")
        name = item["path"]
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("unsafe immutable export inventory path")
        identity = (item["sha256"], item["size"])
        if name in result and result[name] != identity:
            raise ValueError("immutable export inventory metadata changed")
        result[name] = identity
    return {(name, digest, size) for name, (digest, size) in result.items()}


def _inventory_contains(
    tree: Path, path: Path, digest: str, inventory: set[tuple[str, str, int]]
) -> bool:
    relative = path.relative_to(tree).as_posix()
    canonical = relative.removeprefix("exports/")
    return (canonical, digest, path.stat().st_size) in inventory


def _validate_merge_record(record: dict[str, Any], path: Path, manifest: dict[str, Any]) -> None:
    """Bind gameplay identity and frozen rules without trusting filename alone."""
    from research.bl_density.completion import SCIENCE_COMMIT
    from research.bl_density.runner import _seed
    from research.bl_ppo.exports import digest

    identity = record.get("episode_id", "")
    parts = identity.split("/") if isinstance(identity, str) else []
    if len(parts) != 4:
        raise ValueError("malformed episode identity")
    namespace, arm, mix, index = parts
    if (
        path.name != identity.replace("/", "_") + ".json"
        or record.get("arm") != arm
        or record.get("mix") != mix
        or not index.isdecimal()
        or record.get("seed") != _seed(arm, mix, int(index), namespace=namespace)
        or record.get("plan", {}).get("seed") != record.get("seed")
        or record.get("horizon") != 5000
        or record.get("sample_stride") != 200
        or record.get("complete") is not True
    ):
        raise ValueError("episode scientific identity/seed/horizon/stride differs")
    env = manifest.get("env", {})
    bindings = {
        "metrics_sha256": ("METRICS.md", "BL_DENSITY_METRICS_SHA256"),
        "metrics_amendment_sha256": ("METRICS_AMENDMENT_01.md", "BL_DENSITY_AMENDMENT_SHA256"),
        "metrics_capacity_amendment_sha256": (
            "METRICS_AMENDMENT_02.md",
            "BL_DENSITY_CAPACITY_AMENDMENT_SHA256",
        ),
        "metrics_round14_amendment_sha256": (
            "METRICS_AMENDMENT_03.md",
            "BL_DENSITY_ROUND14_AMENDMENT_SHA256",
        ),
        "metrics_replication_amendment_sha256": (
            "METRICS_AMENDMENT_04.md",
            "BL_DENSITY_REPLICATION_AMENDMENT_SHA256",
        ),
    }
    for field, (name, key) in bindings.items():
        expected = digest(Path(__file__).with_name(name))
        if record.get(field) != expected or env.get(key) != expected:
            raise ValueError("episode frozen metrics binding differs: " + field)
    if record.get("metrics_commit") != manifest["commit"]:
        raise ValueError("episode source commit differs")
    if manifest["density"]["phase"] == "production-completion":
        if record.get("metrics_science_commit") != SCIENCE_COMMIT or record.get(
            "metrics_completion_amendment_sha256"
        ) != digest(Path(__file__).with_name("METRICS_AMENDMENT_05.md")):
            raise ValueError("episode completion science binding differs")
    cpu = record.get("cpu_seconds")
    if (
        isinstance(cpu, bool)
        or not isinstance(cpu, (float, int))
        or not np.isfinite(cpu)
        or cpu <= 0
    ):
        raise ValueError("episode CPU is malformed")


def _registered_episode_ids(namespace: str) -> set[str]:
    """Return the frozen DENSITY-2 allocation, never inferred from recovered files."""
    from research.bl_density.runner import DENSITY2_ALLOCATION, DENSITY2_NAMESPACE

    if namespace != DENSITY2_NAMESPACE:
        raise ValueError("merge accepts only the frozen DENSITY-2 namespace")
    return {
        f"{namespace}/{arm}/{mix}/{index:03d}"
        for arm, by_mix in DENSITY2_ALLOCATION.items()
        for mix, count in by_mix.items()
        for index in range(count)
    }


def _load_verified_merge_records(
    trees: Sequence["Path"], completion_trees: Sequence["Path"], namespace: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """Read only sidecar-verified, in-allocation normal and replay exports.

    The normal records are the sole input to the statistical look.  Replay
    records deliberately remain separate because their duplicate IDs are an
    identity control, not a second observation.
    """
    import json
    from pathlib import Path

    from research.bl_ppo.exports import verify_export

    normalized = [Path(tree).resolve() for tree in trees]
    completions = {Path(tree).resolve() for tree in completion_trees}
    if (
        not normalized
        or not completions <= set(normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("trees must contain distinct completion trees")
    if len(normalized) - len(completions) != 1:
        raise ValueError("exactly one original production tree required")
    allowed = _registered_episode_ids(namespace)
    normal: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    receipts = [_merge_receipt(tree) for tree in normalized]
    normal_ids: set[str] = set()
    for tree in normalized:
        manifest = _merge_manifest(tree)
        if (manifest["density"]["phase"] == "production-completion") != (tree in completions):
            raise ValueError("completion manifest must be declared as a completion source")
        inventory = _receipt_inventory(tree)
        for path in _merge_episode_dirs(tree, replay=False):
            digest = verify_export(path)
            if not _inventory_contains(tree, path, digest, inventory):
                raise ValueError("episode export is absent from immutable receipt inventory")
            record = json.loads(path.read_text())
            episode_id = record.get("episode_id")
            if record.get("namespace") != namespace or episode_id not in allowed:
                raise ValueError("foreign or malformed normal episode identity")
            _validate_merge_record(record, path, manifest)
            if episode_id in normal_ids:
                raise ValueError("duplicate episode_id across verified normal exports")
            normal_ids.add(episode_id)
            record["_merge_completion_source"] = tree in completions
            record["_merge_source_tree"] = str(tree)
            normal.append(record)
        if tree in completions:
            for path in _merge_episode_dirs(tree, replay=True):
                digest = verify_export(path)
                if not _inventory_contains(tree, path, digest, inventory):
                    raise ValueError("replay export is absent from immutable receipt inventory")
                record = json.loads(path.read_text())
                episode_id = record.get("episode_id")
                if record.get("namespace") != namespace or episode_id not in allowed:
                    raise ValueError("foreign or malformed replay episode identity")
                _validate_merge_record(record, path, manifest)
                record["_merge_completion_source"] = True
                record["_merge_source_tree"] = str(tree)
                replay.append(record)
    normal.sort(key=lambda record: record["episode_id"])
    return normal, replay, receipts


def _identity_replay(
    normal: Sequence[dict[str, Any]], replay: Sequence[dict[str, Any]], namespace: str
) -> dict[str, Any]:
    """Compare the fixed twelve replay identities after sidecar verification."""
    from research.bl_density.verify import semantic

    expected = {f"{namespace}/A/{mix}/{index:03d}" for mix in MIXES for index in range(4)}
    originals = {str(record.get("episode_id")): record for record in normal}
    replay_by_id = {str(record.get("episode_id")): record for record in replay}
    replay_set_valid = len(replay_by_id) == len(replay) and set(replay_by_id) == expected
    missing_originals = sorted(expected - set(originals))
    if not replay_set_valid or missing_originals:
        return {
            "status": "IDENTITY_FAIL",
            "reason": "fixed_replay_set_or_originals_missing",
            "episode_ids": sorted(expected),
            "matched_episodes": 0,
            "mismatched_episode_ids": [],
            "missing_original_episode_ids": missing_originals,
            "replay_episode_ids": sorted(replay_by_id),
            "original_cpu_seconds": None,
            "replay_cpu_seconds": None,
            "cpu_calibration_ratio": None,
        }

    def canonical(record: dict[str, Any]) -> Any:
        from os.path import basename

        # Merge provenance is Mac-local bookkeeping, not gameplay identity.
        value = {
            key: value
            for key, value in record.items()
            if key not in {"_merge_completion_source", "_merge_source_tree"}
        }
        # Checkpoint bytes are pinned by PUBLIC_INPUTS.json, not session staging paths.
        for parent, field in (("plan", "specs"), ("construction", "roster_specs")):
            if field in value.get(parent, {}):
                value[parent] = dict(value[parent])
                value[parent][field] = [
                    (
                        ["checkpoint", basename(spec[1])]
                        if isinstance(spec, list) and len(spec) == 2 and spec[0] == "checkpoint"
                        else spec
                    )
                    for spec in value[parent][field]
                ]
        return semantic(value)

    mismatches = [
        episode_id
        for episode_id in sorted(expected)
        if canonical(originals[episode_id]) != canonical(replay_by_id[episode_id])
    ]
    original_cpu = sum(_record_cpu(originals[episode_id]) for episode_id in expected)
    replay_cpu = sum(_record_cpu(replay_by_id[episode_id]) for episode_id in expected)
    ratio = replay_cpu / original_cpu if original_cpu > 0 else None
    return {
        "status": "IDENTITY_FAIL" if mismatches else "IDENTICAL",
        "episode_ids": sorted(expected),
        "matched_episodes": len(expected) - len(mismatches),
        "mismatched_episode_ids": mismatches,
        "original_cpu_seconds": original_cpu,
        "replay_cpu_seconds": replay_cpu,
        "cpu_calibration_ratio": ratio,
    }


def _completion_attempt(tree: "Path") -> dict[str, Any]:
    """Load a sidecar-verified, non-outcome completion accounting record."""
    import json

    from research.bl_ppo.exports import verify_export

    candidates = [
        tree / "attempt.json",
        tree / "density" / "attempt.json",
        tree / "exports" / "density" / "attempt.json",
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError("completion tree requires exactly one sidecar-verified attempt.json")
    observed = verify_export(present[0])
    if not _inventory_contains(tree, present[0], observed, _receipt_inventory(tree)):
        raise ValueError("attempt accounting absent from immutable receipt inventory")
    value = json.loads(present[0].read_text())
    if (
        value.get("namespace") != "bl-density-round14b-20261022"
        or value.get("b8_sha256")
        != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
    ):
        raise ValueError("attempt namespace/B8 identity differs")
    if value.get("schema") != "bl-density-attempt/v1":
        raise ValueError("completion attempt schema mismatch")
    count, overhead = value.get("attempted_episode_count"), value.get(
        "attempt_cpu_overhead_seconds"
    )
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("completion attempt count is malformed")
    if (
        isinstance(overhead, bool)
        or not isinstance(overhead, (int, float))
        or not np.isfinite(overhead)
        or overhead < 0
    ):
        raise ValueError("completion attempt overhead is malformed")
    return {
        "path": str(present[0].resolve()),
        "tree": str(tree.resolve()),
        "sha256": verify_export(present[0]),
        **value,
    }


def _cell_completion_counts(records: Sequence[dict[str, Any]], namespace: str) -> dict[str, Any]:
    from research.bl_density.runner import DENSITY2_ALLOCATION

    values: dict[str, Any] = {}
    for arm, by_mix in DENSITY2_ALLOCATION.items():
        values[arm] = {}
        for mix, registered in by_mix.items():
            matching = [
                record
                for record in records
                if record.get("arm") == arm and record.get("mix") == mix
            ]
            completed = sum(bool(record.get("complete")) for record in matching)
            values[arm][mix] = {
                "namespace": namespace,
                "registered_episodes": registered,
                "attempted_episodes": len(matching),
                "complete_episodes": completed,
                "missing_episodes": registered - completed,
            }
    return values


def _reprice_completion_cpu(
    records: Sequence[dict[str, Any]], ratio: float | dict[str, float]
) -> list[dict[str, Any]]:
    """Scale only records produced by the completion session by 1/r."""
    import copy

    value = copy.deepcopy(list(records))
    for record in value:
        if record.pop("_merge_completion_source", False):
            factor = ratio if isinstance(ratio, float) else ratio[record["_merge_source_tree"]]
            record["cpu_seconds"] = _record_cpu(record) / factor
            if "h300_cpu_seconds" in record:
                record["h300_cpu_seconds"] = float(record["h300_cpu_seconds"]) / factor
        else:
            record.pop("_merge_completion_source", None)
        record.pop("_merge_source_tree", None)
    return value


def _refresh_merge_decisions(analysis: dict[str, Any]) -> None:
    """Reapply only the unchanged registered rule after CPU attribution."""
    pooled_a = sum(analysis["cells"]["A"][mix]["witnessed_avoidable_doom_roots"] for mix in MIXES)
    analysis["decisions"] = {
        arm: {
            mix: _decision(
                arm,
                mix,
                analysis["cells"]["A"][mix],
                analysis["cells"][arm][mix],
                analysis["realism"][mix][arm],
                analysis["minimum_map"]["root_assessments"][arm][mix],
                analysis["run_status"],
                pooled_a,
            )
            for mix in MIXES
        }
        for arm in ARMS[1:]
    }


def _registered_overhead(
    analysis: dict[str, Any], records: Sequence[dict[str, Any]], overhead: float
) -> None:
    """Charge an equal overhead amount to each registered slot, including missing slots."""
    from research.bl_density.runner import DENSITY2_ALLOCATION

    total = sum(sum(cells.values()) for cells in DENSITY2_ALLOCATION.values())
    for arm, cells in DENSITY2_ALLOCATION.items():
        for mix, count in cells.items():
            cpu = sum(_record_cpu(r) for r in records if r["arm"] == arm and r["mix"] == mix)
            _set_cpu_rates(analysis["cells"][arm][mix], cpu + overhead * count / total)
    _refresh_merge_decisions(analysis)


def _charge_partial_attempt_cpu(
    analysis: dict[str, Any],
    attempts: Sequence[dict[str, Any]],
    *,
    ratios: dict[str, float] | None = None,
) -> None:
    """Charge verified non-scoring partial primary work to its registered cell."""
    charged = 0.0
    for attempt in attempts:
        scale = 1.0 if ratios is None else ratios[str(attempt["tree"])]
        for item in attempt.get("partial_episode_cpu_seconds", []):
            if not isinstance(item, dict) or item.get("kind") != "primary":
                continue
            parts = str(item.get("episode_id", "")).split("/")
            value = item.get("cpu_seconds")
            if (
                len(parts) != 4
                or parts[1] not in ARMS
                or parts[2] not in MIXES
                or not isinstance(value, (int, float))
            ):
                raise ValueError("malformed partial episode accounting")
            if item["episode_id"] not in _registered_episode_ids("bl-density-round14b-20261022"):
                raise ValueError("partial accounting identity outside allocation")
            cpu = float(value) / scale
            if not np.isfinite(cpu) or cpu < 0:
                raise ValueError("malformed partial episode CPU")
            cell = analysis["cells"][parts[1]][parts[2]]
            cell["partial_episode_cpu_seconds"] += cpu
            cell["attempted_episodes"] += 1
            _set_cpu_rates(cell, cell["attributed_total_cpu_seconds"] + cpu)
            charged += cpu
    if not charged:
        return
    aggregate = analysis["aggregate_complete_episode_metrics"]
    analysis["partial_episode_cpu_seconds"] += charged
    analysis["total_cpu_seconds"] += charged
    _set_cpu_rates(aggregate, aggregate["attributed_total_cpu_seconds"] + charged)
    pooled_a = sum(analysis["cells"]["A"][mix]["witnessed_avoidable_doom_roots"] for mix in MIXES)
    analysis["decisions"] = {
        arm: {
            mix: _decision(
                arm,
                mix,
                analysis["cells"]["A"][mix],
                analysis["cells"][arm][mix],
                analysis["realism"][mix][arm],
                analysis["minimum_map"]["root_assessments"][arm][mix],
                analysis["run_status"],
                pooled_a,
            )
            for mix in MIXES
        }
        for arm in ARMS[1:]
    }


def merge(
    trees: Sequence["Path"],
    *,
    completion_tree: "Path | Sequence[Path]",
    namespace: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Produce the one Mac-side DENSITY-2 look from verified immutable exports.

    Callers must complete the published-ledger fence before calling this
    function.  It never reads a per-attempt summary, which stays sealed.
    """
    from pathlib import Path

    from research.bl_density.runner import DENSITY2_NAMESPACE, FEATURE_NAMES

    active_namespace = namespace or DENSITY2_NAMESPACE
    completions = (
        [completion_tree]
        if not isinstance(completion_tree, (list, tuple))
        else list(completion_tree)
    )
    if not 1 <= len(completions) <= 2 or len(set(map(str, completions))) != len(completions):
        raise ValueError("one or two distinct completion trees required")
    # _load_verified_merge_records accepts one completion source.  Mark every
    # completion record below so a mechanically authorized second attempt is
    # included in the same session adjustment.
    records, replay, receipts = _load_verified_merge_records(trees, completions, active_namespace)
    base = {
        "schema": "bl-density-merge/v1",
        "namespace": active_namespace,
        "immutable_receipts": receipts,
        "completion_counts": _cell_completion_counts(records, active_namespace),
    }
    try:
        attempts = [_completion_attempt(Path(tree)) for tree in completions]
    except ValueError as exc:
        # A killed transport can leave individually verified payloads without
        # the terminal accounting needed to charge their CPU.  Preserve their
        # count, but never open the statistical look on an unpriced union.
        return {
            **base,
            "status": "INCOMPLETE",
            "completion_void": False,
            "reason": "missing_verified_completion_accounting",
            "accounting_error": str(exc),
        }
    replay_by_tree: dict[str, list[dict[str, Any]]] = {}
    for record in replay:
        replay_by_tree.setdefault(record["_merge_source_tree"], []).append(record)
    identities = {
        str(Path(tree).resolve()): _identity_replay(
            records, replay_by_tree.get(str(Path(tree).resolve()), []), active_namespace
        )
        for tree in completions
    }
    base["identity_replay"] = {"sessions": identities}
    for attempt in attempts:
        for item in attempt.get("partial_episode_cpu_seconds", []):
            if isinstance(item, dict) and item.get("kind") == "primary":
                parts = str(item.get("episode_id", "")).split("/")
                if len(parts) == 4 and parts[1] in ARMS and parts[2] in MIXES:
                    base["completion_counts"][parts[1]][parts[2]]["attempted_episodes"] += 1
    if any(value["status"] != "IDENTICAL" for value in identities.values()):
        return {**base, "status": "IDENTITY_FAIL", "completion_void": True}
    # The ruling charges the completion attempt's measured per-attempt overhead
    # to every one of the original 492 registered episodes.
    registered_count = len(_registered_episode_ids(active_namespace))
    # Multiple mechanically authorized attempts remain one transport series:
    # use its measured aggregate per-attempt overhead, then apply that neutral
    # rate to every frozen allocation unit.
    overhead = (
        sum(float(attempt["attempt_cpu_overhead_seconds"]) for attempt in attempts)
        / sum(int(attempt["attempted_episode_count"]) for attempt in attempts)
        * registered_count
    )
    clean_records = _reprice_completion_cpu(records, 1.0)
    raw = finalize_cpu(
        summarize(
            clean_records,
            feature_names or FEATURE_NAMES,
            run_status=(
                "COMPLETE"
                if len(clean_records) == registered_count
                and all(r.get("complete") for r in clean_records)
                else "INCOMPLETE"
            ),
        ),
        clean_records,
        overhead,
    )
    _registered_overhead(raw, clean_records, overhead)
    _charge_partial_attempt_cpu(raw, attempts)
    ratios = {tree: value["cpu_calibration_ratio"] for tree, value in identities.items()}
    if any(ratio is None or not np.isfinite(ratio) or ratio <= 0 for ratio in ratios.values()):
        raise ValueError("identity replay CPU ratio is malformed")
    typed_ratios = {tree: float(ratio) for tree, ratio in ratios.items()}
    rescaled_records = _reprice_completion_cpu(records, typed_ratios)
    rescaled = finalize_cpu(
        summarize(
            rescaled_records,
            feature_names or FEATURE_NAMES,
            run_status=(
                "COMPLETE"
                if len(rescaled_records) == registered_count
                and all(r.get("complete") for r in rescaled_records)
                else "INCOMPLETE"
            ),
        ),
        rescaled_records,
        (
            sum(
                float(attempt["attempt_cpu_overhead_seconds"]) / typed_ratios[str(attempt["tree"])]
                for attempt in attempts
            )
            / sum(int(attempt["attempted_episode_count"]) for attempt in attempts)
            * registered_count
        ),
    )
    _registered_overhead(rescaled, rescaled_records, rescaled["overhead_cpu_seconds"])
    _charge_partial_attempt_cpu(rescaled, attempts, ratios=typed_ratios)
    outside = any(not 0.80 <= ratio <= 1.25 for ratio in typed_ratios.values())
    effective: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        effective[arm] = {}
        for mix in MIXES:
            raw_decision = raw["decisions"][arm][mix]
            scaled_decision = rescaled["decisions"][arm][mix]
            if (
                outside
                and raw_decision["status"] != "REJECTED"
                and (raw_decision["status"] != scaled_decision["status"])
            ):
                decision = {
                    "status": "INSUFFICIENT_SESSION",
                    "reasons": ["raw_and_rescaled_session_decisions_disagree"],
                }
            else:
                decision = raw_decision
            effective[arm][mix] = {
                "status": decision["status"],
                "reasons": decision.get("reasons", []),
                "raw": raw_decision,
                "completion_cpu_rescaled_by_1_over_r": scaled_decision,
            }
    return {
        **base,
        "status": raw["run_status"],
        "completion_void": False,
        "completion_overhead": {
            "attempts": [
                {
                    key: attempt[key]
                    for key in (
                        "path",
                        "sha256",
                        "attempt_cpu_overhead_seconds",
                        "attempted_episode_count",
                    )
                }
                for attempt in attempts
            ],
            "charged_uniformly_over_registered_episodes": registered_count,
            "charged_overhead_cpu_seconds": overhead,
        },
        "session_confound": {
            "cpu_calibration_ratios_r": typed_ratios,
            "outside_registered_band": outside,
            "registered_band": [0.80, 1.25],
            "raw_evaluation": raw,
            "completion_cpu_rescaled_by_1_over_r_evaluation": rescaled,
            "effective_decisions": effective,
        },
    }


def main() -> int:
    """Run the protected, create-only Mac merge entrypoint."""
    import argparse
    from pathlib import Path

    from research.bl_ppo.exports import export_json
    from research.published_ledger import bind_published_ledger_check, require_published_ledger

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("merge", help="perform the one DENSITY-2 Mac look")
    command.add_argument("--tree", type=Path, action="append", required=True)
    command.add_argument("--completion-tree", type=Path, action="append", required=True)
    command.add_argument("--authoring-repo", type=Path, required=True)
    command.add_argument("--ledger", required=True)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or any(
        Path(str(args.out) + suffix).exists() for suffix in (".sha256", ".complete")
    ):
        parser.error("merge summary output is create-only")
    # This fence must precede every export payload read, so a dirty or
    # unpublished registered ledger cannot consume the one outcome look.
    proof = require_published_ledger(args.authoring_repo, args.ledger)
    result = merge(
        args.tree,
        completion_tree=args.completion_tree,
    )
    export_json(args.out, bind_published_ledger_check(result, proof))
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
