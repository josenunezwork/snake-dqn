"""Descriptive density summaries, retaining raw counts and insufficiency reasons."""

from __future__ import annotations

from typing import Any, Sequence

from research.bl_density.statistics import compare, length_summary


def _ratio(numerator: int, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _recall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if isinstance(row.get("natural_doom"), bool)]
    true = [row for row in known if row["natural_doom"]]
    captured = sum(bool(row["prefilter"]) for row in true)
    return {
        "status": "OK" if true else "INSUFFICIENT",
        "scope": "sampled L>=800 grid; exact uninterrupted trajectory labels",
        "eligible_roots": len(rows),
        "known_labels": len(known),
        "censored_labels": len(rows) - len(known),
        "true_dooms": len(true),
        "captured_true_dooms": captured,
        "rejected_true_dooms": len(true) - captured,
        "recall": _ratio(captured, len(true)),
    }


def _doom_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**row, "episode_id": record["episode_id"]}
        for record in records
        for row in record["sampled"]
        if row.get("prefilter") and row.get("h300_doom") is True
    ]


def _cell(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for record in records for row in record["sampled"]]
    doom = _doom_rows(records)
    decisions = sum(record["decisions"] for record in records)
    cpu = sum(record["cpu_seconds"] for record in records)
    lengths = [value for record in records for value in record["lengths"]]
    return {
        "episodes": len(records),
        "hero_decisions": decisions,
        "process_cpu_seconds": cpu,
        "sum_episode_wall_seconds": sum(record["wall_seconds"] for record in records),
        "eligible_sampled_roots": len(rows),
        "prefilter_passes": sum(bool(row["prefilter"]) for row in rows),
        "prefilter_rejections": sum(not row["prefilter"] for row in rows),
        "h300_calls": sum(record["h300_calls"] for record in records),
        "h300_branch_decisions": sum(record["h300_branch_decisions"] for record in records),
        "h300_cpu_seconds": sum(record["h300_cpu_seconds"] for record in records),
        "doom_moments": len(doom),
        "doom_moments_per_cpu_second": _ratio(len(doom), cpu),
        "doom_moments_per_hero_decision": _ratio(len(doom), decisions),
        "doom_moments_per_1000_hero_decisions": _ratio(1000 * len(doom), decisions),
        "hero_length_distribution": length_summary(lengths),
        "sampled_length_distribution": length_summary(row["length"] for row in rows),
        "doom_length_distribution": length_summary(row["length"] for row in doom),
    }


def summarize(records: list[dict[str, Any]], feature_names: Sequence[str]) -> dict[str, Any]:
    """Summarize only complete episodes, and disclose every excluded partial unit."""
    complete = [record for record in records if record["complete"]]
    cells = {
        arm: {
            mix: _cell([r for r in complete if r["arm"] == arm and r["mix"] == mix])
            for mix in ("scripted", "frozen", "mixed")
        }
        for arm in ("A", "B50", "B70", "C")
    }
    a_records = [record for record in complete if record["arm"] == "A"]
    realism: dict[str, Any] = {}
    for mix in ("scripted", "frozen", "mixed"):
        reference = _doom_rows([r for r in a_records if r["mix"] == mix])
        realism[mix] = {}
        for arm in ("B50", "B70", "C"):
            candidate_records = [r for r in complete if r["arm"] == arm and r["mix"] == mix]
            result = compare(reference, _doom_rows(candidate_records), feature_names)
            if arm == "C":
                by_length = {}
                for length in (800, 1200, 1500):
                    subset = [
                        r
                        for r in candidate_records
                        if r["construction"]["fixture"]["body_lengths"][0] == length
                    ]
                    by_length[str(length)] = compare(reference, _doom_rows(subset), feature_names)
                result["by_injected_start_length"] = by_length
            realism[mix][arm] = result
    a_rows = [row for record in a_records for row in record["sampled"]]
    return {
        "scope": "development measurement; no science or release claims",
        "denominator_scope": "complete episodes only; partial episode work disclosed separately",
        "partial_episode_ids": [r["episode_id"] for r in records if not r["complete"]],
        "partial_episode_cpu_seconds": sum(r["cpu_seconds"] for r in records if not r["complete"]),
        "cells": cells,
        "arm_a_prefilter_recall": {
            "overall": _recall(a_rows),
            "by_mix": {
                mix: _recall([row for r in a_records if r["mix"] == mix for row in r["sampled"]])
                for mix in ("scripted", "frozen", "mixed")
            },
        },
        "realism": realism,
        "feature_names": list(feature_names),
        "aggregate_complete_episode_metrics": _cell(complete),
    }
