#!/usr/bin/env python3
"""Validate and compare T16 Colab throughput results with the Mac baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "t16-colab-throughput/v1"
MAC_BASELINE_TRANSITIONS_PER_SECOND = 5_400.0
FOUR_X_TRANSITIONS_PER_SECOND = MAC_BASELINE_TRANSITIONS_PER_SECOND * 4
MAC_BASELINE_UPDATES_PER_SECOND = 21.0
FOUR_X_UPDATES_PER_SECOND = MAC_BASELINE_UPDATES_PER_SECOND * 4
EXPECTED_TRANSITIONS_PER_UPDATE = 256.0
TRANSITIONS_PER_UPDATE_TOLERANCE = 0.02
REQUIRED_CONFIG_NAMES = (
    "hero_w128_a2",
    "hero_w128_a4",
    "hero_w128_a8",
    "selfplay_w64_a2",
    "selfplay_w128_a2",
    "selfplay_w128_a4",
)
SELF_PLAY_CAVEAT = (
    "self-play transitions count every learner-driven snake; the 5,400/s Mac baseline is "
    "the hero-only recipe, so this ratio is wall-clock per training transition, not a "
    "same-configuration comparison until a Mac self-play rate is measured"
)
SMOKE03_REQUIRED_CONFIG_NAMES = (
    "selfplay_w128_a2_g0",
    "selfplay_w128_a2_g3",
    "selfplay_w128_a3_g3",
    "selfplay_w128_a3_g4",
)


class ResultsError(ValueError):
    """Raised when a results document cannot support a throughput conclusion."""


def _valid_tps(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def _within_batch_tolerance(value: float | None) -> bool:
    """Whether the measured replay ratio remains within the smoke-03 invariant."""
    if value is None:
        return False
    return abs(value - EXPECTED_TRANSITIONS_PER_UPDATE) <= (
        EXPECTED_TRANSITIONS_PER_UPDATE * TRANSITIONS_PER_UPDATE_TOLERANCE
    )


def compare_results(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return one conservative verdict per declared timing configuration.

    A row passes only when the runner explicitly completed it and it reports a
    finite throughput at or above the 4x threshold. Failed or malformed rows
    never become a pass merely because another row did.
    """
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise ResultsError(f"expected schema {SCHEMA!r}")
    configs = payload.get("configs")
    if not isinstance(configs, list):
        raise ResultsError("results must contain a configs list")
    overall_status = payload.get("status")
    suite = payload.get("suite", "t16")
    if suite not in ("t16", "smoke03"):
        raise ResultsError(f"unknown suite {suite!r}")
    smoke03 = suite == "smoke03"
    required_names = SMOKE03_REQUIRED_CONFIG_NAMES if smoke03 else REQUIRED_CONFIG_NAMES
    top_level_failure = overall_status == "failed"
    names = [row.get("name") for row in configs if isinstance(row, Mapping)]
    duplicate_names = {name for name in names if isinstance(name, str) and names.count(name) > 1}
    rows = []
    for index, config in enumerate(configs):
        if not isinstance(config, Mapping):
            rows.append({"index": index, "status": "INVALID", "reason": "config is not an object"})
            continue
        name = config.get("name")
        name = name if isinstance(name, str) and name else f"config-{index}"
        status = config.get("status")
        throughput = _valid_tps(config.get("transitions_per_second"))
        updates = _valid_tps(config.get("updates_per_second"))
        transitions_per_update = _valid_tps(config.get("transitions_per_update"))
        if top_level_failure:
            verdict, reason = "INVALID", "top-level smoke status is failed"
        elif name in duplicate_names:
            verdict, reason = "INVALID", "duplicate configuration name"
        elif status != "completed":
            verdict, reason = "INVALID", f"runner status is {status!r}"
        elif throughput is None:
            verdict, reason = "INVALID", "missing or invalid transitions_per_second"
        elif smoke03 and updates is None:
            verdict, reason = "INVALID", "missing or invalid updates_per_second"
        elif smoke03 and not _within_batch_tolerance(transitions_per_update):
            verdict, reason = "INVALID", "transitions_per_update is outside 256 +/-2%"
        elif smoke03 and updates >= FOUR_X_UPDATES_PER_SECOND:
            verdict, reason = (
                "PASS",
                "clears the 4x Mac updates/s threshold at the fixed replay ratio",
            )
        elif smoke03:
            verdict, reason = "FAIL", "below the 4x Mac updates/s threshold"
        elif throughput >= FOUR_X_TRANSITIONS_PER_SECOND:
            verdict, reason = "PASS", "clears the 4x Mac threshold"
        else:
            verdict, reason = "FAIL", "below the 4x Mac threshold"
        row = {
            "name": name,
            "runner_status": status,
            "transitions_per_second": throughput,
            "baseline_transitions_per_second": MAC_BASELINE_TRANSITIONS_PER_SECOND,
            "threshold_transitions_per_second": FOUR_X_TRANSITIONS_PER_SECOND,
            "speedup_vs_mac": (
                None if throughput is None else throughput / MAC_BASELINE_TRANSITIONS_PER_SECOND
            ),
            "updates_per_second": updates,
            "baseline_updates_per_second": MAC_BASELINE_UPDATES_PER_SECOND,
            "threshold_updates_per_second": FOUR_X_UPDATES_PER_SECOND,
            "transitions_per_update": transitions_per_update,
            "verdict": verdict,
            "reason": reason,
        }
        if name.startswith("selfplay"):
            row["caveat"] = SELF_PLAY_CAVEAT
        for key in (
            "tail_transitions_per_second",
            "resources",
            "oversubscription",
            "policy_lag_mean",
            "learn_seconds_per_update",
        ):
            if key in config:
                row[key] = config[key]
        rows.append(row)
    present = {name for name in names if isinstance(name, str)}
    for name in required_names:
        if name not in present:
            rows.append(
                {
                    "name": name,
                    "runner_status": "missing",
                    "transitions_per_second": None,
                    "baseline_transitions_per_second": MAC_BASELINE_TRANSITIONS_PER_SECOND,
                    "threshold_transitions_per_second": FOUR_X_TRANSITIONS_PER_SECOND,
                    "speedup_vs_mac": None,
                    "updates_per_second": None,
                    "baseline_updates_per_second": MAC_BASELINE_UPDATES_PER_SECOND,
                    "threshold_updates_per_second": FOUR_X_UPDATES_PER_SECOND,
                    "transitions_per_update": None,
                    "verdict": "INVALID",
                    "reason": "required T16 configuration is missing",
                }
            )
    return {
        "schema": "t16-colab-throughput-comparison/v1",
        "source_schema": SCHEMA,
        "source_status": overall_status,
        "suite": suite,
        "baseline_transitions_per_second": MAC_BASELINE_TRANSITIONS_PER_SECOND,
        "threshold_transitions_per_second": FOUR_X_TRANSITIONS_PER_SECOND,
        "baseline_updates_per_second": MAC_BASELINE_UPDATES_PER_SECOND,
        "threshold_updates_per_second": FOUR_X_UPDATES_PER_SECOND,
        "expected_transitions_per_update": EXPECTED_TRANSITIONS_PER_UPDATE,
        "configs": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Read a results JSON and print its conservative comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.results.read_text(encoding="utf-8"))
        report = compare_results(payload)
    except (OSError, json.JSONDecodeError, ResultsError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
