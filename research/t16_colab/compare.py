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
    "sp_w128_a2_packed_g0_bs2_fp32",
    "sp_w128_a2_raw_g0_bs2_fp32",
    "sp_w128_a3_raw_g3_bs2_fp32",
    "sp_w128_a4_raw_g3_bs2_fp32",
    "sp_w128_a6_raw_g3_bs2_fp32",
    "sp_w128_a4_raw_g3_bs4_fp32",
    "sp_w128_a4_raw_g3_bs2_bf16",
)
SMOKE03_THROUGHPUT_ONLY_NAMES = frozenset(SMOKE03_REQUIRED_CONFIG_NAMES[-2:])
SMOKE03_BATCH_SCALE_BY_NAME = {
    **{name: 2 for name in SMOKE03_REQUIRED_CONFIG_NAMES[:5]},
    SMOKE03_REQUIRED_CONFIG_NAMES[5]: 4,
    SMOKE03_REQUIRED_CONFIG_NAMES[6]: 2,
}


class ResultsError(ValueError):
    """Raised when a results document cannot support a throughput conclusion."""


def _valid_tps(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def _within_batch_tolerance(value: float | None, expected: float) -> bool:
    """Whether the measured replay ratio remains within the smoke-03 invariant."""
    if value is None:
        return False
    return abs(value - expected) <= expected * TRANSITIONS_PER_UPDATE_TOLERANCE


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
    if smoke03 and payload.get("omitted_smoke03_row5"):
        required_names = tuple(
            name for index, name in enumerate(required_names, start=1) if index != 5
        )
    top_level_failure = overall_status == "failed"
    names = [row.get("name") for row in configs if isinstance(row, Mapping)]
    duplicate_names = {name for name in names if isinstance(name, str) and names.count(name) > 1}
    control = next(
        (
            row
            for row in configs
            if isinstance(row, Mapping) and row.get("name") == SMOKE03_REQUIRED_CONFIG_NAMES[0]
        ),
        None,
    )
    control_policy_lag = (
        _valid_tps(control.get("policy_lag_mean"))
        if control
        and control.get("status") == "completed"
        and control.get("name") not in duplicate_names
        else None
    )
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
        policy_lag = _valid_tps(config.get("policy_lag_mean"))
        batch_scale = SMOKE03_BATCH_SCALE_BY_NAME.get(name, config.get("batch_scale", 2))
        expected_tpu = (
            EXPECTED_TRANSITIONS_PER_UPDATE * float(batch_scale) / 2
            if isinstance(batch_scale, (int, float)) and not isinstance(batch_scale, bool)
            else None
        )
        throughput_only = smoke03 and (
            bool(config.get("throughput_only", False)) or name in SMOKE03_THROUGHPUT_ONLY_NAMES
        )
        recipe_criteria_met = (
            throughput_only
            and status == "completed"
            and updates is not None
            and updates >= FOUR_X_UPDATES_PER_SECOND
            and expected_tpu is not None
            and _within_batch_tolerance(transitions_per_update, expected_tpu)
            and policy_lag is not None
            and control_policy_lag is not None
            and policy_lag <= control_policy_lag
        )
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
        elif smoke03 and (
            expected_tpu is None
            or not _within_batch_tolerance(transitions_per_update, expected_tpu)
        ):
            verdict, reason = (
                "INVALID",
                "transitions_per_update is outside the configured +/-2% range",
            )
        elif smoke03 and policy_lag is None:
            verdict, reason = "INVALID", "missing policy_lag_mean"
        elif throughput_only:
            verdict, reason = "THROUGHPUT_ONLY", "recipe arm; throughput evidence only"
        elif (
            smoke03
            and name != SMOKE03_REQUIRED_CONFIG_NAMES[0]
            and (control_policy_lag is None or policy_lag > control_policy_lag)
        ):
            verdict, reason = "FAIL", "policy_lag_mean exceeds the packed control"
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
            "expected_transitions_per_update": expected_tpu,
            "policy_lag_mean": policy_lag,
            "recipe_criteria_met": recipe_criteria_met if throughput_only else None,
            "verdict": verdict,
            "reason": reason,
        }
        if name.startswith("selfplay"):
            row["caveat"] = SELF_PLAY_CAVEAT
        for key in (
            "tail_transitions_per_second",
            "resources",
            "oversubscription",
            "learn_seconds_per_update",
            "actor_exitcodes",
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
                    "reason": "required smoke configuration is missing",
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
