"""Statistics for controlled observation-value counterfactual probes.

These helpers measure disagreement between two action-value vectors exposed on
the same frozen world.  They deliberately do not estimate natural aliasing,
optimal Q values, or promotion quality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np

_ACTION_COUNT = 6


def _require_real(value: object, label: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not isfinite(number) or (non_negative and number < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _validated_values(values: Sequence[float | None], label: str) -> tuple[float | None, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence with exactly {_ACTION_COUNT} slots")
    if len(values) != _ACTION_COUNT:
        raise ValueError(f"{label} must contain exactly {_ACTION_COUNT} slots")
    return tuple(
        None if value is None else _require_real(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )


def _best_actions(values: tuple[float | None, ...], margin: float) -> list[int]:
    eligible = [value for value in values if value is not None]
    maximum = max(eligible)
    return [
        index
        for index, value in enumerate(values)
        if value is not None and value >= maximum - margin
    ]


def pair_metrics(
    left: Sequence[float | None], right: Sequence[float | None], tie_margin: float = 0.01
) -> dict[str, Any]:
    """Measure action disagreement for a single, controlled world pair.

    ``None`` marks an unavailable action and must occur in the same slot in
    both vectors.  The returned regret is a conditional counterfactual metric,
    not a natural alias-prevalence or policy-quality statistic.
    """
    margin = _require_real(tie_margin, "tie_margin", non_negative=True)
    left_values = _validated_values(left, "left")
    right_values = _validated_values(right, "right")
    if tuple(value is None for value in left_values) != tuple(
        value is None for value in right_values
    ):
        raise ValueError("left and right must have the same unavailable-action domain")
    eligible = [index for index, value in enumerate(left_values) if value is not None]
    if not eligible:
        raise ValueError("at least one action must be eligible")

    left_preferred = _best_actions(left_values, margin)
    right_preferred = _best_actions(right_values, margin)
    averages = {
        index: (float(left_values[index]) + float(right_values[index])) / 2.0 for index in eligible
    }
    best_common_action = min(
        index for index in eligible if averages[index] == max(averages.values())
    )
    regret = (
        max(float(left_values[index]) for index in eligible)
        + max(float(right_values[index]) for index in eligible)
    ) / 2.0 - averages[best_common_action]
    # Arithmetic may introduce a negative residue around zero; a real negative
    # value signals a broken formula and must remain visible.
    if regret < 0.0 and regret > -1e-12:
        regret = 0.0

    disagreements = 0
    for first_position, first in enumerate(eligible):
        for second in eligible[first_position + 1 :]:
            left_delta = float(left_values[first]) - float(left_values[second])
            right_delta = float(right_values[first]) - float(right_values[second])
            if abs(left_delta) <= margin or abs(right_delta) <= margin:
                continue
            if (left_delta > 0.0) != (right_delta > 0.0):
                disagreements += 1

    return {
        "left_preferred_actions": left_preferred,
        "right_preferred_actions": right_preferred,
        "disjoint_best_sets": not set(left_preferred).intersection(right_preferred),
        "best_common_action": best_common_action,
        "best_common_action_regret": regret,
        "directional_order_disagreements": disagreements,
        "number_of_choices": len(eligible),
    }


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_metric_record(metrics: object) -> tuple[float, bool]:
    if not isinstance(metrics, Mapping):
        raise ValueError("record metrics must be a mapping")
    required = {"best_common_action_regret", "disjoint_best_sets"}
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"record metrics missing {sorted(missing)}")
    regret = _require_real(
        metrics["best_common_action_regret"], "best_common_action_regret", non_negative=True
    )
    disjoint = metrics["disjoint_best_sets"]
    if not isinstance(disjoint, bool):
        raise ValueError("disjoint_best_sets must be a boolean")
    return regret, disjoint


def aggregate_world_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_world_ids: Sequence[str],
    bootstrap_seed: int,
    bootstrap_samples: int = 2000,
    regret_threshold: float = 0.01,
    min_eligible_worlds: int = 8,
    min_disjoint_worlds: int = 3,
) -> dict[str, Any]:
    """Aggregate controlled-pair metrics with worlds as the sampling unit.

    Repeated pairs in one world are averaged before bootstrap resampling, which
    prevents pair pseudoreplication.  Worlds with no accepted pair remain
    explicitly uncertain rather than receiving a fabricated zero regret.
    """
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    if isinstance(expected_world_ids, (str, bytes)) or not isinstance(expected_world_ids, Sequence):
        raise ValueError("expected_world_ids must be a sequence")
    expected = tuple(expected_world_ids)
    if not expected or any(not isinstance(world_id, str) or not world_id for world_id in expected):
        raise ValueError("expected_world_ids must contain non-empty strings")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_world_ids must be unique")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    samples = _require_positive_int(bootstrap_samples, "bootstrap_samples")
    threshold = _require_real(regret_threshold, "regret_threshold", non_negative=True)
    minimum_worlds = _require_positive_int(min_eligible_worlds, "min_eligible_worlds")
    minimum_disjoint = _require_positive_int(min_disjoint_worlds, "min_disjoint_worlds")

    grouped: dict[str, list[tuple[float, bool]]] = {world_id: [] for world_id in expected}
    seen_pairs: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        if set(("world_id", "pair_id", "metrics")).difference(record):
            raise ValueError(f"record {index} missing world_id, pair_id, or metrics")
        world_id, pair_id = record["world_id"], record["pair_id"]
        if not isinstance(world_id, str) or not world_id:
            raise ValueError("world_id must be a non-empty string")
        if world_id not in grouped:
            raise ValueError(f"unknown world_id: {world_id}")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("pair_id must be a non-empty string")
        key = (world_id, pair_id)
        if key in seen_pairs:
            raise ValueError(f"duplicate pair_id {pair_id!r} for world {world_id!r}")
        seen_pairs.add(key)
        grouped[world_id].append(_validate_metric_record(record["metrics"]))

    world_rows: list[dict[str, Any]] = []
    eligible_means: list[float] = []
    disjoint_worlds = 0
    disjoint_pairs = 0
    for world_id in expected:
        pairs = grouped[world_id]
        if not pairs:
            world_rows.append(
                {
                    "world_id": world_id,
                    "accepted_pair_count": 0,
                    "mean_best_common_action_regret": None,
                    "any_disjoint_best_sets": False,
                    "uncertain": True,
                }
            )
            continue
        mean_regret = float(np.mean(np.asarray([pair[0] for pair in pairs], dtype=np.float64)))
        any_disjoint = any(pair[1] for pair in pairs)
        eligible_means.append(mean_regret)
        disjoint_worlds += int(any_disjoint)
        disjoint_pairs += sum(int(pair[1]) for pair in pairs)
        world_rows.append(
            {
                "world_id": world_id,
                "accepted_pair_count": len(pairs),
                "mean_best_common_action_regret": mean_regret,
                "any_disjoint_best_sets": any_disjoint,
                "uncertain": False,
            }
        )

    eligible_count = len(eligible_means)
    pair_count = len(seen_pairs)
    if eligible_count:
        means = np.asarray(eligible_means, dtype=np.float64)
        draws = np.random.default_rng(bootstrap_seed).choice(
            means, size=(samples, eligible_count), replace=True
        )
        lower_bound = float(np.quantile(draws.mean(axis=1), 0.05))
    else:
        lower_bound = None
    decision = (
        "ADVANCE_INFORMATION_DIAGNOSTIC"
        if lower_bound is not None
        and eligible_count >= minimum_worlds
        and disjoint_worlds >= minimum_disjoint
        and lower_bound > threshold
        else "INCONCLUSIVE_NOT_ADVANCED"
    )
    return {
        "metric_scope": "conditional_counterfactual_action_value_disagreement",
        "not_claimed": ["natural_alias_prevalence", "optimal_q_values", "model_promotion"],
        "expected_world_count": len(expected),
        "eligible_world_count": eligible_count,
        "uncertain_world_count": len(expected) - eligible_count,
        "accepted_pair_count": pair_count,
        "disjoint_world_count": disjoint_worlds,
        "disjoint_pair_count": disjoint_pairs,
        "disjoint_world_rate": disjoint_worlds / eligible_count if eligible_count else None,
        "disjoint_pair_rate": disjoint_pairs / pair_count if pair_count else None,
        "world_rows": world_rows,
        "best_common_action_regret_l95": lower_bound,
        "bootstrap": {
            "seed": bootstrap_seed,
            "samples": samples,
            "quantile": 0.05,
            "sampling_unit": "eligible_world_pair_mean",
        },
        "decision_settings": {
            "regret_threshold": threshold,
            "min_eligible_worlds": minimum_worlds,
            "min_disjoint_worlds": minimum_disjoint,
            "strict_lower_bound_required": True,
        },
        "decision": decision,
    }
