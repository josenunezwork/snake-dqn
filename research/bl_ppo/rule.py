"""E9 pre-data fixed-controller funding family; no release authority.

E2 and E6 each reserve one-sided family alpha .05, Holm over the two named
slots, including an unavailable slot as p=1. Equal mix weighting is invariant.
"""

from __future__ import annotations

import math
from statistics import mean, variance
from typing import Any

E2_ALPHA = 0.05
E6_ALPHA = 0.05
ARMS = ("R", "F")
RULE = {
    "schema": "bl-ppo-e9-rule/v1",
    "unit": "paired world",
    "mix_weight": 1 / 3,
    "one_look": True,
    "alpha_routes": {"E2": E2_ALPHA, "E6": E6_ALPHA},
    "holm_slots": list(ARMS),
    "E2_GO": {
        "mass_point_min": 10,
        "LB90_strictly_positive": True,
        "holm_positive": True,
        "scripted_episode_delta_min": 0,
        "pooled_alive_LB95_min_exclusive": -0.05,
        "each_mix_alive_LB95_min_exclusive": -0.075,
        "each_mix_absolute_alive_min": 0.30,
    },
    "E2_KILL": "tripwire or mass UB90 <= 0; integrity takes precedence",
    "E6_funding": {
        "early_LB90_positive": True,
        "holm_positive": True,
        "ordinary_point_min": -2,
        "critic_qualified_required": True,
    },
    "family_funding": "any full E2 GO or full Holm-qualified E6 NEEDS_SCALE",
    "family_null_ceiling": 0.10,
    "funding_probability_at_true_40_floor": 0.60,
    "allowed_counts": [
        {"scripted": 96, "frozen": 32, "mixed": 32},
        {"scripted": 128, "frozen": 48, "mixed": 48},
    ],
    "no_release_authority": True,
}


def holm_adjusted(pvalues: dict[str, float]) -> dict[str, float]:
    """Return step-down adjusted p-values, preserving all reserved arm slots."""
    if set(pvalues) != set(ARMS):
        raise ValueError("both R/F reserved slots required; unavailable slot uses p=1")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("p-values must be finite probabilities")
    ordered = sorted(pvalues, key=lambda arm: (pvalues[arm], arm))
    result, previous = {}, 0.0
    for index, arm in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * pvalues[arm]))
        result[arm] = previous
    return result


def holm_rejections(pvalues: dict[str, float], alpha: float = E2_ALPHA) -> dict[str, bool]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be inside (0,1)")
    return {arm: value <= alpha for arm, value in holm_adjusted(pvalues).items()}


def mass_pvalue(stats: dict[str, Any]) -> float:
    """One-sided Student-t p for a positive equal-mix paired mean."""
    from scipy.stats import t

    if stats["se"] == 0:
        return 0.0 if stats["estimate"] > 0 else 1.0
    return float(t.sf(stats["estimate"] / stats["se"], stats["df"]))


def paired_efficacy(groups: dict[str, list[float]]) -> dict[str, Any]:
    """Stratified Welch inference, sum(s_m²/n_m)/9, never row weighted."""
    from scipy.stats import t

    if set(groups) != {"scripted", "frozen", "mixed"}:
        raise ValueError("all three mixes required")
    if any(len(x) < 2 or any(not math.isfinite(v) for v in x) for x in groups.values()):
        raise ValueError("finite paired worlds required")
    terms = [variance(x) / len(x) / 9 for x in groups.values()]
    var = sum(terms)
    denom = sum(v * v / (len(x) - 1) for v, x in zip(terms, groups.values()))
    df = var * var / denom if denom else None
    se = math.sqrt(var)
    estimate = mean(mean(x) for x in groups.values())
    width = float(t.isf(0.1, df)) * se if df else 0.0
    result = {
        "estimate": estimate,
        "se": se,
        "df": df,
        "lower": estimate - width,
        "upper": estimate + width,
    }
    return {**result, "pvalue": mass_pvalue(result)}
