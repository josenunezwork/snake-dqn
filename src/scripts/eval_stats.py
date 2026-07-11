"""Shared paired-evaluation statistics for the promotion gate (blueprint §5.1-§5.2).

Home of the math that ``tournament_eval.py`` (and, eventually, other eval
harnesses) rely on:

* ``mass_integral``     — the headline metric: mean per-frame mass over the TOTAL
                          episode horizon, where dead frames contribute 0. Dying
                          rich no longer outranks surviving (the alive-conditioned
                          ``mean_mass`` bug this replaces).
* ``paired_stats``      — per-seed candidate-minus-baseline deltas with a
                          t-distribution 95% CI and wins/N (ported from the
                          correct math in ``ensemble_eval.py``).
* ``recommended_seed_count`` — pilot-mode power analysis: seeds needed to detect
                          a minimum effect at alpha=0.05 (two-sided), power=0.8.

No SciPy dependency: two-sided 97.5% and one-sided 80% Student-t quantiles are
table-driven (exact to 4 decimals for the tabulated dfs, conservative — next
lower df — in between, normal in the limit).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

# Student-t quantiles, indexed by degrees of freedom. Between tabulated dfs the
# largest tabulated df <= the requested df is used (a slightly WIDER interval,
# i.e. conservative). Beyond the table the normal quantile applies.
_T_975: Dict[int, float] = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
    6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
    11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
    16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
    21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595,
    26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
    40: 2.0211, 50: 2.0086, 60: 2.0003, 80: 1.9901, 100: 1.9840, 120: 1.9799,
}  # fmt: skip
_T_80: Dict[int, float] = {
    1: 1.3764, 2: 1.0607, 3: 0.9785, 4: 0.9410, 5: 0.9195,
    6: 0.9057, 7: 0.8960, 8: 0.8889, 9: 0.8834, 10: 0.8791,
    11: 0.8755, 12: 0.8726, 13: 0.8702, 14: 0.8681, 15: 0.8662,
    16: 0.8647, 17: 0.8633, 18: 0.8620, 19: 0.8610, 20: 0.8600,
    21: 0.8591, 22: 0.8583, 23: 0.8575, 24: 0.8569, 25: 0.8562,
    26: 0.8557, 27: 0.8551, 28: 0.8546, 29: 0.8542, 30: 0.8538,
    40: 0.8507, 50: 0.8489, 60: 0.8477, 80: 0.8461, 100: 0.8452, 120: 0.8446,
}  # fmt: skip
_Z_975 = 1.9600
_Z_80 = 0.8416


def _t_quantile(table: Dict[int, float], normal_limit: float, df: int) -> float:
    """Look up a Student-t quantile for ``df`` degrees of freedom.

    Args:
        table: Quantile table keyed by degrees of freedom.
        normal_limit: The normal quantile used beyond the table.
        df: Degrees of freedom (values < 1 are clamped to 1).

    Returns:
        The quantile for the largest tabulated df <= ``df`` (conservative), or
        ``normal_limit`` past the end of the table.
    """
    df = max(int(df), 1)
    if df > max(table):
        return normal_limit
    key = max(k for k in table if k <= df)
    return table[key]


def t_critical_975(df: int) -> float:
    """Two-sided 95% (upper 97.5%) Student-t critical value for ``df``."""
    return _t_quantile(_T_975, _Z_975, df)


def t_critical_80(df: int) -> float:
    """One-sided 80% (power) Student-t critical value for ``df``."""
    return _t_quantile(_T_80, _Z_80, df)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 when fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def ci95_halfwidth(values: Sequence[float]) -> float:
    """Half-width of the t-distribution 95% CI on the mean of ``values``.

    Returns 0.0 when fewer than two values (a single sample has no CI).
    """
    n = len(values)
    if n < 2:
        return 0.0
    return t_critical_975(n - 1) * sample_std(values) / math.sqrt(n)


def mass_integral(alive_mass_sum: float, total_frames: int) -> float:
    """Headline metric: mean per-frame mass over the TOTAL episode horizon.

    ``alive_mass_sum`` is the sum of the hero's mass over the frames it was
    alive; dead frames contribute 0 mass but STILL count in the denominator,
    so an early death drags the score down even if the snake died rich.

    Args:
        alive_mass_sum: Sum of per-frame mass while alive.
        total_frames: Total frames in the episode horizon (must be > 0).

    Returns:
        alive_mass_sum / total_frames.

    Raises:
        ValueError: If ``total_frames`` is not positive.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    return float(alive_mass_sum) / float(total_frames)


def paired_stats(candidate: Sequence[float], baseline: Sequence[float]) -> Dict[str, Any]:
    """Paired per-seed delta analysis (candidate - baseline) with a 95% t-CI.

    This is the analysis ``ensemble_eval.py`` got right and the old
    ``tournament_eval.py`` discarded: both sequences must come from the SAME
    ordered seed list so element i of each is a common-random-numbers pair.

    Args:
        candidate: Per-seed metric values for the candidate.
        baseline: Per-seed metric values for the baseline (same seed order).

    Returns:
        Dict with ``n``, ``deltas``, ``mean_delta``, ``ci95`` (half-width),
        ``ci_low``/``ci_high``, ``wins`` (deltas > 0), ``losses``,
        ``significant`` (CI excludes 0 in the candidate's favor) and
        ``regression`` (CI entirely below 0).

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    if len(candidate) != len(baseline):
        raise ValueError(
            f"paired analysis needs equal-length runs, got {len(candidate)} vs {len(baseline)}"
        )
    if not candidate:
        raise ValueError("paired analysis needs at least one seed")

    deltas: List[float] = [float(c) - float(b) for c, b in zip(candidate, baseline)]
    n = len(deltas)
    mean_delta = mean(deltas)
    ci = ci95_halfwidth(deltas)
    significant = n >= 2 and (mean_delta - ci) > 0
    regression = n >= 2 and (mean_delta + ci) < 0
    return {
        "n": n,
        "deltas": deltas,
        "mean_delta": mean_delta,
        "ci95": ci,
        "ci_low": mean_delta - ci,
        "ci_high": mean_delta + ci,
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "significant": significant,
        "regression": regression,
    }


def recommended_seed_count(
    delta_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    minimum: int = 2,
) -> int:
    """Seeds needed for a paired t-test to detect ``mde`` (two-sided).

    Iterates n = ceil(((t_{1-alpha/2, n-1} + t_{power, n-1}) * sd / mde)^2)
    starting from the normal approximation until it stabilizes.

    Args:
        delta_std: Estimated standard deviation of the per-seed paired deltas.
        mde: Minimum detectable effect (same units as the metric; must be > 0).
        alpha: Two-sided significance level. Only 0.05 is supported (the
            quantile tables are fixed).
        power: Target power. Only 0.8 is supported.
        minimum: Floor on the returned count.

    Returns:
        Recommended number of paired seeds (>= ``minimum``).

    Raises:
        ValueError: If ``mde`` <= 0, or alpha/power are unsupported.
    """
    if mde <= 0:
        raise ValueError(f"mde must be positive, got {mde}")
    if alpha != 0.05 or power != 0.8:
        raise ValueError("only alpha=0.05, power=0.8 are supported (fixed t-tables)")
    if delta_std <= 0:
        return minimum

    ratio = delta_std / mde
    n = max(int(math.ceil(((_Z_975 + _Z_80) * ratio) ** 2)), minimum)
    for _ in range(10):
        t_sum = t_critical_975(n - 1) + t_critical_80(n - 1)
        new_n = max(int(math.ceil((t_sum * ratio) ** 2)), minimum)
        if new_n == n:
            break
        n = new_n
    return n
