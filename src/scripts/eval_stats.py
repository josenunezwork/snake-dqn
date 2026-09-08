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

The legacy diagnostic helpers retain their table-driven behavior.  Strict E2
decisions use the dependency-free Student-t survival/inverse implementation at
the end of this file, plus raw paired deltas.  The strict APIs deliberately do
not accept aggregate world dictionaries: pairing identity is a runtime concern
and this module receives only already-validated finite deltas.
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

# The strict pilot is a finite-sample t approximation, versioned so a receipt
# says precisely which sizing convention froze its final-world count.  It sizes
# each superiority mix at alpha / 3 and describes *marginal* 80% power only;
# Holm's compound decision has no claimed joint-power guarantee.
PAIRED_DELTA_PILOT_METHOD = "paired-delta-t-marginal-v1"
PAIRED_DELTA_PILOT_POWER = 0.8
PAIRED_DELTA_PILOT_FAMILY_ALPHA = 0.05
PAIRED_DELTA_PILOT_MIXES = 3
PAIRED_DELTA_PILOT_FLOOR = 40


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


# Strict E2 numerical primitives.  These are intentionally local rather than a
# SciPy dependency: promotion receipts must be reproducible in the project venv.
_FPMIN = 1e-300
_BETA_EPS = 3e-14
_BETA_MAX_ITER = 20_000


def _positive_real(value: float, name: str) -> float:
    """Return a finite positive real, rejecting bools and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _finite_deltas(deltas: Sequence[float]) -> List[float]:
    """Copy finite numeric deltas, rejecting bool and nonfinite input."""
    parsed: List[float] = []
    for index, value in enumerate(deltas):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"delta at index {index} must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"delta at index {index} must be finite")
        parsed.append(value)
    return parsed


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction with guarded Lentz steps."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    if not math.isfinite(h):
        raise ArithmeticError("non-finite incomplete-beta continued fraction")
    for m in range(1, _BETA_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < _FPMIN:
            d = _FPMIN
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < _FPMIN:
            d = _FPMIN
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if not all(math.isfinite(value) for value in (aa, c, d, delta, h)):
            raise ArithmeticError("non-finite incomplete-beta continued fraction")
        if abs(delta - 1.0) <= _BETA_EPS:
            return h
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    """Return regularized incomplete beta ``I_x(a, b)`` using symmetry reduction."""
    a, b = _positive_real(a, "a"), _positive_real(b, "b")
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError("x must be a real number")
    x = float(x)
    if math.isnan(x) or not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    try:
        front = math.exp(log_front)
    except OverflowError as error:
        raise ArithmeticError("incomplete-beta prefactor overflow") from error
    if not math.isfinite(front):
        raise ArithmeticError("non-finite incomplete-beta prefactor")
    threshold = (a + 1.0) / (a + b + 2.0)
    if x < threshold:
        value = front * _beta_fraction(a, b, x) / a
    else:
        value = 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b
    if -8.0 * _BETA_EPS <= value <= 1.0 + 8.0 * _BETA_EPS:
        return min(1.0, max(0.0, value))
    raise ArithmeticError(f"regularized beta escaped [0,1]: {value!r}")


def student_t_sf(t: float, df: float) -> float:
    """Return the one-sided Student-t survival probability for finite ``df``."""
    df = _positive_real(df, "df")
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise TypeError("t must be a real number")
    t = float(t)
    if math.isnan(t):
        raise ValueError("t must not be NaN")
    if t == math.inf:
        return 0.0
    if t == -math.inf:
        return 1.0
    if t == 0.0:
        return 0.5
    beta = regularized_beta(df / (df + t * t), df / 2.0, 0.5)
    return beta / 2.0 if t > 0.0 else 1.0 - beta / 2.0


def student_t_isf(probability: float, df: float) -> float:
    """Return positive inverse Student-t survival using a binary64 bracket."""
    df = _positive_real(df, "df")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise TypeError("probability must be a real number")
    probability = float(probability)
    if math.isnan(probability) or not 0.0 < probability < 0.5:
        raise ValueError("probability must be in (0, 0.5)")
    low, high = 0.0, 1.0
    while student_t_sf(high, df) > probability:
        high *= 2.0
        if not math.isfinite(high):
            raise ArithmeticError("could not bracket Student-t inverse")
    for _ in range(4_096):
        if math.nextafter(low, high) == high:
            return high
        midpoint = (low + high) / 2.0
        if midpoint == low or midpoint == high:
            return high
        if student_t_sf(midpoint, df) > probability:
            low = midpoint
        else:
            high = midpoint
    raise ArithmeticError("Student-t inverse bisection did not reach adjacent floats")


def holm(p_values: Sequence[float], alpha: float = 0.05) -> tuple[List[bool], List[float]]:
    """Apply stable Holm step-down and return decisions and adjusted p-values."""
    alpha = _positive_real(alpha, "alpha")
    if alpha > 1.0:
        raise ValueError("alpha must be at most one")
    parsed = []
    for index, value in enumerate(p_values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("p-values must be real numbers")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be finite values in [0,1]")
        parsed.append((value, index))
    ordered = sorted(parsed, key=lambda item: item[0])
    rejected = [False] * len(parsed)
    adjusted = [0.0] * len(parsed)
    running = 0.0
    accepting = True
    total = len(parsed)
    for rank, (value, original_index) in enumerate(ordered):
        if accepting and value <= alpha / (total - rank):
            rejected[original_index] = True
        else:
            accepting = False
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[original_index] = running
    return rejected, adjusted


def _descriptive_ci(deltas: Sequence[float]) -> Dict[str, float | None]:
    """Return the ordinary unadjusted two-sided 95% CI, or null below n=2."""
    count = len(deltas)
    average = mean(deltas)
    if count < 2:
        return {"half_width": None, "low": None, "high": None}
    half_width = student_t_isf(0.025, count - 1) * sample_std(deltas) / math.sqrt(count)
    return {
        "half_width": half_width,
        "low": average - half_width,
        "high": average + half_width,
    }


def paired_delta_test(deltas: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Analyze finite candidate-minus-incumbent deltas with a one-sided t test.

    The returned record is JSON-safe even for zero standard error: its
    ``t_statistic`` is null and ``t_statistic_limit`` records the signed limiting
    value.  A test with fewer than two worlds is invalid and cannot succeed.
    The included ordinary two-sided 95% interval is descriptive only.
    """
    alpha = _positive_real(alpha, "alpha")
    if alpha >= 0.5:
        raise ValueError("alpha must be less than 0.5")
    values = _finite_deltas(deltas)
    count = len(values)
    average = mean(values)
    descriptive = _descriptive_ci(values)
    result: Dict[str, Any] = {
        "n": count,
        "mean_delta": average,
        "sample_std": None,
        "standard_error": None,
        "t_statistic": None,
        "t_statistic_limit": None,
        "p_value": None,
        "critical_value": None,
        "superior": False,
        "valid": count >= 2,
        "descriptive_ci95": descriptive,
    }
    if count < 2:
        result["invalid_reason"] = "at_least_two_paired_deltas_required"
        return result

    std = sample_std(values)
    standard_error = std / math.sqrt(count)
    critical = student_t_isf(alpha, count - 1)
    result.update(
        {
            "sample_std": std,
            "standard_error": standard_error,
            "critical_value": critical,
        }
    )
    if standard_error == 0.0:
        if average > 0.0:
            p_value, limit = 0.0, "positive_infinity"
        elif average < 0.0:
            p_value, limit = 1.0, "negative_infinity"
        else:
            p_value, limit = 0.5, "zero"
        result.update({"p_value": p_value, "t_statistic_limit": limit})
    else:
        statistic = average / standard_error
        result.update({"t_statistic": statistic, "p_value": student_t_sf(statistic, count - 1)})
    result["superior"] = bool(result["p_value"] <= alpha)
    return result


def holm_three_mix_superiority(
    deltas_by_mix: Dict[str, Sequence[float]], alpha: float = 0.05, required_successes: int = 2
) -> Dict[str, Any]:
    """Apply the strict three-mix one-sided superiority rule to raw deltas.

    Exactly three named mixes are required.  Any mix with fewer than two
    observations makes the whole result invalid, instead of silently dropping
    a hypothesis from Holm's family.
    """
    alpha = _positive_real(alpha, "alpha")
    if alpha >= 0.5:
        raise ValueError("alpha must be less than 0.5")
    if isinstance(required_successes, bool) or not isinstance(required_successes, int):
        raise TypeError("required_successes must be an integer")
    if required_successes < 1 or required_successes > 3:
        raise ValueError("required_successes must be in [1, 3]")
    names = list(deltas_by_mix)
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("strict superiority requires exactly three distinct mixes")
    per_mix = {name: paired_delta_test(deltas_by_mix[name], alpha) for name in names}
    invalid = [name for name in names if not per_mix[name]["valid"]]
    output: Dict[str, Any] = {
        "alpha": alpha,
        "required_successes": required_successes,
        "mix_order": names,
        "per_mix": per_mix,
        "rejected": {name: False for name in names},
        "adjusted_p_values": {name: None for name in names},
        "successful_mixes": [],
        "passes": False,
        "valid": not invalid,
    }
    if invalid:
        output["invalid_reason"] = "at_least_two_paired_deltas_required"
        output["invalid_mixes"] = invalid
        return output
    rejected, adjusted = holm([per_mix[name]["p_value"] for name in names], alpha)
    rejected_by_mix = dict(zip(names, rejected))
    adjusted_by_mix = dict(zip(names, adjusted))
    successful = [name for name in names if rejected_by_mix[name]]
    output.update(
        {
            "rejected": rejected_by_mix,
            "adjusted_p_values": adjusted_by_mix,
            "successful_mixes": successful,
            "passes": len(successful) >= required_successes,
        }
    )
    return output


def scripted_noninferiority(
    deltas: Sequence[float], absolute_delta_ni: float, alpha: float = 0.05
) -> Dict[str, Any]:
    """Test scripted noninferiority with a strict one-sided lower-bound rule."""
    alpha = _positive_real(alpha, "alpha")
    margin = _positive_real(absolute_delta_ni, "absolute_delta_ni")
    if alpha >= 0.5:
        raise ValueError("alpha must be less than 0.5")
    paired = paired_delta_test(deltas, alpha)
    output: Dict[str, Any] = {
        "alpha": alpha,
        "absolute_delta_ni": margin,
        "n": paired["n"],
        "mean_delta": paired["mean_delta"],
        "lower_bound": None,
        "critical_value": paired["critical_value"],
        "passes": False,
        "valid": paired["valid"],
    }
    if not paired["valid"]:
        output["invalid_reason"] = paired["invalid_reason"]
        return output
    lower_bound = paired["mean_delta"] - paired["critical_value"] * paired["standard_error"]
    output["lower_bound"] = lower_bound
    # This is intentionally strict: equality to -margin is insufficient.
    output["passes"] = lower_bound > -margin
    return output


def _paired_delta_required_count(delta_std: float, mde: float, alpha: float) -> int:
    """Finite-sample fixed-point t approximation for one marginal test."""
    if delta_std == 0.0:
        return PAIRED_DELTA_PILOT_FLOOR
    ratio = delta_std / mde
    count = PAIRED_DELTA_PILOT_FLOOR
    for _ in range(100):
        critical = student_t_isf(alpha, count - 1) + student_t_isf(
            1.0 - PAIRED_DELTA_PILOT_POWER, count - 1
        )
        next_count = max(PAIRED_DELTA_PILOT_FLOOR, int(math.ceil((critical * ratio) ** 2)))
        if next_count == count:
            return count
        count = next_count
    raise ArithmeticError("paired-delta pilot sizing did not converge")


def paired_delta_pilot_size(
    deltas_by_mix: Dict[str, Sequence[float]], mde_by_mix: Dict[str, float]
) -> Dict[str, Any]:
    """Freeze strict final-world sizing from paired deltas across exactly three mixes.

    This is a per-mix 80% marginal-power approximation at conservative
    ``0.05 / 3`` planning alpha.  It is not a joint-power claim for Holm plus
    scripted noninferiority.  All three mixes must provide at least two raw
    paired pilot deltas; baseline-only variance is never accepted here.
    """
    names = list(deltas_by_mix)
    if len(names) != 3 or len(set(names)) != 3 or set(names) != set(mde_by_mix):
        raise ValueError("pilot sizing needs MDEs and raw deltas for exactly the same three mixes")
    planning_alpha = PAIRED_DELTA_PILOT_FAMILY_ALPHA / PAIRED_DELTA_PILOT_MIXES
    per_mix: Dict[str, Any] = {}
    required_counts: List[int] = []
    for name in names:
        values = _finite_deltas(deltas_by_mix[name])
        if len(values) < 2:
            raise ValueError(f"pilot mix {name!r} needs at least two paired deltas")
        mde = _positive_real(mde_by_mix[name], f"mde for mix {name!r}")
        delta_std = sample_std(values)
        required = _paired_delta_required_count(delta_std, mde, planning_alpha)
        per_mix[name] = {
            "n": len(values),
            "paired_delta_std": delta_std,
            "mde": mde,
            "recommended_n": required,
        }
        required_counts.append(required)
    required_n = max(PAIRED_DELTA_PILOT_FLOOR, max(required_counts))
    return {
        "method": PAIRED_DELTA_PILOT_METHOD,
        "family_alpha": PAIRED_DELTA_PILOT_FAMILY_ALPHA,
        "planning_alpha_per_mix": planning_alpha,
        "declared_per_mix_power": PAIRED_DELTA_PILOT_POWER,
        "joint_power_claim": None,
        "minimum_final_worlds": PAIRED_DELTA_PILOT_FLOOR,
        "mix_order": names,
        "per_mix": per_mix,
        "required_final_worlds": required_n,
    }


def strict_promotion_decision(
    deltas_by_mix: Dict[str, Sequence[float]],
    scripted_mix: str,
    absolute_delta_ni: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Join strict Holm superiority and separate scripted noninferiority.

    The combined interval is descriptive only: it averages the three raw
    deltas at each matched world position and never contributes to ``passes``.
    Runtime validates world identities before it calls this pure numeric helper.
    """
    superiority = holm_three_mix_superiority(deltas_by_mix, alpha=alpha)
    if scripted_mix not in deltas_by_mix:
        raise ValueError(f"scripted mix {scripted_mix!r} is not one of the three strict mixes")
    noninferiority = scripted_noninferiority(
        deltas_by_mix[scripted_mix], absolute_delta_ni, alpha=alpha
    )
    names = list(deltas_by_mix)
    values = [_finite_deltas(deltas_by_mix[name]) for name in names]
    equal_lengths = len({len(item) for item in values}) == 1
    descriptive: Dict[str, Any]
    if equal_lengths:
        combined = [mean(row) for row in zip(*values)] if values and values[0] else []
        descriptive = paired_delta_test(combined, alpha)["descriptive_ci95"]
        descriptive["n"] = len(combined)
    else:
        descriptive = {"n": None, "half_width": None, "low": None, "high": None}
    valid = bool(superiority["valid"] and noninferiority["valid"] and equal_lengths)
    return {
        "decision_method": "strict-promotion-v1",
        "superiority": superiority,
        "scripted_noninferiority": noninferiority,
        "descriptive_combined_ci95": descriptive,
        "valid": valid,
        "passes": bool(valid and superiority["passes"] and noninferiority["passes"]),
    }
