"""Screen statistics: pooled seed effects, learning curves, gate planning, futility.

Opt-in analysis helpers for development-only screens (portfolio review 2026-09-26,
recommendations 7.1 and 7.4).  Nothing here is wired into the strict promotion
gate or ``src/scripts/eval_stats.py``; callers import these functions explicitly.

Four tools, pure numpy/stdlib:

1. :func:`random_effects_pool` -- DerSimonian-Laird random-effects pooling of
   per-seed paired effects with a confidence interval, Cochran's Q, tau^2, I^2
   and an optional prediction interval.  :func:`per_seed_effects` turns raw
   per-world paired deltas into the ``(effect, variance)`` pairs it consumes.
   :func:`crossed_seed_world_mean` handles the common design where every seed
   is evaluated on the *same* world bank (seed effects are then correlated, so
   DerSimonian-Laird's independence assumption does not hold).
   :func:`stratified_mean_of_means` is the serving-time pooled estimate
   (equal-weight mean of per-mix mean deltas, Welch-Satterthwaite CI).
2. :func:`curve_auc`, :func:`last_k_mean` and :func:`learning_curve_contrast`
   -- area under the checkpoint curve and last-k-checkpoint means, contrasted
   across seeds with paired (or Welch) t intervals.
3. :func:`gate_pass_probabilities`, :func:`gate_pass_probabilities_for_effect`
   and :func:`seeds_for_pooled_power` -- planning: how often does a true
   effect pass an all-seeds conjunction gate versus a pooled gate?
4. :func:`futility_plan`, :func:`evaluate_futility_look` and
   :func:`run_futility_sequence` -- a one-sided group-sequential "clear loser"
   stop with pre-declared looks and Lan-DeMets O'Brien-Fleming-type alpha
   spending.  Boundaries are computed exactly (numerical recursive
   integration), not from tables.

Sign convention: every effect is ``candidate - baseline``; positive favours the
candidate.  All intervals are two-sided at ``confidence`` unless stated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from src.scripts.eval_stats import student_t_isf, student_t_sf

__all__ = [
    "FutilityPlan",
    "chi2_sf",
    "crossed_seed_world_mean",
    "curve_auc",
    "evaluate_futility_look",
    "futility_plan",
    "futility_stop_probability",
    "gate_pass_probabilities",
    "gate_pass_probabilities_for_effect",
    "last_k_mean",
    "learning_curve_contrast",
    "obf_spending",
    "per_seed_effects",
    "random_effects_pool",
    "run_futility_sequence",
    "seeds_for_pooled_power",
    "stratified_mean_of_means",
]

_NORMAL = NormalDist()
_GAMMA_EPS = 1e-15
_GAMMA_MAX_ITER = 10_000
_FPMIN = 1e-300


# --------------------------------------------------------------------------- validation


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _probability(value: object, name: str) -> float:
    parsed = _real(value, name)
    if not 0.0 < parsed < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return parsed


def _half_probability(value: object, name: str) -> float:
    parsed = _probability(value, name)
    if parsed >= 0.5:
        raise ValueError(f"{name} must be < 0.5")
    return parsed


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _finite_array(values: object, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


# --------------------------------------------------------------------------- distributions


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    return _NORMAL.inv_cdf(p)


_erfc_vec = np.vectorize(math.erfc, otypes=[float])


def _norm_sf_array(x: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc_vec(np.asarray(x, dtype=float) / math.sqrt(2.0))


def _t_two_sided_critical(confidence: float, df: float) -> float:
    """Two-sided critical value; ``df=inf`` gives the normal quantile."""
    tail = (1.0 - confidence) / 2.0
    if math.isinf(df):
        return _norm_ppf(1.0 - tail)
    return student_t_isf(tail, df)


def _lower_gamma_series(a: float, x: float) -> float:
    total = term = 1.0 / a
    ap = a
    for _ in range(_GAMMA_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _GAMMA_EPS:
            return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    raise ArithmeticError("incomplete-gamma series did not converge")


def _upper_gamma_fraction(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _GAMMA_MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _GAMMA_EPS:
            return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    raise ArithmeticError("incomplete-gamma continued fraction did not converge")


def chi2_sf(statistic: float, df: float) -> float:
    """Chi-square survival function ``P(X >= statistic)`` via the regularized gamma Q."""
    statistic = _real(statistic, "statistic")
    df = _real(df, "df")
    if df <= 0.0:
        raise ValueError("df must be positive")
    if statistic <= 0.0:
        return 1.0
    a, x = df / 2.0, statistic / 2.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _lower_gamma_series(a, x))
    return min(1.0, _upper_gamma_fraction(a, x))


# --------------------------------------------------------------------------- 1. pooling

_CI_METHODS = ("z", "hksj", "hksj_floor")


def per_seed_effects(deltas_by_seed: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    """Summarise per-world paired deltas into one ``(effect, variance)`` per seed.

    ``deltas_by_seed[s]`` holds the paired ``candidate - baseline`` deltas of seed
    ``s`` (one per evaluation world).  The effect is their mean and the variance
    is the squared standard error ``s^2 / n``.  Each seed needs >= 2 worlds.
    """
    if len(deltas_by_seed) == 0:
        raise ValueError("deltas_by_seed must not be empty")
    effects: list[float] = []
    variances: list[float] = []
    counts: list[int] = []
    for index, deltas in enumerate(deltas_by_seed):
        array = _finite_array(deltas, f"deltas_by_seed[{index}]", 1)
        if array.size < 2:
            raise ValueError(f"deltas_by_seed[{index}] needs at least 2 worlds")
        effects.append(float(array.mean()))
        variances.append(float(array.var(ddof=1) / array.size))
        counts.append(int(array.size))
    return {"effects": effects, "variances": variances, "counts": counts}


def random_effects_pool(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    confidence: float = 0.95,
    ci_method: str = "hksj_floor",
) -> dict[str, Any]:
    """DerSimonian-Laird random-effects pool of independent per-seed effects.

    Model: ``y_i = mu + u_i + e_i`` with ``u_i ~ N(0, tau^2)`` (between-seed
    heterogeneity) and ``e_i ~ N(0, v_i)`` (within-seed sampling error, ``v_i``
    treated as known).  DerSimonian-Laird moment estimates::

        w_i = 1 / v_i,  mu_FE = sum(w y) / sum(w),  Q = sum(w (y - mu_FE)^2)
        tau^2 = max(0, (Q - (k - 1)) / (sum(w) - sum(w^2) / sum(w)))
        w*_i = 1 / (v_i + tau^2),  mu = sum(w* y) / sum(w*),  SE = sqrt(1 / sum(w*))
        I^2 = max(0, (Q - (k - 1)) / Q)

    ``ci_method``:

    * ``"z"`` -- textbook DL interval ``mu +/- z * SE``.  Anti-conservative with
      few seeds (k <= 5 is typical here); kept for reference.
    * ``"hksj"`` -- Hartung-Knapp-Sidik-Jonkman: ``SE_HK = sqrt(q / sum(w*))``
      with ``q = sum(w* (y - mu)^2) / (k - 1)`` and a t critical value on
      ``k - 1`` df.
    * ``"hksj_floor"`` (default) -- HKSJ with ``q`` floored at 1, so the
      interval is never narrower than the DL standard error allows.

    The ``p_value`` is two-sided for ``H0: mu = 0`` under the same reference
    distribution.  A Higgins-Thompson-Spiegelhalter prediction interval for a
    *new* seed's effect (``mu +/- t_{k-2} sqrt(tau^2 + SE^2)``) is reported for
    ``k >= 3``.

    Caveat: seeds must be independent.  If every seed was evaluated on the
    same worlds, the per-seed effects share world noise; use
    :func:`crossed_seed_world_mean` instead.
    """
    y = _finite_array(effects, "effects", 1)
    v = _finite_array(variances, "variances", 1)
    if y.shape != v.shape:
        raise ValueError("effects and variances must have the same length")
    k = int(y.size)
    if k < 2:
        raise ValueError("random-effects pooling needs at least 2 seeds")
    if np.any(v <= 0.0):
        raise ValueError("variances must be strictly positive")
    confidence = _probability(confidence, "confidence")
    if ci_method not in _CI_METHODS:
        raise ValueError(f"ci_method must be one of {_CI_METHODS}")

    w = 1.0 / v
    sum_w = float(w.sum())
    mu_fixed = float((w * y).sum() / sum_w)
    q_stat = float((w * (y - mu_fixed) ** 2).sum())
    df_q = k - 1
    c_term = sum_w - float((w**2).sum()) / sum_w
    tau2 = max(0.0, (q_stat - df_q) / c_term) if c_term > 0.0 else 0.0
    i2 = max(0.0, (q_stat - df_q) / q_stat) if q_stat > 0.0 else 0.0

    w_star = 1.0 / (v + tau2)
    sum_w_star = float(w_star.sum())
    mu = float((w_star * y).sum() / sum_w_star)
    se_dl = math.sqrt(1.0 / sum_w_star)

    if ci_method == "z":
        se, ci_df = se_dl, math.inf
    else:
        q_hk = float((w_star * (y - mu) ** 2).sum()) / df_q
        if ci_method == "hksj_floor":
            q_hk = max(q_hk, 1.0)
        se, ci_df = math.sqrt(q_hk / sum_w_star), float(df_q)
    critical = _t_two_sided_critical(confidence, ci_df)
    if se > 0.0:
        statistic = mu / se
        tail = (
            _norm_sf(abs(statistic)) if math.isinf(ci_df) else student_t_sf(abs(statistic), ci_df)
        )
        p_value = min(1.0, 2.0 * tail)
    else:
        statistic, p_value = (math.copysign(math.inf, mu), 0.0) if mu != 0.0 else (0.0, 1.0)

    prediction = None
    if k >= 3:
        half = student_t_isf((1.0 - confidence) / 2.0, k - 2) * math.sqrt(tau2 + se_dl**2)
        prediction = [mu - half, mu + half]

    return {
        "k": k,
        "mu": mu,
        "se": se,
        "se_dl": se_dl,
        "ci": [mu - critical * se, mu + critical * se],
        "ci_method": ci_method,
        "ci_df": ci_df,
        "critical": critical,
        "statistic": statistic,
        "p_value": p_value,
        "mu_fixed": mu_fixed,
        "se_fixed": math.sqrt(1.0 / sum_w),
        "q": q_stat,
        "q_df": df_q,
        "q_p_value": chi2_sf(q_stat, df_q),
        "tau2": tau2,
        "i2": i2,
        "weights": (w_star / sum_w_star).tolist(),
        "prediction_interval": prediction,
        "confidence": confidence,
    }


def crossed_seed_world_mean(deltas: object, *, confidence: float = 0.95) -> dict[str, Any]:
    """Grand mean of a seeds x worlds delta matrix with crossed random effects.

    Use this when every training seed is evaluated on the *same* world bank.
    Model ``d_sw = mu + a_s + b_w + e_sw`` (two-way random effects, no
    replication).  With mean squares ``MS_s``, ``MS_w``, ``MS_e`` the unbiased
    variance of the grand mean is ``(MS_s + MS_w - MS_e) / (S * W)``, i.e.
    ``sigma_s^2 / S + sigma_w^2 / W + sigma_e^2 / (S * W)``.  Negative
    component estimates are truncated at zero (which drops the corresponding
    mean square from the combination) and the CI uses Satterthwaite df.
    """
    matrix = _finite_array(deltas, "deltas", 2)
    n_seeds, n_worlds = matrix.shape
    if n_seeds < 2 or n_worlds < 2:
        raise ValueError("deltas needs at least 2 seeds and 2 worlds")
    confidence = _probability(confidence, "confidence")
    grand = float(matrix.mean())
    seed_means = matrix.mean(axis=1)
    world_means = matrix.mean(axis=0)
    df_s, df_w = n_seeds - 1, n_worlds - 1
    df_e = df_s * df_w
    ms_s = float(n_worlds * ((seed_means - grand) ** 2).sum() / df_s)
    ms_w = float(n_seeds * ((world_means - grand) ** 2).sum() / df_w)
    resid = matrix - seed_means[:, None] - world_means[None, :] + grand
    ms_e = float((resid**2).sum() / df_e)

    keep_s, keep_w = ms_s > ms_e, ms_w > ms_e
    coeffs = (float(keep_s), float(keep_w), 1.0 - float(keep_s) - float(keep_w))
    mean_squares = (ms_s, ms_w, ms_e)
    dfs = (df_s, df_w, df_e)
    cells = float(n_seeds * n_worlds)
    variance = sum(c * m for c, m in zip(coeffs, mean_squares)) / cells
    denom = sum((c * m) ** 2 / d for c, m, d in zip(coeffs, mean_squares, dfs))
    numer = (variance * cells) ** 2
    df = numer / denom if denom > 0.0 else math.inf
    se = math.sqrt(max(variance, 0.0))
    critical = _t_two_sided_critical(confidence, df)
    return {
        "mean": grand,
        "se": se,
        "df": df,
        "ci": [grand - critical * se, grand + critical * se],
        "sigma2_seed": max(0.0, (ms_s - ms_e) / n_worlds),
        "sigma2_world": max(0.0, (ms_w - ms_e) / n_seeds),
        "sigma2_residual": ms_e,
        "ms_seed": ms_s,
        "ms_world": ms_w,
        "ms_residual": ms_e,
        "n_seeds": n_seeds,
        "n_worlds": n_worlds,
        "confidence": confidence,
    }


def stratified_mean_of_means(
    deltas_by_stratum: Mapping[str, Sequence[float]], *, confidence: float = 0.90
) -> dict[str, Any]:
    """Equal-weight mean of per-stratum mean deltas with a Welch-Satterthwaite t CI.

    The serving-time pooled estimate of the governance note (strata = opponent
    mixes, unit = world): ``mu = (1/M) sum_m mean_m`` with
    ``SE^2 = (1/M^2) sum_m s_m^2 / n_m`` and Satterthwaite df.  Strata are
    treated as independent samples; when strata share world seeds this ignores
    their covariance (see :func:`crossed_seed_world_mean` for that design).
    Also reports whether the upper bound is below zero (the governance
    clear-loser filter at 90%) and which strata have an upper bound below zero.
    Each stratum needs >= 2 deltas.
    """
    if not deltas_by_stratum:
        raise ValueError("deltas_by_stratum must not be empty")
    confidence = _probability(confidence, "confidence")
    names = list(deltas_by_stratum)
    means, terms, dfs, per_stratum = [], [], [], {}
    for name in names:
        array = _finite_array(deltas_by_stratum[name], f"deltas_by_stratum[{name!r}]", 1)
        if array.size < 2:
            raise ValueError(f"stratum {name!r} needs at least 2 deltas")
        n = int(array.size)
        mean = float(array.mean())
        var_mean = float(array.var(ddof=1)) / n
        stratum_df = float(n - 1)
        crit = _t_two_sided_critical(confidence, stratum_df)
        half = crit * math.sqrt(var_mean)
        per_stratum[name] = {"n": n, "mean": mean, "se": math.sqrt(var_mean)}
        per_stratum[name]["ci"] = [mean - half, mean + half]
        per_stratum[name]["upper_below_zero"] = bool(mean + half < 0.0)
        means.append(mean)
        terms.append(var_mean)
        dfs.append(stratum_df)
    m = len(names)
    mu = float(sum(means) / m)
    variance = sum(terms) / (m * m)
    se = math.sqrt(variance)
    denom = sum((t / (m * m)) ** 2 / d for t, d in zip(terms, dfs))
    df = variance**2 / denom if denom > 0.0 else math.inf
    critical = _t_two_sided_critical(confidence, df)
    ci = [mu - critical * se, mu + critical * se]
    return {
        "strata": names,
        "mean_of_means": mu,
        "se": se,
        "df": df,
        "ci": ci,
        "confidence": confidence,
        "upper_below_zero": bool(ci[1] < 0.0),
        "strata_upper_below_zero": [n for n in names if per_stratum[n]["upper_below_zero"]],
        "per_stratum": per_stratum,
    }


# --------------------------------------------------------------------------- 2. curves


def _curve_matrix(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    return _finite_array(array, name, 2)


def _positions(positions: object | None, n_checkpoints: int) -> np.ndarray:
    if positions is None:
        return np.arange(n_checkpoints, dtype=float)
    x = _finite_array(positions, "positions", 1)
    if x.size != n_checkpoints:
        raise ValueError("positions must have one entry per checkpoint")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("positions must be strictly increasing")
    return x


def curve_auc(
    values: object, positions: object | None = None, *, normalize: bool = True
) -> np.ndarray:
    """Trapezoidal area under each row's checkpoint curve.

    ``values`` is ``(seeds, checkpoints)`` (a 1-D curve is treated as one seed).
    ``positions`` are the checkpoint x-coordinates (e.g. env steps), default
    ``0..C-1``.  With ``normalize=True`` the area is divided by the x-span, giving
    the curve's time-averaged height in the metric's own units.
    """
    matrix = _curve_matrix(values, "values")
    if matrix.shape[1] < 2:
        raise ValueError("curve_auc needs at least 2 checkpoints")
    x = _positions(positions, matrix.shape[1])
    widths = np.diff(x)
    area = ((matrix[:, 1:] + matrix[:, :-1]) * 0.5 * widths[None, :]).sum(axis=1)
    return area / float(x[-1] - x[0]) if normalize else area


def last_k_mean(values: object, k: int) -> np.ndarray:
    """Mean of each row's last ``k`` checkpoints (robust to acquire-then-lose)."""
    matrix = _curve_matrix(values, "values")
    k = _positive_int(k, "k")
    if k > matrix.shape[1]:
        raise ValueError("k exceeds the number of checkpoints")
    return matrix[:, -k:].mean(axis=1)


def _contrast_summary(
    candidate: np.ndarray, baseline: np.ndarray | None, paired: bool, confidence: float
) -> dict[str, Any]:
    if baseline is None or paired:
        deltas = candidate if baseline is None else candidate - baseline
        n = int(deltas.size)
        if n < 2:
            raise ValueError("a seed-level CI needs at least 2 seeds")
        mean = float(deltas.mean())
        sd = float(deltas.std(ddof=1))
        se = sd / math.sqrt(n)
        df = float(n - 1)
        per_seed = deltas.tolist()
    else:
        n_c, n_b = int(candidate.size), int(baseline.size)
        if n_c < 2 or n_b < 2:
            raise ValueError("Welch contrast needs at least 2 seeds per arm")
        var_c, var_b = float(candidate.var(ddof=1)) / n_c, float(baseline.var(ddof=1)) / n_b
        mean = float(candidate.mean() - baseline.mean())
        se = math.sqrt(var_c + var_b)
        denom = var_c**2 / (n_c - 1) + var_b**2 / (n_b - 1)
        df = (var_c + var_b) ** 2 / denom if denom > 0.0 else math.inf
        sd = float("nan")
        per_seed = None
    critical = _t_two_sided_critical(confidence, df)
    if se > 0.0:
        statistic = mean / se
        tail = _norm_sf(abs(statistic)) if math.isinf(df) else student_t_sf(abs(statistic), df)
        p_value = min(1.0, 2.0 * tail)
    else:
        statistic, p_value = (math.copysign(math.inf, mean), 0.0) if mean != 0.0 else (0.0, 1.0)
    return {
        "mean_delta": mean,
        "sd": sd,
        "se": se,
        "df": df,
        "ci": [mean - critical * se, mean + critical * se],
        "t_statistic": statistic,
        "p_value": p_value,
        "per_seed_delta": per_seed,
    }


def learning_curve_contrast(
    candidate: object,
    baseline: object | None = None,
    *,
    positions: object | None = None,
    last_k: int = 3,
    confidence: float = 0.95,
    paired: bool = True,
) -> dict[str, Any]:
    """Compare learning curves across seeds by AUC, last-k mean and final checkpoint.

    ``candidate`` and ``baseline`` are ``(seeds, checkpoints)`` matrices of an
    evaluation metric at each saved checkpoint.  Each seed's curve is reduced to
    a scalar (normalized AUC, mean of the last ``last_k`` checkpoints, final
    value); the seed scalars are then compared with a t interval:

    * ``paired=True`` (default): seed ``s`` of candidate is paired with seed
      ``s`` of baseline (same init seed / eval bank); CI on ``n - 1`` df.
    * ``paired=False``: Welch interval (independent arms, may differ in size).
    * ``baseline=None``: ``candidate`` already holds paired per-checkpoint deltas
      and the summary is a one-sample interval around zero.

    Seeds are the unit of replication; checkpoint-to-checkpoint correlation
    within a seed is absorbed by reducing each curve to one number first.
    """
    confidence = _probability(confidence, "confidence")
    cand = _curve_matrix(candidate, "candidate")
    base = None
    if baseline is not None:
        base = _curve_matrix(baseline, "baseline")
        if base.shape[1] != cand.shape[1]:
            raise ValueError("candidate and baseline need the same checkpoints")
        if paired and base.shape[0] != cand.shape[0]:
            raise ValueError("paired contrast needs the same number of seeds per arm")
    reducers = {
        "auc": lambda m: curve_auc(m, positions),
        "last_k": lambda m: last_k_mean(m, last_k),
        "final": lambda m: m[:, -1],
    }
    result: dict[str, Any] = {
        "n_seeds_candidate": int(cand.shape[0]),
        "n_seeds_baseline": None if base is None else int(base.shape[0]),
        "n_checkpoints": int(cand.shape[1]),
        "last_k": int(last_k),
        "paired": bool(paired or base is None),
        "confidence": confidence,
    }
    for name, reduce in reducers.items():
        cand_scores = np.asarray(reduce(cand), dtype=float)
        base_scores = None if base is None else np.asarray(reduce(base), dtype=float)
        summary = _contrast_summary(cand_scores, base_scores, paired, confidence)
        summary["candidate_scores"] = cand_scores.tolist()
        summary["baseline_scores"] = None if base_scores is None else base_scores.tolist()
        result[name] = summary
    return result


# --------------------------------------------------------------------------- 3. planning


def gate_pass_probabilities(
    per_seed_power: float | Sequence[float], n_seeds: int | None = None, *, alpha: float = 0.05
) -> dict[str, Any]:
    """Pass probability of an all-seeds conjunction gate vs a pooled gate.

    Assumes one common true effect (no between-seed heterogeneity) and that each
    seed's gate is a one-sided z-test at level ``alpha`` with the stated power,
    so seed ``i``'s standardized effect is ``d_i = z_{1-alpha} + z_{p_i}``.

    * Conjunction (every seed must pass): ``prod(p_i)``; false-pass rate at a
      null effect ``alpha^k``.
    * Pooled inverse-variance gate (one z-test on the pooled estimate at the
      same ``alpha``): ``Phi(sqrt(sum d_i^2) - z_{1-alpha})``; false-pass
      rate ``alpha``.

    Example: ``p = 0.8`` and ``k = 3`` gives 0.512 for the conjunction and
    about 0.996 for the pooled gate.  The conjunction's null false-pass rate
    (0.05^3) is far below what a screen needs; the pooled gate spends its
    error budget on power instead.

    ``per_seed_power`` may be a scalar (then ``n_seeds`` is required) or one
    power per seed (seeds with different world counts).
    """
    alpha = _half_probability(alpha, "alpha")
    if isinstance(per_seed_power, (int, float, np.integer, np.floating)) and not isinstance(
        per_seed_power, bool
    ):
        if n_seeds is None:
            raise ValueError("n_seeds is required with a scalar per_seed_power")
        powers = [_probability(per_seed_power, "per_seed_power")] * _positive_int(
            n_seeds, "n_seeds"
        )
    else:
        powers = [_probability(p, "per_seed_power") for p in per_seed_power]
        if not powers:
            raise ValueError("per_seed_power must not be empty")
        if n_seeds is not None and n_seeds != len(powers):
            raise ValueError("n_seeds disagrees with len(per_seed_power)")
    z_alpha = _norm_ppf(1.0 - alpha)
    standardized = [z_alpha + _norm_ppf(p) for p in powers]
    signs = {math.copysign(1.0, d) for d in standardized if d != 0.0}
    if len(signs) > 1:
        raise ValueError("per-seed powers imply effects of opposite sign")
    sign = signs.pop() if signs else 1.0
    pooled_z = sign * math.sqrt(sum(d * d for d in standardized))
    k = len(powers)
    return {
        "n_seeds": k,
        "alpha": alpha,
        "per_seed_power": powers,
        "conjunction_pass": float(np.prod(powers)),
        "pooled_pass": _norm_cdf(pooled_z - z_alpha),
        "conjunction_false_pass": alpha**k,
        "pooled_false_pass": alpha,
        "per_seed_standardized_effect": standardized,
    }


def _mc_gate_rates(
    effect: float, se: float, tau2: float, k: int, alpha: float, n_sims: int, seed: int
) -> dict[str, float]:
    """Monte Carlo pass rates with estimated tau^2 (equal per-seed variances)."""
    rng = np.random.default_rng(seed)
    y = effect + rng.normal(0.0, math.sqrt(tau2), (n_sims, k)) + rng.normal(0.0, se, (n_sims, k))
    z_alpha = _norm_ppf(1.0 - alpha)
    conjunction = np.all(y / se > z_alpha, axis=1)
    mean = y.mean(axis=1)
    ss = ((y - mean[:, None]) ** 2).sum(axis=1)
    tau2_hat = np.maximum(0.0, (ss / se**2 - (k - 1)) * se**2 / (k - 1))
    total = se**2 + tau2_hat
    se_dl = np.sqrt(total / k)
    q_hk = np.maximum(ss / (total * (k - 1)), 1.0)
    se_hk = np.sqrt(q_hk * total / k)
    t_alpha = student_t_isf(alpha, k - 1)
    return {
        "conjunction_pass": float(conjunction.mean()),
        "pooled_dl_z_pass": float((mean - z_alpha * se_dl > 0.0).mean()),
        "pooled_hksj_floor_pass": float((mean - t_alpha * se_hk > 0.0).mean()),
    }


def gate_pass_probabilities_for_effect(
    effect: float,
    per_seed_se: float,
    n_seeds: int,
    *,
    tau2: float = 0.0,
    alpha: float = 0.05,
    n_sims: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Gate pass probabilities for a true mean effect with seed heterogeneity.

    Seed ``i`` observes ``y_i = effect + u_i + e_i`` with ``u_i ~ N(0, tau2)``
    and ``e_i ~ N(0, per_seed_se^2)``, independent across seeds.  Closed forms
    (known variances, one-sided level ``alpha``):

    * per-seed marginal pass ``p = Phi((effect - z se) / sqrt(se^2 + tau2))``
      and conjunction ``p^k`` (independent seeds);
    * pooled ``Phi(effect / sqrt((se^2 + tau2) / k) - z)``.

    With ``n_sims > 0`` a seeded Monte Carlo also reports the pass rates of the
    gates actually computable from data: the conjunction, a DL z pooled gate
    and the HKSJ-floor pooled gate (both with tau^2 *estimated*), which is what
    :func:`random_effects_pool` with ``confidence = 1 - 2 * alpha`` implements.
    """
    effect = _real(effect, "effect")
    se = _real(per_seed_se, "per_seed_se")
    if se <= 0.0:
        raise ValueError("per_seed_se must be positive")
    tau2 = _real(tau2, "tau2")
    if tau2 < 0.0:
        raise ValueError("tau2 must be non-negative")
    k = _positive_int(n_seeds, "n_seeds")
    alpha = _half_probability(alpha, "alpha")
    z_alpha = _norm_ppf(1.0 - alpha)
    total_sd = math.sqrt(se**2 + tau2)
    per_seed = _norm_cdf((effect - z_alpha * se) / total_sd)
    result: dict[str, Any] = {
        "effect": effect,
        "per_seed_se": se,
        "tau2": tau2,
        "n_seeds": k,
        "alpha": alpha,
        "per_seed_pass": per_seed,
        "conjunction_pass": per_seed**k,
        "pooled_pass": _norm_cdf(effect / (total_sd / math.sqrt(k)) - z_alpha),
        "monte_carlo": None,
    }
    n_sims = int(n_sims)
    if n_sims > 0:
        if k < 2:
            raise ValueError("Monte Carlo pooled gates need at least 2 seeds")
        result["monte_carlo"] = {
            "n_sims": n_sims,
            "seed": int(seed),
            **_mc_gate_rates(effect, se, tau2, k, alpha, n_sims, int(seed)),
        }
    return result


def seeds_for_pooled_power(
    effect: float,
    per_seed_se: float,
    *,
    tau2: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    max_seeds: int = 10_000,
) -> dict[str, Any]:
    """Seeds needed for a one-sided pooled gate to reach ``power``.

    ``z_seeds = ceil((z_{1-alpha} + z_power)^2 (se^2 + tau2) / effect^2)`` is
    the known-variance answer.  ``t_seeds`` is the smallest ``k >= 2`` with
    ``(t_{k-1,1-alpha} + t_{k-1,power})^2 (se^2 + tau2) / effect^2 <= k``, the
    usual t-corrected approximation for a gate on ``k - 1`` df (HKSJ).
    Note that tau2 does not shrink with more worlds per seed, only with more seeds.
    """
    effect = _real(effect, "effect")
    if effect <= 0.0:
        raise ValueError("effect must be positive")
    se = _real(per_seed_se, "per_seed_se")
    if se <= 0.0:
        raise ValueError("per_seed_se must be positive")
    tau2 = _real(tau2, "tau2")
    if tau2 < 0.0:
        raise ValueError("tau2 must be non-negative")
    alpha = _half_probability(alpha, "alpha")
    power = _probability(power, "power")
    ratio = (se**2 + tau2) / effect**2
    z_sum = _norm_ppf(1.0 - alpha) + _norm_ppf(power)
    z_seeds = max(1, math.ceil(z_sum**2 * ratio - 1e-12))
    t_seeds = None
    for k in range(2, _positive_int(max_seeds, "max_seeds", 2) + 1):
        t_sum = student_t_isf(alpha, k - 1)
        if power > 0.5:
            t_sum += student_t_isf(1.0 - power, k - 1)
        elif power < 0.5:
            t_sum -= student_t_isf(power, k - 1)
        if t_sum**2 * ratio <= k:
            t_seeds = k
            break
    return {"z_seeds": z_seeds, "t_seeds": t_seeds, "variance_ratio": ratio}


# --------------------------------------------------------------------------- 4. futility

_GRID_POINTS = 2001  # odd, for Simpson weights
_GRID_SD = 9.0  # lower truncation of the continuation region, in SDs
_MIN_FRACTION_STEP = 0.01
_BOUNDARY_BRACKET = (-10.0, 40.0)
_BISECTION_STEPS = 100


def obf_spending(fraction: float, alpha: float) -> float:
    """Lan-DeMets O'Brien-Fleming-type spending for a one-sided level ``alpha``.

    ``alpha(t) = 2 * (1 - Phi(z_{1 - alpha/2} / sqrt(t)))``, so ``alpha(1) = alpha``
    and almost nothing is spent at early looks.
    """
    fraction = _real(fraction, "fraction")
    alpha = _probability(alpha, "alpha")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if fraction == 0.0:
        return 0.0
    return 2.0 * _norm_sf(_norm_ppf(1.0 - alpha / 2.0) / math.sqrt(fraction))


def _simpson_weights(grid: np.ndarray) -> np.ndarray:
    h = float(grid[1] - grid[0])
    weights = np.ones(grid.size)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * h / 3.0


def _crossing_recursion(
    fractions: Sequence[float],
    drift: float,
    boundaries: Sequence[float] | None = None,
    increments: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Upper first-crossing probabilities of ``Z_k = B(t_k) / sqrt(t_k)``.

    ``B`` is Brownian motion with drift ``drift`` (``B(t) ~ N(drift t, t)``).
    Either ``boundaries`` (z scale) are given and crossing probabilities are
    returned, or ``increments`` (target per-look crossing probabilities) are
    given and boundaries are solved by bisection.  The sub-density of the
    not-yet-stopped process is propagated on a Simpson grid (Armitage,
    McPherson and Rowe recursive integration).
    """
    solve = boundaries is None
    out_bounds: list[float] = []
    out_probs: list[float] = []
    grid = weights = density = None
    previous_t = 0.0
    for index, t in enumerate(fractions):
        root_t = math.sqrt(t)
        step = t - previous_t
        root_step = math.sqrt(step)
        if index == 0:

            def cross(b: float, t: float = t, root_t: float = root_t) -> float:
                return _norm_sf((b * root_t - drift * t) / root_t)

        else:
            mass = weights * density
            shifted = grid + drift * step

            def cross(b: float, root_t: float = root_t, root_step: float = root_step) -> float:
                return float((mass * _norm_sf_array((b * root_t - shifted) / root_step)).sum())

        if solve:
            target = increments[index]
            low, high = _BOUNDARY_BRACKET
            for _ in range(_BISECTION_STEPS):
                mid = 0.5 * (low + high)
                if cross(mid) > target:
                    low = mid
                else:
                    high = mid
            bound = 0.5 * (low + high)
        else:
            bound = boundaries[index]
        out_bounds.append(bound)
        out_probs.append(cross(bound))

        upper = bound * root_t
        lower = min(drift * t - _GRID_SD * root_t, upper - 1e-9)
        new_grid = np.linspace(lower, upper, _GRID_POINTS)
        if index == 0:
            new_density = np.exp(-0.5 * ((new_grid - drift * t) / root_t) ** 2) / (
                root_t * math.sqrt(2.0 * math.pi)
            )
        else:
            kernel = (new_grid[:, None] - shifted[None, :]) / root_step
            new_density = (np.exp(-0.5 * kernel**2) * mass[None, :]).sum(axis=1) / (
                root_step * math.sqrt(2.0 * math.pi)
            )
        grid, density, weights = new_grid, new_density, _simpson_weights(new_grid)
        previous_t = t
    return out_bounds, out_probs


@dataclass(frozen=True)
class FutilityPlan:
    """Pre-declared one-sided "clear loser" stopping plan.

    At look ``k`` the first ``look_sizes[k]`` paired deltas (in the pre-declared
    world order) give a statistic ``z_k`` for ``H0: delta >= null_delta``; the
    candidate is stopped as a clear loser when ``z_k <= -boundaries[k]``.
    ``alpha`` bounds the probability of stopping a candidate whose true effect
    equals ``null_delta``.  There is no efficacy stop, so the plan can only
    lower the chance of a later promotion (it never inflates false promotion).
    """

    look_sizes: tuple[int, ...]
    max_size: int
    fractions: tuple[float, ...]
    alpha: float
    null_delta: float
    boundaries: tuple[float, ...]
    alpha_spent: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "look_sizes": list(self.look_sizes),
            "max_size": self.max_size,
            "fractions": list(self.fractions),
            "alpha": self.alpha,
            "null_delta": self.null_delta,
            "boundaries": list(self.boundaries),
            "alpha_spent": list(self.alpha_spent),
            "spending": "lan_demets_obrien_fleming",
        }


def futility_plan(
    look_sizes: Sequence[int],
    *,
    alpha: float = 0.05,
    null_delta: float = 0.0,
    max_size: int | None = None,
) -> FutilityPlan:
    """Build a Lan-DeMets O'Brien-Fleming-type one-sided futility plan.

    ``look_sizes`` are cumulative world counts at each look (strictly
    increasing, each >= 2).  Information fractions are ``n_k / max_size``
    (``max_size`` defaults to the last look).  Boundaries are exact for the
    canonical joint normal distribution (independent increments); consecutive
    fractions must differ by at least 0.01.  Declare the plan before looking
    at any data and do not change looks after the fact.
    """
    sizes = tuple(_positive_int(n, "look size", 2) for n in look_sizes)
    if not sizes:
        raise ValueError("look_sizes must not be empty")
    if any(b <= a for a, b in zip(sizes, sizes[1:])):
        raise ValueError("look_sizes must be strictly increasing")
    total = sizes[-1] if max_size is None else _positive_int(max_size, "max_size", 2)
    if total < sizes[-1]:
        raise ValueError("max_size must be >= the last look size")
    alpha = _half_probability(alpha, "alpha")
    fractions = tuple(n / total for n in sizes)
    steps = np.diff((0.0,) + fractions)
    if np.any(steps < _MIN_FRACTION_STEP):
        raise ValueError("consecutive information fractions must differ by >= 0.01")
    cumulative = [obf_spending(t, alpha) for t in fractions]
    increments = np.diff([0.0] + cumulative).tolist()
    boundaries, crossings = _crossing_recursion(fractions, 0.0, increments=increments)
    return FutilityPlan(
        look_sizes=sizes,
        max_size=total,
        fractions=fractions,
        alpha=alpha,
        null_delta=_real(null_delta, "null_delta"),
        boundaries=tuple(float(b) for b in boundaries),
        alpha_spent=tuple(float(c) for c in np.cumsum(crossings)),
    )


def _t_to_z(t_stat: float, df: float) -> float:
    """Map a Student-t statistic to the normal quantile with the same tail area."""
    if math.isinf(t_stat):
        return t_stat
    if t_stat == 0.0:
        return 0.0
    tail = student_t_sf(abs(t_stat), df)
    if tail <= 0.0:
        return math.copysign(math.inf, t_stat)
    return math.copysign(-_norm_ppf(tail), t_stat)


def evaluate_futility_look(
    plan: FutilityPlan, look: int, deltas: Sequence[float]
) -> dict[str, Any]:
    """Evaluate one pre-declared look of a :class:`FutilityPlan`.

    ``deltas`` must contain exactly ``plan.look_sizes[look]`` paired
    ``candidate - baseline`` deltas: the first worlds of the pre-declared order.
    The paired t statistic ``(mean - null_delta) / (sd / sqrt(n))`` is mapped to
    the normal scale by matching tail areas (the standard group-sequential
    treatment of an unknown variance) and compared with ``-boundary``.  If all
    deltas are identical the statistic is +/-inf (or 0 when equal to the null).
    """
    if not isinstance(plan, FutilityPlan):
        raise TypeError("plan must be a FutilityPlan")
    look = int(look)
    if not 0 <= look < len(plan.look_sizes):
        raise ValueError("look index out of range")
    values = _finite_array(deltas, "deltas", 1)
    expected = plan.look_sizes[look]
    if values.size != expected:
        raise ValueError(f"look {look} is pre-declared at {expected} deltas, got {values.size}")
    n = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    shift = mean - plan.null_delta
    if sd > 0.0:
        t_stat = shift / (sd / math.sqrt(n))
    else:
        t_stat = math.copysign(math.inf, shift) if shift != 0.0 else 0.0
    z_stat = _t_to_z(t_stat, n - 1)
    boundary = plan.boundaries[look]
    return {
        "look": look,
        "n": n,
        "fraction": plan.fractions[look],
        "mean_delta": mean,
        "sd": sd,
        "t_statistic": t_stat,
        "z_statistic": z_stat,
        "boundary": -boundary,
        "stop_clear_loser": bool(z_stat <= -boundary),
        "alpha_spent": plan.alpha_spent[look],
    }


def run_futility_sequence(plan: FutilityPlan, deltas: Sequence[float]) -> dict[str, Any]:
    """Evaluate every look reachable with ``deltas`` in order, stopping at the first hit.

    ``deltas`` are all paired deltas collected so far in the pre-declared world
    order; looks whose size exceeds ``len(deltas)`` are not evaluated.
    """
    values = _finite_array(deltas, "deltas", 1)
    looks: list[dict[str, Any]] = []
    stop_look = None
    for look, size in enumerate(plan.look_sizes):
        if size > values.size:
            break
        record = evaluate_futility_look(plan, look, values[:size])
        looks.append(record)
        if record["stop_clear_loser"]:
            stop_look = look
            break
    return {
        "looks": looks,
        "stopped": stop_look is not None,
        "stop_look": stop_look,
        "completed_all_looks": stop_look is None and len(looks) == len(plan.look_sizes),
        "plan": plan.as_dict(),
    }


def futility_stop_probability(
    plan: FutilityPlan, true_delta: float, delta_sd: float
) -> dict[str, Any]:
    """Probability that ``plan`` stops a candidate with a given true effect.

    Known-variance normal approximation: with per-world delta SD ``delta_sd``
    the harm drift at full information is
    ``theta = (null_delta - true_delta) / (delta_sd / sqrt(max_size))`` and the
    per-look stop probabilities come from the same recursive integration as
    the boundaries.  ``true_delta == null_delta`` recovers the spent alpha.
    Also returns the expected number of worlds used (to the last look).
    """
    true_delta = _real(true_delta, "true_delta")
    delta_sd = _real(delta_sd, "delta_sd")
    if delta_sd <= 0.0:
        raise ValueError("delta_sd must be positive")
    drift = (plan.null_delta - true_delta) / (delta_sd / math.sqrt(plan.max_size))
    _, per_look = _crossing_recursion(plan.fractions, drift, boundaries=plan.boundaries)
    cumulative = np.cumsum(per_look)
    expected = sum(p * n for p, n in zip(per_look, plan.look_sizes))
    expected += (1.0 - float(cumulative[-1])) * plan.look_sizes[-1]
    return {
        "true_delta": true_delta,
        "delta_sd": delta_sd,
        "drift": drift,
        "per_look_stop": [float(p) for p in per_look],
        "cumulative_stop": [float(c) for c in cumulative],
        "stop_probability": float(cumulative[-1]),
        "expected_worlds": float(expected),
    }
