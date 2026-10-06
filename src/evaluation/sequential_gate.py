"""Group-sequential version of the strict three-mix promotion decision (opt-in).

Proposed by ``docs/research/governance_amendment_sequential_gates_2026-10-02.md``.
Nothing here changes :func:`src.scripts.eval_stats.strict_promotion_decision` or any
existing strict package; a future runner must import these functions explicitly.

Design (all pre-registered in a :class:`SequentialGatePlan` before any final data):

* **Looks.**  ``K`` looks at information fractions ``f_k`` (default 0.25, 0.5, 0.75,
  1.0) of ``n_max`` paired worlds *per mix*.  Look ``k`` uses the first
  ``look_sizes[k] = ceil(f_k * n_max)`` deltas of every mix in the pre-declared
  world order, so every look has equal counts per mix (see :func:`round_robin_plan`).
  Alpha is spent at the *actual* fractions ``look_sizes[k] / n_max``.
* **Efficacy (per mix, one-sided).**  ``H0_m: delta_m <= 0``.  Each mix gets its own
  Lan-DeMets O'Brien-Fleming-type spending at ``family_alpha / n_mixes``.  The z-scale
  boundaries ``c_k`` are exact for the canonical joint normal distribution (recursive
  integration in :func:`src.evaluation.screen_stats.futility_plan`; by symmetry the
  lower "clear loser" boundaries are the upper efficacy boundaries).  A mix *crosses*
  at the first look whose paired one-sided t p-value is ``<= 1 - Phi(c_k)``.  Once
  crossed, ``H0_m`` stays rejected (the usual repeated-significance rule).
* **z/t approximation.**  Variance is estimated, so each look's paired t statistic is
  compared with its own Student-t distribution at the boundary's *nominal* level
  (p-value / tail-area matching, Jennison and Turnbull 2000, section 3.8).  It is exact
  at each single look and only approximately preserves the joint crossing
  probabilities; with ``n >= ~30`` per look the error is negligible and the Monte Carlo
  in ``research/sequential_gate_validation_20261002/`` measures it directly.
* **Multiplicity: Bonferroni across mixes, not Holm.**  Each mix's sequential test has
  level ``alpha / n_mixes`` whatever the other mixes do, so the familywise error rate is
  ``<= alpha`` under any dependence between mixes (the strict mixes share world seeds)
  and any interim stopping, and a PASS (``>= required_successes`` rejections) implies,
  with probability ``>= 1 - alpha``, that every rejected mix is truly superior -- the
  same guarantee Holm gives the fixed-N gate.  A Holm-like step-down *is* valid in a
  group-sequential setting (Maurer and Bretz 2013, graphical approaches; Tamhane,
  Mehta and Liu 2010) but needs boundaries recomputed at the recycled level at every
  look, including past looks; it is not implemented here, which is conservative.
* **Scripted noninferiority.**  ``H0: delta_scripted <= -delta_NI`` gets its own OBF
  spending at ``ni_alpha`` (0.05, as in the fixed gate).  It is established at the first
  look whose lower bound ``mean - t_{n-1}(nominal_k) * se > -delta_NI`` (strict ``>``,
  as in :func:`src.scripts.eval_stats.scripted_noninferiority`).  NI is a conjunct of
  PASS (intersection-union), so it needs no multiplicity share.
* **Futility (non-binding).**  At an interim look a mix that has not crossed is
  *futile* when its conditional power -- the probability of crossing the final
  boundary given the current B-value, under drift ``mde * sqrt(n_max) / sd_hat`` -- is
  below ``futility_cp`` (default 0.10).  CP ignores the remaining interim boundaries,
  so it slightly understates the true chance (a mildly aggressive stop).  The efficacy
  boundaries are computed *ignoring* futility, so stopping or overriding a futility
  stop can only lower the type-I error.  ``plan.futility_policy`` pre-registers whether
  futility stops are ``"followed"`` (default) or ``"overridable"``; evaluating a look
  after a futility stop under a ``"followed"`` plan gives ``valid=False``.
* **Behavioral bands (``band_policy="block_at_stop"``).**  Bands are judged only at the
  *qualifying* look, the first look where ``>= required_successes`` mixes have crossed
  and scripted NI is established.  They never delay a stop: if they fail there, the
  gate ends ``STOP_FAIL_BANDS`` (one band check per run, not one per look).  At an
  interim look a band must hold with the margin of :func:`band_check`,
  ``band_margin_z * sd * (1/sqrt(n_k) - 1/sqrt(n_final))``; at the last look the margin
  is 0 (the fixed-N point rule).  For independent band data, a candidate whose band
  violation the fixed-N gate catches with probability ``>= Phi(band_margin_z)`` is caught
  at an interim stop with at least that probability.
* **Paired noninferiority bands (``band_policy="paired_ni_at_stop"``, opt-in).**  Proposed by
  ``docs/research/governance_amendment_paired_bands_2026-10-03.md``.  Still judged once, at
  the qualifying look, and never delaying a stop, but on the *paired* per-world delta
  (candidate minus incumbent on the same final worlds) instead of the candidate mean
  against a calibration reference.  A band passes iff the one-sided lower confidence bound
  ``mean - t_{n-1}(band_nominal_p[k]) * se > -band_ni_margin`` (strict ``>``, as in
  :func:`src.scripts.eval_stats.scripted_noninferiority`), and, if a ``band_floor`` is
  pre-registered, the candidate mean is ``>= band_floor`` (a mechanical sanity tripwire, not a
  test).  ``band_bound="rci_obf"`` takes ``band_nominal_p`` from a Lan-DeMets OBF spending at
  ``band_alpha`` (a repeated confidence bound: its coverage holds at any data-dependent
  stopping look, including one selected by the correlated mass deltas);
  ``band_bound="pointwise"`` uses ``band_alpha`` at every look (exact at a fixed look, not
  under selection).  ``band_margin_z`` is not used by this policy.  See
  :func:`paired_band_check` and :func:`plan_paired_band_check`.
* **Pooled survival band v2 (``band_policy="pooled_ni_continue"``, opt-in).**  Proposed by
  ``docs/research/governance_amendment_survival_band_v2_2026-10-06.md``.  At a look the bands
  pass iff (1) the pooled paired NI bound clears ``-band_ni_margin``: per world the pooled delta
  is the mean over the mixes of ``candidate - incumbent``, and ``mean - t_{n-1}(p_k) * se >
  -band_ni_margin``; (2) every mix's own paired NI bound clears ``-band_mix_margin`` (the
  per-mix catastrophic floor); (3) every mix's candidate mean is ``>= band_floor`` if a floor is
  pre-registered.  ``band_bound`` must be ``"rci_obf"`` (repeated confidence bounds), because
  the band is judged at the qualifying look **and every later look**: a qualified run whose
  bands fail at an interim look gets the decision ``CONTINUE_BANDS`` (efficacy and NI stay
  established, futility no longer stops it) and passes at the first later look where the bands
  pass, else ``FINAL_FAIL``.  ``STOP_FAIL_BANDS`` never occurs under this policy.  See
  :func:`pooled_band_check` and :func:`plan_pooled_band_check`.
* **Decision at look k.**  At a qualifying interim look ``STOP_PASS`` if the bands pass,
  else ``STOP_FAIL_BANDS`` (``CONTINUE_BANDS`` under ``pooled_ni_continue``); otherwise
  ``STOP_FUTILE`` if so many uncrossed mixes are futile that ``required_successes`` can no
  longer be reached; else ``CONTINUE``.  At the last look: ``FINAL_PASS`` if it qualifies and
  the bands pass, else ``FINAL_FAIL``.

Every function is pure and deterministic: a decision at look ``k`` replays looks
``0..k`` from prefixes of the supplied deltas, so an auditor needs only the plan, the
raw paired deltas and the per-look band results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from src.evaluation.screen_stats import futility_plan
from src.scripts.eval_stats import (
    paired_delta_test,
    scripted_noninferiority,
    student_t_isf,
)

__all__ = [
    "DEFAULT_FRACTIONS",
    "SEQUENTIAL_GATE_METHOD",
    "SequentialGatePlan",
    "band_check",
    "conditional_power",
    "paired_band_check",
    "plan_paired_band_check",
    "plan_pooled_band_check",
    "pooled_band_check",
    "look_sizes_for",
    "round_robin_plan",
    "run_sequential_gate",
    "sequential_decision",
    "sequential_gate_plan",
    "worker_look_counts",
]

SEQUENTIAL_GATE_METHOD = "strict-sequential-obf-bonferroni-v1"
DEFAULT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_MIXES = ("frozen", "scripted", "mixed")
DEFAULT_FUTILITY_CP = 0.10
DEFAULT_BAND_MARGIN_Z = 1.645
FUTILITY_POLICIES = ("followed", "overridable")
BAND_POLICIES = ("block_at_stop", "paired_ni_at_stop", "pooled_ni_continue")
PAIRED_BAND_BOUNDS = ("rci_obf", "pointwise")
POOLED_BAND_POLICY = "pooled_ni_continue"
# Plan fields that exist only under band_policy "paired_ni_at_stop" / "pooled_ni_continue".
# They are omitted from ``as_dict`` under the other policies, so every legacy plan dict (and
# its sha256) is unchanged: "block_at_stop" has none, "paired_ni_at_stop" exactly the five it
# always had, "pooled_ni_continue" those five plus ``band_mix_margin``.
_PAIRED_BAND_FIELDS = ("band_ni_margin", "band_alpha", "band_bound", "band_nominal_p", "band_floor")
_BAND_FIELDS_BY_POLICY = {
    "block_at_stop": (),
    "paired_ni_at_stop": _PAIRED_BAND_FIELDS,
    POOLED_BAND_POLICY: _PAIRED_BAND_FIELDS + ("band_mix_margin",),
}
_ALL_BAND_FIELDS = _BAND_FIELDS_BY_POLICY[POOLED_BAND_POLICY]
_NORMAL = NormalDist()


def _probability(value: object, name: str, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not (math.isfinite(value) and 0.0 < value < upper):
        raise ValueError(f"{name} must be in (0, {upper})")
    return value


def look_sizes_for(n_max: int, fractions: Sequence[float] = DEFAULT_FRACTIONS) -> tuple[int, ...]:
    """Per-mix cumulative world counts ``ceil(f * n_max)`` (last fraction must be 1)."""
    if isinstance(n_max, bool) or not isinstance(n_max, int) or n_max < 2:
        raise ValueError("n_max must be an integer >= 2")
    values = [float(f) for f in fractions]
    if not values or values[-1] != 1.0:
        raise ValueError("the last information fraction must be 1.0")
    if any(not 0.0 < f <= 1.0 for f in values) or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("fractions must be strictly increasing in (0, 1]")
    sizes = tuple(int(math.ceil(f * n_max - 1e-9)) for f in values)
    if sizes[0] < 2 or any(b <= a for a, b in zip(sizes, sizes[1:])):
        raise ValueError("look sizes must be >= 2 and strictly increasing after rounding")
    return sizes


@dataclass(frozen=True)
class SequentialGatePlan:
    """Frozen pre-registration of a group-sequential strict gate."""

    mixes: tuple[str, ...]
    scripted_mix: str
    n_max: int
    look_sizes: tuple[int, ...]
    fractions: tuple[float, ...]
    family_alpha: float
    efficacy_alpha_per_mix: float
    efficacy_boundaries: tuple[float, ...]
    efficacy_nominal_p: tuple[float, ...]
    efficacy_alpha_spent: tuple[float, ...]
    ni_alpha: float
    ni_boundaries: tuple[float, ...]
    ni_nominal_p: tuple[float, ...]
    ni_alpha_spent: tuple[float, ...]
    required_successes: int
    mde: float
    futility_cp: float
    futility_policy: str = "followed"
    band_policy: str = "block_at_stop"
    band_margin_z: float = DEFAULT_BAND_MARGIN_Z
    band_ni_margin: float | None = None
    band_alpha: float | None = None
    band_bound: str | None = None
    band_nominal_p: tuple[float, ...] | None = None
    band_floor: float | None = None
    band_mix_margin: float | None = None

    def __post_init__(self) -> None:
        # Build plans with sequential_gate_plan; this only guards direct construction
        # against band fields that as_dict would silently drop or that cannot be used.
        paired = (self.band_ni_margin, self.band_alpha, self.band_bound, self.band_nominal_p)
        if self.band_policy not in _BAND_FIELDS_BY_POLICY:
            raise ValueError(f"band_policy must be one of {BAND_POLICIES}")
        if self.band_policy != POOLED_BAND_POLICY and self.band_mix_margin is not None:
            raise ValueError(f"band_mix_margin set under band_policy {self.band_policy!r}")
        if self.band_policy == "block_at_stop":
            if any(v is not None for v in paired + (self.band_floor,)):
                raise ValueError(f"paired band fields set under band_policy {self.band_policy!r}")
        elif any(v is None for v in paired) or len(self.band_nominal_p) != len(self.look_sizes):
            raise ValueError(f"{self.band_policy} needs margin, alpha, bound and per-look levels")
        elif self.band_policy == POOLED_BAND_POLICY and (
            self.band_mix_margin is None or self.band_bound != "rci_obf"
        ):
            raise ValueError("pooled_ni_continue needs band_mix_margin and band_bound 'rci_obf'")

    @property
    def n_looks(self) -> int:
        return len(self.look_sizes)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": SEQUENTIAL_GATE_METHOD}
        allowed = _BAND_FIELDS_BY_POLICY[self.band_policy]
        for key, value in self.__dict__.items():
            if key in _ALL_BAND_FIELDS and key not in allowed:
                continue
            out[key] = list(value) if isinstance(value, tuple) else value
        out["spending"] = "lan_demets_obrien_fleming"
        out["multiplicity"] = "bonferroni_across_mixes"
        out["futility"] = "non_binding_conditional_power_under_mde"
        return out


def sequential_gate_plan(
    n_max: int,
    *,
    mde: float,
    mixes: Sequence[str] = DEFAULT_MIXES,
    scripted_mix: str = "scripted",
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    family_alpha: float = 0.05,
    ni_alpha: float = 0.05,
    futility_cp: float = DEFAULT_FUTILITY_CP,
    required_successes: int = 2,
    futility_policy: str = "followed",
    band_policy: str = "block_at_stop",
    band_margin_z: float = DEFAULT_BAND_MARGIN_Z,
    band_ni_margin: float | None = None,
    band_alpha: float | None = None,
    band_bound: str | None = None,
    band_floor: float | None = None,
    band_mix_margin: float | None = None,
) -> SequentialGatePlan:
    """Compute and freeze the looks, boundaries and nominal levels of the gate.

    ``band_ni_margin``, ``band_alpha``, ``band_bound`` and ``band_floor`` belong to
    ``band_policy="paired_ni_at_stop"`` or ``"pooled_ni_continue"`` (margin, alpha and bound
    are then required, the floor is optional) and must be left ``None`` under the default
    ``"block_at_stop"``.  ``band_mix_margin`` (the per-mix catastrophic margin) is required
    under ``"pooled_ni_continue"`` (whose bound must be ``"rci_obf"``; ``band_ni_margin`` is
    then the pooled margin) and must be ``None`` otherwise.
    """
    names = tuple(str(m) for m in mixes)
    if len(names) < 1 or len(set(names)) != len(names):
        raise ValueError("mixes must be distinct and non-empty")
    if scripted_mix not in names:
        raise ValueError(f"scripted mix {scripted_mix!r} is not one of the mixes")
    if isinstance(required_successes, bool) or not isinstance(required_successes, int):
        raise TypeError("required_successes must be an integer")
    if not 1 <= required_successes <= len(names):
        raise ValueError("required_successes must be in [1, n_mixes]")
    family_alpha = _probability(family_alpha, "family_alpha", 0.5)
    ni_alpha = _probability(ni_alpha, "ni_alpha", 0.5)
    futility_cp = _probability(futility_cp, "futility_cp")
    if isinstance(mde, bool) or not isinstance(mde, (int, float)) or not 0.0 < mde < math.inf:
        raise ValueError("mde must be a positive finite number")
    if futility_policy not in FUTILITY_POLICIES:
        raise ValueError(f"futility_policy must be one of {FUTILITY_POLICIES}")
    if band_policy not in BAND_POLICIES:
        raise ValueError(f"band_policy must be one of {BAND_POLICIES}")
    if (
        isinstance(band_margin_z, bool)
        or not isinstance(band_margin_z, (int, float))
        or not 0.0 <= band_margin_z < math.inf
    ):
        raise ValueError("band_margin_z must be a finite number >= 0")
    sizes = look_sizes_for(n_max, fractions)
    paired = _paired_band_parameters(
        band_policy, sizes, n_max, band_ni_margin, band_alpha, band_bound, band_floor
    )
    paired.update(_pooled_band_parameters(band_policy, band_bound, band_mix_margin))
    per_mix = family_alpha / len(names)
    efficacy = futility_plan(sizes, alpha=per_mix, max_size=n_max)
    ni = futility_plan(sizes, alpha=ni_alpha, max_size=n_max)
    return SequentialGatePlan(
        mixes=names,
        scripted_mix=scripted_mix,
        n_max=n_max,
        look_sizes=sizes,
        fractions=efficacy.fractions,
        family_alpha=family_alpha,
        efficacy_alpha_per_mix=per_mix,
        efficacy_boundaries=efficacy.boundaries,
        efficacy_nominal_p=tuple(_NORMAL.cdf(-c) for c in efficacy.boundaries),
        efficacy_alpha_spent=efficacy.alpha_spent,
        ni_alpha=ni_alpha,
        ni_boundaries=ni.boundaries,
        ni_nominal_p=tuple(_NORMAL.cdf(-c) for c in ni.boundaries),
        ni_alpha_spent=ni.alpha_spent,
        required_successes=required_successes,
        mde=float(mde),
        futility_cp=futility_cp,
        futility_policy=futility_policy,
        band_policy=band_policy,
        band_margin_z=float(band_margin_z),
        **paired,
    )


def _paired_band_parameters(
    band_policy: str,
    sizes: tuple[int, ...],
    n_max: int,
    margin: object,
    alpha: object,
    bound: object,
    floor: object,
) -> dict[str, Any]:
    if band_policy == "block_at_stop":
        if any(v is not None for v in (margin, alpha, bound, floor)):
            raise ValueError(
                "band_ni_margin, band_alpha, band_bound and band_floor need "
                'band_policy="paired_ni_at_stop" or "pooled_ni_continue"'
            )
        return {}
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not (math.isfinite(margin) and margin > 0.0)
    ):
        raise ValueError("band_ni_margin must be a positive finite number")
    alpha = _probability(alpha, "band_alpha", 0.5)
    if bound not in PAIRED_BAND_BOUNDS:
        raise ValueError(f"band_bound must be one of {PAIRED_BAND_BOUNDS}")
    if floor is not None and (
        isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor)
    ):
        raise ValueError("band_floor must be None or a finite number")
    if bound == "rci_obf":
        spent = futility_plan(sizes, alpha=alpha, max_size=n_max)
        nominal = tuple(_NORMAL.cdf(-c) for c in spent.boundaries)
    else:
        nominal = (alpha,) * len(sizes)
    return {
        "band_ni_margin": float(margin),
        "band_alpha": alpha,
        "band_bound": bound,
        "band_nominal_p": nominal,
        "band_floor": None if floor is None else float(floor),
    }


def _pooled_band_parameters(band_policy: str, bound: object, mix_margin: object) -> dict[str, Any]:
    if band_policy != POOLED_BAND_POLICY:
        if mix_margin is not None:
            raise ValueError('band_mix_margin needs band_policy="pooled_ni_continue"')
        return {}
    if bound != "rci_obf":
        raise ValueError(
            'pooled_ni_continue judges the bands at several looks: band_bound must be "rci_obf"'
        )
    if (
        isinstance(mix_margin, bool)
        or not isinstance(mix_margin, (int, float))
        or not (math.isfinite(mix_margin) and mix_margin > 0.0)
    ):
        raise ValueError("band_mix_margin must be a positive finite number")
    return {"band_mix_margin": float(mix_margin)}


def _ni_bound(deltas: Sequence[float], nominal_p: float) -> dict[str, Any]:
    n = len(deltas)
    mean = math.fsum(deltas) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    t_crit = student_t_isf(nominal_p, n - 1)
    return {
        "mean_delta": mean,
        "sample_std": sd,
        "standard_error": se,
        "t_critical": t_crit,
        "lower_bound": mean - t_crit * se,
    }


def _positive_margin(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a positive finite number")
    return value


def pooled_band_check(
    candidate: Mapping[str, Sequence[float]],
    incumbent: Mapping[str, Sequence[float]],
    *,
    pooled_margin: float,
    mix_margin: float,
    nominal_p: float,
    floor: float | None = None,
) -> dict[str, Any]:
    """Survival band v2 on the worlds available at a look (all mixes at once).

    ``candidate[m][i]`` and ``incumbent[m][i]`` are the band metric of the two arms on world
    ``i`` of mix ``m`` (world ``i`` is the same world in every mix).  With ``d[m][i] =
    candidate[m][i] - incumbent[m][i]`` and the pooled per-world delta ``p[i] = mean over m of
    d[m][i]``, the bands pass iff ``mean(p) - t_{n-1}(nominal_p) * sd(p) / sqrt(n) >
    -pooled_margin`` and, for every mix, ``mean(d[m]) - t_{n-1}(nominal_p) * sd(d[m]) / sqrt(n)
    > -mix_margin`` and (if a floor is given) ``mean(candidate[m]) >= floor``.  Strict ``>``;
    with zero spread a bound is the mean.
    """
    mixes = list(candidate)
    if not mixes or list(incumbent) != mixes:
        raise ValueError("candidate and incumbent must cover the same mixes, in the same order")
    cand = {m: [float(v) for v in candidate[m]] for m in mixes}
    inc = {m: [float(v) for v in incumbent[m]] for m in mixes}
    n = len(cand[mixes[0]])
    if any(len(cand[m]) != n or len(inc[m]) != n for m in mixes):
        raise ValueError("every mix must pair one value per world, on the same worlds")
    if n < 2 or not all(math.isfinite(v) for m in mixes for v in cand[m] + inc[m]):
        raise ValueError("pooled_band_check needs at least two finite pairs per mix")
    pooled_margin = _positive_margin(pooled_margin, "pooled_margin")
    mix_margin = _positive_margin(mix_margin, "mix_margin")
    nominal_p = _probability(nominal_p, "nominal_p", 0.5)
    if floor is not None and (
        isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor)
    ):
        raise ValueError("floor must be None or a finite number")
    deltas = {m: [c - i for c, i in zip(cand[m], inc[m])] for m in mixes}
    pooled_deltas = [math.fsum(deltas[m][w] for m in mixes) / len(mixes) for w in range(n)]
    pooled = _ni_bound(pooled_deltas, nominal_p)
    pooled["margin"] = pooled_margin
    pooled["passes"] = bool(pooled["lower_bound"] > -pooled_margin)
    per_mix: dict[str, dict[str, Any]] = {}
    for m in mixes:
        row = _ni_bound(deltas[m], nominal_p)
        candidate_mean = math.fsum(cand[m]) / n
        row.update(
            margin=mix_margin,
            candidate_mean=candidate_mean,
            incumbent_mean=math.fsum(inc[m]) / n,
            floor=None if floor is None else float(floor),
            passes_ni=bool(row["lower_bound"] > -mix_margin),
            passes_floor=True if floor is None else bool(candidate_mean >= float(floor)),
        )
        row["passes"] = row["passes_ni"] and row["passes_floor"]
        per_mix[m] = row
    passes_mix = all(r["passes_ni"] for r in per_mix.values())
    passes_floor = all(r["passes_floor"] for r in per_mix.values())
    return {
        "n": n,
        "df": n - 1,
        "mixes": mixes,
        "nominal_p": nominal_p,
        "t_critical": pooled["t_critical"],
        "pooled": pooled,
        "per_mix": per_mix,
        "floor": None if floor is None else float(floor),
        "passes_pooled": pooled["passes"],
        "passes_mix": passes_mix,
        "passes_floor": passes_floor,
        "passes": bool(pooled["passes"] and passes_mix and passes_floor),
    }


def plan_pooled_band_check(
    plan: SequentialGatePlan,
    look: int,
    candidate: Mapping[str, Sequence[float]],
    incumbent: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """:func:`pooled_band_check` with the plan's margins, floor and look-``look`` level.

    ``candidate`` and ``incumbent`` map every plan mix to exactly ``plan.look_sizes[look]``
    values (the look's prefix, pre-declared world order); they are taken in plan mix order.
    """
    if not isinstance(plan, SequentialGatePlan):
        raise TypeError("plan must be a SequentialGatePlan")
    if plan.band_policy != POOLED_BAND_POLICY or plan.band_nominal_p is None:
        raise ValueError('plan_pooled_band_check needs band_policy="pooled_ni_continue"')
    if isinstance(look, bool) or not isinstance(look, int) or not 0 <= look < plan.n_looks:
        raise ValueError("look index out of range")
    if set(candidate) != set(plan.mixes) or set(incumbent) != set(plan.mixes):
        raise ValueError(f"the pooled band needs exactly the mixes {list(plan.mixes)}")
    expected = plan.look_sizes[look]
    for mix in plan.mixes:
        if len(candidate[mix]) != expected or len(incumbent[mix]) != expected:
            raise ValueError(f"look {look} is pre-declared at {expected} worlds per mix")
    assert plan.band_ni_margin is not None and plan.band_mix_margin is not None
    out = pooled_band_check(
        {m: candidate[m] for m in plan.mixes},
        {m: incumbent[m] for m in plan.mixes},
        pooled_margin=plan.band_ni_margin,
        mix_margin=plan.band_mix_margin,
        nominal_p=plan.band_nominal_p[look],
        floor=plan.band_floor,
    )
    out["look"] = look
    out["band_bound"] = plan.band_bound
    out["policy"] = POOLED_BAND_POLICY
    return out


def paired_band_check(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    margin: float,
    nominal_p: float,
    floor: float | None = None,
) -> dict[str, Any]:
    """One paired noninferiority band on the worlds available at a look.

    ``candidate[i]`` and ``incumbent[i]`` are the band metric of the two arms on the same
    world.  Passes iff ``mean(d) - t_{n-1}(nominal_p) * sd(d) / sqrt(n) > -margin`` with
    ``d = candidate - incumbent`` (strict ``>``), and ``mean(candidate) >= floor`` when a floor
    is given.  With zero spread the bound is the mean itself.
    """
    cand = [float(v) for v in candidate]
    inc = [float(v) for v in incumbent]
    if len(cand) != len(inc):
        raise ValueError("candidate and incumbent must pair one value per world")
    if len(cand) < 2 or not all(math.isfinite(v) for v in cand + inc):
        raise ValueError("paired_band_check needs at least two finite pairs")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise TypeError("margin must be a real number")
    margin = float(margin)
    if not (math.isfinite(margin) and margin > 0.0):
        raise ValueError("margin must be a positive finite number")
    nominal_p = _probability(nominal_p, "nominal_p", 0.5)
    if floor is not None and (
        isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor)
    ):
        raise ValueError("floor must be None or a finite number")
    n = len(cand)
    deltas = [c - i for c, i in zip(cand, inc)]
    mean = math.fsum(deltas) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    t_crit = student_t_isf(nominal_p, n - 1)
    lower = mean - t_crit * se
    candidate_mean = math.fsum(cand) / n
    passes_ni = bool(lower > -margin)
    passes_floor = True if floor is None else bool(candidate_mean >= float(floor))
    return {
        "n": n,
        "mean_delta": mean,
        "sample_std": sd,
        "standard_error": se,
        "nominal_p": nominal_p,
        "t_critical": t_crit,
        "lower_bound": lower,
        "margin": margin,
        "candidate_mean": candidate_mean,
        "incumbent_mean": math.fsum(inc) / n,
        "floor": None if floor is None else float(floor),
        "passes_ni": passes_ni,
        "passes_floor": passes_floor,
        "passes": passes_ni and passes_floor,
    }


def plan_paired_band_check(
    plan: SequentialGatePlan,
    look: int,
    candidate: Sequence[float],
    incumbent: Sequence[float],
) -> dict[str, Any]:
    """:func:`paired_band_check` with the plan's margin, floor and look-``look`` level.

    The pair lists must hold exactly ``plan.look_sizes[look]`` worlds (the look's prefix).
    """
    if not isinstance(plan, SequentialGatePlan):
        raise TypeError("plan must be a SequentialGatePlan")
    if plan.band_policy != "paired_ni_at_stop" or plan.band_nominal_p is None:
        raise ValueError('plan_paired_band_check needs band_policy="paired_ni_at_stop"')
    if isinstance(look, bool) or not isinstance(look, int) or not 0 <= look < plan.n_looks:
        raise ValueError("look index out of range")
    expected = plan.look_sizes[look]
    if len(candidate) != expected or len(incumbent) != expected:
        raise ValueError(f"look {look} is pre-declared at {expected} worlds per band")
    assert plan.band_ni_margin is not None
    out = paired_band_check(
        candidate,
        incumbent,
        margin=plan.band_ni_margin,
        nominal_p=plan.band_nominal_p[look],
        floor=plan.band_floor,
    )
    out["look"] = look
    out["band_bound"] = plan.band_bound
    return out


def band_check(
    values: Sequence[float],
    lower: float,
    upper: float,
    *,
    n_final: int,
    margin_z: float = DEFAULT_BAND_MARGIN_Z,
) -> dict[str, Any]:
    """One behavioral band on the candidate values available at a look.

    Passes iff ``lower + m <= mean <= upper - m`` with the interim margin
    ``m = margin_z * sd * (1/sqrt(n) - 1/sqrt(n_final))`` (``m = 0`` once ``n >= n_final``,
    the fixed-N point rule).  If the true mean sits ``D`` final-N standard errors outside
    a bound, the interim pass probability is ``Phi(-(D sqrt(n/n_final) + margin_z (1 -
    sqrt(n/n_final))))``: at most the fixed-N ``Phi(-D)`` while ``D <= margin_z`` and below
    ``Phi(-margin_z)`` beyond it.
    """
    data = [float(v) for v in values]
    if len(data) < 2 or not all(math.isfinite(v) for v in data):
        raise ValueError("band_check needs at least two finite values")
    if isinstance(n_final, bool) or not isinstance(n_final, int) or n_final < 2:
        raise ValueError("n_final must be an integer >= 2")
    if not (math.isfinite(margin_z) and margin_z >= 0.0):
        raise ValueError("margin_z must be a finite number >= 0")
    n = len(data)
    mean = math.fsum(data) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in data) / (n - 1))
    margin = margin_z * sd * max(0.0, 1.0 / math.sqrt(n) - 1.0 / math.sqrt(n_final))
    low, high = float(lower) + margin, float(upper) - margin
    return {
        "n": n,
        "mean": mean,
        "sample_std": sd,
        "margin": margin,
        "lower_effective": low,
        "upper_effective": high,
        "passes": bool(low <= mean <= high),
    }


def conditional_power(z: float, fraction: float, final_boundary: float, drift: float) -> float:
    """Chance that ``Z(1) >= final_boundary`` given ``Z(fraction) = z`` and full-info drift.

    B-value form: ``B(t) = Z(t) sqrt(t)``, ``B(1) | B(t) ~ N(B(t) + drift (1 - t), 1 - t)``.
    Remaining interim boundaries are ignored (CP is slightly understated).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if math.isnan(z) or math.isnan(drift):
        raise ValueError("z and drift must not be NaN")
    if fraction >= 1.0:
        return 1.0 if z >= final_boundary else 0.0
    if math.isinf(z):
        return 1.0 if z > 0 else 0.0
    remaining = 1.0 - fraction
    centre = z * math.sqrt(fraction) + drift * remaining
    return float(_NORMAL.cdf((centre - final_boundary) / math.sqrt(remaining)))


def _z_from_p(p_value: float) -> float:
    if p_value <= 0.0:
        return math.inf
    if p_value >= 1.0:
        return -math.inf
    return -_NORMAL.inv_cdf(p_value)


def _mix_look(
    plan: SequentialGatePlan, look: int, values: list[float], crossed_before: bool
) -> dict[str, Any]:
    test = paired_delta_test(values, alpha=plan.efficacy_nominal_p[look])
    crossed_now = bool(test["valid"] and test["superior"])
    record: dict[str, Any] = {
        "n": test["n"],
        "mean_delta": test["mean_delta"],
        "sample_std": test["sample_std"],
        "t_statistic": test["t_statistic"],
        "p_value": test["p_value"],
        "nominal_p": plan.efficacy_nominal_p[look],
        "z_boundary": plan.efficacy_boundaries[look],
        "crossed_at_this_look": crossed_now,
        "crossed_so_far": crossed_before or crossed_now,
        "conditional_power": None,
        "futile": False,
    }
    last = plan.n_looks - 1
    sd = test["sample_std"]
    if look < last and not record["crossed_so_far"] and test["valid"] and sd:
        drift = plan.mde * math.sqrt(plan.n_max) / sd
        power = conditional_power(
            _z_from_p(test["p_value"]),
            plan.fractions[look],
            plan.efficacy_boundaries[last],
            drift,
        )
        record["conditional_power"] = power
        record["futile"] = power < plan.futility_cp
    return record


def _validated_deltas(
    plan: SequentialGatePlan, look: int, deltas_by_mix: Mapping[str, Sequence[float]]
) -> dict[str, list[float]]:
    if set(deltas_by_mix) != set(plan.mixes):
        raise ValueError(f"deltas must cover exactly the mixes {list(plan.mixes)}")
    expected = plan.look_sizes[look]
    out = {}
    for mix in plan.mixes:
        values = [float(v) for v in deltas_by_mix[mix]]
        if len(values) != expected:
            raise ValueError(
                f"look {look} is pre-declared at {expected} deltas per mix, "
                f"mix {mix!r} has {len(values)}"
            )
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"mix {mix!r} has a non-finite delta")
        out[mix] = values
    return out


def _decide(
    plan: SequentialGatePlan, look: int, successes: int, futile: int, ni: bool, bands: bool
) -> str:
    qualifies = successes >= plan.required_successes and ni
    if look == plan.n_looks - 1:
        return "FINAL_PASS" if qualifies and bands else "FINAL_FAIL"
    if qualifies and plan.band_policy == POOLED_BAND_POLICY:
        # survival band v2: judged at this and every later look; a failure continues
        return "STOP_PASS" if bands else "CONTINUE_BANDS"
    if qualifies:  # block_at_stop / paired_ni_at_stop: bands judged once, never delay
        return "STOP_PASS" if bands else "STOP_FAIL_BANDS"
    if len(plan.mixes) - futile < plan.required_successes:
        return "STOP_FUTILE"
    return "CONTINUE"


def sequential_decision(
    plan: SequentialGatePlan,
    look: int,
    deltas_by_mix: Mapping[str, Sequence[float]],
    absolute_delta_ni: float,
    bands_pass: Sequence[bool],
) -> dict[str, Any]:
    """Decide look ``look`` from raw paired deltas, replaying every earlier look.

    ``deltas_by_mix[m]`` holds exactly ``plan.look_sizes[look]`` candidate-minus-incumbent
    deltas of mix ``m`` in the pre-declared world order; earlier looks are its prefixes.
    ``bands_pass[i]`` is the behavioral-band verdict on the data available at look ``i``, one
    entry per look ``0..look``; only the qualifying look's entry is used.  Under
    ``band_policy="block_at_stop"`` every band goes through :func:`band_check` with
    ``plan.band_margin_z`` and ``n_final = plan.n_max``; under ``"paired_ni_at_stop"`` through
    :func:`plan_paired_band_check` at look ``i``; under ``"pooled_ni_continue"`` through
    :func:`plan_pooled_band_check` at look ``i``, and then every qualified look's entry is used
    (``CONTINUE_BANDS`` while they fail).  If an
    earlier look already reached ``STOP_PASS`` or ``STOP_FAIL_BANDS`` the result is invalid.
    An earlier ``STOP_FUTILE`` that was overridden is listed in ``futility_overrides``; it
    is valid only under ``futility_policy="overridable"`` (futility is non-binding, so
    type-I error is unaffected either way, but an unregistered override is a protocol
    deviation and ``unregistered_futility_override`` is set).
    """
    if not isinstance(plan, SequentialGatePlan):
        raise TypeError("plan must be a SequentialGatePlan")
    if isinstance(look, bool) or not isinstance(look, int) or not 0 <= look < plan.n_looks:
        raise ValueError("look index out of range")
    values = _validated_deltas(plan, look, deltas_by_mix)
    margin = float(absolute_delta_ni)
    if not (math.isfinite(margin) and margin > 0.0):
        raise ValueError("absolute_delta_ni must be positive and finite")
    bands = list(bands_pass)
    if len(bands) != look + 1 or not all(isinstance(b, bool) for b in bands):
        raise ValueError("bands_pass needs one bool per look 0..look")

    crossing_look: dict[str, int | None] = {m: None for m in plan.mixes}
    ni_look: int | None = None
    history: list[dict[str, Any]] = []
    per_mix: dict[str, dict[str, Any]] = {}
    ni_record: dict[str, Any] = {}
    for index in range(look + 1):
        size = plan.look_sizes[index]
        per_mix = {}
        for mix in plan.mixes:
            record = _mix_look(plan, index, values[mix][:size], crossing_look[mix] is not None)
            if crossing_look[mix] is None and record["crossed_at_this_look"]:
                crossing_look[mix] = index
            per_mix[mix] = record
        ni_record = scripted_noninferiority(
            values[plan.scripted_mix][:size], margin, alpha=plan.ni_nominal_p[index]
        )
        if ni_look is None and ni_record["passes"]:
            ni_look = index
        successes = [m for m in plan.mixes if crossing_look[m] is not None]
        futile = [m for m in plan.mixes if per_mix[m]["futile"]]
        decision = _decide(
            plan, index, len(successes), len(futile), ni_look is not None, bands[index]
        )
        history.append(
            {
                "look": index,
                "n_per_mix": size,
                "decision": decision,
                "successful_mixes": successes,
                "futile_mixes": futile,
                "ni_established": ni_look is not None,
                "bands_pass": bands[index],
                "bands_judged": len(successes) >= plan.required_successes and ni_look is not None,
            }
        )

    earlier = history[:-1]
    final_stops = [h for h in earlier if h["decision"] in ("STOP_PASS", "STOP_FAIL_BANDS")]
    overrides = [h["look"] for h in earlier if h["decision"] == "STOP_FUTILE"]
    unregistered = bool(overrides) and plan.futility_policy != "overridable"
    current = history[-1]
    valid = not final_stops and not unregistered
    result: dict[str, Any] = {
        "method": SEQUENTIAL_GATE_METHOD,
        "look": look,
        "final_look": look == plan.n_looks - 1,
        "n_per_mix": plan.look_sizes[look],
        "fraction": plan.fractions[look],
        "decision": current["decision"],
        "valid": valid,
        "stopped": current["decision"] not in ("CONTINUE", "CONTINUE_BANDS"),
        "passes": bool(valid and current["decision"] in ("STOP_PASS", "FINAL_PASS")),
        "successful_mixes": current["successful_mixes"],
        "futile_mixes": current["futile_mixes"],
        "crossing_looks": crossing_look,
        "ni_established": ni_look is not None,
        "ni_crossing_look": ni_look,
        "bands_pass": bands[look],
        "absolute_delta_ni": margin,
        "per_mix": per_mix,
        "scripted_noninferiority": ni_record,
        "futility_overrides": overrides,
        "unregistered_futility_override": unregistered,
        "history": history,
        "plan": plan.as_dict(),
    }
    if final_stops:
        kind = "efficacy" if final_stops[0]["decision"] == "STOP_PASS" else "bands"
        result["invalid_reason"] = f"already_stopped_for_{kind}_at_look_{final_stops[0]['look']}"
    elif unregistered:
        result["invalid_reason"] = (
            f"futility_stop_at_look_{overrides[0]}_overridden_under_policy_followed"
        )
    return result


def run_sequential_gate(
    plan: SequentialGatePlan,
    deltas_by_mix: Mapping[str, Sequence[float]],
    absolute_delta_ni: float,
    bands_pass_by_look: Sequence[bool] | None = None,
    *,
    honor_futility: bool = True,
) -> dict[str, Any]:
    """Walk the looks reachable with the supplied deltas and stop at the first stop.

    ``bands_pass_by_look`` defaults to all-pass.  With ``honor_futility=False`` futility
    stops are recorded and overridden (the type-I-relevant, non-binding reading); under a
    ``futility_policy="followed"`` plan the result is then marked invalid.
    """
    available = min(len(list(deltas_by_mix[m])) for m in plan.mixes)
    bands = [True] * plan.n_looks if bands_pass_by_look is None else list(bands_pass_by_look)
    last: dict[str, Any] | None = None
    for look, size in enumerate(plan.look_sizes):
        if size > available:
            break
        prefixes = {m: list(deltas_by_mix[m])[:size] for m in plan.mixes}
        last = sequential_decision(plan, look, prefixes, absolute_delta_ni, bands[: look + 1])
        if last["decision"] in ("STOP_PASS", "STOP_FAIL_BANDS", "FINAL_PASS", "FINAL_FAIL"):
            break
        if last["decision"] == "STOP_FUTILE" and honor_futility:
            break
    if last is None:
        raise ValueError(f"need at least {plan.look_sizes[0]} deltas per mix for the first look")
    last["worlds_per_mix_used"] = last["n_per_mix"]
    return last


def round_robin_plan(
    n_max: int, mixes: Sequence[str] = DEFAULT_MIXES, workers: int = 2
) -> list[dict[str, Any]]:
    """Deterministic world-major interleaving of ``(mix, world_index)`` units.

    Unit ``u = world_index * n_mixes + mix_position`` is one paired world (incumbent and
    candidate episodes of that mix and world, both run by worker ``u % workers``).  Look
    ``k`` is the unit prefix of length ``n_mixes * look_sizes[k]``, so every look holds
    exactly ``look_sizes[k]`` worlds of every mix.  Worker ``w`` runs the units
    ``u % workers == w`` in order, so the intersection of a look prefix with its shard is a
    prefix of that shard: a worker pauses at a look after exactly
    :func:`worker_look_counts` units and no unit from beyond the look enters the analysis.
    """
    names = tuple(str(m) for m in mixes)
    if not names or len(set(names)) != len(names):
        raise ValueError("mixes must be distinct and non-empty")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(n_max, bool) or not isinstance(n_max, int) or n_max < 1:
        raise ValueError("n_max must be a positive integer")
    units = []
    for world_index in range(n_max):
        for mix in names:
            index = len(units)
            units.append(
                {"unit": index, "world_index": world_index, "mix": mix, "worker": index % workers}
            )
    return units


def worker_look_counts(
    look_sizes: Sequence[int], n_mixes: int = 3, workers: int = 2
) -> list[dict[str, Any]]:
    """Units per worker that complete each look's prefix in :func:`round_robin_plan`."""
    out = []
    for look, size in enumerate(look_sizes):
        total = int(size) * n_mixes
        per_worker = [(total - w + workers - 1) // workers for w in range(workers)]
        out.append(
            {"look": look, "worlds_per_mix": int(size), "units": total, "per_worker": per_worker}
        )
    return out
