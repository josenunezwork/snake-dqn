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
  stop can only lower the type-I error.
* **Decision at look k.**  ``STOP_PASS`` if ``>= required_successes`` mixes have crossed
  AND scripted NI is established AND the behavioral bands pass on the data so far;
  ``STOP_FUTILE`` if so many uncrossed mixes are futile that ``required_successes``
  can no longer be reached; else ``CONTINUE``.  At the last look: ``FINAL_PASS`` under
  the same conjunction, otherwise ``FINAL_FAIL``.

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
from src.scripts.eval_stats import paired_delta_test, scripted_noninferiority

__all__ = [
    "DEFAULT_FRACTIONS",
    "SEQUENTIAL_GATE_METHOD",
    "SequentialGatePlan",
    "conditional_power",
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

    @property
    def n_looks(self) -> int:
        return len(self.look_sizes)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": SEQUENTIAL_GATE_METHOD}
        for key, value in self.__dict__.items():
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
) -> SequentialGatePlan:
    """Compute and freeze the looks, boundaries and nominal levels of the gate."""
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
    sizes = look_sizes_for(n_max, fractions)
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
    )


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
    qualifies = successes >= plan.required_successes and ni and bands
    if look == plan.n_looks - 1:
        return "FINAL_PASS" if qualifies else "FINAL_FAIL"
    if qualifies:
        return "STOP_PASS"
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
    ``bands_pass[i]`` is the behavioral-band verdict on the data available at look ``i``
    (one entry per look ``0..look``).  If an earlier look already reached ``STOP_PASS`` the
    result is invalid (evaluating past an efficacy stop); an earlier ``STOP_FUTILE`` that
    was overridden is allowed (futility is non-binding) and listed in
    ``futility_overrides``.
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
            }
        )

    earlier = history[:-1]
    efficacy_stops = [h["look"] for h in earlier if h["decision"] == "STOP_PASS"]
    current = history[-1]
    valid = not efficacy_stops
    result: dict[str, Any] = {
        "method": SEQUENTIAL_GATE_METHOD,
        "look": look,
        "final_look": look == plan.n_looks - 1,
        "n_per_mix": plan.look_sizes[look],
        "fraction": plan.fractions[look],
        "decision": current["decision"],
        "valid": valid,
        "stopped": current["decision"] != "CONTINUE",
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
        "futility_overrides": [h["look"] for h in earlier if h["decision"] == "STOP_FUTILE"],
        "history": history,
        "plan": plan.as_dict(),
    }
    if not valid:
        result["invalid_reason"] = f"already_stopped_for_efficacy_at_look_{efficacy_stops[0]}"
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
    stops are recorded and overridden (the type-I-relevant, non-binding reading).
    """
    available = min(len(list(deltas_by_mix[m])) for m in plan.mixes)
    bands = [True] * plan.n_looks if bands_pass_by_look is None else list(bands_pass_by_look)
    last: dict[str, Any] | None = None
    for look, size in enumerate(plan.look_sizes):
        if size > available:
            break
        prefixes = {m: list(deltas_by_mix[m])[:size] for m in plan.mixes}
        last = sequential_decision(plan, look, prefixes, absolute_delta_ni, bands[: look + 1])
        if last["decision"] in ("STOP_PASS", "FINAL_PASS", "FINAL_FAIL"):
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
