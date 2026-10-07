"""Group-sequential Phase R for Tier-1 FRP-family screens (opt-in, pure, deterministic).

Proposed by ``docs/research/sequential_phase_r_design_2026-10-07.md`` and
``docs/research/governance_amendment_sequential_phase_r_2026-10-07.md``.  Nothing here changes
an existing study's package or its fixed-N decision rule; a future study opts in by declaring a
:func:`validate_rule` rule spec in its pre-registration and building a
:class:`SequentialPhaseRPlan` with :func:`make_plan` before any Phase R world is played.

Design (every number frozen in the plan):

* **Looks.**  ``K`` looks (default 1/3, 2/3, 1) at ``look_sizes[k] = ceil(f_k * N)`` worlds of
  every (seed, mix) bank, ``N`` the planned worlds per (seed, mix).  Look ``k`` uses the first
  ``look_sizes[k]`` worlds of every bank in its pre-declared order, for every cell, so every
  look is a balanced subset over seeds x mixes x heroes.  Look units are whole nested episodes
  (H10000 records with their exact H5000 prefix blocks), so every look has both horizons.
  Information time is the world fraction ``look_sizes[k] / N`` (conservative when the seeds
  are heterogeneous: shared seed effects raise the correlation between looks).
* **GO (efficacy).**  The rule's efficacy statistic is a Hartung-Knapp pooled seed effect; it
  crosses at look ``k`` when its one-sided lower bound at the nominal level ``go_nominal_p[k]``
  exceeds the threshold.  The nominal levels come from Lan-DeMets O'Brien-Fleming spending of
  ``go_alpha`` (0.10 one-sided, the fixed rule's "HK one-sided 90% lower bound") on the
  canonical joint normal distribution, applied to the HK t statistic by tail-area matching on
  its own ``k - 1`` df (Jennison and Turnbull 2000, section 3.8).  The boundaries ignore KILL
  (non-binding futility), so stopping for KILL can only lower P(GO).
* **GO's other clauses** (point thresholds) are judged at the stopping look with an interim
  margin against them, zero at the final look (so the final look is the fixed point rule):

  - *protective* clauses (survival floors, scripted mean, champion guard -- ``"protective":
    true`` in the rule) use ``protective_margin_z * se_k * sqrt(1 - n_k / N)``: ``se_k^2 (1 -
    t)`` is the variance of (interim mean - final mean), so an interim pass is a pass that the
    full run would confirm with probability about ``Phi(protective_margin_z)``.  This shape
    bounds the extra chances an early look gives a floor-violating candidate (simulated: within
    about +0.005 absolute of the fixed design's GO rate, see the design doc);
  - other point clauses (clause 1's +20, MI10 +20, attribution points) use ``margin_z * se_k *
    (1 - sqrt(n_k / N))`` (the strict gate's ``block_at_stop`` band margin).

  Count clauses (positive seeds) and flags (prefix identity controls) are judged as written.
  The feasibility clause sizes the strict gate from the look's own efficacy bound (a repeated
  confidence bound) and the shrunken point.
* **KILL (futility, non-binding for GO, followed by protocol).**  KILL at look ``k`` when every
  KILL component's one-sided upper bound at ``kill_nominal_p[k]`` is below its threshold; the
  nominal levels spend ``kill_alpha`` (0.10, the fixed rule's "one-sided 90% UPPER bound") with
  the pre-registered spending function, so the bounds are repeated confidence bounds and a
  candidate whose true effect is at a KILL threshold is KILLed with probability <=
  ``kill_alpha`` over all looks (the fixed rule's guarantee).  An interim KILL (or NO_GO) is
  suppressed while a higher-precedence final-only outcome holds on the look's point values.
* **Final-only outcomes** (GO_UNGATEABLE, RECIPE, ...), and PARTIAL, are judged at the final look
  only, in the rule's precedence order: INVALID_ANALYSIS > GO > final outcomes > KILL > PARTIAL.

Every function is pure: :func:`sequential_phase_r_decision` replays looks ``0..k`` from prefixes
of the supplied deltas, so an auditor needs only the plan, the raw per-world paired deltas and
the flags.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from src.evaluation.screen_stats import (
    _crossing_recursion,
    obf_spending,
    random_effects_pool,
    stratified_mean_of_means,
)
from src.scripts.eval_stats import student_t_isf

__all__ = [
    "DEFAULT_FRACTIONS",
    "METHOD",
    "RULE_SCHEMA",
    "SequentialPhaseRPlan",
    "candidate_selection",
    "look_sizes_for",
    "make_plan",
    "n_fixed",
    "required_inputs",
    "sequential_phase_r_decision",
    "spending",
    "stat_value",
    "validate_rule",
]

METHOD = "sequential-phase-r-obf-hk-v1"
RULE_SCHEMA = "sequential-phase-r-rule/v1"
DEFAULT_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0, 1.0)
DEFAULT_GO_ALPHA = 0.10
DEFAULT_KILL_ALPHA = 0.10
DEFAULT_GO_SPENDING = "obf"
DEFAULT_KILL_SPENDING = "pocock"
DEFAULT_MARGIN_Z = 1.2815515655446004  # one-sided 90%, the screens' convention
DEFAULT_PROTECTIVE_MARGIN_Z = 1.6448536269514722  # one-sided 95% (survival / guard floors)
SPENDINGS = ("obf", "pocock")  # plus "hsd:<gamma>" (Hwang-Shih-DeCani)
METRICS = ("mi5", "surv5", "mi10", "surv10")
STAT_KINDS = ("hk", "mix_mean", "stratified", "welch")
CLAUSE_KINDS = ("point", "positive_seeds", "flag", "feasibility")
OPS = (">=", ">", "<=", "<")
STOP, CONTINUE, HALT = "STOP", "CONTINUE", "HALT"
INVALID = "INVALID_ANALYSIS"
_MIN_FRACTION_STEP = 0.01
_NORMAL = NormalDist()


# ----------------------------------------------------------------------------- utilities
def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _alpha(value: object, name: str) -> float:
    value = _real(value, name)
    if not 0.0 < value < 0.5:
        raise ValueError(f"{name} must be in (0, 0.5)")
    return value


def _int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


# ----------------------------------------------------------------------------- spending
def spending(fraction: float, alpha: float, kind: str) -> float:
    """Cumulative one-sided alpha spent by information fraction ``fraction``.

    ``obf``: Lan-DeMets O'Brien-Fleming ``2 (1 - Phi(z_{1-alpha/2} / sqrt(t)))``; ``pocock``:
    Lan-DeMets Pocock-type ``alpha ln(1 + (e - 1) t)``; ``hsd:<gamma>``: Hwang-Shih-DeCani
    ``alpha (1 - e^{-gamma t}) / (1 - e^{-gamma})`` (``gamma != 0``).
    """
    t = _real(fraction, "fraction")
    alpha = _alpha(alpha, "alpha")
    if not 0.0 <= t <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if t == 0.0:
        return 0.0
    if kind == "obf":
        return obf_spending(t, alpha)
    if kind == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    if isinstance(kind, str) and kind.startswith("hsd:"):
        gamma = float(kind[4:])
        if not math.isfinite(gamma) or gamma == 0.0:
            raise ValueError("hsd gamma must be finite and non-zero")
        return alpha * (1.0 - math.exp(-gamma * t)) / (1.0 - math.exp(-gamma))
    raise ValueError(f"unknown spending {kind!r} (obf, pocock, hsd:<gamma>)")


def _boundaries(
    fractions: Sequence[float], alpha: float, kind: str
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """z boundaries, nominal one-sided levels and cumulative alpha spent per look."""
    cumulative = [spending(t, alpha, kind) for t in fractions]
    increments = np.diff([0.0] + cumulative).tolist()
    if any(i <= 0.0 for i in increments):
        raise ValueError("the spending function must spend a positive amount at every look")
    bounds, crossings = _crossing_recursion(list(fractions), 0.0, increments=increments)
    nominal = tuple(float(_NORMAL.cdf(-c)) for c in bounds)
    return (
        tuple(float(b) for b in bounds),
        nominal,
        tuple(float(c) for c in np.cumsum(crossings)),
    )


def look_sizes_for(n_worlds: int, fractions: Sequence[float] = DEFAULT_FRACTIONS) -> tuple:
    """Cumulative worlds per (seed, mix) at each look: ``ceil(f * N)`` (last fraction 1)."""
    n_worlds = _int(n_worlds, "n_worlds", 2)
    values = [float(f) for f in fractions]
    if not values or abs(values[-1] - 1.0) > 1e-12:
        raise ValueError("the last information fraction must be 1.0")
    if any(not 0.0 < f <= 1.0 + 1e-12 for f in values) or any(
        b <= a for a, b in zip(values, values[1:])
    ):
        raise ValueError("fractions must be strictly increasing in (0, 1]")
    sizes = tuple(int(math.ceil(f * n_worlds - 1e-9)) for f in values[:-1]) + (n_worlds,)
    if sizes[0] < 2 or any(b <= a for a, b in zip(sizes, sizes[1:])):
        raise ValueError("look sizes must be >= 2 and strictly increasing after rounding")
    steps = np.diff((0,) + sizes) / n_worlds
    if np.any(steps < _MIN_FRACTION_STEP):
        raise ValueError("consecutive information fractions must differ by >= 0.01")
    return sizes


# ----------------------------------------------------------------------------- rule spec
def _stat_spec(spec: Mapping[str, Any], cells: Mapping[str, Any], mixes: Sequence[str]) -> dict:
    kind = spec.get("kind")
    if kind not in STAT_KINDS:
        raise ValueError(f"stat kind must be one of {STAT_KINDS}, not {kind!r}")
    metric = spec.get("metric")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, not {metric!r}")
    out: dict[str, Any] = {"kind": kind, "metric": metric}
    if kind == "welch":
        pair = list(spec.get("cells") or [])
        if len(pair) != 2 or any(c not in cells for c in pair) or pair[0] == pair[1]:
            raise ValueError("a welch stat needs two distinct declared cells")
        out["cells"] = pair
    else:
        if spec.get("cell") not in cells:
            raise ValueError(f"stat cell {spec.get('cell')!r} is not declared")
        out["cell"] = spec["cell"]
    if kind == "mix_mean":
        if spec.get("mix") not in mixes:
            raise ValueError(f"mix_mean needs one of the mixes {list(mixes)}")
        out["mix"] = spec["mix"]
    extra = set(spec) - set(out) - {"kind", "metric"}
    if extra:
        raise ValueError(f"unknown stat fields {sorted(extra)}")
    return out


def _point(spec: Mapping[str, Any], cells: Mapping[str, Any], mixes: Sequence[str]) -> dict:
    op = spec.get("op")
    if op not in OPS:
        raise ValueError(f"op must be one of {OPS}")
    tolerance = _real(spec.get("tolerance", 0.0), "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    protective = spec.get("protective", False)
    if not isinstance(protective, bool):
        raise ValueError("protective must be a bool")
    return {
        "name": str(spec["name"]),
        "kind": "point",
        "stat": _stat_spec(spec["stat"], cells, mixes),
        "op": op,
        "threshold": _real(spec["threshold"], "threshold"),
        "tolerance": tolerance,
        "protective": protective,
    }


def validate_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a declarative rule spec and return its canonical form.

    Shape (JSON)::

        {"schema": "sequential-phase-r-rule/v1", "study_id": str,
         "mixes": [str, ...], "seeds": [int, ...],
         "cells": {name: {"hero": str, "baseline": str, "seeds": [int, ...]}},
         "statuses": {"go": str, "kill": str, "partial": str},
         "efficacy": {"stat": <hk stat>, "lower_bound_above": float},
         "go_clauses": [<clause>, ...],
         "kill": {"all_of": [{"name": str, "stat": <stat>, "upper_below": float}, ...]},
         "final_outcomes": [{"status": str, "all_of": [clause names],
                             "none_of": [clause names], "extra": [<point clause>, ...],
                             "requires_efficacy": bool}],
         "no_go": {"status": str, "all_of": [{"name", "stat", "upper_below"}]} | null,
         "candidate": {"cell": str, "metric": str, "confidence": float} | null}

    ``no_go`` (optional, interim looks only) is the gate-futility stop: every component's
    repeated upper bound (KILL's nominal levels) below its threshold ends the study early with
    that status (no strict gate; the final-only labels are reported descriptively).

    A ``<stat>`` is ``{"kind": "hk"|"stratified", "cell", "metric"}``, ``{"kind": "mix_mean",
    "cell", "metric", "mix"}`` or ``{"kind": "welch", "cells": [a, b], "metric"}``.  A
    ``<clause>`` is a point clause ``{"name", "kind": "point", "stat", "op", "threshold",
    "tolerance"}``, ``{"name", "kind": "positive_seeds", "cell", "metric", "min"}``, ``{"name",
    "kind": "flag", "flag"}`` or ``{"name", "kind": "feasibility", "cap", "inflation",
    "shrink", "per_mix_alpha", "power"}`` (sized from the efficacy stat's cell and metric).
    """
    if rule.get("schema") != RULE_SCHEMA:
        raise ValueError(f"rule schema must be {RULE_SCHEMA!r}")
    mixes = [str(m) for m in rule["mixes"]]
    if not mixes or len(set(mixes)) != len(mixes):
        raise ValueError("mixes must be distinct and non-empty")
    seeds = [int(s) for s in rule["seeds"]]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct seeds are needed")
    cells: dict[str, Any] = {}
    for name, cell in dict(rule["cells"]).items():
        cell_seeds = [int(s) for s in cell["seeds"]]
        if not cell_seeds or any(s not in seeds for s in cell_seeds):
            raise ValueError(f"cell {name!r} seeds must be a non-empty subset of the seeds")
        if len(set(cell_seeds)) != len(cell_seeds):
            raise ValueError(f"cell {name!r} repeats a seed")
        cells[str(name)] = {
            "hero": str(cell["hero"]),
            "baseline": str(cell["baseline"]),
            "seeds": cell_seeds,
        }
    statuses = {k: str(rule["statuses"][k]) for k in ("go", "kill", "partial")}
    efficacy = {
        "stat": _stat_spec(rule["efficacy"]["stat"], cells, mixes),
        "lower_bound_above": _real(rule["efficacy"]["lower_bound_above"], "lower_bound_above"),
    }
    if efficacy["stat"]["kind"] != "hk":
        raise ValueError("the efficacy statistic must be a Hartung-Knapp pooled effect (hk)")
    if len(cells[efficacy["stat"]["cell"]]["seeds"]) < 2:
        raise ValueError("the efficacy cell needs >= 2 seeds")
    clauses: list[dict[str, Any]] = []
    for spec in rule["go_clauses"]:
        kind = spec.get("kind")
        if kind == "point":
            clauses.append(_point(spec, cells, mixes))
        elif kind == "positive_seeds":
            if spec.get("cell") not in cells or spec.get("metric") not in METRICS:
                raise ValueError("positive_seeds needs a declared cell and a metric")
            clauses.append(
                {
                    "name": str(spec["name"]),
                    "kind": kind,
                    "cell": spec["cell"],
                    "metric": spec["metric"],
                    "min": _int(spec["min"], "min", 1),
                }
            )
        elif kind == "flag":
            clauses.append({"name": str(spec["name"]), "kind": kind, "flag": str(spec["flag"])})
        elif kind == "feasibility":
            clauses.append(
                {
                    "name": str(spec["name"]),
                    "kind": kind,
                    "cap": _int(spec["cap"], "cap", 2),
                    "inflation": _real(spec["inflation"], "inflation"),
                    "shrink": _real(spec["shrink"], "shrink"),
                    "per_mix_alpha": _alpha(spec["per_mix_alpha"], "per_mix_alpha"),
                    "power": _real(spec["power"], "power"),
                }
            )
        else:
            raise ValueError(f"clause kind must be one of {CLAUSE_KINDS}, not {kind!r}")
    names = [c["name"] for c in clauses]
    if len(set(names)) != len(names):
        raise ValueError("clause names must be unique")
    kill_parts = []
    for spec in rule["kill"]["all_of"]:
        kill_parts.append(
            {
                "name": str(spec["name"]),
                "stat": _stat_spec(spec["stat"], cells, mixes),
                "upper_below": _real(spec["upper_below"], "upper_below"),
            }
        )
    if not kill_parts:
        raise ValueError("KILL needs at least one component")
    finals = []
    for spec in rule.get("final_outcomes") or []:
        all_of = [str(n) for n in spec.get("all_of", [])]
        none_of = [str(n) for n in spec.get("none_of", [])]
        unknown = [n for n in all_of + none_of if n not in names]
        if unknown:
            raise ValueError(
                f"final outcome {spec.get('status')!r} names unknown clauses {unknown}"
            )
        extra = [_point(p, cells, mixes) for p in spec.get("extra", [])]
        requires = spec.get("requires_efficacy", False)
        if not isinstance(requires, bool):
            raise ValueError("requires_efficacy must be a bool")
        finals.append(
            {
                "status": str(spec["status"]),
                "all_of": all_of,
                "none_of": none_of,
                "extra": extra,
                "requires_efficacy": requires,
            }
        )
    labels = [statuses["go"], statuses["kill"], statuses["partial"], INVALID] + [
        f["status"] for f in finals
    ]
    if len(set(labels)) != len(labels):
        raise ValueError("status labels must be distinct")
    no_go = rule.get("no_go")
    if no_go is not None:
        parts = [
            {
                "name": str(spec["name"]),
                "stat": _stat_spec(spec["stat"], cells, mixes),
                "upper_below": _real(spec["upper_below"], "upper_below"),
            }
            for spec in no_go["all_of"]
        ]
        if not parts:
            raise ValueError("no_go needs at least one component")
        no_go = {"status": str(no_go["status"]), "all_of": parts}
        labels.append(no_go["status"])
        if len(set(labels)) != len(labels):
            raise ValueError("status labels must be distinct")
    candidate = rule.get("candidate")
    if candidate is not None:
        if candidate.get("cell") not in cells or candidate.get("metric") not in METRICS:
            raise ValueError("candidate selection needs a declared cell and a metric")
        conf = _real(candidate.get("confidence", 0.80), "confidence")
        if not 0.0 < conf < 1.0:
            raise ValueError("candidate confidence must be in (0, 1)")
        candidate = {"cell": candidate["cell"], "metric": candidate["metric"], "confidence": conf}
    return {
        "schema": RULE_SCHEMA,
        "study_id": str(rule["study_id"]),
        "mixes": mixes,
        "seeds": seeds,
        "cells": cells,
        "statuses": statuses,
        "efficacy": efficacy,
        "go_clauses": clauses,
        "kill": {"all_of": kill_parts},
        "final_outcomes": finals,
        "no_go": no_go,
        "candidate": candidate,
    }


def _stats_of_rule(rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = [rule["efficacy"]["stat"]]
    for clause in rule["go_clauses"]:
        if clause["kind"] == "point":
            out.append(clause["stat"])
    out += [p["stat"] for p in rule["kill"]["all_of"]]
    if rule.get("no_go"):
        out += [p["stat"] for p in rule["no_go"]["all_of"]]
    for final in rule["final_outcomes"]:
        out += [p["stat"] for p in final["extra"]]
    return out


def required_inputs(rule: Mapping[str, Any]) -> dict[str, list[str]]:
    """``cell -> sorted metrics`` the decision reads (every seed of the cell, every mix)."""
    need: dict[str, set] = {}
    for stat in _stats_of_rule(rule):
        for cell in stat.get("cells") or [stat["cell"]]:
            need.setdefault(cell, set()).add(stat["metric"])
    for clause in rule["go_clauses"]:
        if clause["kind"] == "positive_seeds":
            need.setdefault(clause["cell"], set()).add(clause["metric"])
        if clause["kind"] == "feasibility":
            eff = rule["efficacy"]["stat"]
            need.setdefault(eff["cell"], set()).add(eff["metric"])
    if rule.get("candidate"):
        need.setdefault(rule["candidate"]["cell"], set()).add(rule["candidate"]["metric"])
    return {c: sorted(m) for c, m in sorted(need.items())}


# ----------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class SequentialPhaseRPlan:
    """Frozen pre-registration of a sequential Phase R (build with :func:`make_plan`)."""

    n_worlds: int
    look_sizes: tuple[int, ...]
    fractions: tuple[float, ...]
    go_alpha: float
    go_spending: str
    go_boundaries: tuple[float, ...]
    go_nominal_p: tuple[float, ...]
    go_alpha_spent: tuple[float, ...]
    kill_alpha: float
    kill_spending: str
    kill_boundaries: tuple[float, ...]
    kill_nominal_p: tuple[float, ...]
    kill_alpha_spent: tuple[float, ...]
    margin_z: float
    protective_margin_z: float
    rule: Mapping[str, Any]

    @property
    def n_looks(self) -> int:
        return len(self.look_sizes)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": METHOD}
        for key, value in self.__dict__.items():
            out[key] = list(value) if isinstance(value, tuple) else value
        out["rule"] = json.loads(_canonical(self.rule))
        out["rule_sha256"] = _sha(out["rule"])
        out["information_time"] = "world_fraction"
        out["futility"] = "non_binding_for_go; kill followed by protocol"
        return out

    def sha256(self) -> str:
        return _sha(self.as_dict())


def make_plan(
    n_worlds: int,
    rule: Mapping[str, Any],
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    go_alpha: float = DEFAULT_GO_ALPHA,
    go_spending: str = DEFAULT_GO_SPENDING,
    kill_alpha: float = DEFAULT_KILL_ALPHA,
    kill_spending: str = DEFAULT_KILL_SPENDING,
    margin_z: float = DEFAULT_MARGIN_Z,
    protective_margin_z: float = DEFAULT_PROTECTIVE_MARGIN_Z,
) -> SequentialPhaseRPlan:
    """Compute and freeze looks, nominal levels and the canonical rule."""
    canonical = validate_rule(rule)
    sizes = look_sizes_for(n_worlds, fractions)
    actual = tuple(n / n_worlds for n in sizes)
    go_alpha = _alpha(go_alpha, "go_alpha")
    kill_alpha = _alpha(kill_alpha, "kill_alpha")
    margin_z = _real(margin_z, "margin_z")
    protective_margin_z = _real(protective_margin_z, "protective_margin_z")
    if margin_z < 0.0 or protective_margin_z < 0.0:
        raise ValueError("margins must be >= 0")
    if go_spending != "obf":
        raise ValueError("GO spends its alpha with O'Brien-Fleming (go_spending='obf')")
    go = _boundaries(actual, go_alpha, go_spending)
    kill = _boundaries(actual, kill_alpha, kill_spending)
    return SequentialPhaseRPlan(
        n_worlds=int(n_worlds),
        look_sizes=sizes,
        fractions=actual,
        go_alpha=go_alpha,
        go_spending=go_spending,
        go_boundaries=go[0],
        go_nominal_p=go[1],
        go_alpha_spent=go[2],
        kill_alpha=kill_alpha,
        kill_spending=str(kill_spending),
        kill_boundaries=kill[0],
        kill_nominal_p=kill[1],
        kill_alpha_spent=kill[2],
        margin_z=margin_z,
        protective_margin_z=protective_margin_z,
        rule=canonical,
    )


# ----------------------------------------------------------------------------- statistics
def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _var(values: Sequence[float]) -> float:
    m = _mean(values)
    return math.fsum((v - m) ** 2 for v in values) / (len(values) - 1)


def _seed_effects(
    deltas: Mapping[str, Any], cell: str, metric: str, rule: Mapping[str, Any]
) -> tuple[list[float], list[float]]:
    """Per seed: mix-stratified mean of per-mix means and its variance ``sum s_m^2/n_m / M^2``."""
    mixes = rule["mixes"]
    effects, variances = [], []
    for seed in rule["cells"][cell]["seeds"]:
        by_mix = deltas[cell][metric][seed]
        effects.append(_mean([_mean(by_mix[m]) for m in mixes]))
        variances.append(
            math.fsum(_var(by_mix[m]) / len(by_mix[m]) for m in mixes) / len(mixes) ** 2
        )
    return effects, variances


def _welch(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    va, vb = _var(a) / len(a), _var(b) / len(b)
    se = math.sqrt(va + vb)
    denom = (va**2 / (len(a) - 1) if va > 0 else 0.0) + (vb**2 / (len(b) - 1) if vb > 0 else 0.0)
    df = (va + vb) ** 2 / denom if denom > 0 else None
    return {"estimate": _mean(a) - _mean(b), "se": se, "df": df}


def stat_value(
    stat: Mapping[str, Any], deltas: Mapping[str, Any], rule: Mapping[str, Any]
) -> dict[str, Any]:
    """Estimate, standard error and t df of one declared statistic on a look's deltas."""
    kind, metric = stat["kind"], stat["metric"]
    mixes = rule["mixes"]
    if kind == "hk":
        effects, variances = _seed_effects(deltas, stat["cell"], metric, rule)
        pooled = random_effects_pool(effects, variances, confidence=0.80)
        return {
            "estimate": pooled["mu"],
            "se": pooled["se"],
            "df": pooled["ci_df"],
            "k": pooled["k"],
            "tau2": pooled["tau2"],
            "seed_effects": effects,
            "seed_variances": variances,
        }
    if kind == "welch":
        first, second = stat["cells"]
        a = _seed_effects(deltas, first, metric, rule)[0]
        b = _seed_effects(deltas, second, metric, rule)[0]
        return {**_welch(a, b), "first_effects": a, "second_effects": b}
    seeds = rule["cells"][stat["cell"]]["seeds"]
    if kind == "mix_mean":
        values = [v for s in seeds for v in deltas[stat["cell"]][metric][s][stat["mix"]]]
        return {
            "estimate": _mean(values),
            "se": math.sqrt(_var(values) / len(values)),
            "df": float(len(values) - 1),
            "n": len(values),
        }
    # stratified: mean of per-mix means over the cell's worlds, Welch-Satterthwaite df
    by_mix = {m: [v for s in seeds for v in deltas[stat["cell"]][metric][s][m]] for m in mixes}
    terms = [_var(by_mix[m]) / len(by_mix[m]) for m in mixes]
    variance = math.fsum(terms) / len(mixes) ** 2
    denom = math.fsum(
        (t / len(mixes) ** 2) ** 2 / (len(by_mix[m]) - 1) for t, m in zip(terms, mixes)
    )
    return {
        "estimate": _mean([_mean(by_mix[m]) for m in mixes]),
        "se": math.sqrt(variance),
        "df": variance**2 / denom if denom > 0 else None,
        "n_by_mix": {m: len(v) for m, v in by_mix.items()},
    }


def _t(p: float, df: float | None) -> float:
    if df is None or math.isinf(df):
        return -_NORMAL.inv_cdf(p)
    return student_t_isf(p, df)


def _bound(value: Mapping[str, Any], nominal_p: float, upper: bool) -> float:
    if value["se"] == 0.0:
        return float(value["estimate"])
    half = _t(nominal_p, value["df"]) * value["se"]
    return float(value["estimate"] + half if upper else value["estimate"] - half)


def _compare(value: float, op: str, threshold: float, tolerance: float) -> bool:
    if op == ">=":
        return bool(value >= threshold - tolerance)
    if op == ">":
        return bool(value > threshold)
    if op == "<=":
        return bool(value <= threshold + tolerance)
    return bool(value < threshold)


def _point_clause(
    clause: Mapping[str, Any],
    deltas: Mapping[str, Any],
    rule: Mapping[str, Any],
    shrink: float,
    margin_z: float,
) -> dict[str, Any]:
    """A point threshold with the interim margin ``margin_z * se * (1 - sqrt(n_k / N))``."""
    value = stat_value(clause["stat"], deltas, rule)
    margin = margin_z * value["se"] * shrink
    toward_pass = -margin if clause["op"] in (">=", ">") else margin
    adjusted = value["estimate"] + toward_pass
    return {
        "name": clause["name"],
        "kind": "point",
        "estimate": value["estimate"],
        "se": value["se"],
        "margin": margin,
        "adjusted": adjusted,
        "op": clause["op"],
        "threshold": clause["threshold"],
        "pass": _compare(adjusted, clause["op"], clause["threshold"], clause["tolerance"]),
        "pass_point": _compare(
            value["estimate"], clause["op"], clause["threshold"], clause["tolerance"]
        ),
    }


def _upper_parts(
    parts: Sequence[Mapping[str, Any]], data: Mapping[str, Any], rule: Mapping[str, Any], p: float
) -> list[dict[str, Any]]:
    out = []
    for part in parts:
        value = stat_value(part["stat"], data, rule)
        upper = _bound(value, p, upper=True)
        out.append(
            {
                "name": part["name"],
                "estimate": value["estimate"],
                "se": value["se"],
                "df": value["df"],
                "upper_bound": upper,
                "threshold": part["upper_below"],
                "pass": bool(upper < part["upper_below"]),
            }
        )
    return out


def _n_need(n: int, sd: float, mde: float, per_mix_alpha: float, power: float) -> float:
    return (
        (student_t_isf(per_mix_alpha, n - 1) + student_t_isf(1.0 - power, n - 1)) * sd / mde
    ) ** 2


def n_fixed(
    sd: float,
    mde: float,
    *,
    per_mix_alpha: float = 0.05 / 3,
    power: float = 0.90,
    limit: int = 1_000_000,
) -> int | None:
    """Smallest ``n >= 2`` with ``n >= ((t_{1-a, n-1} + t_{power, n-1}) sd / mde)^2`` (the frp3
    strict pre-registration's sizing, as FRP-v4 clause 7); ``None`` if ``mde <= 0`` or ``n``
    exceeds ``limit``."""
    sd, mde = float(sd), float(mde)
    if not (math.isfinite(sd) and math.isfinite(mde)) or mde <= 0.0 or sd < 0.0:
        return None
    if sd == 0.0 or 2 >= _n_need(2, sd, mde, per_mix_alpha, power):
        return 2
    lo, hi = 2, 4
    while hi < _n_need(hi, sd, mde, per_mix_alpha, power):
        lo, hi = hi, hi * 2
        if hi > limit:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid >= _n_need(mid, sd, mde, per_mix_alpha, power):
            hi = mid
        else:
            lo = mid
    return hi


def _feasibility(
    clause: Mapping[str, Any],
    deltas: Mapping[str, Any],
    rule: Mapping[str, Any],
    efficacy_value: Mapping[str, Any],
    efficacy_bound: float,
) -> dict[str, Any]:
    stat = rule["efficacy"]["stat"]
    seeds = rule["cells"][stat["cell"]]["seeds"]
    sds = {
        m: math.sqrt(_var([v for s in seeds for v in deltas[stat["cell"]][stat["metric"]][s][m]]))
        for m in rule["mixes"]
    }
    sd_star = max(sds.values())
    mde = max(efficacy_bound, clause["shrink"] * efficacy_value["estimate"])
    nf = n_fixed(sd_star, mde, per_mix_alpha=clause["per_mix_alpha"], power=clause["power"])
    n_max = None if nf is None else int(math.ceil(clause["inflation"] * nf - 1e-9))
    return {
        "name": clause["name"],
        "kind": "feasibility",
        "sd_by_mix": sds,
        "sd_star": sd_star,
        "mde_star": mde,
        "mde_bound_route": efficacy_bound,
        "mde_shrink_route": clause["shrink"] * efficacy_value["estimate"],
        "n_fixed": nf,
        "n_max": n_max,
        "cap": clause["cap"],
        "pass": bool(n_max is not None and n_max <= clause["cap"]),
    }


def candidate_selection(
    deltas: Mapping[str, Any], rule: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The seed with the highest lower end of its own two-sided WS interval (ties: lowest)."""
    spec = rule.get("candidate")
    if not spec:
        return None
    rows = []
    for seed in rule["cells"][spec["cell"]]["seeds"]:
        by_mix = {m: list(deltas[spec["cell"]][spec["metric"]][seed][m]) for m in rule["mixes"]}
        row = stratified_mean_of_means(by_mix, confidence=spec["confidence"])
        rows.append({"seed": int(seed), "lower": row["ci"][0], "effect": row["mean_of_means"]})
    best = max(rows, key=lambda r: (r["lower"], -r["seed"]))
    return {
        "rule": "highest lower end of the seed's two-sided WS interval at the candidate "
        "confidence, on the worlds of the stopping look; ties -> lowest seed",
        "seed": best["seed"],
        "lower": best["lower"],
        "ranking": sorted(rows, key=lambda r: (-r["lower"], r["seed"])),
    }


# ----------------------------------------------------------------------------- one look
def _validated(plan: SequentialPhaseRPlan, look: int, deltas: Mapping[str, Any]) -> dict:
    """Prefix of every (cell, metric, seed, mix) list to the look's size (lists must hold at
    least ``look_sizes[look]`` finite values; exactly that many at the analysed look)."""
    rule = plan.rule
    size = plan.look_sizes[look]
    out: dict[str, Any] = {}
    for cell, metrics in required_inputs(rule).items():
        out[cell] = {}
        for metric in metrics:
            out[cell][metric] = {}
            for seed in rule["cells"][cell]["seeds"]:
                out[cell][metric][seed] = {}
                for mix in rule["mixes"]:
                    try:
                        values = deltas[cell][metric][seed][mix]
                    except (KeyError, TypeError) as exc:
                        raise ValueError(
                            f"missing deltas for {cell}/{metric}/{seed}/{mix}"
                        ) from exc
                    values = [float(v) for v in values]
                    if len(values) < size:
                        raise ValueError(
                            f"{cell}/{metric}/{seed}/{mix} has {len(values)} worlds, look {look} "
                            f"needs {size}"
                        )
                    values = values[:size]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError(f"{cell}/{metric}/{seed}/{mix} has a non-finite delta")
                    out[cell][metric][seed][mix] = values
    return out


def _evaluate_look(
    plan: SequentialPhaseRPlan, look: int, deltas: Mapping[str, Any], flags: Mapping[str, bool]
) -> dict[str, Any]:
    rule = plan.rule
    final = look == plan.n_looks - 1
    n = plan.look_sizes[look]
    shrink = 1.0 - math.sqrt(n / plan.n_worlds)
    shrink_protective = math.sqrt(1.0 - n / plan.n_worlds)
    base: dict[str, Any] = {
        "look": look,
        "final_look": final,
        "n_per_seed_mix": n,
        "fraction": plan.fractions[look],
        "go_nominal_p": plan.go_nominal_p[look],
        "kill_nominal_p": plan.kill_nominal_p[look],
        "margin_shrink": shrink,
        "margin_shrink_protective": shrink_protective,
    }
    try:
        data = _validated(plan, look, deltas)
        eff_value = stat_value(rule["efficacy"]["stat"], data, rule)
        eff_bound = _bound(eff_value, plan.go_nominal_p[look], upper=False)
        efficacy = {
            "estimate": eff_value["estimate"],
            "se": eff_value["se"],
            "df": eff_value["df"],
            "tau2": eff_value["tau2"],
            "seed_effects": eff_value["seed_effects"],
            "lower_bound": eff_bound,
            "threshold": rule["efficacy"]["lower_bound_above"],
            "pass": bool(eff_bound > rule["efficacy"]["lower_bound_above"]),
        }
        clauses: dict[str, dict[str, Any]] = {}
        for clause in rule["go_clauses"]:
            kind = clause["kind"]
            if kind == "point":
                if clause["protective"]:
                    row = _point_clause(
                        clause, data, rule, shrink_protective, plan.protective_margin_z
                    )
                else:
                    row = _point_clause(clause, data, rule, shrink, plan.margin_z)
            elif kind == "positive_seeds":
                effects = _seed_effects(data, clause["cell"], clause["metric"], rule)[0]
                positive = sum(1 for e in effects if e > 0.0)
                row = {
                    "name": clause["name"],
                    "kind": kind,
                    "positive": positive,
                    "seeds": len(effects),
                    "min": clause["min"],
                    "pass": bool(positive >= clause["min"]),
                }
            elif kind == "flag":
                value = flags.get(clause["flag"])
                if not isinstance(value, bool):
                    raise ValueError(f"flag {clause['flag']!r} must be supplied as a bool")
                row = {"name": clause["name"], "kind": kind, "flag": clause["flag"], "pass": value}
            else:
                row = _feasibility(clause, data, rule, eff_value, eff_bound)
            clauses[clause["name"]] = row
        kill_parts = _upper_parts(rule["kill"]["all_of"], data, rule, plan.kill_nominal_p[look])
        no_go_parts = (
            _upper_parts(rule["no_go"]["all_of"], data, rule, plan.kill_nominal_p[look])
            if rule.get("no_go") and not final
            else []
        )
        finals = []
        for outcome in rule["final_outcomes"]:
            extra = [_point_clause(p, data, rule, 0.0, 0.0) for p in outcome["extra"]]
            # judged on point values (no interim margin): an interim KILL / NO_GO must not
            # pre-empt an outcome that ranks above it whenever that outcome holds on the data
            point_pass = {n: r.get("pass_point", r["pass"]) for n, r in clauses.items()}
            holds = (
                (efficacy["pass"] or not outcome["requires_efficacy"])
                and all(point_pass[c] for c in outcome["all_of"])
                and not any(point_pass[c] for c in outcome["none_of"])
                and all(e["pass"] for e in extra)
            )
            finals.append({"status": outcome["status"], "extra": extra, "holds": bool(holds)})
        selection = candidate_selection(data, rule)
    except ValueError as exc:
        return {**base, "status": INVALID, "action": HALT, "reason": str(exc)}
    failed = [name for name, row in clauses.items() if not row["pass"]]
    go = efficacy["pass"] and not failed
    kill = all(p["pass"] for p in kill_parts)
    no_go = bool(no_go_parts) and all(p["pass"] for p in no_go_parts)
    blocking = [f["status"] for f in finals if f["holds"]]
    statuses = rule["statuses"]
    if go:
        status, action = statuses["go"], STOP
    elif final:
        status = blocking[0] if blocking else (statuses["kill"] if kill else statuses["partial"])
        action = STOP
    elif kill and not blocking:
        status, action = statuses["kill"], STOP
    elif no_go and not blocking:
        status, action = rule["no_go"]["status"], STOP
    else:
        status, action = "CONTINUE", CONTINUE
    return {
        **base,
        "status": status,
        "action": action,
        "efficacy": efficacy,
        "clauses": clauses,
        "failed_go_clauses": (["efficacy"] if not efficacy["pass"] else []) + failed,
        "kill": {"parts": kill_parts, "all_below": kill, "suppressed_by": blocking if kill else []},
        "no_go": (
            {"parts": no_go_parts, "all_below": no_go, "suppressed_by": blocking if no_go else []}
            if no_go_parts
            else None
        ),
        "final_outcomes": finals if final else [],
        "final_outcomes_holding": blocking,
        "candidate_selection": selection,
    }


def sequential_phase_r_decision(
    plan: SequentialPhaseRPlan,
    look: int,
    deltas: Mapping[str, Any],
    flags: Mapping[str, bool] | Sequence[Mapping[str, bool]],
) -> dict[str, Any]:
    """Decide look ``look`` from per-world paired deltas, replaying every earlier look.

    ``deltas[cell][metric][seed][mix]`` lists the ``hero - baseline`` per-world deltas of the
    cell in the bank's pre-declared world order, exactly ``plan.look_sizes[look]`` of them (the
    look's prefix; earlier looks use prefixes of it).  ``flags`` maps every flag clause's flag
    to a bool (one mapping for all looks, or one per look ``0..look``).  The result is
    ``valid=False`` when an earlier look already stopped or halted (a later look must not
    exist) or when the analysed look has more worlds than its pre-declared size.
    """
    if not isinstance(plan, SequentialPhaseRPlan):
        raise TypeError("plan must be a SequentialPhaseRPlan")
    if isinstance(look, bool) or not isinstance(look, int) or not 0 <= look < plan.n_looks:
        raise ValueError("look index out of range")
    if isinstance(flags, Mapping):
        per_look = [flags] * (look + 1)
    else:
        per_look = list(flags)
        if len(per_look) != look + 1:
            raise ValueError("flags needs one mapping per look 0..look")
    size = plan.look_sizes[look]
    oversize = []
    for cell, metrics in required_inputs(plan.rule).items():
        for metric in metrics:
            for seed in plan.rule["cells"][cell]["seeds"]:
                for mix in plan.rule["mixes"]:
                    try:
                        count = len(deltas[cell][metric][seed][mix])
                    except (KeyError, TypeError):
                        continue  # reported as INVALID by the look evaluation
                    if count > size:
                        oversize.append(f"{cell}/{metric}/{seed}/{mix}")
    history = []
    for index in range(look + 1):
        row = _evaluate_look(plan, index, deltas, per_look[index])
        history.append(
            {
                "look": index,
                "status": row["status"],
                "action": row["action"],
                "n_per_seed_mix": row["n_per_seed_mix"],
            }
        )
        current = row
    earlier_stop = [h for h in history[:-1] if h["action"] != CONTINUE]
    valid = not earlier_stop and not oversize
    result = {
        "method": METHOD,
        "plan_sha256": plan.sha256(),
        **current,
        "valid": valid,
        "history": history,
    }
    if earlier_stop:
        result["invalid_reason"] = (
            f"look {earlier_stop[0]['look']} already ended the study "
            f"({earlier_stop[0]['status']}, {earlier_stop[0]['action']})"
        )
    elif oversize:
        result["invalid_reason"] = (
            f"data beyond look {look} supplied for {oversize[:3]} (prefix integrity)"
        )
    return result
