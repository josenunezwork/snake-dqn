"""Pure, one-look decision rule for the T7 fresh-bank B8 Phase R study.

This module deliberately consumes completed H10000 records only.  It has no runner or
serving side effects: the caller supplies records and the predeclared bank identity, then
receives a JSON-serialisable decision suitable for the receipt and final report.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from src.evaluation.sequential_phase_r import (
    _bound,
    _feasibility,
    _point_clause,
    _seed_effects,
    _sha,
    candidate_selection,
    stat_value,
)

SEEDS = (0, 1, 2, 3, 4)
MIXES = ("frozen", "mixed", "scripted")
MIX_COUNTS = {"scripted": 192, "frozen": 96, "mixed": 96}
HERO = "B8-noveto"
INCUMBENT = "incumbent"
CHAMPION = "champion"
H5000 = 5000
H10000 = 10000
SURVIVAL_TOLERANCE = 1e-9


class DecisionError(ValueError):
    """The fixed-N input does not satisfy the pre-registered record contract."""


def _point(
    name: str, stat: dict[str, Any], op: str, threshold: float, *, protective: bool = False
) -> dict[str, Any]:
    out = {
        "name": name,
        "kind": "point",
        "stat": stat,
        "op": op,
        "threshold": threshold,
        "tolerance": SURVIVAL_TOLERANCE if "alive" in name else 0.0,
        "protective": protective,
    }
    return out


def rule() -> dict[str, Any]:
    """Return the complete, canonical T7 rule (including its hash for pre-registration)."""
    primary = {"hero": HERO, "baseline": INCUMBENT, "seeds": list(SEEDS)}
    guard = {"hero": HERO, "baseline": CHAMPION, "seeds": [0]}
    clauses = [
        _point("c1_mass_point", {"kind": "hk", "cell": "primary", "metric": "mi5"}, ">=", 20.0),
        _point(
            "c2_scripted_mass",
            {"kind": "mix_mean", "cell": "primary", "metric": "mi5", "mix": "scripted"},
            ">=",
            0.0,
            protective=True,
        ),
    ]
    for mix in MIXES:
        clauses.append(
            _point(
                f"c2_{mix}_alive",
                {"kind": "mix_mean", "cell": "primary", "metric": "surv5", "mix": mix},
                ">=",
                -0.03,
                protective=True,
            )
        )
    clauses += [
        {
            "name": "c4_positive_seeds",
            "kind": "positive_seeds",
            "cell": "primary",
            "metric": "mi5",
            "min": 4,
        },
        {"name": "c5_prefix_controls", "kind": "flag", "flag": "prefix_controls"},
        _point("c5_mi10", {"kind": "hk", "cell": "primary", "metric": "mi10"}, ">=", 20.0),
        # This is deliberately the original M4 seed-0-only mass guard.  Tier 1 replaces only
        # its former champion-survival companion, not this clause.
        _point(
            "c6_guard_mass_seed0",
            {"kind": "stratified", "cell": "guard", "metric": "mi5"},
            ">",
            0.0,
            protective=True,
        ),
        {
            "name": "c7_feasibility",
            "kind": "feasibility",
            "cap": 600,
            "inflation": 1.02,
            "shrink": 0.6,
            "per_mix_alpha": 0.05 / 3,
            "power": 0.90,
        },
    ]
    value: dict[str, Any] = {
        "schema": "redesign-t7-phase-r-rule/v1",
        "study_id": "redesign-t7-phase-r-b8-noveto-20261016",
        "one_look": {"n_worlds": 192, "horizon": H10000, "prefix_horizon": H5000},
        "mixes": list(MIXES),
        "mix_counts": dict(MIX_COUNTS),
        "seeds": list(SEEDS),
        "cells": {"primary": primary, "guard": guard},
        "efficacy": {
            "stat": {"kind": "hk", "cell": "primary", "metric": "mi5"},
            "lower_bound_above": 0.0,
            "confidence": 0.90,
        },
        "go_clauses": clauses,
        "candidate": {"cell": "primary", "metric": "mi5", "confidence": 0.80},
        "tier1": {
            "per_seed_alive_floor": {
                "metric": "surv5",
                "baseline": INCUMBENT,
                "mix": "scripted",
                "threshold": -0.15,
                "tolerance": SURVIVAL_TOLERANCE,
            },
            "absolute_episode_floor": {
                "horizon": H5000,
                "definition": "prefix_h5000.deaths == 0",
                "threshold": 0.50,
            },
        },
        "tier2": {
            "horizon": H5000,
            "mix": "scripted",
            "metric": "episode_survival",
            "baseline": CHAMPION,
            "threshold": -0.05,
        },
        "futility": {
            "stat": {"kind": "hk", "cell": "primary", "metric": "mi5"},
            "upper_below": 50.0,
            "confidence": 0.90,
        },
    }
    value["rule_sha256"] = _sha(value)
    return value


def _finite(value: Any, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionError(f"{what} is not numeric") from exc
    if not math.isfinite(out):
        raise DecisionError(f"{what} is non-finite")
    return out


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise DecisionError("a statistic needs at least two worlds")
    vals = [_finite(v, "statistic") for v in values]
    estimate = math.fsum(vals) / len(vals)
    variance = math.fsum((v - estimate) ** 2 for v in vals) / (len(vals) - 1)
    return {
        "estimate": estimate,
        "se": math.sqrt(variance / len(vals)),
        "df": float(len(vals) - 1),
        "n": len(vals),
    }


def _episode(prefix: Mapping[str, Any]) -> float:
    """H5000 episode survival is no death by H5000, including later H10000 deaths."""
    deaths = _finite(prefix["deaths"], "prefix deaths")
    if deaths not in (0.0, 1.0):
        raise DecisionError("prefix deaths must be terminal 0 or 1")
    return float(deaths == 0.0)


def _metric(entry: Mapping[str, Any], metric: str) -> float:
    if entry.get("horizon") != H10000 or not isinstance(entry.get("prefix_h5000"), Mapping):
        raise DecisionError("decision record lacks exact H10000 / prefix_h5000 contract")
    source = entry["prefix_h5000"] if metric in ("mi5", "surv5") else entry["record"]
    if not isinstance(source, Mapping):
        raise DecisionError("record metric payload is not an object")
    key = "mass_integral" if metric.startswith("mi") else "survival_fraction"
    value = _finite(source.get(key), f"{metric}/{key}")
    if key == "survival_fraction" and not 0.0 <= value <= 1.0:
        raise DecisionError(f"{metric}/survival_fraction is outside [0, 1]")
    return value


def _indexed(
    entries: Sequence[Mapping[str, Any]], banks: Mapping[int, Sequence[int]]
) -> dict[tuple[str, int, str, int], Mapping[str, Any]]:
    expected_banks: dict[int, list[int]] = {}
    for seed in SEEDS:
        try:
            worlds = [int(w) for w in banks[seed]]
        except (KeyError, TypeError, ValueError) as exc:
            raise DecisionError(f"missing bank for seed {seed}") from exc
        if len(worlds) != 192 or len(set(worlds)) != 192:
            raise DecisionError(f"seed {seed} bank must have exactly 192 distinct worlds")
        expected_banks[seed] = worlds
    if set(int(s) for s in banks) != set(SEEDS):
        raise DecisionError("banks must contain exactly seeds 0..4")
    all_worlds = [world for worlds in expected_banks.values() for world in worlds]
    if len(set(all_worlds)) != len(all_worlds):
        raise DecisionError("the five seed banks must be globally disjoint")
    result: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise DecisionError("entry is not an object")
        if entry.get("control"):
            continue
        try:
            key = (
                str(entry["hero"]),
                int(entry["seed"]),
                str(entry["mix"]),
                int(entry["world_seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DecisionError("entry has no valid identity") from exc
        if key in result:
            raise DecisionError(f"duplicate decision record {key}")
        result[key] = entry
    expected: set[tuple[str, int, str, int]] = set()
    for seed, worlds in expected_banks.items():
        for mix, count in MIX_COUNTS.items():
            for world in worlds[:count]:
                for hero in (HERO, INCUMBENT, CHAMPION):
                    expected.add((hero, seed, mix, world))
    if set(result) != expected:
        extra, missing = set(result) - expected, expected - set(result)
        raise DecisionError(
            f"decision identity set mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    for key, entry in result.items():
        if entry.get("horizon") != H10000:
            raise DecisionError(f"{key} horizon is not H10000")
        if not isinstance(entry.get("prefix_h5000"), Mapping):
            raise DecisionError(f"{key} lacks H5000 prefix")
        for metric in ("mi5", "surv5", "mi10"):
            _metric(entry, metric)
        _episode(entry["prefix_h5000"])
    return result


def _deltas(
    index: Mapping[tuple[str, int, str, int], Mapping[str, Any]], banks: Mapping[int, Sequence[int]]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "primary": {m: {} for m in ("mi5", "surv5", "mi10")},
        "guard": {"mi5": {}},
    }
    for seed in SEEDS:
        for metric in ("mi5", "surv5", "mi10"):
            out["primary"][metric][seed] = {}
        for mix, count in MIX_COUNTS.items():
            worlds = [int(w) for w in banks[seed]][:count]
            for metric in ("mi5", "surv5", "mi10"):
                out["primary"][metric][seed][mix] = [
                    _metric(index[(HERO, seed, mix, world)], metric)
                    - _metric(index[(INCUMBENT, seed, mix, world)], metric)
                    for world in worlds
                ]
            if seed == 0:
                out["guard"]["mi5"][seed] = out["guard"]["mi5"].get(seed, {})
                out["guard"]["mi5"][seed][mix] = [
                    _metric(index[(HERO, seed, mix, world)], "mi5")
                    - _metric(index[(CHAMPION, seed, mix, world)], "mi5")
                    for world in worlds
                ]
    return out


def _per_seed_floor(
    deltas: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[dict[str, Any], list[int]]:
    rows: dict[str, Any] = {}
    passing: list[int] = []
    for seed in SEEDS:
        # Tier 1's catastrophic floor is scripted, on this seed's own 192-world bank.
        value = _summary(deltas["primary"]["surv5"][seed]["scripted"])
        floor = spec["tier1"]["per_seed_alive_floor"]
        ok = value["estimate"] >= floor["threshold"] - floor["tolerance"]
        rows[str(seed)] = {**value, "threshold": floor["threshold"], "pass": ok}
        if ok:
            passing.append(seed)
    return rows, passing


def _absolute_and_sensitivity(
    index: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    banks: Mapping[int, Sequence[int]],
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    absolute: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {"incumbent": {}, "champion": {}}
    tier2_values: list[float] = []
    for mix, count in MIX_COUNTS.items():
        hero_episode = [
            _episode(index[(HERO, seed, mix, world)]["prefix_h5000"])
            for seed in SEEDS
            for world in [int(w) for w in banks[seed]][:count]
        ]
        for baseline in (INCUMBENT, CHAMPION):
            alive, episode = [], []
            for seed in SEEDS:
                for world in [int(w) for w in banks[seed]][:count]:
                    hero = index[(HERO, seed, mix, world)]
                    base = index[(baseline, seed, mix, world)]
                    alive.append(_metric(hero, "surv5") - _metric(base, "surv5"))
                    episode.append(_episode(hero["prefix_h5000"]) - _episode(base["prefix_h5000"]))
            sensitivity[baseline][mix] = {
                "alive_frame": _summary(alive),
                "episode": _summary(episode),
            }
            if baseline == CHAMPION and mix == "scripted":
                tier2_values = episode
        row = _summary(hero_episode)
        threshold = spec["tier1"]["absolute_episode_floor"]["threshold"]
        absolute[mix] = {**row, "threshold": threshold, "pass": row["estimate"] >= threshold}
    tier2 = {**_summary(tier2_values), "threshold": spec["tier2"]["threshold"]}
    tier2["pass"] = tier2["estimate"] >= tier2["threshold"] - SURVIVAL_TOLERANCE
    return absolute, sensitivity, tier2


def decide(
    entries: list[dict[str, Any]], banks: dict[int, list[int]], prefix_controls: bool
) -> dict[str, Any]:
    """Apply T7 once at full N; malformed/incomplete inputs refuse without a scientific result.

    Control records may be supplied alongside decision records, but are intentionally excluded from
    all estimates.  ``prefix_controls`` is the independently recomputed exact-prefix result.
    """
    spec = rule()
    try:
        if not isinstance(prefix_controls, bool):
            raise DecisionError("prefix_controls must be bool")
        index = _indexed(entries, banks)
        deltas = _deltas(index, banks)
        efficacy_value = stat_value(spec["efficacy"]["stat"], deltas, spec)
        efficacy_lb = _bound(efficacy_value, 0.10, upper=False)
        efficacy = {
            **efficacy_value,
            "lower_bound": efficacy_lb,
            "threshold": 0.0,
            "pass": efficacy_lb > 0.0,
        }
        clauses: dict[str, Any] = {}
        for clause in spec["go_clauses"]:
            if clause["kind"] == "point":
                clauses[clause["name"]] = _point_clause(clause, deltas, spec, 0.0, 0.0)
            elif clause["kind"] == "positive_seeds":
                effects = _seed_effects(deltas, clause["cell"], clause["metric"], spec)[0]
                positive = sum(effect > 0.0 for effect in effects)
                clauses[clause["name"]] = {
                    "name": clause["name"],
                    "kind": "positive_seeds",
                    "positive": positive,
                    "seeds": len(effects),
                    "min": clause["min"],
                    "pass": positive >= clause["min"],
                }
            elif clause["kind"] == "flag":
                clauses[clause["name"]] = {
                    "name": clause["name"],
                    "kind": "flag",
                    "flag": clause["flag"],
                    "pass": prefix_controls,
                }
            else:
                clauses[clause["name"]] = _feasibility(
                    clause, deltas, spec, efficacy_value, efficacy_lb
                )
        per_seed_floor, eligible = _per_seed_floor(deltas, spec)
        floor = spec["tier1"]["per_seed_alive_floor"]
        clauses["tier1_per_seed_alive_floor"] = {
            "name": "tier1_per_seed_alive_floor",
            "kind": "per_seed_floor",
            "threshold": floor["threshold"],
            "rows": per_seed_floor,
            "pass": len(eligible) == len(SEEDS),
        }
        absolute, sensitivity, tier2 = _absolute_and_sensitivity(index, banks, spec)
        absolute_spec = spec["tier1"]["absolute_episode_floor"]
        clauses["tier1_absolute_episode_floor"] = {
            "name": "tier1_absolute_episode_floor",
            "kind": "absolute_episode_floor",
            "threshold": absolute_spec["threshold"],
            "by_mix": absolute,
            "pass": all(row["pass"] for row in absolute.values()),
        }
        filtered_rule = copy.deepcopy(spec)
        filtered_rule["cells"]["primary"]["seeds"] = eligible
        selection = candidate_selection(deltas, filtered_rule) if eligible else None
        futility_value = stat_value(spec["futility"]["stat"], deltas, spec)
        futility_ub = _bound(futility_value, 0.10, upper=True)
        futility = {
            **futility_value,
            "upper_bound": futility_ub,
            "threshold": 50.0,
            "triggered": futility_ub < 50.0,
        }
        failed = (["efficacy"] if not efficacy["pass"] else []) + [
            name for name, row in clauses.items() if not row["pass"]
        ]
        reasons = list(failed)
        if futility["triggered"] and reasons:
            reasons.append("NO_GO_EARLY: mass_futility")
        return {
            "schema": "redesign-t7-phase-r-decision/v1",
            "rule_sha256": spec["rule_sha256"],
            "valid": True,
            "status": "STRICT_PASS" if not reasons else "FAIL",
            "action": "STOP",
            "reasons": reasons,
            "efficacy": efficacy,
            "clauses": clauses,
            "futility": futility,
            "candidate_selection": selection,
            "candidate_eligible_seeds": eligible,
            "tier2": tier2,
            "sensitivity": sensitivity,
        }
    except (DecisionError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "schema": "redesign-t7-phase-r-decision/v1",
            "rule_sha256": spec["rule_sha256"],
            "valid": False,
            "status": "INVALID_ANALYSIS",
            "action": "HALT",
            "reasons": [f"invalid_analysis: {exc}"],
            "candidate_selection": None,
        }
