"""One-look E2 paired endpoint screen; never a registered or release verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path
from statistics import mean, variance
from typing import Any

from research.bl_ppo import bank as banks
from research.bl_ppo.rule import E2_ALPHA, RULE, holm_rejections, mass_pvalue
from research.published_ledger import bind_published_ledger_check, require_published_ledger

ARMS = ("R", "F", "B8")
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def _number(value: Any) -> float:
    """Return a finite scalar while rejecting bool's accidental numeric coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("ledger requires finite numeric values")
    return float(value)


def _trace_ledger(trace: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    """Validate a terminal trace and expose the information E8 must retain."""
    if not isinstance(trace, list) or not trace or len(trace) > horizon:
        raise ValueError("missing or oversized endpoint trace")
    if [step.get("frame") for step in trace] != list(range(len(trace))):
        raise ValueError("endpoint trace frame sequence is not contiguous")
    alive, masses = [], []
    for step in trace:
        if not isinstance(step, dict) or type(step.get("alive")) is not bool:
            raise ValueError("endpoint trace lacks a boolean alive state")
        mass = _number(step.get("mass"))
        if mass < 0 or (not step["alive"] and mass != 0):
            raise ValueError("endpoint trace has invalid dead-frame mass")
        alive.append(step["alive"])
        masses.append(mass)
    if len(trace) < horizon and alive[-1]:
        raise ValueError("endpoint trace ended before terminal death")
    death = next((frame for frame, value in enumerate(alive) if not value), None)
    if death is not None and any(alive[death:]):
        raise ValueError("terminal-hero trace resurrected after death")
    return {
        "alive": alive,
        "mass": masses,
        "death_frame": death,
        "last_alive_mass": masses[death - 1] if death else (masses[-1] if alive[-1] else 0.0),
    }


def _length_bin(length: float) -> str:
    if length < 400:
        return "lt400"
    if length < 800:
        return "400_799"
    if length < 1200:
        return "800_1199"
    return "ge1200"


def _frame_bin(frame: int) -> str:
    if frame < 2000:
        return "lt2000"
    if frame < 3500:
        return "2000_3499"
    return "ge3500"


def _world_ledger(controllers: dict[str, dict[str, Any]], horizon: int) -> dict[str, Any]:
    """Derive lossless mass and endpoint accounting from all three controller traces."""
    if "B8" not in controllers:
        raise ValueError("mass ledger needs B8 trace")
    traces = {
        arm: _trace_ledger(metrics.get("trace"), horizon) for arm, metrics in controllers.items()
    }
    output: dict[str, Any] = {}
    baseline = traces["B8"]
    for arm, candidate in traces.items():
        if arm == "B8":
            continue
        both = candidate_only = b8_only = 0.0
        occupancy: dict[str, int] = {}
        long_body_deltas = [
            candidate["mass"][frame] - candidate["mass"][frame - 1]
            for frame in range(1, len(candidate["mass"]))
            if candidate["alive"][frame]
            and candidate["alive"][frame - 1]
            and candidate["mass"][frame] >= 800
            and candidate["mass"][frame - 1] >= 800
        ]
        for frame in range(horizon):
            ca = frame < len(candidate["alive"]) and candidate["alive"][frame]
            ba = frame < len(baseline["alive"]) and baseline["alive"][frame]
            cm = candidate["mass"][frame] if ca else 0.0
            bm = baseline["mass"][frame] if ba else 0.0
            if ca and ba:
                both += cm - bm
            elif ca:
                candidate_only += cm
            elif ba:
                b8_only += bm
            if ca:
                key = f"{_frame_bin(frame)}:{_length_bin(cm)}"
                occupancy[key] = occupancy.get(key, 0) + 1
        total = both + candidate_only - b8_only
        expected = _number(controllers[arm]["mass_integral"]) - _number(
            controllers["B8"]["mass_integral"]
        )
        if not math.isclose(total / horizon, expected, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("mass ledger does not reconcile with full-horizon mass integral")
        death = baseline["death_frame"]
        endpoint = None
        if death is not None:
            rescued = death < len(candidate["alive"]) and candidate["alive"][death]
            re_death = bool(rescued and candidate["death_frame"] is not None)
            endpoint = {
                "b8_death_frame_bin": _frame_bin(death),
                "b8_death_length_bin": _length_bin(baseline["last_alive_mass"]),
                "rescued": int(rescued),
                "full_horizon_rescues": int(rescued and candidate["death_frame"] is None),
                "re_death": int(re_death),
            }
        else:
            endpoint = {
                "b8_death_frame_bin": "H5000_survivor",
                "b8_death_length_bin": _length_bin(baseline["last_alive_mass"]),
                "rescued": 0,
                "full_horizon_rescues": 0,
                "re_death": 0,
            }
        output[arm] = {
            "mass": {
                "both_alive_growth": both,
                "candidate_only": candidate_only,
                "b8_only": b8_only,
                "total": total,
                "per_frame": total / horizon,
            },
            "occupancy_by_frame_length": occupancy,
            "endpoint": endpoint,
            "candidate_growth_slope_l_ge_800": (
                sum(long_body_deltas) / len(long_body_deltas) if long_body_deltas else None
            ),
        }
    return output


def _validate_trace_metrics(controllers: dict[str, dict[str, Any]], horizon: int) -> None:
    """Bind compact endpoint metrics to the retained, immutable full traces."""
    for metrics in controllers.values():
        trace = metrics.get("trace")
        parsed = _trace_ledger(trace, horizon)
        if metrics.get("trace_sha256") != banks.canonical_sha(trace):
            raise ValueError("endpoint trace hash does not bind retained trace bytes")
        mass = sum(parsed["mass"]) / horizon
        alive_fraction = sum(parsed["alive"]) / horizon
        if (
            metrics.get("active_frames") != len(trace)
            or not math.isclose(
                _number(metrics.get("mass_integral")), mass, rel_tol=0.0, abs_tol=1e-10
            )
            or not math.isclose(
                _number(metrics.get("alive_frame_fraction")),
                alive_fraction,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            or metrics.get("no_death_episode")
            is not (len(trace) == horizon and all(parsed["alive"]))
        ):
            raise ValueError("endpoint metrics do not reconcile with retained full trace")


def _empty_e8_ledgers() -> dict[str, Any]:
    return {
        "mass_by_arm_mix": {
            arm: {
                mix: {
                    "both_alive_growth": 0.0,
                    "candidate_only": 0.0,
                    "b8_only": 0.0,
                    "total": 0.0,
                    "worlds": 0,
                    "growth_slope_l_ge_800_sum": 0.0,
                    "growth_slope_l_ge_800_count": 0,
                }
                for mix in banks.MIX_COUNTS
            }
            for arm in ("R", "F")
        },
        "occupancy_by_arm_mix": {arm: {mix: {} for mix in banks.MIX_COUNTS} for arm in ("R", "F")},
        "endpoint_by_arm_mix": {arm: {mix: {} for mix in banks.MIX_COUNTS} for arm in ("R", "F")},
    }


def _add_e8_ledger(ledgers: dict[str, Any], mix: str, world: dict[str, Any]) -> None:
    for arm, value in world.items():
        mass = ledgers["mass_by_arm_mix"][arm][mix]
        for key in ("both_alive_growth", "candidate_only", "b8_only", "total"):
            mass[key] += value["mass"][key]
        mass["worlds"] += 1
        slope = value["candidate_growth_slope_l_ge_800"]
        if slope is not None:
            mass["growth_slope_l_ge_800_sum"] += slope
            mass["growth_slope_l_ge_800_count"] += 1
        occupancy = ledgers["occupancy_by_arm_mix"][arm][mix]
        for key, count in value["occupancy_by_frame_length"].items():
            occupancy[key] = occupancy.get(key, 0) + count
        endpoint = value["endpoint"]
        if endpoint is not None:
            key = f"{endpoint['b8_death_frame_bin']}:{endpoint['b8_death_length_bin']}"
            bucket = ledgers["endpoint_by_arm_mix"][arm][mix].setdefault(
                key,
                {
                    "exposure": 0,
                    "rescues": 0,
                    "full_horizon_rescues": 0,
                    "re_deaths": 0,
                    "both_alive_growth": 0.0,
                    "candidate_only": 0.0,
                    "b8_only": 0.0,
                },
            )
            bucket["exposure"] += 1
            bucket["rescues"] += endpoint["rescued"]
            bucket["full_horizon_rescues"] += endpoint["full_horizon_rescues"]
            bucket["re_deaths"] += endpoint["re_death"]
            for name in ("both_alive_growth", "candidate_only", "b8_only"):
                bucket[name] += value["mass"][name]


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_training_status(item: dict[str, Any]) -> None:
    """Check operational counters and actual window evidence for scientific stops."""
    if not _valid_sha(item.get("receipt_sha256")):
        raise ValueError("training status lacks immutable stopped/endpoint receipt hash")
    if item.get("e9_rule_sha256") != banks.canonical_sha(RULE) or not _valid_sha(
        item.get("e9_receipt_sha256")
    ):
        raise ValueError("training status is not bound to the frozen admitted E9 rule and receipt")
    counters = item.get("counters", {})
    transitions = counters.get("hero_transitions")
    actor_transitions = counters.get("actor_transitions")
    critic_transitions = counters.get("critic_refit_transitions")
    critic_fit_transitions = counters.get("critic_refit_fit_transitions")
    if (
        type(transitions) is not int
        or type(actor_transitions) is not int
        or type(critic_transitions) is not int
        or type(critic_fit_transitions) is not int
        or min(transitions, actor_transitions, critic_transitions, critic_fit_transitions) < 0
        or transitions != actor_transitions + critic_transitions
        or critic_fit_transitions > critic_transitions
    ):
        raise ValueError("training counters must reconcile actor and critic-refit transitions")
    target = item.get("target_transitions")
    if type(target) is not int or target not in {4_000_000, 8_000_000, 20_000_000}:
        raise ValueError("training status lacks the E4-selected registered transition target")
    if actor_transitions > target or (item.get("complete") is True and actor_transitions != target):
        raise ValueError("training actor-transition endpoint does not reconcile with its E4 target")
    if item.get("stop_reason") != "tripwire":
        critic = item.get("critic_refit", {})
        heldout = critic.get("heldout_late_body", {})
        if (
            critic.get("required_transitions") != 2_000_000
            or critic.get("observed_transitions") != critic_transitions
            or critic.get("observed_fit_transitions") != critic_fit_transitions
            or critic_transitions < 2_000_000
            or critic_fit_transitions < 2_000_000
            or critic.get("popart") is not True
            or type(heldout.get("count")) is not int
            or heldout["count"] < 1
            or _number(heldout.get("explained_variance")) < 0.1
            or heldout.get("beats_baseline") is not True
            or not _number(heldout.get("mse")) < _number(heldout.get("baseline_mse"))
            or critic.get("qualified") is not True
            or item.get("critic_qualified") is not True
        ):
            raise ValueError(
                "critic must pass PopArt late-body held-out qualification before actor screening"
            )
        boost_floor = item.get("boost_contract_floor", {})
        if (
            boost_floor.get("floor") != 0.005
            or type(boost_floor.get("below_floor")) is not bool
            or _number(boost_floor.get("candidate_rate")) < 0
            or _number(boost_floor.get("b8_rate")) < 0
            or boost_floor["below_floor"] != (boost_floor["candidate_rate"] < boost_floor["floor"])
        ):
            raise ValueError("missing or inconsistent 0.5% boost-contract floor evidence")
    if not isinstance(item.get("tripwires"), list):
        raise ValueError("tripwire evidence must be a list")
    for tripwire in item["tripwires"]:
        if not isinstance(tripwire, dict):
            raise ValueError("tripwires require structured evidence")
        kind = tripwire.get("kind")
        if kind == "nonfinite":
            if tripwire.get("stage") not in {"loss", "logits", "value", "gradient", "parameters"}:
                raise ValueError("nonfinite tripwire requires the failed numerical stage")
            continue
        if kind not in {"action_collapse", "boost_share"}:
            raise ValueError("unsupported tripwire kind")
        counts, count = tripwire.get("action_counts"), tripwire.get("window_count")
        if (
            type(count) is not int
            or count != 100_000
            or count > transitions
            or not isinstance(counts, list)
            or len(counts) != 6
            or any(type(v) is not int or v < 0 for v in counts)
            or sum(counts) != count
        ):
            raise ValueError("tripwire requires a complete matched 100k decision window")
        if kind == "action_collapse" and max(counts) / count <= 0.9:
            raise ValueError("action collapse threshold not crossed")
        if kind == "boost_share":
            b8_count = tripwire.get("b8_boost_count")
            if type(b8_count) is not int or not 0 <= b8_count <= count:
                raise ValueError("invalid matched B8 boost count")
            if sum(counts[3:]) <= 3 * b8_count:
                raise ValueError("boost-share threshold not crossed")


def _tripwired_arms(result: dict[str, Any]) -> set[str]:
    """Recognize an intentional scientific stop, distinct from an interruption."""
    return {
        arm
        for arm in ("R", "F")
        if result.get("training_status", {}).get(arm, {}).get("stop_reason") == "tripwire"
        and result["training_status"][arm].get("tripwires")
    }


def paired_bounds(groups: dict[str, list[float]], alpha: float = 0.1) -> dict[str, Any]:
    """Equal-mix paired-world Welch bounds, allowing unequal mix sample sizes."""
    from src.scripts.eval_stats import student_t_isf

    if set(groups) != set(banks.MIX_COUNTS) or any(len(v) < 2 for v in groups.values()):
        raise ValueError("three paired mixes with at least two worlds each are required")
    if any(not math.isfinite(x) for values in groups.values() for x in values):
        raise ValueError("nonfinite paired delta")
    terms = [variance(values) / len(values) / 9 for values in groups.values()]
    estimate = mean(mean(values) for values in groups.values())
    var = sum(terms)
    denominator = sum(t * t / (len(v) - 1) for t, v in zip(terms, groups.values()))
    df = var * var / denominator if denominator else None
    se = math.sqrt(var)
    critical = student_t_isf(alpha, df) if df is not None else 0.0
    return {
        "estimate": estimate,
        "se": se,
        "df": df,
        "alpha_one_sided": alpha,
        "lower": estimate - critical * se,
        "upper": estimate + critical * se,
        "unit": "paired world",
        "mix_weight": 1 / 3,
        "n_by_mix": {mix: len(v) for mix, v in groups.items()},
    }


def _single_bounds(values: list[float], alpha: float = 0.05) -> dict[str, float]:
    from src.scripts.eval_stats import student_t_isf

    se = math.sqrt(variance(values) / len(values))
    estimate = mean(values)
    width = student_t_isf(alpha, len(values) - 1) * se
    return {
        "estimate": estimate,
        "se": se,
        "lower": estimate - width,
        "upper": estimate + width,
        "alpha_one_sided": alpha,
    }


def _validate_results(bank: dict[str, Any], audit: dict[str, Any], result: dict[str, Any]) -> None:
    banks.check_audit(bank, audit)
    if result.get("schema") != "bl-ppo-screen-results/v2":
        raise ValueError("unrecognized screen result")
    if result.get("bank_semantic_sha256") != bank["allocation_sha256"]:
        raise ValueError("results do not bind the audited bank")
    killed = _tripwired_arms(result)
    active = set(ARMS) - killed
    expected = {row["world_id"]: row for row in bank["rows"]} if len(killed) < 2 else {}
    rows = result.get("rows", [])
    if len(rows) != len(expected) or len({r.get("world_id") for r in rows}) != len(expected):
        raise ValueError("incomplete or duplicate paired worlds")
    if {r.get("world_id") for r in rows} != set(expected):
        raise ValueError("unexpected paired world identities")
    if result.get("complete") is not True or result.get("cap_expired") is not False:
        raise ValueError("incomplete run or exhausted cap")
    if result.get("parity_pass") is not True:
        raise ValueError("runtime parity qualification missing or failed")
    checkpoints = result.get("checkpoint_sha256", {})
    if not active.issubset(checkpoints) or any(not _valid_sha(v) for v in checkpoints.values()):
        raise ValueError("checkpoint identities missing")
    if checkpoints["B8"] != B8_SHA256:
        raise ValueError("incorrect B8 comparator")
    for row in rows:
        world = expected[row["world_id"]]
        if any(row.get(k) != world[k] for k in ("world_seed", "mix", "horizon")):
            raise ValueError("world semantics differ from audited bank")
        if row.get("roster_sha256") != banks.canonical_sha(world["opponents"]):
            raise ValueError("opponent roster differs from audited bank")
        if not row.get("cpu_identity") or set(row.get("controllers", {})) != active:
            raise ValueError("paired world missing one CPU identity or one controller")
        for arm, metrics in row["controllers"].items():
            if metrics.get("checkpoint_sha256") != checkpoints[arm]:
                raise ValueError("endpoint checkpoint substitution")
            if metrics.get("frames_stepped") != 5000:
                raise ValueError("incorrect terminal horizon accounting")
            if type(metrics.get("no_death_episode")) is not bool:
                raise ValueError("invalid no-death indicator")
            for key in ("mass_integral", "alive_frame_fraction"):
                number = metrics.get(key)
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    raise ValueError("missing numerical metric")
                if not math.isfinite(number) or number < 0:
                    raise ValueError("invalid numerical metric")
            if metrics["alive_frame_fraction"] > 1:
                raise ValueError("alive fraction exceeds horizon")
        ledger = row.get("e8_ledger")
        if not isinstance(ledger, dict) or set(ledger) != active - {"B8"}:
            raise ValueError("every paired world requires E8 mass, occupancy and endpoint ledgers")
        for arm, item in ledger.items():
            mass = item.get("mass", {})
            if set(mass) != {
                "both_alive_growth",
                "candidate_only",
                "b8_only",
                "total",
                "per_frame",
            }:
                raise ValueError("incomplete E8 three-term mass ledger")
            for value in mass.values():
                _number(value)
            if (
                not math.isclose(
                    mass["both_alive_growth"] + mass["candidate_only"] - mass["b8_only"],
                    mass["total"],
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                or not math.isclose(
                    mass["total"] / row["horizon"], mass["per_frame"], rel_tol=0.0, abs_tol=1e-10
                )
                or not math.isclose(
                    mass["per_frame"],
                    row["controllers"][arm]["mass_integral"]
                    - row["controllers"]["B8"]["mass_integral"],
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
            ):
                raise ValueError("E8 mass ledger does not reconcile to endpoint metric")
            if not isinstance(item.get("occupancy_by_frame_length"), dict):
                raise ValueError("E8 occupancy ledger is missing")
            endpoint = item.get("endpoint")
            if endpoint is not None and (
                set(endpoint)
                != {
                    "b8_death_frame_bin",
                    "b8_death_length_bin",
                    "rescued",
                    "full_horizon_rescues",
                    "re_death",
                }
                or endpoint["rescued"] not in (0, 1)
                or endpoint["full_horizon_rescues"] not in (0, 1)
                or endpoint["re_death"] not in (0, 1)
                or endpoint["full_horizon_rescues"] > endpoint["rescued"]
                or endpoint["re_death"] > endpoint["rescued"]
            ):
                raise ValueError("E8 endpoint rescue/re-death ledger is invalid")
    status = result.get("training_status", {})
    for arm in ("R", "F"):
        item = status.get(arm, {})
        _validate_training_status(item)
        if arm in killed:
            continue
        if item.get("complete") is not True:
            raise ValueError("training endpoint incomplete without a scientific tripwire")
        if item.get("checkpoint_sha256") != checkpoints[arm]:
            raise ValueError("training status does not bind final endpoint")

    ledgers = result.get("e8_ledgers")
    if not isinstance(ledgers, dict) or set(ledgers) != {
        "mass_by_arm_mix",
        "occupancy_by_arm_mix",
        "endpoint_by_arm_mix",
    }:
        raise ValueError("missing aggregate E8 ledgers")
    for arm in active - {"B8"}:
        for mix in banks.MIX_COUNTS:
            aggregate = ledgers["mass_by_arm_mix"].get(arm, {}).get(mix, {})
            matching = [row["e8_ledger"][arm] for row in rows if row["mix"] == mix]
            if aggregate.get("worlds") != len(matching):
                raise ValueError("E8 aggregate ledger world count does not reconcile")
            for term in ("both_alive_growth", "candidate_only", "b8_only", "total"):
                if not math.isclose(
                    _number(aggregate.get(term)),
                    sum(item["mass"][term] for item in matching),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError("E8 aggregate mass ledger differs from paired world ledgers")
            if not math.isclose(
                _number(aggregate.get("both_alive_growth"))
                + _number(aggregate.get("candidate_only"))
                - _number(aggregate.get("b8_only")),
                _number(aggregate.get("total")),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError("E8 aggregate three-term mass ledger does not reconcile")
    recomputed = _empty_e8_ledgers()
    for row in rows:
        _add_e8_ledger(recomputed, row["mix"], row["e8_ledger"])
    if banks.canonical_sha(recomputed) != banks.canonical_sha(ledgers):
        raise ValueError("E8 occupancy or endpoint aggregates differ from paired world ledgers")


def analyze(bank: dict[str, Any], audit: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Apply only the pre-data E2 rules; invalid/incomplete evidence wins."""
    output: dict[str, Any] = {
        "schema": "bl-ppo-screen-decision/v1",
        "development_only": True,
        "registered_verdict_eligible": False,
        "recipe_claim": False,
        "bank_semantic_sha256": bank.get("allocation_sha256"),
    }
    try:
        _validate_results(bank, audit, result)
    except (ValueError, KeyError, TypeError) as exc:
        return {**output, "status": "INCOMPLETE", "errors": [str(exc)]}
    rows = result["rows"]

    def deltas(left: str, right: str, key: str) -> dict[str, list[float]]:
        return {
            mix: [
                float(r["controllers"][left][key]) - float(r["controllers"][right][key])
                for r in rows
                if r["mix"] == mix
            ]
            for mix in banks.MIX_COUNTS
        }

    arms = {}
    mass_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ("R", "F"):
        if arm in _tripwired_arms(result):
            arms[arm] = {
                "status": "KILL",
                "kill_reasons": result["training_status"][arm]["tripwires"],
                "screened": False,
                "reason": "intentional scientific tripwire before endpoint",
            }
            continue
        mass = paired_bounds(deltas(arm, "B8", "mass_integral"))
        mass["pvalue"] = mass_pvalue(mass)
        mass_by_arm[arm] = mass
        survival_groups = deltas(arm, "B8", "alive_frame_fraction")
        survival = paired_bounds(survival_groups, alpha=0.05)
        per_mix = {mix: _single_bounds(values) for mix, values in survival_groups.items()}
        absolute = {
            mix: mean(
                r["controllers"][arm]["alive_frame_fraction"] for r in rows if r["mix"] == mix
            )
            for mix in banks.MIX_COUNTS
        }
        no_death = mean(deltas(arm, "B8", "no_death_episode")["scripted"])
        band_pass = (
            survival["lower"] > -0.05
            and all(x["lower"] > -0.075 for x in per_mix.values())
            and all(x >= 0.30 for x in absolute.values())
        )
        tripwires = result["training_status"][arm]["tripwires"]
        reasons = list(tripwires)
        if mass["upper"] <= 0:
            reasons.append("pooled_mass_UB90_le_zero")
        status = (
            "KILL"
            if reasons
            else (
                "PENDING_HOLM"
                if mass["estimate"] >= 10 and mass["lower"] > 0 and no_death >= 0 and band_pass
                else "INCONCLUSIVE"
            )
        )
        arms[arm] = {
            "status": status,
            "kill_reasons": reasons,
            "pooled_mass": mass,
            "scripted_no_death_delta": no_death,
            "survival_band_v2": {
                "pass": band_pass,
                "pooled": survival,
                "by_mix": per_mix,
                "candidate_alive_fraction": absolute,
            },
        }
    pvalues = {arm: mass_by_arm.get(arm, {}).get("pvalue", 1.0) for arm in ("R", "F")}
    holm = holm_rejections(pvalues, alpha=E2_ALPHA)
    for arm, item in arms.items():
        if item["status"] == "PENDING_HOLM":
            item["status"] = "GO" if holm[arm] else "INCONCLUSIVE"
        item["holm_e2_alpha_005"] = {"pvalue": pvalues[arm], "rejected": holm[arm]}
        if (
            arm in result["training_status"]
            and "boost_contract_floor" in result["training_status"][arm]
        ):
            item["training_boost_contract_floor"] = result["training_status"][arm][
                "boost_contract_floor"
            ]
        rates = {}
        for mix in banks.MIX_COUNTS:
            metrics = [row["controllers"].get(arm) for row in rows if row["mix"] == mix]
            if not metrics or any(
                m is None
                or type(m.get("boost_decisions")) is not int
                or type(m.get("hero_decisions")) is not int
                or not 0 <= m["boost_decisions"] <= m["hero_decisions"]
                or m["hero_decisions"] < 1
                for m in metrics
            ):
                continue
            rates[mix] = sum(m["boost_decisions"] for m in metrics) / sum(
                m["hero_decisions"] for m in metrics
            )
        pooled_boost = mean(rates.values()) if len(rates) == 3 else None
        item["boost_contract_floor"] = {
            "scope": "greedy final endpoint; equal mix mean of observed action shares",
            "by_mix": rates,
            "candidate_rate": pooled_boost,
            "floor": 0.005,
            "below_floor": pooled_boost < 0.005 if pooled_boost is not None else None,
            "affects_status": False,
        }
    contrast = None if _tripwired_arms(result) else paired_bounds(deltas("R", "F", "mass_integral"))
    return {
        **output,
        "status": "COMPLETE",
        "arms": arms,
        "holm_e2": {"alpha": E2_ALPHA, "pvalues": pvalues, "rejections": holm},
        "R_minus_F": contrast,
        "e8_ledgers": result["e8_ledgers"],
    }


def evaluate_world(
    world: dict[str, Any], checkpoint: Path, arm: str, *, driver_factory: Any = None
) -> dict[str, Any]:
    """Run a terminal hero with deterministic policy and an auditable action trace.

    The driver owns the same frozen roster/GridBatchSim semantics as collection.
    Opponent respawns remain enabled; hero scoring ends at its first death.
    """
    import numpy as np
    import torch

    from research.bl_ppo.exports import digest
    from research.bl_ppo.model import PPOPolicy
    from src.model.ego2s_network import load_ego2s_checkpoint

    if driver_factory is None:
        from research.bl_ppo.collector import make_eval_driver

        driver_factory = make_eval_driver
    model = (
        load_ego2s_checkpoint(str(checkpoint), "cpu")
        if arm == "B8"
        else PPOPolicy.from_checkpoint(checkpoint, "cpu")
    )
    model.eval()
    if arm != "B8" and model.arm != arm:
        raise ValueError("screen checkpoint belongs to a different arm")
    driver = driver_factory(int(world["world_seed"]), world["opponents"], int(world["horizon"]))
    mass, alive_frames, trace, hidden = 0.0, 0, [], None
    started = time.monotonic()
    for frame in range(int(world["horizon"])):
        obs = driver.observe_hero()
        tensors = [torch.as_tensor(obs[k]) for k in ("local", "global", "scalars")]
        with torch.inference_mode():
            if arm == "B8":
                logits = model(*tensors)
            else:
                logits, _, hidden = model(*tensors, hidden=hidden)
        values = logits.detach().cpu().numpy().reshape(6)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite evaluation logits")
        # Hero is bare. B8's advisory v8 planes are observations, not a veto.
        mask = np.asarray(obs["mask"], dtype=bool).reshape(6).copy()
        if not mask.any():
            mask[:3] = True
        action = int(np.where(mask, values.astype(np.float64), -np.inf).argmax())
        observation_digest = hashlib.sha256()
        for key in ("local", "global", "scalars", "mask"):
            array = np.ascontiguousarray(obs[key])
            observation_digest.update(str((key, array.shape, array.dtype.str)).encode())
            observation_digest.update(array.tobytes())
        observed_sha = observation_digest.hexdigest()
        step = driver.step(action)
        alive = bool(step["alive"])
        if alive:
            alive_frames += 1
            mass += float(step["mass"])
        order = np.argsort(values)
        trace.append(
            {
                "frame": frame,
                "action": action,
                "observation_sha256": observed_sha,
                "logits_hex": [float(v).hex() for v in values],
                "top2_margin": float(values[order[-1]] - values[order[-2]]),
                "alive": alive,
                "mass": float(step["mass"]) if alive else 0.0,
            }
        )
        if step["done"]:
            break
    horizon = int(world["horizon"])
    return {
        "checkpoint_sha256": digest(checkpoint),
        "mass_integral": mass / horizon,
        "alive_frame_fraction": alive_frames / horizon,
        "no_death_episode": alive_frames == horizon,
        "frames_stepped": horizon,
        "active_frames": len(trace),
        "hero_decisions": len(trace),
        "boost_decisions": sum(step["action"] >= 3 for step in trace),
        "trace_sha256": banks.canonical_sha(trace),
        "trace": trace,
        "wall_seconds": time.monotonic() - started,
    }


def parity_artifact(
    bank: dict[str, Any], checkpoints: dict[str, Path], *, preflight: bool = False
) -> dict[str, Any]:
    """Generate actual CPU reference traces on disjoint, non-decision worlds."""
    banks.validate_bank(bank)
    if _checkpoint_hashes(checkpoints).get("B8") != B8_SHA256:
        raise ValueError("parity requires the released B8 comparator")
    worlds = bank["excluded_preflight_rows" if preflight else "excluded_parity_rows"]
    result = {
        "schema": "bl-ppo-runtime-parity/v1",
        "preflight_only": preflight,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "source_sha256": _source_hashes(),
        "runtime": _runtime(),
        "checkpoint_sha256": _checkpoint_hashes(checkpoints),
        "rows": [],
    }
    for world in worlds:
        result["rows"].append(
            {
                "world_id": world["world_id"],
                "controllers": {
                    arm: evaluate_world(world, path, arm) for arm, path in checkpoints.items()
                },
            }
        )
    return result


def compare_parity(reference: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Require actual full-H5000 Mac traces; explain mismatches and fail closed.

    Hex logits and top-two margins are retained to investigate float32 ties.
    A tiny preflight never qualifies the development screen.
    """
    errors = []
    numerical_differences = []
    for key in ("schema", "bank_semantic_sha256", "source_sha256", "checkpoint_sha256"):
        if reference.get(key) != runtime.get(key):
            errors.append(f"parity {key} identity differs")
    if reference.get("schema") != "bl-ppo-runtime-parity/v1":
        errors.append("wrong parity schema")
    if reference.get("preflight_only") is not False or runtime.get("preflight_only") is not False:
        errors.append("tiny preflight cannot qualify full screen")
    if reference.get("runtime", {}).get("system") != "Darwin":
        errors.append("reference did not originate on Mac")
    if len(reference.get("rows", [])) != 3 or len(runtime.get("rows", [])) != 3:
        errors.append("parity requires all three excluded H5000 worlds")
    for left, right in zip(reference.get("rows", []), runtime.get("rows", [])):
        if left.get("world_id") != right.get("world_id"):
            errors.append("parity world order differs")
        if set(left.get("controllers", {})) != set(right.get("controllers", {})):
            errors.append("parity controller identities differ")
            continue
        for arm in left.get("controllers", {}):
            a, b = left["controllers"][arm], right["controllers"][arm]
            for item in (a, b):
                trace = item.get("trace", [])
                if item.get("frames_stepped") != 5000 or item.get(
                    "trace_sha256"
                ) != banks.canonical_sha(item.get("trace")):
                    errors.append("incomplete or forged H5000 action trace")
                if (
                    not trace
                    or len(trace) > 5000
                    or [step.get("frame") for step in trace] != list(range(len(trace)))
                    or (len(trace) < 5000 and trace[-1].get("alive") is not False)
                ):
                    errors.append("truncated nonterminal parity trace")
            semantic_keys = ("frame", "action", "observation_sha256", "alive", "mass")
            left_semantics = [
                {k: step.get(k) for k in semantic_keys} for step in a.get("trace", [])
            ]
            right_semantics = [
                {k: step.get(k) for k in semantic_keys} for step in b.get("trace", [])
            ]
            if left_semantics != right_semantics:
                errors.append(
                    f"{left['world_id']}/{arm}: trace differs; "
                    "inspect stored logits and top2 margins"
                )
            differences = [
                abs(float.fromhex(x) - float.fromhex(y))
                for sa, sb in zip(a.get("trace", []), b.get("trace", []))
                for x, y in zip(sa.get("logits_hex", []), sb.get("logits_hex", []))
                if x != y
            ]
            if differences:
                numerical_differences.append(
                    {
                        "world_id": left["world_id"],
                        "arm": arm,
                        "different_logit_values": len(differences),
                        "maximum_absolute_difference": max(differences),
                    }
                )
    return {
        "schema": "bl-ppo-runtime-parity-comparison/v1",
        "pass": not errors,
        "errors": errors,
        "logit_numerical_differences": numerical_differences,
        "scope": "three-world development smoke; no universal platform identity claim",
    }


def _source_hashes() -> dict[str, str]:
    from research.bl_ppo.exports import digest

    return {
        name: digest(Path(__file__).with_name(name))
        for name in ("collector.py", "model.py", "screen.py", "bank.py", "rule.py")
    }


def _runtime() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "host": platform.node(),
        "cpu_threads": 1,
    }


def _checkpoint_hashes(checkpoints: dict[str, Path]) -> dict[str, str]:
    from research.bl_ppo.exports import digest

    return {arm: digest(path) for arm, path in checkpoints.items()}


def execute_screen(
    bank: dict[str, Any],
    audit: dict[str, Any],
    checkpoints: dict[str, Path],
    training_status: dict[str, Any],
    reference: dict[str, Any],
    runtime: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    """Evaluate paired worlds serially with create-only resumable world exports."""
    from research.bl_ppo.exports import export_json, verify_export

    banks.check_audit(bank, audit)
    for arm in ("R", "F"):
        _validate_training_status(training_status[arm])
    killed = _tripwired_arms({"training_status": training_status})
    active = set(ARMS) - killed
    if set(checkpoints) != active:
        raise ValueError("screen needs B8 and exactly the non-tripwired final endpoints")
    checkpoint_hashes = _checkpoint_hashes(checkpoints)
    if checkpoint_hashes.get("B8") != B8_SHA256:
        raise ValueError("screen requires the released B8 comparator")
    for arm in active - {"B8"}:
        if checkpoint_hashes[arm] != training_status[arm].get("checkpoint_sha256"):
            raise ValueError("screen checkpoint differs from completed training endpoint")
    comparison = compare_parity(reference, runtime)
    if (
        not comparison["pass"]
        or reference.get("checkpoint_sha256") != checkpoint_hashes
        or reference.get("bank_semantic_sha256") != bank["allocation_sha256"]
        or reference.get("source_sha256") != _source_hashes()
    ):
        raise ValueError("actual bound Mac/runtime parity evidence required before screen")
    result = {
        "schema": "bl-ppo-screen-results/v2",
        "claim_status": "COMPUTE_ONLY_UNCLAIMED",
        "complete": True,
        "cap_expired": False,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "checkpoint_sha256": checkpoint_hashes,
        "training_status": training_status,
        "parity_pass": True,
        "parity_comparison": comparison,
        "reference_semantic_sha256": banks.canonical_sha(reference),
        "runtime_semantic_sha256": banks.canonical_sha(runtime),
        "source_sha256": _source_hashes(),
        "rows": [],
        "e8_ledgers": _empty_e8_ledgers(),
    }
    for index, world in enumerate(bank["rows"] if len(killed) < 2 else []):
        path = out / "worlds" / f"world-{index:04d}.json"
        binding = {
            "world_id": world["world_id"],
            "bank_semantic_sha256": bank["allocation_sha256"],
            "checkpoint_sha256": checkpoint_hashes,
            "source_sha256": _source_hashes(),
        }
        if path.exists():
            verify_export(path)
            saved = json.loads(path.read_text())
            if saved.get("binding") != binding:
                raise ValueError("stale or substituted partial paired world")
            row = saved["row"]
        else:
            row = {
                **{k: world[k] for k in ("world_id", "world_seed", "mix", "horizon")},
                "roster_sha256": banks.canonical_sha(world["opponents"]),
                "cpu_identity": _runtime(),
                "controllers": {},
            }
            for arm in ARMS:
                if arm in active:
                    row["controllers"][arm] = evaluate_world(world, checkpoints[arm], arm)
            row["e8_ledger"] = _world_ledger(row["controllers"], int(world["horizon"]))
            export_json(path, {"binding": binding, "row": row})
        if "e8_ledger" not in row:
            # v2 world exports are the sole resumable format; a v1 partial lacks
            # trace-derived accounting and cannot be silently promoted.
            raise ValueError("partial paired-world export predates mandatory E8 ledger")
        _validate_trace_metrics(row["controllers"], int(world["horizon"]))
        derived_ledger = _world_ledger(row["controllers"], int(world["horizon"]))
        if banks.canonical_sha(row["e8_ledger"]) != banks.canonical_sha(derived_ledger):
            raise ValueError("cached paired-world E8 ledger differs from retained full trace")
        # Keep full traces in bounded immutable world exports; the aggregate
        # contains only statistics and their trace identities.
        compact = {
            **row,
            "controllers": {
                arm: {k: v for k, v in metrics.items() if k != "trace"}
                for arm, metrics in row["controllers"].items()
            },
        }
        result["rows"].append(compact)
        _add_e8_ledger(result["e8_ledgers"], row["mix"], row["e8_ledger"])
        print(
            json.dumps({"paired_worlds_complete": index + 1, "paired_worlds_total": 160}),
            flush=True,
        )
    _validate_results(bank, audit, result)
    export_json(out / "results.json", result)
    return result


def load_training_status(paths: dict[str, Path]) -> dict[str, Any]:
    """Bind scientific stop/endpoint evidence to its own immutable export bytes."""
    from research.bl_ppo.exports import verify_export

    if set(paths) != {"R", "F"}:
        raise ValueError("both immutable arm status receipts are required")
    result = {}
    for arm, path in paths.items():
        checksum = verify_export(path)
        value = json.loads(path.read_text())
        if value.get("arm") != arm or value.get("scientific") is not True:
            raise ValueError("wrong arm or preflight receipt cannot admit a scientific screen")
        value["receipt_sha256"] = checksum
        _validate_training_status(value)
        result[arm] = value
    return result


def main(argv: list[str] | None = None) -> int:
    """Prepare parity, evaluate paired endpoints, or analyze completed receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--write-parity", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--b8-checkpoint", type=Path)
    parser.add_argument("--r-checkpoint", type=Path)
    parser.add_argument("--f-checkpoint", type=Path)
    parser.add_argument("--training-status", type=Path)
    parser.add_argument("--r-status", type=Path)
    parser.add_argument("--f-status", type=Path)
    parser.add_argument("--mac-reference", type=Path)
    parser.add_argument("--runtime-parity", type=Path)
    parser.add_argument("--authoring-repo", type=Path)
    parser.add_argument("--ledger")
    args = parser.parse_args(argv)
    import os

    from research.bl_ppo.exports import export_json, verify_export

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    for path in (args.bank, args.audit):
        verify_export(path)
    value, audit = (json.loads(p.read_text()) for p in (args.bank, args.audit))
    banks.check_audit(value, audit)
    checkpoints = {
        arm: path
        for arm, path in (
            ("B8", args.b8_checkpoint),
            ("R", args.r_checkpoint),
            ("F", args.f_checkpoint),
        )
        if path is not None
    }
    if args.write_parity:
        if "B8" not in checkpoints:
            parser.error("--write-parity requires --b8-checkpoint")
        export_json(args.out, parity_artifact(value, checkpoints, preflight=args.preflight_only))
        return 0
    if args.evaluate:
        if not all((args.r_status, args.f_status, args.mac_reference, args.runtime_parity)):
            parser.error("--evaluate requires r-status, f-status, mac-reference and runtime-parity")
        for path in (args.mac_reference, args.runtime_parity):
            verify_export(path)
        execute_screen(
            value,
            audit,
            checkpoints,
            load_training_status({"R": args.r_status, "F": args.f_status}),
            json.loads(args.mac_reference.read_text()),
            json.loads(args.runtime_parity.read_text()),
            args.out,
        )
        return 0
    if args.results is None:
        parser.error("analysis requires --results")
    if args.authoring_repo is None or args.ledger is None:
        parser.error("analysis is a claim and requires --authoring-repo and --ledger")
    proof = require_published_ledger(args.authoring_repo, args.ledger)
    verify_export(args.results)
    decision = analyze(*(json.loads(p.read_text()) for p in (args.bank, args.audit, args.results)))
    export_json(args.out, bind_published_ledger_check(decision, proof))
    print(json.dumps({"status": decision["status"], "out": str(args.out)}))
    return 0 if decision["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
