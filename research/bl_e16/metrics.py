"""Frozen E16 whole-life ledger and one-look decision statistics.

The evaluator supplies one complete H5000 length trace per controller and
collision-audited natural world.  This module deliberately knows no gameplay:
it makes the outcome arithmetic inspectable and rejects incomplete or
non-whole-life traces before it computes a decision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, variance
from typing import Any, Iterable, Mapping, Sequence

from src.scripts.eval_stats import holm, student_t_isf, student_t_sf

MIXES = ("scripted", "frozen", "mixed")
HORIZON = 5_000
FAMILY_ALPHA = 0.10
DEATH_FRAME_BINS = ("<2000", "2000-2999", "3000-3999", "4000-4999", "survivor")
LAST_LENGTH_BINS = ("<700", "700-999", "1000-1199", ">=1200")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite real number")
    return parsed


def _lengths(values: Sequence[float], label: str, horizon: int) -> tuple[float, ...]:
    if len(values) != horizon:
        raise ValueError(f"{label} must contain exactly {horizon} whole-life frames")
    parsed = tuple(_number(value, f"{label}[{index}]") for index, value in enumerate(values))
    if any(value < 0.0 for value in parsed):
        raise ValueError(f"{label} cannot contain negative mass")
    # Terminal heroes cannot resurrect.  This turns a common trace truncation or
    # respawn bug into invalid evidence rather than a flattering rescue count.
    dead = False
    for value in parsed:
        if value == 0.0:
            dead = True
        elif dead:
            raise ValueError(f"{label} resurrects after terminal death")
    return parsed


def _death_frame(values: Sequence[float]) -> int | None:
    for frame, value in enumerate(values):
        if value == 0.0:
            return frame
    return None


def _death_frame_bin(frame: int | None) -> str:
    if frame is None:
        return "survivor"
    if frame < 2_000:
        return "<2000"
    if frame < 3_000:
        return "2000-2999"
    if frame < 4_000:
        return "3000-3999"
    return "4000-4999"


def _last_length_bin(length: float) -> str:
    if length < 700.0:
        return "<700"
    if length < 1_000.0:
        return "700-999"
    if length < 1_200.0:
        return "1000-1199"
    return ">=1200"


def whole_life_ledger(
    candidate_lengths: Sequence[float],
    b8_lengths: Sequence[float],
    *,
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """Decompose one natural paired H5000 world into E16's three-term ledger.

    Every mass value is a full-horizon mass integral (sum of length samples
    divided by ``horizon``).  Thus dead frames are zero and the exact identity
    ``total = common_alive + candidate_only - b8_only`` holds in the same unit
    as E16's G/J comparisons.
    """
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    candidate = _lengths(candidate_lengths, "candidate_lengths", horizon)
    b8 = _lengths(b8_lengths, "b8_lengths", horizon)
    common = candidate_only = b8_only = 0.0
    for candidate_mass, b8_mass in zip(candidate, b8):
        if candidate_mass > 0.0 and b8_mass > 0.0:
            common += candidate_mass - b8_mass
        elif candidate_mass > 0.0:
            candidate_only += candidate_mass
        elif b8_mass > 0.0:
            b8_only += b8_mass
    scale = float(horizon)
    b8_death = _death_frame(b8)
    candidate_death = _death_frame(candidate)
    b8_last = 0.0 if b8_death == 0 else (b8[-1] if b8_death is None else b8[b8_death - 1])
    candidate_survives = candidate_death is None
    b8_survives = b8_death is None
    tail_mean = None
    tail_retention = None
    productive = False
    if b8_death is not None and b8_death > 0:
        tail_mean = mean(candidate[b8_death:])
        tail_retention = tail_mean / b8_last if b8_last > 0.0 else None
        productive = candidate_survives and tail_retention is not None and tail_retention >= 0.90
    return {
        "horizon": horizon,
        "common_alive": common / scale,
        "candidate_only": candidate_only / scale,
        "b8_only": b8_only / scale,
        "total_delta": (common + candidate_only - b8_only) / scale,
        "candidate_mass_integral": sum(candidate) / scale,
        "b8_mass_integral": sum(b8) / scale,
        "b8_death_frame": b8_death,
        "candidate_death_frame": candidate_death,
        "b8_last_living_length": b8_last,
        "candidate_tail_mean_length": tail_mean,
        "tail_retention": tail_retention,
        "candidate_rescue": b8_death is not None and candidate_survives,
        "productive_rescue": productive,
        "induced_death": b8_survives and candidate_death is not None,
        # Collateral is evaluated over every natural world, including worlds in
        # which B8 survives; it is never conditioned on a B8 death.
        "collateral": max(0.0, -common / scale) + b8_only / scale,
    }


def ledger_by_mix(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate completed ledger rows, stratified by B8 death frame and length."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mix = row.get("mix")
        if mix not in MIXES:
            raise ValueError(f"unknown or missing mix: {mix!r}")
        ledger = whole_life_ledger(
            row["candidate_lengths"], row["b8_lengths"], horizon=row.get("horizon", HORIZON)
        )
        grouped[mix].append(ledger)
    if set(grouped) != set(MIXES) or any(not grouped[mix] for mix in MIXES):
        raise ValueError("all three mixes require at least one complete paired world")

    output: dict[str, Any] = {}
    for mix in MIXES:
        values = grouped[mix]
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in values:
            strata[
                _death_frame_bin(item["b8_death_frame"]),
                _last_length_bin(item["b8_last_living_length"]),
            ].append(item)
        rescue_tails = [
            item["tail_retention"]
            for item in values
            if item["candidate_rescue"] and item["tail_retention"] is not None
        ]
        all_b8_death_tails = [
            item["tail_retention"] for item in values if item["tail_retention"] is not None
        ]
        strata_rows = []
        for death_bin in DEATH_FRAME_BINS:
            for length_bin in LAST_LENGTH_BINS:
                members = strata[death_bin, length_bin]
                strata_rows.append(
                    {
                        "b8_death_frame_bin": death_bin,
                        "b8_last_living_length_bin": length_bin,
                        "n": len(members),
                        "common_alive": (
                            mean(item["common_alive"] for item in members) if members else None
                        ),
                        "candidate_only": (
                            mean(item["candidate_only"] for item in members) if members else None
                        ),
                        "b8_only": mean(item["b8_only"] for item in members) if members else None,
                        "productive_rescue_count": sum(
                            item["productive_rescue"] for item in members
                        ),
                        "candidate_rescue_count": sum(item["candidate_rescue"] for item in members),
                        "induced_death_count": sum(item["induced_death"] for item in members),
                    }
                )
        output[mix] = {
            "n": len(values),
            "common_alive": mean(item["common_alive"] for item in values),
            "candidate_only": mean(item["candidate_only"] for item in values),
            "b8_only": mean(item["b8_only"] for item in values),
            "total_delta": mean(item["total_delta"] for item in values),
            "collateral": mean(item["collateral"] for item in values),
            "productive_rescues_per_100": 100.0
            * mean(item["productive_rescue"] for item in values),
            "induced_deaths_per_100": 100.0 * mean(item["induced_death"] for item in values),
            # The admission retention readout is over actual survivor rescues.
            # All B8-death tails remain separately visible, including candidate
            # deaths, so a poor tail is not silently discarded from the audit.
            "rescue_tail_retention": rescue_tails,
            "all_b8_death_tail_retention": all_b8_death_tails,
            "b8_deathframe_x_last_length": strata_rows,
        }
    output["equal_mix"] = {
        key: mean(output[mix][key] for mix in MIXES)
        for key in ("common_alive", "candidate_only", "b8_only", "total_delta", "collateral")
    }
    return output


def paired_equal_mix_bounds(
    groups: Mapping[str, Sequence[float]], *, alpha: float = FAMILY_ALPHA
) -> dict[str, Any]:
    """Equal-mix world-clustered bound with the frozen E16 variance formula."""
    if not 0.0 < alpha < 0.5:
        raise ValueError("one-sided alpha must be in (0, .5)")
    if set(groups) != set(MIXES):
        raise ValueError("exactly scripted, frozen, and mixed paired groups are required")
    parsed = {mix: [_number(value, f"{mix} delta") for value in groups[mix]] for mix in MIXES}
    if any(len(values) < 2 for values in parsed.values()):
        raise ValueError("each mix requires at least two paired worlds")
    terms = [variance(parsed[mix]) / (9.0 * len(parsed[mix])) for mix in MIXES]
    estimate = mean(mean(parsed[mix]) for mix in MIXES)
    variance_sum = sum(terms)
    denominator = sum(term * term / (len(parsed[mix]) - 1) for term, mix in zip(terms, MIXES))
    df = None if variance_sum == 0.0 else variance_sum * variance_sum / denominator
    se = math.sqrt(variance_sum)
    critical = 0.0 if variance_sum == 0.0 else student_t_isf(alpha, df)
    statistic = (
        math.inf
        if se == 0.0 and estimate > 0.0
        else (-math.inf if se == 0.0 and estimate < 0.0 else 0.0 if se == 0.0 else estimate / se)
    )
    if statistic == math.inf:
        p_value = 0.0
    elif statistic == -math.inf:
        p_value = 1.0
    elif se == 0.0:
        p_value = 0.5
    else:
        p_value = student_t_sf(statistic, float(df))
    return {
        "estimate": estimate,
        "se": se,
        "variance": variance_sum,
        "df": df,
        "alpha_one_sided": alpha,
        "lower": estimate - critical * se,
        "upper": estimate + critical * se,
        "p_value": p_value,
        "n_by_mix": {mix: len(parsed[mix]) for mix in MIXES},
    }


def simultaneous_bonferroni_bounds(
    claims: Mapping[str, Mapping[str, Any]], *, family_alpha: float = FAMILY_ALPHA
) -> dict[str, dict[str, Any]]:
    """Return conservative simultaneous bounds for E16 admission claims.

    Each claim supplies ``groups`` (the three paired-world mix vectors) and a
    ``direction`` of ``lower`` (J) or ``upper`` (G/collateral).  It is a
    companion to Holm's J-superiority decision: the Bonferroni endpoint is a
    fixed simultaneous report, so it never depends on the observed rank.
    """
    if not claims:
        raise ValueError("at least one simultaneous claim is required")
    if not 0.0 < family_alpha < 0.5:
        raise ValueError("family alpha must be in (0, .5)")
    alpha = family_alpha / len(claims)
    output: dict[str, dict[str, Any]] = {}
    for name, claim in claims.items():
        direction = claim.get("direction")
        if direction not in {"lower", "upper"}:
            raise ValueError(f"{name} direction must be 'lower' or 'upper'")
        result = paired_equal_mix_bounds(claim["groups"], alpha=alpha)
        result.update(
            {
                "simultaneous_family_alpha_one_sided": family_alpha,
                "bonferroni_alpha_one_sided": alpha,
                "direction": direction,
                "simultaneous_bound": result["lower"] if direction == "lower" else result["upper"],
            }
        )
        output[name] = result
    return output


def holm_one_sided_bounds(
    comparisons: Mapping[str, Mapping[str, Sequence[float]]], *, family_alpha: float = FAMILY_ALPHA
) -> dict[str, dict[str, Any]]:
    """Apply the frozen one-sided Holm family over named arm/look comparisons."""
    if not comparisons:
        raise ValueError("at least one admission comparison is required")
    preliminary = {name: paired_equal_mix_bounds(groups) for name, groups in comparisons.items()}
    names = tuple(preliminary)
    rejected, adjusted = holm(
        [float(preliminary[name]["p_value"]) for name in names], alpha=family_alpha
    )
    # The reported lower endpoint uses the comparison's Holm step alpha.  The
    # rejection flag, calculated from the same frozen family, is authoritative.
    ordered = sorted(names, key=lambda name: (float(preliminary[name]["p_value"]), name))
    step_alpha = {name: family_alpha / (len(names) - rank) for rank, name in enumerate(ordered)}
    output: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        result = paired_equal_mix_bounds(comparisons[name], alpha=step_alpha[name])
        result.update(
            {
                "family_alpha_one_sided": family_alpha,
                "holm_step_alpha": step_alpha[name],
                "holm_adjusted_p_value": adjusted[index],
                "holm_rejects_nonpositive": rejected[index],
            }
        )
        output[name] = result
    return output


def survival_band_v2(
    candidate_alive: Mapping[str, Sequence[float]],
    b8_alive: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fixed-look survival guard A: pooled -.05, per-mix -.075, floor .30."""
    differences = {
        mix: [
            _number(c, "candidate alive") - _number(b, "b8 alive")
            for c, b in zip(candidate_alive[mix], b8_alive[mix])
        ]
        for mix in MIXES
    }
    if any(
        len(differences[mix]) != len(candidate_alive[mix])
        or len(differences[mix]) != len(b8_alive[mix])
        for mix in MIXES
    ):
        raise ValueError("candidate and B8 survival rows must be paired within every mix")
    pooled = paired_equal_mix_bounds(differences, alpha=alpha)
    per_mix: dict[str, Any] = {}
    floors: dict[str, float] = {}
    for mix in MIXES:
        values = differences[mix]
        if len(values) < 2:
            raise ValueError("survival band requires at least two worlds per mix")
        se = math.sqrt(variance(values) / len(values))
        bound = mean(values) - student_t_isf(alpha, len(values) - 1) * se
        per_mix[mix] = {"estimate": mean(values), "lower": bound, "n": len(values)}
        floors[mix] = mean(_number(value, "candidate alive") for value in candidate_alive[mix])
    passed = (
        pooled["lower"] > -0.05
        and all(per_mix[mix]["lower"] > -0.075 for mix in MIXES)
        and all(value >= 0.30 for value in floors.values())
    )
    return {"pass": passed, "pooled": pooled, "by_mix": per_mix, "candidate_alive_fraction": floors}


def decide_arm(
    *,
    available: bool,
    valid: bool,
    tripwires: Sequence[str],
    j: Mapping[str, Any] | None,
    g: Mapping[str, Any] | None,
    ledger: Mapping[str, Any] | None,
    collateral: Mapping[str, Any] | None,
    survival: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the frozen E16 precedence without manufacturing a rank for unavailable arms."""
    if not available:
        return {"status": "UNRANKED", "reasons": ["arm_precondition_unmet"]}
    if not valid:
        return {"status": "INCOMPLETE", "reasons": ["integrity_or_cap_or_critic_failure"]}
    if tripwires:
        return {"status": "KILL", "reasons": ["tripwire", *tripwires]}
    if j is None or g is None or ledger is None or collateral is None or survival is None:
        return {"status": "INCOMPLETE", "reasons": ["required_endpoint_evidence_missing"]}
    common_alive = float(ledger["equal_mix"]["common_alive"])
    if survival.get("pass") is not True:
        return {"status": "KILL", "reasons": ["survival_band_v2_failed"]}
    if common_alive < -10.0:
        return {"status": "KILL", "reasons": ["common_alive_loss_below_minus_10"]}
    per_mix = ledger
    productive = all(float(per_mix[mix]["productive_rescues_per_100"]) >= 5.0 for mix in MIXES)
    retention = all(
        values and mean(values) >= 0.90
        for values in (per_mix[mix]["rescue_tail_retention"] for mix in MIXES)
    )
    admitted = (
        j.get("holm_rejects_nonpositive") is True
        and float(j["lower"]) > 0.0
        and float(g["estimate"]) >= 10.0
        and productive
        and retention
        and float(collateral["upper"]) <= 10.0
    )
    artifact_status = (
        "ARTIFACT_REJECTED_AT_BAR" if float(g["upper"]) < 40.0 else "NOT_REJECTED_AT_BAR"
    )
    if admitted:
        return {"status": "ADMIT_100M", "artifact_status": artifact_status, "reasons": []}
    return {"status": "INCONCLUSIVE", "artifact_status": artifact_status, "reasons": []}
