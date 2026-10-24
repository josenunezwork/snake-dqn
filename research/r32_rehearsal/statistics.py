"""Frozen R32-5 prospective statistical comparison utilities.

This module deliberately has no study allocation, dispatch, lifecycle, or artifact
admission code.  It makes prospective sensitivities reproducible and refuses to
recommend a future design without adequate external calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

MIXES = ("scripted", "frozen", "mixed")
PROTOCOL_PATH = Path(__file__).with_name("STATISTICS_PROTOCOL.json")


@dataclass(frozen=True)
class CostModel:
    """Whole-path time decomposition in seconds, kept separate from elapsed time."""

    setup: float
    screen_pair_by_mix: Mapping[str, float]
    confirmation: float
    watch: float
    strict: float
    serving: float
    queue_delay: float = 0.0
    handoff: float = 0.0

    def fixed_total(self, allocation: Mapping[str, int]) -> float:
        return (
            self.setup
            + sum(self.screen_pair_by_mix[m] * allocation[m] for m in MIXES)
            + self.confirmation
            + self.watch
            + self.strict
            + self.serving
            + self.queue_delay
            + self.handoff
        )


def load_protocol(path: Path | str = PROTOCOL_PATH) -> dict[str, Any]:
    """Load the byte-reviewable frozen protocol without mutating it."""
    return json.loads(Path(path).read_text())


def deterministic_scenario_units(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return stable namespace/allocation units for a remote simulation package."""
    namespace = protocol["compute"]["resume_namespace"]
    units = []
    scenarios = protocol["scenarios"]["critical"] + protocol["scenarios"]["sensitivity_only"]
    for index, scenario in enumerate(scenarios):
        unit_id = f"{namespace}/scenario/{index:03d}"
        units.append(
            {
                "unit_id": unit_id,
                "allocation_index": index,
                "scenario": scenario,
                "seed": int.from_bytes(hashlib.sha256(unit_id.encode()).digest()[:8], "big"),
            }
        )
    return units


def frozen_units(tiny: bool = False) -> list[dict[str, Any]]:
    """Expose the frozen, deterministic work units expected by the R32 harness.

    Tiny units preserve the full scenario list but use the protocol's 200-trial
    local preflight. Main units use the mandated 100,000 critical-scenario trials.
    """
    protocol = load_protocol()
    critical = set(protocol["scenarios"]["critical"])
    units = []
    for unit in deterministic_scenario_units(protocol):
        trials = (
            protocol["compute"]["tiny_preflight_trials"]
            if tiny
            else (
                protocol["compute"]["critical_trials_per_scenario"]
                if unit["scenario"] in critical
                else protocol["compute"]["sensitivity_trials_per_scenario"]
            )
        )
        units.append({**unit, "trials": trials})
    return units


def has_adequate_empirical_calibration(evidence: Mapping[str, Any]) -> bool:
    """Require joint pair variance, cost, and full-path/session information."""
    required = {
        "paired_delta_sd",
        "complete_pair_cost_seconds",
        "watch_session_cost_seconds",
        "whole_path_cost_seconds",
    }
    return bool(
        evidence.get("published", False)
        and evidence.get("external_development", False)
        and required <= set(evidence)
    )


def optimal_cost_allocation(
    paired_delta_sd: Mapping[str, float],
    complete_pair_cost_seconds: Mapping[str, float],
    total_pairs: int = 384,
    minimum_per_mix: int = 16,
) -> dict[str, int]:
    """Integer Neyman starting allocation; guard constraints must be checked separately."""
    if total_pairs < minimum_per_mix * len(MIXES):
        raise ValueError("total_pairs is below the frozen per-mix floor")
    weights = np.array(
        [paired_delta_sd[m] / math.sqrt(complete_pair_cost_seconds[m]) for m in MIXES], dtype=float
    )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("SDs and complete-pair costs must be finite positive values")
    free = total_pairs - minimum_per_mix * len(MIXES)
    raw = free * weights / weights.sum()
    counts = np.floor(raw).astype(int) + minimum_per_mix
    for i in np.argsort(-(raw - np.floor(raw)))[: total_pairs - int(counts.sum())]:
        counts[i] += 1
    return dict(zip(MIXES, map(int, counts)))


def equal_weight_variance(
    paired_delta_sd: Mapping[str, float], allocation: Mapping[str, int]
) -> float:
    """Frozen disjoint-bank variance for G = sum(mean_delta_m)/3."""
    if any(allocation[m] <= 0 or paired_delta_sd[m] <= 0 for m in MIXES):
        raise ValueError("all allocations and SDs must be positive")
    return sum((paired_delta_sd[m] ** 2) / allocation[m] for m in MIXES) / 9.0


def welch_satterthwaite_df(
    paired_delta_sd: Mapping[str, float], allocation: Mapping[str, int]
) -> float:
    """df for independent mix-mean variance components, with no invented covariance."""
    components = np.array(
        [(paired_delta_sd[m] ** 2) / allocation[m] / 9.0 for m in MIXES], dtype=float
    )
    denominators = np.array([allocation[m] - 1 for m in MIXES], dtype=float)
    if np.any(denominators <= 0):
        raise ValueError("every mix needs at least two pairs for finite-sample inference")
    return float(components.sum() ** 2 / np.sum(components**2 / denominators))


def favorable_bound(
    cost: CostModel, alternative: Mapping[str, int], reference: Mapping[str, int] | None = None
) -> dict[str, float]:
    """Compute the cheapest possible whole-path saving if every non-screen cost is unchanged."""
    reference = reference or {m: 128 for m in MIXES}
    ref = cost.fixed_total(reference)
    alt = cost.fixed_total(alternative)
    return {
        "reference_seconds": ref,
        "alternative_seconds": alt,
        "saving_fraction": (ref - alt) / ref if ref else 0.0,
    }


def minimum_cost_allocation_for_precision(
    paired_delta_sd: Mapping[str, float],
    complete_pair_cost_seconds: Mapping[str, float],
    *,
    target_variance: float,
    minimum_per_mix: int = 16,
    maximum_per_mix: int = 512,
) -> dict[str, int]:
    """Find the cheapest integer allocation meeting a frozen equal-weight precision target.

    The target should normally be the equal-128/mix variance. This is the relevant
    cost comparison; a fixed 384-pair reallocation cannot by itself save cost.
    """
    best: tuple[float, tuple[int, int, int]] | None = None
    for n0 in range(minimum_per_mix, maximum_per_mix + 1):
        for n1 in range(minimum_per_mix, maximum_per_mix + 1):
            residual = (
                9.0 * target_variance
                - paired_delta_sd[MIXES[0]] ** 2 / n0
                - paired_delta_sd[MIXES[1]] ** 2 / n1
            )
            if residual <= 0:
                continue
            n2 = max(minimum_per_mix, math.ceil(paired_delta_sd[MIXES[2]] ** 2 / residual))
            if n2 > maximum_per_mix:
                continue
            counts = (n0, n1, n2)
            cost = sum(complete_pair_cost_seconds[m] * n for m, n in zip(MIXES, counts))
            if equal_weight_variance(
                paired_delta_sd, dict(zip(MIXES, counts))
            ) <= target_variance and (best is None or cost < best[0]):
                best = (cost, counts)
    if best is None:
        raise ValueError("no allocation meets target variance inside the frozen search bounds")
    return dict(zip(MIXES, best[1]))


def _mass_pass(final_by_mix: np.ndarray, alpha_one_sided: float = 0.10) -> bool:
    """Mass endpoint using independent mix means and Welch-Satterthwaite df."""
    allocation = dict(zip(MIXES, [final_by_mix.shape[1]] * len(MIXES)))
    sd = dict(zip(MIXES, final_by_mix.std(axis=1, ddof=1)))
    pooled = final_by_mix.mean(axis=1).mean()
    se = math.sqrt(equal_weight_variance(sd, allocation))
    lower = (
        pooled
        - student_t.ppf(1.0 - alpha_one_sided, df=welch_satterthwaite_df(sd, allocation)) * se
    )
    return bool(pooled >= 40.0 and lower > 0.0)


def futility_conditional_probability(
    observed_64_by_mix: np.ndarray,
    paired_delta_sd: Mapping[str, float],
    *,
    trials: int,
    seed: int,
    remaining_profile: Mapping[str, float] | None = None,
    heavy_tail_df: int = 5,
) -> dict[str, float | int]:
    """Exact analytic Gaussian predictive calculation for the sole 64/128 look.

    `observed_64_by_mix` has shape (3, 64), with rows in MIXES order. It uses the
    protocol's Gaussian conditional model; `trials` and `seed` remain accepted for
    harness compatibility but do not add nested Monte Carlo. This function neither
    makes an efficacy claim nor handles guards.
    """
    observed = np.asarray(observed_64_by_mix, dtype=float)
    if observed.shape != (3, 64) or not np.all(np.isfinite(observed)):
        raise ValueError("observed_64_by_mix must be finite with shape (3, 64)")
    if trials <= 0 or heavy_tail_df <= 2:
        raise ValueError("trials must be positive and heavy_tail_df must exceed 2")
    profile = remaining_profile or {m: 40.0 for m in MIXES}
    sds = np.array([paired_delta_sd[m] for m in MIXES])
    if not np.all(np.isfinite(sds)) or np.any(sds <= 0):
        raise ValueError("paired_delta_sd must contain positive finite values for every mix")
    observed_mean = observed.mean(axis=1).mean()
    remaining_mean = np.mean([profile[m] for m in MIXES])
    final_mean = 0.5 * (observed_mean + remaining_mean)
    final_allocation = {m: 128 for m in MIXES}
    frozen_sd = dict(zip(MIXES, sds))
    final_se = math.sqrt(equal_weight_variance(frozen_sd, final_allocation))
    threshold = max(
        40.0, student_t.ppf(0.90, welch_satterthwaite_df(frozen_sd, final_allocation)) * final_se
    )
    remaining_se = math.sqrt(sum((sds**2) / 64.0) / 36.0)
    probability = float(student_t.sf((threshold - final_mean) / remaining_se, df=10**9))
    return {
        "conditional_probability": probability,
        "analytic_model": "gaussian",
        "threshold": threshold,
        "conditional_mean": final_mean,
        "conditional_se": remaining_se,
        "trials": trials,
        "futility": probability < 0.05,
    }


def _scenario_profile(scenario: str) -> dict[str, float]:
    """Frozen effect profiles; heterogeneous and partial-null cases stay explicit."""
    profiles = {
        "effect_minus20_homogeneous": {m: -20.0 for m in MIXES},
        "effect_0_homogeneous": {m: 0.0 for m in MIXES},
        "effect_20_homogeneous": {m: 20.0 for m in MIXES},
        "effect_40_homogeneous": {m: 40.0 for m in MIXES},
        "effect_60_homogeneous": {m: 60.0 for m in MIXES},
        "effect_80_homogeneous": {m: 80.0 for m in MIXES},
        "effect_40_heterogeneous_mix": {"scripted": 80.0, "frozen": 40.0, "mixed": 0.0},
        "effect_40_guard_boundary": {m: 40.0 for m in MIXES},
        "effect_0_partial_null": {"scripted": 40.0, "frozen": 0.0, "mixed": -40.0},
        "effect_40_correlated_mass_survival_watch": {m: 40.0 for m in MIXES},
        "effect_40_heavy_tail": {m: 40.0 for m in MIXES},
    }
    if scenario not in profiles:
        raise ValueError(f"unknown frozen critical scenario: {scenario}")
    return profiles[scenario]


def _wilson(successes: int, trials: int, z: float = 1.96) -> dict[str, float]:
    """Wilson uncertainty stays non-degenerate when a simulated rate is zero or one."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2)) / denominator
    return {
        "rate": p,
        "wilson95_low": max(0.0, center - radius),
        "wilson95_high": min(1.0, center + radius),
    }


def _mass_pass_from_means(
    means: np.ndarray, frozen_sd: Mapping[str, float], n_per_mix: int = 128
) -> np.ndarray:
    """Vectorized frozen final mass endpoint for disjoint, equally weighted mixes."""
    allocation = {m: n_per_mix for m in MIXES}
    se = math.sqrt(equal_weight_variance(frozen_sd, allocation))
    df = welch_satterthwaite_df(frozen_sd, allocation)
    pooled = means.mean(axis=1)
    lower = pooled - student_t.ppf(0.90, df) * se
    return (pooled >= 40.0) & (lower > 0.0)


def run_frozen_unit(scenario: str, trials: int) -> dict[str, Any]:
    """Run one deterministic sensitivity unit without accessing study outcomes.

    The synthetic 64-world interim block is generated from the frozen, explicitly
    uncalibrated sensitivity model.  This adapter is for the package harness only;
    it reports a sensitivity calculation and always retains the fixed design.
    """
    protocol = load_protocol()
    unit = next(
        (item for item in deterministic_scenario_units(protocol) if item["scenario"] == scenario),
        None,
    )
    if unit is None:
        raise ValueError(f"scenario is not a frozen critical unit: {scenario}")
    model = protocol["synthetic_sensitivity_model"]
    profile = _scenario_profile(scenario)
    rng = np.random.default_rng(unit["seed"])
    df = int(model["heavy_tail_df"])
    sd = model["paired_delta_sd_by_mix"]
    scales = np.array([sd[m] * math.sqrt((df - 2) / df) for m in MIXES])
    fixed_mass = sequential_mass = guard_a = guard_b = watch = futility_count = 0
    watch_mass = watch_death = 0
    fixed_contract = sequential_contract = 0
    # Batches bound RAM for 100k x 3 x 128 heavy-tail sensitivities.
    for start in range(0, trials, 2000):
        count = min(2000, trials - start)
        heavy_tail = scenario == "effect_40_heavy_tail"
        draw = rng.standard_t if heavy_tail else rng.standard_normal
        draw_scale = scales if heavy_tail else np.array([sd[m] for m in MIXES])
        shared_mass = (
            rng.standard_normal((count, 1, 1)) * 24.0
            if scenario == "effect_40_correlated_mass_survival_watch"
            else 0.0
        )
        observed = draw(size=(count, 3, 64)) * draw_scale[None, :, None] + shared_mass
        observed += np.array([profile[m] for m in MIXES])[None, :, None]
        remaining = draw(size=(count, 3, 64)) * draw_scale[None, :, None] + shared_mass
        remaining += np.array([profile[m] for m in MIXES])[None, :, None]
        # The fixed and sequential designs share these exact complete paired blocks.
        # Therefore sequential final success is pathwise a subset of fixed success.
        full = np.concatenate([observed, remaining], axis=2)
        fixed_ok = _mass_pass_from_means(full.mean(axis=2), sd)
        observed_mean = observed.mean(axis=2).mean(axis=1)
        conditional_mean = 0.5 * (observed_mean + 40.0)
        final_se = math.sqrt(equal_weight_variance(sd, {m: 128 for m in MIXES}))
        threshold = max(
            40.0,
            student_t.ppf(0.90, welch_satterthwaite_df(sd, {m: 128 for m in MIXES})) * final_se,
        )
        remaining_se = math.sqrt(sum((scales * math.sqrt(df / (df - 2))) ** 2 / 64.0) / 36.0)
        conditional_probability = norm.sf((threshold - conditional_mean) / remaining_se)
        futile = conditional_probability < 0.05
        final_means = full.mean(axis=2)
        sequential_ok = _mass_pass_from_means(final_means, sd) & ~futile
        # Frozen correlated synthetic guard sensitivity: a shared latent makes all
        # component outcomes positively correlated, but remains explicitly uncalibrated.
        latent = (
            shared_mass[:, 0, 0][:, None] / 24.0
            if isinstance(shared_mass, np.ndarray)
            else rng.standard_normal((count, 1))
        )
        effect = np.array([profile[m] for m in MIXES])[None, :]
        boundary = -0.025 if scenario == "effect_40_guard_boundary" else 0.0
        survival_mix = -0.075 + effect / 800.0 + boundary + 0.025 * latent
        survival_absolute = 0.30 + effect / 800.0 + boundary + 0.025 * latent
        guard_a_ok = np.all(survival_mix >= -0.075, axis=1) & np.all(
            survival_absolute >= 0.30, axis=1
        )
        guard_b_ok = effect[:, 0] / 800.0 + boundary + 0.02 * latent[:, 0] >= 0.03
        watch_mass_ok = (
            -0.05 + effect.mean(axis=1) / 800.0 + boundary + 0.025 * latent[:, 0] >= -0.05
        )
        watch_death_ok = effect.mean(axis=1) / 1000.0 + boundary + 0.02 * latent[:, 0] >= 0.0
        watch_ok = watch_mass_ok & watch_death_ok
        fixed_mass += int(fixed_ok.sum())
        sequential_mass += int(sequential_ok.sum())
        futility_count += int(futile.sum())
        guard_a += int(guard_a_ok.sum())
        guard_b += int(guard_b_ok.sum())
        watch += int(watch_ok.sum())
        watch_mass += int(watch_mass_ok.sum())
        watch_death += int(watch_death_ok.sum())
        fixed_contract += int((fixed_ok & guard_a_ok & guard_b_ok & watch_ok).sum())
        sequential_contract += int((sequential_ok & guard_a_ok & guard_b_ok & watch_ok).sum())
    futility = _wilson(futility_count, trials)
    expected_screen_pairs = 192.0 + (1.0 - futility["rate"]) * 192.0
    decision = conservative_decision(protocol, {"published": False, "external_development": False})
    return {
        "schema": "r32-r5-statistics-unit-result/v1",
        "unit": {**unit, "trials": trials},
        "synthetic_sensitivity": True,
        "calibration_status": model["status"],
        "observed_profile": profile,
        "futility": futility,
        "component_failure_rates": {
            "fixed_mass": _wilson(trials - fixed_mass, trials),
            "sequential_mass": _wilson(trials - sequential_mass, trials),
            "survival_guard_a": _wilson(trials - guard_a, trials),
            "survival_guard_b": _wilson(trials - guard_b, trials),
            "watch_mass_product": _wilson(trials - watch_mass, trials),
            "watch_death_product": _wilson(trials - watch_death, trials),
        },
        "complete_contract_power": {
            "fixed": _wilson(fixed_contract, trials),
            "futility_64_128": _wilson(sequential_contract, trials),
            "paired_power_loss_fixed_minus_futility": _wilson(
                fixed_contract - sequential_contract, trials
            ),
        },
        "cost_comparison": {
            "equal_fixed": {
                "allocation": {m: 128 for m in MIXES},
                "mac_execution_seconds": None,
                "elapsed_seconds": None,
            },
            "cost_aware_fixed": {
                "status": (
                    "INELIGIBLE_RETAIN_EQUAL: no eligible external paired-delta/cost evidence"
                ),
                "mac_execution_seconds": None,
                "elapsed_seconds": None,
            },
            "futility_64_128": {
                "expected_screen_pairs": expected_screen_pairs,
                "screen_pair_saving_fraction": (384.0 - expected_screen_pairs) / 384.0,
                "mac_execution_seconds": None,
                "elapsed_seconds": None,
            },
            "whole_path_status": (
                "UNKNOWN: no eligible complete-pair/session/strict/serving/queue/handoff cost "
                "evidence; queue and handoff are elapsed-only, never Mac execution."
            ),
        },
        "interpretation": (
            "Synthetic sensitivity only; component rates, power and savings are not empirically "
            "calibrated and cannot support adoption."
        ),
        "decision": decision,
    }


def conservative_decision(
    protocol: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    favorable_saving_fraction: float | None = None,
    power_loss_40: float | None = None,
    power_loss_60: float | None = None,
    thresholds_resolved: bool = False,
) -> dict[str, str]:
    """Apply the ruling's conservative selection guard; sensitivities can never adopt."""
    if not has_adequate_empirical_calibration(evidence):
        return {
            "verdict": "RETAIN_FIXED_DESIGN",
            "reason": (
                "Missing adequate eligible joint paired-delta/cost calibration; sensitivity is not "
                "empirical calibration."
            ),
        }
    if (
        favorable_saving_fraction is None
        or favorable_saving_fraction
        < protocol["decision"]["minimum_expected_whole_path_saving_at_effect_40"]
    ):
        return {
            "verdict": "RETAIN_FIXED_DESIGN",
            "reason": "Favorable whole-path saving does not reach 20%.",
        }
    limit = protocol["decision"]["maximum_complete_contract_power_loss_at_effect_40_and_60"]
    if (
        power_loss_40 is None
        or power_loss_60 is None
        or max(power_loss_40, power_loss_60) > limit
        or not thresholds_resolved
    ):
        return {
            "verdict": "RETAIN_FIXED_DESIGN",
            "reason": "Power-loss threshold or Monte Carlo uncertainty remains unresolved.",
        }
    return {
        "verdict": "RETAIN_FIXED_DESIGN",
        "reason": (
            "A later Astra-judged pre-data registration is required; this module never adopts a "
            "future scientific design."
        ),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument(
        "--print-units", action="store_true", help="print deterministic remote simulation unit IDs"
    )
    parser.add_argument("--tiny", action="store_true", help="print 200-trial preflight units")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.print_units:
        print(json.dumps(frozen_units(tiny=args.tiny), indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "schema": protocol["schema"],
                    "status": protocol["status"],
                    "default_decision": protocol["decision"]["conservative_default"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    _main()
