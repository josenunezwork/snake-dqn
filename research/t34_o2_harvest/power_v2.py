"""Prospective, pre-data O2 V2 sizing from the measured T23 paired SDs.

The two O2 superiority contrasts share the D+O arm, but no covariance for
their paired-world deltas has been measured.  This module therefore does not
claim an independence-based joint power: it chooses a sufficient,
dependence-robust allocation by requiring 90% marginal power for each
contrast.  Bonferroni's inequality then guarantees at least 80% joint power.

This is a planning calculation only.  It consumes no O2 trajectory, bank, or
outcome artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402
from scipy.stats import nct, t  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
T23_SOURCE = ROOT / "research/t23_r0_screen_20261017/results/t23_verdict.json"
T23_SOURCE_SHA256 = "e97234956a3a7434153850d5e4fbb8a8018064728c454a68cee36313e615e37b"
MIXES = ("scripted", "frozen", "mixed")
FAMILY_ALPHA = 0.05
MARGINAL_ALPHA = FAMILY_ALPHA / 2
JOINT_TARGET = 0.80
MARGINAL_TARGET = 0.90
EFFECT_B8 = 40.0
EFFECT_D = 10.0


def _source_check() -> None:
    """Refuse a calculation when the pinned T23 planning source drifted."""
    if hashlib.sha256(T23_SOURCE.read_bytes()).hexdigest() != T23_SOURCE_SHA256:
        raise ValueError("pinned T23 verdict SHA-256 differs")


def measured_sds() -> dict[str, float]:
    """Extract exact marginal paired SDs from the pinned T23 result object."""
    _source_check()
    mixes = json.loads(T23_SOURCE.read_text())["steps"]["8000000"]["mixes"]
    return {mix: math.sqrt(float(mixes[mix]["paired_mass_variance"])) for mix in MIXES}


def _statistics(n_per_mix: int, sds: dict[str, float] | None = None) -> tuple[float, float]:
    """Return equal-mix Welch SE and planning df for balanced strata."""
    if n_per_mix < 2:
        raise ValueError("at least two worlds per mix are required")
    sds = sds or measured_sds()
    components = [sds[mix] ** 2 / n_per_mix for mix in MIXES]
    variance = sum(components) / 9.0
    denominator = sum(component * component / (n_per_mix - 1) for component in components) / 81.0
    return math.sqrt(variance), variance * variance / denominator


def marginal_power(
    n_per_mix: int,
    effect: float,
    alpha: float = MARGINAL_ALPHA,
    sds: dict[str, float] | None = None,
) -> float:
    """One-sided Welch-t rejection probability under the normal planning model."""
    se, df = _statistics(n_per_mix, sds)
    critical = float(t.isf(alpha, df))
    return float(nct.sf(critical, df, effect / se))


def _joint_lower_bound(n_per_mix: int, sds: dict[str, float] | None = None) -> float:
    """Sharp arbitrary-dependence lower bound for two-test Holm rejection.

    Holm succeeds if the p-value regions are low/mid or mid/low (low < .025,
    mid < .05).  The displayed maximum is the corresponding Fr\u00e9chet bound;
    it is sharper than requiring both tests to clear .025.
    """
    b8_low = marginal_power(n_per_mix, EFFECT_B8, 0.025, sds)
    b8_mid = marginal_power(n_per_mix, EFFECT_B8, 0.05, sds)
    d_low = marginal_power(n_per_mix, EFFECT_D, 0.025, sds)
    d_mid = marginal_power(n_per_mix, EFFECT_D, 0.05, sds)
    return max(0.0, b8_low + d_mid - 1.0, b8_mid + d_low - 1.0)


def required_n() -> int:
    """Find the smallest balanced allocation satisfying the robust joint bound."""
    sds = measured_sds()
    n = 2
    while True:
        if _joint_lower_bound(n, sds) >= JOINT_TARGET:
            return n
        n += 1


def _holm_reject(p_b8: np.ndarray, p_d: np.ndarray) -> np.ndarray:
    """Two-hypothesis Holm at family alpha .05 with deterministic tie treatment."""
    lo, hi = np.minimum(p_b8, p_d), np.maximum(p_b8, p_d)
    return (lo < FAMILY_ALPHA / 2.0) & (hi < FAMILY_ALPHA)


def monte_carlo(n_per_mix: int, reps: int, seed: int, sds: dict[str, float]) -> dict[str, Any]:
    """Verify the analytic calculation and strong-null control in a Gaussian proxy.

    The simulation deliberately uses independent contrasts only as a diagnostic.
    The registered sizing claim is the Fr\u00e9chet/Bonferroni lower bound, which needs
    no unmeasured covariance.  Composite-null rows test the actual Holm logic.
    """
    rng = np.random.default_rng(seed)
    se, df = _statistics(n_per_mix, sds)
    variance_components = np.asarray([sds[m] ** 2 / n_per_mix for m in MIXES])
    result: dict[str, Any] = {"reps": reps, "seed": seed, "se": se, "df": df, "scenarios": {}}
    for name, (mu_b8, mu_d) in {
        "target_true40_increment10": (EFFECT_B8, EFFECT_D),
        "global_null": (0.0, 0.0),
        "b8_null_increment10": (0.0, EFFECT_D),
        "true40_increment_null": (EFFECT_B8, 0.0),
    }.items():
        # Independent contrast diagnostic: Welch sample variances have their
        # finite-n chi-square law.  Cross-contrast covariance is deliberately
        # absent here and is not used by the sizing guarantee.
        v = rng.chisquare(n_per_mix - 1, size=(reps, 2, 3)) / (n_per_mix - 1)
        estimates = (v * variance_components).sum(axis=2) / 9.0
        ses = np.sqrt(estimates)
        dfs = estimates**2 / (((v * variance_components) ** 2 / (n_per_mix - 1)).sum(axis=2) / 81.0)
        means = rng.normal(loc=(mu_b8, mu_d), scale=se, size=(reps, 2))
        p_b8 = t.sf(means[:, 0] / ses[:, 0], dfs[:, 0])
        p_d = t.sf(means[:, 1] / ses[:, 1], dfs[:, 1])
        passed = _holm_reject(p_b8, p_d)
        both_marginal = (p_b8 < 0.025) & (p_d < 0.025)
        if name == "target_true40_increment10":
            event = passed
            label = "holm_both_power_independent_proxy"
        elif name == "global_null":
            event = np.minimum(p_b8, p_d) < FAMILY_ALPHA / 2.0
            label = "family_any_false_rejection"
        elif name == "b8_null_increment10":
            event = passed
            label = "false_rejection_of_b8_null"
        else:
            event = passed
            label = "false_rejection_of_increment_null"
        rate = float(event.mean())
        half = 1.959963984540054 * math.sqrt(rate * (1.0 - rate) / reps)
        result["scenarios"][name] = {
            label: rate,
            "normal_approx_95_interval": [max(0.0, rate - half), min(1.0, rate + half)],
            "both_alpha025_marginal_event": float(both_marginal.mean()),
        }
    return result


def run(
    reps: int = 100_000, seed: int = 202610221, cap_worlds: int | None = None
) -> dict[str, Any]:
    """Build the immutable V2 planning object; no O2 study data are read."""
    if reps < 1:
        raise ValueError("positive replicate count required")
    _source_check()
    sds = measured_sds()
    n = required_n()
    se, df = _statistics(n, sds)
    p_b8_low, p_d_low = marginal_power(n, EFFECT_B8, 0.025, sds), marginal_power(
        n, EFFECT_D, 0.025, sds
    )
    p_b8_mid, p_d_mid = marginal_power(n, EFFECT_B8, 0.05, sds), marginal_power(
        n, EFFECT_D, 0.05, sds
    )
    prior_bound = _joint_lower_bound(n - 1, sds)
    result = {
        "schema": "t34-o2-power-v2/v1",
        "pre_data_only": True,
        "source": {
            "path": str(T23_SOURCE.relative_to(ROOT)),
            "sha256": T23_SOURCE_SHA256,
            "measured_paired_mass_sd": sds,
            "extraction": "exact square roots of T23 steps.8000000.mixes.*.paired_mass_variance",
        },
        "rule": {
            "two_one_sided_superiority_contrasts": ["D+O - B8", "D+O - D"],
            "family_method": "Holm step-down",
            "family_alpha": FAMILY_ALPHA,
            "sizing_sufficient_event": (
                "sharp Fr\u00e9chet lower bound for Holm rejection over arbitrary "
                "cross-contrast dependence"
            ),
            "true_effects": {"D+O - B8": EFFECT_B8, "D+O - D": EFFECT_D},
            "balanced_counts": True,
        },
        "dependence": {
            "known_cross_contrast_covariance": False,
            "assumption": "T23 marginal SD used for each contrast; no covariance invented",
            "joint_power_guarantee": "max(0, P_B8(.025)+P_D(.05)-1, P_B8(.05)+P_D(.025)-1) >= .80",
            "joint_target": JOINT_TARGET,
            "note": (
                "The independent Monte Carlo is diagnostic only; the allocation rests on "
                "the dependence-robust bound."
            ),
        },
        "required_allocation": {
            "worlds_per_mix": n,
            "worlds_total": 3 * n,
            "welch_equal_mix_se": se,
            "welch_satterthwaite_df": df,
            "marginal_power": {
                "D+O - B8": {"alpha025": p_b8_low, "alpha05": p_b8_mid},
                "D+O - D": {"alpha025": p_d_low, "alpha05": p_d_mid},
            },
            "dependence_robust_joint_lower_bound": _joint_lower_bound(n, sds),
            "minimality_previous_per_mix": n - 1,
            "minimality_previous_dependence_robust_joint_lower_bound": prior_bound,
            "exceeds_1000_worlds": 3 * n > 1000,
        },
        "point_bar_disclosure": {
            "existing_rule": "observed D+O - B8 mean >= 40",
            "at_true40": (
                "Under symmetric continuous estimation error this point bar has an asymptotic "
                "0.5 pass ceiling; no sample size establishes >=.80 full-GO power while "
                "retaining it."
            ),
            "prospective_correction_required_for_full_GO_claim": (
                "Replace the point bar in a sibling pre-data amendment with a "
                "superiority-inference "
                "rule (for example Holm-rejected D+O-B8 >0 plus a prospectively chosen effect "
                "threshold strictly below +40), or define the .80 target as superiority-only. "
                "Do not "
                "represent this sizing as full-GO power under the retained point bar."
            ),
        },
        "familywise_null": {
            "claim": (
                "Holm at alpha .05 strongly controls family-wise false rejection at <=.05 under "
                "valid marginal p-values, including composite nulls; no cross-contrast "
                "independence "
                "is needed."
            ),
            "composites_checked": [
                "global null",
                "B8 contrast null with +10 increment",
                "+40 B8 contrast with increment null",
            ],
        },
        "monte_carlo": monte_carlo(n, reps, seed, sds),
        "limitations": (
            "T23 is a different controller comparison; its marginal variance is a planning proxy. "
            "Incremental D+O-D variance and covariance with D+O-B8 are unobserved. Actual guards, "
            "fault rules, and point bars can only reduce complete-controller GO probability."
        ),
        "simulator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if cap_worlds is not None:
        if cap_worlds < 6 or cap_worlds % 3:
            raise ValueError("cap_worlds must be a balanced multiple of three >= 6")
        cap_n = cap_worlds // 3
        cap_b8_low, cap_d_low = marginal_power(cap_n, EFFECT_B8, 0.025, sds), marginal_power(
            cap_n, EFFECT_D, 0.025, sds
        )
        cap_b8_mid, cap_d_mid = marginal_power(cap_n, EFFECT_B8, 0.05, sds), marginal_power(
            cap_n, EFFECT_D, 0.05, sds
        )
        result["budget_capped_allocation"] = {
            "worlds_per_mix": cap_n,
            "worlds_total": cap_worlds,
            "status": "UNDERPOWERED_CAP",
            "marginal_power": {
                "D+O - B8": {"alpha025": cap_b8_low, "alpha05": cap_b8_mid},
                "D+O - D": {"alpha025": cap_d_low, "alpha05": cap_d_mid},
            },
            "dependence_robust_joint_lower_bound": _joint_lower_bound(cap_n, sds),
            "reason": (
                "The requested allocation needs 5097 worlds; this is the largest balanced "
                "allocation "
                "supplied by the separately computed <=150-unit packing model."
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=202610221)
    parser.add_argument("--cap-worlds", type=int)
    args = parser.parse_args()
    with args.out.open("x") as handle:
        json.dump(
            run(args.reps, args.seed, args.cap_worlds),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
