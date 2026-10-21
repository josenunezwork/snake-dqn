"""Prospective Z-4 planning only; reuse E9's measured marginal simulation model.

No production rule, bank, manifest or admission consumer imports this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from research.bl_ppo import oc
from research.bl_ppo.exports import export_json, verify_export

DESIGNS = (
    {"id": "a_fixed_192", "counts": dict(zip(oc.MIXES, (192, 64, 64))), "e6_funds": True},
    {"id": "b_uncoupled_128", "counts": dict(zip(oc.MIXES, (128, 48, 48))), "e6_funds": False},
    {
        "id": "c_sequential_96_plus_96",
        "counts": dict(zip(oc.MIXES, (192, 64, 64))),
        "stage_counts": dict(zip(oc.MIXES, (96, 32, 32))),
        "e6_funds": True,
    },
)


def pool_stages(
    mean1: np.ndarray, var1: np.ndarray, mean2: np.ndarray, var2: np.ndarray, n: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact pooled sufficient statistics for two independent equal-sized stages."""
    mean = (mean1 + mean2) / 2
    variance = ((n - 1) * (var1 + var2) + n * (mean1 - mean2) ** 2 / 2) / (2 * n - 1)
    return mean, variance


def inference(
    means: np.ndarray, sampled: np.ndarray, n: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    terms = sampled / n / 9
    variance = terms.sum(axis=1)
    df = variance**2 / (terms**2 / (n - 1)).sum(axis=1)
    return means.mean(axis=1), np.sqrt(variance), df


def continuation_mask(
    means: np.ndarray, se: np.ndarray, df: np.ndarray, effect: float
) -> np.ndarray:
    """Only futility at look 1: equality at UB90=40 continues; never fund here."""
    return means + effect + t.isf(0.1, df) * se >= 40


def survival_guards(
    means: np.ndarray,
    sampled: np.ndarray,
    n: np.ndarray,
    episode: np.ndarray,
    b8_alive: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Same marginal band-v2 and scripted point gates as e9-oc-04."""
    alive = means[:, :, :2] + delta
    pooled, se, df = inference(alive, sampled[:, :, :2], n)
    return (
        (pooled - t.isf(0.05, df) * se > -0.05)
        & (alive - t.isf(0.05, n - 1) * np.sqrt(sampled[:, :, :2] / n) > -0.075).all(axis=1)
        & (b8_alive[None, :, None] + alive >= 0.30).all(axis=1)
        & (episode[:, 0, :2] + delta >= 0)
    )


def simulate(
    inputs: dict[str, Any], design: dict[str, Any], *, reps: int, seed: int
) -> list[dict[str, Any]]:
    if reps < 1000:
        raise ValueError("at least 1000 replicates required")
    rng = np.random.default_rng(seed)
    counts = design["counts"]
    nflat = np.array([counts[m] for m in oc.MIXES], dtype=float)
    n = nflat[None, :, None]
    variances = {
        metric: np.array([inputs["mixes"][m][metric] for m in oc.MIXES])
        for metric in ("paired_mass_variance", "paired_alive_variance", "paired_episode_variance")
    }
    b8 = np.array([inputs["mixes"][m]["b8_alive"] for m in oc.MIXES])
    rows = []
    for rho in (0.0, 0.5, 0.9):
        shared = rng.standard_normal((reps, 3, 1))
        metrics = {}
        interim = None
        for metric, variance in variances.items():
            if "stage_counts" in design:
                half = nflat / 2
                # Independent worlds at the two stages. The E6 shared-reference
                # component correlates with the final, not twice with stage 1.
                shared2 = (
                    rng.standard_normal((reps, 3, 1)) if metric == "paired_mass_variance" else None
                )
                a, av, _ = oc._metric(
                    rng, variance, half, reps, rho, shared if shared2 is not None else None
                )
                b, bv, _ = oc._metric(rng, variance, half, reps, rho, shared2)
                metrics[metric] = pool_stages(a, av, b, bv, n / 2)
                if shared2 is not None:
                    interim = inference(a[:, :, :2], av[:, :, :2], n / 2)
                    final_shared = (shared + shared2) / np.sqrt(2)
            else:
                a, av, _ = oc._metric(
                    rng,
                    variance,
                    nflat,
                    reps,
                    rho,
                    shared if metric == "paired_mass_variance" else None,
                )
                metrics[metric] = a, av
                final_shared = shared
        mass, se, df = inference(*[x[:, :, :2] for x in metrics["paired_mass_variance"]], n)
        en = np.array([44.0, 42.0, 42.0])
        em, es, _ = oc._metric(rng, variances["paired_mass_variance"], en, reps, rho, final_shared)
        early, ese, edf = inference(em[:, :, 2:], es[:, :, 2:], en[None, :, None])
        om, _, _ = oc._metric(rng, variances["paired_mass_variance"], en, reps, rho)
        ordinary = om.mean(axis=1)[:, 2:] >= -2
        for delta in (-0.05, 0.0, 0.03, 0.06):
            guards = survival_guards(
                *metrics["paired_alive_variance"],
                n,
                metrics["paired_episode_variance"][0],
                b8,
                delta,
            )
            for effect in (0.0, 20.0, 40.0, 60.0, 80.0):
                means = mass + effect
                active = np.ones((reps, 2), dtype=bool)
                if interim is not None:
                    im, ise, idf = interim
                    # Stop an arm if its interim UB90 is below the +40 target.
                    # No early efficacy, and stopped arms retain their Holm slot.
                    active = continuation_mask(im, ise, idf, effect)
                p = np.where(active, t.sf(means / se, df), 1.0)
                go = oc._holm(p) & (means >= 10) & (means - t.isf(0.1, df) * se > 0) & guards
                kill = active & (means + t.isf(0.1, df) * se <= 0)
                early_mean = early + effect
                e6 = (
                    oc._holm(t.sf(early_mean / ese, edf))
                    & (early_mean - t.isf(0.1, edf) * ese > 0)
                    & ordinary
                )
                e2_funds, e6_reported = go.any(axis=1), e6.any(axis=1)
                e6_funds = e6_reported & design["e6_funds"]
                family = e2_funds | e6_funds
                # Both endpoints are measured together at each stage; if either
                # continues, all three controllers run all stage-2 worlds.
                continue_stage2 = active.any(axis=1)
                fraction = (0.5 + 0.5 * continue_stage2.mean()) if interim is not None else 1.0
                rows.append(
                    {
                        "rho_shared_B8": rho,
                        "true_mass_and_early_dividend": effect,
                        "true_survival_delta": delta,
                        "family_funding_probability": float(family.mean()),
                        "MC95_interval": oc.wilson_interval(int(family.sum()), reps),
                        "E2_any_GO": float(e2_funds.mean()),
                        "E6_any_NEEDS_SCALE_reported": float(e6_reported.mean()),
                        "E6_is_funding_route": design["e6_funds"],
                        "E2_joint": oc.joint_outcomes(go, kill),
                        "funding_routes_joint": {
                            "E2_and_E6": float((e2_funds & e6_funds).mean()),
                            "E2_only": float((e2_funds & ~e6_funds).mean()),
                            "E6_only": float((~e2_funds & e6_funds).mean()),
                            "neither": float((~family).mean()),
                        },
                        "probability_stage2": (
                            float(continue_stage2.mean()) if interim is not None else None
                        ),
                        "expected_screen_worlds": float(sum(counts.values()) * fraction),
                        "expected_screen_controller_episodes": float(
                            3 * sum(counts.values()) * fraction
                        ),
                        "interim_futility_arm_probability": list(
                            (1 - active.mean(axis=0)).astype(float)
                        ),
                        "futility_in_E2_joint": "INCONCLUSIVE; no funding; not scientific KILL",
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    null = [r for r in rows if r["true_mass_and_early_dividend"] == 0]
    power = [
        r for r in rows if r["true_mass_and_early_dividend"] == 40 and r["true_survival_delta"] >= 0
    ]
    return {
        "worst_family_null_pass": max(r["family_funding_probability"] for r in null),
        "null_MC95_upper": max(r["MC95_interval"][1] for r in null),
        "minimum_joint_funding_at_40_nonnegative_survival": min(
            r["family_funding_probability"] for r in power
        ),
        "power_MC95_lower": min(r["MC95_interval"][0] for r in power),
        "passes_probability_bars": max(r["MC95_interval"][1] for r in null) <= 0.10
        and min(r["MC95_interval"][0] for r in power) >= 0.60,
        "expected_controller_episodes_at_40_range": [
            min(r["expected_screen_controller_episodes"] for r in power),
            max(r["expected_screen_controller_episodes"] for r in power),
        ],
    }


def screening_invoice(episodes: float, rate: float, *, e6: bool) -> dict[str, float]:
    """Conditional L4 invoice: aggregate completed controller episodes per second.

    One 900-second setup allowance, 2x estimate liability, 20% reserve.
    E6's separate 4-unit cap includes its own reserve; training is excluded.
    """
    if episodes <= 0 or rate <= 0 or not np.isfinite(rate):
        raise ValueError("finite positive episode rate required")
    cap = (2 * (episodes / rate + 900) / 3600 * 2) / 0.8 + (4 if e6 else 0)
    return {"aggregate_episode_rate_assumed": rate, "screening_units_including_reserve_and_E6": cap}


def run(prior_path: Path, *, reps: int, seed: int) -> dict[str, Any]:
    prior_sha = verify_export(prior_path)
    prior = json.loads(prior_path.read_bytes())
    inputs = oc.measured_inputs(Path(prior["measured_inputs"]["source"]))
    if inputs != prior["measured_inputs"]:
        raise ValueError("T23 inputs differ from e9-oc-04")
    if (
        prior["source_sha256"]["oc.py"]
        != hashlib.sha256(Path(oc.__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("existing E9 simulator differs from e9-oc-04")
    candidates = []
    for design in DESIGNS:
        rows = simulate(inputs, design, reps=reps, seed=seed)
        episodes = 3 * sum(design["counts"].values())
        candidates.append(
            {
                "design": design,
                "summary": summarize(rows),
                "scenarios": rows,
                "budget": {
                    "measured_screen_episode_rate": None,
                    "status": "UNPRICED_NO_MEASURED_SCREEN_RATE",
                    "maximum_controller_episodes": episodes,
                    "screening_cap_proposed_units": 120,
                    "minimum_aggregate_episodes_per_second_for_120u": episodes
                    / (((120 - 4) * 0.8 / 2 / 2 * 3600) - 900),
                    "E6_cap_included_even_when_diagnostic": 4,
                    "rate_sensitivity_not_measurements": [
                        screening_invoice(episodes, rate, e6=True) for rate in (0.01, 0.1, 1.0)
                    ],
                },
            }
        )
    passing = [c for c in candidates if c["summary"]["passes_probability_bars"]]
    return {
        "schema": "bl-ppo-e9-replacement-oc/v1",
        "status": "PROPOSED_AWAITING_FUNDING_JUDGE",
        "platform": "Mac CPU; one math thread; no gameplay or Colab",
        "replicates": reps,
        "seed": seed,
        "prior_report": {"path": str(prior_path), "sha256": prior_sha},
        "measured_inputs": inputs,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("oc.py", "oc_replacement.py", "rule.py")
        },
        "candidates_in_priority_order": candidates,
        "selection": {
            "first_passing_candidate": passing[0]["design"]["id"] if passing else None,
            "best_simulated_candidate": max(
                candidates,
                key=lambda c: c["summary"]["minimum_joint_funding_at_40_nonnegative_survival"],
            )["design"]["id"],
            "dispatch_admitted": False,
            "E2_status": "BLOCKED",
            "reason": (
                "Proposed planning report only; no tested Z-4 design meets both OC bars."
                if not passing
                else "Funding judge, measured invoice and separate implementation required."
            ),
        },
        "limitations": prior["assumptions"]
        + [
            (
                "Both R and F true effects equal the scenario; joint funding means any eligible "
                "route, not both endpoints passing."
            ),
            (
                "No E6 funding route in candidate b; E6 results are diagnostic only and cannot "
                "select/fund an E2 endpoint."
            ),
            (
                "Global-null pass is simulated, not a strong-FWER proof; E2 Holm .05 and "
                "(a/c only) separate E6 Holm .05 bound efficacy family error by .10 under "
                "valid marginal p-values."
            ),
            (
                "Sequential option has no interim success; independent normal/chi-square stage "
                "summaries are pooled including between-stage sum of squares. Futility can "
                "only remove final rejections."
            ),
            (
                "120-unit feasibility is conditional on measured evaluation rate, setup/export "
                "costs and future liability invoice. Collection transitions/s are not "
                "evaluation episodes/s."
            ),
            (
                "Best means best among the three Z-4 candidates, not a global optimum over "
                "unregistered designs. No claim that every conceivable design below "
                "120 units is impossible."
            ),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    report = run(args.prior, reps=args.replicates, seed=args.seed)
    export_json(args.out, report)
    print(
        json.dumps(
            {
                "selection": report["selection"],
                "summaries": [
                    {"id": c["design"]["id"], **c["summary"]}
                    for c in report["candidates_in_priority_order"]
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
