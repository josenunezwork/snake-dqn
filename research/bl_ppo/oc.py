"""Mac-only, no-gameplay E9 measured-variance operating characteristics.

A planning simulation is not efficacy evidence. T23 supplies measured marginal
variances, not E2/E6 covariance or a predictor of learning. S1 is mandatory for
an admitted final power claim; an absent source is explicitly fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, t

from research.bl_ppo.exports import export_json
from research.bl_ppo.rule import RULE

MIXES = ("scripted", "frozen", "mixed")
COUNTS = ({"scripted": 96, "frozen": 32, "mixed": 32}, {"scripted": 128, "frozen": 48, "mixed": 48})


def measured_inputs(path: Path) -> dict[str, Any]:
    """Read only an explicit approved T23 summary; bind exact source bytes."""
    payload = path.read_bytes()
    data = json.loads(payload)["steps"]["8000000"]["mixes"]
    result = {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "study": "T23 8M fixed endpoints, historical variance only",
        "mixes": {},
    }
    for mix in MIXES:
        row = data[mix]
        result["mixes"][mix] = {
            key: float(row[key])
            for key in (
                "paired_mass_variance",
                "paired_alive_variance",
                "paired_episode_variance",
                "b8_alive",
                "b8_no_death",
                "count",
            )
        }
        if any(not np.isfinite(v) or v < 0 for v in result["mixes"][mix].values()):
            raise ValueError("invalid measured variance input")
    return result


def stratified_variance(variances: dict[str, float], counts: dict[str, int]) -> float:
    """Equal-mix mean variance, including unequal allocation sizes."""
    return sum(variances[m] / counts[m] for m in MIXES) / 9


def _holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Vectorized exact two-slot Holm; last dimension is R,F."""
    order = np.argsort(p, axis=-1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=-1)
    first = ranked[..., 0] <= alpha / 2
    reject = np.stack((first, first & (ranked[..., 1] <= alpha)), axis=-1)
    return np.take_along_axis(reject, np.argsort(order, axis=-1), axis=-1)


def _normals(rng: np.random.Generator, shape: tuple[int, ...], rho: float) -> np.ndarray:
    # A shared -B8 component creates the stated cross-candidate/route correlation.
    return np.sqrt(rho) * rng.standard_normal((*shape[:-1], 1)) + np.sqrt(
        1 - rho
    ) * rng.standard_normal(shape)


def _metric(
    rng: np.random.Generator,
    variances: np.ndarray,
    n: np.ndarray,
    reps: int,
    rho: float,
    shared: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal means plus chi-square variance estimates (Welch marginal model)."""
    shared = rng.standard_normal((reps, 3, 1)) if shared is None else shared
    noise = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal((reps, 3, 4))
    means = noise * np.sqrt(variances / n)[None, :, None]
    sampled = (
        variances[None, :, None]
        * rng.chisquare(n[None, :, None] - 1, size=(reps, 3, 4))
        / (n[None, :, None] - 1)
    )
    terms = sampled / n[None, :, None] / 9
    pooled_var = terms.sum(axis=1)
    df = pooled_var**2 / (terms**2 / (n[None, :, None] - 1)).sum(axis=1)
    return means, sampled, df


def simulate(
    inputs: dict[str, Any], counts: dict[str, int], *, reps: int = 20000, seed: int = 20261021
) -> list[dict[str, Any]]:
    """Full E2/E6 route pass probabilities under registered guard scenarios.

    E6 uses 44/42/42 early source worlds and the same-sized ordinary stratum.
    Its unknown SD is a *proxy sensitivity* from T23, not measured E6 precision.
    Operational completeness and critic qualification are assumed true and are
    printed as assumptions; dispatch admission independently requires them.
    """
    if reps < 1000:
        raise ValueError("at least 1000 Monte Carlo replicates required")
    rng = np.random.default_rng(seed)
    n = np.array([counts[m] for m in MIXES], dtype=float)
    v = lambda key: np.array([inputs["mixes"][m][key] for m in MIXES])
    mass_var, alive_var, episode_var = map(
        v, ("paired_mass_variance", "paired_alive_variance", "paired_episode_variance")
    )
    b8_alive = v("b8_alive")
    records = []
    for rho in (0.0, 0.5, 0.9):
        shared_B8 = rng.standard_normal((reps, 3, 1))
        e2_mean, e2_sample, e2_df = _metric(rng, mass_var, n, reps, rho, shared_B8)
        e2_se = np.sqrt((e2_sample / n[None, :, None] / 9).sum(axis=1))[:, :2]
        e2_noise = e2_mean.mean(axis=1)[:, :2]
        e2_df = e2_df[:, :2]
        en = np.array([44.0, 42.0, 42.0])
        em, es, edf = _metric(rng, mass_var, en, reps, rho, shared_B8)
        e6_noise = em.mean(axis=1)[:, 2:]
        e6_se = np.sqrt((es / en[None, :, None] / 9).sum(axis=1))[:, 2:]
        e6_df = edf[:, 2:]
        # Independent source strata; the ordinary true effect is zero.
        om, _, _ = _metric(rng, mass_var, en, reps, rho)
        ordinary = om.mean(axis=1)[:, 2:] >= -2
        am, av, adf = _metric(rng, alive_var, n, reps, rho)
        ase = np.sqrt((av / n[None, :, None] / 9).sum(axis=1))[:, :2]
        per_se = np.sqrt(av / n[None, :, None])[:, :, :2]
        epm, _, _ = _metric(rng, episode_var, n, reps, rho)
        for survival in (-0.05, 0.0, 0.03, 0.06):
            alive = am[:, :, :2] + survival
            pooled = alive.mean(axis=1)
            bands = (
                (pooled - t.isf(0.05, adf[:, :2]) * ase > -0.05)
                & (alive - t.isf(0.05, n - 1)[None, :, None] * per_se > -0.075).all(axis=1)
                & (b8_alive[None, :, None] + alive >= 0.30).all(axis=1)
            )
            # Scripted no-death point guard; same effect scenarios, no invented precision.
            guards = bands & (epm[:, 0, :2] + survival >= 0)
            for effect in (0.0, 20.0, 40.0, 60.0, 80.0):
                means = e2_noise + effect
                p = t.sf(means / e2_se, e2_df)
                e2_go = _holm(p) & (means >= 10) & (means - t.isf(0.1, e2_df) * e2_se > 0) & guards
                e2_kill = means + t.isf(0.1, e2_df) * e2_se <= 0
                early = e6_noise + effect
                e6_pass = (
                    _holm(t.sf(early / e6_se, e6_df))
                    & (early - t.isf(0.1, e6_df) * e6_se > 0)
                    & ordinary
                )
                family = e2_go.any(axis=1) | e6_pass.any(axis=1)
                p_family = float(family.mean())
                records.append(
                    {
                        "counts": counts,
                        "rho_shared_B8": rho,
                        "true_mass_and_early_dividend": effect,
                        "true_survival_delta": survival,
                        "family_funding_probability": p_family,
                        "mc_se": float(np.sqrt(p_family * (1 - p_family) / reps)),
                        "MC95_interval": wilson_interval(int(family.sum()), reps),
                        "E2_any_GO": float(e2_go.any(axis=1).mean()),
                        "E6_any_NEEDS_SCALE": float(e6_pass.any(axis=1).mean()),
                        "E2_R": {
                            "GO": float(e2_go[:, 0].mean()),
                            "KILL": float(e2_kill[:, 0].mean()),
                            "INCONCLUSIVE": float((~e2_go[:, 0] & ~e2_kill[:, 0]).mean()),
                        },
                        "efficacy_only_any_E2": float(_holm(p).any(axis=1).mean()),
                    }
                )
    return records


def run(path: Path, *, reps: int, seed: int) -> dict[str, Any]:
    inputs = measured_inputs(path)
    records = [row for counts in COUNTS for row in simulate(inputs, counts, reps=reps, seed=seed)]
    summaries = []
    for counts in COUNTS:
        subset = [r for r in records if r["counts"] == counts]
        null = max(
            r["family_funding_probability"]
            for r in subset
            if r["true_mass_and_early_dividend"] == 0
        )
        power = min(
            r["family_funding_probability"]
            for r in subset
            if r["true_mass_and_early_dividend"] == 40 and r["true_survival_delta"] >= 0
        )
        null_upper = max(
            r["MC95_interval"][1] for r in subset if r["true_mass_and_early_dividend"] == 0
        )
        power_lower = min(
            r["MC95_interval"][0]
            for r in subset
            if r["true_mass_and_early_dividend"] == 40 and r["true_survival_delta"] >= 0
        )
        summaries.append(
            {
                "counts": counts,
                "worst_family_null_pass": null,
                "minimum_full_funding_at_40_nonnegative_survival": power,
                "null_MC95_upper": null_upper,
                "power_MC95_lower": power_lower,
                "passes_probability_bars": null_upper <= 0.10 and power_lower >= 0.60,
            }
        )
    selected = COUNTS[0] if summaries[0]["passes_probability_bars"] else COUNTS[1]
    return {
        "schema": "bl-ppo-e9-oc/v1",
        "platform": "Mac CPU, no gameplay",
        "seed": seed,
        "replicates": reps,
        "measured_inputs": inputs,
        "S1_inputs": {"status": "UNAVAILABLE", "required_for_admitted_forecast": True},
        "rule": RULE,
        "rule_semantic_sha256": hashlib.sha256(
            json.dumps(RULE, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("oc.py", "rule.py")
        },
        "selection": {
            "counts": selected,
            "dispatch_admitted": False,
            "reason": "Full-gate power below .60 even after allowed escalation; S1 measured variances unavailable. No further unregistered escalation.",
        },
        "summaries": summaries,
        "scenarios": records,
        "round05_continuation_sensitivities": {
            "status": "INCOMPLETE_NOT_SIMULATED",
            "requested": "E16+E17+32M plateau/continued/forgetting with rho0/.5/.9",
            "missing_inputs": [
                "Measured or registered rescue/retention/collateral joint model",
                "Registered E17 signature forecast distributions and error model",
                "S1 final measured paired variances by mix",
            ],
            "false_funding": None,
            "false_final_efficacy": None,
            "missed_winners": None,
            "expected_units": None,
        },
        "E7_Wilson": {
            "successes": 26,
            "total": 128,
            "point": 26 / 128,
            "LB90": wilson_lower(26, 128),
        },
        "assumptions": [
            "Normal paired-world marginal mean model, estimated variances chi-square; not empirical policy dynamics.",
            "Shared -B8 Gaussian component rho=0/.5/.9; variance-estimate and cross-metric covariance unknown.",
            "E6 early/ordinary variances use measured T23 full-world proxies; not measured E6 precision.",
            "True early dividend equals nominated mass scenario; ordinary true effect=0.",
            "Critic gates, integrity, limits and parity assumed pass; failure makes funding impossible.",
            "S1 final variance artifact unavailable in approved explicit roots; no substituted invented SD.",
            "Continuation rescue, retention, collateral and forecast skill not identified; round05 future-trajectory OC remains explicitly incomplete.",
        ],
    }


def wilson_interval(successes: int, count: int) -> list[float]:
    """Pointwise two-sided 95% Monte Carlo uncertainty, not gameplay precision."""
    lower = wilson_lower(successes, count, 0.025)
    upper = 1 - wilson_lower(count - successes, count, 0.025)
    return [lower, upper]


def wilson_lower(successes: int, count: int, alpha: float = 0.1) -> float:
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("invalid binomial counts")
    z = norm.isf(alpha)
    p = successes / count
    return float(
        (p + z * z / (2 * count) - z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)))
        / (1 + z * z / count)
    )


def validate_admission(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed before scientific collection; templates may still be exported."""
    if report.get("schema") != "bl-ppo-e9-oc/v1" or report.get("rule") != RULE:
        raise ValueError("E9 schema or frozen rule mismatch")
    if report.get("selection", {}).get("dispatch_admitted") is not True:
        raise ValueError(
            "E9 OC admission blocked: "
            + report.get("selection", {}).get("reason", "no positive admission")
        )
    for key in ("measured_inputs", "S1_inputs"):
        source = report.get(key, {})
        if not source.get("sha256") or len(source["sha256"]) != 64 or not source.get("mixes"):
            raise ValueError("E9 missing measured T23/S1 source provenance")
        if set(source["mixes"]) != set(MIXES):
            raise ValueError("E9 source must contain measured variances for each mix")
        for item in source["mixes"].values():
            for metric in (
                "paired_mass_variance",
                "paired_alive_variance",
                "paired_episode_variance",
            ):
                if metric not in item or not np.isfinite(item[metric]) or item[metric] < 0:
                    raise ValueError("E9 missing finite measured variance")
    counts = report["selection"].get("counts")
    if counts not in COUNTS:
        raise ValueError("E9 screen counts outside prospective registered options")
    selected = [x for x in report.get("summaries", []) if x.get("counts") == counts]
    if len(selected) != 1 or not selected[0].get("passes_probability_bars"):
        raise ValueError("E9 selected design lacks passing full-gate OC")
    rows = [x for x in report.get("scenarios", []) if x.get("counts") == counts]
    if any(
        not np.isfinite(x.get("family_funding_probability", float("nan")))
        or not 0 <= x["family_funding_probability"] <= 1
        for x in rows
    ):
        raise ValueError("E9 invalid probability")
    required = {
        (r, g, d)
        for r in (0.0, 0.5, 0.9)
        for g in (0.0, 20.0, 40.0, 60.0, 80.0)
        for d in (-0.05, 0.0, 0.03, 0.06)
    }
    if {
        (
            x.get("rho_shared_B8"),
            x.get("true_mass_and_early_dividend"),
            x.get("true_survival_delta"),
        )
        for x in rows
    } != required:
        raise ValueError("E9 incomplete joint guard-scenario grid")
    null = max(
        x["family_funding_probability"] for x in rows if x["true_mass_and_early_dividend"] == 0
    )
    power = min(
        x["family_funding_probability"]
        for x in rows
        if x["true_mass_and_early_dividend"] == 40 and x["true_survival_delta"] >= 0
    )
    if null > 0.10 or power < 0.60:
        raise ValueError("E9 recomputed full-gate probability bars fail")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t23",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "t23_r0_screen_20261017/results/t23_verdict.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    report = run(args.t23, reps=args.replicates, seed=args.seed)
    export_json(args.out, report)
    print(json.dumps({"selection": report["selection"], "summaries": report["summaries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
