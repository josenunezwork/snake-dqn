"""Pre-data OC sensitivity on independent historical data; never imports gameplay.

The 240-world geometry is prescribed by the Round 0 specification. The
counterfactual two-look design is an informative cost comparison, not adopted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
from scipy.stats import t  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "research/survival_band_v2_20261006/survival_pools_20261006.json"
POOL_SHA = "6295400a2bee43f425e83a6f23dfb275f1d2cdb343e18ea4e7541c69900dc06a"
MIXES = ("scripted", "frozen", "mixed")
COUNTS = (96, 72, 72)
THETAS = (20.0, 26.8, 40.0, 60.0, 80.0)


def intervals(samples: list[np.ndarray], alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack([x.mean(axis=1) for x in samples], axis=1)
    vs = np.stack([x.var(axis=1, ddof=1) / x.shape[1] for x in samples], axis=1)
    pooled = means.mean(axis=1)
    variance = vs.sum(axis=1) / 9
    denominator = sum(vs[:, j] ** 2 / (samples[j].shape[1] - 1) for j in range(3)) / 81
    df = np.divide(variance**2, denominator, out=np.full_like(variance, 237), where=denominator > 0)
    return (
        pooled,
        pooled - t.isf(alpha, df) * np.sqrt(variance),
        means
        - np.stack(
            [t.isf(alpha, x.shape[1] - 1) * np.sqrt(vs[:, j]) for j, x in enumerate(samples)],
            axis=1,
        ),
    )


def _wilson_upper(k: int, n: int) -> float:
    z = 1.959963984540054
    p = k / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (
        1 + z * z / n
    )


def run(reps: int = 20000, seed: int = 202610200) -> dict[str, Any]:
    if reps < 1:
        raise ValueError("positive replicate count required")
    if hashlib.sha256(POOL.read_bytes()).hexdigest() != POOL_SHA:
        raise ValueError("independent historical pool identity mismatch")
    pool = json.loads(POOL.read_text())
    scenarios = {"no_change": (0.0, 0.0, 0.0), "pooled_minus_005": (-0.05,) * 3}
    for j, mix in enumerate(MIXES):
        scenarios[f"{mix}_minus_0075"] = tuple(-0.075 if i == j else 0.0 for i in range(3))
    models = {}
    for name in ("frp2_phase1", "frp2_phase2"):
        models[name] = {
            m: np.array(
                list(
                    zip(
                        pool[name][m]["mass_delta"],
                        np.array(pool[name][m]["candidate_survival"])
                        - np.array(pool[name][m]["incumbent_survival"]),
                    )
                )
            )
            for m in MIXES
        }
    for sd in (0.25, 0.35, 0.50):
        models[f"normal_sensitivity_alive_sd_{sd:g}"] = sd
    result: dict[str, Any] = dict(
        schema="t34-round0-oc/v1",
        reps=reps,
        seed=seed,
        source_pool_sha256=POOL_SHA,
        simulator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unit="independent synthetic paired world within mix",
        no_gameplay=True,
        models={},
    )
    rng = np.random.default_rng(seed)
    for name, model in models.items():
        totals = {f"{theta:g}/{scenario}": 0 for theta in THETAS for scenario in scenarios}
        mass_pass = {str(theta): 0 for theta in THETAS}
        early = 0
        mass_sds = None
        for offset in range(0, reps, 250):
            batch = min(250, reps - offset)
            masses, alive = [], []
            for j, mix in enumerate(MIXES):
                if isinstance(model, dict):
                    raw = model[mix]
                    draw = raw[rng.integers(0, len(raw), (batch, COUNTS[j]))]
                    masses.append(draw[:, :, 0] - raw[:, 0].mean())
                    alive.append(draw[:, :, 1] - raw[:, 1].mean())
                else:
                    # Historical phase2 mass SD, normal survival stress, rho=.97
                    raw = models["frp2_phase2"][mix]
                    z = rng.normal(size=(batch, COUNTS[j]))
                    masses.append(z * raw[:, 0].std(ddof=1))
                    alive.append(
                        model * (0.97 * z + math.sqrt(1 - 0.97**2) * rng.normal(size=z.shape))
                    )
            mass_sds = {
                m: (
                    float(models["frp2_phase2"][m][:, 0].std(ddof=1))
                    if not isinstance(model, dict)
                    else float(model[m][:, 0].std(ddof=1))
                )
                for m in MIXES
            }
            alive_pass = {}
            for scenario, shifts in scenarios.items():
                _, lower, mix_lower = intervals([x + shifts[j] for j, x in enumerate(alive)], 0.05)
                alive_pass[scenario] = (lower > -0.05) & (mix_lower > -0.075).all(axis=1)
            for theta in THETAS:
                mass_mean, mass_lower, _ = intervals([x + theta for x in masses], 0.10)
                passed_mass = (mass_mean >= 40) & (mass_lower > 0)
                mass_pass[str(theta)] += int(passed_mass.sum())
                for scenario, shifts in scenarios.items():
                    passed = passed_mass & alive_pass[scenario]
                    totals[f"{theta:g}/{scenario}"] += int(passed.sum())
            # Counterfactual one interim at N=120; Bonferroni alpha .05+.05.
            first = [x[:, : x.shape[1] // 2] + 40 for x in masses]
            mean, lower, _ = intervals(first, 0.05)
            early += int(((mean >= 40) & (lower > 0)).sum())
        rates = {
            k: dict(rate=v / reps, wilson_upper95=_wilson_upper(v, reps)) for k, v in totals.items()
        }
        error_rates = [v["rate"] for k, v in rates.items() if not k.endswith("/no_change")]
        result["models"][name] = dict(
            mass_sd=mass_sds,
            joint_mass_survival_rates=rates,
            max_protected_regression_joint_error=max(error_rates),
            protected_regression_check_pass=max(error_rates) <= 0.06,
            mass_only_power={k: v / reps for k, v in mass_pass.items()},
            fixed_expected_pairs_at_40=240,
            counterfactual_mass_only_two_look_expected_pairs_at_40=240 - 120 * early / reps,
            counterfactual_limit=(
                "Optimistic mass-only cost; excludes survival/scripted/boost bars; not adopted"
            ),
        )
    result["protected_regression_check_pass"] = all(
        model["protected_regression_check_pass"] for model in result["models"].values()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=202610200)
    args = parser.parse_args()
    payload = run(args.reps, args.seed)
    with args.out.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
