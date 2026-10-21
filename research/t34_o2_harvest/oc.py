"""Pre-data O2 operating characteristics from unsealed historical paired deltas.

No gameplay or O2 outcomes. The source is a pinned 2026-10-06 development
pool, not the sealed Round 0 artifact root. These are model sensitivities,
not an identified distribution for the unobserved O2 incremental contrast.
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
from scipy.stats import norm, t  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "research/survival_band_v2_20261006/survival_pools_20261006.json"
POOL_SHA = "6295400a2bee43f425e83a6f23dfb275f1d2cdb343e18ea4e7541c69900dc06a"
MIXES = ("scripted", "frozen", "mixed")


def intervals(samples: list[np.ndarray], alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack([x.mean(axis=1) for x in samples], axis=1)
    vs = np.stack([x.var(axis=1, ddof=1) / x.shape[1] for x in samples], axis=1)
    mu, variance = means.mean(axis=1), vs.sum(axis=1) / 9
    denominator = sum(vs[:, j] ** 2 / (samples[j].shape[1] - 1) for j in range(3)) / 81
    df = np.divide(variance**2, denominator, out=np.full_like(variance, 93), where=denominator > 0)
    se = np.sqrt(variance)
    statistic = np.divide(
        mu, se, out=np.where(mu > 0, np.inf, np.where(mu < 0, -np.inf, 0.0)), where=se > 0
    )
    return mu, mu - t.isf(alpha, df) * se, t.sf(statistic, df)


def wilson(k: int, n: int) -> list[float]:
    z, p = 1.959963984540054, k / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(center - half) / (1 + z * z / n), (center + half) / (1 + z * z / n)]


def run(reps: int = 20000, seed: int = 202610212) -> dict[str, Any]:
    if reps < 1:
        raise ValueError("positive replicate count required")
    if hashlib.sha256(POOL.read_bytes()).hexdigest() != POOL_SHA:
        raise ValueError("historical development pool SHA mismatch")
    pool = json.loads(POOL.read_text())
    result = {
        "schema": "t34-o2-oc/v1",
        "reps": reps,
        "seed": seed,
        "no_gameplay": True,
        "source": str(POOL.relative_to(ROOT)),
        "source_sha256": POOL_SHA,
        "simulator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixed_worlds": 96,
        "counts_per_mix": 32,
        "family_alpha": 0.10,
        "look_choice": "fixed N, one look as JUDGMENT mandates; no optional stopping",
        "adequacy_target": 0.80,
        "models": {},
        "interpretation": (
            "At a true gain exactly +40, the mean>=40 point bar limits full-pass "
            "probability to about 50% for symmetric continuous estimation error, "
            "irrespective of N. Skew bootstrap models need not be exactly 50%. No N "
            "can promise 80% full GO power at that boundary. Holm superiority-only "
            "power is separately reported. No new bank or extension authorized."
        ),
        "limitations": (
            "Historical vector-policy variance is a sensitivity, not O2 variance. "
            "Incremental harvest variance is unobserved: 0.5x and 1x historical SD "
            "with independent incremental noise are explicit assumptions. Boost and "
            "absolute survival floor held passing; absence of faults assumed; "
            "scripted no-death true gain set +0.06 for power scenarios. These "
            "conditional full-pass rates do not establish actual complete-controller "
            "power."
        ),
    }
    rng = np.random.default_rng(seed)
    scenarios = {
        "global_null": (0.0, 0.0),
        "incremental_null": (80.0, 0.0),
        "true40_increment10": (40.0, 10.0),
        "true40_increment40": (40.0, 40.0),
        "true60_increment20": (60.0, 20.0),
        "true80_increment40": (80.0, 40.0),
    }
    guard_scenarios = {"no_change": (0.0, 0.0, 0.0), "pooled_minus005": (-0.05,) * 3}
    for j, mix in enumerate(MIXES):
        guard_scenarios[mix + "_minus0075"] = tuple(-0.075 if i == j else 0.0 for i in range(3))
    for phase in ("frp2_phase1", "frp2_phase2"):
        raw = {
            m: np.asarray(
                list(
                    zip(
                        pool[phase][m]["mass_delta"],
                        np.asarray(pool[phase][m]["candidate_survival"])
                        - np.asarray(pool[phase][m]["incumbent_survival"]),
                        (np.asarray(pool[phase][m]["candidate_survival"]) >= 1).astype(float)
                        - (np.asarray(pool[phase][m]["incumbent_survival"]) >= 1).astype(float),
                    )
                )
            )
            for m in MIXES
        }
        sds = {m: float(raw[m][:, 0].std(ddof=1)) for m in MIXES}
        se96 = math.sqrt(sum(s * s / 32 for s in sds.values()) / 9)
        planning = {
            str(n): {
                "conservative_first_holm_superiority_power_at40_normal": float(
                    norm.sf(norm.isf(0.05) - 40 / (se96 * math.sqrt(96 / n)))
                ),
                "full_point_bar_ceiling_symmetric_at40": 0.5,
            }
            for n in (96, 384, 1536, 6144)
        }
        n80 = int(
            math.ceil(
                (norm.isf(0.05) + norm.ppf(0.8)) ** 2
                * sum(s * s for s in sds.values())
                / (3 * 40**2)
                / 3
            )
            * 3
        )
        for ratio in (0.5, 1.0):
            key = f"{phase}_increment_sd_ratio_{ratio:g}"
            totals = {
                s: {
                    k: 0
                    for k in (
                        "b8_lb90",
                        "holm_both",
                        "mass_both_with_point40",
                        "conditional_full_pass",
                    )
                }
                for s in scenarios
            }
            guards = {s: 0 for s in guard_scenarios}
            null_any = 0
            for offset in range(0, reps, 250):
                b = min(250, reps - offset)
                masses, incs, alive, deaths = [], [], [], []
                for m in MIXES:
                    r = raw[m]
                    draw = r[rng.integers(0, len(r), (b, 32))]
                    independent = r[rng.integers(0, len(r), (b, 32))]
                    masses.append(draw[:, :, 0] - r[:, 0].mean())
                    incs.append(ratio * (independent[:, :, 0] - r[:, 0].mean()))
                    alive.append(draw[:, :, 1] - r[:, 1].mean())
                    deaths.append(draw[:, :, 2] - r[:, 2].mean())
                for name, shifts in guard_scenarios.items():
                    _, lower, _ = intervals([a + shifts[j] for j, a in enumerate(alive)], 0.05)
                    mix_ok = np.stack(
                        [
                            (a + shifts[j]).mean(axis=1)
                            - t.isf(0.05, 31) * (a.std(axis=1, ddof=1) / math.sqrt(32))
                            > -0.075
                            for j, a in enumerate(alive)
                        ],
                        axis=1,
                    ).all(axis=1)
                    guards[name] += int(((lower > -0.05) & mix_ok).sum())
                _, guardlower, _ = intervals(alive, 0.05)
                guardok = (guardlower > -0.05) & np.stack(
                    [
                        a.mean(axis=1) - t.isf(0.05, 31) * a.std(axis=1, ddof=1) / math.sqrt(32)
                        > -0.075
                        for a in alive
                    ],
                    axis=1,
                ).all(axis=1)
                scriptok = deaths[0].mean(axis=1) + 0.06 >= 0.03
                for name, (gain, increment) in scenarios.items():
                    mean, lower, p1 = intervals([a + gain for a in masses], 0.1)
                    _, _, p2 = intervals([a + increment for a in incs], 0.1)
                    holm = (np.minimum(p1, p2) < 0.05) & (np.maximum(p1, p2) < 0.10)
                    point = holm & (mean >= 40)
                    full = point & guardok & scriptok
                    for flag, array in (
                        ("b8_lb90", lower > 0),
                        ("holm_both", holm),
                        ("mass_both_with_point40", point),
                        ("conditional_full_pass", full),
                    ):
                        totals[name][flag] += int(array.sum())
                    if name == "global_null":
                        null_any += int((np.minimum(p1, p2) < 0.05).sum())
            alive_sds = {m: float(raw[m][:, 1].std(ddof=1)) for m in MIXES}
            z_union = float(norm.isf(0.05) + norm.ppf(0.95))
            guard_total = max(
                z_union**2 * sum(s * s for s in alive_sds.values()) / (3 * 0.05**2),
                3 * z_union**2 * max(s * s for s in alive_sds.values()) / 0.075**2,
            )
            guard_n80 = 3 * int(math.ceil(guard_total / 3))
            result["models"][key] = {
                "source_n_per_mix": {m: len(raw[m]) for m in MIXES},
                "mass_sd": sds,
                "pooled_se96": se96,
                "alive_delta_sd": alive_sds,
                "normal_worlds_guard_a80pct_union_bound_at_no_change": guard_n80,
                "normal_guard_planning_limit": (
                    "Normal known-variance approximation; four lower bounds each assigned at "
                    "most 5% failure for union power >=80%; ignores absolute floor and "
                    "t-estimation. Not registered and not a forecast of O2 variance."
                ),
                "scenarios": {
                    s: {k: {"rate": v / reps, "wilson95": wilson(v, reps)} for k, v in d.items()}
                    for s, d in totals.items()
                },
                "global_null_any_holm_rejection": {
                    "rate": null_any / reps,
                    "wilson95": wilson(null_any, reps),
                },
                "survival_guard_a_acceptance": {
                    s: {"rate": v / reps, "wilson95": wilson(v, reps)} for s, v in guards.items()
                },
                "survival_protected_boundary_check_le006": all(
                    v / reps <= 0.06 for s, v in guards.items() if s != "no_change"
                ),
                "prospective_normal_sizes_not_registered": planning,
                "normal_worlds_for80pct_conservative_lb_at40": n80,
            }
    result["full_go_power_at_true40_adequate"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20000)
    args = parser.parse_args()
    with args.out.open("x") as handle:
        json.dump(run(args.reps), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
