"""Bounded, resumable development-only OC check for fixed-N T34 retention.

No gameplay is imported or executed. Only the original T3 dense development
receipts and historical T15 B8 mixed-Watch6 baseline records may be read.
Empirical T3 arm pools each contain 32 physical seeds; they are never stacked
as 64 independent worlds. No retention/precheck output is an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
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

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
DENSE = ARTIFACTS / "t3-rollout-labels-20261015/dense/ceiling"
WATCH = ARTIFACTS / "t15-mixed-roster-20261017/run-v1/records"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
MIXES = ("frozen", "scripted", "mixed")
THETAS = (20.0, 26.8, 40.0, 60.0, 80.0)
N = 32
BATCH = 250


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _number(value: Any, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("development metric must be finite numeric")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError("development metric out of range")
    return float(value)


def development_pool() -> dict[str, Any]:
    """Validate immutable development schemas and retain every input byte digest."""
    sources: dict[str, str] = {}
    pools = {}
    seed_sets = []
    for arm in ("b8_s0", "b8_s3"):
        values = []
        seeds = set()
        for env in range(32):
            path = DENSE / f"{arm}_{env}.json"
            row = json.loads(path.read_text())
            if (
                row.get("schema") != "t3-ceiling/v1"
                or row.get("status") != "PASS"
                or row.get("trigger_mode") != "dense"
                or row.get("prefix_identity") is not True
                or row.get("arm") != arm
                or row.get("env") != env
            ):
                raise ValueError(f"invalid original dense development receipt: {path}")
            seed = row["world_seed"]
            if type(seed) is not int or seed in seeds or row["baseline"]["world_seed"] != seed:
                raise ValueError("T3 physical world identity mismatch or duplicate")
            if arm == "b8_s3" and row["checkpoint"]["sha256"] != B8_SHA:
                raise ValueError("T3 B8 s3 checkpoint identity mismatch")
            seeds.add(seed)
            baseline = row["baseline"]
            survival = _number(row["alive_frame_fraction"], 0, 1)
            mass = _number(row["mass_integral"], 0)
            base_alive = _number(baseline["survival_fraction"], 0, 1)
            base_mass = _number(baseline["mass_integral"], 0)
            delta = [mass - base_mass, survival - base_alive, base_alive]
            if not math.isclose(delta[0], row["paired_delta"]["mass_integral"], abs_tol=1e-9):
                raise ValueError("T3 mass delta does not reconcile")
            if not math.isclose(
                delta[1], row["paired_delta"]["alive_frame_fraction"], abs_tol=1e-9
            ):
                raise ValueError("T3 survival delta does not reconcile")
            values.append(delta)
            sources[str(path)] = _sha(path)
        pools[arm] = values
        seed_sets.append(seeds)
    if seed_sets[0] != seed_sets[1]:
        raise ValueError("T3 checkpoint arms must share the same 32 physical worlds")
    watch = []
    watch_seeds = set()
    for index in range(25):
        path = WATCH / f"watch-{index:03}.json"
        row = json.loads(path.read_text())
        data = row["data"]
        if (
            row.get("schema") != "t15-mixed-roster-result/v1"
            or row.get("status") != "complete"
            or row.get("kind") != "watch"
            or row.get("index") != index
            or data.get("horizon") != 5000
            or data.get("frames_stepped") != 5000
            or data.get("no_errors") is not True
            or data["roster"].get("hero_checkpoint_sha256") != B8_SHA
            or data["roster"].get("watch_roster") != "mixed"
        ):
            raise ValueError(f"invalid original B8 Watch development record: {path}")
        if row["seed"] in watch_seeds or row["seed"] != data["seed"]:
            raise ValueError("Watch baseline physical seed mismatch or duplicate")
        watch_seeds.add(row["seed"])
        hero = data["hero"]
        hist = hero["decision_length_histogram"]
        if sum(hist.values()) != hero["decisions"] or hero["decisions"] != 5000:
            raise ValueError("Watch baseline decision histogram does not reconcile")
        watch.append(
            {
                "seed": row["seed"],
                "deaths": _number(hero["deaths"], 0),
                "alive_fraction_continuous_respawn": _number(hero["alive_frames"], 0, 5000) / 5000,
                "decision_mean_length": sum(int(k) * v for k, v in hist.items()) / 5000,
            }
        )
        sources[str(path)] = _sha(path)
    return {"t3_pools": pools, "watch_baselines": watch, "source_files": sources}


def intervals(samples: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match analyzer's independent equal-mix Welch interval (rep,mix,world)."""
    n = samples.shape[-1]
    mean = samples.mean(axis=2)
    variance = samples.var(axis=2, ddof=1) / n
    pooled_var = variance.sum(axis=1) / 9
    denominator = (variance * variance).sum(axis=1) / (81 * (n - 1))
    df = np.divide(
        pooled_var * pooled_var,
        denominator,
        out=np.full_like(pooled_var, 3 * (n - 1)),
        where=denominator > 0,
    )
    pooled_mean = mean.mean(axis=1)
    pooled_lower = pooled_mean - t.isf(alpha, df) * np.sqrt(pooled_var)
    mix_lower = mean - t.isf(alpha, n - 1) * np.sqrt(variance)
    return pooled_mean, pooled_lower, mix_lower


def scenarios() -> dict[str, list[float]]:
    """Predeclared amendment scenarios; no tuning to the simulated result."""
    result = {"no_change": [0.0] * 3, "pooled_minus_0.05": [-0.05] * 3}
    for index, mix in enumerate(MIXES):
        shifts = [0.0] * 3
        shifts[index] = -0.075
        result[f"{mix}_minus_0.075"] = shifts
    return result


def _models(pool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = {}
    for arm, values in pool["t3_pools"].items():
        arr = np.array(values)
        models[f"t3_dense_{arm}_scripted_shape_all_mixes"] = {
            "kind": "empirical",
            "values": values,
            "mass_sd": float(arr[:, 0].std(ddof=1)),
            "survival_sd": float(arr[:, 1].std(ddof=1)),
        }
    # Continuous respawn is a distinct estimand. Use its mass-scale variance only
    # as a declared independent-copy sensitivity, never as terminal survival data.
    watch_lengths = np.array([r["decision_mean_length"] for r in pool["watch_baselines"]])
    watch_mass_sd = float(watch_lengths.std(ddof=1) * math.sqrt(2))
    for sd in (0.25, 0.35, 0.50):
        models[f"normal_stress_sd_{sd:g}_watch_mass_scale"] = {
            "kind": "normal",
            "mass_sd": watch_mass_sd,
            "survival_sd": sd,
        }
    return models


def simulate_batch(pool: dict[str, Any], seed: int, offset: int, count: int) -> dict[str, int]:
    """Use batch-indexed streams so interruption never changes completed draws."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, offset]))
    counts: dict[str, int] = {}
    for name, model in _models(pool).items():
        if model["kind"] == "empirical":
            values = np.asarray(model["values"])
            selected = values[rng.integers(0, len(values), size=(count, 3, N))]
            mass = selected[..., 0] - values[:, 0].mean()
            alive = selected[..., 1] - values[:, 1].mean()
            base = selected[..., 2]
        else:
            mass = rng.normal(0, model["mass_sd"], (count, 3, N))
            alive = rng.normal(0, model["survival_sd"], (count, 3, N))
            base = np.full((count, 3, N), 0.8)
        _amean, alower, mixlower = intervals(alive, 0.05)
        mass_intervals = {alpha: intervals(mass, alpha) for alpha in (0.05, 0.10)}
        for scenario, shifts_list in scenarios().items():
            shifts = np.array(shifts_list)
            band = (
                (alower + shifts.mean() > -0.05)
                & np.all(mixlower + shifts > -0.075, axis=1)
                & np.all((base + alive).mean(axis=2) + shifts >= 0.30, axis=1)
            )
            counts[f"{name}|{scenario}|band_only"] = int(band.sum())
            for theta in THETAS:
                for alpha in (0.05, 0.10):
                    mean, lower, _ = mass_intervals[alpha]
                    sign = lower + theta > 0
                    skill = sign & (mean + theta >= 40)
                    key = f"{name}|{scenario}|theta={theta:g}|alpha={alpha:g}"
                    counts[key + "|joint_mass_and_band"] = int((skill & band).sum())
                    near_zero = (mean + theta >= 40) & (
                        np.abs(mass_intervals[0.10][1] + theta) <= 10
                    )
                    if scenario == "no_change":
                        counts[key + "|mass_near_zero_exception"] = int(near_zero.sum())
                        counts[key + "|joint_after_near_zero_exception"] = int(
                            (skill & band & ~near_zero).sum()
                        )
                        counts[key + "|mass_lb_positive"] = int(sign.sum())
                        counts[key + "|mass_point_and_lb"] = int(skill.sum())
        # Illustrative sequential alternative: same max N, two looks, Bonferroni
        # across looks (alpha .025 each), no futility. Early success needs +40
        # point, mass lower >0, and all bands at alpha .025. This is not registered.
        _, early_mass_lb, _ = intervals(mass[..., :16], 0.025)
        early_mass_mean = mass[..., :16].mean(axis=(1, 2))
        _, early_alive_lb, early_mix_lb = intervals(alive[..., :16], 0.025)
        early = (
            (early_mass_mean + 40 >= 40)
            & (early_mass_lb + 40 > 0)
            & (early_alive_lb > -0.05)
            & np.all(early_mix_lb > -0.075, axis=1)
            & np.all((base + alive)[..., :16].mean(axis=2) >= 0.30, axis=1)
        )
        counts[f"{name}|sequential_illustration|early_success"] = int(early.sum())
    return counts


def _wilson(k: int, n: int) -> dict[str, float | int]:
    p = k / n
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {
        "k": k,
        "n": n,
        "rate": p,
        "wilson95_low": center - half,
        "wilson95_high": center + half,
    }


def _atomic_state(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def run(
    out: Path,
    state_path: Path,
    reps: int = 20000,
    seed: int = 3420261020,
    max_seconds: float = 60,
    resume: bool = False,
) -> dict[str, Any]:
    """Checkpoint every <=250 replicates; completion emits a create-only receipt."""
    if sys.platform != "darwin":
        raise RuntimeError("retention OC preparation is Mac only")
    if type(reps) is not int or not 20000 <= reps <= 100000 or reps % BATCH:
        raise ValueError("reps must be a multiple of 250 in [20000,100000]")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 60:
        raise ValueError("max-seconds must be in (0,60]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if out.exists() or out.resolve() == state_path.resolve():
        raise FileExistsError("receipt must be fresh and distinct from state")
    started = time.monotonic()
    pool = development_pool()
    from research.t34_pathb.retention_v2 import source_inventory

    identity = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_hashes": source_inventory(REPO),
        "simulator_hash": _sha(Path(__file__)),
        "world_pool_sha256": _digest(pool),
        "reps": reps,
        "seed": seed,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("resume source/development pool/seed/replicate identity changed")
        if (
            type(state.get("completed")) is not int
            or state["completed"] % BATCH
            or not 0 <= state["completed"] <= reps
        ):
            raise ValueError("invalid completed replicate checkpoint")
    else:
        state = {"identity": identity, "completed": 0, "counts": {}}
        with state_path.open("x") as stream:
            json.dump(state, stream, sort_keys=True)
    while state["completed"] < reps:
        if time.monotonic() - started >= max_seconds:
            return {
                "status": "INCOMPLETE",
                "completed": state["completed"],
                "resume_command": "repeat the same command with --resume",
            }
        counts = simulate_batch(pool, seed, state["completed"], BATCH)
        for key, value in counts.items():
            state["counts"][key] = state["counts"].get(key, 0) + value
        state["completed"] += BATCH
        _atomic_state(state_path, state)
    rates = {key: _wilson(k, reps) for key, k in sorted(state["counts"].items())}
    protected = {
        k: v
        for k, v in rates.items()
        if "|no_change|" not in k and k.endswith("joint_mass_and_band")
    }
    failures = {k: v for k, v in protected.items() if v["rate"] > 0.06}
    costs = {}
    for name in _models(pool):
        early = rates[f"{name}|sequential_illustration|early_success"]["rate"]
        costs[name] = {
            "fixed_worlds_per_mix": 32,
            "sequential_expected_worlds_per_mix": 16 * early + 32 * (1 - early),
            "early_success_probability": early,
            "fixed_paired_worlds_total": 96,
            "sequential_expected_paired_worlds_total": 3 * (32 - 16 * early),
        }
    receipt = {
        "schema": "t34-retention-fixed-n-oc/v1",
        "status": "FAIL" if failures else "PASS",
        **identity,
        "development_only": True,
        "active_arms": ["paced", "unpaced"],
        "arm_model": (
            "common marginal development shape for both arms; no arm-specific "
            "retention observations"
        ),
        "efficacy_rows_produced": False,
        "threads": 1,
        "n_per_mix": N,
        "looks": [N],
        "theta_grid": list(THETAS),
        "alpha_grid": [0.05, 0.10],
        "models": list(_models(pool)),
        "scenarios": scenarios(),
        "completed": reps,
        "acceptance": {
            "max_joint_regression_pass_rate": 0.06,
            "failed_cells": failures,
            "criterion": "every simulated joint rate <=1.2*band_alpha=.06",
        },
        "methods": {
            "survival": (
                "one-look alpha=.05, strict pooled LB>-0.05 and every mix LB>-0.075; " "floor .30"
            ),
            "pooling": "independent mix draws, equal-mix Welch-Satterthwaite, paired world unit",
            "centering": (
                "T3 mass and alive deltas centered separately; joint shape retained "
                "within worlds"
            ),
            "empirical": (
                "each 32-world arm resampled separately; scripted shape reused as proxy "
                "for all mixes"
            ),
            "stress": (
                "independent normal survival SD .25/.35/.50; mass SD sqrt(2)*Watch "
                "baseline mean-length SD"
            ),
            "mass": (
                "one-sided LB at alpha .10 and conservative two-arm Holm first rank .05; "
                "point >=40"
            ),
            "joint": (
                "mass plus band only, all other bars assumed pass; upper bound for "
                "complete GO probability"
            ),
            "near_zero": (
                "additional no-change diagnostic excludes point>=40 and abs(raw alpha .10 "
                "mass LB)<=10, matching analyzer precedence"
            ),
            "rng": "NumPy PCG64 SeedSequence([seed, completed_batch_offset]); batch size 250",
        },
        "development_pool": pool,
        "development_summary": {
            name: {k: v for k, v in model.items() if k != "values"}
            for name, model in _models(pool).items()
        },
        "rates": rates,
        "fixed_vs_sequential": costs,
        "cost_note": (
            "Fixed N is retained as mandated by the spec: 96 paired terminal worlds, plus "
            "16 product sessions and timing, with all arms completing. Sequential illustration "
            "has looks 16/32, alpha .025 per look (Bonferroni across looks, conservative Holm "
            "rank), no futility, and early success only with point mass>=40, positive LB and "
            "survival bands. Saved worlds are an optimistic ceiling: product/timing bars and "
            "the second arm can prevent stopping; look overhead is omitted. It is not a "
            "proposed or authorized execution schedule. At a true +40 effect, the point>=40 "
            "clause alone has about 50% power even when sign power is high."
        ),
        "limitations": [
            (
                "No retention outcomes or precheck outcomes were read. This is design "
                "evidence, not efficacy."
            ),
            (
                "No development paired retention data cover frozen or mixed terminal "
                "rosters. Scripted T3 shape is a proxy."
            ),
            (
                "T3 source arms share 32 physical seeds, use different checkpoints and "
                "are separate sensitivity pools."
            ),
            (
                "Watch baselines are 25 continuous-respawn sessions; their decision "
                "mean-length variance is only a mass-scale sensitivity, not terminal "
                "survival or a paired candidate variance."
            ),
            (
                "Additive centered shifts can leave survival [0,1]; no clipping or "
                "post-result tuning is applied."
            ),
            "Normal stress is a declared distribution model; it is not additional observed worlds.",
            (
                "Wilson intervals report Monte Carlo uncertainty; acceptance uses "
                "amendment's point-rate <=.06 criterion."
            ),
            (
                "Holm-family dependence and all non-mass product/boost/no-death clauses "
                "are not modeled; full GO power can be lower."
            ),
        ],
    }
    # Reject source drift across resumable work rather than publishing a stale binding.
    if source_inventory(REPO) != identity["source_hashes"] or development_pool() != pool:
        raise ValueError("source or development data changed during simulation")
    with out.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "status": receipt["status"],
        "completed": reps,
        "failed_cells": len(failures),
        "out": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3420261020)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run(args.out, args.state, args.reps, args.seed, args.max_seconds, args.resume))
    )


if __name__ == "__main__":
    main()
