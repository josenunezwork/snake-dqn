#!/usr/bin/env python3
"""Monte Carlo validation of ``src.evaluation.sequential_gate`` (numpy, single-threaded).

Operating characteristics of the default group-sequential strict gate (looks at 0.25,
0.5, 0.75, 1.0 of N_max = 243 worlds per mix, OBF spending, Bonferroni alpha 0.05/3 per
mix, scripted NI at 0.05, non-binding CP < 0.10 futility under MDE 30), simulated with
normal paired deltas of SD 140 per mix:

* ``--part mc``: vectorized replicates per scenario.  Type-I error (familywise false
  rejection and false PASS) under null configurations, power and expected worlds at
  deltas 30, 60 and 130, and the same scenarios under the fixed-N strict decision
  (Holm, as ``strict_promotion_decision``) and fixed-N Bonferroni, plus a 2M-replicate
  check of the per-look t critical values at the NI and efficacy nulls.
* ``--part crosscheck``: a few hundred replicates per scenario through the pure
  :func:`run_sequential_gate` and :func:`strict_promotion_decision`, compared with the
  vectorized replica decision-by-decision (they must agree exactly).

The vectorized replica compares t with per-look critical values from
``student_t_isf`` (identical to the pure p-value rule) and maps t to z for conditional
power through a 0.002-spaced table of ``student_t_sf`` per interim df (interpolation
error ~1e-7 in z).  Deterministic numbers (fixed seeds, sorted JSON keys, no
timestamps); only the ``cpu_seconds`` fields vary between runs.
Output goes only to ``--out`` (outside the repo).  Delta_NI = 3.5 is illustrative (the
v5 strict margin was 3.17; the real margin comes from each study's calibration).
"""

from __future__ import annotations

import os

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _name in _THREAD_VARS + ("VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"  # single-threaded numpy, set before numpy is imported

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from statistics import NormalDist  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.evaluation.screen_stats import (  # noqa: E402
    futility_plan,
    futility_stop_probability,
)
from src.evaluation.sequential_gate import (  # noqa: E402
    run_sequential_gate,
    sequential_gate_plan,
)
from src.scripts.eval_stats import (  # noqa: E402
    strict_promotion_decision,
    student_t_isf,
    student_t_sf,
)

N_MAX = 243
SD = 140.0
MDE = 30.0
DELTA_NI = 3.5
ALPHA = 0.05
MIXES = ("frozen", "scripted", "mixed")
SCRIPTED = 1
OUTCOME_KEYS = ("stop_pass", "stop_futile", "final_pass", "final_fail")
CHUNK = 20_000
_NORMAL = NormalDist()

# name -> (per-mix true deltas, between-mix correlation, replicates, seed)
SCENARIOS: Dict[str, Tuple[Tuple[float, float, float], float, int, int]] = {
    "null_indep": ((0.0, 0.0, 0.0), 0.0, 200_000, 11),
    "null_corr50": ((0.0, 0.0, 0.0), 0.5, 200_000, 12),
    "lfc_one_effect_two_null": ((130.0, 0.0, 0.0), 0.0, 200_000, 13),
    "ni_null_scripted_at_margin": ((130.0, -DELTA_NI, 130.0), 0.0, 200_000, 14),
    "delta_30": ((30.0, 30.0, 30.0), 0.0, 50_000, 21),
    "delta_60": ((60.0, 60.0, 60.0), 0.0, 50_000, 22),
    "delta_130": ((130.0, 130.0, 130.0), 0.0, 50_000, 23),
    "delta_30_corr50": ((30.0, 30.0, 30.0), 0.5, 50_000, 24),
}


def wilson(successes: int, total: int) -> Dict[str, float]:
    """Rate with a Wilson 95% interval."""
    if total == 0:
        return {"rate": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"rate": p, "low": centre - half, "high": centre + half, "n": total, "k": successes}


def draw(rng: np.random.Generator, deltas, rho: float, reps: int) -> np.ndarray:
    """Paired deltas (reps, 3, N_MAX) with per-world correlation ``rho`` between mixes."""
    if rho == 0.0:
        noise = rng.standard_normal((reps, 3, N_MAX))
    else:
        shared = rng.standard_normal((reps, 1, N_MAX))
        own = rng.standard_normal((reps, 3, N_MAX))
        noise = math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * own
    return noise * SD + np.asarray(deltas, dtype=float)[None, :, None]


def _z_of_p(p: float) -> float:
    """As ``sequential_gate._z_from_p``, but finite (+/-40) for a 0 or 1 tail."""
    if p <= 0.0:
        return 40.0
    if p >= 1.0:
        return -40.0
    return -_NORMAL.inv_cdf(p)


class Replica:
    """Vectorized replica of :func:`run_sequential_gate` for one frozen plan."""

    def __init__(self, plan) -> None:
        self.plan = plan
        self.sizes = np.asarray(plan.look_sizes)
        self.t_eff = [
            student_t_isf(p, n - 1) for p, n in zip(plan.efficacy_nominal_p, plan.look_sizes)
        ]
        self.t_ni = [student_t_isf(p, n - 1) for p, n in zip(plan.ni_nominal_p, plan.look_sizes)]
        self.grid = np.round(np.arange(-12.0, 12.0 + 1e-9, 0.002), 6)
        self.z_tables = []
        for n in plan.look_sizes[:-1]:
            tails = np.array([student_t_sf(float(t), float(n - 1)) for t in self.grid])
            self.z_tables.append(np.array([_z_of_p(p) for p in tails]))
        self.z_cut = _NORMAL.inv_cdf(plan.futility_cp)

    def t_to_z(self, look: int, t: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(t, -12.0, 12.0), self.grid, self.z_tables[look])

    def look_stats(self, x: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per look: (mean, sd, t) arrays of shape (reps, 3)."""
        out, s1, s2, start = [], 0.0, 0.0, 0
        for n in self.sizes:
            seg = x[:, :, start:n]
            s1 = s1 + seg.sum(axis=2)
            s2 = s2 + (seg * seg).sum(axis=2)
            start = n
            mean = s1 / n
            var = np.maximum(s2 - n * mean * mean, 0.0) / (n - 1)
            sd = np.sqrt(var)
            out.append((mean, sd, mean / (sd / math.sqrt(n))))
        return out

    def decide(self, x: np.ndarray, honor_futility: bool = True) -> Dict[str, np.ndarray]:
        """Outcome codes (1 STOP_PASS, 2 STOP_FUTILE, 3 FINAL_PASS, 4 FINAL_FAIL) per replicate."""
        plan, reps = self.plan, x.shape[0]
        last = len(self.sizes) - 1
        stats = self.look_stats(x)
        crossed = np.zeros((reps, 3), dtype=bool)
        ever_crossed = np.zeros((reps, 3), dtype=bool)  # all looks, ignoring any stop
        ni = np.zeros(reps, dtype=bool)
        active = np.ones(reps, dtype=bool)
        outcome = np.zeros(reps, dtype=np.int8)
        stop_look = np.full(reps, last, dtype=np.int8)
        overridden = np.zeros(reps, dtype=bool)
        c_final = plan.efficacy_boundaries[last]
        for k, n in enumerate(self.sizes):
            mean, sd, t = stats[k]
            hit = t >= self.t_eff[k]
            ever_crossed |= hit
            crossed |= hit & active[:, None]
            lower = mean[:, SCRIPTED] - self.t_ni[k] * sd[:, SCRIPTED] / math.sqrt(n)
            ni |= (lower > -DELTA_NI) & active
            qualifies = (crossed.sum(axis=1) >= plan.required_successes) & ni
            if k == last:
                outcome[active] = np.where(qualifies[active], 3, 4)
                break
            f = plan.fractions[k]
            z = self.t_to_z(k, t)
            theta = plan.mde * math.sqrt(plan.n_max) / sd
            arg = z * math.sqrt(f) + theta * (1.0 - f) - c_final
            futile = (~crossed) & (arg < self.z_cut * math.sqrt(1.0 - f))
            pass_now = active & qualifies
            futile_now = active & ~qualifies & (3 - futile.sum(axis=1) < plan.required_successes)
            outcome[pass_now] = 1
            stop_look[pass_now] = k
            active &= ~pass_now
            if honor_futility:
                outcome[futile_now] = 2
                stop_look[futile_now] = k
                active &= ~futile_now
            else:
                overridden |= futile_now
        return {
            "outcome": outcome,
            "stop_look": stop_look,
            "crossed": crossed,
            "ever_crossed": ever_crossed,
            "overridden": overridden,
            "final_stats": stats[last],
        }


def fixed_n(final_stats, deltas) -> Dict[str, np.ndarray]:
    """Fixed-N strict decision (Holm, as strict_promotion_decision) and fixed Bonferroni."""
    mean, sd, t = final_stats
    df = N_MAX - 1
    c = [student_t_isf(ALPHA / 3, df), student_t_isf(ALPHA / 2, df), student_t_isf(ALPHA, df)]
    order = np.argsort(-t, axis=1)
    sorted_t = np.take_along_axis(t, order, axis=1)
    r1 = sorted_t[:, 0] >= c[0]
    r2 = r1 & (sorted_t[:, 1] >= c[1])
    r3 = r2 & (sorted_t[:, 2] >= c[2])
    count = r1.astype(int) + r2 + r3
    ranks = np.argsort(order, axis=1)
    holm_rej = ranks < count[:, None]
    lower = mean[:, SCRIPTED] - student_t_isf(ALPHA, df) * sd[:, SCRIPTED] / math.sqrt(N_MAX)
    ni = lower > -DELTA_NI
    bonf_rej = t >= c[0]
    return {
        "holm_rejected": holm_rej,
        "holm_pass": (count >= 2) & ni,
        "bonf_rejected": bonf_rej,
        "bonf_pass": (bonf_rej.sum(axis=1) >= 2) & ni,
    }


def summarize(name: str, replica: Replica, honor: bool) -> Dict[str, Any]:
    deltas, rho, reps, seed = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    null_mask = np.asarray(deltas) <= 0.0
    tallies: Dict[str, Any] = {
        k: 0
        for k in (
            "stop_pass",
            "stop_futile",
            "final_pass",
            "final_fail",
            "false_rej_any",
            "ever_false_any",
            "overridden",
            "holm_pass",
            "bonf_pass",
            "holm_false_any",
            "bonf_false_any",
        )
    }
    looks = np.zeros(len(replica.sizes), dtype=np.int64)
    per_mix_ever = np.zeros(3, dtype=np.int64)
    worlds = 0
    done = 0
    while done < reps:
        size = min(CHUNK, reps - done)
        x = draw(rng, deltas, rho, size)
        res = replica.decide(x, honor_futility=honor)
        out = res["outcome"]
        for code, key in (
            (1, "stop_pass"),
            (2, "stop_futile"),
            (3, "final_pass"),
            (4, "final_fail"),
        ):
            tallies[key] += int((out == code).sum())
        looks += np.bincount(res["stop_look"], minlength=len(replica.sizes))
        worlds += int(replica.sizes[res["stop_look"]].sum())
        tallies["false_rej_any"] += int((res["crossed"][:, null_mask]).any(axis=1).sum())
        tallies["ever_false_any"] += int((res["ever_crossed"][:, null_mask]).any(axis=1).sum())
        per_mix_ever += res["ever_crossed"].sum(axis=0)
        tallies["overridden"] += int(res["overridden"].sum())
        fixed = fixed_n(res["final_stats"], deltas)
        tallies["holm_pass"] += int(fixed["holm_pass"].sum())
        tallies["bonf_pass"] += int(fixed["bonf_pass"].sum())
        tallies["holm_false_any"] += int(fixed["holm_rejected"][:, null_mask].any(axis=1).sum())
        tallies["bonf_false_any"] += int(fixed["bonf_rejected"][:, null_mask].any(axis=1).sum())
        done += size
    passes = tallies["stop_pass"] + tallies["final_pass"]
    mean_n = worlds / reps
    out: Dict[str, Any] = {
        "deltas": list(deltas),
        "between_mix_correlation": rho,
        "replicates": reps,
        "seed": seed,
        "honor_futility": honor,
        "sequential_pass": wilson(passes, reps),
        "outcomes": {k: tallies[k] / reps for k in OUTCOME_KEYS},
        "stop_look_distribution": (looks / reps).tolist(),
        "expected_worlds_per_mix": mean_n,
        "expected_paired_worlds_total": 3 * mean_n,
        "fixed_paired_worlds_total": 3 * N_MAX,
        "expected_saving_vs_fixed": 1.0 - mean_n / N_MAX,
        "fixed_holm_pass": wilson(tallies["holm_pass"], reps),
        "fixed_bonferroni_pass": wilson(tallies["bonf_pass"], reps),
        "per_mix_crossing_all_looks_ignoring_stops": (per_mix_ever / reps).tolist(),
    }
    if null_mask.any():
        out["null_mixes"] = [MIXES[i] for i in np.flatnonzero(null_mask)]
        out["sequential_false_rejection_any_null_mix"] = wilson(tallies["false_rej_any"], reps)
        out["sequential_false_crossing_any_null_mix_all_looks"] = wilson(
            tallies["ever_false_any"], reps
        )
        out["fixed_holm_false_rejection_any_null_mix"] = wilson(tallies["holm_false_any"], reps)
        out["fixed_bonferroni_false_rejection_any_null_mix"] = wilson(
            tallies["bonf_false_any"], reps
        )
    if not honor:
        out["futility_stops_overridden"] = tallies["overridden"] / reps
    return out


CODES = {"STOP_PASS": 1, "STOP_FUTILE": 2, "FINAL_PASS": 3, "FINAL_FAIL": 4}
CROSSCHECK = (
    "null_indep",
    "lfc_one_effect_two_null",
    "ni_null_scripted_at_margin",
    "delta_30",
    "delta_60",
)
CROSSCHECK_REPS = 250


def analytic_per_mix(plan) -> Dict[str, Any]:
    """Known-variance per-mix crossing probability (recursive integration, by symmetry)."""
    eff = futility_plan(plan.look_sizes, alpha=plan.efficacy_alpha_per_mix, max_size=N_MAX)
    out = {}
    for delta in (0.0, 30.0, 60.0, 130.0):
        prob = futility_stop_probability(eff, -delta, SD)
        out[f"{delta:g}"] = {
            "cross_probability": prob["stop_probability"],
            "cumulative_by_look": prob["cumulative_stop"],
        }
    return out


def ni_boundary_precision(replica: Replica, reps: int = 2_000_000) -> Dict[str, Any]:
    """Scripted NI alone at its null (delta = -delta_NI): crossing at any of the 4 looks.

    Isolates the z/t approximation for the NI boundaries (target 0.05 exactly) and the
    efficacy boundaries at delta 0 (target alpha / 3) with many more replicates.
    """
    rng = np.random.default_rng(31)
    ni_hits = eff_hits = done = 0
    while done < reps:
        size = min(100_000, reps - done)
        x = rng.standard_normal((size, 1, N_MAX)) * SD
        crossed_ni = np.zeros(size, dtype=bool)
        crossed_eff = np.zeros(size, dtype=bool)
        # With true delta -delta_NI, the NI lower bound (x - delta_NI) - c*se > -delta_NI
        # reduces to t(x) > c on the centred deltas x, so NI at its null is a 0.05 test.
        for k, (_, _, t) in enumerate(replica.look_stats(x)):
            crossed_ni |= t[:, 0] > replica.t_ni[k]
            crossed_eff |= t[:, 0] >= replica.t_eff[k]
        ni_hits += int(crossed_ni.sum())
        eff_hits += int(crossed_eff.sum())
        done += size
    return {
        "replicates": reps,
        "ni_any_look_at_null": wilson(ni_hits, reps),
        "ni_target": ALPHA,
        "efficacy_any_look_at_zero": wilson(eff_hits, reps),
        "efficacy_target": ALPHA / 3,
    }


def part_mc(plan, replica: Replica) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name in SCENARIOS:
        start = time.process_time()
        results[name] = {"honor_futility": summarize(name, replica, True)}
        if SCENARIOS[name][0].count(0.0) or name.startswith("ni_null"):
            results[name]["futility_overridden"] = summarize(name, replica, False)
        results[name]["cpu_seconds"] = round(time.process_time() - start, 2)
    start = time.process_time()
    precision = ni_boundary_precision(replica)
    precision["cpu_seconds"] = round(time.process_time() - start, 2)
    return {
        "scenarios": results,
        "analytic_known_variance_per_mix": analytic_per_mix(plan),
        "boundary_precision": precision,
    }


def part_crosscheck(plan, replica: Replica) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in CROSSCHECK:
        deltas, rho, _, seed = SCENARIOS[name]
        x = draw(np.random.default_rng(seed + 1000), deltas, rho, CROSSCHECK_REPS)
        res = replica.decide(x, honor_futility=True)
        fixed = fixed_n(res["final_stats"], deltas)
        mismatches, holm_mismatches = [], []
        start = time.process_time()
        for i in range(CROSSCHECK_REPS):
            by_mix = {m: x[i, j].tolist() for j, m in enumerate(MIXES)}
            pure = run_sequential_gate(plan, by_mix, DELTA_NI)
            if (CODES[pure["decision"]], pure["look"]) != (
                int(res["outcome"][i]),
                int(res["stop_look"][i]),
            ):
                mismatches.append(i)
            strict = strict_promotion_decision(by_mix, "scripted", DELTA_NI)["passes"]
            if strict != bool(fixed["holm_pass"][i]):
                holm_mismatches.append(i)
        out[name] = {
            "replicates": CROSSCHECK_REPS,
            "sequential_mismatches": mismatches,
            "fixed_holm_mismatches": holm_mismatches,
            "outcome_counts": {k: int((res["outcome"] == v).sum()) for k, v in CODES.items()},
            "cpu_seconds": round(time.process_time() - start, 2),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("mc", "crosscheck"), required=True)
    parser.add_argument("--out", required=True, help="output directory (outside the repo)")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out)
    if out_dir == _REPO_ROOT or out_dir.startswith(_REPO_ROOT + os.sep):
        raise SystemExit("--out must be outside the repository")
    os.makedirs(out_dir, exist_ok=True)
    start = time.process_time()
    plan = sequential_gate_plan(N_MAX, mde=MDE)
    replica = Replica(plan)
    body = part_mc(plan, replica) if args.part == "mc" else part_crosscheck(plan, replica)
    payload = {
        "part": args.part,
        "plan": plan.as_dict(),
        "settings": {
            "n_max": N_MAX,
            "sd": SD,
            "mde": MDE,
            "delta_ni": DELTA_NI,
            "alpha": ALPHA,
            "mixes": list(MIXES),
            "numpy": np.__version__,
        },
        "results": body,
        "cpu_seconds_total": round(time.process_time() - start, 2),
    }
    path = os.path.join(out_dir, f"{args.part}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {path} ({payload['cpu_seconds_total']} s CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
