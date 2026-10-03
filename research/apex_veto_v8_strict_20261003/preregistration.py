#!/usr/bin/env python3
"""Pre-registration inputs of the v8 strict challenge (deterministic; numpy; no game played).

Reads the closed v8 Tier-1 screen (``apex-veto-v8-screen-20261002/run-v1``; its merged
``summary.json`` pinned by sha256) and writes, beside this file:

* ``screen_deltas.json``: the paired v8(8) - v7(4) ``mass_integral`` deltas per mix in world
  order, recomputed from the raw A/B records and cross-checked against the summary.  This is
  the skew-check input (Tier-0 data) frozen by sha256 in the intent.
* ``operating_characteristics.json``: the sizing rule and its result (MDE, N_max, runtime
  projection), the frozen plan, a Monte Carlo of the frozen plan in the style of
  ``research/sequential_gate_validation_20261002/simulate.py`` (its vectorized ``Replica``,
  reused with this study's N_max, per-mix screen SDs and planning delta_NI), a joint
  bootstrap of the screen's world records (efficacy, scripted NI with a resampled
  calibration margin, and the survival bands at the qualifying look) for the expected power
  and duration at the observed screen effects, a decision-by-decision cross-check of that
  bootstrap replica against ``run_sequential_gate``, and a planning-only pre-run of the
  resampling skew check (the binding check runs inside the strict run after calibration).
* ``look1_band_cost.json``: the amendment's early-stop band cost per mix (pilot candidate SD
  of survival_fraction), plus the same with the 16-world calibration reference noise.

Usage: ``python research/apex_veto_v8_strict_20261003/preregistration.py`` (about 10-20 min
of one CPU core).  Nothing here reads strict data (none exists before the intent).
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import NormalDist  # noqa: E402
from typing import Any, Dict, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sequential_gate_validation_20261002 import simulate as sim  # noqa: E402
from research.sequential_gate_validation_20261002 import skew_probe  # noqa: E402
from research.sequential_strict_template.sequential_runner import (  # noqa: E402
    DEFAULT_CAPS,
    N_MAX_CAP,
    WORKER_STOP_MARGIN_SECONDS,
    WORKERS,
)
from src.evaluation.sequential_gate import (  # noqa: E402
    run_sequential_gate,
    sequential_gate_plan,
)
from src.scripts.eval_stats import student_t_isf  # noqa: E402

HERE = Path(__file__).resolve().parent
MIXES = ("frozen", "scripted", "mixed")
SCRIPTED = MIXES.index("scripted")
SCREEN_RUN = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v8-screen-20261002/run-v1"
)
SCREEN_SUMMARY_SHA256 = "ecd205e6ec4e83c070dc67fe9c14bd9134a8464d737b9171d852c5c5c8f0c3c4"
SCREEN_WORLDS = 60
DELTAS_OUT = HERE / "screen_deltas.json"
OC_OUT = HERE / "operating_characteristics.json"
BAND_OUT = HERE / "look1_band_cost.json"

# ---- sizing rule (stated before any strict data exists) ----------------------------------
FAMILY_ALPHA = 0.05
PER_MIX_ALPHA = FAMILY_ALPHA / len(MIXES)
SIZING_POWER = 0.90
SEQUENTIAL_INFLATION = 1.02  # amendment: ~2% for four OBF looks (checked by the MC below)
MDE_STEP = 5.0
MDE_SEARCH_MAX = 100.0
PROJECTION_OVERHEAD = 1.15  # as the v7 strict package
PROJECTION_MAX_FRACTION = 0.70  # the 70% runtime rule
SIZING_RULE = (
    "per-mix SD = sample SD of the v8 screen's 60 paired deltas; sd_max = max over mixes; "
    "n_fixed(MDE) = smallest n with n >= ((t_{1-0.05/3, n-1} + t_{0.90, n-1}) * sd_max / "
    "MDE)^2 (one-sided per-mix Bonferroni alpha, 90% per-mix power at the MDE in every mix); "
    "N_max(MDE) = ceil(1.02 * n_fixed) (amendment: four OBF looks); MDE = the smallest "
    "multiple of 5 with N_max <= 300 AND projected final-stage time per worker <= 70% of the "
    "worker budget (screen A/B mean episode seconds, 2 workers, x1.15 overhead). The screen's "
    "observed mean deltas are not used."
)

# ---- study constants (as the v7 strict) ---------------------------------------------------
NI_FRACTION = 0.03
BAND_BELOW = 0.02
BAND_ABOVE = 1.0
N_CALIBRATION = 16
BAND_MARGIN_Z = 1.645
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
FUTILITY_CP = 0.10

_NORMAL = NormalDist()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


# ---------------------------------------------------------------- screen data


def load_screen() -> Dict[str, Any]:
    """Per mix, world-ordered A (v7) and B (v8) records of the closed screen."""
    summary_path = SCREEN_RUN / "merged" / "summary.json"
    actual = sha256_file(summary_path)
    if actual != SCREEN_SUMMARY_SHA256:
        raise SystemExit(f"screen summary sha256 {actual} is not the pinned one")
    summary = json.loads(summary_path.read_text())
    if summary["decision"] != "ADVANCE" or summary["smoke"] or not summary["complete"]:
        raise SystemExit("the screen is not a complete, real ADVANCE")
    entries: Dict[tuple, Dict[str, Any]] = {}
    for path in sorted(SCREEN_RUN.glob("shard-*/records/*.json")):
        entry = json.loads(path.read_text())
        if entry["arm"] in ("A", "B"):
            key = (entry["arm"], entry["mix"], int(entry["world_index"]))
            if key in entries:
                raise SystemExit(f"duplicate screen record {key}")
            entries[key] = entry
    out: Dict[str, Any] = {"summary_sha256": actual, "per_mix": {}}
    for mix in MIXES:
        rows = {"A": [], "B": []}
        for arm in ("A", "B"):
            for w in range(SCREEN_WORLDS):
                rows[arm].append(entries[(arm, mix, w)])
        for w in range(SCREEN_WORLDS):
            if rows["A"][w]["world_seed"] != rows["B"][w]["world_seed"]:
                raise SystemExit(f"{mix} world {w}: A and B seeds differ")
        deltas = [
            rows["B"][w]["record"]["mass_integral"] - rows["A"][w]["record"]["mass_integral"]
            for w in range(SCREEN_WORLDS)
        ]
        saved = summary["per_mix"][mix]["mass_integral"]["deltas_B_minus_A"]
        if len(saved) != len(deltas) or any(abs(a - b) > 1e-9 for a, b in zip(saved, deltas)):
            raise SystemExit(f"{mix}: raw-record deltas differ from summary.json")
        out["per_mix"][mix] = {
            "deltas": deltas,
            "seeds": [int(rows["A"][w]["world_seed"]) for w in range(SCREEN_WORLDS)],
            "survival_A": [r["record"]["survival_fraction"] for r in rows["A"]],
            "survival_B": [r["record"]["survival_fraction"] for r in rows["B"]],
            "mass_A": [r["record"]["mass_integral"] for r in rows["A"]],
            "wall_A": [r["wall_seconds"] for r in rows["A"]],
            "wall_B": [r["wall_seconds"] for r in rows["B"]],
        }
    seeds = [out["per_mix"][m]["seeds"] for m in MIXES]
    if not all(s == seeds[0] for s in seeds):
        raise SystemExit("mixes do not share world seeds by index (joint bootstrap needs it)")
    return out


def describe(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    centred = x - x.mean()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "skewness": float(np.mean(centred**3) / np.std(centred) ** 3),
        "zeros": int(np.sum(x == 0.0)),
    }


# ---------------------------------------------------------------- sizing


def n_fixed(sd: float, mde: float) -> int:
    n = 2
    while True:
        need = (
            (student_t_isf(PER_MIX_ALPHA, n - 1) + student_t_isf(1 - SIZING_POWER, n - 1))
            * sd
            / mde
        ) ** 2
        if n >= need:
            return n
        n += 1


def runtime_projection(screen: Mapping[str, Any], n_max: int) -> Dict[str, Any]:
    per_mix = {}
    for mix in MIXES:
        a, b = screen["per_mix"][mix]["wall_A"], screen["per_mix"][mix]["wall_B"]
        per_mix[mix] = {
            "incumbent_v7_mean_s": float(np.mean(a)),
            "candidate_v8_mean_s": float(np.mean(b)),
        }
    triplet = sum(r["incumbent_v7_mean_s"] + r["candidate_v8_mean_s"] for r in per_mix.values())
    budget = DEFAULT_CAPS["final"] - WORKER_STOP_MARGIN_SECONDS
    projected = n_max * triplet / WORKERS * PROJECTION_OVERHEAD
    return {
        "basis": "v8 screen A/B record wall_seconds (3 concurrent shard processes, 2 torch "
        "threads each; the screen's v8 arm also ran a diagnostic reference v7, which the "
        "strict candidate does not, so this is conservative)",
        "per_mix_episode_seconds": per_mix,
        "pair_seconds_per_world_triplet": triplet,
        "workers": WORKERS,
        "overhead_factor": PROJECTION_OVERHEAD,
        "projected_seconds_per_worker_at_n_max": projected,
        "worker_budget_seconds": budget,
        "fraction_of_worker_budget": projected / budget,
        "max_fraction": PROJECTION_MAX_FRACTION,
        "within_limit": projected / budget <= PROJECTION_MAX_FRACTION,
    }


def sizing(screen: Mapping[str, Any]) -> Dict[str, Any]:
    sds = {m: describe(screen["per_mix"][m]["deltas"])["sd"] for m in MIXES}
    sd_max = max(sds.values())
    table = []
    chosen = None
    mde = MDE_STEP
    while mde <= MDE_SEARCH_MAX:
        nf = n_fixed(sd_max, mde)
        n_max = math.ceil(SEQUENTIAL_INFLATION * nf - 1e-9)
        proj = runtime_projection(screen, n_max)
        feasible = n_max <= N_MAX_CAP and proj["within_limit"]
        table.append(
            {
                "mde": mde,
                "n_fixed_90": nf,
                "n_max": n_max,
                "runtime_fraction": proj["fraction_of_worker_budget"],
                "feasible": feasible,
            }
        )
        if feasible and chosen is None:
            chosen = {"mde": mde, "n_fixed_90": nf, "n_max": n_max}
        mde += MDE_STEP
    if chosen is None:
        raise SystemExit("no MDE <= 100 is feasible: STOP_INFEASIBLE")
    return {
        "rule": SIZING_RULE,
        "per_mix_delta_sd": sds,
        "sd_max": sd_max,
        "binding_mix": max(sds, key=sds.get),
        "per_mix_alpha": PER_MIX_ALPHA,
        "power": SIZING_POWER,
        "sequential_inflation": SEQUENTIAL_INFLATION,
        "table": [row for row in table if row["mde"] <= chosen["mde"] + 3 * MDE_STEP],
        "chosen": chosen,
        "runtime_projection": runtime_projection(screen, chosen["n_max"]),
        "screen_effects_used": False,
    }


# ---------------------------------------------------------------- normal Monte Carlo


def configure_sim(n_max: int, sds: Sequence[float], mde: float, delta_ni: float) -> None:
    """Point simulate.py's module constants at this study (its code is reused unchanged)."""
    sd_vec = np.asarray(sds, dtype=float)

    def draw(rng: np.random.Generator, deltas, rho: float, reps: int) -> np.ndarray:
        if rho == 0.0:
            noise = rng.standard_normal((reps, 3, n_max))
        else:
            shared = rng.standard_normal((reps, 1, n_max))
            own = rng.standard_normal((reps, 3, n_max))
            noise = math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * own
        return noise * sd_vec[None, :, None] + np.asarray(deltas, dtype=float)[None, :, None]

    sim.N_MAX = n_max
    sim.SD = float(max(sds))
    sim.MDE = mde
    sim.DELTA_NI = delta_ni
    sim.draw = draw


def normal_scenarios(mde: float, delta_ni: float, screen_means: Sequence[float], rho: float):
    big = 130.0
    obs = tuple(float(v) for v in screen_means)
    return {
        "null_indep": ((0.0, 0.0, 0.0), 0.0, 200_000, 11),
        "null_screen_corr": ((0.0, 0.0, 0.0), rho, 200_000, 12),
        "lfc_one_effect_two_null": ((big, 0.0, 0.0), 0.0, 200_000, 13),
        "ni_null_scripted_at_margin": ((big, -delta_ni, big), 0.0, 200_000, 14),
        "delta_mde": ((mde, mde, mde), 0.0, 50_000, 21),
        "delta_mde_screen_corr": ((mde, mde, mde), rho, 50_000, 22),
        "screen_means": (obs, 0.0, 50_000, 23),
        "screen_means_screen_corr": (obs, rho, 50_000, 24),
        "frozen_mean_scripted_zero_mixed_zero": ((obs[0], 0.0, 0.0), 0.0, 50_000, 25),
        "frozen_mean_scripted_mde_mixed_zero": ((obs[0], mde, 0.0), 0.0, 50_000, 26),
        "frozen_lb90_scripted_mean_mixed_mean": ((15.8, obs[1], obs[2]), 0.0, 50_000, 27),
    }


# ---------------------------------------------------------------- joint bootstrap


def bootstrap_decide(replica, x, surv, ref, delta_ni, plan):
    """``Replica.decide`` with a per-replicate delta_NI and the survival bands.

    x: (reps, 3, N) mass deltas; surv: (reps, 3, N) candidate survival; ref: (reps, 3)
    calibration reference; delta_ni: (reps,).  Codes: 1 STOP_PASS, 2 STOP_FUTILE,
    3 FINAL_PASS, 4 FINAL_FAIL, 5 STOP_FAIL_BANDS.  Bands are judged once, at the
    qualifying look (``block_at_stop``), with the ``band_check`` interim margin.
    """
    reps = x.shape[0]
    last = len(replica.sizes) - 1
    stats = replica.look_stats(x)
    crossed = np.zeros((reps, 3), dtype=bool)
    ni = np.zeros(reps, dtype=bool)
    active = np.ones(reps, dtype=bool)
    outcome = np.zeros(reps, dtype=np.int8)
    stop_look = np.full(reps, last, dtype=np.int8)
    c_final = plan.efficacy_boundaries[last]
    lower_bound = ref - BAND_BELOW
    upper_bound = ref + BAND_ABOVE
    for k, n in enumerate(replica.sizes):
        mean, sd, t = stats[k]
        crossed |= (t >= replica.t_eff[k]) & active[:, None]
        lower = mean[:, SCRIPTED] - replica.t_ni[k] * sd[:, SCRIPTED] / math.sqrt(n)
        ni |= (lower > -delta_ni) & active
        qualifies = (crossed.sum(axis=1) >= plan.required_successes) & ni
        s = surv[:, :, :n]
        s_mean = s.mean(axis=2)
        s_sd = s.std(axis=2, ddof=1)
        margin = BAND_MARGIN_Z * s_sd * max(0.0, 1.0 / math.sqrt(n) - 1.0 / math.sqrt(plan.n_max))
        bands = np.all((lower_bound + margin <= s_mean) & (s_mean <= upper_bound - margin), axis=1)
        if k == last:
            outcome[active] = np.where((qualifies & bands)[active], 3, 4)
            break
        f = plan.fractions[k]
        z = replica.t_to_z(k, t)
        theta = plan.mde * math.sqrt(plan.n_max) / sd
        arg = z * math.sqrt(f) + theta * (1.0 - f) - c_final
        futile = (~crossed) & (arg < replica.z_cut * math.sqrt(1.0 - f))
        stop_q = active & qualifies
        futile_now = active & ~qualifies & (3 - futile.sum(axis=1) < plan.required_successes)
        outcome[stop_q] = np.where(bands[stop_q], 1, 5)
        stop_look[stop_q] = k
        active &= ~stop_q
        outcome[futile_now] = 2
        stop_look[futile_now] = k
        active &= ~futile_now
    return {"outcome": outcome, "stop_look": stop_look, "crossed": crossed}


def bootstrap_arrays(screen, rng, reps: int, n_max: int, shift=None):
    """Joint world resampling (same world index for all mixes), plus 16 calibration worlds."""
    deltas = np.asarray([screen["per_mix"][m]["deltas"] for m in MIXES], dtype=float)
    if shift is not None:
        deltas = deltas - deltas.mean(axis=1, keepdims=True) + np.asarray(shift)[:, None]
    surv_b = np.asarray([screen["per_mix"][m]["survival_B"] for m in MIXES], dtype=float)
    surv_a = np.asarray([screen["per_mix"][m]["survival_A"] for m in MIXES], dtype=float)
    mass_a_scripted = np.asarray(screen["per_mix"]["scripted"]["mass_A"], dtype=float)
    idx = rng.integers(0, SCREEN_WORLDS, size=(reps, n_max))
    cidx = rng.integers(0, SCREEN_WORLDS, size=(reps, N_CALIBRATION))
    x = deltas[:, idx].transpose(1, 0, 2)
    surv = surv_b[:, idx].transpose(1, 0, 2)
    ref = surv_a[:, cidx].mean(axis=2).T
    delta_ni = NI_FRACTION * mass_a_scripted[cidx].mean(axis=1)
    return x, surv, ref, delta_ni


def bootstrap_power(screen, plan, replica, reps: int, seed: int, shift=None) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    codes = {
        1: "STOP_PASS",
        2: "STOP_FUTILE",
        3: "FINAL_PASS",
        4: "FINAL_FAIL",
        5: "STOP_FAIL_BANDS",
    }
    counts = {name: 0 for name in codes.values()}
    looks = np.zeros(len(plan.look_sizes), dtype=np.int64)
    per_mix_cross = np.zeros(3, dtype=np.int64)
    worlds = 0
    gate_without_bands = 0
    done = 0
    while done < reps:
        k = min(5_000, reps - done)
        x, surv, ref, dni = bootstrap_arrays(screen, rng, k, plan.n_max, shift)
        res = bootstrap_decide(replica, x, surv, ref, dni, plan)
        for code, name in codes.items():
            counts[name] += int((res["outcome"] == code).sum())
        looks += np.bincount(res["stop_look"], minlength=len(plan.look_sizes))
        worlds += int(np.asarray(plan.look_sizes)[res["stop_look"]].sum())
        per_mix_cross += res["crossed"].sum(axis=0)
        no_band = replica.decide(x, honor_futility=True)  # module DELTA_NI = planning value
        gate_without_bands += int(np.isin(no_band["outcome"], (1, 3)).sum())
        done += k
    passes = counts["STOP_PASS"] + counts["FINAL_PASS"]
    return {
        "replicates": reps,
        "seed": seed,
        "shifted_means": None if shift is None else list(shift),
        "pass_probability": sim.wilson(passes, reps),
        "outcomes": {name: c / reps for name, c in counts.items()},
        "stop_look_distribution": (looks / reps).tolist(),
        "expected_worlds_per_mix": worlds / reps,
        "per_mix_crossing_by_stop": dict(zip(MIXES, (per_mix_cross / reps).tolist())),
        "pass_probability_efficacy_and_ni_only_planning_delta_ni": gate_without_bands / reps,
    }


def crosscheck_bootstrap(screen, plan, replica, reps: int = 300, seed: int = 4242) -> Dict:
    """The bootstrap replica vs the pure ``run_sequential_gate`` (with per-look band verdicts
    from ``band_check``), decision by decision."""
    from src.evaluation.sequential_gate import band_check

    rng = np.random.default_rng(seed)
    x, surv, ref, dni = bootstrap_arrays(screen, rng, reps, plan.n_max)
    res = bootstrap_decide(replica, x, surv, ref, dni, plan)
    names = {
        "STOP_PASS": 1,
        "STOP_FUTILE": 2,
        "FINAL_PASS": 3,
        "FINAL_FAIL": 4,
        "STOP_FAIL_BANDS": 5,
    }
    mismatches = []
    for i in range(reps):
        bands = []
        for n in plan.look_sizes:
            ok = True
            for j in range(3):
                ok &= band_check(
                    surv[i, j, :n].tolist(),
                    float(ref[i, j] - BAND_BELOW),
                    float(ref[i, j] + BAND_ABOVE),
                    n_final=plan.n_max,
                    margin_z=plan.band_margin_z,
                )["passes"]
            bands.append(bool(ok))
        pure = run_sequential_gate(
            plan, {m: x[i, j].tolist() for j, m in enumerate(MIXES)}, float(dni[i]), bands
        )
        if (names[pure["decision"]], pure["look"]) != (
            int(res["outcome"][i]),
            int(res["stop_look"][i]),
        ):
            mismatches.append(i)
    counts = {k: int((res["outcome"] == v).sum()) for k, v in names.items()}
    return {"replicates": reps, "seed": seed, "mismatches": mismatches, "outcome_counts": counts}


# ---------------------------------------------------------------- band cost


def band_cost(screen, plan) -> Dict[str, Any]:
    n1, n_max = plan.look_sizes[0], plan.n_max
    out: Dict[str, Any] = {
        "definition": (
            "amendment: probability that a candidate whose true mean survival_fraction "
            "equals the reference fails the band [ref - 0.02, ref + 1] at a look-1 stop, "
            "with band_check's margin 1.645 * sd * (1/sqrt(n1) - 1/sqrt(N_max)) and sd = "
            "the pilot (screen v8 arm) per-episode SD of survival_fraction; normal approx."
        ),
        "look_sizes": list(plan.look_sizes),
        "band_margin_z": plan.band_margin_z,
        "per_mix": {},
    }
    for mix in MIXES:
        sd_b = float(np.std(screen["per_mix"][mix]["survival_B"], ddof=1))
        sd_a = float(np.std(screen["per_mix"][mix]["survival_A"], ddof=1))
        obs_diff = float(
            np.mean(screen["per_mix"][mix]["survival_B"])
            - np.mean(screen["per_mix"][mix]["survival_A"])
        )
        rows = {}
        for label, n in (("look_1", n1), ("look_2", plan.look_sizes[1]), ("final", n_max)):
            margin = BAND_MARGIN_Z * sd_b * max(0.0, 1 / math.sqrt(n) - 1 / math.sqrt(n_max))
            se_c = sd_b / math.sqrt(n)
            se_ref = sd_a / math.sqrt(N_CALIBRATION)
            se_both = math.sqrt(se_c**2 + se_ref**2)
            rows[label] = {
                "n": n,
                "margin": margin,
                "fail_if_equal_ref_known": _NORMAL.cdf((-BAND_BELOW + margin) / se_c),
                "fail_if_equal_with_calibration_noise": _NORMAL.cdf(
                    (-BAND_BELOW + margin) / se_both
                ),
                "fail_at_screen_survival_difference_with_calibration_noise": _NORMAL.cdf(
                    (-BAND_BELOW + margin - obs_diff) / se_both
                ),
            }
        out["per_mix"][mix] = {
            "pilot_candidate_sd": sd_b,
            "pilot_incumbent_sd": sd_a,
            "screen_survival_difference_v8_minus_v7": obs_diff,
            "calibration_worlds": N_CALIBRATION,
            **rows,
        }
    out["note"] = (
        "The reference is the 16-world calibration incumbent mean (v7 strict rule, kept), so "
        "its own sampling SD (incumbent SD / 4, about 0.05-0.08) is larger than the 0.02 band; "
        "the 'with_calibration_noise' rows include it. These costs are large; they are a "
        "property of the v7 band rule applied at n1 worlds and are disclosed, not changed."
    )
    return out


# ---------------------------------------------------------------- planning-only skew pre-run


def skew_prerun(deltas_by_mix, n_max: int, delta_ni: float, reps: int) -> Dict[str, Any]:
    tmp = HERE / ".skew_prerun_tmp.json"
    out = {}
    try:
        for mix in MIXES:
            tmp.write_text(json.dumps(deltas_by_mix[mix]))
            began = time.process_time()
            res = skew_probe.part_resample(str(tmp), reps, n_max, delta_ni, FRACTIONS)
            res["cpu_seconds"] = round(time.process_time() - began, 1)
            out[mix] = res
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "status": "planning only, not the binding check (that runs in the strict run after "
        "calibration, with the calibrated delta_NI, through the frozen probe)",
        "delta_ni_used": delta_ni,
        "reps": reps,
        "per_mix": out,
        "would_pass_per_template_rule": all(
            out[m]["efficacy_null"]["any_look"] <= 1.2 * PER_MIX_ALPHA for m in MIXES
        )
        and out["scripted"]["ni_null"]["any_look"] <= 0.06,
    }


# ---------------------------------------------------------------- main


def main() -> int:
    began = time.process_time()
    screen = load_screen()
    deltas_by_mix = {m: screen["per_mix"][m]["deltas"] for m in MIXES}
    write_json(DELTAS_OUT, deltas_by_mix)
    size = sizing(screen)
    n_max, mde = size["chosen"]["n_max"], size["chosen"]["mde"]
    plan = sequential_gate_plan(
        n_max,
        mde=mde,
        mixes=MIXES,
        scripted_mix="scripted",
        fractions=FRACTIONS,
        futility_cp=FUTILITY_CP,
        band_margin_z=BAND_MARGIN_Z,
    )
    stats = {m: describe(deltas_by_mix[m]) for m in MIXES}
    sds = [stats[m]["sd"] for m in MIXES]
    means = [stats[m]["mean"] for m in MIXES]
    planning_ni = NI_FRACTION * float(np.mean(screen["per_mix"]["scripted"]["mass_A"]))
    arr = np.asarray([deltas_by_mix[m] for m in MIXES])
    corr = np.corrcoef(arr)
    rho = float(np.mean([corr[0, 1], corr[0, 2], corr[1, 2]]))
    configure_sim(n_max, sds, mde, planning_ni)
    sim.SCENARIOS = normal_scenarios(mde, planning_ni, means, max(rho, 0.0))
    replica = sim.Replica(plan)
    mc = sim.part_mc(plan, replica)
    boot = {
        "screen_observed_effects": bootstrap_power(screen, plan, replica, 40_000, 101),
        "screen_shape_at_mde_all_mixes": bootstrap_power(
            screen, plan, replica, 20_000, 102, shift=(mde, mde, mde)
        ),
        "screen_shape_global_null": bootstrap_power(
            screen, plan, replica, 40_000, 103, shift=(0.0, 0.0, 0.0)
        ),
    }
    cross = crosscheck_bootstrap(screen, plan, replica)
    if cross["mismatches"]:
        raise SystemExit(f"bootstrap replica disagrees with run_sequential_gate: {cross}")
    proj = size["runtime_projection"]
    triplet = proj["pair_seconds_per_world_triplet"]
    exp_worlds = boot["screen_observed_effects"]["expected_worlds_per_mix"]
    hours = {
        "calibration_h": N_CALIBRATION
        * sum(r["incumbent_v7_mean_s"] for r in proj["per_mix_episode_seconds"].values())
        / WORKERS
        * PROJECTION_OVERHEAD
        / 3600,
        "final_expected_h_at_screen_effects": exp_worlds
        * triplet
        / WORKERS
        * PROJECTION_OVERHEAD
        / 3600,
        "final_max_h_at_n_max": proj["projected_seconds_per_worker_at_n_max"] / 3600,
        "per_look_h": [
            (b - a) * triplet / WORKERS * PROJECTION_OVERHEAD / 3600
            for a, b in zip((0,) + tuple(plan.look_sizes[:-1]), plan.look_sizes)
        ],
    }
    skew = skew_prerun(deltas_by_mix, n_max, planning_ni, 300_000)
    report = {
        "schema": "apex-veto-v8-strict-oc/v1",
        "screen": {
            "run": str(SCREEN_RUN),
            "summary_sha256": screen["summary_sha256"],
            "per_mix_deltas": stats,
            "between_mix_delta_correlation_mean": rho,
            "between_mix_delta_correlation": corr.tolist(),
            "screen_deltas_path": str(DELTAS_OUT.relative_to(REPO)),
            "screen_deltas_sha256": sha256_file(DELTAS_OUT),
        },
        "sizing": size,
        "plan": plan.as_dict(),
        "planning_delta_ni": {
            "value": planning_ni,
            "rule": "0.03 x screen v7 (arm A) scripted mean mass_integral; the binding "
            "delta_NI is 0.03 x the calibration incumbent scripted mean",
        },
        "normal_monte_carlo": {
            "method": "simulate.py Replica/summarize/part_mc reused unchanged; normal deltas "
            "with the per-mix screen SDs; bands not simulated here",
            "per_mix_sd": dict(zip(MIXES, sds)),
            **mc,
        },
        "joint_bootstrap": {
            "method": "resample screen world indices jointly across mixes (N_max per "
            "replicate), candidate survival from the same worlds, a 16-world calibration "
            "reference (incumbent survival) and delta_NI (0.03 x incumbent scripted mass) "
            "resampled per replicate; bands judged at the qualifying look with the band_check "
            "margin; futility followed",
            **boot,
            "crosscheck_vs_run_sequential_gate": cross,
        },
        "expected_duration": hours,
        "skew_check_prerun": skew,
        "cpu_seconds_total": round(time.process_time() - began, 1),
    }
    write_json(OC_OUT, report)
    write_json(BAND_OUT, {"schema": "apex-veto-v8-strict-band-cost/v1", **band_cost(screen, plan)})
    print(json.dumps({"n_max": n_max, "mde": mde, "oc": str(OC_OUT), "bands": str(BAND_OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
