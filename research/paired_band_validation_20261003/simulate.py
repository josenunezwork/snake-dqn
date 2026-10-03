"""Paired-band validation: old calibration-reference band vs paired noninferiority band.

Label: diagnostic (Tier 0 style).  Resamples saved paired records of two COMPLETED runs
(v7 strict final stage and the v8 Tier-1 screen); it runs no game and reads nothing from the
running v8 strict gate.  See README.md for the question, method and results.

Parts (all single-threaded, fixed seeds):

* ``extract``    read the saved records once and write the compact data file
                 ``paired_survival_20261003.json`` (survival fraction of both arms, mass delta,
                 per mix and world seed, plus the v7 calibration incumbent survival).
* ``bands``      band-only operating characteristics at fixed look sizes n = 63..249.
* ``gate``       joint (mass, survival) resampling through the whole sequential gate, so the
                 band is judged at the data-dependent qualifying look (selection effect).
* ``crosscheck`` vectorized gate replica vs the pure ``run_sequential_gate`` +
                 ``plan_paired_band_check`` on fresh replicates.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.sequential_gate import (  # noqa: E402
    band_check,
    plan_paired_band_check,
    run_sequential_gate,
    sequential_gate_plan,
)
from src.scripts.eval_stats import student_t_isf  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "paired_survival_20261003.json"
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
V7 = ARTIFACTS / "apex-veto-v7-strict-20261002/run-v1/output"
V8_SCREEN = ARTIFACTS / "apex-veto-v8-screen-20261002/run-v1"
MIXES = ("frozen", "scripted", "mixed")
N_MAX = 249
MDE = 30.0
OLD_BELOW = 0.02  # fixed-N packages: candidate mean >= reference_mean - 0.02
OLD_MARGIN_Z = 1.645  # block_at_stop interim margin
CALIBRATION_WORLDS = 16
V7_DELTA_NI = 5.78794575  # v7 strict calibration.json absolute_delta_ni (mass units)
# True mean survival delta (candidate - incumbent) per mix (frozen, scripted, mixed).
SCENARIOS = {
    "no_regression": (0.0, 0.0, 0.0),
    "improves_0.05": (0.05, 0.05, 0.05),
    "one_mix_-0.02": (-0.02, 0.0, 0.0),
    "one_mix_-0.05": (-0.05, 0.0, 0.0),
    "all_mixes_-0.02": (-0.02, -0.02, -0.02),
    "all_mixes_-0.05": (-0.05, -0.05, -0.05),
    "one_mix_-0.05@scripted": (0.0, -0.05, 0.0),
    "one_mix_-0.05@mixed": (0.0, 0.0, -0.05),
}
# Scenarios the gate part skips (all-mix regressions are less adversarial than one-mix ones).
GATE_SKIP = ("one_mix_-0.02", "all_mixes_-0.02", "all_mixes_-0.05")
PAIRED_FIELDS = ("band_ni_margin", "band_alpha", "band_bound", "band_floor")
# Run-time settings; the defaults reproduce the README.  ``main`` overrides them from the
# command line for a study's own pre-registration check (see the amendment).
CONFIG = {
    "data": DATA,
    "n_max": N_MAX,
    "mde": MDE,
    "delta_ni": V7_DELTA_NI,
    # 0.5, 0.67, 1, 1.5 and 2 x MDE, and the v7 strict look-1-sized effect.
    "thetas": (15.0, 20.0, 30.0, 45.0, 60.0, 130.0),
    "check_rule": None,  # (margin, alpha, bound) of the study's frozen plan
    "plan_params": None,  # sequential_gate_plan kwargs of the study's frozen plan
}
NEW_RULES = tuple(
    (margin, alpha, bound)
    for margin in (0.02, 0.05, 0.075)
    for alpha in (0.05, 0.10)
    for bound in ("pointwise", "rci_obf")
)


def rule_name(margin: float, alpha: float, bound: str) -> str:
    return f"paired_M{margin:g}_a{alpha:g}_{bound}"


# ------------------------------------------------------------------------------- extract


def extract() -> dict:
    def load(paths):
        for path in sorted(paths):
            with open(path) as handle:
                yield json.load(handle)

    v7 = {}
    for rec in load(glob.glob(str(V7 / "final/records/*.json"))):
        v7[(rec["arm"], rec["mix"], int(rec["record"]["seed"]))] = rec["record"]
    calibration = {m: [] for m in MIXES}
    for rec in load(glob.glob(str(V7 / "calibration/records/*.json"))):
        calibration[rec["mix"]].append(rec["record"]["survival_fraction"])
    v8 = {}
    for rec in load(glob.glob(str(V8_SCREEN / "shard-*/records/*.json"))):
        if rec["arm"] in ("A", "B"):  # C/D are repeat-determinism arms of A/B
            v8[(rec["arm"], rec["mix"], int(rec["world_seed"]))] = rec["record"]

    def pool(table, inc_arm, cand_arm):
        seeds = sorted({s for a, m, s in table if m == MIXES[0] and a == inc_arm})
        out = {"world_seeds": seeds}
        for mix in MIXES:
            inc = [table[(inc_arm, mix, s)] for s in seeds]
            cand = [table[(cand_arm, mix, s)] for s in seeds]
            out[mix] = {
                "incumbent_survival": [r["survival_fraction"] for r in inc],
                "candidate_survival": [r["survival_fraction"] for r in cand],
                "mass_delta": [c["mass_integral"] - i["mass_integral"] for c, i in zip(cand, inc)],
            }
        return out

    return {
        "label": "diagnostic; derived from completed runs only (v7 strict final, v8 screen)",
        "sources": {
            "v7_strict": {
                "path": str(V7),
                "incumbent": "v5 veto (served)",
                "candidate": "v7 veto lambda=4",
                "records": len(v7),
            },
            "v8_screen": {
                "path": str(V8_SCREEN),
                "incumbent": "arm A (v7 lambda=4)",
                "candidate": "arm B (v8 lambda=8)",
                "records": len(v8),
            },
        },
        "v7_strict": pool(v7, "incumbent", "candidate"),
        "v7_calibration_incumbent_survival": calibration,
        "v8_screen": pool(v8, "A", "B"),
    }


# ------------------------------------------------------------------------- shared helpers


def load_data() -> dict:
    with open(CONFIG["data"]) as handle:
        return json.load(handle)


def sources(data: dict) -> list:
    """Pools in the data file: every top-level entry with ``world_seeds`` (fails closed)."""
    keys = [k for k in sorted(data) if isinstance(data[k], dict) and "world_seeds" in data[k]]
    for key in keys:
        missing = [m for m in MIXES if m not in data[key]]
        if missing:
            raise SystemExit(f"pool {key!r} lacks mixes {missing}; this script needs {MIXES}")
    if not keys:
        raise SystemExit("no pool with world_seeds in the data file")
    return keys


def scenarios() -> dict:
    out = dict(SCENARIOS)
    if CONFIG["check_rule"] is not None:
        margin = CONFIG["check_rule"][0]
        for i, mix in enumerate(MIXES):
            out[f"one_mix_at_margin@{mix}"] = tuple(-margin if j == i else 0.0 for j in range(3))
    return out


def regressed_mix(shifts) -> str | None:
    """The single regressed mix of a one-mix scenario, else None."""
    negative = [m for m, d in zip(MIXES, shifts) if d < 0]
    return negative[0] if len(negative) == 1 and all(d <= 0 for d in shifts) else None


def rules() -> tuple:
    extra = CONFIG["check_rule"]
    return NEW_RULES + ((extra,) if extra is not None and extra not in NEW_RULES else ())


def arrays(data: dict, source: str):
    """Per mix: incumbent survival, centred paired survival delta, centred mass delta."""
    out = {}
    for mix in MIXES:
        rec = data[source][mix]
        inc = np.asarray(rec["incumbent_survival"])
        d = np.asarray(rec["candidate_survival"]) - inc
        m = np.asarray(rec["mass_delta"])
        out[mix] = (inc, d - d.mean(), m - m.mean())
    return out


def t_crit(p: float, n: int) -> float:
    return student_t_isf(p, n - 1)


def prefix_stats(x: np.ndarray, n: int):
    head = x[:, :n]
    mean = head.mean(axis=1)
    sd = head.std(axis=1, ddof=1)
    return mean, sd


def old_rule(cand_mean, cand_sd, ref, n, interim):
    margin = OLD_MARGIN_Z * cand_sd * max(0.0, 1 / math.sqrt(n) - 1 / math.sqrt(CONFIG["n_max"]))
    margin = margin if interim else 0.0
    return cand_mean >= ref - OLD_BELOW + margin


def new_rule(d_mean, d_sd, n, margin, p):
    return d_mean - t_crit(p, n) * d_sd / math.sqrt(n) > -margin


def base_kwargs() -> dict:
    """Efficacy/NI/futility/look settings: the study's frozen plan, else the defaults."""
    params = CONFIG["plan_params"]
    if params is None:
        return {"n_max": CONFIG["n_max"], "mde": CONFIG["mde"]}
    return {k: v for k, v in params.items() if k not in PAIRED_FIELDS + ("band_policy",)}


def plans():
    kwargs = base_kwargs()
    base = sequential_gate_plan(**kwargs)
    new = {
        rule_name(*r): sequential_gate_plan(
            **kwargs,
            band_policy="paired_ni_at_stop",
            band_ni_margin=r[0],
            band_alpha=r[1],
            band_bound=r[2],
        )
        for r in rules()
    }
    return base, new


def wilson(k: int, n: int) -> list:
    if n == 0:
        return [None, None]
    z, p = 1.959964, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


# --------------------------------------------------------------------------------- bands


def part_bands(reps: int, seed: int) -> dict:
    data = load_data()
    base, new_plans = plans()
    looks = base.look_sizes
    out = {"looks": list(looks), "reps": reps, "results": {}}
    for s_index, source in enumerate(sources(data)):
        pools = arrays(data, source)
        n_worlds = len(data[source]["world_seeds"])
        for d_index, (label, shifts) in enumerate(scenarios().items()):
            rng = np.random.default_rng([seed, s_index, d_index])
            idx = rng.integers(
                0, n_worlds, size=(reps, CONFIG["n_max"])
            )  # shared seeds: joint draw
            cal = rng.integers(0, n_worlds, size=(reps, CALIBRATION_WORLDS))
            passes = {}  # rule -> look -> per-mix bool arrays
            for mix, delta in zip(MIXES, shifts):
                inc, d, _ = pools[mix]
                inc_draw = inc[idx]
                cand_draw = inc_draw + d[idx] + delta
                d_draw = d[idx] + delta
                ref = inc[cal].mean(axis=1)
                for k, n in enumerate(looks):
                    c_mean, c_sd = prefix_stats(cand_draw, n)
                    d_mean, d_sd = prefix_stats(d_draw, n)
                    res = {
                        "old_point": old_rule(c_mean, c_sd, ref, n, interim=False),
                        "old_block_at_stop": old_rule(c_mean, c_sd, ref, n, interim=True),
                    }
                    for name, plan in new_plans.items():
                        res[name] = new_rule(
                            d_mean, d_sd, n, plan.band_ni_margin, plan.band_nominal_p[k]
                        )
                    for name, value in res.items():
                        passes.setdefault(name, {}).setdefault(n, {})[mix] = value
            table = {}
            for name, by_look in passes.items():
                table[name] = {}
                for n, by_mix in by_look.items():
                    per_mix = {m: round(float(v.mean()), 4) for m, v in by_mix.items()}
                    every = np.logical_and.reduce([by_mix[m] for m in MIXES])
                    per_mix["all3"] = round(float(every.mean()), 4)
                    table[name][str(n)] = per_mix
            out["results"][f"{source}|{label}"] = table
    return out


# ---------------------------------------------------------------------------------- gate


def vector_gate(plan, mass: dict, n_reps: int, delta_ni: float):
    """Vectorized replica of run_sequential_gate (futility honored, z approximated by t).

    Returns (stop_look, qualifies) arrays; stop_look is the qualifying look when qualifies.
    """
    looks = plan.look_sizes
    last = plan.n_looks - 1
    crossed = {m: np.zeros(n_reps, bool) for m in MIXES}
    ni = np.zeros(n_reps, bool)
    done = np.zeros(n_reps, bool)
    stop_look = np.full(n_reps, last)
    qualifies = np.zeros(n_reps, bool)
    for k, n in enumerate(looks):
        futile = np.zeros(n_reps, int)
        for mix in MIXES:
            mean, sd = prefix_stats(mass[mix], n)
            se = sd / math.sqrt(n)
            t = mean / se
            crossed[mix] |= t >= t_crit(plan.efficacy_nominal_p[k], n)
            if mix == plan.scripted_mix:
                ni |= mean - t_crit(plan.ni_nominal_p[k], n) * se > -delta_ni
            if k < last:
                drift = plan.mde * math.sqrt(plan.n_max) / sd
                frac, rem = plan.fractions[k], 1.0 - plan.fractions[k]
                centre = t * math.sqrt(frac) + drift * rem
                z = (centre - plan.efficacy_boundaries[last]) / math.sqrt(rem)
                cp = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
                futile += (~crossed[mix]) & (cp < plan.futility_cp)
        successes = sum(crossed[m].astype(int) for m in MIXES)
        q_now = (successes >= plan.required_successes) & ni & ~done
        qualifies |= q_now
        stop_look[q_now] = k
        done |= q_now
        if k < last:
            fut_now = ((len(MIXES) - futile) < plan.required_successes) & ~done
            stop_look[fut_now] = k
            done |= fut_now
    return stop_look, qualifies


def gate_bands_at(plan_new, k_arr, surv: dict, inc: dict, ref: dict, looks, name):
    """Band verdicts per replicate at look k_arr[r] for one rule (old or a new plan).

    Returns (all bands pass, {mix: that band passes}).
    """
    ok = np.ones(len(k_arr), bool)
    per_mix = {m: np.ones(len(k_arr), bool) for m in MIXES}
    for k, n in enumerate(looks):
        rows = k_arr == k
        if not rows.any():
            continue
        for mix in MIXES:
            if name.startswith("old"):
                cand = inc[mix][rows, :n] + surv[mix][rows, :n]
                c_mean, c_sd = cand.mean(axis=1), cand.std(axis=1, ddof=1)
                verdict = old_rule(c_mean, c_sd, ref[mix][rows], n, interim=True)
            else:
                d = surv[mix][rows, :n]
                verdict = new_rule(
                    d.mean(axis=1),
                    d.std(axis=1, ddof=1),
                    n,
                    plan_new.band_ni_margin,
                    plan_new.band_nominal_p[k],
                )
            ok[rows] &= verdict
            per_mix[mix][rows] = verdict
    return ok, per_mix


def draw_gate(pools, rng, n_reps, n_worlds, theta, shifts):
    idx = rng.integers(0, n_worlds, size=(n_reps, CONFIG["n_max"]))
    cal = rng.integers(0, n_worlds, size=(n_reps, CALIBRATION_WORLDS))
    mass, surv, inc, ref = {}, {}, {}, {}
    for mix, delta in zip(MIXES, shifts):
        inc_pool, d, m = pools[mix]
        mass[mix] = m[idx] + theta
        surv[mix] = d[idx] + delta
        inc[mix] = inc_pool[idx]
        ref[mix] = inc_pool[cal].mean(axis=1)
    return mass, surv, inc, ref


def part_gate(reps: int, seed: int) -> dict:
    data = load_data()
    base, new_plans = plans()
    delta_ni = CONFIG["delta_ni"]
    rule_names = ["old_block_at_stop"] + list(new_plans)
    out = {"reps": reps, "delta_ni": delta_ni, "results": {}}
    for s_index, source in enumerate(sources(data)):
        pools = arrays(data, source)
        n_worlds = len(data[source]["world_seeds"])
        for t_index, theta in enumerate(CONFIG["thetas"]):
            for d_index, (label, shifts) in enumerate(scenarios().items()):
                if label in GATE_SKIP:
                    continue
                rng = np.random.default_rng([seed, s_index, t_index, d_index])
                mass, surv, inc, ref = draw_gate(pools, rng, reps, n_worlds, theta, shifts)
                stop, qual = vector_gate(base, mass, reps, delta_ni)
                row = {
                    "p_qualify": round(float(qual.mean()), 4),
                    "stop_look_dist_given_qualify": [
                        round(float((stop[qual] == k).mean()), 4) if qual.any() else None
                        for k in range(base.n_looks)
                    ],
                }
                for name in rule_names:
                    plan = None if name.startswith("old") else new_plans[name]
                    ok, per_mix = gate_bands_at(plan, stop, surv, inc, ref, base.look_sizes, name)
                    k, q = int((ok & qual).sum()), int(qual.sum())
                    row[name] = {
                        "p_band_given_qualify": round(k / q, 4) if q else None,
                        "ci": wilson(k, q),
                        "p_pass": round(k / reps, 4),
                    }
                    bad = regressed_mix(shifts)
                    if bad is not None:
                        # Joint: P(qualify and the regressed band passes), an upper bound on
                        # P(gate PASS | regression).  Conditional: given that the run qualified.
                        kb = int((per_mix[bad] & qual).sum())
                        row[name].update(
                            regressed_mix=bad,
                            p_regressed_band_and_qualify=round(kb / reps, 4),
                            ci_joint=wilson(kb, reps),
                            p_regressed_band_given_qualify=round(kb / q, 4) if q else None,
                            ci_conditional=wilson(kb, q),
                        )
                out["results"][f"{source}|theta={theta:g}|{label}"] = row
    if CONFIG["check_rule"] is not None:
        out["check"] = pre_registration_check(out["results"])
    return out


def pre_registration_check(results: dict) -> dict:
    """Amendment acceptance: joint regressed-band error <= 1.2 x band_alpha.

    Scenarios ``one_mix_at_margin@<mix>`` (one mix regresses by exactly the margin M), every
    pool and every theta.  The judged rate is the joint P(run qualifies and the regressed band
    passes), which bounds P(gate PASS with that regression).  The conditional rate given
    qualification is reported but not judged: it inflates where qualifying is rare.
    """
    margin, alpha, bound = CONFIG["check_rule"]
    name = rule_name(margin, alpha, bound)
    joint, conditional = {}, {}
    for key, row in results.items():
        if "|one_mix_at_margin@" in key:
            joint[key] = row[name]["p_regressed_band_and_qualify"]
            conditional[key] = row[name]["p_regressed_band_given_qualify"]
    threshold = 1.2 * alpha
    return {
        "rule": name,
        "threshold": threshold,
        "joint_rates": joint,
        "conditional_rates_reported_only": conditional,
        "max_joint_rate": max(joint.values()) if joint else None,
        "passes": bool(joint) and max(joint.values()) <= threshold,
    }


# ---------------------------------------------------------------------------- crosscheck


def part_crosscheck(reps: int, seed: int) -> dict:
    data = load_data()
    base, new_plans = plans()
    name = rule_name(0.05, 0.05, "pointwise")  # the recommended rule
    plan = new_plans[name]
    pools = arrays(data, "v7_strict")
    n_worlds = len(data["v7_strict"]["world_seeds"])
    mismatches, checked = [], 0
    cases = (
        (30.0, "one_mix_-0.05"),
        (60.0, "no_regression"),
        (30.0, "no_regression"),
        (15.0, "one_mix_-0.05@scripted"),
    )
    for s_index, (theta, label) in enumerate(cases):
        rng = np.random.default_rng([seed, 99, s_index])
        mass, surv, inc, ref = draw_gate(pools, rng, reps, n_worlds, theta, SCENARIOS[label])
        stop, qual = vector_gate(plan, mass, reps, CONFIG["delta_ni"])
        ok, _ = gate_bands_at(plan, stop, surv, inc, ref, plan.look_sizes, name)
        for r in range(reps):
            deltas = {m: mass[m][r].tolist() for m in MIXES}
            bands = []
            for k, n in enumerate(plan.look_sizes):
                verdicts = [
                    plan_paired_band_check(
                        plan,
                        k,
                        (inc[m][r, :n] + surv[m][r, :n]).tolist(),
                        inc[m][r, :n].tolist(),
                    )["passes"]
                    for m in MIXES
                ]
                bands.append(all(verdicts))
            exact = run_sequential_gate(plan, deltas, CONFIG["delta_ni"], bands)
            mine = (
                ("STOP_PASS" if ok[r] else "STOP_FAIL_BANDS")
                if qual[r] and stop[r] < plan.n_looks - 1
                else ("FINAL_PASS" if qual[r] and ok[r] else None)
            )
            exact_look = exact["look"]
            same = exact_look == int(stop[r]) and (
                mine == exact["decision"]
                or (mine is None and exact["decision"] in ("STOP_FUTILE", "FINAL_FAIL"))
            )
            checked += 1
            if not same:
                mismatches.append(
                    {
                        "scenario": [theta, label],
                        "rep": r,
                        "vector": [int(stop[r]), mine],
                        "exact": [exact_look, exact["decision"]],
                    }
                )
    # The old rule's band function against the module's band_check on a few prefixes.
    rng = np.random.default_rng([seed, 7])
    old_mismatch = 0
    for _ in range(200):
        n = int(rng.choice(base.look_sizes))
        values = rng.uniform(0, 1, n)
        ref = float(rng.uniform(0.5, 0.7))
        module = band_check(values.tolist(), ref - OLD_BELOW, ref + 1.0, n_final=CONFIG["n_max"])[
            "passes"
        ]
        mine = bool(old_rule(values.mean(), values.std(ddof=1), ref, n, interim=True))
        old_mismatch += module != mine
    return {
        "rule": name,
        "replicates": checked,
        "mismatches": len(mismatches),
        "examples": mismatches[:10],
        "old_rule_vs_band_check_mismatches": old_mismatch,
    }


def load_plan_params(path: Path) -> None:
    """Freeze CONFIG to the study's plan; fail closed on any mismatch."""
    with open(path) as handle:
        raw = json.load(handle)
    params = dict(raw.get("plan_parameters", raw))
    plan = sequential_gate_plan(**params)
    if "plan" in raw and plan.as_dict() != raw["plan"]:
        raise SystemExit("rebuilt plan does not equal the frozen plan in --plan-params")
    if tuple(plan.mixes) != MIXES:
        raise SystemExit(f"this script simulates the mixes {MIXES}; plan has {plan.mixes}")
    CONFIG.update(plan_params=params, n_max=plan.n_max, mde=plan.mde)
    if plan.band_policy == "paired_ni_at_stop":
        CONFIG["check_rule"] = (plan.band_ni_margin, plan.band_alpha, plan.band_bound)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", choices=("extract", "bands", "gate", "crosscheck"), required=True)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DATA, help="pooled paired records (JSON)")
    parser.add_argument("--n-max", type=int, default=N_MAX)
    parser.add_argument("--mde", type=float, default=MDE)
    parser.add_argument("--delta-ni", type=float, default=V7_DELTA_NI)
    parser.add_argument(
        "--thetas", default="15,20,30,45,60,130", help="mass effects for --part gate"
    )
    parser.add_argument(
        "--check-rule",
        default=None,
        help="margin,alpha,bound of the study's frozen plan, e.g. 0.05,0.05,pointwise",
    )
    parser.add_argument(
        "--plan-params",
        type=Path,
        default=None,
        help="JSON: sequential_gate_plan kwargs, or {'plan_parameters': ..., 'plan': ...} "
        "(intent.json style; the rebuilt plan must equal 'plan'). Overrides --n-max/--mde "
        "and, under band_policy paired_ni_at_stop, --check-rule.",
    )
    args = parser.parse_args()
    CONFIG.update(
        data=args.data,
        n_max=args.n_max,
        mde=args.mde,
        delta_ni=args.delta_ni,
        thetas=tuple(float(t) for t in args.thetas.split(",")),
    )
    if args.check_rule:
        margin, alpha, bound = args.check_rule.split(",")
        CONFIG["check_rule"] = (float(margin), float(alpha), bound)
    if args.plan_params:
        load_plan_params(args.plan_params)
    start = time.process_time()
    if args.part == "extract":
        result = extract()
        with open(DATA, "w") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
        print(json.dumps({"wrote": str(DATA), "sha256": digest}))
        return
    reps = args.reps or {"bands": 20000, "gate": 10000, "crosscheck": 75}[args.part]
    result = {"bands": part_bands, "gate": part_gate, "crosscheck": part_crosscheck}[args.part](
        reps, args.seed
    )
    result["data_sha256"] = hashlib.sha256(Path(CONFIG["data"]).read_bytes()).hexdigest()
    result["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in CONFIG.items()}
    result["cpu_seconds"] = round(time.process_time() - start, 1)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
