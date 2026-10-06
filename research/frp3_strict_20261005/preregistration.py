#!/usr/bin/env python3
"""Pre-registration inputs of the FRP-v3 strict gate (deterministic; numpy; no game played).

Reads the closed FRP-v3 Phase R (``frp-v3-20261005/phaseR/rp-v1``; merged ``summary.json``
pinned by sha256, decision GO_R) and writes, beside this file:

* ``phase_r_deltas.json``: the candidate's OWN paired deltas (training seed 12, M3@60000 minus
  champion+v8, exact H5000 prefix ``mass_integral``, 32 worlds per mix in world order),
  recomputed from the raw records and cross-checked against the summary's per-seed means. This
  is the skew-check input (Tier-0 data) frozen by sha256 in the intent.
* ``paired_pool.json``: the paired-band check pool (``simulate.py --data`` format): pool
  ``frp3_s12`` (the candidate's own 32 worlds per mix) and pool ``frp3_all_seeds`` (all five
  Phase R seeds, 160 worlds per mix, a sensitivity pool of the same recipe); survival is the
  exact H5000-prefix ``survival_fraction`` of both arms, ``mass_delta`` the H5000 delta.
* ``frozen_plan.json``: ``{"plan_parameters", "plan"}`` exactly as ``intent.json`` will freeze
  them (``sequential_runner.frozen_plan_document``; the ``simulate.py --plan-params`` input).
* ``operating_characteristics.json``: the sizing rule and its result, the frozen plan, a normal
  Monte Carlo of the plan (``research/sequential_gate_validation_20261002/simulate.py``'s
  ``Replica`` reused with this study's N_max, per-mix SDs and development delta_NI), a joint
  bootstrap of the candidate's Phase R worlds through the whole gate WITH survival band v2
  (``pooled_ni_continue``: judged at the qualifying look and every later look, a qualified run
  whose bands fail continues; cross-checked decision by decision against
  ``run_sequential_gate`` + ``plan_pooled_band_check``), sensitivity sizing, the Mac-fallback
  runtime projection and a planning-only pre-run of the resampling skew check.
* ``look1_band_cost.json``: survival band v2's cost per look (pilot SDs of the paired and the
  pooled survival delta): the chance a no-regression candidate fails the band at a look, and
  the chance it has not passed by the final look when it qualifies at a given look.

The band rule and its margins come from the survival band v2 amendment, which chose and
calibrated them on independent pre-FRP-v3 data only. The Phase R pool below is used, as the
template requires, for this study's validity check and for planning (descriptive).

Usage: ``python research/frp3_strict_20261005/preregistration.py`` (single-threaded; about
10-20 min of one core: hold one shared CPU slot). Nothing here reads strict data.
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
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import NormalDist  # noqa: E402
from typing import Any, Dict, List, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sequential_gate_validation_20261002 import simulate as sim  # noqa: E402
from research.sequential_gate_validation_20261002 import skew_probe  # noqa: E402
from research.sequential_strict_template import sequential_runner as R  # noqa: E402
from src.evaluation.sequential_gate import plan_pooled_band_check, run_sequential_gate  # noqa: E402
from src.scripts.eval_stats import student_t_isf  # noqa: E402

HERE = Path(__file__).resolve().parent
MIXES = ("frozen", "scripted", "mixed")
SCRIPTED = MIXES.index("scripted")
PHASE_R = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/frp-v3-20261005/phaseR/rp-v1")
PHASE_R_SUMMARY_SHA256 = "0fcb80f689f47f70877ac4c9f6b87c2a76bafb0e21bf270c1aecc7f27b3ab206"
CANDIDATE_SEED = 12
PHASE_R_SEEDS = (10, 11, 12, 13, 14)
WORLDS_PER_SEED = 32
CANDIDATE_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
VETO_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"
DELTAS_OUT = HERE / "phase_r_deltas.json"
POOL_OUT = HERE / "paired_pool.json"
PLAN_OUT = HERE / "frozen_plan.json"
OC_OUT = HERE / "operating_characteristics.json"
BAND_OUT = HERE / "look1_band_cost.json"

# ---- sizing rule (stated before any strict data exists; the owner's rule) ------------------
FAMILY_ALPHA = 0.05
PER_MIX_ALPHA = FAMILY_ALPHA / len(MIXES)
SIZING_POWER = 0.90
SEQUENTIAL_INFLATION = 1.02  # amendment: ~2% for four OBF looks (checked by the MC below)
MDE_STEP = 5.0
MDE_SEARCH_MAX = 200.0
N_MAX_LIMIT = R.N_MAX_CAP  # 300
SIZING_RULE = (
    "per-mix SD = sample SD of the candidate's own FRP-v3 Phase R paired H5000 deltas (training "
    "seed 12, M3@60000 minus champion+v8, 32 worlds per mix); sd_max = max over mixes; "
    "n_fixed(MDE) = smallest n with n >= ((t_{1-0.05/3, n-1} + t_{0.90, n-1}) * sd_max / "
    "MDE)^2 (one-sided per-mix Bonferroni alpha, 90% per-mix power at the MDE in every mix); "
    "N_max(MDE) = ceil(1.02 * n_fixed) (amendment: four OBF looks); MDE = the smallest "
    "multiple of 5 with N_max <= 300. The Phase R mean deltas are not used for sizing."
)

# ---- study constants -----------------------------------------------------------------------
NI_FRACTION = 0.03
N_CALIBRATION = 16
BAND_MARGIN_Z = 1.645
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
FUTILITY_CP = 0.10
# Survival band v2 (amendment 2026-10-06) option 1: pooled 0.05, per-mix 0.075, alpha 0.05
# rci_obf, floor 0.30 (see spec.PAIRED_BAND)
PAIRED_BAND = dict(R.POOLED_BAND_OPTIONS["1"])
# Reference-bank SDs (v8 - v7, 150 worlds/mix, rp-bank-long), quoted for sensitivity only.
REFERENCE_BANK_SDS = {"frozen": 124.0, "scripted": 141.0, "mixed": 97.0}
# Mac-fallback runtime basis (2 slots, live H5000): the v8 strict run's champion+v8 arm.
V8_STRICT_RECORDS = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v8-strict-20261003/run-v1/"
    "output/final/records"
)
CANDIDATE_TIME_FACTOR = 1.25  # Phase R: the candidate's v8 work per episode ~1.2x (bigger body)
PROJECTION_OVERHEAD = 1.15

_NORMAL = NormalDist()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any, compact: bool = False) -> None:
    if compact:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    else:
        text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")


# ---------------------------------------------------------------- Phase R data


def load_phase_r(root: Path = PHASE_R) -> Dict[str, Any]:
    """Per training seed and mix, world-ordered paired records (candidate vs incumbent)."""
    summary_path = root / "merged" / "summary.json"
    actual = sha256_file(summary_path)
    if actual != PHASE_R_SUMMARY_SHA256:
        raise SystemExit(f"Phase R summary sha256 {actual} is not the pinned one")
    summary = json.loads(summary_path.read_text())
    decision = summary["decision"]
    if decision["status"] != "GO_R" or summary["smoke"] or not summary["tier1_complete"]:
        raise SystemExit("Phase R is not a complete, real GO_R")
    if not summary["prefix_identity"]["pass"]:
        raise SystemExit("Phase R prefix identity did not pass")
    selection = summary["analysis"]["candidate_selection"]
    if selection["seed"] != CANDIDATE_SEED or selection["checkpoint"]["sha256"] != CANDIDATE_SHA256:
        raise SystemExit("Phase R's pre-declared candidate is not seed 12 M3@60000")
    per_seed_saved = {
        row["seed"]: row["per_mix_mean"]
        for row in summary["analysis"]["cells"]["M3@60000"]["h5000_prefix"]["per_seed"]
    }
    records = root / "shard-rp" / "records"
    out: Dict[str, Any] = {"summary_sha256": actual, "seeds": {}}
    for seed in PHASE_R_SEEDS:
        per_mix: Dict[str, Any] = {}
        for mix in MIXES:
            arms: Dict[str, Dict[int, Mapping[str, Any]]] = {"candidate": {}, "incumbent": {}}
            for arm, pattern, sha in (
                ("candidate", f"M3-u60000-s{seed}-{mix}-*.json", None),
                ("incumbent", f"incumbent-s{seed}-{mix}-*.json", CHAMPION_SHA256),
            ):
                for path in sorted(records.glob(pattern)):
                    entry = json.loads(path.read_text())
                    if entry["control"]:
                        continue
                    if arm == "candidate" and seed == CANDIDATE_SEED:
                        sha = CANDIDATE_SHA256
                    if sha is not None and entry["hero_sha256"] != sha:
                        raise SystemExit(f"{path.name}: hero {entry['hero_sha256'][:12]}")
                    method = entry["record"]["probes"]["safety_veto"]["method"]
                    if method != VETO_METHOD:
                        raise SystemExit(f"{path.name}: veto {method}")
                    index = int(entry["world_index"])
                    if index in arms[arm]:
                        raise SystemExit(f"duplicate Phase R record {path.name}")
                    arms[arm][index] = entry
            if sorted(arms["candidate"]) != list(range(WORLDS_PER_SEED)) or sorted(
                arms["incumbent"]
            ) != list(range(WORLDS_PER_SEED)):
                raise SystemExit(f"seed {seed} {mix}: not exactly {WORLDS_PER_SEED} paired worlds")
            cand = [arms["candidate"][w] for w in range(WORLDS_PER_SEED)]
            inc = [arms["incumbent"][w] for w in range(WORLDS_PER_SEED)]
            for c, i in zip(cand, inc):
                if c["world_seed"] != i["world_seed"]:
                    raise SystemExit(f"seed {seed} {mix}: arms on different worlds")
            deltas = [
                c["prefix_h5000"]["mass_integral"] - i["prefix_h5000"]["mass_integral"]
                for c, i in zip(cand, inc)
            ]
            saved = per_seed_saved[seed][mix]
            if abs(float(np.mean(deltas)) - saved) > 1e-6:
                raise SystemExit(f"seed {seed} {mix}: raw-record mean differs from summary.json")
            per_mix[mix] = {
                "deltas": deltas,
                "world_seeds": [int(c["world_seed"]) for c in cand],
                "candidate_survival": [c["prefix_h5000"]["survival_fraction"] for c in cand],
                "incumbent_survival": [i["prefix_h5000"]["survival_fraction"] for i in inc],
                "incumbent_mass": [i["prefix_h5000"]["mass_integral"] for i in inc],
            }
        worlds = [per_mix[m]["world_seeds"] for m in MIXES]
        if not all(w == worlds[0] for w in worlds):
            raise SystemExit(f"seed {seed}: mixes do not share worlds by index")
        out["seeds"][seed] = per_mix
    return out


def pooled(phase_r: Mapping[str, Any], seeds: Sequence[int], key: str, mix: str) -> List[float]:
    return [v for s in seeds for v in phase_r["seeds"][s][mix][key]]


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


def pool_document(phase_r: Mapping[str, Any]) -> Dict[str, Any]:
    def pool(seeds: Sequence[int]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "world_seeds": pooled(phase_r, seeds, "world_seeds", "frozen"),
        }
        for mix in MIXES:
            out[mix] = {
                "candidate_survival": pooled(phase_r, seeds, "candidate_survival", mix),
                "incumbent_survival": pooled(phase_r, seeds, "incumbent_survival", mix),
                "mass_delta": pooled(phase_r, seeds, "deltas", mix),
            }
        return out

    return {
        "label": "Tier-0 development data: FRP-v3 Phase R paired records (exact H5000 prefixes)",
        "sources": {
            "phase_r": str(PHASE_R),
            "summary_sha256": phase_r["summary_sha256"],
            "frp3_s12": "the candidate's own worlds (training seed 12, M3@60000 vs champion+v8)",
            "frp3_all_seeds": "all five Phase R seeds (10-14) of the same recipe vs "
            "champion+v8 (sensitivity pool)",
        },
        "frp3_s12": pool((CANDIDATE_SEED,)),
        "frp3_all_seeds": pool(PHASE_R_SEEDS),
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


def size_for(sds: Mapping[str, float]) -> Dict[str, Any]:
    sd_max = max(sds.values())
    table, chosen, mde = [], None, MDE_STEP
    while mde <= MDE_SEARCH_MAX:
        nf = n_fixed(sd_max, mde)
        n_max = math.ceil(SEQUENTIAL_INFLATION * nf - 1e-9)
        feasible = n_max <= N_MAX_LIMIT
        table.append({"mde": mde, "n_fixed_90": nf, "n_max": n_max, "feasible": feasible})
        if feasible and chosen is None:
            chosen = {"mde": mde, "n_fixed_90": nf, "n_max": n_max}
        mde += MDE_STEP
    if chosen is None:
        raise SystemExit("no MDE <= 200 is feasible: STOP_INFEASIBLE")
    return {
        "per_mix_delta_sd": dict(sds),
        "sd_max": sd_max,
        "binding_mix": max(sds, key=sds.get),
        "table": [r for r in table if chosen["mde"] - 3 * MDE_STEP <= r["mde"]][:7],
        "chosen": chosen,
    }


def sizing(phase_r: Mapping[str, Any]) -> Dict[str, Any]:
    own = {m: describe(phase_r["seeds"][CANDIDATE_SEED][m]["deltas"])["sd"] for m in MIXES}
    all_seeds = {m: describe(pooled(phase_r, PHASE_R_SEEDS, "deltas", m))["sd"] for m in MIXES}
    binding = size_for(own)
    return {
        "rule": SIZING_RULE,
        "per_mix_alpha": PER_MIX_ALPHA,
        "power": SIZING_POWER,
        "sequential_inflation": SEQUENTIAL_INFLATION,
        "basis": "candidate's own Phase R paired SDs (binding)",
        **binding,
        "phase_r_effects_used": False,
        "sensitivity_not_binding": {
            "all_phase_r_seeds_sds": size_for(all_seeds),
            "v8_reference_bank_sds": size_for(REFERENCE_BANK_SDS),
        },
    }


def runtime_projection(n_max: int) -> Dict[str, Any]:
    """Mac-fallback final stage (2 slots, live H5000): champion+v8 episode times measured in
    the v8 strict run (its candidate arm), the FRP-v3 candidate at x1.25, x1.15 overhead."""
    walls: Dict[str, List[float]] = {m: [] for m in MIXES}
    for path in sorted(V8_STRICT_RECORDS.glob("final-candidate-*.json")):
        entry = json.loads(path.read_text())
        walls[entry["mix"]].append(float(entry["wall_seconds"]))
    per_mix = {m: float(np.mean(walls[m])) for m in MIXES}
    triplet = sum(per_mix[m] * (1.0 + CANDIDATE_TIME_FACTOR) for m in MIXES)
    budget = R.DEFAULT_CAPS["final"] - R.WORKER_STOP_MARGIN_SECONDS
    projected = n_max * triplet / R.WORKERS * PROJECTION_OVERHEAD
    calibration = N_CALIBRATION * sum(per_mix.values()) / R.WORKERS * PROJECTION_OVERHEAD
    return {
        "basis": "v8 strict run-v1 final records of its champion+v8 arm (Mac, 2 slots, live "
        f"H5000, {sum(len(v) for v in walls.values())} episodes); the FRP-v3 candidate at "
        f"x{CANDIDATE_TIME_FACTOR} (Phase R: about 1.2x the v8 work per episode, bigger "
        f"bodies); x{PROJECTION_OVERHEAD} overhead",
        "incumbent_episode_seconds": per_mix,
        "mean_episode_seconds_both_arms": triplet / (2 * len(MIXES)),
        "pair_seconds_per_world_triplet": triplet,
        "projected_final_seconds_per_worker_at_n_max": projected,
        "projected_calibration_seconds": calibration,
        "final_cap_budget_seconds": budget,
        "fraction_of_final_cap": projected / budget,
        "note": "disclosed, not a sizing constraint (the owner's rule is N_max <= 300): the "
        "Mac fallback at N_max projects to this fraction of the template's final-stage cap; a "
        "cap reached ends the run INCOMPLETE",
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


def normal_scenarios(mde: float, delta_ni: float, own: Sequence[float], pooled_means, rho):
    big = 4.0 * mde
    own = tuple(float(v) for v in own)
    pool = tuple(float(v) for v in pooled_means)
    return {
        "null_indep": ((0.0, 0.0, 0.0), 0.0, 200_000, 11),
        "null_phase_r_corr": ((0.0, 0.0, 0.0), rho, 200_000, 12),
        "lfc_one_effect_two_null": ((big, 0.0, 0.0), 0.0, 200_000, 13),
        "ni_null_scripted_at_margin": ((big, -delta_ni, big), 0.0, 200_000, 14),
        "delta_mde": ((mde, mde, mde), 0.0, 50_000, 21),
        "delta_mde_phase_r_corr": ((mde, mde, mde), rho, 50_000, 22),
        "candidate_own_means": (own, 0.0, 50_000, 23),
        "all_seed_means": (pool, 0.0, 50_000, 24),
        "half_all_seed_means": (tuple(v / 2 for v in pool), 0.0, 50_000, 25),
        "two_mixes_at_mde_scripted_zero": ((mde, 0.0, mde), 0.0, 50_000, 26),
    }


# ---------------------------------------------------------------- joint bootstrap (paired)


def band_ok_by_look(replica, cand, inc, plan):
    """Survival band v2 verdict per look (list of (reps,) arrays) and, per look, the failing
    component counts input: pooled ok (reps,), per-mix ok (reps, 3)."""
    d = cand - inc
    pooled = d.mean(axis=1)
    out = []
    for k, n in enumerate(replica.sizes):
        t_band = student_t_isf(plan.band_nominal_p[k], int(n) - 1)
        dk = d[:, :, :n]
        mix_lower = dk.mean(axis=2) - t_band * dk.std(axis=2, ddof=1) / math.sqrt(n)
        mix_ok = mix_lower > -plan.band_mix_margin
        if plan.band_floor is not None:
            mix_ok &= cand[:, :, :n].mean(axis=2) >= plan.band_floor
        pk = pooled[:, :n]
        pooled_ok = pk.mean(axis=1) - t_band * pk.std(axis=1, ddof=1) / math.sqrt(n) > -(
            plan.band_ni_margin
        )
        out.append((pooled_ok & np.all(mix_ok, axis=1), pooled_ok, mix_ok))
    return out


def bootstrap_decide(replica, x, cand, inc, delta_ni, plan):
    """``Replica.decide`` with a per-replicate delta_NI and survival band v2.

    x: (reps, 3, N) mass deltas; cand / inc: (reps, 3, N) survival of the arms on the same
    worlds; delta_ni: (reps,). Codes: 1 STOP_PASS, 2 STOP_FUTILE, 3 FINAL_PASS, 4 FINAL_FAIL
    (5 STOP_FAIL_BANDS cannot occur under ``pooled_ni_continue``). Qualification as in the gate;
    from the qualifying look on, the bands (pooled NI > -band_ni_margin, every mix's NI >
    -band_mix_margin, every candidate mean >= floor, at the look's rci_obf level) are judged at
    every look: STOP_PASS at the first look where they pass, FINAL_PASS / FINAL_FAIL at the
    last; a qualified run is never stopped for futility.
    """
    reps = x.shape[0]
    last = len(replica.sizes) - 1
    stats = replica.look_stats(x)
    bands = band_ok_by_look(replica, cand, inc, plan)
    crossed = np.zeros((reps, 3), dtype=bool)
    ni = np.zeros(reps, dtype=bool)
    active = np.ones(reps, dtype=bool)
    qualified = np.zeros(reps, dtype=bool)
    outcome = np.zeros(reps, dtype=np.int8)
    stop_look = np.full(reps, last, dtype=np.int8)
    qualify_look = np.full(reps, -1, dtype=np.int8)
    pooled_fail_q = 0
    band_fail_mix = np.zeros(3, dtype=np.int64)
    c_final = plan.efficacy_boundaries[last]
    for k, n in enumerate(replica.sizes):
        mean, sd, t = stats[k]
        ok, pooled_ok, mix_ok = bands[k]
        crossed |= (t >= replica.t_eff[k]) & active[:, None]
        lower = mean[:, SCRIPTED] - replica.t_ni[k] * sd[:, SCRIPTED] / math.sqrt(n)
        ni |= (lower > -delta_ni) & active
        newly = active & ~qualified & (crossed.sum(axis=1) >= plan.required_successes) & ni
        qualify_look[newly] = k
        pooled_fail_q += int((~pooled_ok[newly]).sum())
        band_fail_mix += (~mix_ok[newly]).sum(axis=0)
        qualified |= newly
        if k == last:
            outcome[active] = np.where((qualified & ok)[active], 3, 4)
            break
        stop_q = active & qualified & ok
        outcome[stop_q] = 1
        stop_look[stop_q] = k
        active &= ~stop_q
        f = plan.fractions[k]
        z = replica.t_to_z(k, t)
        theta = plan.mde * math.sqrt(plan.n_max) / sd
        arg = z * math.sqrt(f) + theta * (1.0 - f) - c_final
        futile = (~crossed) & (arg < replica.z_cut * math.sqrt(1.0 - f))
        futile_now = active & ~qualified & (3 - futile.sum(axis=1) < plan.required_successes)
        outcome[futile_now] = 2
        stop_look[futile_now] = k
        active &= ~futile_now
    return {
        "outcome": outcome,
        "stop_look": stop_look,
        "qualify_look": qualify_look,
        "crossed": crossed,
        "pooled_band_fail_at_qualifying_look": pooled_fail_q,
        "band_fail_by_mix": band_fail_mix,
    }


def bootstrap_arrays(
    phase_r, seeds, rng, reps: int, n_max: int, shift=None, delta_ni=None, survival=None
):
    """Joint world resampling (same world for every mix), plus a 16-world calibration
    delta_NI per replicate (0.03 x resampled incumbent scripted mean) unless fixed.
    ``survival`` (per-mix true survival deltas) recentres the candidate's survival on the
    incumbent's (descriptive what-if; ``None`` keeps the observed deltas)."""
    deltas = np.asarray([pooled(phase_r, seeds, "deltas", m) for m in MIXES], dtype=float)
    if shift is not None:
        deltas = deltas - deltas.mean(axis=1, keepdims=True) + np.asarray(shift)[:, None]
    cand = np.asarray([pooled(phase_r, seeds, "candidate_survival", m) for m in MIXES])
    inc = np.asarray([pooled(phase_r, seeds, "incumbent_survival", m) for m in MIXES])
    if survival is not None:
        d = cand - inc
        cand = inc + d - d.mean(axis=1, keepdims=True) + np.asarray(survival)[:, None]
    mass_inc = np.asarray(pooled(phase_r, PHASE_R_SEEDS, "incumbent_mass", "scripted"))
    worlds = deltas.shape[1]
    idx = rng.integers(0, worlds, size=(reps, n_max))
    x = deltas[:, idx].transpose(1, 0, 2)
    c = cand[:, idx].transpose(1, 0, 2).astype(float)
    i = inc[:, idx].transpose(1, 0, 2).astype(float)
    if delta_ni is None:
        cidx = rng.integers(0, mass_inc.size, size=(reps, N_CALIBRATION))
        dni = NI_FRACTION * mass_inc[cidx].mean(axis=1)
    else:
        dni = np.full(reps, float(delta_ni))
    return x, c, i, dni


CODES = {1: "STOP_PASS", 2: "STOP_FUTILE", 3: "FINAL_PASS", 4: "FINAL_FAIL", 5: "STOP_FAIL_BANDS"}


def bootstrap_power(
    phase_r, seeds, plan, replica, reps, seed, shift=None, survival=None
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    counts = {name: 0 for name in CODES.values()}
    looks = np.zeros(len(plan.look_sizes), dtype=np.int64)
    qlooks = np.zeros(len(plan.look_sizes) + 1, dtype=np.int64)
    per_mix_cross = np.zeros(3, dtype=np.int64)
    band_fail = np.zeros(3, dtype=np.int64)
    pooled_fail = 0
    worlds = done = 0
    while done < reps:
        k = min(5_000, reps - done)
        x, c, i, dni = bootstrap_arrays(
            phase_r, seeds, rng, k, plan.n_max, shift, survival=survival
        )
        res = bootstrap_decide(replica, x, c, i, dni, plan)
        for code, name in CODES.items():
            counts[name] += int((res["outcome"] == code).sum())
        looks += np.bincount(res["stop_look"], minlength=len(plan.look_sizes))
        qlooks += np.bincount(res["qualify_look"] + 1, minlength=len(plan.look_sizes) + 1)
        worlds += int(np.asarray(plan.look_sizes)[res["stop_look"]].sum())
        per_mix_cross += res["crossed"].sum(axis=0)
        band_fail += res["band_fail_by_mix"]
        pooled_fail += res["pooled_band_fail_at_qualifying_look"]
        done += k
    passes = counts["STOP_PASS"] + counts["FINAL_PASS"]
    return {
        "replicates": reps,
        "seed": seed,
        "pool_seeds": list(seeds),
        "shifted_means": None if shift is None else list(shift),
        "survival_deltas": "observed" if survival is None else list(survival),
        "pass_probability": sim.wilson(passes, reps),
        "outcomes": {name: counts[name] / reps for name in CODES.values()},
        "stop_look_distribution": (looks / reps).tolist(),
        "qualifying_look_distribution": {
            "never": qlooks[0] / reps,
            **{f"look_{j + 1}": qlooks[j + 1] / reps for j in range(len(plan.look_sizes))},
        },
        "expected_worlds_per_mix": worlds / reps,
        "per_mix_crossing_by_stop": dict(zip(MIXES, (per_mix_cross / reps).tolist())),
        "band_failures_at_qualifying_look": {
            "pooled": pooled_fail / reps,
            **{f"mix_{m}": v for m, v in zip(MIXES, (band_fail / reps).tolist())},
        },
    }


def _crosscheck_batch(phase_r, plan, replica, reps: int, seed: int, shift) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    x, c, i, dni = bootstrap_arrays(phase_r, (CANDIDATE_SEED,), rng, reps, plan.n_max, shift=shift)
    res = bootstrap_decide(replica, x, c, i, dni, plan)
    names = {v: k for k, v in CODES.items()}
    mismatches = []
    for r in range(reps):
        bands = []
        for look, n in enumerate(plan.look_sizes):
            check = plan_pooled_band_check(
                plan,
                look,
                {m: c[r, j, :n].tolist() for j, m in enumerate(MIXES)},
                {m: i[r, j, :n].tolist() for j, m in enumerate(MIXES)},
            )
            bands.append(bool(check["passes"]))
        pure = run_sequential_gate(
            plan, {m: x[r, j].tolist() for j, m in enumerate(MIXES)}, float(dni[r]), bands
        )
        if (names[pure["decision"]], pure["look"]) != (
            int(res["outcome"][r]),
            int(res["stop_look"][r]),
        ):
            mismatches.append(r)
    counts = {k: int((res["outcome"] == v).sum()) for k, v in names.items()}
    return {
        "replicates": reps,
        "seed": seed,
        "shifted_means": list(shift),
        "mismatches": mismatches,
        "outcome_counts": counts,
    }


def crosscheck_bootstrap(phase_r, plan, replica, reps: int = 300, seed: int = 4242) -> Dict:
    """The bootstrap replica vs the pure ``run_sequential_gate`` with per-look survival band v2
    verdicts from ``plan_pooled_band_check``, decision by decision (candidate's own pool), in
    two batches: shifted to the MDE (passes, continued band failures and final fails occur) and
    to the global null (futility stops and final fails occur)."""
    mde = (plan.mde, plan.mde, plan.mde)
    batches = {
        "at_mde": _crosscheck_batch(phase_r, plan, replica, reps, seed, mde),
        "at_null": _crosscheck_batch(phase_r, plan, replica, reps, seed + 1, (0.0, 0.0, 0.0)),
    }
    counts = {k: sum(b["outcome_counts"][k] for b in batches.values()) for k in CODES.values()}
    return {
        "replicates": 2 * reps,
        "batches": batches,
        "mismatches": [f"{name}:{r}" for name, b in batches.items() for r in b["mismatches"]],
        "outcome_counts": counts,
    }


# ---------------------------------------------------------------- band cost (survival band v2)


def band_cost(phase_r, plan) -> Dict[str, Any]:
    """Normal approximation of survival band v2's per-look cost with the Phase R pilot SDs:
    for the pooled component and each mix's component, P(fail at look k) at a true delta of 0
    and at the pilot's observed delta (descriptive planning input)."""
    out: Dict[str, Any] = {
        "definition": (
            "pooled_ni_continue (survival band v2, amendment 2026-10-06, option 1): at look k "
            "the bands pass iff the pooled paired NI bound (per-world mean over the mixes of "
            "candidate - incumbent survival_fraction) clears -band_ni_margin and every mix's "
            "bound clears -band_mix_margin at the plan's rci_obf level p_k (and every candidate "
            "mean >= floor); normal approximation with the Phase R pilot SDs; per look: "
            "P(fail) of the pooled component and of each mix's component at a true delta of 0 "
            "and at the pilot's observed delta; descriptive planning input (the rule was "
            "chosen on independent data)"
        ),
        "look_sizes": list(plan.look_sizes),
        "band_ni_margin": plan.band_ni_margin,
        "band_mix_margin": plan.band_mix_margin,
        "band_alpha": plan.band_alpha,
        "band_bound": plan.band_bound,
        "band_nominal_p": list(plan.band_nominal_p),
        "floor": plan.band_floor,
        "pools": {},
    }
    for label, seeds in (("candidate_own", (CANDIDATE_SEED,)), ("all_seeds", PHASE_R_SEEDS)):
        cand = np.asarray([pooled(phase_r, seeds, "candidate_survival", m) for m in MIXES])
        inc = np.asarray([pooled(phase_r, seeds, "incumbent_survival", m) for m in MIXES])
        d = cand - inc
        series = {**{m: d[j] for j, m in enumerate(MIXES)}, "pooled": d.mean(axis=0)}
        rows: Dict[str, Any] = {}
        for name, values in series.items():
            sd, obs = float(values.std(ddof=1)), float(values.mean())
            margin = plan.band_ni_margin if name == "pooled" else plan.band_mix_margin
            looks = {}
            for k, n in enumerate(plan.look_sizes):
                se = sd / math.sqrt(n)
                t = student_t_isf(plan.band_nominal_p[k], n - 1)
                looks[f"look_{k + 1}"] = {
                    "n": n,
                    "fail_if_true_delta_0": _NORMAL.cdf(t - margin / se),
                    "fail_at_pilot_delta": _NORMAL.cdf(t - (margin + obs) / se),
                }
            rows[name] = {
                "pilot_sd": sd,
                "pilot_mean_delta": obs,
                "margin": margin,
                "pilot_ties": int(np.sum(values == 0.0)),
                **looks,
            }
        out["pools"][label] = {
            "pilot_worlds": int(d.shape[1]),
            "pilot_candidate_means": dict(zip(MIXES, cand.mean(axis=1).tolist())),
            "components": rows,
        }
    out["note"] = (
        "Under the frozen v2 rule (paired_ni_at_stop, per-mix M 0.05, judged once at the "
        "qualifying look) a no-regression candidate failed the scripted band at look 2 about "
        "91% of the time (superseded_v2_band/). Survival band v2 judges the pooled delta (SD "
        "about 0.21 here) at margin 0.05 and each mix only against a catastrophic 0.075, and a "
        "qualified run whose bands fail continues to the next look instead of stopping."
    )
    return out


# ---------------------------------------------------------------- planning-only skew pre-run


def skew_prerun(deltas_by_mix, n_max: int, delta_ni: float, reps: int) -> Dict[str, Any]:
    handle, name = tempfile.mkstemp(suffix=".json", prefix="frp3strict-skew-prerun-")
    os.close(handle)
    tmp = Path(name)  # outside the repo (never in the source closure)
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


def plan_parameters(n_max: int, mde: float) -> Dict[str, Any]:
    from research.frp3_strict_20261005 import spec as S

    return R.study_plan_parameters(
        S.SPEC,
        n_max=n_max,
        mde=mde,
        fractions=list(FRACTIONS),
        futility_policy="followed",
        band_margin_z=BAND_MARGIN_Z,
        paired_band=dict(PAIRED_BAND),
    )


def development_delta_ni(phase_r: Mapping[str, Any]) -> float:
    """0.03 x the Phase R incumbent (champion+v8) scripted mean H5000 mass, all 160 worlds."""
    return NI_FRACTION * float(
        np.mean(pooled(phase_r, PHASE_R_SEEDS, "incumbent_mass", "scripted"))
    )


def main() -> int:
    began = time.process_time()
    phase_r = load_phase_r()
    own = phase_r["seeds"][CANDIDATE_SEED]
    deltas_by_mix = {m: own[m]["deltas"] for m in MIXES}
    write_json(DELTAS_OUT, deltas_by_mix)
    write_json(POOL_OUT, pool_document(phase_r), compact=True)
    size = sizing(phase_r)
    n_max, mde = size["chosen"]["n_max"], size["chosen"]["mde"]
    params = plan_parameters(n_max, mde)
    write_json(PLAN_OUT, R.frozen_plan_document(params))
    plan = R.plan_from_parameters(params)
    stats = {m: describe(deltas_by_mix[m]) for m in MIXES}
    sds = [stats[m]["sd"] for m in MIXES]
    own_means = [stats[m]["mean"] for m in MIXES]
    all_means = [float(np.mean(pooled(phase_r, PHASE_R_SEEDS, "deltas", m))) for m in MIXES]
    dev_ni = development_delta_ni(phase_r)
    corr = np.corrcoef(np.asarray([deltas_by_mix[m] for m in MIXES]))
    rho = float(np.mean([corr[0, 1], corr[0, 2], corr[1, 2]]))
    configure_sim(n_max, sds, mde, dev_ni)
    sim.SCENARIOS = normal_scenarios(mde, dev_ni, own_means, all_means, max(rho, 0.0))
    replica = sim.Replica(plan)
    mc = sim.part_mc(plan, replica)
    own_seed, all_seeds = (CANDIDATE_SEED,), PHASE_R_SEEDS
    boot = {
        "candidate_own_observed_effects": bootstrap_power(
            phase_r, own_seed, plan, replica, 40_000, 101
        ),
        "candidate_own_observed_mass_survival_no_change": bootstrap_power(
            phase_r, own_seed, plan, replica, 20_000, 107, survival=(0.0, 0.0, 0.0)
        ),
        "candidate_own_observed_mass_pooled_regression_at_margin": bootstrap_power(
            phase_r, own_seed, plan, replica, 20_000, 108, survival=(-0.05, -0.05, -0.05)
        ),
        "candidate_own_observed_mass_scripted_at_mix_margin": bootstrap_power(
            phase_r,
            own_seed,
            plan,
            replica,
            20_000,
            109,
            survival=(0.0, -plan.band_mix_margin, 0.0),
        ),
        "candidate_own_mde_survival_no_change": bootstrap_power(
            phase_r,
            own_seed,
            plan,
            replica,
            20_000,
            110,
            shift=(mde, mde, mde),
            survival=(0.0, 0.0, 0.0),
        ),
        "candidate_own_shape_at_all_seed_means": bootstrap_power(
            phase_r, own_seed, plan, replica, 20_000, 102, shift=tuple(all_means)
        ),
        "candidate_own_shape_at_mde_all_mixes": bootstrap_power(
            phase_r, own_seed, plan, replica, 20_000, 103, shift=(mde, mde, mde)
        ),
        "candidate_own_shape_global_null": bootstrap_power(
            phase_r, own_seed, plan, replica, 40_000, 104, shift=(0.0, 0.0, 0.0)
        ),
        "all_seeds_observed_effects": bootstrap_power(
            phase_r, all_seeds, plan, replica, 20_000, 105
        ),
        "all_seeds_shape_at_mde_all_mixes": bootstrap_power(
            phase_r, all_seeds, plan, replica, 20_000, 106, shift=(mde, mde, mde)
        ),
    }
    cross = crosscheck_bootstrap(phase_r, plan, replica)
    if cross["mismatches"]:
        raise SystemExit(f"bootstrap replica disagrees with run_sequential_gate: {cross}")
    projection = runtime_projection(n_max)
    skew = skew_prerun(deltas_by_mix, n_max, dev_ni, 300_000)
    report = {
        "schema": "frp3-m3s12-strict-oc/v2-survival-band-v2",
        "development_data": {
            "phase_r": str(PHASE_R),
            "summary_sha256": phase_r["summary_sha256"],
            "candidate_seed": CANDIDATE_SEED,
            "candidate_own_per_mix_deltas": stats,
            "all_seeds_per_mix_deltas": {
                m: describe(pooled(phase_r, PHASE_R_SEEDS, "deltas", m)) for m in MIXES
            },
            "between_mix_delta_correlation_mean": rho,
            "between_mix_delta_correlation": corr.tolist(),
            "skew_input_path": str(DELTAS_OUT.relative_to(REPO)),
            "skew_input_sha256": sha256_file(DELTAS_OUT),
            "paired_pool_path": str(POOL_OUT.relative_to(REPO)),
            "paired_pool_sha256": sha256_file(POOL_OUT),
            "frozen_plan_path": str(PLAN_OUT.relative_to(REPO)),
            "frozen_plan_sha256": sha256_file(PLAN_OUT),
        },
        "sizing": size,
        "plan_parameters": params,
        "plan": plan.as_dict(),
        "development_delta_ni": {
            "value": dev_ni,
            "rule": "0.03 x the Phase R incumbent (champion+v8) scripted mean H5000 mass, all "
            "160 worlds; the binding delta_NI is 0.03 x the calibration incumbent scripted mean",
        },
        "normal_monte_carlo": {
            "method": "simulate.py Replica/summarize/part_mc reused unchanged; normal deltas "
            "with the candidate's own per-mix SDs; efficacy and NI only (bands not simulated)",
            "per_mix_sd": dict(zip(MIXES, sds)),
            **mc,
        },
        "joint_bootstrap": {
            "method": "resample Phase R world indices jointly across mixes (N_max per "
            "replicate), both arms' H5000 survival from the same worlds, a 16-world "
            "calibration delta_NI resampled per replicate (0.03 x incumbent scripted mass); "
            "survival band v2 (pooled_ni_continue: pooled M 0.05, per-mix 0.075, alpha 0.05 "
            "rci_obf, floor 0.30) judged at the qualifying look and every later look, a "
            "qualified run whose bands fail continues; futility followed before qualification; "
            "'survival_deltas' = observed, or the candidate's survival recentred on the "
            "incumbent's at the stated true deltas (what-if)",
            **boot,
            "crosscheck_vs_run_sequential_gate": cross,
        },
        "mac_fallback_runtime": projection,
        "skew_check_prerun": skew,
        "cpu_seconds_total": round(time.process_time() - began, 1),
    }
    write_json(OC_OUT, report)
    write_json(BAND_OUT, {"schema": "frp3-m3s12-strict-band-cost/v2", **band_cost(phase_r, plan)})
    print(
        json.dumps(
            {
                "n_max": n_max,
                "mde": mde,
                "development_delta_ni": dev_ni,
                "look_sizes": list(plan.look_sizes),
                "oc": str(OC_OUT),
                "bands": str(BAND_OUT),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
