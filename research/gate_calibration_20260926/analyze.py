#!/usr/bin/env python3
"""Read-only gate-calibration analysis of saved H5000 strict-pilot records.

Descriptive, UNAUDITED statistics over the per-episode JSON records that the
``task-aligned-strict-challenge-20260926`` pilot-v1 run saved:

* ``development/producer/records/incumbent-*.json``: 48 Apex episodes
  (16 worlds per mix, development seed namespace);
* ``pilot/producer/records/{candidate,incumbent}-*.json``: 48 candidate plus
  48 paired Apex episodes on the same (disjoint from development) worlds.

It reports per-mix Apex mean/SD/CV per estimand, paired candidate-vs-Apex
delta SDs, pairing efficiency, required world counts at an MDE of 10% of the
development-bank Apex mean (the pilot's own sizing rule, reusing
``src.scripts.eval_stats._paired_delta_required_count`` at alpha 0.05/3), how
much of the outcome variance death timing explains, the death-cause mix, a
non-binding futility rule evaluated on the pilot, and a throughput budget.

The script only reads records and writes one JSON file at ``--out``.  It uses
the standard library plus numpy, has no randomness, and its output is
byte-identical for identical inputs (sorted keys, no timestamps).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.scripts.eval_stats import (  # noqa: E402  (repo-root import after path setup)
    PAIRED_DELTA_PILOT_FAMILY_ALPHA,
    PAIRED_DELTA_PILOT_MIXES,
    _paired_delta_required_count,
    student_t_isf,
)

SCHEMA_VERSION = "gate-calibration/v1"
MIXES: Tuple[str, ...] = ("frozen", "scripted", "mixed")
ARMS: Tuple[str, ...] = ("incumbent", "candidate")
DEFAULT_SOURCE = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/"
    "task-aligned-strict-challenge-20260926/pilot-v1/output"
)
PROTECTED_ROOT = "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913"
DEFAULT_OUT = (
    "/private/tmp/claude-501/-Users-josenunez-Projects-ml-snake-dqn/"
    "8b38327f-2897-4918-b079-74962b22441c/scratchpad/gate_calibration/gate_calibration.json"
)
MDE_FRACTION = 0.1
PLANNING_ALPHA = PAIRED_DELTA_PILOT_FAMILY_ALPHA / PAIRED_DELTA_PILOT_MIXES
FUTILITY_FAMILY_ALPHA = 0.10
FUTILITY_ONE_SIDED_ALPHA = FUTILITY_FAMILY_ALPHA / len(MIXES)  # per mix, per look
MULTIPLICATIVE_MDE_LADDER: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.50)
SIGN_MDE = 0.1  # mean sign delta 0.1 <=> P(candidate beats Apex) 0.50 -> 0.55
# Projected batch-engine speedup for vector61 gates (docs/research/
# simd_vector61_plan_2026-09-26.md "Measured cost": 3-4x, not 100x; the
# featurizer dominates). Not measured end to end; midpoint used here.
SIMD_SPEEDUP = 3.5
LIVE_SECONDS_PER_EPISODE_FALLBACK = 851.6772085831035 / 48.0

Record = Dict[str, Any]


# --------------------------------------------------------------------------- loading


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def normalize_record(raw: Dict[str, Any], arm: str, stage: str) -> Record:
    """Flatten one saved episode record into the fields this analysis uses."""
    identity = raw.get("world_identity") or {}
    probes = raw.get("probes") or {}
    mix = identity.get("mix_id")
    if mix not in MIXES:
        raise ValueError(f"unknown mix {mix!r} in record seed={raw.get('seed')!r}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    deaths = int(raw.get("deaths", 0))
    cause = probes.get("death_cause")
    return {
        "arm": arm,
        "stage": stage,
        "mix": mix,
        "seed": int(raw["seed"]),
        "mass_integral": float(raw["mass_integral"]),
        "survival_fraction": float(raw["survival_fraction"]),
        "mean_mass_alive": float(raw.get("mean_mass_alive", 0.0)),
        "max_mass": float(raw.get("max_mass", 0.0)),
        "deaths": deaths,
        "kills": int(raw.get("kills", 0)),
        "death_cause": (cause if cause else "none") if deaths > 0 else "alive_at_horizon",
        "food_eaten": probes.get("food_eaten"),
        "boost_frame_fraction": probes.get("boost_frame_fraction"),
    }


def load_stage(stage_dir: str, stage: str) -> Tuple[List[Record], List[Dict[str, str]]]:
    """Load ``<stage_dir>/producer/records/<arm>-<mix>-<seed>.json`` sorted by name."""
    pattern = os.path.join(stage_dir, "producer", "records", "*.json")
    records: List[Record] = []
    sources: List[Dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        arm = name.split("-", 1)[0]
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        record = normalize_record(raw, arm, stage)
        expected = f"{arm}-{record['mix']}-{record['seed']}.json"
        if name != expected:
            raise ValueError(f"record file {name} does not match its content ({expected})")
        records.append(record)
        sources.append({"file": f"{stage}/{name}", "sha256": _sha256_file(path)})
    return records, sources


def stage_elapsed_seconds(stage_dir: str) -> Optional[float]:
    """Wall time the stage reported in ``physical-report.json`` (None if absent)."""
    path = os.path.join(stage_dir, "physical-report.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle).get("elapsed_seconds")
    return float(value) if isinstance(value, (int, float)) else None


# --------------------------------------------------------------------------- statistics


def _f(value: float) -> Optional[float]:
    """JSON-safe float: NaN/inf become None, finite values stay full precision."""
    value = float(value)
    return value if math.isfinite(value) else None


def describe(values: Sequence[float]) -> Dict[str, Any]:
    """n, mean, SD (ddof=1), CV = SD/|mean|, and quantiles."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    q25, median, q75 = (float(x) for x in np.quantile(arr, [0.25, 0.5, 0.75]))
    return {
        "n": n,
        "mean": _f(mean),
        "sd": _f(sd),
        "cv": _f(sd / abs(mean)) if mean != 0.0 else None,
        "median": _f(median),
        "q25": _f(q25),
        "q75": _f(q75),
        "min": _f(arr.min()),
        "max": _f(arr.max()),
    }


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (deterministic)."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return None
    return _f(np.corrcoef(a, b)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(average_ranks(x), average_ranks(y))


def sign_test_two_sided(wins: int, losses: int) -> Optional[float]:
    """Exact two-sided binomial sign test p-value, ties dropped."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def required_count(delta_sd: Optional[float], mde: Optional[float]) -> Optional[int]:
    """Pilot sizing rule: paired t, 80% marginal power, alpha 0.05/3 per mix."""
    if delta_sd is None or mde is None or not mde > 0.0:
        return None
    return int(_paired_delta_required_count(float(delta_sd), float(mde), PLANNING_ALPHA))


def one_sided_upper_bound(deltas: Sequence[float], alpha: float) -> Optional[float]:
    """Upper one-sided (1 - alpha) t bound on the mean paired delta."""
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size < 2:
        return None
    se = float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return _f(float(arr.mean()) + student_t_isf(alpha, arr.size - 1) * se)


# --------------------------------------------------------------------------- estimands

Estimand = Callable[[Record], float]
ESTIMANDS: Dict[str, Estimand] = {
    "mass_integral": lambda r: r["mass_integral"],
    "survival_fraction": lambda r: r["survival_fraction"],
    "log1p_mass": lambda r: math.log1p(r["mass_integral"]),
    # Growth while alive: mass_integral = survival_fraction * mean_mass_alive.
    "mean_mass_alive": lambda r: r["mean_mass_alive"],
}


def by_mix(records: Sequence[Record], arm: str, stage: str) -> Dict[str, List[Record]]:
    out: Dict[str, List[Record]] = {mix: [] for mix in MIXES}
    for record in records:
        if record["arm"] == arm and record["stage"] == stage:
            out[record["mix"]].append(record)
    for mix in MIXES:
        out[mix].sort(key=lambda r: r["seed"])
    return out


def pair_by_world(
    candidates: Sequence[Record], incumbents: Sequence[Record]
) -> List[Tuple[Record, Record]]:
    """Join candidate and incumbent records on (mix, seed); both sides must match 1:1."""
    inc = {(r["mix"], r["seed"]): r for r in incumbents}
    cand = {(r["mix"], r["seed"]): r for r in candidates}
    if len(inc) != len(incumbents) or len(cand) != len(candidates):
        raise ValueError("duplicate (mix, seed) world in one arm")
    if set(inc) != set(cand):
        raise ValueError("candidate and incumbent worlds differ; pairing is undefined")
    return [(cand[key], inc[key]) for key in sorted(cand)]


# --------------------------------------------------------------------------- sections


def apex_descriptives(records: Sequence[Record]) -> Dict[str, Any]:
    """Per stage, per mix, per estimand Apex descriptives (plus pooled ridit ranks)."""
    out: Dict[str, Any] = {}
    for stage in ("development", "pilot"):
        mixes = by_mix(records, "incumbent", stage)
        stage_out: Dict[str, Any] = {}
        for mix in MIXES:
            rows = mixes[mix]
            if not rows:
                continue
            entry = {name: describe([fn(r) for r in rows]) for name, fn in ESTIMANDS.items()}
            stage_out[mix] = entry
        if stage_out:
            out[stage] = stage_out
    # Rank-based view of bank-to-bank stability: pool the development and pilot
    # Apex banks within a mix; if the two banks come from one distribution each
    # bank's mean ridit is ~0.5 (Mann-Whitney AUC of pilot vs development).
    dev = by_mix(records, "incumbent", "development")
    pil = by_mix(records, "incumbent", "pilot")
    bank_ranks: Dict[str, Any] = {}
    for mix in MIXES:
        if not dev[mix] or not pil[mix]:
            continue
        values = [r["mass_integral"] for r in dev[mix]] + [r["mass_integral"] for r in pil[mix]]
        ridits = (average_ranks(values) - 0.5) / len(values)
        n_dev = len(dev[mix])
        bank_ranks[mix] = {
            "development_mean_ridit": _f(ridits[:n_dev].mean()),
            "pilot_mean_ridit": _f(ridits[n_dev:].mean()),
            "pilot_vs_development_mean_mass_ratio": _f(
                np.mean([r["mass_integral"] for r in pil[mix]])
                / np.mean([r["mass_integral"] for r in dev[mix]])
            ),
        }
    out["bank_to_bank_rank_check"] = bank_ranks
    return out


def reference_means(records: Sequence[Record]) -> Dict[str, Dict[str, float]]:
    """Development-bank Apex mean per mix per estimand (the pilot's MDE reference)."""
    dev = by_mix(records, "incumbent", "development")
    return {
        mix: {name: float(np.mean([fn(r) for r in dev[mix]])) for name, fn in ESTIMANDS.items()}
        for mix in MIXES
        if dev[mix]
    }


def _paired_block(
    cand: Sequence[float], inc: Sequence[float], mde: Optional[float]
) -> Dict[str, Any]:
    c = np.asarray(cand, dtype=np.float64)
    a = np.asarray(inc, dtype=np.float64)
    d = c - a
    sd = float(d.std(ddof=1)) if d.size > 1 else None
    var_c = float(c.var(ddof=1)) if c.size > 1 else float("nan")
    var_a = float(a.var(ddof=1)) if a.size > 1 else float("nan")
    unpaired_sd = math.sqrt(var_c + var_a) if math.isfinite(var_c + var_a) else None
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    t_stat = float(d.mean()) / (sd / math.sqrt(d.size)) if sd else None
    return {
        "n": int(d.size),
        "candidate_mean": _f(c.mean()),
        "incumbent_mean": _f(a.mean()),
        "delta": describe(d),
        "delta_t": _f(t_stat) if t_stat is not None else None,
        "wins": wins,
        "losses": losses,
        "ties": int(d.size - wins - losses),
        "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
        "pearson_candidate_vs_incumbent": pearson(c, a),
        "spearman_candidate_vs_incumbent": spearman(c, a),
        "pairing_variance_ratio": (
            _f((var_c + var_a) / (sd * sd)) if sd and unpaired_sd is not None else None
        ),
        "mde": _f(mde) if mde is not None else None,
        "required_worlds_paired": required_count(sd, mde),
        "required_worlds_per_arm_unpaired": required_count(unpaired_sd, mde),
        "standardized_mde": _f(mde / sd) if sd and mde else None,
        "observed_standardized_effect": _f(float(d.mean()) / sd) if sd else None,
    }


def paired_analysis(records: Sequence[Record]) -> Dict[str, Any]:
    """Candidate-vs-Apex on the pilot worlds, per mix, per estimand."""
    ref = reference_means(records)
    cand = [r for r in records if r["arm"] == "candidate" and r["stage"] == "pilot"]
    inc = [r for r in records if r["arm"] == "incumbent" and r["stage"] == "pilot"]
    pairs = pair_by_world(cand, inc)
    out: Dict[str, Any] = {}
    for mix in MIXES:
        mp = [(c, a) for c, a in pairs if c["mix"] == mix]
        if not mp:
            continue
        entry: Dict[str, Any] = {}
        for name, fn in ESTIMANDS.items():
            mde = MDE_FRACTION * abs(ref[mix][name]) if mix in ref else None
            entry[name] = _paired_block([fn(c) for c, _ in mp], [fn(a) for _, a in mp], mde)
        # log1p with a multiplicative MDE: delta log ~ log(1.1) is a 10% mass ratio.
        entry["log1p_mass_multiplicative_mde"] = _paired_block(
            [math.log1p(c["mass_integral"]) for c, _ in mp],
            [math.log1p(a["mass_integral"]) for _, a in mp],
            math.log(1.0 + MDE_FRACTION),
        )
        # Pooled ridit: rank all 2n pilot mass values in the mix together.
        values = [c["mass_integral"] for c, _ in mp] + [a["mass_integral"] for _, a in mp]
        ridits = (average_ranks(values) - 0.5) / len(values)
        n = len(mp)
        apex_ridit_mean = float(ridits[n:].mean())
        entry["ridit_mass"] = _paired_block(
            ridits[:n].tolist(), ridits[n:].tolist(), MDE_FRACTION * apex_ridit_mean
        )
        signs = [float(np.sign(c["mass_integral"] - a["mass_integral"])) for c, a in mp]
        entry["sign_mass"] = _paired_block(signs, [0.0] * n, SIGN_MDE)
        out[mix] = entry
    return {
        "mde_reference": "development-bank Apex mean per mix x 0.10 (pilot sizing rule)",
        "mde_reference_means": ref,
        "planning_alpha_per_mix": PLANNING_ALPHA,
        "per_mix": out,
    }


def death_timing(records: Sequence[Record]) -> Dict[str, Any]:
    """How much of the mass outcome is death timing.

    With ``hero_terminal`` the episode ends at the first death, so
    ``mass_integral = survival_fraction * mean_mass_alive`` exactly and
    ``log m = log s + log a``.  We report corr(mass, survival), its square, and
    the additive decomposition of Var(log m) into Var(log s), Var(log a) and
    2 Cov; for the pilot we also report the same for the paired deltas.
    """
    out: Dict[str, Any] = {}
    groups = [
        ("development", "incumbent"),
        ("pilot", "incumbent"),
        ("pilot", "candidate"),
    ]
    for stage, arm in groups:
        mixes = by_mix(records, arm, stage)
        for mix in MIXES + ("all",):
            rows = [r for m in MIXES for r in mixes[m]] if mix == "all" else mixes.get(mix, [])
            if len(rows) < 3:
                continue
            out[f"{stage}/{arm}/{mix}"] = _timing_block(rows)
    cand = [r for r in records if r["arm"] == "candidate" and r["stage"] == "pilot"]
    inc = [r for r in records if r["arm"] == "incumbent" and r["stage"] == "pilot"]
    if cand and inc:
        pairs = pair_by_world(cand, inc)
        for mix in MIXES + ("all",):
            mp = [(c, a) for c, a in pairs if mix == "all" or c["mix"] == mix]
            if len(mp) < 3:
                continue
            dm = [c["mass_integral"] - a["mass_integral"] for c, a in mp]
            ds = [c["survival_fraction"] - a["survival_fraction"] for c, a in mp]
            dlm = [math.log(c["mass_integral"]) - math.log(a["mass_integral"]) for c, a in mp]
            dls = [
                math.log(c["survival_fraction"]) - math.log(a["survival_fraction"]) for c, a in mp
            ]
            r = pearson(dm, ds)
            rl = pearson(dlm, dls)
            out[f"pilot/paired_delta/{mix}"] = {
                "n": len(mp),
                "pearson_delta_mass_vs_delta_survival": r,
                "r2_delta_mass_on_delta_survival": _f(r * r) if r is not None else None,
                "pearson_delta_log_mass_vs_delta_log_survival": rl,
                "r2_delta_log_mass_on_delta_log_survival": _f(rl * rl) if rl is not None else None,
            }
    return out


def _timing_block(rows: Sequence[Record]) -> Dict[str, Any]:
    mass = [r["mass_integral"] for r in rows]
    surv = [r["survival_fraction"] for r in rows]
    log_m = np.log(np.asarray(mass, dtype=np.float64))
    log_s = np.log(np.asarray(surv, dtype=np.float64))
    log_a = log_m - log_s  # log mean_mass_alive (exact identity under hero_terminal)
    var_m = float(log_m.var(ddof=1))
    cov = float(np.cov(log_s, log_a, ddof=1)[0, 1])
    r = pearson(mass, surv)
    return {
        "n": len(rows),
        "pearson_mass_vs_survival": r,
        "r2_mass_on_survival": _f(r * r) if r is not None else None,
        "spearman_mass_vs_survival": spearman(mass, surv),
        "var_log_mass": _f(var_m),
        "share_var_log_survival": _f(float(log_s.var(ddof=1)) / var_m) if var_m else None,
        "share_var_log_mean_mass_alive": _f(float(log_a.var(ddof=1)) / var_m) if var_m else None,
        "share_2cov": _f(2.0 * cov / var_m) if var_m else None,
        "max_identity_error": _f(
            max(
                abs(r["mass_integral"] - r["survival_fraction"] * r["mean_mass_alive"])
                for r in rows
            )
        ),
    }


def death_causes(records: Sequence[Record]) -> Dict[str, Any]:
    """Death-cause counts per stage/arm/mix and per stage/arm overall."""
    out: Dict[str, Dict[str, int]] = {}
    for r in records:
        for key in (f"{r['stage']}/{r['arm']}/{r['mix']}", f"{r['stage']}/{r['arm']}/all"):
            bucket = out.setdefault(key, {})
            bucket[r["death_cause"]] = bucket.get(r["death_cause"], 0) + 1
    apex = [r for r in records if r["arm"] == "incumbent"]
    total: Dict[str, int] = {}
    for r in apex:
        total[r["death_cause"]] = total.get(r["death_cause"], 0) + 1
    out["all_incumbent"] = total
    return {key: dict(sorted(value.items())) for key, value in sorted(out.items())}


def futility_evaluation(records: Sequence[Record]) -> Dict[str, Any]:
    """Evaluate a non-binding clear-loser stop on the pilot pairs.

    Rule: at a look with n worlds per mix, stop for futility if, in ANY mix,
    the one-sided (1 - 0.10/3) upper t bound on the mean paired delta is below
    zero, so a truly-equal candidate is falsely stopped at most ~10% per look.
    Being non-binding (it can only stop, never promote) it does not inflate the
    promotion type-I error.  Looks: first 8 worlds (by seed order) and all 16.
    """
    cand = [r for r in records if r["arm"] == "candidate" and r["stage"] == "pilot"]
    inc = [r for r in records if r["arm"] == "incumbent" and r["stage"] == "pilot"]
    if not cand:
        return {}
    pairs = pair_by_world(cand, inc)
    estimands = {
        "mass_integral": ESTIMANDS["mass_integral"],
        "log1p_mass": ESTIMANDS["log1p_mass"],
    }
    out: Dict[str, Any] = {"one_sided_alpha": FUTILITY_ONE_SIDED_ALPHA, "looks": {}}
    n_max = min(sum(1 for c, _ in pairs if c["mix"] == mix) for mix in MIXES)
    looks = sorted({max(2, n_max // 2), n_max})
    for look in looks:
        look_out: Dict[str, Any] = {}
        for name, fn in estimands.items():
            per_mix: Dict[str, Any] = {}
            for mix in MIXES:
                mp = [(c, a) for c, a in pairs if c["mix"] == mix][:look]
                deltas = [fn(c) - fn(a) for c, a in mp]
                upper = one_sided_upper_bound(deltas, FUTILITY_ONE_SIDED_ALPHA)
                per_mix[mix] = {
                    "n": len(deltas),
                    "mean_delta": _f(np.mean(deltas)),
                    "upper_bound": upper,
                    "futile": bool(upper is not None and upper < 0.0),
                }
            look_out[name] = {
                "per_mix": per_mix,
                "stop": any(v["futile"] for v in per_mix.values()),
            }
        out["looks"][str(look)] = look_out
    return out


def throughput_budget(
    paired: Dict[str, Any], live_seconds_per_episode: float, simd_speedup: float
) -> Dict[str, Any]:
    """Episodes and compute for a gate that runs max-over-mixes N in every mix.

    Apex episodes are deterministic given the world seed, so the incumbent arm
    of a fixed world bank is computed once and cached; the marginal cost per
    candidate is the candidate arm only (3 * N episodes).
    """
    out: Dict[str, Any] = {
        "live_seconds_per_episode": live_seconds_per_episode,
        "simd_speedup_assumed": simd_speedup,
        "per_estimand": {},
    }
    per_mix = paired.get("per_mix", {})
    names = sorted({name for entry in per_mix.values() for name in entry})
    for name in names:
        counts = {mix: per_mix[mix][name]["required_worlds_paired"] for mix in per_mix}
        if any(v is None for v in counts.values()) or not counts:
            continue
        n = max(counts.values())
        candidate_eps = len(MIXES) * n
        live_h = candidate_eps * live_seconds_per_episode / 3600.0
        out["per_estimand"][name] = {
            "required_worlds_by_mix": counts,
            "gate_worlds_per_mix": n,
            "candidate_episodes": candidate_eps,
            "incumbent_episodes_once_per_bank": candidate_eps,
            "live_cpu_hours_per_candidate": _f(live_h),
            "simd_hours_per_candidate": _f(live_h / simd_speedup),
            "required_episodes_per_second_for_1h_gate": _f(candidate_eps / 3600.0),
            "required_scored_frames_per_second_for_1h_gate_upper_bound": _f(
                candidate_eps * 5000.0 / 3600.0
            ),
        }
    return out


def mass_mde_ladder(records: Sequence[Record], paired: Dict[str, Any]) -> Dict[str, Any]:
    """Required worlds for raw mass_integral at MDE fractions of the POOLED Apex mean.

    The pilot's MDE reference was one 16-world development bank; pooling the
    development and pilot Apex banks (32 worlds per mix) is less noisy.
    """
    out: Dict[str, Any] = {}
    pooled = {
        mix: float(
            np.mean(
                [r["mass_integral"] for r in records if r["arm"] == "incumbent" and r["mix"] == mix]
            )
        )
        for mix in paired.get("per_mix", {})
    }
    for fraction in MULTIPLICATIVE_MDE_LADDER:
        counts = {
            mix: required_count(entry["mass_integral"]["delta"]["sd"], fraction * pooled[mix])
            for mix, entry in paired.get("per_mix", {}).items()
        }
        out[f"{fraction:.2f}"] = {
            "mde_by_mix": {mix: fraction * pooled[mix] for mix in pooled},
            "required_worlds_by_mix": counts,
            "gate_worlds_per_mix": max(counts.values()) if counts else None,
        }
    return {"pooled_apex_mean_by_mix": pooled, "ladder": out}


def log_mde_ladder(paired: Dict[str, Any]) -> Dict[str, Any]:
    """Required worlds for the log1p-mass estimand at several multiplicative MDEs."""
    out: Dict[str, Any] = {}
    for fraction in MULTIPLICATIVE_MDE_LADDER:
        mde = math.log(1.0 + fraction)
        counts = {
            mix: required_count(entry["log1p_mass"]["delta"]["sd"], mde)
            for mix, entry in paired.get("per_mix", {}).items()
        }
        out[f"{fraction:.2f}"] = {
            "delta_log_mde": mde,
            "required_worlds_by_mix": counts,
            "gate_worlds_per_mix": max(counts.values()) if counts else None,
        }
    return out


def aa_across_banks(records: Sequence[Record]) -> Dict[str, Any]:
    """Apex vs Apex across two DISJOINT world banks (development vs pilot).

    Same policy, different worlds: this is the only A/A contrast the saved
    records support.  Welch t on each estimand shows whether the bank-to-bank
    gap is within ordinary world-to-world noise.
    """
    dev = by_mix(records, "incumbent", "development")
    pil = by_mix(records, "incumbent", "pilot")
    out: Dict[str, Any] = {}
    for mix in MIXES:
        if len(dev[mix]) < 2 or len(pil[mix]) < 2:
            continue
        entry: Dict[str, Any] = {}
        for name, fn in ESTIMANDS.items():
            a = np.asarray([fn(r) for r in dev[mix]], dtype=np.float64)
            b = np.asarray([fn(r) for r in pil[mix]], dtype=np.float64)
            va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
            se = math.sqrt(va + vb)
            df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1)) if se else None
            entry[name] = {
                "pilot_minus_development": _f(b.mean() - a.mean()),
                "welch_t": _f((b.mean() - a.mean()) / se) if se else None,
                "welch_df": _f(df) if df is not None else None,
            }
        out[mix] = entry
    return out


# --------------------------------------------------------------------------- driver

DISCLAIMER = (
    "UNAUDITED descriptive statistics computed from saved pilot-v1 records; "
    "n=16 worlds per mix per stage; not a promotion decision and not independently replayed."
)


def _is_under(path: str, root: str) -> bool:
    path = os.path.realpath(os.path.abspath(path))
    root = os.path.realpath(os.path.abspath(root))
    return path == root or path.startswith(root + os.sep)


def analyze(source: str, live_seconds_per_episode: Optional[float] = None) -> Dict[str, Any]:
    """Run every section over ``<source>/{development,pilot}`` and return one dict."""
    records: List[Record] = []
    sources: List[Dict[str, str]] = []
    elapsed: Dict[str, Any] = {}
    for stage in ("development", "pilot"):
        stage_dir = os.path.join(source, stage)
        stage_records, stage_sources = load_stage(stage_dir, stage)
        records.extend(stage_records)
        sources.extend(stage_sources)
        seconds = stage_elapsed_seconds(stage_dir)
        elapsed[stage] = {
            "elapsed_seconds": seconds,
            "episodes": len(stage_records),
            "seconds_per_episode": (
                _f(seconds / len(stage_records)) if seconds and stage_records else None
            ),
        }
    if not records:
        raise ValueError(f"no records found under {source}")
    if live_seconds_per_episode is None:
        live_seconds_per_episode = (
            elapsed["development"]["seconds_per_episode"] or LIVE_SECONDS_PER_EPISODE_FALLBACK
        )
    dev_seeds = {(r["mix"], r["seed"]) for r in records if r["stage"] == "development"}
    pilot_seeds = {(r["mix"], r["seed"]) for r in records if r["stage"] == "pilot"}
    combined = hashlib.sha256(
        "".join(f"{s['file']}:{s['sha256']}\n" for s in sources).encode("utf-8")
    ).hexdigest()
    counts: Dict[str, int] = {}
    for r in records:
        key = f"{r['stage']}/{r['arm']}/{r['mix']}"
        counts[key] = counts.get(key, 0) + 1
    paired = paired_analysis(records) if pilot_seeds else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "disclaimer": DISCLAIMER,
        "inputs": {
            "record_count": len(records),
            "records_sha256_digest": combined,
            "counts": dict(sorted(counts.items())),
            "development_pilot_world_overlap": len(dev_seeds & pilot_seeds),
            "stage_timing": elapsed,
        },
        "apex_descriptives": apex_descriptives(records),
        "paired": paired,
        "death_timing": death_timing(records),
        "death_causes": death_causes(records),
        "futility": futility_evaluation(records),
        "log_mde_ladder": log_mde_ladder(paired),
        "mass_mde_ladder": mass_mde_ladder(records, paired),
        "aa_across_banks": aa_across_banks(records),
        "throughput": (
            throughput_budget(paired, float(live_seconds_per_episode), SIMD_SPEEDUP)
            if paired
            else {}
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="pilot-v1 output directory")
    parser.add_argument("--out", default=DEFAULT_OUT, help="JSON output path (only write)")
    parser.add_argument(
        "--live-seconds-per-episode",
        type=float,
        default=None,
        help="live H5000 seconds per episode (default: development stage wall time / episodes)",
    )
    args = parser.parse_args(argv)
    if _is_under(args.out, PROTECTED_ROOT) or _is_under(args.out, args.source):
        parser.error("--out must not be inside the protected research artifact tree")
    result = analyze(args.source, args.live_seconds_per_episode)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(payload)
    print(f"wrote {args.out} ({len(payload)} bytes, {result['inputs']['record_count']} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
