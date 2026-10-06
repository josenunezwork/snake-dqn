#!/usr/bin/env python3
"""Operating characteristics of survival band v2 (governance amendment 2026-10-06).

Parts (single-threaded numpy; no game is played):

* ``oc`` -- the design study.  Candidate survival-guardrail rules are compared on the
  **pre-existing** calibration pools of ``survival_pools_20261006.json`` (``extract.py``; no
  FRP-v3 data) and on a normal stress model, (1) band only, at the looks of the plan and (2)
  through a vectorized replica of the whole sequential gate (efficacy, scripted NI, futility,
  then the band rule), with joint world resampling (one index per world, shared by the three
  mixes, both arms' survival and the mass delta of that world).  Also the pre-stated selection
  of the per-mix catastrophic margin (``SELECTION_RULE``).
* ``descriptive`` -- the same gate replica on another pool (e.g. the FRP-v3 Phase R pool),
  **labelled descriptive**: it is never an input to the rule or its calibration.
* ``study`` -- the per-study pre-registration check of ``band_policy="pooled_ni_continue"``
  for a frozen plan (``--plan-params``) on the study's pool: joint error P(PASS and a
  regression exactly at a margin) for the pooled regression (-band_ni_margin in every mix) and
  each one-mix regression (-band_mix_margin), at the mass effects of the check; acceptance:
  every joint rate <= 1.2 x band_alpha.
* ``crosscheck`` -- the replica against the pure ``run_sequential_gate`` +
  ``plan_pooled_band_check`` of ``src/evaluation/sequential_gate.py``, decision by decision.

Rules simulated in ``oc`` (survival_fraction, d = candidate - incumbent on the same world):

* ``A_rci_at_stop``: per-mix paired NI, M 0.05, alpha 0.05 rci_obf, judged once at the
  qualifying look, a failure stops the gate (template v2 as frozen for FRP-v3).
* ``A_pointwise_at_stop``: the same with the pointwise bound (the 2026-10-03 recommendation).
* ``B_per_mix_final``: per-mix NI M 0.05 alpha 0.05, judged at N_max only (a run that
  qualifies early continues to N_max for the band).
* ``C_per_mix_recal_final``: per-mix NI with a recalibrated margin/alpha (M 0.075, alpha 0.10)
  at N_max only.
* ``D_pooled_final``: pooled NI (mean over mixes of d, M 0.05) and a per-mix catastrophic NI
  (M 0.075), alpha 0.05 each, at N_max only.
* ``E_pooled_continue`` (proposed): D's components with rci_obf levels, judged at every look
  from the qualifying look on; the first look where all pass is STOP_PASS (FINAL_PASS at the
  last), a qualified run whose bands fail continues; FINAL_FAIL if they never pass.
* ``E_pooled_continue_mix0.1`` / ``_mix0.125`` / ``_mix0.05``: E with another per-mix
  margin (the selection grid; E itself uses 0.075).
* ``F_floor_only``: survival co-reported, only the absolute floor (candidate mean >= 0.30).
* ``none``: no band (efficacy + NI only; the ceiling).

Every rule except ``none`` keeps the absolute floor 0.30 per mix.

Reproduce (one core; hold one shared CPU slot; outputs embed ``cpu_seconds``, every other
number is deterministic)::

  P=./venv/bin/python; D=research/survival_band_v2_20261006
  OMP_NUM_THREADS=1 $P $D/extract.py
  OMP_NUM_THREADS=1 $P $D/simulate.py --part oc --out $D/outputs/oc.json
  OMP_NUM_THREADS=1 $P $D/simulate.py --part descriptive \
      --data research/frp3_strict_20261005/paired_pool.json \
      --out $D/outputs/descriptive_frp3_phase_r.json
  OMP_NUM_THREADS=1 $P $D/simulate.py --part design-plan --out $D/outputs/design_plan_pooled.json
  OMP_NUM_THREADS=1 $P $D/simulate.py --part crosscheck \
      --plan-params $D/outputs/design_plan_pooled.json --delta-ni 9.945893662499998 \
      --pool frp2_phase2 --reps 300 --out $D/outputs/crosscheck.json
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

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import NormalDist  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sequential_gate_validation_20261002.simulate import Replica  # noqa: E402
from src.evaluation.screen_stats import futility_plan  # noqa: E402
from src.evaluation.sequential_gate import sequential_gate_plan  # noqa: E402
from src.scripts.eval_stats import student_t_isf  # noqa: E402

MIXES = ("frozen", "scripted", "mixed")
SCRIPTED = 1
POOLS_PATH = HERE / "survival_pools_20261006.json"
BETWEEN_CHECKPOINT = ("frp2_phase2", "frp2_phase1")
PHASE1_CELLS = tuple(
    f"frp2_phase1_cell_{c}" for c in ("C15000", "C30000", "C60000", "M15000", "M30000", "M60000")
)
SELECTION_POOLS = BETWEEN_CHECKPOINT + PHASE1_CELLS
VETO_CHANGE = ("v8_strict", "v8_refbank", "v7_strict", "v8_screen")
# The design plan: the FRP-v3 strict gate's frozen looks (N_max 275, MDE 65; a design input
# fixed before this amendment) and its development delta_NI 9.946.
DESIGN_N_MAX = 275
DESIGN_MDE = 65.0
DESIGN_DELTA_NI = 9.945893662499998
# Mass SDs the design plan was sized for (frozen in the FRP-v3 protocol): an optional rescale of
# the pools' mass deltas, so that qualification timing matches the plan (mass only).
PLAN_SIZING_MASS_SD = (134.6, 311.2, 196.4)
# Mass effects: multiples of the MDE, and the FRP-v3 protocol's planning effects (mass only).
THETA_MULTIPLES = (0.5, 0.67, 1.0, 1.5, 2.0)
PLANNED_EFFECTS = (98.1, 68.5, 130.5)
FLOOR = 0.30
ALPHA = 0.05
POOLED_MARGIN = 0.05
MIX_MARGIN = 0.075
MIX_MARGIN_GRID = (0.05, 0.075, 0.10, 0.125)
# Pre-stated selection of the per-mix margin (amendment, "Choice of the per-mix margin").
SELECTION_RULE = (
    "smallest band_mix_margin in {0.05, 0.075, 0.10, 0.125} such that, with rci_obf levels at "
    "alpha 0.05 and the pooled margin 0.05, the final-look band (n = N_max = 275, judged alone) "
    "passes a candidate with no survival change with probability >= 0.80 (the owner's target) in "
    "EVERY independent between-checkpoint pool: FRP-v2 Phase 2, FRP-v2 Phase 1 and each of the "
    "six FRP-v2 Phase 1 recipe cells (the realistic range of between-checkpoint paired survival "
    "SDs); the smallest margin protects one-mix regressions best"
)
SELECTION_FLOOR = 0.80
STRESS_GRID = (0.25, 0.30, 0.35, 0.40, 0.441, 0.50)
CHUNK = 4000
CHECK_FACTOR = 1.2
CHECK_MIN_REPS = 20_000
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str  # none | per_mix | pooled | floor_only
    timing: str  # at_stop | final_only | continue
    alpha: float = ALPHA
    bound: str = "pointwise"  # pointwise | rci_obf
    mix_margin: Optional[float] = None
    pooled_margin: Optional[float] = None
    floor: Optional[float] = FLOOR


def design_rules() -> List[Rule]:
    rules = [
        Rule("A_rci_at_stop", "per_mix", "at_stop", bound="rci_obf", mix_margin=0.05),
        Rule("A_pointwise_at_stop", "per_mix", "at_stop", mix_margin=0.05),
        Rule("B_per_mix_final", "per_mix", "final_only", mix_margin=0.05),
        Rule("C_per_mix_recal_final", "per_mix", "final_only", alpha=0.10, mix_margin=0.075),
        Rule(
            "D_pooled_final",
            "pooled",
            "final_only",
            mix_margin=MIX_MARGIN,
            pooled_margin=POOLED_MARGIN,
        ),
        Rule(
            "E_pooled_continue",
            "pooled",
            "continue",
            bound="rci_obf",
            mix_margin=MIX_MARGIN,
            pooled_margin=POOLED_MARGIN,
        ),
    ]
    for m in MIX_MARGIN_GRID:
        if m != MIX_MARGIN:
            rules.append(
                Rule(
                    f"E_pooled_continue_mix{m:g}",
                    "pooled",
                    "continue",
                    bound="rci_obf",
                    mix_margin=m,
                    pooled_margin=POOLED_MARGIN,
                )
            )
    rules.append(Rule("F_floor_only", "floor_only", "at_stop"))
    rules.append(Rule("none", "none", "at_stop", floor=None))
    return rules


def levels(rule: Rule, sizes: Sequence[int], n_max: int) -> List[float]:
    if rule.bound == "rci_obf":
        spent = futility_plan(tuple(sizes), alpha=rule.alpha, max_size=n_max)
        return [_NORMAL.cdf(-c) for c in spent.boundaries]
    return [rule.alpha] * len(sizes)


def wilson(k: int, n: int) -> Dict[str, float]:
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"rate": p, "low": centre - half, "high": centre + half, "k": int(k), "n": int(n)}


# ---------------------------------------------------------------- scenarios


def survival_scenarios(mix_margin: float = MIX_MARGIN) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {
        "no_change": (0.0, 0.0, 0.0),
        "improves_0.05_all": (0.05, 0.05, 0.05),
        "all_-0.025": (-0.025, -0.025, -0.025),
        "all_-0.05": (-0.05, -0.05, -0.05),
    }
    for size in (0.05, 0.075, 0.10, 0.125):
        for j, mix in enumerate(MIXES):
            shift = [0.0, 0.0, 0.0]
            shift[j] = -size
            out[f"one_mix_-{size:g}@{mix}"] = tuple(shift)
    return out


# ---------------------------------------------------------------- per-look band statistics


class LookStats:
    """Prefix statistics of the centred survival deltas (per mix and pooled over mixes)."""

    def __init__(self, d: np.ndarray, inc: np.ndarray, sizes: Sequence[int]):
        # d, inc: (reps, 3, N); d already centred per mix at the pool level
        self.sizes = list(sizes)
        pooled = d.mean(axis=1)
        self.mean, self.sd, self.pmean, self.psd, self.inc_mean = [], [], [], [], []
        for n in self.sizes:
            seg = d[:, :, :n]
            self.mean.append(seg.mean(axis=2))
            self.sd.append(seg.std(axis=2, ddof=1))
            pseg = pooled[:, :n]
            self.pmean.append(pseg.mean(axis=1))
            self.psd.append(pseg.std(axis=1, ddof=1))
            self.inc_mean.append(inc[:, :, :n].mean(axis=2))

    def ok(self, rule: Rule, k: int, shift: np.ndarray, level: float) -> np.ndarray:
        n = self.sizes[k]
        reps = self.mean[k].shape[0]
        ok = np.ones(reps, dtype=bool)
        if rule.kind == "none":
            return ok
        if rule.floor is not None:
            cand_mean = self.inc_mean[k] + self.mean[k] + shift[None, :]
            ok &= np.all(cand_mean >= rule.floor, axis=1)
        if rule.kind == "floor_only":
            return ok
        t = student_t_isf(level, n - 1)
        lower = self.mean[k] + shift[None, :] - t * self.sd[k] / math.sqrt(n)
        ok &= np.all(lower > -rule.mix_margin, axis=1)
        if rule.kind == "pooled":
            plower = self.pmean[k] + float(shift.mean()) - t * self.psd[k] / math.sqrt(n)
            ok &= plower > -rule.pooled_margin
        return ok


# ---------------------------------------------------------------- gate replica


def qualification(
    replica: Replica, plan, x: np.ndarray, delta_ni: float
) -> Tuple[np.ndarray, np.ndarray]:
    """First qualifying look (-1 if none) and the look where a non-qualifying run ends
    (futility stop, policy followed, or the last look)."""
    reps = x.shape[0]
    last = len(replica.sizes) - 1
    stats = replica.look_stats(x)
    crossed = np.zeros((reps, 3), dtype=bool)
    ni = np.zeros(reps, dtype=bool)
    active = np.ones(reps, dtype=bool)
    tau = np.full(reps, -1, dtype=np.int64)
    end = np.full(reps, last, dtype=np.int64)
    c_final = plan.efficacy_boundaries[last]
    for k, n in enumerate(replica.sizes):
        mean, sd, t = stats[k]
        crossed |= (t >= replica.t_eff[k]) & active[:, None]
        lower = mean[:, SCRIPTED] - replica.t_ni[k] * sd[:, SCRIPTED] / math.sqrt(n)
        ni |= (lower > -delta_ni) & active
        qualifies = (crossed.sum(axis=1) >= plan.required_successes) & ni
        now = active & qualifies
        tau[now] = k
        end[now] = k
        active &= ~now
        if k == last:
            break
        f = plan.fractions[k]
        z = replica.t_to_z(k, t)
        theta = plan.mde * math.sqrt(plan.n_max) / sd
        arg = z * math.sqrt(f) + theta * (1.0 - f) - c_final
        futile = (~crossed) & (arg < replica.z_cut * math.sqrt(1.0 - f))
        futile_now = active & (3 - futile.sum(axis=1) < plan.required_successes)
        end[futile_now] = k
        active &= ~futile_now
    return tau, end


def rule_outcome(
    rule: Rule,
    stats: LookStats,
    tau: np.ndarray,
    end: np.ndarray,
    shift: np.ndarray,
    level_by_look: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """(passes, stop look) per replicate for one band rule."""
    last = len(stats.sizes) - 1
    qualified = tau >= 0
    if rule.timing == "final_only":
        ok_last = stats.ok(rule, last, shift, level_by_look[last])
        stop = np.where(qualified, last, end)
        return qualified & ok_last, stop
    oks = [stats.ok(rule, k, shift, level_by_look[k]) for k in range(last + 1)]
    if rule.timing == "at_stop":
        ok_tau = np.zeros_like(qualified)
        for k in range(last + 1):
            ok_tau |= (tau == k) & oks[k]
        return qualified & ok_tau, end
    passes = np.zeros_like(qualified)
    stop = np.where(qualified, last, end)
    decided = ~qualified
    for k in range(last + 1):
        now = ~decided & (tau <= k) & oks[k]
        passes |= now
        stop[now] = k
        decided |= now
    return passes, stop


def resample(
    pool: Mapping[str, Any],
    rng: np.random.Generator,
    reps: int,
    n_max: int,
    theta: Sequence[float],
    mass_sd: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint world resampling: mass deltas shifted to ``theta`` (optionally rescaled per mix to
    ``mass_sd``), centred survival deltas, incumbent survival; one index per world."""
    mass = np.asarray([pool[m]["mass_delta"] for m in MIXES], dtype=float)
    mass = mass - mass.mean(axis=1, keepdims=True)
    if mass_sd is not None:
        mass = mass / mass.std(axis=1, ddof=1, keepdims=True) * np.asarray(mass_sd)[:, None]
    mass = mass + np.asarray(theta, dtype=float)[:, None]
    inc = np.asarray([pool[m]["incumbent_survival"] for m in MIXES], dtype=float)
    cand = np.asarray([pool[m]["candidate_survival"] for m in MIXES], dtype=float)
    d = cand - inc
    d = d - d.mean(axis=1, keepdims=True)
    idx = rng.integers(0, mass.shape[1], size=(reps, n_max))
    return (
        mass[:, idx].transpose(1, 0, 2),
        d[:, idx].transpose(1, 0, 2),
        inc[:, idx].transpose(1, 0, 2),
    )


def gate_cell(
    pool: Mapping[str, Any],
    plan,
    replica: Replica,
    rules: Sequence[Rule],
    scenarios: Mapping[str, Sequence[float]],
    theta: Sequence[float],
    reps: int,
    seed: int,
    delta_ni: float,
    mass_sd: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    sizes = list(plan.look_sizes)
    lv = {r.name: levels(r, sizes, plan.n_max) for r in rules}
    counts = {r.name: {s: 0 for s in scenarios} for r in rules}
    worlds = {r.name: {s: 0 for s in scenarios} for r in rules}
    qualified_total = 0
    tau_hist = np.zeros(len(sizes) + 1, dtype=np.int64)
    band_ok_given_q = {r.name: {s: 0 for s in scenarios} for r in rules}
    done = 0
    while done < reps:
        k = min(CHUNK, reps - done)
        x, d, inc = resample(pool, rng, k, plan.n_max, theta, mass_sd)
        tau, end = qualification(replica, plan, x, delta_ni)
        stats = LookStats(d, inc, sizes)
        qualified_total += int((tau >= 0).sum())
        tau_hist += np.bincount(tau + 1, minlength=len(sizes) + 1)
        for rule in rules:
            for name, shift in scenarios.items():
                passes, stop = rule_outcome(rule, stats, tau, end, np.asarray(shift), lv[rule.name])
                counts[rule.name][name] += int(passes.sum())
                band_ok_given_q[rule.name][name] += int(passes[tau >= 0].sum())
                worlds[rule.name][name] += int(np.asarray(sizes)[stop].sum())
        done += k
    out: Dict[str, Any] = {
        "theta": list(theta),
        "reps": reps,
        "seed": seed,
        "p_qualify": qualified_total / reps,
        "qualifying_look_distribution": {
            "never": tau_hist[0] / reps,
            **{f"look_{j + 1}": tau_hist[j + 1] / reps for j in range(len(sizes))},
        },
        "rules": {},
    }
    for rule in rules:
        out["rules"][rule.name] = {
            name: {
                "p_pass": wilson(counts[rule.name][name], reps),
                "p_band_pass_given_qualified": (
                    band_ok_given_q[rule.name][name] / qualified_total if qualified_total else None
                ),
                "expected_worlds_per_mix": worlds[rule.name][name] / reps,
            }
            for name in scenarios
        }
    return out


# ---------------------------------------------------------------- band only (fixed looks)


def band_only(
    d: np.ndarray, inc: np.ndarray, rules: Sequence[Rule], sizes: Sequence[int], n_max: int
) -> Dict[str, Any]:
    stats = LookStats(d, inc, sizes)
    scen = survival_scenarios()
    out: Dict[str, Any] = {}
    for rule in rules:
        lv = levels(rule, sizes, n_max)
        if rule.timing == "final_only":
            lv = [rule.alpha] * len(sizes)
        row: Dict[str, Any] = {}
        for name, shift in scen.items():
            oks = [stats.ok(rule, k, np.asarray(shift), lv[k]) for k in range(len(sizes))]
            cell = {f"n{n}": float(oks[k].mean()) for k, n in enumerate(sizes)}
            if rule.timing == "continue":
                for j in range(len(sizes)):
                    any_ok = np.zeros_like(oks[0])
                    for k in range(j, len(sizes)):
                        any_ok |= oks[k]
                    cell[f"by_final_from_look_{j + 1}"] = float(any_ok.mean())
            row[name] = cell
        out[rule.name] = row
    return out


def pool_band_only(pool, rules, sizes, n_max, reps, seed) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    _, d, inc = resample(pool, rng, reps, n_max, (0.0, 0.0, 0.0))
    return band_only(d, inc, rules, sizes, n_max)


def stress_band_only(sd: float, rules, sizes, n_max, reps, seed) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((reps, 3, n_max)) * sd
    inc = np.full_like(d, 0.6)
    return band_only(d, inc, rules, sizes, n_max)


def describe_pool(pool: Mapping[str, Any]) -> Dict[str, Any]:
    d = np.asarray(
        [
            np.asarray(pool[m]["candidate_survival"]) - np.asarray(pool[m]["incumbent_survival"])
            for m in MIXES
        ]
    )
    x = np.asarray([pool[m]["mass_delta"] for m in MIXES], dtype=float)
    inc = np.asarray([pool[m]["incumbent_survival"] for m in MIXES])
    cand = np.asarray([pool[m]["candidate_survival"] for m in MIXES])
    corr = np.corrcoef(d)
    binary = (cand >= 1.0).astype(float) - (inc >= 1.0).astype(float)
    return {
        "world_triples": int(d.shape[1]),
        "paired_survival_sd": dict(zip(MIXES, d.std(axis=1, ddof=1).round(4).tolist())),
        "paired_survival_mean": dict(zip(MIXES, d.mean(axis=1).round(4).tolist())),
        "pooled_paired_survival_sd": round(float(d.mean(axis=0).std(ddof=1)), 4),
        "ties": dict(zip(MIXES, (d == 0).mean(axis=1).round(3).tolist())),
        "between_mix_survival_delta_corr": [
            round(float(corr[0, 1]), 3),
            round(float(corr[0, 2]), 3),
            round(float(corr[1, 2]), 3),
        ],
        "mass_delta_sd": dict(zip(MIXES, x.std(axis=1, ddof=1).round(1).tolist())),
        "mass_survival_corr": dict(
            zip(MIXES, [round(float(np.corrcoef(x[j], d[j])[0, 1]), 3) for j in range(3)])
        ),
        "incumbent_survival_mean": dict(zip(MIXES, inc.mean(axis=1).round(4).tolist())),
        "binary_survived_to_h_delta_sd": dict(
            zip(MIXES, binary.std(axis=1, ddof=1).round(4).tolist())
        ),
        "binary_survived_to_h_incumbent_rate": dict(
            zip(MIXES, (inc >= 1.0).mean(axis=1).round(3).tolist())
        ),
    }


# ---------------------------------------------------------------- parts


def design_plan():
    return sequential_gate_plan(DESIGN_N_MAX, mde=DESIGN_MDE)


def thetas_for(mde: float) -> Dict[str, Tuple[float, float, float]]:
    out = {f"{m:g}xMDE": (m * mde,) * 3 for m in THETA_MULTIPLES}
    out["planned_frp3_protocol"] = PLANNED_EFFECTS
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(band_tables: Mapping[str, Any], stress: Mapping[str, Any]) -> Dict[str, Any]:
    final = f"n{DESIGN_N_MAX}"
    rows = []
    chosen = None
    for m in MIX_MARGIN_GRID:
        name = "E_pooled_continue" if m == MIX_MARGIN else f"E_pooled_continue_mix{m:g}"
        rates = {p: band_tables[p][name]["no_change"][final] for p in SELECTION_POOLS}
        ok = all(v >= SELECTION_FLOOR for v in rates.values())
        rows.append(
            {
                "band_mix_margin": m,
                "no_change_final_pass": rates,
                "worst_pool": min(rates, key=rates.get),
                "worst_rate": min(rates.values()),
                "meets": ok,
                "stress_no_change_final_pass_by_sd": {
                    key: stress[key][name]["no_change"][final] for key in stress
                },
                "one_mix_-0.05_final_pass_frp2_phase2": {
                    mix: band_tables["frp2_phase2"][name][f"one_mix_-0.05@{mix}"][final]
                    for mix in MIXES
                },
            }
        )
        if ok and chosen is None:
            chosen = m
    if chosen != MIX_MARGIN:
        raise SystemExit(f"selection gives {chosen}, the module constant is {MIX_MARGIN}")
    return {"rule": SELECTION_RULE, "grid": rows, "chosen_band_mix_margin": chosen}


def part_oc(args) -> Dict[str, Any]:
    data = json.loads(POOLS_PATH.read_text())
    plan = design_plan()
    replica = Replica(plan)
    rules = design_rules()
    sizes = list(plan.look_sizes)
    pools = BETWEEN_CHECKPOINT + PHASE1_CELLS + VETO_CHANGE
    out: Dict[str, Any] = {
        "schema": "survival-band-v2-oc/v1",
        "data": str(POOLS_PATH.relative_to(REPO)),
        "data_sha256": sha256_file(POOLS_PATH),
        "design_plan": {
            "n_max": DESIGN_N_MAX,
            "mde": DESIGN_MDE,
            "look_sizes": sizes,
            "delta_ni": DESIGN_DELTA_NI,
            "note": "the FRP-v3 strict gate's frozen looks, MDE and development delta_NI "
            "(design inputs fixed before this amendment; efficacy/NI/futility as frozen)",
        },
        "rules": [asdict(r) for r in rules],
        "rule_levels": {r.name: levels(r, sizes, plan.n_max) for r in rules},
        "pools": {p: describe_pool(data[p]) for p in pools},
        "band_only": {},
        "stress_band_only": {},
        "gate": {},
    }
    for i, p in enumerate(pools):
        out["band_only"][p] = pool_band_only(data[p], rules, sizes, plan.n_max, args.reps, 100 + i)
    for i, sd in enumerate(STRESS_GRID):
        out["stress_band_only"][f"sd{sd:g}"] = stress_band_only(
            sd, rules, sizes, plan.n_max, args.reps, 200 + i
        )
    out["selection"] = selection(out["band_only"], out["stress_band_only"])
    scen = survival_scenarios()
    gate_pools = pools if args.all_gate_pools else BETWEEN_CHECKPOINT + ("v8_strict",)
    gate_pools = tuple(p for p in gate_pools if p not in PHASE1_CELLS or args.all_gate_pools)
    for i, p in enumerate(gate_pools):
        rows = {}
        for j, (label, theta) in enumerate(thetas_for(plan.mde).items()):
            rows[label] = gate_cell(
                data[p],
                plan,
                replica,
                rules,
                scen,
                theta,
                args.gate_reps,
                1000 + 37 * i + j,
                DESIGN_DELTA_NI,
            )
        out["gate"][p] = {"mass_as_pool": rows}
    # mass rescaled to the plan's sizing SDs (qualification timing of the plan; mass only)
    for i, p in enumerate(BETWEEN_CHECKPOINT):
        rows = {}
        for j, label in enumerate(("1xMDE", "planned_frp3_protocol")):
            theta = thetas_for(plan.mde)[label]
            rows[label] = gate_cell(
                data[p],
                plan,
                replica,
                rules,
                scen,
                theta,
                args.gate_reps,
                3000 + 37 * i + j,
                DESIGN_DELTA_NI,
                mass_sd=PLAN_SIZING_MASS_SD,
            )
        out["gate"][p]["mass_rescaled_to_plan_sizing_sd"] = rows
    return out


def part_descriptive(args) -> Dict[str, Any]:
    """The design study's gate replica on another pool; DESCRIPTIVE only."""
    data = json.loads(Path(args.data).read_text())
    plan = design_plan()
    replica = Replica(plan)
    rules = design_rules()
    scen = survival_scenarios()
    pools = sorted(k for k, v in data.items() if isinstance(v, dict) and "world_seeds" in v)
    out: Dict[str, Any] = {
        "schema": "survival-band-v2-descriptive/v1",
        "label": "DESCRIPTIVE ONLY: not an input to the choice or calibration of survival band v2",
        "data": str(args.data),
        "data_sha256": sha256_file(Path(args.data)),
        "pools": {p: describe_pool(data[p]) for p in pools},
        "gate": {},
    }
    for i, p in enumerate(pools):
        rows = {}
        observed = (0.0, 0.0, 0.0)
        mass = [float(np.mean(data[p][m]["mass_delta"])) for m in MIXES]
        surv = [
            float(
                np.mean(
                    np.asarray(data[p][m]["candidate_survival"])
                    - np.asarray(data[p][m]["incumbent_survival"])
                )
            )
            for m in MIXES
        ]
        cells = {"1xMDE": ((plan.mde,) * 3, scen), "observed_mass": (tuple(mass), scen)}
        for j, (label, (theta, sc)) in enumerate(cells.items()):
            rows[label] = gate_cell(
                data[p],
                plan,
                replica,
                rules,
                sc,
                theta,
                args.gate_reps,
                5000 + 37 * i + j,
                DESIGN_DELTA_NI,
            )
        observed_scen = {"observed_survival": tuple(surv)}
        rows["observed_mass_and_survival"] = gate_cell(
            data[p],
            plan,
            replica,
            rules,
            observed_scen,
            tuple(mass),
            args.gate_reps,
            6000 + i,
            DESIGN_DELTA_NI,
        )
        rows["observed_means"] = {"mass": mass, "survival": surv, "zero_shift": observed}
        out["gate"][p] = rows
    return out


# ---------------------------------------------------------------- per-study check (policy)


def plan_from_document(path: Path):
    from research.sequential_strict_template import sequential_runner as R

    doc = json.loads(Path(path).read_text())
    params = doc["plan_parameters"]
    plan = R.plan_from_parameters(params)
    if plan.as_dict() != doc["plan"]:
        raise SystemExit("--plan-params: the rebuilt plan differs from its frozen 'plan'")
    if plan.band_policy != "pooled_ni_continue":
        raise SystemExit("--plan-params: band_policy is not pooled_ni_continue")
    return plan, params


def policy_rule(plan) -> Rule:
    return Rule(
        "pooled_ni_continue",
        "pooled",
        "continue",
        alpha=plan.band_alpha,
        bound=plan.band_bound,
        mix_margin=plan.band_mix_margin,
        pooled_margin=plan.band_ni_margin,
        floor=plan.band_floor,
    )


def check_scenarios(plan) -> Dict[str, Tuple[float, float, float]]:
    out = {
        "no_change": (0.0, 0.0, 0.0),
        "pooled_at_margin": (-plan.band_ni_margin,) * 3,
    }
    for j, mix in enumerate(MIXES):
        shift = [0.0, 0.0, 0.0]
        shift[j] = -plan.band_mix_margin
        out[f"one_mix_at_margin@{mix}"] = tuple(shift)
    return out


def part_study(args) -> Dict[str, Any]:
    plan, params = plan_from_document(Path(args.plan_params))
    if plan.futility_policy == "overridable":
        raise SystemExit("the study check simulates futility followed only")
    data_path = Path(args.data)
    data = json.loads(data_path.read_text())
    pools = sorted(k for k, v in data.items() if isinstance(v, dict) and "world_seeds" in v)
    if not pools:
        raise SystemExit("--data has no pools")
    replica = Replica(plan)
    rule = policy_rule(plan)
    scen = check_scenarios(plan)
    thetas = [float(t) for t in args.thetas.split(",")]
    results: Dict[str, Any] = {}
    joint: Dict[str, float] = {}
    for i, p in enumerate(pools):
        for j, theta in enumerate(thetas):
            cell = gate_cell(
                data[p],
                plan,
                replica,
                [rule],
                scen,
                (theta,) * 3,
                args.reps,
                7000 + 101 * i + j,
                float(args.delta_ni),
            )
            for name, row in cell["rules"][rule.name].items():
                key = f"{p}|theta={theta:g}|{name}"
                results[key] = {
                    **row,
                    "p_qualify": cell["p_qualify"],
                    "qualifying_look_distribution": cell["qualifying_look_distribution"],
                }
                if name != "no_change":
                    joint[key] = row["p_pass"]["rate"]
    threshold = CHECK_FACTOR * plan.band_alpha
    passes = bool(joint) and all(v <= threshold for v in joint.values())
    return {
        "schema": "survival-band-v2-study-check/v1",
        "config": {
            "plan_params": params,
            "delta_ni": float(args.delta_ni),
            "thetas": thetas,
            "data": str(data_path),
        },
        "data_sha256": sha256_file(data_path),
        "reps": args.reps,
        "results": results,
        "check": {
            "rule": "pooled_ni_continue",
            "acceptance": f"every joint rate P(PASS and a regression exactly at a margin) <= "
            f"{CHECK_FACTOR} x band_alpha",
            "threshold": threshold,
            "joint_rates": joint,
            "max_joint_rate": max(joint.values()) if joint else None,
            "passes": passes,
        },
    }


def part_crosscheck(args) -> Dict[str, Any]:
    """Replica vs ``run_sequential_gate`` with per-look ``plan_pooled_band_check`` verdicts."""
    from src.evaluation.sequential_gate import plan_pooled_band_check, run_sequential_gate

    plan, _ = plan_from_document(Path(args.plan_params))
    data = json.loads(Path(args.data).read_text())
    pool = data[args.pool]
    replica = Replica(plan)
    rule = policy_rule(plan)
    lv = list(plan.band_nominal_p)
    names = {"STOP_PASS": True, "FINAL_PASS": True, "FINAL_FAIL": False, "STOP_FUTILE": False}
    batches = {
        "mde_no_change": ((plan.mde,) * 3, (0.0, 0.0, 0.0)),
        "mde_pooled_at_margin": ((plan.mde,) * 3, (-plan.band_ni_margin,) * 3),
        "2mde_scripted_at_mix_margin": ((2 * plan.mde,) * 3, (0.0, -plan.band_mix_margin, 0.0)),
        "null": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    }
    out: Dict[str, Any] = {"pool": args.pool, "batches": {}, "mismatches": []}
    for b, (label, (theta, shift)) in enumerate(batches.items()):
        rng = np.random.default_rng(9000 + b)
        x, d, inc = resample(pool, rng, args.reps, plan.n_max, theta)
        tau, end = qualification(replica, plan, x, float(args.delta_ni))
        stats = LookStats(d, inc, plan.look_sizes)
        passes, stop = rule_outcome(rule, stats, tau, end, np.asarray(shift), lv)
        counts: Dict[str, int] = {}
        for r in range(args.reps):
            cand = {m: (inc[r, j] + d[r, j] + shift[j]).tolist() for j, m in enumerate(MIXES)}
            incumbent = {m: inc[r, j].tolist() for j, m in enumerate(MIXES)}
            bands = []
            for look, n in enumerate(plan.look_sizes):
                check = plan_pooled_band_check(
                    plan,
                    look,
                    {m: cand[m][:n] for m in MIXES},
                    {m: incumbent[m][:n] for m in MIXES},
                )
                bands.append(bool(check["passes"]))
            pure = run_sequential_gate(
                plan,
                {m: x[r, j].tolist() for j, m in enumerate(MIXES)},
                float(args.delta_ni),
                bands,
            )
            counts[pure["decision"]] = counts.get(pure["decision"], 0) + 1
            if names[pure["decision"]] != bool(passes[r]) or pure["look"] != int(stop[r]):
                out["mismatches"].append(f"{label}:{r}")
        out["batches"][label] = {"replicates": args.reps, "pure_decisions": counts}
    return out


def part_design_plan(args) -> Dict[str, Any]:
    """The design plan under ``pooled_ni_continue`` option 1 as a frozen plan document (the
    ``--plan-params`` input of ``study`` / ``crosscheck`` for the design pools)."""
    from research.sequential_strict_template import sequential_runner as R

    params = R.plan_parameters(
        n_max=DESIGN_N_MAX,
        mde=DESIGN_MDE,
        mixes=MIXES,
        scripted_mix="scripted",
        band_policy="pooled_ni_continue",
        paired_band=dict(R.POOLED_BAND_OPTIONS["1"]),
    )
    return R.frozen_plan_document(params)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--part",
        required=True,
        choices=("oc", "descriptive", "study", "crosscheck", "design-plan"),
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20_000)
    ap.add_argument("--gate-reps", type=int, default=20_000)
    ap.add_argument("--all-gate-pools", action="store_true")
    ap.add_argument("--data", default=str(POOLS_PATH))
    ap.add_argument("--plan-params", default=None)
    ap.add_argument("--delta-ni", default=None)
    ap.add_argument("--thetas", default=None)
    ap.add_argument("--pool", default=None)
    args = ap.parse_args(argv)
    began = time.process_time()
    if args.part == "design-plan":
        Path(args.out).write_text(
            json.dumps(part_design_plan(args), indent=1, sort_keys=True) + "\n"
        )
        print(json.dumps({"out": args.out}))
        return 0
    if args.part in ("study", "crosscheck"):
        if not (args.plan_params and args.delta_ni):
            ap.error(f"--part {args.part} needs --plan-params and --delta-ni")
    if args.part == "study":
        if not args.thetas:
            ap.error("--part study needs --thetas")
        if args.reps < CHECK_MIN_REPS:
            ap.error(f"--part study needs --reps >= {CHECK_MIN_REPS}")
        result = part_study(args)
    elif args.part == "crosscheck":
        if not args.pool:
            ap.error("--part crosscheck needs --pool")
        result = part_crosscheck(args)
    elif args.part == "descriptive":
        result = part_descriptive(args)
    else:
        result = part_oc(args)
    result["cpu_seconds"] = round(time.process_time() - began, 1)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"out": args.out, "cpu_seconds": result["cpu_seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
