#!/usr/bin/env python3
"""Skew sensitivity and resampling operating characteristics of the sequential gate.

Diagnostic (Tier 0 style).  One mix at a time, numpy only, single-threaded, fixed seeds.

* ``--part gamma``: synthetic gamma-shaped paired deltas (SD 140) at the efficacy null
  (mean 0) and at the NI null (mean -delta_NI), for skewness -2.83, -1.0, 0 and +2.83.
  Reports the per-mix any-look efficacy crossing rate (target ``alpha/3``), the look-1
  rate, the NI any-look rate (target 0.05), and the same rates for the fixed-N gate.
  It also reports the larger-first-look remedy (fractions 0.5, 0.75, 1.0).
* ``--part resample --deltas FILE``: the pre-registered check of the governance
  amendment.  ``FILE`` is a JSON list of saved paired deltas (screen or pilot data of the
  study).  They are centred at the efficacy null and at the NI null and resampled with
  replacement under the study's frozen ``--n-max``, ``--delta-ni`` and ``--fractions``.
  ``passes`` applies the pre-registered acceptance thresholds.
* ``--part bands``: the adopted ``block_at_stop`` band policy against two rejected rules.
  One mix; the stop look is its first efficacy crossing; per-world candidate survival is
  normal with correlation ``rho`` to the mass delta, its true mean ``D`` final-N standard
  errors below the band edge
  (``D < 0``: inside the band).  Reports P(band passes | efficacy qualified) under
  ``block_at_stop``, under delaying on band failure with the same interim margin
  ``z * sd * (1/sqrt(n_k) - 1/sqrt(N))`` (rejected), under the first draft's re-check at
  every look with the point estimate (withdrawn), and under the fixed-N gate.

Output is JSON on stdout (or ``--out``).  No game is run and no record is read except
``--deltas``.
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
    os.environ[_name] = "1"  # single-threaded numpy, set before numpy is imported

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable, Dict  # noqa: E402

import numpy as np  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.evaluation.sequential_gate import sequential_gate_plan  # noqa: E402
from src.scripts.eval_stats import student_t_isf  # noqa: E402

N_MAX = 243
SD = 140.0
DELTA_NI = 3.5
EFFICACY_LIMIT = 1.2 * 0.05 / 3  # pre-registered acceptance thresholds (amendment)
NI_LIMIT = 0.06
CHUNK = 25_000
Z_BAND = 1.645  # default SequentialGatePlan.band_margin_z
N_SIZES = (61, 122, 183, 243)  # default look sizes at N_MAX = 243


def crossing_rates(
    sampler: Callable[[np.random.Generator, int], np.ndarray],
    plan,
    *,
    reps: int,
    seed: int,
    centre: float,
    ni: bool,
) -> Dict[str, Any]:
    """Any-look, look-1 and fixed-N crossing rates of one mix for ``plan``.

    ``sampler(rng, k)`` returns ``(k, n_max)`` deltas with mean 0; ``centre`` is added.
    Efficacy crosses when ``t >= t_crit``; NI when ``mean - t_crit * se > -DELTA_NI``.
    """
    sizes = plan.look_sizes
    nominal = plan.ni_nominal_p if ni else plan.efficacy_nominal_p
    crit = [student_t_isf(p, n - 1) for p, n in zip(nominal, sizes)]
    fixed_crit = student_t_isf(0.05 if ni else plan.efficacy_alpha_per_mix, sizes[-1] - 1)
    rng = np.random.default_rng(seed)
    any_look = first = fixed = 0
    done = 0
    while done < reps:
        k = min(CHUNK, reps - done)
        x = sampler(rng, k) + centre
        s1 = np.cumsum(x, axis=1)
        s2 = np.cumsum(x * x, axis=1)
        hit = np.zeros(k, dtype=bool)
        for look, n in enumerate(sizes):
            mean = s1[:, n - 1] / n
            sd = np.sqrt(np.maximum(s2[:, n - 1] - n * mean * mean, 0.0) / (n - 1))
            se = sd / math.sqrt(n)
            now = (mean - crit[look] * se > -DELTA_NI) if ni else (mean >= crit[look] * se)
            if look == 0:
                first += int(now.sum())
            if look == len(sizes) - 1:
                fixed_now = (
                    (mean - fixed_crit * se > -DELTA_NI) if ni else (mean >= fixed_crit * se)
                )
                fixed += int(fixed_now.sum())
            hit |= now
        any_look += int(hit.sum())
        done += k
    return {
        "reps": reps,
        "any_look": any_look / reps,
        "look_1": first / reps,
        "look_1_nominal": nominal[0],
        "fixed_n": fixed / reps,
        "target": 0.05 if ni else plan.efficacy_alpha_per_mix,
    }


def gamma_sampler(skew: float) -> Callable[[np.random.Generator, int], np.ndarray]:
    """Mean-0, SD-140 deltas with the given skewness (0 means normal)."""
    if skew == 0.0:
        return lambda rng, k: rng.standard_normal((k, N_MAX)) * SD
    shape = (2.0 / abs(skew)) ** 2
    sign = 1.0 if skew > 0 else -1.0

    def draw(rng: np.random.Generator, k: int) -> np.ndarray:
        g = rng.gamma(shape, 1.0, size=(k, N_MAX))
        return sign * (g - shape) / math.sqrt(shape) * SD

    return draw


def part_gamma(reps: int) -> Dict[str, Any]:
    plans = {
        "default_0.25_0.5_0.75_1": sequential_gate_plan(N_MAX, mde=30.0),
        "remedy_0.5_0.75_1": sequential_gate_plan(N_MAX, mde=30.0, fractions=(0.5, 0.75, 1.0)),
    }
    out: Dict[str, Any] = {}
    for p_index, (label, plan) in enumerate(plans.items()):
        rows = {}
        for s_index, skew in enumerate((-2.83, -1.0, 0.0, 2.83)):
            sampler = gamma_sampler(skew)
            seed = 1000 + 100 * p_index + 10 * s_index
            rows[f"skew_{skew:+.2f}"] = {
                "efficacy_null": crossing_rates(
                    sampler, plan, reps=reps, seed=seed, centre=0.0, ni=False
                ),
                "ni_null": crossing_rates(
                    sampler, plan, reps=reps, seed=seed + 1, centre=-DELTA_NI, ni=True
                ),
            }
        out[label] = {"look_sizes": list(plan.look_sizes), "rows": rows}
    return out


def part_resample(
    path: str, reps: int, n_max: int, delta_ni: float, fractions: tuple
) -> Dict[str, Any]:
    global N_MAX, DELTA_NI
    N_MAX, DELTA_NI = n_max, delta_ni
    with open(path, encoding="utf-8") as handle:
        data = np.asarray([float(v) for v in json.load(handle)], dtype=float)
    if data.size < 20 or not np.all(np.isfinite(data)):
        raise SystemExit("need at least 20 finite saved deltas")
    centred = data - data.mean()
    plan = sequential_gate_plan(n_max, mde=30.0, fractions=fractions)

    def sampler(rng: np.random.Generator, k: int) -> np.ndarray:
        return rng.choice(centred, size=(k, n_max), replace=True)

    eff = crossing_rates(sampler, plan, reps=reps, seed=7, centre=0.0, ni=False)
    ni = crossing_rates(sampler, plan, reps=reps, seed=8, centre=-delta_ni, ni=True)
    skew = float(np.mean(centred**3) / np.std(centred) ** 3)
    return {
        "n_saved": int(data.size),
        "look_sizes": list(plan.look_sizes),
        "sample_skewness": skew,
        "efficacy_null": eff,
        "ni_null": ni,
        "limits": {"efficacy_any_look": EFFICACY_LIMIT, "ni_any_look": NI_LIMIT},
        "passes": bool(eff["any_look"] <= EFFICACY_LIMIT and ni["any_look"] <= NI_LIMIT),
    }


def _band_chunk(rng, k: int, delta: float, rho: float, d: float, crit, fixed_crit) -> np.ndarray:
    """Counts [qualified, block, crossed, defer, recheck, fixed qualified, fixed pass]."""
    zx = rng.standard_normal((k, N_MAX))
    y = rho * zx + math.sqrt(1 - rho * rho) * rng.standard_normal((k, N_MAX))
    y = y - d / math.sqrt(N_MAX)  # band edge at 0, sigma_y = 1
    x = zx * SD + delta
    sx, sxx, sy, syy = (np.cumsum(a, axis=1) for a in (x, x * x, y, y * y))
    stopped, block, recheck, defer, crossed = (np.zeros(k, dtype=bool) for _ in range(5))
    for look, n in enumerate(N_SIZES):
        mx = sx[:, n - 1] / n
        sdx = np.sqrt(np.maximum(sxx[:, n - 1] - n * mx * mx, 0) / (n - 1))
        my = sy[:, n - 1] / n
        sdy = np.sqrt(np.maximum(syy[:, n - 1] - n * my * my, 0) / (n - 1))
        margin = Z_BAND * sdy * (1 / math.sqrt(n) - 1 / math.sqrt(N_MAX))
        qual = mx >= crit[look] * sdx / math.sqrt(n)
        crossed |= qual
        stop_now = qual & ~stopped
        block |= stop_now & (my >= margin)  # adopted: judged once, at the qualifying look
        recheck |= crossed & (my >= 0.0)  # first draft: point estimate at every look
        defer |= crossed & (my >= margin)  # rejected: delay on failure, with margin
        stopped |= stop_now
    fixed_q = mx >= fixed_crit * sdx / math.sqrt(N_MAX)
    parts = (stopped, block, crossed, defer, recheck, fixed_q, fixed_q & (my >= 0.0))
    return np.array([int(a.sum()) for a in parts])


def part_bands(reps: int) -> Dict[str, Any]:
    plan = sequential_gate_plan(N_MAX, mde=30.0)
    crit = [student_t_isf(p, n - 1) for p, n in zip(plan.efficacy_nominal_p, N_SIZES)]
    fixed_crit = student_t_isf(plan.efficacy_alpha_per_mix, N_MAX - 1)
    rows: Dict[str, Any] = {}
    seed = 500
    for delta in (30.0, 130.0):
        for rho in (0.0, 0.5):
            for d in (-2.0, 0.0, 1.0, 1.645, 2.5):
                seed += 1
                rng = np.random.default_rng(seed)
                c = np.zeros(7, dtype=np.int64)
                for done in range(0, reps, CHUNK):
                    k = min(CHUNK, reps - done)
                    c += _band_chunk(rng, k, delta, rho, d, crit, fixed_crit)
                rows[f"delta_{delta:g}_rho_{rho:g}_D_{d:g}"] = {
                    "p_qualified": float(c[0] / reps),
                    "block_at_stop": float(c[1] / max(c[0], 1)),
                    "defer_with_margin": float(c[3] / max(c[2], 1)),
                    "recheck_every_look": float(c[4] / max(c[2], 1)),
                    "fixed_n": float(c[6] / max(c[5], 1)),
                }
    return {"z_band": Z_BAND, "reps": reps, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--part", choices=("gamma", "resample", "bands"), required=True)
    parser.add_argument("--reps", type=int, default=300_000)
    parser.add_argument("--deltas", help="JSON list of saved paired deltas (resample)")
    parser.add_argument("--n-max", type=int, default=243)
    parser.add_argument("--delta-ni", type=float, default=3.5)
    parser.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--out")
    args = parser.parse_args()
    start = time.process_time()
    if args.part == "gamma":
        result = part_gamma(args.reps)
    elif args.part == "bands":
        result = part_bands(args.reps)
    else:
        if not args.deltas:
            parser.error("--part resample needs --deltas")
        fractions = tuple(float(f) for f in args.fractions.split(","))
        result = part_resample(args.deltas, args.reps, args.n_max, args.delta_ni, fractions)
    result = {"part": args.part, "result": result, "cpu_seconds": time.process_time() - start}
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
