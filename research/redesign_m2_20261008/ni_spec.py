"""PRE-REGISTERED M2 non-inferiority check (redesign scope, doc section 15.3).

Committed before any ego2s student exists (owner decision 2026-10-08). Development
check, not gate evidence: it decides only whether the distilled student is good enough
to carry into M3.

Hypothesis
    The distilled ego2s student WITHOUT a veto is non-inferior to frp3-s12 WITHOUT a
    veto on dev H5000 mass.

Design
    * Profile ``promotion-v2-watch-rect`` at H5000, the pinned deployment config, the
      strict balanced rosters (``dev_screen._design_rows``) of the frozen / scripted /
      mixed mixes.
    * ``WORLDS_PER_MIX`` = 48 fresh worlds per mix from namespace ``NAMESPACE`` (disjoint
      from every distillation seed and from the M1 identity seeds), the same worlds for
      both arms: 144 paired worlds.
    * Engine: ``run_simd_eval(sim_engine="grid", vector61=True)`` (M1: bit-identical to
      BatchSim, itself bit-identical to the live game). frp3-s12 runs vector61 rowwise
      (bit-exact forwards); the student runs the ego2s numba featurizer and a batched CPU
      forward (one torch thread), greedy over the resolved action mask.

Decision rule (PASS iff)
    ``d_i = mass_student_i - mass_frp3_i`` over the 144 paired worlds;
    ``LB = mean(d) - t_{0.90, n-1} * sd(d) / sqrt(n)`` (one-sided 90%, Student t);
    PASS iff ``LB > MARGIN`` with ``MARGIN = -15`` (about 13% of the measured frp3-s12
    no-veto mean of 113.0 on the M1 identity worlds). Every planned world must complete.

Reported, not gated
    Per-mix deltas; student + v8 vs frp3-s12 + v8 (I) on the same worlds; death causes and
    peak lengths; the v8 veto activation counters on the student.

Student under test (fixed before the check runs)
    The checkpoint produced by the round-2 (final DAgger round) fit, selected inside that
    fit by held-out v8-action agreement (held-out = distillation worlds, never NI worlds).
    Its sha256 is written to the check's intent file before the first NI episode runs. No
    tuning on, and no re-run of, the NI worlds; if the check is run again for any reason,
    both runs are reported.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

SCHEMA = "redesign-m2-ni/v1"
NAMESPACE = "redesign-m2-ni/v1"
NAMESPACE_KEY = "worlds"
DISTILL_NAMESPACE = "redesign-m2-distill/v1"
WORLDS_PER_MIX = 48
MIXES = ("frozen", "scripted", "mixed")
MARGIN = -15.0
CONFIDENCE = 0.90
HORIZON = 5000
PROFILE_NAME = "promotion-v2-watch-rect"
FRP3_S12 = (
    "frp3_m3_s12_u60000_20261005.pth",
    "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
)
BASELINE_NO_VETO_MEAN_M1 = 113.0  # frp3-s12, no veto, 24 M1 identity worlds (doc 14.2)


def ni_seeds() -> List[int]:
    """The 48 pre-registered NI world seeds (shared by all three mixes)."""
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.screen_seeds(WORLDS_PER_MIX, NAMESPACE, NAMESPACE_KEY)


def distill_seeds(count: int, offset: int = 0) -> List[int]:
    """Distillation world seeds (disjoint from :func:`ni_seeds`; checked by tests)."""
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.screen_seeds(offset + count, DISTILL_NAMESPACE, NAMESPACE_KEY)[offset:]


def lower_bound(deltas: Sequence[float], confidence: float = CONFIDENCE) -> Dict[str, Any]:
    """One-sided Student-t lower bound of the mean paired delta."""
    from src.scripts.eval_stats import student_t_isf

    n = len(deltas)
    if n < 2:
        raise ValueError("need at least two paired worlds")
    m = sum(deltas) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in deltas) / (n - 1))
    t = student_t_isf(1.0 - confidence, n - 1)
    return {"n": n, "mean": m, "sd": sd, "t": t, "lb": m - t * sd / math.sqrt(n)}


def decide(
    student: Mapping[str, Mapping[int, float]], baseline: Mapping[str, Mapping[int, float]]
) -> Dict[str, Any]:
    """Apply the pre-registered rule to ``{mix: {world_seed: mass_integral}}`` per arm."""
    seeds = ni_seeds()
    deltas: List[float] = []
    per_mix: Dict[str, Any] = {}
    complete = True
    for mix in MIXES:
        s_mix, b_mix = student.get(mix, {}), baseline.get(mix, {})
        mix_deltas = []
        for seed in seeds:
            if seed not in s_mix or seed not in b_mix:
                complete = False
                continue
            mix_deltas.append(float(s_mix[seed]) - float(b_mix[seed]))
        deltas.extend(mix_deltas)
        per_mix[mix] = lower_bound(mix_deltas) if len(mix_deltas) >= 2 else {"n": len(mix_deltas)}
    pooled = lower_bound(deltas)
    passed = complete and len(deltas) == WORLDS_PER_MIX * len(MIXES) and pooled["lb"] > MARGIN
    return {
        "schema_version": SCHEMA,
        "margin": MARGIN,
        "confidence": CONFIDENCE,
        "complete": complete,
        "pooled": pooled,
        "per_mix": per_mix,
        "verdict": "PASS" if passed else ("FAIL" if complete else "INCOMPLETE"),
    }
