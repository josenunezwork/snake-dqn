"""PRE-REGISTERED M2b non-inferiority check (owner decision 2026-10-08, "Option 1").

Committed before any M2b data or ego2s-b student exists. Identical rule to M2
(:mod:`ni_spec`: margin -15, one-sided 90% Student-t LB of the pooled paired delta,
48 worlds per mix, frp3-s12 vs the student, both WITHOUT a veto, dev H5000, profile
``promotion-v2-watch-rect``, grid engine), on FRESH worlds: namespace
``redesign-m2b-ni/v1`` (disjoint from the M2 NI worlds, every distillation round, the
probe worlds and the M1 identity worlds; tested). The M2 FAIL stays on record.

Student under test: the ego2s-b student from the last DAgger round's fit (data rounds
10, 11, 12 = M2b round 0, 1, 2), selected inside that fit by held-out v8-action agreement;
its sha256 goes into the check's intent file before the first NI episode. No probe-based
stopping: the student goes to the check whatever any probe shows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from research.redesign_m2_20261008 import ni_spec

SCHEMA = "redesign-m2b-ni/v1"
NAMESPACE = "redesign-m2b-ni/v1"
NAMESPACE_KEY = ni_spec.NAMESPACE_KEY
WORLDS_PER_MIX = ni_spec.WORLDS_PER_MIX
MIXES = ni_spec.MIXES
MARGIN = ni_spec.MARGIN
CONFIDENCE = ni_spec.CONFIDENCE
HORIZON = ni_spec.HORIZON
FRP3_S12 = ni_spec.FRP3_S12
DATA_ROUNDS = (10, 11, 12)
PROBE_ROUND_INDEX = 51


def ni_seeds() -> List[int]:
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.screen_seeds(WORLDS_PER_MIX, NAMESPACE, NAMESPACE_KEY)


def decide(
    student: Mapping[str, Mapping[int, float]], baseline: Mapping[str, Mapping[int, float]]
) -> Dict[str, Any]:
    """The M2 rule (:func:`ni_spec.decide`) on the M2b worlds."""
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
        per_mix[mix] = (
            ni_spec.lower_bound(mix_deltas) if len(mix_deltas) >= 2 else {"n": len(mix_deltas)}
        )
    pooled = ni_spec.lower_bound(deltas) if len(deltas) >= 2 else {"n": len(deltas)}
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
