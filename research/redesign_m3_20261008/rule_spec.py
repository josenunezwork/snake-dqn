"""M3 Tier-1 sequential Phase R rule spec and plan (pinned by tests/test_m3_rule_spec.py).

The FRP-v4 rule in sequential form (``research/sequential_phase_r/example_rules.frp_v4_like``)
for the M3 study: seeds = the 5 M3 training seeds, candidate cell "primary" = M3 student
(20M transitions) + v8 vs the incumbent frp3-s12 + v8, guard cell = the same candidate vs
champion_a5 + v8 on the first seed's bank, cap 600, NO_GO opted in at G = 50 (owner,
2026-10-08), N = 32 worlds per (seed, mix), default fractions / spendings / margins of the
ratified amendment.
"""

from __future__ import annotations

from research.sequential_phase_r.example_rules import frp_v4_like
from src.evaluation.sequential_phase_r import make_plan

STUDY_ID = "redesign-m3-phase-r"
TRAINING_SEEDS = (0, 1, 2, 3, 4)
HERO = "M3-ego2s-b@20M+v8"
N_WORLDS = 32
CAP = 600
NO_GO_G = 50.0


def m3_rule() -> dict:
    return frp_v4_like(seeds=TRAINING_SEEDS, hero=HERO, cap=CAP, no_go=NO_GO_G, study_id=STUDY_ID)


def m3_plan():
    return make_plan(N_WORLDS, m3_rule())
