"""M4 Tier-1 sequential Phase R rule specs (pinned by tests/test_m4_rule_spec.py).

Two studies, both the FRP-v4 rule in sequential form (``example_rules.frp_v4_like``; the
ratified ``sequential-phase-r-obf-hk-v1`` method) UNCHANGED, each with NO_GO opted in at
G = 50, exactly as M3-C (``research/redesign_m3c_20261008/rule_spec_m3c.py``), with the
selected M4 arm's FIVE training seeds (0-2 from the M4-A pod, 3-4 trained on the Mac with the
same commit and flags; PREREGISTRATION_M4.md):

* **Primary (the redesign goal): ``redesign-m4-phase-r-nv``** -- hero = the selected M4 arm's
  student WITHOUT a veto (``ego2s_mask="advisory"``) vs the incumbent frp3-s12 + v8 on every
  seed's bank; guard = the same student vs champion_a5 + v8 on seed 0's bank.
* **Secondary: ``redesign-m4-phase-r-v8``** -- hero = the selected arm's student + v8 vs the
  same incumbent; same guard construction.

Seeds = (0, 1, 2, 3, 4), N = 32 worlds per (seed, mix), cap 600, NO_GO G = 50. Five seeds is
the seed count frp_v4_like was characterised at, so no OC re-run is required.

Banks: ``redesign-m4-phase-r-nv/v1`` and ``redesign-m4-phase-r-v8/v1``, key ``seed<s>``
(disjoint from every training, probe, M3 and M3-C namespace; tested).
"""

from __future__ import annotations

from research.sequential_phase_r.example_rules import frp_v4_like
from src.evaluation.sequential_phase_r import make_plan

TRAINING_SEEDS = (0, 1, 2, 3, 4)
N_WORLDS = 32
CAP = 600
NO_GO_G = 50.0
PRIMARY_ID = "redesign-m4-phase-r-nv"
SECONDARY_ID = "redesign-m4-phase-r-v8"
HERO_NV = "M4-selected-noveto"
HERO_V8 = "M4-selected+v8"
# Pins (tests/test_m4_rule_spec.py)
PRIMARY_RULE_SHA256 = "f614b80639e1b9b29f32a08ad38426e8b87644d5720cfe54a89d943a66f5e6b4"
SECONDARY_RULE_SHA256 = "e53691a6907e37294f9b720e7959d3e13650539f1501dd0a78d36bf92f865c46"
PRIMARY_PLAN_SHA256 = "72097708d03e81b4d4b56bc2ea013df4736b9b9f6c0c727ef2a1bb938f6f42d2"
SECONDARY_PLAN_SHA256 = "85d20b3a67de87ad24fbeca9cd4c2f3fe9888e14c5bb2e6fab30c6a53d22b5ea"
BANK_NS = {PRIMARY_ID: "redesign-m4-phase-r-nv/v1", SECONDARY_ID: "redesign-m4-phase-r-v8/v1"}


def _rule(hero: str, study_id: str) -> dict:
    return frp_v4_like(seeds=TRAINING_SEEDS, hero=hero, cap=CAP, no_go=NO_GO_G, study_id=study_id)


def primary_rule() -> dict:
    return _rule(HERO_NV, PRIMARY_ID)


def secondary_rule() -> dict:
    return _rule(HERO_V8, SECONDARY_ID)


def primary_plan():
    return make_plan(N_WORLDS, primary_rule())


def secondary_plan():
    return make_plan(N_WORLDS, secondary_rule())
