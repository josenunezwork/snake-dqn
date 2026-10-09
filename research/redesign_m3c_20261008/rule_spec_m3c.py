"""M3-C Tier-1 sequential Phase R rule specs (DRAFT; pinned by tests/test_m3c_rule_spec.py).

Two studies, both the FRP-v4 rule in sequential form (``example_rules.frp_v4_like``; the
ratified ``sequential-phase-r-obf-hk-v1`` method), each with NO_GO opted in at G = 50:

* **Primary (the redesign goal): ``redesign-m3c-phase-r-nv``** -- hero = the selected M3-C
  arm's student WITHOUT a veto (``ego2s_mask="advisory"``, the live selection rule) vs the
  incumbent frp3-s12 + v8 on every seed's bank; guard = the same student vs champion_a5 + v8
  on seed 0's bank.
* **Secondary: ``redesign-m3c-phase-r-v8``** -- hero = the selected arm's student + v8 vs the
  same incumbent; same guard construction.

Seeds = the 5 training seeds of the selected arm, N = 32 worlds per (seed, mix), cap 600.
The hero names are labels; candidate checkpoints and their sha256 enter ``plan.json`` before
the first shard. Banks: ``redesign-m3c-phase-r-nv/v1`` and ``redesign-m3c-phase-r-v8/v1``,
key ``seed<s>`` (disjoint from every training, probe and M3 namespace; tested).
"""

from __future__ import annotations

from research.sequential_phase_r.example_rules import frp_v4_like
from src.evaluation.sequential_phase_r import make_plan

TRAINING_SEEDS = (0, 1, 2, 3, 4)
N_WORLDS = 32
CAP = 600
NO_GO_G = 50.0
PRIMARY_ID = "redesign-m3c-phase-r-nv"
SECONDARY_ID = "redesign-m3c-phase-r-v8"
HERO_NV = "M3C-selected-noveto"
HERO_V8 = "M3C-selected+v8"
# DRAFT pins (re-pinned or confirmed at ratification; tests/test_m3c_rule_spec.py)
PRIMARY_RULE_SHA256 = "5103b4133f819e6dea097487ed5f0192aa2cc9ce3ae75f40d595afa63d84b6c0"
SECONDARY_RULE_SHA256 = "1f50018e803abd501ed0783890bbed9c74f1e850eb0387fbbf0b4eef63b97488"
PRIMARY_PLAN_SHA256 = "70e4b4d78c9b77b411f36479b28a3f183a8fc6cc438c0d1c8368fc65e92350f2"
SECONDARY_PLAN_SHA256 = "4a71565d16693c0ae599ace7591770226f2636594f7e394c1be1d4ae12b8a211"
BANK_NS = {PRIMARY_ID: "redesign-m3c-phase-r-nv/v1", SECONDARY_ID: "redesign-m3c-phase-r-v8/v1"}


def primary_rule() -> dict:
    return frp_v4_like(
        seeds=TRAINING_SEEDS, hero=HERO_NV, cap=CAP, no_go=NO_GO_G, study_id=PRIMARY_ID
    )


def secondary_rule() -> dict:
    return frp_v4_like(
        seeds=TRAINING_SEEDS, hero=HERO_V8, cap=CAP, no_go=NO_GO_G, study_id=SECONDARY_ID
    )


def primary_plan():
    return make_plan(N_WORLDS, primary_rule())


def secondary_plan():
    return make_plan(N_WORLDS, secondary_rule())
