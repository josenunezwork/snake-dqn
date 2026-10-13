"""M4 Tier-1 sequential Phase R rule specs (DRAFT; pinned by tests/test_m4_rule_spec.py).

Two studies, both the FRP-v4 rule in sequential form (``example_rules.frp_v4_like``; the
ratified ``sequential-phase-r-obf-hk-v1`` method), each with NO_GO opted in at G = 50, as
M3-C (``research/redesign_m3c_20261008/rule_spec_m3c.py``), with M4-A's THREE training seeds:

* **Primary (the redesign goal): ``redesign-m4-phase-r-nv``** -- hero = the selected M4 arm's
  student WITHOUT a veto (``ego2s_mask="advisory"``) vs the incumbent frp3-s12 + v8 on every
  seed's bank; guard = the same student vs champion_a5 + v8 on seed 0's bank.
* **Secondary: ``redesign-m4-phase-r-v8``** -- hero = the selected arm's student + v8 vs the
  same incumbent; same guard construction.

Seeds = (0, 1, 2), N = 32 worlds per (seed, mix), cap 600, NO_GO G = 50.

**One departure from ``frp_v4_like``:** its positive-seeds clause (4) is "4 of 5" at five
seeds and otherwise "6" (the 8-seed FRP-v5 value), which three seeds can never meet; here it
is **3 of 3** (every seed positive; FRP-v4 4/5 = 80 %, FRP-v5 6/8 = 75 %, and 2/3 = 67 % would
be looser than both). Governance condition 7 (a different number of seeds) requires the OC
re-run for this rule: ``oc_m4.py``.

Banks: ``redesign-m4-phase-r-nv/v1`` and ``redesign-m4-phase-r-v8/v1``, key ``seed<s>``
(disjoint from every training, probe, M3 and M3-C namespace; tested).
"""

from __future__ import annotations

from research.sequential_phase_r.example_rules import frp_v4_like
from src.evaluation.sequential_phase_r import make_plan

TRAINING_SEEDS = (0, 1, 2)
N_WORLDS = 32
CAP = 600
NO_GO_G = 50.0
MIN_POSITIVE_SEEDS = 3  # of 3 (see the module docstring)
PRIMARY_ID = "redesign-m4-phase-r-nv"
SECONDARY_ID = "redesign-m4-phase-r-v8"
HERO_NV = "M4-selected-noveto"
HERO_V8 = "M4-selected+v8"
# DRAFT pins (re-pinned or confirmed at ratification; tests/test_m4_rule_spec.py)
PRIMARY_RULE_SHA256 = "67dfe274de90858e4e0cd8d2e3b9bd6e81873bae919e7c838a513a6aeaedf08c"
SECONDARY_RULE_SHA256 = "e842a5237f756d17780aaa899e41c4dde80e4a0314df5728b03d6816cae86fe8"
PRIMARY_PLAN_SHA256 = "c8f3595f029d11c91ccbddf69087ca2cc32bb41544a02610df7a411f4da7c36a"
SECONDARY_PLAN_SHA256 = "c4d8095d3e358b8703c32b713abd9e3899318af54938e1bcc58ef058ee907a39"
BANK_NS = {PRIMARY_ID: "redesign-m4-phase-r-nv/v1", SECONDARY_ID: "redesign-m4-phase-r-v8/v1"}


def _rule(hero: str, study_id: str) -> dict:
    rule = frp_v4_like(seeds=TRAINING_SEEDS, hero=hero, cap=CAP, no_go=NO_GO_G, study_id=study_id)
    hits = [c for c in rule["go_clauses"] if c["name"] == "c4_positive_seeds"]
    if len(hits) != 1:
        raise RuntimeError("frp_v4_like changed: expected exactly one c4_positive_seeds clause")
    hits[0]["min"] = MIN_POSITIVE_SEEDS
    return rule


def primary_rule() -> dict:
    return _rule(HERO_NV, PRIMARY_ID)


def secondary_rule() -> dict:
    return _rule(HERO_V8, SECONDARY_ID)


def primary_plan():
    return make_plan(N_WORLDS, primary_rule())


def secondary_plan():
    return make_plan(N_WORLDS, secondary_rule())
