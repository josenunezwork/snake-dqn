"""M3 world namespaces are disjoint from every world any earlier study or probe used."""

from __future__ import annotations

from research.apex_safety_20260926 import dev_screen
from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b
from research.redesign_m3_20261008 import student_record_parity
from research.redesign_scope_20261007 import grid_h5000_identity as gid

TRAIN = "redesign-m3-train/v1"
PHASE_R = "redesign-m3-phase-r/v1"
# Worlds actually used per distillation / probe round index (data manifests, probes).
USED = {0: 416, 1: 208, 2: 208, 9: 32, 10: 416, 11: 208, 12: 208, 50: 16, 51: 16, 52: 16}


def _used_elsewhere() -> set:
    used = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds()) | set(gid.world_seeds(8))
    used |= set(dev_screen.screen_seeds(3, student_record_parity.DOMAIN, "worlds"))
    for r, n in USED.items():
        used |= set(ni_spec.distill_seeds(n, round_index=r))
    return used


def test_training_and_phase_r_worlds_are_fresh():
    used = _used_elsewhere()
    train, bank = set(), set()
    for s in range(5):
        # M3-B: 20M transitions / (64 worlds x 5000 frames) = 63 segments -> 4032 worlds.
        train |= set(dev_screen.screen_seeds(4100, TRAIN, f"seed{s}"))
        bank |= set(dev_screen.screen_seeds(32, PHASE_R, f"seed{s}"))
    assert not train & used
    assert not bank & used
    assert not train & bank
    assert len(bank) == 160
