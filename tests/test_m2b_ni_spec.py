"""The pre-registered M2b NI rule: same rule as M2, fresh disjoint worlds."""

from __future__ import annotations

from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b
from research.redesign_scope_20261007 import grid_h5000_identity as gid


def test_fresh_worlds_disjoint_from_everything():
    seeds = set(ni_spec_m2b.ni_seeds())
    assert len(seeds) == 48
    others = set(ni_spec.ni_seeds()) | set(gid.world_seeds(8))
    for r in (0, 1, 2, 9, 10, 11, 12, 50, 51):
        others |= set(ni_spec.distill_seeds(3000, round_index=r))
    assert not seeds & others


def test_same_rule_as_m2():
    assert ni_spec_m2b.MARGIN == ni_spec.MARGIN == -15.0
    assert ni_spec_m2b.CONFIDENCE == ni_spec.CONFIDENCE == 0.90
    assert ni_spec_m2b.WORLDS_PER_MIX == 48 and ni_spec_m2b.MIXES == ni_spec.MIXES
    seeds = ni_spec_m2b.ni_seeds()
    base = {m: {s: 100.0 + (i % 7) for i, s in enumerate(seeds)} for m in ni_spec.MIXES}
    worse = {m: {s: v - 30.0 for s, v in base[m].items()} for m in ni_spec.MIXES}
    assert ni_spec_m2b.decide(base, base)["verdict"] == "PASS"
    assert ni_spec_m2b.decide(worse, base)["verdict"] == "FAIL"
