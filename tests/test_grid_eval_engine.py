"""``run_simd_eval(sim_engine="grid")`` == the default BatchSim engine (redesign M1).

The full gate-world H5000 check (frp3-s12, vector61, v8) is the research harness
``research/redesign_scope_20261007/grid_h5000_identity.py``; this is its CI-sized
counterpart on the legacy (profile-free) path with scripted agents, plus the harness's
pure comparison helpers.
"""

from __future__ import annotations

import json

import pytest

from research.redesign_scope_20261007 import grid_h5000_identity as gid
from src.simd_env import eval_engine as ee
from src.simd_env.grid_sim import GridBatchSim


@pytest.mark.parametrize("hero", ["greedy_food", "random_safe"])
def test_grid_engine_records_equal_batch_engine(hero):
    opponents = [("scripted", "greedy_food"), ("scripted", "random_safe")] * 2 + [
        ("scripted", "greedy_food")
    ]
    seeds = [11, 12, 13]
    kwargs = dict(frames=400, seeds=seeds)
    batch = ee.run_simd_eval(("scripted", hero), opponents, sim_engine="batch", **kwargs)
    grid = ee.run_simd_eval(("scripted", hero), opponents, sim_engine="grid", **kwargs)
    assert json.dumps(batch, sort_keys=True) == json.dumps(grid, sort_keys=True)
    assert any(r["probes"]["food_eaten"] > 0 for r in batch)


def test_grid_engine_class_is_terminal_hero_grid_sim():
    cls = ee._terminal_hero_sim_class("grid")
    assert issubclass(cls, GridBatchSim) and issubclass(cls, ee._TerminalHeroMixin)
    assert ee._terminal_hero_sim_class("grid") is cls
    assert ee._terminal_hero_sim_class("batch") is ee._TerminalHeroBatchSim
    with pytest.raises(ValueError, match="sim_engine"):
        ee._terminal_hero_sim_class("live")
    with pytest.raises(ValueError, match="sim_engine"):
        ee.run_simd_eval(("scripted", "greedy_food"), [], 10, [1], sim_engine="gpu")


def test_identity_harness_helpers():
    assert gid.strip_timing({"a": 1, "wall_seconds": 2, "b": [{"x_seconds": 1, "y": 2}]}) == {
        "a": 1,
        "b": [{"y": 2}],
    }
    assert gid.diff_paths({"a": 1, "b": [1, 2]}, {"a": 1.0, "b": [1, 2, 3]}) == [
        ("a", 1, 1.0),
        ("b[len]", 2, 3),
    ]
    record = {"mass_integral": 1.5, "veto_diagnostics": {"t_seconds": 3.0, "n": 1}}
    same = {"records": [record], "frame_digests": [["d1", "d2"]]}
    other = {"records": [record], "frame_digests": [["d1", "dX"]]}
    rows = gid.compare_worlds(same, same, [7])
    assert rows[0]["identical"] and rows[0]["first_differing_frame"] is None
    rows = gid.compare_worlds(same, other, [7])
    assert not rows[0]["identical"] and rows[0]["first_differing_frame"] == 1
    assert rows[0]["record_identical"]
