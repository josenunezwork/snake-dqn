"""Independent collision-view inverse checks; no fits or study outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from research.bl_residual.factory_masks import advisory_from_row
from src.simd_env.batch_sim import CARDINAL, BatchSimConfig
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_scenarios import inject_serpentines
from src.simd_env.grid_sim import GridBatchSim


def _fixture(heading, phase, pending, kind, *, jit=False):
    sim = GridBatchSim(
        BatchSimConfig(num_envs=1, num_snakes=2, max_capacity=1200, initial_food=0),
        seeds=[991],
        train_mode=False,
        jit=jit,
    )
    ahead = CARDINAL[heading]
    right = np.array([-ahead[1], ahead[0]])
    origin = np.array([20, 20])
    if kind == "wall":
        origin = np.array([(20, 0), (144, 20), (20, 82), (0, 20)][heading])
    hero = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)]
    enemy = [(5, -5), (5, -4)]
    if kind == "wall":
        enemy = [(5, 5), (5, 6)]
    if kind == "head-one":
        enemy = [(0, -1)]
    elif kind == "tail":
        enemy = [(0, -2), (0, -1)]
    elif kind == "enclosed":
        enemy = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (2, 0)]
    elif kind == "short":
        hero = [(0, 0)]
    elif kind == "overlap":
        enemy = [(2, 0), (1, 0)]
    for slot, cells in enumerate((hero, enemy)):
        coords = [origin + right * x - ahead * y for x, y in cells]
        n = len(coords)
        sim.bodies[0, slot] = 0
        sim.bodies[0, slot, :n] = np.array(coords[::-1])
        sim.head_ptr[0, slot] = n - 1
        sim.seg_count[0, slot] = n
        sim.length[0, slot] = n + (pending if slot == 0 else 0)
        sim.alive[0, slot] = True
        sim.direction[0, slot] = heading
        sim.boost_frames[0, slot] = phase
    sim._rebuild_env_grids(0)
    sim._refresh_action_masks(np.ones(1, bool))
    return sim


def _row(sim, backend):
    # The inverse uses only draft's six local planes and first twelve scalars.
    # Appending zero bv8 diagnostics avoids invoking an unrelated flood kernel.
    obs = build_ego_raster(sim, EgoRasterConfig(), backend=backend, rows=np.array([[0, 0]]))
    return {
        "length": int(sim.length[0, 0]),
        "local": np.concatenate([obs["local"][0], np.zeros((1, 31, 31), np.uint8)]),
        "scalars": np.pad(obs["scalars"][0], (0, 48)),
    }


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_inverse_matches_independent_grid_mask(jit, backend):
    for heading in range(4):
        for phase in range(3):
            for pending in (0, 1, 5, 300):
                for kind in ("coil", "head-one", "tail", "enclosed", "short", "wall", "overlap"):
                    sim = _fixture(heading, phase, pending, kind, jit=jit)
                    expected = sim.get_advisory_action_mask()[0, 0]
                    assert np.array_equal(advisory_from_row(_row(sim, backend)), expected), (
                        heading,
                        phase,
                        pending,
                        kind,
                        jit,
                        backend,
                    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("jit", [False, True])
def test_long_filled_bodies_and_wrapped_ring(backend, jit):
    sim = GridBatchSim(
        BatchSimConfig(num_envs=1, num_snakes=2, max_capacity=1600, initial_food=0),
        seeds=[991],
        train_mode=False,
        jit=jit,
    )
    inject_serpentines([sim], [700, 300], box_height=8)
    for slot in range(2):
        body = sim.get_bodies(0, slot)
        head = sim.cap - 3
        sim.bodies[0, slot] = 0
        for k, cell in enumerate(body):
            sim.bodies[0, slot, (head - k) % sim.cap] = cell
        sim.head_ptr[0, slot] = head
    sim._rebuild_env_grids(0)
    for phase in range(3):
        sim.boost_frames[0, 0] = phase
        sim._refresh_action_masks(np.ones(1, bool))
        row = _row(sim, backend)
        assert row["local"][0].max() == 255
        assert np.array_equal(advisory_from_row(row), sim.get_advisory_action_mask()[0, 0])


def test_inverse_preserves_empty_mask_and_fallback_q_choice():
    sim = _fixture(0, 2, 0, "enclosed")
    mask = advisory_from_row(_row(sim, "numpy"))
    assert not mask.any()
    q = np.array([1, 7, 7, 99, 98, 97], np.float32)
    resolved = mask.copy()
    resolved[:3] = True
    assert np.where(resolved, q, -np.inf).argmax() == 1


@pytest.mark.parametrize("field,value", [("length", 0), ("pending", -1), ("phase", 0.2)])
def test_inverse_refuses_unroundtrippable_metadata(field, value):
    row = _row(_fixture(0, 0, 0, "coil"), "numpy")
    if field == "length":
        row["length"] = value
    else:
        row["scalars"][2 if field == "pending" else 4] = value
    with pytest.raises(ValueError):
        advisory_from_row(row)
