"""Focused, CPU-only checks for the T4A harness primitives."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from research.gpu_sim_spike import t4a_compare as compare
from research.gpu_sim_spike import t4a_fixtures as fixtures


class _RasterNet(nn.Module):
    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalar: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((local.flatten(1)[:, :2], glob.flatten(1)[:, :2], scalar[:, :2]), dim=1)


def test_fixed_forward_pads_each_raster_argument_and_restores_real_rows() -> None:
    wrapped = fixtures.Fixed1024Forward(_RasterNet(), "hero")
    local = torch.ones((4, 1, 2, 2), dtype=torch.uint8)
    glob = torch.ones((4, 1, 2, 2), dtype=torch.uint8)
    scalar = torch.ones((4, 2))
    got = wrapped(local, glob, scalar)
    assert tuple(got.shape) == (4, 6)
    assert wrapped.calls[0].padded_shapes == ((1024, 1, 2, 2), (1024, 1, 2, 2), (1024, 2))
    assert wrapped.calls[0].output_shape == (1024, 6)


@pytest.mark.parametrize("worlds", (4, 64, 1024))
def test_frozen_layouts_have_four_unique_positions(worlds: int) -> None:
    for layout in fixtures.LAYOUTS:
        rows = fixtures.layout_rows(worlds, layout)
        assert len(set(rows)) == 4
        assert all(0 <= row < worlds for row in rows)
    assert fixtures.layout_rows(worlds, "interspersed") == tuple(
        i * (worlds - 1) // 3 for i in range(4)
    )


def test_comparator_detects_one_bit_and_scalar_counter_faults() -> None:
    base = torch.zeros(4, dtype=torch.long)
    changed = base.clone()
    changed[0] = 1
    mismatch = compare.compare(base, changed, "sim.rng_counter")
    assert mismatch is not None
    assert mismatch.field == "sim.rng_counter"
    scalar = torch.tensor(3.0)
    assert compare.digest(scalar)


def test_frozen_roots_are_distinct_by_mode_and_roster() -> None:
    values = {fixtures.seed_for(roster, smoke=False, canary=False) for roster in fixtures.ROSTERS}
    assert values == {9402401, 9402402, 9402403}
    assert fixtures.seed_for("frozen", smoke=True, canary=False) == 9402421
    assert fixtures.HISTORY_OFFSETS == (0, 3, 11, 23)


@pytest.mark.parametrize("shape", [(), (7, 7)])
def test_unknown_persistent_tensor_refuses_even_without_world_axis(shape) -> None:
    from types import SimpleNamespace

    owner = SimpleNamespace(hidden=torch.zeros(shape))
    with pytest.raises(ValueError, match="unknown persistent tensor"):
        compare.persistent_tensors(owner, "sim")


def test_sacrificial_bodies_are_connected_distinct_and_do_not_touch_targets() -> None:
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
    from src.simd_env.batch_sim import BatchSimConfig

    cfg = BatchSimConfig(
        num_envs=64,
        num_snakes=6,
        game_width=1450,
        game_height=830,
        max_capacity=1024,
        initial_food=0,
        max_food=2,
    )
    sim = DeviceSim(cfg, cap=1024)
    runtime = DeviceVectorRuntime()
    runtime.bind(sim)
    before = sim.bodies[:4].clone()
    fixtures._fill_sacrificial(sim, runtime, {0, 1, 2, 3}, 9402401)
    assert torch.equal(before, sim.bodies[:4])
    assert int(sim.seg_count[4:].min()) == 8
    assert int(sim.seg_count[4:].max()) == 511
    for row in (4, 7, 8, 17, 26):
        cells_by_seat = []
        for seat in range(6):
            count = int(sim.seg_count[row, seat])
            rings = (sim.head_ptr[row, seat] - torch.arange(count)) % sim.cap
            cells = sim.bodies[row, seat, rings]
            assert torch.unique(cells, dim=0).shape[0] == count
            assert bool((cells.diff(dim=0).abs().sum(1) == 1).all())
            assert bool(((cells[:, 0] >= 0) & (cells[:, 0] < sim.grid_w)).all())
            assert bool(((cells[:, 1] >= 0) & (cells[:, 1] < sim.grid_h)).all())
            cells_by_seat.append(cells)
        all_cells = torch.cat(cells_by_seat)
        assert torch.unique(all_cells, dim=0).shape[0] == all_cells.shape[0]
