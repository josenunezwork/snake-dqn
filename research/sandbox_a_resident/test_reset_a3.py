"""A-3 sandbox-local initial-food adapter parity tests."""

from __future__ import annotations

import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.sandbox_a_resident.device_sim import ResidentDeviceSim, food_mechanics_parity_probe
from src.simd_env.batch_sim import BatchSimConfig


def make_cfg(worlds: int = 4) -> BatchSimConfig:
    return BatchSimConfig(
        num_envs=worlds,
        num_snakes=6,
        game_width=120,
        game_height=120,
        initial_food=10,
        max_food=10,
        max_capacity=32,
    )


def assert_tensor_state_equal(left: DeviceSim, right: DeviceSim) -> None:
    for name, value in left.__dict__.items():
        if isinstance(value, torch.Tensor):
            other = getattr(right, name)
            assert value.dtype == other.dtype and value.shape == other.shape, name
            assert torch.equal(value, other), name


def copy_tensor_state(source: DeviceSim, destination: DeviceSim) -> None:
    for name, value in source.__dict__.items():
        if isinstance(value, torch.Tensor):
            getattr(destination, name).copy_(value)


def test_batched_empty_initial_food_matches_frozen_counter_stream_and_order() -> None:
    torch.set_num_threads(1)
    frozen = DeviceSim(make_cfg(), device="cpu", train_mode=False, allow_respawn=True)
    candidate = ResidentDeviceSim(make_cfg(), device="cpu", train_mode=False, allow_respawn=True)
    frozen.rng_key.copy_(torch.tensor([19, 31, 47, 59]))
    frozen.rng_counter.copy_(torch.tensor([32, 32, 7, 11]))
    # Rows 0/1 model freshly reset blank worlds.  Rows 2/3 remain fully fed
    # and must retain their physical and food state through the optimized path.
    frozen.food_cells[2, 0] = torch.tensor([7, 7])
    frozen.food_present[2, 0] = True
    frozen.food_count[2] = 10
    frozen.food_cells[2, 1:10] = torch.tensor([8, 8])
    frozen.food_present[2, 1:10] = True
    frozen.food_cells[3, :10] = torch.tensor([9, 9])
    frozen.food_present[3, :10] = True
    frozen.food_count[3] = 10
    copy_tensor_state(frozen, candidate)
    before_live = {
        name: value[2:].clone()
        for name, value in candidate.__dict__.items()
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == candidate.E
    }

    frozen.prepare()
    candidate.prepare()
    assert_tensor_state_equal(frozen, candidate)
    for name, value in before_live.items():
        if name in {"frame", "rng_counter"}:
            continue
        assert torch.equal(getattr(candidate, name)[2:], value), name
    assert frozen.food_count[:2].tolist() == candidate.food_count[:2].tolist()
    assert torch.equal(frozen.food_cells[:2], candidate.food_cells[:2])
    assert torch.equal(frozen.rng_counter, candidate.rng_counter)


def test_batched_append_preserves_rejected_duplicate_occupied_and_boundary_candidates() -> None:
    torch.set_num_threads(1)
    frozen = DeviceSim(make_cfg(1), device="cpu", train_mode=False)
    candidate = ResidentDeviceSim(make_cfg(1), device="cpu", train_mode=False)
    # Existing food creates an occupied-cell rejection.  The stream also has
    # two boundary candidates and a duplicate accepted candidate.
    for sim in (frozen, candidate):
        sim.food_cells[0, 0] = torch.tensor([1, 1])
        sim.food_present[0, 0] = True
        sim.food_count[0] = 1
    stream = torch.tensor([[[-1, 0], [1, 1], [2, 2], [2, 2], [12, 0], [3, 3]]])
    mask = torch.ones((1, stream.shape[1]), dtype=torch.bool)
    in_bounds = mask & (stream[..., 0] >= 0) & (stream[..., 0] < frozen.grid_w)
    in_bounds &= (stream[..., 1] >= 0) & (stream[..., 1] < frozen.grid_h)
    for column in range(stream.shape[1]):
        # Maintenance supplies only in-arena candidates to the old one-at-a-
        # time append.  `_append_many` owns that boundary rejection itself.
        frozen._append_food(stream[:, column], in_bounds[:, column], False)
    candidate._append_many(stream, mask, False)
    assert torch.equal(frozen.food_cells, candidate.food_cells)
    assert torch.equal(frozen.food_present, candidate.food_present)
    assert torch.equal(frozen.food_count, candidate.food_count)
    assert [tuple(cell) for cell in candidate.food_cells[0, :3].tolist()] == [
        (1, 1),
        (2, 2),
        (3, 3),
    ]


def test_initial_food_adapter_preserves_mechanics_across_following_steps() -> None:
    torch.set_num_threads(1)
    frozen = DeviceSim(make_cfg(3), device="cpu", train_mode=False, allow_respawn=True)
    candidate = ResidentDeviceSim(make_cfg(3), device="cpu", train_mode=False, allow_respawn=True)
    frozen.rng_key.copy_(torch.tensor([101, 211, 307]))
    frozen.rng_counter.fill_(32)
    copy_tensor_state(frozen, candidate)
    actions = torch.tensor([[0, 1, 2, 0, 1, 2]]).expand(3, -1).clone()
    for _ in range(4):
        frozen.step(actions)
        candidate.step(actions)
        assert_tensor_state_equal(frozen, candidate)


def test_zero_ambient_with_corpse_food_uses_frozen_path() -> None:
    """A corpse-only row is not an empty reset row and must not take the fast path."""
    torch.set_num_threads(1)
    frozen = DeviceSim(make_cfg(2), device="cpu", train_mode=False)
    candidate = ResidentDeviceSim(make_cfg(2), device="cpu", train_mode=False)
    frozen.food_cells[0, 0] = torch.tensor([4, 4])
    frozen.food_present[0, 0] = True
    frozen.corpse_food[0, 0] = True
    frozen.food_count[0] = 1
    copy_tensor_state(frozen, candidate)
    frozen.prepare()
    candidate.prepare()
    assert_tensor_state_equal(frozen, candidate)


def test_same_device_food_mechanics_probe_returns_pass() -> None:
    torch.set_num_threads(1)
    result = food_mechanics_parity_probe(make_cfg(2), ticks=3)
    assert result == {"status": "PASS", "ticks": 3, "worlds": 2, "device": "cpu"}


def test_compact_food_rows_do_not_reinterpret_slot_constant_when_worlds_equal_seats() -> None:
    """E == S must not turn a slot tensor into an environment-indexed tensor."""
    sim = ResidentDeviceSim(make_cfg(6), device="cpu", train_mode=False)
    sim.slot_constant = torch.arange(sim.S)
    subset = sim._row_subset(torch.tensor([1, 4]))
    assert subset.E == 2
    assert subset.slot_constant.shape == (6,)
    assert torch.equal(subset.slot_constant, sim.slot_constant)


def test_naturally_depleted_food_preserves_inactive_padding_bytes() -> None:
    torch.set_num_threads(1)
    cfg = make_cfg(1)
    frozen = DeviceSim(cfg, device="cpu", train_mode=False)
    candidate = ResidentDeviceSim(cfg, device="cpu", train_mode=False)
    frozen.food_cells.copy_(
        torch.arange(frozen.food_cells.numel()).reshape_as(frozen.food_cells) % 12
    )
    frozen.rng_key.fill_(19)
    frozen.rng_counter.fill_(32)
    copy_tensor_state(frozen, candidate)
    frozen.prepare()
    candidate.prepare()
    assert_tensor_state_equal(frozen, candidate)
