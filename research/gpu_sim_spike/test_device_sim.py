"""Focused CPU-fallback tests for the device-only rollout state."""

import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def _world(*, food: int = 2) -> GridBatchSim:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=2,
        game_width=100,
        game_height=100,
        initial_food=food,
        max_food=3 if food else 0,
        max_capacity=32,
    )
    return GridBatchSim(cfg, seeds=[7], train_mode=True, allow_respawn=False, jit=False)


def test_clone_branch_step_and_reference_bridge() -> None:
    reference = _world()
    sim = DeviceSim.from_reference(reference)
    branches = sim.repeat_interleave(6)
    assert branches.bodies.shape[0] == 6
    sim.step(torch.ones((1, 2), dtype=torch.long))
    sim.copy_env_to_reference(reference)
    assert reference.get_action_mask().shape == (1, 2, 6)
    assert sim.frame.tolist() == [1]


def test_head_path_swap_is_a_head_collision() -> None:
    reference = _world(food=0)
    reference.bodies[0, :, 0] = ((4, 4), (5, 4))
    reference.head_ptr[:] = 0
    reference.seg_count[:] = 1
    reference.length[:] = 1
    reference.alive[:] = True
    reference.direction[0] = (1, 3)  # right, left
    reference._rebuild_env_grids(0)
    sim = DeviceSim.from_reference(reference)
    actions = torch.ones((1, 2), dtype=torch.long)
    reference.step(actions.numpy())
    sim.step(actions)
    assert reference.alive[0].tolist() == [False, False]
    assert sim.alive[0].tolist() == [False, False]


def test_corpse_cap_keeps_newest_combined_stream_entries() -> None:
    cfg = BatchSimConfig(num_envs=1, num_snakes=1, game_width=1000, game_height=1000, max_food=300)
    sim = DeviceSim(cfg)
    ambient = torch.tensor([[[i % 100, 9 + i // 100] for i in range(300)]], dtype=torch.long)
    corpse = torch.tensor([[[i % 100, i // 100] for i in range(900)]], dtype=torch.long)
    sim._append_many(ambient, torch.ones((1, 300), dtype=torch.bool), False)
    sim._append_many(corpse, torch.ones((1, 900), dtype=torch.bool), True)
    assert sim.food_count.tolist() == [450]
    assert sim.corpse_food.sum(dim=1).tolist() == [150]
    assert sim.food_cells[0, 300:450].tolist() == corpse[0, 750:900].tolist()


def test_corpse_fifo_handles_reinsert_after_eviction() -> None:
    cfg = BatchSimConfig(num_envs=1, num_snakes=1, game_width=300, game_height=100, max_food=20)
    sim = DeviceSim(cfg)
    old = torch.tensor([[[i, 0] for i in range(10)]], dtype=torch.long)
    sim._append_many(old, torch.ones((1, 10), dtype=torch.bool), True)
    sim._append_many(torch.tensor([[[10, 0], [0, 0]]]), torch.ones((1, 2), dtype=torch.bool), True)
    assert sim.food_cells[0, :10].tolist() == [[i, 0] for i in range(2, 10)] + [[10, 0], [0, 0]]


def test_corpse_fifo_empty_stream_reinsert_keeps_latest_instance() -> None:
    cfg = BatchSimConfig(num_envs=1, num_snakes=1, game_width=300, game_height=100, max_food=20)
    sim = DeviceSim(cfg)
    cells = torch.tensor([[[i, 0] for i in list(range(11)) + [0]]], dtype=torch.long)
    sim._append_many(cells, torch.ones((1, 12), dtype=torch.bool), True)
    assert sim.food_cells[0, :10].tolist() == [[i, 0] for i in range(2, 11)] + [[0, 0]]


def test_three_way_equal_head_collision_keeps_last_cpu_pair_survivor() -> None:
    # Slots 0/1 collide first and die; slot 2's later pair references a dead
    # opponent and therefore survives under the CPU collision-list order.
    cfg = BatchSimConfig(
        num_envs=1, num_snakes=3, game_width=100, game_height=100, initial_food=0, max_food=0
    )
    reference = GridBatchSim(cfg, seeds=[3], train_mode=True, allow_respawn=False, jit=False)
    reference.bodies[0, :, 0] = ((4, 5), (6, 5), (5, 4))
    reference.head_ptr[:] = 0
    reference.seg_count[:] = 1
    reference.length[:] = 1
    reference.alive[:] = True
    reference.direction[0] = (1, 3, 2)
    reference._rebuild_env_grids(0)
    sim = DeviceSim.from_reference(reference)
    actions = torch.ones((1, 3), dtype=torch.long)
    reference.step(actions.numpy())
    sim.step(actions)
    assert reference.alive[0].tolist() == [False, False, True]
    assert sim.alive[0].tolist() == [False, False, True]


def test_boost_burn_drops_the_v2_tail_pellet() -> None:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=1,
        game_width=200,
        game_height=100,
        initial_food=0,
        max_food=0,
        max_capacity=32,
        min_boost_length=5,
        boost_length_cost_frames=3,
    )
    reference = GridBatchSim(cfg, seeds=[1], train_mode=True, allow_respawn=False, jit=False)
    reference.bodies[0, 0, :5] = ((5, 5), (4, 5), (3, 5), (2, 5), (1, 5))
    reference.head_ptr[:] = 0
    reference.seg_count[:] = 5
    reference.length[:] = 5
    reference.alive[:] = True
    reference.direction[:] = 1
    reference.boost_frames[:] = 2
    reference._rebuild_env_grids(0)
    sim = DeviceSim.from_reference(reference)
    action = torch.tensor([[4]])
    reference.step(action.numpy())
    sim.step(action)
    assert sim.length.tolist() == reference.length.tolist()
    assert [
        tuple(cell) for cell in sim.food_cells[0, : sim.food_count[0]].tolist()
    ] == reference.food_cells[0]
    assert sim.corpse_food[0, : sim.food_count[0]].tolist() == [True]


def test_food_resolution_rechecks_second_boost_cell_after_contest() -> None:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=2,
        game_width=200,
        game_height=100,
        initial_food=0,
        max_food=2,
        max_capacity=32,
        min_boost_length=5,
    )
    reference = GridBatchSim(cfg, seeds=[19], train_mode=True, allow_respawn=False, jit=False)
    reference.bodies[0, 0, :5] = ((5, 5), (4, 5), (3, 5), (2, 5), (1, 5))
    reference.bodies[0, 1, :5] = ((8, 5), (9, 5), (10, 5), (11, 5), (12, 5))
    reference.head_ptr[:] = 0
    reference.seg_count[:] = 5
    reference.length[:] = 5
    reference.alive[:] = True
    reference.direction[0] = (1, 3)
    reference.food_cells[0] = [(6, 5), (7, 5)]
    reference.food_set[0] = set(reference.food_cells[0])
    reference.corpse_cells[0] = set()
    reference._rebuild_env_grids(0)
    sim = DeviceSim.from_reference(reference)
    action = torch.tensor([[4, 1]])  # slot 0 crosses both; slot 1 takes cell 7.
    reference.step(action.numpy())
    sim.step(action)
    assert sim._last_food_ate.tolist() == reference._last_food_ate.tolist()
    assert sim.length.tolist() == reference.length.tolist()
