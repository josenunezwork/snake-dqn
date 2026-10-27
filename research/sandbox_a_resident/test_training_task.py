"""Task-boundary checks for natural H5000 sandbox PPO collection."""

from __future__ import annotations

import torch

from research.sandbox_a_resident.training_task import ResidentTrainingTask, TrainingTaskConfig


def _task() -> ResidentTrainingTask:
    return ResidentTrainingTask(TrainingTaskConfig(worlds=3, tiny=True), device="cpu")


@torch.inference_mode()
def test_task_emits_b8_shapes_and_full_body_capacity() -> None:
    task = _task()
    local, glob, scalars, legal = task.observe()
    assert local.shape == (3, 7, 31, 31)
    assert glob.shape == (3, 4, 37, 37)
    assert scalars.shape == (3, 60)
    assert legal.shape == (3, 6)
    heads = task.sim.heads()[0]
    assert len({tuple(value.tolist()) for value in heads}) == task.sim.S
    assert bool((heads[:, 0] > 0).all() and (heads[:, 0] < task.sim.grid_w - 1).all())
    assert bool((heads[:, 1] > 0).all() and (heads[:, 1] < task.sim.grid_h - 1).all())
    assert task.sim.cap == task.sim.cfg.max_capacity == 10400
    # Cross the historical A-3 physical ring cap and then its wrap boundary.
    task.sim.seg_count[0, 0] = 1025
    task.sim.length[0, 0] = 10400
    task.sim.head_ptr[0, 0] = 10399
    task.sim._push(torch.tensor([[[1, 1]] * 6]), torch.tensor([[True] + [False] * 5] * 3))
    assert task.sim.seg_count[0, 0].item() == 1026
    assert task.sim.head_ptr[0, 0].item() == 0
    assert not task.sim.capacity_overflow[0, 0]


@torch.inference_mode()
def test_forced_terminal_and_horizon_cut_reset_without_trace_leak() -> None:
    task = _task()
    # Force slot zero's next normal move through the left wall.  The returned
    # record describes that terminal frame although the next world is already
    # resident for the following observation.
    task.sim.bodies[0, 0, 0] = torch.tensor([0, 0])
    task.sim.head_ptr[0, 0] = 0
    task.sim.seg_count[0, 0] = 1
    task.sim.length[0, 0] = 1
    task.sim.direction[0, 0] = 3
    result = task.step(torch.ones(3, dtype=torch.long))
    assert result.terminal[0]
    assert result.reset[0]
    assert not result.cut[0]
    assert result.reward[0].item() == 0.0
    assert task.age[0].item() == 0

    task.age.fill_(task.config.horizon - 1)
    result = task.step(torch.ones(3, dtype=torch.long))
    assert result.cut.all()
    assert result.reset.all()
    assert not result.terminal.any()
    assert task.age.eq(0).all()


def test_task_snapshot_round_trip_preserves_next_observation() -> None:
    task = _task()
    task.step(torch.ones(3, dtype=torch.long))
    saved = task.state_dict()
    expected = tuple(value.clone() for value in task.observe())
    task.step(torch.zeros(3, dtype=torch.long))
    task.load_state_dict(saved)
    actual = task.observe()
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))


@torch.inference_mode()
def test_prepared_seam_resume_next_step_is_exact() -> None:
    direct = _task()
    # observe() leaves the task at the exact policy seam used by collection.
    direct.observe()
    assert direct.sim._prepared
    saved = direct.state_dict()
    resumed = _task()
    resumed.load_state_dict(saved)
    assert resumed.sim._prepared
    action = torch.tensor((1, 1, 1), dtype=torch.long)
    left, right = direct.step(action), resumed.step(action)
    assert all(torch.equal(a, b) for a, b in zip(left, right))
    for name in ("frame", "rng_counter", "length", "alive", "food_count", "bodies"):
        assert torch.equal(getattr(direct.sim, name), getattr(resumed.sim, name))


@torch.inference_mode()
def test_sandbox_opponent_respawn_clears_hunger_and_boost_carry() -> None:
    task = _task()
    task.sim.alive[0, 1] = False
    task.sim.respawn_timer[0, 1] = 0
    task.sim.boost_frames[0, 1] = 99
    task.sim.frames_since_food[0, 1] = 999
    task.sim._respawn()
    assert task.sim.alive[0, 1]
    assert task.sim.boost_frames[0, 1].item() == 0
    assert task.sim.frames_since_food[0, 1].item() == 0
