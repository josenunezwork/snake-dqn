"""Task-boundary checks for natural H5000 sandbox PPO collection."""

from __future__ import annotations

import pytest
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
    assert result.reset.all()
    assert torch.all(result.cut | result.terminal)
    assert task.age.eq(0).all()


def test_task_rejects_seed_range_that_can_escape_audited_phase() -> None:
    with torch.no_grad():
        try:
            TrainingTaskConfig(
                worlds=3,
                tiny=True,
                allocation_phase_offset=600_000_000,
                allocation_offset=600_000_064,
                allocation_count=96,
                max_episodes=11,
            )
        except ValueError as error:
            assert "audited phase range" in str(error)
        else:
            raise AssertionError("out-of-range reset allocation was accepted")


@pytest.mark.parametrize(
    ("phase_offset", "phase_count", "worlds"),
    (
        (200_000_000, 5_242_880 + 1024, 24),
        (300_000_000, 8_388_608 + 1024, 24),
        (400_000_000, 3 * 96, 96),
        (500_000_000, 3 * 256, 256),
        (600_000_000, 3 * 32, 32),
    ),
)
def test_every_nontraining_phase_rejects_one_episode_past_its_audited_end(
    phase_offset: int, phase_count: int, worlds: int
) -> None:
    with pytest.raises(ValueError, match="audited phase range"):
        TrainingTaskConfig(
            worlds=worlds,
            allocation_phase_offset=phase_offset,
            allocation_offset=phase_offset,
            allocation_count=phase_count,
            max_episodes=phase_count // worlds + 1,
        )


@torch.inference_mode()
def test_step_reset_never_eagerly_observes_or_erases_terminal_cause(monkeypatch) -> None:
    task = _task()
    task.sim.bodies[0, 0, 0] = torch.tensor([0, 0])
    task.sim.head_ptr[0, 0] = 0
    task.sim.seg_count[0, 0] = 1
    task.sim.length[0, 0] = 1
    task.sim.direction[0, 0] = 3
    monkeypatch.setattr(
        task, "observe", lambda: (_ for _ in ()).throw(AssertionError("eager observe"))
    )
    event = task.step(torch.ones(3, dtype=torch.long))
    assert event.terminal[0]
    assert task.sim._last_death_cause[0, 0].item() != 0


@torch.inference_mode()
def test_reset_refuses_episode_after_audited_cap_before_hashing(monkeypatch) -> None:
    task = ResidentTrainingTask(
        TrainingTaskConfig(worlds=3, tiny=True, allocation_count=3, max_episodes=1), device="cpu"
    )
    task.episode[0] = 1
    import research.sandbox_a_resident.training_task as task_module

    monkeypatch.setattr(
        task_module, "_key", lambda *_: (_ for _ in ()).throw(AssertionError("hashed"))
    )
    with pytest.raises(RuntimeError, match="audited seed allocation"):
        task.reset(torch.tensor([True, False, False]))


@torch.inference_mode()
def test_initial_reset_hashes_only_initial_rows_not_all_episode_slots(monkeypatch) -> None:
    import research.sandbox_a_resident.training_task as task_module

    calls: list[int] = []
    original = task_module._key
    monkeypatch.setattr(
        task_module,
        "_key",
        lambda namespace, index: calls.append(index) or original(namespace, index),
    )
    ResidentTrainingTask(
        TrainingTaskConfig(worlds=3, tiny=True, allocation_count=6, max_episodes=2), device="cpu"
    )
    assert calls == [0, 1, 2]


@torch.inference_mode()
def test_reset_sampler_uses_counter_derived_headings() -> None:
    task = ResidentTrainingTask(TrainingTaskConfig(worlds=24, tiny=True), device="cpu")
    assert torch.any(task.sim.direction != 1)


@torch.inference_mode()
def test_single_life_terminal_reports_no_performed_reset() -> None:
    task = ResidentTrainingTask(
        TrainingTaskConfig(worlds=3, tiny=True, auto_reset=False, max_episodes=1), device="cpu"
    )
    task.observe()
    task.sim.bodies[:, 0, 0] = 0
    task.sim.head_ptr[:, 0] = 0
    task.sim.direction[:, 0] = 3
    result = task.step(torch.ones(3, dtype=torch.long))
    assert result.terminal.all()
    assert not result.reset.any()
    assert not task.sim.alive[:, 0].any()
    assert (task.sim._last_death_cause[:, 0] != 0).all()


@torch.inference_mode()
def test_reset_heading_and_spawn_are_independent_of_reset_grouping() -> None:
    batched, separate = _task(), _task()
    batched.reset()
    for index in range(3):
        separate.reset(torch.arange(3).eq(index))
    for name in ("bodies", "direction", "rng_key", "rng_counter"):
        assert torch.equal(getattr(batched.sim, name), getattr(separate.sim, name)), name


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
