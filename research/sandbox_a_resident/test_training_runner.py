"""Collector semantics for natural terminal/horizon cuts and CPU smoke objects."""

from types import SimpleNamespace

import torch

from research.sandbox_a_resident.train import TrainControl, collect, proxy_kind, rollout_monitoring


class Actor(torch.nn.Module):
    def forward(self, local, glob, scalars):
        return torch.zeros((local.shape[0], 6))


class Critic(torch.nn.Module):
    def forward(self, local, glob, scalars, clock):
        return clock + 1


class Task:
    def __init__(self):
        self.config = SimpleNamespace(horizon=5000)
        self.age = torch.tensor([0, 4999, 1])
        self.roster_id = torch.tensor([0, 1, 2])
        self.tick = 0

    def observe(self):
        return (
            torch.zeros((3, 7, 1, 1), dtype=torch.uint8),
            torch.zeros((3, 4, 1, 1), dtype=torch.uint8),
            torch.zeros((3, 60)),
            torch.ones((3, 6), dtype=torch.bool),
        )

    @property
    def hero_length(self):
        # Deliberately unlike scalar zero: collector diagnostics must use this
        # pre-action logical quantity.
        return self.age + 10

    def step(self, actions):
        self.tick += 1
        terminal = torch.tensor([self.tick == 1, False, False])
        cut = torch.tensor([False, self.tick == 1, False])
        self.age = torch.where(terminal | cut, torch.zeros_like(self.age), self.age + 1)
        return SimpleNamespace(reward=torch.ones(3), terminal=terminal, cut=cut)


def test_collector_marks_death_and_h5000_without_bootstrap() -> None:
    task = Task()
    rollout = collect(task, Actor(), Critic(), 2, torch.Generator().manual_seed(3))
    assert rollout.terminal[0].tolist() == [True, False, False]
    assert rollout.horizon[0].tolist() == [False, True, False]
    assert rollout.bootstrap[0].tolist() == [False, False, True]
    assert rollout.mix_ids.unique().tolist() == [0, 1, 2]
    assert rollout.logical_lengths[:, 0].tolist() == [10, 10]


def test_control_freezes_100m_shape_and_rejects_wrong_task() -> None:
    control = TrainControl(study="sandbox-a-train01", source_commit="a" * 40, allocation_index=0)
    assert control.actor_rows == 100_663_296
    try:
        TrainControl(study="foreign", source_commit="a" * 40, allocation_index=0)
    except ValueError:
        pass
    else:
        raise AssertionError("foreign control accepted")


def test_proxy_schedule_has_only_diagnostic16_and_existing_regular_boundaries() -> None:
    control = TrainControl("sandbox-a-train01", "a" * 40, 0)
    scheduled = [boundary for boundary in range(1, 385) if proxy_kind(boundary, control)]
    assert scheduled == [16, 64, 128, 192, 256, 320, 384]
    assert proxy_kind(16, control) == "diagnostic-update-16"
    assert all(proxy_kind(boundary, control) == "regular" for boundary in scheduled[1:])


def test_collector_stores_normal_action_support_for_boxed_in_rows():
    class BoxedTask(Task):
        def observe(self):
            local, glob, scalars, legal = super().observe()
            legal[0] = False
            return local, glob, scalars, legal

    rollout = collect(BoxedTask(), Actor(), Critic(), 2, torch.Generator().manual_seed(4))
    assert rollout.legal_masks[:, 0].tolist() == [[True, True, True, False, False, False]] * 2
    assert (rollout.actions[:, 0] < 3).all()
    assert rollout.legal_masks.gather(-1, rollout.actions[..., None]).all()


def test_collection_progress_preserves_actions_and_exact_rollout(monkeypatch):
    from research.sandbox_a_resident import train

    expected = collect(Task(), Actor(), Critic(), 2, torch.Generator().manual_seed(3))
    clock = iter(range(0, 10000, 61))
    monkeypatch.setattr(train.time, "monotonic", lambda: next(clock))
    messages = []
    actual = collect(
        Task(),
        Actor(),
        Critic(),
        2,
        torch.Generator().manual_seed(3),
        on_progress=lambda stage, values: messages.append((stage, values)),
    )
    for name, value in vars(expected).items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(getattr(actual, name), value), name
    assert [value["ticks"] for _, value in messages] == [1, 2]


def test_transition_support_has_explicit_mix_denominators() -> None:
    rollout = collect(Task(), Actor(), Critic(), 2, torch.Generator().manual_seed(3))
    support = rollout_monitoring(rollout)
    assert support["rows"] == 6
    assert [support["by_mix"][mix]["rows"] for mix in ("scripted", "frozen", "mixed")] == [2, 2, 2]
    assert sum(row["terminal_count"] for row in support["by_mix"].values()) == 1
    assert sum(row["horizon_count"] for row in support["by_mix"].values()) == 1
