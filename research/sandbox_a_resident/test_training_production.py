"""Production orchestration branches with bounded lightweight dependency seams."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.sandbox_a_resident import train
from research.sandbox_a_resident.test_training_runner import Actor, Critic, Task


@pytest.fixture
def production(monkeypatch, tmp_path):
    """Keep test_only=False; replace expensive science/engine dependencies only."""
    control = train.TrainControl("sandbox-a-train01", "a" * 40, 0)
    # Explicit test seam: production branches, three worlds, two tiny updates.
    for name, value in {
        "worlds": 3,
        "outer_iterations": 2,
        "rollout_horizon": 2,
        "proxy_worlds_per_mix": 3,
        "proxy_horizon": 2,
        "compile_mode": "none",
    }.items():
        object.__setattr__(control, name, value)
    hashes = {
        "b8": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
        "frp": train.FRP3_S12_SHA256,
        "champ": train.CHAMPION_SHA256,
    }
    monkeypatch.setattr(train, "digest", lambda p: hashes[Path(p).name])

    class FakeActor(Actor):
        def __init__(self, checkpoint):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.rand(()))
            self.reference = Actor()
            self.temperature = torch.tensor(1.0)

        def export_ego2s(self, path, metadata):
            path.write_bytes(b"fixture-actor")

    class FakeCritic(Critic):
        def __init__(self, reference):
            super().__init__()

    class FakeTrainer:
        def __init__(self, *args):
            self.actor_rows = 0
            self.critic_optimizer = None

        def update(self, rollout):
            self.actor_rows += rollout.actions.numel()
            return {"actor_rows": self.actor_rows}

        def state_dict(self):
            return {"actor_rows": self.actor_rows}

        def load_state_dict(self, state):
            self.actor_rows = state["actor_rows"]

    class FakeTask(Task):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.sim = SimpleNamespace(cfg=SimpleConfig())
            self.rosters = [SimpleNamespace(identity="fixture")]

        def attach_opponents(self, opponents):
            pass

        def proxy_factory(self, opponents):
            return lambda *args: None

        def state_dict(self):
            return {"age": self.age.clone(), "tick": self.tick}

        def load_state_dict(self, state):
            self.age, self.tick = state["age"].clone(), state["tick"]

    from dataclasses import dataclass

    @dataclass
    class SimpleConfig:
        fixture: bool = True

    class Compile:
        def __init__(self, actor, task, **kwargs):
            self.task, self.report = task, {}

        def runtime_self_check(self, rows):
            assert rows == control.worlds
            return {"rows": rows}

        def observe(self):
            return self.task.observe()

        def actor_forward(self, local, glob, scalars):
            return Actor()(local, glob, scalars)

    calls = []

    def screen(**kwargs):
        calls.append("screen")
        torch.rand(7)  # Would corrupt a restored RNG stream on iteration-zero resume.
        return {"material_gaps": []}

    def prepare(*args, **kwargs):
        calls.append("prepare")
        return SimpleNamespace(
            state=SimpleNamespace(
                state_dict=lambda: {
                    "stage": "ready",
                    "temperature": 1.0,
                    "quality": {"mse": 1.0},
                    "task_state": {
                        "age": torch.tensor([4, 5]),
                        "nested": {"tensor": torch.ones(2)},
                    },
                }
            ),
            critic_optimizer=None,
        )

    for name, replacement in {
        "B8Actor": FakeActor,
        "ClockedPopArtCritic": FakeCritic,
        "PPOTrainer": FakeTrainer,
        "ResidentTrainingTask": FakeTask,
        "DeviceRosterRuntime": lambda *a, **k: None,
        "TrainingCompile": Compile,
        "_load_vector61": lambda *a: None,
        "cpu_distribution_factory": lambda *a, **k: None,
        "run_distribution_screen": screen,
        "prepare": prepare,
        "evaluate": lambda *a: {},
        "serializable": lambda x: x,
    }.items():
        monkeypatch.setattr(train, name, replacement)
    return control, calls


def execute(control, output, **kwargs):
    return train.run(
        control,
        checkpoint=Path("b8"),
        frp3=Path("frp"),
        champion=Path("champ"),
        output=output,
        device="cpu",
        **kwargs,
    )


def test_production_metrics_and_progress_serialize_without_task_tensors(production, tmp_path):
    control, calls = production
    summary = execute(control, tmp_path / "full")
    assert calls == ["screen", "prepare"]
    assert control.test_only is False
    json.dumps(summary)
    progress = json.loads((tmp_path / "full/progress.json").read_text())
    assert progress["preparation"]["stage"] == "ready"
    assert "task_state" not in progress["preparation"]
    assert "task_state" not in summary["metrics"][1]["preparation"]
    # Exact task state is retained once in the opaque snapshot.
    state = torch.load(tmp_path / "full/iteration-0000.pt", weights_only=False)
    assert torch.equal(state["preparation"]["task_state"]["age"], torch.tensor([4, 5]))


def test_iteration_zero_production_resume_skips_screen_and_preserves_rng(
    production, tmp_path, monkeypatch
):
    control, calls = production
    execute(control, tmp_path / "full")
    state_path = tmp_path / "full/iteration-0000.pt"
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *a, **k: {0: {"iteration": 0, "exports": [{"path": str(state_path)}]}},
    )
    saved = torch.load(state_path, weights_only=False)
    expected_rng = saved["rng"]["torch"]
    original_load = torch.load
    placements = []

    def cpu_load(*args, **kwargs):
        placements.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(train.torch, "load", cpu_load)
    original_collect = train.collect
    observed = []

    def checked_collect(*args):
        if not observed:
            observed.append(torch.get_rng_state().clone())
        return original_collect(*args)

    monkeypatch.setattr(train, "collect", checked_collect)
    result = execute(control, tmp_path / "resume", ledger=ledger)
    assert calls == ["screen", "prepare"]
    assert placements == ["cpu"]
    assert torch.equal(observed[0], expected_rng)
    assert sum("distribution_screen" in row for row in result["metrics"]) == 1
    final = torch.load(tmp_path / "full/iteration-0002.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resume/iteration-0002.pt", weights_only=False)
    assert final["metrics"] == resumed["metrics"]
    assert torch.equal(final["rng"]["collection"], resumed["rng"]["collection"])


def test_material_distribution_gap_stops_before_preparation(production, tmp_path, monkeypatch):
    control, calls = production
    monkeypatch.setattr(
        train,
        "run_distribution_screen",
        lambda **kw: {"material_gaps": [{"mix": "frozen", "metric": "alive_fraction"}]},
    )
    with pytest.raises(RuntimeError, match="material distribution gap"):
        execute(control, tmp_path / "failure")
    assert "prepare" not in calls
    assert (
        json.loads((tmp_path / "failure/progress.json").read_text())["stage"]
        == "distribution-screen-failed"
    )
    assert not (tmp_path / "failure/iteration-0000.pt").exists()
