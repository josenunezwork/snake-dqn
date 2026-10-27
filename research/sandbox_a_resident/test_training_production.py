"""Production orchestration branches with bounded lightweight dependency seams."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.sandbox_a_resident import train
from research.sandbox_a_resident.test_training_runner import Actor, Critic, Task
from research.sandbox_a_resident.training_units import TRAIN01_07_SOURCE_COMMIT, source_migration


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

        def update(self, rollout, *, diagnostics=True):
            self.actor_rows += rollout.actions.numel()
            return {"actor_rows": self.actor_rows, "diagnostics": diagnostics}

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

        def proxy_factory(self, opponents, allocations):
            assert {row["phase"] for row in allocations} == {
                "train",
                "calibration",
                "critic",
                "monitor",
                "nomination",
                "engineering",
            }
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


def test_production_tasks_use_exact_audited_phase_bounds(production, tmp_path, monkeypatch):
    control, _calls = production
    original_task, original_prepare = train.ResidentTrainingTask, train.prepare
    configs = []

    def make_task(config, **kwargs):
        configs.append(config)
        return original_task(config, **kwargs)

    def preparation(factory, *args, **kwargs):
        factory("calibration")
        factory("refit")
        return original_prepare(factory, *args, **kwargs)

    monkeypatch.setattr(train, "ResidentTrainingTask", make_task)
    monkeypatch.setattr(train, "prepare", preparation)
    execute(control, tmp_path / "bounds")
    allocations = train.seed_allocations(vars(control))
    assert len(configs) == 3
    assert [config.worlds for config in configs] == [control.worlds, 24, 360]
    for config, allocation in zip(configs, allocations[:3], strict=True):
        assert config.allocation_phase_offset == allocation["offset"]
        assert config.allocation_count == allocation["count"]
        assert config.allocation_offset >= allocation["offset"]
        assert (
            config.allocation_offset + config.worlds * config.max_episodes
            <= allocation["offset"] + allocation["count"]
        )


def test_monitoring_toggle_preserves_training_state_and_rng(production, tmp_path, monkeypatch):
    control, _calls = production
    object.__setattr__(control, "outer_iterations", 17)
    calls = []
    monkeypatch.setattr(train, "evaluate", lambda *args: calls.append(args) or {})
    monkeypatch.setattr(train, "paired", lambda *args: {})
    execute(control, tmp_path / "on", monitoring=True)
    execute(control, tmp_path / "off", monitoring=False)
    monitored = torch.load(tmp_path / "on/iteration-0017.pt", weights_only=False)
    quiet = torch.load(tmp_path / "off/iteration-0017.pt", weights_only=False)

    def same(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and np.array_equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        return left == right

    for key in ("actor", "critic", "trainer", "task", "rng", "action_window"):
        assert same(monitored[key], quiet[key]), key
    # Baseline plus the update-16 proxy run; equality is checked after update 17.
    assert len(calls) == 2
    proxy = next(row["proxy"] for row in monitored["metrics"] if row.get("iteration") == 16)
    assert proxy["kind"] == "diagnostic-update-16" and proxy["diagnostic_only"]


def test_resume_at_16_does_not_repeat_successor_diagnostic(production, tmp_path, monkeypatch):
    control, _calls = production
    object.__setattr__(control, "outer_iterations", 17)
    monkeypatch.setattr(train, "paired", lambda *args: {})
    execute(control, tmp_path / "first")
    saved = tmp_path / "first/iteration-0016.pt"
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *args, **kwargs: {16: {"iteration": 16, "exports": [{"path": str(saved)}]}},
    )
    calls = []
    monkeypatch.setattr(train, "evaluate", lambda *args: calls.append(args) or {})
    result = execute(control, tmp_path / "resumed", ledger=ledger)
    assert calls == []
    assert (
        sum(row.get("proxy", {}).get("kind") == "diagnostic-update-16" for row in result["metrics"])
        == 1
    )


def test_bound_predecessor_proxy_cache_rebinds_without_evaluation(
    production, tmp_path, monkeypatch
):
    control, _calls = production
    object.__setattr__(control, "outer_iterations", 17)
    predecessor = train.TrainControl("sandbox-a-train01", TRAIN01_07_SOURCE_COMMIT, 0)
    for field in (
        "worlds",
        "outer_iterations",
        "rollout_horizon",
        "proxy_worlds_per_mix",
        "proxy_horizon",
        "compile_mode",
    ):
        object.__setattr__(predecessor, field, getattr(control, field))
    monkeypatch.setattr(train, "paired", lambda *args: {})
    execute(predecessor, tmp_path / "old")
    saved = tmp_path / "old/iteration-0016.pt"
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"source_migration": source_migration(asdict(predecessor), asdict(control))})
    )
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *args, **kwargs: {16: {"iteration": 16, "exports": [{"path": str(saved)}]}},
    )
    calls = []
    monkeypatch.setattr(train, "evaluate", lambda *args: calls.append(args) or {})
    result = execute(control, tmp_path / "new", ledger=ledger)
    assert calls == []
    migration = next(
        row["proxy_cache_migration"] for row in result["metrics"] if "proxy_cache_migration" in row
    )
    assert migration["from_source_commit"] == TRAIN01_07_SOURCE_COMMIT
    assert migration["to_source_commit"] == control.source_commit

    state = torch.load(saved, weights_only=False)
    # A forged successor identity must not bypass the bound predecessor check.
    for row in state["metrics"]:
        if "proxy_baseline" in row:
            row["proxy_identity"] = migration["to_proxy_identity"]
    forged = tmp_path / "forged-target.pt"
    torch.save(state, forged)
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *args, **kwargs: {16: {"iteration": 16, "exports": [{"path": str(forged)}]}},
    )
    with pytest.raises(ValueError, match="bound predecessor"):
        execute(control, tmp_path / "forged", ledger=ledger)

    state = torch.load(saved, weights_only=False)
    for row in state["metrics"]:
        if "proxy_baseline" in row:
            row["proxy_identity"] = "0" * 64
    corrupt = tmp_path / "corrupt.pt"
    torch.save(state, corrupt)
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *args, **kwargs: {16: {"iteration": 16, "exports": [{"path": str(corrupt)}]}},
    )
    with pytest.raises(ValueError, match="bound predecessor"):
        execute(control, tmp_path / "corrupt", ledger=ledger)


def test_predecessor_iteration16_catches_up_diagnostic_once(production, tmp_path, monkeypatch):
    control, _calls = production
    object.__setattr__(control, "outer_iterations", 17)
    predecessor = train.TrainControl("sandbox-a-train01", TRAIN01_07_SOURCE_COMMIT, 0)
    for field in (
        "worlds",
        "outer_iterations",
        "rollout_horizon",
        "proxy_worlds_per_mix",
        "proxy_horizon",
        "compile_mode",
    ):
        object.__setattr__(predecessor, field, getattr(control, field))
    monkeypatch.setattr(train, "paired", lambda *args: {})
    execute(predecessor, tmp_path / "old")
    original = torch.load(tmp_path / "old/iteration-0016.pt", weights_only=False)
    for row in original["metrics"]:
        if row.get("iteration") == 16:
            row.pop("proxy", None)
    legacy = tmp_path / "legacy-07-iteration16.pt"
    torch.save(original, legacy)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"source_migration": source_migration(asdict(predecessor), asdict(control))})
    )
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *args, **kwargs: {16: {"iteration": 16, "exports": [{"path": str(legacy)}]}},
    )
    calls = []
    monkeypatch.setattr(train, "evaluate", lambda *args: calls.append(args) or {})
    result = execute(control, tmp_path / "resumed", ledger=ledger)
    # The cached B8 baseline is retained; only the candidate diagnostic runs
    # against the post-update-16 predecessor weights before update 17.
    assert len(calls) == 1
    assert (
        sum(row.get("proxy", {}).get("kind") == "diagnostic-update-16" for row in result["metrics"])
        == 1
    )
