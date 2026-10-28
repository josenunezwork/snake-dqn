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
from research.sandbox_a_resident.training_units import (
    TRAIN01_07_SOURCE_COMMIT,
    TRAIN01_11_SOURCE_COMMIT,
    source_migration,
)


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


def test_train01_11_iteration0_migration_skips_preparation_and_rebinds_proxy_cache(
    production, tmp_path, monkeypatch
):
    control, prep_calls = production
    predecessor = train.TrainControl("sandbox-a-train01", TRAIN01_11_SOURCE_COMMIT, 0)
    for field in (
        "worlds",
        "outer_iterations",
        "rollout_horizon",
        "proxy_worlds_per_mix",
        "proxy_horizon",
        "compile_mode",
    ):
        object.__setattr__(predecessor, field, getattr(control, field))
    execute(predecessor, tmp_path / "attempt11")
    assert prep_calls == ["screen", "prepare"]
    saved = tmp_path / "attempt11/iteration-0000.pt"
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"source_migration": source_migration(asdict(predecessor), asdict(control))})
    )
    monkeypatch.setattr(
        train,
        "load_ledger",
        lambda *args, **kwargs: {0: {"iteration": 0, "exports": [{"path": str(saved)}]}},
    )
    eval_calls = []
    monkeypatch.setattr(train, "evaluate", lambda *args: eval_calls.append(args) or {})
    result = execute(control, tmp_path / "attempt12", ledger=ledger)
    # Screen, preparation, and B8 baseline proxy are not repeated on attempt12.
    assert prep_calls == ["screen", "prepare"]
    assert eval_calls == []
    migration = next(
        row["proxy_cache_migration"] for row in result["metrics"] if "proxy_cache_migration" in row
    )
    assert migration["from_source_commit"] == TRAIN01_11_SOURCE_COMMIT
    assert migration["to_source_commit"] == control.source_commit


def _compact_resume_inputs(control, tmp_path, monkeypatch, *, manifest_edit=None):
    """Encode the iteration-0 snapshot, allowlist it, and stage runtime inputs without the .pt."""
    import shutil

    from research.sandbox_a_resident import training_units
    from research.sandbox_a_resident.training_compact import encode_snapshot
    from research.sandbox_a_resident.training_resume import attach_compact_resume, build_ledger
    from research.sandbox_a_resident.units import digest
    from research.sandbox_a_resident.units import publish as publish_unit

    exports = tmp_path / "exports"
    exports.mkdir()
    for stem in ("iteration-0000.pt", "iteration-0000.json", "iteration-0000-actor.pth"):
        for suffix in ("", ".sha256", ".complete"):
            shutil.copy2(tmp_path / "full" / f"{stem}{suffix}", exports / f"{stem}{suffix}")
    ledger, uploads, latest = build_ledger(exports, asdict(control))
    assert latest == 0
    unit = ledger["units"][0]["id"]
    original = exports / "iteration-0000.pt"
    encode_snapshot(original, unit=unit)
    manifest_path = exports / "iteration-0000.pt.xz.manifest"
    if manifest_edit is not None:
        payload = json.loads(manifest_path.read_text())
        manifest_edit(payload)
        for suffix in ("", ".sha256", ".complete"):
            Path(f"{manifest_path}{suffix}").unlink()
        publish_unit(manifest_path, lambda path: path.write_text(json.dumps(payload)))

    def bound(path, runtime):
        return {"path": runtime, "size": path.stat().st_size, "sha256": digest(path)}

    monkeypatch.setattr(
        training_units,
        "COMPACT_RESUME_ALLOWLIST",
        {
            unit: {
                "original": ledger["units"][0]["exports"][0],
                "derivative": bound(
                    exports / "iteration-0000.pt.xz", "resume/iteration-0000.pt.xz"
                ),
                "manifest": bound(manifest_path, "resume/iteration-0000.pt.xz.manifest"),
            }
        },
    )
    ledger, uploads = attach_compact_resume(ledger, uploads, exports)
    inputs = tmp_path / "inputs"
    for row in uploads:
        (inputs / row["dest"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], inputs / row["dest"])
    assert not (inputs / "resume/iteration-0000.pt").exists()
    ledger_path = inputs / "verified-ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    return original, ledger_path, inputs


def test_compact_resume_decodes_verifies_and_resumes_from_decoded_state(
    production, tmp_path, monkeypatch, capsys
):
    """The real ledger path decodes the derivative, verifies tensors, and restores that state."""
    control, calls = production
    execute(control, tmp_path / "full")
    original, ledger_path, inputs = _compact_resume_inputs(control, tmp_path, monkeypatch)
    loads, verified = [], []
    original_load, original_verify = torch.load, train.verify_decoded_state

    def recording_load(path, *args, **kwargs):
        state = original_load(path, *args, **kwargs)
        loads.append((str(path), state))
        return state

    def recording_verify(state, manifest):
        verified.append((state, manifest["unit"]))
        return original_verify(state, manifest)

    restored = []
    original_restore = train._restore

    def recording_restore(state, *args, **kwargs):
        restored.append(state)
        return original_restore(state, *args, **kwargs)

    monkeypatch.setattr(train.torch, "load", recording_load)
    monkeypatch.setattr(train, "verify_decoded_state", recording_verify)
    monkeypatch.setattr(train, "_restore", recording_restore)
    result = execute(control, tmp_path / "resume", ledger=ledger_path, input_root=inputs)
    decoded = inputs / "resume/iteration-0000.pt"
    assert decoded.read_bytes() == original.read_bytes()
    assert [path for path, _ in loads] == [str(decoded.resolve())]
    # The exact object decoded from the derivative is verified and then restored.
    assert verified[0][0] is loads[0][1] and restored == [loads[0][1]]
    assert verified[0][1] == "a100-dev/sandbox-a/train01/iteration-0000"
    assert calls == ["screen", "prepare"]
    final = original_load(tmp_path / "full/iteration-0002.pt", weights_only=False)
    resumed = original_load(tmp_path / "resume/iteration-0002.pt", weights_only=False)
    assert final["metrics"] == resumed["metrics"]
    assert sum("distribution_screen" in row for row in result["metrics"]) == 1
    assert torch.equal(final["rng"]["collection"], resumed["rng"]["collection"])
    assert torch.equal(final["rng"]["torch"], resumed["rng"]["torch"])
    assert torch.equal(final["task"]["age"], resumed["task"]["age"])
    stages = [
        json.loads(line.removeprefix("TRAIN01_PROGRESS "))["stage"]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("TRAIN01_PROGRESS ")
    ]
    compact_stages = [stage for stage in stages if stage.startswith("compact-resume")]
    assert compact_stages == [
        "compact-resume-decode",
        "compact-resume-decoded",
        "compact-resume-verified",
    ]


def test_compact_resume_refuses_tampered_derivative_before_restore(
    production, tmp_path, monkeypatch
):
    from research.sandbox_a_resident.training_compact import CompactResumeRefused

    control, _calls = production
    execute(control, tmp_path / "full")
    _original, ledger_path, inputs = _compact_resume_inputs(control, tmp_path, monkeypatch)
    derivative = inputs / "resume/iteration-0000.pt.xz"
    raw = bytearray(derivative.read_bytes())
    raw[-20] ^= 0x01
    derivative.write_bytes(bytes(raw))
    monkeypatch.setattr(train, "_restore", lambda *a, **k: pytest.fail("restored tampered state"))
    with pytest.raises(CompactResumeRefused, match="differs from its binding"):
        execute(control, tmp_path / "tampered", ledger=ledger_path, input_root=inputs)
    assert not (inputs / "resume/iteration-0000.pt").exists()


def test_compact_resume_refuses_decoded_state_that_differs_from_manifest(
    production, tmp_path, monkeypatch
):
    """A (re-allowlisted) manifest that disagrees with the original refuses before restore."""
    from research.sandbox_a_resident.training_compact import CompactResumeRefused

    def forge(payload):
        payload["tensors"][0]["sha256"] = "0" * 64

    control, _calls = production
    execute(control, tmp_path / "full")
    _original, ledger_path, inputs = _compact_resume_inputs(
        control, tmp_path, monkeypatch, manifest_edit=forge
    )
    monkeypatch.setattr(train, "_restore", lambda *a, **k: pytest.fail("restored unverified state"))
    with pytest.raises(CompactResumeRefused, match="decoded tensor differs"):
        execute(control, tmp_path / "forged", ledger=ledger_path, input_root=inputs)


def test_independent_replication_seed_domain_shifts_training_and_keeps_proxy_bank_identical(
    production, tmp_path, monkeypatch
):
    base_control, _calls = production
    rep_control = train.TrainControl(
        "sandbox-a-train01",
        base_control.source_commit,
        allocation_index=1,
        train_seed_offset=1_000_000_000,
    )
    for field in (
        "worlds",
        "outer_iterations",
        "rollout_horizon",
        "proxy_worlds_per_mix",
        "proxy_horizon",
        "compile_mode",
    ):
        object.__setattr__(rep_control, field, getattr(base_control, field))
    assert train._seed(rep_control, rep_control.allocation_index) != train._seed(
        base_control, base_control.allocation_index
    )
    assert train.allocation(asdict(rep_control)) != train.allocation(asdict(base_control))

    original_task = train.ResidentTrainingTask
    original_prepare = train.prepare
    task_configs = []
    cpu_offsets = []
    device_screen_roots = []
    proxy_configs = []

    def make_task(config, **kwargs):
        task_configs.append(config)
        created = original_task(config, **kwargs)
        created.proxy_factory = lambda _opponents, _allocations: (
            lambda _mix, _worlds, _horizon, seed: SimpleNamespace(seed=seed)
        )
        return created

    def preparation(factory, *args, **kwargs):
        factory("calibration")
        factory("refit")
        return original_prepare(factory, *args, **kwargs)

    def capture_cpu_factory(*_args, root_offset=600_000_000, **_kwargs):
        cpu_offsets.append(root_offset)
        return None

    def capture_screen(**kwargs):
        for mix in ("scripted", "frozen", "mixed"):
            created = kwargs["device_factory"](mix, 32, 5000)
            device_screen_roots.append(created.seed)
        return {"material_gaps": []}

    def capture_evaluate(_actor, _make_task, config, *_args, **_kwargs):
        proxy_configs.append(config)
        return {}

    monkeypatch.setattr(train, "ResidentTrainingTask", make_task)
    monkeypatch.setattr(train, "prepare", preparation)
    monkeypatch.setattr(train, "cpu_distribution_factory", capture_cpu_factory)
    monkeypatch.setattr(train, "run_distribution_screen", capture_screen)
    monkeypatch.setattr(train, "evaluate", capture_evaluate)
    monkeypatch.setattr(train, "paired", lambda *args: {})

    execute(rep_control, tmp_path / "rep")
    assert cpu_offsets == [1_600_000_000]
    assert device_screen_roots == [1_600_000_000, 1_600_000_032, 1_600_000_064]
    assert [c.allocation_phase_offset for c in task_configs[:3]] == [
        1_000_000_000,
        1_200_000_000,
        1_300_000_000,
    ]
    assert proxy_configs and all(cfg.seed_base == 400_000_000 for cfg in proxy_configs)

    # Foreign ledger/snapshot from base_control must be refused by rep_control.
    execute(base_control, tmp_path / "base")
    base_state = torch.load(tmp_path / "base/iteration-0000.pt", weights_only=False)
    with pytest.raises(ValueError, match="foreign snapshot allocation"):
        train._restore(
            base_state,
            rep_control,
            train.B8Actor(Path("b8")),
            train.ClockedPopArtCritic(None),
            train.PPOTrainer(None, None, None),
            train.ResidentTrainingTask(task_configs[0]),
            torch.Generator(),
        )
