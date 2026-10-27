"""Measured-phase coverage and packet runaway envelopes, without provider work."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.sandbox_a_resident import core
from research.sandbox_a_resident import proof as resume_proof
from research.sandbox_a_resident import run


@pytest.mark.parametrize("truncated", [False, True])
def test_production_phase_requires_and_reports_measured_truncation(
    monkeypatch, tmp_path: Path, truncated: bool
) -> None:
    torch.set_num_threads(1)
    published = []
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        reset_count=torch.tensor(2),
        collect=lambda: {
            "terminal": torch.tensor([[True, False]]),
            "truncated": torch.tensor([[False, truncated]]),
        },
    )
    monkeypatch.setattr(run, "ResidentAssay", lambda *args: runner)
    monkeypatch.setattr(run, "load_ego2s_checkpoint", lambda *args: None)

    def publish(path, writer):
        published.append(path)
        return {"path": str(path), "size": 1, "sha256": "a" * 64}

    monkeypatch.setattr(run, "publish", publish)
    control = {
        "worlds": [2],
        "horizon": 1,
        "minibatch": 2,
        "epochs": 1,
        "episode_limit": 1,
        "cycles": 2,
        "warmups": 1,
        "tiny": False,
    }
    arguments = (
        control,
        2,
        "collection",
        0,
        tmp_path / "b8.pth",
        tmp_path,
        "cpu",
        time.monotonic() + 60,
    )
    result = run.run_cell(*arguments)
    assert len(published) == 2
    if not truncated:
        assert result["status"] == "INVALID"
        assert result["invalid_reasons"] == [
            f"measured cycle {index} did not exercise time-limit bootstrap" for index in range(2)
        ]
    else:
        assert result["status"] == "PASS"
        assert result["measured_terminal_rows"] == 2
        assert result["measured_truncated_rows"] == 2
        assert result["hero_rows"] == 4
        assert result["useful_hero_rows"] == 4
        assert result["cached_rows"] == 0
        assert result["cycle_stability"]["rows_per_second_min"] > 0
        assert (
            result["cycle_stability"]["rows_per_second_max"]
            >= result["cycle_stability"]["rows_per_second_min"]
        )
        assert result["cycles"][0]["hero_rows"] == 2
        assert result["cycles"][0]["hero_rows_per_second"] > 0


@pytest.mark.parametrize("episode_limit", [64, 512])
def test_exact_production_schedule_truncates_every_measured_cycle(
    monkeypatch, tmp_path: Path, episode_limit: int
) -> None:
    """Isolate the production clock schedule; real B8 dynamics have a separate smoke."""
    torch.set_num_threads(1)
    torch.manual_seed(3403)
    worlds = 4

    class ConstantActor(nn.Module):
        def __init__(self, b8):
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(6))
            self.value = nn.Parameter(torch.tensor(3.0))

        def forward(self, *obs):
            return self.logits.expand(obs[0].shape[0], -1), self.critic_only(*obs)

        def critic_only(self, *obs):
            return self.value.expand(obs[0].shape[0])

    def observe(runner, sim=None, advisory_mask=None):
        sim = runner.sim if sim is None else sim
        sim.prepare()
        mask = sim.advisory_mask() if advisory_mask is None else advisory_mask
        return (
            torch.zeros((sim.E, 7, 31, 31), dtype=torch.uint8),
            torch.zeros((sim.E, 4, 37, 37), dtype=torch.uint8),
            torch.zeros((sim.E, 60)),
        ), mask[:, 0]

    collected = []
    original_collect = core.ResidentAssay.collect

    def collect(runner):
        assert runner.recipe.tiny is False
        assert runner.recipe.stagger_initial_age is True
        assert runner.sim.cap == 1024
        assert runner.sim.cfg.max_capacity == 10400
        if not collected:
            cohorts = max(1, episode_limit // min(128, episode_limit))
            assert runner.age.tolist() == [
                (i % cohorts) * (episode_limit // cohorts) for i in range(worlds)
            ]
        data = original_collect(runner)
        # Nonzero critic checks the actual deferred bootstrap/GAE path, not just flags.
        assert data["next_value"][data["truncated"]].ne(0).all()
        collected.append(int(data["truncated"].sum()))
        return data

    def survive(sim, actions):
        # Isolate deterministic time-limit reachability from policy mortality.
        # The original age-zero schedule fails this test even with no deaths.
        # Keep real collection, clock increments, resets, bank, bootstrap and GAE.
        sim._last_done.zero_()
        sim._last_reward.zero_()

    def no_respawn(sim):
        # The no-death fixture has no missing seats. Avoid 32 empty CPU spawn
        # trials per tick without replacing prepare, masks or reset handling.
        assert sim.alive.all()

    monkeypatch.setattr(core, "ActorCritic", ConstantActor)
    monkeypatch.setattr(core.ResidentAssay, "observe", observe)
    monkeypatch.setattr(core.ResidentAssay, "collect", collect)
    monkeypatch.setattr(core.DeviceSim, "step", survive)
    monkeypatch.setattr(core.DeviceSim, "_respawn", no_respawn)
    monkeypatch.setattr(run, "load_ego2s_checkpoint", lambda *args: None)
    control = {
        "worlds": [worlds],
        "horizon": 128,
        "minibatch": 2048,
        "epochs": 1,
        "cycles": 3,
        "warmups": 1,
        "tiny": False,
        "recipe_parameters": {"stagger_initial_age": True},
    }
    result = run.run_cell(
        control,
        worlds,
        "integrated",
        27_000_000,
        tmp_path / "b8.pth",
        tmp_path,
        "cpu",
        time.monotonic() + 60,
        episode_limit,
    )
    assert len(collected) == 4  # One excluded warmup and three measured horizons.
    assert collected[1:] == ([8, 8, 8] if episode_limit == 64 else [1, 1, 1])
    assert [row["truncated_rows"] for row in result["cycles"]] == collected[1:]
    assert result["status"] == "PASS" and not result["invalid_reasons"]
    assert result["measured_truncated_rows"] == sum(collected[1:])
    exported = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert exported["status"] == "PASS"


def test_numerical_mode_restores_callers_switches_on_exception() -> None:
    before = run.numerical_settings()
    with pytest.raises(RuntimeError, match="intentional"):
        with run.numerical_mode(deterministic=not before["deterministic_algorithms"], tf32=True):
            assert (
                run.numerical_settings()["deterministic_algorithms"]
                is not before["deterministic_algorithms"]
            )
            assert run.numerical_settings()["cuda_matmul_allow_tf32"]
            assert run.numerical_settings()["cudnn_allow_tf32"]
            raise RuntimeError("intentional")
    assert run.numerical_settings() == before


def test_one_empty_cycle_invalidates_cell_even_when_total_truncations_are_positive(
    monkeypatch, tmp_path: Path
) -> None:
    outputs = iter([True, False, True])  # Excluded warmup, missing first cycle, valid last.
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        reset_count=torch.tensor(2),
        collect=lambda: {
            "terminal": torch.tensor([[False]]),
            "truncated": torch.tensor([[next(outputs)]]),
        },
    )
    monkeypatch.setattr(run, "ResidentAssay", lambda *args: runner)
    monkeypatch.setattr(run, "load_ego2s_checkpoint", lambda *args: None)
    monkeypatch.setattr(run, "package_unit", lambda path, *args: path.write_bytes(b"state"))
    result = run.run_cell(
        {
            "horizon": 1,
            "minibatch": 1,
            "epochs": 1,
            "episode_limit": 512,
            "cycles": 2,
            "warmups": 1,
            "tiny": False,
        },
        1,
        "collection",
        0,
        tmp_path / "b8",
        tmp_path,
        "cpu",
        time.monotonic() + 60,
    )
    assert result["status"] == "INVALID"
    assert result["measured_truncated_rows"] == 1
    assert result["invalid_reasons"] == ["measured cycle 0 did not exercise time-limit bootstrap"]
    assert json.loads((tmp_path / "e1-collection.json").read_text())["status"] == "INVALID"
    assert (tmp_path / "e1-collection.tar.complete").is_file()


def test_invalid_cell_does_not_abort_remaining_sweep(monkeypatch, tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"worlds": [2], "tiny": True, "timeout_seconds": 60}))
    checkpoint = tmp_path / "b8"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "output"
    called = []

    def cell(*args):
        phase = args[2]
        called.append(phase)
        return {"phase": phase, "status": "INVALID" if phase == "collection" else "PASS"}

    monkeypatch.setattr(run, "run_cell", cell)
    monkeypatch.setattr(run, "prove", lambda *args: {"numerical_settings": {}})
    monkeypatch.setattr(run, "load_ego2s_checkpoint", lambda *args: None)
    monkeypatch.setattr(run, "load_ledger", lambda *args: {})
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--control",
            str(control),
            "--checkpoint",
            str(checkpoint),
            "--ledger",
            str(tmp_path / "ledger"),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    assert run.main() == 0  # Complete exports, including INVALID evidence, before cleanup.
    assert called == ["cached", "collection", "integrated"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "INVALID"
    assert len(summary["results"]) == 3


def test_compile_mode_is_explicit_and_opt_in(monkeypatch, tmp_path: Path) -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(run, "ResidentAssay", lambda *args: runner)
    monkeypatch.setattr(run, "load_ego2s_checkpoint", lambda *args: None)
    control = {
        "worlds": [2],
        "horizon": 1,
        "minibatch": 2,
        "epochs": 1,
        "episode_limit": 1,
        "cycles": 1,
        "warmups": 0,
        "tiny": True,
        "compile_mode": "invalid",
    }
    with pytest.raises(ValueError, match="compile_mode"):
        run.run_cell(
            control,
            2,
            "collection",
            0,
            tmp_path / "b8.pth",
            tmp_path,
            "cpu",
            time.monotonic() + 60,
        )


def test_phase_profile_and_compile_report_are_exported_per_cycle_and_aggregate(
    monkeypatch, tmp_path: Path
) -> None:
    reports = iter(
        [
            {
                "mode": "cuda_event",
                "ticks": 2,
                "milliseconds": {"observe": 10.0},
                "ms_per_tick": {"observe": 5.0},
                "nested_phases": [],
            },
            {
                "mode": "cuda_event",
                "ticks": 2,
                "milliseconds": {"observe": 14.0},
                "ms_per_tick": {"observe": 7.0},
                "nested_phases": [],
            },
        ]
    )

    class ProfiledRunner:
        device = torch.device("cpu")
        reset_count = torch.tensor(0)
        compile_report = {"requested_mode": "none", "status": "eager", "subgraphs": {}}

        @property
        def phase_profile(self):
            return next(reports)

        def collect(self):
            return {
                "terminal": torch.tensor([[False]]),
                "truncated": torch.tensor([[True]]),
            }

    runner = ProfiledRunner()
    monkeypatch.setattr(run, "ResidentAssay", lambda *args: runner)
    monkeypatch.setattr(run, "load_ego2s_checkpoint", lambda *args: None)
    monkeypatch.setattr(run, "package_unit", lambda path, *args: path.write_bytes(b"state"))
    result = run.run_cell(
        {
            "horizon": 2,
            "minibatch": 2,
            "epochs": 1,
            "episode_limit": 2,
            "cycles": 2,
            "warmups": 0,
            "tiny": False,
        },
        1,
        "collection",
        0,
        tmp_path / "b8",
        tmp_path,
        "cpu",
        time.monotonic() + 60,
    )
    assert result["cycles"][0]["phase_profile"]["ms_per_tick"]["observe"] == 5.0
    assert result["phase_profile"] == {
        "mode": "cuda_event",
        "cycles": 2,
        "ticks_per_cycle": 2,
        "milliseconds": {"observe": 12.0},
        "ms_per_tick": {"observe": 6.0},
        "hero_rows_per_second": {"observe": pytest.approx(2 * 1000 / 12)},
    }
    assert result["compile_report"]["status"] == "eager"


def test_resume_proof_hashes_rows_before_borrowed_rollout_can_be_overwritten(monkeypatch) -> None:
    class CorruptSecondCollection:
        def __init__(self, *args, **kwargs) -> None:
            self.collect_count = 0

        def collect(self) -> dict[str, torch.Tensor]:
            self.collect_count += 1
            return {"row": torch.tensor([self.collect_count])}

        def update(self, rows: dict[str, torch.Tensor]) -> None:
            pass

        def snapshot(self) -> dict[str, int]:
            return {"collection": self.collect_count}

        def restore(self, state: dict[str, int]) -> None:
            # Model a fault that only affects the resumed collection.
            self.collect_count = state["collection"] + 1

    monkeypatch.setattr(resume_proof, "ResidentAssay", CorruptSecondCollection)
    with pytest.raises(RuntimeError, match="resume behavior/masks/GAE rows differ"):
        resume_proof.prove(None, "cpu")


def test_compile_cache_reuses_graphs_but_resets_model_and_optimizer(monkeypatch) -> None:
    from research.sandbox_a_resident.runtime_tools import CompiledBlock

    factories = []

    def compile_block(eager, **kwargs):
        factories.append(eager)
        return eager

    monkeypatch.setattr(torch, "compile", compile_block)

    def fresh(worlds, weight):
        actor = nn.Linear(1, 1)
        with torch.no_grad():
            actor.weight.fill_(weight)
        runner = SimpleNamespace(
            actor=actor,
            optimizer=torch.optim.Adam(actor.parameters()),
            device=torch.device("cpu"),
            recipe=SimpleNamespace(worlds=worlds, learning_rate=0.0003),
            sim=SimpleNamespace(cap=32, grid_h=6, grid_w=6),
            compile_report={},
        )

        def enable(mode):
            for attribute, name in zip(
                run.CellCompileCache.attributes,
                (
                    "ActorCritic_collect",
                    "ActorCritic_train",
                    "v8_decide",
                    "build_observation",
                    "anchors",
                ),
                strict=True,
            ):
                eager = actor if attribute.startswith("_forward") else lambda x: x + 1
                setattr(runner, attribute, CompiledBlock(eager, mode, runner.compile_report, name))
            runner.compile_mode = mode

        runner.enable_compile = enable
        return runner

    cache = run.CellCompileCache()
    first = fresh(2, 1)
    assert cache.attach(first, "default") is False
    first._forward_train(torch.ones(2, 1)).sum().backward()
    first.optimizer.step()
    assert first.optimizer.state
    before = run.compile_snapshot(first, "default", "cached")
    assert before["v8_decide"]["status"] == "not_applicable"
    assert first.compile_report["v8_decide"]["status"] == "pending"
    second = fresh(2, 7)
    assert cache.attach(second, "default") is True
    assert second.actor is first.actor
    assert second._forward_train is first._forward_train
    assert not second.optimizer.state
    assert torch.equal(second.actor.weight, torch.full_like(second.actor.weight, 7))
    assert len(factories) == 5
    second.compile_report["ActorCritic_train"]["status"] = "changed after snapshot"
    assert before["ActorCritic_train"]["status"] == "compiled"
    assert cache.attach(fresh(3, 2), "default") is True
    assert len(factories) == 5
    other_geometry = fresh(3, 2)
    other_geometry.sim.grid_h = 8
    assert cache.attach(other_geometry, "default") is False
    assert len(factories) == 10


def test_compile_snapshot_eager_schema_and_inactive_learner() -> None:
    assert run.compile_snapshot(SimpleNamespace(compile_report={}), "none") == {
        "requested_mode": "none",
        "status": "eager",
        "subgraphs": {},
    }
    runner = SimpleNamespace(compile_report={"ActorCritic_train": {"status": "pending"}})
    assert (
        run.compile_snapshot(runner, "default", "collection")["ActorCritic_train"]["status"]
        == "not_applicable"
    )
    assert runner.compile_report["ActorCritic_train"]["status"] == "pending"


def test_compiler_metrics_are_separate_nested_timer_deltas() -> None:
    assert run.compiler_delta(
        {"compile_inner": (2.0, 1), "backward": (1.0, 1)},
        {"compile_inner": (2.0, 1), "backward": (4.0, 2), "backend": (2.5, 1)},
    ) == {
        "backward": {"seconds": 3.0, "calls": 1},
        "backend": {"seconds": 2.5, "calls": 1},
    }
