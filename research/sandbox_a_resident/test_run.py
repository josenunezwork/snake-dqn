"""Measured-phase coverage and packet runaway envelopes, without provider work."""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
    if not truncated:
        with pytest.raises(RuntimeError, match="did not exercise time-limit"):
            run.run_cell(*arguments)
        assert not published
    else:
        result = run.run_cell(*arguments)
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
        assert len(published) == 2


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
