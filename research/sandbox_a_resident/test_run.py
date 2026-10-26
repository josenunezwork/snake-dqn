"""Measured-phase coverage and packet runaway envelopes, without provider work."""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
        assert len(published) == 2
