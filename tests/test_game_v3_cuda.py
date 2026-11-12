"""CUDA vs CPU bitwise parity tests (32 worlds x 1,000 ticks) and cuda_check.py smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from research.game_v3.cuda_check import main as cuda_check_main
from research.game_v3.cuda_check import run_device_vs_cpu_parity

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this machine",
)


@pytest.mark.slow
def test_cuda_32_worlds_x_1000_ticks_vs_cpu_bitwise_parity() -> None:
    """32 worlds x 1,000 ticks at N=24 on CUDA matches CPU bit-for-bit."""
    result = run_device_vs_cpu_parity(
        device="cuda",
        num_envs=32,
        num_ticks=1000,
        num_snakes=24,
        learner_slots=8,
        arena_radius=180.0,
        check_every=100,
    )
    assert result["passed"] is True
    assert result["rebases"] >= 2
    assert result["world_resets"] >= 2
    assert result["total_deaths"] >= 100
    assert result["total_kills"] >= 20
    assert result["total_rollout_truncations"] > 0
    assert all(c > 0 for c in result["action_counts"])


def test_cuda_check_entrypoint_writes_single_json(tmp_path: Path) -> None:
    """cuda_check.py runs parity + benchmark on CUDA and writes one JSON result."""
    out_json = tmp_path / "cuda_check_smoke.json"
    rc = cuda_check_main(
        [
            "--device",
            "cuda",
            "--parity-envs",
            "8",
            "--parity-ticks",
            "50",
            "--bench-envs",
            "16",
            "--bench-snakes",
            "24",
            "--learners",
            "8",
            "--warmup-steps",
            "2",
            "--bench-steps",
            "5",
            "--stage-steps",
            "3",
            "--rebase-reset-envs",
            "16",
            "--json-out",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "game-v3-cuda-check/v1"
    assert payload["plan"]["within_l4_budget"] is True
    assert "early-game memory" in payload["memory_measurement_note"]
    assert payload["parity"]["passed"] is True
    assert payload["parity"]["deterministic_algorithms"] is False
    assert payload["parity"]["rebases"] >= 1
    assert payload["parity"]["world_resets"] >= 1
    assert payload["parity_deterministic"]["passed"] is True
    assert payload["parity_deterministic"]["deterministic_algorithms"] is True
    assert payload["parity_deterministic"]["rebases"] >= 1
    assert payload["parity_deterministic"]["world_resets"] >= 1
    assert len(payload["benchmarks"]) == 1
    assert payload["benchmarks"][0]["deterministic_algorithms"] is False
    assert payload["benchmarks"][0]["cases"][0]["status"] == "ok"
    assert payload["rebase_reset_benchmark"]["status"] == "ok"
    assert payload["rebase_reset_benchmark"]["rebase_subticks"]["shift_applied"] > 0
    assert (
        "max_memory_allocated_after_bytes" in payload["rebase_reset_benchmark"]["rebase_subticks"]
    )
    assert "max_memory_allocated_after_bytes" in payload["rebase_reset_benchmark"]["reset_worlds"]
