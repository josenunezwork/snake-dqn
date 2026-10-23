"""Hermetic contracts for the A3 P1 learner-ceiling packet."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.a3_p1_ceiling import benchmark


def test_parser_requires_independent_unit_and_defaults_to_three_repeats(tmp_path) -> None:
    args = benchmark.build_parser().parse_args(
        [
            "--checkpoint",
            "b8.pth",
            "--output",
            str(tmp_path),
            "--unit-id",
            "cached-a100-000",
            "--mode",
            "cached",
        ]
    )
    assert args.repeats == 3
    assert args.worlds == 128 and args.rollout_steps == 128


def test_production_shape_and_repeat_guards_are_checked_before_cuda(monkeypatch, tmp_path) -> None:
    args = benchmark.build_parser().parse_args(
        [
            "--checkpoint",
            "missing.pth",
            "--output",
            str(tmp_path),
            "--unit-id",
            "bad",
            "--mode",
            "cached",
            "--worlds",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="frozen at 128"):
        benchmark.run_unit(args)
    args = benchmark.build_parser().parse_args(
        [
            "--checkpoint",
            "missing.pth",
            "--output",
            str(tmp_path),
            "--unit-id",
            "bad2",
            "--mode",
            "cached",
            "--repeats",
            "2",
            "--tiny",
        ]
    )
    with pytest.raises(ValueError, match="at least three"):
        benchmark.run_unit(args)


def test_gpu_busy_is_best_effort(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(benchmark.subprocess, "run", unavailable)
    assert benchmark._gpu_busy() is None


def test_summary_uses_median_and_conservative_p95() -> None:
    assert benchmark._summary([3.0, 1.0, 2.0]) == {
        "median_seconds": 2.0,
        "p95_seconds": 3.0,
        "samples": 3.0,
    }


def test_real_shell_device_is_understood_by_qualified_transport(monkeypatch, tmp_path) -> None:
    from research.colab_cli import canary

    entrypoint = Path(benchmark.__file__).with_name("colab.sh").read_text()
    monkeypatch.setattr(canary, "_entrypoint", lambda *_: entrypoint)
    assert canary.job_device({"device": "cuda", "env": {"A3_DEVICE": "cuda"}}, tmp_path) == "cuda"
