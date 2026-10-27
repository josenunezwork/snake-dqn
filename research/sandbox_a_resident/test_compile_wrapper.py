"""Focused contracts for ``CompiledBlock`` setup accounting and validation mode."""

from __future__ import annotations

import torch

from research.sandbox_a_resident.runtime_tools import CompiledBlock


def test_validation_uses_caller_tf32_once_and_keeps_eager_reference_no_grad(monkeypatch) -> None:
    observations: list[tuple[str, bool, bool]] = []

    def eager(value: torch.Tensor) -> torch.Tensor:
        observations.append(
            (
                "eager_grad" if torch.is_grad_enabled() else "eager_no_grad",
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            )
        )
        return value + 1

    def compile_block(fn, **kwargs):
        def compiled(value: torch.Tensor) -> torch.Tensor:
            observations.append(
                (
                    "compiled",
                    torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32,
                )
            )
            return fn(value)

        return compiled

    monkeypatch.setattr(torch, "compile", compile_block)
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        report: dict[str, object] = {}
        block = CompiledBlock(eager, "default", report, "block")
        assert block(torch.tensor([1.0])).tolist() == [2.0]
        assert observations == [
            ("compiled", True, True),
            ("eager_grad", True, True),
            ("eager_no_grad", True, True),
        ]
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        entry = report["block"]
        assert isinstance(entry, dict)
        assert "caller numerical globals" in entry["validation_policy"]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def test_setup_and_validation_accounting_are_separate_and_per_specialization(monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda eager, **kwargs: eager)
    report: dict[str, object] = {}
    block = CompiledBlock(lambda value: value * 2, "default", report, "block")

    assert block(torch.ones(2)).shape == (2,)
    assert block(torch.ones(2)).shape == (2,)
    assert block(torch.ones(3)).shape == (3,)

    entry = report["block"]
    assert isinstance(entry, dict)
    assert entry["status"] == "compiled"
    assert entry["compile_specialization_calls"] == 2
    assert entry["validation_calls"] == 2
    assert entry["compile_factory_seconds"] >= 0.0
    assert entry["compile_specialization_seconds"] >= 0.0
    assert entry["validation_seconds"] >= 0.0


def test_specialization_includes_strides_and_numerical_globals(monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda eager, **kwargs: eager)
    report: dict[str, object] = {}
    block = CompiledBlock(lambda value: value * 2, "default", report, "block")
    source = torch.ones(2, 3)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        block(source)
        block(source.t())
        torch.backends.cuda.matmul.allow_tf32 = True
        block(source)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    entry = report["block"]
    assert isinstance(entry, dict)
    assert entry["compile_specialization_calls"] == 3
    assert entry["validation_calls"] == 3
