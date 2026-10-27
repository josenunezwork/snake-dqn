"""Focused contracts for ``CompiledBlock`` setup accounting and validation mode."""

from __future__ import annotations

import torch

from research.sandbox_a_resident.runtime_tools import CompiledBlock


def test_validation_disables_tf32_for_both_paths_and_restores_it(monkeypatch) -> None:
    observations: list[tuple[str, bool, bool]] = []

    def eager(value: torch.Tensor) -> torch.Tensor:
        observations.append(
            (
                "eager",
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
        assert observations[0] == ("compiled", False, False)
        # The mock compiled callable delegates to eager once; the wrapper then
        # invokes eager a second time as the validation reference.
        assert observations[1:] == [("eager", False, False), ("eager", False, False)]
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        entry = report["block"]
        assert isinstance(entry, dict)
        assert entry["validation_policy"].startswith("TF32 disabled")
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
