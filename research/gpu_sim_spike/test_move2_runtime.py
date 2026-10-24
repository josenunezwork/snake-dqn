"""Focused contracts for the fresh MOVE2 runtime, independent of old MOVE evidence."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from research.gpu_sim_spike import move2_parity


def test_parity_compares_persistent_non_tensor_lifecycle() -> None:
    left = SimpleNamespace(
        value=torch.tensor([1], dtype=torch.int64), _prepared=False, rng_metadata={"draw": 3}
    )
    right = SimpleNamespace(
        value=torch.tensor([1], dtype=torch.int64), _prepared=True, rng_metadata={"draw": 3}
    )
    proof = move2_parity.compare(left, right)
    assert not proof["match"]
    assert proof["stage"] == "persistent_non_tensor_state"


def test_parity_rejects_dtype_even_when_values_match() -> None:
    left = SimpleNamespace(value=torch.tensor([1], dtype=torch.int64), _prepared=False)
    right = SimpleNamespace(value=torch.tensor([1], dtype=torch.int32), _prepared=False)
    proof = move2_parity.compare(left, right)
    assert not proof["match"]
    assert proof["field"] == "value"


def test_field_record_retains_original_dtype_and_bytes() -> None:
    row = move2_parity.field_record(
        torch.tensor([1.0], dtype=torch.float64).numpy(),
        torch.tensor([1.0], dtype=torch.float32).numpy(),
    )
    assert not row["match"]
    assert row["cpu"]["dtype"] == "float64"
    assert row["device"]["dtype"] == "float32"


def test_non_tensor_false_and_zero_are_not_identical() -> None:
    left = SimpleNamespace(value=torch.tensor([1]), lifecycle=False)
    right = SimpleNamespace(value=torch.tensor([1]), lifecycle=0)
    assert not move2_parity.compare(left, right)["match"]


def test_non_tensor_signed_zero_is_not_identical() -> None:
    left = SimpleNamespace(value=torch.tensor([1]), lifecycle=-0.0)
    right = SimpleNamespace(value=torch.tensor([1]), lifecycle=0.0)
    assert not move2_parity.compare(left, right)["match"]
