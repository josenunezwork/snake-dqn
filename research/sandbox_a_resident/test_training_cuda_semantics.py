"""CUDA API contracts checked on CPU through strict relocated-state doubles.

These tests exercise device semantics without claiming execution on CUDA hardware.
Run explicitly with training_preflight --cuda-semantics (also in normal preflight).
"""

import random

import numpy as np
import torch

from research.sandbox_a_resident.train import _restore_rng


class RelocatedState:
    """Model a ByteTensor relocated by torch.load(map_location='cuda')."""

    device = torch.device("cuda:0")
    dtype = torch.uint8

    def __init__(self, value):
        self.value = value
        self.conversions = []

    def to(self, *, device, dtype):
        self.conversions.append((device, dtype))
        return self.value.to(device=device, dtype=dtype)


def test_relocated_rng_states_are_cpu_bytes_at_every_restore_api(monkeypatch):
    original = torch.get_rng_state()
    states = [RelocatedState(original.clone()) for _ in range(4)]
    seen = []

    def require_cpu_byte(value):
        assert isinstance(value, torch.Tensor)
        assert value.device.type == "cpu"
        assert value.dtype == torch.uint8
        seen.append(value)

    class CudaGenerator:
        device = torch.device("cuda:0")

        def set_state(self, value):
            require_cpu_byte(value)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "set_rng_state", require_cpu_byte)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda values: [require_cpu_byte(v) for v in values]
    )
    _restore_rng(
        {
            "torch": states[0],
            "cuda": states[1:3],
            "collection": states[3],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        CudaGenerator(),
    )
    assert len(seen) == 4
    assert all(state.conversions == [("cpu", torch.uint8)] for state in states)
