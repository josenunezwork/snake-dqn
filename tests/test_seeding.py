"""Regression coverage for deterministic C0 seed streams."""

import random

import numpy as np
import pytest
import torch

from src.core.seeding import (
    SeedContext,
    capture_rng_state,
    initialize_run_seed,
    restore_rng_state,
)


def test_same_requested_seed_reproduces_global_and_named_streams():
    first = initialize_run_seed(1234)
    trace_one = (random.random(), float(np.random.random()), float(torch.rand(())))
    child_one = first.numpy_rng("rollout").integers(0, 2**31, size=4).tolist()
    second = initialize_run_seed(1234)
    trace_two = (random.random(), float(np.random.random()), float(torch.rand(())))
    child_two = second.numpy_rng("rollout").integers(0, 2**31, size=4).tolist()
    assert trace_one == trace_two
    assert child_one == child_two
    assert first.stream_seed("rollout") != first.stream_seed("evaluation")


def test_rng_state_capture_restores_cpu_trace():
    initialize_run_seed(9)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(state)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected


def test_omitted_seed_records_effective_replayable_value(monkeypatch):
    monkeypatch.setattr("src.core.seeding.secrets.randbits", lambda _: 42)
    context = initialize_run_seed()
    assert context.requested_seed is None
    assert context.effective_seed == 42


@pytest.mark.parametrize("seed", [True, -1, 2**64])
def test_seed_context_rejects_non_unsigned_64_bit_seeds(seed):
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        SeedContext(effective_seed=seed, requested_seed=None)
