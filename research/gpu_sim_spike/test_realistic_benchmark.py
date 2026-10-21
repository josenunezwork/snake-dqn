"""Focused contract tests for E1's production-shape throughput receipt."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from research.gpu_sim_spike import realistic_benchmark as realistic


def test_realistic_shape_requires_six_actual_long_bodies() -> None:
    worlds, lengths = realistic.validate_shape(
        (256, 1024, 4096), (800, 900, 1000, 1100, 1300, 1500)
    )
    assert worlds == (256, 1024, 4096)
    assert min(lengths) == 800
    with pytest.raises(ValueError, match="six bodies"):
        realistic.validate_shape((256,), (800,) * 5)
    with pytest.raises(ValueError, match="all >=800"):
        realistic.validate_shape((256,), (799, 900, 1000, 1100, 1300, 1500))


def test_p95_uses_nearest_rank_and_does_not_hide_three_tick_tail() -> None:
    assert realistic._p95([0.1, 0.2, 0.9]) == 0.9
    assert realistic._p95([0.4]) == 0.4


def test_resident_reset_copies_same_device_tensor_state_only() -> None:
    destination = SimpleNamespace(
        data=torch.tensor([[1, 2], [3, 4]]),
        scalar=torch.tensor(7),
        note="keep",
    )
    template = SimpleNamespace(
        data=torch.tensor([[8, 9], [10, 11]]),
        scalar=torch.tensor(2),
        note="replace?",
    )
    realistic._resident_reset(destination, template)
    assert destination.data.tolist() == [[8, 9], [10, 11]]
    assert destination.scalar.item() == 2  # Runtime counters reset with the resident template.
    assert destination.note == "keep"


def test_realistic_source_hashes_bind_native_kernels_and_launcher() -> None:
    from research.gpu_sim_spike.benchmark import source_hashes

    sources = source_hashes()
    assert "benchmark.py" in sources
    assert "realistic_benchmark.py" in sources
    assert "colab.sh" in sources
    # The kernel files are created by the sibling kernel slice.  Once present,
    # they are covered by exactly the same source closure as the benchmark.
    for name in ("flood_cuda.cpp", "flood_cuda.cu"):
        if (realistic.Path(__file__).parent / name).exists():
            assert name in sources


def test_floor_requires_actual_production_concurrency_and_all_mixes() -> None:
    cases = [
        {
            "case": mix,
            "requested_worlds": worlds,
            "actual_concurrent_worlds": worlds,
            "median_full_world_steps_per_second": 20000.0,
        }
        for mix in realistic.MIXES
        for worlds in (256, 1024, 4096)
    ]
    assert realistic.floor_status(cases, l4=True) == "FLOOR_MET_PENDING_COST_AND_PARITY_AUDIT"
    assert realistic.floor_status(cases, l4=False) == "PRECHECK_ONLY_NOT_L4_DATA"
    assert realistic.floor_status(cases[:-1], l4=True) == "INCOMPLETE_LONG_BODY_COVERAGE"
    cases[-1]["actual_concurrent_worlds"] = 6  # serial-root branch width cannot qualify.
    assert realistic.floor_status(cases, l4=True) == "STOP_ENGINE_THESIS_THIS_WEEK"
    cases[-1]["actual_concurrent_worlds"] = 4096
    cases[-1]["median_full_world_steps_per_second"] = 19999.99
    assert realistic.floor_status(cases, l4=True) == "STOP_ENGINE_THESIS_THIS_WEEK"
