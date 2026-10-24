"""Small non-GPU checks for I0 benchmark constants and decision boundaries."""

from __future__ import annotations

from research.gpu_sim_spike import integration_bench as bench


def test_i0_fixed_repeats_warmups_orders_and_continuity_are_predata_constants() -> None:
    assert bench.WARMUPS == 3
    assert bench.REPEATS == 3
    assert bench.CONTINUOUS_TICKS == 32
    assert bench.ORDERS == (("R", "I"), ("I", "R"), ("R", "I"))
