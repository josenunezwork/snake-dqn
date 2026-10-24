"""Source contracts for CUDA vector61's explicitly ordered scalar arithmetic.

The native extension is intentionally CUDA-only, so a Mac CPU preflight cannot
execute this code.  These checks make the exactness-critical lowering visible
to that preflight; ``test_cuda_indexed_geometry_bytes_match_canonical_cpu`` in
``test_vector_geometry.py`` remains the executable A100 admission test.
"""

from __future__ import annotations

from pathlib import Path

_CUDA_SOURCE = Path(__file__).with_name("vector_geometry.cu")


def _source() -> str:
    return _CUDA_SOURCE.read_text(encoding="utf-8")


def test_native_feature_scalars_use_explicit_round_to_nearest_double_operations() -> None:
    """Guard the no-FMA, one-operation-at-a-time contract on CPU preflight."""
    source = _source()
    for intrinsic in ("__dadd_rn", "__dsub_rn", "__dmul_rn", "__ddiv_rn", "__dsqrt_rn"):
        assert intrinsic in source
    # The feature buffer is float32, so make the CPU oracle's final IEEE-754
    # binary64 -> binary32 rounding explicit too; an ordinary C++ cast leaves
    # that final instruction choice to nvcc.
    assert "__double2float_rn" in source
    assert "static_cast<float>(" not in source
    # Integer deltas/squares must be retained until the single rounded sqrt.
    assert "double distance_from_squared(long long squared)" in source
    assert "return dsqrt(static_cast<double>(squared));" in source
    # No implicit device sqrt expression remains in the feature kernel; all
    # square roots flow through the explicit ``dsqrt`` wrapper.
    assert " = sqrt(" not in source


def test_native_source_keeps_cpu_feature_operation_order_at_reported_boundaries() -> None:
    """Keep density and danger-bonus chains separate and correctly ordered.

    These are the two operations that produced one-ULP A100 discrepancies in
    the sparse-cap160 canary.  The assertions intentionally match the source
    structure, so replacing the explicit round-to-nearest path with ordinary
    CUDA operators fails even where CUDA hardware is unavailable.
    """
    source = _source()
    assert "const double expected_per_sector = fmax(" in source
    assert "ddiv(static_cast<double>(food_capacity), static_cast<double>(sectors))" in source
    assert "ddiv(static_cast<double>(food_counts[sec]), expected_per_sector)" in source
    assert (
        "const double bonus = dmul(fmin(1.0, ddiv(static_cast<double>(count[sec]), 5.0)), .2);"
        in source
    )
    assert "danger[sec] = fmin(2.0, dadd(danger[sec], bonus));" in source
    assert "const double val = dmul(fmax(0.0, dsub(1.0, ddiv(dist, maxd))), 1.5);" in source
