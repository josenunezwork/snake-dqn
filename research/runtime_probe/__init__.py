"""Runtime probe package for comparing Colab CPU/T4/L4/A100 and Mac CPU throughput."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_INTERVAL",
    "DEFAULT_SEED",
    "PROBE_SCHEMA",
    "RUNTIME_SPECS",
    "TARGETS",
    "build_manifest",
    "check_stack_support",
    "collect_system_info",
    "normalize_gpu_target",
    "prepare_probe_package",
    "run_benchmark",
    "run_watch6_slice",
    "validate_probe_output",
]

_BENCH_EXPORTS = {
    "DEFAULT_HORIZON",
    "DEFAULT_INTERVAL",
    "DEFAULT_SEED",
    "PROBE_SCHEMA",
    "collect_system_info",
    "run_benchmark",
    "run_watch6_slice",
    "validate_probe_output",
}

_PREPARE_EXPORTS = {
    "RUNTIME_SPECS",
    "TARGETS",
    "build_manifest",
    "check_stack_support",
    "normalize_gpu_target",
    "prepare_probe_package",
}


def __getattr__(name: str) -> Any:
    if name in _BENCH_EXPORTS:
        from research.runtime_probe import bench

        return getattr(bench, name)
    if name in _PREPARE_EXPORTS:
        from research.runtime_probe import prepare

        return getattr(prepare, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
