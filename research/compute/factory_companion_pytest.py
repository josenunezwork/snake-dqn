"""Pytest plugin that verifies the factory test lane is single-threaded and CPU-only."""

from __future__ import annotations

import os
import sys

import pytest


def _fail(message: str) -> None:
    raise pytest.UsageError(f"factory companion test refusal: {message}")


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get("SNAKE_DQN_DEVICE") != "cpu":
        _fail("SNAKE_DQN_DEVICE must be cpu")
    required = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_THREAD_LIMIT",
    )
    if any(os.environ.get(name) != "1" for name in required):
        _fail("all numerical thread environment limits must equal one")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        _fail("torch thread limits were not applied")
    if "numba" in sys.modules:
        import numba

        if numba.config.NUMBA_NUM_THREADS != 1:
            _fail("numba thread limit was not applied")
    if "threadpoolctl" in sys.modules:
        from threadpoolctl import threadpool_info

        if any(int(pool.get("num_threads", 1)) > 1 for pool in threadpool_info()):
            _fail("threadpoolctl reports parallel numerical work")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        fixtures = set(getattr(item, "fixturenames", ()))
        if item.get_closest_marker("requires_gpu") or fixtures & {
            "gpu",
            "cuda",
            "mps",
            "gpu_device",
        }:
            _fail(f"GPU/MPS fixture or marker is forbidden: {item.nodeid}")
