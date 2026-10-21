"""Run every E1 device test under CUDA dtype contracts on CPU/meta too."""

from collections.abc import Iterator

import pytest

from research.gpu_sim_spike.test_cuda_dtypes import CudaDtypeRules


@pytest.fixture(autouse=True)
def enforce_cuda_dtype_contracts() -> Iterator[None]:
    """Include simulator, observations, v8, anchors, policies and dense rollout."""
    with CudaDtypeRules():
        yield
