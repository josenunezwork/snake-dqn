"""Two isolated resident teacher lanes overlapping host work and CUDA forwards."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numba
import torch

from research.t33_batched_teacher.inference import FixedInference
from research.t33_batched_teacher.teacher import BatchedTeacher, SearchConfig


class DoubleBufferedTeacher:
    """Run A/B in one process with distinct sims, slabs, streams and event fences.

    CUDA waits release the GIL, allowing the other lane to step/featurize. Each
    lane is limited to one Numba worker; BLAS/Torch must be configured to one by
    the caller. Two host workers stay within the four-thread local budget. This
    is a throughput API, not deterministic strict-gate inference.
    """

    def __init__(self, net: Any, config: SearchConfig, device: str = "cuda") -> None:
        if torch.get_num_threads() != 1:
            raise ValueError("A/B pipeline requires torch intra-op threads=1")
        self.lanes = []
        for _ in range(2):
            adapter = FixedInference(net, device)
            self.lanes.append(
                BatchedTeacher(config, device=device, inference=adapter, opponent_inference=adapter)
            )

    def choose(self, a: list[Any], b: list[Any]) -> tuple[Any, Any]:
        """Complete both fixed resident batches in input order, propagating errors."""

        def run(lane: BatchedTeacher, sources: list[Any]) -> Any:
            numba.set_num_threads(1)
            return lane.choose(sources)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="t33-ab") as pool:
            first = pool.submit(run, self.lanes[0], a)
            second = pool.submit(run, self.lanes[1], b)
            return first.result(), second.result()
