"""Process-local exclusive timing wrappers around the unchanged T3 operator."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Iterator

STAGES = (
    "sim",
    "observation",
    "hero_v8",
    "opponent_v8",
    "vector61_frp3",
    "b8_forward",
    "clone_digest",
)


class StageTimer:
    """Subtract nested calls so every observed second belongs to one stage."""

    def __init__(self) -> None:
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.stack: list[list[Any]] = []
        self.originals: list[tuple[Any, str, Any]] = []

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        entry: list[Any] = [stage, 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.stack.pop()
            self.seconds[stage] += max(0.0, elapsed - entry[1])
            if self.stack:
                self.stack[-1][1] += elapsed

    def patch(self, owner: Any, name: str, stage: str) -> None:
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        @functools.wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            with self.measure(stage):
                return original(*args, **kwargs)

        setattr(owner, name, measured)

    def __enter__(self) -> StageTimer:
        from research.t3_rollout_labels_20261015.bank_engine import CapturedWorld
        from src.simd_env import ego2s_policy, ego_raster, ego_raster_bv8
        from src.simd_env.grid_sim import GridBatchSim
        from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

        self.patch(GridBatchSim, "step_with_policy", "sim")
        self.patch(GridBatchSim, "step", "sim")
        self.patch(CapturedWorld, "clone", "clone_digest")
        self.patch(ego_raster, "build_ego_raster", "observation")
        self.patch(ego_raster_bv8, "v8_scalars", "hero_v8")
        self.patch(ego2s_policy, "forward_q", "b8_forward")
        self.patch(Vector61Runtime, "prepare", "observation")
        self.patch(Vector61Runtime, "observe_step", "observation")
        self.patch(Vector61SimdPolicy, "actions", "vector61_frp3")
        self.patch(Vector61SimdPolicy, "_apply_veto", "opponent_v8")
        return self

    def __exit__(self, *exc: Any) -> None:
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
