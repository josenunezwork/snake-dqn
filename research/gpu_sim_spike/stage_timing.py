"""Exclusive, opt-in synchronized stage attribution and actual CUDA kernel counts."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch

STAGES = (
    "movement",
    "deaths_corpse",
    "food",
    "vector61",
    "v8_floods",
    "opponent_forwards",
    "b8_forward",
    "observation_paint",
    "collection",
    "reset",
)


class StageTiming:
    def __init__(self, device: torch.device, *, enabled: bool = False) -> None:
        self.device, self.enabled = device, enabled
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.launches = dict.fromkeys(STAGES, 0)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        cuda = self.device.type == "cuda"
        profiler = None
        if cuda:
            torch.cuda.synchronize(self.device)
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            profiler.__enter__()
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started
            if profiler is not None:
                profiler.__exit__(None, None, None)
                self.launches[name] = self.launches.get(name, 0) + sum(
                    event.device_type == torch.autograd.DeviceType.CUDA
                    and event.name not in ("[memory]",)
                    and not event.name.startswith(("Memcpy", "Memset"))
                    for event in profiler.events()
                )

    def consume(self) -> dict[str, Any]:
        value = {
            "stage_timers": self.seconds.copy(),
            "launch_counts": self.launches.copy(),
            "host_stages": self.seconds.copy(),
            "instrumented": self.enabled,
        }
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.launches = dict.fromkeys(STAGES, 0)
        return value
