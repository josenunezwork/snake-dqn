"""Disjoint factorial engineering cells and an explicit synchronization census."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class SyncCensus(TorchDispatchMode):
    """Count sync-capable ATen operations, not inferred hardware stall duration."""

    def __init__(self) -> None:
        super().__init__()
        self.scope = "setup_or_export"
        self.counts: dict[str, Counter] = defaultdict(Counter)

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        name = str(func)
        if any(
            token in name
            for token in ("_local_scalar_dense", "nonzero", "bincount", "masked_select")
        ):
            self.counts[self.scope][name] += 1
        if "_to_copy" in name and str(kwargs.get("device")) == "cpu":
            if args and isinstance(args[0], torch.Tensor) and args[0].device.type == "cuda":
                self.counts[self.scope]["cuda_to_host_copy"] += 1
        return func(*args, **kwargs)


class DiagnosticTiming:
    """CUDA events preserve queue ordering without a sync at each stage boundary."""

    def __init__(self, device: torch.device, census: SyncCensus) -> None:
        self.device, self.census = device, census
        self.events: list[tuple[str, Any, Any]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        previous, self.census.scope = self.census.scope, name
        cuda = self.device.type == "cuda"
        start = torch.cuda.Event(enable_timing=True) if cuda else time.perf_counter()
        if cuda:
            start.record()
        try:
            yield
        finally:
            end = torch.cuda.Event(enable_timing=True) if cuda else time.perf_counter()
            if cuda:
                end.record()
            self.events.append((name, start, end))
            self.census.scope = previous

    def consume(self) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds: Counter = Counter()
        for name, start, end in self.events:
            seconds[name] += (
                start.elapsed_time(end) / 1000 if self.device.type == "cuda" else end - start
            )
        return {
            "seconds_by_stage": dict(seconds),
            "sync_capable_ops_by_stage": {k: dict(v) for k, v in self.census.counts.items()},
            "nested_stages_do_not_sum": True,
            "scope": "instrumented diagnostics; op counts are not measured stall durations",
        }


def cell_specs() -> list[dict[str, Any]]:
    return [
        dict(regime=regime, worlds=worlds, roster=roster)
        for regime in ("early", "mature")
        for worlds in (128, 1024)
        for roster in ("mixed", "homogeneous")
    ]


def factorial_ruler(
    cfg: Any, fixtures: list[dict[str, Any]], **kwargs: Any
) -> list[dict[str, Any]]:
    """Eight fixed eight-tick probes, never scientific parents or forecast credit."""
    from research.spatial_student_pilot.collect import collect_batch

    result = []
    homogeneous = [f for f in fixtures if f["roster_index"] == fixtures[0]["roster_index"]]
    for spec in cell_specs():
        count = spec["worlds"]
        pool = fixtures if spec["roster"] == "mixed" else homogeneous
        selected = [pool[i % len(pool)] for i in range(count)]
        census = SyncCensus()
        with census:
            batch = collect_batch(
                cfg,
                range(count),
                **kwargs,
                ruler_ticks=8,
                ruler_fixtures=selected if spec["regime"] == "mature" else None,
                ruler_roster_indices=(
                    [fixtures[0]["roster_index"]] * count
                    if spec["roster"] == "homogeneous"
                    else None
                ),
                ruler_census=census,
            )
        samples = batch["tick_samples"]
        result.append(
            dict(
                spec,
                ticks=len(samples),
                warmup_ticks=2,
                measured_tick_seconds=[s["seconds"] for s in samples[2:]],
                stage_diagnostics=batch["stage_diagnostics"],
                scientific_result=False,
            )
        )
        kwargs["progress"]({"phase": "factorial_cell", **spec, "ticks": len(samples)})
    return result
