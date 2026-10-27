"""Sandbox-only phase events and explicit per-subgraph compiler fallbacks."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Iterator

import torch


class PhaseProfiler:
    """Record stream events without synchronizing within the collection loop."""

    def __init__(self, device: torch.device, ticks: int) -> None:
        self.device, self.ticks = device, ticks
        self.events: list[tuple[str, Any, Any]] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.device.type == "cuda":
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            try:
                yield
            finally:
                end.record()
                self.events.append((name, begin, end))
        else:
            begin = time.perf_counter()
            try:
                yield
            finally:
                self.events.append((name, begin, time.perf_counter()))

    def finish(self) -> dict[str, Any]:
        """The only event synchronization is after all horizon work is queued."""
        cuda = self.device.type == "cuda"
        if cuda and self.events:
            self.events[-1][2].synchronize()
        milliseconds: dict[str, float] = {}
        for name, begin, end in self.events:
            ms = begin.elapsed_time(end) if cuda else (end - begin) * 1000
            milliseconds[name] = milliseconds.get(name, 0.0) + ms
        return {
            "mode": "cuda_event" if cuda else "cpu_wall",
            "ticks": self.ticks,
            "milliseconds": milliseconds,
            "ms_per_tick": {name: ms / self.ticks for name, ms in milliseconds.items()},
            "nested_phases": [],
        }


class CompiledBlock:
    """Compile a static block; record setup/lazy failures and persist eager fallback."""

    def __init__(self, eager: Callable, mode: str, report: dict[str, Any], name: str) -> None:
        self.eager, self.report = eager, report
        self.validated = False
        self.report[name] = {"status": "pending", "mode": mode, "fallback_reason": None}
        self.entry = self.report[name]
        try:
            # No whole-step graph capture: simulator loops remain outside this boundary.
            self.compiled = torch.compile(eager, mode=mode, dynamic=False, fullgraph=True)
        except Exception as exc:
            self._fallback(exc)

    def _fallback(self, exc: Exception) -> None:
        self.compiled = None
        self.entry.update(
            status="eager_fallback", fallback_reason=f"{type(exc).__name__}: {exc}"[:2000]
        )

    @staticmethod
    def _compare(actual: Any, expected: Any) -> None:
        if isinstance(expected, torch.Tensor):
            if expected.is_floating_point():
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            elif not torch.equal(actual, expected):
                raise ValueError("compiled integer/bool output differs from eager")
        elif is_dataclass(expected):
            for field in fields(expected):
                CompiledBlock._compare(getattr(actual, field.name), getattr(expected, field.name))
        elif isinstance(expected, (tuple, list)):
            for left, right in zip(actual, expected, strict=True):
                CompiledBlock._compare(left, right)
        elif actual != expected:
            raise ValueError("compiled output differs from eager")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.compiled is not None:
            try:
                result = self.compiled(*args, **kwargs)
                if not self.validated:
                    self._compare(result, self.eager(*args, **kwargs))
                    self.validated = True
                    self.entry["output_check"] = (
                        "eager match: exact integers; floats rtol=atol=2e-5"
                    )
            except Exception as exc:
                self._fallback(exc)
            else:
                self.entry["status"] = "compiled"
                return result
        return self.eager(*args, **kwargs)
