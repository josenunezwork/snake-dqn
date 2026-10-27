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
    """Compile a static block; validate specializations and persist eager fallback.

    The first invocation for each tensor specialization is a setup path.  Its
    compiler and eager validation work is deliberately excluded from the
    benchmark's steady-state phase timers and reported separately.
    """

    def __init__(
        self,
        eager: Callable,
        mode: str,
        report: dict[str, Any],
        name: str,
        *,
        backend: str | Callable | None = None,
    ) -> None:
        self.eager, self.report = eager, report
        self.validated = False
        self._validated_specializations: set[tuple[Any, ...]] = set()
        self._compiled_specializations: set[tuple[Any, ...]] = set()
        self.report[name] = {
            "status": "pending",
            "mode": mode,
            "backend": backend or "inductor",
            "fallback_reason": None,
            "validation_policy": "compiled output and no-grad eager reference share caller "
            "numerical globals; floats rtol=atol=2e-5",
            "compile_factory_seconds": 0.0,
            "compile_specialization_seconds": 0.0,
            "compile_specialization_calls": 0,
            "validation_seconds": 0.0,
            "validation_calls": 0,
        }
        self.entry = self.report[name]
        started = time.perf_counter()
        try:
            # No whole-step graph capture: simulator loops remain outside this boundary.
            kwargs: dict[str, Any] = {"mode": mode, "dynamic": False, "fullgraph": True}
            if backend is not None:
                kwargs["backend"] = backend
            self.compiled = torch.compile(eager, **kwargs)
        except Exception as exc:
            self._fallback(exc)
        finally:
            self.entry["compile_factory_seconds"] = time.perf_counter() - started

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

    @staticmethod
    def _argument_signature(value: Any) -> Any:
        """Return a stable, non-data-dependent compilation specialization key."""
        if isinstance(value, torch.Tensor):
            return (
                "tensor",
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                str(value.dtype),
                str(value.device),
                value.requires_grad,
            )
        if isinstance(value, (tuple, list)):
            return (
                type(value).__name__,
                tuple(CompiledBlock._argument_signature(item) for item in value),
            )
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (key, CompiledBlock._argument_signature(item))
                    for key, item in sorted(value.items())
                ),
            )
        return (type(value).__name__, repr(value))

    @classmethod
    def _specialization_key(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        return (
            "grad_enabled",
            torch.is_grad_enabled(),
            "inference_mode",
            torch.is_inference_mode_enabled(),
            "numerical_globals",
            (
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.get_float32_matmul_precision(),
            ),
            "args",
            tuple(cls._argument_signature(arg) for arg in args),
            "kwargs",
            tuple((key, cls._argument_signature(value)) for key, value in sorted(kwargs.items())),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.compiled is not None:
            try:
                specialization = self._specialization_key(args, kwargs)
                first_specialization = specialization not in self._compiled_specializations
                if first_specialization:
                    compile_started = time.perf_counter()
                    self._compiled_specializations.add(specialization)
                # Compile and execute once in the caller's exact numerical and
                # autograd state. Global numerical guards (notably TF32) are
                # part of the specialization key, so neither validation nor a
                # later measured call silently compiles a different graph.
                result = self.compiled(*args, **kwargs)
                if first_specialization:
                    self.entry["compile_specialization_calls"] += 1
                    self.entry["compile_specialization_seconds"] += (
                        time.perf_counter() - compile_started
                    )
                if specialization not in self._validated_specializations:
                    validation_started = time.perf_counter()
                    # The reference must not retain a second autograd graph.
                    # It deliberately inherits the caller's TF32 and other
                    # numerical globals, matching the graph just compiled.
                    with torch.no_grad():
                        expected = self.eager(*args, **kwargs)
                        self._compare(result, expected)
                    self._validated_specializations.add(specialization)
                    self.validated = True
                    self.entry["validation_calls"] += 1
                    self.entry["validation_seconds"] += time.perf_counter() - validation_started
                    self.entry["output_check"] = (
                        "no-grad eager match under caller numerical globals"
                    )
            except Exception as exc:
                self._fallback(exc)
            else:
                self.entry["status"] = "compiled"
                return result
        return self.eager(*args, **kwargs)
