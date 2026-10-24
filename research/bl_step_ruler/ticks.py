"""World-step counters for the unchanged F0 collector (Step -1 ruler).

Counting is purely observational. It wraps, inside one worker process only:

* ``WatchDriver.step``: one call is exactly one ``GridBatchSim.step`` of one
  E=1 six-seat world, i.e. one world-step, the unit of the GPU R2/R6 rates.
  ``GateDriver.step`` delegates to it, so gate worlds are counted once.
* ``WatchDriver.restore`` / ``WatchDriver.snapshot``: restore and snapshot counts.
* ``factory._restore_root``: everything inside it is the ``replay`` phase.
* ``factory._branch``: everything inside it is the ``branch`` phase. This covers
  doom alternatives, ordinary alternatives, bare reroll audits and H1440
  continuations; they are deliberately not separated (no outcome-bearing split).

Everything else is the ``play`` phase (natural trajectory, death snapshot and the
post-death restore). The collector's bytes, control flow, RNG and exports are not
changed; the wrappers call the originals with identical arguments.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

PHASES = ("play", "replay", "branch")


class TickCounter:
    """Per-phase counters; phases nest as a stack (innermost wins)."""

    def __init__(self) -> None:
        self.stack: list[str] = ["play"]
        self.ticks = dict.fromkeys(PHASES, 0)
        self.restores = dict.fromkeys(PHASES, 0)
        self.snapshots = dict.fromkeys(PHASES, 0)
        self.snapshot_bytes = dict.fromkeys(PHASES, 0)
        self.calls = {"replay": 0, "branch": 0}
        self.phase_seconds = {"replay": 0.0, "branch": 0.0}

    @property
    def phase(self) -> str:
        return self.stack[-1]

    def world_steps(self) -> int:
        return sum(self.ticks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticks": dict(self.ticks),
            "world_steps": self.world_steps(),
            "restores": dict(self.restores),
            "snapshots": dict(self.snapshots),
            "snapshot_bytes": dict(self.snapshot_bytes),
            "calls": dict(self.calls),
            "phase_seconds": dict(self.phase_seconds),
        }


def _targets() -> tuple[Any, Any]:
    from research.bl_labels import factory
    from research.bl_served_search.b2_engine import WatchDriver

    return WatchDriver, factory


@contextmanager
def instrument(
    counter: TickCounter, *, driver_cls: Any = None, factory_module: Any = None
) -> Iterator[TickCounter]:
    """Install the observational wrappers; always restore the originals."""
    if driver_cls is None or factory_module is None:
        real_driver, real_factory = _targets()
        driver_cls = driver_cls or real_driver
        factory_module = factory_module or real_factory
    step = driver_cls.__dict__["step"]
    snapshot = driver_cls.__dict__["snapshot"]
    restore = driver_cls.__dict__["restore"]
    if not isinstance(restore, classmethod):
        raise TypeError("WatchDriver.restore must be a classmethod")
    restore_root = factory_module._restore_root
    branch = factory_module._branch

    def counted_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        event = step(self, *args, **kwargs)
        counter.ticks[counter.phase] += 1
        return event

    def counted_snapshot(self: Any) -> bytes:
        payload = snapshot(self)
        counter.snapshots[counter.phase] += 1
        counter.snapshot_bytes[counter.phase] += len(payload)
        return payload

    def counted_restore(cls: Any, payload: bytes) -> Any:
        value = restore.__func__(cls, payload)
        counter.restores[counter.phase] += 1
        return value

    def phased(name: str, original: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counter.stack.append(name)
            counter.calls[name] += 1
            started = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                counter.phase_seconds[name] += time.monotonic() - started
                counter.stack.pop()

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    driver_cls.step = counted_step
    driver_cls.snapshot = counted_snapshot
    driver_cls.restore = classmethod(counted_restore)
    factory_module._restore_root = phased("replay", restore_root)
    factory_module._branch = phased("branch", branch)
    try:
        yield counter
    finally:
        driver_cls.step = step
        driver_cls.snapshot = snapshot
        driver_cls.restore = restore
        factory_module._restore_root = restore_root
        factory_module._branch = branch
