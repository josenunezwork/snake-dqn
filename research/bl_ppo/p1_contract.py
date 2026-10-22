"""Frozen P1 execution dimensions; no CLI or environment override is admitted."""

from dataclasses import dataclass


def whole_world_pause(requested: int, worlds: int) -> int:
    """Round up once to a complete collector step, preserving literal zero."""
    if requested < 0 or worlds < 1:
        raise ValueError("invalid whole-world pause")
    return ((requested + worlds - 1) // worlds) * worlds


@dataclass(frozen=True)
class ExecutionShape:
    """Production dimensions, replaceable only by in-process test fixtures."""

    target: int = 100_000_000
    worlds: int = 128
    processes: int = 12
    rollout_steps: int = 128
    qualification_requested: int = 1_000_000
    tranche_size: int = 20_000_000

    @property
    def qualification_pause(self) -> int:
        return whole_world_pause(self.qualification_requested, self.worlds)


SHAPE = ExecutionShape()
