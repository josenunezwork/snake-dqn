"""Coordinator-side supervision and bounded-work accounting for Ape-X runs.

This module deliberately depends only on process-like objects (``is_alive`` and
``exitcode``).  The buffer process and actors remain independently owned, while
the coordinator gets one fail-closed place to decide whether continuing a run is
safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Mapping, Sequence


class ApexRuntimeFailure(RuntimeError):
    """Raised when a supervised Ape-X child exits or stops reporting health."""


@dataclass(frozen=True)
class ApexRunBudgets:
    """Independent upper bounds for work performed by one coordinator run."""

    max_learner_updates: int
    max_environment_transitions: int | None = None
    max_wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_learner_updates <= 0:
            raise ValueError("max_learner_updates must be positive")
        if self.max_environment_transitions is not None and self.max_environment_transitions <= 0:
            raise ValueError("max_environment_transitions must be positive when set")
        if self.max_wall_time_seconds is not None and self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive when set")


@dataclass(frozen=True)
class ApexRuntimeSnapshot:
    """Auditable coordinator counters captured in checkpoints and receipts."""

    learner_updates: int
    environment_transitions: int
    replay_rows_emitted: int
    learner_samples: int
    policy_version: int
    elapsed_seconds: float
    actor_heartbeat_ages: Mapping[int, float | None]
    environment_transition_reservation: int = 0


class ApexRuntimeSupervisor:
    """Fail closed on child exit, stale actor health, or declared work limits."""

    def __init__(
        self,
        *,
        actors: Sequence[object],
        buffer_process: object,
        budgets: ApexRunBudgets,
        heartbeat_timeout_seconds: float = 30.0,
        environment_transition_reservation: int = 0,
        clock=monotonic,
    ) -> None:
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive")
        if environment_transition_reservation < 0:
            raise ValueError("environment_transition_reservation must be non-negative")
        self.actors = list(actors)
        self.buffer_process = buffer_process
        self.budgets = budgets
        self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        self.environment_transition_reservation = int(environment_transition_reservation)
        self._clock = clock
        self.started_at = float(clock())

    @staticmethod
    def _is_alive(child: object) -> bool:
        value = getattr(child, "is_alive", False)
        return bool(value() if callable(value) else value)

    @staticmethod
    def _exit_code(child: object) -> object:
        return getattr(child, "exitcode", getattr(child, "_exitcode", None))

    def heartbeat_ages(
        self, actor_stats: Mapping[int, Mapping[str, object]]
    ) -> dict[int, float | None]:
        """Return per-actor heartbeat age without assuming every actor has reported."""
        now = float(self._clock())
        ages: dict[int, float | None] = {}
        for index, _actor in enumerate(self.actors):
            heartbeat = actor_stats.get(index, {}).get("heartbeat_monotonic")
            ages[index] = max(
                0.0,
                now - (self.started_at if heartbeat is None else float(heartbeat)),
            )
        return ages

    def check_children(
        self, actor_stats: Mapping[int, Mapping[str, object]]
    ) -> dict[int, float | None]:
        """Raise a precise failure before a dead/stalled child can look like warmup."""
        if not self._is_alive(self.buffer_process):
            raise ApexRuntimeFailure(
                f"buffer process exited (exitcode={self._exit_code(self.buffer_process)!r})"
            )
        ages = self.heartbeat_ages(actor_stats)
        for index, actor in enumerate(self.actors):
            if not self._is_alive(actor):
                raise ApexRuntimeFailure(
                    f"actor {index} exited (exitcode={self._exit_code(actor)!r})"
                )
            age = ages[index]
            if age is not None and age > self.heartbeat_timeout_seconds:
                raise ApexRuntimeFailure(
                    f"actor {index} heartbeat stale for {age:.2f}s "
                    f"(limit={self.heartbeat_timeout_seconds:.2f}s)"
                )
        return ages

    def stop_cause(self, snapshot: ApexRuntimeSnapshot) -> str | None:
        """Return the first reached hard budget, preserving declared precedence."""
        if snapshot.learner_updates >= self.budgets.max_learner_updates:
            return "learner_update_budget"
        if (
            self.budgets.max_environment_transitions is not None
            and snapshot.environment_transitions + self.environment_transition_reservation
            >= self.budgets.max_environment_transitions
        ):
            return "environment_transition_budget"
        if (
            self.budgets.max_wall_time_seconds is not None
            and snapshot.elapsed_seconds >= self.budgets.max_wall_time_seconds
        ):
            return "wall_time_budget"
        return None


def stop_processes(processes: Sequence[object], timeout_seconds: float = 5.0) -> None:
    """Join, terminate, then kill all process-like children or raise on survivors."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    survivors: list[object] = []
    for process in processes:
        alive = ApexRuntimeSupervisor._is_alive(process)
        if not alive:
            continue
        join = getattr(process, "join", None)
        if callable(join):
            join(timeout=timeout_seconds)
        if ApexRuntimeSupervisor._is_alive(process):
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
            if callable(join):
                join(timeout=timeout_seconds)
        if ApexRuntimeSupervisor._is_alive(process):
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            if callable(join):
                join(timeout=timeout_seconds)
        if ApexRuntimeSupervisor._is_alive(process):
            survivors.append(process)
    if survivors:
        raise ApexRuntimeFailure(f"failed to stop {len(survivors)} Ape-X child process(es)")
