"""Coordinator-side supervision and bounded-work accounting for Ape-X runs.

This module deliberately depends only on process-like objects (``is_alive`` and
``exitcode``).  The buffer process and actors remain independently owned, while
the coordinator gets one fail-closed place to decide whether continuing a run is
safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic
from typing import Any, Mapping, Sequence


class ApexRuntimeFailure(RuntimeError):
    """Raised when a supervised Ape-X child exits or stops reporting health."""


class ApexControlledStop(RuntimeError):
    """A declared budget ended startup or training without a runtime failure."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


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
        if self.max_wall_time_seconds is not None and (
            not math.isfinite(self.max_wall_time_seconds) or self.max_wall_time_seconds <= 0
        ):
            raise ValueError("max_wall_time_seconds must be positive when set")


@dataclass(frozen=True)
class ApexRuntimeSnapshot:
    """Auditable coordinator counters captured in checkpoints and receipts."""

    learner_updates: int
    environment_transitions: int
    agent_transitions: int
    replay_rows_emitted: int
    learner_samples: int
    policy_version: int
    elapsed_seconds: float
    actor_heartbeat_ages: Mapping[int, float | None]
    reserved_environment_frames: int = 0


class SharedActorProgress:
    """Lossless latest counters and heartbeat for one actor process.

    The detailed stats queue is intentionally not used for supervision: it may
    fill while the coordinator is busy. These locked values are the correctness
    channel and are overwritten, never queued.
    """

    def __init__(self, context: Any) -> None:
        self.lock = context.Lock()
        self.environment_frames = context.Value("q", 0, lock=False)
        self.agent_transitions = context.Value("q", 0, lock=False)
        self.replay_rows_emitted = context.Value("q", 0, lock=False)
        self.policy_version = context.Value("q", 0, lock=False)
        self.heartbeat_monotonic = context.Value("d", 0.0, lock=False)

    def report(
        self,
        *,
        environment_frames: int,
        agent_transitions: int,
        replay_rows_emitted: int,
        policy_version: int,
        heartbeat_monotonic: float,
    ) -> None:
        if not self.lock.acquire(timeout=0.1):
            raise ApexRuntimeFailure("actor progress lock is unavailable")
        try:
            self.environment_frames.value = int(environment_frames)
            self.agent_transitions.value = int(agent_transitions)
            self.replay_rows_emitted.value = int(replay_rows_emitted)
            self.policy_version.value = int(policy_version)
            self.heartbeat_monotonic.value = float(heartbeat_monotonic)
        finally:
            self.lock.release()

    def snapshot(self) -> dict[str, int | float]:
        if not self.lock.acquire(timeout=0.1):
            raise ApexRuntimeFailure("actor progress lock is unavailable")
        try:
            return {
                "environment_frames": int(self.environment_frames.value),
                "agent_transitions": int(self.agent_transitions.value),
                "replay_rows_emitted": int(self.replay_rows_emitted.value),
                "policy_version": int(self.policy_version.value),
                "heartbeat_monotonic": float(self.heartbeat_monotonic.value),
            }
        finally:
            self.lock.release()


class SharedEnvironmentFrameBudget:
    """Globally reserve an environment frame before an actor advances its world."""

    def __init__(self, context: Any, maximum: int) -> None:
        if not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("maximum must be a positive integer")
        self.maximum = maximum
        self.lock = context.Lock()
        self.reserved = context.Value("q", 0, lock=False)

    def reserve_one(self) -> bool:
        if not self.lock.acquire(timeout=0.1):
            raise ApexRuntimeFailure("environment frame budget lock is unavailable")
        try:
            if self.maximum is not None and self.reserved.value >= self.maximum:
                return False
            self.reserved.value += 1
            return True
        finally:
            self.lock.release()

    def reserved_frames(self) -> int:
        if not self.lock.acquire(timeout=0.1):
            raise ApexRuntimeFailure("environment frame budget lock is unavailable")
        try:
            return int(self.reserved.value)
        finally:
            self.lock.release()


class ApexRuntimeSupervisor:
    """Fail closed on child exit, stale actor health, or declared work limits."""

    def __init__(
        self,
        *,
        actors: Sequence[object],
        buffer_process: object,
        budgets: ApexRunBudgets,
        heartbeat_timeout_seconds: float = 30.0,
        clock=monotonic,
    ) -> None:
        if not math.isfinite(heartbeat_timeout_seconds) or heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive")
        self.actors = list(actors)
        self.buffer_process = buffer_process
        self.budgets = budgets
        self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        self.actor_started_at: dict[int, float] = {
            index: float(clock()) for index, _actor in enumerate(self.actors)
        }
        self._clock = clock
        self.started_at = float(clock())

    def register_actor(self, actor: object) -> None:
        """Start monitoring an actor from the instant its process is launched."""
        self.actors.append(actor)
        self.actor_started_at[len(self.actors) - 1] = float(self._clock())

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
                now
                - (
                    self.actor_started_at[index]
                    if heartbeat is None or float(heartbeat) <= 0.0
                    else float(heartbeat)
                ),
            )
        return ages

    def current_budget_cause(
        self,
        *,
        learner_updates: int,
        reserved_environment_frames: int,
    ) -> str | None:
        """Return a declared stop cause before treating a normal actor exit as failure."""
        if learner_updates >= self.budgets.max_learner_updates:
            return "learner_update_budget"
        if (
            self.budgets.max_environment_transitions is not None
            and reserved_environment_frames >= self.budgets.max_environment_transitions
        ):
            return "environment_transition_budget"
        if self._clock() - self.started_at >= (self.budgets.max_wall_time_seconds or float("inf")):
            return "wall_time_budget"
        return None

    def check_children(
        self,
        actor_stats: Mapping[int, Mapping[str, object]],
        expected_stop_cause: str | None = None,
    ) -> dict[int, float | None]:
        """Raise a precise failure before a dead/stalled child can look like warmup."""
        if not self._is_alive(self.buffer_process):
            raise ApexRuntimeFailure(
                f"buffer process exited (exitcode={self._exit_code(self.buffer_process)!r})"
            )
        ages = self.heartbeat_ages(actor_stats)
        for index, actor in enumerate(self.actors):
            if not self._is_alive(actor):
                if (
                    expected_stop_cause == "environment_transition_budget"
                    and self._exit_code(actor) == 0
                ):
                    continue
                raise ApexRuntimeFailure(
                    f"actor {index} exited (exitcode={self._exit_code(actor)!r})"
                )
            age = ages[index]
            if (
                expected_stop_cause is None
                and age is not None
                and age > self.heartbeat_timeout_seconds
            ):
                raise ApexRuntimeFailure(
                    f"actor {index} heartbeat stale for {age:.2f}s "
                    f"(limit={self.heartbeat_timeout_seconds:.2f}s)"
                )
        return ages

    def stop_cause(self, snapshot: ApexRuntimeSnapshot) -> str | None:
        """Return the first reached hard budget, preserving declared precedence."""
        return self.current_budget_cause(
            learner_updates=snapshot.learner_updates,
            reserved_environment_frames=snapshot.reserved_environment_frames,
        )


def stop_processes(processes: Sequence[object], timeout_seconds: float = 5.0) -> None:
    """Join, terminate, then kill all process-like children or raise on survivors."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    def wait_phase(candidates: list[object]) -> None:
        deadline = monotonic() + timeout_seconds
        for process in candidates:
            join = getattr(process, "join", None)
            if callable(join):
                join(timeout=max(0.0, deadline - monotonic()))

    active = [process for process in processes if ApexRuntimeSupervisor._is_alive(process)]
    wait_phase(active)
    active = [process for process in active if ApexRuntimeSupervisor._is_alive(process)]
    for process in active:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
    wait_phase(active)
    active = [process for process in active if ApexRuntimeSupervisor._is_alive(process)]
    for process in active:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
    wait_phase(active)
    survivors = [process for process in active if ApexRuntimeSupervisor._is_alive(process)]
    if survivors:
        raise ApexRuntimeFailure(f"failed to stop {len(survivors)} Ape-X child process(es)")
