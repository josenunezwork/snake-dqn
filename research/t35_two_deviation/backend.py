"""Exact CPU backend over the T3 dense bank (imports T34 shadow / T3 / T1 code; modifies none).

World reconstruction, greedy shadow and alternative expansion are the T34 shadow
M2 functions (``bank_measure``), whose rollouts are T3's ``batched_rollout``
(B8 rowwise greedy continuation, advisory mask, eps 0, no veto, real opponents,
cloned world RNG).  This module only adds a forced-first-action branch with
retained prepared states and a longer confirmation rollout.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from research.t3_rollout_labels_20261015.engine import batched_rollout
from research.t34_shadow import bank_measure as bm


class Deadline:
    """Fail closed when the absolute monotonic deadline passes (duck-types T34 ``Budget``)."""

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("deadline must be positive")
        self.started = time.monotonic()
        self.seconds = float(seconds)

    def check(self) -> None:
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("T35 event wall budget expired")

    def elapsed(self) -> float:
        return time.monotonic() - self.started


class BankBackend:
    """``search.two_deviation_search`` backend on prepared T1 ``Replay`` states."""

    def __init__(self, deadline: Deadline) -> None:
        self.deadline = deadline

    def check(self) -> None:
        self.deadline.check()

    def expand(self, state: Any, decision: int, doom: int) -> dict[str, Any]:
        return bm.expand_snapshot(state, decision, doom, self.deadline)

    def branch(
        self, state: Any, decision: int, action: int, stop: int
    ) -> tuple[int | None, dict[int, Any], int]:
        """Force ``action`` at ``decision``, then B8 greedy; keep every prepared state."""
        self.check()
        branch = state.clone(prepared=True)
        branch.first(int(action))
        steps = 1
        states: dict[int, Any] = {}
        if not bool(branch.sim.alive[0, 0]):
            return int(decision), states, steps
        while int(branch.sim.frame[0]) < stop:
            self.check()
            current = int(branch.sim.frame[0])

            def capture(replay: Any) -> None:
                # Mirrors bank_measure.greedy_shadow: frame is one-based during preparation.
                states[int(replay.sim.frame[0]) - 1] = replay.clone(prepared=True)

            branch.step(observer=capture)
            steps += 1
            if not bool(branch.sim.alive[0, 0]):
                return current, states, steps
        return None, states, steps

    def confirm(self, state: Any, decision: int, action: int, depth: int) -> dict[str, Any]:
        self.check()
        outcome = batched_rollout(state.clone(prepared=True), [int(action)], decision, depth=depth)
        row = outcome[int(action)]
        return {
            "survives": bool(row["survives"]),
            "death_decision": row["death_decision"],
            "env_frames": int(row["env_frames"]),
            "horizon": int(row["horizon"]),
        }


def trimmed_source(row: dict[str, Any], start: int) -> dict[str, Any]:
    """Keep saved triggers through the first one at/after ``start`` (reconstruction stop).

    Later triggers cannot affect the pre-action state at ``start``; every kept
    trigger is still checked by ``reconstruct_sources``.
    """
    decisions = sorted(int(t["decision"]) for t in row["trigger_rows"])
    after = [d for d in decisions if d >= start]
    if not after:
        raise ValueError(("no saved trigger at or after the event start", start))
    limit = after[0]
    return {**row, "trigger_rows": [t for t in row["trigger_rows"] if int(t["decision"]) <= limit]}


def prepare_event(
    world: str, start: int, doom: int, deadline: Deadline, context: Any = None
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Rebuild T3's timeline to ``start`` and return greedy-shadow states for its window."""
    from research.b8_death_census_20261014 import census

    source = bm.source_receipt(world)
    actual = bm.sha(Path(source["checkpoint"]["path"]))
    if actual != source["checkpoint"]["sha256"]:
        raise RuntimeError("T3 source checkpoint bytes no longer match its immutable receipt")
    context = census.load_context() if context is None else context
    started = time.perf_counter()
    snapshots, checks, recon_steps = bm.reconstruct_sources(
        trimmed_source(source, start), {start}, deadline, context
    )
    reconstruct_seconds = time.perf_counter() - started
    started = time.perf_counter()
    shadow_doom, states, step_times = bm.greedy_shadow(snapshots[start], start, deadline)
    if shadow_doom != doom:
        raise AssertionError(("greedy doom mismatch", world, start, doom, shadow_doom))
    return states, {
        "t3_receipt_sha256": bm.sha(bm.T3_CEILING / f"{world}.json"),
        "checkpoint_sha256": actual,
        "saved_trigger_checks": checks,
        "reconstruction_row_steps": recon_steps,
        "reconstruction_seconds": reconstruct_seconds,
        "shadow_row_steps": len(step_times),
        "shadow_seconds": time.perf_counter() - started,
    }
