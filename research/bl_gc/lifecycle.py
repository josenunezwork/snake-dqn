"""d.3 lifecycle/retention qualification for round-25 startup GC freezing."""

from __future__ import annotations

import argparse
import gc
import json
import weakref
from pathlib import Path
from typing import Any

from research.bl_gc.common import initialize, new_session, write_result


class _FrozenCycle:
    """Deliberately cyclic diagnostic object used only to prove the detector."""

    def __init__(self) -> None:
        self.link = self


class LifecycleFailure(RuntimeError):
    """Carry a complete, immutable failure receipt back to the CLI."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__(str(result["failure"]["message"]))


def _collect_twice() -> None:
    gc.collect()
    gc.collect()


def run(policy: str, cycles: int = 20, ticks: int = 50) -> dict[str, Any]:
    """Exercise twenty real session replacement lifecycles in one fresh process."""
    if cycles != 20 or ticks != 50:
        raise ValueError("d.3 requires exactly 20 cycles and 50 ticks per cycle")
    receipt, initial = initialize(policy, "lifecycle")
    baseline_freeze = int(receipt["frozen_count"])
    refs: list[tuple[weakref.ReferenceType[Any], ...]] = []
    object_counts: list[int] = []
    for cycle in range(int(cycles)):
        session, roster = new_session(2026102300 + cycle)
        for _ in range(int(ticks)):
            session.step()
            session.snapshot()
        pre_reset_game = session.game
        pre_reset_hero = next(
            snake for snake in pre_reset_game.snakes if int(snake.id) == int(session.hero_id)
        )
        pre_reset_policy = session.policy
        pre_reset_refs = (
            weakref.ref(pre_reset_game),
            weakref.ref(pre_reset_hero),
            weakref.ref(pre_reset_policy),
        )
        session.reset_game()
        reset_game = session.game
        reset_hero = next(
            snake for snake in reset_game.snakes if int(snake.id) == int(session.hero_id)
        )
        reset_refs = (
            weakref.ref(reset_game),
            weakref.ref(reset_hero),
            weakref.ref(session.policy),
        )
        session._build(session.checkpoint_path, mode=session.mode)
        rebuilt_game = session.game
        rebuilt_hero = next(
            snake for snake in rebuilt_game.snakes if int(snake.id) == int(session.hero_id)
        )
        rebuilt_refs = (
            weakref.ref(rebuilt_game),
            weakref.ref(rebuilt_hero),
            weakref.ref(session.policy),
            weakref.ref(session),
        )
        session.close(wait=True)
        del (
            pre_reset_game,
            pre_reset_hero,
            pre_reset_policy,
            reset_game,
            reset_hero,
            rebuilt_game,
            rebuilt_hero,
            session,
        )
        _collect_twice()
        cycle_refs = pre_reset_refs + reset_refs + rebuilt_refs
        weakrefs_cleared = not any(
            reference() is not None for bundle in refs + [cycle_refs] for reference in bundle
        )
        observed_freeze = int(gc.get_freeze_count())
        if not weakrefs_cleared or observed_freeze != baseline_freeze:
            message = (
                f"cycle {cycle}: gameplay object survived collection"
                if not weakrefs_cleared
                else "freeze count changed during session lifecycle"
            )
            raise LifecycleFailure(
                {
                    "schema": "bl-gc-lifecycle/v1",
                    "status": "FAIL",
                    "policy": policy,
                    "cycles_requested": int(cycles),
                    "ticks_per_cycle": int(ticks),
                    "weakrefs_cleared": weakrefs_cleared,
                    "negative_control_retained_while_frozen": None,
                    "negative_control_released_after_unfreeze": None,
                    "gameplay_objects_released_by_teardown": None,
                    "startup_freeze_count": baseline_freeze,
                    "failure": {
                        "cycle": cycle + 1,
                        "message": message,
                        "weakrefs_cleared": weakrefs_cleared,
                        "expected_freeze_count": baseline_freeze,
                        "observed_freeze_count": observed_freeze,
                    },
                    "attestation": initial,
                    "roster": roster,
                }
            )
        refs.append(cycle_refs)
        object_counts.append(len(gc.get_objects()))

    if len(object_counts) >= 20 and object_counts[-1] > object_counts[4] + 2000:
        raise RuntimeError("tracked object growth after cycle 5 exceeds 2,000")

    negative = _FrozenCycle()
    negative_ref = weakref.ref(negative)
    gc.freeze()
    del negative
    _collect_twice()
    if negative_ref() is None:
        raise RuntimeError("negative control was unexpectedly collected while frozen")
    gc.unfreeze()
    _collect_twice()
    if negative_ref() is not None:
        raise RuntimeError("negative control survived unfreeze plus collection")
    if any(reference() is not None for bundle in refs for reference in bundle):
        raise RuntimeError("unfreeze released retained gameplay object")
    return {
        "schema": "bl-gc-lifecycle/v1",
        "policy": policy,
        "cycles": int(cycles),
        "ticks_per_cycle": int(ticks),
        "weakrefs_cleared": True,
        "negative_control_retained_while_frozen": True,
        "negative_control_released_after_unfreeze": True,
        "gameplay_objects_released_by_teardown": 0,
        "startup_freeze_count": baseline_freeze,
        "tracked_objects": {"cycle_5": object_counts[4], "cycle_20": object_counts[-1]},
        "attestation": {**initial, "final": attest_placeholder(receipt)},
    }


def attest_placeholder(receipt: dict[str, Any]) -> dict[str, Any]:
    """Keep lifecycle's post-teardown receipt explicit without rebuilding a game."""
    return {"gc_policy": receipt, "freeze_count_after_teardown": int(gc.get_freeze_count())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("default", "freeze_startup"), required=True)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.policy, args.cycles, args.ticks)
    except LifecycleFailure as exc:
        write_result(args.out, exc.result)
        print(json.dumps({"status": "FAIL", "out": str(args.out), **exc.result["failure"]}))
        raise SystemExit(1)
    write_result(args.out, result)
    print(json.dumps({"status": "PASS", "out": str(args.out)}))


if __name__ == "__main__":
    main()
