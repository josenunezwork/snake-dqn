"""Amended untimed d.3 lifecycle/retention qualification (JUDGMENT_D3.md)."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import weakref
from collections import Counter
from pathlib import Path
from typing import Any

from research.bl_gc.common import initialize, new_session, write_result


class _FrozenCycle:
    """Deliberately cyclic negative control, never part of serving."""

    def __init__(self) -> None:
        self.link = self


def _collect_twice() -> None:
    gc.collect()
    gc.collect()


def _gameplay_types() -> tuple[type, ...]:
    """Load the census types before the production initialization boundary."""
    from torch.nn import Module

    from src.game.food_manager import FoodManager
    from src.game.game_state import GameState
    from src.game.snake import Snake
    from src.training.apex_policy import ApexPolicy
    from web.backend.ego2s_policy import Ego2sServingPolicy
    from web.backend.raster_policy import RasterServingPolicy
    from web.backend.session import GameSession

    return (
        GameSession,
        GameState,
        Snake,
        FoodManager,
        Ego2sServingPolicy,
        RasterServingPolicy,
        ApexPolicy,
        Module,
    )


def gameplay_census(types: tuple[type, ...]) -> dict[str, Any]:
    """Count instances by type, retaining no objects or census list on return."""
    counts: Counter[str] = Counter()
    for obj in gc.get_objects():
        cls = type(obj)
        module = cls.__module__
        module = module if isinstance(module, str) else ""
        if issubclass(cls, types) or module.startswith(
            ("src.game", "src.model", "web.backend.session")
        ):
            counts[f"{module}.{cls.__qualname__}"] += 1
    return {"total": sum(counts.values()), "types": dict(sorted(counts.items()))}


def freeze_non_increasing(counts: list[int], frozen_count: int) -> bool:
    """I1: every post-startup observation is bounded by its predecessor/receipt."""
    return all(
        0 <= current <= previous <= frozen_count
        for previous, current in zip([frozen_count, *counts], counts)
    )


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def _exercise_cycle(seed: int) -> tuple[tuple[weakref.ReferenceType[Any], ...], dict[str, Any]]:
    """Run the unchanged six-snake, 50-tick reset/rebuild/close/drop sequence."""
    session, roster = new_session(seed)
    for _ in range(50):
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
    reset_hero = next(snake for snake in reset_game.snakes if int(snake.id) == int(session.hero_id))
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
    return pre_reset_refs + reset_refs + rebuilt_refs, roster


def run(policy: str, cycles: int = 20, ticks: int = 50) -> dict[str, Any]:
    """Record all checks, including after a failed invariant; stop only on crash."""
    if cycles != 20 or ticks != 50:
        raise ValueError("d.3 requires exactly 20 cycles and 50 ticks per cycle")
    types = _gameplay_types()
    from web.backend import gc_policy

    result: dict[str, Any] = {
        "schema": "bl-gc-lifecycle/v2",
        "status": "FAIL",
        "policy": policy,
        "cycles_requested": cycles,
        "cycles": 0,
        "ticks_per_cycle": ticks,
        "attestation": _git_state(),
        "startup_receipt": None,
        "birth_census": None,
        "teardown_census_pre_collect": None,
        "startup_freeze_count": None,
        "freeze_calls_before_negative_control": None,
        "freeze_count_trajectory": [],
        "freeze_count_total_decrease": None,
        "cycle_checks": [
            {
                "cycle": c + 1,
                "weakrefs_cleared": None,
                "freeze_count": None,
                "freeze_non_increasing": None,
                "freeze_count_delta": None,
                "freeze_calls": None,
                "single_freeze": None,
                "tracked_objects": None,
            }
            for c in range(cycles)
        ],
        "weakrefs_cleared": None,
        "tracked_object_growth_ok": None,
        "negative_control_retained_while_frozen": None,
        "negative_control_released_after_unfreeze": None,
        "gameplay_objects_released_by_teardown": None,
        "teardown_weakrefs_cleared": None,
        "checks": dict.fromkeys(
            (
                "all_20_cycles",
                "weakrefs",
                "I1",
                "I2",
                "I3",
                "object_growth",
                "negative_retained",
                "negative_released",
                "I4",
                "teardown_weakrefs",
            )
        ),
        "failure": None,
    }
    original_freeze = gc.freeze
    original_apply = gc_policy.apply_startup_policy
    freeze_calls = 0

    def counted_freeze() -> None:
        nonlocal freeze_calls
        freeze_calls += 1
        # Production has just called collect(2). No census objects escape.
        if result["birth_census"] is None:
            result["birth_census"] = gameplay_census(types)
        original_freeze()

    def census_apply(*, stage: str) -> dict[str, Any]:
        if policy == "default":
            gc.collect(2)
            result["birth_census"] = gameplay_census(types)
        return original_apply(stage=stage)

    gc.freeze = counted_freeze
    gc_policy.apply_startup_policy = census_apply
    refs: list[tuple[weakref.ReferenceType[Any], ...]] = []
    try:
        receipt, initial = initialize(policy, "lifecycle")
        # common.initialize is untouched; override its git facts with actual HEAD.
        result["attestation"] = {**initial, **result["attestation"]}
        result["startup_receipt"] = receipt
        baseline = int(receipt["frozen_count"])
        result["startup_freeze_count"] = baseline
        trajectory = result["freeze_count_trajectory"]
        trajectory.append(baseline)
        # Also check the first observation after initialize returns.
        trajectory.append(int(gc.get_freeze_count()))
        expected_calls = int(policy == "freeze_startup")
        checks = result["checks"]
        checks["I3"] = result["birth_census"]["total"] == 0
        checks["I1"] = freeze_non_increasing(trajectory, baseline)
        for cycle in range(cycles):
            cycle_refs, roster = _exercise_cycle(2026102300 + cycle)
            refs.append(cycle_refs)
            _collect_twice()
            observed = int(gc.get_freeze_count())
            row = result["cycle_checks"][cycle]
            row.update(
                weakrefs_cleared=all(r() is None for bundle in refs for r in bundle),
                freeze_count=observed,
                freeze_count_delta=observed - trajectory[-1],
                freeze_non_increasing=freeze_non_increasing([trajectory[-1], observed], baseline),
                freeze_calls=freeze_calls,
                single_freeze=freeze_calls == expected_calls,
                tracked_objects=len(gc.get_objects()),
            )
            trajectory.append(observed)
            result["cycles"] += 1
            result["roster"] = roster
            checks["I1"] = checks["I1"] and row["freeze_non_increasing"]
            checks["weakrefs"] = all(
                r["weakrefs_cleared"] for r in result["cycle_checks"][: cycle + 1]
            )
        checks["all_20_cycles"] = result["cycles"] == 20
        rows = result["cycle_checks"]
        result["freeze_count_total_decrease"] = baseline - trajectory[-1]
        result["freeze_calls_before_negative_control"] = freeze_calls
        result["weakrefs_cleared"] = all(row["weakrefs_cleared"] for row in rows)
        result["tracked_objects"] = {
            "cycle_5": rows[4]["tracked_objects"],
            "cycle_20": rows[19]["tracked_objects"],
        }
        result["tracked_object_growth_ok"] = (
            rows[19]["tracked_objects"] <= rows[4]["tracked_objects"] + 2000
        )
        checks["object_growth"] = result["tracked_object_growth_ok"]
        # I4: reveal the startup generation and census BEFORE any collection.
        gc.unfreeze()
        result["teardown_census_pre_collect"] = gameplay_census(types)
        checks["I4"] = result["teardown_census_pre_collect"]["total"] == 0
        _collect_twice()
        result["teardown_weakrefs_cleared"] = all(r() is None for b in refs for r in b)
        result["gameplay_objects_released_by_teardown"] = sum(
            r() is not None for b in refs for r in b
        )
        checks["teardown_weakrefs"] = result["teardown_weakrefs_cleared"]
        # Counting covers all qualification work; remove only for negative control.
        result["freeze_calls_before_negative_control"] = freeze_calls
        checks["I2"] = freeze_calls == expected_calls and all(row["single_freeze"] for row in rows)
        gc.freeze = original_freeze
        negative = _FrozenCycle()
        negative_ref = weakref.ref(negative)
        gc.freeze()
        del negative
        _collect_twice()
        result["negative_control_retained_while_frozen"] = negative_ref() is not None
        checks["negative_retained"] = result["negative_control_retained_while_frozen"]
        gc.unfreeze()
        _collect_twice()
        result["negative_control_released_after_unfreeze"] = negative_ref() is None
        checks["negative_released"] = result["negative_control_released_after_unfreeze"]
        result["attestation"]["final"] = {"freeze_count_after_teardown": gc.get_freeze_count()}
        result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        gc.freeze = original_freeze
        gc_policy.apply_startup_policy = original_apply
    return result


def diagnose() -> dict[str, Any]:
    """Separate, non-gating baseline and freed-startup-object characterization."""
    types = _gameplay_types()
    result: dict[str, Any] = {
        "schema": "bl-gc-lifecycle-diagnostic/v1",
        "gating": False,
        "historical_unexplained_default_count": 375,
        "attestation": _git_state(),
        "bare_interpreter_freeze_count": int(
            subprocess.run(
                [sys.executable, "-I", "-c", "import gc; print(gc.get_freeze_count())"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        ),
    }
    original_freeze = gc.freeze
    birth: dict[int, str] = {}

    def capture_freeze() -> None:
        result["after_import_closure_freeze_count"] = gc.get_freeze_count()
        result["birth_census"] = gameplay_census(types)
        for obj in gc.get_objects():
            cls = type(obj)
            birth[id(obj)] = f"{cls.__module__}.{cls.__qualname__}"
        original_freeze()

    gc.freeze = capture_freeze
    try:
        receipt, _ = initialize("freeze_startup", "lifecycle-diagnostic")
        result["startup_receipt"] = receipt
        _exercise_cycle(2026102300)
        _collect_twice()
        result["after_cycle_1_freeze_count"] = gc.get_freeze_count()
        gc.unfreeze()
        # Do not collect between unfreeze and the birth-ID comparison.
        live_ids = {id(obj) for obj in gc.get_objects()}
        result["missing_birth_id_type_histogram_after_cycle_1"] = dict(
            sorted(Counter(name for ident, name in birth.items() if ident not in live_ids).items())
        )
        result["after_freeze_unfreeze_pair_count"] = gc.get_freeze_count()
        result["id_reuse_note"] = "ID reuse can under-count; census-only temporaries are included."
    finally:
        gc.freeze = original_freeze
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("default", "freeze_startup"))
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.diagnose and args.policy is None:
        parser.error("--policy required unless --diagnose")
    result = diagnose() if args.diagnose else run(args.policy, args.cycles, args.ticks)
    write_result(args.out, result)
    print(json.dumps({"status": result.get("status", "DIAGNOSTIC"), "out": str(args.out)}))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
