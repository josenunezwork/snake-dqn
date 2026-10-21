"""Speed program S2: native v8 for non-root rows must equal the live Python hook.

Row-level differential: every v8 opponent row decided inside ``native_scope(audit=True)``
also runs the stock live hook on a deep copy of that env's hook and must agree on the
action and on every non-timing counter (``NativeV8Mismatch`` otherwise).  Fixtures cover
exact Q ties (integer-floored opponent Q), opponents with bodies >= 800, and opponent
deaths/respawns.  The default (no scope) path must stay the stock method.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT_CANDIDATES = (
    os.environ.get("T34_B8_CHECKPOINT", ""),
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-a5-20261020/inputs/B8_s3_final.pth",
)
CHECKPOINT = next((Path(p) for p in CHECKPOINT_CANDIDATES if p and Path(p).is_file()), None)
needs_checkpoint = pytest.mark.skipif(CHECKPOINT is None, reason="B8_s3_final.pth not present")


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    import torch

    torch.set_num_threads(1)


def test_wrapper_is_inert_outside_scope_and_disable_restores() -> None:
    from research.t34_speed_s2 import native_v8
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    native_v8.disable()
    original = Vector61SimdPolicy.__dict__["_apply_veto_live_hook"]
    assert not hasattr(original, "__wrapped__")
    native_v8.enable()
    try:
        wrapped = Vector61SimdPolicy.__dict__["_apply_veto_live_hook"]
        assert wrapped.__wrapped__ is original
        assert native_v8._depth() == 0
        with native_v8.native_scope():
            assert native_v8._depth() == 1
            with native_v8.python_scope():
                assert native_v8._depth() == 0
            assert native_v8._depth() == 1
        assert native_v8._depth() == 0
    finally:
        native_v8.disable()
    assert Vector61SimdPolicy.__dict__["_apply_veto_live_hook"] is original


def test_candidate_requires_plain_v8_spec() -> None:
    from research.t34_speed_s2.native_v8 import is_native_v8_candidate

    class Spec:
        lam, reference_lambda = 8.0, None

    class Policy:
        veto_variant, veto_spec, runtime = "v8", Spec(), object()

    assert is_native_v8_candidate(Policy())
    Policy.veto_variant = "v7"
    assert not is_native_v8_candidate(Policy())
    Policy.veto_variant = "v8"
    Spec.reference_lambda = 4.0
    assert not is_native_v8_candidate(Policy())


def _world(body: str, mix: str = "frozen", *, ties: bool = False):
    from research.t34_exit_b8 import profile_cpu
    from research.t34_speed_s2 import identity

    row = next(r for r in profile_cpu._rows() if r["mix"] == mix)
    builder = {
        "short": profile_cpu._reset_world_builder,
        "long": profile_cpu._long_world_builder,
        "xlong": identity._xlong_world_builder,
    }[body]
    world = builder(row, CHECKPOINT)
    if ties:
        for policy in world.opponents:
            if getattr(policy, "veto_slots", frozenset()):
                identity._floor_q_install(policy)
    return world


def _drive(world, frames: int, *, native: bool, audit: bool = False):
    """Step ``world`` greedily; return (action grids, hook states, stats, events)."""
    from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig
    from research.t34_exit_b8.teacher_v2 import _step
    from research.t34_exit_b8.trajectory import prepare_world
    from research.t34_speed_s2.native_v8 import NativeV8Stats, _hook_state, native_scope

    teacher = ExitTeacher(ExitTeacherConfig(horizon=8), device="cpu")
    stats = NativeV8Stats()
    grids, events = [], {"deaths": 0, "respawns": 0}
    for _ in range(frames):
        if not bool(world.sim.alive[0, 0]):
            break
        before = world.sim.alive[:, 1:].copy()
        prepare_world(world)
        action = teacher.evaluate(world).greedy_action()
        if native:
            with native_scope(stats, audit=audit):
                grids.append(_step(world, teacher, action))
        else:
            grids.append(_step(world, teacher, action))
        after = world.sim.alive[:, 1:]
        events["deaths"] += int((before & ~after).sum())
        events["respawns"] += int((~before & after).sum())
    hooks = [_hook_state(getattr(policy, "live_vetoes", {})) for policy in world.opponents]
    last = [copy.deepcopy(getattr(policy, "last_veto", None)) for policy in world.opponents]
    return grids, hooks, last, stats, events


def _same_last_veto(a, b) -> bool:
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    if sorted(a) != sorted(b):
        return False
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.dtype != y.dtype or not np.array_equal(x, y):
            return False
    return True


@needs_checkpoint
@pytest.mark.parametrize(
    "body,mix,ties,frames",
    [
        ("long", "frozen", False, 40),
        ("xlong", "frozen", False, 40),
        ("xlong", "mixed", True, 40),
        ("short", "frozen", True, 60),
    ],
)
def test_native_rows_equal_live_hook(body: str, mix: str, ties: bool, frames: int) -> None:
    stock_world = _world(body, mix, ties=ties)
    native_world = _world(body, mix, ties=ties)
    stock = _drive(stock_world, frames, native=False)
    native = _drive(native_world, frames, native=True, audit=True)
    assert native[0] == stock[0], "action grids differ"
    assert native[1] == stock[1], "live hook counters differ"
    assert all(_same_last_veto(a, b) for a, b in zip(native[2], stock[2]))
    stats = native[3]
    assert stats.rows > 0 and stats.audited_rows == stats.replayed_rows + stats.fallback_rows
    assert stats.python_calls == 0
    if body == "xlong":
        assert stats.long_rows > 0
    if ties:
        assert stats.tie_rows > 0


@needs_checkpoint
def test_full_digest_equal_between_stock_and_native_steps() -> None:
    from research.t34_shadow.shadow import full_digest

    stock_world = _world("long", "mixed", ties=True)
    native_world = _world("long", "mixed", ties=True)
    for _ in range(3):
        _drive(stock_world, 8, native=False)
        _drive(native_world, 8, native=True)
        assert full_digest(native_world) == full_digest(stock_world)


@needs_checkpoint
@pytest.mark.parametrize("body,mix,ties", [("xlong", "frozen", False), ("xlong", "mixed", True)])
def test_branch_rollouts_equal_stock(body: str, mix: str, ties: bool) -> None:
    """Merged branch rows (all six first actions, compaction, deaths) match the stock engine."""
    from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig
    from research.t34_speed_s2.native_v8 import NativeV8ExitTeacher

    config = ExitTeacherConfig(horizon=60)
    stock = ExitTeacher(config, device="cpu")
    native = NativeV8ExitTeacher(config, device="cpu", audit=True)
    choices = [list(range(6))]
    expected = stock._rollout_merged([_world(body, mix, ties=ties)], choices)
    observed = native._rollout_merged([_world(body, mix, ties=ties)], choices)
    assert observed == expected
    assert stock.profile()["active_row_steps"] == native.profile()["active_row_steps"]
    stats = native.native_stats
    assert stats.rows > 0 and stats.python_calls == 0
    assert stats.audited_rows == stats.rows
    assert stats.long_rows > 0


def test_registered_roots_use_python_and_clones_use_native(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.t34_exit_b8.engine import ExitTeacher
    from research.t34_speed_s2 import native_v8

    monkeypatch.setattr(ExitTeacher, "actions", lambda *args: native_v8._depth())
    teacher = native_v8.NativeV8ExitTeacher()
    root, clone = object(), object()
    teacher.register_root(root)
    assert teacher.actions(root, 0) == 0
    assert teacher.actions(clone, 0) == 1
    assert native_v8._depth() == 0
    teacher.release_roots()
    native_v8.disable()


@needs_checkpoint
def test_missing_lazy_table_falls_back_without_partial_counter_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.t34_shadow.shadow import full_digest
    from research.t34_speed_s2 import native_v8

    stock_world = _world("xlong", "mixed", ties=True)
    native_world = _world("xlong", "mixed", ties=True)
    replay = native_v8._replay_row

    def missing(*args, **kwargs):
        # Build a complete local counter plan, then simulate a later missing table.
        # No staged counter delta may leak into the stock fallback's hook state.
        replay(*args, **kwargs)
        raise native_v8._Missing

    monkeypatch.setattr(native_v8, "_replay_row", missing)
    stock = _drive(stock_world, 12, native=False)
    native = _drive(native_world, 12, native=True, audit=True)
    assert native[0] == stock[0]
    assert native[1] == stock[1]
    assert full_digest(stock_world) == full_digest(native_world)
    assert native[3].rows == native[3].fallback_rows == native[3].audited_rows > 0
    assert native[3].replayed_rows == 0


@needs_checkpoint
def test_audit_rejects_counter_corruption(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.t34_speed_s2 import native_v8

    commit = native_v8._commit

    def corrupt(hook, plan):
        commit(hook, plan)
        hook.v8.head_checks += 1

    monkeypatch.setattr(native_v8, "_commit", corrupt)
    with pytest.raises(native_v8.NativeV8Mismatch, match="differs from the live hook"):
        _drive(_world("xlong", "mixed", ties=True), 1, native=True, audit=True)
