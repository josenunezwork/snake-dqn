"""Opt-in native v8 opponent veto for T34 ExIt shadow/branch rows (speed program S2).

Scope (JUDGMENT 2026-10-20 §3 S2, §5 D3/D5)
-------------------------------------------
Frozen v8 opponents (``Vector61SimdPolicy`` with ``veto_variant="v8"``) decide through
``Vector61SimdPolicy._apply_veto_live_hook``: one live Python
:class:`~src.evaluation.safety_veto_v8.SpaceAndHeadVeto` ``apply`` per opponent row,
including ``SimRowSnake`` roster construction and several Python flood fills.
:func:`src.simd_env.v8_nb.v8_decide` is the compiled (``nogil``) form, decision-identical
per ``tests/test_v8_nb.py``.

The live hook's per-env state is *bookkeeping only* (probe counters, v5/v7/v8
diagnostics); it never feeds a decision.  But T34 does consume it: ``research.t34_shadow.
shadow.full_digest`` hashes every opponent policy's ``live_vetoes`` hooks (minus the
wall-clock ``*_seconds*`` fields) and ``last_veto``, and the root loop's
``shadow.confirm`` compares the root world's digest with the shadow frame's digest.  So a
native path must reproduce the counters exactly, not just the actions.

What this module does
---------------------
:func:`native_live_hook` replaces only the per-row hook loop of
``_apply_veto_live_hook``.  The stock caller has already computed the decision-time
``counts``/``cap``/``need``/``spacious``, the masked-Q rows and the base argmax; we

1. run ``v8_decide`` (lazy, ``features=False``) on the rows and require its base, counts,
   cap and need to equal the stock values;
2. *replay* each row's live counter updates in Python from the kernel's evaluated tables
   (landing counts, tail-aware areas and their cap, head-risk flags) using the live pure
   rule functions themselves (``boost_aware_choice``, ``veto_choice``,
   ``rerank_candidates``, ``space_preference_choice``, ``head_replacement_choice``,
   ``area_score``), and require the replayed action to equal the kernel action;
3. commit the counter deltas onto the env's live hook object (created exactly as the
   stock loop creates it) and set ``last_veto`` exactly as the stock method does.

The one diagnostic the kernel does not evaluate -- v7's *static* (no tail release) area,
used only by ``rerank_changes_tail_release_driven`` on rows where the v7 re-rank changes
the action -- is computed by the live hook's own method on that row's ``SimRowSnake``
roster.  Any row whose replay would need a table entry the lazy kernel did not evaluate
falls back to the stock ``hook.apply`` for that row (exact by construction; counted).

Routing (opt-in; the default path is unchanged)
-----------------------------------------------
The class method is wrapped once (:func:`enable`), but the wrapper runs the native path
only inside :func:`native_scope`; everywhere else it calls the original method.
:class:`NativeV8ExitTeacher` enters the scope for every row except the scored root
worlds: root rows keep the live Python hook, so every ``shadow.confirm`` compares a
Python-hook root digest against a native-hook shadow digest -- a built-in, per-decision
identity check of actions *and* counters in production (a mismatch would surface as a
``divergence`` and change the exports).  No ratified file is edited.

Fail-closed: any kernel/stock disagreement raises :class:`NativeV8Mismatch`; ``audit``
additionally runs the stock hook on a deep copy of every native row's hook and compares
the action and every non-timing counter.
"""

from __future__ import annotations

import contextlib
import copy
import threading
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig

NATIVE_V8_RECIPE = "t34-native-v8-shadow-branch/v2"
NUM_DIRECTIONS = 3
NUM_ACTIONS = 6

_LOCAL = threading.local()
_ORIGINAL: Any = None
_LOCK = threading.Lock()


class NativeV8Mismatch(RuntimeError):
    """The compiled v8 path disagreed with the stock path."""


class _Missing(Exception):
    """The replay needs a table entry the lazy kernel did not evaluate."""


@dataclass
class NativeV8Stats:
    """Coverage and differential counters (process-local, per teacher)."""

    calls: int = 0
    rows: int = 0
    replayed_rows: int = 0
    fallback_rows: int = 0
    tail_release_rows: int = 0
    python_calls: int = 0
    audited_rows: int = 0
    overrides: int = 0
    v5_changes: int = 0
    v7_changes: int = 0
    head_changes: int = 0
    no_spacious: int = 0
    tie_rows: int = 0
    tie_overrides: int = 0
    boost_base_rows: int = 0
    long_rows: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


# ----------------------------------------------------------------------------- scope
def _depth() -> int:
    return int(getattr(_LOCAL, "depth", 0))


@contextlib.contextmanager
def native_scope(stats: NativeV8Stats | None = None, *, audit: bool = False) -> Iterator[None]:
    """Run v8 live-hook rows natively inside this block (this thread only)."""
    enable()
    previous = (getattr(_LOCAL, "stats", None), getattr(_LOCAL, "audit", False))
    _LOCAL.depth = _depth() + 1
    _LOCAL.stats, _LOCAL.audit = stats, bool(audit)
    try:
        yield
    finally:
        _LOCAL.depth = _depth() - 1
        _LOCAL.stats, _LOCAL.audit = previous


@contextlib.contextmanager
def python_scope() -> Iterator[None]:
    """Force the stock hook inside a native scope (used for root rows)."""
    saved = _depth()
    _LOCAL.depth = 0
    try:
        yield
    finally:
        _LOCAL.depth = saved


def enable() -> None:
    """Wrap ``Vector61SimdPolicy._apply_veto_live_hook`` once (inert outside a scope)."""
    global _ORIGINAL
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    with _LOCK:
        if _ORIGINAL is not None:
            return
        _ORIGINAL = Vector61SimdPolicy.__dict__["_apply_veto_live_hook"]

        def _apply_veto_live_hook(self, sim, rows, pick, counts, cap, need, spacious,
                                  q_rows, mask_rows, actions):  # fmt: skip
            if _depth() > 0 and is_native_v8_candidate(self):
                return native_live_hook(
                    self, sim, rows, pick, counts, cap, need, spacious, q_rows, mask_rows,
                    actions, stats=getattr(_LOCAL, "stats", None),
                    audit=bool(getattr(_LOCAL, "audit", False)),
                )  # fmt: skip
            stats = getattr(_LOCAL, "stats", None)
            if stats is not None and _depth() > 0:
                stats.python_calls += 1
            return _ORIGINAL(
                self, sim, rows, pick, counts, cap, need, spacious, q_rows, mask_rows, actions
            )

        _apply_veto_live_hook.__wrapped__ = _ORIGINAL  # type: ignore[attr-defined]
        Vector61SimdPolicy._apply_veto_live_hook = _apply_veto_live_hook


def disable() -> None:
    """Restore the original class method (tests)."""
    global _ORIGINAL
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    with _LOCK:
        if _ORIGINAL is None:
            return
        Vector61SimdPolicy._apply_veto_live_hook = _ORIGINAL
        _ORIGINAL = None


def is_native_v8_candidate(policy: Any) -> bool:
    """A vector61 v8 policy whose live hook configuration the replay reproduces."""
    spec = getattr(policy, "veto_spec", None)
    return (
        getattr(policy, "veto_variant", None) == "v8"
        and spec is not None
        and getattr(spec, "lam", None) is not None
        and getattr(spec, "reference_lambda", None) is None
        and hasattr(policy, "runtime")
    )


# ----------------------------------------------------------------------------- replay
class _Plan:
    """Counter deltas for one row (committed only when the whole row replayed)."""

    __slots__ = ("probe", "v7_probe", "v5", "v7", "v8", "action", "tail")

    def __init__(self) -> None:
        self.probe: tuple[int, int, str] | None = None  # hook.counters.record args
        self.v7_probe: tuple[int, int, str] | None = None  # hook.v7.counters.record args
        self.v5: tuple | None = None  # hook.v7.v7.v5.record args (seconds = 0)
        self.v7: dict[str, int] = {}
        self.v8: dict[str, int] = {}
        self.action = -1
        self.tail: tuple | None = None  # (q, candidates, anchor, action) for static areas


def _bump(table: dict[str, int], name: str, value: int = 1) -> None:
    table[name] = table.get(name, 0) + int(value)


def _replay_row(
    q: list[float],
    mask: list[bool],
    spacious: list[bool],
    base: int,
    need: int,
    lam: float,
    landing: np.ndarray,
    area: np.ndarray,
    area_cap: int,
    risky: list[bool],
    waived: list[bool],
) -> _Plan:
    """``SpaceAndHeadVeto.apply`` bookkeeping for one row, from kernel tables."""
    from src.evaluation.safety_veto import (
        OUTCOME_KEPT,
        OUTCOME_NO_SPACIOUS,
        OUTCOME_VETOED,
        veto_choice,
    )
    from src.evaluation.safety_veto_v5 import boost_aware_choice
    from src.evaluation.safety_veto_v6 import REASON_HEAD_VETO
    from src.evaluation.safety_veto_v7 import area_score, rerank_candidates, space_preference_choice
    from src.evaluation.safety_veto_v8 import head_replacement_choice

    plan = _Plan()
    v7: dict[str, int] = plan.v7
    v8: dict[str, int] = plan.v8

    def landing_value(direction: int) -> int | None:
        value = int(landing[direction])
        if value == -1:
            raise _Missing
        return None if value == -2 else value

    # ---- SpacePreferenceVeto.decide (v7, with nested v5) ----
    memo: dict[int, bool] = {}
    tally = {"checks": 0, "failures": 0, "extra": 0, "in_v5": 1}

    def landing_ok(direction: int) -> bool:
        if direction not in memo:
            count = landing_value(direction)
            memo[direction] = count is None or count >= need
            if tally["in_v5"]:
                tally["checks"] += int(count is not None)
                tally["failures"] += int(not memo[direction])
            else:
                tally["extra"] += int(count is not None)
        return memo[direction]

    v5_action, v5_outcome, v5_reason = boost_aware_choice(q, mask, spacious, base, landing_ok)
    v2_action, _ = veto_choice(q, mask, spacious, base)
    plan.v5 = (
        base, v5_outcome, v5_reason, v5_action != v2_action, tally["checks"], tally["failures"]
    )  # fmt: skip
    tally["in_v5"] = 0
    v7_action, v7_outcome = v5_action, v5_outcome
    rerank_changed = False
    if not (v5_outcome == OUTCOME_NO_SPACIOUS or lam == 0.0):
        candidates = rerank_candidates(v5_action, v5_reason, mask, spacious, landing_ok)
        _bump(v7, "extra_landing_checks", tally["extra"])
        if len(candidates) >= 2:
            cap_max = 0

            def v7_score(action: int) -> float:
                nonlocal cap_max
                value = int(area[action])
                if value < 0:
                    raise _Missing
                _bump(v7, "area_evaluations")
                _bump(v7, "area_cap_hits", int(value >= area_cap))
                cap_max = max(cap_max, int(area_cap))
                return area_score(value, area_cap)

            _bump(v7, "rerank_decisions")
            chosen, scored = space_preference_choice(q, candidates, v5_action, lam, v7_score)
            _bump(v7, "rerank_scored" if scored else "rerank_pruned")
            if cap_max:
                v7["area_cap_max"] = cap_max  # max-merged on commit
            if chosen != v5_action:
                plan.tail = (list(candidates), int(v5_action), int(chosen))
                rerank_changed = True
                _bump(v7, "rerank_changes")
                _bump(v7, "rerank_changes_of_v5_kept", int(v5_outcome == OUTCOME_KEPT))
                _bump(v7, "rerank_changes_boost_mode", int(chosen >= NUM_DIRECTIONS))
                v7_action, v7_outcome = int(chosen), OUTCOME_VETOED
    plan.v7_probe = (base, v7_action, v7_outcome)

    # ---- SpaceAndHeadVeto.decide part 2: the head layer ----
    action, outcome = v7_action, v7_outcome
    if v7_outcome != OUTCOME_NO_SPACIOUS:
        current = v7_action
        _bump(v8, "head_checks")
        if waived[current]:
            _bump(v8, "head_risk_waived_hero_wins")
        if risky[current]:
            _bump(v8, "head_risky_decisions")
            eligible = [
                bool(mask[a])
                and bool(spacious[a % NUM_DIRECTIONS])
                and not risky[a]
                and (
                    a < NUM_DIRECTIONS
                    or (lambda c: c is None or c >= need)(landing_value(a % NUM_DIRECTIONS))
                )
                for a in range(NUM_ACTIONS)
            ]

            def head_score(candidate: int) -> float:
                value = int(area[candidate])
                if value < 0:
                    raise _Missing
                _bump(v8, "head_area_evaluations")
                return area_score(value, area_cap)

            chosen, reason, anchor = head_replacement_choice(
                q, eligible, risky, current, lam, head_score
            )
            if reason != REASON_HEAD_VETO:
                _bump(v8, "head_risky_kept_no_alternative")
            else:
                _bump(v8, "head_risky_vetoes")
                _bump(
                    v8,
                    "head_vetoes_speed_switched",
                    int((chosen >= NUM_DIRECTIONS) != (current >= NUM_DIRECTIONS)),
                )
                _bump(v8, "head_vetoes_space_differs_from_highest_q", int(chosen != anchor))
                action = int(chosen)
        if action != v7_action:
            outcome = OUTCOME_VETOED
            _bump(v8, "head_vetoes_of_v7_kept", int(v7_outcome == OUTCOME_KEPT))
            _bump(v8, "head_vetoes_of_v7_rerank", int(rerank_changed))
    _bump(v8, "action_differs_from_v7", int(action != v7_action))
    plan.probe = (base, int(action), outcome)
    plan.action = int(action)
    return plan


def _commit(hook: Any, plan: _Plan) -> None:
    """Apply one replayed row's deltas exactly as the live calls would."""
    base, action, outcome = plan.probe  # type: ignore[misc]
    v7_base, v7_action, v7_outcome = plan.v7_probe  # type: ignore[misc]
    v7_hook = hook.v7
    v7_hook.v7.v5.record(*plan.v5, 0.0)  # type: ignore[misc]
    for name, value in plan.v7.items():
        if name == "area_cap_max":
            v7_hook.v7.area_cap_max = max(v7_hook.v7.area_cap_max, int(value))
        else:
            setattr(v7_hook.v7, name, getattr(v7_hook.v7, name) + int(value))
    v7_hook.counters.record(v7_base, v7_action, v7_outcome)
    v7_hook.v7.count_outcome(v7_outcome)
    for name, value in plan.v8.items():
        setattr(hook.v8, name, getattr(hook.v8, name) + int(value))
    hook.counters.record(base, action, outcome)
    hook.v8.count_outcome(outcome)


def _hook_state(hook: Any) -> Any:
    """Every non-timing field of a live hook (recursive), for audits."""
    if isinstance(hook, (int, float, str, bool, type(None))):
        return hook
    if isinstance(hook, (list, tuple)):
        return [_hook_state(v) for v in hook]
    if isinstance(hook, dict):
        return {k: _hook_state(v) for k, v in hook.items()}
    if hasattr(hook, "__dict__"):
        return {
            k: _hook_state(v)
            for k, v in sorted(vars(hook).items())
            if not k.endswith(("_seconds", "_seconds_total", "_seconds_max"))
        }
    return repr(hook)


def _tie(q: np.ndarray, mask: np.ndarray) -> bool:
    legal = np.asarray(q, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    return len(legal) != len(np.unique(legal))


def native_live_hook(
    policy: Any,
    sim: Any,
    rows: np.ndarray,
    pick: np.ndarray,
    counts: np.ndarray,
    cap: np.ndarray,
    need: np.ndarray,
    spacious: np.ndarray,
    q_rows: np.ndarray,
    mask_rows: np.ndarray,
    actions: np.ndarray,
    *,
    stats: NativeV8Stats | None = None,
    audit: bool = False,
) -> np.ndarray:
    """Drop-in for ``Vector61SimdPolicy._apply_veto_live_hook`` (v8 policies only)."""
    from src.simd_env.v8_nb import v8_decide
    from src.simd_env.vector61_policy import (
        check_live_rule_config,
        live_veto_hook,
        sim_row_roster,
    )

    if getattr(policy, "_rule_config_checked", None) is not sim:
        check_live_rule_config(sim)  # the stock once-per-sim guard, kept
        policy._rule_config_checked = sim
    grid = (int(policy.runtime.featurizer.gw), int(policy.runtime.featurizer.gh))
    base = actions[pick].copy()
    final = base.copy()
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    res = v8_decide(
        sim,
        rows,
        np.asarray(q_rows, dtype=np.float32),
        np.asarray(mask_rows, dtype=bool),
        float(policy.veto_spec.lam),
        grid_shape=grid,
    )
    for name, stock in (("base", base), ("counts", counts), ("cap", cap), ("need", need)):
        if not np.array_equal(np.asarray(getattr(res, name), dtype=np.int64),
                              np.asarray(stock, dtype=np.int64)):  # fmt: skip
            raise NativeV8Mismatch(f"v8_nb {name} differs from the stock path on {rows.tolist()}")
    lam = float(policy.veto_spec.lam)
    for i, (env, slot) in enumerate(rows.tolist()):
        hook = policy.live_vetoes.get(int(env))
        if hook is None:
            hook = policy.live_vetoes[int(env)] = live_veto_hook(policy.veto_spec)
        if not (
            getattr(hook, "head_avoidance", False)
            and getattr(hook, "_reference", None) is None
            and getattr(hook, "reference_lambda", None) is None
            and float(hook.lam) == lam
        ):
            raise NativeV8Mismatch("live hook configuration differs from the native recipe")
        q_list = np.asarray(q_rows[i]).tolist()
        mask_list = [bool(x) for x in np.asarray(mask_rows[i]).tolist()]
        before = copy.deepcopy(hook) if audit else None
        try:
            plan = _replay_row(
                q_list,
                mask_list,
                [bool(x) for x in np.asarray(spacious[i]).tolist()],
                int(base[i]),
                int(need[i]),
                lam,
                res.landing[i],
                res.area[i],
                int(res.area_cap[i]),
                [bool(x) for x in res.head_risky[i].tolist()],
                [bool(x) for x in res.head_waived[i].tolist()],
            )
        except _Missing:
            plan = None
        if plan is None:
            hero, roster = sim_row_roster(sim, env, slot, grid, counts[i], int(cap[i]))
            final[i] = int(hook.apply(hero, roster, q_rows[i], mask_rows[i], int(base[i])))
            if stats is not None:
                stats.fallback_rows += 1
        else:
            if plan.action != int(res.action[i]):
                raise NativeV8Mismatch(
                    f"replayed v8 action {plan.action} != kernel {int(res.action[i])} "
                    f"at row {(env, slot)}"
                )
            _commit(hook, plan)
            if plan.tail is not None:
                candidates, anchor, chosen = plan.tail
                hero, roster = sim_row_roster(sim, env, slot, grid, counts[i], int(cap[i]))
                _tail_release(hook, hero, roster, q_list, candidates, anchor, chosen)
                if stats is not None:
                    stats.tail_release_rows += 1
            final[i] = plan.action
            if stats is not None:
                stats.replayed_rows += 1
        if before is not None:
            hero, roster = sim_row_roster(sim, env, slot, grid, counts[i], int(cap[i]))
            expected = int(before.apply(hero, roster, q_rows[i], mask_rows[i], int(base[i])))
            if expected != int(final[i]) or _hook_state(before) != _hook_state(hook):
                raise NativeV8Mismatch(
                    f"native v8 differs from the live hook at row {(env, slot)}: action "
                    f"{int(final[i])} vs {expected}"
                )
            if stats is not None:
                stats.audited_rows += 1
    if stats is not None:
        stats.calls += 1
        stats.rows += len(rows)
        over = final != base
        stats.overrides += int(over.sum())
        stats.v5_changes += int((res.v5_action != res.base).sum())
        stats.v7_changes += int((res.v7_action != res.v5_action).sum())
        stats.head_changes += int((res.action != res.v7_action).sum())
        stats.no_spacious += int((res.outcome == 2).sum())
        ties = np.array([_tie(q_rows[i], mask_rows[i]) for i in range(len(rows))], dtype=bool)
        stats.tie_rows += int(ties.sum())
        stats.tie_overrides += int((ties & over).sum())
        stats.boost_base_rows += int((base >= NUM_DIRECTIONS).sum())
        stats.long_rows += int((sim.length[rows[:, 0], rows[:, 1]] >= 800).sum())
    policy.last_veto = {
        "rows": rows.copy(),
        "frame": sim.frame[rows[:, 0]].copy(),
        "counts": counts,
        "need": need,
        "spacious": spacious,
        "base": base,
        "final": final,
    }
    out = actions.copy()
    out[pick] = final
    return out


def _tail_release(
    hook: Any,
    hero: Any,
    roster: list[Any],
    q: list[float],
    candidates: list[int],
    anchor: int,
    action: int,
) -> None:
    """v7's static-area diagnostic, by the live hook's own method on the row's roster."""
    from src.evaluation.safety_veto_v3 import grid_for, static_blocked
    from src.evaluation.safety_veto_v7 import area_cap

    others = list(roster)
    grid = grid_for(hero)
    blocked = static_blocked(hero, others, grid)
    context = {"grid": grid, "blocked": blocked, "cap": area_cap(hero.length, grid, blocked)}
    hook.v7._count_tail_release_driven(hero, q, candidates, anchor, action, context)


# ----------------------------------------------------------------------------- teacher
class NativeV8ExitTeacher(ExitTeacher):
    """:class:`ExitTeacher` whose non-root v8 opponent rows run on ``v8_nb``.

    Root worlds (registered with :meth:`register_root`) keep the stock Python hook; every
    other world stepped through :meth:`actions` (shadow trajectories, doom-cache
    reconstruction clones) and every merged branch row (:meth:`_dispatch_merged`) runs
    :func:`native_live_hook`.
    """

    def __init__(
        self,
        config: ExitTeacherConfig = ExitTeacherConfig(),
        *,
        audit: bool = False,
        branch_only: bool = False,
        include_roots: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        if branch_only and include_roots:
            raise ValueError("branch_only and include_roots are exclusive")
        self.include_roots = bool(include_roots)
        self.native_stats = NativeV8Stats()
        self.native_audit = bool(audit)
        self.branch_only = bool(branch_only)
        self._root_ids: set[int] = set()
        self._roots: list[Any] = []  # strong refs: ids stay unique while registered
        enable()

    def register_root(self, world: Any) -> Any:
        self._root_ids.add(id(world))
        self._roots.append(world)
        return world

    def release_roots(self) -> None:
        self._root_ids.clear()
        self._roots.clear()

    def actions(self, world: Any, hero_action: int) -> np.ndarray:
        if self.branch_only or (id(world) in self._root_ids and not self.include_roots):
            with python_scope():
                return super().actions(world, hero_action)
        with native_scope(self.native_stats, audit=self.native_audit):
            return super().actions(world, hero_action)

    def _dispatch_merged(self, merged: Any, grid: np.ndarray, active: np.ndarray) -> None:
        with native_scope(self.native_stats, audit=self.native_audit):
            super()._dispatch_merged(merged, grid, active)


def make_native_teacher(
    config: Mapping[str, Any],
    *,
    audit: bool = False,
    branch_only: bool = False,
    include_roots: bool = False,
) -> NativeV8ExitTeacher:
    """The teacher ``teacher_v2.run_worlds`` would build, with native non-root v8."""
    return NativeV8ExitTeacher(
        ExitTeacherConfig(
            horizon=int(config.get("horizon", 300)), beta=float(config.get("beta", 1.0))
        ),
        device=str(config.get("device", "cpu")),
        policy=config.get("policy"),
        audit=audit,
        branch_only=branch_only,
        include_roots=include_roots,
    )


def native_config(config: Mapping[str, Any], teacher: NativeV8ExitTeacher) -> dict[str, Any]:
    """``config`` with ``teacher`` and a world builder that registers each root world."""
    from research.t34_exit_b8.trajectory import build_world

    if str(config.get("device", "cpu")) != "cpu":
        raise ValueError("native v8 S2 path is validated on CPU only")
    builder = config.get("world_builder", build_world)

    def build(row: Mapping[str, Any], checkpoint: Any) -> Any:
        return teacher.register_root(builder(row, checkpoint))

    return {**dict(config), "teacher": teacher, "world_builder": build}


def run_worlds_native(
    config: Mapping[str, Any],
    *,
    audit: bool = False,
    branch_only: bool = False,
    include_roots: bool = False,
) -> dict[str, Any]:
    """``teacher_v2.run_worlds`` with the opt-in native teacher.

    The world exports are written by the unchanged collector; the returned receipt gains
    ``native_v8`` (recipe, scope, coverage counters).
    """
    from research.t34_exit_b8.teacher_v2 import run_worlds

    if config.get("teacher") is not None:
        raise ValueError("run_worlds_native builds its own teacher")
    teacher = make_native_teacher(
        config, audit=audit, branch_only=branch_only, include_roots=include_roots
    )
    try:
        receipt = run_worlds(native_config(config, teacher))
    finally:
        teacher.release_roots()
    receipt["native_v8"] = {
        "recipe": NATIVE_V8_RECIPE,
        "scope": (
            "branch rows only"
            if branch_only
            else "all rows incl. roots" if include_roots else "all non-root rows"
        ),
        "audit": bool(audit),
        "stats": teacher.native_stats.as_dict(),
    }
    return receipt
