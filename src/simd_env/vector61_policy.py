"""Opt-in batched ``vector61`` (Apex DQN) policy for the SIMD evaluation engine.

This serves a 61-D (or 58-D) Apex checkpoint inside
:func:`src.simd_env.eval_engine.run_simd_eval` with the same decision semantics
as the live :class:`src.game.ai_snake.AISnake` under the profiled Watch loop of
``tournament_eval.rollout``. Nothing here runs unless a caller passes
``vector61=True`` to ``run_simd_eval``; without it a vector checkpoint still
raises exactly as before.

Live call sequence reproduced (see ``docs/research/simd_vector61_plan_2026-09-26.md``)
------------------------------------------------------------------------------------
* **Capture.** At the end of frame *t* every surviving AISnake builds
  ``get_state(update_enemy_memory=True)`` and its exact (no-fallback) safe list
  on the post-collision world (``compute_reward_and_train``) and carries both
  to frame *t+1*. :meth:`Vector61Runtime.observe_step` featurizes the same rows
  right after ``BatchSim.step`` (before the next food maintenance) and stores the
  post-step advisory mask, after resetting the trend memory of rows that died.
* **Selection.** At *t+1* the carried state is reused even though food
  maintenance has since run. It is rebuilt (advancing the trend memory a second
  time) only on the snake's first frame or when any snake of that world
  respawned this frame (``GameState.update`` invalidates every cache).
  :meth:`Vector61Runtime.prepare` applies exactly that rule per env.
* **Action.** The empty exact mask falls back to all normal actions, the
  Q-values come from a batch-1 forward of the checkpoint network built with the
  live loader, and the action is the first argmax of
  ``where(mask, q, INVALID_Q_VALUE)``. The resolved mask ``run_simd_eval``
  passes to policies is NOT used: live vector snakes act on the advisory
  (fatality) mask with the legacy normal-only fallback.
* **Veto (opt-in).** :func:`batched_veto_choice` is
  :func:`src.evaluation.safety_veto.veto_choice` on arrays, with
  :func:`veto_threshold` giving the live ``(cap, need)``. Like the live
  ``FreeSpaceVeto``, the flood counts are recomputed on the decision-time world
  (:meth:`Vector61Runtime.decision_free_space`), not read from the possibly
  carried state, and recovered with the live ``round(feature * cap)``. Only the
  configured slots (the hero, slot 0) are vetoed; counters are per env.
* **v5 veto (opt-in, ``veto_variant="v5"``).** The boost-aware veto of
  :mod:`src.evaluation.safety_veto_v5`, per hero row on the same decision-time
  ``(cap, need, spacious)``: :func:`boost_landing_count` replays the boost's
  two-cell move with the live pure ``simulate_move`` on the row's BatchSim body
  and floods the landing cell (post-move body except the new head plus every
  other live snake blocked, v2's cap), and the choice is the live
  ``boost_aware_choice`` with the same lazy landing callback. The probe is
  v2-shaped with the v5 descriptor; the v5 counters are
  :meth:`Vector61SimdPolicy.veto_diagnostics`.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.simd_env.batch_sim import CARDINAL, BatchSim
from src.simd_env.vector61_featurizer import (
    FREE_SPACE_BFS_CAP,
    FREE_SPACE_MIN_CAP,
    Vector61Featurizer,
    Vector61Memory,
    Vector61Params,
    capped_component_sizes,
)

__all__ = [
    "FORWARD_MODES",
    "SAFETY_VETO_VARIANTS",
    "Vector61Runtime",
    "Vector61SimdPolicy",
    "batched_veto_choice",
    "boost_landing_count",
    "checkpoint_is_vector61",
    "free_space_counts",
    "resolve_safety_veto",
    "spacious_from_counts",
    "vector61_provenance",
    "veto_threshold",
]

NUM_ACTIONS = 6
NUM_DIRECTIONS = 3
# Outcome codes for batched_veto_choice (labels match safety_veto.OUTCOME_*).
VETO_KEPT = 0
VETO_VETOED = 1
VETO_NO_SPACIOUS = 2
_OUTCOME_LABELS = ("kept", "vetoed", "no_spacious")
# "rowwise": one batch-1 forward per row (the live call shape; bit-exact).
# "batched": one forward over all rows (faster; BLAS may change the last ulp).
FORWARD_MODES = ("rowwise", "batched")
# Hero veto variants: "v2" = safety_veto.FreeSpaceVeto (``hero_safety_veto=True``),
# "v5" = safety_veto_v5.BoostAwareFreeSpaceVeto (opt-in, ``hero_safety_veto="v5"``).
SAFETY_VETO_VARIANTS = ("v2", "v5")
# Relative action direction (0 left, 1 straight, 2 right) -> CARDINAL index delta.
_TURN_DELTA = (-1, 0, 1)


def resolve_safety_veto(value: object) -> Optional[str]:
    """Normalize a ``hero_safety_veto`` argument to ``None`` (off), ``"v2"`` or ``"v5"``.

    ``False``/``None`` is off and ``True`` is v2 (the meaning it always had); the
    strings ``"v2"`` and ``"v5"`` name a variant. Anything else raises.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return "v2" if value else None
    if isinstance(value, str) and value in SAFETY_VETO_VARIANTS:
        return value
    raise ValueError(
        f"hero_safety_veto must be a bool or one of {SAFETY_VETO_VARIANTS}, got {value!r}"
    )


def vector61_provenance(forward: str, hero_safety_veto: object) -> Dict[str, object]:
    """SIMD-only record provenance for a ``run_simd_eval(vector61=True)`` run.

    Live records never carry this key, so a saved record shows that the SIMD
    vector61 policy produced it and whether its forwards were the bit-exact
    ``"rowwise"`` call shape (``"batched"`` is never gate evidence). With the v5
    veto it also names the veto method (the v2 and no-veto dicts are unchanged).
    """
    if forward not in FORWARD_MODES:
        raise ValueError(f"forward must be one of {FORWARD_MODES}, got {forward!r}")
    variant = resolve_safety_veto(hero_safety_veto)
    out: Dict[str, object] = {
        "engine": "simd",
        "policy": "Vector61SimdPolicy",
        "forward": forward,
        "bit_exact_forward": forward == "rowwise",
        "hero_safety_veto": variant is not None,
    }
    if variant == "v5":
        from src.evaluation.safety_veto_v5 import VETO_METHOD_V5

        out["safety_veto_method"] = VETO_METHOD_V5
    return out


def checkpoint_is_vector61(checkpoint_path: str) -> bool:
    """Whether a checkpoint declares (or defaults to) the ``vector61`` contract."""
    import torch

    from src.model.inference_agent import InferenceAgent
    from src.model.obs_spec import VECTOR61

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return InferenceAgent._detect_obs_spec(blob) == VECTOR61


def batched_veto_choice(
    q_values: np.ndarray,
    action_mask: np.ndarray,
    spacious: np.ndarray,
    base_action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Row-wise :func:`src.evaluation.safety_veto.veto_choice` on arrays.

    Args:
        q_values: ``(N, 6)`` masked Q-values (only eligible entries are read).
        action_mask: ``(N, 6)`` bool hard mask (after the empty-mask fallback).
        spacious: ``(N, 3)`` bool per relative direction.
        base_action: ``(N,)`` masked-argmax actions.

    Returns:
        ``(actions, outcomes)``: ``(N,)`` int actions and ``(N,)`` outcome codes
        (``VETO_KEPT``, ``VETO_VETOED``, ``VETO_NO_SPACIOUS``). Replacement picks the
        highest-Q eligible action in the base action's speed mode, else the other
        mode; ties go to the lowest index (first ``argmax``), as in the scalar rule.
    """
    q = np.asarray(q_values, dtype=np.float64)
    mask = np.asarray(action_mask, dtype=bool)
    spacious = np.asarray(spacious, dtype=bool)
    base = np.asarray(base_action, dtype=np.int64)
    n = len(base)
    eligible = mask & spacious[:, np.arange(NUM_ACTIONS) % NUM_DIRECTIONS]
    any_eligible = eligible.any(axis=1)
    base_eligible = eligible[np.arange(n), base] if n else np.zeros(0, dtype=bool)
    base_boost = base >= NUM_DIRECTIONS
    is_boost = np.arange(NUM_ACTIONS) >= NUM_DIRECTIONS
    same_mode = eligible & (is_boost[None, :] == base_boost[:, None])
    candidates = np.where(same_mode.any(axis=1)[:, None], same_mode, eligible)
    scored = np.where(candidates, q, -np.inf)
    best = np.argmax(scored, axis=1) if n else np.zeros(0, dtype=np.int64)
    actions = np.where(~any_eligible | base_eligible, base, best).astype(np.int64)
    outcomes = np.where(
        ~any_eligible, VETO_NO_SPACIOUS, np.where(base_eligible, VETO_KEPT, VETO_VETOED)
    ).astype(np.int64)
    return actions, outcomes


def veto_threshold(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Array form of ``safety_veto.free_space_threshold``: ``(cap, need)`` per row.

    ``cap = min(160, max(32, 2 * length))`` and ``need = min(max(1, length), cap)``
    (``Snake._logical_length``), exactly as the live veto computes them.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    cap = np.minimum(FREE_SPACE_BFS_CAP, np.maximum(FREE_SPACE_MIN_CAP, lengths * 2))
    need = np.minimum(np.maximum(1, lengths), cap)
    return cap, need


def spacious_from_counts(counts: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Array form of ``safety_veto.spacious_directions`` on recovered counts.

    A direction is spacious when its capped flood count is at least ``need``
    (the live ``round(feature * cap) >= need``; ``count == need`` is spacious).

    Args:
        counts: ``(N, 3)`` integer counts (as from :func:`free_space_counts`).
        need: ``(N,)`` per-row threshold (as from :func:`veto_threshold`).

    Returns:
        ``(N, 3)`` bool.
    """
    counts = np.asarray(counts, dtype=np.int64)
    need = np.asarray(need, dtype=np.int64)
    return counts >= need[:, None]


def free_space_counts(states: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Recover the live capped flood-fill counts from the free-space columns.

    The live veto computes ``round(f * cap)`` on the float64 feature
    ``count / cap``; the state holds that feature cast to float32. Because
    ``cap <= 160`` the float32 rounding error of ``f * cap`` is below 1e-4,
    so ``round`` returns the same integer count either way.
    """
    cap = np.minimum(FREE_SPACE_BFS_CAP, np.maximum(FREE_SPACE_MIN_CAP, lengths * 2))
    feats = np.asarray(states[:, 58:61], dtype=np.float64)
    return np.rint(feats * cap[:, None]).astype(np.int64), cap


def _body_cells(sim: BatchSim, env: int, slot: int) -> np.ndarray:
    """``(n, 2)`` body cells of one snake, head first (``Snake.segments`` order)."""
    k = np.arange(int(sim.seg_count[env, slot]), dtype=np.int64)
    return sim.bodies[env, slot, (int(sim.head_ptr[env, slot]) - k) % sim.cap]


def other_snakes_blocked(sim: BatchSim, env: int, slot: int, grid_shape: Tuple[int, int]):
    """``[x, y]`` bool grid, True where another LIVE snake of ``env`` has a cell.

    The array form of ``safety_veto_v3.static_blocked`` (other live snakes are
    static walls; the row's own snake is excluded).
    """
    gw, gh = grid_shape
    blocked = np.zeros((gw, gh), dtype=bool)
    for other in range(int(sim.S)):
        if other == slot or not sim.alive[env, other]:
            continue
        cells = _body_cells(sim, env, other)
        ok = (cells[:, 0] >= 0) & (cells[:, 0] < gw) & (cells[:, 1] >= 0) & (cells[:, 1] < gh)
        blocked[cells[ok, 0], cells[ok, 1]] = True
    return blocked


def boost_landing_count(
    sim: BatchSim,
    env: int,
    slot: int,
    direction: int,
    cap: int,
    grid_shape: Tuple[int, int],
    others_blocked: Optional[np.ndarray] = None,
) -> Optional[int]:
    """BatchSim form of ``safety_veto_v5.landing_count`` for one ``(env, slot)`` row.

    Replays boost action ``3 + direction`` with the live pure
    ``safety_veto_v5.simulate_move`` on the row's cells (``segment_size=1``, the
    lattice the live pixels sit on), then floods from the landing cell with every
    cell of the post-move body except the new head and every other live snake's
    cell blocked; the count is ``min(cap, component size)`` (0 for a blocked or
    off-grid landing). Returns ``None`` when the boost would not fire
    (``length < min_boost_length``), as the live function does.

    Args:
        sim: The decision-time BatchSim.
        env, slot: The row.
        direction: Relative direction 0 (left), 1 (straight) or 2 (right).
        cap: The row's v2 cap (``veto_threshold``).
        grid_shape: ``(gw, gh)`` of the free-space grid.
        others_blocked: Optional precomputed :func:`other_snakes_blocked` grid.
    """
    from src.evaluation.safety_veto_v5 import simulate_move

    env, slot = int(env), int(slot)
    heading = (int(sim.direction[env, slot]) + _TURN_DELTA[int(direction)]) % 4
    dx, dy = (int(v) for v in CARDINAL[heading])
    move = simulate_move(
        [(int(x), int(y)) for x, y in _body_cells(sim, env, slot).tolist()],
        (dx, dy),
        int(sim.length[env, slot]),
        int(sim.boost_frames[env, slot]),
        True,
        1,
        int(sim.cfg.min_boost_length),
        int(sim.cfg.boost_length_cost_frames),
    )
    if not move.boosted:
        return None
    gw, gh = grid_shape
    if others_blocked is None:
        others_blocked = other_snakes_blocked(sim, env, slot, grid_shape)
    free = ~others_blocked
    for x, y in move.segments[1:]:
        if 0 <= x < gw and 0 <= y < gh:
            free[int(x), int(y)] = False
    return int(capped_component_sizes(free, [move.segments[0]], int(cap))[0])


class Vector61Runtime:
    """Shared per-run featurization state for every vector61 policy of one batch.

    Owns the featurizer, the trend memory and the per-``(env, slot)`` carry
    cache, for all rows driven by any vector61 policy (rows of different
    checkpoints are disjoint, so one memory serves them all). The engine calls
    :meth:`prepare` once per frame inside the Watch selection phase and
    :meth:`observe_step` once after every ``BatchSim.step``.

    Args:
        params: Featurizer params; default ``Vector61Params.from_game_config``.
        trace: Optional diagnostic callable receiving one dict per frame with
            the selection rows, their states/masks and whether each was fresh.
    """

    def __init__(self, params: Optional[Vector61Params] = None, trace=None) -> None:
        self._params = params
        self.trace = trace
        self.featurizer: Optional[Vector61Featurizer] = None
        self.free_space_featurizer: Optional[Vector61Featurizer] = None
        self.memory: Optional[Vector61Memory] = None
        self.controlled: Optional[np.ndarray] = None
        self.fresh_rows = 0
        self.carried_rows = 0

    # ------------------------------------------------------------------
    def bind(self, sim: BatchSim, controlled: np.ndarray) -> None:
        """Attach to a freshly built sim; ``controlled`` is the ``(E, S)`` row mask."""
        from dataclasses import replace

        params = self._params or Vector61Params.from_game_config(sim.cfg)
        self.featurizer = Vector61Featurizer(params)
        self.free_space_featurizer = (
            self.featurizer
            if params.use_free_space
            else Vector61Featurizer(replace(params, use_free_space=True))
        )
        E, S = int(sim.E), int(sim.S)
        controlled = np.asarray(controlled, dtype=bool)
        if controlled.shape != (E, S):
            raise ValueError(f"controlled must have shape {(E, S)}, got {controlled.shape}")
        self.controlled = controlled.copy()
        self.memory = Vector61Memory(E, S)
        width = self.featurizer.input_size
        self._carry_state = np.zeros((E, S, width), dtype=np.float32)
        self._carry_mask = np.zeros((E, S, NUM_ACTIONS), dtype=bool)
        self._carry_valid = np.zeros((E, S), dtype=bool)
        self._alive_post = sim.get_alive()
        self.sel_state = np.zeros((E, S, width), dtype=np.float32)
        self.sel_mask = np.zeros((E, S, NUM_ACTIONS), dtype=bool)
        self.sel_ready = np.zeros((E, S), dtype=bool)
        self.sel_fresh = np.zeros((E, S), dtype=bool)

    def prepare(self, sim: BatchSim) -> None:
        """Build this frame's selection states (Watch phase, after respawns).

        Carried rows reuse last frame's post-step capture. A row is rebuilt
        fresh (``update_enemy_memory=True``) when it has no carry (first frame
        of the snake) or when any slot of its env respawned this frame.
        """
        if self.featurizer is None:
            raise RuntimeError("Vector61Runtime.prepare called before bind")
        alive = sim.get_alive()
        respawned_env = np.any(alive & ~self._alive_post, axis=1)
        ready = self.controlled & alive
        carried = ready & self._carry_valid & ~respawned_env[:, None]
        fresh = ready & ~carried
        self.sel_state[:] = 0.0
        self.sel_mask[:] = False
        self.sel_state[carried] = self._carry_state[carried]
        self.sel_mask[carried] = self._carry_mask[carried]
        rows = np.argwhere(fresh)
        if len(rows):
            states = self.featurizer.featurize(sim, rows, self.memory, update_memory=True)
            self.sel_state[rows[:, 0], rows[:, 1]] = states
            self.sel_mask[fresh] = sim.get_advisory_action_mask()[fresh]
        self._carry_valid[:] = False
        self.sel_ready = ready
        self.sel_fresh = fresh
        self.fresh_rows += int(fresh.sum())
        self.carried_rows += int(carried.sum())
        if self.trace is not None:
            self.trace(
                {
                    "frame": sim.frame.copy(),
                    "ready": ready.copy(),
                    "fresh": fresh.copy(),
                    "state": self.sel_state.copy(),
                    "mask": self.sel_mask.copy(),
                }
            )

    def observe_step(self, sim: BatchSim) -> None:
        """Capture the post-step carry (live ``compute_reward_and_train``)."""
        if self.featurizer is None:
            raise RuntimeError("Vector61Runtime.observe_step called before bind")
        # A dead snake is never featurized before it respawns, and the live
        # snake resets its trend baseline on respawn: reset at death instead.
        self.memory.reset_where(sim.get_done())
        alive = sim.get_alive()
        capture = self.controlled & alive
        rows = np.argwhere(capture)
        if len(rows):
            states = self.featurizer.featurize(sim, rows, self.memory, update_memory=True)
            self._carry_state[rows[:, 0], rows[:, 1]] = states
            self._carry_mask[capture] = sim.get_advisory_action_mask()[capture]
        self._carry_valid = capture
        self._alive_post = alive

    def selection(self, slots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(states, exact_masks)`` prepared for ``(N, 2)`` rows."""
        slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        e, s = slots[:, 0], slots[:, 1]
        if not np.all(self.sel_ready[e, s]):
            raise RuntimeError("vector61 rows were not prepared for this frame")
        return self.sel_state[e, s], self.sel_mask[e, s]

    def decision_free_space(self, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        """``(N, 3)`` capped flood counts on the current (decision-time) world.

        Mirrors the live veto, which recomputes ``_get_free_space_features`` at
        decision time instead of reading the possibly carried state.
        """
        slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        states = self.free_space_featurizer.featurize(sim, slots, None, update_memory=False)
        counts, _ = free_space_counts(states, sim.length[slots[:, 0], slots[:, 1]])
        return counts


class Vector61SimdPolicy:
    """Greedy masked Apex policy over :class:`Vector61Runtime` selection rows.

    Duck-types :class:`src.simd_env.eval_engine.SimdPolicy` (``actions``). The
    network is built with ``tournament_eval.build_policy_from_checkpoint`` (the
    live loader) and put in the same inference state as
    ``configure_eval_game_state`` (epsilon 0, ``training=False``, ``eval()``).

    Args:
        checkpoint_path: A ``vector61`` Apex checkpoint.
        runtime: The run's shared :class:`Vector61Runtime`.
        veto_slots: Arena slots whose decisions get the free-space veto
            (the engine passes ``{0}`` for ``hero_safety_veto``; empty = off).
        forward: ``"rowwise"`` (default, bit-exact live call shape) or
            ``"batched"``.
        veto_variant: ``"v2"`` (default, ``FreeSpaceVeto``) or ``"v5"``
            (``BoostAwareFreeSpaceVeto``) for the ``veto_slots`` rows.
    """

    def __init__(
        self,
        checkpoint_path: str,
        runtime: Vector61Runtime,
        *,
        veto_slots: Sequence[int] = (),
        forward: str = "rowwise",
        veto_variant: str = "v2",
    ) -> None:
        import torch

        from src.scripts.tournament_eval import build_policy_from_checkpoint

        if forward not in FORWARD_MODES:
            raise ValueError(f"forward must be one of {FORWARD_MODES}, got {forward!r}")
        if veto_variant not in SAFETY_VETO_VARIANTS:
            raise ValueError(
                f"veto_variant must be one of {SAFETY_VETO_VARIANTS}, got {veto_variant!r}"
            )
        if not checkpoint_is_vector61(checkpoint_path):
            raise ValueError(f"{checkpoint_path!r} is not a vector61 checkpoint")
        policy = build_policy_from_checkpoint(checkpoint_path)
        if hasattr(policy, "epsilon"):
            policy.epsilon = 0.0
        policy.training = False
        policy.dqn.eval()
        self._torch = torch
        self.checkpoint_path = str(checkpoint_path)
        self.network = policy.dqn
        self.device = next(policy.dqn.parameters()).device
        self.input_size = int(policy.input_size)
        self.runtime = runtime
        self.veto_slots = frozenset(int(s) for s in veto_slots)
        self.forward = forward
        self.veto_variant = veto_variant
        # Per-env veto bookkeeping (src.evaluation.safety_veto.SafetyVetoCounters).
        self.veto_counters: Dict[int, object] = {}
        # Per-env v5 bookkeeping (safety_veto_v5.BoostAwareCounters); v5 only.
        self.boost_counters: Dict[int, object] = {}
        self.last_veto: Optional[Dict[str, np.ndarray]] = None

    def q_values(self, states: np.ndarray):
        """``(N, 6)`` float32 Q tensor for ``(N, D)`` float32 states (no grad)."""
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            if self.forward == "batched":
                return self.network(x)
            rows: List = [self.network(x[i].unsqueeze(0)).squeeze() for i in range(len(x))]
            return torch.stack(rows) if rows else self.network(x)

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        """Masked-greedy (optionally vetoed) actions for prepared ``slots`` rows.

        ``masks`` (the engine's resolved mask) is intentionally ignored: the
        live AISnake acts on its exact advisory mask with the normal-only
        fallback, which :class:`Vector61Runtime` prepares.
        """
        from src.training.action_mask import INVALID_Q_VALUE

        del masks
        torch = self._torch
        slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        states, exact = self.runtime.selection(slots)
        if states.shape[1] != self.input_size:
            raise ValueError(
                f"checkpoint input_size={self.input_size} does not match the active "
                f"featurizer width {states.shape[1]}"
            )
        mask = exact.copy()
        mask[~mask.any(axis=1), :NUM_DIRECTIONS] = True  # live allow_fallback
        q = self.q_values(states)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=q.device)
        masked = torch.where(mask_t, q, torch.full_like(q, INVALID_Q_VALUE))
        actions = masked.argmax(dim=1).cpu().numpy().astype(np.int64)
        if self.veto_slots:
            actions = self._apply_veto(sim, slots, masked, mask, actions)
        return np.clip(actions, 0, NUM_ACTIONS - 1)

    def _apply_veto(
        self,
        sim: BatchSim,
        slots: np.ndarray,
        masked_q,
        mask: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        from src.evaluation.safety_veto import SafetyVetoCounters

        pick = np.flatnonzero(np.isin(slots[:, 1], list(self.veto_slots)))
        if not len(pick):
            return actions
        rows = slots[pick]
        counts = self.runtime.decision_free_space(sim, rows)
        cap, need = veto_threshold(sim.length[rows[:, 0], rows[:, 1]])
        spacious = spacious_from_counts(counts, need)
        q_rows = masked_q.cpu().numpy()[pick]
        if self.veto_variant == "v5":
            return self._apply_veto_v5(
                sim, rows, pick, counts, cap, need, spacious, q_rows, mask[pick], actions
            )
        vetoed, outcomes = batched_veto_choice(q_rows, mask[pick], spacious, actions[pick])
        # Diagnostics for the last call (read by the parity tests).
        self.last_veto = {
            "rows": rows.copy(),
            "frame": sim.frame[rows[:, 0]].copy(),
            "counts": counts,
            "need": need,
            "spacious": spacious,
            "base": actions[pick].copy(),
            "final": vetoed,
        }
        for (env, _slot), base, final, code in zip(rows, actions[pick], vetoed, outcomes):
            counters = self.veto_counters.setdefault(int(env), SafetyVetoCounters())
            counters.record(int(base), int(final), _OUTCOME_LABELS[int(code)])
        out = actions.copy()
        out[pick] = vetoed
        return out

    def _apply_veto_v5(
        self,
        sim: BatchSim,
        rows: np.ndarray,
        pick: np.ndarray,
        counts: np.ndarray,
        cap: np.ndarray,
        need: np.ndarray,
        spacious: np.ndarray,
        q_rows: np.ndarray,
        mask_rows: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """The live ``BoostAwareFreeSpaceVeto.apply`` rule on each picked row.

        Same inputs as the live hook (masked Q, mask, decision-time
        ``cap``/``need``/``spacious``); the live pure ``boost_aware_choice`` makes
        the choice with a lazy, memoized :func:`boost_landing_count` callback, so
        floods run exactly when (and for the directions) the live hook runs them.
        """
        from src.evaluation.safety_veto import SafetyVetoCounters, veto_choice
        from src.evaluation.safety_veto_v5 import BoostAwareCounters, boost_aware_choice

        grid = (int(self.runtime.featurizer.gw), int(self.runtime.featurizer.gh))
        n = len(rows)
        base = actions[pick].copy()
        final = base.copy()
        # -1: not flooded this decision; -2: boost would not fire (live None).
        landing = np.full((n, NUM_DIRECTIONS), -1, dtype=np.int64)
        reasons: List[str] = []
        for i, (env, slot) in enumerate(rows.tolist()):
            started = time.perf_counter()
            q = [float(v) for v in q_rows[i].tolist()]
            m = [bool(v) for v in mask_rows[i].tolist()]
            sp = [bool(v) for v in spacious[i].tolist()]
            b = int(base[i])
            row_cap, row_need = int(cap[i]), int(need[i])
            memo: Dict[int, bool] = {}
            tally = {"checks": 0, "failures": 0}
            others: List[Optional[np.ndarray]] = [None]

            def landing_ok(direction: int, i=i, env=env, slot=slot) -> bool:
                if direction not in memo:
                    if others[0] is None:
                        others[0] = other_snakes_blocked(sim, env, slot, grid)
                    count = boost_landing_count(
                        sim, env, slot, direction, row_cap, grid, others_blocked=others[0]
                    )
                    landing[i, direction] = -2 if count is None else count
                    memo[direction] = count is None or count >= row_need
                    tally["checks"] += int(count is not None)
                    tally["failures"] += int(not memo[direction])
                return memo[direction]

            action, outcome, reason = boost_aware_choice(q, m, sp, b, landing_ok)
            v2_action, _ = veto_choice(q, m, sp, b)
            self.veto_counters.setdefault(int(env), SafetyVetoCounters()).record(b, action, outcome)
            self.boost_counters.setdefault(int(env), BoostAwareCounters()).record(
                b,
                outcome,
                reason,
                action != v2_action,
                tally["checks"],
                tally["failures"],
                time.perf_counter() - started,
            )
            final[i] = action
            reasons.append(reason)
        self.last_veto = {
            "rows": rows.copy(),
            "frame": sim.frame[rows[:, 0]].copy(),
            "counts": counts,
            "need": need,
            "spacious": spacious,
            "base": base,
            "final": final,
            "landing": landing,
            "reason": reasons,
        }
        out = actions.copy()
        out[pick] = final
        return out

    def veto_record(self, env: int) -> Dict[str, object]:
        """Live-shaped ``probes["safety_veto"]`` record for one env."""
        from src.evaluation.safety_veto import FreeSpaceVeto, SafetyVetoCounters

        counters = self.veto_counters.get(int(env), SafetyVetoCounters())
        if self.veto_variant == "v5":
            from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto

            descriptor = BoostAwareFreeSpaceVeto().descriptor()
        else:
            descriptor = FreeSpaceVeto().descriptor()
        return {**descriptor, "counters": counters.to_dict()}

    def veto_diagnostics(self, env: int) -> Dict[str, object]:
        """v5 only: ``BoostAwareCounters.to_dict()`` for one env (the live
        ``diagnostics_record``; ``apply_seconds_*`` time only the per-row rule)."""
        from src.evaluation.safety_veto_v5 import BoostAwareCounters

        if self.veto_variant != "v5":
            raise ValueError("veto diagnostics exist only for the v5 veto")
        return self.boost_counters.get(int(env), BoostAwareCounters()).to_dict()
