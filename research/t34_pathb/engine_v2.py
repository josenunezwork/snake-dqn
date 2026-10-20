"""Exact, bounded Path B v2 shadow and earliest-rescue search.

The engine deliberately reuses the T3 captured-world seam.  In particular it
does not replace Vector61 opponent dispatch: a native E=K batch retains each
opponent's v8 carry and dispatches v8 through its policy action hook.

This module has no executor and never touches a live game.  The controller
captures a :class:`SearchRequest` on its tick thread and runs ``run_search`` in
its one worker.  That separation makes a request immutable while the shadow
continues to be confirmed or invalidated by later ticks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from research.t34_shadow.shadow import clone_world_exact, full_digest

SearchStatus = Literal["IMPROVING", "NO_IMPROVEMENT", "STALE", "LATE", "INVALID"]
_SCRATCH = threading.local()


def _workspace(envs: int, slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded thread-private native batch scratch, grown but never queued."""
    current = getattr(_SCRATCH, "value", None)
    if current is None or current[0].shape[0] < envs or current[0].shape[1] < slots:
        current = (np.empty((envs, slots), dtype=np.int64), np.empty(envs, dtype=bool))
        _SCRATCH.value = current
    return current[0][:envs, :slots], current[1][:envs]


def _digest(world: Any, checkpoint: Any) -> str:
    """Digest trajectory state without reading or swapping process-global RNG.

    GridBatchSim owns one CPython stream per environment; those streams are
    already included by ``full_digest``.  The process-global stream is neither
    a simulator input nor safe to inspect from the background continuation.
    """
    return full_digest(world, checkpoint=checkpoint, global_rng_state=("t34-v2-isolated",))


@dataclass(frozen=True)
class SearchOutcome:
    """Complete continuation result for one forced first action."""

    action: int
    survives: bool
    alive_steps: int
    mass: float  # Sum of logical terminal-hero mass over alive branch frames.
    horizon: int
    death_frame: int | None = None  # Absolute decision frame, if dead.
    censored: bool = False  # True only when terminal horizon truncated the branch.


@dataclass(frozen=True)
class PreparedDecision:
    """An exact, pre-transition T3 snapshot and its B8 decision metadata."""

    frame: int
    world: Any
    bare_action: int
    q: tuple[float, ...]
    legal: tuple[bool, ...]


@dataclass(frozen=True)
class SearchRequest:
    """Immutable worker input captured from one confirmed shadow."""

    shadow_id: int
    source_digest: str
    target_frame: int
    decision: PreparedDecision
    horizon: int
    replay_root: Any | None = None
    cutoff_frame: int | None = None
    doom_frame: int | None = None
    terminal_horizon: int | None = None
    cancel_event: Any | None = None


@dataclass(frozen=True)
class SearchResult:
    """A complete request outcome; controller validates freshness at consume time."""

    status: SearchStatus
    shadow_id: int
    target_frame: int | None
    action: int | None
    bare_action: int
    outcomes: Mapping[int, SearchOutcome]
    source_digest: str
    completed: bool = True
    intervention_kind: Literal["rescue", "delay", "none"] = "none"


@dataclass(frozen=True)
class ShadowAdvance:
    """Public one-tick shadow state for a nonblocking controller."""

    steps: int
    shadow_id: int | None
    confirmed: bool
    doom_frame: int | None
    first_detection_frame: int | None


@dataclass(frozen=True)
class MaintenanceResult:
    """One worker-owned shadow maintenance result for the controller."""

    source_frame: int
    source_life: int
    source_checkpoint: Any
    source_digest: str
    shadow_id: int | None
    confirmed: bool
    doom_frame: int | None
    first_detection_frame: int | None
    search_request: SearchRequest | None
    expected_digests: tuple[tuple[int, str], ...] = ()


def _thread_cap() -> None:
    """Cap process-local numerical libraries before any branch work starts."""
    if getattr(_SCRATCH, "thread_capped", False):
        return
    # Imports are intentionally local: importing this module must not mutate
    # serving process settings.  Numba's cap applies to its current thread.
    try:
        import numba

        numba.set_num_threads(1)
    except (ImportError, RuntimeError, ValueError):
        pass
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    _SCRATCH.thread_capped = True


def _mass(world: Any) -> float:
    """Return terminal-hero logical mass, with dead frames contributing zero."""
    if not bool(world.sim.alive[0, 0]):
        return 0.0
    return float(world.sim.length[0, 0])


def _prepared_metadata(world: Any) -> tuple[int, tuple[float, ...], tuple[bool, ...]]:
    """Read exact B8 Q and resolved action legality from a prepared selector state."""
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    if not getattr(world.sim, "_policy_phase_prepared", False):
        raise RuntimeError("Path B v2 decision snapshot must be selector-prepared")
    legal = tuple(bool(value) for value in world.sim.get_resolved_action_mask()[0, 0])
    if len(legal) != 6 or not any(legal):
        raise ValueError("prepared action mask is empty or malformed")
    obs = build_ego_raster(
        world.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
    )
    q = tuple(float(value) for value in prepared_q(world, obs))
    if len(q) != 6 or not np.isfinite(q).all():
        raise ValueError("prepared B8 Q is malformed or nonfinite")
    # Ego2s B8 selects on advisory legality and, if it is empty, enables the
    # normal modes irrespective of resolved-mask fatality.  This is distinct
    # from Vector61 opponent dispatch and must remain batch-1 identical.
    advisory = world.sim.get_advisory_action_mask()[0, 0].astype(bool)
    if not advisory.any():
        advisory[:3] = True
    bare = min((action for action in range(6) if advisory[action]), key=lambda a: (-q[a], a))
    return int(bare), q, legal


def _capture_and_step(world: Any) -> PreparedDecision:
    """Clone the exact prepared state inside normal policy dispatch, then step it."""
    from research.t3_rollout_labels_20261015.bank_engine import step_prepared
    from src.simd_env.eval_engine import _dispatch_actions

    # Captured T3 worlds begin selector-prepared.  Later decisions must be
    # observed inside the selector; ``choose_and_step`` is the unmodified exact
    # dispatch path and owns Vector61/v8 prepare/observe carry.
    if getattr(world.sim, "_policy_phase_prepared", False):
        snapshot = clone_world_exact(world)
        bare, q, legal = _prepared_metadata(snapshot)
        frame = int(snapshot.sim.frame[0]) - 1
        # A prepared Vector61 runtime cannot be prepared twice: doing so
        # mutates carry/enemy-memory before the captured decision.
        step_prepared(world, bare)
        return PreparedDecision(frame, snapshot, bare, q, legal)
    captured: dict[str, PreparedDecision] = {}

    def choose(prepared: Any) -> np.ndarray:
        masks = prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        # Clone after the sole prepare and before either hero or opponent
        # dispatch.  Thus branch v8 hooks/counters receive exactly one action.
        snapshot = clone_world_exact(world)
        snapshot.sim._policy_phase_prepared = True
        bare, q, legal = _prepared_metadata(snapshot)
        captured["decision"] = PreparedDecision(
            int(snapshot.sim.frame[0]) - 1, snapshot, bare, q, legal
        )
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return captured["decision"]


def evaluate_prepared(
    decision: PreparedDecision,
    actions: Sequence[int],
    horizon: int,
    *,
    cancel_event: Any | None = None,
) -> dict[int, SearchOutcome]:
    """Evaluate forced actions in lockstep with exact v8/opponent carry.

    A branch has its own simulator and runtime.  Stepping the active branches
    in stable action order is equivalent to a batched row expansion while
    retaining policies whose state is not representable by T33's repeated-env
    fast-anchor path.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    choices = tuple(int(action) for action in actions)
    if not choices or len(set(choices)) != len(choices) or any(a not in range(6) for a in choices):
        raise ValueError("actions must be unique values in [0, 5]")
    if any(not decision.legal[action] and action != decision.bare_action for action in choices):
        raise ValueError("forced action is not resolved-legal")
    # Assemble a real E=K native simulator batch.  T33 deliberately refuses
    # vetoed opponents; this wrapper strips only a private assembly clone then
    # restores a separate carried policy per arena slot below.
    merged = _merge_v8_prepared(decision, choices, horizon)
    if merged is not None:
        return _evaluate_merged_v8(merged, decision, choices, horizon, cancel_event=cancel_event)
    # Amendment 2 has no serial/v1 fallback: a failure to construct the exact
    # native v8 batch is an INVALID request and authorizes bare action only.
    raise RuntimeError("unable to assemble exact native v8 continuation batch")


def _merge_v8_prepared(
    decision: PreparedDecision, actions: tuple[int, ...], horizon: int
) -> Any | None:
    """Use T33's native E=K assembly while preserving v8 policy carry per row."""
    try:
        from research.t33_batched_teacher.teacher import merge_worlds
    except ImportError:
        return None
    # Keep authoritative carried v8 policy state before clearing the disposable
    # assembly clone.  ``clone_world_exact`` deliberately creates independent
    # policy dictionaries, but using its copies as originals would still erase
    # the veto configuration below.
    originals = list(decision.world.opponents)
    source = clone_world_exact(decision.world)
    # T33's rejection protects its teacher from silently dropping veto logic.
    # It is safe to clear it only on this disposable assembly source because we
    # replace all policy groups with v8-aware per-slot clones immediately after.
    for policy in source.opponents:
        if hasattr(policy, "veto_slots"):
            policy.veto_slots = frozenset()
    try:
        legal = np.asarray(decision.legal, dtype=bool).copy()
        legal[decision.bare_action] = True
        merged = merge_worlds([source], 1, [legal], horizon=horizon)
    except (TypeError, ValueError, RuntimeError):
        return None
    import copy

    groups: dict[Any, list[tuple[int, int]]] = {}
    for slot, original in enumerate(originals, start=1):
        # T33 rebuilt fast anchors carry an env-length seed vector.  Retain
        # that instance rather than replacing it with the E=1 source policy.
        if not hasattr(original, "veto_slots"):
            for policy, slots in merged.policy_groups.items():
                if any(int(row[1]) == slot for row in slots):
                    groups.setdefault(policy, []).extend(
                        tuple(map(int, row)) for row in slots if int(row[1]) == slot
                    )
                    break
            continue
        policy = copy.copy(original)
        if hasattr(policy, "__dict__"):
            policy.__dict__ = dict(policy.__dict__)
        if hasattr(policy, "runtime"):
            policy.runtime = merged.runtime
        # v8 state is keyed by env.  Each branch begins from exactly the same
        # prepared source carry, then diverges in the shared merged runtime.
        for name in ("veto_counters", "boost_counters", "live_vetoes"):
            value = getattr(original, name, None)
            if isinstance(value, dict):
                setattr(
                    policy,
                    name,
                    {env: copy.deepcopy(value.get(0)) for env in range(merged.sim.E) if 0 in value},
                )
        rows = [(env, slot) for env in range(merged.sim.E)]
        groups[policy] = rows
        for env in range(merged.sim.E):
            merged.opponents[env][slot - 1] = policy
    merged.policy_groups = {
        policy: np.asarray(rows, dtype=np.int64) for policy, rows in groups.items()
    }
    return merged


def _evaluate_merged_v8(
    merged: Any,
    decision: PreparedDecision,
    actions: tuple[int, ...],
    horizon: int,
    *,
    cancel_event: Any | None = None,
) -> dict[int, SearchOutcome]:
    """Native E=K continuation with v8-aware dispatch and rowwise B8 hero Q."""
    from src.simd_env.ego2s_policy import forward_q
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    sim, runtime = merged.sim, merged.runtime
    grid, active = _workspace(sim.E, sim.S)
    grid.fill(1)
    active[...] = np.asarray(merged.layout.active, dtype=bool)

    def opponents() -> None:
        for policy, slots in merged.policy_groups.items():
            live_slots = slots[active[slots[:, 0]] & sim.alive[slots[:, 0], slots[:, 1]]]
            if len(live_slots):
                masks = sim.get_resolved_action_mask()[live_slots[:, 0], live_slots[:, 1]]
                grid[live_slots[:, 0], live_slots[:, 1]] = policy.actions(masks, sim, live_slots)

    opponents()
    grid[:, 0] = np.asarray(merged.layout.action, dtype=np.int64)
    sim.step(grid, active_env_mask=active)
    runtime.observe_step(sim, rows_mask=active)
    alive_steps = (active & sim.alive[:, 0]).astype(np.int64)
    masses = np.where(active & sim.alive[:, 0], sim.length[:, 0], 0).astype(np.float64)
    active &= sim.alive[:, 0]
    for _ in range(1, horizon):
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("Path B v2 search cancelled")
        if not active.any():
            break

        def choose(prepared: Any) -> np.ndarray:
            runtime.prepare(prepared, rows_mask=active)
            grid.fill(1)
            opponents()
            envs = np.flatnonzero(active)
            rows = np.column_stack((envs, np.zeros(len(envs), dtype=np.int64)))
            obs = build_ego_raster(
                prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=rows
            )
            q = forward_q(merged.sources[0].hero.net, obs, "rowwise").numpy()
            mask = prepared.get_advisory_action_mask()[envs, 0].astype(bool)
            empty = ~mask.any(axis=1)
            # Ego2sSimdPolicy's exact trapped fallback permits normal moves
            # even when the resolved mask calls every action fatal.
            mask[empty, :3] = True
            grid[envs, 0] = np.where(mask, q, -np.inf).argmax(axis=1)
            return grid

        sim.step_with_policy(choose, active_env_mask=active)
        runtime.observe_step(sim, rows_mask=active)
        alive_steps += (active & sim.alive[:, 0]).astype(np.int64)
        masses += np.where(active & sim.alive[:, 0], sim.length[:, 0], 0)
        active &= sim.alive[:, 0]
    result: dict[int, SearchOutcome] = {}
    for env, action in enumerate(merged.layout.action):
        survived = bool(active[env])
        result[int(action)] = SearchOutcome(
            int(action),
            survived,
            int(alive_steps[env]),
            float(masses[env]),
            horizon,
            death_frame=None if survived else int(decision.frame) + int(alive_steps[env]),
            censored=False,
        )
    return result


class PathBSearchEngine:
    """D500 exact shadow with immutable earliest-first rescue requests."""

    def __init__(self, depth: int = 500, horizon: int = 500, *, checkpoint: Any = None) -> None:
        if not 1 <= int(depth) <= 500:
            raise ValueError("shadow depth must be in 1..500")
        if int(horizon) < 1:
            raise ValueError("horizon must be positive")
        self.depth = int(depth)
        self.horizon = int(horizon)
        self.terminal_horizon: int | None = None
        self.checkpoint = checkpoint
        self._shadow: Any | None = None
        # Amendment 2 publishes only the earliest detection.  Retaining a
        # full simulator for every lead step would be unbounded in practice;
        # retain exactly one prepared branch plus compact future digests.
        self._first_decision: PreparedDecision | None = None
        self._expected_digests: list[tuple[int, str]] = []
        self._replay_root: Any | None = None
        self._shadow_id = 0
        self._confirmed = False
        self._doom_frame: int | None = None
        self._first_detection_frame: int | None = None
        self._source_digest = ""
        self._confirmed_digest = ""
        self._cancel = threading.Event()

    @property
    def shadow_id(self) -> int | None:
        return self._shadow_id if self._shadow is not None else None

    @property
    def shadow_confirmed(self) -> bool:
        return self._confirmed

    @property
    def doom_frame(self) -> int | None:
        return self._doom_frame

    @property
    def first_detection_frame(self) -> int | None:
        return self._first_detection_frame

    def cancel(self) -> None:
        """Request cooperative worker shutdown at its next row/candidate boundary."""
        self._cancel.set()

    def fork(self, source: Any, frame: int | None = None, life: int | None = None) -> int:
        """Synchronously bind an exact live clone; future work starts in ``advance``."""
        del frame, life  # Controller owns fences; digest binds the actual state.
        self._shadow = clone_world_exact(source)
        self._cancel.clear()
        self._shadow_id += 1
        self._confirmed = True
        self._doom_frame = None
        self._first_detection_frame = None
        self._first_decision = None
        self._replay_root = clone_world_exact(self._shadow)
        self._expected_digests = []
        self._source_digest = _digest(self._shadow, self.checkpoint)
        return self._shadow_id

    def make_maintenance_request(
        self,
        live_snapshot: Any,
        *,
        frame: int,
        life: int,
        checkpoint: Any = None,
        refork: bool = False,
        confirmations: tuple[tuple[int, int, str], ...] = (),
    ) -> tuple[Any, int, int, Any, bool, tuple[tuple[int, int, str], ...]]:
        """Capture a private exact source for a worker-only maintenance call.

        The caller supplies an adapter-produced prepared snapshot, never a
        mutable live simulator.  Cloning here avoids a later tick mutating the
        branch while it waits for the sole worker.
        """
        return (
            live_snapshot,
            int(frame),
            int(life),
            checkpoint,
            bool(refork),
            tuple(confirmations),
        )

    def maintain(
        self, request: tuple[Any, int, int, Any, bool, tuple[tuple[int, int, str], ...]]
    ) -> MaintenanceResult:
        """Worker-owned confirm/refork/one-step maintenance operation."""
        live_snapshot, frame, life, checkpoint, refork, confirmations = request
        live = clone_world_exact(live_snapshot)
        if checkpoint != self.checkpoint:
            self.checkpoint = checkpoint
            refork = True
        source_digest = _digest(live, self.checkpoint)
        self._confirmed_digest = source_digest
        fresh = refork or self._shadow is None or not self._confirmed
        if not fresh:
            for post_frame, _post_life, digest in confirmations:
                if not self._expected_digests:
                    fresh = True
                    break
                expected_frame, expected_digest = self._expected_digests.pop(0)
                if expected_frame != int(post_frame) or expected_digest != str(digest):
                    fresh = True
                    break
        if fresh:
            self.fork(live, frame, life)
        else:
            # The compact mailbox has already reconciled all live steps.
            pass
        # Catch-up is worker-only.  Subsequent maintenance advances exactly one
        # predicted row per confirmed real tick, preserving the D500 lead.
        self.advance(self.depth if fresh else max(1, len(confirmations)))
        # Roll the compact replay anchor forward after every reconciled live
        # slice.  Replay is therefore bounded by the D500 lead, not session age.
        if self._doom_frame is None:
            self._replay_root = clone_world_exact(live)
        elif self._first_detection_frame is None:
            self._first_detection_frame = frame
        return MaintenanceResult(
            source_frame=frame,
            source_life=life,
            source_checkpoint=self.checkpoint,
            source_digest=source_digest,
            shadow_id=self.shadow_id,
            confirmed=self.shadow_confirmed,
            doom_frame=self.doom_frame,
            first_detection_frame=self.first_detection_frame,
            search_request=self.prepare_search(current_frame=frame),
            expected_digests=tuple(self._expected_digests),
        )

    def invalidate(self, reason: str = "invalidated") -> None:
        """Discard predictions; captured worker requests become stale by id mismatch."""
        del reason
        self._confirmed = False
        self._first_decision = None
        self._expected_digests = []

    def confirm(self, live: Any) -> bool:
        """Confirm the next live state against the corresponding shadow prediction."""
        if not self._confirmed or self._shadow is None:
            return False
        if not self._expected_digests:
            return False
        # ``_shadow`` is D steps ahead.  The FIFO contains post-step state
        # digests in real-tick order, so confirming never compares the live
        # state with the far-future mutable simulator.
        expected_frame, expected = self._expected_digests[0]
        if expected_frame != int(live.sim.frame[0]) or expected != _digest(live, self.checkpoint):
            self.invalidate("full_digest_mismatch")
            return False
        self._expected_digests.pop(0)
        return True

    def advance_one(self) -> ShadowAdvance:
        """Advance at most one row-step and stop permanently after future doom."""
        if not self._confirmed or self._shadow is None or self._doom_frame is not None:
            return ShadowAdvance(
                0, self.shadow_id, self._confirmed, self._doom_frame, self._first_detection_frame
            )
        if len(self._expected_digests) >= self.depth or not bool(self._shadow.sim.alive[0, 0]):
            return ShadowAdvance(
                0, self.shadow_id, self._confirmed, self._doom_frame, self._first_detection_frame
            )
        _thread_cap()
        if self._cancel.is_set():
            return ShadowAdvance(
                0, self.shadow_id, self._confirmed, self._doom_frame, self._first_detection_frame
            )
        # Upkeep needs only the ordinary exact greedy continuation and its
        # post-step digest.  Capturing a full prepared branch / extra B8 Q here
        # would allocate and forward D500 times without an outstanding doom.
        from research.t3_rollout_labels_20261015.bank_engine import choose_and_step

        choose_and_step(self._shadow, capture_obs=False)
        self._expected_digests.append(
            (int(self._shadow.sim.frame[0]), _digest(self._shadow, self.checkpoint))
        )
        if not bool(self._shadow.sim.alive[0, 0]):
            self._doom_frame = int(self._shadow.sim.frame[0])
        return ShadowAdvance(
            1, self.shadow_id, self._confirmed, self._doom_frame, self._first_detection_frame
        )

    def advance(self, max_steps: int = 1) -> ShadowAdvance:
        """Advance up to ``max_steps`` for catch-up callers; controller uses one."""
        if max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        if max_steps == 0:
            return ShadowAdvance(
                0, self.shadow_id, self._confirmed, self._doom_frame, self._first_detection_frame
            )
        total = 0
        state = self.advance_one()
        total += state.steps
        while total < max_steps and state.steps and self._doom_frame is None:
            state = self.advance_one()
            total += state.steps
        return ShadowAdvance(
            total, self.shadow_id, self._confirmed, self._doom_frame, self._first_detection_frame
        )

    def prepare_search(self, *, current_frame: int) -> SearchRequest | None:
        """Capture an immutable earliest-decision request after a confirmed doom."""
        if (
            not self._confirmed
            or self._doom_frame is None
            or self._replay_root is None
            or self._shadow is None
        ):
            return None
        decision = self._first_decision
        # ``decision`` supplies a validated action/Q shape; the worker replays
        # the compact root to the first frame after the latest live cutoff.
        if decision is None:
            decision = _capture_and_step(clone_world_exact(self._replay_root))
        return SearchRequest(
            shadow_id=self._shadow_id,
            source_digest=self._confirmed_digest,
            target_frame=current_frame + 1,
            decision=PreparedDecision(
                current_frame + 1,
                clone_world_exact(self._replay_root),
                decision.bare_action,
                decision.q,
                decision.legal,
            ),
            horizon=max(
                1,
                (
                    min(self.horizon, int(self.terminal_horizon) - (current_frame + 1))
                    if self.terminal_horizon is not None
                    else self.horizon
                ),
            ),
            replay_root=clone_world_exact(self._replay_root),
            cutoff_frame=current_frame,
            doom_frame=self._doom_frame,
            terminal_horizon=self.terminal_horizon,
            cancel_event=self._cancel,
        )

    @staticmethod
    def run_search(request: SearchRequest) -> SearchResult:
        """Evaluate the earliest branch point, returning its first improvement."""
        _thread_cap()
        try:
            root = (
                clone_world_exact(request.replay_root) if request.replay_root is not None else None
            )
            candidate = request.decision
            # Replay from one compact root, discarding already-passed branch
            # points, then search every remaining point in ascending order.
            if root is not None:
                while True:
                    candidate = _capture_and_step(root)
                    cutoff = -1 if request.cutoff_frame is None else int(request.cutoff_frame)
                    if candidate.frame > cutoff:
                        break
                limit = int(request.doom_frame or candidate.frame)
            else:
                limit = candidate.frame
            while candidate.frame <= limit:
                if request.cancel_event is not None and request.cancel_event.is_set():
                    raise TimeoutError("Path B v2 search cancelled")
                actions = tuple(
                    sorted(
                        {
                            candidate.bare_action,
                            *(action for action in range(6) if candidate.legal[action]),
                        }
                    )
                )
                candidate_horizon = min(
                    request.horizon,
                    (
                        int(request.terminal_horizon) - candidate.frame
                        if request.terminal_horizon is not None
                        else request.horizon
                    ),
                )
                if candidate_horizon < 1:
                    break
                outcomes = evaluate_prepared(
                    candidate, actions, candidate_horizon, cancel_event=request.cancel_event
                )
                bare = candidate.bare_action
                baseline = outcomes[bare]
                alternatives = sorted(
                    (action for action in actions if action != bare and candidate.legal[action]),
                    key=lambda action: (-candidate.q[action], action),
                )
                # T3 outcome order: keep bare if it reaches horizon; otherwise a
                # horizon rescue is improving.  If all die, only a strictly longer
                # delay is an improvement; Q/index resolve equal outcomes.
                if baseline.survives:
                    chosen = None
                else:
                    survivors = [action for action in alternatives if outcomes[action].survives]
                    if survivors:
                        chosen = survivors[0]
                    else:
                        longest = max(
                            (outcomes[a].alive_steps for a in alternatives),
                            default=baseline.alive_steps,
                        )
                        longer = [
                            a
                            for a in alternatives
                            if outcomes[a].alive_steps == longest and longest > baseline.alive_steps
                        ]
                        chosen = longer[0] if longer else None
                if chosen is not None:
                    return SearchResult(
                        "IMPROVING",
                        request.shadow_id,
                        candidate.frame,
                        chosen,
                        bare,
                        outcomes,
                        request.source_digest,
                        intervention_kind=("rescue" if outcomes[chosen].survives else "delay"),
                    )
                if root is None:
                    break
                candidate = _capture_and_step(root)
            return SearchResult(
                "NO_IMPROVEMENT",
                request.shadow_id,
                None,
                None,
                candidate.bare_action,
                outcomes,
                request.source_digest,
            )
        except (RuntimeError, TypeError, ValueError, IndexError, KeyError, TimeoutError):
            return SearchResult(
                status="INVALID",
                shadow_id=request.shadow_id,
                target_frame=None,
                action=None,
                bare_action=request.decision.bare_action,
                outcomes={},
                source_digest=request.source_digest,
            )

    def result_current(self, result: SearchResult, *, current_frame: int) -> SearchResult:
        """Return STALE/LATE if a worker result cannot authorize this tick."""
        if not self._confirmed or result.shadow_id != self._shadow_id:
            return SearchResult(
                "STALE",
                result.shadow_id,
                None,
                None,
                result.bare_action,
                result.outcomes,
                result.source_digest,
            )
        if result.status == "IMPROVING" and (
            result.target_frame is None or result.target_frame <= current_frame
        ):
            return SearchResult(
                "LATE",
                result.shadow_id,
                None,
                None,
                result.bare_action,
                result.outcomes,
                result.source_digest,
            )
        return result
