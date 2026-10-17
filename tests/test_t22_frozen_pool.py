"""T22 frozen-pool dispatch parity.

The oracle here is the pre-T22 ``StaticPool.dispatch`` implementation at
``4c1e1b2``: stable policy-identity groups, row-major rows, then one policy
call and scatter per group.  The small stateful policies make a change in
call count, row order, skipped dead row, or accidental per-row dispatch
observable as a different recorded action stream and RNG state.

These tests deliberately do not load the pinned research checkpoints.  Their
byte validation and the real vector61 / v8 implementation parity belong to
the pool-construction and vector/v8 focused suites; keeping this oracle
hermetic makes it a useful CPU-only regression guard.
"""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research.redesign_m3c_20261008.league_pool import (
    LeagueVector61Policy,
    PoolIdentity,
    StaticPool,
    build_static_pool,
    default_pool_identities,
)


POLICY_IDS = (
    "b8-s3",
    "frp3-s12",
    "frp3-s12-v8",
    "champion-v8",
    "scripted:greedy_food",
    "scripted:random_safe",
)
# Captured from commit 4c1e1b2's ``StaticPool.dispatch`` before T22 edits.
RECORDED_PRE_T22_STREAM_SHA256 = "0670d50688b6ec03fe8991642ebeb4f56979a336e550f37b37ca9d827e09ff17"


@dataclass
class _Prepared:
    """Frame marker shared with every policy, like the actor's prepared sim."""

    frame: int
    respawned: np.ndarray


@dataclass
class _RecordedPolicy:
    """A deterministic stand-in with policy-local RNG and veto-like state.

    It is intentionally sensitive to both supplied row order and masks.  A
    policy must receive all of its rows in one call; splitting a group shifts
    its RNG consumption and therefore the later recorded stream.
    """

    policy_id: str
    seed: int
    rng: np.random.Generator = field(init=False)
    calls: list[tuple[int, list[list[int]], str, int]] = field(default_factory=list)
    veto_counters: dict[tuple[int, int], int] = field(default_factory=dict)
    last_forward_gpu_s: float = 0.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def actions(self, masks: np.ndarray, prepared: _Prepared, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        masks = np.asarray(masks, dtype=bool)
        assert masks.shape == (len(rows), 6)
        assert len(rows), "dispatch must never call a policy with an empty group"
        digest = hashlib.sha256(masks.tobytes()).hexdigest()
        self.calls.append((prepared.frame, rows.tolist(), digest, id(prepared)))
        draws = self.rng.integers(0, 6, size=len(rows), dtype=np.int64)
        actions = np.empty(len(rows), dtype=np.int64)
        for index, (env, slot) in enumerate(rows):
            legal = np.flatnonzero(masks[index])
            # The normal-action fallback is part of the production contract.
            # The fixtures always present at least one legal normal action.
            assert len(legal)
            action = int(legal[int(draws[index]) % len(legal)])
            key = (int(env), int(slot))
            # Treat respawn as a new wrapper episode, as live veto wrappers do.
            previous = 0 if prepared.respawned[env, slot] else self.veto_counters.get(key, 0)
            self.veto_counters[key] = previous + int(action >= 3)
            actions[index] = action
        return actions


def _identity(policy_id: str) -> PoolIdentity:
    # Dispatch only needs an identity order.  Scripted identities keep this
    # oracle independent of checkpoint bytes and their machine-specific paths.
    return PoolIdentity(policy_id, "scripted", None, None, "scripted", "resolved")


def _pool() -> tuple[StaticPool, dict[str, _RecordedPolicy]]:
    policies = {
        policy_id: _RecordedPolicy(policy_id, 50_000 + index * 101)
        for index, policy_id in enumerate(POLICY_IDS)
    }
    return StaticPool([_identity(policy_id) for policy_id in POLICY_IDS], policies), policies


def _masks(frame: int) -> np.ndarray:
    """Legal masks with varying normal/boost choices and no empty live rows."""
    masks = np.zeros((3, 4, 6), dtype=bool)
    for env in range(masks.shape[0]):
        for slot in range(masks.shape[1]):
            masks[env, slot, (frame + env + slot) % 3] = True
            masks[env, slot, 3 + ((frame + 2 * env + slot) % 3)] = (env + slot + frame) % 2 == 0
    return masks


def _seats() -> np.ndarray:
    return np.array(
        [
            ["b8-s3", "frp3-s12", None, "scripted:greedy_food"],
            ["frp3-s12-v8", "champion-v8", "scripted:random_safe", "b8-s3"],
            ["scripted:greedy_food", "frp3-s12", "champion-v8", "frp3-s12-v8"],
        ],
        dtype=object,
    )


def _baseline_dispatch(
    pool: StaticPool,
    prepared: _Prepared,
    masks: np.ndarray,
    alive: np.ndarray,
    actions: np.ndarray,
    seats: np.ndarray,
) -> dict[str, Any]:
    """Literal 4c1e1b2 dispatch oracle, kept local rather than in production."""
    groups = pool.groups_for(seats, alive)
    counts = {policy_id: int(len(rows)) for policy_id, rows in groups.items()}
    started = time.perf_counter()
    gpu_seconds = 0.0
    for policy_id, rows in groups.items():
        choice = np.asarray(
            pool.policies[policy_id].actions(masks[rows[:, 0], rows[:, 1]], prepared, rows),
            dtype=np.int64,
        ).reshape(-1)
        gpu_seconds += float(getattr(pool.policies[policy_id], "last_forward_gpu_s", 0.0))
        if len(choice) != len(rows):
            raise RuntimeError(f"{policy_id}: returned {len(choice)} actions for {len(rows)} rows")
        actions[rows[:, 0], rows[:, 1]] = choice
    return {
        "frozen_forward_s": time.perf_counter() - started,
        "frozen_forward_gpu_s": gpu_seconds,
        "frozen_live_rows": counts,
    }


def _run(pool: StaticPool, policies: dict[str, _RecordedPolicy], dispatch: Any) -> tuple:
    """Record a death/respawn stream through either reference or optimized dispatch."""
    seats = _seats()
    # Kill and respawn one seat of *every* policy.  This makes any wrapper
    # state reset or dead-row shortcut visible without relying on stochastic
    # deaths in a heavyweight real checkpoint run.
    alives = [np.ones((3, 4), dtype=bool)]
    for policy_id in POLICY_IDS:
        dead = np.ones((3, 4), dtype=bool)
        dead[tuple(np.argwhere(seats == policy_id)[0])] = False
        alives.extend((dead, np.ones((3, 4), dtype=bool)))
    stream: list[np.ndarray] = []
    previous_alive = np.ones_like(alives[0])
    died: set[str] = set()
    respawned_policy: set[str] = set()
    for frame, alive in enumerate(alives):
        actions = np.full((3, 4), -99, dtype=np.int64)
        respawned = alive & ~previous_alive
        died.update(str(seats[e, s]) for e, s in np.argwhere(previous_alive & ~alive))
        respawned_policy.update(str(seats[e, s]) for e, s in np.argwhere(respawned))
        prepared = _Prepared(frame, respawned)
        timing = dispatch(pool, prepared, _masks(frame), alive, actions, seats)
        assert timing["frozen_live_rows"] == {
            policy_id: int(np.sum((seats == policy_id) & alive))
            for policy_id in POLICY_IDS
            if np.any((seats == policy_id) & alive)
        }
        stream.append(actions)
        prepared_ids: set[int] = set()
        for policy in policies.values():
            prepared_ids.update(call[3] for call in policy.calls if call[0] == frame)
        assert prepared_ids == {id(prepared)}
        # Learner and dead slots never receive a frozen action.
        assert actions[0, 2] == -99
        assert np.all(actions[~alive] == -99)
        previous_alive = alive
    assert died == set(POLICY_IDS)
    assert respawned_policy == set(POLICY_IDS)
    state = {
        policy_id: {
            "rng": copy.deepcopy(policy.rng.bit_generator.state),
            # ``prepared`` is intentionally a distinct object in the two runs;
            # object identity is checked per run above, while this oracle keeps
            # only the semantically relevant trace.
            "calls": [(frame, rows, digest) for frame, rows, digest, _ in policy.calls],
            "veto_counters": dict(policy.veto_counters),
        }
        for policy_id, policy in policies.items()
    }
    return stream, state


def test_optimized_dispatch_matches_pre_t22_reference_action_stream_with_respawns():
    """All six policies retain baseline actions, local RNG and wrapper state."""
    reference_pool, reference_policies = _pool()
    optimized_pool, optimized_policies = _pool()
    expected_stream, expected_state = _run(reference_pool, reference_policies, _baseline_dispatch)
    actual_stream, actual_state = _run(
        optimized_pool,
        optimized_policies,
        lambda pool, *args: pool.dispatch(*args),
    )
    assert len(actual_stream) == len(expected_stream)
    for frame, (expected, actual) in enumerate(zip(expected_stream, actual_stream)):
        assert np.array_equal(actual, expected), frame
    assert actual_state == expected_state
    payload = b"".join(frame.tobytes() for frame in actual_stream)
    assert hashlib.sha256(payload).hexdigest() == RECORDED_PRE_T22_STREAM_SHA256


def test_optimized_dispatch_uses_one_row_major_batch_per_live_policy_group():
    """No per-row fallback may hide under action parity for stateless policies."""
    pool, policies = _pool()
    seats = _seats()
    alive = np.ones((3, 4), dtype=bool)
    actions = np.full((3, 4), -99, dtype=np.int64)
    prepared = _Prepared(17, np.zeros((3, 4), dtype=bool))
    pool.dispatch(prepared, _masks(17), alive, actions, seats)
    for policy_id, policy in policies.items():
        expected_rows = np.argwhere((seats == policy_id) & alive).tolist()
        assert len(policy.calls) == 1, policy_id
        frame, rows, _digest, prepared_id = policy.calls[0]
        assert frame == 17 and rows == expected_rows and prepared_id == id(prepared)
    assert np.all(actions[alive & (seats != None)] >= 0)  # noqa: E711 - elementwise None


@pytest.mark.parametrize(
    "dispatch",
    [_baseline_dispatch, lambda pool, *args: pool.dispatch(*args)],
    ids=["pre-t22-reference", "optimized"],
)
def test_dispatch_preserves_empty_dead_and_learner_rows(dispatch: Any):
    """The optimized path retains the reference's no-write boundaries."""
    pool, policies = _pool()
    seats = np.array([[None, "b8-s3"], ["frp3-s12-v8", None]], dtype=object)
    alive = np.array([[True, False], [False, True]], dtype=bool)
    actions = np.full((2, 2), -17, dtype=np.int64)
    timing = dispatch(
        pool,
        _Prepared(0, np.zeros((2, 2), dtype=bool)),
        np.ones((2, 2, 6), dtype=bool),
        alive,
        actions,
        seats,
    )
    assert actions.tolist() == [[-17, -17], [-17, -17]]
    assert timing["frozen_live_rows"] == {}
    assert all(not policy.calls for policy in policies.values())


@pytest.mark.skipif(
    not (Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth")).is_file(),
    reason="needs the pinned frp3-s12 checkpoint",
)
def test_real_vector61_v8_adapter_matches_live_hook_for_all_slots_and_q_ties():
    """The compiled pool wrapper retains the live v8 decisions on nonzero slots.

    This deliberately uses tied Q values, where an innocent reorder can change
    ``argmax``.  It drives the actual vector61 runtime across several frames;
    the fixed synthetic stream above separately forces a death/respawn for
    every one of the six pool identities without needing six large checkpoints.
    """
    import torch

    from research.redesign_m2_20261008 import ni_spec
    from research.redesign_m3c_20261008 import train_m3c as trainer
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    ctx = gid._context(1)
    args = trainer.default_args()
    args.worlds, args.seed = 4, 741
    args.frozen_frac, args.mixed_frac = 1.0 / 3.0, 1.0 / 3.0
    sim, runtime, _policies, _groups = trainer.build_segment(args, ctx, 0, 0.99, hero_v61=True)
    # The production pool prunes this to its vector61 rows.  This direct
    # adapter test widens it so the same real decision path is exercised in
    # every nonzero slot as well as slot zero.
    runtime.controlled[:] = True
    carrier = ctx["lookup"][ni_spec.FRP3_S12[1]][1]
    live = Vector61SimdPolicy(
        carrier,
        runtime,
        veto_slots=range(sim.S),
        veto_variant="v8",
        veto_lambda=8.0,
        forward="batched",
    )
    compiled_inner = Vector61SimdPolicy(
        carrier,
        runtime,
        veto_slots=range(sim.S),
        veto_variant="v8",
        veto_lambda=8.0,
        forward="batched",
    )
    compiled = LeagueVector61Policy(compiled_inner)

    def tied_q(states: np.ndarray):
        # Ties in every normal and boost pair exercise the tie-sensitive branch
        # called out by the Mac/x86 identity record.
        values = np.tile(np.array([3, 3, 2, 3, 3, 2], dtype=np.float32), (len(states), 1))
        return torch.from_numpy(values)

    live.q_values = tied_q
    compiled_inner.q_values = tied_q
    recorded: list[np.ndarray] = []
    saw_nonzero_slot = False
    for _frame in range(8):
        runtime.prepare(sim)
        alive = sim.get_alive()
        rows = np.argwhere(alive)
        masks = sim.get_resolved_action_mask()[rows[:, 0], rows[:, 1]]
        expected = live.actions(masks, sim, rows)
        actual = compiled.actions(masks, sim, rows)
        assert np.array_equal(actual, expected)
        recorded.append(actual.copy())
        saw_nonzero_slot |= bool(np.any(rows[:, 1] != 0))
        actions = np.ones((sim.E, sim.S), dtype=np.int64)
        actions[rows[:, 0], rows[:, 1]] = expected
        sim.step(actions)
        runtime.observe_step(sim)
    assert saw_nonzero_slot
    assert hashlib.sha256(b"".join(actions.tobytes() for actions in recorded)).hexdigest() == (
        "80422bc3d307b4a25bdafcc84ac7fb01cb55a09810e8b0f37bb12e0edb5c48ca"
    )


@pytest.mark.skipif(
    not (Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")).is_file(),
    reason="needs pinned six-policy pool checkpoints",
)
def test_real_six_policy_pool_recorded_stream_matches_pre_t22_wrappers_with_forced_respawns():
    """Real checkpoints: old v8 hooks vs T22 adapter, B8 cache, anchors and bare FRP."""
    from research.redesign_m3c_20261008 import train_m3c as trainer
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    ctx, args = gid._context(1), trainer.default_args()
    args.worlds, args.seed = 8, 19018
    args.league_checkpoint_dir = "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes"
    args.league_b8_checkpoint = str(Path(args.league_checkpoint_dir) / "b8_s3_t7_20261016.pth")
    sim, runtime, _policies, _groups = trainer.build_segment(args, ctx, 0, 0.99, hero_v61=False)
    identities = default_pool_identities(args.league_checkpoint_dir, args.league_b8_checkpoint)
    optimized = build_static_pool(identities, world_seeds=sim.get_world_seeds(), runtime=runtime, device="cpu")
    baseline = build_static_pool(identities, world_seeds=sim.get_world_seeds(), runtime=runtime, device="cpu")
    # This is the pre-T22 policy shape: v8 was Vector61SimdPolicy directly.
    for policy_id, policy in list(baseline.policies.items()):
        if isinstance(policy, LeagueVector61Policy):
            baseline.policies[policy_id] = policy._policy
    seats = np.array([POLICY_IDS] * sim.E, dtype=object)
    vector_ids = {"frp3-s12", "frp3-s12-v8", "champion-v8"}
    runtime.controlled[:] = np.isin(seats, tuple(vector_ids))
    recorded: list[np.ndarray] = []
    deaths, respawns = set(), set()
    previous_alive = sim.get_alive().copy()
    for frame in range(12):
        active = np.ones(sim.E, dtype=bool)
        sim.frame[active] += 1
        sim._maintain_food()
        sim._respawn_dead()
        sim._refresh_action_masks(active)
        # Each identity is forced dead once and respawns on the next frame;
        # this tests the precise empty-row and runtime-carry boundary cheaply.
        if frame % 2 == 0 and frame // 2 < len(POLICY_IDS):
            slot = frame // 2
            sim.alive[0, slot] = False
            sim.respawn_timer[0, slot] = 1
        runtime.prepare(sim)
        alive, masks = sim.get_alive(), sim.get_resolved_action_mask()
        deaths.update(str(seats[e, s]) for e, s in np.argwhere(previous_alive & ~alive))
        respawns.update(str(seats[e, s]) for e, s in np.argwhere(alive & ~previous_alive))
        expected = np.full((sim.E, sim.S), -1, dtype=np.int64)
        actual = expected.copy()
        _baseline_dispatch(baseline, sim, masks, alive, expected, seats)
        learner_rows = np.argwhere((seats == "b8-s3") & alive)
        optimized.prime_b8_with_learner_rows(sim, learner_rows, alive, seats)
        optimized.dispatch(sim, masks, alive, actual, seats)
        assert np.array_equal(actual, expected), frame
        recorded.append(actual)
        step_actions = np.where(actual >= 0, actual, 1)
        sim.step(step_actions)
        runtime.observe_step(sim)
        previous_alive = alive.copy()
    assert deaths == set(POLICY_IDS) and respawns == set(POLICY_IDS)
    assert hashlib.sha256(b"".join(item.tobytes() for item in recorded)).hexdigest() == (
        "c4e59e36efe016dfa215593eb4f6856862ead686e26a69d210506ecb4d21596f"
    )
