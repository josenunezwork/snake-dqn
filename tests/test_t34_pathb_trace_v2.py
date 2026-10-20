"""Real-world Path B v2 continuation parity evidence.

These tests deliberately use the persisted B8 checkpoint and the T3 captured
world seam.  The assertion is a full state digest after every transition, so
the check covers ordered bodies/food, simulator RNG, Vector61 carry and v8
state in addition to the selected hero action.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared
from research.t34_pathb.engine_v2 import (
    PathBSearchEngine,
    PreparedDecision,
    SearchResult,
    _capture_and_step,
    _prepared_metadata,
    evaluate_prepared,
)
from research.t34_shadow.shadow import clone_world_exact, full_digest

B8_CHECKPOINT = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"

pytestmark = pytest.mark.skipif(
    not Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth"
    ).is_file(),
    reason="released B8 gameplay artifacts are not installed",
)


def _world(mix: str):
    from research.t34_shadow.identity import _new_pair

    return _new_pair(2026101934, mix)


def _capture_prepared(world):
    """Capture one real selector-prepared frame without consuming a move."""

    class StopCapture(Exception):
        pass

    captured = {}

    def choose(sim):
        world.runtime.prepare(sim)
        captured["world"] = clone_world_exact(world)
        captured["world"].sim._policy_phase_prepared = True
        captured["world"].sim._policy_step_in_progress = False
        raise StopCapture

    try:
        world.sim.step_with_policy(choose)
    except StopCapture:
        return captured["world"]
    raise AssertionError("selector did not reach the prepared capture seam")


def _trace_reference(world, frames: int):
    trace = []
    for _ in range(frames):
        before_rng = random.getstate()
        actions, _mask, _obs = choose_and_step(world, capture_obs=False)
        after_rng = random.getstate()
        trace.append(
            {
                "frame": int(world.sim.frame[0]),
                "action": int(actions[0, 0]),
                "digest": full_digest(world, checkpoint=B8_CHECKPOINT),
                "global_rng_after": after_rng,
                "global_rng_changed": before_rng != after_rng,
            }
        )
    return trace


def _trace_engine(world, frames: int):
    trace = []
    for _ in range(frames):
        before_rng = random.getstate()
        decision = _capture_and_step(world)
        after_rng = random.getstate()
        trace.append(
            {
                "frame": int(world.sim.frame[0]),
                "action": int(decision.bare_action),
                "digest": full_digest(world, checkpoint=B8_CHECKPOINT),
                "global_rng_after": after_rng,
                "global_rng_changed": before_rng != after_rng,
            }
        )
    return trace


def test_reference_trace_matches_engine_for_scripted_and_mixed_rosters() -> None:
    """Compare choose_and_step and the optimized capture path frame by frame."""

    for mix in ("scripted", "mixed"):
        live_session, source, _specs, _identity = _world(mix)
        try:
            reference, optimized = clone_world_exact(source), clone_world_exact(source)
            ambient = random.getstate()
            random.seed(7101)
            reference_trace = _trace_reference(reference, 4)
            random.seed(7101)
            optimized_trace = _trace_engine(optimized, 4)
            assert optimized_trace == reference_trace, f"trace mismatch for {mix}"
            assert [row["frame"] for row in optimized_trace] == [1, 2, 3, 4]
        finally:
            random.setstate(ambient)
            live_session.close()


def test_prepared_trace_covers_real_trapped_tie_and_wall_death(monkeypatch) -> None:
    """Exercise prepared dispatch on an actual world, including terminal state."""

    live_session, source, _specs, _identity = _world("scripted")
    try:
        prepared = _capture_prepared(source)
        sim = prepared.sim
        # All advisory actions rejected is the real trapped fallback: the
        # selector must use normal actions and lowest-index Q ties.
        sim._last_mask[0, 0, :] = False
        sim._last_resolved_mask[0, 0, :3] = True
        monkeypatch.setattr(
            "research.t3_rollout_labels_20261015.bank_engine.prepared_q",
            lambda *_args: np.ones(6, dtype=np.float32),
        )
        trapped_reference = clone_world_exact(prepared)
        trapped_engine = clone_world_exact(prepared)
        step_prepared(trapped_reference, 0)
        decision = _capture_and_step(trapped_engine)
        assert decision.bare_action == 0
        assert full_digest(trapped_engine, checkpoint=B8_CHECKPOINT) == full_digest(
            trapped_reference, checkpoint=B8_CHECKPOINT
        )

        # Put the real hero one cell from the wall and force the tied straight
        # action.  The transition is produced by GridBatchSim's actual wall
        # collision path; only the network tie is controlled for reproducibility.
        death_reference = clone_world_exact(prepared)
        death_engine = clone_world_exact(prepared)
        for branch in (death_reference, death_engine):
            branch.sim.bodies[0, 0, branch.sim.head_ptr[0, 0]] = (1, 0)
            branch.sim.direction[0, 0] = 0
            branch.sim._rebuild_traversed_from_heads()
            branch.sim._last_mask[0, 0, :] = True
            branch.sim._last_resolved_mask[0, 0, :3] = True
        monkeypatch.setattr(
            "research.t3_rollout_labels_20261015.bank_engine.prepared_q",
            lambda *_args: np.asarray([0.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        step_prepared(death_reference, 1)
        death_decision = _capture_and_step(death_engine)
        assert death_decision.bare_action == 1
        assert not bool(death_engine.sim.alive[0, 0])
        assert int(death_engine.sim._last_death_cause[0, 0]) == 1
        assert full_digest(death_engine, checkpoint=B8_CHECKPOINT) == full_digest(
            death_reference, checkpoint=B8_CHECKPOINT
        )
    finally:
        live_session.close()


def test_real_shadow_fence_rejects_invalidated_and_late_results() -> None:
    live_session, source, _specs, _identity = _world("mixed")
    try:
        engine = PathBSearchEngine(depth=500, checkpoint=B8_CHECKPOINT)
        shadow_id = engine.fork(source)
        result = SearchResult("IMPROVING", shadow_id, 8, 2, 0, {}, "source")
        assert engine.result_current(result, current_frame=7).status == "IMPROVING"
        engine.invalidate("real-digest-mismatch")
        assert engine.result_current(result, current_frame=7).status == "STALE"

        engine.fork(source)
        fresh_id = engine.shadow_id
        late = SearchResult("IMPROVING", fresh_id, 8, 2, 0, {}, engine._source_digest)
        assert engine.result_current(late, current_frame=8).status == "LATE"
    finally:
        live_session.close()


def test_native_e6_mixed_v8_outcomes_match_independent_reference() -> None:
    """Compare native E=6 branch rollouts with one-by-one exact continuations."""

    live_session, source, _specs, _identity = _world("mixed")
    try:
        prepared = _capture_prepared(source)
        bare, q, legal_mask = _prepared_metadata(prepared)
        actions = tuple(index for index, legal in enumerate(legal_mask) if legal)
        decision = PreparedDecision(
            frame=int(prepared.sim.frame[0]) - 1,
            world=prepared,
            bare_action=bare,
            q=q,
            legal=legal_mask,
        )
        native = evaluate_prepared(decision, actions, horizon=3)
        reference = {}
        for action in actions:
            branch = clone_world_exact(prepared)
            step_prepared(branch, action)
            alive_steps = int(bool(branch.sim.alive[0, 0]))
            mass = int(branch.sim.length[0, 0]) if branch.sim.alive[0, 0] else 0
            executed = 1
            while executed < 3 and bool(branch.sim.alive[0, 0]):
                choose_and_step(branch, capture_obs=False)
                executed += 1
                alive_steps += int(bool(branch.sim.alive[0, 0]))
                mass += int(branch.sim.length[0, 0]) if branch.sim.alive[0, 0] else 0
            reference[action] = (bool(branch.sim.alive[0, 0]), alive_steps, float(mass))
        for action in actions:
            outcome = native[action]
            assert (outcome.survives, outcome.alive_steps, outcome.mass) == reference[action]
    finally:
        live_session.close()
