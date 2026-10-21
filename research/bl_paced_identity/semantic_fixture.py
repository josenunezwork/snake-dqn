"""Small GridBatchSim/BatchSim semantic fixture for Mac preflight only."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from research.b8_death_census_20261014.census import clone_sim
from research.bl_paced_identity.events import canonical_bytes
from research.bl_paced_identity.semantic import SemanticFixture
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


class _SimRuntime:
    """Runtime adapter shared only by the test fixture, never a planner helper."""

    def __init__(self, sim: Any) -> None:
        self.sim = sim

    def clone(self) -> "_SimRuntime":
        return type(self)(clone_sim(self.sim))

    def snapshot(self) -> bytes:
        chunks: list[bytes] = []
        for value in (
            self.sim.frame,
            self.sim.head_ptr,
            self.sim.seg_count,
            self.sim.length,
            self.sim.alive,
            self.sim.direction,
            self.sim.boost_frames,
            self.sim.frames_since_food,
        ):
            array = np.ascontiguousarray(value)
            chunks.extend(
                (
                    array.dtype.str.encode("ascii"),
                    np.asarray(array.shape, dtype="<i8").tobytes(),
                    array.tobytes(),
                )
            )
        for env in range(self.sim.E):
            for slot in range(self.sim.S):
                chunks.append(np.asarray(self.sim.get_bodies(env, slot), dtype="<i8").tobytes())
            chunks.append(np.asarray(self.sim.get_food(env), dtype="<i8").tobytes())
            chunks.append(canonical_bytes(self.sim._rngs[env]._rng.getstate()))
        for name, value in sorted(self.sim.get_step_events().items()):
            chunks.extend((name.encode("ascii"), np.ascontiguousarray(value).tobytes()))
        return b"\0".join(chunks)

    def observation(self) -> bytes:
        """Ordered observation bytes: heads/body order, food order, masks and carry."""
        chunks = [
            np.ascontiguousarray(self.sim.heads(), dtype="<i8").tobytes(),
            np.ascontiguousarray(self.sim.get_action_mask(), dtype=np.bool_).tobytes(),
            np.ascontiguousarray(self.sim.get_resolved_action_mask(), dtype=np.bool_).tobytes(),
            np.ascontiguousarray(self.sim.length, dtype="<i8").tobytes(),
        ]
        for env in range(self.sim.E):
            chunks.append(np.asarray(self.sim.get_food(env), dtype="<i8").tobytes())
            for slot in range(self.sim.S):
                chunks.append(np.asarray(self.sim.get_bodies(env, slot), dtype="<i8").tobytes())
        return b"\0".join(chunks)

    def rank_actions(self) -> Mapping[str, Any]:
        legal = self.sim.get_legal_action_mask()[0, 0].astype(bool).tolist()
        # The governed tie rule is explicit: retain the bare/B8 action first,
        # then action index.  Fixture rankings are intentionally score-free.
        ranking = sorted(range(6), key=lambda action: (not legal[action], action != 1, action))
        return {
            "ranking": ranking,
            "winner": ranking[0],
            "tie_rule": "test-only-score-free",
            "bare_action": 1,
            "q_values": [0.0] * 6,
        }

    def step(self, hero_action: int | None) -> Mapping[str, Any]:
        actions = np.zeros((self.sim.E, self.sim.S), dtype=np.int64)
        legal = self.sim.get_legal_action_mask()
        for env in range(self.sim.E):
            for slot in range(self.sim.S):
                choices = np.flatnonzero(legal[env, slot])
                actions[env, slot] = int(choices[0]) if len(choices) else 0
        # The first branch tick forces one alternative.  Thereafter the fixture
        # independently chooses the bare greedy continuation (action 1 when
        # legal, otherwise the first legal action), matching branch semantics.
        hero_legal = np.flatnonzero(legal[0, 0])
        greedy = 1 if bool(legal[0, 0, 1]) else int(hero_legal[0]) if len(hero_legal) else 0
        actions[0, 0] = greedy if hero_action is None else int(hero_action)
        self.sim.step(actions)
        events = self.sim.get_step_events()
        return {
            "hero_action": int(actions[0, 0]),
            "hero_alive": bool(self.sim.alive[0, 0]),
            "terminal": not bool(self.sim.alive[0, 0]),
            "integer_mass_sum": int(self.sim.length[0, 0]) if bool(self.sim.alive[0, 0]) else 0,
            "events": {name: value.tolist() for name, value in sorted(events.items())},
            "opponent_actions": actions[0, 1:].tolist(),
        }


class GridCandidateRuntime(_SimRuntime):
    """Candidate arm: GridBatchSim's grid-backed implementation."""


class SerialReferenceRuntime(_SimRuntime):
    """Reference arm: BatchSim's serial implementation, separately constructed."""


def tiny_semantic_fixtures() -> list[SemanticFixture]:
    """Return a small, non-qualifying fixture that finishes in seconds on one CPU."""
    config = BatchSimConfig(
        num_envs=1,
        num_snakes=3,
        game_width=200,
        game_height=160,
        initial_food=16,
        max_food=24,
        mechanics_version=2,
        max_capacity=128,
        frame_rate=1,
    )
    return [
        SemanticFixture(
            fixture_id="tiny-grid-vs-serial-seed-741",
            candidate_factory=lambda: GridCandidateRuntime(
                GridBatchSim(config, seeds=[741], train_mode=True, allow_respawn=True, jit=False)
            ),
            reference_factory=lambda: SerialReferenceRuntime(
                BatchSim(config, seeds=[741], train_mode=True, allow_respawn=True)
            ),
            horizon=8,
            metadata={
                "scope": "PRECHECK_ONLY",
                "digest_schema": "partial-fixture-state/v1",
                "full_v3": False,
                "ranking_mode": "test-only-score-free",
                "mix": "synthetic",
                "root_seed": 741,
            },
        )
    ]
