"""Offline exact-world and live Watch adapters for Path B v2.

The Watch adapter drives an independent Grid mirror alongside the actual T15
GameSession and rejects a physical/carry/v8 mismatch. Its full-tick timing
includes that bridge. No HTTP service, score write or release-monitor write is
needed. Only construction happens before the background controller starts.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest.mock import patch

import numpy as np

from research.t34_pathb.controller_v2 import PathBControllerV2, SessionIdentity
from research.t34_pathb.engine_v2 import NonFiniteQ, PathBSearchEngine, _digest
from research.t34_shadow.shadow import clone_world_exact

B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def _finite_output(module: Any, inputs: Any, output: Any) -> None:
    """Detect the contract's unconditional non-finite-Q fault at its producer."""
    import torch

    if not bool(torch.isfinite(output).all()):
        raise NonFiniteQ("nonfinite neural Q")


class RunSession:
    """One isolated world; pacing belongs to the caller, never the controller."""

    def __init__(
        self,
        world: Any,
        arm: str,
        *,
        live: Any = None,
        specs: list | None = None,
        terminal_horizon: int | None = None,
        checkpoint_sha: str = B8_SHA,
    ) -> None:
        if arm not in {"bare", "paced", "unpaced"}:
            raise ValueError("unknown Path B v2 arm")
        # Construction captures before the first selector call. Normalize the
        # non-durable execution sentinel to the same idle value as exact clones.
        world.sim._policy_step_in_progress = False
        self.world, self.arm, self.live = world, arm, live
        self.specs = specs or []
        from research.t3_rollout_labels_20261015.bank_engine import sha256

        checkpoint_path = Path(world.hero.checkpoint_path)
        if sha256(checkpoint_path) != checkpoint_sha:
            raise ValueError("adapter hero differs from its pinned checkpoint")
        self.checkpoint_sha = checkpoint_sha
        self.identity_status = "PASS"
        self.parity_status = "NOT_VERIFIED"
        if _digest(world, checkpoint_sha) != _digest(clone_world_exact(world), checkpoint_sha):
            raise ValueError("adapter clone failed exact source-state identity")
        self.parity_status = "PASS"
        self._finite_hooks = []
        models = {id(world.hero.net): world.hero.net}
        for policy in world.opponents:
            model = getattr(policy, "network", None)
            if model is not None:
                models[id(model)] = model
        for model in models.values():
            self._finite_hooks.append(model.register_forward_hook(_finite_output))
        engine = PathBSearchEngine(checkpoint=checkpoint_sha) if arm != "bare" else None
        if engine is not None:
            engine.terminal_horizon = terminal_horizon
        self.controller = None if engine is None else PathBControllerV2(engine)
        self.life = 0
        self._session_id = f"{world.mix}:{world.seed}:{arm}"
        self.decisions = self.boost_decisions = 0
        self.decisions_delta = self.boost_decisions_delta = 0
        self.bare_action = 1
        self.executed_action = 1
        self.hero_death = False
        self.death_length = 0
        self.last_update_ms = self.shadow_upkeep_ms = 0.0
        self.first_result_latency_ms: float | None = None
        self._shadow_cpu_seconds = 0.0
        self._latency_sequence = 0
        self._live_lives, self._grid_lives = [0] * world.sim.S, [0] * world.sim.S
        self._prev_live = self._prev_grid = [True] * world.sim.S
        self._closed = False
        self._offer_snapshot()

    @property
    def frame(self) -> int:
        return int(self.world.sim.frame[0])

    @property
    def hero_alive(self) -> bool:
        return bool(self.world.sim.alive[0, 0])

    @property
    def hero_length(self) -> int:
        return int(self.world.sim.length[0, 0]) if self.hero_alive else 0

    def identity(self, frame: int | None = None) -> SessionIdentity:
        """Bind a decision/post-step frame to this exact session and hero life."""
        return SessionIdentity(
            self._session_id, self.frame if frame is None else frame, self.life, self.checkpoint_sha
        )

    def _offer_snapshot(self) -> None:
        if self.controller is None:
            return
        # Capture privately before sharing with the worker. Skipping a capture
        # while busy is safe only when every live digest is still observed.
        digest = _digest(self.world, self.checkpoint_sha)
        snapshot = None if self.controller.busy else clone_world_exact(self.world)
        self.controller.after_step(snapshot, self.identity(), digest=digest)

    def step(self) -> None:
        """Execute exactly one hero decision through the v2 action boundary."""
        if self._closed:
            raise RuntimeError("session is closed")
        from src.simd_env.eval_engine import _dispatch_actions

        started = time.perf_counter()
        was_alive = self.hero_alive
        self.decisions_delta = self.boost_decisions_delta = 0
        selected: dict[str, Any] = {}

        def choose(sim: Any) -> np.ndarray:
            self.world.runtime.prepare(sim)
            active = bool(sim.alive[0, 0])
            selected["pre_action_length"] = int(sim.length[0, 0]) if active else 0
            if active and not was_alive:
                self.life += 1
            actions = np.ones((1, sim.S), dtype=np.int64)
            _dispatch_actions(
                sim,
                sim.get_resolved_action_mask(),
                actions,
                [self.world.hero],
                [self.world.opponents],
            )
            self.bare_action = int(actions[0, 0])
            if active:
                mask = sim.get_advisory_action_mask()[0, 0].astype(bool).copy()
                if not mask.any():
                    mask[:3] = True
                if self.controller is not None:
                    action, event = self.controller.decide(
                        self.identity(int(sim.frame[0]) - 1), self.bare_action, mask
                    )
                    actions[0, 0] = action
                    selected["event"] = event
                self.decisions_delta = 1
                self.boost_decisions_delta = int(actions[0, 0] >= 3)
            selected["actions"] = actions.copy()
            self.executed_action = int(actions[0, 0])
            return actions

        self.world.sim.step_with_policy(choose)
        self.world.runtime.observe_step(self.world.sim)
        self.hero_death = bool(self.world.sim.get_transition_valid()[0, 0] and not self.hero_alive)
        self.death_length = int(selected.get("pre_action_length", 0)) if self.hero_death else 0
        self.decisions += self.decisions_delta
        self.boost_decisions += self.boost_decisions_delta
        if self.live is not None:
            self._step_live(selected["actions"])
        upkeep_cpu_started = time.thread_time()
        self._offer_snapshot()
        parent_upkeep_seconds = time.thread_time() - upkeep_cpu_started
        self.first_result_latency_ms = None
        if self.controller is not None:
            state = self.controller.state()
            shadow_cpu = float(state.get("shadow_cpu_seconds", 0.0))
            self.shadow_upkeep_ms = (
                parent_upkeep_seconds + max(0.0, shadow_cpu - self._shadow_cpu_seconds)
            ) * 1000
            self._shadow_cpu_seconds = shadow_cpu
            for sample in state.get("first_result_latency_records", ()):
                if int(sample["sequence"]) > self._latency_sequence:
                    self.first_result_latency_ms = float(sample["milliseconds"])
                    self._latency_sequence = int(sample["sequence"])
        self.last_update_ms = (time.perf_counter() - started) * 1000

    def _step_live(self, actions: np.ndarray) -> None:
        """Execute the same selected hero action in real Watch and audit the bridge."""
        from research.t34_shadow.identity import _carry, _physical, _value_digest, _veto

        hero = self.live.game.snakes[0]
        hero.action_transform = lambda snake, others, q, mask, bare: int(actions[0, 0])
        try:
            self.live.step()
        finally:
            del hero.action_transform
        if self.live.last_error:
            raise RuntimeError(self.live.last_error)
        active_live = [bool(s.is_alive) for s in self.live.game.snakes]
        active_grid = self.world.sim.alive[0].tolist()
        for slot in range(self.world.sim.S):
            self._live_lives[slot] += int(active_live[slot] and not self._prev_live[slot])
            self._grid_lives[slot] += int(active_grid[slot] and not self._prev_grid[slot])
        self._prev_live, self._prev_grid = active_live, active_grid
        pairs = {
            "physical": (
                _physical(
                    self.live.game,
                    self.world,
                    self.specs,
                    self._live_lives,
                    live=True,
                    rng=random.getstate(),
                ),
                _physical(None, self.world, self.specs, self._grid_lives, live=False),
            ),
            "carry": (
                _carry(self.live.game, self.world, self.specs, live=True),
                _carry(None, self.world, self.specs, live=False),
            ),
            "v8": (
                _veto(self.live.game, self.world, self.specs, live=True),
                _veto(None, self.world, self.specs, live=False),
            ),
        }
        for name, (actual, expected) in pairs.items():
            if _value_digest(actual) != _value_digest(expected):
                raise RuntimeError(f"Watch/Grid {name} identity failed at frame {self.frame}")

    def controller_state(self) -> dict[str, Any]:
        """Return bounded decision logs; bare sessions have no controller."""
        return (
            {"active": False, "label": "bare B8 s3"}
            if self.controller is None
            else self.controller.state()
        )

    def settle_between_ticks(self, check_cap: Callable[[], None]) -> None:
        """Hold only the descriptive unpaced clock while its existing job finishes."""
        if self.arm != "unpaced":
            raise RuntimeError("only the descriptive unpaced arm may settle between ticks")
        if self.controller is not None:
            self.controller.settle_between_ticks(self.identity(), check_cap=check_cap)

    def discard_pending_assistance(self) -> bool:
        """Discard this session's outstanding assistance after a host pause.

        The row runner invokes this only at its between-row pause boundary.
        Bare sessions have no assistance to discard.
        """
        return False if self.controller is None else self.controller.discard_pending_assistance()

    def close(self, wait: bool = True) -> None:
        """Retire this adapter's one worker after its last tick."""
        self._closed = True
        if self.controller is not None:
            self.controller.close(wait=wait)
        if wait:
            for hook in self._finite_hooks:
                hook.remove()
            self._finite_hooks.clear()


def open_terminal(row: Mapping[str, Any], fps: int, arm: str) -> RunSession:
    """Construct the exact persisted terminal roster, with bare B8 s3 hero."""
    del fps
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    plan = dict(row.get("roster", row))
    if "specs" not in plan:
        raise ValueError("terminal row requires a materialized roster/specs")
    return RunSession(build_world("b8_s3", plan), arm, terminal_horizon=5000)


def open_precheck(row: Mapping[str, Any], fps: int, arm: str) -> RunSession:
    """Replay one original T3 development source with its original B8 checkpoint."""
    del fps
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t34_pathb.precheck_v2 import validate_t3_row

    validate_t3_row(row)
    world = build_world(str(row["t3_source_arm"]), dict(row))
    return RunSession(
        world, arm, terminal_horizon=5000, checkpoint_sha=str(row["t3_checkpoint"]["sha256"])
    )


def open_watch(row: Mapping[str, Any], fps: int, arm: str) -> RunSession:
    """Construct real continuous Watch6: B8 hero vs five fixed FRP3+v8."""
    from research.apex_veto_serving_20261001.serving_run import parity_config
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t15_mixed_roster_20261017.episodes import new_session
    from src.simd_env import eval_engine as ee
    from src.simd_env.grid_sim import GridBatchSim

    seed = int(row["seed"])
    with parity_config(), patch.dict("os.environ", {"SNAKE_SERVE_PATHB": "off"}):
        session = new_session(seed, 6)
    session.set_speed(fps or 24)
    session.policy.forward = "rowwise"
    spec = ("checkpoint", session.watch_opponent_checkpoint["path"])
    specs = [spec] * 5
    live_rng = random.getstate()
    with patch.object(ee, "_terminal_hero_sim_class", return_value=GridBatchSim):
        world = build_world(
            "b8_s3",
            {"seed": seed, "mix": "mixed", "specs": specs},
            hero_checkpoint=Path(session.served_checkpoint["path"]),
        )
    random.setstate(live_rng)
    for slot, policy in enumerate(world.opponents, 1):
        # Each slot gets independent mutable veto carry; the model stays shared.
        import copy

        policy = copy.copy(policy)
        policy._configure_veto((slot,), "v8", 8.0, None)
        world.opponents[slot - 1] = policy
    return RunSession(world, arm, live=session, specs=specs)
