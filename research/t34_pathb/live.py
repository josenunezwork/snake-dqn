"""Isolated continuations of the actual served GameState, using T3 expansion."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import time
from typing import Any

import numpy as np

from research.t3_rollout_labels_20261015.engine import verify_expand

MAX_PAYLOAD = 64 * 1024 * 1024


def fingerprint(game: Any) -> str:
    """Bind all mutable state, RNG and configuration through snapshot serialization."""
    return hashlib.sha256(snapshot(game, depth=1)).hexdigest()


def snapshot(game: Any, depth: int = 40, deadline: float | None = None) -> bytes:
    """Detach the real world; never serialize the controller or its executor."""
    import cloudpickle

    from src.core.game_config import get_config

    hero = game.snakes[0]
    hook = hero.__dict__.pop("action_transform", None)
    try:
        payload = cloudpickle.dumps(
            (game, get_config(), random.getstate(), np.random.get_state(), int(depth), deadline),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    finally:
        if hook is not None:
            hero.action_transform = hook
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Path B snapshot exceeds 64 MiB")
    return payload


class LiveContinuation:
    """One private branch; globals are switched only inside the isolated worker."""

    def __init__(self, payload: bytes) -> None:
        import cloudpickle

        from src.core.game_config import initialize_config

        self.game, config, self.py_rng, self.np_rng, self.depth, self.deadline = cloudpickle.loads(
            payload
        )
        initialize_config(config)
        self.payload = payload
        self.metadata: dict[str, Any] = {}

    def clone(self) -> "LiveContinuation":
        return LiveContinuation(self.payload)

    @property
    def alive(self) -> bool:
        return bool(self.game.snakes[0].is_alive)

    def step(self, first_action: int | None = None) -> None:
        """Use the unchanged live simulation, opponent policies and wrappers."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Path B speculative deadline expired")
        random.setstate(self.py_rng)
        np.random.set_state(self.np_rng)
        hero = self.game.snakes[0]

        def force(snake: Any, others: Any, q: Any, mask: Any, bare: int) -> int:
            self.metadata = {
                "bare_action": int(bare),
                "mask": mask.detach().cpu().numpy().astype(bool).tolist(),
                "q": q.detach().cpu().numpy().astype(float).tolist(),
                "prepared_fingerprint": prepared_fingerprint(self.game),
            }
            # Keep the source lifecycle in the target-state identity, then
            # prevent speculative credit after the hero's first death.
            snake.auto_respawn = False
            return int(bare if first_action is None else first_action)

        hero.action_transform = force
        try:
            self.game.update(train_mode=False, learn=False, allow_respawn=True)
            self.game.food_manager.trim_ambient(self.game.food_manager.max_food)
        finally:
            del hero.action_transform
        self.py_rng, self.np_rng = random.getstate(), np.random.get_state()


def prepared_state(game: Any) -> Any:
    """Exact decision-world identity, including food order and opponent carry.

    This deliberately excludes policy dispatch caches, telemetry, and the action
    hook: they differ between an observed and a speculative call. Mutable snake
    state is serialized, with policy/model and callback fields omitted.
    """
    from src.core.game_config import get_config

    def canonical(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return [value.dtype.str, list(value.shape), value.tobytes().hex()]
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
        if isinstance(value, (tuple, list)):
            return [canonical(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return [canonical(v) for v in sorted(value, key=repr)]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "__dict__"):
            return canonical(vars(value))
        if hasattr(type(value), "__slots__"):
            return canonical({key: getattr(value, key) for key in type(value).__slots__})
        return str(value)

    rows = []
    for snake in game.snakes:
        row = {}
        for name, value in vars(snake).items():
            if name in {
                "policy",
                "ai",
                "safety_veto",
                "checkpoint_manager",
                "action_transform",
            } or callable(value):
                continue
            # Tensors are encoded by numerical bytes, not storage identifiers.
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            row[name] = value
        rows.append(row)
    value = (
        int(game.frame),
        rows,
        vars(game.food_manager),
        get_config(),
        random.getstate(),
        np.random.get_state(),
    )
    return canonical(value)


def prepared_fingerprint(game: Any) -> str:
    """Hash the canonical decision-time world without model/storage identities."""
    return hashlib.sha256(json.dumps(prepared_state(game), sort_keys=True).encode()).hexdigest()


def evaluate(payload: bytes) -> dict[str, Any]:
    """Run the frozen T3 verify/expand kernel over an isolated live snapshot."""
    # Sync-unlimited is a diagnostic in the parent. Restore its global state too.
    from src.core.game_config import get_config, initialize_config

    config, py_rng, np_rng = get_config(), random.getstate(), np.random.get_state()
    try:
        world = LiveContinuation(payload)
        probe = world.clone()
        probe.step()
        meta = probe.metadata
        if not meta:
            raise ValueError("no living target hero decision")
        result = verify_expand(world, meta["bare_action"], meta["q"], meta["mask"], world.depth)
        return {**meta, **result}
    finally:
        initialize_config(config)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
