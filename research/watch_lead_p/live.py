"""Exact clones of the real product GameSession, with private continuation RNG.

No recorded action tape or future live draws enter a forecast. Every continuation
starts from the submitted snapshot and uses its own copied RNG and policy carry.
"""

from __future__ import annotations

import copy
import hashlib
import random
import threading
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import cloudpickle
import numpy as np
import torch

from research.bl_served_search import ceiling, exact_stop
from research.bl_served_search.ruler_operator import _jsonable
from research.bl_served_search.served_tick_probe import session_state_digest


def rng_state() -> tuple[Any, Any, Any]:
    """Capture all global stochastic consumers used by the live engine."""
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def set_rng(state: tuple[Any, Any, Any]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


class LiveWorld:
    """A real mixed Watch6 session; policy weights are immutable during cloning."""

    def __init__(self, seed: int) -> None:
        from research.bl_k1.m1 import new_watch
        from research.t1_commitment_20261015.replay import configure_cpu
        from src.core.game_config import get_config

        configure_cpu()

        # The historical helper feeds NumPy's scalar uint32 API. Preserve the
        # complete canonical uint64 seed through a SeedSequence, never modulo.
        def seed_all(value: int) -> None:
            random.seed(value)
            np.random.seed(np.random.SeedSequence(value).generate_state(4))
            torch.manual_seed(value)

        with patch("research.t15_mixed_roster_20261017.episodes.set_world_seed", seed_all):
            self.session, self.roster = new_watch(seed)
        if get_config().game.mechanics_version != 2:
            raise ValueError("P requires live deployment mechanics-v2")
        self.rng = rng_state()
        self.life = 0
        self.waiting = False

    @property
    def frame(self) -> int:
        return int(self.session.game.frame)

    @property
    def hero(self) -> Any:
        return self.session.game.snakes[0]

    @contextmanager
    def active(self) -> Iterator[None]:
        external = rng_state()
        set_rng(self.rng)
        try:
            yield
        finally:
            self.rng = rng_state()
            set_rng(external)

    def clone(self) -> LiveWorld:
        """Copy mutable carry, caches, callbacks and game; share only frozen networks."""
        source = self.session
        memo = {
            id(source._lock): threading.RLock(),
            id(source.policy.net): source.policy.net,
            id(source.watch_opponent_policy.dqn): source.watch_opponent_policy.dqn,
        }
        result = copy.copy(self)
        result.session = copy.deepcopy(source, memo)
        result.rng = copy.deepcopy(self.rng)
        # Functions are immutable to deepcopy, but these closures belong to the
        # cloned game. Rebind explicitly rather than reading the live clock.
        game = result.session.game
        for snake in game.snakes:
            snake._get_frame = lambda game=game: game.frame
            snake._set_frame = game._set_frame
        return result

    def snapshot(self) -> bytes:
        """Trusted internal root transport, including weights, RNG and full carry."""
        value = self.clone()
        value.session._lock = None
        return cloudpickle.dumps(value, protocol=5)

    @classmethod
    def restore(cls, payload: bytes) -> LiveWorld:
        value = cloudpickle.loads(payload)
        if type(value) is not cls:
            raise ValueError("foreign P root transport")
        value.session._lock = threading.RLock()
        return value

    def digest(self) -> str:
        with self.active():
            return session_state_digest(self.session)

    def step(self, forced: int | None = None) -> dict[str, Any]:
        """Execute ordinary Watch.step, optionally overriding only the hero proposal."""
        before = self.frame
        observed: dict[str, Any] = {}

        def select(snake: Any, _others: Any, q: Any, mask: Any, bare: int) -> int:
            if self.waiting:
                self.life += 1
                self.waiting = False
            observed.update(
                bare=int(bare),
                q=self.session.policy.q_for(snake.id).astype(np.float32).tolist(),
                mask=mask.detach().cpu().numpy().astype(bool).tolist(),
                life=self.life,
                length=int(snake.length),
            )
            if forced is not None and not bool(mask[int(forced)]):
                raise ValueError("P attempted a masked hero action")
            return int(bare) if forced is None else int(forced)

        hero = self.hero
        if getattr(hero, "action_transform", None) is not None:
            raise ValueError("P refuses a preexisting controller")
        hero.action_transform = select
        try:
            with self.active(), torch.inference_mode():
                self.session.step()
        finally:
            del hero.action_transform
        if self.frame != before + 1 or self.session.last_error:
            raise RuntimeError("P Watch transition failed")
        death = hero.id in self.session.game.frame_death_causes
        if death:
            self.waiting = True
        return {
            **observed,
            "death": death,
            "frame": self.frame,
            "mass": int(hero.length) if hero.is_alive else 0,
            "alive": bool(hero.is_alive),
        }

    def proposal(self) -> dict[str, Any]:
        """Inspect next real selector on an independent clone, without live mutation."""
        return self.clone().step()


def search(root: LiveWorld, proposal: dict[str, Any]) -> dict[str, Any]:
    """Rescue-only H300; reuse the ruler's T3 ranking and exact singleton stopping."""
    q = np.asarray(proposal["q"], dtype=np.float32)
    bare = int(proposal["bare"])
    legal = np.flatnonzero(proposal["mask"]).tolist()

    def evaluate(action: int) -> dict[str, Any]:
        branch = root.clone()
        survived = 0
        for index in range(300):
            event = branch.step(action if index == 0 else None)
            if event["death"]:
                return {"survives": False, "frames_survived": survived}
            survived += 1
        return {"survives": True, "frames_survived": survived}

    result = exact_stop.run(legal, bare, q, evaluate, ceiling._choose)
    # Rescue-only: no longest-delay all-die intervention and no harvest change.
    chosen = int(result["chosen"])
    if not result["outcomes"][chosen].get("survives"):
        chosen = bare
    result["chosen"] = chosen
    # Retain the reusable ruler's typed canonical representation for auditing.
    result["semantic_sha256"] = hashlib.sha256(
        ceiling.canonical(_jsonable(result)).encode()
    ).hexdigest()
    return result
