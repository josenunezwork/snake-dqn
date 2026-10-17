"""Descriptive T13 Watch-roster measurements through the real serving session.

The mixed roster is deliberately a harness-only mutation after both policies have
been constructed through their normal registry paths.  No product default or
registry entry is changed by this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

# Set before imports that reach numpy/torch.  A worker is one CPU slot.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")

from research.apex_veto_serving_20261001.serving_run import set_world_seed  # noqa: E402

ROSTERS = ("all_b8", "all_frp3", "all_champion", "mixed")
_SERVING_ENV = (
    "SNAKE_SERVE_CHECKPOINT",
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
    "SNAKE_SERVE_VETO_VARIANT",
)


@contextlib.contextmanager
def _selection(name: str | None, *, veto: bool | None) -> Iterator[None]:
    """Temporarily select one real registry posture without leaking it."""
    saved = {key: os.environ.pop(key) for key in _SERVING_ENV if key in os.environ}
    try:
        if name:
            os.environ["SNAKE_SERVE_CHECKPOINT"] = name
        if veto is True:
            os.environ["SNAKE_SERVE_VETO_WATCH_HERO"] = "1"
            os.environ["SNAKE_SERVE_VETO_VARIANT"] = "v8"
        elif veto is False:
            os.environ["SNAKE_SERVE_VETO_WATCH_HERO"] = "0"
        yield
    finally:
        for key in _SERVING_ENV:
            os.environ.pop(key, None)
        os.environ.update(saved)


def _sha(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_session(checkpoint: str | None, seed: int, *, veto: bool | None) -> Any:
    from web.backend.session import GameSession

    set_world_seed(seed)
    # This is the T12 Watch12 construction hook.  It changes only the config
    # object handed to this constructor and restores the loader immediately.
    original_loader = __import__("web.backend.session", fromlist=["load_config"]).load_config

    def twelve_snake_config(path: str) -> Any:
        config = original_loader(path)
        return replace(config, game=replace(config.game, num_snakes=12))

    with _selection(checkpoint, veto=veto):
        # Registry resolution is still real; suppress only monitoring side
        # effects because T13 raw data is an offline descriptive measurement.
        with (
            patch("web.backend.session.load_config", side_effect=twelve_snake_config),
            patch("web.backend.session.served_resolution"),
            patch("web.backend.session.watch_tick"),
        ):
            session = GameSession()
    # T13 is an offline measurement.  Do not write release-monitor samples or
    # latency logs merely because a B8 registry session was used to build it.
    session._release_monitoring = False
    return session


def _identity(session: Any) -> dict[str, Any]:
    state = session.safety_veto_state() or {}
    return {
        "registry": dict(session.served_checkpoint or {}),
        "checkpoint_path": session.checkpoint_path,
        "checkpoint_sha256": _sha(session.checkpoint_path),
        "config_path": session.config_path,
        "config_sha256": _sha(session.config_path),
        "obs_spec": str(session.obs_spec),
        "policy_class": type(session.policy).__name__,
        "veto": {
            "active": state.get("active"),
            "variant": state.get("variant"),
            "wrapped_snake_ids": list(state.get("wrapped_snake_ids", [])),
            "reason": state.get("reason"),
        },
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _initial_world(session: Any) -> dict[str, Any]:
    """Stable initial-world and RNG evidence before any episode decisions."""
    import numpy as np

    return {
        "digest": _digest(
            (
                int(session.game.frame),
                [
                    (
                        int(s.id),
                        bool(s.is_alive),
                        int(s.length),
                        tuple(s.segments),
                        tuple(s.direction),
                    )
                    for s in session.game.snakes
                ],
                tuple(sorted(tuple(food) for food in session.game.food)),
            )
        ),
        "python_rng_digest": _digest(random.getstate()),
        "numpy_rng_digest": _digest(np.random.get_state()),
        "effective_num_snakes": len(session.game.snakes),
    }


def _snake_identities(snakes: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "snake_id": int(snake.id),
            "policy_class": type(getattr(snake, "policy", None)).__name__,
            "policy_checkpoint": getattr(getattr(snake, "policy", None), "checkpoint_path", None),
            "veto": (
                None
                if getattr(snake, "safety_veto", None) is None
                else {
                    "class": type(snake.safety_veto).__name__,
                    "method": getattr(snake.safety_veto, "method", None),
                    "lambda": getattr(snake.safety_veto, "lam", None),
                    "head_avoidance": getattr(snake.safety_veto, "head_avoidance", None),
                    "reference_lambda": getattr(snake.safety_veto, "reference_lambda", None),
                }
            ),
        }
        for snake in snakes
    ]


def _assert_roster(session: Any, roster: str, donor: Any | None) -> None:
    """Fail closed if selection, wrapper scope, or policy ownership differs."""
    expected = {
        "all_b8": "b8-s3",
        "all_frp3": "frp3-s12",
        "all_champion": "champion",
        "mixed": "b8-s3",
    }[roster]
    choice = session.served_checkpoint or {}
    if choice.get("name") != expected or choice.get("reason") or choice.get("pin_problems"):
        raise RuntimeError(f"T13 registry selection is not exact {expected}: {choice!r}")
    snakes = list(session.game.snakes)
    if len(snakes) != 12 or int(snakes[0].id) != int(session.hero_id):
        raise RuntimeError("T13 requires the T12 Watch12 roster with slot-zero hero")
    if roster == "all_b8":
        if any(
            s.policy is not session.policy or getattr(s, "safety_veto", None) is not None
            for s in snakes
        ):
            raise RuntimeError("all_b8 roster is not entirely unwrapped served B8")
        return
    if roster in ("all_frp3", "all_champion"):
        if any(s.policy is not session.policy for s in snakes):
            raise RuntimeError("homogeneous roster policy binding changed")
        wrapped = [int(s.id) for s in snakes if getattr(s, "safety_veto", None) is not None]
        hero_veto = getattr(snakes[0], "safety_veto", None)
        if (
            wrapped != [int(session.hero_id)]
            or getattr(hero_veto, "method", None) != "free-space-veto/v8-space-and-head(lambda=8.0)"
        ):
            raise RuntimeError(f"homogeneous v8 scope differs from serving semantics: {wrapped!r}")
        return
    if donor is None or (donor.served_checkpoint or {}).get("name") != "frp3-s12":
        raise RuntimeError("mixed roster has no exact FRP3 donor")
    if (
        snakes[0].policy is not session.policy
        or getattr(snakes[0], "safety_veto", None) is not None
    ):
        raise RuntimeError("mixed roster B8 hero changed")
    for snake in snakes[1:]:
        veto = getattr(snake, "safety_veto", None)
        if (
            snake.policy is not donor.policy
            or getattr(veto, "method", None) != "free-space-veto/v8-space-and-head(lambda=8.0)"
            or getattr(veto, "head_avoidance", None) is not True
            or getattr(veto, "reference_lambda", None) is not None
        ):
            raise RuntimeError("mixed opponent is not exact FRP3 plus default strict v8")


class _DecisionObserver:
    """Count real AI updates, using length/action after each real dispatch."""

    def __init__(self, snakes: list[Any], counters: dict[int, dict[str, Any]]) -> None:
        self.snakes, self.counters, self.original = snakes, counters, {}

    def __enter__(self) -> "_DecisionObserver":
        for snake in self.snakes:
            self.original[id(snake)] = snake.update

            def observed(*args: Any, _snake: Any = snake, **kwargs: Any) -> Any:
                row = self.counters[int(_snake.id)]
                row["decisions"] += 1
                row["length_histogram"][int(_snake.length)] += 1
                result = self.original[id(_snake)](*args, **kwargs)
                action = getattr(_snake, "_pre_collision_action", None)
                if action is not None and int(action) >= 3 and bool(_snake.is_boosting):
                    row["boost_decisions"] += 1
                return result

            snake.update = observed
        return self

    def __exit__(self, *_: Any) -> None:
        for snake in self.snakes:
            snake.update = self.original[id(snake)]


class _MixedDispatchObserver:
    """Fail if B8's positional ego2s queue serves anything except its hero."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.calls: list[int] = []
        self.original = getattr(session.policy, "_next_dispatch_q", None)

    def __enter__(self) -> "_MixedDispatchObserver":
        if not callable(self.original):
            raise RuntimeError("mixed B8 hero lacks ego2s dispatch hook")

        def observed(state: Any) -> Any:
            queue = list(getattr(self.session.policy, "_dispatch", []))
            snake_id = int(queue[0]) if queue else -1
            self.calls.append(snake_id)
            if snake_id != int(self.session.hero_id):
                raise RuntimeError(f"mixed B8 dispatch shifted to {snake_id}")
            return self.original(state)

        self.session.policy._next_dispatch_q = observed
        return self

    def __exit__(self, *_: Any) -> None:
        self.session.policy._next_dispatch_q = self.original


def _counter() -> dict[str, Any]:
    return {
        "decisions": 0,
        "deaths": 0,
        "length_histogram": Counter(),
        "mass_sum": 0,
        "boost_decisions": 0,
        "food_eaten": 0,
        "kills_made": 0,
        "kills_taken": 0,
        "death_causes": Counter(),
    }


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "length_histogram": {str(k): int(v) for k, v in sorted(row["length_histogram"].items())},
        "death_causes": {str(k): int(v) for k, v in sorted(row["death_causes"].items())},
    }


def _install_mixed_opponents(session: Any, donor: Any) -> dict[str, Any]:
    """Put FRP3+v8 on slots 1..N, retaining B8 on Watch hero slot zero."""
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto
    from web.backend.safety_veto_serving import V8_LAMBDA

    donor_veto = donor.safety_veto_state() or {}
    if donor_veto.get("active") is not True:
        raise RuntimeError(f"FRP3 v8 was not installed: {donor_veto!r}")
    opponents = list(session.game.snakes[1:])
    for snake in opponents:
        snake.policy = donor.policy
        snake.ai = donor.policy
        # The product Watch selector wraps its hero only.  This explicit,
        # harness-local attachment realizes the requested *opponents + v8* mix.
        veto = install_space_and_head_veto(snake, V8_LAMBDA)
        if veto.method != "free-space-veto/v8-space-and-head(lambda=8.0)":
            raise RuntimeError("mixed opponent did not receive the strict v8 wrapper")
    attach_game = getattr(donor.policy, "attach_game", None)
    if callable(attach_game):
        attach_game(session.game)
    return {
        "b8": _identity(session),
        "frp3_v8": _identity(donor),
        "opponent_ids": [int(s.id) for s in opponents],
    }


def run_watch(roster: str, seed: int, horizon: int = 5000) -> dict[str, Any]:
    """Run H frames through ``GameSession.step`` plus ``snapshot`` and return raw counters.

    ``mass_sum`` is the post-step sum of live logical lengths (dead snakes add
    zero), so analysis can report both total arena mean and per-snake mean.
    Length histograms are exact integer logical-length samples taken immediately
    before actual AI decisions; quantiles are intentionally left to the analyzer.
    """
    if roster not in ROSTERS:
        raise ValueError(f"roster must be one of {ROSTERS}, got {roster!r}")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    checkpoint, veto = {
        # No override at all: this is the release's actual default selection.
        "all_b8": (None, None),
        "all_frp3": ("frp3-s12", None),
        "all_champion": ("champion", None),
        "mixed": (None, None),
    }[roster]
    # Build the donor before the evaluated B8 world.  The final construction
    # reapplies the seed immediately before its GameSession, so a donor cannot
    # perturb paired initial world RNG or food placement.
    donor = _new_session("frp3-s12", int(seed), veto=None) if roster == "mixed" else None
    session = _new_session(checkpoint, int(seed), veto=veto)
    initial = _identity(session)
    mixed = _install_mixed_opponents(session, donor) if donor is not None else None
    snakes = list(session.game.snakes)
    hero_id = int(session.hero_id)
    _assert_roster(session, roster, donor)
    initial_world = _initial_world(session)
    counters = {int(snake.id): _counter() for snake in snakes}
    dispatch = _MixedDispatchObserver(session) if roster == "mixed" else contextlib.nullcontext()
    with _DecisionObserver(snakes, counters), dispatch as mixed_dispatch:
        for _ in range(int(horizon)):
            frame = int(session.game.frame)
            session.step()
            session.snapshot()
            if int(session.game.frame) != frame + 1:
                raise RuntimeError("GameSession Watch step did not advance one frame")
            for snake in snakes:
                row = counters[int(snake.id)]
                sid = int(snake.id)
                row["food_eaten"] += int(bool(session.game.frame_ate_food.get(sid, False)))
                cause = session.game.frame_death_causes.get(sid)
                if cause is not None:
                    row["deaths"] += 1
                    row["death_causes"][str(cause)] += 1
                row["kills_made"] += len(session.game.frame_kills.get(sid, []))
                row["kills_taken"] += sum(
                    sid in victims for victims in session.game.frame_kills.values()
                )
                row["mass_sum"] += int(snake.length) if snake.is_alive else 0
            if session.last_error:
                raise RuntimeError(f"Watch session error: {session.last_error}")
    raw = {str(sid): _plain(value) for sid, value in sorted(counters.items())}
    arena = _counter()
    for value in counters.values():
        for key in (
            "decisions",
            "deaths",
            "mass_sum",
            "boost_decisions",
            "food_eaten",
            "kills_made",
            "kills_taken",
        ):
            arena[key] += value[key]
        arena["length_histogram"].update(value["length_histogram"])
        arena["death_causes"].update(value["death_causes"])
    return {
        "schema_version": "t13-product-roster-episode/v1",
        "roster": roster,
        "seed": int(seed),
        "horizon": int(horizon),
        "frames_stepped": int(horizon),
        "hero_id": hero_id,
        "served": initial,
        "initial_world": initial_world,
        "snake_identities": _snake_identities(snakes),
        "mixed_policy_identities": mixed,
        "mixed_b8_dispatch_ids": [] if mixed_dispatch is None else mixed_dispatch.calls,
        "per_snake": raw,
        "hero": raw[str(hero_id)],
        "arena": _plain(arena),
        "final_veto": _identity(session)["veto"],
    }
