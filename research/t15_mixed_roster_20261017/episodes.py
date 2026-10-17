"""Live, default-constructor episode evidence for the T15 mixed Watch roster."""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from research.apex_veto_serving_20261001.serving_run import set_world_seed

CHECKPOINT_NAME = "b8-s3"
WATCH_ROSTER = "mixed"
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"
_ENV = (
    "SNAKE_SERVE_CHECKPOINT",
    "SNAKE_SERVE_WATCH_ROSTER",
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
    "SNAKE_SERVE_VETO_VARIANT",
)


@contextlib.contextmanager
def mixed_environment() -> Iterator[None]:
    """Select T15's opt-in Watch roster without changing a process default."""
    saved = {key: os.environ.pop(key) for key in _ENV if key in os.environ}
    try:
        # An unset checkpoint exercises the release registry's B8 selection.
        os.environ["SNAKE_SERVE_WATCH_ROSTER"] = WATCH_ROSTER
        yield
    finally:
        for key in _ENV:
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


def new_session(seed: int, num_snakes: int = 6) -> Any:
    """Construct the real mixed Watch session with only a local count override.

    ``GameSession()`` remains the only session constructor.  The patched loader
    exists just for this constructor call, so neither configuration nor serving
    defaults leak from a qualification episode.
    """
    if not 2 <= int(num_snakes) <= 12:
        raise ValueError("num_snakes must be in 2..12")
    from web.backend.session import GameSession

    set_world_seed(int(seed))
    original_loader = __import__("web.backend.session", fromlist=["load_config"]).load_config

    def local_count(path: str) -> Any:
        config = original_loader(path)
        return replace(config, game=replace(config.game, num_snakes=int(num_snakes)))

    with (
        mixed_environment(),
        patch("web.backend.session.load_config", side_effect=local_count),
        patch("web.backend.session.served_resolution"),
        patch("web.backend.session.watch_tick"),
    ):
        session = GameSession()
    # Qualification must not write release-monitor samples.  This changes only
    # telemetry side effects after construction, never policy or game semantics.
    session._release_monitoring = False
    _assert_mixed(session, int(num_snakes))
    return session


def _veto_state(session: Any) -> dict[str, Any]:
    return dict(session.safety_veto_state() or {})


def roster_identity(session: Any) -> dict[str, Any]:
    """Report actual, per-slot policy and wrapper identity from the live game."""
    return {
        "watch_roster": getattr(session, "watch_roster", None),
        "watch_roster_reason": getattr(session, "watch_roster_reason", None),
        "hero_id": int(session.hero_id),
        "served": dict(session.served_checkpoint or {}),
        "hero_checkpoint_sha256": _sha(getattr(session, "checkpoint_path", None)),
        "opponent_checkpoint": dict(getattr(session, "watch_opponent_checkpoint", None) or {}),
        "veto": _veto_state(session),
        "snakes": [
            {
                "id": int(snake.id),
                "policy_class": type(getattr(snake, "policy", None)).__name__,
                "policy_is_hero": getattr(snake, "policy", None) is session.policy,
                "policy_is_opponent": getattr(snake, "policy", None)
                is getattr(session, "watch_opponent_policy", None),
                "veto_method": getattr(getattr(snake, "safety_veto", None), "method", None),
                "veto_head_avoidance": getattr(
                    getattr(snake, "safety_veto", None), "head_avoidance", None
                ),
                "veto_reference_lambda": getattr(
                    getattr(snake, "safety_veto", None), "reference_lambda", None
                ),
            }
            for snake in session.game.snakes
        ],
    }


def _assert_mixed(session: Any, num_snakes: int | None = None) -> None:
    from web.backend.safety_veto_serving import V8_STRICT_RECEIPT_SOURCE_SHA256S
    from web.backend.served_checkpoint import B8_S3_SHA256, FRP3_S12_SHA256

    choice = dict(session.served_checkpoint or {})
    if choice.get("name") != CHECKPOINT_NAME or choice.get("reason") or choice.get("pin_problems"):
        raise RuntimeError(f"mixed hero was not an exact B8 registry selection: {choice!r}")
    if getattr(session, "watch_roster", None) != WATCH_ROSTER:
        raise RuntimeError(
            f"mixed roster did not activate: {getattr(session, 'watch_roster_reason', None)!r}"
        )
    snakes = list(session.game.snakes)
    if num_snakes is not None and len(snakes) != num_snakes:
        raise RuntimeError(f"mixed roster count {len(snakes)} != requested {num_snakes}")
    if not snakes or int(snakes[0].id) != int(session.hero_id):
        raise RuntimeError("mixed Watch hero must remain slot zero")
    hero, opponents = snakes[0], snakes[1:]
    opponent_policy = getattr(session, "watch_opponent_policy", None)
    opponent_choice = dict(getattr(session, "watch_opponent_checkpoint", None) or {})
    if hero.policy is not session.policy or hero.safety_veto is not None:
        raise RuntimeError("mixed hero must be the unwrapped B8 policy")
    if _sha(getattr(session, "checkpoint_path", None)) != B8_S3_SHA256:
        raise RuntimeError("mixed hero bytes are not the registered B8 SHA")
    if opponent_choice.get("name") != "frp3-s12" or opponent_choice.get("reason"):
        raise RuntimeError(f"mixed opponents were not exact FRP3: {opponent_choice!r}")
    if opponent_choice.get("sha256") != FRP3_S12_SHA256:
        raise RuntimeError("mixed opponent selection does not report the registered FRP3 SHA")
    if opponent_policy is None or any(s.policy is not opponent_policy for s in opponents):
        raise RuntimeError("mixed opponents do not share the pinned FRP3 policy")
    if any(
        getattr(s.safety_veto, "method", None) != V8_METHOD
        or getattr(s.safety_veto, "head_avoidance", None) is not True
        or getattr(s.safety_veto, "reference_lambda", None) is not None
        for s in opponents
    ):
        raise RuntimeError("mixed opponents are not exact v8 wrappers")
    state = _veto_state(session)
    expected_ids = [int(s.id) for s in opponents]
    if (
        state.get("active") is not True
        or state.get("scope") != "watch_opponents"
        or list(state.get("wrapped_snake_ids", [])) != expected_ids
        or state.get("checkpoint_sha256") != FRP3_S12_SHA256
        or state.get("strict_receipt_checkpoint_match") is not True
        or state.get("strict_receipt_wrapper_match") is not True
    ):
        raise RuntimeError(f"mixed veto scope differs from opponents-only contract: {state!r}")
    wrapper = dict(state.get("wrapper") or {})
    expected_source = V8_STRICT_RECEIPT_SOURCE_SHA256S["src/evaluation/safety_veto_v8.py"]
    if wrapper.get("source_sha256") != expected_source:
        raise RuntimeError("mixed v8 wrapper source differs from the strict receipt")
    counters = dict(state.get("counters") or {})
    if set(counters) != {str(sid) for sid in expected_ids} or str(hero.id) in counters:
        raise RuntimeError(f"mixed veto counters escape opponent scope: {counters!r}")


class _DecisionEvidence:
    def __init__(self, snake: Any) -> None:
        self.snake, self.lengths, self._original = snake, [], snake.update

    def __enter__(self) -> "_DecisionEvidence":
        def observed(*args: Any, **kwargs: Any) -> Any:
            self.lengths.append(int(self.snake.length))
            return self._original(*args, **kwargs)

        self.snake.update = observed
        return self

    def __exit__(self, *_: Any) -> None:
        self.snake.update = self._original


class _HeroDispatchEvidence:
    """Assert B8's queue dispatch remains confined to its hero row."""

    def __init__(self, session: Any) -> None:
        self.session, self.ids, self.observation_ids, self.observation_alive = session, [], [], []
        self._original = getattr(session.policy, "_next_dispatch_q", None)
        self._prepare = getattr(session.policy, "prepare_frame", None)

    def __enter__(self) -> "_HeroDispatchEvidence":
        if not callable(self._original):
            raise RuntimeError("mixed B8 hero has no ego2s dispatch hook")

        def observed(state: Any) -> Any:
            queue = list(getattr(self.session.policy, "_dispatch", []))
            snake_id = int(queue[0]) if queue else -1
            self.ids.append(snake_id)
            if snake_id != int(self.session.hero_id):
                raise RuntimeError(f"B8 hero dispatch reached non-hero {snake_id}")
            return self._original(state)

        self.session.policy._next_dispatch_q = observed
        if callable(self._prepare):

            def observed_prepare() -> Any:
                result = self._prepare()
                obs = getattr(self.session.policy, "_obs", None) or {}
                rows = list(obs.get("rows", []))
                snakes = list(self.session.game.snakes)
                ids = [int(snakes[int(row)].id) for row in rows if 0 <= int(row) < len(snakes)]
                hero = next(snake for snake in snakes if int(snake.id) == int(self.session.hero_id))
                alive = bool(hero.is_alive)
                self.observation_ids.append(ids)
                self.observation_alive.append(alive)
                expected = [int(self.session.hero_id)] if alive else []
                if ids != expected:
                    raise RuntimeError(
                        "B8 mixed observation rows differ from hero liveness: "
                        f"{ids!r}, alive={alive}"
                    )
                return result

            self.session.policy.prepare_frame = observed_prepare
        return self

    def __exit__(self, *_: Any) -> None:
        self.session.policy._next_dispatch_q = self._original
        if callable(self._prepare):
            self.session.policy.prepare_frame = self._prepare


def _quantiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "p50": float(ordered[round((len(ordered) - 1) * 0.5)]),
        "p95": float(ordered[round((len(ordered) - 1) * 0.95)]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def run_watch(seed: int, horizon: int = 5000, num_snakes: int = 6) -> dict[str, Any]:
    """Run exact ``step(); snapshot()`` Watch frames against the mixed roster."""
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    session = new_session(seed, num_snakes)
    hero = next(s for s in session.game.snakes if int(s.id) == int(session.hero_id))
    deaths, alive_frames = 0, 0
    with _DecisionEvidence(hero) as decisions, _HeroDispatchEvidence(session) as dispatch:
        for _ in range(int(horizon)):
            before = int(session.game.frame)
            session.step()
            session.snapshot()
            if int(session.game.frame) != before + 1:
                raise RuntimeError("mixed Watch step did not advance exactly one frame")
            if session.last_error:
                raise RuntimeError(f"mixed Watch error: {session.last_error}")
            _assert_mixed(session, int(num_snakes))
            deaths += int(int(hero.id) in session.game.frame_death_causes)
            alive_frames += int(bool(hero.is_alive))
    return {
        "schema_version": "t15-mixed-watch-episode/v1",
        "status": "complete",
        "seed": int(seed),
        "horizon": int(horizon),
        "frames_stepped": int(horizon),
        "identity_checked_frames": int(horizon),
        "no_errors": True,
        "roster": roster_identity(session),
        "hero": {
            "id": int(hero.id),
            "decisions": len(decisions.lengths),
            "deaths": deaths,
            "alive_frames": alive_frames,
            "final_length": int(hero.length),
            "decision_length_histogram": dict(sorted(Counter(decisions.lengths).items())),
            "decision_length_quantiles": _quantiles(decisions.lengths),
        },
        "dispatch": {
            "ids": dispatch.ids,
            "hero_only": all(i == int(hero.id) for i in dispatch.ids),
            "observation_ids": dispatch.observation_ids,
            "observation_alive": dispatch.observation_alive,
            "observations_hero_only": all(
                ids == ([int(hero.id)] if alive else [])
                for ids, alive in zip(dispatch.observation_ids, dispatch.observation_alive)
            ),
        },
        "final_veto": _veto_state(session),
    }
