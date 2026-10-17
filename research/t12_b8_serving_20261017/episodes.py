"""Real-session episode evidence for the T12 B8 serving qualification.

This module deliberately constructs :class:`GameSession` without a checkpoint.
The registry is therefore the sole selector, under the explicit b8-s3 serving
environment used by the qualification.  It is intentionally usable for small
smokes; the orchestrator owns protocol pinning and non-smoke episode counts.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# One qualification process occupies one CPU slot.  These must precede torch/numpy
# imports transitively reached by GameSession.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")

from research.apex_veto_serving_20261001.serving_run import (  # noqa: E402
    ScriptedHuman,
    set_world_seed,
)

CHECKPOINT_NAME = "b8-s3"
_SERVING_ENV = (
    "SNAKE_SERVE_CHECKPOINT",
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
    "SNAKE_SERVE_VETO_VARIANT",
)
_CONN_ID = "t12-b8-qualification"


@contextlib.contextmanager
def serving_env() -> Iterator[None]:
    """Select B8 exactly as an operator does and remove every veto override."""
    saved = {key: os.environ.pop(key) for key in _SERVING_ENV if key in os.environ}
    try:
        os.environ["SNAKE_SERVE_CHECKPOINT"] = CHECKPOINT_NAME
        yield
    finally:
        for key in _SERVING_ENV:
            os.environ.pop(key, None)
        os.environ.update(saved)


def _sha256(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _control(session: Any, action: str, value: Any = None) -> None:
    from web.backend.app import _apply_control

    session.last_error = None
    reply = _apply_control(session, {"type": "control", "action": action, "value": value}, _CONN_ID)
    if reply is not None or session.last_error is not None:
        raise RuntimeError(
            f"control {action!r} refused: reply={reply!r} last_error={session.last_error!r}"
        )


def _observe_train_refusal(session: Any) -> Dict[str, Any]:
    """Record the expected forward-only Train refusal without treating it as success."""
    from web.backend.app import _apply_control

    session.last_error = None
    reply = _apply_control(
        session,
        {"type": "control", "action": "set_mode", "value": "train"},
        _CONN_ID,
    )
    result = {
        "mode": session.mode,
        "last_error": session.last_error,
        "reply": reply,
        "policy_training": bool(getattr(session.policy, "training", True)),
        "epsilon": float(getattr(session.policy, "epsilon", -1.0)),
    }
    if result["mode"] != "watch" or not result["last_error"]:
        raise RuntimeError(f"B8 Train was not refused to Watch: {result!r}")
    if result["policy_training"] or result["epsilon"] != 0.0:
        raise RuntimeError(f"Train refusal left an active learner: {result!r}")
    return result


def _new_session(seed: int) -> Any:
    """Build only the default constructor under the B8 selection environment."""
    from web.backend.session import GameSession

    set_world_seed(seed)
    with serving_env():
        session = GameSession()
    selected = (session.served_checkpoint or {}).get("name")
    if selected != CHECKPOINT_NAME:
        raise RuntimeError(f"B8 was not selected: {selected!r}; {session.served_checkpoint!r}")
    return session


def _served_identity(session: Any) -> Dict[str, Any]:
    checkpoint = dict(session.served_checkpoint or {})
    state = session.safety_veto_state() or {}
    return {
        "registry": checkpoint,
        "checkpoint_path": session.checkpoint_path,
        "checkpoint_sha256": _sha256(session.checkpoint_path),
        "config_path": session.config_path,
        "config_sha256": _sha256(session.config_path),
        "obs_spec": str(session.obs_spec),
        "policy_class": type(session.policy).__name__,
        "mode": session.mode,
        "snake_ids": [int(s.id) for s in session.game.snakes],
        "veto": {
            "active": state.get("active"),
            "reason": state.get("reason"),
            "wrapped_snake_ids": state.get("wrapped_snake_ids", []),
            "counters": state.get("counters", {}),
        },
    }


def _quantiles(values: list[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return float(ordered[round((len(ordered) - 1) * q)])

    return {
        "p50": pick(0.50),
        "p95": pick(0.95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


class _DispatchEvidence:
    """Observe actual ego policy dispatches without changing decision semantics."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.ids: list[int] = []
        self.observation_ids: list[list[int]] = []
        self._original = getattr(session.policy, "_next_dispatch_q", None)
        self._prepare = getattr(session.policy, "prepare_frame", None)

    def __enter__(self) -> "_DispatchEvidence":
        if callable(self._original):

            def observed(state: Any) -> Any:
                queue = list(getattr(self.session.policy, "_dispatch", []))
                self.ids.append(int(queue[0]) if queue else -1)
                return self._original(state)

            self.session.policy._next_dispatch_q = observed
        if callable(self._prepare):

            def observed_prepare() -> Any:
                result = self._prepare()
                obs = getattr(self.session.policy, "_obs", None)
                rows = [] if not obs else list(obs.get("rows", []))
                snakes = list(self.session.game.snakes)
                self.observation_ids.append(
                    [int(snakes[int(row)].id) for row in rows if 0 <= int(row) < len(snakes)]
                )
                return result

            self.session.policy.prepare_frame = observed_prepare
        return self

    def __exit__(self, *_: Any) -> None:
        if callable(self._original):
            self.session.policy._next_dispatch_q = self._original
        if callable(self._prepare):
            self.session.policy.prepare_frame = self._prepare


class _HeroDecisionEvidence:
    """Capture the logical length at the actual AI action call.

    Capturing outside ``GameState.update`` is subtly wrong: continuous Watch can
    respawn a snake at the start of a frame, and a snake may die later in that
    same frame.  Wrapping only this instance's ``update`` observes exactly the
    pre-decision state after the respawn phase.
    """

    def __init__(self, hero: Any) -> None:
        self.hero = hero
        self.lengths: list[int] = []
        self._original = hero.update

    def __enter__(self) -> "_HeroDecisionEvidence":
        def observed(*args: Any, **kwargs: Any) -> Any:
            self.lengths.append(int(getattr(self.hero, "length", len(self.hero.segments))))
            return self._original(*args, **kwargs)

        self.hero.update = observed
        return self

    def __exit__(self, *_: Any) -> None:
        self.hero.update = self._original


def _step_and_snapshot(session: Any) -> None:
    """Mirror Hub.engine_loop ordering; snapshot is part of the serving path."""
    session.step()
    session.snapshot()


def run_watch(seed: int, horizon: int = 5000) -> dict:
    """Run a B8 Watch episode and return raw monitoring evidence.

    Errors are recorded and then raised so an orchestrator cannot accidentally
    turn a failed episode into a passing aggregate.
    """
    record: Dict[str, Any] = {
        "kind": "watch",
        "seed": int(seed),
        "horizon": int(horizon),
        "frames_stepped": 0,
        "status": "error",
        "error": None,
    }
    started = time.monotonic()
    try:
        session = _new_session(seed)
        identity = _served_identity(session)
        if identity["policy_class"] != "Ego2sServingPolicy":
            raise RuntimeError(f"unexpected served policy: {identity['policy_class']}")
        if identity["veto"]["active"] is not False or identity["veto"]["wrapped_snake_ids"]:
            raise RuntimeError(f"B8 must be unwrapped: {identity['veto']!r}")
        if any(getattr(s, "policy", None) is not session.policy for s in session.game.snakes):
            raise RuntimeError("Watch roster is not entirely driven by the selected B8 policy")
        hero = next(s for s in session.game.snakes if int(s.id) == int(session.hero_id))
        deaths = 0
        alive_frames = 0
        with _DispatchEvidence(session) as dispatch, _HeroDecisionEvidence(hero) as hero_actions:
            for _ in range(int(horizon)):
                before_frame = int(session.game.frame)
                _step_and_snapshot(session)
                record["frames_stepped"] += 1
                # The authoritative collision phase stores terminal events here.
                # This survives an immediate later respawn, unlike alive-edge counting.
                deaths += int(int(hero.id) in getattr(session.game, "frame_death_causes", {}))
                alive_frames += int(bool(hero.is_alive))
                if int(session.game.frame) == before_frame:
                    raise RuntimeError("Watch step did not advance the game")
        ai_ids = {int(s.id) for s in session.game.snakes}
        record.update(
            {
                "status": "complete",
                "served": identity,
                "hero": {
                    "id": int(hero.id),
                    "decisions": len(hero_actions.lengths),
                    "deaths": deaths,
                    "alive_frames": alive_frames,
                    "final_length": int(hero.length),
                    "decision_length_histogram": dict(
                        sorted(Counter(hero_actions.lengths).items())
                    ),
                    "decision_length_quantiles": _quantiles(hero_actions.lengths),
                },
                "dispatch": {
                    "calls": len(dispatch.ids),
                    "snake_ids": dispatch.ids,
                    "all_ai": all(sid in ai_ids for sid in dispatch.ids if sid >= 0),
                    "unknown_or_empty_rows": sum(sid < 0 for sid in dispatch.ids),
                    "observation_snake_ids": dispatch.observation_ids,
                },
                "roster": {
                    "all_snakes_share_selected_policy": True,
                    "all_snakes_b8": True,
                    "snake_count": len(session.game.snakes),
                },
                "last_error": session.last_error,
                "final_veto": _served_identity(session)["veto"],
            }
        )
        if session.last_error:
            raise RuntimeError(f"Watch last_error: {session.last_error}")
        if record["final_veto"]["active"] is not False or record["final_veto"]["wrapped_snake_ids"]:
            raise RuntimeError(f"Watch acquired a veto wrapper: {record['final_veto']!r}")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_seconds"] = time.monotonic() - started
        raise RuntimeError(record["error"]) from exc
    record["wall_seconds"] = time.monotonic() - started
    return record


def run_play(seed: int, horizon: int = 5000) -> dict:
    """Run one stand-in-human B8 Play episode through the public control path."""
    record: Dict[str, Any] = {
        "kind": "play",
        "seed": int(seed),
        "horizon": int(horizon),
        "frames_stepped": 0,
        "status": "error",
        "error": None,
    }
    started = time.monotonic()
    try:
        session = _new_session(seed)
        _control(session, "set_mode", "play")
        # The rebuild preserves the registry-selected checkpoint but records a fresh session state.
        identity = _served_identity(session)
        # The world that matters begins at new_game, after the Play rebuild.
        set_world_seed(seed)
        _control(session, "new_game")
        human = session._find_human()
        if human is None:
            raise RuntimeError("Play session has no human")
        opponents = [s for s in session.game.snakes if int(s.id) != int(human.id)]
        if any(getattr(s, "safety_veto", None) is not None for s in opponents):
            raise RuntimeError("B8 Play opponents must be unwrapped")
        if any(getattr(s, "policy", None) is not session.policy for s in opponents):
            raise RuntimeError("Play opponents do not all share the selected B8 policy")
        initial_game_frame = int(session.game.frame)
        stand_in, inputs = ScriptedHuman(seed), 0
        while not session.run_started:
            direction = stand_in.choose(human, session.game, must_turn=True)
            if direction is None or inputs >= 8:
                raise RuntimeError("stand-in could not start Play")
            _control(session, "human_input", direction)
            inputs += 1
        with _DispatchEvidence(session) as dispatch:
            while record["frames_stepped"] < int(horizon) and not session.run_over:
                direction = stand_in.choose(human, session.game, must_turn=False)
                if direction is not None:
                    _control(session, "human_input", direction)
                    inputs += 1
                before_frame = int(session.game.frame)
                _step_and_snapshot(session)
                record["frames_stepped"] += 1
                if int(session.game.frame) == before_frame:
                    raise RuntimeError("Play step did not advance the game")
                if session.last_error:
                    raise RuntimeError(f"Play last_error after step: {session.last_error}")
        ai_ids = {int(s.id) for s in opponents}
        from src.game.ai_snake import AISnake

        terminal = "human_death" if session.run_over else "horizon"
        record.update(
            {
                "status": "complete",
                "served": identity,
                "play": {
                    "human_id": int(human.id),
                    "inputs_sent": inputs,
                    "run_over": bool(session.run_over),
                    "run_frames": int(session.run_frames),
                    "last_run": session.last_run,
                    "terminal": terminal,
                    "human_alive": bool(human.is_alive),
                    "human_final_length": int(human.length),
                    "human_type": type(human).__name__,
                    "opponent_ids": sorted(ai_ids),
                    "opponents_unwrapped": True,
                    "human_policy": getattr(human, "policy", None),
                    "human_is_ai": isinstance(human, AISnake),
                },
                "dispatch": {
                    "calls": len(dispatch.ids),
                    "snake_ids": dispatch.ids,
                    "human_dispatched": int(human.id) in dispatch.ids,
                    "all_opponents": all(sid in ai_ids for sid in dispatch.ids if sid >= 0),
                    "unknown_or_empty_rows": sum(sid < 0 for sid in dispatch.ids),
                    "observation_snake_ids": dispatch.observation_ids,
                    "human_observed": any(int(human.id) in ids for ids in dispatch.observation_ids),
                },
                "last_error": session.last_error,
                "game_frame": int(session.game.frame),
                "game_frame_delta": int(session.game.frame) - initial_game_frame,
            }
        )
        if record["dispatch"]["human_dispatched"] or record["dispatch"]["human_observed"]:
            raise RuntimeError(f"Play human entered an AI policy row: {record['dispatch']!r}")
        if record["play"]["human_policy"] is not None:
            raise RuntimeError("Play human was attached to an AI policy")
        if not record["frames_stepped"]:
            raise RuntimeError("Play run did not advance")
        if record["game_frame_delta"] != record["frames_stepped"]:
            raise RuntimeError("Play game-frame delta disagrees with stepped frames")
        if session.run_over and bool(human.is_alive):
            raise RuntimeError("Play terminal run left the human alive")
        if not session.run_over and record["frames_stepped"] != int(horizon):
            raise RuntimeError("Play ended before the registered horizon without human death")
        if not record["dispatch"]["all_opponents"] or record["dispatch"]["unknown_or_empty_rows"]:
            raise RuntimeError(f"Play dispatch evidence is incomplete: {record['dispatch']!r}")
        if session.last_error:
            raise RuntimeError(f"Play last_error: {session.last_error}")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_seconds"] = time.monotonic() - started
        raise RuntimeError(record["error"]) from exc
    record["wall_seconds"] = time.monotonic() - started
    return record


def run_controls(seed: int) -> dict:
    """Exercise Play roster, Train refusal, and HTTP score persistence in isolation."""
    from fastapi.testclient import TestClient
    from src.data.score_store import ScoreStore
    from web.backend import app

    session = _new_session(seed)
    output: Dict[str, Any] = {"seed": int(seed), "status": "error", "error": None}
    old_session, old_store = app.hub.session, app.hub.store
    store: Optional[ScoreStore] = None
    try:
        watch_train = _observe_train_refusal(session)

        # Exercise adjustable Play rosters only after the Watch-mode Train refusal.
        _control(session, "set_mode", "play")
        roster: Dict[str, int] = {}
        for count in (1, 5, 11):
            _control(session, "set_play_opponents", count)
            roster[str(count)] = len(session.game.snakes) - 1
        play_train = _observe_train_refusal(session)
        if play_train["last_error"] != watch_train["last_error"]:
            raise RuntimeError("Watch and Play Train refusals disagree")

        # Finalize a real Play lifecycle, then submit that server-authored result.
        _control(session, "set_mode", "play")
        set_world_seed(seed)
        _control(session, "new_game")
        human = session._find_human()
        if human is None:
            raise RuntimeError("score-path Play session has no human")
        starter = ScriptedHuman(seed)
        direction = starter.choose(human, session.game, must_turn=True)
        if direction is None:
            raise RuntimeError("score-path stand-in could not select a starting direction")
        _control(session, "human_input", direction)
        # ``die`` is the game object's terminal API.  The following session step
        # invokes _step_play and _finalize_run, rather than manufacturing last_run.
        human.die()
        session.step()
        session.snapshot()
        if not session.run_over or session.last_run is None:
            raise RuntimeError("real Play death did not produce a finalized score")
        if session.last_run.get("served_checkpoint") != CHECKPOINT_NAME:
            raise RuntimeError(f"scored run lost served checkpoint: {session.last_run!r}")
        with tempfile.TemporaryDirectory(prefix="t12-b8-scores-") as temp:
            store = ScoreStore(str(Path(temp) / "scores.db"))
            app.hub.session, app.hub.store = session, store
            client = TestClient(app.app)
            try:
                submitted = client.post("/api/scores", json={"name": "t12-b8"})
                player = client.get("/api/players/t12-b8")
                recent = client.get("/api/scores/recent?limit=1")
            finally:
                client.close()
            score_evidence = {
                "submit_status": submitted.status_code,
                "submit": submitted.json(),
                "player_status": player.status_code,
                "player": player.json(),
                "recent_status": recent.status_code,
                "recent": recent.json(),
                "server_last_run": session.last_run,
            }
            store.close()
            store = None
        output.update(
            {
                "status": "complete",
                "play_opponents": roster,
                "train_refusal": play_train,
                "train_refusals": [watch_train, play_train],
                "score_path": score_evidence,
                "served": _served_identity(session),
            }
        )
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(output["error"]) from exc
    finally:
        if store is not None:
            store.close()
        app.hub.session, app.hub.store = old_session, old_store
    return output
