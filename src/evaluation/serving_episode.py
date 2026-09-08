"""Bounded, source-bound evidence from real raster serving sessions.

This module deliberately drives :class:`web.backend.session.GameSession` rather
than reproducing its game loop.  It produces one receipt per fresh process for
the strict serving-readiness schedule; it does not submit a score or mutate the
web application's shared session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from src.core.runtime_contract import canonical_digest
from src.core.seeding import SeedContext, initialize_run_seed
from src.model.obs_spec import RASTER31V3
from web.backend.session import MODE_PLAY, MODE_WATCH, GameSession

EPISODE_SCHEMA_VERSION = "strict-serving-episode/v3"
SEED_API = "src.core.seeding.initialize_run_seed/v1"
_MODES = (MODE_WATCH, MODE_PLAY)
_SHA256_LENGTH = 64


class ServingEpisodeError(RuntimeError):
    """Failure annotated with the last producer stage that actually ran."""

    def __init__(self, stage: str, last_observed_frame: int | None, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.last_observed_frame = last_observed_frame
        self.cause = cause


@dataclass(frozen=True)
class ServingEpisodeSpec:
    """An immutable instruction for one serving-path compatibility episode."""

    episode_id: str
    mode: Literal["watch", "play"]
    serving_seed: int
    checkpoint_path: Path
    expected_candidate_sha256: str
    profile_digest: str
    source_closure_sha256: str
    frame_limit: int = 5000

    def __post_init__(self) -> None:
        if not self.episode_id or not isinstance(self.episode_id, str):
            raise ValueError("episode_id must be a non-empty string")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        if isinstance(self.serving_seed, bool) or not isinstance(self.serving_seed, int):
            raise ValueError("serving_seed must be an unsigned 64-bit integer")
        if not 0 <= self.serving_seed < 2**64:
            raise ValueError("serving_seed must be an unsigned 64-bit integer")
        if isinstance(self.frame_limit, bool) or not isinstance(self.frame_limit, int):
            raise ValueError("frame_limit must be a positive integer")
        if self.frame_limit <= 0:
            raise ValueError("frame_limit must be a positive integer")
        for name in ("expected_candidate_sha256", "profile_digest", "source_closure_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from exc


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layout(session: GameSession) -> dict[str, Any]:
    """Capture observable frame-zero state without claiming a full world proof."""
    snakes = []
    for slot, snake in enumerate(session.game.snakes):
        snakes.append(
            {
                "slot": slot,
                "snake_id": int(snake.id),
                "controller_class": type(snake).__name__,
                "head": list(snake.head),
                "segments": [list(segment) for segment in snake.segments],
                "direction": list(snake.direction),
                "length": int(snake.length),
                "alive": bool(snake.is_alive),
                # Legacy AI snakes use absence as the normal respawn policy;
                # the serving terminal hero/human receives an explicit False.
                "auto_respawn": bool(getattr(snake, "auto_respawn", True)),
            }
        )
    return {
        "mode": session.mode,
        "frame": int(session.game.frame),
        "ordered_snakes": snakes,
        "food_positions": sorted([list(item) for item in session.game.food_manager.food]),
        "ordered_slot_to_snake_id": [item["snake_id"] for item in snakes],
    }


def _participants(session: GameSession, candidate_sha256: str) -> list[dict[str, Any]]:
    """Describe controller ownership from the actual constructed session."""
    from src.game.ai_snake import AISnake
    from src.game.human_snake import HumanSnake

    rows = []
    for slot, snake in enumerate(session.game.snakes):
        human = isinstance(snake, HumanSnake)
        if human:
            if slot != 0:
                raise RuntimeError("HumanSnake must occupy slot zero in a Play serving roster")
        elif not isinstance(snake, AISnake) or snake.policy is not session.policy:
            raise RuntimeError(f"candidate AI slot {slot} is not the session-bound AISnake")
        rows.append(
            {
                "slot": slot,
                "snake_id": int(snake.id),
                "controller": "human" if human else "candidate_ai",
                "controller_class": type(snake).__name__,
                "policy_binding": None if human else "session_shared_policy",
                "checkpoint_sha256": None if human else candidate_sha256,
            }
        )
    return rows


def _perpendicular_input(session: GameSession) -> dict[str, Any]:
    """Start Play using one accepted public direction input."""
    human = session._find_human()
    if human is None:
        raise RuntimeError("Play session did not construct a HumanSnake")
    before = list(human.direction)
    direction = "up" if abs(int(human.direction[0])) == 1 else "left"
    run_started_before = bool(session.run_started)
    session.human_input(direction)
    if run_started_before or not session.run_started or list(human.direction) == before:
        raise RuntimeError("public human input was not accepted")
    return {
        "direction_name": direction,
        "direction_before": before,
        "direction_after": list(human.direction),
        "run_started_before": run_started_before,
        "run_started_after": bool(session.run_started),
        "accepted_observed": True,
    }


def _instrument_dispatch(session: GameSession, initial_slots: Mapping[int, int]) -> dict[str, Any]:
    """Observe real serving dispatch without changing selection or movement."""
    from src.game.human_snake import HumanSnake

    expected_ai_ids = {
        int(snake.id) for snake in session.game.snakes if not isinstance(snake, HumanSnake)
    }
    calls = {snake_id: 0 for snake_id in expected_ai_ids}
    observation = {
        "human_control_events": [],
        "human_update_calls": 0,
        "candidate_action_context_calls": [],
        "context_id_mismatch_count": 0,
        "unknown_context_id_count": 0,
    }
    original_context = session.policy.action_context_for

    def observed_context(snake_id: int):
        context = original_context(snake_id)
        snake_id = int(snake_id)
        if int(context.snake_id) != snake_id:
            observation["context_id_mismatch_count"] += 1
        if snake_id not in expected_ai_ids or snake_id not in initial_slots:
            observation["unknown_context_id_count"] += 1
        else:
            calls[snake_id] += 1
        return context

    session.policy.action_context_for = observed_context
    human = session._find_human()
    if human is not None:
        original_update = human.update

        def observed_human_update(*args: Any, **kwargs: Any):
            observation["human_update_calls"] += 1
            return original_update(*args, **kwargs)

        human.update = observed_human_update

    def finalise() -> dict[str, Any]:
        observation["candidate_action_context_calls"] = [
            {"slot": initial_slots[snake_id], "snake_id": snake_id, "calls": calls[snake_id]}
            for snake_id in sorted(expected_ai_ids, key=lambda item: initial_slots[item])
        ]
        return observation

    observation["_finalise"] = finalise
    return observation


def _seed_application(context: SeedContext, mode: str) -> dict[str, Any]:
    return {
        "api": SEED_API,
        "requested_seed": context.requested_seed,
        "effective_seed": context.effective_seed,
        "global_stream_seed": context.stream_seed("global"),
        "evaluated_build_hook": (
            "GameSession.__init__" if mode == MODE_WATCH else "GameSession.set_mode(play)"
        ),
    }


def _validate_session_contract(session: GameSession, spec: ServingEpisodeSpec) -> str:
    contract = session.serving_contract
    if session.obs_spec != RASTER31V3 or not isinstance(contract, dict):
        raise RuntimeError("strict serving episodes require a raster31v3 serving session")
    candidate_sha256 = str(contract.get("checkpoint_sha256", ""))
    if candidate_sha256 != spec.expected_candidate_sha256:
        raise RuntimeError("constructed session checkpoint hash differs from frozen candidate")
    if _checkpoint_digest(spec.checkpoint_path) != candidate_sha256:
        raise RuntimeError("checkpoint bytes changed during serving episode construction")
    return candidate_sha256


def _validate_observations(
    *,
    spec: ServingEpisodeSpec,
    frames_completed: int,
    completion: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    participants: Sequence[Mapping[str, Any]],
    dispatch: Mapping[str, Any],
) -> None:
    """Fail before publication when a receipt would only be a matching claim."""
    expected_ai = [row for row in participants if row["controller"] == "candidate_ai"]
    actual_ai = list(dispatch["candidate_action_context_calls"])
    if len(expected_ai) != len(actual_ai):
        raise RuntimeError("dispatch observation is missing a candidate AI slot")
    if dispatch["context_id_mismatch_count"] or dispatch["unknown_context_id_count"]:
        raise RuntimeError("raster action context identity did not match the calling AI")
    for expected, actual in zip(expected_ai, actual_ai):
        if (expected["slot"], expected["snake_id"]) != (actual["slot"], actual["snake_id"]):
            raise RuntimeError("dispatch observation changed the serving roster identity")
        if not 0 < int(actual["calls"]) <= frames_completed:
            raise RuntimeError("candidate AI slot was not observed through raster dispatch")
    if lifecycle["dead_to_alive_transitions"] or lifecycle["frame_rewind_count"]:
        raise RuntimeError("served hero reset or rewound during a bounded episode")
    if lifecycle["game_instance_change_count"]:
        raise RuntimeError("served game instance changed during a bounded episode")
    if spec.mode == MODE_WATCH:
        if dispatch["human_control_events"] or dispatch["human_update_calls"]:
            raise RuntimeError("Watch evidence unexpectedly entered the human control path")
        if completion["reason"] != "external-horizon" or frames_completed != spec.frame_limit:
            raise RuntimeError("Watch evidence must complete the external horizon")
        if lifecycle["alive_to_dead_transitions"] not in (0, 1):
            raise RuntimeError("terminal Watch hero had an invalid lifecycle")
        if completion["backend_run_over"] or completion["backend_run_frames"] != 0:
            raise RuntimeError("Watch receipt claimed human-run counters")
        return
    if len(dispatch["human_control_events"]) != 1:
        raise RuntimeError("Play evidence requires exactly one accepted public input")
    if dispatch["human_update_calls"] != frames_completed:
        raise RuntimeError("Play evidence did not enter the HumanSnake update path every frame")
    if completion["reason"] == "human-terminal":
        if not (lifecycle["alive_to_dead_transitions"] == 1 and not lifecycle["final_alive"]):
            raise RuntimeError("Play terminal receipt does not match the observed human lifecycle")
        if not (
            completion["owner"] == "GameSession._step_play"
            and completion["backend_run_over"]
            and 1 <= frames_completed <= spec.frame_limit
            and completion["backend_run_frames"] == frames_completed - 1
        ):
            raise RuntimeError("Play terminal receipt has invalid GameSession counters")
    elif completion["reason"] == "external-horizon":
        if lifecycle["alive_to_dead_transitions"] or not lifecycle["final_alive"]:
            raise RuntimeError("nonterminal Play horizon has an invalid human lifecycle")
        if not (
            completion["owner"] == "external-evaluator"
            and not completion["backend_run_over"]
            and frames_completed == spec.frame_limit
            and completion["backend_run_frames"] == frames_completed
        ):
            raise RuntimeError("Play external horizon has invalid GameSession counters")
    else:
        raise RuntimeError("Play completion reason is not a supported serving boundary")


def run_serving_episode(spec: ServingEpisodeSpec) -> dict[str, Any]:
    """Run one actual serving world and return an unpersisted v3 receipt.

    ``frame_limit`` is configurable for unit tests.  Strict plan validation owns
    the external 5,000-frame production requirement.
    """
    stage = "checkpoint"
    last_observed_frame: int | None = None
    session: GameSession | None = None
    try:
        checkpoint_path = spec.checkpoint_path.expanduser().resolve(strict=True)
        if _checkpoint_digest(checkpoint_path) != spec.expected_candidate_sha256:
            raise RuntimeError("checkpoint bytes do not match expected_candidate_sha256")
        stage = "construct"
        if spec.mode == MODE_WATCH:
            seed_context = initialize_run_seed(spec.serving_seed)
            session = GameSession(checkpoint=str(checkpoint_path))
        else:
            # The constructor must bootstrap Watch first.  Reseed directly before
            # the public mode switch that constructs the evaluated Play world.
            session = GameSession(checkpoint=str(checkpoint_path))
            seed_context = initialize_run_seed(spec.serving_seed)
            session.set_mode(MODE_PLAY)
        stage = "validate"
        if session.mode != spec.mode:
            raise RuntimeError(f"GameSession did not enter requested {spec.mode!r} mode")
        candidate_sha256 = _validate_session_contract(session, spec)
        initial_layout = _layout(session)
        if initial_layout["frame"] != 0:
            raise RuntimeError("evaluated serving world did not start at frame zero")
        initial_slots = {
            int(snake_id): slot
            for slot, snake_id in enumerate(initial_layout["ordered_slot_to_snake_id"])
        }
        participants = _participants(session, candidate_sha256)
        if spec.mode == MODE_PLAY and (
            not participants or participants[0]["controller"] != "human"
        ):
            raise RuntimeError("Play roster did not contain its required human slot")
        dispatch = _instrument_dispatch(session, initial_slots)
        if spec.mode == MODE_PLAY:
            stage = "input"
            dispatch["human_control_events"].append(_perpendicular_input(session))

        stage = "step"
        hero_id = int(session.hero_id)
        hero = next(snake for snake in session.game.snakes if int(snake.id) == hero_id)
        hero_alive = bool(hero.is_alive)
        if not hero_alive:
            raise RuntimeError("serving hero was not alive at the initial layout")
        start_frame = int(session.game.frame)
        initial_game_id = id(session.game)
        previous_frame = start_frame
        previous_alive = hero_alive
        alive_to_dead = 0
        dead_to_alive = 0
        rewinds = 0
        game_changes = 0
        frames_completed = 0

        while frames_completed < spec.frame_limit and not session.run_over:
            session.step()
            frames_completed += 1
            if id(session.game) != initial_game_id:
                game_changes += 1
            current_frame = int(session.game.frame)
            last_observed_frame = current_frame
            if current_frame < previous_frame:
                rewinds += 1
            if current_frame != previous_frame + 1:
                raise RuntimeError("serving session did not advance exactly one frame per step")
            previous_frame = current_frame
            current_hero = next(snake for snake in session.game.snakes if int(snake.id) == hero_id)
            current_alive = bool(current_hero.is_alive)
            alive_to_dead += int(previous_alive and not current_alive)
            dead_to_alive += int(not previous_alive and current_alive)
            previous_alive = current_alive

        end_frame = int(session.game.frame)
        if end_frame - start_frame != frames_completed:
            raise RuntimeError("served frame counter disagrees with external step count")
        if session.run_over:
            completion = {
                "owner": "GameSession._step_play",
                "reason": "human-terminal",
                "backend_run_over": True,
                "backend_run_frames": int(session.run_frames),
            }
        else:
            completion = {
                "owner": "external-evaluator",
                "reason": "external-horizon",
                "backend_run_over": False,
                "backend_run_frames": int(session.run_frames),
            }
        if frames_completed != spec.frame_limit and not session.run_over:
            raise RuntimeError("serving episode stopped before its external horizon")
        if spec.mode == MODE_WATCH and completion["reason"] != "external-horizon":
            raise RuntimeError("Watch session unexpectedly finalized a human run")

        finalise_dispatch = dispatch.pop("_finalise")
        dispatch = finalise_dispatch()
        lifecycle = {
            "snake_id": hero_id,
            "initial_alive": hero_alive,
            "final_alive": previous_alive,
            "alive_to_dead_transitions": alive_to_dead,
            "dead_to_alive_transitions": dead_to_alive,
            "frame_rewind_count": rewinds,
            "game_instance_change_count": game_changes,
        }
        _validate_observations(
            spec=spec,
            frames_completed=frames_completed,
            completion=completion,
            lifecycle=lifecycle,
            participants=participants,
            dispatch=dispatch,
        )
        if _checkpoint_digest(checkpoint_path) != candidate_sha256:
            raise RuntimeError("checkpoint bytes changed while the serving episode ran")
        stage = "complete"
        return {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": spec.episode_id,
            "mode": spec.mode,
            "serving_seed": spec.serving_seed,
            "checkpoint_path": str(checkpoint_path),
            "candidate_sha256": candidate_sha256,
            "profile_digest": spec.profile_digest,
            "source_closure_sha256": spec.source_closure_sha256,
            "seed_application": _seed_application(seed_context, spec.mode),
            "initial_layout": initial_layout,
            "initial_layout_digest": canonical_digest(initial_layout),
            "frame_limit": spec.frame_limit,
            "start_frame": start_frame,
            "frames_completed": frames_completed,
            "end_frame": end_frame,
            "completion": completion,
            "hero_lifecycle": lifecycle,
            "participants": participants,
            "dispatch_observation": dispatch,
            "obs_spec": session.obs_spec,
            "serving_contract": session.serving_contract,
            "status": "completed",
            "error": None,
        }
    except Exception as exc:
        if isinstance(exc, ServingEpisodeError):
            raise
        raise ServingEpisodeError(stage, last_observed_frame, exc) from exc
    finally:
        if session is not None:
            try:
                session.game.full_cleanup()
            except Exception as exc:
                raise ServingEpisodeError("cleanup", last_observed_frame, exc) from exc


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish canonical JSON once, with no replacement path."""
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"serving receipt path already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # ``link`` creates the destination only if it is still absent.  Unlike
        # replace(), it cannot overwrite a prior attempt between preflight and
        # publication.  Both names share the same already-fsynced inode.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_serving_episode_receipt(path: Path, spec: ServingEpisodeSpec) -> int:
    """Execute a valid spec and atomically retain its completed or failed receipt."""
    if os.path.lexists(path.expanduser()):
        raise FileExistsError(f"serving receipt path already exists: {path}")
    try:
        receipt = run_serving_episode(spec)
    except Exception as exc:
        _write_create_only(path, _failed_receipt(spec, exc))
        return 1
    _write_create_only(path, receipt)
    return 0


def _failed_receipt(spec: ServingEpisodeSpec, error: Exception) -> dict[str, Any]:
    """Retain a same-schema failed attempt without inventing observations."""
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": spec.episode_id,
        "mode": spec.mode,
        "serving_seed": spec.serving_seed,
        "checkpoint_path": str(spec.checkpoint_path.expanduser()),
        "candidate_sha256": None,
        "profile_digest": spec.profile_digest,
        "source_closure_sha256": spec.source_closure_sha256,
        "seed_application": None,
        "initial_layout": None,
        "initial_layout_digest": None,
        "frame_limit": spec.frame_limit,
        "start_frame": None,
        "frames_completed": None,
        "end_frame": None,
        "completion": None,
        "hero_lifecycle": None,
        "participants": None,
        "dispatch_observation": None,
        "obs_spec": None,
        "serving_contract": None,
        "status": "failed",
        "error": {
            "stage": getattr(error, "stage", "preflight"),
            "exception_type": type(getattr(error, "cause", error)).__name__,
            "message": str(error),
            "last_observed_frame": getattr(error, "last_observed_frame", None),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--serving-seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--profile-digest", required=True)
    parser.add_argument("--source-closure-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame-limit", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for one serial serving-plan row, retaining a failure artifact on error."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    spec: ServingEpisodeSpec | None = None
    try:
        spec = ServingEpisodeSpec(
            episode_id=args.episode_id,
            mode=args.mode,
            serving_seed=args.serving_seed,
            checkpoint_path=Path(args.checkpoint),
            expected_candidate_sha256=args.candidate_sha256,
            profile_digest=args.profile_digest,
            source_closure_sha256=args.source_closure_sha256,
            frame_limit=args.frame_limit,
        )
        # Refuse an existing path before session creation, so a prior receipt
        # cannot be overwritten and an invalid retry cannot consume a world.
        if os.path.lexists(args.out.expanduser()):
            raise FileExistsError(f"serving receipt path already exists: {args.out}")
        return write_serving_episode_receipt(args.out, spec)
    except Exception as exc:
        if spec is not None and not os.path.lexists(args.out.expanduser()):
            try:
                _write_create_only(args.out, _failed_receipt(spec, exc))
            except Exception:
                pass
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
