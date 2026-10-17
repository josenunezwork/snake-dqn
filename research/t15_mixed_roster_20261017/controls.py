"""Bounded real-session controls for the T15 mixed Watch qualification.

The controls deliberately exercise selection failures through test-local registry
objects.  They never edit the checked-in pin document, checkpoint registry, or
serving environment outside the lifetime of a case.
"""

from __future__ import annotations

import contextlib
import copy
import io
import logging
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from unittest.mock import patch

ENV_KEYS = (
    "SNAKE_SERVE_CHECKPOINT",
    "SNAKE_SERVE_WATCH_ROSTER",
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
    "SNAKE_SERVE_VETO_VARIANT",
    "SNAKE_SCORES_DB",
)
LOGGER_NAMES = ("web.backend.served_checkpoint", "web.backend.safety_veto_serving")
_CASE_SEED = 0


@contextlib.contextmanager
def _environment(values: Mapping[str, str | None]) -> Iterator[None]:
    """Apply a case environment without leaking it into another control."""
    before = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in values.items():
            if value is not None:
                os.environ[key] = str(value)
        yield
    finally:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in before.items():
            if value is not None:
                os.environ[key] = value


@contextlib.contextmanager
def _score_database(path: str) -> Iterator[None]:
    """Redirect only score persistence while preserving the case selectors."""
    previous = os.environ.get("SNAKE_SCORES_DB")
    try:
        os.environ["SNAKE_SCORES_DB"] = path
        yield
    finally:
        if previous is None:
            os.environ.pop("SNAKE_SCORES_DB", None)
        else:
            os.environ["SNAKE_SCORES_DB"] = previous


@contextlib.contextmanager
def _stderr_capture() -> Iterator[io.StringIO]:
    """Capture relevant build records without changing a caller's logging setup."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    loggers = [logging.getLogger(name) for name in LOGGER_NAMES]
    prior = [(logger, logger.level, logger.propagate) for logger in loggers]
    try:
        for logger in loggers:
            logger.addHandler(handler)
            if logger.level == logging.NOTSET or logger.level > logging.INFO:
                logger.setLevel(logging.INFO)
        yield stream
    finally:
        for logger, level, propagate in prior:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


def _identity(session: Any) -> dict[str, Any]:
    """Retain raw session identities and flags for independent audit."""
    veto = session.safety_veto_state() or {}
    return {
        "mode": session.mode,
        "checkpoint_path": session.checkpoint_path,
        "obs_spec": session.obs_spec,
        "served_checkpoint": dict(session.served_checkpoint or {}),
        "watch_roster": session.watch_roster,
        "watch_roster_reason": session.watch_roster_reason,
        "watch_opponent_checkpoint": dict(session.watch_opponent_checkpoint or {}),
        "hero_id": int(session.hero_id),
        "snake_ids": [int(snake.id) for snake in session.game.snakes],
        "policy_classes": [
            type(getattr(snake, "policy", None)).__name__ for snake in session.game.snakes
        ],
        "hero_wrapped": getattr(session.game.snakes[0], "safety_veto", None) is not None,
        "veto": veto,
        "flags": dict(veto.get("flags") or {}),
        "speed_ceiling": float(session.speed_ceiling),
        "total_ai_count": sum(type(snake).__name__ == "AISnake" for snake in session.game.snakes),
        "last_error": session.last_error,
    }


def _observe(session: Any, steps: int = 1) -> dict[str, Any]:
    """Advance only a bounded real serving probe and retain its raw state."""
    errors: list[str] = []
    advanced = 0
    for _ in range(steps):
        try:
            before = int(session.game.frame)
            session.step()
            session.snapshot()
            advanced += int(int(session.game.frame) == before + 1)
        except Exception as exc:  # retained as qualification evidence, never retried
            errors.append(f"{type(exc).__name__}: {exc}")
            break
    result = _identity(session)
    result.update({"frames_requested": steps, "frames_advanced": advanced, "errors": errors})
    return result


def _case(
    name: str,
    expected: Mapping[str, Any],
    factory: Callable[[], Any],
    *,
    steps: int = 1,
    check: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """Run one non-retrying case and expose expected/observed raw facts."""
    from research.apex_veto_serving_20261001.serving_run import set_world_seed
    from web.backend import session as session_module

    with tempfile.TemporaryDirectory(prefix="t15-controls-scores-") as temporary:
        with _stderr_capture() as stderr:
            with (
                patch.object(session_module, "served_resolution"),
                patch.object(session_module, "watch_tick"),
                patch.object(session_module, "play_tick"),
                patch.object(session_module, "ego2s_exception"),
                _score_database(str(Path(temporary) / "scores.sqlite")),
            ):
                try:
                    set_world_seed(_CASE_SEED)
                    observed = _observe(factory(), steps)
                    no_errors = not observed["errors"]
                except Exception as exc:
                    observed = {
                        "construction_error": f"{type(exc).__name__}: {exc}",
                        "errors": [str(exc)],
                    }
                    no_errors = False
    observed["stderr"] = stderr.getvalue().splitlines()
    observed["world_seed"] = _CASE_SEED
    passed = bool(no_errors and check(observed))
    return {
        "name": name,
        "expected": dict(expected),
        "observed": observed,
        "steps": int(steps),
        "no_errors": no_errors,
        "pass": passed,
    }


def _session() -> Any:
    from web.backend.session import GameSession

    return GameSession()


def _fallback_choice(reason: str) -> Any:
    from web.backend.served_checkpoint import ServedCheckpointChoice

    return ServedCheckpointChoice(requested="frp3-s12", reason=reason)


def _mixed_case(
    name: str,
    expected: Mapping[str, Any],
    values: Mapping[str, str | None],
    *,
    steps: int = 1,
    extra: contextlib.AbstractContextManager[Any] | None = None,
    check: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    with _environment(values):
        with extra if extra is not None else contextlib.nullcontext():
            return _case(name, expected, _session, steps=steps, check=check)


def _all_b8(observed: Mapping[str, Any]) -> bool:
    return observed.get("watch_roster") == "all_b8" and not observed.get("hero_wrapped")


def _mixed(observed: Mapping[str, Any]) -> bool:
    veto = observed.get("veto") or {}
    ids = observed.get("snake_ids") or []
    return (
        observed.get("watch_roster") == "mixed"
        and observed.get("hero_wrapped") is False
        and veto.get("active") is True
        and veto.get("scope") == "watch_opponents"
        and veto.get("wrapped_snake_ids") == ids[1:]
        and any("active=True scope=watch_opponents" in line for line in observed.get("stderr", []))
    )


def _temporary_opponent_refusal(kind: str) -> contextlib.AbstractContextManager[Any]:
    """Use copied in-memory registry/pins and a temporary artifact for one refusal.

    GameSession still runs normally; only its opponent resolver receives the
    resulting failed choice.  This isolates missing/corrupt byte checks from
    checked-in registry and pin state.
    """
    from web.backend import served_checkpoint as registry
    from web.backend import session as session_module

    temp = tempfile.TemporaryDirectory(prefix="t15-controls-")
    root = Path(temp.name)
    artifact = root / "frp3.pth"
    if kind == "corrupt_bytes":
        artifact.write_bytes(b"t15-corrupt-frp3")
    pins = copy.deepcopy(registry.load_pins())
    if kind == "missing_pin":
        pins["checkpoints"].pop(registry.NAME_FRP3_S12, None)
    elif kind == "unfilled_pin":
        pins["checkpoints"][registry.NAME_FRP3_S12]["strict_receipt"] = None
    elif kind == "wrong_pin":
        pins["checkpoints"][registry.NAME_FRP3_S12]["checkpoint_sha256"] = "0" * 64
    entry = registry.REGISTRY[registry.NAME_FRP3_S12]
    copied = replace(entry, artifact_path=str(artifact), filename="missing-frp3.pth")
    stack = contextlib.ExitStack()
    stack.enter_context(patch.dict(registry.REGISTRY, {registry.NAME_FRP3_S12: copied}))
    choice = registry.resolve_mixed_watch_opponent(saved_dir=str(root), pins=pins)
    stack.enter_context(
        patch.object(session_module, "resolve_mixed_watch_opponent", return_value=choice)
    )
    stack.callback(temp.cleanup)
    return stack


def _changed_wrapper_source() -> contextlib.AbstractContextManager[Any]:
    from web.backend import safety_veto_serving as veto

    real = veto.wrapper_identity_v8

    def changed() -> dict[str, Any]:
        value = copy.deepcopy(real())
        value["source_sha256s"]["src/evaluation/safety_veto_v8.py"] = "0" * 64
        return value

    return patch.object(veto, "wrapper_identity_v8", changed)


def _b8_fallback_context() -> contextlib.AbstractContextManager[Any]:
    """Force a B8 refusal through a temporary copied registry entry."""
    from web.backend import served_checkpoint as registry

    temporary = tempfile.TemporaryDirectory(prefix="t15-controls-fallback-")
    root = Path(temporary.name)
    b8 = registry.REGISTRY[registry.NAME_B8_S3]
    corrupt_b8 = replace(
        b8,
        filename="absent-b8.pth",
        artifact_path=str(root / "corrupt-b8.pth"),
    )
    (root / "corrupt-b8.pth").write_bytes(b"corrupt-b8")
    stack = contextlib.ExitStack()
    stack.enter_context(patch.dict(registry.REGISTRY, {registry.NAME_B8_S3: corrupt_b8}))
    stack.callback(temporary.cleanup)
    return stack


def _activate_play(session: Any) -> Any:
    """Start a real Play frame before the control's bounded observation tick."""
    for direction in ("up", "right", "down", "left"):
        session.human_input(direction)
        if session.run_started:
            return session
    raise RuntimeError("could not start the Play control with an accepted human direction")


def run_controls(seed: int) -> dict[str, Any]:
    """Run the protocol check-3 matrix with at most three frames per row.

    ``seed`` is recorded for the registered controls bank.  GameSession uses its
    normal runtime construction; controls do not generate qualification episodes.
    """
    global _CASE_SEED
    _CASE_SEED = int(seed)
    base = {"SNAKE_SERVE_CHECKPOINT": "b8-s3"}
    cases: list[dict[str, Any]] = []
    cases.append(_mixed_case("roster_unset", {"roster": "all_b8"}, base, check=_all_b8))
    cases.append(
        _mixed_case(
            "roster_all_b8",
            {"roster": "all_b8"},
            {**base, "SNAKE_SERVE_WATCH_ROSTER": "all_b8"},
            check=_all_b8,
        )
    )
    cases.append(
        _mixed_case(
            "roster_mixed",
            {"roster": "mixed", "scope": "watch_opponents"},
            {**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed"},
            check=_mixed,
        )
    )
    cases.append(
        _mixed_case(
            "roster_unknown",
            {"scenario": "unknown roster", "roster": "all_b8", "reason_contains": "unknown"},
            {**base, "SNAKE_SERVE_WATCH_ROSTER": "unknown"},
            check=lambda row: _all_b8(row) and "unknown" in str(row.get("watch_roster_reason")),
        )
    )
    cases.append(
        _mixed_case(
            "b8_fallback_chain",
            {"served_name": "frp3-s12", "fallback": "b8->frp3->champion"},
            {"SNAKE_SERVE_CHECKPOINT": "b8-s3"},
            extra=_b8_fallback_context(),
            check=lambda row: (row.get("served_checkpoint") or {}).get("name") == "frp3-s12"
            and row.get("watch_roster") == "all_b8",
        )
    )
    checkpoint_expectations = {
        "b8-s3": {"served_name": "b8-s3", "obs_spec": "ego2s-bv8", "hero_wrapped": False},
        "frp3-s12": {"served_name": "frp3-s12", "obs_spec": "vector61", "hero_wrapped": True},
        "champion": {"served_name": "champion", "obs_spec": "vector61", "hero_wrapped": True},
        "not-a-checkpoint": {
            "scenario": "unknown checkpoint",
            "served_name": "b8-s3",
            "reason_contains": "unknown",
        },
    }
    for name, expected in checkpoint_expectations.items():

        def selected(row: Mapping[str, Any], requested: str = name) -> bool:
            choice = row.get("served_checkpoint") or {}
            if requested == "not-a-checkpoint":
                return choice.get("name") == "b8-s3" and "unknown" in str(choice.get("reason"))
            return (
                choice.get("requested") == requested
                and choice.get("name") == requested
                and row.get("obs_spec") == checkpoint_expectations[requested]["obs_spec"]
                and row.get("hero_wrapped") is checkpoint_expectations[requested]["hero_wrapped"]
            )

        cases.append(
            _mixed_case(
                f"checkpoint_{name}",
                expected,
                {"SNAKE_SERVE_CHECKPOINT": name},
                check=selected,
            )
        )
    for variant in (None, "v2", "v5", "v7", "v8", "unknown"):
        values = {**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed"}
        if variant is not None:
            values["SNAKE_SERVE_VETO_VARIANT"] = variant
        cases.append(
            _mixed_case(
                f"mixed_variant_{variant or 'unset'}",
                {"requested_variant": variant, "hero_wrapped": False},
                values,
                check=lambda row: not row.get("hero_wrapped"),
            )
        )
    cases.append(
        _mixed_case(
            "mixed_watch_hero_off",
            {"hero_wrapped": False},
            {**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed", "SNAKE_SERVE_VETO_WATCH_HERO": "0"},
            check=lambda row: not row.get("hero_wrapped"),
        )
    )
    for kind in ("missing_pin", "unfilled_pin", "wrong_pin", "missing_bytes", "corrupt_bytes"):
        cases.append(
            _mixed_case(
                f"mixed_{kind}",
                {
                    "scenario": "unavailable bytes" if kind == "missing_bytes" else kind,
                    "roster": "all_b8",
                    "no_partial_wrappers": True,
                },
                {**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed"},
                extra=_temporary_opponent_refusal(kind),
                check=lambda row: _all_b8(row)
                and not (row.get("veto") or {}).get("wrapped_snake_ids"),
            )
        )
    cases.append(
        _mixed_case(
            "mixed_changed_wrapper_source",
            {"roster": "all_b8", "reason_contains": "wrapper"},
            {**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed"},
            extra=_changed_wrapper_source(),
            check=lambda row: _all_b8(row)
            and "wrapper" in str(row.get("watch_roster_reason")).lower(),
        )
    )

    def play_transition() -> Any:
        session = _session()
        session.set_mode("play")
        return _activate_play(session)

    transition_env = {**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed"}
    with _environment(transition_env):
        cases.append(
            _case(
                "mode_play_refuses_mixed",
                {"mode": "play", "roster": "all_b8"},
                play_transition,
                check=lambda row: row.get("mode") == "play" and _all_b8(row),
            )
        )
    with _environment({**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed"}):

        def train_refusal() -> Any:
            session = _session()
            session.set_mode("train")
            return session

        cases.append(
            _case(
                "train_refusal_b8",
                {"scenario": "train", "mode": "watch", "error_contains": "forward-only"},
                train_refusal,
                check=lambda row: row.get("mode") == "watch"
                and "forward-only" in str(row.get("last_error")).lower(),
            )
        )
    with _environment(transition_env):

        def round_trip() -> Any:
            session = _session()
            session.set_mode("play")
            session.set_mode("watch")
            return session

        cases.append(
            _case(
                "mode_watch_rebuilds_mixed",
                {"mode": "watch", "roster": "mixed"},
                round_trip,
                check=_mixed,
            )
        )
    with _environment(base):

        def replacement() -> Any:
            session = _session()
            session.load_checkpoint("not-a-checkpoint.pth")
            return session

        cases.append(
            _case(
                "checkpoint_replacement_refused",
                {"existing_session_preserved": True},
                replacement,
                check=lambda row: row.get("served_checkpoint", {}).get("name") == "b8-s3"
                and "Unknown checkpoint" in str(row.get("last_error")),
            )
        )
        from web.backend import served_checkpoint as registry

        def replacement_success() -> Any:
            session = _session()
            session.load_checkpoint(registry.B8_S3_FILENAME)
            return session

        cases.append(
            _case(
                "checkpoint_replacement_success_b8",
                {"scenario": "selected checkpoint replacement", "obs_spec": "ego2s-bv8"},
                replacement_success,
                check=lambda row: row.get("served_checkpoint") == {}
                and row.get("obs_spec") == "ego2s-bv8"
                and row.get("last_error") is None,
            )
        )
    for checkpoint in ("champion", "frp3-s12"):
        with _environment(
            {
                "SNAKE_SERVE_CHECKPOINT": checkpoint,
                "SNAKE_SERVE_VETO_PLAY_AI": "1",
            }
        ):

            def play_veto(name: str = checkpoint) -> Any:
                session = _session()
                session.set_mode("play")
                return _activate_play(session)

            cases.append(
                _case(
                    f"play_opt_in_veto_{'frp3' if checkpoint == 'frp3-s12' else checkpoint}",
                    {
                        "checkpoint": checkpoint,
                        "scope": "play_ai",
                        "active": checkpoint == "champion",
                    },
                    play_veto,
                    check=lambda row, name=checkpoint: (
                        row.get("mode") == "play"
                        and (row.get("served_checkpoint") or {}).get("name") == name
                        and (row.get("veto") or {}).get("active") is (name == "champion")
                        and (row.get("veto") or {}).get("scope") == "play_ai"
                        and row.get("hero_wrapped") is False
                        and (
                            name == "champion"
                            or "no strict-gate evidence"
                            in str((row.get("veto") or {}).get("reason")).lower()
                        )
                    ),
                )
            )
    for size, ceiling in ((6, 24.0), (8, 12.0), (12, 12.0)):
        # Test-local loader override changes only the ephemeral GameSession config.
        from web.backend import session as session_module

        real_loader = session_module.load_config

        def loader(path: str, n: int = size) -> Any:
            config = real_loader(path)
            return replace(config, game=replace(config.game, num_snakes=n))

        with (
            _environment({**base, "SNAKE_SERVE_WATCH_ROSTER": "mixed"}),
            patch.object(session_module, "load_config", side_effect=loader),
        ):
            cases.append(
                _case(
                    f"mixed_roster_count_{size}",
                    {"total_ai_count": size, "speed_ceiling": ceiling},
                    _session,
                    check=lambda row, n=size, c=ceiling: row.get("total_ai_count") == n
                    and row.get("speed_ceiling") == c,
                )
            )
    return {
        "schema_version": "t15-mixed-watch-controls/v1",
        "seed": int(seed),
        "cases": cases,
        "pass": all(case["pass"] for case in cases),
    }
