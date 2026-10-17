"""Focused checks for the post-release B8 log evaluator."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from scripts import monitor_b8_release as monitor
from src.model.obs_spec import EGO2S_B
from web.backend import session as session_mod
from web.backend.session import MODE_PLAY, MODE_WATCH, GameSession


def _healthy_records(*, update_ms: float = 8.0) -> list[dict]:
    return [
        {
            "event": "resolution",
            "requested": None,
            "name": "b8-s3",
            "fallback_reason": None,
        },
        {
            "event": "watch_tick",
            "checkpoint": "b8-s3",
            "roster_size": 6,
            "speed": 12.0,
            "decisions": 1,
            "deaths": 0,
            "update_ms": update_ms,
            "timestamp": 0.0,
            "latency_eligible": True,
        },
    ]


def test_healthy_b8_watch_records_report_all_three_metrics() -> None:
    healthy, messages, metrics = monitor.assess(_healthy_records(), [])

    assert healthy and messages == []
    assert metrics["deaths_per_1000"] == 0.0
    assert metrics["deaths_by_roster"]["watch6"] == {
        "decisions": 1,
        "deaths": 0,
        "deaths_per_1000": 0.0,
        "baseline_deaths_per_1000": 0.424,
        "trigger_per_1000": 0.848,
    }
    assert metrics["latency_by_roster"]["watch6"]["update_p95_ms"] == 8.0
    assert metrics["latency_by_roster"]["watch6"]["trigger_ms"] == (
        1.5 * monitor.UPDATE_P95_BASELINE_MS[("watch", 6)]
    )


def test_any_fallback_or_ego2s_exception_requires_rollback() -> None:
    records = _healthy_records()
    records[0]["fallback_reason"] = "b8-s3 bytes did not verify"
    records.append({"event": "ego2s_exception", "error_type": "RuntimeError"})

    healthy, messages, metrics = monitor.assess(records, [])

    assert not healthy
    assert metrics["fallback_records"] == 1 and metrics["ego2s_exceptions"] == 1
    assert any("fallback" in message for message in messages)
    assert any("ego2s-path exception" in message for message in messages)


def test_death_and_latency_thresholds_are_strictly_greater() -> None:
    records = _healthy_records(update_ms=12.1)
    records[1].update(decisions=1, deaths=1)

    healthy, messages, _ = monitor.assess(records, [])

    assert not healthy
    assert any("deaths/1k" in message for message in messages)
    assert any("update p95" in message for message in messages)


def _watch_death_rate_records(*, roster_size: int, deaths: int) -> list[dict]:
    """Build a 125,000-decision Watch record set without sharing death state."""
    speed = 24.0 if roster_size == 8 else 12.0
    dead_tick = {
        "event": "watch_tick",
        "checkpoint": "b8-s3",
        "roster_size": roster_size,
        "speed": speed,
        "decisions": 1,
        "deaths": 1,
        "update_ms": 8.0,
        "timestamp": 0.0,
        "latency_eligible": True,
    }
    survived_tick = {**dead_tick, "deaths": 0}
    records = [
        _healthy_records()[0],
        *([dead_tick] * deaths),
        *([survived_tick] * (125_000 - deaths)),
    ]
    if roster_size == 8:
        # T13b registers Watch8's death baseline only. Watch6 supplies the
        # existing default-speed latency sample, while Watch8 stays unqualified.
        records.insert(1, _healthy_records()[1])
    return records


@pytest.mark.parametrize(
    ("roster_size", "deaths", "rolls_back"),
    [
        (6, 106, False),  # 0.848/1k: trigger is strictly greater.
        (6, 107, True),
        (12, 192, False),  # 1.536/1k: trigger is strictly greater.
        (12, 193, True),
        (8, 124, False),  # 0.992/1k: T13b Watch8 death trigger.
        (8, 125, True),
    ],
)
def test_watch_death_trigger_boundaries(roster_size: int, deaths: int, rolls_back: bool) -> None:
    healthy, messages, metrics = monitor.assess(
        _watch_death_rate_records(roster_size=roster_size, deaths=deaths), []
    )

    assert healthy is not rolls_back
    assert any("deaths/1k" in message for message in messages) is rolls_back
    if roster_size == 12:
        assert metrics["deaths_by_roster"]["watch12"]["deaths_per_1000"] > 0.848
    if roster_size == 8:
        assert metrics["deaths_by_roster"]["watch8"]["trigger_per_1000"] == 0.992
        assert "watch8" not in metrics["latency_by_roster"]


def test_watch_death_thresholds_are_per_roster_and_cannot_be_diluted() -> None:
    records = _healthy_records()
    records[1].update(decisions=1, deaths=1)
    records.extend(
        {
            "event": "watch_tick",
            "checkpoint": "b8-s3",
            "roster_size": 12,
            "speed": 12.0,
            "decisions": 1,
            "deaths": 0,
            "update_ms": 8.0,
            "timestamp": 0.0,
            "latency_eligible": True,
        }
        for _ in range(10_000)
    )

    healthy, messages, metrics = monitor.assess(records, [])

    assert not healthy
    assert metrics["deaths_per_1000"] < 0.848
    assert metrics["deaths_by_roster"]["watch6"]["deaths_per_1000"] == 1000.0
    assert metrics["deaths_by_roster"]["watch12"]["deaths_per_1000"] == 0.0
    assert any("watch6 deaths/1k" in message for message in messages)


def test_watch12_uses_its_own_death_trigger() -> None:
    records = _healthy_records()
    records[1].update(roster_size=12)

    healthy, messages, metrics = monitor.assess(records, [])

    assert healthy and messages == []
    assert metrics["deaths_by_roster"]["watch12"]["trigger_per_1000"] == 1.536


def test_unknown_watch_roster_fails_closed_even_without_latency_samples() -> None:
    records = _healthy_records()
    records.append(
        {
            "event": "watch_tick",
            "checkpoint": "b8-s3",
            "roster_size": 7,
            "speed": 24.0,
            "decisions": 0,
            "deaths": 0,
            "update_ms": 1.0,
            "timestamp": 0.0,
            "latency_eligible": True,
        }
    )

    healthy, messages, _ = monitor.assess(records, [])

    assert not healthy
    assert "no registered Watch death baseline for watch7" in messages


def test_missing_or_mixed_telemetry_never_reports_healthy() -> None:
    healthy, messages, _ = monitor.assess([], [])
    assert not healthy and any("missing" in message for message in messages)

    mixed = _healthy_records()
    mixed.append(
        {
            "event": "watch_tick",
            "checkpoint": "b8-s3",
            "roster_size": 8,
            "speed": 12.0,
            "decisions": 0,
            "deaths": 0,
            "update_ms": 1.0,
            "timestamp": 0.0,
            "latency_eligible": True,
        }
    )
    healthy, messages, _ = monitor.assess(mixed, [])
    assert not healthy and any(
        "no T12B-qualified update baseline for watch8" in message for message in messages
    )
    assert monitor.WATCH_DEATHS_PER_1K_BASELINE[8] == 0.496


def test_parser_retains_malformed_prefixed_log_as_a_fault() -> None:
    items, faults = monitor.records(
        [
            "ordinary server output\n",
            'b8-release-monitor: {"event":"resolution","name":"b8-s3"}\n',
            "b8-release-monitor: not-json\n",
        ]
    )

    assert items == [{"event": "resolution", "name": "b8-s3"}]
    assert faults == ["malformed monitoring JSON"]


def _b8_session(game: object, *, mode: str = MODE_WATCH, warmup: int = 0) -> GameSession:
    """Construct a minimal B8 session without loading a checkpoint or server."""
    session = GameSession.__new__(GameSession)
    session._lock = threading.RLock()
    session.playing = True
    session.mode = mode
    session.game = game
    session.policy = object()
    session.hero_id = 0
    session.speed = 12.0
    session.obs_spec = EGO2S_B
    session.served_checkpoint = {"name": "b8-s3"}
    session._release_monitoring = True
    session._release_latency_warmup_remaining = warmup
    session.checkpoint_path = "/test/b8.pth"
    session._enforce_food_target = lambda: None
    return session


def test_watch_counts_actual_post_respawn_hero_update(monkeypatch: pytest.MonkeyPatch) -> None:
    class Hero:
        id = 0
        is_alive = False

        def update(self, *_args, **_kwargs) -> None:
            self.updated = True

    hero = Hero()

    class Game:
        snakes = [hero]
        frame_death_causes = {}

        def update(self, **_kwargs) -> None:
            hero.is_alive = True  # GameState's continuous Watch respawn phase.
            hero.update([], [])

    events: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(session_mod, "watch_tick", lambda **kwargs: events.append(kwargs))
    _b8_session(Game()).step()

    assert hero.updated is True
    assert events[0]["decisions"] == 1


def test_watch_restores_hero_update_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Hero:
        id = 0
        is_alive = True

        def update(self, *_args, **_kwargs) -> None:
            raise RuntimeError("forward failed")

    hero = Hero()
    original_update = hero.update

    class Game:
        snakes = [hero]
        frame_death_causes = {}

        def update(self, **_kwargs) -> None:
            hero.update([], [])

    errors: list[BaseException] = []
    monkeypatch.setattr(
        session_mod, "ego2s_exception", lambda **kwargs: errors.append(kwargs["error"])
    )
    session = _b8_session(Game())
    with pytest.raises(RuntimeError, match="forward failed"):
        session.step()

    assert hero.update == original_update
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)


@pytest.mark.parametrize("operation", ["build", "play", "snapshot"])
def test_b8_build_play_and_snapshot_failures_are_durable(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    errors: list[dict] = []
    monkeypatch.setattr(session_mod, "ego2s_exception", lambda **kwargs: errors.append(kwargs))
    if operation == "build":
        session = GameSession.__new__(GameSession)
        session._release_monitoring = True
        session.served_checkpoint = {"name": "b8-s3"}
        session._build_impl = lambda *_args: (_ for _ in ()).throw(RuntimeError("build failed"))
        monkeypatch.setattr(session_mod, "get_config", lambda: object())
        monkeypatch.setattr(session_mod, "initialize_config", lambda _config: None)
        with pytest.raises(RuntimeError, match="build failed"):
            session._build("/test/b8.pth", MODE_WATCH)
    elif operation == "play":
        game = SimpleNamespace(
            update=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("play failed"))
        )
        session = _b8_session(game, mode=MODE_PLAY)
        session.run_started = True
        with pytest.raises(RuntimeError, match="play failed"):
            session._step_play()
    else:
        session = _b8_session(SimpleNamespace())
        monkeypatch.setattr(
            "web.backend.serialize.build_frame",
            lambda _session: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        )
        with pytest.raises(RuntimeError, match="snapshot failed"):
            session.snapshot()

    assert len(errors) == 1
    assert errors[0]["checkpoint"] == "b8-s3"


def test_warmup_and_reset_exclude_latency_but_not_watch_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hero = SimpleNamespace(id=0, is_alive=True, update=lambda *_args: None)
    game = SimpleNamespace(snakes=[hero], frame_death_causes={}, reset=lambda: None)
    game.update = lambda **_kwargs: hero.update([], [])
    events: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(session_mod, "watch_tick", lambda **kwargs: events.append(kwargs))
    session = _b8_session(game, warmup=1)
    session.step()
    session.step()
    session.reset_game()

    assert [event["decisions"] for event in events] == [1, 1]
    assert [event["latency_eligible"] for event in events] == [False, True]
    assert session._release_latency_warmup_remaining == 200


def test_non_b8_watch_does_not_add_monitoring_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    hero = SimpleNamespace(id=0, is_alive=True, update=lambda *_args: None)
    game = SimpleNamespace(snakes=[hero], frame_death_causes={})
    game.update = lambda **_kwargs: hero.update([], [])
    ticks: list[dict] = []
    monkeypatch.setattr(session_mod, "watch_tick", lambda **kwargs: ticks.append(kwargs))
    session = _b8_session(game)
    session.served_checkpoint = {"name": "champion"}
    original_update = hero.update
    session.step()

    assert hero.update == original_update
    assert ticks == []


def test_evaluator_rejects_nonfinite_or_malformed_latency_fields() -> None:
    records = _healthy_records()
    records[1]["update_ms"] = float("nan")
    healthy, messages, _ = monitor.assess(records, [])
    assert not healthy and "incomplete Watch tick telemetry" in messages

    records = _healthy_records()
    records[1]["latency_eligible"] = "false"
    healthy, messages, _ = monitor.assess(records, [])
    assert not healthy and "incomplete Watch tick telemetry" in messages

    records = _healthy_records()
    records.append(
        {
            "event": "play_tick",
            "checkpoint": "b8-s3",
            "driven_ai": 5,
            "speed": 12.0,
            "update_ms": -1.0,
            "latency_eligible": True,
        }
    )
    healthy, messages, _ = monitor.assess(records, [])
    assert not healthy and "incomplete Play tick telemetry" in messages


def test_evaluator_ignores_nondefault_latency_and_triggers_play_or_fallback() -> None:
    records = _healthy_records()
    records.append(
        {
            "event": "watch_tick",
            "checkpoint": "b8-s3",
            "roster_size": 6,
            "speed": 24.0,
            "decisions": 0,
            "deaths": 0,
            "update_ms": 1000.0,
            "timestamp": 0.0,
            "latency_eligible": True,
        }
    )
    healthy, messages, _ = monitor.assess(records, [])
    assert healthy and messages == []

    records.append(
        {
            "event": "play_tick",
            "checkpoint": "b8-s3",
            "driven_ai": 5,
            "speed": 12.0,
            "update_ms": 9.0,
            "latency_eligible": True,
        }
    )
    healthy, messages, _ = monitor.assess(records, [])
    assert not healthy and any("play5 update p95" in message for message in messages)

    records = _healthy_records()
    records.append({"event": "ego2s_fallback", "reason": "empty dispatch queue"})
    healthy, messages, metrics = monitor.assess(records, [])
    assert not healthy and metrics["fallback_records"] == 1
    assert any("fallback" in message for message in messages)


def test_window_completes_only_after_both_time_and_decision_conditions() -> None:
    records = _healthy_records()
    healthy, _messages, metrics = monitor.assess(records, [], now=8 * 86400)
    assert healthy and metrics["window_complete"] is False

    tick = dict(records[1])
    records = [records[0]] + [tick] * 100000
    healthy, _messages, metrics = monitor.assess(records, [], now=8 * 86400)
    assert healthy and metrics["window_complete"] is True
