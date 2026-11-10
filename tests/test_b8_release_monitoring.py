"""Focused checks for the post-release B8 log evaluator."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from scripts import monitor_b8_release as monitor
from src.model.obs_spec import EGO2S_B
from web.backend import session as session_mod
from web.backend.served_checkpoint import B8_S3_SHA256, ServedCheckpointChoice
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
            "roster_mode": "all_b8",
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
        "roster_mode": "all_b8",
    }
    assert metrics["latency_by_roster"]["watch6"]["update_p95_ms"] == 8.0
    assert metrics["latency_by_roster"]["watch6"]["trigger_ms"] == (
        1.5 * monitor.UPDATE_P95_BASELINE_MS[("watch", 6)]
    )


def test_legacy_watch_record_without_a_roster_mode_is_all_b8() -> None:
    records = _healthy_records()
    del records[1]["roster_mode"]
    healthy, messages, metrics = monitor.assess(records, [])

    assert healthy and messages == []
    assert set(metrics["deaths_by_roster"]) == {"watch6"}


def test_any_fallback_or_ego2s_exception_requires_rollback() -> None:
    records = _healthy_records()
    records[0]["fallback_reason"] = "b8-s3 bytes did not verify"
    records.append({"event": "ego2s_exception", "error_type": "RuntimeError"})

    healthy, messages, metrics = monitor.assess(records, [])

    assert not healthy
    assert metrics["fallback_records"] == 1 and metrics["ego2s_exceptions"] == 1
    assert any("fallback" in message for message in messages)
    assert any("ego2s-path exception" in message for message in messages)


def test_real_mixed_refusal_is_durable_and_healthy_all_b8_ticks_cannot_hide_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hero = SimpleNamespace(policy=object(), ai=object(), safety_veto=None)
    opponent = SimpleNamespace(policy=object(), ai=object(), safety_veto=None)
    session = GameSession.__new__(GameSession)
    session.mode = MODE_WATCH
    session.game = SimpleNamespace(snakes=[hero, opponent])
    session.policy = hero.policy
    session._release_monitoring = True
    session.served_checkpoint = {"name": "b8-s3", "path": "/test/b8.pth"}
    session.checkpoint_path = "/test/b8.pth"
    session.obs_spec = EGO2S_B
    session.watch_opponent_checkpoint = None
    session.watch_opponent_policy = None
    events: list[dict] = []
    monkeypatch.setattr(
        session_mod,
        "resolve_mixed_watch_opponent",
        lambda: ServedCheckpointChoice(requested="frp3-s12", reason="bad pin"),
    )
    monkeypatch.setattr(
        session_mod,
        "mixed_roster_fallback",
        lambda **kwargs: events.append({"event": "mixed_roster_fallback", **kwargs}),
    )

    session._install_mixed_watch_roster(B8_S3_SHA256)
    healthy, messages, metrics = monitor.assess(_healthy_records() + events, [])

    assert session.watch_roster == "all_b8"
    assert events == [
        {
            "event": "mixed_roster_fallback",
            "reason": "mixed Watch roster refused: FRP3 opponent pin/bytes failed: bad pin",
        }
    ]
    assert not healthy
    assert metrics["fallback_records"] == 1
    assert any("fallback" in message for message in messages)


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
        "roster_mode": "all_b8",
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
            "roster_mode": "all_b8",
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
            "roster_mode": "all_b8",
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
            "roster_mode": "all_b8",
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


def _mixed_watch_records(*, roster_size: int = 6, update_ms: float = 7.0) -> list[dict]:
    records = _healthy_records()
    records[1].update(roster_mode="mixed", roster_size=roster_size, update_ms=update_ms)
    return records


def test_mixed_watch6_uses_t15_death_and_retained_latency_baselines() -> None:
    healthy, messages, metrics = monitor.assess(_mixed_watch_records(), [])

    assert healthy and messages == []
    assert metrics["deaths_by_roster"]["mixed_watch6"]["baseline_deaths_per_1000"] == 0.272
    assert metrics["deaths_by_roster"]["mixed_watch6"]["trigger_per_1000"] == 0.544
    assert metrics["latency_by_roster"]["mixed_watch6"] == {
        "update_p95_ms": 7.0,
        "qualification_update_p95_ms": 7.792219356633723,
        "trigger_ms": 11.688329034950584,
        "qualification_bound_ms": 66.667,
    }


@pytest.mark.parametrize("roster_size", [8, 12])
def test_mixed_watch_n8_and_n12_deaths_fail_closed_without_invented_baselines(
    roster_size: int,
) -> None:
    healthy, messages, metrics = monitor.assess(_mixed_watch_records(roster_size=roster_size), [])

    label = f"mixed_watch{roster_size}"
    assert not healthy
    assert metrics["deaths_by_roster"][label]["baseline_deaths_per_1000"] is None
    assert metrics["deaths_by_roster"][label]["trigger_per_1000"] is None
    assert metrics["latency_by_roster"][label]["qualification_update_p95_ms"] == (
        9.519156138412654 if roster_size == 8 else 13.141333474777639
    )
    assert f"no registered Watch death baseline for {label}" in messages


def _mixed_death_rate_records(deaths: int) -> list[dict]:
    dead = _mixed_watch_records()[1]
    dead.update(deaths=1)
    survived = {**dead, "deaths": 0}
    return [_healthy_records()[0], *([dead] * deaths), *([survived] * (125_000 - deaths))]


def test_mixed_watch_death_trigger_is_strictly_greater() -> None:
    healthy, messages, _ = monitor.assess(_mixed_death_rate_records(68), [])
    assert healthy and messages == []  # Equality at 0.544/1k is not a rollback.

    healthy, messages, _ = monitor.assess(_mixed_death_rate_records(69), [])
    assert not healthy
    assert any(
        "mixed_watch6 deaths/1k" in message and "exceeds 0.544" in message for message in messages
    )


def test_mode_and_size_observation_windows_cannot_be_pooled() -> None:
    all_b8 = _healthy_records()[1]
    all_b8.update(timestamp=0.0)
    mixed = dict(_mixed_watch_records()[1])
    mixed.update(decisions=1, timestamp=0.0)
    healthy, messages, metrics = monitor.assess(
        [_healthy_records()[0], *([all_b8] * 100_000), mixed], [], now=8 * 86400
    )

    assert healthy and messages == []
    assert metrics["windows_by_roster"]["watch6"]["window_complete"] is True
    assert metrics["windows_by_roster"]["mixed_watch6"]["window_complete"] is False
    assert metrics["window_complete"] is False


def test_qualified_mixed_latency_group_cannot_be_hidden_by_all_b8_samples() -> None:
    records = _healthy_records()
    mixed = dict(_mixed_watch_records()[1])
    mixed["latency_eligible"] = False
    records.append(mixed)

    healthy, messages, _ = monitor.assess(records, [])

    assert not healthy
    assert "missing default-speed eligible Watch update telemetry for mixed_watch6" in messages


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
    assert events[0]["roster_mode"] == "all_b8"


def test_mixed_watch_b8_session_emits_partitioning_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    hero = SimpleNamespace(id=0, is_alive=True, update=lambda *_args: None)
    game = SimpleNamespace(snakes=[hero], frame_death_causes={})
    game.update = lambda **_kwargs: hero.update([], [])
    events: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(session_mod, "watch_tick", lambda **kwargs: events.append(kwargs))
    session = _b8_session(game)
    session.watch_roster = "mixed"
    session.step()

    assert events[0]["roster_mode"] == "mixed"
    assert events[0]["decisions"] == 1


def test_mixed_watch_monitoring_stays_on_b8_slot_zero_when_inspecting_an_opponent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b8 = SimpleNamespace(id=0, is_alive=True, update=lambda *_args: None)
    opponent = SimpleNamespace(id=1, is_alive=True, update=lambda *_args: None)
    game = SimpleNamespace(snakes=[b8, opponent], frame_death_causes={0: "wall"})
    game.update = lambda **_kwargs: (b8.update([], []), b8.update([], []), opponent.update([], []))
    events: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(session_mod, "watch_tick", lambda **kwargs: events.append(kwargs))
    session = _b8_session(game)
    session.watch_roster = "mixed"
    session.hero_id = 1  # Inspector selection; not the fixed B8 monitoring hero.
    session.step()

    assert events[0]["decisions"] == 2
    assert events[0]["deaths"] == 1


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


def _healthy_replay_records(
    *,
    frames: int = 50_000,
    update_p95_ms: float = 0.05,
    first_timestamp: float = 0.0,
    timestamp: float = 0.0,
) -> list[dict]:
    return [
        {
            "event": "resolution",
            "requested": None,
            "name": "b8-s3",
            "fallback_reason": None,
        },
        {
            "event": "watch_replay_summary",
            "checkpoint": "b8-s3",
            "roster_mode": "mixed",
            "roster_size": 6,
            "controller": "exact-stop-l1-os",
            "speed": 24.0,
            "frames": frames,
            "update_p95_ms": update_p95_ms,
            "max_update_ms": update_p95_ms,
            "mean_update_ms": update_p95_ms,
            "first_timestamp": first_timestamp,
            "timestamp": timestamp,
        },
    ]


def test_replay_window_reports_incomplete_rather_than_healthy_below_bars(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    below_frames = _healthy_replay_records(frames=49_999)
    healthy, messages, metrics = monitor.assess(below_frames, [], now=8 * 86400)
    assert not healthy
    assert metrics["replay_mode"] is True
    assert metrics["replay_frames"] == 49_999
    assert metrics["window_complete"] is False
    assert metrics["replay_window_complete"] is False
    assert any("incomplete Watch replay observation window" in msg for msg in messages)

    below_days = _healthy_replay_records(frames=50_000)
    healthy, messages, metrics = monitor.assess(below_days, [], now=6 * 86400)
    assert not healthy
    assert metrics["window_complete"] is False
    assert any("incomplete Watch replay observation window" in msg for msg in messages)

    complete = _healthy_replay_records(frames=50_000)
    healthy, messages, metrics = monitor.assess(complete, [], now=7 * 86400)
    assert healthy and messages == []
    assert metrics["window_complete"] is True
    assert metrics["replay_window_complete"] is True

    incomplete_log = tmp_path / "incomplete.log"
    incomplete_log.write_text(
        "\n".join(f"b8-release-monitor: {__import__('json').dumps(r)}" for r in below_frames)
        + "\n",
        encoding="utf-8",
    )
    assert monitor.main([str(incomplete_log)]) == 2
    out = capsys.readouterr().out
    assert "incomplete Watch replay observation window" in out
    assert out.strip().endswith("NOT_HEALTHY")

    complete_log = tmp_path / "complete.log"
    complete_log.write_text(
        "\n".join(f"b8-release-monitor: {__import__('json').dumps(r)}" for r in complete) + "\n",
        encoding="utf-8",
    )
    assert monitor.main([str(complete_log)]) == 0
    assert capsys.readouterr().out.strip().endswith("HEALTHY")


def test_replay_monitor_flags_each_rollback_trigger() -> None:
    # 1. Fallback reason (resolution, ego2s, mixed Watch, or replay fallback)
    for fb_event in (
        {"event": "ego2s_fallback", "reason": "empty queue"},
        {"event": "mixed_roster_fallback", "reason": "bad opponent pin"},
        {"event": "replay_fallback", "reason": "replay fallback"},
    ):
        records = _healthy_replay_records() + [fb_event]
        healthy, messages, metrics = monitor.assess(records, [], now=8 * 86400)
        assert not healthy and metrics["fallback_records"] == 1
        assert any("ROLLBACK: recorded served-checkpoint or ego2s fallback" in m for m in messages)

    # 2. Replay step failure
    records = _healthy_replay_records() + [
        {
            "event": "replay_step_failure",
            "checkpoint": "b8-s3",
            "roster_mode": "mixed",
            "roster_size": 6,
            "controller": "exact-stop-l1-os",
            "stage": "step",
            "error_type": "ValueError",
            "reason": "corrupt frame",
            "timestamp": 1.0,
        }
    ]
    healthy, messages, metrics = monitor.assess(records, [], now=8 * 86400)
    assert not healthy and metrics["replay_step_failures"] == 1
    assert any("ROLLBACK: recorded Watch replay step failure" in m for m in messages)

    # 3. ego2s exception
    records = _healthy_replay_records() + [
        {"event": "ego2s_exception", "checkpoint": "b8-s3", "error_type": "RuntimeError"}
    ]
    healthy, messages, metrics = monitor.assess(records, [], now=8 * 86400)
    assert not healthy and metrics["ego2s_exceptions"] == 1
    assert any("ROLLBACK: recorded ego2s-path exception" in m for m in messages)

    # 4. Replay update p95 > 41.67 ms (strictly greater)
    at_bound = _healthy_replay_records(update_p95_ms=41.67)
    healthy, messages, _ = monitor.assess(at_bound, [], now=8 * 86400)
    assert healthy and messages == []

    over_bound = _healthy_replay_records(update_p95_ms=41.68)
    healthy, messages, metrics = monitor.assess(over_bound, [], now=8 * 86400)
    assert not healthy and metrics["replay_update_p95_ms"] == 41.68
    assert any("ROLLBACK: Watch replay update p95" in m and "41.67" in m for m in messages)

    # 5. Bank exhaustion
    records = _healthy_replay_records() + [
        {
            "event": "replay_bank_exhausted",
            "checkpoint": "b8-s3",
            "roster_mode": "mixed",
            "roster_size": 6,
            "reason": "No verified replay without an immediate repeat; using live B8.",
            "timestamp": 1.0,
        }
    ]
    healthy, messages, metrics = monitor.assess(records, [], now=8 * 86400)
    assert not healthy and metrics["replay_bank_exhaustions"] == 1
    assert any("ROLLBACK: Watch replay bank exhausted" in m for m in messages)


def test_replay_monitor_evaluates_live_bars_on_live_fallback_frames_only() -> None:
    # Cold-startup 24 fps live-fallback ticks without 12 fps latency samples or 100k decisions
    # remain healthy when the replay window is complete and live-fallback bars are not breached.
    cold_tick = {
        "event": "watch_tick",
        "checkpoint": "b8-s3",
        "roster_mode": "mixed",
        "roster_size": 6,
        "speed": 24.0,
        "decisions": 1,
        "deaths": 0,
        "update_ms": 8.5,
        "timestamp": 0.0,
        "latency_eligible": False,
    }
    records = _healthy_replay_records() + [cold_tick] * 5
    healthy, messages, metrics = monitor.assess(records, [], now=8 * 86400)
    assert healthy and messages == []
    assert metrics["window_complete"] is True
    assert metrics["deaths_by_roster"]["mixed_watch6"]["deaths_per_1000"] == 0.0

    # Breaching the live death bar on live-fallback frames triggers rollback.
    lethal_fallback = dict(cold_tick, deaths=1)
    healthy, messages, _ = monitor.assess(
        _healthy_replay_records() + [lethal_fallback], [], now=8 * 86400
    )
    assert not healthy
    assert any("ROLLBACK: mixed_watch6 deaths/1k" in m for m in messages)

    # Breaching the live 12 fps latency bar on eligible live-fallback frames triggers rollback.
    slow_live_fallback = dict(cold_tick, speed=12.0, latency_eligible=True, update_ms=15.0)
    healthy, messages, _ = monitor.assess(
        _healthy_replay_records() + [slow_live_fallback], [], now=8 * 86400
    )
    assert not healthy
    assert any("ROLLBACK: mixed_watch6 update p95" in m for m in messages)


def test_session_replay_telemetry_aggregates_periodically_and_emits_faults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries: list[dict] = []
    failures: list[dict] = []
    exhaustions: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(
        session_mod, "watch_replay_summary", lambda **kwargs: summaries.append(kwargs)
    )
    monkeypatch.setattr(
        session_mod, "watch_replay_failure", lambda **kwargs: failures.append(kwargs)
    )
    monkeypatch.setattr(
        session_mod, "watch_replay_bank_exhausted", lambda **kwargs: exhaustions.append(kwargs)
    )
    monkeypatch.setattr(session_mod, "watch_tick", lambda **_kwargs: None)

    session = _b8_session(
        SimpleNamespace(snakes=[], frame_death_causes={}, update=lambda **_kw: None)
    )
    session.watch_roster = "mixed"
    session.speed = 24.0
    session._watch_replay = SimpleNamespace(
        header={"controller": "exact-stop-l1-os"},
        step=lambda: True,
        snapshot=lambda _s: {"frame": 1},
        close=lambda: None,
    )
    session._replay_bank = None

    for _ in range(session.REPLAY_TELEMETRY_SUMMARY_FRAMES - 1):
        session.step()
    assert summaries == []

    session.step()
    assert len(summaries) == 1
    assert summaries[0]["frames"] == session.REPLAY_TELEMETRY_SUMMARY_FRAMES
    assert summaries[0]["controller"] == "exact-stop-l1-os"
    assert summaries[0]["roster_mode"] == "mixed"

    # Partial batch flushes on leaving replay or failure.
    session.step()
    session.step()
    assert len(summaries) == 1
    session._watch_replay.step = lambda: (_ for _ in ()).throw(ValueError("bad replay frame"))
    session.step()
    assert len(summaries) == 2 and summaries[1]["frames"] == 2
    assert len(failures) == 1 and failures[0]["stage"] == "step"

    # Snapshot failure and bank exhaustion are recorded.
    session._watch_replay = SimpleNamespace(
        header={"controller": "exact-stop-l1-os"},
        step=lambda: False,
        snapshot=lambda _s: (_ for _ in ()).throw(RuntimeError("bad snapshot")),
        close=lambda: None,
    )
    monkeypatch.setattr("web.backend.serialize.build_frame", lambda _s: {"frame": 0})
    session.snapshot()
    assert len(failures) == 2 and failures[1]["stage"] == "snapshot"

    session._watch_replay = SimpleNamespace(
        header={"controller": "exact-stop-l1-os"},
        step=lambda: False,
        close=lambda: None,
    )
    session.step()
    assert len(exhaustions) == 1


def test_every_telemetry_hook_raising_produces_identical_served_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web.backend import release_monitoring

    def _make_replay(name: str, total_steps: int) -> SimpleNamespace:
        state = {"frame": 0, "closed": False}

        def _step() -> bool:
            if state["frame"] >= total_steps:
                return False
            state["frame"] += 1
            return True

        def _snapshot(_s: object) -> dict:
            return {"source": "replay", "name": name, "frame": state["frame"]}

        def _close() -> None:
            state["closed"] = True

        return SimpleNamespace(
            header={"controller": "exact-stop-l1-os"},
            state=state,
            step=_step,
            snapshot=_snapshot,
            close=_close,
        )

    def _build_pair(total_steps: int = 602):
        pairs = []
        for monitored in (False, True):
            live_state = {"frame": 0}
            hero = SimpleNamespace(id=0, is_alive=True, update=lambda *_a: None)

            def _update(_ls=live_state, _h=hero, **_kw) -> None:
                _ls["frame"] += 1
                _h.update([], [])

            game = SimpleNamespace(snakes=[hero], frame_death_causes={}, update=_update)
            sess = _b8_session(game)
            sess._release_monitoring = monitored
            sess.watch_roster = "mixed"
            sess.speed = 24.0
            sess.live_state = live_state
            r1 = _make_replay("r1", total_steps)
            r2 = _make_replay("r2", 2)
            queue = [r2]
            sess._watch_replay = r1
            sess._replay_bank = SimpleNamespace(
                pick=lambda _q=queue: _q.pop(0) if _q else None,
                pending=False,
                error=None,
            )
            pairs.append((sess, r1, r2))
        return pairs[0], pairs[1]

    monkeypatch.setattr(
        "web.backend.serialize.build_frame",
        lambda s: {"source": "live", "frame": s.live_state["frame"]},
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("telemetry hook boom")

    for hook in (
        "served_resolution",
        "mixed_roster_fallback",
        "play_mixed_roster_fallback",
        "watch_tick",
        "play_tick",
        "ego2s_exception",
        "watch_replay_active",
        "watch_replay_summary",
        "watch_replay_failure",
        "watch_replay_bank_exhausted",
    ):
        monkeypatch.setattr(session_mod, hook, _boom)
        monkeypatch.setattr(release_monitoring, hook, _boom)
    monkeypatch.setattr(release_monitoring, "emit", _boom)

    (base, base_r1, base_r2), (mon, mon_r1, mon_r2) = _build_pair(total_steps=602)

    base_frames = []
    mon_frames = []
    # Step across the 600-frame summary flush boundary, replay rotation (r1 -> r2),
    # and bank exhaustion fallback to live B8.
    for _ in range(606):
        base.step()
        mon.step()
        base_frames.append(base.snapshot())
        mon_frames.append(mon.snapshot())

    assert mon_frames == base_frames
    assert mon_r1.state["closed"] is True and base_r1.state["closed"] is True
    assert mon_r2.state["closed"] is True and base_r2.state["closed"] is True
    assert mon._watch_replay is None and base._watch_replay is None

    # Verify step/snapshot failures and _leave_watch_replay still close/clear _watch_replay
    # even when _flush_watch_replay_telemetry and _record_watch_replay_failure raise directly.
    monkeypatch.setattr(mon, "_flush_watch_replay_telemetry", _boom)
    monkeypatch.setattr(mon, "_record_watch_replay_failure", _boom)
    monkeypatch.setattr(mon, "_record_watch_replay_step", _boom)

    failing_step_replay = _make_replay("fail_step", 5)
    failing_step_replay.step = lambda: (_ for _ in ()).throw(ValueError("corrupt step"))
    mon._watch_replay = failing_step_replay
    base_fail_step = _make_replay("fail_step", 5)
    base_fail_step.step = lambda: (_ for _ in ()).throw(ValueError("corrupt step"))
    base._watch_replay = base_fail_step

    base.step()
    mon.step()
    assert mon.snapshot() == base.snapshot()
    assert failing_step_replay.state["closed"] is True
    assert mon._watch_replay is None
    assert mon.last_error == base.last_error

    failing_snap_replay = _make_replay("fail_snap", 5)
    failing_snap_replay.snapshot = lambda _s: (_ for _ in ()).throw(RuntimeError("corrupt snap"))
    mon._watch_replay = failing_snap_replay
    base_fail_snap = _make_replay("fail_snap", 5)
    base_fail_snap.snapshot = lambda _s: (_ for _ in ()).throw(RuntimeError("corrupt snap"))
    base._watch_replay = base_fail_snap

    assert mon.snapshot() == base.snapshot()
    assert failing_snap_replay.state["closed"] is True
    assert mon._watch_replay is None
    assert mon.last_error == base.last_error

    leave_replay = _make_replay("leave", 5)
    mon._watch_replay = leave_replay
    mon._leave_watch_replay()
    assert leave_replay.state["closed"] is True
    assert mon._watch_replay is None


def test_small_partial_replay_flushes_cannot_set_window_p95() -> None:
    resolution = _healthy_replay_records()[0]
    full_batches = [
        {
            "event": "watch_replay_summary",
            "checkpoint": "b8-s3",
            "roster_mode": "mixed",
            "roster_size": 6,
            "controller": "exact-stop-l1-os",
            "speed": 24.0,
            "frames": 600,
            "update_p95_ms": 0.05,
            "max_update_ms": 0.1,
            "mean_update_ms": 0.04,
            "first_timestamp": float(i * 600),
            "timestamp": float((i + 1) * 600),
        }
        for i in range(84)
    ]
    partial_one_frame = {
        "event": "watch_replay_summary",
        "checkpoint": "b8-s3",
        "roster_mode": "mixed",
        "roster_size": 6,
        "controller": "exact-stop-l1-os",
        "speed": 24.0,
        "frames": 1,
        "update_p95_ms": 85.0,
        "max_update_ms": 85.0,
        "mean_update_ms": 85.0,
        "first_timestamp": 60_000.0,
        "timestamp": 60_001.0,
    }
    partial_fifteen_frames = {
        **partial_one_frame,
        "frames": 15,
        "update_p95_ms": 60.0,
        "max_update_ms": 60.0,
    }

    healthy, messages, metrics = monitor.assess(
        [resolution, *full_batches, partial_one_frame, partial_fifteen_frames],
        [],
        now=8 * 86400,
    )
    assert healthy and messages == []
    assert metrics["replay_frames"] == 84 * 600 + 16
    assert metrics["replay_update_p95_ms"] == 0.05

    # If >= 5% of full-window frames breach the 41.67 ms trigger, rollback triggers.
    slow_batches = [
        {**full_batches[0], "update_p95_ms": 50.0, "max_update_ms": 55.0} for _ in range(5)
    ]
    healthy, messages, metrics = monitor.assess(
        [resolution, *full_batches[:79], *slow_batches],
        [],
        now=8 * 86400,
    )
    assert not healthy
    assert metrics["replay_update_p95_ms"] == 50.0
    assert any("ROLLBACK: Watch replay update p95" in m for m in messages)


def test_replay_selection_emits_presence_and_prevents_premature_healthy_before_first_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(
        session_mod,
        "watch_replay_active",
        lambda **kwargs: emitted.append(
            {"event": "watch_replay_active", "timestamp": 1.0, **kwargs}
        ),
    )
    monkeypatch.setattr(
        session_mod,
        "watch_replay_summary",
        lambda **kwargs: emitted.append(
            {"event": "watch_replay_summary", "timestamp": 2.0, **kwargs}
        ),
    )
    monkeypatch.setattr(
        session_mod,
        "watch_tick",
        lambda **kwargs: emitted.append({"event": "watch_tick", "timestamp": 1.5, **kwargs}),
    )

    hero = SimpleNamespace(id=0, is_alive=True, update=lambda *_a: None)
    game = SimpleNamespace(
        snakes=[hero], frame_death_causes={}, update=lambda **_kw: hero.update([], [])
    )
    session = _b8_session(game)
    session.watch_roster = "mixed"
    session.speed = 12.0

    # 1. Cold replay-bank verification (_replay_pending=True) ticking live B8.
    session._replay_bank = SimpleNamespace(pick=lambda: None, pending=True, error=None)
    session._select_watch_replay()
    session.step()
    assert any(e["event"] == "watch_replay_active" and e["status"] == "pending" for e in emitted)

    healthy, messages, metrics = monitor.assess(
        [_healthy_replay_records()[0], *emitted], [], now=8 * 86400
    )
    assert not healthy
    assert metrics["replay_mode"] is True
    assert metrics["window_complete"] is False
    assert any("incomplete Watch replay observation window" in m for m in messages)

    # 2. Replay selected and stepped for < 600 frames (before any watch_replay_summary flush).
    emitted.clear()
    replay = SimpleNamespace(
        header={"controller": "exact-stop-l1-os"},
        step=lambda: True,
        snapshot=lambda _s: {"frame": 1},
        close=lambda: None,
    )
    session._replay_bank = SimpleNamespace(pick=lambda: replay, pending=False, error=None)
    session._select_watch_replay()
    for _ in range(10):
        session.step()
    assert [e["event"] for e in emitted] == ["watch_replay_active"]
    assert emitted[0]["controller"] == "exact-stop-l1-os"
    assert emitted[0]["status"] == "active"

    healthy, messages, metrics = monitor.assess(
        [_healthy_replay_records()[0], *emitted], [], now=8 * 86400
    )
    assert not healthy
    assert metrics["replay_mode"] is True
    assert metrics["replay_frames"] == 0
    assert metrics["window_complete"] is False
    assert any("incomplete Watch replay observation window" in m for m in messages)


def test_replay_telemetry_snapshots_batch_attributes_at_batch_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda snakes: len(snakes))
    monkeypatch.setattr(
        session_mod, "watch_replay_summary", lambda **kwargs: summaries.append(kwargs)
    )
    monkeypatch.setattr(session_mod, "watch_replay_active", lambda **_kwargs: None)

    session = _b8_session(
        SimpleNamespace(
            snakes=[SimpleNamespace()] * 6, frame_death_causes={}, update=lambda **_kw: None
        )
    )
    session.watch_roster = "mixed"
    session.speed = 24.0
    session._watch_replay = SimpleNamespace(
        header={"controller": "exact-stop-l1-os"},
        step=lambda: True,
        snapshot=lambda _s: {"frame": 1},
        close=lambda: None,
    )
    session._replay_bank = None

    session.step()
    session.step()
    session.step()
    assert summaries == []

    # Mutate session attributes before flushing; buffered batch must retain step-time metadata.
    session.speed = 12.0
    session.watch_roster = "all_b8"
    session.game.snakes = [SimpleNamespace()] * 12
    session._watch_replay = SimpleNamespace(
        header={"controller": "mutated-controller"},
        step=lambda: True,
        close=lambda: None,
    )

    # Stepping with changed attributes flushes the prior batch with its snapshotted values
    # and opens a new batch with the updated values.
    session.step()
    assert len(summaries) == 1
    assert summaries[0]["frames"] == 3
    assert summaries[0]["roster_mode"] == "mixed"
    assert summaries[0]["roster_size"] == 6
    assert summaries[0]["controller"] == "exact-stop-l1-os"
    assert summaries[0]["speed"] == 24.0

    # Mutate again before _leave_watch_replay() flush.
    session.speed = 6.0
    session.watch_roster = "mixed"
    session.game.snakes = [SimpleNamespace()] * 8
    session._leave_watch_replay()
    assert len(summaries) == 2
    assert summaries[1]["frames"] == 1
    assert summaries[1]["roster_mode"] == "all_b8"
    assert summaries[1]["roster_size"] == 12
    assert summaries[1]["controller"] == "mutated-controller"
    assert summaries[1]["speed"] == 12.0


def test_mixed_play_ticks_do_not_fault_or_pollute_all_b8_play_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.game.ai_snake import AISnake
    from web.backend import release_monitoring
    from web.backend.served_checkpoint import (
        PLAY_ROSTER_ALL_B8,
        PLAY_ROSTER_MIXED,
        PLAY_ROSTERS,
    )

    assert monitor.PLAY_ROSTER_ALL_B8 == PLAY_ROSTER_ALL_B8
    assert monitor.PLAY_ROSTER_MIXED == PLAY_ROSTER_MIXED
    assert monitor.PLAY_ROSTERS == PLAY_ROSTERS

    emitted: list[dict] = []
    monkeypatch.setattr(
        release_monitoring,
        "emit",
        lambda event, **fields: emitted.append({"event": event, "timestamp": 1.0, **fields}),
    )

    b8_policy = object()
    opp_policy = object()
    human = SimpleNamespace(id=0, is_alive=True, is_human=True, policy=None)
    b8_rival = AISnake.__new__(AISnake)
    b8_rival.id = 1
    b8_rival.is_alive = True
    b8_rival.policy = b8_policy
    frp_opps = []
    for idx in range(2, 6):
        opp = AISnake.__new__(AISnake)
        opp.id = idx
        opp.is_alive = True
        opp.policy = opp_policy
        frp_opps.append(opp)
    game = SimpleNamespace(
        snakes=[human, b8_rival, *frp_opps],
        frame_death_causes={},
        update=lambda **_kw: None,
    )
    session = _b8_session(game, mode=MODE_PLAY, warmup=0)
    session.policy = b8_policy
    session.play_roster = "mixed"
    session.run_started = True
    session.run_over = False
    session.run_frames = 0
    session._find_human = lambda: human

    session._step_play()
    assert len(emitted) == 1
    assert emitted[0]["event"] == "play_tick"
    assert emitted[0]["driven_ai"] == 1

    _, _, baseline_metrics = monitor.assess(_healthy_records(), [])
    assert baseline_metrics["play_ticks_excluded_by_roster"] == 0

    healthy, messages, metrics = monitor.assess([*_healthy_records(), *emitted], [])
    assert healthy and messages == []
    assert emitted[0]["roster_mode"] == "mixed"
    assert emitted[0]["roster_size"] == 5
    assert metrics["play_ticks_excluded_by_roster"] == 1
    assert "play1" not in metrics["latency_by_roster"]
    assert "play5" not in metrics["latency_by_roster"]

    # Malformed play_tick roster_mode still fails closed, validating against PLAY_ROSTERS
    # rather than WATCH_ROSTERS.
    bad_play = {**emitted[0], "roster_mode": "unknown_roster"}
    healthy, messages, _ = monitor.assess([*_healthy_records(), bad_play], [])
    assert not healthy and "incomplete Play tick telemetry" in messages

    monkeypatch.setattr(monitor, "WATCH_ROSTERS", ("all_b8", "mixed", "watch_only_test"))
    monkeypatch.setattr(monitor, "PLAY_ROSTERS", ("all_b8", "mixed", "play_only_test"))
    watch_only_tick = {**emitted[0], "roster_mode": "watch_only_test"}
    healthy, messages, _ = monitor.assess([*_healthy_records(), watch_only_tick], [])
    assert not healthy and "incomplete Play tick telemetry" in messages
    play_only_tick = {**emitted[0], "roster_mode": "play_only_test"}
    healthy, messages, metrics = monitor.assess([*_healthy_records(), play_only_tick], [])
    assert healthy and messages == []
    assert metrics["play_ticks_excluded_by_roster"] == 1


def test_registered_replay_window_p95_estimator_and_metric_only_over_trigger_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web.backend import release_monitoring

    # F3 / N2: local constant in web/backend avoids importing unshipped scripts/,
    # pinned by three-way equality across monitor, release_monitoring, and session.
    assert (
        release_monitoring.REPLAY_UPDATE_P95_TRIGGER_MS
        == monitor.REPLAY_UPDATE_P95_TRIGGER_MS
        == session_mod.REPLAY_UPDATE_P95_TRIGGER_MS
    )

    resolution = _healthy_replay_records()[0]

    # F1: registered estimator prefers batches >= REPLAY_SUMMARY_FULL_FRAMES (600)
    # when present, falling back to >= REPLAY_SUMMARY_MIN_P95_FRAMES (20) only when
    # no full 600-frame batch exists.
    partial_400 = [
        {
            "event": "watch_replay_summary",
            "checkpoint": "b8-s3",
            "roster_mode": "mixed",
            "roster_size": 6,
            "controller": "exact-stop-l1-os",
            "speed": 24.0,
            "frames": 400,
            "update_p95_ms": 80.0,
            "max_update_ms": 90.0,
            "mean_update_ms": 75.0,
            "first_timestamp": float(i * 400),
            "timestamp": float((i + 1) * 400),
        }
        for i in range(125)
    ]
    one_fast_600 = {
        **partial_400[0],
        "frames": 600,
        "update_p95_ms": 3.0,
        "max_update_ms": 4.0,
        "mean_update_ms": 2.0,
        "first_timestamp": 50_000.0,
        "timestamp": 50_600.0,
    }
    healthy, messages, metrics = monitor.assess(
        [resolution, *partial_400, one_fast_600], [], now=8 * 86400
    )
    assert healthy and messages == []
    assert metrics["replay_window_complete"] is True
    assert metrics["replay_update_p95_ms"] == 3.0
    assert monitor.replay_window_p95([(400, 80.0)], []) == 80.0
    assert monitor.replay_window_p95([(30, 1.0)], [80.0] * 30) == 1.0

    # F2: over_trigger_frames is reported in metrics only, with no new ROLLBACK
    # trigger or verdict impact when the registered replay_update_p95_ms bar passes.
    stealth_batches = [
        {
            **one_fast_600,
            "update_p95_ms": 0.05,
            "max_update_ms": 55.0,
            "over_trigger_frames": 29,
            "first_timestamp": float(i * 600),
            "timestamp": float((i + 1) * 600),
        }
        for i in range(96)
    ]
    stalled_batches = [
        {
            **one_fast_600,
            "update_p95_ms": 50.0,
            "max_update_ms": 55.0,
            "over_trigger_frames": 600,
            "first_timestamp": float((96 + i) * 600),
            "timestamp": float((97 + i) * 600),
        }
        for i in range(4)
    ]
    healthy, messages, metrics = monitor.assess(
        [resolution, *stealth_batches, *stalled_batches], [], now=8 * 86400
    )
    assert healthy and messages == []
    assert metrics["replay_over_trigger_frames"] == 5184
    assert metrics["replay_over_trigger_invalid"] == 0

    # N3: Malformed over_trigger_frames must never fault, change verdict, truncate floats,
    # or crash assess (bad type, bool, negative, > frames, float, Infinity, -Infinity, NaN).
    bad_over_trigger_values: list[object] = [
        "bad",
        "12",
        None,
        True,
        False,
        [1],
        {"count": 1},
        -1,
        601,
        3.7,
        3.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    base_healthy_replay = _healthy_replay_records()
    healthy_plain, plain_messages, plain_metrics = monitor.assess(
        base_healthy_replay, [], now=8 * 86400
    )
    assert healthy_plain and plain_messages == []
    assert plain_metrics["replay_over_trigger_frames"] == 0
    assert plain_metrics["replay_over_trigger_invalid"] == 0

    for idx, bad_value in enumerate(bad_over_trigger_values):
        bad_record = {
            **one_fast_600,
            "frames": 600,
            "over_trigger_frames": bad_value,
            "first_timestamp": float(700_000 + idx * 600),
            "timestamp": float(700_600 + idx * 600),
        }
        ok, msgs, bad_metrics = monitor.assess(
            [*base_healthy_replay, bad_record], [], now=9 * 86400
        )
        assert ok is True, f"bad value {bad_value!r} changed verdict: {msgs}"
        assert msgs == [], f"bad value {bad_value!r} added fault/message: {msgs}"
        assert bad_metrics["replay_over_trigger_frames"] == 0
        assert bad_metrics["replay_over_trigger_invalid"] == 1

    combined_bad = [
        {
            **one_fast_600,
            "frames": 600,
            "over_trigger_frames": bad_value,
            "first_timestamp": float(800_000 + idx * 600),
            "timestamp": float(800_600 + idx * 600),
        }
        for idx, bad_value in enumerate(bad_over_trigger_values)
    ]
    valid_edge_batches = [
        {
            **one_fast_600,
            "frames": 600,
            "over_trigger_frames": 0,
            "first_timestamp": 820_000.0,
            "timestamp": 820_600.0,
        },
        {
            **one_fast_600,
            "frames": 600,
            "over_trigger_frames": 17,
            "first_timestamp": 820_600.0,
            "timestamp": 821_200.0,
        },
    ]
    ok, msgs, combined_metrics = monitor.assess(
        [*base_healthy_replay, *combined_bad, *valid_edge_batches], [], now=10 * 86400
    )
    assert ok is True and msgs == []
    assert combined_metrics["replay_over_trigger_frames"] == 17
    assert combined_metrics["replay_over_trigger_invalid"] == len(bad_over_trigger_values)

    # R2: Non-summary records carrying over_trigger_frames (including as the first record)
    # are ignored by the watch_replay_summary over_trigger_frames block, and older
    # watch_replay_summary records lacking over_trigger_frames have update_val=None so
    # high batch update_p95_ms is not counted as 1 frame.
    non_summary_first_bad = {
        "event": "watch_replay_tick",
        "checkpoint": "b8-s3",
        "roster_mode": "mixed",
        "roster_size": 6,
        "speed": 24.0,
        "update_ms": 5.0,
        "timestamp": 1.0,
        "over_trigger_frames": "bad",
    }
    non_summary_tick_count = {
        **non_summary_first_bad,
        "timestamp": 2.0,
        "over_trigger_frames": 1,
    }
    non_summary_active = {
        "event": "watch_replay_active",
        "checkpoint": "b8-s3",
        "roster_mode": "mixed",
        "roster_size": 6,
        "speed": 24.0,
        "timestamp": 0.5,
        "over_trigger_frames": "bad",
    }
    ok, msgs, non_summary_metrics = monitor.assess(
        [non_summary_first_bad, non_summary_active, non_summary_tick_count, *base_healthy_replay],
        [],
        now=9 * 86400,
    )
    assert ok is True and msgs == []
    assert non_summary_metrics["replay_over_trigger_frames"] == 0
    assert non_summary_metrics["replay_over_trigger_invalid"] == 0

    legacy_fast = [
        {k: v for k, v in batch.items() if k != "over_trigger_frames"} for batch in stealth_batches
    ]
    legacy_high_p95_no_key = {
        k: v for k, v in stalled_batches[0].items() if k != "over_trigger_frames"
    }
    assert "over_trigger_frames" not in legacy_high_p95_no_key
    assert "update_ms" not in legacy_high_p95_no_key
    ok, msgs, legacy_metrics = monitor.assess(
        [resolution, *legacy_fast, legacy_high_p95_no_key], [], now=8 * 86400
    )
    assert ok is True and msgs == []
    assert legacy_metrics["replay_over_trigger_frames"] == 0
    assert legacy_metrics["replay_over_trigger_invalid"] == 0

    # Session flush still emits over_trigger_frames for telemetry reporting.
    summaries: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(
        session_mod, "watch_replay_summary", lambda **kwargs: summaries.append(kwargs)
    )
    monkeypatch.setattr(session_mod, "watch_replay_active", lambda **_kwargs: None)

    hero = SimpleNamespace(id=0, is_alive=True, update=lambda *_a: None)
    game = SimpleNamespace(
        snakes=[hero],
        frame_death_causes={0: "wall"},
        update=lambda **_kw: hero.update([], []),
    )
    session = _b8_session(game)
    session.watch_roster = "mixed"
    session.speed = 24.0

    session._record_watch_replay_step(
        SimpleNamespace(header={"controller": "exact-stop-l1-os"}), 50.0
    )
    session._record_watch_replay_step(
        SimpleNamespace(header={"controller": "exact-stop-l1-os"}), 10.0
    )
    session._flush_watch_replay_telemetry()
    assert len(summaries) == 1
    assert summaries[0]["over_trigger_frames"] == 1


def test_replay_opt_out_preserves_live_b8_watch_tick_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PinnedEgo2sPolicy:
        @property
        def epsilon(self) -> float:
            return 0.0

        @epsilon.setter
        def epsilon(self, _value: float) -> None:
            return

    ticks: list[dict] = []
    monkeypatch.setattr(session_mod, "roster_ai_count", lambda *_args: 6)
    monkeypatch.setattr(session_mod, "watch_tick", lambda **kwargs: ticks.append(kwargs))

    hero = SimpleNamespace(id=0, is_alive=True, update=lambda *_a: None)
    game = SimpleNamespace(
        snakes=[hero],
        frame_death_causes={0: "wall"},
        update=lambda **_kw: hero.update([], []),
    )
    session = _b8_session(game, warmup=0)
    session.policy = _PinnedEgo2sPolicy()
    session.watch_roster = "mixed"
    session.speed = 12.0
    session._replay_bank = SimpleNamespace(pick=lambda: None, pending=False, error=None)
    session._replay_opted_out = False

    session.step()
    assert len(ticks) == 1
    assert ticks[0]["decisions"] == 1
    assert ticks[0]["deaths"] == 1

    # Viewer opt-out on the pinned-epsilon B8 policy still runs clean live B8 and must
    # continue emitting watch_tick latency and hero-death telemetry.
    session.set_epsilon(0.5)
    assert session._replay_opted_out is True
    assert session._is_b8_release_monitoring() is True
    session.step()
    assert len(ticks) == 2
    assert ticks[1]["decisions"] == 1
    assert ticks[1]["deaths"] == 1
