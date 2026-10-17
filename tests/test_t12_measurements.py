"""Focused pure tests for the T12 serving measurement helpers."""

from __future__ import annotations

import os

import pytest

from research.t12_b8_serving_20261017 import measurements


def test_frontend_margin_matches_finite_sort_and_strict_thresholds():
    margin = measurements.frontend_decision_margin([float("nan"), 2.0, 1.8, -2.0])
    assert margin["raw"] == pytest.approx(0.2)
    assert margin["relative"] == pytest.approx(0.05)
    assert measurements.confidence_bucket(0.25) == "moderate"
    assert measurements.confidence_bucket(0.1) == "close"
    assert measurements.confidence_bucket(0.250001) == "confident"
    assert measurements.confidence_bucket(0.100001) == "moderate"


def test_frontend_margin_handles_flat_or_insufficient_values():
    assert measurements.frontend_decision_margin([4.0, 4.0, 4.0]) == {"raw": 0.0, "relative": 0.0}
    assert measurements.frontend_decision_margin([float("nan"), 3.0]) == {
        "raw": 0.0,
        "relative": 0.0,
    }


def test_distribution_preserves_recomputable_summary():
    summary = measurements._distribution([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["p50"] == 2.5
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["max"] == 4.0
    assert summary["mean"] == 2.5


def test_ui_speed_bound_uses_max_tick_and_slider_ceiling():
    assert measurements._highest_ui_speed(1.0) == 60
    assert measurements._highest_ui_speed(20.0) == 49
    assert measurements._highest_ui_speed(1000.0) == 1


def test_tmux_presence_blocks_without_mutation(monkeypatch):
    class Result:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    outcomes = iter([Result(0), Result(0, "0\t999\n")])
    monkeypatch.setattr(measurements.subprocess, "run", lambda *args, **kwargs: next(outcomes))
    with pytest.raises(measurements.LatencyBlocked, match="t11-lanes"):
        measurements._tmux_t11_absent()


def test_tmux_retained_session_with_only_dead_panes_is_allowed(monkeypatch):
    class Result:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    outcomes = iter([Result(0), Result(0, "1\t123\n1\t456\n")])
    monkeypatch.setattr(measurements.subprocess, "run", lambda *args, **kwargs: next(outcomes))
    result = measurements._tmux_t11_absent()
    assert result["present"] is True
    assert result["workload_panes_exited"] is True


def test_tmux_absence_requires_a_real_absence_diagnostic(monkeypatch):
    class Result:
        def __init__(self, code, stderr="", stdout=""):
            self.returncode = code
            self.stderr = stderr
            self.stdout = stdout

    outcomes = iter(
        [
            Result(1, "can't find session: t11-lanes"),
            Result(1, "no server running on /private/tmp/tmux-1/default"),
        ]
    )
    monkeypatch.setattr(measurements.subprocess, "run", lambda *args, **kwargs: next(outcomes))
    result = measurements._tmux_t11_absent()
    assert result["present"] is False
    assert result["has_session_returncode"] == 1


def test_serving_environment_clears_veto_overrides_and_restores(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "champion")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v7")
    monkeypatch.setenv("SNAKE_SERVE_VETO_WATCH_HERO", "1")
    with measurements._served_b8_environment():
        assert os.environ["SNAKE_SERVE_CHECKPOINT"] == "b8-s3"
        assert "SNAKE_SERVE_VETO_VARIANT" not in os.environ
        assert "SNAKE_SERVE_VETO_WATCH_HERO" not in os.environ
    assert os.environ["SNAKE_SERVE_CHECKPOINT"] == "champion"
    assert os.environ["SNAKE_SERVE_VETO_VARIANT"] == "v7"


def test_latency_requires_one_registered_seed_per_case():
    with pytest.raises(ValueError, match="exactly four"):
        measurements.run_latency([1, 2, 3], frames=1, warmup=0)


def test_lid_state_reads_the_exact_clamshell_key_not_other_yes_no_values(monkeypatch):
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0

    outcomes = iter(
        [
            Result("Now drawing from 'AC Power'\n"),
            Result('"OtherState" = Yes\n"AppleClamshellState" = No\n"Other" = Yes\n'),
        ]
    )
    monkeypatch.setattr(measurements.subprocess, "run", lambda *args, **kwargs: next(outcomes))
    observed = measurements._mac_power_and_lid()
    assert observed["ac_power"] is True
    assert observed["lid_closed"] is False


def test_pacing_validation_preserves_initial_and_single_lowered_retry(monkeypatch):
    calls = []

    def probe(fps, *_):
        calls.append(fps)
        return {
            "requested_fps": fps,
            "overruns": 1 if len(calls) == 1 else 0,
            "tick_ms": {"max": 20.0},
        }

    monkeypatch.setattr(measurements, "_paced_probe", probe)
    final_bound, attempts = measurements._validate_pacing(60, 7, "watch", 6, 200)
    assert calls == [60, 49]
    assert final_bound == 49
    assert len(attempts) == 2
    assert attempts[0]["validated"] is False
    assert attempts[1]["validated"] is True
