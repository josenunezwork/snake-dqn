"""Regression for setup consuming the same clock as collection and export."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from research.bl_labels import manifests, runtime


@pytest.mark.parametrize("stage", ["transport_probe", "science"])
def test_stage_caps_apply_two_once_and_cover_session_with_reserve(stage: str) -> None:
    value = manifests.stage_envelope(stage)
    base_seconds = sum(
        value[key] for key in ("setup_seconds", "work_seconds", "export_seconds", "margin_seconds")
    )
    assert value["effective_minutes"] * 60 == 2 * base_seconds
    assert value["effective_units"] == 2 * value["base_units"]
    assert value["effective_reserve_units"] == 2 * value["base_reserve_units"]
    assert value["effective_finalize_seconds"] == 2 * (
        value["export_seconds"] + value["margin_seconds"]
    )
    charged_units = value["effective_minutes"] * 0.08 / 60
    assert charged_units + value["effective_reserve_units"] <= value["effective_units"]


def _qualified_state(tmp_path: Path, started_at: float = 1000) -> None:
    (tmp_path / "spec.json").write_text(
        json.dumps(
            {
                "job_id": "f0-budget-probe",
                "generation": "fenced",
                "root": str(tmp_path),
                "timeout_seconds": 1100,
                "env": {"F0_SESSION_SECONDS": "1200", "F0_FINALIZE_SECONDS": "300"},
            }
        )
    )
    (tmp_path / "state.json").write_text(
        json.dumps({"job_id": "f0-budget-probe", "generation": "fenced", "started_at": started_at})
    )


def test_measured_setup_reduces_runtime_including_cold_bootstrap(monkeypatch, tmp_path) -> None:
    _qualified_state(tmp_path)
    monkeypatch.setattr(runtime.time, "time", lambda: 1180)
    budget = runtime.session_budget(tmp_path, 1200, 300)
    assert budget["runtime_seconds"] == 620
    assert budget["measured_setup_seconds"] == 180
    assert budget["pre_watchdog_setup_and_guard_seconds"] == 100
    assert (
        sum(
            budget[key]
            for key in (
                "runtime_seconds",
                "measured_setup_seconds",
                "pre_watchdog_setup_and_guard_seconds",
                "finalize_seconds",
            )
        )
        == 1200
    )
    monkeypatch.setattr(runtime.time, "time", lambda: 1240)
    assert runtime.session_budget(tmp_path, 1200, 300)["runtime_seconds"] == 560


@pytest.mark.parametrize("now", [1800, 1900, float("nan"), 999])
def test_exhausted_or_invalid_setup_refuses_collection(monkeypatch, tmp_path, now) -> None:
    _qualified_state(tmp_path)
    monkeypatch.setattr(runtime.time, "time", lambda: now)
    with pytest.raises(ValueError, match="F0 (setup exhausted|session budget)"):
        runtime.session_budget(tmp_path, 1200, 300)


def test_foreign_generation_refused(monkeypatch, tmp_path) -> None:
    _qualified_state(tmp_path)
    monkeypatch.setattr(runtime.time, "time", lambda: 1180)
    state = json.loads((tmp_path / "state.json").read_text())
    state["generation"] = "foreign"
    (tmp_path / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="identity differs"):
        runtime.session_budget(tmp_path, 1200, 300)


def test_ledger_verification_never_restarts_absolute_collection_clock(
    monkeypatch, tmp_path
) -> None:
    """A resumed ledger may be large; its verification consumes remaining work time."""
    clock = [1000.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])

    def slow_load(*_args):
        clock[0] += 200
        return set()

    monkeypatch.setattr(runtime, "load_ledger", slow_load)
    monkeypatch.setattr(runtime, "completed_session_parents", lambda *_args: set())
    seen = []

    class Pool:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def imap(self, _target, jobs):
            seen.extend(jobs)
            return [({"allocation": job[0], "status": "COMPLETE"}, None) for job in jobs]

    monkeypatch.setattr(runtime.mp, "get_context", lambda _mode: types.SimpleNamespace(Pool=Pool))
    runtime.run(
        {
            "namespace": "budget-test",
            "workers": 1,
            "allocation": [{"allocation_id": "watch6-000", "horizon": 3}],
        },
        tmp_path / "ledger",
        tmp_path / "b8",
        tmp_path / "frp3",
        tmp_path,
        deadline_monotonic=1620,
    )
    assert clock[0] == 1200
    assert seen[0][-1] == 1620  # 420s remain, not a restarted 620s.
