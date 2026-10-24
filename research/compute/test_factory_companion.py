"""Synthetic acceptance tests for the factory companion operational registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.compute import factory_companion as companion
from research.compute import factory_companion_native as native


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def row(pid: int, *, pgid: int = 10, start: str = "born", cpu: float = 0, rss: int = 100) -> dict:
    return {
        "pid": pid,
        "ppid": 1,
        "pgid": pgid,
        "start": start,
        "cpu": cpu,
        "rss": rss,
        "zombie": False,
        "argv": "python tiny.py",
    }


def supervisor(
    tmp_path: Path, clock: Clock, rows: dict[int, dict]
) -> companion.FactoryCompanionSupervisor:
    return companion.FactoryCompanionSupervisor(
        tmp_path, clock=clock, epoch=clock, process_table=lambda: rows
    )


def running(state: dict, *, klass: str = "test", pid: int = 10) -> None:
    state["registrations"]["t"] = {
        "token": "t",
        "class": klass,
        "status": "running",
        "pid": pid,
        "pgid": pid,
        "start": "born",
        "cpu": 0.0,
        "max_rss": 0,
        "started_monotonic": 0.0,
        "policy_sha256": companion.POLICY_SHA256,
        "lease_identity": "a",
    }


def ready(state: dict, identity: str = "a") -> None:
    state.update(
        {
            "monitor": {"pid": 1, "start": "monitor", "pgid": 1},
            "history_hold_until": 0.0,
            "light_only_since": -60.0,
            "observed_since": -60.0,
            "factory": {
                "identity": identity,
                "policy_sha256": companion.POLICY_SHA256,
                "protected": False,
                "streams": {
                    "d256": {
                        "baseline_ready": True,
                        "fresh_window_ready": True,
                        "window_full": True,
                        "window_started_at": -60.0,
                        "median_ratio": 1.0,
                    }
                },
            },
        }
    )


def test_test_slot_is_atomic_and_requires_light_observation(tmp_path: Path) -> None:
    clock, rows = Clock(), {10: row(10)}
    subject = supervisor(tmp_path, clock, rows)
    initial = subject._empty()
    ready(initial)
    initial["light_only_since"] = 0.0
    with pytest.raises(companion.AdmissionError, match="60 s"):
        subject._admit(initial, "test", companion.POLICY_SHA256, "a")
    state = subject._empty()
    ready(state)
    subject._admit(state, "test", companion.POLICY_SHA256, "a")
    running(state)
    with pytest.raises(companion.AdmissionError, match="global"):
        subject._admit(state, "test", companion.POLICY_SHA256, "a")


def test_duty_reserve_includes_live_reservation_and_persists(tmp_path: Path) -> None:
    clock, rows = Clock(), {10: row(10)}
    subject = supervisor(tmp_path, clock, rows)
    state = subject._empty()
    ready(state)
    state["duty"] = [{"at": 0.0, "cpu": 211.0}]
    with pytest.raises(companion.AdmissionError, match="duty"):
        subject._admit(state, "test", companion.POLICY_SHA256, "a")
    subject._write(state)
    assert subject._read()["duty"] == state["duty"]


def test_short_lived_or_pid_reused_group_fails_closed(tmp_path: Path) -> None:
    clock, rows = Clock(), {10: row(10, cpu=4)}
    subject = supervisor(tmp_path, clock, rows)
    state = subject._empty()
    running(state)
    assert subject._group_usage(state["registrations"]["t"], rows) == (4.0, 100, True)
    assert subject._group_usage(state["registrations"]["t"], {}) == (0.0, 0, False)
    rows[10] = row(10, start="reused")
    assert subject._group_usage(state["registrations"]["t"], rows) == (0.0, 0, False)


def test_sample_breach_revokes_only_owned_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, rows = Clock(), {10: row(10, cpu=20)}
    subject = supervisor(tmp_path, clock, rows)
    state = subject._empty()
    running(state)
    subject._write(state)
    seen: list[str] = []
    monkeypatch.setattr(subject, "terminate", seen.append)
    assert subject.sample() == ["t"]
    assert seen == ["t"]
    assert subject._read()["suspended_until"] == 900


def test_light_and_total_thresholds_are_aggregate_not_per_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, rows = Clock(), {10: row(10, cpu=7), 11: row(11, pgid=11, cpu=7)}
    subject = supervisor(tmp_path, clock, rows)
    state = subject._empty()
    running(state, klass="light", pid=10)
    state["registrations"]["u"] = {
        **state["registrations"]["t"],
        "token": "u",
        "pid": 11,
        "pgid": 11,
    }
    subject._write(state)
    seen: list[str] = []
    monkeypatch.setattr(subject, "terminate", seen.append)
    assert subject.sample() == ["t", "u"]
    assert seen == ["t", "u"]


def test_early_warning_only_sheds_test_and_does_not_preempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, rows = Clock(), {10: row(10)}
    subject = supervisor(tmp_path, clock, rows)
    state = subject._empty()
    running(state)
    state["heartbeat"] = 0.0
    subject._write(state)
    monkeypatch.setattr(subject, "terminate", lambda _token: None)
    # Exercise the state transition directly; facade construction is tested below.
    assert subject.revoke_tests() == ["t"]
    assert subject._read()["suspended_until"] == 900


def test_host_snapshot_is_v2_and_never_claims_no_sol_codex(tmp_path: Path) -> None:
    clock, rows = Clock(), {}
    subject = supervisor(tmp_path, clock, rows)
    subject._write(subject._empty())
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(companion, "FactoryCompanionSupervisor", lambda root: subject)
    try:
        result = companion.host_check_snapshot(
            active_factory_identity="attempt", policy_sha="p", root=tmp_path
        )
    finally:
        monkeypatch.undo()
    assert result["schema"].endswith("/v2")
    assert "no_sol_codex" not in result
    assert result["concurrency_policy_sha256"] == "p"


def test_shell_argv_is_refused_before_launch(tmp_path: Path) -> None:
    clock, rows = Clock(), {}
    subject = supervisor(tmp_path, clock, rows)
    with pytest.raises(companion.AdmissionError, match="validator-approved"):
        subject.launch(
            "light", policy_sha="p", lease_identity="a", worktree="w", argv=["/bin/sh", "-c", "x"]
        )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ("/Applications/ChatGPT.app/Contents/MacOS/ChatGPT --helper research/train.py", None),
        ("/Users/josenunez/.codex/computer-use/agent research/train.py", None),
        ("/Users/josenunez/Library/Application Support/Codex/helper research/train.py", None),
        ("codex exec 'please inspect research/train.py'", "unregistered_client"),
        (
            "node /opt/homebrew/lib/node_modules/@openai/codex/bin/codex.js exec x",
            "unregistered_client",
        ),
        ("python -m research.bl_labels.mac_driver --check", "unmanaged_compute"),
        ("python -m pytest research/compute/test_factory_companion.py", "unmanaged_compute"),
    ],
)
def test_argv_classifier_uses_executable_not_prompt_text(argv: str, expected: str | None) -> None:
    assert companion.classify_local_argv(argv) == expected


def test_native_v2_layout_keeps_child_cpu_and_start_identity() -> None:
    assert native.ctypes.sizeof(native._RusageInfoV2) == 160
    names = [name for name, _kind in native._RusageInfoV2._fields_]
    assert names.index("proc_start_abstime") < names.index("child_user_time")
