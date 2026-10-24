"""Focused, outcome-blind tests for the F0 slowdown escalation ruling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_labels import mac_driver, slowdown_resume, w12_runtime
from research.bl_labels.exports import publish_json
from research.bl_labels.factory_persistence import (
    INTERIM_RULING_SHA256,
    POLICY_ID,
    RULING_SHA256,
    PersistenceTracker,
)

_actual_operational_pin = slowdown_resume._operational_pin


@pytest.fixture(autouse=True)
def _isolated_test_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic receipt probes separate from the reviewer's outer lane."""
    from research.compute import factory_interim_launcher

    monkeypatch.setattr(factory_interim_launcher, "TEST_SLOT_PATH", tmp_path / "test-slot.lock")
    monkeypatch.setattr(slowdown_resume, "_operational_pin", lambda: "1" * 40)
    from research.bl_labels import factory_persistence

    clock = [0.0]
    monkeypatch.setattr(
        factory_persistence,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    monkeypatch.setattr(
        factory_persistence,
        "_native_identity",
        lambda pid: {
            "birth": f"fixture-{pid}",
            "birth_precision": {"unit": "fixture"},
            "executable": "/usr/bin/python3",
        },
    )


def _persistent_guard(monkeypatch: pytest.MonkeyPatch, owned: set[int] | None = None) -> None:
    """Exercise the debounce at 0,5,...30 seconds with native identity mocked."""
    from research.bl_labels import factory_persistence

    clock = [0.0]
    monkeypatch.setattr(
        factory_persistence,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _seconds: None),
    )
    tracker = PersistenceTracker("slowdown-guard-fixture")
    for index in range(7):
        clock[0] = index * 5.0
        mac_driver._exclusive_factory_guard(owned or set(), tracker)


def _attempt(
    state: Path,
    name: str,
    *,
    workers: int,
    records: list[dict[str, object]] | None = None,
    stop_reason: str | None = None,
    clean_intervals: tuple[tuple[float, float], ...] = (),
) -> None:
    start = state / f"attempt-{name}.start.json"
    publish_json(start, {"workers": workers})
    terminal: dict[str, object] = {
        "workers": workers,
        "elapsed_seconds": 1.0,
        "start_sha256": slowdown_resume._verify(start),
        "thermal_trip_epochs": [],
        "thermal_clean_intervals": [list(item) for item in clean_intervals],
        "slowdown_trip_records": records if records is not None else [],
    }
    if stop_reason is not None:
        terminal["stop_reason"] = stop_reason
    publish_json(state / f"attempt-{name}.terminal.json", terminal)


def _trip(stream: str, width: int, wall: float, epoch: float) -> dict[str, object]:
    return {"stream": stream, "width": width, "wall_seconds": wall, "epoch": epoch}


def _host_check(state: Path, epoch: float) -> None:
    state.mkdir(parents=True, exist_ok=True)
    telemetry = state / "slowdown-host-check-999.process.jsonl"
    attempt_id = "synthetic-host-window"
    rows = [
        {
            "policy_id": POLICY_ID,
            "ruling_sha256": RULING_SHA256,
            "interim_ruling_sha256": INTERIM_RULING_SHA256,
            "attempt_id": attempt_id,
            "sample_monotonic": index * 5.0,
            "sample_sequence": index + 1,
            "decision": "observation_complete",
        }
        for index in range(7)
    ]
    payload = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    telemetry.write_bytes(payload)
    publish_json(
        state / "slowdown-host-check-999.json",
        {
            "schema": slowdown_resume.HOST_CHECK_SCHEMA,
            "policy": slowdown_resume.PERSISTENCE_POLICY,
            "policy_id": POLICY_ID,
            "ruling_sha256": RULING_SHA256,
            "interim_ruling_sha256": INTERIM_RULING_SHA256,
            "operational_pin": slowdown_resume._operational_pin(),
            "for_trip_epoch": epoch,
            "checked_epoch": epoch + 1,
            "pmset_therm": "CPU_Speed_Limit = 100\n",
            "top_cpu_consumers": [],
            "no_disallowed_competing_compute": True,
            "serial_test_lane_stopped": True,
            "host_health": {"ac_power": True, "lid_open": True},
            "process_window": {
                "policy_id": POLICY_ID,
                "ruling_sha256": RULING_SHA256,
                "interim_ruling_sha256": INTERIM_RULING_SHA256,
                "attempt_id": attempt_id,
                "window_start_monotonic": 0.0,
                "window_end_monotonic": 30.0,
                "observation_count": 7,
                "final_pending": [],
                "observation_errors": [],
                "stop_reasons": [],
                "telemetry_path": str(telemetry),
                "telemetry_sha256": hashlib.sha256(payload).hexdigest(),
            },
        },
    )


@pytest.mark.parametrize("nice", [-5, 0, 14, 15, 19, 20])
@pytest.mark.parametrize(
    ("argv", "blocked"),
    [
        (
            "/Applications/ChatGPT.app/Contents/Resources/codex-cli/"
            "CodexCLI.app/Contents/MacOS/codex app-server",
            False,
        ),
        (
            "/Applications/ChatGPT.app/Contents/Resources/codex-cli/"
            "CodexCLI.app/Contents/MacOS/codex exec-server",
            False,
        ),
        (
            "/Applications/ChatGPT.app/Contents/Resources/codex-cli/"
            "CodexCLI.app/Contents/MacOS/codex exec task",
            False,
        ),
        ("/Applications/ChatGPT.app/Contents/Frameworks/Codex (Service)", False),
        ("/Applications/ChatGPT.app/Contents/Frameworks/Codex (Renderer)", False),
        (
            "/Users/josenunez/.codex/computer-use/Codex Computer Use.app/Contents/"
            "SharedSupport/SkyComputerUseClient.app/Contents/MacOS/SkyComputerUseClient",
            False,
        ),
        (str(Path.home() / ".codex/computer-use/codex") + " exec task", False),
        (
            str(Path.home() / "Library/Application Support/Codex/codex") + " exec task",
            False,
        ),
        (
            str(Path.home() / "Library/Application Support/Codex/Codex (Service)"),
            False,
        ),
        (
            str(Path.home() / "Library/Application Support/Codex/Codex (Renderer)"),
            False,
        ),
        (
            "/Applications/ChatGPT.app/Contents/MacOS/codex exec CODEX_MODEL=gpt-6.1-sol",
            False,
        ),
        (
            str(Path.home() / ".codex/computer-use/codex-handoff.sh"),
            False,
        ),
        ("/bin/bash /Users/owner/bin/codex-handoff.sh -C /tmp/worktree", True),
        ("/opt/homebrew/bin/codex exec --model gpt-6.1-sol task", True),
        ("/opt/homebrew/lib/node_modules/@openai/codex/bin/codex exec task", True),
        ("codex exec task", True),
        ("env CODEX_MODEL=gpt-6.1-sol /usr/local/bin/worker", True),
        ("/opt/homebrew/bin/codex app-server", False),
        ("/opt/homebrew/bin/codex exec-server", False),
        ("/opt/homebrew/bin/codex", False),
        ("/usr/local/bin/sol unrelated", False),
        ("python -m pytest tests", True),
        ("pytest research/bl_labels/test_slowdown_resume.py", True),
        ("python research/study/train.py", True),
    ],
)
def test_process_checks_match_job_argv_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: str, blocked: bool, nice: int
) -> None:
    blocked = blocked and nice < 15
    process_calls = []

    def command_output(command):
        process_calls.append(command)
        return f"123 1 {nice} {argv}\n"

    monkeypatch.setattr(mac_driver, "_command_output", command_output)
    if blocked:
        with pytest.raises(mac_driver.MacGuardError, match="persistent concurrent compute pid 123"):
            _persistent_guard(monkeypatch)
    else:
        mac_driver._exclusive_factory_guard(set(), PersistenceTracker("single-safe-observation"))
    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])
    calls = []

    def check_output(command: list[str], *, text: bool, errors: str, timeout: int) -> str:
        assert text is True
        assert errors == "surrogateescape"
        assert timeout == 10
        calls.append(command)
        if command == ["/usr/bin/pmset", "-g", "therm"]:
            return "CPU_Speed_Limit = 100\n"
        if command == ["/bin/ps", "-axo", "pid=,pcpu=,comm="]:
            return " 123 2.0 /Applications/ChatGPT.app/Contents/MacOS/codex_café\u2028x\udcff\n"
        raise AssertionError(command)

    monkeypatch.setattr(slowdown_resume.subprocess, "check_output", check_output)
    monkeypatch.setattr(
        mac_driver,
        "_guard_snapshot",
        lambda _: {"ac_power": True, "lid_open": True, "thermal": "CPU_Speed_Limit = 100\n"},
    )
    if blocked:
        with pytest.raises(ValueError, match="host check found unsafe process observation; HOLD"):
            slowdown_resume.record_host_check(tmp_path, checked_epoch=2001)
        assert not (tmp_path / "slowdown-host-check-2001.json").exists()
    else:
        path = slowdown_resume.record_host_check(tmp_path, checked_epoch=2001)
        receipt = slowdown_resume._read_triplet(path)
        assert "no_sol_codex" not in receipt
        assert receipt["no_disallowed_competing_compute"] is True
        assert receipt["serial_test_lane_stopped"] is True
        assert receipt["ruling_sha256"] == RULING_SHA256
        assert receipt["operational_pin"] == slowdown_resume._operational_pin()
        assert receipt["pmset_therm"] == "CPU_Speed_Limit = 100\n"
        assert receipt["top_cpu_consumers"] == [
            " 123 2.0 /Applications/ChatGPT.app/Contents/MacOS/codex_café\u2028x\udcff"
        ]
        assert slowdown_resume.resume_policy(6, tmp_path) == (6, [])
    if blocked:
        assert not calls
    else:
        assert calls[-1] == ["/bin/ps", "-axo", "pid=,pcpu=,comm="]
    assert process_calls
    assert all(
        call == ["/bin/ps", "-A", "-ww", "-o", "pid=,ppid=,nice=,args="] for call in process_calls
    )


def test_w12_pair_uses_common_verified_24_hour_clean_return_rule(tmp_path: Path) -> None:
    _attempt(tmp_path, "one", workers=12, records=[_trip("watch6", 12, 100, 1000)])
    _attempt(
        tmp_path,
        "two",
        workers=12,
        records=[_trip("watch6", 12, 100 + 3600, 2000)],
    )
    assert w12_runtime.resolve_resume_workers(12, tmp_path, now=2000 + 86400)[0] == 6
    _attempt(
        tmp_path,
        "three",
        workers=6,
        clean_intervals=((2000, 2000 + 86400),),
    )
    assert w12_runtime.resolve_resume_workers(12, tmp_path, now=2000 + 86400) == (12, [])


def test_w6_pair_requires_create_only_host_check_before_resume(tmp_path: Path) -> None:
    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])
    with pytest.raises(ValueError, match="host check"):
        slowdown_resume.resume_policy(6, tmp_path)
    with pytest.raises(ValueError, match="host check"):
        slowdown_resume.resume_policy(12, tmp_path)
    _host_check(tmp_path, 2000)
    assert slowdown_resume.resume_policy(6, tmp_path) == (6, [])


def test_effective_w3_pair_uses_w6_host_check_class(tmp_path: Path) -> None:
    # The receipt remains nominal W6 for accounting after a thermal stepdown,
    # while the guard records its actual admission width.
    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 3, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 3, 200, 2000)])
    with pytest.raises(ValueError, match="host check"):
        slowdown_resume.resume_policy(12, tmp_path)
    _host_check(tmp_path, 2000)
    assert slowdown_resume.resume_policy(12, tmp_path) == (12, [])


def test_w12_slowdown_cooldown_is_factory_wide(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    d256 = root / "bl-labels-f0" / "mac-pack-02" / "d256-state-mac-01"
    future = root / "bl-labels-f0-speed" / "d1024-d4096-pack-01" / "d1024-state-mac-01"
    _attempt(d256, "one", workers=12, records=[_trip("watch6", 12, 100, 1000)])
    _attempt(d256, "two", workers=12, records=[_trip("watch6", 12, 200, 2000)])
    _attempt(future, "one", workers=12)
    assert w12_runtime.resolve_resume_workers(12, future, now=2000 + 86400)[0] == 6


def test_pair_requires_same_stream_width_and_six_hour_active_wall_window(tmp_path: Path) -> None:
    _attempt(
        tmp_path,
        "one",
        workers=12,
        records=[_trip("watch6", 12, 100, 1000), _trip("gate", 12, 200, 2000)],
    )
    _attempt(
        tmp_path,
        "two",
        workers=12,
        records=[_trip("watch6", 12, 100 + slowdown_resume.SIX_HOURS + 1, 3000)],
    )
    assert slowdown_resume.escalation(tmp_path) == {"w6_epoch": None, "w12_epoch": None}
    assert slowdown_resume.resume_policy(12, tmp_path) == (12, [])


def test_baseline_sanity_w6_holds_and_w12_is_sticky_w6(tmp_path: Path) -> None:
    _attempt(tmp_path, "one", workers=6, stop_reason="slowdown baseline sanity")
    with pytest.raises(ValueError, match="new judge ruling"):
        slowdown_resume.resume_policy(6, tmp_path)

    second = tmp_path / "second"
    _attempt(second, "one", workers=12, stop_reason="slowdown baseline sanity")
    assert slowdown_resume.resume_policy(12, second) == (
        6,
        ["operational_width_fallback:slowdown baseline sanity"],
    )


def test_legacy_telemetry_counts_only_without_explicit_record_list(tmp_path: Path) -> None:
    for number, wall in enumerate((100.0, 200.0), 1):
        start = tmp_path / f"attempt-{number}.start.json"
        publish_json(start, {"workers": 6, "started_epoch": 1000.0 * number})
        publish_json(
            tmp_path / f"attempt-{number}.terminal.json",
            {
                "workers": 6,
                "elapsed_seconds": 1,
                "start_sha256": slowdown_resume._verify(start),
                "cumulative_elapsed_seconds": wall,
                "telemetry_sealed_no_outcomes": [{"event": "slowdown_guard"}]
                * (2 if number == 2 else 1),
            },
        )
    assert slowdown_resume.escalation(tmp_path)["w6_epoch"] == pytest.approx(2001.0)
    assert len(slowdown_resume.slowdown_trip_records(tmp_path)) == 2


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("schema", "bl-labels-f0-slowdown-host-check/v1"),
        ("schema", "foreign/v1"),
        ("policy", "ALLOW_BOUNDED_CONCURRENCY"),
        ("ruling_sha256", "0" * 64),
        ("operational_pin", "0" * 40),
        ("operational_pin", None),
        ("no_disallowed_competing_compute", False),
        ("serial_test_lane_stopped", False),
        ("host_health", None),
        ("no_sol_codex", True),
    ],
)
def test_recovery_refuses_unbound_or_foreign_interim_receipt(
    tmp_path: Path, field: str, foreign: object
) -> None:
    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])
    valid = tmp_path / "valid"
    _host_check(valid, 2000)
    value = slowdown_resume._read_triplet(valid / "slowdown-host-check-999.json")
    value[field] = foreign
    publish_json(tmp_path / "slowdown-host-check-foreign.json", value)
    for width in (6, 12):
        with pytest.raises(ValueError, match="host check required"):
            slowdown_resume.resume_policy(width, tmp_path)


def test_host_check_refuses_running_serial_test_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.compute.factory_interim_launcher import InterimLaunchError, serial_test_slot

    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])
    monkeypatch.setattr(
        mac_driver, "_guard_snapshot", lambda _: pytest.fail("must not observe health under tests")
    )
    with serial_test_slot():
        with pytest.raises(InterimLaunchError, match=".+"):
            slowdown_resume.record_host_check(tmp_path, checked_epoch=2001)
    assert not (tmp_path / "slowdown-host-check-2001.json").exists()


def test_host_check_requires_actual_health_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])

    def unsafe(_: Path) -> None:
        raise mac_driver.MacGuardError("refusing F0 admission on battery")

    monkeypatch.setattr(mac_driver, "_guard_snapshot", unsafe)
    with pytest.raises(ValueError, match="battery"):
        slowdown_resume.record_host_check(tmp_path, checked_epoch=2001)
    assert not (tmp_path / "slowdown-host-check-2001.json").exists()


@pytest.mark.parametrize("dirty", [" M research/bl_labels/slowdown_resume.py\n", "?? foreign.py\n"])
def test_operational_receipt_pin_refuses_dirty_checkout(
    monkeypatch: pytest.MonkeyPatch, dirty: str
) -> None:
    from types import SimpleNamespace

    def git(command, **kwargs):
        assert command == ["git", "status", "--porcelain", "--untracked-files=all"]
        assert kwargs["timeout"] == 10
        return SimpleNamespace(stdout=dirty)

    monkeypatch.setattr(slowdown_resume.subprocess, "run", git)
    with pytest.raises(ValueError, match="checkout is dirty"):
        _actual_operational_pin()


@pytest.mark.parametrize("tool", ["python -m research.x", "python research/x.py"])
@pytest.mark.parametrize("nice", [14, 15, 19])
def test_foreign_research_descendants_are_individually_priority_checked(
    monkeypatch: pytest.MonkeyPatch, tool: str, nice: int
) -> None:
    monkeypatch.setattr(
        mac_driver,
        "_command_output",
        lambda _: f"123 1 15 codex exec task\n124 123 {nice} {tool}\n",
    )
    if nice < 15:
        with pytest.raises(mac_driver.MacGuardError, match="persistent concurrent compute pid 124"):
            _persistent_guard(monkeypatch)
    else:
        mac_driver._exclusive_factory_guard(set(), PersistenceTracker("safe-descendant"))
