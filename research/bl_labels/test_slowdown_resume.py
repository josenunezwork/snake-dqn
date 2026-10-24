"""Focused, outcome-blind tests for the F0 slowdown escalation ruling."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from research.bl_labels import slowdown_resume, w12_runtime
from research.bl_labels.exports import publish_json


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
    publish_json(
        state / "slowdown-host-check-999.json",
        {
            "schema": slowdown_resume.HOST_CHECK_SCHEMA,
            "for_trip_epoch": epoch,
            "checked_epoch": epoch + 1,
            "concurrency_policy_sha256": slowdown_resume._concurrency_policy_sha256(),
            "no_unmanaged_compute": True,
            "no_active_test": True,
            "companion_healthy": True,
            "registry_complete": True,
            "registry_monitor_identity": "fixture-registry",
            "monitor_heartbeat_age_seconds": 0.0,
            "monitor_identity": {"pid": 1, "pgid": 1, "start": "fixture"},
            "host_health": {
                "ac_power": True,
                "lid_open": True,
                "thermal": "fixture",
                "thermal_parsed": {},
                "free_disk_bytes": 50 * 1024**3,
            },
            "observation": {
                "light_only_seconds": 60,
                "sample_count": 12,
                "started_monotonic": 1.0,
                "ended_monotonic": 61.0,
                "trailing_cpu_core_seconds": 0.0,
                "trailing_light_cpu_core_seconds": 0.0,
                "rss_bytes": 0,
            },
        },
    )


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
    ],
)
def test_process_checks_match_job_argv_only(argv: str, blocked: bool) -> None:
    """Desktop helpers are excluded by argv, never by a process-name shortcut."""
    assert slowdown_resume._sol_codex_job(argv) is blocked


def test_record_host_check_requires_v2_companion_observation(tmp_path: Path, monkeypatch) -> None:
    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])
    monkeypatch.setattr(slowdown_resume, "_require_companion_activation", lambda: None)
    monkeypatch.setattr(
        slowdown_resume,
        "_record_host_health",
        lambda _: {
            "ac_power": True,
            "lid_open": True,
            "thermal": "fixture",
            "thermal_parsed": {},
            "free_disk_bytes": 50 * 1024**3,
        },
    )
    monkeypatch.setattr(slowdown_resume, "_concurrency_policy_sha256", lambda: "a" * 64)

    def snapshot(**_kwargs):
        return {
            "no_unmanaged_compute": True,
            "no_active_test": True,
            "companion_healthy": True,
            "registry_complete": True,
            "light_only_observation_seconds": 60,
            "registry_monitor_identity": "fixture-registry",
            "monitor_heartbeat_age_seconds": 0.0,
            "monitor_identity": {"pid": 1, "pgid": 1, "start": "fixture"},
            "observation": {
                "sample_count": 12,
                "started_monotonic": 1.0,
                "ended_monotonic": 61.0,
                "trailing_cpu_core_seconds": 0.0,
                "trailing_light_cpu_core_seconds": 0.0,
                "rss_bytes": 0,
            },
            "factory_attempt_identity": None,
        }

    monkeypatch.setattr("research.compute.factory_companion.host_check_snapshot", snapshot)
    monkeypatch.setattr(
        "research.compute.factory_companion.bound_factory_identity",
        lambda **_kwargs: "fixture-attempt",
    )
    monkeypatch.setattr(
        "research.compute.factory_companion.control_helper", lambda **_kwargs: nullcontext()
    )
    path = slowdown_resume.record_host_check(tmp_path, checked_epoch=2001)
    receipt = slowdown_resume._read_triplet(path)
    assert receipt["schema"] == slowdown_resume.HOST_CHECK_SCHEMA
    assert receipt["concurrency_policy_sha256"] == "a" * 64
    assert receipt["no_unmanaged_compute"] is True
    assert "no_sol_codex" not in receipt
    assert slowdown_resume.resume_policy(6, tmp_path) == (6, [])


def test_old_host_check_schema_is_not_a_new_policy_resume_proof(tmp_path: Path) -> None:
    _attempt(tmp_path, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(tmp_path, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])
    publish_json(
        tmp_path / "slowdown-host-check-old.json",
        {
            "schema": slowdown_resume.HOST_CHECK_SCHEMA_V1,
            "for_trip_epoch": 2000,
            "checked_epoch": 2001,
            "no_sol_codex": True,
            "pmset_therm": "historical",
            "top_cpu_consumers": [],
        },
    )
    with pytest.raises(ValueError, match="host check"):
        slowdown_resume.resume_policy(6, tmp_path)


def test_unqualified_host_check_refuses_before_any_receipt_or_probe(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    def hold() -> None:
        calls.append("activation")
        raise ValueError("HOLD")

    monkeypatch.setattr(slowdown_resume, "_require_companion_activation", hold)
    monkeypatch.setattr(
        slowdown_resume,
        "_record_host_health",
        lambda _: pytest.fail("host probe must not run while activation is HOLD"),
    )
    with pytest.raises(ValueError, match="HOLD"):
        slowdown_resume.record_host_check(tmp_path, checked_epoch=1)
    assert calls == ["activation"]
    assert not list(tmp_path.glob("slowdown-host-check-*.json"))


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
