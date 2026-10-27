"""Unmocked classified-desktop collector and reducer integration evidence."""

from __future__ import annotations

import hashlib
import os
import platform
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        platform.system() != "Darwin" or os.environ.get("SERVED_DESKTOP_INTEGRATION") != "1",
        reason="set SERVED_DESKTOP_INTEGRATION=1 on the enrolled Mac for real process evidence",
    ),
]


def _artifact_directory() -> Path:
    """Create a fresh external evidence directory and leave it for inspection."""
    parent = Path(os.environ.get("SERVED_DESKTOP_INTEGRATION_ARTIFACT_ROOT", "/tmp"))
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="served-desktop-integration-", dir=parent))


def _trace_row(index: int, start_ns: int, end_ns: int) -> dict:
    """Build one reducer trace row whose window is exactly one captured interval."""
    return {
        "tick": index,
        "target_ns": start_ns,
        "wake_ns": start_ns,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "start_cpu_ns": start_ns,
        "end_cpu_ns": end_ns,
        "spans": {"game_update": {"wall_ns": end_ns - start_ns}},
    }


def test_unmocked_desktop_collector_raw_export_and_reducers():
    """Capture real Mac process evidence and replay every desktop reducer entry point."""
    from research.bl_mac_latency import host
    from research.bl_paced_identity.record import export_json, read_export
    from research.bl_served_search import background_report, host_load, probe_host
    from research.bl_served_search.desktop_apps import DEFAULT_REGISTRY, DesktopRegistry

    artifact_dir = _artifact_directory()
    registry = DesktopRegistry(DEFAULT_REGISTRY)
    registry.verify_all()

    owner_pid = os.getpid()
    screen = host.process_screen(include_identity=True)
    point_violations = probe_host.foreign_load(
        screen,
        owner_pid,
        classified=True,
        desktop_classifier=registry,
        admit_desktop=True,
    )

    previous = host_load.capture_cpu()
    samples = []
    guard = host_load.LoadGuard(desktop_policy=True)
    guard_rejections: list[str] = []
    for _ in range(3):
        time.sleep(host_load.SAMPLE_SECONDS + 0.05)
        current = host_load.capture_cpu()
        sample = host_load.cpu_interval(previous, current, owner_pid, desktop_classifier=registry)
        samples.append(sample)
        try:
            guard.check(sample)
        except RuntimeError as exc:
            guard_rejections.append(f"{type(exc).__name__}: {exc}")
        previous = current

    registry_export = registry.read_export()
    host_evidence = {
        "schema": "served-tick-host/v4",
        "load_policy": host_load.DESKTOP_POLICY,
        "status": "INVALID" if guard_rejections or point_violations else "COMPLETE",
        "reason": "; ".join(guard_rejections) if guard_rejections else None,
        "interval_seconds": 10.0,
        "cpu_interval_seconds": host_load.SAMPLE_SECONDS,
        "max_cpu_interval_seconds": host_load.MAX_SAMPLE_SECONDS,
        "heavy_percent_one_core": 20.0,
        "background_ceiling_percent_one_core": host_load.BACKGROUND_CEILING_PERCENT,
        "background_ceiling_seconds_strictly_greater": host_load.BACKGROUND_CEILING_SECONDS,
        "elapsed_seconds": sum((sample["end_ns"] - sample["start_ns"]) / 1e9 for sample in samples),
        "observations": [{"process_screen": screen}],
        "point_guard_violations": point_violations,
        "cpu_samples": samples,
        "desktop_registry": registry_export["registry"],
        "desktop_registry_sha256": registry_export["registry_sha256"],
        "desktop_registry_evidence": registry_export,
        "cpu_method": "real ps cumulative CPU TIME deltas and real classified point inventory",
    }
    host_path = artifact_dir / "host.json"
    host_sha256 = export_json(host_path, host_evidence)
    replayed_host = read_export(host_path)
    assert replayed_host == host_evidence
    assert hashlib.sha256(host_path.read_bytes()).hexdigest() == host_sha256
    assert (artifact_dir / "host.json.sha256").is_file()
    assert (artifact_dir / "host.json.complete").read_bytes() == b"complete\n"

    from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS

    admitted_apps = {
        process["classification"]["application"]
        for sample in samples
        for process in sample["processes"]
        if process["kind"] == "desktop_app"
        and isinstance(process.get("classification"), dict)
        and process["classification"].get("application")
    }
    assert admitted_apps, "the real enrolled-app inventory admitted no desktop_app rows"
    assert all(
        sample["processes"] for sample in samples
    ), "real interval capture produced no process rows"

    exempt_pids = set(screen.get("probe_exempt_pids", []))
    point_protected_services: set[str] = set()
    for proc in screen["classified_processes"]:
        if proc["pid"] in exempt_pids:
            continue
        cmd = proc["command"]
        cls = proc.get("classification", {})
        if (
            cmd.startswith(SYSTEM_EXECUTABLE_ROOTS)
            and ".." not in Path(cmd).parts
            and proc["kind"] != "agent_workload"
            and not cls.get("identity_binding_changed")
        ):
            assert proc["kind"] == "macos_service", f"system service misclassified: {proc}"
            if proc.get("identity_observation_error"):
                assert proc.get("argv") == []
                point_protected_services.add(Path(cmd).name)
    assert {
        "syspolicyd",
        "trustd",
        "fileproviderd",
    } <= point_protected_services, (
        f"expected protected services in point inventory, got {sorted(point_protected_services)}"
    )

    for sample in samples:
        for proc in sample["processes"]:
            cmd = proc["command"]
            cls = proc.get("classification", {})
            if (
                cmd.startswith(SYSTEM_EXECUTABLE_ROOTS)
                and ".." not in Path(cmd).parts
                and proc["kind"] != "agent_workload"
                and not cls.get("identity_binding_changed")
            ):
                assert (
                    proc["kind"] == "macos_service"
                ), f"interval system service misclassified: {proc}"

    trace = [
        _trace_row(index, sample["start_ns"], sample["end_ns"])
        for index, sample in enumerate(samples)
    ]
    result = {"warmup_ticks": 0, "trace": trace}

    # Complete sealed replay must succeed before contamination can be
    # classified.  This keeps schema/identity regressions visible.
    replayed_samples = background_report._samples(replayed_host)
    assert len(replayed_samples) == len(samples)
    summary = background_report.summarize_background(result, replayed_host)
    assert summary["counts"]["ticks"] == len(samples)

    reducer_results: dict[str, object] = {"summarize_background": summary}
    reducer_errors: dict[str, str] = {}
    for name, reducer in (
        ("validate_desktop_host", background_report.validate_desktop_host),
        ("desktop_overall", background_report.desktop_overall),
    ):
        try:
            reducer_results[name] = reducer(replayed_host)
        except ValueError as exc:
            reducer_errors[name] = f"{type(exc).__name__}: {exc}"

    if guard_rejections:
        # A live host guard rejection is valid evidence.  The raw export above
        # remains immutable; reducers must either replay it or reject the same
        # contaminated evidence rather than silently dropping intervals.
        assert "validate_desktop_host" in reducer_errors
        assert "desktop_overall" in reducer_errors
        assert all(
            any(
                marker in reducer_errors[name]
                for marker in (
                    "ordinary workload/unknown",
                    "prohibited workload",
                    "identity binding",
                    "shared background",
                )
            )
            for name in ("validate_desktop_host", "desktop_overall")
        )
    elif point_violations:
        assert "validate_desktop_host" in reducer_errors
        assert "ordinary workload/unknown" in reducer_errors["validate_desktop_host"]
        assert "desktop_overall" not in reducer_errors
    else:
        assert not reducer_errors, reducer_errors
        assert reducer_results["desktop_overall"]["sample_count"] == len(samples)
        assert reducer_results["summarize_background"]["counts"]["ticks"] == len(samples)

    receipt = {
        "host_path": str(host_path),
        "host_sha256": host_sha256,
        "sample_count": len(samples),
        "admitted_applications": sorted(admitted_apps),
        "point_guard_violation_count": len(point_violations),
        "guard_rejections": guard_rejections,
        "reducer_errors": reducer_errors,
        "scope": "three-second collector integration; synthetic alignment trace; no panel timing",
    }
    export_json(artifact_dir / "reducer-results.json", reducer_results)
    export_json(artifact_dir / "receipt.json", receipt)
    print("UNMOCKED_DESKTOP_EVIDENCE", receipt)
