"""Conformance checks for event-sourced classified-desktop identity history."""

from __future__ import annotations

import copy
import hashlib
import json
import plistlib
import subprocess
from pathlib import Path

import pytest

from research.bl_served_search import background_report, host_load
from research.bl_served_search.desktop_apps import (
    POLICY,
    DesktopRegistry,
    classify_exported,
    replay_history,
)


def _app_payload(
    executable: Path,
    bundle: Path,
    *,
    app_id: str = "mail",
    display_name: str = "Mail",
    bundle_id: str = "example.mail",
    role: str = "ordinary_mail_ui",
) -> dict:
    return {
        "id": app_id,
        "display_name": display_name,
        "bundle_path": str(bundle),
        "bundle_id": bundle_id,
        "version": "1",
        "build": "2",
        "team_identifier": "TEAM",
        "executables": [
            {
                "role": role,
                "relative_executable_path": str(executable.relative_to(bundle)),
                "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "designated_requirement": f'designated => identifier "{bundle_id}"',
                "cdhash": "abc",
                "signing_authorities": ["Test Authority"],
            }
        ],
    }


def _write_registry(tmp_path: Path, applications: list[dict], uid: int = 501) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema": "served-tick-desktop-registry/v1",
                "policy": POLICY,
                "owner_uid": uid,
                "permitted_use_declaration": "ordinary UI only",
                "workload_exclusions": [{"exact_executable": "/usr/local/bin/claude"}],
                "applications": applications,
            }
        )
    )
    return path


def _runner(argv, **_kwargs):
    command = str(argv[-1])
    bundle_id = (
        "example.codex"
        if "Codex" in command
        else ("example.browser" if "Browser" in command else "example.mail")
    )
    if argv[0].endswith("codesign") and "-dv" in argv:
        return subprocess.CompletedProcess(
            argv, 0, "TeamIdentifier=TEAM\nCDHash=abc\nAuthority=Test Authority", ""
        )
    if argv[0].endswith("codesign") and "-dr" in argv:
        return subprocess.CompletedProcess(argv, 0, f'designated => identifier "{bundle_id}"', "")
    return subprocess.CompletedProcess(argv, 0, "accepted", "")


def _make_app(tmp_path: Path, name: str, executable_name: str, bundle_id: str) -> tuple[Path, Path]:
    bundle = tmp_path / "Applications" / f"{name}.app"
    executable = bundle / "Contents" / "MacOS" / executable_name
    executable.parent.mkdir(parents=True)
    executable.write_bytes(f"sealed {name}".encode())
    with (bundle / "Contents" / "Info.plist").open("wb") as output:
        plistlib.dump(
            {
                "CFBundleIdentifier": bundle_id,
                "CFBundleShortVersionString": "1",
                "CFBundleVersion": "2",
            },
            output,
        )
    return executable, bundle


@pytest.fixture
def enrolled(tmp_path: Path):
    executable, bundle = _make_app(tmp_path, "Mail", "Mail", "example.mail")
    registry = DesktopRegistry(
        _write_registry(tmp_path, [_app_payload(executable, bundle)]),
        uid=501,
        runner=_runner,
        trusted_roots=[tmp_path / "Applications"],
        require_seal=False,
    )
    registry.verify_all()
    return (
        registry,
        {"pid": 42, "birth": "birth-a", "uid": 501, "command": str(executable), "argv": []},
        executable,
        bundle,
    )


def _freeze(registry: DesktopRegistry, snapshot: dict) -> dict:
    host_load.freeze_cpu_snapshot(snapshot, 999_999, registry)
    return snapshot


def _snapshot(registry: DesktopRegistry, row: dict, *, admit: bool = False) -> dict:
    return registry.observe_snapshot(row, admit=admit)


def test_enrolled_identity_requires_off_tick_binding_and_replays_reference(enrolled):
    registry, row, *_ = enrolled
    unbound = _snapshot(registry, row)
    admitted = _snapshot(registry, row, admit=True)
    repeated = _snapshot(registry, row)
    assert unbound["classification"]["reason"] == "process_identity_not_revalidated"
    assert admitted["classification"]["kind"] == repeated["classification"]["kind"] == "desktop_app"
    # Stable observations intentionally compact to the same immutable event.
    assert (
        admitted["classification"]["history_index"] == repeated["classification"]["history_index"]
    )
    assert classify_exported(admitted, registry.read_export()) == admitted["classification"]
    assert (
        classify_exported(dict(row), registry.read_export())["reason"]
        == "desktop_history_reference_missing"
    )


@pytest.mark.parametrize("change", ["uid", "path", "argv"])
def test_same_pid_birth_identity_change_is_terminal(enrolled, change):
    registry, row, executable, _ = enrolled
    _snapshot(registry, row, admit=True)
    replacement = (
        {"uid": 502}
        if change == "uid"
        else (
            {"command": str(executable.with_name("copy"))}
            if change == "path"
            else {"argv": ["--changed"]}
        )
    )
    changed = _snapshot(registry, {**row, **replacement})
    restored = _snapshot(registry, row, admit=True)
    assert changed["classification"]["reason"] == "identity_binding_changed"
    assert changed["classification"]["identity_binding_changed"] is True
    assert restored["classification"]["reason"] == "identity_binding_changed"


def test_workload_precedence_retains_terminal_history(enrolled):
    registry, row, *_ = enrolled
    _snapshot(registry, row, admit=True)
    changed = _snapshot(registry, {**row, "command": "/usr/local/bin/claude"})
    assert changed["classification"]["kind"] == "agent_workload"
    assert changed["classification"]["identity_binding_changed"] is True
    assert classify_exported(changed, registry.read_export()) == changed["classification"]


def test_real_enrolled_codex_gui_does_not_inherit_cli_workload_rule(tmp_path: Path):
    mail, mail_bundle = _make_app(tmp_path, "Mail", "Mail", "example.mail")
    codex, codex_bundle = _make_app(tmp_path, "Codex", "Codex", "example.codex")
    registry = DesktopRegistry(
        _write_registry(
            tmp_path,
            [
                _app_payload(mail, mail_bundle),
                _app_payload(
                    codex,
                    codex_bundle,
                    app_id="codex",
                    display_name="Codex desktop",
                    bundle_id="example.codex",
                    role="remote_chat_ui",
                ),
            ],
        ),
        uid=501,
        runner=_runner,
        trusted_roots=[tmp_path / "Applications"],
        require_seal=False,
    )
    registry.verify_all()
    row = {"pid": 43, "birth": "codex", "uid": 501, "command": str(codex), "argv": []}
    admitted = _snapshot(registry, row, admit=True)
    cli = _snapshot(registry, {**row, "pid": 44, "birth": "cli", "command": "/usr/local/bin/codex"})
    assert admitted["classification"]["kind"] == "desktop_app"
    assert cli["classification"]["kind"] == "agent_workload"
    assert classify_exported(admitted, registry.read_export()) == admitted["classification"]


def test_replacement_of_unbound_process_remains_terminal_after_restore(enrolled):
    registry, row, executable, _ = enrolled
    executable.write_bytes(b"replaced")
    replaced = _snapshot(registry, row, admit=True)
    executable.write_bytes(b"sealed Mail")
    restored = _snapshot(registry, row, admit=True)
    # The attempt's frozen verification evidence cannot be refreshed after
    # history starts; a fresh process generation must use admission evidence.
    with pytest.raises(ValueError, match="history"):
        registry.verify_all()
    fresh_row = {**row, "pid": 43, "birth": "fresh"}
    fresh = _snapshot(registry, fresh_row, admit=True)
    steady_one = _snapshot(registry, fresh_row)
    steady_two = _snapshot(registry, fresh_row)
    assert (
        replaced["classification"]["reason"]
        == restored["classification"]["reason"]
        == "executable_replaced"
    )
    export = registry.read_export()
    for snapshot in (fresh, steady_one, steady_two):
        assert snapshot["classification"]["kind"] == "desktop_app"
        assert classify_exported(snapshot, export) == snapshot["classification"]
    # The new generation must not invalidate frozen evidence for the original
    # replacement/revocation events.
    replay_history(export)


def _inventory(at: int, row: dict, cpu_seconds: float) -> dict:
    return {
        "time_ns": at * 1_000_000_000,
        "capture_start_ns": at * 1_000_000_000 - 10,
        "capture_end_ns": at * 1_000_000_000 + 10,
        "processes": [{**row, "ppid": 1, "cpu_seconds": cpu_seconds}],
    }


def _host(registry: DesktopRegistry, samples: list[dict]) -> dict:
    evidence = registry.read_export()
    return {
        "load_policy": host_load.DESKTOP_POLICY,
        "desktop_registry": evidence["registry"],
        "desktop_registry_sha256": evidence["registry_sha256"],
        "desktop_registry_evidence": evidence,
        "cpu_samples": samples,
        "observations": [],
    }


def _paired(enrolled, tmp_path: Path) -> tuple[DesktopRegistry, dict, dict]:
    _, row, executable, bundle = enrolled
    other, other_bundle = _make_app(tmp_path, "Browser", "Browser", "example.browser")
    registry = DesktopRegistry(
        _write_registry(
            tmp_path,
            [
                _app_payload(executable, bundle),
                _app_payload(
                    other,
                    other_bundle,
                    app_id="browser",
                    display_name="Browser",
                    bundle_id="example.browser",
                    role="ordinary_browser_ui",
                ),
            ],
        ),
        uid=501,
        runner=_runner,
        trusted_roots=[tmp_path / "Applications"],
        require_seal=False,
    )
    registry.verify_all()
    return registry, row, {**row, "command": str(other)}


_STATES = (
    "invalid",
    "desktop_app",
    "workload",
    "uid_mismatch",
    "system_service",
    "other_enrolled_app",
)


def _state(label: str, row: dict, other: dict) -> dict:
    if label == "invalid":
        return {**row, "command": str(Path(row["command"]).with_name("missing"))}
    if label == "desktop_app":
        return dict(row)
    if label == "workload":
        return {**row, "command": "/usr/local/bin/claude"}
    if label == "uid_mismatch":
        return {**row, "uid": 0}
    if label == "system_service":
        return {**row, "command": "/usr/sbin/sysctl"}
    if label == "other_enrolled_app":
        return dict(other)
    raise AssertionError(label)


@pytest.mark.parametrize(
    ("before_label", "after_label"), [(a, b) for a in _STATES for b in _STATES]
)
def test_collector_and_reducer_replay_every_identity_pair_and_two_steady_intervals(
    enrolled, tmp_path, before_label, after_label
):
    registry, row, other = _paired(enrolled, tmp_path)
    before, after = _state(before_label, row, other), _state(after_label, row, other)
    first, second = _freeze(registry, _inventory(0, before, 1.0)), _freeze(
        registry, _inventory(1, after, 1.1)
    )
    samples = [host_load.cpu_interval(first, second, 999_999, desktop_classifier=registry)]
    for at, cpu in ((2, 1.2), (3, 1.3)):
        later = _freeze(registry, _inventory(at, after, cpu))
        samples.append(host_load.cpu_interval(second, later, 999_999, desktop_classifier=registry))
        second = later
    exported = [item for sample in samples for item in sample["processes"]]
    assert exported
    for process in exported:
        if "interval_transition" in process:
            transition = process["interval_transition"]
            current = transition["current_classification"]
            assert (
                classify_exported({**process, "classification": current}, registry.read_export())
                == current
            )
            previous, prior = (
                transition["previous"],
                transition["previous_classification"],
            )
            assert previous["observation_ns"] == 0
            assert (
                classify_exported({**previous, "classification": prior}, registry.read_export())
                == prior
            )
        else:
            assert classify_exported(process, registry.read_export()) == process["classification"]
    reduced = background_report._samples(_host(registry, samples))
    assert [[p["kind"] for p in sample["processes"]] for sample in reduced] == [
        [p["kind"] for p in sample["processes"]] for sample in samples
    ]
    if before_label == after_label and before_label in {"invalid", "uid_mismatch"}:
        # Initial malformed/UID verification failures revoke that PID/birth so
        # it cannot later be admitted, but they are ordinary unknown evidence
        # at a low CPU rate, not an observed identity-transition contamination.
        assert all(
            not process["classification"]["identity_binding_changed"] for process in exported
        )
        host = _host(registry, samples)
        host["observations"] = [{"classified_processes": [], "probe_exempt_pids": []}]
        background_report.validate_desktop_host(host)


@pytest.mark.parametrize(("field", "value"), [("pid", []), ("birth", [])])
def test_malformed_process_identity_is_unknown_without_state_machine_crash(enrolled, field, value):
    registry, row, *_ = enrolled
    malformed = _snapshot(registry, {**row, field: value}, admit=True)
    assert malformed["classification"]["kind"] == "unknown"


def test_export_policy_mismatch_and_tampered_actual_history_fail_closed(enrolled):
    registry, row, *_ = enrolled
    snapshot = _snapshot(registry, row, admit=True)
    export = registry.read_export()
    assert classify_exported(snapshot, {**export, "policy": "wrong"})["kind"] == "unknown"
    tampered = copy.deepcopy(export)
    tampered["history_sequence"][snapshot["classification"]["history_index"]]["classification"][
        "reason"
    ] = "forged"
    assert classify_exported(snapshot, tampered)["kind"] == "unknown"


def test_explicit_signed_renderer_role_admits_and_replays(enrolled, tmp_path):
    _, row, executable, bundle = enrolled
    renderer = (
        bundle
        / "Contents"
        / "Frameworks"
        / "Mail Renderer.app"
        / "Contents"
        / "MacOS"
        / "Mail Renderer"
    )
    renderer.parent.mkdir(parents=True)
    renderer.write_bytes(executable.read_bytes())
    registry = DesktopRegistry(
        _write_registry(tmp_path, [_app_payload(renderer, bundle, role="remote_chat_renderer")]),
        uid=501,
        runner=_runner,
        trusted_roots=[tmp_path / "Applications"],
        require_seal=False,
    )
    registry.verify_all()
    snapshot = _snapshot(
        registry, {**row, "pid": 43, "birth": "renderer", "command": str(renderer)}, admit=True
    )
    assert snapshot["classification"]["kind"] == "desktop_renderer"
    assert classify_exported(snapshot, registry.read_export()) == snapshot["classification"]


def test_registry_static_enrollment_is_real_and_ui_roles_only():
    path = (
        Path(__file__).parents[1] / "research/bl_served_search/DESKTOP_APP_REGISTRY_20261027.json"
    )
    payload = json.loads(path.read_text())
    assert payload["policy"] == POLICY
    assert {"Microsoft Outlook", "Claude", "Codex desktop", "Google Chrome"} <= {
        app["display_name"] for app in payload["applications"]
    }
    assert all(
        "gpu" not in role["role"] and "service" not in role["role"]
        for app in payload["applications"]
        for role in app["executables"]
    )


def test_wrong_signing_identity_fails_verification(enrolled):
    registry, *_ = enrolled
    registry._runner = lambda argv, **_: subprocess.CompletedProcess(
        argv, 0, "TeamIdentifier=WRONG", ""
    )
    with pytest.raises(ValueError):
        registry.verify_all()
