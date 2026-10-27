"""Offline conformance checks for sealed classified-desktop-v1 admission."""

from __future__ import annotations

import hashlib
import json
import plistlib
import subprocess
from pathlib import Path

import pytest

from research.bl_served_search import background_report, host_load
from research.bl_served_search.desktop_apps import POLICY, DesktopRegistry, classify_exported


def _write_registry(tmp_path: Path, *, executable: Path, bundle: Path, uid: int = 501) -> Path:
    payload = {
        "schema": "served-tick-desktop-registry/v1",
        "policy": POLICY,
        "owner_uid": uid,
        "permitted_use_declaration": "ordinary mail only",
        "workload_exclusions": [{"exact_executable": "/usr/local/bin/claude"}],
        "applications": [
            {
                "id": "mail",
                "display_name": "Mail",
                "bundle_path": str(bundle),
                "bundle_id": "example.mail",
                "version": "1",
                "build": "2",
                "team_identifier": "TEAM",
                "executables": [
                    {
                        "role": "ordinary_mail_ui",
                        "relative_executable_path": str(executable.relative_to(bundle)),
                        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                        "designated_requirement": 'designated => identifier "example.mail"',
                        "cdhash": "abc",
                        "signing_authorities": ["Test Authority"],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def enrolled(tmp_path: Path):
    bundle = tmp_path / "Applications" / "Mail.app"
    executable = bundle / "Contents" / "MacOS" / "Mail"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"sealed executable")
    info = bundle / "Contents" / "Info.plist"
    with info.open("wb") as output:
        plistlib.dump(
            {
                "CFBundleIdentifier": "example.mail",
                "CFBundleShortVersionString": "1",
                "CFBundleVersion": "2",
            },
            output,
        )
    registry_path = _write_registry(tmp_path, executable=executable, bundle=bundle)

    def runner(argv, **_kwargs):
        if argv[0].endswith("codesign") and "-dv" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "TeamIdentifier=TEAM\nCDHash=abc\nAuthority=Test Authority", ""
            )
        if argv[0].endswith("codesign") and "-dr" in argv:
            return subprocess.CompletedProcess(
                argv, 0, 'designated => identifier "example.mail"', ""
            )
        return subprocess.CompletedProcess(argv, 0, "accepted", "")

    registry = DesktopRegistry(
        registry_path, runner=runner, trusted_roots=[tmp_path / "Applications"], require_seal=False
    )
    registry.verify_all()
    row = {"pid": 42, "birth": "birth-a", "uid": 501, "command": str(executable)}
    return registry, row, executable, bundle


def test_enrolled_identity_requires_off_tick_binding_and_replays(enrolled):
    registry, row, *_ = enrolled
    assert registry.classify(row)["reason"] == "process_identity_not_revalidated"
    decision = registry.admit_process(row)
    assert decision["kind"] == "desktop_app"
    assert decision["role"] == "ordinary_mail_ui"
    assert registry.classify(row) == decision
    assert classify_exported(row, registry.read_export()) == decision


@pytest.mark.parametrize("change", ["uid", "path", "birth"])
def test_wrong_uid_path_or_pid_reuse_fails_closed(enrolled, change):
    registry, row, executable, _ = enrolled
    registry.admit_process(row)
    changed = dict(row)
    if change == "uid":
        changed["uid"] = 502
        expected = "uid_mismatch"
    elif change == "path":
        changed["command"] = str(executable.parent / "same-name-copy")
        expected = "missing_or_noncanonical_executable"
    else:
        changed["birth"] = "birth-b"
        expected = "process_identity_not_revalidated"
    assert registry.classify(changed)["reason"] == expected


def test_workload_precedence_applies_inside_enrolled_app(enrolled):
    registry, row, *_ = enrolled
    row["known_workload"] = True
    decision = registry.admit_process(row)
    assert decision["kind"] == "agent_workload"
    assert decision["reason"] == "known_workload"
    assert decision["identity_binding_changed"] is False


def test_exact_workload_cli_and_unknown_renderer_do_not_inherit(enrolled, tmp_path):
    registry, row, *_ = enrolled
    cli = dict(row, command="/usr/local/bin/claude")
    assert registry.classify(cli)["kind"] == "agent_workload"
    fake_renderer = (
        tmp_path / "Applications" / "Fake.app" / "Contents" / "MacOS" / "Fake (Renderer)"
    )
    fake_renderer.parent.mkdir(parents=True)
    fake_renderer.write_text("not enrolled")
    assert registry.classify(dict(row, command=str(fake_renderer)))["kind"] == "unknown"


def test_verified_codex_gui_is_admitted_but_an_unbound_instance_replays_unknown(enrolled):
    registry, row, executable, bundle = enrolled
    installed_codex = "/Applications/Codex.app/Contents/MacOS/Codex"
    registry._verified[installed_codex] = {}
    assert registry._workload_reason({"command": installed_codex}) is None
    assert (
        classify_exported(
            {"pid": 44, "birth": "unbound", "uid": 501, "command": installed_codex},
            registry.read_export(),
        )["kind"]
        == "unknown"
    )
    codex = bundle / "Contents" / "MacOS" / "Codex"
    codex.write_bytes(b"sealed codex executable")
    identity = dict(registry._verified[str(executable)])
    identity.update(
        {
            "executable": str(codex),
            "sha256": hashlib.sha256(codex.read_bytes()).hexdigest(),
            "file_identity": {
                "device": codex.stat().st_dev,
                "inode": codex.stat().st_ino,
                "size": codex.stat().st_size,
                "mtime_ns": codex.stat().st_mtime_ns,
                "ctime_ns": codex.stat().st_ctime_ns,
            },
        }
    )
    registry._verified[str(codex)] = identity
    codex_row = {**row, "pid": 43, "birth": "codex", "command": str(codex)}
    assert registry.classify(codex_row)["kind"] == "unknown"
    assert registry.admit_process(codex_row)["kind"] == "desktop_app"
    wrong_uid = dict(codex_row, uid=502)
    assert registry.classify(wrong_uid)["kind"] == "unknown"
    assert classify_exported(wrong_uid, registry.read_export())["kind"] == "unknown"
    assert (
        classify_exported(dict(codex_row, pid=44, birth="unbound"), registry.read_export())["kind"]
        == "unknown"
    )
    assert (
        registry.classify(dict(codex_row, command="/usr/local/bin/codex"))["kind"]
        == "agent_workload"
    )


@pytest.mark.parametrize(
    "command",
    [
        "/Applications/Unverified Codex.app/Contents/MacOS/Codex",
        "/Applications/Unverified Claude.app/Contents/MacOS/Claude",
        "/usr/local/bin/codex",
        "/usr/local/bin/claude",
    ],
)
def test_unverified_gui_and_cli_workload_names_match_live_and_exported_replay(enrolled, command):
    registry, row, *_ = enrolled
    observed = {**row, "command": command}
    assert registry.classify(observed)["kind"] == "agent_workload"
    assert classify_exported(observed, registry.read_export())["kind"] == "agent_workload"


def test_replacement_symlink_and_traversal_fail_closed(enrolled, tmp_path):
    registry, row, executable, bundle = enrolled
    registry.admit_process(row)
    executable.write_bytes(b"replaced")
    assert registry.classify(row)["reason"] == "executable_replaced"
    outside = tmp_path / "outside"
    outside.write_text("outside")
    executable.unlink()
    executable.symlink_to(outside)
    # The first replacement is retained as the terminal process reason.  A
    # fresh PID/birth still reports the direct filesystem defect.
    assert registry.classify(row)["reason"] == "executable_replaced"
    assert registry.classify({**row, "pid": 43, "birth": "fresh"})["reason"] == (
        "symlink_or_noncanonical_executable"
    )
    bad = _write_registry(
        tmp_path, executable=bundle / "Contents" / "MacOS" / "Mail", bundle=bundle
    )
    payload = json.loads(bad.read_text())
    payload["applications"][0]["executables"][0]["relative_executable_path"] = "../outside"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="traverses"):
        DesktopRegistry(
            bad, trusted_roots=[tmp_path / "Applications"], require_seal=False
        ).verify_all()


def test_replacement_revokes_active_binding_but_preserves_prior_replay_history(enrolled):
    registry, row, executable, _ = enrolled
    assert registry.admit_process(row)["kind"] == "desktop_app"
    prior = dict(row)
    executable.write_bytes(b"replaced")
    changed = registry.classify(row)
    assert changed["identity_binding_changed"] is True
    assert registry.classify(row)["identity_binding_changed"] is True
    export = registry.read_export()
    assert export["bindings"] == []
    assert len(export["binding_history"]) == 1
    assert export["revoked_bindings"][0]["reason"] == "executable_replaced"
    assert classify_exported(prior, export)["kind"] == "desktop_app"
    assert classify_exported({**row, "classification": changed}, export)["kind"] == "unknown"
    # A replaced process remains revoked even if its original bytes return;
    # subsequent off-tick admission cannot bless the same PID/birth attempt.
    repeated = registry.admit_process(row)
    assert repeated["reason"] == "executable_replaced"
    assert repeated["identity_binding_changed"] is True
    executable.write_bytes(b"sealed executable")
    restored = registry.admit_process(row)
    assert restored["reason"] == "executable_replaced"
    assert restored["identity_binding_changed"] is True
    assert registry.read_export()["bindings"] == []


def _cpu_inventory(at: int, row: dict, cpu_seconds: float) -> dict:
    return {
        "time_ns": at * 1_000_000_000,
        "capture_start_ns": at * 1_000_000_000 - 100,
        "capture_end_ns": at * 1_000_000_000 + 100,
        "processes": [{**row, "ppid": 1, "cpu_seconds": cpu_seconds}],
    }


def _desktop_host(
    registry: DesktopRegistry, samples: list[dict], observations: list[dict] | None = None
):
    evidence = registry.read_export()
    return {
        "load_policy": host_load.DESKTOP_POLICY,
        "desktop_registry": evidence["registry"],
        "desktop_registry_sha256": evidence["registry_sha256"],
        "desktop_registry_evidence": evidence,
        "cpu_samples": samples,
        "observations": observations or [],
    }


def _paired_enrolled_registry(enrolled, tmp_path: Path) -> tuple[DesktopRegistry, dict, dict]:
    """Build two separately enrolled UI executables for exec-transition coverage."""
    registry, row, executable, bundle = enrolled
    other_bundle = tmp_path / "Applications" / "Browser.app"
    other = other_bundle / "Contents" / "MacOS" / "Browser"
    other.parent.mkdir(parents=True)
    other.write_bytes(executable.read_bytes())
    with (other_bundle / "Contents" / "Info.plist").open("wb") as output:
        plistlib.dump(
            {
                "CFBundleIdentifier": "example.browser",
                "CFBundleShortVersionString": "1",
                "CFBundleVersion": "2",
            },
            output,
        )
    path = _write_registry(
        tmp_path,
        executable=executable,
        bundle=bundle,
    )
    payload = json.loads(path.read_text())
    payload["applications"].append(
        {
            "id": "browser",
            "display_name": "Browser",
            "bundle_path": str(other_bundle),
            "bundle_id": "example.browser",
            "version": "1",
            "build": "2",
            "team_identifier": "TEAM",
            "executables": [
                {
                    "role": "ordinary_browser_ui",
                    "relative_executable_path": str(other.relative_to(other_bundle)),
                    "sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
                    "designated_requirement": 'designated => identifier "example.browser"',
                    "cdhash": "abc",
                    "signing_authorities": ["Test Authority"],
                }
            ],
        }
    )
    path.write_text(json.dumps(payload))

    def paired_runner(argv, **kwargs):
        if str(other) in argv and argv[0].endswith("codesign") and "-dr" in argv:
            return subprocess.CompletedProcess(
                argv, 0, 'designated => identifier "example.browser"', ""
            )
        return registry._runner(argv, **kwargs)

    paired = DesktopRegistry(
        path,
        uid=501,
        runner=paired_runner,
        trusted_roots=[bundle.parent, other_bundle.parent],
        require_seal=False,
    )
    paired.verify_all()
    return paired, row, {**row, "command": str(other)}


_TRANSITION_STATES = (
    "invalid",
    "desktop_app",
    "workload",
    "uid_mismatch",
    "system_service",
    "other_enrolled_app",
)

_STATE_KINDS = {
    "invalid": "unknown",
    "desktop_app": "desktop_app",
    "workload": "agent_workload",
    "uid_mismatch": "unknown",
    "system_service": "macos_service",
    "other_enrolled_app": "desktop_app",
}


def _transition_state(label: str, row: dict, other: dict) -> dict:
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
    raise AssertionError(f"unknown transition state {label}")


@pytest.mark.parametrize(
    ("before_label", "after_label"),
    [(before, after) for before in _TRANSITION_STATES for after in _TRANSITION_STATES],
)
def test_real_collector_replays_all_desktop_transition_pairs(
    enrolled, tmp_path, before_label, after_label
):
    """Exercise every state-machine edge through collection and reducer replay."""
    registry, row, other = _paired_enrolled_registry(enrolled, tmp_path)
    before_row = _transition_state(before_label, row, other)
    after_row = _transition_state(after_label, row, other)
    # Simulate the preceding ordinary point observation: eligible, unchanged
    # identities may have been admitted, but transition collection must not
    # create a new binding for the after identity.
    if before_label in {"desktop_app", "other_enrolled_app"}:
        assert registry.admit_process(before_row)["kind"] == "desktop_app"
    sample = host_load.cpu_interval(
        _cpu_inventory(0, before_row, 1.0),
        _cpu_inventory(1, after_row, 1.1),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    process = sample["processes"][0]
    transition = before_label != after_label
    expected_kind = (
        "agent_workload"
        if transition and "workload" in {before_label, after_label}
        else "unknown" if transition else _STATE_KINDS[after_label]
    )
    assert process["kind"] == expected_kind
    if transition:
        assert process["interval_transition"]["previous"] == {
            **before_row,
            "ppid": 1,
            "cpu_seconds": 1.0,
        }
        assert (
            process["interval_transition"]["previous_classification"]["kind"]
            == _STATE_KINDS[before_label]
        )
        assert process["classification"]["identity_binding_changed"] is True
        assert registry.read_export()["bindings"] == []
        history = registry.read_export()["binding_history"]
        assert len(history) == int(before_label in {"desktop_app", "other_enrolled_app"})
        # Neither the transitioned current row nor a repeated off-tick call
        # can re-admit this PID/birth for the current registry attempt.
        for candidate in (after_row, before_row, row, other):
            repeated = registry.admit_process(candidate)
            assert repeated["identity_binding_changed"] is True
            assert repeated["kind"] not in {"desktop_app", "desktop_renderer"}
            assert registry.read_export()["bindings"] == []
            assert registry.read_export()["binding_history"] == history
    else:
        assert "interval_transition" not in process
    host = _desktop_host(
        registry,
        [sample],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    reduced = background_report._samples(host)
    assert reduced[0]["processes"][0]["kind"] == expected_kind
    if transition:
        with pytest.raises(
            ValueError, match="desktop identity binding changed in interval evidence"
        ):
            background_report.validate_desktop_host(host)
    else:
        background_report.validate_desktop_host(host)


@pytest.mark.parametrize(("field", "value"), [("pid", []), ("birth", [])])
def test_malformed_process_identity_fails_closed_without_state_machine_crash(
    enrolled, field, value
):
    registry, row, *_ = enrolled
    malformed = {**row, field: value}
    assert registry.admit_process(malformed)["kind"] == "unknown"
    assert registry.classify(malformed)["kind"] == "unknown"


def test_desktop_exec_to_usr_bin_true_is_unknown_and_replays_to_identity_guard(enrolled):
    """The review's literal system-looking exec example remains fail-closed."""
    registry, row, *_ = enrolled
    assert registry.admit_process(row)["kind"] == "desktop_app"
    after = {**row, "command": "/usr/bin/true"}
    sample = host_load.cpu_interval(
        _cpu_inventory(0, row, 1.0),
        _cpu_inventory(1, after, 1.1),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    process = sample["processes"][0]
    assert process["kind"] == "unknown"
    assert process["classification"]["identity_binding_changed"] is True
    host = _desktop_host(
        registry,
        [sample],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    assert background_report._samples(host)[0]["processes"][0]["kind"] == "unknown"
    with pytest.raises(ValueError, match="desktop identity binding changed in interval evidence"):
        background_report.validate_desktop_host(host)


def test_real_collector_replacement_reducer_reports_identity_guard_not_replay_mismatch(enrolled):
    registry, row, executable, _ = enrolled
    registry.admit_process(row)
    prior = host_load.cpu_interval(
        _cpu_inventory(0, row, 1.0),
        _cpu_inventory(1, row, 1.1),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    executable.write_bytes(b"replaced after valid interval")
    invalid = host_load.cpu_interval(
        _cpu_inventory(1, row, 1.1),
        _cpu_inventory(2, row, 1.2),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    invalid_process = invalid["processes"][0]
    assert invalid_process["classification"]["identity_binding_changed"] is True
    host = _desktop_host(registry, [prior, invalid])
    # The old sample still replays through historical binding evidence, and the
    # changed one reaches the explicit guard instead of failing as a mismatch.
    assert len(background_report._samples(host)) == 2
    with pytest.raises(ValueError, match="desktop identity binding changed in interval evidence"):
        background_report.validate_desktop_host(host)
    point_host = _desktop_host(
        registry,
        [prior],
        [
            {
                "classified_processes": [{**invalid_process, "cpu_percent": 1.0}],
                "probe_exempt_pids": [],
            }
        ],
    )
    with pytest.raises(ValueError, match="desktop identity binding changed in point evidence"):
        background_report.validate_desktop_host(point_host)


def test_registry_static_enrollment_is_real_and_ui_roles_only():
    path = (
        Path(__file__).parents[1] / "research/bl_served_search/DESKTOP_APP_REGISTRY_20261027.json"
    )
    payload = json.loads(path.read_text())
    assert payload["policy"] == POLICY
    names = {app["display_name"] for app in payload["applications"]}
    assert {"Microsoft Outlook", "Claude", "Codex desktop", "Google Chrome"} <= names
    roles = {role["role"] for app in payload["applications"] for role in app["executables"]}
    assert all(
        "gpu" not in role and "service" not in role and "plugin" not in role for role in roles
    )


def test_export_policy_mismatch_and_tampered_binding_fail_closed(enrolled):
    registry, row, *_ = enrolled
    registry.admit_process(row)
    export = registry.read_export()
    assert classify_exported(row, {**export, "policy": "classified-v1"})["kind"] == "unknown"
    export["bindings"][0]["identity"] = None
    assert classify_exported(row, export)["kind"] == "unknown"


def test_existing_same_name_copy_and_unenrolled_helper_cannot_inherit(enrolled, tmp_path):
    registry, row, executable, bundle = enrolled
    registry.admit_process(row)
    for index, copied in enumerate(
        (
            tmp_path / "Applications" / "Copied.app" / "Contents" / "MacOS" / executable.name,
            bundle / "Contents" / "MacOS" / "Unenrolled Helper",
        ),
        start=1,
    ):
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(executable.read_bytes())
        decision = registry.admit_process(
            {**row, "pid": row["pid"] + index, "birth": f"copy-{index}", "command": str(copied)}
        )
        assert decision["kind"] == "unknown"
        assert decision["reason"] == "unenrolled_or_unverified_executable"


def test_explicit_signed_renderer_role_admits_and_replays(enrolled, tmp_path):
    registry, row, executable, bundle = enrolled
    renderer = (
        bundle
        / "Contents"
        / "Frameworks"
        / "Mail Helper (Renderer).app"
        / "Contents"
        / "MacOS"
        / "Mail Helper (Renderer)"
    )
    renderer.parent.mkdir(parents=True)
    renderer.write_bytes(executable.read_bytes())
    path = _write_registry(tmp_path, executable=renderer, bundle=bundle)
    payload = json.loads(path.read_text())
    payload["applications"][0]["executables"][0]["role"] = "remote_chat_renderer"
    path.write_text(json.dumps(payload))
    enrolled_renderer = DesktopRegistry(
        path, uid=501, runner=registry._runner, trusted_roots=[bundle.parent], require_seal=False
    )
    enrolled_renderer.verify_all()
    process = {**row, "pid": 43, "birth": "renderer", "command": str(renderer)}
    decision = enrolled_renderer.admit_process(process)
    assert decision["kind"] == "desktop_renderer"
    assert enrolled_renderer.classify(process) == decision
    assert classify_exported(process, enrolled_renderer.read_export()) == decision


@pytest.mark.parametrize(
    "signature",
    [
        "TeamIdentifier=WRONG\nCDHash=abc\nAuthority=Test Authority",
        "TeamIdentifier=TEAM\nCDHash=wrong\nAuthority=Test Authority",
        "Signature=adhoc\nTeamIdentifier=TEAM\nCDHash=abc\nAuthority=Test Authority",
    ],
)
def test_wrong_signing_identity_fails_verification(enrolled, signature):
    registry, *_ = enrolled
    registry._runner = lambda argv, **_: subprocess.CompletedProcess(
        argv,
        0,
        (
            signature
            if "-dv" in argv
            else 'designated => identifier "example.mail"' if "-dr" in argv else "accepted"
        ),
        "",
    )
    with pytest.raises(ValueError):
        registry.verify_all()


def test_replay_rejects_forged_outside_path_and_known_workload(enrolled, tmp_path):
    registry, row, *_ = enrolled
    registry.admit_process(row)
    export = registry.read_export()
    forged = json.loads(json.dumps(export))
    outside = str(tmp_path / "Microsoft Outlook")
    forged["bindings"][0]["command"] = outside
    forged["bindings"][0]["identity"]["executable"] = outside
    forged["verified"] = {outside: forged["bindings"][0]["identity"]}
    assert classify_exported(dict(row, command=outside), forged)["kind"] == "unknown"
    assert classify_exported(dict(row, known_workload=True), export)["kind"] == "agent_workload"


def test_classify_after_admission_does_not_hash(enrolled, monkeypatch):
    registry, row, *_ = enrolled
    registry.admit_process(row)
    monkeypatch.setattr(
        "research.bl_served_search.desktop_apps._sha256", lambda _: pytest.fail("tick hash")
    )
    assert registry.classify(row)["kind"] == "desktop_app"


def test_pid_reuse_is_not_identity_change_but_same_birth_exec_is(enrolled):
    registry, row, *_ = enrolled
    registry.admit_process(row)
    assert (
        registry.classify(dict(row, birth="new-birth", command="/usr/local/bin/claude"))[
            "identity_binding_changed"
        ]
        is False
    )
    assert (
        registry.classify(dict(row, command="/usr/local/bin/claude"))["identity_binding_changed"]
        is True
    )
