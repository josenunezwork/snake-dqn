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


@pytest.mark.parametrize(
    ("before_label", "after_label"),
    [
        (before, after)
        for before in (
            "invalid",
            "workload",
            "uid_mismatch",
            "system_service",
            "unvalidated_enrolled",
        )
        for after in _TRANSITION_STATES
    ],
)
def test_multi_interval_unadmitted_transition_and_steady_replay(
    enrolled, tmp_path, before_label, after_label
):
    """Unadmitted processes that transition remain revoked in follow-up intervals and replay."""
    from research.bl_served_search.desktop_apps import classify_row

    registry, row, other = _paired_enrolled_registry(enrolled, tmp_path)
    before_row = (
        dict(row)
        if before_label == "unvalidated_enrolled"
        else _transition_state(before_label, row, other)
    )
    after_row = _transition_state(after_label, row, other)
    transition = (before_row["uid"], before_row["command"]) != (
        after_row["uid"],
        after_row["command"],
    )
    if not transition:
        return
    inv0 = _cpu_inventory(0, before_row, 1.0)
    inv1 = _cpu_inventory(1, before_row, 1.1)
    if before_label == "unvalidated_enrolled":
        assert registry.classify(before_row)["reason"] == "process_identity_not_revalidated"
        inv1["_process_history"] = registry.history_snapshot()
        samples = []
    else:
        s1 = host_load.cpu_interval(inv0, inv1, owner_pid=999_999, desktop_classifier=registry)
        samples = [s1]
    inv2 = _cpu_inventory(2, after_row, 1.2)
    s2 = host_load.cpu_interval(inv1, inv2, owner_pid=999_999, desktop_classifier=registry)
    inv3 = _cpu_inventory(3, after_row, 1.3)
    s3 = host_load.cpu_interval(inv2, inv3, owner_pid=999_999, desktop_classifier=registry)
    samples.extend([s2, s3])

    p2 = s2["processes"][0]
    p3 = s3["processes"][0]
    expected_p2_kind = "agent_workload" if "workload" in {before_label, after_label} else "unknown"
    expected_p3_kind = "agent_workload" if after_label == "workload" else "unknown"
    assert p2["kind"] == expected_p2_kind
    assert p2["classification"]["identity_binding_changed"] is True
    assert p3["kind"] == expected_p3_kind
    assert p3["classification"]["identity_binding_changed"] is True

    export = registry.read_export()
    assert classify_exported(p3, export) == p3["classification"]
    bare_after = {k: after_row[k] for k in ("pid", "birth", "uid", "command")}
    replayed_bare = classify_exported(bare_after, export)
    assert replayed_bare["kind"] == expected_p3_kind
    assert replayed_bare["identity_binding_changed"] is True
    r_kind, r_hist, _ = classify_row(p3["history_before"], p3, export)
    assert r_kind == expected_p3_kind
    assert r_hist == p3["history_after"]

    host = _desktop_host(
        registry,
        samples,
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    reduced = background_report._samples(host)
    assert len(reduced) == len(samples)
    assert reduced[-1]["processes"][0]["kind"] == expected_p3_kind
    with pytest.raises(ValueError, match="desktop identity binding changed in interval evidence"):
        background_report.validate_desktop_host(host)


@pytest.mark.parametrize("initial_call", ["classify", "admit", "interval"])
def test_unbound_replaced_executable_revokes_and_file_restore_never_readmits(
    enrolled, initial_call
):
    """Replacing an enrolled binary before admission permanently revokes (pid, birth)."""
    registry, row, executable, _ = enrolled
    sealed_bytes = executable.read_bytes()
    executable.write_bytes(b"replaced before admission")

    if initial_call == "classify":
        first = registry.classify(row)
        assert first["kind"] == "unknown"
        assert first["reason"] == "executable_replaced"
        assert first["identity_binding_changed"] is True
    elif initial_call == "admit":
        first = registry.admit_process(row)
        assert first["kind"] == "unknown"
        assert first["reason"] == "executable_replaced"
        assert first["identity_binding_changed"] is True
    else:
        inv0 = _cpu_inventory(0, row, 1.0)
        inv1 = _cpu_inventory(1, row, 1.1)
        s1 = host_load.cpu_interval(inv0, inv1, owner_pid=999_999, desktop_classifier=registry)
        first = s1["processes"][0]["classification"]
        assert first["kind"] == "unknown"
        assert first["reason"] == "executable_replaced"
        assert first["identity_binding_changed"] is True

    # Restoring the sealed binary on disk must NEVER allow the revoked (pid, birth)
    # to be admitted later in the attempt, while a fresh (pid, birth) is admitted.
    executable.write_bytes(sealed_bytes)
    for attempt in (registry.admit_process(row), registry.classify(row)):
        assert attempt["kind"] == "unknown"
        assert attempt["reason"] == "executable_replaced"
        assert attempt["identity_binding_changed"] is True
    assert registry.read_export()["bindings"] == []

    inv2 = _cpu_inventory(2, row, 1.2)
    inv3 = _cpu_inventory(3, row, 1.3)
    s_restored = host_load.cpu_interval(inv2, inv3, owner_pid=999_999, desktop_classifier=registry)
    proc_restored = s_restored["processes"][0]
    assert proc_restored["kind"] == "unknown"
    assert proc_restored["classification"]["reason"] == "executable_replaced"
    assert proc_restored["classification"]["identity_binding_changed"] is True

    fresh_row = {**row, "pid": 99, "birth": "fresh-birth-after-restore"}
    fresh_decision = registry.admit_process(fresh_row)
    assert fresh_decision["kind"] == "desktop_app"
    assert fresh_decision["reason"] == "enrolled_verified_ui_role"

    host = _desktop_host(
        registry,
        [s_restored],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    reduced = background_report._samples(host)
    assert reduced[0]["processes"][0]["kind"] == "unknown"
    with pytest.raises(ValueError, match="desktop identity binding changed in interval evidence"):
        background_report.validate_desktop_host(host)


def test_bound_revoked_then_file_restored_multi_interval_never_readmits(enrolled):
    """Bound -> replaced -> restored across three intervals keeps old (pid,birth) revoked."""
    registry, row, executable, _ = enrolled
    sealed_bytes = executable.read_bytes()
    inv0 = _cpu_inventory(0, row, 1.0)
    inv1 = _cpu_inventory(1, row, 1.1)
    s1 = host_load.cpu_interval(inv0, inv1, owner_pid=999_999, desktop_classifier=registry)
    assert s1["processes"][0]["kind"] == "desktop_app"

    executable.write_bytes(b"tampered binary")
    inv2 = _cpu_inventory(2, row, 1.2)
    s2 = host_load.cpu_interval(inv1, inv2, owner_pid=999_999, desktop_classifier=registry)
    assert s2["processes"][0]["kind"] == "unknown"
    assert s2["processes"][0]["classification"]["reason"] == "executable_replaced"
    assert s2["processes"][0]["classification"]["identity_binding_changed"] is True

    executable.write_bytes(sealed_bytes)
    fresh = {**row, "pid": 77, "birth": "fresh-birth"}
    inv2_with_fresh = {
        **inv2,
        "processes": [*inv2["processes"], {**fresh, "ppid": 1, "cpu_seconds": 0.5}],
    }
    inv3_with_fresh = {
        "time_ns": 3_000_000_000,
        "capture_start_ns": 3_000_000_000 - 100,
        "capture_end_ns": 3_000_000_000 + 100,
        "processes": [
            {**row, "ppid": 1, "cpu_seconds": 1.3},
            {**fresh, "ppid": 1, "cpu_seconds": 0.6},
        ],
    }
    s3 = host_load.cpu_interval(
        inv2_with_fresh, inv3_with_fresh, owner_pid=999_999, desktop_classifier=registry
    )
    by_pid = {p["pid"]: p for p in s3["processes"]}
    assert by_pid[row["pid"]]["kind"] == "unknown"
    assert by_pid[row["pid"]]["classification"]["reason"] == "executable_replaced"
    assert by_pid[row["pid"]]["classification"]["identity_binding_changed"] is True
    assert by_pid[fresh["pid"]]["kind"] == "desktop_app"

    host = _desktop_host(
        registry,
        [s1, s2, s3],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    reduced = background_report._samples(host)
    assert len(reduced) == 3
    with pytest.raises(ValueError, match="desktop identity binding changed in interval evidence"):
        background_report.validate_desktop_host(host)


def test_previous_row_classified_against_history_as_of_its_own_time(
    enrolled, tmp_path, monkeypatch
):
    """Point check at t1 or disk deletion at t1 cannot corrupt t0 previous_classification."""
    from unittest.mock import Mock

    import psutil

    from research.bl_served_search import probe_host

    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=lambda **kw: []))
    registry, row, other = _paired_enrolled_registry(enrolled, tmp_path)
    assert registry.admit_process(row)["kind"] == "desktop_app"
    sysctl_row = {**row, "command": "/usr/sbin/sysctl"}

    # Point check runs at t1 on the transitioned process BEFORE cpu_interval(t0, t1).
    monkeypatch.setattr(
        probe_host,
        "_desktop_point_inventory",
        lambda: [{**sysctl_row, "ppid": 1, "cpu_percent": 1.0}],
    )
    screen = {"processes": "PID PPID %CPU COMM\n42 1 1.0 /usr/sbin/sysctl\n"}
    violations = probe_host.foreign_load(
        screen, 999_999, classified=True, desktop_classifier=registry, admit_desktop=True
    )
    assert len(violations) == 1
    assert violations[0]["classification"]["identity_binding_changed"] is True

    # Even if the t0 executable is deleted on disk at t1, old at t0 was a valid bound
    # desktop_app as of t0.
    Path(row["command"]).unlink()
    s1 = host_load.cpu_interval(
        _cpu_inventory(0, row, 1.0),
        _cpu_inventory(1, sysctl_row, 1.1),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    p1 = s1["processes"][0]
    prev_cls_1 = p1["interval_transition"]["previous_classification"]
    assert prev_cls_1["kind"] == "desktop_app"
    assert prev_cls_1["reason"] == "enrolled_verified_ui_role"
    assert not prev_cls_1.get("identity_binding_changed")
    assert p1["classification"]["identity_binding_changed"] is True

    # Second transition t1 -> t2 (sysctl -> Browser): at t1 sysctl WAS already revoked,
    # so previous_classification in [t1, t2] must record identity_binding_changed=True.
    inv1 = _cpu_inventory(1, sysctl_row, 1.1)
    inv1["_process_history"] = registry.history_snapshot()
    s2 = host_load.cpu_interval(
        inv1,
        _cpu_inventory(2, other, 1.2),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    p2 = s2["processes"][0]
    prev_cls_2 = p2["interval_transition"]["previous_classification"]
    assert prev_cls_2["kind"] == "unknown"
    assert prev_cls_2["identity_binding_changed"] is True
    assert p2["classification"]["identity_binding_changed"] is True

    host = _desktop_host(registry, [s1, s2], [{"process_screen": screen}])
    reduced = background_report._samples(host)
    assert len(reduced) == 2
    with pytest.raises(ValueError, match="desktop identity binding changed in interval evidence"):
        background_report.validate_desktop_host(host)


def test_seeded_randomized_collector_vs_reducer_equivalence_and_tamper_rejection(
    enrolled, tmp_path
):
    """Randomized multi-interval lifecycles never diverge between collector and reducer."""
    import copy
    import random

    from research.bl_served_search.desktop_apps import classify_row, decision_from_history

    base_registry, row, other = _paired_enrolled_registry(enrolled, tmp_path)
    mail_path = Path(row["command"])
    sealed_bytes = mail_path.read_bytes()

    for seed in range(25):
        rng = random.Random(20261027 + seed)
        mail_path.write_bytes(sealed_bytes)
        registry = DesktopRegistry(
            base_registry.path,
            uid=501,
            runner=base_registry._runner,
            trusted_roots=base_registry._trusted_roots,
            require_seal=False,
        )
        registry.verify_all()
        guard = host_load.LoadGuard(desktop_policy=True)
        guard_failed = False

        active: dict[tuple[int, str], dict] = {}
        next_pid = 100
        previous_inv = None
        samples = []
        pure_history: dict[tuple[int, str], dict] = {}

        for step in range(6):
            if rng.random() < 0.25:
                mail_path.write_bytes(b"mutated-bytes" if rng.random() < 0.5 else sealed_bytes)
            for proc_key in list(active):
                if rng.random() < 0.2 and len(active) > 1:
                    active.pop(proc_key)
                    continue
                cur = active[proc_key]
                cpu_s = cur["cpu_seconds"] + rng.choice([0.0, 0.02, 0.05, 0.25])
                if rng.random() < 0.45:
                    label = rng.choice(_TRANSITION_STATES)
                    cand = _transition_state(label, row, other)
                    cur = {
                        "pid": proc_key[0],
                        "birth": proc_key[1],
                        "uid": cand["uid"],
                        "command": cand["command"],
                        "ppid": 1,
                        "cpu_seconds": cpu_s,
                    }
                    if rng.random() < 0.15:
                        cur["known_workload"] = True
                else:
                    cur = {**cur, "cpu_seconds": cpu_s}
                active[proc_key] = cur

            for _ in range(rng.randint(1, 2)):
                label = rng.choice(_TRANSITION_STATES)
                cand = _transition_state(label, row, other)
                pid = next_pid
                next_pid += 1
                birth = f"b-{seed}-{pid}-{step}"
                proc_row = {
                    "pid": pid,
                    "birth": birth,
                    "uid": cand["uid"],
                    "command": cand["command"],
                    "ppid": 1,
                    "cpu_seconds": rng.choice([0.01, 0.05, 0.25]),
                }
                if rng.random() < 0.1:
                    proc_row["known_workload"] = True
                active[(pid, birth)] = proc_row

            at_ns = (step + 1) * 1_000_000_000
            current_inv = {
                "time_ns": at_ns,
                "capture_start_ns": at_ns - 100,
                "capture_end_ns": at_ns + 100,
                "processes": [dict(v) for v in active.values()],
            }
            if previous_inv is not None:
                sample = host_load.cpu_interval(
                    previous_inv, current_inv, owner_pid=999_999, desktop_classifier=registry
                )
                samples.append(sample)
                try:
                    guard.check(sample)
                except RuntimeError:
                    guard_failed = True
                for proc in sample["processes"]:
                    pkey = (proc["pid"], proc["birth"])
                    if "interval_transition" in proc:
                        tr = proc["interval_transition"]
                        prev_in = {
                            **tr["previous"],
                            "admit": False,
                            "executable_status": tr["previous_executable_status"],
                        }
                        pk, ph, pr = classify_row(tr["previous_history_before"], prev_in, registry)
                        assert ph == tr["previous_history_after"]
                        assert decision_from_history(pk, ph, pr) == tr["previous_classification"]
                        ck, ch, cr = classify_row(ph, {**proc, "admit": False}, registry)
                        assert ch == proc["history_after"]
                    else:
                        ck, ch, cr = classify_row(
                            proc["history_before"],
                            {**proc, "admit": proc["admit"]},
                            registry,
                        )
                        assert ch == proc["history_after"]
                        assert decision_from_history(ck, ch, cr) == proc["classification"]
                    if pkey in pure_history:
                        expected_start = (
                            proc["interval_transition"]["previous_history_before"]
                            if "interval_transition" in proc
                            else proc["history_before"]
                        )
                        assert expected_start == pure_history[pkey]
                    pure_history[pkey] = proc["history_after"]
            else:
                for proc_row in current_inv["processes"]:
                    if rng.random() < 0.5:
                        registry.admit_process(proc_row)
                    else:
                        registry.classify(proc_row)
                current_inv["_process_history"] = registry.history_snapshot()
                pure_history = registry.history_snapshot()
            previous_inv = current_inv

        host = _desktop_host(
            registry,
            samples,
            [{"classified_processes": [], "probe_exempt_pids": []}],
        )
        reduced = background_report._samples(host)
        assert len(reduced) == len(samples)
        for s_orig, s_red in zip(samples, reduced):
            assert [p["kind"] for p in s_orig["processes"]] == [
                p["kind"] for p in s_red["processes"]
            ]
            assert [p["classification"] for p in s_orig["processes"]] == [
                p["classification"] for p in s_red["processes"]
            ]
        if guard_failed:
            with pytest.raises(ValueError):
                background_report.validate_desktop_host(host)
        else:
            background_report.validate_desktop_host(host)
            assert background_report.desktop_overall(host)["sample_count"] == len(samples)

        # Tampering with any exported classification or history must be rejected by reducer.
        tampered = copy.deepcopy(host)
        target_proc = tampered["cpu_samples"][-1]["processes"][0]
        target_proc["classification"]["reason"] = "forged_reason"
        with pytest.raises(ValueError):
            background_report._samples(tampered)
        mail_path.write_bytes(sealed_bytes)


def test_argv_change_and_capture_failure_permanently_revoke_admitted_process(enrolled):
    """Finding 1: argv changes and argv capture failures permanently revoke an admitted process."""
    registry, row, executable, _ = enrolled
    exe = str(executable)
    ui_row = {**row, "argv": [exe, "--ordinary-ui"]}
    bench_row = {**row, "argv": [exe, "--benchmark"]}

    # 1. Point path: [exe, "--ordinary-ui"] -> [exe, "--benchmark"] -> [exe, "--ordinary-ui"]
    admitted = registry.admit_process(ui_row)
    assert admitted["kind"] == "desktop_app"
    assert admitted["reason"] == "enrolled_verified_ui_role"

    changed = registry.classify(bench_row)
    assert changed["kind"] == "unknown"
    assert changed["reason"] == "identity_binding_changed"
    assert changed["identity_binding_changed"] is True

    # Restoring the original argv cannot re-admit the revoked (pid, birth).
    for restored in (registry.classify(ui_row), registry.admit_process(ui_row)):
        assert restored["kind"] == "unknown"
        assert restored["identity_binding_changed"] is True

    # 2. Point path: admitted process loses argv capture (argv_identity_unavailable).
    row2 = {**row, "pid": 43, "birth": "birth-b"}
    ui_row2 = {**row2, "argv": [exe, "--ordinary-ui"]}
    unavail_row2 = {
        **row2,
        "argv": [],
        "identity_observation_error": "argv_identity_unavailable",
    }
    assert registry.admit_process(ui_row2)["kind"] == "desktop_app"
    lost = registry.classify(unavail_row2)
    assert lost["kind"] == "unknown"
    assert lost["reason"] == "argv_identity_unavailable"
    assert lost["identity_binding_changed"] is True
    for restored in (registry.classify(ui_row2), registry.admit_process(ui_row2)):
        assert restored["kind"] == "unknown"
        assert restored["identity_binding_changed"] is True

    # 3. Interval path: argv transition and argv capture failure transition both revoke and replay.
    row3 = {**row, "pid": 44, "birth": "birth-c"}
    ui_row3 = {**row3, "argv": [exe, "--ordinary-ui"]}
    bench_row3 = {**row3, "argv": [exe, "--benchmark"]}
    s1 = host_load.cpu_interval(
        _cpu_inventory(0, ui_row3, 1.0),
        _cpu_inventory(1, ui_row3, 1.1),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    assert s1["processes"][0]["kind"] == "desktop_app"
    s2 = host_load.cpu_interval(
        _cpu_inventory(1, ui_row3, 1.1),
        _cpu_inventory(2, bench_row3, 1.2),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    p2 = s2["processes"][0]
    assert p2["kind"] == "unknown"
    assert p2["classification"]["identity_binding_changed"] is True
    assert p2["interval_transition"]["reason"] == "argv_changed_no_delta"
    s3 = host_load.cpu_interval(
        _cpu_inventory(2, bench_row3, 1.2),
        _cpu_inventory(3, ui_row3, 1.3),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    assert s3["processes"][0]["kind"] == "unknown"
    assert s3["processes"][0]["classification"]["identity_binding_changed"] is True

    host = _desktop_host(
        registry,
        [s1, s2, s3],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    assert len(background_report._samples(host)) == 3
    with pytest.raises(ValueError, match="desktop identity binding changed in interval evidence"):
        background_report.validate_desktop_host(host)


def test_workload_detected_when_argv_capture_fails_via_argv_display_cwd_and_ancestry(enrolled):
    """Finding 2 & 5: missing argv keeps workload detection and protected macOS services."""
    registry, row, executable, _ = enrolled
    exe = str(executable)

    # Concrete reproduction from Finding 2: argv=[], error set, argv_display has /Projects/ path.
    with_argv = {
        **row,
        "pid": 50,
        "birth": "b-argv",
        "argv": [exe, "/Users/owner/Projects/run.js"],
    }
    with_display_only = {
        **row,
        "pid": 51,
        "birth": "b-display",
        "argv": [],
        "identity_observation_error": "argv_identity_unavailable",
        "argv_display": f"{exe} /Users/owner/Projects/run.js",
    }
    assert registry.admit_process(with_argv)["kind"] == "agent_workload"
    decision_display = registry.admit_process(with_display_only)
    assert decision_display["kind"] == "agent_workload"
    assert classify_exported(with_display_only, registry.read_export())["kind"] == "agent_workload"

    # Workload detection from cwd and ancestry when argv capture fails.
    with_cwd = {
        **row,
        "pid": 52,
        "birth": "b-cwd",
        "argv": [],
        "identity_observation_error": "argv_identity_unavailable",
        "cwd": "/Users/owner/Projects/snake-dqn",
    }
    with_ancestry = {
        **row,
        "pid": 53,
        "birth": "b-ancestry",
        "argv": [],
        "identity_observation_error": "argv_identity_unavailable",
        "ancestry": [{"command": "/opt/homebrew/bin/node", "argv_display": "node worker.js"}],
    }
    assert registry.admit_process(with_cwd)["kind"] == "agent_workload"
    assert registry.admit_process(with_ancestry)["kind"] == "agent_workload"

    # Missing argv without workload evidence never admits as desktop_app.
    missing_argv = {
        **row,
        "pid": 54,
        "birth": "b-missing-argv",
        "argv": [],
        "identity_observation_error": "argv_identity_unavailable",
        "argv_display": exe,
    }
    missing_decision = registry.admit_process(missing_argv)
    assert missing_decision["kind"] == "unknown"
    assert missing_decision["reason"] == "argv_identity_unavailable"

    # Protected macOS services whose privileged argv cannot be captured remain macos_service.
    for idx, service_cmd in enumerate(
        (
            "/usr/libexec/syspolicyd",
            "/usr/libexec/trustd",
            "/System/Library/PrivateFrameworks/FileProvider.framework/Support/fileproviderd",
        ),
        start=60,
    ):
        svc_row = {
            "pid": idx,
            "birth": f"svc-{idx}",
            "uid": 0,
            "command": service_cmd,
            "argv": [],
            "identity_observation_error": "argv_identity_unavailable",
            "argv_display": service_cmd,
        }
        svc_decision = registry.admit_process(svc_row)
        assert svc_decision["kind"] == "macos_service"
        assert svc_decision["reason"] == "system_executable_root"
        assert registry.classify(svc_row)["kind"] == "macos_service"
        assert classify_exported(svc_row, registry.read_export())["kind"] == "macos_service"


def test_history_replay_rejects_superseded_identities_history_restarts_and_omitted_transitions(
    enrolled,
):
    """Finding 3: reject superseded states, history restarts/forks, and omitted transitions."""
    import copy

    registry, row, *_ = enrolled
    sysctl_row = {
        "pid": 42,
        "birth": "b1",
        "uid": 0,
        "command": "/usr/sbin/sysctl",
        "argv": ["/usr/sbin/sysctl"],
    }
    s1 = host_load.cpu_interval(
        _cpu_inventory(0, sysctl_row, 1.0),
        _cpu_inventory(1, sysctl_row, 1.1),
        owner_pid=999_999,
        desktop_classifier=registry,
    )
    service_proc = copy.deepcopy(s1["processes"][0])
    assert service_proc["kind"] == "macos_service"

    # Observe (42, b1) running /usr/local/bin/python3 at 2s.
    py_row = {
        "pid": 42,
        "birth": "b1",
        "uid": 0,
        "command": "/usr/local/bin/python3",
        "argv": ["/usr/local/bin/python3"],
        "observation_ns": 2_000_000_000,
    }
    revoked = registry.classify(py_row)
    assert revoked["kind"] == "agent_workload"
    assert revoked["identity_binding_changed"] is True

    # Reuse the earlier [0s, 1s] service row in an interval [3s, 4s] and a point row at 3s.
    reused_interval_sample = {
        "start_ns": 3_000_000_000,
        "end_ns": 4_000_000_000,
        "capture_start_ns": 3_000_000_000,
        "capture_end_ns": 4_000_000_000,
        "coverage_complete": True,
        "process_churn": [],
        "background_cpu_percent": service_proc["cpu_percent"],
        "workload_cpu_percent": 0.0,
        "unknown_cpu_percent": 0.0,
        "macos_service_cpu_percent": service_proc["cpu_percent"],
        "desktop_renderer_cpu_percent": 0.0,
        "desktop_app_cpu_percent": 0.0,
        "processes": [{**service_proc, "observation_ns": 4_000_000_000}],
    }
    host_interval_reused = _desktop_host(
        registry,
        [s1, reused_interval_sample],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    with pytest.raises(ValueError, match="superseded|continuity"):
        background_report.validate_desktop_host(host_interval_reused)

    # Also test when only [3s, 4s] is in cpu_samples (so replayed_proc_history is empty at 3s).
    host_single_reused = _desktop_host(
        registry,
        [reused_interval_sample],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    with pytest.raises(ValueError, match="desktop process state superseded before observation"):
        background_report._samples(host_single_reused)

    # Point row at 3s reusing the superseded [0s, 1s] service state.
    point_reused = {**service_proc, "cpu_percent": 1.0, "observation_ns": 3_000_000_000}
    host_point_reused = _desktop_host(
        registry,
        [s1],
        [{"classified_processes": [point_reused], "probe_exempt_pids": []}],
    )
    with pytest.raises(ValueError, match="desktop process state superseded before observation"):
        background_report.validate_desktop_host(host_point_reused)

    # Appending {before: None, after: original_service_state} after revocation and resetting
    # process_history to original_service_state must be rejected by _validate_registry_history.
    export = registry.read_export()
    original_service_state = copy.deepcopy(export["history_log"][0]["after"])
    restarted = copy.deepcopy(export)
    restarted["history_log"].append(
        {
            "pid": 42,
            "birth": "b1",
            "observation_ns": 3_000_000_000,
            "before": None,
            "after": original_service_state,
            "row": copy.deepcopy(export["history_log"][0]["row"]),
        }
    )
    restarted["process_history"] = [original_service_state]
    with pytest.raises(ValueError, match="desktop history log discontinuity"):
        background_report._validate_registry_history(restarted)

    # Jumping back to an earlier before state (forking from Entry 0 after Entry 1) must fail.
    forked = copy.deepcopy(export)
    forked["history_log"].append(
        {
            "pid": 42,
            "birth": "b1",
            "observation_ns": 3_000_000_000,
            "before": original_service_state,
            "after": original_service_state,
            "row": copy.deepcopy(export["history_log"][0]["row"]),
        }
    )
    forked["process_history"] = [original_service_state]
    with pytest.raises(ValueError, match="desktop history log discontinuity"):
        background_report._validate_registry_history(forked)

    # Omitted-transition subcase: deleting interval_transition from sysctl -> unknown interval.
    registry2, *_ = enrolled
    unknown_row = {
        "pid": 77,
        "birth": "b-trans",
        "uid": 0,
        "command": "/tmp/unknown_binary",
        "argv": ["/tmp/unknown_binary"],
    }
    s_trans = host_load.cpu_interval(
        _cpu_inventory(0, {**sysctl_row, "pid": 77, "birth": "b-trans"}, 1.0),
        _cpu_inventory(1, unknown_row, 1.1),
        owner_pid=999_999,
        desktop_classifier=registry2,
    )
    host_omitted = _desktop_host(
        registry2,
        [s_trans],
        [{"classified_processes": [], "probe_exempt_pids": []}],
    )
    del host_omitted["cpu_samples"][0]["processes"][0]["interval_transition"]
    with pytest.raises(ValueError, match="desktop interval transition evidence missing"):
        background_report._samples(host_omitted)


def test_classify_row_is_pure_and_live_exported_semantics_match_identically(enrolled):
    """Finding 4: classify_row is pure (no disk I/O) and identical for live and exported inputs."""
    from research.bl_served_search.desktop_apps import classify_row

    registry, row, executable, _ = enrolled
    same_row_with_admit_true = {**row, "admit": True}

    # Call classify_row before and after mutating executable on disk; output must not change
    # because classify_row is pure and performs zero filesystem access.
    out1 = classify_row(None, same_row_with_admit_true, registry)
    out1_exported = classify_row(None, same_row_with_admit_true, registry.read_export())
    assert out1 == out1_exported
    assert out1[0] == "desktop_app"
    assert out1[2] == "enrolled_verified_ui_role"
    assert registry._history == {}

    executable.write_bytes(b"mutated on disk")
    out2 = classify_row(None, same_row_with_admit_true, registry)
    out2_exported = classify_row(None, same_row_with_admit_true, registry.read_export())
    assert out2 == out1
    assert out2_exported == out1
    assert registry._history == {}

    # Off-tick normalization (normalize_row / admit_process) observes disk replacement and passes
    # immutable executable_status="executable_replaced" into pure classify_row.
    normalized = registry.normalize_row(row, admit=True)
    assert normalized["executable_status"] == "executable_replaced"
    out_replaced = classify_row(None, normalized, registry)
    assert out_replaced == classify_row(None, normalized, registry.read_export())
    assert out_replaced[0] == "unknown"
    assert out_replaced[2] == "executable_replaced"

    # Parity on uid=None between live DesktopRegistry and exported dict.
    uid_none_row = {**row, "uid": None, "admit": True}
    live_uid_none = classify_row(None, uid_none_row, registry)
    exported_uid_none = classify_row(None, uid_none_row, registry.read_export())
    assert live_uid_none == exported_uid_none
    assert live_uid_none[0] == "unknown"
    assert live_uid_none[2] == "desktop_process_identity_missing"
