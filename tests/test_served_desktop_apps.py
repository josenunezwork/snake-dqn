"""Offline conformance checks for sealed classified-desktop-v1 admission."""

from __future__ import annotations

import hashlib
import json
import plistlib
import subprocess
from pathlib import Path

import pytest

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


def test_replacement_symlink_and_traversal_fail_closed(enrolled, tmp_path):
    registry, row, executable, bundle = enrolled
    registry.admit_process(row)
    executable.write_bytes(b"replaced")
    assert registry.classify(row)["reason"] == "executable_replaced"
    outside = tmp_path / "outside"
    outside.write_text("outside")
    executable.unlink()
    executable.symlink_to(outside)
    assert registry.classify(row)["reason"] == "symlink_or_noncanonical_executable"
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
    for copied in (
        tmp_path / "Applications" / "Copied.app" / "Contents" / "MacOS" / executable.name,
        bundle / "Contents" / "MacOS" / "Unenrolled Helper",
    ):
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(executable.read_bytes())
        decision = registry.admit_process({**row, "command": str(copied)})
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
