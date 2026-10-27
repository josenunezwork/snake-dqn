"""Event-history conformance tests for classified desktop identity policy."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import pytest

from research.bl_served_search import background_report, host_load
from research.bl_served_search.desktop_apps import (
    EMPTY_HISTORY,
    DesktopRegistry,
    classify_exported,
    classify_row,
    replay_history,
)


def _identity(path: Path, role: str = "ordinary_mail_ui") -> dict:
    stat = path.stat()
    bundle = path.parents[2]
    return {
        "entry": "mail",
        "application": "Mail",
        "role": role,
        "bundle_path": str(bundle),
        "executable": str(path),
        "bundle_id": "example.mail",
        "version": "1",
        "build": "2",
        "team_identifier": "TEAM",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "designated_requirement": 'designated => identifier "example.mail"',
        "cdhash": "abc",
        "signing_authorities": ["Test Authority"],
        "file_identity": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        },
    }


def _raw_verification(path: Path, *, digest: bool = False) -> dict:
    stat = path.stat()
    file_identity = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    evidence = {
        "ok": True,
        "canonical": str(path),
        "is_symlink": False,
        "file_identity": file_identity,
    }
    if digest:
        evidence |= {
            "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file_identity_after_digest": dict(file_identity),
        }
    return evidence


@pytest.fixture
def enrolled_history(tmp_path: Path):
    app = tmp_path / "Applications" / "Mail.app" / "Contents" / "MacOS"
    app.mkdir(parents=True)
    ui, renderer = app / "Mail", app / "Mail Renderer"
    ui.write_bytes(b"sealed ui")
    renderer.write_bytes(b"sealed renderer")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "served-tick-desktop-registry/v1",
                "policy": "classified-desktop-v1",
                "owner_uid": 501,
                "permitted_use_declaration": "ordinary mail only",
                "workload_exclusions": [{"exact_executable": "/usr/local/bin/claude"}],
                "applications": [
                    {
                        "id": "mail",
                        "display_name": "Mail",
                        "bundle_path": str(app.parents[1]),
                        "bundle_id": "example.mail",
                        "version": "1",
                        "build": "2",
                        "team_identifier": "TEAM",
                        "executables": [
                            {
                                "role": "ordinary_mail_ui",
                                "relative_executable_path": "Contents/MacOS/Mail",
                                "sha256": hashlib.sha256(ui.read_bytes()).hexdigest(),
                                "designated_requirement": 'designated => identifier "example.mail"',
                                "cdhash": "abc",
                                "signing_authorities": ["Test Authority"],
                            },
                            {
                                "role": "ordinary_browser_renderer",
                                "relative_executable_path": "Contents/MacOS/Mail Renderer",
                                "sha256": hashlib.sha256(renderer.read_bytes()).hexdigest(),
                                "designated_requirement": 'designated => identifier "example.mail"',
                                "cdhash": "abc",
                                "signing_authorities": ["Test Authority"],
                            },
                        ],
                    }
                ],
            }
        )
    )
    registry = DesktopRegistry(registry_path, uid=501, trusted_roots=[tmp_path], require_seal=False)
    registry._verified = {
        str(ui): _identity(ui),
        str(renderer): _identity(renderer, "ordinary_browser_renderer"),
    }
    return registry, ui, renderer


def _export(ui: Path, renderer: Path) -> dict:
    return {
        "policy": "classified-desktop-v1",
        "registry": {
            "owner_uid": 501,
            "workload_exclusions": [{"exact_executable": "/usr/local/bin/claude"}],
            "applications": [
                {
                    "id": "mail",
                    "display_name": "Mail",
                    "bundle_path": str(ui.parents[2]),
                    "bundle_id": "example.mail",
                    "version": "1",
                    "build": "2",
                    "team_identifier": "TEAM",
                    "executables": [
                        {
                            "role": "ordinary_mail_ui",
                            "relative_executable_path": "Contents/MacOS/Mail",
                            "sha256": hashlib.sha256(ui.read_bytes()).hexdigest(),
                            "designated_requirement": 'designated => identifier "example.mail"',
                            "cdhash": "abc",
                            "signing_authorities": ["Test Authority"],
                        },
                        {
                            "role": "ordinary_browser_renderer",
                            "relative_executable_path": "Contents/MacOS/Mail Renderer",
                            "sha256": hashlib.sha256(renderer.read_bytes()).hexdigest(),
                            "designated_requirement": 'designated => identifier "example.mail"',
                            "cdhash": "abc",
                            "signing_authorities": ["Test Authority"],
                        },
                    ],
                }
            ],
        },
        "verified": {
            str(ui): _identity(ui),
            str(renderer): _identity(renderer, "ordinary_browser_renderer"),
        },
    }


def _row(label: str, ui: Path, renderer: Path) -> dict:
    base = {"pid": 71, "birth": "birth-71", "uid": 501, "argv": []}
    if label == "desktop_app":
        return {
            **base,
            "command": str(ui),
            "admit": True,
            "verification": _raw_verification(ui, digest=True),
        }
    if label == "desktop_renderer":
        return {
            **base,
            "command": str(renderer),
            "admit": True,
            "verification": _raw_verification(renderer, digest=True),
        }
    if label == "invalid":
        return {
            **base,
            "command": "/unverified/missing",
            "verification": {"ok": True, "canonical_error": "missing"},
        }
    if label == "uid_mismatch":
        return {**base, "uid": 502, "command": str(ui), "verification": _raw_verification(ui)}
    if label == "workload":
        return {
            **base,
            "command": "/usr/local/bin/claude",
            "verification": {
                "ok": True,
                "canonical": "/usr/local/bin/claude",
                "is_symlink": False,
                "file_identity": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4, "ctime_ns": 5},
            },
        }
    if label == "system_service":
        return {
            **base,
            "command": "/usr/sbin/sysctl",
            "verification": {
                "ok": True,
                "canonical": "/usr/sbin/sysctl",
                "is_symlink": False,
                "file_identity": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4, "ctime_ns": 5},
            },
        }
    raise AssertionError(label)


_LABELS = (
    "desktop_app",
    "desktop_renderer",
    "invalid",
    "uid_mismatch",
    "workload",
    "system_service",
)
_INITIAL_KINDS = {
    "desktop_app": "desktop_app",
    "desktop_renderer": "desktop_renderer",
    "invalid": "unknown",
    "uid_mismatch": "unknown",
    "workload": "agent_workload",
    "system_service": "macos_service",
}


@pytest.mark.parametrize("before,after", [(a, b) for a in _LABELS for b in _LABELS])
def test_pure_classifier_table_covers_every_identity_pair(
    enrolled_history, before: str, after: str
):
    """All 36 pairs are explicit history transitions, never registry side effects."""
    _, ui, renderer = enrolled_history
    export = _export(ui, renderer)
    before_row, after_row = _row(before, ui, renderer), _row(after, ui, renderer)
    before_kind, history, _ = classify_row(EMPTY_HISTORY, before_row, export)
    after_kind, next_history, after_reason = classify_row(history, after_row, export)
    assert before_kind == _INITIAL_KINDS[before]
    assert (after_kind, next_history, after_reason) == classify_row(history, after_row, export)
    changed = before_row["uid"] != after_row["uid"] or before_row["command"] != after_row["command"]
    if changed and after != "workload":
        assert after_kind == "unknown"
        # A bad initial UID/verification is itself terminal; otherwise the
        # observed identity transition supplies the terminal reason.
        assert next_history.revoked_reason
    elif not changed:
        assert after_kind == _INITIAL_KINDS[after]


def test_unadmitted_transition_persists_through_steady_replay(enrolled_history):
    registry, ui, _ = enrolled_history
    first = registry.observe_snapshot(_row("invalid", ui, ui))
    changed = registry.observe_snapshot(_row("system_service", ui, ui))
    steady = registry.observe_snapshot(_row("system_service", ui, ui))
    export = registry.read_export()
    assert [
        first["classification"]["kind"],
        changed["classification"]["kind"],
        steady["classification"]["kind"],
    ] == ["unknown", "unknown", "unknown"]
    # The first ordinary failure is terminal without claiming a binding
    # changed; the subsequent exec transition is itself a binding change.
    assert first["classification"]["identity_binding_changed"] is False
    assert changed["classification"]["identity_binding_changed"] is True
    assert steady["classification"]["identity_binding_changed"] is True
    assert (
        replay_history(export)[steady["classification"]["history_index"]]
        == steady["classification"]
    )
    assert classify_exported(steady, export) == steady["classification"]


def test_replacement_of_unbound_process_is_terminal_after_restore(enrolled_history):
    registry, ui, _ = enrolled_history
    ui.write_bytes(b"replaced")
    replaced = registry.observe_snapshot(_row("desktop_app", ui, ui), admit=True)
    ui.write_bytes(b"sealed ui")
    restored = registry.observe_snapshot(_row("desktop_app", ui, ui), admit=True)
    # This represents the fresh off-tick verification needed before a new
    # process generation can be admitted after the file was restored.
    registry._verified[str(ui)] = _identity(ui)
    fresh = registry.observe_snapshot(
        {**_row("desktop_app", ui, ui), "birth": "fresh-birth"}, admit=True
    )
    assert replaced["classification"]["reason"] == "executable_replaced"
    assert restored["classification"]["kind"] == "unknown"
    assert restored["classification"]["identity_binding_changed"] is True
    assert fresh["classification"]["kind"] == "desktop_app"
    replay_history(registry.read_export())


def test_seeded_event_adapter_matches_independent_replay_and_rejects_tampering(enrolled_history):
    registry, ui, renderer = enrolled_history
    rng = random.Random(83017)
    snapshots = []
    for index in range(48):
        row = _row(rng.choice(_LABELS), ui, renderer)
        row["pid"] = 80 + rng.randrange(4)
        row["birth"] = f"birth-{row['pid']}"
        if rng.randrange(5) == 0:
            row["argv"] = ["--changed", str(index)]
        snapshots.append(registry.observe_snapshot(row, admit=bool(rng.randrange(2))))
    export = registry.read_export()
    replayed = replay_history(export)
    assert 0 < len(replayed) <= len(snapshots)
    for snapshot in snapshots:
        decision = snapshot["classification"]
        assert replayed[decision["history_index"]] == decision
        assert classify_exported(snapshot, export) == decision
    tampered = copy.deepcopy(export)
    tampered["history_sequence"][7]["classification"]["reason"] = "forged"
    with pytest.raises(ValueError, match="desktop history replay mismatch"):
        replay_history(tampered)
    assert classify_exported(snapshots[7], tampered)["kind"] == "unknown"


@pytest.mark.parametrize(
    "verification",
    [
        pytest.param(None, id="missing"),
        pytest.param({}, id="empty"),
        pytest.param({"ok": True}, id="missing-raw-facts"),
        pytest.param({"ok": False}, id="explicit-failure"),
        pytest.param({"ok": 0}, id="nonboolean-ok"),
        pytest.param({"ok": True, "file_identity": []}, id="malformed-file-identity"),
    ],
)
def test_malformed_verification_is_terminal_and_never_admits(enrolled_history, verification):
    _, ui, renderer = enrolled_history
    row = _row("desktop_app", ui, renderer)
    if verification is None:
        row.pop("verification")
    else:
        row["verification"] = verification
    kind, history, _ = classify_row(EMPTY_HISTORY, row, _export(ui, renderer))
    assert kind == "unknown"
    assert history.revoked_reason
    assert history.identity_binding_changed is False
    repeated, later, _ = classify_row(
        history, _row("desktop_app", ui, renderer), _export(ui, renderer)
    )
    assert repeated == "unknown"
    assert later.revoked_reason == history.revoked_reason


def test_compact_steady_events_replay_all_cpu_rows(enrolled_history):
    registry, ui, renderer = enrolled_history
    row = _row("desktop_app", ui, renderer)
    t0, t1, t2 = (_inventory(at, row, 1.0 + at / 10) for at in range(3))
    for snapshot in (t0, t1, t2):
        host_load.freeze_cpu_snapshot(snapshot, 999_999, registry)
    samples = [
        host_load.cpu_interval(t0, t1, 999_999, desktop_classifier=registry),
        host_load.cpu_interval(t1, t2, 999_999, desktop_classifier=registry),
    ]
    host = _host(registry, samples)
    assert len(host["desktop_registry_evidence"]["history_sequence"]) == 1
    reduced = background_report._samples(host)
    assert [[p["classification"] for p in sample["processes"]] for sample in reduced] == [
        [p["classification"] for p in sample["processes"]] for sample in samples
    ]


def test_old_event_reference_is_rejected_after_later_revocation(enrolled_history):
    registry, ui, _ = enrolled_history
    old = registry.observe_snapshot(
        {**_row("desktop_app", ui, ui), "observation_ns": 1}, admit=True
    )
    ui.write_bytes(b"replaced")
    registry.observe_snapshot({**_row("desktop_app", ui, ui), "observation_ns": 2}, admit=True)
    ui.write_bytes(b"sealed ui")
    export = registry.read_export()
    assert len(export["history_sequence"]) == 2
    replay_history(export)
    # A restored process cannot borrow the pre-revocation event merely because
    # its raw fields again match it; its later observation time exposes that
    # the referenced event has been superseded.
    restored = {**old, "observation_ns": 3}
    assert classify_exported(restored, export)["kind"] == "unknown"


def test_unenrolled_executable_replacement_is_terminal_after_restore(enrolled_history, tmp_path):
    """A process seen while an unregistered file changes cannot be re-admitted."""
    _, ui, renderer = enrolled_history
    executable = tmp_path / "ordinary-unenrolled"
    executable.write_bytes(b"original")
    row = {
        "pid": 991,
        "birth": "unenrolled-birth",
        "uid": 501,
        "command": str(executable),
        "argv": [],
        "admit": True,
        "verification": _raw_verification(executable, digest=True),
    }
    kind, history, _ = classify_row(EMPTY_HISTORY, row, _export(ui, renderer))
    assert kind == "unknown"
    executable.write_bytes(b"replaced")
    changed = {**row, "verification": _raw_verification(executable, digest=True)}
    kind, history, _ = classify_row(history, changed, _export(ui, renderer))
    assert kind == "unknown"
    assert history.revoked_reason == "executable_replaced"
    executable.write_bytes(b"original")
    restored = {**row, "verification": _raw_verification(executable, digest=True)}
    kind, final_history, _ = classify_row(history, restored, _export(ui, renderer))
    assert kind == "unknown"
    assert final_history.revoked_reason


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
    }


def test_seeded_collector_and_reducer_match_across_interleaved_intervals(enrolled_history):
    """Frozen snapshots retain own-time history despite later point observations."""
    registry, ui, renderer = enrolled_history
    rng = random.Random(140321)
    previous_row = _row("desktop_app", ui, renderer)
    previous = _inventory(0, previous_row, 1.0)
    # This point screen happens before interval collection and establishes the
    # immutable classification that must be reused for the old interval row.
    host_load.freeze_cpu_snapshot(previous, 999_999, registry)
    samples = []
    cpu = 1.0
    for index in range(1, 17):
        label = rng.choice(_LABELS)
        current_row = _row(label, ui, renderer)
        current_row["argv"] = ["--epoch", str(index)] if index % 4 == 0 else []
        if index == 11:
            # A real new generation is a fresh history stream, unlike a
            # recycled identity that has already been revoked.
            current_row["birth"] = "fresh-generation"
        cpu += 0.1
        current = _inventory(index, current_row, cpu)
        stored_previous = dict(previous["processes"][0]["classification"])
        if index % 3 == 0:
            # A point screen sees the new identity after the previous CPU
            # capture.  It must not retroactively change that stored row.
            registry.admit_process(current["processes"][0])
        host_load.freeze_cpu_snapshot(current, 999_999, registry)
        sample = host_load.cpu_interval(previous, current, 999_999, desktop_classifier=registry)
        if sample["processes"] and "interval_transition" in sample["processes"][0]:
            assert (
                sample["processes"][0]["interval_transition"]["previous_classification"]
                == stored_previous
            )
        samples.append(sample)
        previous = current
    host = _host(registry, samples)
    reduced = background_report._samples(host)
    assert len(reduced) == len(samples)
    for raw, replayed in zip(samples, reduced):
        assert [item["kind"] for item in replayed["processes"]] == [
            item["kind"] for item in raw["processes"]
        ]
        assert [item["classification"] for item in replayed["processes"]] == [
            item["classification"] for item in raw["processes"]
        ]
    # Event references, reasons, order, and complete history are all sealed by
    # the reducer replay rather than trusted from collected rows.
    for mutate in (
        lambda export: export["history_sequence"][0]["classification"].update(reason="forged"),
        lambda export: export["history_sequence"].reverse(),
        lambda export: export["history_sequence"].pop(),
    ):
        tampered = copy.deepcopy(host)
        mutate(tampered["desktop_registry_evidence"])
        with pytest.raises(ValueError):
            background_report._samples(tampered)


def test_collector_rejects_snapshot_not_frozen_at_capture_time(enrolled_history):
    registry, ui, renderer = enrolled_history
    previous = _inventory(0, _row("desktop_app", ui, renderer), 1.0)
    current = _inventory(1, _row("system_service", ui, renderer), 1.1)
    host_load.freeze_cpu_snapshot(current, 999_999, registry)
    with pytest.raises(ValueError, match="desktop CPU snapshot was not classified at capture time"):
        host_load.cpu_interval(previous, current, 999_999, desktop_classifier=registry)


def test_workload_verification_failure_exports_terminal_state_despite_same_decision(
    enrolled_history, tmp_path
):
    registry, _ui, _renderer = enrolled_history
    executable = tmp_path / "node"
    executable.write_bytes(b"workload executable")
    row = {
        "pid": 440,
        "birth": "workload-birth",
        "uid": 501,
        "command": str(executable),
        "argv": [],
    }
    before = registry.admit_process({**row, "observation_ns": 0})
    assert before["kind"] == "agent_workload"
    executable.unlink()
    failed = registry.admit_process({**row, "observation_ns": 1})
    assert failed["kind"] == "agent_workload"
    assert failed["reason"] == before["reason"]
    assert failed["history_index"] != before["history_index"]
    export = registry.read_export()
    assert export["histories"][-1]["revoked_reason"]
    assert replay_history(export)[failed["history_index"]] == failed
    executable.write_bytes(b"workload executable")
    restored = registry.admit_process({**row, "observation_ns": 2})
    assert registry.read_export()["histories"][-1]["revoked_reason"]
    assert replay_history(registry.read_export())[restored["history_index"]] == restored
