"""Fixture tests for the L4 label-lane adapter (judgment §3, §4 steps 2/7/8).

Fake receipts and opaque fixture payloads only; no Colab, no real labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_labels import l4_delegation as dl
from research.bl_labels import l4_import as li
from research.bl_labels import l4_manifests as lm
from research.bl_labels import mac_driver
from research.bl_labels.exports import aggregate_attempts, empty_ledger, publish_json
from research.bl_labels.test_mac_integration import (
    _allocations,
    _install_fake_factory,
    _opaque_life,
    _patch_inline_runtime,
)

NS = "f0-fixture"
PIN = "c" * 40


def _control(count: int = 6) -> dict:
    return {
        "namespace": NS,
        "allocation": _allocations(count),
        "mac_gate": {"stage": "d4096", "collector_source_commit": PIN},
    }


def _commit_parent(root: Path, item: dict, deaths: int = 2) -> None:
    for life in range(deaths):
        publish_json(
            root / "lives" / f"{item['allocation_id']}--life-{life:06d}.json",
            _opaque_life(NS, item, life),
        )
    publish_json(
        root / "lives" / f"{item['allocation_id']}--session.json",
        {
            "schema": "bl-labels-f0-session/v1",
            "status": "COMPLETE",
            "namespace": NS,
            "allocation_id": item["allocation_id"],
            "allocation": item,
            "rows": [],
            "counters": {"deaths": deaths},
        },
    )


def _pass(_receipt, injection_manifest=None):
    return {
        "schema": "l4-label-faults/v2",
        "verdict": "PASS",
        "counted": [],
        "recovered_events": 0,
        "recovered_seconds": 0.0,
        "max_attempts": 1,
    }


def _l4_session(tmp: Path, control: dict, delegation_id: str, committed: list[int]) -> tuple:
    """Fake stopped L4 session over a delegation: control, output, receipt."""
    record = json.loads((tmp / "deleg" / f"delegation-{delegation_id}.json").read_text())
    by_id = {row["allocation_id"]: row for row in control["allocation"]}
    session = tmp / f"session-{delegation_id}"
    publish_json(
        session / "control.json",
        {
            "schema": "bl-labels-f0-control/v1",
            "namespace": NS,
            "allocation": [by_id[p] for p in record["parents"]],
            "workers": 12,
        },
    )
    output = session / "output"
    for index in committed:
        _commit_parent(output, control["allocation"][index])
    publish_json(
        output / "runtime-result.json",
        {
            "schema": "bl-labels-f0-runtime/v1",
            "namespace": NS,
            "status": "COMPLETE",
            "result_class": "SCIENCE",
            "operational_diagnostics": {"code_sha": PIN, "platform": "Linux-x86_64", "nproc": 12},
        },
    )
    receipt = session / "receipt.json"
    receipt.write_text(
        json.dumps({"job_id": "bl-l4-d4096-x", "stop_verified": True, "cleanup_error": None})
    )
    return session / "control.json", output, receipt


# --- delegation records -----------------------------------------------------------


def test_delegation_takes_contiguous_tail_and_caps_share(tmp_path: Path) -> None:
    control = _control(6)
    mac = tmp_path / "mac"
    _commit_parent(mac, control["allocation"][0])
    path = dl.write_delegation(tmp_path / "deleg", control, [mac], count=2)
    record = json.loads(path.read_text())
    assert record["parents"] == ["watch6-004", "watch6-005"]
    assert record["indices"] == [4, 5]
    assert record["collector_source_commit"] == PIN
    # remaining uncommitted = 5; outstanding 2 + 1 = 3 > 2.5 -> refused
    with pytest.raises(dl.DelegationRefused, match="50%"):
        dl.write_delegation(tmp_path / "deleg", control, [mac], count=1)


def test_delegation_refuses_overlap_and_non_tail(tmp_path: Path) -> None:
    control = _control(8)
    mac = tmp_path / "mac"
    _commit_parent(mac, control["allocation"][7])  # a started tail parent
    with pytest.raises(dl.DelegationRefused, match="overlaps"):
        dl.write_delegation(tmp_path / "deleg", control, [mac], start=6, end=7)
    with pytest.raises(dl.DelegationRefused, match="tail"):
        dl.write_delegation(tmp_path / "deleg", control, [mac], start=4, end=5)
    path = dl.write_delegation(tmp_path / "deleg", control, [mac], start=5, end=6)
    assert json.loads(path.read_text())["parents"] == ["watch6-005", "watch6-006"]
    with pytest.raises(dl.DelegationRefused, match="overlaps"):
        dl.write_delegation(tmp_path / "deleg", control, [mac], start=6, end=6)


def test_admitted_parent_is_never_delegated(tmp_path: Path) -> None:
    control = _control(4)
    gate = dl.MacDelegationGate(tmp_path / "deleg", NS, control["allocation"])
    assert gate.claim("watch6-003") is True
    with pytest.raises(dl.DelegationRefused, match="overlaps"):
        dl.write_delegation(tmp_path / "deleg", control, [], start=3, end=3)


def test_f1_ruler_pin_is_refused(tmp_path: Path) -> None:
    control = _control(4)
    control["mac_gate"]["collector_source_commit"] = dl.F1_RULER_PIN
    with pytest.raises(dl.DelegationRefused, match="2af48cd0"):
        dl.write_delegation(tmp_path / "deleg", control, [], count=1)


# --- Mac runner hook -------------------------------------------------------------


def _run(control, ledger, output, tmp: Path, **kwargs):
    b8, frp3 = tmp / "b8", tmp / "frp3"
    b8.write_bytes(b"b8")
    frp3.write_bytes(b"frp3")
    plain = {"namespace": control["namespace"], "allocation": control["allocation"]}
    return mac_driver.run(
        plain,
        ledger,
        b8,
        frp3,
        output,
        workers=1,
        timing_armed=tmp / "never-armed",
        deadline_seconds=1200,
        state_dir=tmp / f"state-{output.name}",
        **kwargs,
    )


def test_no_delegation_dir_is_byte_identical(monkeypatch, tmp_path: Path) -> None:
    _patch_inline_runtime(monkeypatch)
    _install_fake_factory(monkeypatch)
    control = _control(3)
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(NS, control["allocation"]))
    plain = _run(control, ledger, tmp_path / "plain", tmp_path)
    gated = _run(control, ledger, tmp_path / "gated", tmp_path, delegation_dir=tmp_path / "empty")
    assert plain["status"] == gated["status"] == "COMPLETE"
    assert "delegated_parents" not in plain and "delegation_dir" not in plain
    assert set(gated) - set(plain) == {"delegation_dir", "delegated_parents"}
    assert gated["delegated_parents"] == []
    files = sorted(p.name for p in (tmp_path / "plain" / "lives").iterdir())
    assert files == sorted(p.name for p in (tmp_path / "gated" / "lives").iterdir())
    for name in files:
        assert (tmp_path / "plain/lives" / name).read_bytes() == (
            tmp_path / "gated/lives" / name
        ).read_bytes()
    assert not (tmp_path / "plain" / dl.ADMIT_DIR).exists()


def test_skip_import_reclaim_resume_and_union_close(monkeypatch, tmp_path: Path) -> None:
    _patch_inline_runtime(monkeypatch)
    _install_fake_factory(monkeypatch)
    control = _control(6)
    deleg = tmp_path / "deleg"
    dl.write_delegation(deleg, control, [], count=2)  # watch6-004, watch6-005
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(NS, control["allocation"]))
    first = _run(control, ledger, tmp_path / "mac-01", tmp_path, delegation_dir=deleg)
    assert first["status"] == "COMPLETE_EXCEPT_DELEGATED"
    assert first["delegated_parents"] == ["watch6-004", "watch6-005"]
    names = {p.name for p in (tmp_path / "mac-01" / "lives").glob("*.json")}
    assert not any(n.startswith(("watch6-004", "watch6-005")) for n in names)

    # L4 committed watch6-004 only; import (Mac-side hash re-check) then reclaim.
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[4])
    union = tmp_path / "union"
    record_path = li.import_session(
        deleg,
        control,
        "01",
        s_control,
        s_output,
        receipt,
        [tmp_path / "mac-01"],
        union,
        classifier=_pass,
    )
    record = json.loads(record_path.read_text())
    assert record["committed_parents"] == ["watch6-004"]
    assert record["uncommitted_parents"] == ["watch6-005"]
    assert all(row["runtime_sha256"] == row["mac_sha256"] for row in record["units"])
    reclaim = dl.write_reclaim(deleg, control, "01", receipt, record_path)
    assert json.loads(reclaim.read_text())["reclaimed_parents"] == ["watch6-005"]

    # Mac resumes: only the reclaimed parent is admissible; L4's parent stays excluded.
    resume_ledger = tmp_path / "resume" / "verified-ledger.json"
    aggregate_attempts([tmp_path / "mac-01"], resume_ledger, NS, control["allocation"])
    second = _run(control, resume_ledger, tmp_path / "mac-02", tmp_path, delegation_dir=deleg)
    assert second["status"] == "COMPLETE_EXCEPT_DELEGATED"
    assert second["delegated_parents"] == ["watch6-004"]
    assert {p.name.split("--")[0] for p in (tmp_path / "mac-02/lives").glob("*.json")} == {
        "watch6-005"
    }

    result = li.union_close(
        control, [tmp_path / "mac-01", tmp_path / "mac-02"], deleg, union, tmp_path / "close"
    )
    assert result["verdict"] == "CLEAN", result
    assert result["prefix_length"] == 6
    assert result["platform_by_parent_counts"] == {"darwin-arm64": 5, "colab-l4-x86_64": 1}
    from research.bl_labels.exports import load_ledger

    assert (
        len(load_ledger(tmp_path / "close/verified-ledger.json", NS, control["allocation"])) == 18
    )


def test_gate_readmits_reclaimed_parent(tmp_path: Path) -> None:
    control = _control(6)
    deleg = tmp_path / "deleg"
    dl.write_delegation(deleg, control, [], count=1)
    gate = dl.MacDelegationGate(deleg, NS, control["allocation"])
    assert gate.claim("watch6-005") is False
    assert gate.admissible(["watch6-005"]) == set()
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[])
    record = li.import_session(
        deleg,
        control,
        "01",
        s_control,
        s_output,
        receipt,
        [],
        tmp_path / "union",
        classifier=_pass,
    )
    dl.write_reclaim(deleg, control, "01", receipt, record)
    assert gate.admissible(["watch6-005"]) == {"watch6-005"}
    assert gate.claim("watch6-005") is True


# --- import refusals -------------------------------------------------------------


def _delegated(tmp_path: Path, count: int = 6, block: int = 2) -> tuple[dict, Path]:
    control = _control(count)
    deleg = tmp_path / "deleg"
    dl.write_delegation(deleg, control, [], count=block)
    return control, deleg


def test_runtime_vs_mac_hash_mismatch_is_c5(tmp_path: Path) -> None:
    control, deleg = _delegated(tmp_path)
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[4])
    payload = s_output / "lives" / "watch6-004--life-000000.json"
    payload.write_bytes(payload.read_bytes().replace(b"COMPLETE", b"COMPLETX"))
    with pytest.raises(li.C5Breach, match="runtime sha256"):
        li.import_session(
            deleg,
            control,
            "01",
            s_control,
            s_output,
            receipt,
            [],
            tmp_path / "u",
            classifier=_pass,
        )


def test_duplicate_unit_is_c5_never_ignored(tmp_path: Path) -> None:
    control, deleg = _delegated(tmp_path)
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[4])
    mac = tmp_path / "mac"
    _commit_parent(mac, control["allocation"][4])  # Mac also ran the delegated parent
    with pytest.raises(li.C5Breach, match="committed twice"):
        li.import_session(
            deleg,
            control,
            "01",
            s_control,
            s_output,
            receipt,
            [mac],
            tmp_path / "u",
            classifier=_pass,
        )


def test_union_close_flags_duplicate_and_ledger_close(tmp_path: Path) -> None:
    control, deleg = _delegated(tmp_path)
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[4])
    union = tmp_path / "u"
    li.import_session(
        deleg, control, "01", s_control, s_output, receipt, [], union, classifier=_pass
    )
    mac = tmp_path / "mac"
    _commit_parent(mac, control["allocation"][4])
    result = li.union_close(
        control, [mac], deleg, union, tmp_path / "close", declared_pending=["watch6-000"]
    )
    assert result["verdict"] == "INVALID"
    assert any("duplicate Mac+L4" in b for b in result["c5_breaches"])
    assert any("ledger close" in b for b in result["c5_breaches"])
    assert not (tmp_path / "close" / "verified-ledger.json").exists()


def test_prefix_semantics_report_unused_l4(tmp_path: Path) -> None:
    control, deleg = _delegated(tmp_path)
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[4, 5])
    union = tmp_path / "u"
    li.import_session(
        deleg, control, "01", s_control, s_output, receipt, [], union, classifier=_pass
    )
    mac = tmp_path / "mac"
    for index in (0, 1, 3):
        _commit_parent(mac, control["allocation"][index])
    result = li.union_close(control, [mac], deleg, union, tmp_path / "close")
    assert result["verdict"] == "CLEAN"
    assert result["prefix_length"] == 2
    assert result["declared_pending"] == ["watch6-002"]
    assert result["unused_beyond_prefix_l4_parents"] == ["watch6-004", "watch6-005"]
    assert result["beyond_prefix_mac_parents"] == ["watch6-003"]
    assert li.contiguous_prefix(control["allocation"], {"watch6-000"}) == 1


def test_quarantine_and_v2_gates_refuse_import(tmp_path: Path) -> None:
    control, deleg = _delegated(tmp_path)
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[4])
    publish_json(s_output.parent / li.QUARANTINE_MARKER, {"schema": "q"})
    with pytest.raises(dl.DelegationRefused, match="quarantined"):
        li.import_session(
            deleg,
            control,
            "01",
            s_control,
            s_output,
            receipt,
            [],
            tmp_path / "u",
            classifier=_pass,
        )
    (s_output.parent / li.QUARANTINE_MARKER).unlink()

    def fail(classes):
        return lambda _r, injection_manifest=None: {
            "schema": "l4-label-faults/v2",
            "verdict": "FAIL",
            "counted": [{"class": c, "index": 1, "detail": "fixture"} for c in classes],
        }

    with pytest.raises(li.C5Breach, match="INVALID"):
        li.import_session(
            deleg,
            control,
            "01",
            s_control,
            s_output,
            receipt,
            [],
            tmp_path / "u",
            classifier=fail(["C4"]),
        )
    with pytest.raises(dl.DelegationRefused, match="verdict FAIL"):
        li.import_session(
            deleg,
            control,
            "01",
            s_control,
            s_output,
            receipt,
            [],
            tmp_path / "u",
            classifier=fail(["C2"]),
        )
    record = li.import_session(
        deleg,
        control,
        "01",
        s_control,
        s_output,
        receipt,
        [],
        tmp_path / "u",
        classifier=fail(["C2"]),
        allow_transport_fault_import=True,
    )
    assert json.loads(record.read_text())["status"] == "IMPORTED_AFTER_TRANSPORT_FAULT"


def test_import_refuses_unverified_stop_and_foreign_collector(tmp_path: Path) -> None:
    control, deleg = _delegated(tmp_path)
    s_control, s_output, receipt = _l4_session(tmp_path, control, "01", committed=[4])
    receipt.write_text(json.dumps({"stop_verified": False, "cleanup_error": None}))
    with pytest.raises(dl.DelegationRefused, match="verified stop"):
        li.import_session(
            deleg, control, "01", s_control, s_output, receipt, [], tmp_path / "u", classifier=_pass
        )


# --- manifests and identity packet -----------------------------------------------


def test_session_manifest_binds_collector_pin_and_l4(tmp_path: Path) -> None:
    control, deleg = _delegated(tmp_path)
    with pytest.raises(dl.DelegationRefused, match="identical collector"):
        lm.prepare_session(
            deleg,
            control,
            "01",
            tmp_path / "s0",
            commit="d" * 40,
            b8=tmp_path,
            frp3=tmp_path,
            online=False,
        )
    path = lm.prepare_session(
        deleg, control, "01", tmp_path / "s1", commit=PIN, b8=tmp_path, frp3=tmp_path, online=False
    )
    manifest = json.loads(path.read_text())
    assert manifest["commit"] == PIN
    assert manifest["requirements"]["allowed_gpus"] == ["L4"]
    assert manifest["l4_lane"]["fault_definition"] == "l4-label-faults/v2"
    sub = json.loads((tmp_path / "s1" / "control.json").read_text())
    assert [row["allocation_id"] for row in sub["allocation"]] == ["watch6-004", "watch6-005"]
    assert "tools/ops/dispatch.sh" in (tmp_path / "s1" / "DISPATCH.txt").read_text()


def test_identity_packet_select_quarantine_and_compare(tmp_path: Path) -> None:
    control = _control(40)
    mac = tmp_path / "mac"
    for item in control["allocation"][:38]:
        _commit_parent(mac, item)
    packet = tmp_path / "packet"
    lm.prepare_identity_packet(
        control, [mac], packet, commit=PIN, b8=tmp_path, frp3=tmp_path, online=False
    )
    expected = json.loads((packet / "mac-expected.json").read_text())
    assert len(expected["parents"]) == 36
    assert expected["parents"] == lm.select_identity_parents(control, [mac])
    assert (packet / li.QUARANTINE_MARKER).exists()
    manifest = json.loads((packet / "manifest.json").read_text())
    assert manifest["job_id"].startswith(li.QUARANTINE_JOB_PREFIX)

    output = packet / "output"
    by_id = {row["allocation_id"]: row for row in control["allocation"]}
    for parent in expected["parents"]:
        _commit_parent(output, by_id[parent])
    assert lm.compare_identity(packet, output, tmp_path, tmp_path)["verdict"] == "IDENTICAL_ALL"

    # A quarantined output can never be imported, even through a real delegation.
    deleg = tmp_path / "deleg"
    dl.write_delegation(deleg, control, [mac], count=1)
    with pytest.raises(dl.DelegationRefused, match="quarantined"):
        li.import_session(
            deleg,
            control,
            "01",
            packet / "control.json",
            output,
            tmp_path / "r.json",
            [mac],
            tmp_path / "u",
            classifier=_pass,
        )


def test_identity_compare_mismatch_emits_replay_audit(tmp_path: Path) -> None:
    control = _control(36)
    mac = tmp_path / "mac"
    for item in control["allocation"]:
        _commit_parent(mac, item)
    packet = tmp_path / "packet"
    lm.prepare_identity_packet(
        control, [mac], packet, commit=PIN, b8=tmp_path, frp3=tmp_path, online=False
    )
    output = tmp_path / "l4-out"
    for item in control["allocation"]:
        _commit_parent(
            output,
            (
                {**item, "seed": item["seed"] + 1000}
                if item["allocation_id"] == "watch6-007"
                else item
            ),
        )
    result = lm.compare_identity(packet, output, tmp_path / "b8", tmp_path / "frp3")
    assert result["verdict"] == "MISMATCH"
    assert result["mismatch"] and all(n.startswith("watch6-007") for n in result["mismatch"])
    assert "research.bl_labels.replay_audit" in result["replay_audit_command"]


def test_importer_default_classifier_is_the_real_v2_module() -> None:
    """The stubbed import tests above bind to the shared l4-label-faults/v2 classifier."""
    classify = li._default_classifier()
    call = {
        "command": "version",
        "status": "completed",
        "elapsed_seconds": 0.1,
        "timeout_seconds": 30.0,
    }
    ok = {
        "job_id": "j",
        "status": "PASS",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": None,
        "cli_calls": [call],
    }
    result = classify(ok, injection_manifest=None)
    assert result["schema"] == "l4-label-faults/v2" and result["verdict"] == "PASS"
    bad = classify(
        {**ok, "status": "INCOMPLETE", "failure_class": "SUPERVISOR"}, injection_manifest=None
    )
    assert bad["verdict"] == "FAIL" and bad["counted_by_class"]["C3"] == 1
