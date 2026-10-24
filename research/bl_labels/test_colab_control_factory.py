"""Shared classifier integration with startup/runtime/recovery process policy."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from research.bl_labels import factory_persistence as fp
from research.bl_labels import slowdown_resume as sr
from research.bl_labels.exports import publish_json
from research.bl_labels.test_factory_persistence import row, valid_receipt


@dataclass(frozen=True)
class Identity:
    role: str = "supervisor"


def fixture(monkeypatch, *, admitted=True):
    records = {(100, "birth-100"): SimpleNamespace(pane_pid=None)}
    monkeypatch.setattr(fp.colab_control, "load_admissions", lambda: records)

    def candidates(rows, records):
        rows = list(rows)
        found = {pid for pid, _ in records}
        while True:
            children = {r["pid"] for r in rows if r["ppid"] in found}
            if children <= found:
                return found & {r["pid"] for r in rows}
            found.update(children)

    monkeypatch.setattr(fp.colab_control, "candidate_pids", candidates)
    monkeypatch.setattr(
        fp.colab_control, "annotate", lambda *args: {100: Identity()} if admitted else {}
    )
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": f"birth-{pid}",
            "birth_precision": "native tick",
            "executable": "/usr/bin/python3" if pid == 100 else "/usr/bin/opaque",
        },
    )


def test_admitted_root_and_generic_numerical_descendant_stays_foreign(monkeypatch):
    fixture(monkeypatch)
    tracker = fp.PersistenceTracker("tree")
    table = "100 1 0 python -m research.colab_cli.supervisor\n101 100 0 opaque\n"
    for sample in range(6):
        rows = fp.process_rows(lambda _: table, tracker.known)
        tracker.observe(rows, sample * 5, {100, 101})
    assert all(r["ownership_reason"] == "non_owned" for r in tracker.known.values())
    assert (100, "birth-100") not in tracker.pending
    with pytest.raises(fp.ProcessMonitorStop, match="pid 101"):
        tracker.observe(fp.process_rows(lambda _: table, tracker.known), 30, {100, 101})


def test_reparented_generic_control_descendant_cannot_disappear(monkeypatch):
    fixture(monkeypatch)
    tracker = fp.PersistenceTracker("reparent")
    initial = "100 1 0 python -m research.colab_cli.supervisor\n101 100 0 opaque\n"
    tracker.observe(fp.process_rows(lambda _: initial, {}), 0, set())
    monkeypatch.setattr(fp.colab_control, "annotate", lambda *args: {})
    for sample in range(1, 6):
        table = "101 1 0 opaque\n"
        tracker.observe(fp.process_rows(lambda _: table, tracker.known), sample * 5, set())
    with pytest.raises(fp.ProcessMonitorStop, match="pid 101"):
        tracker.observe(fp.process_rows(lambda _: table, tracker.known), 30, set())


def test_later_exemption_never_clears_a_latched_stop():
    tracker = fp.PersistenceTracker("latched")
    for sample in range(6):
        tracker.observe([row()], sample * 5, set())
    with pytest.raises(fp.ProcessMonitorStop):
        tracker.observe([row()], 30, set())
    with pytest.raises(fp.ProcessMonitorStop):
        tracker.observe([{**row(), "control_identity": {"role": "supervisor"}}], 35, set())


def test_exemption_clears_only_unlatched_streak_and_redacts_native_rpc_vector():
    tracker = fp.PersistenceTracker("redaction")
    tracker.observe([row()], 0, set())
    accepted = {
        **row(),
        "control_tree": True,
        "control_identity": {"role": "cli"},
        "argv_vector": ["colab", "exec", "SECRET_RPC_PAYLOAD"],
    }
    tracker.observe([accepted], 5, set())
    assert not tracker.pending
    assert any(r["decision"] == "clear_control" for r in tracker.records)
    assert "SECRET_RPC_PAYLOAD" not in str(tracker.records)


@pytest.mark.parametrize("field", ["control_policy_id", "control_ruling_sha256"])
def test_v4_recovery_requires_control_binding(tmp_path, monkeypatch, field):
    value = valid_receipt(tmp_path, monkeypatch)
    value.pop(field)
    publish_json(tmp_path / "slowdown-host-check-101.json", value)
    assert not sr._host_check_after(tmp_path, 100)


def test_reused_root_pid_does_not_inspect_unrelated_generic_descendants(monkeypatch):
    fixture(monkeypatch, admitted=False)
    inspected = []

    def native(pid):
        inspected.append(pid)
        assert pid == 100, "stale root must not seed its unrelated descendants"
        return {"birth": "reused", "executable": "/usr/bin/python3"}

    monkeypatch.setattr(fp, "_native_identity", native)
    rows = fp.process_rows(lambda _: "100 1 15 python\n101 100 0 opaque\n", {})
    assert inspected == [100]
    assert not any(row.get("control_tree") for row in rows)
