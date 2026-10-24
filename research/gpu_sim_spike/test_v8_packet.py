"""Offline contracts for the V8 packet; no CUDA, allocation, or checkpoint loading."""

from __future__ import annotations

import hashlib
import json

import pytest

from research.gpu_sim_spike import v8_units

COMMIT = "a" * 40


def test_v8_units_are_isolated_by_mode_and_resume_only_complete_payloads(tmp_path):
    ns = v8_units.namespace(COMMIT)
    cells = v8_units.allocation()
    assert len(cells) == 9
    v8_units.publish_unit(tmp_path, ns, cells, cells[0], {"ok": True})
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    ledger = recovery / "ledger.json"
    v8_units.build_ledger(tmp_path, ledger, commit=COMMIT)
    assert v8_units.load_ledger(ledger, ns, cells) == {cells[0]: {"ok": True}}
    assert v8_units.namespace(COMMIT, canary=True) != ns
    assert all(cell.endswith("-e1") for cell in v8_units.allocation(smoke=True))


def test_ledger_copies_and_rechecks_bound_recorded_fixture(tmp_path):
    ns, cells = v8_units.namespace(COMMIT), v8_units.allocation()
    fixture = tmp_path / "recorded-states" / "one.npz"
    fixture.parent.mkdir()
    fixture.write_bytes(b"fixture")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    fixture.with_name(fixture.name + ".sha256").write_text(f"{digest}  {fixture.name}\n")
    fixture.with_name(fixture.name + ".complete").write_text("complete\n")
    content = {"recorded_fixture": {"path": "recorded-states/one.npz", "sha256": digest, "size": 7}}
    v8_units.publish_unit(tmp_path, ns, cells, cells[0], content)
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    ledger = recovery / "ledger.json"
    v8_units.build_ledger(tmp_path, ledger, commit=COMMIT)
    assert (recovery / "recorded-states/one.npz").read_bytes() == b"fixture"
    assert v8_units.load_ledger(ledger, ns, cells)[cells[0]] == content


def test_ledger_refuses_foreign_wrapper_duplicate_and_corrupt_fixture(tmp_path):
    ns, cells = v8_units.namespace(COMMIT), v8_units.allocation()
    v8_units.publish_unit(tmp_path, ns, cells, cells[0], {"ok": True})
    payload = v8_units.payload_path(tmp_path, cells[0])
    body = json.loads(payload.read_text())
    body["namespace"] = "v8-indexed-" + "b" * 40 + "-A100-full"
    payload.unlink()
    payload.with_name(payload.name + ".sha256").unlink()
    payload.with_name(payload.name + ".complete").unlink()
    v8_units.publish_json(payload, body)
    with pytest.raises(ValueError, match="foreign"):
        v8_units.build_ledger(tmp_path, tmp_path / "recover" / "ledger.json", commit=COMMIT)
