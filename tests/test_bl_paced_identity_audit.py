"""Targeted clock-proof audit checks using real sealed fake-clock journals."""

from copy import deepcopy

import pytest

from research.bl_paced_identity import audit
from research.bl_paced_identity.events import sha256
from research.bl_paced_identity.fixtures import run_adversarial_fixtures


@pytest.mark.parametrize("mutation", [None, "proof", "decision", "inventory"])
def test_audit_rechecks_clock_case_journals(tmp_path, monkeypatch, mutation):
    fixtures = run_adversarial_fixtures(tmp_path / "interleavings")
    receipt = {
        "source_sha256": {},
        "reference_sha256": {},
        "oracles": {"semantic": {}, "replay": {"verdict": "PASS"}, "paced": {"verdict": "NOT_RUN"}},
        "verdict": "INCOMPLETE",
        "l3_identity_prerequisites": "BLOCKED",
        "clock_seam": {**deepcopy(fixtures["clock_seam"]), "source_binding_sha256": sha256({})},
    }
    if mutation == "proof":
        receipt["clock_seam"]["cases"]["publication-late"]["accepted"] = True
    elif mutation == "decision":
        row = next(row for row in fixtures["fixtures"] if row["fixture_id"] == "clock-seam-on-time")
        row["decisions"][0]["accepted"] = False
    elif mutation == "inventory":
        fixtures["fixtures"] = [
            row for row in fixtures["fixtures"] if row["fixture_id"] != "clock-seam-wrong-frame"
        ]
    exports = {
        "receipt.json": receipt,
        "manifest.json": {"receipt_sha256": sha256(receipt), "dispatchable": False},
        "semantic.json": {},
        "replay.json": {"verdict": "PASS"},
        "fixtures.json": fixtures,
    }
    monkeypatch.setattr(audit, "read_export", lambda path: exports[path.name])
    monkeypatch.setattr(audit, "replay_journal", lambda *_: {"verdict": "PASS"})
    if mutation is None:
        assert audit.audit_preflight(tmp_path)["audit"] == "PASS"
    else:
        with pytest.raises(ValueError, match="clock seam|fixture decisions"):
            audit.audit_preflight(tmp_path)
