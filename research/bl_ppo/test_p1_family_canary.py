"""Evidence admission failures for the P1 canary-to-source handoff."""

from __future__ import annotations

import json

import pytest

from research.bl_ppo import p1_family_canary as canary
from research.bl_ppo.exports import digest, export_json
from research.colab_cli.supervisor import Refused


def test_historical_pass_does_not_discharge_changed_transport(tmp_path, monkeypatch):
    from research.colab_cli import admission

    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"status": "PASS"}))

    def refuse(_result, _repo):
        raise Refused("final-only acceptance is not bound to the current transport source")

    monkeypatch.setattr(admission, "require_final_only", refuse)
    status = canary.acceptance_status(acceptance)
    assert status["reported_status"] == "PASS"
    assert status["accounting_item"] == "HOLD"
    assert status["sha256"] == digest(acceptance)


@pytest.mark.parametrize("changed", ["source-template.json", "canary-manifest.json"])
def test_preflight_refuses_mutated_preparation_before_qualification(tmp_path, changed):
    for name in ("source-template.json", "canary-manifest.json"):
        (tmp_path / name).write_text("{}")
    export_json(
        tmp_path / "preparation.json",
        {
            "source_template_sha256": digest(tmp_path / "source-template.json"),
            "canary_manifest_sha256": digest(tmp_path / "canary-manifest.json"),
        },
    )
    (tmp_path / changed).write_text('{"mutated":true}')
    with pytest.raises(ValueError, match="differs from the frozen preparation"):
        canary.preflight(tmp_path, tmp_path / "absent-acceptance", tmp_path / "absent-review")


def test_admit_cannot_bind_a_receipt_from_another_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(canary, "preflight", lambda *args: {})
    with pytest.raises(ValueError, match="this preparation's real broker"):
        canary.admit(tmp_path, tmp_path / "other/receipt.json", tmp_path / "a", tmp_path / "r")
    assert not (tmp_path / "source-admitted.json").exists()
