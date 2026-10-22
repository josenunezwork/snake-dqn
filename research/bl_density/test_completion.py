"""Completion packaging is immutable, outcome-blind, and doubly gated."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_density import completion, ledger, manifests
from research.bl_density.runner import DENSITY2_ALLOCATION, DENSITY2_NAMESPACE
from research.bl_ppo.exports import export_json


def _prepare_fixture(monkeypatch, tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    entries = [
        {"episode_id": f"{DENSITY2_NAMESPACE}/{arm}/{mix}/{i:03d}", "sha256": "1" * 64, "size": 5}
        for arm, cells in DENSITY2_ALLOCATION.items()
        for mix, count in cells.items()
        for i in range(count)
        if arm != "C-P1" and not (arm == "B70p" and mix == "mixed" and (i == 22 or i >= 25))
    ]
    value = {"schema": ledger.SCHEMA, "namespace": DENSITY2_NAMESPACE, "entries": entries}
    source = {
        "commit": completion.SCIENCE_COMMIT,
        "density": {"namespace": DENSITY2_NAMESPACE, "round14_amendment_sha256": "a" * 64},
        "env": {
            "BL_DENSITY_ROUND14_AMENDMENT_SHA256": "a" * 64,
            "BL_DENSITY_CANARY_PROOF_JSON": "stale",
        },
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12},
        "budget": {"rate_units_per_hour": 1.54, "other_assignments": {}},
        "script": "research/bl_density/colab.sh",
        "max_export_bytes": 67108864,
    }
    (prior / "manifest.json").write_text(json.dumps(source))
    (prior / "receipt.json").write_text("{}")
    ledger_path = tmp_path / "ledger.json"
    export_json(ledger_path, value)
    monkeypatch.setattr(ledger, "build_ledger", lambda *_a, **_kw: copy.deepcopy(value))
    monkeypatch.setattr(completion, "check_science", lambda *_a: None)
    monkeypatch.setattr(manifests, "source_paths", lambda *_a: ["research/bl_density/colab.sh"])
    monkeypatch.setattr(manifests, "_published_and_clean", lambda *_a, **_kw: None)
    monkeypatch.setattr(manifests, "assert_upload_budget", lambda *_a, **_kw: {})
    return prior, ledger_path


def test_prepare_preserves_science_and_same_bound_inputs(monkeypatch, tmp_path):
    prior, ledger_path = _prepare_fixture(monkeypatch, tmp_path)
    directory = tmp_path / "completion-01"
    result = completion.prepare("b" * 40, directory, ledger_path, prior=prior)
    canary = json.loads(Path(result["skip-ledger-canary"]).read_text())
    science = json.loads(Path(result["production-completion"]).read_text())
    assert canary["inputs"] == science["inputs"]
    assert canary["source_paths"] == science["source_paths"]
    assert science["density"]["missing_registered_episodes"] == 84
    assert science["density"]["attempted_episodes"] == 96
    assert science["density"]["total_episodes"] == 492
    assert science["density"]["science_commit"] == completion.SCIENCE_COMMIT
    assert science["requirements"]["est_minutes"] == 35
    assert science["requirements"]["min_vcpu"] == 12
    assert science["timeout_minutes"] == 180
    assert science["budget"]["unit_cap"] == 5
    assert science["budget"]["reserve_units"] == 1
    assert canary["budget"]["unit_cap"] == 1
    assert canary["budget"]["reserve_units"] == 0.2
    assert canary["timeout_minutes"] == 30
    assert science["density"]["prior_spend_units"] == 4.5
    for value in (science, canary):
        assert value["scientific_admission"] is False
        assert value["env"]["BL_DENSITY_SCIENTIFIC_ADMISSION"] == "0"
        assert "BL_DENSITY_CANARY_PROOF_JSON" not in value["env"]
    with pytest.raises(FileExistsError):
        completion.prepare("b" * 40, directory, ledger_path, prior=prior)


def test_prepare_refuses_metadata_not_equal_verified_original(monkeypatch, tmp_path):
    prior, ledger_path = _prepare_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(ledger, "build_ledger", lambda *_a, **_kw: {"entries": []})
    with pytest.raises(ValueError, match="equal all verified"):
        completion.prepare("b" * 40, tmp_path / "prepared", ledger_path, prior=prior)
    assert not (tmp_path / "prepared").exists()


def test_attempt_limits_and_mechanical_retry_guard(monkeypatch, tmp_path):
    prior, ledger_path = _prepare_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="at most"):
        completion.prepare("b" * 40, tmp_path / "prepared", ledger_path, prior=prior, attempt=3)
    with pytest.raises(ValueError, match="mechanical attempt evidence"):
        completion.prepare("b" * 40, tmp_path / "prepared", ledger_path, prior=prior, attempt=2)


def test_admit_refuses_before_any_qualification_when_transport_unmerged(monkeypatch, tmp_path):
    prior, ledger_path = _prepare_fixture(monkeypatch, tmp_path)
    result = completion.prepare("b" * 40, tmp_path / "prepared", ledger_path, prior=prior)
    value = json.loads(Path(result["skip-ledger-canary"]).read_text())
    monkeypatch.setattr(
        completion,
        "_transport_on_main",
        lambda *_a: (_ for _ in ()).throw(ValueError("telemetry must merge")),
    )
    monkeypatch.setattr(
        manifests,
        "source_bound_transport_receipt",
        lambda *_a: pytest.fail("must refuse before qualification"),
    )
    with pytest.raises(ValueError, match="telemetry must merge"):
        completion.admit(value, qualification=tmp_path / "acceptance")
    assert value["scientific_admission"] is False


def test_completion_requires_skip_canary_proof(monkeypatch, tmp_path):
    prior, ledger_path = _prepare_fixture(monkeypatch, tmp_path)
    result = completion.prepare("b" * 40, tmp_path / "prepared", ledger_path, prior=prior)
    value = json.loads(Path(result["production-completion"]).read_text())
    monkeypatch.setattr(completion, "_transport_on_main", lambda *_a: "tip")
    monkeypatch.setattr(
        manifests, "source_bound_transport_receipt", lambda *_a: {"status": "PASS_FINAL_ONLY"}
    )
    with pytest.raises(ValueError, match="skip-ledger canary proof"):
        completion.admit(value, qualification=tmp_path / "acceptance")
