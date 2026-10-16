"""Contracts for the T4e round-68/69 freshness receipt."""

import json
import subprocess
from pathlib import Path

import pytest

from research.t4e_return_20261016 import seed_audit


def _fixture(monkeypatch):
    """Make every registered source deterministic and disjoint by default."""
    monkeypatch.setattr(
        seed_audit.prior_banks,
        "_registered_t2",
        lambda _repo: {
            "pass": True,
            "schema": "fixture",
            "registered_rounds": {"60": list(range(6000, 6096))},
            "historical_distillation": {"history": {"seeds": list(range(7000, 7096))}},
        },
    )
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    monkeypatch.setattr(ni_spec, "ni_seeds", lambda: [])
    monkeypatch.setattr(ni_spec_m2b, "ni_seeds", lambda: [])
    monkeypatch.setattr(
        ni_spec,
        "distill_seeds",
        lambda count, round_index: list(range(round_index * 1000, round_index * 1000 + count)),
    )
    monkeypatch.setattr(seed_audit.prior_banks, "_manifest_binding", lambda *_args: {"ok": True})


def _namespace(artifacts):
    bank = artifacts / "t3-rollout-labels-20261015/bank"
    bank.mkdir(parents=True)
    (bank / "namespace.json").write_text(
        json.dumps({"schema": "t3-label-namespace/v1", "seeds": [4]})
    )


def test_audit_reserves_both_t4e_rounds_against_history_and_observed_results(tmp_path, monkeypatch):
    _fixture(monkeypatch)
    (tmp_path / "research").mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _namespace(artifacts)
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert result["pass"]
    assert result["checks"]["68"]["seeds"] == list(range(68000, 68096))
    assert result["checks"]["69"]["seeds"] == list(range(69000, 69096))

    consumed = tmp_path / "research/actual-result"
    consumed.mkdir(parents=True)
    (consumed / "rows.json").write_text(json.dumps({"world_seeds": [68000]}))
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert not result["pass"]
    assert result["checks"]["68"]["observed_json_collisions"] == [68000]

    artifact_result = artifacts / "anything/actual.json"
    artifact_result.parent.mkdir()
    artifact_result.write_text(json.dumps({"world_seed": 69000}))
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert result["checks"]["69"]["observed_json_collisions"] == [69000]


def test_audit_excludes_only_exact_documentary_receipt(tmp_path, monkeypatch):
    _fixture(monkeypatch)
    (tmp_path / "research").mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _namespace(artifacts)
    receipt = artifacts / "t4e/seed_audit.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({"world_seeds": [68000]}))
    result = seed_audit.seed_bank_audit(tmp_path, artifacts, receipt)
    assert result["pass"]
    assert result["excluded_output"] == str(receipt)

    actual = receipt.parent / "actual_rows.json"
    actual.write_text(json.dumps({"world_seeds": [68000]}))
    result = seed_audit.seed_bank_audit(tmp_path, artifacts, receipt)
    assert not result["pass"]
    assert result["checks"]["68"]["observed_json_collisions"] == [68000]

    with pytest.raises(ValueError, match="exactly one JSON"):
        seed_audit.seed_bank_audit(tmp_path, artifacts, receipt.parent)


def test_exact_prior_receipt_exclusion_keeps_sibling_actual_result(tmp_path, monkeypatch):
    _fixture(monkeypatch)
    (tmp_path / "research").mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _namespace(artifacts)
    prior = artifacts / "prior-audit.json"
    prior.write_text(json.dumps({"world_seeds": [68000]}))
    actual = artifacts / "actual-result.json"
    actual.write_text(json.dumps({"world_seeds": [68000]}))
    result = seed_audit.seed_bank_audit(tmp_path, artifacts, documentary_exclusions=(prior,))
    assert not result["pass"]
    assert result["checks"]["68"]["observed_json_collisions"] == [68000]
    exclusion = result["exact_documentary_exclusions"][0]
    assert exclusion["path"] == str(prior)
    assert exclusion["bytes"] > 0 and len(exclusion["sha256"]) == 64


def test_malformed_receipt_with_complete_reserved_id_fails_closed(tmp_path, monkeypatch):
    _fixture(monkeypatch)
    (tmp_path / "research").mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _namespace(artifacts)
    malformed = artifacts / "truncated.json"
    malformed.write_bytes(b'{"round_index": 68, "world_seed":')
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert not result["pass"]
    assert result["text_scan"]["reserved_round_metadata"][str(artifacts)]

    malformed.write_bytes(b'{"world_seed": 68000')
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert not result["pass"]
    assert result["checks"]["68"]["observed_json_collisions"] == [68000]

    malformed.write_bytes(b'{"world_seed": 69000')
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert not result["pass"]
    assert result["checks"]["69"]["observed_json_collisions"] == [69000]


def test_registered_history_collision_fails_closed(tmp_path, monkeypatch):
    _fixture(monkeypatch)
    (tmp_path / "research").mkdir()
    original = seed_audit._registered

    def colliding(repo):
        receipt, registered = original(repo)
        return receipt, registered | {68000}

    monkeypatch.setattr(seed_audit, "_registered", colliding)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _namespace(artifacts)
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert not result["pass"]
    assert result["checks"]["68"]["registered_collisions"] == [68000]


def test_missing_t3_source_receipt_fails_closed(tmp_path, monkeypatch):
    _fixture(monkeypatch)
    (tmp_path / "research").mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = seed_audit.seed_bank_audit(tmp_path, artifacts)
    assert not result["pass"]
    assert not result["t3_rollout_training_namespace"]["exists"]


def test_partitioned_scanner_splits_timeouts_and_excludes_only_exact_receipt(tmp_path, monkeypatch):
    root = tmp_path / "scope"
    slow = root / "slow"
    slow.mkdir(parents=True)
    (slow / "used.json").write_text('{"world_seed": 68000}')
    receipt = root / "receipt.json"
    receipt.write_text('{"world_seed": 69000}')
    original = seed_audit.subprocess.run
    timed_out = set()

    def bounded(command, **kwargs):
        target = Path(command[-1])
        if target == slow and target not in timed_out:
            timed_out.add(target)
            raise subprocess.TimeoutExpired(command, 60)
        return original(command, **kwargs)

    monkeypatch.setattr(seed_audit.subprocess, "run", bounded)
    hits, progress = seed_audit.concrete_text_hits(root, ["68000", "69000"], (receipt,))
    assert hits == {"68000"}
    assert {item["status"] for item in progress} >= {"timeout-split", "excluded-exact-receipt"}
