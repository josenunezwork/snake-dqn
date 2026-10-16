"""T4c round-66/67 seed-audit contracts."""

import json

from research.t4c_20261016 import seed_audit as banks


def test_seed_audit_rejects_invalid_or_empty_t3_namespace(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    namespace = artifacts / "t3-rollout-labels-20261015/bank"
    namespace.mkdir(parents=True)
    (namespace / "namespace.json").write_text(json.dumps({"schema": "wrong", "seeds": []}))
    (namespace / "manifest.json").write_text(
        json.dumps({"schema": "t3-label-bank/v1", "files": {}})
    )
    monkeypatch.setattr(
        banks.prior_banks,
        "_registered_t2",
        lambda _repo: {
            "pass": True,
            "schema": "fixture",
            "registered_rounds": {},
            "historical_distillation": {},
        },
    )
    monkeypatch.setattr(banks.prior_banks, "_manifest_binding", lambda *_args: {"ok": True})
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    monkeypatch.setattr(ni_spec, "ni_seeds", lambda: [])
    monkeypatch.setattr(ni_spec_m2b, "ni_seeds", lambda: [])
    monkeypatch.setattr(
        ni_spec,
        "distill_seeds",
        lambda _n, round_index: list(range(round_index * 100, round_index * 100 + 96)),
    )
    result = banks.seed_bank_audit(
        tmp_path, artifacts, artifacts / "redesign-t4c-probes-20261016/audit.json"
    )
    assert not result["pass"]
    assert result["t3_rollout_training_namespace"]["schema"] == "wrong"


def test_seed_audit_excludes_only_its_own_receipt(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    namespace = artifacts / "t3-rollout-labels-20261015/bank"
    namespace.mkdir(parents=True)
    (namespace / "namespace.json").write_text(
        json.dumps({"schema": "t3-label-namespace/v1", "seeds": [4]})
    )
    (namespace / "manifest.json").write_text(
        json.dumps({"schema": "t3-label-bank/v1", "files": {}})
    )
    out = artifacts / "redesign-t4c-probes-20261016/audit.json"
    out.parent.mkdir()
    out.write_text(json.dumps({"world_seeds": [6600]}))
    monkeypatch.setattr(
        banks.prior_banks,
        "_registered_t2",
        lambda _repo: {
            "pass": True,
            "schema": "fixture",
            "registered_rounds": {},
            "historical_distillation": {},
        },
    )
    monkeypatch.setattr(banks.prior_banks, "_manifest_binding", lambda *_args: {"ok": True})
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    monkeypatch.setattr(ni_spec, "ni_seeds", lambda: [])
    monkeypatch.setattr(ni_spec_m2b, "ni_seeds", lambda: [])
    monkeypatch.setattr(
        ni_spec,
        "distill_seeds",
        lambda _n, round_index: list(range(round_index * 100, round_index * 100 + 96)),
    )
    result = banks.seed_bank_audit(tmp_path, artifacts, out)
    assert result["pass"]
    assert result["excluded_output"] == str(out)
    assert result["inventories"][str(artifacts)]["file_count"] == 2
    # Historical evaluations are not guaranteed to use a probe/eval directory name.
    campaign = artifacts / "campaign"
    campaign.mkdir()
    (campaign / "rows.json").write_text(json.dumps({"world_seeds": [6600]}))
    result = banks.seed_bank_audit(tmp_path, artifacts, out)
    assert not result["pass"]
    assert result["checks"]["66"]["observed_json_or_label_collisions"] == [6600]
    (artifacts / "rows.json").write_text(json.dumps({"world_seeds": [6700]}))
    result = banks.seed_bank_audit(tmp_path, artifacts, out)
    assert result["checks"]["67"]["observed_json_or_label_collisions"] == [6700]


def test_own_probe_results_are_not_excluded(tmp_path, monkeypatch):
    """Only the audit receipt is excluded, never actual round-66 evaluations."""
    from research.redesign_m2_20261008 import ni_spec

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    archive = tmp_path / "research/t4c_20261016/results"
    archive.mkdir(parents=True)
    seed = ni_spec.distill_seeds(96, round_index=66)[0]
    (archive / "probes.jsonl").write_text(json.dumps({"world_seeds": [seed]}) + "\n")
    monkeypatch.setattr(
        banks.prior_banks,
        "_registered_t2",
        lambda _repo: {
            "pass": True,
            "schema": "fixture",
            "registered_rounds": {},
            "historical_distillation": {},
        },
    )
    result = banks.seed_bank_audit(tmp_path, artifacts)
    assert seed in result["checks"]["66"]["observed_json_or_label_collisions"]
    assert not result["pass"]


def test_cannot_exclude_an_entire_output_directory(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="never a directory"):
        banks.seed_bank_audit(tmp_path, tmp_path, tmp_path)


def test_streamed_inventory_matches_seed_fields_and_hashes(tmp_path):
    values = {
        "nested": [{"world_seed": 42}, {"world_seeds": [51, 52]}],
        "mass": {"80": 1.2},
        "numeric_array": [1.0, 2.0, 3.0],
    }
    (tmp_path / "large.json").write_text(json.dumps(values))
    (tmp_path / "records.jsonl").write_text(json.dumps({"seed": 61}) + "\n")
    previous = banks.prior_banks._json_inventory(tmp_path)
    current = banks.inventory_json(tmp_path)
    for key in ("file_count", "valid_json_count", "seed_count", "seeds", "digest"):
        assert current[key] == previous[key]
    assert current["invalid_json"] == []
    (tmp_path / "records.jsonl").write_text('{"world_seed": 71}\n{"incomplete":')
    partial = banks.inventory_json(tmp_path)
    assert 71 in partial["seeds"]
    assert partial["invalid_json"] == [str(tmp_path / "records.jsonl")]
