"""T4a round-65/66 seed-audit contracts."""

import json

from research.t4a_screen_20261016 import banks


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
    result = banks.seed_bank_audit(tmp_path, artifacts, artifacts / "redesign-t4a-probes-20261016")
    assert not result["pass"]
    assert result["t3_rollout_training_namespace"]["schema"] == "wrong"


def test_seed_audit_excludes_its_own_probe_output(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    namespace = artifacts / "t3-rollout-labels-20261015/bank"
    namespace.mkdir(parents=True)
    (namespace / "namespace.json").write_text(
        json.dumps({"schema": "t3-label-namespace/v1", "seeds": [4]})
    )
    (namespace / "manifest.json").write_text(
        json.dumps({"schema": "t3-label-bank/v1", "files": {}})
    )
    out = artifacts / "redesign-t4a-probes-20261016"
    out.mkdir()
    (out / "self.json").write_text(json.dumps({"world_seeds": [6500]}))
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
    assert result["inventories"][str(out)]["file_count"] == 0
