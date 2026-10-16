"""Round-64 audit cannot approve reuse or missing label provenance."""

import json

import pytest

from research.t4_curve_20261015 import banks


def test_audit_rejects_observed_round64_reuse(tmp_path, monkeypatch):
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    namespace = artifacts / "t3-rollout-labels-20261015/bank/namespace.json"
    namespace.parent.mkdir(parents=True)
    namespace.write_text('{"schema": "t3-label-namespace/v1", "seeds": [1]}')
    probe = artifacts / "old-probes"
    probe.mkdir()
    seed = banks.ni_spec.distill_seeds(32, round_index=64)[0]
    (probe / "row.json").write_text('{"world_seed": %d}' % seed)
    monkeypatch.setattr(
        banks,
        "t2_audit",
        lambda _repo: {"pass": True, "registered_rounds": {}, "historical_distillation": {}},
    )
    monkeypatch.setattr(banks, "_manifest_binding", lambda *_: {"ok": True})
    result = banks.audit(repo, artifacts, artifacts / "out")
    assert result["checks"]["observed_collisions"] == [seed]
    assert not result["pass"]
    (probe / "row.json").unlink()
    assert banks.audit(repo, artifacts, artifacts / "out")["pass"]
    monkeypatch.setattr(banks, "_manifest_binding", lambda *_: {"ok": False})
    assert not banks.audit(repo, artifacts, artifacts / "out")["pass"]
    for invalid in (
        {"seeds": [1]},
        {"schema": "t3-label-namespace/v1", "seeds": []},
        {"schema": "t3-label-namespace/v1", "seeds": [True]},
    ):
        namespace.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="Invalid or empty"):
            banks.audit(repo, artifacts, artifacts / "out")


def test_reserved_screen_rounds_are_disjoint():
    curve = set(banks.ni_spec.distill_seeds(32, round_index=64))
    assert len(curve) == 32
    for round_index in (62, 63):
        assert not curve & set(banks.ni_spec.distill_seeds(96, round_index=round_index))
