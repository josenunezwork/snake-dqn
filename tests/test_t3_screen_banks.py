"""Contracts for the T3 pre-probe seed-bank audit."""

from __future__ import annotations

import json
from pathlib import Path

from research.t3_screen_20261015 import banks


def test_seed_fields_ignore_unrelated_integers() -> None:
    assert banks._seed_values({"count": 62, "seed": 7, "nested": {"world_seeds": [8, "9"]}}) == {
        7,
        8,
    }


def test_output_exclusion_preserves_pre_probe_audit(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    probe = artifacts / "old-probe"
    probe.mkdir(parents=True)
    (probe / "old.json").write_text(json.dumps({"round_index": 60, "world_seeds": [11]}))
    labels = artifacts / "t3-rollout-labels-20261015/bank"
    labels.mkdir(parents=True)
    (labels / "namespace.json").write_text(
        json.dumps({"schema": "t3-label-namespace/v1", "seeds": [12]})
    )
    out = artifacts / "redesign-t3-probes-20261015"
    out.mkdir()
    (out / "self.json").write_text(json.dumps({"round_index": 62, "world_seeds": [999]}))
    monkeypatch.setattr(
        banks,
        "_registered_t2",
        lambda _repo: {
            "pass": True,
            "schema": "fixture",
            "registered_ni_count": 0,
            "registered_rounds": {},
            "historical_distillation": {},
        },
    )
    from research.redesign_m2_20261008 import ni_spec

    original = ni_spec.distill_seeds
    monkeypatch.setattr(
        ni_spec,
        "distill_seeds",
        lambda _n, round_index: (
            [999] + list(range(6200, 6295)) if round_index == 62 else list(range(6300, 6396))
        ),
    )
    try:
        result = banks.seed_bank_audit(tmp_path, artifacts, out)
    finally:
        monkeypatch.setattr(ni_spec, "distill_seeds", original)
    assert result["pass"]
    assert result["observed_probe_artifacts"][str(out)]["file_count"] == 0
