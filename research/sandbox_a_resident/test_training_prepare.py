"""TRAIN-01 admission cannot broaden old sandbox model input permissions."""

from __future__ import annotations

import copy

import pytest

from research.sandbox_a_resident.training_prepare import STUDY, audit_seeds, seed_allocations
from tools.sandbox import registry
from tools.sandbox.manifest import ManifestError, validate_manifest


def test_seed_universe_covers_one_frame_deaths_and_stays_disjoint() -> None:
    rows = seed_allocations({"worlds": 1024, "outer_iterations": 384, "rollout_horizon": 256})
    assert rows[0]["count"] == 100_664_320
    assert all(a["offset"] + a["count"] < b["offset"] for a, b in zip(rows, rows[1:]))


def test_independent_replication_seed_offset_is_disjoint_and_preserves_proxy_eval_banks() -> None:
    base = seed_allocations(
        {"worlds": 1024, "outer_iterations": 384, "rollout_horizon": 256, "train_seed_offset": 0}
    )
    rep = seed_allocations(
        {
            "worlds": 1024,
            "outer_iterations": 384,
            "rollout_horizon": 256,
            "train_seed_offset": 1_000_000_000,
        }
    )
    base_by_phase = {row["phase"]: row for row in base}
    rep_by_phase = {row["phase"]: row for row in rep}
    assert (
        rep_by_phase["monitor"]
        == base_by_phase["monitor"]
        == {
            "phase": "monitor",
            "offset": 400_000_000,
            "count": 288,
        }
    )
    assert (
        rep_by_phase["nomination"]
        == base_by_phase["nomination"]
        == {
            "phase": "nomination",
            "offset": 500_000_000,
            "count": 768,
        }
    )
    for phase in ("train", "calibration", "critic", "engineering"):
        a, b = base_by_phase[phase], rep_by_phase[phase]
        assert a["count"] == b["count"]
        assert a["offset"] + a["count"] <= b["offset"]
    ordered = sorted(rep, key=lambda row: row["offset"])
    assert all(a["offset"] + a["count"] <= b["offset"] for a, b in zip(ordered, ordered[1:]))
    for bad in (-1, True, 200_000_000, 400_000_000, 500_000_000):
        with pytest.raises(ValueError):
            seed_allocations(
                {
                    "worlds": 1024,
                    "outer_iterations": 384,
                    "rollout_horizon": 256,
                    "train_seed_offset": bad,
                }
            )


def test_exhaustive_seed_audit_rejects_effective_exclusion(monkeypatch, tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    value = {"excluded_seeds": []}
    monkeypatch.setattr(
        "research.sandbox_a_resident.training_prepare.load_registry", lambda p: value
    )
    monkeypatch.setattr(
        "research.sandbox_a_resident.training_prepare.verify_sources", lambda r: None
    )
    rows = [{"phase": "test", "offset": 40, "count": 8}]
    assert audit_seeds(rows, registry_path)["count"] == 8
    value["excluded_seeds"] = [registry.world_seed(STUDY, 44) & 0x7FFF_FFFF_FFFF_FFFF]
    with pytest.raises(ValueError, match="excluded seed"):
        audit_seeds(rows, registry_path)
    with pytest.raises(ValueError, match="overlapping"):
        audit_seeds(rows * 2, registry_path)


def test_canonical_opponents_never_use_old_b8_kind(monkeypatch, tmp_path) -> None:
    path = tmp_path / "fixed.pth"
    path.write_bytes(b"fixed opponent fixture")
    sha = next(iter(registry.TRAINING_OPPONENTS))
    monkeypatch.setattr(registry, "digest", lambda path: sha)
    registry.validate_input({"kind": "frozen-opponent", "path": str(path), "sha256": sha})
    with pytest.raises(ValueError, match="only frozen released B8"):
        registry.validate_input({"kind": "frozen-checkpoint", "path": str(path), "sha256": sha})


def test_opponent_kind_exclusive_to_training_study() -> None:
    import json
    from pathlib import Path

    original = json.loads((Path(__file__).parent / "packet-a3-04/sandbox.json").read_text())
    payload = copy.deepcopy(original)
    payload["inputs"].append(
        {"kind": "frozen-opponent", "path": "/tmp/fixed.pth", "sha256": "a" * 64}
    )
    with pytest.raises(ManifestError, match="exclusive"):
        validate_manifest(payload)
