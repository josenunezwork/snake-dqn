"""Allocation-free E1 manifest budget and preflight identity checks."""

from __future__ import annotations

import hashlib

import pytest

from research.gpu_sim_spike import make_manifests as manifests


@pytest.mark.parametrize("kind,envelope,depth", [("l4", 15, "300"), ("probe", 5, "8")])
def test_e1_l4_manifest_budgets_and_local_supervisor_validation(
    tmp_path, monkeypatch, kind, envelope, depth
) -> None:
    monkeypatch.setattr(manifests, "checkpoint_inputs", lambda: [])
    rule_sha = hashlib.sha256(
        (manifests.ROOT / manifests.PACKAGE / "KILL_RULE.md").read_bytes()
    ).hexdigest()
    result = manifests.make_manifest(
        kind=kind,
        commit=manifests.git_head(),
        output_dir=tmp_path / kind,
        preflight={"kill_rule_sha256": rule_sha, "source_sha256": {}},
    )
    manifests.validate_manifest(result)
    budget = result["budget"]
    assert budget["unit_cap"] + budget["reserve_units"] == envelope
    assert result["timeout_minutes"] * budget["rate_units_per_hour"] / 60 <= budget["unit_cap"]
    assert result["timeout_minutes"] >= 2 * result["requirements"]["est_minutes"]
    assert result["requirements"]["allowed_gpus"] == ["L4"]
    assert result["env"]["GPU_SIM_SPIKE_HORIZON"] == depth
    assert result["env"]["GPU_SIM_SPIKE_CPU_WORKERS"] == "12"
    assert result["transport"]["dispatch_ready"] is False


def test_preflight_rule_drift_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(manifests, "checkpoint_inputs", lambda: [])
    with pytest.raises(manifests.ManifestRefused, match="kill rule differs"):
        manifests.make_manifest(
            kind="l4",
            commit=manifests.git_head(),
            output_dir=tmp_path / "run",
            preflight={"kill_rule_sha256": "0" * 64, "source_sha256": {}},
        )
