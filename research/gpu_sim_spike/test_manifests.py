"""Allocation-free E1 manifest budget and preflight identity checks."""

from __future__ import annotations

import hashlib
import json

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
    assert result["env"]["GPU_SIM_SPIKE_MODE"] == "all"
    assert result["env"]["GPU_SIM_SPIKE_REALISTIC_WORLDS"] == "256,1024,4096"
    assert result["env"]["GPU_SIM_SPIKE_REALISTIC_REPEATS"] == "3"
    assert result["env"]["COLAB_CLI_REQUIREMENTS"].endswith("gpu_sim_spike/requirements.txt")
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


def test_handoff_exports_have_complete_hash_sidecars_and_refuse_overwrite(tmp_path) -> None:
    path = tmp_path / "DISPATCH.txt"
    manifests.write_once(path, "prepared only\n")
    assert path.with_name(path.name + ".complete").read_text() == "complete\n"
    assert path.with_name(path.name + ".sha256").read_text().split()[0] == manifests.sha256_file(
        path
    )
    with pytest.raises(manifests.ManifestRefused, match="overwrite"):
        manifests.write_once(path, "changed\n")


def test_old_dense_only_preflight_cannot_authorize_e1b_manifest(tmp_path) -> None:
    path = tmp_path / "benchmark.json"
    manifests.write_once(
        path,
        json.dumps(
            {
                "status": "PREFLIGHT",
                "device": "cpu",
                "elapsed_seconds": 1.0,
                "parity": {"decisions": 3},
            }
        ),
    )
    with pytest.raises(manifests.ManifestRefused, match="both dense and REALISTIC"):
        manifests.preflight_binding(path)
