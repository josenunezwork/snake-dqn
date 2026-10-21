"""Offline envelope and dispatch-refusal checks; no CLI provider operations."""

import json
from pathlib import Path

import pytest

from research.bl_served_search.exports import digest, export_json
from research.bl_served_search.manifests import (
    bind_l2_plan,
    portable_allocation,
    source_paths,
)
from research.bl_served_search.readiness import check


def test_readiness_rejects_prior_plan_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        check({"schema": "bl-served-search-plan/v2", "dispatchable": True}, tmp_path)


def test_portable_bank_changes_only_path_presentation(tmp_path: Path) -> None:
    allocation, output = tmp_path / "old", tmp_path / "portable"
    original = {
        "rows": [{"world_seed": 22, "plan": {"specs": [["checkpoint", "/Mac/a.pth"]]}}],
        "excluded_preflight_rows": [],
    }
    export_json(allocation / "bank.json", original)
    export_json(
        allocation / "audit.json",
        {"status": "CLEAR", "bank_sha256": digest(allocation / "bank.json")},
    )
    bank_path, audit_path = portable_allocation(allocation, output)
    value = json.loads(bank_path.read_text())
    assert value["rows"][0] == {"world_seed": 22, "plan": {"specs": [["checkpoint", "a.pth"]]}}
    audit = json.loads(audit_path.read_text())
    assert audit["bank_sha256"] == digest(bank_path)
    assert audit["allocation_bank_sha256"] == digest(allocation / "bank.json")


def test_static_closure_contains_t3_runtime_and_exports() -> None:
    paths = source_paths()
    assert "research/bl_upload/qualification.py" in paths
    assert "research/t3_rollout_labels_20261015/bank_engine.py" in paths
    assert "research/t3_rollout_labels_20261015/trigger.py" in paths
    assert "research/bl_ppo/exports.py" in paths
    assert "configs" in paths
    assert "research/colab_cli" in paths


def test_runtime_launchers_materialize_the_public_checkpoint_catalog() -> None:
    package = Path(__file__).parent
    for name in ("run_l2.sh", "run_ruler.sh"):
        script = (package / name).read_text(encoding="utf-8")
        assert "research.bl_upload.inputs --input-dir" in script
        assert "BL_PUBLIC_INPUTS:-0" in script
        assert "public_root/B8_s3_final.pth" in script
        assert "public_root/checkpoints" in script


def test_l2_plan_is_written_once_after_public_input_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_served_search import manifests

    calls: list[Path] = []
    real_export = manifests.export_json

    def counted_export(path: Path, value: dict) -> None:
        calls.append(path)
        real_export(path, value)

    monkeypatch.setattr(manifests, "export_json", counted_export)
    catalog = tmp_path / "public-inputs.json"
    catalog.write_text("{}\n", encoding="utf-8")
    manifest = {
        "inputs": [
            {"dest": "bank.json", "sha256": "bank"},
            {"dest": "public-inputs.json", "sha256": "catalog"},
        ]
    }
    plan = {"input_hashes": {"stale-checkpoint.pth": "stale"}}
    plan_path = tmp_path / "l2-plan.json"

    bind_l2_plan(manifest, plan, plan_path)

    assert calls == [plan_path]
    assert plan["input_hashes"] == {"bank.json": "bank", "public-inputs.json": "catalog"}
    assert manifest["inputs"][-1]["dest"] == "l2-plan.json"
    assert manifest["inputs"][-1]["sha256"] == digest(plan_path)
