"""Offline native preparation gates; never allocate or run timing/gameplay."""

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from research.bl_served_search import native_manifests, native_provenance, native_ruler, ruler
from research.bl_served_search.exports import export_json
from research.bl_served_search.test_native_ruler import paired_receipt
from research.colab_cli.supervisor import Refused

ROOT = Path(__file__).resolve().parents[2]


def _functions(raw: str) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(raw).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_compatibility_does_not_change_measured_computation() -> None:
    old_ruler = subprocess.check_output(
        ["git", "show", "8be8c195^:research/bl_served_search/ruler.py"], cwd=ROOT, text=True
    )
    old_native = subprocess.check_output(
        ["git", "show", "9ef24704:research/bl_served_search/native_ruler.py"], cwd=ROOT, text=True
    )
    assert hashlib.sha256(old_ruler.encode()).hexdigest() == native_provenance.OLD_RULER
    assert hashlib.sha256(old_native.encode()).hexdigest() == native_provenance.OLD_NATIVE
    previous, current = _functions(old_ruler), _functions(Path(ruler.__file__).read_text())
    for name in previous.keys() - {"_main"}:
        assert previous[name] == current[name], name
    previous = _functions(old_native)
    current = _functions(Path(native_ruler.__file__).read_text())
    for name in previous.keys() - {"source_identity", "validate_paired_receipt"}:
        assert previous[name] == current[name], name


def test_legacy_receipt_admitted_only_for_exact_historical_sources() -> None:
    receipt = paired_receipt()
    receipt["source_identity"][native_provenance.RULER_PATH] = native_provenance.OLD_RULER
    receipt["source_identity"][native_provenance.NATIVE_PATH] = native_provenance.OLD_NATIVE
    receipt["source_identity"].pop(native_provenance.SELF_PATH)
    for cell in receipt["cells"].values():
        cell["source_identity"][native_provenance.RULER_PATH] = native_provenance.OLD_RULER
    native_ruler.validate_paired_receipt(receipt)
    for field in receipt["source_identity"]:
        for invalid in ("0" * 64, None):
            broken = copy.deepcopy(receipt)
            broken["source_identity"][field] = invalid
            with pytest.raises(ValueError, match="INVALID"):
                native_ruler.validate_paired_receipt(broken)
    broken = copy.deepcopy(receipt)
    broken["source_identity"].pop("src/simd_env/grid_sim.py")
    with pytest.raises(ValueError, match="INVALID"):
        native_ruler.validate_paired_receipt(broken)


def pending(tmp_path: Path) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "job_id": "native-local-test",
        "commit": commit,
        "script": "research/bl_served_search/run_native_ruler.sh",
        "requirements": {"gpu": "L4", "min_vcpu": 12, "est_minutes": 9},
        "timeout_minutes": 18,
        "budget": {"unit_cap": 1, "rate_units_per_hour": 2, "other_assignments": {}},
        "output_dir": str(tmp_path / "future-runtime"),
        "required_exports": ["ruler-receipt.json"],
        "inputs": [],
        "env": {},
    }


def test_main_structure_validated_without_forging_family_proof(tmp_path: Path) -> None:
    manifest = pending(tmp_path)
    assert native_manifests.validate_pending(manifest)["status"] == "PASS_PRECANARY_STRUCTURE"
    assert "canary" not in manifest
    manifest["budget"]["other_assignments"] = []
    with pytest.raises(ValueError, match="map"):
        native_manifests.validate_pending(manifest)


def test_pending_validation_refuses_other_errors(tmp_path: Path) -> None:
    manifest = pending(tmp_path)
    manifest["timeout_minutes"] = 1
    with pytest.raises(Refused, match=">=3"):
        native_manifests.validate_pending(manifest)


def test_bind_rejects_fabricated_or_unreleased_canary(tmp_path: Path) -> None:
    template = tmp_path / "main.json"
    export_json(template, pending(tmp_path))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "PASS", "stop_verified": False}))
    with pytest.raises(Refused):
        native_manifests.bind(template, receipt, tmp_path / "admitted.json")
    assert not (tmp_path / "admitted.json").exists()


def test_publish_requires_actual_clean_broker_release(tmp_path: Path) -> None:
    saved = tmp_path / "manifest.json"
    saved.write_text("{}")
    (tmp_path / "receipt.json").write_text(json.dumps({"status": "PASS", "stop_verified": False}))
    target = tmp_path / "published" / "ruler-receipt.json"
    with pytest.raises(ValueError, match="verified release"):
        native_manifests.publish(tmp_path, target, tmp_path / "unused-mac.json")
    assert not target.exists()
