"""Amendment-04 shared-input family-pack checks."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.manifests import input_row
from research.bl_ppo.p1_manifests import bundle_inputs


def _control(audit: Path) -> dict:
    return {
        "schema": "bl-p1-control/v1",
        "phase": "y3-source",
        "seed": 20261021,
        "e8_critic_convergence_amendment_03": False,
        "e8_critic_convergence_amendment_04": True,
        "e8_critic_convergence_amendment_04_sha256": "a" * 64,
        "natural_collection_world_id_offset": 3_001_000,
        "natural_collection_burned_roots": [],
        "collection_allocation_audit_sha256": digest(audit),
        "collection_allocation_audit_path": "collection-allocation-audit.json",
        "resume": {
            "unit_kind": "operational-snapshot",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
            "namespace": "source_ns",
            "allocation": "source_allocation",
        },
    }


def _legacy_audit(path: Path, *, offset: int, seed: int = 20261021) -> None:
    export_json(
        path,
        {
            "schema": "bl-p1-collection-allocation-audit/v1",
            "seed": seed,
            "natural_collection_world_id_offset": offset,
            "collision_count": 0,
            "burned_roots": [],
        },
    )


def test_preparation_ships_one_archive_with_two_disjoint_controls(tmp_path, monkeypatch) -> None:
    from research.bl_ppo import e8_04_preparation as prep
    from research.bl_ppo.p1_inputs import unpack_bundle

    source_audit = tmp_path / "collection-allocation-audit.json"
    canary_audit = tmp_path / "canary-collection-allocation-audit.json"
    _legacy_audit(source_audit, offset=prep.SOURCE_OFFSET)
    _legacy_audit(canary_audit, offset=prep.CANARY_OFFSET, seed=2026102203)
    control = tmp_path / "p1-control.json"
    ledger = tmp_path / "verified-ledger.json"
    export_json(control, _control(source_audit))
    export_json(
        ledger,
        {
            "schema": "bl-p1-verified-ledger/v1",
            "namespace": "source_ns",
            "allocation": "source_allocation",
            "units": [],
        },
    )
    source_inputs = tmp_path / "source-inputs"
    source_inputs.mkdir()
    bundle_inputs(
        source_inputs,
        [
            input_row(control, "p1-control.json"),
            input_row(Path(str(control) + ".sha256"), "p1-control.json.sha256"),
            input_row(Path(str(control) + ".complete"), "p1-control.json.complete"),
            input_row(ledger, "verified-ledger.json"),
            input_row(Path(str(ledger) + ".sha256"), "verified-ledger.json.sha256"),
            input_row(Path(str(ledger) + ".complete"), "verified-ledger.json.complete"),
        ],
    )
    source_manifest = tmp_path / "source-manifest.json"
    export_json(
        source_manifest,
        {
            "commit": "f" * 40,
            "job_id": "source",
            "script": "research/bl_ppo/p1_colab.sh",
            "env": {"BL_P1_MODE": "y3-source", "BL_PPO_COMMIT": "f" * 40},
            "inputs": [input_row(source_inputs / "p1-inputs.tar.gz", "p1-inputs.tar.gz")],
            "output_dir": str(tmp_path / "old-output"),
            "required_exports": ["receipt.json"],
            "budget": {"unit_cap": 9.62, "reserve_units": 1.92, "rate_units_per_hour": 1.54},
            "requirements": {
                "est_minutes": 75,
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": 12,
            },
            "timeout_minutes": 300,
            "resume": _control(source_audit)["resume"],
            "source_paths": [],
            "bl_p1": {"control": _control(source_audit)},
        },
    )
    monkeypatch.setattr("research.colab_cli.supervisor.validate", lambda value, _root: value)
    qualify_module = importlib.import_module("research.bl_upload.qualify")
    monkeypatch.setattr(
        qualify_module, "qualify", lambda value, repo: {"manifest": value["job_id"]}
    )
    manifests_module = importlib.import_module("research.bl_ppo.manifests")
    monkeypatch.setattr(manifests_module, "_bounded_source_paths", lambda value: [])
    result = prep.prepare(
        source_manifest=source_manifest,
        root=tmp_path / "prepared",
        commit="f" * 40,
        source_audit=source_audit,
        canary_audit=canary_audit,
    )
    prepared = tmp_path / "prepared"
    source = json.loads((prepared / "source-template.json").read_text())
    canary = json.loads((prepared / "canary-manifest.json").read_text())
    assert source["inputs"] == canary["inputs"]
    assert source["scientific_admission"] is False
    assert canary["scientific_admission"] is False
    assert result["offsets"] == {"canary": 3_000_000, "source": 3_001_000}
    unpacked = tmp_path / "unpacked"
    unpack_bundle(prepared / "p1-inputs.tar.gz", unpacked)
    source_control = json.loads((unpacked / "p1-control.json").read_text())
    canary_control = json.loads((unpacked / "canary-control.json").read_text())
    assert source_control["natural_collection_world_id_offset"] == 3_001_000
    assert canary_control["natural_collection_world_id_offset"] == 3_000_000
    assert canary_control["natural_collection_seed"] == 2026102203
    assert source_control["resume"]["namespace"] != canary_control["resume"]["namespace"]
    assert (unpacked / "verified-ledger.json").is_file()
    assert (unpacked / "canary-verified-ledger.json").is_file()


def test_entrypoint_selects_the_canary_control_only_under_family_canary() -> None:
    script = Path(__file__).with_name("p1_colab.sh").read_text()
    assert 'control="${COLAB_INPUT_DIR}/p1-control.json"' in script
    assert 'if [[ "${SNAKE_COLAB_CANARY:-}" == "1" ]]; then' in script
    assert 'control="${COLAB_INPUT_DIR}/canary-control.json"' in script
    assert script.index('control="${COLAB_INPUT_DIR}/canary-control.json"') < script.index(
        "python - \"${control}\" <<'PY'"
    )


def test_v3_audit_requires_fixed_enumeration_and_cross_lane_proof(tmp_path) -> None:
    from research.bl_ppo import e8_04_preparation as prep

    audit = tmp_path / "collection-allocation-audit.json"
    value = {
        "schema": "bl-p1-collection-allocation-audit/v3",
        "natural_collection_world_id_offset": prep.SOURCE_OFFSET,
        "collision_count": 0,
        "burned_roots": [],
        "enumeration": {
            "episode_start": 0,
            "episode_end": 127,
            "attempt_start": 0,
            "attempt_end": 1,
        },
        "cross_lane": {
            "raw_root_intersection_count": 0,
            "allocated_root_intersection_count": 0,
        },
        "offset_advance_rule": {"advance_both_offsets_by": 10_000, "lane_separation": 1_000},
    }
    export_json(audit, value)
    control = _control(audit)
    prep._require_audit(audit, offset=prep.SOURCE_OFFSET, control=control)
    invalid = tmp_path / "invalid-audit.json"
    value["cross_lane"]["allocated_root_intersection_count"] = 1
    export_json(invalid, value)
    with __import__("pytest").raises(ValueError, match="collision proof"):
        prep._require_audit(invalid, offset=prep.SOURCE_OFFSET, control=_control(invalid))
