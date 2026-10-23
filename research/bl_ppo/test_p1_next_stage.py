import json
import subprocess
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_next_stage import QUALIFIED_CLI_TREE, bind_source, prepare


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def test_prepare_writes_held_records_with_qualified_transport(tmp_path: Path) -> None:
    root = tmp_path / "next"
    value = prepare(root=root, commit=_commit(), repo=Path.cwd())
    assert value["status"] == "PREPARED_HELD_NOT_DISPATCHED"
    assert value["qualified_colab_cli_tree"] == QUALIFIED_CLI_TREE
    replacement = json.loads((root / "y3-replacement-held.json").read_text())
    qualification = json.loads((root / "qual-1m-held.json").read_text())
    assert replacement["status"] == qualification["status"] == "HELD_NOT_ADMITTED"
    assert replacement["dispatch"].startswith("NONE")
    assert "Y3_PASS_BOUNDARY_ONLY" in replacement["forbidden_substitutes"]
    assert qualification["fixed_boundary"] == {
        "requested_actor_transitions": 1_000_000,
        "whole_world_boundary": 1_000_064,
        "base_unit_cap": 4.0,
        "effective_cap_and_timeout": "JUDGE_REQUIRED_NOT_FROZEN",
    }
    assert replacement["pricing"]["status"] == "JUDGE_REQUIRED_NOT_FUNDED"
    assert replacement["pricing"]["exact_pin_canary_plus_source_plus_replacement_units"] > 25.50
    for phase in ("y3_replacement", "qual_1m"):
        manifest = root / value["held_phase_packs"][phase]["manifest"]
        receipt = json.loads((root / value["held_phase_packs"][phase]["receipt"]).read_text())
        held = json.loads((root / value["held_phase_packs"][phase]["held_manifest"]).read_text())
        assert manifest.exists()
        assert receipt["dispatch_ready"] is False
        assert receipt["colab_calls"] == 0
        assert held["scientific_admission"] is False
        assert held["transport"]["dispatch_ready"] is False
        assert held["budget"]["rate_units_per_hour"] == 1.54
        assert held["budget"]["rate_tolerance_units_per_hour"] == 0.0


def test_prepare_refuses_a_commit_without_the_qualified_tree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="qualified Colab CLI tree"):
        prepare(root=tmp_path / "next", commit="35104156", repo=Path.cwd())


def test_bind_source_requires_actor_operational_proof(tmp_path: Path) -> None:
    root = tmp_path / "next"
    prepare(root=root, commit=_commit(), repo=Path.cwd())
    source = tmp_path / "source.json"
    export_json(
        source,
        {
            "schema": "bl-p1-y3-source/v1",
            "status": "COMPLETE_SOURCE_CONTINUATION",
            "operational_snapshots": [{"phase": "refit", "path": "refit.pt"}],
            "operational_trace": {"path": "operational-action-trace.json", "sha256": "a" * 64},
        },
    )
    with pytest.raises(ValueError, match="actor-phase operational snapshot"):
        bind_source(root=root, source_receipt=source)
    source_value = json.loads(source.read_text())
    source_value["operational_snapshots"] = [{"phase": "actor", "path": "actor.pt"}]
    for suffix in ("", ".sha256", ".complete"):
        Path(str(source) + suffix).unlink()
    export_json(source, source_value)
    result = bind_source(root=root, source_receipt=source)
    assert result["status"] == "SOURCE_REFERENCE_BOUND_UNVERIFIED"
    assert result["source_receipt"]["sha256"] == digest(source)
