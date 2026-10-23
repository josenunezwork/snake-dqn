"""Amendment-06 next stages cannot borrow missing or boundary-only evidence."""

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_repair_admission import (
    conservative_charge,
    verify_qualification_prerequisites,
    verify_replacement_prerequisites,
)


def session(path: Path, endpoint: str, exports: list) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "cleanup_error": None,
                "error": None,
                "terminal": {"status": "succeeded", "returncode": 0},
                "elapsed_seconds": 3600,
                "endpoint": endpoint,
                "exports": exports,
                "accounting": {"own_rate_units_per_hour": 1.54},
                "usage_before": {"balance": 100},
                "usage_after": {"balance": 98},
            }
        )
    )
    return path


def test_conservative_debit_never_refunds_gross_charge(tmp_path):
    receipt = session(tmp_path / "session.json", "source", [])
    assert conservative_charge(receipt) == 2.00


@pytest.mark.parametrize("prefix", ["MISMATCH_BLOCKS_REPLACEMENT", None])
def test_replacement_cannot_admit_without_prefix_match(tmp_path, prefix):
    source = tmp_path / "source.json"
    export_json(
        source,
        {
            "schema": "bl-p1-y3-source/v1",
            "status": "COMPLETE_SOURCE_CONTINUATION",
            "origin_mode": "IMPORTED_VERIFIED_ORIGIN",
            "assignment_endpoint": "source#generation=1",
            "prefix_witness": {"status": prefix},
        },
    )
    terminal = session(tmp_path / "terminal.json", "source", [{"sha256": digest(source)}])
    with pytest.raises(ValueError, match="matching prefix"):
        verify_replacement_prerequisites(
            source=source, source_supervisor=terminal, charged_units=22.13
        )


def test_replacement_requires_export_binding_and_actual_residual(tmp_path):
    source = tmp_path / "source.json"
    export_json(
        source,
        {
            "schema": "bl-p1-y3-source/v1",
            "status": "COMPLETE_SOURCE_CONTINUATION",
            "origin_mode": "IMPORTED_VERIFIED_ORIGIN",
            "assignment_endpoint": "source#generation=1",
            "prefix_witness": {"status": "MATCH"},
        },
    )
    terminal = session(tmp_path / "terminal.json", "source", [{"sha256": digest(source)}])
    assert (
        verify_replacement_prerequisites(
            source=source, source_supervisor=terminal, charged_units=22.13
        )["prefix_witness"]["status"]
        == "MATCH"
    )
    with pytest.raises(ValueError, match="actual residual"):
        verify_replacement_prerequisites(
            source=source, source_supervisor=terminal, charged_units=30
        )


@pytest.mark.parametrize("status,op", [("Y3_PASS_BOUNDARY_ONLY", "FAIL"), ("PASS", "FAIL")])
def test_qualification_refuses_incomplete_two_replay_pass(tmp_path, status, op):
    y3 = tmp_path / "y3.json"
    export_json(
        y3,
        {
            "schema": "bl-p1-y3/v1",
            "status": status,
            "operational_replay": {"status": op},
            "source_endpoint": "source",
            "replacement_endpoint": "replacement",
            "source_receipt_sha256": "a" * 64,
        },
    )
    source = session(tmp_path / "source-terminal.json", "source", [{"sha256": "a" * 64}])
    replacement = session(
        tmp_path / "replacement-terminal.json", "replacement", [{"sha256": digest(y3)}]
    )
    with pytest.raises(ValueError, match="full Y3 PASS"):
        verify_qualification_prerequisites(
            y3=y3, source_supervisor=source, replacement_supervisor=replacement
        )
