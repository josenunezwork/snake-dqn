"""Offline guard tests for the pre-workload P1 supervisor retry builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_ppo import p1_kernel_ambiguity_retry as retry


def test_retry_identity_changes_only_job_and_output() -> None:
    manifest = {"job_id": "old", "output_dir": "/old", "science": {"frozen": True}}
    result = retry._retry_identity(manifest, job_id="new", output=Path("/new"))
    assert result == {"job_id": "new", "output_dir": "/new", "science": {"frozen": True}}
    assert manifest["job_id"] == "old"


def test_prepare_refuses_a_prior_non_supervisor_stop(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "canary-output").mkdir()
    (source / "canary-output/receipt.json").write_text(
        json.dumps({"status": "INCOMPLETE", "stop_verified": True, "failure_class": "WORKLOAD"})
    )
    original = retry._read_export
    retry._read_export = lambda _path: {}  # type: ignore[assignment]
    with pytest.raises(ValueError, match="ruled pre-workload supervisor stop"):
        try:
            retry.prepare(
                source=source,
                output=tmp_path / "fresh",
                qualification=tmp_path / "qualification.json",
                repo=tmp_path,
            )
        finally:
            retry._read_export = original  # type: ignore[assignment]
