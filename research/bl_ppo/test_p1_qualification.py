"""A fast collection ruler is insufficient to admit the P1 tranche."""

from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_manifests import QUAL_PAUSE
from research.bl_ppo.p1_qualification import finalize


def inputs(root: Path, *, stop_verified: bool = True, actor_seconds: float = 700) -> dict:
    run = root / "qualification-runtime.json"
    export_json(
        run,
        {
            "schema": "bl-p1-qualification-runtime/v1",
            "status": "PAUSED",
            "critic_qualified": True,
            "actor_transitions": QUAL_PAUSE,
            "target": 100_000_000,
            "source_commit": "a" * 40,
            "actor_phase_seconds": actor_seconds,
            "timing_includes_collection_updates_probes": True,
        },
    )
    y3 = root / "y3.json"
    export_json(y3, {"schema": "bl-p1-y3/v1", "status": "PASS"})
    sessions = []
    for name in ("source", "replacement", "qualification"):
        path = root / f"{name}-supervisor.json"
        export_json(
            path,
            {
                "status": "PASS",
                "stop_verified": stop_verified,
                "cleanup_error": None,
                "error": None,
                "endpoint": name,
                "commit": "a" * 40,
                "terminal": {"status": "succeeded", "returncode": 0},
                "elapsed_seconds": 900,
                "exports": [{"path": run.name, "sha256": digest(run)}],
            },
        )
        sessions.append(path)
    return dict(
        runtime=run,
        supervisor=sessions[2],
        y3=y3,
        source_supervisor=sessions[0],
        replacement_supervisor=sessions[1],
        output=root / "receipt.json",
    )


def test_complete_job_qualification_accounts_for_release_and_overhead(tmp_path: Path) -> None:
    result = finalize(**inputs(tmp_path))
    assert result["status"] == "PASS"
    assert result["per_job_overhead_seconds"] == 200
    assert result["prefix_seconds"] == 2700
    assert result["invoice_units"] <= 120


def test_unverified_release_cannot_fund_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="verified release"):
        finalize(**inputs(tmp_path, stop_verified=False))
    assert not (tmp_path / "receipt.json").exists()


def test_slow_complete_job_is_retained_as_not_ready(tmp_path: Path) -> None:
    result = finalize(**inputs(tmp_path, actor_seconds=899))
    # At ~1112/s, the 120-unit 2x-liability envelope is the binding gate.
    assert result["status"] == "NOT_READY"
    assert "120u" in result["reason"] or "liability" in result["reason"]


def test_runtime_cannot_claim_more_time_than_whole_assignment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection\\+update\\+probe"):
        finalize(**inputs(tmp_path, actor_seconds=901))
