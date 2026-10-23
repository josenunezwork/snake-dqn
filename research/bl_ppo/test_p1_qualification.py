"""A fast collection ruler is insufficient to admit the P1 tranche."""

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_manifests import QUAL_PAUSE
from research.bl_ppo.p1_qualification import finalize


def inputs(
    root: Path, *, stop_verified: bool = True, actor_seconds: float = 700, y3_status: str = "PASS"
) -> dict:
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
            "origin_snapshot_sha256": "b" * 64,
        },
    )
    y3 = root / "y3.json"
    export_json(
        y3,
        {
            "schema": "bl-p1-y3/v1",
            "status": y3_status,
            "operational_replay": {"status": "PASS" if y3_status == "PASS" else "FAIL"},
            "source_endpoint": "source",
            "replacement_endpoint": "replacement",
            "source_commit": "a" * 40,
            "source_receipt_sha256": "c" * 64,
            "origin_snapshot_sha256": "b" * 64,
        },
    )
    sessions = []
    for name in ("source", "replacement", "qualification"):
        path = root / f"{name}-supervisor.json"
        # Match the real supervisor convention: create-only JSON, no sidecars.
        payload = {
            "status": "PASS",
            "stop_verified": stop_verified,
            "cleanup_error": None,
            "error": None,
            "endpoint": name,
            "commit": "a" * 40,
            "terminal": {"status": "succeeded", "returncode": 0},
            "elapsed_seconds": 900,
            "transfers": [{"direction": "download", "status": "verified", "elapsed_seconds": 10}],
            "cli_calls": [
                {"command": "stop", "status": "completed", "elapsed_seconds": 1},
                {"command": "sessions", "status": "completed", "elapsed_seconds": 2},
            ],
            "exports": [{"path": run.name, "sha256": digest(run)}],
        }
        if name == "source":
            payload["exports"] = [{"path": "receipt.json", "sha256": "c" * 64}]
        if name == "replacement":
            payload["exports"] = [{"path": "receipt.json", "sha256": digest(y3)}]
        with path.open("x") as stream:
            json.dump(payload, stream)
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


@pytest.mark.parametrize("which", ["source_supervisor", "replacement_supervisor"])
def test_unrelated_successful_sessions_cannot_supply_y3(tmp_path: Path, which: str) -> None:
    arguments = inputs(tmp_path)
    path = arguments[which]
    receipt = json.loads(path.read_text())
    receipt["exports"] = [{"path": "unrelated.json", "sha256": "d" * 64}]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="terminal exports"):
        finalize(**arguments)


def test_boundary_only_y3_cannot_fund_qualification(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full Y3 PASS"):
        finalize(**inputs(tmp_path, y3_status="Y3_PASS_BOUNDARY_ONLY"))
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("field", ["transfers", "cli_calls"])
def test_missing_final_transfer_or_release_timing_blocks_funding(
    tmp_path: Path, field: str
) -> None:
    arguments = inputs(tmp_path)
    path = arguments["supervisor"]
    value = json.loads(path.read_text())
    del value[field]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="final transfer/release timing"):
        finalize(**arguments)
    assert not arguments["output"].exists()


def test_failed_stop_with_verified_disappearance_has_release_timing(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    path = arguments["supervisor"]
    value = json.loads(path.read_text())
    value["cli_calls"][0]["status"] = "failed"
    path.write_text(json.dumps(value))
    result = finalize(**arguments)
    assert result["final_transfer_release_timing"]["stop_and_disappearance_seconds"] == 3
