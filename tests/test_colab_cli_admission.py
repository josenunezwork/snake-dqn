"""Profile and evidence regression gates for S1's shared transport admission."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import acceptance, admission, supervisor

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def passing_result(tmp_path: Path) -> dict[str, Any]:
    artifact = tmp_path / "final-artifact.bin"
    with artifact.open("wb") as stream:
        stream.truncate(acceptance.FINAL_ARTIFACT_BYTES)
    transfer = {
        "status": "PASS",
        "hardware": "L4",
        "profile": "final-only",
        "path": str(artifact),
        "sha256": acceptance.sha256_file(artifact),
        "exported_bytes": acceptance.FINAL_ARTIFACT_BYTES,
        "elapsed_seconds": 16,
        "bytes_per_second_lower_bound": acceptance.FINAL_ARTIFACT_BYTES // 16,
        "verified_on_mac": True,
    }
    gates = [
        {**gate, "status": "OBSERVED" if gate["required"] else "NOT_REQUIRED"}
        for gate in acceptance.review_gates()
    ]
    for gate in gates:
        if gate["id"] == "final-artifact-download":
            gate["evidence_record"] = transfer
        if gate["id"] == "cold-kernel-metadata-recovery":
            gate["evidence_record"] = {
                "status": "PASS",
                "arms": [
                    {
                        "arm": arm,
                        "metadata_recovery": {
                            "original_kernel_id": f"{arm}-kernel",
                            "recovered_kernel_id": f"{arm}-kernel",
                            "same_kernel_id": True,
                            "tracked_metadata_unchanged": True,
                        },
                        "first_upload_transfer": {
                            "remote": f"/content/{arm}-credential.json",
                            "attempts": 1,
                            "size_bytes": 128,
                            "elapsed_seconds": 1.0,
                            "status": "completed",
                        },
                    }
                    for arm in ("cpu", "L4")
                ],
            }
    execution = []
    for arm in ("cpu", "l4"):
        receipt = tmp_path / (arm + ".json")
        receipt.write_text(json.dumps({"stop_verified": True, "cleanup_error": None}))
        execution.append(
            {
                "job": arm,
                "status": "PASS",
                "receipt": str(receipt),
                "receipt_sha256": acceptance.sha256_file(receipt),
            }
        )
    sources = acceptance.source_manifest(REPO)
    return {
        "schema": "snake-dqn-colab-cli-acceptance-result/v1",
        "profile": "final-only",
        "qualified": True,
        "status": "PASS",
        "return_code": 0,
        "source_manifest": sources,
        "final_artifact_transfer": transfer,
        "final_plan": {
            "profile": "final-only",
            "status": "PASS",
            "gates": gates,
            "source_manifest": sources,
            "execution": execution,
        },
    }


def test_final_only_requires_verified_artifact_and_clean_release(passing_result: dict) -> None:
    assert admission.require_final_only(passing_result, REPO)["exported_bytes"] == 64 * 1024**2
    Path(passing_result["final_artifact_transfer"]["path"]).write_bytes(b"changed")
    with pytest.raises(supervisor.Refused, match="artifact is absent or changed"):
        admission.require_final_only(passing_result, REPO)


@pytest.mark.parametrize(
    "fault",
    [
        "old-profile",
        "incomplete",
        "old-source",
        "missing-gate",
        "pass-omitted-gate",
        "missing-arm",
        "changed-receipt",
        "cold-kernel-upload",
        "cold-kernel-missing-arm",
        "cold-kernel-changed-id",
        "cold-kernel-missing-evidence",
        "wrong-size",
        "inflated-throughput",
        "nan-throughput",
    ],
)
def test_final_only_refuses_unqualified_evidence(passing_result: dict, fault: str) -> None:
    result = copy.deepcopy(passing_result)
    if fault == "old-profile":
        result["profile"] = "full-state"
    elif fault == "incomplete":
        result["qualified"] = False
    elif fault == "old-source":
        result["source_manifest"]["research/colab_cli/supervisor.py"] = "0" * 64
    elif fault == "missing-gate":
        result["final_plan"]["gates"].pop()
    elif fault == "pass-omitted-gate":
        next(g for g in result["final_plan"]["gates"] if not g["required"])["status"] = "OBSERVED"
    elif fault == "missing-arm":
        result["final_plan"]["execution"].pop()
    elif fault == "changed-receipt":
        Path(result["final_plan"]["execution"][0]["receipt"]).write_text("{}")
    elif fault == "cold-kernel-upload":
        cold = next(
            gate
            for gate in result["final_plan"]["gates"]
            if gate["id"] == "cold-kernel-metadata-recovery"
        )
        cold["evidence_record"]["arms"][1]["first_upload_transfer"]["status"] = "failed"
    elif fault == "cold-kernel-missing-arm":
        cold = next(
            gate
            for gate in result["final_plan"]["gates"]
            if gate["id"] == "cold-kernel-metadata-recovery"
        )
        cold["evidence_record"]["arms"].pop()
    elif fault == "cold-kernel-changed-id":
        cold = next(
            gate
            for gate in result["final_plan"]["gates"]
            if gate["id"] == "cold-kernel-metadata-recovery"
        )
        cold["evidence_record"]["arms"][0]["metadata_recovery"]["recovered_kernel_id"] = "other"
    elif fault == "cold-kernel-missing-evidence":
        cold = next(
            gate
            for gate in result["final_plan"]["gates"]
            if gate["id"] == "cold-kernel-metadata-recovery"
        )
        cold.pop("evidence_record")
    elif fault == "wrong-size":
        result["final_artifact_transfer"]["exported_bytes"] -= 1
    elif fault == "inflated-throughput":
        result["final_artifact_transfer"]["bytes_per_second_lower_bound"] *= 2
    elif fault == "nan-throughput":
        result["final_artifact_transfer"]["bytes_per_second_lower_bound"] = float("nan")
    with pytest.raises(supervisor.Refused):
        admission.require_final_only(result, REPO)


def test_s1_shared_admission_requires_hash_bound_profile_input() -> None:
    with pytest.raises(supervisor.Refused, match="cli-acceptance-result.json"):
        admission.check_study_manifest({"script": "research/s1_mature_20261018/train.sh"}, REPO)
    admission.check_study_manifest({"script": "research/other/train.sh"}, REPO)


def test_s1_compatibility_wrapper_validates_separate_size_receipts(
    passing_result: dict, tmp_path: Path
) -> None:
    from research.s1_mature_20261018 import final_artifacts

    raw = tmp_path / "raw-acceptance.json"
    raw.write_text(json.dumps(passing_result))
    sizes = {}
    for arm in ("n3", "n1"):
        path = tmp_path / f"{arm}-size.json"
        path.write_text(
            json.dumps(
                {
                    "schema": final_artifacts.SIZE_SCHEMA,
                    "status": "PASS",
                    "arm": arm,
                    "source_manifest": final_artifacts.source_manifest(final_artifacts.SOURCES),
                    "production_contract_validated": True,
                    "export_contract": final_artifacts.PROTOCOL,
                    "measured_final_bundle_bytes": 12_000_000,
                    "final_bundle_bytes_upper_bound": acceptance.FINAL_ARTIFACT_BYTES,
                    "telemetry_log_allowance_bytes": 1_000_000,
                    "artifact_files": sorted(final_artifacts.REQUIRED_ARTIFACT_FILES),
                }
            )
        )
        sizes[arm] = path
    output = tmp_path / "compatibility.json"
    admission.bind_s1_transfer(raw, sizes, output, REPO)
    wrapped = json.loads(output.read_text())
    assert admission.require_final_only(wrapped, REPO) == passing_result["final_artifact_transfer"]
    assert wrapped["s1_final_artifact_transfer"]["transport_source_manifest"] == (
        final_artifacts.source_manifest(final_artifacts.CLI_SOURCES)
    )
    assert raw.read_text() == json.dumps(passing_result)
    with pytest.raises(FileExistsError):
        admission.bind_s1_transfer(raw, sizes, output, REPO)
    sizes["n1"].write_text("{}")
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="size measurement"):
        admission.bind_s1_transfer(raw, sizes, invalid, REPO)
    assert not invalid.exists()


def test_s1_manifest_checks_accepted_source_against_job_commit(
    passing_result: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "acceptance.json"
    raw.write_text(json.dumps(passing_result))
    checked = []

    def check(plan: dict, repo: Path) -> None:
        checked.append((plan, repo))
        raise supervisor.Refused("test: stale pinned transport")

    monkeypatch.setattr(acceptance, "verify_plan_sources", check)
    with pytest.raises(supervisor.Refused, match="stale pinned transport"):
        admission.check_study_manifest(
            {
                "script": "research/s1_mature_20261018/train.sh",
                "commit": "f" * 40,
                "inputs": [{"dest": "cli-acceptance-result.json", "path": str(raw)}],
            },
            REPO,
        )
    assert checked == [
        ({"source_manifest": passing_result["source_manifest"], "commit": "f" * 40}, REPO)
    ]
