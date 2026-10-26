"""Narrow admission fence for the one accepted F5 plan materialization job."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "ops" / "dispatch.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("ops_f5_plan_admission", SOURCE)
assert SPEC is not None and SPEC.loader is not None
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)

ROOT = Path(__file__).resolve().parents[1]
RULINGS_ROOT = ROOT / "research/bl_residual/f5_round08_rulings_20261026"
RULING_FILES = {
    "acceptance": "ACCEPT_F5_W_NEXT_PACKAGES_20261026_ROUND08.md",
    "capture_qualification": "F5_W_CAPTURE_TRANSPORT_QUALIFIED_20261026.md",
    "parallel_clarification": "F5_W_ROUND08_PARALLEL_SEALING_CLARIFICATION_20261026.md",
    "owner_ruling": "OWNER_RULING_COLAB_BUDGET_20261025.txt",
}


def _reference(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def plan_manifest() -> dict:
    return {
        "scientific_admission": False,
        "commit": "c989f6738f6aad143eb97bc47f6bd9e6687057ce",
        "script": "research/bl_residual/f5_colab.sh",
        "device": "cuda:0",
        "env": dict(dispatch.F5_PLAN_ENV),
        "substeps": ["CPU_PLAN_MATERIALIZATION", "DISJOINT_CUDA_TRANSPORT_QUALIFICATION"],
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 15},
        "timeout_minutes": 30,
        "resume_exemption": {"kind": "short-canary", "reason": "plan only", "timeout_minutes": 30},
        "required_exports": list(dispatch.F5_PLAN_REQUIRED_EXPORTS),
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f5_colab.sh",
        },
        "max_export_bytes": 2 * 1024**3,
        "inputs": [
            {"dest": "selected-training.tar.xz", "sha256": dispatch.F5_SELECTED_TRAINING_SHA256},
            {"dest": "selected-training.tar.xz.sha256", "sha256": "a" * 64},
            {"dest": "selected-training.tar.xz.complete", "sha256": "b" * 64},
            {"dest": "timing-fixtures.json", "sha256": "c" * 64},
            {"dest": "control.json", "sha256": "d" * 64},
            {"dest": "verified-ledger.json", "sha256": "e" * 64},
            {
                "dest": "b8.pth",
                "url": dispatch.F5_B8_URL,
                "sha256": dispatch.F5_B8_SHA256,
                "size_bytes": 6562005,
            },
        ],
        "f5": {
            "base_timeout_minutes": 15,
            "effective_timeout_minutes": 30,
            "automatic_retries": False,
            "private_upload_cap_bytes": 45397643,
            "private_upload_bytes_max": 45397643,
            "transfer_bar_seconds": 150,
            "staging_bar_seconds": 300,
        },
        "timing_calibration": {
            "no_automatic_retry": True,
            "no_allocation_retry": True,
            "no_job_retry": True,
            "no_completion_retry": True,
        },
        "budget": {"other_assignments": {}},
        "operational_admission": {
            "schema": dispatch.F5_PLAN_ADMISSION_SCHEMA,
            "status": "ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY",
            "scientific_fits_authorized": False,
            "numerical_looks_authorized": False,
            "release_authorized": False,
            "rulings": {
                key: _reference(RULINGS_ROOT / filename) for key, filename in RULING_FILES.items()
            },
        },
    }


def test_exact_accepted_plan_admission_passes_with_real_ruling_bytes() -> None:
    dispatch.require_scientific_or_f5_plan_admission(plan_manifest())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update({"draft_only": {}}), "draft_only"),
        (lambda value: value["env"].update({"F5_MODE": "qualification"}), "prepared environment"),
        (lambda value: value.update({"substeps": ["CPU_PLAN_MATERIALIZATION"]}), "substeps"),
        (lambda value: value["timing_calibration"].update({"no_job_retry": False}), "retry fences"),
        (
            lambda value: value["operational_admission"].update({"release_authorized": True}),
            "authority",
        ),
        (
            lambda value: value["inputs"][-1].update({"url": "https://example.invalid/b8.pth"}),
            "HTTPS B8",
        ),
        (lambda value: value["required_exports"].pop(), "export inventory"),
        (
            lambda value: value["operational_admission"]["rulings"]["owner_ruling"].update(
                {"size": 1}
            ),
            "bytes differ",
        ),
    ],
)
def test_plan_admission_rejects_any_widening(mutate, match: str) -> None:
    value = copy.deepcopy(plan_manifest())
    mutate(value)
    with pytest.raises(dispatch.Refused, match=match):
        dispatch.require_scientific_or_f5_plan_admission(value)


def test_false_scientific_admission_without_the_exact_schema_remains_refused() -> None:
    value = plan_manifest()
    value.pop("operational_admission")
    with pytest.raises(dispatch.Refused, match="scientific_admission=false"):
        dispatch.require_scientific_or_f5_plan_admission(value)


def test_live_assignment_drift_requires_a_sealed_successor() -> None:
    original = plan_manifest()
    live = copy.deepcopy(original)
    live["budget"]["other_assignments"] = {"endpoint-1": 1.54}
    with pytest.raises(dispatch.Refused, match="seal a successor"):
        dispatch.require_f5_plan_live_assignment_match(original, live)
