"""Narrow admission fence for the accepted F5 timing qualification package."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "ops" / "dispatch.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("ops_f5_timing_admission", SOURCE)
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


def timing_manifest() -> dict:
    inputs = [
        {"dest": dest, "sha256": digest}
        for dest, digest in dispatch.F5_TIMING_INPUT_SHA256.items()
        if dest != "b8.pth"
    ]
    inputs.extend(
        [
            {
                "dest": "b8.pth",
                "url": dispatch.F5_B8_URL,
                "sha256": dispatch.F5_B8_SHA256,
                "size_bytes": 6562005,
            },
            {"dest": "control.json", "sha256": "a" * 64},
            {"dest": "verified-ledger.json", "sha256": "b" * 64},
        ]
    )
    return {
        "scientific_admission": False,
        "commit": dispatch.F5_PLAN_COMMIT,
        "script": "research/bl_residual/f5_colab.sh",
        "device": "cuda:0",
        "env": dict(dispatch.F5_TIMING_ENV),
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 15},
        "timeout_minutes": 30,
        "resume_exemption": {
            "kind": "short-canary",
            "reason": "timing only",
            "timeout_minutes": 30,
        },
        "required_exports": list(dispatch.F5_TIMING_REQUIRED_EXPORTS),
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f5_colab.sh",
        },
        "max_export_bytes": 2 * 1024**3,
        "inputs": inputs,
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
        "budget": {"other_assignments": {"plan-live-endpoint": 1.54}},
        "operational_admission": {
            "schema": dispatch.F5_TIMING_ADMISSION_SCHEMA,
            "status": "ACCEPTED_FULL_TIMING_PARITY_AND_CALIBRATION_ONLY",
            "scientific_fits_authorized": False,
            "numerical_looks_authorized": False,
            "release_authorized": False,
            "rulings": {
                key: _reference(RULINGS_ROOT / filename) for key, filename in RULING_FILES.items()
            },
        },
    }


def test_exact_accepted_timing_admission_passes_with_real_ruling_bytes() -> None:
    dispatch.require_scientific_or_f5_plan_admission(timing_manifest())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update({"draft_only": {}}), "draft_only"),
        (lambda value: value.update({"substeps": []}), "substeps"),
        (lambda value: value["env"].update({"F5_MODE": "prepare-qualification"}), "environment"),
        (lambda value: value["inputs"].pop(), "input inventory"),
        (
            lambda value: value["inputs"].append(
                {"dest": "selected-training.tar.xz", "sha256": "c" * 64}
            ),
            "input inventory",
        ),
        (
            lambda value: next(
                item for item in value["inputs"] if item["dest"] == "best_apex_fs.pth"
            ).update({"sha256": "d" * 64}),
            "opponent inputs",
        ),
        (lambda value: value["timing_calibration"].update({"no_job_retry": False}), "retry fences"),
        (
            lambda value: value["operational_admission"].update({"release_authorized": True}),
            "authority",
        ),
    ],
)
def test_timing_admission_rejects_any_widening(mutate, match: str) -> None:
    value = copy.deepcopy(timing_manifest())
    mutate(value)
    with pytest.raises(dispatch.Refused, match=match):
        dispatch.require_scientific_or_f5_plan_admission(value)


def test_timing_live_assignment_drift_requires_a_sealed_successor() -> None:
    original = timing_manifest()
    live = copy.deepcopy(original)
    live["budget"]["other_assignments"] = {}
    with pytest.raises(dispatch.Refused, match="seal a successor"):
        dispatch.require_f5_plan_live_assignment_match(original, live)
