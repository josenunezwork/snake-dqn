"""Narrow admission fence for the accepted F5 timing qualification package."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
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


def requalified_timing_manifest(tmp_path: Path, monkeypatch) -> dict:
    """Keep evidence tests hermetic while exercising every successor binding."""
    value = timing_manifest()
    value["commit"] = dispatch.F5_PLAN_REQUAL_COMMIT
    bindings = {}
    for key in dispatch.F5_TIMING_REQUAL_BINDINGS:
        path = tmp_path / key
        path.write_bytes(f"reviewed timing evidence for {key}".encode())
        reference = _reference(path)
        bindings[key] = reference["sha256"]
        value["operational_admission"][key] = reference
    monkeypatch.setattr(dispatch, "F5_TIMING_REQUAL_BINDINGS", bindings)
    return value


def test_requalified_timing_requires_all_immutable_evidence_bindings(tmp_path, monkeypatch) -> None:
    value = requalified_timing_manifest(tmp_path, monkeypatch)
    dispatch.require_scientific_or_f5_plan_admission(value)
    for key in dispatch.F5_TIMING_REQUAL_BINDINGS:
        missing = copy.deepcopy(value)
        missing["operational_admission"].pop(key)
        with pytest.raises(dispatch.Refused, match="exact evidence binding"):
            dispatch.require_scientific_or_f5_plan_admission(missing)


@pytest.mark.parametrize("key", list(dispatch.F5_TIMING_REQUAL_BINDINGS))
@pytest.mark.parametrize("change", ["payload", "sha", "size", "symlink", "relative", "missing"])
def test_requalified_timing_rejects_evidence_drift(tmp_path, monkeypatch, key, change) -> None:
    value = requalified_timing_manifest(tmp_path, monkeypatch)
    reference = value["operational_admission"][key]
    match = "evidence bytes differ"
    if change == "payload":
        Path(reference["path"]).write_bytes(b"unreviewed")
    elif change == "sha":
        reference["sha256"] = "0" * 64
    elif change == "size":
        reference["size"] = 0
    elif change == "symlink":
        link = tmp_path / "link"
        link.symlink_to(reference["path"])
        reference["path"] = str(link)
        match = "regular absolute file"
    elif change == "relative":
        reference["path"] = key
        match = "regular absolute file"
    elif change == "missing":
        Path(reference["path"]).unlink()
        match = "regular absolute file"
    with pytest.raises(dispatch.Refused, match=match):
        dispatch.require_scientific_or_f5_plan_admission(value)


@pytest.mark.parametrize("change", ["source", "scope", "retry", "account"])
def test_requalified_timing_keeps_existing_fences(tmp_path, monkeypatch, change) -> None:
    value = requalified_timing_manifest(tmp_path, monkeypatch)
    if change == "source":
        value["commit"] = "f" * 40
        match = "accepted implementation commit"
    elif change == "scope":
        value["env"]["F5_MODE"] = "transport-qualification"
        match = "qualification environment"
    elif change == "retry":
        value["timing_calibration"]["no_completion_retry"] = False
        match = "retry fences"
    else:
        live = copy.deepcopy(value)
        live["budget"]["other_assignments"] = {}
        with pytest.raises(dispatch.Refused, match="seal a successor"):
            dispatch.require_f5_plan_live_assignment_match(value, live)
        return
    with pytest.raises(dispatch.Refused, match=match):
        dispatch.require_scientific_or_f5_plan_admission(value)


def recovery_manifest(tmp_path: Path, *, canary: bool = False) -> dict:
    """Build the separately-bound timing04 recovery shape without provider I/O."""
    import research.colab_cli.transfer_policy as transfer_policy
    from research.bl_ppo.exports import export_json

    value = timing_manifest()
    value["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    value["env"] = dict(dispatch.F5_TIMING_RECOVERY_ENV)
    if canary:
        value["env"].update(
            {
                "F5_MODE": "qualification-canary",
                "SNAKE_COLAB_CANARY": "1",
                "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
            }
        )
    value["requirements"]["est_minutes"] = 15 if canary else 120
    value["timeout_minutes"] = 30 if canary else 240
    value["f5"].update(
        {
            "base_minutes": 15 if canary else 120,
            "effective_minutes": 30 if canary else 240,
            "transfer_seconds": 300,
            "staging_seconds": 300,
        }
    )
    value["f5"]["sizing_basis"] = "UNMEASURED_ENGINEERING_PLANNING_ALLOWANCE"
    value["timing_calibration"] = {
        "no_automatic_retry": True,
        "no_allocation_retry": True,
        "no_job_retry": True,
        "no_completion_retry": True,
    }
    input_root = tmp_path / "inputs"
    control = export_json(input_root / "control.json", {"control": "test"})
    ledger = export_json(input_root / "verified-ledger.json", {"units": []})
    value["inputs"] = [
        {
            "dest": "b8.pth",
            "url": dispatch.F5_B8_URL,
            "sha256": dispatch.F5_B8_SHA256,
            "size_bytes": 6562005,
        },
        {
            "dest": "control.json",
            "path": str(input_root / "control.json"),
            "sha256": control["sha256"],
            "size_bytes": control["size"],
        },
        {
            "dest": "verified-ledger.json",
            "path": str(input_root / "verified-ledger.json"),
            "sha256": ledger["sha256"],
            "size_bytes": ledger["size"],
        },
    ]
    source_tree = subprocess.check_output(
        ["git", "rev-parse", f"{value['commit']}^{{tree}}"], cwd=ROOT, text=True
    ).strip()
    value["f5_timing_source_tree"] = source_tree
    authority = {
        "schema": "f5-timing-recovery-synthetic-input-inventory/v1",
        "ruling_sha256": dispatch.F5_TIMING_RECOVERY_RULING_SHA256,
        "source_commit": value["commit"],
        "source_tree": source_tree,
        "inputs": [
            {
                **{key: row[key] for key in ("dest", "size_bytes", "sha256")},
                **(
                    {"transport": "https", "url": row["url"], "sha256": row["sha256"]}
                    if "url" in row
                    else {"transport": "upload"}
                ),
            }
            for row in value["inputs"]
        ],
    }
    authority["sha256"] = hashlib.sha256(
        json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    authority_path = tmp_path / "timing-authority.json"
    authority_path.write_text(json.dumps(authority, sort_keys=True) + "\n")
    value["f5_timing_synthetic_inventory"] = authority
    value["f5_timing_transfer_exception"] = transfer_policy.f5_timing_recovery_binding(
        authority["sha256"], "canary" if canary else "timing"
    )
    value["operational_admission"] = {
        "schema": dispatch.F5_TIMING_ADMISSION_SCHEMA,
        "status": (
            "ACCEPTED_F5_TIMING_RECOVERY_CANARY_ONLY"
            if canary
            else "ACCEPTED_F5_TIMING_RECOVERY_ONLY"
        ),
        "scientific_fits_authorized": False,
        "numerical_looks_authorized": False,
        "release_authorized": False,
        "timing_recovery": {
            "schema": dispatch.F5_TIMING_RECOVERY_SCHEMA,
            "ruling_sha256": dispatch.F5_TIMING_RECOVERY_RULING_SHA256,
            "scope": (
                "F5_W_TIMING04_FAMILY_CANARY_ONLY"
                if canary
                else "F5_W_TIMING04_FULL_RESUMABLE_OPERATION_ONLY"
            ),
        },
        "timing_recovery_authority": _reference(authority_path),
    }
    value.pop("resume_exemption")
    value["resume"] = {
        "unit_kind": "f5-timing-unit",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }
    if canary:
        value["canary"] = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f5_colab.sh",
            "record_class": "CANARY_ONLY",
            "tiny_workload": True,
        }
        value["required_exports"] = [
            "qualification/qualification.json",
            "timing-canary/fixture-672.json",
            "timing-canary/fixture-673.json",
        ]
    else:
        value.pop("canary")
        value["required_exports"] = [
            "qualification/qualification.json",
            *[f"timing-units/fixture-{index}.json" for index in range(672, 684)],
            "timing-units/calibration.json",
        ]
    return value


def test_timing_recovery_requires_the_exact_long_resumable_contract(tmp_path) -> None:
    dispatch.require_scientific_or_f5_plan_admission(recovery_manifest(tmp_path))


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["f5"].update({"transfer_seconds": 150}), "300-second"),
        (lambda value: value["f5"].update({"sizing_basis": "measured"}), "unmeasured"),
        (lambda value: value["requirements"].update({"est_minutes": 15}), "120/240"),
        (lambda value: value["requirements"].update({"min_vcpu": 5}), "L4 placement"),
        (lambda value: value.update({"timeout_minutes": 30}), "120/240"),
        (lambda value: value["resume"].update({"unit_kind": "episode"}), "resumable unit"),
        (lambda value: value["resume"].update({"skip_verified": 1}), "resume types"),
        (lambda value: value.update({"resume_exemption": {}}), "forbids exemption"),
        (lambda value: value["env"].update({"F5_LEDGER": "/tmp/foreign"}), "runtime environment"),
        (
            lambda value: value["env"].update({"F5_REQUIREMENTS_LOCK": "/tmp/foreign"}),
            "runtime environment",
        ),
        (lambda value: value["timing_calibration"].update({"no_job_retry": False}), "retry fences"),
        (lambda value: value["required_exports"].pop(), "all thirteen"),
        (
            lambda value: value["operational_admission"].update({"release_authorized": True}),
            "authority",
        ),
        (
            lambda value: value["f5_timing_synthetic_inventory"].update({"source_tree": "c" * 40}),
            "inventory differs",
        ),
        (lambda value: value.update({"f5_timing_source_tree": "c" * 40}), "manifest source tree"),
    ],
)
def test_timing_recovery_refuses_widening_or_drift(tmp_path, mutate, match: str) -> None:
    value = recovery_manifest(tmp_path)
    mutate(value)
    with pytest.raises(dispatch.Refused, match=match):
        dispatch.require_scientific_or_f5_plan_admission(value)


def test_timing_recovery_canary_is_bounded_and_marked_canary_only(tmp_path) -> None:
    dispatch.require_scientific_or_f5_plan_admission(recovery_manifest(tmp_path, canary=True))


def test_timing_recovery_refuses_markerless_ledger_input(tmp_path) -> None:
    value = recovery_manifest(tmp_path)
    ledger = next(item for item in value["inputs"] if item["dest"] == "verified-ledger.json")
    Path(str(ledger["path"]) + ".complete").unlink()
    with pytest.raises(dispatch.Refused, match="immutable sidecars"):
        dispatch.require_scientific_or_f5_plan_admission(value)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["canary"].update({"record_class": "TIMING_UNIT"}), "CANARY_ONLY"),
        (lambda value: value["canary"].update({"tiny_workload": False}), "tiny-workload"),
        (lambda value: value["requirements"].update({"est_minutes": 120}), "15/30"),
        (lambda value: value.update({"timeout_minutes": 240}), "15/30"),
    ],
)
def test_timing_recovery_canary_refuses_noncanary_controls(tmp_path, mutate, match: str) -> None:
    value = recovery_manifest(tmp_path, canary=True)
    mutate(value)
    with pytest.raises(dispatch.Refused, match=match):
        dispatch.require_scientific_or_f5_plan_admission(value)
