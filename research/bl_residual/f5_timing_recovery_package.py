"""Create-only F5 timing04 recovery package builder.

This module prepares no provider work.  It reads only attempt04 manifest
metadata and file statistics, then seals the current-source control, empty
ledgers, timing/canary manifests, and synthetic transport authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from research.bl_ppo.exports import export_bytes, export_json, verify_export
from research.bl_residual import f5_package
from research.bl_residual.f5_r10a_annex import allocate
from research.bl_residual.f5_timing_resume import empty_ledger, identity

RULING_SHA256 = "4f10f264fda3c7190e07f486803fa02cc10645741fb95ef215b1d1b0cc123439"
NAMESPACE = "bl-residual-f5-timing-recovery-20261026"
ENVIRONMENT_SHA256 = "bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5"
ALLOCATION_SHA256 = "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
B8_URL = (
    "https://github.com/josenunezwork/snake-dqn/releases/download/"
    "s1-inputs-20261019/B8_s3_final.pth"
)
ATTEMPT04_ELAPSED_SECONDS = 1756.3287814579671
TIMING03_ELAPSED_SECONDS = 5055.5119253752055
THREADS = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}


def canonical(value: Any) -> bytes:
    """Return the canonical bytes used by broker authority contracts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    """Read a Git blob without text decoding or newline stripping."""
    return subprocess.check_output(["git", *args], cwd=repo)


def _metadata_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt04 manifest is absent or unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("inputs"), list):
        raise ValueError("attempt04 manifest is malformed")
    return value


def _source_inputs(prior: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Copy only typed path/stat/hash metadata from attempt04's full inventory."""
    expected = {
        "b8.pth",
        "timing-fixtures.json",
        "champion_a5_freespace_20260621.pth",
        "best_apex_fs.pth",
        "best_apex_stage1_fs.pth",
        "best_apex_pre_fs.pth",
        "frp3_m3_s12_u60000_20261005.pth",
    }
    result: list[dict[str, Any]] = []
    for row in prior["inputs"]:
        if not isinstance(row, dict) or row.get("dest") not in expected:
            continue
        dest, claimed = row["dest"], row.get("sha256")
        if not isinstance(claimed, str) or len(claimed) != 64:
            raise ValueError("attempt04 input hash is malformed")
        if dest == "b8.pth":
            if (
                row.get("url") != B8_URL
                or claimed != B8_SHA256
                or row.get("size_bytes") != 6_562_005
            ):
                raise ValueError("attempt04 public B8 metadata drift")
            result.append({"dest": dest, "url": B8_URL, "sha256": claimed, "size_bytes": 6_562_005})
            continue
        path = Path(str(row.get("path", "")))
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("attempt04 private input is absent or unsafe")
        # The package never opens a payload.  Existing committed metadata binds
        # its digest and this stat check fixes the exact transfer byte count.
        result.append(
            {"dest": dest, "path": str(path), "sha256": claimed, "size_bytes": path.stat().st_size}
        )
    if {row["dest"] for row in result} != expected:
        raise ValueError("attempt04 full timing inventory is incomplete")
    return sorted(result, key=lambda row: row["dest"])


def _prior_environment_control(prior: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carry the recorded typed policy/capture as provenance, never fabricate one."""
    rows = [
        row
        for row in prior["inputs"]
        if isinstance(row, dict) and row.get("dest") == "control.json"
    ]
    if len(rows) != 1:
        raise ValueError("attempt04 control metadata is absent")
    path = Path(str(rows[0].get("path", "")))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("attempt04 control metadata is absent or unsafe")
    value = json.loads(path.read_text())
    policy, capture = value.get("environment_policy"), value.get("environment_capture")
    if not isinstance(policy, dict) or not isinstance(capture, dict):
        raise ValueError("attempt04 control lacks typed environment provenance")
    return dict(policy), dict(capture)


def _authority(commit: str, tree: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in inputs:
        record = {key: row[key] for key in ("dest", "size_bytes")}
        if "url" in row:
            record.update({"transport": "https", "url": row["url"], "sha256": row["sha256"]})
        else:
            record.update({"transport": "upload", "sha256": row["sha256"]})
        rows.append(record)
    unsigned = {
        "schema": "f5-timing-recovery-synthetic-input-inventory/v1",
        "ruling_sha256": RULING_SHA256,
        "source_commit": commit,
        "source_tree": tree,
        "inputs": rows,
    }
    return {**unsigned, "sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def _export_input(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    exported = export_json(path, value)
    verify_export(path)
    return {
        "dest": path.name,
        "path": str(path),
        "sha256": exported["sha256"],
        "size_bytes": exported["size"],
    }


def _export_sidecars(path: Path) -> list[dict[str, Any]]:
    """Describe the marker/sidecar bytes required by runtime ledger verification."""
    return [
        {
            "dest": path.name + suffix,
            "path": str(Path(str(path) + suffix)),
            "sha256": file_sha256(Path(str(path) + suffix)),
            "size_bytes": Path(str(path) + suffix).stat().st_size,
        }
        for suffix in (".sha256", ".complete")
    ]


def _control(
    source_hashes: dict[str, str],
    runtime_inputs: list[dict[str, Any]],
    input_sha: str,
    fixture_sha256: str,
    environment_policy: dict[str, Any],
    environment_capture: dict[str, Any],
    ledger_sha256: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "bl-residual-f5-runtime/v1",
        "namespace": "bl-residual-f5-w-20261025-v1",
        "identity": {
            "source_sha256": hashlib.sha256(
                json.dumps(source_hashes, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "loss_sha256": source_hashes["research/bl_residual/f5_fit.py"],
            "environment_sha256": ENVIRONMENT_SHA256,
            "allocation_sha256": ALLOCATION_SHA256,
            "corpus_sha256": "0" * 64,
        },
        "source_hashes": source_hashes,
        "qualification_b8_sha256": B8_SHA256,
        "qualification_fixture_sha256": fixture_sha256,
        "qualification_checkpoint_pool": {
            row["dest"]: row["sha256"]
            for row in runtime_inputs
            if row["dest"] not in {"b8.pth", "timing-fixtures.json"}
        },
        "input_hashes": {row["dest"]: row["sha256"] for row in runtime_inputs},
        "timing_input_sha256": input_sha,
        "environment_policy": environment_policy,
        "environment_capture": environment_capture,
        "ledger_sha256": ledger_sha256,
        "status": "HOLD_PENDING_CURRENT_SOURCE_ENVIRONMENT_AND_FAMILY_RECEIPT",
    }
    value["allocation"] = hashlib.sha256(
        json.dumps(
            {
                key: item
                for key, item in value.items()
                if key not in {"allocation", "ledger_sha256"}
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return value


def _admission(authority_path: Path) -> dict[str, Any]:
    return {
        "schema": "bl-f5-timing-operational-admission/v1",
        "status": "ACCEPTED_F5_TIMING_RECOVERY_ONLY",
        "scientific_fits_authorized": False,
        "numerical_looks_authorized": False,
        "release_authorized": False,
        "timing_recovery": {
            "schema": "bl-f5-timing-recovery/v1",
            "ruling_sha256": RULING_SHA256,
            "scope": "F5_W_TIMING04_FULL_RESUMABLE_OPERATION_ONLY",
        },
        "timing_recovery_authority": {
            "path": str(authority_path.resolve()),
            "sha256": file_sha256(authority_path),
            "size": authority_path.stat().st_size,
        },
    }


def build(
    output: Path, attempt04_manifest: Path, *, repo: Path = f5_package.ROOT
) -> dict[str, Any]:
    """Build a fresh, create-only package from a clean committed source closure."""
    if output.exists():
        raise ValueError("recovery package output must be fresh")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("recovery package requires a clean committed source checkout")
    commit, tree = _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")
    prior = _metadata_manifest(attempt04_manifest)
    environment_policy, environment_capture = _prior_environment_control(prior)
    runtime_inputs = _source_inputs(prior)
    hashes = f5_package.source_inventory(repo)
    for relative, expected in hashes.items():
        if hashlib.sha256(_git_bytes(repo, "show", f"{commit}:{relative}")).hexdigest() != expected:
            raise ValueError("source closure is not committed at HEAD")
    output.mkdir(parents=True)
    source_paths = sorted(
        set(hashes)
        | {
            f"{f5_package.ENV}/requirements-f5-l4-cu126.lock",
            f"{f5_package.ENV}/f5-common-environment-v6.json",
        }
    )
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", commit, "--", *source_paths], cwd=repo
    )
    source_archive = export_bytes(output / "source.tar", archive)
    input_sha = hashlib.sha256(
        canonical([{k: row[k] for k in ("dest", "sha256", "size_bytes")} for row in runtime_inputs])
    ).hexdigest()
    control_path = output / "inputs" / "control.json"
    fixture_rows = [row for row in allocate(0) if row.get("timing_only") is True]
    if [row["allocation_index"] for row in fixture_rows] != list(range(672, 684)):
        raise ValueError("frozen timing allocation is not the required twelve fixtures")
    fixtures_path = output / "inputs" / "timing-fixtures.json"
    fixture = _export_input(
        fixtures_path, {"schema": "bl-f5-sealed-timing-fixtures/v1", "fixtures": fixture_rows}
    )
    # The runtime consumes the package-local fixture descriptor.  Its digest is
    # deliberately part of the non-ledger identity, unlike control/ledger.
    runtime_inputs = [row for row in runtime_inputs if row["dest"] != "timing-fixtures.json"] + [
        fixture
    ]
    runtime_inputs.sort(key=lambda row: row["dest"])
    input_sha = hashlib.sha256(
        canonical([{k: row[k] for k in ("dest", "sha256", "size_bytes")} for row in runtime_inputs])
    ).hexdigest()
    provisional_source_sha = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    binding = identity(
        namespace=NAMESPACE,
        allocation=ALLOCATION_SHA256,
        source_sha256=provisional_source_sha,
        environment_sha256=ENVIRONMENT_SHA256,
        input_sha256=input_sha,
    )
    ledger_path = output / "inputs" / "verified-ledger.json"
    ledger_export = _export_input(ledger_path, empty_ledger(binding))
    ledger_sidecars = _export_sidecars(ledger_path)
    control = _control(
        hashes,
        runtime_inputs,
        input_sha,
        fixture["sha256"],
        environment_policy,
        environment_capture,
        ledger_export["sha256"],
    )
    control_export = _export_input(control_path, control)
    inputs = [*runtime_inputs, control_export, ledger_export, *ledger_sidecars]
    # The transport authority names every transferred byte, including the
    # package-local fixture/control/empty-ledger records.  It intentionally
    # excludes itself to avoid a self-referential file digest.
    authority = _authority(commit, tree, sorted(inputs, key=lambda row: row["dest"]))
    authority_path = output / "f5-timing-synthetic-inventory.json"
    export_json(authority_path, authority)
    verify_export(authority_path)
    from research.colab_cli.transfer_policy import f5_timing_recovery_binding

    base = {
        "commit": commit,
        "script": "research/bl_residual/f5_colab.sh",
        "device": "cuda:0",
        "scientific_admission": False,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 120},
        "timeout_minutes": 240,
        "supervisor_state_dir": str(output / "state-01"),
        "inputs": inputs,
        "source_paths": source_paths,
        "env": {
            **THREADS,
            "F5_MODE": "qualification",
            "F5_DEVICE": "cuda:0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "max_export_bytes": 2 * 1024**3,
        "budget": {
            "unit_cap": 100000.0,
            "rate_units_per_hour": 2.0,
            "reserve_units": 0.0,
            "other_assignments": {},
        },
        "f5_timing_synthetic_inventory": authority,
        "f5_timing_source_tree": tree,
        "f5_timing_transfer_exception": f5_timing_recovery_binding(authority["sha256"], "timing"),
        "operational_admission": _admission(authority_path),
        "resume": {
            "unit_kind": "f5-timing-unit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "f5": {
            "transfer_seconds": 300,
            "staging_seconds": 300,
            "base_minutes": 120,
            "effective_minutes": 240,
            "transfer_bar_seconds": 300,
            "staging_bar_seconds": 300,
            "private_upload_cap_bytes": 45_397_643,
            "private_upload_bytes_max": 45_397_643,
            "automatic_retries": False,
            "base_timeout_minutes": 120,
            "effective_timeout_minutes": 240,
            "useful_foreground_seconds": 14_355,
            "sizing_basis": "UNMEASURED_ENGINEERING_PLANNING_ALLOWANCE",
            "technical_budget": (
                "Nonbinding technical runaway accounting; wall envelope is authoritative."
            ),
        },
        "required_exports": [
            "qualification/qualification.json",
            "resume/verified-skip-proof.json",
        ],
        "timing_calibration": {
            "no_automatic_retry": True,
            "no_allocation_retry": True,
            "no_job_retry": True,
            "no_completion_retry": True,
        },
        "status": "HOLD_PENDING_CURRENT_SOURCE_ENVIRONMENT_AND_FAMILY_RECEIPT",
    }
    long = {
        **base,
        "job_id": f"bl-f5-timing-recovery-{commit[:8]}-01",
        "output_dir": str(output / "output-01"),
    }
    canary = {
        **base,
        "job_id": f"bl-f5-timing-recovery-canary-{commit[:8]}-01",
        "output_dir": str(output / "canary-output-01"),
        "supervisor_state_dir": str(output / "canary-state-01"),
        "requirements": {**base["requirements"], "est_minutes": 15},
        "timeout_minutes": 30,
        "inputs": inputs,
        "env": {
            **base["env"],
            "F5_MODE": "qualification-canary",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
        },
        "f5_timing_synthetic_inventory": authority,
        "f5_timing_source_tree": tree,
        "f5_timing_transfer_exception": f5_timing_recovery_binding(authority["sha256"], "canary"),
        "operational_admission": {
            **_admission(authority_path),
            "status": "ACCEPTED_F5_TIMING_RECOVERY_CANARY_ONLY",
            "timing_recovery": {
                "schema": "bl-f5-timing-recovery/v1",
                "ruling_sha256": RULING_SHA256,
                "scope": "F5_W_TIMING04_FAMILY_CANARY_ONLY",
            },
        },
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f5_colab.sh",
            "record_class": "CANARY_ONLY",
            "tiny_workload": True,
        },
        "required_exports": [
            "qualification/qualification.json",
            "resume/verified-skip-proof.json",
        ],
        "f5": {
            **base["f5"],
            "base_minutes": 15,
            "effective_minutes": 30,
            "base_timeout_minutes": 15,
            "effective_timeout_minutes": 30,
        },
        "status": "HOLD_PENDING_FAMILY_TRANSPORT_AND_CURRENT_SOURCE_RECEIPT",
    }
    long_file = export_json(output / "timing-operation.draft.json", long)
    canary_file = export_json(output / "family-canary.draft.json", canary)
    accounting = {
        "schema": "bl-f5-timing04-append-only-accounting/v1",
        "attempt03_elapsed_seconds": TIMING03_ELAPSED_SECONDS,
        "attempt04_elapsed_seconds": ATTEMPT04_ELAPSED_SECONDS,
        "cumulative_elapsed_seconds": TIMING03_ELAPSED_SECONDS + ATTEMPT04_ELAPSED_SECONDS,
        "attempt04_status": "INCOMPLETE_RPC_EXEC_POLL_TIMEOUT_NO_COMPLETED_UNITS",
        "historical_context_only": True,
    }
    accounting_file = export_json(output / "attempt04-append-only-accounting.json", accounting)
    packet = {
        "schema": "bl-f5-timing-recovery-package/v1",
        "status": "HOLD_PENDING_CURRENT_SOURCE_ENVIRONMENT_FAMILY_RECEIPT_AND_TRANSPORT_ACCEPTANCE",
        "source_commit": commit,
        "source_tree": tree,
        "ruling_sha256": RULING_SHA256,
        "source_inventory": hashes,
        "source_archive": source_archive,
        "source_paths": source_paths,
        "predecessor_source_paths": prior.get("source_paths"),
        "authority": {
            "path": str(authority_path),
            "sha256": file_sha256(authority_path),
            "inventory_sha256": authority["sha256"],
        },
        "control": control_export,
        "empty_ledger": ledger_export,
        "timing_operation": long_file,
        "family_canary": canary_file,
        "attempt04_accounting": accounting_file,
        "dispatch_authorized": False,
        "scientific_payload_opened": False,
    }
    return export_json(output / "PACKAGE.json", packet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt04-manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.attempt04_manifest), sort_keys=True))


if __name__ == "__main__":
    main()
