"""Prepare DENSITY-2 manifests offline; admission remains deliberately separate."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.bl_density.canary import verify_proof
from research.bl_ppo.exports import export_json
from research.bl_upload import assert_upload_budget
from research.colab_cli import canary as family_canary
from research.colab_cli.schema import validate_manifest
from research.colab_cli.supervisor import Refused, validate

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/bl_density"
REQUIRED_EXPORTS = ["density/summary.json"]
L4_RATE = 1.54
CANARY_CAP, CANARY_RESERVE, CANARY_TIMEOUT = 0.5, 0.1, 15
PRODUCTION_CAP, PRODUCTION_RESERVE, PRODUCTION_TIMEOUT = 5.0, 1.0, 180
ACCEPTANCE_RESULT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-main-acceptance-20261022/"
    "acceptance-1792680738/acceptance-result.json"
)


def digest(path: Path) -> str:
    """Hash a bounded input or package file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(repo: Path = ROOT) -> list[str]:
    """Include the static local import closure, never an artifact directory."""
    pending = [
        repo / PACKAGE / "runner.py",
        repo / PACKAGE / "canary.py",
        repo / "research/bl_upload/inputs.py",
    ]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for parent in path.parents:
            if parent == repo:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer not in visited:
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    package = list(path.relative_to(repo).parts[:-1])
                    prefix = ".".join(
                        package[: len(package) - node.level + 1] + ([prefix] if prefix else [])
                    )
                if prefix:
                    names = [prefix, *(f"{prefix}.{item.name}" for item in node.names)]
            for name in names:
                base = repo.joinpath(*name.split("."))
                for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                    if candidate.is_file() and candidate not in visited:
                        pending.append(candidate)
    return sorted(
        {str(path.relative_to(repo)) for path in visited}
        | {
            PACKAGE + "/colab.sh",
            PACKAGE + "/requirements.txt",
            PACKAGE + "/METRICS.md",
            PACKAGE + "/METRICS_AMENDMENT_01.md",
            PACKAGE + "/METRICS_AMENDMENT_02.md",
            PACKAGE + "/METRICS_AMENDMENT_03.md",
            PACKAGE + "/METRICS_AMENDMENT_04.md",
            "research/colab_cli",
            "research/t34_exit_b8/PUBLIC_INPUTS.json",
            "research/apex_safety_20260926/deployment.yaml",
            "src",
            "configs",
            "pyproject.toml",
        }
    )


def _published_and_clean(
    commit: str, paths: list[str], repo: Path, *, require_main: bool = True
) -> None:
    """Require a clean pin on origin/main, or a published branch for pending preparation."""
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=repo, text=True
    )
    branches = {line.strip() for line in published.splitlines()}
    if require_main and "origin/main" not in branches:
        raise ValueError("source pin must already be published on origin/main")
    if not require_main and not any(branch.startswith("origin/") for branch in branches):
        raise ValueError("source pin must already be published on an origin branch")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", commit, "--", *paths], cwd=repo, text=True
    )
    if changed.strip():
        raise ValueError("working source differs from manifest pin: " + changed)


def _input(path: Path, dest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "dest": dest, "sha256": digest(path)}


def source_bound_transport_receipt(
    repo: Path = ROOT, receipt_path: Path = ACCEPTANCE_RESULT
) -> dict[str, str]:
    """Verify the current final-only acceptance without uploading its receipt."""
    from research.colab_cli.acceptance import verify_plan_sources
    from research.colab_cli.admission import require_final_only

    receipt_path = receipt_path.resolve()
    result = json.loads(receipt_path.read_text())
    require_final_only(result, repo)
    verify_plan_sources(
        {"source_manifest": result["source_manifest"], "commit": result["final_plan"]["commit"]},
        repo,
    )
    return {
        "path": str(receipt_path),
        "sha256": digest(receipt_path),
        "status": "PASS_FINAL_ONLY",
    }


def build(
    commit: str,
    output_dir: Path,
    *,
    phase: str = "production",
    repo: Path = ROOT,
    pending_requalification: bool = False,
    source_qualification_receipt: Path = ACCEPTANCE_RESULT,
) -> dict[str, Any]:
    """Create a source-pinned, explicitly non-admitted DENSITY-2 template.

    Default preparation requires the current accepted final-only receipt. The
    explicit pending mode is limited to a published branch pin, is never
    scientifically admitted, and records that live requalification remains due.
    """
    if phase not in {"canary", "production"}:
        raise ValueError("phase must be canary or production")
    is_probe = phase == "canary"
    paths = source_paths(repo)
    _published_and_clean(commit, paths, repo, require_main=not pending_requalification)
    catalog = repo / "research/t34_exit_b8/PUBLIC_INPUTS.json"
    metric_sha = digest(repo / PACKAGE / "METRICS.md")
    amendment_sha = digest(repo / PACKAGE / "METRICS_AMENDMENT_01.md")
    capacity_sha = digest(repo / PACKAGE / "METRICS_AMENDMENT_02.md")
    round14_sha = digest(repo / PACKAGE / "METRICS_AMENDMENT_03.md")
    replication_sha = digest(repo / PACKAGE / "METRICS_AMENDMENT_04.md")
    manifest: dict[str, Any] = {
        "job_id": "bl-density2-" + phase + "-" + commit[:12],
        "commit": commit,
        "script": PACKAGE + "/colab.sh",
        "source_paths": paths,
        "claim": "DEVELOPMENT_DENSITY_MEASUREMENT_ZERO_SCIENCE_CLAIMS",
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": 7.5 if is_probe else 75,
        },
        "timeout_minutes": CANARY_TIMEOUT if is_probe else PRODUCTION_TIMEOUT,
        "output_dir": str(output_dir.resolve()),
        "budget": {
            "unit_cap": CANARY_CAP if is_probe else PRODUCTION_CAP,
            "rate_units_per_hour": L4_RATE,
            "reserve_units": CANARY_RESERVE if is_probe else PRODUCTION_RESERVE,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": REQUIRED_EXPORTS,
        "scientific_admission": False,
        "inputs": [_input(catalog, "public-inputs.json")],
        "env": {
            "PYTHONUNBUFFERED": "1",
            "SNAKE_DQN_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "BL_PUBLIC_INPUTS": "1",
            "BL_DENSITY_COMMIT": commit,
            "BL_DENSITY_PHASE": phase,
            "BL_DENSITY_SCIENTIFIC_ADMISSION": "0",
            "BL_DENSITY_ROUND14_AMENDMENT_SHA256": round14_sha,
            "BL_DENSITY_METRICS_SHA256": metric_sha,
            "BL_DENSITY_AMENDMENT_SHA256": amendment_sha,
            "BL_DENSITY_CAPACITY_AMENDMENT_SHA256": capacity_sha,
            "BL_DENSITY_REPLICATION_AMENDMENT_SHA256": replication_sha,
            "COLAB_CLI_REQUIREMENTS": PACKAGE + "/requirements.txt",
        },
        "density": {
            "status": "NOT_READY",
            "scientific_admission": False,
            "phase": phase,
            "metric_sha256": metric_sha,
            "amendment_sha256": amendment_sha,
            "capacity_amendment_sha256": capacity_sha,
            "round14_amendment_sha256": round14_sha,
            "replication_amendment_sha256": replication_sha,
            "sample_stride": 200,
            "episodes_per_mix_by_arm": (
                {arm: 1 for arm in ("A", "B70m", "B70p", "C-P1")}
                if is_probe
                else {
                    "A": {"frozen": 96, "mixed": 72, "scripted": 72},
                    "B70m": {mix: 24 for mix in ("frozen", "mixed", "scripted")},
                    "B70p": {mix: 36 for mix in ("frozen", "mixed", "scripted")},
                    "C-P1": {mix: 24 for mix in ("frozen", "mixed", "scripted")},
                }
            ),
            "total_episodes": 12 if is_probe else 492,
            "workers": 1 if is_probe else 10,
            "horizon": 8 if is_probe else 5000,
            "work_wall_seconds": 600 if is_probe else 10800,
            "namespace": (
                "bl-density-round14b-canary-20261022"
                if is_probe
                else "bl-density-round14b-20261022"
            ),
            "one_terminal_look": not is_probe,
            "pilot_excluded": True,
            "production_gate": "NOT_READY_QUALIFICATION_AND_CANARY_REQUIRED",
        },
    }
    if is_probe:
        # FAMILIES is only a helper registry. Do not mutate it; unchanged broker
        # checks evaluate this real script, family, inputs, exports and device evidence.
        manifest["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
        manifest["canary"] = {
            "schema": family_canary.SCHEMA,
            "mode": "probe",
            "family": family_canary.family(manifest),
        }
    manifest["source_qualification"] = (
        {
            "status": "PENDING_REQUALIFICATION",
            "receipt_path": str(source_qualification_receipt.resolve()),
            "reason": "current source requires a fresh final-only acceptance",
        }
        if pending_requalification
        else source_bound_transport_receipt(repo, source_qualification_receipt)
    )
    manifest["density"]["source_qualification_status"] = manifest["source_qualification"]["status"]
    assert_upload_budget(manifest, repo=repo)
    validate_manifest(manifest)
    return manifest


def validation_report(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Record exact schema, qualification and broker outcomes without bypassing either."""
    report: dict[str, Any] = {
        "schema": "bl-density-validation-report/v2",
        "job_id": manifest.get("job_id"),
        "scientific_admission": manifest.get("scientific_admission") is True,
        "status": manifest.get("density", {}).get("status", "NOT_READY"),
    }
    try:
        validate_manifest(manifest)
        assert_upload_budget(manifest, repo=repo)
    except Exception as exc:
        report["schema_and_upload"] = {"status": "REFUSED", "reason": str(exc)}
    else:
        report["schema_and_upload"] = {"status": "PASS"}
    qualification = manifest.get("source_qualification", {})
    if qualification.get("status") == "PENDING_REQUALIFICATION":
        report["source_qualification"] = dict(qualification)
    else:
        try:
            report["source_qualification"] = {
                "status": "PASS",
                **source_bound_transport_receipt(
                    repo, Path(qualification.get("path", ACCEPTANCE_RESULT))
                ),
            }
        except Exception as exc:
            report["source_qualification"] = {"status": "REFUSED", "reason": str(exc)}
    try:
        validate(copy.deepcopy(manifest), repo=repo)
    except (Refused, ValueError) as exc:
        report["supervisor"] = {"status": "PENDING_OR_REFUSED", "reason": str(exc)}
    else:
        report["supervisor"] = {"status": "PASS"}
    return report


def finalize_production(
    template: dict[str, Any], *, canary_proof: Path, canary_receipt: Path, prior_spend_units: float
) -> dict[str, Any]:
    """Bind recovered proof/receipt while retaining one <=5-unit envelope."""
    if template.get("env", {}).get("BL_DENSITY_PHASE") != "production":
        raise ValueError("production template required")
    if template.get("scientific_admission") is not False:
        raise ValueError("only a non-admitted template may be finalized")
    if not isinstance(prior_spend_units, (int, float)) or not 0 <= prior_spend_units <= CANARY_CAP:
        raise ValueError("prior_spend_units must be the bounded canary charge")
    value = copy.deepcopy(template)
    proof = verify_proof(
        canary_proof, value["commit"], value["density"]["round14_amendment_sha256"]
    )
    if not canary_receipt.is_file():
        raise ValueError("recovered family canary receipt required")
    if proof.get("receipt_sha256") != digest(canary_receipt):
        raise ValueError("canary proof does not bind the supplied receipt")
    receipt = json.loads(canary_receipt.read_text())
    elapsed = receipt.get("elapsed_seconds")
    rate = (receipt.get("accounting") or {}).get("priced_rate_units_per_hour")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise ValueError("canary receipt lacks finite elapsed pricing evidence")
    measured_charge = float(elapsed) * float(rate) / 3600
    if float(prior_spend_units) + 1e-12 < measured_charge:
        raise ValueError("prior_spend_units understates the recovered canary charge")
    effective_cap = PRODUCTION_CAP - PRODUCTION_RESERVE - float(prior_spend_units)
    if effective_cap <= 0:
        raise ValueError("no production units remain after reserve and prior charge")
    priced_timeout = min(PRODUCTION_TIMEOUT, int(60 * effective_cap / L4_RATE))
    if 2 * value["requirements"]["est_minutes"] > priced_timeout:
        raise ValueError(
            "remaining priced cap cannot honor the fixed 75-minute production estimate"
        )
    value["budget"]["unit_cap"] = effective_cap
    value["timeout_minutes"] = priced_timeout
    value["canary"] = {
        "receipt": {"path": str(canary_receipt.resolve()), "sha256": digest(canary_receipt)}
    }
    # Environment binding keeps the actual uploaded input set identical to the canary.
    value["env"]["BL_DENSITY_CANARY_PROOF_JSON"] = json.dumps(
        proof, sort_keys=True, separators=(",", ":")
    )
    value["density"].update(
        status="NOT_READY",
        prior_spend_units=float(prior_spend_units),
        effective_production_unit_cap=effective_cap,
        total_envelope_unit_cap=PRODUCTION_CAP,
        production_gate="NOT_READY_QUALIFICATION_AND_SUPERVISOR_ADMISSION_REQUIRED",
    )
    return value


def admit(
    template: dict[str, Any],
    *,
    source_qualification_receipt: Path = ACCEPTANCE_RESULT,
    repo: Path = ROOT,
    canary_proof: Path | None = None,
    canary_receipt: Path | None = None,
    prior_spend_units: float | None = None,
) -> dict[str, Any]:
    """Admit a previously prepared template only after current real evidence.

    This is deliberately offline: it checks receipts and the supervisor's
    preflight validator but never allocates a runtime. A production template
    must first be finalized against recovered canary evidence and its remaining
    budget before the same source and supervisor checks can admit it.
    """
    if template.get("scientific_admission") is not False:
        raise ValueError("only a non-admitted template may be admitted")
    phase = template.get("env", {}).get("BL_DENSITY_PHASE")
    if phase not in {"canary", "production"}:
        raise ValueError("DENSITY-2 canary or production template required")
    if template.get("env", {}).get("BL_DENSITY_SCIENTIFIC_ADMISSION") != "0":
        raise ValueError("template must retain the disabled scientific admission guard")
    paths = list(template.get("source_paths", []))
    _published_and_clean(str(template.get("commit", "")), paths, repo, require_main=True)
    value = copy.deepcopy(template)
    if phase == "production":
        if canary_proof is None or canary_receipt is None or prior_spend_units is None:
            raise ValueError(
                "production admission requires recovered canary proof, receipt and charge"
            )
        value = finalize_production(
            value,
            canary_proof=canary_proof,
            canary_receipt=canary_receipt,
            prior_spend_units=prior_spend_units,
        )
    elif any(item is not None for item in (canary_proof, canary_receipt, prior_spend_units)):
        raise ValueError("canary admission does not accept production canary evidence")
    qualification = source_bound_transport_receipt(repo, source_qualification_receipt)
    value["source_qualification"] = qualification
    value["density"]["source_qualification_status"] = qualification["status"]
    value["scientific_admission"] = True
    value["density"]["scientific_admission"] = True
    value["density"]["status"] = "ADMITTED_OFFLINE_PENDING_DISPATCH"
    value["density"]["production_gate"] = (
        "FAMILY_CANARY_RUN_REQUIRED" if phase == "canary" else "ADMITTED_OFFLINE_PENDING_DISPATCH"
    )
    value["env"]["BL_DENSITY_SCIENTIFIC_ADMISSION"] = "1"
    assert_upload_budget(value, repo=repo)
    validate_manifest(value)
    validate(value, repo=repo)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--phase", choices=("canary", "production"), default="production")
    parser.add_argument("--pending-requalification", action="store_true")
    parser.add_argument("--source-qualification-receipt", type=Path, default=ACCEPTANCE_RESULT)
    parser.add_argument("--admit-template", type=Path)
    parser.add_argument("--canary-proof", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--prior-spend-units", type=float)
    args = parser.parse_args()
    if args.admit_template:
        if args.pending_requalification:
            parser.error("--pending-requalification cannot admit a template")
        template = json.loads(args.admit_template.read_text())
        manifest = admit(
            template,
            source_qualification_receipt=args.source_qualification_receipt,
            canary_proof=args.canary_proof,
            canary_receipt=args.canary_receipt,
            prior_spend_units=args.prior_spend_units,
        )
    else:
        if not args.commit or args.output_dir is None:
            parser.error("--commit and --output-dir are required when preparing a template")
        manifest = build(
            args.commit,
            args.output_dir,
            phase=args.phase,
            pending_requalification=args.pending_requalification,
            source_qualification_receipt=args.source_qualification_receipt,
        )
    export_json(args.manifest, manifest)
    if args.validation_report:
        export_json(args.validation_report, validation_report(manifest))
    print(json.dumps({"manifest": str(args.manifest), "status": manifest["density"]["status"]}))


if __name__ == "__main__":
    main()
