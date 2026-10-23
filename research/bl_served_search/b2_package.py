"""Offline B2 family manifests: exactly one trajectory per qualified allocation.

Caps are ALREADY doubled by PLAN_v3. No provider calls; no dispatch; no automatic
retry. Missing frozen design, real Mac preflight or family evidence is a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from research.bl_served_search import b2
from research.bl_served_search import b2_admission as admission
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.manifests import source_paths
from research.colab_cli import canary
from research.colab_cli.schema import validate_manifest

SCRIPT = "research/bl_served_search/run_b2.sh"
RATE = 2.0
CANARY_CAP = 1.0
UNIT_CAP = 19.0 / 32  # 32 original allocations + canary <=20; one completion series <=40.
RESERVE = UNIT_CAP * 0.2
TIMEOUT = (UNIT_CAP - RESERVE) / RATE * 60


def qualified_cli(commit: str) -> dict:
    """Pin both the published tree and the Mac's current qualification record."""
    path = admission.QUALIFIED_RECORD
    record = json.loads(path.read_text())
    tree = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"], cwd=b2.ROOT, text=True, timeout=10
    ).strip()
    if record.get("tree_hash") != admission.QUALIFIED_TREE or tree != admission.QUALIFIED_TREE:
        raise ValueError("NOT_READY qualified Colab CLI tree differs")
    accepted = Path(record["acceptance_result"])
    if digest(accepted) != record["acceptance_result_sha256"]:
        raise ValueError("NOT_READY qualified CLI acceptance digest differs")
    if json.loads(accepted.read_text()).get("status") != "PASS":
        raise ValueError("NOT_READY qualified CLI acceptance is not PASS")
    return {"path": str(path), "sha256": digest(path), **record}


def input_row(path: Path, dest: str) -> dict:
    return {"path": str(path.resolve()), "dest": dest, "sha256": digest(path)}


def prepare(
    spec_path: Path,
    ledger_path: Path,
    checkpoints: Path,
    preflight_path: Path,
    out: Path,
    backup: Path,
    commit: str,
    *,
    probe: bool,
    estimate_minutes: float,
    other_assignments: dict,
    canary_receipt: Path | None = None,
) -> list[dict]:
    """Prepare a canary or 32 frozen-index manifests; all execution stays with Claude."""
    spec = b2.validate_spec(b2.read_export(spec_path))
    b2.check(spec_path, ledger_path, checkpoints)
    preflight = b2.read_export(preflight_path)
    if preflight.get("status") != "PASS_TINY_ONLY" or preflight.get("real_units") != 3:
        raise ValueError("NOT_READY real Mac entrypoint preflight required")
    if preflight.get("sources") != source_identity():
        raise ValueError("NOT_READY Mac preflight predates current B2 code")
    backup = b2.drive_backup_root(backup)
    if not isinstance(other_assignments, dict):
        raise ValueError("absolute Drive-for-desktop backup and fresh assignment-rate map required")
    # Do not allow the shell payload to be pinned to an unpublished/local variant.
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=b2.ROOT, text=True, timeout=10
    )
    if "origin/main" not in refs or len(commit) != 40:
        raise ValueError("B2 pin must be a full published origin/main commit")
    qualification = qualified_cli(commit)
    paths = sorted(set(source_paths()) | {SCRIPT, "research/apex_safety_20260926/deployment.yaml"})
    changed = subprocess.check_output(
        ["git", "diff", commit, "--name-only", "--", *paths], cwd=b2.ROOT, text=True, timeout=20
    )
    if changed.strip():
        raise ValueError("B2 local runtime source differs from published pin")
    timeout, cap, reserve = (24.0, 1.0, 0.2) if probe else (TIMEOUT, UNIT_CAP, RESERVE)
    if not 0 < estimate_minutes <= timeout / 2:
        raise ValueError("whole-session measured estimate must fit half the frozen timeout")
    if spec["unit_timeout_seconds"] > (timeout - 3) * 60:
        raise ValueError("B2 unit timeout must leave >=3 minutes for setup/export/release")
    if not probe and canary_receipt is None:
        raise ValueError("NOT_READY same-pin real-transport family canary required")
    admitted = None
    if not probe:
        admitted = admission.evaluate(spec_path, canary_receipt)
        if admitted["status"] != "ADMIT_UNCHANGED" or admitted["commit"] != commit:
            raise ValueError("NOT_READY round-16 canary admission required; rescale/judge first")
        if not math.isclose(estimate_minutes, admitted["estimate_minutes"], abs_tol=1e-9):
            raise ValueError("NOT_READY estimate must equal the measured canary admission")
    inputs = [
        input_row(spec_path, "b2-design.json"),
        input_row(ledger_path, "verified-ledger.json"),
    ]
    for path in (spec_path, ledger_path):
        for suffix in (".sha256", ".complete"):
            inputs.append(input_row(Path(str(path) + suffix), path.name + suffix))
    # The payload's portable names must be identical to their sidecar names.
    if spec_path.name != "b2-design.json" or ledger_path.name != "verified-ledger.json":
        raise ValueError("use portable b2-design.json and verified-ledger.json basenames")
    audit = spec_path.parent / "freshness-audit.json"
    b2.validate_freshness(spec, audit)
    for suffix in ("", ".sha256", ".complete"):
        inputs.append(input_row(Path(str(audit) + suffix), audit.name + suffix))
    inputs += [input_row(checkpoints / name, name) for name in (b2.B8_FILE, b2.FRP_FILE)]
    done = b2.units.validate_ledger(
        b2.read_export(ledger_path), spec["namespace"], spec["allocation_digest"]
    )
    indices = [i for i in range(32) if b2.units.unit_id(spec["namespace"], i) not in done]
    if not indices:
        raise ValueError("B2 allocation already verified; do not dispatch another runtime")
    manifests = []
    for index in ([0] if probe else indices):
        token = hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16]
        mode = "canary" if probe else f"trajectory-{index:02d}"
        manifest = {
            "job_id": f"bl-b2-{token}-{mode}",
            "commit": commit,
            "script": SCRIPT,
            "source_paths": paths,
            "requirements": {
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": spec["workers"] + 1,
                "est_minutes": estimate_minutes,
            },
            "timeout_minutes": timeout,
            "output_dir": str(out.resolve() / mode),
            "backup_dir": str(backup),
            "max_export_bytes": 512 * 1024**2,
            "required_exports": ["unit-status.json"] + (["canary-admission.json"] if probe else []),
            "lease_seconds": 300,
            "budget": {
                "unit_cap": cap,
                "reserve_units": reserve,
                "rate_units_per_hour": RATE,
                "other_assignments": other_assignments,
            },
            "resume": {
                "unit_kind": "trajectory",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
            "env": {
                "BL_B2_INDEX": str(index),
                "BL_B2_CANARY": "1" if probe else "0",
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "SNAKE_DQN_DEVICE": "cpu",
                "PYTHONUNBUFFERED": "1",
                **dict.fromkeys(b2.engine.ceiling.THREAD_ENV, "1"),
            },
            "inputs": inputs,
            "bl_b2": {
                "status": "PREPARED_NOT_DISPATCHED",
                "allocation_digest": spec["allocation_digest"],
                "forecast_phase_units": 20,
                "effective_series_cap": 40,
                "caps_already_doubled": True,
                "max_completion_attempts": 1,
                "loss_bound": "one trajectory per dispatch; verify Drive backup before next unit",
                "preflight_sha256": verify_export(preflight_path),
                "estimate_scope": "whole allocation including setup, export and release",
                "qualified_cli": qualification,
                "admission": admitted,
            },
        }
        if probe:
            manifest["canary"] = {"schema": canary.SCHEMA, "mode": "probe", "family": SCRIPT}
            manifest["env"].update(
                {
                    "SNAKE_COLAB_CANARY": "1",
                    "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_served_search",
                }
            )
        else:
            manifest["canary"] = {
                "receipt": {"path": str(canary_receipt.resolve()), "sha256": digest(canary_receipt)}
            }
        validate_manifest(manifest)
        canary.check(manifest, b2.ROOT)
        manifests.append(manifest)
    out.mkdir(parents=True, exist_ok=False)
    for manifest in manifests:
        export_json(out / (manifest["job_id"] + ".json"), manifest)
    return manifests


def source_identity() -> dict[str, str]:
    """Preflight binds its real payload and math dependency closure."""
    # Source paths includes source directories; expand them deterministically.
    result = {}
    paths = sorted(set(source_paths()) | {SCRIPT, "research/apex_safety_20260926/deployment.yaml"})
    for name in paths:
        path = b2.ROOT / name
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for item in candidates:
            if item.is_file() and item.suffix in (".py", ".sh", ".yaml", ".toml", ".txt"):
                result[str(item.relative_to(b2.ROOT))] = digest(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--estimate-minutes", required=True, type=float)
    parser.add_argument("--other-assignments", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path)
    args = parser.parse_args()
    manifests = prepare(
        args.spec,
        args.ledger,
        args.inputs,
        args.preflight,
        args.out,
        args.backup,
        args.commit,
        probe=args.probe,
        estimate_minutes=args.estimate_minutes,
        other_assignments=json.loads(args.other_assignments.read_text()),
        canary_receipt=args.canary_receipt,
    )
    print(json.dumps({"status": "PREPARED_NOT_DISPATCHED", "count": len(manifests)}))


if __name__ == "__main__":
    main()
