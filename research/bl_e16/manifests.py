"""Offline-only E16 Colab manifest preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_e16.admission import forecast
from research.colab_cli.supervisor import Refused, validate

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/bl_e16/colab.sh"
RATE = 2.0
ROSTER_DESTINATIONS = (
    "checkpoints/champion_a5_freespace_20260621.pth",
    "checkpoints/best_apex_fs.pth",
    "checkpoints/best_apex_stage1_fs.pth",
    "checkpoints/best_apex_pre_fs.pth",
    "checkpoints/frp3_m3_s12_u60000_20261005.pth",
)


class ManifestRefused(ValueError):
    """The concrete E16 inputs do not satisfy its frozen admission boundary."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_head(repo: Path = ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _input(path: Path, dest: str) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ManifestRefused(f"input must be a regular file: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "dest": dest}


def _admission(parent: Path | None, bank: Path | None, audit: Path | None) -> dict[str, Any]:
    available = {
        "e2_f_parent": bool(parent and parent.is_file()),
        "evaluation_bank": bool(bank and bank.is_file()),
        "collision_seed_audit": bool(audit and audit.is_file()),
    }
    missing = [name for name, value in available.items() if not value]
    return {
        "schema": "bl-e16-input-admission/v1",
        "status": "NOT_DISPATCHABLE",
        "available": available,
        "missing": missing
        + ["hash-bound E8/E12/E3/E14/E4/episode-rate evidence and execution plan"],
        "required_preconditions": [
            "E2 nominated F endpoint and E12 critic PASS",
            "fresh collision-audited 96-world-per-mix evaluation bank",
            "frozen family-alpha admission contract",
        ],
    }


def make_manifest(
    *,
    commit: str,
    output_dir: Path,
    arm: str,
    b8_checkpoint: Path,
    roster_checkpoints: tuple[Path, ...],
    parent_checkpoint: Path | None = None,
    evaluation_bank: Path | None = None,
    seed_audit: Path | None = None,
) -> dict[str, Any]:
    """Create a structural template that cannot dispatch until real inputs arrive."""
    if arm not in {"A", "G", "R", "S", "P"}:
        raise ManifestRefused(f"unknown E16 arm: {arm}")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase SHA-1")
    if len(roster_checkpoints) != 5:
        raise ManifestRefused("E16 requires exactly five frozen roster checkpoints")
    admission = _admission(parent_checkpoint, evaluation_bank, seed_audit)
    inputs = [
        _input(b8_checkpoint, "B8_s3_final.pth"),
        *[_input(p, d) for p, d in zip(roster_checkpoints, ROSTER_DESTINATIONS)],
    ]
    for path, dest in (
        (parent_checkpoint, "parent/F.pth"),
        (evaluation_bank, "bank/BANK.json"),
        (seed_audit, "bank/SEED_AUDIT.json"),
    ):
        if path and path.is_file():
            inputs.append(_input(path, dest))
            for suffix in (".complete", ".sha256"):
                sidecar = Path(str(path) + suffix)
                if not sidecar.is_file():
                    raise ManifestRefused(f"input requires immutable sidecar: {sidecar}")
                inputs.append(_input(sidecar, dest + suffix))
    manifest = {
        "job_id": f"bl-e16-{arm.lower()}-blocked-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "E16_PREPARATION_ONLY_NOT_DISPATCHED",
        "transport": {"mode": "local_supervisor_upload", "dispatch_ready": False},
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 3},
        "timeout_minutes": 6,
        "output_dir": str(output_dir.expanduser().resolve()),
        "budget": {
            "unit_cap": 20.0,
            "rate_units_per_hour": RATE,
            "reserve_units": 4.0,
            "other_assignments": {},
        },
        "lease_seconds": 180,
        "max_export_bytes": 1024 * 1024 * 1024,
        "inputs": inputs,
        "source_paths": [
            "src",
            "configs",
            "research/bl_e16",
            "research/bl_ppo",
            "research/bl_info_value",
            "research/colab_cli",
            "requirements.txt",
            "pyproject.toml",
        ],
        "env": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMBA_NUM_THREADS": "2",
            "BL_E16_ARM": arm,
            "BL_E16_PLAN_STATUS": admission["status"],
            "BL_E16_CRITIC_TRANSITIONS": "2000000",
            "BL_E16_ACTOR_TRANSITIONS": "4000000",
            "BL_E16_HORIZON": "5000",
        },
        "e16": {
            "schema": "bl-e16-manifest/v1",
            "purpose": "BLOCKED_TEMPLATE_SCHEMA_VALIDATION_ONLY",
            "estimate_scope": "3 minutes to refuse absent prerequisites, NOT a training estimate",
            "forecast_rate_status": "planning assumptions, not E16 L4 measurements",
            "input_admission": admission,
            "forecast": forecast(2000.0, 800.0),
            "arm_precondition": {
                "A": "E2 F + E12 PASS",
                "G": "E2 F + E12 PASS + E3 GO + E14 pass",
                "R": "U >= 40 + TRANSFER_ADMISSIBLE + tau <= 1",
                "S": "E1 CONTINUE",
                "P": "optional PLR whole-world replay approval",
            }[arm],
        },
    }
    from research.bl_ppo.manifests import _base, _bounded_source_paths

    closure = _base(
        job_id="closure", commit=commit, output=output_dir, estimate=3, timeout=6, cap=20
    )["source_paths"]
    manifest["source_paths"] = sorted(set(closure + manifest["source_paths"]))
    manifest["source_paths"] = _bounded_source_paths(manifest)
    return manifest


def validate_offline(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Schema-validate a blocked template locally; this does not admit execution.

    The unchanged supervisor validator makes no Colab calls. Its input/schema
    success is distinct from require_execution_plan scientific admission.
    """
    if manifest.get("transport", {}).get("dispatch_ready") is not False:
        raise ManifestRefused("E16 templates must remain non-dispatchable")
    try:
        return validate(manifest, repo=repo)
    except Refused as exc:
        raise ManifestRefused(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an undispatched E16 manifest")
    parser.add_argument("--commit", default=git_head())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "G", "R", "S", "P"), required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--roster", type=Path, action="append", required=True)
    a = parser.parse_args()
    print(
        json.dumps(
            make_manifest(
                commit=a.commit,
                output_dir=a.output_dir,
                arm=a.arm,
                b8_checkpoint=a.b8,
                roster_checkpoints=tuple(a.roster),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
