"""Prepare one source-pinned density job; every operation here is offline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_density.canary import verify_proof
from research.bl_ppo.exports import export_json
from research.bl_prefix.make_manifest import qualified_transport
from research.bl_upload import assert_upload_budget
from research.colab_cli.supervisor import validate

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/bl_density"


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
            "research/colab_cli",
            "research/t34_exit_b8/PUBLIC_INPUTS.json",
            "research/apex_safety_20260926/deployment.yaml",
            "src",
            "configs",
            "pyproject.toml",
        }
    )


def build(
    commit: str,
    output_dir: Path,
    *,
    phase: str = "production",
    canary_proof: Path | None = None,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Bind exact transport bytes; unproven production remains runtime-blocked."""
    if phase not in {"canary", "production"}:
        raise ValueError("phase must be canary or production")
    if phase == "canary" and canary_proof is not None:
        raise ValueError("canary cannot consume its own proof")
    canary = phase == "canary"
    qualified_transport(repo, commit)
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=repo, text=True
    )
    if "origin/main" not in published:
        raise ValueError("source pin must already be published on origin/main")
    paths = source_paths(repo)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", commit, "--", *paths], cwd=repo, text=True
    )
    if changed.strip():
        raise ValueError("working source differs from manifest pin: " + changed)
    catalog = repo / "research/t34_exit_b8/PUBLIC_INPUTS.json"
    metric_sha = digest(repo / PACKAGE / "METRICS.md")
    amendment_sha = digest(repo / PACKAGE / "METRICS_AMENDMENT_01.md")
    capacity_sha = digest(repo / PACKAGE / "METRICS_AMENDMENT_02.md")
    round14_sha = digest(repo / PACKAGE / "METRICS_AMENDMENT_03.md")
    manifest = {
        "job_id": "bl-density-r14-" + phase + "-" + commit[:12],
        "commit": commit,
        "script": PACKAGE + "/colab.sh",
        "source_paths": paths,
        "claim": "DEVELOPMENT_DENSITY_MEASUREMENT_ZERO_SCIENCE_CLAIMS",
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": 10 if canary else 60,
        },
        "timeout_minutes": 20 if canary else 120,
        "output_dir": str(output_dir.resolve()),
        "budget": {
            "unit_cap": 1.0 if canary else 5.0,
            "rate_units_per_hour": 2.0,
            "reserve_units": 0.2 if canary else 1.0,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "inputs": [
            {
                "dest": "public-inputs.json",
                "path": str(catalog.resolve()),
                "sha256": digest(catalog),
            }
        ],
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
            "BL_DENSITY_ROUND14_AMENDMENT_SHA256": round14_sha,
            "BL_DENSITY_METRICS_SHA256": metric_sha,
            "BL_DENSITY_AMENDMENT_SHA256": amendment_sha,
            "BL_DENSITY_CAPACITY_AMENDMENT_SHA256": capacity_sha,
            "COLAB_CLI_REQUIREMENTS": PACKAGE + "/requirements.txt",
        },
        "density": {
            "status": "PREPARED_NOT_DISPATCHED",
            "metric_sha256": metric_sha,
            "amendment_sha256": amendment_sha,
            "capacity_amendment_sha256": capacity_sha,
            "round14_amendment_sha256": round14_sha,
            "phase": phase,
            "production_gate": "NOT_APPLICABLE" if canary else "BLOCKED_PENDING_L4_CANARY",
            "sample_stride": 200,
            "episodes_per_mix_by_arm": {
                "A": 1 if canary else 12,
                "B70m": 1 if canary else 6,
                "B70p": 1 if canary else 6,
                "C-P1": 1 if canary else 6,
            },
            "total_episodes": 12 if canary else 90,
            "workers": 1 if canary else 6,
            "work_wall_seconds": 600 if canary else 6000,
            "forecast": {
                "session_minutes": 10 if canary else 60,
                "planned_units": 1.0 / 3 if canary else 2.0,
                "reserve_units": 0.2 if canary else 1.0,
                "worst_case_units": 1.0 if canary else 5.0,
                "rate_status": "owner planning ceiling; no provider calls in preparation",
                "qualification": "tiny Mac preflight is not a measured L4 H5000 throughput",
                "stop": "first supervisor wall/unit limit; incomplete units never imply completion",
            },
        },
    }
    if canary_proof is not None:
        verify_proof(canary_proof, commit, round14_sha)
        for suffix in ("", ".sha256", ".complete"):
            path = Path(str(canary_proof) + suffix).resolve()
            manifest["inputs"].append(
                {"path": str(path), "dest": "canary-proof.json" + suffix, "sha256": digest(path)}
            )
        # Sidecar verifier binds the basename; require a stable transport name.
        if canary_proof.name != "canary-proof.json":
            raise ValueError("canary proof basename must be canary-proof.json")
        manifest["density"]["production_gate"] = "RECOVERED_L4_CANARY_VERIFIED"
    assert_upload_budget(manifest, repo=repo)
    return validate(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("canary", "production"), default="production")
    parser.add_argument("--canary-proof", type=Path)
    args = parser.parse_args()
    manifest = build(args.commit, args.output_dir, phase=args.phase, canary_proof=args.canary_proof)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    export_json(args.manifest, manifest)
    print(json.dumps({"manifest": str(args.manifest), "status": "VALIDATED_OFFLINE"}))


if __name__ == "__main__":
    main()
