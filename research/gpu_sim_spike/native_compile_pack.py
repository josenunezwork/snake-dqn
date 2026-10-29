"""Prepare a cheap L4 compiler-only job; never allocate or launch a workload."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from research.gpu_sim_spike import a1_prof_pack
from research.gpu_sim_spike.make_manifests import sha256_file, write_once

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"


def manifest(commit: str, artifact: Path) -> dict:
    """Price a bounded operational compile check with explicit A100/L4/H100 targets."""
    requirements = ROOT / PACKAGE / "native_compile_requirements.txt"
    return {
        "job_id": f"native-compile-l4-{commit[:8]}",
        "commit": commit,
        "script": f"{PACKAGE}/native_compile.sh",
        "device": "cpu",  # nvcc/linker only; no CUDA kernel execution or action decisions.
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/native_compile_requirements.txt",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": PACKAGE,
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 2, "est_minutes": 5},
        "timeout_minutes": 20,
        "budget": {
            "unit_cap": 1.2,
            "reserve_units": 0.2,
            "rate_units_per_hour": 3.0,
            "other_assignments": {},
        },
        "compile_envelope": {
            "base": {"timeout_minutes": 10, "unit_cap": 0.6, "reserve_units": 0.1},
            "effective": {"timeout_minutes": 20, "unit_cap": 1.2, "reserve_units": 0.2},
            "estimate_basis": "provisional 5-minute build forecast; not measured on this pin",
            "scope": "compile only; charge within remaining sprint allowance; no automatic retries",
        },
        "output_dir": str(artifact / "output"),
        "inputs": [
            {
                "path": str(requirements),
                "sha256": sha256_file(requirements),
                "dest": "compile-requirements.txt",
            }
        ],
        "source_paths": [
            "src",
            PACKAGE,
            "research/colab_cli",
            "requirements.txt",
            "pyproject.toml",
        ],
        "required_exports": ["native-compile.json"],
        "max_export_bytes": 1024 * 1024,
        "lease_seconds": 600,
        "resume_exemption": {
            "kind": "short-canary",
            "reason": "Bounded host compile/link only; no outcomes, gameplay or trajectories",
        },
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": f"{PACKAGE}/native_compile.sh",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    a1_prof_pack.require_source_pin(args.commit)
    artifact = args.artifact_dir.resolve()
    value = manifest(args.commit, artifact)
    from research.colab_cli.supervisor import validate
    from tools.ops.dispatch import require_resume_policy

    require_resume_policy(value)
    validate(value, repo=ROOT)
    artifact.mkdir(parents=True, exist_ok=False)
    write_once(artifact / "native-compile.json", json.dumps(value, indent=2, sort_keys=True) + "\n")
    command = shlex.join(
        [
            str(ROOT / "tools/ops/dispatch.sh"),
            value["job_id"],
            str(artifact / "native-compile.json"),
            "--broker",
            str(ROOT),
            "--cwd",
            str(ROOT),
            "--state-dir",
            str(artifact / "state"),
            "--dispatch-txt",
            str(artifact / "DISPATCH.txt"),
        ]
    )
    write_once(
        artifact / "DISPATCH.txt",
        "# Compiler-only preflight; no workload decisions.\n" + command + "\n",
    )
    print(command)


if __name__ == "__main__":
    main()
