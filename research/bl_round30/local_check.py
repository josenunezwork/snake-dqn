"""One bounded Mac preflight through the existing reviewed interim launcher."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

from research.bl_ppo.exports import export_json

ROOT = Path(__file__).resolve().parents[2]
PY = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"
LAUNCH = str(ROOT / "tools/ops/agent_nice.sh")
NODES = [
    "tests/test_bl_round30_admission.py",
    "tests/test_bl_round30_job.py",
    "tests/test_bl_round30_metrics.py",
    "tests/test_bl_round30_objective.py",
    "tests/test_bl_round30_pack.py",
    "tests/test_bl_round30_factory_check.py",
    "tests/test_colab_cli_canary.py::test_actual_scanner_transport_release_admits",
    "tests/test_colab_cli_canary.py::test_receipt_cannot_admit_an_uncovered_dispatch",
    "tests/test_colab_cli_canary.py::test_gpu_job_without_cuda_decision_refuses",
    "tests/test_colab_cli_canary.py::test_job_device_resolves_declarations",
    "tests/test_colab_cli_canary.py::test_gpu_decision_observation_is_accepted",
    "tests/test_colab_cli_canary.py::test_probe_retains_real_input_inventory_and_real_entrypoint",
    "tests/test_colab_cli_canary.py::test_broker_refuses_science_before_any_allocator_call",
    "tests/test_colab_cli_supervisor.py::test_existing_output_directory_is_refused_as_nonfresh",
]


def source_identity() -> dict[str, str]:
    """Bind package Python/shell/dependencies and the exact executed test sources."""
    paths = [
        p for p in (ROOT / "research/bl_round30").iterdir() if p.suffix in (".py", ".sh", ".txt")
    ]
    paths.extend(ROOT / n.split("::")[0] for n in NODES)
    paths.append(ROOT / "tests/conftest.py")
    for directory in (
        "src",
        "research/compute",
        "research/bl_residual",
        "research/colab_cli",
        "tools/ops",
    ):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in (".py", ".sh"))
    paths.extend(
        ROOT / p
        for p in (
            "research/bl_residual/fit.py",
            "research/bl_residual/model.py",
            "research/bl_residual/data.py",
            "research/bl_ppo/exports.py",
            "research/a100_a2/model.py",
            "research/mac_lease.py",
            "tools/ops/agent_nice.sh",
        )
    )
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
    }


def run(output: Path, factory_start: Path | None = None, factory_pid: int | None = None) -> int:
    """A refused lease remains HOLD; this never waits/retries or bypasses a fence."""
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        raise ValueError("preflight caller must already be nice >=15")
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()
    lines = ["schema=factory-companion-test-manifest/v2", f"source_commit={commit}"]
    lines += [f"fixture={p}|{sha}" for p, sha in identity.items()]
    lines += [f"node={node}" for node in NODES]
    manifest = output / "reviewed-check.txt"
    manifest.write_text("\n".join(lines) + "\n")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    command = [
        LAUNCH,
        "--class",
        "test",
        "--review-manifest",
        str(manifest),
        "--review-sha256",
        sha,
        "--",
        PY,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:xdist",
        "-p",
        "research.compute.factory_companion_pytest",
        *NODES,
    ]
    if (factory_start is None) != (factory_pid is None):
        raise ValueError("factory start and PID must be supplied together")
    if factory_start is not None:
        command = [
            PY,
            "-m",
            "research.bl_round30.factory_check",
            "--manifest",
            str(manifest),
            "--sha",
            sha,
            "--start",
            str(factory_start),
            "--pid",
            str(factory_pid),
            "--evidence",
            str(output / "factory-window.json"),
            "--",
            *command[8:],
        ]
    with (output / "check.log").open("x") as log:
        # The launcher owns its 120s including cleanup; this outer bound is only
        # a guard against a broken launcher, and never authorizes another check.
        error = None
        try:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=125, check=False
            )
            returncode = result.returncode
        except (subprocess.TimeoutExpired, OSError) as exc:
            error, returncode = str(exc), 124
    unchanged = identity == source_identity()
    export_json(
        output / "preflight.json",
        {
            "schema": "r30-mac-preflight/v1",
            "status": "PASS" if returncode == 0 and unchanged else "HOLD",
            "returncode": returncode,
            "error": error,
            "outer_timeout_does_not_prove_cleanup": error is not None,
            "source_identity": identity,
            "source_commit": commit,
            "test_manifest_sha256": sha,
            "log_sha256": hashlib.sha256((output / "check.log").read_bytes()).hexdigest(),
            "numerical_threads": 1,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "shared_lock_and_lease": (
                "host-wide serial lock; existing ordinary factory lease preserved"
                if factory_start
                else "existing factory_interim_launcher"
            ),
            "ordinary_factory_evidence": (
                str(output / "factory-window.json") if factory_start else None
            ),
            "max_elapsed_seconds": 120,
        },
    )
    return returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factory-start", type=Path)
    parser.add_argument("--factory-pid", type=int)
    args = parser.parse_args()
    raise SystemExit(run(args.output.resolve(), args.factory_start, args.factory_pid))


if __name__ == "__main__":
    main()
