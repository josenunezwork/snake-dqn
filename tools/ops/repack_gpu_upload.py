"""Repair upload inventories at an unchanged GPU-sprint runtime pin; never dispatch.

Run this file directly with ``prepare``. The create-only package includes a copy
of this helper so its later compile-receipt binding cannot drift with main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGE = "research/gpu_sim_spike"
SOURCE_SUFFIXES = {".py", ".sh", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".txt"}
# Import-only closure of the unchanged eager package initializer and publication
# helper. Production native loaders are parsed by inventory(), never imported.
COMPILE_IMPORTS = (
    "research/gpu_sim_spike/__init__.py",
    "research/gpu_sim_spike/a1_prof_units.py",
    "research/gpu_sim_spike/device_sim.py",
    "research/gpu_sim_spike/sparse_units.py",
    "research/gpu_sim_spike/stage_timing.py",
    "src/__init__.py",
    "src/core/__init__.py",
    "src/core/config_loader.py",
    "src/core/game_config.py",
    "src/core/mechanics_constants.py",
    "src/core/reward_events.py",
    "src/core/runtime_contract.py",
    "src/model/__init__.py",
    "src/model/apex_network.py",
    "src/model/base_network.py",
    "src/model/checkpoint_io.py",
    "src/model/checkpoint_manager.py",
    "src/model/obs_spec.py",
    "src/simd_env/__init__.py",
    "src/simd_env/batch_sim.py",
    "src/simd_env/featurizer.py",
    "src/simd_env/rng.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: Any) -> None:
    """Publish new package controls without replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x") as stream:
        stream.write(text)
    with Path(str(path) + ".sha256").open("x") as stream:
        stream.write(f"{digest(path)}  {path.name}\n")
    with Path(str(path) + ".complete").open("x") as stream:
        stream.write("complete\n")


def verified_json(path: Path) -> dict[str, Any]:
    if Path(str(path) + ".complete").read_text() != "complete\n" or Path(
        str(path) + ".sha256"
    ).read_text().split()[0] != digest(path):
        raise ValueError(f"incomplete or changed package control: {path}")
    return json.loads(path.read_text())


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo, timeout=30)


def compile_sources(repo: Path, commit: str) -> list[str]:
    from research.gpu_sim_spike.native_compile import inventory

    hashes, _ = inventory(repo, commit)
    return sorted(
        set(hashes)
        | set(COMPILE_IMPORTS)
        | {"requirements.txt", "pyproject.toml", f"{PACKAGE}/requirements.txt"}
    )


def sparse_sources(repo: Path, original: dict[str, Any]) -> list[str]:
    """Keep runtime code/configs and the entire spike source-hash inventory.

    Keep the prior conservative import directories, expanded into exact tracked
    files. Evidence JSON, logs, checkpoints, and large fixture banks are excluded.
    The saved exactness NPZ remains an explicit SHA-bound input, not source.
    """
    tracked = git(repo, "ls-tree", "-r", "--name-only", original["commit"]).decode().splitlines()
    roots = original["source_paths"]
    return sorted(
        name
        for name in tracked
        if any(name == root or name.startswith(root + "/") for root in roots)
        and (
            Path(name).suffix in (SOURCE_SUFFIXES - {".txt"}) | {".yaml", ".toml"}
            or (
                Path(name).suffix == ".txt"
                and (name.startswith(PACKAGE + "/") or Path(name).name.startswith("requirements"))
            )
            or name == "research/t34_exit_b8/PUBLIC_INPUTS.json"
        )
    )


def check_upload(manifest: dict[str, Any], broker: Path, stage: Path) -> dict[str, Any]:
    """Run qualified validate AND the exact pre-allocation staging size check."""
    from research.bl_upload import qualify
    from research.colab_cli.source import prepare
    from research.colab_cli.supervisor import validate
    from tools.ops.dispatch import require_resume_policy

    require_resume_policy(manifest)
    value = validate(manifest, repo=broker)
    plan = prepare(broker, value, stage)
    for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
        (stage / name).write_bytes(
            git(broker, "show", f"{value['commit']}:research/colab_cli/{name}")
        )
    rows = [{"path": str(p), "bytes": p.stat().st_size} for p in sorted(stage.iterdir())]
    rows += [
        {"path": i["path"], "bytes": Path(i["path"]).stat().st_size}
        for i in value["inputs"]
        if "path" in i
    ]
    for row in rows:
        if row["bytes"] > value["max_export_bytes"]:
            raise ValueError(f"UPLOAD_CAP: {row}")
    return {
        "status": "PASS_NOT_DISPATCHED",
        "job_id": value["job_id"],
        "commit": value["commit"],
        "manifest_sha256": hashlib.sha256(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "per_file_bound_bytes": value["max_export_bytes"],
        "largest_upload_bytes": max(row["bytes"] for row in rows),
        "source_files": len(plan["files"]),
        "source_archive_bytes": plan["fallback_size"],
        "source_unpacked_bytes": plan["fallback_unpacked_size"],
        "uploads": rows,
        "time_budget": qualify(value, repo=broker),
    }


def prepare_package(source: Path, artifact: Path, broker: Path, attempt: str) -> None:
    from research.gpu_sim_spike.sparse_pack import _preflight_binding
    from tools.ops.qualification import OPS, check

    compile_job = verified_json(source / "native-compile/native-compile.json")
    sparse_job = verified_json(source / "sparse-bench-canary-pending-compile.json")
    pin = compile_job["commit"]
    if sparse_job["commit"] != pin or git(broker, "rev-parse", "HEAD").decode().strip() != pin:
        raise ValueError("compile, sparse and frozen broker pins must match")
    check(broker, OPS / "colab-cli-qualified.json")
    check(broker, OPS / "colab-cli-qualified.json", ref=pin)
    if "origin/" not in git(broker, "branch", "-r", "--contains", pin).decode():
        raise ValueError("runtime pin must be published")
    if (
        _preflight_binding(Path(sparse_job["mac_preflight"]["path"]), pin)
        != sparse_job["mac_preflight"]
    ):
        raise ValueError("original Mac preflight binding differs")
    artifact.mkdir(parents=True, exist_ok=False)
    compile_job["source_paths"] = compile_sources(broker, pin)
    sparse_job["source_paths"] = sparse_sources(broker, sparse_job)
    for job, output in ((compile_job, "native-compile/output"), (sparse_job, "canary-output")):
        job["job_id"] += "-" + attempt
        job["output_dir"] = str(artifact / output)
    for name, job in (("compile", compile_job), ("sparse-pending-compile", sparse_job)):
        report = check_upload(job, broker, artifact / "upload-audit" / name)
        publish(artifact / "upload-audit" / f"{name}.json", report)
    publish(artifact / "native-compile/native-compile.json", compile_job)
    publish(artifact / "sparse-bench-canary-pending-compile.json", sparse_job)
    publish(artifact / "repack_gpu_upload.py", Path(__file__).read_text())
    py = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"
    dispatch = str(broker / "tools/ops/dispatch.sh")
    compile_command = shlex.join(
        [
            dispatch,
            compile_job["job_id"],
            str(artifact / "native-compile/native-compile.json"),
            "--broker",
            str(broker),
            "--cwd",
            str(broker),
            "--state-dir",
            str(artifact / "native-compile/state"),
            "--dispatch-txt",
            str(artifact / "DISPATCH.txt"),
        ]
    )
    bind_command = shlex.join(
        [
            py,
            "-I",
            str(artifact / "repack_gpu_upload.py"),
            "bind-canary",
            "--artifact-dir",
            str(artifact),
            "--broker",
            str(broker),
        ]
    )
    sparse_command = shlex.join(
        [
            dispatch,
            sparse_job["job_id"],
            str(artifact / "sparse-bench-canary.json"),
            "--broker",
            str(broker),
            "--cwd",
            str(broker),
            "--state-dir",
            str(artifact / "state-canary"),
            "--dispatch-txt",
            str(artifact / "DISPATCH.txt"),
        ]
    )
    for name, commands in (
        ("dispatch-compile.sh", [compile_command]),
        (
            "dispatch-canary.sh",
            [
                "COMPILE_RECEIPT=${NATIVE_COMPILE_RECEIPT:-}",
                'if [[ -z "$COMPILE_RECEIPT" ]]; then COMPILE_RECEIPT='
                + shlex.quote(str(artifact / "native-compile/output/receipt.json"))
                + "; fi",
                bind_command + ' --compile-receipt "$COMPILE_RECEIPT"',
                sparse_command,
            ],
        ),
    ):
        publish(
            artifact / name, "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n"
        )
    publish(
        artifact / "DISPATCH.txt",
        "# Compile first; sparse helper requires same-pin PASS and verified stop.\n"
        + f"nice -n 15 bash {shlex.quote(str(artifact / 'dispatch-compile.sh'))}\n"
        + f"nice -n 15 bash {shlex.quote(str(artifact / 'dispatch-canary.sh'))}\n",
    )
    print(json.dumps({"prepared": str(artifact), "runtime_pin": pin, "dispatched": False}))


def bind_canary(artifact: Path, broker: Path, compile_receipt: Path | None = None) -> None:
    from research.gpu_sim_spike.native_compile import binding, require_for_a100_canary

    path = artifact / "sparse-bench-canary.json"
    if path.exists():
        job = verified_json(path)
        require_for_a100_canary(job, broker)
        if (
            compile_receipt is not None
            and binding(compile_receipt, job["commit"], broker)
            != job["native_cuda_compile"]["receipt"]
        ):
            raise ValueError("immutable canary already binds a different compile receipt")
        from research.colab_cli.supervisor import validate

        validate(job, repo=broker)
        return
    job = verified_json(artifact / "sparse-bench-canary-pending-compile.json")
    proof = binding(
        compile_receipt or artifact / "native-compile/output/receipt.json", job["commit"], broker
    )
    job["native_cuda_compile"] = {"receipt": proof}
    require_for_a100_canary(job, broker)
    report = check_upload(job, broker, artifact / "upload-audit/sparse-bound")
    publish(artifact / "upload-audit/sparse-bound.json", report)
    publish(path, job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "bind-canary"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--attempt", default="uploadfix01")
    parser.add_argument("--compile-receipt", type=Path)
    args = parser.parse_args()
    broker = args.broker.resolve(strict=True)
    # Import the qualified frozen runtime, never the authoring checkout's
    # transport or spike modules. -I on the generated command excludes ambient paths.
    sys.path.insert(0, str(broker))
    artifact = args.artifact_dir.resolve()
    if args.action == "prepare":
        if args.source_dir is None:
            parser.error("prepare requires --source-dir")
        prepare_package(args.source_dir.resolve(strict=True), artifact, broker, args.attempt)
    else:
        bind_canary(artifact, broker, args.compile_receipt)


if __name__ == "__main__":
    main()
