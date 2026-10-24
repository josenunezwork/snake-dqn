"""Add the same-pin L4 compile prerequisite to the prepared I0 packet; never auto-dispatch.

The Mac companion uses the corrected host receipt reader and dispatcher while
all runtime sources and transport validation remain on the frozen broker pin.
Existing hash-bound packet controls are retained; binding creates a sibling.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
ARTIFACT = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-integration-i0-20261024-03"
BROKER = BASE / "snake-dqn-gpuint-qualified-03.noindex"
PY = BASE / "snake-dqn/venv/bin/python"
PIN = "968f02765f7c86ca27d2947347896e2fe4e6ce23"


def frozen(broker: Path, code: str, *args: str) -> dict[str, Any]:
    """Run pinned package/transport checks without importing authoring transport."""
    result = subprocess.run(
        [
            str(PY),
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv.pop(1)); " + code,
            str(broker),
            *args,
        ],
        cwd=broker,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return json.loads(result.stdout)


def upload(job: dict[str, Any], broker: Path, stage: Path) -> dict[str, Any]:
    """Check real supervisor admission and every staged upload's byte bound."""
    with tempfile.TemporaryDirectory(prefix="i0-upload-manifest-") as temporary:
        path = Path(temporary) / "job.json"
        path.write_text(json.dumps(job))
        return frozen(
            broker,
            "import json; from pathlib import Path; "
            "from tools.ops.repack_gpu_upload import check_upload; "
            "print(json.dumps(check_upload(json.loads(Path(sys.argv[1]).read_text()), "
            "Path(sys.path[0]), Path(sys.argv[2]))))",
            str(path),
            str(stage),
        )


def controls(artifact: Path, broker: Path) -> dict[str, Any]:
    """Verify the original packet and unchanged runtime/qualified transport."""
    from tools.ops.qualification import OPS, check, git
    from tools.ops.repack_gpu_upload import verified_json

    job = verified_json(artifact / "packet/integration-canary.json")
    if job["commit"] != PIN or git(broker, "rev-parse", "HEAD") != PIN:
        raise ValueError("integration runtime pin differs")
    check(broker, OPS / "colab-cli-qualified.json")
    check(broker, OPS / "colab-cli-qualified.json", ref=PIN)
    return job


def dispatch_command(
    artifact: Path, broker: Path, job: dict[str, Any], path: Path, lane: str
) -> list[str]:
    """Use host admission fixes with the unchanged qualified runtime broker."""
    return [
        str(PY),
        str(ROOT / "tools/ops/dispatch.py"),
        "gpu-integration-i0-canary-968f0276" if lane == "canary" else job["job_id"],
        str(path),
        "--broker",
        str(broker),
        "--cwd",
        str(broker),
        "--state-dir",
        str(
            artifact
            / ("packet/state-canary" if lane == "canary" else "packet/native-compile/state")
        ),
        "--dispatch-txt",
        str(artifact / "packet/DISPATCH.txt"),
    ]


def prepare(artifact: Path, broker: Path) -> dict[str, Any]:
    """Create compiler controls and replace only the unbound operator wrapper."""
    from tools.ops.repack_gpu_upload import publish, verified_json

    controls(artifact, broker)
    compile_root = artifact / "packet/native-compile"
    job = frozen(
        broker,
        "import json; from pathlib import Path; "
        "from research.gpu_sim_spike.native_compile_pack import manifest; "
        "from tools.ops.repack_gpu_upload import compile_sources; "
        "j=manifest(sys.argv[1], Path(sys.argv[2])); "
        "j['source_paths']=compile_sources(Path(sys.path[0]), sys.argv[1]); "
        "print(json.dumps(j))",
        PIN,
        str(compile_root),
    )
    job["job_id"] += "-integration03"
    path = compile_root / "native-compile.json"
    if path.exists():
        if verified_json(path) != job:
            raise ValueError("immutable compiler manifest differs")
    else:
        report = upload(job, broker, compile_root / "upload-audit")
        publish(compile_root / "UPLOAD_CHECK.json", report)
        publish(path, job)
    for lane in ("compile", "canary"):
        script = artifact / f"dispatch-{lane}.sh"
        command = shlex.join(
            [
                str(PY),
                "-I",
                str(Path(__file__).resolve()),
                lane,
                "--artifact-dir",
                str(artifact),
                "--broker",
                str(broker),
            ]
        )
        content = "#!/usr/bin/env bash\nset -euo pipefail\nexec " + command + ' "$@"\n'
        if script.exists() and script.read_text() != content:
            if lane != "canary":
                raise ValueError("existing compile helper differs")
            backup = artifact / "dispatch-canary.before-compile.sh"
            if backup.exists():
                raise ValueError("canary helper already replaced")
            publish(backup, script.read_text())
        script.write_text(content)
        script.chmod(0o755)
    return {"status": "PREPARED_NOT_DISPATCHED", "runtime_pin": PIN, "compile_manifest": str(path)}


def run(
    lane: str, artifact: Path, broker: Path, dry: bool, receipt: Path | None = None
) -> dict[str, Any]:
    """Validate the full helper path; dry runs never publish a bound canary."""
    from research.gpu_sim_spike import native_compile as native
    from tools.ops.repack_gpu_upload import publish, verified_json

    original = controls(artifact, broker)
    if lane == "canary":
        path = artifact / "packet/integration-canary-compiled.json"
        receipt = receipt or Path(
            os.environ.get("NATIVE_COMPILE_RECEIPT")
            or artifact / "packet/native-compile/output/receipt.json"
        )
        job = dict(original)
        job["native_cuda_compile"] = {"receipt": native.binding(receipt, PIN, broker)}
        native.require_for_a100_canary(job, broker)
    else:
        if receipt is not None:
            raise ValueError("compile receipt is only valid for canary")
        path = artifact / "packet/native-compile/native-compile.json"
        job = verified_json(path)
        if job["commit"] != PIN or job["script"] != "research/gpu_sim_spike/native_compile.sh":
            raise ValueError("compiler identity differs")
    if path.exists() and verified_json(path) != job:
        raise ValueError("immutable bound manifest differs")
    with tempfile.TemporaryDirectory(prefix="i0-compile-validation-") as temporary:
        temporary_root = Path(temporary)
        report = upload(job, broker, temporary_root / "upload")
        validation_manifest = temporary_root / path.name
        validation_manifest.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        command = dispatch_command(artifact, broker, job, validation_manifest, lane)
        # Always execute genuine dispatch validation before publishing controls.
        subprocess.run([*command, "--dry-run"], cwd=ROOT, check=True)
        if not dry:
            if not path.exists():
                publish(path, job)
            subprocess.run(
                dispatch_command(artifact, broker, job, path, lane), cwd=ROOT, check=True
            )
    return {
        "status": "PASS_DRY_RUN" if dry else "DISPATCHED",
        "runtime_pin": PIN,
        "lane": lane,
        "largest_upload_bytes": report["largest_upload_bytes"],
        "source_files": report["source_files"],
        "source_archive_bytes": report["source_archive_bytes"],
        "input_rows": len(job["inputs"]),
        "live_accounting": "NOT_CHECKED" if dry else "CHECKED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compile", "canary"))
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT)
    parser.add_argument("--broker", type=Path, default=BROKER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compile-receipt", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.action == "prepare":
        if args.dry_run or args.compile_receipt:
            parser.error("prepare accepts neither --dry-run nor --compile-receipt")
        result = prepare(args.artifact_dir.resolve(), args.broker.resolve())
    else:
        result = run(
            args.action,
            args.artifact_dir.resolve(),
            args.broker.resolve(),
            args.dry_run,
            args.compile_receipt,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
