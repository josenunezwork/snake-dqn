"""R2 operator companion: L4 compile prerequisite, A100 canary, then A100 main.

Mirrors the I0 companion (``integration_compile.py``): every runtime/transport
check runs from the frozen broker checkout of the R2 runtime pin, while the
host dispatcher (``tools/ops/dispatch.py`` of the ``--host`` checkout, which
has ``--dry-run``) performs governed validation and the single allocation.
Every lane first runs the real dispatcher with ``--dry-run`` (allocations=0)
and publishes a bound manifest only after it passes.  Nothing here dispatches
unless the operator omits ``--dry-run``.
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
BASE = Path("/Users/josenunez/Projects/ml")
ARTIFACT = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-r2-padding-20261024"
BROKER = BASE / "snake-dqn-gpur2-qualified.noindex"
HOST = BASE / "snake-dqn"
PY = BASE / "snake-dqn/venv/bin/python"
PACKAGE = "research/gpu_sim_spike"


def frozen(broker: Path, code: str, *args: str) -> Any:
    """Run pinned package/transport code from the broker without authoring imports."""
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
        timeout=900,
    )
    if result.returncode:
        raise RuntimeError(f"broker helper failed: {result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, timeout=30).strip()


def upload(job: dict[str, Any], broker: Path, stage: Path) -> dict[str, Any]:
    """Real supervisor validation plus the exact staged-upload byte bound."""
    with tempfile.TemporaryDirectory(prefix="r2-upload-manifest-") as temporary:
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


def _verified(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from tools.ops.repack_gpu_upload import verified_json

    return verified_json(path)


def _publish(path: Path, value: Any) -> None:
    sys.path.insert(0, str(ROOT))
    from tools.ops.repack_gpu_upload import publish

    publish(path, value)


def controls(artifact: Path, broker: Path) -> tuple[dict[str, Any], str]:
    """The prepared canary, its pin, and an unchanged qualified broker at that pin."""
    sys.path.insert(0, str(ROOT))
    from tools.ops.qualification import OPS, check

    job = _verified(artifact / "packet/r2-canary.json")
    pin = job["commit"]
    if git(broker, "rev-parse", "HEAD") != pin:
        raise ValueError("broker checkout is not at the R2 runtime pin")
    check(broker, OPS / "colab-cli-qualified.json")
    check(broker, OPS / "colab-cli-qualified.json", ref=pin)
    return job, pin


def dispatch_command(
    host: Path, artifact: Path, broker: Path, name: str, path: Path, lane: str
) -> list[str]:
    state = {
        "compile": "packet/native-compile/state",
        "canary": "packet/state-canary",
        "main": "packet/state-main",
    }[lane]
    return [
        str(PY),
        str(host / "tools/ops/dispatch.py"),
        name,
        str(path),
        "--broker",
        str(broker),
        "--cwd",
        str(broker),
        "--state-dir",
        str(artifact / state),
        "--dispatch-txt",
        str(artifact / "packet/DISPATCH.txt"),
    ]


def prepare(artifact: Path, broker: Path) -> dict[str, Any]:
    """Create the same-pin L4 compile manifest and the operator wrappers."""
    _canary, pin = controls(artifact, broker)
    compile_root = artifact / "packet/native-compile"
    job = frozen(
        broker,
        "import json; from pathlib import Path; "
        "from research.gpu_sim_spike.native_compile_pack import manifest; "
        "from tools.ops.repack_gpu_upload import compile_sources; "
        "j=manifest(sys.argv[1], Path(sys.argv[2])); "
        "j['source_paths']=compile_sources(Path(sys.path[0]), sys.argv[1]); "
        "print(json.dumps(j))",
        pin,
        str(compile_root),
    )
    job["job_id"] += "-r2"
    path = compile_root / "native-compile.json"
    if path.exists():
        if _verified(path) != job:
            raise ValueError("immutable compiler manifest differs")
    else:
        report = upload(job, broker, compile_root / "upload-audit")
        _publish(compile_root / "UPLOAD_CHECK.json", report)
        _publish(path, job)
    for lane in ("compile", "canary", "main"):
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
            raise ValueError(f"existing wrapper differs: {script}")
        script.write_text(content)
        script.chmod(0o755)
    return {"status": "PREPARED_NOT_DISPATCHED", "runtime_pin": pin, "compile_manifest": str(path)}


def _main_manifest(artifact: Path, broker: Path, pin: str, receipt: Path) -> Path:
    """Prepare (once) the canary-bound main through the broker's frozen packer."""
    path = artifact / "packet/r2-main.json"
    if not path.exists():
        frozen(
            broker,
            "import json; from pathlib import Path; "
            "from research.gpu_sim_spike.r2_pack import prepare; "
            "print(json.dumps(prepare(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), "
            "canary_receipt=Path(sys.argv[4]), validate=True)))",
            str(artifact / "packet"),
            pin,
            str(artifact / "mac-preflight/preflight.json"),
            str(receipt),
        )
    return path


def run(
    lane: str,
    artifact: Path,
    broker: Path,
    host: Path,
    dry: bool,
    receipt: Path | None = None,
) -> dict[str, Any]:
    original, pin = controls(artifact, broker)
    name = {
        "canary": f"gpu-r2-canary-{pin[:8]}",
        "main": f"gpu-r2-main-{pin[:8]}",
    }.get(lane)
    if lane == "compile":
        if receipt is not None:
            raise ValueError("a receipt is only valid for canary or main")
        path = artifact / "packet/native-compile/native-compile.json"
        job = _verified(path)
        if job["commit"] != pin or job["script"] != f"{PACKAGE}/native_compile.sh":
            raise ValueError("compiler identity differs")
        name = job["job_id"]
    elif lane == "canary":
        path = artifact / "packet/r2-canary-compiled.json"
        receipt = receipt or Path(
            os.environ.get("NATIVE_COMPILE_RECEIPT")
            or artifact / "packet/native-compile/output/receipt.json"
        )
        job = dict(original)
        job["native_cuda_compile"] = {
            "receipt": frozen(
                broker,
                "import json; from pathlib import Path; "
                "from research.gpu_sim_spike import native_compile as n; "
                "b=n.binding(Path(sys.argv[1]), sys.argv[2], Path(sys.path[0])); "
                "n.require_for_a100_canary({**json.loads(sys.argv[3]), "
                "'native_cuda_compile': {'receipt': b}}, Path(sys.path[0])); "
                "print(json.dumps(b))",
                str(receipt),
                pin,
                json.dumps(original),
            )
        }
    else:
        receipt = receipt or artifact / "packet/canary-output/receipt.json"
        if dry and not receipt.is_file():
            raise ValueError(
                "main dry-run requires the canary receipt (the qualified supervisor binds it)"
            )
        path = _main_manifest(artifact, broker, pin, receipt)
        job = _verified(path)
        if job["commit"] != pin or job.get("scientific_admission") is not True:
            raise ValueError("main manifest identity differs")
    if path.exists() and _verified(path) != job:
        raise ValueError("immutable bound manifest differs")
    with tempfile.TemporaryDirectory(prefix="r2-validation-") as temporary:
        temporary_root = Path(temporary)
        report = upload(job, broker, temporary_root / "upload")
        validation_manifest = temporary_root / path.name
        validation_manifest.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        command = dispatch_command(host, artifact, broker, name, validation_manifest, lane)
        subprocess.run([*command, "--dry-run"], cwd=host, check=True)
        if not dry:
            if not path.exists():
                _publish(path, job)
            subprocess.run(
                dispatch_command(host, artifact, broker, name, path, lane), cwd=host, check=True
            )
    return {
        "status": "PASS_DRY_RUN" if dry else "DISPATCHED",
        "runtime_pin": pin,
        "lane": lane,
        "dispatch_name": name,
        "largest_upload_bytes": report.get("largest_upload_bytes"),
        "source_files": report.get("source_files"),
        "source_archive_bytes": report.get("source_archive_bytes"),
        "input_rows": len(job["inputs"]),
        "live_accounting": "NOT_CHECKED" if dry else "CHECKED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compile", "canary", "main"))
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT)
    parser.add_argument("--broker", type=Path, default=BROKER)
    parser.add_argument("--host", type=Path, default=HOST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.action == "prepare":
        if args.dry_run or args.receipt:
            parser.error("prepare accepts neither --dry-run nor --receipt")
        result = prepare(args.artifact_dir.resolve(), args.broker.resolve())
    else:
        result = run(
            args.action,
            args.artifact_dir.resolve(),
            args.broker.resolve(),
            args.host.resolve(),
            args.dry_run,
            args.receipt,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
