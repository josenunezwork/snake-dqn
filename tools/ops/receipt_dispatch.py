"""Mac-side repairs for two saved dispatch helpers; frozen packets stay intact.

Dry-run executes receipt, source, qualification, upload and supervisor checks.
Live accounting admission is reported separately and remains mandatory for V8.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
EXP = BASE / "snake-dqn-artifacts/big-lever/exp"
SPARSE = EXP / "gpu-sparse-metadata-20261024-02/packet"
V8 = EXP / "gpu-v8-main-amended-20261024-01"
PY = BASE / "snake-dqn/venv/bin/python"


def reviewed_files(artifact: Path, lane: str) -> list[Path]:
    """Exact local implementation closure covered by the new light review."""
    return [
        Path(__file__).resolve(),
        ROOT / "tools/ops/dispatch.py",
        ROOT / "tools/ops/repack_gpu_upload.py",
        ROOT / "research/gpu_sim_spike/native_compile.py",
        artifact / ("dispatch-canary.sh" if lane == "sparse" else "dispatch-main.sh"),
    ]


def require_review(artifact: Path, lane: str) -> None:
    """The repaired wrapper needs its own acceptance; old receipts stay frozen."""
    from research.gpu_sim_spike.sparse_units import _verify

    path = artifact / "CODEX_V_RECEIPT_FIX.json"
    _verify(path)
    value = json.loads(path.read_text())
    expected = {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in reviewed_files(artifact, lane)
    }
    if value.get("status") != "ACCEPT" or value.get("source_sha256") != expected:
        raise ValueError("receipt fix lacks exact Codex-V acceptance")


def load(path: Path, name: str) -> ModuleType:
    """Load a host companion without changing the qualified broker's files."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch(job: dict[str, Any], path: Path, broker: Path, artifact: Path, dry: bool) -> None:
    command = [
        str(PY),
        str(ROOT / "tools/ops/dispatch.py"),
        job["job_id"],
        str(path),
        "--broker",
        str(broker),
        "--cwd",
        str(broker),
        "--state-dir",
        str(artifact / ("state-canary" if artifact == SPARSE else "state-main")),
        "--dispatch-txt",
        str(artifact / "DISPATCH.txt"),
    ]
    if dry:
        command.append("--dry-run")
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def upload_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "status",
            "commit",
            "job_id",
            "manifest_sha256",
            "largest_upload_bytes",
            "per_file_bound_bytes",
            "source_files",
            "source_archive_bytes",
        )
    }


def sparse(dry: bool) -> dict[str, Any]:
    broker = BASE / "snake-dqn-sparse-metadata-qualified-02.noindex"
    sys.path.insert(0, str(broker))
    require_review(SPARSE, "sparse")
    repack = load(SPARSE / "repack_gpu_upload.py", "saved_sparse_repack")
    # Only the host binder changes. inventory(repo, commit) still reads every
    # native source/flag from the compile job's original frozen runtime pin.
    native = load(ROOT / "research/gpu_sim_spike/native_compile.py", "host_native_binding")
    job = repack.verified_json(SPARSE / "sparse-bench-canary-pending-compile.json")
    receipt = Path(
        os.environ.get("NATIVE_COMPILE_RECEIPT")
        or str(SPARSE / "native-compile/output/receipt.json")
    )
    job["native_cuda_compile"] = {"receipt": native.binding(receipt, job["commit"], broker)}
    native.require_for_a100_canary(job, broker)
    path = SPARSE / "sparse-bench-canary.json"
    if path.exists() and repack.verified_json(path) != job:
        raise ValueError("immutable bound canary differs from pending job and compile proof")
    with tempfile.TemporaryDirectory(prefix="sparse-receipt-validation-") as temporary:
        root = Path(temporary)
        upload = repack.check_upload(job, broker, root / "upload")
        if dry:
            temporary_manifest = root / path.name
            temporary_manifest.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
            dispatch(job, temporary_manifest, broker, SPARSE, True)
        else:
            if not path.exists():
                repack.publish(path, job)
            dispatch(job, path, broker, SPARSE, False)
    return {"status": "PASS_DRY_RUN" if dry else "DISPATCHED", "upload": upload_summary(upload)}


def verify_v8_preparation(amend: ModuleType) -> dict[str, Any]:
    """Retain every prepared hash, including the preserved original shell bytes."""
    prep_path = V8 / "PREPARATION.json"
    amend._verify_file(prep_path)
    preparation = json.loads(prep_path.read_text())
    original = V8 / "dispatch-main.before-receipt-fix.sh"
    if (
        preparation["implementation"] != amend.implementation()
        or preparation["manifest_sha256"] != amend._verify_file(V8 / "v8-split-main.json")
        or preparation["dispatch_sha256"] != amend.sha256_file(original)
    ):
        raise ValueError("prepared source/tests/manifest/original dispatch binding differs")
    job = json.loads((V8 / "v8-split-main.json").read_text())
    amend.validate_final(job)
    review_path = V8 / "CODEX_V_LIGHT.json"
    amend._verify_file(review_path)
    review = json.loads(review_path.read_text())
    if (
        review.get("status") != "ACCEPT_OFFLINE"
        or review.get("preparation_sha256") != amend._verify_file(prep_path)
        or review.get("manifest_sha256") != amend._verify_file(V8 / "v8-split-main.json")
        or review.get("implementation") != amend.implementation()
    ):
        raise ValueError("INCOMPLETE_ADMISSION: exact package lacks Codex-V acceptance")
    return job


def v8(dry: bool) -> dict[str, Any]:
    broker = BASE / "snake-dqn-v8-metadata-qualified-02.noindex"
    sys.path.insert(0, str(broker))
    require_review(V8, "v8")
    amend = load(BASE / "snake-dqn-gpu-v8/tools/v8_split_main_amend.py", "saved_v8_amendment")
    job = verify_v8_preparation(amend)
    # The actual canary has payload.json.complete, payload.json.sha256 trios.
    # The unchanged checker already verifies these and reduces PASS_CANARY.
    admission_path = V8 / "OPERATOR_ADMISSION.json"
    admission = "PASS"
    if dry and not admission_path.exists():
        admission = f"NOT_READY: missing {admission_path} (with .sha256 and .complete)"
    else:
        if not admission_path.exists():
            raise ValueError(f"INCOMPLETE_ADMISSION: missing operator accounting: {admission_path}")
        amend.admission()
    repack = load(ROOT / "tools/ops/repack_gpu_upload.py", "v8_upload_checker")
    with tempfile.TemporaryDirectory(prefix="v8-receipt-validation-") as temporary:
        upload = repack.check_upload(job, broker, Path(temporary) / "upload")
    dispatch(job, V8 / "v8-split-main.json", broker, V8, dry)
    return {
        "status": "PASS_OFFLINE_DRY_RUN" if dry else "DISPATCHED",
        "live_admission": admission,
        "dispatch_ready": admission == "PASS",
        "upload": upload_summary(upload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("sparse", "v8"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = (sparse if args.lane == "sparse" else v8)(args.dry_run)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
