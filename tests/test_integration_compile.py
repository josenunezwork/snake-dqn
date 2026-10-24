"""Local I0 acceptance through real helpers; no provider calls or proof publication.

Portable receipt and dispatch regressions live in test_native_compile.py and
test_ops_dispatch.py. These checks additionally use the operator's frozen packet.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from research.gpu_sim_spike import native_compile as native
from tools.ops import integration_compile as helper

SPARSE = helper.BASE / "snake-dqn-artifacts/big-lever/exp/gpu-sparse-metadata-20261024-02/packet"
SPARSE_BROKER = helper.BASE / "snake-dqn-sparse-metadata-qualified-02.noindex"
pytestmark = pytest.mark.skipif(
    not (SPARSE / "native-compile/output/receipt.json").is_file()
    or not (helper.ARTIFACT / "dispatch-compile.sh").is_file()
    or not helper.BROKER.is_dir(),
    reason="requires the frozen local I0 and sparse operator packets",
)


def synthetic_compile(root: Path) -> Path:
    """Model the actual supervisor's payload-only export rows, with real sidecars."""
    from tools.ops.repack_gpu_upload import publish

    real = SPARSE / "native-compile/output"
    receipt = json.loads((real / "receipt.json").read_text())
    assert [row["path"] for row in receipt["exports"]] == ["native-compile.json"]
    assert receipt["terminal"]["exports"] == receipt["exports"]
    report = json.loads((real / "exports/native-compile.json").read_text())
    hashes, specs = native.inventory(helper.BROKER, helper.PIN)
    report.update(commit=helper.PIN, source_sha256=hashes, extensions=specs)
    publish(root / "exports/native-compile.json", report)
    manifest = helper.ARTIFACT / "packet/native-compile/native-compile.json"
    (root / "manifest.json").write_bytes(manifest.read_bytes())
    job = json.loads(manifest.read_text())
    row = {
        "path": "native-compile.json",
        "size": (root / "exports/native-compile.json").stat().st_size,
        "sha256": native.digest(root / "exports/native-compile.json"),
    }
    receipt.update(
        commit=helper.PIN,
        job_id=job["job_id"],
        manifest_sha256=native.digest(root / "manifest.json"),
        exports=[row],
    )
    receipt["terminal"].update(job_id=job["job_id"], exports=[row])
    receipt["terminal"]["export_scanner"]["inventory"] = [row]
    # Kept only as a validation fixture. It is never the canonical compile output.
    path = root / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def test_actual_sparse_compile_pass_is_accepted():
    path = SPARSE / "native-compile/output/receipt.json"
    pin = json.loads(path.read_text())["commit"]
    assert native.binding(path, pin, SPARSE_BROKER)["sha256"] == native.digest(path)


def test_full_generated_helpers_with_realistic_synthetic_receipt(tmp_path):
    receipt = synthetic_compile(tmp_path / "synthetic-compile-output")
    before = {str(p): native.digest(p) for p in helper.ARTIFACT.rglob("*") if p.is_file()}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("NATIVE_COMPILE_RECEIPT", None)
    for lane, extra in (("compile", []), ("canary", ["--compile-receipt", str(receipt)])):
        result = subprocess.run(
            ["bash", str(helper.ARTIFACT / f"dispatch-{lane}.sh"), "--dry-run", *extra],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "allocations=0" in result.stdout
        assert json.loads(result.stdout.splitlines()[-1])["status"] == "PASS_DRY_RUN"
    assert {str(p): native.digest(p) for p in helper.ARTIFACT.rglob("*") if p.is_file()} == before
    assert not (helper.ARTIFACT / "packet/integration-canary-compiled.json").exists()
    assert not (helper.ARTIFACT / "packet/native-compile/output").exists()
