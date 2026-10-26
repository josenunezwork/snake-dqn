"""F5 broker-boundary tests.

The broker receipts in the first group are explicitly *simulated CPU broker
observations*.  They exercise the unchanged canary admission and runtime
export scanner with the F5 qualification manifest shape; they are not CUDA
qualification evidence.  The final test is the only test that can emit real
CUDA observer evidence, and it is skipped when CUDA is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from research.bl_ppo.exports import export_json
from research.bl_residual import f5_package as package
from research.colab_cli import canary, canary_observer, runtime

REPO = Path(__file__).resolve().parents[1]
F5_SCRIPT = "research/bl_residual/f5_colab.sh"
F5_PACKAGE = "research/bl_residual"
F5_EXPORTS = (
    "paired/assignment-environment.json",
    "paired/fit-receipt.json",
    "paired/fit-C.pth",
    "paired/fit-W.pth",
    "paired/f5-state-C.pt",
    "paired/f5-state-W.pt",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _transport_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a pinned transport checkout without committing the working tree."""
    repo = tmp_path / "transport-repo"
    (repo / "research/colab_cli").mkdir(parents=True)
    (repo / "research/bl_residual").mkdir(parents=True)
    shutil.copytree(REPO / "research/colab_cli", repo / "research/colab_cli", dirs_exist_ok=True)
    shutil.copy2(REPO / F5_SCRIPT, repo / F5_SCRIPT)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "F5 broker test")
    _git(repo, "config", "user.email", "f5-broker-test@example.invalid")
    _git(repo, "add", "research")
    _git(repo, "commit", "-qm", "pinned F5 transport")
    return repo, _git(repo, "rev-parse", "HEAD")


def _export(path: Path, value: object) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return export_json(path, value)


def _science_inputs(tmp_path: Path) -> list[dict[str, str]]:
    """Create the complete actual F5 input inventory with verified sidecars."""
    root = tmp_path / "science-inputs"
    root.mkdir()
    rows: list[dict[str, str]] = []
    simple = {
        "control.json": {"schema": "f5-test-control/v1"},
        "b8.pth": {"schema": "f5-test-b8/v1", "checkpoint": "synthetic"},
        "selected-training.tar.xz": {"schema": "f5-test-selected/v1"},
        "f5-batch-plan.pt": {"schema": "f5-test-plan/v1"},
        "acceptance.json": {"schema": "f5-test-acceptance/v1"},
        "verified-ledger.json": {"schema": "bl-f5-verified-ledger/v1", "units": []},
        "timing-fixtures.json": {
            "schema": "bl-f5-sealed-timing-fixtures/v1",
            "fixtures": [],
        },
    }
    immutable = {"selected-training.tar.xz", "f5-batch-plan.pt", "acceptance.json"}
    for name, value in simple.items():
        info = _export(root / name, value)
        rows.append({"path": str(root / name), "dest": name, "sha256": info["sha256"]})
        if name in immutable:
            for suffix in (".sha256", ".complete"):
                sidecar = root / (name + suffix)
                rows.append({"path": str(sidecar), "dest": name + suffix, "sha256": _sha(sidecar)})
    return rows


def _manifest_pair(
    tmp_path: Path, repo: Path, commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _science_inputs(tmp_path)
    source_paths = ["research/bl_residual/f5_qualification.py"]
    science: dict[str, Any] = {
        "job_id": "bl-f5-w-qualification-test-01",
        "commit": commit,
        "script": F5_SCRIPT,
        "source_paths": source_paths,
        "device": "cpu",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "est_minutes": 1},
        "timeout_minutes": 20,
        "output_dir": str(tmp_path / "science-output"),
        "inputs": rows,
        "required_exports": list(F5_EXPORTS),
        "env": {"F5_MODE": "qualification", "F5_DEVICE": "cpu"},
    }
    qualification = {
        **science,
        "job_id": "bl-f5-w-qualification-canary-test-01",
        "output_dir": str(tmp_path / "canary-output"),
        "env": {
            "F5_MODE": "qualification",
            "F5_DEVICE": "cpu",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": F5_PACKAGE,
        },
        "canary": {
            "schema": canary.SCHEMA,
            "mode": "probe",
            "family": F5_SCRIPT,
        },
    }
    assert repo.is_dir()
    return science, qualification


def _broker_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    repo, commit = _transport_repo(tmp_path)
    science, probe = _manifest_pair(tmp_path, repo, commit)
    # Bind the complete inventory through the production F5 package helper.
    bound = package.bind_qualification_inventory(science, probe)
    assert bound["inventory_status"] == "BOUND_EXACT_ACTUAL_SCIENCE_UPLOADS"
    output = tmp_path / "canary-output"
    output.mkdir()
    _export(output / "manifest.json", probe)
    _export(
        output / "upload-measurement.json",
        {"elapsed_seconds": 12.0, "transfer_seconds": 4.0, "source_delivery_seconds": 3.0},
    )
    _export(output / "hardware.json", {"hardware": {"kind": "L4"}})

    remote = tmp_path / "remote"
    (remote / "output").mkdir(parents=True)
    for name in F5_EXPORTS:
        payload = remote / "output" / name
        _export(payload, {"schema": "f5-test-producer/v1", "path": name})
    inventory = runtime._exports(remote, 1024 * 1024)
    for row in inventory:
        target = output / "exports" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((remote / "output" / row["path"]).read_bytes())
    receipt: dict[str, Any] = {
        "status": "PASS",
        "stop_verified": True,
        "failure_class": None,
        "cleanup_error": None,
        "generation": "f5-test-generation",
        "commit": commit,
        "job_id": probe["job_id"],
        "manifest_sha256": _sha(output / "manifest.json"),
        "exports": inventory,
        "terminal": {
            "job_id": probe["job_id"],
            "generation": "f5-test-generation",
            "returncode": 0,
            "status": "succeeded",
            "export_scanner": {"name": "runtime._exports", "inventory": inventory},
            # This is intentionally empty: this fixture is simulated CPU
            # broker evidence and is never CUDA qualification proof.
            "canary_observations": [],
        },
        "transfers": [
            {
                "direction": "upload",
                "status": "verified",
                "remote": f"/content/f5-test/inputs/{row['dest']}",
                "size_bytes": Path(row["path"]).stat().st_size,
                "sha256": row["sha256"],
            }
            for row in probe["inputs"]
        ],
    }
    receipt["canary_evidence"] = canary.evidence(probe, receipt, output, repo)
    # Keep the broker-produced proof in the probe-shaped receipt object for
    # assertions; ``canary.check`` recomputes it from the unchanged evidence
    # and receipt files below.
    probe["canary_evidence"] = receipt["canary_evidence"]
    receipt_path = output / "receipt.json"
    _export(receipt_path, receipt)
    science["canary"] = {"receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)}}
    return repo, science, probe, output


def test_f5_cpu_simulated_broker_evidence_covers_actual_inventory_and_runtime_exports(
    tmp_path: Path,
) -> None:
    """SIMULATED CPU broker observations; this is not actual CUDA qualification proof."""
    repo, science, probe, output = _broker_fixture(tmp_path)
    assert len(probe["inputs"]) == 13
    assert {row["dest"] for row in probe["inputs"]} >= {
        "control.json",
        "b8.pth",
        "selected-training.tar.xz",
        "selected-training.tar.xz.sha256",
        "selected-training.tar.xz.complete",
        "f5-batch-plan.pt",
        "f5-batch-plan.pt.sha256",
        "f5-batch-plan.pt.complete",
        "acceptance.json",
        "acceptance.json.sha256",
        "acceptance.json.complete",
        "verified-ledger.json",
        "timing-fixtures.json",
    }
    assert canary.input_inventory(probe) == probe["canary_evidence"]["inputs"]
    assert set(probe["canary_evidence"]["exports"]) == set(F5_EXPORTS)
    assert probe["canary_evidence"]["cuda_observations"] == []
    canary.check(science, repo)


@pytest.mark.parametrize("fault", ["upload_missing", "upload_hash", "export_missing", "source"])
def test_f5_cpu_simulated_broker_refuses_incomplete_or_foreign_evidence(
    tmp_path: Path, fault: str
) -> None:
    """These are simulated CPU receipt faults, never CUDA qualification evidence."""
    repo, science, probe, output = _broker_fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if fault == "upload_missing":
        receipt["transfers"].pop()
    elif fault == "upload_hash":
        receipt["transfers"][0]["sha256"] = "0" * 64
    elif fault == "export_missing":
        receipt["exports"] = receipt["exports"][:-1]
        receipt["terminal"]["export_scanner"]["inventory"] = receipt["exports"]
    else:
        science["source_paths"] = ["research/bl_residual/foreign.py"]
    if fault == "source":
        with pytest.raises(canary.CanaryError):
            canary.check(science, repo)
    else:
        with pytest.raises(canary.CanaryError):
            canary.evidence(probe, receipt, output, repo)


def test_f5_runtime_scanner_refuses_mismatched_producer_sidecar(tmp_path: Path) -> None:
    """The unchanged runtime scanner must reject altered producer bytes."""
    remote = tmp_path / "remote"
    output = remote / "output" / "paired"
    output.mkdir(parents=True)
    payload = output / "fit-C.pth"
    payload.write_bytes(b"producer-payload")
    (Path(str(payload) + ".sha256")).write_text("0" * 64 + "  fit-C.pth\n")
    (Path(str(payload) + ".complete")).write_text("complete\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        runtime._exports(remote, 1024)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA runtime")
def test_real_cuda_observer_records_source_native_f5_selection(tmp_path: Path) -> None:
    """Run the real observer against copied source-native F5 code on CUDA."""
    repo = tmp_path / "repo"
    (repo / "research").mkdir(parents=True)
    shutil.copytree(REPO / "research/colab_cli", repo / "research/colab_cli")
    shutil.copytree(REPO / "research/bl_residual", repo / "research/bl_residual")
    shutil.copytree(REPO / "research/bl_ppo", repo / "research/bl_ppo")
    (repo / "src").symlink_to(REPO / "src", target_is_directory=True)
    root = tmp_path / "colab-root"
    (root / "repo").mkdir(parents=True)
    shutil.copytree(repo / "research", root / "repo/research")
    (root / "repo/src").symlink_to(REPO / "src", target_is_directory=True)
    env = os.environ.copy()
    env.update(
        {
            "COLAB_CLI_ROOT": str(root),
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": F5_PACKAGE,
        }
    )
    script = """
import json
import torch
from research.colab_cli import canary_observer
from research.bl_residual import f5_qualification
from src.model.ego2s_network import Ego2sNet

canary_observer.install()
net = Ego2sNet(local_channels=7, n_scalars=60, n_out=6).cuda().eval()
obs, mask = f5_qualification._native_observation(3403)
q, action = f5_qualification._select_cuda(net, obs, mask)
assert q.shape == (6,)
assert 0 <= action < 6
print(json.dumps({"action": action, "shape": list(q.shape)}))
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root / "repo",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    records = canary_observer.collect(root, F5_PACKAGE)
    assert records
    assert any(row["operation"] == "Tensor.argmax" for row in records)
    assert all(row["shape"][-1] == 6 for row in records)
    assert all(row["caller"].startswith(F5_PACKAGE + "/") for row in records)
