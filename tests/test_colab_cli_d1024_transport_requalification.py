"""D1024 transport-repair acceptance package regressions (no live Colab)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from research.colab_cli import acceptance, supervisor
from tests import test_colab_cli_fault_properties as harness

_isolate_account = harness._isolate_account
pinned_repo = harness.pinned_repo

REPO = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _transport_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "broker"
    transport = repo / "research/colab_cli"
    transport.mkdir(parents=True)
    source = transport / "only.py"
    source.write_text("VALUE = 'qualified'\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Acceptance Test")
    _git(repo, "config", "user.email", "acceptance@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "qualified transport")
    return repo, {"research/colab_cli/only.py": hashlib.sha256(source.read_bytes()).hexdigest()}


def _passing_d1024_result(
    tmp_path: Path, sources: dict[str, str], commit: str, tree: str
) -> dict[str, object]:
    artifact = tmp_path / "final-artifact.bin"
    with artifact.open("wb") as stream:
        stream.truncate(acceptance.FINAL_ARTIFACT_BYTES)
    transfer = {
        "status": "PASS",
        "hardware": "L4",
        "profile": "final-only",
        "path": str(artifact),
        "sha256": acceptance.sha256_file(artifact),
        "exported_bytes": acceptance.FINAL_ARTIFACT_BYTES,
        "elapsed_seconds": 16,
        "bytes_per_second_lower_bound": acceptance.FINAL_ARTIFACT_BYTES // 16,
        "verified_on_mac": True,
    }
    gates = [
        {**gate, "status": "OBSERVED" if gate["required"] else "NOT_REQUIRED"}
        for gate in acceptance.review_gates()
    ]
    for gate in gates:
        if gate["id"] == "final-artifact-download":
            gate["evidence_record"] = transfer
        if gate["id"] == "cold-kernel-metadata-recovery":
            gate["evidence_record"] = {
                "status": "PASS",
                "arms": [
                    {
                        "arm": arm,
                        "metadata_recovery": {
                            "original_kernel_id": f"{arm}-kernel",
                            "recovered_kernel_id": f"{arm}-kernel",
                            "same_kernel_id": True,
                            "tracked_metadata_unchanged": True,
                        },
                        "first_upload_transfer": {
                            "remote": f"/content/{arm}-credential.json",
                            "attempts": 1,
                            "size_bytes": 128,
                            "elapsed_seconds": 1.0,
                            "status": "completed",
                        },
                    }
                    for arm in ("cpu", "L4")
                ],
            }
    inventory = acceptance.d1024_synthetic_inventory()
    output_root = tmp_path / "outputs"
    execution = []
    l4_receipt = None
    for arm in ("cpu", "l4"):
        output = output_root / arm
        output.mkdir(parents=True, exist_ok=True)
        receipt = output / "receipt.json"
        transfers = []
        if arm == "l4":
            transfers = [
                {
                    "direction": "upload",
                    "status": "verified",
                    "remote": f"/content/fake/inputs/{row['dest']}",
                    "size_bytes": row["size_bytes"],
                    "sha256": "a" * 64,
                }
                for row in inventory["inputs"]
                if row["transport"] == "upload"
            ]
        receipt.write_text(
            json.dumps(
                {
                    "commit": commit,
                    "stop_verified": True,
                    "cleanup_error": None,
                    "transfers": transfers,
                }
            )
        )
        execution.append(
            {
                "job": arm,
                "status": "PASS",
                "receipt": str(receipt),
                "receipt_sha256": acceptance.sha256_file(receipt),
            }
        )
        if arm == "l4":
            l4_receipt = receipt
    assert l4_receipt is not None
    measurement_path = l4_receipt.parent / "upload-measurement.json"
    measurement_path.write_text(
        json.dumps(
            {
                "elapsed_seconds": 250.0,
                "transfer_seconds": 149.0,
                "source_delivery_seconds": 21.0,
            }
        )
    )
    staging_inputs = [
        (
            {"dest": row["dest"], "transport": "https", "status": "manifest-bound"}
            if row["transport"] == "https"
            else {
                "dest": row["dest"],
                "transport": "upload",
                "status": "verified",
                "size_bytes": row["size_bytes"],
                "sha256": "a" * 64,
                "remote": f"/content/fake/inputs/{row['dest']}",
            }
        )
        for row in inventory["inputs"]
    ]
    gates.append(
        {
            "id": "d1024-synthetic-full-inventory-staging",
            "required": True,
            "status": "OBSERVED",
            "evidence_record": {
                "status": "PASS",
                "inventory_sha256": inventory["sha256"],
                "public_input_preflight": "/synthetic/preflight.json",
                "inputs": staging_inputs,
                "upload_measurement": {
                    "path": str(measurement_path),
                    "sha256": acceptance.sha256_file(measurement_path),
                    "elapsed_seconds": 250.0,
                    "transfer_seconds": 149.0,
                    "source_delivery_seconds": 21.0,
                },
                "receipt": {
                    "path": str(l4_receipt),
                    "sha256": acceptance.sha256_file(l4_receipt),
                },
            },
        }
    )
    return {
        "schema": "snake-dqn-colab-cli-acceptance-result/v1",
        "status": "PASS",
        "qualified": True,
        "return_code": 0,
        "profile": "final-only",
        "qualified_commit": commit,
        "qualified_tree_hash": tree,
        "source_manifest": sources,
        "d1024_synthetic_inventory_sha256": inventory["sha256"],
        "final_artifact_transfer": transfer,
        "final_plan": {
            "profile": "final-only",
            "status": "PASS",
            "commit": commit,
            "tree_hash": tree,
            "source_manifest": sources,
            "d1024_synthetic_inventory": inventory,
            "gates": gates,
            "execution": execution,
        },
    }


def _write_qualified_registry(
    tmp_path: Path, result: dict[str, object], commit: str, tree: str
) -> tuple[Path, Path]:
    result_path = tmp_path / "acceptance-result.json"
    result_path.write_text(json.dumps(result))
    registry_path = tmp_path / "qualified.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": acceptance.QUALIFIED_RECORD_SCHEMA,
                "qualified_commit": commit,
                "tree_hash": tree,
                "acceptance_result": str(result_path),
                "acceptance_result_sha256": acceptance.sha256_file(result_path),
            }
        )
    )
    return result_path, registry_path


def _refresh_l4_receipt_binding(result: dict[str, object]) -> None:
    plan = result["final_plan"]
    assert isinstance(plan, dict)
    execution = plan["execution"]
    assert isinstance(execution, list)
    l4 = next(row for row in execution if row["job"] == "l4")
    receipt = Path(l4["receipt"])
    digest = acceptance.sha256_file(receipt)
    l4["receipt_sha256"] = digest
    staging = next(
        row for row in plan["gates"] if row["id"] == "d1024-synthetic-full-inventory-staging"
    )
    staging["evidence_record"]["receipt"]["sha256"] = digest


def test_d1024_synthetic_plan_has_all_metadata_inventory_roles_and_sizes(tmp_path: Path) -> None:
    record = acceptance.build_plan(REPO, tmp_path / "state", d1024_inventory=True)
    plan = record["plan"]
    inventory = plan["d1024_synthetic_inventory"]
    assert inventory["sha256"] == acceptance.D1024_F4_INVENTORY_SHA256
    assert [(row["dest"], row["size_bytes"], row["transport"]) for row in inventory["inputs"]] == [
        ("b8.pth", 6_562_005, "https"),
        ("factory-verified-ledger.json", 2_405_362, "upload"),
        ("train-1024.npz", 45_397_643, "upload"),
        ("metadata.tar", 2_211_840, "upload"),
    ]
    manifest = json.loads(Path(plan["jobs"][0]["manifest"]).read_text())
    synthetic = {
        row["dest"]: row
        for row in manifest["inputs"]
        if row["dest"]
        in {"b8.pth", "factory-verified-ledger.json", "train-1024.npz", "metadata.tar"}
    }
    assert synthetic["b8.pth"]["url"] == inventory["inputs"][0]["url"]
    assert "path" not in synthetic["b8.pth"]
    for name in ("factory-verified-ledger.json", "train-1024.npz", "metadata.tar"):
        assert Path(synthetic[name]["path"]).stat().st_size == synthetic[name]["size_bytes"]


def test_source_bound_pass_record_refuses_hash_copy_and_rejected_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sources = _transport_repo(tmp_path)
    monkeypatch.setattr(acceptance, "source_manifest", lambda root: sources)
    commit = _git(repo, "rev-parse", "HEAD")
    tree = acceptance.transport_tree(repo)
    result = _passing_d1024_result(tmp_path, sources, commit, tree)
    result_path = tmp_path / "acceptance-result.json"
    result_path.write_text(json.dumps(result))
    registry = {
        "schema": acceptance.QUALIFIED_RECORD_SCHEMA,
        "qualified_commit": commit,
        "tree_hash": tree,
        "acceptance_result": str(result_path),
        "acceptance_result_sha256": acceptance.sha256_file(result_path),
    }
    registry_path = tmp_path / "qualified.json"
    registry_path.write_text(json.dumps(registry))
    verified = acceptance.verify_qualified_transport_record(repo, registry_path)
    assert verified["tree_hash"] == tree
    with pytest.raises(ValueError, match="rejected tree"):
        acceptance.verify_qualified_transport_record(repo, registry_path, reject_tree=tree)
    result["final_plan"]["gates"].pop()
    result_path.write_text(json.dumps(result))
    registry["acceptance_result_sha256"] = acceptance.sha256_file(result_path)
    registry_path.write_text(json.dumps(registry))
    with pytest.raises(Exception, match="gate|inventory"):
        acceptance.verify_qualified_transport_record(repo, registry_path)
    result = _passing_d1024_result(tmp_path, sources, commit, tree)
    result_path.write_text(json.dumps({**result, "qualified": False}))
    with pytest.raises(ValueError, match="hash differs"):
        acceptance.verify_qualified_transport_record(repo, registry_path)


@pytest.mark.parametrize("arm", ["cpu", "l4"])
def test_qualified_transport_record_rejects_receipt_from_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    repo, sources = _transport_repo(tmp_path)
    monkeypatch.setattr(acceptance, "source_manifest", lambda root: sources)
    commit = _git(repo, "rev-parse", "HEAD")
    tree = acceptance.transport_tree(repo)
    result = _passing_d1024_result(tmp_path, sources, commit, tree)
    plan = result["final_plan"]
    assert isinstance(plan, dict)
    execution = next(row for row in plan["execution"] if row["job"] == arm)
    receipt_path = Path(execution["receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["commit"] = "0" * 40
    receipt_path.write_text(json.dumps(receipt))
    _refresh_l4_receipt_binding(result)
    if arm == "cpu":
        execution["receipt_sha256"] = acceptance.sha256_file(receipt_path)
    _, registry_path = _write_qualified_registry(tmp_path, result, commit, tree)
    with pytest.raises(ValueError, match=f"{arm} receipt is not from the qualified commit"):
        acceptance.verify_qualified_transport_record(repo, registry_path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", "duplicate or missing L4 upload receipt"),
        ("duplicate", "duplicate or missing L4 upload receipt"),
        ("conflict", "conflicts with L4 receipt"),
    ],
)
def test_qualified_transport_record_binds_synthetic_upload_rows_to_l4_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, message: str
) -> None:
    repo, sources = _transport_repo(tmp_path)
    monkeypatch.setattr(acceptance, "source_manifest", lambda root: sources)
    commit = _git(repo, "rev-parse", "HEAD")
    tree = acceptance.transport_tree(repo)
    result = _passing_d1024_result(tmp_path, sources, commit, tree)
    plan = result["final_plan"]
    assert isinstance(plan, dict)
    l4 = next(row for row in plan["execution"] if row["job"] == "l4")
    receipt_path = Path(l4["receipt"])
    receipt = json.loads(receipt_path.read_text())
    if mutation == "missing":
        receipt["transfers"].pop()
    elif mutation == "duplicate":
        receipt["transfers"].append(dict(receipt["transfers"][0]))
    else:
        receipt["transfers"][0]["sha256"] = "b" * 64
    receipt_path.write_text(json.dumps(receipt))
    _refresh_l4_receipt_binding(result)
    _, registry_path = _write_qualified_registry(tmp_path, result, commit, tree)
    with pytest.raises(ValueError, match=message):
        acceptance.verify_qualified_transport_record(repo, registry_path)


def test_qualified_transport_record_rejects_measurement_not_adjacent_to_l4_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sources = _transport_repo(tmp_path)
    monkeypatch.setattr(acceptance, "source_manifest", lambda root: sources)
    commit = _git(repo, "rev-parse", "HEAD")
    tree = acceptance.transport_tree(repo)
    result = _passing_d1024_result(tmp_path, sources, commit, tree)
    plan = result["final_plan"]
    assert isinstance(plan, dict)
    staging = next(
        row for row in plan["gates"] if row["id"] == "d1024-synthetic-full-inventory-staging"
    )
    measurement = staging["evidence_record"]["upload_measurement"]
    original = Path(measurement["path"])
    unrelated = tmp_path / "unrelated" / "upload-measurement.json"
    unrelated.parent.mkdir()
    unrelated.write_bytes(original.read_bytes())
    measurement["path"] = str(unrelated)
    measurement["sha256"] = acceptance.sha256_file(unrelated)
    _, registry_path = _write_qualified_registry(tmp_path, result, commit, tree)
    with pytest.raises(ValueError, match="timing is not from the L4 receipt"):
        acceptance.verify_qualified_transport_record(repo, registry_path)


def test_d1024_synthetic_staging_requires_verified_local_hashes_and_public_role(
    tmp_path: Path,
) -> None:
    record = acceptance.build_plan(REPO, tmp_path / "state", d1024_inventory=True)
    plan = record["plan"]
    job = next(row for row in plan["jobs"] if row["name"] == "l4")
    manifest = json.loads(Path(job["manifest"]).read_text())
    output = Path(manifest["output_dir"])
    output.mkdir(parents=True)
    (output / "public-input-preflight.json").write_text("{}")
    (output / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 250.0,
                "transfer_seconds": 149.0,
                "source_delivery_seconds": 21.0,
            }
        )
    )
    transfers = []
    for row in manifest["inputs"]:
        if row["dest"] not in {
            "factory-verified-ledger.json",
            "train-1024.npz",
            "metadata.tar",
        }:
            continue
        transfers.append(
            {
                "direction": "upload",
                "status": "verified",
                "remote": f"/content/fake/inputs/{row['dest']}",
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
        )
    receipt = {"transfers": transfers}
    (output / "receipt.json").write_text(json.dumps(receipt))
    evidence = acceptance.d1024_synthetic_staging_evidence(plan, job, receipt, output)
    assert evidence["status"] == "PASS"
    original_digest = transfers[0]["sha256"]
    transfers[0]["sha256"] = "0" * 64
    (output / "receipt.json").write_text(json.dumps({"transfers": transfers}))
    with pytest.raises(Exception, match="verified upload"):
        acceptance.d1024_synthetic_staging_evidence(plan, job, {"transfers": transfers}, output)
    transfers[0]["sha256"] = original_digest
    (output / "receipt.json").write_text(json.dumps({"transfers": transfers}))
    (output / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 250.0,
                "transfer_seconds": 170.0,
                "source_delivery_seconds": 21.0,
            }
        )
    )
    with pytest.raises(Exception, match="150-second"):
        acceptance.d1024_synthetic_staging_evidence(plan, job, {"transfers": transfers}, output)


def test_d1024_synthetic_inventory_is_a_required_live_acceptance_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pinned_repo: Path,
    _isolate_account: harness._TransportClock,
) -> None:
    """The fake L4 watchdog exception must still record the synthetic gate."""
    public = b"b" * 6_562_005
    public_path = tmp_path / "synthetic-public-b8.bin"
    public_path.write_bytes(public)
    inventory = json.loads(json.dumps(acceptance.D1024_SYNTHETIC_INVENTORY))
    inventory["inputs"][0]["sha256"] = hashlib.sha256(public).hexdigest()
    monkeypatch.setattr(acceptance, "D1024_SYNTHETIC_INVENTORY", inventory)
    monkeypatch.setattr(
        supervisor,
        "input_preflight",
        lambda manifest: [{"dest": "b8.pth", "status": "verified-synthetic"}],
    )
    monkeypatch.setenv("FAKE_COLAB_PUBLIC_INPUT_PATH", str(public_path))
    binary = harness._fake_env(monkeypatch, tmp_path, {"seed": 0, "max_latency_seconds": 0})
    plan = acceptance.build_plan(
        pinned_repo, tmp_path / "acceptance", None, profile="final-only", d1024_inventory=True
    )["plan"]
    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab=binary,
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python=str(harness.HELPER),
    )
    assert rc == 0
    result = json.loads(
        (Path(plan["jobs"][0]["manifest"]).parent.parent / "acceptance-result.json").read_text()
    )
    gate = next(
        row
        for row in result["final_plan"]["gates"]
        if row["id"] == "d1024-synthetic-full-inventory-staging"
    )
    assert gate["status"] == "OBSERVED"
    assert gate["evidence_record"]["status"] == "PASS"
    registry_path = tmp_path / "qualified-transport.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": acceptance.QUALIFIED_RECORD_SCHEMA,
                "qualified_commit": plan["commit"],
                "tree_hash": plan["tree_hash"],
                "acceptance_result": str(
                    Path(plan["jobs"][0]["manifest"]).parent.parent / "acceptance-result.json"
                ),
                "acceptance_result_sha256": acceptance.sha256_file(
                    Path(plan["jobs"][0]["manifest"]).parent.parent / "acceptance-result.json"
                ),
            }
        )
    )
    verified = acceptance.verify_qualified_transport_record(
        pinned_repo,
        registry_path,
        required_inventory_sha256=plan["d1024_synthetic_inventory"]["sha256"],
    )
    assert verified["qualified_commit"] == plan["commit"]
    harness._assert_safe(tmp_path, plan)
