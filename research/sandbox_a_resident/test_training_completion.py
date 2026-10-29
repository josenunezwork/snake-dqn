"""Admission verifies actual uploaded snapshot bytes without Torch decoding."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from research.sandbox_a_resident import training_preflight
from research.sandbox_a_resident.training_completion import prepare_completion
from research.sandbox_a_resident.training_preflight import validate_packet, validate_uploaded_ledger
from research.sandbox_a_resident.training_resume import build_ledger
from research.sandbox_a_resident.training_units import allocation, publish_snapshot
from research.sandbox_a_resident.units import publish
from tools.sandbox import preflight as sandbox_preflight
from tools.sandbox.registry import digest, validate_input


def test_relative_resume_uploads_are_verified_on_mac_and_foreign_inputs_refused(tmp_path):
    control = {"source_commit": "a" * 40, "outer_iterations": 4, "snapshot_interval": 1}
    output = tmp_path / "exports"
    publish_snapshot(
        output,
        1,
        lambda path: path.write_bytes(b"opaque full state"),
        metadata={
            "allocation": allocation(control),
            "source_commit": control["source_commit"],
        },
    )
    publish(output / "iteration-0001-actor.pth", lambda path: path.write_bytes(b"opaque actor"))
    ledger, uploads, _ = build_ledger(output, control)
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "verified-ledger.json").write_text(json.dumps(ledger))
    validate_uploaded_ledger(packet, {"inputs": uploads}, control)
    with pytest.raises(ValueError, match="sidecars"):
        validate_uploaded_ledger(packet, {"inputs": uploads[:-1]}, control)
    foreign = {
        "path": str(Path(uploads[0]["path"])),
        "dest": "resume/foreign.pt",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="unreferenced"):
        validate_uploaded_ledger(packet, {"inputs": uploads + [foreign]}, control)


def test_prepare_completion_keeps_required_sidecars_and_preflights_successor(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Exercise the create-only completion writer with its TRAIN-01 preflight.

    The ordinary sandbox preflight binds source bytes to a committed pin.  This
    fixture keeps that expensive external source check out of scope while using
    the successor's actual sandbox/dispatch/ledger records and all local
    completion-sidecar validation.
    """
    root = Path(__file__).parent
    repo = root.parents[1]
    work = Path(tempfile.mkdtemp(prefix=".test-train-completion-", dir=repo))
    request.addfinalizer(lambda: shutil.rmtree(work, ignore_errors=True))
    original = work / "original"
    shutil.copytree(root / "packet-train01", original)
    # Keep the generic preflight's real source-byte binding in scope without
    # coupling this completion fixture to every changed training source file.
    # These source bytes are unchanged at the frozen packet pin.
    original_manifest_path = original / "dispatch-manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text())
    original_manifest["source_paths"] = ["src/core/game_config.py"]
    primary = repo.parent / "snake-dqn"
    source_paths = {
        "b8-s3.pth": str(primary / "saved_snakes/b8_s3_t7_20261016.pth"),
        "frp3-s12.pth": str(primary / "saved_snakes/frp3_m3_s12_u60000_20261005.pth"),
        "champion.pth": str(primary / "saved_snakes/champion_a5_freespace_20260621.pth"),
        "registry.json": str(
            Path(__file__).parents[2] / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
        ),
        "verified-ledger.json": str(original / "verified-ledger.json"),
    }
    for row in original_manifest["inputs"]:
        row["path"] = source_paths[row["dest"]]
    original_manifest_path.write_text(
        json.dumps(original_manifest, indent=2, sort_keys=True) + "\n"
    )
    control = json.loads((original / "control.json").read_text())
    exports = work / "exports"
    publish_snapshot(
        exports,
        16,
        lambda path: path.write_bytes(b"opaque full state"),
        metadata={"allocation": allocation(control), "source_commit": control["source_commit"]},
    )
    publish(exports / "iteration-0016-actor.pth", lambda path: path.write_bytes(b"opaque actor"))
    manifest = json.loads(original_manifest_path.read_text())
    executed = work / "executed" / "manifest.json"
    executed.parent.mkdir()
    executed.write_text(json.dumps(manifest, sort_keys=True))
    receipt = executed.parent / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "job_id": manifest["job_id"],
                "commit": manifest["commit"],
                "terminal": None,
                "generation": "fixture-generation",
                "manifest_sha256": digest(executed),
                "status": "INCOMPLETE",
                "failure_class": "RPC_EXEC_POLL_TIMEOUT",
                "stop_verified": True,
                "cleanup_error": None,
            }
        )
    )
    successor = work / "successor"
    output = work / "successor-output"
    prepare_completion(
        original,
        receipt,
        exports,
        successor,
        index=1,
        job_id="sandbox-a-train01-fixture-completion-1",
        output=output,
    )

    sandbox = json.loads((successor / "sandbox.json").read_text())
    resume = [
        row
        for row in sandbox["inputs"]
        if isinstance(row.get("path"), str) and row["path"].startswith(str(exports))
    ]
    assert len(resume) == 9
    assert sum(row["kind"] == "training-export" for row in resume) == 3
    assert sum(row["kind"] == "training-export-sidecar" for row in resume) == 6
    for row in resume:
        validate_input(row, {})

    # The actual generic admission checks manifest kinds, sidecar sibling
    # integrity and exact broker-upload equality.  Only tracked-file guards
    # are patched because this deliberately create-only temporary packet is
    # not itself committed.
    monkeypatch.setattr(sandbox_preflight, "verify_committed_manifest", lambda *_args, **_kw: None)
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _path: None)
    assert validate_packet(successor)["status"] == "PASS"


def test_validate_packet_admits_independent_seed_domain_and_freezes_recipe(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    from dataclasses import asdict

    from research.sandbox_a_resident.train import TrainControl
    from research.sandbox_a_resident.training_prepare import STUDY, seed_allocations
    from research.sandbox_a_resident.training_units import empty_ledger

    root = Path(__file__).parent
    repo = root.parents[1]
    work = Path(tempfile.mkdtemp(prefix=".test-train-domain-", dir=repo))
    request.addfinalizer(lambda: shutil.rmtree(work, ignore_errors=True))
    packet = work / "packet"
    shutil.copytree(root / "packet-train01", packet)
    broker_path = packet / "dispatch-manifest.json"
    broker = json.loads(broker_path.read_text())
    broker["source_paths"] = ["src/core/game_config.py"]
    control = asdict(
        TrainControl(
            study=STUDY,
            source_commit=broker["commit"],
            allocation_index=1,
            train_seed_offset=1_000_000_000,
        )
    )
    (packet / "control.json").write_text(json.dumps(control, indent=2, sort_keys=True) + "\n")
    ledger = empty_ledger(control)
    ledger_path = packet / "verified-ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    allocations = seed_allocations(control)
    registry_path = repo / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
    check = {
        "schema": "sandbox-a-train-seed-audit/v1",
        "status": "PASS",
        "study": STUDY,
        "allocations": allocations,
        "count": sum(row["count"] for row in allocations),
        "canonical_uint64_sha256": "b" * 64,
        "registry_sha256": digest(registry_path),
        "effective_seed_bits": 63,
        "aliases": 0,
        "excluded_intersection": 0,
    }
    check_path = packet / "registry-check.json"
    check_path.write_text(json.dumps(check, indent=2, sort_keys=True) + "\n")
    primary = repo.parent / "snake-dqn"
    source_paths = {
        "b8-s3.pth": str(primary / "saved_snakes/b8_s3_t7_20261016.pth"),
        "frp3-s12.pth": str(primary / "saved_snakes/frp3_m3_s12_u60000_20261005.pth"),
        "champion.pth": str(primary / "saved_snakes/champion_a5_freespace_20261005.pth"),
    }
    source_paths["champion.pth"] = str(primary / "saved_snakes/champion_a5_freespace_20260621.pth")
    source_paths["registry.json"] = str(registry_path)
    source_paths["verified-ledger.json"] = str(ledger_path)
    for row in broker["inputs"]:
        row["path"] = source_paths[row["dest"]]
        row["sha256"] = digest(Path(row["path"]))
    broker["env"]["SANDBOX_A_CONTROL_JSON"] = json.dumps(control, sort_keys=True)
    broker["sandbox_a"] = {
        "seed_allocations": allocations,
        "seed_audit_sha256": digest(check_path),
    }
    broker["sandbox_admission"]["manifest_path"] = str(packet / "sandbox.json")
    broker_path.write_text(json.dumps(broker, indent=2, sort_keys=True) + "\n")
    sandbox = json.loads((packet / "sandbox.json").read_text())
    roots = [row["offset"] for row in allocations]
    from tools.sandbox.registry import world_seed

    sandbox["inputs"] = [
        {
            "kind": "worlds",
            "study": STUDY,
            "indices": roots,
            "seeds": [world_seed(STUDY, i) for i in roots],
        }
    ] + [
        {
            "kind": (
                "frozen-checkpoint"
                if row["dest"] == "b8-s3.pth"
                else (
                    "registry"
                    if row["dest"] == "registry.json"
                    else (
                        "verified-ledger"
                        if row["dest"] == "verified-ledger.json"
                        else "frozen-opponent"
                    )
                )
            ),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in broker["inputs"]
    ]
    sandbox["outputs"]["broker_manifest"] = {
        "path": str(broker_path),
        "sha256": digest(broker_path),
    }
    (packet / "sandbox.json").write_text(
        "\n".join(
            ("{" if i == 0 else " ")
            + json.dumps(k)
            + ": "
            + json.dumps(v)
            + ("," if i < 9 else "}")
            for i, (k, v) in enumerate(sandbox.items())
        )
        + "\n"
    )
    monkeypatch.setattr(sandbox_preflight, "verify_committed_manifest", lambda *_args, **_kw: None)
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _path: None)
    assert validate_packet(packet)["status"] == "PASS"

    tampered = dict(control, proxy_worlds_per_mix=64)
    (packet / "control.json").write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="frozen production recipe"):
        validate_packet(packet)
