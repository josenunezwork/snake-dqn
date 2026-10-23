"""Lossless upload projection and prior-allocation accounting, one process only."""

import io
import json
import tarfile

import numpy as np
import pytest

from research.a100_a2 import make_pack, transfer
from research.bl_ppo.exports import digest


def shard(conflict=False):
    buffer = io.BytesIO()
    packed = np.asarray([[1, 2], [1, 2], [3, 4]], np.uint8)
    if conflict:
        packed[1, 0] = 5
    np.savez_compressed(
        buffer,
        source_decision=np.array([7, 7, 8], np.int64),
        state_packed=packed,
        state_scalars=np.array([[0.125], [0.125], [0.375]], np.float32),
        unused_terminal=np.ones((3, 100), np.float32),
    )
    return buffer.getvalue()


def test_projection_keeps_exact_used_bytes_and_order():
    source = shard()
    payload, counts = transfer.project_shard(source)
    assert counts == {"original_rows": 3, "unique_rows": 2}
    with np.load(io.BytesIO(source)) as old, np.load(io.BytesIO(payload)) as new:
        assert new.files == list(transfer.COLUMNS)
        for key in transfer.COLUMNS:
            assert new[key].dtype == old[key].dtype
            assert new[key].tobytes() == old[key][[0, 2]].tobytes()
    with pytest.raises(ValueError, match="conflicting duplicate"):
        transfer.project_shard(shard(conflict=True))


def test_compact_archive_preserves_original_members_and_rejects_foreign_source(tmp_path):
    root = tmp_path / "original"
    path = root / "t4e/banks/train/world-000001.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(shard())
    manifest = path.parent / "manifest.json"
    manifest.write_text(json.dumps({"shards": {path.name: digest(path)}}))
    a5 = root / "a5/world.json"
    a5.parent.mkdir()
    a5.write_bytes(b'{ "opaque": "preserve exact bytes" }\n')
    files = [
        {"path": str(p.relative_to(root)), "sha256": digest(p), "bytes": p.stat().st_size}
        for p in (path, manifest, a5)
    ]
    (root / "bundle.json").write_text(json.dumps({"schema": "a2-input-bundle/v1", "files": files}))
    source = tmp_path / "original.tar.gz"
    with tarfile.open(source, "x:gz") as archive:
        archive.add(root, arcname="labels")
    compact = tmp_path / "compact.tar.xz"
    receipt = transfer.compact_bundle(source, compact)
    assert receipt["source_archive_sha256"] == digest(source)
    assert transfer.verify_compact_archive(compact, digest(source))["shards"] == 1
    with pytest.raises(ValueError, match="source identity"):
        transfer.verify_compact_archive(compact, "f" * 64)
    unpack = tmp_path / "unpack"
    unpack.mkdir()
    target = transfer._safe_extract(compact, unpack)
    assert (target / "a5/world.json").read_bytes() == a5.read_bytes()
    projection = json.loads((target / "projection.json").read_text())
    projection["shards"] = {}
    (target / "projection.json").write_text(json.dumps(projection))
    with pytest.raises(ValueError, match="hash mismatch"):
        transfer.verify_projection(target)
    with pytest.raises(FileExistsError):
        transfer.compact_bundle(source, compact)


def prior(tmp_path, name, status, failure, elapsed, parent=None):
    root = tmp_path / name
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "a2": {
                    "allocation": "frozen",
                    "remaining_envelope": {"prior_canary_receipt": str(parent)} if parent else {},
                }
            }
        )
    )
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "job_id": "a100-a2-" + name,
                "status": status,
                "failure_class": failure,
                "stop_verified": True,
                "elapsed_seconds": elapsed,
                "exports": [],
                "error": "ModuleNotFoundError" if failure == "CHILD_NONZERO_EXIT" else None,
                "manifest_sha256": digest(manifest),
            }
        )
    )
    return receipt


def test_recovery_charges_all_allocations_and_refuses_missing_parent(tmp_path):
    first = prior(tmp_path, "canary-01", "INCOMPLETE", "CHILD_NONZERO_EXIT", 792.4886501249857)
    second = prior(tmp_path, "canary-02", "PASS", None, 527.8322866659728, first)
    main = prior(tmp_path, "main-02", "INCOMPLETE", "UPLOAD_CAP", 352.604429583007, first)
    bound = main.parent / "manifest.json"
    manifest = json.loads(bound.read_text())
    manifest["canary"] = {"receipt": {"path": str(second), "sha256": digest(second)}}
    bound.write_text(json.dumps(manifest))
    value = json.loads(main.read_text())
    value["manifest_sha256"] = digest(bound)
    main.write_text(json.dumps(value))
    envelope = make_pack.transfer_recovery_envelope([first, second, main])
    assert envelope["main_minutes"] == 242
    assert envelope["main_units"] == 40.35
    assert envelope["prior_charged_units_at_ceiling"] == pytest.approx(4.64701490659435)
    with pytest.raises(ValueError, match="missing earlier"):
        make_pack.transfer_recovery_envelope([second, main])
    with pytest.raises(ValueError, match="charged family canary"):
        make_pack.transfer_recovery_envelope([first, main])
    with pytest.raises(ValueError, match="duplicate"):
        make_pack.transfer_recovery_envelope([first, second, main, main])
    value = json.loads(main.read_text())
    value["stop_verified"] = False
    main.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="stopped mechanical"):
        make_pack.transfer_recovery_envelope([first, second, main])
