"""Safety and integrity coverage for the T16 Colab input bundle."""

import io
import json
import os
import tarfile

import pytest

from research.t16_colab import bundle


def test_export_then_verify_extract_round_trips_all_declared_inputs(tmp_path):
    source = tmp_path / "model.pth"
    source.write_bytes(b"weights")
    archive = tmp_path / "inputs.tar.gz"

    manifest = bundle.export_bundle(archive, {"init/model.pth": source})
    target = bundle.verify_and_extract(archive, tmp_path / "unpacked")

    assert manifest["schema"] == bundle.SCHEMA
    assert (target / "init/model.pth").read_bytes() == b"weights"
    extracted_manifest = json.loads((target / "manifest.json").read_text())
    assert extracted_manifest["files"][0]["path"] == "init/model.pth"


def test_export_is_reproducible_across_names_and_source_timestamps(tmp_path):
    source = tmp_path / "model.pth"
    source.write_bytes(b"unchanged weights")
    first, second = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    bundle.export_bundle(first, {"init/model.pth": source})
    os.utime(source, (12345, 12345))
    bundle.export_bundle(second, {"init/model.pth": source})
    assert first.read_bytes() == second.read_bytes()


def test_tampered_payload_is_never_extracted(tmp_path):
    source = tmp_path / "model.pth"
    source.write_bytes(b"expected")
    archive = tmp_path / "inputs.tar.gz"
    manifest = bundle.export_bundle(archive, {"init/model.pth": source})
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as out:
        info = tarfile.TarInfo(bundle.MANIFEST_NAME)
        info.size = len(manifest_bytes)
        out.addfile(info, io.BytesIO(manifest_bytes))
        info = tarfile.TarInfo(f"{bundle.ROOT}/init/model.pth")
        info.size = len(b"tampered")
        out.addfile(info, io.BytesIO(b"tampered"))

    destination = tmp_path / "unpacked"
    with pytest.raises(bundle.BundleError, match="SHA-256 mismatch"):
        bundle.verify_and_extract(tampered, destination)
    assert not (destination / bundle.ROOT).exists()


def test_unsafe_extra_member_is_rejected_before_extraction(tmp_path):
    source = tmp_path / "model.pth"
    source.write_bytes(b"weights")
    archive = tmp_path / "inputs.tar.gz"
    manifest = bundle.export_bundle(archive, {"init/model.pth": source})
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, "w:gz") as out:
        info = tarfile.TarInfo(bundle.MANIFEST_NAME)
        info.size = len(manifest_bytes)
        out.addfile(info, io.BytesIO(manifest_bytes))
        payload = b"weights"
        info = tarfile.TarInfo(f"{bundle.ROOT}/init/model.pth")
        info.size = len(payload)
        out.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo("../../outside")
        info.size = 1
        out.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(bundle.BundleError, match="not declared"):
        bundle.verify_and_extract(unsafe, tmp_path / "unpacked")
    assert not (tmp_path / "outside").exists()


def test_manifest_cannot_declare_a_traversal_path(tmp_path):
    payload = b"x"
    manifest = {
        "schema": bundle.SCHEMA,
        "files": [
            {
                "path": "../outside",
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
    }
    archive_path = tmp_path / "traversal.tar.gz"
    manifest_bytes = (json.dumps(manifest) + "\n").encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(bundle.MANIFEST_NAME)
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))

    with pytest.raises(bundle.BundleError, match="unsafe archive path"):
        bundle.verify_and_extract(archive_path, tmp_path / "unpacked")
