"""Prove bundled transport preserves the frozen roots and refuses bad inputs."""

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from research.bl_served_search import fixture_bundle, ruler
from research.bl_served_search.exports import digest, export_json
from research.bl_served_search.readiness import fixture_bank
from research.bl_served_search.test_ruler import _fixtures


def _exports(directory: Path) -> None:
    ruler.export_fixtures(directory, {**_fixtures(), "source_identity": ruler.source_identity()})


def test_bundle_roundtrip_is_byte_identical_deterministic_and_create_only(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _exports(original)
    first = fixture_bundle.pack(original, tmp_path / "first.tar.gz")
    second = fixture_bundle.pack(original, tmp_path / "second.tar.gz")
    assert first.read_bytes() == second.read_bytes()
    target = tmp_path / "restored"
    manifest = fixture_bundle.unpack(first, target, digest(first))
    assert manifest == target / "ruler-fixtures.json"
    expected = {path.name: path.read_bytes() for path in original.iterdir()}
    assert {path.name: path.read_bytes() for path in target.iterdir()} == expected
    with pytest.raises(FileExistsError):
        fixture_bundle.unpack(first, target, digest(first))


def test_bundle_hash_mismatch_never_publishes_fixtures(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _exports(original)
    bundle = fixture_bundle.pack(original, tmp_path / "fixtures.tar.gz")
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="size/hash"):
        fixture_bundle.unpack(bundle, target, "0" * 64)
    assert not target.exists()


def test_l2_readiness_reads_only_hash_bound_bundled_fixtures(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _exports(original)
    bundle = fixture_bundle.pack(original, tmp_path / fixture_bundle.BUNDLE_NAME)
    hashes = {fixture_bundle.BUNDLE_NAME: digest(bundle)}
    assert fixture_bank(tmp_path, hashes) == json.loads(
        (original / "ruler-fixtures.json").read_text()
    )
    assert not (tmp_path / "ruler-fixtures").exists()
    with pytest.raises(ValueError, match="size/hash"):
        fixture_bank(tmp_path, {fixture_bundle.BUNDLE_NAME: "0" * 64})
    with pytest.raises(ValueError, match="must be hash-bound"):
        fixture_bank(tmp_path, {})


def test_checksums_cannot_hide_a_root_disagreement(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _exports(original)
    root_path = next(original.glob("*.root.json"))
    value = json.loads(root_path.read_text())
    value["root_sha256"] = "0" * 64
    for suffix in ("", ".sha256", ".complete"):
        Path(str(root_path) + suffix).unlink()
    export_json(root_path, value)
    with pytest.raises(ValueError, match="disagrees"):
        fixture_bundle.pack(original, tmp_path / "fixtures.tar.gz")


@pytest.mark.parametrize("unsafe", ["../escaped", "/absolute", "link", "duplicate"])
def test_unsafe_archive_members_never_publish(tmp_path: Path, unsafe: str) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for index in range(30):
            name = unsafe if index == 0 else f"file-{index}"
            if unsafe == "duplicate":
                name = "same"
            member = tarfile.TarInfo(name)
            if unsafe == "link" and index == 0:
                member.type = tarfile.SYMTYPE
                member.linkname = "../escaped"
            archive.addfile(member, io.BytesIO())
    bundle = tmp_path / "bad.tar.gz"
    bundle.write_bytes(gzip.compress(stream.getvalue()))
    target = tmp_path / "restored"
    with pytest.raises((ValueError, FileExistsError)):
        fixture_bundle.unpack(bundle, target, digest(bundle))
    assert not target.exists()
    assert not (tmp_path / "escaped").exists()


def test_decompression_is_bounded_before_tar_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixture_bundle, "MAX_UNPACKED_BYTES", 1024)
    bundle = tmp_path / "large.tar.gz"
    bundle.write_bytes(gzip.compress(b"x" * 1025))
    with pytest.raises(ValueError, match="unpacked byte limit"):
        fixture_bundle.unpack(bundle, tmp_path / "restored", digest(bundle))


def test_sparse_logical_size_cannot_escape_decompression_limit(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        sparse = tarfile.TarInfo("sparse")
        sparse.size = 1
        sparse.pax_headers = {
            "GNU.sparse.map": "0,1",
            "GNU.sparse.size": str(fixture_bundle.MAX_UNPACKED_BYTES + 1),
        }
        archive.addfile(sparse, io.BytesIO(b"x"))
        for index in range(29):
            archive.addfile(tarfile.TarInfo(f"file-{index}"), io.BytesIO())
    bundle = tmp_path / "sparse.tar.gz"
    bundle.write_bytes(gzip.compress(stream.getvalue()))
    assert len(stream.getvalue()) < fixture_bundle.MAX_UNPACKED_BYTES
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="extracted byte limit|unsafe"):
        fixture_bundle.unpack(bundle, target, digest(bundle))
    assert not target.exists()
