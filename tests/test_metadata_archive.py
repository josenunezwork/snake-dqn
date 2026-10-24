"""Fail-closed package metadata staging; no transport or research execution."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from research import metadata_archive as bundle
from research.bl_ppo.exports import export_json, verify_export


def _inputs(root: Path) -> list[dict]:
    return [
        {"path": str(p), "dest": p.relative_to(root).as_posix(), "sha256": bundle.sha256(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def _archive(tmp_path: Path) -> tuple[Path, list[dict]]:
    source = tmp_path / "source"
    export_json(source / "inputs/verified-ledger.json", {"namespace": "frozen", "units": []})
    (source / "fixture.bin").write_bytes(b"\x00exact\xff")
    rows = _inputs(source)
    archive = tmp_path / "metadata.tar"
    assert len(bundle.bundle_inputs(rows, archive)) == 1
    return archive, rows


def _rewrite(path: Path, mutation: str) -> Path:
    with tarfile.open(path) as tar:
        members = [(copy.copy(m), tar.extractfile(m).read()) for m in tar.getmembers()]
    manifest = json.loads(members[0][1])
    member, data = members[-1]
    if mutation == "digest":
        data = b"x" * len(data)
    elif mutation == "size":
        manifest["files"][-1]["size"] += 1
    elif mutation == "duplicate":
        members.append((copy.copy(member), data))
    elif mutation == "extra":
        extra = tarfile.TarInfo("foreign.json")
        members.append((extra, b"extra"))
    elif mutation == "missing":
        members.pop()
    elif mutation in {"traversal", "absolute", "normalized", "backslash"}:
        name = {
            "traversal": "../escape",
            "absolute": "/escape",
            "normalized": "a//b",
            "backslash": "a\\b",
        }[mutation]
        member.name = name
        manifest["files"][-1]["path"] = name
    elif mutation in {"symlink", "hardlink", "directory", "fifo"}:
        member.type = {
            "symlink": tarfile.SYMTYPE,
            "hardlink": tarfile.LNKTYPE,
            "directory": tarfile.DIRTYPE,
            "fifo": tarfile.FIFOTYPE,
        }[mutation]
        member.linkname = "/tmp/foreign"
    elif mutation == "foreign-schema":
        manifest["schema"] = "foreign"
    elif mutation in {"bad-marker", "bad-sidecar"}:
        suffix = ".complete" if mutation == "bad-marker" else ".sha256"
        for index, (item, _) in enumerate(members):
            if item.name.endswith(suffix):
                payload = b"invalid\n"
                members[index] = item, payload
                for row in manifest["files"]:
                    if row["path"] == item.name:
                        row.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                break
    else:
        raise AssertionError(mutation)
    if mutation == "digest":
        members[-1] = member, data
    members[0] = members[0][0], json.dumps(manifest).encode()
    output = path.with_name(mutation + ".tar")
    with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as tar:
        for member, payload in members:
            member.size = len(payload) if member.isfile() else 0
            tar.addfile(member, io.BytesIO(payload) if member.isfile() else None)
    return output


def test_nested_bytes_sidecars_and_deterministic_archive(tmp_path: Path) -> None:
    archive, rows = _archive(tmp_path)
    other = tmp_path / "other.tar"
    bundle.bundle_inputs(list(reversed(rows)), other)
    assert archive.read_bytes() == other.read_bytes()
    root = tmp_path / "vm"
    root.mkdir()
    result = bundle.unpack(archive, root, bundle.sha256(archive))
    assert result["files"] == 4
    for row in rows:
        assert (root / row["dest"]).read_bytes() == Path(row["path"]).read_bytes()
    verify_export(root / "inputs/verified-ledger.json")
    with pytest.raises(FileExistsError):
        bundle.unpack(archive, root, bundle.sha256(archive))
    with pytest.raises(FileExistsError):
        bundle.bundle_inputs(rows, archive)


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "size",
        "duplicate",
        "extra",
        "missing",
        "traversal",
        "absolute",
        "normalized",
        "backslash",
        "symlink",
        "hardlink",
        "directory",
        "fifo",
        "foreign-schema",
        "bad-marker",
        "bad-sidecar",
    ],
)
def test_invalid_tar_publishes_nothing(tmp_path: Path, mutation: str) -> None:
    archive, _ = _archive(tmp_path)
    bad = _rewrite(archive, mutation)
    root = tmp_path / "vm"
    root.mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        bundle.unpack(bad, root, bundle.sha256(bad))
    assert list(root.iterdir()) == []


def test_outer_digest_and_conflict_refusal(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    root = tmp_path / "vm"
    root.mkdir()
    with pytest.raises(ValueError, match="archive hash"):
        bundle.unpack(archive, root, "0" * 64)
    target = root / "fixture.bin"
    target.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        bundle.unpack(archive, root, bundle.sha256(archive))
    assert target.read_bytes() == b"preserve"
    assert sorted(p.name for p in root.iterdir()) == ["fixture.bin"]


def test_large_payload_and_public_catalog_stay_direct(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "model.bin"
    payload.write_bytes(b"x" * (bundle.MAX_FILE_BYTES + 1))
    Path(str(payload) + ".sha256").write_text(f"{bundle.sha256(payload)}  model.bin\n")
    Path(str(payload) + ".complete").write_bytes(b"complete\n")
    (source / "PUBLIC_INPUTS.json").write_bytes(b"[]\n")
    direct_url = {"url": "https://example.invalid/frozen", "dest": "remote.pth", "sha256": "f" * 64}
    rows = _inputs(source)
    for row in rows:
        if row["dest"] == "PUBLIC_INPUTS.json":
            row["dest"] = "public-inputs.json"
    archive = tmp_path / "metadata.tar"
    result = bundle.bundle_inputs([*rows, direct_url], archive)
    assert [r["dest"] for r in result] == [
        "public-inputs.json",
        "model.bin",
        "remote.pth",
        "metadata.tar",
    ]
    root = tmp_path / "vm"
    root.mkdir()
    (root / "model.bin").write_bytes(payload.read_bytes())
    bundle.unpack(archive, root, bundle.sha256(archive))
    verify_export(root / "model.bin")
    root2 = tmp_path / "vm-missing"
    root2.mkdir()
    with pytest.raises(ValueError, match="missing or unsafe"):
        bundle.unpack(archive, root2, bundle.sha256(archive))
    assert not list(root2.iterdir())


@pytest.mark.parametrize("kind", ["source", "destination", "parent"])
def test_symlink_refusal(tmp_path: Path, kind: str) -> None:
    archive, rows = _archive(tmp_path)
    root = tmp_path / "vm"
    root.mkdir()
    if kind == "source":
        original = Path(rows[0]["path"])
        link = tmp_path / "link"
        link.symlink_to(original)
        rows[0]["path"] = str(link)
        with pytest.raises(ValueError, match="hash/type"):
            bundle.bundle_inputs(rows, tmp_path / "other.tar")
    else:
        (root / ("fixture.bin" if kind == "destination" else "inputs")).symlink_to(
            tmp_path / "foreign"
        )
        with pytest.raises(ValueError, match="symlink"):
            bundle.unpack(archive, root, bundle.sha256(archive))


def test_duplicate_and_prefix_destinations_refused(tmp_path: Path) -> None:
    _, rows = _archive(tmp_path)
    for names in (["same", "same"], ["parent", "parent/child"]):
        candidates = [dict(rows[0], dest=n) for n in names]
        with pytest.raises(ValueError, match="duplicate|overlapping"):
            bundle.bundle_inputs(candidates, tmp_path / "bad.tar")


def test_reuse_requires_complete_identical_metadata_bytes(tmp_path: Path) -> None:
    archive, rows = _archive(tmp_path)
    expected = bundle.sha256(archive)
    assert bundle.bundle_inputs(rows, archive, reuse_verified=True)[-1]["sha256"] == expected
    payload = Path(rows[0]["path"])
    payload.write_bytes(b"changed")
    changed = [dict(row, sha256=bundle.sha256(Path(row["path"]))) for row in rows]
    with pytest.raises(ValueError, match="immutable metadata archive differs"):
        bundle.bundle_inputs(changed, archive, reuse_verified=True)
    assert bundle.sha256(archive) == expected
    Path(str(archive) + ".complete").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        bundle.bundle_inputs(changed, archive, reuse_verified=True)
