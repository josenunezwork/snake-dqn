"""Focused transport tests for compact E3 v2 input conversion."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from research.bl_info_value.compact import array_binding, compact, hydrate
from research.bl_info_value.io import export_bytes, sha256


def _archive(payloads: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as bundle:
            for name, payload in payloads:
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mtime = 0
                bundle.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def _npz() -> bytes:
    stream = io.BytesIO()
    np.savez(
        stream,
        **{
            "decision": np.array([11, 12, 13], dtype=np.int64),
            "action": np.array([1, 2, 3], dtype=np.int8),
            "frame": np.array([100, 101, 102], dtype=np.int64),
            "decisive": np.array([True, False, True]),
            "outcome_death_frame": np.array([130, 131, 132], dtype=np.int64),
            "local": np.arange(12, dtype=np.float32).reshape(3, 2, 2),
            "global": np.arange(6, dtype=np.int16).reshape(3, 2),
            "scalars": np.arange(9, dtype=np.float32).reshape(3, 3),
            "mask": np.array([[True, False], [False, True], [True, True]]),
        },
    )
    return stream.getvalue()


def _v1(tmp_path: Path) -> tuple[Path, str, dict, dict[str, np.ndarray]]:
    source = _npz()
    arrays = np.load(io.BytesIO(source), allow_pickle=False)
    checkpoints = {"checkpoints/a.pth": b"A", "checkpoints/b.pth": b"B"}
    catalog = {
        "schema": "bl-e3-input/v1",
        "smoke": True,
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "source_manifest_sha256": "f" * 64,
        "checkpoints": {
            name: hashlib.sha256(value).hexdigest() for name, value in checkpoints.items()
        },
        "files": {
            "worlds/w0.npz": hashlib.sha256(source).hexdigest(),
            **{name: hashlib.sha256(value).hexdigest() for name, value in checkpoints.items()},
        },
        "tasks": [
            {
                "key": "w0",
                "world": 0,
                "split": 0,
                "arm": "fixture",
                "plan": {"specs": [["checkpoint", "checkpoints/b.pth"]]},
                "checkpoint": "checkpoints/a.pth",
                "source_npz": "worlds/w0.npz",
                "indices": [2, 0],
                "death_decision": 99,
                "depth": 300,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "receipt_sha256": "e" * 64,
            }
        ],
    }
    payload = _archive(
        [("catalog.json", (json.dumps(catalog, sort_keys=True) + "\n").encode())]
        + [("worlds/w0.npz", source)]
        + list(checkpoints.items())
    )
    path = tmp_path / "v1.tar.gz"
    export_bytes(path, payload)
    return path, sha256(path), catalog, {name: np.array(arrays[name]) for name in arrays.files}


def test_compact_binds_original_order_and_hydrates_exact_checkpoints(tmp_path: Path) -> None:
    archive, digest, v1, arrays = _v1(tmp_path)
    catalog = compact(archive, tmp_path / "compact", digest)
    assert catalog["schema"] == "bl-e3-input/v2"
    assert catalog["source_archive_sha256"] == digest
    assert catalog["files"] == {}
    task = catalog["tasks"][0]
    assert task["indices"] == [2, 0]
    assert [row["decision"] for row in task["row_bindings"]] == [13, 11]
    assert task["row_bindings"][0]["outcome_death_frame"] == 132
    assert task["row_bindings"][0]["obs_hashes"]["local"] == array_binding(arrays["local"][2])
    assert task["source_sha256"] == v1["tasks"][0]["source_sha256"]
    compact_input = tmp_path / "compact" / "inputs.tar.gz"
    from research.bl_info_value.catalog import unpack

    extracted = tmp_path / "extracted"
    unpack(compact_input, extracted, sha256(compact_input))
    hydrate(catalog, extracted, tmp_path / "compact" / "checkpoints.tar.gz")
    assert (extracted / "checkpoints" / "a.pth").read_bytes() == b"A"
    assert (extracted / "checkpoints" / "b.pth").read_bytes() == b"B"


def test_compact_refuses_tampered_source_archive(tmp_path: Path) -> None:
    archive, digest, _, _ = _v1(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest mismatch"):
        compact(archive, tmp_path / "compact", digest)


@pytest.mark.parametrize("member", ["../escape.pth", "checkpoints/a.pth"])
def test_hydrate_rejects_unsafe_or_duplicate_member(tmp_path: Path, member: str) -> None:
    payload = _archive([("checkpoints/a.pth", b"A"), (member, b"B")])
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(payload)
    catalog = {
        "checkpoints": {"checkpoints/a.pth": hashlib.sha256(b"A").hexdigest()},
        "checkpoint_archive": {"sha256": sha256(archive), "size_bytes": archive.stat().st_size},
    }
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(ValueError, match="members differ"):
        hydrate(catalog, inputs, archive)


def test_hydrate_refuses_checkpoint_archive_size_or_digest_drift(tmp_path: Path) -> None:
    archive = tmp_path / "checkpoints.tar.gz"
    archive.write_bytes(_archive([("checkpoints/a.pth", b"A")]))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    catalog = {
        "checkpoints": {"checkpoints/a.pth": hashlib.sha256(b"A").hexdigest()},
        "checkpoint_archive": {"sha256": "0" * 64, "size_bytes": archive.stat().st_size},
    }
    with pytest.raises(ValueError, match="digest mismatch"):
        hydrate(catalog, inputs, archive)
    catalog["checkpoint_archive"] = {"sha256": sha256(archive), "size_bytes": 1}
    with pytest.raises(ValueError, match="size mismatch"):
        hydrate(catalog, inputs, archive)


def test_hydrate_refuses_member_digest_drift_after_archive_verification(tmp_path: Path) -> None:
    archive = tmp_path / "checkpoints.tar.gz"
    archive.write_bytes(_archive([("checkpoints/a.pth", b"changed")]))
    catalog = {
        "checkpoints": {"checkpoints/a.pth": hashlib.sha256(b"expected").hexdigest()},
        "checkpoint_archive": {"sha256": sha256(archive), "size_bytes": archive.stat().st_size},
    }
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        hydrate(catalog, inputs, archive)
    assert not (inputs / "checkpoints").exists()
