"""Custody checks use opaque payloads and reject corruption before import."""

import hashlib
import json
from pathlib import Path

import pytest

from research.watch_lead_p.closeout_09e.audit import read_metadata, triplet, verify_unit


def publish(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{checksum}  {path.name}\n")
    Path(str(path) + ".complete").write_text("complete\n")


def test_outcomes_are_never_deserialized(tmp_path: Path) -> None:
    path = tmp_path / "session-00.json"
    # Deliberately not JSON: verification concerns custody, never outcome content.
    publish(path, b"opaque sealed scientific payload")
    ledger = {"namespace": "test/", "allocation": "frozen", "construction": {"v": 2}}
    metadata = {"unit_id": "test/frozen/00", "construction": {"v": 2}, **triplet(path)}
    publish(Path(str(path) + ".metadata.json"), json.dumps(metadata).encode())
    assert verify_unit(path, 0, ledger)["allocation_index"] == 0
    with pytest.raises(ValueError, match="outcome read prohibited"):
        read_metadata(path)
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_unit(path, 1, ledger)


@pytest.mark.parametrize("component", ["payload", "sha256", "complete", "missing", "symlink"])
def test_corrupt_or_partial_export_refused(tmp_path: Path, component: str) -> None:
    path = tmp_path / "session-00.json"
    publish(path, b"opaque sealed scientific payload")
    if component == "payload":
        path.write_bytes(b"altered")
    elif component == "sha256":
        Path(str(path) + ".sha256").write_text(f"{'0' * 64}  {path.name}\n")
    elif component == "complete":
        Path(str(path) + ".complete").write_text("partial\n")
    elif component == "missing":
        Path(str(path) + ".complete").unlink()
    else:
        other = tmp_path / "other"
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises(ValueError):
        triplet(path)
