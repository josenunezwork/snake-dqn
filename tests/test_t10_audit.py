"""Artifact integrity gates fail closed without running numerical diagnostics."""

import hashlib

import pytest

pytest.importorskip("sklearn")

from research.t10_certificate_horizon_20261017.audit import verify_hashes  # noqa: E402


def test_hash_audit_detects_changed_and_missing_artifacts(tmp_path) -> None:
    artifact = tmp_path / "receipt.json"
    artifact.write_bytes(b'{"complete": true}\n')
    expected = {"receipt.json": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    assert verify_hashes(expected, tmp_path) == 1
    artifact.write_bytes(b'{"complete": false}\n')
    with pytest.raises(AssertionError, match="hash mismatch"):
        verify_hashes(expected, tmp_path)
    artifact.unlink()
    with pytest.raises(FileNotFoundError):
        verify_hashes(expected, tmp_path)
