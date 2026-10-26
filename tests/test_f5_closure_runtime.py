"""Synthetic boundary tests for the F5 runtime recovery evidence tool."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "research/bl_residual/f5_r10a_closure_20261026/runtime_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("f5_runtime_recovery", MODULE_PATH)
assert SPEC and SPEC.loader
runtime_recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_recovery)


def test_receipt_binding_never_decodes_forbidden_receipt_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    forbidden = b'{"outcome_value":"DO_NOT_DECODE","terminal":{"mass":999}}\n'
    receipt.write_bytes(forbidden)
    digest = hashlib.sha256(forbidden).hexdigest()
    (tmp_path / "receipt.json.sha256").write_text(f"{digest}  receipt.json\n")
    original = Path.read_bytes

    def guard(path: Path) -> bytes:
        if path == receipt:
            raise AssertionError("receipt content must never be read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)

    binding = runtime_recovery.receipt_binding(receipt)

    assert binding == {
        "path": str(receipt),
        "sha256_from_sidecar": digest,
        "receipt_decoded": False,
    }
    assert "outcome_value" not in str(binding)


def test_roster_projection_rejects_missing_checkpoint_identity() -> None:
    control = {
        "allocation": [
            {
                "allocation_id": "synthetic-0",
                "allocation_index": 0,
                "seed": 7,
                "stream": "gate",
                "plan": {"specs": [["checkpoint", "input:a.pth"]], "member_sha256s": []},
            }
        ]
    }

    with pytest.raises(ValueError, match="ordered roster"):
        runtime_recovery.roster_rows(
            control,
            [{"kind": "checkpoint", "reference": "fixture.pth", "sha256": "a" * 64}] * 6,
        )


def test_unsealed_receipt_retains_unknown_identity_without_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"not JSON; forbidden scientific content need not be decodable")

    def refuse_read(path: Path) -> bytes:
        raise AssertionError("unsealed receipt must not be read")

    monkeypatch.setattr(Path, "read_bytes", refuse_read)
    binding = runtime_recovery.receipt_binding(receipt)
    assert binding["disposition"] == "UNBOUND_NO_SIDECAR"
    assert binding["sha256_from_sidecar"] is None
    assert not binding["receipt_decoded"]


def test_origin_linkage_rejects_mismatched_pack_and_manifest_commit() -> None:
    refs = {"dose": 256}
    pack_record = {"commit": "pack-commit"}
    pack = {"doses": [256], "input_hashes": {"b8.pth": "b8"}}
    manifest = {
        "commit": "different-commit",
        "f4": {"dose": 256},
        "env": {"METADATA_ARCHIVE_SHA256": "metadata"},
        "inputs": [
            {"dest": "b8.pth", "sha256": "b8"},
            {"dest": "metadata.tar", "sha256": "metadata"},
        ],
    }

    with pytest.raises(ValueError, match="commit differ"):
        runtime_recovery.validate_links("synthetic", refs, pack_record, pack, manifest)


def test_present_contradictory_metadata_archive_is_rejected() -> None:
    refs = {"dose": 256}
    pack_record = {"commit": "same", "metadata_archive_sha256": "different"}
    pack = {"doses": [256], "input_hashes": {"b8.pth": "b8"}}
    manifest = {
        "commit": "same",
        "f4": {"dose": 256},
        "env": {"METADATA_ARCHIVE_SHA256": "metadata"},
        "inputs": [
            {"dest": "b8.pth", "sha256": "b8"},
            {"dest": "metadata.tar", "sha256": "metadata"},
        ],
    }
    with pytest.raises(ValueError, match="pack metadata archive"):
        runtime_recovery.validate_links("synthetic", refs, pack_record, pack, manifest)
