"""Tests for the no-unpickle release-asset publication scan."""

from __future__ import annotations

import io
import json
import pickle
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo.p1_release_scan import scan_asset


def _torch_like(path: Path, metadata: object) -> None:
    """Make a torch-save-shaped ZIP without importing/deserializing torch."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(metadata, protocol=4))
        archive.writestr("archive/data/0", b"\x00\xff" * 4096)


def test_pickle_metadata_secret_blocks_without_exposing_value(tmp_path: Path) -> None:
    asset = tmp_path / "checkpoint.pt"
    _torch_like(asset, {"metadata": {"client_secret": "sensitive-token-value-12345"}})
    result = scan_asset(asset)
    assert result["status"] == "BLOCKED"
    assert result["publication_blocked"] is True
    assert any(row["kind"] == "credential_value" for row in result["findings"])
    assert "sensitive-token-value-12345" not in json.dumps(result)
    assert result["method_counts"]["pickle"] == 1


def test_tar_and_npy_metadata_paths_and_pii_are_redacted(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    np.save(source, np.array([1, 2, 3], dtype=np.int64))
    asset = tmp_path / "p1-inputs.tar.gz"
    with tarfile.open(asset, "w:gz") as archive:
        info = tarfile.TarInfo("settings.json")
        body = b'{"owner":"person@example.test","root":"/Users/researcher/private"}'
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
        archive.add(source, arcname="numeric.npy")
    result = scan_asset(asset)
    assert result["status"] == "PASS_NO_SECRETS_WITH_PATH_OR_PII_DISCLOSURES"
    assert result["publication_blocked"] is False
    assert result["method_counts"]["numeric_npy_storage_skipped"] == 1
    rendered = json.dumps(result)
    assert "person@example.test" in rendered  # PII is an indicator, never credential value.
    assert "/Users/researcher/private" in rendered


def test_clean_numeric_npz_passes_and_never_interprets_storage_as_text(tmp_path: Path) -> None:
    asset = tmp_path / "rows.npz"
    np.savez(asset, values=np.arange(32, dtype=np.float32))
    result = scan_asset(asset)
    assert result["status"] == "PASS_NO_SECRETS_OR_PII_FOUND"
    assert result["method_counts"]["numeric_npy_storage_skipped"] == 1
    assert not result["findings"] and not result["failures"]


def test_unicode_npy_payload_secret_blocks_without_scanning_numeric_storage(tmp_path: Path) -> None:
    asset = tmp_path / "labels.npy"
    np.save(asset, np.array(["ghp_abcdefghijklmnopqrstuvwxyz"], dtype="U32"))
    result = scan_asset(asset)
    assert result["status"] == "BLOCKED"
    assert any(row["kind"] == "github_token" for row in result["findings"])


@pytest.mark.parametrize(
    "metadata",
    [
        {"env": {"GOOGLE_OAUTH_ACCESS_TOKEN": "ya29.a-secret-oauth-token-123456"}},
        {"access_token": b"ya29.a-secret-oauth-token-123456"},
    ],
)
def test_credential_values_block_in_json_and_pickle_without_value_disclosure(
    tmp_path: Path, metadata: object
) -> None:
    asset = tmp_path / (
        "metadata.json" if isinstance(metadata, dict) and "env" in metadata else "data.pkl"
    )
    if asset.suffix == ".json":
        asset.write_text(json.dumps(metadata))
    else:
        asset.write_bytes(pickle.dumps(metadata, protocol=4))
    result = scan_asset(asset)
    assert result["publication_blocked"] is True
    assert "ya29.a-secret-oauth-token-123456" not in json.dumps(result)


def test_absolute_path_with_double_slash_is_disclosure_not_oauth_token(tmp_path: Path) -> None:
    asset = tmp_path / "paths.json"
    asset.write_text(json.dumps({"existing_output_path": "/Users/a/run-1//source-repair-03"}))
    result = scan_asset(asset)
    assert result["publication_blocked"] is False
    assert result["status"] == "PASS_NO_SECRETS_WITH_PATH_OR_PII_DISCLOSURES"


def test_long_standalone_google_refresh_token_blocks(tmp_path: Path) -> None:
    asset = tmp_path / "token.json"
    asset.write_text(json.dumps({"refresh_token": "1//" + "a" * 30}))
    result = scan_asset(asset)
    assert result["publication_blocked"] is True
    assert any(row["kind"] == "oauth_token" for row in result["findings"])


def test_pickle_credential_assignment_never_leaks_value_in_indicator(tmp_path: Path) -> None:
    asset = tmp_path / "metadata.pkl"
    secret = "super-sensitive-value-12345"
    asset.write_bytes(pickle.dumps({"metadata": "api_key=" + secret}, protocol=4))
    result = scan_asset(asset)
    assert result["publication_blocked"] is True
    assert secret not in json.dumps(result)


def test_unknown_or_malformed_metadata_fails_closed(tmp_path: Path) -> None:
    asset = tmp_path / "unknown.bin"
    asset.write_bytes(b"not an approved release serialization")
    result = scan_asset(asset)
    assert result["status"] == "BLOCKED"
    assert result["failures"]
