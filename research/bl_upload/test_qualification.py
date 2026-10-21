"""Focused tests for the shared Big Lever upload qualification boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import research.bl_upload.qualification as qualification
from research.bl_upload.qualification import (
    NORMAL_UPLOAD_FORECAST_SECONDS,
    assert_upload_budget,
    convert_public_checkpoints,
    qualify,
    verify_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _manifest(tmp_path: Path, *, body: bytes = b"tiny") -> dict:
    payload = tmp_path / "payload.json"
    payload.write_bytes(body)
    return {
        "job_id": "bl-upload-test",
        "commit": _commit(),
        "script": "research/bl_ppo/colab.sh",
        "env": {},
        "inputs": [
            {
                "path": str(payload),
                "dest": "payload.json",
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ],
        "source_paths": [
            "research/bl_ppo",
            "research/colab_cli",
            "src",
            "requirements.txt",
            "pyproject.toml",
        ],
    }


def test_budget_binds_pinned_bootstrap_and_rejects_bad_local_hash(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = assert_upload_budget(manifest)
    assert result["normal_forecast_seconds"] <= NORMAL_UPLOAD_FORECAST_SECONDS
    assert {row["dest"] for row in result["files"]} >= {"bootstrap.sh", "source.py", "payload.json"}
    manifest["inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        assert_upload_budget(manifest)


def test_full_qualification_binds_source_plan_and_create_only_receipt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    receipt = tmp_path / "upload-preflight.json"
    result = qualify(manifest, receipt_path=receipt)
    saved = json.loads(receipt.read_text())
    assert saved["manifest_sha256"] == result["manifest_sha256"]
    assert result["source_plan"]["commit"] == manifest["commit"]
    assert result["fallback_total_bound_seconds"] <= 300
    assert verify_receipt(manifest, receipt)["status"] == "PASS_NOT_DISPATCHED"
    with pytest.raises(ValueError, match="already exists"):
        qualify(manifest, receipt_path=receipt)


def test_public_conversion_requires_all_six_and_preserves_small_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = {
        "inputs": [
            {
                "dest": f"checkpoints/{index}.pth",
                "size_bytes": 1,
                "sha256": hashlib.sha256(bytes([index])).hexdigest(),
            }
            for index in range(6)
        ]
    }
    monkeypatch.setattr(qualification, "_catalog_rows", lambda _repo: catalog["inputs"])
    inputs = []
    for index, row in enumerate(catalog["inputs"]):
        path = tmp_path / f"checkpoint-{index}"
        path.write_bytes(bytes([index]))
        inputs.append(
            {
                "path": str(path),
                "dest": row["dest"],
                "sha256": row["sha256"],
            }
        )
    extra = tmp_path / "registration.json"
    extra.write_text("{}")
    manifest = {
        "inputs": [
            *inputs,
            {
                "path": str(extra),
                "dest": "registration.json",
                "sha256": hashlib.sha256(b"{}").hexdigest(),
            },
        ],
        "env": {},
        "source_paths": [],
    }
    converted = convert_public_checkpoints(manifest)
    assert converted["env"]["BL_PUBLIC_INPUTS"] == "1"
    assert {row["dest"] for row in converted["inputs"]} == {
        "public-inputs.json",
        "registration.json",
    }
    assert "research/bl_upload" in converted["source_paths"]
    partial = {**manifest, "inputs": inputs[:-1], "env": {}, "source_paths": []}
    assert convert_public_checkpoints(partial)["inputs"] == partial["inputs"]


@pytest.mark.parametrize(
    "row",
    [
        {"dest": "unbound.dat", "size_bytes": 10**9, "sha256": "a" * 64},
        {
            "dest": "remote.dat",
            "url": "http://example.com/input",
            "size_bytes": 1,
            "sha256": "a" * 64,
        },
        {
            "dest": "remote.dat",
            "url": "https://example.com/input",
            "size_bytes": True,
            "sha256": "a" * 64,
        },
    ],
)
def test_budget_refuses_unbound_or_malformed_remote_inputs(tmp_path: Path, row: dict) -> None:
    manifest = _manifest(tmp_path)
    manifest["inputs"] = [row]
    with pytest.raises((ValueError, RuntimeError)):
        assert_upload_budget(manifest)


def test_fallback_must_fit_half_cap_not_only_whole_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        qualification,
        "_source_plan",
        lambda *_args: (
            {"commit": manifest["commit"], "tree": "f" * 40, "fallback_size": 4_000_000},
            4_000_000,
            "a" * 64,
        ),
    )
    with pytest.raises(ValueError, match="fallback upload forecast.*150"):
        qualify(manifest)
