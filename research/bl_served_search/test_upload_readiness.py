"""Focused tests for the served-search package-local upload gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.bl_served_search import upload_readiness


def _manifest(tmp_path: Path) -> dict[str, object]:
    payload = tmp_path / "small.json"
    payload.write_text("{}\n", encoding="utf-8")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True
    ).strip()
    return {
        "commit": commit,
        "inputs": [
            {
                "path": str(payload),
                "dest": "small.json",
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
    }


def test_check_delegates_to_the_qualified_upload_budget(tmp_path: Path) -> None:
    result = upload_readiness.check(_manifest(tmp_path))
    assert result["normal_forecast_seconds"] <= result["normal_budget_seconds"]


def test_check_refuses_a_shared_budget_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("upload forecast exceeds qualified budget")

    monkeypatch.setattr(upload_readiness, "assert_upload_budget", refused)
    with pytest.raises(ValueError, match="exceeds qualified budget"):
        upload_readiness.check(_manifest(tmp_path))


def test_receipt_option_runs_full_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    seen: dict[str, Path] = {}

    def full(manifest: dict[str, object], *, repo: Path, receipt_path: Path) -> dict[str, str]:
        assert manifest["commit"]
        assert repo == upload_readiness.ROOT
        seen["receipt"] = receipt_path
        return {"status": "PASS_NOT_DISPATCHED"}

    monkeypatch.setattr(upload_readiness, "qualify", full)
    monkeypatch.setattr(
        sys,
        "argv",
        ["upload_readiness", "--manifest", str(manifest_path), "--receipt", str(receipt)],
    )

    upload_readiness.main()

    assert seen["receipt"] == receipt
