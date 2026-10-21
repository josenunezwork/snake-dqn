"""Focused tests for O2's standalone upload admission helper."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from research.t34_o2_harvest import upload_readiness


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


def test_o2_check_uses_the_shared_budget_without_changing_registration(tmp_path: Path) -> None:
    result = upload_readiness.check(_manifest(tmp_path))
    assert result["normal_forecast_seconds"] <= result["normal_budget_seconds"]


def test_o2_check_refuses_over_budget_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("upload forecast exceeds qualified budget")

    monkeypatch.setattr(upload_readiness, "assert_upload_budget", refused)
    with pytest.raises(ValueError, match="exceeds qualified budget"):
        upload_readiness.check(_manifest(tmp_path))
