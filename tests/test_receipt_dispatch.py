"""Fail-closed review and live-accounting boundaries of the saved helper repair."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.gpu_sim_spike.sparse_units import publish_json
from tools.ops import receipt_dispatch as helper


@pytest.mark.parametrize("mutation", [None, "wrapper", "marker", "missing-review"])
def test_repair_requires_current_wrapper_and_complete_review(tmp_path: Path, mutation: str) -> None:
    wrapper = tmp_path / "dispatch-canary.sh"
    wrapper.write_text("saved wrapper\n")
    review = tmp_path / "CODEX_V_RECEIPT_FIX.json"
    if mutation != "missing-review":
        publish_json(
            review,
            {
                "status": "ACCEPT",
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in helper.reviewed_files(tmp_path, "sparse")
                },
            },
        )
    if mutation == "wrapper":
        wrapper.write_text("changed after review\n")
    elif mutation == "marker":
        Path(str(review) + ".complete").unlink()
    if mutation is None:
        helper.require_review(tmp_path, "sparse")
    else:
        with pytest.raises((ValueError, OSError)):
            helper.require_review(tmp_path, "sparse")


def test_missing_live_accounting_never_reaches_dispatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(helper, "V8", tmp_path)
    monkeypatch.setattr(helper, "require_review", lambda *args: None)
    monkeypatch.setattr(helper, "load", lambda *args: SimpleNamespace())
    monkeypatch.setattr(helper, "verify_v8_preparation", lambda *args: {})
    monkeypatch.setattr(helper, "dispatch", lambda *args: pytest.fail("dispatch reached"))
    with pytest.raises(ValueError, match="INCOMPLETE_ADMISSION: missing operator accounting"):
        helper.v8(False)
