"""Inert contract checks for the S18 KL-contrast audit operational outer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[1] / "verification/s18_kl_contrast_audit_outer_v1.py"
# The subject imports the M3-D research package, which lives only in the archived tag
# archive/m3d-autonomous-20261010 (not merged into main): skip cleanly when it is absent.
pytest.importorskip(
    "research.redesign_m3d_20261010",
    reason="M3-D package archived (tag archive/m3d-autonomous-20261010)",
)


@pytest.fixture
def outer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s18_kl_contrast_audit_outer_test_subject", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / "base"
    base.mkdir()
    (base / "state.json").write_text('{"active_local_job": null}\n')
    audit = tmp_path / "inert_audit.py"
    audit.write_text("""import argparse, json\np=argparse.ArgumentParser(); p.add_argument('--out'); p.add_argument('--deadline-monotonic'); p.add_argument('--global-monotonic-start'); a=p.parse_args(); json.dump({'schema':'m3d-s18-kl-contrast-independent-audit-raw/v1','status':'PASS_S18_KL_CONTRAST_ARTIFACT_AUDIT_RAW'},open(a.out,'w'))\n""")
    slot = tmp_path / "slot.py"
    slot.write_text("import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[2:]))\n")
    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "CANONICAL_AUDIT", audit)
    monkeypatch.setattr(module, "SLOTRUN", slot)
    monkeypatch.setattr(module, "PYTHON", sys.executable)
    monkeypatch.setattr(module.run_census, "default_guard", lambda: [])
    return module, audit


def test_inert_audit_preserves_raw_status_and_cleans_lease(outer, tmp_path):
    module, audit = outer
    start = time.monotonic() - 0.1
    raw, receipt = tmp_path / "raw.json", tmp_path / "outer.json"
    code = module.main([
        "--audit-source", str(audit), "--audit-sha", hashlib.sha256(audit.read_bytes()).hexdigest(),
        "--pin", str(module.SELF_SOURCE), module.sha256(module.SELF_SOURCE),
        "--raw-out", str(raw), "--out", str(receipt), "--deadline", repr(start + 120.0),
        "--global-monotonic-start", repr(start),
    ])
    result = json.loads(receipt.read_text())
    assert code == 0
    assert result["status"] == "PASS_S18_KL_CONTRAST_AUDIT_OUTER_RAW"
    assert result["raw_audit"]["status"] == "PASS_S18_KL_CONTRAST_ARTIFACT_AUDIT_RAW"
    assert result["lease"]["clear"] is True
    assert result["normal_data_qualification_allowed"] is False


def test_canonical_audit_and_clock_cannot_be_overridden(outer):
    module, _ = outer
    assert module.CANONICAL_AUDIT.name == "audit_s18_kl_contrast_v1.py"
    assert module.RAW_SCHEMA == "m3d-s18-kl-contrast-independent-audit-raw/v1"
    with pytest.raises(ValueError):
        module.protected_child_args(["--deadline-monotonic=1"])
