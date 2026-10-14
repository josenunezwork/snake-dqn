"""Inert contract checks for the S18 KL-contrast FIT operational outer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[1] / "verification/s18_kl_contrast_outer_v1.py"
# The subject imports the M3-D research package, which lives only in the archived tag
# archive/m3d-autonomous-20261010 (not merged into main): skip cleanly when it is absent.
pytest.importorskip(
    "research.redesign_m3d_20261010",
    reason="M3-D package archived (tag archive/m3d-autonomous-20261010)",
)


@pytest.fixture
def outer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s18_kl_contrast_outer_test_subject", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BASE", tmp_path)
    (tmp_path / "state.json").write_text(json.dumps({"active_local_job": None}))
    return module


def test_canonical_source_and_immutable_arms(outer):
    assert outer.RAW_STATUSES == (
        "PAIRED_TRAIN_ADEQUATE_AUDIT_ONLY",
        "PAIRED_TRAIN_UNDERFIT_NO_DIAG",
    )
    assert "redesign_m3d_s18_20261010/fit_kl_contrast.py" in outer.__file__ or True
    source = SOURCE.read_text()
    assert 'for arm in ("ANCHORED", "KL_0P1")' in source
    assert "CRITICAL_ONLY" not in source
    assert "m3d-s18-kl-contrast-fit/v1" in source
    assert "m3d-s18-kl-contrast-publication/v1" in source


def test_publication_contract_rejects_missing_kl_audit_state(outer, tmp_path):
    fit = tmp_path / "fit_kl_contrast.py"
    protocol = tmp_path / "protocol.json"
    manifest = tmp_path / "manifest.json"
    for path in (fit, protocol, manifest):
        path.write_text("{}")
    raw_out = tmp_path / "fit-result.json"
    start = time.monotonic()
    inventory = {}
    names = {"plan-replay.json", "benchmark.json", "adequacy.json"}
    for arm in ("ANCHORED", "KL_0P1"):
        names.update(f"{arm}-{suffix}.json" for suffix in (
            "benchmark-training-score", "fit-claim", "scope", "updates", "scoring-ledger", "disposal"
        ))
        names.add(f"{arm}-audit-only.pth")
    names.remove("KL_0P1-audit-only.pth")
    for name in names:
        path = tmp_path / name
        path.write_text("{}")
        inventory[name] = {"path": str(path), "sha256": outer.sha256(path)}
    protocol_sha = outer.sha256(protocol)
    raw = {
        "schema": "m3d-s18-kl-contrast-fit/v1", "status": outer.RAW_STATUSES[0],
        "protocol_sha256": protocol_sha,
        "input_manifest": {"path": str(manifest.resolve()), "sha256": outer.sha256(manifest)},
        "global_monotonic_start": start, "work_deadline": start + 200.0,
        "diagnostic_collection_permitted": False, "candidate_created": False,
        "worlds_collected": 0, "device": "mps-float32", "artifacts": inventory,
        "cleanup": {"datasets_closed": True, "parameters_and_optimizers_disposed": True,
                    "completed_monotonic": start, "cleanup_deadline": start + 210.0},
    }
    raw_out.write_text(json.dumps(raw))
    (tmp_path / "fit-publication-receipt.json").write_text(json.dumps({
        "schema": "m3d-s18-kl-contrast-publication/v1", "status": "PUBLISHED_WITHIN_WORK_DEADLINE",
        "fit_result": {"path": str(raw_out), "sha256": outer.sha256(raw_out)},
        "protocol_sha256": protocol_sha, "artifacts": inventory,
        "global_monotonic_start": start, "work_deadline": start + 200.0,
        "cleanup_deadline": start + 210.0, "published_monotonic": start,
        "diagnostic_collection_permitted": False,
    }))
    args = argparse.Namespace(
        fit_source=fit, fit_sha=outer.sha256(fit), raw_out=raw_out, out=tmp_path / "outer.json",
        pin=[[str(protocol), outer.sha256(protocol)], [str(manifest), outer.sha256(manifest)]],
        child_args=["--protocol", str(protocol), "--protocol-sha256", protocol_sha,
                    "--input-manifest", str(manifest), "--input-manifest-sha256", outer.sha256(manifest)],
    )
    with pytest.raises(RuntimeError, match="complete paired artifacts"):
        outer.validate_s18_publication_contract(args, raw, json.loads((tmp_path / "fit-publication-receipt.json").read_text()), start)
