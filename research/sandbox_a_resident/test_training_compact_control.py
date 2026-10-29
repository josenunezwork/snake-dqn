"""Full-control compact transport for fresh TRAIN-01 trajectory boundaries."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict

import pytest
import torch

from research.sandbox_a_resident.test_training_compact import sample_state
from research.sandbox_a_resident.train import TrainControl
from research.sandbox_a_resident.training_compact import (
    encode_snapshot,
    materialize_compact_resume,
    verify_decoded_state,
)
from research.sandbox_a_resident.training_preflight import validate_uploaded_ledger
from research.sandbox_a_resident.training_resume import attach_compact_resume, build_ledger
from research.sandbox_a_resident.training_units import (
    allocation,
    load_ledger,
    publish_snapshot,
    unit_id,
    validate_compact_resume,
)
from research.sandbox_a_resident.units import publish


@pytest.fixture
def fresh_compact(tmp_path):
    control = asdict(
        TrainControl("sandbox-a-train01", "a" * 40, 2, train_seed_offset=2_000_000_000)
    )
    exports = tmp_path / "exports"
    for iteration in (16, 64):
        publish_snapshot(
            exports,
            iteration,
            lambda path: torch.save(sample_state(), path),
            metadata={"allocation": allocation(control), "source_commit": control["source_commit"]},
        )
        publish(exports / f"iteration-{iteration:04d}-actor.pth", lambda p: p.write_bytes(b"actor"))
    original = exports / "iteration-0064.pt"
    encode_snapshot(original, unit=unit_id(64))
    ledger, uploads, latest = build_ledger(exports, control)
    assert latest == 64
    # The manual completion builder retains exactly the latest eligible unit.
    ledger["units"] = [ledger["units"][-1]]
    uploads = [r for r in uploads if "iteration-0064" in r["dest"]]
    ledger, uploads = attach_compact_resume(ledger, uploads, exports, control=control)
    return control, original, ledger, uploads


def test_fresh_seed_late_compact_roundtrip_without_allowlist_patch(fresh_compact, tmp_path):
    control, original, ledger, uploads = fresh_compact
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "verified-ledger.json").write_text(json.dumps(ledger))
    validate_uploaded_ledger(packet, {"inputs": uploads}, control)
    assert all(not r["dest"].endswith(".pt") for r in uploads)
    inputs = tmp_path / "inputs"
    for row in uploads:
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], target)
    manifest = materialize_compact_resume(ledger, inputs)
    decoded = inputs / "resume/iteration-0064.pt"
    assert decoded.read_bytes() == original.read_bytes()
    verify_decoded_state(torch.load(decoded, weights_only=False), manifest)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    assert list(load_ledger(path, control, input_root=inputs)) == [64]
    foreign = {**control, "train_seed_offset": 3_000_000_000}
    with pytest.raises(ValueError, match="foreign ledger"):
        load_ledger(path, foreign, input_root=inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        "allocation",
        "namespace",
        "foreign_control",
        "duplicate",
        "cadence",
        "path",
        "size",
        "digest",
        "original",
        "migration",
    ],
)
def test_fresh_compact_refuses_foreign_or_malformed_bindings(fresh_compact, mutation):
    _, _, original, _ = fresh_compact
    ledger = copy.deepcopy(original)
    entry = ledger["compact_resume"]
    if mutation in {"allocation", "namespace"}:
        ledger[mutation] = "foreign"
    elif mutation == "foreign_control":
        entry["control"]["allocation_index"] = 3
    elif mutation == "duplicate":
        ledger["units"] *= 2
    elif mutation == "cadence":
        ledger["units"][0]["iteration"] = 63
    elif mutation == "path":
        entry["derivative"]["path"] = "resume/../outside.xz"
    elif mutation == "size":
        entry["derivative"]["size"] = 128 * 1024**2 + 1
    elif mutation == "digest":
        entry["manifest"]["sha256"] = "g" * 64
    elif mutation == "original":
        entry["original"]["sha256"] = "f" * 64
    else:
        ledger["source_migration"] = {}
    with pytest.raises(ValueError):
        validate_compact_resume(ledger)
