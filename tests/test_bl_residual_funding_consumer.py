"""Canonical R6.3 funding receipts must admit the unchanged I3 consumer.

The producer and consumer use different representations of the fixed reserve:
the held-out look retains structured ancestry, whereas I3 intentionally compares
the frozen fit controls' ordered life-ID digest.  This test derives both values
from real synthetic D256/D1024 package controls and carries the producer output
directly into the existing I3 staging fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json
from research.bl_residual import f4_d1024_package as d1024_package
from research.bl_residual import funding, job
from research.bl_residual.protocol import F4_REGISTRATION_SHA
from research.bl_residual.ra_guard import reserve_identity
from tests import f4_d1024_fixtures as fx
from tests import test_bl_residual_collection_receipts as i3


def _frozen_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, object], tuple[Path, Path, Path, Path]]:
    """Build the two real synthetic fit controls and derive their shared reserve."""
    world = fx.build_world(tmp_path / "world")
    calls: dict[str, object] = {}
    fx.patch_package(monkeypatch, world, calls)
    d256_pack = fx.build_d256_pack(tmp_path, world)
    staging = fx.build_staging(tmp_path, d256_pack)
    ledgers = tmp_path / "ledgers"
    ledgers.mkdir()
    for name in ("d256", "d1024"):
        export_json(ledgers / f"{name}.json", {"synthetic_ledger": name})
    d1024_pack = tmp_path / "d1024-pack"
    d1024_package._prepare(
        d1024_pack,
        d256_ledger=ledgers / "d256.json",
        d1024_ledger=ledgers / "d1024.json",
        d1024_union_sha="d" * 64,
        staging=staging,
        commit="HEAD",
        preflight_receipt=tmp_path / "preflight.json",
    )
    paths = (
        d256_pack / "inputs/control.json",
        d1024_pack / "inputs/control.json",
        d256_pack / "inputs/life-allocation.json",
        d1024_pack / "inputs/life-allocation.json",
    )
    return reserve_identity(*paths), paths


def _funding_look(
    *,
    dose: int,
    checkpoint_sha256: str,
    identity: dict[str, object],
    allocation: Path,
    retained_rows: int,
) -> dict[str, object]:
    """Make a complete no-outcome held-out receipt over the derived reserve."""
    contents = json.loads(allocation.read_text())
    parents = contents["parents"]
    life_ids = identity["reserve_life_ids"]
    assert isinstance(life_ids, list)
    retention = [
        {
            "life_id": life,
            "parent_world": parents[life],
            "decisive_rows": 1,
            "retained_rows": retained_rows,
            "retention": float(retained_rows),
        }
        for life in life_ids
    ]
    return {
        "schema": "bl-residual-heldout/v2",
        "dose": dose,
        "namespace": identity["series_id"],
        "recipe_sha256": fx.recipe_sha(),
        "registration_sha256": F4_REGISTRATION_SHA,
        # This is intentionally the evaluator's structured ancestry digest;
        # funding.evaluate must preserve it separately from the control digest.
        "reserve_allocation_sha256": identity["reserve_ancestry_sha256"],
        "reserve_life_ids": life_ids,
        "checkpoint_sha256": checkpoint_sha256,
        "archive_receipt_sha256": "a" * 64,
        "table_sha256": "b" * 64,
        "natural_archive_complete": True,
        "metrics": {
            "life_retention": retention,
            "reserve_lives_with_zero_decisive_rows": 0,
            "harmful_override_count_natural": 0,
        },
    }


def test_canonical_funding_producer_enters_i3_with_frozen_control_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A producer result derived from controls passes I3 without matched constants."""
    identity, paths = _frozen_controls(monkeypatch, tmp_path)
    d256_control, d1024_control, d256_allocation, d1024_allocation = paths
    before = {path: path.read_bytes() for path in paths}
    left = _funding_look(
        dose=256,
        checkpoint_sha256="2" * 64,
        identity=identity,
        allocation=d256_allocation,
        retained_rows=0,
    )
    right = _funding_look(
        dose=1024,
        checkpoint_sha256="1" * 64,
        identity=identity,
        allocation=d1024_allocation,
        retained_rows=1,
    )
    result = funding.evaluate(
        left,
        right,
        {},
        reserve_identity=identity,
    )
    assert result["verdict"] == "PASS"
    assert result["reserve_allocation_sha256"] == identity["reserve_allocation_sha256"]
    assert result["reserve_ancestry_sha256"] == identity["reserve_ancestry_sha256"]

    # The I3 fixture creates and binds all downstream receipts from the identity
    # produced above.  Its strict equality check remains the real consumer.
    # The package fixture temporarily substitutes only the synthetic B8 digest
    # for construction; restore I3's consumer contract before exercising it.
    monkeypatch.setattr(job, "B8_SHA", i3.B8_SHA)
    monkeypatch.setattr(i3, "SERIES", identity["series_id"])
    monkeypatch.setattr(i3, "RESERVE", identity["reserve_allocation_sha256"])
    control = i3.staging(tmp_path / "i3")
    i3.replace_bound(tmp_path / "i3", control, "D4096_FUNDING", result)
    i3.assert_dependent_seals(tmp_path / "i3", control)
    job._validate_staging(tmp_path / "i3", control, [4096], i3.RECIPE, set())

    assert {path: path.read_bytes() for path in paths} == before

    foreign = {**left, "reserve_life_ids": list(reversed(left["reserve_life_ids"]))}
    with pytest.raises(ValueError, match="reserve"):
        funding.evaluate(foreign, right, {}, reserve_identity=identity)
