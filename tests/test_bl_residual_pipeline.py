"""F3 allocation, immutable-unit and actual-entrypoint plumbing contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo.exports import digest
from research.bl_residual import job
from research.bl_residual.data import allocate_lives, validate_rows, world_fold
from research.bl_residual.make_pack import prepare_inputs
from research.bl_residual.protocol import (
    allocation_hash,
    build_ledger,
    read_ledger,
    require_judgment,
)

B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")


def catalog(n: int) -> list[dict]:
    return [
        {
            "life_id": f"life-{i}",
            "parent_world": f"parent-{i // 2}",
            "stream": "S-Watch6" if i % 3 == 0 else "S-gate",
            "rescue_bearing": True,
            "allocation_index": i,
        }
        for i in range(n)
    ]


def test_nested_doses_and_append_only_holdout() -> None:
    first = allocate_lives(catalog(1500))
    second = allocate_lives(catalog(6000), prior=first)
    for dose in (256, 1024):
        assert first["doses"][str(dose)] == second["doses"][str(dose)]
    d = second["doses"]
    assert len(d["256"]["heldout_lives"]) == 64
    assert d["1024"]["train_lives"] == d["4096"]["train_lives"][:960]
    train_parents = {second["parents"][life] for life in d["4096"]["train_lives"]}
    hold_parents = {second["parents"][life] for life in d["256"]["heldout_lives"]}
    assert not train_parents & hold_parents
    assert all(world_fold(p) == 0 for p in hold_parents)
    changed = catalog(6000)
    changed[0]["parent_world"] = "foreign"
    with pytest.raises(ValueError, match="changed"):
        allocate_lives(changed, prior=first)


def test_insufficient_holdout_never_shrinks_N() -> None:
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        allocate_lives(catalog(10))
    value = allocate_lives(catalog(500))
    assert value["doses"]["256"]["ready"]
    assert not value["doses"]["4096"]["ready"]
    assert value["doses"]["4096"]["train_lives"] == []


def test_draft_cannot_authorize_factory_fit(tmp_path: Path) -> None:
    registration = tmp_path / "F4_DRAFT.md"
    registration.write_text("UNRATIFIED")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema": "bl-residual-f4-judgment/v1",
                "decision": "APPROVE",
                "registration_sha256": digest(registration),
                "recipe_sha256": "f" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="UNRATIFIED"):
        require_judgment(approval, registration, "f" * 64)


@pytest.mark.skipif(not B8.exists(), reason="local pinned B8 fixture unavailable")
def test_real_synthetic_entrypoint_resume_skips_exact_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    control = prepare_inputs(inputs, B8, doses=(256,))
    first = tmp_path / "first"
    result = job.run(inputs, first, mode="smoke")
    assert result["done"] == result["expected_units"] == 1
    assert result["heldout_outcomes_opened"] is False
    ledger = tmp_path / "ledger.json"
    verified = build_ledger([first], ledger, control["namespace"], control["allocation"])
    assert len(verified["units"]) == 1
    # Resume only changes the separately bound ledger, never the fit allocation.
    (inputs / "verified-ledger.json").write_bytes(ledger.read_bytes())
    control["ledger_sha256"] = digest(ledger)
    assert allocation_hash(control) == control["allocation"]
    (inputs / "control.json").write_text(json.dumps(control))
    monkeypatch.setattr(job, "synthetic_table", lambda *_: pytest.fail("verified unit recomputed"))
    resumed = job.run(inputs, tmp_path / "second", mode="smoke")
    assert resumed["skipped"] == [0]
    assert resumed["completed"] == []
    with pytest.raises(ValueError, match="foreign"):
        read_ledger(ledger, "foreign", control["allocation"])
    with pytest.raises(ValueError, match="duplicate"):
        build_ledger(
            [first, first],
            tmp_path / "duplicates.json",
            control["namespace"],
            control["allocation"],
        )


def test_all_scientific_controls_are_allocation_bound() -> None:
    control = {"namespace": "test", "doses": [256], "train_lives": {"256": ["a"]}}
    original = allocation_hash(control)
    control["train_lives"]["256"] = ["b"]
    assert original != allocation_hash(control)


def test_heldout_refuses_synthetic_foreign_and_changed_reserve(tmp_path: Path) -> None:
    from research.bl_residual.heldout import validate_candidate

    allocation = tmp_path / "allocation.json"
    allocation.write_text("frozen allocator fixture")
    control = {
        "synthetic": False,
        "doses": [256],
        "namespace": "series",
        "recipe_sha256": "a" * 64,
        "input_hashes": {"life-allocation.json": digest(allocation)},
    }
    control["allocation"] = allocation_hash(control)
    meta = {
        "schema": "bl-residual-fit/v1",
        "synthetic": False,
        "dose": 256,
        "index": 0,
        "updates": 2000,
        "namespace": "series",
        "recipe_sha256": "a" * 64,
        "allocation": control["allocation"],
    }
    validate_candidate(meta, control, allocation)
    with pytest.raises(ValueError, match="foreign/synthetic"):
        validate_candidate({**meta, "synthetic": True}, control, allocation)
    with pytest.raises(ValueError, match="foreign/synthetic"):
        validate_candidate({**meta, "updates": 3}, control, allocation)
    allocation.write_text("other reserve")
    with pytest.raises(ValueError, match="held-out allocation"):
        validate_candidate(meta, control, allocation)


@pytest.mark.skipif(not B8.exists(), reason="local pinned B8 fixture unavailable")
def test_raw_adapter_refuses_unknown_status_and_cross_parent_life() -> None:
    from src.model.ego2s_network import load_ego2s_checkpoint

    table = job.synthetic_table(load_ego2s_checkpoint(str(B8)))
    validate_rows(table)
    fp64 = {**table, "q": table["q"].astype(np.float64)}
    with pytest.raises(ValueError, match="fp32"):
        validate_rows(fp64)
    table["status"][0, 0] = "BAD"
    with pytest.raises(ValueError, match="status"):
        validate_rows(table)
    table["status"][0, 0] = "EXACT"
    table["parent_world"][0] = "foreign"
    with pytest.raises(ValueError, match="ancestry"):
        validate_rows(table)
    assert table["local"].dtype == np.uint8
