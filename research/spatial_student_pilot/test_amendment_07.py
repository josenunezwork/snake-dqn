"""Prospective support, diagnostics, provenance and outage falsifications."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from research.spatial_student_pilot.diagnostics import (
    bc_summary,
    heldout_bc_agreement,
    label_confidence,
    policy_record,
)
from research.spatial_student_pilot.fault_test import run_fault_test
from research.spatial_student_pilot.resume import UnitStore, empty_ledger, hardware_provenance
from research.spatial_student_pilot.test_search_readout import endpoint, read


def test_death_support_precedes_all_scientific_clauses_but_preserves_diagnostics() -> None:
    roots, worlds = endpoint()
    for row in roots:
        row["b8_survival"] = [1, 1]
    for row in worlds:
        row["student_mass"] = -100
    result = read(roots, worlds)
    assert result["verdict"] == "INCOMPLETE"
    assert result["authorization"] == "none"
    assert not result["clauses"]["equal_mix_gameplay_mass_at_least_b8"]
    assert sum(s["roots"] for s in result["secondary"]["confidence_strata"]) == 384
    for row in roots[:16]:
        row["b8_survival"] = [0, 0]
    assert read(roots, worlds)["verdict"] == "KILL"
    roots[0]["student_logits"] = [float("nan")] * 6
    assert read(roots, worlds)["verdict"] == "INVALID"


def test_confidence_is_selection_only_and_boundary_is_strict() -> None:
    roots, _ = endpoint()
    row = roots[0]
    assert label_confidence(row) == "high"
    row["student_mass"] = [-1e9, 1e9]
    row["b8_survival"] = [0, 0]
    assert label_confidence(row) == "high"
    # Differences [0,2] have mean1 and SE1: not greater than twice SE.
    row["selection_mass"][1] = [100, 102]
    assert label_confidence(row) == "low"
    row["selection_mass"][1] = [101, 103]  # mean2 == twice SE.
    assert label_confidence(row) == "low"
    row["selection_mass"][1] = [101.01, 103.01]
    assert label_confidence(row) == "high"
    row["legal_mask"] = [False, True, False, False, False, False]
    assert label_confidence(row) == "high"


def test_policy_export_uses_legal_top_two_not_raw_illegal_maximum() -> None:
    row = policy_record([2, 4, 100, 7, 6, 5], [True, True, False, False, False, False])
    assert row["student_action"] == 1 and row["student_top2_margin"] == 2
    assert row["student_logits"][2] == 100
    assert (
        policy_record([1] * 6, [True, False, False, False, False, False])["student_top2_margin"]
        == 0
    )
    roots, worlds = endpoint()
    roots[0]["student_top2_margin"] = 3
    assert read(roots, worlds)["verdict"] == "INVALID"
    roots[0]["student_top2_margin"] = 2
    roots[0]["student_action"] = 0
    assert read(roots, worlds)["verdict"] == "INVALID"


def test_bc_summary_is_parent_split_and_refuses_foreign_or_illegal_predictions() -> None:
    roots = [
        dict(
            uid=str(i),
            world_id=i // 2,
            b8_action=i % 2,
            mask=[True, True, True, False, False, False],
        )
        for i in range(4)
    ]
    predictions = [dict(uid=r["uid"], world_id=r["world_id"], action=0) for r in roots]
    result = bc_summary(roots, predictions)
    assert result["by_parent"] == [dict(world_id=i, agreed=1, total=2) for i in range(2)]
    wrong = deepcopy(predictions)
    wrong[0]["action"] = 5
    with pytest.raises(ValueError, match="action differs"):
        bc_summary(roots, wrong)
    with pytest.raises(ValueError, match="UIDs differ"):
        bc_summary(roots, predictions[:-1] + [predictions[0]])
    roots[0]["partition"] = "assessment"
    with pytest.raises(ValueError, match="disjoint"):
        heldout_bc_agreement(None, roots[:1], {0}, device="cpu", microbatch=1)


def test_unit_hardware_bound_inside_payload_and_gpu_sku_recorded(tmp_path: Path) -> None:
    from research.spatial_student_pilot.config import PilotConfig

    cfg = PilotConfig().development("preflight")
    store = UnitStore(tmp_path / "out", cfg, empty_ledger(cfg), tmp_path)
    store.publish("collect-0000", {})
    with patch(
        "torch.load",
        return_value=dict(
            identity={
                "namespace": cfg.namespace,
                "allocation_index": 0,
                "config_sha256": cfg.sha256,
                "unit_id": "collect-0000",
                "schema": "spatial-immutable-unit/v1",
            },
            hardware={"gpu_sku": "forged"},
            payload={},
        ),
    ):
        with pytest.raises(ValueError, match="hardware differs"):
            store.get("collect-0000")
    with (
        patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        hardware = hardware_provenance("cuda")
    assert hardware["gpu_sku"] == "NVIDIA H100 80GB HBM3"
    assert hardware["capability"] == [9, 0]


def test_outage_export_and_resume_offline(tmp_path: Path) -> None:
    assert run_fault_test(tmp_path)["status"] == "PASS"


def test_ruler_diagnostic_units_are_publishable_only_in_ruler_namespace(tmp_path: Path) -> None:
    from research.spatial_student_pilot.config import PilotConfig

    ruler = PilotConfig().development("ruler")
    store = UnitStore(tmp_path / "ruler", ruler, empty_ledger(ruler), tmp_path)
    for name in ("early-0000", "mature-0000"):
        store.publish(name, {"development_only": True})
        assert store.get(name) == {"development_only": True}
    scientific = PilotConfig()
    store = UnitStore(tmp_path / "pilot", scientific, empty_ledger(scientific), tmp_path)
    with pytest.raises(ValueError, match="ruler-only"):
        store.publish("early-0000", {})
