"""Admission-boundary tests; no live pilot data or endpoint look is performed."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from research.spatial_student_pilot.aggregate import (
    _exact_rows,
    _verify_seed_audit,
    aggregate,
    expected_units,
)
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.resume import empty_ledger
from research.spatial_student_pilot.run import publish_json


def test_exact_frozen_endpoint_inventory() -> None:
    units = expected_units(PilotConfig())
    assert len(units) == 86
    assert sum(u.startswith("collect-") for u in units) == 8
    assert sum(u.startswith("search-") for u in units) == 32
    assert sum(u.startswith("fit-") for u in units) == 32
    assert sum(u.startswith("assess-") for u in units) == 12
    assert sum(u.startswith("gameplay-") for u in units) == 2
    assert "fit-0008000" in units
    assert "fit-0000250" in units


def test_empty_ledger_is_incomplete_without_loading_or_readout(tmp_path: Path) -> None:
    cfg = PilotConfig()
    with (
        patch("research.spatial_student_pilot.aggregate.platform.system", return_value="Darwin"),
        patch("research.spatial_student_pilot.aggregate.evaluate_readout") as readout,
        patch("research.spatial_student_pilot.aggregate.UnitStore") as store,
    ):
        result = aggregate(cfg, empty_ledger(cfg), tmp_path, tmp_path / "actor", None, {})
    assert result["verdict"] == "INCOMPLETE"
    assert len(result["missing_units"]) == 86
    readout.assert_not_called()
    store.assert_not_called()


def test_foreign_ledger_and_non_mac_are_invalid(tmp_path: Path) -> None:
    cfg = PilotConfig()
    ledger = empty_ledger(cfg)
    ledger["config_sha256"] = "f" * 64
    with patch("research.spatial_student_pilot.aggregate.platform.system", return_value="Darwin"):
        assert aggregate(cfg, ledger, tmp_path, "actor", None, {})["verdict"] == "INVALID"
        assert (
            aggregate(cfg.development("preflight"), {}, tmp_path, "actor", None, {})["verdict"]
            == "INVALID"
        )
    with patch("research.spatial_student_pilot.aggregate.platform.system", return_value="Linux"):
        assert (
            aggregate(cfg, empty_ledger(cfg), tmp_path, "actor", None, {})["verdict"] == "INVALID"
        )


def test_selected_uid_and_payload_integrity() -> None:
    row = dict(
        uid="root0",
        world_id=1,
        partition="train",
        kind="late",
        mix="frozen",
        b8_action=1,
        compact={"body": torch.tensor([1, 2])},
    )
    selected = [row]
    _exact_rows([deepcopy(row)], selected, assessment=False)
    altered = deepcopy(row)
    altered["compact"]["body"][0] = 9
    with pytest.raises(ValueError, match="payload"):
        _exact_rows([altered], selected, assessment=False)
    with pytest.raises(ValueError, match="UID"):
        _exact_rows([dict(row, uid="foreign")], selected, assessment=False)
    with pytest.raises(ValueError, match="duplicate"):
        _exact_rows([row, row], selected, assessment=False)
    assessment = dict(row, partition="assessment", scoring_suffixes=[0, 1])
    with pytest.raises(ValueError, match="suffixes"):
        _exact_rows([assessment], [assessment], assessment=True)


def test_seed_audit_is_rebuilt_not_boolean_trusted() -> None:
    cfg = PilotConfig()
    audit = dict(
        schema="spatial-seed-audit/v1",
        namespace=cfg.namespace,
        allocation_index=0,
        config_sha256=cfg.sha256,
        effective_keys=[1, 2],
        protected_count=10,
        inventories=["historical-seeds.json"],
        collision_count=0,
        registry_sha256="a" * 64,
    )
    with patch("research.spatial_student_pilot.aggregate.audit_seeds", return_value=audit):
        _verify_seed_audit(cfg, audit)
        wrong = dict(audit, effective_keys=[1, 3])
        with pytest.raises(ValueError, match="effective_keys"):
            _verify_seed_audit(cfg, wrong)
        with pytest.raises(ValueError, match="registry_sha256"):
            _verify_seed_audit(cfg, dict(audit, registry_sha256="b" * 64))


def test_final_json_publication_is_create_only_and_sealed(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    publish_json(path, {"verdict": "INCOMPLETE"})
    assert Path(str(path) + ".sha256").is_file()
    assert Path(str(path) + ".complete").read_text() == "complete\n"
    with pytest.raises(FileExistsError):
        publish_json(path, {"verdict": "GO"})
