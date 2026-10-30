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


def diagnostic_fixture():
    from research.spatial_student_pilot.latency import (
        DIAGNOSTIC_PROTOCOL,
        DIAGNOSTIC_SOFTWARE,
        _stats,
        source_identity,
    )

    cfg = PilotConfig()
    roots = [dict(uid=f"root{i}", world_id=i // 2) for i in range(128)]
    metadata = dict(config_sha256=cfg.sha256, seed=1901, updates=8000, namespace=cfg.namespace)
    raw = [6.533] * 12800
    report = {
        **DIAGNOSTIC_SOFTWARE,
        "schema": "spatial-student-latency/v1",
        "device": "mps",
        "threads": 1,
        "fixture_count": 128,
        "warmups_per_fixture": 12,
        "repeats_per_fixture": 100,
        "scope": DIAGNOSTIC_PROTOCOL["scope"],
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "diagnostic_complete": True,
        "source_sha256": source_identity(),
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata": metadata,
        "config_sha256": cfg.sha256,
        "namespace": cfg.namespace,
        "allocation_index": 0,
        "root_uids": [r["uid"] for r in roots],
        "parent_count": 64,
        "load_observation": "fixture desktop load",
        "statistics": _stats(raw),
        "per_fixture": [_stats(raw[:100]) for _ in roots],
        "sample_ms": raw,
        "within_p95_limit": False,
    }
    return cfg, roots, metadata, report


def test_complete_slow_diagnostic_admitted_without_threshold_flag() -> None:
    from research.spatial_student_pilot.latency import verify_diagnostic

    cfg, roots, metadata, report = diagnostic_fixture()
    assert "threshold_qualified" not in report
    assert verify_diagnostic(report, cfg, roots, "a" * 64, metadata) == 6.533


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "f" * 64),
        ("namespace", "foreign"),
        ("allocation_index", 1),
        ("checkpoint_metadata", {}),
        ("config_sha256", "wrong"),
        ("root_uids", ["foreign"] * 128),
        ("parent_count", 1),
        ("source_sha256", {}),
        ("device", "cpu"),
        ("threads", 2),
        ("warmups_per_fixture", 13),
        ("repeats_per_fixture", 101),
        ("fixture_count", 127),
        ("diagnostic_protocol", {}),
        ("diagnostic_complete", False),
        ("scope", "model-only"),
        ("system", "Linux"),
        ("torch", "different"),
        ("python", "different"),
        ("load_observation", ""),
        ("sample_ms", [1.0]),
        ("sample_ms", [float("nan")] * 12800),
        ("sample_ms", [float("inf")] * 12800),
        ("sample_ms", [True] * 12800),
        ("sample_ms", [-1.0] * 12800),
        ("sample_ms", ["6.533"] * 12800),
        ("statistics", {"p95_ms": 6.533}),
        ("per_fixture", []),
        ("within_p95_limit", True),
    ],
)
def test_diagnostic_malformed_or_identity_invalid_is_rejected(field, value) -> None:
    from research.spatial_student_pilot.latency import verify_diagnostic

    cfg, roots, metadata, report = diagnostic_fixture()
    report[field] = value
    with pytest.raises(ValueError):
        verify_diagnostic(report, cfg, roots, "a" * 64, metadata)


def test_diagnostic_requires_raw_samples_order_and_statistics() -> None:
    from research.spatial_student_pilot.latency import verify_diagnostic

    cfg, roots, metadata, report = diagnostic_fixture()
    report.pop("sample_ms")
    with pytest.raises(ValueError, match="raw latency"):
        verify_diagnostic(report, cfg, roots, "a" * 64, metadata)
    cfg, roots, metadata, report = diagnostic_fixture()
    report["root_uids"].reverse()
    with pytest.raises(ValueError, match="identity/protocol"):
        verify_diagnostic(report, cfg, roots, "a" * 64, metadata)


def test_diagnostic_rejects_boolean_numeric_identities_and_statistics() -> None:
    from research.spatial_student_pilot.latency import _stats, verify_diagnostic

    cfg, roots, metadata, report = diagnostic_fixture()
    report["threads"] = True
    with pytest.raises(ValueError, match="identity/protocol"):
        verify_diagnostic(report, cfg, roots, "a" * 64, metadata)
    cfg, roots, metadata, report = diagnostic_fixture()
    report["sample_ms"] = [1.0] * 12800
    report["statistics"] = _stats(report["sample_ms"])
    report["per_fixture"] = [_stats([1.0] * 100) for _ in roots]
    report["within_p95_limit"] = True
    assert verify_diagnostic(report, cfg, roots, "a" * 64, metadata) == 1.0
    report["statistics"]["p95_ms"] = True
    with pytest.raises(ValueError, match="statistics"):
        verify_diagnostic(report, cfg, roots, "a" * 64, metadata)
