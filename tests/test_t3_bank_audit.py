"""Unit checks for the independent T3 bank audit boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.t3_rollout_labels_20261015 import audit_bank


def test_manifest_audit_requires_world_shard_hash_binding(tmp_path: Path) -> None:
    """A receipt hash alone cannot make an absent world shard auditable."""
    bank = tmp_path / "bank"
    worlds = bank / "worlds"
    worlds.mkdir(parents=True)
    receipt = worlds / "arm-scripted-c0-i0-s7.json"
    receipt.write_text(
        json.dumps(
            {
                "key": "arm-scripted-c0-i0-s7",
                "rows_sha256": "missing-world-shard",
            }
        )
    )
    manifest = {
        "schema": "t3-label-bank/v1",
        "files": {},
        "shards": {},
        "world_receipts": {receipt.name: audit_bank.sha256(receipt)},
    }
    (bank / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(audit_bank.AuditError, match="world receipt has no world shard"):
        audit_bank._load_manifest(bank)


def test_provenance_rejects_duplicate_source_decision(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    np.savez_compressed(bank / "bank-00000.npz", action=np.zeros(0, dtype=np.int64))
    (bank / "bank-00000.provenance.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row": 0,
                        "source_key": "world",
                        "decision": 12,
                        "frame": 2000,
                        "decisive": True,
                    }
                ),
                json.dumps(
                    {
                        "row": 1,
                        "source_key": "world",
                        "decision": 12,
                        "frame": 2005,
                        "decisive": False,
                    }
                ),
            ]
        )
        + "\n"
    )
    with pytest.raises(audit_bank.AuditError, match="duplicate final provenance"):
        audit_bank._provenance(bank, ("bank-00000.npz",))


def test_expected_training_mix_is_exactly_27_9_4() -> None:
    assert [audit_bank._expected_mix(i) for i in range(40)] == ["scripted"] * 27 + ["mixed"] * 9 + [
        "frozen"
    ] * 4


def _world_fixture(
    tmp_path: Path, *, decisive: bool, export: bool = True
) -> tuple[Path, dict, dict]:
    """Build one current world and optional raw final export for helper tests."""
    bank = tmp_path / "bank"
    worlds = bank / "worlds"
    worlds.mkdir(parents=True)
    key = "b8_s0-scripted-c0-i0-s7"
    local = np.zeros((1, 7, 31, 31), dtype=np.uint8)
    global_obs = np.zeros((1, 4, 37, 37), dtype=np.uint8)
    scalars = np.zeros((1, 60), dtype=np.float32)
    scalars[0, 1] = 0.7
    mask = np.array([[True, True, False, False, False, False]], dtype=bool)
    surviving = np.array(
        (
            [[False, True, False, False, False, False]]
            if decisive
            else [[True, True, False, False, False, False]]
        ),
        dtype=bool,
    )
    source = {
        "local": local,
        "global": global_obs,
        "scalars": scalars,
        "mask": mask,
        "surviving": surviving,
        "action": np.array([0], dtype=np.int64),
        "decisive": np.array([decisive], dtype=bool),
        "decision": np.array([1], dtype=np.int64),
        "frame": np.array([2000], dtype=np.int64),
        "q": np.array([[5, 1, 0, 0, 0, 0]], dtype=np.float32),
        "length": np.array([700], dtype=np.int64),
    }
    world = worlds / f"{key}.npz"
    np.savez_compressed(world, **source)
    receipt = {
        "schema": "t3-label-world/v1",
        "status": "PASS",
        "key": key,
        "arm": "b8_s0",
        "mix": "scripted",
        "cycle": 0,
        "seed": 7,
        "roster": {
            "seed": 7,
            "mix": "scripted",
            "world_index": 0,
            "specs": [],
            "member_sha256s": [],
        },
        "run_identity": {"depth": 300, "horizon": 5000},
        "rows": 1,
        "decisive_rows": int(decisive),
        "rows_sha256": audit_bank.sha256(world),
    }
    (worlds / f"{key}.json").write_text(json.dumps(receipt))
    if export:
        final = {
            name: value.copy()
            for name, value in source.items()
            if name not in {"decisive", "decision", "frame"}
        }
        final["weight"] = np.ones(1, dtype=np.float32)
        np.savez_compressed(bank / "bank-00000.npz", **final)
        (bank / "bank-00000.provenance.jsonl").write_text(
            json.dumps(
                {
                    "row": 0,
                    "source_key": key,
                    "decision": 1,
                    "frame": 2000,
                    "decisive": decisive,
                }
            )
            + "\n"
        )
    return bank, {"bank-00000.npz": "unused"}, {key: receipt}


def test_empty_legacy_source_synthesizes_missing_row_keys(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    worlds = bank / "worlds"
    worlds.mkdir(parents=True)
    key = "legacy"
    np.savez_compressed(
        worlds / f"{key}.npz",
        local=np.zeros((0, 7, 31, 31), np.uint8),
        global_=np.zeros((0, 4, 37, 37), np.uint8),
        scalars=np.zeros((0, 60), np.float32),
        mask=np.zeros((0, 6), bool),
        surviving=np.zeros((0, 6), bool),
        action=np.zeros(0, np.int64),
        decisive=np.zeros(0, bool),
    )
    with np.load(worlds / f"{key}.npz") as archive:
        arrays = {
            "global": archive["global_"],
            **{name: archive[name] for name in archive.files if name != "global_"},
        }
    np.savez_compressed(worlds / f"{key}.npz", **arrays)
    calibration = bank / "calibration"
    calibration.mkdir()
    sidecar = calibration / f"{key}.npz"
    np.savez_compressed(
        sidecar,
        q=np.zeros((0, 6), np.float32),
        length=np.zeros(0, np.int64),
        decision=np.zeros(0, np.int64),
        frame=np.zeros(0, np.int64),
    )
    rows = audit_bank._load_source_rows(bank, key, {}, {key}, {key: audit_bank.sha256(sidecar)})
    assert rows == {}


def test_export_audit_rejects_counterweight_without_legal_alternative(tmp_path: Path) -> None:
    bank, shards, receipts = _world_fixture(tmp_path, decisive=False)
    with pytest.raises(audit_bank.AuditError, match="counterweight relation"):
        audit_bank._audit_export(
            bank, {"shards": shards}, receipts, set(), set(), {}, float("-inf")
        )


def test_export_audit_rejects_dropped_decisive_source_row(tmp_path: Path) -> None:
    bank, shards, receipts = _world_fixture(tmp_path, decisive=True, export=False)
    np.savez_compressed(
        bank / "bank-00000.npz",
        local=np.zeros((0, 7, 31, 31), np.uint8),
        **{"global": np.zeros((0, 4, 37, 37), np.uint8)},
        scalars=np.zeros((0, 60), np.float32),
        mask=np.zeros((0, 6), bool),
        surviving=np.zeros((0, 6), bool),
        action=np.zeros(0, np.int64),
        weight=np.zeros(0, np.float32),
        q=np.zeros((0, 6), np.float32),
        length=np.zeros(0, np.int64),
    )
    (bank / "bank-00000.provenance.jsonl").write_text("")
    with pytest.raises(audit_bank.AuditError, match="decisive source rows were dropped"):
        audit_bank._audit_export(
            bank, {"shards": shards}, receipts, set(), set(), {}, float("-inf")
        )


def test_export_audit_allows_unfiltered_death_window_decisive_row(tmp_path: Path) -> None:
    bank, shards, receipts = _world_fixture(tmp_path, decisive=True)
    key = next(iter(receipts))
    receipts[key]["death_decision"] = 200
    report = audit_bank._audit_export(bank, {"shards": shards}, receipts, set(), {key}, {}, 0.0)
    assert report["stats"]["decisive_rows"] == 1
    assert report["stats"]["counterweight_rows"] == 0


def test_death_window_identity_requires_mode_sources_and_validation_binding() -> None:
    receipt = {
        "run_identity": {
            "collection_mode": "death-window",
            "counterweight_ratio": 0.5,
            "source_sha256": {
                "research/t3_rollout_labels_20261015/death_window.py": "module-digest",
                "docs/research/specs/t3_death_window_spec_2026-10-15.md": "spec-digest",
            },
            "death_window_validation_sha256": "receipt-digest",
            "window_maxlen": 60,
            "window_depth": 300,
            "window_strict": True,
            "counter_reservoir": 32,
            "b8_control_every": 20,
        },
        "death_window": {
            "maxlen": 60,
            "depth": 300,
            "strict": True,
            "window_batches": 20,
            "counter_reservoir_seen": 1,
            "counter_reservoir_size": 1,
            "counter_batches": 1,
            "controls_checked": 1,
        },
        "b8_control_checks": 1,
        "batches": 21,
        "death_decision": 20,
        "b8_control_receipts": [
            {
                "decision": 10,
                "survives": False,
                "frames_executed": 11,
                "expected_death_decision": 20,
            }
        ],
    }
    audit_bank._audit_death_window_identity(receipt, "dw")

    receipt["run_identity"]["counterweight_ratio"] = 1.0
    with pytest.raises(audit_bank.AuditError, match="counterweight ratio"):
        audit_bank._audit_death_window_identity(receipt, "dw")


def test_death_window_counterweights_must_be_half_the_decisive_rows() -> None:
    assert audit_bank.COUNTERWEIGHT_RATIO == 0.5
    assert int(np.floor(7 * audit_bank.COUNTERWEIGHT_RATIO)) == 3


def test_death_window_decisive_boundary_is_strictly_less_than_depth() -> None:
    receipt = {"death_decision": 500}
    row = {"decisive": True, "q": np.zeros(6, dtype=np.float32)}
    audit_bank._audit_death_window_row(receipt, row, 0.0, "dw", 201)
    with pytest.raises(audit_bank.AuditError, match="outside the fatal window"):
        audit_bank._audit_death_window_row(receipt, row, 0.0, "dw", 200)
