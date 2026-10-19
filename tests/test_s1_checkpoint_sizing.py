"""Actual serializer/capacity contracts and fail-closed measured export pricing."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.s1_mature_20261018 import checkpoint_sizing as sizing


@pytest.mark.parametrize("arm,actors,seats", [("n3", 6, 3), ("n1", 3, 1)])
def test_registered_raw_capacity_matches_actual_ring_layout(arm, actors, seats) -> None:
    result = sizing.capacity(arm)
    rows = int(1.5 * 500_000 / actors) + 128 * seats * (2 + 8 + 16)
    row_bytes = 7 * 31 * 31 + 4 * 37 * 37 + 60 * 4 + 6
    assert result["ring_row_bytes"] == row_bytes
    assert result["fixed_capacity_array_bytes"] == actors * rows * row_bytes + 20_800_000
    assert result["demonstration_buffer_bytes"] == 0
    assert result["snapshot_targets"] == [2_000_000, 4_000_000, 6_000_000, 8_000_000]


def test_default_numpy_serializer_requires_latin1_expansion_bound() -> None:
    low, high = io.BytesIO(), io.BytesIO()
    torch.save(np.zeros(100_000, np.uint8), low)
    torch.save(np.full(100_000, 255, np.uint8), high)
    assert len(high.getvalue()) > 1.9 * len(low.getvalue())
    assert len(high.getvalue()) < 2 * 100_000 + 4096


def _write(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value))
    return sizing.binding(path)


def qualified(tmp_path: Path) -> tuple[dict, dict]:
    sources = {
        relative: sizing.binding(sizing.ROOT / relative)["sha256"] for relative in sizing.SOURCES
    }
    transfer = {
        "status": "PASS",
        "hardware": "L4",
        "protocol": "s1-chunked-export/v1",
        "verified_on_mac": True,
        "transport_source_sha256": sources["research/s1_mature_20261018/train.py"],
        "transport_source_manifest": {
            relative: sizing.binding(sizing.ROOT / relative)["sha256"]
            for relative in sizing.CLI_SOURCES
        },
        "exported_bytes": 2 * 1024**3,
        "elapsed_seconds": 20,
        "bytes_per_second_lower_bound": 100_000_000,
    }
    receipt = {
        "schema": "s1-full-replay-transport/v1",
        "status": "PASS",
        "source_manifest": sources,
        "timing_receipt_sha256": "timing",
        "cli_acceptance_receipt": _write(
            tmp_path / "acceptance.json", {"qualified": True, "s1_full_replay_transfer": transfer}
        ),
        "arms": {},
    }
    for arm in ("n3", "n1"):
        raw = sizing.capacity(arm)["fixed_capacity_array_bytes"]
        observation = {
            "status": "PASS",
            "arm": arm,
            "production_shape_validated": True,
            "source_manifest": sources,
            "capacity": sizing.capacity(arm),
            "deferred_barrier_packets_included": True,
            "hardware": "L4",
            "preparation_scope": "capture-save-hash-archive-chunk-copy",
            "snapshot_bytes": raw,
            "prepare_seconds": 10,
            "final_archive_bytes": 100_000_000,
            "final_archive_prepare_seconds": 1,
        }
        receipt["arms"][arm] = {
            "full_size_checkpoint_measurement": _write(tmp_path / f"{arm}.json", observation),
            "full_horizon_size_bound": _write(
                tmp_path / f"{arm}-size-bound.json",
                {
                    "schema": "s1-full-horizon-checkpoint-size-bound/v1",
                    "status": "PASS",
                    "arm": arm,
                    "source_manifest": sources,
                    "snapshot_targets": list(sizing.SNAPSHOTS),
                    "independent_validation": True,
                    "includes_all_noncapacity_state_and_framing": True,
                    "additional_serialized_bytes_upper_bound": 100_000_000,
                },
            ),
            "snapshot_bytes_upper_bound": 2 * raw + 100_000_000,
            "snapshot_prepare_seconds_upper_bound": 20,
            "final_archive_bytes_upper_bound": 200_000_000,
            "final_archive_prepare_seconds_upper_bound": 2,
        }
    return receipt, transfer


@pytest.mark.parametrize("extra", [1000, 0, 99_999_999])
def test_one_observed_snapshot_cannot_bound_all_future_snapshots(
    tmp_path: Path, extra: int
) -> None:
    receipt, _ = qualified(tmp_path)
    raw = sizing.capacity("n3")["fixed_capacity_array_bytes"]
    receipt["arms"]["n3"]["snapshot_bytes_upper_bound"] = (
        raw + extra if extra == 1000 else 2 * raw + extra
    )
    with pytest.raises(ValueError, match="full-horizon array expansion or additional state"):
        sizing.transport_bound(receipt, timing_sha256="timing")


def test_all_registered_snapshot_targets_need_independent_size_validation(tmp_path: Path) -> None:
    receipt, _ = qualified(tmp_path)
    proof = receipt["arms"]["n3"]["full_horizon_size_bound"]
    value = json.loads(Path(proof["path"]).read_text())
    value["snapshot_targets"] = [2_000_000]
    receipt["arms"]["n3"]["full_horizon_size_bound"] = _write(Path(proof["path"]), value)
    with pytest.raises(ValueError, match="validated full-horizon"):
        sizing.transport_bound(receipt, timing_sha256="timing")


def test_all_four_snapshots_and_final_are_charged_additively(tmp_path: Path) -> None:
    receipt, _ = qualified(tmp_path)
    result = sizing.transport_bound(receipt, timing_sha256="timing")
    for arm, minutes in (("n3", 30), ("n1", 68)):
        row = result["qualified_arms"][arm]
        size = row["snapshot_archive_bytes_upper_bound"]
        assert row["total_export_bytes_upper_bound"] == 4 * size + 200_000_000
        assert row["export_seconds_upper_bound"] == (4 * size + 200_000_000) / 100_000_000
        assert row["per_run_gpu_hours_upper_bound"] == pytest.approx(
            minutes / 60 + (4 * 20 + 2 + (4 * size + 200_000_000) / 100_000_000) / 3600
        )


@pytest.mark.parametrize("rate", [None, 0, -1, float("nan"), float("inf"), True, 200_000_000])
def test_invalid_or_optimistic_acceptance_throughput_refuses(tmp_path: Path, rate) -> None:
    receipt, transfer = qualified(tmp_path)
    transfer["bytes_per_second_lower_bound"] = rate
    receipt["cli_acceptance_receipt"] = _write(
        tmp_path / "acceptance.json", {"qualified": True, "s1_full_replay_transfer": transfer}
    )
    with pytest.raises(ValueError):
        sizing.transport_bound(receipt, timing_sha256="timing")


def test_missing_transfer_field_and_changed_receipt_refuse(tmp_path: Path) -> None:
    receipt, _ = qualified(tmp_path)
    receipt["cli_acceptance_receipt"] = _write(tmp_path / "acceptance.json", {"qualified": True})
    with pytest.raises(ValueError, match="throughput"):
        sizing.transport_bound(receipt, timing_sha256="timing")
    (tmp_path / "acceptance.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        sizing.transport_bound(receipt, timing_sha256="timing")


def test_capacity_formula_alone_cannot_qualify_full_state(tmp_path: Path) -> None:
    receipt, _ = qualified(tmp_path)
    del receipt["arms"]["n3"]["full_size_checkpoint_measurement"]
    with pytest.raises(ValueError, match="requires a path"):
        sizing.transport_bound(receipt, timing_sha256="timing")


def test_transfer_and_preparation_source_or_hardware_cannot_drift(tmp_path: Path) -> None:
    receipt, transfer = qualified(tmp_path)
    transfer["transport_source_manifest"] = {}
    receipt["cli_acceptance_receipt"] = _write(
        tmp_path / "acceptance.json", {"qualified": True, "s1_full_replay_transfer": transfer}
    )
    with pytest.raises(ValueError, match="source identity"):
        sizing.transport_bound(receipt, timing_sha256="timing")
    receipt, _ = qualified(tmp_path)
    measurement = json.loads((tmp_path / "n3.json").read_text())
    measurement["hardware"] = "Mac"
    receipt["arms"]["n3"]["full_size_checkpoint_measurement"] = _write(
        tmp_path / "n3.json", measurement
    )
    with pytest.raises(ValueError, match="production-shape"):
        sizing.transport_bound(receipt, timing_sha256="timing")
