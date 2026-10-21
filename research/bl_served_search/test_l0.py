"""Focused tests for L0' saved-smoke bookkeeping and fail-closed census rules."""

from pathlib import Path

import pytest

from research.bl_served_search.exports import verify_export
from research.bl_served_search.l0 import (
    SMOKE,
    dense_demand_census,
    inspect,
    own_deadline_table,
    reachability_curves,
    reachable_windows,
    run,
)


def _record(identifier: str, *, rescue: bool = False) -> dict:
    return {
        "id": identifier,
        "historical_survival_rescue": rescue,
        "detection_live_frame_at_24fps": 100,
        "detections": [{"actual_live_frame": 100, "wall": 1.6}],
        "fully_evaluated_improvements": [{"frame": 112, "wall": 2.0}],
        "time_to_first_fully_evaluated_improving_action_seconds": 0.4,
        "doom": 400,
        "historical_lad_d300": 160,
    }


def test_own_deadline_table_marks_missing_fields_invalid() -> None:
    complete = _record("complete")
    incomplete = _record("incomplete")
    incomplete["fully_evaluated_improvements"] = []
    table = own_deadline_table([complete, incomplete])
    assert table["valid_rows"] == 1
    assert table["rows"][0]["own_target_slack_frames"] == 12
    assert table["rows"][0]["compute_within_own_target"] is True
    assert "fully_evaluated_improvements.frame/wall" in table["rows"][1]["missing"]


def test_reachable_windows_requires_all_rescue_rows() -> None:
    records = [_record(str(index), rescue=True) for index in range(34)]
    windows = reachable_windows(records)
    assert len(windows["rows"]) == 34
    assert windows["rows"][0]["windows"][0]["periodic_root_count_upper_bound"] == 8
    with pytest.raises(ValueError, match="expected 34"):
        reachable_windows(records[:-1])


def test_retained_smoke_exports_authorized_l0_prime_generation(tmp_path: Path) -> None:
    accounting, hashes = inspect(SMOKE)
    assert accounting["own_deadline"]["valid_rows"] == 62
    assert len(accounting["per_doom_reachable_windows"]["rows"]) == 34
    assert [row["reachable"] for row in accounting["lad_reachability"]["exact_h500"]] == [
        73,
        71,
        67,
        61,
        53,
        29,
        2,
    ]
    assert [
        row["reachable"] for row in accounting["lad_reachability"]["d300_survival_rescue_only"]
    ] == [29, 27, 27, 27, 21, 17, 1]
    assert accounting["dense_receipt_demand_census"]["status"].startswith("INCOMPLETE")
    assert len(hashes) == 240
    receipt = run(SMOKE, tmp_path / "amendment08-l0")
    result = tmp_path / "amendment08-l0" / "RESULT_L0_PRIME.md"
    payload = tmp_path / "amendment08-l0" / "l0_prime_receipt.json"
    assert receipt["verdict"] == "INCOMPLETE_MISSING_PER_BATCH_DENSE_RECEIPTS"
    assert verify_export(result)
    assert verify_export(payload)
    with pytest.raises(FileExistsError):
        run(SMOKE, tmp_path / "amendment08-l0")


def test_dense_census_never_uses_aggregate_n6_fields() -> None:
    census = dense_demand_census(SMOKE)
    assert census["n6_receipts"] == 80
    assert len(census["receipts_without_complete_batch_flags"]) == 80


def test_lad_exactly_at_latency_boundary_is_reachable() -> None:
    record = _record("boundary", rescue=True)
    record["v2_lad"] = {"lad": 124}
    record["historical_lad_d300"] = 124
    curve = reachability_curves([record])
    assert curve["exact_h500"][0]["reachable"] == 1
    assert curve["d300_survival_rescue_only"][0]["reachable"] == 1
    rows = [_record(str(index), rescue=True) for index in range(34)]
    for row in rows:
        row["historical_lad_d300"] = 124
    assert reachable_windows(rows)["rows"][0]["windows"][0]["periodic_root_count_upper_bound"] == 1
