"""Synthetic v2 supervision reporting checks; no timing panel is executed here."""

from __future__ import annotations

import pytest

from research.bl_served_search.supervision_report import (
    _charges,
    _rolling_cpu_max,
    summarize_scope,
    summarize_supervision,
)


def _row(tick: int, start: int, end: int, **extra: int) -> dict:
    return {
        "tick": tick,
        "target_ns": start,
        "wake_ns": start,
        "start_ns": start,
        "end_ns": end,
        "start_cpu_ns": 1,
        "end_cpu_ns": 2,
        "spans": {"game_update": {"wall_ns": 1}},
        **extra,
    }


def _host(samples: list[tuple[int, int, float]]) -> dict:
    rows = []
    for start, end, background in samples:
        rows.append(
            {
                "start_ns": start,
                "end_ns": end,
                "coverage_complete": True,
                "process_churn": [],
                "background_cpu_percent": background,
                "workload_cpu_percent": 0.0,
                "unknown_cpu_percent": 0.0,
                "processes": [
                    {
                        "pid": 1,
                        "ppid": 0,
                        "command": "/System/Library/mds",
                        "kind": "macos_service",
                        "cpu_percent": background,
                    }
                ],
            }
        )
    return {"load_policy": "classified-supervised-v2", "cpu_samples": rows}


def _replay(intervals: list[tuple[int, int, float]], operations: list[dict] | None = None) -> dict:
    interval_rows = [
        {
            "start_ns": start,
            "end_ns": end,
            "observed_cpu_pct": cpu,
            "cpu_pct": cpu,
            "observed_cpu_s": cpu * (end - start) / 1e11,
            "conservative_cpu_s": cpu * (end - start) / 1e11,
            "conservative_cpu_numerator": int(cpu * (end - start) / 100),
            "counter_denominator": 1,
            "roots": {
                "root-a": {
                    "observed_cpu_s": cpu * (end - start) / 1e11,
                    "conservative_cpu_s": cpu * (end - start) / 1e11,
                }
            },
        }
        for start, end, cpu in intervals
    ]
    return {
        "policy": "classified-supervised-v2",
        "valid": True,
        "complete": True,
        "intervals": interval_rows,
        "charges": [
            {
                "id": f"charge-{index}",
                "root_id": "root-a",
                "start_ns": start,
                "end_ns": end,
                "cpu_s": cpu * (end - start) / 1e11,
                "conservative_cpu_s": cpu * (end - start) / 1e11,
            }
            for index, (start, end, cpu) in enumerate(intervals)
        ],
        "operations": {
            str(row["id"]): {
                **row,
                "observed_cpu_s": row.pop("cpu_s"),
                "conservative_cpu_s": row.get("conservative_cpu_s", row.get("cpu_s", 0.0)),
                "complete": True,
            }
            for row in (operations or [])
        },
        "roots": {
            "root-a": {
                "observed_cpu_s": sum(row["observed_cpu_s"] for row in interval_rows),
                "conservative_cpu_s": sum(row["conservative_cpu_s"] for row in interval_rows),
                "process_count": 1,
            }
        },
        "observed": {"max_10s_cpu_s": 0.2, "max_60s_cpu_s": 0.2},
        "conservative": {"max_10s_cpu_s": 0.2, "max_60s_cpu_s": 0.2},
    }


def test_all_ticks_are_retained_while_poll_overlap_is_only_a_subset_exclusion() -> None:
    report = summarize_supervision(
        {
            "warmup_ticks": 0,
            "trace": [_row(0, 0, 1_000_000_000), _row(1, 1_000_000_000, 2_000_000_000)],
        },
        _host([(0, 2_000_000_000, 5)]),
        _replay(
            [(0, 1_000_000_000, 5), (1_000_000_000, 2_000_000_000, 5)],
            [
                {
                    "id": "poll",
                    "root_id": "root-a",
                    "start_ns": 100_000_000,
                    "end_ns": 200_000_000,
                    "cpu_s": 0.1,
                }
            ],
        ),
    )
    assert report["timing"]["overall"]["count"] == 2
    assert report["counts"]["operation_overlap"] == 2  # one-second padding reaches both ticks
    assert report["timing"]["low_background"]["count"] == 0


def test_expanded_operation_boundary_and_empty_subset_are_explicit() -> None:
    report = summarize_supervision(
        {"warmup_ticks": 0, "trace": [_row(0, 0, 500_000_000)]},
        _host([(0, 500_000_000, 1)]),
        _replay(
            [(0, 500_000_000, 1)],
            [
                {
                    "id": "edge",
                    "root_id": "root-a",
                    "start_ns": 1_400_000_000,
                    "end_ns": 1_500_000_000,
                    "cpu_s": 0.0,
                }
            ],
        ),
    )
    assert report["ticks"][0]["operation_overlap_ids"] == ["edge"]
    assert report["timing"]["low_background"]["count"] == 0
    assert report["timing"]["low_background"]["metrics"]["tick_wall_ns"]["p95"] is None


def test_scope_reports_root_and_operation_accounting() -> None:
    replay = _replay(
        [(0, 1_000_000_000, 10)],
        [{"id": "heartbeat", "root_id": "root-a", "start_ns": 10, "end_ns": 20, "cpu_s": 0.25}],
    )
    scope = summarize_scope(0, 1_000_000_000, replay)
    assert scope["roots"]["root-a"]["operation_count"] == 1
    assert scope["max_operation_cpu_s"] == 0.25
    assert scope["observed_max_10s_cpu_s"] == 0.1


def test_conservative_rolling_debt_carries_into_the_next_cell() -> None:
    replay = _replay([(0, 1_000_000_000, 10)])
    replay["charges"] = [
        {
            "id": "unplaced-final-cpu",
            "root_id": "root-a",
            "start_ns": 0,
            "end_ns": 1_000_000_000,
            "cpu_s": 0.1,
            "conservative_cpu_s": 0.7,
        }
    ]
    scope = summarize_scope(5_000_000_000, 6_000_000_000, replay)
    assert scope["conservative_max_10s_cpu_s"] == 0.7


def test_scope_root_conservative_cpu_uses_full_charge_not_interval_fraction() -> None:
    replay = _replay([(0, 1_000_000_000, 10)])
    replay["charges"][0]["conservative_cpu_s"] = 0.7
    scope = summarize_scope(100_000_000, 200_000_000, replay)
    assert scope["roots"]["root-a"]["root_conservative_cpu_s"] == 0.7


def test_exact_rational_supervision_over_twenty_is_not_float_rounded_down() -> None:
    result = {"warmup_ticks": 0, "trace": [_row(0, 0, 1_000_000_000)]}
    replay = _replay([(0, 1_000_000_000, 20)])
    interval = replay["intervals"][0]
    interval["counter_denominator"] = 10**12
    interval["conservative_cpu_numerator"] = (
        20 * interval["counter_denominator"] * 1_000_000_000 // 100 + 1
    )
    assert interval["cpu_pct"] == 20  # the legacy float presentation rounds here
    report = summarize_supervision(result, _host([(0, 1_000_000_000, 0)]), replay)
    assert report["timing"]["low_background"]["count"] == 0


def test_rolling_max_checks_strictly_between_charge_boundaries() -> None:
    charges = [
        {
            "root_id": "root-a",
            "start_ns": 0,
            "end_ns": 1_000_000_000,
            "cpu_s": 0.2,
            "conservative_cpu_s": 0.2,
        },
        {
            "root_id": "root-a",
            "start_ns": 10_000_000_000,
            "end_ns": 11_000_000_000,
            "cpu_s": 0.3,
            "conservative_cpu_s": 0.3,
        },
    ]
    assert (
        _rolling_cpu_max(charges, 0, 11_000_000_000, 10_000_000_000, "cpu_s", conservative=False)
        == 0.3
    )
    assert (
        _rolling_cpu_max(
            charges, 0, 11_000_000_000, 10_000_000_000, "conservative_cpu_s", conservative=True
        )
        == 0.5
    )


def test_repeated_operation_or_root_charge_ids_remain_distinct() -> None:
    charges = _charges(
        {
            "charges": [
                {
                    "operation_id": "poll",
                    "start_ns": 0,
                    "end_ns": 1,
                    "cpu_s": 0.1,
                    "conservative_cpu_s": 0.1,
                },
                {
                    "operation_id": "poll",
                    "start_ns": 1,
                    "end_ns": 2,
                    "cpu_s": 0.1,
                    "conservative_cpu_s": 0.1,
                },
                {
                    "operation_id": None,
                    "start_ns": 2,
                    "end_ns": 3,
                    "cpu_s": 0.1,
                    "conservative_cpu_s": 0.1,
                },
            ]
        }
    )
    assert [charge["id"] for charge in charges] == ["charge-0", "charge-1", "charge-2"]


def test_combined_background_and_supervision_boundary_is_conservative() -> None:
    base = {"warmup_ticks": 0, "trace": [_row(0, 0, 1_000_000_000)]}
    equal = summarize_supervision(
        base, _host([(0, 1_000_000_000, 15)]), _replay([(0, 1_000_000_000, 5)])
    )
    above = summarize_supervision(
        base, _host([(0, 1_000_000_000, 15.1)]), _replay([(0, 1_000_000_000, 5)])
    )
    assert equal["timing"]["low_background"]["count"] == 1
    assert above["timing"]["low_background"]["count"] == 0


def test_exact_v2_policy_and_conservative_cpu_are_required() -> None:
    result = {"warmup_ticks": 0, "trace": [_row(0, 0, 1_000_000_000)]}
    with pytest.raises(ValueError, match="supervised host"):
        summarize_supervision(
            result,
            {**_host([(0, 1_000_000_000, 1)]), "load_policy": "classified-v1"},
            _replay([(0, 1_000_000_000, 1)]),
        )
    replay = _replay([(0, 1_000_000_000, 1)])
    del replay["intervals"][0]["cpu_pct"]
    with pytest.raises(ValueError, match="conservative"):
        summarize_supervision(result, _host([(0, 1_000_000_000, 1)]), replay)
    replay = _replay([(0, 1_000_000_000, 1)])
    replay["intervals"][0]["counter_denominator"] = 0
    with pytest.raises(ValueError, match="denominator"):
        summarize_supervision(result, _host([(0, 1_000_000_000, 1)]), replay)


def test_production_gap_does_not_bridge_a_permitted_operation() -> None:
    rows = [
        _row(
            tick,
            tick * 2_000_000_000,
            tick * 2_000_000_000 + 100,
            snapshot_end_ns=tick * 2_000_000_000 + 100,
            snapshot_start_cpu_ns=1,
            snapshot_end_cpu_ns=2,
        )
        for tick in range(2)
    ]
    result = {
        "warmup_ticks": 0,
        "trace": rows,
        "client_receive": [
            {"source": "hub_broadcast", "frame": 0, "receive_ns": 100},
            {"source": "hub_broadcast", "frame": 1, "receive_ns": 2_000_000_100},
        ],
        "frame_send": [
            {"frame": 0, "start_ns": 50, "status": "ok"},
            {"frame": 1, "start_ns": 2_000_000_050, "status": "ok"},
        ],
    }
    report = summarize_supervision(
        result,
        _host([(0, 2_000_001_000, 1)]),
        _replay(
            [(0, 2_000_001_000, 1)],
            [
                {
                    "id": "poll",
                    "root_id": "root-a",
                    "start_ns": 900_000_000,
                    "end_ns": 900_000_100,
                    "cpu_s": 0.0,
                }
            ],
        ),
        production=True,
    )
    assert report["timing"]["overall"]["metrics"]["client_frame_gap_ns"]["count"] == 1
    assert report["timing"]["low_background"]["metrics"]["client_frame_gap_ns"]["count"] == 0
