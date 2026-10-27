"""Synthetic alignment checks for the served-tick background covariate report."""

import pytest

from research.bl_served_search.background_report import summarize_background


def _sample(start, end, background=10, workload=0, unknown=0, complete=True):
    return {
        "start_ns": start,
        "end_ns": end,
        "background_cpu_percent": background,
        "workload_cpu_percent": workload,
        "unknown_cpu_percent": unknown,
        "coverage_complete": complete,
        "process_churn": [],
        "processes": [
            {
                "pid": 1,
                "ppid": 0,
                "command": "/System/mds",
                "kind": "macos_service",
                "cpu_percent": background,
            }
        ],
    }


def _row(tick, start, end, **extra):
    return {
        "tick": tick,
        "target_ns": start,
        "wake_ns": start,
        "start_ns": start,
        "end_ns": end,
        "start_cpu_ns": 10,
        "end_cpu_ns": 20,
        "spans": {"game_update": {"wall_ns": 5}},
        **extra,
    }


def _result(rows, warmup=0):
    return {"warmup_ticks": warmup, "trace": rows}


def _host(samples):
    return {"load_policy": "classified-v1", "cpu_samples": samples}


def test_weighted_boundary_overlap_and_low_window():
    report = summarize_background(
        _result([_row(0, 50, 250)]), _host([_sample(0, 100, 10), _sample(100, 300, 30)])
    )
    tick = report["ticks"][0]
    assert tick["sample_indexes"] == [0, 1]
    assert tick["background_cpu_percent_time_weighted_mean"] == pytest.approx(25)
    assert tick["background_cpu_percent_max"] == 30
    assert not tick["low_background"]
    assert report["timing"]["overall"]["count"] == 1


def test_missing_or_long_coverage_cannot_be_low_background():
    report = summarize_background(
        _result([_row(0, 0, 100), _row(1, 100, 200)]),
        _host([_sample(0, 100, 1), _sample(150, 200, 1)]),
    )
    assert report["ticks"][0]["low_background"]
    assert not report["ticks"][1]["coverage_complete"]
    assert report["timing"]["low_background"]["count"] == 1
    report = summarize_background(
        _result([_row(0, 0, 100)]), _host([_sample(0, 6_000_000_000, 1, complete=False)])
    )
    assert not report["ticks"][0]["coverage_complete"]


def test_measured_trace_is_not_sliced_again_and_no_low_windows_is_reported():
    report = summarize_background(
        _result([_row(0, 0, 10), _row(1, 10, 20)], warmup=1), _host([_sample(0, 30, 21)])
    )
    assert [tick["tick"] for tick in report["ticks"]] == [0, 1]
    assert report["timing"]["low_background"]["count"] == 0
    assert report["timing"]["low_background"]["metrics"]["tick_wall_ns"]["count"] == 0
    assert report["timing"]["low_background"]["metrics"]["tick_wall_ns"]["p95"] is None


def test_production_window_includes_snapshot_and_client_receipt():
    row = _row(
        4,
        100,
        120,
        snapshot_end_ns=140,
        snapshot_start_cpu_ns=30,
        snapshot_end_cpu_ns=35,
        game_age=9,
    )
    result = {
        "warmup_ticks": 1,
        "trace": [row],
        "client_receive": [
            {"source": "hub_broadcast", "receive_ns": 50, "frame": 8},
            {"source": "hub_broadcast", "receive_ns": 180, "frame": 9},
        ],
        "frame_send": [
            {"frame": 8, "start_ns": 40, "status": "ok"},
            {"frame": 9, "start_ns": 150, "status": "ok"},
        ],
    }
    report = summarize_background(result, _host([_sample(100, 200)]), production=True)
    assert report["ticks"][0]["window_end_ns"] == 180
    assert report["timing"]["overall"]["metrics"]["snapshot_thread_cpu_ns"]["max"] == 5
    assert report["timing"]["overall"]["metrics"]["step_to_client_ns"]["max"] == 80
    assert "absolute_overrun_ns" not in report["timing"]["overall"]["metrics"]


def test_low_background_client_gaps_do_not_bridge_excluded_ticks():
    rows = [
        _row(
            tick,
            tick * 100,
            tick * 100 + 10,
            snapshot_end_ns=tick * 100 + 20,
            snapshot_start_cpu_ns=30,
            snapshot_end_cpu_ns=35,
        )
        for tick in range(3)
    ]
    result = {
        "warmup_ticks": 0,
        "trace": rows,
        "client_receive": [
            {"source": "hub_broadcast", "receive_ns": tick * 100 + 50, "frame": tick}
            for tick in range(3)
        ],
        "frame_send": [
            {"frame": tick, "start_ns": tick * 100 + 30, "status": "ok"} for tick in range(3)
        ],
    }
    report = summarize_background(
        result,
        _host([_sample(0, 100, 1), _sample(100, 200, 30), _sample(200, 300, 1)]),
        production=True,
    )
    assert report["timing"]["overall"]["metrics"]["client_frame_gap_ns"]["count"] == 2
    assert report["timing"]["low_background"]["metrics"]["client_frame_gap_ns"]["count"] == 0


def test_low_background_gap_requires_clean_interval_between_low_tick_endpoints():
    rows = [
        _row(
            tick,
            tick * 100,
            tick * 100 + 10,
            snapshot_end_ns=tick * 100 + 20,
            snapshot_start_cpu_ns=30,
            snapshot_end_cpu_ns=35,
        )
        for tick in (0, 1)
    ]
    result = {
        "warmup_ticks": 0,
        "trace": rows,
        "client_receive": [
            {"source": "hub_broadcast", "receive_ns": 50, "frame": 0},
            {"source": "hub_broadcast", "receive_ns": 150, "frame": 1},
        ],
        "frame_send": [
            {"frame": 0, "start_ns": 30, "status": "ok"},
            {"frame": 1, "start_ns": 130, "status": "ok"},
        ],
    }
    report = summarize_background(
        result,
        _host([_sample(0, 50, 1), _sample(50, 100, 30), _sample(100, 200, 1)]),
        production=True,
    )
    assert report["timing"]["low_background"]["count"] == 2
    assert report["timing"]["low_background"]["metrics"]["client_frame_gap_ns"]["count"] == 0


def test_coverage_false_retains_observed_covariates_and_cpu_rows_must_reconcile():
    report = summarize_background(
        _result([_row(0, 0, 100)]), _host([_sample(0, 100, 17, complete=False)])
    )
    tick = report["ticks"][0]
    assert not tick["coverage_complete"]
    assert tick["coverage_fraction"] == 0
    assert tick["background_cpu_percent_time_weighted_mean"] == 17
    bad = _sample(0, 100, 17)
    bad["background_cpu_percent"] = 18
    with pytest.raises(ValueError, match="reconcile"):
        summarize_background(_result([_row(0, 0, 100)]), _host([bad]))


@pytest.mark.parametrize(
    "samples",
    [
        [_sample(0, 10), _sample(9, 20)],
        [
            {
                "start_ns": 0,
                "end_ns": 10,
                "background_cpu_percent": 0,
                "workload_cpu_percent": 0,
                "unknown_cpu_percent": 0,
                "coverage_complete": True,
                "processes": [{"pid": 1}],
            }
        ],
    ],
)
def test_invalid_host_evidence_fails_closed(samples):
    with pytest.raises(ValueError):
        summarize_background(_result([_row(0, 0, 10)]), _host(samples))


def test_invalid_trace_times_fail_closed():
    with pytest.raises(ValueError, match="trace end"):
        summarize_background(_result([_row(0, 20, 10)]), _host([_sample(0, 30)]))


def test_zero_client_gap_is_retained_in_overall_and_low_background():
    result = {
        "warmup_ticks": 0,
        "trace": [
            _row(
                tick,
                tick * 100,
                tick * 100 + 10,
                snapshot_end_ns=tick * 100 + 20,
                snapshot_start_cpu_ns=30,
                snapshot_end_cpu_ns=35,
            )
            for tick in range(2)
        ],
        "client_receive": [
            {"source": "hub_broadcast", "receive_ns": 150, "frame": tick} for tick in range(2)
        ],
        "frame_send": [
            {"frame": tick, "start_ns": tick * 100 + 30, "status": "ok"} for tick in range(2)
        ],
    }
    report = summarize_background(result, _host([_sample(0, 200)]), production=True)
    assert report["timing"]["overall"]["metrics"]["client_frame_gap_ns"]["max"] == 0
    overall = report["timing"]["overall"]["metrics"]["client_frame_gap_ns"]
    low = report["timing"]["low_background"]["metrics"]["client_frame_gap_ns"]
    assert low == overall
    assert low["count"] == 1
    assert low["max"] == 0


@pytest.mark.parametrize("fps", [12, 30, 60])
def test_nonproduction_overruns_use_configured_rational_fps(fps):
    from research.bl_served_search.tick_trace import reduce_trace

    epoch = 100
    first_deadline = epoch + 1_000_000_000 // fps
    second_deadline = epoch + 2_000_000_000 // fps
    rows = [
        _row(0, epoch, first_deadline - 1, spans={}),
        _row(1, first_deadline, second_deadline + 7, spans={}),
    ]
    report = summarize_background(
        _result(rows), _host([_sample(0, second_deadline + 100)]), fps=fps
    )
    summary = reduce_trace(rows, expected_ticks=2, fps=fps, require_update=False)
    assert summary["verdict"] == "FAIL"
    timing = report["timing"]["overall"]
    assert timing["absolute_overrun_count"] == summary["absolute_overruns"] == 1
    assert timing["metrics"]["absolute_overrun_severity_ns"]["max"] == 7
    assert summary["overrun_severity_ns"]["max"] == 7


@pytest.mark.parametrize("fps", [0, -1])
def test_background_report_refuses_nonpositive_fps(fps):
    from research.bl_served_search.background_report import _metric_rows

    rows = [_row(0, 0, 100)]
    with pytest.raises(ValueError, match="fps"):
        summarize_background(_result(rows), _host([_sample(0, 200)]), fps=fps)
    with pytest.raises(ValueError, match="fps"):
        _metric_rows(rows, False, [], [], fps=fps)


@pytest.mark.parametrize("churn", [None, {}, "churn", 1])
def test_background_report_refuses_nonlist_process_churn(churn):
    sample = _sample(0, 100)
    sample["process_churn"] = churn
    with pytest.raises(ValueError, match="churn"):
        summarize_background(_result([_row(0, 0, 10)]), _host([sample]))


@pytest.mark.parametrize("change", ["missing_churn", "churn", "long"])
def test_background_report_refuses_inconsistent_complete_coverage(change):
    sample = _sample(0, 100)
    if change == "missing_churn":
        del sample["process_churn"]
    elif change == "churn":
        sample["process_churn"] = [{"pid": 20, "reason": "new_or_exec_process_no_delta"}]
    else:
        sample["end_ns"] = 5_000_000_001
    with pytest.raises(ValueError):
        summarize_background(_result([_row(0, 0, 10)]), _host([sample]))


def test_incomplete_churn_sample_retains_measured_cpu_but_cannot_be_low():
    sample = _sample(0, 100, complete=False)
    sample["process_churn"] = [{"pid": 20, "reason": "new_or_exec_process_no_delta"}]
    report = summarize_background(_result([_row(0, 0, 10)]), _host([sample]))
    assert not report["ticks"][0]["low_background"]
    assert report["ticks"][0]["background_cpu_percent_time_weighted_mean"] == 10
