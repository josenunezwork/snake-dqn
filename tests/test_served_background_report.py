"""Synthetic alignment checks for the served-tick background covariate report."""

import pytest

from research.bl_served_search.background_report import (
    DESKTOP_POLICY,
    desktop_overall,
    summarize_background,
    validate_desktop_host,
)


@pytest.fixture(autouse=True)
def _fixture_desktop_registry_replay(monkeypatch):
    """Keep synthetic covariate tests independent of a sealed history fixture.

    The ``a...a`` digest is reserved for the small, deliberately synthetic
    covariate fixtures in this module.  Production exports always replay their
    complete history sequence through the real pure classifier.  These mocks
    model only the pre-existing fixture semantics, including workload and
    system precedence, so tests which alter an ordinary command still exercise
    the report's exact-classification check.
    """
    from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS
    from research.bl_served_search import desktop_apps

    original_history = desktop_apps.replay_history
    original_reference = desktop_apps.replay_reference
    fixture_sha = "a" * 64
    fixture_desktops = {
        "/Applications/Microsoft Outlook.app/Contents/MacOS/Microsoft Outlook",
        "/Applications/Claude.app/Contents/MacOS/Claude",
        "/Applications/X.app/Renderer",
    }

    def replay_history(registry):
        if registry.get("registry_sha256") == fixture_sha:
            return {}
        return original_history(registry)

    def replay_reference(row, registry, decisions=None):
        if registry.get("registry_sha256") != fixture_sha:
            return original_reference(row, registry, decisions)
        command = row.get("command")
        if not isinstance(command, str):
            return {"kind": "unknown", "reason": "missing_command"}
        if row.get("known_workload") or row.get("workload"):
            return {"kind": "agent_workload", "reason": "known_workload"}
        if any(
            exclusion.get("exact_executable") == command
            for exclusion in registry.get("registry", {}).get("workload_exclusions", [])
        ):
            return {"kind": "agent_workload", "reason": "workload_exclusion"}
        if command.startswith(SYSTEM_EXECUTABLE_ROOTS) and ".." not in command.split("/"):
            return {"kind": "macos_service", "reason": "system_executable"}
        classification = row.get("classification")
        if (
            command in fixture_desktops
            or command == "python"
            or (
                isinstance(classification, dict)
                and isinstance(classification.get("classifier_decision"), dict)
            )
        ) and isinstance(classification, dict):
            return dict(classification)
        return {"kind": "unknown"}

    monkeypatch.setattr(desktop_apps, "replay_history", replay_history)
    monkeypatch.setattr(desktop_apps, "replay_reference", replay_reference)


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


def _desktop_host(samples):
    registry = {"policy": DESKTOP_POLICY, "workload_exclusions": []}
    bindings = []
    for sample in samples:
        for process in sample["processes"]:
            if process["kind"] == "desktop_app":
                bindings.append(
                    {
                        **process["classification"],
                        "pid": process["pid"],
                        "birth": process["birth"],
                        "uid": process["uid"],
                        "command": process["command"],
                    }
                )
    return {
        "load_policy": DESKTOP_POLICY,
        "desktop_registry": registry,
        "desktop_registry_sha256": "a" * 64,
        "desktop_registry_evidence": {
            "policy": DESKTOP_POLICY,
            "registry": registry,
            "registry_sha256": "a" * 64,
            "bindings": bindings,
        },
        "cpu_samples": samples,
    }


def _desktop_sample(start, end, *, app=0, renderer=0, service=0, workload=0, unknown=0):
    processes = []
    if service:
        processes.append(
            {
                "pid": 1,
                "ppid": 0,
                "command": "/System/mds",
                "kind": "macos_service",
                "cpu_percent": service,
            }
        )
    if renderer:
        processes.append(
            {
                "pid": 2,
                "ppid": 1,
                "command": "/Applications/X.app/Renderer",
                "kind": "desktop_renderer",
                "cpu_percent": renderer,
            }
        )
    if app:
        processes.append(
            {
                "pid": 3,
                "ppid": 1,
                "birth": "fixture-birth",
                "uid": 501,
                "command": "/Applications/Microsoft Outlook.app/Contents/MacOS/Microsoft Outlook",
                "kind": "desktop_app",
                "cpu_percent": app,
                "entry": "outlook-main",
                "application": "Microsoft Outlook",
                "role": "mail-ui",
                "reason": "enrolled_verified_ui_role",
                "identity": {"signed": True},
            }
        )
    if workload:
        processes.append(
            {
                "pid": 4,
                "ppid": 1,
                "command": "python",
                "kind": "agent_workload",
                "cpu_percent": workload,
            }
        )
    if unknown:
        processes.append(
            {"pid": 5, "ppid": 1, "command": "/tmp/x", "kind": "unknown", "cpu_percent": unknown}
        )
    for process in processes:
        process["observation_ns"] = end
        process["classification"] = {"kind": process["kind"]}
        if process["kind"] == "macos_service":
            process["classification"]["reason"] = "system_executable"
        if process["kind"] == "desktop_app":
            process["classification"].update(
                {
                    key: process[key]
                    for key in ("reason", "entry", "application", "role", "identity")
                }
            )
    return {
        "start_ns": start,
        "end_ns": end,
        "background_cpu_percent": service + renderer + app,
        "macos_service_cpu_percent": service,
        "desktop_renderer_cpu_percent": renderer,
        "desktop_app_cpu_percent": app,
        "workload_cpu_percent": workload,
        "unknown_cpu_percent": unknown,
        "coverage_complete": True,
        "process_churn": [],
        "processes": processes,
    }


def test_desktop_identity_is_nested_as_exported_by_collector():
    sample = _desktop_sample(0, 1_000_000_000, app=1)
    process = sample["processes"][0]
    for key in ("entry", "application", "role", "reason", "identity"):
        process.pop(key)
    host = _desktop_host([sample])
    host["observations"] = [{"classified_processes": [process], "probe_exempt_pids": []}]
    validate_desktop_host(host)
    overall = desktop_overall(host)
    report = summarize_background(_result([_row(0, 0, 100)]), host)
    assert overall["applications"][0]["entry"] == "outlook-main"
    assert report["process_observations"][0]["desktop_identity"] == process["classification"]


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


def test_desktop_policy_reconciles_one_shared_pool_and_role_evidence():
    host = _desktop_host([_desktop_sample(0, 100, service=10, renderer=15, app=25)])
    report = summarize_background(_result([_row(0, 0, 100)]), host)
    tick = report["ticks"][0]
    assert report["load_policy"] == DESKTOP_POLICY
    assert tick["service_cpu_percent_max"] == 10
    assert tick["desktop_renderer_cpu_percent_max"] == 15
    assert tick["desktop_app_cpu_percent_max"] == 25
    assert tick["background_cpu_percent_max"] == 50
    assert not tick["low_background"]
    observed = report["process_observations"]
    assert observed[-1]["desktop_identity"]["role"] == "mail-ui"
    host["cpu_samples"][0]["desktop_app_cpu_percent"] = 24
    with pytest.raises(ValueError, match="desktop CPU totals"):
        summarize_background(_result([_row(0, 0, 100)]), host)


def test_desktop_policy_rejects_missing_registry_or_app_identity_evidence():
    host = _desktop_host([_desktop_sample(0, 100, app=10)])
    del host["desktop_registry_sha256"]
    with pytest.raises(ValueError, match="registry"):
        summarize_background(_result([_row(0, 0, 100)]), host)
    host = _desktop_host([_desktop_sample(0, 100, app=10)])
    del host["cpu_samples"][0]["processes"][0]["identity"]
    del host["cpu_samples"][0]["processes"][0]["classification"]["identity"]
    with pytest.raises(ValueError, match="identity"):
        summarize_background(_result([_row(0, 0, 100)]), host)


def test_desktop_overall_strict_shared_pool_boundaries_and_all_samples():
    # Exactly 150% never starts the strict >150% clock.  Sixty 1s high
    # intervals also remain below the strict >60s duration boundary.
    samples = [
        _desktop_sample(tick * 1_000_000_000, (tick + 1) * 1_000_000_000, app=151)
        for tick in range(60)
    ]
    samples.append(_desktop_sample(60_000_000_000, 61_000_000_000, app=151, workload=19))
    with pytest.raises(ValueError, match="shared background"):
        desktop_overall(_desktop_host(samples))
    boundary = desktop_overall(
        _desktop_host(
            [
                _desktop_sample(tick * 1_000_000_000, (tick + 1) * 1_000_000_000, app=150)
                for tick in range(60)
            ]
        )
    )
    assert not boundary["background_pool"]["shared_background_pool"]["exceeded"]


def test_desktop_guard_keeps_per_process_rule_without_aggregate_workload_abort():
    host = _desktop_host([_desktop_sample(0, 100, workload=19, unknown=19)])
    second = dict(host["cpu_samples"][0]["processes"][-2])
    second["pid"] = 40
    host["cpu_samples"][0]["processes"].append(second)
    host["cpu_samples"][0]["workload_cpu_percent"] = 38
    # No new aggregate ordinary-work guard exists: two workloads at 19% each
    # remain admissible, while the low subset is excluded by their total.
    report = summarize_background(_result([_row(0, 0, 100)]), host)
    assert not report["ticks"][0]["low_background"]
    assert desktop_overall(host)["sample_count"] == 1


def test_desktop_point_replay_rejects_false_complete_unknown_and_relabelled_service():
    host = _desktop_host([_desktop_sample(0, 100, service=1)])
    unknown = {
        "pid": 90,
        "ppid": 1,
        "uid": 501,
        "birth": "point",
        "observation_ns": 0,
        "command": "/tmp/unknown",
        "kind": "unknown",
        "cpu_percent": 20,
        "classification": {"kind": "unknown"},
    }
    host["observations"] = [
        {"process_screen": {"probe_exempt_pids": [], "classified_processes": [unknown]}}
    ]
    with pytest.raises(ValueError, match=">=20%.*point"):
        validate_desktop_host(host)
    unknown["cpu_percent"] = 1
    unknown.update(
        {
            "kind": "desktop_renderer",
            "classification": {"kind": "desktop_renderer"},
        }
    )
    with pytest.raises(ValueError, match="classification replay"):
        validate_desktop_host(host)
    unknown.update(
        {
            "command": "/System/Library/Frameworks/X.framework/service",
            "kind": "unknown",
            "classification": {"kind": "unknown"},
        }
    )
    with pytest.raises(ValueError, match="classification replay"):
        validate_desktop_host(host)
    unknown.update(
        {
            "command": "/Applications/Fake.app/Contents/Frameworks/Fake Helper (Renderer)",
            "kind": "desktop_renderer",
            "classification": {"kind": "desktop_renderer"},
        }
    )
    host["desktop_registry_sha256"] = "b" * 64
    host["desktop_registry_evidence"]["registry_sha256"] = "b" * 64
    # Changing the synthetic digest leaves the fixture scope, where a complete
    # exported history is required before any classification can be trusted.
    with pytest.raises(ValueError):
        validate_desktop_host(host)


def test_scoped_application_reporting_counts_only_cell_overlaps():
    host = _desktop_host(
        [
            _desktop_sample(0, 100, app=10),
            _desktop_sample(100, 200, app=11),
            _desktop_sample(200, 300, app=12),
        ]
    )
    report = summarize_background(_result([_row(0, 100, 200)]), host)
    assert [row["sample_index"] for row in report["process_observations"]] == [1]
    assert report["background_pool"]["cpu_percent_max"]["desktop_app"] == 11
    assert report["applications"][0]["affected_tick_count"] == 1


def test_enrolled_gui_named_claude_is_not_relabelled_as_cli_workload():
    from research.bl_served_search.background_report import _replay_kind

    row = {
        "pid": 3,
        "birth": "fixture-birth",
        "uid": 501,
        "command": "/Applications/Claude.app/Contents/MacOS/Claude",
        "kind": "desktop_app",
        "classification": {"kind": "desktop_app"},
    }
    assert _replay_kind(row, {"registry_sha256": "a" * 64})["kind"] == "desktop_app"
    row["known_workload"] = True
    assert _replay_kind(row, {"registry_sha256": "a" * 64})["kind"] == "agent_workload"


@pytest.mark.parametrize(
    "classification",
    [
        {"kind": "agent_workload", "reason": "known_workload"},
        {"kind": "unknown", "classifier_decision": {"identity_binding_changed": True}},
    ],
)
def test_interval_guard_rejects_low_cpu_prohibited_identity_evidence(classification):
    host = _desktop_host(
        [
            _desktop_sample(
                0,
                100,
                workload=1 if classification["kind"] == "agent_workload" else 0,
                unknown=1 if classification["kind"] == "unknown" else 0,
            )
        ]
    )
    process = next(
        row for row in host["cpu_samples"][0]["processes"] if row["kind"] == classification["kind"]
    )
    process["classification"] = classification
    host["observations"] = []
    with pytest.raises(ValueError, match="prohibited|identity binding"):
        validate_desktop_host(host)


def test_replay_exact_system_path_workload_exclusion_precedes_service_class():
    from research.bl_served_search.background_report import _replay_kind

    command = "/usr/libexec/computeworker"
    decision = _replay_kind(
        {"command": command},
        {
            "registry_sha256": "a" * 64,
            "registry": {"workload_exclusions": [{"exact_executable": command}]},
        },
    )
    assert decision == {"kind": "agent_workload", "reason": "workload_exclusion"}


def test_exact_workload_exclusion_precedes_system_root_in_replay():
    from research.bl_served_search.background_report import _replay_kind

    row = {"command": "/usr/libexec/computeworker"}
    evidence = {
        "registry_sha256": "a" * 64,
        "registry": {"workload_exclusions": [{"exact_executable": row["command"]}]},
    }
    assert _replay_kind(row, evidence)["kind"] == "agent_workload"


def test_desktop_host_cannot_omit_all_historical_point_evidence():
    host = _desktop_host([_desktop_sample(0, 100, service=1)])
    host["observations"] = []
    with pytest.raises(ValueError, match="point observations missing"):
        validate_desktop_host(host)


@pytest.mark.parametrize(
    "classification",
    [
        {"reason": "known_workload"},
        {"identity_binding_changed": True},
    ],
)
def test_interval_replay_rejects_prohibited_activity_or_admitted_identity_drift_below_20(
    classification,
):
    from research.bl_served_search.background_report import _validate_desktop_guard

    with pytest.raises(ValueError, match="prohibited workload|identity binding changed"):
        _validate_desktop_guard(
            [
                {
                    "processes": [
                        {"kind": "unknown", "cpu_percent": 1, "classification": classification}
                    ],
                    "start_ns": 0,
                    "end_ns": 1,
                    "background_cpu_percent": 0,
                }
            ]
        )
