"""Offline exact-boundary and lifecycle tests for the prospective CPU ledger."""

from __future__ import annotations

from copy import deepcopy

import pytest

from research.bl_served_search.supervision_cpu import POLICY, SCHEMA, SECOND, replay_cpu


def evidence(seconds: int = 65, roots: tuple[str, ...] = ("root",)) -> dict:
    return {
        "schema": SCHEMA,
        "policy": POLICY,
        "counter_unit": "nanoseconds",
        "counter_scope": "exclusive",
        "counter_resolution_ns": 1,
        "provenance_complete": True,
        "exposure_start_ns": 60 * SECOND,
        "roots": list(roots),
        "processes": [
            {"id": rid, "root_id": rid, "parent_id": None, "operation_id": None, "exit_ns": None}
            for rid in roots
        ],
        "operations": [],
        "samples": [
            {
                "time_ns": t * SECOND,
                "capture_start_ns": t * SECOND,
                "capture_end_ns": t * SECOND,
                "cpu_ns": dict.fromkeys(roots, 0),
            }
            for t in range(seconds + 1)
        ],
    }


def helper(
    e: dict,
    *,
    pid: str = "helper",
    oid: str = "poll",
    rid: str = "root",
    parent: str | None = None,
    launch: int = SECOND // 10,
    exit: int | None = 9 * SECOND // 10,
    cpu: int | None = SECOND // 10,
    scope: str = "self",
    sampled: dict[int, int] | None = None,
) -> None:
    if not any(op["id"] == oid for op in e["operations"]):
        e["operations"].append({"id": oid, "root_id": rid, "start_ns": launch, "end_ns": exit})
    e["processes"].append(
        {
            "id": pid,
            "root_id": rid,
            "parent_id": parent or rid,
            "operation_id": oid,
            "launch_ns": launch,
            "exit_ns": exit,
            "final_cpu_ns": cpu,
            "final_scope": scope,
            "descendants_complete": True,
        }
    )
    for sample in e["samples"]:
        t = sample["time_ns"]
        if launch <= t and (exit is None or t < exit):
            sample["cpu_ns"][pid] = (sampled or {}).get(t, 0)

    # Synthetic inclusion witnesses model observed wait/reap propagation.
    for row in e["processes"]:
        if row.get("final_scope") != "subtree":
            continue
        descendants = []
        pending = [row["id"]]
        while pending:
            owner = pending.pop()
            for child in e["processes"]:
                if child.get("parent_id") == owner:
                    pending.append(child["id"])
                    descendants.append(
                        {
                            "process_id": child["id"],
                            "parent_id": owner,
                            "reaped_ns": child["exit_ns"],
                        }
                    )
        row["final_inclusions"] = descendants


def root_increments(e: dict, increments: dict[int, int], rid: str = "root") -> None:
    total = 100 * SECOND  # Lifetime baseline is neither charged nor reset later.
    for row in e["samples"]:
        total += increments.get(row["time_ns"], 0)
        row["cpu_ns"][rid] = total


def error(result: dict, text: str) -> bool:
    return any(text in value for value in result["errors"])


@pytest.mark.parametrize("amount", [SECOND // 5 - 1, SECOND // 5, SECOND // 5 + 1])
def test_verified_supervision_does_not_apply_ordinary_twenty_percent(amount: int) -> None:
    e = evidence()
    helper(e, cpu=amount)
    assert replay_cpu(e)["valid"]


@pytest.mark.parametrize(
    "amount,valid", [(SECOND // 2 - 1, True), (SECOND // 2, True), (SECOND // 2 + 1, False)]
)
def test_exact_aggregate_fifty_percent_boundary(amount: int, valid: bool) -> None:
    e = evidence()
    helper(e, cpu=amount)
    r = replay_cpu(e)
    assert r["valid"] is valid
    assert error(r, ">50%") is not valid


@pytest.mark.parametrize("amount,valid", [(SECOND - 1, True), (SECOND, True), (SECOND + 1, False)])
def test_exact_operation_cpu_boundary_with_sampled_live_enforcement(
    amount: int, valid: bool
) -> None:
    e = evidence()
    per_tick = amount // 4
    helper(
        e,
        launch=0,
        exit=4 * SECOND,
        cpu=amount,
        sampled={t * SECOND: t * per_tick for t in range(4)},
    )
    r = replay_cpu(e)
    assert r["valid"] is valid
    assert error(r, "operation poll >1.0") is not valid
    assert r["observed_cpu_s"] == amount / SECOND


@pytest.mark.parametrize("delta,valid", [(-1, True), (0, True), (1, False)])
def test_exact_two_cpu_seconds_rolling_ten(delta: int, valid: bool) -> None:
    e = evidence()
    root_increments(e, {t * SECOND: SECOND // 4 + (delta if t == 8 else 0) for t in range(1, 9)})
    r = replay_cpu(e)
    assert r["valid"] is valid
    assert r["maxima"]["conservative_10s_cpu_s"] == (2 * SECOND + delta) / SECOND
    assert error(r, "rolling 10s") is not valid


@pytest.mark.parametrize("delta,valid", [(-1, True), (0, True), (1, False)])
def test_exact_three_cpu_seconds_rolling_sixty(delta: int, valid: bool) -> None:
    e = evidence()
    root_increments(e, {t * SECOND: SECOND // 20 + (delta if t == 60 else 0) for t in range(1, 61)})
    r = replay_cpu(e)
    assert r["valid"] is valid
    assert r["maxima"]["conservative_60s_cpu_s"] == (3 * SECOND + delta) / SECOND
    assert error(r, "rolling 60s") is not valid


def test_concurrent_lanes_and_root_bookkeeping_are_aggregated() -> None:
    e = evidence(roots=("a", "b"))
    helper(e, pid="heartbeat", oid="beat", rid="a", cpu=SECOND // 5)
    helper(e, pid="telemetry", oid="read", rid="b", cpu=SECOND // 5)
    root_increments(e, {SECOND: SECOND // 10}, rid="a")
    r = replay_cpu(e)
    assert r["valid"]
    assert r["observed_cpu_s"] == 0.5
    assert r["roots"]["a"]["observed_cpu_s"] == 0.3
    assert r["roots"]["b"]["observed_cpu_s"] == 0.2
    assert r["maxima"]["conservative_burst_pct"] == 50
    root_increments(e, {SECOND: SECOND // 10 + 1}, rid="a")
    assert error(replay_cpu(e), ">50%")


def test_ongoing_operation_cannot_wait_for_exit_to_trigger() -> None:
    e = evidence(10)
    helper(
        e, launch=0, exit=None, cpu=None, sampled={t * SECOND: t * 110_000_000 for t in range(11)}
    )
    r = replay_cpu(e, final=False)
    assert error(r, "operation poll >1.0")
    assert r["operations"]["poll"]["complete"] is False
    assert not replay_cpu(e)["valid"]


def test_between_samples_child_final_is_charged_without_exit_sample() -> None:
    e = evidence()
    helper(e, cpu=SECOND // 4)
    r = replay_cpu(e)
    assert r["valid"]
    assert len(r["charges"]) == 1
    assert r["charges"][0]["kind"] == "final"
    assert r["observed_cpu_s"] == 0.25


def test_final_residual_not_full_final_is_added_after_samples() -> None:
    e = evidence()
    helper(e, launch=0, exit=2 * SECOND, cpu=300_000_000, sampled={0: 0, SECOND: 100_000_000})
    r = replay_cpu(e)
    assert r["valid"]
    assert [c["cpu_ns"] for c in r["charges"]] == [100_000_000, 200_000_000]
    assert r["observed_cpu_s"] == 0.3


def test_recursive_inclusive_finals_remove_descendants_exactly_once() -> None:
    e = evidence()
    helper(
        e,
        pid="a",
        launch=0,
        exit=3 * SECOND,
        cpu=600_000_000,
        scope="subtree",
        sampled={SECOND: 50_000_000, 2 * SECOND: 100_000_000},
    )
    helper(
        e,
        pid="b",
        parent="a",
        launch=SECOND // 10,
        exit=2 * SECOND,
        cpu=400_000_000,
        scope="subtree",
        sampled={SECOND: 100_000_000},
    )
    helper(e, pid="c", parent="b", launch=SECOND // 5, exit=9 * SECOND // 10, cpu=100_000_000)
    r = replay_cpu(e)
    assert r["valid"]
    assert r["observed_cpu_s"] == 0.6
    by_process = {
        pid: sum(c["cpu_ns"] for c in r["charges"] if c["process_id"] == pid)
        for pid in ("a", "b", "c")
    }
    assert by_process == {"a": 200_000_000, "b": 300_000_000, "c": 100_000_000}


def test_inclusive_final_must_cover_sampled_self_plus_all_descendants() -> None:
    e = evidence()
    helper(
        e,
        pid="a",
        launch=0,
        exit=2 * SECOND,
        cpu=200_000_000,
        scope="subtree",
        sampled={SECOND: 150_000_000},
    )
    helper(e, pid="b", parent="a", launch=SECOND // 10, exit=9 * SECOND // 10, cpu=100_000_000)
    assert error(replay_cpu(e), "regression")
    e["processes"][1]["final_cpu_ns"] = 50_000_000
    assert error(replay_cpu(e), "smaller than descendant")


def test_uncertain_span_full_charge_is_duplicated_only_in_bounds() -> None:
    e = evidence()
    helper(
        e,
        launch=SECOND // 2,
        exit=5 * SECOND // 2,
        cpu=300_000_000,
        sampled={SECOND: 100_000_000, 2 * SECOND: 200_000_000},
    )
    e["samples"][1]["capture_start_ns"] = 3 * SECOND // 4
    e["samples"][2]["capture_start_ns"] = 7 * SECOND // 4
    r = replay_cpu(e)
    assert r["valid"]
    assert r["observed_cpu_s"] == 0.3
    assert [i["conservative_cpu_s"] for i in r["intervals"][:3]] == [0.2, 0.2, 0.1]
    assert [i["observed_cpu_s"] for i in r["intervals"][:3]] == [0.1, 0.1, 0.1]


def test_off_grid_rolling_window_and_trailing_support() -> None:
    e = evidence()
    # Both full residuals can fit into one 10s window despite their endpoints
    # being >10s apart. Keep each helper below the individual cap.
    for index in range(3):
        helper(
            e,
            pid=f"left{index}",
            oid=f"left{index}",
            launch=SECOND // 2,
            exit=5 * SECOND // 4,
            cpu=400_000_000,
        )
        helper(
            e,
            pid=f"right{index}",
            oid=f"right{index}",
            launch=10 * SECOND + SECOND // 4,
            exit=11 * SECOND + 3 * SECOND // 4,
            cpu=400_000_000,
        )
    r = replay_cpu(e)
    assert r["maxima"]["observed_10s_cpu_s"] == 1.2
    assert r["maxima"]["conservative_10s_cpu_s"] == 2.4
    assert error(r, "rolling 10s")


def test_repeated_children_churn_does_not_clear_debt_or_cells() -> None:
    e = evidence(90)
    for index in range(16):
        helper(
            e,
            pid=f"child{index}",
            oid=f"poll{index}",
            launch=(55 + index * 2) * SECOND + 1,
            exit=(55 + index * 2) * SECOND + SECOND // 2,
            cpu=200_000_000,
        )
    e["cells"] = [
        {"start_ns": 60 * SECOND, "end_ns": 70 * SECOND},
        {"start_ns": 80 * SECOND, "end_ns": 90 * SECOND},
    ]
    r = replay_cpu(e)
    assert error(r, "rolling 60s")
    assert r["observed_cpu_s"] == 3.2


def test_root_cpu_counts_inside_and_between_operations() -> None:
    e = evidence()
    helper(
        e,
        launch=SECOND // 2,
        exit=5 * SECOND // 2,
        cpu=900_000_000,
        sampled={SECOND: 0, 2 * SECOND: 0},
    )
    root_increments(e, {SECOND: 100_000_000, 2 * SECOND: 100_000_000, 20 * SECOND: 100_000_000})
    r = replay_cpu(e)
    assert r["observed_cpu_s"] == 1.2
    assert r["operations"]["poll"]["conservative_cpu_s"] == 1.1
    assert error(r, "operation poll >1.0")


@pytest.mark.parametrize("bad", [-1, True, 1.0, float("nan"), float("inf")])
def test_nonexact_negative_nonfinite_boolean_cpu_refused(bad: object) -> None:
    e = evidence()
    helper(e, cpu=bad)
    assert not replay_cpu(e)["valid"]
    e = evidence()
    e["samples"][1]["cpu_ns"]["root"] = bad
    assert not replay_cpu(e)["valid"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("counter_resolution_ns", 10_000_000),
        ("counter_resolution_ns", True),
        ("counter_scope", "subtree"),
        ("provenance_complete", False),
        ("policy", "classified-v1"),
        ("schema", "old"),
    ],
)
def test_ambiguous_counter_policy_or_journal_provenance_refused(field: str, value: object) -> None:
    e = evidence()
    e[field] = value
    assert not replay_cpu(e)["valid"]


def test_sample_and_final_regression_refused() -> None:
    e = evidence()
    root_increments(e, {SECOND: 100, 2 * SECOND: -1})
    assert error(replay_cpu(e), "regression")
    e = evidence()
    helper(e, launch=0, exit=2 * SECOND, cpu=99, sampled={SECOND: 100})
    assert error(replay_cpu(e), "regression")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_root",
        "missing_helper",
        "unknown_helper",
        "missing_exit",
        "gap",
        "short_tail",
        "restart",
        "missing_prior",
    ],
)
def test_missing_coverage_never_becomes_zero_or_reset(mutation: str) -> None:
    e = evidence()
    helper(e, launch=0, exit=2 * SECOND, cpu=100, sampled={SECOND: 50})
    if mutation == "missing_root":
        del e["samples"][1]["cpu_ns"]["root"]
    elif mutation == "missing_helper":
        del e["samples"][1]["cpu_ns"]["helper"]
    elif mutation == "unknown_helper":
        e["samples"][1]["cpu_ns"]["unobserved"] = 0
    elif mutation == "missing_exit":
        e["processes"][1]["exit_ns"] = None
    elif mutation == "gap":
        e["samples"] = e["samples"][:1] + e["samples"][6:]
    elif mutation == "short_tail":
        e["samples"][-1]["time_ns"] = e["samples"][-2]["time_ns"] + SECOND // 2 - 1
    elif mutation == "restart":
        e["processes"][0]["exit_ns"] = SECOND
    else:
        e["processes"][1]["launch_ns"] = -1
    assert not replay_cpu(e)["valid"]


def test_admission_and_online_prefix_complete_final_are_distinct() -> None:
    e = evidence(65)
    helper(e, launch=61 * SECOND, exit=None, cpu=None)
    r = replay_cpu(e, final=False)
    assert r["valid"] and r["admission_ready"] and not r["complete"]
    assert error(replay_cpu(e), "missing operation end")
    e["exposure_start_ns"] = 60 * SECOND - 1
    assert not replay_cpu(e, final=False)["valid"]


def test_replay_is_deterministic_and_does_not_mutate_input() -> None:
    e = evidence()
    helper(e)
    before = deepcopy(e)
    assert replay_cpu(e) == replay_cpu(e)
    assert e == before


def test_exact_sampling_interval_boundaries() -> None:
    for step in (SECOND // 2, 5 * SECOND):
        e = evidence(2)
        for i, row in enumerate(e["samples"]):
            row["time_ns"] = row["capture_start_ns"] = row["capture_end_ns"] = i * step
        assert replay_cpu(e)["valid"]


@pytest.mark.parametrize("denominator", [3, 125])
@pytest.mark.parametrize("numerator_delta,valid", [(-1, True), (0, True), (1, False)])
def test_rational_clock_preserves_exact_fifty_percent_boundary(
    denominator: int, numerator_delta: int, valid: bool
) -> None:
    e = evidence()
    e.update(counter_unit="rational_nanoseconds", counter_denominator=denominator)
    helper(e, cpu=SECOND * denominator // 2 + numerator_delta)
    r = replay_cpu(e)
    assert r["valid"] is valid
    assert r["charges"][0]["counter_denominator"] == denominator
    assert r["charges"][0]["cpu_numerator"] == SECOND * denominator // 2 + numerator_delta


@pytest.mark.parametrize("ceiling,width", [(2, 10), (3, 60)])
@pytest.mark.parametrize("delta,valid", [(-1, True), (0, True), (1, False)])
def test_rational_rolling_and_operation_thresholds_are_exact(
    ceiling: int, width: int, delta: int, valid: bool
) -> None:
    e = evidence()
    e.update(counter_unit="rational_nanoseconds", counter_denominator=3)
    count = 8 if width == 10 else 60
    each = ceiling * SECOND * 3 // count
    root_increments(
        e, {t * SECOND: each + (delta if t == count else 0) for t in range(1, count + 1)}
    )
    r = replay_cpu(e)
    assert r["valid"] is valid
    assert error(r, f"rolling {width}s") is not valid
    e = evidence()
    e.update(counter_unit="rational_nanoseconds", counter_denominator=3)
    amount = 3 * SECOND + delta
    helper(
        e,
        launch=0,
        exit=4 * SECOND,
        cpu=amount,
        sampled={t * SECOND: t * (amount // 4) for t in range(4)},
    )
    assert error(replay_cpu(e), "operation poll >1.0") is not valid


@pytest.mark.parametrize("denominator", [0, -1, True, 1.5, float("inf")])
def test_invalid_rational_clock_denominator_refused(denominator: object) -> None:
    e = evidence()
    e.update(counter_unit="rational_nanoseconds", counter_denominator=denominator)
    assert not replay_cpu(e)["valid"]


def test_required_capture_brackets_and_uncovered_tail_refused() -> None:
    e = evidence()
    del e["samples"][1]["capture_start_ns"]
    assert not replay_cpu(e)["valid"]
    e = evidence()
    e["end_ns"] = e["samples"][-1]["time_ns"] + SECOND // 4
    assert error(replay_cpu(e), "endpoints")
    e = evidence()
    e["samples"][1]["capture_end_ns"] += 1
    assert error(replay_cpu(e), "completion")


def test_active_before_exposure_is_included_and_before_stream_is_refused() -> None:
    e = evidence()
    helper(
        e,
        launch=59 * SECOND,
        exit=61 * SECOND,
        cpu=100_000_000,
        sampled={59 * SECOND: 0, 60 * SECOND: 50_000_000},
    )
    r = replay_cpu(e)
    assert r["valid"] and r["admission_ready"]
    assert r["operations"]["poll"]["observed_cpu_s"] == 0.1
    e["operations"][0]["start_ns"] = -1
    assert not replay_cpu(e)["valid"]


def test_online_exclusive_parent_exit_may_precede_child_exit() -> None:
    e = evidence()
    helper(e, pid="a", launch=0, exit=2 * SECOND, cpu=100)
    e["operations"][0]["end_ns"] = None
    helper(e, pid="b", parent="a", launch=SECOND, exit=None, cpu=None)
    assert replay_cpu(e, final=False)["valid"]
    e["processes"][1]["final_scope"] = "subtree"
    e["processes"][1]["final_inclusions"] = [
        {"process_id": "b", "parent_id": "a", "reaped_ns": 2 * SECOND}
    ]
    assert error(replay_cpu(e, final=False), "outstanding descendant")


def test_helper_operation_root_must_match_parent_and_process_root() -> None:
    e = evidence(roots=("root", "foreign"))
    helper(e)
    e["operations"][0]["root_id"] = "foreign"
    assert error(replay_cpu(e), "crosses registered roots")


def test_helper_cannot_launch_after_its_observed_parent_exits() -> None:
    e = evidence()
    helper(e, pid="a", launch=0, exit=SECOND, cpu=0)
    e["operations"][0]["end_ns"] = 3 * SECOND
    helper(e, pid="b", parent="a", launch=2 * SECOND, exit=3 * SECOND, cpu=0)
    assert error(replay_cpu(e), "after parent exit")


@pytest.mark.parametrize(
    "mutation", ["missing", "unwaited", "late_reap", "wrong_parent", "foreign"]
)
def test_inclusive_scope_requires_complete_observed_reaping_propagation(mutation: str) -> None:
    e = evidence()
    helper(e, pid="a", launch=0, exit=3 * SECOND, cpu=800_000_000, scope="subtree")
    helper(e, pid="b", parent="a", launch=1, exit=SECOND, cpu=300_000_000)
    parent = e["processes"][1]
    if mutation == "missing":
        del parent["final_inclusions"]
    elif mutation == "unwaited":
        parent["final_inclusions"] = []
    elif mutation == "late_reap":
        parent["final_inclusions"][0]["reaped_ns"] = 4 * SECOND
    elif mutation == "wrong_parent":
        parent["final_inclusions"][0]["parent_id"] = "root"
    else:
        parent["final_inclusions"][0]["process_id"] = "unknown"
    assert not replay_cpu(e)["valid"]


def test_per_root_self_and_helper_totals_and_maxima_remain_separate() -> None:
    e = evidence(roots=("root", "lane2"))
    helper(e, cpu=200_000_000)
    root_increments(e, {SECOND: 100_000_000})
    root_increments(e, {2 * SECOND: 100_000_000}, rid="lane2")
    r = replay_cpu(e)
    assert r["valid"]
    assert r["roots"]["root"]["self_cpu_s"] == 0.1
    assert r["roots"]["root"]["helper_cpu_s"] == 0.2
    assert r["roots"]["root"]["maxima"]["conservative_burst_pct"] == 30
    assert r["roots"]["lane2"]["maxima"]["conservative_10s_cpu_s"] == 0.1
    assert r["maxima"]["conservative_10s_cpu_s"] == 0.4


@pytest.mark.parametrize("case", ["born_during_sweep", "exit_during_sweep", "exit_at_start"])
def test_capture_bracket_birth_exit_races_keep_all_final_cpu(case) -> None:
    e = evidence()
    if case == "born_during_sweep":
        helper(e, launch=SECOND - 25_000_000, exit=SECOND + SECOND // 2, cpu=SECOND // 10)
        e["samples"][1]["capture_start_ns"] = SECOND - 50_000_000
        e["samples"][1]["cpu_ns"].pop("helper")
    elif case == "exit_during_sweep":
        helper(e, launch=SECOND // 2, exit=SECOND - 25_000_000, cpu=SECOND // 10)
        e["samples"][1]["capture_start_ns"] = SECOND - 50_000_000
        e["samples"][1]["cpu_ns"]["helper"] = SECOND // 20
    else:
        helper(e, launch=SECOND // 2, exit=SECOND, cpu=SECOND // 10)
        e["samples"][1]["cpu_ns"]["helper"] = SECOND // 20
    result = replay_cpu(e)
    assert result["valid"], result["errors"]
    assert result["observed_cpu_s"] == 0.1
    assert sum(c["cpu_ns"] for c in result["charges"]) == SECOND // 10
