"""Composite source/lifecycle/CPU binding proofs; no live supervisor is touched."""

from __future__ import annotations

from copy import deepcopy

import pytest

from research.bl_served_search.remote_supervision import POLICY, digest, replay
from research.bl_served_search.supervision_identity import process_id, verify_identity
from tests.test_served_supervision_identity import qualified_fixture

SECOND = 1_000_000_000


def fixture():
    registry, journal, observations = qualified_fixture()
    root = registry["roots"][0]
    rid = process_id(root)
    hid = process_id(journal[2]["identity"])
    journal[3].update(final_cpu_ns=SECOND // 10, final_scope="self", descendants_complete=True)
    verified = verify_identity(registry, journal, observations)
    assert verified["valid"], verified["errors"]
    cpu = {
        "schema": "served-supervision-cpu/v2",
        "policy": POLICY,
        "counter_unit": "nanoseconds",
        "counter_resolution_ns": 1,
        "counter_scope": "exclusive",
        "provenance_complete": True,
        "exposure_start_ns": 60 * SECOND,
        "roots": [rid],
        "processes": [
            {"id": rid, "root_id": rid, "parent_id": None, "operation_id": None, "exit_ns": None},
            {
                "id": hid,
                "root_id": rid,
                "parent_id": rid,
                "operation_id": "poll-1",
                "launch_ns": 2 * SECOND,
                "exit_ns": 3 * SECOND,
                "final_cpu_ns": SECOND // 10,
                "final_scope": "self",
                "descendants_complete": True,
            },
        ],
        "operations": [{"id": "poll-1", "root_id": rid, "start_ns": SECOND, "end_ns": 4 * SECOND}],
        "samples": [
            {
                "time_ns": t * SECOND,
                "capture_start_ns": t * SECOND,
                "capture_end_ns": t * SECOND,
                "cpu_ns": {rid: 0, **({hid: 0} if t == 2 else {})},
            }
            for t in range(66)
        ],
    }
    points = [
        {
            "time_ns": t * SECOND,
            "processes": [
                {
                    "pid": root["pid"],
                    "ppid": 1,
                    "birth_ns": root["birth_ns"],
                    "command": root["interpreter"],
                    "cpu_percent": 99.0,
                }
            ],
        }
        for t in range(0, 66, 5)
    ]
    intervals = [
        {
            "start_ns": t * SECOND,
            "end_ns": (t + 1) * SECOND,
            "processes": [],
            "coverage_complete": True,
            "process_churn": [],
        }
        for t in range(65)
    ]
    for sample in intervals:
        for proc, amount, parent in (
            (root, 0, 1),
            (journal[2]["identity"], SECOND // 10, root["pid"]),
        ):
            if proc is not root and sample["start_ns"] != 2 * SECOND:
                continue
            sample["processes"].append(
                {
                    "pid": proc["pid"],
                    "ppid": parent,
                    "birth_ns": proc["birth_ns"],
                    "command": proc["interpreter"],
                    "cpu_percent": amount * 100 / SECOND,
                    "cpu_delta_numerator": amount,
                    "counter_denominator": 1,
                }
            )
    return registry, journal, cpu, observations, points, intervals


def run(parts):
    r, j, c, o, p, i = parts
    return replay(r, j, c, observations=o, point_screens=p, ordinary_intervals=i)


def test_matching_full_stream_is_offline_valid_but_never_live_admission():
    result = run(fixture())
    assert result["valid"], result["errors"]
    assert result["status"] == "OFFLINE_VALID"
    assert result["live_capture_capability"] == "NOT_ADMISSIBLE"
    assert result["observed_cpu_s"] == 0.1
    assert len(result["ordinary_cpu_samples"]) == 65
    assert result["maxima"]["conservative_burst_pct"] == 10


@pytest.mark.parametrize(
    "mutation",
    ["process", "parent", "operation", "launch", "exit", "final", "root", "span", "coverage"],
)
def test_individually_plausible_but_unbound_cpu_inventory_refused(mutation):
    parts = fixture()
    cpu = parts[2]
    if mutation == "process":
        cpu["processes"][1]["id"] = "foreign"
    elif mutation == "parent":
        cpu["processes"][1]["parent_id"] = "foreign"
    elif mutation == "operation":
        cpu["operations"][0]["id"] = "foreign"
    elif mutation == "launch":
        cpu["processes"][1]["launch_ns"] += 1
    elif mutation == "exit":
        cpu["processes"][1]["exit_ns"] += 1
    elif mutation == "final":
        cpu["processes"][1]["final_cpu_ns"] += 1
    elif mutation == "root":
        cpu["roots"] = ["foreign"]
    elif mutation == "span":
        cpu["operations"][0]["end_ns"] += 1
    else:
        parts[0]["capture_end_ns"] += 1
    assert not run(parts)["valid"]


@pytest.mark.parametrize("kind", ["python", "pytest", "/bin/bash", "/tmp/helper"])
@pytest.mark.parametrize("cpu,valid", [(19.999, True), (20.0, False), (20.001, False)])
def test_ordinary_point_twenty_percent_boundary(kind, cpu, valid):
    parts = fixture()
    parts[4][0]["processes"].append(
        {"pid": 999, "ppid": 1, "birth_ns": 3, "command": kind, "cpu_percent": cpu}
    )
    assert run(parts)["valid"] is valid


@pytest.mark.parametrize("cpu,valid", [(19.999, True), (20.0, False), (20.001, False)])
def test_ordinary_interval_twenty_percent_boundary(cpu, valid):
    parts = fixture()
    parts[5][1]["processes"].append(
        {"pid": 999, "ppid": 1, "birth_ns": 3, "command": "python", "cpu_percent": cpu}
    )
    assert run(parts)["valid"] is valid


@pytest.mark.parametrize("change", ["birth", "command"])
def test_same_pid_reuse_or_changed_executable_never_point_exempt(change):
    parts = fixture()
    row = parts[4][0]["processes"][0]
    if change == "birth":
        row["birth_ns"] += 1
    else:
        row["command"] = "pytest"
    assert not run(parts)["valid"]


def test_raw_screens_are_preserved_and_verified_cpu_has_its_own_ledger():
    parts = fixture()
    saved = deepcopy(parts)
    result = run(parts)
    assert parts == saved
    assert result["valid"]
    assert result["identity"]["verified_operations"]["poll-1"]["response_bytes"] == 2
    assert digest(parts[0]) == result["registry_digest"]


def test_source_policy_assignment_or_lost_observation_invalidates_whole_replay():
    for key in ("policy_source_hash", "assignment", "generation", "source_hashes"):
        parts = fixture()
        row = parts[3][-1]
        if key == "policy_source_hash":
            row[key] = "0" * 64
        else:
            row["identity"][key] = "mismatch"
        assert not run(parts)["valid"]


def test_host_and_cpu_streams_must_cover_the_same_intervals():
    parts = fixture()
    parts[5].pop()
    assert not run(parts)["valid"]


def test_final_replay_requires_the_full_host_only_admission():
    parts = fixture()
    parts[2]["samples"] = parts[2]["samples"][:6]
    assert not run(parts)["valid"]


@pytest.mark.parametrize("mutation", [None, "amount", "percent", "rounded", "denominator"])
def test_independent_supervision_interval_counter_must_reconcile(mutation):
    parts = fixture()
    helper = parts[1][2]["identity"]
    row = {
        "pid": helper["pid"],
        "ppid": 101,
        "birth_ns": helper["birth_ns"],
        "command": helper["interpreter"],
        "cpu_percent": 10.0,
        "cpu_delta_numerator": SECOND // 10,
        "counter_denominator": 1,
    }
    if mutation == "amount":
        row["cpu_delta_numerator"] += 1
    elif mutation == "percent":
        row["cpu_percent"] = 100.0
    elif mutation == "rounded":
        row.pop("cpu_delta_numerator")
    elif mutation == "denominator":
        row["counter_denominator"] = 3
    parts[5][2]["processes"][1] = row
    result = run(parts)
    assert result["valid"] is (mutation is None)
    if mutation is None:
        assert (
            next(
                r for r in result["sampler_reconciliation"] if r["process_id"] == process_id(helper)
            )["ledger_delta_numerator"]
            == SECOND // 10
        )
        assert not result["ordinary_cpu_samples"][2]["processes"]


def test_actual_ledger_projection_and_report_keep_poll_ticks_and_root_bookkeeping():
    from research.bl_served_search.supervision_report import summarize_supervision

    parts = fixture()
    root = parts[0]["roots"][0]
    rid = process_id(root)
    for index, sample in enumerate(parts[2]["samples"]):
        sample["cpu_ns"][rid] = index * SECOND // 100
    for sample in parts[5]:
        sample["processes"][0].update(cpu_percent=1.0, cpu_delta_numerator=SECOND // 100)
    replayed = run(parts)
    assert replayed["valid"], replayed["errors"]
    rows = [
        {
            "tick": index,
            "target_ns": at,
            "wake_ns": at,
            "start_ns": at,
            "end_ns": at + 1_000_000,
            "start_cpu_ns": 0,
            "end_cpu_ns": 100,
            "spans": {"game_update": {"wall_ns": 100}},
        }
        for index, at in enumerate((2 * SECOND + SECOND // 2, 61 * SECOND))
    ]
    report = summarize_supervision(
        {"trace": rows, "warmup_ticks": 0},
        {"load_policy": POLICY, "cpu_samples": parts[5]},
        replayed,
    )
    assert report["timing"]["overall"]["count"] == 2
    assert report["timing"]["low_background"]["count"] == 1
    assert report["counts"]["poll_overlap"] == 1
    assert not report["ticks"][0]["low_background"]
    assert report["ticks"][1]["low_background"]


@pytest.mark.parametrize(
    "mutation", ["omit_root", "omit_helper", "duplicate_root", "duplicate_helper"]
)
def test_every_permitted_interval_process_must_reconcile_exactly_once(mutation):
    parts = fixture()
    rows = parts[5][2]["processes"]
    index = 0 if mutation.endswith("root") else 1
    if mutation.startswith("omit"):
        rows.pop(index)
    else:
        rows.append(deepcopy(rows[index]))
    result = run(parts)
    assert not result["valid"]
    assert any("permitted process" in error for error in result["errors"])


@pytest.mark.parametrize("bad", [None, [], 3])
def test_nondict_process_and_interval_rows_return_invalid(bad):
    for target in ("point", "interval", "process"):
        parts = fixture()
        if target == "point":
            parts[4][0]["processes"].append(bad)
        elif target == "interval":
            parts[5][0] = bad
        else:
            parts[5][0]["processes"].append(bad)
        assert not run(parts)["valid"]
