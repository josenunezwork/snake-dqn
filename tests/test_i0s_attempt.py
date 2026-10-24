"""Adversarial offline tests for the I0S adapter and admission; zero provider calls."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests import i0s_synthetic_tree as tree
from tools import i0s_attempt as i0s
from tools import small_bench_register as register

pytestmark = pytest.mark.skipif(
    not (i0s.RULING.is_file() and i0s.ROUND36.is_file() and i0s.BUDGET_RULING.is_file()),
    reason="I0S ruling or round-36 judgment absent on this machine",
)


def template() -> dict:
    return {
        "admission_reason": "TEMPLATE_ONLY",
        "commit": i0s.PIN,
        "transport_pin": i0s.TRANSPORT_PIN,
        "job_id": "gpu-integration-i0-main-968f02765f7c-01",
        "scientific_admission": False,
        "canary": {"required": "same-pin native I0 canary receipt"},
        "env": dict(i0s.FLAGS, PYTHONUNBUFFERED="1"),
        "inputs": [
            {"dest": "public-inputs.json", "sha256": i0s.PUBLIC_INPUTS_SHA},
            {"dest": "metadata.tar", "sha256": i0s.METADATA_SHA},
        ],
        "requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "est_minutes": 50},
        "timeout_minutes": 120,
        "budget": {"unit_cap": 20.0, "rate_units_per_hour": 10.0, "reserve_units": 10 / 3},
        "output_dir": "/old/main-output",
        "lease_seconds": 600,
    }


CANARY = {"path": str(i0s.RECEIPT), "sha256": i0s.RECEIPT_SHA, "analysis": {"verdict": "x"}}


def manifest(attempt: int = 1) -> dict:
    output = i0s.attempt_dir(attempt) / "main-output"
    return i0s.build_manifest(template(), copy.deepcopy(CANARY), attempt, output)


# ----------------------------------------------------------------------- manifest


def test_manifest_changes_only_ruling_fields() -> None:
    value = manifest()
    assert i0s.outside_ruling(template(), value) == set()
    top = {path.split(".")[0] for path in i0s.delta_paths(template(), value)}
    assert top == {
        "admission_reason",
        "budget",
        "canary",
        "i0s_envelope",
        "job_id",
        "output_dir",
        "requirements",
        "scientific_admission",
        "timeout_minutes",
    }
    assert value["env"]["INTEGRATION_BENCH_SECONDS"] == "5400"
    assert value["requirements"]["est_minutes"] == 110
    assert value["timeout_minutes"] == 220
    assert value["budget"]["unit_cap"] == 37.0
    assert math.isclose(value["budget"]["reserve_units"], 37 / 6)
    assert value["budget"]["rate_units_per_hour"] == 10.0
    assert value["job_id"] == "gpu-integration-i0s-main-968f02765f7c-a1"
    assert value["inputs"] == template()["inputs"]


def test_owner_2x_applied_once_and_supervisor_floors_hold() -> None:
    assert i0s.TIMEOUT_MINUTES == 2 * i0s.EST_MINUTES
    assert i0s.TIMEOUT_MINUTES * i0s.RATE_CEILING / 60 <= i0s.UNIT_CAP
    assert i0s.UNIT_CAP == math.ceil(i0s.TIMEOUT_MINUTES * i0s.RATE_CEILING / 60)
    assert i0s.RESERVATION == {"units": 42.0, "wall_minutes": 250.0}
    record = i0s.envelope_record()
    assert record["base"]["timeout_minutes"] == i0s.EST_MINUTES
    assert record["effective"]["timeout_minutes"] == i0s.TIMEOUT_MINUTES


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("requirements", "est_minutes"), 111),
        (("timeout_minutes",), 240),
        (("budget", "unit_cap"), 40.0),
        (("budget", "unit_cap"), 30.0),
        (("budget", "rate_units_per_hour"), 12.0),
        (("budget", "reserve_units"), 2 * 37 / 6),
        (("env", "INTEGRATION_BENCH_SECONDS"), "1440"),
        (("env", "INTEGRATION_BENCH_SECONDS"), "7200"),
        (("env", "METADATA_ARCHIVE_SHA256"), "f" * 64),
        (("env", "GPU_SIM_V8_INDEXED"), "1"),
        (("commit",), "0" * 40),
        (("lease_seconds",), 300),
        (("extra_field",), True),
    ],
)
def test_manifest_refuses_any_other_value(path: tuple, value: object) -> None:
    bad = manifest()
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(i0s.Hold):
        i0s.check_manifest(template(), bad, 1)


def test_nonempty_ledger_upload_is_refused() -> None:
    bad = manifest()
    bad["inputs"][1]["sha256"] = "e" * 64  # any bundled ledger changes metadata.tar
    with pytest.raises(i0s.Hold, match="inventory|outside"):
        i0s.check_manifest(template(), bad, 1)
    extra = manifest()
    extra["inputs"].append({"dest": "inputs/verified-ledger.json", "sha256": "d" * 64})
    with pytest.raises(i0s.Hold):
        i0s.check_manifest(template(), extra, 1)


def test_template_deadline_change_is_refused_at_build() -> None:
    changed = template()
    changed["env"]["INTEGRATION_BENCH_SECONDS"] = "1440"
    with pytest.raises(i0s.Hold):
        i0s.build_manifest(changed, CANARY, 1, i0s.attempt_dir(1) / "main-output")


@pytest.mark.parametrize("attempt", [0, 3, 4, -1])
def test_third_attempt_never_exists(attempt: int) -> None:
    for call in (i0s.job_id, i0s.reservation_id, i0s.attempt_dir, i0s.check_attempt_number):
        with pytest.raises(i0s.Hold):
            call(attempt)


def test_retry_manifest_must_equal_attempt_one() -> None:
    first, second = manifest(1), manifest(2)
    i0s.check_manifest(template(), second, 2, first)
    assert i0s.delta_paths(first, second) == {"job_id", "output_dir"}
    second["canary"]["receipt"]["analysis"] = {"verdict": "other"}
    with pytest.raises(i0s.Hold, match="identical"):
        i0s.check_manifest(template(), second, 2, first)
    with pytest.raises(i0s.Hold):
        i0s.check_manifest(template(), manifest(1), 2, first)  # attempt-1 job id as retry


def test_wrong_canary_receipt_or_output_refused() -> None:
    bad = manifest()
    bad["canary"]["receipt"]["sha256"] = "0" * 64
    with pytest.raises(i0s.Hold):
        i0s.check_manifest(template(), bad, 1)
    bad = manifest()
    bad["output_dir"] = "/tmp/elsewhere"
    with pytest.raises(i0s.Hold):
        i0s.check_manifest(template(), bad, 1)


# ---------------------------------------------------------------------- admission


def test_scopes_delta_records_parents_once(tmp_path: Path) -> None:
    rows = tree.make_register(tmp_path / "r.jsonl")
    state = register.state(rows)
    assert state["scopes"]["i0"]["ceiling"] == {"units": 66.0, "wall_minutes": 396.0}
    assert state["scopes"]["first_wave"]["ceiling"] == {"units": 88.8, "wall_minutes": 648.0}
    assert math.isclose(state["scopes"]["i0"]["spent"]["units"], 2.17994, abs_tol=1e-4)
    assert state["holds"]["first-wave-O1"]["account_units"] == 0
    with pytest.raises(i0s.Hold, match="already recorded"):
        i0s.scopes_delta(rows)


def test_attempt_one_reserves_42_units_250_minutes(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    tree.make_register(path)
    row = tree.admit(path, 1, tmp_path / "packet")
    assert row["reservation"]["pair"] == {"units": 42.0, "wall_minutes": 250.0}
    assert row["reservation"]["scopes"] == ["global", "q4", "first_wave", "i0"]
    parents = row["i0s"]["parents_before"]
    assert all(value["fits_reservation"] for value in parents.values())


def first_wave_total(path: Path) -> dict:
    fw = i0s.first_wave_state(register.read(path))
    return {key: fw["spent"][key] + fw["held"][key] for key in register.PAIR_KEYS}


def assert_first_wave_within_ceiling(path: Path) -> None:
    total = first_wave_total(path)
    assert total["units"] <= 88.8 + 1e-9 and total["wall_minutes"] <= 648 + 1e-9, total


def test_live_shape_admits_with_junior_o1_mirror(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    rows = tree.make_register(path, o1=(48.0, 288.0))  # live register shape
    mirror = register.state(rows)["holds"]["first-wave-O1"]
    assert mirror["pair"] == {"units": 44.62, "wall_minutes": 288.0}
    assert mirror["account_units"] == 0 and mirror["full_pair"] == {
        "units": 48.0,
        "wall_minutes": 288.0,
    }
    assert mirror["junior_to"].startswith("I0S")
    assert mirror["ruling"]["sha256"] == i0s.BUDGET_RULING_SHA
    assert i0s.BUDGET_RULING_SHA in {source["sha256"] for source in rows[-1]["sources"]}
    row = tree.admit(path, 1, tmp_path / "packet")
    available = row["i0s"]["parents_before"]["first_wave"]["available"]["units"]
    assert 42.0 <= available < 42.0001
    assert_first_wave_within_ceiling(path)


@pytest.mark.parametrize(("scope", "units"), [("i0", 25.0), ("q4", 130.0), ("global", 1010.0)])
def test_parent_shortfall_is_refused(tmp_path: Path, scope: str, units: float) -> None:
    path = tmp_path / "r.jsonl"
    tree.make_register(path)
    register.append(
        path,
        {
            "type": "delta",
            "id": "spend",
            "sources": tree.SOURCE,
            "spent_additions": {scope: {"units": units, "wall_minutes": 0.0}},
        },
    )
    with pytest.raises(i0s.Hold, match=scope):
        tree.admit(path, 1, tmp_path / "packet")


def test_scopes_must_be_recorded_before_admission(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    register.append(path, tree.init_row())
    with pytest.raises(i0s.Hold, match="register-scopes"):
        tree.admit(path, 1, tmp_path / "packet")


def test_latest_dispatch_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    tree.make_register(path)
    late = datetime(2026, 10, 26, 20, 0, tzinfo=timezone.utc)
    with pytest.raises(i0s.Hold, match="latest dispatch"):
        tree.admit(path, 1, tmp_path / "packet", now=late)
    tree.admit(path, 1, tmp_path / "packet", now=late - timedelta(minutes=1))


def test_stale_capture_and_low_balance_refused(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    tree.make_register(path)
    rows = register.read(path)
    stale = tree.package(1)
    stale["capture_utc"] = (tree.NOW - timedelta(minutes=11)).isoformat()
    with pytest.raises(register.Refused, match="stale"):
        i0s.admission_event(rows, stale, 1, tree.NOW, packet=tmp_path)
    with pytest.raises(register.Refused, match="balance"):
        i0s.admission_event(rows, tree.package(1, balance=60.0), 1, tree.NOW, packet=tmp_path)


def test_wrong_reservation_shape_or_identity_refused(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    tree.make_register(path)
    rows = register.read(path)
    bigger = tree.package(1)
    bigger["cap"] = {"units": 40.0, "wall_minutes": 220.0}
    with pytest.raises(i0s.Hold, match="42"):
        i0s.admission_event(rows, bigger, 1, tree.NOW, packet=tmp_path)
    with pytest.raises(i0s.Hold, match="identity"):
        i0s.admission_event(rows, tree.package(2), 1, tree.NOW, packet=tmp_path)


def test_refresh_reuses_reservation_and_never_double_reserves(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    tree.make_register(path)
    tree.admit(path, 1, tmp_path / "packet")
    later = tree.NOW + timedelta(minutes=5)
    row = tree.admit(path, 1, tmp_path / "packet", now=later)
    assert row["type"] == "refresh"
    assert len(register.state(register.read(path))["reservations"]) == 1


def mechanical_attempt_one(packet: Path, **fields: object) -> None:
    output = i0s.attempt_dir(1, packet) / "main-output"
    defaults = {"failure_class": "RPC_EXEC_POLL_TIMEOUT", "status": "INCOMPLETE"}
    defaults.update(fields)
    tree.receipt(output, **defaults)


def test_retry_admitted_only_after_settled_mechanical_zero_unit(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    mechanical_attempt_one(packet)
    i0s.attempt_dir(2, packet).mkdir(parents=True)
    with pytest.raises(i0s.Hold, match="old I0 hold"):
        tree.admit(path, 2, packet)
    tree.release_old_i0(path)
    with pytest.raises(i0s.Hold, match="not settled"):
        tree.admit(path, 2, packet)
    tree.settle(path, 1)
    row = tree.admit(path, 2, packet)
    assert row["reservation"]["id"] == i0s.reservation_id(2)
    with pytest.raises(i0s.Hold, match="after attempt 2"):
        tree.admit(path, 1, packet)


@pytest.mark.parametrize(
    "case", ["timeout", "units", "child_exit", "pass", "invalid", "no_receipt"]
)
def test_retry_refused_for_non_mechanical_or_unit_bearing_attempt(
    tmp_path: Path, case: str
) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    output = i0s.attempt_dir(1, packet) / "main-output"
    if case == "timeout":
        tree.timeout_tree(output, {"cell-frozen-e1024": (0.4, [2500.0] * 3)})
    elif case == "units":
        tree.timeout_tree(
            output, {"cell-frozen-e1024": (0.4, [2500.0] * 3)}, failure_class="SUPERVISOR"
        )
    elif case == "child_exit":
        tree.receipt(output, failure_class="CHILD_NONZERO_EXIT")
    elif case == "pass":
        tree.receipt(output, failure_class=None, status="PASS")
    elif case == "invalid":
        tree.timeout_tree(
            output, {}, failure="COMBINED_PARITY_MISMATCH", failure_class="SUPERVISOR"
        )
    else:
        output.mkdir(parents=True)
    tree.settle(path, 1)
    tree.release_old_i0(path)
    verdict = i0s.retry_predicate(register.read(path), packet, now=tree.NOW)
    assert not verdict["eligible"]
    i0s.attempt_dir(2, packet).mkdir(parents=True)
    with pytest.raises(i0s.Hold, match="retry predicate"):
        tree.admit(path, 2, packet)


def test_retry_refused_after_latest_dispatch(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    mechanical_attempt_one(packet)
    tree.settle(path, 1)
    late = datetime(2026, 10, 26, 20, 30, tzinfo=timezone.utc)
    assert not i0s.retry_predicate(register.read(path), packet, now=late)["eligible"]


def test_retry_needs_envelope_headroom_after_settlement(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    mechanical_attempt_one(packet)
    tree.settle(path, 1, charged=30.0)  # 66 - 2.18 - 30 < 42
    tree.release_old_i0(path)
    i0s.attempt_dir(2, packet).mkdir(parents=True)
    with pytest.raises(i0s.Hold, match="i0"):
        tree.admit(path, 2, packet)


def test_settled_attempt_cannot_be_readmitted(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    tree.settle(path, 1)
    with pytest.raises(i0s.Hold, match="already settled"):
        tree.admit(path, 1, packet)


# --------------------------------------------------------------------- settlement


def test_conservative_charge_uses_ceiling_or_balance_drop() -> None:
    charged = i0s.conservative_charge([{"elapsed_seconds": 6600.0}])
    assert math.isclose(charged["units"], 6600 / 3600 * 10)
    assert math.isclose(charged["wall_minutes"], 110.0)
    drop = i0s.conservative_charge(
        [
            {
                "elapsed_seconds": 600.0,
                "usage_before": {"balance": 1000.0},
                "usage_after": {"balance": 990.0},
            }
        ]
    )
    assert math.isclose(drop["units"], 10.0)
    two = i0s.conservative_charge([{"elapsed_seconds": 60.0}, {"elapsed_seconds": 6000.0}])
    assert math.isclose(two["wall_minutes"], 101.0)


def test_integer_balance_drop_is_counted() -> None:
    charged = i0s.conservative_charge(
        [
            {
                "elapsed_seconds": 600.0,
                "usage_before": {"balance": 1000},
                "usage_after": {"balance": 980},
            }
        ]
    )
    assert math.isclose(charged["units"], 20.0)


@pytest.mark.parametrize("elapsed", [None, float("nan"), -1.0, "600"])
def test_unknown_elapsed_never_means_zero(elapsed: object) -> None:
    with pytest.raises(i0s.Hold, match="unknown"):
        i0s.conservative_charge([{"elapsed_seconds": elapsed}])


def test_charge_above_reservation_is_escalated() -> None:
    with pytest.raises(i0s.Hold, match="exceeds"):
        i0s.conservative_charge([{"elapsed_seconds": 251 * 60.0}])


# ---------------------------------------------------------------- classification


def test_classification_is_path_only(tmp_path: Path) -> None:
    output = tmp_path / "main-output"
    tree.timeout_tree(output, {"cell-frozen-e1024": (0.4, [2500.0] * 3)})
    payload = output / "exports/integration-units/cell-frozen-e1024.json"
    payload.write_text("not json")  # classification must never parse payloads
    value = i0s.classify_attempt([output])
    assert value["unit_exports"] == 1 and not value["mechanical_zero_unit"]
    assert json.dumps(value)  # serializable, no rates
    assert "seconds" not in json.dumps(value)


# ------------------------------------------------- first-wave budget ruling (R-1..R-4)


def live_admitted(tmp_path: Path) -> tuple[Path, Path]:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path, o1=(48.0, 288.0))
    tree.admit(path, 1, packet)
    return path, packet


def restore(path: Path, attempt: int = 1) -> dict:
    event = i0s.restore_o1_delta(register.read(path), attempt, tree.SOURCE[0])
    register.append(path, event)
    return event


def test_ruling_constants_unchanged() -> None:
    assert i0s.RESERVATION == {"units": 42.0, "wall_minutes": 250.0}
    assert i0s.FIRST_WAVE == {"units": 88.8, "wall_minutes": 648.0}
    assert i0s.I0_CEILING == {"units": 66.0, "wall_minutes": 396.0}
    assert (i0s.EST_MINUTES, i0s.TIMEOUT_MINUTES, i0s.UNIT_CAP) == (110, 220, 37.0)
    assert i0s.OUTSIDE_TAIL == {"units": 5.0, "wall_minutes": 30.0}
    assert i0s.RATE_CEILING == 10.0 and i0s.LATEST_DISPATCH == "2026-10-26T20:00:00+00:00"
    assert i0s.O1_MIRROR == {"units": 44.62, "wall_minutes": 288.0}
    history = sum(row["pair"]["units"] for row in tree.HISTORY)
    assert 0 < 88.8 - history - 42 - i0s.O1_MIRROR["units"] < 1e-4


@pytest.mark.parametrize("charged", [18.3, 30.0, 41.0, 42.0])
def test_first_wave_never_exceeds_ceiling_through_settlement(
    tmp_path: Path, charged: float
) -> None:
    path, _ = live_admitted(tmp_path)
    assert_first_wave_within_ceiling(path)
    tree.release_old_i0(path)
    tree.settle(path, 1, charged=charged)
    assert_first_wave_within_ceiling(path)
    restore(path)
    assert_first_wave_within_ceiling(path)


def test_restore_returns_o1_to_full_at_expected_charge(tmp_path: Path) -> None:
    path, _ = live_admitted(tmp_path)
    with pytest.raises(i0s.Hold, match="not settled"):
        i0s.restore_o1_delta(register.read(path), 1, tree.SOURCE[0])
    tree.settle(path, 1, charged=18.3)
    event = restore(path)
    added = event["holds_add"][0]
    assert added["id"] == "first-wave-O1-restore" and added["account_units"] == 0
    assert math.isclose(added["pair"]["units"], 48 - 44.62)
    assert added["pair"]["wall_minutes"] == 0.0
    fw = i0s.first_wave_state(register.read(path))
    assert math.isclose(fw["mirror_total"]["units"], 48.0)
    with pytest.raises(i0s.Hold, match="already restored"):
        i0s.restore_o1_delta(register.read(path), 1, tree.SOURCE[0])


def test_restore_never_exceeds_remaining_headroom(tmp_path: Path) -> None:
    path, _ = live_admitted(tmp_path)
    tree.settle(path, 1, charged=41.0)
    event = restore(path)
    headroom = 88.8 - (sum(r["pair"]["units"] for r in tree.HISTORY) + 41.0) - 44.62
    assert math.isclose(event["holds_add"][0]["pair"]["units"], headroom)
    assert event["holds_add"][0]["pair"]["units"] < 48 - 44.62
    assert event["o1_short_in_first_wave"]["units"] > 0
    assert_first_wave_within_ceiling(path)


def test_restore_refused_when_first_wave_exhausted_or_o1_full(tmp_path: Path) -> None:
    path, _ = live_admitted(tmp_path)
    tree.settle(path, 1, charged=42.0)
    register.append(
        path,
        {
            "type": "delta",
            "id": "r1-spend",
            "sources": tree.SOURCE,
            "spent_additions": {"first_wave": {"units": 0.0, "wall_minutes": 0.0}},
        },
    )
    small = tmp_path / "small"
    small.mkdir()
    spath = small / "r.jsonl"
    tree.make_register(spath)  # O1 = 8 u: mirror already full
    tree.admit(spath, 1, small / "packet")
    tree.settle(spath, 1)
    with pytest.raises(i0s.Hold, match="nothing to restore"):
        i0s.restore_o1_delta(register.read(spath), 1, tree.SOURCE[0])


def test_retry_reduce_keeps_retry_inside_first_wave(tmp_path: Path) -> None:
    path, packet = live_admitted(tmp_path)
    tree.release_old_i0(path)
    tree.receipt(i0s.attempt_dir(1, packet) / "main-output", failure_class="RPC_EXEC_POLL_TIMEOUT")
    tree.settle(path, 1, charged=18.3)  # i0 keeps 45.5 u; first_wave forces a reduce
    restore(path)
    event = i0s.retry_o1_reduce_delta(register.read(path), tree.SOURCE[0])
    register.append(path, event)
    fw = i0s.first_wave_state(register.read(path))
    target = 88.8 - fw["spent"]["units"] - 42
    assert math.isclose(fw["mirror_total"]["units"], target)
    assert fw["mirrors"]["first-wave-O1-restore"]["pair"]["units"] == 0
    assert all(hold["account_units"] == 0 for hold in fw["mirrors"].values())
    i0s.attempt_dir(2, packet).mkdir(parents=True)
    tree.admit(path, 2, packet)
    assert_first_wave_within_ceiling(path)


def test_retry_needing_negative_o1_hold_is_refused(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)  # O1 mirror 8 u
    tree.admit(path, 1, packet)
    tree.settle(path, 1, charged=40.0)
    register.append(
        path,
        {
            "type": "delta",
            "id": "other-first-wave-spend",
            "sources": tree.SOURCE,
            "spent_additions": {"first_wave": {"units": 10.0, "wall_minutes": 0.0}},
        },
    )
    with pytest.raises(i0s.Hold, match="negative"):
        i0s.retry_o1_reduce_delta(register.read(path), tree.SOURCE[0])


def test_retry_reduce_is_no_op_when_retry_already_fits(tmp_path: Path) -> None:
    path, _ = live_admitted(tmp_path)
    tree.settle(path, 1, charged=1.0)  # 88.8 - 3.18 - 42 = 43.62 < 44.62 -> reduce
    assert i0s.retry_o1_reduce_delta(register.read(path), tree.SOURCE[0]) is not None
    small = tmp_path / "small"
    small.mkdir()
    tree.make_register(small / "r.jsonl")
    tree.admit(small / "r.jsonl", 1, small / "packet")
    tree.settle(small / "r.jsonl", 1)
    assert i0s.retry_o1_reduce_delta(register.read(small / "r.jsonl"), tree.SOURCE[0]) is None


def test_release_old_i0_hold(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    tree.make_register(path, o1=(48.0, 288.0))
    with pytest.raises(i0s.Hold, match="not reserved"):
        i0s.release_old_i0_delta(register.read(path), tree.SOURCE[0])
    tree.admit(path, 1, tmp_path / "packet")
    event = i0s.release_old_i0_delta(register.read(path), tree.SOURCE[0])
    assert event["settlement_attestation"] == i0s.OLD_I0_ATTESTATION
    assert {s["sha256"] for s in event["settlement_evidence"]} >= {
        i0s.BUDGET_RULING_SHA,
        i0s.RULING_SHA,
    }
    register.append(path, event)
    hold = register.state(register.read(path))["holds"]["I0"]
    assert hold["pair"] == {"units": 0.0, "wall_minutes": 0.0} and hold["account_units"] == 0
    assert hold["scopes"] == ["global", "q4"]
    with pytest.raises(i0s.Hold, match="already released"):
        i0s.release_old_i0_delta(register.read(path), tree.SOURCE[0])


def test_global_o1_hold_untouched_by_every_operation(tmp_path: Path) -> None:
    path, packet = live_admitted(tmp_path)
    original = copy.deepcopy(register.state(register.read(path))["holds"]["O1"])
    tree.release_old_i0(path)
    tree.receipt(i0s.attempt_dir(1, packet) / "main-output", failure_class="SUPERVISOR")
    tree.settle(path, 1, charged=35.0)
    restore(path)
    reduce_event = i0s.retry_o1_reduce_delta(register.read(path), tree.SOURCE[0])
    if reduce_event is not None:
        register.append(path, reduce_event)
    state = register.state(register.read(path))
    assert state["holds"]["O1"] == original
    assert original["pair"] == {"units": 48.0, "wall_minutes": 288.0}
    assert original["account_units"] == 48.0 and original["scopes"] == ["global"]
    assert state["scopes"]["first_wave"]["ceiling"] == {"units": 88.8, "wall_minutes": 648.0}


def test_latest_review_version_supersedes_rejected_original(tmp_path: Path) -> None:
    base = tmp_path / "CLAUDE_V_STANDIN_I0S_REVIEW.json"
    with pytest.raises(i0s.Hold):
        i0s.latest_review(base)
    base.write_text("{}")
    assert i0s.latest_review(base) == base
    for name in ("CLAUDE_V_STANDIN_I0S_REVIEW_02.json", "CLAUDE_V_STANDIN_I0S_REVIEW_10.json"):
        (tmp_path / name).write_text("{}")
    assert i0s.latest_review(base).name == "CLAUDE_V_STANDIN_I0S_REVIEW_10.json"
