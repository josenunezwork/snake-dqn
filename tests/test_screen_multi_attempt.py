"""Bank-v2 confirmation + multi-attempt tooling: allocation, pool safety, session budget,
rejected checkpoints, ledger builder and merge."""

from __future__ import annotations

import json
import textwrap

import pytest

from research.watch_replay import screen_l1_os as screen
from research.watch_replay import screen_ledger
from research.watch_replay.seed_audit import (
    BANK_V2_CONFIRM_NAMESPACE,
    BANK_V2_DEV_NAMESPACE,
    BANK_V2_SCREEN_NAMESPACE,
    bank_v2_disjointness,
)
from tools.ops.package_spec import PackageRefused, publish_json, unit_id, verify_immutable


def _result(contract: dict, unit: dict, mass: float = 100.0) -> dict:
    return {
        "schema": screen.SCHEMA,
        "unit_id": unit_id(contract["namespace"], screen.ALLOCATION, unit["index"]),
        "allocation": screen.identity(contract),
        "index": unit["index"],
        "world": unit["world"],
        "arm": unit["arm"],
        "seed": unit["seed"],
        "horizon": contract["horizon"],
        "hero_mass_integral": mass,
        "hero_deaths": 0,
    }


# --- 1. confirmation allocation -----------------------------------------------------------


def test_confirmation_study_allocation_and_namespace_disjointness():
    spec = screen.STUDIES["confirm"]
    assert spec == {"namespace": BANK_V2_CONFIRM_NAMESPACE, "count": 96, "horizon": 14400}
    assert BANK_V2_CONFIRM_NAMESPACE == "t34-bank-v2-confirm-20261105-v1"
    contract = screen.allocation(**spec)
    assert len(contract["units"]) == 192 and contract["horizon"] == 14400
    # The screen allocation (and so its readout identity) is unchanged.
    assert screen.allocation() == screen.allocation(**screen.STUDIES["screen"])
    report = bank_v2_disjointness()
    assert report["status"] == "CLEAR"
    confirm = set(report["namespaces"][BANK_V2_CONFIRM_NAMESPACE])
    assert len(confirm) == 96
    for other in (BANK_V2_SCREEN_NAMESPACE, BANK_V2_DEV_NAMESPACE):
        assert not confirm & set(report["namespaces"][other])
    assert {u["seed"] for u in contract["units"]} == confirm
    with pytest.raises(ValueError):
        screen.allocation(BANK_V2_CONFIRM_NAMESPACE, 96, 14401)


def test_confirmation_forecast_session_plan():
    from research.watch_replay.prepare_screen import SESSION_SECONDS, forecast

    assert SESSION_SECONDS == 9000
    central = forecast("confirm", "central", shards=2)
    high = forecast("confirm", "high", shards=2)
    assert central["units"] == 192
    assert central["sessions_per_shard"] == high["sessions_per_shard"] == 4
    assert central["session_minutes"] <= 180  # each host session stays under ~3 h
    assert 200 < central["units_total"] < 260


# --- 3. pool-safe scheduler and session budget -------------------------------------------

FAKE = textwrap.dedent("""
    import os, time
    from pathlib import Path
    from research.watch_replay import screen_l1_os as screen
    from tools.ops.package_spec import publish_json, unit_id

    def _publish(contract, unit, out):
        result = {
            "schema": screen.SCHEMA,
            "unit_id": unit_id(contract["namespace"], screen.ALLOCATION, unit["index"]),
            "allocation": screen.identity(contract),
            "index": unit["index"], "world": unit["world"], "arm": unit["arm"],
            "seed": unit["seed"], "horizon": contract["horizon"],
        }
        publish_json(out / "units" / f"{unit['index']:06d}.json", result)
        return result

    def hang_and_crash(contract, unit, out, check, **kwargs):
        if unit["index"] == 1:
            time.sleep(3600)          # hung without heartbeat or check()
        if unit["index"] == 3:
            os._exit(7)               # e.g. faulthandler hard exit
        return _publish(contract, unit, out)

    def slow(contract, unit, out, check, **kwargs):
        try:
            for _ in range(600):
                check()
                kwargs["heartbeat"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["heartbeat"].touch()
                time.sleep(0.05)
        except InterruptedError:
            marker = out / "checkpoints" / f"{unit['index']:06d}" / "stopped"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("checkpoint")
            raise
        return _publish(contract, unit, out)
    """)


@pytest.fixture
def fakemod(tmp_path, monkeypatch):
    (tmp_path / "fake_screen_units.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp_path))
    return "fake_screen_units"


def test_hung_or_crashing_unit_fails_alone_and_pool_continues(tmp_path, fakemod):
    contract = screen.allocation("t34-bank-v2-unit-test", 3, 10)
    out = tmp_path / "screen"
    report = screen.run(
        contract,
        out,
        3,
        runner=f"{fakemod}:hang_and_crash",
        hang_seconds=4.0,
        poll_seconds=0.2,
    )
    assert report["completed_now"] == [0, 2, 4, 5]
    by_index = {e["index"]: e for e in report["exclusions"]}
    assert set(by_index) == {1, 3}
    assert "hang" in by_index[1]["error"]
    assert by_index[3]["exitcode"] == 7
    # A second attempt retries only the failed units (resume by default).
    pending, skipped = screen.pending_units(contract, out, None)
    assert skipped == [0, 2, 4, 5] and sorted(u["index"] for u in pending) == [1, 3]


def test_session_budget_checkpoints_running_units_and_leaves_rest(tmp_path, fakemod):
    contract = screen.allocation("t34-bank-v2-unit-test", 3, 10)
    out = tmp_path / "screen"
    report = screen.run(
        contract,
        out,
        2,
        runner=f"{fakemod}:slow",
        session_seconds=3.0,
        poll_seconds=0.2,
    )
    assert report["stopped"] == "session budget"
    assert report["completed_now"] == []
    stopped = sorted(e["index"] for e in report["exclusions"])
    assert len(stopped) == 2 and all(e["exitcode"] == 3 for e in report["exclusions"])
    assert sorted(report["not_started"] + stopped) == list(range(6))
    for index in stopped:
        assert (out / "checkpoints" / f"{index:06d}" / "stopped").read_text() == "checkpoint"


def test_in_process_stalled_unit_does_not_stop_the_run(tmp_path, monkeypatch):
    contract = screen.allocation("t34-bank-v2-unit-test", 2, 10)

    def fake(contract, unit, out, check, **kwargs):
        if unit["index"] == 1:
            raise screen.UnitStalled("no frame")
        result = _result(contract, unit)
        publish_json(out / "units" / f"{unit['index']:06d}.json", result)
        return result

    monkeypatch.setattr(screen, "run_unit", fake)
    report = screen.run(contract, tmp_path / "s", 1)
    assert report["completed_now"] == [0, 2, 3] and report["stopped"] is None
    assert [e["index"] for e in report["exclusions"]] == [1]


def test_shard_selects_worlds_mod_n(tmp_path, monkeypatch):
    contract = screen.allocation("t34-bank-v2-unit-test", 4, 10)
    seen = []

    def fake(contract, unit, out, check, **kwargs):
        seen.append(unit["world"])
        result = _result(contract, unit)
        publish_json(out / "units" / f"{unit['index']:06d}.json", result)
        return result

    monkeypatch.setattr(screen, "run_unit", fake)
    screen.run(contract, tmp_path / "s", 1, shard=(1, 2))
    assert sorted(set(seen)) == [1, 3] and len(seen) == 4


# --- should-fix 3: rejected checkpoints are never offered again ----------------------------


def _ckpt(contract, index, frame, sha, hardware="other-host"):
    return {
        "schema": screen.CHECKPOINT_SCHEMA,
        "allocation": screen.identity(contract),
        "index": index,
        "frame": frame,
        "frames_sha256": sha,
        "hardware": hardware,
    }


def test_rejected_checkpoint_is_skipped_by_resume_and_pack(tmp_path):
    import tarfile

    contract = screen.allocation("t34-bank-v2-unit-test", 1, 10)
    out = tmp_path / "out"
    root = out / "checkpoints"
    good = _ckpt(contract, 1, 5, "a" * 64)
    bad = _ckpt(contract, 1, 9, "b" * 64)
    publish_json(root / "000001" / "frame-000005.json", good)
    publish_json(root / "000001" / "frame-000009.json", bad)
    assert screen.latest_checkpoint([root], 1, contract)["frame"] == 9
    screen.reject_checkpoint(out, 1, bad, "frames 1..9 differ")
    assert [c["frame"] for c in screen.checkpoint_candidates([root], 1, contract)] == [5]
    # The marker also applies when the bad checkpoint comes from another source dir.
    other = tmp_path / "attempt1"
    publish_json(other / "000001" / "frame-000009.json", bad)
    assert screen.latest_checkpoint([root, other], 1, contract)["frame"] == 5
    archive = tmp_path / "ck.tar.gz"
    assert screen.pack_checkpoint_resume([root], archive, exclude=set()) == 1
    with tarfile.open(archive) as tar:
        assert "000001/frame-000005.json" in tar.getnames()
        assert "000001/frame-000009.json" not in tar.getnames()


def test_same_hardware_checkpoint_preferred_over_newer_foreign(tmp_path):
    contract = screen.allocation("t34-bank-v2-unit-test", 1, 10)
    root = tmp_path / "checkpoints"
    here = screen.hardware_fingerprint()
    publish_json(root / "000000" / "frame-000004.json", _ckpt(contract, 0, 4, "c" * 64, here))
    publish_json(root / "000000" / "frame-000008.json", _ckpt(contract, 0, 8, "d" * 64))
    assert [c["frame"] for c in screen.checkpoint_candidates([root], 0, contract)] == [4, 8]


# --- 2. ledger builder and merge ------------------------------------------------------------


def _attempt_dir(tmp_path, name, contract, indices, *, layout="exports/screen", mass=100.0):
    root = tmp_path / name / layout
    publish_json(root / "contract.json", contract)
    for unit in contract["units"]:
        if unit["index"] in indices:
            publish_json(
                root / "units" / f"{unit['index']:06d}.json", _result(contract, unit, mass)
            )
    return tmp_path / name


def test_ledger_and_merge_union_attempts_end_to_end(tmp_path):
    contract = {**screen.allocation("t34-bank-v2-unit-test", 3, 10), "commit": "c" * 40}
    a1 = _attempt_dir(tmp_path, "attempt1", contract, {0, 1, 2, 3})
    a2 = _attempt_dir(tmp_path, "attempt2", contract, {3, 4})
    mac = _attempt_dir(tmp_path, "mac", contract, {5}, layout=".")
    # A partial publication (no sidecars) is ignored, never counted as verified.
    (a2 / "exports/screen/units/000005.json").write_text("{}")
    ledger = screen_ledger.build_ledger(contract, [a1, a2])
    assert [u["index"] for u in ledger["units"]] == [0, 1, 2, 3, 4]
    publish_json(tmp_path / "ledger.json", ledger)
    pending, skipped = screen.pending_units(contract, tmp_path / "next", tmp_path / "ledger.json")
    assert skipped == [0, 1, 2, 3, 4] and [u["index"] for u in pending] == [5]
    merged = tmp_path / "merged"
    report = screen_ledger.merge(contract, [a1, a2, mac], merged)
    assert report["copied"] == list(range(6)) and report["missing"] == []
    for i in range(6):
        verify_immutable(merged / "units" / f"{i:06d}.json")
    again = screen_ledger.merge(contract, [a1, a2, mac], merged)
    assert again["copied"] == [] and again["already_present"] == list(range(6))
    # The merged dir is itself a valid ledger source and analysis input.
    assert len(screen_ledger.build_ledger(contract, [merged])["units"]) == 6
    from research.watch_replay.analyze_screen import load_units

    assert len(load_units(merged, contract)) == 6


def test_ledger_refuses_conflicting_duplicates_and_foreign_sources(tmp_path):
    contract = screen.allocation("t34-bank-v2-unit-test", 2, 10)
    a1 = _attempt_dir(tmp_path, "attempt1", contract, {0, 1})
    a2 = _attempt_dir(tmp_path, "attempt2", contract, {1}, mass=101.0)
    with pytest.raises(PackageRefused, match="differs across attempts"):
        screen_ledger.build_ledger(contract, [a1, a2])
    with pytest.raises(PackageRefused, match="differs across attempts"):
        screen_ledger.merge(contract, [a1, a2], tmp_path / "merged")
    other = screen.allocation("t34-bank-v2-other-test", 2, 10)
    foreign = _attempt_dir(tmp_path, "foreign", other, {0})
    with pytest.raises(PackageRefused, match="different allocation"):
        screen_ledger.build_ledger(contract, [foreign])
    # A unit result whose seed does not match the allocation is refused.
    bad = tmp_path / "bad" / "units" / "000000.json"
    publish_json(bad, {**_result(contract, contract["units"][0]), "seed": 7})
    with pytest.raises(PackageRefused, match="foreign or mismatched"):
        screen_ledger.build_ledger(contract, [tmp_path / "bad"])


def test_ledger_cli_writes_a_prepare_compatible_ledger(tmp_path):
    import subprocess
    import sys

    contract = screen.allocation("t34-bank-v2-unit-test", 1, 10)
    a1 = _attempt_dir(tmp_path, "attempt1", contract, {0})
    publish_json(tmp_path / "contract.json", contract)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.watch_replay.screen_ledger",
            "ledger",
            "--contract",
            str(tmp_path / "contract.json"),
            "--sources",
            str(a1),
            "--out",
            str(tmp_path / "ledger.json"),
        ],
        check=True,
        capture_output=True,
    )
    from tools.ops.package_spec import read_ledger

    rows = read_ledger(
        tmp_path / "ledger.json",
        contract["namespace"],
        screen.ALLOCATION,
        screen.identity(contract),
    )
    assert [r["index"] for r in rows] == [0]
    assert json.loads((tmp_path / "ledger.json").read_text())["units"][0]["size"] > 0
