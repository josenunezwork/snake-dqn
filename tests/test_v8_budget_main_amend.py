"""Exact host amendment deltas, exercising the unchanged saved canary checks."""

from __future__ import annotations

import copy
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.gpu_sim_spike import v8_split_pack

BROKER = Path("/Users/josenunez/Projects/ml/snake-dqn-v8-metadata-qualified-02.noindex")
if v8_split_pack.ROOT != BROKER:
    pytest.skip("requires exact qualified broker and importlib mode", allow_module_level=True)

SOURCE = Path(__file__).resolve().parents[1] / "tools/v8_budget_main_amend.py"
spec = importlib.util.spec_from_file_location("v8_budget_amendment", SOURCE)
amend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amend)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return amend.build(amend.previous().accept_package())


def test_exact_authorized_packet_and_preserved_runtime(manifest: dict) -> None:
    evidence = amend.validate_final(manifest)
    assert evidence["forecast_minutes"] == 49.23401820446816
    assert evidence["canary"]["analysis"]["verdict"] == "PASS_CANARY"
    assert manifest["commit"] == amend.PIN
    assert manifest["timeout_minutes"] == 120
    assert manifest["requirements"]["est_minutes"] == 50
    assert manifest["budget"]["unit_cap"] == 20
    assert manifest["budget"]["reserve_units"] == 10 / 3
    assert manifest["env"]["V8_SPLIT_BENCH_SECONDS"] == "5400"
    assert manifest["v8_split"]["effective_envelope"] == {"wall_minutes": 150, "units": 25}
    assert manifest["v8_split"]["canary"]["wall_minutes"] == 30
    assert manifest["v8_split"]["canary"]["units"] == 5
    assert manifest["admission_budget_amendment"]["historical_canary_charge_ceiling"] == {
        "wall_minutes": 30,
        "units": 5,
    }


def test_exact_local_delta_from_previous_package(manifest: dict) -> None:
    old = amend.previous()
    previous = old.build(old.accept_package(), old.authority(old.RULING, old.RULING_SHA))
    restored = copy.deepcopy(manifest)
    del restored["admission_budget_amendment"]
    for key in ("job_id", "output_dir", "cap_amendment"):
        restored[key] = previous[key]
    restored["v8_split"]["effective_envelope"] = previous["v8_split"]["effective_envelope"]
    restored["v8_split"]["canary"] = previous["v8_split"]["canary"]
    assert restored == previous
    assert restored["inputs"] == previous["inputs"]
    assert len(restored["v8_split"]["matrix"]["order"]) == 9


@pytest.mark.parametrize(
    "path,value",
    [
        (("admission_budget_amendment", "ruling", "sha256"), "0" * 64),
        (("admission_budget_amendment", "packet_cap", "units"), 26),
        (("admission_budget_amendment", "historical_canary_charge_ceiling", "units"), 6),
        (("admission_budget_amendment", "effective_topup", "units"), 2),
        (("admission_budget_amendment", "previous_split_main_ruling", "sha256"), "0" * 64),
        (("v8_split", "canary", "units"), 4),
        (("v8_split", "canary", "wall_minutes"), 31),
        (("v8_split", "effective_envelope", "wall_minutes"), 151),
        (("timeout_minutes",), 119),
        (("timeout_minutes",), 121),
        (("budget", "unit_cap"), 19.65),
        (("budget", "unit_cap"), 21),
        (("budget", "reserve_units"), 0),
        (("requirements", "est_minutes"), 49),
        (("requirements", "min_vcpu"), 11),
        (("requirements", "gpu"), "L4"),
        (("commit",), "a" * 40),
        (("env", "V8_SPLIT_BENCH_SECONDS"), "7200"),
        (("env", "V8_SPLIT_BENCH_LEDGER_PATH"), "foreign.json"),
        (("v8_split", "matrix", "repeats"), 4),
        (("v8_split", "matrix", "order"), ["cell-frozen-e1024"]),
        (("resume", "skip_verified"), False),
    ],
)
def test_foreign_or_reduced_contract_refused(manifest: dict, path: tuple, value: object) -> None:
    changed = copy.deepcopy(manifest)
    target = changed
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="exact budget amendment"):
        amend.validate_final(changed)


def test_missing_budget_authority_refused(manifest: dict) -> None:
    changed = copy.deepcopy(manifest)
    del changed["admission_budget_amendment"]
    with pytest.raises(ValueError, match="exact budget amendment"):
        amend.validate_final(changed)


def test_missing_static_or_live_acceptance_refuses() -> None:
    with pytest.raises((ValueError, OSError)):
        amend.admission()


@pytest.mark.parametrize("mode", ["accepted", "freshness", "calendar"])
def test_dispatch_wait_stays_inside_live_and_calendar_boundary(mode: str) -> None:
    now = datetime.now(timezone.utc)
    body = {
        "valid_until_utc": (now + timedelta(minutes=11)).isoformat(),
        "calendar_deadline_utc": (now + timedelta(minutes=131)).isoformat(),
    }
    if mode == "accepted":
        amend.dispatch_window(body, now)
        return
    key = "valid_until_utc" if mode == "freshness" else "calendar_deadline_utc"
    body[key] = (now + timedelta(minutes=9 if mode == "freshness" else 129)).isoformat()
    with pytest.raises(ValueError, match="window"):
        amend.dispatch_window(body, now)
