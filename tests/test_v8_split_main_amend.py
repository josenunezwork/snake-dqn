"""Offline acceptance and refusal proofs for the exact V8 main cap amendment.

Use the qualified broker as cwd/PYTHONPATH; this file and the companion are
outside its shipped runtime tree. No allocation, native CUDA or main is run.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.colab_cli.supervisor import Refused
from research.gpu_sim_spike import v8_split_pack

BROKER = Path("/Users/josenunez/Projects/ml/snake-dqn-v8-metadata-qualified-02.noindex")
if v8_split_pack.ROOT != BROKER:
    pytest.skip(
        "exact saved-package tests require qualified broker cwd/PYTHONPATH and importlib mode",
        allow_module_level=True,
    )

SOURCE = Path(__file__).resolve().parents[1] / "tools/v8_split_main_amend.py"
spec = importlib.util.spec_from_file_location("v8_main_amend", SOURCE)
amend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amend)


@pytest.fixture(scope="module")
def main_manifest() -> dict:
    ruling = amend.authority(amend.RULING, amend.RULING_SHA)
    return amend.build(amend.accept_package(), ruling)


def test_preserved_acceptance_checker_ast() -> None:
    original = inspect.getsource(amend.pack._canary_binding)
    allowed = original.replace("20 * 60", "50 * 60").replace(
        "bounded 20-minute main estimate", "bounded 50-minute main estimate"
    )
    assert ast.dump(ast.parse(inspect.getsource(amend._canary_binding))) == ast.dump(
        ast.parse(allowed)
    )


def test_exact_amended_identity_and_frozen_supervisor(main_manifest) -> None:
    evidence = amend.validate_final(main_manifest)
    assert evidence["forecast_minutes"] == 49.23401820446816
    assert evidence["canary"]["analysis"]["verdict"] == "PASS_CANARY"
    assert evidence["pre_main"]["no_main_unit_has_run"] is True
    assert main_manifest["commit"] == amend.PIN
    assert main_manifest["requirements"]["est_minutes"] == 50
    assert main_manifest["timeout_minutes"] == 120
    assert main_manifest["budget"]["unit_cap"] == 20
    assert main_manifest["budget"]["reserve_units"] == 10 / 3
    assert main_manifest["v8_split"]["matrix"]["order"] == list(amend.v8_split_units.allocation())


def test_manifest_delta_is_only_the_ruled_fields(main_manifest) -> None:
    original = amend.template()
    restored = copy.deepcopy(main_manifest)
    for field in ("job_id", "output_dir", "canary", "timeout_minutes"):
        restored[field] = original[field]
    restored["requirements"]["est_minutes"] = original["requirements"]["est_minutes"]
    for field in ("unit_cap", "reserve_units"):
        restored["budget"][field] = original["budget"][field]
    restored["env"]["V8_SPLIT_BENCH_SECONDS"] = original["env"]["V8_SPLIT_BENCH_SECONDS"]
    for field in ("effective_envelope", "main"):
        restored["v8_split"][field] = original["v8_split"][field]
    del restored["cap_amendment"]
    assert restored == original


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_missing_or_wrong_amendment_refused(main_manifest, mode) -> None:
    value = copy.deepcopy(main_manifest)
    if mode == "missing":
        del value["cap_amendment"]
    else:
        value["cap_amendment"]["ruling"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authority hash"):
        amend.validate_final(value)


@pytest.mark.parametrize(
    "elapsed,benchmark",
    [(3001, 1), (100, 100), (float("nan"), 1), (float("inf"), 1), (True, 1), (1, 2)],
)
def test_forecast_above_50_or_malformed_refused(elapsed, benchmark) -> None:
    with pytest.raises(ValueError):
        amend.check_forecast(elapsed, benchmark)


@pytest.mark.parametrize(
    "fields,value",
    [
        (("timeout_minutes",), 121),
        (("budget", "unit_cap"), 20.01),
        (("budget", "reserve_units"), 1),
        (("requirements", "est_minutes"), 20),
        (("requirements", "gpu"), "L4"),
        (("requirements", "min_vcpu"), 11),
        (("env", "V8_SPLIT_BENCH_SECONDS"), "1200"),
        (("commit",), "a" * 40),
        (("env", "V8_SPLIT_BENCH_LEDGER_PATH"), "foreign-ledger.json"),
        (("v8_split", "matrix", "order"), ["cell-frozen-e1024"]),
        (("v8_split", "matrix", "repeats"), 4),
        (("v8_split", "matrix", "warmups"), 1),
        (("v8_split", "matrix", "arms"), ["R", "G"]),
        (("v8_split", "effective_envelope", "units"), 25),
        (("resume", "skip_verified"), False),
    ],
)
def test_altered_budget_identity_science_or_coverage_refused(main_manifest, fields, value) -> None:
    changed = copy.deepcopy(main_manifest)
    target = changed
    for field in fields[:-1]:
        target = target[field]
    target[fields[-1]] = value
    with pytest.raises(ValueError, match="exact amended"):
        amend.validate_final(changed)


def test_changed_archive_refused_by_both_paths(main_manifest, tmp_path) -> None:
    changed = copy.deepcopy(main_manifest)
    row = next(row for row in changed["inputs"] if row["dest"] == "metadata.tar")
    payload = bytearray(Path(row["path"]).read_bytes())
    payload[-1] ^= 1
    archive = tmp_path / "metadata.tar"
    archive.write_bytes(payload)
    row.update(path=str(archive), sha256=amend.sha256_file(archive))
    changed["env"]["METADATA_ARCHIVE_SHA256"] = row["sha256"]
    with pytest.raises(ValueError, match="exact amended"):
        amend.validate_final(changed)
    with pytest.raises(Refused, match="canary does not cover real upload bytes"):
        amend.supervisor_validate(changed, repo=amend.BROKER)


@pytest.mark.parametrize("change", ["inventory", "native", "producer", "release", "discrepancy"])
def test_forged_or_missing_canary_evidence_refused(main_manifest, tmp_path, change) -> None:
    original = amend.SOURCE / "canary-output"
    for name in ("receipt.json", "manifest.json"):
        shutil.copyfile(original / name, tmp_path / name)
    shutil.copytree(original / "exports", tmp_path / "exports")
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if change == "release":
        receipt["stop_verified"] = False
    elif change == "inventory":
        receipt["exports"][0]["size"] += 1
    else:
        name = {
            "native": "native-parity.json",
            "producer": "v8-split-bench.json",
            "discrepancy": "discrepancy-register.json",
        }[change]
        (tmp_path / "exports" / name).unlink()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, OSError)):
        amend._canary_binding(receipt_path, amend.PIN, main_manifest["mac_preflight"])


@pytest.mark.parametrize("field", ["namespace", "allocation"])
def test_foreign_ledger_or_allocation_refused(tmp_path, field) -> None:
    units = amend.v8_split_units
    ledger = units.empty_ledger(units.namespace(amend.PIN), units.allocation())
    ledger[field] = "foreign" if field == "namespace" else list(units.allocation())[1:]
    path = tmp_path / "verified-ledger.json"
    units.publish_json(path, ledger)
    with pytest.raises(ValueError, match="foreign"):
        units.load_ledger(path, units.namespace(amend.PIN), units.allocation())


def test_missing_live_accounting_refuses_without_allocation() -> None:
    with pytest.raises((ValueError, OSError)):
        amend.admission()


def test_unchanged_frozen_guard_still_refuses() -> None:
    preflight = amend.template()["mac_preflight"]
    with pytest.raises(ValueError, match="bounded 20-minute"):
        amend.pack._canary_binding(
            amend.SOURCE / "canary-output/receipt.json", amend.PIN, preflight
        )


@pytest.mark.parametrize("mode", ["accepted", "expired", "future", "too_late", "naive", "missing"])
def test_live_accounting_time_gate(mode) -> None:
    now = datetime(2026, 10, 24, tzinfo=timezone.utc)
    body = {
        "accounting_as_of_utc": (now - timedelta(minutes=1)).isoformat(),
        "valid_until_utc": (now + timedelta(minutes=5)).isoformat(),
        "calendar_deadline_utc": (now + timedelta(minutes=130)).isoformat(),
    }
    if mode == "accepted":
        amend.accounting_time(body, now)
        return
    if mode == "expired":
        body["valid_until_utc"] = (now - timedelta(seconds=1)).isoformat()
    elif mode == "future":
        body["accounting_as_of_utc"] = (now + timedelta(seconds=1)).isoformat()
    elif mode == "too_late":
        body["calendar_deadline_utc"] = (now + timedelta(minutes=119)).isoformat()
    elif mode == "naive":
        body["calendar_deadline_utc"] = now.replace(tzinfo=None).isoformat()
    else:
        del body["accounting_as_of_utc"]
    with pytest.raises(ValueError, match="INCOMPLETE_ADMISSION"):
        amend.accounting_time(body, now)


@pytest.mark.parametrize(
    "mode", ["accepted", "static_missing", "static_wrong", "live_missing", "live_wrong_review"]
)
def test_dispatch_requires_exact_static_and_live_reviews(tmp_path, monkeypatch, mode) -> None:
    # Only the Mac companion's artifact root moves to this inert fixture. Frozen
    # packer, broker and validator functions are never patched.
    monkeypatch.setattr(amend, "ART", tmp_path)
    publish = amend.v8_split_units.publish_json
    publish(tmp_path / "PREPARATION.json", {"synthetic_fixture": True})
    publish(tmp_path / "v8-split-main.json", {"synthetic_fixture": True})
    static = {
        "status": "ACCEPT_OFFLINE",
        "implementation": amend.implementation(),
        "preparation_sha256": amend.sha256_file(tmp_path / "PREPARATION.json"),
        "manifest_sha256": amend.sha256_file(tmp_path / "v8-split-main.json"),
    }
    if mode == "static_wrong":
        static["preparation_sha256"] = "0" * 64
    if mode != "static_missing":
        publish(tmp_path / "CODEX_V_LIGHT.json", static)
    now = datetime.now(timezone.utc)
    body = {
        "status": "ADMIT_INITIAL_MAIN",
        "manifest_sha256": static["manifest_sha256"],
        "ruling_sha256": amend.RULING_SHA,
        "no_main_unit_has_run": True,
        "previous_endpoint_release_verified": True,
        "a100_serial_sprint": True,
        "original_t0": "synthetic",
        "historical_expenditure": {"units": 1},
        "liabilities_and_reservations": [],
        "q2_q4_remaining": 30,
        "v8_series_remaining": 30,
        "live_unencumbered_units": 100,
        "current_assignments_and_rates": [],
        "accounting_evidence": "synthetic",
        "reserved_main": {"wall_minutes": 120, "units": 20},
        "fits_all_original_parent_caps_and_deadlines": True,
        "accounting_as_of_utc": (now - timedelta(minutes=1)).isoformat(),
        "valid_until_utc": (now + timedelta(minutes=5)).isoformat(),
        "calendar_deadline_utc": (now + timedelta(minutes=130)).isoformat(),
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    review = tmp_path / "accounting-review.json"
    publish(
        review,
        {
            "status": "ACCEPT",
            "accounting_sha256": "0" * 64 if mode == "live_wrong_review" else digest,
        },
    )
    body["codex_v_review"] = {"path": str(review), "sha256": amend.sha256_file(review)}
    if mode != "live_missing":
        publish(tmp_path / "OPERATOR_ADMISSION.json", body)
    if mode == "accepted":
        amend.admission()
    else:
        with pytest.raises((ValueError, OSError)):
            amend.admission()
