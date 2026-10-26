"""Refusal coverage for the sealed F5 step-3 driver boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_residual import f5_environment, f5_package, f5_r10a_annex
from research.bl_residual import f5_step3_driver as subject
from research.bl_residual import f5_timing, f5_timing_supplement


def _control(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "b8.pth").write_bytes(b"fixture-b8")
    export_json(inputs / "qualification.json", {"fixture": "qualification"})
    fixtures = tmp_path / "fixtures.json"
    allocation = {row["allocation_index"]: row for row in f5_r10a_annex.allocate(0)}
    allowed = {"allocation_index", "group", "group_index", "parent_id", "timing_only"}
    export_json(
        fixtures,
        {
            "fixtures": [
                {
                    **{key: value for key, value in row.items() if key in allowed},
                    "seed": allocation[row["allocation_index"]]["seed"],
                }
                for row in f5_timing._canonical_timing_parents()
            ]
        },
    )
    source_hashes = f5_package.source_inventory(Path(subject.__file__).resolve().parents[2])
    control = {
        "schema": subject.SCHEMA,
        "namespace": "step3-fixture",
        "allocation": "a" * 64,
        "source_hashes": source_hashes,
        "source_sha256": subject._canonical_sha(source_hashes),
        "fixture_descriptor_sha256": digest(fixtures),
        "input_hashes": {
            "b8.pth": {"sha256": digest(inputs / "b8.pth"), "size": 10},
            "qualification.json": {
                "sha256": digest(inputs / "qualification.json"),
                "size": (inputs / "qualification.json").stat().st_size,
            },
        },
        "environment_policy": __import__(
            "research.bl_residual.f5_environment", fromlist=["policy_binding"]
        ).policy_binding(),
        "environment_reference": {"fixture": True},
        "execution_envelope": {
            "host_workers": 6,
            "threads_per_worker": 1,
            "base_timeout_minutes": 60,
            "effective_timeout_minutes": 120,
            "forecast_status": "HOLD",
        },
    }
    control_path = tmp_path / "control.json"
    export_json(control_path, control)
    ledger = tmp_path / "verified-ledger.json"
    export_json(ledger, subject.empty_ledger(control))
    return control_path, fixtures, inputs, control


def test_dry_run_validates_exact_frozen_fixtures_without_runtime_calls(tmp_path: Path) -> None:
    control, fixtures, inputs, _ = _control(tmp_path)
    assert (
        subject.run(
            control_path=control,
            fixtures=fixtures,
            inputs=inputs,
            output=tmp_path / "out",
            ledger=tmp_path / "verified-ledger.json",
            mode="dry-run",
        )["status"]
        == "DRY_RUN_VALID"
    )


def test_source_or_input_mutation_refuses_before_runtime(tmp_path: Path) -> None:
    control, fixtures, inputs, _ = _control(tmp_path)
    (inputs / "b8.pth").write_bytes(b"changed-b8")
    with pytest.raises(subject.Step3Refusal, match="uploaded input differs"):
        subject.validate_control(control, fixtures, inputs)


def _record(fixture_id: int) -> dict[str, Any]:
    fixture = next(
        x for x in f5_timing._canonical_timing_parents() if x["allocation_index"] == fixture_id
    )
    stream = fixture["group"].split("/", 1)[1]
    return {
        "schema": "bl-f5-targeted-long-body-supplement-draft/v1",
        "component_id": "f5-targeted-long-body-first-crossing-v1",
        "component_key": f"f5-targeted-long-body-first-crossing-v1/{fixture_id}",
        "allocation_index": fixture_id,
        "parent_id": fixture["parent_id"],
        "stream": stream,
        "branch_source_seam": (
            "scripted-batched-one-action"
            if stream == "scripted"
            else "common-native-serial-one-action"
        ),
        "horizon": 5000,
        "long_body_minimum": 700,
        "fit_usable": False,
        "science_admission": False,
        "fixture_descriptor_sha256": "b" * 64,
        "source_provenance": {
            "source_sha256": "c" * 64,
            "environment_sha256": "d" * 64,
            "native_v6": True,
        },
        "kernel_provenance": {
            "assignment_raw_sha256": "e" * 64,
            "platform": "Linux",
            "release": "6.6",
        },
        "crossing": "ABSENT_WITHIN_H5000",
        "crossing_decision": None,
    }


def _unit(output: Path, control: dict[str, Any], fixture_id: int) -> dict[str, Any]:
    record = _record(fixture_id)
    record["fixture_descriptor_sha256"] = control["fixture_descriptor_sha256"]
    record["source_provenance"]["source_sha256"] = control["source_sha256"]
    record["operational_payload_sha256"] = __import__(
        "research.bl_residual.f5_timing_supplement", fromlist=["operational_payload_sha256"]
    ).operational_payload_sha256(record)
    path = output / "units" / f"fixture-{fixture_id}.json"
    exported = export_json(
        path,
        {
            "schema": subject.UNIT_SCHEMA,
            "component_id": subject.COMPONENT_ID,
            "identity": subject._binding(control),
            "fixture_id": fixture_id,
            "record": record,
            "elapsed_seconds": 1.0,
            "collection_elapsed_seconds": 1.0,
        },
    )
    return {
        "id": fixture_id,
        "path": path.relative_to(output).as_posix(),
        "sha256": exported["sha256"],
        "size": exported["size"],
    }


def test_completed_absent_unit_is_verified_and_skip_eligible(tmp_path: Path) -> None:
    _, _, _, control = _control(tmp_path)
    output = tmp_path / "prior"
    item = _unit(output, control, 672)
    ledger = output / "verified-ledger.json"
    export_json(ledger, {**subject.empty_ledger(control), "units": [item]})
    assert set(subject.load_verified_units(ledger, control)) == {672}


@pytest.mark.parametrize("fault", ["duplicate", "marker-loss", "partial"])
def test_component_ledger_refuses_duplicate_marker_loss_or_partial_export(
    tmp_path: Path, fault: str
) -> None:
    _, _, _, control = _control(tmp_path)
    output = tmp_path / "prior"
    item = _unit(output, control, 672)
    ledger = output / "verified-ledger.json"
    units = [item, item] if fault == "duplicate" else [item]
    export_json(ledger, {**subject.empty_ledger(control), "units": units})
    if fault == "marker-loss":
        (output / "units/fixture-672.json.complete").unlink()
    elif fault == "partial":
        (output / "units/fixture-672.json.sha256").unlink()
    with pytest.raises(subject.Step3Refusal):
        subject.load_verified_units(ledger, control)


@pytest.mark.parametrize("extra", ["label", "q_values", "action_trace"])
def test_component_ledger_refuses_rehashed_extra_wrapper_payload(
    tmp_path: Path, extra: str
) -> None:
    _, _, _, control = _control(tmp_path)
    output = tmp_path / "prior"
    item = _unit(output, control, 672)
    path = output / item["path"]
    payload = json.loads(path.read_text())
    payload[extra] = "forbidden"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    exported = export_json(path, payload)
    item.update(sha256=exported["sha256"], size=exported["size"])
    ledger = output / "verified-ledger.json"
    export_json(ledger, {**subject.empty_ledger(control), "units": [item]})
    with pytest.raises(subject.Step3Refusal, match="wrapper differs"):
        subject.load_verified_units(ledger, control)


def test_component_ledger_refuses_rehashed_foreign_nested_source(tmp_path: Path) -> None:
    _, _, _, control = _control(tmp_path)
    output = tmp_path / "prior"
    item = _unit(output, control, 672)
    path = output / item["path"]
    payload = json.loads(path.read_text())
    payload["record"]["source_provenance"]["source_sha256"] = "f" * 64
    payload["record"]["operational_payload_sha256"] = (
        f5_timing_supplement.operational_payload_sha256(payload["record"])
    )
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    exported = export_json(path, payload)
    item.update(sha256=exported["sha256"], size=exported["size"])
    ledger = output / "verified-ledger.json"
    export_json(ledger, {**subject.empty_ledger(control), "units": [item]})
    with pytest.raises(subject.Step3Refusal, match="source or fixture identity"):
        subject.load_verified_units(ledger, control)


class _InlineExecutor:
    def __init__(self, **_: Any) -> None:
        pass

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def map(self, function: Any, rows: Any, checkpoints: Any) -> list[Any]:
        return [
            function(row, checkpoint) for row, checkpoint in zip(rows, checkpoints, strict=True)
        ]


def _measure_control(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    control_path, fixtures, inputs, control = _control(tmp_path)
    qualification = {
        "status": "PASS",
        "source_sha256": control["source_sha256"],
        "fixture_descriptor_sha256": control["fixture_descriptor_sha256"],
        "native_cuda_cpu_parity": "PASS",
    }
    for suffix in ("", ".sha256", ".complete"):
        Path(str(inputs / "qualification.json") + suffix).unlink()
    export_json(inputs / "qualification.json", qualification)
    control["qualification_sha256"] = digest(inputs / "qualification.json")
    control["input_hashes"]["qualification.json"] = {
        "sha256": control["qualification_sha256"],
        "size": (inputs / "qualification.json").stat().st_size,
    }
    control["execution_envelope"] = {
        "host_workers": 6,
        "threads_per_worker": 1,
        "forecast_status": "QUALIFIED_COMPLETE",
        "unknown_costs": [],
        "measured_total_seconds": 1.0,
        "base_timeout_minutes": 60,
        "effective_timeout_minutes": 120,
    }
    control["input_hashes"]["external-acceptance.json"] = {"sha256": "0" * 64, "size": 1}
    acceptance = {
        "status": "ACCEPTED",
        "source_sha256": control["source_sha256"],
        "fixture_descriptor_sha256": control["fixture_descriptor_sha256"],
        "acceptance_target_sha256": subject._acceptance_target_sha256(control),
        "execution_envelope_sha256": subject._canonical_sha(control["execution_envelope"]),
        "qualification_sha256": control["qualification_sha256"],
    }
    export_json(inputs / "external-acceptance.json", acceptance)
    control["input_hashes"]["external-acceptance.json"] = {
        "sha256": digest(inputs / "external-acceptance.json"),
        "size": (inputs / "external-acceptance.json").stat().st_size,
    }
    for suffix in ("", ".sha256", ".complete"):
        Path(str(control_path) + suffix).unlink()
    export_json(control_path, control)
    return control_path, fixtures, inputs, control


def test_measure_carries_absent_through_two_fresh_outputs_without_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_path, fixtures, inputs, control = _measure_control(tmp_path)
    prior = tmp_path / "prior"
    first = _unit(prior, control, 672)
    prior_ledger = prior / "verified-ledger.json"
    export_json(prior_ledger, {**subject.empty_ledger(control), "units": [first]})
    calls: list[int] = []

    def fake_measure(row: dict[str, Any], _: Path) -> dict[str, Any]:
        calls.append(row["allocation_index"])
        return _record(row["allocation_index"])

    monkeypatch.setattr(subject.concurrent.futures, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        subject,
        "_qualification",
        lambda *args, **kwargs: (
            {"status": "PASS"},
            {"platform": "Linux", "os_kernel": {"release": "6.6"}},
        ),
    )
    monkeypatch.setattr(f5_environment, "receipt_sha256", lambda _: "d" * 64)
    monkeypatch.setattr(f5_timing_supplement, "measure_fixture", fake_measure)
    first_output = tmp_path / "attempt-one"
    subject.run(
        control_path=control_path,
        fixtures=fixtures,
        inputs=inputs,
        output=first_output,
        ledger=prior_ledger,
        mode="measure",
    )
    assert 672 not in calls and set(calls) == set(range(673, 684))
    successor = first_output / "verified-ledger-final.json"
    assert set(subject.load_verified_units(successor, control)) == set(range(672, 684))
    monkeypatch.setattr(
        f5_timing_supplement, "measure_fixture", lambda *_: pytest.fail("recomputed")
    )
    second_output = tmp_path / "attempt-two"
    subject.run(
        control_path=control_path,
        fixtures=fixtures,
        inputs=inputs,
        output=second_output,
        ledger=successor,
        mode="measure",
    )
    assert set(
        subject.load_verified_units(second_output / "verified-ledger-final.json", control)
    ) == set(range(672, 684))
