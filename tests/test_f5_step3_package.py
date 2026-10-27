"""Offline package guards do not imply a measured runtime forecast."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, export_bytes, export_json
from research.bl_residual import f5_step3_driver as driver
from research.bl_residual import f5_step3_package as package
from research.bl_residual.f5_environment_reference import CAPTURE05_SIZE
from research.colab_cli import canary, runtime
from tests.test_f5_step3_driver import _control


def test_export_envelope_fits_the_known_raw_native_environment_size() -> None:
    assert package.MAX_EXPORT_BYTES == 64 * 1024 * 1024
    assert package.MAX_EXPORT_BYTES > CAPTURE05_SIZE


def test_finite_envelope_doubles_once_and_preserves_archive_guards() -> None:
    value = package.envelope()
    assert value["effective_timeout_minutes"] == 2 * value["base_timeout_minutes"]
    assert value["aggregate_effective_timeout_minutes"] == 3 * value["effective_timeout_minutes"]
    assert value["global_completion_rounds_maximum"] == 2
    assert value["archive_stage_base_hours"] == 96
    assert value["archive_stage_effective_hours"] == 192
    assert value["archive_shards"] == 16
    assert value["unknown_costs"]
    assert value["forecast_status"].startswith("HOLD_")
    assert (
        value["family_canary_effective_timeout_minutes"]
        == 2 * value["family_canary_base_timeout_minutes"]
    )
    assert value["maximum_family_canaries"] == 3
    assert value["aggregate_effective_with_qualification_and_family_canaries_minutes"] == 480


def test_package_refuses_existing_destination_before_reading_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh"):
        package.build(tmp_path)


def test_input_reference_refuses_symlink(tmp_path: Path) -> None:
    original = tmp_path / "input.json"
    original.write_text("{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="unsafe"):
        package.reference(alias)


def _target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provisional: bool = False,
    stale_qualification: bool = False,
) -> Path:
    """Synthetic admission evidence only; no production authority is generated."""
    path, fixtures, inputs, control = _control(tmp_path)
    export_json(inputs / "control.json", control)
    export_bytes(inputs / "timing-fixtures.json", fixtures.read_bytes())
    manifest = {
        "job_id": "synthetic-measure",
        "commit": "a" * 40,
        "script": "research/bl_residual/f5_step3.sh",
        "device": "cuda:0",
        "env": {"F5_STEP3_MODE": "measure"},
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 60},
        "timeout_minutes": 120,
        "max_export_bytes": package.MAX_EXPORT_BYTES,
        "budget": {
            "unit_cap": 8.0,
            "rate_units_per_hour": 2.0,
            "reserve_units": 0.0,
            "other_assignments": {},
        },
        "output_dir": str(tmp_path / "original-output"),
        "source_paths": sorted(control["source_hashes"]),
        "required_exports": ["qualification.json", "supplement-summary.json"],
        "inputs": [
            package.reference(inputs / name, name)
            for name in ("control.json", "b8.pth", "timing-fixtures.json")
        ],
        "resume": {
            "unit_kind": "fixture",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
    }
    export_json(tmp_path / "manifest.json", manifest)
    qualification = tmp_path / "qualified.json"
    export_json(
        qualification,
        {
            "status": "PASS",
            "source_sha256": "f" * 64 if stale_qualification else control["source_sha256"],
            "fixture_descriptor_sha256": control["fixture_descriptor_sha256"],
            "native_cuda_cpu_parity": "PASS",
        },
    )
    forecast = tmp_path / "forecast.json"
    export_json(
        forecast,
        (
            package.provisional_envelope()
            if provisional
            else {
                **package.envelope(),
                "forecast_status": "QUALIFIED_COMPLETE",
                "unknown_costs": [],
                "measured_total_seconds": 1200.0,
            }
        ),
    )
    monkeypatch.setattr(package, "verify", lambda *a, **k: {})
    target = tmp_path / "target"
    package.prepare_target(
        tmp_path,
        target,
        qualification=qualification,
        forecast=forecast,
        ledger=tmp_path / "verified-ledger.json",
    )
    return target


def _final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = _target(tmp_path, monkeypatch)
    authority = json.loads((target / "TARGET.json").read_text())
    acceptance = tmp_path / "external-acceptance.json"
    export_json(acceptance, {**authority, "status": "ACCEPTED"})
    output = tmp_path / "final"
    package.seal_final(target, output, acceptance=acceptance, attempt=1)
    return output


def test_provisional_target_retains_unknowns_and_binds_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path, monkeypatch, provisional=True)
    authority = json.loads((target / "TARGET.json").read_text())
    control = json.loads((target / "inputs/control.json").read_text())
    envelope = control["execution_envelope"]
    assert envelope["measured_total_seconds"] is None
    assert envelope["unknown_costs"] == list(driver.PROVISIONAL_UNKNOWN_COSTS)
    assert envelope["science_admission"] is False
    assert authority["ordering_disposition_sha256"] == driver.ORDERING_DISPOSITION_SHA256
    assert authority["execution_envelope_sha256"] == driver._canonical_sha(envelope)
    assert authority["acceptance_target_sha256"] == driver._acceptance_target_sha256(control)
    kwargs = dict(
        control_path=target / "inputs/control.json",
        fixtures=target / "inputs/timing-fixtures.json",
        inputs=target / "inputs",
        ledger=target / "inputs/verified-ledger.json",
        output=tmp_path / "unused",
        mode="dry-run",
    )
    with pytest.raises(driver.Step3Refusal, match="independent acceptance"):
        driver.run(**kwargs)
    acceptance = tmp_path / "external-acceptance.json"
    export_json(acceptance, {**authority, "status": "ACCEPTED"})
    final = tmp_path / "provisional-final"
    package.seal_final(target, final, acceptance=acceptance, attempt=1)
    package.verify_final(final)
    main = json.loads((final / "manifest.json").read_text())
    probe = json.loads((final / "family-canary-manifest.json").read_text())
    assert main["timeout_minutes"] == 120
    assert probe["timeout_minutes"] == 30
    assert canary.input_inventory(main) == canary.input_inventory(probe)
    canary.check(_broker_proof(final, tmp_path, monkeypatch), tmp_path)


def test_provisional_target_refuses_prior_source_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(driver.Step3Refusal, match="qualification input does not match"):
        _target(tmp_path, monkeypatch, provisional=True, stale_qualification=True)
    assert not (tmp_path / "target/TARGET.json").exists()


def test_provisional_envelope_refuses_mutated_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "ruling.md"
    changed.write_text("unaccepted change")
    monkeypatch.setattr(package, "ORDERING_DISPOSITION", changed)
    with pytest.raises(ValueError, match="disposition digest changed"):
        package.provisional_envelope()


def _broker_proof(output: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Exercise unchanged evidence/check with synthetic scanner/transfer observations."""
    probe = json.loads((output / "family-canary-manifest.json").read_text())
    main = json.loads((output / "manifest.json").read_text())
    monkeypatch.setattr(driver, "_qualification", lambda *a, **k: ({"status": "PASS"}, {}))
    driver.run(
        control_path=output / "inputs/control.json",
        fixtures=output / "inputs/timing-fixtures.json",
        inputs=output / "inputs",
        ledger=output / "inputs/verified-ledger.json",
        output=tmp_path / "remote/output",
        mode="family-canary",
    )
    inventory = runtime._exports(tmp_path / "remote", package.MAX_EXPORT_BYTES)
    broker = tmp_path / "broker"
    broker.mkdir()
    (broker / "exports").mkdir()
    for row in inventory:
        (broker / "exports" / row["path"]).write_bytes(
            (tmp_path / "remote/output" / row["path"]).read_bytes()
        )
    for name, value in (
        ("manifest.json", probe),
        ("hardware.json", {"hardware": {"kind": "L4"}}),
        (
            "upload-measurement.json",
            {"elapsed_seconds": 10, "transfer_seconds": 2, "source_delivery_seconds": 5},
        ),
    ):
        (broker / name).write_text(json.dumps(value))
    receipt = {
        "status": "PASS",
        "stop_verified": True,
        "failure_class": None,
        "commit": probe["commit"],
        "job_id": probe["job_id"],
        "generation": "synthetic",
        "manifest_sha256": digest(broker / "manifest.json"),
        "exports": inventory,
        "terminal": {
            "job_id": probe["job_id"],
            "generation": "synthetic",
            "returncode": 0,
            "status": "completed",
            "export_scanner": {"name": "runtime._exports", "inventory": inventory},
            "canary_observations": [
                {
                    "cuda_decisions": 1,
                    "device": "cuda:0",
                    "shape": [1, 6],
                    "package": "research/bl_residual",
                    "operation": "Tensor.argmax",
                    "caller": "research/bl_residual/f5_step3_driver.py",
                    "root": "/content/snake-cli/synthetic",
                }
            ],
        },
        "transfers": [
            {
                "direction": "upload",
                "status": "verified",
                "remote": "/content/synthetic/inputs/" + row["dest"],
                "size_bytes": Path(row["path"]).stat().st_size,
                "sha256": row["sha256"],
            }
            for row in probe["inputs"]
        ],
    }
    monkeypatch.setattr(canary, "executable_identity", lambda *a: {"synthetic": "identity"})
    monkeypatch.setattr(canary, "_entrypoint", lambda *a: "")
    receipt["canary_evidence"] = canary.evidence(probe, receipt, broker, tmp_path)
    (broker / "receipt.json").write_text(json.dumps(receipt))
    main["canary"] = {
        "receipt": {"path": str(broker / "receipt.json"), "sha256": digest(broker / "receipt.json")}
    }
    return main


def test_final_pair_exact_inventory_and_real_scanner_exports_admit_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _final(tmp_path, monkeypatch)
    assert package.verify_final(output)["attempt"] == 1
    main = json.loads((output / "manifest.json").read_text())
    probe = json.loads((output / "family-canary-manifest.json").read_text())
    assert canary.input_inventory(main) == canary.input_inventory(probe)
    assert main["required_exports"] == probe["required_exports"]
    assert {
        "control.json",
        "qualification.json",
        "external-acceptance.json",
        "verified-ledger.json",
    } <= {row["dest"] for row in main["inputs"]}
    canary.check(_broker_proof(output, tmp_path, monkeypatch), tmp_path)


@pytest.mark.parametrize("fault", ["ledger", "control", "acceptance", "qualification", "export"])
def test_final_canary_cannot_cover_changed_inputs_or_unproduced_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    output = _final(tmp_path, monkeypatch)
    main = _broker_proof(output, tmp_path, monkeypatch)
    if fault == "export":
        main["required_exports"].append("not-produced.json")
    else:
        dest = {
            "ledger": "verified-ledger.json",
            "control": "control.json",
            "acceptance": "external-acceptance.json",
            "qualification": "qualification.json",
        }[fault]
        item = next(row for row in main["inputs"] if row["dest"] == dest)
        item["sha256"] = "f" * 64
    with pytest.raises(canary.CanaryError, match="upload bytes|producer export"):
        canary.check(main, tmp_path)


def test_final_package_refuses_forged_acceptance_and_fourth_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path, monkeypatch)
    acceptance = tmp_path / "external-acceptance.json"
    export_json(acceptance, {"status": "ACCEPTED"})
    with pytest.raises(driver.Step3Refusal, match="independent acceptance"):
        package.seal_final(target, tmp_path / "bad", acceptance=acceptance, attempt=1)
    with pytest.raises(ValueError, match="two mechanical"):
        package.seal_final(target, tmp_path / "four", acceptance=acceptance, attempt=4)
