"""Synthetic operational metadata proofs for append-only F5 timing accounting."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from research.bl_ppo.exports import export_json
from research.bl_residual import f5_timing_resume as subject
from tests.test_bl_residual_f5_environment import _receipt
from tests.test_bl_residual_f5_timing_resume import _binding, _calibration, _fixture


def _save(root: Path, name: str, value: Any) -> dict[str, Any]:
    path = root / name
    export_json(path, value)
    return subject.metadata_reference(path)


def _export_entry(root: Path, path: str) -> dict[str, Any]:
    item = root / path
    return {"path": path, "size": item.stat().st_size, "sha256": subject.verify_export(item)}


def _publish(root, binding, key, payload, **kwargs):
    from research.bl_residual import f5_environment
    from tests.test_bl_residual_f5_timing_resume import _attested

    if "environment_before" not in payload:
        payload = _attested(root, payload)
    else:
        raw = _receipt()
        for name in ("environment_before", "environment_after"):
            link = payload[name]
            path = root / link["path"]
            if not path.exists():
                export_json(path, raw)
            assert path.stat().st_size == link["size"]
            assert subject.verify_export(path) == link["sha256"]
        payload = {
            **payload,
            "kernel_provenance": {
                "assignment_raw_sha256": f5_environment.receipt_sha256(raw),
                "os_kernel": dict(raw["os_kernel"]),
                "platform": raw["platform"],
                "gpu_uuid": raw["gpu_uuid"],
                "before": payload["environment_before"],
                "after": payload["environment_after"],
            },
        }
    return subject.publish_unit(root, binding, key, payload, **kwargs)


def _setup(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    binding = _binding()
    environment = _save(root, "environment-reference.json", _receipt())
    before = {
        "path": "environment/before.json",
        "size": environment["size"],
        "sha256": environment["sha256"],
    }
    entries = []
    for index in range(672, 684):
        payload = {
            **_fixture(index),
            "timing_attempt_id": "attempt-0" if index < 676 else "attempt-1",
            "environment_before": before,
            "environment_after": {
                "path": f"environment/after-fixture-{index}.json",
                "size": environment["size"],
                "sha256": environment["sha256"],
            },
        }
        entries.append(_publish(root, binding, index, payload))
    entries.append(
        _publish(
            root,
            binding,
            subject.CALIBRATION_KEY,
            {
                **_calibration(),
                "timing_attempt_id": "attempt-1",
                "environment_before": before,
                "environment_after": {
                    "path": "environment/after-calibration.json",
                    "size": environment["size"],
                    "sha256": environment["sha256"],
                },
            },
        )
    )
    subject.build_ledger(root, binding, entries)
    return root / "verified-ledger.json", environment, binding


def _attempt(
    root: Path, index: int, binding: dict[str, str], environment: dict[str, Any]
) -> dict[str, Any]:
    name = f"attempt-{index}"
    control = _save(
        root,
        f"{name}-control.json",
        {
            "identity": {
                "source_sha256": binding["source_sha256"],
                "environment_sha256": binding["environment_sha256"],
                "allocation_sha256": binding["allocation"],
            },
            "timing_input_sha256": binding["input_sha256"],
            "timing_attempt_id": name,
        },
    )
    manifest = _save(
        root,
        f"{name}-manifest.json",
        {
            "job_id": name,
            "commit": "e" * 40,
            "inputs": [{"dest": "control.json", "sha256": control["sha256"]}],
        },
    )
    # Attempt zero contributes four one-second fixtures; attempt one contributes
    # eight one-second fixtures and the entire two-second coupled calibration.
    useful = 4 if index == 0 else 10
    unit_paths = [
        f"timing-units/fixture-{item}.json"
        for item in (range(672, 676) if index == 0 else range(676, 684))
    ]
    if index == 1:
        unit_paths.append("timing-units/calibration.json")
    exports = [_export_entry(root, entry) for entry in unit_paths]
    exports.extend(
        [
            {
                "path": "environment/before.json",
                "size": environment["size"],
                "sha256": environment["sha256"],
            },
            {
                "path": "environment/after.json",
                "size": environment["size"],
                "sha256": environment["sha256"],
            },
        ]
    )
    for item in range(672, 684):
        if (index == 0 and item < 676) or (index == 1 and item >= 676):
            exports.append(
                {
                    "path": f"environment/after-fixture-{item}.json",
                    "size": environment["size"],
                    "sha256": environment["sha256"],
                }
            )
    if index == 1:
        exports.append(
            {
                "path": "environment/after-calibration.json",
                "size": environment["size"],
                "sha256": environment["sha256"],
            }
        )
    before_export = next(item for item in exports if item["path"] == "environment/before.json")
    after_export = next(item for item in exports if item["path"] == "environment/after.json")
    return {
        "attempt_index": index,
        "attempt_id": name,
        "manifest": manifest,
        "control": control,
        "broker_receipt": _save(
            root,
            f"{name}-receipt.json",
            {
                "job_id": name,
                "manifest_sha256": manifest["sha256"],
                "commit": "e" * 40,
                "status": "INCOMPLETE" if index == 0 else "PASS",
                "terminal": None if index == 0 else {"returncode": 0},
                "stop_verified": True,
                "cleanup_error": None,
                "elapsed_seconds": useful + 6.0,
                "exports": exports,
            },
        ),
        "upload_measurement": _save(
            root,
            f"{name}-upload.json",
            {
                "setup_seconds": 1.0,
                "transfer_seconds": 2.0,
            },
        ),
        "accounting_measurement": _save(
            root,
            f"{name}-accounting.json",
            {
                "identity": binding,
                "attempt_id": name,
                "setup_seconds": 1.0,
                "io_seconds": 2.0,
                "imbalance_seconds": 3.0,
            },
        ),
        "environment_before": environment,
        "environment_after": environment,
        "environment_before_export": before_export,
        "environment_after_export": after_export,
    }


def _history(
    root: Path, binding: dict[str, str], environment: dict[str, Any], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    previous = None
    for index, attempt in enumerate(attempts):
        previous = subject.append_accounting_history(
            root / f"history-{index}.json",
            binding,
            attempt,
            environment_reference=environment,
            previous_history=previous,
        )
    assert previous is not None
    return previous


def test_verified_two_attempt_history_counts_all_exposure_and_allows_incomplete_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    ledger, environment, binding = _setup(tmp_path)
    history = _history(
        tmp_path,
        binding,
        environment,
        [_attempt(tmp_path, i, binding, environment) for i in range(2)],
    )
    result = subject.reduce_ledger(ledger, binding, accounting_history=history)
    assert result["status"] == "COMPLETE_UNITS_PENDING_MAC_FORECAST"
    assert result["accounting_verified"] is True
    assert result["all_attempt_setup_io_imbalance_seconds"] == 12.0
    assert (
        sum(row["elapsed_seconds"] for row in result["accounting_evidence"]["measured_attempts"])
        == 26.0
    )
    assert result["accounting_evidence"]["measured_attempts"][0]["status"] == "INCOMPLETE"


def test_unbound_attempt_mappings_never_qualify_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    ledger, _, binding = _setup(tmp_path)
    result = subject.reduce_ledger(ledger, binding)
    assert result["status"] == "HOLD_UNVERIFIED_ATTEMPT_ACCOUNTING"
    assert result["archive_registered_conservative_hours"] is None
    assert result["all_attempt_setup_io_imbalance_seconds"] is None


def test_missing_explicit_imbalance_measurement_holds_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    ledger, environment, binding = _setup(tmp_path)
    attempts = [_attempt(tmp_path, i, binding, environment) for i in range(2)]
    attempts[0]["accounting_measurement"] = None
    history = _history(tmp_path, binding, environment, attempts)
    result = subject.reduce_ledger(ledger, binding, accounting_history=history)
    assert result["status"] == "HOLD_UNVERIFIED_ATTEMPT_ACCOUNTING"
    assert result["accounting_evidence"]["missing_measurements"] == ["attempt-0"]


@pytest.mark.parametrize(
    "kind",
    [
        "omitted",
        "duplicate",
        "noncontiguous",
        "missing-stop",
        "hash",
        "io",
        "conservation",
        "foreign-control",
        "environment",
        "missing-unit-export",
        "duplicate-export",
        "tampered-unit-export",
        "missing-before-attestation",
    ],
)
def test_reducer_refuses_unverified_or_incomplete_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    ledger, environment, binding = _setup(tmp_path)
    attempts = [_attempt(tmp_path, i, binding, environment) for i in range(2)]
    if kind == "omitted":
        attempts = attempts[:1]
    elif kind == "duplicate":
        attempts[1]["attempt_id"] = "attempt-0"
    elif kind == "noncontiguous":
        attempts[1]["attempt_index"] = 2
    elif kind == "hash":
        attempts[0]["upload_measurement"]["sha256"] = "0" * 64
    elif kind in {"io", "conservation"}:
        attempts[0]["accounting_measurement"] = _save(
            tmp_path,
            "bad-accounting.json",
            {
                "identity": binding,
                "attempt_id": "attempt-0",
                "setup_seconds": 1.0,
                "io_seconds": 1.0 if kind == "io" else 2.0,
                "imbalance_seconds": 4.0,
            },
        )
    elif kind == "missing-stop":
        attempts[0]["broker_receipt"] = _save(
            tmp_path,
            "bad-receipt.json",
            {
                "job_id": "attempt-0",
                "manifest_sha256": attempts[0]["manifest"]["sha256"],
                "commit": "e" * 40,
                "status": "INCOMPLETE",
                "stop_verified": False,
                "cleanup_error": None,
                "elapsed_seconds": 10,
            },
        )
    elif kind == "foreign-control":
        attempts[0]["control"] = attempts[1]["control"]
    elif kind == "environment":
        foreign = _receipt()
        foreign["cpu_count"] += 1
        attempts[0]["environment_before"] = _save(tmp_path, "bad-environment.json", foreign)
    elif kind in {
        "missing-unit-export",
        "duplicate-export",
        "tampered-unit-export",
        "missing-before-attestation",
    }:
        receipt = subject._bound_metadata(attempts[0]["broker_receipt"])
        exports = receipt["exports"]
        if kind == "missing-unit-export":
            receipt["exports"] = [
                row for row in exports if row["path"] != "timing-units/fixture-672.json"
            ]
        elif kind == "duplicate-export":
            receipt["exports"] = [*exports, dict(exports[0])]
        elif kind == "tampered-unit-export":
            receipt["exports"] = [
                (
                    {**row, "sha256": "0" * 64}
                    if row["path"] == "timing-units/fixture-672.json"
                    else row
                )
                for row in exports
            ]
        else:
            receipt["exports"] = [
                row for row in exports if row["path"] != "environment/before.json"
            ]
        attempts[0]["broker_receipt"] = _save(tmp_path, f"{kind}-receipt.json", receipt)
    with pytest.raises(ValueError):
        history = _history(tmp_path, binding, environment, attempts)
        subject.reduce_ledger(ledger, binding, accounting_history=history)


@pytest.mark.parametrize("field", ["parent_id", "group", "group_index", "seed"])
def test_scanner_refuses_fixture_metadata_drift_before_skip(field: str, tmp_path: Path) -> None:
    payload = _fixture(672)
    payload[field] = "foreign" if field != "seed" else int(payload[field]) + 1
    with pytest.raises(subject.ResumeRefusal, match="allocation identity"):
        _publish(tmp_path, _binding(), 672, payload)


def test_sealed_attempt_keeps_overlapping_export_downloads_diagnostic_only(tmp_path: Path) -> None:
    binding = _binding()
    exports = tmp_path / "exports"
    export_json(exports / "environment/before.json", _receipt())
    export_json(exports / "environment/after.json", _receipt())
    before = _export_entry(exports, "environment/before.json")
    export_json(exports / "environment/after-fixture-672.json", _receipt())
    after = _export_entry(exports, "environment/after-fixture-672.json")
    unit = _publish(
        exports,
        binding,
        672,
        {
            **_fixture(672),
            "timing_attempt_id": "attempt-0",
            "environment_before": before,
            "environment_after": after,
            "elapsed_seconds": 10.0,
        },
    )
    manifest = _save(tmp_path, "manifest.json", {"job_id": "attempt-0"})
    control = _save(tmp_path, "control.json", {"timing_attempt_id": "attempt-0"})
    upload = _save(tmp_path, "upload.json", {"setup_seconds": 1.0, "transfer_seconds": 2.0})
    receipt = _save(
        tmp_path,
        "receipt.json",
        {
            "job_id": "attempt-0",
            "status": "PASS",
            "elapsed_seconds": 14.0,
            "exports": [
                {field: unit[field] for field in ("path", "size", "sha256")},
                before,
                after,
                _export_entry(exports, "environment/after.json"),
            ],
            "transfers": [{"direction": "download", "elapsed_seconds": 5.0}],
        },
    )
    sealed = subject.seal_attempt_accounting(
        tmp_path / "sealed",
        binding,
        manifest=Path(manifest["path"]),
        broker_receipt=Path(receipt["path"]),
        upload_measurement=Path(upload["path"]),
        control=Path(control["path"]),
        exports_root=exports,
        attempt_index=0,
    )
    measurement = subject._bound_metadata(sealed["accounting_measurement"], immutable=True)
    assert measurement["io_seconds"] == 2.0
    assert measurement["imbalance_seconds"] == 1.0
    assert measurement["observed_incremental_export_download_seconds"] == 5.0
    assert measurement["incremental_export_downloads_may_overlap_useful_runtime"] is True


def test_reducer_refuses_earlier_unit_replaced_by_later_attempt_ledger_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later ledger cannot hide a complete unit already receipt-bound earlier."""
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    ledger, environment, binding = _setup(tmp_path)
    attempts = [_attempt(tmp_path, index, binding, environment) for index in range(2)]
    unit_path = tmp_path / "timing-units/fixture-672.json"
    raw = subject._bound_metadata(subject.metadata_reference(unit_path), immutable=True)
    raw["payload"]["timing_attempt_id"] = "attempt-1"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(unit_path) + suffix).unlink()
    export_json(unit_path, raw)
    replacement = _export_entry(tmp_path, "timing-units/fixture-672.json")
    receipt = subject._bound_metadata(attempts[1]["broker_receipt"])
    receipt["exports"].append(replacement)
    attempts[1]["broker_receipt"] = _save(tmp_path, "attempt-1-replaced-receipt.json", receipt)
    for suffix in ("", ".sha256", ".complete"):
        Path(str(ledger) + suffix).unlink()
    entries = [
        _export_entry(tmp_path, f"timing-units/fixture-{index}.json") | {"key": index}
        for index in range(672, 684)
    ]
    entries.append(
        _export_entry(tmp_path, "timing-units/calibration.json") | {"key": subject.CALIBRATION_KEY}
    )
    subject.build_ledger(tmp_path, binding, entries)
    history = _history(tmp_path, binding, environment, attempts)
    with pytest.raises(subject.ResumeRefusal, match="differs from verified final ledger"):
        subject.reduce_ledger(ledger, binding, accounting_history=history)


def test_frozen_history_refuses_count_drift_and_prefix_rewrite(tmp_path: Path) -> None:
    _, environment, binding = _setup(tmp_path)
    history = _history(
        tmp_path,
        binding,
        environment,
        [_attempt(tmp_path, i, binding, environment) for i in range(2)],
    )
    changed = copy.deepcopy(history)
    changed["attempt_count"] = 1
    with pytest.raises(subject.ResumeRefusal):
        subject.verify_accounting_history(changed, binding)
    raw = subject._bound_metadata({k: v for k, v in history.items() if k != "attempt_count"})
    raw["attempts"][0]["attempt_id"] = "rewritten"
    changed = {**_save(tmp_path, "rewritten-history.json", raw), "attempt_count": 2}
    with pytest.raises(subject.ResumeRefusal, match="rewrites"):
        subject.verify_accounting_history(changed, binding)
    with pytest.raises(FileExistsError):
        subject.append_accounting_history(
            Path(history["path"]),
            binding,
            {"attempt_index": 0, "attempt_id": "new"},
            environment_reference=environment,
        )
