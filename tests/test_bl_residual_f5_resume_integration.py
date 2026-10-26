"""Offline integration coverage for F5's real completed-unit qualification loop.

The expensive CUDA/model/world seams are replaced at their outer boundaries.
Everything below those boundaries is the production ``qualification.run`` loop:
it publishes immutable units, scans verified ledgers, and selects the exact
fixture or coupled-calibration work that remains.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research.bl_ppo.exports import export_json
from research.bl_residual import f5_qualification as qualification
from research.bl_residual import f5_timing_resume as resume
from research.bl_residual.f5_fit import F5Identity
from research.bl_residual.f5_r10a_annex import allocate


def _rows() -> list[dict[str, Any]]:
    return [dict(row) for row in allocate(0) if row["timing_only"] is True]


def _identity() -> F5Identity:
    return F5Identity("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)


def _binding(*, canary: bool = False) -> dict[str, str]:
    identity = _identity()
    return resume.identity(
        namespace="bl-residual-f5-timing-recovery-20261026" + ("/canary" if canary else ""),
        allocation=identity.allocation_sha256,
        source_sha256=identity.source_sha256,
        environment_sha256=identity.environment_sha256,
        input_sha256="f" * 64,
    )


def _calibration() -> dict[str, Any]:
    return {
        "elapsed_seconds": 3.0,
        "arms": {
            arm: {
                "serialized_continuation_exact": True,
                "full_seconds": 1.0,
                "serialized_continuation_seconds": 1.0,
            }
            for arm in ("C", "W")
        },
    }


def _install_outer_seams(
    monkeypatch: pytest.MonkeyPatch, executed: list[int], calibrations: list[str]
) -> None:
    """Keep admission/loop/export/scanner real while removing CUDA and worlds."""
    from research.bl_residual import f5_calibration, f5_job
    from src.model import ego2s_network

    monkeypatch.setattr(qualification, "_require_cuda", lambda device: None)
    monkeypatch.setattr(qualification, "load_fixtures", lambda path: (_rows(), "1" * 64))
    monkeypatch.setattr(qualification, "_verify_uploaded_checkpoint_pool", lambda value: {})
    monkeypatch.setattr(
        qualification, "_native_observation", lambda seed: ({}, np.ones(6, dtype=bool))
    )
    monkeypatch.setattr(qualification, "_select_cuda", lambda model, obs, mask: (np.zeros(6), 0))
    monkeypatch.setattr(qualification, "_cpu_action", lambda model, obs, mask: (np.zeros(6), 0))

    def source_timing(row: dict[str, Any], checkpoint: Path, progress: object = None):
        index = int(row["allocation_index"])
        executed.append(index)
        return (
            {
                "t_root": 0.1,
                "t_branch": 0.1,
                "t_nat": 0.1,
                "full_h300": True,
                "full_h5000_natural": True,
                "long_body": True,
                "watch_fixture_frames": 14_400,
                "watch_seconds": 0.1,
            },
            {},
        )

    monkeypatch.setattr(qualification, "_full_source_timing", source_timing)
    monkeypatch.setattr(qualification, "_producer_exports", lambda *args, **kwargs: [])
    monkeypatch.setattr(qualification, "digest", lambda path: f5_job.B8_SHA256)
    monkeypatch.setattr(ego2s_network, "load_ego2s_checkpoint", lambda *args: _FakeModel())
    monkeypatch.setattr(ego2s_network, "spec_for", lambda model: "ego2s-bv8")
    monkeypatch.setattr(f5_job, "_control", lambda inputs: _control())
    monkeypatch.setattr(f5_job, "_source_admission", lambda control: None)
    monkeypatch.setattr(f5_job, "_identity", lambda control: _identity())
    monkeypatch.setattr(f5_job, "_environment_admission", lambda control: {"assignment": "offline"})
    monkeypatch.setattr(f5_job, "_environment_evidence", lambda control, actual: {"offline": True})
    monkeypatch.setattr(f5_job, "_assignment_boundary", lambda actual: {})

    def export_boundary(reference: dict, output: Path, name: str) -> dict:
        path = output / "environment" / name
        exported = export_json(path, reference)
        return {**exported, "path": path.relative_to(output).as_posix()}

    monkeypatch.setattr(qualification, "_export_assignment_boundary", export_boundary)

    def fake_calibration(*args: object, **kwargs: object) -> dict[str, Any]:
        calibrations.append("whole-cw")
        return _calibration()

    monkeypatch.setattr(f5_calibration, "run", fake_calibration)


class _FakeModel:
    def eval(self) -> "_FakeModel":
        return self

    def cuda(self) -> "_FakeModel":
        return self


def _control() -> dict[str, Any]:
    return {
        "qualification_fixture_sha256": "1" * 64,
        "qualification_b8_sha256": (
            "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
        ),
        "qualification_checkpoint_pool": {},
        "timing_input_sha256": "f" * 64,
        "timing_attempt_id": "offline-timing-attempt",
        "source_hashes": {"research/bl_residual/f5_qualification.py": "0" * 64},
        "ledger_sha256": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    }


def _inputs(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "b8.pth").write_bytes(b"offline-pinned-checkpoint")
    return inputs


def _empty_ledger(root: Path, *, canary: bool = False) -> Path:
    path = root / "verified-ledger.json"
    export_json(path, resume.empty_ledger(_binding(canary=canary)))
    return path


def _semantic_records(
    report: dict[str, Any],
) -> list[tuple[int, str, tuple[int, ...], bool, bool, bool]]:
    return [
        (
            row["allocation_index"],
            row["parity"],
            tuple(row["q_shape"]),
            row["full_h300"],
            row["full_h5000_natural"],
            row["long_body"],
        )
        for row in report["samples"]
    ]


def test_run_exports_prefix_skips_exact_units_restarts_coupled_calibration_and_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the actual producer/export/scanner/restart loop without CUDA."""
    executed: list[int] = []
    calibrations: list[str] = []
    _install_outer_seams(monkeypatch, executed, calibrations)
    inputs = _inputs(tmp_path)

    # The uninterrupted reference proves resumed timing-free coverage semantics.
    reference_ledger = _empty_ledger(tmp_path / "reference-inputs")
    uninterrupted = qualification.run(
        inputs, inputs / "unused-fixtures.json", tmp_path / "uninterrupted", ledger=reference_ledger
    )
    assert executed == list(range(672, 684))
    assert calibrations == ["whole-cw"]

    # Inject failure only after the third immutable fixture record exists.
    original_publish = resume.publish_unit
    failed_root = tmp_path / "failed-after-three"

    def fail_after_three(*args: object, **kwargs: object) -> dict[str, Any]:
        published = original_publish(*args, **kwargs)
        if published["key"] == 674:
            raise RuntimeError("synthetic interruption after third fixture export")
        return published

    monkeypatch.setattr(resume, "publish_unit", fail_after_three)
    with pytest.raises(RuntimeError, match="third fixture"):
        qualification.run(
            inputs,
            inputs / "unused-fixtures.json",
            failed_root,
            ledger=_empty_ledger(tmp_path / "failed-inputs"),
        )
    prefix = [
        {"key": key, "path": f"timing-units/fixture-{key}.json", "sha256": "", "size": 0}
        for key in range(672, 675)
    ]
    # Build from the real exported paths, then use that verified prefix as the new attempt's ledger.
    for entry in prefix:
        payload = failed_root / entry["path"]
        entry["sha256"] = resume.verify_export(payload)
        entry["size"] = payload.stat().st_size
        record = json.loads(payload.read_text())["payload"]
        for field in ("environment_before", "environment_after"):
            attestation = failed_root / record[field]["path"]
            assert resume.verify_export(attestation) == record[field]["sha256"]
            assert attestation.stat().st_size == record[field]["size"]
    assert not (failed_root / "environment/after.json").exists()
    resume.build_ledger(failed_root, _binding(), prefix)
    monkeypatch.setattr(resume, "publish_unit", original_publish)

    executed.clear()
    calibrations.clear()
    resumed = qualification.run(
        inputs,
        inputs / "unused-fixtures.json",
        tmp_path / "resume-fixtures",
        ledger=failed_root / "verified-ledger.json",
    )
    assert executed == list(range(675, 684))
    assert calibrations == ["whole-cw"]
    assert _semantic_records(resumed) == _semantic_records(uninterrupted)

    # A failed coupled C/W calibration leaves all fixture units reusable,
    # but no partial calibration.
    fixture_attempt = tmp_path / "fixture-attempt"
    shutil.copytree(failed_root / "timing-units", fixture_attempt / "timing-units")
    shutil.copytree(
        tmp_path / "resume-fixtures/timing-units",
        fixture_attempt / "timing-units",
        dirs_exist_ok=True,
    )
    fixture_entries = []
    for key in range(672, 684):
        path = fixture_attempt / "timing-units" / f"fixture-{key}.json"
        fixture_entries.append(
            {
                "key": key,
                "path": path.relative_to(fixture_attempt).as_posix(),
                "sha256": resume.verify_export(path),
                "size": path.stat().st_size,
            }
        )
    resume.build_ledger(fixture_attempt, _binding(), fixture_entries)

    from research.bl_residual import f5_calibration

    def fail_calibration(*args: object, **kwargs: object) -> dict[str, Any]:
        calibrations.append("partial-cw-failed")
        raise RuntimeError("synthetic partial coupled calibration")

    monkeypatch.setattr(f5_calibration, "run", fail_calibration)
    executed.clear()
    with pytest.raises(RuntimeError, match="partial coupled"):
        qualification.run(
            inputs,
            inputs / "unused-fixtures.json",
            tmp_path / "calibration-failed",
            ledger=fixture_attempt / "verified-ledger.json",
        )
    assert executed == []
    assert not (tmp_path / "calibration-failed/timing-units/calibration.json").exists()

    def complete_calibration(*args: object, **kwargs: object) -> dict[str, Any]:
        calibrations.append("whole-cw-restarted")
        return _calibration()

    monkeypatch.setattr(f5_calibration, "run", complete_calibration)
    calibration_complete = tmp_path / "calibration-complete"
    qualification.run(
        inputs,
        inputs / "unused-fixtures.json",
        calibration_complete,
        ledger=fixture_attempt / "verified-ledger.json",
    )
    assert calibrations[-1] == "whole-cw-restarted"

    # Assemble a new uploaded ledger from immutable completed paths;
    # a completed C/W unit is skipped whole.
    combined = tmp_path / "combined-ledger"
    shutil.copytree(fixture_attempt / "timing-units", combined / "timing-units")
    shutil.copy2(
        calibration_complete / "timing-units/calibration.json",
        combined / "timing-units/calibration.json",
    )
    shutil.copy2(
        calibration_complete / "timing-units/calibration.json.sha256",
        combined / "timing-units/calibration.json.sha256",
    )
    shutil.copy2(
        calibration_complete / "timing-units/calibration.json.complete",
        combined / "timing-units/calibration.json.complete",
    )
    all_entries = list(fixture_entries)
    calibration_path = combined / "timing-units/calibration.json"
    all_entries.append(
        {
            "key": resume.CALIBRATION_KEY,
            "path": "timing-units/calibration.json",
            "sha256": resume.verify_export(calibration_path),
            "size": calibration_path.stat().st_size,
        }
    )
    resume.build_ledger(combined, _binding(), all_entries)
    executed.clear()
    calibrations.clear()
    qualification.run(
        inputs,
        inputs / "unused-fixtures.json",
        tmp_path / "completed-skip",
        ledger=combined / "verified-ledger.json",
    )
    assert executed == [] and calibrations == []

    # The actual CANARY_ONLY path starts from its empty ledger, publishes 672,
    # scans it, skips it, then exports 673.
    canary_ledger = _empty_ledger(tmp_path / "canary-inputs")
    canary = qualification.run(
        inputs,
        inputs / "unused-fixtures.json",
        tmp_path / "canary",
        ledger=canary_ledger,
        transport_only=True,
        canary_tiny_workload=True,
    )
    assert [entry["key"] for entry in canary["canary_only"]] == [672, 673]
    assert (tmp_path / "canary/timing-canary/verified-ledger.json.complete").is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        "namespace",
        "allocation",
        "source",
        "environment",
        "input",
        "duplicate",
        "hash",
        "size",
        "marker",
        "schema",
    ],
)
def test_scanner_refuses_foreign_or_incomplete_completed_unit_contract(
    tmp_path: Path, mutation: str
) -> None:
    """All refusal classes are checked by the same scanner used by ``run``."""
    binding = _binding()
    payload = {
        **_rows()[0],
        "allocation_index": 672,
        "elapsed_seconds": 1.0,
        "parity": "PASS",
        "q_shape": [1, 6],
        "full_h300": True,
        "long_body": True,
        "full_h5000_natural": True,
        "watch_fixture_frames": 14_400,
    }
    entry = resume.publish_unit(tmp_path, binding, 672, payload)
    ledger = resume.empty_ledger(binding)
    ledger["units"] = [copy.deepcopy(entry)]
    if mutation in {"namespace", "allocation", "source", "environment", "input"}:
        name = {
            "namespace": "namespace",
            "allocation": "allocation",
            "source": "source_sha256",
            "environment": "environment_sha256",
            "input": "input_sha256",
        }[mutation]
        ledger["identity"][name] = "9" * 64 if name != "namespace" else "foreign"
    elif mutation == "duplicate":
        ledger["units"].append(copy.deepcopy(entry))
    elif mutation == "hash":
        ledger["units"][0]["sha256"] = "0" * 64
    elif mutation == "size":
        ledger["units"][0]["size"] += 1
    elif mutation == "marker":
        (tmp_path / "timing-units/fixture-672.json.complete").unlink()
    elif mutation == "schema":
        ledger["schema"] = "foreign-schema"
    ledger_path = tmp_path / "bad-ledger.json"
    export_json(ledger_path, ledger)
    with pytest.raises(resume.ResumeRefusal):
        resume.load_ledger(ledger_path, binding)
