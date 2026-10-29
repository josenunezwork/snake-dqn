"""Native accounting-rate guard admission and scientific-fault precedence."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import capture_supervisor_goldens as capture
from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.units import digest


@pytest.fixture(scope="module")
def native_guard(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-rate-guard") / "capture"
    # 041a01b6 prospectively makes rate drift warning-only. The stopped r6
    # guard remains historical continuation evidence, so replay the unchanged
    # native capture used by accepted 0a0f5faf rather than recreating an error
    # that the current producer no longer raises.
    recorded = Path(__file__).parent / "fixtures/supervisor_goldens/historical-rate-drift"
    shutil.copytree(recorded, root)
    provenance = json.loads((root / "provenance.json").read_text())
    for relative, expected in provenance["files"].items():
        payload = (root / relative).read_bytes()
        assert len(payload) == expected["size"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    output = root / "output"
    receipt = json.loads((output / "receipt.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert provenance["receipt_error"] == r.ACCOUNTING_RATE_DRIFT_ERROR
    assert provenance["failure_class"] == "ACCOUNTING"
    assert receipt["stop_verified"] is True
    assert receipt["terminal"] is None
    assert receipt["cleanup_error"] is None
    assert json.loads((root / "provider-sessions.json").read_text()) == {}
    # The producer intentionally used a private commit object store. Restore
    # only its captured commit header into another private store for source scans.
    repo = Path(__file__).resolve().parents[2]
    objects = root.parent / "objects"
    objects.mkdir()
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo,
        text=True,
    ).strip()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_OBJECT_DIRECTORY", str(objects))
        mp.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(Path(common) / "objects"))
        restored = (
            subprocess.check_output(
                ["git", "hash-object", "-w", "-t", "commit", "--stdin"],
                cwd=repo,
                input=(root / "pinned-commit.raw").read_bytes(),
            )
            .decode()
            .strip()
        )
        assert restored == manifest["commit"]
        yield root, receipt, manifest, digest(output / "manifest.json")


def test_native_guard_admits_full_producer_root(native_guard) -> None:
    root, receipt, manifest, sha = native_guard
    r.verify_mechanical_stop(receipt, manifest, sha, output_root=root / "output")
    r._check_supervisor_log_fault_evidence(
        root / "supervisor.log",
        expected_job_id=manifest["job_id"],
        expected_generation=receipt["generation"],
        validated_receipt=receipt,
        expected_output_root=Path(manifest["output_dir"]),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"error": r.ACCOUNTING_RATE_DRIFT_ERROR + "; NovelDiagnosticException('x')"},
        {"error": r.ACCOUNTING_RATE_DRIFT_ERROR.replace("AccountingError", "SupervisorError")},
        {"error": "AccountingError('supervised assignment disappeared from sessions')"},
        {"error": "AccountingError('unknown rate drift')"},
        {"terminal": {"reason": "timeout"}},
        {"stop_verified": False},
        {"cleanup_error": "stop failed"},
        {"status": "COMPLETED"},
        {"failure_class": "SUPERVISOR"},
        {"returncode": 124},
    ],
)
def test_native_guard_envelope_fault_twins(native_guard, change: dict) -> None:
    _, receipt, manifest, sha = native_guard
    r.verify_mechanical_stop(receipt, manifest, sha)
    with pytest.raises(r.ResumeRefused):
        r.verify_mechanical_stop({**receipt, **change}, manifest, sha)


@pytest.mark.parametrize("surface", ["receipt", "progress", "poll", "header", "log"])
@pytest.mark.parametrize(
    "fault",
    [
        {"error": "NovelDiagnosticException('x')"},
        {"finite": False},
        {"material_gaps": ["distribution support"]},
        {"status": "failed"},
    ],
)
def test_scientific_fault_always_overrides_rate_guard(
    tmp_path: Path, native_guard, surface: str, fault: dict
) -> None:
    root, receipt, manifest, sha = native_guard
    output = tmp_path / "output"
    shutil.copytree(root / "output", output)
    r.verify_mechanical_stop(receipt, manifest, sha, output_root=output)
    if surface == "receipt":
        changed = copy.deepcopy(receipt)
        changed["diagnostic"] = fault

        def action() -> None:
            r.verify_mechanical_stop(changed, manifest, sha, output_root=output)

    elif surface == "log":
        log = tmp_path / "supervisor.log"
        log.write_bytes((root / "supervisor.log").read_bytes())
        with log.open("a") as stream:
            stream.write(json.dumps(fault) + "\n")

        def action() -> None:
            r.verify_mechanical_stop(
                receipt, manifest, sha, output_root=output, supervisor_log_path=log
            )

    else:
        target = {
            "progress": output / "exports/training/progress.json",
            "poll": output / "poll-999999.json",
            "header": output / "exports/training/iteration-0336.json",
        }[surface]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(fault))

        def action() -> None:
            r.verify_mechanical_stop(receipt, manifest, sha, output_root=output)

    with pytest.raises(r.ResumeRefused):
        action()


def test_rate_guard_text_is_not_admissible_as_trainer_error(native_guard) -> None:
    _, receipt, manifest, sha = native_guard
    changed = {**receipt, "diagnostic": {"error": r.ACCOUNTING_RATE_DRIFT_ERROR}}
    with pytest.raises(r.ResumeRefused):
        r.verify_mechanical_stop(changed, manifest, sha)


@pytest.mark.parametrize("surface", ["progress", "poll", "header", "supervisor-log"])
def test_rate_guard_error_cannot_be_relocated(tmp_path: Path, native_guard, surface: str) -> None:
    root, receipt, manifest, sha = native_guard
    output = tmp_path / "output"
    shutil.copytree(root / "output", output)
    target = {
        "progress": output / "exports/training/progress.json",
        "poll": output / "poll-999999.json",
        "header": output / "exports/training/iteration-0336.json",
        "supervisor-log": tmp_path / "supervisor.log",
    }[surface]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"error": r.ACCOUNTING_RATE_DRIFT_ERROR}) + "\n")
    message = (
        "progress.json records explicit scientific/non-mechanical fault"
        if surface == "progress"
        else "outside the validated receipt root"
    )
    with pytest.raises(r.ResumeRefused, match=message):
        r.verify_mechanical_stop(
            receipt,
            manifest,
            sha,
            output_root=output,
            supervisor_log_path=target if surface == "supervisor-log" else None,
        )


def test_current_producer_rate_drift_is_warning_only(tmp_path):
    provenance = capture.capture_one(tmp_path / "current-rate-warning", "rate-drift")
    assert provenance["receipt_error"] is None
    assert provenance["failure_class"] is None
