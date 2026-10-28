"""Positive producer captures and defect twins for continuation admission.

These bytes came from the qualified Broker with the existing subprocess fake,
including real pinned-runtime CPU training orchestration/proxy/export outputs.
No receipt in this corpus was assembled by an admission test helper.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import training_resume as resume

GOLDENS = Path(__file__).parent / "fixtures/supervisor_goldens"
ELIGIBLE = (
    "wall-cap",
    "helper-deadline",
    "cli-deadline",
    "runaway-guard",
    "calibration-reserve",
    "lease-renewal-timeout",
    "rpc-exec-poll-timeout",
)
# Retention was artificially held by the first harness; these are schema tests,
# never proof that the qualified producer retains runtime self-release terminals.
RETAINED_TERMINAL_NON_PRODUCER = ("runtime-timeout", "runtime-lease-expiry")
ALL_CASES = (
    "clean-completion",
    *ELIGIBLE,
    *RETAINED_TERMINAL_NON_PRODUCER,
    "child-nonzero",
    "provider-loss-no-receipt",
)


@pytest.fixture(scope="module", autouse=True)
def _restore_capture_commit_objects(tmp_path_factory):
    """Preserve temporary helper pins across Git GC without refs/index changes.

    Every temporary pin has the same tree as the qualified capture baseline;
    the tiny raw commit headers are enough to reconstruct those actual objects.
    """
    repo = Path(__file__).resolve().parents[2]
    objects = tmp_path_factory.mktemp("capture-git-objects")
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=repo, text=True
    ).strip()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_OBJECT_DIRECTORY", str(objects))
        mp.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(Path(common) / "objects"))
        for case in ALL_CASES:
            root = GOLDENS / case
            manifest = json.loads((root / "output/manifest.json").read_text())
            commit = manifest["commit"]
            raw = (root / "pinned-commit.raw").read_bytes()
            header = b"commit " + str(len(raw)).encode() + b"\0"
            assert hashlib.sha1(header + raw).hexdigest() == commit
            exists = subprocess.run(
                ["git", "cat-file", "-e", commit], cwd=repo, capture_output=True
            )
            if exists.returncode:
                restored = (
                    subprocess.check_output(
                        ["git", "hash-object", "-w", "-t", "commit", "--stdin"],
                        input=raw,
                        cwd=repo,
                        text=False,
                    )
                    .decode()
                    .strip()
                )
                assert restored == commit
        yield


def _bytes_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(case: str) -> tuple[Path, dict, dict, str]:
    output = GOLDENS / case / "output"
    manifest = json.loads((output / "manifest.json").read_text())
    receipt = json.loads((output / "receipt.json").read_text())
    return output, receipt, manifest, _bytes_hash(output / "manifest.json")


@pytest.mark.parametrize("case", ALL_CASES)
def test_golden_bytes_match_capture_inventory(case: str) -> None:
    root = GOLDENS / case
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["receipt_fabricated"] is False
    assert provenance["admission_baseline_sha256"] == (
        "c6aed498752109a8adadcc4ac4dfd84f902aaa96b955db28807a47f50df97473"
    )
    assert provenance["source_commit"] == "e971d41f1adbbdc3e6bf655f66f33cc3849c04e5"
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", provenance["source_commit"], "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )

    assert provenance["provider"].startswith("tests/fixtures/fake_colab.py")
    assert provenance["source_sha256"]["research/colab_cli/supervisor.py"] == _bytes_hash(
        Path(__file__).parents[1] / "colab_cli/supervisor.py"
    )
    files = provenance["files"]
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "provenance.json"
    }
    assert set(files) == actual
    for name, descriptor in files.items():
        path = root / name
        assert path.stat().st_size == descriptor["size"]
        assert _bytes_hash(path) == descriptor["sha256"]


@pytest.mark.parametrize("case", ELIGIBLE)
def test_real_producer_mechanical_stop_is_admitted_with_all_retained_bytes(case: str) -> None:
    output, receipt, manifest, manifest_hash = _evidence(case)
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["stop_verified"] is True
    resume.verify_mechanical_stop(
        receipt,
        manifest,
        manifest_hash,
        output_root=output,
        supervisor_log_path=GOLDENS / case / "supervisor.log",
    )


@pytest.mark.parametrize("case", ALL_CASES)
def test_every_real_producer_case_has_scientific_fault_twin(tmp_path: Path, case: str) -> None:
    original = GOLDENS / case
    root = tmp_path / case
    shutil.copytree(original / "output", root)
    if case in {"clean-completion", "provider-loss-no-receipt"}:
        manifest = json.loads((root / "manifest.json").read_text())
        rec_path = root / "receipt.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
        resume._check_output_root_fault_evidence(
            root,
            expected_job_id=manifest["job_id"],
            expected_generation=rec["generation"] if rec else None,
            validated_receipt=rec,
            allow_completed=case == "clean-completion",
        )
    fault = root / "retained-scientific-diagnostic.log"
    fault.write_text("FloatingPointError: non-finite golden injected fault\n")
    # Every case, including completion/unexplained child/provider loss, must
    # diagnose this injected fault specifically rather than any pre-existing
    # reason that makes a particular case ineligible for continuation.
    with pytest.raises(resume.ResumeRefused, match="explicit scientific/non-mechanical fault"):
        resume._scan_retained_text(fault, output_root=root)
    if case in ELIGIBLE:
        _, receipt, manifest, manifest_hash = _evidence(case)
        with pytest.raises(resume.ResumeRefused, match="explicit scientific/non-mechanical fault"):
            resume.verify_mechanical_stop(receipt, manifest, manifest_hash, output_root=root)
    elif case in {"clean-completion", "provider-loss-no-receipt"}:
        with pytest.raises(resume.ResumeRefused, match="explicit scientific/non-mechanical fault"):
            resume._check_output_root_fault_evidence(
                root,
                expected_job_id=manifest["job_id"],
                expected_generation=rec["generation"] if rec else None,
                validated_receipt=rec,
                allow_completed=case == "clean-completion",
            )


def test_real_clean_completion_is_complete_and_cannot_be_continued() -> None:
    output, receipt, manifest, manifest_hash = _evidence("clean-completion")
    assert receipt["status"] == "PASS"
    assert receipt["terminal"]["returncode"] == 0
    summary = json.loads((output / "exports/training/summary.json").read_text())
    assert any("proxy" in row for row in summary["metrics"])
    assert receipt["terminal"]["export_scanner"]["name"] == "runtime._exports"
    resume._check_output_root_fault_evidence(
        output,
        expected_job_id=receipt["job_id"],
        expected_generation=receipt["generation"],
        validated_receipt=receipt,
        allow_completed=True,
    )
    with pytest.raises(resume.ResumeRefused, match="verified released incomplete run"):
        resume.verify_mechanical_stop(receipt, manifest, manifest_hash, output_root=output)


@pytest.mark.parametrize(
    "case,reason", [("runtime-timeout", "timeout"), ("runtime-lease-expiry", "lease_expired")]
)
def test_nonproducer_retained_terminal_shape_has_proxy_snapshot(case: str, reason: str) -> None:
    output, receipt, _, _ = _evidence(case)
    assert receipt["terminal"]["reason"] == reason
    assert receipt["terminal"]["returncode"] != 0
    assert not (output / "exports/training/summary.json").exists()
    metadata = json.loads((output / "exports/training/iteration-0016.json").read_text())
    assert "proxy" in metadata["metadata"]["metrics"]
    assert receipt["terminal"]["export_scanner"]["name"] == "runtime._exports"


def test_unreasoned_real_child_nonzero_refuses() -> None:
    output, receipt, manifest, manifest_hash = _evidence("child-nonzero")
    assert receipt["failure_class"] == "CHILD_NONZERO_EXIT"
    assert receipt["terminal"]["reason"] is None
    with pytest.raises(resume.ResumeRefused, match="non-mechanical child failure"):
        resume.verify_mechanical_stop(receipt, manifest, manifest_hash, output_root=output)


def test_provider_loss_cannot_manufacture_an_ordinary_receipt() -> None:
    root = GOLDENS / "provider-loss-no-receipt"
    assert json.loads((root / "provider-sessions.json").read_text()) == {}
    assert not (root / "output/receipt.json").exists()
    manifest_path = root / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with pytest.raises(resume.ResumeRefused, match="verified released incomplete run"):
        resume.verify_mechanical_stop(
            {}, manifest, _bytes_hash(manifest_path), output_root=root / "output"
        )


def test_accounting_failure_class_does_not_explain_arbitrary_errors() -> None:
    output, receipt, manifest, manifest_hash = _evidence("runaway-guard")
    receipt["error"] = "AccountingError('provider account attribution mismatched unexpectedly')"
    with pytest.raises(resume.ResumeRefused, match="lacks the exact priced session runaway guard"):
        resume.verify_mechanical_stop(receipt, manifest, manifest_hash)


def test_capture_tool_replays_unmodified_broker_against_subprocess_fake(tmp_path: Path) -> None:
    from research.sandbox_a_resident.capture_supervisor_goldens import capture_one

    captured = tmp_path / "capture"
    result = capture_one(captured, "helper-deadline")
    assert result["receipt_error"] == "Refused('whole-session deadline reached before helper')"
    assert result["failure_class"] == "SUPERVISOR"
    receipt = json.loads((captured / "output/receipt.json").read_text())
    assert receipt["stop_verified"] is True
    assert result["receipt_sha256"] == _bytes_hash(captured / "output/receipt.json")
    assert (
        result["source_sha256"]["research/colab_cli/supervisor.py"] == receipt["supervisor_sha256"]
    )
