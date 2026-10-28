"""Corrected native captures, retained separately from immutable first captures."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from research.sandbox_a_resident import training_resume as resume
from research.sandbox_a_resident.capture_supervisor_goldens import REPO, capture_one

CAPTURES = Path(__file__).parent / "fixtures/supervisor_goldens_v2"
CASES = ("clean-completion", "child-nonzero", "runtime-timeout", "runtime-lease-expiry")


@pytest.fixture(scope="module", autouse=True)
def private_capture_objects(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO, text=True
    ).strip()
    objects = tmp_path_factory.mktemp("capture-v2-git-objects")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_OBJECT_DIRECTORY", str(objects))
        mp.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(Path(common) / "objects"))
        for case in CASES:
            root = CAPTURES / case
            manifest = json.loads((root / "output/manifest.json").read_text())
            restored = (
                subprocess.check_output(
                    ["git", "hash-object", "-w", "-t", "commit", "--stdin"],
                    input=(root / "pinned-commit.raw").read_bytes(),
                    cwd=REPO,
                )
                .decode()
                .strip()
            )
            assert restored == manifest["commit"]
            subprocess.run(["git", "cat-file", "-e", restored + "^{tree}"], cwd=REPO, check=True)
        yield


@pytest.mark.parametrize("case", CASES)
def test_corrected_capture_inventory_and_native_policy(case: str) -> None:
    root = CAPTURES / case
    provenance = json.loads((root / "provenance.json").read_text())
    assert "receipt_fabricated" not in provenance
    assert "working_tree_status" in provenance
    assert len(provenance["head_tree"]) == 40
    assert provenance["version_boundary"].startswith("runtime from manifest commit")
    for name, entry in provenance["files"].items():
        path = root / name
        assert path.stat().st_size == entry["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    receipt = json.loads((root / "output/receipt.json").read_text())
    manifest_path = root / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if case == "clean-completion":
        assert receipt["status"] == "PASS"
        assert receipt["terminal"]["reason"] is None
        assert receipt["terminal"]["export_scanner"]["name"] == "runtime._exports"
        resume._check_output_root_fault_evidence(
            root / "output",
            expected_job_id=receipt["job_id"],
            expected_generation=receipt["generation"],
            validated_receipt=receipt,
            allow_completed=True,
        )
    elif case == "child-nonzero":
        assert receipt["terminal"]["returncode"] == 23
        assert receipt["terminal"]["reason"] is None
        assert "started_at" in receipt["terminal"]
        assert receipt["terminal"]["export_scanner"]["name"] == "runtime._exports"
        with pytest.raises(resume.ResumeRefused, match="non-mechanical child failure"):
            resume.verify_mechanical_stop(
                receipt, manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            )
    else:
        assert json.loads((root / "provider-sessions.json").read_text()) == {}
        assert receipt["terminal"] is None
        assert (
            receipt["error"] == "AccountingError('supervised assignment disappeared from sessions')"
        )
        with pytest.raises(
            resume.ResumeRefused, match="lacks the exact priced session runaway guard"
        ):
            resume.verify_mechanical_stop(
                receipt, manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            )


@pytest.mark.parametrize(
    "scenario,reason", [("runtime-timeout", "timeout"), ("runtime-lease-expiry", "lease_expired")]
)
def test_live_offline_capture_preserves_self_release_observation(
    tmp_path: Path, scenario: str, reason: str
) -> None:
    root = tmp_path / scenario
    result = capture_one(root, scenario)
    terminal = json.loads((root / "runtime-observation/terminal.json").read_text())
    assert terminal["reason"] == reason
    assert terminal["returncode"] != 0
    assert json.loads((root / "provider-sessions.json").read_text()) == {}
    assert (
        result["receipt_error"]
        == "AccountingError('supervised assignment disappeared from sessions')"
    )


def test_loss_after_snapshot_and_partial_chunk_is_not_recovery_authority(tmp_path: Path) -> None:
    root = tmp_path / "provider-loss-after-download"
    result = capture_one(root, "provider-loss-after-download")
    subprocess.run(
        ["git", "hash-object", "-w", "-t", "commit", "--stdin"],
        input=(root / "pinned-commit.raw").read_bytes(),
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    output = root / "output"
    training = output / "exports/training"
    assert (training / "iteration-0032.pt.xz").stat().st_size > 17 * 1024 * 1024
    assert (training / "iteration-0016.pt").is_file()
    assert (training / "iteration-0016.json.complete").read_text() == "complete\n"
    assert list(training.glob(".transfer-*.*"))
    assert not (output / "receipt.json").exists()
    assert result["receipt_sha256"] is None
    resume._check_output_root_fault_evidence(output)
    fault = output / "diagnostic.log"
    fault.write_text("NovelDiagnosticException: injected synthetic fault\n")
    with pytest.raises(resume.ResumeRefused, match="explicit scientific/non-mechanical fault"):
        resume._check_output_root_fault_evidence(output)
    manifest_path = output / "manifest.json"
    with pytest.raises(resume.ResumeRefused, match="verified released incomplete run"):
        resume.verify_mechanical_stop(
            {},
            json.loads(manifest_path.read_text()),
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            output_root=output,
        )


def test_failed_capture_is_marked_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "provider-loss-no-receipt"

    def failed(*args, **kwargs) -> subprocess.CompletedProcess:
        root.mkdir()
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="offline host loss child returned 0"):
        capture_one(root, "provider-loss-no-receipt")
    assert not root.exists()
    assert root.with_name(root.name + ".failed").is_dir()
