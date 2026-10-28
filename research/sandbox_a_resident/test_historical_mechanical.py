"""Full retained historical evidence, including native transport and ops logs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import training_resume as resume

FIXTURES = Path(__file__).parent / "fixtures/historical_mechanical"
ATTEMPTS = ("attempt-08", "attempt-09")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def _preserve_original_manifest_commit() -> None:
    """Restore the captured original commit object if local Git GC removed it."""
    repo = Path(__file__).resolve().parents[2]
    for attempt in ATTEMPTS:
        root = FIXTURES / attempt
        manifest = json.loads((root / "output/manifest.json").read_text())
        raw = (root / "pinned-commit.raw").read_bytes()
        identity = hashlib.sha1(b"commit " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        assert identity == manifest["commit"]
        exists = subprocess.run(
            ["git", "cat-file", "-e", identity], cwd=repo, capture_output=True, check=False
        )
        if exists.returncode:
            restored = subprocess.check_output(
                ["git", "hash-object", "-w", "-t", "commit", "--stdin"], input=raw, cwd=repo
            )
            assert restored.decode().strip() == identity


@pytest.mark.parametrize("attempt", ATTEMPTS)
def test_historical_capture_matches_complete_hash_inventory(attempt: str) -> None:
    root = FIXTURES / attempt
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["receipt_fabricated"] is False
    assert provenance["capture_kind"].startswith("byte-for-byte copy of real historical")
    files = provenance["files"]
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "provenance.json"
    }
    assert set(files) == actual
    for name, descriptor in files.items():
        path = root / name
        assert not path.is_symlink()
        assert path.stat().st_size == descriptor["size"]
        assert _sha256(path) == descriptor["sha256"]


def _admit(output: Path, supervisor_log: Path) -> None:
    manifest_path = output / "manifest.json"
    resume.verify_mechanical_stop(
        json.loads((output / "receipt.json").read_text()),
        json.loads(manifest_path.read_text()),
        _sha256(manifest_path),
        output_root=output,
        supervisor_log_path=supervisor_log,
    )


@pytest.mark.parametrize("attempt", ATTEMPTS)
def test_historical_mechanical_stop_with_complete_output_and_log_admits(attempt: str) -> None:
    root = FIXTURES / attempt
    receipt = json.loads((root / "output/receipt.json").read_text())
    assert receipt["failure_class"] == "RPC_EXEC_POLL_TIMEOUT"
    assert receipt["contents_transfers"][1]["status"] == "failed"
    assert receipt["contents_transfers"][1]["operation"] == "renewal-heartbeat"
    log = (root / "supervisor.log").read_text()
    assert "INCOMPLETE: Refused('kernel remains busy/unknown after telemetry hiccup')" in log
    assert log.rstrip().endswith("EXIT=1")
    _admit(root / "output", root / "supervisor.log")


@pytest.mark.parametrize("attempt", ATTEMPTS)
@pytest.mark.parametrize("fault_location", ("retained-output", "supervisor-log"))
def test_historical_mechanical_scientific_fault_twin_refuses_exactly(
    tmp_path: Path, attempt: str, fault_location: str
) -> None:
    original = FIXTURES / attempt
    output = tmp_path / "output"
    shutil.copytree(original / "output", output)
    supervisor_log = tmp_path / "supervisor.log"
    shutil.copyfile(original / "supervisor.log", supervisor_log)
    fault = "FloatingPointError: injected historical scientific fault\n"
    if fault_location == "retained-output":
        diagnostic = output / "injected-fault.log"
        diagnostic.write_text(fault)
        expected = (
            "prior evidence records explicit scientific/non-mechanical fault in "
            f"{diagnostic}: 'floatingpointerror'"
        )
    else:
        lines = supervisor_log.read_text().splitlines()
        line_number = len(lines) - 1
        lines.insert(-2, fault.rstrip())
        supervisor_log.write_text("\n".join(lines) + "\n")
        expected = (
            "prior evidence records explicit scientific/non-mechanical fault in "
            f"supervisor_log:{line_number}: 'floatingpointerror'"
        )
    with pytest.raises(resume.ResumeRefused, match=re.escape(expected)) as caught:
        _admit(output, supervisor_log)
    assert str(caught.value) == expected


@pytest.mark.parametrize("attempt", ATTEMPTS)
@pytest.mark.parametrize("defect", ("foreign-receipt-path", "foreign-error", "middle-exit"))
def test_historical_wrapper_footer_negative_twin_refuses(
    tmp_path: Path, attempt: str, defect: str
) -> None:
    original = FIXTURES / attempt
    output = tmp_path / "output"
    shutil.copytree(original / "output", output)
    supervisor_log = tmp_path / "supervisor.log"
    lines = (original / "supervisor.log").read_text().splitlines()
    assert lines[-1] == "EXIT=1"
    if defect == "foreign-receipt-path":
        lines[-2] += ".foreign"
    elif defect == "foreign-error":
        lines[-2] = lines[-2].replace("telemetry hiccup", "foreign telemetry hiccup")
    else:
        lines.insert(-2, "EXIT=2")
    supervisor_log.write_text("\n".join(lines) + "\n")
    with pytest.raises(resume.ResumeRefused, match="nonzero exit signal"):
        _admit(output, supervisor_log)
