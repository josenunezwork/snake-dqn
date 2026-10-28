"""Byte-exact real A100 golden admission and healthy/scientific-fault twins."""

from __future__ import annotations

import copy
import json
import lzma
import os
import shutil
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_cuda_capture, ct_resume
from research.sandbox_a_resident.ct_cuda_observed import BOOTSTRAP_LINES, CAPTURE_LABELS

GOLDEN = Path(__file__).resolve().parents[2] / "tests/fixtures/ct-cuda-smoke-08"
FAULT = "LookupError: injected scientific invariant"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def decoded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Lossless archival compression keeps the 255-MiB payload out of Git."""
    root = tmp_path_factory.mktemp("cuda-golden")
    shutil.copytree(GOLDEN / "output", root / "output")
    inventory = _read(GOLDEN / "inventory.json")
    for row in inventory["files"]:
        path = root / "output" / row["path"]
        if row.get("storage") == "xz-byte-exact":
            archive = Path(str(path) + ".golden.xz")
            assert ct_resume.digest(archive) == row["compressed_sha256"]
            with lzma.open(archive, "rb") as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            archive.unlink()
        assert path.stat().st_size == row["size"]
        assert ct_resume.digest(path) == row["sha256"]
    shutil.copyfile(GOLDEN / "supervisor.log", root / "supervisor.log")
    assert ct_resume.digest(root / "supervisor.log") == inventory["supervisor_log"]["sha256"]
    return root


@pytest.fixture
def playback(decoded: Path, tmp_path: Path) -> Path:
    # Only the large binary is shared. Every mutable diagnostic has private bytes.
    def copy(source: str, target: str) -> str:
        if Path(source).stat().st_size > 2_000_000:
            os.link(source, target)
            return target
        return shutil.copyfile(source, target)

    shutil.copytree(decoded / "output", tmp_path / "output", copy_function=copy)
    shutil.copyfile(decoded / "supervisor.log", tmp_path / "supervisor.log")
    return tmp_path


def _admit(root: Path) -> dict:
    return ct_cuda_capture.validate_capture(root / "output", root / "supervisor.log")


def _republish(root: Path, path: Path, value: dict) -> None:
    """Keep hash layers valid so a fault twin reaches diagnostic admission."""
    _write(path, value)
    sha = ct_resume.digest(path)
    Path(str(path) + ".sha256").write_text(f"{sha}  {path.name}\n")
    Path(str(path) + ".complete").write_text("complete\n")
    output = root / "output"
    rel = str(path.relative_to(output / "exports"))
    receipt, terminal = _read(output / "receipt.json"), _read(output / "terminal.json")
    for rows in (receipt["exports"], terminal["exports"], terminal["export_scanner"]["inventory"]):
        for row in rows:
            if row["path"] == rel:
                row.update(sha256=sha, size=path.stat().st_size)
    receipt["terminal"] = terminal
    _write(output / "receipt.json", receipt)
    _write(output / "terminal.json", terminal)


def test_real_capture_full_admission_and_success_cannot_resume(playback: Path) -> None:
    report = _admit(playback)
    assert report["status"] == "PASS_CUDA_CAPTURE_ADMISSION"
    assert report["exports"] == 30 and report["checkpoint_bytes"] == 267434164
    output = playback / "output"
    with pytest.raises(ct_resume.CTResumeRefused, match="not a qualified mechanical trigger"):
        ct_resume.verify_mechanical_stop(
            _read(output / "receipt.json"),
            _read(output / "manifest.json"),
            ct_resume.digest(output / "manifest.json"),
            export_root=output,
            supervisor_log_path=playback / "supervisor.log",
        )


@pytest.mark.parametrize(
    "surface",
    [
        "last_error_line",
        "last_traceback_line",
        "poll-transport",
        "receipt-transport",
        "job_log_tail",
        "transport_summary",
        "canary-label",
        "canary-observation",
        "progress-history",
        "fixture-runtime",
        "summary",
        "signal-marker",
    ],
)
@pytest.mark.parametrize("fault", [False, True])
def test_real_cuda_surfaces_healthy_and_fault_twins(
    playback: Path, surface: str, fault: bool
) -> None:
    output = playback / "output"
    if fault:
        if surface in {"last_error_line", "last_traceback_line", "canary-observation"}:
            terminal = _read(output / "terminal.json")
            if surface == "canary-observation":
                terminal["canary_observations"][0]["detail"] = FAULT
            else:
                terminal[surface] = FAULT
            receipt = _read(output / "receipt.json")
            receipt["terminal"] = terminal
            _write(output / "receipt.json", receipt)
            _write(output / "terminal.json", terminal)
        elif surface in {"poll-transport", "receipt-transport", "canary-label"}:
            receipt = _read(output / "receipt.json")
            if surface == "canary-label":
                receipt["canary_evidence"]["family"] += "; " + FAULT
            else:
                record = copy.deepcopy(receipt["cli_calls"][0])
                record["detail"] = FAULT
                if surface == "poll-transport":
                    _write(output / "poll-999999.json", record)
                else:
                    receipt["cli_calls"].append(record)
            _write(output / "receipt.json", receipt)
        elif surface in {"job_log_tail", "transport_summary"}:
            if surface == "job_log_tail":
                _write(output / "poll-999999.json", {"job_log_tail": FAULT})
            else:
                lines = (playback / "supervisor.log").read_text().splitlines()
                row = json.loads(lines[1])
                row["job_log_tail"] = FAULT
                lines[1] = json.dumps(row)
                (playback / "supervisor.log").write_text("\n".join(lines) + "\n")
        elif surface == "progress-history":
            path = output / "exports/ct-cuda-canary/signal/progress-history.log"
            path.write_text(path.read_text() + FAULT + "\n")
        else:
            names = {
                "fixture-runtime": "signal/fixture-runtime.json",
                "summary": "complete/summary.json",
                "signal-marker": "signal/sigusr1-marker.json",
            }
            path = output / "exports/ct-cuda-canary" / names[surface]
            value = _read(path)
            value["detail"] = FAULT
            _republish(playback, path, value)
        with pytest.raises(
            ValueError,
            match=(
                "scientific/non-mechanical|job_log_tail|signal marker|"
                "runtime evidence|stop marker"
            ),
        ):
            _admit(playback)
    else:
        _admit(playback)


@pytest.mark.parametrize("line", sorted(BOOTSTRAP_LINES))
@pytest.mark.parametrize("fault", [False, True])
def test_every_real_bootstrap_line_full_admission(playback: Path, line: str, fault: bool) -> None:
    _write(
        playback / "output/poll-999999.json",
        {"job_log_tail": line + ("; " + FAULT if fault else "")},
    )
    if fault:
        with pytest.raises(
            ct_resume.CTResumeRefused, match="scientific/non-mechanical|job_log_tail"
        ):
            _admit(playback)
    else:
        _admit(playback)


@pytest.mark.parametrize("line", sorted(CAPTURE_LABELS))
def test_capture_labels_are_exact_and_never_resume_exemptions(line: str) -> None:
    ct_resume._scan_json_faults(line, label="capture-label", fixture_completion=True)
    with pytest.raises(ct_resume.CTResumeRefused, match="scientific/non-mechanical"):
        ct_resume._scan_json_faults(
            line + "; " + FAULT, label="capture-label", fixture_completion=True
        )
    with pytest.raises(ct_resume.CTResumeRefused, match="scientific/non-mechanical"):
        ct_resume._scan_json_faults(line, label="resume-label")


@pytest.mark.parametrize(
    "mutation",
    [
        "false-marker",
        "slow-marker",
        "slow-exit",
        "false-cuda",
        "source",
        "export-hash",
        "foreign-job",
    ],
)
def test_real_capture_integrity_and_timing_twins(playback: Path, mutation: str) -> None:
    output = playback / "output"
    base = output / "exports/ct-cuda-canary"
    if mutation in {"false-marker", "slow-marker"}:
        marker = _read(base / "signal/sigusr1-marker.json")
        marker["within_15s_sla" if mutation == "false-marker" else "shutdown_elapsed_sec"] = (
            False if mutation == "false-marker" else 15.0
        )
        _republish(playback, base / "signal/sigusr1-marker.json", marker)
    elif mutation in {"slow-exit", "false-cuda"}:
        capture = _read(base / "capture.json")
        if mutation == "slow-exit":
            capture["probes"][1]["signal_to_exit_seconds"] = 15.0
        else:
            capture["real_cuda"] = False
        _republish(playback, base / "capture.json", capture)
    elif mutation == "source":
        value = _read(output / "staging/source-plan.json")
        value["tree"] = "0" * 40
        _write(output / "staging/source-plan.json", value)
    elif mutation == "foreign-job":
        value = _read(output / "poll-000001.json")
        value["state"]["job_id"] = "foreign"
        _write(output / "poll-000001.json", value)
    else:
        (base / "signal/stage-0001.pt.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(
        ValueError, match="SLA|stop marker|real-CUDA|source members|corrupt export|foreign"
    ):
        _admit(playback)


def test_real_history_bytes_are_retained_and_tail_requires_witness(tmp_path: Path) -> None:
    history = GOLDEN / "output/exports/ct-cuda-canary/signal/progress-history.log"
    assert ct_resume._scan_diagnostic_bytes(history) == history.read_text()
    poll = _read(GOLDEN / "output/poll-000032.json")
    with pytest.raises(ct_resume.CTResumeRefused, match="job_log_tail"):
        ct_resume._scan_job_tail(poll["job_log_tail"], label="unwitnessed", fixture_completion=True)


REAL_PROGRESS = tuple(
    line
    for name in ("complete", "signal")
    for line in (GOLDEN / f"output/exports/ct-cuda-canary/{name}-entrypoint.log")
    .read_text()
    .splitlines()
    if line.startswith(("CT_PROGRESS ", "CT_SIGUSR1_CLEAN_EXIT "))
)


@pytest.mark.parametrize("line", REAL_PROGRESS)
@pytest.mark.parametrize("fault", [False, True])
def test_every_real_cuda_progress_record_full_admission(
    playback: Path, line: str, fault: bool
) -> None:
    if fault:
        prefix, _, payload = line.partition(" ")
        row = json.loads(payload)
        row["extra_science_diagnostic"] = FAULT
        line = prefix + " " + json.dumps(row)
    _write(playback / "output/poll-999999.json", {"job_log_tail": line})
    if fault:
        with pytest.raises(
            ct_resume.CTResumeRefused, match="scientific/non-mechanical|job_log_tail"
        ):
            _admit(playback)
    else:
        _admit(playback)
