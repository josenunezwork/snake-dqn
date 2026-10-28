"""Second real A100 capture: startup cadence and full-admission fault twins."""

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

GOLDEN = Path(__file__).resolve().parents[2] / "tests/fixtures/ct-cuda-smoke-09"
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
    return ct_cuda_capture.validate_capture(
        root / "output",
        root / "supervisor.log",
        packet=ct_cuda_capture.ROOT / "research/sandbox_a_resident/packet-ct-cuda-smoke-1028-09",
        binding=GOLDEN,
    )


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
    assert report["exports"] == 31 and report["checkpoint_bytes"] == 267434164
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
                "scientific/non-mechanical|job_log_tail|signal marker|capture report shape|"
                "runtime evidence|stop marker|qualified-transport proof|capture probe shape|"
                "host log differs|cadence"
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
        ValueError,
        match="SLA|stop marker|real-CUDA|source members|corrupt export|foreign|host log differs",
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
    for name in ("complete", "signal", "progress-history")
    for line in (
        GOLDEN
        / "output/exports/ct-cuda-canary"
        / ("progress-history.log" if name == "progress-history" else f"{name}-entrypoint.log")
    )
    .read_text()
    .splitlines()
    if line.startswith(("CT_PROGRESS ", "CT_HEARTBEAT ", "CT_SIGUSR1_CLEAN_EXIT "))
)


def _line_shapes() -> tuple[str, ...]:
    shapes = {}
    for line in REAL_PROGRESS:
        prefix, _, payload = line.partition(" ")
        row = json.loads(payload)
        key = (prefix, row.get("phase"), row.get("stage"), tuple(sorted(row)))
        shapes.setdefault(key, line)
    return tuple(shapes.values())


@pytest.mark.parametrize("line", _line_shapes())
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


def _transport_shapes() -> tuple[tuple[str, dict], ...]:
    receipt = _read(GOLDEN / "output/receipt.json")
    seen = {}
    for surface in ("cli_calls", "contents_transfers", "transfers", "control_operations"):
        for row in receipt[surface]:
            key = (surface, tuple(sorted(row)), row.get("operation"), row.get("status"))
            seen.setdefault(key, (surface, row))
    return tuple(seen.values())


@pytest.mark.parametrize("surface,record", _transport_shapes())
@pytest.mark.parametrize("fault", [False, True])
def test_every_real_cuda_transport_shape_full_admission(
    playback: Path, surface: str, record: dict, fault: bool
) -> None:
    receipt = _read(playback / "output/receipt.json")
    row = copy.deepcopy(record)
    if fault:
        row["diagnostic"] = FAULT
    receipt[surface].append(row)
    _write(playback / "output/receipt.json", receipt)
    if fault:
        with pytest.raises(ct_resume.CTResumeRefused, match="scientific/non-mechanical"):
            _admit(playback)
    else:
        _admit(playback)


def test_all_real_lines_have_ast_derived_shapes() -> None:
    from research.sandbox_a_resident.ct_progress import (
        capture_inventory,
        heartbeat_inventory,
        validate_heartbeat,
        validate_progress,
    )

    assert set(capture_inventory()) == {"report", "cadence", "complete", "signal", "observation"}
    observed = set()
    for line in REAL_PROGRESS:
        prefix, _, text = line.partition(" ")
        row = json.loads(text)
        if prefix == "CT_HEARTBEAT":
            validate_heartbeat(row)
            observed.add(row["phase"])
        elif prefix == "CT_PROGRESS":
            validate_progress(row, fixture_completion=True)
        else:
            ct_resume._validate_signal_marker(row)
    assert observed == heartbeat_inventory()[1]


def test_renaming_binding_directory_cannot_skip_cadence(playback: Path) -> None:
    renamed = playback / "renamed-binding"
    renamed.mkdir()
    for name in (
        "dispatch-original.json",
        "dispatch-live.json",
        "accounting.json",
        "prepared-source-plan.json",
    ):
        shutil.copyfile(GOLDEN / name, renamed / name)
    report = ct_cuda_capture.validate_capture(
        playback / "output",
        playback / "supervisor.log",
        packet=ct_cuda_capture.ROOT / "research/sandbox_a_resident/packet-ct-cuda-smoke-1028-09",
        binding=renamed,
    )
    assert report["liveness"]["host_observed_max_gap_seconds"] < 18


@pytest.mark.parametrize(
    "fault",
    [None, "gap", "claim", "hash", "shape", "child-clock", "parent-gap", "parent-shape", "poll"],
)
def test_cadence_record_full_admission_twins(playback: Path, fault: str | None) -> None:
    path = playback / "output/exports/ct-cuda-canary/capture.json"
    capture = _read(path)
    cadence = capture["probes"][0]["progress_cadence"]
    if fault == "gap":
        cadence["observations"][20]["received_seconds"] += 70
    elif fault == "claim":
        cadence["max_gap_seconds"] = 0
    elif fault == "hash":
        cadence["observations"][0]["line_sha256"] = "0" * 64
    elif fault == "shape":
        cadence["extra"] = 0
    elif fault == "parent-gap":
        capture["startup_heartbeat_cadence"]["max_gap_seconds"] = 60.1
    elif fault == "parent-shape":
        capture["startup_heartbeat_cadence"]["extra"] = "benign"
    elif fault == "child-clock":
        path = playback / "output/exports/ct-cuda-canary/complete-entrypoint.log"
        text = path.read_text()
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("CT_HEARTBEAT "))
        row = json.loads(lines[index].partition(" ")[2])
        row["elapsed_sec"] = 75.0
        lines[index] = "CT_HEARTBEAT " + json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
    elif fault == "poll":
        path = playback / "output/poll-000015.json"
        poll = _read(path)
        poll["job_log_tail"] += "; " + FAULT
        _write(path, poll)
    if fault not in {"child-clock", "poll"}:
        _republish(playback, path, capture)
    if fault:
        with pytest.raises(ValueError, match="cadence|capture cadence shape|hash|host log differs"):
            _admit(playback)
    else:
        report = _admit(playback)["liveness"]
        assert report["host_observed_max_gap_seconds"] == pytest.approx(17.3)
        assert report["host_first_ct_line_seconds"] == 180.1
        assert report["probes"][0]["parent_receive_max_gap_seconds"] < 15.1


@pytest.mark.parametrize("surface", ["last_traceback_line", "last_error_line"])
@pytest.mark.parametrize(
    "mutation", [None, "append", "embedded", "unknown-phase", "unknown-key", "non-object"]
)
def test_terminal_heartbeat_closed_schema_twins(
    playback: Path, surface: str, mutation: str | None
) -> None:
    output = playback / "output"
    terminal = _read(output / "terminal.json")
    line = terminal["last_traceback_line"]
    row = json.loads(line.partition(" ")[2])
    if mutation == "embedded":
        row["phase"] = FAULT
    elif mutation == "unknown-phase":
        row["phase"] = "new-phase"
    elif mutation == "unknown-key":
        row["extra"] = "benign"
    terminal[surface] = "CT_HEARTBEAT " + json.dumps(row)
    if mutation == "non-object":
        terminal[surface] = "CT_HEARTBEAT []"
    if mutation == "append":
        terminal[surface] += "; " + FAULT
    receipt = _read(output / "receipt.json")
    receipt["terminal"] = terminal
    _write(output / "receipt.json", receipt)
    _write(output / "terminal.json", terminal)
    if mutation:
        with pytest.raises(ValueError, match="scientific/non-mechanical|non-mechanical"):
            _admit(playback)
    else:
        _admit(playback)


@pytest.mark.parametrize("mutation", [None, "duplicate", "missing", "hash"])
def test_export_inventory_order_and_integrity_twins(playback: Path, mutation: str | None) -> None:
    output = playback / "output"
    terminal = _read(output / "terminal.json")
    rows = terminal["export_scanner"]["inventory"]
    rows.reverse()
    if mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "hash":
        rows[0]["sha256"] = "0" * 64
    receipt = _read(output / "receipt.json")
    receipt["terminal"] = terminal
    _write(output / "receipt.json", receipt)
    _write(output / "terminal.json", terminal)
    if mutation:
        with pytest.raises(ValueError, match="export scanner disagreement"):
            _admit(playback)
    else:
        _admit(playback)
