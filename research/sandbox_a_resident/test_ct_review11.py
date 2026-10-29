"""Packet-10 review corrections: realistic slow host polls and fault twins."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from research.sandbox_a_resident import ct_json, ct_liveness, ct_resume
from research.sandbox_a_resident.test_ct_review10 import _lines


def _stream(seconds: int = 7200) -> tuple[list[int], list[str], str]:
    lines = _lines(seconds)
    offsets, ends, full = [], [], ""
    index = 0
    for elapsed in range(0, seconds + 1, 15):
        full += lines[index]
        index += 1
        if elapsed % 45 == 0:
            full += lines[index]
            index += 1
        offsets.append(len(full.encode()))
        ends.append(full)
    return offsets, ends, full


def _segments(base: Path, ends: list[str], until: int) -> list[Path]:
    publisher = ct_liveness.ProgressSegments(base)
    previous = ""
    for tick in range(8, until + 1, 8):  # immutable publication every 120 seconds
        text = ends[tick]
        for line in text[len(previous) :].splitlines():
            publisher.append(line)
        publisher.publish()
        previous = text
    return sorted(base.glob("progress-history-*.log"))


def _polls(base: Path, ends: list[str], gap_start: int, gap_ticks: int) -> None:
    count = 0
    for tick, text in enumerate(ends):
        # One healthy poll takes >=300 s, including the ordinary 15 s sleep.
        if gap_start < tick < gap_start + gap_ticks:
            continue
        count += 1
        (base / f"poll-{count:06d}.json").write_text(
            json.dumps({"snapshot": {"job_log_tail": text.encode()[-4096:].decode()}})
        )


@pytest.mark.parametrize("gap_seconds", [315, 615])
def test_real_cadence_many_slow_poll_placements(tmp_path: Path, gap_seconds: int) -> None:
    _, ends, _ = _stream(3600)
    _segments(tmp_path, ends, len(ends) - 1)
    assert not (tmp_path / "progress-history.log").exists()
    # 134 distinct placements per gap duration; one poll per 15-second tick.
    # 315 = full 300-second qualified operation plus its ordinary 15-second sleep.
    # 615 includes additional account-lock/credential/accounting latency.
    # Use production scans for immutable witnesses and the identical nested poll
    # validator in memory, avoiding thousands of redundant filesystem writes.
    for placement in range(24, 158):
        token = ct_resume._EVIDENCE_CACHE.set({})
        try:
            ct_resume._check_attempt_base_fault_evidence(tmp_path)
            for tick in range(
                max(0, placement - 2), min(len(ends), placement + gap_seconds // 15 + 3)
            ):
                if placement < tick < placement + gap_seconds // 15:
                    continue
                tail = ends[tick].encode()[-4096:].decode()
                ct_resume._scan_json_faults(
                    {"snapshot": {"job_log_tail": tail}}, label="healthy-poll"
                )
            for fault in (tail + "\nLookupError: injected", "x" + tail[1:]):
                with pytest.raises(ct_resume.CTResumeRefused):
                    ct_resume._scan_json_faults(
                        {"snapshot": {"job_log_tail": fault}}, label="fault-poll"
                    )
        finally:
            ct_resume._EVIDENCE_CACHE.reset(token)


@pytest.mark.parametrize(
    "fault", ["hash", "marker", "gap", "partial", "fault-text", "fragment", "poll-fragment"]
)
def test_immutable_segment_fault_twins(tmp_path: Path, fault: str) -> None:
    _, ends, _ = _stream(1800)
    segments = _segments(tmp_path, ends, len(ends) - 1)
    _polls(tmp_path, ends, 30, 21)
    ct_resume._check_attempt_base_fault_evidence(tmp_path)
    path = segments[3]
    if fault == "hash":
        path.write_text(path.read_text().replace('"elapsed_sec": 360.0', '"elapsed_sec": 361.0'))
        path.write_text(path.read_text() + " ")
    elif fault == "marker":
        Path(f"{path}.complete").unlink()
    elif fault == "gap":
        for member in (path, Path(f"{path}.sha256"), Path(f"{path}.complete")):
            member.unlink()
    elif fault == "poll-fragment":
        poll = sorted(tmp_path.glob("poll-*.json"))[-1]
        row = json.loads(poll.read_text())
        text = row["snapshot"]["job_log_tail"]
        assert not text.startswith("CT_")
        row["snapshot"]["job_log_tail"] = "x" + text[1:]
        poll.write_text(json.dumps(row))
    else:
        if fault == "partial":
            path.write_text(path.read_text().rstrip("\n"))
        elif fault == "fault-text":
            path.write_text(path.read_text() + "LookupError: injected\n")
        elif fault == "fragment":
            path.write_text(path.read_text()[40:])
        Path(f"{path}.sha256").write_text(ct_resume.digest(path) + "\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_host_stop_without_final_history_full_admission_twins(tmp_path: Path) -> None:
    from research.sandbox_a_resident.test_ct_hardening import _continuation_fixture

    _, _, _, _, kwargs = _continuation_fixture(tmp_path)
    base = tmp_path / "a0"
    receipt = kwargs["terminal_receipt"]
    receipt.update(
        failure_class="SUPERVISOR", terminal=None, error="whole-session wall cap reached"
    )
    (base / "receipt.json").write_text(json.dumps(receipt))
    _, ends, _ = _stream(3 * 3600)
    segments = _segments(base, ends, len(ends) - 5)  # last <120 s can remain unexported
    _polls(base, ends, 137, 21)
    assert not list(base.rglob("progress-history.log"))
    ct_resume.write_continuation_packet(tmp_path / "healthy", **kwargs)
    segment = segments[40]
    segment.write_text(segment.read_text() + "LookupError: injected\n")
    Path(f"{segment}.sha256").write_text(ct_resume.digest(segment) + "\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume.write_continuation_packet(tmp_path / "fault", **kwargs)


def test_live_segment_publisher_completion_last_and_stop_flush(tmp_path: Path, monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(ct_liveness.time, "monotonic", lambda: now[0])
    current = ct_liveness.start_status(tmp_path / "out")
    try:
        for tick in range(1, 18):
            now[0] = tick * 15.0
            current._emit()
            if tick % 8 == 0:
                path = current.output / f"progress-history-{tick // 8 - 1:04d}.log"
                ct_resume._descriptor(path, runtime_path=path.name)
                assert path.read_text().endswith("\n")
        first = (current.output / "progress-history-0000.log").read_bytes()
    finally:
        ct_liveness.finish_status()
    assert (current.output / "progress-history-0000.log").read_bytes() == first
    assert len(list(current.output.glob("progress-history-*.log"))) == 3
    ct_resume._check_attempt_base_fault_evidence(current.output)
    # Fail closed on namespace collisions; never replace an earlier segment.
    duplicate = ct_liveness.ProgressSegments(current.output)
    duplicate.append("CT_HEARTBEAT " + json.dumps({"phase": "imports"}))
    with pytest.raises(FileExistsError):
        duplicate.publish()
    assert (current.output / "progress-history-0000.log").read_bytes() == first


@pytest.mark.parametrize(
    "line",
    [
        "Export inventory {not JSON}",
        "Downloaded file with {braces} in its name",
        "2026-10-28T12:00:00Z info: {not JSON}",
    ],
)
def test_prose_braces_healthy_fault_twins(tmp_path: Path, line: str) -> None:
    path = tmp_path / "supervisor.log"
    path.write_text(line + "\n")
    ct_resume._scan_supervisor_log(path, job_id=None, generation=None)
    path.write_text(line + " LookupError: injected\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._scan_supervisor_log(path, job_id=None, generation=None)


@pytest.mark.parametrize("prefix", ["", "2026-10-28T12:00:00Z ", "2026-10-28T12:00:00.123+00:00 "])
def test_actual_json_log_lines_remain_strict(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "supervisor.log"
    path.write_text(prefix + '{"elapsed": 1.5}\n')
    ct_resume._scan_supervisor_log(path, job_id=None, generation=None)
    for bad in ('{"elapsed": NaN}', "{broken}", '{"LookupError: injected":0}'):
        path.write_text(prefix + bad + "\n")
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_supervisor_log(path, job_id=None, generation=None)


@pytest.mark.parametrize("kind", [float, np.float32, np.float64, Decimal])
def test_all_real_and_decimal_finite_twins(kind) -> None:
    ct_json.require_finite({"x": [kind("1.5"), True, False]})
    for token in ("nan", "inf", "-inf"):
        value = kind(token)
        for row in ({"x": [value]}, {value: "key"}):
            with pytest.raises(ValueError, match="non-finite"):
                ct_json.require_finite(row)
            with pytest.raises(ct_resume.CTResumeRefused, match="non-finite"):
                ct_resume._scan_json_faults(row, label="decoded")


def test_uv_capture_matches_closed_warning_and_recorded_version() -> None:
    capture = json.loads(
        (Path(__file__).parents[2] / "tests/fixtures/ct-uv-packet11/capture.json").read_text()
    )
    assert capture["returncode"] == 0
    ct_resume._scan_job_tail(capture["version"], label="historical-Mac-uv-version")
    warning = capture["stderr"].splitlines()[0]
    ct_resume._scan_job_tail(warning, label="qualified-uv-warning")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._scan_job_tail(warning + " LookupError: injected", label="uv-fault")


def test_segment_publication_is_completion_last(tmp_path: Path, monkeypatch) -> None:
    publisher = ct_liveness.ProgressSegments(tmp_path)
    publisher.append(_lines(0)[0].strip())
    original = Path.open
    observed = []

    def open_path(path, *args, **kwargs):
        if path.name.endswith(".complete") and args and args[0] == "x":
            payload = Path(str(path).removesuffix(".complete"))
            assert payload.is_file()
            sidecar = Path(f"{payload}.sha256")
            assert sidecar.read_text().strip() == ct_resume.digest(payload)
            assert not path.exists()
            observed.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    publisher.publish()
    assert observed == ["progress-history-0000.log.complete"]


def test_live_segment_publisher_size_and_count_fault_twins(tmp_path: Path, monkeypatch) -> None:
    publisher = ct_liveness.ProgressSegments(tmp_path)
    publisher.append(_lines(0)[0].strip())
    publisher.publish()
    for constant in ("SEGMENT_LIMIT_BYTES", "HISTORY_LIMIT_BYTES", "MAX_SEGMENTS"):
        with monkeypatch.context() as patch:
            patch.setattr(ct_liveness, constant, 0)
            with pytest.raises(ValueError, match="segment.*cap"):
                publisher.append(_lines(0)[0].strip())
                publisher.publish()
