"""Healthy/fault twins for Argon's packet-09 rejection, without science data."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_json, ct_resume, ct_units

HERE = Path(__file__).parent


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
@pytest.mark.parametrize(
    "name",
    [
        "progress.json",
        "stall-watchdog.json",
        "stage-0001.json",
        "poll-000001.json",
        "state.json",
        "terminal.json",
        "receipt.json",
        "events.jsonl",
        "stage-0001.pt.xz.manifest",
        "supervisor.log",
        "job.log",
    ],
)
def test_nonfinite_retained_surface_twins(tmp_path: Path, name: str, token: str) -> None:
    path = tmp_path / name
    # An otherwise valid real receipt avoids a vacuous refusal before the injected leaf.
    healthy = {"counters": {"elapsed": 12.5}}
    if name == "receipt.json":
        healthy.update(
            status="INCOMPLETE",
            stop_verified=True,
            cleanup_error=None,
            failure_class="SUPERVISOR",
            error="whole-session wall cap reached",
        )
    if name in {"state.json", "terminal.json"}:
        healthy.update(job_id="ct0", generation="g0", status="running")
    text = json.dumps(healthy)
    if name == "supervisor.log":
        text = "2026-10-28T00:00:00Z " + text
    path.write_text(text + "\n")
    ct_resume._check_attempt_base_fault_evidence(tmp_path)
    path.write_text(text.replace("12.5", token) + "\n")
    with pytest.raises(ct_resume.CTResumeRefused, match="non-finite|corrupt supervisor"):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_decoded_object_and_producer_twins(value: float, tmp_path: Path) -> None:
    healthy = {"metadata": [{"elapsed": 1.5}]}
    ct_resume._scan_json_faults(healthy, label="decoded")
    assert ct_json.loads(ct_json.dumps(healthy)) == healthy
    faulty = {"metadata": [{"elapsed": value}]}
    with pytest.raises(ct_resume.CTResumeRefused, match="non-finite"):
        ct_resume._scan_json_faults(faulty, label="decoded")
    with pytest.raises(ValueError, match="non-finite"):
        ct_json.dumps(faulty)
    from research.sandbox_a_resident.ct_run import _write_progress

    _write_progress(tmp_path, healthy)
    before = (tmp_path / "progress.json").read_bytes()
    with pytest.raises(ValueError, match="non-finite"):
        _write_progress(tmp_path, faulty)
    assert (tmp_path / "progress.json").read_bytes() == before


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_ledger_and_verified_stage_header_twins(tmp_path: Path, token: str) -> None:
    control = {"source_commit": "a" * 40}
    ledger = ct_units.empty_ledger(control)
    path = tmp_path / "verified-ledger.json"
    path.write_text(json.dumps(ledger))
    assert ct_units.load_ledger(path, control)[0] == {}
    path.write_text(json.dumps({**ledger, "diagnostic": "TOKEN"}).replace('"TOKEN"', token))
    with pytest.raises(ValueError, match="non-finite"):
        ct_units.load_ledger(path, control)
    # Rehash the fault header: hash verification alone must never admit its JSON.
    ct_units.publish_stage_snapshot(
        tmp_path,
        1,
        lambda p: p.write_bytes(b"payload"),
        metadata={
            "allocation": ct_units.allocation(control),
            "source_commit": "a" * 40,
            "elapsed": 12.5,
        },
    )
    header = tmp_path / "stage-0001.json"
    exports = [
        ct_units.verified_descriptor(tmp_path / f"stage-0001{ext}", ct_units.unit_id(1))
        for ext in (".pt", ".json")
    ]
    ledger["units"] = [{"id": ct_units.unit_id(1), "exports": exports}]
    path.write_text(json.dumps(ledger))
    assert set(ct_units.load_ledger(path, control)[0]) == {1}
    header.write_text(header.read_text().replace("12.5", token))
    Path(str(header) + ".sha256").write_text(ct_units.digest(header) + "\n")
    ledger["units"][0]["exports"][1] = ct_units.verified_descriptor(header, ct_units.unit_id(1))
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="non-finite"):
        ct_units.load_ledger(path, control)


@pytest.mark.parametrize("name", ["diagnostic.json", "events.jsonl", "supervisor.log", "job.log"])
@pytest.mark.parametrize("key", ["LookupError: injected", "FloatingPointError: injected"])
def test_raw_and_escaped_fault_keys_in_json_log_twins(tmp_path: Path, name: str, key: str) -> None:
    path = tmp_path / name
    prefix = "2026-10-28T00:00:00Z " if name.endswith(".log") else ""
    path.write_text(prefix + '{"diagnostic": 0}\n')
    ct_resume._check_attempt_base_fault_evidence(tmp_path)
    # Escape the first letter so a raw marker scan by itself misses the fault.
    encoded = json.dumps({key: 0}).replace(key[0], "\\u%04x" % ord(key[0]), 1)
    path.write_text(prefix + encoded + "\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_all_ct_json_surfaces_use_strict_codec() -> None:
    for path in HERE.glob("ct_*.py"):
        if path.name == "ct_json.py":
            continue
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(n, ast.Import) and any(a.name == "json" for a in n.names)
            for n in ast.walk(tree)
        ), path
    for text in ('{"x":0,"x":1}', '{"x":' + "[" * 130 + "0" + "]" * 130 + "}"):
        with pytest.raises(ValueError):
            ct_json.loads(text)


@pytest.mark.parametrize(
    "line",
    [
        "Installed Python 3.12.17 in 990ms",
        "Installed Python 3.12.18 in 12.831s",
        "Downloading cpython-3.12.17-linux-x86_64-gnu (download) (33.1MiB)",
        " Downloaded cpython-3.12.17-linux-x86_64-gnu (download)",
        " + cpython-3.12.17-linux-x86_64-gnu (python3.12)",
        "Using CPython 3.12.17",
        " + pip==26.2.3",
    ],
)
def test_bootstrap_structural_healthy_and_fault_twins(line: str) -> None:
    ct_resume._scan_job_tail(line, label="uv")
    for bad in (
        line + " LookupError: injected",
        "LookupError: injected " + line,
        line + " unrecognized",
        line.replace("3.12", "3.13"),
    ):
        if bad != line:
            with pytest.raises(ct_resume.CTResumeRefused):
                ct_resume._scan_job_tail(bad, label="uv-fault")


def _lines(seconds: int) -> list[str]:
    result = []
    for elapsed in range(0, seconds + 1, 15):
        result.append(
            "CT_HEARTBEAT "
            + json.dumps(
                {
                    "schema": "t34-cf-heartbeat/v1",
                    "study": "t34-cf-1028",
                    "phase": "collection",
                    "elapsed_sec": float(elapsed),
                },
                sort_keys=True,
            )
            + "\n"
        )
        if elapsed % 45 == 0:
            result.append(
                "CT_PROGRESS "
                + json.dumps(
                    {
                        "schema": "t34-cf-progress/v1",
                        "study": "t34-cf-1028",
                        "stage": "collection-progress",
                        "elapsed_sec": elapsed,
                        "latest_durable_stage": 0,
                        "tick": elapsed // 15,
                        "total_ticks": 5760,
                        "charged_rows": 1024 * elapsed // 15,
                        "phase": "collection",
                        "hero_deaths": 0,
                        "eligible_deaths_observed": 0,
                        "retained_death_candidates": 0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return result


def test_24_hour_closed_poll_witnesses_without_history_export(tmp_path: Path) -> None:
    # Realistic 15 s heartbeat / 45 s progress over the complete 24-hour cap.
    # No wall-clock sleep and no scientific trajectories are generated.
    lines = _lines(86400)
    full = "".join(lines)
    assert len(full.encode()) > 1024 * 1024
    token = ct_resume._EVIDENCE_CACHE.set({})
    try:
        text = ""
        for index, line in enumerate(lines):
            text += line
            tail = text.encode()[-4096:].decode()
            ct_resume._scan_job_tail(tail, label="poll")
            # Materialize the first 3h of actual retained polls for the complete scanner.
            if index < 961:
                (tmp_path / f"poll-{index+1:06d}.json").write_text(
                    json.dumps({"snapshot": {"job_log_tail": tail}})
                )
        terminal_tail = full.encode()[-65536:].decode()
        ct_resume._scan_job_tail(terminal_tail, label="terminal")
        first, rest = terminal_tail.split("\n", 1)
        bad = "LookupError: injected".rjust(len(first), "x") + "\n" + rest
        assert len(bad.encode()) == 65536
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_job_tail(bad, label="terminal-fault")
    finally:
        ct_resume._EVIDENCE_CACHE.reset(token)
    # A fresh admission cache must reconstruct witnesses before reading state/log.
    last = json.loads(sorted(tmp_path.glob("poll-*.json"))[-1].read_text())
    last.update(job_id="ct0", generation="g0", status="running")
    (tmp_path / "state.json").write_text(json.dumps(last))
    (tmp_path / "supervisor.log").write_text(json.dumps(last) + "\n")
    ct_resume._check_attempt_base_fault_evidence(tmp_path)
    row = json.loads((tmp_path / "poll-000700.json").read_text())
    row["snapshot"]["job_log_tail"] += "\nLookupError: injected"
    (tmp_path / "poll-000700.json").write_text(json.dumps(row))
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_full_duration_history_real_shell_seal_and_reserve(tmp_path: Path) -> None:
    history = tmp_path / "output/ct/progress-history.log"
    history.parent.mkdir(parents=True)
    history.write_text("".join(_lines(86400)))
    script = (HERE / "ct_job.sh").read_text()
    function = script[script.index("seal_stall_diagnostics() {") : script.index("\non_exit() {")]
    env = {**os.environ, "COLAB_CLI_ROOT": str(tmp_path)}
    command = 'diagnostics="$COLAB_CLI_ROOT/output/ct/stall-watchdog"\n' + function
    result = subprocess.run(
        ["bash", "-c", command + "\nseal_stall_diagnostics"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "sealed_dir=" in result.stderr
    assert ct_resume._descriptor(history, runtime_path=history.name)["size"] > 1024 * 1024
    ct_resume._check_attempt_base_fault_evidence(history.parent)
    assert ct_units.STALL_DIAGNOSTIC_SEAL_RESERVE_BYTES >= 6 * 1024**2 + 3 * 74
    history.write_text(history.read_text() + "\nLookupError: injected\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(history.parent)


def test_import_stall_native_watchdog_healthy_fault_twins(tmp_path: Path) -> None:
    script = """
import runpy, sys, time
from research.sandbox_a_resident import ct_entrypoint, ct_liveness
from pathlib import Path
out = Path(sys.argv[1]); stall = sys.argv[2] == 'stall'
def boundary(*args, **kwargs):
    assert 'torch' not in sys.modules
    assert ct_liveness._CURRENT._thread.is_alive()
    assert (out / 'stall-watchdog/stall-watchdog.json').exists()
    if stall: time.sleep(10)
    return {'main': lambda: 0}
runpy.run_module = boundary
sys.argv = ['entry', 'runner', '--output', str(out), '--stall-timeout-seconds', '0.25']
ct_entrypoint.main()
"""
    for kind, expected in (("healthy", 0), ("stall", 1)):
        out = tmp_path / kind
        result = subprocess.run(
            [sys.executable, "-c", script, str(out), kind],
            text=True,
            capture_output=True,
            timeout=5,
        )
        assert result.returncode == expected, result.stderr
        stacks = (out / "stall-watchdog/faulthandler-stacks.log").read_text()
        assert bool(stacks) == (kind == "stall")
        assert "CT_HEARTBEAT " in result.stdout


def test_accounting_exact_cause_and_rpc_refusal_twins() -> None:
    healthy = dict(
        status="INCOMPLETE",
        stop_verified=True,
        cleanup_error=None,
        failure_class="ACCOUNTING",
        terminal=None,
        error="AccountingError('priced session unit cap reached')",
    )
    ct_resume._check_retained_receipt(healthy, label="receipt")
    for delta in (
        {"error": "AccountingError('balance dropped')"},
        {"error": healthy["error"] + " LookupError: injected"},
        {"terminal": {"reason": "timeout", "returncode": 138}},
        {"failure_class": "RPC_EXEC_POLL_TIMEOUT", "error": "TimeoutExpired('poll', 60)"},
    ):
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._check_retained_receipt({**healthy, **delta}, label="fault")


@pytest.mark.parametrize(
    "cause",
    [
        "whole-session wall cap reached",
        "priced session unit cap reached",
        "study reserve reached",
        "accounting",
    ],
)
def test_full_continuation_host_stop_without_history_twins(tmp_path: Path, cause: str) -> None:
    from research.sandbox_a_resident.test_ct_hardening import _continuation_fixture

    _packet, _broker, _control, _exports, kwargs = _continuation_fixture(tmp_path)
    base = tmp_path / "a0"
    receipt = kwargs["terminal_receipt"]
    receipt.update(
        failure_class="ACCOUNTING" if cause == "accounting" else "SUPERVISOR",
        terminal=None,
        error=(
            "AccountingError('priced session unit cap reached')" if cause == "accounting" else cause
        ),
    )
    text = "Installed Python 3.12.17 in 990ms\n"
    for index, line in enumerate(_lines(3 * 3600)):
        text += line
        (base / f"poll-{index+1:06d}.json").write_text(
            json.dumps({"snapshot": {"job_log_tail": text.encode()[-4096:].decode()}})
        )
    (base / "receipt.json").write_text(json.dumps(receipt))
    assert not list(base.rglob("progress-history.log"))
    ct_resume.write_continuation_packet(tmp_path / "healthy-long", **kwargs)
    row = json.loads((base / "poll-000700.json").read_text())
    row["snapshot"]["job_log_tail"] += "\nLookupError: injected"
    (base / "poll-000700.json").write_text(json.dumps(row))
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume.write_continuation_packet(tmp_path / "fault-long", **kwargs)


@pytest.mark.parametrize("name", ["supervisor.log", "job.log"])
@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "1e999", '[{"x": NaN}]'])
def test_nonfinite_scalar_and_array_log_twins(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text('{"x": 1.0}\n')
    ct_resume._check_attempt_base_fault_evidence(tmp_path)
    path.write_text(text + "\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


@pytest.mark.parametrize("duration", ["990ms", "1.18s", "850ms", "31.204s"])
def test_dynamic_bootstrap_lines_become_closed_suffix_witnesses(duration: str) -> None:
    line = f"Installed Python 3.12.17 in {duration}"
    text = (line + "\n") * 200
    tail = text.encode()[-4096:].decode()

    @ct_resume._evidence_session
    def scan() -> None:
        ct_resume._scan_job_tail(line, label="earlier-complete-bootstrap")
        ct_resume._scan_job_tail(tail, label="later-byte-cut-bootstrap")
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_job_tail("x" + tail[1:], label="changed-fragment")

    scan()


@pytest.mark.parametrize(
    "key",
    [
        "canary",
        "canary_evidence",
        "canary_observations",
        "SNAKE_COLAB_CANARY",
        "SNAKE_COLAB_CANARY_PACKAGE",
    ],
)
def test_qualified_key_exceptions_never_exempt_values_or_extended_keys(key: str) -> None:
    ct_resume._scan_json_faults({key: {"elapsed": 1.5}}, label="healthy-inventory")
    for row in (
        {key: {"detail": "LookupError: injected"}},
        {key: {"elapsed": float("nan")}},
        {key + " LookupError: injected": {"elapsed": 1.5}},
    ):
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_json_faults(row, label="fault-inventory")
