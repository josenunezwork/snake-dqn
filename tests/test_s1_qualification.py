"""Focused terminal-receipt and CPU-fixture checks for S1 qualification."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.q0q1_20261017 import q0b_cuda_fixtures
from research.s1_mature_20261018 import train


def test_real_failed_child_produces_completed_diagnostic_export(tmp_path: Path) -> None:
    receipt = tmp_path / "t22-cuda.json"
    command = [
        sys.executable,
        "-c",
        "import json, pathlib, sys; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps("
        "{'status':'INCOMPLETE','error':'fixture dependency missing'})); sys.exit(2)",
        str(receipt),
    ]
    with pytest.raises(subprocess.CalledProcessError):
        train._run_qualification(
            command, cwd=tmp_path, env={}, timeout=10, receipt=receipt, exports=tmp_path / "out"
        )
    exported = tmp_path / "out/t22-cuda.json"
    assert json.loads(exported.read_text())["error"] == "fixture dependency missing"
    assert exported.with_suffix(".json.sha256").read_text().split()[0] == train.sha256_file(
        exported
    )
    assert exported.with_suffix(".json.complete").is_file()


@pytest.mark.parametrize("error", ["x" * 70000, chr(34) * 70000, chr(0x1F40D) * 70000])
def test_qualification_error_cannot_exceed_diagnostic_cap(tmp_path: Path, error: str) -> None:
    receipt = tmp_path / "runtime-parity.json"
    receipt.write_text(json.dumps({"status": "INCOMPLETE", "error": error}))
    target, packet = train._qualification_failure_receipt(
        receipt, command=["python", "-m", "fixture"], failure=RuntimeError("failed")
    )
    assert target.stat().st_size <= train.QUALIFICATION_DIAGNOSTIC_MAX_BYTES
    assert packet["error_truncated"] is True
    assert packet["error"] == error[:4096]


def test_cpu_preflight_is_explicitly_not_a_cuda_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        q0b_cuda_fixtures,
        "run_cuda_pool",
        lambda *, device: calls.append(f"pool:{device}"),
    )
    monkeypatch.setattr(q0b_cuda_fixtures, "run_t22", lambda: calls.append("t22"))
    monkeypatch.setattr(q0b_cuda_fixtures, "run_t24", lambda: calls.append("t24"))
    monkeypatch.setattr(
        q0b_cuda_fixtures,
        "fixture_sources",
        lambda fixture: {fixture: "fixture-sha"},
    )

    out = tmp_path / "cpu-preflight.json"
    assert q0b_cuda_fixtures.main(["--cpu-preflight", "--out", str(out)]) == 0

    assert calls == ["pool:cpu", "t22", "t24"]
    receipt = json.loads(out.read_text())
    assert receipt["schema"] == "q0b-cpu-preflight/v1"
    assert receipt["status"] == "CPU_PREFLIGHT_PASS"
    assert receipt["cuda_qualified"] is False
    assert receipt["target_runtime"] == "cpu"


def test_failed_qualification_exports_and_logs_its_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = tmp_path / "t22-cuda.json"
    receipt.write_text(json.dumps({"status": "INCOMPLETE", "error": "missing pytest"}) + "\n")
    exports = tmp_path / "exports"
    command = ["python", "-m", "fixture"]

    monkeypatch.setattr(
        train.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(command, returncode=2),
    )

    with pytest.raises(subprocess.CalledProcessError):
        train._run_qualification(
            command,
            cwd=tmp_path,
            env={},
            timeout=1,
            receipt=receipt,
            exports=exports,
        )

    terminal = json.loads((exports / receipt.name).read_text())
    assert terminal["error"] == "missing pytest"
    assert terminal["qualification_command"] == command
    assert terminal["terminal_export"] is True
    assert (exports / f"{receipt.name}.sha256").is_file()
    assert (exports / f"{receipt.name}.complete").is_file()
    logged = capsys.readouterr().out
    assert '"qualification_failure"' in logged
    assert '"error": "missing pytest"' in logged


def test_failed_qualification_writes_a_diagnostic_when_receipt_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "runtime-parity.json"
    exports = tmp_path / "exports"
    command = ["python", "-m", "parity"]

    monkeypatch.setattr(
        train.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(command, returncode=1),
    )

    with pytest.raises(subprocess.CalledProcessError):
        train._run_qualification(
            command,
            cwd=tmp_path,
            env={},
            timeout=1,
            receipt=receipt,
            exports=exports,
        )

    terminal = json.loads((exports / "runtime-parity-failure.json").read_text())
    assert terminal["status"] == "INCOMPLETE"
    assert "CalledProcessError" in terminal["error"]
    assert terminal["receipt_path"] == receipt.name


def test_failed_qualification_exports_bounded_diagnostic_for_large_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "runtime-parity.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "parity/v1",
                "status": "INCOMPLETE",
                "comparison": {"reason": "current runtime differs from registered reference"},
                "rows": "x" * 70000,
            }
        )
    )
    exports = tmp_path / "exports"
    command = ["python", "-m", "parity"]
    monkeypatch.setattr(
        train.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(command, returncode=1),
    )

    with pytest.raises(subprocess.CalledProcessError):
        train._run_qualification(
            command,
            cwd=tmp_path,
            env={},
            timeout=1,
            receipt=receipt,
            exports=exports,
        )

    diagnostic = exports / "runtime-parity-failure.json"
    assert diagnostic.stat().st_size <= train.QUALIFICATION_DIAGNOSTIC_MAX_BYTES
    terminal = json.loads(diagnostic.read_text())
    assert terminal["receipt_path"] == receipt.name
    assert terminal["receipt_size_bytes"] > train.QUALIFICATION_DIAGNOSTIC_MAX_BYTES
    assert len(terminal["receipt_sha256"]) == 64
    assert terminal["comparison_summary"]["field"] == "comparison"
    assert "differs from registered reference" in terminal["comparison_summary"]["value"]
