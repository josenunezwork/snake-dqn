"""Focused liveness watchdog tests; the expiry check uses an isolated process."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.sandbox_a_resident.stall_watchdog import StallWatchdog


def _assert_runtime_scans_stall_diagnostics(runtime_root: Path, diagnostics: Path) -> None:
    """Check the same immutable-export scanner used by the runtime supervisor."""
    from research.colab_cli import runtime

    exports = runtime._exports(runtime_root, 1024 * 1024)
    expected = []
    for name in ("faulthandler-stacks.log", "stall-watchdog.json"):
        payload = diagnostics / name
        expected.append(
            {
                "path": f"training/stall-watchdog/{name}",
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "size": payload.stat().st_size,
            }
        )
    assert exports == expected


def test_progress_exports_latest_stage_and_rearms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "research.sandbox_a_resident.stall_watchdog.faulthandler.cancel_dump_traceback_later",
        lambda: calls.append(("cancel", None)),
    )
    monkeypatch.setattr(
        "research.sandbox_a_resident.stall_watchdog.faulthandler.dump_traceback_later",
        lambda *args, **kwargs: calls.append(("arm", (args, kwargs))),
    )
    watchdog = StallWatchdog(tmp_path / "diagnostics", 45)

    watchdog.start("prepare")
    watchdog.progress("screen:mixed:4864", {"frame": 4864, "total": 5000})
    record = json.loads((tmp_path / "diagnostics/stall-watchdog.json").read_text())
    watchdog.close()

    assert record["stage"] == "screen:mixed:4864"
    assert record["counters"] == {"frame": 4864, "total": 5000}
    assert record["timeout_seconds"] == 45
    assert record["pid"] == os.getpid()
    assert [name for name, _ in calls] == ["cancel", "arm", "cancel", "arm", "cancel"]
    args, kwargs = calls[3][1]
    assert args == (45.0,)
    assert kwargs["repeat"] is False
    assert kwargs["exit"] is True


def test_close_cancels_without_leaving_live_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "research.sandbox_a_resident.stall_watchdog.faulthandler.cancel_dump_traceback_later",
        lambda: calls.append("cancel"),
    )
    monkeypatch.setattr(
        "research.sandbox_a_resident.stall_watchdog.faulthandler.dump_traceback_later",
        lambda *args, **kwargs: calls.append("arm"),
    )
    watchdog = StallWatchdog(tmp_path, 1)
    watchdog.start()
    watchdog.close()
    watchdog.close()

    assert calls == ["cancel", "arm", "cancel"]


@pytest.mark.parametrize(
    ("name", "suffix"),
    (
        ("stall-watchdog.json", ".sha256"),
        ("stall-watchdog.json", ".complete"),
        ("faulthandler-stacks.log", ".sha256"),
        ("faulthandler-stacks.log", ".complete"),
    ),
)
def test_start_refuses_sealed_diagnostics_without_mutating_any_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, suffix: str
) -> None:
    """A restart must leave the sealed export and timer state completely untouched."""
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    payload = diagnostics / name
    payload.write_bytes(b"original immutable diagnostic\n")
    sidecar = Path(f"{payload}{suffix}")
    sidecar.write_bytes(b"original immutable sidecar\n")
    before = {path.name: path.read_bytes() for path in diagnostics.iterdir()}
    timer_calls: list[str] = []
    monkeypatch.setattr(
        "research.sandbox_a_resident.stall_watchdog.faulthandler.cancel_dump_traceback_later",
        lambda: timer_calls.append("cancel"),
    )
    monkeypatch.setattr(
        "research.sandbox_a_resident.stall_watchdog.faulthandler.dump_traceback_later",
        lambda *args, **kwargs: timer_calls.append("arm"),
    )

    with pytest.raises(RuntimeError, match="sealed stall diagnostics"):
        StallWatchdog(diagnostics, 45).start()

    assert {path.name: path.read_bytes() for path in diagnostics.iterdir()} == before
    assert timer_calls == []


def test_timeout_dumps_stacks_and_exits_nonzero(tmp_path: Path) -> None:
    """The native watchdog must still terminate a CPU/native-style busy wait."""
    diagnostics = tmp_path / "diagnostics"
    code = "\n".join(
        (
            "from pathlib import Path",
            "import ctypes",
            "from research.sandbox_a_resident.stall_watchdog import StallWatchdog",
            f"watchdog = StallWatchdog(Path({str(diagnostics)!r}), 0.15)",
            "watchdog.start('before-native-stall')",
            "ctypes.PyDLL(None).sleep(30)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode != 0
    dump = (diagnostics / "faulthandler-stacks.log").read_text()
    assert "Timeout" in dump
    assert 'File "<string>"' in dump
    assert (
        json.loads((diagnostics / "stall-watchdog.json").read_text())["stage"]
        == "before-native-stall"
    )


def test_job_shell_seals_diagnostics_after_watchdog_style_child_exit(tmp_path: Path) -> None:
    """The parent shell publishes diagnostic sidecars after a child ``os._exit``."""
    real_python = Path(sys.executable)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    diagnostics = tmp_path / "runtime/output/training/stall-watchdog"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"research.sandbox_a_resident.train"* ]]; then\n'
        '  mkdir -p "$FAKE_DIAGNOSTICS"\n'
        '  printf \'%s\\n\' \'{"stage": "mixed:4864"}\' > "$FAKE_DIAGNOSTICS/stall-watchdog.json"\n'
        "  printf '%s\\n' 'Timeout fake stack' > \"$FAKE_DIAGNOSTICS/faulthandler-stacks.log\"\n"
        "  exit 73\n"
        "fi\n"
        'exec "$REAL_PYTHON" "$@"\n'
    )
    fake_python.chmod(0o755)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    completed = subprocess.run(
        ["bash", "research/sandbox_a_resident/train_job.sh"],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "COLAB_CLI_ROOT": str(runtime_root),
            "COLAB_INPUT_DIR": str(tmp_path / "input"),
            "SANDBOX_A_SOURCE_COMMIT": "a" * 40,
            "SANDBOX_A_CONTROL_JSON": "{}",
            "FAKE_DIAGNOSTICS": str(diagnostics),
            "REAL_PYTHON": str(real_python),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 73
    assert "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=" in completed.stderr
    for name in ("stall-watchdog.json", "faulthandler-stacks.log"):
        payload = diagnostics / name
        assert Path(f"{payload}.complete").read_text() == "complete\n"
        sidecar = Path(f"{payload}.sha256").read_text().split()
        assert sidecar == [hashlib.sha256(payload.read_bytes()).hexdigest(), name]
    _assert_runtime_scans_stall_diagnostics(runtime_root, diagnostics)


def test_job_shell_seals_real_faulthandler_timeout_for_runtime_export(tmp_path: Path) -> None:
    """A real watchdog expiry reaches the parent finalizer and export scanner."""
    real_python = Path(sys.executable)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    diagnostics = tmp_path / "runtime/output/training/stall-watchdog"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"research.sandbox_a_resident.train"* ]]; then\n'
        '  exec "$REAL_PYTHON" -c \'\n'
        "from pathlib import Path\n"
        "import ctypes\n"
        "from research.sandbox_a_resident.stall_watchdog import StallWatchdog\n"
        'watchdog = StallWatchdog(Path("\'"$FAKE_DIAGNOSTICS"\'"), 0.15)\n'
        'watchdog.start("before-native-stall")\n'
        "ctypes.PyDLL(None).sleep(30)\n"
        "'\n"
        "fi\n"
        'exec "$REAL_PYTHON" "$@"\n'
    )
    fake_python.chmod(0o755)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    completed = subprocess.run(
        ["bash", "research/sandbox_a_resident/train_job.sh"],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "COLAB_CLI_ROOT": str(runtime_root),
            "COLAB_INPUT_DIR": str(tmp_path / "input"),
            "SANDBOX_A_SOURCE_COMMIT": "a" * 40,
            "SANDBOX_A_CONTROL_JSON": "{}",
            "FAKE_DIAGNOSTICS": str(diagnostics),
            "REAL_PYTHON": str(real_python),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=" in completed.stderr
    assert "Timeout" in (diagnostics / "faulthandler-stacks.log").read_text()
    _assert_runtime_scans_stall_diagnostics(runtime_root, diagnostics)


@pytest.mark.parametrize("timeout", (0, -1, math.inf, -math.inf, math.nan))
def test_rejects_nonfinite_or_nonpositive_timeout(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        StallWatchdog(tmp_path, timeout)
