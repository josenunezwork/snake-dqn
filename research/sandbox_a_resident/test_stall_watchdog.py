"""Focused liveness watchdog tests; the expiry check uses an isolated process."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
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
    assert "TRAIN01_DIAGNOSTICS_SEALED child_status=" in completed.stderr
    for name in ("stall-watchdog.json", "faulthandler-stacks.log"):
        payload = diagnostics / name
        assert Path(f"{payload}.complete").read_text() == "complete\n"
        sidecar = Path(f"{payload}.sha256").read_text().split()
        assert sidecar == [hashlib.sha256(payload.read_bytes()).hexdigest(), name]
    _assert_runtime_scans_stall_diagnostics(runtime_root, diagnostics)


@pytest.mark.parametrize("expiry", [False, True])
def test_job_shell_seals_real_faulthandler_timeout_for_runtime_export(
    tmp_path: Path, expiry: bool
) -> None:
    """The same sealing marker follows either ordinary failure or real expiry."""
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
        "import ctypes, os\n"
        "from research.sandbox_a_resident.stall_watchdog import StallWatchdog\n"
        'with StallWatchdog(Path("\'"$FAKE_DIAGNOSTICS"\'"), 0.15) as watchdog:\n'
        '    watchdog.progress("before-native-stall")\n'
        '    if os.environ["WATCHDOG_EXPIRY"] == "True":\n'
        "        ctypes.PyDLL(None).sleep(30)\n"
        '    raise RuntimeError("fixture preparation failure")\n'
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
            "WATCHDOG_EXPIRY": str(expiry),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert "TRAIN01_DIAGNOSTICS_SEALED child_status=" in completed.stderr
    dump = (diagnostics / "faulthandler-stacks.log").read_text()
    if expiry:
        assert "Timeout" in dump
    else:
        assert dump == ""
        assert "fixture preparation failure" in completed.stderr
    _assert_runtime_scans_stall_diagnostics(runtime_root, diagnostics)


@pytest.mark.parametrize("timeout", (0, -1, math.inf, -math.inf, math.nan))
def test_rejects_nonfinite_or_nonpositive_timeout(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        StallWatchdog(tmp_path, timeout)


@pytest.mark.parametrize("blocked_dump", [False, True])
@pytest.mark.parametrize("closing", [False, True])
def test_hard_exit_is_independent_of_dump_and_cancel(
    tmp_path: Path, blocked_dump: bool, closing: bool
) -> None:
    """Real full-pipe dump + held GIL cannot obstruct the independent hard stop."""
    script = r"""
import ctypes, json, os, sys, time
from pathlib import Path
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
out = Path(sys.argv[1])
watchdog = StallWatchdog(out, 0.2)
watchdog.start('real-work-stalled')
if sys.argv[2] == 'True':
    # Fill a private pipe and keep its reader open without consuming anything.
    # The real faulthandler C thread must block writing its timeout header.
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    try:
        while True: os.write(write_fd, b'x' * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(write_fd, True)
    watchdog._stack_file.close()
    watchdog._stack_file = os.fdopen(write_fd, 'w')
    watchdog.progress('real-work-stalled')
if sys.argv[3] == 'True':
    time.sleep(0.3)
    watchdog.close()  # cancel_dump_traceback_later joins the stuck C thread.
else:
    ctypes.PyDLL(None).sleep(30)  # holds the target GIL; Python timers cannot run.
raise SystemExit('hard stop failed')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(blocked_dump), str(closing)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == (-signal.SIGKILL if blocked_dump else 1)
    record = json.loads((tmp_path / "stall-watchdog.json").read_text())
    assert record["stage"] == "real-work-stalled"
    assert record["timeout_seconds"] == 0.2
    if not blocked_dump:
        assert "Timeout" in (tmp_path / "faulthandler-stacks.log").read_text()


def test_real_progress_rearms_and_close_reaps_guard_without_late_kill(tmp_path: Path) -> None:
    """Healthy work crosses multiple original deadlines and then safely disarms."""
    script = r"""
import time
from pathlib import Path
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
watchdog = StallWatchdog(Path(__import__('sys').argv[1]), 0.2)
watchdog.start('startup')
helper = watchdog._exit_guard.process
for i in range(10):
    time.sleep(0.05)
    watchdog.progress('completed-unit', {'i': i})
watchdog.close()
watchdog.close()
assert helper.poll() == 0
time.sleep(2.5)
print('healthy-rearmed-closed', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "healthy-rearmed-closed\n"
    assert (tmp_path / "faulthandler-stacks.log").read_bytes() == b""
    assert json.loads((tmp_path / "stall-watchdog.json").read_text())["counters"] == {"i": 9}


def test_dead_guard_refuses_new_progress_and_still_cleans_up(tmp_path: Path) -> None:
    watchdog = StallWatchdog(tmp_path, 10)
    watchdog.start()
    helper = watchdog._exit_guard.process
    helper.kill()
    helper.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="guard exited unexpectedly"):
            watchdog.progress("unit")
    finally:
        watchdog.close()
    assert helper.poll() == -signal.SIGKILL


def test_stack_close_error_still_reaps_guard(tmp_path: Path) -> None:
    """An ordinary diagnostic close failure must not leave a future kill armed."""
    watchdog = StallWatchdog(tmp_path, 10)
    watchdog.start()
    helper = watchdog._exit_guard.process
    original = watchdog._stack_file

    class BrokenClose:
        def close(self) -> None:
            original.close()
            raise OSError("fixture diagnostic close failed")

    watchdog._stack_file = BrokenClose()
    with pytest.raises(OSError, match="fixture diagnostic close failed"):
        watchdog.close()
    assert helper.poll() == 0
    watchdog.close()


def test_invalid_initial_progress_cleans_up_started_guard(tmp_path: Path) -> None:
    watchdog = StallWatchdog(tmp_path, 10)
    with pytest.raises(ValueError, match="nonempty"):
        watchdog.start("")
    assert watchdog._exit_guard.process.poll() == 0


def test_guard_source_is_in_training_runtime_closure() -> None:
    from research.sandbox_a_resident import training_prepare

    root = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    helper = "research/sandbox_a_resident/ct_watchdog_guard.py"
    # These are the actual Git-derived closures shipped to the VM.
    assert helper in training_prepare.source_paths(pin)


@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGINT])
def test_group_preemption_keeps_guard_alive_for_progress(tmp_path: Path, signum: int) -> None:
    """Signal the real group, then finish work and rearm the independent guard."""
    script = r"""
import os, signal, sys, time
from pathlib import Path
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
signum = int(sys.argv[2])
received = []
signal.signal(signum, lambda *_: received.append(True))
watchdog = StallWatchdog(Path(sys.argv[1]), 0.5)
watchdog.start('before-group-stop')
helper = watchdog._exit_guard.process
assert os.getpgid(helper.pid) == os.getpgrp()
os.killpg(os.getpgrp(), signum)
time.sleep(0.05)
assert received and helper.poll() is None
watchdog.progress('after-group-stop')
watchdog.close()
assert helper.poll() == 0
print('group-stop-progress-clean', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(signum)],
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "group-stop-progress-clean\n"
    assert json.loads((tmp_path / "stall-watchdog.json").read_text())["stage"] == "after-group-stop"


def test_slow_dump_finishes_within_grace_and_stalled_twin_still_dies(tmp_path: Path) -> None:
    """A real dump blocked for >0.25s gets time to finish; forever-blocked twin dies."""
    from research.sandbox_a_resident.stall_watchdog import DUMP_GRACE_SECONDS

    assert DUMP_GRACE_SECONDS == 2.0
    script = r"""
import ctypes, os, sys, threading, time
from pathlib import Path
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
watchdog = StallWatchdog(Path(sys.argv[1]), 0.2)
watchdog.start('slow-dump')
r, w = os.pipe()
os.set_blocking(w, False)
try:
    while True: os.write(w, b'x' * 4096)
except BlockingIOError:
    pass
os.set_blocking(w, True)
watchdog._stack_file.close()
watchdog._stack_file = os.fdopen(w, 'w')
watchdog.progress('slow-dump')
if sys.argv[2] == 'drain':
    def drain():
        time.sleep(0.8)  # Native dump is blocked beyond the former grace.
        while os.read(r, 4096): pass
    threading.Thread(target=drain, daemon=True).start()
    time.sleep(30)
else:
    ctypes.PyDLL(None).sleep(30)
"""
    for mode, status in (("drain", 1), ("stuck", -signal.SIGKILL)):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / mode), mode],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == status, result.stderr


def test_guard_keeps_final_group_cleanup_and_eof_disarm(tmp_path: Path) -> None:
    """Ignoring preemption must preserve TERM cleanup and EOF parent-death disarm."""
    helper = Path(__file__).with_name("ct_watchdog_guard.py")
    for termination in ("term", "eof"):
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(helper), str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            assert os.read(process.stdout.fileno(), 1) == b"R"
            os.killpg(process.pid, signal.SIGUSR1)
            os.killpg(process.pid, signal.SIGINT)
            if termination == "term":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.stdin.close()  # Same private-pipe EOF as unexpected parent death.
            process.wait(timeout=5)
            assert process.returncode == (-signal.SIGTERM if termination == "term" else 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout, process.stderr):
                pipe.close()
