"""Real tiny OS process-group checks; no simulator, production work or Mac lease."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from research.bl_labels.mac_driver import _command_group_live, _stop_command_group


def test_cleanup_kills_descendant_after_successful_leader_exit(tmp_path: Path) -> None:
    pid_path = tmp_path / "grandchild.pid"
    grandchild = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    script = (
        "import subprocess,sys,time,pathlib; "
        f'p=subprocess.Popen([sys.executable,"-c",{grandchild!r}]); '
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(.15)"
    )
    child = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    try:
        child.wait(timeout=5)
        assert child.returncode == 0
        assert _command_group_live(child.pid)
        _stop_command_group(child)
        assert not _command_group_live(child.pid)
    finally:
        # Exact group created above only; no pattern or unrelated process kill.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def test_cleanup_is_idempotent_after_group_has_exited() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    child.wait(timeout=5)
    _stop_command_group(child)
    assert not _command_group_live(child.pid)
