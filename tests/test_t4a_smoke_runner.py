"""The CPU smoke runner owns and cleans up only its own subprocess session."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.t4a_two_sided_20261016.real_bank_smoke import _run_bounded


def test_owned_smoke_timeout_cleans_parent_and_child(tmp_path: Path) -> None:
    script = tmp_path / "owned.py"
    script.write_text(
        "import pathlib, signal, subprocess, sys, time\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "def stopped(*_):\n"
        "    p.write_text('stopped')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, stopped)\n"
        "if p.name == 'parent':\n"
        "    subprocess.Popen([sys.executable, __file__, str(p.with_name('child'))])\n"
        "p.with_suffix('.ready').write_text('ready')\n"
        "while True: time.sleep(.01)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded(
            [sys.executable, str(script), str(tmp_path / "parent")],
            tmp_path,
            dict(os.environ),
            timeout=2,
        )
    assert (tmp_path / "parent.ready").exists()
    assert (tmp_path / "child.ready").exists()
    assert (tmp_path / "parent").read_text() == "stopped"
    assert (tmp_path / "child").read_text() == "stopped"


def test_owned_smoke_normal_exit(tmp_path: Path) -> None:
    result = _run_bounded(
        [sys.executable, "-c", "print('cpu-runner-ok')"], tmp_path, dict(os.environ)
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "cpu-runner-ok"
