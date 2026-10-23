"""Permanent subprocess d.3 retention check."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run(tmp_path: Path, policy: str) -> dict:
    path = tmp_path / f"lifecycle-{policy}.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.bl_gc.lifecycle",
            "--policy",
            policy,
            "--out",
            str(path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, SNAKE_SERVE_GC_POLICY=policy, SNAKE_SERVE_PATHB="0"),
        capture_output=True,
        text=True,
        timeout=480,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]
    assert path.with_suffix(".json.sha256").is_file()
    assert path.with_suffix(".json.complete").read_text() == "complete\n"
    return json.loads(path.read_text())


@pytest.mark.parametrize("policy", ["default", "freeze_startup"])
def test_watch_session_replacements_release_under_both_gc_policies(
    tmp_path: Path, policy: str
) -> None:
    result = _run(tmp_path, policy)
    assert result["cycles"] == 20
    assert result["tracked_objects"]["cycle_20"] <= result["tracked_objects"]["cycle_5"] + 2000
