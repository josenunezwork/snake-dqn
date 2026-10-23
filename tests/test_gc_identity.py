"""Permanent small subprocess d.1 identity check."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(tmp_path: Path, policy: str, attempt: str = "") -> dict:
    path = tmp_path / f"identity-{policy}{attempt}.json"
    env = dict(os.environ, SNAKE_SERVE_GC_POLICY=policy, SNAKE_SERVE_PATHB="0")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.bl_gc.identity",
            "--policy",
            policy,
            "--seed",
            "2026102300",
            "--ticks",
            "600",
            "--out",
            str(path),
        ],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert path.with_suffix(".json.sha256").is_file()
    assert path.with_suffix(".json.complete").read_text() == "complete\n"
    return json.loads(path.read_text())


def test_startup_freeze_preserves_600_tick_watch_identity(tmp_path: Path) -> None:
    default = _run(tmp_path, "default")
    frozen = _run(tmp_path, "freeze_startup")
    control = _run(tmp_path, "default", "-control")
    assert len(default["chain"]) == len(frozen["chain"]) == 600
    assert default["final_chain_sha256"] == frozen["final_chain_sha256"]

    assert default["final_chain_sha256"] == control["final_chain_sha256"]
