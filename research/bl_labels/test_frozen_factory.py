"""Hermetic subprocess contracts for the worker-only frozen factory loader."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], *, cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, text=True, check=True, capture_output=True).stdout.strip()


def _old_tree(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "old"
    (root / "research" / "bl_labels").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "research" / "__init__.py").write_text("")
    (root / "research" / "bl_labels" / "__init__.py").write_text("")
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "marker.py").write_text("VALUE = 'old-src'\n")
    (root / "research" / "bl_labels" / "factory.py").write_text(
        "from src.marker import VALUE\nMARKER = 'old-factory:' + VALUE\n"
    )
    _run(["git", "init", "-q"], cwd=root)
    _run(["git", "add", "."], cwd=root)
    _run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "old",
        ],
        cwd=root,
    )
    pin = _run(["git", "rev-parse", "HEAD"], cwd=root)
    _run(["git", "checkout", "--detach", "-q", pin], cwd=root)
    return root, pin


def _activate(root: Path, pin: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    code = """
import os, sys
from research.bl_labels.frozen_factory import activate
module = activate(sys.argv[1], sys.argv[2])
print(module.MARKER)
print(module.__file__)
"""
    env = {**os.environ, "BL_FROZEN_FACTORY_WORKER": "1"}
    return subprocess.run(
        [sys.executable, "-c", code, str(root), pin],
        cwd=cwd,
        text=True,
        capture_output=True,
        env=env,
    )


def test_activate_imports_complete_old_closure_in_worker_subprocess(tmp_path: Path) -> None:
    old, pin = _old_tree(tmp_path)
    result = _activate(old, pin, cwd=Path.cwd())
    assert result.returncode == 0, result.stderr
    marker, origin = result.stdout.splitlines()
    assert marker == "old-factory:old-src"
    assert Path(origin).resolve().is_relative_to(old.resolve())


def test_activate_refuses_lock_and_nonworker_context(tmp_path: Path) -> None:
    old, pin = _old_tree(tmp_path)
    (old / ".IN_USE_LOCK").write_text("active\n")
    locked = _activate(old, pin, cwd=Path.cwd())
    assert locked.returncode != 0
    assert "in use" in locked.stderr
    (old / ".IN_USE_LOCK").unlink()
    code = "from research.bl_labels.frozen_factory import activate; activate('unused')"
    plain = subprocess.run(
        [sys.executable, "-c", code], cwd=Path.cwd(), text=True, capture_output=True
    )
    assert plain.returncode != 0
    assert "worker-only" in plain.stderr


def test_activate_refuses_the_named_live_d256_worktree(tmp_path: Path) -> None:
    old, pin = _old_tree(tmp_path)
    live = old.with_name("snake-dqn-run-bl-f0-mac-d256-02.noindex")
    old.rename(live)
    result = _activate(live, pin, cwd=Path.cwd())
    assert result.returncode != 0
    assert "live D256" in result.stderr
