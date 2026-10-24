"""Load the pre-speed F0 factory only inside an explicitly isolated worker.

This is an operational fallback for a sealed T-EQ FAIL.  It is deliberately
not a compatibility shim: the caller must bind a distinct, clean detached
worktree at the frozen pin before importing the old factory closure.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

OLD_PIN = "61900af28932e47e22e332840fc4406559adb062"
WORKER_ENV = "BL_FROZEN_FACTORY_WORKER"


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(f"frozen factory git check failed: {' '.join(args)}")
    return result.stdout.strip()


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _validate(root: Path, expected_pin: str) -> Path:
    if not isinstance(expected_pin, str) or len(expected_pin) != 40:
        raise ValueError("frozen factory expected pin must be a full commit SHA")
    root = root.expanduser().resolve()
    candidate = Path.cwd().resolve()
    if root == candidate:
        raise ValueError("frozen factory root must differ from candidate worktree")
    if "snake-dqn-run-bl-f0-mac-d256-02.noindex" in root.parts:
        raise ValueError("frozen factory cannot use the live D256 worktree")
    if not root.is_dir():
        raise ValueError("frozen factory root is not a directory")
    if any(root.rglob(".IN_USE_LOCK")):
        raise RuntimeError("frozen factory root is in use")
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("frozen factory root is not a Git worktree")
    if _git(root, "rev-parse", "HEAD") != expected_pin:
        raise ValueError("frozen factory root pin differs")
    detached = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if detached.returncode == 0:
        raise ValueError("frozen factory root must be detached")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("frozen factory root is not clean")
    return root


def _purge_candidate_modules() -> None:
    for name in tuple(sys.modules):
        if (
            name == "research"
            or name.startswith("research.")
            or name == "src"
            or name.startswith("src.")
        ):
            del sys.modules[name]


def _assert_old_origins(root: Path) -> None:
    foreign: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if (
            name != "research"
            and not name.startswith("research.")
            and name != "src"
            and not name.startswith("src.")
        ):
            continue
        origin = getattr(module, "__file__", None)
        if origin and not _under(Path(origin), root):
            foreign.append(f"{name}={origin}")
    if foreign:
        raise RuntimeError(
            "frozen factory imported candidate science helper: " + ", ".join(foreign)
        )


def activate(root: Path | str, expected_pin: str = OLD_PIN) -> ModuleType:
    """Return the old ``factory`` module after replacing candidate science imports.

    This function is intentionally worker-only.  Calling it in an orchestration
    process could replace imports held by unrelated operational code.
    """
    if os.environ.get(WORKER_ENV) != "1":
        raise RuntimeError("frozen factory activation is worker-only")
    old_root = _validate(Path(root), expected_pin)
    _purge_candidate_modules()
    sys.path.insert(0, str(old_root))
    os.chdir(old_root)
    importlib.invalidate_caches()
    module = importlib.import_module("research.bl_labels.factory")
    origin = getattr(module, "__file__", None)
    if not origin or not _under(Path(origin), old_root):
        raise RuntimeError("frozen factory module origin differs")
    _assert_old_origins(old_root)
    return module
