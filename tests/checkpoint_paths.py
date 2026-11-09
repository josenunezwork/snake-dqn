"""Shared checkpoint path resolution for worktree and main-checkout test runs."""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_MAIN_CHECKOUT = "SNAKE_MAIN_CHECKOUT"
ENV_CKPT_DIR = "SNAKE_CKPT_DIR"

_CKPT_SITE_HOOK = """\
import importlib.abc
import importlib.machinery
import os
import sys

_ckpt_dir = os.environ.get("SNAKE_CKPT_DIR", "").strip()
if _ckpt_dir:
    class _CkptDirPostImportFinder(importlib.abc.MetaPathFinder):
        _active = False

        def find_spec(self, fullname, path, target=None):
            if self._active or fullname not in (
                "web.backend.served_checkpoint",
                "web.backend.session",
            ):
                return None
            self._active = True
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            finally:
                self._active = False
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def _exec_module(module):
                orig_exec(module)
                if fullname == "web.backend.served_checkpoint":
                    module.SAVED_DIR = _ckpt_dir
                    module._candidate_paths = lambda entry, saved_dir: [
                        os.path.join(saved_dir, entry.filename)
                    ]
                elif fullname == "web.backend.session":
                    module.SAVED_DIR = _ckpt_dir
                    module.DEFAULT_CHECKPOINT = os.path.join(
                        _ckpt_dir, "champion_a5_freespace_20260621.pth"
                    )

            spec.loader.exec_module = _exec_module
            return spec

    sys.meta_path.insert(0, _CkptDirPostImportFinder())
"""


@functools.lru_cache(maxsize=1)
def _ckpt_site_dir() -> Path:
    """Create a per-process sitecustomize directory that propagates SNAKE_CKPT_DIR to children."""
    site_dir = Path(tempfile.mkdtemp(prefix="snake_ckpt_site_"))
    (site_dir / "sitecustomize.py").write_text(_CKPT_SITE_HOOK, encoding="utf-8")
    return site_dir


@functools.lru_cache(maxsize=1)
def _git_main_checkout_root() -> Path:
    """Resolve the main checkout root as the parent of git's common dir."""
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if common_dir:
            common_path = Path(common_dir).resolve()
            if common_path.name == ".git" or (common_path.parent / "saved_snakes").is_dir():
                return common_path.parent
    except (OSError, subprocess.SubprocessError):
        pass
    return REPO_ROOT


def main_checkout_root() -> Path:
    """Return the main checkout root, honoring SNAKE_MAIN_CHECKOUT if set."""
    override = os.environ.get(ENV_MAIN_CHECKOUT, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _git_main_checkout_root()


def saved_snakes_dirs() -> tuple[Path, ...]:
    """Return candidate saved_snakes directories in priority order.

    Honors SNAKE_CKPT_DIR exclusively when set.
    """
    ckpt_override = os.environ.get(ENV_CKPT_DIR, "").strip()
    if ckpt_override:
        return (Path(ckpt_override).expanduser().resolve(),)
    candidates: list[Path] = [REPO_ROOT / "saved_snakes"]
    main_saved = main_checkout_root() / "saved_snakes"
    if main_saved not in candidates:
        candidates.append(main_saved)
    return tuple(candidates)


def resolve_saved_snakes_dir(*filenames: str) -> Path:
    """Return the first candidate directory containing the requested checkpoint files."""
    dirs = saved_snakes_dirs()
    for directory in dirs:
        if filenames:
            if all((directory / name).is_file() for name in filenames):
                return directory
        elif directory.is_dir() and any(directory.glob("*.pth")):
            return directory
    return dirs[0]


def saved_checkpoint_path(filename: str) -> Path:
    """Return the path for filename in the highest-priority matching saved_snakes dir."""
    return resolve_saved_snakes_dir(filename) / filename


def has_saved_checkpoints(*filenames: str) -> bool:
    """Return True iff every named checkpoint file exists in the resolved directory."""
    if not filenames:
        directory = resolve_saved_snakes_dir()
        return directory.is_dir() and any(directory.glob("*.pth"))
    directory = resolve_saved_snakes_dir(*filenames)
    return all((directory / name).is_file() for name in filenames)


def use_saved_checkpoints(monkeypatch: pytest.MonkeyPatch, *filenames: str) -> Path:
    """Ensure SNAKE_CKPT_DIR and serving checkpoint paths match the skip-check directory."""
    resolved = resolve_saved_snakes_dir(*filenames)
    worktree_saved = REPO_ROOT / "saved_snakes"
    ckpt_override = os.environ.get(ENV_CKPT_DIR, "").strip()
    if not ckpt_override and resolved != worktree_saved and resolved.is_dir():
        monkeypatch.setenv(ENV_CKPT_DIR, str(resolved))
    if ckpt_override or (resolved != worktree_saved and resolved.is_dir()):
        b8_candidate = resolved / "b8_s3_t7_20261016.pth"
        if "RULER_B8_PATH" not in os.environ and b8_candidate.is_file():
            monkeypatch.setenv("RULER_B8_PATH", str(b8_candidate))
        site_dir = str(_ckpt_site_dir())
        existing_py_path = os.environ.get("PYTHONPATH", "")
        parts = [p for p in existing_py_path.split(os.pathsep) if p]
        if site_dir not in parts:
            new_py_path = (
                f"{site_dir}{os.pathsep}{existing_py_path}" if existing_py_path else site_dir
            )
            monkeypatch.setenv("PYTHONPATH", new_py_path)
        try:
            from web.backend import served_checkpoint as registry
            from web.backend import session as session_mod

            monkeypatch.setattr(registry, "SAVED_DIR", str(resolved))
            monkeypatch.setattr(
                registry,
                "_candidate_paths",
                lambda entry, saved_dir: [os.path.join(saved_dir, entry.filename)],
            )
            monkeypatch.setattr(session_mod, "SAVED_DIR", str(resolved))
            monkeypatch.setattr(
                session_mod,
                "DEFAULT_CHECKPOINT",
                str(resolved / registry.CHAMPION_FILENAME),
            )
        except ImportError:
            pass
    return resolved
