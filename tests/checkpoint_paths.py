"""Shared checkpoint path resolution for worktree and main-checkout test runs."""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_MAIN_CHECKOUT = "SNAKE_MAIN_CHECKOUT"
ENV_CKPT_DIR = "SNAKE_CKPT_DIR"


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
            return Path(common_dir).resolve().parent
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
    """Return candidate saved_snakes directories in priority order."""
    candidates: list[Path] = []
    ckpt_override = os.environ.get(ENV_CKPT_DIR, "").strip()
    if ckpt_override:
        candidates.append(Path(ckpt_override).expanduser())
    candidates.append(REPO_ROOT / "saved_snakes")
    main_saved = main_checkout_root() / "saved_snakes"
    if main_saved not in candidates:
        candidates.append(main_saved)
    return tuple(candidates)


def saved_checkpoint_path(filename: str) -> Path:
    """Return the first existing path for filename across worktree and main saved_snakes."""
    for directory in saved_snakes_dirs():
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return REPO_ROOT / "saved_snakes" / filename


def has_saved_checkpoints(*filenames: str) -> bool:
    """Return True iff every named checkpoint file exists in saved_snakes_dirs()."""
    return all(saved_checkpoint_path(name).is_file() for name in filenames)
