"""Canonical, containment-safe checkpoint catalog for the web backend."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class Checkpoint:
    """An advertised checkpoint and the client name used to load it."""

    name: str
    path: str


def _is_within(path: str, root: str) -> bool:
    """Whether canonical ``path`` is contained by canonical ``root``."""
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _saved_checkpoints(saved_dir: str) -> Iterable[Checkpoint]:
    root = os.path.realpath(saved_dir)
    for candidate in sorted(glob.glob(os.path.join(saved_dir, "*.pth"))):
        path = os.path.realpath(candidate)
        # The saved-snakes namespace intentionally has no subdirectories.  A
        # symlink is admitted only when its target remains directly in it.
        if os.path.dirname(path) == root and os.path.isfile(path):
            yield Checkpoint(name=os.path.basename(candidate), path=path)


def _run_checkpoints(repo_root: str, runs_dir: str) -> Iterable[Checkpoint]:
    root = os.path.realpath(runs_dir)
    repo = os.path.realpath(repo_root)
    display_repo = os.path.abspath(repo_root)
    for candidate in sorted(
        glob.glob(os.path.join(runs_dir, "**", "latest_pqn.pth"), recursive=True)
    ):
        path = os.path.realpath(candidate)
        if not (os.path.isfile(path) and _is_within(path, root)):
            continue
        # This must remain a repo-relative client name even if a caller passed
        # a non-canonical runs_dir.  A repo outside the canonical parent cannot
        # safely advertise a loadable name.
        if not _is_within(path, repo):
            continue
        # Keep the lexical ``latest_pqn.pth`` alias that was discovered.  The
        # canonical target is only for containment and later loading, so an
        # in-root alias remains usable by the client that saw it in the list.
        yield Checkpoint(name=os.path.relpath(candidate, display_repo), path=path)


def list_checkpoint_catalog(repo_root: str, saved_dir: str, runs_dir: str) -> list[Checkpoint]:
    """Return exactly the checkpoints that the web client may request.

    Names are either a saved-snakes basename or ``runs/**/latest_pqn.pth``
    relative to ``repo_root``.  Canonical-path checks keep symlink escapes out
    of both the listing and loader namespace.
    """
    entries = [*_saved_checkpoints(saved_dir), *_run_checkpoints(repo_root, runs_dir)]
    # A duplicate advertised name should have one stable catalog entry.
    unique: dict[str, Checkpoint] = {}
    for entry in entries:
        unique.setdefault(entry.name, entry)
    return [unique[name] for name in sorted(unique)]


def resolve_checkpoint_name(
    name: object, repo_root: str, saved_dir: str, runs_dir: str
) -> Optional[Checkpoint]:
    """Resolve one untrusted client name only when it is catalog-advertised."""
    if not isinstance(name, str) or not name or os.path.isabs(name):
        return None
    # Reject traversal and alternate spellings rather than normalizing them to
    # a catalog entry.  This makes the advertised name the sole load contract.
    if any(part == ".." for part in name.replace("\\", "/").split("/")):
        return None
    for entry in list_checkpoint_catalog(repo_root, saved_dir, runs_dir):
        if name == entry.name:
            return entry
    return None
