"""Isolated Git tests for the R-A source-pin identity guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest
from research.bl_residual import ra_guard

SOURCE_FILES = (
    "research/bl_residual/model.py",
    "research/bl_residual/funding.py",
    "research/bl_labels/d256_resume.py",
    "research/bl_labels/d4096_resume.py",
)


def _git(root: Path, *args: str) -> str:
    """Run a local-only Git command and return its stripped stdout."""
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    """Create a committed tree and publish its main ref to a local bare origin."""
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "source-guard@example.invalid")
    _git(root, "config", "user.name", "Source Guard")
    for relative, payload in {
        "research/bl_residual/model.py": "MODEL = 'frozen'\n",
        "research/bl_residual/funding.py": "FUNDING = 'frozen'\n",
        "research/bl_labels/d256_resume.py": "HASH_HELPER = 'frozen'\n",
        "research/bl_labels/d4096_resume.py": "HASH_IMPORT = 'frozen'\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "Freeze source identity")
    _git(root, "branch", "-M", "main")
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-qu", "origin", "main")
    _git(root, "fetch", "-q", "origin")
    return root, _git(root, "rev-parse", "HEAD")


def _guard_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(ra_guard, "ROOT", root)
    monkeypatch.setattr(ra_guard, "SOURCE_FILES", SOURCE_FILES)


def test_source_identity_accepts_exact_head_published_on_origin_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, pin = _repository(tmp_path)
    _guard_root(monkeypatch, root)

    assert ra_guard.source_identity(pin) == {
        relative: digest(root / relative) for relative in SOURCE_FILES
    }


def test_source_identity_refuses_source_pin_that_is_not_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, pin = _repository(tmp_path)
    (root / SOURCE_FILES[0]).write_text("MODEL = 'newer'\n")
    _git(root, "commit", "-am", "Advance head")
    _git(root, "push", "-q", "origin", "main")
    _guard_root(monkeypatch, root)

    with pytest.raises(ValueError, match="executing HEAD"):
        ra_guard.source_identity(pin)


@pytest.mark.parametrize("relative", SOURCE_FILES)
def test_source_identity_refuses_dirty_representative_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relative: str
) -> None:
    root, pin = _repository(tmp_path)
    (root / relative).write_text("SOURCE = 'dirty'\n")
    _guard_root(monkeypatch, root)

    with pytest.raises(ValueError, match="source differs"):
        ra_guard.source_identity(pin)


def test_source_identity_includes_collector_hash_implementation() -> None:
    assert {
        "research/bl_labels/d256_resume.py",
        "research/bl_labels/d4096_resume.py",
    } <= set(ra_guard.SOURCE_FILES)


def test_source_identity_refuses_head_not_published_to_origin_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _ = _repository(tmp_path)
    _git(root, "checkout", "-qb", "unpublished")
    (root / SOURCE_FILES[1]).write_text("FUNDING = 'unpublished'\n")
    _git(root, "commit", "-am", "Keep unpublished source local")
    _guard_root(monkeypatch, root)

    with pytest.raises(subprocess.CalledProcessError):
        ra_guard.source_identity(_git(root, "rev-parse", "HEAD"))


def test_source_identity_refuses_malformed_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _ = _repository(tmp_path)
    _guard_root(monkeypatch, root)

    with pytest.raises(ValueError, match="full published commit"):
        ra_guard.source_identity("not-a-full-pin")
