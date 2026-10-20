"""Focused source-closure and frozen-worktree tests for the T34 second fixes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.t34_pathb import source_binding_v2 as binding


def _run(repo: Path, *args: str) -> str:
    return subprocess.check_output(args, cwd=repo, text=True).strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "research/t34_pathb").mkdir(parents=True)
    (repo / "src/pkg").mkdir(parents=True)
    (repo / "research/t34_shadow").mkdir(parents=True)
    for directory in ("research", "research/t34_pathb", "src", "src/pkg", "research/t34_shadow"):
        (repo / directory / "__init__.py").write_text("")
    (repo / "src/pkg/__init__.py").write_text("from . import child\n")
    for module in binding.ENTRYPOINTS:
        relative = Path(*module.split(".")).with_suffix(".py")
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("from src.pkg import child\n")
    (repo / "src/pkg/child.py").write_text(
        "def load():\n    from research.t34_shadow import helper\n    return helper\n"
    )
    (repo / "research/t34_shadow/helper.py").write_text("VALUE = 1\n")
    (repo / "configs").mkdir()
    (repo / "configs/runtime.yaml").write_text("game: fixture\n")
    for name in binding.DECLARED_DATA_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bound\n")
    _run(repo, "git", "init", "-q")
    _run(repo, "git", "add", ".")
    _run(
        repo,
        "git",
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-qm",
        "fixture",
    )
    commit = _run(repo, "git", "rev-parse", "HEAD")
    _run(repo, "git", "checkout", "--detach", "-q")
    return repo, commit


def test_source_inventory_follows_transitive_function_scope_imports(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)

    inventory = binding.source_inventory(repo)

    assert "src/pkg/child.py" in inventory
    assert "research/t34_shadow/helper.py" in inventory
    assert "research/t34_pathb/RETENTION_PROFILE_V2.json" in inventory
    assert "configs/runtime.yaml" in inventory
    assert list(inventory) == sorted(inventory)
    assert all(len(digest) == 64 for digest in inventory.values())

    before = inventory["configs/runtime.yaml"]
    (repo / "configs/runtime.yaml").write_text("game: changed\n")
    assert binding.source_inventory(repo)["configs/runtime.yaml"] != before


def test_source_inventory_rejects_dynamic_or_missing_local_imports(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    (repo / "research/t34_pathb/timing_v2.py").write_text("__import__(name)\n")
    with pytest.raises(binding.SourceBindingError, match="dynamic import"):
        binding.source_inventory(repo)

    (repo / "research/t34_pathb/timing_v2.py").write_text("import research.not_here\n")
    with pytest.raises(binding.SourceBindingError, match="unresolved repository-local import"):
        binding.source_inventory(repo)


def test_frozen_guard_requires_detached_exact_clean_tree(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    binding.verify_frozen_worktree(repo, commit)

    (repo / "untracked-data.json").write_text("mutation\n")
    with pytest.raises(binding.SourceBindingError, match="tracked or untracked"):
        binding.verify_frozen_worktree(repo, commit)
    (repo / "untracked-data.json").unlink()

    (repo / "configs/runtime.yaml").write_text("tracked mutation\n")
    with pytest.raises(binding.SourceBindingError, match="tracked or untracked"):
        binding.verify_frozen_worktree(repo, commit)
    _run(repo, "git", "restore", "configs/runtime.yaml")

    _run(repo, "git", "checkout", "-q", "-b", "not-detached")
    with pytest.raises(binding.SourceBindingError, match="detached"):
        binding.verify_frozen_worktree(repo, commit)


def test_guard_check_is_cached_and_final_check_rechecks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, commit = _repo(tmp_path)
    guard = binding.FrozenWorktreeGuard(repo, commit, interval_seconds=60)
    calls = 0
    original = binding.verify_frozen_worktree

    def counted(path: Path, sha: str) -> None:
        nonlocal calls
        calls += 1
        original(path, sha)

    monkeypatch.setattr(binding, "verify_frozen_worktree", counted)
    guard.check()
    assert calls == 0
    guard.final_check()
    assert calls == 1
