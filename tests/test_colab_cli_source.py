"""Tests for the bounded Colab CLI source-delivery transport."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Callable

import pytest

from research.colab_cli import source


def _run(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    _run(repo, "config", "user.email", "tests@example.invalid")
    _run(repo, "config", "user.name", "Tests")
    files = {
        "src/pkg/main.py": "print('ok')\n",
        "research/study/run.sh": "#!/bin/sh\necho run\n",
        "research/colab_cli/source.py": "# source\n",
        "research/colab_cli/runtime.py": "# runtime\n",
        "research/colab_cli/bootstrap.sh": "#!/bin/sh\n",
        "research/colab_cli/release.py": "# release\n",
        "requirements.txt": "pytest\n",
        "pyproject.toml": "[project]\nname='tiny'\n",
        "docs/large-evidence.bin": "x" * (source.MAX_FALLBACK_BYTES + 1),
    }
    for name, value in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    (repo / "research/study/run.sh").chmod(0o755)
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "tiny")
    return repo


def _manifest(repo: Path) -> dict[str, object]:
    return {
        "commit": _run(repo, "rev-parse", "HEAD"),
        "script": "research/study/run.sh",
        "env": {},
        "budget": {"rate_units_per_hour": 1.0, "unit_cap": 1.0, "reserve_units": 0.0},
    }


def _rewrite_archive(
    source_archive: Path,
    target: Path,
    mutate: Callable[[str, bytes], bytes],
    *,
    extra: bool = False,
) -> None:
    """Make a controlled archive mutation while preserving its other members."""
    with tarfile.open(source_archive, "r:gz") as original, tarfile.open(target, "w:gz") as output:
        for member in original.getmembers():
            payload = original.extractfile(member)
            assert payload is not None
            data = mutate(member.name, payload.read())
            info = tarfile.TarInfo(member.name)
            info.mode, info.size, info.mtime = member.mode, len(data), 0
            output.addfile(info, io.BytesIO(data))
        if extra:
            info = tarfile.TarInfo("unexpected.txt")
            info.size, info.mode = 1, 0o644
            output.addfile(info, io.BytesIO(b"x"))


def _archive_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_builds_small_allowlisted_fallback(repo: Path, tmp_path: Path) -> None:
    directory = tmp_path / "delivery"
    plan = source.prepare(repo, _manifest(repo), directory)

    assert plan["fallback_size"] < source.MAX_FALLBACK_BYTES
    assert plan["fallback_unpacked_size"] < source.MAX_FALLBACK_BYTES
    assert "docs/large-evidence.bin" not in plan["files"]
    assert "src/pkg/main.py" in plan["files"]
    assert "research/study/run.sh" in plan["files"]
    assert json.loads((directory / "source-plan.json").read_text()) == plan
    assert json.loads((directory / "schema-validation.json").read_text()) == {
        "manifest_sha256": hashlib.sha256(
            json.dumps(_manifest(repo), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "required_exports": [],
        "schema": "colab-cli-manifest/v1",
        "status": "PASS",
    }
    with tarfile.open(directory / "source.tar.gz") as archive:
        assert set(archive.getnames()) == set(plan["files"])


def test_prepare_rejects_broad_research_override(repo: Path, tmp_path: Path) -> None:
    manifest = _manifest(repo)
    manifest["source_paths"] = ["research"]
    with pytest.raises(ValueError, match="unsafe source path|broad research"):
        source.prepare(repo, manifest, tmp_path / "delivery")


def test_explicit_allowlist_overrides_defaults_and_large_path_fails_before_read(
    repo: Path, tmp_path: Path
) -> None:
    manifest = _manifest(repo)
    manifest["source_paths"] = ["research/study"]
    plan = source.prepare(repo, manifest, tmp_path / "small")
    assert "research/study/run.sh" in plan["files"]
    assert "src/pkg/main.py" not in plan["files"]

    manifest["source_paths"] = ["docs/large-evidence.bin"]
    with pytest.raises(ValueError, match="unpacked cap"):
        source.prepare(repo, manifest, tmp_path / "large")


def test_prepare_rejects_selected_symlink(repo: Path, tmp_path: Path) -> None:
    link = repo / "research/study/link"
    link.symlink_to("run.sh")
    _run(repo, "add", "research/study/link")
    _run(repo, "commit", "-qm", "symlink")
    with pytest.raises(ValueError, match="symlink"):
        source.prepare(repo, _manifest(repo), tmp_path / "delivery")


def test_fetch_uses_pinned_github_identity(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivery = tmp_path / "delivery"
    plan = source.prepare(repo, _manifest(repo), delivery)
    bare = tmp_path / "remote.git"
    _run(tmp_path, "clone", "--bare", str(repo), str(bare))
    monkeypatch.setattr(source, "GITHUB_URL", str(bare))

    result = source.fetch(tmp_path / "runtime", plan)

    assert result == {"transport": "github", "commit": plan["commit"], "tree": plan["tree"]}
    assert (tmp_path / "runtime/repo/src/pkg/main.py").read_text() == "print('ok')\n"


def test_fetch_refuses_identity_mismatch_without_fallback(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivery = tmp_path / "delivery"
    plan = source.prepare(repo, _manifest(repo), delivery)
    bare = tmp_path / "remote.git"
    _run(tmp_path, "clone", "--bare", str(repo), str(bare))
    monkeypatch.setattr(source, "GITHUB_URL", str(bare))
    plan["tree"] = "0" * 40

    with pytest.raises(ValueError, match="identity"):
        source.fetch(tmp_path / "runtime", plan)
    assert not (tmp_path / "runtime/repo").exists()


def test_fetch_semantic_failure_refuses_fallback_and_outage_requests_it(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivery = tmp_path / "delivery"
    plan = source.prepare(repo, _manifest(repo), delivery)
    bare = tmp_path / "remote.git"
    _run(tmp_path, "clone", "--bare", str(repo), str(bare))
    monkeypatch.setattr(source, "GITHUB_URL", str(bare))
    missing = {**plan, "commit": "f" * 40}
    with pytest.raises(ValueError, match="without an outage"):
        source.fetch(tmp_path / "semantic", missing)

    original = source._run

    def unreachable(argv, *, cwd, timeout):
        if "fetch" in argv:
            raise subprocess.CalledProcessError(128, argv, output="fatal: Could not resolve host")
        return original(argv, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(source, "_run", unreachable)
    outcome = source.fetch(tmp_path / "outage", plan)
    assert outcome["transport"] == "fallback_required"
    assert not (tmp_path / "outage/repo").exists()


@pytest.mark.parametrize(
    ("message", "fallback"),
    [
        ("fatal: requested URL returned error: 503", True),
        ("fatal: requested URL returned error: 401", False),
        ("fatal: requested URL returned error: 403", False),
        ("fatal: requested URL returned error: 404", False),
    ],
)
def test_fetch_http_status_only_falls_back_for_5xx(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str, fallback: bool
) -> None:
    plan = source.prepare(repo, _manifest(repo), tmp_path / "delivery")
    original = source._run

    def failed_fetch(argv, *, cwd, timeout):
        if "fetch" in argv:
            raise subprocess.CalledProcessError(128, argv, output=message)
        return original(argv, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(source, "_run", failed_fetch)
    root = tmp_path / ("yes" if fallback else "no")
    if fallback:
        assert source.fetch(root, plan)["transport"] == "fallback_required"
    else:
        with pytest.raises(ValueError, match="without an outage"):
            source.fetch(root, plan)


def test_install_fallback_rejects_extra_member_and_accepts_exact_inventory(
    repo: Path, tmp_path: Path
) -> None:
    delivery = tmp_path / "delivery"
    plan = source.prepare(repo, _manifest(repo), delivery)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    archive = delivery / "source.tar.gz"
    target_archive = runtime / "source.tar.gz"
    target_archive.write_bytes(archive.read_bytes())
    result = source.install_fallback(runtime, plan)
    assert result["transport"] == "fallback"
    assert result["inventory_verified"] is True
    assert (runtime / "repo/research/study/run.sh").stat().st_mode & 0o111

    extra = tmp_path / "extra.tar.gz"
    _rewrite_archive(archive, extra, lambda _name, data: data, extra=True)
    plan_extra = {**plan, "fallback_sha256": _archive_digest(extra)}
    extra_runtime = tmp_path / "extra-runtime"
    extra_runtime.mkdir()
    (extra_runtime / "source.tar.gz").write_bytes(extra.read_bytes())
    with pytest.raises(ValueError, match="inventory"):
        source.install_fallback(extra_runtime, plan_extra)


def test_install_fallback_refuses_pinned_file_digest_mismatch(repo: Path, tmp_path: Path) -> None:
    delivery = tmp_path / "delivery"
    plan = source.prepare(repo, _manifest(repo), delivery)
    changed = tmp_path / "changed.tar.gz"
    _rewrite_archive(
        delivery / "source.tar.gz",
        changed,
        lambda name, data: b"x" * len(data) if name == "pyproject.toml" else data,
    )
    changed_plan = {**plan, "fallback_sha256": _archive_digest(changed)}
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "source.tar.gz").write_bytes(changed.read_bytes())
    with pytest.raises(ValueError, match="digest"):
        source.install_fallback(runtime, changed_plan)
