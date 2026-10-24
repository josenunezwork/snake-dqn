"""Focused command-policy tests; these use no factory process or workload."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from research.compute.factory_companion_commands import CommandValidationError, validate_command


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, *, node: str = "tests/test_small.py::test_one") -> tuple[Path, str]:
    (tmp_path / "tests").mkdir()
    fixture = tmp_path / "tests/test_small.py"
    fixture.write_text("def test_one(): pass\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = tmp_path / "review.txt"
    manifest.write_text(
        "schema=factory-companion-test-manifest/v2\n"
        f"source_commit={commit}\nfixture=tests/test_small.py|{_sha(fixture)}\nnode={node}\n"
    )
    return manifest, _sha(manifest)


def test_light_rejects_preprocessor_and_external_git_diff(tmp_path: Path) -> None:
    with pytest.raises(CommandValidationError):
        validate_command("light", ["rg", "--pre", "/bin/false", "x"], str(tmp_path))
    with pytest.raises(CommandValidationError):
        validate_command("light", ["git", "diff", "--ext-diff"], str(tmp_path))


def test_light_rejects_git_environment_and_direct_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_PAGER", "evil")
    with pytest.raises(CommandValidationError):
        validate_command("light", ["git", "status"], str(tmp_path))
    monkeypatch.delenv("GIT_PAGER")
    with pytest.raises(CommandValidationError):
        validate_command("light", ["python", "-c", "pass"], str(tmp_path))
    with pytest.raises(CommandValidationError):
        validate_command("light", ["./git", "status"], str(tmp_path))


def test_test_requires_reviewed_manifest_and_explicit_plugin(tmp_path: Path) -> None:
    manifest, digest = _manifest(tmp_path)
    argv = [
        "/x/venv/bin/python",
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:xdist",
        "-p",
        "research.compute.factory_companion_pytest",
        "tests/test_small.py::test_one",
    ]
    result = validate_command("test", argv, str(tmp_path), str(manifest), digest)
    assert result["executable_class"] == "pytest" and set(result) == {
        "class",
        "executable_class",
        "argv_sha256",
    }
    with pytest.raises(CommandValidationError):
        validate_command(
            "test",
            argv[:-2] + ["tests/test_small.py::test_one"],
            str(tmp_path),
            str(manifest),
            digest,
        )


def test_client_accepts_only_client_or_reviewed_script(tmp_path: Path) -> None:
    manifest, digest = _manifest(tmp_path)
    assert (
        validate_command("client", ["codex", "exec"], str(tmp_path), str(manifest), digest)[
            "executable_class"
        ]
        == "codex"
    )
    with pytest.raises(CommandValidationError):
        validate_command("client", ["python", "-c", "pass"], str(tmp_path), str(manifest), digest)
    child = "a" * 64
    script = tmp_path / "handoff.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"# factory-companion-reviewed-child-sha256: {child}\n"
        "exec codex exec\n"
    )
    manifest.write_text(manifest.read_text() + f"script=handoff.sh|{_sha(script)}|{child}\n")
    digest = _sha(manifest)
    assert (
        validate_command(
            "client", ["/bin/bash", "handoff.sh"], str(tmp_path), str(manifest), digest
        )["executable_class"]
        == "reviewed-handoff"
    )
