"""Command-policy validation for bounded factory companion work.

The supervisor calls this module before creating an owned process group.  It
returns only digest and classification metadata so operational state never keeps
prompts or a full argv vector.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class CommandValidationError(ValueError):
    """The requested companion command is outside the reviewed policy."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argv_sha(argv: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(argv).encode()).hexdigest()


def _worktree_commit(worktree: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise CommandValidationError("worktree has no resolved Git commit")
    return result.stdout.strip()


def _safe_relative(worktree: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise CommandValidationError("reviewed path must be relative to its worktree")
    candidate = (worktree / path).resolve()
    if worktree.resolve() not in candidate.parents and candidate != worktree.resolve():
        raise CommandValidationError("reviewed path escapes its worktree")
    return candidate


def _manifest(
    worktree: Path, review_manifest: str | None, review_sha: str | None, *, require_nodes: bool
) -> tuple[dict[str, str], dict[str, tuple[str, str]], set[str]]:
    if not review_manifest or not review_sha:
        raise CommandValidationError("reviewed manifest path and SHA256 are required")
    if len(review_sha) != 64 or any(char not in "0123456789abcdef" for char in review_sha):
        raise CommandValidationError("reviewed manifest SHA256 must be lowercase hex")
    manifest = Path(review_manifest)
    if not manifest.is_file() or _sha256(manifest) != review_sha:
        raise CommandValidationError("reviewed manifest SHA does not match")
    fields: dict[str, str] = {}
    scripts: dict[str, tuple[str, str]] = {}
    nodes: set[str] = set()
    fixtures: set[str] = set()
    for line in manifest.read_text().splitlines():
        if line == "schema=factory-companion-test-manifest/v2":
            fields["schema"] = line
        elif line.startswith("source_commit="):
            fields["source_commit"] = line.removeprefix("source_commit=")
        elif line.startswith("fixture="):
            value = line.removeprefix("fixture=")
            path, separator, digest = value.partition("|")
            if not separator or len(digest) != 64:
                raise CommandValidationError("malformed reviewed fixture")
            fixture = _safe_relative(worktree, path)
            if not fixture.is_file() or _sha256(fixture) != digest:
                raise CommandValidationError("reviewed fixture hash does not match")
            fixtures.add(path)
        elif line.startswith("node="):
            nodes.add(line.removeprefix("node="))
        elif line.startswith("script="):
            path, separator, rest = line.removeprefix("script=").partition("|")
            digest, separator2, child_digest = rest.partition("|")
            if not separator or not separator2 or len(digest) != 64 or len(child_digest) != 64:
                raise CommandValidationError("malformed reviewed script")
            script = _safe_relative(worktree, path)
            if not script.is_file() or _sha256(script) != digest:
                raise CommandValidationError("reviewed script hash does not match")
            scripts[path] = (digest, child_digest)
        else:
            raise CommandValidationError("unrecognized reviewed manifest line")
    if fields.get("schema") != "schema=factory-companion-test-manifest/v2":
        raise CommandValidationError("reviewed manifest schema is not v2")
    if fields.get("source_commit") != _worktree_commit(worktree):
        raise CommandValidationError("reviewed manifest source commit does not match")
    if require_nodes and (
        not nodes or any(node.split("::", 1)[0] not in fixtures for node in nodes)
    ):
        raise CommandValidationError("every reviewed test node must pin its test source file")
    return fields, scripts, nodes


def _trusted_executable(argv0: str, names: set[str]) -> str:
    path = Path(argv0)
    if path.name not in names or path.parent not in {Path("."), Path("")}:
        raise CommandValidationError("companion executable must be a trusted command name")
    found = shutil.which(argv0)
    trusted_prefixes = ("/usr/bin/", "/bin/", "/opt/homebrew/bin/", "/usr/local/bin/")
    if not found or not any(found.startswith(prefix) for prefix in trusted_prefixes):
        raise CommandValidationError("companion executable is outside a trusted system path")
    return path.name


def _reject_git_side_effects(argv: Sequence[str], worktree: Path) -> None:
    forbidden = {
        "-c",
        "--config-env",
        "--ext-diff",
        "--paginate",
        "--no-pager",
        "--no-optional-locks",
    }
    if any(item in forbidden or item.startswith("--config-env=") for item in argv[1:]):
        raise CommandValidationError(
            "light git command enables configuration, pager, hooks, or external diff"
        )
    if any(
        name in os.environ
        for name in ("GIT_EXTERNAL_DIFF", "GIT_PAGER", "GIT_CONFIG_COUNT", "GIT_CONFIG_NOSYSTEM")
    ) or any(
        name.startswith("GIT_CONFIG_KEY_") or name.startswith("GIT_CONFIG_VALUE_")
        for name in os.environ
    ):
        raise CommandValidationError(
            "light git command inherits a forbidden Git environment override"
        )
    for key in ("diff.external", "core.pager"):
        probe = subprocess.run(
            ["git", "-C", str(worktree), "config", "--get", key], check=False, capture_output=True
        )
        if probe.returncode == 0 and probe.stdout.strip():
            raise CommandValidationError(
                "light git command has a configured external diff or pager"
            )
    probe = subprocess.run(
        ["git", "-C", str(worktree), "config", "--get-regexp", r"^diff\..*\.textconv$"],
        check=False,
        capture_output=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        raise CommandValidationError("light git command has configured text conversion")


def _light(argv: Sequence[str], worktree: Path) -> str:
    name = (
        _trusted_executable(argv[0], {"rg", "cat", "sed", "jq", "git"})
        if argv[0] != "/bin/bash"
        else "bash"
    )
    if name == "rg":
        if any(
            item == "--pre" or item.startswith("--pre=") or item.startswith("--pre-glob")
            for item in argv[1:]
        ):
            raise CommandValidationError("light rg cannot use a preprocessor")
    elif name == "cat":
        pass
    elif name == "sed":
        if any(
            item in {"-e", "-f", "--expression", "--file"} or item.startswith("--expression=")
            for item in argv[1:]
        ):
            raise CommandValidationError("light sed cannot evaluate a program or file")
    elif name == "jq":
        if any(item in {"-f", "--from-file", "--run-tests"} for item in argv[1:]):
            raise CommandValidationError("light jq cannot execute program or test files")
    elif name == "git":
        if len(argv) < 2 or argv[1] not in {"status", "diff", "show", "log"}:
            raise CommandValidationError("light git is limited to status, diff, show, and log")
        _reject_git_side_effects(argv, worktree)
    elif argv[0] == "/bin/bash" and len(argv) == 3 and argv[1] == "-n":
        pass
    else:
        raise CommandValidationError("unclassified executable is not light companion work")
    return name


def _client(argv: Sequence[str], worktree: Path, scripts: dict[str, tuple[str, str]]) -> str:
    if argv[0] in {"codex", "claude"}:
        found = shutil.which(argv[0])
        allowed = ("/usr/local/bin/", "/opt/homebrew/bin/", str(Path.home() / ".codex") + "/")
        if not found or not any(found.startswith(prefix) for prefix in allowed):
            raise CommandValidationError("client executable is outside a reviewed client path")
        return argv[0]
    if argv[:1] != ["/bin/bash"] or len(argv) < 2:
        raise CommandValidationError("client must be codex, claude, or a reviewed handoff script")
    script_name = str(Path(argv[1]))
    if script_name not in scripts:
        raise CommandValidationError("client script is absent from the reviewed manifest")
    _digest, child_digest = scripts[script_name]
    script = _safe_relative(worktree, script_name)
    marker = f"# factory-companion-reviewed-child-sha256: {child_digest}"
    content = script.read_text()
    if marker not in content or any(
        token in content for token in ("eval ", "source ", "`", "$(", " -c ")
    ):
        raise CommandValidationError(
            "reviewed handoff script has an unpinned or unsafe child command"
        )
    return "reviewed-handoff"


def _test(argv: Sequence[str], nodes: set[str]) -> str:
    if len(argv) < 10 or not argv[0].endswith("/venv/bin/python"):
        raise CommandValidationError("test requires an absolute project venv Python")
    prefix = [
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:xdist",
        "-p",
        "research.compute.factory_companion_pytest",
    ]
    if list(argv[1:10]) != prefix:
        raise CommandValidationError("test must use the reviewed pytest prefix and factory plugin")
    requested = argv[10:]
    if not requested or any(item.startswith("-") or item not in nodes for item in requested):
        raise CommandValidationError(
            "test nodes must be explicit and present in the reviewed manifest"
        )
    return "pytest"


def validate_command(
    klass: str,
    argv: Sequence[str],
    worktree: str,
    review_manifest: str | None = None,
    review_sha: str | None = None,
) -> dict[str, str]:
    """Reject unreviewed local work and return safe metadata for the registry."""
    if (
        klass not in {"light", "client", "test"}
        or not argv
        or any(not value or "\0" in value for value in argv)
    ):
        raise CommandValidationError("invalid companion command")
    root = Path(worktree).resolve()
    if not root.is_dir():
        raise CommandValidationError("companion worktree is unavailable")
    if klass == "light":
        executable_class = _light(argv, root)
    else:
        _fields, scripts, nodes = _manifest(
            root, review_manifest, review_sha, require_nodes=klass == "test"
        )
        executable_class = _client(argv, root, scripts) if klass == "client" else _test(argv, nodes)
    return {"class": klass, "executable_class": executable_class, "argv_sha256": _argv_sha(argv)}
