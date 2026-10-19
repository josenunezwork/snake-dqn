"""Pinned, bounded source delivery for Colab CLI jobs.

The normal transport is a commit-addressed GitHub fetch.  The archive produced
here is deliberately a small, inspected escape hatch for a GitHub outage; it
is not a copy of the repository.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

GITHUB_URL = "https://github.com/josenunezwork/snake-dqn.git"
MAX_FALLBACK_BYTES = 32 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 120
ARCHIVE_NAME = "source.tar.gz"
PLAN_NAME = "source-plan.json"
_REQUIRED_COLAB = (
    "research/colab_cli/source.py",
    "research/colab_cli/runtime.py",
    "research/colab_cli/bootstrap.sh",
    "research/colab_cli/release.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, timeout=15).strip()


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value in {".", "research"}:
        raise ValueError(f"unsafe source path: {value!r}")
    normalized = path.as_posix().rstrip("/")
    if normalized in {"", ".", ".git"} or normalized.startswith(".git/"):
        raise ValueError(f"unsafe source path: {value!r}")
    if normalized == "research" or normalized.startswith("research/") and len(path.parts) == 1:
        raise ValueError(f"unsafe source path: {value!r}")
    return normalized


def _path_allowed(value: str) -> None:
    value = _safe_path(value)
    # A study directory must be specific.  `research/` would quietly recreate
    # the old whole-repository transport through the evidence tree.
    if value == "research" or value.startswith("research/") and len(value.split("/")) < 2:
        raise ValueError(f"broad research source path: {value!r}")


def _matches(path: str, allow: list[str]) -> bool:
    return any(path == entry or path.startswith(entry + "/") for entry in allow)


def _tracked(repo: Path, commit: str) -> dict[str, tuple[str, str, str]]:
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-z", commit], cwd=repo, timeout=20)
    result: dict[str, tuple[str, str, str]] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        meta, name = item.split(b"\t", 1)
        mode, kind, digest = meta.decode().split()
        result[name.decode()] = (mode, digest, kind)
    return result


def _defaults(manifest: dict[str, Any], tracked: dict[str, tuple[str, str, str]]) -> list[str]:
    paths = ["src", "research/colab_cli", "pyproject.toml"]
    script = str(manifest["script"])
    if script.startswith("research/"):
        parts = script.split("/")
        if len(parts) >= 3:
            paths.append("/".join(parts[:2]))
    paths.extend(
        name
        for name in tracked
        if "/" not in name and name.startswith("requirements") and name.endswith(".txt")
    )
    if any(name.startswith("requirements/") for name in tracked):
        paths.append("requirements")
    return paths


def _required(manifest: dict[str, Any]) -> list[str]:
    requirements = str(manifest.get("env", {}).get("COLAB_CLI_REQUIREMENTS", "requirements.txt"))
    return [
        *_REQUIRED_COLAB,
        str(manifest["script"]),
        *([requirements] if requirements else []),
        "pyproject.toml",
    ]


def prepare(repo: Path, manifest: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Write an immutable plan and bounded sparse fallback archive for *manifest*."""
    repo, directory = repo.resolve(), directory.resolve()
    commit = str(manifest["commit"])
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("source delivery requires a full lowercase commit SHA")
    if _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}") != commit:
        raise ValueError("manifest commit is unavailable locally")
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}")
    tracked = _tracked(repo, commit)
    provided = manifest.get("source_paths")
    if provided is not None and (
        not isinstance(provided, list) or not all(isinstance(x, str) for x in provided)
    ):
        raise ValueError("source_paths must be a list of repository-relative strings")
    allow = list(provided) if provided is not None else _defaults(manifest, tracked)
    allow.extend(_required(manifest))
    allow = sorted({_safe_path(path) for path in allow})
    for path in allow:
        _path_allowed(path)

    if provided is not None:
        unknown = [path for path in allow if not any(_matches(name, [path]) for name in tracked)]
        if unknown:
            raise ValueError(f"source_paths select no tracked files: {unknown}")

    selected = {name: tracked[name] for name in tracked if _matches(name, allow)}
    missing = [path for path in _required(manifest) if path not in selected]
    if missing:
        raise ValueError(f"required source paths are not tracked by pinned commit: {missing}")
    for name, (mode, _digest, kind) in selected.items():
        if kind != "blob":
            raise ValueError(f"non-blob prohibited in fallback source: {name}")
        if mode == "120000":
            raise ValueError(f"symlink prohibited in fallback source: {name}")
        if mode not in {"100644", "100755"}:
            raise ValueError(f"unsupported tracked mode {mode} for {name}")

    unpacked = 0
    for _name, (_mode, digest, _kind) in selected.items():
        unpacked += int(_git(repo, "cat-file", "-s", digest))
        if unpacked > MAX_FALLBACK_BYTES:
            raise ValueError("sparse fallback exceeds 32 MiB unpacked cap")
    if unpacked > MAX_FALLBACK_BYTES:
        raise ValueError("sparse fallback exceeds 32 MiB unpacked cap")
    blobs: dict[str, bytes] = {}
    for name, (_mode, _digest, _kind) in selected.items():
        blobs[name] = subprocess.check_output(
            ["git", "show", f"{commit}:{name}"], cwd=repo, timeout=20
        )
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / ARCHIVE_NAME
    with tarfile.open(archive, "x:gz", format=tarfile.PAX_FORMAT) as output:
        for name in sorted(selected):
            blob = blobs[name]
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(blob), int(selected[name][0], 8) & 0o777, 0
            output.addfile(info, io.BytesIO(blob))
    if archive.stat().st_size > MAX_FALLBACK_BYTES:
        archive.unlink()
        raise ValueError("sparse fallback exceeds 32 MiB compressed cap")
    plan: dict[str, Any] = {
        "version": 1,
        "commit": commit,
        "tree": tree,
        "paths": allow,
        "fallback_archive": ARCHIVE_NAME,
        "fallback_sha256": _sha256(archive),
        "fallback_size": archive.stat().st_size,
        "fallback_unpacked_size": unpacked,
        "files": {
            name: {
                "sha256": hashlib.sha256(blobs[name]).hexdigest(),
                "mode": f"{int(mode, 8) & 0o777:04o}",
                "size": len(blobs[name]),
            }
            for name, (mode, _digest, _kind) in sorted(selected.items())
        },
    }
    encoded = json.dumps(plan, sort_keys=True, indent=2) + "\n"
    plan_path = directory / PLAN_NAME
    with plan_path.open("x") as output:
        output.write(encoded)
    plan_path.chmod(0o444)
    return plan


def _run(argv: list[str], *, cwd: Path, timeout: float) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.check_output(
        argv, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=timeout, env=env
    )


def fetch(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Fetch and verify the public pinned commit, requesting fallback only on outage."""
    root = root.resolve()
    target = root / "repo"
    if target.exists():
        raise ValueError("runtime source destination already exists")
    target.mkdir(parents=True)
    started = time.monotonic()
    # Local setup failures are configuration defects, never a reason to use a
    # fallback whose purpose is only a public-GitHub outage.
    _run(["git", "init", "--quiet", str(target)], cwd=root, timeout=10)
    _run(["git", "-C", str(target), "config", "core.hooksPath", "/dev/null"], cwd=root, timeout=5)
    # Fetching the same pinned object is idempotent. One shared budget covers
    # all attempts and backoff; a timeout never restarts this clock.
    for attempt in range(3):
        try:
            remain = FETCH_TIMEOUT_SECONDS - (time.monotonic() - started)
            if remain <= 0:
                raise subprocess.TimeoutExpired("git fetch", FETCH_TIMEOUT_SECONDS)
            _run(
                ["git", "-C", str(target), "fetch", "--depth=1", GITHUB_URL, str(plan["commit"])],
                cwd=root,
                timeout=remain,
            )
            break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            detail = str(getattr(exc, "output", "") or "").lower()
            outage = (
                isinstance(exc, subprocess.TimeoutExpired)
                or any(
                    marker in detail
                    for marker in (
                        "could not resolve",
                        "failed to connect",
                        "connection timed out",
                        "connection reset",
                        "http 5",
                        "service unavailable",
                    )
                )
                or re.search(r"requested url returned error: 5\d\d", detail) is not None
            )
            if not outage:
                shutil.rmtree(target)
                raise ValueError("GitHub fetch failed without an outage; fallback refused") from exc
            remaining = FETCH_TIMEOUT_SECONDS - (time.monotonic() - started)
            if attempt == 2 or remaining <= 0:
                shutil.rmtree(target)
                return {"transport": "fallback_required", "reason": str(exc)[:500]}
            time.sleep(min((2, 5)[attempt], remaining))
        except OSError:
            shutil.rmtree(target)
            raise
    head = _run(["git", "-C", str(target), "rev-parse", "FETCH_HEAD"], cwd=root, timeout=5).strip()
    tree = _run(
        ["git", "-C", str(target), "rev-parse", "FETCH_HEAD^{tree}"], cwd=root, timeout=5
    ).strip()
    if head != plan["commit"] or tree != plan["tree"]:
        shutil.rmtree(target)
        raise ValueError("GitHub source identity does not match pinned commit/tree")
    _run(["git", "-C", str(target), "checkout", "--detach", "--quiet", head], cwd=root, timeout=10)
    return {"transport": "github", "commit": head, "tree": tree}


def install_fallback(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Install only the exact hash-bound sparse inventory from the fallback archive."""
    root = root.resolve()
    archive = root / str(plan.get("fallback_archive", ARCHIVE_NAME))
    if not archive.is_file() or _sha256(archive) != plan["fallback_sha256"]:
        raise ValueError("fallback archive SHA-256 mismatch")
    expected: dict[str, dict[str, str]] = plan["files"]
    target = root / "repo"
    if target.exists():
        raise ValueError("runtime source destination already exists")
    target.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError("fallback archive inventory differs from source plan")
            for member in members:
                name = _safe_path(member.name)
                spec = expected[name]
                if name != member.name or not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"unsafe fallback member: {member.name!r}")
                if member.size != spec["size"] or member.mode != int(spec["mode"], 8):
                    raise ValueError(f"fallback metadata mismatch: {name}")
                if member.size > MAX_FALLBACK_BYTES:
                    raise ValueError("fallback member exceeds size cap")
                payload = source.extractfile(member)
                if payload is None:
                    raise ValueError("unreadable fallback member")
                data = payload.read()
                if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                    raise ValueError(f"fallback digest mismatch: {name}")
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                destination.chmod(int(spec["mode"], 8) & 0o777)
    except Exception:
        shutil.rmtree(target)
        raise
    return {
        "transport": "fallback",
        "commit": plan["commit"],
        "tree": plan["tree"],
        "inventory_verified": True,
    }
