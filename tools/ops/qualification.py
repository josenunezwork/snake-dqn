#!/usr/bin/env python3
"""Check the qualified transport Git tree without allocating a runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

OPS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops")
TRANSPORT = "research/colab_cli"


def git(repo: Path, *args: str) -> str:
    """Run a bounded Git inspection."""
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE, timeout=30
    ).strip()


def tree(repo: Path, ref: str = "HEAD") -> str:
    """Resolve a transport subtree, requiring a tree rather than a blob."""
    value = git(repo, "rev-parse", "--verify", f"{ref}:{TRANSPORT}")
    if git(repo, "cat-file", "-t", value) != "tree":
        raise ValueError("transport path is not a Git tree")
    return value


def clean_transport(repo: Path) -> None:
    """Refuse staged, unstaged, and nonignored untracked transport changes."""
    if git(repo, "status", "--porcelain", "--untracked-files=all", "--", TRANSPORT):
        raise ValueError("broker transport has working-tree changes")


def check(repo: Path, record: Path, ref: str = "HEAD", *, runtime: bool = False) -> str:
    """Require the recorded tree, independent of the enclosing commit."""
    saved = json.loads(record.read_text())
    if not isinstance(saved, dict):
        raise ValueError("invalid qualification registry")
    expected = saved.get("tree_hash", "")
    if (
        not isinstance(expected, str)
        or saved.get("schema") != "snake-dqn-colab-cli-qualified/v1"
        or not re.fullmatch(r"[0-9a-f]{40}", expected)
    ):
        raise ValueError("invalid qualification registry")
    actual = tree(repo, ref)
    if actual != expected and runtime and re.fullmatch(r"[0-9a-f]{40}", ref):
        # Only frozen runtime pins may use the current registry's hash-bound
        # previous qualification. HEAD/remote broker checks never use history.
        previous = saved.get("previous_qualification", {})
        path = Path(previous.get("path", ""))
        if path.is_absolute() and not path.is_symlink() and path.is_file():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() == previous.get("sha256"):
                historical = json.loads(raw)
                if historical.get("schema") == "snake-dqn-colab-cli-qualified/v1":
                    expected = historical.get("tree_hash")
    if actual != expected:
        raise ValueError(f"COLAB_CLI_UNQUALIFIED ref={ref} tree={actual} qualified={expected}")
    if ref == "HEAD":
        clean_transport(repo)
    return actual


def save(record: Path, value: dict[str, Any]) -> None:
    """Atomically replace only the mutable ops registry, never an acceptance receipt."""
    record.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".qualified-", dir=record.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, record)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    """Expose the same check to dispatch and watchd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    try:
        if args.fetch:
            git(args.repo, "fetch", "--quiet", "origin", "main")
        print(check(args.repo, args.record, args.ref, runtime=args.runtime))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: COLAB_CLI_UNQUALIFIED {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
