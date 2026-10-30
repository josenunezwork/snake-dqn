#!/usr/bin/env python3
"""Check the qualified transport Git tree without allocating a runtime."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

OPS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops")
TRANSPORT = "research/colab_cli"
V1 = "snake-dqn-colab-cli-qualified/v1"
V2 = "snake-dqn-colab-cli-qualified/v2"


def entries(saved: dict[str, Any]) -> list[dict[str, Any]]:
    """Read legacy singleton records and explicit multi-tree registries."""
    if not isinstance(saved, dict):
        raise ValueError("invalid qualification registry")
    if saved.get("registry_schema") == V2 or saved.get("schema") == V2:
        rows = saved.get("trees")
        if not isinstance(rows, list) or not rows:
            raise ValueError("invalid qualification registry trees")
    elif saved.get("schema") == V1:
        rows = [saved]
    else:
        raise ValueError("invalid qualification registry schema")
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema") != V1
            or not isinstance(row.get("tree_hash"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", row["tree_hash"])
            or row["tree_hash"] in seen
            or row.get("state", "qualified") not in {"qualified", "retired"}
        ):
            raise ValueError("invalid qualification registry entry")
        seen.add(row["tree_hash"])
    return rows


def select(saved: dict[str, Any], actual: str, *, offline: bool = False) -> dict[str, Any]:
    """Select one reviewed, non-retired tree; never borrow another tree's approval."""
    rows = entries(saved)
    row = next((row for row in rows if row["tree_hash"] == actual), None)
    if row is None or row.get("state") == "retired":
        raise ValueError(f"COLAB_CLI_UNQUALIFIED tree={actual}")
    status = row.get("independent_review_status")
    if status != "ACCEPT" and not (offline and status == "PENDING"):
        raise ValueError(f"COLAB_CLI_INDEPENDENT_REVIEW_REQUIRED status={status!r}")
    return row


def publish(record: Path, value: dict[str, Any], *, retire: str | None = None) -> None:
    """Add a qualification or explicitly retire a tree, with locked backup and replace."""
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.with_suffix(record.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = record.read_bytes() if record.exists() else None
        rows = [dict(row) for row in entries(json.loads(old))] if old else []
        if retire is not None:
            row = next((row for row in rows if row["tree_hash"] == retire), None)
            if row is None or not value.get("reason"):
                raise ValueError("retirement requires an existing tree and a reason")
            row.update(state="retired", retirement=value)
        else:
            entries(value)
            if value.get("schema") != V1:
                raise ValueError("publish requires one tree qualification")
            if any(row["tree_hash"] == value["tree_hash"] for row in rows):
                raise ValueError(
                    "tree already exists; retirement or review update must be explicit"
                )
            rows.append(value)
        if old is not None:
            backup = record.with_name(record.name + ".backup-" + uuid.uuid4().hex)
            with backup.open("xb") as stream:
                stream.write(old)
                stream.flush()
                os.fsync(stream.fileno())
        # Keep the original primary as a legacy v1 view while it is qualified.
        # Historical dispatchers queued on that tree can still read this file.
        primary = next((row for row in rows if row.get("state") != "retired"), rows[0])
        legacy = {**primary, "registry_schema": V2, "trees": rows}
        if primary.get("state") == "retired":
            legacy["independent_review_status"] = "REJECT"
        save(record, legacy)


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


def check(repo: Path, record: Path, ref: str = "HEAD", *, offline: bool = False) -> str:
    """Require the recorded tree, independent of the enclosing commit."""
    actual = tree(repo, ref)
    select(json.loads(record.read_text()), actual, offline=offline)
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
        directory = os.open(record.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    """Expose the same check to dispatch and watchd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "add", "retire"])
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--entry", type=Path)
    parser.add_argument("--tree")
    parser.add_argument("--reason")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--offline", action="store_true", help="read-only pending-review validation"
    )
    args = parser.parse_args()
    try:
        if args.action == "add":
            if args.entry is None:
                raise ValueError("add requires --entry")
            publish(args.record, json.loads(args.entry.read_text()))
            return 0
        if args.action == "retire":
            if not args.tree:
                raise ValueError("retire requires --tree")
            publish(args.record, {"reason": args.reason}, retire=args.tree)
            return 0
        if args.repo is None:
            raise ValueError("check requires --repo")
        if args.fetch:
            git(args.repo, "fetch", "--quiet", "origin", "main")
        print(check(args.repo, args.record, args.ref, offline=args.offline))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: COLAB_CLI_UNQUALIFIED {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
