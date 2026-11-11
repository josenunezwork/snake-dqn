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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

OPS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops")
TRANSPORT = "research/colab_cli"
V1 = "snake-dqn-colab-cli-qualified/v1"
V2 = "snake-dqn-colab-cli-qualified/v2"
KNOWN_LEGACY_CAP6_TREES = frozenset(
    {
        "3a7b2188e3afd47a57ed87cff46d95deb49146aa",
        "d2992d6a24dd548d2be1cd546b324e33aeb454cb",
        "36285bc6cbb637b392b234f67b55cbaa03b37614",
    }
)
_GIT_TRANSPORT_CACHE: dict[tuple[str, str], tuple[str, int]] = {}
_GIT_TREE_CAP_CACHE: dict[tuple[str, str], int] = {}


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
            cap = value.get("account_assignment_cap")
            if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
                raise ValueError("publish requires positive integer account_assignment_cap")
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


def _git_tree_cap(repo: Path, tree_hash: str) -> int | None:
    """Resolve ACCOUNT_ASSIGNMENT_CAP from <tree_hash>:accounting.py, caching only successes."""
    key = (str(repo.resolve()), tree_hash)
    if key in _GIT_TREE_CAP_CACHE:
        return _GIT_TREE_CAP_CACHE[key]
    try:
        src = git(repo, "show", f"{tree_hash}:accounting.py")
        match = re.search(r"^ACCOUNT_ASSIGNMENT_CAP\s*=\s*(\d+)\b", src, re.M)
        if match is not None:
            cap = int(match.group(1))
            _GIT_TREE_CAP_CACHE[key] = cap
            return cap
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _git_transport_info(repo: Path, commit: str) -> tuple[str | None, int | None]:
    """Resolve (tree_hash, accounting_cap) for commit in repo without caching failures."""
    key = (str(repo.resolve()), commit)
    if key in _GIT_TRANSPORT_CACHE:
        return _GIT_TRANSPORT_CACHE[key]
    try:
        tree_hash = tree(repo, commit)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None
    cap = _git_tree_cap(repo, tree_hash)
    if cap is not None:
        _GIT_TRANSPORT_CACHE[key] = (tree_hash, cap)
    return tree_hash, cap


def _registry_row_cap(row: dict[str, Any]) -> int | None:
    """Return explicit positive cap or known legacy cap=6; never infer 6 from v2 schema."""
    raw = row.get("account_assignment_cap")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    if row.get("tree_hash") in KNOWN_LEGACY_CAP6_TREES:
        return 6
    return None


def tree_transport_cap(
    tree_hash: Any,
    *,
    ops: Path,
    repos: Iterable[Path] = (),
) -> int | None:
    """Return the transport tree's account-wide assignment cap for tree_hash, if determinable."""
    if not isinstance(tree_hash, str) or not re.fullmatch(r"[0-9a-f]{40}", tree_hash):
        return None
    record = ops / "colab-cli-qualified.json"
    if record.is_file():
        try:
            rows = entries(json.loads(record.read_text()))
        except (OSError, ValueError):
            rows = []
        for row in rows:
            if row["tree_hash"] == tree_hash:
                row_cap = _registry_row_cap(row)
                if row_cap is not None:
                    return row_cap
    if tree_hash in KNOWN_LEGACY_CAP6_TREES:
        return 6
    seen_repos: set[str] = set()
    for candidate in [*repos, Path(__file__).resolve().parents[2]]:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if str(resolved) in seen_repos or not (resolved / ".git").exists():
            continue
        seen_repos.add(str(resolved))
        code_cap = _git_tree_cap(resolved, tree_hash)
        if code_cap is not None:
            return code_cap
    return None


def commit_transport_cap(
    commit: Any,
    *,
    ops: Path,
    repos: Iterable[Path] = (),
) -> int | None:
    """Return the transport tree's account-wide assignment cap for commit, if determinable."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    record = ops / "colab-cli-qualified.json"
    rows: list[dict[str, Any]] = []
    if record.is_file():
        try:
            rows = entries(json.loads(record.read_text()))
        except (OSError, ValueError):
            rows = []
    for row in rows:
        if row.get("qualified_commit") == commit:
            row_cap = _registry_row_cap(row)
            if row_cap is not None:
                return row_cap
    seen_repos: set[str] = set()
    candidates = [*repos, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if str(resolved) in seen_repos or not (resolved / ".git").exists():
            continue
        seen_repos.add(str(resolved))
        tree_hash, code_cap = _git_transport_info(resolved, commit)
        if tree_hash is None:
            continue
        for row in rows:
            if row["tree_hash"] == tree_hash:
                row_cap = _registry_row_cap(row)
                if row_cap is not None:
                    return row_cap
        if tree_hash in KNOWN_LEGACY_CAP6_TREES:
            return 6
        if code_cap is not None:
            return code_cap
    return None


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
