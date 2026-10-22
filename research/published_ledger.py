"""Publication fence for registered analysis claims.

The worker that computes a result must not also open its registered look.  A
claim is made only from a clean authoring checkout after the selected ledger's
local bytes have been compared with its freshly fetched ``origin/main`` bytes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


class PublishedLedgerRefused(RuntimeError):
    """The authoring checkout cannot safely open a registered analysis claim."""


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PublishedLedgerRefused(
            f"published-ledger git check failed: {' '.join(args)}"
        ) from exc


def _ledger_relative(value: str | Path) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PublishedLedgerRefused("ledger must be a repository-relative path")
    return path.as_posix()


def _canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def require_published_ledger(authoring_repo: str | Path, ledger: str | Path) -> dict[str, str]:
    """Return a hash-bound publication proof or refuse before a look is consumed.

    ``authoring_repo`` is intentionally explicit.  Analysis runners execute in
    detached upload/runtime directories, where a local file cannot establish
    that its ledger was published.  Fetching only refreshes the tracking ref;
    this helper never writes a worktree file or creates a claim.
    """
    repo = Path(authoring_repo).resolve()
    if not repo.is_dir():
        raise PublishedLedgerRefused("authoring repository is not a directory")
    top = Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != repo:
        raise PublishedLedgerRefused("authoring repository must be the checkout root")
    relative = _ledger_relative(ledger)
    local = repo / relative
    if local.is_symlink() or not local.is_file():
        raise PublishedLedgerRefused("local ledger is missing or unsafe")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise PublishedLedgerRefused("authoring checkout has tracked or untracked changes")
    initial_head = _git(repo, "rev-parse", "HEAD").decode().strip()
    _git(
        repo, "fetch", "--quiet", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main"
    )
    origin_main = _git(repo, "rev-parse", "origin/main").decode().strip()
    try:
        published = _git(repo, "show", f"origin/main:{relative}")
    except PublishedLedgerRefused as exc:
        raise PublishedLedgerRefused("ledger is not published on origin/main") from exc
    local_bytes = local.read_bytes()
    if local_bytes != published:
        raise PublishedLedgerRefused("local ledger bytes differ from origin/main")
    # Fetch can overlap with another authoring process. Recheck all local
    # facts after reading the published bytes, before any look is consumed.
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise PublishedLedgerRefused("authoring checkout changed during published-ledger check")
    if _git(repo, "rev-parse", "HEAD").decode().strip() != initial_head:
        raise PublishedLedgerRefused(
            "authoring checkout HEAD changed during published-ledger check"
        )
    if local.read_bytes() != published:
        raise PublishedLedgerRefused("local ledger changed during published-ledger check")
    if _git(repo, "rev-parse", "origin/main").decode().strip() != origin_main:
        raise PublishedLedgerRefused("origin/main changed during published-ledger check")
    proof: dict[str, str] = {
        "schema": "published-ledger-check/v1",
        "ledger": relative,
        "ledger_sha256": hashlib.sha256(local_bytes).hexdigest(),
        "authoring_head": initial_head,
        "origin_main": origin_main,
    }
    proof["proof_sha256"] = _canonical_sha(proof)
    return proof


def bind_published_ledger_check(receipt: dict[str, Any], proof: dict[str, str]) -> dict[str, Any]:
    """Attach the exact pre-claim proof to a newly created analysis receipt."""
    if proof.get("schema") != "published-ledger-check/v1":
        raise PublishedLedgerRefused("invalid published-ledger proof")
    unsigned = {key: value for key, value in proof.items() if key != "proof_sha256"}
    if proof.get("proof_sha256") != _canonical_sha(unsigned):
        raise PublishedLedgerRefused("published-ledger proof digest differs")
    if "published_ledger_check" in receipt or "published_ledger_check_sha256" in receipt:
        raise PublishedLedgerRefused("analysis receipt already has a ledger check")
    return {
        **receipt,
        "published_ledger_check": proof,
        "published_ledger_check_sha256": proof["proof_sha256"],
    }
