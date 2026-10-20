"""Append-only published terminal history for the fixed Mac retention study.

The registered authoring checkout owns the ledger. The detached executable
checkout stays clean. Every admission fetches origin/main and compares the
published ledger with all retained local phase seals and their receipt bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


class RegistrationError(RuntimeError):
    """A retention registration or its immutable history cannot admit work."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_policy(registration: Mapping[str, Any]) -> tuple[Path, str]:
    """Validate a fixed repo-relative ledger name and separate authoring root."""
    item = registration.get("history_ledger")
    study = registration.get("study_root")
    if not isinstance(item, dict) or set(item) != {"repo", "path"} or not isinstance(study, str):
        raise RegistrationError("registered history_ledger repo and path are required")
    repo, name = item["repo"], item["path"]
    if (
        not isinstance(repo, str)
        or not Path(repo).is_absolute()
        or str(Path(repo).resolve()) != repo
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", Path(study).name)
        or name != f"research/t34_pathb/ledger/{Path(study).name}.jsonl"
    ):
        raise RegistrationError(
            "history ledger must use the registered study ID in its authoring repo"
        )
    return Path(repo), name


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RegistrationError(
            f"history publication verification failed: {' '.join(args)}"
        ) from exc


def _entries(text: str) -> list[dict[str, Any]]:
    try:
        result = [json.loads(line) for line in text.splitlines()]
        if any(not isinstance(entry, dict) for entry in result):
            raise ValueError("ledger entries must be objects")
        return result
    except (ValueError, TypeError) as exc:
        raise RegistrationError("malformed terminal history ledger") from exc


def _local_ledger(repo: Path, name: str) -> str:
    path = repo / name
    if path.resolve() != path or path.is_symlink():
        raise RegistrationError("history ledger cannot use links")
    return path.read_text() if path.exists() else ""


def _published_ledger(repo: Path, name: str) -> str:
    if _git(repo, "rev-parse", "--show-toplevel").strip() != str(repo):
        raise RegistrationError("history ledger repo is not its own Git checkout")
    # Explicit refspec prevents a stale remote-tracking main or alternate fetch
    # configuration from authorizing another phase. Network failure refuses.
    _git(repo, "fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main")
    names = _git(repo, "ls-tree", "--name-only", "origin/main", "--", name).splitlines()
    current = _git(repo, "show", f"origin/main:{name}") if names == [name] else ""
    previous = ""
    # Verify published history itself is append-only, including deletions.
    for commit in reversed(
        _git(repo, "log", "--format=%H", "origin/main", "--", name).splitlines()
    ):
        present = _git(repo, "ls-tree", "--name-only", commit, "--", name).splitlines()
        snapshot = _git(repo, "show", f"{commit}:{name}") if present == [name] else ""
        if not snapshot.startswith(previous) or (snapshot and not snapshot.endswith("\n")):
            raise RegistrationError("origin/main terminal ledger history is not append-only")
        _entries(snapshot)
        previous = snapshot
    if current != previous:
        raise RegistrationError("origin/main terminal ledger history was removed")
    return current


def _phase_roots(registration: Mapping[str, Any]) -> list[Path]:
    root = Path(registration["study_root"])
    timing = root / "timing"
    roots = sorted(timing.iterdir()) if timing.exists() else []
    for phase in ("precheck", "efficacy"):
        if (root / phase).exists() or (root / phase).is_symlink():
            roots.append(root / phase)
    return roots


def terminal_entry(out: Path, registration: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a terminal seal and every file it binds, without trusting labels."""
    seal = out / "phase_terminal.json"
    try:
        if out.resolve() != out or seal.is_symlink() or not out.is_dir():
            raise ValueError("linked phase history")
        payload = json.loads(seal.read_text())
        if (
            payload.get("schema") != "t34-retention-terminal/v1"
            or payload.get("phase_path") != str(out)
            or payload.get("registration_sha256") != _registration_sha(registration)
            or not isinstance(payload.get("files"), dict)
            or not payload["files"]
        ):
            raise ValueError("foreign or incomplete terminal seal")
        actual = {}
        for path in out.rglob("*"):
            if path.is_symlink():
                raise ValueError("linked phase evidence")
            if path.is_file() and path != seal:
                actual[path.relative_to(out).as_posix()] = _sha(path)
        if actual != payload["files"]:
            raise ValueError("terminal receipt files changed or disappeared")
        return {"phase_path": str(out), "terminal_sha256": _sha(seal)}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise RegistrationError(f"missing or changed terminal history: {out}: {exc}") from exc


def _registration_sha(registration: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(registration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_published_history(registration: Mapping[str, Any]) -> None:
    """Refuse erased, altered, unpublished, or locally unfinished phase history."""
    repo, name = ledger_policy(registration)
    if repo == Path(__file__).resolve().parents[2]:
        raise RegistrationError(
            "ledger authoring checkout must differ from frozen execution checkout"
        )
    try:
        published = _published_ledger(repo, name)
        local = _local_ledger(repo, name)
        if local != published:
            raise RegistrationError("terminal ledger must be committed and pushed to origin/main")
        expected = [terminal_entry(out, registration) for out in _phase_roots(registration)]
        if _entries(published) != expected:
            raise RegistrationError(
                "origin/main ledger differs from retained local terminal history"
            )
    except OSError as exc:
        raise RegistrationError(f"unreadable terminal history: {exc}") from exc


def finalize_phase(out: Path, registration: Mapping[str, Any]) -> None:
    """Create a seal after source-fence closure, then append its hash durably."""
    repo, name = ledger_policy(registration)
    ledger = repo / name
    if repo == Path(__file__).resolve().parents[2]:
        raise RegistrationError(
            "ledger authoring checkout must differ from frozen execution checkout"
        )
    # Exclusive append under a Git-common-dir lock avoids lost writes across
    # callers, without placing mutable lock files inside either checkout.
    import fcntl

    common = Path(_git(repo, "rev-parse", "--git-common-dir").strip())
    if not common.is_absolute():
        common = repo / common
    with (common / "t34-retention-history.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prior = _entries(_local_ledger(repo, name))
        roots = _phase_roots(registration)
        if not roots or roots[-1] != out:
            raise RegistrationError("terminal phase is not the latest fixed phase directory")
        expected = [terminal_entry(path, registration) for path in roots[:-1]]
        if prior != expected:
            raise RegistrationError("append-only terminal ledger prefix differs from local history")
        seal = out / "phase_terminal.json"
        if seal.exists():
            raise RegistrationError("phase terminal seal already exists")
        files = {
            path.relative_to(out).as_posix(): _sha(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        }
        if any(path.is_symlink() for path in out.rglob("*")) or not files:
            raise RegistrationError("cannot seal linked or empty phase evidence")
        with seal.open("x") as handle:
            json.dump(
                {
                    "schema": "t34-retention-terminal/v1",
                    "phase_path": str(out),
                    "registration_sha256": _registration_sha(registration),
                    "files": files,
                },
                handle,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
        entry = terminal_entry(out, registration)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        if ledger.resolve() != ledger:
            raise RegistrationError("history ledger cannot use links")
        with ledger.open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def claim_phase(out: Path, registration: Mapping[str, Any]) -> None:
    """Publish a create-only admission tag before any gameplay can begin.

    A killed process cannot append a terminal receipt. This remote claim makes
    that unfinished invocation durable even if every local artifact is erased.
    It grants no retry; absence of a measured terminal cause means Claim B KILL.
    """
    import uuid

    repo, _ = ledger_policy(registration)
    root = Path(registration["study_root"])
    relative = out.relative_to(root).as_posix()
    if not re.fullmatch(r"precheck|efficacy|timing/attempt-0[1-3]", relative):
        raise RegistrationError("admission claim must use a fixed phase directory")
    source = registration.get("source_commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise RegistrationError("admission claim requires the registered source commit")
    payload = json.dumps(
        {
            "schema": "t34-retention-admission/v1",
            "phase_path": str(out),
            "registration_sha256": _registration_sha(registration),
            "interruption_without_measured_terminal_cause": "KILL",
            "nonce": uuid.uuid4().hex,
        },
        sort_keys=True,
    )
    # A unique commit per invocation makes a repeated tag push fail even if
    # registration and source identity have not changed. No ref is force-pushed.
    commit = _git(
        repo,
        "-c",
        "user.name=T34 retention admission",
        "-c",
        "user.email=t34-retention@localhost",
        "commit-tree",
        f"{source}^{{tree}}",
        "-p",
        source,
        "-m",
        payload,
    ).strip()
    ref = f"refs/tags/t34-retention/{root.name}/{relative}"
    try:
        _git(repo, "push", "--porcelain", "origin", f"{commit}:{ref}")
    except RegistrationError as exc:
        raise RegistrationError(
            "remote phase admission claim exists or could not be published; "
            "unmeasured unfinished admission is Claim B KILL; no reroll"
        ) from exc
