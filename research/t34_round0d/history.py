"""Create-only Round 0-D reservations and published receipt-hash history.

This is Round 0's integrity machinery (``research/t34_round0/history.py``) with
only the study identities changed: remote claims live under
``refs/tags/t34-round0d/<study-root>/<phase>`` (never matched by Round 0's
``refs/tags/t34-round0/*`` admission glob), the ledger is
``research/t34_round0d/ledger/<study-root>.jsonl``, and the fixed phases add
the second-L4 ``replay`` job. The Round 0 module hard-codes its own tag prefix
and ledger path, so it cannot be imported for these functions without
modifying it; generic primitives (ledger publication checks, sampling) are
imported unchanged.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Mapping

from research.t34_pathb.retention_history_v2 import (
    _entries,
    _git,
    _local_ledger,
    _published_ledger,
    _registration_sha,
)
from research.t34_round0.history import selected_rows
from research.t34_round0d.common import LEDGER_DIR, ROOT, TAG_PREFIX, read_json, sha256, write_once

__all__ = [
    "CapExpiredHistory",
    "HistoryError",
    "finalize",
    "phase_name",
    "policy",
    "require_active_phase",
    "require_chunks",
    "require_published_history",
    "reserve",
    "run_phase",
    "selected_rows",
]

_ACTIVE_PHASE: ContextVar[Path | None] = ContextVar("round0d_active_phase", default=None)
_LOCK_DEPTH: ContextVar[int] = ContextVar("round0d_history_lock_depth", default=0)
TERMINAL_SCHEMA = "t34-round0d-terminal/v1"
ADMISSION_SCHEMA = "t34-round0d-admission/v1"
STATUSES = {"COMPLETE", "MECHANICAL_FAILURE", "INCONCLUSIVE", "KILL"}


class HistoryError(ValueError):
    """An immutable reservation or its published evidence is unsafe."""


class CapExpiredHistory(HistoryError):
    """A registered first cap ended the study without an outcome look."""


@contextmanager
def history_lock(registration: Mapping[str, Any]) -> Iterator[None]:
    """Serialize history checks, reservations and seals across concurrent lane dispatchers."""
    if _LOCK_DEPTH.get():  # re-entrant within one dispatcher (e.g. seal under the lock)
        yield
        return
    _, repo, _ = policy(registration)
    common = Path(_git(repo, "rev-parse", "--git-common-dir").strip())
    if not common.is_absolute():
        common = repo / common
    with (common / "t34-round0d-history.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        token = _LOCK_DEPTH.set(1)
        try:
            yield
        finally:
            _LOCK_DEPTH.reset(token)
            fcntl.flock(lock, fcntl.LOCK_UN)


class InflightClaim:
    """An attempt's liveness lock, held by its dispatcher from reservation to seal.

    Round 0 dispatches serially, so any unsealed claim is an interruption. Round 0-D
    runs two lanes; an unsealed attempt whose lock is still held is in flight and
    is skipped by history checks, while an unsealed attempt whose lock is free is
    exactly Round 0's unfinished claim (KILL).
    """

    def __init__(self, out: Path) -> None:
        self.path = out / "inflight.lock"
        self.handle = self.path.open("x")
        fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        if not self.handle.closed:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def in_flight(out: Path) -> bool:
    """Return whether another live dispatcher holds this attempt's liveness lock."""
    path = out / "inflight.lock"
    if not path.is_file() or (out / "phase_terminal.json").exists():
        return False
    # Shared probe: concurrent probes never block each other, so a dead attempt
    # cannot read as in flight because another probe happens to hold the file.
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def in_flight_count(registration: Mapping[str, Any]) -> int:
    """Number of chunk attempts whose dispatcher currently holds its liveness lock."""
    root, _, _ = policy(registration)
    return sum(in_flight(a) for a in (root / "chunks").glob("chunk-*/attempt-0[12]"))


def require_active_phase(registration: Mapping[str, Any], phase: str) -> None:
    """Prevent unrecorded outcome looks through imported consumer APIs."""
    root, _, _ = policy(registration)
    if _ACTIVE_PHASE.get() != root / phase:
        raise HistoryError("outcome consumer requires its create-only reserved phase")


def policy(registration: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """Require canonical absolute roots and the study-specific repo ledger."""
    root = Path(registration.get("study_root", ""))
    item = registration.get("history_ledger", {})
    repo = Path(item.get("repo", ""))
    name = item.get("path")
    if (
        not root.is_absolute()
        or root.resolve() != root
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", root.name)
        or not repo.is_absolute()
        or repo.resolve() != repo
        or name != f"{LEDGER_DIR}/{root.name}.jsonl"
    ):
        raise HistoryError("ratification requires canonical study_root and authoring ledger")
    return root, repo, name


def phase_name(out: Path, registration: Mapping[str, Any]) -> str:
    """Accept only fixed chunk attempts, the replay job and the one analysis."""
    root, _, _ = policy(registration)
    if out.resolve() != out or not out.is_relative_to(root):
        raise HistoryError("phase must be an unlinked path under the fixed study root")
    relative = out.relative_to(root).as_posix()
    if not re.fullmatch(r"chunks/chunk-\d{3}/attempt-0[12]|ceiling|replay|analysis", relative):
        raise HistoryError("phase must use a fixed chunk/attempt or replay/analysis path")
    return relative


def _roots(registration: Mapping[str, Any]) -> list[Path]:
    root, _, _ = policy(registration)
    found = sorted((root / "chunks").glob("*/*")) if (root / "chunks").exists() else []
    for path in found:
        phase_name(path, registration)
    return found + [root / p for p in ("ceiling", "replay", "analysis") if (root / p).exists()]


def _claims(registration: Mapping[str, Any]) -> dict[str, str]:
    root, repo, _ = policy(registration)
    prefix = f"refs/tags/{TAG_PREFIX}/{root.name}/"
    output = _git(repo, "ls-remote", "--refs", "origin", f"{prefix}*")
    return {
        ref[len(prefix) :]: commit for commit, ref in (line.split() for line in output.splitlines())
    }


def _reject_other_study_roots(registration: Mapping[str, Any]) -> None:
    """A Round 0-D remote claim binds the one root frozen in the ratification."""
    root, repo, _ = policy(registration)
    prefix = f"refs/tags/{TAG_PREFIX}/"
    output = _git(repo, "ls-remote", "--refs", "origin", f"{prefix}*")
    for line in output.splitlines():
        _commit, ref = line.split()
        tag_root, separator, _phase = ref.removeprefix(prefix).partition("/")
        if not separator or tag_root != root.name:
            raise HistoryError("remote Round 0-D admission belongs to a different study_root")


def terminal_entry(out: Path, registration: Mapping[str, Any]) -> dict[str, str]:
    """Check the complete immutable inventory, including supervisor receipts."""
    relative = phase_name(out, registration)
    seal = out / "phase_terminal.json"
    try:
        value = read_json(seal)
    except FileNotFoundError as exc:
        raise HistoryError("unfinished claim is missing its terminal history") from exc
    if (
        value.get("schema") != TERMINAL_SCHEMA
        or value.get("phase_path") != str(out)
        or value.get("registration_sha256") != _registration_sha(registration)
        or value.get("status") not in STATUSES
    ):
        raise HistoryError("foreign or unfinished terminal history")
    paths = list(out.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise HistoryError("linked terminal history")
    files = {p.relative_to(out).as_posix(): sha256(p) for p in paths if p.is_file() and p != seal}
    if not files or files != value.get("files"):
        raise HistoryError("terminal evidence changed, disappeared, or was added")
    return {"phase": relative, "terminal_sha256": sha256(seal)}


def require_published_history(registration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fetch and validate append-only history and every durable remote claim."""
    _, repo, name = policy(registration)
    if repo == ROOT:
        raise HistoryError("authoring checkout must differ from frozen execution checkout")
    local, published = _local_ledger(repo, name), _published_ledger(repo, name)
    if local != published:
        raise HistoryError("receipt-hash ledger must be committed and pushed before continuing")
    entries = _entries(published)
    roots = _roots(registration)
    settled = [out for out in roots if out != _ACTIVE_PHASE.get() and not in_flight(out)]
    actual = [terminal_entry(out, registration) for out in settled]
    if len({e.get("phase") for e in entries}) != len(entries) or sorted(
        entries, key=lambda e: e["phase"]
    ) != sorted(actual, key=lambda e: e["phase"]):
        raise HistoryError("published ledger differs from retained fixed phase history")
    local_claims = {}
    for out in roots:
        admission = read_json(out / "admission.json")
        if admission.get("registration_sha256") != _registration_sha(registration):
            raise HistoryError("foreign admission claim")
        local_claims[phase_name(out, registration)] = admission.get("claim_commit")
    if _claims(registration) != local_claims:
        raise HistoryError("unfinished or erased remote admission is KILL; no reset or reroll")
    return [read_json(out / "phase_terminal.json") for out in settled]


def reserve(
    out: Path, registration: Mapping[str, Any], *, world_ids: list[str] | None = None
) -> None:
    """Publish a unique create-only tag before reserving local phase evidence."""
    relative = phase_name(out, registration)
    root, repo, _ = policy(registration)
    _reject_other_study_roots(registration)
    if out.exists():
        raise HistoryError("phase directory is create-only")
    if world_ids is not None and (
        not world_ids
        or any(not isinstance(world_id, str) or not world_id for world_id in world_ids)
        or len(set(world_ids)) != len(world_ids)
    ):
        raise HistoryError("chunk admission world_ids must be non-empty unique strings")
    source = registration.get("commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise HistoryError("admission requires a full ratified source commit")
    payload: dict[str, Any] = {
        "schema": ADMISSION_SCHEMA,
        "phase_path": str(out),
        "registration_sha256": _registration_sha(registration),
        "unexplained_interruption": "KILL",
        "dispatched_at_unix_seconds": time.time(),
        "nonce": uuid.uuid4().hex,
    }
    if world_ids is not None:
        payload["world_ids"] = world_ids
    commit = _git(
        repo,
        "-c",
        "user.name=T34 Round0D admission",
        "-c",
        "user.email=t34-round0d@localhost",
        "commit-tree",
        f"{source}^{{tree}}",
        "-p",
        source,
        "-m",
        json.dumps(payload, sort_keys=True),
    ).strip()
    _git(
        repo,
        "push",
        "--porcelain",
        "origin",
        f"{commit}:refs/tags/{TAG_PREFIX}/{root.name}/{relative}",
    )
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "admission.json", dict(payload, claim_commit=commit))


def finalize(out: Path, registration: Mapping[str, Any], status: str, reason: str) -> None:
    """Seal after closing the source fence and durably append the receipt hash."""
    _, repo, name = policy(registration)
    if status not in STATUSES:
        raise HistoryError("unknown phase terminal status")
    with history_lock(registration):
        prior = _entries(_local_ledger(repo, name))
        if any(e.get("phase") == phase_name(out, registration) for e in prior):
            raise HistoryError("phase already appears in the append-only ledger")
        for entry in prior:
            if (
                terminal_entry(Path(registration["study_root"]) / entry["phase"], registration)
                != entry
            ):
                raise HistoryError("prior terminal history differs")
        if status in {"KILL", "INCONCLUSIVE"} or reason.startswith(
            "explained unconditional controller fault"
        ):
            write_once(out / "study_closed.json", {"status": status})
        if status == "COMPLETE" and phase_name(out, registration).startswith("chunks/"):
            write_once(out / "schedule_complete.json", {"status": status})
        paths = list(out.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise HistoryError("cannot seal linked evidence")
        files = {p.relative_to(out).as_posix(): sha256(p) for p in paths if p.is_file()}
        write_once(
            out / "phase_terminal.json",
            {
                "schema": TERMINAL_SCHEMA,
                "phase_path": str(out),
                "registration_sha256": _registration_sha(registration),
                "status": status,
                "reason": reason,
                "files": files,
            },
        )
        ledger = repo / name
        if ledger.resolve() != ledger:
            raise HistoryError("ledger cannot use links")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as handle:
            handle.write(json.dumps(terminal_entry(out, registration), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def seal_ceiling(registration: Mapping[str, Any], spend: Mapping[str, Any]) -> None:
    """Settle a schedule that settled charges alone make infeasible: sealed INCONCLUSIVE.

    Called by dispatch under ``history_lock`` with no attempt in flight. The
    create-only ``ceiling`` phase is claimed, records the spend arithmetic only
    (no outcome), and is sealed INCONCLUSIVE so analysis settles INCONCLUSIVE via
    ``CapExpiredHistory`` (first cap wins; no extension).
    """
    root, _, _ = policy(registration)
    out = root / "ceiling"
    reserve(out, registration)
    write_once(
        out / "ceiling.json",
        {
            "schema": "t34-round0d-ceiling/v1",
            "status": "INCONCLUSIVE",
            "no_extension": True,
            "spend": dict(spend),
        },
    )
    finalize(
        out, registration, "INCONCLUSIVE", "registered study spend ceiling reached; first cap wins"
    )


def require_chunks(chunks: list[Path], registration: Mapping[str, Any]) -> None:
    """Require the sole admissible completed attempt of every fixed chunk."""
    root, _, _ = policy(registration)
    terminals = require_published_history(registration)
    if any(t["status"] == "KILL" for t in terminals):
        raise HistoryError("operator abort or unexplained interruption is KILL")
    if any(t["status"] == "INCONCLUSIVE" for t in terminals):
        raise CapExpiredHistory("registered first cap expired; INCONCLUSIVE with no extension")
    expected_output = Path("job/exports") / "round0d"
    chunks_root = root / "chunks"
    retained, scientific_stop = [], False
    if chunks_root.exists():
        for group in sorted(chunks_root.iterdir()):
            if not group.is_dir():
                continue
            attempts = sorted(path for path in group.iterdir() if path.is_dir())
            if not attempts:
                continue
            terminal = read_json(attempts[-1] / "phase_terminal.json")
            if terminal.get("status") == "COMPLETE":
                retained.append(attempts[-1] / expected_output)
            scientific_stop = scientific_stop or terminal.get("reason", "").startswith(
                "explained unconditional controller fault"
            )
    if scientific_stop:
        if len(set(chunks)) != len(chunks) or set(chunks) != set(retained):
            raise HistoryError("scientific STOP must consume exactly all retained completed chunks")
        return
    if (root / "analysis").exists() and _ACTIVE_PHASE.get() != root / "analysis":
        raise HistoryError("the sole outcome look has already been reserved")
    frozen_chunks = registration.get("sizing", {}).get("chunks")
    if not isinstance(frozen_chunks, list) or not frozen_chunks:
        raise HistoryError("ratification lacks a fixed chunk schedule")
    expected_names = [f"chunk-{index:03d}" for index in range(len(frozen_chunks))]
    try:
        groups = sorted(chunks_root.iterdir())
    except FileNotFoundError as exc:
        raise HistoryError("unfinished claim is missing its terminal history") from exc
    if any(not g.is_dir() for g in groups) or [g.name for g in groups] != expected_names:
        raise HistoryError("fixed chunk schedule is missing a terminal claim")
    expected = []
    for group in groups:
        attempts = sorted(group.iterdir())
        if [p.name for p in attempts] not in (["attempt-01"], ["attempt-01", "attempt-02"]):
            raise HistoryError("fixed chunks permit at most one fresh mechanical attempt")
        try:
            seals = [read_json(p / "phase_terminal.json") for p in attempts]
        except FileNotFoundError as exc:
            raise HistoryError("unfinished claim is missing its terminal history") from exc
        if len(seals) == 2 and seals[0]["status"] != "MECHANICAL_FAILURE":
            raise HistoryError("second attempt was not an outcome-independent mechanical rerun")
        if seals[-1]["status"] != "COMPLETE":
            raise HistoryError("latest chunk attempt is incomplete")
        expected.append(attempts[-1] / expected_output)
    if len(set(chunks)) != len(chunks) or set(chunks) != set(expected):
        raise HistoryError("supplied chunks differ from the fixed retained attempt history")


def _analysis_reservation_allowed(registration: Mapping[str, Any]) -> None:
    """Refuse a replay/analysis claim while the in-time chunk schedule is still pending."""
    root, _, _ = policy(registration)
    chunks = registration.get("sizing", {}).get("chunks")
    approved_at = registration.get("approved_at")
    if not isinstance(chunks, list) or not chunks or not isinstance(approved_at, str):
        raise HistoryError("ratification lacks chunk schedule or approval time")
    from research.t34_round0.make_manifest import dispatch_deadline

    try:
        deadline = dispatch_deadline(approved_at)
    except ValueError as exc:
        raise HistoryError("ratification lacks a valid dispatch deadline") from exc
    chunks_root = root / "chunks"
    groups = (
        {path.name: path for path in chunks_root.iterdir() if path.is_dir()}
        if chunks_root.exists()
        else {}
    )
    expected = {f"chunk-{index:03d}" for index in range(len(chunks))}
    remote = {
        phase
        for phase in _claims(registration)
        if re.fullmatch(r"chunks/chunk-\d{3}/attempt-0[12]", phase)
    }
    attempts = [a for g in groups.values() for a in g.iterdir() if a.is_dir()]
    if any(in_flight(a) for a in attempts):
        raise HistoryError("a chunk attempt is still in flight; no claim to reserve")
    local = {a.relative_to(root).as_posix() for a in attempts}
    unfinished = any(not (a / "phase_terminal.json").exists() for a in attempts) or bool(
        remote - local
    )
    settled_early = (
        any((a / "study_closed.json").is_file() for a in attempts)
        or (root / "ceiling" / "study_closed.json").is_file()
    )
    pending = bool(expected - set(groups))
    for name in expected & set(groups):
        latest = sorted(p for p in groups[name].iterdir() if p.is_dir())
        if not latest or not (latest[-1] / "schedule_complete.json").is_file():
            pending = True
    if time.time() <= deadline and pending and not unfinished and not settled_early:
        raise HistoryError("registered chunk schedule is still pending; no claim to reserve")


def run_phase(registration: Mapping[str, Any], phase: str, action: Any) -> Any:
    """Fence and seal a single create-only replay or decision phase."""
    from research.t34_pathb.source_binding_v2 import FrozenWorktreeGuard

    root, _, _ = policy(registration)
    out = root / phase
    with FrozenWorktreeGuard(ROOT, registration["commit"]), history_lock(registration):
        _analysis_reservation_allowed(registration)
        if phase != "analysis":
            terminals = require_published_history(registration)
            if any(terminal["status"] == "INCONCLUSIVE" for terminal in terminals):
                raise CapExpiredHistory("registered first cap expired; skip replay")
        reserve(out, registration)
    status, reason = "KILL", "operator abort or unexplained interruption"
    token = _ACTIVE_PHASE.set(out)
    try:
        with FrozenWorktreeGuard(ROOT, registration["commit"]):
            result = action(out)
        write_once(out / ("decision.json" if phase == "analysis" else "replay.json"), result)
        status, reason = "COMPLETE", "source-fenced phase complete"
        return result
    finally:
        _ACTIVE_PHASE.reset(token)
        if status == "KILL" and phase == "analysis" and not (out / "decision.json").exists():
            write_once(
                out / "decision.json",
                {
                    "schema": "t34-round0d-decision/v1",
                    "decision": "KILL",
                    "reason": reason,
                    "release_authorized": False,
                },
            )
        if status == "KILL":
            write_once(out / "interruption.json", {"decision": "KILL", "reason": reason})
        finalize(out, registration, status, reason)
