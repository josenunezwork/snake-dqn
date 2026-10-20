"""Create-only Round 0 reservations and published receipt-hash history.

Execution uses a detached checkout; only the registered authoring checkout may
append the ledger. A remote admission survives deletion of all local evidence.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping

from research.t34_pathb.retention_history_v2 import (
    _entries,
    _git,
    _local_ledger,
    _published_ledger,
    _registration_sha,
)
from research.t34_round0.common import ROOT, read_json, sha256, write_once

_ACTIVE_PHASE: ContextVar[Path | None] = ContextVar("round0_active_phase", default=None)


class HistoryError(ValueError):
    """An immutable reservation or its published evidence is unsafe."""


class CapExpiredHistory(HistoryError):
    """A registered first cap ended the study without an outcome look."""


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
        or name != f"research/t34_round0/ledger/{root.name}.jsonl"
    ):
        raise HistoryError(
            "ratification requires canonical study_root and authoring history ledger"
        )
    return root, repo, name


def phase_name(out: Path, registration: Mapping[str, Any]) -> str:
    """Accept only fixed chunk attempts, replay and the one analysis directory."""
    root, _, _ = policy(registration)
    if out.resolve() != out or not out.is_relative_to(root):
        raise HistoryError("phase must be an unlinked path under the fixed study root")
    relative = out.relative_to(root).as_posix()
    if not re.fullmatch(r"chunks/chunk-\d{3}/attempt-0[12]|replay|analysis", relative):
        raise HistoryError("phase must use a fixed chunk/attempt or replay/analysis path")
    return relative


def _roots(registration: Mapping[str, Any]) -> list[Path]:
    root, _, _ = policy(registration)
    found = sorted((root / "chunks").glob("*/*")) if (root / "chunks").exists() else []
    for path in found:
        phase_name(path, registration)
    return found + [root / p for p in ("replay", "analysis") if (root / p).exists()]


def _claims(registration: Mapping[str, Any]) -> dict[str, str]:
    root, repo, _ = policy(registration)
    prefix = f"refs/tags/t34-round0/{root.name}/"
    output = _git(repo, "ls-remote", "--refs", "origin", f"{prefix}*")
    return {
        ref[len(prefix) :]: commit for commit, ref in (line.split() for line in output.splitlines())
    }


def _reject_other_study_roots(registration: Mapping[str, Any]) -> None:
    """A Round 0 remote claim binds the one root frozen in the ratification."""
    root, repo, _ = policy(registration)
    prefix = "refs/tags/t34-round0/"
    output = _git(repo, "ls-remote", "--refs", "origin", f"{prefix}*")
    for line in output.splitlines():
        _commit, ref = line.split()
        relative = ref.removeprefix(prefix)
        tag_root, separator, _phase = relative.partition("/")
        if not separator or tag_root != root.name:
            raise HistoryError("remote Round 0 admission belongs to a different study_root")


def terminal_entry(out: Path, registration: Mapping[str, Any]) -> dict[str, str]:
    """Check the complete immutable inventory, including supervisor receipts."""
    relative = phase_name(out, registration)
    seal = out / "phase_terminal.json"
    try:
        value = read_json(seal)
    except FileNotFoundError as exc:
        raise HistoryError("unfinished claim is missing its terminal history") from exc
    if (
        value.get("schema") != "t34-round0-terminal/v1"
        or value.get("phase_path") != str(out)
        or value.get("registration_sha256") != _registration_sha(registration)
        or value.get("status") not in {"COMPLETE", "MECHANICAL_FAILURE", "INCONCLUSIVE", "KILL"}
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
    actual = [terminal_entry(out, registration) for out in roots if out != _ACTIVE_PHASE.get()]
    if len({e.get("phase") for e in entries}) != len(entries) or sorted(
        entries, key=lambda e: e["phase"]
    ) != sorted(actual, key=lambda e: e["phase"]):
        raise HistoryError("published ledger differs from retained fixed phase history")
    claims = _claims(registration)
    local_claims = {}
    for out in roots:
        admission = read_json(out / "admission.json")
        if admission.get("registration_sha256") != _registration_sha(registration):
            raise HistoryError("foreign admission claim")
        local_claims[phase_name(out, registration)] = admission.get("claim_commit")
    if claims != local_claims:
        raise HistoryError("unfinished or erased remote admission is KILL; no reset or reroll")
    return [read_json(out / "phase_terminal.json") for out in roots if out != _ACTIVE_PHASE.get()]


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
    payload = {
        "schema": "t34-round0-admission/v1",
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
        "user.name=T34 Round0 admission",
        "-c",
        "user.email=t34-round0@localhost",
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
        f"{commit}:refs/tags/t34-round0/{root.name}/{relative}",
    )
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "admission.json", dict(payload, claim_commit=commit))


def finalize(out: Path, registration: Mapping[str, Any], status: str, reason: str) -> None:
    """Seal after closing the source fence and durably append the receipt hash."""
    _, repo, name = policy(registration)
    if status not in {"COMPLETE", "MECHANICAL_FAILURE", "INCONCLUSIVE", "KILL"}:
        raise HistoryError("unknown phase terminal status")
    common = Path(_git(repo, "rev-parse", "--git-common-dir").strip())
    if not common.is_absolute():
        common = repo / common
    with (common / "t34-round0-history.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prior = _entries(_local_ledger(repo, name))
        # Parallel lanes may finish in either order; each append retains all
        # previously sealed entries and cannot overwrite or duplicate an entry.
        if any(e.get("phase") == phase_name(out, registration) for e in prior):
            raise HistoryError("phase already appears in the append-only ledger")
        for entry in prior:
            if (
                terminal_entry(Path(registration["study_root"]) / entry["phase"], registration)
                != entry
            ):
                raise HistoryError("prior terminal history differs")
        # A structural marker lets the pre-reservation schedule check identify
        # early settlement without reading any terminal entry or outcome file.
        # Its bytes are included in the immutable terminal inventory below.
        if status in {"KILL", "INCONCLUSIVE"} or reason.startswith(
            "explained unconditional controller fault"
        ):
            write_once(out / "study_closed.json", {"status": status})
        paths = list(out.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise HistoryError("cannot seal linked evidence")
        files = {p.relative_to(out).as_posix(): sha256(p) for p in paths if p.is_file()}
        write_once(
            out / "phase_terminal.json",
            {
                "schema": "t34-round0-terminal/v1",
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


def selected_rows(
    rows: list[dict[str, Any]], chunks: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Freeze a ceil(10%) bank-order sample independently within every chunk."""
    import math

    from research.t34_round0.bank import ordered_rows

    ordered = ordered_rows(rows)
    selected, covered = [], []
    for chunk in sorted(chunks, key=lambda c: c["slice"]["start"]):
        start, count = chunk["slice"]["start"], chunk["slice"]["count"]
        if type(start) is not int or type(count) is not int or start < 0 or count < 1:
            raise HistoryError("chunk slice must contain positive integer counts")
        group = ordered[start : start + count]
        if len(group) != count or chunk["slice"].get("world_ids") != [r["world_id"] for r in group]:
            raise HistoryError("chunk slice differs from frozen bank")
        covered.extend(r["world_id"] for r in group)
        # Preserve A2's mix coverage rather than selecting only the prefix mix
        # of a contiguous chunk. Largest remainders fix the exact ceil(10%) N;
        # ties follow the frozen scripted/frozen/mixed order, without outcomes.
        sample_count = math.ceil(count / 10)
        mixes = [
            mix
            for mix in ("scripted", "frozen", "mixed", None)
            if any(row.get("mix") == mix for row in group)
        ]
        groups = {mix: [row for row in group if row.get("mix") == mix] for mix in mixes}
        quotas = {mix: sample_count * len(groups[mix]) // count for mix in mixes}
        remaining = sample_count - sum(quotas.values())
        ranked = sorted(mixes, key=lambda mix: -(sample_count * len(groups[mix]) % count))
        for mix in ranked[:remaining]:
            quotas[mix] += 1
        wanted = {row["world_id"] for mix in mixes for row in groups[mix][: quotas[mix]]}
        selected.extend(row for row in group if row["world_id"] in wanted)
    if covered != [r["world_id"] for r in ordered]:
        raise HistoryError("chunks must partition the whole bank in canonical order")
    return selected


def require_chunks(chunks: list[Path], registration: Mapping[str, Any]) -> None:
    """Require the sole admissible completed attempt of every fixed chunk."""
    root, _, _ = policy(registration)
    terminals = require_published_history(registration)
    if any(t["status"] == "KILL" for t in terminals):
        raise HistoryError("operator abort or unexplained interruption is KILL")
    if any(t["status"] == "INCONCLUSIVE" for t in terminals):
        raise CapExpiredHistory("registered first cap expired; INCONCLUSIVE with no extension")
    retained_outputs, scientific_stop = [], False
    chunks_root = root / "chunks"
    if chunks_root.exists():
        for group in sorted(chunks_root.iterdir()):
            if not group.is_dir():
                continue
            attempts = sorted(path for path in group.iterdir() if path.is_dir())
            if not attempts:
                continue
            terminal = read_json(attempts[-1] / "phase_terminal.json")
            if terminal.get("status") == "COMPLETE":
                retained_outputs.append(attempts[-1] / "job/exports/round0")
            scientific_stop = scientific_stop or terminal.get("reason", "").startswith(
                "explained unconditional controller fault"
            )
    if scientific_stop:
        if len(set(chunks)) != len(chunks) or set(chunks) != set(retained_outputs):
            raise HistoryError("scientific STOP must consume exactly all retained completed chunks")
        return
    if (root / "analysis").exists() and _ACTIVE_PHASE.get() != root / "analysis":
        raise HistoryError("the sole outcome look has already been reserved")
    frozen_chunks = registration.get("sizing", {}).get("chunks")
    if frozen_chunks is not None and (
        not isinstance(frozen_chunks, list)
        or not frozen_chunks
        or any(
            not isinstance(item, Mapping)
            or type(item.get("start")) is not int
            or type(item.get("count")) is not int
            or item["start"] < 0
            or item["count"] < 1
            for item in frozen_chunks
        )
    ):
        raise HistoryError("ratification lacks a fixed chunk schedule")
    expected_names = (
        [f"chunk-{index:03d}" for index in range(len(frozen_chunks))]
        if frozen_chunks is not None
        else None
    )
    chunks_root = root / "chunks"
    try:
        groups = sorted(chunks_root.iterdir())
    except FileNotFoundError as exc:
        raise HistoryError("unfinished claim is missing its terminal history") from exc
    if (
        any(not group.is_dir() for group in groups)
        or expected_names is not None
        and [group.name for group in groups] != expected_names
    ):
        raise HistoryError("fixed chunk schedule is missing a terminal claim")
    expected = []
    for group in groups:
        attempts = sorted(group.iterdir())
        if not attempts or [p.name for p in attempts] not in (
            ["attempt-01"],
            ["attempt-01", "attempt-02"],
        ):
            raise HistoryError("fixed chunks permit at most one fresh mechanical attempt")
        try:
            seals = [read_json(p / "phase_terminal.json") for p in attempts]
        except FileNotFoundError as exc:
            raise HistoryError("unfinished claim is missing its terminal history") from exc
        if len(seals) == 2 and seals[0]["status"] != "MECHANICAL_FAILURE":
            raise HistoryError("second attempt was not an outcome-independent mechanical rerun")
        if seals[-1]["status"] != "COMPLETE":
            raise HistoryError("latest chunk attempt is incomplete")
        expected.append(attempts[-1] / "job/exports/round0")
    if len(set(chunks)) != len(chunks) or set(chunks) != set(expected):
        raise HistoryError("supplied chunks differ from the fixed retained attempt history")


def _analysis_reservation_allowed(registration: Mapping[str, Any]) -> None:
    """Refuse a still-pending schedule without consuming the sole outcome look."""
    root, _, _ = policy(registration)
    chunks = registration.get("sizing", {}).get("chunks")
    approved_at = registration.get("approved_at")
    if not isinstance(chunks, list) or not chunks or not isinstance(approved_at, str):
        return
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
    remote_chunk_claims = {
        phase
        for phase in _claims(registration)
        if re.fullmatch(r"chunks/chunk-\d{3}/attempt-0[12]", phase)
    }
    local_attempts = {
        attempt.relative_to(root).as_posix()
        for group in groups.values()
        for attempt in group.iterdir()
        if attempt.is_dir()
    }
    unfinished_claim = any(
        not (attempt / "phase_terminal.json").exists()
        for group in groups.values()
        for attempt in group.iterdir()
        if attempt.is_dir()
    ) or bool(remote_chunk_claims - local_attempts)
    settled_early = any(
        (attempt / "study_closed.json").is_file()
        for group in groups.values()
        for attempt in group.iterdir()
        if attempt.is_dir()
    )
    if (
        time.time() <= deadline
        and expected - set(groups)
        and not unfinished_claim
        and not settled_early
    ):
        raise HistoryError(
            "registered chunk schedule is still pending; analysis has no claim to reserve"
        )


def run_phase(registration: Mapping[str, Any], phase: str, action: Any) -> Any:
    """Fence and seal a single create-only replay or decision phase."""
    from research.t34_pathb.source_binding_v2 import FrozenWorktreeGuard

    root, _, _ = policy(registration)
    out = root / phase
    with FrozenWorktreeGuard(ROOT, registration["commit"]):
        if phase == "analysis":
            _analysis_reservation_allowed(registration)
        else:
            terminals = require_published_history(registration)
            if any(terminal["status"] == "INCONCLUSIVE" for terminal in terminals):
                raise CapExpiredHistory(
                    "registered first cap expired; skip replay and settle INCONCLUSIVE"
                )
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
                    "schema": "t34-round0-decision/v1",
                    "decision": "KILL",
                    "reason": reason,
                    "release_authorized": False,
                },
            )
        if status == "KILL":
            write_once(out / "interruption.json", {"decision": "KILL", "reason": reason})
        finalize(out, registration, status, reason)
