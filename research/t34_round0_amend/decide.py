"""Sole entrypoint for the pre-look Round 0 boost-floor analysis amendment.

Runs once, after the registered sole analysis (frozen checkout) has sealed
``analysis/decision.json`` and its ledger entry is published on origin/main. It
re-evaluates only the delivered-boost clause under the amended relative floor
and writes the governing ``amendment-boost-floor/decision.json``.

Before its create-only remote claim it parses no outcome: it checks only sealed
inventories (by hash), published ledgers, remote claims and source bytes. After
the claim it reads the registered decision and, only when that decision rests on
clauses, re-reads the hash-bound pair records the registered analysis consumed,
reproduces the registered summary exactly and changes only the boost clause.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from research.t34_pathb.retention_history_v2 import (
    _entries,
    _git,
    _local_ledger,
    _published_ledger,
    _registration_sha,
)
from research.t34_round0.analyze import _bound_path, summarize
from research.t34_round0.common import ROOT, read_json, sha256, write_once
from research.t34_round0.history import _claims, policy, terminal_entry
from research.t34_round0_amend.rule import CEILING, CLAUSE, amend, pooled_boosts

AMENDMENT_NAME = "research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json"
AMENDMENT = ROOT / AMENDMENT_NAME
PHASE = "amendment-boost-floor"
SCHEMA = "t34-round0-decision-amended/v1"
TERMINAL_SCHEMA = "t34-round0-amend-terminal/v1"


class AmendmentRefused(RuntimeError):
    """A precondition failed before the claim; nothing was claimed or written."""


class AmendmentIntegrity(ValueError):
    """Bound evidence does not reproduce the registered analysis."""


def ledger_name(root: Path) -> str:
    return f"research/t34_round0_amend/ledger/{root.name}.jsonl"


def tag_ref(root: Path) -> str:
    # Outside refs/tags/t34-round0/<root>/ so Round 0's claim inventory is untouched.
    return f"refs/tags/t34-round0-amend/{root.name}/boost-floor"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_amendment(
    amendment: Mapping[str, Any], registration: Mapping[str, Any], ratification: Path
) -> None:
    if (
        amendment.get("schema") != "t34-round0-analysis-amendment/v1"
        or amendment.get("status") != "DECLARED_PRE_LOOK"
        or amendment.get("ratification_canonical_sha256") != _registration_sha(registration)
        or amendment.get("ratification_file_sha256") != sha256(ratification)
        or amendment.get("ratified_commit") != registration.get("commit")
        or amendment.get("study_root") != registration.get("study_root")
        or registration.get("status") != "RATIFIED"
    ):
        raise AmendmentRefused("amendment does not bind this exact Round 0 ratification")


def verify_sources(checkout: Path, amendment: Mapping[str, Any], commit: str) -> None:
    """The imported registered code must be the ratified commit's exact bytes."""
    sources = amendment.get("registered_sources")
    if not isinstance(sources, dict) or "research/t34_round0/analyze.py" not in sources:
        raise AmendmentRefused("amendment lacks its registered source binding")
    for name, expected in sources.items():
        try:
            frozen = subprocess.run(
                ["git", "cat-file", "blob", f"{commit}:{name}"],
                cwd=checkout,
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise AmendmentRefused(f"ratified commit lacks registered source {name}") from exc
        if sha256(checkout / name) != expected or hashlib.sha256(frozen).hexdigest() != expected:
            raise AmendmentRefused(f"registered source differs from ratified commit: {name}")


def verify_checkout(checkout: Path, amendment_path: Path) -> None:
    """Run from a clean checkout whose HEAD is published on origin/main."""
    if _git(checkout, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise AmendmentRefused("amendment checkout has tracked or untracked changes")
    _git(checkout, "fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main")
    try:
        _git(checkout, "merge-base", "--is-ancestor", "HEAD", "origin/main")
    except RuntimeError as exc:
        raise AmendmentRefused("amendment checkout HEAD is not published on origin/main") from exc
    if amendment_path.resolve() != (checkout / AMENDMENT_NAME).resolve():
        raise AmendmentRefused("amendment must be read from its own checkout")


def verify_precedence(repo: Path, amendment_path: Path, admission: Mapping[str, Any]) -> None:
    """The amendment reached origin/main, unchanged, before the analysis claim."""
    history = _git(
        repo, "log", "--first-parent", "--format=%H %ct", "origin/main", "--", AMENDMENT_NAME
    ).splitlines()
    if len(history) != 1:
        raise AmendmentRefused("amendment must be published once on origin/main and never edited")
    commit, committed = history[0].split()
    published = _git(repo, "show", f"{commit}:{AMENDMENT_NAME}")
    current = _git(repo, "show", f"origin/main:{AMENDMENT_NAME}")
    if published != current or current != amendment_path.read_text():
        raise AmendmentRefused("amendment bytes differ from their origin/main publication")
    dispatched = admission.get("dispatched_at_unix_seconds")
    if (
        isinstance(dispatched, bool)
        or not isinstance(dispatched, (int, float))
        or int(committed) >= int(dispatched)
    ):
        raise AmendmentRefused("amendment was not published before the analysis was reserved")


def preconditions(
    registration: Mapping[str, Any], amendment_path: Path
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Structural checks only: no outcome file is parsed before the claim."""
    root, repo, name = policy(registration)
    out = root / PHASE
    analysis = root / "analysis"
    if out.exists():
        raise AmendmentRefused("amendment phase is create-only; it already exists")
    if not (analysis / "phase_terminal.json").is_file():
        raise AmendmentRefused("the registered sole analysis has not been sealed")
    entry = terminal_entry(analysis, registration)
    if entry not in _entries(_published_ledger(repo, name)):
        raise AmendmentRefused("registered analysis ledger entry is not published on origin/main")
    admission = read_json(analysis / "admission.json")
    if admission.get("registration_sha256") != _registration_sha(registration) or _claims(
        registration
    ).get("analysis") != admission.get("claim_commit"):
        raise AmendmentRefused("registered analysis lacks its remote admission claim")
    verify_precedence(repo, amendment_path, admission)
    amend_ledger = ledger_name(root)
    local, published = _local_ledger(repo, amend_ledger), _published_ledger(repo, amend_ledger)
    if local != published or any(e.get("phase") == PHASE for e in _entries(published)):
        raise AmendmentRefused("amendment ledger already records a look or is unpublished")
    if _git(repo, "ls-remote", "--refs", "origin", tag_ref(root)).strip():
        raise AmendmentRefused("amendment remote claim already exists; no second look")
    return root, repo, analysis, entry


def reserve(
    out: Path, repo: Path, checkout: Path, registration: Mapping[str, Any], amendment_sha: str
) -> None:
    """Publish a unique create-only remote tag, then create the local phase."""
    head = _git(checkout, "rev-parse", "HEAD").strip()
    payload = {
        "schema": "t34-round0-amend-admission/v1",
        "phase_path": str(out),
        "registration_sha256": _registration_sha(registration),
        "amendment_sha256": amendment_sha,
        "amendment_checkout_commit": head,
        "unexplained_interruption": "KILL",
        "claimed_at_unix_seconds": time.time(),
        "nonce": uuid.uuid4().hex,
    }
    commit = _git(
        checkout,
        "-c",
        "user.name=T34 Round0 amendment",
        "-c",
        "user.email=t34-round0@localhost",
        "commit-tree",
        f"{head}^{{tree}}",
        "-p",
        head,
        "-m",
        json.dumps(payload, sort_keys=True),
    ).strip()
    # A plain (non-force) push refuses to move an existing tag: the claim is create-only.
    _git(checkout, "push", "--porcelain", "origin", f"{commit}:{tag_ref(out.parent)}")
    out.mkdir(parents=False, exist_ok=False)
    write_once(out / "admission.json", dict(payload, claim_commit=commit))


def load_records(registered: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Re-read exactly the hash-bound pair records the registered analysis consumed."""
    chunk_hashes = registered.get("chunk_sha256")
    if not isinstance(chunk_hashes, dict) or not chunk_hashes:
        raise AmendmentIntegrity("registered decision lacks its chunk hash binding")
    records, seen = [], set()
    for receipt_name, digest in sorted(chunk_hashes.items()):
        receipt_path = Path(receipt_name)
        if receipt_path.name != "chunk.json" or sha256(receipt_path) != digest:
            raise AmendmentIntegrity("chunk receipt differs from the registered analysis")
        receipt = read_json(receipt_path)
        for entry in receipt["pairs"]:
            # Same SHA-256 binding the registered analysis used for every pair.
            pair_path = _bound_path(receipt_path.parent, entry)
            if pair_path in seen:
                raise AmendmentIntegrity("duplicate pair record")
            seen.add(pair_path)
            pair = read_json(pair_path)
            if (
                pair.get("schema") != "t34-round0-pair/v1"
                or pair.get("status") != "complete"
                or pair.get("preflight") is not False
            ):
                raise AmendmentIntegrity("non-scientific or incomplete pair")
            records.append(pair)
    return records


def amended_decision(registered: Mapping[str, Any], amendment_sha: str) -> dict[str, Any]:
    """Pass non-clause verdicts through; otherwise re-evaluate the boost clause only."""
    header = dict(
        schema=SCHEMA,
        governing=True,
        amendment_sha256=amendment_sha,
        registered_schema=registered.get("schema"),
        registered_decision=registered.get("decision"),
        registered_reason=registered.get("reason"),
    )
    if (
        registered.get("schema") != "t34-round0-decision/v1"
        or registered.get("evidence_authority") != "TIER_1_DEVELOPMENT"
        or not isinstance(registered.get("clauses"), dict)
        or registered.get("unconditional_faults")
    ):
        # KILL, INCONCLUSIVE and every fault STOP precede the clauses: unchanged.
        return dict(
            registered,
            **header,
            boost_floor_amendment=dict(applied=False, reason="registered verdict precedes clauses"),
        )
    records = load_records(registered)
    summary = json.loads(_canonical(summarize(records)))
    for key, value in summary.items():
        if key != "evidence_authority" and _canonical(value) != _canonical(registered.get(key)):
            raise AmendmentIntegrity(f"bound records do not reproduce registered {key}")
    teacher, bare = pooled_boosts(records, "teacher"), pooled_boosts(records, "bare")
    result = amend(registered, teacher, bare)
    result.update(
        header,
        registered_clauses=dict(registered["clauses"]),
        boost_floor_amendment=dict(
            applied=True,
            rule="0.5 * bare_share <= teacher_share <= 0.10 (inclusive; bare 0 -> floor 0)",
            teacher_boost_decisions=teacher[0],
            teacher_decisions=teacher[1],
            teacher_boost_share=teacher[0] / teacher[1],
            bare_boost_decisions=bare[0],
            bare_decisions=bare[1],
            bare_boost_share=bare[0] / bare[1],
            floor=0.5 * bare[0] / bare[1],
            ceiling=CEILING,
            registered_clause=registered["clauses"][CLAUSE],
            amended_clause=result["clauses"][CLAUSE],
        ),
    )
    return result


def seal(out: Path, repo: Path, registration: Mapping[str, Any], status: str, reason: str) -> None:
    """Seal the inventory and append its hash to the amendment ledger (publish next)."""
    name = ledger_name(out.parent)
    common = Path(_git(repo, "rev-parse", "--git-common-dir").strip())
    if not common.is_absolute():
        common = repo / common
    with (common / "t34-round0-amend-history.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(e.get("phase") == PHASE for e in _entries(_local_ledger(repo, name))):
            raise AmendmentIntegrity("amendment already appears in its append-only ledger")
        paths = list(out.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise AmendmentIntegrity("cannot seal linked evidence")
        files = {p.relative_to(out).as_posix(): sha256(p) for p in paths if p.is_file()}
        seal_path = out / "phase_terminal.json"
        write_once(
            seal_path,
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
            raise AmendmentIntegrity("ledger cannot use links")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as handle:
            entry = {"phase": PHASE, "terminal_sha256": sha256(seal_path)}
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def run(
    ratification: Path,
    out_file: Path,
    *,
    checkout: Path = ROOT,
    amendment_path: Path = AMENDMENT,
) -> dict[str, Any]:
    """Refuse without side effects, or claim once and always seal one decision."""
    try:
        registration = read_json(ratification)
        amendment = read_json(amendment_path)
        verify_amendment(amendment, registration, ratification)
        verify_checkout(checkout, amendment_path)
        verify_sources(checkout, amendment, registration["commit"])
        root, repo, analysis, entry = preconditions(registration, amendment_path)
        if _git(repo, "remote", "get-url", "origin") != _git(
            checkout, "remote", "get-url", "origin"
        ):
            raise AmendmentRefused("amendment checkout and history ledger use different origins")
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
        if isinstance(exc, AmendmentRefused):
            raise
        raise AmendmentRefused(f"precondition failed: {exc}") from exc
    out = root / PHASE
    if out_file != out / "decision.json":
        raise AmendmentRefused("governing output must be <study_root>/amendment-boost-floor/")
    amendment_sha = sha256(amendment_path)
    reserve(out, repo, checkout, registration, amendment_sha)
    status, reason = "KILL", "operator abort or unexplained interruption"
    result: dict[str, Any] = {}
    try:
        registered = read_json(analysis / "decision.json")
        if terminal_entry(analysis, registration) != entry:
            raise AmendmentIntegrity("registered analysis changed after the claim")
        result = amended_decision(registered, amendment_sha)
        status, reason = "COMPLETE", "boost-floor amendment complete"
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError, IndexError) as exc:
        reason = f"amendment integrity failure: {exc}"
    finally:
        if status != "COMPLETE":
            result = dict(
                schema=SCHEMA,
                governing=True,
                decision="KILL",
                reason=reason,
                evidence_authority="INTEGRITY_FAILURE",
                release_authorized=False,
                amendment_sha256=amendment_sha,
            )
        write_once(out / "decision.json", result)
        seal(out, repo, registration, status, reason)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.ratification, args.out)
    except AmendmentRefused as exc:
        parser.exit(2, f"REFUSED (nothing claimed or written): {exc}\n")
    # Publish the amendment ledger before reading or reporting the governing decision.
    print(f"sealed {args.out}; publish the amendment ledger before reporting")


if __name__ == "__main__":
    main()
