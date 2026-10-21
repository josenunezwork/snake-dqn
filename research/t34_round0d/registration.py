"""Draft and (later, by the operator only) ratify Round 0-D; never ratifies on import.

``draft`` writes the committed DRAFT registration (status DRAFT, not ratified,
no dispatch authority). ``ratify`` is the operator's later command after the
independent review/Fable approval, the A5-D receipt and a source-current A6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from research.t34_round0d.bank import check_audit
from research.t34_round0d.common import (
    LEDGER_DIR,
    PACKAGE,
    ROOT,
    head,
    read_json,
    sha256,
    source_hashes,
    write_once,
)

PREREG = PACKAGE / "PREREGISTRATION_ROUND0D.md"
RATIFICATION = PACKAGE / "REGISTRATION_ROUND0D.json"
DRAFT = PACKAGE / "REGISTRATION_ROUND0D_DRAFT.json"
SCHEMA = "t34-round0d-ratification/v1"
STUDY_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0d-study-01")
BANK = PACKAGE / "BANK_ROUND0D.json"
AUDIT = PACKAGE / "SEED_AUDIT_ROUND0D.json"


def require_registration(
    registration: Mapping[str, Any], bank_path: Path, audit_path: Path, *, repo: Path = ROOT
) -> dict[str, Any]:
    """Reject drafts, changed rules, source, sizing, A5-D or bank identities."""
    bank = check_audit(bank_path, audit_path)
    required = {
        "schema": SCHEMA,
        "status": "RATIFIED",
        "registration_sha256": sha256(repo / "research/t34_round0d/PREREGISTRATION_ROUND0D.md"),
        "bank_sha256": sha256(bank_path),
        "audit_sha256": sha256(audit_path),
    }
    if any(registration.get(k) != v for k, v in required.items()):
        raise ValueError("Round 0-D is not ratified for these exact rules and bank")
    if not all(
        isinstance(registration.get(k), str) and registration[k].strip()
        for k in ("operator", "approved_at", "commit")
    ):
        raise ValueError("operator identity, date and commit are required")
    commit = registration["commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("ratification needs a full source SHA")
    if head(repo) != commit:
        raise ValueError("checkout differs from ratified source commit")
    hashes = registration.get("source_hashes")
    if not isinstance(hashes, dict) or hashes != source_hashes(repo):
        raise ValueError("source inventory differs from ratified executable closure")
    preflight = registration.get("preflight", {})
    if preflight.get("source_hashes") != hashes or len(str(preflight.get("sha256", ""))) != 64:
        raise ValueError("ratification lacks the source-bound real A6 preflight")
    a5d = registration.get("a5d", {})
    if len(str(a5d.get("sha256", ""))) != 64:
        raise ValueError("ratification lacks the A5-D evidence binding")
    from research.t34_round0d.history import policy

    policy(registration)
    sizing = registration.get("sizing")
    if not isinstance(sizing, dict) or not sizing.get("chunks"):
        raise ValueError("ratification lacks the frozen chunk sizing")
    if sizing.get("study_unit_ceiling") != 48.0 or sizing.get("expected_units", 99) > 40.0:
        raise ValueError("ratified sizing breaks the 40 u expected / 48 u charged ceiling")
    return bank


def require_published_registration(
    registration: Mapping[str, Any], ratification_path: Path
) -> None:
    """Require the canonical ratification JSON on origin/main after its frozen source."""
    from research.t34_round0d.history import policy

    _, repo, _ = policy(registration)
    relative = RATIFICATION.relative_to(ROOT).as_posix()
    expected = (repo / relative).resolve()
    if ratification_path.resolve() != expected:
        raise ValueError("ratification must use the canonical REGISTRATION_ROUND0D.json")
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", "+refs/heads/main:refs/remotes/origin/main"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        published = subprocess.run(
            ["git", "show", f"origin/main:{relative}"], cwd=repo, check=True, capture_output=True
        ).stdout
        source = str(registration.get("commit"))
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source, "origin/main"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        later = subprocess.run(
            ["git", "log", "--format=%H", f"{source}..origin/main", "--", relative],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("ratification must be committed and pushed to origin/main") from exc
    if expected.read_bytes() != published or read_json(expected) != dict(registration):
        raise ValueError("ratification bytes differ from origin/main")
    if not later:
        raise ValueError("ratification must be committed to main after its frozen source")


def draft(out: Path = DRAFT, *, study_root: Path = STUDY_ROOT) -> dict[str, Any]:
    """Write the non-dispatchable DRAFT registration with ESTIMATED sizing only."""
    from research.t34_round0d.sizing import (
        EXPECTED_UNIT_CEILING,
        STUDY_UNIT_CEILING,
        estimated_plan,
    )

    bank = check_audit(BANK, AUDIT)
    payload = {
        "schema": "t34-round0d-registration-draft/v1",
        "status": "DRAFT",
        "ratified": False,
        "dispatch_authorized": False,
        "release_authorized": False,
        "dispatch_condition": (
            "Round 0 published sole analysis decision is a scientific STOP or INCONCLUSIVE "
            "(NO-GO branch B or C, docs/research/specs/2026-10-20_t34_nogo_branch_plan.md §3); "
            "a Round 0 GO shelves Round 0-D unrun"
        ),
        "preregistration": {
            "path": PREREG.relative_to(ROOT).as_posix(),
            "sha256": sha256(PREREG),
        },
        "study_root": str(study_root),
        "history_ledger": {
            "repo": "<authoring checkout>",
            "path": f"{LEDGER_DIR}/{study_root.name}.jsonl",
        },
        "remote_claim_prefix": "refs/tags/t34-round0d/<study_root>/<phase>",
        "bank": {
            "path": BANK.relative_to(ROOT).as_posix(),
            "sha256": sha256(BANK),
            "namespace": bank["namespace"],
            "counts": bank["counts"],
        },
        "audit": {"path": AUDIT.relative_to(ROOT).as_posix(), "sha256": sha256(AUDIT)},
        "round0_rules_copied_by_import": {
            "summarize": "research.t34_round0.analyze.summarize (bars, clauses, decision order)",
            "chunk_sizing": "research.t34_round0.make_manifest.size_round",
            "replay_sample": "research.t34_round0.history.selected_rows",
            "bare_arm": "research.t34_round0.runner._bare_trace",
            "metrics": "research.t34_round0.runner._metrics_from_trace",
            "oc": {
                "path": "research/t34_round0/OC_ROUND0.json",
                "sha256": sha256(ROOT / "research/t34_round0/OC_ROUND0.json"),
            },
        },
        "controller": "research.t34_round0d.controller.DenseT3Controller",
        "sizing_estimated": {
            "low_rate_15_per_hour": estimated_plan(15.0),
            "high_rate_23_per_hour": estimated_plan(23.0),
        },
        "unit_ceilings": {
            "expected_units_max_for_ratification": EXPECTED_UNIT_CEILING,
            "study_charged_units_ceiling": STUDY_UNIT_CEILING,
            "a5d_unit_cap": 3.0,
            "replay_unit_cap": 2.0,
        },
        "dispatch_window_hours": 48,
        "blockers": [
            "Round 0 sole analysis must be published with a NO-GO scientific code",
            "independent review and Fable judge approval of PREREGISTRATION_ROUND0D.md",
            "A5-D fidelity PASS and throughput >= 8 dense worlds/h/lane",
            "source-current excluded A6 Mac preflight",
            "operator ratification JSON committed and pushed to main before first dispatch",
        ],
    }
    write_once(out, payload)
    return payload


def ratify(
    *,
    operator: str,
    a5d_path: Path,
    preflight_record: Path,
    out: Path,
    study_root: Path = STUDY_ROOT,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Operator-only: freeze sizing, source, A5-D and A6 after approval (not run here)."""
    from research.t34_round0d.sizing import plan, validate

    if not operator.strip():
        raise ValueError("operator name is required")
    check_audit(BANK, AUDIT)
    a5d = validate(read_json(a5d_path))
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("ratification requires committed tracked source")
    hashes = source_hashes(repo)
    verification = read_json(preflight_record)
    if (
        verification.get("schema") != "t34-round0d-preflight-validation/v1"
        or verification.get("status") != "PASS"
        or verification.get("source_hashes") != hashes
    ):
        raise ValueError("ratification requires a source-bound successful real A6 preflight")
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if (
        a5d["a6_measurement_sha256"] != verification["preflight"]["sha256"]
        or a5d.get("a6_source_sha256") != digest
    ):
        raise ValueError("A5-D sizing must bind this exact source-current A6 measurement")
    sizing = plan(a5d)
    payload = {
        "schema": SCHEMA,
        "status": "RATIFIED",
        "operator": operator,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "commit": head(repo),
        "study_root": str(study_root),
        "history_ledger": {
            "repo": str(repo.resolve()),
            "path": f"{LEDGER_DIR}/{study_root.name}.jsonl",
        },
        "sizing": sizing,
        "registration_sha256": sha256(PREREG),
        "bank_sha256": sha256(BANK),
        "audit_sha256": sha256(AUDIT),
        "a5d": {"path": str(a5d_path.resolve()), "sha256": sha256(a5d_path)},
        "dense_r1_plan": a5d["dense_r1_plan"],
        "preflight": {
            "path": str(preflight_record.resolve()),
            "sha256": sha256(preflight_record),
            "source_hashes": hashes,
        },
        "source_hashes": hashes,
        "release_authorized": False,
        "scope": "Round 0-D development premise test only; dispatch only on a Round 0 NO-GO",
    }
    require_registration(payload, BANK, AUDIT, repo=repo)
    write_once(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("draft")
    rat = sub.add_parser("ratify")
    rat.add_argument("--operator", required=True)
    rat.add_argument("--a5d", type=Path, required=True)
    rat.add_argument("--preflight-record", type=Path, required=True)
    rat.add_argument("--out", type=Path, default=RATIFICATION)
    args = parser.parse_args()
    if args.command == "draft":
        result = draft()
        print(json.dumps({"status": result["status"], "path": str(DRAFT)}))
    else:
        ratify(
            operator=args.operator,
            a5d_path=args.a5d,
            preflight_record=args.preflight_record,
            out=args.out,
        )


if __name__ == "__main__":
    main()
