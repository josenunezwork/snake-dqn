"""Draft and (later, operator only) activate the Round 0-D registration; never on import.

Two stages (PREREGISTRATION_ROUND0D.md "Approval and activation"):

* **Approval** (before the Round 0 sole look): the judge approves these exact rule,
  code and pin bytes on main. ``draft`` writes the DRAFT record of that state.
* **Activation** (``ratify``, mechanical): only after (a) the governing amended
  Round 0 verdict is published and routes to Round 0-D (scientific STOP or
  INCONCLUSIVE), (b) a source-current A6 and (c) an A5-D receipt whose frozen
  rate-to-action table says RATIFY. It binds the governing decision.json SHA-256,
  its phase_terminal SHA-256 and the amendment-ledger commit, freezes sizing, and
  starts the 48 h dispatch clock. No Round 0 outcome number is read; only the
  verdict code fields named in RULING_round0_go_file.md §3.
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
from research.t34_round0d.pins import PINNED, check_pins

PREREG = PACKAGE / "PREREGISTRATION_ROUND0D.md"
RATIFICATION = PACKAGE / "REGISTRATION_ROUND0D.json"
DRAFT = PACKAGE / "REGISTRATION_ROUND0D_DRAFT.json"
SCHEMA = "t34-round0d-ratification/v2"
STUDY_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0d-study-01")
ROUND0_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01")
GOVERNING_PHASE = "amendment-boost-floor"
GOVERNING_LEDGER = f"research/t34_round0_amend/ledger/{ROUND0_ROOT.name}.jsonl"
FALLBACK_LEDGER = f"research/t34_round0/ledger/{ROUND0_ROOT.name}.jsonl"
BANK = PACKAGE / "BANK_ROUND0D.json"
FALLBACK_REFUSAL_TEXT = "amendment was not published before the analysis was reserved"
AUDIT = PACKAGE / "SEED_AUDIT_ROUND0D.json"


def round0_route(decision: Mapping[str, Any], *, governing: bool) -> str:
    """Route a governing Round 0 verdict by its code fields only (RULING §3)."""
    code = decision.get("decision")
    authority = decision.get("evidence_authority")
    faults = decision.get("unconditional_faults") or []
    reason = str(decision.get("reason", ""))
    if governing and decision.get("governing") is not True:
        return "NOT_GOVERNING"
    if code == "GO" and authority == "TIER_1_DEVELOPMENT":
        return "GO_PLAN_SHELVE_ROUND0D"
    if code == "STOP" and (faults or reason.startswith("unconditional")):
        return "BUG_BRANCH"
    if code in ("STOP", "INCONCLUSIVE") and authority == "TIER_1_DEVELOPMENT":
        return "ROUND0D"
    return "MECHANICAL_BRANCH"


def bind_round0_verdict(
    decision_path: Path, ledger_repo: Path, *, fallback_refusal: Path | None = None
) -> dict[str, Any]:
    """Verify the published governing verdict and return its hash binding (Mac only)."""
    from research.t34_pathb.retention_history_v2 import _entries, _git, _published_ledger

    governing = fallback_refusal is None
    phase = GOVERNING_PHASE if governing else "analysis"
    ledger = GOVERNING_LEDGER if governing else FALLBACK_LEDGER
    if decision_path.resolve() != ROUND0_ROOT / phase / "decision.json":
        raise ValueError("Round 0 verdict must be the governing amendment-boost-floor decision")
    if not governing:
        # RULING §2.2: the only admissible refusal is late amendment publication, and it
        # must be preserved and logged in CAMPAIGN_LOG before any Round 0-conditioned lane.
        text = fallback_refusal.read_text() if fallback_refusal.is_file() else ""
        log = (ROOT / "docs/research/final_run_20261019/CAMPAIGN_LOG.md").read_text()
        if FALLBACK_REFUSAL_TEXT not in text or sha256(fallback_refusal) not in log:
            raise ValueError(
                "registered-file fallback needs the preserved late-publication decide refusal, "
                "logged by SHA-256 in CAMPAIGN_LOG.md"
            )
    seal_path = decision_path.parent / "phase_terminal.json"
    seal = read_json(seal_path)
    if seal.get("status") != "COMPLETE" or seal.get("files", {}).get("decision.json") != sha256(
        decision_path
    ):
        raise ValueError("Round 0 verdict is not a sealed COMPLETE phase")
    entry = {"phase": phase, "terminal_sha256": sha256(seal_path)}
    if entry not in _entries(_published_ledger(ledger_repo, ledger)):
        raise ValueError("Round 0 verdict ledger entry is not published on origin/main")
    commit = _git(ledger_repo, "log", "-1", "--format=%H", "origin/main", "--", ledger).strip()
    decision = read_json(decision_path)
    route = round0_route(decision, governing=governing)
    if route != "ROUND0D":
        raise ValueError(f"governing Round 0 verdict routes to {route}, not Round 0-D")
    binding = {
        "path": str(decision_path),
        "sha256": sha256(decision_path),
        "phase_terminal_sha256": sha256(seal_path),
        "ledger": ledger,
        "ledger_commit": commit,
        "decision": decision["decision"],
        "route": route,
        "governing": governing,
    }
    if not governing:
        binding["fallback_refusal"] = {
            "path": str(fallback_refusal),
            "sha256": sha256(fallback_refusal),
        }
    return binding


def require_registration(
    registration: Mapping[str, Any],
    bank_path: Path,
    audit_path: Path,
    *,
    repo: Path = ROOT,
    mac_side: bool = False,
) -> dict[str, Any]:
    """Reject drafts, changed rules/pins/source, non-RATIFY sizing or an unbound Round 0 verdict."""
    bank = check_audit(bank_path, audit_path)
    check_pins(repo)
    required = {
        "schema": SCHEMA,
        "status": "RATIFIED",
        "registration_sha256": sha256(repo / "research/t34_round0d/PREREGISTRATION_ROUND0D.md"),
        "bank_sha256": sha256(bank_path),
        "audit_sha256": sha256(audit_path),
        "pins": PINNED,
    }
    if any(registration.get(k) != v for k, v in required.items()):
        raise ValueError("Round 0-D is not ratified for these exact rules, pins and bank")
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
    if len(str(registration.get("a5d", {}).get("sha256", ""))) != 64:
        raise ValueError("ratification lacks the A5-D evidence binding")
    verdict = registration.get("round0_verdict", {})
    if verdict.get("route") != "ROUND0D" or len(str(verdict.get("sha256", ""))) != 64:
        raise ValueError("ratification lacks a bound governing Round 0 scientific NO-GO")
    if mac_side and sha256(Path(verdict["path"])) != verdict["sha256"]:
        raise ValueError("governing Round 0 verdict bytes changed after activation")
    from research.t34_round0d.history import policy

    policy(registration)
    sizing = registration.get("sizing")
    if (
        not isinstance(sizing, dict)
        or not sizing.get("chunks")
        or sizing.get("action") != "RATIFY"
        or sizing.get("expected_units", 1e9) > 120.0
        or sizing.get("derived_ceiling", 1e9) > 120.0
        or sizing.get("study_unit_ceiling") != 120.0
        or sizing.get("chunk_admission_ceiling")
        != 120.0 - float(sizing.get("replay", {}).get("unit_cap", 1e9))
        or sizing.get("first_stranded_chunk") is not None
        or sizing.get("straggler_total_units", 1e9) > 120.0
        or sizing.get("max_in_flight") != 2
        or sizing.get("wall_hours_forecast", 1e9) > 24.0
        or sizing.get("teacher_worlds_per_hour", 0) < 5.0
    ):
        raise ValueError("frozen rate-to-action table does not say RATIFY")
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
    """Write the non-dispatchable DRAFT record with the ESTIMATED rate-to-action table."""
    from research.t34_round0d.sizing import (
        KILL_WORLDS_PER_HOUR,
        STUDY_UNIT_CAP,
        WALL_HOURS_CAP,
        dense_r1_plan,
        estimated_plan,
    )

    bank = check_audit(BANK, AUDIT)
    check_pins(ROOT)
    table = {}
    for rate in (4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 15.0, 23.0):
        row = estimated_plan(rate)["plan"]
        table[f"{rate:g}"] = {
            key: row.get(key)
            for key in (
                "action",
                "reason",
                "chunk_pairs",
                "expected_units",
                "derived_ceiling",
                "straggler_total_units",
                "first_stranded_chunk",
                "chunk_admission_ceiling",
                "study_unit_ceiling",
                "wall_hours_forecast",
            )
        } | {"chunks": len(row.get("chunks", []))}
    payload = {
        "schema": "t34-round0d-registration-draft/v3",
        "status": "DRAFT",
        "ratified": False,
        "dispatch_authorized": False,
        "release_authorized": False,
        "built_from_commit": head(),
        "dispatch_condition": (
            "the published governing Round 0 verdict "
            "(<round0 root>/amendment-boost-floor/decision.json, governing: true) is a "
            "scientific STOP or INCONCLUSIVE (RULING_round0_go_file.md §3); every other code "
            "shelves Round 0-D"
        ),
        "preregistration": {"path": PREREG.relative_to(ROOT).as_posix(), "sha256": sha256(PREREG)},
        "pins": PINNED,
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
        "rate_to_action": {
            "kill_below_worlds_per_lane_hour": KILL_WORLDS_PER_HOUR,
            "expected_units_cap": STUDY_UNIT_CAP,
            "wall_hours_cap_on_2_lanes": WALL_HOURS_CAP,
            "derived_ceiling_gate": "1.2 x expected + (lanes - 1) x max_chunk(cap - expected) <= 120",  # noqa: E501
            "straggler_schedule_gate": (
                "chunks charged setup + ceil(n/12) x max(slowest, wave) at 2.0 u/h; 2-lane "
                "admission (in-flight and new chunk at full cap) against 120 - replay cap admits "
                "every chunk, and straggler total + replay cap <= 120; else KILL"
            ),
            "wall_gate": "straggler makespans + 10 min gap per chunk on the longer lane + replay <= 24 h",  # noqa: E501
            "replay_cap_reserved_inside_120u": True,
            "operative_study_ceiling_units": 120.0,
            "chunking": "multiples of 12 pairs; ceil(n/12) whole waves; timeouts from A5-D slowest world",  # noqa: E501
            "setup_minutes_rule": "max(15, ceil(observed A5-D setup) + 5)",
            "estimated_rows_with_placeholder_L4_costs": table,
        },
        "dense_r1_examples": {
            "r8_yield0.66_bearing0.5": dense_r1_plan(8.0, 0.66, 0.5),
            "r15_yield0.66_bearing0.5": dense_r1_plan(15.0, 0.66, 0.5),
        },
        "a5d": {
            "timeout_minutes": 300,
            "unit_cap": 10.0,
            "may_run_pre_verdict": True,
            "hold_resolution": "pre-registered Mac re-run (a5d.py resolve-hold)",
        },
        "dispatch_window_hours": 48,
        "blockers": [
            "independent review and judge approval of these exact bytes on main",
            "published governing Round 0 verdict routing to Round 0-D",
            "source-current excluded A6 Mac preflight after the final commit",
            "A5-D receipt with fidelity PASS/EXPLAINED_TIE and a RATIFY table row",
            "operator activation JSON committed and pushed to main before first dispatch",
        ],
    }
    write_once(out, payload)
    return payload


def ratify(
    *,
    operator: str,
    a5d_path: Path,
    preflight_record: Path,
    round0_decision: Path,
    ledger_repo: Path,
    out: Path,
    study_root: Path = STUDY_ROOT,
    repo: Path = ROOT,
    fallback_refusal: Path | None = None,
) -> dict[str, Any]:
    """Operator-only activation after approval, the governing verdict, A6 and A5-D."""
    from research.t34_round0d.sizing import plan, validate

    if not operator.strip():
        raise ValueError("operator name is required")
    check_audit(BANK, AUDIT)
    check_pins(repo)
    verdict = bind_round0_verdict(round0_decision, ledger_repo, fallback_refusal=fallback_refusal)
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
    if sizing.get("action") != "RATIFY":
        raise ValueError(
            f"frozen rate-to-action table: {sizing.get('action')} ({sizing.get('reason')})"
        )
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
        "round0_verdict": verdict,
        "pins": PINNED,
        "sizing": sizing,
        "registration_sha256": sha256(PREREG),
        "bank_sha256": sha256(BANK),
        "audit_sha256": sha256(AUDIT),
        "a5d": {"path": str(a5d_path.resolve()), "sha256": sha256(a5d_path)},
        "preflight": {
            "path": str(preflight_record.resolve()),
            "sha256": sha256(preflight_record),
            "source_hashes": hashes,
        },
        "source_hashes": hashes,
        "release_authorized": False,
        "scope": "Round 0-D development premise test only (route ROUND0D of the governing verdict)",
    }
    require_registration(payload, BANK, AUDIT, repo=repo, mac_side=True)
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
    rat.add_argument("--round0-decision", type=Path, required=True)
    rat.add_argument("--round0-ledger-repo", type=Path, required=True)
    rat.add_argument("--fallback-refusal", type=Path)
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
            round0_decision=args.round0_decision,
            ledger_repo=args.round0_ledger_repo,
            fallback_refusal=args.fallback_refusal,
            out=args.out,
        )


if __name__ == "__main__":
    main()
