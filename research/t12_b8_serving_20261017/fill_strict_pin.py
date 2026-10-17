"""Fill the b8-s3 serving pin only from the completed T7 decision evidence.

This owner-only tool binds the selected B8 seed-3 checkpoint to T7's producer decision,
isolated audit, and independent result review. It refuses unless the independent audit passes
all structural checks and the independent review recomputes ``STRICT_PASS``. Existing pins are
immutable: a filled entry is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NAME = "b8-s3"
CHECKPOINT_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
PASS_WORD = "STRICT_PASS"
FILES = ("decision", "audit_report", "independent_review")
ARTIFACT_ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t7-phase-r-b8-20261016"
)
IMMUTABLE_SHA256 = {
    "decision": "8fd97ecc564fdc31a1d77ed50b8112dcffac42d736db20b2b410b67c6ef139fb",
    "audit_report": "c631930c0af441e72e304a08cea401ed1a494c02c3e73071a8a459fb33dde07e",
    "independent_review": "135b1b8b5736619dbf1d57460d606c3a60a4ff4f59019c4ee51392dc5e2aecbe",
    "plan": "76731e5f25d6327e6e4e21cd084151333c6889b1e26936e3f00d065982275327",
    "start": "ae9e195d27525325ae08d1c141c8aaa510303521096db2663c237f2afc951344",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refusing: {path} is not readable JSON: {exc}")


def _same_sha(left: Path, right: Path, label: str, out: List[str]) -> None:
    if not right.is_file() or _sha(left) != _sha(right):
        out.append(f"{label} bytes do not match the immutable artifact copy")


def _recompute_t7(
    root: Path,
    decision_path: Path,
    decision: Mapping[str, Any],
    out: List[str],
) -> None:
    """Re-run the frozen T7 pure rule against hash-bound raw records, read-only."""
    from research.redesign_t7_phase_r_20261016 import phase_r_t7, rule_spec_t7

    plan_path, start_path = root / "plan.json", root / "start.json"
    if not plan_path.is_file() or not start_path.is_file():
        out.append("decision root lacks plan.json or start.json")
        return
    plan, start = _json(plan_path), _json(start_path)
    artifact_plan = ARTIFACT_ROOT / "plan.json"
    artifact_start = ARTIFACT_ROOT / "start.json"
    artifact_decision = ARTIFACT_ROOT / "decision.json"
    for source, artifact, label in (
        (plan_path, artifact_plan, "plan"),
        (start_path, artifact_start, "start"),
        (decision_path, artifact_decision, "decision"),
    ):
        _same_sha(source, artifact, label, out)
    if not isinstance(plan, Mapping) or plan.get("schema") != "t7-phase-r-plan/v1":
        out.append("plan schema is invalid")
        return
    if decision.get("plan_sha256") != _sha(plan_path) or start.get("plan_sha256") != _sha(
        plan_path
    ):
        out.append("decision/start do not bind plan bytes")
    if decision.get("start_sha256") != _sha(start_path):
        out.append("decision does not bind start bytes")
    selected = (plan.get("checkpoints") or {}).get("B8-noveto|s3")
    if not isinstance(selected, Mapping) or selected.get("sha256") != CHECKPOINT_SHA256:
        out.append("plan B8-noveto|s3 checkpoint is not the registry sha256")
    registered_rule = rule_spec_t7.rule()
    if plan.get("rule") != registered_rule or plan.get("rule_sha256") != phase_r_t7.canonical_sha(
        registered_rule
    ):
        out.append("current frozen T7 rule does not match the plan")
        return
    inventory = decision.get("records")
    if not isinstance(inventory, list) or len(inventory) != 5805:
        out.append("decision record inventory is not the complete 5805-record set")
        return
    records = []
    for item in inventory:
        rel = item.get("path") if isinstance(item, Mapping) else None
        digest = item.get("sha256") if isinstance(item, Mapping) else None
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            out.append("decision inventory has an unsafe record path")
            return
        path = ARTIFACT_ROOT / rel
        if not path.is_file() or _sha(path) != digest:
            out.append(f"record inventory hash mismatch: {rel}")
            return
        records.append(_json(path))
    nested = {
        (row.get("hero"), row.get("seed"), row.get("mix"), row.get("world_seed")): row
        for row in records
        if not row.get("control")
    }
    controls = [row for row in records if row.get("control")]
    prefix_ok = len(controls) == 45 and all(
        phase_r_t7.prefix_identical(
            row.get("record", {}),
            nested.get(
                (row.get("hero"), row.get("seed"), row.get("mix"), row.get("world_seed")), {}
            ).get("prefix_h5000", {}),
        )
        for row in controls
    )
    recomputed = rule_spec_t7.decide(
        [row for row in records if not row.get("control")],
        {int(key): value for key, value in plan["banks"].items()},
        prefix_ok,
    )
    if recomputed != decision.get("decision"):
        out.append("T7 rule recomputation differs from decision.json")
    if recomputed.get("status") != PASS_WORD or recomputed.get("valid") is not True:
        out.append("T7 rule recomputation is not a valid STRICT_PASS")


def evidence_problems(root: Path, files: Mapping[str, str]) -> List[str]:
    """Return reasons T7 evidence cannot authorize b8-s3 ([] authorizes it)."""
    out: List[str] = []
    paths: Dict[str, Path] = {}
    for key in FILES:
        rel = files.get(key)
        if (
            not isinstance(rel, str)
            or not rel
            or Path(rel).is_absolute()
            or ".." in Path(rel).parts
        ):
            out.append(f"{key} path must be relative to the root without '..'")
            continue
        paths[key] = root / rel
        if not paths[key].is_file():
            out.append(f"{key} file missing: {paths[key]}")
    if out:
        return out
    decision, audit, review = (_json(paths[key]) for key in FILES)
    if not isinstance(audit, Mapping) or audit.get("schema") != "t7-phase-r-audit/v1":
        out.append("isolated audit schema is not t7-phase-r-audit/v1")
    elif audit.get("coverage") != {"expected": 5805, "found": 5805, "exact": True} or audit.get(
        "errors"
    ) not in (None, []):
        out.append("isolated audit coverage or errors are not PASS")
    if not isinstance(audit, Mapping) or not all(
        audit.get(key) is True for key in ("plan_valid", "records_valid", "decision_valid")
    ):
        out.append("isolated audit is not PASS")
    if not isinstance(decision, Mapping):
        out.append("decision is not an object")
        return out
    _recompute_t7(root, paths["decision"], decision, out)
    decision_body = decision.get("decision")
    if not isinstance(decision_body, Mapping) or decision_body.get("status") != PASS_WORD:
        out.append(f"decision.status is not {PASS_WORD}")
    selection = (
        decision_body.get("candidate_selection") if isinstance(decision_body, Mapping) else None
    )
    if not isinstance(selection, Mapping) or selection.get("seed") != 3:
        out.append("decision candidate_selection.seed is not 3")
    if not isinstance(review, Mapping) or review.get("verdict") != "PASS":
        out.append("independent review verdict is not PASS")
    if not isinstance(review, Mapping) or review.get("errors") != []:
        out.append("independent review errors are not empty")
    if not isinstance(review, Mapping) or review.get("candidate_seed") != 3:
        out.append("independent review candidate_seed is not 3")
    if not isinstance(review, Mapping) or review.get("decision") != PASS_WORD:
        out.append(f"independent review does not recompute {PASS_WORD}")
    evidence = review.get("evidence") if isinstance(review, Mapping) else None
    if not isinstance(evidence, Mapping):
        out.append("independent review has no evidence hashes")
    else:
        expected = {
            "decision": "decision.json",
            "audit_report": "audit.json",
            "plan": "plan.json",
            "start": "start.json",
        }
        for key, filename in expected.items():
            review_key = filename
            evidence_path = paths[key] if key in paths else root / filename
            if not evidence_path.is_file() or evidence.get(review_key) != _sha(evidence_path):
                out.append(f"independent review {review_key} sha256 does not bind {key}")
    for key, path in {
        **paths,
        "plan": root / "plan.json",
        "start": root / "start.json",
    }.items():
        if not path.is_file() or _sha(path) != IMMUTABLE_SHA256[key]:
            out.append(f"{key} sha256 is not the immutable T7 evidence hash")
    return out


def build_pin(
    root: Path, decision: str, audit_report: str, independent_review: str
) -> Dict[str, Any]:
    root = root.resolve()
    files = {
        "decision": decision,
        "audit_report": audit_report,
        "independent_review": independent_review,
    }
    problems = evidence_problems(root, files)
    if problems:
        raise SystemExit(f"refusing: {problems}")
    receipt: Dict[str, Any] = {"verdict": PASS_WORD, "root": str(root)}
    for key, rel in files.items():
        receipt[key] = {"path": rel, "sha256": _sha(root / rel)}
    return receipt


def fill(pins_path: Path, pin: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    from web.backend import served_checkpoint as registry

    doc = _json(pins_path)
    entry = (doc.get("checkpoints") or {}).get(NAME) if isinstance(doc, Mapping) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"refusing: {pins_path} has no {NAME} entry")
    if entry.get("strict_receipt") is not None:
        raise SystemExit(f"refusing: the {NAME} pin is already filled (never overwritten)")
    entry["strict_receipt"] = pin
    problems = registry.strict_pin_problems(NAME, doc)
    if problems:
        raise SystemExit(f"refusing: the filled pin would not validate: {problems}")
    if not dry_run:
        pins_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    from web.backend import served_checkpoint as registry

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--independent-review", required=True)
    parser.add_argument("--pins", type=Path, default=Path(registry.PINS_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    pin = build_pin(args.root, args.decision, args.audit_report, args.independent_review)
    doc = fill(args.pins, pin, dry_run=args.dry_run)
    print(json.dumps(doc["checkpoints"][NAME], indent=2))
    print("validated (dry run, nothing written)" if args.dry_run else f"wrote {args.pins}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
