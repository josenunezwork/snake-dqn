"""Owner tool: record the frp3-s12 strict receipt in the served-checkpoint pins (once).

Run ONLY after the checkpoint swap's strict gate (frp3-s12 + released v8 vs the champion +
v8) closed STRICT_PASS. It reads the strict run's receipt, intent and audit report and refuses
unless :func:`evidence_problems` is empty (the independent audit says PASS with recomputed
outcome STRICT_PASS, the receipt binds the intent and audit report bytes, and the intent's
candidate arm is frp3-s12 with the v8 method), then replaces the ``null`` placeholder of
``checkpoints["frp3-s12"].strict_receipt`` in ``web/backend/served_checkpoint_pins.json``
with the three files' paths and sha256s. It refuses to overwrite a filled pin. Commit the
pins file afterwards; nothing is served until a session is built with
``SNAKE_SERVE_CHECKPOINT=frp3-s12`` (or the release step flips the default).

Usage::

    ./venv/bin/python research/frp3_checkpoint_serving_20261005/fill_strict_pin.py \\
        --root <strict run root> --receipt output/receipt.json --intent intent.json \\
        --audit-report <path of the audit report, relative to root> [--dry-run]
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

NAME = "frp3-s12"
CHECKPOINT_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"
PASS_WORD = "STRICT_PASS"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refusing: {path} is not readable JSON: {exc}")


def evidence_problems(root: Path, files: Mapping[str, str]) -> List[str]:
    """Structured STRICT_PASS check of a strict run (the sequential strict template's shapes,
    as the v8 strict run wrote them). [] = the evidence authorizes frp3-s12 + v8.

    * ``audit_report``: ``status == "PASS"`` and ``recomputed.expected_outcome ==
      "STRICT_PASS"`` (the independent audit's verdict, not the producer's claim);
    * ``receipt``: ``intent_sha256`` and ``audit_report_sha256`` equal the two files' bytes;
    * ``intent``: ``arms.candidate.checkpoint_sha256`` is frp3-s12's and
      ``arms.candidate.method`` is the released v8 method (frp3 is the challenger arm).
    """
    out = []
    paths = {}
    for key in ("receipt", "intent", "audit_report"):
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
    receipt, intent, report = (_json(paths[k]) for k in ("receipt", "intent", "audit_report"))
    if not isinstance(report, Mapping) or report.get("status") != "PASS":
        out.append("audit report status is not PASS")
    elif (report.get("recomputed") or {}).get("expected_outcome") != PASS_WORD:
        out.append(f"audit report recomputed.expected_outcome is not {PASS_WORD}")
    if not isinstance(receipt, Mapping):
        out.append("receipt is not an object")
    else:
        if receipt.get("intent_sha256") != _sha(paths["intent"]):
            out.append("receipt.intent_sha256 is not the intent file's sha256")
        if receipt.get("audit_report_sha256") != _sha(paths["audit_report"]):
            out.append("receipt.audit_report_sha256 is not the audit report's sha256")
    candidate = ((intent.get("arms") or {}) if isinstance(intent, Mapping) else {}).get(
        "candidate"
    ) or {}
    if candidate.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        out.append(f"intent arms.candidate.checkpoint_sha256 is not {NAME}'s")
    if candidate.get("method") != V8_METHOD:
        out.append(f"intent arms.candidate.method is not {V8_METHOD!r}")
    return out


def build_pin(root: Path, receipt: str, intent: str, audit_report: str) -> Dict[str, Any]:
    """The filled ``strict_receipt`` value (raises ``SystemExit`` on any refusal)."""
    root = root.resolve()
    files = {"receipt": receipt, "intent": intent, "audit_report": audit_report}
    problems = evidence_problems(root, files)
    if problems:
        raise SystemExit(f"refusing: {problems}")
    out: Dict[str, Any] = {"verdict": PASS_WORD, "root": str(root)}
    for key, rel in files.items():
        out[key] = {"path": rel, "sha256": _sha(root / rel)}
    return out


def fill(pins_path: Path, pin: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    from web.backend import served_checkpoint as registry

    doc = json.loads(pins_path.read_text(encoding="utf-8"))
    entry = (doc.get("checkpoints") or {}).get(NAME)
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
    parser.add_argument("--root", type=Path, required=True, help="the strict run root")
    parser.add_argument("--receipt", required=True, help="receipt path relative to --root")
    parser.add_argument("--intent", required=True, help="intent path relative to --root")
    parser.add_argument("--audit-report", required=True, help="audit report relative to --root")
    parser.add_argument("--pins", type=Path, default=Path(registry.PINS_PATH))
    parser.add_argument("--dry-run", action="store_true", help="validate and print only")
    args = parser.parse_args(argv)
    pin = build_pin(args.root, args.receipt, args.intent, args.audit_report)
    doc = fill(args.pins, pin, dry_run=args.dry_run)
    print(json.dumps(doc["checkpoints"][NAME], indent=2))
    print(("validated (dry run, nothing written)" if args.dry_run else f"wrote {args.pins}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
