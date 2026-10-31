"""Mac-side binding of Astra's pre-data confirmation ruling for Watch6 V3.

Parses an explicit approved verdict (`APPROVE` or `APPROVE-WITH-CONDITIONS`) from
Astra's written ruling Markdown and refuses `REJECT`, `NOT_RATIFIED`, missing, or
conflicting verdicts both when recording `astra-ruling.json` and when validating it
before live dispatch or final analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from tools.ops.package_spec import digest, publish_json

SCHEMA = "watch6-v3-pre-data-ruling/v2"
ALLOWED_VERDICTS = ("APPROVE", "APPROVE-WITH-CONDITIONS")
ALLOWED_STATUS_TOKENS = {
    "RATIFIED",
    "APPROVED",
    "APPROVE",
    "APPROVE-WITH-CONDITIONS",
    "APPROVE_WITH_CONDITIONS",
    "RATIFIED-WITH-CONDITIONS",
    "RATIFIED_WITH_CONDITIONS",
}
DEFAULT_RULING_JSON = (
    p.ROOT.parent / "snake-dqn-artifacts/t34-sol/argon-v3-confirm-fix1/astra-ruling.json"
)

_VERDICT_LINE_RE = re.compile(
    r"^\s*(?:[-*+]\s+|#+\s+)?\*{0,2}`?(?:verdict|decision)`?\*{0,2}\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_STATUS_LINE_RE = re.compile(
    r"^\s*(?:[-*+]\s+|#+\s+)?\*{0,2}`?status`?\*{0,2}\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_LEADING_TOKEN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)")


def _clean_token(raw: str) -> str:
    stripped = raw.strip().lstrip("*`_~ ").rstrip("*`_~ ")
    match = _LEADING_TOKEN_RE.match(stripped)
    if match is None:
        return ""
    return match.group(1).upper()


def parse_ruling_verdict(markdown_text: str) -> str:
    """Parse explicit `Verdict: APPROVE` or `Verdict: APPROVE-WITH-CONDITIONS` and refuse else."""
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        raise ValueError("REFUSE: empty ruling markdown")

    verdicts: list[str] = []
    in_fence = False
    for line in markdown_text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        status_match = _STATUS_LINE_RE.match(line)
        if status_match is not None:
            status_raw = status_match.group(1)
            status_token = _clean_token(status_raw)
            if status_token not in ALLOWED_STATUS_TOKENS:
                raise ValueError(f"REFUSE: ruling STATUS is not ratified: {status_raw!r}")

        verdict_match = _VERDICT_LINE_RE.match(line)
        if verdict_match is not None:
            verdict_raw = verdict_match.group(1)
            verdict_token = _clean_token(verdict_raw).replace("_", "-")
            if verdict_token not in ALLOWED_VERDICTS:
                raise ValueError(
                    f"REFUSE: ruling verdict must be APPROVE or APPROVE-WITH-CONDITIONS, "
                    f"got {verdict_raw!r}"
                )
            verdicts.append(verdict_token)

    if not verdicts:
        raise ValueError(
            "REFUSE: ruling markdown lacks explicit 'Verdict: APPROVE' or "
            "'Verdict: APPROVE-WITH-CONDITIONS' line"
        )
    if len(set(verdicts)) != 1:
        raise ValueError(f"REFUSE: conflicting Verdict lines in ruling markdown: {verdicts}")
    return verdicts[0]


def resolve_ruling_json(override: Path | None = None) -> Path:
    """Resolve the active Mac-side ruling receipt path."""
    if override is not None:
        return override.resolve()
    env = os.environ.get("WATCH6_V3_RULING_JSON")
    if env:
        return Path(env).resolve()
    return DEFAULT_RULING_JSON.resolve()


def record_ruling(ruling_md: Path, out: Path | None = None, *, judge: str = "Astra") -> dict:
    """Bind Astra's approved pre-data ruling markdown to REGISTRATION_V3.md and science."""
    ruling_md = ruling_md.resolve(strict=True)
    if not ruling_md.is_file() or ruling_md.stat().st_size == 0:
        raise ValueError(f"ruling markdown must be a nonempty regular file: {ruling_md}")
    if judge not in ("Astra", "Fable"):
        raise ValueError("independent pre-data judge must be Astra (or Fable stand-in)")
    verdict = parse_ruling_verdict(ruling_md.read_text(encoding="utf-8"))
    target = resolve_ruling_json(out)
    payload = {
        "schema": SCHEMA,
        "status": "RATIFIED",
        "verdict": verdict,
        "judge": judge,
        "implementer": "Argon",
        "independent_implementation_review": True,
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "registration_path": str(p.registration()),
        "registration_sha256": digest(p.registration()),
        "ruling_path": str(ruling_md),
        "ruling_sha256": digest(ruling_md),
    }
    publish_json(target, payload)
    return payload


def validate_ruling_receipt(path: Path | None = None) -> dict:
    """Fail closed if Astra's pre-data ruling receipt is absent, unapproved, or mismatched."""
    target = resolve_ruling_json(path)
    if not target.is_file():
        raise ValueError(
            f"REFUSE: missing ratified Astra pre-data ruling receipt at {target} "
            "(run `python -m research.watch6_search_efficacy_v3.ratify --ruling <RULING_V3.md>`)"
        )
    receipt = json.loads(target.read_text(encoding="utf-8"))
    ruling_md = Path(receipt.get("ruling_path", ""))
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "RATIFIED"
        or receipt.get("verdict") not in ALLOWED_VERDICTS
        or receipt.get("judge") not in ("Astra", "Fable")
        or receipt.get("implementer") != "Argon"
        or receipt.get("namespace") != p.NAMESPACE
        or receipt.get("science") != p.identity()
        or receipt.get("registration_sha256") != digest(p.registration())
        or not ruling_md.is_absolute()
        or not ruling_md.is_file()
        or digest(ruling_md) != receipt.get("ruling_sha256")
    ):
        raise ValueError("REFUSE: Astra pre-data ruling receipt is stale, invalid, or mismatched")
    parsed_verdict = parse_ruling_verdict(ruling_md.read_text(encoding="utf-8"))
    if parsed_verdict != receipt.get("verdict"):
        raise ValueError("REFUSE: ruling markdown verdict does not match recorded receipt")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruling", type=Path, required=True, help="Path to Astra's RULING_V3.md")
    parser.add_argument("--out", type=Path, default=None, help="Output path for astra-ruling.json")
    parser.add_argument("--judge", default="Astra", choices=("Astra", "Fable"))
    args = parser.parse_args()
    receipt = record_ruling(args.ruling, args.out, judge=args.judge)
    validate_ruling_receipt(args.out)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
