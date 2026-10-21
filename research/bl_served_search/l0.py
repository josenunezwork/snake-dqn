"""Saved-timeline L0 replay for the Path B deliverability smoke.

This is development-only accounting.  It never reconstructs a game, changes a
controller, or substitutes fields from another smoke artifact for a missing
``lad-*.json`` timeline.  The latter rule is deliberate: Round 5 section (f)
requires an INCOMPLETE result if the retained LAD records cannot support the
counterfactual replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import export_bytes, export_json

SMOKE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-smoke-20261021")
ARTIFACT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2")
SPEEDUPS = (1, 2, 3, 5, 10, 20)
LAD_WINDOWS = ((16, 21), (88, 100), (248, 254))
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def digest(path: Path) -> str:
    """Return the SHA-256 of one immutable input."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one explicitly named JSON source."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"object expected: {path}")
    return value


def case_id(path: Path, prefix: str) -> str:
    """Return the common interval id from an immutable smoke filename."""
    if not path.name.startswith(prefix) or path.suffix != ".json":
        raise ValueError(f"unexpected smoke filename: {path.name}")
    return path.stem[len(prefix) :]


def timeline_entries(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Accept only a complete recorded evaluation timeline with required fields."""
    entries = record.get("evaluation_timeline")
    if not isinstance(entries, list) or not entries:
        return None
    if not all(
        isinstance(entry, dict)
        and isinstance(entry.get("completion_seconds"), (int, float))
        and isinstance(entry.get("candidate_frame"), int)
        for entry in entries
    ):
        return None
    return entries


def inspect(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Audit retained LAD schema and fixed T3 rescue/delay denominator metadata."""
    lad_paths = sorted(source.glob("lad-*.json"))
    n1_paths = sorted(source.glob("n1-*.json"))
    hashes = {str(path): digest(path) for path in [*lad_paths, *n1_paths]}
    lad = {case_id(path, "lad-"): load_json(path) for path in lad_paths}
    n1 = {case_id(path, "n1-"): load_json(path) for path in n1_paths}
    missing = [
        {
            "id": identifier,
            "missing": ["evaluation_timeline", "completion_seconds", "candidate_frame"],
        }
        for identifier, record in lad.items()
        if timeline_entries(record) is None
    ]
    t3 = [record for record in n1.values() if record.get("group") == "T3"]
    rescue = sum(bool(record.get("historical_survival_rescue")) for record in t3)
    delay = len(t3) - rescue
    # N1 is only a schema diagnostic: its search rows describe request batches
    # and its fully_evaluated list retains improving candidates only.  It is not
    # an allowed replacement for the LAD-record timeline.
    n1_diagnostic = {
        "linked_lad_ids": len(set(lad) & set(n1)),
        "n1_complete_rows": sum(record.get("status") == "COMPLETE" for record in n1.values()),
        "n1_search_fields": sorted(
            {key for record in n1.values() for row in record.get("searches", []) for key in row}
        ),
        "n1_improving_fields": sorted(
            {
                key
                for record in n1.values()
                for row in record.get("fully_evaluated_improvements", [])
                for key in row
            }
        ),
        "n1_is_not_substituted": True,
    }
    return {
        "lad_records": len(lad),
        "n1_records": len(n1),
        "rescue_capable": rescue,
        "delay_only": delay,
        "other_records": len(n1) - len(t3),
        "missing_timeline_records": len(missing),
        "missing_timeline_examples": missing[:5],
        "n1_diagnostic": n1_diagnostic,
        "can_replay": not missing and len(lad) == 80 and rescue == 34 and delay == 41,
    }, hashes


def result_markdown(receipt: dict[str, Any]) -> str:
    """Render the compact human delivery without asserting unavailable estimates."""
    audit = receipt["input_audit"]
    return "\n".join(
        [
            "# L0 speed-up-to-delivery replay — 2026-10-21",
            "",
            (
                "**Verdict: INCOMPLETE.** The 80 retained `lad-*.json` records do not contain "
                "a per-evaluation completion timeline or candidate frames, so Round 5 section (f) "
                "forbids the k× replay."
            ),
            "",
            (
                "This development-only result is excluded from registered verdicts and makes no "
                "serving, "
                "release, paced-identity, or trajectory-efficacy claim."
            ),
            "",
            (
                f"- Inputs audited: {audit['lad_records']} LAD records; "
                f"{audit['n1_records']} linked "
                "N=1 controller records used only to establish the immutable 75-T3 denominator and "
                "schema diagnostic."
            ),
            (
                f"- Fixed T3 accounting: {audit['rescue_capable']} rescue-capable intervals and "
                f"{audit['delay_only']} delay-only intervals; "
                f"{audit['other_records']} M3 records are "
                "outside that denominator."
            ),
            (
                f"- Missing required timelines: {audit['missing_timeline_records']}/"
                f"{audit['lad_records']}. "
                "Each LAD record has only `kind`, `lad`, `outcomes`, `probes`, and `rule`."
            ),
            (
                "- The linked N=1 records retain batch `searches` and improving-only observations, "
                "not a complete per-evaluation timeline. They were not substituted for "
                "the required "
                "LAD data."
            ),
            (
                "- Therefore no chronological or LAD-first delivery rate, no 50%/80% crossing k, "
                "and no k=20 doom-to-LAD kill decision is estimable from these records."
            ),
            "",
            (
                "The prescribed replay grid remains k={1,2,3,5,10,20}; LAD-first priorities "
                "remain the 16–21, 88–100, and 248–254 decision-before-doom windows. A future "
                "measurement needs "
                "to retain every requested candidate frame and its completion time prospectively."
            ),
            "",
        ]
    )


def run(source: Path, artifact_root: Path) -> dict[str, Any]:
    """Perform the bounded saved-data audit and publish create-only exports."""
    started = time.monotonic()
    audit, hashes = inspect(source)
    receipt = {
        "schema": "bl-served-search-l0/v1",
        "development_only": True,
        "excluded_from_registered_verdicts": True,
        "status": "COMPLETE" if audit["can_replay"] else "INCOMPLETE",
        "verdict": "INCOMPLETE_MISSING_LAD_EVALUATION_TIMELINES",
        "rule": (
            "Round-05 JUDGMENT.md section (f): missing completion times or candidate frames => "
            "INCOMPLETE"
        ),
        "speedups": list(SPEEDUPS),
        "lad_first_windows_before_doom": [list(window) for window in LAD_WINDOWS],
        "input_audit": audit,
        "input_sha256": hashes,
        "execution": {
            "nice": os.nice(0),
            "max_processes": 4,
            "numerical_threads": 1,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    if receipt["status"] == "COMPLETE":
        raise RuntimeError(
            "timeline replay publication requires a separately reviewed complete path"
        )
    replay_dir = artifact_root / "l0"
    if artifact_root.exists() and any(
        (artifact_root / name).exists() for name in ("RESULT_L0.md", "l0_receipt.json")
    ):
        raise FileExistsError("L0 parent exports already exist")
    if replay_dir.exists():
        raise FileExistsError(f"L0 export directory already exists: {replay_dir}")
    export_json(replay_dir / "input_audit.json", receipt)
    export_bytes(artifact_root / "RESULT_L0.md", result_markdown(receipt).encode())
    receipt_export = export_json(artifact_root / "l0_receipt.json", receipt)
    receipt["receipt_export"] = receipt_export
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SMOKE)
    parser.add_argument("--out", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    if os.nice(0) < 10:
        parser.error("run under nice -n 10 or lower priority")
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        parser.error("all numerical thread caps must explicitly equal one")
    receipt = run(args.source, args.out)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "verdict": receipt["verdict"],
                "input_audit": receipt["input_audit"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
