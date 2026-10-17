#!/usr/bin/env python3
"""Refresh the fail-closed T11 screen audit immediately before probe launch.

The immutable LOCAL30 audit owns the registered-history scan.  This wrapper
selects one active round and, for the sole extension, permits only a fully
validated transcript of the completed initial round to be excluded from that
scan.  It allocates no worlds and starts no gameplay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.redesign_m2_20261008 import dev_probe
from research.t11_local30_20261017 import seed_audit

SCHEMA = "t11-screen-freshness-audit/v1"
INITIAL_ROUND = seed_audit.INITIAL_ROUND
EXTENSION_ROUND = seed_audit.EXTENSION_ROUND
ARM = "LOCAL30"
_MIX_WORLDS = {"scripted": 96, "frozen": 32, "mixed": 32}
_INITIAL_NAMES = (
    *[f"CTRL_s{seed}" for seed in range(10, 15)],
    "baseline",
    *[f"{ARM}_s{seed}" for seed in range(10, 15)],
)


def _sha256(path: Path) -> str:
    """Return a SHA256 digest for a bound evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_initial_jobs() -> set[tuple[str, str]]:
    """Return the complete registered round-74 checkpoint/mix matrix."""
    return {(name, mix) for name in _INITIAL_NAMES for mix in _MIX_WORLDS}


def _validated_round74_evidence(
    evidence: str | Path, repo: Path
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Validate exact completed round-74 evidence before allowing its exclusion.

    Excluding a transcript is safe only when it proves every registered
    round-74 row, all with the fixed roster and mix cardinality.  The colocated
    initial receipt is independently bound as well, preventing a hand-written
    partial transcript from hiding a prior allocation.
    """
    path = Path(evidence).resolve()
    if not path.is_file():
        raise FileNotFoundError("completed round-74 probe transcript is missing")
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ValueError("completed round-74 transcript is not JSONL") from error
    rosters = seed_audit.candidate_rosters()
    expected_seeds = set(rosters[str(INITIAL_ROUND)])
    expected_jobs = _expected_initial_jobs()
    observed_jobs: set[tuple[str, str]] = set()
    rows_by_job: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("completed round-74 transcript contains a non-object row")
        name, mix = row.get("name"), row.get("mix")
        if (
            row.get("round_index") != INITIAL_ROUND
            or not isinstance(name, str)
            or mix not in _MIX_WORLDS
        ):
            raise ValueError("completed round-74 transcript has invalid job metadata")
        key = (name, mix)
        if key in observed_jobs:
            raise ValueError("completed round-74 transcript contains duplicate jobs")
        observed_jobs.add(key)
        rows_by_job[key] = row
        worlds = _MIX_WORLDS[mix]
        if row.get("worlds") != worlds:
            raise ValueError("completed round-74 transcript has invalid world count")
        expected_row_seeds = set(dev_probe.probe_seeds(worlds, INITIAL_ROUND))
        for field in ("mass", "deaths", "alive_frames"):
            values = row.get(field)
            if not isinstance(values, dict):
                raise ValueError(f"completed round-74 transcript lacks {field} map")
            try:
                actual = {int(seed) for seed in values}
            except (TypeError, ValueError) as error:
                raise ValueError("completed round-74 transcript has invalid seed keys") from error
            if actual != expected_row_seeds:
                raise ValueError(
                    "completed round-74 transcript does not bind its registered roster"
                )
    if observed_jobs != expected_jobs:
        raise ValueError("completed round-74 transcript does not bind every registered job")

    # The watcher saves every completed row before materializing probes_round74.jsonl.
    # They contain seed literals and round metadata too, so extension freshness can
    # exclude them only after byte-for-data binding them to the validated transcript.
    from research.t2_screen_20261015.watch_t2b import row_path

    raw_rows: list[Path] = []
    for name, mix in sorted(expected_jobs):
        raw_path = row_path(
            path.parent,
            {
                "name": name,
                "mix": mix,
                "worlds": _MIX_WORLDS[mix],
                "round_index": INITIAL_ROUND,
            },
        ).resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(f"completed round-74 row is missing: {raw_path}")
        try:
            raw = json.loads(raw_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"completed round-74 row is not JSON: {raw_path}") from error
        if raw != rows_by_job[(name, mix)]:
            raise ValueError(f"completed round-74 row differs from transcript: {raw_path}")
        raw_rows.append(raw_path)

    initial_audit = path.parent / "seed_bank_audit_round74.json"
    if not initial_audit.is_file():
        raise FileNotFoundError("completed round-74 audit receipt is missing")
    try:
        receipt = json.loads(initial_audit.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("completed round-74 audit receipt is not JSON") from error
    check = (
        receipt.get("checks", {}).get(str(INITIAL_ROUND), {}) if isinstance(receipt, dict) else {}
    )
    if not (
        isinstance(receipt, dict)
        and receipt.get("schema") == SCHEMA
        and receipt.get("round_index") == INITIAL_ROUND
        and receipt.get("pass") is True
        and check.get("seeds") == sorted(expected_seeds)
    ):
        raise ValueError("completed round-74 audit receipt does not bind the expected roster")
    return (
        (path, initial_audit.resolve(), *raw_rows),
        [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "kind": "validated-owned-round74-probe-transcript",
                "rows": len(rows),
                "world_count": len(expected_seeds),
            },
            {
                "path": str(initial_audit.resolve()),
                "sha256": _sha256(initial_audit),
                "kind": "validated-owned-round74-audit-receipt",
            },
            {
                "paths": [str(raw_path) for raw_path in raw_rows],
                "sha256": {str(raw_path): _sha256(raw_path) for raw_path in raw_rows},
                "kind": "validated-owned-round74-raw-rows",
                "rows": len(raw_rows),
            },
        ],
    )


def _base_audit_with_exclusions(exclusions: tuple[Path, ...], **kwargs: Any) -> dict[str, Any]:
    """Run the immutable audit while excluding only validated exact evidence files."""
    original_tokens = seed_audit.t6_seed_audit.concrete_text_hits
    original_metadata = seed_audit.t6_seed_audit._partitioned_t6_rg
    original_scope = seed_audit._round_metadata_scope

    def tokens(root: Path, values: list[str], ignored: tuple[Path, ...] = ()) -> Any:
        return original_tokens(root, values, tuple(dict.fromkeys((*ignored, *exclusions))))

    def metadata(root: Path, command: list[str], ignored: tuple[Path, ...], label: str) -> Any:
        return original_metadata(
            root, command, tuple(dict.fromkeys((*ignored, *exclusions))), label
        )

    def scope(root: Path, ignored: tuple[Path, ...]) -> dict[str, list[str]]:
        return original_scope(root, tuple(dict.fromkeys((*ignored, *exclusions))))

    seed_audit.t6_seed_audit.concrete_text_hits = tokens
    seed_audit.t6_seed_audit._partitioned_t6_rg = metadata
    seed_audit._round_metadata_scope = scope
    try:
        return seed_audit.audit(**kwargs)
    finally:
        seed_audit.t6_seed_audit.concrete_text_hits = original_tokens
        seed_audit.t6_seed_audit._partitioned_t6_rg = original_metadata
        seed_audit._round_metadata_scope = original_scope


def audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    out: str | Path | None = None,
    round_index: int = INITIAL_ROUND,
    consumed_round74_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """Return the current T11 round-74 or conditional-round-75 freshness receipt."""
    if round_index not in seed_audit.ROUNDS:
        raise ValueError(f"round_index must be one of {seed_audit.ROUNDS}")
    if round_index == INITIAL_ROUND and consumed_round74_evidence is not None:
        raise ValueError("round 74 cannot exclude prior-round evidence")
    if round_index == EXTENSION_ROUND and consumed_round74_evidence is None:
        raise ValueError("round 75 requires exact completed round-74 evidence")
    repo_path = Path(repo).resolve()
    output = Path(out).resolve() if out is not None else None
    documentary_paths: tuple[Path, ...] = ()
    documentary: list[dict[str, Any]] = []
    if round_index == EXTENSION_ROUND:
        documentary_paths, documentary = _validated_round74_evidence(
            consumed_round74_evidence, repo_path  # type: ignore[arg-type]
        )
    base = _base_audit_with_exclusions(
        documentary_paths, repo=repo_path, artifacts=artifacts, out=output
    )
    check = base.get("checks", {}).get(str(round_index), {})
    passed = bool(
        base.get("pass") is True and check.get("count") == 96 and check.get("unique") is True
    )
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": _sha256(Path(__file__).resolve()),
        "scope": (
            f"T11 evaluation round {round_index}; immutable registered and observed-history "
            "scan, including all historical namespaces and the live T7 namespace."
        ),
        "excluded_output": str(output) if output is not None else None,
        "round_index": round_index,
        "consumed_round74_evidence": (
            str(Path(consumed_round74_evidence).resolve())
            if consumed_round74_evidence is not None
            else None
        ),
        "exclusion_policy": (
            "Only exact evidence validated as this screen's complete round-74 transcript and "
            "receipt may be excluded for round 75; the immutable audit's exact preregistration "
            "and caller output exclusions remain in force."
        ),
        "validated_documentary_exclusions": documentary,
        "base_t11_audit": base,
        "checks": base.get("checks", {}),
        "limit": (
            "No claim is made about unrecorded experiments, opaque binary files, or external "
            "systems."
        ),
    }


def main() -> int:
    """Write one refreshable receipt without starting evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--round-index", type=int, default=INITIAL_ROUND)
    parser.add_argument("--consumed-round74-evidence", type=Path)
    args = parser.parse_args()
    receipt = audit(
        artifacts=args.artifacts,
        out=args.out,
        round_index=args.round_index,
        consumed_round74_evidence=args.consumed_round74_evidence,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
