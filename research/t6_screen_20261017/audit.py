#!/usr/bin/env python3
"""Refresh the fail-closed T6 probe-world audit for rounds 72 and 73.

The original T6 preparation audit binds the historical evaluation, training,
label, return, and snapshot-store namespaces.  This launch-time wrapper adds
the subsequently reserved T7 Phase R namespace, which is live while T6 is
probed.  It allocates no worlds and starts no gameplay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.apex_safety_20260926 import dev_screen  # noqa: E402
from research.redesign_t7_phase_r_20261016 import seed_audit as t7_seed_audit  # noqa: E402
from research.t6_restart_20261016 import seed_audit as t6_seed_audit  # noqa: E402

SCHEMA = "t6-screen-freshness-audit/v1"
T7_RECEIPT = Path("research/redesign_t7_phase_r_20261016/SEED_AUDIT_T7.json")
LEGACY_T6_RECEIPT = Path("docs/research/t6_seed_audit_2026-10-16.json")


def _sha256(path: Path) -> str:
    """Return the SHA256 of one evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _t7_namespace(repo: Path) -> tuple[dict[str, Any], set[int]]:
    """Bind T7's deterministic candidates and its committed audit receipt."""
    generated = t7_seed_audit.generate_banks()
    banks = generated.get("banks")
    if not isinstance(banks, dict) or set(banks) != {f"s{index}" for index in range(5)}:
        raise ValueError("T7 Phase R generated-bank roster is incomplete")
    seeds: set[int] = set()
    for name, bank in sorted(banks.items()):
        values = bank.get("seeds") if isinstance(bank, dict) else None
        if not isinstance(values, list) or len(values) != t7_seed_audit.SEED_COUNT:
            raise ValueError(f"T7 Phase R bank {name} is malformed")
        if not all(type(seed) is int and seed >= 0 for seed in values):
            raise ValueError(f"T7 Phase R bank {name} contains invalid seed identities")
        seeds.update(values)
    if len(seeds) != 5 * t7_seed_audit.SEED_COUNT:
        raise ValueError("T7 Phase R generated-bank roster contains duplicate identities")

    receipt_path = repo / T7_RECEIPT
    receipt: dict[str, Any] = {"path": str(receipt_path), "exists": receipt_path.is_file()}
    if receipt_path.is_file():
        try:
            payload = json.loads(receipt_path.read_text())
        except json.JSONDecodeError as error:
            receipt.update({"valid": False, "error": str(error)})
        else:
            recorded_banks = payload.get("banks") if isinstance(payload, dict) else None
            recorded_seeds: set[int] = set()
            roster_valid = isinstance(recorded_banks, dict) and set(recorded_banks) == set(banks)
            if roster_valid:
                for name in banks:
                    row = recorded_banks.get(name)
                    listed = row.get("seeds") if isinstance(row, dict) else None
                    if not isinstance(listed, list) or not all(
                        type(seed) is int for seed in listed
                    ):
                        roster_valid = False
                        break
                    recorded_seeds.update(listed)
            roster_matches = roster_valid and recorded_seeds == seeds
            receipt.update(
                {
                    "valid": (
                        isinstance(payload, dict)
                        and payload.get("schema") == t7_seed_audit.SCHEMA
                        and payload.get("namespace") == t7_seed_audit.BANK_NS
                        and payload.get("pass") is True
                        and roster_matches
                    ),
                    "sha256": _sha256(receipt_path),
                    "schema": payload.get("schema") if isinstance(payload, dict) else None,
                    "pass": payload.get("pass") if isinstance(payload, dict) else None,
                    "roster_matches_generated": roster_matches,
                }
            )
    else:
        receipt["valid"] = False
    result = {
        "namespace": t7_seed_audit.BANK_NS,
        "bank_key": t7_seed_audit.BANK_KEY,
        "bank_count": len(banks),
        "world_count": len(seeds),
        "seed_digest": t7_seed_audit.seed_digest(seeds),
        "receipt": receipt,
    }
    result["pass"] = receipt.get("valid") is True
    return result, seeds


def _additional_historical_namespaces(artifacts: Path) -> tuple[dict[str, Any], set[int]]:
    """Bind later FRP/T4f/T7 exclusion evidence omitted by the old T6 receipt."""
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise FileNotFoundError("historical FRP training exclusion inventories incomplete")
    values: set[int] = set()
    inventories: list[dict[str, Any]] = []
    for path in exclusions:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list) or not all(type(seed) is int for seed in payload):
            raise ValueError(f"invalid FRP training exclusions: {path}")
        seeds = set(payload)
        values.update(seeds)
        inventories.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "count": len(seeds),
                "seed_digest": t7_seed_audit.seed_digest(seeds),
            }
        )
    t4f_manifests = [
        artifacts / "t4f-return-bank-v3-20261016/banks/train/manifest.json",
        artifacts / "t4f-return-bank-v3-20261016/banks/holdout/manifest.json",
    ]
    if not all(path.is_file() for path in t4f_manifests):
        raise FileNotFoundError("T4f v3 train and holdout manifests are required")
    for path in t4f_manifests:
        payload = json.loads(path.read_text())
        if payload.get("collection_schema") != "t4f-return-bank/v3":
            raise ValueError(f"unexpected T4f v3 manifest schema: {path}")

    # These were generated/rejected before the live round-76 namespace.  They
    # remain excluded identities even though their audit receipt is documentary.
    rejected: set[int] = set()
    for round_index in (74, 75):
        for run_seed in range(5):
            rejected.update(
                dev_screen.screen_seeds(
                    t7_seed_audit.SEED_COUNT,
                    t7_seed_audit.BANK_NS,
                    f"round{round_index}-seed{run_seed}",
                )
            )
    if len(rejected) != 2 * 5 * t7_seed_audit.SEED_COUNT:
        raise ValueError("T7 rejected-round namespace contains duplicate identities")
    values.update(rejected)
    return (
        {
            "frp_training_exclusion_inventories": inventories,
            "t4f_v3_manifests": [
                {"path": str(path), "sha256": _sha256(path)} for path in t4f_manifests
            ],
            "t7_rejected_rounds": [74, 75],
            "t7_rejected_world_count": len(rejected),
            "t7_rejected_seed_digest": t7_seed_audit.seed_digest(rejected),
            "pass": True,
        },
        values,
    )


def _validated_documentary(
    repo: Path, round_index: int, consumed_round72_evidence: str | Path | None
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Validate only owned receipts that may contain otherwise-fresh seed literals."""
    from research.redesign_m2_20261008 import ni_spec

    legacy = repo / LEGACY_T6_RECEIPT
    if not legacy.is_file():
        raise FileNotFoundError("binding legacy T6 seed-audit receipt is missing")
    payload = json.loads(legacy.read_text())
    expected = {
        str(value): sorted(ni_spec.distill_seeds(96, round_index=value))
        for value in t6_seed_audit.ROUNDS
    }
    valid_legacy = (
        isinstance(payload, dict)
        and payload.get("schema") == "t6-restart-seed-bank-audit/v1"
        and payload.get("pass") is True
        and isinstance(payload.get("checks"), dict)
        and all(
            payload["checks"].get(key, {}).get("seeds") == seeds for key, seeds in expected.items()
        )
    )
    if not valid_legacy:
        raise ValueError("legacy T6 receipt does not bind the expected fresh rosters")
    paths = [legacy]
    rows: list[dict[str, Any]] = [
        {"path": str(legacy), "sha256": _sha256(legacy), "kind": "validated-legacy-t6-audit"}
    ]
    if round_index == t6_seed_audit.EXTENSION_ROUND:
        evidence = Path(consumed_round72_evidence).resolve()  # type: ignore[arg-type]
        if not evidence.is_file():
            raise FileNotFoundError("completed round-72 probe transcript is missing")
        records = [json.loads(line) for line in evidence.read_text().splitlines() if line.strip()]
        if not records or not all(
            isinstance(row, dict) and row.get("round_index") == 72 for row in records
        ):
            raise ValueError("completed round-72 transcript has invalid rows")
        expected72 = set(expected["72"])
        observed: set[int] = set()
        for row in records:
            for key in ("mass", "deaths"):
                values = row.get(key)
                if not isinstance(values, dict):
                    raise ValueError(f"completed round-72 transcript lacks {key} map")
                try:
                    mapped = {int(seed) for seed in values}
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "completed round-72 transcript has invalid seed keys"
                    ) from error
                if not mapped or not mapped <= expected72:
                    raise ValueError("completed round-72 transcript has unregistered seed keys")
                observed.update(mapped)
        if observed != expected72:
            raise ValueError("completed round-72 transcript does not bind the full roster")
        paths.append(evidence)
        rows.append(
            {
                "path": str(evidence),
                "sha256": _sha256(evidence),
                "kind": "validated-owned-round72-probe-transcript",
                "rows": len(records),
                "world_count": len(observed),
            }
        )
        initial_audit = evidence.parent / "seed_bank_audit_round72.json"
        if not initial_audit.is_file():
            raise FileNotFoundError("completed round-72 audit receipt is missing")
        initial = json.loads(initial_audit.read_text())
        if not (
            isinstance(initial, dict)
            and initial.get("schema") == SCHEMA
            and initial.get("round_index") == 72
            and initial.get("pass") is True
            and initial.get("checks", {}).get("72", {}).get("seeds") == expected["72"]
        ):
            raise ValueError("completed round-72 audit receipt does not bind the expected roster")
        paths.append(initial_audit)
        rows.append(
            {
                "path": str(initial_audit),
                "sha256": _sha256(initial_audit),
                "kind": "validated-owned-round72-audit-receipt",
            }
        )
    return tuple(paths), rows


def _base_audit_with_documentary_exclusions(
    exclusions: tuple[Path, ...], active_round: int, **kwargs: Any
) -> dict[str, Any]:
    """Run the legacy audit while excluding only validated exact documentary files."""
    original_tokens = t6_seed_audit.concrete_text_hits
    original_metadata = t6_seed_audit._reserved_round_metadata

    def tokens(root: Path, values: list[str], _ignored: tuple[Path, ...] = ()) -> Any:
        if active_round == t6_seed_audit.EXTENSION_ROUND:
            from research.redesign_m2_20261008 import ni_spec

            values = [
                str(seed)
                for seed in sorted(ni_spec.distill_seeds(96, round_index=active_round))
            ]
        return original_tokens(root, values, tuple(dict.fromkeys((*_ignored, *exclusions))))

    def metadata(root: Path, _ignored: tuple[Path, ...] = ()) -> Any:
        if active_round == t6_seed_audit.EXTENSION_ROUND:
            hits, progress = t6_seed_audit._partitioned_t6_rg(
                root,
                [
                    "--pcre2",
                    r'"(?:round_index|round)"\s*:\s*73(?![0-9])',
                    "--glob",
                    "*.json",
                    "--glob",
                    "*.jsonl",
                    "--only-matching",
                ],
                tuple(dict.fromkeys((*_ignored, *exclusions))),
                "round73-metadata",
            )
            return bool(hits), progress
        return original_metadata(root, tuple(dict.fromkeys((*_ignored, *exclusions))))

    t6_seed_audit.concrete_text_hits = tokens
    t6_seed_audit._reserved_round_metadata = metadata
    try:
        return t6_seed_audit.seed_bank_audit(**kwargs)
    finally:
        t6_seed_audit.concrete_text_hits = original_tokens
        t6_seed_audit._reserved_round_metadata = original_metadata


def audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    out: str | Path | None = None,
    snapshot_store: str | Path | None = None,
    t4e_source_bank: str | Path | None = None,
    round_index: int = t6_seed_audit.INITIAL_ROUND,
    consumed_round72_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """Return the current T6 rounds-72/73 freshness receipt.

    ``out`` may name only the exact receipt currently being written.  It is
    passed through to the base audit as its sole scan exclusion, so rerunning
    this function refreshes the scan without hiding previous evidence.
    """
    if round_index not in t6_seed_audit.ROUNDS:
        raise ValueError(f"round_index must be one of {t6_seed_audit.ROUNDS}")
    if round_index == t6_seed_audit.INITIAL_ROUND and consumed_round72_evidence is not None:
        raise ValueError("round 72 cannot exclude prior-round evidence")
    if round_index == t6_seed_audit.EXTENSION_ROUND and consumed_round72_evidence is None:
        raise ValueError("round 73 requires exact completed round-72 evidence")
    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    output = Path(out).resolve() if out is not None else None
    documentary_paths, documentary = _validated_documentary(
        repo_path, round_index, consumed_round72_evidence
    )
    base = _base_audit_with_documentary_exclusions(
        documentary_paths,
        active_round=round_index,
        repo=repo_path,
        artifacts=artifacts,
        snapshot_store=snapshot_store,
        t4e_source_bank=t4e_source_bank,
        exclude_out=output,
    )
    t7, t7_seeds = _t7_namespace(repo_path)
    historical, historical_seeds = _additional_historical_namespaces(artifact_root)
    checks: dict[str, dict[str, Any]] = {}
    for audit_round, original in base["checks"].items():
        row = dict(original)
        candidate = set(row["seeds"])
        row["t7_phase_r_namespace_collisions"] = sorted(candidate & t7_seeds)
        row["additional_historical_namespace_collisions"] = sorted(candidate & historical_seeds)
        checks[audit_round] = row
    passed = bool(
        base.get("pass") is True
        and t7.get("pass") is True
        and historical.get("pass") is True
        and all(
            not check["t7_phase_r_namespace_collisions"]
            and not check["additional_historical_namespace_collisions"]
            for check in checks.values()
        )
    )
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": _sha256(Path(__file__).resolve()),
        "scope": (
            f"T6 evaluation round {round_index} with both reserved rosters checked; "
            "base registered and observed history; "
            "T4e collection worlds; all 319 bound T4b snapshot source worlds; and the "
            "running T7 Phase R namespace."
        ),
        "excluded_output": str(output) if output is not None else None,
        "round_index": round_index,
        "consumed_round72_evidence": (
            str(Path(consumed_round72_evidence).resolve())
            if consumed_round72_evidence is not None
            else None
        ),
        "exclusion_policy": "Only the exact audit receipt named by out is self-excluded.",
        "validated_documentary_exclusions": documentary,
        "base_t6_audit": base,
        "t7_phase_r_namespace": t7,
        "additional_historical_namespaces": historical,
        "checks": checks,
        "limit": "No claim is made about unrecorded experiments or external systems.",
    }


def main() -> int:
    """Write a receipt and return a nonzero status on any freshness failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--snapshot-store", type=Path)
    parser.add_argument("--t4e-source-bank", type=Path)
    parser.add_argument("--round-index", type=int, default=t6_seed_audit.INITIAL_ROUND)
    parser.add_argument("--consumed-round72-evidence", type=Path)
    args = parser.parse_args()
    receipt = audit(
        artifacts=args.artifacts,
        out=args.out,
        snapshot_store=args.snapshot_store,
        t4e_source_bank=args.t4e_source_bank,
        round_index=args.round_index,
        consumed_round72_evidence=args.consumed_round72_evidence,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
