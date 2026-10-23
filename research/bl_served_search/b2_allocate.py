"""Prepare one fresh, ratified B2 seed allocation without dispatching any work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.bl_served_search import b2
from research.bl_served_search import b2_units as units
from research.bl_served_search import bank
from research.bl_served_search.ceiling import digest_value
from research.bl_served_search.exports import digest, export_json, verify_export

NAMESPACE = "bl-c15-b2-natural-roots-20261022/v1"
RULING_STATUS = "RATIFIED_BEFORE_DATA"
PRODUCTION_TRAJECTORIES = 32
MAX_ROUNDS = 32
ROSTER = "bare-b8-s3-vs-5-frp3-s12-v8"
_SCIENTIFIC_KEYS = (
    "trajectory_frames",
    "root_stride",
    "escape_total_frames",
    "unit_timeout_seconds",
)


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")


def derive(
    round_index: int, namespace: str = NAMESPACE
) -> tuple[list[int], int, list[dict[str, Any]]]:
    """Derive all production and excluded-canary seeds for a complete round."""
    if type(round_index) is not int or not 0 <= round_index < MAX_ROUNDS:
        raise ValueError("B2 round must be a non-boolean integer from 0 through 31")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("B2 namespace must be nonempty")
    prefix = f"{namespace}/round-{round_index}"
    seeds = [_seed(f"{prefix}/production/trajectory-{index:02d}") for index in range(32)]
    canary_seed = _seed(f"{prefix}/excluded-canary")
    rows = [
        {
            "world_id": f"{prefix}/production/trajectory-{index:02d}",
            "world_seed": seed,
            "allocation": "production",
            "trajectory_index": index,
        }
        for index, seed in enumerate(seeds)
    ]
    rows.append(
        {
            "world_id": f"{prefix}/excluded-canary",
            "world_seed": canary_seed,
            "allocation": "excluded-canary",
            "trajectory_index": None,
        }
    )
    if len({row["world_seed"] for row in rows}) != len(rows):
        raise ValueError("B2 derived an internal whole-round seed collision")
    return seeds, canary_seed, rows


def read_ruling(path: Path) -> tuple[dict[str, Any], str]:
    """Load only a complete, owner-ratified, immutable B2 parameter ruling."""
    path = Path(path)
    checksum = verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("status") != RULING_STATUS:
        raise ValueError("B2 requires a RATIFIED_BEFORE_DATA ruling")
    if value.get("plan_sha256") != b2.PLAN_SHA256:
        raise ValueError("B2 ruling does not bind the current PLAN_v3 authority")
    for key in _SCIENTIFIC_KEYS:
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(f"B2 ruling missing positive {key}")
    if value["escape_total_frames"] not in (1440, 1740):
        raise ValueError("B2 ruling must freeze a legal H1440 continuation horizon")
    if value.get("root_grid") != "absolute-trajectory-decision":
        raise ValueError("B2 ruling must freeze absolute-trajectory-decision root grid")
    return value, checksum


def _design(
    ruling: dict[str, Any],
    ruling_path: Path,
    ruling_sha256: str,
    namespace: str,
    round_index: int,
    seeds: list[int],
    canary_seed: int,
    freshness: dict[str, Any],
    freshness_sha256: str,
) -> dict[str, Any]:
    design: dict[str, Any] = {
        "schema": b2.SCHEMA + "/design",
        "plan_sha256": b2.PLAN_SHA256,
        "status": "FROZEN_BEFORE_DATA",
        "namespace": namespace,
        "round_index": round_index,
        "roster": ROSTER,
        "workers": 6,
        "seeds": seeds,
        "canary_seed": canary_seed,
        "freshness": freshness,
        "freshness_report_sha256": freshness_sha256,
        "ruling": {"path": ruling_path.name, "sha256": ruling_sha256},
        **{key: ruling[key] for key in _SCIENTIFIC_KEYS},
    }
    design["root_grid"] = ruling["root_grid"]
    design["allocation_digest"] = digest_value(design)
    b2.validate_spec(design)
    return design


def validate_audit(audit: dict[str, Any], seeds: list[int], canary_seed: int) -> None:
    """Require the actual whole-allocation bank report, never a CLEAR assertion."""
    if not isinstance(audit, dict) or audit.get("schema") != "bl-served-search-seed-audit/v1":
        raise ValueError("B2 requires the bank.audit schema")
    if (
        audit.get("status") != "CLEAR"
        or audit.get("candidate_count") != 33
        or audit.get("unique_count") != 33
        or audit.get("collisions") != []
        or audit.get("internal_collision") is not False
    ):
        raise ValueError("B2 whole-allocation freshness audit is not CLEAR")
    explicit = audit.get("explicit_banks")
    if (
        not isinstance(explicit, dict)
        or len(explicit) < 4
        or any(
            not isinstance(row, dict) or row.get("collisions") != [] for row in explicit.values()
        )
    ):
        raise ValueError("B2 freshness audit lacks clear explicit-bank coverage")
    history = audit.get("history")
    if (
        not isinstance(history, dict)
        or not isinstance(history.get("roots"), list)
        or not history["roots"]
        or not isinstance(history.get("hits"), dict)
        or history.get("never_opened") != bank.SEALED
        or not isinstance(audit.get("coverage"), str)
        or not audit["coverage"]
    ):
        raise ValueError("B2 freshness audit lacks history coverage proof")
    exclusions = audit.get("training_exclusions")
    if (
        not isinstance(exclusions, list)
        or len(exclusions) < 7
        or any(not isinstance(row, dict) or row.get("collisions") != [] for row in exclusions)
    ):
        raise ValueError("B2 freshness audit lacks clear training-exclusion coverage")


def allocate(
    ruling_path: Path,
    out: Path,
    *,
    namespace: str = NAMESPACE,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    """Audit complete rounds until one is clear, then export B2 design and ledger."""
    if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_ROUNDS:
        raise ValueError("B2 max_rounds must be a non-boolean integer from 1 through 32")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("B2 namespace must be nonempty")
    # Refuse before audit side effects when the authority is incomplete or foreign.
    ruling, ruling_sha256 = read_ruling(ruling_path)
    out = Path(out)
    if out.exists():
        raise FileExistsError("B2 allocation output must be a fresh directory")
    out.mkdir(parents=True)
    for round_index in range(max_rounds):
        seeds, canary_seed, rows = derive(round_index, namespace)
        audit = bank.audit(rows)
        if not isinstance(audit, dict) or audit.get("status") not in {"CLEAR", "REJECTED"}:
            raise ValueError("B2 freshness auditor returned an invalid whole-round report")
        report = {
            **audit,
            "namespace": namespace,
            "round_index": round_index,
            "seed_allocation_sha256": digest_value([*seeds, canary_seed]),
            "candidate_count": len(rows),
        }
        export_json(out / f"audit-round-{round_index:02d}.json", report)
        if report["status"] != "CLEAR":
            continue
        validate_audit(audit, seeds, canary_seed)
        freshness = {
            **audit,
            "namespace": namespace,
            "round_index": round_index,
            "seeds": [*seeds, canary_seed],
            "seed_allocation_sha256": digest_value([*seeds, canary_seed]),
        }
        freshness_path = out / "freshness-audit.json"
        export_json(freshness_path, freshness)
        design = _design(
            ruling,
            Path(ruling_path),
            ruling_sha256,
            namespace,
            round_index,
            seeds,
            canary_seed,
            freshness,
            digest(freshness_path),
        )
        b2.validate_freshness(design, freshness_path)
        export_json(out / "b2-design.json", design)
        units.export_ledger(
            out / "verified-ledger.json", units.empty_ledger(namespace, design["allocation_digest"])
        )
        return design
    raise RuntimeError("B2 has no CLEAR whole allocation; rejected round receipts are retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruling", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = parser.parse_args()
    design = allocate(args.ruling, args.out, namespace=args.namespace, max_rounds=args.max_rounds)
    print(json.dumps({"status": design["status"], "round_index": design["round_index"]}))


if __name__ == "__main__":
    main()
