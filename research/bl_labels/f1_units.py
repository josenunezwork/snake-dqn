"""Outcome-blind immutable completion ledgers for F1 worker schedules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .exports import _materialize_triplet, _verify, publish_json

LABELS = ("one", "half", "all")
SCHEMA = "bl-labels-f1-verified-ledger/v1"


def empty_ledger(
    bank_sha256: str,
    platform: str,
    root_ids: list[str] | None = None,
    source_commit: str = "local-functional",
) -> dict[str, Any]:
    if not re.fullmatch("[0-9a-f]{64}", bank_sha256) or platform not in {"cpu", "l4", "mac"}:
        raise ValueError("F1 ledger requires frozen bank digest and platform")
    return {
        "schema": SCHEMA,
        "namespace": f"f1/{bank_sha256}/{platform}/{source_commit}",
        "source_commit": source_commit,
        "bank_sha256": bank_sha256,
        "platform": platform,
        "allocation": list(LABELS),
        "units": [],
    }


def load_ledger(
    path: Path, bank_sha256: str, platform: str, source_commit: str = "local-functional"
) -> dict[str, dict[str, Any]]:
    _verify(path)
    value = json.loads(path.read_text())
    expected = empty_ledger(bank_sha256, platform, source_commit=source_commit)
    if {k: v for k, v in value.items() if k != "units"} != {
        k: v for k, v in expected.items() if k != "units"
    }:
        raise ValueError("foreign F1 ledger namespace/allocation")
    if not isinstance(value.get("units"), list):
        raise ValueError("malformed F1 ledger units")
    found = {}
    for row in value["units"]:
        label = row.get("unit_id")
        if label not in LABELS or label in found or row.get("path") != f"units/config-{label}.json":
            raise ValueError("duplicate or foreign F1 schedule unit")
        payload = path.parent / row["path"]
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("F1 ledger unit differs from verified metadata")
        unit = json.loads(payload.read_text())
        if (
            unit.get("source_commit") != source_commit
            or unit.get("bank_sha256") != bank_sha256
            or unit.get("platform") != platform
            or unit.get("schedule_label") != label
            or unit.get("root_count") != 64
            or unit.get("status") != "COMPLETE"
            or unit.get("transport_probe") is not False
        ):
            raise ValueError("F1 schedule payload has foreign identity or incomplete evidence")
        found[label] = unit
    return found


def build_ledger(
    attempts: list[Path],
    out: Path,
    bank_sha256: str,
    platform: str,
    source_commit: str = "local-functional",
) -> dict[str, Any]:
    """Union only paths/sizes/digests; inspect no timing or label outcomes."""
    value = empty_ledger(bank_sha256, platform, source_commit=source_commit)
    seen = set()
    for attempt in attempts:
        if list(attempt.glob("INVALID*.json")):
            raise ValueError("INVALID F1 attempt cannot contribute verified units")
        for path in sorted((attempt / "units").glob("*.json")):
            label = path.stem.removeprefix("config-")
            if label not in LABELS or label in seen:
                raise ValueError("duplicate or foreign F1 completed unit")
            # Open identity wrappers only; never use timing values for unit selection.
            digest = _verify(path)
            unit = json.loads(path.read_text())
            if (
                unit.get("source_commit") != source_commit
                or unit.get("bank_sha256") != bank_sha256
                or unit.get("platform") != platform
                or unit.get("schedule_label") != label
                or unit.get("status") != "COMPLETE"
                or unit.get("transport_probe") is not False
                or unit.get("root_count") != 64
            ):
                raise ValueError("foreign or incomplete F1 completed unit")
            target = out.parent / "units" / path.name
            _materialize_triplet(path, target)
            value["units"].append(
                {
                    "unit_id": label,
                    "path": f"units/{path.name}",
                    "sha256": digest,
                    "size": path.stat().st_size,
                }
            )
            seen.add(label)
    publish_json(out, value)
    return value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, action="append", required=True)
    parser.add_argument("--bank-sha256", required=True)
    parser.add_argument("--platform", choices=("cpu", "l4", "mac"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    build_ledger(args.exports, args.out, args.bank_sha256, args.platform, args.source_commit)


if __name__ == "__main__":
    main()
