"""Admit the inventory-bound sibling only after genuine family and debit proof."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_family_canary import admit, preflight


def reconcile(history: Path, receipt: Path | None = None) -> dict[str, Any]:
    """Charge every terminal Y3 attempt, including the later staging pair."""
    paths: set[Path] = set()
    for attempt in history.iterdir():
        if attempt.name == "t34-round0-study-01" or attempt.is_symlink() or not attempt.is_dir():
            continue
        for suffix in (
            "colab-output/receipt.json",
            "canary-output/receipt.json",
            "source-output/receipt.json",
            "family-canary/canary-output/receipt.json",
            "family-canary/source-output/receipt.json",
            "final/canary-output/receipt.json",
            "final/source-output/receipt.json",
        ):
            path = attempt / suffix
            if path.exists():
                paths.add(path)
    if receipt is not None:
        paths.add(receipt)
    rows = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe or missing terminal receipt")
        value = json.loads(path.read_text())
        if value.get("stop_verified") is not True or value.get("cleanup_error") is not None:
            raise ValueError("attempt has not proved release")
        if value.get("status") not in {"PASS", "INCOMPLETE"}:
            raise ValueError("unexpected terminal state; return for review")
        elapsed = value.get("elapsed_seconds")
        rate = value.get("accounting", {}).get("own_rate_units_per_hour")
        before = value.get("usage_before", {}).get("balance")
        after = value.get("usage_after", {}).get("balance")
        if (
            any(
                type(number) not in (int, float) or not math.isfinite(number) or number < 0
                for number in (elapsed, rate, before, after)
            )
            or elapsed <= 0
            or rate <= 0
        ):
            raise ValueError("terminal receipt lacks finite accounting")
        charged = math.ceil((max(elapsed * rate / 3600, before - after, 0) - 1e-9) * 100) / 100
        rows.append({"path": str(path), "sha256": digest(path), "charged_units": charged})
    if not rows:
        raise ValueError("empty Y3 terminal history")
    total = round(sum(row["charged_units"] for row in rows), 2)
    reserved = round(total + 9.62 + 3.86 + (2.0 if receipt is None else 0), 2)
    if reserved > 25.50:
        raise ValueError("all-attempt debit plus remaining reservations exceeds 25.50u")
    return {
        "schema": "bl-p1-e8-04-all-attempt-debit/v1",
        "attempts": rows,
        "charged_units": total,
        "reserved_total_units": reserved,
        "envelope_units": 25.50,
        "warning": "WARNING_UNEXPLAINED_DEBIT",
        "live_usage_sessions_rate_calendar_checks": "Claude before each dispatch",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.receipt.resolve() != root / "canary-output/receipt.json":
        raise ValueError("requires this sibling's actual canary terminal receipt")
    if not args.receipt.is_file():
        raise ValueError("canary has not completed; source remains unadmitted")
    template = preflight(root, args.acceptance, args.review)
    if template.get("scientific_admission") is not False:
        raise ValueError("source template must retain its dispatch fuse")
    debit = reconcile(root.parent.parent, args.receipt.resolve())
    # Existing admission proves exact-source family exports, acceptance and release.
    admitted = admit(root, args.receipt, args.acceptance, args.review)
    value = json.loads(admitted.read_text())
    if value["bl_p1"]["control"].get("scientific_admission") is not True:
        raise ValueError("source runtime control is not provisioned for admission")
    value["scientific_admission"] = True
    value["transport"]["dispatch_ready"] = True
    export_json(root / "source-dispatch-debit.json", debit)
    export_json(root / "source-dispatch.json", value)
    print(root / "source-dispatch.json")


if __name__ == "__main__":
    main()
