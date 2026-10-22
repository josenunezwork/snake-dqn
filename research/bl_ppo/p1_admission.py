"""Local, evidence-bound discharge of P1 source review and shared Y3 debit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, verify_export

RECEIPT_NAMES = {"source-02", "source-latebody-02"}
LIVE_BLOCKER = "Claude: fresh usage + sessions; confirm empty inventory and rate <=1.54u/h"


def reconcile(receipts: list[Path]) -> dict[str, Any]:
    """Charge full elapsed ceiling or gross account debit, whichever is larger."""
    import json

    rows = []
    roots = {p.resolve().parents[2] for p in receipts}
    supplied = {p.resolve() for p in receipts}
    discovered = {p.resolve() for root in roots for p in root.glob("*/colab-output/receipt.json")}
    if discovered != supplied:
        raise ValueError("Y3 terminal receipt inventory changed; reprice every spent attempt")
    for path in receipts:
        # Supervisor write_once terminal receipts have no export sidecars.
        # Preserve those bytes; the new debit receipt binds their SHA-256.
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or unsafe supervisor terminal receipt")
        if any(Path(str(path) + suffix).exists() for suffix in (".sha256", ".complete")):
            verify_export(path)
        value = json.loads(path.read_text())
        name = path.parent.parent.name
        if name not in RECEIPT_NAMES or value.get("stop_verified") is not True:
            raise ValueError("Y3 reconciliation requires both known, released attempts")
        if value.get("cleanup_error") or value.get("status") != "INCOMPLETE":
            raise ValueError("unexpected Y3 predecessor terminal state")
        accounting = value["accounting"]
        elapsed = value["elapsed_seconds"]
        rate = accounting["own_rate_units_per_hour"]
        observed = accounting["observed_rate_units_per_hour"]
        before, after = value["usage_before"]["balance"], value["usage_after"]["balance"]
        numbers = (elapsed, rate, observed, before, after)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in numbers):
            raise ValueError("invalid Y3 debit inputs")
        if elapsed <= 0 or rate <= 0 or observed != 1.54:
            raise ValueError("Y3 L4 rate evidence differs from the repriced ceiling")
        gross = max(0.0, before - after)
        # Do not credit concurrent baseline debit or provider lag. Round UP to cents.
        charge = math.ceil((max(elapsed * rate / 3600, gross) - 1e-9) * 100) / 100
        rows.append(
            {
                "attempt": name,
                "path": str(path.resolve()),
                "sha256": digest(path),
                "endpoint": value["endpoint"],
                "stop_verified": True,
                "elapsed_seconds": elapsed,
                "priced_rate_units_per_hour": rate,
                "observed_rate_units_per_hour": observed,
                "gross_account_debit_units": gross,
                "last_crosscheck": accounting.get("last_crosscheck"),
                "charged_units": charge,
            }
        )
    if len(rows) != 2 or {r["attempt"] for r in rows} != RECEIPT_NAMES:
        raise ValueError("Y3 reconciliation requires each spent attempt exactly once")
    spent = round(sum(r["charged_units"] for r in rows), 2)
    remaining = round(8.0 - spent, 2)
    # 150min source and 40min replacement at the measured ceiling, plus 20%
    # of each cap retained as reserve. These ceilings sum to the actual remainder.
    source_cap, replacement_cap = 4.81, 1.29
    if remaining != round(source_cap + replacement_cap, 2):
        raise ValueError("changed Y3 debit requires a fresh price, never a larger allowance")
    return {
        "schema": "bl-p1-y3-debit/v1",
        "status": "RECONCILED",
        "envelope_units": 8.0,
        "attempts": sorted(rows, key=lambda r: r["attempt"]),
        "spent_units": spent,
        "remaining_units": remaining,
        "pricing_rate_units_per_hour": 1.54,
        "source": {
            "estimate_minutes": 75,
            "timeout_minutes": 150,
            "unit_cap": source_cap,
            "reserve_units": 0.96,
        },
        "replacement": {
            "estimate_minutes": 20,
            "timeout_minutes": 40,
            "unit_cap": replacement_cap,
            "reserve_units": 0.26,
        },
        "policy": "Full elapsed ceiling or gross account debit, maximum, rounded up to cents; "
        "no subtraction for concurrent assignments; first cap wins. "
        "Replacement is a reservation, not dispatch admission or a new allowance.",
    }


def verify_source_admission(
    review: dict[str, Any], debit: dict[str, Any], identity: dict[str, str]
) -> dict[str, Any]:
    """Verify actual review hashes and recompute the complete shared debit locally."""
    if review.get("schema") != "bl-p1-codex-v-review/v1" or review.get("status") != "PASS":
        raise ValueError("source requires independent Codex-V PASS review")
    if review.get("source_sha256") != identity or review.get("unresolved_findings") != []:
        raise ValueError("Codex-V review does not cover this exact executable source")
    if review.get("no_new_cuda_path") is not True:
        raise ValueError("round-14 requires no-new-CUDA-path review")
    actual = reconcile([Path(row["path"]) for row in debit.get("attempts", [])])
    if debit != actual:
        raise ValueError("Y3 debit differs from immutable terminal receipts")
    for name in ("source", "replacement"):
        item = debit[name]
        if item["timeout_minutes"] < 2 * item["estimate_minutes"]:
            raise ValueError("Y3 timeout lacks 2x estimate")
        if item["timeout_minutes"] * 1.54 / 60 + item["reserve_units"] > item["unit_cap"] + 1e-9:
            raise ValueError("Y3 liability plus reserve exceeds its cap")
    return debit["source"]
