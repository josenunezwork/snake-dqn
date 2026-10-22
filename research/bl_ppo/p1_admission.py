"""Local, evidence-bound discharge of P1 source review and shared Y3 debit."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, verify_export

RECEIPT_NAMES = {"source-02", "source-latebody-02"}
LIVE_BLOCKER = "Claude: fresh usage + sessions; confirm empty inventory and rate <=1.54u/h"
Y3_05_ENVELOPE = 25.50
Y3_05_RATE = 1.54


def reconcile_y3_05(receipts: list[Path]) -> dict[str, Any]:
    """Bind the four real predecessor receipts under the prospective Y3-05 envelope.

    This is deliberately a new schema: old eight-unit debit artifacts remain
    historical evidence and their parsers/tests retain their original rules.
    """
    expected = ("source-02", "source-latebody-02", "canary-03", "source-03")
    expected_locations = (
        "source-02/colab-output/receipt.json",
        "source-latebody-02/colab-output/receipt.json",
        "source-latebody-03-canary-01/canary-output/receipt.json",
        "source-latebody-03-canary-01/source-output/receipt.json",
    )
    snapshot_retry = len(receipts) == 6
    if snapshot_retry:
        expected += ("canary-05", "source-05")
        prefix = "source-latebody-05-kernel-watchdog-repin-20261022/family-canary/"
        expected_locations += (
            prefix + "canary-output/receipt.json",
            prefix + "source-output/receipt.json",
        )
    if len(receipts) != len(expected):
        raise ValueError("Y3-05 requires exactly the four known real receipts")
    spend = subprocess.run(
        ["python3", "tools/ops/spend.py", *map(str, receipts)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    if spend.returncode:
        raise ValueError(f"spend reconciliation refused: {spend.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for name, location, path in zip(expected, expected_locations, receipts):
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or unsafe Y3-05 predecessor receipt")
        if not str(path.resolve()).endswith(location):
            raise ValueError("Y3-05 predecessor receipt identity/order differs")
        value = json.loads(path.read_text())
        expected_status = "PASS" if name.startswith("canary-") else "INCOMPLETE"
        if (
            value.get("status") != expected_status
            or value.get("stop_verified") is not True
            or value.get("cleanup_error") is not None
        ):
            raise ValueError("predecessor receipt is not a released terminal attempt")
        accounting = value.get("accounting")
        elapsed = value.get("elapsed_seconds")
        if not isinstance(accounting, dict) or type(elapsed) not in (int, float) or elapsed <= 0:
            raise ValueError("invalid predecessor receipt accounting")
        rate = accounting.get("own_rate_units_per_hour")
        if type(rate) not in (int, float) or rate <= 0:
            raise ValueError("invalid predecessor own rate")
        own = elapsed * rate / 3600
        before = value.get("usage_before", {}).get("balance")
        after = value.get("usage_after", {}).get("balance")
        if type(before) not in (int, float) or type(after) not in (int, float):
            raise ValueError("predecessor receipt lacks account balances")
        gross = max(0.0, before - after)
        charged = math.ceil((max(own, gross) - 1e-9) * 100) / 100
        rows.append(
            {
                "attempt": name,
                "path": str(path.resolve()),
                "sha256": digest(path),
                "own_units": own,
                "gross_account_debit_units": gross,
                "legacy_conservative_charge_units": charged,
            }
        )
    spent = round(sum(row["legacy_conservative_charge_units"] for row in rows), 2)
    if round(sum(row["legacy_conservative_charge_units"] for row in rows[:4]), 2) != 4.67:
        raise ValueError("known Y3-05 predecessor debit differs from frozen 4.67u")
    result = {
        "schema": (
            "bl-p1-y3-debit/v3-snapshot-bound-retry"
            if snapshot_retry
            else "bl-p1-y3-debit/v2-source-latebody-05"
        ),
        "status": "RECONCILED_NOT_DISPATCHED",
        "envelope_units": Y3_05_ENVELOPE,
        "base_envelope_units": 12.75,
        "pricing_rate_units_per_hour": Y3_05_RATE,
        "attempts": rows,
        "legacy_conservative_spend_units": spent,
        "remaining_after_predecessors_units": round(Y3_05_ENVELOPE - spent, 2),
        "canary": {
            "base_cap_units": 1.0,
            "effective_cap_units": 2.0,
            "base_reserve_units": 0.2,
            "effective_reserve_units": 0.4,
            "forecast_minutes": 15,
            "timeout_minutes": 30,
            "base_timeout_minutes": 15,
        },
        "source": {
            "forecast_minutes": 75,
            "timeout_minutes": 300,
            "base_timeout_minutes": 150,
            "base_cap_units": 4.81,
            "effective_cap_units": 9.62,
            "base_reserve_units": 0.96,
            "effective_reserve_units": 1.92,
        },
        "replacement": {
            "forecast_minutes": 30,
            "timeout_minutes": 120,
            "base_timeout_minutes": 60,
            "base_cap_units": 1.93,
            "effective_cap_units": 3.86,
            "base_reserve_units": 0.39,
            "effective_reserve_units": 0.78,
        },
        "retry": {
            "max_outcome_blind_retries": 2,
            "attempts": "one original plus at most two completion attempts",
            "max_additional_source_attempts": 1,
            "source_cap_units": 9.62,
            "source_timeout_minutes": 300,
            "source_reserve_units": 1.92,
            "replacement_cap_units": 3.86,
            "replacement_timeout_minutes": 120,
            "replacement_reserve_units": 0.78,
            "preconditions": [
                "verified SUPERVISOR/transport failure",
                "verified stop with no cleanup error",
                "missing units",
                "no actor outcome read",
                "full invoice fits actual residual",
            ],
            "second_consecutive_post_fix_failure": "STOP_FOR_ROOT_CAUSE_REVIEW",
        },
        "spend_report": spend.stdout,
    }
    reserved = spent + 2.0 + 9.62 + 3.86
    if reserved > Y3_05_ENVELOPE + 1e-9:
        raise ValueError("Y3-05 envelope cannot fund canary/source/replacement")
    result["reserved_initial_path_units"] = reserved
    result["unallocated_after_initial_path_units"] = round(Y3_05_ENVELOPE - reserved, 2)
    return result


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
    if debit.get("schema") in {
        "bl-p1-y3-debit/v2-source-latebody-05",
        "bl-p1-y3-debit/v3-snapshot-bound-retry",
    }:
        actual = reconcile_y3_05([Path(row["path"]) for row in debit.get("attempts", [])])
        if debit != actual:
            raise ValueError("Y3 debit differs from immutable terminal receipts")
        return {
            "estimate_minutes": 75,
            "timeout_minutes": 300,
            "unit_cap": 9.62,
            "reserve_units": 1.92,
        }
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
