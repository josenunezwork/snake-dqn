"""Create-only cumulative P1 attempt-2 accounting and admission evidence.

This companion intentionally does not call :func:`p1_qualification.terminal`
for the -04 failed repair: that success-only gate must remain unchanged.  It
instead binds the immutable failed receipt and its verified release proof into
a separate failure-accounting record, then uses that record for every -05
admission calculation.  This module never contacts a provider or dispatches.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.p1_repair_admission import RULING_SHA256, conservative_charge

ATTEMPT2_RULING_SHA256 = "55b13197d9d92f8eeb1c895180f01993ca11908f51c74a60d9a67065a0550d13"
ENVELOPE_UNITS = 34.50
PRIOR_CHARGED_UNITS = 10.34
FAILED_REPAIR_CHARGE_UNITS = 0.17
CANARY_LIABILITY_UNITS = 2.00
REPAIR_LIABILITY_UNITS = 6.16
REPLACEMENT_LIABILITY_UNITS = 3.86
REPLACEMENT_RETRY_LIABILITY_UNITS = 3.86
FAILED_RECEIPT_SHA256 = "b675b84e98d155251f43d1b44d00640240dbf2769cde2f37465edf6dc08b66b1"
FAILURE_REPORT_SHA256 = "b40f8abb4d9a6b942da17680778811dd3ecd1e3eb7a9a5a482e7c529c36392d5"


def _finite_number(value: object, name: str, *, nonnegative: bool = True) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _ceil_cents(value: float) -> float:
    """Round a debit upward to the nearest cent; never undercharge by rounding."""
    return math.ceil((value - 1e-9) * 100) / 100


def _export(path: Path) -> dict[str, Any]:
    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected immutable object evidence")
    return value


def _raw_receipt(path: Path) -> dict[str, Any]:
    """Read an unchanged supervisor receipt without pretending it is an export."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or unsafe supervisor receipt")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected supervisor receipt object")
    return value


def _release_proved(receipt: dict[str, Any]) -> None:
    if receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None:
        raise ValueError("failed receipt lacks verified release")


def _failure_report_binds_receipt(report: dict[str, Any], receipt: Path) -> None:
    if report.get("schema") != "bl-p1-repair-failure-independent-codex-v/v1":
        raise ValueError("unexpected independent repair-failure report")
    if report.get("status") != "VERIFIED_NO_RETRY_UNDER_CURRENT_RULES":
        raise ValueError("failure report is not the immutable no-retry finding")
    receipt_sha = digest(receipt)
    if not any(
        item.get("path") == str(receipt.resolve()) and item.get("sha256") == receipt_sha
        for item in report.get("evidence", [])
        if isinstance(item, dict)
    ):
        raise ValueError("failure report does not bind the supplied failed receipt")


def record_failed_repair(
    *,
    failure_report: Path,
    failed_receipt: Path,
    output: Path | None,
) -> Path:
    """Publish the ruled -04 charge without relaxing the success-only terminal gate."""
    report = _export(failure_report)
    receipt = _raw_receipt(failed_receipt)
    if digest(failure_report) != FAILURE_REPORT_SHA256:
        raise ValueError("failure report does not match its ruling-pinned bytes")
    if digest(failed_receipt) != FAILED_RECEIPT_SHA256:
        raise ValueError("failed receipt does not match its ruling-pinned bytes")
    _failure_report_binds_receipt(report, failed_receipt)
    _release_proved(receipt)
    if receipt.get("status") != "INCOMPLETE" or receipt.get("terminal") is not None:
        raise ValueError("failure accounting requires the unchanged incomplete nonterminal receipt")
    accounting = report.get("accounting", {})
    failed = accounting.get("failed_repair", {}) if isinstance(accounting, dict) else {}
    own = _finite_number(failed.get("own_units"), "failed own debit")
    gross = _finite_number(failed.get("gross_units"), "failed gross debit")
    ruled = _finite_number(failed.get("conservative_rounded_units"), "failed rounded debit")
    own_raw = (
        _finite_number(
            receipt.get("elapsed_seconds"), "failed receipt elapsed seconds", nonnegative=False
        )
        * _finite_number(
            receipt.get("accounting", {}).get("own_rate_units_per_hour"),
            "failed receipt own rate",
            nonnegative=False,
        )
        / 3600
    )
    before = _finite_number(receipt.get("usage_before", {}).get("balance"), "failed balance before")
    after = _finite_number(receipt.get("usage_after", {}).get("balance"), "failed balance after")
    gross_raw = max(before - after, 0.0)
    if (
        ruled != FAILED_REPAIR_CHARGE_UNITS
        or ruled != _ceil_cents(max(own, gross))
        or not math.isclose(own, own_raw, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(gross, gross_raw, rel_tol=0.0, abs_tol=1e-12)
        or ruled != _ceil_cents(max(own_raw, gross_raw))
    ):
        raise ValueError("failed repair debit does not equal the ruled conservative 0.17 u")
    if accounting.get("legacy_units") != 10.11:
        raise ValueError("failure report does not preserve the 10.11 u legacy debit")
    value = {
        "schema": "bl-p1-attempt2-failure-accounting/v1",
        "status": "PASS_FAILURE_ACCOUNTING_ONLY",
        "ruling_sha256": ATTEMPT2_RULING_SHA256,
        "finalisation_ruling_sha256": RULING_SHA256,
        "failure_report": {"path": str(failure_report.resolve()), "sha256": digest(failure_report)},
        "failed_receipt": {"path": str(failed_receipt.resolve()), "sha256": digest(failed_receipt)},
        "ruling_pins": {
            "failure_report_sha256": FAILURE_REPORT_SHA256,
            "failed_receipt_sha256": FAILED_RECEIPT_SHA256,
        },
        "release_proof": {"stop_verified": True, "cleanup_error": None},
        "charges": {
            "prior_charged_units": PRIOR_CHARGED_UNITS,
            "failed_repair_conservative_units": ruled,
            "charged_units": PRIOR_CHARGED_UNITS + ruled,
        },
        "warning": "WARNING_UNEXPLAINED_DEBIT",
        "note": (
            "Bound local receipt arithmetic only; not a live account inventory "
            "or dispatch approval."
        ),
    }
    if output is None:
        return Path()
    export_json(output, value)
    return output


def _failure_accounting(path: Path) -> dict[str, Any]:
    value = _export(path)
    charges = value.get("charges", {})
    if (
        value.get("schema") != "bl-p1-attempt2-failure-accounting/v1"
        or value.get("status") != "PASS_FAILURE_ACCOUNTING_ONLY"
        or value.get("ruling_sha256") != ATTEMPT2_RULING_SHA256
        or value.get("finalisation_ruling_sha256") != RULING_SHA256
        or value.get("warning") != "WARNING_UNEXPLAINED_DEBIT"
        or charges.get("prior_charged_units") != PRIOR_CHARGED_UNITS
        or charges.get("failed_repair_conservative_units") != FAILED_REPAIR_CHARGE_UNITS
        or charges.get("charged_units") != PRIOR_CHARGED_UNITS + FAILED_REPAIR_CHARGE_UNITS
        or value.get("ruling_pins", {}).get("failure_report_sha256") != FAILURE_REPORT_SHA256
        or value.get("ruling_pins", {}).get("failed_receipt_sha256") != FAILED_RECEIPT_SHA256
    ):
        raise ValueError("invalid attempt-2 failure accounting evidence")
    failed_receipt = Path(value.get("failed_receipt", {}).get("path", ""))
    failure_report = Path(value.get("failure_report", {}).get("path", ""))
    if digest(failed_receipt) != value["failed_receipt"].get("sha256") or digest(
        failure_report
    ) != value["failure_report"].get("sha256"):
        raise ValueError("failure accounting source binding changed")
    # Reopen the source evidence on every admission instead of trusting an
    # earlier PASS label; these defaults enforce the ruling pins.
    record_failed_repair(
        failure_report=failure_report,
        failed_receipt=failed_receipt,
        output=None,
    )
    return value


def _reconciliation(path: Path, *, additional_attributable_debit_units: float) -> dict[str, Any]:
    value = _export(path)
    declared = _finite_number(
        value.get("additional_attributable_debit_units"), "reconciliation additional debit"
    )
    if (
        value.get("schema") != "bl-p1-attempt2-live-reconciliation/v1"
        or value.get("status") != "PASS"
        or value.get("warning") != "WARNING_UNEXPLAINED_DEBIT"
        or declared != additional_attributable_debit_units
    ):
        raise ValueError("fresh reconciliation is absent or differs from the supplied debit")
    return value


def cumulative_admission(
    *,
    stage: str,
    held_manifest: Path,
    failure_accounting: Path,
    reconciliation: Path,
    additional_attributable_debit_units: float,
    output: Path,
    canary_receipt: Path | None = None,
) -> Path:
    """Publish exact-manifest cumulative charge-plus-liability admission evidence."""
    if stage not in {"canary", "repair"}:
        raise ValueError("attempt-2 admission accepts canary or repair only")
    manifest = _export(held_manifest)
    failure = _failure_accounting(failure_accounting)
    additional = _ceil_cents(
        _finite_number(additional_attributable_debit_units, "additional attributable debit")
    )
    _reconciliation(reconciliation, additional_attributable_debit_units=additional)
    if stage == "canary":
        if canary_receipt is not None:
            raise ValueError("canary admission cannot consume a canary receipt")
        liabilities = {
            "canary": CANARY_LIABILITY_UNITS,
            "repair": REPAIR_LIABILITY_UNITS,
            "replacement": REPLACEMENT_LIABILITY_UNITS,
            "replacement_retry": REPLACEMENT_RETRY_LIABILITY_UNITS,
        }
        live_charge = 0.0
    else:
        if canary_receipt is None:
            raise ValueError("repair admission requires the exact -05 canary terminal receipt")
        live_charge = conservative_charge(canary_receipt)
        liabilities = {
            "repair": REPAIR_LIABILITY_UNITS,
            "replacement": REPLACEMENT_LIABILITY_UNITS,
            "replacement_retry": REPLACEMENT_RETRY_LIABILITY_UNITS,
        }
    # All published unit figures are cents: provider debit is conservatively
    # rounded before it arrives here, and exact decimal display avoids binary
    # float artifacts such as 15.879999999999999 in a binding receipt.
    charged = round(failure["charges"]["charged_units"] + additional + live_charge, 2)
    reserved = round(sum(liabilities.values()), 2)
    total = round(charged + reserved, 2)
    if total > ENVELOPE_UNITS + 1e-9:
        raise ValueError("cumulative charge plus outstanding liabilities exceeds 34.50 u envelope")
    value: dict[str, Any] = {
        "schema": "bl-p1-attempt2-cumulative-admission/v1",
        "status": "PASS",
        "stage": stage,
        "ruling_sha256": ATTEMPT2_RULING_SHA256,
        "finalisation_ruling_sha256": RULING_SHA256,
        "held_manifest": {"path": str(held_manifest.resolve()), "sha256": digest(held_manifest)},
        "failure_accounting": {
            "path": str(failure_accounting.resolve()),
            "sha256": digest(failure_accounting),
        },
        "reconciliation": {"path": str(reconciliation.resolve()), "sha256": digest(reconciliation)},
        "additional_attributable_debit_units": additional,
        "live_stage_charge_units": live_charge,
        "charged_units": charged,
        "outstanding_liabilities": liabilities,
        "outstanding_liabilities_units": reserved,
        "charged_plus_reserved_units": total,
        "envelope_units": ENVELOPE_UNITS,
        "remaining_headroom_units": round(ENVELOPE_UNITS - total, 2),
        "warning": "WARNING_UNEXPLAINED_DEBIT",
        "live_inventory_and_rate_check_required": True,
        "manifest_commit": manifest.get("commit"),
    }
    if canary_receipt is not None:
        value["canary_receipt"] = {
            "path": str(canary_receipt.resolve()),
            "sha256": digest(canary_receipt),
        }
    export_json(output, value)
    return output


def verify_cumulative_admission(path: Path, *, stage: str, held_manifest: Path) -> dict[str, Any]:
    """Refuse a cumulative admission not bound to this exact held manifest and stage."""
    value = _export(path)
    if value.get("held_manifest", {}).get("path") != str(held_manifest.resolve()):
        raise ValueError("cumulative admission names a different held manifest")
    _export(held_manifest)
    failure_path = Path(value.get("failure_accounting", {}).get("path", ""))
    reconciliation_path = Path(value.get("reconciliation", {}).get("path", ""))
    if digest(failure_path) != value.get("failure_accounting", {}).get("sha256") or digest(
        reconciliation_path
    ) != value.get("reconciliation", {}).get("sha256"):
        raise ValueError("cumulative admission input binding changed")
    failure = _failure_accounting(failure_path)
    additional = _ceil_cents(
        _finite_number(
            value.get("additional_attributable_debit_units"), "additional attributable debit"
        )
    )
    _reconciliation(reconciliation_path, additional_attributable_debit_units=additional)
    if stage == "canary":
        live_charge = 0.0
        liabilities = {
            "canary": CANARY_LIABILITY_UNITS,
            "repair": REPAIR_LIABILITY_UNITS,
            "replacement": REPLACEMENT_LIABILITY_UNITS,
            "replacement_retry": REPLACEMENT_RETRY_LIABILITY_UNITS,
        }
    else:
        receipt_path = Path(value.get("canary_receipt", {}).get("path", ""))
        if digest(receipt_path) != value.get("canary_receipt", {}).get("sha256"):
            raise ValueError("cumulative admission canary receipt binding changed")
        live_charge = conservative_charge(receipt_path)
        liabilities = {
            "repair": REPAIR_LIABILITY_UNITS,
            "replacement": REPLACEMENT_LIABILITY_UNITS,
            "replacement_retry": REPLACEMENT_RETRY_LIABILITY_UNITS,
        }
    charged = round(failure["charges"]["charged_units"] + additional + live_charge, 2)
    reserved = round(sum(liabilities.values()), 2)
    total = round(charged + reserved, 2)
    if (
        value.get("schema") != "bl-p1-attempt2-cumulative-admission/v1"
        or value.get("status") != "PASS"
        or value.get("stage") != stage
        or value.get("ruling_sha256") != ATTEMPT2_RULING_SHA256
        or value.get("held_manifest", {}).get("sha256") != digest(held_manifest)
        or value.get("live_stage_charge_units") != live_charge
        or value.get("outstanding_liabilities") != liabilities
        or value.get("charged_units") != charged
        or value.get("outstanding_liabilities_units") != reserved
        or value.get("charged_plus_reserved_units") != total
        or total > ENVELOPE_UNITS + 1e-9
    ):
        raise ValueError("cumulative admission does not bind the exact manifest within envelope")
    return value


def main() -> None:
    """Create only a failure-accounting or exact-manifest cumulative record."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    failure = commands.add_parser("failure-record")
    failure.add_argument("--failure-report", required=True, type=Path)
    failure.add_argument("--failed-receipt", required=True, type=Path)
    failure.add_argument("--output", required=True, type=Path)
    cumulative = commands.add_parser("cumulative")
    cumulative.add_argument("stage", choices=("canary", "repair"))
    cumulative.add_argument("--held-manifest", required=True, type=Path)
    cumulative.add_argument("--failure-accounting", required=True, type=Path)
    cumulative.add_argument("--reconciliation", required=True, type=Path)
    cumulative.add_argument("--additional-attributable-debit-units", required=True, type=float)
    cumulative.add_argument("--output", required=True, type=Path)
    cumulative.add_argument("--canary-receipt", type=Path)
    args = parser.parse_args()
    if args.command == "failure-record":
        print(
            record_failed_repair(
                failure_report=args.failure_report,
                failed_receipt=args.failed_receipt,
                output=args.output,
            )
        )
    else:
        print(
            cumulative_admission(
                stage=args.stage,
                held_manifest=args.held_manifest,
                failure_accounting=args.failure_accounting,
                reconciliation=args.reconciliation,
                additional_attributable_debit_units=args.additional_attributable_debit_units,
                output=args.output,
                canary_receipt=args.canary_receipt,
            )
        )


if __name__ == "__main__":
    main()
