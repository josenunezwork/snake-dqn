"""Mac-only, fail-closed operator accounting for the amended V8 split main.

This companion never invokes Colab or dispatches.  ``draft`` writes the immutable
historical reconciliation and an editable live template.  ``check`` validates an
operator-filled body and prints the exact body digest that Codex-V must accept.
``finalize`` verifies that independent acceptance before publishing the create-only
OPERATOR_ADMISSION.json trio consumed by v8_split_main_amend.admission().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools import v8_admission_budget as budget
from tools.v8_admission_evidence import sourced, validate_evidence

BASE = Path("/Users/josenunez/Projects/ml")
ART = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-v8-main-budget-20261024-02"
RULING = budget.BUDGET_RULING
RULING_SHA = budget.BUDGET_RULING_SHA
PRIOR_RULING = BASE / "snake-dqn-artifacts/big-lever/exp/JUDGMENT_V8_SPLIT_MAIN_20261024.md"
PRIOR_RULING_SHA = budget.SPLIT_RULING_SHA
MANIFEST = ART / "v8-split-main.json"
MAIN_CAP = {"wall_minutes": 120, "units": 20}
V8_ALLOWANCE = {"wall_minutes": 216, "units": 36}
PACKET_ALLOWANCE = {"wall_minutes": 150, "units": 25}
HISTORICAL_CANARY_CAP = {"wall_minutes": 30, "units": 5}

# All six count against the original 216/36 ceiling. The boolean identifies
# split-packet canaries, which additionally count against the amended 150/25.
RECEIPTS = (
    (
        "metadata_canary",
        BASE
        / "snake-dqn-artifacts/big-lever/exp/gpu-v8-metadata-20261024-02"
        / "canary-output/receipt.json",
        True,
    ),
    (
        "split01_canary",
        BASE
        / "snake-dqn-artifacts/big-lever/exp/gpu-v8-split-20261023-01/canary-output/receipt.json",
        True,
    ),
    (
        "split02_canary",
        BASE
        / "snake-dqn-artifacts/big-lever/exp/gpu-v8-split-20261023-02/canary-output/receipt.json",
        True,
    ),
    (
        "sprint02_failed_allocation",
        BASE
        / "snake-dqn-artifacts/big-lever/exp/gpu-v8-sprint-20261023-02/canary-output/receipt.json",
        False,
    ),
    (
        "sprint03_canary",
        BASE
        / "snake-dqn-artifacts/big-lever/exp/gpu-v8-sprint-20261023-03"
        / "canary-output-n03/receipt.json",
        False,
    ),
    (
        "sprint03_main",
        BASE
        / "snake-dqn-artifacts/big-lever/exp/gpu-v8-sprint-20261023-03"
        / "main-output-n03/receipt.json",
        False,
    ),
)


class Refused(ValueError):
    """An accounting record lacks evidence necessary for admission."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def body_digest(value: dict[str, Any]) -> str:
    copied = dict(value)
    copied.pop("codex_v_review", None)
    return hashlib.sha256(json.dumps(copied, sort_keys=True, allow_nan=False).encode()).hexdigest()


def publish(path: Path, value: dict[str, Any]) -> None:
    """Publish the repository's payload/hash/completion trio exactly once."""
    for target in (
        path,
        path.with_suffix(path.suffix + ".sha256"),
        path.with_suffix(path.suffix + ".complete"),
    ):
        if target.exists():
            raise Refused(f"create-only target exists: {target}")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    value_hash = hashlib.sha256(encoded.encode()).hexdigest()
    for target, payload in (
        (path, encoded),
        (path.with_suffix(path.suffix + ".sha256"), value_hash + "  " + path.name + "\n"),
        (path.with_suffix(path.suffix + ".complete"), "complete\n"),
    ):
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload.encode())
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary.name, target)
        finally:
            Path(temporary.name).unlink()


def verified(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    complete = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not sidecar.is_file() or not complete.is_file():
        raise Refused(f"missing immutable trio: {path}")
    actual = digest(path)
    if (
        sidecar.read_text().strip() != f"{actual}  {path.name}"
        or complete.read_text().strip() != "complete"
    ):
        raise Refused(f"forged or stale immutable trio: {path}")
    return actual


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value <= 0)
    ):
        raise Refused(f"invalid {name}")
    return float(value)


def receipt_rows() -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for name, path, post_anchor in RECEIPTS:
        if path in seen:
            raise Refused("duplicate V8 receipt path")
        seen.add(path)
        raw = json.loads(path.read_text())
        elapsed = _number(raw.get("elapsed_seconds"), f"{name}.elapsed_seconds", positive=True)
        rate = _number(
            raw.get("accounting", {}).get("own_rate_units_per_hour"), name + ".rate", positive=True
        )
        if rate != 10 or raw.get("stop_verified") is not True:
            raise Refused("V8 historical rate/release differs from reconciliation")
        rows.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest(path),
                "status": raw.get("status"),
                "job_id": raw.get("job_id"),
                "elapsed_seconds": elapsed,
                "own_rate_units_per_hour": rate,
                "field_pointers": {
                    "elapsed_seconds": "/elapsed_seconds",
                    "rate": "/accounting/own_rate_units_per_hour",
                    "stop_verified": "/stop_verified",
                    "job_id": "/job_id",
                },
                "at_10_units_per_hour": elapsed * rate / 3600,
                "split_packet": post_anchor,
            }
        )
    return rows


def receipt_hashes() -> dict[str, str]:
    """Map immutable broker job identities to terminal receipt bytes."""
    result = {}
    for row in receipt_rows():
        job_id = row["job_id"]
        if not isinstance(job_id, str) or not job_id or job_id in result:
            raise Refused("missing or duplicate V8 receipt job_id")
        result[job_id] = row["sha256"]
    return result


def draft_record() -> dict[str, Any]:
    if digest(RULING) != RULING_SHA:
        raise Refused("cap ruling hash differs")
    if digest(PRIOR_RULING) != PRIOR_RULING_SHA:
        raise Refused("prior split ruling hash differs")
    rows = receipt_rows()
    post = [row for row in rows if row["split_packet"]]
    packet_minutes = sum(row["elapsed_seconds"] for row in post) / 60
    packet_units = sum(row["at_10_units_per_hour"] for row in post)
    failed_receipt = next(row for row in rows if row["name"] == "sprint02_failed_allocation")
    handoff_path = BASE / "snake-dqn-gpu-v8/research/gpu_sim_spike/V8_SPLIT_HANDOFF_20261023.md"
    handoff_seconds = 55.95090962498216
    conservative_seconds = max(failed_receipt["elapsed_seconds"], handoff_seconds)
    receipt_total_seconds = sum(row["elapsed_seconds"] for row in rows)
    conservative_total_seconds = (
        receipt_total_seconds - failed_receipt["elapsed_seconds"] + conservative_seconds
    )
    prior_path = (
        BASE
        / "snake-dqn-artifacts/a100-retention-20261023/acceptance-01/boundary-admission.draft.json"
    )
    prior = json.loads(prior_path.read_text())
    round28 = BASE / "snake-dqn-artifacts/big-lever/round-28/JUDGMENT.md"
    intent = BASE / "snake-dqn-gpu-v8/research/gpu_sim_spike/V8_INTENT_20261023.md"
    return {
        "schema": "v8-budget-operator-admission-draft/v1",
        "status": "INCOMPLETE_ADMISSION",
        "note": (
            "Historical reconciliation only. Null is unknown, never zero; "
            "no dispatch is authorized."
        ),
        "ruling": {"path": str(RULING), "sha256": RULING_SHA},
        "prior_split_ruling": {"path": str(PRIOR_RULING), "sha256": PRIOR_RULING_SHA},
        "manifest": {"path": str(MANIFEST), "sha256": digest(MANIFEST)},
        "authority_kind": "ASTRA_RECONSTRUCTION_NOT_RECOVERED_SEAL",
        "policy_version": "v8-admission-budget-20261024-v1",
        "historical_original_sealed_admission": None,
        "historical_original_t0": None,
        "historical_sealed_spendable_units": None,
        "reconstruction_anchor_utc": "2026-10-24T01:00:00Z",
        "original_t0": "2026-10-24T01:00:00Z",
        "candidate_freeze_deadline_utc": "2026-10-26T01:00:00Z",
        "integrated_speed_deadline_utc": "2026-10-27T01:00:00Z",
        "utility_deadline_utc": "2026-10-29T01:00:00Z",
        "packet_cap": PACKET_ALLOWANCE,
        "historical_canary_charge_ceiling": HISTORICAL_CANARY_CAP,
        "reserved_main": MAIN_CAP,
        "base_topup": {"wall_minutes": 3, "units": 0.5},
        "effective_topup": {"wall_minutes": 6, "units": 1},
        "reconstruction": None,
        "ledger": None,
        "authority_sources": {
            "original_v8_ceiling": {
                "path": str(intent),
                "sha256": digest(intent),
                "lines": "58-69",
                "value": V8_ALLOWANCE,
            },
            "parent_ceilings": {
                "path": str(round28),
                "sha256": digest(round28),
                "lines": "45-74",
                "q2": {"wall_minutes": 864, "units": 144},
                "q4": {"wall_minutes": 1152, "units": 192},
                "global_maximum": {"wall_minutes": 7200, "units": 1100},
            },
            "prior_reconciliation": {
                "path": str(prior_path),
                "sha256": digest(prior_path),
                "pointer": "/reconciliation",
            },
        },
        "historical_unknown_sources": {
            key: {
                "source": {
                    "path": str(prior_path),
                    "sha256": digest(prior_path),
                    "pointer": pointer,
                },
                "value": None,
            }
            for key, pointer in (
                ("original_sealed_admission", "/original_sealed_admission"),
                ("original_t0", "/t0_epoch"),
                (
                    "sealed_spendable_effective_limit_units",
                    "/reconciliation/round_28_allowance/sealed_spendable_effective_limit_units",
                ),
                (
                    "booked_outstanding_units",
                    "/reconciliation/liabilities/booked_outstanding_units",
                ),
            )
        },
        "prior_liability_references_not_booked_balances": prior["reconciliation"]["liabilities"],
        "historical_expenditure": {
            "all_known_v8_receipts": rows,
            "receipt_only_elapsed_minutes": receipt_total_seconds / 60,
            "receipt_only_at_10_units_per_hour": receipt_total_seconds / 360,
            "conservative_failed_allocation_elapsed_seconds": conservative_seconds,
            "conservative_elapsed_minutes": conservative_total_seconds / 60,
            "conservative_at_10_units_per_hour": conservative_total_seconds / 360,
            "failed_allocation_discrepancy": {
                "receipt_elapsed_seconds": failed_receipt["elapsed_seconds"],
                "handoff_claim_elapsed_seconds": handoff_seconds,
                "handoff_source": {
                    "path": str(handoff_path),
                    "sha256": digest(handoff_path),
                    "lines": "66-68",
                },
                "resolution": (
                    "UNRESOLVED; reserve max once, never add both; "
                    "provider debit remains unverified."
                ),
            },
            "provider_debit_reconciliation": None,
        },
        "v8_series": {
            "historical_ceiling_not_a_new_grant": V8_ALLOWANCE,
            "all_known_receipts_conservatively_charged_once": [row["name"] for row in rows],
            "provisional_after_all_known_v8_receipts": {
                "wall_minutes": V8_ALLOWANCE["wall_minutes"] - conservative_total_seconds / 60,
                "units": V8_ALLOWANCE["units"] - conservative_total_seconds / 360,
            },
            "warning": (
                "216/36 is historical authority only, never a reset. "
                "Original sealed T0 and liabilities must reconcile this provisional calculation."
            ),
        },
        "split_packet": {
            "amended_cap": PACKET_ALLOWANCE,
            "historical_canary_charge_ceiling": HISTORICAL_CANARY_CAP,
            "known_post_anchor_spend": {"wall_minutes": packet_minutes, "units": packet_units},
            "provisional_remainder": {
                "wall_minutes": PACKET_ALLOWANCE["wall_minutes"] - packet_minutes,
                "units": PACKET_ALLOWANCE["units"] - packet_units,
            },
            "required_main_reservation": MAIN_CAP,
            "admissible_from_known_receipts": (
                PACKET_ALLOWANCE["wall_minutes"] - packet_minutes >= MAIN_CAP["wall_minutes"]
                and PACKET_ALLOWANCE["units"] - packet_units >= MAIN_CAP["units"]
            ),
        },
        "liabilities_and_reservations": None,
        "global_remaining": None,
        "q2_q4_remaining": None,
        "live_unencumbered_units": None,
        "current_assignments_and_rates": None,
        "calendar_deadline_utc": "2026-10-26T01:00:00Z",
        "blockers": [
            "Create the immutable reconstructed T0/C record; never reset it from a later balance.",
            "All liabilities/reservations and Q2/Q4 balances are unknown.",
            "Live usage/sessions/rates, release and account-cap evidence are operator-only.",
        ],
    }


def template(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "INCOMPLETE_ADMISSION",
        "manifest_sha256": draft["manifest"]["sha256"],
        "ruling_sha256": RULING_SHA,
        "prior_split_ruling_sha256": PRIOR_RULING_SHA,
        "packet_cap": PACKET_ALLOWANCE,
        "historical_canary_charge_ceiling": HISTORICAL_CANARY_CAP,
        "reserved_main": MAIN_CAP,
        "original_t0": draft["original_t0"],
        "reconstructed_spendable_effective_limit_units": None,
        "effective_spendable_limit_units": None,
        "historical_expenditure": draft["historical_expenditure"],
        "liabilities_and_reservations": None,
        "q2_q4_remaining": None,
        "v8_series_remaining": None,
        "split_packet_remaining": None,
        "global_remaining": None,
        "live_unencumbered_units": None,
        "verified_main_rate_units_per_hour": None,
        "previous_endpoint": None,
        "current_assignments_and_rates": None,
        "calendar_deadline_utc": draft["calendar_deadline_utc"],
        "accounting_evidence": {
            "reconstruction": None,
            "ledger": None,
            "v8_charge_reconciliation": None,
            "usage": None,
            "sessions": None,
            "active_records": [],
            "main_rate": None,
            "previous_endpoint_release": None,
            "outside_cap_main_release_liability": None,
            "later_tightening": {
                "earlier_authenticated_clock_source": None,
                "smaller_historical_limit_source": None,
            },
        },
        "accounting_as_of_utc": None,
        "valid_until_utc": None,
        "no_main_unit_has_run": None,
        "previous_endpoint_release_verified": None,
        "a100_serial_sprint": None,
        "fits_all_original_parent_caps_and_deadlines": None,
        "outside_cap_main_release_liability": {"wall_minutes": None, "units": None},
        "main_reservation": {
            "reservation_id": None,
            "scope_memberships": ["packet", "v8", "q2", "q4", "global"],
            "charged": MAIN_CAP,
        },
        "codex_v_review": {"path": None, "sha256": None},
    }


def write_draft(output: Path = ART) -> dict[str, Any]:
    record = draft_record()
    publish(output / "OPERATOR_ADMISSION.draft.json", record)
    publish(output / "OPERATOR_ADMISSION.template.json", template(record))
    implementation = {
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": digest(Path(__file__).resolve()),
        },
        "tests": {
            "path": str(Path(__file__).resolve().parents[1] / "tests/test_v8_split_admission.py"),
            "sha256": digest(
                Path(__file__).resolve().parents[1] / "tests/test_v8_split_admission.py"
            ),
        },
        "draft_sha256": verified(output / "OPERATOR_ADMISSION.draft.json"),
        "template_sha256": verified(output / "OPERATOR_ADMISSION.template.json"),
        "evidence_helper": {
            "path": str(Path(__file__).resolve().with_name("v8_admission_evidence.py")),
            "sha256": digest(Path(__file__).resolve().with_name("v8_admission_evidence.py")),
        },
        "evidence_tests": {
            "path": str(
                Path(__file__).resolve().parents[1] / "tests/test_v8_admission_evidence.py"
            ),
            "sha256": digest(
                Path(__file__).resolve().parents[1] / "tests/test_v8_admission_evidence.py"
            ),
        },
        "budget_helper": {
            "path": str(Path(__file__).resolve().with_name("v8_admission_budget.py")),
            "sha256": digest(Path(__file__).resolve().with_name("v8_admission_budget.py")),
        },
        "budget_tests": {
            "path": str(Path(__file__).resolve().parents[1] / "tests/test_v8_admission_budget.py"),
            "sha256": digest(
                Path(__file__).resolve().parents[1] / "tests/test_v8_admission_budget.py"
            ),
        },
    }
    publish(
        output / "PREPARATION_ACCOUNTING.json",
        {
            "status": "PREPARED_ACCOUNTING_DRAFT_NOT_ADMITTED",
            "implementation": implementation,
            "blockers": record["blockers"],
            "allocations": 0,
        },
    )
    return record


def _required_object(body: dict[str, Any], key: str) -> Any:
    value = body.get(key)
    if value is None or value == "":
        raise Refused(f"missing {key}")
    return value


def validate_live(body: dict[str, Any], now: datetime | None = None) -> str:
    """Validate exactly the evidence old amendment admission() expects, before publish."""
    if (
        body.get("status") != "ADMIT_INITIAL_MAIN"
        or body.get("manifest_sha256") != digest(MANIFEST)
        or body.get("ruling_sha256") != RULING_SHA
        or body.get("prior_split_ruling_sha256") != PRIOR_RULING_SHA
    ):
        raise Refused("wrong admitted identity")
    if (
        body.get("reserved_main") != MAIN_CAP
        or body.get("packet_cap") != PACKET_ALLOWANCE
        or body.get("historical_canary_charge_ceiling") != HISTORICAL_CANARY_CAP
    ):
        raise Refused("full 120-minute/20-unit main cap is not reserved")
    for key in (
        "original_t0",
        "historical_expenditure",
        "liabilities_and_reservations",
        "q2_q4_remaining",
        "v8_series_remaining",
        "live_unencumbered_units",
        "current_assignments_and_rates",
        "calendar_deadline_utc",
        "accounting_evidence",
    ):
        _required_object(body, key)
    for key in (
        "no_main_unit_has_run",
        "previous_endpoint_release_verified",
        "a100_serial_sprint",
        "fits_all_original_parent_caps_and_deadlines",
    ):
        if body.get(key) is not True:
            raise Refused(f"required true field absent: {key}")
    now = now or datetime.now(timezone.utc)
    times = {}
    for key in ("accounting_as_of_utc", "valid_until_utc", "calendar_deadline_utc"):
        try:
            times[key] = datetime.fromisoformat(body[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise Refused(f"invalid {key}") from exc
        if times[key].tzinfo is None or times[key].utcoffset() != timedelta(0):
            raise Refused(f"non-UTC {key}")
    if not times["accounting_as_of_utc"] <= now <= times["valid_until_utc"]:
        raise Refused("stale accounting")
    if times["valid_until_utc"] - times["accounting_as_of_utc"] > timedelta(
        minutes=15
    ) or now - times["accounting_as_of_utc"] > timedelta(minutes=15):
        raise Refused("accounting freshness exceeds 15 minutes")
    if now + timedelta(minutes=120) > times["calendar_deadline_utc"]:
        raise Refused("deadline cannot fit full main")
    try:
        outside = body.get("outside_cap_main_release_liability")
        outside_source = body.get("accounting_evidence", {}).get(
            "outside_cap_main_release_liability"
        )
        if not isinstance(outside, dict) or sourced(outside_source) != outside:
            raise Refused("missing sourced outside-cap release liability")
        outside_minutes = _number(outside.get("wall_minutes"), "outside-cap release minutes")
        outside_units = _number(outside.get("units"), "outside-cap release units")
        reservation = body.get("main_reservation")
        if (
            not isinstance(reservation, dict)
            or not isinstance(reservation.get("reservation_id"), str)
            or not reservation["reservation_id"]
            or reservation.get("charged") != MAIN_CAP
            or set(reservation.get("scope_memberships", []))
            != {"packet", "v8", "q2", "q4", "global"}
        ):
            raise Refused("missing shared initial-main reservation")
        ledger = sourced(body["accounting_evidence"]["ledger"])
        if any(
            row.get("reservation_id") == reservation["reservation_id"]
            for row in [*ledger.get("rows", []), *ledger.get("protected_holds", [])]
            if isinstance(row, dict)
        ):
            raise Refused("initial-main reservation was prebooked in liabilities")
        charges = sourced(body["accounting_evidence"]["v8_charge_reconciliation"])
        if not isinstance(charges, list):
            raise Refused("missing historical V8 charge list")
        charges_by_job = {row.get("job_id"): row for row in charges if isinstance(row, dict)}
        rows = receipt_rows()
        failed = next(row for row in rows if row["name"] == "sprint02_failed_allocation")
        conservative_failed_seconds = max(failed["elapsed_seconds"], 55.95090962498216)
        for row in rows:
            expected_seconds = (
                conservative_failed_seconds
                if row["name"] == "sprint02_failed_allocation"
                else row["elapsed_seconds"]
            )
            charge = charges_by_job.get(row["job_id"])
            if not isinstance(charge, dict):
                raise Refused("missing historical V8 conservative charge")
            actual = charge.get("charged")
            expected_minutes, expected_units = expected_seconds / 60, expected_seconds / 360
            if (
                not isinstance(actual, dict)
                or _number(actual.get("wall_minutes"), "charge minutes") < expected_minutes
                or _number(actual.get("units"), "charge units") < expected_units
            ):
                raise Refused("historical V8 charge is below conservative receipt floor")
        validate_evidence(body, now=now, receipt_hashes=receipt_hashes())
        reconstruction = budget.validate_reconstruction(
            body["accounting_evidence"]["reconstruction"], path=budget.RECONSTRUCTION
        )
        if (
            body.get("reconstructed_spendable_effective_limit_units")
            != reconstruction["reconstructed_spendable_effective_limit_units"]
        ):
            raise Refused("frozen C differs from reconstruction")
        for field in (
            "split_packet_remaining",
            "v8_series_remaining",
            "global_remaining",
            "q2_q4_remaining",
        ):
            available = body[field]
            if (
                _number(available.get("wall_minutes"), field) < 120 + outside_minutes
                or _number(available.get("units"), field) < 20 + outside_units
            ):
                raise Refused("scope cannot cover main plus outside-cap release liability")
        if _number(body.get("live_unencumbered_units"), "live units") < 20 + outside_units:
            raise Refused("account cannot cover main plus outside-cap release liability")
    except ValueError as error:
        raise Refused(str(error)) from error
    return body_digest(body)


def finalize(input_path: Path, review_path: Path, output: Path = ART) -> str:
    review_path = review_path.resolve(strict=True)
    # The scoped companion validates every frozen runtime/canary/source/export
    # invariant.  It is intentionally the only preparation authority here.
    from tools import v8_budget_main_amend as amend

    amend.verify_preparation(require_review=True)
    accounting_prep = output / "PREPARATION_ACCOUNTING.json"
    draft_path = output / "OPERATOR_ADMISSION.draft.json"
    offline_review_path = output / "CODEX_V_ADMISSION_REVIEW.json"
    accounting_prep_hash = verified(accounting_prep)
    draft_hash = verified(draft_path)
    verified(offline_review_path)
    accounting = json.loads(accounting_prep.read_text())
    offline_review = json.loads(offline_review_path.read_text())
    implementation = accounting.get("implementation")
    if (
        not isinstance(implementation, dict)
        or accounting.get("status") != "PREPARED_ACCOUNTING_DRAFT_NOT_ADMITTED"
    ):
        raise Refused("missing accounting preparation binding")
    expected_paths = {
        "source": Path(__file__).resolve(),
        "tests": Path(__file__).resolve().parents[1] / "tests/test_v8_split_admission.py",
        "evidence_helper": Path(__file__).resolve().with_name("v8_admission_evidence.py"),
        "evidence_tests": Path(__file__).resolve().parents[1]
        / "tests/test_v8_admission_evidence.py",
        "budget_helper": Path(__file__).resolve().with_name("v8_admission_budget.py"),
        "budget_tests": Path(__file__).resolve().parents[1] / "tests/test_v8_admission_budget.py",
    }
    for key in expected_paths:
        binding = implementation.get(key, {})
        if (
            not isinstance(binding, dict)
            or binding.get("path") != str(expected_paths[key])
            or digest(Path(binding.get("path", ""))) != binding.get("sha256")
        ):
            raise Refused(f"changed accounting implementation: {key}")
    if implementation.get("draft_sha256") != draft_hash:
        raise Refused("accounting preparation binds a different draft")
    if implementation.get("template_sha256") != verified(
        output / "OPERATOR_ADMISSION.template.json"
    ):
        raise Refused("accounting preparation binds a different template")
    for row in json.loads(draft_path.read_text())["historical_expenditure"][
        "all_known_v8_receipts"
    ]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise Refused("historical receipt changed after independent draft acceptance")
    if (
        offline_review.get("status") != "ACCEPT_OFFLINE_DRAFT"
        or offline_review.get("draft_sha256") != draft_hash
        or offline_review.get("preparation_accounting_sha256") != accounting_prep_hash
    ):
        raise Refused("no independent Codex-V accounting draft acceptance")
    expected_impl = {
        binding["path"]: binding["sha256"]
        for key, binding in implementation.items()
        if key in expected_paths
    }
    if offline_review.get("implementation_sha256") != expected_impl:
        raise Refused("Codex-V draft review binds different implementation")
    body = json.loads(input_path.read_text())
    if (
        body.get("historical_expenditure")
        != json.loads(draft_path.read_text())["historical_expenditure"]
    ):
        raise Refused(
            "historical accounting differs from accepted draft; use sourced charge reconciliation"
        )
    body["codex_v_review"] = {"path": str(review_path), "sha256": verified(review_path)}
    accounting_sha = validate_live(body)
    review = json.loads(review_path.read_text())
    if review.get("status") != "ACCEPT" or review.get("accounting_sha256") != accounting_sha:
        raise Refused("no independent Codex-V ACCEPT for exact accounting body")
    publish(output / "OPERATOR_ADMISSION.json", body)
    return accounting_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("draft")
    check = commands.add_parser("check")
    check.add_argument("--input", type=Path, required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--input", type=Path, required=True)
    final.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "draft":
        result: Any = write_draft()
    elif args.command == "check":
        result = {
            "status": "REVIEW_REQUIRED",
            "accounting_sha256": validate_live(json.loads(args.input.read_text())),
        }
    else:
        result = {"status": "FINALIZED", "accounting_sha256": finalize(args.input, args.review)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
