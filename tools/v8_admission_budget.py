"""Immutable reconstructed-budget boundary for the V8 initial-main admission.

The helper is deliberately local and offline.  It creates the one record that
freezes C; later accounting can consume C but cannot recompute it from a newer
provider balance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path("/Users/josenunez/Projects/ml")
ART = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-v8-main-budget-20261024-02"
RECONSTRUCTION = ART / "RECONSTRUCTION.json"
CAMPAIGN_LOG_SNAPSHOT = ART / "CAMPAIGN_LOG_AT_RECONSTRUCTION.md"
BUDGET_RULING = BASE / "snake-dqn-artifacts/big-lever/exp/JUDGMENT_V8_ADMISSION_BUDGET_20261024.md"
BUDGET_RULING_SHA = "c6e6683fb4df8e5831c1d55c640050fe43c265253d58fde586f0041de17a3e59"
SPLIT_RULING_SHA = "f8c1386dc6b82c3cd0f0b74a491b18433d78da48cd3fb28472f16c3a68db22b4"
ANCHOR = "2026-10-24T01:00:00+00:00"


class Refused(ValueError):
    """The operator did not supply an immutable, conservative budget record."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise Refused(f"INCOMPLETE_ADMISSION: invalid nonnegative {name}")
    return float(value)


def _utc(value: Any, name: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise Refused(f"INCOMPLETE_ADMISSION: invalid {name}") from error
    if result.utcoffset() != timedelta(0):
        raise Refused(f"INCOMPLETE_ADMISSION: {name} must be UTC")
    return result


def _zulu(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def sourced(binding: Any) -> Any:
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise Refused("INCOMPLETE_ADMISSION: missing source binding")
    path = Path(binding["path"])
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or sha256(path) != binding.get("sha256")
    ):
        raise Refused("INCOMPLETE_ADMISSION: source bytes changed")
    pointer = binding.get("pointer")
    if not isinstance(pointer, str):
        raise Refused("INCOMPLETE_ADMISSION: missing RFC6901 source pointer")
    value: Any = json.loads(path.read_text())
    if pointer:
        if not pointer.startswith("/"):
            raise Refused("INCOMPLETE_ADMISSION: invalid RFC6901 source pointer")
        for part in pointer[1:].split("/"):
            value = (
                value[int(part)]
                if isinstance(value, list)
                else value[part.replace("~1", "/").replace("~0", "~")]
            )
    return value


def _publish(path: Path, body: dict[str, Any]) -> None:
    targets = (
        path,
        path.with_suffix(path.suffix + ".sha256"),
        path.with_suffix(path.suffix + ".complete"),
    )
    if any(target.exists() for target in targets):
        raise Refused("INCOMPLETE_ADMISSION: reconstruction is create-only")
    payload = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for target, contents in (
        (path, payload),
        (targets[1], hashlib.sha256(payload.encode()).hexdigest() + "  " + path.name + "\n"),
        (targets[2], "complete\n"),
    ):
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(contents.encode())
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(handle.name, target)
        finally:
            Path(handle.name).unlink()


def _verified(path: Path) -> str:
    if (
        not path.is_file()
        or not path.with_suffix(path.suffix + ".sha256").is_file()
        or not path.with_suffix(path.suffix + ".complete").is_file()
    ):
        raise Refused("INCOMPLETE_ADMISSION: missing immutable reconstruction trio")
    value = sha256(path)
    if (
        path.with_suffix(path.suffix + ".sha256").read_text().strip() != f"{value}  {path.name}"
        or path.with_suffix(path.suffix + ".complete").read_text().strip() != "complete"
    ):
        raise Refused("INCOMPLETE_ADMISSION: stale reconstruction sidecar")
    return value


def _check_record(record: dict[str, Any]) -> dict[str, Any]:
    if (
        record.get("authority_kind") != "ASTRA_RECONSTRUCTION_NOT_RECOVERED_SEAL"
        or record.get("policy_version") != "v8-admission-budget-20261024-v1"
    ):
        raise Refused("INCOMPLETE_ADMISSION: wrong reconstruction authority")
    historical_fields = (
        "historical_original_sealed_admission",
        "historical_original_t0",
        "historical_sealed_spendable_units",
    )
    if any(key not in record or record[key] is not None for key in historical_fields):
        raise Refused("INCOMPLETE_ADMISSION: reconstructed record claims recovered history")
    if sha256(BUDGET_RULING) != BUDGET_RULING_SHA:
        raise Refused("INCOMPLETE_ADMISSION: budget ruling bytes differ")
    authority = record.get("authority")
    if (
        not isinstance(authority, dict)
        or authority.get("budget_ruling_sha256") != BUDGET_RULING_SHA
        or authority.get("split_ruling_sha256") != SPLIT_RULING_SHA
    ):
        raise Refused("INCOMPLETE_ADMISSION: missing ruling bindings")
    for key in ("budget_ruling", "split_ruling", "round28", "campaign_log"):
        binding = authority.get(key)
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("sha256"), str)
        ):
            raise Refused("INCOMPLETE_ADMISSION: missing reconstruction authority descriptor")
        path = Path(binding["path"])
        if not path.is_absolute() or not path.is_file() or sha256(path) != binding["sha256"]:
            raise Refused("INCOMPLETE_ADMISSION: reconstruction authority bytes changed")
    campaign = authority["campaign_log"]
    if Path(campaign["path"]).resolve() != CAMPAIGN_LOG_SNAPSHOT.resolve() or not isinstance(
        campaign.get("original_live_path"), str
    ):
        raise Refused("INCOMPLETE_ADMISSION: campaign authority must use immutable snapshot")
    accounting, sources = record.get("accounting"), record.get("accounting_sources")
    if not isinstance(accounting, dict) or not isinstance(sources, dict):
        raise Refused("INCOMPLETE_ADMISSION: missing reconstruction accounting")
    values = {
        key: _number(accounting.get(key), key) for key in ("S", "B", "L_external", "L_sprint")
    }
    if any(_number(sourced(sources.get(key)), key) != value for key, value in values.items()):
        raise Refused("INCOMPLETE_ADMISSION: unsourced reconstruction amount")
    if (
        not isinstance(record.get("accounting_as_of_utc"), str)
        or not isinstance(record.get("raw_usage"), dict)
        or not isinstance(record.get("raw_sessions"), dict)
        or not isinstance(record.get("ledger"), dict)
    ):
        raise Refused("INCOMPLETE_ADMISSION: reconstruction lacks fresh account/ledger evidence")
    _utc(record["accounting_as_of_utc"], "accounting_as_of_utc")
    for key in ("raw_usage", "raw_sessions", "ledger"):
        source = record[key]
        if not isinstance(source.get("path"), str) or not isinstance(source.get("sha256"), str):
            raise Refused("INCOMPLETE_ADMISSION: malformed reconstruction evidence")
        source_path = Path(source["path"])
        if (
            not source_path.is_absolute()
            or not source_path.is_file()
            or sha256(source_path) != source["sha256"]
        ):
            raise Refused("INCOMPLETE_ADMISSION: reconstruction evidence bytes changed")
    raw_balance = re.search(
        r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)", Path(record["raw_usage"]["path"]).read_text()
    )
    if (
        raw_balance is None
        or _number(float(raw_balance.group(1)), "raw provider balance") != values["B"]
    ):
        raise Refused("INCOMPLETE_ADMISSION: frozen B differs from raw provider balance")
    frozen_ledger = sourced(record["ledger"])
    if not isinstance(frozen_ledger, dict) or not isinstance(
        frozen_ledger.get("reconciliation"), dict
    ):
        raise Refused("INCOMPLETE_ADMISSION: frozen ledger is malformed")
    if (
        frozen_ledger.get("schema") != "v8-admission-ledger/v1"
        or frozen_ledger.get("history_basis") != "INVENTORIED_CONSERVATIVE_BOUNDS"
        or frozen_ledger.get("complete_history") is not True
        or frozen_ledger.get("complete_liabilities") is not True
    ):
        raise Refused("INCOMPLETE_ADMISSION: frozen ledger inventory is incomplete")
    frozen_reconciliation = frozen_ledger["reconciliation"]
    try:
        frozen_spent = _number(frozen_reconciliation["spent"]["global"]["units"], "frozen ledger S")
        frozen_liability = _number(
            frozen_reconciliation["liabilities"]["global"]["units"], "frozen ledger L_sprint"
        )
    except (KeyError, TypeError) as error:
        raise Refused("INCOMPLETE_ADMISSION: frozen ledger global totals missing") from error
    seen = set()
    derived_spent = 0.0
    derived_liability = 0.0
    for row in frozen_ledger.get("rows", []):
        if not isinstance(row, dict) or not isinstance(row.get("job_id"), str) or not row["job_id"]:
            raise Refused("INCOMPLETE_ADMISSION: frozen ledger row identity missing")
        if row["job_id"] in seen:
            raise Refused("INCOMPLETE_ADMISSION: duplicate frozen ledger identity")
        seen.add(row["job_id"])
        memberships = row.get("scope_memberships")
        if not isinstance(memberships, list) or "global" not in memberships:
            raise Refused("INCOMPLETE_ADMISSION: frozen ledger row lacks global membership")
        charged = row.get("charged")
        liability = row.get("remaining_liability")
        if sourced(row.get("charge_source")) != charged:
            raise Refused("INCOMPLETE_ADMISSION: frozen row charge differs from bound source")
        if not isinstance(row.get("evidence"), list) or not row["evidence"]:
            raise Refused("INCOMPLETE_ADMISSION: frozen row evidence missing")
        for descriptor in row["evidence"]:
            sourced(descriptor)
        derived_spent += _number(
            charged.get("units") if isinstance(charged, dict) else None, "frozen row charge"
        )
        derived_liability += _number(
            liability.get("units") if isinstance(liability, dict) else None,
            "frozen row remaining liability",
        )
    for hold in frozen_ledger.get("protected_holds", []):
        if not isinstance(hold, dict) or "global" not in hold.get("scope_memberships", []):
            raise Refused("INCOMPLETE_ADMISSION: frozen protected hold lacks global membership")
        charged = hold.get("charged")
        if sourced(hold.get("source")) != charged:
            raise Refused("INCOMPLETE_ADMISSION: frozen hold differs from bound source")
        derived_liability += _number(
            charged.get("units") if isinstance(charged, dict) else None,
            "frozen protected hold",
        )
    if (
        frozen_spent != values["S"]
        or frozen_liability != values["L_sprint"]
        or derived_spent != frozen_spent
        or derived_liability != frozen_liability
    ):
        raise Refused("INCOMPLETE_ADMISSION: frozen S/L do not match bound ledger")
    try:
        frozen_external = _number(
            frozen_reconciliation["external_liabilities_units"],
            "frozen ledger external liabilities",
        )
        frozen_account = _number(
            frozen_reconciliation["account_outstanding_units"], "frozen account liabilities"
        )
    except (KeyError, TypeError) as error:
        raise Refused("INCOMPLETE_ADMISSION: frozen external liabilities missing") from error
    if (
        frozen_external != values["L_external"]
        or frozen_account != frozen_external + frozen_liability
    ):
        raise Refused("INCOMPLETE_ADMISSION: frozen L_external does not match bound ledger")
    historical = accounting.get("historical_authenticated_limit_units")
    if historical is not None:
        historical = _number(historical, "historical authenticated limit")
        if (
            _number(
                sourced(sources.get("historical_authenticated_limit_units")),
                "historical authenticated limit",
            )
            != historical
        ):
            raise Refused("INCOMPLETE_ADMISSION: unsourced historical limit")
    expected = min(
        1100.0,
        values["S"] + values["B"] - values["L_external"],
        historical if historical is not None else float("inf"),
    )
    if (
        expected < 0
        or _number(record.get("reconstructed_spendable_effective_limit_units"), "C") != expected
    ):
        raise Refused("INCOMPLETE_ADMISSION: frozen C differs from conservative formula")
    if (
        _number(record.get("sealed_spendable_effective_limit_units"), "sealed compatibility limit")
        != expected
    ):
        raise Refused("INCOMPLETE_ADMISSION: compatibility sealed C differs")
    if (
        _number(record.get("base_limit_units"), "base limit") != expected / 2
        or _number(record.get("included_reserve_units"), "included reserve") != expected / 6
    ):
        raise Refused("INCOMPLETE_ADMISSION: C components are not derived")
    earlier = record.get("earlier_authenticated_clock_source")
    clock = _utc(record.get("original_t0"), "original_t0")
    anchor = _utc(ANCHOR, "anchor")
    if earlier is None:
        if clock != anchor:
            raise Refused("INCOMPLETE_ADMISSION: reconstructed T0 moved later")
    else:
        earlier_clock = _utc(sourced(earlier), "earlier authenticated clock")
        if earlier_clock > anchor or clock != earlier_clock:
            raise Refused("INCOMPLETE_ADMISSION: later clock cannot extend reconstruction")
    if (
        record.get("reconstruction_anchor_utc") != "2026-10-24T01:00:00Z"
        or record.get("candidate_freeze_deadline_utc") != _zulu(clock + timedelta(hours=48))
        or record.get("integrated_speed_deadline_utc") != _zulu(clock + timedelta(hours=72))
        or record.get("utility_deadline_utc") != _zulu(clock + timedelta(hours=120))
    ):
        raise Refused("INCOMPLETE_ADMISSION: reconstruction deadlines differ")
    return record


def freeze_reconstruction(input_path: Path, output_path: Path = RECONSTRUCTION) -> dict[str, Any]:
    if output_path.resolve() != RECONSTRUCTION.resolve():
        raise Refused("INCOMPLETE_ADMISSION: production reconstruction path is fixed")
    record = _check_record(json.loads(input_path.read_text()))
    age = (
        datetime.now(timezone.utc) - _utc(record["accounting_as_of_utc"], "accounting_as_of_utc")
    ).total_seconds()
    if age < 0 or age > 15 * 60:
        raise Refused("INCOMPLETE_ADMISSION: reconstruction boundary is older than 15 minutes")
    _publish(output_path, record)
    return record


def validate_reconstruction(binding: Any, *, path: Path = RECONSTRUCTION) -> dict[str, Any]:
    if (
        not isinstance(binding, dict)
        or binding.get("path") != str(path)
        or binding.get("pointer") != ""
    ):
        raise Refused("INCOMPLETE_ADMISSION: foreign reconstruction binding")
    if _verified(path) != binding.get("sha256"):
        raise Refused("INCOMPLETE_ADMISSION: reconstruction bytes changed")
    return _check_record(json.loads(path.read_text()))


def reconstruction_working_template() -> dict[str, Any]:
    """Editable operator input; it contains no asserted live accounting values."""
    round28 = BASE / "snake-dqn-artifacts/big-lever/round-28/JUDGMENT.md"
    campaign = BASE / "snake-dqn-gpu-v8/docs/research/final_run_20261019/CAMPAIGN_LOG.md"
    return {
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
        "earlier_authenticated_clock_source": None,
        "authority": {
            "budget_ruling_sha256": BUDGET_RULING_SHA,
            "split_ruling_sha256": SPLIT_RULING_SHA,
            "budget_ruling": {"path": str(BUDGET_RULING), "sha256": sha256(BUDGET_RULING)},
            "split_ruling": {
                "path": str(
                    BASE / "snake-dqn-artifacts/big-lever/exp/JUDGMENT_V8_SPLIT_MAIN_20261024.md"
                ),
                "sha256": SPLIT_RULING_SHA,
            },
            "round28": {"path": str(round28), "sha256": sha256(round28)},
            "campaign_log": {
                "path": str(CAMPAIGN_LOG_SNAPSHOT),
                "sha256": None,
                "original_live_path": str(campaign),
            },
        },
        "accounting_as_of_utc": None,
        "raw_usage": None,
        "raw_sessions": None,
        "ledger": None,
        "accounting": {
            "S": None,
            "B": None,
            "L_external": None,
            "L_sprint": None,
            "historical_authenticated_limit_units": None,
        },
        "accounting_sources": {
            "S": None,
            "B": None,
            "L_external": None,
            "L_sprint": None,
            "historical_authenticated_limit_units": None,
        },
        "reconstructed_spendable_effective_limit_units": None,
        "sealed_spendable_effective_limit_units": None,
        "base_limit_units": None,
        "included_reserve_units": None,
    }


def ledger_working_template() -> dict[str, Any]:
    """Editable inventory shape. Every zero hold needs a separate sourced release."""
    zero = {"wall_minutes": None, "units": None}
    return {
        "schema": "v8-admission-ledger/v1",
        "history_basis": "INVENTORIED_CONSERVATIVE_BOUNDS",
        "complete_history": False,
        "complete_liabilities": False,
        "rows": [
            {
                "job_id": None,
                "endpoint": None,
                "authority_sha256": None,
                "manifest_sha256": None,
                "scope_memberships": ["global"],
                "charged": zero,
                "charge_basis": None,
                "charge_source": None,
                "evidence": [],
                "terminal_status": None,
                "release_status": None,
                "remaining_liability": {"wall_minutes": 0, "units": 0},
            }
        ],
        "protected_holds": [
            {
                "name": name,
                "scope_memberships": ["global", "q4"] if name == "I0" else ["global"],
                "charged": zero,
                "source": None,
                "released_or_completed_source": None,
            }
            for name in ("I0", "Q3", "Q5", "Q6", "O1")
        ],
        "reconciliation": None,
    }


def charge_working_template() -> list[dict[str, Any]]:
    return [
        {
            "job_id": None,
            "receipt_sha256": None,
            "charged": {"wall_minutes": None, "units": None},
            "source": None,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--input", type=Path, required=True)
    commands.add_parser("templates")
    args = parser.parse_args()
    if args.command == "freeze":
        result: Any = freeze_reconstruction(args.input)
    else:
        result = {
            "reconstruction": reconstruction_working_template(),
            "ledger": ledger_working_template(),
            "charges": charge_working_template(),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
