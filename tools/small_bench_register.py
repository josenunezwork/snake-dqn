"""Append-only, local financial register for small benchmark admissions.

The register is deliberately provider-agnostic.  A caller captures provider state,
then holds the *shared account* lock around ``read -> reserve -> append``.  This
module only records the bounded accounting decision; it never allocates or queries
an endpoint.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


class Refused(ValueError):
    """Raised when a register row or proposed admission is not conservative."""


class AuditGateWarning(UserWarning):
    """Accounting-audit gate breach; recorded and emitted but non-blocking."""


PAIR_KEYS = ("units", "wall_minutes")
# A ceiling_raise row may carry only these keys: it can never add scopes, holds,
# spend, settlements, debts, timeouts or anything else (owner 2026-10-24: no budgets).
CEILING_RAISE_KEYS = frozenset(
    {
        "type",
        "id",
        "ceilings",
        "sources",
        "attestation",
        "owner_words",
        "note",
        "revision",
        "prev_hash",
        "row_hash",
    }
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Refused(f"invalid finite {label}")
    result = float(value)
    if result < 0 or (positive and result <= 0):
        raise Refused(f"invalid nonnegative {label}")
    return result


def _pair(value: Any, label: str, *, positive: bool = False) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(PAIR_KEYS):
        raise Refused(f"invalid {label} pair")
    return {key: _number(value[key], f"{label}.{key}", positive=positive) for key in PAIR_KEYS}


def _plus(*pairs: dict[str, float]) -> dict[str, float]:
    return {key: sum(pair[key] for pair in pairs) for key in PAIR_KEYS}


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise Refused(f"missing {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Refused(f"invalid {label}") from error
    if parsed.tzinfo is None:
        raise Refused(f"timezone required for {label}")
    return parsed.astimezone(timezone.utc)


def _identity(event: dict[str, Any]) -> str:
    value = event.get("id")
    if not isinstance(value, str) or not value:
        raise Refused("event requires nonempty id")
    return value


def _event_hash(event: dict[str, Any]) -> str:
    body = dict(event)
    body.pop("row_hash", None)
    return _sha(body)


def _sources(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise Refused("nonempty sources required")
    for source in value:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise Refused("invalid source path")
        digest = source.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise Refused("invalid source sha256")


def _validate_init(row: dict[str, Any]) -> None:
    scopes = row.get("scopes")
    if not isinstance(scopes, dict) or not scopes:
        raise Refused("initial row requires scopes")
    for name, scope in scopes.items():
        if not isinstance(name, str) or not name or not isinstance(scope, dict):
            raise Refused("invalid initial scope")
        _pair(scope.get("ceiling"), f"scope {name} ceiling")
        _pair(scope.get("spent"), f"scope {name} spent")
    holds = row.get("holds")
    if not isinstance(holds, list):
        raise Refused("initial row requires holds list")
    for hold in holds:
        _validate_hold(hold)
    if len({hold["id"] for hold in holds}) != len(holds):
        raise Refused("duplicate initial hold id")
    _number(row.get("external_commitments_units", 0), "external commitments")
    _number(row.get("unsettled_debit_holds_units", 0), "unsettled debit holds")
    if (
        not isinstance(row.get("completeness_attestation"), str)
        or not row["completeness_attestation"].strip()
    ):
        raise Refused("initial row requires completeness attestation")
    _sources(row.get("sources"))


def _validate_hold(hold: Any) -> None:
    if not isinstance(hold, dict) or not isinstance(hold.get("id"), str) or not hold["id"]:
        raise Refused("hold requires id")
    if not isinstance(hold.get("scopes"), list) or not hold["scopes"]:
        raise Refused("hold requires scopes")
    _pair(hold.get("pair"), "hold")
    _number(hold.get("account_units", hold["pair"]["units"]), "hold account units")
    if not isinstance(hold.get("kind"), str) or not hold["kind"]:
        raise Refused("hold requires kind")


def read(path: Path | str) -> list[dict[str, Any]]:
    """Read and verify a complete JSONL hash chain; absent file is an empty register."""
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    previous = None
    for line_number, line in enumerate(target.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Refused(f"invalid JSONL row {line_number}") from error
        if not isinstance(row, dict):
            raise Refused(f"invalid register row {line_number}")
        if row.get("revision") != line_number - 1 or row.get("prev_hash") != previous:
            raise Refused("broken register revision/hash chain")
        if row.get("row_hash") != _event_hash(row):
            raise Refused("broken register row hash")
        rows.append(row)
        previous = row["row_hash"]
    state(rows)  # validates semantic history too
    return rows


@contextlib.contextmanager
def reservation_lock(path: Path | str) -> Iterator[None]:
    """Acquire caller-selected shared account lock without waiting for another operator."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Refused("account reservation lock is already held") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def append(path: Path | str, event: dict[str, Any]) -> dict[str, Any]:
    """Atomically append an event by replacing the fully fsynced JSONL file.

    The caller must hold ``reservation_lock`` when an event affects admission.
    """
    target = Path(path)
    rows = read(target)
    row = dict(event)
    row.pop("revision", None)
    row.pop("prev_hash", None)
    row.pop("row_hash", None)
    row["revision"] = len(rows)
    row["prev_hash"] = rows[-1]["row_hash"] if rows else None
    row["row_hash"] = _event_hash(row)
    state([*rows, row])
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(_canonical(existing) for existing in [*rows, row])
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return row


def state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive conservative current totals and validate all supported row types."""
    if not rows:
        return {"initialized": False, "reservations": {}, "outstanding": {}, "audit_due": None}
    if rows[0].get("type") != "init":
        raise Refused("first row must be init")
    _identity(rows[0])
    _validate_init(rows[0])
    scopes = {
        name: {"ceiling": _pair(value["ceiling"], name), "spent": _pair(value["spent"], name)}
        for name, value in rows[0]["scopes"].items()
    }
    holds: dict[str, dict[str, Any]] = {hold["id"]: dict(hold) for hold in rows[0]["holds"]}
    external_commitments = _number(
        rows[0].get("external_commitments_units", 0), "external commitments"
    )
    unsettled_debit_holds = _number(
        rows[0].get("unsettled_debit_holds_units", 0), "unsettled debit holds"
    )
    reservations: dict[str, dict[str, Any]] = {}
    seen_ids = {_identity(rows[0])}
    material_defect = False
    audits: list[dict[str, Any]] = []
    for row in rows[1:]:
        identifier = _identity(row)
        if identifier in seen_ids:
            raise Refused("duplicate register event id")
        seen_ids.add(identifier)
        kind = row.get("type")
        if kind == "admission":
            reservation = row.get("reservation")
            if not isinstance(reservation, dict):
                raise Refused("admission requires reservation")
            rid = reservation.get("id")
            if not isinstance(rid, str) or not rid:
                raise Refused("admission requires reservation id")
            _validate_hold(reservation)
            package = row.get("package")
            if not isinstance(package, dict):
                raise Refused("admission requires package")
            identity = (
                package.get("job_id"),
                package.get("manifest_sha256"),
                package.get("packet_id"),
            )
            if not all(isinstance(value, str) and value for value in identity):
                raise Refused("admission package identity missing")
            if rid in reservations:
                if reservations[rid]["identity"] != identity:
                    raise Refused("reservation id reused for changed package")
                raise Refused("duplicate reservation admission")
            reservations[rid] = {
                "identity": identity,
                "row": row,
                "reservation": reservation,
                "settled": False,
            }
        elif kind == "refresh":
            rid = row.get("reservation_id")
            package = row.get("package")
            if rid not in reservations or not isinstance(package, dict):
                raise Refused("refresh requires an existing reservation")
            identity = (
                package.get("job_id"),
                package.get("manifest_sha256"),
                package.get("packet_id"),
            )
            if reservations[rid]["identity"] != identity:
                raise Refused("refresh changed package identity")
        elif kind == "settlement":
            rid = row.get("reservation_id")
            if rid not in reservations or reservations[rid]["settled"]:
                raise Refused("invalid settlement reservation")
            _sources(row.get("sources"))
            charged = _pair(row.get("charged"), "settlement charged")
            remaining = _pair(row.get("remaining_hold"), "settlement remaining hold")
            prior = _pair(reservations[rid]["reservation"]["pair"], "reservation")
            if any(charged[key] + remaining[key] > prior[key] + 1e-9 for key in PAIR_KEYS):
                raise Refused("settlement exceeds original reservation")
            if (
                row.get("remaining_account_units", charged["units"] + remaining["units"])
                < remaining["units"]
            ):
                raise Refused("settlement cannot remove remaining exposure account coverage")
            for scope_name in reservations[rid]["reservation"]["scopes"]:
                scopes[scope_name]["spent"] = _plus(scopes[scope_name]["spent"], charged)
            reservations[rid].update(
                {
                    "settled": True,
                    "remaining": remaining,
                    "remaining_account_units": _number(
                        row.get("remaining_account_units", charged["units"] + remaining["units"]),
                        "remaining account units",
                    ),
                }
            )
        elif kind == "delta":
            _sources(row.get("sources"))
            for name, scope in row.get("scopes_add", {}).items():
                if not isinstance(name, str) or not name or name in scopes:
                    raise Refused("delta new scope identity already exists or is invalid")
                scopes[name] = {
                    "ceiling": _pair(scope["ceiling"], "new scope ceiling"),
                    "spent": _pair(scope["spent"], "new scope spent"),
                }
            additions = row.get("spent_additions", {})
            if not isinstance(additions, dict):
                raise Refused("delta spent additions must be a mapping")
            for scope_name, pair in additions.items():
                if scope_name not in scopes:
                    raise Refused("delta references unknown scope")
                scopes[scope_name]["spent"] = _plus(
                    scopes[scope_name]["spent"], _pair(pair, "delta spent")
                )
            additions_holds = row.get("holds_add", [])
            if not isinstance(additions_holds, list):
                raise Refused("delta holds add must be a list")
            for hold in additions_holds:
                _validate_hold(hold)
                if hold["id"] in holds:
                    raise Refused("delta duplicates hold")
                holds[hold["id"]] = dict(hold)
            external_commitments += _number(
                row.get("external_commitments_units_add", 0), "delta external commitments"
            )
            unsettled_debit_holds += _number(
                row.get("unsettled_debit_holds_units_add", 0), "delta unsettled debit holds"
            )
            replacements = row.get("holds_replace", [])
            external_set = row.get("external_commitments_units_set")
            unsettled_set = row.get("unsettled_debit_holds_units_set")
            if replacements or external_set is not None or unsettled_set is not None:
                _sources(row.get("settlement_evidence"))
                if (
                    not isinstance(row.get("settlement_attestation"), str)
                    or not row["settlement_attestation"].strip()
                ):
                    raise Refused("reduction requires settlement attestation")
            if not isinstance(replacements, list):
                raise Refused("delta holds replace must be a list")
            for replacement in replacements:
                _validate_hold(replacement)
                prior = holds.get(replacement["id"])
                if prior is None or replacement["scopes"] != prior["scopes"]:
                    raise Refused("hold replacement requires existing matching hold")
                old_pair = _pair(prior["pair"], "existing hold")
                new_pair = _pair(replacement["pair"], "replacement hold")
                if any(new_pair[key] > old_pair[key] for key in PAIR_KEYS) or _number(
                    replacement.get("account_units", new_pair["units"]), "replacement account units"
                ) > _number(
                    prior.get("account_units", old_pair["units"]), "existing account units"
                ):
                    raise Refused("hold replacement cannot enlarge hold")
                holds[replacement["id"]] = dict(replacement)
            if external_set is not None:
                candidate = _number(external_set, "external commitments set")
                if candidate > external_commitments:
                    raise Refused("external commitment set cannot enlarge")
                external_commitments = candidate
            if unsettled_set is not None:
                candidate = _number(unsettled_set, "unsettled debit hold set")
                if candidate > unsettled_debit_holds:
                    raise Refused("unsettled debit hold set cannot enlarge")
                unsettled_debit_holds = candidate
        elif kind == "reconcile":
            rid = row.get("reservation_id")
            if rid not in reservations or not reservations[rid]["settled"]:
                raise Refused("reconcile requires settled reservation")
            _sources(row.get("sources"))
            _sources(row.get("settlement_evidence"))
            if (
                not isinstance(row.get("settlement_attestation"), str)
                or not row["settlement_attestation"].strip()
            ):
                raise Refused("reconcile requires explicit payment/settlement attestation")
            amount = _number(row.get("remaining_account_units"), "reconciled unpaid debt")
            if amount < reservations[rid]["remaining"]["units"]:
                raise Refused("reconcile cannot remove remaining exposure account coverage")
            if amount > reservations[rid]["remaining_account_units"]:
                raise Refused("reconcile cannot enlarge debt")
            reservations[rid]["remaining_account_units"] = amount
        elif kind == "audit":
            if row.get("status") != "ACCEPT" or not isinstance(row.get("prefix_revision"), int):
                raise Refused("audit must be ACCEPT with prefix revision")
            if row["prefix_revision"] < 0 or row["prefix_revision"] >= row["revision"]:
                raise Refused("audit prefix revision invalid")
            if row.get("prefix_hash") != rows[row["prefix_revision"]].get("row_hash"):
                raise Refused("audit prefix hash does not close prefix")
            source = row.get("independent_source")
            if not isinstance(source, dict) or source.get("kind") in (None, "self-generated"):
                raise Refused("audit requires independent source")
            _sources([source])
            audits.append(row)
        elif kind == "status":
            if row.get("material_defect") is True:
                material_defect = True
            elif row.get("material_defect") not in (False, None):
                raise Refused("invalid material defect status")
        elif kind == "tighten":
            changes = row.get("ceilings")
            if not isinstance(changes, dict) or not changes:
                raise Refused("tighten requires ceilings")
            for name, ceiling in changes.items():
                if name not in scopes:
                    raise Refused("unknown tightened scope")
                candidate = _pair(ceiling, "tightened ceiling")
                if any(candidate[key] > scopes[name]["ceiling"][key] for key in PAIR_KEYS):
                    raise Refused("tighten cannot enlarge ceiling")
                scopes[name]["ceiling"] = candidate
        elif kind == "ceiling_raise":
            unexpected = set(row) - CEILING_RAISE_KEYS
            if unexpected:
                raise Refused(
                    "ceiling_raise may carry only ceilings, sources, attestation, owner words "
                    f"and a note; refused fields {sorted(unexpected)}"
                )
            _sources(row.get("sources"))
            for field in ("attestation", "owner_words"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise Refused(f"ceiling_raise requires written {field}")
            changes = row.get("ceilings")
            if not isinstance(changes, dict) or not changes:
                raise Refused("ceiling_raise requires ceilings")
            raised: dict[str, dict[str, float]] = {}
            for name, ceiling in changes.items():
                if name not in scopes:
                    raise Refused("ceiling_raise references unknown scope")
                candidate = _pair(ceiling, "raised ceiling")
                if any(candidate[key] < scopes[name]["ceiling"][key] for key in PAIR_KEYS):
                    raise Refused("ceiling_raise cannot lower a ceiling")
                raised[name] = candidate
            if not any(
                candidate[key] > scopes[name]["ceiling"][key]
                for name, candidate in raised.items()
                for key in PAIR_KEYS
            ):
                raise Refused("ceiling_raise must raise at least one ceiling")
            for name, candidate in raised.items():
                scopes[name]["ceiling"] = candidate
        else:
            raise Refused("unsupported register event")
    outstanding: dict[str, dict[str, Any]] = {f"hold:{key}": value for key, value in holds.items()}
    for rid, value in reservations.items():
        if value["settled"]:
            pair = value["remaining"]
            account_units = value["remaining_account_units"]
        else:
            pair = _pair(value["reservation"]["pair"], "reservation")
            account_units = _number(
                value["reservation"].get("account_units", pair["units"]), "account units"
            )
        if pair["units"] or pair["wall_minutes"] or account_units:
            outstanding[rid] = {
                **value["reservation"],
                "pair": pair,
                "account_units": account_units,
                "kind": (
                    "unsettled_debit"
                    if value["settled"] and not pair["units"] and not pair["wall_minutes"]
                    else value["reservation"]["kind"]
                ),
            }
    for scope_name, scope in scopes.items():
        held = (
            _plus(
                *(entry["pair"] for entry in outstanding.values() if scope_name in entry["scopes"])
            )
            if outstanding
            else {"units": 0.0, "wall_minutes": 0.0}
        )
        if any(scope["spent"][key] + held[key] > scope["ceiling"][key] + 1e-9 for key in PAIR_KEYS):
            raise Refused(f"scope {scope_name} is over ceiling")
    closed_prefix = max((audit["prefix_revision"] for audit in audits), default=-1)
    unaudited = [
        value["row"] for value in reservations.values() if value["row"]["revision"] > closed_prefix
    ]
    first = min((_utc(row["admitted_utc"], "admitted_utc") for row in unaudited), default=None)
    return {
        "initialized": True,
        "scopes": scopes,
        "holds": holds,
        "reservations": reservations,
        "outstanding": outstanding,
        "external_commitments_units": external_commitments,
        "unsettled_debit_holds_units": unsettled_debit_holds,
        "material_defect": material_defect,
        "first_unaudited_admission": first,
        "unaudited_gross_units": sum(
            value["reservation"]["pair"]["units"]
            for value in reservations.values()
            if value["row"]["revision"] > closed_prefix
        ),
    }


def reserve(
    rows: list[dict[str, Any]], package: dict[str, Any], reservation_id: str, now: datetime | str
) -> dict[str, Any]:
    """Create one fail-closed admission event; caller appends it under the account lock."""
    current = state(rows)
    if not current["initialized"]:
        raise Refused("register must be initialized")
    if current["material_defect"]:
        raise Refused("material defect blocks admission")
    instant = _utc(now, "now") if isinstance(now, str) else now.astimezone(timezone.utc)
    required = ("job_id", "manifest_sha256", "packet_id", "deadline_utc", "capture_utc", "operator")
    for key in required:
        if not isinstance(package.get(key), str) or not package[key]:
            raise Refused(f"package requires {key}")
    for key in ("manifest_sha256",):
        if len(package[key]) != 64:
            raise Refused("invalid package sha256")
    if not isinstance(package.get("scopes"), list) or not package["scopes"]:
        raise Refused("package requires scopes")
    if len(set(package["scopes"])) != len(package["scopes"]):
        raise Refused("duplicate package scopes")
    cap = _pair(package.get("cap"), "package cap", positive=True)
    tail = _pair(package.get("outside_tail"), "outside tail")
    required_pair = _plus(cap, tail)
    _number(package.get("live_balance_units"), "live balance")
    bindings = package.get("bindings")
    if not isinstance(bindings, (dict, list)) or not bindings:
        raise Refused("package requires bindings")
    # Accounting-audit gate: computed, recorded on the event and emitted as a
    # warning, but non-blocking (owner standing rule 2026-10-24, commit 38b075ac).
    gate_warnings: list[str] = []
    first_unaudited = current["first_unaudited_admission"]
    if first_unaudited and instant >= first_unaudited + timedelta(hours=24):
        gate_warnings.append("accounting audit overdue")
    deadline = _utc(package["deadline_utc"], "deadline")
    capture = _utc(package["capture_utc"], "capture")
    wait = _number(package.get("allocation_wait_minutes", 10), "allocation wait minutes")
    if (
        deadline <= instant
        or capture > instant
        or instant - capture > timedelta(minutes=10)
        or instant + timedelta(minutes=required_pair["wall_minutes"] + wait) > deadline
    ):
        raise Refused("expired deadline or stale live capture")
    identity = (package["job_id"], package["manifest_sha256"], package["packet_id"])
    existing = current["reservations"].get(reservation_id)
    if existing is not None:
        if existing["identity"] != identity:
            raise Refused("reservation id changed package identity")
        if existing["settled"]:
            raise Refused("settled reservation cannot refresh")
        frozen = existing["row"]["package"]
        for key in ("cap", "outside_tail", "scopes", "bindings"):
            if package.get(key) != frozen.get(key):
                raise Refused("refresh changed frozen package material")
        if deadline > _utc(frozen["deadline_utc"], "frozen deadline"):
            raise Refused("refresh cannot extend deadline")
    if existing is None and any(
        entry["kind"] == "small_a100" for entry in current["outstanding"].values()
    ):
        raise Refused("another small A100 reservation remains outstanding")
    for name in package["scopes"]:
        if name not in current["scopes"]:
            raise Refused("package references unknown scope")
        scope = current["scopes"][name]
        held = (
            _plus(
                *(
                    entry["pair"]
                    for held_id, entry in current["outstanding"].items()
                    if name in entry["scopes"] and (existing is None or held_id != reservation_id)
                )
            )
            if current["outstanding"]
            else {"units": 0.0, "wall_minutes": 0.0}
        )
        if any(
            scope["ceiling"][key] - scope["spent"][key] - held[key] + 1e-9 < required_pair[key]
            for key in PAIR_KEYS
        ):
            raise Refused(f"insufficient scope {name}")
    account_holds = sum(
        entry["account_units"]
        for held_id, entry in current["outstanding"].items()
        if existing is None or held_id != reservation_id
    )
    available = (
        package["live_balance_units"]
        - current["external_commitments_units"]
        - account_holds
        - current["unsettled_debit_holds_units"]
    )
    if available + 1e-9 < required_pair["units"]:
        raise Refused("insufficient live account balance")
    gross_after = current["unaudited_gross_units"] + (
        required_pair["units"] if existing is None else 0.0
    )
    if existing is None and gross_after > 100 + 1e-9:
        gate_warnings.append("audit required before gross reservations exceed 100 units")
    gate = {
        "blocking": False,
        "status": "WARN" if gate_warnings else "OK",
        "warnings": gate_warnings,
        "first_unaudited_admission_utc": first_unaudited.isoformat() if first_unaudited else None,
        "audit_due_utc": (
            (first_unaudited + timedelta(hours=24)).isoformat() if first_unaudited else None
        ),
        "unaudited_gross_units_after": gross_after,
        "gross_limit_units": 100,
    }
    for message in gate_warnings:
        warnings.warn(f"{message} (non-blocking)", AuditGateWarning, stacklevel=2)
    if existing is not None:
        return {
            "type": "refresh",
            "id": f"refresh:{reservation_id}:{instant.isoformat()}",
            "reservation_id": reservation_id,
            "refreshed_utc": instant.isoformat(),
            "package": dict(package),
            "accounting_audit_gate": gate,
        }
    return {
        "accounting_audit_gate": gate,
        "type": "admission",
        "id": f"admission:{reservation_id}",
        "admitted_utc": instant.isoformat(),
        "package": dict(package),
        "reservation": {
            "id": reservation_id,
            "scopes": list(package["scopes"]),
            "pair": required_pair,
            "account_units": required_pair["units"],
            "kind": "small_a100",
        },
    }


def _bound_on_disk(source: dict[str, Any]) -> None:
    path = Path(source["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Refused(f"source must be an absolute regular file: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise Refused(f"source sha256 mismatch: {path}")


def append_ceiling_raise(
    register_path: Path | str,
    lock_path: Path | str,
    record: dict[str, Any],
    expect_head: str | None = None,
) -> dict[str, Any]:
    """Append one raise-only ceiling row under the shared reservation lock."""
    if record.get("type") != "ceiling_raise":
        raise Refused("record type must be ceiling_raise")
    _sources(record.get("sources"))
    for source in record["sources"]:
        _bound_on_disk(source)
    with reservation_lock(lock_path):
        rows = read(register_path)
        if expect_head is not None and (not rows or rows[-1]["row_hash"] != expect_head):
            raise Refused("register head moved; re-read and rebuild the record")
        return append(register_path, record)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    verify = sub.add_parser("verify", help="read-only chain + semantic verification")
    verify.add_argument("--register", required=True, type=Path)
    raise_parser = sub.add_parser("ceiling-raise", help="append one raise-only ceiling row")
    raise_parser.add_argument("--register", required=True, type=Path)
    raise_parser.add_argument("--lock", required=True, type=Path)
    raise_parser.add_argument("--record", required=True, type=Path)
    raise_parser.add_argument("--expect-head", help="row_hash the register head must equal")
    args = parser.parse_args(argv)
    if args.operation == "verify":
        rows = read(args.register)
        current = state(rows)
        first = current.get("first_unaudited_admission")
        print(
            json.dumps(
                {
                    "revision": rows[-1]["revision"] if rows else None,
                    "head": rows[-1]["row_hash"] if rows else None,
                    "scopes": current.get("scopes", {}),
                    "first_unaudited_admission_utc": first.isoformat() if first else None,
                    "unaudited_gross_units": current.get("unaudited_gross_units", 0),
                },
                sort_keys=True,
                indent=1,
            )
        )
        return
    record = json.loads(args.record.read_text())
    row = append_ceiling_raise(args.register, args.lock, record, args.expect_head)
    print(json.dumps({"revision": row["revision"], "row_hash": row["row_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
