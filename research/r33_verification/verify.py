"""Outcome-blind Mac ledger, separate terminal consumer, and bounded completions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.r33_verification import inventory
from research.r33_verification.contract import ATTEMPTS
from research.r33_verification.pilot_runtime import shard_payload
from tools.ops.package_spec import (
    PackageRefused,
    digest,
    empty_ledger,
    publish_json,
    read_ledger,
    unit_id,
    verify_immutable,
)


def union(
    attempts: list[Path],
    out: Path,
    namespace: str,
    allocation: str,
    identity: dict[str, str],
    expected_ids: set[str],
) -> dict[str, Any]:
    """Verify paths/sizes/hashes only; never open a shard outcome for the ledger."""
    if len({p.resolve() for p in attempts}) != len(attempts):
        raise PackageRefused("duplicate attempt root")
    rows, seen = [], set()
    for attempt in attempts:
        for path in sorted((attempt / "shards").glob("*.json")):
            if not Path(f"{path}.complete").exists():
                continue
            try:
                index = int(path.stem.removeprefix("shard-"))
            except ValueError as exc:
                raise PackageRefused("foreign shard path") from exc
            ident = unit_id(namespace, allocation, index)
            if path.name != f"shard-{index:06d}.json" or ident not in expected_ids or ident in seen:
                raise PackageRefused("foreign or duplicate R33 shard")
            evidence = verify_immutable(path)
            seen.add(ident)
            rows.append(
                {
                    "index": index,
                    "unit_id": ident,
                    "sha256": evidence["sha256"],
                    "size": evidence["size"],
                }
            )
    ledger = empty_ledger(namespace, allocation, identity)
    ledger["units"] = sorted(rows, key=lambda row: row["index"])
    publish_json(out, ledger)
    return ledger


def decide(
    attempts: list[Path],
    ledger: Path,
    out: Path,
    namespace: str,
    allocation: str,
    identity: dict[str, str],
) -> dict[str, Any]:
    """The Mac consumer alone admits the exact union after integrity verification."""
    shards = inventory.packet()["shards"]
    rows = read_ledger(ledger, namespace, allocation, identity)
    if {row["index"] for row in rows} != set(range(len(shards))):
        raise PackageRefused("INCOMPLETE: exact frozen packet is missing shards")
    for row in rows:
        index = row["index"]
        paths = [a / "shards" / f"shard-{index:06d}.json" for a in attempts]
        paths = [p for p in paths if Path(f"{p}.complete").is_file()]
        if len(paths) != 1:
            raise PackageRefused("INVALID: missing or duplicate verified unit")
        evidence = verify_immutable(paths[0])
        if any(evidence[key] != row[key] for key in ("sha256", "size")):
            raise PackageRefused("INVALID: ledger no longer matches downloaded bytes")
        value = json.loads(paths[0].read_text())
        if value != shard_payload(namespace, allocation, identity, index, shards[index]["nodes"]):
            raise PackageRefused("INVALID: foreign identity, changed checks or false PASS")
    terminals = []
    for attempt in attempts:
        terminal_path = attempt / "terminal-receipt.json"
        if not terminal_path.exists():
            continue  # An interrupted prefix still contributes verified immutable units.
        verify_immutable(terminal_path)
        terminal = json.loads(terminal_path.read_text())
        created, skipped = terminal.get("new_indices"), terminal.get("skipped_verified_indices")
        if (
            terminal.get("schema") != "r33-attempt-terminal/v1"
            or terminal.get("status") != "ATTEMPT_COMPLETE"
            or terminal.get("identity") != identity
            or terminal.get("namespace") != namespace
            or terminal.get("allocation") != allocation
            or not isinstance(created, list)
            or not isinstance(skipped, list)
            or any(type(index) is not int for index in created + skipped)
            or len(set(created + skipped)) != len(created + skipped)
            or set(created + skipped) != set(range(len(shards)))
        ):
            raise PackageRefused("INVALID: foreign or incomplete terminal receipt")
        actual = {
            int(path.stem.removeprefix("shard-"))
            for path in (attempt / "shards").glob("shard-*.json")
            if Path(f"{path}.complete").is_file()
        }
        if actual != set(created):
            raise PackageRefused("INVALID: terminal inventory differs from immutable exports")
        terminals.append(terminal)
    if not terminals:
        raise PackageRefused("INCOMPLETE: no verified current-pin terminal receipt")
    result = {
        "schema": "r33-mac-packet-decision/v1",
        "status": "PASS_LINUX_CONTRACTS",
        "namespace": namespace,
        "allocation": allocation,
        "identity": identity,
        "verified_units": len(rows),
        "checks_passed": len(inventory.nodes()),
        "native_pending": inventory.packet()["native_pending"],
        "operational_acceptance": "HOLD_UNTIL_SUPERVISOR_RECEIPT_AND_RELEASE",
    }
    publish_json(out, result)
    return result


def completion_admission(
    previous_receipts: list[Path],
    verified_ledger: Path,
    *,
    attempts_used: int,
    series_minutes: float,
    series_units: float,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require verified interrupted children/stop; uncertainty and test failures refuse."""
    if not previous_receipts or type(attempts_used) is not int or not 1 <= attempts_used <= 2:
        raise PackageRefused("completion attempt limit or prior receipt invalid")
    if len(previous_receipts) != attempts_used or len(set(previous_receipts)) != attempts_used:
        raise PackageRefused("missing or duplicate prior attempt receipts")
    for path in previous_receipts:
        receipt = json.loads(path.read_text())
        if (
            receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error")
            or receipt.get("status") != "INCOMPLETE"
            or receipt.get("failure_class") != "CHILD_NONZERO_EXIT"
            or receipt.get("terminal", {}).get("returncode") not in {-9, -15, 124, 137, 143}
            or not receipt.get("generation")
            or receipt.get("terminal", {}).get("generation") != receipt.get("generation")
        ):
            raise PackageRefused(
                "mechanical completion requires verified stop and interrupted child"
            )
    if manifest is None:
        raise PackageRefused("frozen original manifest required")
    identity = manifest["r33"]["identity"]
    rows = read_ledger(
        verified_ledger, manifest["r33"]["namespace"], manifest["r33"]["allocation"], identity
    )
    missing = sorted(set(range(len(inventory.packet()["shards"]))) - {r["index"] for r in rows})
    if not missing:
        raise PackageRefused("no missing verification shards; completion refused")
    spent_minutes = spent_units = 0.0
    seen_jobs = set()
    for attempt_index, path in enumerate(previous_receipts):
        receipt = json.loads(path.read_text())
        prior_manifest_path = path.parent / "manifest.json"
        if not prior_manifest_path.is_file() or digest(prior_manifest_path) != receipt.get(
            "manifest_sha256"
        ):
            raise PackageRefused("receipt has no matching immutable manifest snapshot")
        prior = json.loads(prior_manifest_path.read_text())
        if (
            receipt.get("commit") != manifest["commit"]
            or prior.get("r33", {}).get("identity") != identity
            or receipt.get("job_id") in seen_jobs
            or prior.get("job_id") != receipt.get("job_id")
        ):
            raise PackageRefused("foreign or duplicate source/allocation/attempt receipt")
        if prior.get("r33", {}).get("attempt") != attempt_index:
            raise PackageRefused("noncontiguous or foreign completion attempt number")
        envelope = ATTEMPTS[attempt_index]
        if (
            prior.get("timeout_minutes") != envelope["effective_minutes"]
            or prior.get("budget", {}).get("unit_cap") != envelope["unit_cap"]
            or prior.get("budget", {}).get("rate_units_per_hour") != 0.08
        ):
            raise PackageRefused("per-attempt frozen envelope changed")
        seen_jobs.add(receipt["job_id"])
        seconds = receipt.get("elapsed_seconds")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise PackageRefused("unknown charged attempt elapsed time")
        if (
            seconds > envelope["effective_minutes"] * 60
            or seconds * 0.08 / 3600 > envelope["unit_cap"] + 1e-12
        ):
            raise PackageRefused("per-attempt hard cap exceeded")
        accounting = receipt.get("accounting") or {}
        if accounting.get("priced_rate_units_per_hour") != 0.08:
            raise PackageRefused("unknown or changed CPU rate; full reforecast required")
        ledger_rows = {row["index"]: row for row in rows}
        for entry in receipt.get("exports", []):
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise PackageRefused("foreign prior export path")
            if relative.parent == Path("shards") and relative.suffix == ".json":
                index = int(relative.stem.removeprefix("shard-"))
                evidence = verify_immutable(path.parent / "exports" / relative)
                row = ledger_rows.get(index)
                if row is None or any(
                    evidence[key] != entry[key] or row[key] != entry[key]
                    for key in ("sha256", "size")
                ):
                    raise PackageRefused("verified prior unit omitted from resume ledger")
        spent_minutes += seconds / 60
        spent_units += seconds * 0.08 / 3600
    # Both receipt-derived charges and declared charges must agree; no NaN, refund,
    # omitted attempt or cap redistribution can buy another completion.
    if (
        not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (series_minutes, series_units)
        )
        or abs(series_minutes - spent_minutes) > 1e-6
        or abs(series_units - spent_units) > 1e-9
        or spent_minutes + 24 > 288
        or spent_units + 0.032 > 0.384
    ):
        raise PackageRefused("series accounting or remaining frozen liability differs")
    return {
        "status": "MECHANICAL_COMPLETION_ELIGIBLE",
        "missing_indices": missing,
        "operator_owned": True,
        "attempt": attempts_used,
        "spent_minutes": spent_minutes,
        "spent_units": spent_units,
        "effective_minutes": 24,
        "unit_cap": 0.032,
    }


def accept_supervisor(
    manifest: dict[str, Any],
    receipt: Path,
    decision: Path,
    out: Path,
    release_evidence: Path,
    prior_receipts: list[Path] | None = None,
) -> dict[str, Any]:
    """Require the real broker contract, verified local exports, accounting and release."""
    value = json.loads(receipt.read_text())
    verify_immutable(decision)
    outcome = json.loads(decision.read_text())
    verify_immutable(release_evidence)
    released = json.loads(release_evidence.read_text())
    snapshot = receipt.parent / "manifest.json"
    if not snapshot.is_file() or digest(snapshot) != value.get("manifest_sha256"):
        raise PackageRefused("HOLD: missing hash-bound supervisor manifest")
    actual = json.loads(snapshot.read_text())
    for key in ("job_id", "commit", "script", "env", "r33", "timeout_minutes"):
        if actual.get(key) != manifest.get(key):
            raise PackageRefused("INVALID: broker manifest changed a frozen obligation")
    terminal = value.get("terminal") or {}
    accounting = value.get("accounting") or {}
    endpoint = value.get("endpoint")
    rate = accounting.get("observed_rate_units_per_hour")
    if (
        value.get("status") != "PASS"
        or value.get("stop_verified") is not True
        or value.get("cleanup_error")
        or value.get("error")
        or value.get("commit") != manifest["commit"]
        or value.get("job_id") != manifest["job_id"]
        or not value.get("session")
        or not value.get("generation")
        or not endpoint
        or terminal.get("generation") != value["generation"]
        or terminal.get("returncode") != 0
        or terminal.get("status") not in {"completed", "success", "succeeded", "PASS"}
        or accounting.get("allocated_endpoint") != endpoint
        or accounting.get("priced_rate_units_per_hour") != 0.08
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not 0 < rate <= 0.08
        or accounting.get("unaccounted_endpoints") != []
        or accounting.get("calibration", {}).get("status") != "PASS"
        or not isinstance(accounting.get("last_crosscheck"), dict)
        or outcome.get("status") != "PASS_LINUX_CONTRACTS"
        or outcome.get("identity") != manifest["r33"]["identity"]
        or outcome.get("verified_units") != len(inventory.packet()["shards"])
        or released.get("schema") != "r33-release-observation/v1"
        or released.get("endpoint") != endpoint
        or released.get("supervisor_receipt_sha256") != digest(receipt)
        or not isinstance(released.get("sessions"), dict)
        or endpoint in released["sessions"]
    ):
        raise PackageRefused("HOLD: terminal, accounting or assignment-release proof incomplete")
    entries = value.get("exports")
    if not isinstance(entries, list) or not entries:
        raise PackageRefused("INVALID: missing broker export inventory")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PackageRefused("INVALID: unsafe export inventory path")
        evidence = verify_immutable(receipt.parent / "exports" / relative)
        if any(evidence[key] != entry[key] for key in ("size", "sha256")):
            raise PackageRefused("INVALID: broker export identity differs")
    if "terminal-receipt.json" not in {e["path"] for e in entries}:
        raise PackageRefused("INCOMPLETE: terminal export absent")
    history = list(prior_receipts or [])
    if len(history) != manifest["r33"]["attempt"] or [digest(path) for path in history] != manifest[
        "r33"
    ].get("prior_receipt_sha256", []):
        raise PackageRefused("HOLD: incomplete prior charge history")
    records = [json.loads(path.read_text()) for path in history] + [value]
    seconds = 0.0
    for index, record in enumerate(records):
        elapsed = record.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or not 0 <= elapsed <= ATTEMPTS[index]["effective_minutes"] * 60
            or elapsed * 0.08 / 3600 > ATTEMPTS[index]["unit_cap"] + 1e-12
        ):
            raise PackageRefused("INVALID: unknown elapsed or per-attempt cap exceeded")
        seconds += elapsed
    if seconds / 60 > 288 or seconds * 0.08 / 3600 > 0.384:
        raise PackageRefused("INVALID: pilot series hard cap exceeded")
    result = {
        "schema": "r33-pilot-acceptance/v1",
        "status": "PASS",
        "supervisor_receipt_sha256": digest(receipt),
        "decision_sha256": digest(decision),
        "release_evidence_sha256": digest(release_evidence),
        "identity": manifest["r33"]["identity"],
        "units_charged_at_ceiling": seconds * 0.08 / 3600,
        "allocated_minutes": seconds / 60,
        "native_pending": outcome["native_pending"],
    }
    publish_json(out, result)
    return result


def observe_release(receipt: Path, out: Path, colab: str) -> dict[str, Any]:
    """Read-only, current account observation; never allocate, stop, or authenticate."""
    import re
    import subprocess
    import time

    value = json.loads(receipt.read_text())
    result = subprocess.run(
        [colab, "--auth", "oauth2", "sessions"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = re.findall(r"^\[([^\]]+)\]\s+(\S+)\s+\| Hardware:", result.stdout, re.M)
    if not rows and "No active sessions found on server." not in result.stdout:
        raise PackageRefused("HOLD: unrecognized account assignment output")
    sessions = {endpoint: name for name, endpoint in rows}
    if not value.get("endpoint") or value["endpoint"] in sessions:
        raise PackageRefused("HOLD: expected released endpoint is still listed")
    evidence = {
        "schema": "r33-release-observation/v1",
        "endpoint": value["endpoint"],
        "sessions": sessions,
        "observed_epoch": time.time(),
        "supervisor_receipt_sha256": digest(receipt),
        "raw_sessions_sha256": __import__("hashlib").sha256(result.stdout.encode()).hexdigest(),
    }
    publish_json(out, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    ledger = actions.add_parser("ledger")
    ledger.add_argument("--manifest", type=Path, required=True)
    ledger.add_argument("--attempt", type=Path, action="append", required=True)
    ledger.add_argument("--out", type=Path, required=True)
    ledger.add_argument("--decision", type=Path)
    observe = actions.add_parser("observe-release")
    observe.add_argument("--receipt", type=Path, required=True)
    observe.add_argument("--out", type=Path, required=True)
    observe.add_argument("--colab", default="/Users/josenunez/.local/bin/colab")
    accept = actions.add_parser("accept")
    for name in ("manifest", "receipt", "decision", "release-evidence", "out"):
        accept.add_argument("--" + name, type=Path, required=True)
    accept.add_argument("--prior-receipt", type=Path, action="append")
    args = parser.parse_args()
    if args.action == "observe-release":
        result = observe_release(args.receipt, args.out, args.colab)
    elif args.action == "accept":
        result = accept_supervisor(
            json.loads(args.manifest.read_text()),
            args.receipt,
            args.decision,
            args.out,
            args.release_evidence,
            args.prior_receipt,
        )
    else:
        manifest = json.loads(args.manifest.read_text())
        contract = manifest["r33"]
        expected = {
            unit_id(contract["namespace"], contract["allocation"], shard["index"])
            for shard in inventory.packet()["shards"]
        }
        result = union(
            args.attempt,
            args.out,
            contract["namespace"],
            contract["allocation"],
            contract["identity"],
            expected,
        )
        if args.decision:
            result = decide(
                args.attempt,
                args.out,
                args.decision,
                contract["namespace"],
                contract["allocation"],
                contract["identity"],
            )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
