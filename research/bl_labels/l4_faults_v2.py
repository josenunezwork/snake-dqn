"""``l4-label-faults/v2``: the prospective L4/CPU label-lane fault definition.

Authority: ``bl-labels-f0/JUDGMENT_L4_FAULT_DEFINITION_20261024.md`` sha256
``fe8579c2aec5489ea25149b5d88036cb22daad0faa5310913231f3388716728e`` §2.

Pure, hash-only and outcome-blind: this module reads supervisor call/transfer
records, terminal fields, optional runtime RPC marker counts and optional unit
hash evidence. It never opens label payloads. It is shared by the
rpc_qualification checker, the F1-bis analyzer and the L4 production importer.

Counted classes (any one fails the session):

C1 bound failure: elapsed > timeout + 2.0 s, or no recorded positive bound.
C2 unrecovered request: a failed call not resolved within <= 4 bounded attempts
   and <= 150 s from its first failure (exec: under the same request ID with an
   identical request sha256; idempotent reads: a later completed read).
C3 supervisor termination: any non-PASS terminal, unverified stop, cleanup
   error or terminal failure class.
C4 execution-identity breach: re-issue with a different code hash; marker counts
   (started != 1 when executed, done > 1, done without started).
C5 unit integrity breach: duplicate commit, missing .complete/.sha256, Mac vs
   runtime sha mismatch, ledger-close mismatch, restore/bare-audit mismatch,
   broken worker invariance.
C6 unverified transfer: an upload/download that does not end with a verified
   match; the pre-runtime heartbeat PUT is the only carve-out.

Recovered (resolved, C4-clean) failures are *recorded, not counted*. Injected
faults are excluded only by identity in a create-only, pre-dispatch injection
manifest; anything unlisted is real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA = "l4-label-faults/v2"
INJECTION_SCHEMA = "l4-label-faults-injection-manifest/v1"
AUTHORITY_SHA256 = "fe8579c2aec5489ea25149b5d88036cb22daad0faa5310913231f3388716728e"
BOUND_SLACK_SECONDS = 2.0
RECOVERY_MAX_ATTEMPTS = 4
RECOVERY_MAX_SECONDS = 150.0
READ_COMMANDS = frozenset({"version", "whoami", "sessions", "usage", "url"})
TRANSFER_COMMANDS = frozenset({"upload", "download"})
# Supervisor-only records written by the recovery path (not CLI subprocesses).
RECOVERY_COMMANDS = frozenset({"kernel_idle", "recovery"})
HEARTBEAT_SUFFIX = "/lease-heartbeat.json"
NON_TRANSPORT_TERMINALS = frozenset({"CHILD_NONZERO_EXIT"})


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _fault(cls: str, detail: str, index: int | None = None, record: Any = None) -> dict[str, Any]:
    row: dict[str, Any] = {"class": cls, "detail": detail}
    if index is not None:
        row["index"] = index
    if record is not None:
        row["record"] = record
    return row


def load_injection_manifest(path: Path, *, job_id: str | None = None) -> dict[str, Any]:
    """Load a create-only injection manifest with its sha256 sidecar."""
    raw = path.read_bytes()
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.is_file():
        recorded = sidecar.read_text().split()[0]
        if recorded != hashlib.sha256(raw).hexdigest():
            raise ValueError("injection manifest differs from its sha256 sidecar")
    manifest = json.loads(raw)
    validate_injection_manifest(manifest, job_id=job_id)
    return manifest


def validate_injection_manifest(
    manifest: Any, *, job_id: str | None = None, allow_retroactive: bool = False
) -> None:
    """``allow_retroactive`` exists only for the §2.4 sanity fixture of attempt 1."""
    if not isinstance(manifest, dict) or manifest.get("schema") != INJECTION_SCHEMA:
        raise ValueError("foreign injection manifest")
    if manifest.get("written_before_dispatch") is not True and not (
        allow_retroactive and manifest.get("provenance") == "retroactive-sanity-fixture"
    ):
        raise ValueError("injection manifest is not a pre-dispatch record")
    if job_id is not None and manifest.get("session_job_id") != job_id:
        raise ValueError("injection manifest names a different session")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("injection manifest lacks entries")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("malformed injection entry")
        ident = entry.get("injection_id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("injection entries need unique IDs")
        seen.add(ident)
        if entry.get("command") not in {"exec", "upload", "download", "terminal"}:
            raise ValueError("injection entry names an unsupported command")
        if entry["command"] == "terminal":
            if not (
                isinstance(entry.get("failure_class"), str)
                and isinstance(entry.get("error_contains"), str)
                and entry["error_contains"]
                and entry.get("max_calls", 1) == 1
            ):
                raise ValueError("terminal injections need failure_class + error_contains")
            if not isinstance(entry.get("type"), str) or not entry["type"]:
                raise ValueError("injection entry lacks its type")
            continue
        if not isinstance(entry.get("type"), str) or not entry["type"]:
            raise ValueError("injection entry lacks its type")
        limit = entry.get("max_calls", 1)
        if type(limit) is not int or limit < 1:
            raise ValueError("injection entry max_calls must be a positive integer")
        if entry["command"] == "exec" and not isinstance(entry.get("request_id"), str):
            raise ValueError("exec injections are identified by their pre-dispatch request ID")
        if entry["command"] != "exec" and not isinstance(entry.get("remote_suffix"), str):
            raise ValueError("transfer injections are identified by remote suffix")
        if "call_index" in entry and (
            type(entry["call_index"]) is not int or entry["call_index"] < 0
        ):
            raise ValueError("injection call_index must be a nonnegative integer")


def _transfer_remotes(receipt: dict[str, Any], calls: list[dict[str, Any]]) -> dict[int, str]:
    """Attribute each upload/download call to its remote path.

    The fix-pin supervisor records ``remote`` on the call itself. Older receipts
    are paired per direction with ``contents_transfers`` (the i-th upload call
    with the i-th upload row); a pairing whose statuses disagree is unreliable
    (the heartbeat thread appends concurrently), and that call stays
    unattributed, which can only make the verdict stricter.
    """
    mapping: dict[int, str] = {}
    rows = receipt.get("contents_transfers")
    rows = rows if isinstance(rows, list) else []
    for direction in TRANSFER_COMMANDS:
        wanted = [i for i, call in enumerate(calls) if call.get("command") == direction]
        matching = [r for r in rows if isinstance(r, dict) and r.get("direction") == direction]
        paired = len(wanted) == len(matching)
        for position, index in enumerate(wanted):
            own = calls[index].get("remote")
            if isinstance(own, str) and own:
                mapping[index] = own
                continue
            if not paired:
                continue
            row = matching[position]
            remote = row.get("remote")
            if isinstance(remote, str) and row.get("status") == calls[index].get("status"):
                mapping[index] = remote
    return mapping


def _match_injection(
    index: int,
    call: dict[str, Any],
    remote: str | None,
    entries: list[dict[str, Any]],
    used: dict[str, list[int]],
) -> dict[str, Any] | None:
    for entry in entries:
        if entry["command"] == "terminal" or call.get("command") != entry["command"]:
            continue
        if "call_index" in entry and entry["call_index"] != index:
            continue
        if entry["command"] == "exec":
            if call.get("request_id") != entry["request_id"]:
                continue
            if (
                entry.get("request_sha256")
                and call.get("request_sha256") != entry["request_sha256"]
            ):
                continue
        elif remote is None or not remote.endswith(entry["remote_suffix"]):
            continue
        if len(used.setdefault(entry["injection_id"], [])) >= entry.get("max_calls", 1):
            continue
        used[entry["injection_id"]].append(index)
        return entry
    return None


def _heartbeat_carveout(index: int, calls: list[dict[str, Any]], remotes: dict[int, str]) -> bool:
    """The ruled pre-runtime heartbeat PUT: the first heartbeat, before any staged input."""
    if calls[index].get("command") != "upload" or not remotes.get(index, "").endswith(
        HEARTBEAT_SUFFIX
    ):
        return False
    for earlier in range(index):
        if calls[earlier].get("command") not in TRANSFER_COMMANDS:
            continue
        if calls[earlier].get("command") != "upload" or earlier not in remotes:
            return False
        remote = remotes[earlier]
        if remote.endswith(HEARTBEAT_SUFFIX):
            return False  # not the first heartbeat
        if not remote.endswith("-release.json"):
            return False  # a staged input already moved: not pre-runtime
    # It must have returned inside its own bound (a failure, not a stall).
    elapsed, timeout = _finite(calls[index].get("elapsed_seconds")), _finite(
        calls[index].get("timeout_seconds")
    )
    return elapsed is not None and timeout is not None and elapsed < timeout


def _verified_transfer(receipt: dict[str, Any], remote: str, direction: str) -> bool:
    for row in receipt.get("transfers") or []:
        if (
            isinstance(row, dict)
            and row.get("remote") == remote
            and row.get("direction") == direction
            and row.get("status") == "verified"
        ):
            return True
    return False


def _exec_episodes(
    calls: list[dict[str, Any]],
    excluded: set[int],
    counted: list[dict[str, Any]],
    recovered: list[dict[str, Any]],
) -> None:
    """Resolve every failed exec by its durable request identity (C2/C4)."""
    summaries = {
        call.get("request_id"): (index, call)
        for index, call in enumerate(calls)
        if call.get("command") == "recovery" and isinstance(call.get("request_id"), str)
    }
    hashes: dict[str, set[str]] = {}
    for index, call in enumerate(calls):
        if call.get("command") == "exec" and isinstance(call.get("request_id"), str):
            hashes.setdefault(call["request_id"], set()).add(str(call.get("request_sha256")))
    for request_id, values in hashes.items():
        if len(values) != 1 or "None" in values:
            counted.append(
                _fault("C4", f"request {request_id} re-issued with differing/missing code hash")
            )
    index = 0
    while index < len(calls):
        call = calls[index]
        if call.get("command") != "exec" or call.get("status") == "completed" or index in excluded:
            index += 1
            continue
        request_id = call.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            counted.append(_fault("C2", "failed exec without a durable request ID", index, call))
            index += 1
            continue
        attempts, seconds, resolved_at = 0, 0.0, None
        for later in range(index + 1, len(calls)):
            other = calls[later]
            seconds += _finite(other.get("elapsed_seconds")) or 0.0
            if other.get("command") == "exec" and other.get("request_id") == request_id:
                attempts += 1
                if other.get("status") == "completed":
                    resolved_at = later
                    break
            elif other.get("command") == "kernel_idle" and other.get("request_id") == request_id:
                continue  # probes are part of an attempt; the summary record counts them
        source = "derived"
        summary = summaries.get(request_id)
        if summary is not None and summary[0] > index:
            source = "recovery-record"
            attempts = summary[1].get("attempts")
            seconds = _finite(summary[1].get("recovery_seconds"))
            if summary[1].get("status") != "completed":
                resolved_at = None
        if (
            resolved_at is None
            or type(attempts) is not int
            or seconds is None
            or attempts > RECOVERY_MAX_ATTEMPTS
            or seconds > RECOVERY_MAX_SECONDS
        ):
            counted.append(
                _fault(
                    "C2",
                    f"request {request_id} unresolved within {RECOVERY_MAX_ATTEMPTS} attempts/"
                    f"{RECOVERY_MAX_SECONDS:.0f} s (attempts={attempts}, seconds={seconds}, "
                    f"source={source})",
                    index,
                    call,
                )
            )
            # Skip later failures of the same unresolved episode: one C2 per request.
            index = len(calls) if resolved_at is None else resolved_at + 1
            continue
        recovered.append(
            {
                "kind": "exec",
                "request_id": request_id,
                "first_failure_index": index,
                "resolved_index": resolved_at,
                "attempts": attempts,
                "seconds": seconds,
                "source": source,
            }
        )
        index = resolved_at + 1


def _marker_faults(markers: dict[str, Any], counted: list[dict[str, Any]]) -> None:
    for request_id, value in sorted(markers.items()):
        if not isinstance(value, dict):
            counted.append(_fault("C4", f"malformed marker evidence for {request_id}"))
            continue
        started, done = value.get("started"), value.get("done")
        if type(started) is not int or type(done) is not int:
            counted.append(_fault("C4", f"marker counts missing for {request_id}"))
            continue
        hashes = value.get("code_sha256", [])
        if done > 1 or (done and started != 1) or started > 1:
            counted.append(_fault("C4", f"{request_id}: started={started} done={done}"))
        elif isinstance(hashes, list) and len(set(hashes)) > 1:
            counted.append(_fault("C4", f"{request_id}: marker code hash differs"))


def unit_integrity_faults(units: dict[str, Any]) -> list[dict[str, Any]]:
    """C5 from hash-only unit evidence (never payload contents)."""
    faults: list[dict[str, Any]] = []
    committed = units.get("committed", [])
    seen: dict[str, int] = {}
    for row in committed:
        unit = row.get("unit_id")
        seen[unit] = seen.get(unit, 0) + 1
        if row.get("complete") is not True or row.get("sha256_sidecar_ok") is not True:
            faults.append(_fault("C5", f"unit {unit} lacks create-only .complete/.sha256"))
        mac, runtime = row.get("mac_sha256"), row.get("runtime_sha256")
        if mac is not None or runtime is not None:
            if not (isinstance(mac, str) and mac == runtime):
                faults.append(_fault("C5", f"unit {unit} Mac sha256 differs from runtime sha256"))
    for unit, count in sorted(seen.items()):
        if count > 1:
            faults.append(_fault("C5", f"unit {unit} committed {count} times"))
    if "allocated" in units:
        allocated = set(units["allocated"])
        closed = set(seen) | set(units.get("pending", []))
        if allocated != closed or set(seen) & set(units.get("pending", [])):
            faults.append(_fault("C5", "ledger close: allocated != committed ∪ declared-pending"))
    integrity = units.get("integrity", {})
    for key in ("restore_digest_mismatches", "bare_audit_mismatches"):
        if integrity.get(key, 0) != 0:
            faults.append(_fault("C5", f"{key}={integrity.get(key)}"))
    if integrity.get("worker_invariance", True) is not True:
        faults.append(_fault("C5", "worker invariance broken"))
    return faults


def classify_session(
    receipt: dict[str, Any],
    injection_manifest: dict[str, Any] | None = None,
    *,
    markers: dict[str, Any] | None = None,
    units: dict[str, Any] | None = None,
    allow_retroactive_manifest: bool = False,
) -> dict[str, Any]:
    """Classify one supervisor receipt under v2. Raises ValueError if malformed."""
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    calls = receipt.get("cli_calls")
    if not isinstance(calls, list) or not calls or not all(isinstance(c, dict) for c in calls):
        raise ValueError("v2 needs the complete supervisor call census")
    entries: list[dict[str, Any]] = []
    if injection_manifest is not None:
        validate_injection_manifest(
            injection_manifest,
            job_id=receipt.get("job_id"),
            allow_retroactive=allow_retroactive_manifest,
        )
        entries = injection_manifest["entries"]
    counted: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    carve_outs: list[dict[str, Any]] = []
    used: dict[str, list[int]] = {}
    remotes = _transfer_remotes(receipt, calls)
    excluded_indices: set[int] = set()

    # C1 applies to every call, injected or not: bounds must hold even for injections.
    for index, call in enumerate(calls):
        if call.get("status") not in {"completed", "failed"}:
            raise ValueError(f"call {index} has an unknown status")
        elapsed, timeout = _finite(call.get("elapsed_seconds")), _finite(
            call.get("timeout_seconds")
        )
        if elapsed is None or elapsed < 0:
            raise ValueError(f"call {index} lacks a finite elapsed time")
        if timeout is None or timeout <= 0:
            counted.append(_fault("C1", "call has no recorded bound", index, call))
        elif elapsed > timeout + BOUND_SLACK_SECONDS:
            counted.append(
                _fault("C1", f"elapsed {elapsed:.3f} > bound {timeout:.3f} + 2", index, call)
            )

    # Identity-based injection exclusion (failed calls only).
    for index, call in enumerate(calls):
        if call.get("status") == "completed" or call.get("command") in RECOVERY_COMMANDS:
            continue
        remote = remotes.get(index) if remotes else None
        entry = _match_injection(index, call, remote, entries, used)
        if entry is not None:
            excluded_indices.add(index)
            excluded.append(
                {"index": index, "injection_id": entry["injection_id"], "type": entry["type"]}
            )

    _exec_episodes(calls, excluded_indices, counted, recovered)

    for index, call in enumerate(calls):
        command = call.get("command")
        if call.get("status") == "completed" or index in excluded_indices or command == "exec":
            continue
        if command in RECOVERY_COMMANDS:
            continue  # resolved/unresolved through the exec episode and its summary
        if command in READ_COMMANDS:
            attempts, seconds, ok = 0, 0.0, None
            for later in range(index + 1, len(calls)):
                seconds += _finite(calls[later].get("elapsed_seconds")) or 0.0
                if calls[later].get("command") == command:
                    attempts += 1
                    if calls[later].get("status") == "completed":
                        ok = later
                        break
            if ok is None or attempts > RECOVERY_MAX_ATTEMPTS or seconds > RECOVERY_MAX_SECONDS:
                counted.append(_fault("C2", f"idempotent {command} read unresolved", index, call))
            else:
                recovered.append(
                    {
                        "kind": command,
                        "first_failure_index": index,
                        "resolved_index": ok,
                        "attempts": attempts,
                        "seconds": seconds,
                        "source": "derived",
                    }
                )
            continue
        if command == "stop":
            if receipt.get("stop_verified") is True:
                recovered.append(
                    {
                        "kind": "stop",
                        "first_failure_index": index,
                        "resolved_index": None,
                        "attempts": 1,
                        "seconds": 0.0,
                        "source": "stop_verified",
                    }
                )
            else:
                counted.append(
                    _fault("C2", "stop failed and release was not verified", index, call)
                )
            continue
        if command in TRANSFER_COMMANDS:
            if _heartbeat_carveout(index, calls, remotes):
                carve_outs.append({"index": index, "rule": "pre-runtime heartbeat PUT"})
                continue
            remote = remotes.get(index) if remotes else None
            if remote is not None and _verified_transfer(receipt, remote, command):
                recovered.append(
                    {
                        "kind": command,
                        "first_failure_index": index,
                        "resolved_index": None,
                        "attempts": 1,
                        "seconds": 0.0,
                        "source": "later verified transfer",
                        "remote": remote,
                    }
                )
            else:
                detail = "unverified transfer" + (f" {remote}" if remote else " (unattributable)")
                counted.append(_fault("C6", detail, index, call))
            continue
        counted.append(_fault("C2", f"failed {command} call has no recovery rule", index, call))

    for index, row in enumerate(receipt.get("transfers") or []):
        if not isinstance(row, dict):
            raise ValueError("malformed transfers row")
        if row.get("status") != "verified" and not _verified_transfer(
            receipt, str(row.get("remote")), str(row.get("direction"))
        ):
            entry = next(
                (
                    e
                    for e in entries
                    if e["command"] in TRANSFER_COMMANDS
                    and e["command"] == row.get("direction")
                    and isinstance(row.get("remote"), str)
                    and row["remote"].endswith(e["remote_suffix"])
                    and len(used.setdefault(e["injection_id"] + "#transfers", []))
                    < e.get("max_calls", 1)
                ),
                None,
            )
            if entry is not None:
                used[entry["injection_id"] + "#transfers"].append(index)
                used.setdefault(entry["injection_id"], [])
                excluded.append(
                    {
                        "transfers_index": index,
                        "injection_id": entry["injection_id"],
                        "type": entry["type"],
                    }
                )
                continue
            counted.append(
                _fault(
                    "C6",
                    f"transfer {row.get('remote')} not verified",
                    None,
                    {"transfers_index": index, "status": row.get("status")},
                )
            )

    terminal_class = receipt.get("failure_class")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or terminal_class
    ):
        kind = "NON_TRANSPORT" if terminal_class in NON_TRANSPORT_TERMINALS else "TRANSPORT"
        designed = next(
            (
                e
                for e in entries
                if e["command"] == "terminal"
                and receipt.get("status") != "PASS"
                and terminal_class == e["failure_class"]
                and e["error_contains"] in str(receipt.get("error"))
                # A designed release never excuses an unverified stop or cleanup error.
                and receipt.get("stop_verified") is True and receipt.get("cleanup_error") is None
            ),
            None,
        )
        if designed is not None:
            used.setdefault(designed["injection_id"], []).append(-1)
            excluded.append(
                {
                    "terminal": True,
                    "injection_id": designed["injection_id"],
                    "type": designed["type"],
                }
            )
        else:
            counted.append(
                _fault(
                    "C3",
                    f"terminal status={receipt.get('status')} failure_class={terminal_class} "
                    f"stop_verified={receipt.get('stop_verified')} cleanup_error="
                    f"{receipt.get('cleanup_error')} ({kind})",
                    None,
                    {"error": receipt.get("error"), "terminal_kind": kind},
                )
            )

    if markers is not None:
        _marker_faults(markers, counted)
    if units is not None:
        counted.extend(unit_integrity_faults(units))

    unrealized = [
        e["injection_id"]
        for e in entries
        if not used.get(e["injection_id"]) and not used.get(e["injection_id"] + "#transfers")
    ]
    return {
        "schema": SCHEMA,
        "authority_sha256": AUTHORITY_SHA256,
        "job_id": receipt.get("job_id"),
        "verdict": "FAIL" if counted else "PASS",
        "counted": counted,
        "counted_by_class": {
            cls: sum(1 for row in counted if row["class"] == cls)
            for cls in ("C1", "C2", "C3", "C4", "C5", "C6")
        },
        "recovered": recovered,
        "recovered_events": len(recovered),
        "recovered_seconds": round(sum(row["seconds"] for row in recovered), 6),
        "max_attempts": max((row["attempts"] for row in recovered), default=0),
        "excluded_injections": excluded,
        "unrealized_injections": unrealized,
        "carve_outs": carve_outs,
        "markers_checked": markers is not None,
        "units_checked": units is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--injection-manifest", type=Path)
    args = parser.parse_args(argv)
    receipt = json.loads(args.receipt.read_text())
    manifest = (
        load_injection_manifest(args.injection_manifest, job_id=receipt.get("job_id"))
        if args.injection_manifest
        else None
    )
    result = classify_session(receipt, manifest)
    json.dump(result, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
