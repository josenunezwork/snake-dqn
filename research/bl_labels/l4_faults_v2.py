"""``l4-label-faults/v2``: the prospective L4/CPU label-lane fault definition.

Authority: ``bl-labels-f0/JUDGMENT_L4_FAULT_DEFINITION_20261024.md`` sha256
``fe8579c2aec5489ea25149b5d88036cb22daad0faa5310913231f3388716728e`` §2.

Pure, hash-only and outcome-blind: this module reads supervisor call/transfer
records, terminal fields, mandatory prospective RPC marker counts and optional unit
hash evidence. It never opens label payloads. It is shared by the
rpc_qualification checker, the F1-bis analyzer and the L4 production importer.

Counted classes (any one fails the session):

C1 bound failure: elapsed > timeout + 2.0 s, or no recorded positive bound.
C2 unrecovered request: a failed call not resolved within <= 4 bounded attempts
   and <= 150 s from its first failure (exec: under the same request ID with an
   identical request sha256; reads: the same logical operation and durable reply).
C3 supervisor termination: covered transport/supervisor termination, unverified
   stop or cleanup error. A child-only nonzero exit fails outside this class.
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
TRANSPORT_TERMINALS = frozenset(
    {
        "FOREIGN_GENERATION",
        "CONTROL_RPC_TIMEOUT",
        "CONTROL_RPC_ERROR",
        "RPC_EXEC_POLL_TIMEOUT",
        "UPLOAD_CAP",
        "ACCOUNTING",
        "SUPERVISOR",
    }
)


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
            if not allow_retroactive:
                for key in ("watchdog_event_id", "watchdog_event_sha256"):
                    if not isinstance(entry.get(key), str) or not entry[key]:
                        raise ValueError(f"terminal injection lacks {key}")
            continue
        if not isinstance(entry.get("type"), str) or not entry["type"]:
            raise ValueError("injection entry lacks its type")
        limit = entry.get("max_calls", 1)
        if type(limit) is not int or limit < 1:
            raise ValueError("injection entry max_calls must be a positive integer")
        if entry["command"] == "exec":
            if not isinstance(entry.get("request_id"), str):
                raise ValueError("exec injections are identified by their pre-dispatch request ID")
            if not allow_retroactive:
                for key in ("request_sha256", "failure_type"):
                    if not isinstance(entry.get(key), str) or not entry[key]:
                        raise ValueError(f"exec injection lacks full {key} identity")
        if entry["command"] != "exec":
            if allow_retroactive:
                if not isinstance(entry.get("remote_suffix"), str):
                    raise ValueError("legacy transfer injections need remote suffix")
            else:
                for key in ("remote", "operation_id", "attempt_id", "failure_type"):
                    if not isinstance(entry.get(key), str) or not entry[key]:
                        raise ValueError(f"transfer injection lacks full {key} identity")
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
    *,
    allow_retroactive: bool,
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
            if not allow_retroactive and call.get("failure_type") != entry["failure_type"]:
                continue
        elif allow_retroactive:
            if remote is None or not remote.endswith(entry["remote_suffix"]):
                continue
        elif not (
            remote == entry["remote"]
            and call.get("operation_id") == entry["operation_id"]
            and call.get("attempt_id") == entry["attempt_id"]
            and call.get("failure_type") == entry["failure_type"]
        ):
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


def _verified_transfer(
    receipt: dict[str, Any], calls: list[dict[str, Any]], index: int, remote: str, direction: str
) -> bool:
    """Require a verification tied to a later, matching logical transfer.

    A path is not an identity: repeated uploads of the same path may contain
    different bytes.  The receipt therefore has to bind the failed attempt to
    a later completed attempt and bind that completed attempt to its hash
    verification record.
    """
    call = calls[index]
    operation_id, digest = call.get("operation_id"), call.get("sha256")
    started = _finite(call.get("finished_monotonic"))
    if (
        not all(isinstance(value, str) and value for value in (operation_id, digest))
        or started is None
    ):
        return False
    attempts = 0
    for later in range(index + 1, len(calls)):
        candidate = calls[later]
        if candidate.get("command") == direction and candidate.get("operation_id") == operation_id:
            attempts += 1
        if not (
            candidate.get("command") == direction
            and candidate.get("status") == "completed"
            and candidate.get("operation_id") == operation_id
            and candidate.get("sha256") == digest
            and _transfer_remotes(receipt, calls).get(later) == remote
        ):
            continue
        finished = _finite(candidate.get("finished_monotonic"))
        if (
            finished is None
            or finished < started
            or not (1 <= attempts <= RECOVERY_MAX_ATTEMPTS)
            or finished - started > RECOVERY_MAX_SECONDS
        ):
            continue
        for row in receipt.get("transfers") or []:
            if (
                isinstance(row, dict)
                and row.get("status") == "verified"
                and row.get("remote") == remote
                and row.get("direction") == direction
                and row.get("operation_id") == operation_id
                and row.get("sha256") == digest
                and row.get("verified_call_index") == later
            ):
                return True
    return False


def _completed_transfer_verified(
    receipt: dict[str, Any], calls: list[dict[str, Any]], index: int, remote: str
) -> bool:
    """Every completed transfer needs its own hash-verification correlation."""
    call = calls[index]
    if not all(
        isinstance(value, str) and value for value in (call.get("operation_id"), call.get("sha256"))
    ):
        return False
    return any(
        isinstance(row, dict)
        and row.get("status") == "verified"
        and row.get("remote") == remote
        and row.get("direction") == call.get("command")
        and row.get("operation_id") == call.get("operation_id")
        and row.get("sha256") == call.get("sha256")
        and row.get("verified_call_index") == index
        for row in receipt.get("transfers") or []
    )


def _read_recovery(
    calls: list[dict[str, Any]], index: int, *, legacy: bool
) -> tuple[int, int, float] | None:
    """Return bounded recovery chain evidence for one idempotent read."""
    call = calls[index]
    if legacy:
        seconds = 0.0
        for later in range(index + 1, len(calls)):
            seconds += _finite(calls[later].get("elapsed_seconds")) or 0.0
            if (
                calls[later].get("command") == call.get("command")
                and calls[later].get("status") == "completed"
            ):
                return (later, 1, seconds)
        return None
    operation_id, digest = call.get("operation_id"), call.get("operation_sha256")
    started = _finite(call.get("finished_monotonic"))
    if (
        not all(isinstance(value, str) and value for value in (operation_id, digest))
        or started is None
    ):
        return None
    attempts = 0
    for later in range(index + 1, len(calls)):
        other = calls[later]
        if other.get("command") != call.get("command") or other.get("operation_id") != operation_id:
            continue
        attempts += 1
        if (
            other.get("status") == "completed"
            and other.get("operation_sha256") == digest
            and other.get("durable_final_reply") is True
        ):
            finished = _finite(other.get("finished_monotonic"))
            if finished is not None and finished >= started:
                return (later, attempts, finished - started)
    return None


def _exec_episodes(
    calls: list[dict[str, Any]],
    excluded: set[int],
    counted: list[dict[str, Any]],
    recovered: list[dict[str, Any]],
    *,
    legacy_projection: bool,
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
    handled_requests: set[str] = set()
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
        if request_id in handled_requests:
            index += 1
            continue
        attempts, seconds, resolved_at = 0, 0.0 if legacy_projection else None, None
        probe_attempts, reissue_attempts, bounded_attempts = 0, 0, True
        valid_idle_before_reissue = True
        previous_reissue = index
        for later in range(index + 1, len(calls)):
            other = calls[later]
            if legacy_projection:
                seconds += _finite(other.get("elapsed_seconds")) or 0.0
            if other.get("command") == "exec" and other.get("request_id") == request_id:
                reissue_attempts += 1
                if legacy_projection:
                    attempts += 1
                timeout = _finite(other.get("timeout_seconds"))
                bounded_attempts = bounded_attempts and timeout is not None and timeout <= 30.0
                if not legacy_projection and reissue_attempts:
                    valid_idle_before_reissue = valid_idle_before_reissue and any(
                        probe.get("command") == "kernel_idle"
                        and probe.get("request_id") == request_id
                        and probe.get("request_sha256") == call.get("request_sha256")
                        and probe.get("idle") is True
                        and probe.get("status") == "completed"
                        for probe in calls[previous_reissue + 1 : later]
                    )
                previous_reissue = later
                if other.get("status") == "completed" and (
                    legacy_projection
                    or (
                        other.get("request_sha256") == call.get("request_sha256")
                        and other.get("durable_final_reply") is True
                    )
                ):
                    resolved_at = later
                    started = _finite(call.get("finished_monotonic"))
                    finished = _finite(other.get("finished_monotonic"))
                    if started is not None and finished is not None and finished >= started:
                        seconds = finished - started
                    break
            elif other.get("command") == "kernel_idle" and other.get("request_id") == request_id:
                probe_attempts += 1
                timeout = _finite(other.get("timeout_seconds"))
                bounded_attempts = bounded_attempts and timeout is not None and timeout <= 30.0
        # A recovery iteration is an idle proof followed, when idle, by its
        # reissue. Count the larger side so a run of busy probes cannot be
        # hidden by a single later reissue, without double-counting pairs.
        if not legacy_projection:
            attempts = max(probe_attempts, reissue_attempts)
            if not bounded_attempts or not valid_idle_before_reissue:
                resolved_at = None
        source = "derived"
        actual_attempts, actual_seconds = attempts, seconds
        summary = summaries.get(request_id)
        if summary is not None and summary[0] > index:
            source = "recovery-record"
            attempts = summary[1].get("attempts")
            seconds = _finite(summary[1].get("recovery_seconds"))
            started = _finite(summary[1].get("first_failure_monotonic"))
            finished = _finite(summary[1].get("resolved_monotonic"))
            if not legacy_projection and not (
                summary[1].get("status") == "completed"
                and summary[1].get("request_sha256") == call.get("request_sha256")
                and summary[1].get("final_request_sha256") == call.get("request_sha256")
                and summary[1].get("durable_final_reply") is True
                and summary[1].get("attempts") == actual_attempts
                and resolved_at is not None
                and started is not None
                and finished is not None
                and _finite(call.get("finished_monotonic")) == started
                and _finite(calls[resolved_at].get("finished_monotonic")) == finished
                and finished >= started
                and seconds is not None
                and math.isclose(seconds, finished - started, abs_tol=1e-6)
            ):
                resolved_at = None
            elif actual_seconds is not None and not math.isclose(
                seconds, actual_seconds, abs_tol=1e-6
            ):
                resolved_at = None
        if (
            resolved_at is None
            or type(attempts) is not int
            or seconds is None
            or attempts < 1
            or seconds < 0
            or not bounded_attempts
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
            handled_requests.add(request_id)
            index += 1
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
        handled_requests.add(request_id)
        # Do not jump to the resolution: another request can have failed
        # between this request's first failed call and its later completion.
        index += 1


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
        if started < 0 or done < 0 or done > 1 or (done and started != 1) or started > 1:
            counted.append(_fault("C4", f"{request_id}: started={started} done={done}"))
        elif isinstance(hashes, list) and len(set(hashes)) > 1:
            counted.append(_fault("C4", f"{request_id}: marker code hash differs"))


def _marker_coverage_faults(
    calls: list[dict[str, Any]], markers: dict[str, Any] | None, counted: list[dict[str, Any]]
) -> None:
    """Require the prospective marker census to cover each executed request."""
    if markers is None:
        counted.append(_fault("C4", "prospective session lacks marker census"))
        return
    _marker_faults(markers, counted)
    request_hashes: dict[str, set[Any]] = {}
    for call in calls:
        request_id = call.get("request_id")
        if call.get("command") == "exec" and isinstance(request_id, str):
            request_hashes.setdefault(request_id, set()).add(call.get("request_sha256"))
    for request_id, hashes in request_hashes.items():
        evidence = markers.get(request_id)
        if not isinstance(evidence, dict):
            counted.append(_fault("C4", f"{request_id}: marker evidence absent from census"))
            continue
        marker_hashes = evidence.get("code_sha256")
        marker_hashes = (
            {marker_hashes} if isinstance(marker_hashes, str) else set(marker_hashes or [])
        )
        if len(hashes) != 1 or not isinstance(next(iter(hashes)), str) or marker_hashes != hashes:
            counted.append(
                _fault("C4", f"{request_id}: marker code hash does not bind call identity")
            )
            continue
        completed = any(
            call.get("command") == "exec"
            and call.get("request_id") == request_id
            and call.get("status") == "completed"
            for call in calls
        )
        if completed and (evidence.get("started") != 1 or evidence.get("done") != 1):
            counted.append(
                _fault("C4", f"{request_id}: completed request lacks exactly-once markers")
            )


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
    legacy_projection: bool = False,
) -> dict[str, Any]:
    """Classify one supervisor receipt under v2. Raises ValueError if malformed."""
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    if markers is None and "marker_census" in receipt:
        markers = receipt["marker_census"]
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
        if call.get("command") == "exec" and (
            not isinstance(call.get("request_id"), str)
            or not call.get("request_id")
            or not isinstance(call.get("request_sha256"), str)
            or not call.get("request_sha256")
        ):
            counted.append(_fault("C4", "exec lacks durable request ID or code hash", index, call))

    if not legacy_projection:
        for index, call in enumerate(calls):
            if call.get("command") not in TRANSFER_COMMANDS or call.get("status") != "completed":
                continue
            remote = remotes.get(index)
            if remote is None or not _completed_transfer_verified(receipt, calls, index, remote):
                counted.append(
                    _fault("C6", "completed transfer lacks correlated verification", index, call)
                )

    # Identity-based injection exclusion (failed calls only).
    for index, call in enumerate(calls):
        if call.get("status") == "completed" or call.get("command") in RECOVERY_COMMANDS:
            continue
        remote = remotes.get(index) if remotes else None
        entry = _match_injection(
            index, call, remote, entries, used, allow_retroactive=allow_retroactive_manifest
        )
        if entry is not None:
            excluded_indices.add(index)
            excluded.append(
                {"index": index, "injection_id": entry["injection_id"], "type": entry["type"]}
            )

    _exec_episodes(calls, excluded_indices, counted, recovered, legacy_projection=legacy_projection)

    for index, call in enumerate(calls):
        command = call.get("command")
        if call.get("status") == "completed" or index in excluded_indices or command == "exec":
            continue
        if command in RECOVERY_COMMANDS:
            continue  # resolved/unresolved through the exec episode and its summary
        if command in READ_COMMANDS:
            chain = _read_recovery(calls, index, legacy=legacy_projection)
            if chain is None:
                counted.append(_fault("C2", f"idempotent {command} read unresolved", index, call))
            else:
                ok, attempts, seconds = chain
                if not (
                    1 <= attempts <= RECOVERY_MAX_ATTEMPTS and 0 <= seconds <= RECOVERY_MAX_SECONDS
                ):
                    counted.append(
                        _fault("C2", f"idempotent {command} read unresolved", index, call)
                    )
                    continue
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
            if remote is not None and _verified_transfer(receipt, calls, index, remote, command):
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
        if row.get("status") != "verified":
            entry = next(
                (
                    e
                    for e in entries
                    if e["command"] in TRANSFER_COMMANDS
                    and e["command"] == row.get("direction")
                    and (
                        allow_retroactive_manifest
                        and isinstance(row.get("remote"), str)
                        and row["remote"].endswith(e["remote_suffix"])
                        or not allow_retroactive_manifest
                        and row.get("remote") == e.get("remote")
                        and row.get("operation_id") == e.get("operation_id")
                        and row.get("attempt_id") == e.get("attempt_id")
                        and row.get("failure_type") == e.get("failure_type")
                    )
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
    session_failures: list[dict[str, Any]] = []
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or terminal_class
    ):
        kind = (
            "NON_TRANSPORT"
            if terminal_class in NON_TRANSPORT_TERMINALS
            else "TRANSPORT" if terminal_class in TRANSPORT_TERMINALS else "UNKNOWN_TERMINAL"
        )
        designed = next(
            (
                e
                for e in entries
                if e["command"] == "terminal"
                and receipt.get("status") != "PASS"
                and terminal_class == e["failure_class"]
                and e["error_contains"] in str(receipt.get("error"))
                and (
                    allow_retroactive_manifest
                    or any(
                        isinstance(event, dict)
                        and event.get("event_id") == e.get("watchdog_event_id")
                        and event.get("sha256") == e.get("watchdog_event_sha256")
                        and event.get("observed") is True
                        for event in receipt.get("watchdog_events") or []
                    )
                )
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
        elif (
            terminal_class in TRANSPORT_TERMINALS
            or receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error") is not None
        ):
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
        else:
            session_failures.append(
                {
                    "kind": kind,
                    "detail": f"terminal failure_class={terminal_class} is outside C3 scope",
                }
            )

    if not legacy_projection:
        _marker_coverage_faults(calls, markers, counted)
    elif markers is not None:
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
        "verdict": "FAIL" if counted or session_failures else "PASS",
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
        "session_failures": session_failures,
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
