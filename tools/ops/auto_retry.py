#!/usr/bin/env python3
"""Prepare outcome-blind zero-export completion attempts for the ready queue."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.colab_cli.accounting import (  # noqa: E402
    ACCOUNT_ASSIGNMENT_CAP,
    POLLING_ASSIGNMENT_CAP_ERROR,
    AccountingError,
)
from research.colab_cli.cli_guard import (  # noqa: E402
    KERNEL_CREATION_GUARD_ERROR,
    GuardError,
)

STALE_ACCOUNTED_ENDPOINTS_PREFIX = "accounted assignment endpoints are no longer active: "


def read(path: Path) -> dict[str, Any]:
    """Read a JSON object, refusing partial or malformed evidence."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    """Hash evidence without interpreting scientific payloads."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stale_accounted_endpoints(error: Any) -> tuple[str, ...] | None:
    """Parse exact pre-allocation stale other_assignments AccountingError endpoints."""
    if not isinstance(error, str):
        return None
    try:
        call = ast.parse(error, mode="eval").body
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "AccountingError"
            or len(call.args) != 1
            or call.keywords
        ):
            return None
        detail = ast.literal_eval(call.args[0])
    except (ValueError, SyntaxError):
        return None
    if not isinstance(detail, str) or not detail.startswith(STALE_ACCOUNTED_ENDPOINTS_PREFIX):
        return None
    raw = detail[len(STALE_ACCOUNTED_ENDPOINTS_PREFIX) :]
    if not raw:
        return None
    parts = tuple(raw.split(", "))
    if (
        any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", ep) is None for ep in parts)
        or list(parts) != sorted(set(parts))
        or repr(AccountingError(f"{STALE_ACCOUNTED_ENDPOINTS_PREFIX}{', '.join(parts)}")) != error
    ):
        return None
    return parts


def _readonly_preadmission_calls(calls: Any) -> bool:
    """Prove only completed read-only version/sessions/usage calls ran (never colab new)."""
    return (
        isinstance(calls, list)
        and len(calls) == 3
        and all(
            isinstance(call, dict)
            and call.get("command") == command
            and call.get("operation") == "control"
            and call.get("status") == "completed"
            and type(call.get("attempt")) is int
            and call.get("attempt") == 1
            for call, command in zip(calls, ("version", "sessions", "usage"))
        )
    )


def _has_broker_record(account: Path, kind: str, session: str, output: Path) -> bool:
    """Check whether any pending/active broker record references this session or output."""
    if (account / f"{kind}-{session}.json").exists():
        return True
    if not account.is_dir():
        return False
    resolved_output = output.resolve()
    for record_path in account.glob(f"{kind}-*.json"):
        try:
            record = read(record_path)
        except (OSError, ValueError):
            if record_path.name == f"{kind}-{session}.json":
                return True
            continue
        if record.get("session") == session:
            return True
        record_output = record.get("output")
        if isinstance(record_output, str) and record_output:
            if Path(record_output).resolve() == resolved_output:
                return True
    return False


def blocked_before_allocation(manifest: dict[str, Any], output: Path, log: Path) -> bool:
    """Recognize exact pre-allocation refusals/accounting races that never ran colab new.

    The cap and stale-other-assignment checks permit only completed read-only
    version/sessions/usage calls with accounting=null, endpoint=null, and
    stop_verified=false. None of these paths acquired an assignment, so there is
    none to stop. Any identity residue or runtime operation keeps the attempt ineligible.
    """
    path = output / "receipt.json"
    if not path.is_file() or not (output / "manifest.json").is_file() or not log.is_file():
        return False
    receipt = read(path)
    session = receipt.get("session", "")
    if not isinstance(session, str):
        return False
    account = Path.home() / ".config/colab-cli/snake"
    calls = receipt.get("cli_calls")
    pending_fence = (
        receipt.get("error")
        == "Refused('unresolved allocation/stop: reconcile pending records first')"
        and receipt.get("failure_class") == "SUPERVISOR"
        and receipt.get("usage_before") is None
        and calls == []
    )
    account_cap = (
        receipt.get("error")
        in {
            f"Refused('account-wide assignment cap {cap} (including notebooks)')"
            for cap in (ACCOUNT_ASSIGNMENT_CAP, 3)
        }
        and receipt.get("failure_class") == "ACCOUNTING"
        and isinstance(receipt.get("usage_before"), dict)
        and _readonly_preadmission_calls(calls)
    )
    stale_other_assignments = (
        stale_accounted_endpoints(receipt.get("error")) is not None
        and receipt.get("failure_class") == "ACCOUNTING"
        and isinstance(receipt.get("usage_before"), dict)
        and _readonly_preadmission_calls(calls)
    )
    return (
        re.search(r"^EXIT=[1-9]\d*$", log.read_text(), re.MULTILINE) is not None
        and (pending_fence or account_cap or stale_other_assignments)
        and receipt.get("status") == "INCOMPLETE"
        and receipt.get("endpoint") is None
        and receipt.get("stop_verified") is False
        and receipt.get("accounting") is None
        and receipt.get("usage_after") is None
        and receipt.get("cleanup_error") is None
        and receipt.get("terminal") is None
        and all(
            receipt.get(k) == []
            for k in (
                "control_operations",
                "exports",
                "transfers",
                "contents_transfers",
            )
        )
        and receipt.get("manifest_sha256") == digest(output / "manifest.json")
        and all(receipt.get(k) == manifest.get(k) for k in ("job_id", "commit"))
        and re.fullmatch(r"snake-[A-Za-z0-9._-]+-[a-f0-9]{12}", session) is not None
        and not _has_broker_record(account, "pending", session, output)
        and not _has_broker_record(account, "active", session, output)
        and not any(
            (output / name).exists()
            for name in (
                "hardware.json",
                "terminal.json",
                "accounting-baseline.json",
                "rate-calibration.json",
                "usage-allocated.json",
            )
        )
        and not (output / "exports").is_symlink()
        and not (output / "exports").is_file()
        and not ((output / "exports").exists() and any((output / "exports").rglob("*")))
    )


_TRANSIENT_COLAB_EXEC_EXCEPTION_RE = re.compile(
    r"^(?:[\w.]+\.)?(?:"
    r"WebSocket\w*(?:Error|Exception)"
    r"|ConnectionClosed\w*"
    r"|TimeoutError"
    r"|TimeoutExpired"
    r"|ConnectTimeout"
    r"|ReadTimeout"
    r"|WriteTimeout"
    r"|PoolTimeout"
    r"|TimeoutException"
    r"|RpcTimeoutError"
    r"|ConnectionError"
    r"|ConnectionResetError"
    r"|ConnectionAbortedError"
    r"|ConnectionRefusedError"
    r"|BrokenPipeError"
    r"|EOFError"
    r"|RemoteDisconnected"
    r"|IncompleteRead"
    r"|SSLError"
    r"|SSLEOFError"
    r"|gaierror"
    r"|NetworkError"
    r"|ConnectError"
    r"|ReadError"
    r"|WriteError"
    r"|RemoteProtocolError"
    r"|LocalProtocolError"
    r"|ProtocolError"
    r"|TransportError"
    r"|ChunkedEncodingError"
    r"|SessionLostError"
    r"|URLError"
    r"|ProxyError"
    r"|RetryError"
    r"|MaxRetryError"
    r"|NewConnectionError"
    r"|ClosedResourceError"
    r"|BrokenResourceError"
    r")\b"
)

_TRANSIENT_COLAB_EXEC_HTTP_RE = re.compile(
    r"^(?:[\w.]+\.)?(?:HTTPError|HTTPStatusError|ColabRequestError)\b"
)

_NON_RETRYABLE_HTTP_4XX_RE = re.compile(r"\b(?:40[0-79]|41\d|42[0-8]|43\d|45\d)\b")

_TRANSIENT_COLAB_EXEC_DETAIL_RE = re.compile(
    r"connection (?:is )?already closed"
    r"|socket (?:is )?already closed"
    r"|websocket[^\n]*(?:closed|reset|timed? out|disconnect|eof|broken)"
    r"|connection (?:reset|refused|aborted|closed|timed? out|lost|broken)"
    r"|broken pipe"
    r"|unexpected eof"
    r"|eof occurred in violation of protocol"
    r"|remote end closed connection"
    r"|read timed? out"
    r"|connect timed? out"
    r"|operation timed? out"
    r"|transport[^\n]*(?:timed? out|timeout|closed|reset|lost|disconnect)"
    r"|502 bad gateway"
    r"|503 service unavailable"
    r"|504 gateway timeout",
    re.I,
)


def _extract_refused_detail(error: str) -> str | None:
    try:
        call = ast.parse(error, mode="eval").body
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "Refused"
            or len(call.args) != 1
            or call.keywords
        ):
            return None
        detail = ast.literal_eval(call.args[0])
    except (ValueError, SyntaxError):
        return None
    return detail if isinstance(detail, str) else None


def is_kernel_creation_guard_error(error: Any) -> bool:
    """Match the shared cli_guard kernel-creation refusal, never a Rich source excerpt."""
    if not isinstance(error, str):
        return False
    if error in {
        KERNEL_CREATION_GUARD_ERROR,
        repr(GuardError(KERNEL_CREATION_GUARD_ERROR)),
        f"Refused({KERNEL_CREATION_GUARD_ERROR!r})",
    }:
        return True
    detail = _extract_refused_detail(error)
    return (
        isinstance(detail, str)
        and detail.startswith("colab exec failed (1): ")
        and re.search(
            r"(?:\n|: )GuardError: " + re.escape(KERNEL_CREATION_GUARD_ERROR) + r"\s*\Z",
            detail,
        )
        is not None
    )


def is_transient_colab_exec_error(error: Any) -> bool:
    """Match a transient colab exec transport/WebSocket failure, never a child/study error."""
    if not isinstance(error, str):
        return False
    detail = _extract_refused_detail(error)
    if detail is None:
        detail = error
    match = re.match(r"^colab exec failed \(\d+\):\s*(.*)\Z", detail, re.S)
    if match is None:
        return False
    body = match.group(1)
    if re.search(
        r'File "[^"]*(?:/|^)(?:src/|web/|job\.sh\b|research/(?!colab_cli/))',
        body,
    ):
        return False
    raw_lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not raw_lines:
        return False
    border_indices = [
        idx for idx, line in enumerate(raw_lines) if re.match(r"^╰[─╰╭│┼┤├┬┴╴╶\s]*$", line)
    ]
    if border_indices and border_indices[-1] + 1 < len(raw_lines):
        terminal = " ".join(raw_lines[border_indices[-1] + 1 :])
    else:
        non_border = [
            line for line in raw_lines if not re.match(r"^[╭╰│─┼┤├┬┴╴╶\s]+$", line)
        ]
        if not non_border:
            return False
        terminal = non_border[-1]
    if _TRANSIENT_COLAB_EXEC_EXCEPTION_RE.search(terminal):
        return True
    if _TRANSIENT_COLAB_EXEC_HTTP_RE.search(terminal) and not _NON_RETRYABLE_HTTP_4XX_RE.search(
        terminal
    ):
        return True
    return _TRANSIENT_COLAB_EXEC_DETAIL_RE.search(terminal) is not None


def is_transient_colab_read_budget_error(error: Any) -> bool:
    """Match a transient colab usage/sessions read operation budget exhaustion or timeout."""
    if not isinstance(error, str):
        return False
    detail = _extract_refused_detail(error)
    if detail is None:
        detail = error
    return (
        re.match(
            r"^colab (?:usage|sessions) (?:operation budget exhausted|timed out after \d+s)\Z",
            detail.strip(),
        )
        is not None
    )


def eligible(
    manifest: dict[str, Any], output: Path, log: Path, unallocated_proof: Path | None = None
) -> dict[str, Any]:
    """Require a bound, exited, stopped mechanical attempt with no outputs."""
    receipt = read(output / "receipt.json")
    executed = output / "manifest.json"
    unallocated = blocked_before_allocation(manifest, output, log)
    if unallocated_proof is not None:
        if __package__:
            from . import dispatch
        else:
            import dispatch
        proof = read(unallocated_proof)
        unallocated = (
            proof.get("schema") == "colab-allocation-unavailable/v1"
            and proof.get("mode") == "LIVE"
            and proof.get("recovery") == {"endpoint": None}
            and proof.get("endpoint_ever_assigned") is False
            and proof.get("vm_compute_started") is False
            and proof.get("receipt", {}).get("path") == str(output / "receipt.json")
            and proof.get("receipt", {}).get("sha256") == digest(output / "receipt.json")
            and dispatch.allocation_unavailable(output, log, manifest["requirements"]["gpu"])
        )
        if not unallocated:
            raise ValueError("retry requires exact live unallocated proof")
    if (
        not re.search(r"^EXIT=\d+$", log.read_text(), re.MULTILINE)
        or receipt.get("status") != "INCOMPLETE"
        or (not unallocated and receipt.get("stop_verified") is not True)
        or (not unallocated and receipt.get("cleanup_error") is not None)
        or receipt.get("exports") != []
        or receipt.get("terminal") is not None
        or (output / "terminal.json").exists()
        or (output / "exports").is_symlink()
        or ((output / "exports").exists() and any((output / "exports").rglob("*")))
        or receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("manifest_sha256") != digest(executed)
    ):
        raise ValueError("retry requires exited, bound, verified stop and zero outputs")
    actual = read(executed)
    if any(actual.get(key) != manifest.get(key) for key in ("job_id", "commit", "output_dir")):
        raise ValueError("executed identity differs from dispatched manifest")
    if unallocated:
        return receipt
    # Some historical receipts mislabel any TimeoutExpired as RPC_EXEC_POLL_TIMEOUT.
    # Require positive transport evidence, and veto child/science/integrity errors.
    error = str(receipt.get("error", ""))
    if re.search(
        r"nan\b|non.?finite|overflow|integrity|assertion|child.*exit|"
        r"pytest|test.*fail|sha.?256.*mismatch|hash.*mismatch|numerical|out of memory|\boom\b",
        error,
        re.I,
    ):
        raise ValueError("scientific/integrity failure is never retryable")
    category = receipt.get("failure_class")
    if category == "SUPERVISOR":
        # Kernel-creation guard or transient colab exec/read transport drop;
        # stop_verified + zero exports are checked above.
        if not (
            is_kernel_creation_guard_error(error)
            or is_transient_colab_exec_error(error)
            or is_transient_colab_read_budget_error(error)
        ):
            raise ValueError("unknown or ambiguous mechanical failure")
        return receipt
    if category == "ACCOUNTING":
        if error not in {
            POLLING_ASSIGNMENT_CAP_ERROR,
            repr(AccountingError(POLLING_ASSIGNMENT_CAP_ERROR)),
        }:
            raise ValueError("unknown or ambiguous mechanical failure")
        return receipt
    patterns = {
        "RPC_EXEC_POLL_TIMEOUT": (
            r"telemetry|kernel|rpc|colab.*(?:exec|usage|sessions)|transport|poll timed out"
        ),
        "TRANSPORT_TIMEOUT": r"transport|rpc|colab.*(?:exec|usage|sessions)|telemetry",
        "UPLOAD_CAP": r"upload|transfer.*cap",
        "ALLOCATION_REFUSED": r"allocation|assign|capacity|unavailable|503",
        "ALLOCATION_UNAVAILABLE": r"allocation|assign|capacity|unavailable|503",
    }
    if category not in patterns or not re.search(patterns[category], error, re.I):
        raise ValueError("unknown or ambiguous mechanical failure")
    return receipt


def completion_limit(manifest: dict[str, Any], requested: int | None = None) -> tuple[int, int]:
    """Return consumed completions and the narrowest declared package limit."""
    attempt = manifest.get("completion_attempt", 0)
    if type(attempt) is not int or not 0 <= attempt <= 2:
        raise ValueError("invalid completion_attempt")
    limits = [2]
    if manifest.get("timing_calibration", {}).get("no_automatic_retry") is True:
        limits.append(0)
    if "max_completion_attempts" in manifest:
        limits.append(manifest["max_completion_attempts"])
    resume = manifest.get("resume")
    if isinstance(resume, dict):
        limits.append(resume.get("max_outcome_blind_retries", 0))
    elif requested is None:
        raise ValueError("exempt canary requires an explicit package retry limit")
    envelope = manifest.get("technical_envelope", {})
    if "series_max_original_plus_completions" in envelope:
        limits.append(envelope["series_max_original_plus_completions"] - 1)
    if requested is not None:
        limits.append(requested)
    if any(type(limit) is not int or not 0 <= limit <= 2 for limit in limits):
        raise ValueError("invalid package completion limit")
    return attempt, min(limits)


def package_check(manifest: dict[str, Any], broker: Path, python: str) -> None:
    """Use the package's own outcome-blind resume/admission helpers when present.

    No exports means the original uploaded ledger remains valid, including any
    snapshots it already contains. Repacking that ledger would change canary
    input identity unnecessarily. These checks refuse newly completed units.
    """
    script = manifest.get("script", "")
    program = ""
    if script.startswith("research/doom_auc_smoke/"):
        program = (
            "from research.doom_auc_smoke.evidence import build_ledger\n"
            "cfg=json.loads(Path(next(i['path'] for i in m['inputs'] "
            "if i['dest']=='config.json')).read_text())\n"
            "assert build_ledger([Path(m['output_dir'])], cfg['identity'])['units']==[]\n"
        )
    elif script.startswith("research/spatial_student_pilot/"):
        program = (
            "from research.spatial_student_pilot.config import PilotConfig\n"
            "from research.spatial_student_pilot.resume import validate_ledger\n"
            "inputs={i['dest']:Path(i['path']) for i in m['inputs']}\n"
            "cfg=PilotConfig.from_dict(json.loads(inputs['config.json'].read_text()))\n"
            "validate_ledger(json.loads(inputs['verified-ledger.json'].read_text()), cfg)\n"
        )
    elif script.startswith("research/watch6_search_efficacy/"):
        program = "from research.watch6_search_efficacy.admission import admit\nadmit(m)\n"
    if program:
        subprocess.run(
            [
                python,
                "-c",
                "import json,sys\nfrom pathlib import Path\n"
                "sys.path.insert(0,sys.argv[1])\nm=json.loads(sys.argv[2])\n" + program,
                str(broker),
                json.dumps(manifest),
            ],
            cwd=broker,
            check=True,
            timeout=180,
        )


def prepare(
    manifest: Path,
    log: Path,
    destination: Path,
    broker: Path,
    *,
    limit: int | None = None,
    python: str = sys.executable,
    unallocated_proof: Path | None = None,
) -> Path:
    """Create a fresh immutable attempt, preserving source, inputs and all caps."""
    value = read(manifest)
    output = Path(value["output_dir"])
    eligible(value, output, log, unallocated_proof)
    attempt, maximum = completion_limit(value, limit)
    if attempt >= maximum:
        raise ValueError("package completion limit exhausted")
    package_check(value, broker, python)
    # Use the executed packet so supervisor defaults/accounting refresh are retained.
    successor = copy.deepcopy(read(output / "manifest.json"))
    token = uuid.uuid4().hex[:10]
    successor["job_id"] = value["job_id"][:42] + "-auto-" + token
    successor["output_dir"] = str(destination / "output")
    # Keep the scientific canary manifest unchanged apart from operational identity.
    # Main packages that already track completion_attempt advance it explicitly.
    successor["completion_attempt"] = attempt + 1
    if successor.get("requirements", {}).get("gpu") == "A100" and successor["requirements"].get(
        "prefer"
    ) == ["H100", "A100"]:
        # The sibling already consumed H100 preference. Retry its admitted A100
        # envelope directly; variants() correctly rejects an A100 primary.
        successor["requirements"].pop("prefer")
    if not destination.is_absolute():
        raise ValueError("fresh absolute retry destination required")
    destination.mkdir(parents=True, exist_ok=False)
    result = destination / "manifest.json"
    result.write_text(json.dumps(successor, indent=2, sort_keys=True) + "\n")
    proof = {
        "schema": "ops-zero-export-retry/v1",
        "completion_attempt": attempt + 1,
        "max_completion_attempts": maximum,
        "prior_manifest": {"path": str(manifest), "sha256": digest(manifest)},
        "prior_receipt": {
            "path": str(output / "receipt.json"),
            "sha256": digest(output / "receipt.json"),
        },
        "manifest_sha256": digest(result),
        "unallocated_proof": (
            {"path": str(unallocated_proof), "sha256": digest(unallocated_proof)}
            if unallocated_proof is not None
            else None
        ),
        "ledger_action": "preserve original uploaded ledger; zero new exports",
    }
    (destination / "RETRY.json").write_text(json.dumps(proof, indent=2) + "\n")
    for path in (result, destination / "RETRY.json"):
        path.with_name(path.name + ".sha256").write_text(digest(path) + "\n")
        path.with_name(path.name + ".complete").write_text("complete\n")
    return result


def main() -> int:
    """Expose preparation and integrated queue scanning without direct CLI mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        print(prepare(args.manifest, args.log, args.destination, args.broker, limit=args.limit))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
