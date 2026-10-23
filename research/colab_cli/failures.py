"""Explicit terminal failure categories persisted by the Colab supervisor."""

from __future__ import annotations

import re
import subprocess
from enum import StrEnum
from typing import Any


class FailureClass(StrEnum):
    """Receipt-stable classes used by recovery and postmortem tooling."""

    CHILD_NONZERO_EXIT = "CHILD_NONZERO_EXIT"
    FOREIGN_GENERATION = "FOREIGN_GENERATION"
    CONTROL_RPC_TIMEOUT = "CONTROL_RPC_TIMEOUT"
    CONTROL_RPC_ERROR = "CONTROL_RPC_ERROR"
    RPC_EXEC_POLL_TIMEOUT = "RPC_EXEC_POLL_TIMEOUT"
    UPLOAD_CAP = "UPLOAD_CAP"
    ACCOUNTING = "ACCOUNTING"
    SUPERVISOR = "SUPERVISOR"


class ClassifiedFailure(RuntimeError):
    """A refusal with a non-ambiguous receipt failure class."""

    def __init__(self, failure_class: FailureClass, detail: str) -> None:
        super().__init__(detail)
        self.failure_class = failure_class


def terminal_failure(terminal: dict[str, Any], generation: str) -> ClassifiedFailure | None:
    """Classify a terminal receipt before declaring a remote job successful."""
    if terminal.get("generation") != generation:
        return ClassifiedFailure(
            FailureClass.FOREIGN_GENERATION,
            "terminal generation does not match this supervised attempt",
        )
    returncode = terminal.get("returncode")
    if returncode != 0:
        line = terminal.get("last_traceback_line") or terminal.get("last_error_line")
        if not isinstance(line, str) or not line.strip():
            line = str(terminal.get("reason") or "no traceback line published")
        return ClassifiedFailure(
            FailureClass.CHILD_NONZERO_EXIT,
            f"remote child exited {returncode}: {line.strip().splitlines()[-1]}",
        )
    if terminal.get("status") not in {"completed", "success", "succeeded", "PASS"}:
        return ClassifiedFailure(
            FailureClass.SUPERVISOR,
            f"remote terminal not successful: {terminal.get('status')}",
        )
    return None


def control_timeout(exc: BaseException) -> bool:
    """Identify control-channel timeouts without treating a provider 5xx as one."""
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return True
    message = str(exc).lower()
    # Rich tracebacks include nearby source (e.g. timeout=timeout). Prefer the
    # final exception line so a provider 5xx is not mislabeled as a stall.
    terminal = re.findall(
        r"(?:^|\n)(?:cli_guard: )?([\w.]+(?:error|exception|expired): [^\n]+)", message
    )
    if terminal:
        message = terminal[-1]
    return bool(
        re.search(
            r"timed out|timeouterror|timeoutexpired|"
            r"(?:execution|kernel|gateway) timeout|^timeout\b",
            message,
        )
    )


def classify_exception(exc: BaseException) -> FailureClass:
    """Map known boundaries to receipt classes without stringifying secrets."""
    if isinstance(exc, ClassifiedFailure):
        return exc.failure_class
    if exc.__cause__ is not None:
        cause_class = classify_exception(exc.__cause__)
        if cause_class in {FailureClass.CONTROL_RPC_TIMEOUT, FailureClass.CONTROL_RPC_ERROR}:
            return cause_class
    try:
        from research.colab_cli.accounting import AccountingError

        if isinstance(exc, AccountingError):
            return FailureClass.ACCOUNTING
    except ImportError:  # pragma: no cover - defensive for isolated imports
        pass
    if isinstance(exc, subprocess.TimeoutExpired):
        command = exc.cmd
        if isinstance(command, (list, tuple)) and "exec" in command:
            return FailureClass.CONTROL_RPC_TIMEOUT
        return FailureClass.RPC_EXEC_POLL_TIMEOUT
    if "colab exec failed" in str(exc).lower():
        return (
            FailureClass.CONTROL_RPC_TIMEOUT
            if control_timeout(exc)
            else FailureClass.CONTROL_RPC_ERROR
        )
    detail = str(exc).lower()
    if "generation fence mismatch" in detail or "lease_generation_changed" in detail:
        return FailureClass.FOREIGN_GENERATION
    if (
        "upload exceeds bounded transfer size" in detail
        or "pinned upload exceeds bounded" in detail
        or "total upload time cap" in detail
        or "control upload exceeds" in detail
    ):
        return FailureClass.UPLOAD_CAP
    if any(
        marker in detail
        for marker in (
            "telemetry recovery",
            "kernel completion not acknowledged",
            "kernel remains busy/unknown",
            "exec operation budget exhausted",
            "colab exec operation budget exhausted",
            "exec timed out",
            "rpc timed out",
            "poll timed out",
        )
    ):
        return FailureClass.RPC_EXEC_POLL_TIMEOUT
    if "account" in detail and any(
        marker in detail
        for marker in ("rate", "assignment", "balance", "reserve", "accounted", "attribution")
    ):
        return FailureClass.ACCOUNTING
    return FailureClass.SUPERVISOR
