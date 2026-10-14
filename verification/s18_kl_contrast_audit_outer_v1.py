"""Bound the independent S16 KL contrast auditor under its inherited clock.

This is an operational outer only.  It preserves the auditor's raw receipt and raw
status; a root supervisor must separately qualify the outer process exit and create
the consumer-facing qualified audit records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path("/Users/josenunez/.codex/worktrees/m3d-autonomous/snake-dqn")
BASE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-autonomous-20261010")
PYTHON = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"
SLOTRUN = REPO / "research/redesign_m3_20261008/slotrun.py"
CANONICAL_AUDIT = REPO / "verification/audit_s18_kl_contrast_v2.py"
SELF_SOURCE = Path(__file__).resolve()
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RAW_SCHEMA = "m3d-s16-kl-contrast-independent-audit-raw/v1"
MODE_CONFIG = {
    "normal": {
        "source": CANONICAL_AUDIT,
        "schema": RAW_SCHEMA,
        "statuses": (
            "PASS_S18_KL_CONTRAST_ARTIFACT_AUDIT_RAW",
            "FAIL_S18_KL_CONTRAST_ARTIFACT_AUDIT_RAW",
        ),
        "pass_status": "PASS_S18_KL_CONTRAST_ARTIFACT_AUDIT_RAW",
    },
}
OUTER_SCHEMA = "m3d-s16-kl-contrast-audit-outer-raw/v1"
OUTER_LAUNCH_SCHEMA = "m3d-s16-kl-contrast-audit-outer-launch-raw/v1"
WORK_SECONDS = 112.0
GLOBAL_SECONDS = 120.0

sys.path.insert(0, str(REPO))
from research.redesign_m3d_20261010 import process_cleanup, run_census  # noqa: E402


def sha256(path: Path) -> str:
    """Return a file's byte-exact SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one JSON receipt and refuse replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def lease_path() -> Path:
    """Return the shared S16 lease path."""
    return (BASE / "state.json").with_name("state.json.m3d-s7.lock")


def replace_state(value: Mapping[str, Any]) -> None:
    """Atomically replace the shared state document."""
    temporary = (BASE / "state.json").with_suffix(".s16-kl-contrast-audit.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(BASE / "state.json")


def require_lease(lock: Path) -> dict[str, Any]:
    """Require that the current process owns the exact shared lease."""
    if not lock.is_file() or json.loads(lock.read_text()).get("controller_pid") != os.getpid():
        raise RuntimeError("S16 KL-CONTRAST-audit lease owner differs")
    state = json.loads((BASE / "state.json").read_text())
    active = state.get("active_local_job") or {}
    if active.get("controller_pid") != os.getpid() or active.get("lease_lock") != str(lock):
        raise RuntimeError("S16 KL-CONTRAST-audit active lease owner differs")
    return state


def pin_map(audit_source: Path, audit_sha: str, pins: Sequence[Sequence[str]]) -> dict[Path, str]:
    """Validate the canonical audit source plus supplemental operational pins."""
    expected: dict[Path, str] = {audit_source.resolve(): audit_sha}
    for raw_path, digest in pins:
        path = Path(raw_path).resolve()
        if path in expected and expected[path] != digest:
            raise ValueError(f"conflicting source pin for {path}")
        expected[path] = digest
    if SELF_SOURCE not in expected:
        raise ValueError("mandatory supplemental self source pin is missing")
    if expected[SELF_SOURCE] != sha256(SELF_SOURCE):
        raise ValueError("supplemental self source pin differs")
    for path, digest in expected.items():
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"malformed source pin for {path}")
        if not path.is_file():
            raise FileNotFoundError(f"pinned source is not a file: {path}")
    return expected


def check_pins(expected: Mapping[Path, str]) -> dict[str, str | None]:
    """Return all physical pin mismatches."""
    return {
        str(path): (sha256(path) if path.is_file() else None)
        for path, digest in expected.items()
        if not path.is_file() or sha256(path) != digest
    }


def validate_clock(start: float, deadline: float, now: float | None = None) -> tuple[float, float]:
    """Validate the inherited start and return exact work/global deadlines."""
    now = time.monotonic() if now is None else now
    if (
        type(start) not in (int, float)
        or not math.isfinite(start)
        or type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or deadline != start + GLOBAL_SECONDS
        or start > now
        or now >= start + WORK_SECONDS
    ):
        raise TimeoutError(
            "invalid, future, expired, or reset inherited S16 KL-CONTRAST-audit clock"
        )
    return start + WORK_SECONDS, start + GLOBAL_SECONDS


def protected_child_args(child_args: Sequence[str]) -> None:
    """Reject child clock/output overrides that would break inherited ownership."""
    protected = ("--deadline-monotonic", "--global-monotonic-start", "--out")
    for item in child_args:
        for name in protected:
            if item == name or item.startswith(name + "="):
                raise ValueError(f"child arguments may not override {name}")


def child_command(args: argparse.Namespace, work_deadline: float) -> list[str]:
    """Build the slotrun command with the inherited raw output and work deadline."""
    protected_child_args(args.child_args)
    return [
        PYTHON,
        str(SLOTRUN),
        "--",
        PYTHON,
        str(args.audit_source.resolve()),
        *args.child_args,
        "--global-monotonic-start",
        repr(args.global_monotonic_start),
        "--deadline-monotonic",
        repr(work_deadline),
        "--out",
        str(args.raw_out),
    ]


def read_raw(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    """Read and physically hash the raw receipt without changing its bytes."""
    if not args.raw_out.is_file() or args.raw_out.is_symlink():
        return None, "raw audit receipt was not a regular file"
    try:
        raw = json.loads(args.raw_out.read_text())
    except BaseException as error:
        return None, f"raw audit JSON: {type(error).__name__}: {error}"
    if not isinstance(raw, dict):
        return None, "raw audit JSON must be an object"
    config = MODE_CONFIG[args.audit_mode]
    if raw.get("schema") != config["schema"]:
        return raw, "raw audit schema differs"
    if raw.get("status") not in args.expected_status:
        return raw, f"raw audit status was {raw.get('status')!r}"
    return raw, None


def build_receipt(
    args: argparse.Namespace,
    *,
    start: float,
    work_deadline: float,
    global_deadline: float,
    command: list[str],
    expected: Mapping[Path, str],
    preflight: Mapping[str, str | None],
    postflight: Mapping[str, str | None],
    raw: dict[str, Any] | None,
    raw_error: str | None,
    returncode: int | None,
    outer_pid: int,
    outer_pgid: int,
    audit_pid: int | None,
    audit_pgid: int | None,
    closed: bool,
    lease_clear: bool,
    reserved: bool,
    state_error: str | None,
    errors: list[str],
) -> dict[str, Any]:
    """Construct the raw operational receipt; never upgrade the auditor status."""
    raw_ref = {
        "path": str(args.raw_out),
        "sha256": sha256(args.raw_out) if args.raw_out.is_file() else None,
        "status": None if raw is None else raw.get("status"),
    }
    config = MODE_CONFIG[args.audit_mode]
    pass_status = (
        returncode == 0
        and not errors
        and raw is not None
        and raw.get("status") == config["pass_status"]
        and closed
        and lease_clear
        and not postflight
    )
    return {
        "schema": OUTER_SCHEMA,
        "status": (
            "PASS_S18_KL_CONTRAST_AUDIT_OUTER_RAW"
            if pass_status
            else "FAIL_S18_KL_CONTRAST_AUDIT_OUTER_RAW"
        ),
        "errors": errors,
        "actual_tool_exitcode": None,
        "root_tool_exit_qualified": False,
        "actual_returncode": returncode,
        "actual_worker_exitcode": returncode,
        "audit_mode": args.audit_mode,
        "normal_data_qualification_allowed": False,
        "audit_source": {"path": str(args.audit_source.resolve()), "sha256": args.audit_sha},
        "source_pins": {
            "expected": {str(path): digest for path, digest in expected.items()},
            "before": dict(preflight),
            "after": dict(postflight),
        },
        "raw_audit": raw_ref,
        "original_raw": raw_ref,
        "command": command,
        "clock": {
            "started_monotonic": start,
            "work_deadline": work_deadline,
            "global_deadline": global_deadline,
            "cleanup_seconds": 8.0,
            "elapsed_seconds": time.monotonic() - start,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "process": {
            "outer_pid": outer_pid,
            "outer_pgid": outer_pgid,
            "audit_pid": audit_pid,
            "audit_pgid": audit_pgid,
            "owned_group_absent": closed,
        },
        "lease": {"reserved": reserved, "clear": lease_clear, "state_error": state_error},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit inherited-clock and source-pin ABI."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--audit-source", type=Path, required=True)
    parser.add_argument("--audit-sha", required=True)
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--global-monotonic-start", type=float, required=True)
    parser.add_argument("--audit-mode", choices=tuple(MODE_CONFIG), default="normal")
    parser.add_argument("--pin", nargs=2, action="append", default=[], metavar=("PATH", "SHA"))
    parser.add_argument("--expected-status", action="append")
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.child_args[:1] == ["--"]:
        args.child_args = args.child_args[1:]
    config = MODE_CONFIG[args.audit_mode]
    if args.expected_status is None:
        args.expected_status = list(config["statuses"])
    elif any(status not in config["statuses"] for status in args.expected_status):
        parser.error(f"expected statuses do not match audit mode {args.audit_mode}")
    if args.raw_out.is_symlink() or args.out.is_symlink():
        parser.error("raw and outer output paths must not be symlinks")
    args.audit_source = args.audit_source.resolve()
    args.raw_out = args.raw_out.absolute()
    args.out = args.out.absolute()
    expected_source = CANONICAL_AUDIT
    if args.audit_source != expected_source.resolve():
        parser.error(f"outer source does not match audit mode {args.audit_mode}")
    protected_child_args(args.child_args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded operational outer and return its unqualified process result."""
    args = parse_args(argv)
    start = args.global_monotonic_start
    errors: list[str] = []
    pending: list[int] = []
    child: subprocess.Popen[Any] | None = None
    expected: dict[Path, str] = {}
    preflight: dict[str, str | None] = {}
    postflight: dict[str, str | None] = {}
    returncode: int | None = None
    closed = False
    reserved = False
    lease_clear = False
    lock_owned = False
    state_error: str | None = None
    original_handlers: dict[int, Any] = {}
    terminal_failure = False
    lock = lease_path()
    launch_path = args.out.with_name(args.out.stem + "-launch.json")
    log_path = args.out.with_name(args.out.stem + ".log")
    try:
        work_deadline, global_deadline = validate_clock(start, args.deadline)
        if run_census.default_guard():
            raise RuntimeError("S16 physical guard blocks KL-CONTRAST-audit admission")
        expected = pin_map(args.audit_source, args.audit_sha, args.pin)
        preflight = check_pins(expected)
        if preflight:
            raise RuntimeError(f"pinned source drift before lease: {preflight}")
        if any(
            path.exists() or path.is_symlink()
            for path in (args.raw_out, args.out, launch_path, log_path, lock)
        ):
            raise FileExistsError(
                "S16 KL-CONTRAST-audit output, launch, log, or lease already exists"
            )
        for signum in (signal.SIGTERM, signal.SIGINT):
            original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda number, _frame: pending.append(number))
        if pending or time.monotonic() >= work_deadline:
            raise TimeoutError("signal or work deadline before lease reservation")
        with lock.open("x") as stream:
            json.dump(
                {"controller_pid": os.getpid(), "stage": "s16-kl-contrast-audit"}, stream
            )
            stream.write("\n")
        lock_owned = True
        state = json.loads((BASE / "state.json").read_text())
        if state.get("active_local_job") is not None:
            raise RuntimeError("active_local_job is already reserved")
        state["active_local_job"] = {
            "controller_pid": os.getpid(),
            "stage": "s16-kl-contrast-audit",
            "deadline_monotonic": global_deadline,
            "lease_lock": str(lock),
            "children": {},
        }
        replace_state(state)
        reserved = True
        command = child_command(args, work_deadline)
        environment = dict(
            os.environ,
            PYTHONPATH=str(REPO),
            SNAKE_DQN_DEVICE="mps",
            PYTORCH_ENABLE_MPS_FALLBACK="0",
            **{name: "1" for name in THREAD_CAPS},
        )
        with log_path.open("x") as log:
            require_lease(lock)
            if pending or time.monotonic() >= work_deadline:
                raise TimeoutError("signal or work deadline before final child admission")
            if run_census.default_guard():
                raise RuntimeError(
                    "S16 KL-CONTRAST-audit admission changed before child launch"
                )
            prelaunch = check_pins(expected)
            if prelaunch:
                raise RuntimeError(f"pinned source drift before child launch: {prelaunch}")
            if pending or time.monotonic() >= work_deadline:
                raise TimeoutError("signal or work deadline during final child admission")
            child = subprocess.Popen(
                command,
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state = require_lease(lock)
        active = dict(state["active_local_job"])
        active["children"] = {"audit": {"pid": child.pid, "pgid": child.pid}}
        state["active_local_job"] = active
        replace_state(state)
        create_json(
            launch_path,
            {
                "schema": OUTER_LAUNCH_SCHEMA,
                "audit_source": str(args.audit_source),
                "audit_source_sha256": args.audit_sha,
                "pins": {str(path): digest for path, digest in expected.items()},
                "command": command,
                "outer_pid": os.getpid(),
                "outer_pgid": os.getpid(),
                "audit_pid": child.pid,
                "audit_pgid": child.pid,
                "started_monotonic": start,
                "work_deadline": work_deadline,
                "global_deadline": global_deadline,
                "cleanup_deadline": global_deadline,
                "thread_caps": {name: environment[name] for name in THREAD_CAPS},
            },
        )
        next_guard = time.monotonic()
        while child.poll() is None:
            if pending:
                raise InterruptedError(f"received signals {pending}")
            if time.monotonic() >= work_deadline:
                raise TimeoutError("S16 KL-CONTRAST-audit work deadline ended")
            now = time.monotonic()
            if now >= next_guard:
                require_lease(lock)
                if run_census.default_guard() or check_pins(expected):
                    raise RuntimeError(
                        "S16 KL-CONTRAST-audit guard or source drift during child"
                    )
                next_guard = now + 5.0
            try:
                child.wait(timeout=min(0.25, max(0.01, work_deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        if pending or time.monotonic() >= work_deadline:
            raise TimeoutError("signal or work deadline after KL-CONTRAST-audit child exit")
        require_lease(lock)
        if run_census.default_guard() or check_pins(expected):
            raise RuntimeError("S16 KL-CONTRAST-audit guard or source drift after child")
        returncode = child.returncode
        if returncode != 0:
            errors.append(f"actual KL-CONTRAST-audit worker return code: {returncode}")
    except BaseException as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if child is not None:
            try:
                state = require_lease(lock)
                active = dict(state["active_local_job"])
                active["children"] = {"audit": {"pid": child.pid, "pgid": child.pid}}
                state["active_local_job"] = active
                replace_state(state)
            except BaseException as error:
                errors.append(f"child ownership persistence: {type(error).__name__}: {error}")
            cleanup_deadline = min(
                global_deadline if "global_deadline" in locals() else start + GLOBAL_SECONDS,
                time.monotonic() + 8.0,
            )
            try:
                process_cleanup.close_process_group(child, deadline=cleanup_deadline)
                closed = not process_cleanup._live_members(child.pid)
            except BaseException as error:
                errors.append(f"cleanup: {type(error).__name__}: {error}")
        else:
            closed = True
        if not closed:
            errors.append("owned KL-CONTRAST-audit process group closure is unverified")
        if reserved and closed:
            try:
                state = require_lease(lock)
                if (state.get("active_local_job") or {}).get("controller_pid") == os.getpid():
                    state["active_local_job"] = None
                    replace_state(state)
                    lock.unlink()
                    lock_owned = False
                else:
                    errors.append("runtime lease owner differs; lease left intact")
            except BaseException as error:
                errors.append(f"lease cleanup: {type(error).__name__}: {error}")
        elif reserved:
            errors.append("runtime lease retained because owned-group closure is unverified")
        elif lock_owned:
            try:
                if json.loads(lock.read_text()).get("controller_pid") == os.getpid():
                    lock.unlink()
                    lock_owned = False
            except BaseException as error:
                errors.append(f"unreserved lease cleanup: {type(error).__name__}: {error}")
        try:
            lease_clear = (
                json.loads((BASE / "state.json").read_text()).get("active_local_job") is None
            )
        except BaseException as error:
            state_error = f"{type(error).__name__}: {error}"
            errors.append(f"state inspection: {state_error}")
        try:
            postflight = check_pins(expected) if expected else {}
        except BaseException as error:
            postflight = {"<postflight-pin-check>": None}
            errors.append(f"postflight pin check: {type(error).__name__}: {error}")
        if postflight:
            errors.append(f"pinned source drift after child: {postflight}")
        if pending:
            errors.append(f"pending signals: {pending}")
        if "global_deadline" in locals() and time.monotonic() >= global_deadline:
            errors.append("outer global deadline exceeded")
        raw, raw_error = read_raw(args)
        if raw_error:
            errors.append(raw_error)
        elif raw is not None and raw.get("status") != MODE_CONFIG[args.audit_mode]["pass_status"]:
            errors.append(f"raw audit did not pass: {raw.get('status')!r}")
        try:
            receipt = build_receipt(
                args,
                start=start,
                work_deadline=locals().get("work_deadline", start + WORK_SECONDS),
                global_deadline=locals().get("global_deadline", start + GLOBAL_SECONDS),
                command=locals().get("command", []),
                expected=expected,
                preflight=preflight,
                postflight=postflight,
                raw=raw,
                raw_error=raw_error,
                returncode=returncode,
                outer_pid=os.getpid(),
                outer_pgid=os.getpid(),
                audit_pid=None if child is None else child.pid,
                audit_pgid=None if child is None else child.pid,
                closed=closed,
                lease_clear=lease_clear,
                reserved=reserved,
                state_error=state_error,
                errors=errors,
            )
            create_json(args.out, receipt)
            if pending:
                terminal_failure = True
            if time.monotonic() >= receipt["clock"]["global_deadline"]:
                terminal_failure = True
            print(
                json.dumps({"status": receipt["status"], "errors": receipt["errors"]}), flush=True
            )
            if pending:
                terminal_failure = True
            if time.monotonic() >= receipt["clock"]["global_deadline"]:
                terminal_failure = True
        except BaseException as error:
            terminal_failure = True
            print(
                json.dumps(
                    {
                        "status": "FAIL_S18_KL_CONTRAST_AUDIT_OUTER_RAW",
                        "receipt_error": str(error),
                    }
                )
            )
        finally:
            for signum, handler in original_handlers.items():
                signal.signal(signum, handler)
    return 0 if returncode == 0 and not errors and not terminal_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
