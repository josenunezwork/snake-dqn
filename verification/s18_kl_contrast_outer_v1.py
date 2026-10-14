"""Bound one separately admitted S16 FIT worker using the qualified S7 outer lifecycle.

This wrapper owns one FIT child and its matching shared runtime lease. It never launches
DATA or SCORE, interprets scientific results, extends a stage clock, or ratifies a protocol.
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
from typing import Any

REPO = Path("/Users/josenunez/.codex/worktrees/m3d-autonomous/snake-dqn")
BASE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-autonomous-20261010")
PYTHON = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RAW_STATUSES = ("PAIRED_TRAIN_ADEQUATE_AUDIT_ONLY", "PAIRED_TRAIN_UNDERFIT_NO_DIAG")

sys.path.insert(0, str(REPO))
from research.redesign_m3d_20261010 import process_cleanup, run_census  # noqa: E402


def sha256(path: Path) -> str:
    """Return a file's byte-exact SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_json(path: Path, value: dict[str, Any]) -> None:
    """Write a create-only JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def replace_state(value: dict[str, Any]) -> None:
    """Atomically replace only the shared runtime state document."""
    state = BASE / "state.json"
    temporary = state.with_suffix(".s16-fit-outer.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(state)


def lease_path() -> Path:
    """Return the shared exclusive, recoverable state-admission lock."""
    state = BASE / "state.json"
    return state.with_name(state.name + ".m3d-s7.lock")


def require_lease(lock: Path) -> dict[str, Any]:
    """Confirm that this wrapper still owns the shared runtime state lease."""
    if not lock.is_file():
        raise RuntimeError("FIT state lock disappeared")
    if json.loads(lock.read_text()).get("controller_pid") != os.getpid():
        raise RuntimeError("FIT state lock owner differs")
    state = json.loads((BASE / "state.json").read_text())
    active = state.get("active_local_job") or {}
    if active.get("controller_pid") != os.getpid() or active.get("lease_lock") != str(lock):
        raise RuntimeError("FIT state lease owner differs")
    return state


def pin_map(fit_source: Path, fit_sha: str, pins: list[list[str]]) -> dict[Path, str]:
    """Validate pin syntax and return a de-duplicated absolute-path mapping."""
    expected: dict[Path, str] = {fit_source.resolve(): fit_sha}
    for raw_path, digest in pins:
        path = Path(raw_path).resolve()
        if path in expected and expected[path] != digest:
            raise ValueError(f"conflicting pin for {path}")
        expected[path] = digest
    for path, digest in expected.items():
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"malformed SHA-256 pin for {path}")
        if not path.is_file():
            raise FileNotFoundError(f"pinned source is not a file: {path}")
    return expected


def check_pins(expected: dict[Path, str]) -> dict[str, str | None]:
    """Read every pinned input and return all mismatches without accepting drift."""
    mismatches: dict[str, str | None] = {}
    for path, digest in expected.items():
        actual = sha256(path) if path.is_file() else None
        if actual != digest:
            mismatches[str(path)] = actual
    return mismatches


def validate_s16_publication_contract(
    args: argparse.Namespace, raw: dict[str, Any], publication: dict[str, Any], started: float
) -> None:
    """Bind the paired S16 producer's metadata and opaque mandatory artifact family."""
    root = args.raw_out.parent
    for path in (args.raw_out, root / "fit-publication-receipt.json"):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("S16 FIT result/publication must be regular physical files")
    expected = pin_map(args.fit_source, args.fit_sha, args.pin)
    refs: dict[str, dict[str, str]] = {}
    for flag in ("--protocol", "--input-manifest"):
        values = []
        for name in (flag, flag + "-sha256"):
            if args.child_args.count(name) != 1 or any(
                item.startswith(name + "=") for item in args.child_args
            ):
                raise RuntimeError(f"S16 FIT requires exactly one pinned {name} pair")
            index = args.child_args.index(name)
            if index + 1 >= len(args.child_args) or args.child_args[index + 1].startswith("--"):
                raise RuntimeError(f"S16 FIT lacks {name} value")
            values.append(args.child_args[index + 1])
        path, digest = Path(values[0]).resolve(), values[1]
        if expected.get(path) != digest or not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"S16 FIT independent physical pin differs for {flag}")
        refs[flag] = {"path": str(path), "sha256": digest}
    if (
        raw.get("protocol_sha256") != refs["--protocol"]["sha256"]
        or raw.get("input_manifest") != refs["--input-manifest"]
        or raw.get("global_monotonic_start") != started
        or raw.get("work_deadline") != started + 200.0
        or raw.get("diagnostic_collection_permitted") is not False
        or raw.get("candidate_created") is not False
        or type(raw.get("worlds_collected")) is not int
        or raw.get("worlds_collected") != 0
        or raw.get("device") != "mps-float32"
    ):
        raise RuntimeError("S16 FIT result manifest/clock/TRAIN-only policy differs")
    required = {"plan-replay.json", "benchmark.json", "adequacy.json"}
    required.update(
        f"{arm}-{suffix}.json"
        for arm in ("ANCHORED", "KL_0P1")
        for suffix in (
            "benchmark-training-score",
            "fit-claim",
            "scope",
            "updates",
            "scoring-ledger",
            "disposal",
        )
    )
    required.update(f"{arm}-audit-only.pth" for arm in ("ANCHORED", "KL_0P1"))
    inventory = publication["artifacts"]
    if not required.issubset(inventory):
        raise RuntimeError("S16 FIT publication lacks complete paired artifacts and audit states")
    for name, ref in inventory.items():
        if not isinstance(name, str) or not isinstance(ref, dict):
            raise RuntimeError("S16 FIT artifact inventory is malformed")
        path = Path(ref.get("path", ""))
        if (
            not path.is_absolute()
            or path.parent != root
            or path.name != name
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RuntimeError("S16 FIT artifact is not a direct regular physical output")
    cleanup = raw.get("cleanup")
    completed = cleanup.get("completed_monotonic") if isinstance(cleanup, dict) else None
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("datasets_closed") is not True
        or cleanup.get("parameters_and_optimizers_disposed") is not True
        or cleanup.get("cleanup_deadline") != started + 210.0
        or type(completed) not in (int, float)
        or not math.isfinite(completed)
        or not started <= completed <= publication["published_monotonic"]
    ):
        raise RuntimeError("S16 FIT cleanup metadata must precede publication")


def validate_publication(args: argparse.Namespace, raw: dict[str, Any], started: float) -> None:
    """Require original-clock publication and physical artifact hashes without decoding states."""
    root = args.raw_out.parent
    failure = root / "fit-failure.json"
    if failure.exists() or failure.is_symlink():
        raise RuntimeError("S16 FIT failure marker overrides publication")
    publication = json.loads((root / "fit-publication-receipt.json").read_text())
    work_deadline = started + 200.0
    permitted = False
    if (
        raw.get("schema") != "m3d-s16-kl-contrast-fit/v1"
        or raw.get("status") not in RAW_STATUSES
        or not isinstance(publication, dict)
        or publication.get("schema") != "m3d-s16-kl-contrast-publication/v1"
        or publication.get("status") != "PUBLISHED_WITHIN_WORK_DEADLINE"
        or publication.get("fit_result")
        != {"path": str(args.raw_out), "sha256": sha256(args.raw_out)}
        or publication.get("global_monotonic_start") != started
        or publication.get("work_deadline") != work_deadline
        or publication.get("cleanup_deadline") != started + 210.0
        or publication.get("diagnostic_collection_permitted") is not permitted
        or publication.get("protocol_sha256") != raw.get("protocol_sha256")
        or not isinstance(raw.get("protocol_sha256"), str)
    ):
        raise RuntimeError("S16 FIT publication schema/status/hash/original clock differs")
    published = publication.get("published_monotonic")
    if (
        type(published) not in (int, float)
        or not math.isfinite(published)
        or not started <= published <= min(work_deadline, time.monotonic())
    ):
        raise RuntimeError("S16 FIT result publication is late, future, or nonfinite")
    inventory = publication.get("artifacts")
    if not isinstance(inventory, dict) or not inventory or raw.get("artifacts") != inventory:
        raise RuntimeError("S16 FIT publication artifact inventory is missing or differs")
    validate_s16_publication_contract(args, raw, publication, started)
    for ref in inventory.values():
        if time.monotonic() >= started + 210.0:
            raise TimeoutError("S16 inherited global deadline during publication rehash")
        if not isinstance(ref, dict):
            raise RuntimeError("S16 FIT artifact reference must be an object")
        path = Path(ref["path"])
        if (
            not path.is_absolute()
            or not path.resolve().is_relative_to(root.resolve())
            or sha256(path) != ref["sha256"]
        ):
            raise RuntimeError("S16 FIT published artifact root/hash differs")
    if failure.exists() or failure.is_symlink():
        raise RuntimeError("S16 FIT failure marker overrides publication")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fit-source", type=Path, required=True)
    parser.add_argument("--fit-sha", required=True)
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--deadline",
        type=float,
        required=True,
        help="External absolute time.monotonic() deadline.",
    )
    parser.add_argument("--pin", nargs=2, action="append", default=[], metavar=("PATH", "SHA"))
    parser.add_argument("--global-monotonic-start", type=float, required=True)
    parser.add_argument("--expected-status", action="append", choices=RAW_STATUSES)
    parser.add_argument(
        "child_args",
        nargs=argparse.REMAINDER,
        help="Pass child-FIT options after --.",
    )
    args = parser.parse_args()
    if args.child_args[:1] == ["--"]:
        args.child_args = args.child_args[1:]
    if args.expected_status is None:
        args.expected_status = list(RAW_STATUSES)
    args.raw_out, args.out = args.raw_out.resolve(), args.out.resolve()
    if args.raw_out.name != "fit-result.json":
        parser.error("--raw-out must name FIT_DIRECTORY/fit-result.json")
    protected = (
        "--deadline-monotonic",
        "--work-deadline",
        "--global-monotonic-start",
        "--out",
        "--output-root",
    )
    if any(
        item == name
        or item.startswith(name + "=")
        or (item.startswith("--") and name.startswith(item.split("=", 1)[0]))
        for item in args.child_args
        for name in protected
    ):
        parser.error("child arguments may not override inherited clock or output")
    return args


def main() -> int:
    args = parse_args()
    # Root takes this clock once before stage admission; parsing and rehashing are charged.
    started = args.global_monotonic_start
    global_deadline = started + 210.0
    work_deadline = started + 200.0
    errors: list[str] = []
    pending: list[int] = []
    child: subprocess.Popen[Any] | None = None
    cleanup: str | None = None
    closed = False
    reserved = False
    returncode: int | None = None
    preflight_pins: dict[str, str | None] = {}
    postflight_pins: dict[str, str | None] = {}
    launch_path = args.out.with_name(args.out.stem + "-launch.json")
    log_path = args.out.with_name(args.out.stem + ".log")
    expected: dict[Path, str] = {}
    lease_clear = False
    publication_sha: str | None = None
    state_error: str | None = None
    original_handlers: dict[int, Any] = {}
    lock = lease_path()
    lock_owned = False

    try:
        if (
            not math.isfinite(started)
            or not math.isfinite(args.deadline)
            or args.deadline != global_deadline
            or started > time.monotonic()
            or time.monotonic() >= work_deadline
        ):
            raise TimeoutError("invalid, future, expired or reset inherited S16 FIT clock")
        if run_census.default_guard():
            raise RuntimeError("S16 physical guard blocks FIT admission")
        if (
            args.fit_source.resolve()
            != (REPO / "research/redesign_m3d_s18_20261010/fit_kl_contrast.py").resolve()
        ):
            raise ValueError("outer launches only the canonical S16 FIT source")
        expected = pin_map(args.fit_source, args.fit_sha, args.pin)
        preflight_pins = check_pins(expected)
        if preflight_pins:
            raise RuntimeError(f"pinned source drift before lease: {preflight_pins}")
        if any(
            path.exists() or path.is_symlink()
            for path in (
                args.raw_out,
                args.out,
                launch_path,
                log_path,
                args.raw_out.parent / "fit-publication-receipt.json",
                args.raw_out.parent / "fit-failure.json",
            )
        ):
            raise FileExistsError("FIT raw output, receipt, launch receipt, or log already exists")
        for signum in (signal.SIGTERM, signal.SIGINT):
            original_handlers[signum] = signal.signal(
                signum, lambda number, _frame: pending.append(number)
            )

        if pending or time.monotonic() >= work_deadline:
            raise TimeoutError("signal or work deadline before lease reservation")
        if run_census.default_guard():
            raise RuntimeError("S16 physical guard blocks FIT reservation")

        with lock.open("x") as stream:
            json.dump(
                {
                    "controller_pid": os.getpid(),
                    "stage": "s18-kl-contrast-fit",
                },
                stream,
            )
            stream.write("\n")
        lock_owned = True
        state = json.loads((BASE / "state.json").read_text())
        if state.get("active_local_job") is not None:
            raise RuntimeError("active_local_job is already reserved")
        state["active_local_job"] = {
            "controller_pid": os.getpid(),
            "stage": "s18-kl-contrast-fit",
            "deadline_monotonic": global_deadline,
            "lease_lock": str(lock),
            "children": {},
        }
        replace_state(state)
        reserved = True

        if pending or time.monotonic() >= work_deadline:
            raise TimeoutError("signal or work deadline before child launch")
        prelaunch_pins = check_pins(expected)
        if prelaunch_pins:
            raise RuntimeError(f"pinned source drift before child launch: {prelaunch_pins}")
        environment = dict(
            os.environ,
            PYTHONPATH=str(REPO),
            SNAKE_DQN_DEVICE="mps",
            PYTORCH_ENABLE_MPS_FALLBACK="0",
            **{name: "1" for name in THREAD_CAPS},
        )
        command = [
            PYTHON,
            str(REPO / "research/redesign_m3_20261008/slotrun.py"),
            "--",
            PYTHON,
            str(args.fit_source.resolve()),
            *args.child_args,
            "--global-monotonic-start",
            repr(started),
            "--deadline-monotonic",
            repr(work_deadline),
            "--out",
            str(args.raw_out),
        ]
        with log_path.open("x") as log:
            if pending or time.monotonic() >= work_deadline:
                raise TimeoutError("signal or work deadline after FIT log admission")
            if check_pins(expected):
                raise RuntimeError("pinned source drift immediately before child launch")
            require_lease(lock)
            if run_census.default_guard():
                raise RuntimeError("S16 physical guard blocks FIT launch")
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
        active["children"] = {"fit": {"pid": child.pid, "pgid": child.pid}}
        state["active_local_job"] = active
        replace_state(state)
        create_json(
            launch_path,
            {
                "schema": "m3d-s16-kl-contrast-outer-launch-raw/v1",
                "fit_source": str(args.fit_source.resolve()),
                "fit_source_sha256": args.fit_sha,
                "pins": {str(path): digest for path, digest in expected.items()},
                "command": command,
                "fit_pid": child.pid,
                "fit_pgid": child.pid,
                "outer_pid": os.getpid(),
                "started_monotonic": started,
                "work_deadline": work_deadline,
                "global_deadline": global_deadline,
                "thread_caps": {name: environment[name] for name in THREAD_CAPS},
            },
        )
        next_guard = time.monotonic()
        while child.poll() is None:
            if pending:
                raise InterruptedError(f"received signals {pending}")
            if time.monotonic() >= work_deadline:
                raise TimeoutError("FIT work deadline ended")
            now = time.monotonic()
            if now >= next_guard:
                if run_census.default_guard():
                    raise RuntimeError("S16 physical guard reached during FIT")
                require_lease(lock)
                next_guard = now + 5.0
            try:
                child.wait(timeout=min(0.25, max(0.01, work_deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        returncode = child.returncode
        if pending or time.monotonic() >= work_deadline:
            raise TimeoutError("signal or original FIT work deadline after child exit")
        if returncode != 0:
            errors.append(f"actual FIT return code: {returncode}")
    except BaseException as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if child is not None:
            try:
                state = require_lease(lock)
                active = dict(state["active_local_job"])
                active["children"] = {"fit": {"pid": child.pid, "pgid": child.pid}}
                state["active_local_job"] = active
                replace_state(state)
            except BaseException as error:
                errors.append(f"child ownership persistence: {type(error).__name__}: {error}")
            cleanup_deadline = min(global_deadline, time.monotonic() + 10.0)
            try:
                cleanup = process_cleanup.close_process_group(child, deadline=cleanup_deadline)
                closed = not process_cleanup._live_members(child.pid)
            except BaseException as error:
                errors.append(f"cleanup: {type(error).__name__}: {error}")
        else:
            # No Popen occurred, so there is no owned group whose absence is uncertain.
            closed = True
        if not closed:
            errors.append("owned FIT process group closure is unverified")
        if reserved and closed:
            try:
                state = require_lease(lock)
                active = state.get("active_local_job") or {}
                if active.get("controller_pid") == os.getpid():
                    state["active_local_job"] = None
                    replace_state(state)
                    lock.unlink()
                    lock_owned = False
                else:
                    errors.append("runtime lease owner differs; lease left intact")
            except BaseException as error:
                errors.append(f"lease cleanup: {type(error).__name__}: {error}")
        elif reserved:
            errors.append("runtime lease retained because closure is unverified")
        elif lock_owned:
            try:
                if json.loads(lock.read_text()).get("controller_pid") != os.getpid():
                    raise RuntimeError("unreserved lock owner changed; foreign lock preserved")
                lock.unlink()
                lock_owned = False
            except BaseException as error:
                errors.append(f"unreserved lock cleanup: {type(error).__name__}: {error}")
        try:
            lease_clear = (
                json.loads((BASE / "state.json").read_text()).get("active_local_job") is None
            )
        except BaseException as error:
            state_error = f"{type(error).__name__}: {error}"
            errors.append(f"state inspection: {state_error}")
        try:
            postflight_pins = check_pins(expected) if expected else {}
        except BaseException as error:
            postflight_pins = {"<postflight-pin-check>": None}
            errors.append(f"postflight pin check: {type(error).__name__}: {error}")
        if postflight_pins:
            errors.append(f"pinned source drift after child: {postflight_pins}")
        if pending:
            errors.append(f"pending signals: {pending}")
        if time.monotonic() > global_deadline:
            errors.append("outer global deadline exceeded")
        raw: dict[str, Any] | None = None
        if args.raw_out.is_file():
            try:
                raw = json.loads(args.raw_out.read_text())
            except BaseException as error:
                errors.append(f"raw FIT JSON: {type(error).__name__}: {error}")
        else:
            errors.append("raw FIT receipt was not created")
        if raw is not None and not isinstance(raw, dict):
            errors.append("raw FIT JSON must be an object")
            raw = None
        if raw is not None and raw.get("status") not in args.expected_status:
            errors.append(
                f"raw FIT status was {raw.get('status')!r}, expected {args.expected_status!r}"
            )
        if raw is not None:
            try:
                validate_publication(args, raw, started)
                publication_sha = sha256(args.raw_out.parent / "fit-publication-receipt.json")
            except BaseException as error:
                errors.append(f"FIT publication: {type(error).__name__}: {error}")
        if not lease_clear:
            errors.append("root runtime lease is not clear")
    try:
        raw_sha = sha256(args.raw_out) if args.raw_out.is_file() else None
    except BaseException as error:
        raw_sha = None
        errors.append(f"raw FIT hash: {type(error).__name__}: {error}")
    receipt = {
        "schema": "m3d-s16-kl-contrast-outer-raw/v1",
        "status": (
            "PASS_S18_KL_CONTRAST_OUTER_RAW"
            if not errors
            else "FAIL_S18_KL_CONTRAST_OUTER_RAW"
        ),
        "expected_raw_statuses": args.expected_status,
        "actual_returncode": returncode,
        "errors": errors,
        "fit_source": {
            "path": str(args.fit_source.resolve()),
            "sha256": args.fit_sha,
        },
        "source_pins": {
            "expected": {str(path): digest for path, digest in expected.items()},
            "before": preflight_pins,
            "after": postflight_pins,
        },
        "raw_fit": {
            "path": str(args.raw_out),
            "sha256": raw_sha,
            "status": None if raw is None else raw.get("status"),
        },
        "publication": {
            "path": str(args.raw_out.parent / "fit-publication-receipt.json"),
            "sha256": publication_sha,
        },
        "clock": {
            "started_monotonic": started,
            "work_deadline": work_deadline,
            "global_deadline": global_deadline,
            "elapsed_seconds": time.monotonic() - started,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "process": {
            "fit_pid": None if child is None else child.pid,
            "fit_pgid": None if child is None else child.pid,
            "cleanup": cleanup,
            "owned_group_absent": closed,
        },
        "lease": {
            "reserved": reserved,
            "clear": lease_clear,
            "state_error": state_error,
        },
    }
    try:
        create_json(args.out, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "errors": errors,
                    "receipt": str(args.out),
                }
            )
        )
    except BaseException as error:
        print(
            json.dumps(
                {"status": "FAIL_S18_KL_CONTRAST_OUTER_RAW", "receipt_error": str(error)}
            )
        )
        return 2
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
    final_gate_failed = bool(pending) or time.monotonic() > global_deadline
    return 0 if not errors and not final_gate_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
