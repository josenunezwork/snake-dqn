#!/usr/bin/env python3
"""Dispatch a qualified supervisor with bounded account allocation exclusion."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

if __package__:
    from .qualification import OPS
else:
    from qualification import OPS

HERE = Path(__file__).resolve().parent
EXACT_ERROR = "AccountingError('assignment endpoints changed during allocation')"
BLOCK = re.compile(r"^\s*(?:BLOCK|HOLD)\b|RUNTIME BLOCKED|do not run this|PREPARED_NOT_DISPATCHED")


class Refused(ValueError):
    """The wrapper cannot prove that the next operational step is allowed."""


def command(
    args: list[str], *, deadline: float | None = None, **kwargs: Any
) -> subprocess.CompletedProcess:
    """Bound every subprocess while the account lock is held."""
    remaining = 60.0 if deadline is None else min(60.0, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("allocation lock deadline expired")
    return subprocess.run(args, timeout=remaining, check=True, **kwargs)


def note(log: Path, value: str) -> None:
    """Append an ops observation without replacing supervisor evidence."""
    with log.open("a") as stream:
        stream.write(value + "\n")


@contextlib.contextmanager
def allocation_lock(path: Path, seconds: float) -> Iterator[float]:
    """Own the flock here; termination, including SIGKILL, closes the descriptor."""
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused(f"account allocation lock is held: {path}") from exc
        try:
            yield time.monotonic() + seconds
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def qualified(broker: Path, ops: Path, deadline: float | None = None, ref: str = "HEAD") -> None:
    """Check actual broker bytes both before validation and under allocation lock."""
    command(
        [
            sys.executable,
            str(HERE / "qualification.py"),
            "check",
            "--repo",
            str(broker),
            "--record",
            str(ops / "colab-cli-qualified.json"),
            "--ref",
            ref,
        ],
        deadline=deadline,
        stdout=subprocess.DEVNULL,
    )


def validate(
    broker: Path, cwd: Path, state: Path, manifest: Path, log: Path, deadline: float | None = None
) -> None:
    """Run real validation; launch.sh --check would only print and return."""
    with log.open("x") as stream:
        command(
            [
                str(broker / "research/colab_cli/launch.sh"),
                "--state-dir",
                str(state),
                "--colab",
                str(Path.home() / ".local/bin/colab"),
                "validate",
                str(manifest),
            ],
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            deadline=deadline,
        )


def register(ops: Path, log: Path, name: str) -> None:
    """Register each attempt separately, including its eventual EXIT marker."""
    with (ops / "watch-logs.txt").open("a") as stream:
        stream.write(f"{log} {name}\n")


def start(
    broker: Path, cwd: Path, state: Path, manifest: Path, log: Path, name: str, deadline: float
) -> None:
    """Submit one shell command with preserved argument boundaries and exit evidence."""
    log.touch(exist_ok=False)
    register(log.parent.parent, log, name)
    launcher = [
        str(broker / "research/colab_cli/launch.sh"),
        "--state-dir",
        str(state),
        "--colab",
        str(Path.home() / ".local/bin/colab"),
        "run",
        str(manifest),
    ]
    environment = shlex.join(["env", f"HOME={Path.home()}", f'PATH={os.environ["PATH"]}'])
    body = (
        f"{{ cd {shlex.quote(str(cwd))} && {environment} {shlex.join(launcher)}; }} "
        f">> {shlex.quote(str(log))} 2>&1; rc=$?; "
        f'printf "EXIT=%s\\n" "$rc" >> {shlex.quote(str(log))}; exit "$rc"'
    )
    try:
        command(
            ["tmux", "new-session", "-d", "-s", f"job-{name}", shlex.join(["bash", "-c", body])],
            deadline=deadline,
        )
    except (OSError, TimeoutError, subprocess.SubprocessError):
        note(log, "OPS_ALLOCATION_FAILURE launch submission uncertain")
        raise


def read_json(path: Path) -> dict:
    """Require an object; malformed local evidence never authorizes a retry."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Refused(f"evidence is not an object: {path}")
    return value


# Owner's 2026-10-22 A0 work order explicitly selects PLAN_A100 §3's
# A1 acceptance envelope over the earlier generic 60-minute exemption ceiling.
A1_AUTHORITY_SHA256 = "1c87898be554d8e66788da6f0a09ee7b842a514e530c1d0e1c624bcf58c3962c"


def a1_acceptance_limit(manifest: dict[str, Any], exemption: dict[str, Any]) -> bool:
    """Recognize only the source-bound A1 ruler's authorized 2 h / 20 u envelope."""
    return (
        exemption.get("kind") == "acceptance"
        and exemption.get("authority_sha256") == A1_AUTHORITY_SHA256
        and manifest.get("script") == "research/gpu_sim_spike/colab.sh"
        and manifest.get("env", {}).get("GPU_SIM_SPIKE_ENTRYPOINT") == "a1-ruler"
        and manifest.get("requirements", {}).get("gpu") == "A100"
        and manifest.get("requirements", {}).get("allowed_gpus") == ["A100"]
        and manifest.get("budget", {}).get("unit_cap") == 20
        and manifest.get("a1", {}).get("shapes") == [6, 256, 1024]
        and manifest.get("a1", {}).get("mixes") == ["scripted", "frozen", "mixed"]
        and manifest.get("a1", {}).get("repeats") == 3
        and manifest.get("a1", {}).get("ticks_per_repeat") == 8
    )


def require_resume_policy(manifest: dict[str, Any]) -> None:
    """Require a resume contract or a declared, bounded operational exemption."""
    if "resume" in manifest and "resume_exemption" in manifest:
        raise Refused("declare resume or resume_exemption, never both")
    if "resume_exemption" in manifest:
        exemption = manifest["resume_exemption"]
        if not isinstance(exemption, dict) or exemption.get("kind") not in (
            "short-canary",
            "acceptance",
        ):
            raise Refused("resume_exemption.kind must be short-canary or acceptance")
        reason = exemption.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise Refused("resume_exemption.reason must be nonempty")
        timeout = manifest.get("timeout_minutes")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise Refused("resume exemption requires a finite positive timeout_minutes")
        # Owner's 2x rule: ruling's 20-minute canary; acceptance's 30-minute maximum.
        limit = 40 if exemption["kind"] == "short-canary" else 60
        if a1_acceptance_limit(manifest, exemption):
            limit = 120
        if timeout > limit:
            raise Refused(
                f"{exemption['kind']} resume exemption requires timeout_minutes <= {limit}"
            )
        return
    resume = manifest.get("resume")
    if not isinstance(resume, dict):
        raise Refused("manifest requires a resume block (or declared resume_exemption)")
    for field in ("unit_kind", "ledger_path"):
        if not isinstance(resume.get(field), str) or not resume[field].strip():
            raise Refused(f"resume.{field} must be nonempty")
    if resume.get("skip_verified") is not True:
        raise Refused("resume.skip_verified must be true")
    retries = resume.get("max_outcome_blind_retries")
    if type(retries) is not int or retries != 2:
        raise Refused("resume.max_outcome_blind_retries must be the integer 2")


def wait_allocation(output: Path, log: Path, deadline: float, poll: float) -> str:
    """An assignment record is too early: accounting calibration must finish."""
    while time.monotonic() < deadline:
        calibration = output / "rate-calibration.json"
        if calibration.is_file() and read_json(calibration).get("status") == "PASS":
            note(log, f"OPS_ALLOCATION_SUCCESS calibration={calibration}")
            return "success"
        receipt = output / "receipt.json"
        if receipt.is_file():
            value = read_json(receipt)
            retry = (
                value.get("status") == "INCOMPLETE"
                and value.get("failure_class") == "ACCOUNTING"
                and value.get("error") == EXACT_ERROR
                and value.get("cleanup_error") is None
                and value.get("stop_verified") is True
            )
            # Receipt publication precedes shell exit; wait before a retry.
            exited = bool(re.search(r"^EXIT=\d+$", log.read_text(), re.MULTILINE))
            if retry and not exited:
                time.sleep(min(poll, max(0, deadline - time.monotonic())))
                continue
            note(log, "OPS_ALLOCATION_FAILURE_RETRYABLE" if retry else "OPS_ALLOCATION_FAILURE")
            return "retry" if retry else "failed"
        if re.search(r"^EXIT=\d+$", log.read_text(), re.MULTILINE):
            note(log, "OPS_ALLOCATION_FAILURE no-terminal-receipt")
            return "failed"
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    note(log, "OPS_ALLOCATION_WAIT_TIMEOUT")
    return "timeout"


def ensure_watcher(ops: Path, broker: Path) -> None:
    """Start the watcher outside the allocation critical section."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", "watchd"],
        timeout=10,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        command(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                "watchd",
                shlex.join(
                    ["env", f"OPS_DIR={ops}", f"OPS_REPO={broker}", str(HERE / "watchd.sh")]
                ),
            ]
        )


def dispatch(args: argparse.Namespace) -> str:
    """Validate first, then serialize one allocation and at most one exact retry."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        raise Refused("NAME must contain only letters, digits, dot, underscore, hyphen")
    ops = Path(os.environ.get("OPS_DIR", OPS)).resolve()
    manifest, broker, cwd = args.manifest.resolve(), args.broker.resolve(), args.cwd.resolve()
    state = (args.state_dir or ops / "state" / args.name).resolve()
    dispatch_txt = args.dispatch_txt or manifest.parent / "DISPATCH.txt"
    if dispatch_txt.is_file():
        blocked = [
            line
            for line in dispatch_txt.read_text().splitlines()
            if BLOCK.search(line) and "live inventory" not in line.lower()
        ]
        if blocked:
            raise Refused(f"{dispatch_txt} has BLOCK/HOLD instructions: {blocked}")
    original_bytes = manifest.read_bytes()
    value = json.loads(original_bytes)
    if not isinstance(value, dict):
        raise Refused("manifest must be an object")
    require_resume_policy(value)
    if value.get("scientific_admission") is False:
        raise Refused("manifest scientific_admission=false")
    if not isinstance(value.get("budget", {}).get("other_assignments", {}), dict):
        raise Refused("budget.other_assignments must be a map")
    output = value.get("output_dir")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise Refused("manifest output_dir must be absolute")
    output = str(Path(output).resolve())
    if not isinstance(value.get("job_id"), str) or not value["job_id"]:
        raise Refused("manifest job_id must be nonempty")
    for subdir in ("logs", "state", "manifests", "dispatch"):
        (ops / subdir).mkdir(parents=True, exist_ok=True)
    commit = value.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refused("manifest commit must be a full published SHA")
    qualified(broker, ops)
    qualified(broker, ops, ref=commit)
    # Create-only name reservation also fences two clients dispatched concurrently.
    (ops / "dispatch" / args.name).mkdir()
    log = ops / "logs" / f"{args.name}.log"
    if log.exists():
        raise Refused(f"log already exists (name reused): {log}")
    manifest = ops / "manifests" / f"{args.name}.json"
    with manifest.open("xb") as stream:
        stream.write(original_bytes)
    validate(broker, cwd, state, manifest, ops / "logs" / f"{args.name}.validate")
    seconds = float(os.environ.get("OPS_ALLOC_LOCK_TIMEOUT_SECONDS", "600"))
    poll = float(os.environ.get("OPS_ALLOC_POLL_INTERVAL_SECONDS", "1"))
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise Refused("allocation timeout must be positive and at most 600 seconds")
    if not math.isfinite(poll) or poll <= 0:
        raise Refused("allocation poll interval must be positive and finite")
    expected = value
    with allocation_lock(ops / "alloc.lock", seconds) as deadline:
        for attempt in range(2):
            name = args.name if attempt == 0 else f"{args.name}-retry-1"
            if read_json(manifest) != expected:
                raise Refused("manifest changed after validation")
            qualified(broker, ops, deadline)
            qualified(broker, ops, deadline, ref=commit)
            start(broker, cwd, state, manifest, log, name, deadline)
            outcome = wait_allocation(Path(output), log, deadline, poll)
            if attempt == 1 and outcome == "retry":
                note(log, "OPS_ALLOCATION_RETRY_EXHAUSTED")
                outcome = "failed"
            if outcome != "retry":
                break
            retry_id = value["job_id"] + "-retry-1"
            if len(retry_id) > 64:
                digest = hashlib.sha256(value["job_id"].encode()).hexdigest()[:8]
                retry_id = f'{value["job_id"][:47]}-{digest}-retry-1'
            retry = dict(value, output_dir=f"{output}-retry-1", job_id=retry_id)
            expected = retry
            manifest = ops / "manifests" / f"{name}-retry-1.json"
            if Path(retry["output_dir"]).exists():
                raise Refused("retry output_dir already exists")
            with manifest.open("x") as stream:
                json.dump(retry, stream, indent=2, sort_keys=True)
                stream.write("\n")
            output = retry["output_dir"]
            state = Path(f"{state}-retry-1")
            log = ops / "logs" / f"{args.name}-retry-1.log"
            note(
                ops / "logs" / f"{args.name}.log",
                f"OPS_ALLOCATION_RETRY manifest={manifest} output_dir={output} log={log}",
            )
            validate(
                broker,
                cwd,
                state,
                manifest,
                ops / "logs" / f"{args.name}-retry-1.validate",
                deadline,
            )
    ensure_watcher(ops, broker)
    return f"DISPATCHED {args.name} allocation={outcome} log={log}"


def main() -> int:
    """Keep CLI compatible with dispatch.sh's existing interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dispatch-txt", type=Path)
    parser.add_argument(
        "--broker", type=Path, default=Path("/Users/josenunez/Projects/ml/snake-dqn")
    )
    parser.add_argument("--cwd", type=Path, default=Path("/Users/josenunez/Projects/ml/snake-dqn"))
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    try:
        print(dispatch(args))
    except (OSError, ValueError, TypeError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
