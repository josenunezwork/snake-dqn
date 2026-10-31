"""Serialized-staging Watch6 V3 dispatcher wrapping qualified operator tools/ops/dispatch.sh.

Enforces three invariants before/around live Colab dispatch without emitting any
extra stdout/stderr lines that would perturb ready_queue.py refusal parsing:
1. V3 package admission (`admission.admit(manifest)`).
2. Pre-data Astra ruling verification (`ratify.validate_ruling_receipt`, live only).
3. Account-wide serialized staging (`<ops_dir>/watch6-staging.lock`, Astra RULING_SHARD6 §2):
   at most one Colab runtime stages/uploads at a time across the account; computing
   shards continue concurrently up to ACCOUNT_ASSIGNMENT_CAP=6.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import admit
from research.watch6_search_efficacy_v3.ratify import validate_ruling_receipt

DEFAULT_OPERATOR_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-transport-hiccup")
DEFAULT_OPS_DIR = p.ROOT.parent / "snake-dqn-artifacts/ops"
STAGING_WAIT_TIMEOUT_SECONDS = 420.0
STAGING_STALE_MANIFEST_SECONDS = 600.0
EXIT_RE = re.compile(r"^EXIT=\d+\s*$", re.MULTILINE)


def _active_output_dir(raw_output_dir: str) -> Path:
    retry_dir = Path(raw_output_dir + "-retry-1")
    return retry_dir if retry_dir.exists() else Path(raw_output_dir)


def _log_has_exit(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    try:
        tail = log_path.read_text(errors="replace")[-4096:]
    except OSError:
        return False
    return bool(EXIT_RE.search(tail))


def job_is_actively_staging(
    manifest_file: Path, ops_dir: Path, *, now: float | None = None
) -> bool:
    """True iff an ops/manifests/<name>.json job is currently inside its staging window."""
    if manifest_file.name.endswith(".live.json") or not manifest_file.is_file():
        return False
    now = time.time() if now is None else now
    try:
        stat = manifest_file.stat()
        if now - stat.st_mtime > STAGING_STALE_MANIFEST_SECONDS:
            return False
        data = json.loads(manifest_file.read_text())
    except (OSError, ValueError):
        return False
    raw_out = data.get("output_dir")
    if not isinstance(raw_out, str) or not raw_out:
        return False
    out = _active_output_dir(raw_out)
    if (out / "upload-measurement.json").is_file() or (out / "receipt.json").is_file():
        return False
    name = manifest_file.stem
    if _log_has_exit(ops_dir / "logs" / f"{name}.log"):
        return False
    return True


def wait_other_jobs_staged(
    ops_dir: Path,
    exclude_name: str,
    *,
    timeout: float = STAGING_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = 2.0,
) -> None:
    """Block silently until no other active job in ops/manifests is still staging."""
    manifests_dir = ops_dir / "manifests"
    if not manifests_dir.is_dir():
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        busy = [
            mf
            for mf in sorted(manifests_dir.glob("*.json"))
            if mf.stem != exclude_name and job_is_actively_staging(mf, ops_dir)
        ]
        if not busy:
            return
        time.sleep(poll_seconds)


def wait_own_job_staged(
    raw_output_dir: str,
    log_path: Path,
    *,
    timeout: float = STAGING_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = 1.0,
) -> None:
    """Hold watch6-staging.lock silently until this runtime completes staging or exits."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = _active_output_dir(raw_output_dir)
        if (
            (out / "upload-measurement.json").is_file()
            or (out / "receipt.json").is_file()
            or _log_has_exit(log_path)
        ):
            return
        time.sleep(poll_seconds)


def _parse_cli(
    argv: list[str],
) -> tuple[str, Path, list[str], bool, Path, Path | None]:
    """Parse V3 options while preserving exact tools/ops/dispatch.sh pass-through flags."""
    operator_override: Path | None = None
    ruling_override: Path | None = None
    filtered: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--operator-root":
            if idx + 1 >= len(argv):
                raise ValueError("--operator-root requires a path")
            operator_override = Path(argv[idx + 1]).resolve()
            idx += 2
            continue
        if arg == "--ruling-json":
            if idx + 1 >= len(argv):
                raise ValueError("--ruling-json requires a path")
            ruling_override = Path(argv[idx + 1]).resolve()
            idx += 2
            continue
        filtered.append(arg)
        idx += 1

    dry_run = "--dry-run" in filtered
    positionals = []
    idx = 0
    while idx < len(filtered):
        token = filtered[idx]
        if token in ("--broker", "--cwd", "--state-dir", "--dispatch-txt"):
            idx += 2
            continue
        if token.startswith("-"):
            idx += 1
            continue
        positionals.append(token)
        idx += 1

    if len(positionals) == 1 and Path(positionals[0]).suffix == ".json":
        manifest_path = Path(positionals[0]).resolve()
        manifest_data = json.loads(manifest_path.read_text())
        name = str(manifest_data["job_id"])
        flags = [a for a in filtered if a != positionals[0]]
        forward = [
            name,
            str(manifest_path),
            "--broker",
            str(p.ROOT),
            "--cwd",
            str(p.ROOT),
            *flags,
        ]
    elif len(positionals) >= 2:
        name = positionals[0]
        manifest_path = Path(positionals[1]).resolve()
        forward = list(filtered)
    else:
        raise ValueError("usage: dispatch.sh <NAME> <MANIFEST> --broker ... --cwd ... [--dry-run]")

    if operator_override is not None:
        operator_root = operator_override
    elif os.environ.get("WATCH6_OPERATOR_ROOT"):
        operator_root = Path(os.environ["WATCH6_OPERATOR_ROOT"]).resolve()
    elif (DEFAULT_OPERATOR_ROOT / "tools/ops/dispatch.sh").is_file():
        operator_root = DEFAULT_OPERATOR_ROOT.resolve()
    else:
        operator_root = p.ROOT
    return name, manifest_path, forward, dry_run, operator_root, ruling_override


def run_dispatch(
    argv: list[str],
    *,
    ops_dir: Path | None = None,
    staging_timeout: float = STAGING_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = 1.0,
) -> int:
    """Execute V3 admission + serialized staging around operator tools/ops/dispatch.sh."""
    name, manifest_path, forward, dry_run, operator_root, ruling_override = _parse_cli(argv)
    manifest = json.loads(manifest_path.read_text())
    admit(manifest)
    target_dispatch = operator_root / "tools/ops/dispatch.sh"
    if not target_dispatch.is_file():
        raise ValueError(f"missing operator dispatch.sh: {target_dispatch}")

    if dry_run:
        completed = subprocess.run([str(target_dispatch), *forward], check=False)
        return completed.returncode

    validate_ruling_receipt(ruling_override)
    resolved_ops = (
        ops_dir.resolve()
        if ops_dir is not None
        else Path(os.environ.get("OPS_DIR", str(DEFAULT_OPS_DIR))).resolve()
    )
    resolved_ops.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_ops / "watch6-staging.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        wait_other_jobs_staged(
            resolved_ops, name, timeout=staging_timeout, poll_seconds=max(poll_seconds, 0.05)
        )
        completed = subprocess.run([str(target_dispatch), *forward], check=False)
        if completed.returncode == 0:
            wait_own_job_staged(
                manifest["output_dir"],
                resolved_ops / "logs" / f"{name}.log",
                timeout=staging_timeout,
                poll_seconds=poll_seconds,
            )
        return completed.returncode
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def main() -> None:
    try:
        rc = run_dispatch(sys.argv[1:])
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(2)
    sys.exit(rc)


if __name__ == "__main__":
    main()
