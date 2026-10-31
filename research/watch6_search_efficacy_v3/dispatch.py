"""Fail-closed serialized-staging Watch6 V3 dispatcher wrapping tools/ops/dispatch.sh.

Enforces three invariants before/around live Colab dispatch:
1. V3 package admission (`admission.admit(manifest)`).
2. Pre-data Astra ruling verification (`ratify.validate_ruling_receipt`, live only).
3. Fail-CLOSED account-wide serialized staging (`<ops_dir>/watch6-staging.lock`):
   - Never silently ignores stale (>600s) or corrupt manifests.
   - On wait expiry (either waiting for prior staging jobs or waiting for own job to
     finish staging), writes durable `<ops_dir>/watch6-staging-reconcile.json` and
     refuses closed (`exit 2`), blocking any subsequent dispatch until reconciled via
     `python -m research.watch6_search_efficacy_v3.dispatch --reconcile-staging`.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import admit
from research.watch6_search_efficacy_v3.ratify import validate_ruling_receipt

DEFAULT_OPERATOR_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-transport-hiccup")
DEFAULT_OPS_DIR = p.ROOT.parent / "snake-dqn-artifacts/ops"
# Covers 180s warmup + 150s rate calibration + 300s staging cap + 90s stop margin (< 4500s cap).
STAGING_WAIT_TIMEOUT_SECONDS = 720.0
RECONCILE_MARKER_NAME = "watch6-staging-reconcile.json"
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


def _has_stopped_or_unassigned_receipt(receipt_path: Path) -> bool:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(receipt, dict) or receipt.get("cleanup_error") is not None:
        return False
    if receipt.get("stop_verified") is True:
        return True
    return receipt.get("endpoint") is None and receipt.get("stop_verified") is False


def _has_live_unallocated_proof(ops_dir: Path, name: str, out_dir: Path) -> bool:
    if (out_dir / "receipt.json").exists():
        return False
    candidates = [
        ops_dir / "dispatch" / name / "unavailable.json",
        *sorted(ops_dir.glob(f"unallocated-{name}*.json")),
    ]
    for proof_path in candidates:
        if not proof_path.is_file():
            continue
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(proof, dict)
            and proof.get("schema") == "colab-allocation-unavailable/v1"
            and proof.get("mode") == "LIVE"
            and proof.get("endpoint_ever_assigned") is False
            and proof.get("vm_compute_started") is False
        ):
            return True
    return False


def _write_reconcile_fence(
    ops_dir: Path, *, reason: str, job_name: str, details: dict[str, object]
) -> Path:
    ops_dir.mkdir(parents=True, exist_ok=True)
    fence_path = ops_dir / RECONCILE_MARKER_NAME
    payload = {
        "schema": "watch6-v3-staging-reconcile-fence/v1",
        "created_at": time.time(),
        "reason": reason,
        "job_name": job_name,
        "details": details,
        "reconcile_command": (
            f"python -m research.watch6_search_efficacy_v3.dispatch "
            f"--reconcile-staging --ops-dir {ops_dir}"
        ),
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ops_dir, prefix=".reconcile-", delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, fence_path)
    return fence_path


def _require_no_reconcile_fence(ops_dir: Path) -> None:
    fence_path = ops_dir / RECONCILE_MARKER_NAME
    if fence_path.exists() or fence_path.is_symlink():
        raise RuntimeError(
            f"REFUSE: unreconciled staging serialization fence present at {fence_path}; "
            "verify all active sessions/staging outputs and run "
            "`python -m research.watch6_search_efficacy_v3.dispatch "
            f"--reconcile-staging --ops-dir {ops_dir}`"
        )


def job_is_actively_staging(manifest_file: Path, ops_dir: Path) -> bool:
    """True iff an ops/manifests/<name>.json job is currently inside its staging window.

    Never ignores old manifests by mtime. Corrupt manifests or exited jobs whose output
    directory exists without upload-measurement.json, stop-verified receipt.json, or
    LIVE unallocated proof fail CLOSED immediately.
    """
    if manifest_file.name.endswith(".live.json"):
        return False
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise RuntimeError(
            f"REFUSE: unsafe or unreadable ops manifest during staging check: {manifest_file}"
        )
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"REFUSE: corrupt ops manifest blocks staging serialization: {manifest_file}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"REFUSE: non-object ops manifest blocks staging: {manifest_file}")
    raw_out = data.get("output_dir")
    if not isinstance(raw_out, str) or not raw_out.strip():
        raise RuntimeError(
            f"REFUSE: ops manifest missing output_dir blocks staging: {manifest_file}"
        )

    name = manifest_file.stem
    separate_retry_manifest = (ops_dir / "manifests" / f"{name}-retry-1.json").is_file()
    out = Path(raw_out) if separate_retry_manifest else _active_output_dir(raw_out)

    if (out / "upload-measurement.json").is_file():
        return False
    if _has_stopped_or_unassigned_receipt(out / "receipt.json"):
        return False
    if _has_live_unallocated_proof(ops_dir, name, out):
        return False

    log_exited = _log_has_exit(ops_dir / "logs" / f"{name}.log")
    if log_exited:
        any_dir_created = Path(raw_out).exists() or (
            not separate_retry_manifest and Path(raw_out + "-retry-1").exists()
        )
        if not any_dir_created:
            return False
        fence = _write_reconcile_fence(
            ops_dir,
            reason="exited_manifest_missing_staging_or_stop_evidence",
            job_name=name,
            details={"manifest": str(manifest_file), "output_dir": str(out)},
        )
        raise RuntimeError(
            f"REFUSE: exited ops manifest {manifest_file.name} left unreconciled output {out} "
            f"without upload-measurement.json or stopped receipt.json (fence: {fence})"
        )
    return True


def wait_other_jobs_staged(
    ops_dir: Path,
    exclude_name: str,
    *,
    timeout: float = STAGING_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = 2.0,
) -> None:
    """Wait until zero other jobs are staging; on expiry fail CLOSED and write reconcile fence."""
    _require_no_reconcile_fence(ops_dir)
    manifests_dir = ops_dir / "manifests"
    if not manifests_dir.is_dir():
        return
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        busy = [
            mf
            for mf in sorted(manifests_dir.glob("*.json"))
            if mf.stem != exclude_name and job_is_actively_staging(mf, ops_dir)
        ]
        if not busy:
            return
        if time.monotonic() >= deadline:
            blocking = [mf.name for mf in busy]
            fence = _write_reconcile_fence(
                ops_dir,
                reason="wait_other_jobs_staged_expired",
                job_name=exclude_name,
                details={
                    "timeout_seconds": timeout,
                    "blocking_manifests": blocking,
                },
            )
            raise RuntimeError(
                f"REFUSE: staging serialization wait expired after {timeout:g}s "
                f"for {exclude_name}; active/stale staging manifests require "
                f"reconciliation: {blocking} (fence: {fence})"
            )
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))


def wait_own_job_staged(
    ops_dir: Path,
    job_name: str,
    raw_output_dir: str,
    log_path: Path,
    *,
    timeout: float = STAGING_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = 1.0,
) -> None:
    """Hold watch6-staging.lock until own job finishes staging; on expiry fail CLOSED with fence."""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        out = _active_output_dir(raw_output_dir)
        if (
            (out / "upload-measurement.json").is_file()
            or _has_stopped_or_unassigned_receipt(out / "receipt.json")
            or _has_live_unallocated_proof(ops_dir, job_name, out)
        ):
            return
        if _log_has_exit(log_path):
            if not Path(raw_output_dir).exists() and not Path(raw_output_dir + "-retry-1").exists():
                return
            fence = _write_reconcile_fence(
                ops_dir,
                reason="own_job_exited_without_staging_or_stop_evidence",
                job_name=job_name,
                details={"output_dir": str(out), "log_path": str(log_path)},
            )
            raise RuntimeError(
                f"REFUSE: own job {job_name} exited during staging without upload-measurement.json "
                f"or stopped receipt.json ({out}); reconcile required (fence: {fence})"
            )
        if time.monotonic() >= deadline:
            fence = _write_reconcile_fence(
                ops_dir,
                reason="wait_own_job_staged_expired",
                job_name=job_name,
                details={
                    "timeout_seconds": timeout,
                    "output_dir": str(out),
                    "log_path": str(log_path),
                },
            )
            raise RuntimeError(
                f"REFUSE: own staging wait expired after {timeout:g}s for {job_name} ({out}); "
                f"reconcile runtime before any further dispatch (fence: {fence})"
            )
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))


def reconcile_staging(ops_dir: Path) -> dict[str, object]:
    """Verify under watch6-staging.lock that zero manifests remain staging, then clear fence."""
    resolved_ops = ops_dir.resolve()
    resolved_ops.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_ops / "watch6-staging.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        manifests_dir = resolved_ops / "manifests"
        busy: list[str] = []
        checked = 0
        if manifests_dir.is_dir():
            for mf in sorted(manifests_dir.glob("*.json")):
                if mf.name.endswith(".live.json"):
                    continue
                checked += 1
                if job_is_actively_staging(mf, resolved_ops):
                    busy.append(mf.name)
        if busy:
            raise RuntimeError(
                "REFUSE: cannot clear staging reconciliation fence; "
                f"manifests still staging: {busy}"
            )
        fence_path = resolved_ops / RECONCILE_MARKER_NAME
        had_fence = fence_path.exists()
        fence_path.unlink(missing_ok=True)
        return {
            "status": "RECONCILED",
            "ops_dir": str(resolved_ops),
            "checked_manifests": checked,
            "cleared_fence": had_fence,
        }
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


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
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    """Execute V3 admission + fail-closed serialized staging around tools/ops/dispatch.sh."""
    if "--reconcile-staging" in argv:
        target_ops = ops_dir
        if "--ops-dir" in argv:
            pos = argv.index("--ops-dir")
            if pos + 1 >= len(argv):
                raise ValueError("--ops-dir requires a path")
            target_ops = Path(argv[pos + 1])
        resolved = (
            target_ops.resolve()
            if target_ops is not None
            else Path(os.environ.get("OPS_DIR", str(DEFAULT_OPS_DIR))).resolve()
        )
        print(json.dumps(reconcile_staging(resolved), indent=2), flush=True)
        return 0

    name, manifest_path, forward, dry_run, operator_root, ruling_override = _parse_cli(argv)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            resolved_ops, name, timeout=staging_timeout, poll_seconds=max(poll_seconds, 0.02)
        )
        completed = subprocess.run([str(target_dispatch), *forward], check=False)
        if completed.returncode == 0:
            wait_own_job_staged(
                resolved_ops,
                name,
                manifest["output_dir"],
                resolved_ops / "logs" / f"{name}.log",
                timeout=staging_timeout,
                poll_seconds=max(poll_seconds, 0.02),
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
