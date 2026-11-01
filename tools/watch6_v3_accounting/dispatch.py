"""Fail-closed serialized-staging Watch6 V3 dispatcher adapter wrapping tools/ops/dispatch.sh.

Preserves all Watch6 V3 qualified dispatcher checks, staging locks (`watch6-staging.lock`),
reconciliation fences (`watch6-staging-reconcile.json`), canary verification, L4-only W4
admission, and actual-byte broker closure verification bound to
``/Users/josenunez/Projects/ml/snake-dqn-argon-v3-confirm`` at ``357e3504`` (refusing the main
checkout and any cross-root substitution) while recognizing PROVEN never-allocated attempts
(Ruling Point 4) and requiring Astra's re-ratified accounting receipt before live dispatch.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tools.watch6_v3_accounting import (
    ADAPTER_ROOT,
    V3_BROKER_ROOT,
    bind_broker_protocol,
    ensure_cli_dash_b,
    isolated_broker_execution,
)

# isort: split

from research.watch6_search_efficacy_v3 import dispatch as v3_dispatch
from tools.ops.package_spec import digest
from tools.watch6_v3_accounting.admission import admit
from tools.watch6_v3_accounting.proof import (
    FORBIDDEN_OUTPUT_ENTRIES,
    has_forbidden_or_contradictory_output_artifacts,
    is_proven_never_allocated_attempt,
    verify_no_receipt_unallocated_proof,
)
from tools.watch6_v3_accounting.ratify import (
    validate_ruling_receipt,
    verify_amendment_document,
)

DEFAULT_OPERATOR_ROOT = v3_dispatch.DEFAULT_OPERATOR_ROOT
DEFAULT_OPS_DIR = v3_dispatch.DEFAULT_OPS_DIR
STAGING_WAIT_TIMEOUT_SECONDS = v3_dispatch.STAGING_WAIT_TIMEOUT_SECONDS
RECONCILE_MARKER_NAME = v3_dispatch.RECONCILE_MARKER_NAME
FALLBACK_SUFFIX = v3_dispatch.FALLBACK_SUFFIX
RETRY_SUFFIX = v3_dispatch.RETRY_SUFFIX
ALLOCATION_UNAVAILABLE = v3_dispatch.ALLOCATION_UNAVAILABLE
POST_ALLOCATION_ARTIFACTS = FORBIDDEN_OUTPUT_ENTRIES

_fallback_sibling_name = v3_dispatch._fallback_sibling_name
_read_regular_json_object = v3_dispatch._read_regular_json_object
_log_has_exit = v3_dispatch._log_has_exit
_has_upload_measurement = v3_dispatch._has_upload_measurement
_is_aborted_prestart_dispatch = v3_dispatch._is_aborted_prestart_dispatch
_load_ops_manifest = v3_dispatch._load_ops_manifest
_collect_attempts = v3_dispatch._collect_attempts
_collect_own_attempts = v3_dispatch._collect_own_attempts
_parse_reconcile_ops_dir = v3_dispatch._parse_reconcile_ops_dir


def _has_post_allocation_artifacts(out_dir: Path) -> bool:
    return has_forbidden_or_contradictory_output_artifacts(out_dir)


def _has_stopped_or_unassigned_receipt(receipt_path: Path) -> bool:
    if not v3_dispatch._has_stopped_or_unassigned_receipt(receipt_path):
        return False
    receipt = _read_regular_json_object(receipt_path)
    if receipt is None:
        return False
    if receipt.get("stop_verified") is True:
        return True
    job_id = receipt.get("job_id")
    session = receipt.get("session")
    return not has_forbidden_or_contradictory_output_artifacts(
        receipt_path.parent,
        job_id=job_id if isinstance(job_id, str) else None,
        session=session if isinstance(session, str) else None,
    )


def _has_live_unallocated_proof(ops_dir: Path, name: str, out_dir: Path) -> bool:
    receipt_path = out_dir / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        return is_proven_never_allocated_attempt(None, out_dir, ops_dir=ops_dir, ops_name=name)
    if _has_post_allocation_artifacts(out_dir):
        return False
    disp_dir = ops_dir / "dispatch" / name
    raw_candidates = [
        disp_dir / "unavailable.json",
        *(sorted(disp_dir.glob("unallocated*.json")) if disp_dir.is_dir() else []),
        *sorted(ops_dir.glob(f"unallocated-{name}*.json")),
    ]
    existing: list[Path] = []
    seen: set[Path] = set()
    for cand in raw_candidates:
        if cand.is_symlink():
            return False
        if cand.exists():
            if not cand.is_file():
                return False
            key = cand.resolve()
            if key not in seen:
                seen.add(key)
                existing.append(cand)
    if not existing:
        return False
    if len({digest(c) for c in existing}) != 1:
        return False
    return all(verify_no_receipt_unallocated_proof(cand, out_dir) for cand in existing)


def _attempt_state(
    ops_dir: Path,
    name: str,
    manifest_file: Path | None,
    data: dict[str, object],
    out_dir: Path,
    log_path: Path,
) -> str:
    del data
    receipt_path = out_dir / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        if _has_stopped_or_unassigned_receipt(receipt_path):
            return "DONE"
        if is_proven_never_allocated_attempt(None, out_dir, ops_dir=ops_dir, ops_name=name):
            return "DONE"
        return "FENCE"
    if _has_upload_measurement(out_dir):
        return "DONE"
    if _has_live_unallocated_proof(ops_dir, name, out_dir):
        return "DONE"
    if _log_has_exit(log_path):
        if not out_dir.exists() and not out_dir.is_symlink():
            return "DONE"
        return "FENCE"
    if manifest_file is not None and _is_aborted_prestart_dispatch(
        manifest_file, ops_dir, name, out_dir
    ):
        return "DONE"
    return "STAGING"


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
            f"python -m tools.watch6_v3_accounting.dispatch "
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
            "`python -m tools.watch6_v3_accounting.dispatch "
            f"--reconcile-staging --ops-dir {ops_dir}`"
        )


def job_is_actively_staging(manifest_file: Path, ops_dir: Path) -> bool:
    """True iff an ops/manifests/<name>.json job (or its sibling) is inside its staging window."""
    if manifest_file.name.endswith(".live.json"):
        return False
    attempts = _collect_attempts(manifest_file, ops_dir)
    any_staging = False
    for attempt_name, attempt_mf, attempt_data, attempt_out, attempt_log in attempts:
        state = _attempt_state(
            ops_dir, attempt_name, attempt_mf, attempt_data, attempt_out, attempt_log
        )
        if state == "FENCE":
            fence = _write_reconcile_fence(
                ops_dir,
                reason="exited_manifest_missing_staging_or_stop_evidence",
                job_name=manifest_file.stem,
                details={
                    "manifest": str(manifest_file),
                    "attempt": attempt_name,
                    "output_dir": str(attempt_out),
                },
            )
            raise RuntimeError(
                f"REFUSE: exited ops manifest {manifest_file.name} left unreconciled output "
                f"{attempt_out} without upload-measurement.json or stopped receipt.json "
                f"(fence: {fence})"
            )
        if state == "STAGING":
            any_staging = True
    return any_staging


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
        attempts = _collect_own_attempts(ops_dir, job_name, raw_output_dir, log_path)
        staging_attempts: list[tuple[Path, Path]] = []
        for attempt_name, attempt_mf, attempt_data, attempt_out, attempt_log in attempts:
            state = _attempt_state(
                ops_dir, attempt_name, attempt_mf, attempt_data, attempt_out, attempt_log
            )
            if state == "FENCE":
                fence = _write_reconcile_fence(
                    ops_dir,
                    reason="own_job_exited_without_staging_or_stop_evidence",
                    job_name=job_name,
                    details={
                        "attempt": attempt_name,
                        "output_dir": str(attempt_out),
                        "log_path": str(attempt_log),
                    },
                )
                raise RuntimeError(
                    f"REFUSE: own job {job_name} exited during staging without "
                    f"upload-measurement.json or stopped receipt.json ({attempt_out}); "
                    f"reconcile required (fence: {fence})"
                )
            if state == "STAGING":
                staging_attempts.append((attempt_out, attempt_log))
        if not staging_attempts:
            return
        if time.monotonic() >= deadline:
            active_out, active_log = staging_attempts[-1]
            fence = _write_reconcile_fence(
                ops_dir,
                reason="wait_own_job_staged_expired",
                job_name=job_name,
                details={
                    "timeout_seconds": timeout,
                    "output_dir": str(active_out),
                    "log_path": str(active_log),
                },
            )
            raise RuntimeError(
                f"REFUSE: own staging wait expired after {timeout:g}s for {job_name} "
                f"({active_out}); reconcile runtime before any further dispatch (fence: {fence})"
            )
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))


def reconcile_staging(ops_dir: Path) -> dict[str, object]:
    """Verify under watch6-staging.lock that zero manifests remain staging, then clear fence."""
    with isolated_broker_execution("dispatch-reconcile"):
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
            had_fence = fence_path.exists() or fence_path.is_symlink()
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
    *,
    default_broker_root: Path = V3_BROKER_ROOT,
) -> tuple[str, Path, list[str], bool, Path, Path | None, Path | None, Path | None, Path, bool]:
    """Parse adapter options and bind --broker/--cwd strictly to one verified V3 broker root."""
    operator_override: Path | None = None
    ruling_override: Path | None = None
    amendment_override: Path | None = None
    evidence_override: Path | None = None
    broker_candidates: list[tuple[str, Path]] = []
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
        if arg == "--amendment":
            if idx + 1 >= len(argv):
                raise ValueError("--amendment requires a path")
            amendment_override = Path(argv[idx + 1]).resolve()
            idx += 2
            continue
        if arg == "--evidence-dir":
            if idx + 1 >= len(argv):
                raise ValueError("--evidence-dir requires a path")
            raw_ev = Path(argv[idx + 1])
            if raw_ev.is_symlink():
                raise ValueError(f"REFUSE: symlinked --evidence-dir forbidden: {raw_ev}")
            evidence_override = raw_ev.resolve()
            idx += 2
            continue
        if arg == "--broker-root":
            if idx + 1 >= len(argv):
                raise ValueError("--broker-root requires a path")
            raw_br = Path(argv[idx + 1])
            if raw_br.is_symlink():
                raise ValueError(f"REFUSE: symlinked --broker-root forbidden: {raw_br}")
            broker_candidates.append(("--broker-root", raw_br.resolve()))
            idx += 2
            continue
        filtered.append(arg)
        idx += 1

    dry_run = "--dry-run" in filtered
    remaining_flags: list[str] = []
    positionals: list[str] = []
    idx = 0
    while idx < len(filtered):
        token = filtered[idx]
        if token in ("--broker", "--cwd"):
            if idx + 1 >= len(filtered):
                raise ValueError(f"{token} requires a path")
            raw_p = Path(filtered[idx + 1])
            if raw_p.is_symlink():
                raise ValueError(f"REFUSE: symlinked {token} forbidden: {raw_p}")
            broker_candidates.append((token, raw_p.resolve()))
            idx += 2
            continue
        if token in ("--state-dir", "--dispatch-txt"):
            if idx + 1 >= len(filtered):
                raise ValueError(f"{token} requires a path")
            remaining_flags.extend([token, filtered[idx + 1]])
            idx += 2
            continue
        if token.startswith("-"):
            remaining_flags.append(token)
            idx += 1
            continue
        positionals.append(token)
        idx += 1

    if broker_candidates:
        distinct_brokers = {resolved for _, resolved in broker_candidates}
        if len(distinct_brokers) != 1:
            formatted = ", ".join(f"{flag}={path}" for flag, path in broker_candidates)
            raise ValueError(
                "REFUSE: cross-root substitution forbidden; conflicting broker/cwd roots: "
                f"{formatted}"
            )
        effective_broker = broker_candidates[0][1]
        explicit_broker_given = True
    else:
        if default_broker_root.is_symlink():
            raise ValueError(f"REFUSE: symlinked V3 broker root forbidden: {default_broker_root}")
        effective_broker = default_broker_root.resolve()
        explicit_broker_given = False

    if len(positionals) == 1 and Path(positionals[0]).suffix == ".json":
        manifest_path = Path(positionals[0]).resolve()
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = str(manifest_data["job_id"])
        extra_positionals: list[str] = []
    elif len(positionals) >= 2:
        name = positionals[0]
        manifest_path = Path(positionals[1]).resolve()
        extra_positionals = positionals[2:]
    else:
        raise ValueError("usage: dispatch.sh <NAME> <MANIFEST> --broker ... --cwd ... [--dry-run]")

    forward = [
        name,
        str(manifest_path),
        *extra_positionals,
        "--broker",
        str(effective_broker),
        "--cwd",
        str(effective_broker),
        *remaining_flags,
    ]

    if operator_override is not None:
        operator_root = operator_override
    elif os.environ.get("WATCH6_OPERATOR_ROOT"):
        operator_root = Path(os.environ["WATCH6_OPERATOR_ROOT"]).resolve()
    elif (DEFAULT_OPERATOR_ROOT / "tools/ops/dispatch.sh").is_file():
        operator_root = DEFAULT_OPERATOR_ROOT.resolve()
    else:
        operator_root = effective_broker
    return (
        name,
        manifest_path,
        forward,
        dry_run,
        operator_root,
        ruling_override,
        amendment_override,
        evidence_override,
        effective_broker,
        explicit_broker_given,
    )


def run_dispatch(
    argv: list[str],
    *,
    ops_dir: Path | None = None,
    staging_timeout: float = STAGING_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = 1.0,
    repo: Path | None = None,
    broker_root: Path | None = None,
) -> int:
    """Execute actual-byte broker check + V3 admission + fail-closed staging around dispatch.sh."""
    if broker_root is not None and broker_root.is_symlink():
        raise ValueError(f"REFUSE: symlinked broker_root forbidden: {broker_root}")
    default_broker = broker_root if broker_root is not None else V3_BROKER_ROOT

    with isolated_broker_execution("dispatch", broker_root=default_broker) as iso:
        if "--reconcile-staging" in argv:
            resolved = _parse_reconcile_ops_dir(argv, ops_dir)
            print(json.dumps(reconcile_staging(resolved), indent=2), flush=True)
            return 0

        adapter_repo = (repo or ADAPTER_ROOT).resolve()
        (
            name,
            manifest_path,
            forward,
            dry_run,
            operator_root,
            ruling_override,
            amendment_override,
            evidence_override,
            effective_broker,
            explicit_broker_given,
        ) = _parse_cli(argv, default_broker_root=default_broker)

        if (
            broker_root is not None
            and explicit_broker_given
            and broker_root.resolve() != effective_broker
        ):
            raise ValueError(
                "REFUSE: cross-root substitution forbidden; "
                f"broker_root={broker_root.resolve()} != CLI broker/cwd={effective_broker}"
            )

        iso.verify_outside(effective_broker)
        with bind_broker_protocol(effective_broker, require_head_at_baseline=True):
            resolved_ops = (
                ops_dir.resolve()
                if ops_dir is not None
                else Path(os.environ.get("OPS_DIR", str(DEFAULT_OPS_DIR))).resolve()
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if evidence_override is not None:
                admit(
                    manifest,
                    ops_dir=resolved_ops,
                    evidence_dir=evidence_override,
                    broker_root=effective_broker,
                )
            else:
                admit(manifest, ops_dir=resolved_ops, broker_root=effective_broker)

            target_dispatch = operator_root / "tools/ops/dispatch.sh"
            if not target_dispatch.is_file():
                raise ValueError(f"missing operator dispatch.sh: {target_dispatch}")

            child_env = iso.subprocess_env(prepend_wrapper_to_path=dry_run)
            if dry_run:
                if amendment_override is not None:
                    verify_amendment_document(
                        amendment_override,
                        repo=adapter_repo,
                        broker_root=effective_broker,
                    )
                completed = subprocess.run(
                    [str(target_dispatch), *forward],
                    env=child_env,
                    check=False,
                )
                return completed.returncode

            validate_ruling_receipt(
                ruling_override,
                amendment_override=amendment_override,
                repo=adapter_repo,
                broker_root=effective_broker,
            )
            resolved_ops.mkdir(parents=True, exist_ok=True)
            lock_path = resolved_ops / "watch6-staging.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                wait_other_jobs_staged(
                    resolved_ops,
                    name,
                    timeout=staging_timeout,
                    poll_seconds=max(poll_seconds, 0.02),
                )
                completed = subprocess.run(
                    [str(target_dispatch), *forward],
                    env=child_env,
                    check=False,
                )
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
    ensure_cli_dash_b("tools.watch6_v3_accounting.dispatch")
    try:
        rc = run_dispatch(sys.argv[1:])
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(2)
    sys.exit(rc)


if __name__ == "__main__":
    main()
