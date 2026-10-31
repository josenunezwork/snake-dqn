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

import ast
import fcntl
import hashlib
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
FALLBACK_SUFFIX = "-a100-fallback"
RETRY_SUFFIX = "-retry-1"
EXIT_RE = re.compile(r"^EXIT=\d+\s*$", re.MULTILINE)
NONZERO_EXIT_RE = re.compile(r"^EXIT=[1-9]\d*\s*$", re.MULTILINE)
ALLOCATION_UNAVAILABLE = (
    "[colab] Allocation refused (precondition failed). This can mean "
    "too many active sessions, or a temporary usage or capacity "
    "limit for the requested runtime. Run `colab stop` to free up a "
    "session, wait and retry, or try a different accelerator."
)
POST_ALLOCATION_ARTIFACTS = (
    "hardware.json",
    "terminal.json",
    "rate-calibration.json",
    "warmup-measurement.json",
    "upload-measurement.json",
    "source-delivery.json",
    "input-hashes.json",
    "exports",
)


def _fallback_sibling_name(job: str) -> str:
    """Derive the deterministic <=64-char H100->A100 fallback sibling name."""
    if len(job + FALLBACK_SUFFIX) <= 64:
        return job + FALLBACK_SUFFIX
    digest = hashlib.sha256(job.encode("utf-8")).hexdigest()[:8]
    return job[: 64 - len(FALLBACK_SUFFIX) - 9] + "-" + digest + FALLBACK_SUFFIX


def _read_regular_json_object(path: Path) -> dict[str, object] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _log_has_exit(log_path: Path, *, nonzero_only: bool = False) -> bool:
    if not log_path.is_file():
        return False
    try:
        tail = log_path.read_text(errors="replace")[-4096:]
    except OSError:
        return False
    pattern = NONZERO_EXIT_RE if nonzero_only else EXIT_RE
    return bool(pattern.search(tail))


def _has_upload_measurement(out_dir: Path) -> bool:
    return _read_regular_json_object(out_dir / "upload-measurement.json") is not None


def _has_stopped_or_unassigned_receipt(receipt_path: Path) -> bool:
    receipt = _read_regular_json_object(receipt_path)
    if receipt is None or receipt.get("cleanup_error") is not None:
        return False
    if receipt.get("stop_verified") is True:
        return True
    accounting = receipt.get("accounting")
    if isinstance(accounting, dict) and accounting.get("allocated_endpoint") is not None:
        return False
    return receipt.get("endpoint") is None and receipt.get("stop_verified") is False


def _has_post_allocation_artifacts(out_dir: Path) -> bool:
    if any(
        (out_dir / name).exists() or (out_dir / name).is_symlink()
        for name in POST_ALLOCATION_ARTIFACTS
    ):
        return True
    return out_dir.is_dir() and any(out_dir.glob("poll-*.json"))


def _has_live_unallocated_proof(ops_dir: Path, name: str, out_dir: Path) -> bool:
    if _has_post_allocation_artifacts(out_dir):
        return False
    receipt_path = out_dir / "receipt.json"
    receipt: dict[str, object] | None = None
    receipt_sha: str | None = None
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _read_regular_json_object(receipt_path)
        if receipt is None:
            return False
        accounting = receipt.get("accounting")
        if (
            receipt.get("endpoint") is not None
            or (isinstance(accounting, dict) and accounting.get("allocated_endpoint") is not None)
            or receipt.get("stop_verified") is not False
        ):
            return False
        try:
            receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        except OSError:
            return False

    candidates = [
        ops_dir / "dispatch" / name / "unavailable.json",
        *sorted(ops_dir.glob(f"unallocated-{name}*.json")),
    ]
    for proof_path in candidates:
        proof = _read_regular_json_object(proof_path)
        if proof is None:
            continue
        if (
            proof.get("schema") != "colab-allocation-unavailable/v1"
            or proof.get("mode") != "LIVE"
            or proof.get("endpoint_ever_assigned") is not False
            or proof.get("vm_compute_started") is not False
        ):
            continue
        if "recovery" in proof and proof.get("recovery") != {"endpoint": None}:
            continue
        if receipt is not None:
            bound = proof.get("receipt")
            bound_path = bound.get("path") if isinstance(bound, dict) else None
            if not isinstance(bound, dict) or not isinstance(bound_path, str) or not bound_path:
                continue
            if (
                Path(bound_path).resolve() != receipt_path.resolve()
                or bound.get("sha256") != receipt_sha
            ):
                continue
        return True
    return False


def _has_preassignment_refusal_receipt(
    manifest_data: dict[str, object], out_dir: Path, log_path: Path
) -> bool:
    """Positive proof that colab new was refused by the server before any endpoint assignment."""
    if not _log_has_exit(log_path, nonzero_only=True) or _has_post_allocation_artifacts(out_dir):
        return False
    receipt_path = out_dir / "receipt.json"
    exec_path = out_dir / "manifest.json"
    receipt = _read_regular_json_object(receipt_path)
    exec_manifest = _read_regular_json_object(exec_path)
    if receipt is None or exec_manifest is None:
        return False
    try:
        exec_sha = hashlib.sha256(exec_path.read_bytes()).hexdigest()
    except OSError:
        return False
    job_id = manifest_data.get("job_id")
    commit = manifest_data.get("commit")
    if (
        receipt.get("manifest_sha256") != exec_sha
        or not isinstance(job_id, str)
        or not job_id
        or receipt.get("job_id") != exec_manifest.get("job_id")
        or exec_manifest.get("job_id") != job_id
        or not isinstance(commit, str)
        or not commit
        or receipt.get("commit") != exec_manifest.get("commit")
        or exec_manifest.get("commit") != commit
    ):
        return False

    req = manifest_data.get("requirements")
    accelerator = req.get("gpu") if isinstance(req, dict) else None
    if not isinstance(accelerator, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", accelerator):
        return False
    session = receipt.get("session")
    if not isinstance(session, str) or not re.fullmatch(r"snake-[a-z0-9_-]+-[a-f0-9]{12}", session):
        return False
    error = receipt.get("error", "")
    if not isinstance(error, str) or not error.startswith("Refused(") or not error.endswith(")"):
        return False
    try:
        detail = ast.literal_eval(error[len("Refused(") : -1])
    except (ValueError, SyntaxError):
        return False
    if not isinstance(detail, str):
        return False
    expected = (
        f"colab new failed (1): [colab] Creating session '{session}'...\n"
        f"{ALLOCATION_UNAVAILABLE}\n"
    )
    service_unavailable = detail.startswith("colab new failed (1): ")
    if service_unavailable:
        lines = detail.removeprefix("colab new failed (1): ").strip().splitlines()
        service_unavailable = (
            re.fullmatch(
                r"ColabRequestError: Failed to issue request POST "
                r"https://colab\.research\.google\.com/tun/m/assign"
                r"\?nbh=[a-f0-9_.-]+&variant=GPU&accelerator=" + re.escape(accelerator) + r": "
                r"(?:Service Unavailable|503(?: Service Unavailable)?)",
                lines[-1] if lines else "",
            )
            is not None
        )
    calls = receipt.get("cli_calls")
    return (
        (detail == expected or service_unavailable)
        and receipt.get("status") == "INCOMPLETE"
        and receipt.get("failure_class") == "SUPERVISOR"
        and receipt.get("endpoint") is None
        and isinstance(receipt.get("accounting"), dict)
        and receipt["accounting"].get("allocated_endpoint") is None
        and receipt.get("terminal") is None
        and receipt.get("cleanup_error") == "ALLOCATION_UNRESOLVED: retained pending record"
        and receipt.get("stop_verified") is False
        and all(receipt.get(k) == [] for k in ("exports", "transfers", "contents_transfers"))
        and receipt.get("control_operations", []) == []
        and isinstance(calls, list)
        and all(isinstance(call, dict) for call in calls)
        and sum(call.get("command") == "new" for call in calls) == 1
        and all(call.get("command") in {"version", "sessions", "usage", "new"} for call in calls)
        and next(call for call in calls if call["command"] == "new").get("status") == "failed"
    )


def _has_subsequent_independent_manifest(manifest_file: Path, ops_dir: Path, name: str) -> bool:
    """True iff a later non-sibling ops manifest proves a subsequent alloc.lock critical section."""
    manifests_dir = ops_dir / "manifests"
    if not manifests_dir.is_dir():
        return False
    try:
        own_mtime_ns = manifest_file.stat().st_mtime_ns
    except OSError:
        return False
    fb_own = _fallback_sibling_name(name)
    retry_own = f"{name}{RETRY_SUFFIX}"
    for other in manifests_dir.glob("*.json"):
        if other.name.endswith(".live.json"):
            continue
        stem = other.stem
        if stem in {name, fb_own, retry_own}:
            continue
        if _fallback_sibling_name(stem) == name or f"{stem}{RETRY_SUFFIX}" == name:
            continue
        try:
            if other.stat().st_mtime_ns > own_mtime_ns:
                return True
        except OSError:
            continue
    return False


def _alloc_lock_is_unlocked(ops_dir: Path) -> bool:
    lock_path = ops_dir / "alloc.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _has_active_preference_primary_lease(ops_dir: Path, fallback_name: str) -> bool:
    if not fallback_name.endswith(FALLBACK_SUFFIX):
        return False
    direct_primary = ops_dir / "dispatch" / fallback_name[: -len(FALLBACK_SUFFIX)]
    if direct_primary.is_dir() and any(direct_primary.glob("*allocation-lease.json")):
        return True
    dispatch_root = ops_dir / "dispatch"
    if not dispatch_root.is_dir():
        return False
    for pref_path in dispatch_root.glob("*/gpu-preference.json"):
        pref = _read_regular_json_object(pref_path)
        variants = pref.get("variants") if isinstance(pref, dict) else None
        if (
            isinstance(variants, list)
            and any(isinstance(v, dict) and v.get("name") == fallback_name for v in variants)
            and any(pref_path.parent.glob("*allocation-lease.json"))
        ):
            return True
    return False


def _is_aborted_prestart_dispatch(
    manifest_file: Path, ops_dir: Path, name: str, out_dir: Path
) -> bool:
    """Positive non-time-based proof that tools/ops/dispatch.py aborted before start()."""
    if out_dir.exists() or out_dir.is_symlink() or (ops_dir / "logs" / f"{name}.log").exists():
        return False
    disp_dir = ops_dir / "dispatch" / name
    if not disp_dir.is_dir():
        base_name = name[: -len(RETRY_SUFFIX)] if name.endswith(RETRY_SUFFIX) else ""
        if (
            base_name
            and (ops_dir / "dispatch" / base_name).is_dir()
            and (ops_dir / "logs" / f"{name}.validate").is_file()
        ):
            disp_dir = ops_dir / "dispatch" / base_name
        else:
            return False
    if any(disp_dir.glob("*allocation-lease.json")) or _has_active_preference_primary_lease(
        ops_dir, name
    ):
        return False
    return _has_subsequent_independent_manifest(
        manifest_file, ops_dir, name
    ) or _alloc_lock_is_unlocked(ops_dir)


def _load_ops_manifest(manifest_file: Path) -> tuple[dict[str, object], str]:
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
    return data, raw_out.rstrip("/")


def _collect_attempts(
    manifest_file: Path, ops_dir: Path
) -> list[tuple[str, Path | None, dict[str, object], Path, Path]]:
    data, raw_out = _load_ops_manifest(manifest_file)
    name = manifest_file.stem
    base_log = ops_dir / "logs" / f"{name}.log"
    attempts: list[tuple[str, Path | None, dict[str, object], Path, Path]] = [
        (name, manifest_file, data, Path(raw_out), base_log)
    ]

    if not name.endswith(RETRY_SUFFIX) and not name.endswith(FALLBACK_SUFFIX):
        retry_name = f"{name}{RETRY_SUFFIX}"
        retry_mf = ops_dir / "manifests" / f"{retry_name}.json"
        retry_out = Path(raw_out + RETRY_SUFFIX)
        retry_log = ops_dir / "logs" / f"{retry_name}.log"
        if retry_mf.exists() or retry_mf.is_symlink():
            r_data, r_raw = _load_ops_manifest(retry_mf)
            attempts.append((retry_name, retry_mf, r_data, Path(r_raw), retry_log))
        elif retry_out.exists() or retry_out.is_symlink() or retry_log.exists():
            eff_log = retry_log if retry_log.exists() else base_log
            attempts.append(
                (retry_name, None, {**data, "output_dir": str(retry_out)}, retry_out, eff_log)
            )

    if not name.endswith(FALLBACK_SUFFIX):
        fb_name = _fallback_sibling_name(name)
        fb_mf = ops_dir / "manifests" / f"{fb_name}.json"
        fb_out = Path(raw_out + FALLBACK_SUFFIX)
        fb_log = ops_dir / "logs" / f"{fb_name}.log"
        if fb_mf.exists() or fb_mf.is_symlink():
            f_data, f_raw = _load_ops_manifest(fb_mf)
            attempts.append((fb_name, fb_mf, f_data, Path(f_raw), fb_log))
        elif fb_out.exists() or fb_out.is_symlink() or fb_log.exists():
            attempts.append((fb_name, None, {**data, "output_dir": str(fb_out)}, fb_out, fb_log))

    return attempts


def _attempt_state(
    ops_dir: Path,
    name: str,
    manifest_file: Path | None,
    data: dict[str, object],
    out_dir: Path,
    log_path: Path,
) -> str:
    if _has_upload_measurement(out_dir):
        return "DONE"
    if _has_stopped_or_unassigned_receipt(out_dir / "receipt.json"):
        return "DONE"
    if _has_live_unallocated_proof(ops_dir, name, out_dir):
        return "DONE"
    if _has_preassignment_refusal_receipt(data, out_dir, log_path):
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
    """True iff an ops/manifests/<name>.json job (or its sibling) is inside its staging window.

    Never ignores old manifests by mtime. Corrupt manifests or exited attempts whose output
    directory exists without upload-measurement.json, stop-verified/unassigned receipt.json,
    bound LIVE unallocated proof, or verified pre-assignment server refusal fail CLOSED.
    """
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


def _collect_own_attempts(
    ops_dir: Path, job_name: str, raw_output_dir: str, log_path: Path
) -> list[tuple[str, Path | None, dict[str, object], Path, Path]]:
    own_mf = ops_dir / "manifests" / f"{job_name}.json"
    if own_mf.exists() or own_mf.is_symlink():
        return _collect_attempts(own_mf, ops_dir)
    clean_raw = raw_output_dir.rstrip("/")
    base_data: dict[str, object] = {"job_id": job_name, "output_dir": clean_raw}
    attempts: list[tuple[str, Path | None, dict[str, object], Path, Path]] = [
        (job_name, None, base_data, Path(clean_raw), log_path)
    ]
    if not job_name.endswith(RETRY_SUFFIX) and not job_name.endswith(FALLBACK_SUFFIX):
        retry_name = f"{job_name}{RETRY_SUFFIX}"
        retry_out = Path(clean_raw + RETRY_SUFFIX)
        retry_log = ops_dir / "logs" / f"{retry_name}.log"
        if retry_out.exists() or retry_out.is_symlink() or retry_log.exists():
            eff_log = retry_log if retry_log.exists() else log_path
            attempts.append(
                (retry_name, None, {**base_data, "output_dir": str(retry_out)}, retry_out, eff_log)
            )
    if not job_name.endswith(FALLBACK_SUFFIX):
        fb_name = _fallback_sibling_name(job_name)
        fb_out = Path(clean_raw + FALLBACK_SUFFIX)
        fb_log = ops_dir / "logs" / f"{fb_name}.log"
        if fb_out.exists() or fb_out.is_symlink() or fb_log.exists():
            attempts.append(
                (fb_name, None, {**base_data, "output_dir": str(fb_out)}, fb_out, fb_log)
            )
    return attempts


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


def _parse_reconcile_ops_dir(argv: list[str], ops_dir: Path | None) -> Path:
    target_ops = ops_dir
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--reconcile-staging":
            idx += 1
            continue
        if arg == "--ops-dir":
            if idx + 1 >= len(argv) or not argv[idx + 1]:
                raise ValueError("--ops-dir requires a path")
            target_ops = Path(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--ops-dir="):
            raw = arg.split("=", 1)[1]
            if not raw:
                raise ValueError("--ops-dir requires a path")
            target_ops = Path(raw)
            idx += 1
            continue
        raise ValueError(f"unexpected argument with --reconcile-staging: {arg}")
    if target_ops is not None:
        return target_ops.resolve()
    return Path(os.environ.get("OPS_DIR", str(DEFAULT_OPS_DIR))).resolve()


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
        resolved = _parse_reconcile_ops_dir(argv, ops_dir)
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
