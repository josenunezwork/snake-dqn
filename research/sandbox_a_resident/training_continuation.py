"""One-command sealed TRAIN-01 continuation packet builder; never dispatches.

Given a completed mechanical-stop attempt directory and its latest completed,
Mac-verified periodic snapshot iteration, this builder:
1. Verifies the terminal receipt's mechanical stop and clean VM release.
2. Verifies the export directory's completion-last snapshot triples (torch-free)
   and requires ``--iteration`` to equal the latest resumable snapshot ``< 384``.
3. Creates (or re-verifies) the lossless ``iteration-NNNN.pt.xz`` stream and
   per-tensor ``.pt.xz.manifest`` derivative with round-trip byte + tensor check.
4. Creates fresh detached ``runtime`` and ``broker`` git worktrees, registers the
   ``(source_packet_id, iteration, original_sha256)`` compact entry in the
   runtime worktree's ``compact_allowlist.json``, and seals a fresh successor
   packet with remaining-update ``2x`` envelopes and identical seed/proxy banks.
5. Runs a 3x local ``background-sparse-v1`` ``< 2.0 s`` launch-exec bound test,
   checks the 18-file whole-staging projection under the ``300 s`` cap, validates
   committed packet bindings, and runs ``tools/ops/dispatch.sh --dry-run``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from research.colab_cli import source
from research.sandbox_a_resident.train import DIAGNOSTIC_PROXY_UPDATE, TrainControl
from research.sandbox_a_resident.training_compact import (
    CompactResumeRefused,
    decode_stream,
    encode_snapshot,
    load_manifest,
    verify_compact_snapshot_export,
    verify_decoded_state,
)
from research.sandbox_a_resident.training_prepare import STUDY, source_paths, write_once
from research.sandbox_a_resident.training_resume import (
    ACCOUNT_RELEASE_CAPTURE_SCHEMA,
    HOST_ENOSPC_FAILURE_CLASS,
    HOST_ENOSPC_RECOVERY_SCHEMA,
    ResumeRefused,
    build_ledger,
    completion_packet,
    compute_retained_evidence_inventory,
    verify_host_enospc_recovery,
    verify_mechanical_stop,
)
from research.sandbox_a_resident.training_task import seed_allocations
from research.sandbox_a_resident.training_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_SCHEMA_V2,
    NAMESPACE,
    allocation,
    register_compact_allowlist_entry,
    unit_id,
    validate_compact_source_packet,
)
from research.sandbox_a_resident.units import publish
from tools.sandbox import source_delivery
from tools.sandbox.registry import TRANSPORT_TREE, digest, safe_path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PROXY_UPDATES = (64, 128, 192, 256, 320, 384)
PROXY_CHECKPOINTS = (DIAGNOSTIC_PROXY_UPDATE, *CANDIDATE_PROXY_UPDATES)
WORKLOAD_MANIFEST_KEYS = (
    "job_id",
    "commit",
    "script",
    "env",
    "source_paths",
    "timeout_minutes",
    "requirements",
    "inputs",
    "resume",
)
SLUG_RE = re.compile(r"^train01-(?:11|15|16|17)-c[12]-[a-z0-9][a-z0-9_-]*$")
SOURCE_PACKET_BASE_RE = re.compile(
    r"^packet-(train01-(?:11|15|16|17))(?:-c[12](?:-[a-z0-9][a-z0-9_-]*)?)?$"
)
# Prospectively frozen whole-input staging bounds reconciled with measured
# A100 upload-measurement.json anchors (train01-attempt-15 18-file compact
# staging: 31,438,403 B in 57.91 s actual, 8,804,704 B .pt.xz in 6.87 s;
# train01-attempt-13 212 chunks / 444.6 MB at 1,624,000 B/s sustained 2-worker
# rate; train01-attempt-{10,11} conservative 810,734 B/s + 3.670 s/file bound).
# Real occupied-simulator-ring periodic snapshots (e.g. iteration-0048.pt
# 1.343 GB -> 38.33 MB .pt.xz, 66.60 MB across 15 private inputs) project
# ~85.5 s nominal / ~154.7 s conservative; 100 MB private inputs project
# ~105.9 s nominal / ~195.9 s conservative across 18 files, staying strictly
# within the <= 120 s nominal and <= 210 s (< 300 s hard cap) bounds while
# rejecting uncompressed ~1.34 GB snapshots (> 860 s nominal).
MAX_CONTINUATION_PRIVATE_INPUT_BYTES = 100_000_000
STAGING_HARD_CAP_SECONDS = 300.0
STAGING_DESIGN_NOMINAL_SECONDS = 120.0
STAGING_DESIGN_WORST_SECONDS = 210.0
DEFAULT_STORAGE_SAFETY_MARGIN_BYTES = 4 * 1024**3
COMPACT_20_SNAPSHOT_NOMINAL_CEILING_BYTES = 1_000_000_000
TELEMETRY_AND_LOGS_BUDGET_BYTES = 100 * 1024 * 1024
TRANSFER_TEMP_FLOOR_BYTES = 256 * 1024 * 1024
SINGLE_THREAD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "SNAKE_DQN_DEVICE": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


class ContinuationRefused(ValueError):
    """A continuation precheck, staging bound, or dry-run validation failed."""


class _LocalContentsCLI:
    """Offline Contents/RPC stand-in measuring foreground background-sparse launch."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.session = "continuation-launch-check"
        self.deadline = None
        self.priced_deadline = None
        self._control_context = SimpleNamespace(deadline=float("inf"))
        self.calls: list[str] = []
        self.rpc_elapsed: list[float] = []

    def operation(self, _name: str):
        class _Op:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *_args):
                return False

        return _Op()

    def recoverable_rpc(self, code: str, timeout: float = 300.0):
        del timeout
        self.calls.append(code)
        namespace: dict[str, Any] = {}
        started = time.monotonic()
        exec(compile(code, "<continuation-local-launch>", "exec"), namespace)
        self.rpc_elapsed.append(time.monotonic() - started)
        return namespace["result"]

    def call(
        self,
        operation: str,
        _session_flag: str,
        _session: str,
        remote: str,
        local: str,
        **_kwargs: Any,
    ) -> str:
        if operation != "download":
            raise ContinuationRefused(f"unexpected local CLI operation: {operation}")
        path = Path(remote)
        if not path.is_file():
            raise FileNotFoundError(remote)
        shutil.copyfile(path, local)
        return ""


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ContinuationRefused(f"git {' '.join(args)} failed in {repo}: {detail}")
    return completed.stdout.strip()


def _ensure_git_identity(repo: Path, source_repo: Path) -> None:
    for key, fallback in (("user.name", "Argon"), ("user.email", "argon@local")):
        current = subprocess.run(
            ["git", "-C", str(repo), "config", "--get", key],
            text=True,
            capture_output=True,
            check=False,
        )
        if current.returncode == 0 and current.stdout.strip():
            continue
        inherited = subprocess.run(
            ["git", "-C", str(source_repo), "config", "--get", key],
            text=True,
            capture_output=True,
            check=False,
        )
        value = inherited.stdout.strip() if inherited.returncode == 0 else ""
        _git(repo, "config", key, value or fallback)


def _require_clean_qualified_checkout(repo: Path) -> str:
    tree = _git(repo, "rev-parse", "--verify", "HEAD:research/colab_cli")
    if tree != TRANSPORT_TREE:
        raise ContinuationRefused(
            f"checkout {repo} transport tree {tree} differs from {TRANSPORT_TREE}"
        )
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ContinuationRefused(f"checkout {repo} is not clean:\n{status}")
    return _git(repo, "rev-parse", "--verify", "HEAD")


def _complete_sidecars(path: Path) -> tuple[int, str]:
    sidecar = Path(f"{path}.sha256")
    complete = Path(f"{path}.complete")
    if not (path.is_file() and sidecar.is_file() and complete.is_file()):
        raise ContinuationRefused(f"incomplete artifact or sidecars: {path.name}")
    if any(item.is_symlink() for item in (path, sidecar, complete)):
        raise ContinuationRefused(f"symlinked artifact or sidecar: {path.name}")
    expected = sidecar.read_text().strip().split()[0]
    if digest(path) != expected:
        raise ContinuationRefused(f"corrupt artifact SHA-256: {path.name}")
    return path.stat().st_size, expected


def _resolve_attempt_output(source_attempt: Path) -> Path:
    safe_path(source_attempt)
    resolved = source_attempt.resolve()
    safe_path(resolved)
    if (resolved / "output" / "receipt.json").is_file() or (resolved / "output").is_dir():
        return resolved / "output"
    return resolved


def _resolve_source_packet(
    executed_manifest: dict[str, Any],
    *,
    repo_root: Path,
    source_packet_override: Path | None,
) -> Path:
    if source_packet_override is not None:
        safe_path(source_packet_override)
        resolved = source_packet_override.resolve()
        safe_path(resolved)
        if not (resolved / "dispatch-manifest.json").is_file():
            raise ContinuationRefused(f"source packet missing dispatch-manifest.json: {resolved}")
        return resolved
    admission = executed_manifest.get("sandbox_admission", {})
    manifest_path = admission.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ContinuationRefused("executed manifest lacks sandbox_admission.manifest_path")
    candidate = Path(manifest_path).parent
    safe_path(candidate)
    if (candidate / "dispatch-manifest.json").is_file():
        return candidate.resolve()
    fallback = (repo_root / "research/sandbox_a_resident" / candidate.name).resolve()
    safe_path(fallback)
    if (fallback / "dispatch-manifest.json").is_file():
        return fallback
    raise ContinuationRefused(f"cannot locate frozen source packet directory for {manifest_path}")


def encode_or_verify_compact_snapshot(
    original_pt: Path,
    *,
    source_packet_id: str,
    iteration: int,
    source_commit: str,
) -> dict[str, Any]:
    """Create-only encode or re-verify a Mac-encoded or compact-at-source snapshot."""
    torch.set_num_threads(1)
    derivative = Path(str(original_pt) + COMPACT_DERIVATIVE_SUFFIX)
    manifest_path = Path(str(original_pt) + COMPACT_MANIFEST_SUFFIX)
    if original_pt.is_symlink():
        raise CompactResumeRefused(f"symlinked full-state snapshot: {original_pt}")
    if not original_pt.exists():
        for suffix in (".sha256", ".complete"):
            orphan = Path(f"{original_pt}{suffix}")
            if orphan.exists() or orphan.is_symlink():
                raise CompactResumeRefused(
                    f"orphan raw full-state sidecar on compact-at-source export: {orphan.name}"
                )
        verified = verify_compact_snapshot_export(
            original_pt.parent,
            iteration,
            source_packet_id=source_packet_id,
            source_commit=source_commit,
            require_actor=True,
        )
        return dict(verified["allowlist_entry"])
    if not derivative.exists() and not manifest_path.exists():
        encoded = encode_snapshot(
            original_pt,
            source_packet_id=source_packet_id,
            iteration=iteration,
            source_commit=source_commit,
        )
        return dict(encoded["allowlist_entry"])
    orig_size, orig_sha = _complete_sidecars(original_pt)
    deriv_size, deriv_sha = _complete_sidecars(derivative)
    man_size, man_sha = _complete_sidecars(manifest_path)
    runtime_stem = f"resume/iteration-{iteration:04d}.pt"
    binding = {
        "schema": COMPACT_RESUME_SCHEMA_V2,
        "source_packet_id": source_packet_id,
        "iteration": iteration,
        "unit": unit_id(iteration),
        "original": {"path": runtime_stem, "size": orig_size, "sha256": orig_sha},
        "derivative": {
            "path": runtime_stem + COMPACT_DERIVATIVE_SUFFIX,
            "size": deriv_size,
            "sha256": deriv_sha,
        },
        "manifest": {
            "path": runtime_stem + COMPACT_MANIFEST_SUFFIX,
            "size": man_size,
            "sha256": man_sha,
        },
    }
    loaded_manifest = load_manifest(manifest_path, binding)
    decode_stream(
        derivative, lambda _block: None, expected_size=orig_size, expected_sha256=orig_sha
    )
    state = torch.load(original_pt, map_location="cpu", weights_only=False)
    verify_decoded_state(state, loaded_manifest)
    return {
        "source_packet_id": source_packet_id,
        "iteration": iteration,
        "unit": binding["unit"],
        "original": binding["original"],
        "derivative": binding["derivative"],
        "manifest": binding["manifest"],
        "source_commit": source_commit,
    }


def compute_storage_projection(
    output_dir: Path,
    *,
    remaining_snapshot_count: int,
    reference_original_pt_bytes: int,
    reference_derivative_bytes: int,
    reference_manifest_bytes: int,
    reference_actor_bytes: int,
    reference_json_bytes: int,
    snapshot_format: str = "compact",
    safety_margin_bytes: int = DEFAULT_STORAGE_SAFETY_MARGIN_BYTES,
) -> dict[str, Any]:
    """Fail-closed Mac destination disk-capacity admission check with stated margin."""
    if snapshot_format not in {"compact", "raw"}:
        raise ContinuationRefused(f"unsupported snapshot_format: {snapshot_format!r}")
    if type(remaining_snapshot_count) is not int or remaining_snapshot_count < 1:
        raise ContinuationRefused("remaining_snapshot_count must be a positive integer")
    for label, val in (
        ("reference_original_pt_bytes", reference_original_pt_bytes),
        ("reference_derivative_bytes", reference_derivative_bytes),
        ("reference_manifest_bytes", reference_manifest_bytes),
        ("reference_actor_bytes", reference_actor_bytes),
        ("reference_json_bytes", reference_json_bytes),
        ("safety_margin_bytes", safety_margin_bytes),
    ):
        if type(val) is not int or val < 0:
            raise ContinuationRefused(f"{label} must be a nonnegative integer")

    probe = output_dir
    while not probe.exists():
        if probe.parent == probe:
            raise ContinuationRefused(
                f"cannot locate existing filesystem ancestor for {output_dir}"
            )
        probe = probe.parent
    if probe.is_symlink():
        raise ContinuationRefused(f"symlinked destination ancestor: {probe}")
    resolved_ancestor = probe.resolve()
    usage = shutil.disk_usage(resolved_ancestor)
    st_dev = os.stat(resolved_ancestor).st_dev

    raw_sidecar_overhead = 3 * 2 * 4096
    raw_per_snapshot_nominal = (
        reference_original_pt_bytes
        + reference_actor_bytes
        + reference_json_bytes
        + raw_sidecar_overhead
    )
    raw_counterfactual_exports_bytes = remaining_snapshot_count * raw_per_snapshot_nominal

    if snapshot_format == "compact":
        sidecar_overhead = 4 * 2 * 4096
        per_snapshot_nominal = (
            reference_derivative_bytes
            + reference_manifest_bytes
            + reference_actor_bytes
            + reference_json_bytes
            + sidecar_overhead
        )
        per_snapshot_conservative = (
            int(math.ceil(1.25 * reference_derivative_bytes))
            + int(math.ceil(1.50 * reference_manifest_bytes))
            + int(math.ceil(1.05 * reference_actor_bytes))
            + int(math.ceil(1.50 * reference_json_bytes))
            + sidecar_overhead
        )
        transfer_temp_bytes = max(
            TRANSFER_TEMP_FLOOR_BYTES,
            int(math.ceil(1.25 * max(reference_derivative_bytes, reference_actor_bytes))),
        )
        mac_verify_decode_temp_bytes = int(math.ceil(1.05 * reference_original_pt_bytes))
    else:
        per_snapshot_nominal = raw_per_snapshot_nominal
        per_snapshot_conservative = (
            int(math.ceil(1.05 * reference_original_pt_bytes))
            + int(math.ceil(1.05 * reference_actor_bytes))
            + int(math.ceil(1.50 * reference_json_bytes))
            + raw_sidecar_overhead
        )
        transfer_temp_bytes = max(
            TRANSFER_TEMP_FLOOR_BYTES,
            int(math.ceil(1.10 * reference_original_pt_bytes)),
        )
        mac_verify_decode_temp_bytes = 0

    remaining_exports_nominal = remaining_snapshot_count * per_snapshot_nominal
    remaining_exports_conservative = remaining_snapshot_count * per_snapshot_conservative
    if (
        snapshot_format == "compact"
        and remaining_snapshot_count <= 20
        and remaining_exports_nominal >= COMPACT_20_SNAPSHOT_NOMINAL_CEILING_BYTES
    ):
        raise ContinuationRefused(
            f"compact periodic exports ({remaining_exports_nominal} B across "
            f"{remaining_snapshot_count} snapshots) exceed "
            f"{COMPACT_20_SNAPSHOT_NOMINAL_CEILING_BYTES} B (<1 GB) design ceiling"
        )

    temp_headroom_bytes = transfer_temp_bytes + mac_verify_decode_temp_bytes
    required_nominal_bytes = (
        remaining_exports_nominal + TELEMETRY_AND_LOGS_BUDGET_BYTES + temp_headroom_bytes
    )
    required_conservative_bytes = (
        remaining_exports_conservative + TELEMETRY_AND_LOGS_BUDGET_BYTES + temp_headroom_bytes
    )
    required_conservative_with_margin_bytes = required_conservative_bytes + safety_margin_bytes
    free_bytes = int(usage.free)
    if free_bytes < required_conservative_with_margin_bytes:
        raise ContinuationRefused(
            f"insufficient Mac storage on {resolved_ancestor}: free={free_bytes} B "
            f"({free_bytes / (1024**3):.2f} GiB) < required_with_margin="
            f"{required_conservative_with_margin_bytes} B "
            f"({required_conservative_with_margin_bytes / (1024**3):.2f} GiB; "
            f"exports={remaining_exports_conservative} B, temp={temp_headroom_bytes} B, "
            f"logs={TELEMETRY_AND_LOGS_BUDGET_BYTES} B, margin={safety_margin_bytes} B)"
        )
    persistent_after_conservative_bytes = free_bytes - (
        remaining_exports_conservative + TELEMETRY_AND_LOGS_BUDGET_BYTES
    )
    free_above_margin_at_temp_peak_bytes = free_bytes - required_conservative_with_margin_bytes
    reduction_factor = (
        round(raw_counterfactual_exports_bytes / remaining_exports_nominal, 2)
        if remaining_exports_nominal > 0
        else None
    )
    return {
        "schema": "sandbox-a-train01-continuation-mac-storage-projection/v1",
        "status": "PASS",
        "snapshot_format": snapshot_format,
        "destination_output_dir": str(output_dir),
        "checked_filesystem_ancestor": str(resolved_ancestor),
        "filesystem_st_dev": int(st_dev),
        "filesystem_total_bytes": int(usage.total),
        "filesystem_free_bytes": free_bytes,
        "filesystem_free_gib": round(free_bytes / (1024**3), 3),
        "remaining_snapshot_count": remaining_snapshot_count,
        "reference_snapshot_bytes": {
            "raw_original_pt": reference_original_pt_bytes,
            "compact_derivative_pt_xz": reference_derivative_bytes,
            "compact_manifest": reference_manifest_bytes,
            "actor_pth": reference_actor_bytes,
            "envelope_json": reference_json_bytes,
        },
        "per_snapshot_nominal_bytes": per_snapshot_nominal,
        "per_snapshot_conservative_bytes": per_snapshot_conservative,
        "remaining_exports_nominal_bytes": remaining_exports_nominal,
        "remaining_exports_nominal_mb": round(remaining_exports_nominal / 1_000_000.0, 2),
        "remaining_exports_conservative_bytes": remaining_exports_conservative,
        "remaining_exports_conservative_gib": round(remaining_exports_conservative / (1024**3), 3),
        "under_1gb_nominal_for_remaining_exports": (
            remaining_exports_nominal < COMPACT_20_SNAPSHOT_NOMINAL_CEILING_BYTES
        ),
        "telemetry_and_logs_bytes": TELEMETRY_AND_LOGS_BUDGET_BYTES,
        "transfer_temp_headroom_bytes": transfer_temp_bytes,
        "mac_verify_decode_temp_bytes": mac_verify_decode_temp_bytes,
        "temp_headroom_bytes": temp_headroom_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "safety_margin_gib": round(safety_margin_bytes / (1024**3), 3),
        "required_nominal_bytes": required_nominal_bytes,
        "required_conservative_bytes": required_conservative_bytes,
        "required_conservative_with_margin_bytes": required_conservative_with_margin_bytes,
        "required_conservative_with_margin_gib": round(
            required_conservative_with_margin_bytes / (1024**3), 3
        ),
        "persistent_free_after_conservative_run_bytes": persistent_after_conservative_bytes,
        "persistent_free_after_conservative_run_gib": round(
            persistent_after_conservative_bytes / (1024**3), 3
        ),
        "free_above_safety_margin_at_temp_peak_bytes": free_above_margin_at_temp_peak_bytes,
        "free_above_safety_margin_at_temp_peak_gib": round(
            free_above_margin_at_temp_peak_bytes / (1024**3), 3
        ),
        "raw_counterfactual_exports_bytes": raw_counterfactual_exports_bytes,
        "raw_counterfactual_exports_gib": round(raw_counterfactual_exports_bytes / (1024**3), 3),
        "export_byte_reduction_factor": reduction_factor,
    }


def _discover_default_supervisor_log(
    job_id: Any, *, output_root: Path, repo_root: Path
) -> Path | None:
    if not isinstance(job_id, str) or not job_id:
        return None
    candidates = [repo_root.parent / "snake-dqn-artifacts/ops/logs" / f"{job_id}.log"]
    if len(output_root.parents) >= 3:
        candidates.insert(0, output_root.parents[2] / "ops/logs" / f"{job_id}.log")
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    return None


def build_host_enospc_recovery_certificate(
    source_attempt: Path,
    *,
    account_status_capture_path: Path,
    certificate_output_path: Path,
    recover_receipt_path: Path,
    supervisor_log_path: Path | None = None,
    state_dir: Path | None = None,
    account_root: Path | None = None,
    source_packet: Path | None = None,
    repo_root: Path = ROOT,
    now_unix_seconds: float | None = None,
    max_capture_age_seconds: float | None = 86400.0,
) -> dict[str, Any]:
    """Create-only publish and immediately validate a host ENOSPC recovery certificate."""
    repo_root = safe_path(repo_root).resolve()
    output_root = _resolve_attempt_output(source_attempt)
    attempt_root = output_root.parent if output_root.name == "output" else output_root
    if (output_root / "receipt.json").exists() or (output_root / "receipt.json").is_symlink():
        raise ContinuationRefused("receipt.json already exists; refusing ENOSPC recovery builder")
    if (output_root / "terminal.json").exists() or (output_root / "terminal.json").is_symlink():
        raise ContinuationRefused("terminal.json already exists; refusing ENOSPC recovery builder")
    executed_manifest_path = output_root / "manifest.json"
    if executed_manifest_path.is_symlink() or not executed_manifest_path.is_file():
        raise ContinuationRefused(f"missing executed manifest: {executed_manifest_path}")
    executed_manifest = json.loads(executed_manifest_path.read_text())
    source_packet_dir = _resolve_source_packet(
        executed_manifest, repo_root=repo_root, source_packet_override=source_packet
    )
    source_packet_id = source_packet_dir.name
    source_packet_manifest_path = source_packet_dir / "dispatch-manifest.json"
    source_control = json.loads((source_packet_dir / "control.json").read_text())

    resolved_state = (attempt_root / "state") if state_dir is None else state_dir.resolve()
    assignment_path = resolved_state / "assignment.json"
    if (
        state_dir is None
        and not assignment_path.exists()
        and isinstance(executed_manifest.get("job_id"), str)
    ):
        lane_assignment = resolved_state / executed_manifest["job_id"] / "assignment.json"
        if lane_assignment.is_file() and not lane_assignment.is_symlink():
            assignment_path = lane_assignment
    if assignment_path.is_symlink() or not assignment_path.is_file():
        raise ContinuationRefused(f"missing assignment.json: {assignment_path}")
    assignment = json.loads(assignment_path.read_text())

    resolved_sup_log = (
        _discover_default_supervisor_log(
            executed_manifest.get("job_id"), output_root=output_root, repo_root=repo_root
        )
        if supervisor_log_path is None
        else supervisor_log_path.resolve()
    )
    if resolved_sup_log is None or not resolved_sup_log.is_file() or resolved_sup_log.is_symlink():
        raise ContinuationRefused("host ENOSPC recovery certificate requires supervisor_log_path")
    raw_lines = resolved_sup_log.read_text().splitlines()
    nonempty = [line.strip() for line in raw_lines if line.strip()]
    if len(nonempty) < 3:
        raise ContinuationRefused("supervisor_log lacks poll history and terminal ENOSPC lines")

    safe_path(account_status_capture_path)
    resolved_capture = account_status_capture_path.resolve()
    cap_size, cap_sha = _complete_sidecars(resolved_capture)
    capture_payload = json.loads(resolved_capture.read_text())
    if capture_payload.get("schema") != ACCOUNT_RELEASE_CAPTURE_SCHEMA:
        raise ContinuationRefused("account_status_capture_path has wrong schema")

    resolved_account_root = (
        (Path.home() / ".config/colab-cli/snake").resolve()
        if account_root is None
        else account_root.resolve()
    )
    session = str(assignment["session"])
    endpoint = str(assignment["endpoint"])
    active_file = resolved_account_root / f"active-{session}.json"
    if active_file.exists() or active_file.is_symlink():
        raise ContinuationRefused("qualified reconciliation required; active record still exists")
    safe_path(recover_receipt_path)
    if recover_receipt_path.is_symlink() or not recover_receipt_path.is_file():
        raise ContinuationRefused("missing or symlinked qualified recovery receipt")
    matched_rec = recover_receipt_path.resolve()
    if matched_rec.parent != output_root.resolve() or not (
        matched_rec.name.startswith("recovery-") and matched_rec.suffix == ".json"
    ):
        raise ContinuationRefused(
            "explicit recovery receipt must be recovery-*.json in output_root"
        )
    broker_reconciliation = {
        "mode": "qualified_recover_receipt",
        "account_root": str(resolved_account_root),
        "recover_receipt": {
            "path": str(matched_rec),
            "size": matched_rec.stat().st_size,
            "sha256": digest(matched_rec),
        },
    }

    retained = compute_retained_evidence_inventory(output_root)
    verified_ledger, _uploads, latest = build_ledger(
        output_root / "exports/training", source_control
    )
    latest_export = verified_ledger["units"][-1]["exports"][0]

    certificate = {
        "schema": HOST_ENOSPC_RECOVERY_SCHEMA,
        "original_attempt_state": {
            "prior_attempt_status": "INCOMPLETE",
            "failure_class": HOST_ENOSPC_FAILURE_CLASS,
            "original_receipt_present": False,
            "original_terminal_status": "UNKNOWN_UNAVAILABLE",
            "original_supervisor_error": "UNKNOWN_UNAVAILABLE",
            "original_cleanup_error": "UNKNOWN_UNAVAILABLE",
            "post_hoc_release_verified": True,
        },
        "identities": {
            "study": source_control["study"],
            "namespace": NAMESPACE,
            "job_id": executed_manifest["job_id"],
            "source_packet_id": source_packet_id,
            "source_packet_manifest_sha256": digest(source_packet_manifest_path),
            "executed_manifest_sha256": digest(executed_manifest_path),
            "source_commit": executed_manifest["commit"],
            "control_allocation": allocation(source_control),
            "output_dir": str(output_root.resolve()),
            "generation": assignment["generation"],
            "endpoint": endpoint,
            "session": session,
            "assignment_record": {
                "path": str(assignment_path.resolve()),
                "size": assignment_path.stat().st_size,
                "sha256": digest(assignment_path),
            },
        },
        "supervisor_log": {
            "path": str(resolved_sup_log),
            "size": resolved_sup_log.stat().st_size,
            "sha256": digest(resolved_sup_log),
            "line_count": len(raw_lines),
            "terminal_enospc_line": nonempty[-2],
            "terminal_exit_line": nonempty[-1],
        },
        "retained_evidence": retained,
        "release_evidence": {
            "account_status_capture": {
                "path": str(resolved_capture),
                "size": cap_size,
                "sha256": cap_sha,
                "captured_at_utc": capture_payload.get("captured_at_utc"),
                "captured_unix_seconds": capture_payload.get("captured_unix_seconds"),
            },
            "broker_reconciliation": broker_reconciliation,
        },
        "resume_binding": {
            "verified_snapshot_iterations": [
                int(row["iteration"]) for row in verified_ledger["units"]
            ],
            "latest_eligible_iteration": latest,
            "latest_eligible_unit": unit_id(latest),
            "latest_full_state_size": latest_export["size"],
            "latest_full_state_sha256": latest_export["sha256"],
        },
    }
    safe_path(certificate_output_path)
    certificate_output_path.parent.mkdir(parents=True, exist_ok=True)
    publish(
        certificate_output_path,
        lambda dst: dst.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n"),
    )
    try:
        return verify_host_enospc_recovery(
            certificate_output_path,
            executed_manifest,
            digest(executed_manifest_path),
            output_root=output_root,
            source_control=source_control,
            source_packet_id=source_packet_id,
            source_packet_manifest_sha256=digest(source_packet_manifest_path),
            now_unix_seconds=now_unix_seconds,
            max_capture_age_seconds=max_capture_age_seconds,
        )
    except ResumeRefused as error:
        raise ContinuationRefused(str(error)) from error


def compute_remaining_envelope(
    iteration: int,
    *,
    outer_iterations: int = 384,
    snapshot_interval: int = 16,
    deferred_proxy: bool = False,
    needs_baseline_proxy: bool = False,
) -> dict[str, Any]:
    """Size remaining A100 base and 2x effective envelopes per colab-limits §4.3-4.4."""
    if type(iteration) is not int or not 0 <= iteration < outer_iterations:
        raise ContinuationRefused(
            f"continuation iteration must lie in 0..{outer_iterations - 1}, got {iteration!r}"
        )
    remaining_updates = outer_iterations - iteration
    scheduled_proxies = [step for step in PROXY_CHECKPOINTS if iteration < step <= outer_iterations]
    proxy_count = (
        len(scheduled_proxies) + (1 if deferred_proxy else 0) + (1 if needs_baseline_proxy else 0)
    )
    next_snapshot = ((iteration // snapshot_interval) + 1) * snapshot_interval
    remaining_snapshots = list(range(next_snapshot, outer_iterations + 1, snapshot_interval))
    if outer_iterations not in remaining_snapshots:
        remaining_snapshots.append(outer_iterations)

    # Calibrated from train01-attempt-15's measured steady-state A100 periodic
    # exports (iteration-0016: 204.78 s collect / 7.39 s learner; iteration-0032:
    # 207.20 s collect / 9.11 s learner; iteration-0048: 204.30 s collect /
    # 8.59 s learner) plus ~38.3 MB periodic .pt.xz stage + decode startup.
    phases_nominal = {
        "alloc_warmup_launch_ratecal_compact_stage_decode": 240.0,
        "compile_self_check_resumed": 90.0,
        "ppo_rollout_collection": round(remaining_updates * 205.5, 2),
        "ppo_learner_float64_kl": round(remaining_updates * 8.5, 2),
        "detached_h5000_e96_proxies": round(proxy_count * 2049.4, 2),
        "snapshot_local_writes": round(len(remaining_snapshots) * 5.0, 2),
        "post_terminal_final_export_download": 371.7,
        "cleanup_and_stop_verification": 60.0,
    }
    phases_conservative = {
        "alloc_warmup_launch_ratecal_compact_stage_decode": 480.0,
        "compile_self_check_resumed": 150.0,
        "ppo_rollout_collection": round(remaining_updates * 210.0, 2),
        "ppo_learner_float64_kl": round(remaining_updates * 9.5, 2),
        "detached_h5000_e96_proxies": round(proxy_count * 2100.0, 2),
        "snapshot_local_writes": round(len(remaining_snapshots) * 7.0, 2),
        "post_terminal_final_export_download_worst_1p66mbs": 879.2,
        "cleanup_and_stop_verification": 60.0,
    }
    nominal_seconds = round(sum(phases_nominal.values()), 2)
    conservative_seconds = round(sum(phases_conservative.values()), 2)
    est_minutes = max(60, int(math.ceil((conservative_seconds / 60.0) / 10.0) * 10))
    timeout_minutes = 2 * est_minutes
    base_seconds = est_minutes * 60
    effective_seconds = timeout_minutes * 60
    unit_cap_base = math.ceil(((est_minutes / 60.0) * 8.0 + 0.5) * 2.0) / 2.0
    unit_cap_effective = round(2.0 * unit_cap_base, 1)
    if unit_cap_effective < (timeout_minutes / 60.0) * 8.0:
        raise ContinuationRefused("effective unit_cap violates 8.0 u/h priced timeout admission")
    return {
        "schema": "sandbox-a-train01-continuation-wall-time-projection/v1",
        "resume_from_iteration": iteration,
        "outer_iterations": outer_iterations,
        "remaining_updates": remaining_updates,
        "remaining_actor_rows": remaining_updates * 1024 * 256,
        "scheduled_proxy_updates": scheduled_proxies,
        "deferred_proxy": deferred_proxy,
        "needs_baseline_proxy": needs_baseline_proxy,
        "proxy_count": proxy_count,
        "remaining_snapshot_count": len(remaining_snapshots),
        "phases_nominal_seconds": phases_nominal,
        "phases_conservative_seconds": phases_conservative,
        "nominal_seconds": nominal_seconds,
        "nominal_hours": round(nominal_seconds / 3600.0, 2),
        "conservative_base_seconds": conservative_seconds,
        "conservative_base_hours": round(conservative_seconds / 3600.0, 2),
        "est_minutes": est_minutes,
        "timeout_minutes": timeout_minutes,
        "base_seconds": base_seconds,
        "effective_seconds": effective_seconds,
        "unit_cap_base_at_8p0": unit_cap_base,
        "unit_cap": unit_cap_effective,
        "expected_nominal_units_at_5p3": round((nominal_seconds / 3600.0) * 5.3, 2),
    }


def compute_staging_projection(
    broker_manifest: dict[str, Any],
    runtime_checkout: Path,
    *,
    source_tar_gz_bytes: int,
) -> dict[str, Any]:
    """Compute exact whole-staging time per colab-limits §3.1 and §4.1."""
    private_inputs: dict[str, int] = {}
    for row in broker_manifest["inputs"]:
        private_inputs[row["dest"]] = Path(row["path"]).stat().st_size
    private_bytes = sum(private_inputs.values())
    private_count = len(private_inputs)
    bootstrap_bytes = (runtime_checkout / "research/colab_cli/bootstrap.sh").stat().st_size
    source_py_bytes = (runtime_checkout / "research/colab_cli/source.py").stat().st_size
    file_count = private_count + 3
    total_bytes = private_bytes + bootstrap_bytes + source_py_bytes + source_tar_gz_bytes
    nominal_seconds = round(1.30 + (total_bytes / 1_635_817.0) + 2.193 * file_count + 3.45, 2)
    worst_seconds = round(1.82 + (total_bytes / 810_734.0) + 3.670 * file_count + 3.47, 2)
    if private_bytes > MAX_CONTINUATION_PRIVATE_INPUT_BYTES:
        raise ContinuationRefused(
            f"private inputs ({private_bytes} B) exceed "
            f"{MAX_CONTINUATION_PRIVATE_INPUT_BYTES} B whole-staging ceiling"
        )
    if (
        nominal_seconds > STAGING_DESIGN_NOMINAL_SECONDS
        or worst_seconds > STAGING_DESIGN_WORST_SECONDS
        or worst_seconds >= STAGING_HARD_CAP_SECONDS
    ):
        raise ContinuationRefused(
            f"whole-staging projection exceeds target bounds: "
            f"nominal={nominal_seconds}s worst={worst_seconds}s"
        )
    return {
        "schema": "sandbox-a-train01-continuation-upload-projection/v1",
        "skill_reference": "colab-limits §3.1, §4.1",
        "status": "PASS",
        "hard_cap_seconds": STAGING_HARD_CAP_SECONDS,
        "canary_bar_seconds": 150.0,
        "design_target_nominal_seconds": STAGING_DESIGN_NOMINAL_SECONDS,
        "design_target_worst_seconds": STAGING_DESIGN_WORST_SECONDS,
        "max_private_input_bytes": MAX_CONTINUATION_PRIVATE_INPUT_BYTES,
        "measured_calibration_anchors": {
            "attempt_15_compact_18files_bytes": 31_438_403,
            "attempt_15_compact_18files_elapsed_seconds": 57.91,
            "attempt_15_pt_xz_8804704b_transfer_seconds": 6.87,
            "attempt_13_sustained_2worker_chunk_bytes_per_second": 1_624_000,
            "conservative_bytes_per_second": 810_734,
            "conservative_per_file_overhead_seconds": 3.67,
        },
        "private_input_count": private_count,
        "private_input_bytes": private_bytes,
        "private_inputs": dict(sorted(private_inputs.items())),
        "supervisor_pinned_files": {
            "bootstrap.sh": bootstrap_bytes,
            "source.py": source_py_bytes,
            "source.tar.gz": source_tar_gz_bytes,
        },
        "total_upload_file_count": file_count,
        "total_upload_bytes": total_bytes,
        "nominal_seconds": nominal_seconds,
        "worst_case_seconds": worst_seconds,
        "fraction_of_300s_cap_nominal": round(nominal_seconds / STAGING_HARD_CAP_SECONDS, 4),
        "fraction_of_300s_cap_worst": round(worst_seconds / STAGING_HARD_CAP_SECONDS, 4),
    }


def measure_launch_exec_bound(
    runtime_checkout: Path,
    broker_manifest: dict[str, Any],
    *,
    runs: int = 3,
    bound_seconds: float = 2.0,
) -> dict[str, Any]:
    """Build sparse archive and prove 3x local background-sparse launch < 2.0 s."""
    with tempfile.TemporaryDirectory(prefix="train01-cont-launch-") as temp_dir:
        temp_root = Path(temp_dir).resolve()
        prep_dir = temp_root / "prepared"
        plan = source.prepare(runtime_checkout, broker_manifest, prep_dir)
        archive_path = prep_dir / str(plan["fallback_archive"])
        archive_bytes = archive_path.stat().st_size
        source_py = runtime_checkout / "research/colab_cli/source.py"
        records: list[dict[str, Any]] = []
        for index in range(runs):
            run_root = temp_root / f"content-{index}"
            run_root.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(source_py, run_root / "source.py")
            shutil.copyfile(archive_path, run_root / archive_path.name)
            cli = _LocalContentsCLI(run_root)
            started = time.monotonic()
            delivery = source_delivery.install_background(
                cli, str(run_root), plan, cli.recoverable_rpc
            )
            completion_seconds = time.monotonic() - started
            if (
                delivery.get("transport") != "fallback"
                or delivery.get("inventory_verified") is not True
                or len(cli.rpc_elapsed) != 1
            ):
                raise ContinuationRefused("background-sparse launch probe failed verification")
            launch_seconds = cli.rpc_elapsed[0]
            if launch_seconds >= bound_seconds:
                raise ContinuationRefused(
                    f"background-sparse launch {launch_seconds:.4f}s >= {bound_seconds:.1f}s bound"
                )
            records.append(
                {
                    "index": index,
                    "launch_seconds": launch_seconds,
                    "completion_seconds": completion_seconds,
                    "status": "completed",
                }
            )
        for locked in prep_dir.iterdir():
            if locked.is_file():
                locked.chmod(0o644)
    return {
        "status": "PASS",
        "bound_seconds": bound_seconds,
        "files_verified": len(plan["files"]),
        "source_tar_gz_bytes": archive_bytes,
        "source_unpacked_bytes": int(plan["fallback_unpacked_size"]),
        "runs": records,
    }


def _single_thread_process_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(SINGLE_THREAD_ENV)
    python_exe = Path(os.path.abspath(sys.executable))
    env["PATH"] = f"{python_exe.parent}:{env.get('PATH', '')}"
    env["SNAKE_DQN_PYTHON"] = str(python_exe)
    return env


def run_packet_validation(broker_checkout: Path, packet_dir: Path) -> str:
    """Validate committed packet bindings inside broker_checkout without pytest recursion."""
    code = (
        "import sys; from pathlib import Path; "
        "from research.sandbox_a_resident.training_preflight import validate_packet; "
        "check = validate_packet(Path(sys.argv[1])); "
        "print(f'PASS_VALIDATE_PACKET seeds={check[\"count\"]} allocations=0')"
    )
    python_exe = os.path.abspath(sys.executable)
    completed = subprocess.run(
        ["nice", "-n", "15", python_exe, "-c", code, str(packet_dir)],
        cwd=broker_checkout,
        env=_single_thread_process_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0 or "PASS_VALIDATE_PACKET" not in completed.stdout:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ContinuationRefused(f"committed packet validation failed: {detail}")
    return completed.stdout.strip()


def run_dispatch_dry_run(
    broker_checkout: Path,
    runtime_checkout: Path,
    packet_dir: Path,
    state_dir: Path,
    job_id: str,
) -> str:
    """Run tools/ops/dispatch.sh --dry-run and require PASS_DRY_RUN with allocations=0."""
    dispatch_sh = broker_checkout / "tools/ops/dispatch.sh"
    completed = subprocess.run(
        [
            "nice",
            "-n",
            "15",
            str(dispatch_sh),
            job_id,
            str(packet_dir / "dispatch-manifest.json"),
            "--broker",
            str(broker_checkout),
            "--cwd",
            str(runtime_checkout),
            "--state-dir",
            str(state_dir),
            "--dispatch-txt",
            str(packet_dir / "DISPATCH.txt"),
            "--dry-run",
        ],
        cwd=broker_checkout,
        env=_single_thread_process_env(),
        text=True,
        capture_output=True,
        check=False,
        # Both offline admission passes fit inside the 540s outer launcher
        # window, with one minute for this dry-run process's own checks.
        timeout=600,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0 or "PASS_DRY_RUN" not in output or "allocations=0" not in output:
        detail = completed.stderr.strip() or output
        raise ContinuationRefused(f"dispatch.sh --dry-run failed: {detail}")
    return output


def _format_dispatch_command(
    broker_checkout: Path,
    runtime_checkout: Path,
    packet_dir: Path,
    state_dir: Path,
    job_id: str,
) -> str:
    return (
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 "
        f"TORCHINDUCTOR_COMPILE_THREADS=1 nice -n 15 {broker_checkout}/tools/ops/dispatch.sh "
        f"{job_id} {packet_dir}/dispatch-manifest.json "
        f"--broker {broker_checkout} "
        f"--cwd {runtime_checkout} "
        f"--state-dir {state_dir} "
        f"--dispatch-txt {packet_dir}/DISPATCH.txt"
    )


def _write_ten_line_sandbox(path: Path, sandbox: dict[str, Any]) -> None:
    if len(sandbox) != 10:
        raise ContinuationRefused("adopted sandbox.json requires exactly 10 top-level keys")
    lines = [
        ("{" if index == 0 else " ")
        + json.dumps(key)
        + ": "
        + json.dumps(value)
        + ("," if index < 9 else "}")
        for index, (key, value) in enumerate(sandbox.items())
    ]
    with path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def _derive_identities(
    source_packet_id: str,
    *,
    completion_index: int,
    iteration: int,
    slug_override: str | None,
    job_id_override: str | None,
    date_tag: str | None,
) -> tuple[str, str, str]:
    match = SOURCE_PACKET_BASE_RE.fullmatch(source_packet_id)
    if match is None:
        raise ContinuationRefused(f"unsupported source packet id: {source_packet_id!r}")
    base_run = match.group(1)
    slug = (
        f"{base_run}-c{completion_index}-i{iteration:04d}"
        if slug_override is None
        else slug_override
    )
    if not SLUG_RE.fullmatch(slug):
        raise ContinuationRefused(
            f"successor slug {slug!r} must match train01-(11|15|16|17)-c(1|2)-<suffix>"
        )
    short_suffix = slug.removeprefix("train01-")
    day = dt.date.today().strftime("%Y%m%d") if date_tag is None else date_tag
    job_id = (
        f"sandbox-a-train01-{day}-{short_suffix}" if job_id_override is None else job_id_override
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,54}", job_id):
        raise ContinuationRefused(f"invalid or overly long job_id: {job_id!r}")
    return slug, short_suffix, job_id


def build_continuation_packet(
    source_attempt: Path,
    iteration: int,
    *,
    source_packet: Path | None = None,
    slug: str | None = None,
    job_id: str | None = None,
    date_tag: str | None = None,
    repo_root: Path = ROOT,
    checkout_parent: Path | None = None,
    artifacts_root: Path | None = None,
    skip_dry_run: bool = False,
    launch_runs: int = 3,
    recovery_evidence: Path | None = None,
    supervisor_log: Path | None = None,
    safety_margin_bytes: int = DEFAULT_STORAGE_SAFETY_MARGIN_BYTES,
) -> dict[str, Any]:
    """Seal a fresh TRAIN-01 continuation packet in detached worktrees; never dispatch."""
    if type(launch_runs) is not int or launch_runs < 1:
        raise ContinuationRefused("launch_runs must be a positive integer")
    repo_root = safe_path(repo_root).resolve()
    safe_path(repo_root)
    _require_clean_qualified_checkout(repo_root)

    output_root = _resolve_attempt_output(source_attempt)
    executed_manifest_path = output_root / "manifest.json"
    receipt_path = output_root / "receipt.json"
    export_root = output_root / "exports/training"
    safe_path(executed_manifest_path)
    safe_path(receipt_path)
    if executed_manifest_path.is_symlink() or not executed_manifest_path.is_file():
        raise ContinuationRefused(
            f"missing or symlinked source attempt evidence: {executed_manifest_path}"
        )
    safe_path(export_root)
    if export_root.is_symlink() or not export_root.is_dir():
        raise ContinuationRefused(f"missing or symlinked training export directory: {export_root}")

    executed_manifest = json.loads(executed_manifest_path.read_text())
    source_packet_dir = _resolve_source_packet(
        executed_manifest, repo_root=repo_root, source_packet_override=source_packet
    )
    source_packet_id = source_packet_dir.name
    try:
        validate_compact_source_packet(source_packet_id, iteration)
    except ValueError as error:
        raise ContinuationRefused(str(error)) from error

    frozen_source_manifest_path = source_packet_dir / "dispatch-manifest.json"
    frozen_source_manifest = json.loads(frozen_source_manifest_path.read_text())
    for key in WORKLOAD_MANIFEST_KEYS:
        if executed_manifest.get(key) != frozen_source_manifest.get(key):
            raise ContinuationRefused(f"executed manifest differs from frozen source packet: {key}")

    source_control = json.loads((source_packet_dir / "control.json").read_text())
    resolved_sup_log: Path | None = None
    if supervisor_log is not None:
        safe_path(supervisor_log)
        resolved_sup_log = supervisor_log.resolve()
        safe_path(resolved_sup_log)
    else:
        resolved_sup_log = _discover_default_supervisor_log(
            executed_manifest.get("job_id"), output_root=output_root, repo_root=repo_root
        )

    receipt: dict[str, Any] | None = None
    resolved_recovery: Path | None = None
    recovery_summary: dict[str, Any] | None = None
    if recovery_evidence is None:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ContinuationRefused(
                f"missing or symlinked source attempt evidence: {receipt_path}"
            )
        receipt = json.loads(receipt_path.read_text())
        try:
            verify_mechanical_stop(
                receipt,
                executed_manifest,
                digest(executed_manifest_path),
                output_root=output_root,
                supervisor_log_path=resolved_sup_log,
            )
        except ResumeRefused as error:
            raise ContinuationRefused(str(error)) from error
    else:
        safe_path(recovery_evidence)
        resolved_recovery = recovery_evidence.resolve()
        safe_path(resolved_recovery)
        if receipt_path.exists() or receipt_path.is_symlink():
            raise ContinuationRefused(
                "receipt.json exists in source attempt; refusing narrow host ENOSPC recovery path"
            )
        try:
            recovery_summary = verify_host_enospc_recovery(
                resolved_recovery,
                executed_manifest,
                digest(executed_manifest_path),
                output_root=output_root,
                source_control=source_control,
                source_packet_id=source_packet_id,
                source_packet_manifest_sha256=digest(frozen_source_manifest_path),
            )
        except ResumeRefused as error:
            raise ContinuationRefused(str(error)) from error

    previous_index = frozen_source_manifest.get("completion_attempt", {}).get("index", 0)
    completion_index = previous_index + 1
    if completion_index not in (1, 2):
        raise ContinuationRefused("only two manual outcome-blind completion attempts are permitted")

    try:
        _source_ledger, _source_uploads, latest = build_ledger(export_root, source_control)
    except ResumeRefused as error:
        raise ContinuationRefused(str(error)) from error
    outer_iterations = int(source_control["outer_iterations"])
    if latest >= outer_iterations:
        raise ContinuationRefused("latest verified snapshot already reaches the frozen endpoint")
    if iteration != latest:
        raise ContinuationRefused(
            f"requested iteration {iteration} is not the latest verified resumable snapshot "
            f"({latest}); outcome-blind continuation must resume the latest snapshot"
        )

    snapshot_json_path = export_root / f"iteration-{iteration:04d}.json"
    snapshot_actor_path = export_root / f"iteration-{iteration:04d}-actor.pth"
    snapshot_info = json.loads(snapshot_json_path.read_text())
    snapshot_meta = snapshot_info.get("metadata", {})
    deferred_proxy = snapshot_meta.get("deferred_proxy") is not None
    needs_baseline_proxy = iteration == 0 and snapshot_meta.get("prepare_phase") not in {
        "ready",
        "skipped",
    }

    resolved_slug, short_suffix, resolved_job_id = _derive_identities(
        source_packet_id,
        completion_index=completion_index,
        iteration=iteration,
        slug_override=slug,
        job_id_override=job_id,
        date_tag=date_tag,
    )
    packet_name = f"packet-{resolved_slug}"
    resolved_parent = (repo_root.parent if checkout_parent is None else checkout_parent).resolve()
    resolved_artifacts = (
        resolved_parent / "snake-dqn-artifacts/sandbox-a-resident"
        if artifacts_root is None
        else artifacts_root.resolve()
    )
    for directory in (resolved_parent, resolved_artifacts):
        safe_path(directory)

    runtime_checkout = resolved_parent / f"snake-dqn-{resolved_slug}-runtime.noindex"
    broker_checkout = resolved_parent / f"snake-dqn-{resolved_slug}-broker.noindex"
    attempt_dir = resolved_artifacts / f"train01-attempt-{short_suffix}"
    state_dir = attempt_dir / "state"
    output_dir = attempt_dir / "output"
    for fresh in (
        runtime_checkout,
        broker_checkout,
        attempt_dir,
        state_dir,
        output_dir,
        repo_root / "research/sandbox_a_resident" / packet_name,
    ):
        safe_path(fresh)
        if fresh.exists() or fresh.is_symlink():
            raise ContinuationRefused(f"refusing to reuse existing continuation target: {fresh}")

    try:
        allowlist_entry = encode_or_verify_compact_snapshot(
            export_root / f"iteration-{iteration:04d}.pt",
            source_packet_id=source_packet_id,
            iteration=iteration,
            source_commit=str(source_control["source_commit"]),
        )
    except (CompactResumeRefused, FileExistsError) as error:
        raise ContinuationRefused(str(error)) from error

    envelope = compute_remaining_envelope(
        iteration,
        outer_iterations=outer_iterations,
        snapshot_interval=int(source_control["snapshot_interval"]),
        deferred_proxy=deferred_proxy,
        needs_baseline_proxy=needs_baseline_proxy,
    )
    storage_projection = compute_storage_projection(
        output_dir,
        remaining_snapshot_count=int(envelope["remaining_snapshot_count"]),
        reference_original_pt_bytes=int(allowlist_entry["original"]["size"]),
        reference_derivative_bytes=int(allowlist_entry["derivative"]["size"]),
        reference_manifest_bytes=int(allowlist_entry["manifest"]["size"]),
        reference_actor_bytes=snapshot_actor_path.stat().st_size,
        reference_json_bytes=snapshot_json_path.stat().st_size,
        snapshot_format="compact",
        safety_margin_bytes=safety_margin_bytes,
    )

    _git(repo_root, "worktree", "add", "--detach", str(runtime_checkout), "HEAD")
    _ensure_git_identity(runtime_checkout, repo_root)
    runtime_allowlist = runtime_checkout / "research/sandbox_a_resident/compact_allowlist.json"
    register_compact_allowlist_entry(allowlist_entry, path=runtime_allowlist)
    if _git(runtime_checkout, "status", "--porcelain", "--untracked-files=all"):
        _git(runtime_checkout, "add", "research/sandbox_a_resident/compact_allowlist.json")
        _git(
            runtime_checkout,
            "commit",
            "-m",
            f"Allowlist compact resume for {source_packet_id} iteration-{iteration:04d}",
        )
    runtime_pin = _require_clean_qualified_checkout(runtime_checkout)

    _git(repo_root, "worktree", "add", "--detach", str(broker_checkout), runtime_pin)
    _ensure_git_identity(broker_checkout, repo_root)

    successor_control = {**source_control, "source_commit": runtime_pin}
    expected_control = asdict(
        TrainControl(
            study=STUDY,
            source_commit=runtime_pin,
            allocation_index=int(successor_control.get("allocation_index", 0)),
            train_seed_offset=int(successor_control.get("train_seed_offset", 0)),
        )
    )
    if successor_control != expected_control:
        raise ContinuationRefused("successor control differs from frozen production recipe")
    allocations = seed_allocations(successor_control)
    if allocations != seed_allocations(source_control):
        raise ContinuationRefused("successor seed/proxy bank allocations changed")

    broker_registry = (broker_checkout / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json").resolve()
    registry_check = json.loads((source_packet_dir / "registry-check.json").read_text())
    if (
        registry_check.get("status") != "PASS"
        or registry_check.get("allocations") != allocations
        or registry_check.get("registry_sha256") != digest(broker_registry)
    ):
        raise ContinuationRefused("source registry-check.json does not bind successor registry")

    template_inputs: list[dict[str, str]] = []
    for row in copy.deepcopy(executed_manifest["inputs"]):
        dest = row["dest"]
        if dest == "verified-ledger.json" or dest.startswith("resume/"):
            continue
        if dest == "registry.json":
            template_inputs.append(
                {
                    "path": str(broker_registry),
                    "dest": "registry.json",
                    "sha256": digest(broker_registry),
                }
            )
        else:
            resolved_input = safe_path(row["path"]).resolve()
            if digest(resolved_input) != row["sha256"]:
                raise ContinuationRefused(f"fixed input SHA-256 drifted: {dest}")
            template_inputs.append(
                {"path": str(resolved_input), "dest": dest, "sha256": row["sha256"]}
            )

    template_env = dict(executed_manifest["env"])
    template_env["SANDBOX_A_CONTROL_JSON"] = json.dumps(successor_control, sort_keys=True)
    template_env["SANDBOX_A_SOURCE_COMMIT"] = runtime_pin
    template_env.setdefault("SANDBOX_A_STALL_TIMEOUT_SECONDS", "900")

    successor_template = copy.deepcopy(executed_manifest)
    successor_template.update(
        {
            "commit": runtime_pin,
            "source_paths": source_paths(runtime_pin, repo_root=runtime_checkout),
            "env": template_env,
            "inputs": template_inputs,
            "sandbox_source_delivery": "background-sparse-v1",
            "timeout_minutes": envelope["timeout_minutes"],
            "requirements": {
                **copy.deepcopy(executed_manifest["requirements"]),
                "est_minutes": envelope["est_minutes"],
            },
            "budget": {
                "rate_units_per_hour": 8.0,
                "unit_cap": float(envelope["unit_cap"]),
                "reserve_units": 0,
                "other_assignments": {},
            },
            "technical_envelope": {
                "base_seconds": envelope["base_seconds"],
                "effective_seconds": envelope["effective_seconds"],
                "endpoint": "100663296 actor rows plus bounded preparation and proxy",
            },
        }
    )

    try:
        resume_packet = completion_packet(
            executed_manifest,
            receipt,
            export_root,
            source_control,
            completion_index=completion_index,
            job_id=resolved_job_id,
            output_dir=output_dir,
            original_manifest_sha256=digest(executed_manifest_path),
            successor_manifest=successor_template,
            successor_control=successor_control,
            compact_source_packet_id=source_packet_id,
            allowlist_path=runtime_allowlist,
            output_root=output_root,
            supervisor_log_path=resolved_sup_log,
            recovery_evidence_path=resolved_recovery,
            source_packet_id=source_packet_id,
            source_packet_manifest_sha256=digest(frozen_source_manifest_path),
        )
    except ResumeRefused as error:
        raise ContinuationRefused(str(error)) from error

    packet_dir = broker_checkout / "research/sandbox_a_resident" / packet_name
    packet_dir.mkdir(parents=True, exist_ok=False)
    write_once(packet_dir / "control.json", successor_control)
    write_once(packet_dir / "registry-check.json", registry_check)
    write_once(packet_dir / "verified-ledger.json", resume_packet.ledger)
    committed_recovery_sha256: str | None = None
    if resolved_recovery is not None:
        packet_recovery_file = packet_dir / "recovery-evidence.json"
        publish(packet_recovery_file, lambda dst: dst.write_bytes(resolved_recovery.read_bytes()))
        committed_recovery_sha256 = digest(packet_recovery_file)
        assert recovery_summary is not None
        if committed_recovery_sha256 != recovery_summary["certificate_sha256"]:
            raise ContinuationRefused("recovery-evidence.json digest drifted during copy")

    broker_manifest = resume_packet.manifest
    broker_manifest["inputs"].append(
        {
            "path": str(packet_dir / "verified-ledger.json"),
            "dest": "verified-ledger.json",
            "sha256": digest(packet_dir / "verified-ledger.json"),
        }
    )
    broker_manifest["sandbox_a"] = {
        "seed_allocations": allocations,
        "seed_audit_sha256": digest(packet_dir / "registry-check.json"),
    }
    broker_manifest["sandbox_admission"] = {
        "schema": "standing-a100-development-sandbox-admission/v1",
        "manifest_path": str(packet_dir / "sandbox.json"),
    }
    write_once(packet_dir / "dispatch-manifest.json", broker_manifest)

    source_sandbox = json.loads((source_packet_dir / "sandbox.json").read_text())
    old_kinds = {
        row["sha256"]: row["kind"]
        for row in source_sandbox["inputs"]
        if "sha256" in row and row["kind"] in {"frozen-checkpoint", "frozen-opponent", "registry"}
    }
    old_kinds[digest(broker_registry)] = "registry"
    worlds = [row for row in source_sandbox["inputs"] if row["kind"] == "worlds"]
    sandbox = copy.deepcopy(source_sandbox)
    sandbox["inputs"] = worlds + [
        {
            "kind": (
                "training-export-sidecar"
                if row["dest"].startswith("resume/")
                and row["dest"].endswith((".sha256", ".complete"))
                else (
                    "training-export-derivative"
                    if row["dest"].startswith("resume/")
                    and row["dest"].endswith((COMPACT_DERIVATIVE_SUFFIX, COMPACT_MANIFEST_SUFFIX))
                    else (
                        "training-export"
                        if row["dest"].startswith("resume/")
                        else (
                            "verified-ledger"
                            if row["dest"] == "verified-ledger.json"
                            else old_kinds[row["sha256"]]
                        )
                    )
                )
            ),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in broker_manifest["inputs"]
    ]
    sandbox["timeout"] = {
        "base_seconds": envelope["base_seconds"],
        "seconds": envelope["effective_seconds"],
        "purpose": "runaway-guard",
    }
    sandbox["outputs"] = {
        "label": "simulator-D",
        "paths": [str(output_dir)],
        "broker_manifest": {
            "path": str(packet_dir / "dispatch-manifest.json"),
            "sha256": digest(packet_dir / "dispatch-manifest.json"),
        },
    }
    _write_ten_line_sandbox(packet_dir / "sandbox.json", sandbox)

    launch_bound = measure_launch_exec_bound(runtime_checkout, broker_manifest, runs=launch_runs)
    staging_projection = compute_staging_projection(
        broker_manifest,
        runtime_checkout,
        source_tar_gz_bytes=int(launch_bound["source_tar_gz_bytes"]),
    )

    dispatch_command = _format_dispatch_command(
        broker_checkout, runtime_checkout, packet_dir, state_dir, resolved_job_id
    )
    dispatch_lines = [
        f"TRAIN-01 continuation {packet_name}: prepared only; operator dispatch required.",
        f"Runtime pin: {runtime_pin}",
        (
            f"Continues {source_packet_id} ({executed_manifest['job_id']}) from Mac-verified "
            f"iteration-{iteration:04d} via lossless compact derivative "
            f"({allowlist_entry['derivative']['size']} B .pt.xz + "
            f"{allowlist_entry['manifest']['size']} B tensor manifest; "
            f"completion_attempt={completion_index}/2)."
        ),
        (
            f"Remaining workload: {envelope['remaining_updates']}/{outer_iterations} updates "
            f"({envelope['remaining_actor_rows']} actor rows), {envelope['proxy_count']} scheduled "
            f"E96 H5000 proxies {envelope['scheduled_proxy_updates']}; same seed domain "
            f"(allocation_index={successor_control['allocation_index']}, "
            f"train_seed_offset={successor_control['train_seed_offset']}) and identical "
            f"monitor (400000000, 288) / nomination (500000000, 768) banks."
        ),
        (
            f"Source delivery: background-sparse-v1 (3x local spawn <2.0 s, max "
            f"{max(r['launch_seconds'] for r in launch_bound['runs']):.4f} s); "
            f"{staging_projection['private_input_count']} private inputs "
            f"({staging_projection['private_input_bytes']} B) + supervisor/fallback files = "
            f"{staging_projection['total_upload_bytes']} B across "
            f"{staging_projection['total_upload_file_count']} uploads "
            f"({staging_projection['nominal_seconds']:.2f} s nominal / "
            f"{staging_projection['worst_case_seconds']:.2f} s conservative, "
            f"{staging_projection['fraction_of_300s_cap_nominal'] * 100:.1f}%/"
            f"{staging_projection['fraction_of_300s_cap_worst'] * 100:.1f}% of 300 s cap)."
        ),
        (
            f"Mac destination storage projection (compact-at-source periodic exports): "
            f"{storage_projection['remaining_snapshot_count']} remaining snapshots = "
            f"{storage_projection['remaining_exports_nominal_bytes']} B "
            f"({storage_projection['remaining_exports_nominal_mb']:.2f} MB < 1 GB; "
            f"{storage_projection['remaining_exports_conservative_gib']:.3f} GiB conservative; "
            f"{storage_projection['export_byte_reduction_factor']}x smaller than "
            f"{storage_projection['raw_counterfactual_exports_gib']:.2f} GiB raw); "
            f"required with temp+logs+{storage_projection['safety_margin_gib']:.1f} GiB margin = "
            f"{storage_projection['required_conservative_with_margin_gib']:.3f} GiB vs "
            f"{storage_projection['filesystem_free_gib']:.3f} GiB free "
            f"(+{storage_projection['free_above_safety_margin_at_temp_peak_gib']:.3f} GiB above "
            f"safety margin)."
        ),
        (
            f"Remaining A100 wall-time projection: {envelope['nominal_hours']:.2f} h nominal "
            f"({envelope['conservative_base_hours']:.2f} h conservative) -> "
            f"est_minutes={envelope['est_minutes']} / "
            f"timeout_minutes={envelope['timeout_minutes']} "
            f"(2x guard), unit_cap={envelope['unit_cap']} u; progress <=60 s; "
            f"faulthandler stall watchdog 900 s (exit 1); SIGUSR1 <15 s path active."
        ),
    ]
    if recovery_summary is not None:
        dispatch_lines.append(
            f"Prior attempt stop evidence: host ENOSPC recovery certificate "
            f"({recovery_summary['certificate_sha256'][:16]}...; prior status=INCOMPLETE, "
            f"original receipt absent as-is, post-hoc release verified: sessions empty, "
            f"assignments=0, rate=0.0, {recovery_summary['polls_scanned']} retained polls + "
            f"full supervisor log scanned with 0 scientific-failure hits)."
        )
        if recovery_summary.get("stale_active_present_now") and recovery_summary.get(
            "operator_reconcile_command"
        ):
            dispatch_lines.extend(
                [
                    "",
                    "Pre-dispatch stale broker record reconciliation (run once before live "
                    "dispatch):",
                    str(recovery_summary["operator_reconcile_command"]),
                ]
            )
    dispatch_lines.extend(
        [
            "",
            "Exact operator dispatch (from any shell):",
            dispatch_command,
            "",
            "Dry-run: append --dry-run. Preserve detached checkouts and prior evidence.",
        ]
    )
    with (packet_dir / "DISPATCH.txt").open("x") as stream:
        stream.write("\n".join(dispatch_lines) + "\n")

    source_sha256 = {
        rel_path: hashlib.sha256((runtime_checkout / rel_path).read_bytes()).hexdigest()
        for rel_path in broker_manifest["source_paths"]
    }
    if receipt is not None:
        prior_stop_kind = "supervisor_receipt"
        prior_receipt_present = True
        prior_receipt_sha256: str | None = digest(receipt_path)
        prior_failure_class = receipt.get("failure_class")
        prior_release_proof: dict[str, Any] = {
            "kind": "supervisor_receipt",
            "original_receipt_present": True,
            "stop_verified": receipt.get("stop_verified"),
            "cleanup_error": receipt.get("cleanup_error"),
        }
    else:
        assert recovery_summary is not None and resolved_recovery is not None
        prior_stop_kind = "host_enospc_recovery_certificate"
        prior_receipt_present = False
        prior_receipt_sha256 = None
        prior_failure_class = HOST_ENOSPC_FAILURE_CLASS
        prior_release_proof = {
            "kind": "post_hoc_host_enospc_recovery",
            "prior_attempt_status": "INCOMPLETE",
            "original_receipt_present": False,
            "original_terminal_status": "UNKNOWN_UNAVAILABLE",
            "original_supervisor_error": "UNKNOWN_UNAVAILABLE",
            "original_cleanup_error": "UNKNOWN_UNAVAILABLE",
            "post_hoc_release_verified": True,
            "account_status_capture_sha256": recovery_summary["account_status_capture_sha256"],
            "captured_at_utc": recovery_summary["captured_at_utc"],
            "broker_reconciliation_mode": recovery_summary["broker_reconciliation_mode"],
            "recover_receipt": recovery_summary["recover_receipt"],
            "supervisor_log_sha256": recovery_summary["supervisor_log_sha256"],
            "polls_scanned": recovery_summary["polls_scanned"],
            "polls_sha256": recovery_summary["polls_sha256"],
        }
    seal: dict[str, Any] = {
        "schema": "sandbox-a-train01-continuation-seal/v1",
        "prepared_not_dispatched": True,
        "dispatch_identity": resolved_job_id,
        "packet_name": packet_name,
        "prior_packet": source_packet_id,
        "prior_job_id": executed_manifest["job_id"],
        "prior_output_dir": str(output_root),
        "prior_supervisor_log": (
            {
                "path": str(resolved_sup_log),
                "size": resolved_sup_log.stat().st_size,
                "sha256": digest(resolved_sup_log),
            }
            if resolved_sup_log is not None
            else None
        ),
        "prior_executed_manifest_sha256": digest(executed_manifest_path),
        "prior_stop_evidence_kind": prior_stop_kind,
        "prior_receipt_present": prior_receipt_present,
        "prior_receipt_sha256": prior_receipt_sha256,
        "prior_failure_class": prior_failure_class,
        "prior_recovery_evidence_schema": (
            HOST_ENOSPC_RECOVERY_SCHEMA if recovery_summary is not None else None
        ),
        "prior_recovery_evidence_source_path": (
            str(resolved_recovery) if resolved_recovery is not None else None
        ),
        "prior_recovery_evidence_sha256": committed_recovery_sha256,
        "prior_release_proof": prior_release_proof,
        "completion_attempt_index": completion_index,
        "resume_from_iteration": iteration,
        "remaining_updates": envelope["remaining_updates"],
        "scheduled_proxy_updates": envelope["scheduled_proxy_updates"],
        "periodic_export_format": "compact",
        "compact_allowlist_entry": allowlist_entry,
        "runtime_checkout": str(runtime_checkout),
        "runtime_pin": runtime_pin,
        "broker_checkout": str(broker_checkout),
        "qualified_transport_tree": TRANSPORT_TREE,
        "sandbox_source_delivery": "background-sparse-v1",
        "local_launch_bound_seconds": 2,
        "launch_exec_bound": launch_bound,
        "upload_staging_projection": staging_projection,
        "mac_storage_projection": storage_projection,
        "remaining_wall_time_projection": envelope,
        "est_minutes": envelope["est_minutes"],
        "timeout_minutes": envelope["timeout_minutes"],
        "unit_cap": envelope["unit_cap"],
        "technical_envelope": broker_manifest["technical_envelope"],
        "control_sha256": digest(packet_dir / "control.json"),
        "control_allocation_sha256": allocation(successor_control),
        "registry_audit_sha256": digest(packet_dir / "registry-check.json"),
        "ledger_sha256": digest(packet_dir / "verified-ledger.json"),
        "manifest_sha256": digest(packet_dir / "dispatch-manifest.json"),
        "sandbox_sha256": digest(packet_dir / "sandbox.json"),
        "dispatch_txt_sha256": digest(packet_dir / "DISPATCH.txt"),
        "source_sha256": source_sha256,
    }
    write_once(packet_dir / "SEAL.json", seal)

    _git(broker_checkout, "add", f"research/sandbox_a_resident/{packet_name}")
    _git(
        broker_checkout,
        "commit",
        "-m",
        f"Seal TRAIN-01 continuation {packet_name} from {source_packet_id} "
        f"iteration-{iteration:04d}",
    )
    broker_seal = _require_clean_qualified_checkout(broker_checkout)

    venv_target = Path(sys.prefix)
    if not (venv_target / "bin/python").exists():
        venv_target = Path(os.path.abspath(sys.executable)).parents[1]
    if (venv_target / "bin/python").exists():
        for checkout in (runtime_checkout, broker_checkout):
            ignored = subprocess.run(
                ["git", "-C", str(checkout), "check-ignore", "-q", "venv"],
                check=False,
                capture_output=True,
            )
            if ignored.returncode != 0:
                common_raw = _git(checkout, "rev-parse", "--git-common-dir")
                common_dir = Path(common_raw)
                if not common_dir.is_absolute():
                    common_dir = (checkout / common_dir).resolve()
                exclude_file = common_dir / "info/exclude"
                exclude_file.parent.mkdir(parents=True, exist_ok=True)
                existing_exclude = exclude_file.read_text() if exclude_file.is_file() else ""
                if "venv" not in existing_exclude.splitlines():
                    prefix = "" if not existing_exclude or existing_exclude.endswith("\n") else "\n"
                    exclude_file.write_text(f"{existing_exclude}{prefix}venv\n")
            checkout_venv = checkout / "venv"
            if not checkout_venv.exists() and not checkout_venv.is_symlink():
                checkout_venv.symlink_to(venv_target)

    _require_clean_qualified_checkout(runtime_checkout)
    _require_clean_qualified_checkout(broker_checkout)

    validation_output = run_packet_validation(broker_checkout, packet_dir)
    dry_run_output = (
        "SKIPPED_DRY_RUN (--skip-dry-run; committed packet validation PASS)"
        if skip_dry_run
        else run_dispatch_dry_run(
            broker_checkout, runtime_checkout, packet_dir, state_dir, resolved_job_id
        )
    )
    if state_dir.exists() or output_dir.exists():
        raise ContinuationRefused("dry-run unexpectedly created state_dir or output_dir")

    return {
        "schema": "sandbox-a-train01-continuation-result/v1",
        "status": "PREPARED_NOT_DISPATCHED",
        "allocations": 0,
        "source_packet_id": source_packet_id,
        "source_job_id": executed_manifest["job_id"],
        "prior_stop_evidence_kind": prior_stop_kind,
        "prior_receipt_present": prior_receipt_present,
        "prior_recovery_evidence_sha256": committed_recovery_sha256,
        "resume_from_iteration": iteration,
        "completion_attempt_index": completion_index,
        "packet_name": packet_name,
        "packet_dir": str(packet_dir),
        "job_id": resolved_job_id,
        "runtime_checkout": str(runtime_checkout),
        "runtime_pin": runtime_pin,
        "broker_checkout": str(broker_checkout),
        "broker_seal": broker_seal,
        "qualified_transport_tree": TRANSPORT_TREE,
        "state_dir": str(state_dir),
        "output_dir": str(output_dir),
        "state_exists": state_dir.exists(),
        "output_exists": output_dir.exists(),
        "periodic_export_format": "compact",
        "compact_allowlist_entry": allowlist_entry,
        "launch_exec_bound": launch_bound,
        "upload_staging_projection": staging_projection,
        "mac_storage_projection": storage_projection,
        "remaining_envelope": envelope,
        "packet_validation": validation_output,
        "dry_run_output": dry_run_output,
        "dispatch_command": dispatch_command,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-attempt",
        required=True,
        type=Path,
        help="completed mechanical-stop attempt directory (or its output/ subdirectory)",
    )
    parser.add_argument(
        "--iteration",
        required=True,
        type=int,
        help="latest completed Mac-verified periodic snapshot iteration (< 384) to continue from",
    )
    parser.add_argument(
        "--source-packet",
        type=Path,
        help="optional override for frozen source packet directory",
    )
    parser.add_argument(
        "--recovery-evidence",
        type=Path,
        help="explicit create-only host ENOSPC recovery certificate for a receipt-less attempt",
    )
    parser.add_argument(
        "--supervisor-log",
        type=Path,
        help="optional explicit supervisor log file for full retained fault-history scan",
    )
    parser.add_argument(
        "--slug",
        help="optional successor slug (default: train01-<15|16|17>-c<1|2>-i<NNNN>)",
    )
    parser.add_argument("--job-id", help="optional successor dispatch job_id")
    parser.add_argument("--date-tag", help="optional YYYYMMDD tag for default job_id")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--checkout-parent", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument(
        "--skip-dry-run",
        action="store_true",
        help="validate committed packet bindings without spawning dispatch.sh --dry-run",
    )
    args = parser.parse_args(argv)
    result = build_continuation_packet(
        args.source_attempt,
        args.iteration,
        source_packet=args.source_packet,
        slug=args.slug,
        job_id=args.job_id,
        date_tag=args.date_tag,
        repo_root=args.repo_root,
        checkout_parent=args.checkout_parent,
        artifacts_root=args.artifacts_root,
        skip_dry_run=args.skip_dry_run,
        recovery_evidence=args.recovery_evidence,
        supervisor_log=args.supervisor_log,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
