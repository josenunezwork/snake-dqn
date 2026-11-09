"""Synthetic positive, negative, tamper, lineage, live-proof, and baseline-equivalence tests
for the Mac-only Watch6 V3 accounting adapter (Astra Ruling Points 1-7).
"""

from __future__ import annotations

import builtins
import functools
import importlib.machinery
import importlib.util
import io
import json
import os
import pickle
import pickletools
import platform
import py_compile
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import admission as v3_admission
from research.watch6_search_efficacy_v3 import final_look as v3_final_look
from research.watch6_search_efficacy_v3 import (
    mirror,
)
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3 import ratify as v3_ratify
from research.watch6_search_efficacy_v3 import resume as v3_resume
from research.watch6_search_efficacy_v3 import (
    run,
)
from tests.test_watch6_search_efficacy_v3 import (
    VALID_TRANSPORT_TRACEBACK,
    _synthetic_cost_report,
    _synthetic_v3_rows,
)
from tools.ops.package_spec import (
    digest,
    empty_ledger,
    publish_immutable,
    publish_json,
)
from tools.watch6_v3_accounting import (
    ADAPTER_FILES,
    ADAPTER_ROOT,
    AMENDMENT_V5_REL,
    AMENDMENT_V5_SHA256,
    AMENDMENT_V6_REL,
    AMENDMENT_V6_SHA256,
    BASELINE_COMMIT,
    BASELINE_NAMESPACE,
    BASELINE_SCIENCE,
    MAIN_CHECKOUT_ROOT,
    PINNED_SUPERVISOR_REL,
    PINNED_SUPERVISOR_SHA256,
    REGISTRATION_V3_SHA256,
    RETAINED_PENDING_CLEANUP_ERROR,
    RULING_2_REL,
    RULING_2_SHA256,
    RULING_3_REL,
    RULING_3_SHA256,
    RULING_VERIFY_5_REL,
    RULING_VERIFY_5_SHA256,
    RULING_VERIFY_6_REL,
    RULING_VERIFY_6_SHA256,
    SHARD00_EXEC_MANIFEST_SHA256,
    SHARD00_PATH_RULING_REL,
    SHARD00_PATH_RULING_SHA256,
    SHARD00_PROOF_REL,
    SHARD00_PROOF_SHA256,
    SHARD00_RECEIPT_REL,
    SHARD00_RECEIPT_SHA256,
    SHARD12_EVIDENCE_SIDECAR_REL,
    SHARD12_EVIDENCE_SIDECAR_SHA256,
    SHARD12_EXEC_MANIFEST_SHA256,
    SHARD12_LIVE_MANIFEST_SHA256,
    SHARD12_OPS_LOG_SHA256,
    SHARD12_OPS_MANIFEST_SHA256,
    SHARD12_PATH_RULING_REL,
    SHARD12_PATH_RULING_SHA256,
    SHARD12_RECEIPT_REL,
    SHARD12_RECEIPT_SHA256,
    V3_BROKER_ROOT,
    active_broker_isolation,
    adapter_commit,
    adapter_files_map,
    adapter_inventory_sha256,
    admission,
    artifacts_root,
    bind_broker_protocol,
    compute_broker_identity,
    dispatch,
    final_look,
    isolated_broker_execution,
    prepare,
    proof,
    ratify,
    recover,
    resume,
    used_numba_cache_dirs,
    verify_baseline_closure,
)


def _render_amendment_markdown(repo: Path | None = None, commit_sha: str | None = None) -> str:
    root = (repo or ADAPTER_ROOT).resolve()
    c_sha = commit_sha or adapter_commit(root, require_clean=False)
    f_map = adapter_files_map(root)
    inv_sha = adapter_inventory_sha256(root)
    lines = [
        "# Watch6 V3 Pre-Look Accounting Amendment V6.1 (2026-11-01)",
        "",
        f"- Namespace: `{BASELINE_NAMESPACE}`",
        f"- V3 Broker Root: `{V3_BROKER_ROOT}`",
        f"- Baseline Runtime Commit: `{BASELINE_COMMIT}`",
        f"- Baseline Science Identity: `{BASELINE_SCIENCE}`",
        f"- Original Registration (`REGISTRATION_V3.md`) SHA256: `{REGISTRATION_V3_SHA256}`",
        f"- Ruling 2 (`{RULING_2_REL}`) SHA256: `{RULING_2_SHA256}`",
        f"- Ruling 3 (`{RULING_3_REL}`) SHA256: `{RULING_3_SHA256}`",
        (
            f"- Shard-00 Path Ruling (`{SHARD00_PATH_RULING_REL}`) SHA256: "
            f"`{SHARD00_PATH_RULING_SHA256}`"
        ),
        f"- Shard-00 Unallocated Proof (`{SHARD00_PROOF_REL}`) SHA256: `{SHARD00_PROOF_SHA256}`",
        f"- Shard-00 Refused Receipt (`{SHARD00_RECEIPT_REL}`) SHA256: `{SHARD00_RECEIPT_SHA256}`",
        f"- Shard-00 Executed Manifest SHA256: `{SHARD00_EXEC_MANIFEST_SHA256}`",
        f"- Prior Amendment V5 (`{AMENDMENT_V5_REL}`) SHA256: `{AMENDMENT_V5_SHA256}`",
        f"- Prior Verify-5 Ruling (`{RULING_VERIFY_5_REL}`) SHA256: `{RULING_VERIFY_5_SHA256}`",
        f"- Prior Amendment V6 (`{AMENDMENT_V6_REL}`) SHA256: `{AMENDMENT_V6_SHA256}`",
        f"- Prior Verify-6 Ruling (`{RULING_VERIFY_6_REL}`) SHA256: `{RULING_VERIFY_6_SHA256}`",
        (
            f"- Shard-12 Path Ruling (`{SHARD12_PATH_RULING_REL}`) SHA256: "
            f"`{SHARD12_PATH_RULING_SHA256}`"
        ),
        (
            f"- Shard-12 Transport Evidence Sidecar (`{SHARD12_EVIDENCE_SIDECAR_REL}`) SHA256: "
            f"`{SHARD12_EVIDENCE_SIDECAR_SHA256}`"
        ),
        f"- Shard-12 Receipt (`{SHARD12_RECEIPT_REL}`) SHA256: `{SHARD12_RECEIPT_SHA256}`",
        f"- Shard-12 Executed Manifest SHA256: `{SHARD12_EXEC_MANIFEST_SHA256}`",
        f"- Shard-12 Ops Source Manifest SHA256: `{SHARD12_OPS_MANIFEST_SHA256}`",
        f"- Shard-12 Ops Live Manifest SHA256: `{SHARD12_LIVE_MANIFEST_SHA256}`",
        f"- Shard-12 Operational Log SHA256: `{SHARD12_OPS_LOG_SHA256}`",
        f"- Pinned Supervisor (`{PINNED_SUPERVISOR_REL}`) SHA256: `{PINNED_SUPERVISOR_SHA256}`",
        f"- Adapter Commit: `{c_sha}`",
        f"- Adapter Inventory SHA256: `{inv_sha}`",
        "",
        "## Complete 10-File Adapter Inventory",
    ]
    for rel_path, sha in sorted(f_map.items()):
        lines.append(f"- `{rel_path}`: `{sha}`")
    lines.append("")
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def _cached_synthetic_v3_rows_by_slot() -> dict[int, dict]:
    with bind_broker_protocol(V3_BROKER_ROOT):
        return {r["allocation"]["index"]: r for r in _synthetic_v3_rows()}


@functools.lru_cache(maxsize=1)
def _cached_broker_source_paths() -> list[str]:
    with bind_broker_protocol(V3_BROKER_ROOT):
        return list(p.source_paths())


@functools.lru_cache(maxsize=1)
def _cached_broker_allocation() -> list[dict]:
    with bind_broker_protocol(V3_BROKER_ROOT):
        return list(p.allocation())


def _write_immutable_text(path: Path, content: str) -> Path:
    publish_immutable(path, content.encode("utf-8"))
    return path


def _write_accounting_ruling_fixture(
    tmp_path: Path,
    *,
    verdict: str = "APPROVE",
) -> tuple[Path, Path]:
    commit_sha = adapter_commit(ADAPTER_ROOT, require_clean=False)
    amend_path = _write_immutable_text(
        tmp_path / "WATCH6_V3_ACCOUNTING_AMENDMENT_V5_20261031.md",
        _render_amendment_markdown(ADAPTER_ROOT, commit_sha),
    )
    ruling_md = tmp_path / "ASTRA_RERATIFICATION_RULING.md"
    ruling_md.write_text(
        "# Watch6 V3 Accounting Adapter Re-Ratification\n\n"
        f"Verdict: {verdict}\n"
        "STATUS: RATIFIED\n"
        "Judge: Astra · 2026-10-31\n",
        encoding="utf-8",
    )
    ruling_json = tmp_path / "astra-accounting-ruling.json"
    ratify.record_ruling(
        ruling_md,
        amendment_md=amend_path,
        out=ruling_json,
        require_clean_git=False,
    )
    return amend_path, ruling_json


def _write_v3_attempt(
    out_dir: Path,
    *,
    shard: int,
    mode: str = "pass",
    status: str | None = None,
    failure_class: str | None = None,
    error: str | None = None,
    endpoint: str | None = "m-s-test-endpoint",
    stop_verified: bool = True,
    cleanup_error: str | None = None,
    export_slots: list[int] | None = None,
    pair_rows_by_slot: dict[int, dict] | None = None,
    prestart_retries_used: int = 0,
    compute_completion_attempt: int = 0,
    prior_receipts: list[dict] | None = None,
    completion_attempt: int | None = None,
    commit: str = BASELINE_COMMIT,
    gpu: str = "L4",
    prefer: list[str] | None = None,
    job_log_tail: str | None = None,
) -> Path:
    """Create a realistic V3 supervisor output directory pinned to BASELINE_COMMIT & L4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    priors = list(prior_receipts or [])
    tot_dispatches = 1 + prestart_retries_used + compute_completion_attempt
    req: dict[str, object] = {
        "gpu": gpu,
        "allowed_gpus": [gpu],
        "min_vcpu": 12,
        "est_minutes": 117,
    }
    if prefer is not None:
        req["prefer"] = list(prefer)

    broker_science = BASELINE_SCIENCE
    manifest: dict = {
        "job_id": f"watch6-v3-shard-{shard:02d}-{out_dir.name}"[:64],
        "output_dir": str(out_dir),
        "commit": commit,
        "script": p.SCRIPT,
        "source_paths": list(_cached_broker_source_paths()),
        "device": "cpu",
        "timeout_minutes": 234,
        "requirements": req,
        "budget": {
            "unit_cap": 9.75,
            "rate_units_per_hour": 2.5,
            "reserve_units": 1.95,
            "other_assignments": {},
        },
        "env": {
            "WATCH6_FIXTURE": "0",
            "WATCH6_SHARD": str(shard),
            "WATCH6_SHARDS": str(p.SHARDS),
            "WATCH6_WORKERS": str(p.WORKERS),
            "WATCH6_SCIENCE": broker_science,
            "SNAKE_WATCH_SOURCE": "live",
        },
        "watch6_v3": {
            "namespace": p.NAMESPACE,
            "science": broker_science,
            "primary_allocation": list(_cached_broker_allocation()),
            "shard": shard,
            "phase": "primary",
            "prestart_retries_used": prestart_retries_used,
            "compute_completion_attempt": compute_completion_attempt,
            "total_dispatches_for_shard": tot_dispatches,
            "prior_receipts": priors,
            "base_minutes": 117,
            "base_unit_cap": 4.875,
            "base_reserve_units": 0.975,
        },
    }
    if completion_attempt is not None:
        manifest["completion_attempt"] = completion_attempt

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    cli_calls: list[dict] = []
    terminal_obj: dict | None = None

    if mode == "pass":
        eff_status = status or "PASS"
        eff_fcls = failure_class
        eff_err = error
        (out_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
        (out_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
        (out_dir / "upload-measurement.json").write_text('{"elapsed_seconds": 65.0}\n')
        (out_dir / "source-delivery.json").write_text('{"transport": "github"}\n')
        (out_dir / "input-hashes.json").write_text("{}\n")
        (out_dir / "hardware.json").write_text('{"ready": true, "hardware": {"kind": "L4"}}\n')
        (out_dir / "poll-000000.json").write_text(
            json.dumps(
                {
                    "state": {
                        "status": "running",
                        "returncode": None,
                        "reason": None,
                        "error": None,
                        "unassign_error": None,
                    },
                    "terminal": None,
                    "job_log_tail": (
                        job_log_tail
                        if job_log_tail is not None
                        else f"WATCH6_V3_PROGRESS shard={shard} done=4/4 skipped=0 elapsed=90.0s\n"
                    ),
                    "supervisor_log_tail": "",
                }
            )
            + "\n"
        )
        terminal_obj = {"returncode": 0, "reason": None, "error": None}
        (out_dir / "terminal.json").write_text(json.dumps(terminal_obj) + "\n")
    elif mode == "prestart_staging":
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "RPC_EXEC_POLL_TIMEOUT"
        eff_err = error or VALID_TRANSPORT_TRACEBACK
        (out_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
        (out_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
        (out_dir / "upload-measurement.json").write_text(
            '{"elapsed_seconds": 120.0, "staging_completed": false}\n'
        )
        cli_calls = [{"subcommand": "upload", "returncode": 1}]
    elif mode == "prestart_unassigned":
        endpoint = None
        stop_verified = False
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "ACCOUNTING"
        eff_err = error or "AccountingError('account-wide assignment cap 6 (including notebooks)')"
    elif mode == "compute_transport":
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "RPC_EXEC_POLL_TIMEOUT"
        eff_err = error if error is not None else VALID_TRANSPORT_TRACEBACK
        (out_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
        (out_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
        (out_dir / "upload-measurement.json").write_text('{"elapsed_seconds": 65.0}\n')
        (out_dir / "source-delivery.json").write_text('{"transport": "github"}\n')
        (out_dir / "input-hashes.json").write_text("{}\n")
        (out_dir / "hardware.json").write_text('{"ready": true, "hardware": {"kind": "L4"}}\n')
        (out_dir / "poll-000000.json").write_text(
            json.dumps(
                {
                    "state": {
                        "status": "running",
                        "returncode": None,
                        "reason": None,
                        "error": None,
                        "unassign_error": None,
                    },
                    "terminal": None,
                    "job_log_tail": (
                        job_log_tail
                        if job_log_tail is not None
                        else f"WATCH6_V3_PROGRESS shard={shard} done=1/4 skipped=0 elapsed=55.0s\n"
                    ),
                    "supervisor_log_tail": "",
                }
            )
            + "\n"
        )
        cli_calls = [{"subcommand": "exec", "returncode": 1}]
    elif mode == "child_fail":
        eff_status = "INCOMPLETE"
        eff_fcls = "CHILD_NONZERO_EXIT"
        eff_err = error or "ValueError: mirror mismatch at frame 42"
        (out_dir / "hardware.json").write_text('{"ready": true, "hardware": {"kind": "L4"}}\n')
        terminal_obj = {"returncode": 1, "reason": "nonzero_exit", "error": eff_err}
        (out_dir / "terminal.json").write_text(json.dumps(terminal_obj) + "\n")
    else:
        raise ValueError(f"unknown mode: {mode}")

    exports = []
    alloc_table = _cached_broker_allocation()
    for slot in export_slots or []:
        alloc = alloc_table[slot]
        payload = (
            pair_rows_by_slot[slot]
            if pair_rows_by_slot and slot in pair_rows_by_slot
            else {"sealed": True, "index": slot}
        )
        pair_path = out_dir / "exports/efficacy" / f"pair-{slot:02d}.json"
        desc = publish_json(pair_path, payload)
        meta_desc = publish_json(
            pair_path.with_suffix(".meta.json"),
            {
                "namespace": p.NAMESPACE,
                "science": broker_science,
                "seed_kind": "primary",
                "allocation": alloc,
                "fixture": False,
                "sha256": desc["sha256"],
                "size": desc["size"],
            },
        )
        exports.append(
            {
                "path": f"efficacy/pair-{slot:02d}.json",
                "sha256": desc["sha256"],
                "size": desc["size"],
            }
        )
        exports.append(
            {
                "path": f"efficacy/pair-{slot:02d}.meta.json",
                "sha256": meta_desc["sha256"],
                "size": meta_desc["size"],
            }
        )

    receipt = {
        "job_id": manifest["job_id"],
        "commit": manifest["commit"],
        "manifest_sha256": digest(manifest_path),
        "endpoint": endpoint,
        "status": eff_status,
        "failure_class": eff_fcls,
        "error": eff_err,
        "stop_verified": stop_verified,
        "cleanup_error": cleanup_error,
        "terminal": terminal_obj,
        "exports": exports,
        "transfers": [],
        "contents_transfers": [],
        "cli_calls": cli_calls,
        "control_operations": [],
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_path


def _write_never_allocated_shard_fixture(
    base_dir: Path,
    *,
    ops_dir: Path,
    ops_name: str = "watch6-v3-fix3-shard-00",
    shard: int = 0,
    with_update_banner: bool = True,
    refusal_kind: str = "precondition",
    notebook_hash: str = "2e58e2f564224b838cd96581c0f8bd29",
) -> tuple[Path, Path, Path, Path, Path]:
    """Build a complete synthetic Ruling Point 4 never-allocated attempt + accounted sessions."""
    (ops_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (ops_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ops_dir / "dispatch" / ops_name).mkdir(parents=True, exist_ok=True)

    # Accounted active peer session so before/sessions_stdout/membership.retained are exercised:
    peer_out = base_dir / "peer-active-out"
    peer_out.mkdir(parents=True, exist_ok=True)
    (peer_out / "upload-measurement.json").write_text(
        '{"elapsed_seconds": 40.0}\n', encoding="utf-8"
    )
    peer_mf = {
        "job_id": "peer-active-l4-job",
        "commit": BASELINE_COMMIT,
        "output_dir": str(peer_out),
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 60},
    }
    peer_ops_mf = ops_dir / "manifests/peer-active-l4-job.json"
    peer_ops_mf.write_text(json.dumps(peer_mf, sort_keys=True) + "\n", encoding="utf-8")
    peer_exec_mf = peer_out / "manifest.json"
    peer_exec_mf.write_text(json.dumps(peer_mf, sort_keys=True) + "\n", encoding="utf-8")

    account_dir = base_dir / "colab-account"
    account_dir.mkdir(parents=True, exist_ok=True)
    peer_active_rec = account_dir / "active-snake-peer-active-l4-job-aaaaaaaaaaaa.json"
    peer_active_rec.write_text('{"endpoint": "m-s-peer1111"}\n', encoding="utf-8")

    out_dir = base_dir / "output-shard-00"
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"watch6-v3-shard-{shard:02d}-pkg01"
    generation = "d271c5f9b85947908f90f023c6c0f76c"
    session = f"snake-{job_id}-{generation[:12]}"

    broker_science = BASELINE_SCIENCE
    manifest = {
        "job_id": job_id,
        "output_dir": str(out_dir),
        "commit": BASELINE_COMMIT,
        "script": p.SCRIPT,
        "source_paths": list(_cached_broker_source_paths()),
        "device": "cpu",
        "timeout_minutes": 234,
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": 117,
        },
        "budget": {
            "unit_cap": 9.75,
            "rate_units_per_hour": 2.5,
            "reserve_units": 1.95,
            "other_assignments": {},
        },
        "env": {
            "WATCH6_FIXTURE": "0",
            "WATCH6_SHARD": str(shard),
            "WATCH6_SHARDS": str(p.SHARDS),
            "WATCH6_WORKERS": str(p.WORKERS),
            "WATCH6_SCIENCE": broker_science,
            "SNAKE_WATCH_SOURCE": "live",
        },
        "watch6_v3": {
            "namespace": p.NAMESPACE,
            "science": broker_science,
            "primary_allocation": list(_cached_broker_allocation()),
            "shard": shard,
            "phase": "primary",
            "prestart_retries_used": 0,
            "compute_completion_attempt": 0,
            "total_dispatches_for_shard": 1,
            "prior_receipts": [],
            "base_minutes": 117,
            "base_unit_cap": 4.875,
            "base_reserve_units": 0.975,
        },
    }
    exec_mf_path = out_dir / "manifest.json"
    exec_mf_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    ops_mf_path = ops_dir / "manifests" / f"{ops_name}.json"
    ops_mf_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    log_path = ops_dir / "logs" / f"{ops_name}.log"
    log_path.write_text("supervisor refused allocation\nEXIT=1\n", encoding="utf-8")

    banner = (
        "[colab] A new version of Colab CLI is available: 0.7.5 -> 0.7.7\n"
        "[colab] Run 'colab update' or visit https://github.com/googlecolab/colab-cli/releases\n"
        '[colab] To silence this check, set "enable_update_check": false in config.json\n\n'
        if with_update_banner
        else ""
    )
    if refusal_kind == "precondition":
        detail = (
            f"colab new failed (1): {banner}"
            f"[colab] Creating session '{session}'...\n"
            f"{dispatch.ALLOCATION_UNAVAILABLE}\n"
        )
    elif refusal_kind == "503_padded":
        padded_nbh = (
            f"{notebook_hash[:8]}_{notebook_hash[8:12]}_{notebook_hash[12:16]}_"
            f"{notebook_hash[16:20]}_{notebook_hash[20:]}........"
        )
        detail = (
            f"colab new failed (1): {banner}"
            f"[colab] Creating session '{session}'...\n"
            "ColabRequestError: Failed to issue request POST "
            "https://colab.research.google.com/tun/m/assign"
            f"?nbh={padded_nbh}&variant=GPU&accelerator=L4: "
            "503 Service Unavailable\n"
        )
    else:
        detail = (
            f"colab new failed (1): {banner}"
            f"[colab] Creating session '{session}'...\n"
            "ColabRequestError: Failed to issue request POST "
            "https://colab.research.google.com/tun/m/assign"
            f"?nbh={notebook_hash}&variant=GPU&accelerator=L4: "
            "503 Service Unavailable\n"
        )

    baseline_assignments = [
        {
            "endpoint": "m-s-peer1111",
            "rate_units_per_hour": 1.94,
            "session": "?",
            "source": "broker.active",
        }
    ]
    (out_dir / "accounting-baseline.json").write_text(
        json.dumps(
            {
                "assignments": baseline_assignments,
                "baseline_balance_units": 900.0,
                "baseline_rate_units_per_hour": 1.94,
                "declared_other_rate_units_per_hour": 1.94,
                "unaccounted_endpoints": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = {
        "accounting": {
            "allocated_endpoint": None,
            "assignments": baseline_assignments,
            "calibration": None,
            "last_crosscheck": None,
            "observed_rate_units_per_hour": None,
            "other_assignments": {},
            "rate_units_per_hour": 2.5,
            "reserve_units": 1.95,
            "unaccounted_endpoints": [],
            "unit_cap": 9.75,
        },
        "cleanup_error": RETAINED_PENDING_CLEANUP_ERROR,
        "cli_calls": [
            {"attempt": 1, "command": "version", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "sessions", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "usage", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "new", "operation": "new", "status": "failed"},
        ],
        "commit": BASELINE_COMMIT,
        "contents_transfers": [],
        "control_operations": [],
        "endpoint": None,
        "error": f"Refused({detail!r})",
        "exports": [],
        "failure_class": "SUPERVISOR",
        "generation": generation,
        "job_id": job_id,
        "manifest_sha256": digest(exec_mf_path),
        "session": session,
        "status": "INCOMPLETE",
        "stop_verified": False,
        "terminal": None,
        "transfers": [],
        "usage_after": None,
        "usage_before": {"active_assignments": 1, "balance": 900.0},
        "warmup_seconds": 0.0,
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    lane_state = ops_dir / "state" / ops_name / job_id
    lane_state.mkdir(parents=True, exist_ok=True)
    pending_path = account_dir / f"pending-{session}.json"
    # Note: pending_path is NOT created on disk because pending_action == "unlink_after_archive".
    pending_record = {
        "assignment_record": str(lane_state / "sessions.assignment.json"),
        "before": ["m-s-peer1111"],
        "config": str(lane_state / "sessions.json"),
        "notebook_hash": notebook_hash,
        "output": str(out_dir),
        "session": session,
    }

    proof_payload = {
        "classification": "ALLOCATION_UNAVAILABLE",
        "endpoint_ever_assigned": False,
        "membership": {
            "added": {},
            "removed": {},
            "retained": {
                "m-s-peer1111": {
                    "active": str(peer_active_rec),
                    "active_sha256": digest(peer_active_rec),
                    "executed_manifest": str(peer_exec_mf),
                    "executed_manifest_sha256": digest(peer_exec_mf),
                    "job_id": "peer-active-l4-job",
                    "manifest": str(peer_ops_mf),
                    "manifest_sha256": digest(peer_ops_mf),
                    "output": str(peer_out),
                    "session": "snake-peer-active-l4-job-aaaaaaaaaaaa",
                }
            },
        },
        "mode": "LIVE",
        "pending": {
            "path": str(pending_path),
            "record": pending_record,
            "sha256": proof.canonical_pending_sha256(pending_record),
        },
        "pending_action": "unlink_after_archive",
        "receipt": {
            "path": str(receipt_path),
            "sha256": digest(receipt_path),
        },
        "recovery": {"endpoint": None},
        "requested_accelerator": "L4",
        "schema": proof.PROOF_SCHEMA,
        "sessions_stdout": (
            "[snake-peer-active-l4-job-aaaaaaaaaaaa] m-s-peer1111 | "
            "Hardware: NVIDIA L4 | Shape: High-RAM | Variant: GPU\n"
        ),
        "vm_compute_started": False,
    }
    proof_top = ops_dir / f"unallocated-{ops_name}-proof-01.json"
    proof_disp = ops_dir / "dispatch" / ops_name / "unallocated-proof-01.json"
    proof_bytes = json.dumps(proof_payload, indent=2, sort_keys=True) + "\n"
    proof_top.write_text(proof_bytes, encoding="utf-8")
    proof_disp.write_text(proof_bytes, encoding="utf-8")
    return receipt_path, proof_top, proof_disp, pending_path, lane_state


def test_baseline_closure_identity_and_adapter_inventory_isolation(tmp_path: Path) -> None:
    assert compute_broker_identity(V3_BROKER_ROOT) == BASELINE_SCIENCE
    assert p.NAMESPACE == BASELINE_NAMESPACE
    assert digest(V3_BROKER_ROOT / "research/watch6_search_efficacy_v3/REGISTRATION_V3.md") == (
        REGISTRATION_V3_SHA256
    )
    assert not hasattr(p, "ACCEPTED_PRIOR_SCIENCES")
    assert all("watch6_v3_accounting" not in entry for entry in p.source_paths())

    closure = verify_baseline_closure()
    assert closure == {
        "broker_root": str(V3_BROKER_ROOT.resolve()),
        "baseline_commit": BASELINE_COMMIT,
        "science": BASELINE_SCIENCE,
        "namespace": BASELINE_NAMESPACE,
        "registration_sha256": REGISTRATION_V3_SHA256,
    }

    f_map = adapter_files_map(ADAPTER_ROOT)
    assert tuple(sorted(f_map.keys())) == tuple(sorted(ADAPTER_FILES))
    assert len(f_map) == 10
    assert "tools/watch6_v3_accounting/analyze.py" not in f_map
    assert len(adapter_inventory_sha256(ADAPTER_ROOT)) == 64

    # Simulated ACCEPTED_PRIOR_SCIENCES bypass or science rotation MUST fail closed:
    with patch.object(p, "ACCEPTED_PRIOR_SCIENCES", frozenset({BASELINE_SCIENCE}), create=True):
        with pytest.raises(ValueError, match="ACCEPTED_PRIOR_SCIENCES"):
            verify_baseline_closure(V3_BROKER_ROOT)

    with patch.object(p, "identity", return_value="0" * 64):
        p.identity.cache_clear = lambda: None  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="rotated"):
            verify_baseline_closure(V3_BROKER_ROOT)

    # Unexpected extra file (e.g. standalone analyze.py) in adapter package MUST be refused:
    fake_repo = tmp_path / "fake-repo"
    (fake_repo / "tools/watch6_v3_accounting").mkdir(parents=True)
    (fake_repo / "tests").mkdir(parents=True)
    for rel in ADAPTER_FILES:
        (fake_repo / rel).write_text("# ok\n", encoding="utf-8")
    assert len(adapter_files_map(fake_repo)) == 10
    (fake_repo / "tools/watch6_v3_accounting/analyze.py").write_text("# forbidden\n")
    with pytest.raises(ValueError, match="unexpected file in adapter package directory"):
        adapter_files_map(fake_repo)


def test_live_shard00_proof_and_receipt_verified_readonly() -> None:
    art = artifacts_root(ADAPTER_ROOT)
    live_proof = art / SHARD00_PROOF_REL
    live_receipt = art / SHARD00_RECEIPT_REL
    if not (live_proof.is_file() and live_receipt.is_file()):
        pytest.skip("live shard-00 artifacts not mounted on this host")

    before_sha = digest(live_receipt)
    before_stat = live_receipt.stat()
    before_proof_sha = digest(live_proof)
    before_proof_stat = live_proof.stat()
    assert before_sha == SHARD00_RECEIPT_SHA256
    assert before_proof_sha == SHARD00_PROOF_SHA256

    ops_dir = art / "ops"
    summary = proof.verify_never_allocated_attempt(live_receipt.parent, ops_dir=ops_dir)
    assert summary["receipt_sha256"] == SHARD00_RECEIPT_SHA256
    assert summary["executed_manifest_sha256"] == SHARD00_EXEC_MANIFEST_SHA256
    assert summary["proof_sha256"] == SHARD00_PROOF_SHA256
    assert len(summary["proof_paths"]) == 2

    rec_obj = json.loads(live_receipt.read_text(encoding="utf-8"))
    assert proof.is_proven_never_allocated_attempt(rec_obj, live_receipt.parent, ops_dir=ops_dir)
    assert admission.is_prestart_transport_failure(rec_obj, live_receipt.parent, ops_dir=ops_dir)
    _, verified_rec = resume.verify_bound_receipt(live_receipt, ops_dir=ops_dir)
    assert verified_rec["stop_verified"] is False
    assert verified_rec["cleanup_error"] == RETAINED_PENDING_CLEANUP_ERROR
    assert (
        dispatch._attempt_state(
            ops_dir,
            "watch6-v3-fix3-shard-00",
            ops_dir / "manifests/watch6-v3-fix3-shard-00.json",
            {},
            live_receipt.parent,
            ops_dir / "logs/watch6-v3-fix3-shard-00.log",
        )
        == "DONE"
    )

    # Verify receipt.json and unallocated proof were NEVER rewritten or touched:
    after_stat = live_receipt.stat()
    after_proof_stat = live_proof.stat()
    assert digest(live_receipt) == before_sha
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_size == before_stat.st_size
    assert digest(live_proof) == before_proof_sha
    assert after_proof_stat.st_mtime_ns == before_proof_stat.st_mtime_ns
    assert after_proof_stat.st_size == before_proof_stat.st_size


@pytest.mark.parametrize(
    ("with_banner", "refusal_kind"),
    [(True, "precondition"), (False, "precondition"), (True, "503"), (True, "503_padded")],
)
def test_never_allocated_proof_positive_variants(
    tmp_path: Path,
    with_banner: bool,
    refusal_kind: str,
) -> None:
    ops_dir = tmp_path / "ops"
    r_path, p_top, p_disp, _, _ = _write_never_allocated_shard_fixture(
        tmp_path,
        ops_dir=ops_dir,
        with_update_banner=with_banner,
        refusal_kind=refusal_kind,
    )
    # Optional explicit auth_ok=True, sessions_ok=True and consistent pre-alloc state/events pass:
    if refusal_kind == "503_padded":
        obj = json.loads(p_top.read_text(encoding="utf-8"))
        obj["auth_ok"] = True
        obj["sessions_ok"] = True
        raw = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        p_top.write_text(raw, encoding="utf-8")
        p_disp.write_text(raw, encoding="utf-8")
        (r_path.parent / "state.json").write_text(
            json.dumps({"status": "REFUSED", "endpoint": None, "polls": 0}) + "\n",
            encoding="utf-8",
        )
        (r_path.parent / "events.jsonl").write_text(
            json.dumps({"event": "new", "status": "failed", "returncode": 1, "endpoint": None})
            + "\n",
            encoding="utf-8",
        )
    rec = json.loads(r_path.read_text(encoding="utf-8"))
    assert proof.is_proven_never_allocated_attempt(rec, r_path.parent, ops_dir=ops_dir) is True
    assert admission.is_prestart_transport_failure(rec, r_path.parent, ops_dir=ops_dir) is True
    _, bound_rec = resume.verify_bound_receipt(r_path, ops_dir=ops_dir)
    assert bound_rec["stop_verified"] is False
    assert bound_rec["cleanup_error"] == RETAINED_PENDING_CLEANUP_ERROR
    assert dispatch.reconcile_staging(ops_dir)["status"] == "RECONCILED"


def _assert_all_callers_refuse_never_allocated(ops_dir: Path, r_path: Path) -> None:
    rec = json.loads(r_path.read_text(encoding="utf-8"))
    with isolated_broker_execution("assert-never-allocated-refused"):
        with pytest.raises(ValueError):
            proof.verify_never_allocated_attempt(r_path.parent, receipt=rec, ops_dir=ops_dir)
        assert proof.is_proven_never_allocated_attempt(rec, r_path.parent, ops_dir=ops_dir) is False
        assert admission.is_prestart_transport_failure(rec, r_path.parent, ops_dir=ops_dir) is False
        with pytest.raises(ValueError):
            resume.verify_bound_receipt(r_path, ops_dir=ops_dir)
        with pytest.raises(RuntimeError, match="REFUSE"):
            dispatch.reconcile_staging(ops_dir)
        (ops_dir / dispatch.RECONCILE_MARKER_NAME).unlink(missing_ok=True)
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE"):
            final_look._verify_attempt_stop_and_hashes(
                r_path.parent,
                ops_dir=ops_dir,
                ops_name="watch6-v3-fix3-shard-00",
            )


def _assert_live_shard00_virtual_proof_mutation_refused(mutate_fn) -> None:
    """Apply an in-memory mutation to the real shard-00 proof without touching any file on disk."""
    art = artifacts_root(ADAPTER_ROOT)
    live_proof = art / SHARD00_PROOF_REL
    live_receipt = art / SHARD00_RECEIPT_REL
    if not (live_proof.is_file() and live_receipt.is_file()):
        return
    ops_dir = art / "ops"
    before_rec_sha = digest(live_receipt)
    before_proof_sha = digest(live_proof)
    orig_reader = proof._read_regular_json_dict

    def patched_reader(path: Path, label: str) -> dict:
        data = json.loads(json.dumps(orig_reader(path, label)))
        if label == "unallocated proof":
            mutate_fn(data)
        return data

    rec = json.loads(live_receipt.read_text(encoding="utf-8"))
    with patch.object(proof, "_read_regular_json_dict", side_effect=patched_reader):
        with pytest.raises(ValueError):
            proof.verify_never_allocated_attempt(live_receipt.parent, ops_dir=ops_dir)
        assert (
            proof.is_proven_never_allocated_attempt(rec, live_receipt.parent, ops_dir=ops_dir)
            is False
        )
        assert (
            admission.is_prestart_transport_failure(rec, live_receipt.parent, ops_dir=ops_dir)
            is False
        )
        with pytest.raises(ValueError):
            resume.verify_bound_receipt(live_receipt, ops_dir=ops_dir)
    assert digest(live_receipt) == before_rec_sha
    assert digest(live_proof) == before_proof_sha


def _assert_live_shard00_virtual_file_refused(filename: str, content: str) -> None:
    """Inject a virtual file into real shard-00 output_dir in memory without touching disk."""
    art = artifacts_root(ADAPTER_ROOT)
    live_proof = art / SHARD00_PROOF_REL
    live_receipt = art / SHARD00_RECEIPT_REL
    if not (live_proof.is_file() and live_receipt.is_file()):
        return
    ops_dir = art / "ops"
    before_rec_sha = digest(live_receipt)
    before_proof_sha = digest(live_proof)
    target = live_receipt.parent / filename
    orig_exists = Path.exists
    orig_is_file = Path.is_file
    orig_read_text = Path.read_text

    rec = json.loads(live_receipt.read_text(encoding="utf-8"))
    with (
        patch.object(Path, "exists", lambda self: True if self == target else orig_exists(self)),
        patch.object(Path, "is_file", lambda self: True if self == target else orig_is_file(self)),
        patch.object(
            Path,
            "read_text",
            lambda self, *a, **kw: content if self == target else orig_read_text(self, *a, **kw),
        ),
    ):
        with pytest.raises(ValueError):
            proof.verify_never_allocated_attempt(live_receipt.parent, ops_dir=ops_dir)
        assert (
            proof.is_proven_never_allocated_attempt(rec, live_receipt.parent, ops_dir=ops_dir)
            is False
        )
        assert (
            admission.is_prestart_transport_failure(rec, live_receipt.parent, ops_dir=ops_dir)
            is False
        )
        with pytest.raises(ValueError):
            resume.verify_bound_receipt(live_receipt, ops_dir=ops_dir)
    assert digest(live_receipt) == before_rec_sha
    assert digest(live_proof) == before_proof_sha


@isolated_broker_execution("test-probe-1")
def test_astra_probe_1_refuses_false_auth_and_sessions_flags(tmp_path: Path) -> None:
    """Astra Probe 1: LIVE proof with auth_ok/sessions_ok not exactly boolean True fails closed."""
    _assert_live_shard00_virtual_proof_mutation_refused(
        lambda d: d.update(auth_ok=False, sessions_ok=False)
    )
    for idx, bad_flags in enumerate(
        (
            {"auth_ok": False, "sessions_ok": False},
            {"auth_ok": False, "sessions_ok": True},
            {"auth_ok": True, "sessions_ok": False},
            {"auth_ok": False},
            {"sessions_ok": False},
            {"auth_ok": True},
            {"sessions_ok": True},
            {"auth_ok": 1, "sessions_ok": True},
            {"auth_ok": True, "sessions_ok": "true"},
            {"auth_ok": None, "sessions_ok": None},
        )
    ):
        case_dir = tmp_path / f"probe1-{idx}"
        ops_dir = case_dir / "ops"
        r_path, p_top, p_disp, _, _ = _write_never_allocated_shard_fixture(
            case_dir, ops_dir=ops_dir
        )
        obj = json.loads(p_top.read_text(encoding="utf-8"))
        obj.update(bad_flags)
        raw = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        p_top.write_text(raw, encoding="utf-8")
        p_disp.write_text(raw, encoding="utf-8")
        _assert_all_callers_refuse_never_allocated(ops_dir, r_path)


def test_astra_probe_2_refuses_zero_or_mismatched_pending_sha256(tmp_path: Path) -> None:
    """Astra Probe 2: pending.sha256 not equal to actual SHA-256 of archived pending payload."""
    _assert_live_shard00_virtual_proof_mutation_refused(
        lambda d: d["pending"].update(sha256="0" * 64)
    )
    for idx, bad_sha in enumerate(("0" * 64, "f" * 64)):
        case_dir = tmp_path / f"probe2-{idx}"
        ops_dir = case_dir / "ops"
        r_path, p_top, p_disp, _, _ = _write_never_allocated_shard_fixture(
            case_dir, ops_dir=ops_dir
        )
        obj = json.loads(p_top.read_text(encoding="utf-8"))
        obj["pending"]["sha256"] = bad_sha
        raw = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        p_top.write_text(raw, encoding="utf-8")
        p_disp.write_text(raw, encoding="utf-8")
        _assert_all_callers_refuse_never_allocated(ops_dir, r_path)


def test_astra_probe_3_refuses_foreign_or_unbound_notebook_hash(tmp_path: Path) -> None:
    """Astra Probe 3: foreign/non-UUIDv4/unbound notebook_hash/nbh fails closed even if rehashed."""
    _assert_live_shard00_virtual_proof_mutation_refused(
        lambda d: d["pending"]["record"].update(notebook_hash="0" * 32)
    )

    def _mutate_live_foreign_uuid4(d: dict) -> None:
        d["pending"]["record"]["notebook_hash"] = "3f1aebc3989f44b39163a8838dc302bb"
        d["pending"]["sha256"] = proof.canonical_pending_sha256(d["pending"]["record"])

    def _mutate_live_fresh_uuid4(d: dict) -> None:
        d["pending"]["record"]["notebook_hash"] = "ffffffffffff4fff8fffffffffffffff"
        d["pending"]["sha256"] = proof.canonical_pending_sha256(d["pending"]["record"])

    _assert_live_shard00_virtual_proof_mutation_refused(_mutate_live_foreign_uuid4)
    _assert_live_shard00_virtual_proof_mutation_refused(_mutate_live_fresh_uuid4)

    # 3a. Exact Astra Probe 3 ('0' * 32) on synthetic fixture (both raw and with recomputed sha256):
    for rehash in (False, True):
        case_dir = tmp_path / f"probe3-zeros-{rehash}"
        ops_dir = case_dir / "ops"
        r_path, p_top, p_disp, _, _ = _write_never_allocated_shard_fixture(
            case_dir, ops_dir=ops_dir
        )
        obj = json.loads(p_top.read_text(encoding="utf-8"))
        obj["pending"]["record"]["notebook_hash"] = "0" * 32
        if rehash:
            obj["pending"]["sha256"] = proof.canonical_pending_sha256(obj["pending"]["record"])
        raw = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        p_top.write_text(raw, encoding="utf-8")
        p_disp.write_text(raw, encoding="utf-8")
        _assert_all_callers_refuse_never_allocated(ops_dir, r_path)

    # 3b. Valid UUIDv4 notebook_hash with recomputed sha256 that disagrees with 503 URL nbh:
    case_503 = tmp_path / "probe3-503-nbh-mismatch"
    ops_503 = case_503 / "ops"
    r_503, p_top_503, p_disp_503, _, _ = _write_never_allocated_shard_fixture(
        case_503, ops_dir=ops_503, refusal_kind="503"
    )
    obj_503 = json.loads(p_top_503.read_text(encoding="utf-8"))
    obj_503["pending"]["record"]["notebook_hash"] = "9b2f5a9164224b838cd96581c0f8bd29"
    obj_503["pending"]["sha256"] = proof.canonical_pending_sha256(obj_503["pending"]["record"])
    raw_503 = json.dumps(obj_503, indent=2, sort_keys=True) + "\n"
    p_top_503.write_text(raw_503, encoding="utf-8")
    p_disp_503.write_text(raw_503, encoding="utf-8")
    _assert_all_callers_refuse_never_allocated(ops_503, r_503)

    # 3c. Replayed UUIDv4 notebook_hash belonging to a foreign assignment or unallocated proof:
    case_replay = tmp_path / "probe3-replayed-nbh"
    ops_replay = case_replay / "ops"
    r_rep, p_top_rep, p_disp_rep, _, _ = _write_never_allocated_shard_fixture(
        case_replay, ops_dir=ops_replay
    )
    foreign_nbh = "7c8e5a9164224b838cd96581c0f8bd29"
    foreign_lane = ops_replay / "state/peer-active-l4-job/peer-active-l4-job"
    foreign_lane.mkdir(parents=True, exist_ok=True)
    (foreign_lane / "sessions.assignment.json").write_text(
        json.dumps({"notebook_hash": foreign_nbh, "endpoint": "m-s-peer1111"}) + "\n",
        encoding="utf-8",
    )
    obj_rep = json.loads(p_top_rep.read_text(encoding="utf-8"))
    obj_rep["pending"]["record"]["notebook_hash"] = foreign_nbh
    obj_rep["pending"]["sha256"] = proof.canonical_pending_sha256(obj_rep["pending"]["record"])
    raw_rep = json.dumps(obj_rep, indent=2, sort_keys=True) + "\n"
    p_top_rep.write_text(raw_rep, encoding="utf-8")
    p_disp_rep.write_text(raw_rep, encoding="utf-8")
    _assert_all_callers_refuse_never_allocated(ops_replay, r_rep)

    # 3d. Pending config path pointing to a foreign job lane (even with recomputed pending.sha256):
    case_lane = tmp_path / "probe3-foreign-lane"
    ops_lane = case_lane / "ops"
    r_ln, p_top_ln, p_disp_ln, _, _ = _write_never_allocated_shard_fixture(
        case_lane, ops_dir=ops_lane
    )
    bad_lane = ops_lane / "state/foreign-lane/foreign-job"
    bad_lane.mkdir(parents=True, exist_ok=True)
    obj_ln = json.loads(p_top_ln.read_text(encoding="utf-8"))
    obj_ln["pending"]["record"]["config"] = str(bad_lane / "sessions.json")
    obj_ln["pending"]["record"]["assignment_record"] = str(bad_lane / "sessions.assignment.json")
    obj_ln["pending"]["sha256"] = proof.canonical_pending_sha256(obj_ln["pending"]["record"])
    raw_ln = json.dumps(obj_ln, indent=2, sort_keys=True) + "\n"
    p_top_ln.write_text(raw_ln, encoding="utf-8")
    p_disp_ln.write_text(raw_ln, encoding="utf-8")
    _assert_all_callers_refuse_never_allocated(ops_lane, r_ln)


@isolated_broker_execution("test-probe-4")
def test_astra_probe_4_refuses_runtime_measurement_and_supervisor_postalloc_artifacts(
    tmp_path: Path,
) -> None:
    """Astra Probe 4: runtime-measurement.json + every supervisor post-allocation artifact."""
    _assert_live_shard00_virtual_file_refused("runtime-measurement.json", "{}")
    for idx, artifact_name in enumerate(
        (
            "runtime-measurement.json",
            "backup.json",
            "session-release.json",
            "ready.json",
            "rate-calibration.json",
            "rate-calibration-000.json",
            "source-delivery.json",
            "input-hashes.json",
            "usage-allocated.json",
            "usage-000000.json",
            "recovery-179348-99.json",
            "spec.json",
            "lease.json",
            "job.log",
            "supervisor.log",
        )
    ):
        case_dir = tmp_path / f"probe4-{idx}"
        ops_dir = case_dir / "ops"
        r_path, _, _, _, _ = _write_never_allocated_shard_fixture(case_dir, ops_dir=ops_dir)
        (r_path.parent / artifact_name).write_text("{}\n", encoding="utf-8")
        _assert_all_callers_refuse_never_allocated(ops_dir, r_path)


def test_astra_probe_5_refuses_scientific_fault_artifact(tmp_path: Path) -> None:
    """Astra Probe 5: scientific-fault.json fails closed across never-allocated & retry paths."""
    fault_payload = '{"error":"ValueError: scientific failure"}'
    _assert_live_shard00_virtual_file_refused("scientific-fault.json", fault_payload)

    case_dir = tmp_path / "probe5"
    ops_dir = case_dir / "ops"
    r_path, _, _, _, _ = _write_never_allocated_shard_fixture(case_dir, ops_dir=ops_dir)
    (r_path.parent / "scientific-fault.json").write_text(fault_payload + "\n", encoding="utf-8")
    _assert_all_callers_refuse_never_allocated(ops_dir, r_path)

    # Also verify allocated prestart/compute transport classifiers refuse scientific-fault.json:
    comp_rec_path = _write_v3_attempt(
        tmp_path / "probe5-compute", shard=1, mode="compute_transport"
    )
    (comp_rec_path.parent / "scientific-fault.json").write_text(
        fault_payload + "\n", encoding="utf-8"
    )
    comp_rec = json.loads(comp_rec_path.read_text(encoding="utf-8"))
    assert admission.is_external_transport_compute_failure(comp_rec, comp_rec_path.parent) is False


@isolated_broker_execution("test-probe-6")
def test_astra_probe_6_refuses_allocated_or_contradictory_state_json(tmp_path: Path) -> None:
    """Astra Probe 6: contradictory state.json (allocated endpoint, polls>0, assigned status)."""
    _assert_live_shard00_virtual_file_refused(
        "state.json", '{"endpoint":"allocated-but-unstopped","polls":1}'
    )
    for idx, state_payload in enumerate(
        (
            {"endpoint": "allocated-but-unstopped", "polls": 1},
            {"status": "allocated", "endpoint": None},
            {"status": "assigned", "endpoint": None},
            {"status": "running", "pid": 4242},
            {"status": "REFUSED", "allocated": True},
            {"status": "REFUSED", "assigned": True},
            {"status": "REFUSED", "polls": 1},
            {"status": "REFUSED", "uploaded": 1},
        )
    ):
        case_dir = tmp_path / f"probe6-{idx}"
        ops_dir = case_dir / "ops"
        r_path, _, _, _, _ = _write_never_allocated_shard_fixture(case_dir, ops_dir=ops_dir)
        (r_path.parent / "state.json").write_text(
            json.dumps(state_payload) + "\n", encoding="utf-8"
        )
        _assert_all_callers_refuse_never_allocated(ops_dir, r_path)


@isolated_broker_execution("test-probe-7")
def test_astra_probe_7_refuses_exec_or_command_event_in_events_jsonl(tmp_path: Path) -> None:
    """Astra Probe 7: contradictory events.jsonl (exec/command/post-new/allocated event)."""
    _assert_live_shard00_virtual_file_refused(
        "events.jsonl", '{"endpoint":"allocated-but-unstopped","event":"exec"}\n'
    )
    for idx, event_payload in enumerate(
        (
            {"endpoint": "allocated-but-unstopped", "event": "exec"},
            {"event": "exec"},
            {"command": "exec"},
            {"event": "command"},
            {"operation": "exec"},
            {"event": "allocated"},
            {"event": "assigned"},
            {"event": "upload"},
            {"event": "stop"},
            {"event": "new", "status": "completed", "returncode": 0},
        )
    ):
        case_dir = tmp_path / f"probe7-{idx}"
        ops_dir = case_dir / "ops"
        r_path, _, _, _, _ = _write_never_allocated_shard_fixture(case_dir, ops_dir=ops_dir)
        (r_path.parent / "events.jsonl").write_text(
            json.dumps(event_payload) + "\n", encoding="utf-8"
        )
        _assert_all_callers_refuse_never_allocated(ops_dir, r_path)


@isolated_broker_execution("test-never-allocated-neg")
def test_never_allocated_proof_negative_and_tamper_vectors(tmp_path: Path) -> None:
    def make_case(name: str) -> tuple[Path, Path, Path, Path, Path, Path]:
        case_dir = tmp_path / name
        ops_dir = case_dir / "ops"
        r_path, p_top, p_disp, pend, lane = _write_never_allocated_shard_fixture(
            case_dir, ops_dir=ops_dir
        )
        return ops_dir, r_path, p_top, p_disp, pend, lane

    def assert_refused(ops_dir: Path, r_path: Path) -> None:
        _assert_all_callers_refuse_never_allocated(ops_dir, r_path)

    # 1. Missing LIVE proof files:
    ops, r_p, p_top, p_disp, _, _ = make_case("missing-proof")
    p_top.unlink()
    p_disp.unlink()
    assert_refused(ops, r_p)

    # 2. Divergent / non-identical proof copies:
    ops, r_p, p_top, p_disp, _, _ = make_case("divergent-proof")
    tampered = json.loads(p_disp.read_text(encoding="utf-8"))
    tampered["sessions_stdout"] += "\n"
    p_disp.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    assert_refused(ops, r_p)

    # 3-7. Proof contract violations (DRY_RUN, endpoint_ever_assigned, vm_compute_started,
    #      non-null recovery.endpoint, receipt sha256 mismatch):
    for label, patch_dict in (
        ("dry-run", {"mode": "DRY_RUN"}),
        ("assigned", {"endpoint_ever_assigned": True}),
        ("compute-started", {"vm_compute_started": True}),
        ("recovery-ep", {"recovery": {"endpoint": "m-s-leaked"}}),
        ("bad-rec-sha", {"receipt": {"path": "", "sha256": "0" * 64}}),
    ):
        ops, r_p, p_top, p_disp, _, _ = make_case(f"proof-{label}")
        obj = json.loads(p_top.read_text(encoding="utf-8"))
        if label == "bad-rec-sha":
            obj["receipt"]["sha256"] = "0" * 64
        else:
            obj.update(patch_dict)
        raw = json.dumps(obj, sort_keys=True) + "\n"
        p_top.write_text(raw, encoding="utf-8")
        p_disp.write_text(raw, encoding="utf-8")
        assert_refused(ops, r_p)

    # 8. Surviving pending record on disk:
    ops, r_p, _, _, pend, _ = make_case("surviving-pending")
    pend.write_text("{}\n", encoding="utf-8")
    assert_refused(ops, r_p)

    # 9. Surviving active session record on disk:
    ops, r_p, _, _, pend, _ = make_case("surviving-active")
    session = json.loads(r_p.read_text(encoding="utf-8"))["session"]
    (pend.parent / f"active-{session}.json").write_text("{}\n", encoding="utf-8")
    assert_refused(ops, r_p)

    # 10. Lane assignment record on disk:
    ops, r_p, _, _, _, lane = make_case("surviving-lane-assignment")
    (lane / "sessions.assignment.json").write_text("{}\n", encoding="utf-8")
    assert_refused(ops, r_p)

    # 11. Refused session present in sessions_stdout:
    ops, r_p, p_top, p_disp, _, _ = make_case("session-in-stdout")
    session = json.loads(r_p.read_text(encoding="utf-8"))["session"]
    obj = json.loads(p_top.read_text(encoding="utf-8"))
    obj[
        "sessions_stdout"
    ] += f"[{session}] m-s-ghost | Hardware: NVIDIA L4 | Shape: High-RAM | Variant: GPU\n"
    raw = json.dumps(obj, sort_keys=True) + "\n"
    p_top.write_text(raw, encoding="utf-8")
    p_disp.write_text(raw, encoding="utf-8")
    assert_refused(ops, r_p)

    # 12. Unaccounted endpoint in sessions_stdout / tampered membership manifest hash:
    ops, r_p, p_top, p_disp, _, _ = make_case("unaccounted-session")
    obj = json.loads(p_top.read_text(encoding="utf-8"))
    obj["sessions_stdout"] += (
        "[snake-mystery-111111111111] m-s-mystery | "
        "Hardware: NVIDIA L4 | Shape: High-RAM | Variant: GPU\n"
    )
    raw = json.dumps(obj, sort_keys=True) + "\n"
    p_top.write_text(raw, encoding="utf-8")
    p_disp.write_text(raw, encoding="utf-8")
    assert_refused(ops, r_p)

    ops, r_p, p_top, p_disp, _, _ = make_case("tampered-peer-manifest")
    (ops / "manifests/peer-active-l4-job.json").write_text('{"tampered": true}\n', encoding="utf-8")
    assert_refused(ops, r_p)

    # 13-16. Receipt tampering (allocated endpoint, transfers, scientific error, cli_calls):
    for label, rec_mut in (
        ("nonnull-endpoint", lambda r: r.update(endpoint="m-s-assigned")),
        ("nonnull-alloc-ep", lambda r: r["accounting"].update(allocated_endpoint="m-s-assigned")),
        ("has-transfer", lambda r: r.update(transfers=[{"dest": "x"}])),
        ("scientific-err", lambda r: r.update(error="Refused('ValueError: mirror mismatch')")),
        (
            "exec-cli-call",
            lambda r: r["cli_calls"].append(
                {"attempt": 1, "command": "exec", "operation": "exec", "status": "failed"}
            ),
        ),
    ):
        ops, r_p, p_top, p_disp, _, _ = make_case(f"rec-{label}")
        rec = json.loads(r_p.read_text(encoding="utf-8"))
        rec_mut(rec)
        r_p.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
        obj = json.loads(p_top.read_text(encoding="utf-8"))
        obj["receipt"]["sha256"] = digest(r_p)
        raw = json.dumps(obj, sort_keys=True) + "\n"
        p_top.write_text(raw, encoding="utf-8")
        p_disp.write_text(raw, encoding="utf-8")
        assert_refused(ops, r_p)

    # 17. Post-allocation artifact present in output_dir:
    for forbidden_name in (
        "hardware.json",
        "poll-000000.json",
        "upload-measurement.json",
        "warmup-measurement.json",
        "terminal.json",
    ):
        ops, r_p, _, _, _, _ = make_case(f"artifact-{forbidden_name}")
        (r_p.parent / forbidden_name).write_text("{}\n", encoding="utf-8")
        assert_refused(ops, r_p)

    # 18. Allocated attempt lacking verified stop is ALWAYS refused:
    alloc_unstopped = _write_v3_attempt(
        tmp_path / "allocated-unstopped",
        shard=1,
        mode="compute_transport",
        stop_verified=False,
        cleanup_error=RETAINED_PENDING_CLEANUP_ERROR,
    )
    rec_alloc = json.loads(alloc_unstopped.read_text(encoding="utf-8"))
    assert admission.is_prestart_transport_failure(rec_alloc, alloc_unstopped.parent) is False
    assert (
        admission.is_external_transport_compute_failure(rec_alloc, alloc_unstopped.parent) is False
    )
    with pytest.raises(ValueError, match="runtime stop not verified"):
        resume.verify_bound_receipt(alloc_unstopped)


def test_ratify_refuses_approve_path_and_tampered_amendment_and_accepts_approve(
    tmp_path: Path,
) -> None:
    commit_sha = adapter_commit(ADAPTER_ROOT, require_clean=False)
    amend_path = _write_immutable_text(
        tmp_path / "WATCH6_V3_ACCOUNTING_AMENDMENT_20261031.md",
        _render_amendment_markdown(ADAPTER_ROOT, commit_sha),
    )

    # 1. APPROVE-PATH (including live watch6-v3-shard00-ruling/RULING.md!) MUST be refused!
    approve_path_md = tmp_path / "RULING_APPROVE_PATH.md"
    approve_path_md.write_text(
        "Verdict: APPROVE-PATH\n\nAstra · 2026-10-31 · binding PRE-LOOK path.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="REFUSE"):
        ratify.record_ruling(
            approve_path_md,
            amendment_md=amend_path,
            out=tmp_path / "refused-path.json",
            require_clean_git=False,
        )
    assert not (tmp_path / "refused-path.json").exists()

    live_path_ruling = artifacts_root(ADAPTER_ROOT) / SHARD00_PATH_RULING_REL
    if live_path_ruling.is_file():
        with pytest.raises(ValueError, match="REFUSE"):
            ratify.record_ruling(
                live_path_ruling,
                amendment_md=amend_path,
                out=tmp_path / "refused-live-path.json",
                require_clean_git=False,
            )

    # 2. Old watch6-v3-pre-data-ruling/v2 receipt is refused by accounting adapter:
    old_ruling_md = tmp_path / "OLD_V3.md"
    old_ruling_md.write_text("Verdict: APPROVE\nSTATUS: RATIFIED\n", encoding="utf-8")
    old_json = tmp_path / "old-v3-ruling.json"
    with bind_broker_protocol(V3_BROKER_ROOT):
        v3_ratify.record_ruling(old_ruling_md, old_json)
    with pytest.raises(ValueError, match="REFUSE"):
        ratify.validate_ruling_receipt(old_json, require_clean_git=False)

    # 3. Amendment inside hashed V3 source closure or missing adapter file hash is refused:
    with pytest.raises(ValueError, match="inside hashed V3 source closure"):
        ratify._assert_outside_source_closure(
            ADAPTER_ROOT / "research/watch6_search_efficacy_v3/AMENDMENT.md",
            ADAPTER_ROOT,
        )

    incomplete_amend = _write_immutable_text(
        tmp_path / "INCOMPLETE_AMENDMENT.md",
        _render_amendment_markdown(ADAPTER_ROOT, commit_sha).replace(
            "tools/watch6_v3_accounting/proof.py", "OMITTED_FILE"
        ),
    )
    with pytest.raises(ValueError, match="missing exact hash binding"):
        ratify.verify_amendment_document(incomplete_amend, require_clean_git=False)

    # 4. Explicit APPROVE and APPROVE-WITH-CONDITIONS succeed and detect post-ratification tamper:
    for idx, verdict in enumerate(("APPROVE", "APPROVE-WITH-CONDITIONS")):
        _, ruling_json = _write_accounting_ruling_fixture(tmp_path / f"ok-{idx}", verdict=verdict)
        validated = ratify.validate_ruling_receipt(ruling_json, require_clean_git=False)
        assert validated["verdict"] == verdict
        assert validated["adapter_inventory_sha256"] == adapter_inventory_sha256(ADAPTER_ROOT)

    # Post-ratification adapter file hash drift is caught fail-closed:
    _, drift_json = _write_accounting_ruling_fixture(tmp_path / "drift")
    fake_map = {**adapter_files_map(ADAPTER_ROOT), "tools/watch6_v3_accounting/proof.py": "f" * 64}
    with patch.object(ratify, "adapter_files_map", return_value=fake_map):
        with pytest.raises(ValueError, match="REFUSE"):
            ratify.validate_ruling_receipt(drift_json, require_clean_git=False)


def _setup_24_shard_accounting_ops(
    tmp_path: Path,
    *,
    shard0_extra_compute_failures: int = 0,
    sealed_placeholders: bool = False,
) -> tuple[Path, Path, Path, dict[int, dict]]:
    """Build a complete 24-shard V3 ops registry where shard 00 has a never-allocated A_1."""
    ops_dir = tmp_path / "ops"
    (ops_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (ops_dir / "logs").mkdir(parents=True, exist_ok=True)
    amend_path, ruling_json = _write_accounting_ruling_fixture(tmp_path / "governance")

    # Passing L4 canary:
    canary_out = tmp_path / "output-canary"
    canary_out.mkdir(parents=True)
    broker_science = BASELINE_SCIENCE
    canary_mf = {
        "job_id": "watch6-v3-canary",
        "output_dir": str(canary_out),
        "commit": BASELINE_COMMIT,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 15},
        "canary": {"schema": canary.SCHEMA, "mode": "probe", "family": p.SCRIPT},
        "env": {"WATCH6_FIXTURE": "1"},
        "watch6_v3": {
            "namespace": p.NAMESPACE,
            "science": broker_science,
            "primary_allocation": p.allocation(),
        },
    }
    (canary_out / "manifest.json").write_text(json.dumps(canary_mf, sort_keys=True) + "\n")
    (canary_out / "receipt.json").write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-canary",
                "commit": BASELINE_COMMIT,
                "manifest_sha256": digest(canary_out / "manifest.json"),
                "endpoint": "m-s-canary-l4",
                "status": "PASS",
                "failure_class": None,
                "error": None,
                "stop_verified": True,
                "cleanup_error": None,
                "terminal": {"returncode": 0, "reason": None, "error": None},
                "exports": [],
            }
        )
        + "\n"
    )
    (ops_dir / "manifests/watch6-v3-canary.json").write_text(json.dumps(canary_mf) + "\n")

    all_rows = {} if sealed_placeholders else _cached_synthetic_v3_rows_by_slot()
    rows_arg = None if sealed_placeholders else all_rows

    # Shard 00: A_1 is PROVEN never-allocated (`cleanup_error` retained, `stop_verified=False`),
    # optionally followed by `shard0_extra_compute_failures` stopped compute transport failures,
    # followed by passing terminal completion on L4 W4!
    s0_a1, _, _, _, _ = _write_never_allocated_shard_fixture(
        tmp_path / "shard00-a1",
        ops_dir=ops_dir,
        ops_name="watch6-v3-fix3-shard-00",
        shard=0,
    )
    priors = [
        {"receipt_path": str(s0_a1), "receipt_sha256": digest(s0_a1), "classification": "prestart"}
    ]
    for c_idx in range(shard0_extra_compute_failures):
        c_rec = _write_v3_attempt(
            tmp_path / f"out-shard-00-compute-{c_idx + 1}",
            shard=0,
            mode="compute_transport",
            export_slots=[0] if c_idx == 0 else [],
            pair_rows_by_slot=rows_arg,
            prestart_retries_used=1,
            compute_completion_attempt=c_idx,
            prior_receipts=priors,
        )
        (ops_dir / f"manifests/watch6-v3-shard-00-compute-{c_idx + 1}.json").write_text(
            (c_rec.parent / "manifest.json").read_text()
        )
        priors = [
            *priors,
            {
                "receipt_path": str(c_rec),
                "receipt_sha256": digest(c_rec),
                "classification": "external_transport_compute",
            },
        ]

    rem_slots = [24, 48, 72] if shard0_extra_compute_failures > 0 else p.shard_indices(0)
    s0_pass = _write_v3_attempt(
        tmp_path / "out-shard-00-recovered",
        shard=0,
        mode="pass",
        export_slots=rem_slots,
        pair_rows_by_slot=rows_arg,
        prestart_retries_used=1,
        compute_completion_attempt=shard0_extra_compute_failures,
        prior_receipts=priors,
    )
    (ops_dir / "manifests/watch6-v3-fix3-shard-00-recovered.json").write_text(
        (s0_pass.parent / "manifest.json").read_text()
    )

    for s in range(1, p.SHARDS):
        r_path = _write_v3_attempt(
            tmp_path / f"out-shard-{s:02d}",
            shard=s,
            mode="pass",
            export_slots=p.shard_indices(s),
            pair_rows_by_slot=rows_arg,
        )
        (ops_dir / f"manifests/watch6-v3-fix3-shard-{s:02d}.json").write_text(
            (r_path.parent / "manifest.json").read_text()
        )

    return ops_dir, ruling_json, amend_path, all_rows


def test_recover_and_final_look_lineage_retry_caps_and_zero_outcome_peek(tmp_path: Path) -> None:
    # 1. Recover shard 00 from proven never-allocated receipt (Ruling Point 6):
    rec_case = tmp_path / "recover-case"
    ops_rec = rec_case / "ops"
    s0_a1, _, _, _, _ = _write_never_allocated_shard_fixture(rec_case, ops_dir=ops_rec, shard=0)
    before_a1_sha = digest(s0_a1)

    base_pkg = rec_case / "package-v3-fix3-01"
    base_pkg.mkdir()
    with bind_broker_protocol(V3_BROKER_ROOT):
        publish_json(
            base_pkg / "verified-ledger.json",
            empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity()),
        )
        broker_exec_identity = mirror.execution_identity()
        broker_science = p.identity()
    dummy_in = base_pkg / "input-metadata.tar.gz"
    dummy_in.write_bytes(b"metadata")
    (base_pkg / "canary.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "path": str(dummy_in),
                        "dest": "input-metadata.tar.gz",
                        "sha256": digest(dummy_in),
                    }
                ]
            }
        )
    )
    (base_pkg / "output-canary").mkdir()
    (base_pkg / "output-canary/receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "cleanup_error": None,
                "commit": BASELINE_COMMIT,
            }
        )
    )
    bench = rec_case / "benchmark.json"
    bench.write_text(json.dumps({"science": broker_science}))
    mir = rec_case / "mac-mirror.json"
    mir.write_text(
        json.dumps(
            {
                "schema": "watch6-full-mirror/v3",
                "status": "PASS",
                "platform": "Darwin",
                "execution_identity": broker_exec_identity,
                "seed": p.FIXTURE_SEEDS[0],
                "horizon": p.HORIZON,
                "mirror_identity_frames": p.HORIZON,
                "mirror_mismatches": 0,
                "respawns_per_slot": [3, 3, 3, 3, 3, 3],
            }
        )
    )
    cost_path = rec_case / "costs.json"
    cost_path.write_text(json.dumps(_synthetic_cost_report()))

    captured_prepare_kwargs: dict = {}

    def fake_prepare(out: Path, *args, **kwargs) -> dict:
        captured_prepare_kwargs.update(kwargs)
        out.mkdir(parents=True, exist_ok=True)
        return {"shards": 24, "workers": 4, "base_minutes_per_shard": 117}

    with patch.object(recover, "prepare", side_effect=fake_prepare):
        plan = recover.recover(
            base_package=base_pkg,
            receipts=[s0_a1],
            target_shards=[0],
            out=rec_case / "recovery-shard-00",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit=BASELINE_COMMIT,
            ops_dir=ops_rec,
        )
    assert plan["target_shards"] == [0]
    assert plan["verified_units_in_ledger"] == 0
    assert plan["reused_initial_canary"] is True
    assert plan["shard_meta"]["0"]["prestart_retries_used"] == 1
    assert plan["shard_meta"]["0"]["compute_completion_attempt"] == 0
    assert plan["shard_meta"]["0"]["total_dispatches_for_shard"] == 2
    assert len(plan["never_allocated_proofs"]) == 1
    assert captured_prepare_kwargs["target_shards"] == [0]
    assert digest(s0_a1) == before_a1_sha

    # Refuses non-357e3504 commit in recover and prepare:
    with pytest.raises(ValueError, match=BASELINE_COMMIT):
        recover.recover(
            base_package=base_pkg,
            receipts=[s0_a1],
            target_shards=[0],
            out=rec_case / "recovery-bad-commit",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit="a" * 40,
            ops_dir=ops_rec,
        )
    with pytest.raises(ValueError, match=BASELINE_COMMIT):
        prepare._require_published_pin("a" * 40, ADAPTER_ROOT)

    # 2. Full 24-shard final_look PASS with retained shard-00 never-allocated A_1 + 2 retries:
    ops_ok, ruling_ok, amend_ok, _ = _setup_24_shard_accounting_ops(
        tmp_path / "full-ok", shard0_extra_compute_failures=2
    )
    s0_retained_rec = tmp_path / "full-ok/shard00-a1/output-shard-00/receipt.json"
    retained_before = digest(s0_retained_rec)
    decision = final_look.run_final_look(
        ops_dir=ops_ok,
        out_dir=tmp_path / "full-ok/look-out",
        ruling_json=ruling_ok,
        amendment_md=amend_ok,
        require_clean_git=False,
    )
    assert decision["verdict"] == "GO_EFFICACY_ONLY"
    assert decision["n"] == 96
    assert decision["total_main_attempts"] == 27  # 23 single-pass shards + 4 attempts on shard 00
    assert len(decision["retained_never_allocated_attempts"]) == 1
    assert digest(s0_retained_rec) == retained_before
    assert json.loads(s0_retained_rec.read_text())["stop_verified"] is False

    # 3. Exceeded compute retry cap (3 compute failures after never-allocated prestart)
    #    MUST raise INVALID_OR_INCOMPLETE with ZERO calls to decide_v3!
    ops_cap, ruling_cap, amend_cap, rows_cap = _setup_24_shard_accounting_ops(
        tmp_path / "full-cap", shard0_extra_compute_failures=2, sealed_placeholders=True
    )
    # Replace terminal pass with a 3rd failed compute auto-retry + 4th pass auto-retry:
    auto1_dir = tmp_path / "full-cap/auto-retry-1"
    prev_pass_mf = ops_cap / "manifests/watch6-v3-fix3-shard-00-recovered.json"
    prev_c2_rec = (tmp_path / "full-cap/out-shard-00-compute-2/receipt.json").resolve()
    c3_rec = _write_v3_attempt(
        auto1_dir / "output",
        shard=0,
        mode="compute_transport",
        prestart_retries_used=1,
        compute_completion_attempt=1,
        prior_receipts=json.loads(
            (tmp_path / "full-cap/out-shard-00-compute-2/manifest.json").read_text()
        )["watch6_v3"]["prior_receipts"],
        completion_attempt=1,
    )
    publish_json(
        auto1_dir / "manifest.json", json.loads((c3_rec.parent / "manifest.json").read_text())
    )
    publish_json(
        auto1_dir / "RETRY.json",
        {
            "schema": "ops-zero-export-retry/v1",
            "completion_attempt": 1,
            "max_completion_attempts": 2,
            "prior_manifest": {"path": str(prev_pass_mf), "sha256": digest(prev_pass_mf)},
            "prior_receipt": {"path": str(prev_c2_rec), "sha256": digest(prev_c2_rec)},
            "manifest_sha256": digest(auto1_dir / "manifest.json"),
            "unallocated_proof": None,
        },
    )
    (ops_cap / "manifests/watch6-v3-shard-00-auto1.json").write_text(
        (auto1_dir / "manifest.json").read_text()
    )
    auto2_dir = tmp_path / "full-cap/auto-retry-2"
    p4_rec = _write_v3_attempt(
        auto2_dir / "output",
        shard=0,
        mode="pass",
        export_slots=[24, 48, 72],
        pair_rows_by_slot=rows_cap,
        prestart_retries_used=1,
        compute_completion_attempt=1,
        prior_receipts=json.loads((c3_rec.parent / "manifest.json").read_text())["watch6_v3"][
            "prior_receipts"
        ],
        completion_attempt=2,
    )
    publish_json(
        auto2_dir / "manifest.json", json.loads((p4_rec.parent / "manifest.json").read_text())
    )
    publish_json(
        auto2_dir / "RETRY.json",
        {
            "schema": "ops-zero-export-retry/v1",
            "completion_attempt": 2,
            "max_completion_attempts": 2,
            "prior_manifest": {
                "path": str(auto1_dir / "manifest.json"),
                "sha256": digest(auto1_dir / "manifest.json"),
            },
            "prior_receipt": {"path": str(c3_rec.resolve()), "sha256": digest(c3_rec)},
            "manifest_sha256": digest(auto2_dir / "manifest.json"),
            "unallocated_proof": None,
        },
    )
    prev_pass_mf.write_text((auto2_dir / "manifest.json").read_text())

    with patch.object(final_look, "decide_v3", side_effect=AssertionError("OUTCOME PEEKED!")):
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE.*exceeded compute completion"):
            final_look.run_final_look(
                ops_dir=ops_cap,
                out_dir=tmp_path / "full-cap/look-out",
                ruling_json=ruling_cap,
                amendment_md=amend_cap,
                require_clean_git=False,
            )

    # 4. Incomplete slot coverage (95/96) MUST write incomplete-status.json & never call decide_v3:
    ops_inc, ruling_inc, amend_inc, _ = _setup_24_shard_accounting_ops(
        tmp_path / "full-inc", sealed_placeholders=True
    )
    rec_s1_path = tmp_path / "full-inc/out-shard-01/receipt.json"
    rec_s1 = json.loads(rec_s1_path.read_text())
    rec_s1["exports"] = [e for e in rec_s1["exports"] if "pair-73" not in e["path"]]
    rec_s1_path.write_text(json.dumps(rec_s1, sort_keys=True) + "\n")
    for sfx in (
        "",
        ".sha256",
        ".complete",
        ".meta.json",
        ".meta.json.sha256",
        ".meta.json.complete",
    ):
        (tmp_path / f"full-inc/out-shard-01/exports/efficacy/pair-73.json{sfx}").unlink(
            missing_ok=True
        )
    with patch.object(final_look, "decide_v3", side_effect=AssertionError("OUTCOME PEEKED!")):
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE.*95/96"):
            final_look.run_final_look(
                ops_dir=ops_inc,
                out_dir=tmp_path / "full-inc/look-out",
                ruling_json=ruling_inc,
                amendment_md=amend_inc,
                require_clean_git=False,
            )
    assert (tmp_path / "full-inc/look-out/incomplete-status.json").is_file()

    # 5. A100 substitution in ops registry MUST fail closed with INVALID_OR_INCOMPLETE:
    ops_a100, ruling_a100, amend_a100, _ = _setup_24_shard_accounting_ops(
        tmp_path / "full-a100", sealed_placeholders=True
    )
    mf_s2_path = ops_a100 / "manifests/watch6-v3-fix3-shard-02.json"
    mf_s2 = json.loads(mf_s2_path.read_text())
    mf_s2["requirements"]["gpu"] = "A100"
    mf_s2["requirements"]["allowed_gpus"] = ["A100"]
    mf_s2_path.write_text(json.dumps(mf_s2) + "\n")
    with patch.object(final_look, "decide_v3", side_effect=AssertionError("OUTCOME PEEKED!")):
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE.*non-L4 or A100"):
            final_look.run_final_look(
                ops_dir=ops_a100,
                out_dir=tmp_path / "full-a100/look-out",
                ruling_json=ruling_a100,
                amendment_md=amend_a100,
                require_clean_git=False,
            )


def test_baseline_equivalence_with_v3_host_modules(tmp_path: Path) -> None:
    # 1. Classification equivalence across all non-ALLOCATION_UNRESOLVED attempt shapes:
    for idx, mode in enumerate(
        ("pass", "prestart_staging", "prestart_unassigned", "compute_transport", "child_fail")
    ):
        rp = _write_v3_attempt(tmp_path / f"eq-mode-{idx}", shard=2, mode=mode)
        rec = json.loads(rp.read_text(encoding="utf-8"))
        assert admission.is_prestart_transport_failure(
            rec, rp.parent
        ) == v3_admission.is_prestart_transport_failure(rec, rp.parent)
        assert admission.is_external_transport_compute_failure(
            rec, rp.parent
        ) == v3_admission.is_external_transport_compute_failure(rec, rp.parent)

    # 2. Full 24-shard ledger + final_look decision equivalence when zero retained-pending exist:
    ops_base = tmp_path / "base/ops"
    (ops_base / "manifests").mkdir(parents=True)
    (ops_base / "logs").mkdir(parents=True)
    amend_path, acct_ruling_json = _write_accounting_ruling_fixture(tmp_path / "base/acct-gov")
    v3_md = tmp_path / "base/V3_APPROVED.md"
    v3_md.write_text("Verdict: APPROVE\nSTATUS: RATIFIED\n", encoding="utf-8")
    v3_ruling_json = tmp_path / "base/v3-ruling.json"
    with bind_broker_protocol(V3_BROKER_ROOT):
        v3_ratify.record_ruling(v3_md, v3_ruling_json)
        broker_science = p.identity()
        rows = {r["allocation"]["index"]: r for r in _synthetic_v3_rows()}

    canary_out = tmp_path / "base/output-canary"
    canary_out.mkdir(parents=True)
    canary_mf = {
        "job_id": "watch6-v3-canary",
        "output_dir": str(canary_out),
        "commit": BASELINE_COMMIT,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 15},
        "canary": {"schema": canary.SCHEMA, "mode": "probe", "family": p.SCRIPT},
        "env": {"WATCH6_FIXTURE": "1"},
        "watch6_v3": {
            "namespace": p.NAMESPACE,
            "science": broker_science,
            "primary_allocation": p.allocation(),
        },
    }
    (canary_out / "manifest.json").write_text(json.dumps(canary_mf, sort_keys=True) + "\n")
    (canary_out / "receipt.json").write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-canary",
                "commit": BASELINE_COMMIT,
                "manifest_sha256": digest(canary_out / "manifest.json"),
                "endpoint": "m-s-canary",
                "status": "PASS",
                "failure_class": None,
                "error": None,
                "stop_verified": True,
                "cleanup_error": None,
                "terminal": {"returncode": 0, "reason": None, "error": None},
                "exports": [],
            }
        )
        + "\n"
    )
    (ops_base / "manifests/canary.json").write_text(json.dumps(canary_mf) + "\n")

    receipt_paths: list[Path] = []
    unit_paths: list[Path] = []
    for s in range(p.SHARDS):
        rp = _write_v3_attempt(
            tmp_path / f"base/out-shard-{s:02d}",
            shard=s,
            mode="pass",
            export_slots=p.shard_indices(s),
            pair_rows_by_slot=rows,
        )
        receipt_paths.append(rp)
        for slot in p.shard_indices(s):
            unit_paths.append(rp.parent / f"exports/efficacy/pair-{slot:02d}.json")
        (ops_base / f"manifests/shard-{s:02d}.json").write_text(
            (rp.parent / "manifest.json").read_text()
        )

    with bind_broker_protocol(V3_BROKER_ROOT):
        expected_v3_ledger = v3_resume.build(unit_paths, receipt_paths)
        dec_v3 = v3_final_look.run_final_look(
            ops_dir=ops_base,
            out_dir=tmp_path / "base/v3-out",
            ruling_json=v3_ruling_json,
        )
    assert resume.build(unit_paths, receipt_paths, ops_dir=ops_base) == expected_v3_ledger
    dec_acct = final_look.run_final_look(
        ops_dir=ops_base,
        out_dir=tmp_path / "base/acct-out",
        ruling_json=acct_ruling_json,
        amendment_md=amend_path,
        require_clean_git=False,
    )
    for stat_key in (
        "verdict",
        "replay_serving_decision",
        "n",
        "science",
        "namespace",
        "death_delta_per_1k",
        "death_delta_ci95",
        "death_pass",
        "mass_relative_difference",
        "mass_relative_ci95_fieller",
        "mass_pass",
        "primary_pairs_count",
        "reserve_substitutions_count",
        "final_ledger_sha256",
    ):
        assert dec_acct[stat_key] == dec_v3[stat_key]


def test_dispatch_dry_run_unratified_live_refusal_and_staging_with_never_allocated_job(
    tmp_path: Path,
) -> None:
    ops_dir = tmp_path / "ops"
    _write_never_allocated_shard_fixture(
        tmp_path / "prior-refused",
        ops_dir=ops_dir,
        ops_name="watch6-v3-fix3-shard-00",
        shard=0,
    )
    # Give peer-active-l4-job an upload-measurement so it is not blocking staging:
    (tmp_path / "prior-refused/peer-active-out/upload-measurement.json").write_text("{}\n")

    op_root = tmp_path / "operator"
    (op_root / "tools/ops").mkdir(parents=True)
    called_log = tmp_path / "called.txt"
    fake_dispatch = op_root / "tools/ops/dispatch.sh"
    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'echo "$@" >> "{called_log}"\n'
        'if [[ "$*" == *"--dry-run"* ]]; then exit 0; fi\n'
        "PY_EXPR=\"import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])\"\n"
        'OUT=$(python3 -c "$PY_EXPR" "$2")\n'
        'mkdir -p "$OUT"\n'
        'echo "{}" > "$OUT/upload-measurement.json"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_dispatch.chmod(0o755)

    mf = tmp_path / "shard-00.json"
    mf.write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-shard-00-rec",
                "commit": BASELINE_COMMIT,
                "output_dir": str(tmp_path / "out-rec"),
            }
        ),
        encoding="utf-8",
    )

    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        # 1. --dry-run succeeds before ratification against V3_BROKER_ROOT:
        rc_dry = dispatch.run_dispatch(
            [
                "watch6-v3-shard-00-rec",
                str(mf),
                "--broker",
                str(V3_BROKER_ROOT),
                "--cwd",
                str(V3_BROKER_ROOT),
                "--operator-root",
                str(op_root),
                "--ruling-json",
                str(tmp_path / "nonexistent-ruling.json"),
                "--dry-run",
            ],
            ops_dir=ops_dir,
        )
        assert rc_dry == 0

        # 2. Live dispatch WITHOUT ratified Astra receipt MUST refuse before invoking dispatch.sh!
        before_calls = called_log.read_text(encoding="utf-8").splitlines()
        with pytest.raises(ValueError, match="REFUSE: missing ratified Astra accounting"):
            dispatch.run_dispatch(
                [
                    "watch6-v3-shard-00-rec",
                    str(mf),
                    "--broker",
                    str(V3_BROKER_ROOT),
                    "--cwd",
                    str(V3_BROKER_ROOT),
                    "--operator-root",
                    str(op_root),
                    "--ruling-json",
                    str(tmp_path / "nonexistent-ruling.json"),
                ],
                ops_dir=ops_dir,
            )
        assert called_log.read_text(encoding="utf-8").splitlines() == before_calls

        # 3. Live dispatch WITH ratified Astra receipt passes staging check (prior shard-00
        #    never-allocated attempt is recognized as DONE, not FENCE/STAGING) and succeeds:
        amend_path, ruling_json = _write_accounting_ruling_fixture(tmp_path / "gov")
        with patch.object(
            dispatch,
            "validate_ruling_receipt",
            side_effect=lambda r, amendment_override=None, repo=None, broker_root=None: (
                ratify.validate_ruling_receipt(
                    r,
                    amendment_override=amendment_override,
                    repo=repo,
                    broker_root=broker_root,
                    require_clean_git=False,
                )
            ),
        ):
            rc_live = dispatch.run_dispatch(
                [
                    "watch6-v3-shard-00-rec",
                    str(mf),
                    "--broker",
                    str(V3_BROKER_ROOT),
                    "--cwd",
                    str(V3_BROKER_ROOT),
                    "--operator-root",
                    str(op_root),
                    "--ruling-json",
                    str(ruling_json),
                    "--amendment",
                    str(amend_path),
                ],
                ops_dir=ops_dir,
                poll_seconds=0.02,
            )
        assert rc_live == 0
        assert (tmp_path / "out-rec/upload-measurement.json").is_file()


def _create_clean_baseline_clone(dest: Path) -> Path:
    """Create a fast local --shared git clone checked out at 357e3504 without touching broker."""
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(ADAPTER_ROOT),
            str(dest),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run(
        [
            "git",
            "sparse-checkout",
            "set",
            "--no-cone",
            "/.gitignore",
            "/pyproject.toml",
            "/research/**/*.py",
            "/research/colab_cli/",
            "/research/watch6_search_efficacy_v3/",
            "/research/bl_served_search/requirements.txt",
            "/src/",
            "/web/backend/",
            "/configs/",
            "/tools/ops/",
            "/docs/research/final_run_20261019/",
        ],
        cwd=dest,
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", BASELINE_COMMIT],
        cwd=dest,
        check=True,
        timeout=30,
    )
    return dest


@pytest.fixture(scope="module")
def clean_baseline_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("clean_baseline_tpl") / "broker_357e3504"
    return _create_clean_baseline_clone(base)


def _fresh_baseline_clone(template: Path, dest: Path) -> Path:
    shutil.copytree(template, dest, symlinks=True)
    return dest


def test_live_v3_broker_identity_and_main_checkout_refusal(tmp_path: Path) -> None:
    """Verify real V3 broker computes b26f7fa0... cleanly at 357e3504 and main checkout refuses."""
    assert V3_BROKER_ROOT.is_dir()
    before_status = subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored",
        ],
        cwd=V3_BROKER_ROOT,
        text=True,
        timeout=15,
    )
    broker_head = subprocess.check_output(
        ["git", "--no-optional-locks", "rev-parse", "HEAD"],
        cwd=V3_BROKER_ROOT,
        text=True,
        timeout=10,
    ).strip()
    assert broker_head == BASELINE_COMMIT

    computed = compute_broker_identity(V3_BROKER_ROOT)
    assert computed == BASELINE_SCIENCE
    closure = verify_baseline_closure(V3_BROKER_ROOT, require_head_at_baseline=True)
    assert closure["science"] == BASELINE_SCIENCE
    assert closure["baseline_commit"] == BASELINE_COMMIT
    assert closure["broker_root"] == str(V3_BROKER_ROOT.resolve())

    after_status = subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored",
        ],
        cwd=V3_BROKER_ROOT,
        text=True,
        timeout=15,
    )
    assert after_status == before_status

    if MAIN_CHECKOUT_ROOT.is_dir():
        main_identity = compute_broker_identity(MAIN_CHECKOUT_ROOT)
        if (MAIN_CHECKOUT_ROOT / "src/snake_dqn.egg-info").is_dir():
            assert main_identity != BASELINE_SCIENCE
        with pytest.raises(ValueError, match="REFUSE: main checkout"):
            verify_baseline_closure(MAIN_CHECKOUT_ROOT)
        with pytest.raises(ValueError, match="REFUSE: main checkout"):
            dispatch.run_dispatch(
                [
                    "watch6-v3-main-refused",
                    str(tmp_path / "missing.json"),
                    "--broker",
                    str(MAIN_CHECKOUT_ROOT),
                    "--cwd",
                    str(MAIN_CHECKOUT_ROOT),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
            )
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE.*main checkout"):
            final_look.run_final_look(
                ops_dir=tmp_path / "ops",
                out_dir=tmp_path / "look-out",
                broker_root=MAIN_CHECKOUT_ROOT,
                require_clean_git=False,
            )


def test_same_root_pass_on_clean_copy_and_stray_untracked_or_ignored_refusal(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Point 3 requirement (4): same-root pass on clean copy; refuse untracked/ignored drift."""
    clean_copy = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clean-v3-copy")

    # 1. Cross-root pass (p.ROOT != clean_copy) and same-root pass (p.ROOT == clean_copy):
    assert compute_broker_identity(clean_copy) == BASELINE_SCIENCE
    cross_ok = verify_baseline_closure(clean_copy, require_head_at_baseline=True)
    assert cross_ok["science"] == BASELINE_SCIENCE

    with patch.object(p, "ROOT", clean_copy):
        p.source_paths.cache_clear()
        p.identity.cache_clear()
        assert p.identity() == BASELINE_SCIENCE
        same_ok = verify_baseline_closure(clean_copy, require_head_at_baseline=True)
        assert same_ok["science"] == BASELINE_SCIENCE
    p.source_paths.cache_clear()
    p.identity.cache_clear()

    # 2. Stray git-ignored files (`src/snake_dqn.egg-info/*.txt`) MUST change identity (zero
    #    exclusions!) and MUST be refused in BOTH same-root and cross-root verification:
    egg_dir = clean_copy / "src/snake_dqn.egg-info"
    egg_dir.mkdir(parents=True)
    for fname in (
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "requires.txt",
        "top_level.txt",
    ):
        (egg_dir / fname).write_text(f"synthetic-egg-info-{fname}\n", encoding="utf-8")

    # Confirm git status without --ignored treats egg-info as clean/ignored:
    porcelain = subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
        cwd=clean_copy,
        text=True,
        timeout=10,
    ).strip()
    assert porcelain == ""

    polluted_egg_identity = compute_broker_identity(clean_copy)
    assert polluted_egg_identity != BASELINE_SCIENCE

    # Cross-root check (while p.ROOT is clean) MUST refuse (no hardcoded BASELINE_SCIENCE!):
    with pytest.raises(ValueError, match=polluted_egg_identity):
        verify_baseline_closure(clean_copy, require_head_at_baseline=True)

    # Same-root check (p.ROOT == clean_copy) MUST refuse with the exact same identity:
    with patch.object(p, "ROOT", clean_copy):
        p.source_paths.cache_clear()
        p.identity.cache_clear()
        with pytest.raises(ValueError, match=polluted_egg_identity):
            verify_baseline_closure(clean_copy, require_head_at_baseline=True)
    p.source_paths.cache_clear()
    p.identity.cache_clear()

    # Removing the ignored egg-info directory restores exact b26f7fa0... pass:
    shutil.rmtree(egg_dir)
    assert compute_broker_identity(clean_copy) == BASELINE_SCIENCE
    assert verify_baseline_closure(clean_copy, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )

    # 3. Stray untracked file inside hashed closure (`src/stray_untracked.txt`) MUST change
    #    identity and refuse in BOTH same-root and cross-root verification:
    stray_txt = clean_copy / "src/stray_untracked.txt"
    stray_txt.write_text("stray untracked payload\n", encoding="utf-8")
    stray_identity = compute_broker_identity(clean_copy)
    assert stray_identity != BASELINE_SCIENCE

    with pytest.raises(ValueError, match=stray_identity):
        verify_baseline_closure(clean_copy, require_head_at_baseline=True)

    with patch.object(p, "ROOT", clean_copy):
        p.source_paths.cache_clear()
        p.identity.cache_clear()
        with pytest.raises(ValueError, match=stray_identity):
            verify_baseline_closure(clean_copy, require_head_at_baseline=True)
    p.source_paths.cache_clear()
    p.identity.cache_clear()
    stray_txt.unlink()

    # 4. Tampering with V3 protocol.py (e.g. adding an exclusion or hardcoded identity) refuses:
    proto_file = clean_copy / "research/watch6_search_efficacy_v3/protocol.py"
    orig_proto_bytes = proto_file.read_bytes()
    proto_file.write_text(
        orig_proto_bytes.decode("utf-8") + "\n# unauthorized edit\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="V3 protocol.py modified"):
        verify_baseline_closure(clean_copy, require_head_at_baseline=True)
    proto_file.write_bytes(orig_proto_bytes)


def test_refusal_of_any_cross_root_substitution_across_entrypoints(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Point 3 req (4): refuse every cross-root substitution across dispatch/recover/look."""
    clean_copy = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clean-broker")
    dirty_copy = _fresh_baseline_clone(clean_baseline_template, tmp_path / "dirty-egg-broker")
    (dirty_copy / "src/snake_dqn.egg-info").mkdir(parents=True)
    (dirty_copy / "src/snake_dqn.egg-info/SOURCES.txt").write_text(
        "src/main.py\n",
        encoding="utf-8",
    )
    dirty_identity = compute_broker_identity(dirty_copy)
    assert dirty_identity != BASELINE_SCIENCE

    op_root = tmp_path / "operator"
    (op_root / "tools/ops").mkdir(parents=True)
    called_log = tmp_path / "dispatch-invocations.log"
    fake_dispatch = op_root / "tools/ops/dispatch.sh"
    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n" f'echo "$@" >> "{called_log}"\n' "exit 0\n",
        encoding="utf-8",
    )
    fake_dispatch.chmod(0o755)

    mf = tmp_path / "shard-00.json"
    mf.write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-cross-root-probe",
                "commit": BASELINE_COMMIT,
                "output_dir": str(tmp_path / "out-probe"),
            }
        ),
        encoding="utf-8",
    )

    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        # 1. Clean host checkout + dirty --broker (ignored egg-info) MUST refuse before dispatch.sh:
        with pytest.raises(ValueError, match=dirty_identity):
            dispatch.run_dispatch(
                [
                    "watch6-v3-cross-root-probe",
                    str(mf),
                    "--broker",
                    str(dirty_copy),
                    "--cwd",
                    str(dirty_copy),
                    "--operator-root",
                    str(op_root),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
            )
        assert not called_log.exists()

        # 2. Clean --broker + different --cwd MUST refuse (no --broker/--cwd split substitution):
        with pytest.raises(ValueError, match="cross-root substitution forbidden"):
            dispatch.run_dispatch(
                [
                    "watch6-v3-cross-root-probe",
                    str(mf),
                    "--broker",
                    str(clean_copy),
                    "--cwd",
                    str(dirty_copy),
                    "--operator-root",
                    str(op_root),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
            )
        assert not called_log.exists()

        # 3. Clean broker_root parameter + conflicting CLI --broker MUST refuse:
        with pytest.raises(ValueError, match="cross-root substitution forbidden"):
            dispatch.run_dispatch(
                [
                    "watch6-v3-cross-root-probe",
                    str(mf),
                    "--broker",
                    str(V3_BROKER_ROOT),
                    "--cwd",
                    str(V3_BROKER_ROOT),
                    "--operator-root",
                    str(op_root),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
                broker_root=clean_copy,
            )
        assert not called_log.exists()

        # 4. Non-git --broker directory (previously skipped by .git check) refuses:
        non_git_dir = tmp_path / "non-git-broker"
        non_git_dir.mkdir()
        with pytest.raises(ValueError, match="REFUSE"):
            dispatch.run_dispatch(
                [
                    "watch6-v3-cross-root-probe",
                    str(mf),
                    "--broker",
                    str(non_git_dir),
                    "--cwd",
                    str(non_git_dir),
                    "--operator-root",
                    str(op_root),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
            )
        assert not called_log.exists()

        # 5. Worktree not at HEAD == 357e3504 (both science-identical ahead clone and ADAPTER_ROOT)
        #    refused as execution broker:
        ahead_copy = _fresh_baseline_clone(clean_baseline_template, tmp_path / "ahead-broker")
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "ahead-of-baseline",
            ],
            cwd=ahead_copy,
            check=True,
            timeout=10,
        )
        with pytest.raises(ValueError, match="HEAD is"):
            dispatch.run_dispatch(
                [
                    "watch6-v3-cross-root-probe",
                    str(mf),
                    "--broker",
                    str(ahead_copy),
                    "--cwd",
                    str(ahead_copy),
                    "--operator-root",
                    str(op_root),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
            )
        assert not called_log.exists()

        if (
            subprocess.check_output(
                ["git", "--no-optional-locks", "rev-parse", "HEAD"],
                cwd=ADAPTER_ROOT,
                text=True,
            ).strip()
            != BASELINE_COMMIT
        ):
            with pytest.raises(
                ValueError,
                match=r"(HEAD is|protocol\.identity\(\) rotated|main checkout .* is forbidden)",
            ):
                dispatch.run_dispatch(
                    [
                        "watch6-v3-cross-root-probe",
                        str(mf),
                        "--broker",
                        str(ADAPTER_ROOT),
                        "--cwd",
                        str(ADAPTER_ROOT),
                        "--operator-root",
                        str(op_root),
                        "--dry-run",
                    ],
                    ops_dir=tmp_path / "ops",
                )
            assert not called_log.exists()

    # 6. Dirty broker_root refused across recover, final_look, admission, resume, prepare, ratify:
    with pytest.raises(ValueError, match=dirty_identity):
        recover.recover(
            base_package=tmp_path,
            receipts=[],
            target_shards=[0],
            out=tmp_path / "rec-dirty",
            benchmark=tmp_path / "b.json",
            mirror_receipt=tmp_path / "m.json",
            empirical_costs=tmp_path / "c.json",
            commit=BASELINE_COMMIT,
            broker_root=dirty_copy,
        )
    with pytest.raises(ValueError, match=f"INVALID_OR_INCOMPLETE.*{dirty_identity}"):
        final_look.run_final_look(
            ops_dir=tmp_path / "ops",
            out_dir=tmp_path / "look-dirty",
            broker_root=dirty_copy,
            require_clean_git=False,
        )
    with pytest.raises(ValueError, match=dirty_identity):
        admission.admit({}, broker_root=dirty_copy)
    with pytest.raises(ValueError, match=dirty_identity):
        resume.build([], [], broker_root=dirty_copy)
    with pytest.raises(ValueError, match=dirty_identity):
        prepare.prepare(
            tmp_path / "prep-dirty",
            tmp_path / "b.json",
            BASELINE_COMMIT,
            tmp_path / "m.json",
            tmp_path / "c.json",
            broker_root=dirty_copy,
        )

    amend_path, ruling_json = _write_accounting_ruling_fixture(tmp_path / "gov-cross")
    with pytest.raises(ValueError, match=dirty_identity):
        ratify.verify_amendment_document(
            amend_path,
            broker_root=dirty_copy,
            require_clean_git=False,
        )
    # Ruling receipt bound to V3_BROKER_ROOT refuses validation under a different broker_root:
    with pytest.raises(ValueError, match="ruling receipt broker_root"):
        ratify.validate_ruling_receipt(
            ruling_json,
            broker_root=clean_copy,
            require_clean_git=False,
        )

    # 7. Invoking from a dirty host checkout (p.ROOT == dirty_copy, as on main checkout) still
    #    refuses dirty_copy while default binding to V3_BROKER_ROOT attests V3_BROKER_ROOT's bytes:
    with patch.object(p, "ROOT", dirty_copy):
        p.source_paths.cache_clear()
        p.identity.cache_clear()
        assert p.identity() == dirty_identity
        with pytest.raises(ValueError, match=dirty_identity):
            verify_baseline_closure(dirty_copy)

        validated = ratify.validate_ruling_receipt(ruling_json, require_clean_git=False)
        assert validated["broker_root"] == str(V3_BROKER_ROOT.resolve())
        assert validated["science"] == BASELINE_SCIENCE

        with (
            patch.object(dispatch, "admit", return_value={"status": "PASS"}),
            patch.object(
                dispatch,
                "verify_amendment_document",
                side_effect=lambda a, repo=None, broker_root=None: (
                    ratify.verify_amendment_document(
                        a,
                        repo=repo,
                        broker_root=broker_root,
                        require_clean_git=False,
                    )
                ),
            ),
        ):
            rc = dispatch.run_dispatch(
                [
                    str(mf),
                    "--operator-root",
                    str(op_root),
                    "--amendment",
                    str(amend_path),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
            )
            assert rc == 0
    p.source_paths.cache_clear()
    p.identity.cache_clear()
    logged = called_log.read_text(encoding="utf-8")
    assert f"--broker {V3_BROKER_ROOT.resolve()} --cwd {V3_BROKER_ROOT.resolve()}" in logged


def _assert_all_broker_entrypoints_refuse_clone(
    clone: Path,
    tmp_path: Path,
    *,
    expected_match: str,
) -> None:
    """Assert verify_baseline_closure (cross-root & same-root) + all entrypoints refuse clone."""
    with pytest.raises(ValueError, match=expected_match):
        verify_baseline_closure(clone, require_head_at_baseline=True)

    with patch.object(p, "ROOT", clone):
        p.source_paths.cache_clear()
        p.identity.cache_clear()
        with pytest.raises(ValueError, match=expected_match):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    p.source_paths.cache_clear()
    p.identity.cache_clear()

    with pytest.raises(ValueError, match=expected_match):
        with bind_broker_protocol(clone):
            pass

    op_root = tmp_path / "op"
    (op_root / "tools/ops").mkdir(parents=True, exist_ok=True)
    fake_dispatch = op_root / "tools/ops/dispatch.sh"
    fake_dispatch.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_dispatch.chmod(0o755)
    mf = tmp_path / "probe-mf.json"
    mf.write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-point3-probe",
                "commit": BASELINE_COMMIT,
                "output_dir": str(tmp_path / "probe-out"),
            }
        ),
        encoding="utf-8",
    )
    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        with pytest.raises(ValueError, match=expected_match):
            dispatch.run_dispatch(
                [
                    "watch6-v3-point3-probe",
                    str(mf),
                    "--broker",
                    str(clone),
                    "--cwd",
                    str(clone),
                    "--operator-root",
                    str(op_root),
                    "--dry-run",
                ],
                ops_dir=tmp_path / "ops",
            )

    with pytest.raises(ValueError, match=expected_match):
        recover.recover(
            base_package=tmp_path,
            receipts=[],
            target_shards=[0],
            out=tmp_path / "rec-out",
            benchmark=tmp_path / "b.json",
            mirror_receipt=tmp_path / "m.json",
            empirical_costs=tmp_path / "c.json",
            commit=BASELINE_COMMIT,
            broker_root=clone,
        )
    with pytest.raises(ValueError, match=f"INVALID_OR_INCOMPLETE.*{expected_match}"):
        final_look.run_final_look(
            ops_dir=tmp_path / "ops",
            out_dir=tmp_path / "look-out",
            broker_root=clone,
            require_clean_git=False,
        )
    with pytest.raises(ValueError, match=expected_match):
        admission.admit({}, broker_root=clone)
    with pytest.raises(ValueError, match=expected_match):
        resume.build([], [], broker_root=clone)
    with pytest.raises(ValueError, match=expected_match):
        prepare.prepare(
            tmp_path / "prep-out",
            tmp_path / "b.json",
            BASELINE_COMMIT,
            tmp_path / "m.json",
            tmp_path / "c.json",
            broker_root=clone,
        )
    amend_path, _ = _write_accounting_ruling_fixture(tmp_path / "gov-probe")
    with pytest.raises(ValueError, match=expected_match):
        ratify.verify_amendment_document(
            amend_path,
            broker_root=clone,
            require_clean_git=False,
        )


def test_point3_refuses_lone_egg_info_pkg_info_unmasked(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Astra Ruling 3 Point 3 reproducer 1: lone ignored src/snake_dqn.egg-info/PKG-INFO."""
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-lone-pkg-info")
    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )

    pkg_info = clone / "src/snake_dqn.egg-info/PKG-INFO"
    pkg_info.parent.mkdir(parents=True, exist_ok=True)
    pkg_info.write_text("Metadata-Version: 2.1\nName: snake-dqn\n", encoding="utf-8")

    # Ensure strictly unmasked: no .txt siblings, git status without --ignored is empty,
    # and unmodified protocol.identity() still hashes to BASELINE_SCIENCE:
    assert list(pkg_info.parent.iterdir()) == [pkg_info]
    porcelain = subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
        cwd=clone,
        text=True,
        timeout=10,
    ).strip()
    assert porcelain == ""
    assert compute_broker_identity(clone) == BASELINE_SCIENCE

    _assert_all_broker_entrypoints_refuse_clone(
        clone,
        tmp_path / "ep-pkg-info",
        expected_match=r"packaging metadata.*src/snake_dqn\.egg-info",
    )


def test_point3_refuses_lone_extension_module_so_shadowing_py_unmasked(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Astra Ruling 3 Point 3 reproducer 2: lone ignored src/model/ego2s_network.so."""
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-lone-ego2s-so")
    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )

    planted_so = clone / "src/model/ego2s_network.so"
    planted_so.write_bytes(b"\xcf\xfa\xed\xfe\x00foreign-macho-stub")

    # Confirm git status without --ignored omits *.so AND protocol.identity() stays b26f7fa0...:
    porcelain = subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
        cwd=clone,
        text=True,
        timeout=10,
    ).strip()
    assert porcelain == ""
    assert compute_broker_identity(clone) == BASELINE_SCIENCE

    # Confirm Python's FileFinder prefers .so over tracked ego2s_network.py (without loading it):
    if ".so" in importlib.machinery.EXTENSION_SUFFIXES:
        finder = importlib.machinery.FileFinder(
            str(clone / "src/model"),
            (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES),
            (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES),
        )
        spec = finder.find_spec("ego2s_network")
        assert spec is not None and spec.origin == str(planted_so)

    _assert_all_broker_entrypoints_refuse_clone(
        clone,
        tmp_path / "ep-ego2s-so",
        expected_match=r"extension module.*src/model/ego2s_network\.so",
    )


def test_point3_refuses_each_foreign_closure_artifact_class_alone(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Each foreign/ignored closure file or directory class is refused strictly ALONE."""
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-foreign-classes")
    ext_tag = sys.implementation.cache_tag

    def _assert_clean() -> None:
        res = verify_baseline_closure(clone, require_head_at_baseline=True)
        assert res["science"] == BASELINE_SCIENCE

    _assert_clean()

    lone_file_cases: list[tuple[str, bytes, bool, str]] = [
        # (relative_path, payload, identity_unchanged, expected_error_regex)
        (
            f"src/model/ego2s_network.{ext_tag}-darwin.so",
            b"\xcf\xfa\xed\xfe\x00abi-so",
            True,
            r"extension module.*ego2s_network",
        ),
        (
            "src/model/ego2s_network.pyd",
            b"MZ\x90\x00foreign-pyd",
            True,
            r"extension module.*src/model/ego2s_network\.pyd",
        ),
        (
            "src/simd_env/libgrid_override.dylib",
            b"\xcf\xfa\xed\xfe\x00dylib",
            True,
            r"extension module.*src/simd_env/libgrid_override\.dylib",
        ),
        (
            "tools/ops/package_spec.so",
            b"\xcf\xfa\xed\xfe\x00ops-so",
            True,
            r"extension module.*tools/ops/package_spec\.so",
        ),
        (
            "research/bl_served_search/analyze.so",
            b"\xcf\xfa\xed\xfe\x00research-so",
            True,
            r"extension module.*research/bl_served_search/analyze\.so",
        ),
        (
            "src/foreign_hook.pth",
            b"import os\n",
            True,
            r"path-hook.*src/foreign_hook\.pth",
        ),
        (
            "foreign_top_level.pth",
            b"import sys\n",
            True,
            r"path-hook.*foreign_top_level\.pth",
        ),
        (
            "src/foreign_bundle.egg",
            b"PK\x03\x04egg",
            True,
            r"packaging metadata.*src/foreign_bundle\.egg",
        ),
        (
            "src/snake_dqn-0.1.0.dist-info/METADATA",
            b"Metadata-Version: 2.1\nName: snake-dqn\n",
            True,
            r"packaging metadata.*src/snake_dqn-0\.1\.0\.dist-info",
        ),
        (
            "sitecustomize.py",
            b"# foreign top-level startup module\n",
            True,
            r"stray Python source or bytecode.*sitecustomize\.py",
        ),
    ]

    for rel_path, payload, expect_same_science, err_regex in lone_file_cases:
        target = clone / rel_path
        created_parent = not target.parent.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        try:
            if expect_same_science:
                assert compute_broker_identity(clone) == BASELINE_SCIENCE
            with pytest.raises(ValueError, match=err_regex):
                verify_baseline_closure(clone, require_head_at_baseline=True)
        finally:
            target.unlink(missing_ok=True)
            if created_parent:
                shutil.rmtree(target.parent, ignore_errors=True)
        _assert_clean()

    # Empty ignored src/snake_dqn.egg-info/ directory alone:
    empty_egg = clone / "src/snake_dqn.egg-info"
    empty_egg.mkdir(parents=True)
    try:
        assert compute_broker_identity(clone) == BASELINE_SCIENCE
        with pytest.raises(ValueError, match=r"packaging metadata.*src/snake_dqn\.egg-info"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        empty_egg.rmdir()
    _assert_clean()

    # Git-ignored stray .py (via .git/info/exclude) inside a single-file-entry closure parent dir
    # (`tools/ops/shadow_helper.py`), where BOTH git status and protocol.identity() stay unchanged:
    exclude_file = clone / ".git/info/exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    orig_exclude = exclude_file.read_text(encoding="utf-8") if exclude_file.is_file() else ""
    ignored_py = clone / "tools/ops/shadow_helper.py"
    try:
        exclude_file.write_text(orig_exclude + "\nshadow_helper.py\n", encoding="utf-8")
        ignored_py.write_text("SHADOW = 1\n", encoding="utf-8")
        porcelain = subprocess.check_output(
            ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
            cwd=clone,
            text=True,
            timeout=10,
        ).strip()
        assert porcelain == ""
        assert compute_broker_identity(clone) == BASELINE_SCIENCE
        with pytest.raises(
            ValueError,
            match=r"stray Python source or bytecode.*tools/ops/shadow_helper\.py",
        ):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        ignored_py.unlink(missing_ok=True)
        exclude_file.write_text(orig_exclude, encoding="utf-8")
    _assert_clean()


def _snapshot_tree_state(root: Path) -> dict[str, tuple[str, int]]:
    """Snapshot all non-.git directory and file entries in ``root`` for zero-write assertions."""
    state: dict[str, tuple[str, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        rel_dp = dp.relative_to(root)
        if rel_dp.parts and rel_dp.parts[0] == ".git":
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        if rel_dp != Path("."):
            state[rel_dp.as_posix()] = ("dir", 0)
        for fn in sorted(filenames):
            fp = dp / fn
            rel_fp = fp.relative_to(root).as_posix()
            if fp.is_symlink():
                state[rel_fp] = ("symlink", len(os.readlink(fp)))
            else:
                state[rel_fp] = ("file", fp.stat().st_size)
    return state


def _compile_genuine_coarse_words_cache(clone: Path, out_cache_dir: Path) -> tuple[Path, Path]:
    """Compile tracked ``src/simd_env/ego_raster_nb.py`` ``_coarse_words_kernel`` out-of-tree."""
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    pycache_out = out_cache_dir.parent / "compile_cpython_pycache"
    pycache_out.mkdir(parents=True, exist_ok=True)
    before_clone = _snapshot_tree_state(clone)
    script = textwrap.dedent("""
        import sys
        import numpy as np
        from src.simd_env import ego_raster_nb

        owner_w = np.full((1, 16, 4), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        owner_w[0, 4, 1] = np.uint64(0xFFFFFFFFFFFF0000)
        food_w = np.zeros((1, 16, 2), dtype=np.uint64)
        food_w[0, 4, 0] = np.uint64(1)
        env_needed = np.array([True], dtype=np.bool_)
        counts = np.zeros((1, 2, 4, 4), dtype=np.int32)
        total = np.zeros((1, 4, 4), dtype=np.int32)
        fcounts = np.zeros((1, 4, 4), dtype=np.int32)
        ego_raster_nb._coarse_words_kernel(
            owner_w, food_w, env_needed, 2, 8, 8, 2, counts, total, fcounts
        )
        assert int(total.sum()) >= 1
        """)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(clone)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(pycache_out)
    env["NUMBA_CACHE_DIR"] = str(out_cache_dir)
    env["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
    subprocess.run(
        [os.path.abspath(sys.executable), "-B", "-c", script],
        cwd=clone,
        env=env,
        check=True,
        timeout=60,
    )
    assert _snapshot_tree_state(clone) == before_clone
    nbis = sorted(out_cache_dir.rglob("ego_raster_nb._coarse_words_kernel-233.py*.nbi"))
    nbcs = sorted(out_cache_dir.rglob("ego_raster_nb._coarse_words_kernel-233.py*.1.nbc"))
    assert len(nbis) == 1 and len(nbcs) == 1
    return nbis[0], nbcs[0]


class _InnerEvalGadget:
    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (builtins.eval, (self.expr,))


def _build_numba_serialize_loads_pickle(inner_pickle: bytes) -> bytes:
    """Craft Astra Ruling 4 Point 3 probe (b) outer pickle: numba.core.serialize.loads(inner)."""
    import struct

    return (
        b"\x80\x04"
        b"\x8c\x14numba.core.serialize"
        b"\x8c\x05loads"
        b"\x93"
        b"B" + struct.pack("<I", len(inner_pickle)) + inner_pickle + b"\x85R."
    )


def test_live_v3_broker_passes_after_quarantine_with_zero_caches() -> None:
    """Ruling 4 req (3)-(4): live V3 broker has zero caches after quarantine and passes closure."""
    assert V3_BROKER_ROOT.is_dir()
    found_caches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(V3_BROKER_ROOT, followlinks=False):
        dp = Path(dirpath)
        rel_dp = dp.relative_to(V3_BROKER_ROOT)
        if rel_dp.parts and rel_dp.parts[0] == ".git":
            dirnames[:] = []
            continue
        for dn in sorted(dirnames):
            if dn == "__pycache__":
                found_caches.append((dp / dn).relative_to(V3_BROKER_ROOT).as_posix())
        for fn in sorted(filenames):
            if fn.lower().endswith((".nbi", ".nbc", ".pyc", ".pyo")):
                found_caches.append((dp / fn).relative_to(V3_BROKER_ROOT).as_posix())
    assert found_caches == [], f"unexpected cache artifacts in live V3 broker: {found_caches}"

    closure = verify_baseline_closure(V3_BROKER_ROOT, require_head_at_baseline=True)
    assert closure == {
        "broker_root": str(V3_BROKER_ROOT.resolve()),
        "baseline_commit": BASELINE_COMMIT,
        "science": BASELINE_SCIENCE,
        "namespace": BASELINE_NAMESPACE,
        "registration_sha256": REGISTRATION_V3_SHA256,
    }


@pytest.fixture(scope="module")
def compiled_coarse_words_cache(
    clean_baseline_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("compiled_coarse_words")
    return _compile_genuine_coarse_words_cache(clean_baseline_template, base / "ext-nb-cache")


def test_ruling4_probe_a_mutated_macho_nbc_and_genuine_numba_cache_refused(
    clean_baseline_template: Path,
    compiled_coarse_words_cache: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Astra Ruling 4 Point 3 probe (a): flipped Mach-O instruction in .nbc (+ genuine) refused."""
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-probe-a")
    src_nbi, src_nbc = compiled_coarse_words_cache

    genuine_nbi_bytes = src_nbi.read_bytes()
    genuine_nbc_bytes = src_nbc.read_bytes()
    tracked_ego_sha = (
        __import__("hashlib")
        .sha256((clone / "src/simd_env/ego_raster_nb.py").read_bytes())
        .digest()
    )
    assert tracked_ego_sha in genuine_nbi_bytes

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        orig_insn = bytes.fromhex("ff4306d1")
        flipped_insn = bytes.fromhex("fe4306d1")
        assert orig_insn in genuine_nbc_bytes
        mutated_nbc_bytes = genuine_nbc_bytes.replace(orig_insn, flipped_insn, 1)
    else:
        mutated_nbc_bytes = (
            genuine_nbc_bytes[:-16]
            + bytes([genuine_nbc_bytes[-16] ^ 0x01])
            + genuine_nbc_bytes[-15:]
        )
    assert mutated_nbc_bytes != genuine_nbc_bytes

    simd_pycache = clone / "src/simd_env/__pycache__"
    simd_pycache.mkdir(parents=True, exist_ok=True)
    planted_nbi = simd_pycache / src_nbi.name
    planted_nbc = simd_pycache / src_nbc.name

    # Probe (a): genuine matching-stamp .nbi + Mach-O instruction-flipped .nbc:
    planted_nbi.write_bytes(genuine_nbi_bytes)
    planted_nbc.write_bytes(mutated_nbc_bytes)
    porcelain = subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
        cwd=clone,
        text=True,
        timeout=10,
    ).strip()
    assert porcelain == ""
    assert compute_broker_identity(clone) == BASELINE_SCIENCE

    _assert_all_broker_entrypoints_refuse_clone(
        clone,
        tmp_path / "ep-probe-a-mutated",
        expected_match=r"Numba native cache payload.*ego_raster_nb\._coarse_words_kernel-233",
    )

    # Unmutated genuine .nbi + .nbc pair is ALSO refused across all entrypoints (no whitelist):
    planted_nbc.write_bytes(genuine_nbc_bytes)
    _assert_all_broker_entrypoints_refuse_clone(
        clone,
        tmp_path / "ep-probe-a-genuine",
        expected_match=r"Numba native cache payload.*ego_raster_nb\._coarse_words_kernel-233",
    )

    # Lone .nbi and lone .nbc (inside __pycache__ and outside __pycache__) each refuse alone:
    planted_nbc.unlink()
    with pytest.raises(ValueError, match=r"Numba native cache payload.*\.nbi"):
        verify_baseline_closure(clone, require_head_at_baseline=True)
    planted_nbi.unlink()

    planted_nbc.write_bytes(genuine_nbc_bytes)
    with pytest.raises(ValueError, match=r"Numba native cache payload.*\.nbc"):
        verify_baseline_closure(clone, require_head_at_baseline=True)
    planted_nbc.unlink()
    simd_pycache.rmdir()

    stray_nbc = clone / "src/simd_env" / src_nbc.name
    stray_nbc.write_bytes(mutated_nbc_bytes)
    try:
        with pytest.raises(ValueError, match=r"Numba native cache payload.*\.nbc"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        stray_nbc.unlink()

    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )


def test_ruling4_probe_b_nested_numba_serialize_eval_nbc_refused(
    clean_baseline_template: Path,
    compiled_coarse_words_cache: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Astra Ruling 4 Point 3 probe (b): numba.core.serialize.loads(builtins.eval('6*7')) .nbc."""
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-probe-b")
    src_nbi, src_nbc = compiled_coarse_words_cache

    inner_42 = pickle.dumps(_InnerEvalGadget("6*7"), protocol=4)
    outer_42_nbc = _build_numba_serialize_loads_pickle(inner_42)
    # Confirm Astra's exact nested-pickle gadget evaluates 6*7 == 42 while exposing only numba.*
    # at the outer pickle stream level:
    assert pickle.loads(outer_42_nbc) == 42
    outer_globals: list[str] = []
    stack: list[object] = []
    for op, arg, _ in pickletools.genops(io.BytesIO(outer_42_nbc)):
        if op.name in ("MEMOIZE", "FRAME", "PROTO"):
            continue
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            stack.append(arg)
        elif op.name == "GLOBAL":
            outer_globals.append(str(arg))
            stack.append(None)
        elif op.name == "STACK_GLOBAL":
            attr_n = stack.pop()
            mod_n = stack.pop()
            outer_globals.append(f"{mod_n}.{attr_n}")
            stack.append(None)
        else:
            stack.append(None)
    assert outer_globals == ["numba.core.serialize.loads"]

    simd_pycache = clone / "src/simd_env/__pycache__"
    simd_pycache.mkdir(parents=True, exist_ok=True)
    planted_nbi = simd_pycache / src_nbi.name
    planted_nbc = simd_pycache / src_nbc.name
    planted_nbi.write_bytes(src_nbi.read_bytes())
    planted_nbc.write_bytes(outer_42_nbc)

    assert compute_broker_identity(clone) == BASELINE_SCIENCE
    _assert_all_broker_entrypoints_refuse_clone(
        clone,
        tmp_path / "ep-probe-b-42",
        expected_match=r"Numba native cache payload.*ego_raster_nb\._coarse_words_kernel-233",
    )

    # Also prove the adapter NEVER unpickles .nbc during refusal:
    sentinel = tmp_path / "UNPICKLED_SENTINEL"
    inner_sentinel = pickle.dumps(
        _InnerEvalGadget(f"open({str(sentinel)!r}, 'w').write('boom')"),
        protocol=4,
    )
    outer_sentinel_nbc = _build_numba_serialize_loads_pickle(inner_sentinel)
    planted_nbc.write_bytes(outer_sentinel_nbc)
    _assert_all_broker_entrypoints_refuse_clone(
        clone,
        tmp_path / "ep-probe-b-sentinel",
        expected_match=r"Numba native cache payload.*ego_raster_nb\._coarse_words_kernel-233",
    )
    assert not sentinel.exists()

    shutil.rmtree(simd_pycache)
    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )


def test_point3_refuses_all_pycache_pyc_and_non_cpython_caches_outright(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Simplest fail-closed cache rule under -B: refuse all __pycache__, .pyc, and non-CPython."""
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-all-caches")
    tag = sys.implementation.cache_tag
    tracked_py = clone / "src/model/ego2s_network.py"
    pycache_dir = clone / "src/model/__pycache__"
    valid_pyc = pycache_dir / f"ego2s_network.{tag}.pyc"

    # 1. Genuine CPython .pyc in __pycache__/ (TIMESTAMP, CHECKED_HASH, UNCHECKED_HASH) is REFUSED:
    for idx, mode in enumerate(
        (
            py_compile.PycInvalidationMode.TIMESTAMP,
            py_compile.PycInvalidationMode.CHECKED_HASH,
            py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
    ):
        py_compile.compile(
            str(tracked_py),
            cfile=str(valid_pyc),
            dfile="src/model/ego2s_network.py",
            doraise=True,
            invalidation_mode=mode,
        )
        try:
            assert valid_pyc.is_file()
            assert compute_broker_identity(clone) == BASELINE_SCIENCE
            if idx == 0:
                _assert_all_broker_entrypoints_refuse_clone(
                    clone,
                    tmp_path / "ep-pyc-refused",
                    expected_match=r"CPython bytecode cache.*src/model/__pycache__/ego2s_network",
                )
            else:
                with pytest.raises(
                    ValueError,
                    match=r"CPython bytecode cache.*src/model/__pycache__/ego2s_network",
                ):
                    verify_baseline_closure(clone, require_head_at_baseline=True)
        finally:
            shutil.rmtree(pycache_dir, ignore_errors=True)

    # 2. Stray .pyc outside __pycache__/ is REFUSED:
    stray_pyc = clone / "src/model/ego2s_network.pyc"
    py_compile.compile(str(tracked_py), cfile=str(stray_pyc), doraise=True)
    try:
        with pytest.raises(
            ValueError,
            match=r"CPython bytecode cache.*src/model/ego2s_network\.pyc",
        ):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        stray_pyc.unlink()

    # 3. Non-CPython cache payloads inside __pycache__/ (.so, .pkl) are REFUSED:
    for fname in ("stealth.so", "cached_blob.pkl"):
        pycache_dir.mkdir(parents=True, exist_ok=True)
        foreign_cache = pycache_dir / fname
        foreign_cache.write_bytes(b"\x80\x04\x95\x02\x00\x00\x00\x00\x00\x00\x00N.")
        try:
            with pytest.raises(ValueError, match=r"forbidden non-CPython cache payload"):
                verify_baseline_closure(clone, require_head_at_baseline=True)
        finally:
            shutil.rmtree(pycache_dir, ignore_errors=True)

    # 4. Empty __pycache__/ directories (in src/, research/, or top-level) are REFUSED:
    for empty_rel in (
        "src/simd_env/__pycache__",
        "research/watch6_search_efficacy_v3/__pycache__",
        "__pycache__",
    ):
        empty_pc = clone / empty_rel
        empty_pc.mkdir(parents=True, exist_ok=True)
        try:
            with pytest.raises(ValueError, match=r"forbidden __pycache__ directory"):
                verify_baseline_closure(clone, require_head_at_baseline=True)
        finally:
            empty_pc.rmdir()

    # 5. Empty untracked non-__pycache__ directory inside closure is REFUSED:
    untracked_cache_dir = clone / "src/simd_env/.numba_cache"
    untracked_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(
            ValueError,
            match=r"foreign untracked or ignored directory.*src/simd_env/\.numba_cache",
        ):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        untracked_cache_dir.rmdir()

    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )


def test_fresh_cache_entrypoints_execute_broker_numba_and_write_nothing_in_tree(
    clean_baseline_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling 4 req (2) & (4): fresh empty NUMBA_CACHE_DIR + -B + PYTHONDONTWRITEBYTECODE=1."""
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-fresh-cache")
    before_clone_state = _snapshot_tree_state(clone)
    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )
    assert _snapshot_tree_state(clone) == before_clone_state

    # Poison caller env to prove entrypoints override NUMBA_CACHE_DIR, PYTHONDONTWRITEBYTECODE,
    # and even a TMPDIR pointing inside the broker tree:
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(clone / "src/simd_env/__pycache__"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.setenv("TMPDIR", str(clone))

    # 1. Dispatch preflight executing real broker code + @njit(cache=True) (_coarse_words_kernel):
    op_root = tmp_path / "op-fresh"
    (op_root / "tools/ops").mkdir(parents=True, exist_ok=True)
    telemetry_jsonl = tmp_path / "preflight-cache-telemetry.jsonl"
    fake_dispatch = op_root / "tools/ops/dispatch.sh"
    fake_dispatch.write_text(
        textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            exec python3 - "{clone}" "{telemetry_jsonl}" <<'PY'
            import json, os, sys
            from pathlib import Path
            import numpy as np

            broker = Path(sys.argv[1]).resolve()
            out_log = Path(sys.argv[2])
            nb_dir = Path(os.environ["NUMBA_CACHE_DIR"]).resolve()
            is_valid_nb_dir = nb_dir.is_dir() and not nb_dir.is_symlink()
            empty_before = is_valid_nb_dir and list(nb_dir.iterdir()) == []

            sys.path.insert(0, str(broker))
            from src.simd_env import ego_raster_nb

            owner_w = np.full((1, 16, 4), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
            owner_w[0, 4, 1] = np.uint64(0xFFFFFFFFFFFF0000)
            food_w = np.zeros((1, 16, 2), dtype=np.uint64)
            env_needed = np.array([True], dtype=np.bool_)
            counts = np.zeros((1, 2, 4, 4), dtype=np.int32)
            total = np.zeros((1, 4, 4), dtype=np.int32)
            fcounts = np.zeros((1, 4, 4), dtype=np.int32)
            ego_raster_nb._coarse_words_kernel(
                owner_w, food_w, env_needed, 2, 8, 8, 2, counts, total, fcounts
            )
            nbis = [p.name for p in nb_dir.rglob("*.nbi")]
            nbcs = [p.name for p in nb_dir.rglob("*.nbc")]
            record = {{
                "dont_write_bytecode_flag": int(sys.flags.dont_write_bytecode),
                "sys_dont_write_bytecode": bool(sys.dont_write_bytecode),
                "env_pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
                "numba_cache_dir": str(nb_dir),
                "numba_locator": os.environ.get("NUMBA_CACHE_LOCATOR_CLASSES"),
                "empty_before_use": empty_before,
                "nbi_files": nbis,
                "nbc_files": nbcs,
                "kernel_total": int(total.sum()),
            }}
            with out_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\\n")
            PY
            """),
        encoding="utf-8",
    )
    fake_dispatch.chmod(0o755)

    mf = tmp_path / "fresh-cache-mf.json"
    mf.write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-fresh-cache-preflight",
                "commit": BASELINE_COMMIT,
                "output_dir": str(tmp_path / "fresh-out"),
            }
        ),
        encoding="utf-8",
    )

    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        rc1 = dispatch.run_dispatch(
            [
                "watch6-v3-fresh-cache-preflight",
                str(mf),
                "--broker",
                str(clone),
                "--cwd",
                str(clone),
                "--operator-root",
                str(op_root),
                "--dry-run",
            ],
            ops_dir=tmp_path / "ops",
        )
        rc2 = dispatch.run_dispatch(
            [
                "watch6-v3-fresh-cache-preflight",
                str(mf),
                "--broker",
                str(clone),
                "--cwd",
                str(clone),
                "--operator-root",
                str(op_root),
                "--dry-run",
            ],
            ops_dir=tmp_path / "ops",
        )
    assert rc1 == 0 and rc2 == 0
    records = [
        json.loads(line)
        for line in telemetry_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    for rec in records:
        nb_path = Path(rec["numba_cache_dir"])
        assert rec["dont_write_bytecode_flag"] == 1
        assert rec["sys_dont_write_bytecode"] is True
        assert rec["env_pythondontwritebytecode"] == "1"
        assert rec["numba_locator"] == "UserProvidedCacheLocator"
        assert rec["empty_before_use"] is True
        assert len(rec["nbi_files"]) >= 1 and len(rec["nbc_files"]) >= 1
        assert rec["kernel_total"] >= 1
        assert not nb_path.is_relative_to(clone.resolve())
        assert not nb_path.is_relative_to(V3_BROKER_ROOT.resolve())
        assert not nb_path.is_relative_to(ADAPTER_ROOT.resolve())
        assert not nb_path.exists()
    assert records[0]["numba_cache_dir"] != records[1]["numba_cache_dir"]
    assert _snapshot_tree_state(clone) == before_clone_state

    # 2. Every adapter entrypoint creates a distinct, verified-empty, non-reused NUMBA_CACHE_DIR:
    import tools.watch6_v3_accounting as acct_pkg

    observed_entrypoint_dirs: list[str] = []
    real_vctc = acct_pkg._verify_clean_tracked_closure

    def _spy_isolation(*args: object, **kwargs: object) -> None:
        iso = active_broker_isolation()
        assert iso is not None
        assert iso.verified_empty_before_use is True
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        assert sys.dont_write_bytecode is True
        assert os.environ.get("NUMBA_CACHE_DIR") == str(iso.numba_cache_dir)
        assert os.environ.get("NUMBA_CACHE_LOCATOR_CLASSES") == "UserProvidedCacheLocator"
        assert not iso.numba_cache_dir.is_relative_to(clone.resolve())
        observed_entrypoint_dirs.append(str(iso.numba_cache_dir))
        return real_vctc(*args, **kwargs)  # type: ignore[arg-type]

    amend_clone = _write_immutable_text(
        tmp_path / "WATCH6_V3_ACCOUNTING_AMENDMENT_V5_20261031.md",
        _render_amendment_markdown(ADAPTER_ROOT),
    )

    seen_before = set(used_numba_cache_dirs())
    with patch.object(acct_pkg, "_verify_clean_tracked_closure", side_effect=_spy_isolation):
        # ratify:
        ratify.verify_amendment_document(
            amend_clone,
            broker_root=clone,
            require_clean_git=False,
        )
        # resume:
        assert resume.build([], [], broker_root=clone)["units"] == []
        # admission (lineage + admit):
        admission.verify_prior_receipts_lineage(0, [], 0, 0, 1, broker_root=clone)
        with pytest.raises(ValueError, match="Watch6 V3 runtime commit"):
            admission.admit({"commit": "bad"}, broker_root=clone)
        # prepare:
        with pytest.raises(ValueError, match="main preparation requires"):
            prepare.prepare(
                tmp_path / "prep-fresh",
                tmp_path / "b.json",
                BASELINE_COMMIT,
                tmp_path / "m.json",
                tmp_path / "c.json",
                broker_root=clone,
            )
        # recover:
        with pytest.raises(ValueError, match="recovery requires at least one"):
            recover.recover(
                base_package=tmp_path,
                receipts=[],
                target_shards=[0],
                out=tmp_path / "rec-fresh",
                benchmark=tmp_path / "b.json",
                mirror_receipt=tmp_path / "m.json",
                empirical_costs=tmp_path / "c.json",
                commit=BASELINE_COMMIT,
                broker_root=clone,
            )
        # final_look:
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE"):
            final_look.run_final_look(
                ops_dir=tmp_path / "ops",
                out_dir=tmp_path / "look-fresh",
                broker_root=clone,
                require_clean_git=False,
            )

    newly_allocated = set(used_numba_cache_dirs()) - seen_before
    assert len(newly_allocated) == 7
    assert set(observed_entrypoint_dirs) == newly_allocated
    for alloc_dir in newly_allocated:
        assert not Path(alloc_dir).exists()
        assert not Path(alloc_dir).is_relative_to(clone.resolve())

    # 3. Real CLI invocation WITHOUT -B and WITH poisoned caller env re-execs under -B, succeeds,
    #    and writes zero files/directories into clone:
    cli_env = dict(os.environ)
    cli_env["PYTHONPATH"] = str(ADAPTER_ROOT)
    cli_env["PYTHONDONTWRITEBYTECODE"] = "0"
    cli_env["NUMBA_CACHE_DIR"] = str(clone / "src/simd_env/__pycache__")
    cli_env.pop("_WATCH6_V3_ACCT_DASH_B_REEXEC", None)
    ledger_cli_out = tmp_path / "cli-empty-ledger.json"
    cli_proc = subprocess.run(
        [
            os.path.abspath(sys.executable),
            "-m",
            "tools.watch6_v3_accounting.resume",
            "--broker-root",
            str(clone),
            "--out",
            str(ledger_cli_out),
        ],
        cwd=clone,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert cli_proc.returncode == 0, cli_proc.stderr
    assert json.loads(ledger_cli_out.read_text(encoding="utf-8"))["units"] == []

    assert _snapshot_tree_state(clone) == before_clone_state
    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )


# ---------------------------------------------------------------------------
# V6 Shard-12 Renewal-WebSocket Zero-Export Transport Evidence Test Suite
# ---------------------------------------------------------------------------


def _build_truncated_renewal_ws_refusal(
    *,
    terminal_line: str = proof.EXACT_TERMINAL_WS_LINE,
    send_code: str = proof.EXACT_WS_SEND_CODE,
    raise_code: str = proof.EXACT_WS_RAISE_CODE,
    wrapper_prefix: str = proof.RENEWAL_WS_WRAPPER_PREFIX,
    truncation_len: int = proof.RENEWAL_WS_TRUNCATED_BODY_LEN,
    extra_box_lines: int = 0,
    pointer_on_raise: bool = True,
) -> str:
    """Simulate Rich COLUMNS=240 websocket send traceback sliced by supervisor [-2000:]."""
    width = 240

    def box_line(inner: str) -> str:
        padded = inner[: width - 2].ljust(width - 2)
        return f"│{padded}│"

    top_border = "╭" + ("─" * (width - 2)) + "╮"
    bottom_border = "╰" + ("─" * (width - 2)) + "╯"
    ptr_send = "   " if pointer_on_raise else " ❱ "
    ptr_raise = " ❱ " if pointer_on_raise else "   "

    box_rows = [
        top_border,
        *[
            box_line(f"   {100 + i} │   # transport context pad {i}")
            for i in range(12 + extra_box_lines)
        ],
        box_line("   290 │   def send_text(self, text_data: str) -> int:"),
        box_line(f"{ptr_send}292 │   │   {send_code}"),
        box_line(f"{ptr_raise}293 │   │   │   {raise_code}"),
        box_line("   294 │   │   return len(text_data)"),
        box_line(" "),
        bottom_border,
        terminal_line,
    ]
    full_cli_text = "\n".join(box_rows)
    assert len(full_cli_text) > truncation_len + 40
    sliced = full_cli_text[-truncation_len:]
    inner = f"{wrapper_prefix}{sliced}"
    return f"Refused({inner!r})"


def _write_renewal_ws_shard_fixture(
    base_dir: Path,
    *,
    ops_dir: Path,
    evidence_dir: Path,
    shard: int = 12,
    ops_name: str | None = None,
    poll_count: int = 4,
    prestart_retries_used: int = 0,
    compute_completion_attempt: int = 0,
    prior_receipts: list[dict] | None = None,
    completion_attempt: int | None = None,
    extra_box_lines: int = 0,
    publish_sidecar: bool = True,
    with_live_and_accounting: bool = True,
) -> tuple[Path, Path]:
    """Create a complete realistic zero-export renewal-WS attempt + immutable sidecar."""
    eff_ops_name = ops_name or f"watch6-v3-fix3-shard-{shard:02d}"
    (ops_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (ops_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ops_dir / "dispatch" / eff_ops_name).mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    out_dir = base_dir / f"output-shard-{shard:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    priors = list(prior_receipts or [])
    tot_dispatches = 1 + prestart_retries_used + compute_completion_attempt
    job_id = f"watch6-v3-shard-{shard:02d}-{base_dir.name}"[:60]
    generation = f"{shard:02x}" + "9f0b1c2d3e4f5a6b7c8d9e0f1a2b3c"
    session = f"snake-{job_id}-{generation[:12]}"
    endpoint = f"m-s-l4shard{shard:02d}ep"

    source_manifest: dict[str, object] = {
        "job_id": job_id,
        "output_dir": str(out_dir),
        "commit": BASELINE_COMMIT,
        "script": p.SCRIPT,
        "source_paths": list(_cached_broker_source_paths()),
        "device": "cpu",
        "timeout_minutes": 234,
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": 117,
        },
        "budget": {
            "unit_cap": 9.75,
            "rate_units_per_hour": 2.5,
            "reserve_units": 1.95,
            "other_assignments": {},
        },
        "env": {
            "WATCH6_FIXTURE": "0",
            "WATCH6_SHARD": str(shard),
            "WATCH6_SHARDS": str(p.SHARDS),
            "WATCH6_WORKERS": str(p.WORKERS),
            "WATCH6_SCIENCE": BASELINE_SCIENCE,
            "SNAKE_WATCH_SOURCE": "live",
        },
        "watch6_v3": {
            "namespace": p.NAMESPACE,
            "science": BASELINE_SCIENCE,
            "primary_allocation": list(_cached_broker_allocation()),
            "shard": shard,
            "phase": "primary",
            "prestart_retries_used": prestart_retries_used,
            "compute_completion_attempt": compute_completion_attempt,
            "total_dispatches_for_shard": tot_dispatches,
            "prior_receipts": priors,
            "base_minutes": 117,
            "base_unit_cap": 4.875,
            "base_reserve_units": 0.975,
        },
    }
    if completion_attempt is not None:
        source_manifest["completion_attempt"] = completion_attempt

    other_assignments = {"m-s-peer01": {"rate_units_per_hour": 1.94, "session": "peer-01"}}
    exec_manifest = json.loads(json.dumps(source_manifest))
    exec_manifest["budget"]["other_assignments"] = other_assignments

    src_mf_path = ops_dir / "manifests" / f"{eff_ops_name}.json"
    src_mf_path.write_text(json.dumps(source_manifest, sort_keys=True) + "\n", encoding="utf-8")
    exec_mf_path = out_dir / "manifest.json"
    exec_mf_path.write_text(json.dumps(exec_manifest, sort_keys=True) + "\n", encoding="utf-8")

    if with_live_and_accounting:
        live_mf_path = ops_dir / "manifests" / f"{eff_ops_name}.live.json"
        live_mf_path.write_text(json.dumps(exec_manifest, sort_keys=True) + "\n", encoding="utf-8")
        acct_rec_path = ops_dir / "dispatch" / eff_ops_name / f"{eff_ops_name}.accounting.json"
        acct_rec_path.write_text(
            json.dumps(
                {
                    "source_manifest": str(src_mf_path),
                    "source_manifest_sha256": digest(src_mf_path),
                    "manifest": str(live_mf_path),
                    "manifest_sha256": digest(live_mf_path),
                    "other_assignments": other_assignments,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    (out_dir / "hardware.json").write_text(
        json.dumps({"ready": True, "hardware": {"kind": "L4", "vcpu": 12}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "warmup-measurement.json").write_text(
        '{"status": "PASS", "warmup_seconds": 14.2}\n', encoding="utf-8"
    )
    (out_dir / "rate-calibration.json").write_text(
        '{"status": "PASS", "observed_rate_units_per_hour": 1.94}\n', encoding="utf-8"
    )
    (out_dir / "upload-measurement.json").write_text(
        '{"elapsed_seconds": 48.5, "staging_completed": true}\n', encoding="utf-8"
    )
    (out_dir / "source-delivery.json").write_text(
        json.dumps({"transport": "github", "commit": BASELINE_COMMIT}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "input-hashes.json").write_text(
        json.dumps({"inputs/verified-ledger.json": "a" * 64}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "spend-000000.json").write_text(
        '{"unit_cap_status": "OK", "provider_transition_mismatch_polls": 0}\n',
        encoding="utf-8",
    )
    (out_dir / "usage-000000.json").write_text(
        '{"active_assignments": 2, "balance": 880.5}\n',
        encoding="utf-8",
    )
    (out_dir / "exports").mkdir(exist_ok=True)

    poll_log_lines: list[str] = []
    t0 = 1793480000.0
    for idx in range(poll_count):
        p_status = "starting" if idx == 0 else "running"
        tail_msg = f"WATCH6_V3_START shard={shard} poll={idx}\n"
        (out_dir / f"poll-{idx:06d}.json").write_text(
            json.dumps(
                {
                    "exports": [],
                    "terminal": None,
                    "job_log_tail": tail_msg,
                    "supervisor_log_tail": "",
                    "state": {
                        "status": p_status,
                        "job_id": job_id,
                        "generation": generation,
                        "returncode": None,
                        "reason": None,
                        "error": None,
                        "unassign_error": None,
                        "started_at": t0,
                        "updated_at": t0 + 120.0 * idx,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        poll_log_lines.append(
            json.dumps(
                {
                    "job": job_id,
                    "status": p_status,
                    "exports": 0,
                    "terminal": False,
                    "job_log_tail": tail_msg,
                },
                sort_keys=True,
            )
        )

    refusal_error = _build_truncated_renewal_ws_refusal(extra_box_lines=extra_box_lines)
    receipt = {
        "job_id": job_id,
        "commit": BASELINE_COMMIT,
        "manifest_sha256": digest(exec_mf_path),
        "supervisor_sha256": PINNED_SUPERVISOR_SHA256,
        "generation": generation,
        "session": session,
        "endpoint": endpoint,
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": refusal_error,
        "traceback": None,
        "last_traceback_line": None,
        "stop_verified": True,
        "cleanup_error": None,
        "terminal": None,
        "exports": [],
        "transfers": [],
        "contents_transfers": [
            {"direction": "upload", "dest": "inputs/verified-ledger.json", "bytes": 128}
        ],
        "accounting": {
            "allocated_endpoint": endpoint,
            "unaccounted_endpoints": [],
            "other_assignments": other_assignments,
            "rate_units_per_hour": 2.5,
            "reserve_units": 1.95,
            "unit_cap": 9.75,
        },
        "control_operations": [
            {
                "operation": "poll",
                "thread": "MainThread",
                "status": "completed",
                "started_monotonic": 500.0,
                "ended_monotonic": 503.0,
            },
            {
                "operation": "renewal",
                "thread": "MainThread",
                "status": "failed",
                "started_monotonic": 612.0,
                "ended_monotonic": 620.5,
            },
        ],
        "cli_calls": [
            {
                "command": "new",
                "operation": "new",
                "thread": "MainThread",
                "attempt": 1,
                "status": "completed",
                "returncode": 0,
                "started_monotonic": 10.0,
                "ended_monotonic": 25.0,
                "elapsed_seconds": 15.0,
                "timeout_seconds": 180.0,
            },
            {
                "command": "exec",
                "operation": "poll",
                "thread": "MainThread",
                "attempt": 1,
                "status": "completed",
                "returncode": 0,
                "started_monotonic": 500.2,
                "ended_monotonic": 502.8,
                "elapsed_seconds": 2.6,
                "timeout_seconds": 90.0,
            },
            {
                "command": "exec",
                "operation": "renewal",
                "thread": "MainThread",
                "attempt": 1,
                "status": "failed",
                "returncode": 1,
                "started_monotonic": 612.4,
                "ended_monotonic": 620.1,
                "elapsed_seconds": 7.7,
                "timeout_seconds": 90.0,
            },
            {
                "command": "stop",
                "operation": "control",
                "thread": "MainThread",
                "attempt": 1,
                "status": "completed",
                "returncode": 0,
                "started_monotonic": 620.8,
                "ended_monotonic": 625.0,
                "elapsed_seconds": 4.2,
                "timeout_seconds": 120.0,
            },
            {
                "command": "sessions",
                "operation": "control",
                "thread": "MainThread",
                "attempt": 1,
                "status": "completed",
                "returncode": 0,
                "started_monotonic": 625.1,
                "ended_monotonic": 627.4,
                "elapsed_seconds": 2.3,
                "timeout_seconds": 60.0,
            },
        ],
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    calib_path = (out_dir / "rate-calibration.json").resolve()
    oplog_lines = [
        f"OPS_ALLOCATION_SUCCESS calibration={calib_path}",
        *poll_log_lines[: max(1, poll_count)],
        f"INCOMPLETE: {refusal_error}; receipt: {receipt_path.resolve()}",
        "EXIT=1",
    ]
    (ops_dir / "logs" / f"{eff_ops_name}.log").write_text(
        "\n".join(oplog_lines) + "\n",
        encoding="utf-8",
    )

    sidecar_path = evidence_dir / f"transport-evidence-{eff_ops_name}-01.json"
    if publish_sidecar:
        proof.publish_renewal_ws_transport_evidence(
            out_dir,
            sidecar_path,
            ops_dir=ops_dir,
            ops_name=eff_ops_name,
        )
    return receipt_path, sidecar_path


def test_live_shard12_evidence_verified_readonly(tmp_path: Path) -> None:
    """Ruling Points 2, 3, 4: real shard-12 evidence passes with immutable sidecar & zero writes."""
    art = artifacts_root(ADAPTER_ROOT)
    live_receipt = art / SHARD12_RECEIPT_REL
    ops_dir = art / "ops"
    live_ops_mf = ops_dir / "manifests/watch6-v3-fix3-shard-12.json"
    live_live_mf = ops_dir / "manifests/watch6-v3-fix3-shard-12.live.json"
    live_oplog = ops_dir / "logs/watch6-v3-fix3-shard-12.log"
    if not (live_receipt.is_file() and live_ops_mf.is_file() and live_oplog.is_file()):
        pytest.skip("live shard-12 artifacts not mounted on this host")

    out12 = live_receipt.parent
    watched_files = [
        live_receipt,
        out12 / "manifest.json",
        out12 / "hardware.json",
        *sorted(out12.glob("poll-*.json")),
        live_ops_mf,
        live_live_mf,
        live_oplog,
    ]
    snapshots_before = {
        p_item: (digest(p_item), p_item.stat().st_mtime_ns, p_item.stat().st_size)
        for p_item in watched_files
    }
    assert snapshots_before[live_receipt][0] == SHARD12_RECEIPT_SHA256
    assert snapshots_before[out12 / "manifest.json"][0] == SHARD12_EXEC_MANIFEST_SHA256
    assert snapshots_before[live_ops_mf][0] == SHARD12_OPS_MANIFEST_SHA256
    assert snapshots_before[live_live_mf][0] == SHARD12_LIVE_MANIFEST_SHA256
    assert snapshots_before[live_oplog][0] == SHARD12_OPS_LOG_SHA256

    rec12 = json.loads(live_receipt.read_text(encoding="utf-8"))
    # Baseline V3 refuses both prestart and compute; adapter also refuses prestart and refuses
    # compute WITHOUT an immutable verified sidecar:
    assert v3_admission.is_prestart_transport_failure(rec12, out12) is False
    assert admission.is_prestart_transport_failure(rec12, out12, ops_dir=ops_dir) is False
    assert v3_admission.is_external_transport_compute_failure(rec12, out12) is False
    empty_ev = tmp_path / "empty-ev"
    empty_ev.mkdir()
    assert (
        admission.is_external_transport_compute_failure(
            rec12,
            out12,
            ops_dir=ops_dir,
            evidence_dir=empty_ev,
        )
        is False
    )

    # Publish immutable sidecar to tmp_path and verify full conjunction:
    ev_dir = tmp_path / "ev"
    ev_dir.mkdir()
    sidecar_tmp = ev_dir / "transport-evidence-watch6-v3-fix3-shard-12-01.json"
    summary = proof.publish_renewal_ws_transport_evidence(out12, sidecar_tmp, ops_dir=ops_dir)
    assert summary["sidecar_sha256"] == SHARD12_EVIDENCE_SIDECAR_SHA256
    assert summary["receipt_sha256"] == SHARD12_RECEIPT_SHA256
    assert summary["executed_manifest_sha256"] == SHARD12_EXEC_MANIFEST_SHA256
    assert summary["supervisor_sha256"] == PINNED_SUPERVISOR_SHA256
    assert summary["polls_count"] == 4
    assert summary["shard"] == 12
    assert (
        admission.is_external_transport_compute_failure(
            rec12,
            out12,
            ops_dir=ops_dir,
            evidence_dir=ev_dir,
        )
        is True
    )
    _, bound_rec12 = resume.verify_bound_receipt(live_receipt, ops_dir=ops_dir)
    assert bound_rec12["stop_verified"] is True
    assert bound_rec12["cleanup_error"] is None

    # If the canonical V6 sidecar is already published in argon-v3-accounting-v6, verify automatic
    # default discovery without passing evidence_dir:
    published_sidecar = art / SHARD12_EVIDENCE_SIDECAR_REL
    if published_sidecar.is_file():
        auto_res = proof.verify_renewal_ws_transport_evidence(out12, ops_dir=ops_dir)
        assert auto_res["sidecar_sha256"] == SHARD12_EVIDENCE_SIDECAR_SHA256
        assert (
            admission.is_external_transport_compute_failure(rec12, out12, ops_dir=ops_dir) is True
        )

    # Virtual in-memory fault injections on real shard 12 MUST fail closed without touching disk:
    orig_exists = Path.exists
    orig_is_file = Path.is_file
    orig_read_text = Path.read_text
    for virtual_fault in ("scientific-fault.json", "runtime-measurement.json", "terminal.json"):
        v_target = out12 / virtual_fault
        with (
            patch.object(
                Path, "exists", lambda self, t=v_target: True if self == t else orig_exists(self)
            ),
            patch.object(
                Path, "is_file", lambda self, t=v_target: True if self == t else orig_is_file(self)
            ),
            patch.object(
                Path,
                "read_text",
                lambda self, *a, t=v_target, **kw: (
                    '{"error":"fault"}\n' if self == t else orig_read_text(self, *a, **kw)
                ),
            ),
        ):
            assert (
                admission.is_external_transport_compute_failure(
                    rec12,
                    out12,
                    ops_dir=ops_dir,
                    evidence_dir=ev_dir,
                )
                is False
            )

    for p_item, before_tuple in snapshots_before.items():
        st = p_item.stat()
        assert (digest(p_item), st.st_mtime_ns, st.st_size) == before_tuple


@pytest.mark.parametrize(("shard", "extra_lines"), [(5, 0), (12, 3), (19, 8)])
def test_renewal_ws_synthetic_positive_and_prospective_shards(
    tmp_path: Path,
    shard: int,
    extra_lines: int,
) -> None:
    """Prospective application across shards 0..23 + lineage + recover (Point 5 & Point 6)."""
    ops_dir = tmp_path / "ops"
    ev_dir = tmp_path / "evidence"
    r_path, sidecar_path = _write_renewal_ws_shard_fixture(
        tmp_path / "pkg",
        ops_dir=ops_dir,
        evidence_dir=ev_dir,
        shard=shard,
        extra_box_lines=extra_lines,
    )
    rec = json.loads(r_path.read_text(encoding="utf-8"))
    assert sidecar_path.is_file()
    assert admission.is_prestart_transport_failure(rec, r_path.parent, ops_dir=ops_dir) is False
    assert (
        admission.is_external_transport_compute_failure(
            rec,
            r_path.parent,
            ops_dir=ops_dir,
            evidence_dir=ev_dir,
        )
        is True
    )
    admission.verify_prior_receipts_lineage(
        shard,
        [
            {
                "receipt_path": str(r_path),
                "receipt_sha256": digest(r_path),
                "classification": "external_transport_compute",
            }
        ],
        0,
        1,
        2,
        ops_dir=ops_dir,
        evidence_dir=ev_dir,
    )

    # Verify recover.recover produces exact Ruling Point 5 counters (prestart=0, compute=1, tot=2):
    base_pkg = tmp_path / "base-pkg"
    base_pkg.mkdir()
    with bind_broker_protocol(V3_BROKER_ROOT):
        publish_json(
            base_pkg / "verified-ledger.json",
            empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity()),
        )
        exec_id = mirror.execution_identity()
    dummy_in = base_pkg / "input-metadata.tar.gz"
    dummy_in.write_bytes(b"meta")
    (base_pkg / "canary.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "path": str(dummy_in),
                        "dest": "input-metadata.tar.gz",
                        "sha256": digest(dummy_in),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (base_pkg / "output-canary").mkdir()
    (base_pkg / "output-canary/receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "cleanup_error": None,
                "commit": BASELINE_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    bench = tmp_path / "benchmark.json"
    bench.write_text(json.dumps({"science": BASELINE_SCIENCE}), encoding="utf-8")
    mir = tmp_path / "mac-mirror.json"
    mir.write_text(
        json.dumps(
            {
                "schema": "watch6-full-mirror/v3",
                "status": "PASS",
                "platform": "Darwin",
                "execution_identity": exec_id,
                "seed": p.FIXTURE_SEEDS[0],
                "horizon": p.HORIZON,
                "mirror_identity_frames": p.HORIZON,
                "mirror_mismatches": 0,
                "respawns_per_slot": [3, 3, 3, 3, 3, 3],
            }
        ),
        encoding="utf-8",
    )
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps(_synthetic_cost_report()), encoding="utf-8")

    def fake_prepare(out: Path, *args: object, **kwargs: object) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        return {"shards": 24, "workers": 4, "base_minutes_per_shard": 117}

    with patch.object(recover, "prepare", side_effect=fake_prepare):
        plan = recover.recover(
            base_package=base_pkg,
            receipts=[r_path],
            target_shards=[shard],
            out=tmp_path / f"recovery-shard-{shard:02d}-01",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=costs,
            commit=BASELINE_COMMIT,
            ops_dir=ops_dir,
            evidence_dir=ev_dir,
        )
    s_key = str(shard)
    assert plan["verified_units_in_ledger"] == 0
    assert plan["reused_initial_canary"] is True
    assert plan["shard_meta"][s_key]["prestart_retries_used"] == 0
    assert plan["shard_meta"][s_key]["compute_completion_attempt"] == 1
    assert plan["shard_meta"][s_key]["total_dispatches_for_shard"] == 2
    assert len(plan["transport_evidence_proofs"]) == 1
    assert plan["transport_evidence_proofs"][0]["shard"] == shard


def _assert_renewal_ws_case_refused(
    out_dir: Path,
    ops_dir: Path,
    ev_dir: Path,
    *,
    ops_name: str = "watch6-v3-fix3-shard-12",
    republish: bool = True,
) -> None:
    sidecar = ev_dir / f"transport-evidence-{ops_name}-01.json"
    with isolated_broker_execution("assert-renewal-ws-refused"):
        if republish:
            for sfx in ("", ".sha256", ".complete"):
                Path(f"{sidecar}{sfx}").unlink(missing_ok=True)
            with pytest.raises(ValueError):
                proof.publish_renewal_ws_transport_evidence(
                    out_dir,
                    sidecar,
                    ops_dir=ops_dir,
                    ops_name=ops_name,
                )
        else:
            with pytest.raises(ValueError):
                proof.verify_renewal_ws_transport_evidence(
                    out_dir,
                    ops_dir=ops_dir,
                    ops_name=ops_name,
                    evidence_dir=ev_dir,
                )
        rec = json.loads((out_dir / "receipt.json").read_text(encoding="utf-8"))
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE"):
            final_look._classify_attempt(
                out_dir,
                rec,
                ops_dir=ops_dir,
                ops_name=ops_name,
                evidence_dir=ev_dir,
            )


@isolated_broker_execution("test-renewal-ws-faults")
def test_renewal_ws_fault_injection_and_exclusion_vectors(tmp_path: Path) -> None:
    """Ruling Points 2, 4, 6: child/scientific/timeout vetoes, partial-export, general-WS."""

    def fresh_case(label: str) -> tuple[Path, Path, Path, Path]:
        c_dir = tmp_path / label
        ops = c_dir / "ops"
        ev = c_dir / "ev"
        r_p, s_p = _write_renewal_ws_shard_fixture(
            c_dir / "pkg",
            ops_dir=ops,
            evidence_dir=ev,
            shard=12,
            publish_sidecar=False,
        )
        return r_p.parent, ops, ev, s_p

    def update_receipt_and_oplog(out_dir: Path, ops_dir: Path, mutator) -> None:
        rp = out_dir / "receipt.json"
        rec = json.loads(rp.read_text(encoding="utf-8"))
        mutator(rec)
        rp.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
        oplog = ops_dir / "logs/watch6-v3-fix3-shard-12.log"
        lines = oplog.read_text(encoding="utf-8").splitlines()
        new_lines = [
            (
                f"INCOMPLETE: {rec.get('error')}; receipt: {rp.resolve()}"
                if ln.startswith("INCOMPLETE: ")
                else ln
            )
            for ln in lines
        ]
        oplog.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 1. Forbidden files (terminal.json, scientific-fault.json, runtime-measurement.json, logs):
    for idx, bad_file in enumerate(
        (
            "terminal.json",
            "scientific-fault.json",
            "runtime-measurement.json",
            "job.log",
            "supervisor.log",
            "recovery-179348-1.json",
            "telemetry-000000.json",
        )
    ):
        out_d, ops_d, ev_d, _ = fresh_case(f"forbid-file-{idx}")
        (out_d / bad_file).write_text('{"status": "bad"}\n', encoding="utf-8")
        _assert_renewal_ws_case_refused(out_d, ops_d, ev_d)

    # 2. Child / scientific / integrity / wall-timeout / kernel-error vetoes across files:
    for label, fault_text in (
        ("nan", "NaN detected in policy evaluation"),
        ("mirror", "ValueError: mirror mismatch at frame 128"),
        ("wall-timeout", "WALL_TIMEOUT: whole-session wall cap reached"),
        ("faulthandler", "faulthandler dump_traceback_later fired"),
        ("kernel-err", "kernel error despite CLI exit zero: child crashed"),
    ):
        # 2a. In poll job_log_tail:
        out_d, ops_d, ev_d, _ = fresh_case(f"poll-fault-{label}")
        p0 = out_d / "poll-000001.json"
        p0_obj = json.loads(p0.read_text(encoding="utf-8"))
        p0_obj["job_log_tail"] = fault_text
        p0.write_text(json.dumps(p0_obj, sort_keys=True) + "\n", encoding="utf-8")
        _assert_renewal_ws_case_refused(out_d, ops_d, ev_d)

        # 2b. In spend-000000.json diagnostic string:
        out_s, ops_s, ev_s, _ = fresh_case(f"spend-fault-{label}")
        sp0 = out_s / "spend-000000.json"
        sp_obj = json.loads(sp0.read_text(encoding="utf-8"))
        sp_obj["note"] = fault_text
        sp0.write_text(json.dumps(sp_obj, sort_keys=True) + "\n", encoding="utf-8")
        _assert_renewal_ws_case_refused(out_s, ops_s, ev_s)

    # 2c. spend unit_cap_status != OK:
    out_cap, ops_cap, ev_cap, _ = fresh_case("spend-cap-exceeded")
    (out_cap / "spend-000000.json").write_text(
        '{"unit_cap_status": "EXCEEDED"}\n', encoding="utf-8"
    )
    _assert_renewal_ws_case_refused(out_cap, ops_cap, ev_cap)

    # 2d. queue-*.log mentioning non-transport fault for this job:
    out_q, ops_q, ev_q, _ = fresh_case("queue-log-fault")
    q_job = json.loads((out_q / "receipt.json").read_text(encoding="utf-8"))["job_id"]
    (ops_q / "logs/queue-20261031.log").write_text(
        f"{q_job}: ValueError: mirror mismatch\n",
        encoding="utf-8",
    )
    _assert_renewal_ws_case_refused(out_q, ops_q, ev_q)

    # 2e. Contradictory state.json / events.jsonl:
    out_st, ops_st, ev_st, _ = fresh_case("bad-state-json")
    (out_st / "state.json").write_text('{"terminal": {"returncode": 1}}\n', encoding="utf-8")
    _assert_renewal_ws_case_refused(out_st, ops_st, ev_st)

    out_evj, ops_evj, ev_evj, _ = fresh_case("bad-events-jsonl")
    (out_evj / "events.jsonl").write_text(
        '{"event": "poll", "note": "WALL_TIMEOUT"}\n', encoding="utf-8"
    )
    _assert_renewal_ws_case_refused(out_evj, ops_evj, ev_evj)

    # 3. Partial-export exclusion (Ruling Point 6):
    # 3a. Non-empty exports/ directory on disk:
    out_pe1, ops_pe1, ev_pe1, _ = fresh_case("partial-export-dir")
    (out_pe1 / "exports/efficacy").mkdir(parents=True, exist_ok=True)
    (out_pe1 / "exports/efficacy/pair-12.json").write_text('{"partial": true}\n', encoding="utf-8")
    _assert_renewal_ws_case_refused(out_pe1, ops_pe1, ev_pe1)

    # 3b. Non-empty receipt.exports:
    out_pe2, ops_pe2, ev_pe2, _ = fresh_case("partial-export-receipt")
    update_receipt_and_oplog(
        out_pe2,
        ops_pe2,
        lambda r: r.update(
            exports=[{"path": "efficacy/pair-12.json", "sha256": "a" * 64, "size": 2}]
        ),
    )
    _assert_renewal_ws_case_refused(out_pe2, ops_pe2, ev_pe2)

    # 3c. Non-empty poll.exports:
    out_pe3, ops_pe3, ev_pe3, _ = fresh_case("partial-export-poll")
    p_pe = out_pe3 / "poll-000002.json"
    p_pe_obj = json.loads(p_pe.read_text(encoding="utf-8"))
    p_pe_obj["exports"] = ["efficacy/pair-12.json"]
    p_pe.write_text(json.dumps(p_pe_obj, sort_keys=True) + "\n", encoding="utf-8")
    _assert_renewal_ws_case_refused(out_pe3, ops_pe3, ev_pe3)

    # 3d. Nonzero exports count in operational log telemetry line:
    out_pe4, ops_pe4, ev_pe4, _ = fresh_case("partial-export-oplog")
    oplog_pe = ops_pe4 / "logs/watch6-v3-fix3-shard-12.log"
    oplog_pe.write_text(
        oplog_pe.read_text(encoding="utf-8").replace('"exports": 0', '"exports": 1', 1),
        encoding="utf-8",
    )
    _assert_renewal_ws_case_refused(out_pe4, ops_pe4, ev_pe4)

    # 3e. Download transfer recorded in receipt.transfers:
    out_pe5, ops_pe5, ev_pe5, _ = fresh_case("partial-export-download-tx")
    update_receipt_and_oplog(
        out_pe5,
        ops_pe5,
        lambda r: r.update(transfers=[{"direction": "download", "remote": "out.json"}]),
    )
    _assert_renewal_ws_case_refused(out_pe5, ops_pe5, ev_pe5)

    # 4. General-WebSocket & signature/conjunction exclusions (Ruling Points 2 & 6):
    for ws_idx, bad_error in enumerate(
        (
            # Other WebSocket exception classes:
            _build_truncated_renewal_ws_refusal(
                terminal_line="WebSocketTimeoutException: Connection timed out."
            ),
            _build_truncated_renewal_ws_refusal(
                terminal_line="WebSocketBadStatusException: Handshake status 502 Bad Gateway"
            ),
            _build_truncated_renewal_ws_refusal(
                terminal_line="WebSocketProtocolException: Invalid close frame."
            ),
            # Different WebSocketConnectionClosedException message:
            _build_truncated_renewal_ws_refusal(
                terminal_line="WebSocketConnectionClosedException: socket is already closed."
            ),
            # Exception-substring-only without Rich box / 2,000-char truncation:
            (
                "Refused('colab exec failed (1): "
                "WebSocketConnectionClosedException: Connection is already closed.')"
            ),
            # Untruncated (< 2000 chars) Rich box:
            _build_truncated_renewal_ws_refusal(truncation_len=1999),
            # Overlong (> 2000 chars) Rich box:
            _build_truncated_renewal_ws_refusal(truncation_len=2001),
            # Wrong wrapper prefix:
            _build_truncated_renewal_ws_refusal(wrapper_prefix="colab upload failed (1): "),
            _build_truncated_renewal_ws_refusal(wrapper_prefix="colab exec failed (2): "),
            # Missing websocket send guard line:
            _build_truncated_renewal_ws_refusal(send_code="if self.sock is None:"),
            # Active pointer on wrong line:
            _build_truncated_renewal_ws_refusal(pointer_on_raise=False),
            # Kernel error prefix smuggling WebSocket text:
            (
                "Refused('kernel error despite CLI exit zero: "
                "WebSocketConnectionClosedException: Connection is already closed.')"
            ),
        )
    ):
        out_ws, ops_ws, ev_ws, _ = fresh_case(f"ws-sig-excl-{ws_idx}")
        update_receipt_and_oplog(out_ws, ops_ws, lambda r, e=bad_error: r.update(error=e))
        _assert_renewal_ws_case_refused(out_ws, ops_ws, ev_ws)

    # 5. Monotonic renewal interval, timeout, and stop/sessions conjunction exclusions:
    for conj_label, rec_mut in (
        ("terminal-nonnull", lambda r: r.update(terminal={"returncode": 1})),
        ("child-fcls", lambda r: r.update(failure_class="CHILD_NONZERO_EXIT")),
        ("unstopped", lambda r: r.update(stop_verified=False)),
        ("cleanup-err", lambda r: r.update(cleanup_error="stop failed")),
        ("ctrl-not-renewal", lambda r: r["control_operations"][-1].update(operation="poll")),
        ("cli-not-renewal", lambda r: r["cli_calls"][2].update(operation="poll")),
        ("cli-not-exec", lambda r: r["cli_calls"][2].update(command="upload")),
        (
            "cli-outside-ctrl-interval",
            lambda r: r["cli_calls"][2].update(started_monotonic=610.0),
        ),
        (
            "cli-timed-out",
            lambda r: r["cli_calls"][2].update(elapsed_seconds=90.0, timeout_seconds=90.0),
        ),
        ("missing-post-sessions", lambda r: r["cli_calls"].pop()),
        (
            "stop-before-ctrl-end",
            lambda r: r["cli_calls"][3].update(started_monotonic=619.0),
        ),
    ):
        out_cj, ops_cj, ev_cj, _ = fresh_case(f"conj-{conj_label}")
        update_receipt_and_oplog(out_cj, ops_cj, rec_mut)
        _assert_renewal_ws_case_refused(out_cj, ops_cj, ev_cj)


@isolated_broker_execution("test-renewal-ws-tamper")
def test_renewal_ws_missing_evidence_tamper_and_ratify_guards(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Ruling Points 1, 3, 4: missing evidence, 1-byte tamper, foreign/ambiguous sidecar, ratify."""
    # 1. Missing sidecar / missing .sha256 / missing .complete:
    c_miss = tmp_path / "missing-sidecar"
    r_miss, s_miss = _write_renewal_ws_shard_fixture(
        c_miss / "pkg",
        ops_dir=c_miss / "ops",
        evidence_dir=c_miss / "ev",
        shard=12,
        publish_sidecar=True,
    )
    Path(f"{s_miss}.complete").unlink()
    _assert_renewal_ws_case_refused(r_miss.parent, c_miss / "ops", c_miss / "ev", republish=False)
    s_miss.unlink()
    Path(f"{s_miss}.sha256").unlink()
    _assert_renewal_ws_case_refused(r_miss.parent, c_miss / "ops", c_miss / "ev", republish=False)

    # 2. Missing polls / poll sequence gap / missing hardware / missing ops manifest / missing log:
    for miss_label, remover in (
        ("zero-polls", lambda out_d, _ops_d: [p.unlink() for p in out_d.glob("poll-*.json")]),
        ("poll-gap", lambda out_d, _ops_d: (out_d / "poll-000001.json").unlink()),
        ("no-hardware", lambda out_d, _ops_d: (out_d / "hardware.json").unlink()),
        ("no-warmup", lambda out_d, _ops_d: (out_d / "warmup-measurement.json").unlink()),
        (
            "no-ops-mf",
            lambda _out_d, ops_d: (ops_d / "manifests/watch6-v3-fix3-shard-12.json").unlink(),
        ),
        (
            "no-oplog",
            lambda _out_d, ops_d: (ops_d / "logs/watch6-v3-fix3-shard-12.log").unlink(),
        ),
    ):
        c_m = tmp_path / f"miss-{miss_label}"
        r_m, _ = _write_renewal_ws_shard_fixture(
            c_m / "pkg",
            ops_dir=c_m / "ops",
            evidence_dir=c_m / "ev",
            shard=12,
            publish_sidecar=True,
        )
        remover(r_m.parent, c_m / "ops")
        _assert_renewal_ws_case_refused(r_m.parent, c_m / "ops", c_m / "ev", republish=False)

    # 3. Post-publication 1-byte / field tamper in every bound artifact & extra unlisted poll:
    for tamp_label, tamper_fn in (
        (
            "receipt",
            lambda out_d, _ops_d: (out_d / "receipt.json").write_text(
                (out_d / "receipt.json").read_text(encoding="utf-8") + "\n", encoding="utf-8"
            ),
        ),
        (
            "exec-mf",
            lambda out_d, _ops_d: (out_d / "manifest.json").write_text(
                (out_d / "manifest.json").read_text(encoding="utf-8") + "\n", encoding="utf-8"
            ),
        ),
        (
            "ops-src-mf",
            lambda _out_d, ops_d: (ops_d / "manifests/watch6-v3-fix3-shard-12.json").write_text(
                (ops_d / "manifests/watch6-v3-fix3-shard-12.json").read_text(encoding="utf-8")
                + "\n",
                encoding="utf-8",
            ),
        ),
        (
            "ops-live-mf",
            lambda _out_d, ops_d: (
                ops_d / "manifests/watch6-v3-fix3-shard-12.live.json"
            ).write_text(
                (ops_d / "manifests/watch6-v3-fix3-shard-12.live.json").read_text(encoding="utf-8")
                + "\n",
                encoding="utf-8",
            ),
        ),
        (
            "ops-acct-rec",
            lambda _out_d, ops_d: (
                ops_d / "dispatch/watch6-v3-fix3-shard-12/watch6-v3-fix3-shard-12.accounting.json"
            ).write_text(
                (
                    ops_d
                    / "dispatch/watch6-v3-fix3-shard-12/watch6-v3-fix3-shard-12.accounting.json"
                ).read_text(encoding="utf-8")
                + "\n",
                encoding="utf-8",
            ),
        ),
        (
            "hardware",
            lambda out_d, _ops_d: (out_d / "hardware.json").write_text(
                '{"hardware": {"kind": "L4", "vcpu": 16}, "ready": true}\n', encoding="utf-8"
            ),
        ),
        (
            "poll-tamper",
            lambda out_d, _ops_d: (out_d / "poll-000002.json").write_text(
                (out_d / "poll-000002.json").read_text(encoding="utf-8") + "\n", encoding="utf-8"
            ),
        ),
        (
            "extra-unlisted-poll",
            lambda out_d, _ops_d: (out_d / "poll-000004.json").write_text(
                (out_d / "poll-000003.json").read_text(encoding="utf-8"), encoding="utf-8"
            ),
        ),
        (
            "oplog-tamper",
            lambda _out_d, ops_d: (ops_d / "logs/watch6-v3-fix3-shard-12.log").write_text(
                "\n" + (ops_d / "logs/watch6-v3-fix3-shard-12.log").read_text(encoding="utf-8"),
                encoding="utf-8",
            ),
        ),
    ):
        c_t = tmp_path / f"tamper-{tamp_label}"
        r_t, _ = _write_renewal_ws_shard_fixture(
            c_t / "pkg",
            ops_dir=c_t / "ops",
            evidence_dir=c_t / "ev",
            shard=12,
            publish_sidecar=True,
        )
        tamper_fn(r_t.parent, c_t / "ops")
        _assert_renewal_ws_case_refused(r_t.parent, c_t / "ops", c_t / "ev", republish=False)

    # 4. Tampered sidecar (even re-published with valid .sha256 + .complete) & duplicate:
    c_sc = tmp_path / "tamper-sidecar-republished"
    r_sc, s_sc = _write_renewal_ws_shard_fixture(
        c_sc / "pkg",
        ops_dir=c_sc / "ops",
        evidence_dir=c_sc / "ev",
        shard=12,
        publish_sidecar=True,
    )
    sc_payload = json.loads(s_sc.read_text(encoding="utf-8"))
    sc_payload["supervisor"]["supervisor_sha256"] = "f" * 64
    for sfx in ("", ".sha256", ".complete"):
        Path(f"{s_sc}{sfx}").unlink()
    publish_json(s_sc, sc_payload)
    _assert_renewal_ws_case_refused(r_sc.parent, c_sc / "ops", c_sc / "ev", republish=False)

    # Conflicting second sidecar in ops/dispatch/<run_name>/:
    c_dup = tmp_path / "ambiguous-sidecars"
    r_dup, s_dup = _write_renewal_ws_shard_fixture(
        c_dup / "pkg",
        ops_dir=c_dup / "ops",
        evidence_dir=c_dup / "ev",
        shard=12,
        publish_sidecar=True,
    )
    bad_copy = c_dup / "ops/dispatch/watch6-v3-fix3-shard-12/transport-evidence-conflict.json"
    dup_obj = json.loads(s_dup.read_text(encoding="utf-8"))
    dup_obj["identity"]["shard"] = 13
    publish_json(bad_copy, dup_obj)
    _assert_renewal_ws_case_refused(r_dup.parent, c_dup / "ops", c_dup / "ev", republish=False)

    # Modified supervisor.py in broker clone MUST fail closed in both proof and ratify:
    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-bad-sup")
    sup_file = clone / PINNED_SUPERVISOR_REL
    sup_file.write_text(sup_file.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    c_sup = tmp_path / "bad-broker-sup"
    r_sup, _ = _write_renewal_ws_shard_fixture(
        c_sup / "pkg",
        ops_dir=c_sup / "ops",
        evidence_dir=c_sup / "ev",
        shard=12,
        publish_sidecar=True,
    )
    with pytest.raises(ValueError):
        proof.verify_renewal_ws_transport_evidence(
            r_sup.parent,
            ops_dir=c_sup / "ops",
            evidence_dir=c_sup / "ev",
            broker_root=clone,
        )

    # 5. Ratification refuses live watch6-v3-shard12-ruling/RULING.md (APPROVE-PATH) and V5 ruling:
    amend_v6 = _write_immutable_text(
        tmp_path / "WATCH6_V3_ACCOUNTING_AMENDMENT_V6_20261101.md",
        _render_amendment_markdown(ADAPTER_ROOT),
    )
    art = artifacts_root(ADAPTER_ROOT)
    live_shard12_ruling = art / SHARD12_PATH_RULING_REL
    if live_shard12_ruling.is_file():
        with pytest.raises(ValueError, match="REFUSE"):
            ratify.record_ruling(
                live_shard12_ruling,
                amendment_md=amend_v6,
                out=tmp_path / "refused-shard12-path.json",
                require_clean_git=False,
            )
    old_v5_ruling_json = art / "t34-sol/argon-v3-accounting-fix4/astra-accounting-ruling.json"
    if old_v5_ruling_json.is_file():
        with pytest.raises(ValueError, match="REFUSE"):
            ratify.validate_ruling_receipt(old_v5_ruling_json, require_clean_git=False)


def test_v6_full_24_shard_lineage_retry_cap_and_zero_peek(tmp_path: Path) -> None:
    """Full 24-shard V6 registry: shard 00 prestart + shard 12 & shard 19 renewal-WS."""
    ops_dir, ruling_json, amend_path, rows = _setup_24_shard_accounting_ops(tmp_path / "v6-ok")
    ev_dir = tmp_path / "v6-ok/evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    # Replace shard 12 single-pass with A_1 renewal-WS failure + A_2 recovered PASS:
    shutil.rmtree(tmp_path / "v6-ok/out-shard-12")
    (ops_dir / "manifests/watch6-v3-fix3-shard-12.json").unlink()
    s12_a1, s12_sidecar = _write_renewal_ws_shard_fixture(
        tmp_path / "v6-ok/s12-a1",
        ops_dir=ops_dir,
        evidence_dir=ev_dir,
        shard=12,
        ops_name="watch6-v3-fix3-shard-12",
    )
    s12_priors = [
        {
            "receipt_path": str(s12_a1),
            "receipt_sha256": digest(s12_a1),
            "classification": "external_transport_compute",
        }
    ]
    s12_pass = _write_v3_attempt(
        tmp_path / "v6-ok/out-shard-12-recovered",
        shard=12,
        mode="pass",
        export_slots=p.shard_indices(12),
        pair_rows_by_slot=rows,
        prestart_retries_used=0,
        compute_completion_attempt=1,
        prior_receipts=s12_priors,
    )
    (ops_dir / "manifests/watch6-v3-fix3-shard-12-recovered.json").write_text(
        (s12_pass.parent / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Replace prospective shard 19 with TWO renewal-WS compute failures + A_3 recovered PASS:
    shutil.rmtree(tmp_path / "v6-ok/out-shard-19")
    (ops_dir / "manifests/watch6-v3-fix3-shard-19.json").unlink()
    s19_a1, _ = _write_renewal_ws_shard_fixture(
        tmp_path / "v6-ok/s19-a1",
        ops_dir=ops_dir,
        evidence_dir=ev_dir,
        shard=19,
        ops_name="watch6-v3-fix3-shard-19",
    )
    s19_p1 = [
        {
            "receipt_path": str(s19_a1),
            "receipt_sha256": digest(s19_a1),
            "classification": "external_transport_compute",
        }
    ]
    s19_a2, _ = _write_renewal_ws_shard_fixture(
        tmp_path / "v6-ok/s19-a2",
        ops_dir=ops_dir,
        evidence_dir=ev_dir,
        shard=19,
        ops_name="watch6-v3-fix3-shard-19-rec1",
        compute_completion_attempt=1,
        prior_receipts=s19_p1,
    )
    s19_p2 = [
        *s19_p1,
        {
            "receipt_path": str(s19_a2),
            "receipt_sha256": digest(s19_a2),
            "classification": "external_transport_compute",
        },
    ]
    s19_pass = _write_v3_attempt(
        tmp_path / "v6-ok/out-shard-19-rec2",
        shard=19,
        mode="pass",
        export_slots=p.shard_indices(19),
        pair_rows_by_slot=rows,
        prestart_retries_used=0,
        compute_completion_attempt=2,
        prior_receipts=s19_p2,
    )
    (ops_dir / "manifests/watch6-v3-fix3-shard-19-rec2.json").write_text(
        (s19_pass.parent / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    decision = final_look.run_final_look(
        ops_dir=ops_dir,
        out_dir=tmp_path / "v6-ok/look-out",
        ruling_json=ruling_json,
        amendment_md=amend_path,
        evidence_dir=ev_dir,
        require_clean_git=False,
    )
    assert decision["verdict"] == "GO_EFFICACY_ONLY"
    assert decision["n"] == 96
    assert len(decision["retained_never_allocated_attempts"]) == 1
    assert len(decision["retained_transport_evidence_attempts"]) == 3
    assert decision["total_main_attempts"] == 28  # 21*1 + 2 (s00) + 2 (s12) + 3 (s19)

    # 1. Tampering shard 12's sidecar MUST fail pre-look with ZERO pair-*.json payload reads:
    Path(f"{s12_sidecar}.complete").unlink()
    orig_read_text = Path.read_text

    def forbid_pair_payload_read(self: Path, *args: object, **kwargs: object) -> str:
        if resume.PAIR_NAME_RE.fullmatch(self.name) is not None:
            raise AssertionError(f"OUTCOME PEEKED BEFORE PRE-LOOK AUDIT: {self}")
        return orig_read_text(self, *args, **kwargs)

    with (
        patch.object(Path, "read_text", forbid_pair_payload_read),
        patch.object(final_look, "decide_v3", side_effect=AssertionError("OUTCOME PEEKED!")),
    ):
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE"):
            final_look.run_final_look(
                ops_dir=ops_dir,
                out_dir=tmp_path / "v6-ok/look-tampered",
                ruling_json=ruling_json,
                amendment_md=amend_path,
                evidence_dir=ev_dir,
                require_clean_git=False,
            )

    # Restore sidecar completion marker, then inject a 3rd failed renewal-WS attempt on shard 19
    # (exceeding MAX_COMPUTE_COMPLETIONS_PER_SHARD == 2) and verify pre-look refusal + zero peek:
    Path(f"{s12_sidecar}.complete").write_bytes(b"complete\n")
    shutil.rmtree(tmp_path / "v6-ok/out-shard-19-rec2")
    (ops_dir / "manifests/watch6-v3-fix3-shard-19-rec2.json").unlink()

    auto1_dir = tmp_path / "v6-ok/s19-auto1"
    s19_a3, _ = _write_renewal_ws_shard_fixture(
        auto1_dir,
        ops_dir=ops_dir,
        evidence_dir=ev_dir,
        shard=19,
        ops_name="watch6-v3-fix3-shard-19-auto1",
        compute_completion_attempt=1,
        prior_receipts=s19_p1,
        completion_attempt=1,
    )
    publish_json(
        auto1_dir / "manifest.json",
        json.loads((s19_a3.parent / "manifest.json").read_text(encoding="utf-8")),
    )
    prev_s19_mf = ops_dir / "manifests/watch6-v3-fix3-shard-19-rec1.json"
    publish_json(
        auto1_dir / "RETRY.json",
        {
            "schema": "ops-zero-export-retry/v1",
            "completion_attempt": 1,
            "max_completion_attempts": 2,
            "prior_manifest": {"path": str(prev_s19_mf), "sha256": digest(prev_s19_mf)},
            "prior_receipt": {"path": str(s19_a2.resolve()), "sha256": digest(s19_a2)},
            "manifest_sha256": digest(auto1_dir / "manifest.json"),
            "unallocated_proof": None,
        },
    )

    auto2_dir = tmp_path / "v6-ok/s19-auto2"
    s19_p4 = _write_v3_attempt(
        auto2_dir / "output",
        shard=19,
        mode="pass",
        export_slots=p.shard_indices(19),
        pair_rows_by_slot=rows,
        prestart_retries_used=0,
        compute_completion_attempt=1,
        prior_receipts=s19_p1,
        completion_attempt=2,
    )
    publish_json(
        auto2_dir / "manifest.json",
        json.loads((s19_p4.parent / "manifest.json").read_text(encoding="utf-8")),
    )
    publish_json(
        auto2_dir / "RETRY.json",
        {
            "schema": "ops-zero-export-retry/v1",
            "completion_attempt": 2,
            "max_completion_attempts": 2,
            "prior_manifest": {
                "path": str(auto1_dir / "manifest.json"),
                "sha256": digest(auto1_dir / "manifest.json"),
            },
            "prior_receipt": {"path": str(s19_a3.resolve()), "sha256": digest(s19_a3)},
            "manifest_sha256": digest(auto2_dir / "manifest.json"),
            "unallocated_proof": None,
        },
    )
    (ops_dir / "manifests/watch6-v3-fix3-shard-19-auto2.json").write_text(
        (auto2_dir / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with (
        patch.object(Path, "read_text", forbid_pair_payload_read),
        patch.object(final_look, "decide_v3", side_effect=AssertionError("OUTCOME PEEKED!")),
    ):
        with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE.*exceeded compute completion"):
            final_look.run_final_look(
                ops_dir=ops_dir,
                out_dir=tmp_path / "v6-ok/look-cap-exceeded",
                ruling_json=ruling_json,
                amendment_md=amend_path,
                evidence_dir=ev_dir,
                require_clean_git=False,
            )


def _rebuild_and_immutably_publish_forged_sidecar(
    out_dir: Path,
    ops_dir: Path,
    ev_dir: Path,
    *,
    ops_name: str = "watch6-v3-fix3-shard-12",
    shard: int = 12,
    sync_manifests_and_log: bool = True,
) -> Path:
    """Re-sync hashes on mutated synthetic evidence and immutably publish a matching sidecar."""
    rec_path = out_dir / "receipt.json"
    exec_mf_path = out_dir / "manifest.json"
    src_mf_path = ops_dir / "manifests" / f"{ops_name}.json"
    live_mf_path = ops_dir / "manifests" / f"{ops_name}.live.json"
    acct_path = ops_dir / "dispatch" / ops_name / f"{ops_name}.accounting.json"
    oplog_path = ops_dir / "logs" / f"{ops_name}.log"

    exec_mf = json.loads(exec_mf_path.read_text(encoding="utf-8"))
    rec = json.loads(rec_path.read_text(encoding="utf-8"))

    if sync_manifests_and_log:
        src_mf = json.loads(json.dumps(exec_mf))
        if isinstance(src_mf.get("budget"), dict):
            src_mf["budget"]["other_assignments"] = {}
        src_mf_path.write_text(json.dumps(src_mf, sort_keys=True) + "\n", encoding="utf-8")
        exec_mf_path.write_text(json.dumps(exec_mf, sort_keys=True) + "\n", encoding="utf-8")
        if live_mf_path.exists():
            live_mf_path.write_text(json.dumps(exec_mf, sort_keys=True) + "\n", encoding="utf-8")
        if acct_path.exists():
            acct_obj = json.loads(acct_path.read_text(encoding="utf-8"))
            acct_obj["source_manifest"] = str(src_mf_path)
            acct_obj["source_manifest_sha256"] = digest(src_mf_path)
            if live_mf_path.exists():
                acct_obj["manifest"] = str(live_mf_path)
                acct_obj["manifest_sha256"] = digest(live_mf_path)
            acct_path.write_text(json.dumps(acct_obj, sort_keys=True) + "\n", encoding="utf-8")
        rec["manifest_sha256"] = digest(exec_mf_path)
        rec_path.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
        lines = oplog_path.read_text(encoding="utf-8").splitlines()
        new_lines = [
            (
                f"INCOMPLETE: {rec.get('error')}; receipt: {rec_path.resolve()}"
                if ln.startswith("INCOMPLETE: ")
                else ln
            )
            for ln in lines
        ]
        oplog_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def fdesc(p_item: Path) -> dict[str, object]:
        resolved = p_item.resolve()
        return {"path": str(resolved), "sha256": digest(resolved), "size": resolved.stat().st_size}

    polls_meta = []
    for idx, poll_p in enumerate(sorted(out_dir.glob("poll-*.json"))):
        p_obj = json.loads(poll_p.read_text(encoding="utf-8"))
        polls_meta.append(
            {
                "index": idx,
                **fdesc(poll_p),
                "status": str(p_obj["state"]["status"]),
                "updated_at": p_obj["state"]["updated_at"],
                "exports_count": 0,
                "terminal": None,
            }
        )

    ctrl_ops = rec["control_operations"]
    failed_ctrl_idx = len(ctrl_ops) - 1
    failed_ctrl = ctrl_ops[failed_ctrl_idx]
    cli_calls = rec["cli_calls"]
    failed_cli_idx = next(
        i
        for i, c in enumerate(cli_calls)
        if c.get("thread") == "MainThread" and c.get("status") != "completed"
    )
    failed_cli = cli_calls[failed_cli_idx]
    stop_cli = cli_calls[failed_cli_idx + 1]
    sess_cli = cli_calls[failed_cli_idx + 2]

    poll_log_count = sum(
        1
        for ln in oplog_path.read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith("{")
    )
    sidecar_payload = {
        "schema": proof.TRANSPORT_EVIDENCE_SCHEMA,
        "classification": proof.TRANSPORT_EVIDENCE_CLASSIFICATION,
        "identity": {
            "namespace": BASELINE_NAMESPACE,
            "science": BASELINE_SCIENCE,
            "commit": BASELINE_COMMIT,
            "shard": shard,
            "job_id": rec["job_id"],
            "session": rec["session"],
            "endpoint": rec["endpoint"],
            "generation": rec["generation"],
        },
        "receipt": fdesc(rec_path),
        "executed_manifest": fdesc(exec_mf_path),
        "ops_manifests": {
            "ops_name": ops_name,
            "source_manifest": fdesc(src_mf_path),
            "live_manifest": fdesc(live_mf_path) if live_mf_path.exists() else None,
            "accounting_record": fdesc(acct_path) if acct_path.exists() else None,
            "relationship": proof.OPS_MANIFEST_RELATIONSHIP,
        },
        "hardware": {
            **fdesc(out_dir / "hardware.json"),
            "ready": True,
            "kind": "L4",
        },
        "polls": polls_meta,
        "operational_log": {
            **fdesc(oplog_path),
            "exit_code": 1,
            "poll_lines_count": poll_log_count,
            "max_reported_exports": 0,
        },
        "supervisor": {
            "supervisor_sha256": PINNED_SUPERVISOR_SHA256,
            "supervisor_relpath": PINNED_SUPERVISOR_REL,
            "provenance": dict(proof.EXPECTED_SUPERVISOR_PROVENANCE),
        },
        "timing": {
            "failed_control_index": failed_ctrl_idx,
            "failed_control_started_monotonic": failed_ctrl["started_monotonic"],
            "failed_control_ended_monotonic": failed_ctrl["ended_monotonic"],
            "failed_cli_index": failed_cli_idx,
            "failed_cli_started_monotonic": failed_cli["started_monotonic"],
            "failed_cli_ended_monotonic": failed_cli["ended_monotonic"],
            "failed_cli_elapsed_seconds": failed_cli["elapsed_seconds"],
            "failed_cli_timeout_seconds": failed_cli["timeout_seconds"],
            "stop_cli_index": failed_cli_idx + 1,
            "stop_cli_started_monotonic": stop_cli["started_monotonic"],
            "stop_cli_ended_monotonic": stop_cli["ended_monotonic"],
            "sessions_cli_index": failed_cli_idx + 2,
            "sessions_cli_started_monotonic": sess_cli["started_monotonic"],
            "sessions_cli_ended_monotonic": sess_cli["ended_monotonic"],
        },
        "error_signature": proof._parse_renewal_ws_error_signature(rec["error"]),
    }

    sidecar_path = ev_dir / f"transport-evidence-{ops_name}-01.json"
    for sfx in ("", ".sha256", ".complete"):
        Path(f"{sidecar_path}{sfx}").unlink(missing_ok=True)
    publish_json(sidecar_path, sidecar_payload)
    return sidecar_path


def test_v6_1_same_invocation_closure_reverification_refuses_py_mutation_and_empty_pycache(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Astra V6 Reject Point 1: same-invocation re-verification refuses .py edit & empty pycache."""
    clone_py = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-same-inv-py")
    clone_pc = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-same-inv-pc")

    amend_md = _write_immutable_text(
        tmp_path / "WATCH6_V3_ACCOUNTING_AMENDMENT_V6_1_20261101.md",
        _render_amendment_markdown(ADAPTER_ROOT),
    )
    r_py, _ = _write_renewal_ws_shard_fixture(
        tmp_path / "ev-py/pkg",
        ops_dir=tmp_path / "ev-py/ops",
        evidence_dir=tmp_path / "ev-py/ev",
        shard=12,
    )
    r_pc, _ = _write_renewal_ws_shard_fixture(
        tmp_path / "ev-pc/pkg",
        ops_dir=tmp_path / "ev-pc/ops",
        evidence_dir=tmp_path / "ev-pc/ev",
        shard=12,
    )

    with isolated_broker_execution("v6-1-same-invocation-closure") as iso:
        assert not hasattr(iso, "verified_closures")
        assert not hasattr(iso, "verified_protocols")
        assert not hasattr(iso, "verified_bound_sciences")

        # --- Probe 1A: tracked .py mutation within the SAME invocation ---
        assert verify_baseline_closure(clone_py)["science"] == BASELINE_SCIENCE
        assert compute_broker_identity(clone_py) == BASELINE_SCIENCE
        with bind_broker_protocol(clone_py) as bound_ok:
            assert bound_ok["science"] == BASELINE_SCIENCE
        proof.verify_renewal_ws_transport_evidence(
            r_py.parent,
            ops_dir=tmp_path / "ev-py/ops",
            evidence_dir=tmp_path / "ev-py/ev",
            broker_root=clone_py,
        )
        ratify.verify_amendment_document(
            amend_md,
            broker_root=clone_py,
            require_clean_git=False,
        )

        target_py = clone_py / "src/model/ego2s_network.py"
        orig_py_bytes = target_py.read_bytes()
        target_py.write_bytes(orig_py_bytes + b"\n# mutated tracked py bytes\n")
        assert active_broker_isolation() is iso

        assert compute_broker_identity(clone_py) != BASELINE_SCIENCE
        with pytest.raises(ValueError, match="REFUSE"):
            verify_baseline_closure(clone_py)
        with pytest.raises(ValueError, match="REFUSE"):
            with bind_broker_protocol(clone_py):
                pass
        with pytest.raises(ValueError, match="REFUSE"):
            proof.verify_renewal_ws_transport_evidence(
                r_py.parent,
                ops_dir=tmp_path / "ev-py/ops",
                evidence_dir=tmp_path / "ev-py/ev",
                broker_root=clone_py,
            )
        with pytest.raises(ValueError, match="REFUSE"):
            ratify.verify_amendment_document(
                amend_md,
                broker_root=clone_py,
                require_clean_git=False,
            )

        # Restore src/ .py, confirm pass, then mutate tracked .py in tools/ops/ within same iso:
        target_py.write_bytes(orig_py_bytes)
        assert verify_baseline_closure(clone_py)["science"] == BASELINE_SCIENCE
        tools_py = clone_py / "tools/ops/package_spec.py"
        orig_tools_bytes = tools_py.read_bytes()
        tools_py.write_bytes(orig_tools_bytes + b"\n# mutated tools py bytes\n")
        assert active_broker_isolation() is iso
        assert compute_broker_identity(clone_py) != BASELINE_SCIENCE
        with pytest.raises(ValueError, match="REFUSE"):
            verify_baseline_closure(clone_py)
        tools_py.write_bytes(orig_tools_bytes)
        assert verify_baseline_closure(clone_py)["science"] == BASELINE_SCIENCE

        # --- Probe 1B: empty __pycache__ created within the SAME invocation ---
        assert verify_baseline_closure(clone_pc)["science"] == BASELINE_SCIENCE
        with bind_broker_protocol(clone_pc) as bound_pc_ok:
            assert bound_pc_ok["science"] == BASELINE_SCIENCE
        proof.verify_renewal_ws_transport_evidence(
            r_pc.parent,
            ops_dir=tmp_path / "ev-pc/ops",
            evidence_dir=tmp_path / "ev-pc/ev",
            broker_root=clone_pc,
        )
        ratify.verify_amendment_document(
            amend_md,
            broker_root=clone_pc,
            require_clean_git=False,
        )

        for rel_pycache in (
            "src/__pycache__",
            "research/watch6_search_efficacy_v3/__pycache__",
            "__pycache__",
        ):
            empty_pc = clone_pc / rel_pycache
            empty_pc.mkdir()
            assert list(empty_pc.iterdir()) == []
            assert active_broker_isolation() is iso

            with pytest.raises(ValueError, match="REFUSE.*__pycache__"):
                verify_baseline_closure(clone_pc)
            with pytest.raises(ValueError, match="REFUSE.*__pycache__"):
                with bind_broker_protocol(clone_pc):
                    pass
            with pytest.raises(ValueError, match="REFUSE.*__pycache__"):
                proof.verify_renewal_ws_transport_evidence(
                    r_pc.parent,
                    ops_dir=tmp_path / "ev-pc/ops",
                    evidence_dir=tmp_path / "ev-pc/ev",
                    broker_root=clone_pc,
                )
            with pytest.raises(ValueError, match="REFUSE.*__pycache__"):
                ratify.verify_amendment_document(
                    amend_md,
                    broker_root=clone_pc,
                    require_clean_git=False,
                )
            empty_pc.rmdir()
            assert verify_baseline_closure(clone_pc)["science"] == BASELINE_SCIENCE


@isolated_broker_execution("test-v6-1-point2-conjunction")
def test_v6_1_exact_evidence_conjunction_refuses_astra_four_probes(tmp_path: Path) -> None:
    """Astra V6 Reject Point 2: failed exec rc=0, stop rc=null, workers=8, hardware vcpu=1."""

    def fresh_case(label: str) -> tuple[Path, Path, Path]:
        base = tmp_path / label
        ops_d = base / "ops"
        ev_d = base / "ev"
        r_p, _ = _write_renewal_ws_shard_fixture(
            base / "pkg",
            ops_dir=ops_d,
            evidence_dir=ev_d,
            shard=12,
            publish_sidecar=False,
        )
        return r_p.parent, ops_d, ev_d

    def assert_refused_both_publish_and_prepublished_sidecar(
        out_d: Path,
        ops_d: Path,
        ev_d: Path,
    ) -> None:
        sidecar = ev_d / "transport-evidence-watch6-v3-fix3-shard-12-01.json"
        with pytest.raises(ValueError):
            proof.publish_renewal_ws_transport_evidence(
                out_d,
                sidecar,
                ops_dir=ops_d,
                ops_name="watch6-v3-fix3-shard-12",
            )
        _rebuild_and_immutably_publish_forged_sidecar(
            out_d,
            ops_d,
            ev_d,
            sync_manifests_and_log=False,
        )
        _assert_renewal_ws_case_refused(out_d, ops_d, ev_d, republish=False)

    # Positive control: unmutated fresh_case passes both publish and rebuilt immutable sidecar:
    out_ok, ops_ok, ev_ok = fresh_case("positive-control")
    _rebuild_and_immutably_publish_forged_sidecar(out_ok, ops_ok, ev_ok)
    assert (
        proof.verify_renewal_ws_transport_evidence(
            out_ok,
            ops_dir=ops_ok,
            ops_name="watch6-v3-fix3-shard-12",
            evidence_dir=ev_ok,
        )["shard"]
        == 12
    )
    # Positive control for pinned-supervisor 9-key cli_calls (no returncode) + GPU hardware.json:
    out_pin, ops_pin, ev_pin = fresh_case("positive-pinned-schema")
    rec_pin = json.loads((out_pin / "receipt.json").read_text(encoding="utf-8"))
    for c_item in rec_pin["cli_calls"]:
        c_item.pop("returncode", None)
    (out_pin / "receipt.json").write_text(
        json.dumps(rec_pin, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_pin / "hardware.json").write_text(
        json.dumps(
            {
                "ready": True,
                "hardware": {"kind": "L4", "name": "NVIDIA L4", "compute_cap": "8.9"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _rebuild_and_immutably_publish_forged_sidecar(out_pin, ops_pin, ev_pin)
    assert (
        proof.verify_renewal_ws_transport_evidence(
            out_pin,
            ops_dir=ops_pin,
            ops_name="watch6-v3-fix3-shard-12",
            evidence_dir=ev_pin,
        )["shard"]
        == 12
    )

    # Probe 2.1: failed exec/renewal CLI call must have failing returncode (not 0, null, or 2):
    for bad_rc_label, bad_rc in (
        ("zero", 0),
        ("null", None),
        ("two", 2),
        ("missing", "__DELETE__"),
    ):
        out_1, ops_1, ev_1 = fresh_case(f"failed-exec-rc-{bad_rc_label}")
        rec_1 = json.loads((out_1 / "receipt.json").read_text(encoding="utf-8"))
        if bad_rc == "__DELETE__":
            rec_1["cli_calls"][2].pop("returncode", None)
        else:
            rec_1["cli_calls"][2]["returncode"] = bad_rc
        (out_1 / "receipt.json").write_text(
            json.dumps(rec_1, sort_keys=True) + "\n", encoding="utf-8"
        )
        assert_refused_both_publish_and_prepublished_sidecar(out_1, ops_1, ev_1)

    # Probe 2.2: stop/sessions verification must have real successful returncode 0 (not null/1):
    for call_idx, call_name in ((3, "stop"), (4, "sessions")):
        for rc_label, bad_rc in (("null", None), ("nonzero", 1), ("missing", "__DELETE__")):
            out_2, ops_2, ev_2 = fresh_case(f"{call_name}-rc-{rc_label}")
            rec_2 = json.loads((out_2 / "receipt.json").read_text(encoding="utf-8"))
            if bad_rc == "__DELETE__":
                rec_2["cli_calls"][call_idx].pop("returncode", None)
            else:
                rec_2["cli_calls"][call_idx]["returncode"] = bad_rc
            (out_2 / "receipt.json").write_text(
                json.dumps(rec_2, sort_keys=True) + "\n", encoding="utf-8"
            )
            assert_refused_both_publish_and_prepublished_sidecar(out_2, ops_2, ev_2)

    # Probe 2.3: workers must equal manifest's W (4); workers=8 refused everywhere:
    for w_label, mf_mutator in (
        ("env-workers-8", lambda m: m["env"].update(WATCH6_WORKERS="8")),
        ("env-workers-missing", lambda m: m["env"].pop("WATCH6_WORKERS", None)),
        (
            "frozen-layout-workers-8",
            lambda m: m["env"].update(
                WATCH6_FROZEN_LAYOUT=json.dumps(
                    {"base_minutes_per_shard": 209, "shards": 24, "workers": 8}
                )
            ),
        ),
        ("watch6-v3-workers-8", lambda m: m["watch6_v3"].update(workers=8)),
        ("top-manifest-workers-8", lambda m: m.update(workers=8)),
    ):
        out_3, ops_3, ev_3 = fresh_case(f"workers-{w_label}")
        mf_3 = json.loads((out_3 / "manifest.json").read_text(encoding="utf-8"))
        mf_mutator(mf_3)
        (out_3 / "manifest.json").write_text(
            json.dumps(mf_3, sort_keys=True) + "\n", encoding="utf-8"
        )
        _rebuild_and_immutably_publish_forged_sidecar(
            out_3,
            ops_3,
            ev_3,
            sync_manifests_and_log=True,
        )
        _assert_renewal_ws_case_refused(out_3, ops_3, ev_3, republish=False)
        for sfx in ("", ".sha256", ".complete"):
            Path(f"{ev_3 / 'transport-evidence-watch6-v3-fix3-shard-12-01.json'}{sfx}").unlink(
                missing_ok=True
            )
        with pytest.raises(ValueError):
            proof.publish_renewal_ws_transport_evidence(
                out_3,
                ev_3 / "transport-evidence-watch6-v3-fix3-shard-12-01.json",
                ops_dir=ops_3,
                ops_name="watch6-v3-fix3-shard-12",
            )

    # Probe 2.4: hardware vCPU must meet min_vcpu (12); vcpu=1 refused:
    for hw_label, bad_hw in (
        ("vcpu-1", {"ready": True, "hardware": {"kind": "L4", "vcpu": 1}}),
        ("vcpu-11", {"ready": True, "hardware": {"kind": "L4", "vcpu": 11}}),
        ("vcpu-null", {"ready": True, "hardware": {"kind": "L4", "vcpu": None}}),
        (
            "top-vcpu-1",
            {
                "ready": True,
                "vcpu": 1,
                "hardware": {"kind": "L4", "name": "NVIDIA L4", "compute_cap": "8.9"},
            },
        ),
        ("missing-vcpu-and-gpu-meta", {"ready": True, "hardware": {"kind": "L4"}}),
    ):
        out_4, ops_4, ev_4 = fresh_case(f"hw-{hw_label}")
        (out_4 / "hardware.json").write_text(
            json.dumps(bad_hw, sort_keys=True) + "\n", encoding="utf-8"
        )
        assert_refused_both_publish_and_prepublished_sidecar(out_4, ops_4, ev_4)


@isolated_broker_execution("test-v6-1-point3-fault-veto")
def test_v6_1_overriding_fault_veto_scans_all_operational_evidence(tmp_path: Path) -> None:
    """Astra V6 Reject Point 3: overriding fault veto across ALL operational JSON/log fields."""

    def fresh_case(label: str) -> tuple[Path, Path, Path]:
        base = tmp_path / label
        ops_d = base / "ops"
        ev_d = base / "ev"
        r_p, _ = _write_renewal_ws_shard_fixture(
            base / "pkg",
            ops_dir=ops_d,
            evidence_dir=ev_d,
            shard=12,
            publish_sidecar=False,
        )
        return r_p.parent, ops_d, ev_d

    def assert_fault_refused_everywhere(out_d: Path, ops_d: Path, ev_d: Path) -> None:
        sidecar = ev_d / "transport-evidence-watch6-v3-fix3-shard-12-01.json"
        with pytest.raises(ValueError):
            proof.publish_renewal_ws_transport_evidence(
                out_d,
                sidecar,
                ops_dir=ops_d,
                ops_name="watch6-v3-fix3-shard-12",
            )
        _rebuild_and_immutably_publish_forged_sidecar(
            out_d,
            ops_d,
            ev_d,
            sync_manifests_and_log=False,
        )
        _assert_renewal_ws_case_refused(out_d, ops_d, ev_d, republish=False)

    # Positive control: unmutated fresh_case passes before any fault injection:
    out_ok, ops_ok, ev_ok = fresh_case("positive-control")
    _rebuild_and_immutably_publish_forged_sidecar(
        out_ok, ops_ok, ev_ok, sync_manifests_and_log=False
    )
    assert (
        proof.verify_renewal_ws_transport_evidence(
            out_ok,
            ops_dir=ops_ok,
            ops_name="watch6-v3-fix3-shard-12",
            evidence_dir=ev_ok,
        )["shard"]
        == 12
    )

    # 1. Astra's exact Point 3 probe: error="RuntimeError: child scientific failure" in oplog JSON:
    for oplog_label, extra_fields in (
        ("astra-error-field", {"error": "RuntimeError: child scientific failure"}),
        ("runtime-error-non-sci", {"error": "RuntimeError: child worker crashed"}),
        ("custom-note-field", {"diagnostic": "RuntimeError: child worker crashed"}),
        ("nested-list-fault", {"warnings": ["RuntimeError: child scientific failure"]}),
        ("non-string-error-obj", {"error": {"code": 1}}),
    ):
        out_l, ops_l, ev_l = fresh_case(f"oplog-{oplog_label}")
        oplog_p = ops_l / "logs/watch6-v3-fix3-shard-12.log"
        mut_lines = []
        injected = False
        for ln in oplog_p.read_text(encoding="utf-8").splitlines():
            if not injected and ln.strip().startswith("{"):
                j_obj = json.loads(ln)
                j_obj.update(extra_fields)
                mut_lines.append(json.dumps(j_obj, sort_keys=True))
                injected = True
            else:
                mut_lines.append(ln)
        assert injected
        oplog_p.write_text("\n".join(mut_lines) + "\n", encoding="utf-8")
        assert_fault_refused_everywhere(out_l, ops_l, ev_l)

    # 2. Queue log and run-specific sibling log RuntimeError probes:
    out_ql, ops_ql, ev_ql = fresh_case("queue-runtime-error")
    job_ql = json.loads((out_ql / "receipt.json").read_text(encoding="utf-8"))["job_id"]
    (ops_ql / "logs/queue-20261101.log").write_text(
        f"2026-11-01 {job_ql} RuntimeError: child worker crashed\n",
        encoding="utf-8",
    )
    assert_fault_refused_everywhere(out_ql, ops_ql, ev_ql)

    out_sl, ops_sl, ev_sl = fresh_case("sibling-run-log-fault")
    (ops_sl / "logs/watch6-v3-fix3-shard-12-child.log").write_text(
        "RuntimeError: child scientific failure\n",
        encoding="utf-8",
    )
    assert_fault_refused_everywhere(out_sl, ops_sl, ev_sl)

    # 3. Measurements, hardware, polls, receipt nested fields, and ops accounting record:
    for art_label, mutator in (
        (
            "warmup-fault",
            lambda out_d, _ops_d: (out_d / "warmup-measurement.json").write_text(
                '{"status": "PASS", "warmup_seconds": 14.2, '
                '"note": "RuntimeError: child scientific failure"}\n',
                encoding="utf-8",
            ),
        ),
        (
            "calib-fault",
            lambda out_d, _ops_d: (out_d / "rate-calibration.json").write_text(
                '{"status": "PASS", "observed_rate_units_per_hour": 1.94, '
                '"error": "RuntimeError: child scientific failure"}\n',
                encoding="utf-8",
            ),
        ),
        (
            "upload-fault",
            lambda out_d, _ops_d: (out_d / "upload-measurement.json").write_text(
                '{"elapsed_seconds": 48.5, "staging_completed": true, '
                '"note": "RuntimeError: child scientific failure"}\n',
                encoding="utf-8",
            ),
        ),
        (
            "hardware-fault",
            lambda out_d, _ops_d: (out_d / "hardware.json").write_text(
                '{"ready": true, "hardware": {"kind": "L4", "vcpu": 12, '
                '"note": "RuntimeError: child scientific failure"}}\n',
                encoding="utf-8",
            ),
        ),
        (
            "poll-sup-tail-fault",
            lambda out_d, _ops_d: (out_d / "poll-000001.json").write_text(
                (out_d / "poll-000001.json")
                .read_text(encoding="utf-8")
                .replace(
                    '"supervisor_log_tail": ""',
                    '"supervisor_log_tail": "RuntimeError: child worker crashed"',
                ),
                encoding="utf-8",
            ),
        ),
        (
            "receipt-nested-fault",
            lambda out_d, _ops_d: (out_d / "receipt.json").write_text(
                json.dumps(
                    {
                        **json.loads((out_d / "receipt.json").read_text(encoding="utf-8")),
                        "usage_after": {"note": "RuntimeError: child scientific failure"},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            ),
        ),
        (
            "ops-accounting-fault",
            lambda _out_d, ops_d: (
                ops_d / "dispatch/watch6-v3-fix3-shard-12/watch6-v3-fix3-shard-12.accounting.json"
            ).write_text(
                json.dumps(
                    {
                        **json.loads(
                            (
                                ops_d / "dispatch/watch6-v3-fix3-shard-12/"
                                "watch6-v3-fix3-shard-12.accounting.json"
                            ).read_text(encoding="utf-8")
                        ),
                        "note": "RuntimeError: child scientific failure",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            ),
        ),
    ):
        out_m, ops_m, ev_m = fresh_case(f"art-{art_label}")
        mutator(out_m, ops_m)
        assert_fault_refused_everywhere(out_m, ops_m, ev_m)
