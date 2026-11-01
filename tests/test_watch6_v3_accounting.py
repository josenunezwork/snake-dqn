"""Synthetic positive, negative, tamper, lineage, live-proof, and baseline-equivalence tests
for the Mac-only Watch6 V3 accounting adapter (Astra Ruling Points 1-7).
"""

from __future__ import annotations

import functools
import importlib.machinery
import importlib.util
import json
import marshal
import py_compile
import shutil
import struct
import subprocess
import sys
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
    BASELINE_COMMIT,
    BASELINE_NAMESPACE,
    BASELINE_SCIENCE,
    MAIN_CHECKOUT_ROOT,
    REGISTRATION_V3_SHA256,
    RETAINED_PENDING_CLEANUP_ERROR,
    RULING_2_REL,
    RULING_2_SHA256,
    RULING_3_REL,
    RULING_3_SHA256,
    SHARD00_EXEC_MANIFEST_SHA256,
    SHARD00_PATH_RULING_REL,
    SHARD00_PATH_RULING_SHA256,
    SHARD00_PROOF_REL,
    SHARD00_PROOF_SHA256,
    SHARD00_RECEIPT_REL,
    SHARD00_RECEIPT_SHA256,
    V3_BROKER_ROOT,
    adapter_commit,
    adapter_files_map,
    adapter_inventory_sha256,
    admission,
    artifacts_root,
    bind_broker_protocol,
    compute_broker_identity,
    dispatch,
    final_look,
    prepare,
    proof,
    ratify,
    recover,
    resume,
    verify_baseline_closure,
)


def _render_amendment_markdown(repo: Path | None = None, commit_sha: str | None = None) -> str:
    root = (repo or ADAPTER_ROOT).resolve()
    c_sha = commit_sha or adapter_commit(root, require_clean=False)
    f_map = adapter_files_map(root)
    inv_sha = adapter_inventory_sha256(root)
    lines = [
        "# Watch6 V3 Pre-Look Accounting Amendment V4 (2026-10-31)",
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
        tmp_path / "WATCH6_V3_ACCOUNTING_AMENDMENT_V4_20261031.md",
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
        "source_paths": p.source_paths(),
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
            "primary_allocation": p.allocation(),
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
    for slot in export_slots or []:
        alloc = p.allocation()[slot]
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
        "source_paths": p.source_paths(),
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
            "primary_allocation": p.allocation(),
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

        # 5. Worktree not at HEAD == 357e3504 (e.g. ADAPTER_ROOT) refused as execution broker:
        if (
            subprocess.check_output(
                ["git", "--no-optional-locks", "rev-parse", "HEAD"],
                cwd=ADAPTER_ROOT,
                text=True,
            ).strip()
            != BASELINE_COMMIT
        ):
            with pytest.raises(ValueError, match="HEAD is"):
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


def test_point3_pycache_and_bytecode_verification_rules(
    clean_baseline_template: Path,
    tmp_path: Path,
) -> None:
    """Verify safe __pycache__/.pyc + Numba .nbi/.nbc rules: genuine caches pass, forgeries fail."""
    # 1. Untouched live V3 broker passes as-is (with existing Numba src/simd_env/__pycache__):
    assert verify_baseline_closure(V3_BROKER_ROOT, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )

    clone = _fresh_baseline_clone(clean_baseline_template, tmp_path / "clone-pycache")
    tag = sys.implementation.cache_tag
    tracked_py = clone / "src/model/ego2s_network.py"
    pycache_dir = clone / "src/model/__pycache__"
    valid_pyc = pycache_dir / f"ego2s_network.{tag}.pyc"

    # 2. Genuine CPython .pyc compiled from tracked ego2s_network.py (TIMESTAMP, CHECKED_HASH,
    #    and UNCHECKED_HASH modes) PASSES:
    for mode in (
        py_compile.PycInvalidationMode.TIMESTAMP,
        py_compile.PycInvalidationMode.CHECKED_HASH,
        py_compile.PycInvalidationMode.UNCHECKED_HASH,
    ):
        py_compile.compile(
            str(tracked_py),
            cfile=str(valid_pyc),
            dfile="src/model/ego2s_network.py",
            doraise=True,
            invalidation_mode=mode,
        )
        assert valid_pyc.is_file()
        assert compute_broker_identity(clone) == BASELINE_SCIENCE
        assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
            BASELINE_SCIENCE
        )
        valid_pyc.unlink()

    # 3. Stray .pyc outside __pycache__/ is unconditionally REFUSED:
    stray_pyc = clone / "src/model/ego2s_network.pyc"
    py_compile.compile(str(tracked_py), cfile=str(stray_pyc), doraise=True)
    try:
        assert compute_broker_identity(clone) == BASELINE_SCIENCE
        with pytest.raises(
            ValueError, match=r"stray Python source or bytecode.*src/model/ego2s_network\.pyc"
        ):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        stray_pyc.unlink()

    # 4. Orphan .pyc in __pycache__/ without a tracked baseline .py is REFUSED:
    orphan_pyc = pycache_dir / f"ghost_module.{tag}.pyc"
    py_compile.compile(str(tracked_py), cfile=str(orphan_pyc), doraise=True)
    try:
        assert compute_broker_identity(clone) == BASELINE_SCIENCE
        with pytest.raises(ValueError, match=r"orphan \.pyc.*ghost_module"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        orphan_pyc.unlink()

    # 5. Foreign interpreter tag .pyc is REFUSED:
    foreign_tag_pyc = pycache_dir / "ego2s_network.cpython-311.pyc"
    py_compile.compile(str(tracked_py), cfile=str(foreign_tag_pyc), doraise=True)
    try:
        with pytest.raises(ValueError, match="foreign interpreter cache tag"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        foreign_tag_pyc.unlink()

    # 6. Stale timestamp/size .pyc header is REFUSED:
    py_compile.compile(
        str(tracked_py),
        cfile=str(valid_pyc),
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    raw_valid = bytearray(valid_pyc.read_bytes())
    orig_mtime, orig_size = struct.unpack("<II", raw_valid[8:16])
    raw_valid[8:16] = struct.pack("<II", (orig_mtime + 1) & 0xFFFFFFFF, orig_size)
    valid_pyc.write_bytes(bytes(raw_valid))
    try:
        with pytest.raises(ValueError, match=r"stale or mismatched timestamp \.pyc header"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        valid_pyc.unlink()

    # 7. Mismatched PEP-552 source-hash .pyc (UNCHECKED_HASH bit_field=1) is REFUSED:
    forged_evil_code = marshal.dumps(compile("EVIL_PAYLOAD = 999\n", "ego2s_network.py", "exec"))
    bad_hash_pyc = (
        importlib.util.MAGIC_NUMBER + struct.pack("<I", 1) + (b"\xaa" * 8) + forged_evil_code
    )
    valid_pyc.write_bytes(bad_hash_pyc)
    try:
        with pytest.raises(ValueError, match="mismatched PEP-552 source_hash"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        valid_pyc.unlink()

    # 8. FORGED-HEADER .pyc (exact MAGIC_NUMBER + exact tracked mtime/size OR exact PEP-552
    #    SipHash of tracked ego2s_network.py, but marshalled evil CodeType!) is REFUSED:
    st = tracked_py.stat()
    tracked_bytes = tracked_py.read_bytes()
    for forged_header in (
        importlib.util.MAGIC_NUMBER
        + struct.pack("<III", 0, int(st.st_mtime) & 0xFFFFFFFF, st.st_size & 0xFFFFFFFF),
        importlib.util.MAGIC_NUMBER
        + struct.pack("<I", 1)
        + importlib.util.source_hash(tracked_bytes),
    ):
        valid_pyc.write_bytes(forged_header + forged_evil_code)
        try:
            assert compute_broker_identity(clone) == BASELINE_SCIENCE
            with pytest.raises(
                ValueError, match="bytecode body .* does not match compiled tracked"
            ):
                verify_baseline_closure(clone, require_head_at_baseline=True)
        finally:
            valid_pyc.unlink()

    # 9. Foreign non-bytecode file inside __pycache__/ (e.g. .so) is REFUSED:
    pycache_so = pycache_dir / "stealth.so"
    pycache_so.write_bytes(b"\xcf\xfa\xed\xfe")
    try:
        with pytest.raises(ValueError, match=r"foreign file inside __pycache__"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        pycache_so.unlink()

    # 10. Numba cache (.nbi/.nbc) verification: genuine pair passes; tampered stamp, orphan .nbc,
    #     and unsafe pickle global opcode each FAIL CLOSED without executing pickle.loads:
    broker_simd_cache = V3_BROKER_ROOT / "src/simd_env/__pycache__"
    clone_simd_cache = clone / "src/simd_env/__pycache__"
    clone_simd_cache.mkdir(parents=True, exist_ok=True)

    orphan_nbc = clone_simd_cache / "v8_nb.fake_kernel-1.py312.1.nbc"
    orphan_nbc.write_bytes(b"\x80\x04\x95\x02\x00\x00\x00\x00\x00\x00\x00N.")
    try:
        with pytest.raises(ValueError, match=r"orphan Numba \.nbc"):
            verify_baseline_closure(clone, require_head_at_baseline=True)
    finally:
        orphan_nbc.unlink()

    sample_nbis = sorted(broker_simd_cache.glob("v8_nb.*.nbi"))
    if sample_nbis:
        src_nbi = sample_nbis[0]
        stem = src_nbi.name[: -len(".nbi")]
        sibling_nbcs = sorted(broker_simd_cache.glob(f"{stem}.*.nbc"))
        copied: list[Path] = []
        try:
            dst_nbi = clone_simd_cache / src_nbi.name
            shutil.copy2(src_nbi, dst_nbi)
            copied.append(dst_nbi)
            for s_nbc in sibling_nbcs:
                d_nbc = clone_simd_cache / s_nbc.name
                shutil.copy2(s_nbc, d_nbc)
                copied.append(d_nbc)
            # Genuine Numba cache pair copied onto clean baseline clone PASSES:
            assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
                BASELINE_SCIENCE
            )

            # Mutating the 32-byte SHA-256 source stamp inside .nbi REFUSES:
            v8_sha = (
                __import__("hashlib")
                .sha256((clone / "src/simd_env/v8_nb.py").read_bytes())
                .digest()
            )
            orig_nbi_bytes = dst_nbi.read_bytes()
            assert v8_sha in orig_nbi_bytes
            corrupted_stamp = bytes([v8_sha[0] ^ 0xFF]) + v8_sha[1:]
            dst_nbi.write_bytes(orig_nbi_bytes.replace(v8_sha, corrupted_stamp, 1))
            with pytest.raises(ValueError, match="stale or mismatched SHA-256 source stamp"):
                verify_baseline_closure(clone, require_head_at_baseline=True)

            # Unsafe pickle GLOBAL opcode (e.g. `cos\nsystem\n`) REFUSES safely via pickletools:
            dst_nbi.write_bytes(b"cos\nsystem\n(S'echo NO'\ntR.")
            with pytest.raises(ValueError, match="foreign pickle global"):
                verify_baseline_closure(clone, require_head_at_baseline=True)
        finally:
            for p_item in copied:
                p_item.unlink(missing_ok=True)

    assert verify_baseline_closure(clone, require_head_at_baseline=True)["science"] == (
        BASELINE_SCIENCE
    )
