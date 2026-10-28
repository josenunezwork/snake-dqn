"""Unit tests for C/T seed audit, runner, compact XZ resume, SIGUSR1 <15s, and fault precedence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident.ct_prepare import (
    CT_DOMAIN_PREFIX,
    STUDY,
    audit_ct_seeds,
    ct_endpoint_steps,
    write_once,
)
from research.sandbox_a_resident.ct_resume import (
    CTResumeRefused,
    build_stage_ledger,
    completion_packet,
    verify_mechanical_stop,
    write_continuation_packet,
)
from research.sandbox_a_resident.ct_run import SIGUSR1_EXIT_CODE, run_ct_experiment
from research.sandbox_a_resident.ct_task import (
    CTControl,
    ct_seed_allocations,
    train01_protected_eval_seeds,
    world_partition_table,
)
from research.sandbox_a_resident.ct_topology import TOPOLOGY_SPEC
from research.sandbox_a_resident.ct_units import (
    CompactResumeRefused,
    allocation,
    empty_ledger,
    encode_compact_stage_snapshot,
    load_ledger,
    publish_stage_snapshot,
    unit_id,
)
from research.sandbox_a_resident.units import digest
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
from tools.sandbox.manifest import ManifestError, validate_manifest
from tools.sandbox.registry import DEFAULT_REGISTRY, validate_input, world_seed

ROOT = Path(__file__).resolve().parents[2]


def _save_tiny_checkpoints(tmp_path: Path) -> tuple[Path, Path, Path]:
    torch.manual_seed(31)
    b8_path = tmp_path / "b8_tiny.pth"
    frp3_path = tmp_path / "frp3_tiny.pth"
    champ_path = tmp_path / "champ_tiny.pth"

    b8 = Ego2sNet(local_channels=7, n_scalars=60).eval()
    save_ego2s_checkpoint(b8_path, b8)

    for p in (frp3_path, champ_path):
        lin = torch.nn.Linear(61, 6).eval()
        torch.save({"tiny_linear_vector61": True, "state_dict": lin.state_dict()}, p)
    return b8_path, frp3_path, champ_path


def test_seed_domain_disjointness_partition_and_train01_exclusion() -> None:
    ctrl = CTControl()
    assert ct_endpoint_steps(ctrl) == 39_485_440
    labels, partitions, per_mix = world_partition_table(1024)
    assert len(labels) == 1024
    assert len(partitions["train"]) == 820
    assert len(partitions["assessment"]) == 204
    assert set(partitions["train"]).isdisjoint(set(partitions["assessment"]))
    assert sum(per_mix["train"].values()) == 820
    assert sum(per_mix["assessment"].values()) == 204
    assert per_mix["assessment"] == {"scripted": 68, "frozen": 68, "mixed": 68}

    protected_train01 = train01_protected_eval_seeds()
    assert len(protected_train01) == 1056

    tiny_ctrl = CTControl(worlds=12, collection_ticks=16, max_body_capacity=128, tiny=True)
    allocs = ct_seed_allocations(tiny_ctrl)
    audit = audit_ct_seeds(allocs, DEFAULT_REGISTRY)
    assert audit["status"] == "PASS"
    assert audit["aliases"] == 0
    assert audit["excluded_intersection"] == 0
    assert audit["train01_protected_count"] == 1056
    assert audit["train01_excluded_intersection"] == 0


def test_tiny_runner_end_to_end_and_compact_xz_continuation_resume(tmp_path: Path) -> None:
    b8_p, frp3_p, champ_p = _save_tiny_checkpoints(tmp_path)
    ctrl = CTControl(
        source_commit="a" * 40,
        worlds=6,
        collection_ticks=12,
        max_body_capacity=128,
        max_deaths=2,
        lookbacks=(2, 4),
        max_ordinary_roots=4,
        anchor_rows=16,
        min_death_length=1,
        min_held_out_deaths=0,
        min_held_out_death_worlds=0,
        min_held_out_ordinary_roots=1,
        min_held_out_anchor_rows=1,
        branch_horizon=3,
        minibatches=2,
        searched_batch_size=2,
        anchor_batch_size=4,
        bootstrap_resamples=50,
        scoring_chunks=2,
        branch_batch_size=16,
        snapshot_store_max_bytes=16 * 1024 * 1024,
        tiny=True,
    )

    # 1) Full uninterrupted tiny run (stage-0001 collect, stage-0002..0003 score, stage-0004 fit).
    full_out = tmp_path / "full_out"
    ledger_init = tmp_path / "empty_ledger.json"
    ledger_init.write_text(json.dumps(empty_ledger(ctrl.to_dict())))
    rc, summary_full = run_ct_experiment(
        ctrl,
        b8_checkpoint=b8_p,
        frp3_checkpoint=frp3_p,
        champion_checkpoint=champ_p,
        output=full_out,
        device="cpu",
        ledger_path=ledger_init,
    )
    assert rc == 0
    assert (full_out / "summary.json").is_file()
    assert (full_out / "summary.json.sha256").is_file()
    assert (full_out / "summary.json.complete").is_file()
    assert (full_out / "stage-0001.pt").is_file()
    assert (full_out / "stage-0002.pt").is_file()
    assert (full_out / "stage-0003.pt").is_file()
    assert (full_out / "stage-0004.pt").is_file()

    # 2) Simulate mechanical stop after stage-0001..0002 (before stage-0003 and summary.json):
    partial_export = tmp_path / "attempt1_exports"
    partial_export.mkdir()
    for s in (1, 2):
        for ext in (".pt", ".pt.sha256", ".pt.complete", ".json", ".json.sha256", ".json.complete"):
            src = full_out / f"stage-{s:04d}{ext}"
            (partial_export / src.name).write_bytes(src.read_bytes())

    # Encode stage-0001 into compact .pt.xz + .pt.xz.manifest!
    encode_compact_stage_snapshot(partial_export / "stage-0001.pt", unit=unit_id(1))
    assert (partial_export / "stage-0001.pt.xz").is_file()
    assert (partial_export / "stage-0001.pt.xz.manifest").is_file()

    orig_manifest = {
        "job_id": "sandbox-ct-cf-test-01",
        "commit": "a" * 40,
        "inputs": [
            {"path": str(b8_p), "dest": "b8-s3.pth", "sha256": digest(b8_p)},
            {"path": str(frp3_p), "dest": "frp3-s12.pth", "sha256": digest(frp3_p)},
            {"path": str(champ_p), "dest": "champion.pth", "sha256": digest(champ_p)},
            {
                "path": str(ledger_init),
                "dest": "verified-ledger.json",
                "sha256": digest(ledger_init),
            },
        ],
    }
    orig_sha = hashlib.sha256(json.dumps(orig_manifest, sort_keys=True).encode()).hexdigest()
    mech_receipt = {
        "job_id": "sandbox-ct-cf-test-01",
        "commit": "a" * 40,
        "manifest_sha256": orig_sha,
        "status": "INCOMPLETE",
        "failure_class": "CHILD_NONZERO_EXIT",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": 1,
        "terminal": {
            "job_id": "sandbox-ct-cf-test-01",
            "generation": 1,
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        },
    }
    resume_pkt = completion_packet(
        orig_manifest,
        mech_receipt,
        partial_export,
        ctrl.to_dict(),
        completion_index=1,
        job_id="sandbox-ct-cf-test-02",
        output_dir=tmp_path / "attempt2_out",
        original_manifest_sha256=orig_sha,
        use_compact_xz_for_stages=[1],
    )
    assert resume_pkt.verified_stages == (1, 2)
    assert "compact_resume" in resume_pkt.ledger
    ledger_inputs = [
        item for item in resume_pkt.manifest["inputs"] if item["dest"] == "verified-ledger.json"
    ]
    assert len(ledger_inputs) == 1
    assert ledger_inputs[0]["sha256"] == resume_pkt.ledger_sha256

    # Stage uploaded inputs in a fresh input_root (note: NO raw stage-0001.pt uploaded!).
    input_root = tmp_path / "vm_inputs"
    (input_root / "resume").mkdir(parents=True)
    for item in resume_pkt.input_records:
        dst = input_root / item["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(item["path"]).read_bytes())
    assert not (input_root / "resume/stage-0001.pt").exists()
    assert (input_root / "resume/stage-0001.pt.xz").is_file()

    resume_ledger_path = input_root / "verified-ledger.json"
    resume_ledger_path.write_text(json.dumps(resume_pkt.ledger, indent=2))

    # Foreign control refusal test before running valid resume.
    wrong_ctrl = CTControl(**{**ctrl.to_dict(), "fit_seed": 999})
    with pytest.raises(ValueError, match="foreign C/T ledger"):
        load_ledger(resume_ledger_path, wrong_ctrl.to_dict())

    # Tampered .pt.xz refusal test on a scratch copy.
    bad_root = tmp_path / "bad_vm_inputs"
    (bad_root / "resume").mkdir(parents=True)
    for item in resume_pkt.input_records:
        dst = bad_root / item["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(item["path"]).read_bytes())
    xz_bad = bad_root / "resume/stage-0001.pt.xz"
    raw_xz = bytearray(xz_bad.read_bytes())
    raw_xz[-5] ^= 0xFF
    xz_bad.write_bytes(bytes(raw_xz))
    with pytest.raises(CompactResumeRefused):
        load_ledger(resume_ledger_path, ctrl.to_dict(), input_root=bad_root)

    # Valid resume from compact stage-0001 + raw stage-0002 -> skips stages 1 & 2, runs 3 & 4!
    resumed_out = tmp_path / "attempt2_out"
    rc2, summary_resumed = run_ct_experiment(
        ctrl,
        b8_checkpoint=b8_p,
        frp3_checkpoint=frp3_p,
        champion_checkpoint=champ_p,
        output=resumed_out,
        device="cpu",
        ledger_path=resume_ledger_path,
        input_root=input_root,
    )
    assert rc2 == 0
    assert summary_resumed["resumed_stages"] == [1, 2]
    assert not (resumed_out / "stage-0001.pt").exists()
    assert not (resumed_out / "stage-0002.pt").exists()
    assert (resumed_out / "stage-0003.pt").is_file()
    assert (resumed_out / "stage-0004.pt").is_file()
    assert summary_resumed["verdict"] == summary_full["verdict"]
    assert summary_resumed["evaluation"]["gates"] == summary_full["evaluation"]["gates"]


def test_incomplete_support_stops_before_branch_scoring_and_blocks_continuation(
    tmp_path: Path,
) -> None:
    b8_p, frp3_p, champ_p = _save_tiny_checkpoints(tmp_path)
    ctrl = CTControl(
        source_commit="b" * 40,
        worlds=6,
        collection_ticks=8,
        max_body_capacity=128,
        min_held_out_deaths=32,  # Impossible in 6 worlds x 8 ticks -> triggers INCOMPLETE_SUPPORT
        scoring_chunks=2,
        tiny=True,
    )
    out = tmp_path / "inc_out"
    rc, summary = run_ct_experiment(
        ctrl,
        b8_checkpoint=b8_p,
        frp3_checkpoint=frp3_p,
        champion_checkpoint=champ_p,
        output=out,
        device="cpu",
    )
    assert rc == 0
    assert summary["verdict"] == "INCOMPLETE_SUPPORT"
    assert (out / "stage-0001.pt").is_file()
    assert not (out / "stage-0002.pt").exists()

    # Scientific-fault precedence: even if summary.json is missing under RPC_EXEC_POLL_TIMEOUT,
    # stage-0001.json's support_status="INCOMPLETE_SUPPORT" MUST refuse continuation!
    only_stage1 = tmp_path / "only_stage1"
    only_stage1.mkdir()
    for ext in (".pt", ".pt.sha256", ".pt.complete", ".json", ".json.sha256", ".json.complete"):
        src = out / f"stage-0001{ext}"
        (only_stage1 / src.name).write_bytes(src.read_bytes())

    manifest = {"job_id": "job-inc", "commit": "b" * 40, "inputs": []}
    m_sha = "c" * 64
    timeout_receipt = {
        "job_id": "job-inc",
        "commit": "b" * 40,
        "manifest_sha256": m_sha,
        "status": "INCOMPLETE",
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "stop_verified": True,
        "cleanup_error": None,
    }
    with pytest.raises(CTResumeRefused, match="INCOMPLETE_SUPPORT"):
        completion_packet(
            manifest,
            timeout_receipt,
            only_stage1,
            ctrl.to_dict(),
            completion_index=1,
            job_id="job-inc-2",
            output_dir=tmp_path / "next",
            original_manifest_sha256=m_sha,
        )


def test_scientific_fault_precedence_over_rpc_timeout(tmp_path: Path) -> None:
    manifest = {"job_id": "job-1", "commit": "d" * 40}
    m_sha = "e" * 64
    timeout_receipt = {
        "job_id": "job-1",
        "commit": "d" * 40,
        "manifest_sha256": m_sha,
        "status": "INCOMPLETE",
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "stop_verified": True,
        "cleanup_error": None,
    }

    # Empty 0-byte faulthandler-stacks.log created at startup is allowed on clean mechanical stop.
    clean_dir = tmp_path / "clean_ct"
    (clean_dir / "stall-watchdog").mkdir(parents=True)
    (clean_dir / "stall-watchdog/faulthandler-stacks.log").write_bytes(b"")
    verify_mechanical_stop(timeout_receipt, manifest, m_sha, export_root=clean_dir)

    # Non-empty faulthandler-stacks.log MUST refuse continuation even under RPC_EXEC_POLL_TIMEOUT!
    stall_dir = tmp_path / "stall_ct"
    (stall_dir / "stall-watchdog").mkdir(parents=True)
    (stall_dir / "stall-watchdog/faulthandler-stacks.log").write_text("Thread 0x01 stack...\n")
    with pytest.raises(CTResumeRefused, match="faulthandler stack dump"):
        verify_mechanical_stop(timeout_receipt, manifest, m_sha, export_root=stall_dir)

    # fault.json MUST refuse continuation even under RPC_EXEC_POLL_TIMEOUT!
    fault_dir = tmp_path / "fault_ct"
    fault_dir.mkdir()
    (fault_dir / "fault.json").write_text('{"status": "INVALID"}\n')
    with pytest.raises(CTResumeRefused, match="fault diagnostic present"):
        verify_mechanical_stop(timeout_receipt, manifest, m_sha, export_root=fault_dir)

    # summary.json MUST refuse continuation even under RPC_EXEC_POLL_TIMEOUT!
    sum_dir = tmp_path / "sum_ct"
    sum_dir.mkdir()
    (sum_dir / "summary.json").write_text('{"verdict": "STOP"}\n')
    with pytest.raises(CTResumeRefused, match="scientific summary already present"):
        verify_mechanical_stop(timeout_receipt, manifest, m_sha, export_root=sum_dir)


def test_sigusr1_graceful_fence_under_15s(tmp_path: Path) -> None:
    b8_p, frp3_p, champ_p = _save_tiny_checkpoints(tmp_path)
    ctrl = CTControl(
        source_commit="f" * 40,
        worlds=6,
        collection_ticks=200,
        max_body_capacity=128,
        scoring_chunks=2,
        tiny=True,
    )
    ctrl_path = tmp_path / "control.json"
    ctrl_path.write_text(json.dumps(ctrl.to_dict()))
    out_dir = tmp_path / "sig_out"

    env = dict(os.environ)
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    ):
        env[k] = "1"
    env["SNAKE_DQN_DEVICE"] = "cpu"

    proc = subprocess.Popen(
        [
            "nice",
            "-n",
            "15",
            sys.executable,
            "-m",
            "research.sandbox_a_resident.ct_run",
            "--checkpoint",
            str(b8_p),
            "--frp3",
            str(frp3_p),
            "--champion",
            str(champ_p),
            "--control",
            str(ctrl_path),
            "--output",
            str(out_dir),
            "--device",
            "cpu",
            "--test-delay-per-heartbeat",
            "0.04",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait until startup self-checks pass and Stage-1 collection begins.
        deadline = time.monotonic() + 25.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            prog_file = out_dir / "progress.json"
            if prog_file.is_file():
                try:
                    data = json.loads(prog_file.read_text())
                    if data.get("stage") in ("stage-0001-collection-start", "collection-progress"):
                        ready = True
                        break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.05)
        assert ready, f"Subprocess exited or did not reach collection: rc={proc.returncode}"

        t_sig = time.monotonic()
        proc.send_signal(signal.SIGUSR1)
        out_text, _ = proc.communicate(timeout=12.0)
        elapsed = time.monotonic() - t_sig

        assert proc.returncode == SIGUSR1_EXIT_CODE, f"Unexpected rc={proc.returncode}: {out_text}"
        assert elapsed < 15.0, f"SIGUSR1 shutdown exceeded 15s grace: {elapsed:.3f}s"
        marker = out_dir / "sigusr1-marker.json"
        assert marker.is_file()
        assert (out_dir / "sigusr1-marker.json.sha256").is_file()
        assert (out_dir / "sigusr1-marker.json.complete").is_file()
        m_payload = json.loads(marker.read_text())
        assert m_payload["status"] == "PREEMPTED_SIGUSR1"
        assert m_payload["within_15s_sla"] is True
        assert m_payload["allocation"] == allocation(ctrl.to_dict())
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_admission_exclusivity_for_ct_and_train01(tmp_path: Path) -> None:
    base = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": STUDY,
        "inputs": [
            {
                "kind": "worlds",
                "study": STUDY,
                "indices": [0],
                "seeds": [12345],
            },
            {
                "kind": "ct-export-derivative",
                "path": "exports/stage-0001.pt.xz",
                "sha256": "a" * 64,
            },
        ],
        "salt": "a100-dev/t34-cf-1028/",
        "endpoint": {"kind": "steps", "value": 39_485_440},
        "timeout": {"base_seconds": 43_200, "seconds": 86_400, "purpose": "runaway-guard"},
        "excluded_studies": ["F5", "F4-fold-0", "R30", "strict", "probe", "serving"],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        "outputs": {
            "label": "simulator-D",
            "paths": ["/tmp/ct-out"],
            "broker_manifest": {"path": "/tmp/broker.json", "sha256": "b" * 64},
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    assert validate_manifest(base)["study"] == STUDY

    # ct-export-derivative must be refused for sandbox-a-train01
    wrong_study = {
        **base,
        "study": "sandbox-a-train01",
        "inputs": [
            {"kind": "worlds", "study": "sandbox-a-train01", "indices": [0], "seeds": [12345]},
            base["inputs"][1],
        ],
    }
    with pytest.raises(ManifestError, match="exclusive to t34-cf-1028"):
        validate_manifest(wrong_study)

    # Registry validate_input for ct-export, ct-export-sidecar, ct-export-derivative
    stage_pt = tmp_path / "stage-0001.pt"
    stage_pt.write_bytes(b"stage-1-bytes")
    pt_sha = hashlib.sha256(stage_pt.read_bytes()).hexdigest()
    (tmp_path / "stage-0001.pt.sha256").write_text(pt_sha + "\n")
    (tmp_path / "stage-0001.pt.complete").write_text("complete\n")
    validate_input({"kind": "ct-export", "path": str(stage_pt), "sha256": pt_sha})
    sidecar = tmp_path / "stage-0001.pt.complete"
    validate_input({"kind": "ct-export-sidecar", "path": str(sidecar), "sha256": digest(sidecar)})
    deriv = tmp_path / "stage-0001.pt.xz"
    deriv.write_bytes(b"xz-bytes")
    d_sha = digest(deriv)
    (tmp_path / "stage-0001.pt.xz.sha256").write_text(d_sha + "\n")
    (tmp_path / "stage-0001.pt.xz.complete").write_text("complete\n")
    validate_input({"kind": "ct-export-derivative", "path": str(deriv), "sha256": d_sha})


def test_qualified_mechanical_stop_receipts_and_fault_precedence(tmp_path: Path) -> None:
    """Fix 4 regression: admit normal qualified timeout/SIGUSR1 receipts; refuse faults."""
    manifest = {"job_id": "sandbox-ct-cf-qual-01", "commit": "d" * 40}
    m_sha = "e" * 64
    clean_dir = tmp_path / "clean_exports"
    (clean_dir / "stall-watchdog").mkdir(parents=True)
    (clean_dir / "stall-watchdog/faulthandler-stacks.log").write_bytes(b"")

    def _base_receipt(**overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "job_id": "sandbox-ct-cf-qual-01",
            "commit": "d" * 40,
            "manifest_sha256": m_sha,
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "generation": "gen-01",
        }
        base.update(overrides)
        return base

    # 1) Normal qualified runtime wall-timeout / SIGUSR1 exit 138 -> CHILD_NONZERO_EXIT
    #    with terminal.status="failed", reason="timeout", returncode=138, plus ct_job.sh's
    #    normal STALL_WATCHDOG_DIAGNOSTICS exit footer.
    for mech_reason in ("timeout", "lease_expired", "release_credential_expiring"):
        qualified_sigusr1 = _base_receipt(
            failure_class="CHILD_NONZERO_EXIT",
            error=(
                "remote child exited 138: "
                "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/out/ct/stall-watchdog"
            ),
            terminal={
                "job_id": "sandbox-ct-cf-qual-01",
                "generation": "gen-01",
                "status": "failed",
                "reason": mech_reason,
                "returncode": 138,
                "last_traceback_line": (
                    "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/out/ct/stall-watchdog"
                ),
            },
        )
        verify_mechanical_stop(qualified_sigusr1, manifest, m_sha, export_root=clean_dir)

    # 2) RPC_EXEC_POLL_TIMEOUT both without terminal and with mechanical terminal.status="failed".
    verify_mechanical_stop(
        _base_receipt(failure_class="RPC_EXEC_POLL_TIMEOUT"),
        manifest,
        m_sha,
        export_root=clean_dir,
    )
    verify_mechanical_stop(
        _base_receipt(
            failure_class="RPC_EXEC_POLL_TIMEOUT",
            terminal={
                "job_id": "sandbox-ct-cf-qual-01",
                "generation": "gen-01",
                "status": "failed",
                "reason": "timeout",
                "returncode": 138,
            },
        ),
        manifest,
        m_sha,
        export_root=clean_dir,
    )

    # 3) Mechanical SUPERVISOR stops (wall/unit/lease/read-retry or mechanical terminal).
    for sup_err in (
        "whole-session wall cap reached",
        "priced session unit cap reached",
        "retention lease elapsed",
        "read retry budget exhausted",
    ):
        verify_mechanical_stop(
            _base_receipt(failure_class="SUPERVISOR", error=sup_err),
            manifest,
            m_sha,
            export_root=clean_dir,
        )

    # 4) Refuse invented / non-qualified failure classes previously allowlisted by mistake.
    for bogus_cls in ("SESSION_LOST", "WALL_CLOCK_TIMEOUT", "SIGUSR1_PREEMPTED"):
        with pytest.raises(CTResumeRefused, match="not a qualified mechanical trigger"):
            verify_mechanical_stop(
                _base_receipt(failure_class=bogus_cls),
                manifest,
                m_sha,
                export_root=clean_dir,
            )

    # 5) Refuse CHILD_NONZERO_EXIT without mechanical terminal reason or with watchdog rc=124.
    with pytest.raises(CTResumeRefused, match="non-mechanical child failure"):
        verify_mechanical_stop(
            _base_receipt(
                failure_class="CHILD_NONZERO_EXIT",
                terminal={
                    "job_id": "sandbox-ct-cf-qual-01",
                    "generation": "gen-01",
                    "status": "failed",
                    "reason": None,
                    "returncode": 1,
                },
            ),
            manifest,
            m_sha,
            export_root=clean_dir,
        )
    with pytest.raises(CTResumeRefused, match="watchdog timeout exit code 124"):
        verify_mechanical_stop(
            _base_receipt(
                failure_class="CHILD_NONZERO_EXIT",
                terminal={
                    "job_id": "sandbox-ct-cf-qual-01",
                    "generation": "gen-01",
                    "status": "failed",
                    "reason": "timeout",
                    "returncode": 124,
                },
            ),
            manifest,
            m_sha,
            export_root=clean_dir,
        )

    # 6) Scientific/runtime fault evidence in poll-*.json or supervisor log overrides timeout.
    poll_fault_dir = tmp_path / "poll_fault_out" / "exports" / "ct"
    poll_fault_dir.mkdir(parents=True)
    (tmp_path / "poll_fault_out" / "poll-0001.json").write_text(
        json.dumps(
            {
                "status": "running",
                "job_log_tail": "Traceback (most recent call last):\nRuntimeError: boom\n",
            }
        )
    )
    with pytest.raises(CTResumeRefused, match="explicit scientific/non-mechanical fault"):
        verify_mechanical_stop(
            _base_receipt(failure_class="RPC_EXEC_POLL_TIMEOUT"),
            manifest,
            m_sha,
            export_root=poll_fault_dir,
        )

    sup_log = tmp_path / "supervisor-fault.log"
    sup_log.write_text("INFO starting\nCUDA out of memory while allocating tensor\n")
    with pytest.raises(CTResumeRefused, match="cuda out of memory"):
        verify_mechanical_stop(
            _base_receipt(failure_class="RPC_EXEC_POLL_TIMEOUT"),
            manifest,
            m_sha,
            export_root=clean_dir,
            supervisor_log_path=sup_log,
        )


def _write_tiny_initial_packet(
    packet_dir: Path,
    ctrl: CTControl,
    checkpoints: tuple[Path, Path, Path],
    output_dir: Path,
) -> None:
    """Create a 6-file initial tiny packet directory for write_continuation_packet testing."""
    b8_p, frp3_p, champ_p = checkpoints
    control = ctrl.to_dict()
    allocations = ct_seed_allocations(ctrl)
    check = audit_ct_seeds(allocations, DEFAULT_REGISTRY)
    packet_dir.mkdir(parents=True, exist_ok=False)
    write_once(packet_dir / "control.json", control)
    write_once(packet_dir / "registry-check.json", check)
    write_once(packet_dir / "topology-manifest.json", dict(TOPOLOGY_SPEC))
    write_once(packet_dir / "verified-ledger.json", empty_ledger(control))
    uploads = [
        {"path": str(b8_p.resolve()), "dest": "b8-s3.pth", "sha256": digest(b8_p)},
        {"path": str(frp3_p.resolve()), "dest": "frp3-s12.pth", "sha256": digest(frp3_p)},
        {"path": str(champ_p.resolve()), "dest": "champion.pth", "sha256": digest(champ_p)},
        {
            "path": str(DEFAULT_REGISTRY.resolve()),
            "dest": "registry.json",
            "sha256": digest(DEFAULT_REGISTRY),
        },
        {
            "path": str((packet_dir / "verified-ledger.json").resolve()),
            "dest": "verified-ledger.json",
            "sha256": digest(packet_dir / "verified-ledger.json"),
        },
    ]
    manifest = {
        "job_id": "tiny-ct-00",
        "commit": ctrl.source_commit,
        "script": "research/sandbox_a_resident/ct_job.sh",
        "device": "cuda",
        "source_paths": ["research/sandbox_a_resident/ct_run.py"],
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 2,
            "est_minutes": 720,
        },
        "timeout_minutes": 1440,
        "output_dir": str(output_dir.resolve()),
        "max_export_bytes": 1024**3,
        "inputs": uploads,
        "env": {
            "SANDBOX_CT_CONTROL_JSON": json.dumps(control, sort_keys=True),
            "SANDBOX_A_SOURCE_COMMIT": ctrl.source_commit,
            "SNAKE_DQN_DEVICE": "cuda",
            "MAX_JOBS": "1",
            "COLAB_CLI_REQUIREMENTS": "research/sandbox_a_resident/requirements.txt",
            "SANDBOX_A_STALL_TIMEOUT_SECONDS": "900",
        },
        "required_exports": ["ct/summary.json"],
        "budget": {
            "rate_units_per_hour": 8.0,
            "unit_cap": 193.0,
            "reserve_units": 0,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "ct-stage-snapshot",
            "ledger_path": "verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "sandbox_admission": {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str((packet_dir / "sandbox.json").resolve()),
        },
        "sandbox_execution": "tiny C/T continuation test",
        "requirements_file": "research/sandbox_a_resident/requirements.txt",
        "sandbox_a": {
            "seed_allocations": allocations,
            "seed_audit_sha256": digest(packet_dir / "registry-check.json"),
            "topology_manifest_sha256": digest(packet_dir / "topology-manifest.json"),
        },
        "sandbox_source_delivery": "background-sparse-v1",
    }
    write_once(packet_dir / "dispatch-manifest.json", manifest)
    roots = [row["offset"] for row in allocations]
    sb_inputs = [
        {
            "kind": "worlds",
            "study": STUDY,
            "indices": roots,
            "seeds": [world_seed(STUDY, i) for i in roots],
        },
        {"kind": "frozen-checkpoint", "path": uploads[0]["path"], "sha256": uploads[0]["sha256"]},
        {"kind": "frozen-opponent", "path": uploads[1]["path"], "sha256": uploads[1]["sha256"]},
        {"kind": "frozen-opponent", "path": uploads[2]["path"], "sha256": uploads[2]["sha256"]},
        {"kind": "registry", "path": uploads[3]["path"], "sha256": uploads[3]["sha256"]},
        {"kind": "verified-ledger", "path": uploads[4]["path"], "sha256": uploads[4]["sha256"]},
    ]
    sandbox = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": STUDY,
        "inputs": sb_inputs,
        "salt": f"{CT_DOMAIN_PREFIX}/",
        "endpoint": {"kind": "steps", "value": ct_endpoint_steps(ctrl)},
        "timeout": {"base_seconds": 43_200, "seconds": 86_400, "purpose": "runaway-guard"},
        "excluded_studies": [
            "F5",
            "F4-fold-0",
            "29-census-parents",
            "R30",
            "strict",
            "probe",
            "serving",
            "D4096",
            "t34-round0-study-01",
            "sandbox-a-train01-monitor",
            "sandbox-a-train01-nomination",
        ],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        "outputs": {
            "label": "simulator-D",
            "paths": [str(output_dir.resolve())],
            "broker_manifest": {
                "path": str((packet_dir / "dispatch-manifest.json").resolve()),
                "sha256": digest(packet_dir / "dispatch-manifest.json"),
            },
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    lines = [
        ("{" if i == 0 else " ") + json.dumps(k) + ": " + json.dumps(v) + ("," if i < 9 else "}")
        for i, (k, v) in enumerate(sandbox.items())
    ]
    (packet_dir / "sandbox.json").write_text("\n".join(lines) + "\n")


def _stage_manifest_inputs(manifest: dict[str, object], input_root: Path) -> Path:
    """Materialize every manifest['inputs'] entry into input_root just as runtime staging does."""
    input_root.mkdir(parents=True, exist_ok=True)
    inputs = manifest["inputs"]
    assert isinstance(inputs, list)
    for item in inputs:
        assert isinstance(item, dict)
        src = Path(str(item["path"]))
        dst = input_root / str(item["dest"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        assert digest(dst) == item["sha256"]
    ledger_dst = input_root / "verified-ledger.json"
    assert ledger_dst.is_file()
    return ledger_dst


def test_two_successive_completions_preserve_union_and_pass_production_admission_dry_run(
    tmp_path: Path,
) -> None:
    """Fix 5 regression: two successive completions preserve stage union and pass dry-run."""
    # --- Part A: Numerical 2-completion chain without recomputation ---
    ckpts = _save_tiny_checkpoints(tmp_path)
    b8_p, frp3_p, champ_p = ckpts
    ctrl = CTControl(
        source_commit="a" * 40,
        worlds=6,
        collection_ticks=12,
        max_body_capacity=128,
        max_deaths=2,
        lookbacks=(2, 4),
        max_ordinary_roots=4,
        anchor_rows=16,
        min_death_length=1,
        min_held_out_deaths=0,
        min_held_out_death_worlds=0,
        min_held_out_ordinary_roots=1,
        min_held_out_anchor_rows=1,
        branch_horizon=3,
        minibatches=2,
        searched_batch_size=2,
        anchor_batch_size=4,
        bootstrap_resamples=50,
        scoring_chunks=2,
        branch_batch_size=16,
        snapshot_store_max_bytes=16 * 1024 * 1024,
        tiny=True,
    )

    full_out = tmp_path / "tiny_full"
    init_ledger = tmp_path / "tiny_empty_ledger.json"
    init_ledger.write_text(json.dumps(empty_ledger(ctrl.to_dict())))
    rc0, summary_full = run_ct_experiment(
        ctrl,
        b8_checkpoint=b8_p,
        frp3_checkpoint=frp3_p,
        champion_checkpoint=champ_p,
        output=full_out,
        device="cpu",
        ledger_path=init_ledger,
    )
    assert rc0 == 0

    packet_00 = tmp_path / "tiny-packet-00"
    _write_tiny_initial_packet(packet_00, ctrl, ckpts, tmp_path / "tiny_a0_out")

    # Attempt 1 exports ONLY stage-0001 before qualified SIGUSR1 wall timeout.
    attempt1_exports = tmp_path / "tiny_a0_out" / "exports" / "ct"
    attempt1_exports.mkdir(parents=True)
    for ext in (".pt", ".pt.sha256", ".pt.complete", ".json", ".json.sha256", ".json.complete"):
        src = full_out / f"stage-0001{ext}"
        (attempt1_exports / src.name).write_bytes(src.read_bytes())

    receipt_1 = {
        "job_id": "tiny-ct-00",
        "commit": ctrl.source_commit,
        "manifest_sha256": digest(packet_00 / "dispatch-manifest.json"),
        "status": "INCOMPLETE",
        "failure_class": "CHILD_NONZERO_EXIT",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": "gen-a0",
        "terminal": {
            "job_id": "tiny-ct-00",
            "generation": "gen-a0",
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        },
    }
    packet_c1 = tmp_path / "tiny-packet-c1"
    c1_info = write_continuation_packet(
        packet_c1,
        prior_packet_dir=packet_00,
        terminal_receipt=receipt_1,
        export_root=attempt1_exports,
        completion_index=1,
        job_id="tiny-ct-c1",
        output_dir=tmp_path / "tiny_c1_out",
        use_compact_xz_for_stages=(1,),
        auto_encode_compact_xz=True,
        validate_registry_inputs=False,
    )
    assert c1_info["verified_stages"] == [1]
    c1_manifest = json.loads((packet_c1 / "dispatch-manifest.json").read_text())
    assert c1_manifest["sandbox_admission"]["manifest_path"] == str(
        (packet_c1 / "sandbox.json").resolve()
    )

    # Execute Completion 1 from staged manifest inputs ONLY (no manual ledger wiring).
    vm_c1_inputs = tmp_path / "vm_c1_inputs"
    c1_ledger_staged = _stage_manifest_inputs(c1_manifest, vm_c1_inputs)
    c1_run_out = tmp_path / "tiny_c1_full_run"
    rc_c1, _ = run_ct_experiment(
        ctrl,
        b8_checkpoint=vm_c1_inputs / "b8-s3.pth",
        frp3_checkpoint=vm_c1_inputs / "frp3-s12.pth",
        champion_checkpoint=vm_c1_inputs / "champion.pth",
        output=c1_run_out,
        device="cpu",
        ledger_path=c1_ledger_staged,
        input_root=vm_c1_inputs,
    )
    assert rc_c1 == 0
    assert not (c1_run_out / "stage-0001.pt").exists()
    assert (c1_run_out / "stage-0002.pt").is_file()

    # Completion 1 times out after exporting ONLY stage-0002 (no stage-0001 in attempt2_exports!).
    attempt2_exports = tmp_path / "tiny_c1_out" / "exports" / "ct"
    attempt2_exports.mkdir(parents=True)
    for ext in (".pt", ".pt.sha256", ".pt.complete", ".json", ".json.sha256", ".json.complete"):
        src = c1_run_out / f"stage-0002{ext}"
        (attempt2_exports / src.name).write_bytes(src.read_bytes())

    # Single-directory scan of attempt2_exports alone lacks stage-0001, as Astra observed.
    with pytest.raises(CTResumeRefused, match="without verified stage-0001"):
        build_stage_ledger(attempt2_exports, ctrl.to_dict())

    # Cross-attempt union [attempt1_exports, attempt2_exports] preserves stages (1, 2)!
    receipt_2 = {
        "job_id": "tiny-ct-c1",
        "commit": ctrl.source_commit,
        "manifest_sha256": digest(packet_c1 / "dispatch-manifest.json"),
        "status": "INCOMPLETE",
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": "gen-c1",
        "terminal": {
            "job_id": "tiny-ct-c1",
            "generation": "gen-c1",
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        },
    }
    packet_c2 = tmp_path / "tiny-packet-c2"
    c2_info = write_continuation_packet(
        packet_c2,
        prior_packet_dir=packet_c1,
        terminal_receipt=receipt_2,
        export_root=[attempt1_exports, attempt2_exports],
        completion_index=2,
        job_id="tiny-ct-c2",
        output_dir=tmp_path / "tiny_c2_out",
        use_compact_xz_for_stages=(1,),
        auto_encode_compact_xz=True,
        validate_registry_inputs=False,
    )
    assert c2_info["verified_stages"] == [1, 2]
    c2_manifest = json.loads((packet_c2 / "dispatch-manifest.json").read_text())
    assert c2_manifest["sandbox_admission"]["manifest_path"] == str(
        (packet_c2 / "sandbox.json").resolve()
    )

    # Execute Completion 2 from staged manifest inputs ONLY -> skips [1, 2], runs [3, 4].
    vm_c2_inputs = tmp_path / "vm_c2_inputs"
    c2_ledger_staged = _stage_manifest_inputs(c2_manifest, vm_c2_inputs)
    c2_run_out = tmp_path / "tiny_c2_out"
    rc_c2, summary_c2 = run_ct_experiment(
        ctrl,
        b8_checkpoint=vm_c2_inputs / "b8-s3.pth",
        frp3_checkpoint=vm_c2_inputs / "frp3-s12.pth",
        champion_checkpoint=vm_c2_inputs / "champion.pth",
        output=c2_run_out,
        device="cpu",
        ledger_path=c2_ledger_staged,
        input_root=vm_c2_inputs,
    )
    assert rc_c2 == 0
    assert summary_c2["resumed_stages"] == [1, 2]
    assert not (c2_run_out / "stage-0001.pt").exists()
    assert not (c2_run_out / "stage-0002.pt").exists()
    assert (c2_run_out / "stage-0003.pt").is_file()
    assert (c2_run_out / "stage-0004.pt").is_file()
    assert summary_c2["verdict"] == summary_full["verdict"]
    assert summary_c2["evaluation"]["gates"] == summary_full["evaluation"]["gates"]

    # --- Part B: Production committed admission + dispatch.sh --dry-run on c1 & c2 ---
    canonical_ckpt_dir = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
    if (
        os.environ.get("CT_PREFLIGHT_SUBPROCESS") == "1"
        or not (canonical_ckpt_dir / "b8_s3_t7_20261016.pth").is_file()
    ):
        return

    ws = tmp_path / "ws"
    broker_wt = ws / "broker"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(broker_wt)],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(broker_wt),
                "sparse-checkout",
                "set",
                "--no-cone",
                "/src/",
                "/tools/",
                "/research/colab_cli/",
                "/research/compute/",
                "/research/sandbox_a_resident/",
                "/research/sandbox_b_heads/",
                "/research/a100_a2/model.py",
                "/research/bl_labels/__init__.py",
                "/research/bl_labels/exports.py",
                "/research/bl_residual/",
                "/research/gpu_sim_spike/",
                "/docs/research/governance_a100_sandbox_20261026/",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(broker_wt), "checkout", "-q", "HEAD"], check=True)
        subprocess.run(
            ["git", "-C", str(broker_wt), "config", "user.name", "CT Continuation Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(broker_wt), "config", "user.email", "ct@test.invalid"],
            check=True,
        )

        # Overlay current working-tree C/T package files so uncommitted fixes are tested live.
        src_pkg = ROOT / "research/sandbox_a_resident"
        dst_pkg = broker_wt / "research/sandbox_a_resident"
        for name in (
            *[p.name for p in src_pkg.glob("ct_*.py")],
            *[p.name for p in src_pkg.glob("test_ct_*.py")],
            "ct_job.sh",
            "requirements.txt",
        ):
            src_f = src_pkg / name
            if src_f.is_file():
                dst_f = dst_pkg / name
                dst_f.write_bytes(src_f.read_bytes())
                dst_f.chmod(src_f.stat().st_mode)

        for rel_pkt in ("packet-train01-06", "packet-train01-09"):
            mf = dst_pkg / rel_pkt / "dispatch-manifest.json"
            if mf.is_file():
                mf.write_text(
                    mf.read_text().replace(
                        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/",
                        f"{ws}/snake-dqn-artifacts/",
                    )
                )

        subprocess.run(
            ["git", "-C", str(broker_wt), "add", "research/sandbox_a_resident"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(broker_wt),
                "commit",
                "--allow-empty",
                "-qm",
                "test ct impl pin",
            ],
            check=True,
        )
        impl_pin = subprocess.check_output(
            ["git", "-C", str(broker_wt), "rev-parse", "HEAD"], text=True
        ).strip()

        art_root = ws / "snake-dqn-artifacts/t34-cf-1028"
        prod_p0 = dst_pkg / "packet-test-ct-00"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.sandbox_a_resident.ct_prepare",
                "--packet",
                str(prod_p0),
                "--commit",
                impl_pin,
                "--checkpoint-dir",
                str(canonical_ckpt_dir),
                "--job-id",
                "sandbox-ct-cf-test-p0",
                "--output-dir",
                str(art_root / "attempt-00/output"),
                "--sandbox-source-delivery",
                "background-sparse-v1",
                "--stall-timeout-seconds",
                "900",
            ],
            cwd=broker_wt,
            check=True,
        )

        prod_ctrl = json.loads((prod_p0 / "control.json").read_text())
        exp_a0 = art_root / "attempt-00-prior/exports/ct"
        exp_c1 = art_root / "attempt-c1-prior/exports/ct"
        publish_stage_snapshot(
            exp_a0,
            1,
            lambda dst: torch.save({"kind": "collection", "marker": torch.tensor([1])}, dst),
            metadata={
                "kind": "collection",
                "allocation": allocation(prod_ctrl),
                "source_commit": impl_pin,
                "support_status": "PASS",
            },
        )
        publish_stage_snapshot(
            exp_c1,
            2,
            lambda dst: torch.save({"kind": "scoring_chunk", "chunk": torch.tensor([0])}, dst),
            metadata={
                "kind": "scoring_chunk",
                "chunk_idx": 0,
                "allocation": allocation(prod_ctrl),
                "source_commit": impl_pin,
            },
        )

        rec_p0 = ws / "receipt-p0.json"
        rec_p0.write_text(
            json.dumps(
                {
                    "job_id": "sandbox-ct-cf-test-p0",
                    "commit": impl_pin,
                    "manifest_sha256": digest(prod_p0 / "dispatch-manifest.json"),
                    "status": "INCOMPLETE",
                    "failure_class": "CHILD_NONZERO_EXIT",
                    "stop_verified": True,
                    "cleanup_error": None,
                    "generation": "gen-p0",
                    "terminal": {
                        "job_id": "sandbox-ct-cf-test-p0",
                        "generation": "gen-p0",
                        "status": "failed",
                        "reason": "timeout",
                        "returncode": 138,
                    },
                }
            )
        )
        prod_c1 = dst_pkg / "packet-test-ct-c1"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.sandbox_a_resident.ct_resume",
                "--packet-dir",
                str(prod_c1),
                "--prior-packet-dir",
                str(prod_p0),
                "--receipt",
                str(rec_p0),
                "--export-dir",
                str(exp_a0),
                "--completion-index",
                "1",
                "--job-id",
                "sandbox-ct-cf-test-c1",
                "--output-dir",
                str(art_root / "attempt-c1/output"),
                "--compact-stage",
                "1",
            ],
            cwd=broker_wt,
            check=True,
        )

        rec_c1 = ws / "receipt-c1.json"
        rec_c1.write_text(
            json.dumps(
                {
                    "job_id": "sandbox-ct-cf-test-c1",
                    "commit": impl_pin,
                    "manifest_sha256": digest(prod_c1 / "dispatch-manifest.json"),
                    "status": "INCOMPLETE",
                    "failure_class": "RPC_EXEC_POLL_TIMEOUT",
                    "stop_verified": True,
                    "cleanup_error": None,
                    "generation": "gen-c1",
                    "terminal": {
                        "job_id": "sandbox-ct-cf-test-c1",
                        "generation": "gen-c1",
                        "status": "failed",
                        "reason": "timeout",
                        "returncode": 138,
                    },
                }
            )
        )
        prod_c2 = dst_pkg / "packet-test-ct-c2"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.sandbox_a_resident.ct_resume",
                "--packet-dir",
                str(prod_c2),
                "--prior-packet-dir",
                str(prod_c1),
                "--receipt",
                str(rec_c1),
                "--export-dir",
                str(exp_a0),
                "--export-dir",
                str(exp_c1),
                "--completion-index",
                "2",
                "--job-id",
                "sandbox-ct-cf-test-c2",
                "--output-dir",
                str(art_root / "attempt-c2/output"),
                "--compact-stage",
                "1",
            ],
            cwd=broker_wt,
            check=True,
        )

        subprocess.run(
            [
                "git",
                "-C",
                str(broker_wt),
                "add",
                str(prod_p0),
                str(prod_c1),
                str(prod_c2),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(broker_wt), "commit", "-qm", "seal c1 and c2 test packets"],
            check=True,
        )

        # Always run committed ct_preflight.validate_packet on both c1 and c2.
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from research.sandbox_a_resident.ct_preflight import validate_packet; "
                    f"validate_packet(Path({str(prod_c1)!r}), require_committed=True); "
                    f"validate_packet(Path({str(prod_c2)!r}), require_committed=True)"
                ),
            ],
            cwd=broker_wt,
            check=True,
        )

        # Run full tools/ops/dispatch.sh --dry-run on BOTH c1 and c2 at top level
        # (skip only inside nested ct_preflight subprocesses to avoid recursion).
        if os.environ.get("CT_PREFLIGHT_SUBPROCESS") != "1":
            ops_dir = ws / "ops"
            ops_dir.mkdir(parents=True)
            transport_tree = subprocess.check_output(
                ["git", "-C", str(broker_wt), "rev-parse", "HEAD:research/colab_cli"],
                text=True,
            ).strip()
            (ops_dir / "colab-cli-qualified.json").write_text(
                json.dumps(
                    {
                        "schema": "snake-dqn-colab-cli-qualified/v1",
                        "tree_hash": transport_tree,
                    }
                )
            )
            shims = ws / "shims"
            shims.mkdir()
            (shims / "python3").symlink_to(Path(sys.executable))
            (shims / "pmset").write_text("#!/bin/sh\necho \"Now drawing from 'AC Power'\"\n")
            (shims / "pmset").chmod(0o755)
            (shims / "ioreg").write_text("#!/bin/sh\necho '  \"AppleClamshellState\" = No'\n")
            (shims / "ioreg").chmod(0o755)
            (shims / "caffeinate").write_text('#!/bin/sh\nshift\nexec "$@"\n')
            (shims / "caffeinate").chmod(0o755)

            dry_env = dict(os.environ)
            dry_env["OPS_DIR"] = str(ops_dir)
            dry_env["SNAKE_DQN_PYTHON"] = str(Path(sys.executable))
            dry_env["PATH"] = f"{shims}:{os.environ['PATH']}"

            for name, pkt, st_dir in (
                ("sandbox-ct-cf-test-c1", prod_c1, art_root / "attempt-c1/state"),
                ("sandbox-ct-cf-test-c2", prod_c2, art_root / "attempt-c2/state"),
            ):
                res = subprocess.run(
                    [
                        str(broker_wt / "tools/ops/dispatch.sh"),
                        name,
                        str(pkt / "dispatch-manifest.json"),
                        "--broker",
                        str(broker_wt),
                        "--cwd",
                        str(broker_wt),
                        "--state-dir",
                        str(st_dir),
                        "--dry-run",
                    ],
                    cwd=broker_wt,
                    env=dry_env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode != 0:
                    diag = subprocess.run(
                        [
                            str(broker_wt / "tools/sandbox/launch.sh"),
                            "--state-dir",
                            str(ws / f"diag-{name}"),
                            "--colab",
                            str(Path.home() / ".local/bin/colab"),
                            "validate",
                            str(pkt / "dispatch-manifest.json"),
                        ],
                        cwd=broker_wt,
                        env=dry_env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    raise AssertionError(
                        f"dry-run failed for {name}: {res.stdout}\n{res.stderr}\n"
                        f"LAUNCH_VALIDATE_DIAG (rc={diag.returncode}):\n"
                        f"{diag.stdout}\n{diag.stderr}"
                    )
                assert f"PASS_DRY_RUN {name} allocations=0" in res.stdout
    finally:
        shutil.rmtree(ws, ignore_errors=True)
