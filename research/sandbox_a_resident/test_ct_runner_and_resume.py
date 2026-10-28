"""Unit tests for C/T seed audit, runner, compact XZ resume, SIGUSR1 <15s, and fault precedence."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident.ct_prepare import STUDY, audit_ct_seeds, ct_endpoint_steps
from research.sandbox_a_resident.ct_resume import (
    CTResumeRefused,
    completion_packet,
    verify_mechanical_stop,
)
from research.sandbox_a_resident.ct_run import SIGUSR1_EXIT_CODE, run_ct_experiment
from research.sandbox_a_resident.ct_task import (
    CTControl,
    ct_seed_allocations,
    train01_protected_eval_seeds,
    world_partition_table,
)
from research.sandbox_a_resident.ct_units import (
    CompactResumeRefused,
    allocation,
    empty_ledger,
    encode_compact_stage_snapshot,
    load_ledger,
    unit_id,
)
from research.sandbox_a_resident.units import digest
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
from tools.sandbox.manifest import ManifestError, validate_manifest
from tools.sandbox.registry import DEFAULT_REGISTRY, validate_input

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
        "failure_class": "SESSION_LOST",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": 1,
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
