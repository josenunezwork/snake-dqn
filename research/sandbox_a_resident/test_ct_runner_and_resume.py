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

    (partial_export / "exports/ct").mkdir(parents=True)
    orig_manifest = {
        "job_id": "sandbox-ct-cf-test-01",
        "output_dir": str(partial_export),
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
        "generation": "gen-1",
        "terminal": {
            "job_id": "sandbox-ct-cf-test-01",
            "generation": "gen-1",
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

    # Scientific-fault precedence: even if summary.json is missing under a host wall cap,
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
        "failure_class": "SUPERVISOR",
        "error": "whole-session wall cap reached",
        "stop_verified": True,
        "cleanup_error": None,
    }
    with pytest.raises(CTResumeRefused, match="(?i)incomplete_support"):
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


def test_scientific_fault_precedence_over_host_wall_cap(tmp_path: Path) -> None:
    manifest = {"job_id": "job-1", "commit": "d" * 40}
    m_sha = "e" * 64
    timeout_receipt = {
        "job_id": "job-1",
        "commit": "d" * 40,
        "manifest_sha256": m_sha,
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": "whole-session wall cap reached",
        "stop_verified": True,
        "cleanup_error": None,
    }

    # Empty 0-byte faulthandler-stacks.log created at startup is allowed on clean mechanical stop.
    clean_dir = tmp_path / "clean_ct"
    (clean_dir / "stall-watchdog").mkdir(parents=True)
    (clean_dir / "stall-watchdog/faulthandler-stacks.log").write_bytes(b"")
    verify_mechanical_stop(
        timeout_receipt, {**manifest, "output_dir": str(clean_dir)}, m_sha, export_root=clean_dir
    )

    # Non-empty faulthandler-stacks.log MUST refuse continuation even under a host wall cap!
    stall_dir = tmp_path / "stall_ct"
    (stall_dir / "stall-watchdog").mkdir(parents=True)
    (stall_dir / "stall-watchdog/faulthandler-stacks.log").write_text("Thread 0x01 stack...\n")
    with pytest.raises(CTResumeRefused, match="faulthandler stack dump"):
        verify_mechanical_stop(
            timeout_receipt,
            {**manifest, "output_dir": str(stall_dir)},
            m_sha,
            export_root=stall_dir,
        )

    # fault.json MUST refuse continuation even under a host wall cap!
    fault_dir = tmp_path / "fault_ct"
    fault_dir.mkdir()
    (fault_dir / "fault.json").write_text('{"status": "INVALID"}\n')
    with pytest.raises(CTResumeRefused, match="fault diagnostic present"):
        verify_mechanical_stop(
            timeout_receipt,
            {**manifest, "output_dir": str(fault_dir)},
            m_sha,
            export_root=fault_dir,
        )

    # summary.json MUST refuse continuation even under a host wall cap!
    sum_dir = tmp_path / "sum_ct"
    sum_dir.mkdir()
    (sum_dir / "summary.json").write_text('{"verdict": "STOP"}\n')
    with pytest.raises(CTResumeRefused, match="scientific summary already present"):
        verify_mechanical_stop(
            timeout_receipt, {**manifest, "output_dir": str(sum_dir)}, m_sha, export_root=sum_dir
        )


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
    manifest = {
        "job_id": "sandbox-ct-cf-qual-01",
        "commit": "d" * 40,
        "output_dir": str(tmp_path / "clean_exports"),
    }
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

    # RPC timeouts remain ambiguous provider failures, even with mechanical terminal text.
    for terminal in (
        None,
        {
            "job_id": "sandbox-ct-cf-qual-01",
            "generation": "gen-01",
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        },
    ):
        with pytest.raises(CTResumeRefused, match="not a qualified mechanical trigger"):
            verify_mechanical_stop(
                _base_receipt(failure_class="RPC_EXEC_POLL_TIMEOUT", terminal=terminal),
                manifest,
                m_sha,
                export_root=clean_dir,
            )

    # 3) Mechanical SUPERVISOR stops (wall/unit/lease/read-retry or mechanical terminal).
    for sup_err in (
        "whole-session wall cap reached",
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
        with pytest.raises(
            CTResumeRefused,
            match=(
                f"^prior receipt failure_class {bogus_cls!r} "
                "is not a qualified mechanical trigger$"
            ),
        ):
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
    manifest["output_dir"] = str(tmp_path / "poll_fault_out")
    with pytest.raises(CTResumeRefused, match="explicit scientific/non-mechanical fault"):
        verify_mechanical_stop(
            _base_receipt(failure_class="SUPERVISOR", error="whole-session wall cap reached"),
            manifest,
            m_sha,
            export_root=poll_fault_dir,
        )

    manifest["output_dir"] = str(clean_dir)
    sup_log = tmp_path / "supervisor-fault.log"
    sup_log.write_text("INFO starting\nCUDA out of memory while allocating tensor\n")
    with pytest.raises(CTResumeRefused, match="cuda out of memory"):
        verify_mechanical_stop(
            _base_receipt(failure_class="SUPERVISOR", error="whole-session wall cap reached"),
            manifest,
            m_sha,
            export_root=clean_dir,
            supervisor_log_path=sup_log,
        )


def _clean_supervisor_log(directory: Path, name: str = "supervisor.log") -> Path:
    path = directory / name
    if not path.exists():
        path.write_text("qualified transport stop observed\n")
    return path


def _seal_initial_packet(packet: Path) -> None:
    bindings = {
        "manifest_sha256": "dispatch-manifest.json",
        "sandbox_sha256": "sandbox.json",
        "control_sha256": "control.json",
        "ledger_sha256": "verified-ledger.json",
        "registry_audit_sha256": "registry-check.json",
        "topology_manifest_sha256": "topology-manifest.json",
    }
    write_once(
        packet / "SEAL.json",
        {
            "schema": "t34-cf-1028-prep-seal/v1",
            **{key: digest(packet / name) for key, name in bindings.items()},
        },
    )


def _rebind_forged_packet(packet: Path, broker: dict) -> None:
    """Keep forged packets complete so only the intended semantic guard refuses."""
    (packet / "dispatch-manifest.json").write_text(json.dumps(broker))
    sandbox_path = packet / "sandbox.json"
    sandbox = json.loads(sandbox_path.read_text())
    sandbox["outputs"]["broker_manifest"] = {
        "path": str(packet / "dispatch-manifest.json"),
        "sha256": digest(packet / "dispatch-manifest.json"),
    }
    uploads = {row["dest"]: row for row in broker["inputs"]}
    for row in sandbox["inputs"]:
        if row.get("kind") == "verified-ledger":
            row.update(
                path=uploads["verified-ledger.json"]["path"],
                sha256=uploads["verified-ledger.json"]["sha256"],
            )
    sandbox_path.write_text(json.dumps(sandbox))
    seal = json.loads((packet / "SEAL.json").read_text())
    seal.update(
        manifest_sha256=digest(packet / "dispatch-manifest.json"),
        ledger_sha256=digest(packet / "verified-ledger.json"),
    )
    (packet / "SEAL.json").write_text(json.dumps(seal))


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
    _seal_initial_packet(packet_dir)


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
        supervisor_log_path=_clean_supervisor_log(tmp_path),
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
        "failure_class": "SUPERVISOR",
        "error": "whole-session wall cap reached",
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
        supervisor_log_path=_clean_supervisor_log(tmp_path),
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

    retained_ws = os.environ.get("CT_DRY_RUN_WORKSPACE")
    ws = Path(retained_ws) if retained_ws else tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=False)
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
                "/tests/fixtures/ct-supervisor-06-exports/",
                "/tests/fixtures/ct-supervisor-06-ct-bound/",
                "/tests/fixtures/ct-supervisor-06-entrypoint/",
                "/tests/fixtures/ct-supervisor-06-main/",
                "/tests/fixtures/ct-supervisor-06-entrypoint-main/",
                "/tests/fixtures/ct-cuda-smoke-08/",
                "/tests/fixtures/ct-cuda-smoke-09/",
                "/tests/fixtures/ct-uv-packet11/",
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

        _seal_initial_packet(prod_p0)
        prod_ctrl = json.loads((prod_p0 / "control.json").read_text())
        exp_a0 = art_root / "attempt-00/output/exports/ct"
        exp_c1 = art_root / "attempt-c1/output/exports/ct"
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
            exp_a0,
            2,
            lambda dst: dst.write_bytes(b"prior verified stage 2"),
            metadata={"allocation": allocation(prod_ctrl), "source_commit": impl_pin},
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
                "--supervisor-log",
                str(_clean_supervisor_log(ws, "supervisor-p0.log")),
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
                        "independent_review_status": "ACCEPT",
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

            subprocess.run(
                ["git", "-C", str(broker_wt), "add", str(prod_p0), str(prod_c1)], check=True
            )
            subprocess.run(
                ["git", "-C", str(broker_wt), "commit", "-qm", "seal fresh c1 test packet"],
                check=True,
            )
            fresh = subprocess.run(
                [
                    str(broker_wt / "tools/ops/dispatch.sh"),
                    "sandbox-ct-cf-test-c1",
                    str(prod_c1 / "dispatch-manifest.json"),
                    "--broker",
                    str(broker_wt),
                    "--cwd",
                    str(broker_wt),
                    "--state-dir",
                    str(art_root / "attempt-c1/state"),
                    "--dry-run",
                ],
                cwd=broker_wt,
                env=dry_env,
                capture_output=True,
                text=True,
            )
            evidence = ws / "dry-run-evidence"
            evidence.mkdir()
            (evidence / "fresh-c1.stdout.txt").write_text(fresh.stdout)
            (evidence / "fresh-c1.stderr.txt").write_text(fresh.stderr)
            assert fresh.returncode == 0, fresh.stdout + fresh.stderr
            assert "PASS_DRY_RUN" in fresh.stdout
            assert not (art_root / "attempt-c1/output").exists()

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
        # Completion 2 adds stage 3 while retaining admitted [1, 2].
        for old in list(exp_c1.glob("stage-0002*")):
            old.rename(old.with_name(old.name.replace("0002", "0003")))
        header = exp_c1 / "stage-0003.json"
        info = json.loads(header.read_text())
        info["stage"], info["id"] = 3, unit_id(3)
        info["snapshot"]["id"] = unit_id(3)
        header.write_text(json.dumps(info))
        Path(str(header) + ".sha256").write_text(digest(header) + "\n")

        rec_c1 = ws / "receipt-c1.json"
        rec_c1.write_text(
            json.dumps(
                {
                    "job_id": "sandbox-ct-cf-test-c1",
                    "commit": impl_pin,
                    "manifest_sha256": digest(prod_c1 / "dispatch-manifest.json"),
                    "status": "INCOMPLETE",
                    "failure_class": "SUPERVISOR",
                    "error": "whole-session wall cap reached",
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
                "--supervisor-log",
                str(_clean_supervisor_log(ws, "supervisor-c1.log")),
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

        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from research.sandbox_a_resident.test_ct_runner_and_resume import "
                    "_ct3_committed_admission_refusal_probes; "
                    f"_ct3_committed_admission_refusal_probes(Path({str(prod_c2)!r}))"
                ),
            ],
            cwd=broker_wt,
            check=True,
        )

        # Run full tools/ops/dispatch.sh --dry-run on BOTH c1 and c2 at top level
        # (skip only inside nested ct_preflight subprocesses to avoid recursion).
        if os.environ.get("CT_PREFLIGHT_SUBPROCESS") != "1":
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
                # Preserve the wrapper's stderr AND its validation logs, even
                # on success. stdout is never written to ops/logs/<name>.log.
                evidence = ws / "dry-run-evidence"
                evidence.mkdir(exist_ok=True)
                with (evidence / f"{name}.stdout.txt").open("x") as stream:
                    stream.write(res.stdout)
                with (evidence / f"{name}.stderr.txt").open("x") as stream:
                    stream.write(res.stderr)
                with (evidence / f"{name}.command.json").open("x") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "argv": res.args,
                                "cwd": str(broker_wt),
                                "env": {
                                    k: dry_env[k] for k in ("OPS_DIR", "SNAKE_DQN_PYTHON", "PATH")
                                },
                                "returncode": res.returncode,
                            },
                            indent=2,
                        )
                    )
                if name == "sandbox-ct-cf-test-c1":
                    # c1 has already produced the canonical evidence consumed by c2.
                    # Re-dispatch must refuse that used output; c2 remains fresh.
                    assert res.returncode == 1 and "dispatch identity already used" in res.stderr
                    continue
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
        if not retained_ws:
            shutil.rmtree(ws, ignore_errors=True)


def test_qualified_nested_poll_shapes_fault_precedence_and_all_polls_scan(tmp_path: Path) -> None:
    """Re-review Finding 1: inspect qualified nested poll state & supervisor_log_tail."""
    from research.colab_cli import runtime as colab_runtime

    manifest = {"job_id": "sandbox-ct-cf-qual-poll-01", "commit": "d" * 40}
    m_sha = "e" * 64
    rpc_timeout_receipt: dict[str, object] = {
        "job_id": "sandbox-ct-cf-qual-poll-01",
        "commit": "d" * 40,
        "manifest_sha256": m_sha,
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": "whole-session wall cap reached",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": "gen-p1",
    }

    # 1) Astra's exact verbatim counterexample (REVIEW.md §1):
    #    <attempt>/poll-000001.json + <attempt>/exports/ct + SUPERVISOR wall cap w/ terminal=None.
    astra_attempt = tmp_path / "astra_attempt_01"
    astra_export_root = astra_attempt / "exports" / "ct"
    astra_export_root.mkdir(parents=True)
    manifest["output_dir"] = str(astra_attempt)
    (astra_attempt / "poll-000001.json").write_text(
        json.dumps(
            {
                "state": {
                    "status": "failed",
                    "reason": "runtime_error",
                    "error": "FloatingPointError('non-finite tensor')",
                },
                "terminal": None,
                "exports": [],
                "job_log_tail": "",
                "supervisor_log_tail": (
                    "Traceback (most recent call last):\nFloatingPointError: non-finite tensor"
                ),
            }
        )
    )
    with pytest.raises(CTResumeRefused, match="floatingpointerror|non-finite"):
        verify_mechanical_stop(rpc_timeout_receipt, manifest, m_sha, export_root=astra_export_root)

    def _produce_real_qualified_poll(
        case_name: str,
        *,
        state: dict[str, object],
        job_log: str = "",
        supervisor_log: str = "",
        terminal: dict[str, object] | None = None,
        compact: bool = True,
    ) -> dict[str, object]:
        vm_root = tmp_path / f"vm_{case_name}"
        vm_root.mkdir(parents=True, exist_ok=True)
        gen = str(state.get("generation", "gen-p1"))
        full_state = {"job_id": "sandbox-ct-cf-qual-poll-01", "generation": gen, **state}
        (vm_root / "state.json").write_text(json.dumps(full_state))
        (vm_root / "lease.json").write_text(json.dumps({"generation": gen, "renewed_at": 1.0}))
        if job_log:
            (vm_root / "job.log").write_text(job_log)
        if supervisor_log:
            (vm_root / "supervisor.log").write_text(supervisor_log)
        if terminal is not None:
            (vm_root / "terminal.json").write_text(json.dumps(terminal))
        return colab_runtime.read_state(vm_root, gen, renew=False, compact=compact)

    def _attempt_with_polls(name: str, polls: list[dict[str, object]]) -> Path:
        att = tmp_path / name
        manifest["output_dir"] = str(att)
        exp = att / "exports" / "ct"
        exp.mkdir(parents=True)
        for idx, p_obj in enumerate(polls, start=1):
            (att / f"poll-{idx:06d}.json").write_text(json.dumps(p_obj))
        return exp

    # 2) Nested state runtime fault alone (empty supervisor_log_tail & job_log_tail, no terminal).
    poll_state_only = _produce_real_qualified_poll(
        "state_fault_only",
        state={
            "status": "failed",
            "reason": "runtime_error",
            "error": "FloatingPointError('non-finite tensor')",
        },
    )
    exp_state_only = _attempt_with_polls("att_state_only", [poll_state_only])
    with pytest.raises(CTResumeRefused, match="floatingpointerror|runtime_error"):
        verify_mechanical_stop(rpc_timeout_receipt, manifest, m_sha, export_root=exp_state_only)

    # 3) supervisor_log_tail fault alone while state.status == "running" (traceback & runtime.py).
    for idx_sup, sup_err in enumerate(
        (
            "Traceback (most recent call last):\nFloatingPointError: non-finite tensor\n",
            "colab-cli runtime: runtime supervisor failure: OSError('No space left on device')\n",
        )
    ):
        poll_sup_only = _produce_real_qualified_poll(
            f"sup_only_{idx_sup}",
            state={"status": "running", "pid": 4444},
            supervisor_log=sup_err,
        )
        exp_sup_only = _attempt_with_polls(f"att_sup_only_{idx_sup}", [poll_sup_only])
        with pytest.raises(CTResumeRefused):
            verify_mechanical_stop(rpc_timeout_receipt, manifest, m_sha, export_root=exp_sup_only)

    # 4) Child crash in nested state (status="failed", reason=None, returncode=1) w/o terminal.
    poll_child_crash = _produce_real_qualified_poll(
        "child_crash",
        state={"status": "failed", "reason": None, "returncode": 1},
    )
    exp_child_crash = _attempt_with_polls("att_child_crash", [poll_child_crash])
    with pytest.raises(CTResumeRefused, match="non-mechanical child failure"):
        verify_mechanical_stop(rpc_timeout_receipt, manifest, m_sha, export_root=exp_child_crash)

    # 5) Completed, post-terminal, watchdog-124, or invalid nested state w/o terminal/final exports.
    bad_states: list[tuple[str, dict[str, object], bool]] = [
        ("succeeded", {"status": "succeeded", "returncode": 0, "reason": None}, True),
        ("completed", {"status": "completed", "returncode": None, "reason": None}, True),
        (
            "post_terminal",
            {
                "status": "lease_expired_after_terminal",
                "returncode": 0,
                "reason": "lease_expired",
            },
            True,
        ),
        ("invalid", {"status": "invalid", "returncode": None, "reason": None}, True),
        ("watchdog_124", {"status": "failed", "returncode": 124, "reason": "timeout"}, True),
        (
            "unassign_err",
            {"status": "failed", "reason": "timeout", "unassign_error": "URLError"},
            True,
        ),
        (
            "state_exports_summary",
            {
                "status": "running",
                "exports": [{"path": "ct/summary.json", "size": 512, "sha256": "a" * 64}],
            },
            False,
        ),
        (
            "state_exports_watchdog",
            {
                "status": "running",
                "exports": [
                    {
                        "path": "ct/stall-watchdog/faulthandler-stacks.log",
                        "size": 128,
                        "sha256": "b" * 64,
                    }
                ],
            },
            False,
        ),
    ]
    for tag, st_dict, is_compact in bad_states:
        poll_bad = _produce_real_qualified_poll(f"bad_{tag}", state=st_dict, compact=is_compact)
        exp_bad = _attempt_with_polls(f"att_bad_{tag}", [poll_bad])
        with pytest.raises(CTResumeRefused):
            verify_mechanical_stop(rpc_timeout_receipt, manifest, m_sha, export_root=exp_bad)

    # 6) ALL retained polls scanned: fault in poll-000001.json is NOT masked by clean poll-000002!
    clean_running_poll = _produce_real_qualified_poll(
        "clean_running",
        state={
            "status": "running",
            "pid": 7777,
            "exports": [{"path": "ct/stage-0001.pt", "size": 4096, "sha256": "c" * 64}],
        },
        job_log=(
            'CT_PROGRESS {"phase":"scoring_chunk_0_selection","slabs_completed":1,'
            '"logical_branch_steps":42,"executed_sim_steps":42,"elapsed_sec": 60.0, '
            '"latest_durable_stage": 1, '
            '"schema": "t34-cf-progress/v1", "stage": "scoring-progress", "study": "t34-cf-1028"}\n'
        ),
        compact=False,
    )
    exp_multi_fault = _attempt_with_polls(
        "att_multi_poll_fault_first", [poll_state_only, clean_running_poll]
    )
    with pytest.raises(CTResumeRefused, match="floatingpointerror|runtime_error"):
        verify_mechanical_stop(rpc_timeout_receipt, manifest, m_sha, export_root=exp_multi_fault)

    # 7) Ordinary verified timeout/lease/credential-expiring stops in qualified poll shape pass!
    empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    poll_starting = _produce_real_qualified_poll("ok_start", state={"status": "starting"})
    for mech_reason in ("timeout", "lease_expired", "release_credential_expiring"):
        mech_exports = [
            {"path": "ct/stage-0001.pt", "size": 4096, "sha256": "c" * 64},
            {
                "path": "ct/stall-watchdog/faulthandler-stacks.log",
                "size": 0,
                "sha256": empty_sha,
            },
        ]
        mech_term = {
            "export_scanner": {"name": "runtime._exports", "inventory": mech_exports},
            "canary_observations": [],
            "last_traceback_line": (
                "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/out/ct/stall-watchdog"
            ),
            "job_id": "sandbox-ct-cf-qual-poll-01",
            "generation": "gen-p1",
            "status": "failed",
            "returncode": 138,
            "reason": mech_reason,
            "exports": mech_exports,
        }
        poll_mech_end = _produce_real_qualified_poll(
            f"ok_end_{mech_reason}",
            state={
                "status": "failed",
                "returncode": 138,
                "reason": mech_reason,
                "exports": mech_exports,
            },
            job_log=(
                'CT_PROGRESS {"phase":"scoring_chunk_0_scoring","slabs_completed":2,'
                '"logical_branch_steps":84,"executed_sim_steps":84,"elapsed_sec": 120.0, '
                '"latest_durable_stage": 1, '
                '"schema": "t34-cf-progress/v1", "stage": "scoring-progress", '
                '"study": "t34-cf-1028"}\n'
                "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/out/ct/stall-watchdog\n"
            ),
            terminal=mech_term,
            compact=True,
        )
        exp_ok = _attempt_with_polls(
            f"att_ok_{mech_reason}", [poll_starting, clean_running_poll, poll_mech_end]
        )
        verify_mechanical_stop(rpc_timeout_receipt, manifest, m_sha, export_root=exp_ok)
        verify_mechanical_stop(
            {
                **rpc_timeout_receipt,
                "failure_class": "CHILD_NONZERO_EXIT",
                "terminal": mech_term,
            },
            manifest,
            m_sha,
            export_root=exp_ok,
        )


def test_successor_packet_refuses_omitted_or_conflicting_prior_coverage_and_hidden_provenance_fault(
    tmp_path: Path,
) -> None:
    """Re-review Finding 2 regression: successor packets must preserve & bind admitted coverage."""
    import copy

    from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger

    canonical_ckpt_dir = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
    if (canonical_ckpt_dir / "b8_s3_t7_20261016.pth").is_file():
        ckpts = (
            canonical_ckpt_dir / "b8_s3_t7_20261016.pth",
            canonical_ckpt_dir / "frp3_m3_s12_u60000_20261005.pth",
            canonical_ckpt_dir / "champion_a5_freespace_20260621.pth",
        )
        use_registry_validation = True
    else:
        ckpts = _save_tiny_checkpoints(tmp_path)
        use_registry_validation = False

    ctrl = CTControl(
        source_commit="a" * 40, worlds=6, collection_ticks=12, scoring_chunks=4, tiny=True
    )
    ctrl_dict = ctrl.to_dict()
    alloc = allocation(ctrl_dict)

    a0_out = tmp_path / "attempt_00_out"
    packet_00 = tmp_path / "packet-00"
    _write_tiny_initial_packet(packet_00, ctrl, ckpts, a0_out)
    validate_uploaded_ledger(
        packet_00,
        json.loads((packet_00 / "dispatch-manifest.json").read_text()),
        ctrl_dict,
        require_committed=False,
    )

    dir_s1 = a0_out / "exports_s1"
    dir_s2 = a0_out / "exports_s2"
    dir_s2_conflict = a0_out / "exports_s2_conflict"
    dir_s3 = tmp_path / "attempt_c1_out/exports_s3"
    publish_stage_snapshot(
        dir_s1,
        1,
        lambda dst: torch.save({"stage": 1, "v": torch.tensor([11])}, dst),
        metadata={
            "allocation": alloc,
            "source_commit": ctrl.source_commit,
            "support_status": "PASS",
        },
    )
    publish_stage_snapshot(
        dir_s2,
        2,
        lambda dst: torch.save({"stage": 2, "v": torch.tensor([22])}, dst),
        metadata={"allocation": alloc, "source_commit": ctrl.source_commit},
    )
    publish_stage_snapshot(
        dir_s2_conflict,
        2,
        lambda dst: torch.save({"stage": 2, "v": torch.tensor([999])}, dst),
        metadata={"allocation": alloc, "source_commit": ctrl.source_commit},
    )
    publish_stage_snapshot(
        dir_s3,
        3,
        lambda dst: torch.save({"stage": 3, "v": torch.tensor([33])}, dst),
        metadata={"allocation": alloc, "source_commit": ctrl.source_commit},
    )

    (
        Path(json.loads((packet_00 / "dispatch-manifest.json").read_text())["output_dir"])
        / "exports/ct"
    ).mkdir(parents=True)

    rec_00 = {
        "job_id": "tiny-ct-00",
        "commit": ctrl.source_commit,
        "manifest_sha256": digest(packet_00 / "dispatch-manifest.json"),
        "status": "INCOMPLETE",
        "failure_class": "CHILD_NONZERO_EXIT",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": "g0",
        "terminal": {
            "job_id": "tiny-ct-00",
            "generation": "g0",
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        },
    }
    c1_out = tmp_path / "attempt_c1_out"
    packet_c1 = tmp_path / "packet-c1"
    c1_res = write_continuation_packet(
        packet_c1,
        prior_packet_dir=packet_00,
        terminal_receipt=rec_00,
        export_root=[dir_s1, dir_s2],
        completion_index=1,
        job_id="tiny-ct-c1",
        output_dir=c1_out,
        use_compact_xz_for_stages=(1,),
        auto_encode_compact_xz=True,
        validate_registry_inputs=use_registry_validation,
        supervisor_log_path=_clean_supervisor_log(tmp_path),
    )
    assert c1_res["verified_stages"] == [1, 2]
    c1_manifest = json.loads((packet_c1 / "dispatch-manifest.json").read_text())
    c1_sha = digest(packet_c1 / "dispatch-manifest.json")
    validate_uploaded_ledger(
        packet_c1, copy.deepcopy(c1_manifest), ctrl_dict, require_committed=False
    )

    (c1_out / "exports/ct").mkdir(parents=True)

    rec_c1 = {
        "job_id": "tiny-ct-c1",
        "commit": ctrl.source_commit,
        "manifest_sha256": c1_sha,
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": "whole-session wall cap reached",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": "g1",
    }

    # 1) Astra's exact omission counterexample (REVIEW.md §2):
    #    Prior c1 has verified_stages=[1, 2]; supplying ONLY dir_s1 for completion 2 MUST be refused
    #    by both completion_packet() and write_continuation_packet() (including legacy c1 shape)!
    with pytest.raises(CTResumeRefused, match=r"omits previously verified stage\(s\) \[2\]"):
        completion_packet(
            c1_manifest,
            rec_c1,
            dir_s1,
            ctrl_dict,
            completion_index=2,
            job_id="tiny-ct-c2-omit-direct",
            output_dir=tmp_path / "c2_omit_direct_out",
            original_manifest_sha256=c1_sha,
        )

    minimal_c1_manifest = {
        **copy.deepcopy(c1_manifest),
        "completion_attempt": {"index": 1, "verified_stages": [1, 2]},
    }
    minimal_c1_sha = hashlib.sha256(
        json.dumps(minimal_c1_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with pytest.raises(CTResumeRefused, match="unbound retained evidence scan root"):
        completion_packet(
            minimal_c1_manifest,
            {**rec_c1, "manifest_sha256": minimal_c1_sha},
            dir_s1,
            ctrl_dict,
            completion_index=2,
            job_id="tiny-ct-c2-omit-legacy",
            output_dir=tmp_path / "c2_omit_legacy_out",
            original_manifest_sha256=minimal_c1_sha,
        )

    packet_c2_omit = tmp_path / "packet-c2-omit"
    with pytest.raises(CTResumeRefused, match=r"omits previously verified stage\(s\) \[2\]"):
        write_continuation_packet(
            packet_c2_omit,
            prior_packet_dir=packet_c1,
            terminal_receipt=rec_c1,
            export_root=dir_s1,
            completion_index=2,
            job_id="tiny-ct-c2-omit",
            output_dir=tmp_path / "c2_omit_out",
            use_compact_xz_for_stages=(1,),
            auto_encode_compact_xz=True,
            validate_registry_inputs=use_registry_validation,
            supervisor_log_path=_clean_supervisor_log(tmp_path),
        )
    assert not packet_c2_omit.exists()

    # 2) Conflict / recomputation counterexample:
    #    Supplying [dir_s1, dir_s2_conflict] (altered/recomputed stage-0002) MUST be refused!
    with pytest.raises(
        CTResumeRefused, match="stage-0002 conflicts with previously verified coverage"
    ):
        completion_packet(
            c1_manifest,
            rec_c1,
            [dir_s1, dir_s2_conflict],
            ctrl_dict,
            completion_index=2,
            job_id="tiny-ct-c2-conflict-direct",
            output_dir=tmp_path / "c2_conflict_direct_out",
            original_manifest_sha256=c1_sha,
        )

    packet_c2_conflict = tmp_path / "packet-c2-conflict"
    with pytest.raises(
        CTResumeRefused, match="stage-0002 conflicts with previously verified coverage"
    ):
        write_continuation_packet(
            packet_c2_conflict,
            prior_packet_dir=packet_c1,
            terminal_receipt=rec_c1,
            export_root=[dir_s1, dir_s2_conflict],
            completion_index=2,
            job_id="tiny-ct-c2-conflict",
            output_dir=tmp_path / "c2_conflict_out",
            use_compact_xz_for_stages=(1,),
            auto_encode_compact_xz=True,
            validate_registry_inputs=use_registry_validation,
            supervisor_log_path=_clean_supervisor_log(tmp_path),
        )
    assert not packet_c2_conflict.exists()

    # 3) Valid completion 2 preserving [1, 2] and adding [3] passes both builder and preflight!
    packet_c2_ok = tmp_path / "packet-c2-ok"
    c2_res = write_continuation_packet(
        packet_c2_ok,
        prior_packet_dir=packet_c1,
        terminal_receipt=rec_c1,
        export_root=[dir_s1, dir_s2, dir_s3],
        completion_index=2,
        job_id="tiny-ct-c2-ok",
        output_dir=tmp_path / "c2_ok_out",
        use_compact_xz_for_stages=(1,),
        auto_encode_compact_xz=True,
        validate_registry_inputs=use_registry_validation,
        supervisor_log_path=_clean_supervisor_log(tmp_path),
    )
    assert c2_res["verified_stages"] == [1, 2, 3]
    c2_manifest = json.loads((packet_c2_ok / "dispatch-manifest.json").read_text())
    assert c2_manifest["completion_attempt"]["prior_verified_stages"] == [1, 2]
    validate_uploaded_ledger(
        packet_c2_ok, copy.deepcopy(c2_manifest), ctrl_dict, require_committed=False
    )

    # 4) Forged/tampered packet_c2 dropping or mutating stage 2 MUST be refused by ct_preflight!
    forged_drop_dir = tmp_path / "packet-c2-forged-drop"
    shutil.copytree(packet_c2_ok, forged_drop_dir)
    forged_ledger = json.loads((forged_drop_dir / "verified-ledger.json").read_text())
    forged_ledger["units"] = [u for u in forged_ledger["units"] if u["stage"] != 2]
    (forged_drop_dir / "verified-ledger.json").write_text(
        json.dumps(forged_ledger, indent=2, sort_keys=True) + "\n"
    )
    forged_lsha = digest(forged_drop_dir / "verified-ledger.json")
    forged_broker = copy.deepcopy(c2_manifest)
    forged_broker["inputs"] = [
        (
            {
                "path": str((forged_drop_dir / "verified-ledger.json").resolve()),
                "dest": "verified-ledger.json",
                "sha256": forged_lsha,
            }
            if r["dest"] == "verified-ledger.json"
            else r
        )
        for r in forged_broker["inputs"]
        if not str(r["dest"]).startswith("resume/stage-0002")
    ]
    forged_broker["completion_attempt"]["verified_stages"] = [1, 3]
    forged_broker["completion_attempt"]["ledger_sha256"] = forged_lsha
    _rebind_forged_packet(forged_drop_dir, forged_broker)
    with pytest.raises(ValueError, match="^continuation ledger omits previously verified stage 2$"):
        validate_uploaded_ledger(forged_drop_dir, forged_broker, ctrl_dict, require_committed=False)

    # Even if an attacker ALSO edits completion_attempt.prior_verified_units/stages, on-disk
    # prior_manifest_path -> prior verified-ledger.json cross-check in ct_preflight catches it!
    forged_broker_2 = copy.deepcopy(forged_broker)
    forged_broker_2["completion_attempt"]["prior_verified_stages"] = [1]
    forged_broker_2["completion_attempt"]["prior_verified_units"] = [
        u for u in forged_broker_2["completion_attempt"]["prior_verified_units"] if u["stage"] != 2
    ]
    _rebind_forged_packet(forged_drop_dir, forged_broker_2)
    with pytest.raises(
        ValueError,
        match="^prior_verified_units differs from bound prior verified-ledger.json$",
    ):
        validate_uploaded_ledger(
            forged_drop_dir, forged_broker_2, ctrl_dict, require_committed=False
        )

    # Forged conflicting stage 2 in packet_c2 -> refused by ct_preflight!
    forged_conf_dir = tmp_path / "packet-c2-forged-conflict"
    shutil.copytree(packet_c2_ok, forged_conf_dir)
    conf_ledger, conf_uploads, _ = build_stage_ledger([dir_s1, dir_s2_conflict, dir_s3], ctrl_dict)
    (forged_conf_dir / "verified-ledger.json").write_text(
        json.dumps(conf_ledger, indent=2, sort_keys=True) + "\n"
    )
    conf_lsha = digest(forged_conf_dir / "verified-ledger.json")
    conf_broker = copy.deepcopy(c2_manifest)
    conf_broker["inputs"] = [
        r
        for r in conf_broker["inputs"]
        if r["dest"] != "verified-ledger.json" and not str(r["dest"]).startswith("resume/")
    ] + [
        {
            "path": str((forged_conf_dir / "verified-ledger.json").resolve()),
            "dest": "verified-ledger.json",
            "sha256": conf_lsha,
        },
        *conf_uploads,
    ]
    conf_broker["completion_attempt"]["ledger_sha256"] = conf_lsha
    _rebind_forged_packet(forged_conf_dir, conf_broker)
    with pytest.raises(
        ValueError,
        match="^continuation ledger conflicts with previously verified stage-0002$",
    ):
        validate_uploaded_ledger(forged_conf_dir, conf_broker, ctrl_dict, require_committed=False)

    # 5) Bound prior provenance fault inspection: if an earlier attempt's output_dir (a0_out) holds
    #    retained qualified poll fault evidence, BOTH write_continuation_packet(c2) and ct_preflight
    #    refuse continuation even though dir_s1/dir_s2 live outside a0_out!
    a0_out.mkdir(parents=True, exist_ok=True)
    (a0_out / "poll-000001.json").write_text(
        json.dumps(
            {
                "state": {
                    "status": "failed",
                    "reason": "runtime_error",
                    "error": "FloatingPointError('non-finite tensor')",
                },
                "terminal": None,
                "exports": [],
                "job_log_tail": "",
                "supervisor_log_tail": "FloatingPointError: non-finite tensor",
            }
        )
    )
    with pytest.raises(CTResumeRefused, match="floatingpointerror|non-finite"):
        write_continuation_packet(
            tmp_path / "packet-c2-hidden-a0-fault",
            prior_packet_dir=packet_c1,
            terminal_receipt=rec_c1,
            export_root=[dir_s1, dir_s2, dir_s3],
            completion_index=2,
            job_id="tiny-ct-c2-hidden-fault",
            output_dir=tmp_path / "c2_hidden_fault_out",
            use_compact_xz_for_stages=(1,),
            auto_encode_compact_xz=True,
            validate_registry_inputs=use_registry_validation,
            supervisor_log_path=_clean_supervisor_log(tmp_path),
        )
    with pytest.raises(ValueError, match="floatingpointerror|non-finite"):
        validate_uploaded_ledger(
            packet_c2_ok, copy.deepcopy(c2_manifest), ctrl_dict, require_committed=False
        )


@pytest.mark.parametrize(
    "relative,content",
    [
        (
            "exports/ct/progress.json.partial",
            '{"stage":"scoring-failed","error":"FloatingPointError: non-finite tensor tripwire"}',
        ),
        (
            "exports/ct/progress.json.partial",
            '{"stage":"scoring-failed","error":"FloatingPointError: non-finite',
        ),
        ("exports/ct/progress.json.partial", '{"stage":"scoring-progress"'),
        (
            "exports/ct/job.log",
            "Traceback (most recent call last):\nFloatingPointError: non-finite",
        ),
        ("supervisor.log", "FloatingPointError: non-finite tensor tripwire"),
        ("exports/ct/calibrated-boundary.json", '{"error":"FloatingPointError: non-finite"}'),
        ("exports/ct/stage-0002.json", '{"error":"FloatingPointError: non-finite"}'),
        ("exports/ct/progress.json", '{"metadata":{"error":"FloatingPointError: non-finite"}}'),
        ("exports/ct/unknown.json", '{"metadata":{"error":"opaque failure"}}'),
        ("exports/ct/unknown.json", '{"metadata":{"finite":false}}'),
        ("poll-000001.json", '{"state":{"job_id":"foreign","generation":"g0","status":"running"}}'),
        (
            "terminal.json",
            '{"job_id":"tiny-ct-00","generation":"foreign","status":"failed",'
            '"reason":"timeout","returncode":138}',
        ),
        (
            "state.json",
            '{"job_id":"tiny-ct-00","generation":"g0","status":"failed","returncode":7}',
        ),
    ],
)
def test_ct3_retained_fault_probes_refuse_writer_and_admission(
    tmp_path: Path,
    relative: str,
    content: str,
) -> None:
    """CT3 MAJOR-1: retained evidence is mandatory through both production boundaries."""
    from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger

    ctrl = CTControl(source_commit="a" * 40, worlds=6, collection_ticks=12, tiny=True)
    initial, successor = tmp_path / "initial", tmp_path / "successor"
    prior_output = tmp_path / "prior-output"
    exports = prior_output / "exports/ct"
    _write_tiny_initial_packet(initial, ctrl, _save_tiny_checkpoints(tmp_path), prior_output)
    publish_stage_snapshot(
        exports,
        1,
        lambda p: p.write_bytes(b"hash-only numerical payload"),
        metadata={"allocation": allocation(ctrl.to_dict()), "source_commit": ctrl.source_commit},
    )
    receipt = {
        "job_id": "tiny-ct-00",
        "commit": ctrl.source_commit,
        "manifest_sha256": digest(initial / "dispatch-manifest.json"),
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": "whole-session wall cap reached",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": "g0",
    }
    kwargs = dict(
        prior_packet_dir=initial,
        terminal_receipt=receipt,
        export_root=exports,
        completion_index=1,
        job_id="ct-c1",
        output_dir=tmp_path / "next-output",
        use_compact_xz_for_stages=None,
        validate_registry_inputs=False,
        supervisor_log_path=_clean_supervisor_log(tmp_path),
    )
    write_continuation_packet(successor, **kwargs)
    manifest = json.loads((successor / "dispatch-manifest.json").read_text())
    validate_uploaded_ledger(successor, manifest, ctrl.to_dict(), require_committed=False)
    fault = prior_output / relative
    fault.parent.mkdir(parents=True, exist_ok=True)
    fault.write_text(content)
    with pytest.raises(ValueError):
        validate_uploaded_ledger(successor, manifest, ctrl.to_dict(), require_committed=False)
    with pytest.raises(ValueError):
        write_continuation_packet(tmp_path / "refused", **kwargs)
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("cap", [5 * 1024 * 1024, 1024])
def test_ct3_oversized_exception_export_obeys_recursive_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cap: int,
) -> None:
    """CT3 MINOR: the actual startup exception handler cannot bypass the hard total."""
    from research.sandbox_a_resident import ct_run
    from research.sandbox_a_resident.ct_units import recursive_export_bytes

    def explode(_device: object) -> None:
        raise ValueError("X" * (3 * 1024 * 1024))

    monkeypatch.setattr(ct_run, "forced_state_determinism_suite", explode)
    output = tmp_path / "fault-export"
    ctrl = CTControl(
        source_commit="a" * 40, tiny=True, max_stage_export_bytes=cap, max_total_export_bytes=cap
    )
    with pytest.raises(ValueError):
        run_ct_experiment(
            ctrl,
            b8_checkpoint=tmp_path / "unused",
            frp3_checkpoint=tmp_path / "unused",
            champion_checkpoint=tmp_path / "unused",
            output=output,
            device="cpu",
        )
    assert recursive_export_bytes(output) <= cap
    # No fault bundle fits the small cap, so publication must refuse atomically.
    if cap == 1024:
        assert not (output / "fault.json").exists()
        assert sorted(p.name for p in output.iterdir()) == [
            "fixture-runtime.json",
            "progress-history.log",
            "progress.json",
        ]
    else:
        assert (output / "fault.json.complete").is_file()
        record = json.loads((output / "fault.json").read_text())
        assert len(record["error_message"]) <= 8192
        assert len(record["traceback"]) <= 16384
        assert recursive_export_bytes(output) <= cap


def _ct3_committed_admission_refusal_probes(packet: Path) -> None:
    """Exercise CT3 forgeries as committed packets in the disposable production clone."""
    import copy

    from research.sandbox_a_resident.ct_preflight import validate_packet
    from research.sandbox_a_resident.units import digest
    from tools.sandbox.preflight import PreflightError

    files = {p: p.read_bytes() for p in packet.iterdir() if p.is_file()}
    original = json.loads((packet / "dispatch-manifest.json").read_text())
    ancestor = Path(original["completion_attempt"]["prior_manifest_path"]).parent
    root_packet = Path(
        json.loads((ancestor / "dispatch-manifest.json").read_text())["completion_attempt"][
            "prior_manifest_path"
        ]
    ).parent
    ancestor_files = {
        p: p.read_bytes()
        for directory in (ancestor, root_packet)
        for p in directory.iterdir()
        if p.is_file()
    }
    original_output = Path(
        json.loads(Path(original["completion_attempt"]["prior_manifest_path"]).read_text())[
            "completion_attempt"
        ]["prior_output_dirs"][0]
    )
    original_output_existed = original_output.exists()
    for attack in (
        "missing_reference_and_stage",
        "omitted_output_dirs",
        "omitted_export_roots",
        "missing_proof",
        "missing_seal",
        "unbound_proof",
        "inventory_deletion",
        "nested_fault_rebound_inventory",
        "shadow_prior",
        "missing_root_seal",
        "uncommitted_root_seal",
        "uncommitted_prior_manifest",
        "omitted_supervisor_log",
        "relative_supervisor_log",
        "rebound_log_fault",
    ):
        for path, data in {**files, **ancestor_files}.items():
            path.write_bytes(data)
        broker = copy.deepcopy(original)
        comp = broker["completion_attempt"]
        if attack == "missing_reference_and_stage":
            comp.pop("prior_manifest_path")
            ledger_path = packet / "verified-ledger.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["units"] = [u for u in ledger["units"] if u["stage"] != 2]
            ledger_path.write_text(json.dumps(ledger))
            comp["verified_stages"] = [1, 3]
            comp["prior_verified_stages"] = [1]
            comp["prior_verified_units"] = [
                u for u in comp["prior_verified_units"] if u["stage"] != 2
            ]
            comp["ledger_sha256"] = digest(ledger_path)
            broker["inputs"] = [
                u for u in broker["inputs"] if not u["dest"].startswith("resume/stage-0002")
            ]
            next(u for u in broker["inputs"] if u["dest"] == "verified-ledger.json")["sha256"] = (
                digest(ledger_path)
            )
        elif attack == "omitted_output_dirs":
            comp["prior_output_dirs"] = []
        elif attack == "omitted_export_roots":
            comp["prior_export_roots"] = comp["prior_export_roots"][-1:]
        elif attack in {"missing_proof", "missing_seal"}:
            (packet / ("provenance.json" if attack == "missing_proof" else "SEAL.json")).unlink()
        elif attack == "unbound_proof":
            comp.pop("provenance_sha256")
        elif attack == "inventory_deletion":
            proof = json.loads((packet / "provenance.json").read_text())
            proof["retained"] = {}
            (packet / "provenance.json").write_text(json.dumps(proof))
            comp["provenance_sha256"] = digest(packet / "provenance.json")
        elif attack == "shadow_prior":
            shadow = ancestor / "shadow.json"
            shadow.write_bytes((ancestor / "dispatch-manifest.json").read_bytes())
            comp["prior_manifest_path"] = str(shadow)
            proof = json.loads((packet / "provenance.json").read_text())
            proof["prior_manifest_path"] = str(shadow)
            (packet / "provenance.json").write_text(json.dumps(proof))
            comp["provenance_sha256"] = digest(packet / "provenance.json")
        elif attack == "missing_root_seal":
            (root_packet / "SEAL.json").unlink()
        elif attack in {"uncommitted_root_seal", "uncommitted_prior_manifest"}:
            pass  # Mutated after the packet commit, below.
        elif attack in {"omitted_supervisor_log", "relative_supervisor_log", "rebound_log_fault"}:
            proof = json.loads((packet / "provenance.json").read_text())
            if attack == "omitted_supervisor_log":
                proof["supervisor_log"] = None
            elif attack == "relative_supervisor_log":
                proof["supervisor_log"]["path"] = "supervisor.log"
            else:
                log = Path(proof["supervisor_log"]["path"])
                original_log = log.read_bytes()
                log.write_text('{"status":"failed","returncode":7}\n')
                proof["supervisor_log"].update(size=log.stat().st_size, sha256=digest(log))
            (packet / "provenance.json").write_text(json.dumps(proof))
            comp["provenance_sha256"] = digest(packet / "provenance.json")
        else:
            # Rebind *all* inventories after injecting a fault: semantic scan must still refuse.
            from research.sandbox_a_resident.ct_resume import retained_inventory

            original_output.mkdir(parents=True, exist_ok=True)
            fault = original_output / "unselected.json.partial"
            fault.write_text('{"metadata":{"error":"FloatingPointError: non-finite"}}')
            for proof_path in (
                packet / "provenance.json",
                Path(comp["prior_manifest_path"]).parent / "provenance.json",
            ):
                # Parent remains immutable; c2 admission must recursively re-open its evidence.
                if proof_path.parent != packet:
                    continue
                proof = json.loads(proof_path.read_text())
                proof["retained"] = {r: retained_inventory(Path(r)) for r in proof["retained"]}
                proof_path.write_text(json.dumps(proof))
            comp["provenance_sha256"] = digest(packet / "provenance.json")
        (packet / "dispatch-manifest.json").write_text(json.dumps(broker))
        sandbox = json.loads((packet / "sandbox.json").read_text())
        sandbox["outputs"]["broker_manifest"]["sha256"] = digest(packet / "dispatch-manifest.json")
        if attack == "missing_reference_and_stage":
            sandbox["inputs"] = [
                u for u in sandbox["inputs"] if "stage-0002" not in u.get("path", "")
            ]
            next(u for u in sandbox["inputs"] if u.get("kind") == "verified-ledger")["sha256"] = (
                digest(packet / "verified-ledger.json")
            )
        (packet / "sandbox.json").write_text(
            "\n".join(
                ("{" if i == 0 else " ")
                + json.dumps(k)
                + ": "
                + json.dumps(v)
                + ("," if i < 9 else "}")
                for i, (k, v) in enumerate(sandbox.items())
            )
            + "\n"
        )
        if (packet / "SEAL.json").exists():
            seal = json.loads((packet / "SEAL.json").read_text())
            seal.update(
                manifest_sha256=digest(packet / "dispatch-manifest.json"),
                ledger_sha256=digest(packet / "verified-ledger.json"),
                provenance_sha256=comp.get("provenance_sha256"),
            )
            (packet / "SEAL.json").write_text(json.dumps(seal))
        # Commit only the known components; shadow.json deliberately stays untracked.
        subprocess.run(["git", "add", "-A", str(packet), *map(str, ancestor_files)], check=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-qm", "committed refusal probe " + attack],
            check=True,
        )
        uncommitted = (
            root_packet / "SEAL.json"
            if attack == "uncommitted_root_seal"
            else ancestor / "dispatch-manifest.json"
        )
        if attack in {"uncommitted_root_seal", "uncommitted_prior_manifest"}:
            uncommitted.write_text(uncommitted.read_text() + " ")
        try:
            validate_packet(packet, require_committed=True)
        except (ValueError, FileNotFoundError, PreflightError) as error:
            print("REFUSED_COMMITTED", attack, type(error).__name__, str(error), flush=True)
        else:
            raise AssertionError("ADMITTED " + attack)
        if attack == "shadow_prior":
            shadow.unlink()
        if attack == "rebound_log_fault":
            log.write_bytes(original_log)
        if attack == "nested_fault_rebound_inventory":
            fault.unlink()
            if not original_output_existed:
                original_output.rmdir()
    for path, data in {**files, **ancestor_files}.items():
        path.write_bytes(data)
    subprocess.run(["git", "add", "-A", str(packet), *map(str, ancestor_files)], check=True)
    subprocess.run(["git", "commit", "-qm", "restore accepted fixture"], check=True)
    validate_packet(packet, require_committed=True)
