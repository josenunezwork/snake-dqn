"""Tests for `research.arena_ppo.ruler`, `run_canary_suite`, and 3-mode `prepare.py`."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from research.arena_ppo import ruler as ruler_mod
from research.arena_ppo import runtime as runtime_mod
from research.arena_ppo.ppo import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    StopRequestedError,
    run_training_job,
)
from research.arena_ppo.prepare import (
    CANARY_REQUIRED_EXPORTS,
    ROOT,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
    prepare_arena_ppo_packet,
)
from research.arena_ppo.ruler import _WorldMaturityTracker, run_canary_suite, run_ruler
from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.arena_ppo.sim import StepInfo, TrainSim
from research.colab_cli import canary as colab_canary
from research.colab_cli import schema as colab_schema
from research.colab_cli import supervisor as colab_supervisor
from research.colab_cli.runtime import _exports as supervisor_scan_exports
from tools.ops import dispatch as ops_dispatch


def test_ruler_mps_mature_burnin_stage_split_dual_mb_and_compile_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify M3 burn-in/maturity/stages/untracked/dual-mb + cheapest-first + _compile grid."""
    compile_calls: list[str] = []

    def _fast_compile(fn: Any, **kwargs: Any) -> Any:
        compile_calls.append(str(kwargs.get("mode", "default")))
        return fn

    monkeypatch.setattr(torch, "compile", _fast_compile)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = tmp_path / "ruler_mps_out"
    res = run_ruler(
        output_dir=out_dir,
        device=dev,
        envs_list=[4, 2],  # deliberately unsorted to verify cheapest-first ordering
        net_sizes=["M", "S"],
        burnin_ticks=35,
        warmup_ticks=1,
        collect_ticks=3,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,  # < 4*4*6=96 so larger-minibatch probe runs as well
        actor_chunk_envs=2,
        max_capacity=96,
        include_best_compile=True,
        measure_larger_minibatch=True,
        seed=123,
    )

    assert verify_completed_artifact(out_dir / "ruler_result.json")
    assert res["partial"] is False
    assert res["stopped_reason"] == "completed"
    assert res["batch6_forward"]["S"]["warmup_iters"] == 10
    assert res["batch6_forward"]["S"]["bench_iters"] == 40

    # Cheapest-first: E2_S (2), E4_S (4), E{best}_S_compile, E2_M (6), E4_M (12), E{best}_M_compile
    cell_ids = [c["cell_id"] for c in res["cells"]]
    assert len(cell_ids) == 6
    assert cell_ids[0] == "E2_S"
    assert cell_ids[1] == "E4_S"
    assert cell_ids[2].endswith("_S_compile")
    assert cell_ids[3] == "E2_M"
    assert cell_ids[4] == "E4_M"
    assert cell_ids[5].endswith("_M_compile")
    assert len(compile_calls) >= 2

    expected_stages = (
        "sim_step",
        "rollout_record",
        "obs_build",
        "policy_forward",
        "gae",
        "learner",
    )
    for cell in res["cells"]:
        cid = cell["cell_id"]
        assert verify_completed_artifact(out_dir / "cells" / f"{cid}.json")
        assert cell["oom"] is False
        assert cell["rollout_buffer_bytes"] > 0
        assert cell["train_rollout_len"] == 256
        assert cell["train_rollout_buffer_bytes"] > cell["rollout_buffer_bytes"]
        coll = cell["collection_only"]
        integ = cell["integrated"]
        assert (
            cell["projected_train_peak_vram_bytes"]
            >= integ["peak_vram_bytes"]
            + cell["train_rollout_buffer_bytes"]
            - cell["rollout_buffer_bytes"]
        )
        assert coll["agent_steps_per_sec"] > 0.0
        assert integ["agent_steps_per_sec"] > 0.0
        assert coll["world_maturity"]["mean_length"] > 3.0
        assert integ["world_maturity"]["mean_length"] > 3.0
        assert integ["world_maturity"]["p95_length"] >= integ["world_maturity"]["mean_length"]
        assert "deaths_count" in integ["world_maturity"]
        assert "mean_corpse_count" in integ["world_maturity"]
        assert isinstance(integ["untracked_seconds"], float)
        assert isinstance(integ["epoch0_approx_kl"], float)
        for stg in expected_stages:
            assert stg in integ["stage_seconds"]
            assert stg in integ["stage_peak_vram_bytes"]
        assert integ["learner_default_minibatch"]["minibatch_size"] == 12
        assert integ["learner_larger_minibatch"] is not None
        assert integ["learner_larger_minibatch"]["oom"] is False
        assert integ["learner_larger_minibatch"]["minibatch_size"] > 12


def test_b2_ruler_incremental_cells_sigusr1_partial_export_and_projected_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify B2 per-cell persistence, SIGUSR1 partial ruler_result.json, reuse, and skip."""
    if hasattr(signal, "SIGUSR1"):
        sig_dir = tmp_path / "ruler_sig"
        res_sig = run_ruler(
            output_dir=sig_dir,
            device="cpu",
            envs_list=[2, 4],
            net_sizes=["S"],
            burnin_ticks=5,
            warmup_ticks=1,
            collect_ticks=2,
            rollout_len=3,
            ppo_epochs=1,
            minibatch_size=12,
            max_capacity=64,
            include_best_compile=False,
            measure_larger_minibatch=False,
            step_hook=lambda cid: os.kill(os.getpid(), signal.SIGUSR1) if cid == "E2_S" else None,
        )
        assert res_sig["partial"] is True
        assert res_sig["stopped_reason"] == "signal:SIGUSR1"
        assert [c["cell_id"] for c in res_sig["cells"]] == ["E2_S"]
        assert verify_completed_artifact(sig_dir / "cells" / "E2_S.json")
        assert verify_completed_artifact(sig_dir / "ruler_result.json")

        # Re-run in fresh output dir with input_dir=sig_dir -> reuses E2_S and runs only E4_S
        resume_dir = tmp_path / "ruler_resumed"
        res_resumed = run_ruler(
            output_dir=resume_dir,
            device="cpu",
            input_dir=sig_dir,
            envs_list=[2, 4],
            net_sizes=["S"],
            burnin_ticks=5,
            warmup_ticks=1,
            collect_ticks=2,
            rollout_len=3,
            ppo_epochs=1,
            minibatch_size=12,
            max_capacity=64,
            include_best_compile=False,
            measure_larger_minibatch=False,
        )
        assert res_resumed["partial"] is False
        assert res_resumed["reused_cells"] == ["E2_S"]
        assert [c["cell_id"] for c in res_resumed["cells"]] == ["E2_S", "E4_S"]
        assert verify_completed_artifact(resume_dir / "cells" / "E4_S.json")

    # Projected-time skip: E2_S fits (0s projection on 1st cell), E64_M projects > budget -> skip
    monkeypatch.setattr(
        ruler_mod,
        "_project_cell_seconds",
        lambda **kw: 0.0 if kw["num_envs"] == 2 else 500.0,
    )
    proj_dir = tmp_path / "ruler_proj"
    res_proj = run_ruler(
        output_dir=proj_dir,
        device="cpu",
        envs_list=[2, 64],
        net_sizes=["S"],
        burnin_ticks=4,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=3,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=64,
        include_best_compile=False,
        measure_larger_minibatch=False,
        max_wall_seconds=60.0,
    )
    assert res_proj["partial"] is True
    assert res_proj["stopped_reason"] == "wall_projection_guard"
    assert [c["cell_id"] for c in res_proj["cells"]] == ["E2_S"]
    assert len(res_proj["skipped_cells"]) == 1
    assert res_proj["skipped_cells"][0]["cell_id"] == "E64_S"
    assert res_proj["skipped_cells"][0]["reason"] == "projected_wall_budget_skip"
    assert verify_completed_artifact(proj_dir / "cells" / "E2_S.json")
    assert verify_completed_artifact(proj_dir / "ruler_result.json")


def test_m1_ruler_oom_halving_retries_and_stage_peak_vram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify M1 OOM retry halves actor_chunk_envs & minibatch_size and records stage VRAM."""
    fake_vram = {"bytes": 128_000_000}
    monkeypatch.setattr(ruler_mod, "_peak_vram_bytes", lambda _dev: fake_vram["bytes"])

    orig_integ_once = ruler_mod._measure_integrated_once
    calls: list[tuple[int, int]] = []

    def _flaky_integ_once(**kwargs: Any) -> dict[str, Any]:
        calls.append((int(kwargs["actor_chunk_envs"]), int(kwargs["minibatch_size"])))
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory: simulated initial chunk/mb OOM")
        fake_vram["bytes"] = 96_000_000
        return orig_integ_once(**kwargs)

    monkeypatch.setattr(ruler_mod, "_measure_integrated_once", _flaky_integ_once)
    oom_retry_dir = tmp_path / "oom_retry"
    res = run_ruler(
        output_dir=oom_retry_dir,
        device="cpu",
        envs_list=[128],
        net_sizes=["S"],
        burnin_ticks=2,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=384,
        actor_chunk_envs=128,
        max_capacity=64,
        include_best_compile=False,
        measure_larger_minibatch=False,
    )
    cell = res["cells"][0]
    assert cell["oom"] is False
    assert calls == [(128, 384), (64, 192)]
    integ = cell["integrated"]
    assert integ["oom_retries"] == 1
    assert integ["actor_chunk_envs_used"] == 64
    assert integ["minibatch_size_used"] == 192
    assert integ["stage_peak_vram_bytes"]["learner"] == 96_000_000

    # Also verify unrecoverable OOM at floor records oom=True and continues to next cell
    orig_coll_once = ruler_mod._measure_collection_only_once

    def _hard_oom_on_e4(**kwargs: Any) -> dict[str, Any]:
        if kwargs["num_envs"] == 4:
            raise torch.OutOfMemoryError("CUDA out of memory even at floor")
        return orig_coll_once(**kwargs)

    monkeypatch.setattr(ruler_mod, "_measure_integrated_once", orig_integ_once)
    monkeypatch.setattr(ruler_mod, "_measure_collection_only_once", _hard_oom_on_e4)
    hard_dir = tmp_path / "oom_hard"
    res_hard = run_ruler(
        output_dir=hard_dir,
        device="cpu",
        envs_list=[2, 4],
        net_sizes=["S"],
        burnin_ticks=2,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=3,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=64,
        include_best_compile=False,
        measure_larger_minibatch=False,
    )
    assert res_hard["cells"][0]["oom"] is False
    assert res_hard["cells"][1]["oom"] is True
    assert res_hard["cells"][1]["collection_only"]["oom"] is True


def test_canary_suite_real_exports_no_stubs_and_custom_board_ruler(tmp_path: Path) -> None:
    """Verify M2/M4/MINOR 8 real canary suite (zero stubs) + Coordinator #7 custom board ruler."""
    assert not hasattr(ruler_mod, "write_canary_export_probes")

    canary_dir = tmp_path / "canary_real_out"
    suite = run_canary_suite(
        output_dir=canary_dir,
        device="cpu",
        num_envs=4,
        rollout_len=6,
        seed=2026,
    )
    assert suite["indexed_parity"]["status"] == "PASS"
    assert suite["train_result"]["canary_export_probe"] is False
    assert suite["train_result"]["updates_completed"] == 2

    for rel in CANARY_REQUIRED_EXPORTS:
        assert verify_completed_artifact(canary_dir / rel)
    assert verify_completed_artifact(canary_dir / "cells" / "E4_ego_only_S.json")
    assert verify_completed_artifact(canary_dir / "cells" / "E4_shared_trunk_S.json")
    assert verify_completed_artifact(canary_dir / "cells" / "E4_per_seat_S.json")
    assert list((canary_dir / "checkpoints").glob("ckpt_u*.pt")) == []

    # Prove final.pt from canary_suite is a genuine loadable trainer checkpoint, not a stub
    cfg = ArenaPPOConfig.from_dict(suite["train_result"]["config"])
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    loaded = trainer.load_checkpoint(canary_dir / "checkpoints" / "final.pt")
    assert loaded["counters"]["updates_completed"] == 2

    # Coordinator #7: ruler at custom board size (48x64) and snake count (4)
    custom_dir = tmp_path / "custom_board_ruler"
    res_custom = run_ruler(
        output_dir=custom_dir,
        device="cpu",
        envs_list=[2],
        net_sizes=["S"],
        grid_h=48,
        grid_w=64,
        num_snakes=4,
        burnin_ticks=10,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=8,
        max_capacity=64,
        include_best_compile=False,
        measure_larger_minibatch=False,
    )
    assert res_custom["batch6_forward"]["S"]["batch_size"] == 4
    assert res_custom["batch6_forward"]["S"]["token_grid"] == [6, 8]
    assert res_custom["cells"][0]["grid_h"] == 48
    assert res_custom["cells"][0]["grid_w"] == 64
    assert res_custom["cells"][0]["num_snakes"] == 4
    assert res_custom["cells"][0]["token_grid"] == [6, 8]


def _stage_manifest_inputs(mf: dict[str, Any], remote_inputs_dir: Path) -> Path:
    """Copy manifest `inputs` into simulated remote `$COLAB_INPUT_DIR`."""
    remote_inputs_dir.mkdir(parents=True, exist_ok=True)
    for inp in mf["inputs"]:
        dst = remote_inputs_dir / str(inp["dest"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inp["path"], dst)
    return remote_inputs_dir


def _build_synthetic_canary_receipt(
    canary_mf: dict[str, Any],
    gpu_kind: str,
) -> Path:
    """Create a broker-verified canary receipt from real `run_canary_suite` exports."""
    canary_out = Path(canary_mf["output_dir"])
    canary_exports_dir = canary_out / "exports"
    remote_inputs_dir = _stage_manifest_inputs(canary_mf, canary_out / "remote_inputs")
    ledger_cand = remote_inputs_dir / "verified-ledger.json"
    suite_res = run_canary_suite(
        output_dir=canary_exports_dir,
        device="cpu",
        input_dir=remote_inputs_dir,
        ledger_path=ledger_cand if ledger_cand.is_file() else None,
        num_envs=2,
        rollout_len=4,
        seed=777,
    )
    (canary_out / "canary_suite_summary.json").write_text(
        json.dumps(suite_res, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    saved_probe_mf = canary_out / "manifest.json"
    saved_probe_mf.write_text(json.dumps(canary_mf, indent=2, sort_keys=True) + "\n")
    (canary_out / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 3.0,
                "transfer_seconds": 1.0,
                "source_delivery_seconds": 1.0,
                "staging_completed": True,
            }
        )
    )
    (canary_out / "hardware.json").write_text(json.dumps({"hardware": {"kind": gpu_kind}}))
    export_inventory = []
    for fpath in sorted(canary_exports_dir.rglob("*")):
        if fpath.is_file():
            export_inventory.append(
                {
                    "path": fpath.relative_to(canary_exports_dir).as_posix(),
                    "size": fpath.stat().st_size,
                    "sha256": sha256_file(fpath),
                }
            )
    transfers = []
    for inp in canary_mf["inputs"]:
        transfers.append(
            {
                "direction": "upload",
                "remote": f"/content/snake-cli/gen-test-01/inputs/{inp['dest']}",
                "status": "verified",
                "size_bytes": Path(inp["path"]).stat().st_size,
                "sha256": inp["sha256"],
            }
        )
    receipt_obj: dict[str, Any] = {
        "status": "PASS",
        "stop_verified": True,
        "error": None,
        "cleanup_error": None,
        "failure_class": None,
        "commit": canary_mf["commit"],
        "job_id": canary_mf["job_id"],
        "generation": "gen-test-01",
        "manifest_sha256": sha256_file(saved_probe_mf),
        "exports": export_inventory,
        "transfers": transfers,
        "terminal": {
            "job_id": canary_mf["job_id"],
            "generation": "gen-test-01",
            "returncode": 0,
            "status": "completed",
            "export_scanner": {"name": "runtime._exports", "inventory": export_inventory},
            "canary_observations": [
                {
                    "cuda_decisions": 12,
                    "device": "cuda:0",
                    "package": "research/arena_ppo",
                    "caller": "research/arena_ppo/net.py:act",
                    "operation": "torch.multinomial",
                    "shape": [12, 6],
                    "root": "/content/snake-cli/gen-test-01",
                }
            ],
        },
    }
    receipt_obj["canary_evidence"] = colab_canary.evidence(canary_mf, receipt_obj, canary_out, ROOT)
    receipt_path = canary_out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt_obj, indent=2, sort_keys=True) + "\n")
    return receipt_path


def test_prepare_three_modes_l4_a100_t4_rejected_and_offline_supervisor_validate(
    tmp_path: Path,
) -> None:
    """Validate canary, ruler, and train packets on L4 & A100 against schema & supervisor."""
    for exports in (CANARY_REQUIRED_EXPORTS, RULER_REQUIRED_EXPORTS, TRAIN_REQUIRED_EXPORTS):
        for rel in exports:
            assert not rel.endswith(".sha256")
            assert not rel.endswith(".complete")

    # MINOR 10: T4 has no bf16 and must be rejected
    with pytest.raises(ValueError, match="T4"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_t4_refused",
            commit="HEAD",
            mode="canary",
            gpu="T4",
            output_dir=tmp_path / "out_t4",
            allow_unmerged_commit=True,
        )

    # Unmerged HEAD must be refused when allow_unmerged_commit=False
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    is_merged = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", head_sha, "origin/main"],
            capture_output=True,
        ).returncode
        == 0
    )
    if not is_merged:
        with pytest.raises(ValueError, match="origin/main"):
            prepare_arena_ppo_packet(
                tmp_path / "refused_pkt",
                commit="HEAD",
                mode="canary",
                gpu="L4",
                output_dir=tmp_path / "refused_out",
                allow_unmerged_commit=False,
            )

    # 1. Prepare `canary` on L4 and A100 (<= 12 min, instrumented)
    canary_l4 = prepare_arena_ppo_packet(
        tmp_path / "pkt_canary_l4",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_canary_l4",
        allow_unmerged_commit=True,
    )
    canary_a100 = prepare_arena_ppo_packet(
        tmp_path / "pkt_canary_a100",
        commit="HEAD",
        mode="canary",
        gpu="A100",
        output_dir=tmp_path / "out_canary_a100",
        allow_unmerged_commit=True,
    )
    mf_canary_l4 = json.loads(Path(canary_l4["manifest_path"]).read_text(encoding="utf-8"))
    mf_canary_a100 = json.loads(Path(canary_a100["manifest_path"]).read_text(encoding="utf-8"))

    for mf in (mf_canary_l4, mf_canary_a100):
        assert mf["timeout_minutes"] == 30.0
        assert mf["requirements"]["est_minutes"] == 15.0
        assert mf["env"]["ARENA_PPO_MAX_WALL_SECONDS"] == "720"
        assert mf["env"]["SNAKE_COLAB_CANARY"] == "1"
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)

    rcpt_l4 = _build_synthetic_canary_receipt(mf_canary_l4, "L4")
    rcpt_a100 = _build_synthetic_canary_receipt(mf_canary_a100, "A100")

    # 2. Prepare `ruler` science packets on L4 and A100 bound to their canary receipts
    ruler_l4 = prepare_arena_ppo_packet(
        tmp_path / "pkt_ruler_l4",
        commit="HEAD",
        mode="ruler",
        gpu="L4",
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_ruler_l4",
        allow_unmerged_commit=True,
    )
    ruler_a100 = prepare_arena_ppo_packet(
        tmp_path / "pkt_ruler_a100",
        commit="HEAD",
        mode="ruler",
        gpu="A100",
        canary_receipt=rcpt_a100,
        output_dir=tmp_path / "out_ruler_a100",
        allow_unmerged_commit=True,
    )
    mf_ruler_l4 = json.loads(Path(ruler_l4["manifest_path"]).read_text(encoding="utf-8"))
    mf_ruler_a100 = json.loads(Path(ruler_a100["manifest_path"]).read_text(encoding="utf-8"))

    for mf in (mf_ruler_l4, mf_ruler_a100):
        assert "SNAKE_COLAB_CANARY" not in mf["env"]
        assert mf["timeout_minutes"] == 60.0
        assert mf["env"]["ARENA_PPO_MAX_WALL_SECONDS"] == "3120"
        assert mf["resume_exemption"]["kind"] == "acceptance"
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)

    # Hardware mismatch between --gpu and canary receipt must be refused
    with pytest.raises(ValueError, match="does not match canary receipt hardware"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_hw_mismatch",
            commit="HEAD",
            mode="ruler",
            gpu="A100",
            canary_receipt=rcpt_l4,
            output_dir=tmp_path / "out_hw_mismatch",
            allow_unmerged_commit=True,
        )

    # 3. Prepare `train` science packet bound to L4 canary receipt with sized max_export_bytes
    train_l4 = prepare_arena_ppo_packet(
        tmp_path / "pkt_train_l4",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=6.0,
        expected_steps_per_sec=150_000.0,
        canary_receipt=rcpt_l4,
        measured_checkpoint_bytes=40 * 1024 * 1024,
        output_dir=tmp_path / "out_train_l4",
        allow_unmerged_commit=True,
    )
    assert "training_plan" in train_l4
    assert train_l4["training_plan"]["expected_updates"] > 100
    mf_train_l4 = json.loads(Path(train_l4["manifest_path"]).read_text(encoding="utf-8"))
    assert "SNAKE_COLAB_CANARY" not in mf_train_l4["env"]
    assert mf_train_l4["timeout_minutes"] == 360.0
    assert mf_train_l4["env"]["ARENA_PPO_HOURS"] == "6.0"
    assert mf_train_l4["env"]["ARENA_PPO_EXPECTED_STEPS_PER_SEC"] == "150000.0"
    parsed_train_cfg = json.loads(mf_train_l4["env"]["ARENA_PPO_CONFIG_JSON"])
    assert parsed_train_cfg["total_updates"] > 100
    assert mf_train_l4["max_export_bytes"] < 2 * 1024**3
    colab_schema.validate_manifest(mf_train_l4)
    ops_dispatch.require_resume_policy(mf_train_l4)
    colab_supervisor.validate(mf_train_l4)


def test_major1_r3_end_to_end_continuations_both_prior_kinds(tmp_path: Path) -> None:
    """MAJOR-1: end-to-end continuation canary + resumed train/ruler for BOTH prior kinds."""
    # Ambiguous directory containing BOTH checkpoints/ and cells/ must refuse --prior-kind auto
    ambig_dir = tmp_path / "ambig_prior"
    run_canary_suite(output_dir=ambig_dir, device="cpu", num_envs=2, rollout_len=4, seed=11)
    with pytest.raises(ValueError, match="Ambiguous prior_output_dir"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_ambig_auto",
            commit="HEAD",
            mode="canary",
            gpu="L4",
            prior_output_dir=ambig_dir,
            prior_kind="auto",
            output_dir=tmp_path / "out_ambig_auto",
            allow_unmerged_commit=True,
        )

    # ---- Case A: Train checkpoint continuation (attempt-01 -> cont canary -> attempt-02) ----
    prior_train_out = tmp_path / "prior_train_attempt01"
    prior_train_cfg = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="S",
        num_envs=6,  # distinct from canary's num_envs=2 to prove 0-update staged load verification
        rollout_len=8,
        ppo_epochs=1,
        minibatch_size=24,
        actor_chunk_envs=6,
        max_capacity=96,
        total_updates=2,
        seed=20261110,
        deterministic=True,
    )
    run_training_job(
        cfg=prior_train_cfg,
        output_dir=prior_train_out,
        work_dir=tmp_path / "prior_train_work",
        device="cpu",
    )
    assert verify_completed_artifact(prior_train_out / "checkpoints" / "final.pt")

    # Mismatched prior_kind for mode=train must be rejected
    with pytest.raises(ValueError, match="mode='train'"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_bad_train_prior",
            commit="HEAD",
            mode="train",
            gpu="L4",
            prior_output_dir=prior_train_out,
            prior_kind="ruler_cells",
            output_dir=tmp_path / "out_bad_train_prior",
            allow_unmerged_commit=True,
        )

    cont_canary_ckpt = prepare_arena_ppo_packet(
        tmp_path / "pkt_cont_canary_ckpt",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        prior_output_dir=prior_train_out,
        prior_kind="auto",  # auto-detects "checkpoint" when only checkpoints/ exist
        output_dir=tmp_path / "out_cont_canary_ckpt",
        attempt="attempt-02",
        allow_unmerged_commit=True,
    )
    mf_cont_canary_ckpt = json.loads(
        Path(cont_canary_ckpt["manifest_path"]).read_text(encoding="utf-8")
    )
    assert len(mf_cont_canary_ckpt["inputs"]) == 2
    colab_schema.validate_manifest(mf_cont_canary_ckpt)
    colab_supervisor.validate(mf_cont_canary_ckpt)

    rcpt_cont_ckpt = _build_synthetic_canary_receipt(mf_cont_canary_ckpt, "L4")
    suite_ckpt_summary = json.loads(
        (Path(mf_cont_canary_ckpt["output_dir"]) / "canary_suite_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert suite_ckpt_summary["staged_input_verification"]["kind"] == "checkpoint"
    assert suite_ckpt_summary["staged_input_verification"]["updates_completed"] == 2
    assert suite_ckpt_summary["train_result"]["updates_completed"] == 2

    cont_train = prepare_arena_ppo_packet(
        tmp_path / "pkt_cont_train_02",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        prior_output_dir=prior_train_out,
        prior_kind="checkpoint",
        canary_receipt=rcpt_cont_ckpt,
        ppo_config=ArenaPPOConfig(**{**prior_train_cfg.to_dict(), "total_updates": 4}),
        explicit_total_updates=True,
        output_dir=tmp_path / "out_cont_train_02",
        attempt="attempt-02",
        allow_unmerged_commit=True,
    )
    mf_cont_train = json.loads(Path(cont_train["manifest_path"]).read_text(encoding="utf-8"))
    colab_canary.check(mf_cont_train, ROOT)
    colab_schema.validate_manifest(mf_cont_train)
    ops_dispatch.require_resume_policy(mf_cont_train)
    colab_supervisor.validate(mf_cont_train)

    # Execute resumed training job offline from the staged manifest inputs (updates 2 -> 4)
    staged_train_inputs = _stage_manifest_inputs(mf_cont_train, tmp_path / "exec_cont_train_in")
    res_cont_train = run_training_job(
        cfg=ArenaPPOConfig.from_json(mf_cont_train["env"]["ARENA_PPO_CONFIG_JSON"]),
        output_dir=Path(mf_cont_train["output_dir"]),
        work_dir=tmp_path / "exec_cont_train_work",
        device="cpu",
        input_dir=staged_train_inputs,
        ledger_path=staged_train_inputs / "verified-ledger.json",
    )
    assert res_cont_train["resumed_from"] is not None
    assert res_cont_train["updates_completed"] == 4

    # ---- Case B: Partial ruler cells continuation (attempt-01 -> cont canary -> attempt-02) ----
    prior_ruler_out = tmp_path / "prior_ruler_attempt01"
    run_ruler(
        output_dir=prior_ruler_out,
        device="cpu",
        envs_list=[2, 4],
        net_sizes=["S"],
        burnin_ticks=4,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=3,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=64,
        include_best_compile=False,
        measure_larger_minibatch=False,
        step_hook=lambda cid: (
            os.kill(os.getpid(), signal.SIGUSR1)
            if cid == "E2_S" and hasattr(signal, "SIGUSR1")
            else None
        ),
    )
    assert verify_completed_artifact(prior_ruler_out / "cells" / "E2_S.json")
    assert not (prior_ruler_out / "cells" / "E4_S.json").exists()

    cont_canary_cells = prepare_arena_ppo_packet(
        tmp_path / "pkt_cont_canary_cells",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        prior_output_dir=prior_ruler_out,
        prior_kind="auto",  # auto-detects "ruler_cells" when only cells/*.json exist!
        output_dir=tmp_path / "out_cont_canary_cells",
        attempt="ruler-attempt-02",
        allow_unmerged_commit=True,
    )
    mf_cont_canary_cells = json.loads(
        Path(cont_canary_cells["manifest_path"]).read_text(encoding="utf-8")
    )
    assert any(inp["dest"] == "cells/E2_S.json" for inp in mf_cont_canary_cells["inputs"])
    colab_schema.validate_manifest(mf_cont_canary_cells)
    colab_supervisor.validate(mf_cont_canary_cells)

    rcpt_cont_cells = _build_synthetic_canary_receipt(mf_cont_canary_cells, "L4")
    suite_cells_summary = json.loads(
        (Path(mf_cont_canary_cells["output_dir"]) / "canary_suite_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert suite_cells_summary["staged_input_verification"] == {
        "kind": "ruler_cells",
        "verified_cells": ["E2_S"],
    }

    cont_ruler = prepare_arena_ppo_packet(
        tmp_path / "pkt_cont_ruler_02",
        commit="HEAD",
        mode="ruler",
        gpu="L4",
        prior_output_dir=prior_ruler_out,
        prior_kind="ruler_cells",
        canary_receipt=rcpt_cont_cells,
        output_dir=tmp_path / "out_cont_ruler_02",
        attempt="attempt-02",
        allow_unmerged_commit=True,
    )
    mf_cont_ruler = json.loads(Path(cont_ruler["manifest_path"]).read_text(encoding="utf-8"))
    colab_canary.check(mf_cont_ruler, ROOT)
    colab_schema.validate_manifest(mf_cont_ruler)
    ops_dispatch.require_resume_policy(mf_cont_ruler)
    colab_supervisor.validate(mf_cont_ruler)

    staged_ruler_inputs = _stage_manifest_inputs(mf_cont_ruler, tmp_path / "exec_cont_ruler_in")
    res_cont_ruler = run_ruler(
        output_dir=Path(mf_cont_ruler["output_dir"]),
        device="cpu",
        input_dir=staged_ruler_inputs,
        envs_list=[2, 4],
        net_sizes=["S"],
        burnin_ticks=4,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=3,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=64,
        include_best_compile=False,
        measure_larger_minibatch=False,
    )
    assert res_cont_ruler["reused_cells"] == ["E2_S"]
    assert [c["cell_id"] for c in res_cont_ruler["cells"]] == ["E2_S", "E4_S"]


def test_minor3_minor5_minor10_r3_nanquantile_compile180s_and_mid_tick_ruler_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify MINOR 3 nanquantile, MINOR 5 >=180s compile est, MINOR 10 mid-tick ruler signal."""
    # MINOR 3: _WorldMaturityTracker stores fixed-shape NaN-masked tensors (no boolean indexing)
    sim = TrainSim.create(2, device="cpu", seed=123, max_capacity=64)
    sim.prepare()
    sim.sim.alive[:] = torch.tensor(
        [[True, False, True, True, False, True], [False, True, False, False, True, False]]
    )
    sim.sim.length[:] = torch.tensor([[10, 999, 20, 30, 888, 40], [777, 50, 666, 555, 100, 444]])
    tracker = _WorldMaturityTracker(torch.device("cpu"))
    died_zero = torch.zeros((2, 6), dtype=torch.bool)
    tracker.record_tick(sim, died_zero, sample_stride=1)
    assert len(tracker._length_samples) == 1
    assert tracker._length_samples[0].shape == (12,)
    assert int(torch.isnan(tracker._length_samples[0]).sum().item()) == 6
    summary = tracker.summarize()
    # Alive lengths are [10, 20, 30, 40, 50, 100] -> mean = 250 / 6 = 41.667
    assert abs(summary["mean_length"] - (250.0 / 6.0)) < 1e-2
    assert 50.0 <= summary["p95_length"] <= 100.0

    # All-dead tick must produce finite 0.0 p95_length without NaN crash
    empty_tracker = _WorldMaturityTracker(torch.device("cpu"))
    sim.sim.alive.fill_(False)
    empty_tracker.record_tick(sim, died_zero, sample_stride=1)
    assert empty_tracker.summarize()["p95_length"] == 0.0

    # MINOR 5: compile projection adds >= 180s
    cpu_dev = torch.device("cpu")
    est_no_compile = ruler_mod._project_cell_seconds(
        num_envs=1024, net_sz="S", is_compile=False, device=cpu_dev, sec_per_env_by_net={}
    )
    est_compile = ruler_mod._project_cell_seconds(
        num_envs=1024, net_sz="S", is_compile=True, device=cpu_dev, sec_per_env_by_net={}
    )
    assert est_no_compile == 0.0
    assert est_compile >= 180.0

    est_base = ruler_mod._project_cell_seconds(
        num_envs=1024,
        net_sz="S",
        is_compile=False,
        device=cpu_dev,
        sec_per_env_by_net={"S": 0.02},
    )
    est_comp_after_base = ruler_mod._project_cell_seconds(
        num_envs=1024,
        net_sz="S",
        is_compile=True,
        device=cpu_dev,
        sec_per_env_by_net={"S": 0.02},
    )
    assert (est_comp_after_base - est_base) >= 180.0

    # MINOR 10: mid-tick SIGUSR1 inside run_ruler (during cell E4_S after E2_S completes)
    if hasattr(signal, "SIGUSR1"):
        orig_step = TrainSim.step
        fired = {"done": False}

        def _mid_tick_ruler_sig(self: TrainSim, actions: torch.Tensor) -> StepInfo:
            out = orig_step(self, actions)
            # Fire inside timed collection of E4_S (burnin_ticks=0 -> very first E=4 step is timed!)
            if self.E == 4 and not fired["done"]:
                fired["done"] = True
                os.kill(os.getpid(), signal.SIGUSR1)
            return out

        monkeypatch.setattr(TrainSim, "step", _mid_tick_ruler_sig)
        mid_sig_dir = tmp_path / "ruler_mid_tick_sig"
        res_mid = run_ruler(
            output_dir=mid_sig_dir,
            device="cpu",
            envs_list=[2, 4],
            net_sizes=["S"],
            burnin_ticks=0,
            warmup_ticks=1,
            collect_ticks=3,
            rollout_len=4,
            ppo_epochs=1,
            minibatch_size=12,
            max_capacity=64,
            include_best_compile=False,
            measure_larger_minibatch=False,
        )
        assert fired["done"] is True
        assert res_mid["partial"] is True
        assert res_mid["stopped_reason"] == "signal:SIGUSR1"
        assert [c["cell_id"] for c in res_mid["cells"]] == ["E2_S"]
        assert verify_completed_artifact(mid_sig_dir / "cells" / "E2_S.json")
        assert not (mid_sig_dir / "cells" / "E4_S.json").exists()
        assert verify_completed_artifact(mid_sig_dir / "ruler_result.json")


def test_multi_arch_ruler_grid_cheapest_first_one_epoch_cell_learner_report_and_prepare(
    tmp_path: Path,
) -> None:
    """Verify --net-arch x size x E cheapest-first order, 1-epoch probe, FLOPs, MPS, and prepare."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = tmp_path / "multi_arch_ruler"
    res = run_ruler(
        output_dir=out_dir,
        device=dev,
        envs_list=[4, 2],
        net_sizes=["S"],
        net_archs=["per_seat", "shared_trunk", "ego_only"],
        burnin_ticks=4,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=4,
        ppo_epochs=2,
        include_one_epoch_cell=True,
        minibatch_size=12,
        actor_chunk_envs=2,
        max_capacity=64,
        include_best_compile=False,
        measure_larger_minibatch=False,
        seed=2026,
    )

    if torch.backends.mps.is_available():
        canary_mps_out = tmp_path / "canary_mps_shared_trunk"
        suite_mps = run_canary_suite(
            output_dir=canary_mps_out,
            device="mps",
            num_envs=4,
            rollout_len=4,
            seed=2026,
        )
        assert suite_mps["train_result"]["updates_completed"] == 2
        assert suite_mps["train_result"]["config"]["net_arch"] == "shared_trunk"

    cell_ids = [c["cell_id"] for c in res["cells"]]
    assert cell_ids == [
        "E2_ego_only_S",
        "E2_shared_trunk_S",
        "E2_shared_trunk_S_ep1",
        "E4_ego_only_S",
        "E4_shared_trunk_S",
        "E2_per_seat_S",
        "E4_per_seat_S",
    ]
    assert res["include_one_epoch_cell"] is True
    assert res["one_epoch_comparison"] is not None
    assert res["one_epoch_comparison"]["base_cell_id"] == "E2_shared_trunk_S"
    assert res["one_epoch_comparison"]["one_epoch_cell_id"] == "E2_shared_trunk_S_ep1"
    assert res["one_epoch_comparison"]["base_ppo_epochs"] == 2

    # Learner settings & FLOPs reported in top-level and per-cell artifacts
    assert res["learner_settings"]["ppo_epochs"] == 2
    assert "shared_trunk_S" in res["learner_settings"]["per_variant_defaults"]
    for bkey in ("ego_only_S", "shared_trunk_S", "per_seat_S"):
        assert res["batch6_forward"][bkey]["forward_flops_per_row"] > 0
        assert res["batch6_forward"][bkey]["learner_flops_per_row"] > 0
    for cell in res["cells"]:
        assert cell["forward_flops_per_row"] > 0
        assert cell["learner_flops_per_row"] > 0
        assert cell["learner_settings"]["minibatch_size"] == 12
        assert cell["learner_settings"]["world_ticks_per_minibatch"] == 2

    # Verify job.sh and prepare.py set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + --net-arch
    job_sh_text = (ROOT / "research" / "arena_ppo" / "job.sh").read_text(encoding="utf-8")
    assert "expandable_segments:True" in job_sh_text

    pkt_r = tmp_path / "pkt_ruler_arch"
    info_r = prepare_arena_ppo_packet(
        pkt_r,
        commit="HEAD",
        mode="ruler",
        gpu="L4",
        net_arch="ego_only,shared_trunk",
        output_dir=tmp_path / "out_ruler_arch",
        allow_unmerged_commit=True,
    )
    mf_r = json.loads(Path(info_r["manifest_path"]).read_text(encoding="utf-8"))
    assert mf_r["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert mf_r["env"]["ARENA_PPO_RULER_ARCHS"] == "ego_only,shared_trunk"

    pkt_t = tmp_path / "pkt_train_arch"
    info_t = prepare_arena_ppo_packet(
        pkt_t,
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=6.0,
        expected_steps_per_sec=20000.0,
        net_arch="shared_trunk",
        net_size="S",
        num_envs=2048,
        output_dir=tmp_path / "out_train_arch",
        allow_unmerged_commit=True,
    )
    mf_t = json.loads(Path(info_t["manifest_path"]).read_text(encoding="utf-8"))
    assert mf_t["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert mf_t["env"]["ARENA_PPO_NET_ARCH"] == "shared_trunk"
    cfg_t = json.loads(mf_t["env"]["ARENA_PPO_CONFIG_JSON"])
    assert cfg_t["net_arch"] == "shared_trunk"
    assert cfg_t["num_envs"] == 2048
    assert cfg_t["actor_chunk_envs"] == 2048
    assert cfg_t["minibatch_size"] == 3072


def test_m1_ruler_isolates_non_oom_cell_and_bench_errors_and_canary_probes_all_archs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: per-cell/per-bench non-OOM isolation, exit 0 if >=1 cell succeeds, 3-arch canary."""
    # 1. Verify prepare.py canary docstring and exemption reason match 15/30 (not <= 12 min)
    assert prepare_arena_ppo_packet.__doc__ is not None
    assert "<= 12 min" not in prepare_arena_ppo_packet.__doc__
    assert "15 min" in prepare_arena_ppo_packet.__doc__
    assert "30 min" in prepare_arena_ppo_packet.__doc__
    pkt_c = prepare_arena_ppo_packet(
        tmp_path / "pkt_canary_15_30",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_canary_15_30",
        allow_unmerged_commit=True,
    )
    mf_c = json.loads(Path(pkt_c["manifest_path"]).read_text(encoding="utf-8"))
    reason_text = mf_c["resume_exemption"]["reason"]
    assert "<= 12 min" not in reason_text
    assert "15 min" in reason_text and "30 min" in reason_text

    # 2. Non-OOM exceptions in batch6_forward (ego_only) and in cells (ego_only collection,
    # per_seat integrated) must NOT abort shared_trunk; CLI exits 0 and ruler_result.json PASSes.
    orig_bench = ruler_mod.measure_batch6_forward_ms
    orig_coll = ruler_mod._measure_collection_only_cell
    orig_integ = ruler_mod._measure_integrated_cell
    long_detail = "x" * 2400

    def _faulty_bench(device: torch.device, net_sz: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("net_arch") == "ego_only":
            raise ValueError(f"synthetic batch6 bug on ego_only {long_detail}")
        return orig_bench(device, net_sz, **kwargs)

    def _faulty_coll(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("net_arch") == "ego_only":
            raise RuntimeError(f"synthetic collection bug on ego_only {long_detail}")
        return orig_coll(**kwargs)

    def _faulty_integ(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("net_arch") == "per_seat":
            raise ArithmeticError("synthetic integrated learner bug on per_seat")
        return orig_integ(**kwargs)

    monkeypatch.setattr(ruler_mod, "measure_batch6_forward_ms", _faulty_bench)
    monkeypatch.setattr(ruler_mod, "_measure_collection_only_cell", _faulty_coll)
    monkeypatch.setattr(ruler_mod, "_measure_integrated_cell", _faulty_integ)

    out_partial = tmp_path / "ruler_partial_errors"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research.arena_ppo.ruler",
            "--output-dir",
            str(out_partial),
            "--device",
            "cpu",
            "--envs",
            "2",
            "--nets",
            "S",
            "--net-arch",
            "ego_only,shared_trunk,per_seat",
            "--burnin-ticks",
            "2",
            "--warmup-ticks",
            "1",
            "--collect-ticks",
            "2",
            "--rollout-len",
            "2",
            "--ppo-epochs",
            "1",
            "--no-one-epoch-cell",
            "--no-best-compile",
            "--no-larger-minibatch",
            "--max-capacity",
            "64",
        ],
    )
    rc = ruler_mod.main()
    assert rc == 0
    assert verify_completed_artifact(out_partial / "ruler_result.json")
    for cid in ("E2_ego_only_S", "E2_shared_trunk_S", "E2_per_seat_S"):
        assert verify_completed_artifact(out_partial / "cells" / f"{cid}.json")

    res_p = json.loads((out_partial / "ruler_result.json").read_text(encoding="utf-8"))
    assert res_p["succeeded_cell_ids"] == ["E2_shared_trunk_S"]
    assert res_p["failed_cell_ids"] == ["E2_ego_only_S", "E2_per_seat_S"]
    assert len(res_p["failed_benchmarks"]) == 1
    assert res_p["failed_benchmarks"][0]["benchmark_key"] == "ego_only_S"
    assert res_p["failed_benchmarks"][0]["error"].startswith("ValueError: synthetic batch6 bug")
    assert 0 < len(res_p["failed_benchmarks"][0]["traceback"]) <= 2000

    ego_cell = json.loads(
        (out_partial / "cells" / "E2_ego_only_S.json").read_text(encoding="utf-8")
    )
    assert ego_cell["error"].startswith("RuntimeError: synthetic collection bug on ego_only")
    assert 0 < len(ego_cell["traceback"]) <= 2000
    assert ego_cell["integrated"] == {"skipped_after_collection_error": True}

    ps_cell = json.loads((out_partial / "cells" / "E2_per_seat_S.json").read_text(encoding="utf-8"))
    assert ps_cell["error"] == "ArithmeticError: synthetic integrated learner bug on per_seat"
    assert 0 < len(ps_cell["traceback"]) <= 2000

    assert res_p["status_summary"].startswith("PARTIAL_PASS: 1/3 cells succeeded")
    assert "E2_ego_only_S" in res_p["status_summary"]
    assert "E2_per_seat_S" in res_p["status_summary"]

    # 3. With that same per_seat bug active, run_canary_suite MUST fail loudly so CUDA-only bugs
    # in ANY of the 3 architectures fail the canary immediately after probing all 3.
    with pytest.raises(RuntimeError, match="E2_per_seat_S"):
        run_canary_suite(
            output_dir=tmp_path / "canary_catches_arch_bug",
            device="cpu",
            num_envs=2,
            rollout_len=2,
            seed=99,
        )

    # 4. If ZERO cells succeed (every cell raises non-OOM error), ruler writes ruler_result.json
    # and raises RuntimeError (exits non-zero).
    monkeypatch.setattr(
        ruler_mod,
        "_measure_collection_only_cell",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("all cells broken")),
    )
    all_fail_dir = tmp_path / "ruler_all_fail"
    with pytest.raises(RuntimeError, match="FAIL: 0/1 cells succeeded"):
        run_ruler(
            output_dir=all_fail_dir,
            device="cpu",
            envs_list=[2],
            net_sizes=["S"],
            net_archs=["shared_trunk"],
            burnin_ticks=2,
            warmup_ticks=1,
            collect_ticks=2,
            rollout_len=2,
            ppo_epochs=1,
            include_one_epoch_cell=False,
            include_best_compile=False,
            measure_larger_minibatch=False,
            max_capacity=64,
        )
    assert verify_completed_artifact(all_fail_dir / "ruler_result.json")

    # 5. StopRequestedError and KeyboardInterrupt must still re-raise (never swallowed as cell err)
    monkeypatch.setattr(ruler_mod, "measure_batch6_forward_ms", orig_bench)
    monkeypatch.setattr(
        ruler_mod,
        "_measure_collection_only_cell",
        lambda **_kw: (_ for _ in ()).throw(KeyboardInterrupt("ctrl-c")),
    )
    with pytest.raises(KeyboardInterrupt, match="ctrl-c"):
        run_ruler(
            output_dir=tmp_path / "ruler_kb_interrupt",
            device="cpu",
            envs_list=[2],
            net_sizes=["S"],
            net_archs=["shared_trunk"],
            burnin_ticks=2,
            warmup_ticks=1,
            collect_ticks=2,
            rollout_len=2,
            ppo_epochs=1,
            include_one_epoch_cell=False,
            include_best_compile=False,
            measure_larger_minibatch=False,
            max_capacity=64,
        )
    monkeypatch.setattr(
        ruler_mod,
        "_measure_collection_only_cell",
        lambda **_kw: (_ for _ in ()).throw(StopRequestedError("sigusr1")),
    )
    stop_res = run_ruler(
        output_dir=tmp_path / "ruler_stop_req",
        device="cpu",
        envs_list=[2],
        net_sizes=["S"],
        net_archs=["shared_trunk"],
        burnin_ticks=2,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=2,
        ppo_epochs=1,
        include_one_epoch_cell=False,
        include_best_compile=False,
        measure_larger_minibatch=False,
        max_capacity=64,
    )
    assert stop_res["stopped_reason"].startswith("signal:")
    assert stop_res["failed_cell_ids"] == []

    # 6. Restore real functions and verify healthy 3-arch canary runs 1 minibatch per arch
    monkeypatch.setattr(ruler_mod, "_measure_collection_only_cell", orig_coll)
    monkeypatch.setattr(ruler_mod, "_measure_integrated_cell", orig_integ)
    healthy_canary_dir = tmp_path / "canary_3arch_ok"
    healthy_suite = run_canary_suite(
        output_dir=healthy_canary_dir,
        device="cpu",
        num_envs=4,
        rollout_len=4,
        seed=1234,
    )
    rr = healthy_suite["ruler_result"]
    assert rr["succeeded_cell_ids"] == [
        "E4_ego_only_S",
        "E4_shared_trunk_S",
        "E4_per_seat_S",
    ]
    for cell in rr["cells"]:
        ls = cell["integrated"]["learner_settings"]
        assert ls["ppo_epochs"] == 1
        assert ls["minibatch_size"] == ls["total_rows_per_epoch"]


def test_m2_five_session_continuation_chain_and_fast_gpu_with_fake_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2: 5-session continuation chain + fast single session stop on wall guard, never cap."""
    from research.arena_ppo import ppo as ppo_mod

    fake_now = {"t": 1_000.0}
    step_wall_advance = {"sec": 110.0}
    monkeypatch.setattr(ppo_mod.time, "monotonic", lambda: fake_now["t"])

    base_cfg = ArenaPPOConfig(
        net_arch="ego_only",
        net_size="S",
        num_envs=1,
        num_snakes=4,
        grid_h=24,
        grid_w=32,
        rollout_len=2,  # 8 agent-steps per update
        ppo_epochs=1,
        minibatch_size=8,
        actor_chunk_envs=1,
        max_capacity=32,
        seed=20261111,
        deterministic=True,
    )

    # At expected_steps_per_sec = 0.08 (8 / 0.08 = 100 s/update), 1h = 36 nominal updates ->
    # old prepare.py set total_updates = max(100, ceil(36 * 3) + 16) = 124, so session 4 stopped
    # early and session 5 exited immediately with 0 updates as completed_target_updates!
    prior_out: Path | None = None
    prev_updates = 0
    for sess_idx in range(1, 6):
        pkt_info = prepare_arena_ppo_packet(
            tmp_path / f"pkt_chain_{sess_idx:02d}",
            commit="HEAD",
            mode="train",
            gpu="L4",
            hours=1.0,
            expected_steps_per_sec=0.08,
            prior_output_dir=prior_out,
            prior_kind="checkpoint" if prior_out is not None else "auto",
            ppo_config=base_cfg if prior_out is None else None,
            output_dir=tmp_path / f"out_chain_{sess_idx:02d}",
            attempt=f"attempt-{sess_idx:02d}",
            allow_unmerged_commit=True,
        )
        mf = json.loads(Path(pkt_info["manifest_path"]).read_text(encoding="utf-8"))
        staged_in = _stage_manifest_inputs(mf, tmp_path / f"in_chain_{sess_idx:02d}")
        cfg_sess = ArenaPPOConfig.from_json(mf["env"]["ARENA_PPO_CONFIG_JSON"])
        max_wall = float(mf["env"]["ARENA_PPO_MAX_WALL_SECONDS"])

        res = run_training_job(
            cfg=cfg_sess,
            output_dir=Path(mf["output_dir"]),
            work_dir=tmp_path / f"work_chain_{sess_idx:02d}",
            device="cpu",
            input_dir=staged_in,
            ledger_path=staged_in / "verified-ledger.json",
            max_wall_seconds=max_wall,
            hours=1.0,
            expected_steps_per_sec=0.08,
            step_hook=lambda _tr, _u: fake_now.__setitem__(
                "t", fake_now["t"] + step_wall_advance["sec"]
            ),
        )
        assert res["stopped_reason"] == "wall_budget_guard", f"session {sess_idx}: {res}"
        delta_updates = int(res["updates_completed"]) - prev_updates
        assert delta_updates >= 25, f"session {sess_idx} ran only {delta_updates} updates"
        if sess_idx > 1:
            assert res["resumed_from"] is not None
        prev_updates = int(res["updates_completed"])
        prior_out = Path(mf["output_dir"])

    assert prev_updates >= 150  # > 124 old cap

    # Single session running 4x faster than expected_steps_per_sec (25 s/update vs 100 s est):
    # must run > 124 updates and stop ONLY on wall_budget_guard, never completed_target_updates!
    step_wall_advance["sec"] = 25.0
    fast_pkt = prepare_arena_ppo_packet(
        tmp_path / "pkt_fast_single",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=1.0,
        expected_steps_per_sec=0.08,
        ppo_config=base_cfg,
        output_dir=tmp_path / "out_fast_single",
        attempt="fast-01",
        allow_unmerged_commit=True,
    )
    mf_fast = json.loads(Path(fast_pkt["manifest_path"]).read_text(encoding="utf-8"))
    staged_fast = _stage_manifest_inputs(mf_fast, tmp_path / "in_fast_single")
    res_fast = run_training_job(
        cfg=ArenaPPOConfig.from_json(mf_fast["env"]["ARENA_PPO_CONFIG_JSON"]),
        output_dir=Path(mf_fast["output_dir"]),
        work_dir=tmp_path / "work_fast_single",
        device="cpu",
        input_dir=staged_fast,
        ledger_path=staged_fast / "verified-ledger.json",
        max_wall_seconds=float(mf_fast["env"]["ARENA_PPO_MAX_WALL_SECONDS"]),
        hours=1.0,
        expected_steps_per_sec=0.08,
        step_hook=lambda _tr, _u: fake_now.__setitem__(
            "t", fake_now["t"] + step_wall_advance["sec"]
        ),
    )
    assert res_fast["stopped_reason"] == "wall_budget_guard"
    assert res_fast["updates_completed"] > 124


def test_m3_prepare_scales_max_update_seconds_and_refuses_over_half_session_update(
    tmp_path: Path,
) -> None:
    """M3: prepare sets max_update_seconds = max(1200, 3x update_sec) and refuses >0.5*session."""
    # E=2048, T=256, S=6 -> 3,145,728 agent-steps/update.
    # At 2,000 steps/s, 1 update = 1,572.864 s (> old 1200 s hardcoded cap) on a 6 h (21,600 s) run.
    slow_info = prepare_arena_ppo_packet(
        tmp_path / "pkt_m3_slow_ok",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=6.0,
        expected_steps_per_sec=2000.0,
        net_arch="shared_trunk",
        net_size="S",
        num_envs=2048,
        output_dir=tmp_path / "out_m3_slow_ok",
        allow_unmerged_commit=True,
    )
    mf_slow = json.loads(Path(slow_info["manifest_path"]).read_text(encoding="utf-8"))
    cfg_slow = json.loads(mf_slow["env"]["ARENA_PPO_CONFIG_JSON"])
    expected_limit = round(3.0 * (2048 * 256 * 6) / 2000.0, 2)  # 4718.59 s
    assert cfg_slow["max_update_seconds"] == expected_limit
    assert cfg_slow["max_update_seconds"] > 1200.0

    # At 25,000 steps/s, 3x update_sec = 377.49 s -> 1200.0 s floor applies.
    fast_info = prepare_arena_ppo_packet(
        tmp_path / "pkt_m3_fast_floor",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=6.0,
        expected_steps_per_sec=25000.0,
        net_arch="shared_trunk",
        net_size="S",
        num_envs=2048,
        output_dir=tmp_path / "out_m3_fast_floor",
        allow_unmerged_commit=True,
    )
    mf_fast = json.loads(Path(fast_info["manifest_path"]).read_text(encoding="utf-8"))
    cfg_fast = json.loads(mf_fast["env"]["ARENA_PPO_CONFIG_JSON"])
    assert cfg_fast["max_update_seconds"] == 1200.0

    # Single update (3,145,728 / 1,500 = 2,097.15 s) > half of 1.0 h session (1,800.0 s) -> refused!
    with pytest.raises(ValueError, match="exceeds half"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_m3_refused",
            commit="HEAD",
            mode="train",
            gpu="L4",
            hours=1.0,
            expected_steps_per_sec=1500.0,
            net_arch="shared_trunk",
            net_size="S",
            num_envs=2048,
            output_dir=tmp_path / "out_m3_refused",
            allow_unmerged_commit=True,
        )


def test_minor3_continuation_canary_verifies_staged_checkpoint_without_rollout_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minor (3): run_canary_suite verifies staged ckpt without production rollout buffers."""
    from research.arena_ppo import ppo as ppo_mod

    prod_out = tmp_path / "prod_out"
    prod_cfg = ArenaPPOConfig(
        net_arch="shared_trunk",
        net_size="S",
        num_envs=10,
        rollout_len=12,
        ppo_epochs=1,
        minibatch_size=24,
        actor_chunk_envs=10,
        max_capacity=64,
        total_updates=1,
        seed=2026,
        deterministic=True,
    )
    run_training_job(
        cfg=prod_cfg,
        output_dir=prod_out,
        work_dir=tmp_path / "prod_work",
        device="cpu",
    )

    pkt_canary = prepare_arena_ppo_packet(
        tmp_path / "pkt_m3_cont_canary",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        prior_output_dir=prod_out,
        prior_kind="checkpoint",
        output_dir=tmp_path / "out_m3_cont_canary",
        allow_unmerged_commit=True,
    )
    mf_canary = json.loads(Path(pkt_canary["manifest_path"]).read_text(encoding="utf-8"))
    staged_inputs = _stage_manifest_inputs(mf_canary, tmp_path / "staged_m3_inputs")

    allocated_buffers: list[tuple[int, int]] = []
    orig_buf_init = ppo_mod.CompactRolloutBuffer.__init__

    def _spy_buf_init(self: Any, *args: Any, **kwargs: Any) -> None:
        allocated_buffers.append((int(kwargs["num_envs"]), int(kwargs["rollout_len"])))
        orig_buf_init(self, *args, **kwargs)

    monkeypatch.setattr(ppo_mod.CompactRolloutBuffer, "__init__", _spy_buf_init)

    suite = run_canary_suite(
        output_dir=tmp_path / "canary_exec_out",
        device="cpu",
        input_dir=staged_inputs,
        ledger_path=staged_inputs / "verified-ledger.json",
        num_envs=2,
        rollout_len=4,
        seed=99,
    )
    assert suite["staged_input_verification"] is not None
    assert suite["staged_input_verification"]["kind"] == "checkpoint"
    assert suite["staged_input_verification"]["updates_completed"] == 1
    # Production (num_envs=10, rollout_len=12) rollout buffer must NEVER be allocated!
    assert (10, 12) not in allocated_buffers
    assert all(ne <= 2 for ne, _ in allocated_buffers)

    # Verify cross-arch continuation packet preparation is refused immediately on Mac (Minor 5a)
    with pytest.raises(ValueError, match="Continuation checkpoint config mismatch on 'net_arch'"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_cross_arch_refused",
            commit="HEAD",
            mode="train",
            gpu="L4",
            hours=2.0,
            net_arch="ego_only",
            prior_output_dir=prod_out,
            prior_kind="checkpoint",
            output_dir=tmp_path / "out_cross_arch_refused",
            allow_unmerged_commit=True,
        )


def test_task_a_ruler_fp32_cells_comparisons_and_short_l4_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default bf16 plain cells + `_fp32` cells, paired comparisons, and short L4 prepare CLI."""
    from research.arena_ppo import prepare as prepare_mod

    out_dual = tmp_path / "ruler_dual_precision_out"
    res = run_ruler(
        output_dir=out_dual,
        device="cpu",
        envs_list=[2],
        net_sizes=["S"],
        net_archs=["ego_only", "shared_trunk"],
        burnin_ticks=4,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=4,
        ppo_epochs=2,
        minibatch_size=24,
        actor_chunk_envs=2,
        max_capacity=64,
        include_fp32=True,
        include_one_epoch_cell=False,
        include_best_compile=False,
        measure_larger_minibatch=False,
        seed=314,
    )
    assert verify_completed_artifact(out_dual / "ruler_result.json")
    assert res["stopped_reason"] == "completed"
    assert res["amp"] is True
    assert res["include_fp32"] is True
    cell_ids = [c["cell_id"] for c in res["cells"]]
    assert cell_ids == [
        "E2_ego_only_S",
        "E2_ego_only_S_fp32",
        "E2_shared_trunk_S",
        "E2_shared_trunk_S_fp32",
    ]
    for c in res["cells"]:
        assert verify_completed_artifact(out_dual / "cells" / f"{c['cell_id']}.json")
        assert c["amp"] is (not c["cell_id"].endswith("_fp32"))
        assert c["is_fp32_probe"] is c["cell_id"].endswith("_fp32")
        assert c["learner_rows_per_sec"] > 0.0
        assert c["integrated_agent_steps_per_sec"] > 0.0
        assert isinstance(c["epoch0_approx_kl"], float)
        assert isinstance(c["value_loss"], float)

    comps = res["precision_comparisons"]
    assert len(comps) == 2
    assert [(x["default_cell_id"], x["fp32_cell_id"]) for x in comps] == [
        ("E2_ego_only_S", "E2_ego_only_S_fp32"),
        ("E2_shared_trunk_S", "E2_shared_trunk_S_fp32"),
    ]
    for comp in comps:
        assert comp["amp_learner_rows_per_sec"] > 0.0
        assert comp["fp32_learner_rows_per_sec"] > 0.0
        assert comp["learner_speedup_ratio"] > 0.0
        assert comp["amp_integrated_agent_steps_per_sec"] > 0.0
        assert comp["fp32_integrated_agent_steps_per_sec"] > 0.0
        assert comp["integrated_speedup_ratio"] > 0.0
        assert isinstance(comp["amp_epoch0_approx_kl"], float)
        assert isinstance(comp["fp32_epoch0_approx_kl"], float)
        assert isinstance(comp["amp_value_loss"], float)
        assert isinstance(comp["fp32_value_loss"], float)

    # Short E=2048 default (`bf16`) vs `_fp32` ruler packet via prepare.py CLI (< 25 min)
    canary_pkt = prepare_arena_ppo_packet(
        tmp_path / "pkt_canary_for_short_ruler",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_canary_for_short_ruler",
        allow_unmerged_commit=True,
    )
    canary_mf = json.loads(Path(canary_pkt["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_short = _build_synthetic_canary_receipt(canary_mf, "L4")

    pkt_ruler_dir = tmp_path / "pkt_short_ruler_fp32"
    out_ruler_dir = tmp_path / "out_short_ruler_fp32"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research.arena_ppo.prepare",
            "--packet",
            str(pkt_ruler_dir),
            "--commit",
            "HEAD",
            "--mode",
            "ruler",
            "--gpu",
            "L4",
            "--net-arch",
            "ego_only,shared_trunk",
            "--ruler-envs",
            "2048",
            "--ruler-nets",
            "S",
            "--include-fp32",
            "--no-one-epoch-cell",
            "--no-best-compile",
            "--no-larger-minibatch",
            "--ruler-base-minutes",
            "12",
            "--canary-receipt",
            str(rcpt_short),
            "--output-dir",
            str(out_ruler_dir),
            "--allow-unmerged-commit",
        ],
    )
    assert prepare_mod.main() == 0
    mf_path = pkt_ruler_dir / "dispatch-manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    colab_canary.check(mf, ROOT)
    colab_schema.validate_manifest(mf)
    ops_dispatch.require_resume_policy(mf)
    colab_supervisor.validate(mf)
    assert mf["timeout_minutes"] == 24.0
    env = mf["env"]
    assert env["ARENA_PPO_RULER_ARCHS"] == "ego_only,shared_trunk"
    assert env["ARENA_PPO_RULER_ENVS"] == "2048"
    assert env["ARENA_PPO_RULER_NETS"] == "S"
    assert env["ARENA_PPO_AMP"] == "1"
    assert env["ARENA_PPO_INCLUDE_FP32"] == "1"
    assert env["ARENA_PPO_NO_ONE_EPOCH_CELL"] == "1"
    assert env["ARENA_PPO_NO_BEST_COMPILE"] == "1"
    assert env["ARENA_PPO_NO_LARGER_MINIBATCH"] == "1"

    # Train packet defaults to amp=True (ARENA_PPO_AMP=1), and --no-amp turns it off (0)
    pkt_train_def = prepare_arena_ppo_packet(
        tmp_path / "pkt_train_default_amp",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        canary_receipt=rcpt_short,
        output_dir=tmp_path / "out_train_default_amp",
        allow_unmerged_commit=True,
    )
    mf_train_def = json.loads(Path(pkt_train_def["manifest_path"]).read_text(encoding="utf-8"))
    colab_supervisor.validate(mf_train_def)
    assert mf_train_def["env"]["ARENA_PPO_AMP"] == "1"
    cfg_def = json.loads(mf_train_def["env"]["ARENA_PPO_CONFIG_JSON"])
    assert cfg_def["amp"] is True and cfg_def["use_bf16_autocast"] is True

    pkt_train_fp32 = prepare_arena_ppo_packet(
        tmp_path / "pkt_train_fp32",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        amp=False,
        canary_receipt=rcpt_short,
        output_dir=tmp_path / "out_train_fp32",
        allow_unmerged_commit=True,
    )
    mf_train_fp32 = json.loads(Path(pkt_train_fp32["manifest_path"]).read_text(encoding="utf-8"))
    colab_supervisor.validate(mf_train_fp32)
    assert mf_train_fp32["env"]["ARENA_PPO_AMP"] == "0"
    cfg_fp32 = json.loads(mf_train_fp32["env"]["ARENA_PPO_CONFIG_JSON"])
    assert cfg_fp32["amp"] is False and cfg_fp32["use_bf16_autocast"] is False


def test_task_b1_and_a3_canary_3arch_train_late_reset_resume_and_cuda_fp32_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3-design canary train w/ time-limit + reservoir + resume + dual-precision CUDA probe."""
    canary_out = tmp_path / "canary_3arch_out"
    suite = run_canary_suite(
        output_dir=canary_out,
        device="cpu",
        num_envs=4,
        rollout_len=4,
        seed=2026,
    )
    assert verify_completed_artifact(canary_out / "canary_checks.json")
    checks = json.loads((canary_out / "canary_checks.json").read_text(encoding="utf-8"))
    assert checks["fp32_canary_probe"]["executed"] is False
    assert "CUDA" in checks["fp32_canary_probe"]["skipped_reason"]
    assert suite["fp32_canary_probe"] == checks["fp32_canary_probe"]

    assert set(suite["train_by_arch"].keys()) == {"ego_only", "per_seat", "shared_trunk"}
    for arch in ("ego_only", "per_seat", "shared_trunk"):
        tr = suite["train_by_arch"][arch]
        assert tr["updates_completed"] == 2
        assert tr["resumed_from_checkpoint"] is True
        assert tr["config"]["amp"] is True
        assert tr["config"]["use_bf16_autocast"] is True
        assert tr["config"]["max_world_frames"] == 4
        assert tr["config"]["late_reset_enabled"] is True
        assert tr["config"]["late_reset_min_length"] == 1
        assert tr["total_time_limit_resets"] >= 1, f"{arch} did not exercise time-limit reset"
        assert tr["total_harvested"] >= 1, f"{arch} did not harvest into late-state reservoir"
        assert tr["total_restored"] >= 1, f"{arch} did not restore from late-state reservoir"

    # Enable CUDA `_fp32` canary probe gate to verify both default and `_fp32` cells execute
    assert ruler_mod._canary_runs_fp32_probe(torch.device("cpu")) is False
    assert ruler_mod._canary_runs_fp32_probe(torch.device("cuda", 0)) is True
    monkeypatch.setattr(ruler_mod, "_canary_runs_fp32_probe", lambda _dev: True)

    canary_cuda_out = tmp_path / "canary_cuda_sim_out"
    suite_cuda = run_canary_suite(
        output_dir=canary_cuda_out,
        device="cpu",
        num_envs=4,
        rollout_len=4,
        seed=2026,
    )
    assert suite_cuda["fp32_canary_probe"]["executed"] is True
    assert suite_cuda["fp32_canary_probe"]["fp32_cell_ids"] == [
        "E4_ego_only_S_fp32",
        "E4_shared_trunk_S_fp32",
        "E4_per_seat_S_fp32",
    ]
    assert suite_cuda["ruler_result"]["succeeded_cell_ids"] == [
        "E4_ego_only_S",
        "E4_ego_only_S_fp32",
        "E4_shared_trunk_S",
        "E4_shared_trunk_S_fp32",
        "E4_per_seat_S",
        "E4_per_seat_S_fp32",
    ]

    # Verify a failure specifically inside a secondary `_fp32` cell aborts the CUDA canary loudly
    orig_integ = ruler_mod._measure_integrated_cell

    def _fail_fp32_cell(**kw: Any) -> dict[str, Any]:
        if (not kw.get("amp")) and kw.get("net_arch") == "shared_trunk":
            raise RuntimeError("synthetic failure in shared_trunk _fp32")
        return orig_integ(**kw)

    monkeypatch.setattr(ruler_mod, "_measure_integrated_cell", _fail_fp32_cell)
    with pytest.raises(RuntimeError, match="E4_shared_trunk_S_fp32"):
        run_canary_suite(
            output_dir=tmp_path / "canary_cuda_fp32_fail",
            device="cpu",
            num_envs=4,
            rollout_len=4,
            seed=2026,
        )


def test_task_b4_fatal_non_oom_cuda_error_stops_grid_and_cleanup_fault_preserves_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task B.4: non-OOM CUDA error stops grid, marks skipped_cells, and survives cleanup fault."""
    orig_coll = ruler_mod._measure_collection_only_cell
    orig_cleanup = ruler_mod._cleanup_device_memory
    cuda_poisoned = {"active": False}

    def _poison_on_second_cell(**kw: Any) -> dict[str, Any]:
        if kw.get("num_envs") == 2 and kw.get("net_arch") == "shared_trunk":
            cuda_poisoned["active"] = True
            raise RuntimeError("CUDA error: device-side assert triggered")
        if kw.get("num_envs") == 4:
            raise AssertionError("Grid failed to stop after non-OOM CUDA device-side assert!")
        return orig_coll(**kw)

    def _poisoned_cleanup(dev: torch.device) -> None:
        if cuda_poisoned["active"]:
            raise RuntimeError("CUDA error: device-side assert triggered inside empty_cache")
        orig_cleanup(dev)

    monkeypatch.setattr(ruler_mod, "_measure_collection_only_cell", _poison_on_second_cell)
    monkeypatch.setattr(ruler_mod, "_cleanup_device_memory", _poisoned_cleanup)

    out_dir = tmp_path / "ruler_cuda_assert_stop"
    res = run_ruler(
        output_dir=out_dir,
        device="cpu",
        envs_list=[2, 4],
        net_sizes=["S"],
        net_archs=["ego_only", "shared_trunk"],
        burnin_ticks=2,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=2,
        ppo_epochs=1,
        minibatch_size=12,
        actor_chunk_envs=2,
        max_capacity=64,
        include_one_epoch_cell=False,
        include_best_compile=False,
        measure_larger_minibatch=False,
        seed=77,
    )

    # Current failing cell record MUST be persisted intact despite poisoned cleanup raising!
    assert verify_completed_artifact(out_dir / "cells" / "E2_ego_only_S.json")
    assert verify_completed_artifact(out_dir / "cells" / "E2_shared_trunk_S.json")
    assert verify_completed_artifact(out_dir / "ruler_result.json")

    poisoned_cell = json.loads(
        (out_dir / "cells" / "E2_shared_trunk_S.json").read_text(encoding="utf-8")
    )
    assert poisoned_cell["fatal_cuda_error"] is True
    assert "device-side assert" in poisoned_cell["error"]
    assert "empty_cache" in poisoned_cell["cleanup_error"]

    assert res["partial"] is True
    assert res["stopped_reason"] == "cuda_error_stop"
    assert [c["cell_id"] for c in res["cells"]] == ["E2_ego_only_S", "E2_shared_trunk_S"]
    assert res["succeeded_cell_ids"] == ["E2_ego_only_S"]
    assert res["failed_cell_ids"] == ["E2_shared_trunk_S"]

    skipped = res["skipped_cells"]
    assert [s["cell_id"] for s in skipped] == ["E4_ego_only_S", "E4_shared_trunk_S"]
    for s in skipped:
        assert s["reason"] == "stopped_after_cuda_error"
        assert s["trigger_cell_id"] == "E2_shared_trunk_S"
        assert "device-side assert" in s["trigger_error"]


def test_canary_disables_projection_skip_under_huge_projected_cell_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canary tiny ruler must run all 3 designs even when projection exceeds max_wall_seconds."""
    proj_calls: list[tuple[str, float]] = []

    def _huge_projection(**kwargs: Any) -> float:
        arch = str(kwargs.get("net_arch", "per_seat"))
        est = 0.0 if arch == "ego_only" else 10_000.0
        proj_calls.append((arch, est))
        return est

    monkeypatch.setattr(ruler_mod, "_project_cell_seconds", _huge_projection)

    # Default run_ruler (disable_projection_skip=False) still skips on huge projection.
    default_dir = tmp_path / "ruler_default_skips"
    res_default = run_ruler(
        output_dir=default_dir,
        device="cpu",
        envs_list=[2],
        net_sizes=["S"],
        net_archs=["ego_only", "shared_trunk", "per_seat"],
        burnin_ticks=2,
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=2,
        ppo_epochs=1,
        include_one_epoch_cell=False,
        include_best_compile=False,
        measure_larger_minibatch=False,
        max_capacity=64,
        max_wall_seconds=120.0,
    )
    assert res_default["partial"] is True
    assert res_default["stopped_reason"] == "wall_projection_guard"
    assert res_default["succeeded_cell_ids"] == ["E2_ego_only_S"]
    assert [sc["cell_id"] for sc in res_default["skipped_cells"]] == [
        "E2_shared_trunk_S",
        "E2_per_seat_S",
    ]
    assert all(sc["reason"] == "projected_wall_budget_skip" for sc in res_default["skipped_cells"])

    # Canary path sets disable_projection_skip=True and must still run all three designs.
    proj_calls.clear()
    canary_dir = tmp_path / "canary_ignores_projection"
    suite = run_canary_suite(
        output_dir=canary_dir,
        device="cpu",
        max_wall_seconds=540.0,
        num_envs=2,
        rollout_len=4,
        seed=20261111,
    )
    rr = suite["ruler_result"]
    assert [arch for arch, est in proj_calls if est >= 10_000.0] == [
        "shared_trunk",
        "per_seat",
    ]
    assert rr["partial"] is False
    assert rr["stopped_reason"] == "completed"
    assert rr["skipped_cells"] == []
    assert rr["succeeded_cell_ids"] == [
        "E2_ego_only_S",
        "E2_shared_trunk_S",
        "E2_per_seat_S",
    ]
    for rel in CANARY_REQUIRED_EXPORTS:
        assert verify_completed_artifact(canary_dir / rel)


def test_canary_error_names_skipped_cells_reasons_and_missing_designs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canary failure message must name skipped cells + reasons and genuinely missing designs."""
    orig_monotonic = ruler_mod.time.monotonic
    orig_integ = ruler_mod._measure_integrated_cell
    clock_offset = {"sec": 0.0}

    monkeypatch.setattr(ruler_mod.time, "monotonic", lambda: orig_monotonic() + clock_offset["sec"])

    def _slow_second_cell(**kwargs: Any) -> dict[str, Any]:
        out = orig_integ(**kwargs)
        if kwargs.get("net_arch") == "shared_trunk":
            clock_offset["sec"] += 400.0
        return out

    monkeypatch.setattr(ruler_mod, "_measure_integrated_cell", _slow_second_cell)

    # 1. Hard elapsed-time stop inside canary ruler skips per_seat -> error names cell, reason, arch
    with pytest.raises(RuntimeError) as exc_skip:
        run_canary_suite(
            output_dir=tmp_path / "canary_hard_wall_skip",
            device="cpu",
            max_wall_seconds=540.0,
            num_envs=2,
            rollout_len=4,
            seed=7,
        )
    msg_skip = str(exc_skip.value)
    assert "Canary multi-arch ruler probe failed:" in msg_skip
    assert "missing designs: per_seat" in msg_skip
    assert "E2_per_seat_S (elapsed_wall_budget_skip)" in msg_skip

    # 2. Genuinely missing design (no failed/skipped cells) -> error names the missing design
    clock_offset["sec"] = 0.0
    monkeypatch.setattr(ruler_mod, "_measure_integrated_cell", orig_integ)
    orig_run_ruler = ruler_mod.run_ruler

    def _drop_per_seat(**kwargs: Any) -> dict[str, Any]:
        kwargs["net_archs"] = ["ego_only", "shared_trunk"]
        return orig_run_ruler(**kwargs)

    monkeypatch.setattr(ruler_mod, "run_ruler", _drop_per_seat)

    with pytest.raises(RuntimeError) as exc_missing:
        run_canary_suite(
            output_dir=tmp_path / "canary_missing_design",
            device="cpu",
            max_wall_seconds=540.0,
            num_envs=2,
            rollout_len=4,
            seed=7,
        )
    msg_missing = str(exc_missing.value)
    assert "Canary multi-arch ruler probe failed:" in msg_missing
    assert "missing designs: per_seat" in msg_missing
    assert "skipped cells: none" in msg_missing


def test_continuation_canary_completed_exports_never_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: continuation canary must never mutate sha256 or stat of completed exports."""
    prior_train_out = tmp_path / "prior_ego_s_train"
    prior_cfg = ArenaPPOConfig(
        net_arch="ego_only",
        net_size="S",
        num_envs=4,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=24,
        actor_chunk_envs=4,
        max_capacity=64,
        total_updates=2,
        seed=20261111,
        deterministic=True,
    )
    run_training_job(
        cfg=prior_cfg,
        output_dir=prior_train_out,
        work_dir=tmp_path / "prior_ego_s_work",
        device="cpu",
    )
    assert verify_completed_artifact(prior_train_out / "checkpoints" / "final.pt")

    cont_pkt = prepare_arena_ppo_packet(
        tmp_path / "pkt_cont_canary_a100",
        commit="HEAD",
        mode="canary",
        gpu="A100",
        prior_output_dir=prior_train_out,
        prior_kind="checkpoint",
        output_dir=tmp_path / "host_cont_canary_out",
        attempt="attempt-02",
        allow_unmerged_commit=True,
    )
    mf = json.loads(Path(cont_pkt["manifest_path"]).read_text(encoding="utf-8"))
    max_export_bytes = int(mf["max_export_bytes"])

    cli_root = tmp_path / "remote_cli_root"
    remote_in = _stage_manifest_inputs(mf, cli_root / "inputs")
    remote_out = cli_root / "output"
    remote_out_resolved = remote_out.resolve()

    initial_sha256: dict[str, str] = {}
    supervisor_cache: dict[str, Any] = {}
    orig_fsync_dir = runtime_mod._fsync_directory

    def _check_completed_exports() -> None:
        if not remote_out.is_dir():
            return
        seen_now: set[str] = set()
        for marker in sorted(remote_out.rglob("*.complete")):
            candidate = marker.with_name(marker.name[: -len(".complete")])
            rel = candidate.relative_to(remote_out).as_posix()
            seen_now.add(rel)
            observed_sha = sha256_file(candidate)
            if rel not in initial_sha256:
                initial_sha256[rel] = observed_sha
            else:
                assert (
                    observed_sha == initial_sha256[rel]
                ), f"Completed export {rel} sha256 changed after .complete"
        assert set(initial_sha256).issubset(
            seen_now
        ), f"Completed export removed after .complete: {set(initial_sha256) - seen_now}"
        supervisor_scan_exports(cli_root, max_export_bytes, supervisor_cache)

    def _hooked_fsync_dir(directory: Path) -> None:
        orig_fsync_dir(directory)
        resolved = directory.resolve()
        if resolved == remote_out_resolved or remote_out_resolved in resolved.parents:
            _check_completed_exports()

    monkeypatch.setattr(runtime_mod, "_fsync_directory", _hooked_fsync_dir)

    suite = run_canary_suite(
        output_dir=remote_out,
        device="cpu",
        input_dir=remote_in,
        ledger_path=remote_in / "verified-ledger.json",
        num_envs=2,
        rollout_len=4,
        seed=20261111,
    )
    _check_completed_exports()

    assert suite["staged_input_verification"]["kind"] == "checkpoint"
    assert suite["staged_input_verification"]["updates_completed"] == 2
    assert len(initial_sha256) >= len(CANARY_REQUIRED_EXPORTS)
    for rel in CANARY_REQUIRED_EXPORTS:
        assert rel in initial_sha256
        assert verify_completed_artifact(remote_out / rel)
    assert list((remote_out / "checkpoints").glob("ckpt_u*.pt")) == []
