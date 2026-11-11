"""Tests for `research.arena_ppo.ruler`, `run_canary_suite`, and 3-mode `prepare.py`."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from research.arena_ppo import ruler as ruler_mod
from research.arena_ppo.ppo import ArenaPPOConfig, ArenaPPOTrainer, run_training_job
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
    assert verify_completed_artifact(canary_dir / "cells" / "E4_S.json")
    assert verify_completed_artifact(canary_dir / "checkpoints" / "ckpt_u0000001.pt")

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
        assert mf["timeout_minutes"] == 12.0
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
