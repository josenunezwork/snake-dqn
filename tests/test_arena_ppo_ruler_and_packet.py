"""Tests for `research.arena_ppo.ruler` (MPS end-to-end, wall guard, OOM) and `prepare.py`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from research.arena_ppo import ruler as ruler_mod
from research.arena_ppo.ppo import ArenaPPOConfig, ArenaPPOTrainer, run_training_job
from research.arena_ppo.prepare import (
    ROOT,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
    prepare_arena_ppo_packet,
)
from research.arena_ppo.ruler import run_ruler
from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.colab_cli import canary as colab_canary
from research.colab_cli import schema as colab_schema
from research.colab_cli import supervisor as colab_supervisor
from tools.ops import dispatch as ops_dispatch


def test_ruler_end_to_end_on_mps_and_canary_probes(tmp_path: Path) -> None:
    """Run real ruler on MPS (or CPU fallback if headless) and verify exports + probes."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = tmp_path / "ruler_mps_out"
    res = run_ruler(
        output_dir=out_dir,
        device=dev,
        envs_list=[2, 4],
        net_sizes=["S", "M"],
        warmup_ticks=1,
        collect_ticks=3,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        actor_chunk_envs=4,
        max_capacity=64,
        write_canary_probes=True,
        seed=123,
    )

    assert verify_completed_artifact(out_dir / "ruler_result.json")
    for rel in TRAIN_REQUIRED_EXPORTS:
        assert verify_completed_artifact(out_dir / rel)

    assert len(res["cells"]) == 4
    assert res["skipped_cells"] == []
    for cell in res["cells"]:
        assert cell["oom"] is False
        assert cell["collection_only"]["agent_steps_per_sec"] > 0.0
        assert cell["integrated"]["agent_steps_per_sec"] > 0.0
        for stage in ("sim_step", "obs_build", "policy_forward", "learner"):
            assert stage in cell["integrated"]["stage_seconds"]
            assert cell["integrated"]["stage_seconds"][stage] >= 0.0

    # Canary probe checkpoint must be refused by trainer resume and prepare staging
    cfg = ArenaPPOConfig(num_envs=2, rollout_len=4, max_capacity=64)
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    with pytest.raises(ValueError, match="canary_export_probe"):
        trainer.load_checkpoint(out_dir / "checkpoints" / "final.pt")


def test_ruler_wall_budget_skips_and_oom_catch_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify wall budget skips remaining cells and OOM in one cell is caught & recorded."""
    wall_dir = tmp_path / "ruler_wall"
    res_wall = run_ruler(
        output_dir=wall_dir,
        device="cpu",
        envs_list=[2, 4],
        net_sizes=["S", "M"],
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=64,
        max_wall_seconds=0.01,
        write_canary_probes=False,
    )
    assert verify_completed_artifact(wall_dir / "ruler_result.json")
    assert len(res_wall["skipped_cells"]) == 4
    assert len(res_wall["cells"]) == 0

    # Test OOM catch-record-continue on E=4 while E=2 succeeds
    orig_coll = ruler_mod._measure_collection_only_cell

    def _fake_coll(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["num_envs"] == 4:
            raise torch.OutOfMemoryError("CUDA out of memory: simulated test OOM")
        return orig_coll(**kwargs)

    monkeypatch.setattr(ruler_mod, "_measure_collection_only_cell", _fake_coll)
    oom_dir = tmp_path / "ruler_oom"
    res_oom = run_ruler(
        output_dir=oom_dir,
        device="cpu",
        envs_list=[2, 4],
        net_sizes=["S"],
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=64,
        write_canary_probes=False,
    )
    assert verify_completed_artifact(oom_dir / "ruler_result.json")
    assert res_oom["cells"][0]["oom"] is False
    assert res_oom["cells"][1]["oom"] is True
    assert res_oom["cells"][1]["collection_only"]["oom"] is True

    # Completed ruler_result.json must never be overwritten
    with pytest.raises(FileExistsError):
        run_ruler(
            output_dir=oom_dir,
            device="cpu",
            envs_list=[2],
            net_sizes=["S"],
            max_wall_seconds=0.01,
            write_canary_probes=False,
        )


def test_prepare_manifests_supervisor_validate_and_prior_resume(tmp_path: Path) -> None:
    """Validate ruler and train manifests against Colab schema, resume policy, and supervisor."""
    for exports in (RULER_REQUIRED_EXPORTS, TRAIN_REQUIRED_EXPORTS):
        for rel in exports:
            assert not rel.endswith(".sha256")
            assert not rel.endswith(".complete")

    # Unmerged HEAD must be refused when allow_unmerged_commit=False (unless HEAD <= origin/main)
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
                mode="ruler",
                gpu="L4",
                output_dir=tmp_path / "refused_out",
                allow_unmerged_commit=False,
            )

    # Prepare ruler (L4 and A100) and train (L4) packets and validate via supervisor.validate
    ruler_l4 = prepare_arena_ppo_packet(
        tmp_path / "pkt_ruler_l4",
        commit="HEAD",
        mode="ruler",
        gpu="L4",
        output_dir=tmp_path / "out_ruler_l4",
        allow_unmerged_commit=True,
    )
    ruler_a100 = prepare_arena_ppo_packet(
        tmp_path / "pkt_ruler_a100",
        commit="HEAD",
        mode="ruler",
        gpu="A100",
        output_dir=tmp_path / "out_ruler_a100",
        allow_unmerged_commit=True,
    )
    train_l4 = prepare_arena_ppo_packet(
        tmp_path / "pkt_train_l4",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=3.0,
        output_dir=tmp_path / "out_train_l4",
        allow_unmerged_commit=True,
    )

    mf_ruler_l4 = json.loads(Path(ruler_l4["manifest_path"]).read_text(encoding="utf-8"))
    mf_ruler_a100 = json.loads(Path(ruler_a100["manifest_path"]).read_text(encoding="utf-8"))
    mf_train_l4 = json.loads(Path(train_l4["manifest_path"]).read_text(encoding="utf-8"))

    assert mf_ruler_l4["timeout_minutes"] <= 28.0
    assert mf_ruler_a100["requirements"]["gpu"] == "A100"
    assert mf_train_l4["timeout_minutes"] == 180.0

    # Family canary invariants: identical script, source_paths, and initial input SHA-256
    assert mf_ruler_l4["script"] == mf_train_l4["script"] == "research/arena_ppo/job.sh"
    assert mf_ruler_l4["source_paths"] == mf_train_l4["source_paths"]
    assert len(mf_ruler_l4["inputs"]) == len(mf_train_l4["inputs"]) == 1
    assert mf_ruler_l4["inputs"][0]["dest"] == mf_train_l4["inputs"][0]["dest"]
    assert mf_ruler_l4["inputs"][0]["sha256"] == mf_train_l4["inputs"][0]["sha256"]

    for mf in (mf_ruler_l4, mf_ruler_a100):
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)

    # Without a canary receipt yet, `train` passes schema + resume policy and reaches canary gate
    colab_schema.validate_manifest(mf_train_l4)
    ops_dispatch.require_resume_policy(mf_train_l4)
    with pytest.raises(colab_supervisor.Refused, match="SHA-bound family canary receipt"):
        colab_supervisor.validate(mf_train_l4)

    # Build a complete synthetic family canary receipt using real `run_ruler` probe exports
    canary_out = Path(mf_ruler_l4["output_dir"])
    canary_exports_dir = canary_out / "exports"
    run_ruler(
        output_dir=canary_exports_dir,
        device="cpu",
        envs_list=[2],
        net_sizes=["S"],
        warmup_ticks=1,
        collect_ticks=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=64,
        write_canary_probes=True,
    )
    saved_probe_mf = canary_out / "manifest.json"
    saved_probe_mf.write_text(json.dumps(mf_ruler_l4, indent=2, sort_keys=True) + "\n")
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
    (canary_out / "hardware.json").write_text(json.dumps({"hardware": {"kind": "L4"}}))
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
    inp0 = mf_ruler_l4["inputs"][0]
    receipt_obj = {
        "status": "PASS",
        "stop_verified": True,
        "error": None,
        "cleanup_error": None,
        "failure_class": None,
        "commit": mf_ruler_l4["commit"],
        "job_id": mf_ruler_l4["job_id"],
        "generation": "gen-test-01",
        "manifest_sha256": sha256_file(saved_probe_mf),
        "exports": export_inventory,
        "transfers": [
            {
                "direction": "upload",
                "remote": f"/content/snake-cli/gen-test-01/inputs/{inp0['dest']}",
                "status": "verified",
                "size_bytes": Path(inp0["path"]).stat().st_size,
                "sha256": inp0["sha256"],
            }
        ],
        "terminal": {
            "job_id": mf_ruler_l4["job_id"],
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
    receipt_obj["canary_evidence"] = colab_canary.evidence(
        mf_ruler_l4, receipt_obj, canary_out, ROOT
    )
    receipt_path = canary_out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt_obj, indent=2, sort_keys=True) + "\n")

    # Now `train` bound to the `ruler` canary receipt passes full `supervisor.validate`!
    train_l4_bound = prepare_arena_ppo_packet(
        tmp_path / "pkt_train_l4_bound",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=3.0,
        canary_receipt=receipt_path,
        output_dir=tmp_path / "out_train_l4_bound",
        allow_unmerged_commit=True,
    )
    mf_train_bound = json.loads(Path(train_l4_bound["manifest_path"]).read_text(encoding="utf-8"))
    colab_schema.validate_manifest(mf_train_bound)
    ops_dispatch.require_resume_policy(mf_train_bound)
    colab_supervisor.validate(mf_train_bound)

    # Test `--prior-output-dir` staging of a real training checkpoint (and refusal of canary probes)
    with pytest.raises(FileExistsError):
        # Re-using existing packet_dir is refused
        prepare_arena_ppo_packet(
            tmp_path / "pkt_ruler_l4",
            commit="HEAD",
            mode="ruler",
            gpu="L4",
            output_dir=tmp_path / "out_dup",
            allow_unmerged_commit=True,
        )
    with pytest.raises(FileNotFoundError, match="No verified non-probe"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_refuse_probe_resume",
            commit="HEAD",
            mode="train",
            gpu="L4",
            prior_output_dir=canary_out,
            output_dir=tmp_path / "out_refuse_probe_resume",
            allow_unmerged_commit=True,
        )

    prior_train_out = tmp_path / "prior_real_train"
    run_training_job(
        cfg=ArenaPPOConfig(
            num_envs=2,
            rollout_len=4,
            ppo_epochs=1,
            minibatch_size=12,
            max_capacity=64,
            total_updates=1,
        ),
        output_dir=prior_train_out,
        device="cpu",
    )
    train_resume = prepare_arena_ppo_packet(
        tmp_path / "pkt_train_resume",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        prior_output_dir=prior_train_out,
        output_dir=tmp_path / "out_train_resume",
        attempt="attempt-02",
        allow_unmerged_commit=True,
    )
    ruler_resume = prepare_arena_ppo_packet(
        tmp_path / "pkt_ruler_resume",
        commit="HEAD",
        mode="ruler",
        gpu="L4",
        prior_output_dir=prior_train_out,
        output_dir=tmp_path / "out_ruler_resume",
        attempt="attempt-02",
        allow_unmerged_commit=True,
    )
    mf_resume = json.loads(Path(train_resume["manifest_path"]).read_text(encoding="utf-8"))
    mf_ruler_resume = json.loads(Path(ruler_resume["manifest_path"]).read_text(encoding="utf-8"))
    colab_schema.validate_manifest(mf_resume)
    ops_dispatch.require_resume_policy(mf_resume)
    colab_supervisor.validate(mf_ruler_resume)
    assert len(mf_resume["inputs"]) == 2
    assert [r["sha256"] for r in mf_resume["inputs"]] == [
        r["sha256"] for r in mf_ruler_resume["inputs"]
    ]
