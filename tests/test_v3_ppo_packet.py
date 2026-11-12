"""Preflight tests for `research.v3_ppo.prepare`, `ruler`, `job.sh`, VRAM & export-cap refusals."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import research.v3_ppo.ruler as ruler_mod
from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.colab_cli import canary as colab_canary
from research.colab_cli import schema as colab_schema
from research.colab_cli import source as colab_source
from research.colab_cli import supervisor as colab_supervisor
from research.game_v3.config import GameV3Config
from research.v3_ppo.config import (
    ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_BYTES_PER_SEC,
    ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS,
    ASSUMED_STAGED_INPUT_UPLOAD_RATE_BYTES_PER_SEC,
    ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS,
    MAX_TRAIN_HEADROOM_SESSION_FRACTION,
    NONE_MODE_MAX_TOTAL_EXPORT_BYTES,
    TRAIN_FINAL_EVAL_AND_SAVE_ALLOWANCE_SECONDS,
    TRAIN_FINAL_SAVE_AFTER_EVAL_ALLOWANCE_SECONDS,
    TRAIN_FINAL_TELEMETRY_EXPORT_BYTES,
    TRAIN_SETUP_ALLOWANCE_SECONDS,
    TRAINER_CHECKPOINT_RESERVE_SECONDS,
    V3PPOConfig,
    estimate_checkpoint_export_caps,
    estimate_train_wall_headroom,
    estimate_train_wall_headroom_by_mode,
    estimate_training_memory,
)
from research.v3_ppo.job import build_arg_parser as build_train_arg_parser
from research.v3_ppo.job import resolve_effective_train_config
from research.v3_ppo.ppo import (
    V3PPOTrainer,
)
from research.v3_ppo.ppo import main as ppo_main
from research.v3_ppo.ppo import (
    publish_checkpoint_to_output,
    run_training_job,
    verify_full_checkpoint_set,
)
from research.v3_ppo.prepare import (
    ROOT,
)
from research.v3_ppo.prepare import main as prepare_main
from research.v3_ppo.prepare import (
    prepare_v3_ppo_packet,
    runtime_source_paths,
)
from research.v3_ppo.ruler import (
    CANARY_REQUIRED_EXPORTS,
    DEFAULT_RULER_ROLLOUT_LEN,
    DEFAULT_RULER_TIMED_UPDATES,
    DEFAULT_RULER_WARMUP_UPDATES,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
)
from research.v3_ppo.ruler import build_arg_parser as build_ruler_arg_parser
from research.v3_ppo.ruler import main as ruler_main
from research.v3_ppo.ruler import (
    resolve_effective_ruler_config,
    run_canary_suite,
    run_ruler,
)
from tools.ops import dispatch as ops_dispatch


def _stage_manifest_inputs(mf: dict[str, Any], remote_inputs_dir: Path) -> Path:
    remote_inputs_dir.mkdir(parents=True, exist_ok=True)
    for inp in mf["inputs"]:
        dst = remote_inputs_dir / str(inp["dest"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inp["path"], dst)
    return remote_inputs_dir


def _build_synthetic_canary_receipt(
    canary_mf: dict[str, Any],
    gpu_kind: str,
    *,
    shared_exports_src: Path | None = None,
) -> Path:
    """Create a broker-verified canary receipt from real `run_canary_suite` exports."""
    canary_out = Path(canary_mf["output_dir"])
    canary_exports_dir = canary_out / "exports"
    remote_in = _stage_manifest_inputs(canary_mf, canary_out / "remote_inputs")

    if shared_exports_src is not None:
        shutil.copytree(shared_exports_src, canary_exports_dir)
    else:
        fast_eval_override = V3PPOConfig(
            game_config=GameV3Config(arena_radius=60.0, num_snakes=6),
            eval_envs=4,
            eval_burnin_ticks=4,
            eval_ticks=12,
            eval_learner_slots=2,
        )
        canary_res = run_canary_suite(
            output_dir=canary_exports_dir,
            device="cpu",
            input_dir=remote_in,
            ledger_path=remote_in / "verified-ledger.json",
            canary_envs=2,
            canary_arena_radius=60.0,
            canary_num_snakes=6,
            canary_rollout_len=4,
            canary_eval_cfg_override=fast_eval_override,
        )
        assert canary_res["default_eval_benchmark"]["aborted"] is False
        assert canary_res["default_eval_benchmark"]["wall_seconds"] > 0.0
        assert canary_res["default_eval_wall_seconds"] > 0.0

    saved_probe_mf = canary_out / "manifest.json"
    saved_probe_mf.write_text(
        json.dumps(canary_mf, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (canary_out / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 2.0,
                "transfer_seconds": 0.5,
                "source_delivery_seconds": 0.8,
                "staging_completed": True,
            }
        ),
        encoding="utf-8",
    )
    (canary_out / "hardware.json").write_text(
        json.dumps({"hardware": {"kind": gpu_kind}}), encoding="utf-8"
    )

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
    colab_schema.validate_export_inventory(canary_mf, export_inventory)

    transfers = [
        {
            "direction": "upload",
            "remote": f"/content/snake-cli/gen-test-01/inputs/{inp['dest']}",
            "status": "verified",
            "size_bytes": Path(inp["path"]).stat().st_size,
            "sha256": inp["sha256"],
        }
        for inp in canary_mf["inputs"]
    ]
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
                    "package": "research/v3_ppo",
                    "caller": "research/v3_ppo/ruler.py:_verify_cardinal6_canary_decisions",
                    "operation": "Tensor.argmax",
                    "shape": [2, 6, 6],
                    "root": "/content/snake-cli/gen-test-01",
                }
            ],
        },
    }
    receipt_obj["canary_evidence"] = colab_canary.evidence(canary_mf, receipt_obj, canary_out, ROOT)
    receipt_path = canary_out / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt_path


def test_prepare_canary_ruler_train_and_continuation_validate_with_colab_cli(
    tmp_path: Path,
) -> None:
    """Validate L4 and A100 canary, ruler, train, & sharded FULL continuation packets."""
    for rel in (*CANARY_REQUIRED_EXPORTS, *RULER_REQUIRED_EXPORTS, *TRAIN_REQUIRED_EXPORTS):
        assert not rel.endswith(".sha256")
        assert not rel.endswith(".complete")

    canary_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_l4",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_canary_l4",
        allow_unmerged_commit=True,
    )
    canary_a100 = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_a100",
        commit="HEAD",
        mode="canary",
        gpu="A100",
        output_dir=tmp_path / "out_canary_a100",
        allow_unmerged_commit=True,
    )
    mf_c_l4 = json.loads(Path(canary_l4["manifest_path"]).read_text(encoding="utf-8"))
    mf_c_a100 = json.loads(Path(canary_a100["manifest_path"]).read_text(encoding="utf-8"))

    for mf in (mf_c_l4, mf_c_a100):
        assert mf["timeout_minutes"] <= colab_canary.MAX_CANARY_MINUTES
        assert mf["env"]["SNAKE_COLAB_CANARY"] == "1"
        assert mf["env"]["SNAKE_COLAB_CANARY_PACKAGE"] == "research/v3_ppo"
        assert mf["env"]["V3_PPO_MODE"] == "canary"
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)
        colab_canary.check(mf, ROOT)

    # Verify sparse source archive excludes legacy grid sim & old obs modules!
    source_plan = colab_source.prepare(ROOT, mf_c_l4, tmp_path / "source_plan_l4")
    files_set = set(source_plan["files"])
    assert "research/v3_ppo/job.sh" in files_set
    assert "research/v3_ppo/ppo.py" in files_set
    assert "research/v3_ppo/ruler.py" in files_set
    assert "research/game_v3/sim.py" in files_set
    assert "research/arena_ppo/runtime.py" in files_set
    assert "research/arena_ppo/sim.py" not in files_set
    assert "research/arena_ppo/obs.py" not in files_set
    assert "research/arena_ppo/net.py" not in files_set
    assert "research/arena_ppo/ppo.py" not in files_set
    assert not any(
        name.startswith(("src/", "web/", "research/gpu_sim_spike/")) for name in files_set
    )

    rcpt_l4 = _build_synthetic_canary_receipt(mf_c_l4, "L4")
    l4_exports = Path(mf_c_l4["output_dir"]) / "exports"
    rcpt_a100 = _build_synthetic_canary_receipt(mf_c_a100, "A100", shared_exports_src=l4_exports)

    # Bind ruler & default train on L4, and E=256 train on A100
    ruler_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_ruler_l4",
        commit="HEAD",
        mode="ruler",
        gpu="L4",
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_ruler_l4",
        allow_unmerged_commit=True,
    )
    train_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_l4",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        expected_steps_per_sec=25000.0,
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_train_l4",
        allow_unmerged_commit=True,
    )
    train_a100 = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_a100",
        commit="HEAD",
        mode="train",
        gpu="A100",
        num_envs=256,
        net_size="M",
        hours=4.0,
        canary_receipt=rcpt_a100,
        output_dir=tmp_path / "out_train_a100",
        allow_unmerged_commit=True,
    )

    # Prepare a continuation train packet staging a multi-shard FULL checkpoint set
    prior_sim_dir = tmp_path / "prior_train_exports"
    prior_cfg = V3PPOConfig(
        game_config=GameV3Config(arena_radius=60.0, num_snakes=6),
        num_envs=4,
        rollout_len=4,
        eval_every_minutes=0.0,
    )
    prior_tr = V3PPOTrainer(prior_cfg, device="cpu")
    prior_tr.train_step()
    probe_idx = prior_tr.save_full_checkpoint(tmp_path / "prior_probe", tag="u0000001")
    prior_shard_cap = verify_full_checkpoint_set(probe_idx)["shard_paths"][0].stat().st_size - 1024
    work_c = tmp_path / "prior_work_ckpts"
    idx_prior = prior_tr.save_full_checkpoint(
        work_c, tag="u0000001", max_shard_bytes=prior_shard_cap
    )
    assert verify_full_checkpoint_set(idx_prior)["num_shards"] >= 2
    publish_checkpoint_to_output(
        idx_prior,
        prior_sim_dir / "exports" / "checkpoints",
        reason="completed",
        counters={"updates_completed": 1, "agent_steps_collected": 96, "world_steps_collected": 4},
    )

    canary_cont_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_cont_l4",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        prior_output_dir=prior_sim_dir,
        output_dir=tmp_path / "out_canary_cont_l4",
        allow_unmerged_commit=True,
    )
    mf_c_cont_l4 = json.loads(Path(canary_cont_l4["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_cont_l4 = _build_synthetic_canary_receipt(mf_c_cont_l4, "L4")

    cont_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_cont_l4",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        prior_output_dir=prior_sim_dir,
        canary_receipt=rcpt_cont_l4,
        output_dir=tmp_path / "out_cont_l4",
        allow_unmerged_commit=True,
    )
    cont_ledger = json.loads(
        (Path(cont_l4["packet_dir"]) / "verified-ledger.json").read_text(encoding="utf-8")
    )
    assert cont_ledger["resume_checkpoint"]["checkpoint_kind"] == "full"
    assert len(cont_ledger["resume_checkpoint"]["companion_files"]) >= 3  # light + >=2 shards

    # Verify `prepare_v3_ppo_packet` falls back cleanly to staging LIGHT if a FULL shard is missing
    broken_prior_dir = tmp_path / "broken_prior_exports"
    shutil.copytree(prior_sim_dir, broken_prior_dir)
    next((broken_prior_dir / "exports" / "checkpoints").glob("full_*.sim_shard_*.pt")).unlink()
    cont_light_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_cont_light_l4",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        prior_output_dir=broken_prior_dir,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_cont_light_l4",
        allow_unmerged_commit=True,
    )
    light_ledger = json.loads(
        (Path(cont_light_l4["packet_dir"]) / "verified-ledger.json").read_text(encoding="utf-8")
    )
    assert light_ledger["resume_checkpoint"]["checkpoint_kind"] == "light"
    assert light_ledger["resume_checkpoint"].get("companion_files", []) == []

    for summary, expected_mode in (
        (ruler_l4, "ruler"),
        (train_l4, "train"),
        (train_a100, "train"),
        (cont_l4, "train"),
    ):
        mf = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
        assert "SNAKE_COLAB_CANARY" not in mf["env"]
        assert mf["env"]["V3_PPO_MODE"] == expected_mode
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)
        colab_canary.check(mf, ROOT)


def test_prepare_refuses_checkpoint_shard_and_session_export_cap_violations(
    tmp_path: Path,
) -> None:
    """Verify FULL shard/session caps fire in `final`/`periodic` and NOT in default `none` mode."""
    # 1. Shard smaller than 1 single worst-case world (~30.22 MiB at R=600, N=24):
    #    refused in `periodic`/`final` modes, accepted in default `none` mode!
    tiny_shard_cfg = V3PPOConfig(num_envs=8, checkpoint_shard_bytes=4 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds checkpoint_shard_bytes"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_bad_shard",
            commit="HEAD",
            mode="train",
            gpu="L4",
            full_exports="periodic",
            ppo_config=tiny_shard_cfg,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_bad_shard",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_bad_shard").exists()

    ok_tiny_none = prepare_v3_ppo_packet(
        tmp_path / "pkt_tiny_shard_none",
        commit="HEAD",
        mode="train",
        gpu="L4",
        ppo_config=tiny_shard_cfg,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_tiny_shard_none",
        allow_unmerged_commit=True,
    )
    assert ok_tiny_none["checkpoint_export_caps"]["max_full_exports"] == 0

    # 2. Shard cap whose 1.25x file envelope exceeds the 2.00 GiB supervisor per-file ceiling:
    #    refused in `final`/`periodic` modes, accepted in default `none` mode!
    huge_shard_cfg = V3PPOConfig(num_envs=16, checkpoint_shard_bytes=1800 * 1024 * 1024)
    with pytest.raises(ValueError, match="supervisor 2.00 GiB per-file limit"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_huge_shard",
            commit="HEAD",
            mode="train",
            gpu="L4",
            full_exports="final",
            ppo_config=huge_shard_cfg,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_huge_shard",
            allow_unmerged_commit=True,
        )

    ok_huge_none = prepare_v3_ppo_packet(
        tmp_path / "pkt_huge_shard_none",
        commit="HEAD",
        mode="train",
        gpu="L4",
        ppo_config=huge_shard_cfg,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_huge_shard_none",
        allow_unmerged_commit=True,
    )
    assert ok_huge_none["checkpoint_export_caps"]["per_file_max_export_bytes"] == 80 * 1024 * 1024
    assert (
        ok_huge_none["checkpoint_export_caps"]["session_total_export_cap_bytes"]
        == NONE_MODE_MAX_TOTAL_EXPORT_BYTES
    )

    # 3. Total session export volume exceeding GPU session cap:
    #    (3a) Excessive `max_periodic_full_exports` (`6` periodic + `1` stop = `7` FULLs > `12 GiB`)
    #         refused when periodic FULLs are enabled, ignored in default `none` mode!
    bloated_full_cfg = V3PPOConfig(num_envs=128, max_periodic_full_exports=6)
    with pytest.raises(ValueError, match="exceeds L4 session export cap"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_bloated_session",
            commit="HEAD",
            mode="train",
            gpu="L4",
            hours=8.0,
            full_exports="periodic",
            ppo_config=bloated_full_cfg,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_bloated_session",
            allow_unmerged_commit=True,
        )
    ok_bloated_full_none = prepare_v3_ppo_packet(
        tmp_path / "pkt_bloated_session_none",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=8.0,
        ppo_config=bloated_full_cfg,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_bloated_session_none",
        allow_unmerged_commit=True,
    )
    assert ok_bloated_full_none["checkpoint_export_caps"]["max_full_exports"] == 0

    #    (3b) Update-based LIGHT checkpoint setting (`checkpoint_every_updates=1`) STILL refuses
    #         in default `none` mode against the shrunken 4.00 GiB LIGHT-only session cap!
    bloated_upd_cfg = V3PPOConfig(num_envs=128, checkpoint_every_updates=1)
    with pytest.raises(ValueError, match="exceeds L4 session export cap"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_bloated_updates",
            commit="HEAD",
            mode="train",
            gpu="L4",
            hours=2.0,
            ppo_config=bloated_upd_cfg,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_bloated_updates",
            allow_unmerged_commit=True,
        )

    # 4. Explicit high `measured_packed_bytes_per_world`: refuses in `periodic`, accepted in `none`
    with pytest.raises(ValueError, match="exceeds L4 session export cap"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_high_measured",
            commit="HEAD",
            mode="train",
            gpu="L4",
            num_envs=128,
            full_exports="periodic",
            measured_packed_bytes_per_world=8 * 1024 * 1024,  # 8 MiB * 4.8x = 38.4 MiB/world
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_high_measured",
            allow_unmerged_commit=True,
        )
    ok_high_meas_none = prepare_v3_ppo_packet(
        tmp_path / "pkt_high_measured_none",
        commit="HEAD",
        mode="train",
        gpu="L4",
        num_envs=128,
        measured_packed_bytes_per_world=8 * 1024 * 1024,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_high_measured_none",
        allow_unmerged_commit=True,
    )
    assert ok_high_meas_none["checkpoint_export_caps"]["max_full_exports"] == 0

    # 5. Verify 8 h and 24 h sessions fit cleanly on L4 (E=128) and A100 (E=256) for both S and M
    #    in both default `none` mode (shrunken 80/85 MiB per-file & 4 GiB session caps) and
    #    `periodic` mode (320 MiB per-file & 12/24 GiB session caps).
    for fe_mode in ("none", "periodic"):
        for gpu_kind, n_envs in (("L4", 128), ("A100", 256)):
            for h_val in (8.0, 24.0):
                for net_sz in ("S", "M"):
                    slug = f"{fe_mode}_{gpu_kind.lower()}_{int(h_val)}h_{net_sz.lower()}"
                    ok_info = prepare_v3_ppo_packet(
                        tmp_path / f"pkt_{slug}",
                        commit="HEAD",
                        mode="train",
                        gpu=gpu_kind,
                        hours=h_val,
                        num_envs=n_envs,
                        net_size=net_sz,
                        full_exports=fe_mode,  # type: ignore[arg-type]
                        allow_missing_canary_receipt=True,
                        output_dir=tmp_path / f"out_{slug}",
                        allow_unmerged_commit=True,
                    )
                    caps = ok_info["checkpoint_export_caps"]
                    assert caps["total_session_fits_cap"] is True
                    assert caps["max_light_exports"] == (17 if h_val == 8.0 else 49)
                    mf_ok = json.loads(Path(ok_info["manifest_path"]).read_text(encoding="utf-8"))
                    if fe_mode == "none":
                        assert caps["max_full_exports"] == 0
                        assert (
                            caps["session_total_export_cap_bytes"]
                            == NONE_MODE_MAX_TOTAL_EXPORT_BYTES
                        )
                        exp_per_file = (80 if net_sz == "S" else 85) * 1024 * 1024
                        assert caps["per_file_max_export_bytes"] == exp_per_file
                        assert mf_ok["max_export_bytes"] == exp_per_file
                    else:
                        assert caps["max_full_exports"] == 4
                        assert caps["per_file_max_export_bytes"] == 320 * 1024 * 1024
                    assert int(mf_ok["env"]["V3_PPO_MAX_SESSION_EXPORT_BYTES"]) == int(
                        caps["session_total_export_cap_bytes"]
                    )


def test_ruler_unsynced_headline_vram_precheck_and_single_slot_mature_state_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruler uses T=32 defaults, unsynced headline steps, VRAM precheck, and 1-slot state cache."""
    assert DEFAULT_RULER_ROLLOUT_LEN == 32
    assert DEFAULT_RULER_WARMUP_UPDATES == 1
    assert DEFAULT_RULER_TIMED_UPDATES == 2

    # Default ruler packet uses training rollout_len=32 (so env override is absent at 32)
    ruler_pkt = prepare_v3_ppo_packet(
        tmp_path / "pkt_ruler_def",
        commit="HEAD",
        mode="ruler",
        gpu="A100",
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_ruler_def",
        allow_unmerged_commit=True,
    )
    ruler_mf = json.loads(Path(ruler_pkt["manifest_path"]).read_text(encoding="utf-8"))
    assert "V3_PPO_RULER_ROLLOUT_LEN" not in ruler_mf["env"]

    gcfg = GameV3Config(arena_radius=60.0, num_snakes=6)
    est_e2 = estimate_training_memory(
        V3PPOConfig(
            game_config=gcfg,
            num_envs=2,
            learner_slots=4,
            net_size="S",
            amp=False,
            use_channels_last=False,
            rollout_len=4,
            ppo_epochs=1,
            minibatch_size=8,
            eval_every_minutes=0.0,
            eval_every_updates=0,
        )
    )["estimated_peak_bytes"]
    est_e4 = estimate_training_memory(
        V3PPOConfig(
            game_config=gcfg,
            num_envs=4,
            learner_slots=2,
            net_size="S",
            amp=False,
            use_channels_last=False,
            rollout_len=4,
            ppo_epochs=1,
            minibatch_size=8,
            eval_every_minutes=0.0,
            eval_every_updates=0,
        )
    )["estimated_peak_bytes"]
    assert est_e4 > est_e2
    mid_vram_budget = (est_e2 + est_e4) // 2

    built_envs: list[int] = []
    real_build_packed = ruler_mod._build_packed_mature_sim_state

    def _spy_build_packed(**kwargs: Any) -> dict[str, Any]:
        built_envs.append(int(kwargs["num_envs"]))
        packed = real_build_packed(**kwargs)
        assert "_zlib_tensors" in packed
        return packed

    monkeypatch.setattr(ruler_mod, "_build_packed_mature_sim_state", _spy_build_packed)

    train_step_stage_timings: list[bool] = []
    synced_rollout_calls: list[bool] = []
    orig_train_step = V3PPOTrainer.train_step
    orig_collect_rollout = V3PPOTrainer.collect_rollout

    def _spy_train_step(  # type: ignore[no-untyped-def]
        self: V3PPOTrainer, *, should_stop=None, stage_timing: bool = False
    ):
        train_step_stage_timings.append(bool(stage_timing))
        return orig_train_step(self, should_stop=should_stop, stage_timing=stage_timing)

    def _spy_collect_rollout(  # type: ignore[no-untyped-def]
        self: V3PPOTrainer, *, should_stop=None, stage_timing: bool = False
    ):
        if stage_timing:
            synced_rollout_calls.append(True)
        return orig_collect_rollout(self, should_stop=should_stop, stage_timing=stage_timing)

    monkeypatch.setattr(V3PPOTrainer, "train_step", _spy_train_step)
    monkeypatch.setattr(V3PPOTrainer, "collect_rollout", _spy_collect_rollout)

    res = run_ruler(
        output_dir=tmp_path / "ruler_exec",
        device="cpu",
        envs_list=[2, 4],
        net_sizes=["S"],
        learners_list=[4, 2],
        arena_radius=60.0,
        num_snakes=6,
        burnin_ticks=4,
        warmup_ticks=1,
        collect_ticks=2,
        warmup_updates=1,
        timed_updates=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=8,
        amp=False,
        use_channels_last=False,
        vram_budget_bytes=mid_vram_budget,
    )

    assert res["status"] == "PASS"
    assert res["warmup_updates"] == 1
    assert res["timed_updates"] == 2
    assert set(res["succeeded_cell_ids"]) == {"E2_L2_S", "E2_L4_S"}
    assert set(res["oom_cell_ids"]) == {"E4_L2_S", "E4_L4_S"}
    # Mature sim state built ONLY once for E=2 (shared across L2 & L4), and NEVER built for E=4!
    assert built_envs == [2]

    by_id = {c["cell_id"]: c for c in res["cells"]}
    ok_cell = by_id["E2_L4_S"]
    assert ok_cell["precheck_oom"] is False
    integ = ok_cell["integrated"]
    assert integ["warmup_updates"] == 1
    assert integ["timed_updates"] == 2
    # 2 timed updates * 4 ticks * 2 worlds * 6 snakes = 96 agent steps
    assert integ["agent_steps"] == 2 * 4 * 2 * 6
    assert integ["collect_seconds"] > 0.0
    assert integ["learn_seconds"] > 0.0
    assert integ["headline_stage_timing_syncs"] is False
    assert ok_cell["collection_only"]["headline_stage_timing_syncs"] is False
    assert ok_cell["collect_seconds"] == integ["collect_seconds"]
    assert ok_cell["learn_seconds"] == integ["learn_seconds"]
    # All warmup + timed `train_step` calls ran with `stage_timing=False` (2 * 3 = 6),
    # and each succeeded cell ran 1 separate `collect_rollout(..., stage_timing=True)` pass (= 2).
    assert train_step_stage_timings == [False] * 6
    assert len(synced_rollout_calls) == 2

    oom_cell = by_id["E4_L2_S"]
    assert oom_cell["oom"] is True
    assert oom_cell["precheck_oom"] is True
    assert oom_cell["integrated"] is None


def test_canary_suite_default_eval_uses_full_production_eval_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_canary_suite` without override invokes `evaluate_policy` at full production defaults."""
    seen_canary_eval_cfgs: list[V3PPOConfig] = []

    def _fake_default_eval(_net: Any, eval_cfg: V3PPOConfig, **_kw: Any) -> dict[str, Any]:
        seen_canary_eval_cfgs.append(eval_cfg)
        return {
            "aborted": False,
            "wall_seconds": 1.234,
            "eval_envs": eval_cfg.eval_envs,
            "matchup_envs": {"vs_food_seeking": 96, "vs_random_safe": 96},
            "eval_burnin_ticks": eval_cfg.eval_burnin_ticks,
            "eval_ticks": eval_cfg.eval_ticks,
            "eval_learner_slots": eval_cfg.resolved_eval_learner_slots,
            "matchups": {
                "vs_food_seeking": {"completed_lives": 320},
                "vs_random_safe": {"completed_lives": 350},
            },
        }

    monkeypatch.setattr(ruler_mod, "evaluate_policy", _fake_default_eval)
    canary_def = run_canary_suite(
        output_dir=tmp_path / "canary_def_eval_check",
        device="cpu",
        canary_envs=2,
        canary_arena_radius=60.0,
        canary_num_snakes=6,
        canary_rollout_len=4,
    )
    assert len(seen_canary_eval_cfgs) == 1
    def_cfg = seen_canary_eval_cfgs[0]
    ref_default = V3PPOConfig()
    assert def_cfg.eval_envs == ref_default.eval_envs == 192
    assert def_cfg.eval_burnin_ticks == ref_default.eval_burnin_ticks == 250
    assert def_cfg.eval_ticks == ref_default.eval_ticks == 2000
    assert def_cfg.game_config.arena_radius == 600.0
    assert def_cfg.num_snakes == 24
    assert canary_def["default_eval_benchmark"]["wall_seconds"] == 1.234
    assert canary_def["default_eval_benchmark"]["vs_random_safe_completed_lives"] == 350


def test_prepare_refuses_bad_configs_and_oom_vram_budgets(tmp_path: Path) -> None:
    """Verify `prepare` refuses VRAM overflows, missing/wrong receipts, T4, and bad configs."""
    # 1. E=256 x 24 learners x T=32 (~27.4 GB) MUST be refused on 22 GB L4, accepted on 40 GB A100
    with pytest.raises(ValueError, match="exceeds L4 memory budget"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_oom_l4",
            commit="HEAD",
            mode="train",
            gpu="L4",
            num_envs=256,
            learner_slots=24,
            rollout_len=32,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_oom_l4",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_oom_l4").exists(), "Failed prepare must clean up packet_dir!"

    # 2. E=512 x 24 learners x T=32 (~51.9 GB) MUST be refused on 40 GB A100, accepted on 80 GB H100
    with pytest.raises(ValueError, match="exceeds A100 memory budget"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_oom_a100",
            commit="HEAD",
            mode="train",
            gpu="A100",
            num_envs=512,
            learner_slots=24,
            rollout_len=32,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_oom_a100",
            allow_unmerged_commit=True,
        )

    ok_h100 = prepare_v3_ppo_packet(
        tmp_path / "pkt_ok_h100",
        commit="HEAD",
        mode="train",
        gpu="H100",
        hours=8.0,
        num_envs=512,
        learner_slots=24,
        rollout_len=32,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_ok_h100",
        allow_unmerged_commit=True,
    )
    assert ok_h100["memory_estimate"]["fits_gpu_budget"] is True

    # 3. Ruler packet where zero cells fit on 22 GB L4 (`E=1024`) is refused
    with pytest.raises(ValueError, match="zero requested ruler cells fit"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_oom_ruler_l4",
            commit="HEAD",
            mode="ruler",
            gpu="L4",
            ruler_envs="1024",
            ruler_learners="24",
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_oom_ruler_l4",
            allow_unmerged_commit=True,
        )

    # 4. Unsupported GPU (`T4`) refused
    with pytest.raises(ValueError, match="T4"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_t4",
            commit="HEAD",
            mode="canary",
            gpu="T4",
            output_dir=tmp_path / "out_t4",
            allow_unmerged_commit=True,
        )

    # 5. Missing `--canary-receipt` refused on ruler & train via both Python API and CLI
    for mode in ("ruler", "train"):
        with pytest.raises(ValueError, match="canary_receipt"):
            prepare_v3_ppo_packet(
                tmp_path / f"pkt_no_rcpt_{mode}",
                commit="HEAD",
                mode=mode,
                gpu="L4",
                output_dir=tmp_path / f"out_no_rcpt_{mode}",
                allow_unmerged_commit=True,
            )
        with pytest.raises(SystemExit):
            prepare_main(
                [
                    "--mode",
                    mode,
                    "--gpu",
                    "L4",
                    "--packet",
                    str(tmp_path / f"cli_no_rcpt_{mode}"),
                    "--allow-unmerged-commit",
                ]
            )

    # 6. Wrong GPU hardware in canary receipt refused (binding L4 receipt to A100 packet)
    c_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_c_l4_hw",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_c_l4_hw",
        allow_unmerged_commit=True,
    )
    mf_c_l4 = json.loads(Path(c_l4["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_l4 = _build_synthetic_canary_receipt(mf_c_l4, "L4")
    with pytest.raises(ValueError, match="does not match canary receipt hardware"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_wrong_hw",
            commit="HEAD",
            mode="ruler",
            gpu="A100",
            canary_receipt=rcpt_l4,
            output_dir=tmp_path / "out_wrong_hw",
            allow_unmerged_commit=True,
        )

    # 7. Unmerged commit refused when `allow_unmerged_commit=False`
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    is_merged = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", head_sha, "origin/main"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    if not is_merged:
        with pytest.raises(ValueError, match="origin/main"):
            prepare_v3_ppo_packet(
                tmp_path / "pkt_unmerged",
                commit="HEAD",
                mode="canary",
                gpu="L4",
                output_dir=tmp_path / "out_unmerged",
                allow_unmerged_commit=False,
            )


def test_non_pass_outcomes_exit_nonzero(tmp_path: Path) -> None:
    """Both `ppo` and `ruler` CLIs must return non-zero exit code when status != 'PASS'."""
    out_ppo = tmp_path / "ppo_fail_out"
    rc_ppo = ppo_main(
        [
            "--output-dir",
            str(out_ppo),
            "--work-dir",
            str(tmp_path / "ppo_fail_work"),
            "--device",
            "cpu",
            "--num-envs",
            "2",
            "--num-snakes",
            "6",
            "--arena-radius",
            "60.0",
            "--rollout-len",
            "4",
            "--total-updates",
            "5",
            "--eval-every-minutes",
            "0.0",
            "--eval-every-updates",
            "0",
            "--max-wall-seconds",
            "0.0001",
        ]
    )
    assert rc_ppo != 0
    train_res_path = out_ppo / "train_result.json"
    verify_completed_artifact(train_res_path)
    train_res = json.loads(train_res_path.read_text(encoding="utf-8"))
    assert train_res["status"] == "FAIL"

    out_ruler = tmp_path / "ruler_fail_out"
    rc_ruler = ruler_main(
        [
            "--output-dir",
            str(out_ruler),
            "--device",
            "cpu",
            "--envs",
            "2",
            "--nets",
            "S",
            "--learners",
            "6",
            "--arena-radius",
            "60.0",
            "--num-snakes",
            "6",
            "--max-wall-seconds",
            "0.0001",
        ]
    )
    assert rc_ruler != 0
    ruler_res_path = out_ruler / "ruler_result.json"
    verify_completed_artifact(ruler_res_path)
    ruler_res = json.loads(ruler_res_path.read_text(encoding="utf-8"))
    assert ruler_res["status"] == "FAIL"


def test_canary_completed_exports_never_mutate_for_supervisor_and_have_st_nlink_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed canary exports never mutate under supervisor polling and keep st_nlink == 1."""
    import research.arena_ppo.runtime as arena_rt
    from research.colab_cli.runtime import _exports as scan_supervisor_exports

    cli_root = tmp_path / "colab_cli_canary_root"
    out_dir = cli_root / "output"
    max_export_bytes = 320 * 1024 * 1024
    supervisor_cache: dict[str, Any] = {}

    orig_fsync = arena_rt._fsync_directory
    scan_invocations = 0

    def _polling_fsync(directory: Path) -> None:
        nonlocal scan_invocations
        orig_fsync(directory)
        if out_dir.exists():
            scan_invocations += 1
            scan_supervisor_exports(cli_root, max_export_bytes, supervisor_cache)

    monkeypatch.setattr(arena_rt, "_fsync_directory", _polling_fsync)

    fast_eval_override = V3PPOConfig(
        game_config=GameV3Config(arena_radius=60.0, num_snakes=6),
        rollout_len=4,
        eval_envs=4,
        eval_burnin_ticks=4,
        eval_ticks=12,
        eval_learner_slots=2,
    )
    canary_res = run_canary_suite(
        output_dir=out_dir,
        device="cpu",
        canary_envs=2,
        canary_arena_radius=60.0,
        canary_num_snakes=6,
        canary_rollout_len=4,
        canary_train_envs=4,
        canary_eval_cfg_override=fast_eval_override,
    )
    assert canary_res["status"] == "PASS"

    # Poll again AFTER `run_canary_suite`'s `finally: shutil.rmtree(tmp_dir)` has run!
    final_records = scan_supervisor_exports(cli_root, max_export_bytes, supervisor_cache)
    assert scan_invocations >= 15
    exported_relpaths = {rec["path"] for rec in final_records}
    for req in CANARY_REQUIRED_EXPORTS:
        assert req in exported_relpaths
        verify_completed_artifact(out_dir / req)

    completed_markers = sorted(out_dir.rglob("*.complete"))
    assert len(completed_markers) == len(supervisor_cache)
    for marker in completed_markers:
        payload = marker.with_name(marker.name[: -len(".complete")])
        sha_sidecar = Path(f"{payload}.sha256")
        assert payload.stat().st_nlink == 1, f"Expected st_nlink==1 on {payload}"
        assert sha_sidecar.stat().st_nlink == 1, f"Expected st_nlink==1 on {sha_sidecar}"
        assert marker.stat().st_nlink == 1, f"Expected st_nlink==1 on {marker}"


def test_canary_eval_resident_train_sim_combined_peak_and_prepare_eval_estimate_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canary default eval holds GPU-default training sim resident and propagates wall estimate."""
    # 1. Verify `prepare --mode canary` sets GPU-default training world count (`128`/`256`/`512`)
    expected_gpu_envs = {"L4": "128", "A100": "256", "H100": "512"}
    canary_manifests: dict[str, dict[str, Any]] = {}
    for gpu_kind, exp_envs in expected_gpu_envs.items():
        pkt_info = prepare_v3_ppo_packet(
            tmp_path / f"pkt_canary_{gpu_kind.lower()}",
            commit="HEAD",
            mode="canary",
            gpu=gpu_kind,
            output_dir=tmp_path / f"out_canary_{gpu_kind.lower()}",
            allow_unmerged_commit=True,
        )
        mf = json.loads(Path(pkt_info["manifest_path"]).read_text(encoding="utf-8"))
        assert mf["env"]["V3_PPO_CANARY_TRAIN_ENVS"] == exp_envs
        canary_manifests[gpu_kind] = mf

    # 2. Verify `run_canary_suite` keeps the resident training `V3PPOTrainer` (`sim` + `buffer`)
    #    allocated concurrently while `evaluate_policy` runs!
    import weakref

    live_trainers: weakref.WeakSet[V3PPOTrainer] = weakref.WeakSet()
    orig_trainer_cls = ruler_mod.V3PPOTrainer

    def _tracking_trainer(*args: Any, **kwargs: Any) -> V3PPOTrainer:
        inst = orig_trainer_cls(*args, **kwargs)
        live_trainers.add(inst)
        return inst

    monkeypatch.setattr(ruler_mod, "V3PPOTrainer", _tracking_trainer)

    real_eval_policy = ruler_mod.evaluate_policy
    live_sim_world_counts_during_eval: list[list[int]] = []

    def _spy_eval_policy(model: Any, eval_cfg: V3PPOConfig, **kwargs: Any) -> dict[str, Any]:
        live_sim_world_counts_during_eval.append(
            sorted(int(tr.sim.E) for tr in live_trainers if tr.buffer.nbytes_total() > 0)
        )
        res = real_eval_policy(model, eval_cfg, **kwargs)
        # Override wall_seconds to a distinctive non-95 value to prove propagation into `prepare`!
        res["wall_seconds"] = 37.25
        return res

    monkeypatch.setattr(ruler_mod, "evaluate_policy", _spy_eval_policy)

    canary_l4_mf = canary_manifests["L4"]
    canary_exports_dir = tmp_path / "shared_canary_exports"
    remote_in = _stage_manifest_inputs(canary_l4_mf, tmp_path / "shared_remote_inputs")

    fast_eval_override = V3PPOConfig(
        game_config=GameV3Config(arena_radius=60.0, num_snakes=6),
        rollout_len=4,
        eval_envs=4,
        eval_burnin_ticks=4,
        eval_ticks=12,
        eval_learner_slots=2,
    )
    canary_res = run_canary_suite(
        output_dir=canary_exports_dir,
        device="cpu",
        input_dir=remote_in,
        ledger_path=remote_in / "verified-ledger.json",
        canary_envs=2,
        canary_arena_radius=60.0,
        canary_num_snakes=6,
        canary_rollout_len=4,
        canary_train_envs=6,
        canary_eval_cfg_override=fast_eval_override,
    )
    monkeypatch.setattr(ruler_mod, "evaluate_policy", real_eval_policy)

    # Resident 6-world training `GameV3Sim` MUST be live when `evaluate_policy` starts!
    assert live_sim_world_counts_during_eval == [[6]]
    assert canary_res["default_eval_resident_train_envs"] == 6
    assert canary_res["default_eval_wall_seconds"] == pytest.approx(37.25)
    assert canary_res["default_eval_peak_memory_bytes"] > 0
    assert canary_res["default_eval_peak_memory_mib"] > 0.0
    assert canary_res["default_eval_benchmark"]["resident_train_envs"] == 6
    assert canary_res["default_eval_benchmark"]["peak_memory_bytes"] > 0

    rcpt_l4 = _build_synthetic_canary_receipt(
        canary_manifests["L4"],
        "L4",
        shared_exports_src=canary_exports_dir,
    )
    assert rcpt_l4.is_file()

    # 3. `prepare --mode train` takes `eval_estimated_seconds` (`37.25`) from bound canary receipt!
    train_pkt = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_from_canary_est",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_train_from_canary_est",
        allow_unmerged_commit=True,
    )
    train_mf = json.loads(Path(train_pkt["manifest_path"]).read_text(encoding="utf-8"))
    train_cfg = json.loads(train_mf["env"]["V3_PPO_CONFIG_JSON"])
    assert train_cfg["eval_estimated_seconds"] == pytest.approx(37.25)
    assert train_mf["technical_envelope"]["canary_eval_wall_seconds"] == pytest.approx(37.25)
    assert train_mf["technical_envelope"]["canary_eval_peak_memory_bytes"] == int(
        canary_res["default_eval_peak_memory_bytes"]
    )

    # Explicit caller override (`eval_estimated_seconds=110.0`) still takes precedence when passed!
    train_override_pkt = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_explicit_eval_est",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        eval_estimated_seconds=110.0,
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_train_explicit_eval_est",
        allow_unmerged_commit=True,
    )
    override_mf = json.loads(Path(train_override_pkt["manifest_path"]).read_text(encoding="utf-8"))
    override_cfg = json.loads(override_mf["env"]["V3_PPO_CONFIG_JSON"])
    assert override_cfg["eval_estimated_seconds"] == pytest.approx(110.0)


def _materialize_baseline_root(tmp_path: Path, commit_ref: str) -> Path:
    """Create an isolated repo tree with ``commit_ref``'s ``research/v3_ppo/`` extracted."""
    baseline_root = tmp_path / f"baseline_{commit_ref}"
    baseline_root.mkdir(parents=True)
    (baseline_root / "src").symlink_to(ROOT / "src")
    (baseline_root / "tools").symlink_to(ROOT / "tools")
    (baseline_root / "research").mkdir()
    for src_p in (ROOT / "research").iterdir():
        if src_p.name != "v3_ppo":
            (baseline_root / "research" / src_p.name).symlink_to(src_p)
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", commit_ref, "research/v3_ppo"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-xf", "-", "-C", str(baseline_root)],
        input=archive.stdout,
        check=True,
    )
    return baseline_root


def _write_effective_packet_probe_script(probe_script: Path) -> Path:
    """Write helper script capturing `job.sh` effective configs across canary/ruler/train."""
    probe_script.write_text(
        """from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
work_root = Path(sys.argv[2]).resolve()
out_json = Path(sys.argv[3]).resolve()
git_repo = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(repo_root))

import torch
import research.v3_ppo.ppo as ppo_mod
import research.v3_ppo.prepare as prep_mod
import research.v3_ppo.ruler as ruler_mod
from research.v3_ppo.config import V3PPOConfig

prep_mod.ROOT = git_repo

def run_job_sh_and_capture(manifest_path: Path, pkt_dir: Path, run_dir: Path) -> dict:
    mf = json.loads(manifest_path.read_text(encoding="utf-8"))
    bin_dir = run_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    cap_path = run_dir / "cap.json"
    stub_py = bin_dir / "python"
    stub_py.write_text(
        "#!/usr/bin/env bash\\n"
        f"{sys.executable} -c 'import json, os, sys; "
        f"open({json.dumps(str(cap_path))}, \\"w\\").write(json.dumps({{"
        "\\"argv\\": sys.argv[1:], "
        "\\"env\\": {k: v for k, v in os.environ.items() "
        "if k.startswith((\\"V3_PPO_\\", \\"SNAKE_\\"))}"
        "}))' \\"$@\\"\\n",
        encoding="utf-8",
    )
    stub_py.chmod(0o755)
    cli_root = run_dir / "cli_root"
    (cli_root / "output").mkdir(parents=True, exist_ok=True)
    (cli_root / "work").mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": os.environ.get("HOME", "/tmp"),
        "COLAB_CLI_ROOT": str(cli_root),
        "COLAB_INPUT_DIR": str(pkt_dir / "inputs"),
    }
    env.update({k: str(v) for k, v in mf["env"].items()})
    subprocess.run(
        ["bash", str(git_repo / "research" / "v3_ppo" / "job.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    cap = json.loads(cap_path.read_text(encoding="utf-8"))
    argv = cap["argv"]
    saved_env = os.environ.copy()
    captured_call: dict = {}
    try:
        for k in list(os.environ):
            if k.startswith(("V3_PPO_", "SNAKE_")):
                del os.environ[k]
        os.environ.update(cap["env"])
        if argv[:2] == ["-m", "research.v3_ppo.ppo"]:
            orig_fn = ppo_mod.run_training_job
            def _spy_train(**kw):
                captured_call["entry"] = "run_training_job"
                captured_call["cfg"] = kw["cfg"].to_dict()
                captured_call["device"] = str(kw["device"])
                captured_call["max_wall_seconds"] = kw["max_wall_seconds"]
                captured_call["max_session_export_bytes"] = kw["max_session_export_bytes"]
                return {"status": "PASS"}
            ppo_mod.run_training_job = _spy_train
            try:
                assert ppo_mod.main(argv[2:]) == 0
            finally:
                ppo_mod.run_training_job = orig_fn
        elif argv[:2] == ["-m", "research.v3_ppo.ruler"]:
            orig_canary = ruler_mod.run_canary_suite
            orig_ruler = ruler_mod.run_ruler
            def _spy_canary(**kw):
                captured_call["entry"] = "run_canary_suite"
                captured_call["kwargs"] = {
                    k: v for k, v in kw.items()
                    if k not in {"output_dir", "input_dir", "ledger_path"}
                }
                return {"status": "PASS"}
            def _spy_ruler(**kw):
                captured_call["entry"] = "run_ruler"
                captured_call["kwargs"] = {
                    k: v for k, v in kw.items()
                    if k not in {"output_dir", "input_dir"}
                }
                return {"status": "PASS"}
            ruler_mod.run_canary_suite = _spy_canary
            ruler_mod.run_ruler = _spy_ruler
            try:
                assert ruler_mod.main(argv[2:]) == 0
            finally:
                ruler_mod.run_canary_suite = orig_canary
                ruler_mod.run_ruler = orig_ruler
        else:
            raise AssertionError(f"Unexpected argv: {argv}")
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
    return captured_call

results = {}
s_canary = prep_mod.prepare_v3_ppo_packet(
    work_root / "pkt_canary", mode="canary", gpu="L4",
    output_dir=work_root / "out_canary", allow_unmerged_commit=True,
)
results["canary_l4"] = run_job_sh_and_capture(
    Path(s_canary["manifest_path"]), Path(s_canary["packet_dir"]), work_root / "r_canary"
)

s_ruler = prep_mod.prepare_v3_ppo_packet(
    work_root / "pkt_ruler", mode="ruler", gpu="A100",
    ruler_nets="S,M", ruler_envs="512,1024", amp=False, use_channels_last=False,
    compile_model=True, fast_learner_ops=False,
    output_dir=work_root / "out_ruler",
    allow_missing_canary_receipt=True, allow_unmerged_commit=True,
)
results["ruler_a100_custom"] = run_job_sh_and_capture(
    Path(s_ruler["manifest_path"]), Path(s_ruler["packet_dir"]), work_root / "r_ruler"
)

s_train_s = prep_mod.prepare_v3_ppo_packet(
    work_root / "pkt_train_s", mode="train", gpu="L4", net_size="S",
    output_dir=work_root / "out_train_s",
    allow_missing_canary_receipt=True, allow_unmerged_commit=True,
)
results["train_s_default"] = run_job_sh_and_capture(
    Path(s_train_s["manifest_path"]), Path(s_train_s["packet_dir"]), work_root / "r_train_s"
)

s_train_m = prep_mod.prepare_v3_ppo_packet(
    work_root / "pkt_train_m", mode="train", gpu="A100", net_size="M", num_envs=128,
    compile_model=True, amp=False, use_channels_last=False, fast_learner_ops=False,
    output_dir=work_root / "out_train_m",
    allow_missing_canary_receipt=True, allow_unmerged_commit=True,
)
results["train_m_custom"] = run_job_sh_and_capture(
    Path(s_train_m["manifest_path"]), Path(s_train_m["packet_dir"]), work_root / "r_train_m"
)

out_json.write_text(json.dumps(results, sort_keys=True), encoding="utf-8")
""",
        encoding="utf-8",
    )
    return probe_script


def test_prepare_and_job_sh_effective_configs_match_e2865f93_baseline(tmp_path: Path) -> None:
    """Prove canary/ruler/train packets match e2865f93 effective configs (with earlier wall guard)."""
    baseline_root = _materialize_baseline_root(tmp_path, "e2865f93")
    probe_script = _write_effective_packet_probe_script(tmp_path / "probe_effective_packets.py")

    sub_env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    before_json = tmp_path / "before.json"
    after_json = tmp_path / "after.json"
    subprocess.run(
        [
            sys.executable,
            str(probe_script),
            str(baseline_root),
            str(tmp_path / "w_before"),
            str(before_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(baseline_root)},
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(probe_script),
            str(ROOT),
            str(tmp_path / "w_after"),
            str(after_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(ROOT)},
        check=True,
    )
    before = json.loads(before_json.read_text(encoding="utf-8"))
    after = json.loads(after_json.read_text(encoding="utf-8"))
    assert after["canary_l4"] == before["canary_l4"]
    assert after["ruler_a100_custom"] == before["ruler_a100_custom"]

    for train_key, gpu_kind, net_sz in (
        ("train_s_default", "L4", "S"),
        ("train_m_custom", "A100", "M"),
    ):
        assert after[train_key]["entry"] == before[train_key]["entry"]
        assert after[train_key]["device"] == before[train_key]["device"]
        assert after[train_key]["cfg"] == {**before[train_key]["cfg"], "full_exports": "none"}
        assert after[train_key]["max_session_export_bytes"] == NONE_MODE_MAX_TOTAL_EXPORT_BYTES
        exp_wall = float(
            estimate_train_wall_headroom(
                V3PPOConfig(net_size=net_sz, num_envs=128),  # type: ignore[arg-type]
                gpu=gpu_kind,
                session_seconds=7200.0,
            )["max_wall_seconds"]
        )
        assert after[train_key]["max_wall_seconds"] == pytest.approx(exp_wall)


def test_canary_and_ruler_packets_byte_for_byte_identical_vs_c85c6837(tmp_path: Path) -> None:
    """Prove canary and ruler packets are unchanged byte-for-byte vs c85c6837."""
    baseline_root = _materialize_baseline_root(tmp_path, "c85c6837")
    probe_script = _write_effective_packet_probe_script(tmp_path / "probe_effective_c85c6837.py")

    sub_env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    eff_before_json = tmp_path / "eff_c85c6837.json"
    eff_after_json = tmp_path / "eff_head.json"
    subprocess.run(
        [
            sys.executable,
            str(probe_script),
            str(baseline_root),
            str(tmp_path / "eff_w_before"),
            str(eff_before_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(baseline_root)},
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(probe_script),
            str(ROOT),
            str(tmp_path / "eff_w_after"),
            str(eff_after_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(ROOT)},
        check=True,
    )
    eff_before = json.loads(eff_before_json.read_text(encoding="utf-8"))
    eff_after = json.loads(eff_after_json.read_text(encoding="utf-8"))
    assert eff_after["canary_l4"] == eff_before["canary_l4"]
    assert eff_after["ruler_a100_custom"] == eff_before["ruler_a100_custom"]

    # Now verify literal byte-for-byte identity of every file in canary and ruler packet dirs
    byte_script = tmp_path / "byte_compare_canary_ruler.py"
    byte_script.write_text(
        """from __future__ import annotations
import base64, json, shutil, sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
shared_pkt_root = Path(sys.argv[2]).resolve()
out_json = Path(sys.argv[3]).resolve()
git_repo = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(repo_root))

import research.v3_ppo.prepare as prep_mod
prep_mod.ROOT = git_repo

if shared_pkt_root.exists():
    shutil.rmtree(shared_pkt_root)
shared_pkt_root.mkdir(parents=True)

cases = {
    "canary_l4": dict(mode="canary", gpu="L4"),
    "canary_a100_flags": dict(
        mode="canary", gpu="A100", compile_model=True, fast_learner_ops=False
    ),
    "canary_h100": dict(mode="canary", gpu="H100"),
    "ruler_l4_default": dict(
        mode="ruler", gpu="L4", allow_missing_canary_receipt=True
    ),
    "ruler_a100_custom": dict(
        mode="ruler", gpu="A100", ruler_nets="S,M", ruler_envs="512,1024",
        amp=False, use_channels_last=False, compile_model=True, fast_learner_ops=False,
        allow_missing_canary_receipt=True,
    ),
}

snapshots = {}
for name, kw in cases.items():
    pkt_dir = shared_pkt_root / f"pkt_{name}"
    out_dir = shared_pkt_root / f"out_{name}"
    summary = prep_mod.prepare_v3_ppo_packet(
        pkt_dir,
        commit="c85c6837",
        output_dir=out_dir,
        allow_unmerged_commit=True,
        **kw,
    )
    files_b64 = {}
    for p in sorted(pkt_dir.rglob("*")):
        if p.is_file():
            files_b64[p.relative_to(pkt_dir).as_posix()] = base64.b64encode(
                p.read_bytes()
            ).decode("ascii")
    snapshots[name] = {
        "summary_json": json.dumps(summary, indent=2, sort_keys=True),
        "files_b64": files_b64,
    }

out_json.write_text(json.dumps(snapshots, sort_keys=True), encoding="utf-8")
""",
        encoding="utf-8",
    )

    shared_pkt_root = tmp_path / "shared_byte_pkts"
    snap_before_json = tmp_path / "snap_c85c6837.json"
    snap_after_json = tmp_path / "snap_head.json"
    subprocess.run(
        [
            sys.executable,
            str(byte_script),
            str(baseline_root),
            str(shared_pkt_root),
            str(snap_before_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(baseline_root)},
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(byte_script),
            str(ROOT),
            str(shared_pkt_root),
            str(snap_after_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(ROOT)},
        check=True,
    )
    snap_before = json.loads(snap_before_json.read_text(encoding="utf-8"))
    snap_after = json.loads(snap_after_json.read_text(encoding="utf-8"))
    assert snap_after.keys() == snap_before.keys()
    for case_name in snap_before:
        assert (
            snap_after[case_name]["files_b64"] == snap_before[case_name]["files_b64"]
        ), f"Packet file bytes changed vs c85c6837 for {case_name}"
        assert (
            snap_after[case_name]["summary_json"] == snap_before[case_name]["summary_json"]
        ), f"Summary JSON changed vs c85c6837 for {case_name}"


def test_default_train_packet_differs_from_c85c6837_only_in_documented_ways(
    tmp_path: Path,
) -> None:
    """Prove default `train` packets differ from `c85c6837` strictly in documented `none` fields."""
    baseline_root = _materialize_baseline_root(tmp_path, "c85c6837")
    train_diff_script = tmp_path / "train_diff_c85c6837.py"
    train_diff_script.write_text(
        """from __future__ import annotations
import base64, json, shutil, sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
shared_pkt_root = Path(sys.argv[2]).resolve()
out_json = Path(sys.argv[3]).resolve()
git_repo = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(repo_root))

import research.v3_ppo.prepare as prep_mod
prep_mod.ROOT = git_repo

if shared_pkt_root.exists():
    shutil.rmtree(shared_pkt_root)
shared_pkt_root.mkdir(parents=True)

cases = {
    "train_l4_default": dict(mode="train", gpu="L4", allow_missing_canary_receipt=True),
    "train_a100_8h_m": dict(
        mode="train", gpu="A100", num_envs=256, net_size="M", hours=8.0,
        allow_missing_canary_receipt=True,
    ),
}

snapshots = {}
for name, kw in cases.items():
    pkt_dir = shared_pkt_root / f"pkt_{name}"
    out_dir = shared_pkt_root / f"out_{name}"
    prep_mod.prepare_v3_ppo_packet(
        pkt_dir,
        commit="c85c6837",
        output_dir=out_dir,
        allow_unmerged_commit=True,
        **kw,
    )
    files_b64 = {}
    for p in sorted(pkt_dir.rglob("*")):
        if p.is_file():
            files_b64[p.relative_to(pkt_dir).as_posix()] = base64.b64encode(
                p.read_bytes()
            ).decode("ascii")
    snapshots[name] = {
        "manifest": json.loads((pkt_dir / "dispatch-manifest.json").read_text(encoding="utf-8")),
        "files_b64": files_b64,
    }

out_json.write_text(json.dumps(snapshots, sort_keys=True), encoding="utf-8")
""",
        encoding="utf-8",
    )

    sub_env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    shared_root = tmp_path / "shared_train_diff_pkts"
    before_path = tmp_path / "train_c85c6837.json"
    after_path = tmp_path / "train_head.json"
    subprocess.run(
        [
            sys.executable,
            str(train_diff_script),
            str(baseline_root),
            str(shared_root),
            str(before_path),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(baseline_root)},
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(train_diff_script),
            str(ROOT),
            str(shared_root),
            str(after_path),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(ROOT)},
        check=True,
    )
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    expected_case_specs = {
        "train_l4_default": ("L4", 128, "S", 2.0, 80 * 1024 * 1024),
        "train_a100_8h_m": ("A100", 256, "M", 8.0, 85 * 1024 * 1024),
    }
    for case_name, (gpu_kind, n_envs, net_sz, h_val, exp_per_file) in expected_case_specs.items():
        b_files = before[case_name]["files_b64"]
        a_files = after[case_name]["files_b64"]
        expected_pkt_files = {
            "dispatch-manifest.json",
            "inputs/verified-ledger.json",
            "inputs/verified-ledger.json.sha256",
            "inputs/verified-ledger.json.complete",
            "verified-ledger.json",
        }
        assert set(a_files.keys()) == set(b_files.keys()) == expected_pkt_files
        # Every non-manifest packet file is byte-for-byte identical vs c85c6837!
        for rel_non_mf in sorted(expected_pkt_files - {"dispatch-manifest.json"}):
            assert a_files[rel_non_mf] == b_files[rel_non_mf]

        mf_b = before[case_name]["manifest"]
        mf_a = after[case_name]["manifest"]
        documented_top_diffs = {"max_export_bytes", "env", "technical_envelope"}
        assert {k: v for k, v in mf_a.items() if k not in documented_top_diffs} == {
            k: v for k, v in mf_b.items() if k not in documented_top_diffs
        }
        assert mf_a["max_export_bytes"] == exp_per_file

        # `env` differs ONLY in `V3_PPO_MAX_WALL_SECONDS`, `V3_PPO_MAX_SESSION_EXPORT_BYTES`, and
        # `V3_PPO_CONFIG_JSON` (`"full_exports": "none"` added).
        documented_env_diffs = {
            "V3_PPO_MAX_WALL_SECONDS",
            "V3_PPO_MAX_SESSION_EXPORT_BYTES",
            "V3_PPO_CONFIG_JSON",
        }
        assert {k: v for k, v in mf_a["env"].items() if k not in documented_env_diffs} == {
            k: v for k, v in mf_b["env"].items() if k not in documented_env_diffs
        }
        cfg_b = json.loads(mf_b["env"]["V3_PPO_CONFIG_JSON"])
        cfg_a = json.loads(mf_a["env"]["V3_PPO_CONFIG_JSON"])
        assert "max_periodic_full_exports" not in cfg_a
        assert cfg_a == {
            **{k: v for k, v in cfg_b.items() if k != "max_periodic_full_exports"},
            "full_exports": "none",
        }
        assert (
            int(mf_a["env"]["V3_PPO_MAX_SESSION_EXPORT_BYTES"]) == NONE_MODE_MAX_TOTAL_EXPORT_BYTES
        )

        exp_cfg = V3PPOConfig(num_envs=n_envs, net_size=net_sz, full_exports="none")  # type: ignore[arg-type]
        exp_caps = estimate_checkpoint_export_caps(
            exp_cfg, gpu=gpu_kind, max_wall_seconds=h_val * 3600.0
        )
        exp_hr = estimate_train_wall_headroom(exp_cfg, gpu=gpu_kind, session_seconds=h_val * 3600.0)
        assert int(mf_a["env"]["V3_PPO_MAX_WALL_SECONDS"]) == int(exp_hr["max_wall_seconds"])

        # `technical_envelope` differs ONLY in session export caps & `wall_headroom`.
        documented_tech_diffs = {
            "session_total_export_cap_bytes",
            "worst_case_session_export_bytes",
            "wall_headroom",
        }
        assert {
            k: v for k, v in mf_a["technical_envelope"].items() if k not in documented_tech_diffs
        } == {k: v for k, v in mf_b["technical_envelope"].items() if k not in documented_tech_diffs}
        assert (
            mf_a["technical_envelope"]["session_total_export_cap_bytes"]
            == NONE_MODE_MAX_TOTAL_EXPORT_BYTES
        )
        assert mf_a["technical_envelope"]["worst_case_session_export_bytes"] == int(
            exp_caps["worst_case_session_export_bytes"]
        )
        assert mf_a["technical_envelope"]["wall_headroom"] == exp_hr


def test_train_wall_headroom_arithmetic_gpu_defaults_summary_and_40pct_refusal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify headroom arithmetic per mode (`none`/`final`/`periodic`), warnings, & >40% refusal."""
    assert TRAIN_SETUP_ALLOWANCE_SECONDS == 330.0
    assert TRAINER_CHECKPOINT_RESERVE_SECONDS == 90.0
    assert TRAIN_FINAL_EVAL_AND_SAVE_ALLOWANCE_SECONDS == 300.0
    assert TRAIN_FINAL_SAVE_AFTER_EVAL_ALLOWANCE_SECONDS == 150.0
    assert TRAIN_FINAL_TELEMETRY_EXPORT_BYTES == 4 * 1024 * 1024
    assert ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS == 1.0
    assert ASSUMED_STAGED_INPUT_UPLOAD_RATE_BYTES_PER_SEC == 1_000_000.0
    assert ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS == 1.0
    assert ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_BYTES_PER_SEC == 1_000_000.0
    assert MAX_TRAIN_HEADROOM_SESSION_FRACTION == 0.40

    expected_cases = [
        # (gpu, envs, net, hours, none_bytes, none_wall, full_bytes, full_wall, full_fits_40)
        ("L4", 128, "S", 4.0, 20_971_520, 13_749, 2_164_785_152, 11_605, True),
        ("L4", 128, "S", 8.0, 20_971_520, 28_149, 2_164_785_152, 26_005, True),
        ("L4", 128, "M", 4.0, 75_497_472, 13_694, 2_219_311_104, 11_550, True),
        ("L4", 128, "M", 8.0, 75_497_472, 28_094, 2_219_311_104, 25_950, True),
        ("A100", 256, "S", 4.0, 20_971_520, 13_749, 4_312_793_088, 9_457, True),
        ("A100", 256, "S", 8.0, 20_971_520, 28_149, 4_312_793_088, 23_857, True),
        ("A100", 256, "M", 4.0, 75_497_472, 13_694, 4_367_319_040, 9_402, True),
        ("A100", 256, "M", 8.0, 75_497_472, 28_094, 4_367_319_040, 23_802, True),
        ("H100", 512, "S", 4.0, 20_971_520, 13_749, 8_608_808_960, 5_161, False),
        ("H100", 512, "S", 8.0, 20_971_520, 28_149, 8_608_808_960, 19_561, True),
        ("H100", 512, "M", 4.0, 75_497_472, 13_694, 8_663_334_912, 5_106, False),
        ("H100", 512, "M", 8.0, 75_497_472, 28_094, 8_663_334_912, 19_506, True),
    ]

    for (
        gpu_kind,
        n_envs,
        net_sz,
        h_val,
        none_bytes,
        none_wall,
        full_bytes,
        full_wall,
        full_fits,
    ) in expected_cases:
        sess_sec = h_val * 3600.0
        by_mode = estimate_train_wall_headroom_by_mode(
            V3PPOConfig(num_envs=n_envs, net_size=net_sz),  # type: ignore[arg-type]
            gpu=gpu_kind,
            session_seconds=sess_sec,
        )
        assert tuple(by_mode.keys()) == ("none", "final", "periodic")

        for fe_mode in ("none", "final", "periodic"):
            cfg = V3PPOConfig(
                num_envs=n_envs,
                net_size=net_sz,  # type: ignore[arg-type]
                full_exports=fe_mode,  # type: ignore[arg-type]
            )
            caps = estimate_checkpoint_export_caps(cfg, gpu=gpu_kind, max_wall_seconds=sess_sec)
            est = estimate_train_wall_headroom(cfg, gpu=gpu_kind, session_seconds=sess_sec)
            assert by_mode[fe_mode] == est

            exp_bytes = none_bytes if fe_mode == "none" else full_bytes
            exp_wall = none_wall if fe_mode == "none" else full_wall
            exp_fits = True if fe_mode == "none" else full_fits

            assert est["full_exports"] == fe_mode
            assert est["final_light_bytes"] == int(caps["est_light_checkpoint_bytes"])
            if fe_mode == "none":
                assert est["final_full_bytes"] == 0
                assert est["final_telemetry_bytes"] == TRAIN_FINAL_TELEMETRY_EXPORT_BYTES
            else:
                assert est["final_full_bytes"] == int(caps["est_full_sim_bytes"])
                assert est["final_telemetry_bytes"] == 0
            assert est["final_export_bytes"] == exp_bytes
            assert est["setup_allowance_seconds"] == 330.0
            assert est["staged_input_bytes"] == 0
            assert est["staged_input_upload_seconds"] == 0.0
            assert est["trainer_checkpoint_reserve_seconds"] == 90.0
            assert est["final_eval_and_save_allowance_seconds"] == 300.0
            assert est["final_export_download_seconds"] == pytest.approx(exp_bytes / 1_000_000.0)
            assert est["headroom_seconds"] == pytest.approx(330.0 + 300.0 + exp_bytes / 1_000_000.0)
            assert est["max_wall_seconds"] == exp_wall
            assert est["fits_max_headroom_fraction"] is exp_fits

            slug = f"{fe_mode}_{gpu_kind.lower()}_{n_envs}_{net_sz.lower()}_{int(h_val)}h"
            pkt_path = tmp_path / f"pkt_{slug}"
            capsys.readouterr()
            if exp_fits:
                summary = prepare_v3_ppo_packet(
                    pkt_path,
                    commit="HEAD",
                    mode="train",
                    gpu=gpu_kind,
                    hours=h_val,
                    num_envs=n_envs,
                    net_size=net_sz,  # type: ignore[arg-type]
                    full_exports=None if fe_mode == "none" else fe_mode,  # type: ignore[arg-type]
                    allow_missing_canary_receipt=True,
                    output_dir=tmp_path / f"out_{slug}",
                    allow_unmerged_commit=True,
                )
                prep_err = capsys.readouterr().err
                assert summary["timeout_minutes"] == pytest.approx(h_val * 60.0)
                assert summary["max_wall_seconds"] == exp_wall
                wh = summary["wall_headroom"]
                assert wh["headroom_seconds"] == pytest.approx(est["headroom_seconds"])
                assert summary["wall_headroom_by_mode"] == by_mode
                assert "V3_PPO_HEADROOM_BY_MODE " in prep_err
                if fe_mode == "periodic":
                    assert "WARNING [V3_PPO_PERIODIC_FULL_EXPORTS]" in prep_err
                    assert "UPLOAD_CAP" in prep_err
                else:
                    assert "V3_PPO_PERIODIC_FULL_EXPORTS" not in prep_err
                mf = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
                assert mf["timeout_minutes"] == pytest.approx(h_val * 60.0)
                assert int(mf["env"]["V3_PPO_MAX_WALL_SECONDS"]) == exp_wall
                assert mf["technical_envelope"]["wall_headroom"] == wh
                staged_cfg_json = json.loads(mf["env"]["V3_PPO_CONFIG_JSON"])
                if fe_mode == "none":
                    assert "max_periodic_full_exports" not in staged_cfg_json
                else:
                    assert "max_periodic_full_exports" in staged_cfg_json
            else:
                with pytest.raises(
                    ValueError,
                    match=(
                        rf"Refusing train packet whose wall-guard headroom.*"
                        rf"exceeds 40% of the {int(h_val)} h session"
                    ),
                ):
                    prepare_v3_ppo_packet(
                        pkt_path,
                        commit="HEAD",
                        mode="train",
                        gpu=gpu_kind,
                        hours=h_val,
                        num_envs=n_envs,
                        net_size=net_sz,  # type: ignore[arg-type]
                        full_exports=fe_mode,  # type: ignore[arg-type]
                        allow_missing_canary_receipt=True,
                        output_dir=tmp_path / f"out_{slug}",
                        allow_unmerged_commit=True,
                    )
                assert not pkt_path.exists()

    # Short sessions on A100 (256 worlds @ 2h) and L4 (128 worlds @ 1h):
    # accepted in default `none` mode, refused in `final`/`periodic` modes!
    for short_gpu, short_envs, short_hours in (("A100", 256, 2.0), ("L4", 128, 1.0)):
        ok_short = prepare_v3_ppo_packet(
            tmp_path / f"pkt_ok_none_{short_gpu.lower()}_{int(short_hours)}h",
            commit="HEAD",
            mode="train",
            gpu=short_gpu,
            hours=short_hours,
            num_envs=short_envs,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / f"out_ok_none_{short_gpu.lower()}",
            allow_unmerged_commit=True,
        )
        assert ok_short["wall_headroom"]["full_exports"] == "none"

        bad_pkt = tmp_path / f"pkt_refuse_{short_gpu.lower()}_{int(short_hours)}h"
        with pytest.raises(ValueError, match="exceeds 40%"):
            prepare_v3_ppo_packet(
                bad_pkt,
                commit="HEAD",
                mode="train",
                gpu=short_gpu,
                hours=short_hours,
                num_envs=short_envs,
                full_exports="final",
                allow_missing_canary_receipt=True,
                output_dir=tmp_path / f"out_refuse_{short_gpu.lower()}",
                allow_unmerged_commit=True,
            )
        assert not bad_pkt.exists()

    # Bound canary receipt provides measured evaluation seconds -> term 2 = eval_wall_s + 150 s!
    canary_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_c_l4_hr",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_c_l4_hr",
        allow_unmerged_commit=True,
    )
    mf_c_l4 = json.loads(Path(canary_l4["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_l4 = _build_synthetic_canary_receipt(mf_c_l4, "L4")
    canary_checks = json.loads(
        (Path(mf_c_l4["output_dir"]) / "exports" / "canary_checks.json").read_text(encoding="utf-8")
    )
    measured_eval_s = float(canary_checks["default_eval_wall_seconds"])
    capsys.readouterr()

    # Default CLI (`--full-exports none`) uses LIGHT+telemetry download term and emits NO warning
    rc_cli_none = prepare_main(
        [
            "--mode",
            "train",
            "--gpu",
            "L4",
            "--num-envs",
            "128",
            "--net-size",
            "S",
            "--hours",
            "4.0",
            "--canary-receipt",
            str(rcpt_l4),
            "--output-dir",
            str(tmp_path / "out_cli_l4_4h_none"),
            "--packet",
            str(tmp_path / "pkt_cli_l4_4h_none"),
            "--allow-unmerged-commit",
        ]
    )
    assert rc_cli_none == 0
    captured_none = capsys.readouterr()
    cli_summary_none = json.loads(captured_none.out)
    cli_wh_none = cli_summary_none["wall_headroom"]
    assert cli_wh_none["full_exports"] == "none"
    assert cli_wh_none["final_eval_and_save_source"] == "canary_receipt_eval_plus_150s"
    assert cli_wh_none["setup_allowance_seconds"] == 330.0
    assert cli_wh_none["final_eval_and_save_allowance_seconds"] == pytest.approx(
        measured_eval_s + 150.0
    )
    assert cli_wh_none["final_export_download_seconds"] == pytest.approx(20_971_520 / 1_000_000.0)
    assert cli_wh_none["headroom_seconds"] == pytest.approx(
        330.0 + (measured_eval_s + 150.0) + 20_971_520 / 1_000_000.0
    )
    assert set(cli_summary_none["wall_headroom_by_mode"].keys()) == {"none", "final", "periodic"}
    assert "V3_PPO_HEADROOM " in captured_none.err
    assert "V3_PPO_HEADROOM_BY_MODE " in captured_none.err
    assert "V3_PPO_PERIODIC_FULL_EXPORTS" not in captured_none.err

    # CLI `--full-exports periodic` uses FULL+LIGHT download term and prints loud warning!
    rc_cli_per = prepare_main(
        [
            "--mode",
            "train",
            "--gpu",
            "L4",
            "--num-envs",
            "128",
            "--net-size",
            "S",
            "--hours",
            "4.0",
            "--full-exports",
            "periodic",
            "--canary-receipt",
            str(rcpt_l4),
            "--output-dir",
            str(tmp_path / "out_cli_l4_4h_per"),
            "--packet",
            str(tmp_path / "pkt_cli_l4_4h_per"),
            "--allow-unmerged-commit",
        ]
    )
    assert rc_cli_per == 0
    captured_per = capsys.readouterr()
    cli_summary_per = json.loads(captured_per.out)
    cli_wh_per = cli_summary_per["wall_headroom"]
    assert cli_wh_per["full_exports"] == "periodic"
    assert cli_wh_per["final_export_download_seconds"] == pytest.approx(2_164_785_152 / 1_000_000.0)
    assert "WARNING [V3_PPO_PERIODIC_FULL_EXPORTS]" in captured_per.err


def test_light_only_prior_output_continuation_prepare_canary_and_train_without_error_warnings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """LIGHT-only prior output resumes cleanly in `prepare`, canary, and `ppo` without WARN tags."""
    prior_root = tmp_path / "prior_light_only_session"
    prior_exports = prior_root / "exports"
    prior_cfg = V3PPOConfig(
        game_config=GameV3Config(arena_radius=60.0, num_snakes=6),
        num_envs=3,
        learner_slots=4,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=8,
        amp=False,
        use_channels_last=False,
        total_updates=2,
        eval_every_minutes=0.0,
        eval_every_updates=0,
        full_exports="none",
    )
    res_prior = run_training_job(
        cfg=prior_cfg,
        output_dir=prior_exports,
        work_dir=tmp_path / "prior_work",
        device="cpu",
    )
    assert res_prior["status"] == "PASS"
    assert list(prior_exports.rglob("full_*")) == []
    assert (prior_exports / "checkpoints" / "light_u0000002.pt").is_file()
    capsys.readouterr()

    # 1. `prepare --mode canary --prior-output-dir`: zero WARN/ERROR lines
    canary_cont = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_light_cont",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        prior_output_dir=prior_root,
        output_dir=tmp_path / "out_canary_light_cont",
        allow_unmerged_commit=True,
    )
    prep_c_io = capsys.readouterr()
    assert "WARN" not in prep_c_io.out and "WARN" not in prep_c_io.err
    assert "ERROR" not in prep_c_io.out and "ERROR" not in prep_c_io.err

    mf_c_cont = json.loads(Path(canary_cont["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_cont = _build_synthetic_canary_receipt(mf_c_cont, "L4")
    canary_io = capsys.readouterr()
    assert "worlds reinitialised" in canary_io.out
    assert "WARN" not in canary_io.out and "WARN" not in canary_io.err
    assert "ERROR" not in canary_io.out and "ERROR" not in canary_io.err

    canary_checks = json.loads(
        (Path(mf_c_cont["output_dir"]) / "exports" / "canary_checks.json").read_text(
            encoding="utf-8"
        )
    )
    cont_probe = canary_checks["staged_input_verification"]
    assert cont_probe is not None
    assert cont_probe["kind"] == "light"
    assert cont_probe["num_shards"] == 0
    assert cont_probe["updates_completed"] == 2

    # 2. `prepare --mode train --prior-output-dir` + `run_training_job`: zero WARN/ERROR lines,
    #    and `"worlds reinitialised"` stays in stdout!
    train_cont = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_light_cont",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        prior_output_dir=prior_root,
        canary_receipt=rcpt_cont,
        output_dir=tmp_path / "out_train_light_cont",
        allow_unmerged_commit=True,
    )
    prep_t_io = capsys.readouterr()
    assert "WARN" not in prep_t_io.out and "WARN" not in prep_t_io.err
    assert "ERROR" not in prep_t_io.out and "ERROR" not in prep_t_io.err

    mf_t_cont = json.loads(Path(train_cont["manifest_path"]).read_text(encoding="utf-8"))
    prior_light_bytes = int((prior_exports / "checkpoints" / "light_u0000002.pt").stat().st_size)
    cont_wh = train_cont["wall_headroom"]
    assert cont_wh["staged_input_bytes"] == prior_light_bytes
    assert cont_wh["staged_input_upload_seconds"] == pytest.approx(prior_light_bytes / 1_000_000.0)
    assert cont_wh["headroom_seconds"] == pytest.approx(
        330.0
        + prior_light_bytes / 1_000_000.0
        + cont_wh["final_eval_and_save_allowance_seconds"]
        + cont_wh["final_export_download_seconds"]
    )
    staged_in = _stage_manifest_inputs(mf_t_cont, tmp_path / "staged_train_light_in")
    cont_cfg = V3PPOConfig.from_dict(
        {**json.loads(mf_t_cont["env"]["V3_PPO_CONFIG_JSON"]), "total_updates": 4}
    )
    assert cont_cfg.full_exports == "none"

    res_cont = run_training_job(
        cfg=cont_cfg,
        output_dir=tmp_path / "exec_train_light_cont_out",
        work_dir=tmp_path / "exec_train_light_cont_work",
        device="cpu",
        input_dir=staged_in,
        ledger_path=staged_in / "verified-ledger.json",
    )
    run_t_io = capsys.readouterr()
    assert res_cont["status"] == "PASS"
    assert res_cont["updates_completed"] == 4
    assert res_cont["updates_this_session"] == 2
    assert res_cont["resumed_from"]["checkpoint_kind"] == "light"
    assert res_cont["resumed_from"]["worlds_reinitialized"] is True
    assert "worlds reinitialised" in run_t_io.out
    assert "WARN" not in run_t_io.out and "WARN" not in run_t_io.err
    assert "ERROR" not in run_t_io.out and "ERROR" not in run_t_io.err


def _resolve_project_module_relpaths(dotted: str) -> list[str]:
    """Resolve a ``research.*`` or ``src.*`` module name to tracked repository relpaths."""
    if not (dotted in {"research", "src"} or dotted.startswith(("research.", "src."))):
        return []
    parts = dotted.split(".")
    relpaths: list[str] = []
    for depth in range(1, len(parts)):
        pkg_init = ROOT / Path(*parts[:depth]) / "__init__.py"
        if pkg_init.is_file():
            relpaths.append(pkg_init.relative_to(ROOT).as_posix())
    mod_file = (ROOT / Path(*parts)).with_suffix(".py")
    pkg_dir = ROOT / Path(*parts)
    pkg_file = pkg_dir / "__init__.py"
    if mod_file.is_file():
        relpaths.append(mod_file.relative_to(ROOT).as_posix())
    elif pkg_file.is_file():
        relpaths.append(pkg_file.relative_to(ROOT).as_posix())
    elif pkg_dir.is_dir():
        # PEP 420 namespace package directory (e.g., `research/`): require at least one `.py` child
        # when imported directly without a submodule, otherwise resolve via submodule imports.
        pass
    else:
        raise AssertionError(f"Unresolvable project import {dotted!r}")
    return relpaths


def _collect_imports_from_nodes(
    nodes: list[ast.AST],
    cur_pkg_parts: list[str],
    referenced_symbols: dict[str, set[str]],
    module_aliases: dict[str, str],
) -> list[str]:
    """Extract imported project module names and symbol references from ``nodes``."""
    candidate_modules: list[str] = []
    for root_node in nodes:
        for node in ast.walk(root_node):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidate_modules.append(alias.name)
                    if alias.name.startswith(("research.", "src.")):
                        bound = alias.asname or alias.name.split(".")[0]
                        module_aliases[bound] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    ascend = node.level - 1
                    base_parts = (
                        cur_pkg_parts[: len(cur_pkg_parts) - ascend]
                        if ascend > 0
                        else cur_pkg_parts
                    )
                    base_mod = ".".join(
                        base_parts + (node.module.split(".") if node.module else [])
                    )
                else:
                    base_mod = node.module or ""
                if base_mod:
                    candidate_modules.append(base_mod)
                    for alias in node.names:
                        if alias.name == "*":
                            referenced_symbols.setdefault(base_mod, set()).add("*")
                        else:
                            sub_cand = f"{base_mod}.{alias.name}"
                            sub_parts = sub_cand.split(".")
                            if (ROOT / Path(*sub_parts)).with_suffix(".py").is_file() or (
                                ROOT / Path(*sub_parts) / "__init__.py"
                            ).is_file():
                                candidate_modules.append(sub_cand)
                                module_aliases[alias.asname or alias.name] = sub_cand
                            else:
                                referenced_symbols.setdefault(base_mod, set()).add(alias.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                target_mod = module_aliases.get(node.value.id)
                if target_mod is not None:
                    referenced_symbols.setdefault(target_mod, set()).add(node.attr)
    return candidate_modules


def _reachable_nodes_in_shared_module(tree: ast.Module, wanted_symbols: set[str]) -> list[ast.AST]:
    """Return top-level non-def statements plus defs reachable from ``wanted_symbols``."""
    if "*" in wanted_symbols:
        return list(tree.body)
    defs: dict[str, ast.AST] = {}
    always_nodes: list[ast.AST] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[stmt.name] = stmt
        else:
            always_nodes.append(stmt)

    seen_syms: set[str] = set()
    sym_queue = [s for s in wanted_symbols if s in defs]
    reachable_defs: list[ast.AST] = []
    while sym_queue:
        sym = sym_queue.pop()
        if sym in seen_syms:
            continue
        seen_syms.add(sym)
        def_node = defs[sym]
        reachable_defs.append(def_node)
        for sub in ast.walk(def_node):
            if isinstance(sub, ast.Name) and sub.id in defs and sub.id not in seen_syms:
                sym_queue.append(sub.id)
    return always_nodes + reachable_defs


def _walk_transitive_project_imports(entry_relpaths: list[str]) -> set[str]:
    """Walk transitive project imports of ``entry_relpaths`` across ``research.*`` and ``src.*``."""
    owned_prefixes = ("research/v3_ppo/", "research/game_v3/")
    discovered: set[str] = set()
    referenced_symbols: dict[str, set[str]] = {}
    processed_shared_syms: dict[str, set[str]] = {}
    queue: list[str] = []

    def _enqueue(rel: str) -> None:
        parts = Path(rel).with_suffix("").parts
        for depth in range(1, len(parts)):
            pkg_init = ROOT / Path(*parts[:depth]) / "__init__.py"
            if pkg_init.is_file():
                init_rel = pkg_init.relative_to(ROOT).as_posix()
                if init_rel not in discovered:
                    discovered.add(init_rel)
                    queue.append(init_rel)
        if rel not in discovered:
            discovered.add(rel)
            queue.append(rel)

    for entry in entry_relpaths:
        _enqueue(entry)

    while queue:
        rel = queue.pop()
        file_path = ROOT / rel
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=rel)
        cur_pkg_parts = list(Path(rel).parent.parts)
        mod_dotted = ".".join(Path(rel).with_suffix("").parts)
        if mod_dotted.endswith(".__init__"):
            mod_dotted = mod_dotted[: -len(".__init__")]

        if rel.startswith(owned_prefixes):
            nodes_to_scan: list[ast.AST] = list(tree.body)
        else:
            wanted = set(referenced_symbols.get(mod_dotted, set()))
            processed_shared_syms[rel] = set(wanted)
            nodes_to_scan = _reachable_nodes_in_shared_module(tree, wanted)

        aliases: dict[str, str] = {}
        prev_sym_snapshot = {k: set(v) for k, v in referenced_symbols.items()}
        candidate_modules = _collect_imports_from_nodes(
            nodes_to_scan, cur_pkg_parts, referenced_symbols, aliases
        )

        for mod_name in candidate_modules:
            for resolved_rel in _resolve_project_module_relpaths(mod_name):
                if resolved_rel not in discovered:
                    discovered.add(resolved_rel)
                    if resolved_rel.endswith(".py"):
                        queue.append(resolved_rel)
                elif not resolved_rel.startswith(owned_prefixes):
                    target_dotted = ".".join(Path(resolved_rel).with_suffix("").parts)
                    if target_dotted.endswith(".__init__"):
                        target_dotted = target_dotted[: -len(".__init__")]
                    cur_syms = referenced_symbols.get(target_dotted, set())
                    if not cur_syms.issubset(processed_shared_syms.get(resolved_rel, set())):
                        queue.append(resolved_rel)

        # Re-queue any already-discovered shared module that gained newly referenced symbols
        for mod_key, sym_set in referenced_symbols.items():
            if sym_set != prev_sym_snapshot.get(mod_key, set()):
                for resolved_rel in _resolve_project_module_relpaths(mod_key):
                    if (
                        resolved_rel in discovered
                        and not resolved_rel.startswith(owned_prefixes)
                        and not sym_set.issubset(processed_shared_syms.get(resolved_rel, set()))
                    ):
                        queue.append(resolved_rel)

    return discovered


def test_runtime_source_paths_covers_all_entrypoint_imports(tmp_path: Path) -> None:
    """Every transitive project import of `ppo` and `ruler` is shipped in `runtime_source_paths`."""
    shipped = set(runtime_source_paths("HEAD"))
    imported = _walk_transitive_project_imports(
        ["research/v3_ppo/ppo.py", "research/v3_ppo/ruler.py"]
    )
    missing = sorted(imported - shipped)
    assert not missing, f"Entrypoint imports missing from runtime_source_paths: {missing}"

    # Stage ONLY the shipped source list into an isolated directory and execute entrypoints from it.
    isolated = tmp_path / "isolated_colab_tree"
    for rel in sorted(shipped):
        dst = isolated / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    iso_probe = (
        "import pathlib, sys\n"
        f"repo_root = pathlib.Path({str(ROOT)!r}).resolve()\n"
        f"iso_root = pathlib.Path({str(isolated)!r}).resolve()\n"
        "sys.path[:] = [\n"
        "    str(iso_root),\n"
        "    *(\n"
        "        p for p in sys.path\n"
        "        if p and pathlib.Path(p).resolve() != repo_root\n"
        "    ),\n"
        "]\n"
        "import research.v3_ppo.ppo as ppo_m\n"
        "import research.v3_ppo.ruler as ruler_m\n"
        "from research.game_v3.config import GameV3Config\n"
        "from research.v3_ppo.config import V3PPOConfig\n"
        "assert pathlib.Path(ppo_m.__file__).resolve().is_relative_to(iso_root)\n"
        "assert pathlib.Path(ruler_m.__file__).resolve().is_relative_to(iso_root)\n"
        "ppo_m.build_arg_parser()\n"
        "ruler_m.build_arg_parser()\n"
        "cfg = V3PPOConfig(\n"
        "    game_config=GameV3Config(arena_radius=60.0, num_snakes=6),\n"
        "    num_envs=2, learner_slots=4, rollout_len=2, ppo_epochs=1, minibatch_size=4,\n"
        "    amp=False, use_channels_last=False, eval_every_minutes=0.0, eval_every_updates=0,\n"
        ")\n"
        "tr = ppo_m.V3PPOTrainer(cfg, device='cpu')\n"
        "tr.train_step()\n"
    )
    sub_env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": str(isolated),
    }
    subprocess.run(
        [sys.executable, "-c", iso_probe],
        cwd=str(isolated),
        env=sub_env,
        check=True,
        capture_output=True,
        text=True,
    )

    # Deleting a shipped module from the isolated tree MUST fail with ModuleNotFoundError.
    (isolated / "research" / "v3_ppo" / "exports.py").unlink()
    failed = subprocess.run(
        [sys.executable, "-c", iso_probe],
        cwd=str(isolated),
        env=sub_env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "ModuleNotFoundError" in failed.stderr


def test_job_sh_env_mapping_matches_prepare_across_all_modes(tmp_path: Path) -> None:
    """Every env var emitted by `prepare` is mapped by `job.sh` into matching CLI args."""
    job_sh_path = ROOT / "research" / "v3_ppo" / "job.sh"
    job_sh_text = job_sh_path.read_text(encoding="utf-8")
    supervisor_only_env_keys = {
        "COLAB_CLI_REQUIREMENTS",
        "SNAKE_COLAB_CANARY",
        "SNAKE_COLAB_CANARY_PACKAGE",
    }

    def _capture_job_sh_invocation(
        manifest_path: Path, pkt_dir: Path, run_dir: Path
    ) -> dict[str, Any]:
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in mf["env"]:
            if key not in supervisor_only_env_keys:
                assert key in job_sh_text, f"{key} emitted by prepare is missing from job.sh"

        bin_dir = run_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        cap_path = run_dir / "cap.json"
        stub_py = bin_dir / "python"
        stub_py.write_text(
            "#!/usr/bin/env bash\n"
            f"{sys.executable} -c 'import json, os, sys; "
            f'open({json.dumps(str(cap_path))}, "w").write(json.dumps({{'
            '"argv": sys.argv[1:], '
            '"env": {k: v for k, v in os.environ.items() '
            'if k.startswith(("V3_PPO_", "SNAKE_"))}'
            '}))\' "$@"\n',
            encoding="utf-8",
        )
        stub_py.chmod(0o755)
        cli_root = run_dir / "cli_root"
        (cli_root / "output").mkdir(parents=True, exist_ok=True)
        (cli_root / "work").mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": os.environ.get("HOME", "/tmp"),
            "COLAB_CLI_ROOT": str(cli_root),
            "COLAB_INPUT_DIR": str(pkt_dir / "inputs"),
            **{k: str(v) for k, v in mf["env"].items()},
        }
        subprocess.run(
            ["bash", str(job_sh_path)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(cap_path.read_text(encoding="utf-8"))

    # 1. Canary packet with non-default compile + fast-learner flags on A100
    canary_info = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_map",
        commit="HEAD",
        mode="canary",
        gpu="A100",
        compile_model=True,
        fast_learner_ops=False,
        output_dir=tmp_path / "out_canary_map",
        allow_unmerged_commit=True,
    )
    canary_cap = _capture_job_sh_invocation(
        Path(canary_info["manifest_path"]),
        Path(canary_info["packet_dir"]),
        tmp_path / "run_canary_map",
    )
    assert canary_cap["argv"][:2] == ["-m", "research.v3_ppo.ruler"]
    canary_eff = resolve_effective_ruler_config(
        build_ruler_arg_parser().parse_args(canary_cap["argv"][2:]),
        environ=canary_cap["env"],
    )
    assert canary_eff["mode"] == "canary"
    assert canary_eff["canary_train_envs"] == 256
    assert canary_eff["compile_model"] is True
    assert canary_eff["fast_learner_ops"] is False

    # 2. Ruler packet exercising every ruler env override
    ruler_info = prepare_v3_ppo_packet(
        tmp_path / "pkt_ruler_map",
        commit="HEAD",
        mode="ruler",
        gpu="A100",
        ruler_envs="128,256",
        ruler_nets="S,M",
        ruler_learners="12,24",
        arena_radius=450.0,
        num_snakes=24,
        ruler_rollout_len=16,
        ruler_warmup_updates=2,
        ruler_timed_updates=3,
        amp=False,
        use_channels_last=False,
        include_fp32=True,
        compile_model=True,
        fast_learner_ops=False,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_ruler_map",
        allow_unmerged_commit=True,
    )
    ruler_cap = _capture_job_sh_invocation(
        Path(ruler_info["manifest_path"]),
        Path(ruler_info["packet_dir"]),
        tmp_path / "run_ruler_map",
    )
    assert ruler_cap["argv"][:2] == ["-m", "research.v3_ppo.ruler"]
    ruler_eff = resolve_effective_ruler_config(
        build_ruler_arg_parser().parse_args(ruler_cap["argv"][2:]),
        environ=ruler_cap["env"],
    )
    assert ruler_eff["mode"] == "ruler"
    assert ruler_eff["envs_list"] == [128, 256]
    assert ruler_eff["net_sizes"] == ["S", "M"]
    assert ruler_eff["learners_list"] == [12, 24]
    assert ruler_eff["arena_radius"] == 450.0
    assert ruler_eff["num_snakes"] == 24
    assert ruler_eff["rollout_len"] == 16
    assert ruler_eff["warmup_updates"] == 2
    assert ruler_eff["timed_updates"] == 3
    assert ruler_eff["amp"] is False
    assert ruler_eff["use_channels_last"] is False
    assert ruler_eff["include_fp32"] is True
    assert ruler_eff["compile_model"] is True
    assert ruler_eff["fast_learner_ops"] is False

    # 3. Train packet exercising train env overrides + config JSON
    train_info = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_map",
        commit="HEAD",
        mode="train",
        gpu="A100",
        hours=4.0,
        net_size="M",
        num_envs=128,
        learner_slots=18,
        arena_radius=500.0,
        num_snakes=20,
        rollout_len=16,
        amp=False,
        use_channels_last=False,
        compile_model=True,
        fast_learner_ops=False,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_train_map",
        allow_unmerged_commit=True,
    )
    train_cap = _capture_job_sh_invocation(
        Path(train_info["manifest_path"]),
        Path(train_info["packet_dir"]),
        tmp_path / "run_train_map",
    )
    assert train_cap["argv"][:2] == ["-m", "research.v3_ppo.ppo"]
    train_eff = resolve_effective_train_config(
        build_train_arg_parser().parse_args(train_cap["argv"][2:]),
        environ=train_cap["env"],
    )
    assert train_eff.cfg.net_size == "M"
    assert train_eff.cfg.num_envs == 128
    assert train_eff.cfg.learner_slots == 18
    assert train_eff.cfg.game_config.arena_radius == 500.0
    assert train_eff.cfg.num_snakes == 20
    assert train_eff.cfg.rollout_len == 16
    assert train_eff.cfg.amp is False
    assert train_eff.cfg.use_channels_last is False
    assert train_eff.cfg.compile_model is True
    assert train_eff.cfg.fast_learner_ops is False
    assert train_eff.max_session_export_bytes == int(
        train_info["checkpoint_export_caps"]["session_total_export_cap_bytes"]
    )


def test_prepare_recipe_json_overrides_summary_vram_guard_and_runtime_recording(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--recipe-json` CLI/file overrides, summary output, VRAM checks, and runtime logs."""
    # 1. Default train packet prints `recipe overrides: none` and omits `recipe_overrides` key
    capsys.readouterr()
    rc_none = prepare_main(
        [
            "--packet",
            str(tmp_path / "pkt_recipe_none"),
            "--commit",
            "HEAD",
            "--mode",
            "train",
            "--gpu",
            "L4",
            "--hours",
            "4.0",
            "--allow-missing-canary-receipt",
            "--output-dir",
            str(tmp_path / "out_recipe_none"),
            "--allow-unmerged-commit",
        ]
    )
    io_none = capsys.readouterr()
    assert rc_none == 0
    assert "recipe overrides: none" in io_none.err
    summary_none = json.loads(io_none.out)
    assert "recipe_overrides" not in summary_none

    # 2. Inline JSON & file path `--recipe-json` flow into staged config, summary, and VRAM/plan
    recipe_obj = {
        "reward_alive_length_coef": 0.0,
        "reward_death_penalty": 2.0,
        "gamma": 0.995,
        "entropy_coef": 0.005,
        "lr": 1.5e-4,
        "ppo_epochs": 3,
        "minibatch_size": 512,
        "rollout_len": 64,
        "clip_ratio": 0.15,
    }
    recipe_path = tmp_path / "custom_recipe.json"
    recipe_path.write_text(json.dumps(recipe_obj), encoding="utf-8")

    capsys.readouterr()
    rc_ov = prepare_main(
        [
            "--packet",
            str(tmp_path / "pkt_recipe_ov"),
            "--commit",
            "HEAD",
            "--mode",
            "train",
            "--gpu",
            "A100",
            "--hours",
            "4.0",
            "--net-size",
            "M",
            "--expected-steps-per-sec",
            "20000",
            "--recipe-json",
            str(recipe_path),
            "--allow-missing-canary-receipt",
            "--output-dir",
            str(tmp_path / "out_recipe_ov"),
            "--allow-unmerged-commit",
        ]
    )
    io_ov = capsys.readouterr()
    assert rc_ov == 0
    assert (
        "recipe overrides: clip_ratio=0.15, entropy_coef=0.005, gamma=0.995, lr=0.00015, "
        "minibatch_size=512, ppo_epochs=3, reward_alive_length_coef=0.0, "
        "reward_death_penalty=2.0, rollout_len=64"
    ) in io_ov.err
    summary_ov = json.loads(io_ov.out)
    assert summary_ov["recipe_overrides"] == recipe_obj
    # Overridden rollout_len=64 doubles agent_steps_per_update (64 * 128 * 24 = 196,608)
    assert summary_ov["training_plan"]["agent_steps_per_update"] == 64 * 128 * 24

    mf_ov = json.loads(
        (tmp_path / "pkt_recipe_ov" / "dispatch-manifest.json").read_text(encoding="utf-8")
    )
    eff_ov = resolve_effective_train_config(
        build_train_arg_parser().parse_args(["--output-dir", str(tmp_path / "eff_ov_out")]),
        environ=mf_ov["env"],
    )
    staged_cfg_ov = eff_ov.cfg
    for key, val in recipe_obj.items():
        assert getattr(staged_cfg_ov, key) == val
    assert staged_cfg_ov.net_size == "M"
    assert staged_cfg_ov.resolved_minibatch_size == 512
    assert eff_ov.recipe_overrides == recipe_obj

    # 2b. Setting `--recipe-json` to code-default values produces `recipe overrides: none` in both
    #     `prepare` and `resolve_effective_train_config` / `V3_PPO_RECIPE`, while `--rollout-len 64`
    #     (without `--recipe-json`) reports `rollout_len=64` in both!
    capsys.readouterr()
    summary_def_eq = prepare_v3_ppo_packet(
        tmp_path / "pkt_recipe_def_eq",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=4.0,
        recipe_json=json.dumps({"gamma": 0.99, "minibatch_size": 2048, "rollout_len": 32}),
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_recipe_def_eq",
        allow_unmerged_commit=True,
    )
    err_def_eq = capsys.readouterr().err
    assert "recipe overrides: none" in err_def_eq
    assert "recipe_overrides" not in summary_def_eq
    mf_def_eq = json.loads(Path(summary_def_eq["manifest_path"]).read_text(encoding="utf-8"))
    eff_def_eq = resolve_effective_train_config(
        build_train_arg_parser().parse_args(["--output-dir", str(tmp_path / "eff_def_eq_out")]),
        environ=mf_def_eq["env"],
    )
    assert eff_def_eq.recipe_overrides == {}

    capsys.readouterr()
    summary_rl64 = prepare_v3_ppo_packet(
        tmp_path / "pkt_recipe_rl64",
        commit="HEAD",
        mode="train",
        gpu="A100",
        hours=4.0,
        rollout_len=64,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_recipe_rl64",
        allow_unmerged_commit=True,
    )
    err_rl64 = capsys.readouterr().err
    assert "recipe overrides: rollout_len=64" in err_rl64
    assert summary_rl64["recipe_overrides"] == {"rollout_len": 64}
    mf_rl64 = json.loads(Path(summary_rl64["manifest_path"]).read_text(encoding="utf-8"))
    eff_rl64 = resolve_effective_train_config(
        build_train_arg_parser().parse_args(["--output-dir", str(tmp_path / "eff_rl64_out")]),
        environ=mf_rl64["env"],
    )
    assert eff_rl64.recipe_overrides == {"rollout_len": 64}

    # 3. Memory budget check uses overridden rollout_len and minibatch_size!
    with pytest.raises(ValueError, match="exceeds L4 memory budget"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_vram_rollout",
            commit="HEAD",
            mode="train",
            gpu="L4",
            num_envs=128,
            recipe_json=json.dumps({"rollout_len": 512}),
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_vram_rollout",
            allow_unmerged_commit=True,
        )
    with pytest.raises(ValueError, match="exceeds L4 memory budget"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_vram_mb",
            commit="HEAD",
            mode="train",
            gpu="L4",
            num_envs=128,
            net_size="M",
            recipe_json=json.dumps({"minibatch_size": 65536}),
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_vram_mb",
            allow_unmerged_commit=True,
        )

    # 4. Non-train modes and invalid recipe JSONs are refused at prepare time
    for non_train_mode in ("canary", "ruler"):
        with pytest.raises(ValueError, match="only supported in --mode train"):
            prepare_v3_ppo_packet(
                tmp_path / f"pkt_bad_mode_{non_train_mode}",
                commit="HEAD",
                mode=non_train_mode,
                gpu="L4",
                recipe_json=json.dumps({"gamma": 0.99}),
                allow_missing_canary_receipt=(non_train_mode != "canary"),
                output_dir=tmp_path / f"out_bad_mode_{non_train_mode}",
                allow_unmerged_commit=True,
            )
    for idx_bad, bad_json in enumerate(
        (
            json.dumps({"value_coef": 0.5}),
            json.dumps({"ppo_epochs": True}),
            json.dumps({"rollout_len": "32"}),
            json.dumps({"gamma": 1.5}),
            json.dumps({"reward_death_penalty": -0.1}),
        )
    ):
        with pytest.raises(ValueError):
            prepare_v3_ppo_packet(
                tmp_path / f"pkt_bad_recipe_{idx_bad}",
                commit="HEAD",
                mode="train",
                gpu="L4",
                recipe_json=bad_json,
                allow_missing_canary_receipt=True,
                output_dir=tmp_path / f"out_bad_recipe_{idx_bad}",
                allow_unmerged_commit=True,
            )

    # 5. Prepare a tiny CPU packet, resolve its manifest env via `resolve_effective_train_config`,
    #    and verify `prepare` and `V3_PPO_RECIPE` report the exact same overrides and record them
    #    in checkpoints and `train_result.json`
    tiny_base_cfg = V3PPOConfig(
        game_config=GameV3Config(arena_radius=300.0, num_snakes=6),
        num_envs=2,
        learner_slots=4,
        net_size="S",
        amp=False,
        use_channels_last=False,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=4,
        total_updates=2,
        checkpoint_every_updates=1,
        eval_every_minutes=0.0,
        eval_every_updates=0,
    )
    capsys.readouterr()
    tiny_pkt_summary = prepare_v3_ppo_packet(
        tmp_path / "pkt_tiny_recipe_job",
        commit="HEAD",
        mode="train",
        gpu="L4",
        ppo_config=tiny_base_cfg,
        explicit_total_updates=True,
        recipe_json=json.dumps(
            {
                "reward_alive_length_coef": 0.0,
                "reward_death_penalty": 2.5,
                "gamma": 0.995,
            }
        ),
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "remote_tiny_recipe_job",
        allow_unmerged_commit=True,
    )
    prep_tiny_err = capsys.readouterr().err
    expected_tiny_ov_str = (
        "gamma=0.995, minibatch_size=4, ppo_epochs=1, reward_alive_length_coef=0.0, "
        "reward_death_penalty=2.5, rollout_len=4"
    )
    assert f"recipe overrides: {expected_tiny_ov_str}" in prep_tiny_err

    mf_tiny = json.loads(Path(tiny_pkt_summary["manifest_path"]).read_text(encoding="utf-8"))
    run_out = tmp_path / "tiny_recipe_job_out"
    eff_tiny = resolve_effective_train_config(
        build_train_arg_parser().parse_args(
            [
                "--output-dir",
                str(run_out),
                "--work-dir",
                str(tmp_path / "tiny_recipe_job_work"),
                "--input-dir",
                str(tmp_path / "pkt_tiny_recipe_job" / "inputs"),
                "--ledger",
                str(tmp_path / "pkt_tiny_recipe_job" / "inputs" / "verified-ledger.json"),
            ]
        ),
        environ={**mf_tiny["env"], "SNAKE_DQN_DEVICE": "cpu"},
    )
    assert eff_tiny.recipe_overrides == tiny_pkt_summary["recipe_overrides"]

    capsys.readouterr()
    res_tiny = run_training_job(
        cfg=eff_tiny.cfg,
        output_dir=eff_tiny.output_dir,
        work_dir=eff_tiny.work_dir,
        device=eff_tiny.device,
        input_dir=eff_tiny.input_dir,
        ledger_path=eff_tiny.ledger_path,
        allow_recipe_change=eff_tiny.allow_recipe_change,
    )
    job_stdout = capsys.readouterr().out
    recipe_log_lines = [ln for ln in job_stdout.splitlines() if ln.startswith("V3_PPO_RECIPE ")]
    assert len(recipe_log_lines) == 1
    assert f'overrides="{expected_tiny_ov_str}"' in recipe_log_lines[0]
    assert res_tiny["status"] == "PASS"
    assert res_tiny["config"]["reward_alive_length_coef"] == 0.0
    assert res_tiny["config"]["reward_death_penalty"] == 2.5
    assert res_tiny["config"]["gamma"] == 0.995

    import torch as _torch

    saved_light = _torch.load(
        run_out / "checkpoints" / "light_u0000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved_light["config"]["reward_alive_length_coef"] == 0.0
    assert saved_light["config"]["reward_death_penalty"] == 2.5
    assert saved_light["config"]["gamma"] == 0.995


def test_no_recipe_json_packets_byte_for_byte_identical_vs_d11d603d(tmp_path: Path) -> None:
    """Prove omitting `--recipe-json` produces byte-for-byte identical packets vs `d11d603d`."""
    baseline_root = _materialize_baseline_root(tmp_path, "d11d603d")
    byte_script = tmp_path / "byte_check_d11d603d.py"
    byte_script.write_text(
        """from __future__ import annotations
import base64, json, shutil, sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
shared_pkt_root = Path(sys.argv[2]).resolve()
out_json = Path(sys.argv[3]).resolve()
git_repo = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(repo_root))

import research.v3_ppo.prepare as prep_mod
prep_mod.ROOT = git_repo

if shared_pkt_root.exists():
    shutil.rmtree(shared_pkt_root)
shared_pkt_root.mkdir(parents=True)

cases = {
    "canary_l4": dict(mode="canary", gpu="L4"),
    "ruler_a100": dict(mode="ruler", gpu="A100", allow_missing_canary_receipt=True),
    "train_l4_none": dict(mode="train", gpu="L4", hours=4.0, allow_missing_canary_receipt=True),
    "train_a100_final": dict(
        mode="train", gpu="A100", num_envs=256, net_size="M", hours=8.0,
        full_exports="final", allow_missing_canary_receipt=True,
    ),
    "train_l4_periodic": dict(
        mode="train", gpu="L4", hours=4.0, full_exports="periodic",
        allow_missing_canary_receipt=True,
    ),
}

snapshots = {}
for name, kw in cases.items():
    pkt_dir = shared_pkt_root / f"pkt_{name}"
    out_dir = shared_pkt_root / f"out_{name}"
    summary = prep_mod.prepare_v3_ppo_packet(
        pkt_dir,
        commit="d11d603d",
        output_dir=out_dir,
        allow_unmerged_commit=True,
        **kw,
    )
    files_b64 = {}
    for p in sorted(pkt_dir.rglob("*")):
        if p.is_file():
            files_b64[p.relative_to(pkt_dir).as_posix()] = base64.b64encode(
                p.read_bytes()
            ).decode("ascii")
    snapshots[name] = {
        "summary_json": json.dumps(summary, indent=2, sort_keys=True),
        "files_b64": files_b64,
    }

out_json.write_text(json.dumps(snapshots, sort_keys=True), encoding="utf-8")
""",
        encoding="utf-8",
    )

    sub_env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    shared_root = tmp_path / "shared_d11d603d_pkts"
    before_json = tmp_path / "snap_d11d603d.json"
    after_json = tmp_path / "snap_head_d11d603d.json"
    subprocess.run(
        [
            sys.executable,
            str(byte_script),
            str(baseline_root),
            str(shared_root),
            str(before_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(baseline_root)},
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(byte_script),
            str(ROOT),
            str(shared_root),
            str(after_json),
            str(ROOT),
        ],
        env={**sub_env, "PYTHONPATH": str(ROOT)},
        check=True,
    )
    snap_before = json.loads(before_json.read_text(encoding="utf-8"))
    snap_after = json.loads(after_json.read_text(encoding="utf-8"))
    assert snap_after.keys() == snap_before.keys()
    # Canary and ruler packets remain 100% byte-for-byte identical vs d11d603d!
    for unchanged_case in ("canary_l4", "ruler_a100"):
        assert (
            snap_after[unchanged_case]["files_b64"] == snap_before[unchanged_case]["files_b64"]
        ), f"Packet file bytes changed vs d11d603d for {unchanged_case}"
        assert (
            snap_after[unchanged_case]["summary_json"]
            == snap_before[unchanged_case]["summary_json"]
        ), f"Summary JSON changed vs d11d603d for {unchanged_case}"

    # Train packets vs d11d603d: every non-manifest file is byte-identical, and manifest differs
    # ONLY in Item 4's raised 330 s setup allowance (-210 s max_wall_seconds + staged_input/reserve
    # metadata) and dropping inert `max_periodic_full_exports` from `none`-mode V3_PPO_CONFIG_JSON.
    import base64 as _b64

    for train_case in ("train_l4_none", "train_a100_final", "train_l4_periodic"):
        b_files = snap_before[train_case]["files_b64"]
        a_files = snap_after[train_case]["files_b64"]
        assert set(a_files.keys()) == set(b_files.keys())
        for rel_p in sorted(set(a_files.keys()) - {"dispatch-manifest.json"}):
            assert a_files[rel_p] == b_files[rel_p]
        mf_b = json.loads(_b64.b64decode(b_files["dispatch-manifest.json"]).decode("utf-8"))
        mf_a = json.loads(_b64.b64decode(a_files["dispatch-manifest.json"]).decode("utf-8"))
        assert {k: v for k, v in mf_a.items() if k not in {"env", "technical_envelope"}} == {
            k: v for k, v in mf_b.items() if k not in {"env", "technical_envelope"}
        }
        assert int(mf_a["env"]["V3_PPO_MAX_WALL_SECONDS"]) == (
            int(mf_b["env"]["V3_PPO_MAX_WALL_SECONDS"]) - 210
        )
        cfg_b = json.loads(mf_b["env"]["V3_PPO_CONFIG_JSON"])
        cfg_a = json.loads(mf_a["env"]["V3_PPO_CONFIG_JSON"])
        if train_case == "train_l4_none":
            assert cfg_a == {k: v for k, v in cfg_b.items() if k != "max_periodic_full_exports"}
        else:
            assert cfg_a == cfg_b

    # And on HEAD, omitting `--recipe-json` vs passing `--recipe-json "{}"` is 100% byte-identical!
    pkt_no_r = tmp_path / "shared_recipe_eq" / "pkt"
    out_no_r = tmp_path / "shared_recipe_eq" / "out"
    s_no_r = prepare_v3_ppo_packet(
        pkt_no_r,
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=4.0,
        allow_missing_canary_receipt=True,
        output_dir=out_no_r,
        allow_unmerged_commit=True,
    )
    bytes_no_r = {
        p.relative_to(pkt_no_r).as_posix(): p.read_bytes()
        for p in sorted(pkt_no_r.rglob("*"))
        if p.is_file()
    }
    shutil.rmtree(pkt_no_r.parent)
    s_empty_r = prepare_v3_ppo_packet(
        pkt_no_r,
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=4.0,
        recipe_json="{}",
        allow_missing_canary_receipt=True,
        output_dir=out_no_r,
        allow_unmerged_commit=True,
    )
    bytes_empty_r = {
        p.relative_to(pkt_no_r).as_posix(): p.read_bytes()
        for p in sorted(pkt_no_r.rglob("*"))
        if p.is_file()
    }
    assert bytes_empty_r == bytes_no_r
    assert s_empty_r == s_no_r


def test_prepare_minibatch_remainder_warning_and_25pct_drop_refusal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`prepare` warns (1 stderr line + summary) on mb remainder <= 25% and refuses > 25%."""
    # 1. Exact divisibility (98,304 % 2048 == 0): no warning and no summary key
    capsys.readouterr()
    s_div = prepare_v3_ppo_packet(
        tmp_path / "pkt_mb_div",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=4.0,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_mb_div",
        allow_unmerged_commit=True,
    )
    err_div = capsys.readouterr().err
    assert "V3_PPO_MINIBATCH_REMAINDER" not in err_div
    assert "minibatch_remainder" not in s_div

    # 2. Non-divisible with <= 25% dropped (98,304 % 20,000 == 18,304 -> 18.62% dropped)
    capsys.readouterr()
    s_warn = prepare_v3_ppo_packet(
        tmp_path / "pkt_mb_warn",
        commit="HEAD",
        mode="train",
        gpu="A100",
        num_envs=128,
        hours=4.0,
        recipe_json=json.dumps({"minibatch_size": 20000}),
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_mb_warn",
        allow_unmerged_commit=True,
    )
    err_warn = capsys.readouterr().err
    warn_lines = [ln for ln in err_warn.splitlines() if "V3_PPO_MINIBATCH_REMAINDER" in ln]
    assert len(warn_lines) == 1
    assert "dropping 18304/98304 rows (18.62%) per PPO epoch" in warn_lines[0]
    assert s_warn["minibatch_remainder"] == {
        "rows_per_update": 98304,
        "minibatch_size": 20000,
        "dropped_rows_per_epoch": 18304,
        "dropped_fraction": round(18304 / 98304, 6),
        "dropped_percent": 18.62,
        "padded_rows_per_epoch": 0,
    }

    # 3. Non-divisible with > 25% dropped (98,304 % 32,769 == 32,766 -> 33.33% > 25%): refused!
    refused_pkt = tmp_path / "pkt_mb_refused"
    with pytest.raises(
        ValueError,
        match=r"32766/98304 rows \(33\.33%\) per PPO epoch, exceeding the 25% cap",
    ):
        prepare_v3_ppo_packet(
            refused_pkt,
            commit="HEAD",
            mode="train",
            gpu="A100",
            num_envs=128,
            hours=4.0,
            recipe_json=json.dumps({"minibatch_size": 32769}),
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_mb_refused",
            allow_unmerged_commit=True,
        )
    assert not refused_pkt.exists()
