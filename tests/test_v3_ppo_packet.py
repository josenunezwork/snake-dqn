"""Preflight tests for `research.v3_ppo.prepare`, `ruler`, `job.sh`, VRAM & export-cap refusals."""

from __future__ import annotations

import json
import shutil
import subprocess
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
from research.v3_ppo.config import V3PPOConfig, estimate_training_memory
from research.v3_ppo.ppo import (
    V3PPOTrainer,
)
from research.v3_ppo.ppo import main as ppo_main
from research.v3_ppo.ppo import (
    publish_checkpoint_to_output,
    verify_full_checkpoint_set,
)
from research.v3_ppo.prepare import (
    ROOT,
)
from research.v3_ppo.prepare import main as prepare_main
from research.v3_ppo.prepare import (
    prepare_v3_ppo_packet,
)
from research.v3_ppo.ruler import (
    CANARY_REQUIRED_EXPORTS,
    DEFAULT_RULER_ROLLOUT_LEN,
    DEFAULT_RULER_TIMED_UPDATES,
    DEFAULT_RULER_WARMUP_UPDATES,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
)
from research.v3_ppo.ruler import main as ruler_main
from research.v3_ppo.ruler import (
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
        hours=2.0,
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
    """Verify `prepare` enforces single-world shard, 2 GiB supervisor, and session export caps."""
    # 1. Shard smaller than 1 single worst-case world (~30.22 MiB at R=600, N=24) -> refused
    tiny_shard_cfg = V3PPOConfig(num_envs=8, checkpoint_shard_bytes=4 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds checkpoint_shard_bytes"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_bad_shard",
            commit="HEAD",
            mode="train",
            gpu="L4",
            ppo_config=tiny_shard_cfg,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_bad_shard",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_bad_shard").exists()

    # 2. Shard cap whose 1.25x file envelope exceeds the 2.00 GiB supervisor per-file ceiling
    huge_shard_cfg = V3PPOConfig(num_envs=16, checkpoint_shard_bytes=1800 * 1024 * 1024)
    with pytest.raises(ValueError, match="supervisor 2.00 GiB per-file limit"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_huge_shard",
            commit="HEAD",
            mode="train",
            gpu="L4",
            ppo_config=huge_shard_cfg,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_huge_shard",
            allow_unmerged_commit=True,
        )

    # 3. Total session export volume exceeding GPU session cap (`12.00 GiB` on L4):
    #    (3a) Excessive `max_periodic_full_exports` (`6` periodic + `1` stop = `7` FULLs > `12 GiB`)
    bloated_full_cfg = V3PPOConfig(num_envs=128, max_periodic_full_exports=6)
    with pytest.raises(ValueError, match="exceeds L4 session export cap"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_bloated_session",
            commit="HEAD",
            mode="train",
            gpu="L4",
            hours=8.0,
            ppo_config=bloated_full_cfg,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_bloated_session",
            allow_unmerged_commit=True,
        )

    #    (3b) Minor 4: Update-based checkpoint setting (`checkpoint_every_updates=1`) exceeding cap
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

    # 4. Explicit high `measured_packed_bytes_per_world` pushing E=128 over 12.00 GiB L4 cap
    with pytest.raises(ValueError, match="exceeds L4 session export cap"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_high_measured",
            commit="HEAD",
            mode="train",
            gpu="L4",
            num_envs=128,
            measured_packed_bytes_per_world=8 * 1024 * 1024,  # 8 MiB * 4.8x = 38.4 MiB/world
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_high_measured",
            allow_unmerged_commit=True,
        )

    # 5. Verify 8 h and 24 h sessions fit cleanly on L4 (E=128) and A100 (E=256) for both S and M
    for gpu_kind, n_envs in (("L4", 128), ("A100", 256)):
        for h_val in (8.0, 24.0):
            for net_sz in ("S", "M"):
                slug = f"{gpu_kind.lower()}_{int(h_val)}h_{net_sz.lower()}"
                ok_info = prepare_v3_ppo_packet(
                    tmp_path / f"pkt_{slug}",
                    commit="HEAD",
                    mode="train",
                    gpu=gpu_kind,
                    hours=h_val,
                    num_envs=n_envs,
                    net_size=net_sz,
                    allow_missing_canary_receipt=True,
                    output_dir=tmp_path / f"out_{slug}",
                    allow_unmerged_commit=True,
                )
                caps = ok_info["checkpoint_export_caps"]
                assert caps["total_session_fits_cap"] is True
                assert caps["max_light_exports"] == (17 if h_val == 8.0 else 49)
                assert caps["max_full_exports"] == 4
                mf_ok = json.loads(Path(ok_info["manifest_path"]).read_text(encoding="utf-8"))
                assert int(mf_ok["env"]["V3_PPO_MAX_SESSION_EXPORT_BYTES"]) == int(
                    caps["session_total_export_cap_bytes"]
                )


def test_ruler_defaults_warmup_timed_split_precheck_and_single_slot_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ruler default T=32, warmup>=1, timed>=2, collect/learn split, & VRAM precheck."""
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
    # Minor 6: ALL warmup + timed `train_step` calls ran with `stage_timing=False` (2 * 3 = 6),
    # and each succeeded cell ran 1 separate `collect_rollout(..., stage_timing=True)` pass (= 2)!
    assert train_step_stage_timings == [False] * 6
    assert len(synced_rollout_calls) == 2

    # M2-r2: Verify `run_canary_suite` without override invokes `evaluate_policy` at DEFAULT config
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

    oom_cell = by_id["E4_L2_S"]
    assert oom_cell["oom"] is True
    assert oom_cell["precheck_oom"] is True
    assert oom_cell["integrated"] is None


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
    """B1-r3: Poll `colab_cli.runtime._exports` on publish + after cleanup (`st_nlink == 1`)."""
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
    """m3-r3 & m5-r3: Canary default eval holds GPU-default train sim resident & propagates wall."""
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


def test_prepare_and_job_sh_effective_configs_match_e2865f93_baseline(tmp_path: Path) -> None:
    """Prove canary/ruler/train packets produce identical effective runtime configs vs e2865f93."""
    import os
    import sys

    baseline_root = tmp_path / "baseline_e2865f93"
    baseline_root.mkdir(parents=True)
    # Symlink shared read-only packages and extract e2865f93's `research/v3_ppo/`
    (baseline_root / "src").symlink_to(ROOT / "src")
    (baseline_root / "tools").symlink_to(ROOT / "tools")
    (baseline_root / "research").mkdir()
    for src_p in (ROOT / "research").iterdir():
        if src_p.name != "v3_ppo":
            (baseline_root / "research" / src_p.name).symlink_to(src_p)
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "e2865f93", "research/v3_ppo"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-xf", "-", "-C", str(baseline_root)],
        input=archive.stdout,
        check=True,
    )

    probe_script = tmp_path / "probe_effective_packets.py"
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
    assert after == before
