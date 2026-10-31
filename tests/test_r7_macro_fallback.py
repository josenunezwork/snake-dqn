"""Unit tests for R7 macro-option fallback, split hygiene, bit-exact 2-ply search, and resume."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from research.gpu_sim_spike import r2_cache
from research.r7_macro_fallback import ALLOCATION_NAME, NAMESPACE, R7_KILL_THRESHOLDS
from research.r7_macro_fallback import uploads as r7_uploads
from research.r7_macro_fallback.options import (
    WorldGridState,
    evaluate_all_options,
    run_mac_timing_probe,
)
from research.r7_macro_fallback.prepare import ROOT, r7_source_paths
from research.r7_macro_fallback.readout import (
    collect_verified_batch_units,
    evaluate_r7_readout,
)
from research.r7_macro_fallback.search import (
    assert_train_only_roots,
    batch_unit_id,
    evaluate_root_batch,
    publish_verified_ledger,
    run_search_from_files,
    run_two_ply_macro_rollouts,
)
from research.sandbox_a_resident.ct_mechanics import (
    CTDeviceSim,
    CTOpponentRuntime,
    derive_branch_keys,
)
from research.sandbox_a_resident.ct_snapshots import capture_world_snapshot
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    training_rosters,
)
from src.model.apex_network import ApexNetwork
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
from src.simd_env.batch_sim import BatchSimConfig
from tools.ops.package_spec import (
    Input,
    PackageRefused,
    publish_json,
    read_ledger,
    stage_inputs,
    verify_immutable,
)


def _make_grid_state(
    *,
    hero_length: int,
    hero_seg_count: int,
    hero_body_xy: list[tuple[int, int]],
    direction: int = 1,
    boost_frames: int = 0,
    min_boost_length: int = 10,
    boost_cost_frames: int = 10,
) -> WorldGridState:
    alive = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    length = np.array([hero_length, 0, 0, 0, 0, 0], dtype=np.int32)
    seg_count = np.array([hero_seg_count, 0, 0, 0, 0, 0], dtype=np.int32)
    dirs = np.array([direction, 1, 1, 1, 1, 1], dtype=np.int32)
    bframes = np.array([boost_frames, 0, 0, 0, 0, 0], dtype=np.int32)
    offsets = np.array(
        [
            0,
            len(hero_body_xy),
            len(hero_body_xy),
            len(hero_body_xy),
            len(hero_body_xy),
            len(hero_body_xy),
            len(hero_body_xy),
        ],
        dtype=np.int32,
    )
    body_xy = np.asarray(hero_body_xy, dtype=np.int16)
    food_xy = np.array([[25, 20]], dtype=np.int16)
    food_corpse = np.array([1], dtype=np.uint8)
    return WorldGridState(
        grid_h=40,
        grid_w=40,
        min_boost_length=min_boost_length,
        boost_cost_frames=boost_cost_frames,
        alive=alive,
        length=length,
        seg_count=seg_count,
        direction=dirs,
        boost_frames=bframes,
        body_offsets=offsets,
        body_xy=body_xy,
        food_xy=food_xy,
        food_corpse=food_corpse,
    )


def test_r7_pending_growth_and_boost_burn_legality() -> None:
    # Hero facing East (direction=1) at (10, 10), curled so tail is at (10, 9)
    # (Action 0 = Turn Left = North). Body ordered head -> tail:
    curled_body = [(10, 10), (9, 10), (9, 9), (10, 9)]

    # Case A: seg_count == length (4 == 4) -> tail at (10, 9) vacates on normal move,
    # so Action 0 (North -> (10, 9)) is safe.
    st_vacates = _make_grid_state(
        hero_length=4,
        hero_seg_count=4,
        hero_body_xy=curled_body,
        direction=1,
        min_boost_length=10,
    )
    _, scores_vac, _ = evaluate_all_options(st_vacates)
    assert np.all(scores_vac[:, 0] > -1e17)
    # Length 4 < min_boost_length 10 -> all boost actions 3..5 must be illegal!
    assert np.all(scores_vac[:, 3:] <= -1e18)

    # Case B: pending growth seg_count < length (4 < 6) -> tail at (10, 9) DOES NOT vacate!
    # Action 0 (North -> (10,9)) must be flagged fatal across all 6 options!
    st_pending = _make_grid_state(
        hero_length=6,
        hero_seg_count=4,
        hero_body_xy=curled_body,
        direction=1,
        min_boost_length=10,
    )
    chosen_pend, scores_pend, _ = evaluate_all_options(st_pending)
    assert np.all(scores_pend[:, 0] <= -1e18)
    assert all(int(a) != 0 for a in chosen_pend.tolist())

    # Case C: boost-burn boundary (length >= min_boost_length and boost_frames + 1 >= cost)
    long_body = [(20 - k, 20) for k in range(12)]
    st_boost_burn = _make_grid_state(
        hero_length=12,
        hero_seg_count=12,
        hero_body_xy=long_body,
        direction=1,
        boost_frames=9,
        min_boost_length=10,
        boost_cost_frames=10,
    )
    chosen_bb, scores_bb, _ = evaluate_all_options(st_boost_burn)
    # O0/O1 are strictly normal-speed options; O2..O5 allow legal boost (action 4).
    assert np.all(scores_bb[:2, 4] <= -1e18)
    assert np.all(scores_bb[2:, 4] > -1e17)
    assert all(0 <= int(a) < 6 for a in chosen_bb.tolist())

    # Timing probe smoke
    probe = run_mac_timing_probe([st_boost_burn], ticks=50, warmup_ticks=5)
    assert probe["spacious_interrupt_enabled"] is False
    assert probe["ticks"] == 50


def test_r7_refuses_assessment_and_non_all_die_roots() -> None:
    valid = {"root_uid": "r0", "partition": "train", "is_assessment": False, "all_die_sel0": True}
    assert_train_only_roots([valid])

    with pytest.raises(ValueError, match="Refusing non-train root"):
        assert_train_only_roots([{**valid, "partition": "assessment"}])

    with pytest.raises(ValueError, match="Refusing non-train root"):
        assert_train_only_roots([{**valid, "is_assessment": True}])

    with pytest.raises(ValueError, match="all_die_sel0 is False"):
        assert_train_only_roots([{**valid, "all_die_sel0": False}])


def test_r7_split_ply1_ply2_matches_continuous_rollout_and_ledger_skip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    dev = torch.device("cpu")
    b8_net = Ego2sNet(local_channels=7, n_scalars=60).eval()
    v61_net = ApexNetwork(input_size=61, hidden_size=512, output_size=6).eval()
    ckpt_nets = {FRP3_S12_SHA256: v61_net, CHAMPION_SHA256: v61_net}

    cfg = BatchSimConfig(
        num_envs=3,
        num_snakes=6,
        game_width=240,
        game_height=240,
        initial_food=10,
        max_food=18,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    rosters_all = training_rosters(6)
    sim = CTDeviceSim(
        cfg, device=dev, train_mode=False, allow_respawn=True, move_backend="reference"
    )
    r2_cache.enable(sim)
    sim.rng_key.copy_(torch.tensor([101, 202, 303], dtype=torch.long))
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    opp = CTOpponentRuntime(sim, rosters_all[:3], b8_network=b8_net, checkpoint_networks=ckpt_nets)
    sim.prepare()

    root_records: list[dict] = []
    for b_idx in range(3):
        snap = capture_world_snapshot(sim, opp, b_idx, world_id=b_idx, life_id=0, tick=10)
        root_records.append(
            {
                "root_index": b_idx,
                "batch_index": b_idx,
                "root_uid": f"synth-train-root-{b_idx}",
                "partition": "train",
                "is_assessment": False,
                "kind": "death_lookback",
                "world_id": b_idx,
                "mix": rosters_all[b_idx].mix,
                "life_id": 0,
                "tick": 10,
                "lookback": 15,
                "hero_length": 850,
                "length_bin": "[800,1000)",
                "all_die_sel0": True,
                "exact_stop_action_sel0": 1,
                "exact_stop_s_sel0": 0.0,
                "exact_stop_s_score": 0.0,
                "exact_stop_m_score": 120.0,
                "snapshot": snap,
            }
        )

    batch_eval = evaluate_root_batch(
        root_records,
        rosters_by_world=rosters_all,
        horizon=10,
        macro_ticks=3,
        top_m_ply1=3,
        selection_suffix=0,
        scoring_suffixes=[4, 5, 6, 7],
        compaction_interval=2,
        max_slab_envs=16,
        canary_only=False,
        base_cfg=cfg,
        device=dev,
        b8_network=b8_net,
        checkpoint_networks=ckpt_nets,
    )

    # Bit-exactness check: continuous 10-step rollout of frozen (o1_star, o2_star) on suffix 0
    # must match the 2-stage split search (Ply 1 snapshot -> Ply 2 + B8 continuation) on suffix 0!
    bkeys0 = [derive_branch_keys(r["root_uid"], 0) for r in root_records]
    cont = run_two_ply_macro_rollouts(
        [r["snapshot"] for r in root_records],
        [rosters_all[r["world_id"]] for r in root_records],
        [r["o1_star"] for r in batch_eval["roots"]],
        [r["o2_star"] for r in batch_eval["roots"]],
        sim_rng_keys=[bk.sim_rng_key for bk in bkeys0],
        opp_rng_keys=[bk.opp_rng_key for bk in bkeys0],
        horizon=10,
        macro_ticks=3,
        compaction_interval=2,
        base_cfg=cfg,
        device=dev,
        b8_network=b8_net,
        checkpoint_networks=ckpt_nets,
    )
    for i, r_out in enumerate(batch_eval["roots"]):
        if r_out["retained_ply1_options"]:
            assert float(cont["survive"][i]) == pytest.approx(float(r_out["sel0_survive"]), abs=0.0)
            assert float(cont["mean_mass"][i]) == pytest.approx(
                float(r_out["sel0_mean_mass"]), abs=1e-4
            )
            assert int(cont["ply1_nonfatal_ticks"][i]) == int(r_out["sel0_ply1_nonfatal_ticks"])
            assert int(cont["two_ply_nonfatal_ticks"][i]) == int(
                r_out["sel0_two_ply_nonfatal_ticks"]
            )

    # End-to-end file runner + resume ledger skip test
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    save_ego2s_checkpoint(str(in_dir / "b8-s3.pth"), b8_net)
    v61_payload = {
        "obs_spec": "vector61",
        "input_size": 61,
        "hidden_size": 512,
        "output_size": 6,
        "state_dict": v61_net.state_dict(),
    }
    torch.save(v61_payload, in_dir / "frp3-s12.pth")
    torch.save(v61_payload, in_dir / "champion.pth")

    serializable_roots = [{**r, "snapshot": r["snapshot"].to_state_dict()} for r in root_records]
    roots_buf = io.BytesIO()
    torch.save(
        {
            "schema": "t34-r7-train-all-die-roots/v1",
            "assessment_roots_used": 0,
            "total_worlds": 6,
            "max_body_capacity": 256,
            "num_batches": 3,
            "num_roots": 3,
            "roots": serializable_roots,
        },
        roots_buf,
    )
    (in_dir / "train-all-die-roots.pt.xz").write_bytes(
        lzma.compress(roots_buf.getvalue(), preset=6)
    )
    cfg_path = in_dir / "test-config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "schema": "t34-r7-search-config/v1",
                "canary_only": False,
                "verify_checkpoint_hashes": False,
                "roots_file": "train-all-die-roots.pt.xz",
                "batch_indices": [0, 1, 2],
                "max_roots_per_batch": None,
                "horizon": 10,
                "macro_ticks": 3,
                "top_m_ply1": 2,
                "selection_suffix": 0,
                "scoring_suffixes": [4, 5, 6, 7],
                "compaction_interval": 2,
                "max_slab_envs": 8,
            }
        ),
        encoding="utf-8",
    )

    ledger0 = in_dir / "ledger-0.json"
    publish_verified_ledger(ledger0, {})
    attempt1_out = tmp_path / "attempt-1" / "exports" / "r7"
    s1 = run_search_from_files(
        input_dir=in_dir,
        config_path=cfg_path,
        ledger_path=ledger0,
        output_dir=attempt1_out,
        device_str="cpu",
    )
    assert s1["computed_unit_ids"] == [batch_unit_id(0), batch_unit_id(1), batch_unit_id(2)]
    assert s1["skipped_unit_ids"] == []

    # Simulate partial attempt-1 (remove batch-0001 and seal stopped receipt) -> resume attempt-2
    for suffix in ("", ".sha256", ".complete"):
        (attempt1_out / "units" / f"batch-0001.json{suffix}").unlink()
    (tmp_path / "attempt-1" / "receipt.json").write_text(
        json.dumps({"status": "INCOMPLETE", "stop_verified": True, "cleanup_error": None}),
        encoding="utf-8",
    )

    entries1, _, _ = collect_verified_batch_units([tmp_path / "attempt-1"])
    assert sorted(entries1) == [0, 2]

    ledger1 = in_dir / "ledger-1.json"
    publish_verified_ledger(ledger1, entries1)
    attempt2_out = tmp_path / "attempt-2" / "exports" / "r7"
    s2 = run_search_from_files(
        input_dir=in_dir,
        config_path=cfg_path,
        ledger_path=ledger1,
        output_dir=attempt2_out,
        device_str="cpu",
    )
    assert s2["skipped_unit_ids"] == [batch_unit_id(0), batch_unit_id(2)]
    assert s2["computed_unit_ids"] == [batch_unit_id(1)]
    (tmp_path / "attempt-2" / "receipt.json").write_text(
        json.dumps({"status": "PASS", "stop_verified": True, "cleanup_error": None}),
        encoding="utf-8",
    )

    merged_entries, merged_payloads, _ = collect_verified_batch_units(
        [tmp_path / "attempt-1", tmp_path / "attempt-2"]
    )
    assert sorted(merged_entries) == [0, 1, 2]
    assert len(merged_payloads) == 3

    # Foreign namespace ledger refusal
    bad_ledger = tmp_path / "bad-ledger.json"
    bad_payload = json.loads(ledger1.read_text(encoding="utf-8"))
    bad_payload["namespace"] = "a100-dev/foreign-study"
    publish_json(bad_ledger, bad_payload)
    with pytest.raises(PackageRefused, match="foreign or malformed verified ledger"):
        read_ledger(bad_ledger, NAMESPACE, ALLOCATION_NAME)


def test_r7_readout_kill_thresholds_and_zero_ordinary_flips() -> None:
    assert R7_KILL_THRESHOLDS["min_independent_suffix_H300_rescue_L800_all_die"] == 0.25
    assert R7_KILL_THRESHOLDS["min_total_death_root_rescue_lift_over_exact_stop_pp"] == 10.0
    assert R7_KILL_THRESHOLDS["max_newly_caused_ordinary_lethal_flips"] == 0
    assert R7_KILL_THRESHOLDS["max_mac_1thread_tracker_p95_ms_per_tick"] == 1.0
    assert R7_KILL_THRESHOLDS["min_realized_nonfatal_macro_duration_ticks"] == 10.0

    baseline = {
        "schema": "t34-r7-train-exact-stop-baseline/v1",
        "assessment_roots_used": 0,
        "train_roots": [
            {
                "root_uid": "d0",
                "partition": "train",
                "kind": "death_lookback",
                "mix": "frozen",
                "hero_length": 900,
                "all_die_sel0": True,
                "exact_stop_s_score": 0.0,
                "exact_stop_m_score": 300.0,
                "ct4_a_star_s_score": 0.0,
            },
            {
                "root_uid": "d1",
                "partition": "train",
                "kind": "death_lookback",
                "mix": "mixed",
                "hero_length": 920,
                "all_die_sel0": False,
                "exact_stop_s_score": 1.0,
                "exact_stop_m_score": 920.0,
                "ct4_a_star_s_score": 1.0,
            },
            {
                "root_uid": "o0",
                "partition": "train",
                "kind": "ordinary",
                "mix": "scripted",
                "hero_length": 880,
                "all_die_sel0": False,
                "exact_stop_s_score": 1.0,
                "exact_stop_m_score": 880.0,
                "ct4_a_star_s_score": 1.0,
            },
        ],
    }
    mac_ok = {
        "schema": "t34-r7-mac-tracker-timing/v1",
        "p95_ms_per_tick": 0.22,
    }
    uid0 = batch_unit_id(0)
    units_pass = {
        uid0: {
            "roots": [
                {
                    "root_uid": "d0",
                    "partition": "train",
                    "macro_triggered": True,
                    "o1_name": "O0_delay_coil",
                    "o2_name": "O2_open_voronoi",
                    "policy_s_score": 0.50,
                    "policy_m_score": 850.0,
                    "indep_mean_ply1_nonfatal_ticks": 15.0,
                    "indep_mean_two_ply_nonfatal_ticks": 30.0,
                }
            ]
        }
    }
    rep_pass = evaluate_r7_readout(
        train_baseline=baseline,
        unit_payloads=units_pass,
        mac_timing=mac_ok,
        expected_batches=1,
    )
    assert rep_pass["verdict"] == "PASS"
    assert rep_pass["metrics"]["newly_caused_ordinary_lethal_flips"] == 0
    assert rep_pass["metrics"]["l800_all_die_indep_rescue_rate"] == pytest.approx(0.50)
    assert rep_pass["metrics"]["total_death_root_lift_l800"] == pytest.approx(0.25)

    # Low rescue (<25%) -> KILL
    units_low = {
        uid0: {
            "roots": [
                {
                    **units_pass[uid0]["roots"][0],
                    "policy_s_score": 0.15,
                }
            ]
        }
    }
    rep_kill = evaluate_r7_readout(
        train_baseline=baseline,
        unit_payloads=units_low,
        mac_timing=mac_ok,
        expected_batches=1,
    )
    assert rep_kill["verdict"] == "KILL"
    assert "passes_l800_all_die_rescue_ge_25pct" in rep_kill["failed_bars"]


def test_r7_repack_upload_budget_canary_and_main(tmp_path: Path) -> None:
    assert r7_uploads.MAX_TRANSFERS == 8
    assert r7_uploads.SECONDS_PER_TRANSFER == 6.0
    assert r7_uploads.BYTES_PER_SECOND == 800_000
    assert r7_uploads.MAX_ESTIMATED_TRANSFER_SECONDS == 60.0
    assert r7_uploads.PART_BYTES == 2 * 1024**2

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Deterministic 520 KB pseudo-random payload (>= real 506 KB train-all-die-roots.pt.xz).
    rng = np.random.default_rng(3407)
    roots_file = raw_dir / "train-all-die-roots.pt.xz"
    roots_file.write_bytes(rng.bytes(520_000))
    canary_cfg = raw_dir / "canary-config.json"
    canary_cfg.write_text('{"schema":"t34-r7-search-config/v1","canary_only":true}\n')
    main_cfg = raw_dir / "main-config.json"
    main_cfg.write_text('{"schema":"t34-r7-search-config/v1","canary_only":false}\n')
    ledger = raw_dir / "verified-ledger.json"
    publish_verified_ledger(ledger, {})
    public_catalog = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"

    staged_rows = stage_inputs(
        tmp_path / "staged",
        [
            Input(roots_file, "train-all-die-roots.pt.xz"),
            Input(canary_cfg, "canary-config.json"),
            Input(main_cfg, "main-config.json"),
            Input(ledger, "verified-ledger.json"),
            Input(public_catalog, "public-inputs.json"),
        ],
    )
    packed = r7_uploads.pack_inputs(staged_rows, tmp_path / "packed" / r7_uploads.METADATA)
    assert len(packed) == 1
    assert packed[0]["dest"] == r7_uploads.METADATA
    meta_sha = str(packed[0]["sha256"])

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = r7_source_paths(head, repo=ROOT)
    assert "research/bl_upload" not in source_paths
    manifest_inputs = [{k: row[k] for k in ("path", "dest", "sha256")} for row in packed]

    for role, cfg_name in (("canary", "canary-config.json"), ("main", "main-config.json")):
        manifest = {
            "job_id": f"r7-macro-fallback-{role}-test",
            "commit": head,
            "script": "research/r7_macro_fallback/job.sh",
            "device": "cuda",
            "source_paths": source_paths,
            "requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "min_vcpu": 2},
            "timeout_minutes": 30 if role == "canary" else 50,
            "output_dir": str(tmp_path / f"out-{role}"),
            "max_export_bytes": r7_uploads.MAX_FILE_BYTES,
            "inputs": manifest_inputs,
            "required_exports": ["r7/run-summary.json", "r7/skip-proof.json"],
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/r7_macro_fallback/requirements.txt",
                "R7_SOURCE_COMMIT": head,
                "R7_CONFIG_NAME": cfg_name,
                "R7_METADATA_SHA256": meta_sha,
            },
        }
        report = r7_uploads.check_uploads(manifest, repo=ROOT)
        assert report["status"] == "PASS"
        assert report["normal_transfer_count"] == 3
        assert report["transfer_count"] == 4
        assert report["normal_transfer_count"] <= 8
        assert report["transfer_count"] <= 8
        assert report["normal_estimated_transfer_seconds"] <= 60.0
        assert report["estimated_transfer_seconds"] <= 60.0

    # Budget gate must refuse > 8 transfers or > 60 s forecast.
    too_many_inputs = [
        {"path": packed[0]["path"], "dest": f"extra-{idx}.tar.gz", "sha256": meta_sha}
        for idx in range(7)
    ]
    with pytest.raises(ValueError, match="R7 upload transfer forecast refuses offline"):
        r7_uploads.check_uploads({**manifest, "inputs": too_many_inputs}, repo=ROOT)

    # Round-trip unpack verification + tamper refusal.
    runtime_in = tmp_path / "runtime-inputs"
    runtime_in.mkdir()
    shutil.copyfile(packed[0]["path"], runtime_in / r7_uploads.METADATA)
    with pytest.raises(ValueError, match="SHA256 or size mismatch"):
        r7_uploads.stage_metadata(runtime_in, "0" * 64)

    unpacked = r7_uploads.stage_metadata(runtime_in, meta_sha)
    assert unpacked["split_payloads"] == []
    for name in r7_uploads.BASE_PAYLOADS:
        verify_immutable(runtime_in / name)
    assert (runtime_in / "train-all-die-roots.pt.xz").read_bytes() == roots_file.read_bytes()


def test_r7_split_over_2mib_and_slim_three_checkpoint_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rng = np.random.default_rng(99)
    # 2.4 MiB incompressible payload (> 2 MiB) -> must split into 2 parts (2 MiB + 0.4 MiB),
    # while small files + all sidecars remain inside input-metadata.tar.gz.
    big_bytes = rng.bytes(int(2.4 * 1024**2))
    roots_big = raw_dir / "train-all-die-roots.pt.xz"
    roots_big.write_bytes(big_bytes)
    canary_cfg = raw_dir / "canary-config.json"
    canary_cfg.write_text('{"canary":true}\n')
    main_cfg = raw_dir / "main-config.json"
    main_cfg.write_text('{"canary":false}\n')
    ledger = raw_dir / "verified-ledger.json"
    publish_verified_ledger(ledger, {})
    public_catalog = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"

    staged = stage_inputs(
        tmp_path / "staged",
        [
            Input(roots_big, "train-all-die-roots.pt.xz"),
            Input(canary_cfg, "canary-config.json"),
            Input(main_cfg, "main-config.json"),
            Input(ledger, "verified-ledger.json"),
            Input(public_catalog, "public-inputs.json"),
        ],
    )
    packed = r7_uploads.pack_inputs(
        staged,
        tmp_path / "packed" / r7_uploads.METADATA,
        parts_dir=tmp_path / "packed" / "parts",
    )
    assert len(packed) == 3  # 2 split parts + 1 input-metadata.tar.gz (<= 8 transfers!)
    for row in packed:
        assert int(row["size_bytes"]) <= r7_uploads.PART_BYTES

    runtime_in = tmp_path / "runtime-split"
    runtime_in.mkdir()
    meta_sha = ""
    for row in packed:
        dest_path = runtime_in / str(row["dest"])
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], dest_path)
        if row["dest"] == r7_uploads.METADATA:
            meta_sha = str(row["sha256"])

    res = r7_uploads.stage_metadata(runtime_in, meta_sha)
    assert res["split_payloads"] == ["train-all-die-roots.pt.xz"]
    verify_immutable(runtime_in / "train-all-die-roots.pt.xz")
    assert (runtime_in / "train-all-die-roots.pt.xz").read_bytes() == big_bytes

    # Slim checkpoint materialization: verify ONLY the 3 needed checkpoints are extracted
    # and the 3 unused Apex league members are dropped.
    b8_bytes = b"synthetic-b8-checkpoint"
    frp3_bytes = b"synthetic-frp3-checkpoint"
    champ_bytes = b"synthetic-champion-checkpoint"
    unused_bytes = b"unused-apex-league-checkpoint"
    b8_sha = hashlib.sha256(b8_bytes).hexdigest()
    frp3_sha = hashlib.sha256(frp3_bytes).hexdigest()
    champ_sha = hashlib.sha256(champ_bytes).hexdigest()

    league_buf = io.BytesIO()
    with tarfile.open(fileobj=league_buf, mode="w:gz") as tar:
        for member_name, data in (
            ("checkpoints/frp3_m3_s12_u60000_20261005.pth", frp3_bytes),
            ("checkpoints/champion_a5_freespace_20260621.pth", champ_bytes),
            ("checkpoints/best_apex_fs.pth", unused_bytes),
            ("checkpoints/best_apex_stage1_fs.pth", unused_bytes),
            ("checkpoints/best_apex_pre_fs.pth", unused_bytes),
        ):
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    league_bytes = league_buf.getvalue()
    league_sha = hashlib.sha256(league_bytes).hexdigest()

    monkeypatch.setattr(
        r7_uploads,
        "REQUIRED_B8_CHECKPOINT",
        ("b8-s3.pth", len(b8_bytes), b8_sha),
    )
    monkeypatch.setattr(
        r7_uploads,
        "REQUIRED_LEAGUE_CHECKPOINTS",
        {
            "frp3_m3_s12_u60000_20261005.pth": ("frp3-s12.pth", len(frp3_bytes), frp3_sha),
            "champion_a5_freespace_20260621.pth": ("champion.pth", len(champ_bytes), champ_sha),
        },
    )
    monkeypatch.setattr(
        r7_uploads,
        "load_public_catalog",
        lambda _p: (
            [],
            {
                "B8_s3_final.pth": {
                    "name": "B8_s3_final.pth",
                    "url": "https://example.invalid/B8_s3_final.pth",
                    "size_bytes": len(b8_bytes),
                    "sha256": b8_sha,
                },
                "league.tar.gz": {
                    "name": "league.tar.gz",
                    "url": "https://example.invalid/league.tar.gz",
                    "size_bytes": len(league_bytes),
                    "sha256": league_sha,
                },
            },
        ),
    )

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.close()

    def _fake_opener(url: str, timeout: int = 60) -> _FakeResp:
        assert timeout == 60
        if url.endswith("B8_s3_final.pth"):
            return _FakeResp(b8_bytes)
        if url.endswith("league.tar.gz"):
            return _FakeResp(league_bytes)
        raise AssertionError(f"unexpected URL {url}")

    published = r7_uploads.materialize_r7_checkpoints(runtime_in, opener=_fake_opener)
    assert set(published) == {"b8-s3.pth", "frp3-s12.pth", "champion.pth"}
    assert (runtime_in / "b8-s3.pth").read_bytes() == b8_bytes
    assert (runtime_in / "frp3-s12.pth").read_bytes() == frp3_bytes
    assert (runtime_in / "champion.pth").read_bytes() == champ_bytes
    for dropped in (
        "best_apex_fs.pth",
        "best_apex_stage1_fs.pth",
        "best_apex_pre_fs.pth",
        "league.tar.gz",
    ):
        assert not (runtime_in / dropped).exists()
