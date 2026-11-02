"""Determinism, signal, collision/death/respawn, identity, and negative tests."""

from __future__ import annotations

import copy
import io
import json
import logging
import os
import pickle
import pickletools
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
import torch

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.run import DEFAULT_LOCAL_SAVED_DIR, world_unit
from research.watch6_search_efficacy.product import open_watch
from research.world_checkpoint import (
    WORLD_CHECKPOINT_SCHEMA,
    CheckpointInterrupted,
    capture_runtime_identity,
    configure_cpu,
    find_latest_checkpoint,
    highest_checkpoint_seq_on_disk,
    inject_live_and_mirror_serpentines,
    load_checkpoint,
    measure_checkpoint_scaling,
    quarantine_incomplete_checkpoints,
    restore_run_session_state,
)
from research.world_checkpoint import run as run_job
from research.world_checkpoint import (
    run_paired_world_resumable,
    save_checkpoint,
    snapshot_run_session_state,
    stage_exported_checkpoints,
    verify_checkpoint,
    verify_restored_types,
    verify_session_parity,
)
from research.world_checkpoint.checkpoint import _PrimitiveUnpickler
from tools.ops.package_spec import empty_ledger, publish_json


@pytest.fixture(autouse=True)
def _checkpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure single-thread CPU execution and stage Watch6 checkpoints."""
    configure_cpu()
    if "SNAKE_CKPT_DIR" not in os.environ and DEFAULT_LOCAL_SAVED_DIR.is_dir():
        monkeypatch.setenv("SNAKE_CKPT_DIR", str(DEFAULT_LOCAL_SAVED_DIR))
    monkeypatch.setenv("SNAKE_WATCH_SOURCE", "live")
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")


def _strip_wall_clock_timings(world_result: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(world_result)
    cleaned.pop("elapsed_seconds", None)
    for arm_data in cleaned.get("arms", {}).values():
        timing = arm_data.get("timing")
        if isinstance(timing, dict):
            for key in list(timing.keys()):
                if "seconds" in key:
                    timing.pop(key, None)
    return cleaned


def _setup_collision_death_respawn_and_long_body_hook(session: Any, _arm: str) -> None:
    """Deterministic hook creating >=600 hero, >=1000 opp, collisions, deaths, and respawns."""
    import random

    from research.bl_ppo.collector import _prepare_sim

    # 1. Inject serpentine bodies: hero=620 (>=600 so L2'/L1 search on frame 1), opp1=1020
    inject_live_and_mirror_serpentines(session, (620, 1020, 300, 200, 150, 100))

    sim = session.world.sim
    ss = int(sim.s)
    game = session.live.game

    # 2. Hero head is at (57, 6) heading RIGHT (1, 0), with hero body along (53..59, 5) above it.
    #    Place snake 4 (slot 4) at y=7 from x=55..59 and blocking (59..64, 6) so hero steps once
    #    on frame 1 (running L2'/L1 search at length 620), collides with snake 4 on frame 2
    #    (collisions={0: 'body'}, kills={4: [0]}), and respawns on frame 3!
    cells4 = [
        (55, 7),
        (56, 7),
        (57, 7),
        (58, 7),
        (59, 7),
        (59, 6),
        (60, 6),
        (61, 6),
        (62, 6),
        (63, 6),
        (64, 6),
    ]
    n4 = len(cells4)
    sim.bodies[0, 4] = 0
    sim.bodies[0, 4, :n4] = np.array(cells4[::-1], dtype=np.int64)
    sim.head_ptr[0, 4] = n4 - 1
    sim.seg_count[0, 4] = n4
    sim.length[0, 4] = n4
    sim._reward_prev_length[0, 4] = n4
    sim.direction[0, 4] = 3

    occupied = set(cells4)
    keep = [c for c in sim.food_cells[0] if c not in occupied]
    sim.food_cells[0][:] = keep
    sim.food_set[0].intersection_update(keep)
    sim.corpse_cells[0].intersection_update(keep)
    sim._rebuild_env_grids(0)
    all_envs = np.ones(sim.E, dtype=bool)
    sim._rebuild_traversed_from_heads(all_envs)
    sim._refresh_action_masks(all_envs)
    _prepare_sim(sim)

    game.food_manager.food = [(int(c[0]) * ss, int(c[1]) * ss) for c in sim.food_cells[0]]
    game.food_manager._corpse_positions = {
        (int(c[0]) * ss, int(c[1]) * ss) for c in sim.corpse_cells[0]
    }
    for sidx, snk in enumerate(game.snakes):
        if sidx == 4:
            snk.segments = [(int(cx) * ss, int(cy) * ss) for cx, cy in cells4]
            snk.direction = (-1, 0)
            snk.length = n4
            snk.is_alive = True
            snk._reward_prev_length = n4
        snk._prev_nearest_enemy_dist = float("inf")
        snk._prev_nearest_enemy_id = None
        snk._carried_selection = None
        if hasattr(snk, "_free_space_cache"):
            delattr(snk, "_free_space_cache")

    rt = session.world.runtime
    rt._carry_valid[...] = False
    rt._alive_post[...] = sim.alive
    rt.memory.prev_dist[...] = float("inf")
    rt.memory.prev_id[...] = -1
    random.setstate(sim._rngs[0]._rng.getstate())
    verify_session_parity(session)


def test_save_load_verify_checkpoint_and_corruption_rejection(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_path = ckpt_dir / "ckpt-seq-000001-arm-bare-frame-00010.ckpt"

    payload = {
        "checkpoint_schema": WORLD_CHECKPOINT_SCHEMA,
        "checkpoint_seq": 1,
        "index": 0,
        "seed": p.FIXTURE_SEEDS[0],
        "current_arm": "bare",
        "current_frame": 10,
        "sample_array": np.arange(12, dtype=np.int64).reshape(3, 4),
        "sample_tensor": torch.tensor([True, False, True, True, False, True]),
    }
    meta = save_checkpoint(payload, ckpt_path)
    assert Path(meta["path"]).is_file()
    assert Path(f"{ckpt_path}.sha256").is_file()
    assert Path(f"{ckpt_path}.complete").is_file()
    assert Path(f"{ckpt_path}.complete").read_text(encoding="utf-8") == "complete\n"

    # F7: Inspect pickle opcode stream and assert zero GLOBAL / STACK_GLOBAL class imports
    raw_bytes = ckpt_path.read_bytes()
    opcodes = [op.name for op, _, _ in pickletools.genops(raw_bytes)]
    assert "GLOBAL" not in opcodes
    assert "STACK_GLOBAL" not in opcodes

    # Verify _PrimitiveUnpickler refuses any external class
    forbidden_pickle = pickle.dumps(Path("/tmp"))
    with pytest.raises(pickle.UnpicklingError, match="forbidden class reference"):
        _PrimitiveUnpickler(io.BytesIO(forbidden_pickle)).load()

    sha = verify_checkpoint(ckpt_path)
    assert sha == meta["sha256"]
    loaded = load_checkpoint(ckpt_path)
    assert loaded["checkpoint_seq"] == 1
    assert np.array_equal(loaded["sample_array"], payload["sample_array"])
    assert "identity" in loaded
    assert find_latest_checkpoint(ckpt_dir) == ckpt_path

    # Overwrite mode atomically updates payload, .sha256, and .complete
    updated = dict(payload, checkpoint_seq=2, current_frame=20)
    meta2 = save_checkpoint(updated, ckpt_path, overwrite=True)
    assert verify_checkpoint(ckpt_path) == meta2["sha256"]
    assert load_checkpoint(ckpt_path)["current_frame"] == 20

    # Write a second higher-sequence checkpoint and an incomplete third checkpoint + temp file
    ckpt2 = ckpt_dir / "ckpt-seq-000003-arm-bare-frame-00030.ckpt"
    save_checkpoint(dict(payload, checkpoint_seq=3, current_frame=30), ckpt2)

    incomplete = ckpt_dir / "ckpt-seq-000004-arm-bare-frame-00040.ckpt"
    incomplete.write_bytes(b"partial-crash-bytes")
    Path(f"{incomplete}.sha256").write_text("0" * 64 + f"  {incomplete.name}\n", encoding="utf-8")
    leftover_tmp = ckpt_dir / ".r33-ckpt-seq-000004.tmp"
    leftover_tmp.write_bytes(b"temp-bytes")

    # Missing .complete -> find_latest_checkpoint must log warning, ignore it, and return ckpt2
    with caplog.at_level(logging.WARNING):
        assert find_latest_checkpoint(ckpt_dir) == ckpt2
    with pytest.raises(ValueError, match="missing completion marker"):
        load_checkpoint(incomplete)

    # F6: quarantine_incomplete_checkpoints moves partial files and preserves highest seq on disk
    assert highest_checkpoint_seq_on_disk(ckpt_dir) == 4
    quarantined = quarantine_incomplete_checkpoints(ckpt_dir)
    assert len(quarantined) == 3
    assert not incomplete.exists()
    assert not leftover_tmp.exists()
    assert highest_checkpoint_seq_on_disk(ckpt_dir) == 4

    # Corrupting payload bytes after .complete fails SHA-256 and falls back to ckpt2 with warning
    corrupted = ckpt_dir / "ckpt-seq-000005-arm-bare-frame-00050.ckpt"
    save_checkpoint(dict(payload, checkpoint_seq=5, current_frame=50), corrupted)
    corrupted.write_bytes(corrupted.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_checkpoint(corrupted)
    with caplog.at_level(logging.WARNING):
        assert find_latest_checkpoint(ckpt_dir) == ckpt2
    assert any("Skipping invalid or corrupt checkpoint" in rec.message for rec in caplog.records)


def test_identity_binding_and_thread_caps_refuse_mismatch(tmp_path: Path) -> None:
    """Verify F3/F9: thread caps on import, and refusing mismatched model/code/platform/roster."""
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("VECLIB_MAXIMUM_THREADS") == "1"
    assert os.environ.get("NUMEXPR_NUM_THREADS") == "1"
    assert os.environ.get("NUMBA_NUM_THREADS") == "1"
    assert int(torch.get_num_threads()) == 1
    assert int(numba.get_num_threads()) == 1

    seed = p.FIXTURE_SEEDS[0]
    ckpt_dir = tmp_path / "identity_world"
    with pytest.raises(CheckpointInterrupted) as exc_info:
        run_paired_world_resumable(
            index=0,
            seed=seed,
            arm_order=("bare", "l2_prime", "l1"),
            horizon=10,
            fixture=True,
            checkpoint_dir=ckpt_dir,
            stop_at=("bare", 4),
        )
    ckpt_path = exc_info.value.checkpoint_path
    base_data = load_checkpoint(ckpt_path)
    identity = capture_runtime_identity()
    assert base_data["identity"] == identity

    # 1. Tamper with model SHA-256
    bad_model = copy.deepcopy(base_data)
    first_pin = sorted(bad_model["identity"]["model_sha256s"].keys())[0]
    bad_model["identity"]["model_sha256s"][first_pin] = "0" * 64
    bad_model_path = tmp_path / "bad_model.ckpt"
    save_checkpoint(bad_model, bad_model_path)
    with pytest.raises(ValueError, match="checkpoint identity mismatch in model_sha256s"):
        load_checkpoint(bad_model_path)

    # 2. Tamper with code identity
    bad_code = copy.deepcopy(base_data)
    bad_code["identity"]["code_identity"]["world_checkpoint_code_sha256"] = "f" * 64
    bad_code_path = tmp_path / "bad_code.ckpt"
    save_checkpoint(bad_code, bad_code_path)
    with pytest.raises(ValueError, match="checkpoint identity mismatch in code_identity"):
        load_checkpoint(bad_code_path)

    # 3. Tamper with host platform fingerprint
    bad_plat = copy.deepcopy(base_data)
    bad_plat["identity"]["platform_fingerprint"]["cpu_model"] = "Different-CPU-Arch-9999"
    bad_plat_path = tmp_path / "bad_plat.ckpt"
    save_checkpoint(bad_plat, bad_plat_path)
    with pytest.raises(ValueError, match="checkpoint identity mismatch in platform_fingerprint"):
        load_checkpoint(bad_plat_path)

    # 4. Tamper with saved in-progress roster and verify run_paired_world_resumable refuses resume
    bad_roster_dir = tmp_path / "bad_roster_world"
    bad_roster = copy.deepcopy(base_data)
    bad_roster["in_progress_arm"]["counters"]["roster"]["veto_variant"] = "tampered-v0"
    save_checkpoint(
        bad_roster,
        bad_roster_dir / "ckpt-seq-000001-arm-bare-frame-00004.ckpt",
    )
    with pytest.raises(
        ValueError,
        match="checkpoint in-progress roster does not match newly loaded model roster",
    ):
        run_paired_world_resumable(
            index=0,
            seed=seed,
            arm_order=("bare", "l2_prime", "l1"),
            horizon=10,
            fixture=True,
            checkpoint_dir=bad_roster_dir,
        )


def test_negative_corrupted_restored_fields_and_type_invariants_in_process(
    tmp_path: Path,
) -> None:
    """In-process negative/mutation tests verifying corrupted fields or types fail."""
    seed = p.FIXTURE_SEEDS[0]
    session = open_watch({"seed": int(seed)}, 24, "bare")
    try:
        inject_live_and_mirror_serpentines(session, (650, 500, 400, 300, 200, 150))
        for _ in range(3):
            session.step()

        snap = snapshot_run_session_state(session)

        # 1. Corrupting restored sim.frames_since_food breaks physical parity
        restore_run_session_state(session, snap)
        session.world.sim.frames_since_food[0, 0] += 1
        with pytest.raises(RuntimeError, match="Watch/Grid physical identity failed"):
            verify_session_parity(session)

        # 2. Corrupting restored runtime.memory.prev_dist on opponent slots breaks carry parity
        restore_run_session_state(session, snap)
        session.world.runtime.memory.prev_dist[0, 1:] = 99999.0
        with pytest.raises(RuntimeError, match="Watch/Grid carry identity failed"):
            verify_session_parity(session)

        # 3. Corrupting restored opponent v8 veto state breaks v8 parity
        restore_run_session_state(session, snap)
        for opp in session.world.opponents:
            hook = opp.live_vetoes.get(0)
            if hook is not None:
                hook.reference_lambda = 999.0
                break
        with pytest.raises(RuntimeError, match="Watch/Grid v8 identity failed"):
            verify_session_parity(session)

        # 4. Corrupting restored field types triggers verify_restored_types TypeError (F1/F4)
        restore_run_session_state(session, snap)
        session.live.game.frame_collisions = [(1, "body", 2)]  # type: ignore[assignment]
        with pytest.raises(TypeError, match="invalid game.frame_collisions type"):
            verify_restored_types(session, snap)

        restore_run_session_state(session, snap)
        session.live.game.frame_ate_food = {0: 1}  # type: ignore[dict-item]
        with pytest.raises(TypeError, match="invalid game.frame_ate_food type"):
            verify_restored_types(session, snap)

        restore_run_session_state(session, snap)
        session.live.game.snakes[0].last_next_action_mask = [True] * 6  # type: ignore[assignment]
        with pytest.raises(TypeError, match="last_next_action_mask must be Tensor"):
            verify_restored_types(session, snap)

        restore_run_session_state(session, snap)
        was_tuple = isinstance(session.world.sim._last_kill_victim_len[0, 0], tuple)
        session.world.sim._last_kill_victim_len[0, 0] = [10] if was_tuple else (10,)
        with pytest.raises(TypeError, match="tuple/list mismatch"):
            verify_restored_types(session, snap)
    finally:
        session.close()

    # 5. Corrupting DenseTrigger state in a saved checkpoint alters resumed output vs baseline
    long_hero_hook = lambda s, _a: inject_live_and_mirror_serpentines(  # noqa: E731
        s, (650, 500, 400, 300, 200, 150)
    )
    baseline = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=("l2_prime", "bare", "l1"),
        horizon=6,
        fixture=True,
        initial_session_hook=long_hero_hook,
    )
    corrupt_dir = tmp_path / "corrupt_trigger_world"
    with pytest.raises(CheckpointInterrupted) as exc_info:
        run_paired_world_resumable(
            index=0,
            seed=seed,
            arm_order=("l2_prime", "bare", "l1"),
            horizon=6,
            fixture=True,
            checkpoint_dir=corrupt_dir,
            stop_at=("l2_prime", 1),
            initial_session_hook=long_hero_hook,
        )
    corrupt_data = load_checkpoint(exc_info.value.checkpoint_path)
    # Shift first_decision to 3 and min_length to 9999 so no further searches fire on frames 2..6
    corrupt_data["in_progress_arm"]["counters"]["trigger"]["first_decision"] = 3
    corrupt_data["in_progress_arm"]["counters"]["trigger"]["min_length"] = 9999
    save_checkpoint(
        corrupt_data,
        corrupt_dir / "ckpt-seq-000002-arm-l2_prime-frame-00001.ckpt",
    )
    corrupted_resumed = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=("l2_prime", "bare", "l1"),
        horizon=6,
        fixture=True,
        checkpoint_dir=corrupt_dir,
        initial_session_hook=long_hero_hook,
    )
    assert (
        corrupted_resumed["arms"]["l2_prime"]["batches"] != baseline["arms"]["l2_prime"]["batches"]
    )


def test_collisions_deaths_respawns_and_long_body_search_determinism(
    tmp_path: Path,
) -> None:
    """Verify F1/F4/F5: stop/resume at frames with collisions, deaths, respawns, and long bodies."""
    seed = p.FIXTURE_SEEDS[0]
    horizon = 12
    arm_order = ("bare", "l2_prime", "l1")

    # Probe exact frame numbers where hero collision/death and respawn occur
    probe = open_watch({"seed": int(seed)}, 24, "bare")
    try:
        _setup_collision_death_respawn_and_long_body_hook(probe, "bare")
        collision_frames: list[int] = []
        respawn_frames: list[int] = []
        for f in range(1, horizon + 1):
            prev_alive = [bool(s.is_alive) for s in probe.live.game.snakes]
            probe.step()
            if probe.live.game.frame_collisions:
                collision_frames.append(f)
            now_alive = [bool(s.is_alive) for s in probe.live.game.snakes]
            if any(now_alive[i] and not prev_alive[i] for i in range(6)):
                respawn_frames.append(f)
    finally:
        probe.close()

    assert collision_frames, "Expected at least one collision frame"
    assert respawn_frames, "Expected at least one respawn frame"

    baseline_digests: dict[tuple[str, int], str] = {}
    baseline = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        on_frame_digest=lambda a, f, d: baseline_digests.__setitem__((a, f), d),
        initial_session_hook=_setup_collision_death_respawn_and_long_body_hook,
    )
    assert baseline["arms"]["bare"]["deaths"] >= 1
    assert baseline["arms"]["bare"]["long_body_deaths_ge600"] >= 1
    assert sum(baseline["arms"]["bare"]["respawns_per_slot"]) >= 1
    assert baseline["arms"]["l2_prime"]["batches_ge600"] >= 1
    assert baseline["arms"]["l1"]["batches_ge600"] >= 1

    ckpt_dir = tmp_path / "collision_death_respawn_world"
    resumed_digests: dict[tuple[str, int], str] = {}
    stop_points = [
        ("bare", collision_frames[0]),
        ("bare", respawn_frames[0]),
        ("l2_prime", collision_frames[0]),
        ("l2_prime", respawn_frames[0]),
        ("l1", collision_frames[0]),
        ("l1", respawn_frames[0]),
    ]
    for stop_arm, stop_frame in stop_points:
        with pytest.raises(CheckpointInterrupted):
            run_paired_world_resumable(
                index=0,
                seed=seed,
                arm_order=arm_order,
                horizon=horizon,
                fixture=True,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=3,
                on_frame_digest=lambda a, f, d: resumed_digests.__setitem__((a, f), d),
                stop_at=(stop_arm, stop_frame),
                initial_session_hook=_setup_collision_death_respawn_and_long_body_hook,
            )

    resumed = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=3,
        on_frame_digest=lambda a, f, d: resumed_digests.__setitem__((a, f), d),
        initial_session_hook=_setup_collision_death_respawn_and_long_body_hook,
    )
    assert resumed_digests == baseline_digests
    assert _strip_wall_clock_timings(resumed) == _strip_wall_clock_timings(baseline)

    # F5: Test SIGUSR1 arriving mid-search on the exact respawn frame (`not was_alive`)
    # On arm "l1" (`min_length=0`), when hero dies on collision_frames[0] and respawns on
    # respawn_frames[0], choose() sees `active=True and not was_alive`, creates a fresh
    # DenseTrigger(min_length=0), and runs dense_batch search on respawn_frames[0]!
    f5_ckpt_dir = tmp_path / "f5_respawn_search_signal"
    f5_digests: dict[tuple[str, int], str] = {}
    hero_respawn_frame = respawn_frames[0]
    with pytest.raises(CheckpointInterrupted) as exc_f5:
        run_paired_world_resumable(
            index=0,
            seed=seed,
            arm_order=arm_order,
            horizon=horizon,
            fixture=True,
            checkpoint_dir=f5_ckpt_dir,
            on_frame_digest=lambda a, f, d: f5_digests.__setitem__((a, f), d),
            signal_on_search=("l1", hero_respawn_frame, signal.SIGUSR1),
            initial_session_hook=_setup_collision_death_respawn_and_long_body_hook,
        )
    assert exc_f5.value.arm == "l1"
    assert exc_f5.value.frame_index == hero_respawn_frame - 1
    verify_checkpoint(exc_f5.value.checkpoint_path)

    f5_resumed = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        checkpoint_dir=f5_ckpt_dir,
        on_frame_digest=lambda a, f, d: f5_digests.__setitem__((a, f), d),
        initial_session_hook=_setup_collision_death_respawn_and_long_body_hook,
    )
    assert f5_digests == baseline_digests
    assert _strip_wall_clock_timings(f5_resumed) == _strip_wall_clock_timings(baseline)


def test_paired_world_determinism_uninterrupted_vs_resumed(tmp_path: Path) -> None:
    seed = p.FIXTURE_SEEDS[0]
    horizon = 300
    arm_order = ("bare", "l2_prime", "l1")

    uninterrupted_digests: dict[tuple[str, int], str] = {}

    def record_uninterrupted(arm: str, frame_index: int, digest_hex: str) -> None:
        assert (arm, frame_index) not in uninterrupted_digests
        uninterrupted_digests[(arm, frame_index)] = digest_hex

    uninterrupted = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        checkpoint_dir=None,
        on_frame_digest=record_uninterrupted,
    )
    assert len(uninterrupted_digests) == 3 * horizon
    assert uninterrupted["arms"]["l1"]["batches"] == 60

    reference = world_unit(
        {"index": 0, "seed": seed, "horizon": horizon, "arm_order": list(arm_order)},
        fixture=True,
    )
    assert _strip_wall_clock_timings(uninterrupted) == _strip_wall_clock_timings(reference)

    ckpt_dir = tmp_path / "resumed_world"
    resumed_digests: dict[tuple[str, int], str] = {}

    def record_resumed(arm: str, frame_index: int, digest_hex: str) -> None:
        assert (arm, frame_index) not in resumed_digests
        resumed_digests[(arm, frame_index)] = digest_hex

    stop_points = [
        ("bare", 113),
        ("bare", 300),
        ("l2_prime", 197),
        ("l2_prime", 300),
        ("l1", 127),
        ("l1", 233),
    ]

    for expected_arm, expected_frame in stop_points:
        with pytest.raises(CheckpointInterrupted) as exc_info:
            run_paired_world_resumable(
                index=0,
                seed=seed,
                arm_order=arm_order,
                horizon=horizon,
                fixture=True,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=90,
                on_frame_digest=record_resumed,
                stop_at=(expected_arm, expected_frame),
            )
        interrupted = exc_info.value
        assert interrupted.arm == expected_arm
        assert interrupted.frame_index == expected_frame
        assert interrupted.checkpoint_path.is_file()
        verify_checkpoint(interrupted.checkpoint_path)

    resumed_result = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=90,
        on_frame_digest=record_resumed,
    )

    assert len(resumed_digests) == 3 * horizon
    assert resumed_digests == uninterrupted_digests
    assert _strip_wall_clock_timings(resumed_result) == _strip_wall_clock_timings(uninterrupted)


def test_sigusr1_and_sigterm_graceful_checkpoint_and_resume(tmp_path: Path) -> None:
    seed = p.FIXTURE_SEEDS[1]
    horizon = 60
    arm_order = ("l1", "bare", "l2_prime")

    baseline_digests: dict[tuple[str, int], str] = {}
    baseline = run_paired_world_resumable(
        index=1,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        on_frame_digest=lambda a, f, d: baseline_digests.__setitem__((a, f), d),
    )

    ckpt_dir = tmp_path / "signal_world"
    resumed_digests: dict[tuple[str, int], str] = {}

    t0 = time.monotonic()
    with pytest.raises(CheckpointInterrupted) as exc_usr1:
        run_paired_world_resumable(
            index=1,
            seed=seed,
            arm_order=arm_order,
            horizon=horizon,
            fixture=True,
            checkpoint_dir=ckpt_dir,
            on_frame_digest=lambda a, f, d: resumed_digests.__setitem__((a, f), d),
            signal_on_search=("l1", 16, signal.SIGUSR1),
        )
    grace_elapsed = time.monotonic() - t0
    assert grace_elapsed < 15.0
    assert exc_usr1.value.signal_number == signal.SIGUSR1
    assert exc_usr1.value.arm == "l1"
    assert exc_usr1.value.frame_index == 15
    verify_checkpoint(exc_usr1.value.checkpoint_path)

    def send_sigterm(arm: str, frame_index: int) -> None:
        if arm == "l1" and frame_index == 35:
            os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(CheckpointInterrupted) as exc_term:
        run_paired_world_resumable(
            index=1,
            seed=seed,
            arm_order=arm_order,
            horizon=horizon,
            fixture=True,
            checkpoint_dir=ckpt_dir,
            on_progress=send_sigterm,
            on_frame_digest=lambda a, f, d: resumed_digests.__setitem__((a, f), d),
        )
    assert exc_term.value.signal_number == signal.SIGTERM
    assert exc_term.value.arm == "l1"
    assert exc_term.value.frame_index == 35
    verify_checkpoint(exc_term.value.checkpoint_path)

    final = run_paired_world_resumable(
        index=1,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        checkpoint_dir=ckpt_dir,
        on_frame_digest=lambda a, f, d: resumed_digests.__setitem__((a, f), d),
    )

    assert resumed_digests == baseline_digests
    assert _strip_wall_clock_timings(final) == _strip_wall_clock_timings(baseline)


def test_long_body_checkpoint_scaling_and_restore_parity() -> None:
    rows = measure_checkpoint_scaling(
        seed=p.FIXTURE_SEEDS[0],
        hero_lengths=(1, 150, 450, 900, 1200, 1500),
        opponent_lengths=(900, 700, 500, 400, 300),
        steps_after_inject=3,
        repeats=3,
    )
    assert len(rows) == 6
    sizes = [r["size_bytes"] for r in rows]
    for prev_size, next_size in zip(sizes, sizes[1:]):
        assert next_size > prev_size
    for row in rows:
        assert row["save_ms_median"] < 200.0
        assert row["load_ms_median"] < 200.0


def test_job_entrypoint_and_job_sh_export_and_resume(tmp_path: Path) -> None:
    """Verify F2: job.py and job.sh publish checkpoints with .sha256/.complete and resume."""
    horizon = 6
    ref0 = world_unit(
        {
            "index": 0,
            "seed": p.FIXTURE_SEEDS[0],
            "horizon": horizon,
            "arm_order": p.arm_order_for_index(0),
        },
        fixture=True,
    )
    ref1 = world_unit(
        {
            "index": 1,
            "seed": p.FIXTURE_SEEDS[1],
            "horizon": horizon,
            "arm_order": p.arm_order_for_index(1),
        },
        fixture=True,
    )

    # Attempt 1: interrupt world 0 mid-arm at frame 3 of l2_prime, exporting checkpoints
    attempt1_out = tmp_path / "attempt1_out"
    with pytest.raises(CheckpointInterrupted) as exc_info:
        run_job(
            out=attempt1_out / "efficacy",
            ledger_path=None,
            workers=1,
            shards=1,
            shard=0,
            fixture=True,
            fixture_horizon=horizon,
            fixture_count=2,
            local_smoke=True,
            checkpoint_interval=2,
            stop_at_by_index={0: ("l2_prime", 3)},
        )
    assert exc_info.value.arm == "l2_prime"
    assert exc_info.value.frame_index == 3
    verify_checkpoint(exc_info.value.checkpoint_path)

    # Verify exported checkpoints under attempt1_out/efficacy/checkpoints/fixture-world-00
    exported_ckpts = list(
        (attempt1_out / "efficacy" / "checkpoints" / "fixture-world-00").glob("*.ckpt.complete")
    )
    assert len(exported_ckpts) >= 2

    # Stage Attempt 1's exported checkpoints into Attempt 2's COLAB_INPUT_DIR/world-checkpoints
    attempt2_in = tmp_path / "attempt2_in"
    attempt2_out = tmp_path / "attempt2_out"
    attempt2_in.mkdir(parents=True, exist_ok=True)
    attempt2_out.mkdir(parents=True, exist_ok=True)

    staged = stage_exported_checkpoints(attempt1_out, attempt2_in / "world-checkpoints")
    assert len(staged) == 1
    assert staged[0]["world_tag"] == "fixture-world-00"

    # Write empty verified-ledger.json for Attempt 2
    publish_json(
        attempt2_in / "verified-ledger.json",
        empty_ledger(p.NAMESPACE, "watch6-fixture", {"science": p.identity()}),
    )

    # Run Attempt 2 via research/world_checkpoint/job.sh subprocess
    repo_root = Path(__file__).resolve().parents[1]
    job_sh = repo_root / "research" / "world_checkpoint" / "job.sh"
    env = dict(os.environ)
    env.update(
        {
            "COLAB_INPUT_DIR": str(attempt2_in),
            "COLAB_OUTPUT_DIR": str(attempt2_out),
            "WATCH6_FIXTURE": "1",
            "WATCH6_LOCAL_SMOKE": "1",
            "WATCH6_FIXTURE_COUNT": "2",
            "WATCH6_FIXTURE_HORIZON": str(horizon),
            "WATCH6_CHECKPOINT_INTERVAL": "2",
            "WATCH6_WORKERS": "1",
            "WATCH6_SHARDS": "1",
            "WATCH6_SHARD": "0",
            "PYTHONPATH": str(repo_root),
            "PATH": f"{Path(sys.executable).parent}:{env.get('PATH', '')}",
        }
    )
    proc = subprocess.run(
        ["bash", str(job_sh)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0

    w0_path = attempt2_out / "efficacy" / "fixture-world-00.json"
    w1_path = attempt2_out / "efficacy" / "fixture-world-01.json"
    assert Path(f"{w0_path}.complete").is_file()
    assert Path(f"{w1_path}.complete").is_file()

    w0 = json.loads(w0_path.read_text(encoding="utf-8"))
    w1 = json.loads(w1_path.read_text(encoding="utf-8"))
    assert _strip_wall_clock_timings(w0) == _strip_wall_clock_timings(ref0)
    assert _strip_wall_clock_timings(w1) == _strip_wall_clock_timings(ref1)


@pytest.mark.slow
def test_natural_horizon_collision_death_respawn_around_frame_1984(
    tmp_path: Path,
) -> None:
    """Natural 2005-frame horizon stopping/resuming around natural collisions (1984, 2001)."""
    from research.watch6_l1_confirm.engine import run_arm

    seed = p.FIXTURE_SEEDS[0]
    horizon = 2005
    arm_order = ("bare", "l2_prime")

    baseline_digests: dict[tuple[str, int], str] = {}

    def record_window_baseline(arm: str, frame_index: int, digest_hex: str) -> None:
        if frame_index >= 1980:
            baseline_digests[(arm, frame_index)] = digest_hex

    baseline = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        on_frame_digest=record_window_baseline,
    )
    # Verify natural opponent respawns occurred at frames 1985 (slot 1) and 2002 (slot 3)
    assert baseline["arms"]["bare"]["respawns_per_slot"] == [0, 1, 0, 1, 0, 0]
    assert baseline["arms"]["l2_prime"]["respawns_per_slot"] == [0, 1, 0, 1, 0, 0]

    # Also verify exact match against stock research.watch6_l1_confirm.engine.run_arm
    ref_bare = dict(run_arm(seed, "bare", lambda: None, horizon=horizon))
    ref_bare.pop("roster", None)
    assert _strip_wall_clock_timings({"arms": {"bare": baseline["arms"]["bare"]}}) == (
        _strip_wall_clock_timings({"arms": {"bare": ref_bare}})
    )

    ckpt_dir = tmp_path / "natural_1984_world"
    resumed_digests: dict[tuple[str, int], str] = {}

    def record_window_resumed(arm: str, frame_index: int, digest_hex: str) -> None:
        if frame_index >= 1980:
            resumed_digests[(arm, frame_index)] = digest_hex

    stop_points = [
        ("bare", 1983),
        ("bare", 1984),
        ("bare", 1985),
        ("bare", 2001),
        ("bare", 2002),
        ("l2_prime", 1984),
        ("l2_prime", 1985),
        ("l2_prime", 2001),
        ("l2_prime", 2002),
    ]
    for stop_arm, stop_frame in stop_points:
        with pytest.raises(CheckpointInterrupted) as exc_info:
            run_paired_world_resumable(
                index=0,
                seed=seed,
                arm_order=arm_order,
                horizon=horizon,
                fixture=True,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=1000,
                on_frame_digest=record_window_resumed,
                stop_at=(stop_arm, stop_frame),
            )
        verify_checkpoint(exc_info.value.checkpoint_path)

    resumed = run_paired_world_resumable(
        index=0,
        seed=seed,
        arm_order=arm_order,
        horizon=horizon,
        fixture=True,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=1000,
        on_frame_digest=record_window_resumed,
    )
    assert resumed_digests == baseline_digests
    assert _strip_wall_clock_timings(resumed) == _strip_wall_clock_timings(baseline)
