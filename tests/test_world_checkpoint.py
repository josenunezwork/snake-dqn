"""Determinism, signal-handling, atomic file, and long-body scaling tests for world_checkpoint."""

from __future__ import annotations

import copy
import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.run import DEFAULT_LOCAL_SAVED_DIR, world_unit
from research.world_checkpoint import (
    WORLD_CHECKPOINT_SCHEMA,
    CheckpointInterrupted,
    find_latest_checkpoint,
    load_checkpoint,
    measure_checkpoint_scaling,
    run_paired_world_resumable,
    save_checkpoint,
    verify_checkpoint,
)


@pytest.fixture(autouse=True)
def _checkpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_save_load_verify_checkpoint_and_corruption_rejection(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_path = ckpt_dir / "ckpt-seq-000001-arm-bare-frame-00010.ckpt"

    payload = {
        "checkpoint_schema": WORLD_CHECKPOINT_SCHEMA,
        "checkpoint_seq": 1,
        "index": 0,
        "seed": p.FIXTURE_SEEDS[0],
        "current_arm": "bare",
        "current_frame": 10,
    }
    meta = save_checkpoint(payload, ckpt_path)
    assert Path(meta["path"]).is_file()
    assert Path(f"{ckpt_path}.sha256").is_file()
    assert Path(f"{ckpt_path}.complete").is_file()
    assert Path(f"{ckpt_path}.complete").read_text(encoding="utf-8") == "complete\n"

    sha = verify_checkpoint(ckpt_path)
    assert sha == meta["sha256"]
    loaded = load_checkpoint(ckpt_path)
    assert loaded == payload
    assert find_latest_checkpoint(ckpt_dir) == ckpt_path

    # Overwrite mode atomically updates payload, .sha256, and .complete
    updated = dict(payload, checkpoint_seq=2, current_frame=20)
    meta2 = save_checkpoint(updated, ckpt_path, overwrite=True)
    assert verify_checkpoint(ckpt_path) == meta2["sha256"]
    assert load_checkpoint(ckpt_path)["current_frame"] == 20

    # Write a second higher-sequence checkpoint and an incomplete third checkpoint
    ckpt2 = ckpt_dir / "ckpt-seq-000003-arm-bare-frame-00030.ckpt"
    save_checkpoint(dict(payload, checkpoint_seq=3, current_frame=30), ckpt2)

    incomplete = ckpt_dir / "ckpt-seq-000004-arm-bare-frame-00040.ckpt"
    incomplete.write_bytes(b"partial-crash-bytes")
    Path(f"{incomplete}.sha256").write_text("0" * 64 + f"  {incomplete.name}\n", encoding="utf-8")
    # Missing .complete -> find_latest_checkpoint must ignore it and return ckpt2
    assert find_latest_checkpoint(ckpt_dir) == ckpt2
    with pytest.raises(ValueError, match="missing completion marker"):
        load_checkpoint(incomplete)

    # Corrupting payload bytes after .complete must fail SHA-256 verification
    corrupted = ckpt_dir / "ckpt-seq-000005-arm-bare-frame-00050.ckpt"
    save_checkpoint(dict(payload, checkpoint_seq=5, current_frame=50), corrupted)
    corrupted.write_bytes(corrupted.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_checkpoint(corrupted)
    # find_latest_checkpoint skips the corrupted checkpoint 5 and falls back to ckpt2
    assert find_latest_checkpoint(ckpt_dir) == ckpt2


def test_paired_world_determinism_uninterrupted_vs_resumed(tmp_path: Path) -> None:
    seed = p.FIXTURE_SEEDS[0]
    horizon = 300
    arm_order = ("bare", "l2_prime", "l1")

    # 1. Run uninterrupted and record per-frame state hashes for every frame of every arm.
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

    # Also verify parity against the stock Watch6-L1-V1 world_unit engine.
    stock = world_unit(
        {
            "index": 0,
            "seed": seed,
            "horizon": horizon,
            "arm_order": list(arm_order),
        },
        fixture=True,
    )
    assert _strip_wall_clock_timings(uninterrupted) == _strip_wall_clock_timings(stock)

    # 2. Run interrupted-and-resumed across 6 stop points (mid-arm and arm boundaries):
    #    - mid-arm 1: ("bare", 110)
    #    - arm 1 boundary: ("bare", 300)
    #    - mid-arm 2: ("l2_prime", 175)
    #    - arm 2 boundary: ("l2_prime", 300)
    #    - mid-arm 3 (active exact-stop search): ("l1", 127)
    #    - mid-arm 3 (active exact-stop search): ("l1", 233)
    #    - completion: ("l1", 300)
    ckpt_dir = tmp_path / "resumable_world"
    resumed_digests: dict[tuple[str, int], str] = {}

    def record_resumed(arm: str, frame_index: int, digest_hex: str) -> None:
        assert (
            arm,
            frame_index,
        ) not in resumed_digests, f"duplicate frame execution ({arm}, {frame_index}) on resume"
        resumed_digests[(arm, frame_index)] = digest_hex

    stop_points = [
        ("bare", 110),
        ("bare", 300),
        ("l2_prime", 175),
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

    # Final resume to completion
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

    # 1. Trigger SIGUSR1 right inside dense_batch on l1 frame 16 (decision 15)
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
    # Aborted mid-search on frame 16 -> rolled back and saved verified checkpoint at frame 15
    assert exc_usr1.value.frame_index == 15
    verify_checkpoint(exc_usr1.value.checkpoint_path)

    # 2. Resume and deliver SIGTERM on frame 35 of l1 via on_progress
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

    # 3. Resume to completion
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
    # Verify size grows monotonically with snake length from L=150 to L=1500
    sizes = [r["size_bytes"] for r in rows]
    for prev_size, next_size in zip(sizes, sizes[1:]):
        assert next_size > prev_size
    # Even at max length (L=1500 hero + 2800 opponents = 4300 total segments),
    # save and load times on Mac are well under 100 ms.
    for row in rows:
        assert row["save_ms_median"] < 200.0
        assert row["load_ms_median"] < 200.0
