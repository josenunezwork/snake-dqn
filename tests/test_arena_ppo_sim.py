"""Tests for `research.arena_ppo.sim` (`TrainSim` exact v2 dynamics, masks, opponents)."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import pytest
import torch

from research.arena_ppo import runtime as arena_runtime
from research.arena_ppo.runtime import (
    SignalStopFlag,
    preload_native_sim_extensions,
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.arena_ppo.sim import (
    _SIM_STATE_TENSORS,
    LateStateReservoir,
    TrainSim,
    arena_v2_config,
    derive_arena_world_seeds,
    verify_indexed_vs_reference_parity,
)
from research.gpu_sim_spike import event_cuda, move_cuda
from research.gpu_sim_spike.anchors_device import choose_anchor
from research.gpu_sim_spike.device_sim import DeviceSim


def _make_mature_reference_device_sim(num_envs: int, *, seed: int = 1234) -> DeviceSim:
    """Build a CPU exact-v2 DeviceSim with mature snake lengths to exercise boost & corpses."""
    cfg = arena_v2_config(num_envs, max_capacity=512)
    sim = DeviceSim(
        cfg,
        device="cpu",
        train_mode=False,
        allow_respawn=True,
        cap=cfg.max_capacity,
        move_backend="reference",
        hero_respawn=True,
        exact_mechanics=True,
    )
    seeds = derive_arena_world_seeds(seed, num_envs)
    sim.rng_key = torch.tensor(seeds, dtype=torch.long) & 0x7FFF_FFFF_FFFF_FFFF
    sim.rng_counter.zero_()
    sim.alive.fill_(False)
    sim.respawn_timer.zero_()
    sim.seg_count.zero_()
    sim.length.fill_(1)
    sim.food_present.fill_(False)
    sim.corpse_food.fill_(False)
    sim.food_count.zero_()
    sim._respawn_exact()
    sim._refill_food_rows(torch.ones((num_envs,), dtype=torch.bool))
    sim.frame.zero_()
    sim._prepared = False

    # Assign varied lengths (including boostable >=5 and 1.15x size-ratio contests)
    base_lengths = torch.tensor([1, 4, 5, 12, 20, 24], dtype=torch.long)
    sim.length.copy_(base_lengths[None, :].expand(num_envs, -1).clone())
    sim._reward_prev_length.copy_(sim.length)
    return sim


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_trainsim_dynamics_byte_identical_to_devicesim_v2_64x500(device_name: str) -> None:
    """Prove TrainSim dynamics == DeviceSim v2 byte-for-byte on 64 worlds x 500 ticks."""
    num_envs = 64
    ticks = 500
    ref = _make_mature_reference_device_sim(num_envs, seed=20261110)
    train_sim = TrainSim.from_reference(ref, device=device_name)

    gen = torch.Generator(device="cpu").manual_seed(987654321)
    action_tape = torch.randint(0, 6, (ticks, num_envs, 6), generator=gen, dtype=torch.long)

    total_deaths = 0
    total_boosts = 0
    for tick in range(ticks):
        act_cpu = action_tape[tick]
        ref.step(act_cpu)
        info = train_sim.step(act_cpu.to(train_sim.device))
        total_deaths += int(info.died.sum().cpu().item())
        total_boosts += int(info.boosted.sum().cpu().item())

        if (tick + 1) % 100 == 0 or tick == ticks - 1:
            for field in _SIM_STATE_TENSORS:
                expected = getattr(ref, field)
                observed = getattr(train_sim.sim, field).detach().cpu()
                assert expected.dtype == observed.dtype, f"dtype mismatch on {field} at {tick}"
                assert expected.shape == observed.shape, f"shape mismatch on {field} at {tick}"
                if expected.is_floating_point():
                    assert torch.allclose(
                        expected, observed, atol=1e-6, rtol=0.0
                    ), f"float mismatch on {field} at tick={tick}"
                else:
                    assert torch.equal(
                        expected, observed
                    ), f"exact tensor mismatch on {field} at tick={tick}"

    assert total_deaths > 50, "64x500 tape must exercise snake deaths, corpses, and respawns"
    assert total_boosts > 100, "64x500 tape must exercise boost steps and paid-tail pellets"


def test_legal_action_masks_and_opponent_routing() -> None:
    """Verify legal boost masking and per-world opponent routing (`self`/`snapshot`/`scripted`)."""
    train_sim = TrainSim.create(
        6,
        device="cpu",
        seed=77,
        max_capacity=256,
        opponent_kinds=["self", "self", "snapshot", "snapshot", "scripted", "scripted"],
    )
    train_sim.prepare()
    train_sim.sim.length[:] = torch.tensor(
        [
            [1, 4, 5, 6, 10, 3],
            [5, 5, 4, 2, 8, 1],
            [3, 7, 5, 4, 9, 6],
            [5, 1, 2, 5, 5, 4],
            [6, 5, 3, 8, 4, 12],
            [4, 4, 4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    train_sim.sim.alive[1, 0] = False

    mask = train_sim.legal_action_mask()
    assert mask.shape == (6, 6, 6)
    assert mask[:, :, :3].all(), "normal turns 0..2 must remain legal for all rows"
    expected_boost = train_sim.sim.alive & (train_sim.sim.length >= 5)
    assert torch.equal(mask[:, :, 3], expected_boost)
    assert torch.equal(mask[:, :, 4], expected_boost)
    assert torch.equal(mask[:, :, 5], expected_boost)
    assert not bool(mask[1, 0, 3:].any()), "dead seat cannot boost"

    learner_mask = train_sim.learner_seat_mask()
    assert learner_mask[:2].all()
    assert learner_mask[2:, 0].all()
    assert not bool(learner_mask[2:, 1:].any())

    self_act = torch.zeros((6, 6), dtype=torch.long)
    snap_act = torch.full((6, 6), 2, dtype=torch.long)
    scr_act = torch.full((6, 6), 1, dtype=torch.long)
    routed = train_sim.compose_world_actions(
        self_act, snapshot_actions=snap_act, scripted_actions=scr_act
    )
    assert (routed[:2] == 0).all()
    assert (routed[2:4, 0] == 0).all() and (routed[2:4, 1:] == 2).all()
    assert (routed[4:6, 0] == 0).all() and (routed[4:6, 1:] == 1).all()


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_scripted_greedy_food_matches_anchor_oracle(device_name: str) -> None:
    """Verify vectorized greedy_food matches anchors_device.choose_anchor across all 6 seats."""
    ref = _make_mature_reference_device_sim(16, seed=555)
    train_sim = TrainSim.from_reference(ref, device=device_name)
    gen = torch.Generator(device="cpu").manual_seed(42)
    for _ in range(15):
        act = torch.randint(0, 6, (16, 6), generator=gen, dtype=torch.long)
        ref.step(act)
        train_sim.step(act.to(train_sim.device))
        ref.prepare()
        train_sim.prepare()
        vec_actions = train_sim.scripted_greedy_food_actions(chunk_envs=5).cpu()
        oracle_actions = torch.stack(
            [choose_anchor(ref, slot, "greedy_food") for slot in range(6)], dim=1
        )
        assert torch.equal(vec_actions, oracle_actions)


def test_trainsim_state_dict_roundtrip_and_immutable_artifacts(tmp_path: Path) -> None:
    """Verify TrainSim state_dict restore reproduces identical rollouts and create-only IO."""
    sim_a = TrainSim.create(8, device="cpu", seed=99, max_capacity=256)
    gen = torch.Generator(device="cpu").manual_seed(31415)
    for _ in range(25):
        sim_a.step(torch.randint(0, 6, (8, 6), generator=gen, dtype=torch.long))

    saved = sim_a.state_dict()
    assert "bodies_live_i16" in saved
    assert saved["bodies_live_i16"].dtype == torch.int16
    full_i64_bytes = 8 * 6 * 256 * 2 * 8
    assert saved["bodies_live_i16"].numel() * 2 < full_i64_bytes // 10
    sim_b = TrainSim.create(8, device="cpu", seed=1, max_capacity=256)
    sim_b.load_state_dict(saved)

    for _ in range(30):
        act = torch.randint(0, 6, (8, 6), generator=gen, dtype=torch.long)
        info_a = sim_a.step(act)
        info_b = sim_b.step(act)
        assert torch.equal(info_a.reward, info_b.reward)
        assert torch.equal(sim_a.prev_heads, sim_b.prev_heads)
        assert torch.equal(sim_a.prev_alive, sim_b.prev_alive)

    out_json = tmp_path / "sample.json"
    digest = write_create_only_json(out_json, {"ok": True})
    assert verify_completed_artifact(out_json) == digest
    with pytest.raises(FileExistsError):
        write_create_only_artifact(out_json, b"{}")


def test_b1_preload_native_sim_extensions_monkeypatched_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression B1: preload_native_sim_extensions calls move_cuda._extension & event_cuda load."""
    calls: list[str] = []

    class _FakeCudaDevice:
        type = "cuda"
        index = 0

    monkeypatch.setattr(arena_runtime, "ensure_cuda_initialized", lambda dev: calls.append("init"))
    monkeypatch.setattr(arena_runtime, "synchronize_device", lambda dev: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "Fake L4")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))

    monkeypatch.setattr(move_cuda, "_extension", lambda: calls.append("move_ext") or object())
    monkeypatch.setattr(event_cuda, "_EXTENSION", None)

    import torch.utils.cpp_extension as cpp_ext

    def _fake_load(
        *,
        name: str,
        sources: list[str],
        extra_cuda_cflags: list[str],
        extra_cflags: list[str],
        verbose: bool,
    ) -> Any:
        calls.append(f"load:{name}")
        assert any(src.endswith("event_cuda.cpp") for src in sources)
        assert any(src.endswith("event_cuda.cu") for src in sources)
        return {"loaded": name}

    monkeypatch.setattr(cpp_ext, "load", _fake_load)

    meta = preload_native_sim_extensions(_FakeCudaDevice())  # type: ignore[arg-type]
    assert meta["arch"] == "8.9"
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"
    assert "move_ext" in calls
    assert "load:snake_dqn_a0_event_v1" in calls
    assert event_cuda._EXTENSION == {"loaded": "snake_dqn_a0_event_v1"}
    monkeypatch.setattr(event_cuda, "_EXTENSION", None)


def test_minor4_step_info_does_not_alias_live_sim_tensors() -> None:
    """Regression MINOR 4: StepInfo.survived and post_length do not alias live sim state."""
    sim = TrainSim.create(4, device="cpu", seed=123, max_capacity=128)
    sim.prepare()
    # Force snake (0, 0) onto right border facing right so stepping straight kills it on the wall.
    sim.sim.bodies[0, 0, sim.sim.head_ptr[0, 0]] = torch.tensor(
        [sim.grid_w - 1, 10], dtype=torch.long
    )
    sim.sim.direction[0, 0] = 1
    sim.sim.length[0, 0] = 9

    info = sim.step(torch.ones((4, 6), dtype=torch.long))
    assert not bool(info.survived[0, 0].item())
    assert int(info.post_length[0, 0].item()) == 9
    survived_snapshot = info.survived.clone()
    length_snapshot = info.post_length.clone()

    # Subsequent prepare/step decrements respawn timer and respawns seat (0, 0) in-place!
    sim.step(torch.ones((4, 6), dtype=torch.long))
    sim.prepare()
    assert bool(sim.sim.alive[0, 0].item()), "seat (0, 0) should have respawned in live sim"
    assert int(sim.sim.length[0, 0].item()) == 1

    # StepInfo from the earlier step must remain completely unmutated.
    assert torch.equal(info.survived, survived_snapshot)
    assert torch.equal(info.post_length, length_snapshot)


def test_minor5_capacity_overflow_checked_and_saved() -> None:
    """Regression MINOR 5: capacity_overflow is in _SIM_STATE_TENSORS, saved, and raises."""
    assert "capacity_overflow" in _SIM_STATE_TENSORS
    sim = TrainSim.create(2, device="cpu", seed=7, max_capacity=6)
    sim.assert_no_capacity_overflow()
    sim.prepare()
    sim.sim.length[:] = 10
    for _ in range(8):
        sim.step(torch.ones((2, 6), dtype=torch.long))

    assert bool(sim.sim.capacity_overflow.any().item())
    with pytest.raises(RuntimeError, match="capacity_overflow"):
        sim.assert_no_capacity_overflow()

    saved = sim.state_dict()
    sim2 = TrainSim.create(2, device="cpu", seed=8, max_capacity=6)
    sim2.assert_no_capacity_overflow()
    sim2.load_state_dict(saved)
    with pytest.raises(RuntimeError, match="capacity_overflow"):
        sim2.assert_no_capacity_overflow()


def test_minor8_indexed_vs_reference_collision_parity() -> None:
    """Regression MINOR 8: verify_indexed_vs_reference_parity checks 4x100 mature worlds."""
    report = verify_indexed_vs_reference_parity("cpu", num_envs=4, ticks=100, seed=20261110)
    assert report["status"] == "PASS"
    assert report["total_deaths"] > 0
    assert report["total_boosts"] > 0


def test_signal_stop_flag_real_sigusr1() -> None:
    """Verify SignalStopFlag catches a real os.kill(os.getpid(), SIGUSR1)."""
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("SIGUSR1 unavailable on platform")
    with SignalStopFlag() as stop_flag:
        assert not stop_flag.stopped
        os.kill(os.getpid(), signal.SIGUSR1)
        assert stop_flag.stopped
        assert stop_flag.signal_name == "SIGUSR1"


def test_parameterized_board_size_and_snake_count_trainsim() -> None:
    """Coordinator #7: verify custom (grid_h=48, grid_w=64, num_snakes=4) parity vs DeviceSim."""
    cfg = arena_v2_config(8, grid_h=48, grid_w=64, num_snakes=4, max_capacity=256)
    assert cfg.game_height == 480
    assert cfg.game_width == 640
    assert cfg.num_snakes == 4

    train_sim = TrainSim.create(
        8, device="cpu", seed=4242, grid_h=48, grid_w=64, num_snakes=4, max_capacity=256
    )
    ref = DeviceSim(
        cfg,
        device="cpu",
        train_mode=False,
        allow_respawn=True,
        cap=cfg.max_capacity,
        move_backend="reference",
        hero_respawn=True,
        exact_mechanics=True,
    )
    for name in _SIM_STATE_TENSORS:
        if hasattr(train_sim.sim, name):
            setattr(ref, name, getattr(train_sim.sim, name).clone())
    ref._prepared = bool(train_sim.sim._prepared)

    gen = torch.Generator(device="cpu").manual_seed(909)
    for _ in range(40):
        act = torch.randint(0, 6, (8, 4), generator=gen, dtype=torch.long)
        ref.step(act)
        info = train_sim.step(act)
        assert info.reward.shape == (8, 4)
        ref.prepare()
        train_sim.prepare()
        vec_g = train_sim.scripted_greedy_food_actions(chunk_envs=3)
        oracle_g = torch.stack(
            [choose_anchor(ref, slot, "greedy_food") for slot in range(4)], dim=1
        )
        assert torch.equal(vec_g, oracle_g)


def test_minor8_r3_reset_worlds_matches_fresh_create_and_prepare_exact() -> None:
    """MINOR 8: reset_worlds matches fresh create()+prepare() bit-for-bit incl. frame-1 food."""
    # Tight 12x12 board with 4 snakes so initial food placement & frame-1 _maintain_food interact
    sim = TrainSim.create(4, device="cpu", seed=2026, grid_h=12, grid_w=12, num_snakes=4)
    gen = torch.Generator(device="cpu").manual_seed(8080)
    sim.prepare()
    sim.sim.length[:] = 8
    sim.sim._reward_prev_length.copy_(sim.sim.length)
    for _ in range(12):
        # Include boost actions (3..5) so _boosted_this_step, _last_died, _last_ate get dirty
        sim.step(torch.randint(0, 6, (4, 4), generator=gen, dtype=torch.long))
    sim.prepare()

    # Force dirty step flags before reset_worlds to verify they are cleared on reset rows
    sim.sim._boosted_this_step[:] = True
    sim.sim._last_done[:] = True
    sim.sim._last_kills[:] = 5

    non_hit_before = {
        name: getattr(sim.sim, name)[[1, 3]].clone()
        for name in _SIM_STATE_TENSORS
        if getattr(sim.sim, name).ndim >= 1 and getattr(sim.sim, name).shape[0] == 4
    }

    hit = torch.tensor([True, False, True, False], dtype=torch.bool)
    pre_keys = sim.sim.rng_key[[0, 2]].clone()
    pre_counters = sim.sim.rng_counter[[0, 2]].clone()
    sim.reset_worlds(hit)

    # Build a fresh reference TrainSim initialized with those pre-reset RNG streams + prepare()
    fresh = TrainSim.create(2, device="cpu", seed=1, grid_h=12, grid_w=12, num_snakes=4)
    fresh.sim.rng_key.copy_(pre_keys)
    fresh.sim.rng_counter.copy_(pre_counters)
    fresh.sim.alive.fill_(False)
    fresh.sim.respawn_timer.zero_()
    fresh.sim.seg_count.zero_()
    fresh.sim.length.fill_(1)
    fresh.sim.food_present.fill_(False)
    fresh.sim.corpse_food.fill_(False)
    fresh.sim.food_count.zero_()
    fresh.sim._respawn_exact()
    fresh.sim._refill_food_rows(torch.ones((2,), dtype=torch.bool))
    fresh.sim.frame.zero_()
    fresh.sim._prepared = False
    fresh.prepare()

    for name in _SIM_STATE_TENSORS:
        got = getattr(sim.sim, name)
        exp = getattr(fresh.sim, name)
        if got.ndim >= 1 and got.shape[0] == 4:
            assert torch.equal(
                got[[0, 2]], exp
            ), f"reset_worlds mismatch vs create()+prepare() on {name}"
            assert torch.equal(got[[1, 3]], non_hit_before[name]), f"non-hit rows mutated on {name}"

    assert not bool(sim.prev_alive[[0, 2]].any().item())
    assert int(sim.prev_heads[[0, 2]].abs().sum().item()) == 0
    assert not bool(sim.sim._boosted_this_step[[0, 2]].any().item())


def test_minor11a_r3_state_dict_gathers_live_bodies_on_device_before_cpu_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MINOR 11a: TrainSim.state_dict must gather/cast to int16 BEFORE calling Tensor.cpu()."""
    sim = TrainSim.create(4, device="cpu", seed=71, max_capacity=512)
    sim.prepare()
    for _ in range(5):
        sim.step(torch.ones((4, 6), dtype=torch.long))

    orig_cpu = torch.Tensor.cpu
    copied_shapes_dtypes: list[tuple[tuple[int, ...], torch.dtype]] = []

    def _spy_cpu(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        copied_shapes_dtypes.append((tuple(self.shape), self.dtype))
        return orig_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", _spy_cpu)
    sd = sim.state_dict()
    assert sd["bodies_live_i16"].shape[2] < 32
    # Full int64 bodies tensor has shape (4, 6, 512, 2) and dtype int64; must NEVER hit .cpu()!
    assert ((4, 6, 512, 2), torch.int64) not in copied_shapes_dtypes


def test_late_state_reservoir_harvest_restore_and_exact_mechanics_continuation() -> None:
    """Feature (c): GPU reservoir harvests >= threshold worlds and restored worlds step exact."""
    sim = TrainSim.create(6, device="cpu", seed=20261110, max_capacity=256)
    reservoir = LateStateReservoir(
        max_size=4,
        min_length=600,
        num_snakes=sim.S,
        max_capacity=sim.cap,
        food_capacity=sim.sim.food_capacity,
        device=sim.device,
        seed=99,
    )
    assert reservoir.nbytes() > 0

    sim.prepare()
    # Below threshold (max length = 599): nothing harvested
    sim.sim.length[:] = 12
    sim.sim.length[0, 0] = 599
    assert reservoir.harvest(sim) == 0
    assert reservoir.size == 0

    # Step several ticks with mature lengths & boosts so corpses, trail food, and ring ptrs are rich
    gen = torch.Generator(device="cpu").manual_seed(12345)
    for _ in range(8):
        sim.step(torch.randint(0, 6, (6, 6), generator=gen, dtype=torch.long))
    sim.prepare()
    sim.sim.length[1, 2] = 600
    sim.sim.length[4, 0] = 725
    sim.sim._reward_prev_length.copy_(sim.sim.length)

    added = reservoir.harvest(sim)
    assert added == 2
    assert reservoir.size == 2

    # Round-trip reservoir state_dict
    res_sd = reservoir.state_dict()
    reservoir_reloaded = LateStateReservoir(
        max_size=4,
        min_length=600,
        num_snakes=sim.S,
        max_capacity=sim.cap,
        food_capacity=sim.sim.food_capacity,
        device=sim.device,
        seed=1,
    )
    reservoir_reloaded.load_state_dict(res_sd)
    assert reservoir_reloaded.size == 2

    non_hit_snapshot = {
        name: getattr(sim.sim, name)[[0, 2, 5]].clone()
        for name in _SIM_STATE_TENSORS
        if getattr(sim.sim, name).ndim >= 1 and getattr(sim.sim, name).shape[0] == 6
    }

    hit = torch.tensor([False, True, False, True, True, False], dtype=torch.bool)
    n_restored, n_fresh = reservoir_reloaded.reset_or_restore_worlds(sim, hit, restore_prob=1.0)
    assert (n_restored, n_fresh) == (3, 0)
    assert int((sim.sim.length[[1, 3, 4]].max(dim=1).values >= 600).sum().item()) == 3
    assert (sim.sim.frame[[1, 3, 4]] == 1).all()

    for name, snap in non_hit_snapshot.items():
        assert torch.equal(getattr(sim.sim, name)[[0, 2, 5]], snap), f"non-hit mutated on {name}"

    # Prove restored worlds continue with bit-exact DeviceSim mechanics over 40 ticks
    ref_cfg = arena_v2_config(6, max_capacity=256)
    ref = DeviceSim(
        ref_cfg,
        device="cpu",
        train_mode=False,
        allow_respawn=True,
        cap=256,
        move_backend="reference",
        hero_respawn=True,
        exact_mechanics=True,
    )
    for name in _SIM_STATE_TENSORS:
        setattr(ref, name, getattr(sim.sim, name).clone())
    ref._prepared = bool(sim.sim._prepared)

    for tick in range(40):
        act = torch.randint(0, 6, (6, 6), generator=gen, dtype=torch.long)
        ref.step(act)
        sim.step(act)
        if (tick + 1) % 10 == 0:
            for name in _SIM_STATE_TENSORS:
                exp = getattr(ref, name)
                got = getattr(sim.sim, name)
                if exp.is_floating_point():
                    assert torch.allclose(exp, got, atol=1e-6, rtol=0.0)
                else:
                    assert torch.equal(
                        exp, got
                    ), f"Restored world mechanics diverged on {name}@{tick}"
