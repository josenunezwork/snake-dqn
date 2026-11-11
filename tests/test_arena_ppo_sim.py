"""Tests for `research.arena_ppo.sim` (`TrainSim` exact v2 dynamics, masks, opponents)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.arena_ppo.runtime import (
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.arena_ppo.sim import (
    _SIM_STATE_TENSORS,
    TrainSim,
    arena_v2_config,
    derive_arena_world_seeds,
)
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
    """Verify TrainSim state_dict restore reproduces identical future rollouts and create-only IO."""
    sim_a = TrainSim.create(8, device="cpu", seed=99, max_capacity=256)
    gen = torch.Generator(device="cpu").manual_seed(31415)
    for _ in range(25):
        sim_a.step(torch.randint(0, 6, (8, 6), generator=gen, dtype=torch.long))

    saved = sim_a.state_dict()
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
