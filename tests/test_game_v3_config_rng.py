"""Unit tests for GameV3Config (v3.1), shared integer tables, and counter RNG parity."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from research.game_v3.config import (
    FP_ONE,
    MAX_OCC_SUBTICK,
    MAX_SNAKES,
    NUM_ACTIONS,
    OCC_OWNER_BITS,
    OCC_OWNER_MASK,
    TRIG_ONE,
    GameV3Config,
    build_length_tables_np,
    build_trig_tables_np,
    decode_occ_owner_hi,
    decode_occ_owner_lo,
    decode_occ_subtick,
    occ_entry_active,
    pack_occ_hi,
    pack_occ_lo,
)
from research.game_v3.rng import (
    PURPOSE_AMBIENT_FOOD,
    PURPOSE_INIT_FOOD,
    PURPOSE_INIT_SNAKE,
    PURPOSE_KILL_CREDIT,
    PURPOSE_RESPAWN,
    PURPOSE_SCRIPTED,
    rng_angle_idx_np,
    rng_angle_idx_torch,
    rng_int_np,
    rng_int_torch,
    rng_u63_np,
    rng_u63_torch,
    rng_uniform_f32_np,
    rng_uniform_f32_torch,
)


def test_config_defaults_and_json_roundtrip() -> None:
    """Default v3.1 config matches spec constants, removes size margin, and round-trips JSON."""
    cfg = GameV3Config()
    assert cfg.arena_radius == 600.0
    assert cfg.num_snakes == 24
    assert cfg.cell_size == 1.0
    assert cfg.start_length == 10
    assert cfg.max_length == 2048
    assert cfg.spawn_wall_margin == 20.0
    assert cfg.max_life_ticks == 0
    assert cfg.respawn_candidates == 8
    assert cfg.angle_steps == 4096
    assert cfg.speed == 1.0
    assert cfg.boost_speed == 2.0
    assert cfg.turn_rate == 0.12
    assert cfg.turn_exponent == 0.24
    assert cfg.boost_drop_fraction == 0.5
    assert cfg.food_pickup_mult == 1.5
    assert cfg.obs_fine_size == 64
    assert cfg.obs_coarse_size == 32
    assert cfg.obs_scales == (1, 4, 16)
    assert cfg.num_actions == NUM_ACTIONS == 10
    assert not hasattr(cfg, "head_on_tie_margin")
    assert not hasattr(cfg, "head_on_ratio_bp")
    assert cfg.grid_size % 16 == 0
    assert cfg.half_grid % 16 == 0

    with pytest.raises(FrozenInstanceError):
        cfg.arena_radius = 800.0  # type: ignore[misc]

    custom = GameV3Config(
        arena_radius=1200.0,
        num_snakes=48,
        max_length=1536,
        turn_rate=0.10,
        turn_exponent=0.28,
        min_boost_length=20,
        boost_drop_fraction=0.4,
        food_pickup_mult=1.75,
        corpse_mass_fraction=0.6,
        max_life_ticks=1800,
    )
    payload = custom.to_json()
    restored = GameV3Config.from_json(payload)
    assert restored == custom
    assert hash(restored) == hash(custom)


def test_config_validation_bounds() -> None:
    """Out-of-range configuration fields raise ValueError."""
    with pytest.raises(ValueError):
        GameV3Config(arena_radius=2001.0)
    with pytest.raises(ValueError):
        GameV3Config(arena_radius=0.0)
    with pytest.raises(ValueError):
        GameV3Config(num_snakes=5)
    with pytest.raises(ValueError):
        GameV3Config(num_snakes=65)
    with pytest.raises(ValueError):
        GameV3Config(max_length=5, start_length=10)
    with pytest.raises(ValueError):
        GameV3Config(corpse_mass_fraction=1.5)
    with pytest.raises(ValueError):
        GameV3Config(boost_drop_fraction=1.2)
    with pytest.raises(ValueError):
        GameV3Config(food_pickup_mult=0.0)
    with pytest.raises(ValueError):
        GameV3Config(ambient_pellet_value=256)  # ambient_grid is uint8 [0, 255].
    with pytest.raises(ValueError):
        GameV3Config(angle_steps=4000)  # Must be power of 2.


def test_occupancy_bit_layout_pack_decode_and_derived_snake_cap() -> None:
    """Occupancy pack/decode helpers round-trip all 64 slots and derive MAX_SNAKES."""
    assert MAX_SNAKES == (1 << OCC_OWNER_BITS) == 64
    assert OCC_OWNER_MASK == MAX_SNAKES - 1 == 63
    assert GameV3Config(num_snakes=MAX_SNAKES).num_snakes == MAX_SNAKES
    with pytest.raises(ValueError, match=str(MAX_SNAKES)):
        GameV3Config(num_snakes=MAX_SNAKES + 1)

    slots = torch.arange(MAX_SNAKES, dtype=torch.int32)
    for st in (0, 1, 42, MAX_OCC_SUBTICK):
        hi = pack_occ_hi(st, slots)
        lo = pack_occ_lo(st, slots)
        assert hi.dtype == torch.int32 and lo.dtype == torch.int32
        assert bool((hi >= 0).all()) and bool((lo >= 0).all())
        assert torch.equal(decode_occ_owner_hi(hi), slots)
        assert torch.equal(decode_occ_owner_lo(lo), slots)
        assert torch.equal(decode_occ_subtick(hi), torch.full_like(slots, st))
        assert torch.equal(decode_occ_subtick(lo), torch.full_like(slots, st))
        assert bool(occ_entry_active(hi, st).all())
        assert bool(occ_entry_active(lo, st).all())
        if st < MAX_OCC_SUBTICK:
            assert not bool(occ_entry_active(hi, st + 1).any())
            assert not bool(occ_entry_active(lo, st + 1).any())

    empty = torch.full((MAX_SNAKES,), -1, dtype=torch.int32)
    assert not bool(occ_entry_active(empty, 0).any())


def test_body_radius_turn_slowdown_and_integer_tables() -> None:
    """Turn radius roughly triples from start_length=10 to length=1000; integer tables exact."""
    cfg = GameV3Config()
    assert pytest.approx(cfg.turn_rate_f64(cfg.start_length), rel=1e-12) == cfg.turn_rate
    assert pytest.approx(cfg.body_radius_f64(150), rel=1e-12) == 2.0
    assert cfg.body_radius_f64(50) > cfg.body_radius_f64(10)
    assert cfg.body_radius_f64(100_000) == cfg.max_body_radius

    q_start = cfg.turn_quantum(cfg.start_length)
    q_1000 = cfg.turn_quantum(1000)
    r_start = cfg.speed / (2.0 * math.sin(math.pi * q_start / cfg.angle_steps))
    r_1000 = cfg.speed / (2.0 * math.sin(math.pi * q_1000 / cfg.angle_steps))
    ratio = r_1000 / r_start
    assert 2.75 <= ratio <= 3.25, f"Expected turn radius ~3x at L=1000, got {ratio:.3f}x"

    sin_tbl, cos_tbl = build_trig_tables_np(cfg.angle_steps)
    assert sin_tbl.dtype == np.int32 and cos_tbl.dtype == np.int32
    q = cfg.angle_steps // 4
    assert (sin_tbl[0], cos_tbl[0]) == (0, TRIG_ONE)
    assert (sin_tbl[q], cos_tbl[q]) == (TRIG_ONE, 0)
    assert (sin_tbl[2 * q], cos_tbl[2 * q]) == (0, -TRIG_ONE)
    assert (sin_tbl[3 * q], cos_tbl[3 * q]) == (-TRIG_ONE, 0)

    lt = build_length_tables_np(cfg)
    assert int(lt["speed_fp"][10]) == FP_ONE
    assert int(lt["boost_speed_fp"][10]) == 2 * FP_ONE
    # 5 turn levels must be strictly monotonic [+full, +half, 0, -half, -full].
    for length in (10, 100, 1000, 2048):
        row = lt["turn_deltas"][length]
        assert row[0] > row[1] > row[2] == 0 > row[3] > row[4]
        assert row[0] == -row[4] and row[1] == -row[3]


@pytest.mark.parametrize(
    "purpose",
    [
        PURPOSE_INIT_SNAKE,
        PURPOSE_INIT_FOOD,
        PURPOSE_AMBIENT_FOOD,
        PURPOSE_RESPAWN,
        PURPOSE_SCRIPTED,
        PURPOSE_KILL_CREDIT,
    ],
)
def test_counter_rng_exact_parity_numpy_vs_torch_cpu_and_mps(purpose: int) -> None:
    """Counter RNG matches draw-for-draw across NumPy, Torch CPU, and Torch MPS."""
    seeds_np = np.array([0, 1, 42, 999999937, (1 << 60) + 17], dtype=np.int64)[:, None]
    ticks_np = np.array([0, 1, 57, 999, 123456], dtype=np.int64)[:, None]
    slots_np = np.arange(64, dtype=np.int64)[None, :]
    high_arr_np = (np.arange(64, dtype=np.int64) % 7 + 1)[None, :]

    u63_np = rng_u63_np(seeds_np, ticks_np, purpose, slots_np)
    ri_np = rng_int_np(seeds_np, ticks_np, purpose, slots_np, 1216)
    ri_var_np = rng_int_np(seeds_np, ticks_np, purpose, slots_np, high_arr_np)
    uf_np = rng_uniform_f32_np(seeds_np, ticks_np, purpose, slots_np)
    ang_np = rng_angle_idx_np(seeds_np, ticks_np, purpose, slots_np, 4096)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for dev in devices:
        seeds_th = torch.as_tensor(seeds_np, dtype=torch.int64, device=dev)
        ticks_th = torch.as_tensor(ticks_np, dtype=torch.int64, device=dev)
        slots_th = torch.as_tensor(slots_np, dtype=torch.int64, device=dev)
        high_th = torch.as_tensor(high_arr_np, dtype=torch.int32, device=dev)

        u63_th = rng_u63_torch(seeds_th, ticks_th, purpose, slots_th).cpu().numpy()
        ri_th = rng_int_torch(seeds_th, ticks_th, purpose, slots_th, 1216).cpu().numpy()
        ri_var_th = rng_int_torch(seeds_th, ticks_th, purpose, slots_th, high_th).cpu().numpy()
        uf_th = rng_uniform_f32_torch(seeds_th, ticks_th, purpose, slots_th).cpu().numpy()
        ang_th = rng_angle_idx_torch(seeds_th, ticks_th, purpose, slots_th, 4096).cpu().numpy()

        np.testing.assert_array_equal(u63_np, u63_th)
        np.testing.assert_array_equal(ri_np, ri_th)
        np.testing.assert_array_equal(ri_var_np, ri_var_th)
        np.testing.assert_array_equal(uf_np.view(np.uint32), uf_th.view(np.uint32))
        np.testing.assert_array_equal(ang_np, ang_th)
        scalar_th = rng_u63_torch(seeds_th[:, 0], 57, purpose, 19).cpu().numpy()
        scalar_np = rng_u63_np(seeds_np[:, 0], 57, purpose, 19)
        np.testing.assert_array_equal(scalar_np, scalar_th)
        assert np.all((uf_np >= 0.0) & (uf_np < 1.0))
        assert np.all((ang_np >= 0) & (ang_np < 4096))
