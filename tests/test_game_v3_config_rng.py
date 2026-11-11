"""Unit tests for GameV3Config and stateless counter-based RNG parity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from research.game_v3.config import GameV3Config
from research.game_v3.rng import (
    PURPOSE_AMBIENT_FOOD,
    PURPOSE_INIT_FOOD,
    PURPOSE_INIT_SNAKE,
    PURPOSE_RESPAWN,
    PURPOSE_SCRIPTED,
    rng_angle_f32_np,
    rng_angle_f32_torch,
    rng_int_np,
    rng_int_torch,
    rng_u63_np,
    rng_u63_torch,
    rng_uniform_f32_np,
    rng_uniform_f32_torch,
)


def test_config_defaults_and_json_roundtrip() -> None:
    """Default config matches spec constants, is frozen, and round-trips via JSON."""
    cfg = GameV3Config()
    assert cfg.arena_radius == 600.0
    assert cfg.num_snakes == 24
    assert cfg.cell_size == 1.0
    assert cfg.speed == 1.0
    assert cfg.boost_speed == 2.0
    assert cfg.turn_rate == 0.12
    assert cfg.head_on_tie_margin == 0.15
    assert cfg.obs_raster_size == 64
    assert cfg.obs_scales == (1.0, 4.0, 16.0)
    assert cfg.grid_size % 16 == 0
    assert cfg.half_grid % 16 == 0

    with pytest.raises(FrozenInstanceError):
        cfg.arena_radius = 800.0  # type: ignore[misc]

    custom = GameV3Config(
        arena_radius=1200.0,
        num_snakes=48,
        turn_rate=0.10,
        min_boost_length=20,
        corpse_mass_fraction=0.6,
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


def test_body_radius_and_turn_rate_monotonicity() -> None:
    """Body radius grows with length up to cap and turn rate shrinks mildly."""
    cfg = GameV3Config()
    assert pytest.approx(cfg.turn_rate_f64(cfg.start_length), rel=1e-12) == cfg.turn_rate
    assert pytest.approx(cfg.body_radius_f64(150), rel=1e-12) == 2.0
    assert cfg.body_radius_f64(50) > cfg.body_radius_f64(10)
    assert cfg.body_radius_f64(100_000) == cfg.max_body_radius
    assert cfg.turn_rate_f64(200) < cfg.turn_rate_f64(cfg.start_length)


@pytest.mark.parametrize(
    "purpose",
    [
        PURPOSE_INIT_SNAKE,
        PURPOSE_INIT_FOOD,
        PURPOSE_AMBIENT_FOOD,
        PURPOSE_RESPAWN,
        PURPOSE_SCRIPTED,
    ],
)
def test_counter_rng_exact_parity_numpy_vs_torch_cpu_and_mps(purpose: int) -> None:
    """Counter RNG matches draw-for-draw across NumPy, Torch CPU, and Torch MPS."""
    seeds_np = np.array([0, 1, 42, 999999937, (1 << 60) + 17], dtype=np.int64)[:, None]
    ticks_np = np.array([0, 1, 57, 999, 123456], dtype=np.int64)[:, None]
    slots_np = np.arange(64, dtype=np.int64)[None, :]

    u63_np = rng_u63_np(seeds_np, ticks_np, purpose, slots_np)
    ri_np = rng_int_np(seeds_np, ticks_np, purpose, slots_np, 1216)
    uf_np = rng_uniform_f32_np(seeds_np, ticks_np, purpose, slots_np)
    ang_np = rng_angle_f32_np(seeds_np, ticks_np, purpose, slots_np)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for dev in devices:
        seeds_th = torch.as_tensor(seeds_np, dtype=torch.int64, device=dev)
        ticks_th = torch.as_tensor(ticks_np, dtype=torch.int64, device=dev)
        slots_th = torch.as_tensor(slots_np, dtype=torch.int64, device=dev)

        u63_th = rng_u63_torch(seeds_th, ticks_th, purpose, slots_th).cpu().numpy()
        ri_th = rng_int_torch(seeds_th, ticks_th, purpose, slots_th, 1216).cpu().numpy()
        uf_th = rng_uniform_f32_torch(seeds_th, ticks_th, purpose, slots_th).cpu().numpy()
        ang_th = rng_angle_f32_torch(seeds_th, ticks_th, purpose, slots_th).cpu().numpy()

        np.testing.assert_array_equal(u63_np, u63_th)
        np.testing.assert_array_equal(ri_np, ri_th)
        np.testing.assert_array_equal(uf_np.view(np.uint32), uf_th.view(np.uint32))
        np.testing.assert_array_equal(ang_np.view(np.uint32), ang_th.view(np.uint32))
        # Scalar tick and scalar slot parity check.
        scalar_th = rng_u63_torch(seeds_th[:, 0], 57, purpose, 19).cpu().numpy()
        scalar_np = rng_u63_np(seeds_np[:, 0], 57, purpose, 19)
        np.testing.assert_array_equal(scalar_np, scalar_th)
        assert np.all((uf_np >= 0.0) & (uf_np < 1.0))
        assert np.all((ang_np >= -np.float32(np.pi)) & (ang_np < np.float32(np.pi)))
