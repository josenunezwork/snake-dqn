"""Stateless counter-based RNG keyed by (world_seed, tick, purpose, slot).

Both the PyTorch batched simulator and the plain-NumPy single-world reference
use this exact SplitMix64-style 4-key integer bijection so every random draw
agrees bit-for-bit across NumPy, Torch CPU, Torch MPS, and Torch CUDA.
"""

from __future__ import annotations

import numpy as np
import torch

PURPOSE_INIT_SNAKE: int = 1
PURPOSE_INIT_FOOD: int = 2
PURPOSE_AMBIENT_FOOD: int = 3
PURPOSE_RESPAWN: int = 4
PURPOSE_SCRIPTED: int = 5

# Signed 64-bit two's-complement representations of SplitMix64 / Murmur constants:
#   0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0xC6BC279692B5C323.
_K1: int = -7046029254386353131
_K2: int = -4658895280553007687
_K3: int = -7723592293110705685
_K4: int = -4132994306676758123

_INV_2POW24_F32: float = float(np.float32(1.0 / 16777216.0))


def _s64_mul_scalar(a: int, b: int) -> int:
    """Multiply two Python ints with exact two's-complement signed int64 wraparound."""
    u = (a * b) & 0xFFFF_FFFF_FFFF_FFFF
    return u - (1 << 64) if u >= (1 << 63) else u


def _term_torch(val: torch.Tensor | int, k: int) -> torch.Tensor | int:
    """Multiply a Torch int64 tensor or Python int by constant k in signed int64."""
    if isinstance(val, torch.Tensor):
        return val * k
    return _s64_mul_scalar(int(val), k)


def rng_u63_torch(
    world_seed: torch.Tensor,
    tick: torch.Tensor | int,
    purpose: int,
    slot: torch.Tensor | int,
) -> torch.Tensor:
    """Return deterministic positive 63-bit int64 draws on any Torch device.

    Args:
        world_seed: Int64 seed tensor broadcastable with ``tick`` and ``slot``.
        tick: Int64 tensor or Python int tick index.
        purpose: Integer stream purpose tag.
        slot: Int64 tensor or Python int draw slot index.

    Returns:
        Int64 tensor in ``[0, 2**63 - 1]``.
    """
    z = (
        world_seed * _K1
        + _term_torch(tick, _K2)
        + _s64_mul_scalar(int(purpose), _K3)
        + _term_torch(slot, _K4)
    )
    z = (z ^ ((z >> 30) & 0x3_FFFF_FFFF)) * _K2
    z = (z ^ ((z >> 27) & 0x1F_FFFF_FFFF)) * _K3
    z = z ^ ((z >> 31) & 0x1_FFFF_FFFF)
    return z & 0x7FFF_FFFF_FFFF_FFFF


def rng_int_torch(
    world_seed: torch.Tensor,
    tick: torch.Tensor | int,
    purpose: int,
    slot: torch.Tensor | int,
    high: int,
) -> torch.Tensor:
    """Return deterministic int32 draws in ``[0, high)`` via 31-bit multiply-shift."""
    u63 = rng_u63_torch(world_seed, tick, purpose, slot)
    u31 = (u63 >> 32) & 0x7FFF_FFFF
    return ((u31 * int(high)) >> 31).to(torch.int32)


def rng_angle_idx_torch(
    world_seed: torch.Tensor,
    tick: torch.Tensor | int,
    purpose: int,
    slot: torch.Tensor | int,
    angle_steps: int = 4096,
) -> torch.Tensor:
    """Return deterministic int32 heading angle indices in ``[0, angle_steps)``."""
    return rng_int_torch(world_seed, tick, purpose, slot, int(angle_steps))


def rng_uniform_f32_torch(
    world_seed: torch.Tensor,
    tick: torch.Tensor | int,
    purpose: int,
    slot: torch.Tensor | int,
) -> torch.Tensor:
    """Return deterministic float32 draws in ``[0.0, 1.0)`` using 24 mantissa bits."""
    u63 = rng_u63_torch(world_seed, tick, purpose, slot)
    u24 = ((u63 >> 39) & 0xFF_FFFF).to(torch.float32)
    return u24 * _INV_2POW24_F32


def rng_u63_np(
    world_seed: np.ndarray | int,
    tick: np.ndarray | int,
    purpose: int,
    slot: np.ndarray | int,
) -> np.ndarray:
    """Return deterministic positive 63-bit int64 draws in NumPy."""
    s64 = np.asarray(world_seed, dtype=np.int64)
    t64 = np.asarray(tick, dtype=np.int64)
    p64 = np.int64(purpose)
    sl64 = np.asarray(slot, dtype=np.int64)
    k1 = np.int64(_K1)
    k2 = np.int64(_K2)
    k3 = np.int64(_K3)
    k4 = np.int64(_K4)
    with np.errstate(over="ignore"):
        z = s64 * k1 + t64 * k2 + p64 * k3 + sl64 * k4
        z = (z ^ ((z >> np.int64(30)) & np.int64(0x3_FFFF_FFFF))) * k2
        z = (z ^ ((z >> np.int64(27)) & np.int64(0x1F_FFFF_FFFF))) * k3
        z = z ^ ((z >> np.int64(31)) & np.int64(0x1_FFFF_FFFF))
        return z & np.int64(0x7FFF_FFFF_FFFF_FFFF)


def rng_int_np(
    world_seed: np.ndarray | int,
    tick: np.ndarray | int,
    purpose: int,
    slot: np.ndarray | int,
    high: int,
) -> np.ndarray:
    """Return deterministic int32 draws in ``[0, high)`` in NumPy."""
    u63 = rng_u63_np(world_seed, tick, purpose, slot)
    u31 = (u63 >> np.int64(32)) & np.int64(0x7FFF_FFFF)
    with np.errstate(over="ignore"):
        return ((u31 * np.int64(high)) >> np.int64(31)).astype(np.int32)


def rng_angle_idx_np(
    world_seed: np.ndarray | int,
    tick: np.ndarray | int,
    purpose: int,
    slot: np.ndarray | int,
    angle_steps: int = 4096,
) -> np.ndarray:
    """Return deterministic int32 heading angle indices in ``[0, angle_steps)`` in NumPy."""
    return rng_int_np(world_seed, tick, purpose, slot, int(angle_steps))


def rng_uniform_f32_np(
    world_seed: np.ndarray | int,
    tick: np.ndarray | int,
    purpose: int,
    slot: np.ndarray | int,
) -> np.ndarray:
    """Return deterministic float32 draws in ``[0.0, 1.0)`` in NumPy."""
    u63 = rng_u63_np(world_seed, tick, purpose, slot)
    u24 = ((u63 >> np.int64(39)) & np.int64(0xFF_FFFF)).astype(np.float32)
    return (u24 * np.float32(_INV_2POW24_F32)).astype(np.float32)
