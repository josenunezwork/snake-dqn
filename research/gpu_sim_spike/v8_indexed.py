"""Optional native indexed construction for the device v8 feature bundle.

The normal :mod:`v8_device` path remains the CPU oracle and default CUDA path.
With ``GPU_SIM_V8_INDEXED=1`` on CUDA, this module reads each ring in native
code instead of materialising ``(row, slot, cap, xy)`` and six post-move rings.
It deliberately constructs only the union/release tensors consumed by the
unchanged v8 decision code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_EXTENSION: object | None = None
_ERROR: Exception | None = None


def enabled() -> bool:
    """Return the explicit experiment opt-in; the conservative default is off."""
    return os.environ.get("GPU_SIM_V8_INDEXED", "0") == "1"


def _load() -> Any:
    global _ERROR, _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if _ERROR is not None:
        raise RuntimeError("v8 indexed CUDA extension previously failed to load") from _ERROR
    try:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        _EXTENSION = load(
            name="snake_dqn_v8_indexed_v1",
            sources=[str(root / "v8_indexed.cpp"), str(root / "v8_indexed.cu")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=os.environ.get("GPU_SIM_SPIKE_CUDA_VERBOSE") == "1",
        )
    except Exception as exc:  # pragma: no cover - requires a CUDA compiler
        _ERROR = exc
        raise RuntimeError("v8 indexed CUDA extension could not be loaded") from exc
    return _EXTENSION


def construct(sim: Any, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Build v8 occupancy, post-tail, and release maps by direct ring reads.

    Returns ``all_occ, other_occ, post_tail_occ, release, lpost, boosted``.
    The caller retains the frozen Python decision/ranking and clears each new
    head from ``release`` exactly as before.  This function is CUDA-only by
    design, making accidental CPU-oracle substitution impossible.
    """
    if rows.device.type != "cuda":
        raise ValueError("v8 indexed construction is CUDA-only")
    all_occ, other_occ, post_tail_occ, release, lpost, boosted, status = _load().construct(
        sim.bodies.contiguous(),
        sim.head_ptr.contiguous(),
        sim.seg_count.contiguous(),
        sim.length.contiguous(),
        sim.alive.contiguous(),
        sim.direction.contiguous(),
        sim.boost_frames.contiguous(),
        rows.contiguous(),
        int(sim.grid_h),
        int(sim.grid_w),
        int(sim.cfg.min_boost_length),
        int(sim.cfg.boost_length_cost_frames),
    )
    torch._assert_async(status.eq(0), "v8 indexed ring construction rejected invalid input")
    return all_occ.bool(), other_occ.bool(), post_tail_occ.bool(), release, lpost, boosted.bool()
