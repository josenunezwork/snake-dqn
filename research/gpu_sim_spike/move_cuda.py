"""MOVE indexed kernels and a CPU portability twin; never a silent CUDA fallback.

The RNG and resolver match the frozen device engine, including its intentional
counter-RNG difference from GridBatchSim. CPU portability is not native proof.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.mechanics_constants import HEADON_SIZE_RATIO
from src.simd_env.batch_sim import DEATH_BODY, DEATH_HEAD, DEATH_SELF, DEATH_WALL

_EXTENSION: Any = None


def _extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        _EXTENSION = load(
            name="snake_dqn_move_v1",
            sources=[str(root / "move_cuda.cpp"), str(root / "move_cuda.cu")],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXTENSION


def _respawn_fields(sim: Any) -> list[torch.Tensor]:
    return [
        getattr(sim, name)
        for name in (
            "bodies",
            "head_ptr",
            "seg_count",
            "length",
            "alive",
            "direction",
            "boost_frames",
            "frames_since_food",
            "respawn_timer",
            "_reward_prev_length",
            "terminal_slots",
            "rng_counter",
            "rng_key",
        )
    ]


def respawn(sim: Any) -> None:
    """Build occupancy once only in ready worlds, preserving all 32 RNG draws."""
    if sim.device.type == "cuda":
        occupancy = torch.empty(
            (sim.E, sim.grid_h * sim.grid_w), dtype=torch.int32, device=sim.device
        )
        _extension().respawn(_respawn_fields(sim), occupancy, sim.grid_w, sim.grid_h)
        return
    if sim.device.type != "cpu":
        raise ValueError("indexed MOVE supports CPU or CUDA only")
    # NumPy views alias CPU tensor storage; no tensor state is replaced.
    arrays = [tensor.numpy() for tensor in _respawn_fields(sim)]
    (
        body,
        ptr,
        count,
        length,
        alive,
        direction,
        boost,
        hunger,
        timer,
        prev,
        terminal,
        counter,
        key,
    ) = arrays
    for e in range(sim.E):
        timer[e] -= ((~alive[e]) & (timer[e] > 0)).astype(np.int64)
        ready = ~alive[e] & (timer[e] <= 0) & ~terminal[e]
        old_counter = int(counter[e])
        # Inactive worlds still advance the exact counter, with int64 wrap.
        counter[e] = ((old_counter + 32 + (1 << 63)) % (1 << 64)) - (1 << 63)
        if not ready.any():
            continue
        occ = np.zeros((sim.grid_h, sim.grid_w), dtype=np.bool_)
        for j in range(sim.S):
            if alive[e, j]:
                for k in range(min(int(count[e, j]), sim.cap)):
                    x, y = body[e, j, (int(ptr[e, j]) - k) % sim.cap]
                    if 0 <= x < sim.grid_w and 0 <= y < sim.grid_h:
                        occ[y, x] = True
        placed = np.zeros(sim.S, dtype=np.bool_)
        for trial in range(32):
            z = (
                int(key[e]) * 6364136223846793005 + (old_counter + trial + 1) * 1442695040888963407
            ) & ((1 << 63) - 1)
            bx, by = z % sim.grid_w, (z // sim.grid_w) % sim.grid_h
            candidates = [
                ((bx + j) % sim.grid_w, (by + j // sim.grid_w) % sim.grid_h) for j in range(sim.S)
            ]
            # Uniqueness is against ALL earlier slots, even ineligible ones.
            for j, (x, y) in enumerate(candidates):
                if ready[j] and not placed[j] and not occ[y, x] and (x, y) not in candidates[:j]:
                    body[e, j, 0] = (x, y)
                    ptr[e, j] = 0
                    count[e, j] = length[e, j] = prev[e, j] = 1
                    alive[e, j] = True
                    direction[e, j] = 1
                    boost[e, j] = hunger[e, j] = timer[e, j] = 0
                    placed[j] = True
                    occ[y, x] = True


def collisions(
    sim: Any,
    head0: torch.Tensor,
    traversed: torch.Tensor,
    valid: torch.Tensor,
    in_arena: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inspect ring entries in parallel, then resolve each world in fixed order."""
    if sim.device.type == "cuda":
        return tuple(
            _extension().collisions(
                [
                    sim.bodies,
                    sim.head_ptr,
                    sim.seg_count,
                    sim.length,
                    sim.alive,
                    head0.contiguous(),
                    traversed.contiguous(),
                    valid.contiguous(),
                    in_arena.contiguous(),
                ],
                sim.v2,
                float(HEADON_SIZE_RATIO),
            )
        )
    if sim.device.type != "cpu":
        raise ValueError("indexed MOVE supports CPU or CUDA only")
    body, ptr, count, length, alive = (
        value.numpy() for value in (sim.bodies, sim.head_ptr, sim.seg_count, sim.length, sim.alive)
    )
    h0, path, active, arena = (value.numpy() for value in (head0, traversed, valid, in_arena))
    death = np.zeros((sim.E, sim.S), dtype=np.int64)
    rank_out = np.full_like(death, sim.S * sim.S + sim.S)
    dead = np.zeros((sim.E, sim.S), dtype=np.bool_)
    kills, mass = np.zeros_like(death), np.zeros_like(death)
    for e in range(sim.E):
        self_hit = np.zeros(sim.S, dtype=np.bool_)
        hp = np.zeros((sim.S, sim.S), dtype=np.bool_)
        bp = hp.copy()
        for i in range(sim.S):
            for j in range(sim.S):
                if alive[e, j]:
                    rings = (int(ptr[e, j]) - np.arange(min(int(count[e, j]), sim.cap))) % sim.cap
                    cells = body[e, j, rings]
                    for t in range(2):
                        if active[e, i, t]:
                            hits = np.all(cells == path[e, i, t], axis=-1)
                            if i == j:
                                self_hit[i] |= hits[3:].any()
                            else:
                                bp[i, j] |= hits.any()
                if i != j:
                    for t in range(2):
                        for u in range(2):
                            hp[i, j] |= bool(
                                active[e, i, t]
                                and active[e, j, u]
                                and np.array_equal(path[e, i, t], path[e, j, u])
                            )
                            hp[i, j] |= bool(
                                np.array_equal(h0[e, i], path[e, j, u])
                                and np.array_equal(path[e, i, t], h0[e, j])
                            )
        rank = 0
        for i in range(sim.S):
            for hit, cause in ((not arena[e, i], DEATH_WALL), (self_hit[i], DEATH_SELF)):
                if alive[e, i] and not dead[e, i] and hit:
                    death[e, i], rank_out[e, i], dead[e, i] = cause, rank, True
                rank += 1
            for j in range(sim.S):
                if i == j:
                    continue
                eligible = alive[e, i] and not dead[e, i]
                head = eligible and alive[e, j] and not dead[e, j] and hp[i, j]
                bh = (
                    eligible
                    and (sim.v2 or (alive[e, j] and not dead[e, j]))
                    and not hp[i, j]
                    and bp[i, j]
                )
                iw = bool(
                    head
                    and sim.v2
                    and np.float32(length[e, i])
                    >= np.float32(HEADON_SIZE_RATIO) * np.float32(length[e, j])
                )
                jw = bool(
                    head
                    and sim.v2
                    and np.float32(length[e, j])
                    >= np.float32(HEADON_SIZE_RATIO) * np.float32(length[e, i])
                )
                mutual = head and not iw and not jw
                if bh or mutual or jw:
                    death[e, i], rank_out[e, i], dead[e, i] = (
                        DEATH_BODY if bh else DEATH_HEAD,
                        rank,
                        True,
                    )
                if mutual or iw:
                    death[e, j], rank_out[e, j], dead[e, j] = DEATH_HEAD, rank, True
                kills[e, i] += int(iw)
                kills[e, j] += int(jw or bh)
                mass[e, i] += int(length[e, j]) if iw else 0
                mass[e, j] += int(length[e, i]) if jw or bh else 0
                rank += 1
    return tuple(torch.from_numpy(value) for value in (death, rank_out, dead, kills, mass))
