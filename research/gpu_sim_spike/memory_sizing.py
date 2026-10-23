"""CPU-only tensor-size arithmetic for the A1 vector61 memory repair.

Run with ``./venv/bin/python -m research.gpu_sim_spike.memory_sizing``.  This
does not create a CUDA context or allocate the represented tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

GIB = 1 << 30
# A1's S0 long-world fixture comes from the production 1450 x 830 arena with
# 10-pixel cells. Capacity 10,400 is the body ring, not board area.
A1_GRID_W = 145
A1_GRID_H = 83


def tensor_bytes(*shape: int, itemsize: int) -> int:
    """Return contiguous tensor storage bytes without allocating a tensor."""
    total = itemsize
    for dim in shape:
        total *= dim
    return total


@dataclass(frozen=True)
class VectorScratch:
    """Largest vector61 feature scratch tensors at one row-block size."""

    rows: int
    snakes: int
    capacity: int

    @property
    def action_delta(self) -> int:
        return tensor_bytes(self.rows, 3, self.snakes, self.capacity, 2, itemsize=8)

    @property
    def body_delta(self) -> int:
        return tensor_bytes(self.rows, self.snakes, self.capacity, 2, itemsize=8)

    @property
    def action_distance(self) -> int:
        return tensor_bytes(self.rows, 3, self.snakes, self.capacity, itemsize=4)

    @property
    def body_distance(self) -> int:
        return tensor_bytes(self.rows, self.snakes, self.capacity, itemsize=4)

    @property
    def conservative_peak(self) -> int:
        """Conservative live-set bound for row-local long/float reductions.

        It retains four action-delta-sized long intermediates plus four
        body-delta-sized long intermediates and the two principal float
        distance arrays.  This is intentionally an allocation-plan bound, not
        a measured CUDA peak.
        """
        return (
            4 * self.action_delta + 4 * self.body_delta + self.action_distance + self.body_distance
        )


def a1_report(
    worlds: int = 1024, snakes: int = 6, capacity: int = 10400, chunk: int = 64
) -> dict[str, int]:
    """Return exact no-allocation byte counts for the A1 maximum shape."""
    rows = worlds * snakes
    full = VectorScratch(rows, snakes, capacity)
    blocked = VectorScratch(min(rows, chunk), snakes, capacity)
    # bool plane for (E, S, relative-action, H, W), excluding padded-neighbor
    # intermediates.
    bfs_plane = tensor_bytes(worlds, snakes, 3, A1_GRID_H, A1_GRID_W, itemsize=1)
    return {
        "rows": rows,
        "full_action_delta_bytes": full.action_delta,
        "full_body_delta_bytes": full.body_delta,
        "full_action_distance_bytes": full.action_distance,
        "chunk_action_delta_bytes": blocked.action_delta,
        "chunk_body_delta_bytes": blocked.body_delta,
        "chunk_conservative_row_scratch_bytes": blocked.conservative_peak,
        "bfs_bool_plane_bytes": bfs_plane,
    }


def legacy_named_locals(
    worlds: int = 1024, snakes: int = 6, capacity: int = 10400, food_capacity: int = 450
) -> dict[str, int]:
    """Named pre-repair locals that remained reachable near the OOM site.

    These are source-level storage sizes, not a claim that CUDA's allocator
    reports each one separately at one instruction.  In particular, temporary
    expressions and allocator blocks have different lifetimes.
    """
    rows = worlds * snakes
    plane = tensor_bytes(worlds, snakes, 3, A1_GRID_H, A1_GRID_W, itemsize=1)
    return {
        "body_view_cells": tensor_bytes(worlds, snakes, capacity, 2, itemsize=8),
        "row_body": tensor_bytes(rows, snakes, capacity, 2, itemsize=8),
        "danger_bdelta": tensor_bytes(rows, snakes, capacity, 2, itemsize=8),
        "danger_bdist": tensor_bytes(rows, snakes, capacity, itemsize=4),
        "danger_sector": tensor_bytes(rows, snakes, capacity, itemsize=8),
        "danger_values": tensor_bytes(rows, snakes, capacity, itemsize=4),
        "danger_mask": tensor_bytes(rows, snakes, capacity, itemsize=1),
        "danger_near": tensor_bytes(rows, snakes, capacity, itemsize=1),
        "food_onehot": tensor_bytes(rows, food_capacity, 16, itemsize=4),
        "bfs_plane": plane,
        "per_action_delta_request": tensor_bytes(rows, 3, snakes, capacity, 2, itemsize=8),
    }


def repaired_plan_bound(
    worlds: int = 1024, snakes: int = 6, capacity: int = 10400, chunk: int = 64
) -> dict[str, int]:
    """Conservative source-level simultaneous plan, excluding opaque kernels.

    The B8 and CUDA-workspace allowance is deliberately explicit rather than
    misrepresented as a measured allocation.  The ruler's actual peak receipt
    remains the fit authority.
    """
    block = VectorScratch(min(worlds * snakes, chunk), snakes, capacity)
    body = tensor_bytes(worlds, snakes, capacity, 2, itemsize=8)
    # Current flood keeps frontier/seen plus four padded neighbor candidates;
    # reserve eight planes for expression overlap and allocator rounding.
    bfs_allowance = 8 * tensor_bytes(worlds, snakes, 3, A1_GRID_H, A1_GRID_W, itemsize=1)
    return {
        "sim_and_template_bodies": 2 * body,
        "one_body_view": body,
        "chunk_row_scratch_conservative": block.conservative_peak,
        "bfs_expression_allowance": bfs_allowance,
        "b8_and_cuda_workspace_allowance": 2 * GIB,
    }


def legacy_pre_request_bytes(values: dict[str, int]) -> int:
    """Sum old-call locals that existed before the failed action-delta request."""
    return sum(
        value
        for name, value in values.items()
        if name not in {"bfs_plane", "per_action_delta_request"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--snakes", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=10400)
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()
    report = a1_report(args.worlds, args.snakes, args.capacity, args.chunk)
    legacy = legacy_named_locals(args.worlds, args.snakes, args.capacity)
    repaired = repaired_plan_bound(args.worlds, args.snakes, args.capacity, args.chunk)
    for name, value in report.items():
        if name == "rows":
            print(f"{name}={value}")
        else:
            print(f"{name}={value} ({value / GIB:.9f} GiB)")
    print("legacy_named_locals_gib=")
    for name, value in legacy.items():
        print(f"  {name}={value / GIB:.9f}")
    print(f"  subtotal_before_failed_request={legacy_pre_request_bytes(legacy) / GIB:.9f}")
    print(f"  failed_request={legacy['per_action_delta_request'] / GIB:.9f}")
    print(f"  bfs_later_stage_not_in_pre_request={legacy['bfs_plane'] / GIB:.9f}")
    print("repaired_plan_bound_gib=")
    for name, value in repaired.items():
        print(f"  {name}={value / GIB:.9f}")
    print(f"  subtotal={sum(repaired.values()) / GIB:.9f}")


if __name__ == "__main__":
    main()
