"""Native CUDA admission point for E1 flood kernels.

The CPU oracle deliberately stays in :mod:`ego_device` and :mod:`v8_device`.
CUDA is different: an E1 run must use the compiled kernels, rather than quietly
falling back to the Python wavefront when a compiler is missing.  Loading is
lazy so CPU and FakeTensor tests do not need a CUDA toolkit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

_EXTENSION: object | None = None
_ERROR: Exception | None = None
_DIAGNOSTICS_ENABLED = False


@contextmanager
def _native_nvtx(label: str):
    """Mirror the optional native record-function scope without plain-tick cost."""
    if _DIAGNOSTICS_ENABLED and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def available() -> bool:
    """Return whether the native kernels can be loaded on this host."""
    try:
        _load()
    except RuntimeError:
        return False
    return True


def _load() -> object:
    """Build/load the extension once, retaining the actionable compiler error."""
    global _EXTENSION, _ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _ERROR is not None:
        raise RuntimeError(_message()) from _ERROR
    if not torch.cuda.is_available() or torch.version.cuda is None:
        _ERROR = RuntimeError("PyTorch was not built with an available CUDA runtime")
        raise RuntimeError(_message()) from _ERROR
    try:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        # Do not use an implicit Python/Torch fallback here.  The generated
        # module is the evidence that a CUDA E1 benchmark exercised the native
        # queue/union-find implementation.
        _EXTENSION = load(
            name="snake_dqn_e1_flood_cuda_v2",
            sources=[str(root / "flood_cuda.cpp"), str(root / "flood_cuda.cu")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=os.environ.get("GPU_SIM_SPIKE_CUDA_VERBOSE") == "1",
        )
    except Exception as exc:  # pragma: no cover - exercised by L4 preflight
        _ERROR = exc
        raise RuntimeError(_message()) from exc
    return _EXTENSION


def _message() -> str:
    return (
        "E1 CUDA flood kernels are required for a CUDA run but could not be loaded; "
        "install a CUDA-enabled PyTorch matching nvcc and ninja, then rerun the preflight. "
        f"original error: {_ERROR}"
    )


def flood(
    start: torch.Tensor,
    blocked: torch.Tensor,
    cap: torch.Tensor,
    release: torch.Tensor | None,
    max_cap: int = 4096,
) -> torch.Tensor:
    """Run one sparse FIFO frontier per ``(world, slot, action)`` on CUDA."""
    if release is None:
        release = torch.empty((0,), dtype=torch.int64, device=start.device)
    label = f"gpu_sim.native.flood.sparse_frontier_cap{int(max_cap)}"
    with _native_nvtx(label):
        result, status = _load().flood(start, blocked, cap, release, int(max_cap))
    torch._assert_async(status.eq(0), "E1 native frontier watchdog, queue, or lane failure")
    return result


def components(
    allowed: torch.Tensor,
    heads: torch.Tensor,
    direction: torch.Tensor,
    alive: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Run tiled union-find region sizing for the three ego action starts."""
    with _native_nvtx("gpu_sim.native.flood.components"):
        result, status = _load().components(allowed, heads, direction, alive, rows)
    torch._assert_async(status.eq(0), "E1 native union-find watchdog or invalid-root failure")
    return result


def static_capped(start: torch.Tensor, occupied: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    """Count indexed static components without expanding occupancy by slot/action.

    ``start`` is ``(E,S,3,2)``, ``occupied`` is world-shaped ``(E,H,W)``, and
    each lane's cap is ``(E,S)``.  This specialized path deliberately has no
    release map or distance state; callers needing either must retain ``flood``.
    """
    with _native_nvtx("gpu_sim.native.flood.sparse_frontier_cap160"):
        result, status = _load().static_capped(
            start.contiguous(), occupied.contiguous(), cap.contiguous()
        )
    torch._assert_async(status.eq(0), "E2 static capped frontier watchdog or queue failure")
    return result


def set_launch_diagnostics(enabled: bool, reset: bool = False, capture_events: bool = True) -> None:
    """Enable optional actual-stream native-launch event accounting."""
    global _DIAGNOSTICS_ENABLED
    _DIAGNOSTICS_ENABLED = bool(enabled)
    _load().set_launch_diagnostics(bool(enabled), bool(reset), bool(capture_events))


def get_launch_diagnostics() -> list[dict[str, object]]:
    """Drain completed native-launch event records after the diagnostic tick."""
    return list(_load().get_launch_diagnostics())


def _selfcheck(output: Path | None = None) -> None:
    """Compile and execute a small native CUDA frontier proof before E1 cost work."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--selfcheck requires CUDA; CPU preflight must not claim native readiness"
        )
    # A winding path crosses the 256-thread initialization tile.  The expected
    # count is derived here rather than relying on the candidate Torch flood.
    h, w = 19, 23
    blocked = torch.ones((1, 1, h, w), dtype=torch.bool, device="cuda")
    blocked[0, 0, 2, 1:22] = False
    blocked[0, 0, 3:18, 21] = False
    blocked[0, 0, 17, 2:22] = False
    result = flood(
        torch.tensor([[[1, 2]]], device="cuda"),
        blocked,
        torch.tensor([[4096]], device="cuda"),
        None,
        4096,
    )

    # Count independently on CPU; do not reuse the candidate's Torch flood.
    def count(start_xy: tuple[int, int], blocked_cpu: torch.Tensor) -> int:
        from collections import deque

        todo, seen = deque([start_xy]), {start_xy}
        while todo:
            x, y = todo.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and not bool(blocked_cpu[0, 0, ny, nx])
                    and (nx, ny) not in seen
                ):
                    seen.add((nx, ny))
                    todo.append((nx, ny))
        return len(seen)

    expected = count((1, 2), blocked.cpu())
    if int(result.cpu()[0, 0]) != expected:
        raise RuntimeError(
            f"native E1 flood selfcheck mismatch: {int(result.cpu()[0, 0])} != {expected}"
        )
    # A release-rejected cell must remain unseen.  A longer route reaches this
    # cell at its eligible distance, proving the atomic dedupe happens after
    # the release check rather than at the rejected early attempt.
    retry_blocked = torch.tensor([[[[False, False, False], [False, False, True]]]], device="cuda")
    retry_release = torch.tensor([[[[0, 3, 0], [0, 0, 0]]]], device="cuda", dtype=torch.long)
    retry = flood(
        torch.tensor([[[0, 0]]], device="cuda"),
        retry_blocked,
        torch.tensor([[4096]], device="cuda"),
        retry_release,
        4096,
    )
    if int(retry.cpu()[0, 0]) != 5:
        raise RuntimeError("native E1 release-retry selfcheck mismatch")
    # Also force production geometry/caps through allocation and dispatch.  It
    # catches dynamic-memory/launch failures before the expensive CPU label
    # phase, without claiming this microcheck is a throughput measurement.
    for cap in (160, 4096):
        wide_start = torch.tensor([[[0, 0]]], device="cuda")
        wide_blocked = torch.zeros((1, 1, 83, 145), dtype=torch.bool, device="cuda")
        wide = flood(wide_start, wide_blocked, torch.tensor([[cap]], device="cuda"), None, cap)
        if int(wide.cpu()[0, 0]) != cap:
            raise RuntimeError(f"native E1 cap selfcheck mismatch at cap={cap}")
    # E2 vector61 path: one world occupancy grid is indexed by every slot and
    # action.  These cases cover cap 32/160, blocked and OOB starts, a winding
    # corridor, and repeated/disconnected queries without reusing the kernel.
    static_open = torch.ones((1, 83, 145), dtype=torch.bool, device="cuda")
    static_open[0, 2, 1:82] = False
    static_open[0, 3:80, 81] = False
    static_open[0, 79, 2:82] = False
    static_start = torch.tensor(
        [[[[1, 2], [1, 2], [-1, 2]], [[80, 79], [0, 0], [144, 82]]]],
        dtype=torch.long,
        device="cuda",
    )
    static_cap = torch.tensor([[32, 160]], dtype=torch.long, device="cuda")
    static_got = static_capped(static_start, static_open, static_cap).cpu().numpy()
    static_blocked = static_open.cpu().numpy()[0]

    def static_count(start_xy: tuple[int, int], cap_value: int) -> int:
        from collections import deque

        sx, sy = start_xy
        if not (0 <= sx < 145 and 0 <= sy < 83) or static_blocked[sy, sx]:
            return 0
        todo, seen = deque([(sx, sy)]), {(sx, sy)}
        while todo and len(seen) < cap_value:
            x, y = todo.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (
                    0 <= nx < 145
                    and 0 <= ny < 83
                    and not static_blocked[ny, nx]
                    and (nx, ny) not in seen
                ):
                    seen.add((nx, ny))
                    todo.append((nx, ny))
                    if len(seen) == cap_value:
                        break
        return len(seen)

    static_expected = np.array(
        [
            [
                [static_count((1, 2), 32), static_count((1, 2), 32), 0],
                [static_count((80, 79), 160), 0, 0],
            ]
        ],
        dtype=np.int64,
    )
    if not np.array_equal(static_got, static_expected):
        raise RuntimeError(
            f"native E2 static capped mismatch: {static_got.tolist()} "
            f"!= {static_expected.tolist()}"
        )
    # Two worlds deliberately share starts but not occupancy.  The open world
    # proves exact 159/160/161 component boundaries and cap saturation; the
    # blocked world proves queries are indexed by world, never broadcast from
    # its neighbour.  Slots retain independent cap32/cap160 contracts.
    boundary_occ = torch.zeros((2, 19, 23), dtype=torch.bool, device="cuda")
    boundary_occ[1, :, 3] = True
    boundary_start = torch.tensor(
        [
            [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]],
            [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]],
        ],
        dtype=torch.long,
        device="cuda",
    )
    boundary_cap = torch.tensor([[32, 160], [32, 160]], dtype=torch.long, device="cuda")
    boundary_got = static_capped(boundary_start, boundary_occ, boundary_cap).cpu().numpy()
    boundary_expected = np.array(
        [[[32, 32, 32], [160, 160, 160]], [[32, 32, 32], [57, 57, 57]]], dtype=np.int64
    )
    if not np.array_equal(boundary_got, boundary_expected):
        raise RuntimeError(f"native E2 indexed/cap selfcheck mismatch: {boundary_got.tolist()}")
    for component_size, expected_size in ((159, 159), (160, 160), (161, 160)):
        line_occ = torch.zeros((1, 1, component_size), dtype=torch.bool, device="cuda")
        line_start = torch.zeros((1, 1, 3, 2), dtype=torch.long, device="cuda")
        line_cap = torch.tensor([[160]], dtype=torch.long, device="cuda")
        line_got = static_capped(line_start, line_occ, line_cap).cpu().numpy()
        if not bool((line_got == expected_size).all()):
            raise RuntimeError(
                f"native E2 boundary mismatch size={component_size}: " f"{line_got.tolist()}"
            )
    # Two disconnected components cross 16-cell tiles.  The repeated rows
    # prove row gathering is separate from one-per-world component labelling.
    allowed = torch.zeros((1, 19, 23), dtype=torch.bool, device="cuda")
    allowed[0, 1:18, 1:11] = True
    allowed[0, 2:5, 15:20] = True
    heads = torch.tensor([[[0, 9]]], dtype=torch.long, device="cuda")
    direction = torch.tensor([[1]], dtype=torch.long, device="cuda")
    alive = torch.tensor([[True]], dtype=torch.bool, device="cuda")
    rows = torch.tensor([[0, 0], [0, 0]], dtype=torch.long, device="cuda")
    component = components(allowed, heads, direction, alive, rows).cpu()
    # Only straight starts at (1, 9), inside the 17*10 rectangle.
    if not bool((component[:, 1] == 170).all()) or not bool((component[:, (0, 2)] == 0).all()):
        raise RuntimeError(f"native E1 union-find selfcheck mismatch: {component.tolist()}")
    production_component = components(
        torch.ones((1, 83, 145), dtype=torch.bool, device="cuda"),
        torch.tensor([[[0, 0]]], dtype=torch.long, device="cuda"),
        direction,
        alive,
        rows[:1],
    ).cpu()
    if production_component.tolist() != [[0, 83 * 145, 83 * 145]]:
        raise RuntimeError("native E1 production-size component selfcheck mismatch")
    receipt = {
        "kind": "e1_native_cuda_flood_selfcheck",
        "status": "OK",
        "device": torch.cuda.get_device_name(),
        "production_geometry": [83, 145],
        "caps": [160, 4096],
        "independent_fifo_expected": expected,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("flood_cuda.cpp"),
                Path(__file__).with_name("flood_cuda.cu"),
            )
        },
    }
    if output is not None:
        # Reuse the benchmark's immutable sidecar writer only after native
        # compilation has succeeded; importing it has no CUDA side effects.
        from research.gpu_sim_spike.benchmark import write_receipt

        write_receipt(output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - L4 launcher entrypoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.selfcheck:
        parser.error("pass --selfcheck")
    _selfcheck(args.output)
