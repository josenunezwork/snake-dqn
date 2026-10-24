"""Lazy native FIFO for actual corpse candidates; CUDA runs never silently fall back."""

from pathlib import Path
from typing import Any

import torch

_EXTENSION: Any = None
_LAUNCH_DIAGNOSTICS = False
_CAPTURE_LAUNCH_EVENTS = False
_EVENT_POOL_SIZE = 256
_EVENT_POOL: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_PENDING_LAUNCHES: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]] = []
_EVENT_CURSOR = 0

_FIFO_SCOPE = "gpu_sim.native.event.fifo"


def set_launch_diagnostics(enabled: bool, reset: bool = False, capture_events: bool = True) -> None:
    """Enable separate same-stream CUDA-event timing around native FIFO launches.

    This is intentionally opt-in and used only by the A1-PROF diagnostic pass.
    Plain timing ticks never create these events or NVTX ranges.  Callers must
    drain the launch stream once before consuming :func:`get_launch_diagnostics`.
    """
    global _CAPTURE_LAUNCH_EVENTS, _LAUNCH_DIAGNOSTICS, _EVENT_CURSOR
    if type(enabled) is not bool or type(reset) is not bool or type(capture_events) is not bool:
        raise TypeError("launch diagnostics flags must be bool")
    if reset:
        _PENDING_LAUNCHES.clear()
        _EVENT_CURSOR = 0
    if enabled and capture_events and not _EVENT_POOL:
        # Event allocation happens before the diagnostic tick.  A fixed pool
        # makes accidental per-launch allocator work an explicit failure.
        _EVENT_POOL.extend(
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(_EVENT_POOL_SIZE)
        )
        # Force lazy CUDA-event handle allocation before the diagnostic tick.
        # Re-recording the same events for launches is valid after this drain.
        for start, end in _EVENT_POOL:
            start.record()
            end.record()
        torch.cuda.synchronize()
    _LAUNCH_DIAGNOSTICS = enabled
    _CAPTURE_LAUNCH_EVENTS = enabled and capture_events


def get_launch_diagnostics() -> list[dict[str, Any]]:
    """Resolve FIFO launch events after the caller's single tick-end drain."""
    rows = [
        {"name": name, "count": 1, "elapsed_ms": start.elapsed_time(end), "stream": stream}
        for name, stream, start, end in _PENDING_LAUNCHES
    ]
    _PENDING_LAUNCHES.clear()
    return rows


def append_fifo(
    tag: torch.Tensor,
    queue: torch.Tensor,
    count: torch.Tensor,
    cells: torch.Tensor,
    mask: torch.Tensor,
    accepted: torch.Tensor,
    old_food: int,
    cap: int,
    corpse: bool,
) -> None:
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        _EXTENSION = load(
            name="snake_dqn_a0_event_v1",
            sources=[str(root / "event_cuda.cpp"), str(root / "event_cuda.cu")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    global _EVENT_CURSOR
    # Materialize arguments before the launch event: any defensive contiguous
    # clone is ordinary framework work, never falsely charged to FIFO.
    contiguous_cells, contiguous_mask = cells.contiguous(), mask.contiguous()
    start = end = None
    if _CAPTURE_LAUNCH_EVENTS:
        # Event.record() uses PyTorch's current stream, the exact stream the
        # extension obtains with getCurrentCUDAStream below.
        if _EVENT_CURSOR >= len(_EVENT_POOL):
            raise RuntimeError("event CUDA launch diagnostic event pool exhausted")
        start, end = _EVENT_POOL[_EVENT_CURSOR]
        _EVENT_CURSOR += 1
        stream = torch.cuda.current_stream(tag.device)
        start.record(stream)
    if _LAUNCH_DIAGNOSTICS:
        torch.cuda.nvtx.range_push(_FIFO_SCOPE)
    try:
        _EXTENSION.append_fifo(
            tag,
            queue,
            count,
            contiguous_cells,
            contiguous_mask,
            accepted,
            old_food,
            cap,
            corpse,
            _LAUNCH_DIAGNOSTICS,
        )
        if end is not None:
            # event_cuda.cu obtains this same PyTorch current stream with
            # getCurrentCUDAStream(tag.get_device()) for its native launch.
            end.record(stream)
            _PENDING_LAUNCHES.append((_FIFO_SCOPE, int(stream.cuda_stream), start, end))
    finally:
        if _LAUNCH_DIAGNOSTICS:
            torch.cuda.nvtx.range_pop()


def selfcheck(output: Path) -> None:
    """Execute native ordered FIFO against frozen E1b on production-capacity states."""
    import hashlib

    from research.gpu_sim_spike.benchmark import write_receipt
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.gpu_sim_spike.device_sim_frozen import DeviceSim as FrozenDeviceSim
    from src.simd_env.batch_sim import BatchSimConfig

    if not torch.cuda.is_available():
        raise RuntimeError("native A0 oracle selfcheck requires CUDA")
    frozen_path = Path(__file__).with_name("device_sim_frozen.py")
    expected = "7fbabf67b439c6e8ebf2cbb252abed5dadcc1ce08194a8d6177605ab4bbe8d88"
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("A0 frozen oracle source identity mismatch")
    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=6,
        game_width=1450,
        game_height=830,
        max_food=300,
        max_capacity=10400,
        mechanics_version=2,
    )
    cpu = FrozenDeviceSim(cfg, device="cpu", allow_respawn=False)
    cuda = DeviceSim(cfg, device="cuda", allow_respawn=False)
    # Ambient prefix + full corpse FIFO. Native work must retain old ambient,
    # evict oldest corpses, ignore duplicates, then permit an evicted reinsert.
    initial = torch.tensor([[(i % 145, i // 145) for i in range(450)]]).repeat(2, 1, 1)
    for sim in (cpu, cuda):
        sim.food_cells[:, :450] = initial.to(sim.device)
        sim.food_present[:, :450] = True
        sim.corpse_food[:, 300:450] = True
        sim.food_count[:] = 450
    stream = [500 + i for i in range(2300)] + [300, 500, 2800, 300]
    cells = torch.tensor([[(i % 145, i // 145) for i in stream]]).repeat(2, 1, 1)
    mask = torch.ones((2, len(stream)), dtype=torch.bool)
    mask[1, 100:] = False  # per-world padding and unequal actual dead lengths
    cpu._append_many(cells, mask, True)
    # Production-shaped padded input; only actual masked segments reach FIFO.
    padded = torch.zeros((2, 6 * 10400, 2), dtype=torch.long, device="cuda")
    padded_mask = torch.zeros((2, 6 * 10400), dtype=torch.bool, device="cuda")
    padded[:, : cells.shape[1]] = cells.to("cuda")
    padded_mask[:, : cells.shape[1]] = mask.to("cuda")
    cuda._append_many(padded, padded_mask, True)
    torch.cuda.synchronize()
    for name in ("food_cells", "food_present", "corpse_food", "food_count", "rng_counter"):
        left, right = getattr(cpu, name), getattr(cuda, name).cpu()
        if left.numpy().tobytes() != right.numpy().tobytes():
            raise RuntimeError(f"A0 native FIFO oracle mismatch: {name}")
    write_receipt(
        output,
        {
            "schema": "a0-native-fifo-oracle/v1",
            "status": "PASS",
            "production_capacity": 10400,
            "baseline_sha256": expected,
            "actual_candidates": mask.sum(1).tolist(),
            "padded_candidates": 62400,
            "device": torch.cuda.get_device_name(0),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).with_suffix(".cpp"),
                    Path(__file__).with_suffix(".cu"),
                )
            },
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selfcheck(args.output)
