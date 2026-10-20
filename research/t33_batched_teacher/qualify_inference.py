"""Collect >=1000 calibration states and qualify the requested B8 device path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.phase0_search_20261019 import bank, teacher
from research.t33_batched_teacher.ablation import B8, B8_SHA, calibration_rows, sha
from research.t33_batched_teacher.evidence import identity
from research.t33_batched_teacher.inference import FixedInference, agreement
from research.t33_batched_teacher.observations import ObservationBuffer


def collect(checkpoint: Path, count: int) -> tuple[Any, dict[str, np.ndarray], np.ndarray]:
    """Sample actual, disjoint calibration trajectories at prepared Watch seams."""
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    chunks: dict[str, list] = {k: [] for k in ("local", "global", "scalars")}
    masks = []
    net = None
    rows = calibration_rows()
    for index, row in enumerate(rows):
        wanted = count // len(rows) + (index < count % len(rows))
        world = teacher.build_world(row, checkpoint)
        net = world.hero.net
        sampled = 0

        class Sample:
            def choose(self, prepared: Any) -> tuple[int, dict]:
                nonlocal sampled
                obs = build_ego_raster(
                    prepared.sim,
                    EgoRasterConfig(version="bv8"),
                    backend="numba",
                    rows=np.array([[0, 0]]),
                )
                for key in chunks:
                    chunks[key].append(obs[key].copy())
                q, legal, select = teacher._q_and_legal(prepared)
                if not select.any():
                    select[:3] = True
                masks.append(select.copy())
                sampled += 1
                return teacher._mask_argmax(q, select), {"searched": False, "scores": {}}

        for frame in range(5000):
            if sampled >= wanted or not world.sim.alive[0, 0]:
                break
            teacher.step_world(world, Sample() if frame % 3 == 0 else None)
        if sampled != wanted:
            raise RuntimeError(f"calibration trajectory ended before {wanted} states")
    return net, {key: np.concatenate(values) for key, values in chunks.items()}, np.asarray(masks)


def qualify(
    net: Any, obs: dict[str, np.ndarray], masks: np.ndarray, *, device: str, batch_size: int = 64
) -> dict:
    """Compare fixed-shape inference to eager fp32 on the same device and states."""
    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    candidate = FixedInference(net, device)
    eager = FixedInference(net, device, compile_cuda=False, half_cuda=False)
    buffer = ObservationBuffer(batch_size, pinned=device == "cuda")
    reference, actual = [], []
    started = time.perf_counter()
    for offset in range(0, len(masks), batch_size):
        count = min(batch_size, len(masks) - offset)
        buffer.zero_()
        for key in obs:
            buffer.arrays[key][:count] = obs[key][offset : offset + count]
        reference.append(eager.q(buffer)[:count])
        actual.append(candidate.q(buffer)[:count])
    metrics = agreement(np.concatenate(reference), np.concatenate(actual), masks)
    cuda = device == "cuda"
    return {
        **metrics,
        "cuda_qualification": cuda and metrics["threshold_pass"],
        "status": (
            ("CUDA_PASS" if metrics["threshold_pass"] else "CUDA_FAIL")
            if cuda
            else "FALLBACK_ONLY_CUDA_UNVERIFIED"
        ),
        "device": device,
        "batch_size": batch_size,
        "seconds": time.perf_counter() - started,
        "effective": candidate.profile(),
        "torch": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=B8)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.states < 1000:
        parser.error("qualification requires at least 1000 states")
    if sha(args.checkpoint) != B8_SHA:
        raise ValueError("B8 checkpoint hash mismatch")
    teacher.configure_cpu()
    net, obs, masks = collect(args.checkpoint, args.states)
    result = qualify(net, obs, masks, device=args.device, batch_size=args.batch_size)
    receipt = {
        "schema": "t33-inference-qualification/v1",
        "checkpoint_sha256": B8_SHA,
        "platform": platform.platform(),
        "scope": "disjoint-calibration-inference-only",
        "state_sha256": {
            k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in {**obs, "masks": masks}.items()
        },
        "source": identity(),
        **result,
    }
    json.dumps(receipt, allow_nan=False)
    bank.write_create(args.out, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("source", "state_sha256")}))
    return 0 if args.device != "cuda" or result["cuda_qualification"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
