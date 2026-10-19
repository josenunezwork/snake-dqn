"""Fixed-shape, one-slab B8 inference; CUDA is separately qualified."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from research.t33_batched_teacher.observations import ObservationBuffer


class _HalfRasterForward(nn.Module):
    """Keep CUDA raster normalization fp16 while preserving fp32 scalars."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        local = local.to(torch.float16) * (1.0 / 255.0)
        glob = glob.to(torch.float16) * (1.0 / 255.0)
        fused = self.net.fuse(
            torch.cat((self.net.local(local), self.net.glob(glob), scalars.float()), dim=1)
        )
        advantage = self.net.adv(fused)
        return self.net.value(fused) + advantage - advantage.mean(dim=1, keepdim=True)


class FixedInference:
    """Own immutable device weights and one persistent fixed-shape input slab.

    CPU/MPS use eager fp32. CUDA requests fp16 autocast plus reduce-overhead
    compilation by default; compilation errors propagate, never become timings
    falsely described as compiled. Instances bind to one resident row shape.
    """

    def __init__(
        self,
        net: nn.Module,
        device: str = "cpu",
        *,
        compile_cuda: bool = True,
        half_cuda: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        self.half = self.device.type == "cuda" and half_cuda
        self.compiled = self.device.type == "cuda" and compile_cuda
        self.net = copy.deepcopy(net).to(self.device).eval()
        model = _HalfRasterForward(self.net) if self.half else self.net
        self.model = (
            torch.compile(model, mode="reduce-overhead", dynamic=False, fullgraph=True)
            if self.compiled
            else model
        )
        self.rows: int | None = None
        self.slab: torch.Tensor | None = None
        self.vector_models: dict[tuple[str, int], Any] = {}
        self.counters: dict[str, float | int] = {}
        self.reset_profile()

    def reset_profile(self) -> None:
        """Exclude first-use JIT, compilation, and warmup from timed receipts."""
        self.counters = {
            "calls": 0,
            "h2d_s": 0.0,
            "gpu_s": 0.0,
            "inference_wall_s": 0.0,
            "vector_calls": 0,
            "vector_h2d_s": 0.0,
            "vector_gpu_s": 0.0,
        }

    def profile(self) -> dict[str, Any]:
        """Return event timings and effective execution mode."""
        return {
            **self.counters,
            "device": str(self.device),
            "rows": self.rows,
            "precision": "fp16-autocast" if self.half else "fp32",
            "compile_mode": "reduce-overhead" if self.compiled else "eager",
        }

    @torch.inference_mode()
    def q(self, buffer: ObservationBuffer) -> np.ndarray:
        """Return CPU Q values after one nonblocking CUDA H2D slab copy."""
        if self.rows is None:
            self.rows = buffer.rows
        elif self.rows != buffer.rows:
            raise ValueError("FixedInference cannot change resident batch shape")
        started = time.perf_counter()
        events = None
        if self.device.type == "cuda":
            if not buffer.storage.is_pinned():
                raise ValueError("CUDA inference requires a pinned ObservationBuffer")
            if self.slab is None:
                self.slab = torch.empty(
                    buffer.storage.numel(), dtype=torch.uint8, device=self.device
                )
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            self.slab.copy_(buffer.storage, non_blocking=True)
            events[1].record()
            views = buffer.tensor_views(self.slab)
        elif self.device.type == "mps":
            if self.slab is None:
                self.slab = torch.empty(
                    buffer.storage.numel(), dtype=torch.uint8, device=self.device
                )
            self.slab.copy_(buffer.storage)
            views = buffer.tensor_views(self.slab)
        else:
            views = buffer.tensor_views()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half):
            q = self.model(views["local"], views["global"], views["scalars"])
        if events:
            events[2].record()
        result = q.float().cpu().numpy().copy()
        if events:
            events[2].synchronize()
            self.counters["h2d_s"] += events[0].elapsed_time(events[1]) / 1000
            self.counters["gpu_s"] += events[1].elapsed_time(events[2]) / 1000
        self.counters["calls"] += 1
        self.counters["inference_wall_s"] += time.perf_counter() - started
        return result

    @torch.inference_mode()
    def vector_q(self, policy: Any, states: np.ndarray) -> np.ndarray:
        """Batch frozen Vector61 opponents with a fixed shape per checkpoint."""
        # The teacher has already verified checkpoint identity when grouping.
        identity = str(getattr(policy, "checkpoint_path", id(policy)))
        key = (identity, len(states))
        if key not in self.vector_models:
            model = copy.deepcopy(policy.network).to(self.device).eval()
            if self.compiled:
                model = torch.compile(model, mode="reduce-overhead", dynamic=False, fullgraph=True)
            host = torch.empty(
                states.shape, dtype=torch.float32, pin_memory=self.device.type == "cuda"
            )
            self.vector_models[key] = (model, host, torch.empty_like(host, device=self.device))
        model, host, target = self.vector_models[key]
        host.numpy()[:] = states
        events = None
        if self.device.type == "cuda":
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
        target.copy_(host, non_blocking=self.device.type == "cuda")
        if events:
            events[1].record()
        # Opponents remain fp32: the requested fp16 screen qualifies B8 only.
        result = model(target)
        if events:
            events[2].record()
        result = result.float().cpu().numpy().copy()
        if events:
            events[2].synchronize()
            self.counters["vector_h2d_s"] += events[0].elapsed_time(events[1]) / 1000
            self.counters["vector_gpu_s"] += events[1].elapsed_time(events[2]) / 1000
        self.counters["vector_calls"] += 1
        return result


def agreement(reference: np.ndarray, candidate: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    """Describe drift without silently accepting failed or undersized screens."""
    if reference.shape != candidate.shape or reference.shape != masks.shape:
        raise ValueError("Q and mask shapes must agree")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite Q values invalidate inference qualification")
    allowed = masks.astype(bool).copy()
    allowed[~allowed.any(axis=1), :3] = True
    ref = np.where(allowed, reference, -np.inf)
    cand = np.where(allowed, candidate, -np.inf)
    same = ref.argmax(axis=1) == cand.argmax(axis=1)
    raw_same = reference.argmax(axis=1) == candidate.argmax(axis=1)
    count = len(reference)
    if not count:
        raise ValueError("empty qualification sample")
    ranked = np.sort(ref, axis=1)
    margins = ranked[:, -1] - ranked[:, -2]
    bad = np.flatnonzero(~same)
    return {
        "states": count,
        "masked_matches": int(same.sum()),
        "masked_agreement": float(same.mean()),
        "raw_matches": int(raw_same.sum()),
        "raw_agreement": float(raw_same.mean()),
        "max_abs_q_error": float(np.max(np.abs(reference - candidate))),
        "mismatch_examples": [
            {
                "row": int(i),
                "reference": int(ref[i].argmax()),
                "candidate": int(cand[i].argmax()),
                "reference_margin": float(margins[i]),
            }
            for i in bad[:32]
        ],
        "threshold_pass": bool(count >= 1000 and same.mean() >= 0.995 and raw_same.mean() >= 0.995),
    }
