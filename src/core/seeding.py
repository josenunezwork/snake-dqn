"""Deterministic run seeding with stable named child streams."""

from __future__ import annotations

import hashlib
import random
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


def _seed_bytes(seed: int, stream: str) -> bytes:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("Seeds must be unsigned 64-bit integers")
    if not stream or not isinstance(stream, str):
        raise ValueError("Named RNG streams require a non-empty string")
    return seed.to_bytes(8, "big") + b"\0" + stream.encode("utf-8")


def derive_seed(seed: int, stream: str) -> int:
    """Derive a stable child seed without Python's process-randomized hash."""
    return int.from_bytes(hashlib.sha256(_seed_bytes(seed, stream)).digest()[:8], "big")


@dataclass(frozen=True)
class SeedContext:
    """Effective run seed and deterministic child-stream derivation."""

    effective_seed: int
    requested_seed: int | None

    def __post_init__(self) -> None:
        _seed_bytes(self.effective_seed, "seed-context")
        if self.requested_seed is not None:
            _seed_bytes(self.requested_seed, "requested-seed")

    def stream_seed(self, stream: str) -> int:
        """Return this context's stable child seed for ``stream``."""
        return derive_seed(self.effective_seed, stream)

    def numpy_rng(self, stream: str) -> np.random.Generator:
        """Return an independent NumPy generator for the named stream."""
        return np.random.default_rng(self.stream_seed(stream))


def initialize_run_seed(requested_seed: int | None = None) -> SeedContext:
    """Seed Python, NumPy and Torch before environment/model construction.

    An omitted requested seed creates fresh cryptographic entropy, but returns it
    so callers can persist the effective value with their run provenance.
    """
    effective_seed = secrets.randbits(64) if requested_seed is None else requested_seed
    context = SeedContext(effective_seed=effective_seed, requested_seed=requested_seed)
    # Libraries accept narrower values for their global state; deriving named
    # streams keeps the full effective identity available to the caller.
    global_seed = context.stream_seed("global")
    random.seed(global_seed)
    np.random.seed(global_seed % 2**32)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)
    return context


def capture_rng_state() -> dict[str, Any]:
    """Capture supported local RNG state (CPU and available CUDA generators).

    This helper intentionally captures CPU and CUDA state only. It does not
    capture MPS state, so a resumed MPS continuation resets that generator even
    though current PyTorch exposes MPS RNG functions.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state returned by :func:`capture_rng_state`.

    CUDA state is restored only when CUDA is available. No cross-device promise
    is made, and callers must restore before constructing stochastic consumers.
    """
    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"RNG state missing required keys: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
