"""Q1 task-aligned single-learner collection contract.

This module is deliberately small and policy-free.  It records the pieces which
make Q1 C/D a different collection treatment from the legacy hero collector:
fresh namespaces, equal-probability gate mixes, terminal hero worlds, and the
registered H5000 bootstrap boundary.  Roster materialisation remains owned by
``dev_screen._design_rows`` so the training construction cannot drift from the
historical gate's wrappers or mixed-slot rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np


PROFILE = "task-aligned-n1"
MIXES = ("scripted", "frozen", "mixed")
HORIZON = 5_000
NAMESPACE_PREFIX = "q1-task-aligned-n1-20261017"


@dataclass(frozen=True)
class WorldPlan:
    """One reset's immutable task-collection decisions."""

    mix: str
    terminal_hero: bool = True
    horizon: int = HORIZON


def fresh_namespace(arm: str, seed: int) -> str:
    """Return the disjoint Q1 namespace for an arm/seed block."""
    if arm not in ("A", "B", "C", "D"):
        raise ValueError("Q1 arm must be A, B, C, or D")
    if int(seed) < 0:
        raise ValueError("Q1 seed must be non-negative")
    return f"{NAMESPACE_PREFIX}/{arm}/seed{int(seed)}"


def equal_mix_assignments(seed: int, reset_index: Sequence[int]) -> np.ndarray:
    """Choose a gate mix independently and uniformly for each world reset.

    This uses an isolated deterministic RNG stream.  It intentionally does not
    balance a finite batch: equal reset *probability*, rather than equal
    transition exposure or an artificial per-segment quota, is the Q1 contract.
    """
    ids = np.asarray(reset_index, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20_261_017, 1]))
    # Draw through the largest requested reset ordinal so batching/reset order
    # does not alter a world's assigned mix.
    if len(ids) == 0:
        return np.empty(0, dtype=object)
    if np.any(ids < 0):
        raise ValueError("reset indexes must be non-negative")
    draws = rng.integers(0, len(MIXES), size=int(ids.max()) + 1)
    return np.asarray([MIXES[int(draws[index])] for index in ids], dtype=object)


def reset_mask_after_step(hero_done: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Return worlds to reset after one step under terminal-hero semantics.

    Hero death is terminal.  A surviving hero at the registered H5000 horizon
    is *not* terminal: its n-step pending rows are flushed with the existing
    bootstrap rule before the world resets.
    """
    done = np.asarray(hero_done, dtype=bool)
    frames = np.asarray(frame, dtype=np.int64)
    if done.shape != frames.shape:
        raise ValueError("hero_done and frame must have the same shape")
    return done | (frames >= HORIZON)


def reset_seed(namespace: str, training_seed: int, actor_segment: int, env: int, ordinal: int) -> int:
    """Return a fresh, order-independent uint32 seed for one terminal reset."""
    key = f"{namespace}|seed{int(training_seed)}|actor{int(actor_segment)}|env{int(env)}|reset{int(ordinal)}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def mix_for_reset(training_seed: int, actor_segment: int, env: int, ordinal: int) -> str:
    """Independently sample one of the three gate mixes for a reset."""
    rng = np.random.default_rng(
        np.random.SeedSequence([int(training_seed), 20_261_017, int(actor_segment), int(env), int(ordinal)])
    )
    return MIXES[int(rng.integers(len(MIXES)))]

