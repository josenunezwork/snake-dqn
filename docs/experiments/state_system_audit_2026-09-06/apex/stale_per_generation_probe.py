#!/usr/bin/env python3
"""Reproduce stale distributed-PER indices after a replay-ring overwrite."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.apex_buffer import SharedPrioritizedBuffer


def transition(marker: float) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
    """Build a valid transition whose state marker identifies its slot occupant."""
    state = np.full(61, marker, dtype=np.float32)
    next_state = np.full(61, marker + 0.5, dtype=np.float32)
    return state, 0, 0.0, next_state, False


def add(buffer: SharedPrioritizedBuffer, marker: float) -> None:
    """Insert one transition with equal initial sampling priority."""
    state, action, reward, next_state, done = transition(marker)
    buffer.add(state, action, reward, next_state, done, priority=1.0)


def slot_markers(buffer: SharedPrioritizedBuffer, indices: list[int]) -> list[float]:
    """Read the current transition marker at each returned data index."""
    return [float(buffer._tree.data[index][0][0]) for index in indices]


def slot_priorities(buffer: SharedPrioritizedBuffer, indices: list[int]) -> list[float]:
    """Read the current leaf priority at each returned data index."""
    offset = buffer._tree.capacity - 1
    return [float(buffer._tree.tree[offset + index]) for index in indices]


def main() -> int:
    np.random.seed(7)
    buffer = SharedPrioritizedBuffer(
        capacity=2,
        alpha=1.0,
        beta_start=0.4,
        beta_end=1.0,
        beta_frames=100,
        state_size=61,
    )

    add(buffer, 1.0)
    add(buffer, 2.0)
    batch, sampled_indices, _weights = buffer.sample(batch_size=2)
    sampled_markers = [float(state[0]) for state in batch["states"]]

    # A full ring rotation replaces both sampled transitions before their
    # asynchronous TD-error update returns from the learner.
    add(buffer, 101.0)
    add(buffer, 102.0)
    markers_before_update = slot_markers(buffer, sampled_indices)
    stale_td_errors = np.array([20.0, 30.0], dtype=np.float32)
    buffer.update_priorities(sampled_indices, stale_td_errors)
    priorities_after_update = slot_priorities(buffer, sampled_indices)

    overwritten = all(
        sampled != current
        for sampled, current in zip(sampled_markers, markers_before_update)
    )
    stale_update_applied = bool(
        overwritten
        and np.allclose(priorities_after_update, [20.000001, 30.000001])
    )
    result = {
        "probe": "distributed_per_stale_slot_identity",
        "status": "reproduced" if stale_update_applied else "not_reproduced",
        "capacity": 2,
        "sampled_indices": sampled_indices,
        "sampled_transition_markers": sampled_markers,
        "slot_markers_after_full_ring_overwrite": markers_before_update,
        "stale_td_errors_applied": stale_td_errors.tolist(),
        "slot_priorities_after_stale_update": priorities_after_update,
        "finding": (
            "Sample returns bare slot indices. After those slots are overwritten, "
            "update_priorities accepts the old indices and changes the new occupants."
        ),
        "source_refs": {
            "sample_indices": "src/training/apex_buffer.py:573-675",
            "priority_update": "src/training/apex_buffer.py:677-696",
            "ring_overwrite": "src/training/sum_tree.py:55-75",
            "index_only_update": "src/training/sum_tree.py:82-97",
        },
    }

    output_path = Path(__file__).with_name("stale_per_generation_probe.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if stale_update_applied else 1


if __name__ == "__main__":
    raise SystemExit(main())
