"""Streaming replay-regeneration data path for P4.

This module deliberately keeps a bounded world/batch buffer.  It never builds
an array of all observations, which is essential because a real bank can be
one million roughly-65-KB regenerated rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .interface import ReplaySource, ReplayWorld, validate_row


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    worlds_seen: int
    worlds_dropped: int
    rows_seen: int
    rows_yielded: int
    elapsed_seconds: float
    epoch_wall_seconds: float

    @property
    def drop_rate(self) -> float:
        return self.worlds_dropped / max(1, self.worlds_seen)


class ReplayDivergenceError(RuntimeError):
    """More than two percent of complete worlds diverged during replay."""


def verify_world_proof(world: ReplayWorld, *, allow_fake: bool) -> bool:
    """Check the engine's complete-world proof before consuming any row."""
    if not world.verified:
        return False
    if not allow_fake and (
        len(world.hash_expected) != 64
        or len(world.hash_actual) != 64
        or world.hash_expected != world.hash_actual
    ):
        raise ValueError(f"verified production world {world.world_id} lacks matching sha256 proof")
    return True


def sample_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Apply fixed DAgger/death-window/decisive weighting to a selected row set.

    Current round is 1.0, old rounds 0.5.  Every death window in a world sums
    to 10 *after* applying its round factor, including the 20x decisive boost.
    """
    age = np.asarray([1.0 if int(r["round_age"]) == 0 else 0.5 for r in rows], dtype=np.float32)
    base = age * np.asarray([20.0 if bool(r["decisive"]) else 1.0 for r in rows], dtype=np.float32)
    groups: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        if bool(row["death_window"]):
            groups.setdefault(
                (str(row["world_id"]), int(row.get("death_window_id", 0))), []
            ).append(index)
    for indices in groups.values():
        # Decisive rows are 20x inside the window too.  Its final mass is 10
        # for current and 5 for historical (the fixed 1.0 / 0.5 DAgger rule).
        base[indices] *= (10.0 * age[indices[0]]) / float(base[indices].sum())
    return base


def select_training_rows(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> list[Mapping[str, Any]]:
    """Keep all searched rows and a deterministic 1:1 sample of unsearched rows."""
    searched = [r for r in rows if bool(r["searched"])]
    unsearched = [r for r in rows if not bool(r["searched"])]
    if not searched or len(unsearched) < len(searched):
        raise ValueError("export cannot satisfy the required 1:1 searched/unsearched row ratio")
    indices = np.sort(rng.choice(len(unsearched), size=len(searched), replace=False))
    selected = searched + [unsearched[int(i)] for i in indices]
    weights = sample_weights(selected)
    return [dict(row, _sample_weight=float(weight)) for row, weight in zip(selected, weights)]


def _stack(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    fields = ("local", "global", "scalars", "q_prev", "forbidden_mask")
    result = {key: np.stack([np.asarray(row[key]) for row in rows]) for key in fields}
    result["action_star"] = np.asarray([row["action_star"] for row in rows], dtype=np.int64)
    result["length"] = np.asarray([row["length"] for row in rows], dtype=np.int64)
    result["decisive"] = np.asarray([row["decisive"] for row in rows], dtype=bool)
    result["sample_weight"] = np.asarray([row["_sample_weight"] for row in rows], dtype=np.float32)
    if all("full_sight" in row for row in rows):
        result["full_sight"] = np.stack([np.asarray(row["full_sight"]) for row in rows])
    elif any("full_sight" in row for row in rows):
        raise ValueError("world mixes rows with and without full_sight")
    aux_target = "hindsight_log_free_region"
    aux_mask = "hindsight_log_free_region_mask"
    if all(aux_target in row and aux_mask in row for row in rows):
        result[aux_target] = np.asarray([row[aux_target] for row in rows], dtype=np.float32)
        result[aux_mask] = np.asarray([row[aux_mask] for row in rows], dtype=bool)
    elif any(aux_target in row or aux_mask in row for row in rows):
        raise ValueError("hindsight target and mask must be present together on every row")
    return result


class StreamingReplayLoader:
    """Bounded per-epoch replay loader with end-of-epoch divergence enforcement."""

    def __init__(self, source: ReplaySource, batch_size: int = 1024, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.source, self.batch_size, self.seed = source, batch_size, seed
        self.last_stats: EpochStats | None = None

    def epoch(self, epoch: int) -> Iterator[dict[str, np.ndarray]]:
        started = perf_counter()
        regen_seconds = 0.0
        segment_started = started
        rng = np.random.default_rng(self.seed + epoch)
        worlds_seen = worlds_dropped = rows_seen = rows_yielded = 0
        chunks: dict[str, list[int]] = {}
        buffered: list[Mapping[str, Any]] = []
        for world in self.source.replay_epoch(epoch):
            worlds_seen += 1
            chunk = world.chunk_id or "unpartitioned"
            chunks.setdefault(chunk, [0, 0])[0] += 1
            if not verify_world_proof(
                world, allow_fake=bool(getattr(self.source, "is_fake", False))
            ):
                worlds_dropped += 1
                chunks[chunk][1] += 1
                continue
            for row in world.rows:
                validate_row(row)
            rows_seen += len(world.rows)
            buffered.extend(select_training_rows(world.rows, rng))
            while len(buffered) >= self.batch_size:
                selection = buffered[: self.batch_size]
                del buffered[: self.batch_size]
                rng.shuffle(selection)
                rows_yielded += len(selection)
                packed = _stack(selection)
                regen_seconds += perf_counter() - segment_started
                yield packed
                segment_started = perf_counter()
        if buffered:
            rng.shuffle(buffered)
            rows_yielded += len(buffered)
            packed = _stack(buffered)
            regen_seconds += perf_counter() - segment_started
            yield packed
            segment_started = perf_counter()
        regen_seconds += perf_counter() - segment_started
        stats = EpochStats(
            epoch,
            worlds_seen,
            worlds_dropped,
            rows_seen,
            rows_yielded,
            regen_seconds,
            perf_counter() - started,
        )
        self.last_stats = stats
        # Intentionally assess only the final cumulative count, never first/one world.
        bad_chunks = {
            name: counts for name, counts in chunks.items() if counts[1] / counts[0] > 0.02
        }
        if bad_chunks:
            raise ReplayDivergenceError(
                f"replay hash mismatch rate exceeds 2% in complete chunk(s): {bad_chunks}; "
                f"epoch total {stats.worlds_dropped}/{stats.worlds_seen}"
            )


def to_device(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert a bounded loader batch while preserving bool/int fields."""
    return {
        key: torch.from_numpy(np.ascontiguousarray(value)).to(device)
        for key, value in batch.items()
    }
