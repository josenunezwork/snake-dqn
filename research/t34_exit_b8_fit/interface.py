"""Stable, deliberately small boundary between the ExIt engine and the fitter.

The production engine is built in the sibling worktree and is not imported at
module import time.  It must provide a factory returning :class:`ReplaySource`.
Each world is hash-verified by the engine *before* it is exposed to this
package.  A mismatching world is returned as ``verified=False`` and is dropped;
the loader reports its cumulative rate after the complete epoch.

The local fake below is solely a deterministic smoke fixture.  It has no game
semantics and cannot be used to make a research or gate claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch

EXPORT_SCHEMA = "exit-b8-export/v1"
REQUIRED_ROW_FIELDS = frozenset(
    {
        "local",
        "global",
        "scalars",
        "q_prev",
        "action_star",
        "forbidden_mask",
        "searched",
        "death_window",
        "decisive",
        "round_age",
        "length",
        "world_id",
    }
)


@dataclass(frozen=True)
class ReplayWorld:
    """One fully regenerated exported world and its hash verification result."""

    world_id: str
    rows: Sequence[Mapping[str, Any]]
    verified: bool
    hash_expected: str = ""
    hash_actual: str = ""
    reason: str = ""
    chunk_id: str = ""


class ReplaySource(Protocol):
    """Streaming producer; implementations must replay/regenerate per epoch.

    Row arrays stay immutable for their lifetime, including after the iterator advances;
    the bounded loader may retain a final partial batch while requesting the
    following world.  Engines that reuse backing buffers must yield owned rows.

    An optional ``training_row_counts(epoch) -> tuple[int, int]`` supplies the
    searched/unsearched counts of the verified retained stream, enabling exact
    bank-wide 1:1 sampling without another observation bank in memory.  Counts
    must exclude hash-rejected worlds.  Count drift invalidates the entire fit
    even when the ordinary world-drop rate is <=2%.  Without this metadata the
    loader requires each retained world to independently support a 1:1 sample.
    """

    is_fake: bool

    def replay_epoch(self, epoch: int) -> Iterable[ReplayWorld]:
        """Regenerate worlds for ``epoch``; never materialise the whole bank."""


class BatchModel(Protocol):
    """The tower and S1b models share this thin fitting-side call surface."""

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return raw Q values with shape ``(N, 6)``."""


def validate_row(row: Mapping[str, Any]) -> None:
    """Reject an incomplete export row before it can influence a fit."""
    missing = REQUIRED_ROW_FIELDS.difference(row)
    if missing:
        raise ValueError(f"exit-fit row missing required fields: {sorted(missing)}")
    q = np.asarray(row["q_prev"], dtype=np.float32)
    forbidden = np.asarray(row["forbidden_mask"], dtype=bool)
    if q.shape != (6,) or forbidden.shape != (6,):
        raise ValueError("q_prev and forbidden_mask must each contain six actions")
    if not np.isfinite(q).all() or not 0 <= int(row["action_star"]) < 6:
        raise ValueError("row has non-finite q_prev or invalid action_star")
    if forbidden[int(row["action_star"])]:
        raise ValueError("forbidden actions must exclude action_star")
    local, glob, scalars = (
        np.asarray(row["local"]),
        np.asarray(row["global"]),
        np.asarray(row["scalars"]),
    )
    if local.shape != (7, 31, 31) or glob.shape != (4, 37, 37) or scalars.shape != (60,):
        raise ValueError("P4 rows require ego2s-bv8 local/global/scalar shapes")
    if local.dtype != np.uint8 or glob.dtype != np.uint8 or not np.isfinite(scalars).all():
        raise ValueError("raster planes must be uint8 and scalars finite")
    if "full_sight" in row:
        full_sight = np.asarray(row["full_sight"])
        if full_sight.shape != (4, 88, 152) or full_sight.dtype != np.uint8:
            raise ValueError("full_sight must be uint8 (4, 88, 152)")
    if bool(row["decisive"]) and not bool(row["searched"]):
        raise ValueError("decisive labels must be searched decisions")
    if bool(row["death_window"]) and "death_window_id" not in row:
        raise ValueError("every death-window row requires its explicit window identifier")


def forward_batch(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Call either the P5 model API or the existing three-input S1b network."""
    method = getattr(model, "forward_batch", None)
    q = (
        method(batch)
        if callable(method)
        else model(batch["local"], batch["global"], batch["scalars"])
    )
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f"student model must return (N, 6) Q values, got {tuple(q.shape)}")
    return q


class FakeReplaySource:
    """Small deterministic fake for wiring/preflight tests only."""

    is_fake = True

    def __init__(self, worlds: int = 2, rows_per_world: int = 128, seed: int = 0) -> None:
        self.worlds, self.rows_per_world, self.seed = worlds, rows_per_world, seed

    def replay_epoch(self, epoch: int) -> Iterable[ReplayWorld]:
        # Regeneration must be stable across epochs; only loader ordering changes.
        rng = np.random.default_rng(self.seed)
        for w in range(self.worlds):
            rows = []
            for i in range(self.rows_per_world):
                q = rng.normal(size=6).astype(np.float32)
                action = int(q.argmax())
                forbidden = np.zeros(6, dtype=bool)
                if i % 5 == 0:
                    forbidden[(action + 1) % 6] = True
                rows.append(
                    {
                        "local": rng.integers(0, 255, (7, 31, 31), dtype=np.uint8),
                        "global": rng.integers(0, 255, (4, 37, 37), dtype=np.uint8),
                        "scalars": rng.normal(size=60).astype(np.float32),
                        "full_sight": rng.integers(0, 255, (4, 88, 152), dtype=np.uint8),
                        "q_prev": q,
                        "action_star": action,
                        "forbidden_mask": forbidden,
                        "searched": i % 2 == 0,
                        "death_window": i % 11 == 0,
                        "death_window_id": i // 11,
                        "decisive": i % 10 == 0,
                        "round_age": 0 if w == 0 else 1,
                        "length": int(rng.integers(5, 1300)),
                        "world_id": f"fake-{w}",
                    }
                )
            yield ReplayWorld(
                world_id=f"fake-{w}", rows=rows, verified=True, reason="FAKE_SMOKE_ONLY"
            )
