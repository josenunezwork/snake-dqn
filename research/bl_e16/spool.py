"""Bounded disk spool for complete-life E16 critic targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from research.bl_e16.training import E16Batch


def write_life(path: Path, batch: E16Batch) -> Path:
    """Write a completed critic life once; targets must already be exact MC returns."""
    if batch.critic_targets is None or batch.actions.shape[1] != 1:
        raise ValueError("spool accepts one complete critic life with explicit targets")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    arrays: dict[str, Any] = {}
    for key in (
        "local",
        "global_",
        "scalars",
        "frame",
        "actions",
        "logp",
        "rewards",
        "dones",
        "values",
        "action_masks",
        "critic_targets",
    ):
        value = getattr(batch, key)
        arrays[key] = value.detach().cpu().numpy()
    for key in ("geometry", "history"):
        value = getattr(batch, key)
        if value is not None:
            arrays[key] = value.detach().cpu().numpy()
    if batch.horizon_cutoff is not None:
        arrays["horizon_cutoff"] = batch.horizon_cutoff.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)
    return path


def batches(path: Path, rows: int = 32) -> Iterator[E16Batch]:
    """Read bounded training batches from a complete-life spool."""
    if rows < 1:
        raise ValueError("rows must be positive")
    with np.load(path, allow_pickle=False) as value:
        count = len(value["actions"])
        for start in range(0, count, rows):
            end = min(count, start + rows)

            def get(key: str) -> torch.Tensor:
                return torch.from_numpy(np.array(value[key][start:end], copy=True))

            def optional(key: str) -> torch.Tensor | None:
                return get(key) if key in value.files else None

            yield E16Batch(
                local=get("local"),
                global_=get("global_"),
                scalars=get("scalars"),
                frame=get("frame"),
                actions=get("actions"),
                logp=get("logp"),
                rewards=get("rewards"),
                dones=get("dones"),
                values=get("values"),
                last_value=torch.zeros(1),
                action_masks=get("action_masks"),
                geometry=optional("geometry"),
                history=optional("history"),
                horizon_cutoff=optional("horizon_cutoff"),
                critic_targets=get("critic_targets"),
            )


class StreamingLife:
    """Store at most 32 physical observations in RAM while awaiting MC targets.

    NPZ is split into bounded chunks because slicing a compressed 5000-row NPZ
    would decompress the entire geometry array on every minibatch.
    """

    def __init__(self, directory: Path, rows: int = 32) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.rows = rows
        self.buffer: list[dict[str, np.ndarray]] = []
        self.rewards: list[float] = []
        self.chunks: list[Path] = []
        self.closed = False

    def append(self, record: dict[str, Any], reward: float, done: bool, frame: int) -> None:
        if self.closed:
            raise ValueError("completed life is immutable")
        names = {
            "local": "local",
            "global_": "global_",
            "scalars": "scalars",
            "actions": "action",
            "logp": "logp",
            "values": "value",
            "action_masks": "mask",
            "geometry": "geometry",
            "history": "history",
        }
        row = {
            key: value.detach().cpu().numpy()
            for key, source in names.items()
            if (value := record[source]) is not None
        }
        row.update(
            frame=np.array([frame]),
            rewards=np.array([reward], dtype=np.float32),
            dones=np.array([done]),
            horizon_cutoff=np.array([frame + 1 >= 5000]),
        )
        self.buffer.append(row)
        self.rewards.append(reward)
        if len(self.buffer) >= self.rows:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        path = self.directory / f"chunk-{len(self.chunks):05d}.npz"
        np.savez_compressed(
            path, **{key: np.stack([row[key] for row in self.buffer]) for key in self.buffer[0]}
        )
        self.chunks.append(path)
        self.buffer.clear()

    def finish(self) -> None:
        if self.closed:
            raise ValueError("life already complete")
        self._flush()
        targets = np.cumsum(np.array(self.rewards, dtype=np.float64)[::-1])[::-1].astype(np.float32)
        np.save(self.directory / "targets.npy", targets)
        self.closed = True

    def batches(self) -> Iterator[E16Batch]:
        if not self.closed:
            raise ValueError("complete terminal life required before critic fitting")
        targets = np.load(self.directory / "targets.npy", allow_pickle=False)
        offset = 0
        for chunk in self.chunks:
            with np.load(chunk, allow_pickle=False) as saved:
                arrays = {
                    key: torch.from_numpy(np.array(saved[key], copy=True)) for key in saved.files
                }
            count = len(arrays["actions"])
            arrays["critic_targets"] = torch.from_numpy(
                targets[offset : offset + count].copy()
            ).view(count, 1)
            arrays["last_value"] = torch.zeros(1)
            yield E16Batch(**arrays)
            offset += count

    def snapshot(self) -> dict[str, Any]:
        """Embed compressed pending chunks so resume needs no old runtime disk."""
        return {
            "buffer": self.buffer,
            "rows": self.rows,
            "rewards": list(self.rewards),
            "closed": self.closed,
            "chunks": [path.read_bytes() for path in self.chunks],
        }

    @classmethod
    def restore(cls, directory: Path, state: dict[str, Any]) -> "StreamingLife":
        spool = cls(directory, rows=int(state["rows"]))
        spool.rewards = list(state["rewards"])
        spool.buffer = list(state.get("buffer", []))
        for index, blob in enumerate(state["chunks"]):
            path = directory / f"chunk-{index:05d}.npz"
            path.write_bytes(blob)
            spool.chunks.append(path)
        if state["closed"]:
            spool.finish()
        return spool
