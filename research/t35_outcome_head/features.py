"""Frozen-B8-trunk features for every labelled hero decision of a world.

Each world is replayed through the audited ``research.t34_exit_b8.replay``
seam (checkpoint SHA and every 250-frame physical-state hash verified), then
the bv8 observations go once through B8's frozen trunk. Only the 256-D fuse
vector and B8's Q are kept, so the cache is ~1 KB per row.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.t34_exit_b8.exports import load_world
from research.t34_exit_b8.replay import replay_world
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

from .data import world_targets

FEATURE_DIM = 256


def trunk_forward(
    net: Ego2sNet, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fuse_features, q)`` exactly as :meth:`Ego2sNet.forward` computes Q."""
    local = local.float() * (1.0 / 255.0)
    glob = glob.float() * (1.0 / 255.0)
    h = net.fuse(torch.cat([net.local(local), net.glob(glob), scalars.float()], dim=1))
    a = net.adv(h)
    return h, net.value(h) + a - a.mean(dim=1, keepdim=True)


def _cache_name(world_id: str) -> str:
    return world_id.replace("/", "__") + ".npz"


def build_world_features(
    world_path: str, checkpoint: str, cache_dir: str, threads: int = 1, batch: int = 1024
) -> dict[str, Any]:
    """Replay + featurize one world into ``cache_dir`` (create-only). Returns a receipt."""
    torch.set_num_threads(int(threads))
    started = time.perf_counter()
    world = load_world(world_path)
    targets = world_targets(world)
    out = Path(cache_dir) / _cache_name(targets.world_id)
    result = replay_world(world, checkpoint)
    replay_s = time.perf_counter() - started
    if not result.accepted:
        return {
            "world_id": targets.world_id,
            "accepted": False,
            "mismatch": result.mismatch,
            "replay_seconds": replay_s,
        }
    obs = result.observations
    decisions = np.asarray([o["decision"] for o in obs], dtype=np.int64)
    if not np.array_equal(decisions, targets.decisions):
        raise ValueError(f"{targets.world_id}: replayed decisions differ from labels")
    net = load_ego2s_checkpoint(checkpoint, "cpu")
    feats = np.zeros((len(obs), FEATURE_DIM), dtype=np.float32)
    q = np.zeros((len(obs), 6), dtype=np.float32)
    t1 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(obs), batch):
            chunk = obs[s : s + batch]
            h, qq = trunk_forward(
                net,
                torch.from_numpy(np.stack([o["local"] for o in chunk])),
                torch.from_numpy(np.stack([o["global"] for o in chunk])),
                torch.from_numpy(np.stack([o["scalars"] for o in chunk])),
            )
            feats[s : s + len(chunk)] = h.numpy()
            q[s : s + len(chunk)] = qq.numpy()
    forward_s = time.perf_counter() - t1
    mask = np.stack([o["mask"] for o in obs]).astype(bool)
    with open(out, "xb") as stream:
        np.savez(
            stream, world_id=targets.world_id, decisions=decisions, feats=feats, q=q, mask=mask
        )
    return {
        "world_id": targets.world_id,
        "accepted": True,
        "rows": int(len(obs)),
        "replay_seconds": replay_s,
        "forward_seconds": forward_s,
        "total_seconds": time.perf_counter() - started,
        "pid": os.getpid(),
    }


def load_world_features(cache_dir: str | Path, world_id: str) -> dict[str, np.ndarray]:
    with np.load(Path(cache_dir) / _cache_name(world_id)) as z:
        return {k: z[k] for k in ("decisions", "feats", "q", "mask")}
