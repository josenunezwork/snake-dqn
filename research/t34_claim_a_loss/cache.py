"""Diagnostic row cache from hash-verified v2 world exports.

Each world is regenerated through the ratified A2 regenerator
(``research.t34_exit_b8.replay.replay_world``: checkpoint SHA and every saved
state hash must match).  Only a subset of decisions is kept:

* every searched row (``lad``/``transition``): decisive ``horizon_rescue`` rows
  and zero-weight ``all_die_delay`` rows;
* every ``anchor_stride``-th ``shadow_confirm`` row (KL-anchor rows).

Row semantics come from the ratified ``normalize_row`` (forbidden mask,
decisive flag, delay flag).  The carrier B8 forward gives ``q_prev`` and the
256-D fused trunk feature ``h`` (exactly the vector B8's value/advantage heads
read), so frozen-trunk arms train on ``h`` without touching rasters.  The
recorded label Q must match the regenerated B8 Q within ``1e-3 + 1e-4|q|``.

Output: one create-only ``.npz`` per world plus ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

KIND_ANCHOR, KIND_DECISIVE, KIND_DELAY = 0, 1, 2
SEARCHED_KINDS = frozenset({"lad", "transition"})
SCHEMA = "t34-claim-a-loss-cache/v1"


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discover(roots: list[Path]) -> list[Path]:
    """Complete world exports under ``roots`` (``.complete`` marker), stable order."""
    paths: list[Path] = []
    for root in roots:
        found = sorted(Path(root).rglob("world-*.json"))
        paths.extend(p for p in found if p.with_name(p.name + ".complete").is_file())
    return paths


def keep_label(label: Mapping[str, Any], anchor_stride: int) -> bool:
    kind = label.get("label_kind")
    if kind in SEARCHED_KINDS:
        return True
    return kind == "shadow_confirm" and int(label["decision"]) % anchor_stride == 0


def row_fields(label: Mapping[str, Any], q_prev: np.ndarray) -> dict[str, Any]:
    """Ratified v2 row semantics for one kept label (forbidden mask, kind, a*)."""
    from research.t34_exit_b8_fit.interface import normalize_row

    row = {k: label[k] for k in label}
    row["q_prev"] = q_prev
    norm = normalize_row(row)
    if label.get("label_kind") == "shadow_confirm":
        kind = KIND_ANCHOR
    elif bool(norm.get("delay_only", False)):
        kind = KIND_DELAY
    elif bool(norm["decisive"]):
        kind = KIND_DECISIVE
    else:  # bare_survives searched rows: train as anchors (bare kept)
        kind = KIND_ANCHOR
    survive = np.zeros(6, dtype=bool)
    tested = np.zeros(6, dtype=bool)
    for a, value in (label.get("per_action") or {}).items():
        if value.get("tested") is True:
            tested[int(a)] = True
            survive[int(a)] = bool(value.get("survives", value.get("survive", False)))
    return {
        "kind": kind,
        "a_star": int(norm["action_star"]),
        "forbidden": np.asarray(norm["forbidden_mask"], dtype=bool),
        "tested": tested,
        "survive": survive,
        "student_action": int(label.get("student_action", int(np.argmax(q_prev)))),
        "shadow_id": int(label.get("shadow_id", -1) if label.get("shadow_id") is not None else -1),
    }


def build_world(
    path: str, checkpoint: str, out_dir: str, anchor_stride: int, q_tol: float = 1e-3
) -> dict[str, Any]:
    """Replay one world, keep the diagnostic rows, write ``<world>.npz`` (create-only)."""
    import torch

    from research.t34_exit_b8.exports import load_world
    from research.t34_exit_b8.replay import _default_observation, replay_world
    from src.model.ego2s_network import load_ego2s_checkpoint

    torch.set_num_threads(1)
    started = time.perf_counter()
    exported = load_world(path)
    world_id = str(exported["world_id"])

    def observe(world: Any, label: Mapping[str, Any]) -> dict[str, Any]:
        if not keep_label(label, anchor_stride):
            return {"skip": True}
        obs = _default_observation(world, label)
        obs["length"] = int(world.sim.length[0, 0])
        return obs

    result = replay_world(exported, checkpoint, observe=observe)
    replay_seconds = time.perf_counter() - started
    if not result.accepted:
        return {"world_id": world_id, "accepted": False, "mismatch": result.mismatch}
    obs = [o for o in result.observations if not o.get("skip")]
    net = load_ego2s_checkpoint(checkpoint, "cpu")
    local = np.stack([o["local"] for o in obs])
    glob = np.stack([o["global"] for o in obs])
    scalars = np.stack([o["scalars"] for o in obs]).astype(np.float32)
    q = np.zeros((len(obs), 6), dtype=np.float32)
    h = np.zeros((len(obs), 256), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(obs), 512):
            hh, qq = trunk_forward(
                net,
                torch.from_numpy(local[s : s + 512]),
                torch.from_numpy(glob[s : s + 512]),
                torch.from_numpy(scalars[s : s + 512]),
            )
            h[s : s + len(hh)] = hh.numpy()
            q[s : s + len(qq)] = qq.numpy()
    recorded = np.stack([np.asarray(o["label"]["q"], dtype=np.float32) for o in obs])
    err = np.abs(recorded - q)
    if not (err <= q_tol + 1e-4 * np.abs(recorded)).all():
        return {
            "world_id": world_id,
            "accepted": False,
            "mismatch": f"q_prev mismatch {float(err.max()):.3g}",
        }
    fields = [row_fields(o["label"], q[i]) for i, o in enumerate(obs)]
    out = Path(out_dir) / (world_id.replace("/", "__") + ".npz")
    arrays = {
        "world_id": np.asarray(world_id),
        "decision": np.asarray([o["decision"] for o in obs], dtype=np.int64),
        "length": np.asarray([o["length"] for o in obs], dtype=np.int64),
        "local": local,
        "global": glob,
        "scalars": scalars,
        "q_prev": q,
        "h": h,
        "kind": np.asarray([f["kind"] for f in fields], dtype=np.int8),
        "a_star": np.asarray([f["a_star"] for f in fields], dtype=np.int64),
        "forbidden": np.stack([f["forbidden"] for f in fields]),
        "tested": np.stack([f["tested"] for f in fields]),
        "survive": np.stack([f["survive"] for f in fields]),
        "student_action": np.asarray([f["student_action"] for f in fields], dtype=np.int64),
        "shadow_id": np.asarray([f["shadow_id"] for f in fields], dtype=np.int64),
    }
    with open(out, "xb") as stream:
        np.savez(stream, **arrays)
    kinds = arrays["kind"]
    return {
        "world_id": world_id,
        "accepted": True,
        "export": str(path),
        "export_sha256": sha256_file(path),
        "npz": out.name,
        "npz_sha256": sha256_file(out),
        "rows": int(len(obs)),
        "anchors": int((kinds == KIND_ANCHOR).sum()),
        "decisive": int((kinds == KIND_DECISIVE).sum()),
        "delay": int((kinds == KIND_DELAY).sum()),
        "max_abs_q_err": float(err.max()),
        "replay_seconds": replay_seconds,
        "seconds": time.perf_counter() - started,
        "pid": os.getpid(),
    }


def trunk_forward(net: Any, local: Any, glob: Any, scalars: Any) -> tuple[Any, Any]:
    """``(h, q)`` exactly as ``Ego2sNet.forward`` computes Q (h = fused 256-D)."""
    import torch

    local = local.float() * (1.0 / 255.0)
    glob = glob.float() * (1.0 / 255.0)
    h = net.fuse(torch.cat([net.local(local), net.glob(glob), scalars.float()], dim=1))
    a = net.adv(h)
    return h, net.value(h) + a - a.mean(dim=1, keepdim=True)


def _job(args: tuple[str, str, str, int]) -> dict[str, Any]:
    return build_world(*args)


def build_cache(
    roots: list[Path],
    checkpoint: Path,
    out_dir: Path,
    *,
    anchor_stride: int = 8,
    workers: int = 2,
    expect_worlds: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    paths = discover(roots)
    if expect_worlds is not None and len(paths) != expect_worlds:
        raise ValueError(f"expected {expect_worlds} complete worlds, found {len(paths)}")
    jobs = [(str(p), str(checkpoint), str(out_dir), anchor_stride) for p in paths]
    started = time.perf_counter()
    if workers <= 1:
        receipts = [_job(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            receipts = []
            for receipt in pool.imap_unordered(_job, jobs):
                receipts.append(receipt)
                print(json.dumps({"phase": "cache", "done": len(receipts), "of": len(jobs)}))
    receipts.sort(key=lambda r: r["world_id"])
    ids = [r["world_id"] for r in receipts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate world ids across export roots")
    manifest = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "roots": [str(r) for r in roots],
        "anchor_stride": anchor_stride,
        "worlds": receipts,
        "accepted": sum(bool(r["accepted"]) for r in receipts),
        "rejected": [r for r in receipts if not r["accepted"]],
        "totals": {
            k: sum(int(r.get(k, 0)) for r in receipts)
            for k in ("rows", "anchors", "decisive", "delay")
        },
        "wall_seconds": time.perf_counter() - started,
    }
    with (out_dir / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expect-worlds", type=int)
    args = parser.parse_args()
    manifest = build_cache(
        args.root,
        args.checkpoint,
        args.out,
        anchor_stride=args.anchor_stride,
        workers=args.workers,
        expect_worlds=args.expect_worlds,
    )
    print(json.dumps({k: manifest[k] for k in ("accepted", "totals", "wall_seconds")}))


if __name__ == "__main__":
    main()
