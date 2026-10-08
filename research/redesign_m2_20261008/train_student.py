#!/usr/bin/env python3
"""M2 student fit: distil frp3-s12(+v8) labels into an Ego2sNet on the Mac GPU (doc 15.2).

Loss per sample = Huber(Q_s(o, .) - Q_T(o, .)) over all 6 actions (the teacher's RAW
scale, no standardization) + LAMBDA * DQfD margin
``max_{a in A} [Q_s(o, a) + m * 1(a != a_v8)] - Q_s(o, a_v8)`` with ``A`` = the resolved
mask plus ``a_v8``; ``m = 0.1 * median per-state teacher Q range`` measured on round-0
data (written once to ``<out>/margin.json`` and reused by later rounds).

Data: every shard of the given rounds; held-out = worlds with ``world_seed % 10 == 0``
(distillation worlds only; the NI worlds are a disjoint namespace). Shards are streamed:
a background thread decompresses shards in random order into a shuffle pool.

Selection: the epoch with the best held-out v8-action agreement (masked argmax over the
resolved mask vs ``a_v8``) is saved as ``student.pth``; per-epoch metrics go to
``metrics.json``. Not gate evidence.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.model.ego2s_network import save_ego2s_checkpoint  # noqa: E402
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint  # noqa: E402

LAMBDA = 1.0
MARGIN_FRACTION = 0.1
KEYS = ("local", "global", "scalars", "q_teacher", "a_v8", "a_base", "mask_resolved", "length")


def shard_paths(data: Path, rounds: List[int]) -> List[Path]:
    paths: List[Path] = []
    for r in rounds:
        paths += sorted((data / f"round-{r}").glob("chunk-*-part-*.npz"))
    if not paths:
        raise SystemExit("no shards")
    return paths


def load_shard(path: Path, held_out: bool) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        keep = (z["world_seed"] % 10 == 0) == held_out
        return {k: z[k][keep] for k in KEYS}


def concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def stream(paths: List[Path], rng: np.random.Generator, batch: int, pool_shards: int = 4):
    """Yield shuffled minibatches over one epoch (decompression in a background thread)."""
    order = [paths[i] for i in rng.permutation(len(paths))]
    q: "queue.Queue[Optional[Dict[str, np.ndarray]]]" = queue.Queue(maxsize=pool_shards + 2)

    def reader() -> None:
        for p in order:
            q.put(load_shard(p, held_out=False))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    pool: List[Dict[str, np.ndarray]] = []
    done = False
    while not done or pool:
        while not done and len(pool) < pool_shards:
            item = q.get()
            if item is None:
                done = True
            else:
                pool.append(item)
        if not pool:
            break
        data = concat(pool)
        pool = []
        idx = rng.permutation(len(data["a_v8"]))
        for start in range(0, len(idx) - batch + 1, batch):
            sel = idx[start : start + batch]
            yield {k: v[sel] for k, v in data.items()}


def to_device(b: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}


def losses(net: Ego2sNet, b: Dict[str, torch.Tensor], margin: float):
    q = net(b["local"], b["global"], b["scalars"])
    qt = b["q_teacher"]
    reg = torch.nn.functional.smooth_l1_loss(q, qt)
    a = b["a_v8"].long()
    onehot = torch.nn.functional.one_hot(a, 6).bool()
    allowed = b["mask_resolved"] | onehot
    bonus = (~onehot).float() * margin
    hi = torch.where(allowed, q + bonus, torch.full_like(q, -1e9)).max(dim=1).values
    marg = (hi - q.gather(1, a[:, None]).squeeze(1)).mean()
    return reg + LAMBDA * marg, reg, marg, q


@torch.no_grad()
def evaluate(net: Ego2sNet, val: Dict[str, np.ndarray], device, margin: float) -> Dict[str, float]:
    net.eval()
    agree, n, reg, ss_res, big_agree, big_n, ov_agree, ov_n = 0, 0, 0.0, 0.0, 0, 0, 0, 0
    qs: List[np.ndarray] = []
    for start in range(0, len(val["a_v8"]), 2048):
        b = {k: v[start : start + 2048] for k, v in val.items()}
        t = to_device(b, device)
        _, r, _, q = losses(net, t, margin)
        q = q.cpu().numpy()
        qs.append(q)
        m = b["mask_resolved"].copy()
        m[~m.any(axis=1), :3] = True
        pred = np.where(m, q, -np.inf).argmax(axis=1)
        ok = pred == b["a_v8"]
        agree += int(ok.sum())
        n += len(ok)
        over = b["a_v8"] != b["a_base"]
        ov_agree += int(ok[over].sum())
        ov_n += int(over.sum())
        big = b["length"] > 500
        big_agree += int(ok[big].sum())
        big_n += int(big.sum())
        reg += float(r) * len(ok)
        ss_res += float(((q - b["q_teacher"]) ** 2).sum())
    qt = val["q_teacher"]
    r2 = 1.0 - ss_res / float(((qt - qt.mean()) ** 2).sum())
    net.train()
    return {
        "agree_v8": agree / n,
        "agree_v8_len_gt_500": big_agree / big_n if big_n else None,
        "n_len_gt_500": big_n,
        "agree_v8_on_veto_overrides": ov_agree / ov_n if ov_n else None,
        "n_veto_overrides": ov_n,
        "huber": reg / n,
        "q_r2": r2,
        "n": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--rounds", default="0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", default=None, help="warm-start checkpoint (previous round)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--val-max", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    rounds = [int(r) for r in args.rounds.split(",")]
    paths = shard_paths(args.data, rounds)
    args.out.mkdir(parents=True, exist_ok=True)

    margin_file = args.data / "margin.json"
    if margin_file.exists():
        margin = json.loads(margin_file.read_text())["margin"]
    else:
        if rounds != [0]:
            raise SystemExit("margin.json must be fixed from round-0 data first")
        ranges = []
        for p in paths:
            with np.load(p) as z:
                q = z["q_teacher"]
                ranges.append(q.max(axis=1) - q.min(axis=1))
        median = float(np.median(np.concatenate(ranges)))
        margin = MARGIN_FRACTION * median
        margin_file.write_text(
            json.dumps({"margin": margin, "median_q_range": median, "fraction": MARGIN_FRACTION})
        )

    val_parts = [load_shard(p, held_out=True) for p in paths]
    val = concat(val_parts)
    if len(val["a_v8"]) > args.val_max:
        keep = np.sort(rng.choice(len(val["a_v8"]), args.val_max, replace=False))
        val = {k: v[keep] for k, v in val.items()}

    if args.init:
        net = load_ego2s_checkpoint(args.init, device)
    else:
        net = Ego2sNet()
        with torch.no_grad():  # start the value head at the teacher's mean Q (raw scale)
            net.value.bias.fill_(float(val["q_teacher"].mean()))
        net = net.to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_train = 0
    for p in paths:
        with np.load(p) as z:
            n_train += int((z["world_seed"] % 10 != 0).sum())
    total_steps = max(1, args.epochs * (n_train // args.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    history = []
    best = -1.0
    before = evaluate(net, val, device, margin)
    history.append({"epoch": 0, **before})
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(1, args.epochs + 1):
        t0, steps, run = time.perf_counter(), 0, np.zeros(3)
        for b in stream(paths, rng, args.batch):
            loss, reg, marg = losses(net, to_device(b, device), margin)[:3]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            steps += 1
            if steps % 50 == 1:
                run += [float(loss), float(reg), float(marg)]
        metrics = evaluate(net, val, device, margin)
        row = {
            "epoch": epoch,
            "steps": steps,
            "samples_per_s": steps * args.batch / (time.perf_counter() - t0),
            "train_loss_reg_margin": (run / max(1, (steps + 49) // 50)).tolist(),
            **metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["agree_v8"] > best:
            best = metrics["agree_v8"]
            meta = {
                "rounds": rounds,
                "epoch": epoch,
                "margin": margin,
                "metrics": metrics,
                "n_train": n_train,
                "init": args.init,
            }
            save_ego2s_checkpoint(args.out / "student.pth", net, meta)
    (args.out / "metrics.json").write_text(
        json.dumps({"history": history, "best_agree_v8": best, "margin": margin}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
