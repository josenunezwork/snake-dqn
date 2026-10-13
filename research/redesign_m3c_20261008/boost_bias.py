#!/usr/bin/env python3
"""M3-C boost screen tool (B0; development only): an exact serving-time boost bias.

Adding ``beta`` to the dueling advantage head's bias of the three boost actions raises every
boost-vs-normal Q gap by exactly ``beta`` (``Q = V + A - mean(A)``: boost Q +beta/2, normal Q
-beta/2) and changes nothing else, so a biased checkpoint IS "the same student, boosting
more", playable by every existing tool (dev_probe, Phase R harness) without code changes.

* ``calibrate``: greedy boost rate on held-out demonstration states (rounds 10 + 12, world
  seeds % 10 == 0) for a grid of beta; reports the beta whose rate matches a target (the
  teacher's own greedy boost rate on the same states by default, ~0.95%; the v8-action
  labels are 1.4%).
* ``write``: the biased checkpoint (meta records the source sha256 and beta).

  ./venv/bin/python research/redesign_m3c_20261008/boost_bias.py calibrate --ckpt X.pth
  ./venv/bin/python research/redesign_m3c_20261008/boost_bias.py write --ckpt X.pth \\
      --beta 0.08 --out X_boost0.08.pth
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOOST = slice(3, 6)


def biased_net(path: str, beta: float):
    import torch

    from src.model.ego2s_network import load_ego2s_checkpoint

    net = load_ego2s_checkpoint(path, "cpu")
    with torch.no_grad():
        net.adv.bias[BOOST] += float(beta)
    return net


def boost_rate(q: np.ndarray, mask: np.ndarray, beta: float) -> float:
    m = mask.copy()
    m[~m.any(axis=1), :3] = True
    qb = q.copy()
    qb[:, BOOST] += beta
    return float((np.where(m, qb, -np.inf).argmax(axis=1) >= 3).mean())


def cmd_calibrate(args) -> int:
    import torch

    from research.redesign_m3c_20261008.diag_q import load_states
    from src.model.ego2s_network import load_ego2s_checkpoint

    torch.set_num_threads(1)
    st = load_states(args.shards_per_round)
    net = load_ego2s_checkpoint(args.ckpt, "cpu")
    with torch.no_grad():
        q = np.concatenate(
            [
                net(
                    torch.from_numpy(st["local"][i : i + 2048]),
                    torch.from_numpy(st["global"][i : i + 2048]),
                    torch.from_numpy(st["scalars"][i : i + 2048]),
                ).numpy()
                for i in range(0, len(st["scalars"]), 2048)
            ]
        )
    teacher = boost_rate(st["q_teacher"], st["mask_resolved"], 0.0)
    grid = [round(x, 3) for x in np.arange(0.0, args.max_beta + 1e-9, args.step)]
    rates = {b: boost_rate(q, st["mask_resolved"], b) for b in grid}
    target = args.target if args.target is not None else teacher
    best = min(grid, key=lambda b: abs(rates[b] - target))
    out = {
        "ckpt": args.ckpt,
        "n_states": int(len(q)),
        "teacher_greedy_boost_rate": teacher,
        "target": target,
        "rates": rates,
        "beta_at_target": best,
    }
    print(json.dumps(out))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def cmd_write(args) -> int:
    import torch

    from src.model.ego2s_network import save_ego2s_checkpoint

    if args.out.exists():
        raise SystemExit(f"refusing: {args.out} exists")
    src_sha = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = biased_net(args.ckpt, args.beta)
    meta = dict(blob.get("meta") or {})
    meta["boost_bias"] = {"beta": float(args.beta), "source": args.ckpt, "source_sha256": src_sha}
    save_ego2s_checkpoint(args.out, net, meta)
    print(json.dumps({"out": str(args.out), "beta": args.beta, "source_sha256": src_sha}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--shards-per-round", type=int, default=3)
    c.add_argument("--max-beta", type=float, default=0.4)
    c.add_argument("--step", type=float, default=0.01)
    c.add_argument("--target", type=float, default=None)
    c.add_argument("--out", type=Path, default=None)
    w = sub.add_parser("write")
    w.add_argument("--ckpt", required=True)
    w.add_argument("--beta", type=float, required=True)
    w.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return cmd_calibrate(args) if args.cmd == "calibrate" else cmd_write(args)


if __name__ == "__main__":
    raise SystemExit(main())
