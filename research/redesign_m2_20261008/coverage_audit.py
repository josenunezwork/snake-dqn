#!/usr/bin/env python3
"""M2b teacher-feature -> raster coverage audit (owner request, before regenerating data).

Question: which of the teacher's 61 inputs can be recovered from the student's observation?
Data: shards recorded with ``gen_data.py --obs-version b --record-state61`` (the teacher's
exact 61-D input at each hero decision next to the ego2s-b observation). Two probe nets
(the Ego2sNet trunk with a 61-wide linear head) regress the standardized 61 features from

* ``draft``: the ego2s-draft part only (local channels 0-5, scalars 0-11), the M2 input;
* ``b``: the full ego2s-b observation;

trained on MPS, evaluated on held-out worlds (``world_seed % 4 == 0``). Reports per-feature
held-out R^2. Expected non-recoverable by design: features in the ABSOLUTE frame (heading
one-hot 0-3, boundary distances 40-43, enemy relative x/y and heading in absolute axes)
are only recoverable up to the rotation the ego frame removes (the arena's aspect ratio
leaks orientation through the wall planes); the policy is rotation-equivariant, so these
need no new input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.model.ego2s_network import Ego2sNet  # noqa: E402

NAMES = (
    ["dir_up", "dir_right", "dir_down", "dir_left", "length_capped"]
    + ["food_rel_x", "food_rel_y", "food_dist"]
    + [f"food_density_{i}" for i in range(16)]
    + [f"danger_{i}" for i in range(16)]
    + ["wall_left", "wall_right", "wall_top", "wall_bottom"]
    + ["enemy1_rel_x", "enemy1_rel_y", "enemy1_size", "enemy1_head_dx", "enemy1_head_dy"]
    + ["enemy1_trend", "enemy2_rel_x", "enemy2_rel_y", "enemy2_size", "kill_opportunity"]
    + ["danger_left", "danger_straight", "danger_right", "boost_available"]
    + ["free_space_left", "free_space_straight", "free_space_right"]
)
ABSOLUTE_FRAME = {0, 1, 2, 3, 5, 6, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51} | set(range(8, 40))


def load(data: Path):
    parts = [np.load(p) for p in sorted(data.glob("round-*/chunk-*-part-*.npz"))]
    keys = ("local", "global", "scalars", "state61", "world_seed")
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def fit(d, variant: str, device, epochs: int, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held = d["world_seed"] % 4 == 0
    loc = d["local"] if variant == "b" else d["local"][:, :6]
    sca = d["scalars"] if variant == "b" else d["scalars"][:, :12]
    y = d["state61"].astype(np.float64)
    mu, sd = y[~held].mean(0), y[~held].std(0) + 1e-6
    ys = ((y - mu) / sd).astype(np.float32)
    net = Ego2sNet(local_channels=loc.shape[1], n_scalars=sca.shape[1], n_out=61).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    tr = np.flatnonzero(~held)

    def head(b):
        out = net.fuse(
            torch.cat(
                [
                    net.local(torch.from_numpy(loc[b]).to(device).float() / 255.0),
                    net.glob(torch.from_numpy(d["global"][b]).to(device).float() / 255.0),
                    torch.from_numpy(sca[b]).to(device),
                ],
                dim=1,
            )
        )
        return net.adv(out)

    for _ in range(epochs):
        perm = rng.permutation(tr)
        for i in range(0, len(perm) - 511, 512):
            b = np.sort(perm[i : i + 512])
            loss = torch.nn.functional.mse_loss(head(b), torch.from_numpy(ys[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    te = np.flatnonzero(held)
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 2048):
            preds.append(head(te[i : i + 2048]).cpu().numpy())
    p = np.concatenate(preds)
    t = ys[te]
    ss_res = ((p - t) ** 2).sum(0)
    ss_tot = ((t - t.mean(0)) ** 2).sum(0)
    r2 = np.where(ss_tot > 1e-9 * len(t), 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    return r2, y[te].std(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    torch.set_num_threads(1)
    d = load(args.data)
    device = torch.device(args.device)
    r2d, sd = fit(d, "draft", device, args.epochs)
    r2b, _ = fit(d, "b", device, args.epochs)
    rows = []
    for i, name in enumerate(NAMES):
        rows.append(
            {
                "index": i,
                "name": name,
                "absolute_frame": i in ABSOLUTE_FRAME,
                "heldout_sd": float(sd[i]),
                "r2_draft": None if np.isnan(r2d[i]) else float(r2d[i]),
                "r2_b": None if np.isnan(r2b[i]) else float(r2b[i]),
            }
        )
    summary = {"samples": int(len(d["state61"])), "heldout": int((d["world_seed"] % 4 == 0).sum())}
    args.out.write_text(json.dumps({"summary": summary, "features": rows}, indent=1))
    for r in rows:
        print(
            f"{r['index']:2d} {r['name']:22s} abs={int(r['absolute_frame'])} "
            f"draft={r['r2_draft'] if r['r2_draft'] is None else round(r['r2_draft'], 3)} "
            f"b={r['r2_b'] if r['r2_b'] is None else round(r['r2_b'], 3)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
