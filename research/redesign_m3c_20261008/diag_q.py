#!/usr/bin/env python3
"""M3-C diagnostic D1: what does M3-B's "Q drift" consist of? (development only).

Under the v2 reward (``gamma * Phi(L') - Phi(L)``, ``Phi = L / 10``, ``Phi(death) = 0``,
``-3`` on death, ``+0.3 * victim length`` per kill) the discounted return telescopes to
``-Phi(L_0) + U`` with ``U = -3 * gamma**T_death + sum_k gamma**k * kill_k``: the value of a
state is minus its own potential plus a small, size-independent "unshaped" part. A rising
``max |Q|`` can therefore be (a) bigger snakes in the data (``|Q| ~ L / 10``) or (b) a rising
``U`` (over-estimation or a genuine change in kill / death expectations).

This script evaluates checkpoints on a fixed set of held-out demonstration states (world
seeds ``% 10 == 0`` of rounds 10 and 12) and reports, per checkpoint: mean / max |Q|, the
greedy ``U = max_a Q + Phi(L)``, its quantiles and its mean in length bins, and the same for
the teacher Q stored in the shards. One thread, CPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEMO = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/data")
BINS = (0, 50, 150, 300, 600, 5000)


def load_states(shards_per_round: int) -> dict:
    parts = []
    for r in (10, 12):
        for p in sorted((DEMO / f"round-{r}").glob("chunk-*-part-*.npz"))[:shards_per_round]:
            with np.load(p) as z:
                keep = z["world_seed"] % 10 == 0
                parts.append(
                    {
                        k: z[k][keep]
                        for k in ("local", "global", "scalars", "q_teacher", "mask_resolved")
                    }
                )
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarize(q: np.ndarray, mask: np.ndarray, phi: np.ndarray) -> dict:
    m = mask.copy()
    m[~m.any(axis=1), :3] = True
    qg = np.where(m, q, -np.inf).max(axis=1)
    u = qg + phi
    L = phi * 10
    out = {
        "mean_q_greedy": float(qg.mean()),
        "max_abs_q": float(np.abs(q).max()),
        "frac_q_greedy_negative": float((qg < 0).mean()),
        "corr_qgreedy_phi": float(np.corrcoef(qg, phi)[0, 1]),
        "u_mean": float(u.mean()),
        "u_q": {str(p): float(np.quantile(u, p / 100)) for p in (1, 10, 50, 90, 99)},
        "u_by_length": {},
        "boost_gap_mean": float(
            (np.where(m[:, 3:], q[:, 3:], -np.inf).max(axis=1) - q[:, :3].max(axis=1))[
                m[:, 3:].any(axis=1)
            ].mean()
        ),
    }
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (L >= lo) & (L < hi)
        if sel.sum():
            out["u_by_length"][f"{lo}-{hi}"] = {
                "n": int(sel.sum()),
                "u_mean": float(u[sel].mean()),
                "q_greedy_mean": float(qg[sel].mean()),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpts", nargs="+", help="label=path")
    ap.add_argument("--shards-per-round", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    from src.model.ego2s_network import load_ego2s_checkpoint

    torch.set_num_threads(1)
    st = load_states(args.shards_per_round)
    phi = st["scalars"][:, 1].astype(np.float64) * 100.0  # scalars[1] = L / 1000
    res = {
        "n_states": int(len(phi)),
        "teacher": summarize(st["q_teacher"], st["mask_resolved"], phi),
    }
    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        net = load_ego2s_checkpoint(path, "cpu").eval()
        qs = []
        with torch.no_grad():
            for i in range(0, len(phi), 2048):
                sl = slice(i, i + 2048)
                qs.append(
                    net(
                        torch.from_numpy(st["local"][sl]),
                        torch.from_numpy(st["global"][sl]),
                        torch.from_numpy(st["scalars"][sl]),
                    ).numpy()
                )
        q = np.concatenate(qs).astype(np.float64)
        res[label] = summarize(q, st["mask_resolved"], phi)
        print(
            label, json.dumps({k: res[label][k] for k in ("mean_q_greedy", "max_abs_q", "u_mean")})
        )
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
