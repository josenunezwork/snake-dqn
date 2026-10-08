#!/usr/bin/env python3
"""Why does the ego2s-b student under-boost? (pre-M3 item 3; held-out distillation worlds).

Evaluates a student on the held-out worlds (``world_seed % 10 == 0``) of the given data
rounds and reports, for boost-labelled decisions (``a_v8 >= 3``) vs the rest:

* label rate; the teacher's Q gap between its best boost and best normal action;
* the student's same gap, its predicted-boost rate, agreement, and teacher-Q regret;
* breakdown by body length, kill opportunity (scalar 19) and nearest-enemy distance.
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

from src.model.ego2s_network import load_ego2s_checkpoint  # noqa: E402

KEYS = ("local", "global", "scalars", "q_teacher", "a_v8", "mask_resolved", "length")


def load_val(data: Path, rounds):
    parts = []
    for r in rounds:
        for p in sorted((data / f"round-{r}").glob("chunk-*-part-*.npz")):
            with np.load(p) as z:
                k = z["world_seed"] % 10 == 0
                if k.any():
                    parts.append({n: z[n][k] for n in KEYS})
    return {n: np.concatenate([p[n] for p in parts]) for n in KEYS}


def summarize(sel, d, q):
    m = d["mask_resolved"].copy()
    m[~m.any(1), :3] = True
    pred = np.where(m, q, -np.inf).argmax(1)
    qt = d["q_teacher"]
    idx = np.arange(len(pred))
    t_gap = np.where(m[:, 3:], qt[:, 3:], -np.inf).max(1) - np.where(
        m[:, :3], qt[:, :3], -np.inf
    ).max(1)
    s_gap = np.where(m[:, 3:], q[:, 3:], -np.inf).max(1) - np.where(
        m[:, :3], q[:, :3], -np.inf
    ).max(1)
    regret = qt[idx, d["a_v8"]] - qt[idx, pred]
    fin = np.isfinite(t_gap) & sel
    return {
        "n": int(sel.sum()),
        "agree": float((pred == d["a_v8"])[sel].mean()) if sel.any() else None,
        "pred_boost": float((pred >= 3)[sel].mean()) if sel.any() else None,
        "teacher_boost_gap_median": float(np.median(t_gap[fin])) if fin.any() else None,
        "student_boost_gap_median": float(np.median(s_gap[fin])) if fin.any() else None,
        "student_gap_positive": float((s_gap[fin] > 0).mean()) if fin.any() else None,
        "regret_mean": float(regret[sel].mean()) if sel.any() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--rounds", default="10,11,12")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    d = load_val(args.data, [int(r) for r in args.rounds.split(",")])
    net = load_ego2s_checkpoint(args.student, args.device)
    qs = []
    with torch.no_grad():
        for i in range(0, len(d["a_v8"]), 4096):
            t = [
                torch.from_numpy(np.ascontiguousarray(d[k][i : i + 4096])).to(args.device)
                for k in ("local", "global", "scalars")
            ]
            qs.append(net(*t).float().cpu().numpy())
    q = np.concatenate(qs)
    boost = d["a_v8"] >= 3
    can = d["mask_resolved"][:, 3:].any(1)
    out = {
        "n": int(len(boost)),
        "boost_label_rate": float(boost.mean()),
        "boost_legal_rate": float(can.mean()),
        "boost_labels": summarize(boost, d, q),
        "normal_labels_boost_legal": summarize(~boost & can, d, q),
        "by_length": {},
    }
    for lo, hi in ((0, 50), (50, 200), (200, 500), (500, 10**6)):
        sel_len = (d["length"] >= lo) & (d["length"] < hi)
        out["by_length"][f"{lo}-{hi}"] = {
            "boost_label_rate": float(boost[sel_len].mean()) if sel_len.any() else None,
            "boost_labels": summarize(boost & sel_len, d, q),
        }
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
