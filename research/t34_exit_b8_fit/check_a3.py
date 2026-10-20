#!/usr/bin/env python3
"""A3: verify a zero-initialized ExIt tower is exactly its B8 initializer.

The input is a directory containing memory-mappable ``local.npy``, ``global.npy``
and ``scalars.npy`` arrays captured from actual replay states.  It is streamed in
batches, so the 10,000-state acceptance check does not materialise a full
full-sight bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.t34_exit_b8_fit.full_sight import FULL_SIGHT_SHAPE
from research.t34_exit_b8_fit.network import ExitFitTowerNet
from research.t34_exit_b8_fit.prepare_a3_rows import sha256
from src.model.ego2s_network import load_ego2s_checkpoint


def _rows(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.load(directory / f"{name}.npy", mmap_mode="r") for name in ("local", "global", "scalars")
    )
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("A3 row arrays have inconsistent first dimensions")
    return arrays  # type: ignore[return-value]


def _checked_delta(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return an A3 delta, refusing non-finite output instead of masking it."""
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("A3 cannot pass non-finite B8 or tower Q values")
    delta = (candidate - reference).abs().max()
    if not torch.isfinite(delta):  # defensive if a future torch operation changes above semantics
        raise ValueError("A3 cannot pass a non-finite Q delta")
    return float(delta.item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b8", type=Path, required=True, help="published ego2s-bv8 B8 checkpoint")
    parser.add_argument(
        "--rows", type=Path, required=True, help="directory of actual replay-state .npy arrays"
    )
    parser.add_argument("--states", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.states <= 0 or args.batch <= 0:
        raise SystemExit("--states and --batch must be positive")
    torch.set_num_threads(1)
    local, glob, scalars = _rows(args.rows)
    if len(local) < args.states:
        raise SystemExit(f"A3 requires {args.states} actual states, got {len(local)}")
    base = load_ego2s_checkpoint(str(args.b8), "cpu").eval()
    tower = ExitFitTowerNet(base).eval()
    maximum = 0.0
    compared = 0
    with torch.inference_mode():
        for start in range(0, args.states, args.batch):
            stop = min(start + args.batch, args.states)
            loc = torch.from_numpy(np.ascontiguousarray(local[start:stop]))
            glo = torch.from_numpy(np.ascontiguousarray(glob[start:stop]))
            sca = torch.from_numpy(np.ascontiguousarray(scalars[start:stop]))
            # Eval's zero-fuse branch cannot consume the tower input. Allocate only
            # this batch nevertheless so the exact callable contract is exercised.
            sight = torch.zeros((stop - start, *FULL_SIGHT_SHAPE), dtype=torch.uint8)
            delta = _checked_delta(base(loc, glo, sca), tower(loc, glo, sca, sight))
            maximum = max(maximum, delta)
            compared += stop - start
    result = {
        "acceptance": "A3",
        "states": compared,
        "checkpoint_sha256": sha256(args.b8),
        "observation_sha256": {
            name: sha256(args.rows / f"{name}.npy") for name in ("local", "global", "scalars")
        },
        "max_abs_q_delta": maximum,
        "pass": maximum == 0.0,
    }
    print(json.dumps(result, sort_keys=True))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
