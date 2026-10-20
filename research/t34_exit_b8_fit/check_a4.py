"""A4: actually fit B8's own labels with the tower off and measure Q drift.

Uses memory-mapped actual observations produced by ``prepare_a3_rows``. Checks
loss, backward pass and Adam update. No new gameplay or gate worlds are used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .loss import LossConfig, exit_fit_loss
from .prepare_a3_rows import sha256


def a4_identity(
    checkpoint: Path,
    rows: Path,
    *,
    states: int = 1024,
    batch_size: int = 32,
    updates: int = 20,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Run genuine no-tower self-distillation and compare all chosen states."""
    from src.model.ego2s_network import load_ego2s_checkpoint

    if min(states, batch_size, updates) < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("states, batch_size and updates must be positive; tolerance finite >=0")
    torch.set_num_threads(1)
    arrays = {
        name: np.load(rows / f"{name}.npy", mmap_mode="r")
        for name in ("local", "global", "scalars")
    }
    provenance = json.loads((rows / "provenance.json").read_text())
    for name in arrays:
        if provenance.get("arrays", {}).get(name) != sha256(rows / f"{name}.npy"):
            raise ValueError(f"A4 observation bytes differ from provenance: {name}")
    if any(len(value) < states for value in arrays.values()):
        raise ValueError(f"A4 needs at least {states} actual states")
    previous = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
    model = copy.deepcopy(previous).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)
    largest_loss = 0.0

    def tensors(start: int, stop: int) -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(np.array(value[start:stop], copy=True))
            for name, value in arrays.items()
        }

    for update in range(updates):
        start = (update * batch_size) % states
        batch = tensors(start, min(start + batch_size, states))
        with torch.no_grad():
            target = previous(batch["local"], batch["global"], batch["scalars"])
        batch.update(
            q_prev=target,
            action_star=target.argmax(1),
            forbidden_mask=torch.zeros_like(target, dtype=torch.bool),
            sample_weight=torch.ones(len(target)),
        )
        predicted = model(batch["local"], batch["global"], batch["scalars"])
        loss, _ = exit_fit_loss(predicted, batch, LossConfig())
        if not torch.isfinite(loss):
            raise ValueError("non-finite A4 fit loss")
        largest_loss = max(largest_loss, abs(float(loss.detach())))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    maximum = 0.0
    mismatches = 0
    with torch.inference_mode():
        for start in range(0, states, batch_size):
            batch = tensors(start, min(start + batch_size, states))
            before = previous(batch["local"], batch["global"], batch["scalars"])
            after = model(batch["local"], batch["global"], batch["scalars"])
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise ValueError("non-finite A4 Q values")
            maximum = max(maximum, float((before - after).abs().max()))
            mismatches += int((before.argmax(1) != after.argmax(1)).sum())
    return {
        "schema": "exit-fit-a4/v2",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "rows_provenance": provenance,
        "states": states,
        "optimizer_updates": updates,
        "batch_size": batch_size,
        "tower": False,
        "labels": "B8 argmax; empty forbidden set; B8 Q anchor",
        "max_abs_q_error": maximum,
        "argmax_mismatches": mismatches,
        "max_abs_loss": largest_loss,
        "tolerance": tolerance,
        "passed": maximum <= tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--states", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = a4_identity(
        args.checkpoint, args.rows, states=args.states, batch_size=args.batch, updates=args.updates
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
