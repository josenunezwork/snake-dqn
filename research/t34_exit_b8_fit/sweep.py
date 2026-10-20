"""Execute the registered 3x3 tau/margin grid on actual saved T3 observations.

This offline sensitivity tool does real fitting and reports behavior at every
recorded length. Saved label rows do not replace the required full H5000 probe.
No new gameplay, gate data, cloud allocation, or serving change occurs here.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from .diagnostics import trajectory_diagnostics
from .loss import LossConfig, exit_fit_loss
from .prepare_a3_rows import sha256

TAUS = (0.02, 0.05, 0.1)
MARGINS = (0.05, 0.1, 0.25)


def run_sweep(run_one: Callable[[LossConfig], dict[str, Any]]) -> list[dict[str, Any]]:
    """Run each registered pair in fixed order and require measured diagnostics."""
    result = []
    for tau, margin in itertools.product(TAUS, MARGINS):
        row = dict(run_one(LossConfig(tau=tau, margin=margin)))
        if {"all_length_disagreement", "boost_share"}.difference(row):
            raise ValueError("sweep must report all_length_disagreement and boost_share")
        result.append({"tau": tau, "margin": margin, **row})
    return result


@contextmanager
def open_t3_shard(path: Path) -> Iterator[dict[str, np.ndarray]]:
    """Extract named NPY members once and mmap; never decompress a whole bank into RAM."""
    with tempfile.TemporaryDirectory(prefix="t34-t3-sweep-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            available = set(archive.namelist())
            observation = ("packed",) if "packed.npy" in available else ("local", "global")
            keys = (*observation, "scalars", "mask", "surviving", "length")
            for key in keys:
                with (
                    archive.open(key + ".npy") as reader,
                    (root / (key + ".npy")).open("xb") as writer,
                ):
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
        arrays = {key: np.load(root / (key + ".npy"), mmap_mode="r") for key in keys}
        if len({len(value) for value in arrays.values()}) != 1:
            raise ValueError("T3 arrays have inconsistent row counts")
        yield arrays


def t3_batch(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, torch.Tensor]:
    """Unpack only one fit batch of the original lossless T3 representation."""
    if "packed" in arrays:
        from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref

        local, glob = unpack_ref(arrays["packed"][indices], train_layout())
    else:
        local, glob = arrays["local"][indices], arrays["global"][indices]
    return {
        "local": torch.from_numpy(np.array(local, copy=True)),
        "global": torch.from_numpy(np.array(glob, copy=True)),
        "scalars": torch.from_numpy(np.array(arrays["scalars"][indices], copy=True)),
    }


def fit_pair(
    previous: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    config: LossConfig,
    *,
    epochs: int,
    batch_size: int,
    max_rows: int | None,
    max_updates: int | None,
    seed: int,
) -> dict[str, Any]:
    """Fit one B8 initializer and measure raw plus advisory-masked behavior."""
    model = copy.deepcopy(previous).train()
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)
    count = min(len(arrays["scalars"]), max_rows or len(arrays["scalars"]))
    if count < 10:
        raise ValueError("need at least ten T3 rows for empirical length deciles")
    rng = np.random.default_rng(seed)
    updates = 0
    started = time.perf_counter()
    for _ in range(epochs):
        order = rng.permutation(count)
        for start in range(0, count, batch_size):
            picked = order[start : start + batch_size]
            batch = {key: value.to(device) for key, value in t3_batch(arrays, picked).items()}
            with torch.no_grad():
                anchor = previous(batch["local"], batch["global"], batch["scalars"])
            surviving = torch.as_tensor(
                np.array(arrays["surviving"][picked], copy=True), device=device
            ).bool()
            legal = torch.as_tensor(
                np.array(arrays["mask"][picked], copy=True), device=device
            ).bool()
            if not surviving.any(1).all() or (surviving & ~legal).any():
                raise ValueError("T3 surviving sets must be nonempty and advisory-legal")
            target = anchor.masked_fill(~surviving, -torch.inf).argmax(1)
            batch.update(
                q_prev=anchor,
                action_star=target,
                forbidden_mask=legal & ~surviving,
                sample_weight=torch.ones(len(picked), device=device),
            )
            predicted = model(batch["local"], batch["global"], batch["scalars"])
            loss, _ = exit_fit_loss(predicted, batch, config)
            if not torch.isfinite(loss):
                raise ValueError("non-finite T3 sweep loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            if max_updates is not None and updates >= max_updates:
                break
        if max_updates is not None and updates >= max_updates:
            break
    old, new, old_masked, new_masked, lengths = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            picked = np.arange(start, min(start + batch_size, count))
            batch = {key: value.to(device) for key, value in t3_batch(arrays, picked).items()}
            anchor = previous(batch["local"], batch["global"], batch["scalars"])
            student = model(batch["local"], batch["global"], batch["scalars"])
            if not torch.isfinite(student).all() or not torch.isfinite(anchor).all():
                raise ValueError("non-finite sweep diagnostic Q values")
            legal = torch.as_tensor(
                np.array(arrays["mask"][picked], copy=True), device=device
            ).bool()
            if not legal.any(1).all():
                raise ValueError("T3 diagnostic row lacks advisory-legal actions")
            old.extend(anchor.argmax(1).cpu().tolist())
            new.extend(student.argmax(1).cpu().tolist())
            old_masked.extend(anchor.masked_fill(~legal, -torch.inf).argmax(1).cpu().tolist())
            new_masked.extend(student.masked_fill(~legal, -torch.inf).argmax(1).cpu().tolist())
            lengths.extend(arrays["length"][picked].tolist())
    diagnostics = trajectory_diagnostics(
        previous_actions=np.asarray(old), actions=np.asarray(new), lengths=np.asarray(lengths)
    )
    return {
        "rows": count,
        "optimizer_updates": updates,
        "elapsed_seconds": time.perf_counter() - started,
        "all_length_disagreement": float(np.mean(np.asarray(old) != np.asarray(new))),
        "advisory_masked_disagreement": float(
            np.mean(np.asarray(old_masked) != np.asarray(new_masked))
        ),
        "advisory_masked_boost_share": float(np.mean(np.asarray(new_masked) >= 3)),
        **diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True, help="one original T3 bank NPZ")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=None, help="tiny smoke prefix only")
    parser.add_argument(
        "--updates", type=int, default=None, help="tiny smoke updates per grid cell"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if not 4 <= args.epochs <= 6 or args.batch < 1:
        parser.error("epochs must be 4..6 and batch positive")
    if (args.updates is not None and args.updates < 1) or (
        args.max_rows is not None and args.max_rows < 10
    ):
        parser.error("updates positive; max-rows >=10")
    from src.model.ego2s_network import load_ego2s_checkpoint

    torch.set_num_threads(1)
    previous = load_ego2s_checkpoint(str(args.checkpoint), args.device).eval()
    with open_t3_shard(args.bank) as arrays:
        result = run_sweep(
            lambda config: fit_pair(
                previous,
                arrays,
                config,
                epochs=args.epochs,
                batch_size=args.batch,
                max_rows=args.max_rows,
                max_updates=args.updates,
                seed=args.seed,
            )
        )
    report = {
        "schema": "exit-fit-t3-sweep/v2",
        "smoke_only": args.max_rows is not None or args.updates is not None,
        "scope": "All lengths in saved T3 rows; not the full H5000 tripwire probe",
        "weighting": (
            "Uniform legacy T3 rows; P4 DAgger/death-window weights belong to the export loader"
        ),
        "bank_sha256": sha256(args.bank),
        "checkpoint_sha256": sha256(args.checkpoint),
        "epochs_requested": args.epochs,
        "rows": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps({"out": str(args.out), "cells": len(result), "smoke_only": report["smoke_only"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
