"""Raw-raster D300 classifier with nested world splits and T5's exact OOF folds."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from research.redesign_m3c_20261008.obs_pack import (  # noqa: E402
    train_layout, unpack)
from research.t5_separability_20261016.fit import bootstrap_auc  # noqa: E402
from research.t5b_risk_and_fuse_20261016.fuse import (BANK_ROOT,  # noqa: E402
                                                      T5_DATA, T5_FOLDS,
                                                      load_t5_data,
                                                      saved_folds, sha256,
                                                      source_packed)

SEED = 20261017
OUT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t8-probes-20261017/cnn")


class DeathCNN(nn.Module):
    """Two small convolution towers and an action-conditioned binary head.

    Towers are evaluated once per state; each of six logits explicitly receives
    its own one-hot action. The legal mask is used only in the loss/evaluation.
    """

    def __init__(self) -> None:
        super().__init__()

        def tower(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.local = tower(7)
        self.global_ = tower(4)
        self.head = nn.Sequential(
            nn.Linear(32 * (4 * 4 + 5 * 5) + 60 + 6, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 1),
        )
        self.register_buffer("actions", torch.eye(6))

    def forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((self.local(local), self.global_(glob), scalars), dim=1)
        expanded = features[:, None, :].expand(-1, 6, -1)
        actions = self.actions[None].expand(len(features), -1, -1)
        return self.head(torch.cat((expanded, actions), dim=2)).squeeze(-1)


def inner_split(world: np.ndarray, fold: np.ndarray, current: int) -> tuple[np.ndarray, ...]:
    """Label-independent 20% inner validation worlds; outer worlds never participate."""
    outer = fold == current
    worlds = np.unique(world[~outer].astype(str))
    rng = np.random.default_rng(SEED + current)
    valid_worlds = rng.permutation(worlds)[: max(1, int(np.ceil(len(worlds) * 0.2)))]
    valid = ~outer & np.isin(world.astype(str), valid_worlds)
    train = ~outer & ~valid
    if not train.any() or not valid.any() or not outer.any():
        raise ValueError("empty nested split")
    return np.flatnonzero(train), np.flatnonzero(valid), np.flatnonzero(outer)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Every observed legal branch has equal BCE weight; illegal labels have no effect."""
    if not legal.any():
        raise ValueError("batch has no legal branch")
    return F.binary_cross_entropy_with_logits(logits[legal], labels[legal])


def batch_inputs(
    local: np.ndarray, glob: np.ndarray, scalars: np.ndarray, rows: np.ndarray, device: str
) -> tuple[torch.Tensor, ...]:
    """Convert raw byte planes to [0,1]; scale reach-time by its fixed byte maximum."""
    loc = torch.from_numpy(local[rows]).to(device=device, dtype=torch.float32) / 255.0
    # Reach-time uses only 0..31; this invertible rescaling preserves all raster signal.
    loc[:, 5] *= 255.0 / 31.0
    return (
        loc,
        torch.from_numpy(glob[rows]).to(device=device, dtype=torch.float32) / 255.0,
        torch.from_numpy(scalars[rows]).to(device),
    )


def predict(
    net: DeathCNN,
    local: np.ndarray,
    glob: np.ndarray,
    scalars: np.ndarray,
    rows: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    net.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            inputs = batch_inputs(local, glob, scalars, rows[start : start + batch_size], device)
            result.append(net(*inputs).cpu().numpy())
    return np.concatenate(result)


def benchmark(batch_size: int = 128) -> dict[str, Any]:
    """Choose device by end-to-end forward/backward timing, never by outcome metrics."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    local = rng.integers(0, 256, (batch_size, 7, 31, 31), dtype=np.uint8)
    glob = rng.integers(0, 256, (batch_size, 4, 37, 37), dtype=np.uint8)
    scalars = np.zeros((batch_size, 60), np.float32)
    times = {}
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        torch.manual_seed(SEED)
        net = DeathCNN().to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        labels = torch.zeros((batch_size, 6), device=device)
        for iteration in range(24):
            if iteration == 4:
                if device == "mps":
                    torch.mps.synchronize()
                start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logits = net(*batch_inputs(local, glob, scalars, np.arange(batch_size), device))
            F.binary_cross_entropy_with_logits(logits, labels).backward()
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        times[device] = (time.monotonic() - start) / 20
    return {
        "seconds_per_batch": times,
        "selected": min(times, key=times.get),
        "batch_size": batch_size,
        "warmup_batches": 4,
        "timed_batches": 20,
    }


def run(
    out: Path,
    device: str,
    epochs: int = 80,
    patience: int = 8,
    batch_size: int = 128,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Fit five fresh diagnostic classifiers with inner-only early stopping."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite a diagnostic run: {out}")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    started = time.monotonic()
    paths = [
        Path(__file__),
        T5_DATA,
        T5_FOLDS,
        Path("research/t5b_risk_and_fuse_20261016/fuse.py"),
        Path("research/t5_separability_20261016/fit.py"),
        Path("research/redesign_m3c_20261008/obs_pack.py"),
    ]
    paths += [BANK_ROOT / "banks" / split / "manifest.json" for split in ("train", "holdout")]
    before = {str(p): sha256(p) for p in paths}
    timing = benchmark(batch_size)
    if device == "auto":
        device = timing["selected"]
    if device == "mps" and timing["selected"] != "mps":
        raise ValueError("spec permits MPS only if faster; benchmark selected CPU")
    data = load_t5_data(T5_DATA)
    fold = saved_folds(data)
    packed = source_packed(data, BANK_ROOT)
    local, glob = unpack(packed, np.arange(len(packed)), train_layout())
    np.savez_compressed(out / "inputs.npz", packed=packed, state_id=data["state_id"])
    del packed
    y, legal = data["died"].astype(np.float32), data["legal"].astype(bool)
    state, action = np.nonzero(legal)
    oof = np.full_like(y, np.nan)
    report: dict[str, Any] = {
        "authority": "supervised diagnostic only; no policy training or gate evidence",
        "device": device,
        "device_benchmark": timing,
        "seed": SEED,
        "parameters": sum(p.numel() for p in DeathCNN().parameters()),
        "max_epochs": epochs,
        "patience": patience,
        "batch_size_states": batch_size,
        "optimizer": "AdamW lr=0.001 weight_decay=0.0001; unweighted legal-branch BCE",
        "early_stopping": "minimum inner-world validation BCE; min_delta=0.0001",
        "source_sha256_before": before,
        "folds": [],
    }
    print(json.dumps({"setup": report}), flush=True)
    for current in range(5):
        torch.manual_seed(SEED + current)
        rng = np.random.default_rng(SEED + current)
        train, valid, test = inner_split(data["world"], fold, current)
        mean = data["scalars"][train].mean(0)
        std = data["scalars"][train].std(0)
        std[std < 1e-6] = 1
        scalars = np.ascontiguousarray((data["scalars"] - mean) / std, dtype=np.float32)
        net = DeathCNN().to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
        history = []
        for epoch in range(1, epochs + 1):
            epoch_start = time.monotonic()
            net.train()
            shuffled = rng.permutation(train)
            loss_sum, count = 0.0, 0
            for start in range(0, len(train), batch_size):
                rows = shuffled[start : start + batch_size]
                inputs = batch_inputs(local, glob, scalars, rows, device)
                labels = torch.from_numpy(y[rows]).to(device)
                mask = torch.from_numpy(legal[rows]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(net(*inputs), labels, mask)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite diagnostic loss")
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
                n = int(legal[rows].sum())
                loss_sum += float(loss.detach().cpu()) * n
                count += n
            logits = predict(net, local, glob, scalars, valid, device, batch_size)
            vl, vy = logits[legal[valid]], y[valid][legal[valid]]
            val_loss = float(np.mean(np.logaddexp(0.0, vl) - vy * vl))
            entry = {
                "epoch": epoch,
                "train_bce": loss_sum / count,
                "validation_bce": val_loss,
                "validation_auc": float(roc_auc_score(vy, vl)),
                "elapsed_s": time.monotonic() - epoch_start,
            }
            history.append(entry)
            print(json.dumps({"fold": current, **entry}), flush=True)
            if val_loss < best_loss - 0.0001:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("no finite validation checkpoint")
        net.load_state_dict(best_state)
        logits = predict(net, local, glob, scalars, test, device, batch_size)
        # Sigmoid is evaluated stably in float64, preserving ranking across folds.
        oof[test] = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
        torch.save(
            {
                "kind": "diagnostic_death_classifier_NOT_POLICY",
                "state_dict": best_state,
                "scalar_mean": mean,
                "scalar_std": std,
                "best_epoch": best_epoch,
            },
            out / f"fold-{current}.pt",
        )
        np.savez_compressed(out / f"fold-{current}.split.npz", train=train, valid=valid, test=test)
        report["folds"].append(
            {
                "fold": current,
                "best_epoch": best_epoch,
                "train_worlds": len(np.unique(data["world"][train])),
                "valid_worlds": len(np.unique(data["world"][valid])),
                "test_worlds": len(np.unique(data["world"][test])),
                "history": history,
            }
        )
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    scores = oof[state, action].astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("missing OOF predictions")
    np.savez_compressed(
        out / "predictions.npz",
        y=y[state, action].astype(np.uint8),
        state=state,
        action=action,
        world=data["world"][state],
        fold=fold[state],
        cnn=scores,
    )
    report["auc"] = bootstrap_auc(
        y[state, action].astype(np.uint8), {"cnn": scores}, data["world"][state], n_boot
    )
    after = {str(p): sha256(p) for p in paths}
    if before != after:
        raise RuntimeError("bound input or source changed during diagnostic fit")
    report["source_sha256_after"] = after
    report["elapsed_s"] = time.monotonic() - started
    report["artifact_sha256"] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"final": report["auc"], "elapsed_s": report["elapsed_s"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.benchmark_only:
            print(json.dumps(benchmark(), indent=2))
        else:
            run(args.out, args.device)


if __name__ == "__main__":
    main()
