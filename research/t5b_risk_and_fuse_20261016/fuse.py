"""Evaluate B8 fuse representations on immutable T5 return-bank states.

This is a CPU-only development diagnostic.  It forwards fixed checkpoints over
the already-restored T5 states and fits fixed supervised classifiers; it never
rolls out a policy or changes a bank label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
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
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from research.redesign_m3c_20261008.obs_pack import train_layout, unpack  # noqa: E402
from research.t5_separability_20261016.fit import (  # noqa: E402
    bootstrap_auc,
    estimator,
    world_folds,
)

ARTIFACT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5b-risk-and-fuse-20261016")
T5_DATA = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/features.npz"
)
T5_FOLDS = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/fits/folds.npz"
)
BANK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-return-bank-20261016")
FINALS = {
    f"B8_s{k}": Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
        f"redesign-m4b-mac-20261013/B8_s{k}/final.pth"
    )
    for k in range(5)
}
EXPECTED_DATA_SHA = "e4ecbe991061c90bb2ae3cdb4b236d0025f1219d61fa806b7e96f43962d9e02a"
EXPECTED_MANIFESTS = {
    "train": "708e84d9463ebb890edeb3aa2ed99111cfbd259c939a3c2c211d6bf41c236a3d",
    "holdout": "c0ae22541d20927d1a7996f1a5a91c2aceb9591de8eb743c514def86e2c95e6c",
}
EXPECTED_FINALS = {
    "B8_s0": "bf6a42b387e6d643e943a03620a30128ec6032f984abb5a08559b9b0ac865f4f",
    "B8_s1": "a7b04a84c94ad6cf90f6461a2ab23b214a28c4c59ac9e7bc69c61ca3ae617753",
    "B8_s2": "5840ba615efd600325d5c5bcedf87b2cbb452fe0969d954d5797ab12f6917a0e",
    "B8_s3": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    "B8_s4": "6b6d3f37c6db129de941c3fb205ca6af77c6506db0793b1291656f767e4fb28f",
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_t5_data(path: Path) -> dict[str, np.ndarray]:
    """Load and pin the published T5 feature/label table."""
    if sha256(path) != EXPECTED_DATA_SHA:
        raise ValueError("T5 feature table SHA differs from the published restored-bank dataset")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if data["scalars"].shape != (19976, 60) or int(data["legal"].sum()) != 103264:
        raise ValueError("unexpected T5 state or legal-branch count")
    return data


def saved_folds(data: dict[str, np.ndarray], path: Path = T5_FOLDS) -> np.ndarray:
    """Load T5's persisted fold receipt and reject a regenerated-only mismatch."""
    with np.load(path, allow_pickle=False) as archive:
        world, state_id, fold = archive["world"], archive["state_id"], archive["fold"]
    if not np.array_equal(world, data["world"]) or not np.array_equal(state_id, data["state_id"]):
        raise ValueError("saved T5 folds do not align with canonical state rows")
    if not np.array_equal(fold, world_folds(data["world"])):
        raise ValueError("saved T5 folds differ from the published deterministic allocation")
    return fold


def source_packed(data: dict[str, np.ndarray], bank_root: Path) -> np.ndarray:
    """Hash-validate T4e shards and join each canonical state to packed bytes.

    Source shards may split one world into decisive/outside publications.  The
    same source decision must therefore agree byte-for-byte if it appears twice.
    """
    lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "holdout"):
        directory = bank_root / "banks" / split
        manifest_path = directory / "manifest.json"
        if sha256(manifest_path) != EXPECTED_MANIFESTS[split]:
            raise ValueError(f"{split} manifest differs from T5 restoration receipt")
        manifest = json.loads(manifest_path.read_text())
        paths = sorted(directory.glob("*.npz"))
        if set(manifest["shards"]) != {path.name for path in paths}:
            raise ValueError(f"{split} T4e shard set differs from manifest")
        for path in paths:
            shard_hash = sha256(path)
            if shard_hash != manifest["shards"][path.name]:
                raise ValueError(f"T4e shard hash mismatch: {path.name}")
            world = int(path.name.split("-")[1])
            with np.load(path, allow_pickle=False) as archive:
                packed, scalars, decisions = (
                    archive["state_packed"],
                    archive["state_scalars"],
                    archive["source_decision"],
                )
                for index, decision in enumerate(decisions):
                    key = (world, int(decision))
                    value = (packed[index].copy(), scalars[index].copy())
                    old = lookup.get(key)
                    if old is not None and (
                        not np.array_equal(old[0], value[0]) or not np.array_equal(old[1], value[1])
                    ):
                        raise ValueError(f"conflicting source observation at {key}")
                    lookup[key] = value
            if sha256(path) != shard_hash:
                raise RuntimeError(f"T4e shard changed while joining: {path.name}")
    selected: list[np.ndarray] = []
    for state_id, scalars in zip(data["state_id"], data["scalars"]):
        name, decision = str(state_id).split(":d")
        key = (int(name.split("-")[1]), int(decision))
        value = lookup.get(key)
        if value is None:
            raise ValueError(f"published T5 state missing from T4e shards: {state_id}")
        if value[1].dtype != scalars.dtype or value[1].tobytes() != scalars.tobytes():
            raise ValueError(f"T5 scalar byte mismatch for {state_id}")
        selected.append(value[0])
    return np.stack(selected)


def forward_final(
    checkpoint: Path, packed: np.ndarray, scalars: np.ndarray, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact T1C post-ReLU fuse output and Q values for immutable rows."""
    from src.model.ego2s_network import load_ego2s_checkpoint

    torch.set_num_threads(1)
    net = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
    if not isinstance(net.fuse[2], torch.nn.ReLU):
        raise ValueError("T1C fuse hook requires fuse[2] to be ReLU")
    local, glob = unpack(packed, np.arange(len(packed)), train_layout())
    features, values = [], []
    captured: list[np.ndarray] = []

    def hook(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        row = output.detach().cpu().numpy().copy()
        if row.ndim != 2 or row.shape[1] != 256 or not np.isfinite(row).all() or (row < 0).any():
            raise ValueError("invalid post-ReLU 256-d T1C fuse output")
        captured.append(row.astype(np.float32))

    handle = net.fuse[2].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0, len(packed), batch_size):
                end = min(start + batch_size, len(packed))
                q = net(
                    torch.from_numpy(local[start:end]),
                    torch.from_numpy(glob[start:end]),
                    torch.from_numpy(np.ascontiguousarray(scalars[start:end])),
                )
                values.append(q.detach().cpu().numpy().astype(np.float32))
    finally:
        handle.remove()
    features = np.concatenate(captured)
    q = np.concatenate(values)
    if features.shape != (len(packed), 256) or q.shape != (len(packed), 6):
        raise ValueError("fuse/Q row alignment failed")
    return features, q


def design(
    data: dict[str, np.ndarray], fuse: np.ndarray, q: np.ndarray, full: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create legal action rows with shared six-action context, as in T5."""
    if fuse.shape != (len(data["scalars"]), 256) or q.shape != (len(fuse), 6):
        raise ValueError("fuse/Q state alignment mismatch")
    state, action = np.nonzero(data["legal"])
    base = np.concatenate((fuse, q), axis=1)
    if full:
        base = np.concatenate((base, data["scalars"], data["features"]), axis=1)
    x = np.concatenate((base[state], np.eye(6, dtype=np.float32)[action]), axis=1)
    y = data["died"][state, action].astype(np.uint8)
    if x.shape[1] != (352 if full else 268) or len(np.unique(y)) != 2:
        raise ValueError("invalid fuse probe design")
    return x, y, state, action


def fit(
    data: dict[str, np.ndarray], fuse: np.ndarray, q: np.ndarray, output: Path, n_boot: int = 2000
) -> dict[str, Any]:
    """Fit fixed grouped-fold GBM/LR probes and write compact prediction receipts."""
    output.mkdir(parents=True, exist_ok=True)
    fold = saved_folds(data)
    result: dict[str, Any] = {
        "schema": "t5b-fuse-v1",
        "authority": "diagnostic only; not gate evidence",
        "data_sha256": EXPECTED_DATA_SHA,
        "fold_counts": np.bincount(fold, minlength=5).tolist(),
        "results": {},
    }
    for variant, full in (("fuse_q", False), ("fuse_q_60_24", True)):
        x, y, state, action = design(data, fuse, q, full)
        predictions: dict[str, np.ndarray] = {}
        for model in ("hist_gradient_boosting", "logistic_regression"):
            pred = np.full(len(y), np.nan)
            for current in range(5):
                train, test = fold[state] != current, fold[state] == current
                fitted = estimator(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    fitted.fit(x[train], y[train])
                pred[test] = fitted.predict_proba(x[test])[:, 1]
            if not np.isfinite(pred).all():
                raise ValueError("missing grouped OOF prediction")
            predictions[model] = pred
        np.savez_compressed(
            output / f"{variant}.predictions.npz",
            y=y,
            state=state,
            action=action,
            world=data["world"][state],
            fold=fold[state],
            **predictions,
        )
        # Reuse T5's tie-aware world bootstrap exactly; aliases avoid an irrelevant delta field.
        metrics = bootstrap_auc(
            y,
            {
                "fuse": predictions["hist_gradient_boosting"],
                "full": predictions["logistic_regression"],
            },
            data["world"][state],
            n_boot,
        )
        result["results"][variant] = {
            "rows": int(len(y)),
            "positive": int(y.sum()),
            "width": int(x.shape[1]),
            "hist_gradient_boosting": metrics["metrics"]["fuse"],
            "logistic_regression": metrics["metrics"]["full"],
        }
    (output / "fits.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(
    data_path: Path, bank_root: Path, out: Path, arms: list[str], n_boot: int = 2000
) -> dict[str, Any]:
    """Join, forward, and fit every requested final without changing source labels."""
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    bound_paths = [
        data_path,
        T5_FOLDS,
        Path(__file__),
        Path("research/t5_separability_20261016/fit.py"),
        Path("research/redesign_m3c_20261008/obs_pack.py"),
        Path("src/model/ego2s_network.py"),
    ]
    bound_paths += [bank_root / "banks" / split / "manifest.json" for split in ("train", "holdout")]
    before = {str(path): sha256(path) for path in bound_paths}
    if before[str(data_path)] != EXPECTED_DATA_SHA:
        raise ValueError("T5 feature table SHA differs from the published restored-bank dataset")
    data = load_t5_data(data_path)
    packed = source_packed(data, bank_root)
    summary: dict[str, Any] = {
        "authority": "diagnostic only; not gate evidence",
        "t5_data_sha256": sha256(data_path),
        "arms": {},
        "source_bank": str(bank_root),
        "states": len(packed),
        "legal_branches": int(data["legal"].sum()),
        "source_sha256_before": before,
    }
    for arm in arms:
        checkpoint = FINALS[arm]
        checkpoint_before = sha256(checkpoint)
        if checkpoint_before != EXPECTED_FINALS[arm]:
            raise ValueError(f"published B8 final hash mismatch: {arm}")
        fuse, q = forward_final(checkpoint, packed, data["scalars"])
        arm_out = out / arm
        np.savez_compressed(
            arm_out.with_suffix(".features.npz"),
            fuse=fuse,
            q=q,
            state_id=data["state_id"],
        )
        checkpoint_after = sha256(checkpoint)
        if checkpoint_after != checkpoint_before:
            raise RuntimeError(f"checkpoint changed while forwarding: {arm}")
        summary["arms"][arm] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_before,
            "feature_shape": list(fuse.shape),
            "q_shape": list(q.shape),
            "fits": fit(data, fuse, q, arm_out, n_boot),
        }
    summary["elapsed_s"] = time.monotonic() - start
    after = {str(path): sha256(path) for path in bound_paths}
    if before != after:
        raise RuntimeError("a bound source, T5 fold receipt, or bank manifest changed during run")
    summary["source_sha256_after"] = after
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=T5_DATA)
    parser.add_argument("--bank", type=Path, default=BANK_ROOT)
    parser.add_argument("--out", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--arms", nargs="+", choices=tuple(FINALS), default=list(FINALS))
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(run(args.data, args.bank, args.out, args.arms, args.boot), indent=2))


if __name__ == "__main__":
    main()
