"""Fixed-sample separation diagnostic for the M6 CTRL and T3 label finals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_env] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

from research.t3_rollout_labels_20261015.bank import LabelBank  # noqa: E402

ARMS = ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
SEEDS = range(10, 15)


def sha256(path: Path) -> str:
    """Return the SHA-256 for an input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def proportional_sample_indices(
    surviving: np.ndarray, action: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Choose a reproducible decisive/counter sample at its bank proportion."""
    if count < 1 or count > len(action):
        raise ValueError("sample count must be within the bank size")
    rows = np.arange(len(action))
    decisive = rows[~surviving[rows, action]]
    counter = rows[surviving[rows, action]]
    if not len(decisive) or not len(counter):
        raise ValueError("bank must contain both decisive and counter rows")
    decisive_count = min(max(int(round(count * len(decisive) / len(rows))), 1), len(decisive))
    counter_count = count - decisive_count
    if counter_count < 1 or counter_count > len(counter):
        raise ValueError("requested proportional sample cannot include both row classes")
    rng = np.random.default_rng(seed)
    indices = np.sort(
        np.concatenate(
            [
                rng.choice(decisive, decisive_count, replace=False),
                rng.choice(counter, counter_count, replace=False),
            ]
        )
    )
    return indices, {
        "total": int(len(rows)),
        "decisive_total": int(len(decisive)),
        "counter_total": int(len(counter)),
        "sample_total": int(len(indices)),
        "decisive_sample": decisive_count,
        "counter_sample": counter_count,
    }


def _quartiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"q25": None, "q50": None, "q75": None}
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75)}


def summarize_bank_q(
    q: np.ndarray, surviving: np.ndarray, mask: np.ndarray, b8_action: np.ndarray
) -> dict[str, Any]:
    """Summarize separation and raw/masked surviving-set agreement for Q rows."""
    rows = np.arange(len(q))
    best_surviving = np.where(surviving, q, -np.inf).max(axis=1)
    gap = best_surviving - q[rows, b8_action]
    raw_action = q.argmax(axis=1)
    masked_action = np.where(mask, q, -np.inf).argmax(axis=1)
    decisive = ~surviving[rows, b8_action]

    def group(select: np.ndarray) -> dict[str, Any]:
        selected_rows = rows[select]
        return {
            "n": int(len(selected_rows)),
            "separation_quartiles": _quartiles(gap[select]),
            "raw_agreement": float(surviving[selected_rows, raw_action[select]].mean()),
            "masked_agreement": float(surviving[selected_rows, masked_action[select]].mean()),
        }

    return {
        "semantics": {
            "separation": "max Q over stored surviving set minus Q of stored B8 action",
            "raw_agreement": (
                "stored surviving set contains unmasked Q argmax; matches learner_m3c bank_agree"
            ),
            "masked_agreement": (
                "stored surviving set contains advisory-mask Q argmax; supplemental"
            ),
        },
        "overall": group(np.ones(len(q), dtype=bool)),
        "decisive": group(decisive),
        "counter": group(~decisive),
    }


def _checkpoint_paths(ctrl_root: Path, t3_root: Path) -> dict[str, Path]:
    paths = {f"CTRL_s{seed}": ctrl_root / f"CTRL_s{seed}" / "final.pth" for seed in SEEDS}
    paths.update(
        {
            f"{arm}_s{seed}": t3_root / f"{arm}_s{seed}" / "final.pth"
            for arm in ARMS
            for seed in SEEDS
        }
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final checkpoints: {missing}")
    return paths


def require_finished_status(status_path: Path) -> None:
    """Fail closed unless all three T3 arms report successful seeds 10 through 14."""
    if not status_path.is_file():
        raise FileNotFoundError(f"missing T3 completion receipt: {status_path}")
    final: dict[str, int] = {}
    for line in status_path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[0].endswith(tuple(f"_s{seed}" for seed in SEEDS)):
            continue
        if not fields[1].startswith("exit="):
            continue
        try:
            final[fields[0]] = int(fields[1].removeprefix("exit="))
        except ValueError:
            continue
    missing = [
        f"{arm}_s{seed} exit=0"
        for arm in ARMS
        for seed in SEEDS
        if final.get(f"{arm}_s{seed}") != 0
    ]
    if missing:
        raise RuntimeError(f"not all T3 finals are complete: {missing}")


def run_bank_diagnostic(
    bank_dir: Path,
    ctrl_root: Path,
    t3_root: Path,
    status_path: Path,
    out: Path,
    *,
    sample_count: int = 2000,
    sample_seed: int = 20261016,
    batch_size: int = 256,
    margin_json: Path | None = None,
    margin_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate every final checkpoint on one fixed, provenance-preserved bank sample."""
    require_finished_status(status_path)
    import torch

    from src.core.device_manager import DeviceManager
    from src.model.ego2s_network import load_ego2s_checkpoint

    DeviceManager.override_device(torch.device("cpu"))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from research.redesign_m3c_20261008.obs_pack import train_layout

    bank = LabelBank(bank_dir, 60, pack=train_layout())
    indices, counts = proportional_sample_indices(
        bank.surviving, bank.action, sample_count, sample_seed
    )
    paths = _checkpoint_paths(ctrl_root, t3_root)
    local, global_, scalars = bank.local[indices], bank.global_[indices], bank.scalars[indices]
    bank_shards = {p.name: sha256(p) for p in sorted(bank_dir.glob("*.npz"))}
    checkpoint_hashes = {name: sha256(path) for name, path in paths.items()}
    result: dict[str, Any] = {
        "schema": "t3-bank-separation-diagnostic/v1",
        "device": "cpu",
        "torch_threads": 1,
        "sample": {
            "method": "numpy.default_rng choice without replacement, stratified decisive/counter",
            "seed": sample_seed,
            "counts": counts,
            "indices": indices.tolist(),
            "indices_sha256": hashlib.sha256(
                indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "inputs": {
            "bank_dir": str(bank_dir),
            "bank_shards": bank_shards,
            "status_path": str(status_path),
            "status_sha256": sha256(status_path),
            "checkpoints": {
                name: {"path": str(path), "sha256": checkpoint_hashes[name]}
                for name, path in paths.items()
            },
        },
        "results": {},
        "arm_pooled": {},
    }
    if margin_json is not None:
        margin = json.loads(margin_json.read_text())["margin"]
        result["inputs"]["margin"] = {
            "path": str(margin_json),
            "sha256": sha256(margin_json),
            "base_value": float(margin),
            "label_bank_margin_scale": float(margin_scale),
            "effective_value": float(margin) * float(margin_scale),
        }
    pooled_q: dict[str, list[np.ndarray]] = {"CTRL": []} | {arm: [] for arm in ARMS}
    for name, path in paths.items():
        net = load_ego2s_checkpoint(str(path), "cpu").eval()
        q = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                stop = min(start + batch_size, len(indices))
                q.append(
                    net(
                        torch.from_numpy(local[start:stop]),
                        torch.from_numpy(global_[start:stop]),
                        torch.from_numpy(scalars[start:stop]),
                    )
                    .cpu()
                    .numpy()
                )
        values = np.concatenate(q)
        result["results"][name] = summarize_bank_q(
            values, bank.surviving[indices], bank.mask[indices], bank.action[indices]
        )
        pooled_q[name.rsplit("_s", 1)[0]].append(values)
    for arm, values in pooled_q.items():
        result["arm_pooled"][arm] = summarize_bank_q(
            np.concatenate(values),
            np.tile(bank.surviving[indices], (len(values), 1)),
            np.tile(bank.mask[indices], (len(values), 1)),
            np.tile(bank.action[indices], len(values)),
        )
    if bank_shards != {p.name: sha256(p) for p in sorted(bank_dir.glob("*.npz"))}:
        raise RuntimeError("label-bank shard changed while diagnostic was running")
    if result["inputs"]["status_sha256"] != sha256(status_path):
        raise RuntimeError("T3 status receipt changed while diagnostic was running")
    if checkpoint_hashes != {name: sha256(path) for name, path in paths.items()}:
        raise RuntimeError("final checkpoint changed while diagnostic was running")
    if margin_json is not None and result["inputs"]["margin"]["sha256"] != sha256(margin_json):
        raise RuntimeError("margin receipt changed while diagnostic was running")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--ctrl-root", type=Path, required=True)
    parser.add_argument("--t3-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--sample-seed", type=int, default=20261016)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--margin-json", type=Path)
    parser.add_argument("--margin-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    result = run_bank_diagnostic(
        args.bank,
        args.ctrl_root,
        args.t3_root,
        args.status,
        args.out,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
        margin_json=args.margin_json,
        margin_scale=args.margin_scale,
    )
    print(json.dumps({"out": str(args.out), "sample": result["sample"]["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
