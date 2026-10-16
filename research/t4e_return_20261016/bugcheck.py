"""CPU-only, read-only numerical audit of the immutable T4e return bank.

No optimizer, actors, training entry point, or process pool is used. Checkpoints
are evaluated on identical saved observations; no gameplay score is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.redesign_m3c_20261008.obs_pack import PackLayout, unpack_ref  # noqa: E402
from research.t4d_return_regression_20261016.bank import realized_returns  # noqa: E402
from src.model.ego2s_network import load_ego2s_checkpoint  # noqa: E402


def sha256(path: Path) -> str:
    """Hash a saved input or output without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, Any]:
    """Describe every finite observation, with explicit population size."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("expected a nonempty finite population")
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        **dict(
            zip(
                ("min", "p05", "p25", "median", "p75", "p95", "max"),
                map(float, np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])),
            )
        ),
    }


def predict(net: Any, packed: np.ndarray, scalars: np.ndarray, layout: PackLayout) -> np.ndarray:
    """Bound memory to 128 rows and evaluate without gradients on CPU."""
    result = []
    with torch.inference_mode():
        for start in range(0, len(packed), 128):
            local, global_ = unpack_ref(packed[start : start + 128], layout)
            result.append(
                net(
                    torch.from_numpy(local),
                    torch.from_numpy(global_),
                    torch.from_numpy(scalars[start : start + 128]),
                ).numpy()
            )
    return np.concatenate(result) if result else np.empty((0, 6), np.float32)


def summarize(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Separate branch-weighted distributions from equal-state paired gaps."""
    result = {}
    for family, code in (("decisive", 0), ("outside", 1)):
        selected = rows["family"] == code
        if not selected.any():
            continue
        groups = {}
        for name, boost in (("nonboost", False), ("boost", True)):
            keep = selected & ((rows["action"] >= 3) == boost)
            groups[name] = {
                "survival_rate": float((~rows["died"][keep]).mean()),
                "y": distribution(rows["y"][keep]),
                "G": distribution(rows["G"][keep]),
                "bootstrap": distribution(rows["bootstrap"][keep]),
                "weighted_y_mean": float(np.average(rows["y"][keep], weights=rows["weight"][keep])),
                "b8_qa": distribution(rows["b8_qa"][keep]),
                "weighted_huber_derivative_mean": float(
                    np.mean(
                        0.0032
                        * rows["weight"][keep]
                        * np.clip(rows["b8_qa"][keep] - rows["y"][keep], -100, 100)
                    )
                ),
            }
        gaps, weighted_gaps = [], []
        for state in np.unique(rows["state_id"][selected]):
            keep = selected & (rows["state_id"] == state)
            boost = keep & (rows["action"] >= 3)
            normal = keep & (rows["action"] < 3)
            if boost.any() and normal.any():
                gap = rows["y"][boost].mean() - rows["y"][normal].mean()
                gaps.append(gap)
                weighted_gaps.append(rows["weight"][keep][0])
        groups["mean_y_gap_boost_minus_nonboost"] = (
            groups["boost"]["y"]["mean"] - groups["nonboost"]["y"]["mean"]
        )
        groups["paired_state_gap"] = distribution(np.asarray(gaps))
        groups["paired_state_positive_fraction"] = float((np.asarray(gaps) > 0).mean())
        groups["paired_state_weighted_gap_mean"] = float(np.average(gaps, weights=weighted_gaps))
        for model in ("b8", "ctrl_2m", "ctrl_final"):
            # Every branch is compared with that action's Q on exactly its s0.
            groups[model + "_qa"] = distribution(rows[model + "_qa"][selected])
            groups[model + "_y_minus_qa"] = distribution(
                rows["y"][selected] - rows[model + "_qa"][selected]
            )
        result[family] = groups
    return result


def audit(art: Path, out: Path) -> dict[str, Any]:
    """Stream all shards, verifying their manifest hashes before evaluation."""
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    inputs, models = {}, {}
    for index in range(5):
        paths = {
            "b8": art / f"redesign-m4b-mac-20261013/B8_s{index}/final.pth",
            "ctrl_2m": art / f"redesign-m6-mac-20261014/CTRL_s{index + 10}/ckpt_2000k.pth",
            "ctrl_final": art / f"redesign-m6-mac-20261014/CTRL_s{index + 10}/final.pth",
        }
        models[index] = {}
        for label, path in paths.items():
            inputs[str(path)] = sha256(path)
            models[index][label] = load_ego2s_checkpoint(str(path), "cpu")
    summary = {
        "schema": "t4e-bugcheck/v1",
        "device": "cpu",
        "processes": 1,
        "model_assignment": "source B8_sN; CTRL_s(10+N); online=target=B8 for bootstrap",
        "halted_checkpoint_evaluation": "UNAVAILABLE: neither halted run saved a checkpoint",
        "inputs": inputs,
        "banks": {},
    }
    summary["halted_runs"] = {}
    for seed in (10, 11):
        run = art / f"redesign-t4e-mac-20261016/HALTED_boost_RETURN_s{seed}"
        checkpoints = sorted(run.rglob("*.pth"))
        if checkpoints:
            raise ValueError(f"halted checkpoint now available; extend audit: {checkpoints}")
        files = {str(path): sha256(path) for path in sorted(run.rglob("*")) if path.is_file()}
        inputs.update(files)
        logs = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        summary["halted_runs"][str(seed)] = {"files": files, "logs": logs, "checkpoints": []}
    summary["source_hashes"] = {str(Path(__file__).relative_to(Path.cwd())): sha256(Path(__file__))}
    for split in ("train", "holdout"):
        root = art / "t4e-return-bank-20261016/banks" / split
        manifest = json.loads((root / "manifest.json").read_text())
        inputs[str(root / "manifest.json")] = sha256(root / "manifest.json")
        assert set(manifest["shards"]) == {p.name for p in root.glob("*.npz")}
        layout = PackLayout.from_dict(manifest["layout"])
        accumulated = defaultdict(list)
        state_values = defaultdict(list)
        state_count = 0
        for shard_index, (name, expected) in enumerate(sorted(manifest["shards"].items())):
            path = root / name
            assert sha256(path) == expected, path
            arm = int(re.search(r"B8_s(\d+)", name).group(1))
            with np.load(path, allow_pickle=False) as z:
                data = {key: z[key] for key in z.files if key != "manifest_json"}
            _, unique, inverse = np.unique(
                data["source_decision"], return_index=True, return_inverse=True
            )
            state_ids = inverse + state_count
            state_count += len(unique)
            state_values["family"].append(data["family"][unique])
            state_values["source_arm"].append(np.full(len(unique), arm))
            for label, net in models[arm].items():
                q = predict(
                    net, data["state_packed"][unique], data["state_scalars"][unique], layout
                )
                accumulated[label + "_qa"].append(q[inverse, data["action"]])
                state_values[label + "_all_q"].append(q)
            alive = ~data["died"]
            q_terminal = predict(
                models[arm]["b8"],
                data["terminal_packed"][alive],
                data["terminal_scalars"][alive],
                layout,
            )
            mask = data["terminal_mask"][alive].copy()
            mask[~mask.any(axis=1), :3] = True
            actions = np.where(mask, q_terminal, -1e9).argmax(axis=1)
            bootstrap = np.zeros(len(alive), np.float64)
            bootstrap[alive] = (
                manifest["gamma"] ** data["T"][alive] * q_terminal[np.arange(len(actions)), actions]
            )
            G = realized_returns(
                data["T"],
                data["died"],
                data["S"],
                data["L0"],
                data["L_T"],
                manifest["gamma"],
                manifest["kappa"],
            )
            for key in ("action", "family", "died", "T", "L0", "L_T", "S"):
                accumulated[key].append(data[key])
            for key, value in {
                "weight": data["row_weight"],
                "G": G,
                "bootstrap": bootstrap,
                "y": G + bootstrap,
                "state_id": state_ids,
                "source_arm": np.full(len(alive), arm),
            }.items():
                accumulated[key].append(value)
            if shard_index % 25 == 0:
                print(f"{split}: {shard_index + 1}/{len(manifest['shards'])} shards", flush=True)
        rows = {key: np.concatenate(value) for key, value in accumulated.items()}
        states = {key: np.concatenate(value) for key, value in state_values.items()}
        np.savez_compressed(out / f"{split}-rows.npz", **rows)
        np.savez_compressed(out / f"{split}-states.npz", **states)
        summary["banks"][split] = {
            "rows": len(rows["y"]),
            "states": state_count,
            "by_family": summarize(rows),
            "by_source_arm": {
                str(i): summarize({k: v[rows["source_arm"] == i] for k, v in rows.items()})
                for i in range(5)
            },
            "unique_state_all_six_q": {
                family: {
                    label: distribution(states[label + "_all_q"][states["family"] == code])
                    for label in ("b8", "ctrl_2m", "ctrl_final")
                }
                for family, code in (("decisive", 0), ("outside", 1))
                if (states["family"] == code).any()
            },
        }
    summary["wall_seconds"] = time.monotonic() - started
    summary["outputs"] = {p.name: sha256(p) for p in sorted(out.glob("*.npz"))}
    (out / "statistics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifacts, args.out)
    print(json.dumps({"wall_seconds": result["wall_seconds"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
