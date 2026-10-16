"""Describe full source-B8 bootstrapped branch targets, never just realised G."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.redesign_m3c_20261008.obs_pack import PackLayout  # noqa: E402
from research.t4d_return_regression_20261016.bank import realized_returns  # noqa: E402
from research.t4e_return_20261016.bugcheck import distribution, predict, sha256  # noqa: E402
from src.model.ego2s_network import load_ego2s_checkpoint  # noqa: E402


def statistics(bank: Path, plans: Path) -> dict:
    """Use each row's frozen source B8 as selector and evaluator for descriptive y."""
    torch.set_num_threads(1)
    started = time.monotonic()
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    jobs = {
        j["plan_index"]: j
        for p in sorted(plans.glob("cycle-*.json"))
        for j in json.loads(p.read_text())["jobs"]
    }
    layout = PackLayout.from_dict(manifest["layout"])
    models, checkpoint_hashes = {}, {}
    families = {
        0: {"normal": [], "boost": [], "paired": [], "normal_alive": [], "boost_alive": []},
        1: {"normal": [], "boost": [], "paired": [], "normal_alive": [], "boost_alive": []},
    }
    if set(manifest["shards"]) != {p.name for p in bank.glob("*.npz")}:
        raise ValueError("shard set differs from manifest")
    for ordinal, (name, expected) in enumerate(sorted(manifest["shards"].items()), 1):
        path = bank / name
        if sha256(path) != expected:
            raise ValueError(f"changed bank shard {name}")
        index = int(name.split("-")[1])
        checkpoint = Path(jobs[index]["student"])
        if str(checkpoint) not in models:
            models[str(checkpoint)] = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
            checkpoint_hashes[str(checkpoint)] = sha256(checkpoint)
        model = models[str(checkpoint)]
        with np.load(path, allow_pickle=False) as data:
            gamma = float(manifest["gamma"])
            y = realized_returns(
                data["T"], data["died"], data["S"], data["L0"], data["L_T"], gamma, 1.0
            )
            live = ~data["died"]
            if live.any():
                q = predict(
                    model, data["terminal_packed"][live], data["terminal_scalars"][live], layout
                )
                mask = data["terminal_mask"][live].copy()
                mask[~mask.any(axis=1), :3] = True
                selected = np.where(mask, q, -np.inf).argmax(axis=1)
                y[live] += np.power(gamma, data["T"][live]) * q[np.arange(len(q)), selected]
            for family in (0, 1):
                group = families[family]
                keep = data["family"] == family
                for label, boost in (("normal", False), ("boost", True)):
                    selected = keep & ((data["action"] >= 3) == boost)
                    group[label].extend(y[selected].tolist())
                    group[label + "_alive"].extend(live[selected].tolist())
                for decision in np.unique(data["source_decision"][keep]):
                    state = keep & (data["source_decision"] == decision)
                    boost, normal = state & (data["action"] >= 3), state & (data["action"] < 3)
                    if boost.any() and normal.any():
                        group["paired"].append(float(y[boost].mean() - y[normal].mean()))
        if ordinal % 25 == 0:
            print(
                json.dumps(
                    {"bank": str(bank), "shards_done": ordinal, "shards": len(manifest["shards"])}
                ),
                flush=True,
            )
    result = {}
    for code, label in ((0, "decisive"), (1, "outside")):
        group = families[code]
        if not group["normal"]:
            continue
        result[label] = {
            "normal_y": distribution(group["normal"]),
            "boost_y": distribution(group["boost"]),
            "pooled_boost_minus_normal": float(np.mean(group["boost"]) - np.mean(group["normal"])),
            "paired_state_gap": distribution(group["paired"]),
            "normal_survival": float(np.mean(group["normal_alive"])),
            "boost_survival": float(np.mean(group["boost_alive"])),
        }
    return {
        "schema": "t4f-target-statistics/v1",
        "bank": str(bank),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": sha256(Path(__file__)),
        "target_convention": (
            "realized G + gamma^T masked source-B8 Q for survivor; death has no bootstrap"
        ),
        "families": result,
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = statistics(args.bank.resolve(), args.plans.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
