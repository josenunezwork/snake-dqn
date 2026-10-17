"""Recheck saved T8 predictions, world isolation and bootstrap figures independently."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from research.redesign_m3c_20261008.obs_pack import train_layout, unpack  # noqa: E402
from research.t5b_risk_and_fuse_20261016.fuse import (  # noqa: E402
    T5_DATA,
    load_t5_data,
    saved_folds,
    sha256,
)
from research.t8_probes_20261017.cnn import DeathCNN, inner_split, predict  # noqa: E402

ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t8-probes-20261017")


def independent_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], world: np.ndarray) -> dict:
    """Weighted midrank implementation, independent of T5's classwise reduction kernel."""
    unique, groups = np.unique(world.astype(str), return_inverse=True)
    prepared = {}
    draws = {key: [] for key in scores}
    for key, values in scores.items():
        order = np.argsort(values, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
        sizes = np.diff(np.r_[starts, len(y)])
        prepared[key] = (order, starts, sizes)
    rng = np.random.default_rng(20261017)
    for _ in range(2000):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[groups]
        for key, (order, starts, sizes) in prepared.items():
            w, labels = weights[order].astype(float), y[order]
            grouped_weight = np.add.reduceat(w, starts)
            midrank = np.repeat(np.cumsum(grouped_weight) - grouped_weight / 2, sizes)
            positives = np.sum(w * labels)
            negatives = np.sum(w * (1 - labels))
            auc = (np.sum(w * labels * midrank) - positives**2 / 2) / (positives * negatives)
            draws[key].append(auc)
    results = {
        key: {
            "auc": float(roc_auc_score(y, values)),
            "ci95": np.quantile(draws[key], [0.025, 0.975]).tolist(),
        }
        for key, values in scores.items()
    }
    if "arrival" in scores:
        results["delta"] = {
            "auc_delta": results["arrival"]["auc"] - results["baseline"]["auc"],
            "ci95": np.quantile(
                np.array(draws["arrival"]) - draws["baseline"], [0.025, 0.975]
            ).tolist(),
        }
    return results


def check_predictions(path: Path, data: dict, fold: np.ndarray) -> dict:
    state, action = np.nonzero(data["legal"])
    with np.load(path, allow_pickle=False) as z:
        for key, expected in (
            ("state", state),
            ("action", action),
            ("y", data["died"][state, action].astype(np.uint8)),
            ("world", data["world"][state]),
            ("fold", fold[state]),
        ):
            np.testing.assert_array_equal(z[key], expected)
        keys = [key for key in ("cnn", "baseline", "arrival") if key in z.files]
        scores = {key: z[key] for key in keys}
        assert all(np.isfinite(value).all() for value in scores.values())
        return independent_bootstrap(z["y"], scores, z["world"])


def run(root: Path, cnn_only: bool) -> dict:
    torch.set_num_threads(1)
    data = load_t5_data(T5_DATA)
    fold = saved_folds(data)
    result = {"status": "PASS", "cnn": {}, "arrival": {}}
    cnn_dir = root / "cnn"
    summary = json.loads((cnn_dir / "summary.json").read_text())
    result["cnn"]["metrics"] = check_predictions(cnn_dir / "predictions.npz", data, fold)
    metric = result["cnn"]["metrics"]["cnn"]
    for key in ("auc", "ci95"):
        np.testing.assert_allclose(metric[key], summary["auc"]["metrics"]["cnn"][key], atol=1e-12)
    with np.load(cnn_dir / "inputs.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["state_id"], data["state_id"])
        local, glob = unpack(z["packed"], np.arange(len(data["state_id"])), train_layout())
    with np.load(cnn_dir / "predictions.npz", allow_pickle=False) as z:
        matrix = np.full((len(data["state_id"]), 6), np.nan)
        matrix[z["state"], z["action"]] = z["cnn"]
    maximum = 0.0
    for current in range(5):
        train, valid, test = inner_split(data["world"], fold, current)
        with np.load(cnn_dir / f"fold-{current}.split.npz", allow_pickle=False) as z:
            for key, expected in (("train", train), ("valid", valid), ("test", test)):
                np.testing.assert_array_equal(z[key], expected)
        sets = [set(data["world"][idx]) for idx in (train, valid, test)]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        # Trusted local artifact produced by this task; deliberately not a policy checkpoint.
        saved = torch.load(cnn_dir / f"fold-{current}.pt", map_location="cpu", weights_only=False)
        assert saved["kind"] == "diagnostic_death_classifier_NOT_POLICY"
        np.testing.assert_array_equal(saved["scalar_mean"], data["scalars"][train].mean(0))
        std = data["scalars"][train].std(0)
        std[std < 1e-6] = 1
        np.testing.assert_array_equal(saved["scalar_std"], std)
        history = summary["folds"][current]["history"]
        best, epoch, stale = float("inf"), 0, 0
        for row in history:
            if row["validation_bce"] < best - 0.0001:
                best, epoch, stale = row["validation_bce"], row["epoch"], 0
            else:
                stale += 1
        assert epoch == saved["best_epoch"] == summary["folds"][current]["best_epoch"]
        assert stale == summary["patience"] or len(history) == summary["max_epochs"]
        net = DeathCNN().eval()
        net.load_state_dict(saved["state_dict"])
        scalars = np.ascontiguousarray((data["scalars"] - saved["scalar_mean"]) / std)
        chosen = test[np.linspace(0, len(test) - 1, 12, dtype=int)]
        logits = predict(net, local, glob, scalars, chosen, "cpu", 12)
        prob = 1 / (1 + np.exp(-logits.astype(float)))
        mask = data["legal"][chosen]
        difference = float(np.max(np.abs(prob[mask] - matrix[chosen][mask])))
        assert difference < 2e-5, difference
        maximum = max(maximum, difference)
    result["cnn"]["cpu_vs_mps_60_state_max_abs_prediction_difference"] = maximum
    result["cnn"]["nested_split_and_train_only_scaling"] = "PASS"
    if not cnn_only:
        arrival_dir = root / "arrival-fit"
        reported = json.loads((arrival_dir / "summary.json").read_text())
        for model in ("hist_gradient_boosting", "logistic_regression"):
            metrics = check_predictions(arrival_dir / f"{model}.predictions.npz", data, fold)
            for actual, reference in (
                ("baseline", "baseline_60_24"),
                ("arrival", "arrival_60_24_18"),
                ("delta", "delta_vs_60_24"),
            ):
                for key, value in metrics[actual].items():
                    np.testing.assert_allclose(
                        value, reported["results"][model][reference][key], atol=1e-12
                    )
            result["arrival"][model] = metrics
    result["audited_predictions_sha256"] = {
        str(p): sha256(p) for p in sorted(root.glob("*/**/*predictions.npz"))
    }
    path = root / ("cnn-audit.json" if cnn_only else "audit.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cnn-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(run(args.root, args.cnn_only), indent=2))


if __name__ == "__main__":
    main()
