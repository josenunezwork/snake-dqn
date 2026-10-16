"""CPU-only T5 grouped separability fits; no policy training or checkpoint changes.

Primary estimate pools five world-held-out predictions over the complete bank. An
additional train-to-original-holdout estimate preserves the bank's historical split.
All model choices are fixed, with no hyperparameter search or row-level early stopping.
"""

from __future__ import annotations

import os

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

import argparse  # noqa: E402 - thread limits must precede numerical imports
import hashlib  # noqa: E402 - thread limits must precede numerical imports
import json  # noqa: E402 - thread limits must precede numerical imports
import time  # noqa: E402 - thread limits must precede numerical imports
import warnings  # noqa: E402 - thread limits must precede numerical imports
from pathlib import Path  # noqa: E402 - thread limits must precede numerical imports
from typing import Any  # noqa: E402 - thread limits must precede numerical imports

import numpy as np  # noqa: E402 - thread limits must precede numerical imports
from sklearn import (  # noqa: E402 - thread limits must precede numerical imports
    __version__ as sklearn_version,
)
from sklearn.ensemble import (  # noqa: E402 - thread limits must precede numerical imports
    HistGradientBoostingClassifier,
)
from sklearn.exceptions import (  # noqa: E402 - thread limits must precede numerical imports
    ConvergenceWarning,
)
from sklearn.linear_model import (  # noqa: E402 - thread limits must precede numerical imports
    LogisticRegression,
)
from sklearn.metrics import (  # noqa: E402 - thread limits must precede numerical imports
    roc_auc_score,
)
from sklearn.pipeline import (  # noqa: E402 - thread limits must precede numerical imports
    make_pipeline,
)
from sklearn.preprocessing import (  # noqa: E402 - thread limits must precede numerical imports
    StandardScaler,
)
from threadpoolctl import (  # noqa: E402 - thread limits must precede numerical imports
    threadpool_limits,
)

SEED = 20261016
FEATURE_SETS = ("60", "60+24", "24")
MODELS = ("hist_gradient_boosting", "logistic_regression")


def world_folds(world: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Assign shuffled unique worlds to folds, independent of outcomes and row counts."""
    unique, inverse = np.unique(world.astype(str), return_inverse=True)
    if len(unique) < n_splits:
        raise ValueError("fewer worlds than folds")
    rng = np.random.default_rng(SEED)
    assignment = np.empty(len(unique), dtype=np.int16)
    assignment[rng.permutation(len(unique))] = np.arange(len(unique)) % n_splits
    return assignment[inverse]


def design(data: dict[str, np.ndarray], target: str, features: str) -> tuple:
    """Make label rows; action one-hot is common context in all death comparisons."""
    scalars, extra = data["scalars"], data["features"]
    if scalars.shape != (len(extra), 60) or extra.shape[1:] != (24,):
        raise ValueError("expected 60 existing scalars and 24 new features per state")
    matrix = {"60": scalars, "60+24": np.concatenate((scalars, extra), axis=1), "24": extra}[
        features
    ]
    legal, died, horizon = data["legal"], data["died"], data["horizon"]
    if not (legal.shape == died.shape == horizon.shape == (len(matrix), 6)):
        raise ValueError("branch arrays must have shape (N, 6)")
    if np.any(legal & ((horizon < 1) | (horizon > 300) | (~died & (horizon != 300)))):
        raise ValueError("legal branch lacks a complete D300 death/survival outcome")
    if target == "death_d300":
        state, action = np.nonzero(legal)
        x = np.concatenate((matrix[state], np.eye(6, dtype=np.float32)[action]), axis=1)
        y = died[state, action].astype(np.uint8)
    elif target == "boost_only_escape":
        # Quantification is over the bank's actually legal branches. At least one
        # normal branch is required; illegal actions are neither survived nor died.
        state = np.flatnonzero(legal[:, :3].any(axis=1))
        action = np.full(len(state), -1, dtype=np.int8)
        normal_survives = (legal[:, :3] & ~died[:, :3]).any(axis=1)
        boost_survives = (legal[:, 3:] & ~died[:, 3:]).any(axis=1)
        y = (boost_survives & ~normal_survives)[state].astype(np.uint8)
        x = matrix[state]
    else:
        raise ValueError(target)
    if not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ValueError("nonfinite design or only one label class")
    return x, y, state, action


def estimator(model: str) -> Any:
    """Fixed, untuned fit settings shared by all input variants."""
    if model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=SEED,
        )
    if model == "logistic_regression":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
        )
    raise ValueError(model)


def weighted_auc(y: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Exact weighted AUC including ties (also used to test the bootstrap kernel)."""
    order = np.argsort(scores, kind="stable")
    values, labels, weight = scores[order], y[order], weights[order]
    ends = np.r_[np.flatnonzero(np.diff(values)) + 1, len(values)]
    starts = np.r_[0, ends[:-1]]
    pos = np.add.reduceat(weight * labels, starts)
    neg = np.add.reduceat(weight * (1 - labels), starts)
    total = pos.sum() * neg.sum()
    if total == 0:
        return float("nan")
    return float(np.dot(pos, np.cumsum(neg) - 0.5 * neg) / total)


def bootstrap_auc(
    y: np.ndarray, predictions: dict[str, np.ndarray], world: np.ndarray, n_boot: int = 2000
) -> dict:
    """Paired percentile 95% CIs by resampling entire worlds with replacement."""
    unique, inverse = np.unique(world.astype(str), return_inverse=True)
    # Sort/tie-group once per prediction vector; each replicate changes world weights.
    prepared = {}
    for name, scores in predictions.items():
        order = np.argsort(scores, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
        prepared[name] = (inverse[order], y[order], starts)
    draws = {name: np.empty(n_boot, dtype=np.float64) for name in predictions}
    rng = np.random.default_rng(SEED + 1)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        for name, (group, label, starts) in prepared.items():
            weights = counts[group]
            pos = np.add.reduceat(weights * label, starts)
            neg = np.add.reduceat(weights * (1 - label), starts)
            denom = pos.sum() * neg.sum()
            draws[name][b] = (
                np.dot(pos.astype(float), np.cumsum(neg) - 0.5 * neg) / denom if denom else np.nan
            )
    result = {}
    for name, scores in predictions.items():
        good = draws[name][np.isfinite(draws[name])]
        result[name] = {
            "auc": float(roc_auc_score(y, scores)),
            "ci95": np.quantile(good, [0.025, 0.975]).tolist() if len(good) else None,
            "valid_bootstraps": len(good),
        }
    if "60+24" in draws and "60" in draws:
        delta = draws["60+24"] - draws["60"]
        good = delta[np.isfinite(delta)]
        result["delta_60+24_minus_60"] = {
            "auc_delta": result["60+24"]["auc"] - result["60"]["auc"],
            "ci95": np.quantile(good, [0.025, 0.975]).tolist() if len(good) else None,
            "valid_bootstraps": len(good),
        }
    return {"worlds": len(unique), "rows": len(y), "positive": int(y.sum()), "metrics": result}


def run(data_path: Path, output: Path, n_splits: int = 5, n_boot: int = 2000) -> dict:
    """Write predictions, exact fold assignment, provenance, and AUC receipts."""
    output.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    fold = world_folds(data["world"], n_splits)
    historical_train = data["split"].astype(str) == "train"
    historical_holdout = data["split"].astype(str) == "holdout"
    if not np.all(historical_train | historical_holdout):
        raise ValueError("unknown historical split")
    if set(data["world"][historical_train]) & set(data["world"][historical_holdout]):
        raise ValueError("historical train/holdout world overlap")
    np.savez_compressed(
        output / "folds.npz", world=data["world"], state_id=data["state_id"], fold=fold
    )
    report = {
        "schema": "t5-separability-v1",
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_path": str(data_path.resolve()),
        "sklearn_version": sklearn_version,
        "seed": SEED,
        "n_folds": n_splits,
        "bootstrap_world_draws": n_boot,
        "primary": "pooled world-grouped out-of-fold predictions over train + holdout",
        "secondary": "fit historical train only, score historical holdout only",
        "action_context": (
            "six one-hot action indicators appended equally to each per-action design"
        ),
        "sample_weight": "one per observed legal branch; one per state for boost-only",
        "boost_label": (
            "some legal boost survives D300 and all legal normals die; require a legal normal"
        ),
        "models": {name: str(estimator(name)) for name in MODELS},
        "results": {},
    }
    start = time.monotonic()
    for target in ("death_d300", "boost_only_escape"):
        target_result = {}
        for model in MODELS:
            predictions, held_predictions = {}, {}
            for features in FEATURE_SETS:
                x, y, state, action = design(data, target, features)
                row_fold = fold[state]
                pred = np.full(len(y), np.nan)
                for k in range(n_splits):
                    train, test = row_fold != k, row_fold == k
                    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                        raise ValueError(f"{target} fold {k} lacks both classes")
                    fitted = estimator(model)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        fitted.fit(x[train], y[train])
                    pred[test] = fitted.predict_proba(x[test])[:, 1]
                    print(
                        json.dumps(
                            {
                                "target": target,
                                "model": model,
                                "features": features,
                                "fold": k,
                                "auc": roc_auc_score(y[test], pred[test]),
                                "elapsed_s": round(time.monotonic() - start, 2),
                            }
                        ),
                        flush=True,
                    )
                if not np.isfinite(pred).all():
                    raise ValueError("missing OOF prediction")
                predictions[features] = pred
                train, held = historical_train[state], historical_holdout[state]
                fitted = estimator(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    fitted.fit(x[train], y[train])
                held_predictions[features] = fitted.predict_proba(x[held])[:, 1]
            np.savez_compressed(
                output / f"{target}.{model}.predictions.npz",
                y=y,
                state=state,
                action=action,
                world=data["world"][state],
                fold=fold[state],
                heldout=held,
                **{f"oof_{key}": value for key, value in predictions.items()},
                **{f"holdout_{key}": value for key, value in held_predictions.items()},
            )
            target_result[model] = {
                "grouped_oof": bootstrap_auc(y, predictions, data["world"][state], n_boot),
                "original_holdout": (
                    bootstrap_auc(y[held], held_predictions, data["world"][state][held], n_boot)
                    if len(np.unique(y[held])) == 2
                    else {
                        "auc": None,
                        "reason": "original holdout has only one class",
                        "rows": int(held.sum()),
                        "positive": int(y[held].sum()),
                    }
                ),
            }
            report["results"][target] = target_result
            (output / "fits.json").write_text(json.dumps(report, indent=2) + "\n")
    primary = report["results"]["death_d300"]["hist_gradient_boosting"]["grouped_oof"]["metrics"]
    delta = primary["delta_60+24_minus_60"]["auc_delta"]
    alone = primary["24"]["auc"]
    report["advisor_bar_verbatim"] = (
        "Proceed if the 60 + 24 AUC on per-action death beats the 60-scalar AUC by at least "
        "0.05 and the 24-alone AUC is at least 0.80."
    )
    report["separability_verdict"] = "PASS" if delta >= 0.05 and alone >= 0.80 else "FAIL"
    report["elapsed_s"] = time.monotonic() - start
    (output / "fits.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.out)


if __name__ == "__main__":
    main()
