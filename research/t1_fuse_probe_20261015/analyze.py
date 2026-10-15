#!/usr/bin/env python3
"""Analyze T1C fuse-layer captures as a development diagnostic, not gate evidence.

The probe uses world-held-out predictions.  It intentionally shares a fold assignment
for a numeric world across arms, so a pooled B8 result cannot train on another arm's
copy of its held-out world.
"""

from __future__ import annotations

import os

# Set every native-pool limit before importing numpy or scikit-learn.  The capture
# and two unrelated training lanes can run while this diagnostic is fitting.
for _ENV in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_ENV] = "2"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import (  # noqa: E402
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np  # noqa: E402

DEFAULT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t1-fuse-probe-20261015/runs")
HORIZONS = (50, 100, 300)
CS = (0.01, 0.1, 1.0)


def _sklearn_dependencies() -> None:
    """Make the study's isolated sklearn target available when the venv lacks it."""
    if importlib.util.find_spec("sklearn") is not None:
        return
    target = (
        Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
        / "t1-fuse-probe-20261015/python-deps"
    )
    if target.is_dir() and str(target) not in sys.path:
        sys.path.insert(0, str(target))


def _thread_limit():
    """Constrain BLAS/OpenMP pools too, when threadpoolctl is available."""
    _sklearn_dependencies()
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=2)
    except ImportError:
        return nullcontext()


class WeightedAUC:
    """Tie-aware AUC whose world bootstrap only changes row weights.

    Ranks are computed once.  Bootstrap iterations therefore only perform compact
    ``bincount`` operations and never re-fit a classifier.
    """

    def __init__(self, score: np.ndarray, y: np.ndarray) -> None:
        self._y = np.asarray(y, dtype=np.float64)
        safe_score = np.asarray(score, dtype=np.float64)
        _, self._tie_group = np.unique(safe_score, return_inverse=True)
        self._groups = int(self._tie_group.max()) + 1 if len(safe_score) else 0

    def __call__(self, row_weight: np.ndarray) -> Optional[float]:
        positive = np.bincount(
            self._tie_group, weights=row_weight * self._y, minlength=self._groups
        )
        negative = np.bincount(
            self._tie_group, weights=row_weight * (1.0 - self._y), minlength=self._groups
        )
        n_positive, n_negative = positive.sum(), negative.sum()
        if n_positive <= 0.0 or n_negative <= 0.0:
            return None
        below = np.cumsum(negative) - negative
        return float((positive * (below + 0.5 * negative)).sum() / (n_positive * n_negative))


def folds_by_world(worlds: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Return deterministic folds while assigning every occurrence of a world together."""
    worlds = np.asarray(worlds)
    unique = np.unique(worlds)
    if len(unique) < 2:
        return np.zeros(len(worlds), dtype=np.int8)
    n_folds = min(int(k), len(unique))
    order = np.random.default_rng(seed).permutation(unique)
    mapping = {int(world): index % n_folds for index, world in enumerate(order)}
    return np.asarray([mapping[int(world)] for world in worlds], dtype=np.int8)


def _model(c_value: float):
    _sklearn_dependencies()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=0),
    )


def _safe_probability(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, c_value: float
) -> Tuple[np.ndarray, Optional[str]]:
    """Fit LR or return the observed prevalence when a train split is single-class."""
    classes = np.unique(y_train)
    if len(classes) < 2:
        prevalence = float(np.mean(y_train)) if len(y_train) else 0.0
        return np.full(len(X_test), prevalence, dtype=np.float64), (
            "single-class training fold; used training prevalence"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _thread_limit():
            fitted = _model(c_value).fit(X_train, y_train)
    messages = [str(item.message) for item in caught if "converg" in str(item.message).lower()]
    issue = "; ".join(messages) if messages else None
    return fitted.predict_proba(X_test)[:, 1].astype(np.float64), issue


def _auc(score: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    return WeightedAUC(score, y)(np.ones(len(y), dtype=np.float64))


def nested_oof(
    X: np.ndarray,
    y: np.ndarray,
    worlds: np.ndarray,
    outer_folds: int = 5,
    inner_folds: int = 3,
    c_grid: Sequence[float] = CS,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[int, float], List[str]]:
    """Nested group-world LR OOF predictions, selected Cs, and explicit fallbacks.

    Inner selection maximizes the valid pooled inner-OOF AUC.  If every inner AUC
    is undefined, the central fixed-grid value (0.1 when available) is used.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    worlds = np.asarray(worlds)
    fold = folds_by_world(worlds, outer_folds, seed)
    prediction = np.empty(len(y), dtype=np.float64)
    selected: Dict[int, float] = {}
    notes: List[str] = []
    default_c = 0.1 if 0.1 in c_grid else float(c_grid[len(c_grid) // 2])
    for outer in np.unique(fold):
        train, test = fold != outer, fold == outer
        train_worlds = worlds[train]
        inner = folds_by_world(train_worlds, inner_folds, seed + int(outer) + 1)
        scores: Dict[float, List[float]] = {float(c): [] for c in c_grid}
        for c_value in c_grid:
            inner_prediction = np.empty(int(train.sum()), dtype=np.float64)
            for current in np.unique(inner):
                inner_train, inner_test = inner != current, inner == current
                p, issue = _safe_probability(
                    X[train][inner_train],
                    y[train][inner_train],
                    X[train][inner_test],
                    float(c_value),
                )
                inner_prediction[inner_test] = p
                if issue:
                    notes.append(f"outer {int(outer)} inner {int(current)}: {issue}")
            value = _auc(inner_prediction, y[train])
            if value is not None:
                scores[float(c_value)].append(value)
        means = {c: float(np.mean(values)) for c, values in scores.items() if values}
        c_best = max(means, key=means.get) if means else default_c
        if not means:
            notes.append(f"outer {int(outer)}: all inner AUCs undefined; selected C={c_best}")
        selected[int(outer)] = float(c_best)
        p, issue = _safe_probability(X[train], y[train], X[test], float(c_best))
        prediction[test] = p
        if issue:
            notes.append(f"outer {int(outer)}: {issue}")
    return prediction, selected, notes


def _hgb_oof(X: np.ndarray, y: np.ndarray, worlds: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Fixed modest HGB comparison with the same five world-held-out folds."""
    _sklearn_dependencies()
    from sklearn.ensemble import HistGradientBoostingClassifier

    fold = folds_by_world(worlds)
    prediction = np.empty(len(y), dtype=np.float64)
    notes: List[str] = []
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    for current in np.unique(fold):
        train, test = fold != current, fold == current
        if len(np.unique(y[train])) < 2:
            prediction[test] = float(np.mean(y[train]))
            notes.append(f"HGB outer {int(current)}: single-class train; used prevalence")
            continue
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
        with _thread_limit():
            model.fit(X[train], y[train])
        prediction[test] = model.predict_proba(X[test])[:, 1]
    return prediction, notes


def auc_ci(
    score: np.ndarray, y: np.ndarray, worlds: np.ndarray, boot: int = 2000, seed: int = 1
) -> Dict[str, Optional[float]]:
    """Point AUC and world-clustered percentile 95% CI, without bootstrap refitting."""
    score, y, worlds = np.asarray(score), np.asarray(y), np.asarray(worlds)
    calculator = WeightedAUC(score, y)
    point = calculator(np.ones(len(y), dtype=np.float64))
    if point is None:
        return {
            "auc": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_valid": 0,
            "bootstrap_discarded": int(boot),
        }
    unique, inverse = np.unique(worlds, return_inverse=True)
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(int(boot)):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        value = calculator(counts[inverse].astype(np.float64))
        if value is not None:
            values.append(value)
    if not values:
        return {
            "auc": point,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_valid": 0,
            "bootstrap_discarded": int(boot),
        }
    return {
        "auc": point,
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "bootstrap_valid": len(values),
        "bootstrap_discarded": int(boot) - len(values),
    }


def reliability(prediction: np.ndarray, y: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    """Fixed-width reliability table with null observed rate for empty bins."""
    prediction, y = np.asarray(prediction), np.asarray(y)
    result: List[Dict[str, Any]] = []
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        mask = (prediction >= lo) & ((prediction < hi) if index < bins - 1 else (prediction <= hi))
        count = int(mask.sum())
        result.append(
            {
                "bin": index,
                "lo": lo,
                "hi": hi,
                "n": count,
                "mean_prediction": float(prediction[mask].mean()) if count else None,
                "observed_rate": float(y[mask].mean()) if count else None,
            }
        )
    return result


def _load_capture(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as capture:
        required = {"features", "arm", "world", "frame", "length", "q", "action"}
        required |= {f"{prefix}{horizon}" for prefix in ("y", "keep") for horizon in HORIZONS}
        missing = sorted(required - set(capture.files))
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        features = capture["features"]
        if features.ndim != 2 or features.shape[1] != 256:
            raise ValueError(f"{path}: expected features (N, 256), got {features.shape}")
        data = {key: capture[key] for key in required}
    sizes = {len(value) for value in data.values() if np.asarray(value).ndim > 0}
    if len(sizes) != 1:
        raise ValueError(f"{path}: inconsistent row counts {sorted(sizes)}")
    return data


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 input identity without loading another NPZ copy."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scope_data(
    captures: Mapping[str, Dict[str, np.ndarray]], arms: Iterable[str]
) -> Dict[str, np.ndarray]:
    chosen = [captures[arm] for arm in arms]
    keys = tuple(chosen[0])
    return {key: np.concatenate([np.asarray(item[key]) for item in chosen]) for key in keys}


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _q_signals(q: np.ndarray) -> Dict[str, np.ndarray]:
    ordered = np.sort(np.asarray(q, dtype=np.float64), axis=1)
    return {"max_q": ordered[:, -1], "top2_gap": ordered[:, -1] - ordered[:, -2]}


def _analyze_scope(
    name: str,
    data: Dict[str, np.ndarray],
    output_root: Path,
    boot: int,
    completed: Optional[Callable[[str, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"rows_raw": int(len(data["world"])), "by_horizon": {}}
    q_signals = _q_signals(data["q"])
    for horizon in HORIZONS:
        keep = np.asarray(data[f"keep{horizon}"], dtype=bool)
        X = data["features"][keep]
        y = data[f"y{horizon}"][keep].astype(np.int8)
        world = data["world"][keep]
        if not len(y):
            output["by_horizon"][str(horizon)] = {"rows": 0, "warning": "no kept rows"}
            if completed:
                completed(name, horizon, output)
            continue
        started = time.perf_counter()
        print(f"{name} H{horizon}: fitting nested LR", flush=True)
        lr_prediction, selected, notes = nested_oof(X, y, world)
        model_metrics: Dict[str, Any] = {
            "lr_nested": {
                **auc_ci(lr_prediction, y, world, boot),
                "selected_c_by_outer_fold": selected,
                "warnings": notes,
            }
        }
        try:
            print(f"{name} H{horizon}: fitting fixed HGB", flush=True)
            hgb_prediction, hgb_notes = _hgb_oof(X, y, world)
            model_metrics["hgb_fixed"] = {
                **auc_ci(hgb_prediction, y, world, boot),
                "warnings": hgb_notes,
            }
        except ImportError:
            hgb_prediction = np.full(len(y), np.nan)
            model_metrics["hgb_fixed"] = {"available": False, "warning": "scikit-learn unavailable"}
        signals: Dict[str, Any] = {}
        for signal_name, signal in q_signals.items():
            raw = signal[keep]
            signals[signal_name] = {
                "orientation": "higher raw score predicts death",
                "auc_raw_score": auc_ci(raw, y, world, boot),
                "auc_negative_score": auc_ci(-raw, y, world, boot),
            }
        result: Dict[str, Any] = {
            "rows": int(len(y)),
            "positives": int(y.sum()),
            "worlds": int(len(np.unique(world))),
            "models": model_metrics,
            "q_signals": signals,
            "fit_seconds": time.perf_counter() - started,
        }
        if horizon == 100:
            result["lr_reliability_10bin"] = reliability(lr_prediction, y)
        output["by_horizon"][str(horizon)] = result
        np.savez_compressed(
            output_root / f"oof_{name}_h{horizon}.npz",
            world=world.astype(np.int64),
            fold=folds_by_world(world).astype(np.int8),
            arm=np.asarray(data["arm"])[keep],
            frame=np.asarray(data["frame"])[keep].astype(np.int16),
            y=y.astype(np.int8),
            lr_prediction=lr_prediction.astype(np.float64),
            hgb_prediction=hgb_prediction.astype(np.float64),
        )
        print(f"{name} H{horizon}: complete in {result['fit_seconds']:.1f}s", flush=True)
        if completed:
            completed(name, horizon, output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="directory containing arm NPZs"
    )
    parser.add_argument("--boot", type=int, default=2000, help="world-cluster bootstrap replicates")
    parser.add_argument(
        "--resume", action="store_true", help="retain completed scopes in metrics.json"
    )
    args = parser.parse_args(argv)
    if args.boot < 1:
        parser.error("--boot must be positive")
    root = args.root
    paths = sorted(path for path in root.glob("*.npz") if not path.name.startswith("oof_"))
    captures = {path.stem: _load_capture(path) for path in paths}
    capture_sha256 = {path.stem: _sha256(path) for path in paths}
    if not captures:
        raise SystemExit(f"No capture NPZs under {root}")
    started = time.perf_counter()
    scopes: Dict[str, List[str]] = {arm: [arm] for arm in sorted(captures)}
    expected_b8 = [f"B8_s{index}" for index in range(5)]
    pooled_warning = None
    if all(arm in captures for arm in expected_b8):
        scopes["pooled_B8"] = expected_b8
    else:
        pooled_warning = "pooled_B8 omitted: requires all B8_s0 through B8_s4"
    metrics = root / "metrics.json"
    previous: Dict[str, Any] = {}
    if args.resume and metrics.exists():
        previous = json.loads(metrics.read_text())
    output: Dict[str, Any] = {
        "schema": "t1c-fuse-probe/v1",
        "authority": "development diagnostic only; not gate evidence",
        "methods": {
            "folds": "5 outer folds by numeric world, shared across arms; 3 inner world folds",
            "lr": {"standardized": True, "penalty": "L2", "c_grid": list(CS)},
            "hgb": {
                "model": "HistGradientBoostingClassifier",
                "learning_rate": 0.06,
                "max_iter": 150,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 80,
                "l2_regularization": 1.0,
                "early_stopping": False,
            },
            "ci": f"{args.boot}-sample world-cluster percentile bootstrap, no bootstrap refits",
            "thread_limit": 2,
            "numpy_version": np.__version__,
        },
        "capture_sha256": capture_sha256,
        "scopes": previous.get("scopes", {}) if args.resume else {},
    }
    if pooled_warning:
        output["warnings"] = [pooled_warning]
    try:
        _sklearn_dependencies()
        import sklearn

        output["methods"]["sklearn_version"] = sklearn.__version__
    except ImportError:
        output["methods"]["sklearn_version"] = None

    def scope_complete(scope: str) -> bool:
        """Only retain a resume scope when all horizons match this analysis setup."""
        old = previous.get("scopes", {}).get(scope, {})
        old_horizons = old.get("by_horizon", {})
        same_methods = previous.get("methods") == output["methods"]
        same_inputs = previous.get("capture_sha256") == output["capture_sha256"]
        oof_present = all((root / f"oof_{scope}_h{horizon}.npz").is_file() for horizon in HORIZONS)
        return (
            same_methods
            and same_inputs
            and oof_present
            and all(str(horizon) in old_horizons for horizon in HORIZONS)
        )

    def save_progress(scope: str, horizon: int, result: Dict[str, Any]) -> None:
        output["scopes"][scope] = result
        metrics.write_text(json.dumps(_json_clean(output), indent=2, allow_nan=False) + "\n")
        progress = root / "analysis_progress.json"
        progress.write_text(
            json.dumps({"scope": scope, "horizon": horizon, "complete": True}, indent=2) + "\n"
        )

    for name, arm_names in scopes.items():
        if args.resume and scope_complete(name):
            print(f"resuming: retaining completed scope {name}", flush=True)
            continue
        print(f"analyzing {name}: {', '.join(arm_names)}", flush=True)
        output["scopes"][name] = _analyze_scope(
            name, _scope_data(captures, arm_names), root, args.boot, save_progress
        )
    output["wall_seconds"] = time.perf_counter() - started
    metrics.write_text(json.dumps(_json_clean(output), indent=2, allow_nan=False) + "\n")
    print(f"wrote {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
