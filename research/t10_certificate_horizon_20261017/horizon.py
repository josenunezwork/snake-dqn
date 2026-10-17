"""T10 horizon-stratified T5-bank GBM diagnostic.

This refits the fixed T5 histogram gradient booster at five known outcome
horizons.  It is a diagnostic only: it does not restore or roll out a policy,
and it never changes the immutable T5 outcomes.
"""

from __future__ import annotations

import argparse
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
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from research.t5_separability_20261016.fit import bootstrap_auc  # noqa: E402
from research.t5_separability_20261016.fit import estimator  # noqa: E402
from research.t8_probes_20261017.arrival_fit import align_arrival  # noqa: E402

from research.t5b_risk_and_fuse_20261016.fuse import (  # isort: skip # noqa: E402
    T5_DATA,
    T5_FOLDS,
    load_t5_data,
    saved_folds,
    sha256,
)

HORIZONS = (30, 60, 100, 200, 300)
T8_ARRIVAL = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t8-probes-20261017/arrival/features.npz"
)
T5B_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5b-risk-and-fuse-20261016")
DEFAULT_FUSES = {
    f"fuse_q_B8_s{seed}": T5B_ROOT / "runs" / f"B8_s{seed}" / f"B8_s{seed}.features.npz"
    for seed in range(5)
}


def death_within(data: dict[str, np.ndarray], horizon: int) -> np.ndarray:
    """Return known legal branch labels for death on or before ``horizon``.

    T5 records every legal branch through D300.  A death at the horizon is
    positive, including D300, while a surviving branch is negative at every
    requested horizon.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon {horizon}; expected one of {HORIZONS}")
    legal, died, branch_horizon = data["legal"], data["died"], data["horizon"]
    if not (legal.shape == died.shape == branch_horizon.shape and legal.ndim == 2):
        raise ValueError("T5 branch arrays must have the same two-dimensional shape")
    complete = legal & (
        (died & (branch_horizon >= 1) & (branch_horizon <= 300)) | (~died & (branch_horizon == 300))
    )
    if not np.array_equal(complete, legal):
        raise ValueError("legal branch lacks a complete D300 death/survival outcome")
    return (died & (branch_horizon <= horizon)).astype(np.uint8)


def load_fuse(data: dict[str, np.ndarray], path: Path) -> np.ndarray:
    """Load one T5b fuse+Q feature receipt in canonical T5 state order."""
    with np.load(path, allow_pickle=False) as archive:
        fuse, q, state_id = archive["fuse"], archive["q"], archive["state_id"]
    if fuse.shape != (len(data["state_id"]), 256) or q.shape != (len(fuse), 6):
        raise ValueError(f"unexpected fuse/Q shape in {path}")
    if not np.array_equal(state_id, data["state_id"]):
        raise ValueError(f"T5b fuse state identities differ from canonical T5 bank: {path}")
    matrix = np.concatenate((fuse, q), axis=1)
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError(f"invalid fuse values in {path}")
    return matrix


def designs(
    data: dict[str, np.ndarray], arrival: np.ndarray | None, fuses: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Build state-level features; action identity is appended after row expansion."""
    scalars, extra = data["scalars"], data["features"]
    if scalars.shape != (len(extra), 60) or extra.shape != (len(scalars), 24):
        raise ValueError("expected canonical T5 60 scalar and 24 feature columns")
    result = {"60": scalars, "60+24": np.concatenate((scalars, extra), axis=1)}
    if arrival is not None:
        if arrival.shape != (len(scalars), 18):
            raise ValueError("arrival features must be canonical (N,18)")
        result["60+24+18_arrival"] = np.concatenate((scalars, extra, arrival), axis=1)
    for name, matrix in fuses.items():
        if matrix.shape != (len(scalars), 262):
            raise ValueError(f"{name} must contain 256 fuse plus six Q columns")
        result[name] = matrix
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite diagnostic design")
    return result


def action_rows(
    matrix: np.ndarray, legal: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand a state feature matrix to actual legal per-action rows."""
    state, action = np.nonzero(legal)
    x = np.concatenate((matrix[state], np.eye(6, dtype=np.float32)[action]), axis=1)
    y = labels[state, action]
    if len(np.unique(y)) != 2:
        raise ValueError("horizon target has only one class")
    return x, y, state, action


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    """Hash every source input once, rejecting missing files before a fit starts."""
    return {str(path): sha256(path) for path in paths}


def run(
    output: Path,
    *,
    data_path: Path = T5_DATA,
    arrival_path: Path | None = T8_ARRIVAL,
    fuse_paths: dict[str, Path] | None = None,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Fit all requested horizons and persist OOF predictions plus provenance."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    fuse_paths = DEFAULT_FUSES if fuse_paths is None else fuse_paths
    sources = [
        data_path,
        T5_FOLDS,
        Path(__file__),
        Path("research/t5_separability_20261016/fit.py"),
        Path("research/t5b_risk_and_fuse_20261016/fuse.py"),
        Path("research/t8_probes_20261017/arrival_fit.py"),
    ]
    if arrival_path is not None:
        sources.append(arrival_path)
    sources.extend(fuse_paths.values())
    before = _source_hashes(sources)
    output.mkdir(parents=True)
    started = time.monotonic()
    data = load_t5_data(data_path)
    # saved_folds intentionally verifies T5's original persisted allocation, not a re-split.
    fold = saved_folds(data)
    arrival = align_arrival(data, arrival_path) if arrival_path is not None else None
    fuses = {name: load_fuse(data, path) for name, path in fuse_paths.items()}
    state_designs = designs(data, arrival, fuses)
    report: dict[str, Any] = {
        "schema": "t10-horizon-v1",
        "authority": "diagnostic only; no policy training, rollout, gate, or serving evidence",
        "data_sha256": before[str(data_path)],
        "fold_receipt_sha256": before[str(T5_FOLDS)],
        "horizons": list(HORIZONS),
        "bootstrap_world_draws": n_boot,
        "bootstrap": "T5 whole-world percentile bootstrap; seed 20261017; fixed OOF predictions",
        "model": str(estimator("hist_gradient_boosting")),
        "feature_sets": {name: int(matrix.shape[1] + 6) for name, matrix in state_designs.items()},
        "source_sha256_before": before,
        "results": {},
    }
    for horizon in HORIZONS:
        labels = death_within(data, horizon)
        predictions: dict[str, np.ndarray] = {}
        row_state = row_action = row_y = None
        for name, matrix in state_designs.items():
            x, y, state, action = action_rows(matrix, data["legal"], labels)
            if row_state is None:
                row_state, row_action, row_y = state, action, y
            elif not (
                np.array_equal(row_state, state)
                and np.array_equal(row_action, action)
                and np.array_equal(row_y, y)
            ):
                raise RuntimeError("diagnostic designs expanded to incompatible branch rows")
            prediction = np.full(len(y), np.nan)
            for current in range(5):
                train, test = fold[state] != current, fold[state] == current
                if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                    raise ValueError(f"D{horizon} fold {current} lacks both classes")
                fitted = estimator("hist_gradient_boosting")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    fitted.fit(x[train], y[train])
                prediction[test] = fitted.predict_proba(x[test])[:, 1]
                print(
                    json.dumps(
                        {
                            "horizon": horizon,
                            "features": name,
                            "fold": current,
                            "auc": roc_auc_score(y[test], prediction[test]),
                            "elapsed_s": round(time.monotonic() - started, 2),
                        }
                    ),
                    flush=True,
                )
            if not np.isfinite(prediction).all():
                raise RuntimeError("missing grouped out-of-fold prediction")
            predictions[name] = prediction
        assert row_state is not None and row_action is not None and row_y is not None
        np.savez_compressed(
            output / f"D{horizon}.predictions.npz",
            y=row_y,
            state=row_state,
            action=row_action,
            world=data["world"][row_state],
            fold=fold[row_state],
            **{f"oof_{name}": values for name, values in predictions.items()},
        )
        summary = bootstrap_auc(row_y, predictions, data["world"][row_state], n_boot)
        summary["label_counts"] = {
            "legal_branches": int(len(row_y)),
            "death_within_horizon": int(row_y.sum()),
            "survives_beyond_horizon": int((1 - row_y).sum()),
        }
        report["results"][f"D{horizon}"] = summary
        (output / "horizon_fits.json").write_text(json.dumps(report, indent=2) + "\n")
    after = _source_hashes(sources)
    if before != after:
        raise RuntimeError("a bound source artifact or implementation file changed during the fit")
    report["source_sha256_after"] = after
    report["elapsed_s"] = time.monotonic() - started
    report["artifact_sha256"] = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "horizon_fits.json"
    }
    report["status"] = "PASS"
    (output / "horizon_fits.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def named_fuse_paths(paths: list[Path]) -> dict[str, Path]:
    """Name selected T5b receipts from their B8 final parent directory."""
    selected = {f"fuse_q_{path.parent.name}": path for path in paths}
    if len(selected) != len(paths):
        raise ValueError("duplicate T5b fuse final names")
    return selected


def main() -> None:
    """Run the single-process CPU-only horizon diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--no-arrival", action="store_true")
    parser.add_argument(
        "--fuse",
        action="append",
        type=Path,
        default=None,
        help="optional T5b feature receipt; repeat to include selected B8 finals",
    )
    args = parser.parse_args()
    selected = None
    if args.fuse is not None:
        selected = named_fuse_paths(args.fuse)
    with threadpool_limits(limits=1):
        result = run(
            args.out,
            arrival_path=None if args.no_arrival else T8_ARRIVAL,
            fuse_paths=selected,
            n_boot=args.boot,
        )
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
