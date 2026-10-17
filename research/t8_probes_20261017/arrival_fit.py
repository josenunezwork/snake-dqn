"""Fixed T5 GBM/LR arrival-race diagnostic and saved T1 late-LAD comparison."""

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
from threadpoolctl import threadpool_limits  # noqa: E402

from research.t5_separability_20261016.fit import bootstrap_auc  # noqa: E402
from research.t5_separability_20261016.fit import design, estimator  # noqa: E402
from research.t5_separability_20261016.t1_check import load_feature_state  # noqa: E402
from research.t5b_risk_and_fuse_20261016.fuse import T5_DATA  # noqa: E402
from research.t5b_risk_and_fuse_20261016.fuse import (  # noqa: E402
    T5_FOLDS,
    load_t5_data,
    saved_folds,
    sha256,
)
from src.simd_env.t8_features import compute_slot_features  # noqa: E402

T5_ROOT = T5_DATA.parent
ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t8-probes-20261017")
T1_STATES = T5_ROOT / "t1_feature_states.final.npz"
T1_CHECK = T5_ROOT / "t1_check.final.json"
T1_STATES_SHA = "17b62967ed2b8d00c78a0b5dbc397bce7bdf9b4eb45aeb6c05c648e7637539ce"
T1_CHECK_SHA = "46513a39c5cb5f21b12c79f8d3bb39a7fbff3ac52f0eb63fff6d248f2ff10c66"


def align_arrival(data: dict[str, np.ndarray], path: Path) -> np.ndarray:
    """Join exact state identities, rejecting missing, extra or duplicate rows."""
    with np.load(path, allow_pickle=False) as z:
        ids, world, split, features = z["state_id"], z["world"], z["split"], z["features"]
    if len(set(ids.tolist())) != len(ids) or set(ids) != set(data["state_id"]):
        raise ValueError("arrival state identities differ from the complete canonical T5 bank")
    if features.shape != (len(ids), 18) or features.dtype != np.float32:
        raise ValueError("arrival features must be float32 (N,18)")
    lookup = {str(value): i for i, value in enumerate(ids)}
    index = np.array([lookup[str(value)] for value in data["state_id"]])
    if not (
        np.array_equal(world[index], data["world"]) and np.array_equal(split[index], data["split"])
    ):
        raise ValueError("arrival world or historical split mismatch")
    if not np.isfinite(features).all():
        raise ValueError("nonfinite arrival features")
    return features[index]


def t1_check(out: Path) -> dict[str, Any]:
    """Compare the fixed T1 best survivor to its chosen action in all 46 saved LADs."""
    if sha256(T1_STATES) != T1_STATES_SHA or sha256(T1_CHECK) != T1_CHECK_SHA:
        raise ValueError("T1 saved state or identity receipt differs from published T5")
    canonical = json.loads(T1_CHECK.read_text())["rows"]
    lookup = {(r["arm"], r["env"], r["decision"]): r for r in canonical}
    with np.load(T1_STATES, allow_pickle=False) as z:
        metadata = json.loads(z["metadata"].tobytes())
    if len(metadata) != 46 or len(lookup) != 46:
        raise ValueError("expected exactly 46 saved late LADs")
    rows, seen = [], set()
    for i in range(46):
        identity, sim = load_feature_state(T1_STATES, i)
        key = (identity["arm"], identity["env"], identity["decision"])
        if key in seen or key not in lookup:
            raise ValueError("duplicate or unexpected T1 state")
        seen.add(key)
        canonical_row = lookup[key]
        for field in ("frame", "world_seed"):
            if identity[field] != canonical_row[field]:
                raise ValueError(f"T1 {field} mismatch")
        # Compare raw counts: the prescribed log transform saturates above 4096.
        features = compute_slot_features(sim, 0, 0, raw_area=True).reshape(6, 3)
        chosen, best = canonical_row["chosen_action"], canonical_row["best_surviving_action"]
        rows.append(
            {
                **identity,
                "chosen_action": chosen,
                "best_surviving_action": best,
                "chosen": features[chosen].tolist(),
                "best_surviving": features[best].tolist(),
                "survivor_larger_voronoi": bool(features[best, 0] > features[chosen, 0]),
            }
        )
    count = sum(r["survivor_larger_voronoi"] for r in rows)
    result = {
        "authority": "diagnostic only",
        "late_lads": 46,
        "larger_voronoi": count,
        "share": count / 46,
        "rows": rows,
        "source_sha256": {str(T1_STATES): T1_STATES_SHA, str(T1_CHECK): T1_CHECK_SHA},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(arrival_path: Path, out: Path, n_boot: int = 2000) -> dict[str, Any]:
    """Refit identical T5 baselines and expanded designs on exactly the saved folds."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite run: {out}")
    out.mkdir(parents=True)
    started = time.monotonic()
    paths = [
        arrival_path,
        T5_DATA,
        T5_FOLDS,
        Path(__file__),
        Path("research/t5_separability_20261016/fit.py"),
        Path("src/simd_env/t8_features.py"),
    ]
    before = {str(p): sha256(p) for p in paths}
    data = load_t5_data(T5_DATA)
    arrival = align_arrival(data, arrival_path)
    fold = saved_folds(data)
    baseline, y, state, action = design(data, "death_d300", "60+24")
    full = np.concatenate((baseline[:, :-6], arrival[state], baseline[:, -6:]), axis=1)
    report: dict[str, Any] = {
        "authority": "diagnostic only; no policy training or gate evidence",
        "source_sha256_before": before,
        "width": full.shape[1],
        "feature_design": "60+24+18 action-major race features + 6 one-hot",
        "results": {},
    }
    for model in ("hist_gradient_boosting", "logistic_regression"):
        predictions = {}
        for name, x in (("baseline", baseline), ("arrival", full)):
            pred = np.full(len(y), np.nan)
            for current in range(5):
                train, test = fold[state] != current, fold[state] == current
                fit = estimator(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    fit.fit(x[train], y[train])
                pred[test] = fit.predict_proba(x[test])[:, 1]
                print(
                    json.dumps(
                        {
                            "model": model,
                            "design": name,
                            "fold": current,
                            "elapsed_s": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if not np.isfinite(pred).all():
                raise ValueError("incomplete arrival prediction vector")
            predictions[name] = pred
        prior = T5_ROOT / "fits" / f"death_d300.{model}.predictions.npz"
        with np.load(prior, allow_pickle=False) as z:
            for key, expected in (
                ("y", y),
                ("state", state),
                ("action", action),
                ("world", data["world"][state]),
                ("fold", fold[state]),
            ):
                if not np.array_equal(z[key], expected):
                    raise ValueError(f"T5 saved predictions {key} mismatch")
            max_diff = float(np.max(np.abs(z["oof_60+24"] - predictions["baseline"])))
            if max_diff > 1e-10:
                raise ValueError(f"T5 baseline reproduction failed: {max_diff}")
            scalar_predictions = z["oof_60"].copy()
        np.savez_compressed(
            out / f"{model}.predictions.npz",
            y=y,
            state=state,
            action=action,
            world=data["world"][state],
            fold=fold[state],
            **predictions,
        )
        # T5's bootstrap has a fixed delta naming convention; alias the paired vectors.
        metrics = bootstrap_auc(
            y,
            {"60": predictions["baseline"], "60+24": predictions["arrival"]},
            data["world"][state],
            n_boot,
        )["metrics"]
        scalar_delta = bootstrap_auc(
            y,
            {"60": scalar_predictions, "60+24": predictions["arrival"]},
            data["world"][state],
            n_boot,
        )["metrics"]
        report["results"][model] = {
            "baseline_60_24": metrics["60"],
            "arrival_60_24_18": metrics["60+24"],
            "delta_vs_60_24": metrics["delta_60+24_minus_60"],
            "delta_vs_60": scalar_delta["delta_60+24_minus_60"],
            "baseline_max_abs_prediction_difference": max_diff,
            "baseline_predictions_sha256": sha256(prior),
        }
    report["t1"] = t1_check(out / "t1_check.json")
    after = {str(p): sha256(p) for p in paths}
    if before != after:
        raise RuntimeError("arrival fit source or inputs changed")
    report["source_sha256_after"] = after
    report["elapsed_s"] = time.monotonic() - started
    report["artifact_sha256"] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "arrival" / "features.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "arrival-fit")
    parser.add_argument("--t1-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.t1_only:
            result = t1_check(args.out)
            print(json.dumps({k: result[k] for k in ("late_lads", "larger_voronoi", "share")}))
        else:
            run(args.data, args.out)


if __name__ == "__main__":
    main()
