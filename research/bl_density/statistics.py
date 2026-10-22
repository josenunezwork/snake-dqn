"""Pure development-measurement summaries for the BL density screen.

These summaries deliberately make no inferential or scientific claim.  They
describe the observed rows, use episode-grouped held-out classification only
as a two-sample diagnostic, and fail closed when the diagnostic is too small
or malformed to be useful.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

_MIN_EPISODES_PER_CLASS = 4


def length_summary(lengths: Iterable[float]) -> Dict[str, Any]:
    """Return a descriptive long-body length distribution.

    Non-finite values make the summary explicitly insufficient rather than
    silently changing the denominator.
    """
    values = np.asarray(list(lengths), dtype=float)
    if values.size == 0:
        return {"status": "INSUFFICIENT", "reason": "empty_lengths", "count": 0}
    if not np.isfinite(values).all():
        return {"status": "INSUFFICIENT", "reason": "nonfinite_lengths", "count": int(values.size)}

    names = ("lt_800", "800_999", "1000_1199", "1200_1499", "ge_1500")
    masks = (
        values < 800,
        (values >= 800) & (values < 1000),
        (values >= 1000) & (values < 1200),
        (values >= 1200) & (values < 1500),
        values >= 1500,
    )
    quantiles = {
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }
    return {
        "status": "OK",
        "count": int(values.size),
        "quantiles": quantiles,
        "histogram": {name: int(mask.sum()) for name, mask in zip(names, masks)},
    }


def _ks_statistic(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the two-sample empirical-CDF maximum distance (no p-value)."""
    combined = np.sort(np.concatenate((reference, candidate)))
    # Evaluating both right-continuous ECDFs at every observed support point is
    # exact, including ties.
    reference_cdf = np.searchsorted(np.sort(reference), combined, side="right") / reference.size
    candidate_cdf = np.searchsorted(np.sort(candidate), combined, side="right") / candidate.size
    return float(np.max(np.abs(reference_cdf - candidate_cdf)))


def ks_compare(
    reference_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    names: Sequence[str],
) -> Dict[str, Any]:
    """Return KS-only diagnostics for fields deliberately excluded from AUC.

    Geometry is a separate realism signal.  In particular, it must never be
    appended to the frozen 35-feature classifier after data collection has
    started.  Like :func:`compare`, malformed or missing values fail closed.
    """
    fields = list(names)
    if not fields or len(set(fields)) != len(fields):
        return {"status": "INSUFFICIENT", "reason": "invalid_geometry_names"}

    def matrix(rows: Sequence[Dict[str, Any]]) -> tuple[np.ndarray, str]:
        values: list[list[float]] = []
        for row in rows:
            try:
                vector = [float(row[name]) for name in fields]
            except (KeyError, TypeError, ValueError):
                return np.empty((0, len(fields))), "missing_or_nonnumeric_geometry"
            if not np.isfinite(vector).all():
                return np.empty((0, len(fields))), "nonfinite_geometry"
            values.append(vector)
        if not values:
            return np.empty((0, len(fields))), "empty_rows"
        return np.asarray(values, dtype=float), ""

    reference, reference_error = matrix(reference_rows)
    candidate, candidate_error = matrix(candidate_rows)
    if reference_error or candidate_error:
        return {
            "status": "INSUFFICIENT",
            "reason": reference_error or candidate_error,
            "n_reference_rows": int(reference.shape[0]),
            "n_candidate_rows": int(candidate.shape[0]),
        }
    return {
        "status": "OK",
        "fields": {
            name: {
                "statistic": _ks_statistic(reference[:, index], candidate[:, index]),
                "n_reference": int(reference.shape[0]),
                "n_candidate": int(candidate.shape[0]),
            }
            for index, name in enumerate(fields)
        },
    }


def _rows(rows: Sequence[Dict[str, Any]], width: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Validate and convert public feature-row shape without dropping rows."""
    features: List[List[float]] = []
    episode_ids: List[str] = []
    for row in rows:
        if not isinstance(row, dict) or "episode_id" not in row or "features" not in row:
            return np.empty((0, width)), np.empty(0, dtype=str), "malformed_row"
        values = row["features"]
        if not isinstance(values, (list, tuple)) or len(values) != width:
            return np.empty((0, width)), np.empty(0, dtype=str), "feature_width"
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError):
            return np.empty((0, width)), np.empty(0, dtype=str), "non_numeric_feature"
        if not np.isfinite(vector).all():
            return np.empty((0, width)), np.empty(0, dtype=str), "nonfinite_feature"
        features.append(vector)
        episode_ids.append(str(row["episode_id"]))
    if not features:
        return np.empty((0, width)), np.empty(0, dtype=str), "empty_rows"
    return np.asarray(features, dtype=float), np.asarray(episode_ids, dtype=str), ""


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC with average ranks for ties, without scipy."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = y == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def _classifier(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_ids: np.ndarray,
    candidate_ids: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    ref_episodes = sorted(set(reference_ids.tolist()))
    candidate_episodes = sorted(set(candidate_ids.tolist()))
    result: Dict[str, Any] = {
        "n_reference_episodes": len(ref_episodes),
        "n_candidate_episodes": len(candidate_episodes),
        "n_reference_rows": int(reference.shape[0]),
        "n_candidate_rows": int(candidate.shape[0]),
    }
    if (
        len(ref_episodes) < _MIN_EPISODES_PER_CLASS
        or len(candidate_episodes) < _MIN_EPISODES_PER_CLASS
    ):
        return {"status": "INSUFFICIENT", "reason": "fewer_than_4_episodes_per_class", **result}
    overlap = sorted(set(ref_episodes) & set(candidate_episodes))
    if overlap:
        return {
            "status": "INSUFFICIENT",
            "reason": "episode_identity_in_both_classes",
            "overlapping_episode_ids": overlap,
            **result,
        }

    # Fixed by METRICS_AMENDMENT_01: lexicographically sort each class, hold
    # out its final ceil(25%) episode IDs, and retain every row from a group.
    # Thus repeated/nearby decisions can never leak across train and test.
    n_ref_test = max(1, int(ceil(len(ref_episodes) * 0.25)))
    n_candidate_test = max(1, int(ceil(len(candidate_episodes) * 0.25)))
    ref_test = set(ref_episodes[-n_ref_test:])
    candidate_test = set(candidate_episodes[-n_candidate_test:])
    ref_is_test = np.asarray([episode in ref_test for episode in reference_ids])
    candidate_is_test = np.asarray([episode in candidate_test for episode in candidate_ids])
    train_x = np.vstack((reference[~ref_is_test], candidate[~candidate_is_test]))
    test_x = np.vstack((reference[ref_is_test], candidate[candidate_is_test]))
    train_y = np.concatenate((np.zeros((~ref_is_test).sum()), np.ones((~candidate_is_test).sum())))
    test_y = np.concatenate((np.zeros(ref_is_test.sum()), np.ones(candidate_is_test.sum())))
    if not train_x.size or not test_x.size or len(np.unique(test_y)) != 2:
        return {"status": "INSUFFICIENT", "reason": "invalid_group_holdout", **result}

    train_constant = np.ptp(train_x, axis=0) == 0
    dropped = [name for name, constant in zip(feature_names, train_constant) if constant]
    kept = [name for name, constant in zip(feature_names, train_constant) if not constant]
    if not kept:
        return {
            "status": "INSUFFICIENT",
            "reason": "all_training_features_constant",
            "dropped_constant_feature_names": dropped,
            **result,
        }
    train_x = train_x[:, ~train_constant]
    test_x = test_x[:, ~train_constant]
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=0, max_iter=1000, class_weight="balanced"),
        )
        model.fit(train_x, train_y)
        scores = model.predict_proba(test_x)[:, 1]
        method = "sklearn_logistic_regression"
    except ImportError:
        return {"status": "INSUFFICIENT", "reason": "sklearn_unavailable", **result}
    except ValueError as exc:
        return {"status": "INSUFFICIENT", "reason": str(exc), **result}
    result.update(
        {
            "status": "OK",
            "auc": _auc(test_y, np.asarray(scores, dtype=float)),
            "method": method,
            "dropped_constant_feature_names": dropped,
            "used_feature_names": kept,
            "split": {
                "reference_test_episode_ids": sorted(ref_test),
                "candidate_test_episode_ids": sorted(candidate_test),
                "reference_train_episode_ids": sorted(set(ref_episodes) - ref_test),
                "candidate_train_episode_ids": sorted(set(candidate_episodes) - candidate_test),
                "n_reference_train_rows": int((~ref_is_test).sum()),
                "n_candidate_train_rows": int((~candidate_is_test).sum()),
                "n_reference_test_rows": int(ref_is_test.sum()),
                "n_candidate_test_rows": int(candidate_is_test.sum()),
            },
        }
    )
    return result


def compare(
    reference_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Describe feature distances and a grouped two-sample AUC diagnostic.

    ``reference_rows`` and ``candidate_rows`` contain ``episode_id`` and a
    feature vector matching ``feature_names``.  This function never discards
    malformed rows: it reports ``INSUFFICIENT`` so upstream collection faults
    remain visible.
    """
    names = list(feature_names)
    if not names or len(set(names)) != len(names):
        return {"status": "INSUFFICIENT", "reason": "invalid_feature_names"}
    reference, reference_ids, reference_error = _rows(reference_rows, len(names))
    candidate, candidate_ids, candidate_error = _rows(candidate_rows, len(names))
    if reference_error or candidate_error:
        return {
            "status": "INSUFFICIENT",
            "reason": reference_error or candidate_error,
            "n_reference_rows": int(reference.shape[0]),
            "n_candidate_rows": int(candidate.shape[0]),
        }
    feature_ks = {
        name: {
            "statistic": _ks_statistic(reference[:, index], candidate[:, index]),
            "n_reference": int(reference.shape[0]),
            "n_candidate": int(candidate.shape[0]),
        }
        for index, name in enumerate(names)
    }
    classifier = _classifier(reference, candidate, reference_ids, candidate_ids, names)
    return {
        "status": "OK" if classifier["status"] == "OK" else "INSUFFICIENT",
        "feature_ks": feature_ks,
        "classifier": classifier,
        "n_reference_rows": int(reference.shape[0]),
        "n_candidate_rows": int(candidate.shape[0]),
        "n_reference_episodes": len(set(reference_ids.tolist())),
        "n_candidate_episodes": len(set(candidate_ids.tolist())),
    }
