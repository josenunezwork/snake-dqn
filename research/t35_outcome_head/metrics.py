"""Registered metrics: pooled AUC, outcome-argmax agreement, world bootstrap, verdict."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

GO_AUC = 0.80
GO_AUC_LB = 0.77
GO_AGREE = 0.85
GO_MARGIN = 0.03
KILL_AUC = 0.77
KILL_AGREE = 0.70


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank statistic with average ranks for ties."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def within_row_auc(p: np.ndarray, y: np.ndarray, tested: np.ndarray) -> float:
    """Mean per-row AUC over rows whose tested actions have mixed outcomes."""
    vals = []
    for i in range(p.shape[0]):
        t = tested[i]
        yi = y[i, t]
        if 0 < yi.sum() < len(yi):
            vals.append(auc(p[i, t], yi))
    return float(np.mean(vals)) if vals else float("nan")


def outcome_argmax(p_surv: np.ndarray, q: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Mirror the teacher's rule on predictions.

    Predicted survivors S = {legal a : p_surv[a] >= 0.5}; choose argmax Q_B8 over S
    (lowest index on ties). If S is empty, choose argmax p_surv over legal actions.
    """
    out = np.empty(p_surv.shape[0], dtype=np.int64)
    for i in range(p_surv.shape[0]):
        lg = legal[i] if legal[i].any() else np.ones_like(legal[i])
        surv = lg & (p_surv[i] >= 0.5)
        if surv.any():
            out[i] = int(np.argmax(np.where(surv, q[i], -np.inf)))
        else:
            out[i] = int(np.argmax(np.where(lg, p_surv[i], -np.inf)))
    return out


def world_bootstrap(
    world_index: np.ndarray,
    stat: Callable[[np.ndarray], float],
    reps: int = 2000,
    seed: int = 20261020,
) -> tuple[float, float]:
    """Percentile 95% CI of ``stat(row_indices)`` resampling worlds with replacement."""
    worlds = np.unique(world_index)
    rows_by_world = [np.flatnonzero(world_index == w) for w in worlds]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(worlds), size=len(worlds))
        idx = np.concatenate([rows_by_world[k] for k in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def verdict(auc_teacher: float, auc_lb: float, agree: float, margin: float) -> str:
    """Registered decision rule (PREREGISTRATION.md section 6)."""
    if not (np.isfinite(auc_teacher) and np.isfinite(agree)):
        return "ABORT"
    if auc_teacher <= KILL_AUC or agree < KILL_AGREE:
        return "KILL"
    if auc_teacher >= GO_AUC and auc_lb > GO_AUC_LB and agree >= GO_AGREE and margin >= GO_MARGIN:
        return "GO"
    return "INCONCLUSIVE"


def pairs(values: np.ndarray, rows: Sequence[int], tested: np.ndarray) -> np.ndarray:
    """Flatten the tested (row, action) entries of ``values`` for ``rows``."""
    rows = np.asarray(rows)
    return values[rows][tested[rows]]
