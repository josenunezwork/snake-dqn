"""Fail-closed research/product boundary and paired continuation audit hook."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import t

from .config import RoundConfig


def teacher_audit(
    cfg: RoundConfig, report: dict[str, Any], student_sha: str, opponents_sha: str
) -> dict[str, Any]:
    """Verify parent-paired teacher gain under student vs B8 continuation.

    The independent report must use the same roots, first-action search design,
    and independent scoring suffixes in both arms. Parent means are the units;
    the one-sided 90% lower bound must exceed the frozen degradation tolerance.
    This hook admits a prospective audit; it does not generate an audit or GO.
    """
    if (
        report.get("schema") != "spatial-exit-teacher-audit/v1"
        or not isinstance(report.get("independent_judge"), str)
        or not report["independent_judge"].strip()
        or report.get("namespace") != cfg.namespace
        or report.get("student_sha256") != student_sha
        or report.get("opponents_sha256") != opponents_sha
        or report.get("branch_horizon") != cfg.branch_horizon
        or report.get("selection_suffixes") != list(cfg.selection_suffixes)
        or report.get("scoring_suffixes") != list(cfg.scoring_suffixes)
    ):
        raise ValueError("paired teacher-degradation audit identity mismatch")
    rows = report.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("paired teacher audit rows required")
    seen, parents = set(), defaultdict(list)
    for row in rows:
        if (
            not isinstance(row.get("uid"), str)
            or not row["uid"]
            or row["uid"] in seen
            or not isinstance(row.get("parent_id"), str)
            or not row["parent_id"]
        ):
            raise ValueError("duplicate/missing paired teacher audit identity")
        seen.add(row["uid"])
        scores = [row.get("student_teacher_gain"), row.get("b8_teacher_gain")]
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in scores
        ):
            raise ValueError("nonfinite paired teacher audit")
        parents[row["parent_id"]].append(scores[0] - scores[1])
    if len(parents) < max(2, cfg.teacher_audit_min_parents):
        raise ValueError("paired teacher audit lacks independent parent support")
    values = np.array([np.mean(v) for v in parents.values()])
    lower = float(
        values.mean() - t.ppf(0.90, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    )
    if lower < -cfg.teacher_max_degradation:
        raise ValueError("paired teacher-degradation audit failed")
    return dict(parent_count=len(values), lower90=lower, passed=True)


def require_training_scope(
    cfg: RoundConfig,
    rows: Sequence[dict[str, Any]],
    scope: str,
    replay_receipt: Path | None = None,
) -> None:
    """Research labels stay quarantined until an exact-product replay is bound.

    Product training requires an independently admitted replay receipt replacing
    each original search target. This package has no exact-product replay tool,
    so it deliberately refuses that path even when quarantine is disabled.
    """
    if scope not in {"research", "product"}:
        raise ValueError("unknown training scope")
    if scope == "product":
        if cfg.quarantine_search_targets or any(r.get("quarantined", True) for r in rows):
            raise ValueError("search targets quarantined pending exact product dynamics replay")
        raise ValueError(
            "exact product replay admission is not implemented; product training refused"
        )
