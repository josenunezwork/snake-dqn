"""Frozen allocation, delivery identity and terminal-look rules for package P."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tools.sandbox.registry import world_seed

STUDY = "t34-watch-lead-1028"
NAMESPACE = f"a100-dev/{STUDY}/"
ARMS = ("bare", "P0", "P48")
SEEDS = tuple(world_seed(STUDY, index) for index in range(32))
FIXTURE_SEED = world_seed(STUDY, 32)
FRAMES = 6000
DEPTH = 300
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 3431
MAX_ROOT_BYTES = 16 * 1024 * 1024
DECISION_ORDER = (
    "INVALID/INCOMPLETE first; then STOP this delayed controller if UB90(benefit)<10% "
    "or mass upper95<−5%; GO to real paced development if benefit≥10%, LB90>0 and "
    "mass lower95≥−5%; otherwise INCONCLUSIVE, with no automatic expansion."
)


@dataclass(frozen=True)
class Identity:
    """All acceptance fields refer to the predicted target, never the source action."""

    session: str
    frame: int
    life: int
    checkpoint: str
    state: str
    mask: tuple[bool, ...]
    transport: str = "live-session-deepcopy-rng/v1"
    fallback: str = "bare-B8-advisory-normal-fallback/v1"

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Certificate:
    """One W1/K1 result, valid at exactly one target frame."""

    source_frame: int
    target: Identity
    action: int
    ready_frame: int
    root_bytes: int


def accept(cert: Certificate, actual: Identity, *, legal: bool = True) -> str:
    """Fail closed on late, stale, foreign, divergent, overflowed or masked work."""
    if cert.root_bytes > MAX_ROOT_BYTES:
        return "overflow"
    if actual.frame > cert.target.frame or cert.ready_frame > cert.target.frame:
        return "late"
    if actual.frame < cert.target.frame:
        return "pending"
    if (
        actual.session != cert.target.session
        or actual.life != cert.target.life
        or actual.checkpoint != cert.target.checkpoint
    ):
        return "stale"
    if actual != cert.target:
        return "divergent"
    if not legal or not 0 <= cert.action < 6 or not actual.mask[cert.action]:
        return "masked"
    return "accepted"


def decision(metrics: dict[str, Any], *, valid: bool, complete: bool) -> str:
    """Apply the ruling's INVALID/STOP/GO/INCONCLUSIVE order without P0 rescue."""
    if not valid or not complete:
        return "INVALID/INCOMPLETE"
    if metrics["mass_upper95"] < -0.05 or (
        metrics.get("support", False) and metrics["benefit_ub90"] < 0.10
    ):
        return "STOP this delayed controller"
    if not metrics.get("support", False):
        return "INCONCLUSIVE"
    if (
        metrics["benefit"] >= 0.10
        and metrics["benefit_lb90"] > 0
        and metrics["mass_lower95"] >= -0.05
    ):
        return "GO to real paced development"
    return "INCONCLUSIVE"


def statistics(rows: list[dict[str, Any]], arm: str = "P48") -> dict[str, Any]:
    """Resample complete paired sessions; mass is relative paired pooled horizon mass."""
    bare = np.asarray([r["arms"]["bare"]["deaths"] for r in rows], dtype=float)
    other = np.asarray([r["arms"][arm]["deaths"] for r in rows], dtype=float)
    bm = np.asarray([r["arms"]["bare"]["mass"] for r in rows], dtype=float)
    pm = np.asarray([r["arms"][arm]["mass"] for r in rows], dtype=float)
    if not len(rows) or not np.isfinite(np.concatenate((bare, other, bm, pm))).all():
        raise ValueError("nonempty finite paired sessions required")
    if np.any(bare < 0) or np.any(other < 0) or np.any(bm <= 0) or np.any(pm < 0):
        raise ValueError("invalid deaths/mass")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(rows), size=(BOOTSTRAPS, len(rows))
    )
    denominator = bare[indices].mean(axis=1)
    # A zero-death bootstrap denominator is unsupported, not zero benefit.
    # Keep conservative extended-real bounds rather than discarding resamples.
    numer = other[indices].mean(axis=1)
    benefit = np.divide(
        denominator - numer,
        denominator,
        out=np.full(BOOTSTRAPS, np.nan),
        where=denominator > 0,
    )
    lower = np.where(denominator > 0, benefit, -np.inf)
    upper = np.where(denominator > 0, benefit, np.inf)
    paired_mass = (pm - bm) / bm
    relative_mass = paired_mass[indices].mean(axis=1)
    support = bool(bare.mean() > 0)

    def bound(values: np.ndarray, quantile: float) -> float | None:
        value = float(np.quantile(values, quantile, method="inverted_cdf"))
        return value if np.isfinite(value) else None

    lb, ub = bound(lower, 0.10), bound(upper, 0.90)
    return {
        "arm": arm,
        "sessions": len(rows),
        "support": support and lb is not None and ub is not None,
        "zero_baseline_bootstraps": int((denominator == 0).sum()),
        "benefit": float((bare.mean() - other.mean()) / bare.mean()) if support else None,
        "benefit_lb90": lb,
        "benefit_ub90": ub,
        "mass_relative": float(paired_mass.mean()),
        "mass_lower95": bound(relative_mass, 0.025),
        "mass_upper95": bound(relative_mass, 0.975),
        "bare_deaths_per_1k": float(bare.mean() / FRAMES * 1000),
        "arm_deaths_per_1k": float(other.mean() / FRAMES * 1000),
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
