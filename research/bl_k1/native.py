"""Thread-local branch opt-in; S2's counter-exact mode remains unchanged.

Kernel-only skips live_vetoes bookkeeping, never runtime memory or last_veto.
Only rollout_all enters native_scope; roots/shadows remain on the stock hook.
The adapter lives here to respect WP-1/S2 file ownership.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from research.t34_speed_s2 import native_v8 as s2

MODES = ("stock", "replay", "kernel")
_LOCAL = threading.local()
_LOCK = threading.Lock()
_INSTALLED: Any = None


@dataclass
class RowStats(s2.NativeV8Stats):
    """Native/fallback coverage; rows, never inferred from an enable flag."""

    kernel_rows: int = 0
    stock_rows: int = 0
    noncandidate_rows: int = 0
    root_stock_rows: int = 0


@dataclass
class Settings:
    mode: str = "stock"
    stats: RowStats | None = None


_SETTINGS: contextvars.ContextVar[Settings | None] = contextvars.ContextVar("k1", default=None)


@contextlib.contextmanager
def configure(mode: str, stats: RowStats | None = None) -> Iterator[None]:
    """Opt in a caller; only frozen/mixed rollout_all branches consume this."""
    if mode not in MODES:
        raise ValueError(f"unknown K-1 mode: {mode}")
    token = _SETTINGS.set(Settings(mode, stats))
    try:
        yield
    finally:
        _SETTINGS.reset(token)


def _kernel_hook(
    policy: Any,
    sim: Any,
    rows: np.ndarray,
    pick: np.ndarray,
    counts: np.ndarray,
    cap: np.ndarray,
    need: np.ndarray,
    spacious: np.ndarray,
    q_rows: np.ndarray,
    mask_rows: np.ndarray,
    actions: np.ndarray,
    *,
    stats: RowStats | None = None,
    audit: bool = False,
) -> np.ndarray:
    """Call v8_decide directly, retaining the stock configuration/input guards."""
    from src.simd_env.v8_nb import v8_decide
    from src.simd_env.vector61_policy import check_live_rule_config

    if audit:
        raise ValueError("kernel identity uses independent stock trajectories, not counter replay")
    if getattr(policy, "_rule_config_checked", None) is not sim:
        check_live_rule_config(sim)
        policy._rule_config_checked = sim
    # Existing hooks may carry a non-default config: never silently ignore it.
    for hook in policy.live_vetoes.values():
        if not (
            getattr(hook, "head_avoidance", False)
            and getattr(hook, "_reference", None) is None
            and getattr(hook, "reference_lambda", None) is None
            and float(hook.lam) == float(policy.veto_spec.lam)
        ):
            raise s2.NativeV8Mismatch("kernel live hook configuration differs")
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    base = actions[pick].copy()
    res = v8_decide(
        sim,
        rows,
        np.asarray(q_rows, dtype=np.float32),
        np.asarray(mask_rows, dtype=bool),
        float(policy.veto_spec.lam),
        grid_shape=(int(policy.runtime.featurizer.gw), int(policy.runtime.featurizer.gh)),
    )
    for name, stock in (("base", base), ("counts", counts), ("cap", cap), ("need", need)):
        if not np.array_equal(getattr(res, name), stock):
            raise s2.NativeV8Mismatch(f"kernel {name} differs from stock inputs")
    final = res.action.copy()
    policy.last_veto = {
        "rows": rows.copy(),
        "frame": sim.frame[rows[:, 0]].copy(),
        "counts": counts,
        "need": need,
        "spacious": spacious,
        "base": base,
        "final": final,
    }
    if stats is not None:
        stats.calls += 1
        stats.rows += len(rows)
        stats.kernel_rows += len(rows)
        stats.overrides += int((final != base).sum())
        stats.long_rows += int((sim.length[rows[:, 0], rows[:, 1]] >= 800).sum())
    out = actions.copy()
    out[pick] = final
    return out


def _install() -> None:
    """Install an inert thread-local adapter without changing S2 source bytes."""
    global _INSTALLED
    with _LOCK:
        from src.simd_env.vector61_policy import Vector61SimdPolicy

        if _INSTALLED is Vector61SimdPolicy._apply_veto_live_hook:
            return
        s2.enable()
        original = Vector61SimdPolicy._apply_veto_live_hook

        def routed(policy: Any, sim: Any, rows: np.ndarray, *args: Any) -> np.ndarray:
            settings = _SETTINGS.get()
            active = getattr(_LOCAL, "active", False)
            stats = settings.stats if settings is not None else None
            if not active and stats is not None and s2._depth() == 0:
                stats.root_stock_rows += len(rows)
            if active and settings.mode == "stock":
                if stats is not None:
                    stats.stock_rows += len(rows)
            elif active and not s2.is_native_v8_candidate(policy):
                if stats is not None:
                    stats.noncandidate_rows += len(rows)
            elif active and getattr(_LOCAL, "kernel", False) and s2._depth() > 0:
                return _kernel_hook(policy, sim, rows, *args, stats=stats)
            return original(policy, sim, rows, *args)

        Vector61SimdPolicy._apply_veto_live_hook = routed
        _INSTALLED = routed


@contextlib.contextmanager
def branch_scope(mix: str) -> Iterator[None]:
    """Enter S2 only for frozen/mixed branch rows, never parent/root dispatch."""
    settings = _SETTINGS.get()
    if settings is None or mix not in ("frozen", "mixed"):
        with s2.python_scope():
            yield
        return
    _install()
    previous = (getattr(_LOCAL, "kernel", False), getattr(_LOCAL, "active", False))
    _LOCAL.kernel, _LOCAL.active = settings.mode == "kernel", True
    try:
        scope = s2.python_scope() if settings.mode == "stock" else s2.native_scope(settings.stats)
        with scope:
            yield
    finally:
        _LOCAL.kernel, _LOCAL.active = previous
