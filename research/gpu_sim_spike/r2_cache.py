"""Round-36 R2 pure-optimization layer: live windows, per-state memo, device constants.

Authority: ``JUDGMENT_POST_I0S_DIRECTION_20261024.md`` step 1 (L1+L2+L3 as one
package).  Every function here is an exact re-expression of the pinned I0
runtime (``968f0276``); nothing changes an output or a persistent byte.

* **L1, live-length windows.**  Ring gathers read ``B = min(cap,
  round_up(max seg_count + GROWTH, 256))`` logical offsets instead of all
  ``cap`` slots.  Every offset ``k >= seg_count`` was already masked out and
  painted as zero into ``amax`` reductions with a zero initial value, so the
  skipped offsets cannot change a byte.  ``GROWTH = 2`` covers the post-move
  v5 view, whose valid prefix is at most ``seg_count + 2`` long.  Ring storage
  stays ``cap`` slots, so saved state is unchanged; ``cap`` also stays in the
  advisory-mask rank encoding and in every ring modulus.
* **L2, per-state memo.**  An explicit generation counter is bumped by every
  simulator mutator (step, prepare, respawn, food maintenance, reset/restore,
  clone, repeat).  Cached values are keyed on that generation, so one advisory
  mask and one per-slot occupancy count plane serve every consumer of the same
  state.  ``debug`` mode (``enable(..., debug=True)`` or ``GPU_SIM_R2_DEBUG=1``)
  recomputes each cached value with the pinned full-capacity code on every
  lookup and raises :class:`R2CacheError` on any difference, which catches a
  missed invalidation or a consumer mutating a shared value.
* **L3, device constants.**  Small constant tables are created once per device
  instead of being copied from the host on every call.

The mode is per simulator object and is held in a weak side table, never in
``sim.__dict__``: state inventories, clone/reset copies and digests are
byte-for-byte those of the pinned engine.  A simulator that was never enabled
runs the unmodified pinned code paths.
"""

from __future__ import annotations

import itertools
import os
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

GROWTH = 2
ALIGN = 256


class R2CacheError(RuntimeError):
    """A cached R2 value differs from its pinned full-capacity recomputation."""


@dataclass
class _Entry:
    debug: bool
    generation: int
    cache: dict[str, tuple[int, Any]] = field(default_factory=dict)
    stats: dict[str, int] = field(
        default_factory=lambda: {"hits": 0, "misses": 0, "debug_checks": 0, "bumps": 0}
    )


_ENTRIES: "weakref.WeakKeyDictionary[Any, _Entry]" = weakref.WeakKeyDictionary()
_GENERATIONS = itertools.count(1)
_CONSTANTS: dict[tuple[Any, ...], torch.Tensor] = {}


def _entry(sim: Any) -> _Entry | None:
    """Registered entry, or ``None`` (including objects that cannot be weakly referenced)."""
    try:
        return _ENTRIES.get(sim)
    except TypeError:
        return None


def _env_debug() -> bool:
    return os.environ.get("GPU_SIM_R2_DEBUG", "0") == "1"


def enable(sim: Any, *, debug: bool | None = None) -> None:
    """Route ``sim`` through R2.  A fresh generation means no inherited cache."""
    _ENTRIES[sim] = _Entry(
        debug=_env_debug() if debug is None else bool(debug), generation=next(_GENERATIONS)
    )


def disable(sim: Any) -> None:
    if _entry(sim) is not None:
        del _ENTRIES[sim]


def enabled(sim: Any) -> bool:
    return _entry(sim) is not None


def debug(sim: Any) -> bool:
    entry = _entry(sim)
    return entry is not None and (entry.debug or _env_debug())


def generation(sim: Any) -> int | None:
    entry = _entry(sim)
    return None if entry is None else entry.generation


def stats(sim: Any) -> dict[str, int]:
    entry = _entry(sim)
    return {} if entry is None else dict(entry.stats)


def bump(sim: Any) -> None:
    """Invalidate every cached value of ``sim``; a no-op for pinned-mode objects."""
    entry = _entry(sim)
    if entry is None:
        return
    entry.generation = next(_GENERATIONS)
    entry.cache.clear()
    entry.stats["bumps"] += 1


def inherit(target: Any, source: Any) -> None:
    """Give a clone the source's mode, with its own fresh generation and empty cache."""
    entry = _entry(source)
    if entry is None:
        disable(target)
    else:
        enable(target, debug=entry.debug)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.device == right.device
            and bool(torch.equal(left, right))
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_same(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def check(label: str, value: Any, expected: Any) -> None:
    """Raise unless an R2 value equals its pinned recomputation exactly."""
    if not _same(value, expected):
        raise R2CacheError(f"R2 {label} differs from pinned full-capacity recomputation")


def memo(
    sim: Any,
    key: str,
    compute: Callable[[], Any],
    reference: Callable[[], Any] | None = None,
) -> Any:
    """Return ``compute()`` once per state generation; debug-check every lookup."""
    entry = _entry(sim)
    if entry is None:
        return compute()
    saved = entry.cache.get(key)
    if saved is not None and saved[0] == entry.generation:
        entry.stats["hits"] += 1
        value = saved[1]
    else:
        entry.stats["misses"] += 1
        value = compute()
        entry.cache[key] = (entry.generation, value)
    if reference is not None and (entry.debug or _env_debug()):
        entry.stats["debug_checks"] += 1
        check(key, value, reference())
    return value


def _host_scalar_safe(value: torch.Tensor) -> bool:
    return value.device.type != "meta" and type(value).__name__ != "FakeTensor"


def _fake_mode_active() -> bool:
    try:
        from torch._guards import detect_fake_mode

        return detect_fake_mode(None) is not None
    except Exception:  # pragma: no cover - private API drift fails safe (no caching)
        return True


def const(values: Any, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """L3: a cached device copy of a small constant table (never mutated by callers)."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or _fake_mode_active():
        return torch.tensor(values, dtype=dtype, device=device)
    key = (values, dtype, device.type, device.index)
    tensor = _CONSTANTS.get(key)
    if tensor is None:
        with torch.inference_mode(False):
            tensor = torch.tensor(values, dtype=dtype, device=device)
        _CONSTANTS[key] = tensor
    return tensor


def _window_value(sim: Any) -> int:
    cap = int(sim.cap)
    if not _host_scalar_safe(sim.seg_count) or sim.seg_count.numel() == 0:
        return cap
    longest = int(sim.seg_count.max().item())
    return min(cap, (longest + GROWTH + ALIGN - 1) // ALIGN * ALIGN)


def window(sim: Any) -> int:
    """L1: logical ring offsets to read; ``cap`` for pinned-mode simulators."""
    if not enabled(sim):
        return int(sim.cap)
    span = memo(sim, "window", lambda: _window_value(sim))
    if span < int(sim.cap) and _host_scalar_safe(sim.seg_count):
        # Guard against a stale generation: the window must still cover every
        # live prefix plus the post-move growth bound.
        torch._assert_async(
            sim.seg_count.max() + GROWTH <= span, "R2 live window below seg_count + growth"
        )
    return span


def ring_view(sim: Any, span: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Head-to-tail cells and live flags for the first ``span`` logical offsets."""
    k = torch.arange(span, device=sim.bodies.device)[None, None, :]
    ring = (sim.head_ptr[..., None] - k) % sim.cap
    cells = sim.bodies.gather(2, ring[..., None].expand(-1, -1, -1, 2))
    valid = sim.alive[..., None] & (k < sim.seg_count[..., None])
    return cells, valid


def _slot_planes(sim: Any, span: int) -> torch.Tensor:
    """``(E,S,H,W)`` per-slot union occupancy painted from a ``span`` ring view."""
    cells, valid = ring_view(sim, span)
    e, s = int(sim.E), int(sim.S)
    h, w = int(sim.grid_h), int(sim.grid_w)
    x, y = cells[..., 0], cells[..., 1]
    ok = valid & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    flat = torch.zeros((e * s, h * w), dtype=torch.uint8, device=sim.bodies.device)
    index = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).reshape(e * s, span)
    flat.scatter_reduce_(
        1, index, ok.reshape(e * s, span).to(torch.uint8), reduce="amax", include_self=True
    )
    return flat.reshape(e, s, h, w) != 0


def _reference_planes(sim: Any) -> torch.Tensor:
    """Pinned v8 painter applied per slot at full capacity (debug reference)."""
    from research.gpu_sim_spike.v8_device import _body_cells, _paint

    env = torch.arange(int(sim.E), device=sim.bodies.device)
    cells, valid = _body_cells(sim, env)
    planes = [
        _paint(cells[:, slot : slot + 1], valid[:, slot : slot + 1], sim.grid_h, sim.grid_w)
        for slot in range(int(sim.S))
    ]
    return torch.stack(planes, dim=1)


def slot_occupancy(sim: Any) -> torch.Tensor:
    """L2: ``(E,S,H,W)`` bool occupancy of every live body, once per state."""
    return memo(
        sim,
        "slot_occupancy",
        lambda: _slot_planes(sim, window(sim)),
        reference=lambda: _reference_planes(sim),
    )


def occupancy_counts(sim: Any) -> torch.Tensor:
    """L2: ``(E,H,W)`` uint8 number of slots covering each cell (at most S)."""

    def compute() -> torch.Tensor:
        planes = slot_occupancy(sim)
        return planes.to(torch.uint8).sum(dim=1, dtype=torch.uint8)

    return memo(sim, "occupancy_counts", compute)


def all_occupancy(sim: Any) -> torch.Tensor:
    """L2: ``(E,H,W)`` union of every live body; equals the pinned v8 ``all_occ``."""

    def reference() -> torch.Tensor:
        from research.gpu_sim_spike.v8_device import _body_cells, _paint

        env = torch.arange(int(sim.E), device=sim.bodies.device)
        return _paint(*_body_cells(sim, env), sim.grid_h, sim.grid_w)

    return memo(sim, "all_occupancy", lambda: occupancy_counts(sim) != 0, reference=reference)


def other_occupancy(sim: Any, env: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """Union of every live body except ``slot``'s own, for each ``(env, slot)`` row.

    Counts preserve duplicate occupancy: a cell stays blocked whenever any other
    slot covers it, including cells shared with the excluded snake.
    """
    own = slot_occupancy(sim)[env, slot].to(torch.uint8)
    return (occupancy_counts(sim)[env] - own) != 0
