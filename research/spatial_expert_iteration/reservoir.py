"""Pre-data reservoir admission and exact pre-search export sizing."""

from __future__ import annotations

import io
from typing import Any, Sequence

import torch

from .config import RoundConfig
from .data import select_anchors


def serialized_size(value: Any) -> int:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.tell()


def capacity_forecast(cfg: RoundConfig, roots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reserve all candidate roots before collection; never evict old parents.

    Bytes are a forecast, not a bound on future natural body lengths: reserve
    the larger of the largest existing serialized root and a maturity-sized
    six-snake row (snapshot + compact bodies, food, 64 KiB metadata allowance).
    Exact serialized union sizing runs after collection, before search/fit.
    """
    select_anchors(cfg, roots)
    maximum = len(roots) + cfg.candidate_roots
    if maximum > cfg.reservoir_max_roots:
        raise ValueError("pre-data natural reservoir root cap exceeded")
    sim = cfg.sim_config(1)
    maturity_bytes = (
        8 * sim.num_snakes * min(cfg.body_capacity, cfg.mature_length)
        + 12 * ((sim.game_width // sim.segment_size) * (sim.game_height // sim.segment_size))
        + 65536
    )
    per_root = max([maturity_bytes, *(serialized_size(r) for r in roots)])
    current = serialized_size(dict(schema="spatial-exit-reservoir/v1", roots=list(roots)))
    forecast = current + cfg.candidate_roots * per_root
    if forecast > cfg.reservoir_max_bytes:
        raise ValueError("pre-data natural reservoir byte forecast exceeds cap")
    return dict(
        policy="append-only-no-eviction",
        existing_roots=len(roots),
        maximum_roots=maximum,
        current_bytes=current,
        forecast_bytes_per_new_root=per_root,
        forecast_bytes=forecast,
        forecast_method="max-existing-serialized-root-or-maturity-row-plus-64KiB/v1",
        exact_check="after-collection-before-search-and-at-export",
    )


def reservoir_value(
    cfg: RoundConfig, previous: Sequence[dict[str, Any]], new: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Preserve old root bytes and reject exact overflow before expensive search."""
    union = {r["uid"]: r for r in previous}
    for row in new:
        union.setdefault(row["uid"], row)
    if len(union) > cfg.reservoir_max_roots:
        raise ValueError("frozen natural reservoir root cap exceeded")
    value = dict(
        schema="spatial-exit-reservoir/v1",
        namespace=cfg.namespace,
        round_index=cfg.round_index,
        roots=list(union.values()),
    )
    if serialized_size(value) > cfg.reservoir_max_bytes:
        raise ValueError("frozen natural reservoir byte cap exceeded before search")
    return value
