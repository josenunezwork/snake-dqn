"""Prospective ruler accounting: full slabs, mature bodies and bounded exports."""

from __future__ import annotations

import gzip
import io
import math
from typing import Any

import torch

from research.spatial_student_pilot.resume import (
    MAX_DECODED_BYTES,
    MAX_EXPORT_BYTES,
    MAX_UNIT_BYTES,
)


def late_collection_rate(batch: dict[str, Any]) -> dict[str, Any]:
    """Use only the final quarter of ticks, requiring living mature-body exposure."""
    samples = batch["tick_samples"]
    tail = samples[len(samples) * 3 // 4 :]
    qualified = len(tail) >= 20 and all(r["mature_worlds"] > 0 for r in tail)
    return {
        "qualified": qualified,
        "start_tick": tail[0]["tick"] if tail else None,
        "ticks": len(tail),
        "seconds_per_tick": sum(r["seconds"] for r in tail) / max(1, len(tail)),
        "minimum_mature_worlds": min((r["mature_worlds"] for r in tail), default=0),
    }


def support_projection(cfg: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Partial-ruler reach fractions are lower bounds; never extrapolate growth."""
    partitions = cfg.partitions()
    reached = set(batch["reached_l600_world_ids"])
    result: dict[str, Any] = {}
    for part, total, required in (("train", 768, 256), ("assessment", 256, 64)):
        ids = [i for i in batch["world_ids"] if partitions[i] == part]
        count = sum(i in reached for i in ids)
        estimate = total * count / max(1, len(ids))
        result[part] = dict(
            observed_worlds=len(ids),
            reached_l600=count,
            projected_worlds=estimate,
            required_worlds=required,
        )
    result["qualified"] = all(
        result[p]["observed_worlds"] > 0
        and result[p]["projected_worlds"] >= result[p]["required_worlds"]
        for p in ("train", "assessment")
    )
    return result


def historical_support_projection(batch: dict[str, Any]) -> dict[str, Any]:
    """Frozen historical reach fractions, with separately observed early growth.

    Restored mature worlds never enter this denominator. Neither early linear
    growth nor the historical split promises fresh pilot parent support.
    """
    samples = batch["tick_samples"]
    growth = (samples[-1]["mean_hero_length"] - samples[0]["mean_hero_length"]) if samples else 0
    result = {
        p: dict(
            observed_worlds=n,
            reached_l600=k,
            projected_worlds=total * k / n,
            required_worlds=required,
        )
        for p, n, k, total, required in (
            ("train", 820, 554, 768, 256),
            ("assessment", 204, 141, 256, 64),
        )
    }
    result.update(
        method="frozen historical reach fractions plus independent early growth diagnostic",
        historical_only=True,
        fresh_support_proven=False,
        early_ticks=len(samples),
        early_mean_length_gain=growth,
        early_seconds=sum(r["seconds"] for r in samples),
        historical_maturation_ticks=[2650, 3300],
        qualified=len(samples) >= 20
        and growth > 0
        and all(
            result[p]["projected_worlds"] >= result[p]["required_worlds"]
            for p in ("train", "assessment")
        ),
    )
    return result


def tick_forecast(
    *,
    collection_tick: float,
    b8_branch_tick: float,
    student_branch_tick: float,
    gameplay_ticks: dict[str, float],
    fit_update: float,
    student_forward_tick: float = 0.0,
    cfg: Any = None,
) -> dict[str, float]:
    """Whole production slabs, no alive-frame discounts or concurrency credit.

    Assessment charges independent selection/B8+teacher work plus the maximum
    missing student and closed-loop work, with no assumed action agreement.
    Paired gameplay has one E384 loop; its mature floor overcharges E1024's
    opponent/hero work plus an additional E192 student forward.
    """
    from research.spatial_student_pilot.config import PilotConfig

    cfg = cfg or PilotConfig()

    def slabs(roots: int, jobs: int) -> int:
        return sum(
            math.ceil(min(cfg.search_chunk, roots - start) * jobs / cfg.branch_batch_size)
            for start in range(0, roots, cfg.search_chunk)
        )

    train = cfg.train_late + cfg.train_ordinary
    assessment = cfg.assessment_late + cfg.assessment_ordinary
    return dict(
        collection=math.ceil(cfg.worlds / cfg.world_batch_size) * cfg.horizon * collection_tick,
        search=slabs(train, 12) * cfg.branch_horizon * b8_branch_tick,
        fit=(cfg.bc_updates + cfg.improvement_updates) * fit_update,
        assessment=(
            (slabs(assessment, 12) + slabs(assessment, 4)) * cfg.branch_horizon * b8_branch_tick
            + slabs(assessment, 4) * cfg.branch_horizon * student_branch_tick
        ),
        gameplay=math.ceil(cfg.gameplay_worlds / cfg.world_batch_size)
        * cfg.horizon
        * max(
            gameplay_ticks["b8"], gameplay_ticks["student"], collection_tick + student_forward_tick
        ),
    )


def _sizes(payload: Any) -> tuple[int, int]:
    raw = io.BytesIO()
    torch.save(payload, raw)
    content = raw.getvalue()
    return len(content), len(gzip.compress(content, compresslevel=1, mtime=0))


def export_projection(
    batch: dict[str, Any],
    labels: list[dict[str, Any]],
    fit_payloads: list[Any],
    actor_bytes: int,
    *,
    assessment_rows: list[Any] | None = None,
    gameplay_rows: list[Any] | None = None,
    checkpoint_samples: list[dict[str, Any]] | None = None,
    cfg: Any = None,
) -> dict[str, Any]:
    """Conservative record-wise serialization, extrapolated to >=1000-body span.

    Per-record gzip loses cross-record compression and repeated metadata savings.
    Apply the mature span growth factor to all bytes (including constant tensors),
    then25% headroom. These estimates gate admission; runtime caps remain binding.
    """
    from research.spatial_student_pilot.config import PilotConfig

    cfg = cfg or PilotConfig()
    mature = [r for r in batch["roots"] if r["hero_length"] >= 600]
    if not mature or not labels or not fit_payloads or not assessment_rows or not gameplay_rows:
        return {"qualified": False, "reason": "missing mature or fit size samples"}
    span = max(int(r["compact"]["body_cells"].shape[2]) for r in mature)
    target = max(1000, span)
    factor = 1.25 * target / max(1, span)

    def record_bound(rows: list[Any]) -> tuple[int, int]:
        sizes = [_sizes(r) for r in rows]
        return tuple(math.ceil(max(s[i] for s in sizes) * factor) for i in (0, 1))

    # Include every selected ruler root; labels carry the same snapshot/compact.
    root = record_bound(mature + labels)
    demo = record_bound([{k: v for k, v in r.items() if k != "snapshot"} for r in mature])
    fit = tuple(max(s[i] for s in map(_sizes, fit_payloads)) for i in (0, 1))

    def measured_rows(rows: list[Any], count: int) -> dict[str, int]:
        measured = [_sizes(row) for row in rows]
        return dict(
            decoded=math.ceil(max(s[0] for s in measured) * count * 1.25) + 16384,
            gzip=math.ceil(max(s[1] for s in measured) * count * 1.25) + 16384,
        )

    units = {
        "collect": dict(
            count=8,
            decoded=4096 * demo[0] + 512 * root[0] + 16384,
            gzip=4096 * demo[1] + 512 * root[1] + 16384,
        ),
        "search": dict(
            count=math.ceil(1024 / cfg.search_chunk),
            decoded=cfg.search_chunk * root[0] + 16384,
            gzip=cfg.search_chunk * root[1] + 16384,
        ),
        "fit": dict(
            count=32,
            decoded=math.ceil(fit[0] * 1.05) + 16384,
            gzip=math.ceil(fit[1] * 1.05) + 16384,
        ),
        "aselect": dict(
            count=math.ceil(384 / cfg.search_chunk),
            decoded=cfg.search_chunk * root[0] + 16384,
            gzip=cfg.search_chunk * root[1] + 16384,
        ),
        "ascore": dict(
            count=math.ceil(384 / cfg.search_chunk),
            decoded=cfg.search_chunk * root[0] + 65536,
            gzip=cfg.search_chunk * root[1] + 65536,
        ),
        "assess": dict(
            count=math.ceil(384 / cfg.search_chunk),
            **measured_rows(assessment_rows, cfg.search_chunk),
        ),
        "gameplay": dict(count=1, **measured_rows(gameplay_rows, 192)),
        # Full-state reservoir + sim snapshots every500 ticks. Each logical
        # shard is bounded separately, and the seal is published last.
        "cstate": dict(
            count=9 * 8,
            decoded=3136 * demo[0] + 512 * root[0] + 131072,
            gzip=3136 * demo[1] + 512 * root[1] + 131072,
        ),
        "cseal": dict(count=9, decoded=16384, gzip=16384),
    }
    if checkpoint_samples is not None:
        try:
            units["cstate"] = dict(count=9 * 8, **checkpoint_bound(checkpoint_samples, factor))
        except ValueError:
            return {"qualified": False, "reason": "missing real checkpoint component samples"}
    for row in units.values():
        row["within_caps"] = row["decoded"] <= MAX_DECODED_BYTES and row["gzip"] <= MAX_UNIT_BYTES
    total = sum(r["count"] * r["gzip"] for r in units.values()) + actor_bytes + 96 * 1024**2
    return dict(
        qualified=all(r["within_caps"] for r in units.values()) and total <= MAX_EXPORT_BYTES,
        observed_mature_body_span=span,
        extrapolated_body_span=target,
        record_size_factor=factor,
        checkpoint_size_method=(
            "measured batched components"
            if checkpoint_samples is not None
            else "per-record fallback bound"
        ),
        units=units,
        projected_export_bytes=total,
        limits=dict(
            decoded_unit=MAX_DECODED_BYTES, gzip_unit=MAX_UNIT_BYTES, total_export=MAX_EXPORT_BYTES
        ),
        miscellaneous_export_reserve_bytes=96 * 1024**2,
    )


def checkpoint_components(payload: dict[str, Any]) -> dict[str, Any]:
    """Measure real batched trajectory fields, retaining counts and serializer bytes."""
    groups = {
        "demo_records": [r for key, r in payload["records"].items() if key.endswith("/demo")],
        "root_records": [r for key, r in payload["records"].items() if not key.endswith("/demo")],
        "snapshots": payload["snapshots"],
        "metadata": {
            key: value for key, value in payload.items() if key not in {"records", "snapshots"}
        },
    }
    result = {}
    for key, value in groups.items():
        decoded, compressed = _sizes(value)
        result[key] = dict(
            count=len(value) if key != "metadata" else len(payload["tick_samples"]),
            decoded=decoded,
            gzip=compressed,
        )
    return result


def checkpoint_bound(samples: list[dict[str, Any]], factor: float) -> dict[str, int]:
    """Scale measured component batches, without per-record pickle overhead.

    Count ceilings per128-world shard:3136 new demos in500 ticks, at most512
    newly referenced retained roots,128 live states,5000 metadata tick records.
    Component-wise compression loses cross-component sharing conservatively.
    """
    targets = dict(demo_records=3136, root_records=512, snapshots=128, metadata=5000)
    if not samples or any(s[k]["count"] <= 0 for s in samples for k in targets):
        raise ValueError("missing checkpoint component samples")
    return {
        mode: sum(
            math.ceil(max(s[k][mode] / s[k]["count"] for s in samples) * target * factor)
            for k, target in targets.items()
        )
        + 16384
        for mode in ("decoded", "gzip")
    }
