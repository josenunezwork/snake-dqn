"""Immutable L2' four-chunk assignment and validation.

The selection bank remains one collision-audited allocation.  Chunks are a
transport boundary only: each receives a deterministic, interleaved subset so
that a completed receipt always contains 16 frozen, 16 mixed and 8 scripted
worlds.  They must be merged only after all four terminal receipts exist.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

CHUNK_COUNT = 4
CHUNK_SIZE = 40
CHUNK_MIX_COUNTS = {"frozen": 16, "mixed": 16, "scripted": 8}
_MIX_ORDER = ("frozen", "mixed", "scripted")


def ordered_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the fixed interleaved allocation order from a complete bank."""
    by_mix = {
        mix: sorted(
            (row for row in rows if row["mix"] == mix),
            key=lambda row: (int(row["world_seed"]), str(row["world_id"])),
        )
        for mix in _MIX_ORDER
    }
    expected = {mix: CHUNK_MIX_COUNTS[mix] * CHUNK_COUNT for mix in _MIX_ORDER}
    if {mix: len(values) for mix, values in by_mix.items()} != expected:
        raise ValueError("L2' chunking requires the complete 64/64/32 allocation")
    result: list[dict[str, Any]] = []
    for chunk_index in range(CHUNK_COUNT):
        # Eight full cycles make all three mixes visible early in every chunk;
        # the remaining frozen/mixed rows retain the same pairwise cadence.
        for offset in range(CHUNK_MIX_COUNTS["frozen"]):
            result.append(by_mix["frozen"][chunk_index * 16 + offset])
            result.append(by_mix["mixed"][chunk_index * 16 + offset])
            if offset < CHUNK_MIX_COUNTS["scripted"]:
                result.append(by_mix["scripted"][chunk_index * 8 + offset])
    return result


def chunk_rows(rows: Iterable[dict[str, Any]], chunk_index: int) -> list[dict[str, Any]]:
    """Select one exact 40-world chunk, preserving its deterministic order."""
    if not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("invalid L2' chunk index")
    ordered = ordered_rows(rows)
    chunk = ordered[chunk_index * CHUNK_SIZE : (chunk_index + 1) * CHUNK_SIZE]
    validate_chunk_rows(chunk)
    return chunk


def world_ids_sha256(rows: Iterable[dict[str, Any]]) -> str:
    """Hash ordered world identities without importing evaluator implementation."""
    payload = "\n".join(str(row["world_id"]) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_chunk_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Require exactly the registered chunk composition and unique identities."""
    values = list(rows)
    if (
        len(values) != CHUNK_SIZE
        or Counter(str(row.get("mix")) for row in values) != CHUNK_MIX_COUNTS
    ):
        raise ValueError("wrong L2' chunk composition")
    ids = [str(row.get("world_id")) for row in values]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate L2' chunk world identity")
