"""Explicit compatibility for the prospective partition-only source change."""

from __future__ import annotations

from typing import Any

LEGACY_CHUNKS_SHA256 = "977ce4df49e534f1588b6d516bf546d42e3b5e062b71ff10317da65bdf3cb39e"
FROZEN_BANK_SHA256 = "0a8d3059e1feb7f9c954171093e0c21258b6cf89ea43e1c20f8e035776fcb8d3"
CHUNKS_PATH = "research/bl_served_search/chunks.py"


def compatible_source(saved: dict[str, Any], current: dict[str, str]) -> bool:
    """Accept only the registered old partition hash; every other byte agrees."""
    if saved == current:
        return True
    return (
        set(saved) == set(current)
        and saved.get(CHUNKS_PATH) == LEGACY_CHUNKS_SHA256
        and all(
            saved[name] == checksum for name, checksum in current.items() if name != CHUNKS_PATH
        )
    )
