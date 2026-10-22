"""Exact pre-lease receipt compatibility; never rewrite measured evidence.

The process-gate merge added a Mac CLI lease to ruler.py after Y1-Mac.
Neither its computational functions nor the native paired runner changed.
The native validator now calls this compatibility boundary, so its own former
hash is also admitted explicitly. All other dependencies must match exactly.
"""

from __future__ import annotations

from typing import Mapping

OLD_RULER = "d224d94b840a2a91051ec96eea4fbec480fbd55200d3e52c97ced69809b0d0b4"
OLD_NATIVE = "b7c730ba6b63f3000b941cf4f18e3d29fca19bee4cc6304300bcad2b1062e65b"
RULER_PATH = "research/bl_served_search/ruler.py"
NATIVE_PATH = "research/bl_served_search/native_ruler.py"
SELF_PATH = "research/bl_served_search/native_provenance.py"


def compatible(saved: Mapping[str, str], current: Mapping[str, str]) -> bool:
    """Allow only the two named historical bytes and the new helper omission."""
    if set(saved) - set(current):
        return False
    missing = set(current) - set(saved)
    if missing and (missing != {SELF_PATH} or saved.get(NATIVE_PATH) != OLD_NATIVE):
        return False
    legacy = {RULER_PATH: OLD_RULER, NATIVE_PATH: OLD_NATIVE}
    return all(
        value == current[key] or (key in legacy and value == legacy[key])
        for key, value in saved.items()
    )


def same_sources(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    """Compare both inventories to current bytes, retaining originals in reports."""
    from research.bl_served_search.native_ruler import source_identity

    current = source_identity()
    return compatible(left, current) and compatible(right, current)
