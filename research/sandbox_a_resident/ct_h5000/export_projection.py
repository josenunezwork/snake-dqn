"""Prospective mature-body export sizing; projections are not measured compression ratios."""

from __future__ import annotations

from typing import Any


def project_stage_bytes() -> dict[str, Any]:
    """Conservatively add full root and raw six-snake body costs to predecessor stage."""
    predecessor_bytes = 98_500_000
    roots = 2560
    root_bytes = 28 * 1024
    # Int16 x/y snapshots use four raw bytes per segment. Count all six snakes
    # at the stated length, without credit for delta/byte-shuffle compression.
    projections = {
        str(length): predecessor_bytes + roots * (root_bytes + 6 * length * 4)
        for length in (600, 1700, 10400)
    }
    return {
        "predecessor_stage_bytes_approx": predecessor_bytes,
        "predecessor_root_bytes_approx": root_bytes,
        "full_bank_roots": roots,
        "formula": "98.5e6 + 2560 * (28*1024 + 6*length*4)",
        "projected_stage_bytes_by_all_six_snake_length": projections,
        "stage_cap_bytes": 896 * 1024**2,
        "recursive_cap_bytes": 1024**3,
        "transfer_cap_bytes": 1536 * 1024**2,
        "diagnostic_bound_bytes": 96 * 1024**2,
        "notes": "Deliberately double-counts predecessor roots/body bytes as headroom; "
        "anchor/topology growth and serialization remain unmeasured. "
        "Existing memory/store caps still apply; no worst-case capacity guarantee.",
    }
