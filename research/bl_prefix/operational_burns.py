"""Operationally exposed IDs excluded prospectively from scientific manifests."""

from __future__ import annotations

from typing import Iterable

# See OPERATIONAL_BURN_ADDENDUM.md; preserve the original audit bytes.
BURNED_IDENTITIES = frozenset({1102882751176, 1100083338059})


def require_unburned(values: Iterable[int]) -> None:
    """Reject retired identities before any scientific continuation starts."""
    if BURNED_IDENTITIES.intersection(values):
        raise ValueError("scientific plan includes an operationally burned RNG identity")
