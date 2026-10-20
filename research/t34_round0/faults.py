"""Compatibility re-export for fault imports bound to the Round 0 package.

The canonical shared fault classes live in :mod:`research.t34_exit_b8.faults`
so ExIt teacher code never depends on a study package.  Keep these bindings
identical for source closures that recorded the original import path.
"""

from __future__ import annotations

from research.t34_exit_b8.faults import (
    CapExpired,
    CapExpiry,
    ExportSerializationError,
    IllegalActionFault,
    NonFiniteQLossFault,
    OneActionShareFault,
    OperationalFailure,
)

__all__ = [
    "CapExpired",
    "CapExpiry",
    "ExportSerializationError",
    "IllegalActionFault",
    "NonFiniteQLossFault",
    "OneActionShareFault",
    "OperationalFailure",
]
