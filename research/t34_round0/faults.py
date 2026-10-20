"""Typed terminal conditions for the T34 Round 0 producer.

These classes make the scientific controller tripwires distinguishable from
operational failures after spawned worker exceptions cross process boundaries.
"""

from __future__ import annotations


class NonFiniteQLossFault(ValueError):
    """The controller produced a non-finite Q value or loss."""


class IllegalActionFault(ValueError):
    """The controller produced an action outside the six-action contract."""


class OneActionShareFault(RuntimeError):
    """One action exceeded the registered 90 percent share in a 100k window."""


class OperationalFailure(RuntimeError):
    """A non-scientific producer failure with no controller conclusion."""


class ExportSerializationError(OperationalFailure):
    """A trace or its create-only export could not be serialized or loaded."""


class CapExpiry(OperationalFailure):
    """The registered per-chunk operational wall cap was reached."""


# Kept as an import-compatible spelling for the first implementation draft.
CapExpired = CapExpiry
