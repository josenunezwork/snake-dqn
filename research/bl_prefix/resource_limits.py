"""Single source of registered execution ceilings for prefix development jobs.

Resource amendments are prospective and documented in CAP_NOTE.md. Neither a
resource profile nor a prepared manifest authorizes incomplete scientific inputs.
"""

from __future__ import annotations

# Total units, reserved units, usable allocation minutes. Includes setup/export.
ORIGINAL_LIMITS = {"e6": (4.0, 0.8, 96.0), "e7": (20.0, 4.0, 480.0)}
# Prospective execution profile, prepared under CAP_NOTE.md; scientific plans
# remain blocked pending real inputs and cap ratification.
LIMITS = {"e6": (43.4, 8.68, 1041.6), "e7": (28.5, 5.7, 684.0)}
SESSION_LIMIT_MINUTES = 24 * 60


def usable_seconds(experiment: str) -> float:
    """Return the same first-cap wall ceiling used by the manifest builder."""
    cap, reserve, minutes = LIMITS[experiment]
    if minutes > SESSION_LIMIT_MINUTES or minutes > (cap - reserve) * 30 + 1e-9:
        raise ValueError("resource profile exceeds its session or usable-unit ceiling")
    return minutes * 60
