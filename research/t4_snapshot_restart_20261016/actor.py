"""Actor-local accounting for the opt-in T4 snapshot restart curriculum.

The trainer keeps this deliberately separate from the normal actor counters so a
run without ``--restart-frac`` has exactly the pre-T4 message shape and RNG
schedule.  Lengths are decision-time hero lengths; death lengths are the
corresponding pre-step lengths.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

RESTART_KEYS = (
    "restart_decisions",
    "fresh_decisions",
    "restart_deaths",
    "fresh_deaths",
    "restart_length_sum",
    "fresh_length_sum",
    "restart_death_length_sum",
    "fresh_death_length_sum",
)


class RestartActorTelemetry:
    """Per-message restart/fresh counters with integer, merge-safe fields."""

    def __init__(self) -> None:
        self.values: Dict[str, int] = {key: 0 for key in RESTART_KEYS}

    def record(
        self, worlds: np.ndarray, lengths: np.ndarray, died: np.ndarray, restarted: np.ndarray
    ) -> None:
        worlds = np.asarray(worlds, dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        died = np.asarray(died, dtype=bool)
        is_restart = np.asarray(restarted, dtype=bool)[worlds]
        for prefix, selected in (("restart", is_restart), ("fresh", ~is_restart)):
            self.values[f"{prefix}_decisions"] += int(selected.sum())
            self.values[f"{prefix}_length_sum"] += int(lengths[selected].sum())
            dead = selected & died
            self.values[f"{prefix}_deaths"] += int(dead.sum())
            self.values[f"{prefix}_death_length_sum"] += int(lengths[dead].sum())

    def take(self) -> Dict[str, int]:
        out = self.values
        self.values = {key: 0 for key in RESTART_KEYS}
        return out
