"""Frozen 100k-decision action/boost windows with matched B8 decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ActionWindows:
    """State is included in full boundary checkpoints and never reset at PPO updates."""

    count: int = 0
    action_counts: list[int] = field(default_factory=lambda: [0] * 6)
    b8_boost_count: int = 0
    completed_windows: int = 0

    def observe(self, actions: Any, b8_actions: Any) -> list[dict[str, Any]]:
        actions = np.asarray(actions).reshape(-1)
        baseline = np.asarray(b8_actions).reshape(-1)
        if actions.shape != baseline.shape or np.any((actions < 0) | (actions >= 6)):
            raise ValueError("tripwire actions must be matched legal six-action arrays")
        if np.any((baseline < 0) | (baseline >= 6)):
            raise ValueError("invalid matched B8 action")
        events = []
        offset = 0
        while offset < len(actions):
            take = min(100_000 - self.count, len(actions) - offset)
            chunk = actions[offset : offset + take].astype(np.int64)
            self.action_counts = (
                np.asarray(self.action_counts) + np.bincount(chunk, minlength=6)
            ).tolist()
            self.b8_boost_count += int((baseline[offset : offset + take] >= 3).sum())
            self.count += take
            offset += take
            if self.count == 100_000:
                evidence = {
                    "window_count": self.count,
                    "action_counts": list(self.action_counts),
                    "b8_boost_count": self.b8_boost_count,
                    "window_index": self.completed_windows,
                }
                if max(self.action_counts) > 90_000:
                    events.append({"kind": "action_collapse", **evidence})
                if sum(self.action_counts[3:]) > 3 * self.b8_boost_count:
                    events.append({"kind": "boost_share", **evidence})
                self.count, self.action_counts, self.b8_boost_count = 0, [0] * 6, 0
                self.completed_windows += 1
        return events
