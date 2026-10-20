"""Prospective execution amendment regressions; no gameplay is consumed."""

from __future__ import annotations

import math

from research.t34_pathb.retention_v2 import (
    MAC_ARMS,
    MAX_CPU_HOURS,
    MAX_ELAPSED_HOURS,
    PRECHECK_MAX_ELAPSED_HOURS,
    unpaced_colab_policy,
)


def test_colab_cost_is_separate_smoke_priced_and_reserved() -> None:
    policy = unpaced_colab_policy()
    base = policy["smoke_seconds_per_world"] * policy["worlds"] / 3600 * 2.0
    assert policy["unit_cap"] == math.ceil(base * 1.2 * 1e6) / 1e6
    assert policy["reserve_units"] == math.ceil(base * 0.2 * 1e6) / 1e6
    assert policy["descriptive_only"] is True
    assert policy["max_account_assignments"] == 3
    assert MAC_ARMS == ("bare", "paced")
    assert (MAX_ELAPSED_HOURS, MAX_CPU_HOURS, PRECHECK_MAX_ELAPSED_HOURS) == (40.0, 60.0, 8.0)
