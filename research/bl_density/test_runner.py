from __future__ import annotations

import numpy as np

from research.bl_density.runner import cheap_prefilter, effective_q_features, retrospective_doom


def test_cheap_prefilter_boundaries() -> None:
    assert cheap_prefilter(np.array([159, 500, 500]), np.ones(6, dtype=bool))
    assert not cheap_prefilter(
        np.array([160, 500, 500]), np.array([True, True, True, False, False, False])
    )
    assert cheap_prefilter(
        np.array([160, 500, 500]), np.array([True, True, False, True, True, True])
    )


def test_retrospective_doom_window_is_transition_exclusive_at_300() -> None:
    assert retrospective_doom(20, 20)
    assert retrospective_doom(319, 20)
    assert not retrospective_doom(320, 20)
    assert not retrospective_doom(19, 20)


def test_q_fallback_uses_normal_actions_and_single_action_gap_zero() -> None:
    action, values = effective_q_features(np.array([1, 4, 3, 99, 98, 97]), np.zeros(6, dtype=bool))
    assert action == 1 and values[0] == 1.0
    action, values = effective_q_features(
        np.array([1, 4, 3, 99, 98, 97]), np.array([False, True, False, False, False, False])
    )
    assert action == 1 and values[0] == 0.0


def test_frozen_pool_rotates_across_the_whole_cell() -> None:
    from pathlib import Path

    from research.apex_safety_20260926 import dev_screen as ds
    from research.bl_density.runner import planned_jobs
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    context = {"ds": ds, "lookup": ds.agent_lookup({sha: name for name, sha in ds.POOL})}
    jobs = planned_jobs(
        context,
        b8=Path("b8.pth"),
        arms=["A"],
        mixes=["frozen"],
        episodes=6,
        horizon=5000,
        tiny=False,
        deadline=None,
    )
    seeds = [job["plan"]["seed"] for job in jobs]
    expected = planned_rosters(context, seeds, ["frozen"] * 6)
    assert [job["plan"] for job in jobs] == expected
    assert len({tuple(tuple(spec) for spec in job["plan"]["specs"]) for job in jobs}) > 1
