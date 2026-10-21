import numpy as np

from research.bl_ppo.tripwires import ActionWindows


def test_windows_cross_updates_and_strict_collapse_threshold() -> None:
    monitor = ActionWindows()
    actions = np.concatenate([np.ones(90_000, dtype=int), np.zeros(10_000, dtype=int)])
    assert monitor.observe(actions[:40_000], actions[:40_000]) == []
    assert monitor.observe(actions[40_000:], actions[40_000:]) == []
    actions[-1] = 1
    event = monitor.observe(actions, actions)
    assert event[0]["kind"] == "action_collapse"
    assert event[0]["window_index"] == 1 and sum(event[0]["action_counts"]) == 100_000


def test_boost_is_strictly_more_than_three_times_matched_b8() -> None:
    actions = np.arange(100_000) % 3
    actions[:30] = 3
    b8 = np.arange(100_000) % 3
    b8[:10] = 3
    assert ActionWindows().observe(actions, b8) == []
    actions[31] = 3
    assert ActionWindows().observe(actions, b8)[0]["kind"] == "boost_share"
    assert ActionWindows().observe(actions, np.zeros_like(actions))[0]["b8_boost_count"] == 0
