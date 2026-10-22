"""Counts and denominators must remain visible even with no natural doom events."""

from research.bl_density.analysis import summarize


def _record(arm="A", complete=True):
    return {
        "episode_id": arm + "/scripted/0",
        "arm": arm,
        "mix": "scripted",
        "complete": complete,
        "decisions": 100,
        "cpu_seconds": 2.0,
        "wall_seconds": 3.0,
        "h300_calls": 1,
        "h300_branch_decisions": 50,
        "h300_cpu_seconds": 0.5,
        "lengths": [100, 800],
        "sampled": [
            {
                "length": 800,
                "prefilter": True,
                "h300_doom": True,
                "natural_doom": True,
                "features": [1.0],
            },
            {"length": 900, "prefilter": False, "natural_doom": True},
        ],
    }


def test_rates_use_all_live_decisions_and_cpu_not_wall_time():
    value = summarize([_record()], ["q_gap"])
    cell = value["cells"]["A"]["scripted"]
    assert cell["doom_moments_per_cpu_second"] == 0.5
    assert cell["doom_moments_per_hero_decision"] == 0.01
    recall = value["arm_a_prefilter_recall"]["overall"]
    assert recall["recall"] == 0.5
    assert recall["rejected_true_dooms"] == 1
    assert value["realism"]["scripted"]["B50"]["status"] == "INSUFFICIENT"


def test_partial_episodes_are_disclosed_and_excluded_from_cell_metrics():
    value = summarize([_record(complete=False)], ["q_gap"])
    assert value["cells"]["A"]["scripted"]["hero_decisions"] == 0
    assert value["partial_episode_cpu_seconds"] == 2
    assert value["arm_a_prefilter_recall"]["overall"]["recall"] is None
