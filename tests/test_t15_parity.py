"""Focused no-artifact checks for the T15 mixed-roster identity harness."""

import json

from research.t15_mixed_roster_20261017 import parity


def test_first_difference_handles_equal_and_length_mismatches():
    assert parity._first_difference(["a", "b"], ["a", "b"]) is None
    assert parity._first_difference(["a", "b"], ["a", "c"]) == 1
    assert parity._first_difference(["a"], ["a", "b"]) == 1


def test_comparison_is_hash_bound_and_reports_a_numeric_packet_difference():
    left = [{"slot": 1, "state_sha256": "a", "q_sha256": "b", "action": 2}]
    same = parity._comparison(left, list(left))
    changed = parity._comparison(left, [{**left[0], "action": 1}])
    assert same["identical"] is True and same["first_difference"] is None
    assert changed["identical"] is False and changed["first_difference"] == 0
    assert changed["live_sha256"] != changed["eval_sha256"]


def test_packet_chain_hash_survives_sorted_key_json_roundtrip():
    packets = [{"slot": 1, "q_sha256": "q", "state_sha256": "state", "action": 2}, None]
    reloaded = json.loads(json.dumps(packets, sort_keys=True, separators=(",", ":")))
    assert parity._digest(packets) == parity._digest(reloaded)
    assert parity._comparison(packets, reloaded)["identical"] is True


def test_mixed_environment_is_scoped(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "champion")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v7")
    with parity._mixed_environment():
        assert parity.os.environ["SNAKE_SERVE_WATCH_ROSTER"] == "mixed"
        assert "SNAKE_SERVE_CHECKPOINT" not in parity.os.environ
    assert parity.os.environ["SNAKE_SERVE_CHECKPOINT"] == "champion"
    assert parity.os.environ["SNAKE_SERVE_VETO_VARIANT"] == "v7"


def test_terminal_hero_and_inactive_opponents_keep_frame_aligned_evidence():
    """A terminal hero has no later action, but the registered horizon is retained."""
    hero = {"slot": 0, "state_sha256": "state", "q_sha256": "q", "action": 1}
    opponent = {"slot": 1, "state_sha256": "vstate", "q_sha256": "vq", "action": 2}
    hero_result = parity._comparison([hero, None, None], [hero, None, None])
    vector_result = parity._comparison([[opponent], [], []], [[opponent], [], []])
    assert hero_result["identical"] is True
    assert hero_result["live_count"] == 3
    assert vector_result["identical"] is True
    assert vector_result["eval_count"] == 3
    active_result = parity._comparison(
        list(zip([True, False, False], [[1], [], []])),
        list(zip([True, False, False], [[1], [], []])),
    )
    assert active_result["identical"] is True
