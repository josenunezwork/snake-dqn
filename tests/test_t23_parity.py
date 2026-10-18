"""Focused contract tests for the portable T23 parity comparator."""

from __future__ import annotations

from research.t23_r0_screen_20261017.parity import (
    B8_SHA256,
    HORIZON,
    SCHEMA,
    _decision_diagnostic,
    compare_artifacts,
)


def _artifact() -> dict:
    worlds = []
    for index, mix in enumerate(("scripted", "scripted", "frozen", "mixed")):
        worlds.append(
            {
                "world_id": f"reserved-{index}",
                "world_seed": 900 + index,
                "mix": mix,
                "roster_digest": f"r{index}",
                "world_digest": f"w{index}",
                "record": {"mass_integral": index},
                "frame_digests": [f"f{index}" for _ in range(HORIZON)],
                "actions": [
                    {
                        "frame": 1,
                        "slot": 0,
                        "policy": "ego2s",
                        "action": 1,
                        "q": [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                    }
                ],
            }
        )
    return {
        "schema": SCHEMA,
        "horizon": HORIZON,
        "bank_sha256": "bank",
        "b8_sha256": B8_SHA256,
        "profile_digest": "profile",
        "source_sha256": "source",
        "worlds": worlds,
    }


def test_equal_artifacts_pass():
    result = compare_artifacts(_artifact(), _artifact())
    assert result["pass"]


def test_action_mismatch_fails_and_records_tie_information():
    right = _artifact()
    right["worlds"][0]["actions"][0]["action"] = 2
    right["worlds"][0]["actions"][0]["q"] = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    result = compare_artifacts(_artifact(), right)
    assert not result["pass"]
    assert "action mismatch" in " ".join(result["errors"])
    assert result["diagnostics"][0]["tie_diagnostic"]["colab_top2_ulp"] == 0


def test_missing_or_different_identity_fails_closed():
    right = _artifact()
    right["worlds"].pop()
    right["bank_sha256"] = "other"
    result = compare_artifacts(_artifact(), right)
    assert not result["pass"]
    assert any("identity mismatch" in message for message in result["errors"])
    assert any("world identities" in message for message in result["errors"])


def test_tie_diagnostic_uses_eligible_actions_not_an_illegal_high_q():
    packet = {
        "frame": 1,
        "slot": 0,
        "policy": "ego2s",
        "action": 1,
        "q": [100.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "mask": [False, True, True, False, False, False],
    }
    diagnostic = _decision_diagnostic(packet, packet)
    assert diagnostic["mac_top_action"] == 2
    assert diagnostic["mac_top2_ulp"] == 0
