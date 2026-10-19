from __future__ import annotations

import json

from research.phase0_search_20261019 import analyze, bank


def test_bank_has_disjoint_registered_selection_and_fixture_worlds() -> None:
    value = bank.make_bank()
    assert {
        mix: sum(row["mix"] == mix for row in value["selection"]) for mix in bank.MIX_COUNTS
    } == bank.MIX_COUNTS
    all_rows = value["selection"] + [row for rows in value["fixtures"].values() for row in rows]
    assert len(all_rows) == len({row["world_seed"] for row in all_rows}) == 172
    assert len(value["fixtures"]["calibration"]) == len(value["fixtures"]["timing"]) == 3
    assert json.dumps(value, allow_nan=False)


def _complete_input(delta: float = 100.0) -> dict:
    value = bank.make_bank()
    file_sha = "a" * 64
    plan_sha = "b" * 64
    rows = []
    for world in value["selection"]:
        b8 = 100.0
        rows.append(
            {
                "plan_sha256": plan_sha,
                "bank_sha256": file_sha,
                "world_id": world["world_id"],
                "world_seed": world["world_seed"],
                "mix": world["mix"],
                "world_digest": world["world_semantic_sha256"],
                "world_identity": {
                    "horizon": 5000,
                    "world_semantic_sha256": world["world_semantic_sha256"],
                    "opponent_semantic_sha256": world["opponent_semantic_sha256"],
                },
                "opponent_semantics": world["opponent_semantic_sha256"],
                "complete": True,
                "arms": {
                    "b8": {
                        "mass_integral": b8,
                        "survived": world["mix"] != "scripted",
                        "frames_scored": 5000,
                    },
                    "search": {
                        "mass_integral": b8 + delta,
                        "survived": True,
                        "frames_scored": 5000,
                    },
                },
                "paired": {"mass_delta": delta},
            }
        )
    result = {
        "schema": analyze.INPUT_SCHEMA,
        "complete": True,
        "plan_sha256": plan_sha,
        "bank_file_sha256": file_sha,
        "bank": value,
        "bank_audit": {
            "pass": True,
            "world_bank_semantic_sha256": value["allocation_semantic_sha256"],
        },
        "plan_binding": {
            "kind": "selection",
            "config": {},
            "input_sha256": {
                "bank": file_sha,
                "audit": "c" * 64,
                "config": "d" * 64,
                "preregistration": "e" * 64,
                "checkpoint": analyze.B8_SHA256,
            },
            "source_closure_sha256": {"runner.py": "1" * 64},
        },
        "worlds": rows,
    }
    result["merged_sha256"] = analyze._runner_digest(result)
    return result


def test_analysis_uses_one_world_level_equal_mix_effect_and_go_precedence() -> None:
    result = analyze.analyze(_complete_input())
    assert result["decision"] == "GO"
    assert result["statistics"]["equal_mix_mass_effect"] == 100.0
    assert result["statistics"]["welch_df"] is None


def test_analysis_rejects_incomplete_or_tampered_merged_receipt() -> None:
    value = _complete_input()
    value["worlds"][0]["arms"]["search"]["mass_integral"] += 1
    assert analyze.analyze(value)["decision"] == "INVALID_INCOMPLETE"
