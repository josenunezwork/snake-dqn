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


def _resign(value: dict) -> dict:
    value.pop("merged_sha256", None)
    value["merged_sha256"] = analyze._runner_digest(value)
    return value


def test_decision_mass_boundaries_and_stop_precedence() -> None:
    for delta, expected in [
        (39.0, "STOP_TRAINING"),
        (40.0, "INCONCLUSIVE"),
        (79.0, "INCONCLUSIVE"),
        (80.0, "GO"),
    ]:
        assert analyze.analyze(_complete_input(delta))["decision"] == expected
    value = _complete_input(100.0)
    for row in value["worlds"]:
        row["arms"]["b8"]["survived"] = True
    assert analyze.analyze(_resign(value))["decision"] == "STOP_TRAINING"


def test_equal_mix_weighting_does_not_weight_scripted_three_times() -> None:
    value = _complete_input()
    effects = {"scripted": 0.0, "frozen": 60.0, "mixed": 180.0}
    for row in value["worlds"]:
        delta = effects[row["mix"]]
        row["arms"]["search"]["mass_integral"] = 100.0 + delta
        row["paired"]["mass_delta"] = delta
    result = analyze.analyze(_resign(value))
    assert result["statistics"]["equal_mix_mass_effect"] == 80.0
    assert result["decision"] == "GO"


def test_missing_hash_wrong_checkpoint_and_nonfinite_are_invalid() -> None:
    value = _complete_input()
    value.pop("merged_sha256")
    assert analyze.analyze(value)["decision"] == "INVALID_INCOMPLETE"
    value = _complete_input()
    value["plan_binding"]["input_sha256"]["checkpoint"] = "0" * 64
    assert analyze.analyze(_resign(value))["decision"] == "INVALID_INCOMPLETE"
    value = _complete_input()
    value["worlds"][0]["arms"]["search"]["mass_integral"] = float("nan")
    assert analyze.analyze(value)["decision"] == "INVALID_INCOMPLETE"


def test_collision_scan_matches_whole_decimal_world_id(tmp_path) -> None:
    candidate = bank.make_bank()["selection"][0]["world_seed"]
    (tmp_path / "prior.json").write_text(json.dumps({"world_seed": candidate}))
    (tmp_path / "near.json").write_text(json.dumps({"world_seed": f"{candidate}1"}))
    assert bank._scan_roots([(tmp_path, ("*.json",))], {candidate}) == {candidate}
