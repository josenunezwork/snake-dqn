from __future__ import annotations

import json

import pytest

from research.phase0_search_20261019 import parity, runner


def _world() -> dict:
    return {
        "world_id": "smoke-scripted-0000-7",
        "world_seed": 7,
        "mix": "scripted",
        "horizon": 2,
        "opponents": [["scripted", "greedy_food"]],
        "world_semantic_sha256": "world",
        "opponent_semantic_sha256": "opponents",
    }


def _result(plan: str) -> dict:
    world = _world()
    digest = "a" * 64
    trace = [
        {
            "frame": frame,
            "action": 0,
            "b8_action": 0,
            "searched": False,
            "q": [0.0] * 6,
            "mask": [True] * 6,
            "selection_mask": [True] * 6,
            "pre_world_digest": digest,
            "post_world_digest": digest,
        }
        for frame in (1, 2)
    ]
    value = {
        "complete": True,
        "plan_sha256": plan,
        "world_id": world["world_id"],
        "world_seed": world["world_seed"],
        "mix": world["mix"],
        "bank_sha256": "bank",
        "world_digest": world["world_semantic_sha256"],
        "world_identity": {
            key: world[key]
            for key in ("opponents", "horizon", "world_semantic_sha256", "opponent_semantic_sha256")
        },
        "arms": {
            arm: {
                "mass_integral": 1.0,
                "mass_sum": 2.0,
                "survived": True,
                "frames_scored": 2,
                "frames_executed": 2,
                "start_world_digest": digest,
                "final_world_digest": digest,
                "trace": trace,
            }
            for arm in ("b8", "search")
        },
    }
    for evidence in value["arms"].values():
        evidence["trace_sha256"] = runner.canonical_sha256(evidence["trace"])
    value["result_sha256"] = runner.canonical_sha256(runner._semantic_result(value))
    return value


def test_worker_limits_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.worker_limit("mac", 6) == 6
    with pytest.raises(runner.RunError):
        runner.worker_limit("mac", 7)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    assert runner.worker_limit("cpu-pod", 32) == 32


def test_resume_refuses_tampered_complete_partial(tmp_path) -> None:
    path = tmp_path / "world.json"
    path.write_text(json.dumps(_result("plan")))
    assert runner._valid_existing(
        path, {"plan_sha256": "plan", "input_sha256": {"bank": "bank"}}, _world()
    )
    value = json.loads(path.read_text())
    value["arms"]["search"]["mass_integral"] = 999.0
    path.write_text(json.dumps(value))
    assert not runner._valid_existing(
        path, {"plan_sha256": "plan", "input_sha256": {"bank": "bank"}}, _world()
    )


def test_tie_is_reported_but_not_accepted() -> None:
    trace_a = {
        "pre_world_digest": "same",
        "post_world_digest": "a",
        "mask": [True, True],
        "selection_mask": [True, True],
        "searched": False,
        "q": [1.0, 1.0],
        "action": 0,
    }
    trace_b = {
        "pre_world_digest": "same",
        "post_world_digest": "b",
        "mask": [True, True],
        "selection_mask": [True, True],
        "searched": False,
        "q": [1.0, 1.0],
        "action": 1,
    }

    def record(trace: dict) -> dict:
        return {
            "world_id": "w",
            "world_digest": "d",
            "arms": {"b8": {"trace": [trace]}, "search": {"trace": [trace]}},
        }

    result = parity.compare_records(record(trace_a), record(trace_b))
    assert result["pass"] is False
    assert result["tie_eligible"] is True
