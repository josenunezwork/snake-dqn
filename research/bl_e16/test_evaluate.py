"""Closed-loop terminal-harness tests with deterministic tiny synthetic worlds."""

from typing import Any

import numpy as np
import pytest
import torch

from research.bl_e16.evaluate import analyze, evaluate_controller


class Driver:
    def __init__(self, world: dict[str, Any]) -> None:
        self.horizon, self.frame, self.last_action = world["horizon"], 0, 0

    def observe_hero(self) -> dict[str, np.ndarray]:
        # Policy reacts to its own preceding decision; an action tape differs.
        return {
            "local": np.array([[[[self.last_action]]]], dtype=np.uint8),
            "global": np.zeros((1, 1, 1, 1)),
            "scalars": np.zeros((1, 1)),
            "mask": np.ones((1, 6), dtype=bool),
        }

    def step(self, action: int) -> dict[str, Any]:
        self.frame += 1
        self.last_action = action
        alive = self.frame < 3
        return {"done": not alive, "alive": alive, "mass": 10.0 if alive else 0.0}


class Model:
    def __call__(self, local: torch.Tensor, *args: Any) -> torch.Tensor:
        logits = torch.zeros(1, 6)
        logits[0, (int(local.item()) + 1) % 6] = 10.0
        return logits


def test_candidate_reacts_and_dead_tail_is_zero() -> None:
    world = {"horizon": 5}
    result = evaluate_controller(world, "B8", Model(), Driver)
    assert result["lengths"] == [10.0, 10.0, 0.0, 0.0, 0.0]
    assert result["active_frames"] == 3
    assert result["mass_integral"] == 4.0
    assert result["alive_fraction"] == 0.4
    assert len(result["trace_sha256"]) == 64


def test_analysis_rejects_missing_pairs_wrong_family_and_respawn() -> None:
    world = {"world_id": "x", "world_seed": 123, "mix": "scripted", "horizon": 5}
    with pytest.raises(ValueError, match="missing"):
        analyze([], ["A"], scientific=False, expected_worlds=[world], training_status={})
    outcome = evaluate_controller(world, "B8", Model(), Driver)
    row = {**world, "controllers": {"B8": outcome, "parent": outcome, "A": outcome}}
    smoke = analyze([row], ["A"], scientific=False, expected_worlds=[world], training_status={})
    assert smoke["status"] == "SMOKE_ONLY"
    row["controllers"] = dict(row["controllers"])
    row["controllers"]["A"] = {**outcome, "lengths": [10.0, 0.0, 10.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="resurrects"):
        analyze([row], ["A"], scientific=False, expected_worlds=[world], training_status={})


def test_forged_alive_fraction_cannot_pass_guard() -> None:
    world = {"world_id": "x", "world_seed": 123, "mix": "scripted", "horizon": 5}
    outcome = evaluate_controller(world, "B8", Model(), Driver)
    row = {**world, "controllers": {name: dict(outcome) for name in ("B8", "parent", "A")}}
    row["controllers"]["A"]["alive_fraction"] = 1.0
    with pytest.raises(ValueError, match="alive-frame counter"):
        analyze([row], ["A"], scientific=False, expected_worlds=[world], training_status={})
