"""Hermetic E1 checkpoint parity and inference RNG regression."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from src.core.game_config import initialize_config
from src.core.runtime_contract import EffectiveWorldConfig
from src.core.seeding import initialize_run_seed
from src.evaluation.anchors import ScriptedAnchor
from src.evaluation.protocol import promotion_v2_watch_rect
from src.model.raster_network import RasterDuelingNetwork
from src.scripts import tournament_eval as te
from src.simd_env import eval_engine as ee
from src.training.pqn_trainer import PQNConfig, PQNTrainer
from web.backend.session import _v3_serving_config

pytestmark = pytest.mark.usefixtures("setup_config")


def _record_anchor(context: Any, action: int) -> dict[str, Any]:
    """Serialize all inputs consumed by the shared scripted anchor."""
    return {
        "world_seed": int(context.world_seed),
        "slot": int(context.slot),
        "frame": int(context.frame),
        "head": tuple(int(v) for v in context.head_cell),
        "heading": tuple(int(v) for v in context.heading),
        "food": tuple(tuple(int(v) for v in cell) for cell in context.food_cells),
        "allowed": tuple(bool(v) for v in context.allowed_mask),
        "action": int(action),
    }


def _build_checkpoint(tmp_path):
    """Build the same zero-update raster31v3 artifact used by E1 acceptance."""
    seed = initialize_run_seed(827411)
    config = PQNConfig(
        num_envs=1,
        num_snakes=6,
        rollout_len=2,
        obs_spec="raster31v3",
        recipe="corrected-v3",
        flip_augment=False,
        max_frames=1234,
        starvation_max=77,
        max_length=66,
        seed=seed.effective_seed,
        requested_device="cpu",
        effective_device="cpu",
    )
    trainer = PQNTrainer(config, device=torch.device("cpu"))
    path = tmp_path / "actual_initial_pqn.pth"
    trainer.save_checkpoint(str(path))
    return path, torch.load(path, map_location="cpu", weights_only=False)


def test_actual_checkpoint_live_simd_inputs_and_anchor_trace_are_identical(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real wrappers agree through food replacement without inference RNG draws."""
    checkpoint_path, checkpoint = _build_checkpoint(tmp_path)
    world = EffectiveWorldConfig(**checkpoint["effective_world"])
    initialize_config(_v3_serving_config(world))
    profile = replace(
        promotion_v2_watch_rect(world),
        scored_horizon=3,
        observation_progress_horizon=1234,
    )
    hero = ("checkpoint", str(checkpoint_path))
    opponents = [("scripted", "greedy_food")] * 5
    phase = {"name": "live"}
    anchors: dict[str, list[dict[str, Any]]] = {"live": [], "simd": []}
    inputs: dict[str, list[dict[str, np.ndarray]]] = {"live": [], "simd": []}
    random_draws: dict[str, list[float]] = {"live": [], "simd": []}
    actual_anchor = ScriptedAnchor.action
    actual_forward = RasterDuelingNetwork.forward
    actual_random = random.random

    def trace_anchor(anchor: ScriptedAnchor, context: Any) -> int:
        action = actual_anchor(anchor, context)
        anchors[phase["name"]].append(_record_anchor(context, action))
        return action

    def trace_forward(network: Any, *values: Any, **kwargs: Any) -> Any:
        tensors = (
            values[0]
            if len(values) == 1
            else dict(zip(("tactical", "strategic", "scalars"), values))
        )
        inputs[phase["name"]].append(
            {name: tensors[name][0].detach().cpu().numpy().copy() for name in tensors}
        )
        return actual_forward(network, *values, **kwargs)

    def trace_random() -> float:
        value = actual_random()
        random_draws[phase["name"]].append(value)
        return value

    monkeypatch.setattr(ScriptedAnchor, "action", trace_anchor)
    monkeypatch.setattr(RasterDuelingNetwork, "forward", trace_forward)
    monkeypatch.setattr(random, "random", trace_random)
    live = te.rollout(hero, opponents, frames=3, seed=14033, profile=profile)
    phase["name"] = "simd"
    simd = ee.run_simd_eval(hero, opponents, frames=3, seeds=[14033], profile=profile)[0]

    assert len(inputs["live"]) == len(inputs["simd"]) == 3
    for frame, (left, right) in enumerate(zip(inputs["live"], inputs["simd"]), start=1):
        for name in ("tactical", "strategic", "scalars"):
            np.testing.assert_array_equal(left[name], right[name], err_msg=f"{name} frame {frame}")
    assert anchors["live"] == anchors["simd"]
    assert random_draws == {"live": [], "simd": []}
    assert live["world_identity"] == simd["world_identity"]
    for key in ("mass_integral", "kills", "deaths"):
        assert live[key] == simd[key]
    assert live["probes"]["food_eaten"] == simd["probes"]["food_eaten"]
