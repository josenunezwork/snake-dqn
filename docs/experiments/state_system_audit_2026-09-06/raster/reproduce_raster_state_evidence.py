#!/usr/bin/env python3
"""Reproduce the compact raster31v2 state-representation audit evidence.

This is an audit artifact, not a product test.  It deliberately uses one CPU
thread, tiny hand-built worlds, and two one-frame BatchSim transitions.  The
existence proofs below establish observation aliases or raster priority
failures; they do not estimate how often the cases occur or their effect on
trained-policy performance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.simd_env.batch_sim import BatchSim, BatchSimConfig  # noqa: E402
from src.simd_env.featurizer import (  # noqa: E402
    CODE_ENEMY_BODY,
    CODE_ENEMY_PRED,
    CODE_WALL,
    ObsInputs,
    build_observations,
    obs_inputs_from_batch_sim,
)

OBS_FIELDS = ("tactical_uint8", "strategic_uint8", "scalars", "mask")


def _jsonable(value: Any) -> Any:
    """Convert NumPy values recursively for deterministic JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write(name: str, payload: Dict[str, Any]) -> None:
    (ARTIFACT_DIR / name).write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _entity(label: str, body: Sequence[Tuple[int, int]], heading: int) -> Dict[str, Any]:
    return {"label": label, "body_head_to_tail": list(body), "heading": heading}


def _static_inputs(
    entities: Sequence[Dict[str, Any]],
    *,
    grid_w: int = 80,
    grid_h: int = 80,
    food: Sequence[Tuple[int, int]] = (),
    corpse: Iterable[Tuple[int, int]] = (),
    arena_type_flag: float = 0.0,
) -> ObsInputs:
    """Build a one-environment ObsInputs snapshot from compact entity arrays."""
    snake_count = len(entities)
    max_body = max(len(entity["body_head_to_tail"]) for entity in entities)
    bodies = np.zeros((1, snake_count, max_body, 2), dtype=np.int64)
    body_len = np.zeros((1, snake_count), dtype=np.int64)
    heading = np.zeros((1, snake_count), dtype=np.int64)
    for slot, entity in enumerate(entities):
        body = np.asarray(entity["body_head_to_tail"], dtype=np.int64)
        bodies[0, slot, : len(body)] = body
        body_len[0, slot] = len(body)
        heading[0, slot] = int(entity["heading"])

    food_slots = max(1, len(food))
    food_cells = np.zeros((1, food_slots, 2), dtype=np.int64)
    food_mass = np.zeros((1, food_slots), dtype=np.float64)
    food_is_corpse = np.zeros((1, food_slots), dtype=bool)
    corpse_set = set(corpse)
    for index, cell in enumerate(food):
        food_cells[0, index] = cell
        food_mass[0, index] = 1.0
        food_is_corpse[0, index] = cell in corpse_set

    return ObsInputs(
        heads=bodies[:, :, 0].copy(),
        bodies=bodies,
        body_len=body_len,
        lengths=body_len.copy(),
        alive=np.ones((1, snake_count), dtype=bool),
        heading=heading,
        boost_frames=np.zeros((1, snake_count), dtype=np.int64),
        frames_since_food=np.zeros((1, snake_count), dtype=np.int64),
        boosting=np.zeros((1, snake_count), dtype=bool),
        food_cells=food_cells,
        food_mass=food_mass,
        food_is_corpse=food_is_corpse,
        grid_w=grid_w,
        grid_h=grid_h,
        max_snakes=snake_count,
        starvation_max=500,
        max_length=400,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.zeros(1, dtype=np.int64),
        max_frames=5000,
        arena_type_flag=arena_type_flag,
    )


def _hero_comparison(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field in OBS_FIELDS:
        a = np.asarray(left[field])[0, 0]
        b = np.asarray(right[field])[0, 0]
        result[field] = {
            "equal": bool(np.array_equal(a, b)),
            "different_elements": int(np.count_nonzero(a != b)),
        }
    return result


def _hero_field_equal(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, bool]:
    return {
        field: bool(np.array_equal(np.asarray(left[field])[0, 0], np.asarray(right[field])[0, 0]))
        for field in OBS_FIELDS
    }


def _slot_permutation_probe(commit: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hero = _entity("hero", [(20, 20)], 0)
    body_enemy = _entity("enemy_body_source", [(24, 14), (25, 14), (26, 14)], 3)
    pred_enemy = _entity("enemy_prediction_source", [(25, 15), (25, 16), (25, 17)], 0)
    order_body_then_prediction = [hero, body_enemy, pred_enemy]
    order_prediction_then_body = [hero, pred_enemy, body_enemy]

    left = build_observations(_static_inputs(order_body_then_prediction))
    right = build_observations(_static_inputs(order_prediction_then_body))
    row, col = 17, 20
    left_code = int(left["tactical_uint8"][0, 0, 0, row, col])
    right_code = int(right["tactical_uint8"][0, 0, 0, row, col])
    comparison = _hero_comparison(left, right)
    diff_indices = np.argwhere(
        left["tactical_uint8"][0, 0] != right["tactical_uint8"][0, 0]
    )

    assert left_code == CODE_ENEMY_PRED
    assert right_code == CODE_ENEMY_BODY
    assert comparison["tactical_uint8"]["different_elements"] == 2
    assert all(comparison[field]["equal"] for field in OBS_FIELDS[1:])

    evidence = {
        "source_commit": commit,
        "classification": "confirmed source-slot permutation dependence",
        "interpretation": (
            "The same physical world yields a different hero raster solely because enemy slots "
            "are swapped. This is an invariant failure, not a measured policy-performance loss."
        ),
        "world": {
            "grid_cells": [80, 80],
            "observer_slot": 0,
            "entities_by_identity": [hero, body_enemy, pred_enemy],
            "slot_order_body_then_prediction": [entity["label"] for entity in order_body_then_prediction],
            "slot_order_prediction_then_body": [entity["label"] for entity in order_prediction_then_body],
        },
        "physical_world_same": True,
        "only_enemy_slot_order_swapped": True,
        "contested_world_cell": [25, 14],
        "contested_raster_cell": [row, col],
        "type_code_body_first_then_prediction": left_code,
        "type_code_prediction_first_then_body": right_code,
        "declared_codes": {"enemy_body": CODE_ENEMY_BODY, "enemy_prediction": CODE_ENEMY_PRED},
        "hero_observation_comparison": comparison,
        "hero_tactical_diff_indices_plane_row_col": diff_indices,
    }

    priority = {
        "world": evidence["world"],
        "contested_world_cell": [25, 14],
        "contested_raster_cell": [row, col],
        "expected_by_declared_priority": CODE_ENEMY_BODY,
        "actual_code": left_code,
        "confirmed_violation": left_code != CODE_ENEMY_BODY,
    }
    return evidence, priority


def _wall_overwrite_probe() -> Dict[str, Any]:
    entities = [
        _entity("hero", [(20, 2)], 0),
        _entity("enemy", [(25, 0)], 0),
    ]
    obs = build_observations(_static_inputs(entities))
    row, col = 20, 20
    actual = int(obs["tactical_uint8"][0, 0, 0, row, col])
    assert actual == CODE_ENEMY_PRED
    return {
        "world": {"grid_cells": [80, 80], "observer_slot": 0, "entities": entities},
        "predicted_world_cell": [25, -1],
        "cell_raster": [row, col],
        "wall_code": CODE_WALL,
        "prediction_code": CODE_ENEMY_PRED,
        "actual_code": actual,
        "confirmed_wall_erased": actual != CODE_WALL,
        "interpretation": (
            "A geometrically impossible enemy prediction and the wall occupy one categorical "
            "cell; the emitted type plane retains only the prediction. This proves information "
            "loss but does not quantify a control-performance effect."
        ),
    }


def _set_single_segment_world(
    sim: BatchSim,
    heads: Sequence[Tuple[int, int]],
    headings: Sequence[int],
    food: Sequence[Tuple[int, int]] = (),
    corpse: Iterable[Tuple[int, int]] = (),
) -> None:
    sim.bodies[:] = 0
    sim.head_ptr[:] = 0
    sim.seg_count[:] = 1
    sim.length[:] = 1
    sim.alive[:] = True
    sim.direction[0] = np.asarray(headings, dtype=np.int64)
    sim.boost_frames[:] = 0
    sim.frames_since_food[:] = 0
    sim._boosted_this_step[:] = False
    sim._reward_prev_length[:] = 1
    for slot, head in enumerate(heads):
        sim.bodies[0, slot, 0] = head
    sim.food_cells[0] = list(food)
    sim.food_set[0] = set(food)
    sim.corpse_cells[0] = set(corpse)
    sim._rebuild_traversed_from_heads()
    sim._last_mask = sim._compute_action_masks()


def _observe(sim: BatchSim) -> Dict[str, np.ndarray]:
    return build_observations(obs_inputs_from_batch_sim(sim), sim.get_action_mask())


def _remote_heading_probe(commit: str) -> Dict[str, Any]:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=2,
        game_width=800,
        game_height=800,
        initial_food=0,
        max_food=0,
        mechanics_version=2,
    )
    up = BatchSim(cfg, seeds=[281])
    down = BatchSim(cfg, seeds=[281])
    heads = [(20, 20), (20, 50)]
    _set_single_segment_world(up, heads, [0, 0])
    _set_single_segment_world(down, heads, [0, 2])

    current_up, current_down = _observe(up), _observe(down)
    current_equal = _hero_field_equal(current_up, current_down)
    joint_action = np.asarray([[1, 1]], dtype=np.int64)
    up.step(joint_action)
    down.step(joint_action)
    next_up, next_down = _observe(up), _observe(down)
    next_equal = _hero_field_equal(next_up, next_down)
    next_scalar_diff = np.flatnonzero(
        next_up["scalars"][0, 0] != next_down["scalars"][0, 0]
    )

    assert all(current_equal.values())
    assert not all(next_equal.values())
    assert np.array_equal(up.get_reward()[:, 0], down.get_reward()[:, 0])
    assert np.array_equal(up.get_done()[:, 0], down.get_done()[:, 0])

    return {
        "source_commit": commit,
        "classification": "confirmed exact observation alias with divergent next observation",
        "interpretation": (
            "The controlled hero receives exactly the same raster31v2 observation in both worlds. "
            "The same full joint relative-action vector produces different next observations. "
            "The first-step hero reward and done are equal. This is an alias-existence proof; it "
            "does not measure frequency, preferred-action disagreement, or performance loss."
        ),
        "initial_worlds": [
            {
                "label": "enemy_heading_up",
                "entities": [
                    _entity("hero", [heads[0]], 0),
                    _entity("enemy", [heads[1]], 0),
                ],
                "food": [],
            },
            {
                "label": "enemy_heading_down",
                "entities": [
                    _entity("hero", [heads[0]], 0),
                    _entity("enemy", [heads[1]], 2),
                ],
                "food": [],
            },
        ],
        "hidden_enemy_headings": [0, 2],
        "same_relative_actions": joint_action[0],
        "current_hero_observation_fields_equal": current_equal,
        "next_enemy_heads": [up.get_heads()[0, 1], down.get_heads()[0, 1]],
        "next_hero_observation_fields_equal": next_equal,
        "next_hero_scalar_different_indices": next_scalar_diff,
        "hero_rewards": [float(up.get_reward()[0, 0]), float(down.get_reward()[0, 0])],
        "hero_done": [bool(up.get_done()[0, 0]), bool(down.get_done()[0, 0])],
    }


def _food_class_probe(commit: str) -> Dict[str, Any]:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=1,
        game_width=800,
        game_height=800,
        initial_food=0,
        max_food=1,
        mechanics_version=2,
    )
    ambient = BatchSim(cfg, seeds=[177])
    corpse = BatchSim(cfg, seeds=[177])
    head = tuple(int(v) for v in ambient.get_heads()[0, 0])
    corners = [(1, 1), (78, 1), (1, 78), (78, 78)]
    food_cell = max(corners, key=lambda cell: max(abs(cell[0] - head[0]), abs(cell[1] - head[1])))
    heading = int(ambient.get_directions()[0, 0])
    _set_single_segment_world(ambient, [head], [heading], [food_cell], [])
    _set_single_segment_world(corpse, [head], [heading], [food_cell], [food_cell])

    current_ambient, current_corpse = _observe(ambient), _observe(corpse)
    current_equal = _hero_field_equal(current_ambient, current_corpse)
    safe_actions = np.flatnonzero(ambient.get_action_mask()[0, 0])
    action = int(safe_actions[0])
    one_action = np.asarray([[action]], dtype=np.int64)
    ambient.step(one_action)
    corpse.step(one_action)
    next_ambient, next_corpse = _observe(ambient), _observe(corpse)
    next_equal = _hero_field_equal(next_ambient, next_corpse)
    next_scalar_diff = np.flatnonzero(
        next_ambient["scalars"][0, 0] != next_corpse["scalars"][0, 0]
    )

    assert all(current_equal.values())
    assert not all(next_equal.values())
    assert np.array_equal(ambient.get_reward(), corpse.get_reward())
    assert np.array_equal(ambient.get_done(), corpse.get_done())
    assert [len(ambient.get_food(0)), len(corpse.get_food(0))] == [1, 2]

    return {
        "source_commit": commit,
        "classification": "confirmed exact observation alias with divergent next observation",
        "interpretation": (
            "The hero receives exactly the same raster31v2 observation when a remote pellet's "
            "hidden class is ambient versus corpse. Under the same action, v2 food maintenance "
            "produces different next states because corpse food is cap-exempt. Reward and done are "
            "equal on this step. This proves an alias exists; it does not measure its frequency or "
            "control-performance loss."
        ),
        "initial_worlds": [
            {
                "label": "ambient",
                "entities": [_entity("hero", [head], heading)],
                "food": [{"cell": food_cell, "class": "ambient"}],
            },
            {
                "label": "corpse",
                "entities": [_entity("hero", [head], heading)],
                "food": [{"cell": food_cell, "class": "corpse"}],
            },
        ],
        "seed": 177,
        "hero_head": head,
        "hero_heading": heading,
        "food_cell": food_cell,
        "hidden_classes": ["ambient", "corpse"],
        "same_hero_action": action,
        "current_hero_observation_fields_equal": current_equal,
        "next_food_cells": [ambient.get_food(0), corpse.get_food(0)],
        "next_food_counts": [len(ambient.get_food(0)), len(corpse.get_food(0))],
        "next_hero_observation_fields_equal": next_equal,
        "next_hero_scalar_different_indices": next_scalar_diff,
        "reward_equal": bool(np.array_equal(ambient.get_reward(), corpse.get_reward())),
        "done_equal": bool(np.array_equal(ambient.get_done(), corpse.get_done())),
    }


def _strategic_footprint_probe() -> Dict[str, Any]:
    hero = _entity("hero", [(70, 40)], 0)
    near = build_observations(
        _static_inputs([hero, _entity("enemy", [(132, 40)], 0)], grid_w=145, grid_h=83)
    )
    far = build_observations(
        _static_inputs([hero, _entity("enemy", [(133, 40)], 0)], grid_w=145, grid_h=83)
    )
    near_sum = int(near["strategic_uint8"][0, 0, 0].sum())
    far_sum = int(far["strategic_uint8"][0, 0, 0].sum())
    assert near_sum == 10 and far_sum == 0
    return {
        "board_width_cells": 145,
        "hero_head": [70, 40],
        "enemy_heads": [[132, 40], [133, 40]],
        "lateral_62_channel_sum": near_sum,
        "lateral_63_channel_sum": far_sum,
    }


def _circular_capacity_probe() -> Dict[str, Any]:
    rectangular = _static_inputs([_entity("hero", [(20, 20)], 0)])
    circular = replace(rectangular, arena_type_flag=1.0)
    a = build_observations(rectangular)
    b = build_observations(circular)
    scalar_diff = np.flatnonzero(a["scalars"][0, 0] != b["scalars"][0, 0])
    assert np.array_equal(a["tactical_uint8"], b["tactical_uint8"])
    assert np.array_equal(a["strategic_uint8"], b["strategic_uint8"])
    assert scalar_diff.tolist() == [25]
    return {
        "obsinputs_has_circle_geometry": False,
        "tactical_equal": True,
        "strategic_equal": True,
        "scalar_diff_indices": scalar_diff,
    }


def _network_probe(commit: str) -> Dict[str, Any]:
    import torch.nn as nn

    from src.model.raster_network import RasterDuelingNetwork

    network = RasterDuelingNetwork()
    names = (
        "tactical_conv",
        "tactical_fc",
        "strategic_conv",
        "strategic_fc",
        "fuse_fc",
        "value_stream",
        "advantage_stream",
    )
    counts = {
        name: sum(parameter.numel() for parameter in getattr(network, name).parameters())
        for name in names
    }
    total = sum(parameter.numel() for parameter in network.parameters())
    counts["total"] = total
    return {
        "source_commit": commit,
        "parameters": counts,
        "fractions": {name: round(counts[name] / total, 4) for name in names},
        "conv_layernorm_count": sum(
            isinstance(module, nn.LayerNorm)
            for trunk in (network.tactical_conv, network.strategic_conv)
            for module in trunk.modules()
        ),
        "fc_layernorm_count": sum(isinstance(module, nn.LayerNorm) for module in network.modules()),
        "tactical_receptive_field_raw_cells": 9,
        "tactical_final_stride_raw_cells": 4,
        "strategic_receptive_field_coarse_cells": 7,
        "strategic_final_stride_coarse_cells": 4,
        "strategic_receptive_field_raw_cells": 35,
    }


def main() -> None:
    started = time.perf_counter()
    commit = _commit()
    slot_evidence, priority_evidence = _slot_permutation_probe(commit)
    wall_evidence = _wall_overwrite_probe()
    heading_evidence = _remote_heading_probe(commit)
    food_evidence = _food_class_probe(commit)

    outputs = {
        "enemy_slot_permutation.json": slot_evidence,
        "priority_and_wall.json": {
            "source_commit": commit,
            "body_prediction_priority": priority_evidence,
            "outside_prediction_overwrites_wall": wall_evidence,
        },
        "remote_heading_transition.json": heading_evidence,
        "food_class_transition.json": food_evidence,
        "probes.json": {
            "source_commit": commit,
            "priority_overwrite": priority_evidence,
            "outside_prediction_overwrites_wall": wall_evidence,
            "remote_enemy_heading_alias": {
                "world_states": {
                    "hero_head": heading_evidence["initial_worlds"][0]["entities"][0][
                        "body_head_to_tail"
                    ][0],
                    "enemy_head": heading_evidence["initial_worlds"][0]["entities"][1][
                        "body_head_to_tail"
                    ][0],
                    "hidden_headings": heading_evidence["hidden_enemy_headings"],
                },
                "same_relative_actions_hero_enemy": heading_evidence["same_relative_actions"],
                "controlled_hero_current_fields_equal": heading_evidence[
                    "current_hero_observation_fields_equal"
                ],
                "controlled_hero_next_fields_equal": heading_evidence[
                    "next_hero_observation_fields_equal"
                ],
            },
            "remote_food_class_alias": {
                "food_cell": food_evidence["food_cell"],
                "hidden_classes": food_evidence["hidden_classes"],
                "controlled_hero_fields_equal": food_evidence[
                    "current_hero_observation_fields_equal"
                ],
            },
            "strategic_footprint": _strategic_footprint_probe(),
            "circular_geometry_capacity": _circular_capacity_probe(),
            "interpretation": (
                "Alias cases prove representation non-Markovness by construction; they do not "
                "by themselves establish occurrence rate, preferred-action disagreement, or "
                "trained-policy performance loss."
            ),
        },
        "network_analysis.json": _network_probe(commit),
    }

    for name, payload in outputs.items():
        _write(name, payload)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_commit": commit,
                "elapsed_seconds": round(elapsed, 3),
                "outputs": sorted(outputs),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
