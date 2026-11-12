"""Bitwise golden-trace regression check recorded from pre-refactor Game V3.1 code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.obs import LearnerIndex, build_obs
from research.game_v3.sim import GameV3Sim

GOLDEN_DATA_PATH = Path(__file__).with_name("test_game_v3_golden_data.py")

_TENSOR_ATTRS: tuple[str, ...] = (
    "seeds",
    "world_ticks",
    "head_pos",
    "theta",
    "alive",
    "boosting",
    "length",
    "seg_count",
    "head_ptr",
    "oldest_active_subtick",
    "dist_accum_fp",
    "boost_ticks",
    "boost_drop_fp_accum",
    "respawn_timer",
    "life_age",
    "life_start",
    "truncated_pending",
    "ring_pos",
    "ring_cell",
    "ring_r_sq",
    "ring_subtick",
    "prev_alive",
    "prev_boosting",
    "prev_head_pos",
    "prev_theta",
    "prev_length",
    "occ_hi",
    "occ_lo",
    "occ_subtick_4x",
    "occ_subtick_16x",
    "ambient_grid",
    "ambient_grid_4x",
    "ambient_grid_16x",
    "corpse_grid",
    "corpse_grid_4x",
    "corpse_grid_16x",
    "ambient_count",
    "total_spawned_snake_mass",
    "total_spawned_ambient_mass",
    "total_corpse_decay_sink",
    "total_boost_decay_sink",
    "total_length_cap_sink",
    "total_truncation_sink",
)

_OBS_FIELDS: tuple[str, ...] = (
    "raster_s1",
    "raster_s4",
    "raster_s16",
    "scalars",
    "action_mask",
)

_STEP_INFO_FIELDS: tuple[str, ...] = (
    "alive",
    "life_start",
    "died",
    "truncated",
    "in_respawn_delay",
    "length",
    "prev_length",
    "mass_eaten",
    "killer_id",
    "kill_credit",
    "life_age",
)

_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "default_small",
        "cfg": GameV3Config(
            arena_radius=85.0,
            num_snakes=12,
            respawn_delay=5,
            max_life_ticks=25,
            ambient_food_density=0.018,
        ),
        "num_envs": 4,
        "seeds": [101, 202, 303, 404],
        "action_seed": 2026111101,
        "ticks": 60,
        "stride": 15,
        "learner_slots": 4,
    },
    {
        "name": "larger_arena",
        "cfg": GameV3Config(
            arena_radius=260.0,
            num_snakes=24,
            start_length=25,
            respawn_delay=6,
            ambient_food_density=0.008,
        ),
        "num_envs": 3,
        "seeds": [1101, 1202, 1303],
        "action_seed": 2026111102,
        "ticks": 60,
        "stride": 15,
        "learner_slots": 8,
    },
    {
        "name": "snakes_64",
        "cfg": GameV3Config(
            arena_radius=140.0,
            num_snakes=64,
            respawn_delay=4,
            ambient_food_density=0.015,
        ),
        "num_envs": 2,
        "seeds": [2101, 2202],
        "action_seed": 2026111103,
        "ticks": 60,
        "stride": 15,
        "learner_slots": 16,
    },
)


def _tensor_digest(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    shape_s = "x".join(str(int(d)) for d in arr.shape)
    return f"{arr.dtype.name}:{shape_s}:{digest}"


def compute_golden_traces() -> dict[str, Any]:
    """Run all three deterministic CPU golden-trace configurations and return sha256 digests."""
    traces: dict[str, Any] = {}
    for spec in _CASES:
        cfg: GameV3Config = spec["cfg"]
        num_envs = int(spec["num_envs"])
        ticks = int(spec["ticks"])
        stride = int(spec["stride"])
        learner_slots = int(spec["learner_slots"])

        sim = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=spec["seeds"])
        full_mask = torch.ones((num_envs, cfg.num_snakes), dtype=torch.bool)
        sub_mask = torch.zeros((num_envs, cfg.num_snakes), dtype=torch.bool)
        sub_mask[:, :learner_slots] = True
        sub_index = LearnerIndex(sim, sub_mask)

        rng = np.random.default_rng(int(spec["action_seed"]))
        checkpoints: dict[str, Any] = {}

        def _snapshot(last_info: Any | None) -> dict[str, Any]:
            state_digests = {name: _tensor_digest(getattr(sim, name)) for name in _TENSOR_ATTRS}
            obs_full = build_obs(sim, full_mask)
            obs_sub = build_obs(sim, sub_index)
            snap: dict[str, Any] = {
                "tick": int(sim.tick),
                "state": state_digests,
                "obs_full": {f: _tensor_digest(getattr(obs_full, f)) for f in _OBS_FIELDS},
                "obs_learner": {f: _tensor_digest(getattr(obs_sub, f)) for f in _OBS_FIELDS},
            }
            if last_info is not None:
                snap["step_info"] = {
                    f: _tensor_digest(getattr(last_info, f)) for f in _STEP_INFO_FIELDS
                }
            return snap

        checkpoints["0"] = _snapshot(None)
        for t in range(1, ticks + 1):
            actions_np = rng.integers(
                0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64
            )
            actions = torch.from_numpy(actions_np)
            truncate = t == ticks // 2
            info = sim.step(actions, truncate=truncate)
            if t % stride == 0 or t == ticks:
                checkpoints[str(t)] = _snapshot(info)

        traces[str(spec["name"])] = checkpoints
    return traces


def write_golden_data_module(target: Path = GOLDEN_DATA_PATH) -> None:
    """Write ``GOLDEN_TRACE`` JSON payload into a tracked test data module."""
    traces = compute_golden_traces()
    payload_json = json.dumps(traces, indent=2, sort_keys=True)
    content = (
        '"""Stored pre-refactor sha256 golden trace digests for Game V3.1."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "from typing import Any\n\n"
        f'GOLDEN_TRACE: dict[str, Any] = json.loads(r"""\n{payload_json}\n""")\n'
    )
    target.write_text(content, encoding="utf-8")


def test_simulator_and_observations_match_pre_refactor_golden_trace() -> None:
    """Verify state, StepInfo, and observation tensors match pre-refactor sha256 digests."""
    from tests.test_game_v3_golden_data import GOLDEN_TRACE

    actual = compute_golden_traces()
    assert set(actual.keys()) == set(GOLDEN_TRACE.keys())
    for case_name, expected_ckpts in GOLDEN_TRACE.items():
        got_ckpts = actual[case_name]
        assert set(got_ckpts.keys()) == set(expected_ckpts.keys()), f"case={case_name}"
        for tick_key, exp_snap in expected_ckpts.items():
            got_snap = got_ckpts[tick_key]
            assert got_snap["tick"] == exp_snap["tick"], f"{case_name} tick={tick_key}"
            for group in ("state", "obs_full", "obs_learner", "step_info"):
                if group not in exp_snap:
                    continue
                for field_name, exp_digest in exp_snap[group].items():
                    got_digest = got_snap[group][field_name]
                    assert got_digest == exp_digest, (
                        f"Golden trace mismatch in case={case_name!r} tick={tick_key} "
                        f"{group}.{field_name}: got {got_digest} != expected {exp_digest}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Generate and write tests/test_game_v3_golden_data.py from current code.",
    )
    args = parser.parse_args()
    if args.write:
        write_golden_data_module()
        print(f"Wrote {GOLDEN_DATA_PATH}")
    else:
        test_simulator_and_observations_match_pre_refactor_golden_trace()
        print("Golden trace matches stored baseline.")


if __name__ == "__main__":
    main()
