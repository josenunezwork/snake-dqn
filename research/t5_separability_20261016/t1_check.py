"""Replay the T1 late LADs and measure T5's action-area signal.

This is deliberately a deterministic replay measurement, not a new rollout
study.  Each state is reconstructed from the immutable T1 census action tape,
then checked against its recorded frame, action, length, and world seed before
the NumPy-only T5 features are read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

# This module is a local CPU-only measurement even when invoked outside the
# campaign launcher.  Set these before importing NumPy/Torch transitively.
os.environ["SNAKE_DQN_DEVICE"] = "cpu"
for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402

from research.b8_death_census_20261014 import census  # noqa: E402
from research.t1_commitment_20261015.replay import (  # noqa: E402
    ARTIFACTS as T1_ARTIFACTS,
)
from research.t1_commitment_20261015.replay import (  # noqa: E402
    assert_same_world,
    build_replay,
    configure_cpu,
    load_inputs,
)
from src.simd_env.t5_features import compute_slot_features  # noqa: E402

DEFAULT_OUT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016")
FEATURES_PER_ACTION = 4
AREA_LOG = 0
BOOST3_AREA_LOG = 1


def _late_lads(rows_path: Path) -> list[dict[str, Any]]:
    """Load exactly the canonical 46 late T1 last-avoidable decisions."""
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    selected = [row for row in rows if row.get("stratum") == "late" and row.get("lad")]
    if len(selected) != 46:
        raise ValueError(f"expected 46 late LADs, found {len(selected)}")
    keys = {(str(row["arm"]), int(row["env"]), int(row["lad"]["decision"])) for row in selected}
    if len(keys) != len(selected):
        raise ValueError("late LAD identity is not unique")
    return selected


def _action_values(features: np.ndarray, action: int) -> dict[str, float]:
    offset = int(action) * FEATURES_PER_ACTION
    return {
        "area_log": float(features[offset + AREA_LOG]),
        "boost3_area_log": float(features[offset + BOOST3_AREA_LOG]),
    }


_STATE_ARRAYS = (
    "alive",
    "seg_count",
    "head_ptr",
    "bodies",
    "length",
    "boost_frames",
    "direction",
)


def _save_feature_states(
    path: Path, states: list[tuple[dict[str, Any], dict[str, np.ndarray]]]
) -> None:
    """Persist the exact minimal simulator state T5's read-only feature API uses."""
    metadata = []
    arrays: dict[str, np.ndarray] = {}
    for index, (identity, state) in enumerate(states):
        metadata.append(identity)
        for name, value in state.items():
            arrays[f"s{index}_{name}"] = value
    arrays["metadata"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_feature_state(path: Path, index: int) -> tuple[dict[str, Any], Any]:
    """Load one saved T1 boundary into the minimal object accepted by the feature API."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
        identity = metadata[index]
        state = {name: archive[f"s{index}_{name}"].copy() for name in _STATE_ARRAYS}
    sim = SimpleNamespace(
        E=1,
        S=int(identity["slots"]),
        cap=int(identity["cap"]),
        grid_w=int(identity["grid_w"]),
        grid_h=int(identity["grid_h"]),
        cfg=SimpleNamespace(
            min_boost_length=int(identity["min_boost_length"]),
            boost_length_cost_frames=int(identity["boost_length_cost_frames"]),
        ),
        **state,
    )
    return identity, sim


def _measure_arm(
    arm: str, rows: Iterable[dict[str, Any]], context: Any
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, np.ndarray]]]]:
    """Replay one B8 arm once, observing every requested T1 LAD boundary."""
    requested = list(rows)
    by_target = {int(row["env"]): row for row in requested}
    if len(by_target) != len(requested):
        raise ValueError(f"duplicate T1 environment in {arm}")
    envs = sorted(by_target)
    _data, series = load_inputs(arm)
    replay = build_replay(arm, envs, context)
    locations = {env: index for index, env in enumerate(envs)}
    targets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in requested:
        targets[int(row["lad"]["decision"])].append(row)
    results: list[dict[str, Any]] = []
    states: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []

    for decision in range(max(targets) + 1):

        def observe(current: Any) -> None:
            if decision not in targets:
                return
            for row in targets[decision]:
                env = int(row["env"])
                local = locations[env]
                lad = row["lad"]
                if int(current.sim.frame[local]) != int(lad["frame"]):
                    raise AssertionError(
                        (arm, env, "frame", current.sim.frame[local], lad["frame"])
                    )
                actual_length = (
                    int(current.sim.length[local, 0]) if current.sim.alive[local, 0] else 0
                )
                if actual_length != int(lad["hero_length"]):
                    raise AssertionError((arm, env, "length", actual_length, lad["hero_length"]))
                if int(series["action"][env, decision]) != int(lad["chosen_action"]):
                    raise AssertionError((arm, env, "chosen action differs from tape"))
                if int(current.seeds[local]) != int(row["world_seed"]):
                    raise AssertionError((arm, env, "world seed differs from T1 row"))

                # The clone is an in-memory snapshot restoration.  It catches missing
                # ring/grid/RNG state before treating a feature value as a T1 value.
                restored = current.clone()
                assert_same_world(current, restored)
                state_features = compute_slot_features(current.sim, local, 0)
                restored_features = compute_slot_features(restored.sim, local, 0)
                np.testing.assert_array_equal(state_features, restored_features)

                identity = {
                    "arm": arm,
                    "env": env,
                    "world_seed": int(row["world_seed"]),
                    "decision": decision,
                    "frame": int(lad["frame"]),
                    "slots": int(current.sim.S),
                    "cap": int(current.sim.cap),
                    "grid_w": int(current.sim.grid_w),
                    "grid_h": int(current.sim.grid_h),
                    "min_boost_length": int(current.sim.cfg.min_boost_length),
                    "boost_length_cost_frames": int(current.sim.cfg.boost_length_cost_frames),
                }
                state = {
                    name: np.asarray(getattr(current.sim, name)[local : local + 1]).copy()
                    for name in _STATE_ARRAYS
                }
                state["features"] = state_features[None, :].copy()
                states.append((identity, state))

                chosen = int(lad["chosen_action"])
                best = int(lad["best_surviving_action"])
                chosen_values = _action_values(state_features, chosen)
                best_values = _action_values(state_features, best)
                higher_area = best_values["area_log"] > chosen_values["area_log"]
                higher_boost3 = best_values["boost3_area_log"] > chosen_values["boost3_area_log"]
                results.append(
                    {
                        "arm": arm,
                        "env": env,
                        "world_seed": int(row["world_seed"]),
                        "decision": decision,
                        "frame": int(lad["frame"]),
                        "chosen_action": chosen,
                        "best_surviving_action": best,
                        "surviving_actions": [int(a) for a in lad["surviving_actions"]],
                        "chosen": chosen_values,
                        "best_surviving": best_values,
                        "best_higher_area_log": higher_area,
                        "best_higher_boost3_area_log": higher_boost3,
                        "passes_either": bool(higher_area or higher_boost3),
                        "restored_matches": {
                            "deterministic_tape_action": True,
                            "frame": True,
                            "hero_length": True,
                            "world_seed": True,
                            "full_clone_state": True,
                            "features_bitwise_after_clone": True,
                        },
                    }
                )

        replay.step(series["action"][envs, decision], observe)
    if len(results) != len(requested):
        raise AssertionError((arm, "captured rows", len(results), len(requested)))
    return results, states


def run(rows_path: Path | None = None, state_out: Path | None = None) -> dict[str, Any]:
    """Return the T1 feature comparison receipt, using one CPU thread."""
    configure_cpu()
    started = time.perf_counter()
    path = rows_path or T1_ARTIFACTS / "rows.jsonl"
    rows = _late_lads(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    context = census.load_context()
    evidence = []
    feature_states = []
    for arm in sorted(grouped):
        arm_evidence, arm_states = _measure_arm(arm, grouped[arm], context)
        evidence.extend(arm_evidence)
        feature_states.extend(arm_states)
    evidence.sort(key=lambda item: (item["arm"], item["env"], item["decision"]))
    passed = sum(item["passes_either"] for item in evidence)
    count = len(evidence)
    if state_out is not None:
        _save_feature_states(state_out, feature_states)
        for index in range(len(feature_states)):
            _identity, saved = load_feature_state(state_out, index)
            with np.load(state_out, allow_pickle=False) as archive:
                expected = archive[f"s{index}_features"]
            np.testing.assert_array_equal(compute_slot_features(saved, 0, 0)[None, :], expected)
    return {
        "schema": "t5-t1-area-check/v1",
        "authority": "development diagnostic, not gate evidence",
        "status": "COMPLETE",
        "validation_status": "PASS",
        "check_verdict": "PASS" if passed / count >= 0.70 else "FAIL",
        "source": str(path),
        "comparison": (
            "T1 best_surviving_action has strictly higher area_log or boost3_area_log "
            "than T1 chosen_action"
        ),
        "bar": {"minimum_share": 0.70, "met": passed / count >= 0.70},
        "late_lads": count,
        "higher_either": passed,
        "share": passed / count,
        "rows": evidence,
        "feature_state_store": str(state_out) if state_out is not None else None,
        "feature_state_count": len(feature_states),
        "wall_seconds": time.perf_counter() - started,
        "execution": "deterministic T1 replay; CPU only; one process; no training; no numba port",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=T1_ARTIFACTS / "rows.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "t1_check.json")
    parser.add_argument("--states-out", type=Path, default=DEFAULT_OUT / "t1_feature_states.npz")
    args = parser.parse_args()
    receipt = run(args.rows, args.states_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: receipt[k] for k in ("status", "late_lads", "higher_either", "share", "bar")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
