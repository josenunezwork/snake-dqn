#!/usr/bin/env python3
"""Deterministic, CPU-only reproductions for raster serving contract defects."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Make the worktree root importable when this nested script is executed by path.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.core.device_manager import DeviceManager
from src.game.ai_snake import AISnake
from src.model.obs_spec import RASTER31V2, RASTER31V2_SHAPES
from src.model.raster_network import RasterDuelingNetwork
from web.backend.session import GameSession


SOURCE_FILES = (
    "web/backend/raster_policy.py",
    "web/backend/session.py",
    "src/game/game_state.py",
    "src/simd_env/featurizer.py",
    "src/training/pqn_trainer.py",
)


def _source_hashes() -> dict[str, str]:
    """Hash the source files whose behavior these probes exercise."""
    return {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def _write_raster_checkpoint(path: Path, mechanics_version: int) -> None:
    """Write a disposable, structurally valid raster checkpoint."""
    torch.manual_seed(1729)
    network = RasterDuelingNetwork()
    payload = {
        "dqn_state_dict": network.state_dict(),
        "obs_spec": RASTER31V2,
        "output_size": 6,
        "mechanics_version": mechanics_version,
        "reward_version": mechanics_version,
        **RASTER31V2_SHAPES.to_metadata(),
    }
    torch.save(payload, path)


def _play_dispatch_probe(checkpoint: Path) -> dict[str, object]:
    """Show that the first AI consumes the human row in raster Play mode."""
    session = GameSession(checkpoint=str(checkpoint))
    session.set_mode("play")
    policy = session.policy
    policy._ensure_frame()

    roster = [
        {
            "type": type(snake).__name__,
            "id": int(snake.id),
            "alive": bool(snake.is_alive),
            "cache_row": int(policy._cache_id_to_row[int(snake.id)]),
        }
        for snake in session.game.snakes
    ]
    queue_before = list(policy._dispatch_queue)
    first_ai = next(snake for snake in session.game.snakes if isinstance(snake, AISnake))
    expected_row = int(policy._cache_id_to_row[int(first_ai.id)])

    # Give every row a distinct, deterministic argmax. This separates row identity
    # from random network initialization and makes the failure unambiguous.
    row_q = np.zeros((len(roster), 6), dtype=np.float32)
    for row in range(len(roster)):
        row_q[row, row % 6] = 10.0
    policy._cache_q = row_q
    returned = policy._next_dispatch_q(torch.zeros((1, 61), dtype=torch.float32))
    actual_argmax = int(returned.argmax().item())
    expected_argmax = expected_row % 6

    return {
        "roster": roster,
        "dispatch_queue_before_first_ai_call": queue_before,
        "dispatch_queue_after_first_ai_call": list(policy._dispatch_queue),
        "first_ai_id": int(first_ai.id),
        "expected_ai_row": expected_row,
        "expected_argmax": expected_argmax,
        "actual_argmax": actual_argmax,
        "reproduced_foreign_row": actual_argmax != expected_argmax,
    }


def _mechanics_probe(checkpoint: Path) -> dict[str, object]:
    """Show that serving ignores a raster checkpoint's mechanics version."""
    checkpoint_blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    session = GameSession(checkpoint=str(checkpoint))
    state = session.control_state()
    recorded = int(checkpoint_blob["mechanics_version"])
    served = int(state["mechanics_version"])
    return {
        "checkpoint_mechanics_version": recorded,
        "served_mechanics_version": served,
        "served_config": state["config"],
        "session_error": state["error"],
        "reproduced_silent_mismatch": recorded != served and state["error"] is None,
    }


def _serialized_mask_probe(checkpoint: Path) -> dict[str, object]:
    """Show that hero_observation advertises all-true instead of the live mask."""
    session = GameSession(checkpoint=str(checkpoint))
    hero = session.game.snakes[0]
    hero.segments = [(10, 100), (20, 100), (30, 100), (40, 100), (50, 100)]
    hero.length = 5
    hero.direction = (-1, 0)
    session.policy._invalidate_cache()

    shown = np.asarray(session.policy.hero_observation(hero.id)["mask"], dtype=bool)
    safe_actions = hero._get_safe_actions(session.game.snakes, allow_fallback=False)
    actual = np.asarray([action in safe_actions for action in range(6)], dtype=bool)
    return {
        "hero_id": int(hero.id),
        "hero_head": [int(v) for v in hero.head],
        "hero_direction": [int(v) for v in hero.direction],
        "safe_actions": [int(v) for v in safe_actions],
        "serialized_raster_mask": shown.tolist(),
        "actual_live_safe_mask": actual.tolist(),
        "mismatched_action_indices": np.flatnonzero(shown != actual).astype(int).tolist(),
        "reproduced_mask_mismatch": not np.array_equal(shown, actual),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    np.random.seed(1729)
    DeviceManager.override_device(torch.device("cpu"))
    try:
        test_log = REPO_ROOT / "runs/state_system_audit_20260906/serving/targeted_tests.log"
        with tempfile.TemporaryDirectory(prefix="snake-serving-probe-") as temp_dir:
            checkpoint = Path(temp_dir) / "mechanics1_raster.pth"
            _write_raster_checkpoint(checkpoint, mechanics_version=1)
            result = {
                "schema_version": 1,
                "purpose": "Reproduce raster web serving identity and contract mismatches",
                "reproduction_command": (
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                    "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python "
                    "runs/state_system_audit_20260906/serving/serving_contract_probe.py "
                    "--output runs/state_system_audit_20260906/serving/serving_contract_probe.json"
                ),
                "targeted_test_receipt": {
                    "command": (
                        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                        "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q "
                        "tests/test_web_raster_serving.py tests/test_obs_parity.py "
                        "tests/test_inference_agent.py tests/test_inference_agent_raster.py "
                        "tests/test_web_checkpoints.py"
                    ),
                    "path": str(test_log.relative_to(REPO_ROOT)),
                    "sha256": hashlib.sha256(test_log.read_bytes()).hexdigest()
                    if test_log.exists()
                    else None,
                },
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
                "device": "cpu",
                "torch_threads": int(torch.get_num_threads()),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "seed": 1729,
                "synthetic_checkpoint_retained": False,
                "source_sha256": _source_hashes(),
                "play_dispatch": _play_dispatch_probe(checkpoint),
                "checkpoint_mechanics": _mechanics_probe(checkpoint),
                "serialized_action_mask": _serialized_mask_probe(checkpoint),
            }
            result["all_three_reproduced"] = all(
                (
                    result["play_dispatch"]["reproduced_foreign_row"],
                    result["checkpoint_mechanics"]["reproduced_silent_mismatch"],
                    result["serialized_action_mask"]["reproduced_mask_mismatch"],
                )
            )
            result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    finally:
        DeviceManager.reset_for_testing()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_three_reproduced": result["all_three_reproduced"]}))
    return 0 if result["all_three_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
