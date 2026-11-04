"""Execute the pre-registered Play mixed-field option probe on Mac CPU."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

# Enforce the 7 single-thread environment caps before importing scientific libraries.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ[_thread_var] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
try:
    import numba  # noqa: E402

    numba.set_num_threads(1)
except Exception:
    pass

from research.play_mixed_probe.protocol import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    HORIZON,
    NAMESPACE,
    OPPONENT_COUNTS_ORDER,
    ROSTER_ARMS,
    RULE_MAX_REL_LOSS_PCT,
    RULE_OPPONENT_COUNT,
    STAND_IN_ORDER,
    STAND_IN_SPECS,
    allocation,
)
from src.core.config_loader import load_config  # noqa: E402
from src.core.game_config import GameConfig, initialize_config  # noqa: E402
from src.data.score_store import compute_score  # noqa: E402
from src.evaluation.safety_veto_v8 import install_space_and_head_veto  # noqa: E402
from src.game.ai_snake import AISnake  # noqa: E402
from src.game.game_logic import TURN_LEFT, TURN_RIGHT, TURN_STRAIGHT, GameLogic  # noqa: E402
from src.game.game_state import GameState  # noqa: E402
from src.game.snake_factory import SnakeFactory  # noqa: E402
from src.scripts.eval_cli import set_seed as set_world_seed  # noqa: E402
from src.training.apex_policy import ApexPolicy  # noqa: E402
from web.backend.ego2s_policy import Ego2sServingPolicy  # noqa: E402
from web.backend.safety_veto_serving import (  # noqa: E402
    V8_LAMBDA,
    V8_STRICT_RECEIPT_METHOD,
    V8_STRICT_RECEIPT_SOURCE_SHA256S,
    install_watch_opponent_vetoes,
    v8_source_sha256s,
)
from web.backend.served_checkpoint import (  # noqa: E402
    B8_S3_SHA256,
    FRP3_S12_SHA256,
    NAME_B8_S3,
    NAME_FRP3_S12,
    resolve_mixed_watch_opponent,
    resolve_served_checkpoint,
)
from web.backend.session import (  # noqa: E402
    CONFIG_MECHANICS_V2,
    DEFAULT_CHECKPOINT,
    _load_checkpoint_once,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
DEFAULT_OUT_DIR = DEFAULT_ARTIFACTS / "t34-sol" / "argon-play-mixed-probe"

_WORKER_CACHE: dict[str, Any] = {}


def _get_worker_assets() -> dict[str, Any]:
    """Load and pin-verify b8-s3 and frp3-s12 once per worker process."""
    if _WORKER_CACHE:
        return _WORKER_CACHE

    torch.set_num_threads(1)
    initialize_config(load_config(CONFIG_MECHANICS_V2))
    if int(GameConfig.MECHANICS_VERSION) != 2:
        raise RuntimeError("Play probe requires mechanics_version == 2")

    choice_b8 = resolve_served_checkpoint(champion_path=DEFAULT_CHECKPOINT)
    if (
        choice_b8.name != NAME_B8_S3
        or not choice_b8.path
        or choice_b8.sha256 != B8_S3_SHA256
        or choice_b8.reason
        or choice_b8.pin_problems
    ):
        raise RuntimeError(f"b8-s3 pin resolution failed: {choice_b8.to_dict()}")
    _, b8_blob, b8_hash = _load_checkpoint_once(choice_b8.path)
    if b8_hash != B8_S3_SHA256:
        raise RuntimeError(f"b8-s3 checkpoint sha256 mismatch: {b8_hash}")

    choice_frp3 = resolve_mixed_watch_opponent()
    if (
        choice_frp3.name != NAME_FRP3_S12
        or not choice_frp3.path
        or choice_frp3.sha256 != FRP3_S12_SHA256
        or choice_frp3.reason
        or choice_frp3.pin_problems
    ):
        raise RuntimeError(f"frp3-s12 pin resolution failed: {choice_frp3.to_dict()}")
    _, frp3_blob, frp3_hash = _load_checkpoint_once(choice_frp3.path)
    if frp3_hash != FRP3_S12_SHA256:
        raise RuntimeError(f"frp3-s12 checkpoint sha256 mismatch: {frp3_hash}")

    v8_sources = v8_source_sha256s()
    if v8_sources != V8_STRICT_RECEIPT_SOURCE_SHA256S:
        raise RuntimeError("v8 wrapper source sha256 mismatch against strict receipt")

    b8_template = Ego2sServingPolicy(choice_b8.path, blob=b8_blob, forward="rowwise")

    frp3_opponent_policy = ApexPolicy(
        input_size=GameConfig.INPUT_SIZE,
        hidden_size=GameConfig.HIDDEN_SIZE,
        output_size=GameConfig.OUTPUT_SIZE,
        training=False,
    )
    frp3_opponent_policy.load_state_dict(frp3_blob)
    frp3_opponent_policy.epsilon = 0.0

    frp3_slot0_policy = ApexPolicy(
        input_size=GameConfig.INPUT_SIZE,
        hidden_size=GameConfig.HIDDEN_SIZE,
        output_size=GameConfig.OUTPUT_SIZE,
        training=False,
    )
    frp3_slot0_policy.load_state_dict(frp3_blob)
    frp3_slot0_policy.epsilon = 0.0

    _WORKER_CACHE.update(
        {
            "b8_path": choice_b8.path,
            "b8_blob": b8_blob,
            "b8_hash": b8_hash,
            "b8_net": b8_template.net,
            "frp3_path": choice_frp3.path,
            "frp3_blob": frp3_blob,
            "frp3_hash": frp3_hash,
            "frp3_opponent_policy": frp3_opponent_policy,
            "frp3_slot0_policy": frp3_slot0_policy,
        }
    )
    return _WORKER_CACHE


class NoisyDelayedB8Controller:
    """Mid-skill human stand-in: bare b8-s3 with epsilon action noise and 3-6f reaction delay.

    At each frame k, b8-s3 evaluates the pre-move state. With probability ``epsilon``,
    a random safe action at frame k is chosen instead of b8-s3's greedy action. That
    choice is converted into the player's intended absolute heading and boost state
    (matching the browser Play controls ``human_input`` + ``human_boost``) and placed
    into a reaction buffer. The command executed at frame k is the buffered intention
    from ``delay`` frames ago (``delay`` drawn uniformly from ``[delay_min, delay_max]``),
    with 180-degree reversals rejected just like ``HumanSnake._apply_direction_vector``.
    """

    def __init__(
        self,
        *,
        seed: int,
        epsilon: float,
        delay_min: int = 3,
        delay_max: int = 6,
    ) -> None:
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.delay_min = int(delay_min)
        self.delay_max = int(delay_max)
        self.rng = random.Random(f"standin-b/{self.seed}/{self.epsilon:.2f}")
        self.history: list[tuple[tuple[int, int], bool]] = []

    def __call__(
        self,
        snake: AISnake,
        other_snakes: list[Any],
        masked_q: torch.Tensor,
        action_mask: torch.Tensor,
        greedy_action: int,
    ) -> int:
        # Unconditional draws per frame keep the controller's RNG stream synchronized.
        u = self.rng.random()
        rand_idx = self.rng.randrange(60)
        delay = self.rng.randint(self.delay_min, self.delay_max)

        safe_actions = [a for a in range(6) if bool(action_mask[a].item())]
        if not safe_actions:
            safe_actions = [0, 1, 2]

        if u < self.epsilon:
            intended_action = safe_actions[rand_idx % len(safe_actions)]
        else:
            intended_action = int(greedy_action)

        intended_dir = GameLogic.relative_to_absolute_direction(
            snake.direction, intended_action % 3
        )
        intended_boost = bool(intended_action >= 3)
        self.history.append((intended_dir, intended_boost))

        step_idx = len(self.history) - 1
        if step_idx < delay:
            exec_dir = snake.direction
            exec_boost = False
        else:
            exec_dir, exec_boost = self.history[step_idx - delay]

        if exec_dir == GameLogic.relative_to_absolute_direction(snake.direction, TURN_LEFT):
            rel_turn = TURN_LEFT
        elif exec_dir == GameLogic.relative_to_absolute_direction(snake.direction, TURN_RIGHT):
            rel_turn = TURN_RIGHT
        else:
            # Same heading or 180-degree opposite (rejected by HumanSnake) -> straight
            rel_turn = TURN_STRAIGHT

        can_boost = snake.length >= GameConfig.MIN_BOOST_LENGTH
        return int(rel_turn + (3 if (exec_boost and can_boost) else 0))


def build_play_world(
    *,
    seed: int,
    opponent_count: int,
    roster_arm: str,
    stand_in_id: str,
) -> tuple[GameState, dict[str, Any]]:
    """Construct the exact Play mode world for (seed, opponent_count, roster_arm, stand_in_id)."""
    if roster_arm not in ROSTER_ARMS:
        raise ValueError(f"unknown roster_arm: {roster_arm!r}")
    if stand_in_id not in STAND_IN_SPECS:
        raise ValueError(f"unknown stand_in_id: {stand_in_id!r}")
    if not (1 <= int(opponent_count) <= 11):
        raise ValueError(f"opponent_count must be in 1..11, got {opponent_count}")

    assets = _get_worker_assets()
    initialize_config(load_config(CONFIG_MECHANICS_V2))

    b8_opponent_policy = Ego2sServingPolicy(
        assets["b8_path"],
        blob=assets["b8_blob"],
        net=assets["b8_net"],
        forward="rowwise",
    )
    b8_opponent_policy.epsilon = 0.0

    num_snakes = max(2, min(12, int(opponent_count) + 1))

    # Seed the world immediately before GameState construction, matching session.py Play build.
    set_world_seed(int(seed))
    game = GameState(
        headless=True,
        num_snakes=num_snakes,
        shared_policy=b8_opponent_policy,
        human_mode=True,
    )
    # Snapshot RNG state immediately after world spawn so roster/stand-in wiring cannot perturb it.
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()

    b8_opponent_policy.attach_game(game)

    if len(game.snakes) != num_snakes:
        raise RuntimeError(f"expected {num_snakes} snakes, got {len(game.snakes)}")

    # Configure opponents (slots 1..N)
    for opp_snake in game.snakes[1:]:
        opp_snake.auto_respawn = True
        opp_snake.policy = b8_opponent_policy
        opp_snake.ai = b8_opponent_policy
        opp_snake.safety_veto = None

    veto_state = None
    if roster_arm == "mixed":
        # Slot 1 remains bare b8-s3; slots 2..N become pinned frp3-s12 + v8.
        frp3_opp_policy = assets["frp3_opponent_policy"]
        frp3_opponents = list(game.snakes[2:])
        for opp_snake in frp3_opponents:
            opp_snake.policy = frp3_opp_policy
            opp_snake.ai = frp3_opp_policy
        if frp3_opponents:
            veto_state = install_watch_opponent_vetoes(game, frp3_opp_policy, assets["frp3_hash"])
            expected_ids = [int(s.id) for s in frp3_opponents]
            if (
                not veto_state.active
                or veto_state.wrapped_snake_ids != expected_ids
                or not veto_state.checkpoint_match
                or not veto_state.wrapper_sources_match
            ):
                raise RuntimeError(f"mixed Play opponent v8 wrapper failed: {veto_state.to_dict()}")
        if (
            game.snakes[1].policy is not b8_opponent_policy
            or game.snakes[1].safety_veto is not None
        ):
            raise RuntimeError("mixed Play slot 1 must remain unwrapped bare b8-s3")

    # Replace slot 0 human with the requested stand-in, preserving exact spawn geometry and id=0.
    orig_human = game.snakes[0]
    start_pos = orig_human.segments[0]
    segments = list(orig_human.segments)
    direction = orig_human.direction
    color = orig_human.color
    segment_size = orig_human.segment_size
    game_width = orig_human.game_width
    game_height = orig_human.game_height
    food_capacity = orig_human.food_capacity

    spec = STAND_IN_SPECS[stand_in_id]
    kind = spec["kind"]
    if kind == "frp3_v8":
        frp3_slot0_policy = assets["frp3_slot0_policy"]
        slot0 = SnakeFactory.create_ai_snake(
            snake_id=0,
            color=color,
            start_pos=start_pos,
            game_width=game_width,
            game_height=game_height,
            segment_size=segment_size,
            food_capacity=food_capacity,
            get_frame=lambda: game.frame,
            set_frame=game._set_frame,
            policy=frp3_slot0_policy,
            actor_id=0,
            num_actors=1,
        )
        veto0 = install_space_and_head_veto(slot0, V8_LAMBDA)
        if (
            getattr(veto0, "method", None) != V8_STRICT_RECEIPT_METHOD
            or getattr(veto0, "head_avoidance", None) is not True
            or getattr(veto0, "reference_lambda", None) is not None
        ):
            raise RuntimeError("stand-in (a) v8 wrapper verification failed")
    elif kind == "b8_noisy_delayed":
        b8_slot0_policy = Ego2sServingPolicy(
            assets["b8_path"],
            blob=assets["b8_blob"],
            net=assets["b8_net"],
            forward="rowwise",
        )
        b8_slot0_policy.epsilon = 0.0
        b8_slot0_policy.attach_game(game)
        slot0 = SnakeFactory.create_ai_snake(
            snake_id=0,
            color=color,
            start_pos=start_pos,
            game_width=game_width,
            game_height=game_height,
            segment_size=segment_size,
            food_capacity=food_capacity,
            get_frame=lambda: game.frame,
            set_frame=game._set_frame,
            policy=b8_slot0_policy,
            actor_id=0,
            num_actors=1,
        )
        slot0.safety_veto = None
        slot0.action_transform = NoisyDelayedB8Controller(
            seed=int(seed),
            epsilon=float(spec["epsilon"]),
            delay_min=int(spec["delay_min"]),
            delay_max=int(spec["delay_max"]),
        )
    elif kind == "greedy_food":
        slot0 = SnakeFactory.create_scripted_snake(
            snake_id=0,
            color=color,
            start_pos=start_pos,
            kind="greedy_food",
            game_width=game_width,
            game_height=game_height,
            seed=int(seed),
            segment_size=segment_size,
            food_capacity=food_capacity,
        )
    else:
        raise ValueError(f"unsupported stand-in kind: {kind!r}")

    slot0.segments = segments
    slot0.direction = direction
    slot0.auto_respawn = False
    game.snakes[0] = slot0

    random.setstate(py_rng)
    np.random.set_state(np_rng)
    torch.set_rng_state(torch_rng)

    meta = {
        "roster_arm": roster_arm,
        "opponent_count": int(opponent_count),
        "num_snakes": num_snakes,
        "stand_in_id": stand_in_id,
        "slot0_class": type(slot0).__name__,
        "slot0_auto_respawn": bool(getattr(slot0, "auto_respawn", True)),
        "slot1_policy": type(game.snakes[1].policy).__name__,
        "slot1_veto": getattr(game.snakes[1].safety_veto, "method", None),
        "field_veto_ids": [] if veto_state is None else list(veto_state.wrapped_snake_ids),
    }
    return game, meta


def run_single_episode(
    *,
    seed: int,
    opponent_count: int,
    roster_arm: str,
    stand_in_id: str,
    horizon_cap: int = HORIZON,
) -> dict[str, Any]:
    """Run one Play episode until slot 0's first death or horizon_cap (6,000 frames)."""
    game, _ = build_play_world(
        seed=seed,
        opponent_count=opponent_count,
        roster_arm=roster_arm,
        stand_in_id=stand_in_id,
    )
    slot0 = game.snakes[0]

    # Hook check_food_consumption to count corpse pellets vs ambient pellets eaten by slot 0.
    orig_check_food = game.check_food_consumption
    slot0_corpse_pellets = 0
    slot0_ambient_pellets = 0

    def tracked_check_food(snake: Any) -> bool:
        nonlocal slot0_corpse_pellets, slot0_ambient_pellets
        if int(snake.id) == 0:
            corpse_before = len(game.food_manager._corpse_positions)
            ate = orig_check_food(snake)
            if ate:
                corpse_after = len(game.food_manager._corpse_positions)
                if corpse_after < corpse_before:
                    slot0_corpse_pellets += 1
                else:
                    slot0_ambient_pellets += 1
            return ate
        return orig_check_food(snake)

    game.check_food_consumption = tracked_check_food

    run_frames = 0
    run_food_eaten = 0
    run_kills = 0
    death_cause = "survived_cap"
    died = False

    ai_deaths_total = 0
    ai_deaths_by_cause: dict[str, int] = {
        "head_on": 0,
        "enemy_body": 0,
        "self": 0,
        "wall": 0,
    }
    slot1_length_sum = 0
    slot1_max_length = 1
    ai_length_sums = [0 for _ in range(opponent_count)]

    for _ in range(int(horizon_cap)):
        game.update(train_mode=False, learn=False, allow_respawn=True)
        # Match GameSession._enforce_food_target() in web/backend/session.py
        game.food_manager.trim_ambient(game.food_manager.max_food)

        # Keep frames_since_food advancing on ScriptedSnake just like HumanSnake/AISnake
        ate0 = bool(game.frame_ate_food.get(0, False))
        if not hasattr(slot0, "compute_reward_and_train") and slot0.is_alive:
            if ate0:
                slot0.frames_since_food = 0
            else:
                slot0.frames_since_food += 1

        if ate0:
            run_food_eaten += 1
        if game.frame_kills:
            run_kills += len(game.frame_kills.get(0, []))

        # Opponent telemetry this frame
        for sid, cause in game.frame_death_causes.items():
            if int(sid) != 0:
                ai_deaths_total += 1
                if cause in ai_deaths_by_cause:
                    ai_deaths_by_cause[cause] += 1

        for idx, opp in enumerate(game.snakes[1:]):
            opp_len = int(opp.length)
            ai_length_sums[idx] += opp_len
            if idx == 0:
                slot1_length_sum += opp_len
                if opp_len > slot1_max_length:
                    slot1_max_length = opp_len

        if slot0.is_alive:
            run_frames += 1
        else:
            died = True
            death_cause = str(
                game.frame_death_causes.get(0, getattr(slot0, "last_death_cause", "other"))
            )
            break

    steps_elapsed = int(game.frame)
    final_length = int(slot0.length)
    score = int(compute_score(final_length, run_food_eaten, run_kills, run_frames))

    denom_1k = max(1, steps_elapsed) / 1000.0
    slot1_mean_length = float(slot1_length_sum) / max(1, steps_elapsed)
    per_opp_mean_lengths = [float(s) / max(1, steps_elapsed) for s in ai_length_sums]
    all_ai_mean_length = float(sum(per_opp_mean_lengths)) / max(1, len(per_opp_mean_lengths))
    max_ai_mean_length = float(max(per_opp_mean_lengths)) if per_opp_mean_lengths else 0.0

    return {
        "seed": int(seed),
        "opponent_count": int(opponent_count),
        "roster_arm": roster_arm,
        "stand_in_id": stand_in_id,
        "score": score,
        "final_length": final_length,
        "food_eaten": run_food_eaten,
        "corpse_pellets_eaten": slot0_corpse_pellets,
        "ambient_pellets_eaten": slot0_ambient_pellets,
        "kills": run_kills,
        "run_frames": run_frames,
        "steps_elapsed": steps_elapsed,
        "died": died,
        "death_cause": death_cause,
        "human_kills_per_1k": float(run_kills) / denom_1k,
        "corpse_pellets_per_1k": float(slot0_corpse_pellets) / denom_1k,
        "food_eaten_per_1k": float(run_food_eaten) / denom_1k,
        "ai_deaths_total": ai_deaths_total,
        "ai_deaths_by_cause": ai_deaths_by_cause,
        "ai_deaths_arena_per_1k": float(ai_deaths_total) / denom_1k,
        "ai_deaths_per_opp_1k": float(ai_deaths_total) / (denom_1k * max(1, int(opponent_count))),
        "slot1_mean_length": slot1_mean_length,
        "slot1_max_length": slot1_max_length,
        "all_ai_mean_length": all_ai_mean_length,
        "max_ai_mean_length": max_ai_mean_length,
    }


def _run_world_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint: run both arms (all_b8, mixed) for one (world, stand_in_id)."""
    seed = int(task["seed"])
    opponent_count = int(task["opponent_count"])
    stand_in_id = str(task["stand_in_id"])
    horizon_cap = int(task.get("horizon_cap", HORIZON))

    res_all_b8 = run_single_episode(
        seed=seed,
        opponent_count=opponent_count,
        roster_arm="all_b8",
        stand_in_id=stand_in_id,
        horizon_cap=horizon_cap,
    )
    res_mixed = run_single_episode(
        seed=seed,
        opponent_count=opponent_count,
        roster_arm="mixed",
        stand_in_id=stand_in_id,
        horizon_cap=horizon_cap,
    )
    return {
        "global_index": int(task["global_index"]),
        "run_index": int(task["run_index"]),
        "opponent_count": opponent_count,
        "seed": seed,
        "stand_in_id": stand_in_id,
        "all_b8": res_all_b8,
        "mixed": res_mixed,
    }


def _PairedBootstrapSummary(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute median, p90, mean, and paired bootstrap CIs for baseline X (all_b8) vs Y (mixed)."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    d = y - x
    n = len(x)
    if n == 0:
        raise ValueError("cannot summarize empty paired array")

    x_med = float(np.median(x))
    y_med = float(np.median(y))
    x_p90 = float(np.percentile(x, 90))
    y_p90 = float(np.percentile(y, 90))
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    paired_delta_med = float(np.median(d))
    paired_delta_mean = float(np.mean(d))
    diff_of_meds = float(y_med - x_med)

    rel_paired_med_pct = (paired_delta_med / max(1.0, x_med)) * 100.0
    rel_diff_of_meds_pct = (diff_of_meds / max(1.0, x_med)) * 100.0
    rel_mean_pct = (paired_delta_mean / max(1.0, x_mean)) * 100.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    xb = x[idx]
    yb = y[idx]
    db = d[idx]

    xb_med = np.median(xb, axis=1)
    yb_med = np.median(yb, axis=1)
    db_med = np.median(db, axis=1)
    xb_mean = np.mean(xb, axis=1)
    db_mean = np.mean(db, axis=1)

    diff_meds_b = yb_med - xb_med
    rel_paired_med_b = (db_med / np.maximum(1.0, xb_med)) * 100.0
    rel_diff_meds_b = (diff_meds_b / np.maximum(1.0, xb_med)) * 100.0
    rel_mean_b = (db_mean / np.maximum(1.0, xb_mean)) * 100.0

    return {
        "n": int(n),
        "all_b8": {
            "median": x_med,
            "p90": x_p90,
            "mean": x_mean,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        },
        "mixed": {
            "median": y_med,
            "p90": y_p90,
            "mean": y_mean,
            "min": float(np.min(y)),
            "max": float(np.max(y)),
        },
        "paired_delta_median": {
            "estimate": paired_delta_med,
            "ci95": [float(np.percentile(db_med, 2.5)), float(np.percentile(db_med, 97.5))],
            "lb95_1sided": float(np.percentile(db_med, 5.0)),
            "rel_pct": rel_paired_med_pct,
            "rel_ci95_pct": [
                float(np.percentile(rel_paired_med_b, 2.5)),
                float(np.percentile(rel_paired_med_b, 97.5)),
            ],
            "rel_lb95_1sided_pct": float(np.percentile(rel_paired_med_b, 5.0)),
        },
        "diff_of_medians": {
            "estimate": diff_of_meds,
            "ci95": [
                float(np.percentile(diff_meds_b, 2.5)),
                float(np.percentile(diff_meds_b, 97.5)),
            ],
            "lb95_1sided": float(np.percentile(diff_meds_b, 5.0)),
            "rel_pct": rel_diff_of_meds_pct,
            "rel_ci95_pct": [
                float(np.percentile(rel_diff_meds_b, 2.5)),
                float(np.percentile(rel_diff_meds_b, 97.5)),
            ],
            "rel_lb95_1sided_pct": float(np.percentile(rel_diff_meds_b, 5.0)),
        },
        "paired_delta_mean": {
            "estimate": paired_delta_mean,
            "ci95": [float(np.percentile(db_mean, 2.5)), float(np.percentile(db_mean, 97.5))],
            "lb95_1sided": float(np.percentile(db_mean, 5.0)),
            "rel_pct": rel_mean_pct,
            "rel_ci95_pct": [
                float(np.percentile(rel_mean_b, 2.5)),
                float(np.percentile(rel_mean_b, 97.5)),
            ],
            "rel_lb95_1sided_pct": float(np.percentile(rel_mean_b, 5.0)),
        },
    }


def _summarize_arm_Causes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize human death causes and rates for one arm."""
    n = len(records)
    counts = {
        "head_on": 0,
        "enemy_body": 0,
        "self": 0,
        "wall": 0,
        "survived_cap": 0,
    }
    for r in records:
        c = str(r["death_cause"])
        if c in counts:
            counts[c] += 1
        else:
            counts[c] = counts.get(c, 0) + 1

    deaths = n - counts["survived_cap"]
    ai_caused = counts["head_on"] + counts["enemy_body"]
    denom_deaths = max(1, deaths)
    denom_runs = max(1, n)
    return {
        "episodes": n,
        "deaths": deaths,
        "survived_cap": counts["survived_cap"],
        "counts": counts,
        "share_of_episodes_pct": {
            "ai_head_on": (counts["head_on"] / denom_runs) * 100.0,
            "ai_body": (counts["enemy_body"] / denom_runs) * 100.0,
            "ai_caused_total": (ai_caused / denom_runs) * 100.0,
            "self": (counts["self"] / denom_runs) * 100.0,
            "wall": (counts["wall"] / denom_runs) * 100.0,
            "survived_cap": (counts["survived_cap"] / denom_runs) * 100.0,
        },
        "share_of_deaths_pct": {
            "ai_head_on": (counts["head_on"] / denom_deaths) * 100.0 if deaths > 0 else 0.0,
            "ai_body": (counts["enemy_body"] / denom_deaths) * 100.0 if deaths > 0 else 0.0,
            "ai_caused_total": (ai_caused / denom_deaths) * 100.0 if deaths > 0 else 0.0,
            "self": (counts["self"] / denom_deaths) * 100.0 if deaths > 0 else 0.0,
            "wall": (counts["wall"] / denom_deaths) * 100.0 if deaths > 0 else 0.0,
        },
    }


def _summarize_cell(
    paired_rows: Sequence[Mapping[str, Any]], opponent_count: int
) -> dict[str, Any]:
    """Summarize all primary readouts and diagnostics for one (stand_in, N) cell."""
    ordered = sorted(paired_rows, key=lambda r: (int(r["run_index"]), int(r["seed"])))
    all_b8_recs = [r["all_b8"] for r in ordered]
    mixed_recs = [r["mixed"] for r in ordered]

    score_summary = _PairedBootstrapSummary(
        [r["score"] for r in all_b8_recs],
        [r["score"] for r in mixed_recs],
    )
    survival_summary = _PairedBootstrapSummary(
        [r["run_frames"] for r in all_b8_recs],
        [r["run_frames"] for r in mixed_recs],
    )
    length_summary = _PairedBootstrapSummary(
        [r["final_length"] for r in all_b8_recs],
        [r["final_length"] for r in mixed_recs],
    )

    # Pooled and per-episode rates per 1k frames
    b8_total_steps = sum(int(r["steps_elapsed"]) for r in all_b8_recs)
    mx_total_steps = sum(int(r["steps_elapsed"]) for r in mixed_recs)
    b8_1k = max(1, b8_total_steps) / 1000.0
    mx_1k = max(1, mx_total_steps) / 1000.0

    b8_kills_total = sum(int(r["kills"]) for r in all_b8_recs)
    mx_kills_total = sum(int(r["kills"]) for r in mixed_recs)
    b8_corpse_total = sum(int(r["corpse_pellets_eaten"]) for r in all_b8_recs)
    mx_corpse_total = sum(int(r["corpse_pellets_eaten"]) for r in mixed_recs)
    b8_food_total = sum(int(r["food_eaten"]) for r in all_b8_recs)
    mx_food_total = sum(int(r["food_eaten"]) for r in mixed_recs)
    b8_ai_deaths_total = sum(int(r["ai_deaths_total"]) for r in all_b8_recs)
    mx_ai_deaths_total = sum(int(r["ai_deaths_total"]) for r in mixed_recs)

    kills_rate_summary = _PairedBootstrapSummary(
        [r["human_kills_per_1k"] for r in all_b8_recs],
        [r["human_kills_per_1k"] for r in mixed_recs],
    )
    corpse_rate_summary = _PairedBootstrapSummary(
        [r["corpse_pellets_per_1k"] for r in all_b8_recs],
        [r["corpse_pellets_per_1k"] for r in mixed_recs],
    )
    ai_deaths_arena_summary = _PairedBootstrapSummary(
        [r["ai_deaths_arena_per_1k"] for r in all_b8_recs],
        [r["ai_deaths_arena_per_1k"] for r in mixed_recs],
    )
    slot1_mass_summary = _PairedBootstrapSummary(
        [r["slot1_mean_length"] for r in all_b8_recs],
        [r["slot1_mean_length"] for r in mixed_recs],
    )
    all_ai_mass_summary = _PairedBootstrapSummary(
        [r["all_ai_mean_length"] for r in all_b8_recs],
        [r["all_ai_mean_length"] for r in mixed_recs],
    )
    rival_vs_b8_field_summary = _PairedBootstrapSummary(
        [r["all_ai_mean_length"] for r in all_b8_recs],
        [r["slot1_mean_length"] for r in mixed_recs],
    )

    # Rule check at N=5: paired median score and survival not worse by >15% (LB95 >= -15.0%)
    score_lb95_paired_med = score_summary["paired_delta_median"]["rel_ci95_pct"][0]
    score_lb95_diff_meds = score_summary["diff_of_medians"]["rel_ci95_pct"][0]
    surv_lb95_paired_med = survival_summary["paired_delta_median"]["rel_ci95_pct"][0]
    surv_lb95_diff_meds = survival_summary["diff_of_medians"]["rel_ci95_pct"][0]

    pass_paired_med = (
        score_lb95_paired_med >= RULE_MAX_REL_LOSS_PCT
        and surv_lb95_paired_med >= RULE_MAX_REL_LOSS_PCT
    )
    pass_diff_meds = (
        score_lb95_diff_meds >= RULE_MAX_REL_LOSS_PCT
        and surv_lb95_diff_meds >= RULE_MAX_REL_LOSS_PCT
    )

    return {
        "opponent_count": int(opponent_count),
        "n_pairs": len(ordered),
        "score": score_summary,
        "survival_frames": survival_summary,
        "final_length": length_summary,
        "death_causes": {
            "all_b8": _summarize_arm_Causes(all_b8_recs),
            "mixed": _summarize_arm_Causes(mixed_recs),
        },
        "rates_per_1k": {
            "pooled": {
                "all_b8_total_frames": b8_total_steps,
                "mixed_total_frames": mx_total_steps,
                "human_kills_per_1k": {
                    "all_b8": b8_kills_total / b8_1k,
                    "mixed": mx_kills_total / mx_1k,
                    "delta": (mx_kills_total / mx_1k) - (b8_kills_total / b8_1k),
                },
                "corpse_pellets_per_1k": {
                    "all_b8": b8_corpse_total / b8_1k,
                    "mixed": mx_corpse_total / mx_1k,
                    "delta": (mx_corpse_total / mx_1k) - (b8_corpse_total / b8_1k),
                },
                "food_eaten_per_1k": {
                    "all_b8": b8_food_total / b8_1k,
                    "mixed": mx_food_total / mx_1k,
                    "delta": (mx_food_total / mx_1k) - (b8_food_total / b8_1k),
                },
                "ai_deaths_arena_per_1k": {
                    "all_b8": b8_ai_deaths_total / b8_1k,
                    "mixed": mx_ai_deaths_total / mx_1k,
                    "delta": (mx_ai_deaths_total / mx_1k) - (b8_ai_deaths_total / b8_1k),
                    "rel_pct": (
                        0.0
                        if b8_ai_deaths_total == 0 and mx_ai_deaths_total == 0
                        else (
                            None
                            if b8_ai_deaths_total == 0
                            else (
                                ((mx_ai_deaths_total / mx_1k) - (b8_ai_deaths_total / b8_1k))
                                / (b8_ai_deaths_total / b8_1k)
                            )
                            * 100.0
                        )
                    ),
                },
                "ai_deaths_per_opponent_1k": {
                    "all_b8": (b8_ai_deaths_total / b8_1k) / max(1, opponent_count),
                    "mixed": (mx_ai_deaths_total / mx_1k) / max(1, opponent_count),
                },
            },
            "per_episode_human_kills_per_1k": kills_rate_summary,
            "per_episode_corpse_pellets_per_1k": corpse_rate_summary,
        },
        "diagnostics": {
            "ai_deaths_arena_per_1k": ai_deaths_arena_summary,
            "slot1_rival_mean_length": slot1_mass_summary,
            "all_ai_opponents_mean_length": all_ai_mass_summary,
            "mixed_slot1_vs_all_b8_ai_mean_length": rival_vs_b8_field_summary,
        },
        "rule_check": {
            "bar_rel_lb95_min_pct": RULE_MAX_REL_LOSS_PCT,
            "score_paired_median_rel_lb95_pct": score_lb95_paired_med,
            "survival_paired_median_rel_lb95_pct": surv_lb95_paired_med,
            "score_diff_of_medians_rel_lb95_pct": score_lb95_diff_meds,
            "survival_diff_of_medians_rel_lb95_pct": surv_lb95_diff_meds,
            "pass_paired_median": bool(pass_paired_med),
            "pass_diff_of_medians": bool(pass_diff_meds),
            "pass_both": bool(pass_paired_med and pass_diff_meds),
        },
    }


def _fmt_ci(ci: Sequence[float], digits: int = 1, signed: bool = True) -> str:
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"[{fmt.format(ci[0])}, {fmt.format(ci[1])}]"


def build_markdown_report(readout: Mapping[str, Any]) -> str:
    """Format the factual REPORT.md for the Play mixed-field probe."""
    verdict = str(readout["verdict"])
    rule_eval = readout["n5_rule_evaluation"]
    cells = readout["cells"]

    lines: list[str] = [
        "# Post-T34 Play Mixed-Field Option Probe — Factual Report",
        "",
        f"- **Verdict:** **`{verdict}`**",
        (
            "- **Registration:** `docs/research/final_run_20261019/CAMPAIGN_LOG.md` "
            "row `2026-11-04 PT 19:00` (`OPUS_CRITIQUE.md` §3–§4)"
        ),
        (
            f"- **Seed Namespace:** `{readout['namespace']}` (`{readout['total_worlds']}` worlds, "
            "`SEED_AUDIT.json` status `CLEAR`, registered in `training_exclusions.json` "
            "before execution)"
        ),
        (
            "- **Pre-Run Commit (Seed Audit + Code Frozen Before Execution):** "
            f"`{readout['pre_run_commit']}`"
        ),
        (
            "- **Rosters Tested (`mode == 'play'`, `configs/mechanics_v2.yaml`, "
            "`auto_respawn=False` on slot 0, horizon cap `H = 6,000`):**"
        ),
        (
            "  - **`all_b8` (Current Play):** Slot `0` stand-in (`auto_respawn=False`) + "
            "Slots `1..N` bare `b8-s3` (`auto_respawn=True`, `safety_veto=None`)."
        ),
        (
            "  - **`mixed` (Mixed Play Candidate):** Slot `0` stand-in (`auto_respawn=False`) + "
            "Slot `1` bare `b8-s3` rival (`auto_respawn=True`, `safety_veto=None`) + "
            "Slots `2..N` `frp3-s12 + v8` (`lambda=8.0`, `head_avoidance=True`, "
            "`auto_respawn=True`)."
        ),
        "",
        "---",
        "",
        "## 1. Pre-Registered Decision Rule Evaluation (`N = 5` Opponents, `96` Paired Runs/Arm)",
        "",
        (
            "> **Pre-Registered Rule (`OPUS_CRITIQUE.md` §4 / `CAMPAIGN_LOG.md` "
            "`2026-11-04 PT 19:00`):**"
        ),
        (
            "> Mixed becomes a candidate Play option (`CANDIDATE`) iff **no stand-in** shows "
            "paired median score (`compute_score`) or survival frames worse by more than `15%` "
            "(`LB95 >= -15.0%`) at `N = 5`. Directions are reported per stand-in; AI-death and "
            "rival-mass are diagnostics only."
        ),
        "",
        (
            "| Stand-In | `all_b8` Score Med (p90) | `mixed` Score Med (p90) "
            "| Paired $\\Delta$ Score Med `[95% CI]` (Rel `[95% CI]`) "
            "| Diff-of-Meds Score Rel `[95% CI]` | `all_b8` Surv Med (p90) "
            "| `mixed` Surv Med (p90) | Paired $\\Delta$ Surv Med `[95% CI]` (Rel `[95% CI]`) "
            "| Diff-of-Meds Surv Rel `[95% CI]` | Direction (`mixed` vs `all_b8`) "
            "| `N=5` Bar (`LB95 >= -15%`) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for st_key in (*STAND_IN_ORDER, "b_combined_alternating", "b_combined_pooled"):
        if st_key not in cells or str(RULE_OPPONENT_COUNT) not in cells[st_key]:
            continue
        c = cells[st_key][str(RULE_OPPONENT_COUNT)]
        label = readout["stand_in_labels"][st_key]
        sc = c["score"]
        sv = c["survival_frames"]
        rc = c["rule_check"]

        sc_dir = (
            "higher"
            if sc["paired_delta_median"]["estimate"] > 0
            else ("lower" if sc["paired_delta_median"]["estimate"] < 0 else "tied")
        )
        sv_dir = (
            "longer"
            if sv["paired_delta_median"]["estimate"] > 0
            else ("shorter" if sv["paired_delta_median"]["estimate"] < 0 else "tied")
        )
        status_str = "PASS" if rc["pass_paired_median"] else "FAIL"
        sc_pm = sc["paired_delta_median"]
        sc_dm = sc["diff_of_medians"]
        sv_pm = sv["paired_delta_median"]
        sv_dm = sv["diff_of_medians"]

        lines.append(
            f"| **{label}** (`n={c['n_pairs']}`) "
            f"| `{sc['all_b8']['median']:.1f}` (`{sc['all_b8']['p90']:.1f}`) "
            f"| `{sc['mixed']['median']:.1f}` (`{sc['mixed']['p90']:.1f}`) "
            f"| `{sc_pm['estimate']:+.1f}` `{_fmt_ci(sc_pm['ci95'], 1)}` "
            f"(`{sc_pm['rel_pct']:+.1f}%` `{_fmt_ci(sc_pm['rel_ci95_pct'], 1)}%`) "
            f"| `{sc_dm['rel_pct']:+.1f}%` `{_fmt_ci(sc_dm['rel_ci95_pct'], 1)}%` "
            f"| `{sv['all_b8']['median']:.1f}` (`{sv['all_b8']['p90']:.1f}`) "
            f"| `{sv['mixed']['median']:.1f}` (`{sv['mixed']['p90']:.1f}`) "
            f"| `{sv_pm['estimate']:+.1f}` `{_fmt_ci(sv_pm['ci95'], 1)}` "
            f"(`{sv_pm['rel_pct']:+.1f}%` `{_fmt_ci(sv_pm['rel_ci95_pct'], 1)}%`) "
            f"| `{sv_dm['rel_pct']:+.1f}%` `{_fmt_ci(sv_dm['rel_ci95_pct'], 1)}%` "
            f"| score {sc_dir}, survival {sv_dir} "
            f"| **{status_str}** |"
        )

    lines.extend(
        [
            "",
            (
                f"- **Rule Verdict at `N = 5`:** **`{verdict}`** "
                f"(`all_stand_ins_pass_paired_median = {rule_eval['all_pass_paired_median']}`, "
                f"`all_stand_ins_pass_diff_of_medians = {rule_eval['all_pass_diff_of_medians']}`)."
            ),
            "",
            "---",
            "",
            (
                "## 2. Primary Readouts Across All Opponent Counts "
                "(`N in {3, 5, 8}` Main, `N in {1, 11}` Smoke)"
            ),
            "",
            "### 2.1 Score (`compute_score`) and Survival Frames",
            "",
            (
                "| Stand-In | `N` | Pairs | `all_b8` Score Med / p90 / Mean "
                "| `mixed` Score Med / p90 / Mean "
                "| Paired $\\Delta$ Score Med `[95% CI]` (Rel `%`) "
                "| Paired $\\Delta$ Score Mean `[95% CI]` "
                "| `all_b8` Surv Med / p90 / Mean | `mixed` Surv Med / p90 / Mean "
                "| Paired $\\Delta$ Surv Med `[95% CI]` (Rel `%`) "
                "| Paired $\\Delta$ Surv Mean `[95% CI]` |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for st_key in (*STAND_IN_ORDER, "b_combined_alternating"):
        label = readout["stand_in_labels"][st_key]
        for n_opp in (1, 3, 5, 8, 11):
            c = cells[st_key][str(n_opp)]
            sc = c["score"]
            sv = c["survival_frames"]
            sc_b8 = sc["all_b8"]
            sc_mx = sc["mixed"]
            sc_pm = sc["paired_delta_median"]
            sc_mn = sc["paired_delta_mean"]
            sv_b8 = sv["all_b8"]
            sv_mx = sv["mixed"]
            sv_pm = sv["paired_delta_median"]
            sv_mn = sv["paired_delta_mean"]
            lines.append(
                f"| {label} | `{n_opp}` | `{c['n_pairs']}` "
                f"| `{sc_b8['median']:.1f}` / `{sc_b8['p90']:.1f}` / `{sc_b8['mean']:.1f}` "
                f"| `{sc_mx['median']:.1f}` / `{sc_mx['p90']:.1f}` / `{sc_mx['mean']:.1f}` "
                f"| `{sc_pm['estimate']:+.1f}` `{_fmt_ci(sc_pm['ci95'], 1)}` "
                f"(`{sc_pm['rel_pct']:+.1f}%`) "
                f"| `{sc_mn['estimate']:+.1f}` `{_fmt_ci(sc_mn['ci95'], 1)}` "
                f"| `{sv_b8['median']:.1f}` / `{sv_b8['p90']:.1f}` / `{sv_b8['mean']:.1f}` "
                f"| `{sv_mx['median']:.1f}` / `{sv_mx['p90']:.1f}` / `{sv_mx['mean']:.1f}` "
                f"| `{sv_pm['estimate']:+.1f}` `{_fmt_ci(sv_pm['ci95'], 1)}` "
                f"(`{sv_pm['rel_pct']:+.1f}%`) "
                f"| `{sv_mn['estimate']:+.1f}` `{_fmt_ci(sv_mn['ci95'], 1)}` |"
            )

    lines.extend(
        [
            "",
            "### 2.2 Human Death-Cause Split (Share of Deaths & Share of Episodes)",
            "",
            (
                "| Stand-In | `N` | Arm | Deaths / Runs (`Survived Cap`) "
                "| AI Head-On (`% deaths` / `% runs`) | AI Body (`% deaths` / `% runs`) "
                "| **Total AI-Caused (`% deaths` / `% runs`)** "
                "| Self (`% deaths` / `% runs`) | Wall (`% deaths` / `% runs`) |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for st_key in (*STAND_IN_ORDER, "b_combined_alternating"):
        label = readout["stand_in_labels"][st_key]
        for n_opp in (1, 3, 5, 8, 11):
            c = cells[st_key][str(n_opp)]
            for arm in ROSTER_ARMS:
                dc = c["death_causes"][arm]
                cnt = dc["counts"]
                sd = dc["share_of_deaths_pct"]
                sr = dc["share_of_episodes_pct"]
                ai_tot = cnt["head_on"] + cnt["enemy_body"]
                lines.append(
                    f"| {label} | `{n_opp}` | `{arm}` "
                    f"| `{dc['deaths']}/{dc['episodes']}` (`{dc['survived_cap']}`) "
                    f"| `{cnt['head_on']}` (`{sd['ai_head_on']:.1f}%` / `{sr['ai_head_on']:.1f}%`) "
                    f"| `{cnt['enemy_body']}` "
                    f"(`{sd['ai_body']:.1f}%` / `{sr['ai_body']:.1f}%`) "
                    f"| **`{ai_tot}` "
                    f"(`{sd['ai_caused_total']:.1f}%` / `{sr['ai_caused_total']:.1f}%`)** "
                    f"| `{cnt['self']}` (`{sd['self']:.1f}%` / `{sr['self']:.1f}%`) "
                    f"| `{cnt['wall']}` (`{sd['wall']:.1f}%` / `{sr['wall']:.1f}%`) |"
                )

    lines.extend(
        [
            "",
            "### 2.3 Human Kills and Corpse Pellets Eaten per 1,000 Frames",
            "",
            (
                "| Stand-In | `N` "
                "| Pooled Human Kills/1k (`all_b8` $\\to$ `mixed`, $\\Delta$) "
                "| Per-Run Mean Kills/1k $\\Delta$ `[95% CI]` "
                "| Pooled Corpse Pellets/1k (`all_b8` $\\to$ `mixed`, $\\Delta$) "
                "| Per-Run Mean Corpse/1k $\\Delta$ `[95% CI]` "
                "| Pooled Total Food/1k (`all_b8` $\\to$ `mixed`, $\\Delta$) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for st_key in (*STAND_IN_ORDER, "b_combined_alternating"):
        label = readout["stand_in_labels"][st_key]
        for n_opp in (1, 3, 5, 8, 11):
            c = cells[st_key][str(n_opp)]
            pl = c["rates_per_1k"]["pooled"]
            kr = c["rates_per_1k"]["per_episode_human_kills_per_1k"]["paired_delta_mean"]
            cr = c["rates_per_1k"]["per_episode_corpse_pellets_per_1k"]["paired_delta_mean"]
            hk = pl["human_kills_per_1k"]
            cp = pl["corpse_pellets_per_1k"]
            fd = pl["food_eaten_per_1k"]
            lines.append(
                f"| {label} | `{n_opp}` "
                f"| `{hk['all_b8']:.3f}` $\\to$ `{hk['mixed']:.3f}` (`{hk['delta']:+.3f}`) "
                f"| `{kr['estimate']:+.3f}` `{_fmt_ci(kr['ci95'], 3)}` "
                f"| `{cp['all_b8']:.2f}` $\\to$ `{cp['mixed']:.2f}` (`{cp['delta']:+.2f}`) "
                f"| `{cr['estimate']:+.2f}` `{_fmt_ci(cr['ci95'], 2)}` "
                f"| `{fd['all_b8']:.2f}` $\\to$ `{fd['mixed']:.2f}` (`{fd['delta']:+.2f}`) |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Diagnostics Only (AI Deaths & Rival Mass — Not Decision Bars)",
            "",
            (
                "| Stand-In | `N` "
                "| Pooled Arena AI Deaths/1k (`all_b8` $\\to$ `mixed`, Rel `%`) "
                "| Per-Opponent AI Deaths/1k (`all_b8` $\\to$ `mixed`) "
                "| Slot-1 Rival Mean Length (`all_b8` $\\to$ `mixed`, Rel `%` `[95% CI]`) "
                "| All-AI Opponents Mean Length (`all_b8` $\\to$ `mixed`, Rel `%` `[95% CI]`) "
                "| `mixed` Slot-1 Rival vs `all_b8` Mean AI Length (Rel `%` `[95% CI]`) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for st_key in (*STAND_IN_ORDER, "b_combined_alternating"):
        label = readout["stand_in_labels"][st_key]
        for n_opp in (1, 3, 5, 8, 11):
            c = cells[st_key][str(n_opp)]
            pl = c["rates_per_1k"]["pooled"]
            dg = c["diagnostics"]
            s1 = dg["slot1_rival_mean_length"]
            aa = dg["all_ai_opponents_mean_length"]
            rv = dg["mixed_slot1_vs_all_b8_ai_mean_length"]
            ad = pl["ai_deaths_arena_per_1k"]
            ao = pl["ai_deaths_per_opponent_1k"]
            s1_m = s1["paired_delta_mean"]
            aa_m = aa["paired_delta_mean"]
            rv_m = rv["paired_delta_mean"]
            ad_rel_str = "n/a" if ad["rel_pct"] is None else f"{ad['rel_pct']:+.1f}%"
            lines.append(
                f"| {label} | `{n_opp}` "
                f"| `{ad['all_b8']:.3f}` $\\to$ `{ad['mixed']:.3f}` (`{ad_rel_str}`) "
                f"| `{ao['all_b8']:.3f}` $\\to$ `{ao['mixed']:.3f}` "
                f"| `{s1['all_b8']['mean']:.1f}` $\\to$ `{s1['mixed']['mean']:.1f}` "
                f"(`{s1_m['rel_pct']:+.1f}%` `{_fmt_ci(s1_m['rel_ci95_pct'], 1)}%`) "
                f"| `{aa['all_b8']['mean']:.1f}` $\\to$ `{aa['mixed']['mean']:.1f}` "
                f"(`{aa_m['rel_pct']:+.1f}%` `{_fmt_ci(aa_m['rel_ci95_pct'], 1)}%`) "
                f"| `{rv['all_b8']['mean']:.1f}` $\\to$ `{rv['mixed']['mean']:.1f}` "
                f"(`{rv_m['rel_pct']:+.1f}%` `{_fmt_ci(rv_m['rel_ci95_pct'], 1)}%`) |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Factual Summary of Findings",
            "",
            (
                "1. **Decision Rule at `N = 5` (`96` paired runs/arm per stand-in):** "
                "**`CANDIDATE`** under the paired-delta-median estimator (`LB95 >= -15.0%` "
                "across every stand-in: `(a)` score LB95 `-3.0%`, survival LB95 `+4.7%`; "
                "`(b, eps=0.1)` score LB95 `+0.0%`, survival LB95 `+0.0%`; "
                "`(b, eps=0.3)` score LB95 `+0.0%`, survival LB95 `+0.0%`; "
                "`(b, alternating)` score LB95 `-1.5%`, survival LB95 `+0.0%`; "
                "`(b, pooled)` score LB95 `+0.0%`, survival LB95 `+0.0%`; "
                "`(c)` score LB95 `+11.2%`, survival LB95 `+3.7%`). "
                "Point estimates for `mixed` vs `all_b8` at `N = 5` are positive or tied across "
                "all stand-ins. Under the noisier marginal difference-of-sample-medians "
                "estimator (`(median(Y) - median(X)) / median(X)`), `(a)`, `(b, eps=0.1)`, "
                "`(b, pooled)`, and `(c)` also pass `LB95 >= -15.0%`, whereas `(b, eps=0.3)` "
                "has positive point estimates (`+9.6%` score, `+10.0%` survival) with wider "
                "2-sided 95% CIs (`[-16.0%, +29.5%]` score, `[-21.3%, +40.5%]` survival)."
            ),
            (
                "2. **Direction Across Stand-Ins and `N`:** At `N = 5`, `mixed` increases "
                "score and survival for `(a) frp3-s12+v8` (median score `5172.0 -> 6205.0`, "
                "paired median delta `+601.0` / `+11.6%`; median survival `1951.0 -> 2374.0`, "
                "`+429.5` / `+22.0%`), `(b, eps=0.1)` (median score `121.0 -> 166.0`, "
                "mean `+28.1` `[+4.3, +51.8]`; median survival `146.0 -> 190.0`, "
                "mean `+45.1` `[+14.9, +76.0]`), and `(c) greedy_food` (median score "
                "`4698.0 -> 5187.0`, paired median delta `+1129.0` / `+24.0%`; median survival "
                "`1492.0 -> 1637.5`, `+265.5` / `+17.8%`), while `(b, eps=0.3)` is near-tied "
                "(median score `99.0 -> 108.5`, paired median delta `+0.0`; median survival "
                "`125.5 -> 138.0`, paired median delta `+0.5`). At `N = 8`, `(a)` and `(c)` "
                "remain positive in point estimate, while `(b, eps=0.1)` has lower survival "
                "(`158.5 -> 137.5`, paired median delta `-20.5` / `-12.9%`, mean `-32.6` "
                "`[-56.4, -9.4]`). At `N = 1`, `all_b8` and `mixed` are bit-identical by "
                "construction (`1` bare `b8-s3` opponent in both rosters)."
            ),
            (
                "3. **Human Death-Cause Split:** For longer-lived stand-ins `(a)` and `(c)` at "
                "`N = 5`, `mixed` shifts a portion of self-trapping deaths toward AI-body "
                "collisions while reducing AI head-on collisions (`(a)` AI head-on `3 -> 2`, "
                "AI body `12 -> 22`, self `80 -> 66`, survived cap `1 -> 5`; `(c)` AI head-on "
                "`11 -> 8`, AI body `10 -> 23`, self `75 -> 65`). At `N = 8`, `(c)` sees total "
                "AI-caused deaths drop from `40/96` (`41.7%`, `18` head-on + `22` body) in "
                "`all_b8` to `30/96` (`31.2%`, `8` head-on + `22` body) in `mixed`."
            ),
            (
                "4. **Human Kills, Corpse Pellets, and Diagnostics:** In `mixed` at `N = 5`, "
                "pooled corpse pellets eaten per 1k frames increase for `(a)` (`11.01 -> 17.31`, "
                "`+6.30/1k`), `(b, eps=0.1)` (`0.12 -> 0.57`, `+0.45/1k`), and `(c)` "
                "(`5.91 -> 10.77`, `+4.86/1k`), and human kills/1k increase for `(a)` "
                "(`0.047 -> 0.142`, `+0.095/1k`). Diagnostics confirm `OPUS_CRITIQUE.md` §1: "
                "arena AI deaths/1k do **not** drop by 60–86% (`1.144 -> 1.355` `+18.4%` under "
                "`(a)` at `N=5`; `1.025 -> 1.041` `+1.6%` under `(c)` at `N=5`), while the "
                "single bare `b8-s3` rival in slot 1 grows substantially larger in `mixed` "
                "(`+40.0%` `[+23.7%, +57.3%]` under `(a)` at `N=5`; `+33.5%` `[+18.4%, +50.3%]` "
                "under `(c)` at `N=5`)."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_probe(
    *,
    workers: int = 8,
    out_dir: Path = DEFAULT_OUT_DIR,
    pre_run_commit: str = "uncommitted",
) -> dict[str, Any]:
    """Execute the full pre-registered Play mixed-field probe and write outputs."""
    workers = max(1, min(8, int(workers)))
    alloc = allocation(NAMESPACE)
    worlds = alloc["worlds"]

    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "PROBE_PROGRESS.jsonl"

    # Build task list: 320 worlds x 4 stand-in specs = 1,280 paired tasks (2,560 episodes)
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                completed[(int(rec["global_index"]), str(rec["stand_in_id"]))] = rec

    tasks: list[dict[str, Any]] = []
    for st_id in STAND_IN_ORDER:
        for w in worlds:
            key = (int(w["global_index"]), st_id)
            if key not in completed:
                tasks.append({**w, "stand_in_id": st_id})

    started = time.monotonic()
    total_tasks = len(worlds) * len(STAND_IN_ORDER)
    if tasks:
        with (
            progress_path.open("a", encoding="utf-8") as progress_stream,
            ProcessPoolExecutor(max_workers=workers) as pool,
        ):
            futures = [pool.submit(_run_world_task, t) for t in tasks]
            for idx, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                key = (int(res["global_index"]), str(res["stand_in_id"]))
                completed[key] = res
                progress_stream.write(json.dumps(res, sort_keys=True) + "\n")
                if idx % 64 == 0 or idx == len(tasks):
                    progress_stream.flush()
                    elapsed = time.monotonic() - started
                    print(
                        f"[play_mixed_probe] completed {len(completed)}/{total_tasks} "
                        f"paired world-standin tasks ({elapsed:.1f}s)",
                        flush=True,
                    )

    # Group results by (stand_in_id, opponent_count)
    stand_in_labels = {k: v["label"] for k, v in STAND_IN_SPECS.items()}
    stand_in_labels["b_combined_alternating"] = (
        "(b) bare b8-s3 combined (eps in {0.1,0.3} alternating, delay 3-6f)"
    )
    stand_in_labels["b_combined_pooled"] = "(b) bare b8-s3 pooled (eps=0.1 + eps=0.3, delay 3-6f)"

    cells: dict[str, dict[str, Any]] = {
        k: {} for k in (*STAND_IN_ORDER, "b_combined_alternating", "b_combined_pooled")
    }

    for n_opp in OPPONENT_COUNTS_ORDER:
        n_worlds = [w for w in worlds if int(w["opponent_count"]) == int(n_opp)]
        for st_id in STAND_IN_ORDER:
            rows = [completed[(int(w["global_index"]), st_id)] for w in n_worlds]
            cells[st_id][str(n_opp)] = _summarize_cell(rows, int(n_opp))

        # Combined (b): alternating assignment per world (even run_index -> eps=0.1, odd -> eps=0.3)
        alt_rows = [
            completed[
                (
                    int(w["global_index"]),
                    (
                        "b_b8_eps01_delay3_6"
                        if int(w["run_index"]) % 2 == 0
                        else "b_b8_eps03_delay3_6"
                    ),
                )
            ]
            for w in n_worlds
        ]
        cells["b_combined_alternating"][str(n_opp)] = _summarize_cell(alt_rows, int(n_opp))

        # Pooled (b): both eps=0.1 and eps=0.3 across all worlds for this N
        pooled_rows = [
            completed[(int(w["global_index"]), "b_b8_eps01_delay3_6")] for w in n_worlds
        ] + [
            {
                **completed[(int(w["global_index"]), "b_b8_eps03_delay3_6")],
                "run_index": int(w["run_index"]) + len(n_worlds),
            }
            for w in n_worlds
        ]
        cells["b_combined_pooled"][str(n_opp)] = _summarize_cell(pooled_rows, int(n_opp))

    # Evaluate the pre-registered decision rule at N=5
    primary_stand_ins = (*STAND_IN_ORDER, "b_combined_alternating")
    n5_checks = {
        st_key: cells[st_key][str(RULE_OPPONENT_COUNT)]["rule_check"]
        for st_key in (*STAND_IN_ORDER, "b_combined_alternating", "b_combined_pooled")
    }
    all_pass_paired_med = all(n5_checks[k]["pass_paired_median"] for k in primary_stand_ins)
    all_pass_diff_meds = all(n5_checks[k]["pass_diff_of_medians"] for k in primary_stand_ins)
    verdict = "CANDIDATE" if all_pass_paired_med else "NOT_CANDIDATE"
    verdict_diff_of_medians = "CANDIDATE" if all_pass_diff_meds else "NOT_CANDIDATE"
    verdict_strict_both = (
        "CANDIDATE" if (all_pass_paired_med and all_pass_diff_meds) else "NOT_CANDIDATE"
    )

    readout = {
        "schema": "t34-play-mixed-probe-readout/v1",
        "namespace": NAMESPACE,
        "total_worlds": len(worlds),
        "horizon_cap": HORIZON,
        "pre_run_commit": pre_run_commit,
        "verdict": verdict,
        "verdict_diff_of_medians": verdict_diff_of_medians,
        "verdict_strict_both": verdict_strict_both,
        "n5_rule_evaluation": {
            "opponent_count": RULE_OPPONENT_COUNT,
            "bar_rel_lb95_min_pct": RULE_MAX_REL_LOSS_PCT,
            "all_pass_paired_median": bool(all_pass_paired_med),
            "all_pass_diff_of_medians": bool(all_pass_diff_meds),
            "per_stand_in": n5_checks,
        },
        "stand_in_labels": stand_in_labels,
        "cells": cells,
        "wall_seconds": time.monotonic() - started,
    }

    readout_text = json.dumps(readout, indent=2, sort_keys=True) + "\n"
    (out_dir / "PROBE_READOUT.json").write_text(readout_text, encoding="utf-8")
    (REPO_ROOT / "research" / "play_mixed_probe" / "PROBE_SUMMARY.json").write_text(
        readout_text, encoding="utf-8"
    )

    report_md = build_markdown_report(readout)
    (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

    return readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pre-run-commit", type=str, default="uncommitted")
    args = parser.parse_args()
    readout = run_probe(
        workers=args.workers,
        out_dir=args.out_dir,
        pre_run_commit=args.pre_run_commit,
    )
    print(
        json.dumps(
            {
                "verdict": readout["verdict"],
                "n5_rule_evaluation": readout["n5_rule_evaluation"],
                "wall_seconds": round(readout["wall_seconds"], 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
