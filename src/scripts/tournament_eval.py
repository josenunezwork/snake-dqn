#!/usr/bin/env python3
"""Promotion gate: paired candidate-vs-baseline eval over diverse opponent mixes.

Repaired per blueprint §5.1-§5.2 (P0). What changed vs the old harness:

  * Headline metric is the MASS INTEGRAL: mean per-frame mass over the TOTAL
    episode horizon, dead frames contributing 0 — replacing the alive-
    conditioned mean_mass under which dying rich outranked surviving.
  * PAIRED analysis: every candidate plays the same seeds as a designated
    --baseline (default: the incumbent champion); per-seed deltas of the mass
    integral get a t-distribution 95% CI and wins/N (math shared with
    ensemble_eval via src/scripts/eval_stats.py).
  * OPPONENT DIVERSITY: instead of 5 clones of one checkpoint, the candidate
    is evaluated round-robin over opponent MIXES:
       frozen    slots cycle over the --opponents checkpoint pool
       scripted  all slots are greedy_food scripted anchors (ungameable)
       mixed     alternating frozen-pool and random_safe scripted slots
  * PROMOTION RULE (printed, and enforced with --gate): promote iff the paired
    mass-integral delta > 0 at 95% CI on >= 2 mixes AND no regression vs the
    scripted anchor mix.
  * --pilot runs the baseline only and recommends a seed count for a minimum
    detectable effect of 3% of baseline mass integral (alpha 0.05, power 0.8).
  * Behavioral probes (src/training/behavior_probes.py) run alongside: boost
    fraction, death causes, food eaten, kill opportunities, entrapment events.

Agents (candidates, --baseline, --opponents entries) are checkpoint paths or
scripted stand-ins ``scripted:greedy_food`` / ``scripted:random_safe`` — the
latter need no checkpoint, which is how the gate calibrates itself
(champion > greedy anchor > random_safe) and how tests run hermetically.

Usage:
  # Gate a candidate against the incumbent champion on the default mixes
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py \
    saved_snakes/latest_apex.pth --gate \
    --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9 \
    --json-output logs/gate_latest.json

  # Pilot: how many seeds do we need?
  ./venv/bin/python src/scripts/tournament_eval.py --pilot --frames 3000

  # Calibration: the champion must beat the scripted anchor
  ./venv/bin/python src/scripts/tournament_eval.py \
    saved_snakes/champion_a5_freespace_20260621.pth \
    --baseline scripted:greedy_food --mixes scripted,mixed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import load_and_initialize_config  # noqa: E402
from src.core.game_config import GameConfig, get_config  # noqa: E402
from src.core.runtime_contract import (  # noqa: E402
    ActionMaskSet,
    EffectiveWorldConfig,
    RuntimeModeContract,
    canonical_digest,
)
from src.evaluation.anchors import (  # noqa: E402
    SCRIPTED_ANCHOR_VERSION,
    AnchorContext,
    ScriptedAnchor,
)
from src.evaluation.artifacts import EvaluationArtifacts, SnapshotError  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    EvaluationMetricsAccumulator,
    PostStepState,
    StepEvents,
)
from src.evaluation.protocol import (  # noqa: E402
    PROMOTION_V2_WATCH_RECT,
    EvaluationProfile,
    legacy_diagnostic_profile,
    promotion_v2_watch_rect,
)
from src.game.game_state_factory import (  # noqa: E402
    configure_eval_game_state,
    create_training_game_state,
)
from src.game.scripted_snake import SCRIPTED_KINDS  # noqa: E402
from src.game.snake_factory import SnakeFactory  # noqa: E402
from src.scripts.eval_cli import parse_seed_list, set_seed  # noqa: E402
from src.scripts.eval_stats import (  # noqa: E402
    ci95_halfwidth,
    mass_integral,
    mean,
    paired_stats,
    recommended_seed_count,
    sample_std,
)
from src.training.behavior_probes import BehaviorProbes  # noqa: E402

# Backward-compat alias: ensemble_eval imports ci95 from this module. The stats
# now live in eval_stats (and are t-based rather than the old normal approx).
ci95 = ci95_halfwidth

DEFAULT_CONFIG = "configs/eval_free_space.yaml"
DEFAULT_BASELINE = "saved_snakes/champion_a5_freespace_20260621.pth"
# The frozen 61-D pool (58-D checkpoints need widen_input.py first; see
# configs/eval_free_space.yaml header).
DEFAULT_OPPONENTS = (
    "saved_snakes/champion_a5_freespace_20260621.pth",
    "saved_snakes/best_apex_fs.pth",
    "saved_snakes/best_apex_stage1_fs.pth",
    "saved_snakes/best_apex_pre_fs.pth",
)
MIX_NAMES = ("frozen", "scripted", "mixed")
SCRIPTED_PREFIX = "scripted:"
SCRIPTED_ANCHOR_MIX = "scripted"
# Per-seed metric keys reported for every rollout.
METRIC_KEYS = ("mass_integral", "max_mass", "kills", "deaths", "survival_fraction")

# (kind, ref) agent spec: ("checkpoint", path) or ("scripted", scripted kind).
AgentSpec = Tuple[str, str]


def materialize_opponent_specs_by_world(
    opponent_specs: Sequence[AgentSpec], seeds: Sequence[int]
) -> Dict[int, List[AgentSpec]]:
    """Freeze an ordered opponent roster per world before paired evaluation."""
    roster = [(str(kind), str(reference)) for kind, reference in opponent_specs]
    return {int(seed): list(roster) for seed in seeds}


def _slot_content_hash(kind: str, reference: str) -> str:
    """Return stable content identity without exposing snapshot-local paths."""
    if kind == "checkpoint":
        digest = hashlib.sha256()
        with Path(reference).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    return canonical_digest(
        {"scripted_anchor": reference, "anchor_version": SCRIPTED_ANCHOR_VERSION}
    )


def _roster_identity(
    seed: int, specs: Sequence[AgentSpec], mix_id: str = "unspecified"
) -> Dict[str, Any]:
    hashes = [_slot_content_hash(kind, ref) for kind, ref in specs]
    return {
        "seed_namespace": "evaluation-world/v1",
        "seed": int(seed),
        "mix_id": str(mix_id),
        "ordered_slot_content_hashes": hashes,
        "roster_id": canonical_digest({"slots": hashes}),
    }


def _evaluation_world_from_config() -> EffectiveWorldConfig:
    """Materialize the complete live-world identity from the resolved config."""
    config = GameConfig
    pqn = get_config().pqn
    return EffectiveWorldConfig(
        width=int(config.WIDTH),
        height=int(config.HEIGHT),
        segment_size=int(config.SEGMENT_SIZE),
        wall_thickness=int(config.WALL_THICKNESS),
        arena_type=str(config.ARENA_TYPE),
        mechanics_version=int(config.MECHANICS_VERSION),
        num_snakes=int(config.NUM_SNAKES),
        max_frames=int(config.MAX_FRAMES),
        initial_food=int(config.INITIAL_FOOD),
        max_food=int(config.MAX_FOOD),
        min_boost_length=int(config.MIN_BOOST_LENGTH),
        boost_length_cost_frames=int(config.BOOST_LENGTH_COST_FRAMES),
        frame_rate=int(config.FRAME_RATE),
        max_length=int(config.MAX_LENGTH),
        starvation_max_frames=int(config.STARVATION_MAX_FRAMES),
        # BatchSim capacity is an implementation/storage limit. It is not the
        # serving max-length normalizer; the corrected PQN default is 400 when
        # YAML does not explicitly select a different capacity.
        max_capacity=int(pqn.max_capacity if pqn.max_capacity is not None else 400),
        kill_scale=float(pqn.kill_scale if pqn.kill_scale is not None else 0.3),
        death_value=float(pqn.death_value if pqn.death_value is not None else -3.0),
        normalization={
            "max_frames": float(config.MAX_FRAMES),
            "starvation_max": float(config.STARVATION_MAX_FRAMES),
            "max_length": float(config.MAX_LENGTH),
        },
    )


def evaluation_profile_for_name(name: str, frames: int) -> EvaluationProfile:
    """Return an explicit profile, retaining legacy diagnostics as the default."""
    world = _evaluation_world_from_config()
    if name == PROMOTION_V2_WATCH_RECT:
        profile = promotion_v2_watch_rect(world)
        if frames != profile.scored_horizon:
            raise ValueError("promotion-v2-watch-rect requires --frames 5000")
        return profile
    return legacy_diagnostic_profile(
        world,
        RuntimeModeContract(
            mode="legacy_diagnostic",
            training=True,
            respawn=True,
            hero_terminal=True,
            population_floor=False,
            reset_strategy="legacy",
        ),
        scored_horizon=frames,
        observation_progress_horizon=int(world.normalization["max_frames"]),
    )


def parse_mix_list(value: str) -> List[str]:
    """Parse a comma-separated opponent-mix list (deduplicated, order-preserving).

    Duplicates are dropped so a repeated mix can neither double-count its
    paired deltas in the combined CI nor satisfy the ">= 2 mixes" promotion
    clause with a single opponent mix.
    """
    mixes = list(dict.fromkeys(m.strip() for m in value.split(",") if m.strip()))
    for mix in mixes:
        if mix not in MIX_NAMES:
            raise argparse.ArgumentTypeError(f"unknown mix {mix!r}; expected one of {MIX_NAMES}")
    if not mixes:
        raise argparse.ArgumentTypeError("at least one mix is required")
    return mixes


def parse_agent_spec(value: str) -> AgentSpec:
    """Parse an agent reference: a checkpoint path or ``scripted:<kind>``."""
    if value.startswith(SCRIPTED_PREFIX):
        kind = value[len(SCRIPTED_PREFIX) :]
        if kind not in SCRIPTED_KINDS:
            raise argparse.ArgumentTypeError(
                f"unknown scripted kind {kind!r}; expected one of {SCRIPTED_KINDS}"
            )
        return ("scripted", kind)
    return ("checkpoint", value)


def agent_label(spec: AgentSpec) -> str:
    """Human-readable label for an agent spec."""
    kind, ref = spec
    return f"scripted:{ref}" if kind == "scripted" else ref


def _ckpt_value(ck: dict, key, default):
    """Read a contract value from a checkpoint (top-level or apex_config/config)."""
    if key in ck:
        return ck[key]
    for sub in ("apex_config", "config"):
        d = ck.get(sub)
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


def build_policy_from_checkpoint(checkpoint_path: str):
    """Build an inference feedforward policy matching the checkpoint's input size.

    Inference (training=False) skips the TD/reward contract; the input_size shape
    check still runs. We read input_size FROM the checkpoint (not the active config)
    so a feedforward net is always built with the right width (including the 61-D
    free-space variant) instead of relying on the active config.
    """
    import torch

    from src.training.apex_policy import ApexPolicy

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden = int(_ckpt_value(ck, "hidden_size", GameConfig.HIDDEN_SIZE))
    input_size = int(_ckpt_value(ck, "input_size", GameConfig.INPUT_SIZE))

    policy = ApexPolicy(
        input_size,
        hidden,
        GameConfig.OUTPUT_SIZE,
        training=False,
    )
    policy.load_state_dict(ck)
    if hasattr(policy, "epsilon"):
        policy.epsilon = 0.0
    return policy


def checkpoint_obs_spec(checkpoint_path: str) -> str:
    """Return a checkpoint's declared observation contract.

    Legacy Apex checkpoints do not record ``obs_spec`` and therefore resolve
    to ``vector61``. This header read lets the CLI reject an unsafe live roster
    before creating a GameState or beginning a costly paired evaluation.
    """
    import torch

    from src.model.inference_agent import InferenceAgent

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return InferenceAgent._detect_obs_spec(blob)


def _checkpoint_obs_spec_or_unknown(checkpoint_path: str) -> str:
    """Read an obs spec without hiding the SIMD CLI's current error behavior."""
    try:
        return checkpoint_obs_spec(checkpoint_path)
    except Exception:
        return "unknown"


def build_mix_specs(
    mix: str, num_opponents: int, opponent_pool: Sequence[AgentSpec]
) -> List[AgentSpec]:
    """Build the per-slot opponent specs for one mix.

    Args:
        mix: One of ``MIX_NAMES``.
        num_opponents: Number of opponent slots in the arena.
        opponent_pool: The frozen pool (checkpoints or scripted stand-ins)
            cycled round-robin by the ``frozen`` and ``mixed`` mixes.

    Returns:
        One AgentSpec per opponent slot.

    Raises:
        ValueError: If a mix that needs the frozen pool gets an empty one, or
            the mix name is unknown.
    """
    if mix == "scripted":
        return [("scripted", "greedy_food")] * num_opponents
    if not opponent_pool:
        raise ValueError(f"mix {mix!r} needs a non-empty --opponents pool")
    if mix == "frozen":
        return [opponent_pool[i % len(opponent_pool)] for i in range(num_opponents)]
    if mix == "mixed":
        specs: List[AgentSpec] = []
        pool_i = 0
        for slot in range(num_opponents):
            if slot % 2 == 0:
                specs.append(opponent_pool[pool_i % len(opponent_pool)])
                pool_i += 1
            else:
                specs.append(("scripted", "random_safe"))
        return specs
    raise ValueError(f"unknown mix {mix!r}; expected one of {MIX_NAMES}")


def _attach_agent(
    gs,
    slot: int,
    spec: AgentSpec,
    seed: int,
    policy_cache: Dict[str, Any],
    profile: EvaluationProfile | None = None,
) -> None:
    """Attach an agent spec to arena slot ``slot`` (hero is slot 0).

    Checkpoint agents keep the roster's AISnake and get the frozen policy;
    scripted agents replace the roster snake in place (same id/color/spawn).

    A raster serving policy is valid only for the hero (slot 0). It computes
    every snake's raster Q rows once, then dispatches rows in ``GameState``'s
    pre-move, alive-roster order. If a raster policy were attached to an
    opponent, earlier vector snakes would not consume its dispatch queue and
    that opponent could receive another snake's Q row. The CLI rejects this
    roster before rollout; retain this guard for direct callers too.
    """
    snake = gs.snakes[slot]
    kind, ref = spec
    if kind == "checkpoint":
        from src.model.obs_spec import RASTER31V2, RASTER31V3

        obs_spec = checkpoint_obs_spec(ref)
        if obs_spec == RASTER31V2 and slot != 0:
            raise ValueError(
                "live raster checkpoints are supported only as the hero (slot 0); "
                "raster opponents are unsafe because RasterServingPolicy dispatches "
                "Q rows in pre-move roster order"
            )
        if ref not in policy_cache:
            if obs_spec == RASTER31V2:
                from src.model.inference_agent import InferenceAgent
                from web.backend.raster_policy import RasterServingPolicy

                policy_cache[ref] = RasterServingPolicy(InferenceAgent.from_checkpoint(ref))
            elif obs_spec == RASTER31V3:
                if profile is None:
                    raise ValueError("raster31v3 live evaluation requires an explicit profile")
                from src.simd_env.eval_engine import validate_v3_checkpoint_for_profile

                validate_v3_checkpoint_for_profile(ref, profile)
                normalization = {
                    key: int(value) for key, value in profile.world.normalization.items()
                }
                normalization["max_frames"] = profile.observation_progress_horizon
                policy_cache[ref] = RasterServingPolicy(
                    InferenceAgent.from_checkpoint(ref), normalization=normalization
                )
            else:
                policy_cache[ref] = build_policy_from_checkpoint(ref)
        snake.policy = policy_cache[ref]
        snake.ai = snake.policy  # AISnake's compatibility alias follows policy replacement.
        attach_game = getattr(snake.policy, "attach_game", None)
        if attach_game is not None:
            attach_game(gs)
        return
    scripted = SnakeFactory.create_scripted_snake(
        snake_id=snake.id,
        color=snake.color,
        start_pos=tuple(snake.segments[0]),
        kind=ref,
        game_width=gs._game_width,
        game_height=gs._game_height,
        seed=seed * 1000 + slot,
        food_capacity=gs._effective_max_food,
    )
    if profile is not None:
        anchor = ScriptedAnchor(ref)

        def choose_action(self, other_snakes, food):
            legal = np.zeros(6, dtype=bool)
            legal[:3] = True
            if self.length >= GameConfig.MIN_BOOST_LENGTH:
                legal[3:] = True
            advisory = np.zeros(6, dtype=bool)
            for action in self._get_safe_actions(other_snakes, allow_fallback=False):
                advisory[action] = True
            allowed = ActionMaskSet(
                legal=legal[None, :], advisory=advisory[None, :], dead=np.array([False])
            ).resolved()[0]
            return anchor.action(
                AnchorContext(
                    world_seed=seed,
                    slot=slot,
                    # GameState increments before agents act; the shared SIMD
                    # anchor observes its pre-step frame counter.
                    frame=int(gs.frame) - 1,
                    head_cell=(
                        int(self.head[0] // self.segment_size),
                        int(self.head[1] // self.segment_size),
                    ),
                    heading=tuple(int(value) for value in self.direction),
                    food_cells=tuple(
                        (int(x // self.segment_size), int(y // self.segment_size)) for x, y in food
                    ),
                    allowed_mask=allowed,
                )
            )

        scripted._choose_action = types.MethodType(choose_action, scripted)
    gs.snakes[slot] = scripted


def rollout(
    hero_spec: AgentSpec,
    opponent_specs: Sequence[AgentSpec],
    frames: int,
    seed: int,
    profile: EvaluationProfile | None = None,
    world_identity: Dict[str, Any] | None = None,
    mix_id: str = "unspecified",
) -> Dict[str, Any]:
    """One paired rollout: hero (slot 0) vs the mix's opponents.

    The hero's death is terminal (no respawn) — the mass integral integrates 0
    over its dead frames. Opponents DO respawn so arena pressure stays constant
    for the whole horizon instead of decaying as opponents die off.

    Args:
        hero_spec: Candidate or baseline agent for slot 0.
        opponent_specs: One spec per opponent slot (len == num_snakes - 1).
        frames: Total episode horizon.
        seed: World seed (paired across hero specs).

    Returns:
        Per-seed metrics: ``mass_integral``, ``max_mass``, ``mean_mass_alive``
        (legacy diagnostic), ``kills``, ``deaths``, ``survival_fraction`` and a
        ``probes`` sub-dict from BehaviorProbes.
    """
    if profile is not None and profile.legacy_diagnostic:
        raise ValueError("legacy diagnostic profiles must use the legacy evaluation path")
    set_seed(seed)
    gs = create_training_game_state(eval_mode=False)
    if len(gs.snakes) != len(opponent_specs) + 1:
        gs.full_cleanup()
        raise ValueError(
            f"arena has {len(gs.snakes)} snakes but mix defines {len(opponent_specs)} opponents"
        )

    if profile is not None and _evaluation_world_from_config().digest != profile.world.digest:
        gs.full_cleanup()
        raise ValueError("evaluation profile world does not match the configured live world")
    derived_identity = (
        _roster_identity(seed, opponent_specs, mix_id) if profile is not None else None
    )
    if world_identity is not None:
        if profile is None or world_identity != derived_identity:
            gs.full_cleanup()
            raise ValueError("world_identity does not match the materialized opponent roster")

    try:
        policy_cache: Dict[str, Any] = {}
        _attach_agent(gs, 0, hero_spec, seed, policy_cache, profile)
        for slot, spec in enumerate(opponent_specs, start=1):
            _attach_agent(gs, slot, spec, seed, policy_cache, profile)
        # Disable centralized training so no policy is ever updated.
        gs._shared_policy = None
        # Greedy, inference-only for every policy now attached to the roster.
        configure_eval_game_state(gs)

        hero = gs.snakes[0]
        hero.auto_respawn = False  # hero death is terminal; opponents keep respawning
        hero_id = hero.id

        probes = BehaviorProbes()
        accumulator = EvaluationMetricsAccumulator(profile.scored_horizon) if profile else None
        max_mass = len(hero.segments)
        mass_sum = 0.0
        alive_frames = 0
        deaths = 0
        kills = 0
        prev_alive = hero.is_alive

        for _ in range(frames):
            gs.update(
                train_mode=profile.runtime.training if profile else True,
                learn=False,
                allow_respawn=profile.runtime.respawn if profile else True,
            )
            if profile is not None:
                # The named Watch transition includes its session-side ambient-food
                # enforcement. Corpse food is intentionally retained by this helper.
                gs.food_manager.trim_ambient(profile.world.max_food)
                if any(
                    snake._logical_length() >= profile.world.max_capacity for snake in gs.snakes
                ):
                    raise RuntimeError("evaluation world exceeded its declared max_capacity")
            probes.observe(gs)
            alive = hero.is_alive
            if accumulator is not None:
                cause = gs.frame_death_causes.get(hero_id)
                accumulator.observe(
                    pre_alive=bool(prev_alive),
                    post=PostStepState(
                        alive=bool(alive), logical_mass=float(hero._logical_length())
                    ),
                    events=StepEvents(
                        food_eaten=int(bool(gs.frame_ate_food.get(hero_id, False))),
                        boost_executed=bool(prev_alive and hero.is_boosting),
                        kills=len(gs.frame_kills.get(hero_id, [])),
                        death=bool(prev_alive and not alive),
                        death_cause=cause,
                    ),
                    acted=bool(prev_alive),
                )
            if alive:
                m = len(hero.segments)
                max_mass = max(max_mass, m)
                mass_sum += m
                alive_frames += 1
            if prev_alive and not alive:
                deaths += 1
            prev_alive = alive
            kills += len(gs.frame_kills.get(hero_id, []))

        records = probes.finalize_episode()
        hero_record = next((r for r in records if r["snake_id"] == hero_id), {})
        if accumulator is not None:
            result = accumulator.result()
            result.update(
                {
                    "seed": seed,
                    "evaluation_profile": profile.descriptor(),
                    "evaluation_profile_digest": profile.digest,
                }
            )
            result["world_identity"] = derived_identity
            return result
        return {
            "seed": seed,
            "mass_integral": mass_integral(mass_sum, frames),
            "max_mass": float(max_mass),
            "mean_mass_alive": float(mass_sum / alive_frames) if alive_frames else 0.0,
            "kills": float(kills),
            "deaths": float(deaths),
            "survival_fraction": float(alive_frames / frames),
            "probes": {
                "death_cause": hero_record.get("death_cause"),
                "boost_frame_fraction": hero_record.get("boost_frame_fraction", 0.0),
                "food_eaten": hero_record.get("food_eaten", 0),
                "kill_opportunity_count": hero_record.get("kill_opportunity_count", 0),
                "entrapment_event": bool(hero_record.get("entrapment_event", False)),
                "peak_length": hero_record.get("peak_length", 0),
            },
        }
    finally:
        gs.full_cleanup()


def run_mix(
    hero_spec: AgentSpec,
    mix_specs: Sequence[AgentSpec],
    frames: int,
    seeds: Sequence[int],
    engine: str = "live",
    profile: EvaluationProfile | None = None,
    mix_id: str = "unspecified",
) -> List[Dict[str, Any]]:
    """Run one hero over all seeds of one opponent mix.

    Args:
        hero_spec: Candidate/baseline agent for slot 0.
        mix_specs: One spec per opponent slot.
        frames: Total episode horizon.
        seeds: World seeds (paired across hero specs).
        engine: ``"live"`` (default) steps the Python :class:`GameState`
            per-seed; ``"simd"`` runs all seeds as parallel envs in the
            vectorized :class:`~src.simd_env.batch_sim.BatchSim` (blueprint
            §5.4, ~100x cheaper), computing the SAME gate metrics from the
            batch arrays. ``"simd"`` supports scripted agents only today — the
            raster network for checkpoints is unbuilt (blueprint P3), so a
            checkpoint hero/opponent raises there. ``"live"`` is unchanged.

    Returns:
        One per-seed metric dict per seed (in ``seeds`` order).
    """
    assignments = materialize_opponent_specs_by_world(mix_specs, seeds)
    if engine == "simd":
        from src.simd_env.eval_engine import run_simd_eval

        return run_simd_eval(
            hero_spec,
            list(mix_specs),
            frames,
            list(seeds),
            gamma=float(GameConfig.GAMMA),
            profile=profile,
            opponent_specs_by_world=assignments,
            world_identities=(
                {
                    int(seed): _roster_identity(int(seed), assignments[int(seed)], mix_id)
                    for seed in seeds
                }
                if profile is not None
                else None
            ),
            mix_id=mix_id,
        )
    return [
        rollout(
            hero_spec,
            assignments[int(seed)],
            frames,
            int(seed),
            profile=profile,
            world_identity=(
                _roster_identity(int(seed), assignments[int(seed)], mix_id)
                if profile is not None
                else None
            ),
            mix_id=mix_id,
        )
        for seed in seeds
    ]


def summarize_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed rollout metrics (mean + t-CI) plus probe aggregates."""
    out: Dict[str, Any] = {"n": len(runs)}
    for key in METRIC_KEYS:
        vals = [r[key] for r in runs]
        out[key] = mean(vals)
        out[f"{key}_ci"] = ci95_halfwidth(vals)
    death_causes: Dict[str, int] = {}
    for r in runs:
        cause = r["probes"]["death_cause"]
        if cause is not None:
            death_causes[cause] = death_causes.get(cause, 0) + 1

    def _mean_available(key: str):
        values = [r["probes"][key] for r in runs]
        return None if any(value is None for value in values) else mean([float(v) for v in values])

    out["probes"] = {
        "boost_frame_fraction": mean([r["probes"]["boost_frame_fraction"] for r in runs]),
        "food_eaten": mean([float(r["probes"]["food_eaten"]) for r in runs]),
        "kill_opportunity_count": _mean_available("kill_opportunity_count"),
        "entrapment_events": (
            None
            if any(r["probes"]["entrapment_event"] is None for r in runs)
            else sum(1 for r in runs if r["probes"]["entrapment_event"])
        ),
        "deaths_by_cause": death_causes,
    }
    reasons = [r.get("probe_unavailable_reasons", {}) for r in runs]
    unavailable = {key: value for reason in reasons for key, value in reason.items()}
    if unavailable:
        out["probe_unavailable_reasons"] = unavailable
    return out


def combined_paired_stats(
    candidate_runs_by_mix: Dict[str, Sequence[Dict[str, Any]]],
    baseline_runs_by_mix: Dict[str, Sequence[Dict[str, Any]]],
    mixes: Sequence[str],
) -> Dict[str, Any]:
    """Compute one equal-weight paired delta per world across opponent mixes.

    A world seed is reused across mixes, so its mix-specific results are
    correlated observations rather than extra independent samples. The
    descriptive combined interval therefore averages each world's
    candidate-minus-baseline deltas across the requested mixes, then computes
    the t interval over those distinct world averages. Per-mix paired results
    remain the promotion authority.

    Args:
        candidate_runs_by_mix: Candidate rollout records keyed by mix.
        baseline_runs_by_mix: Baseline rollout records keyed by mix.
        mixes: Predeclared opponent mixes, each given equal weight per world.

    Returns:
        The usual ``paired_stats`` schema over one aggregate delta per seed.

    Raises:
        ValueError: If mixes are missing, a run omits its seed, a seed is
            duplicated, or candidate/baseline and cross-mix seed sets differ.
    """
    mixes = list(dict.fromkeys(mixes))
    if not mixes:
        raise ValueError("combined paired analysis needs at least one mix")

    def scores_by_seed(runs: Sequence[Dict[str, Any]], label: str) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for run in runs:
            if "seed" not in run:
                raise ValueError(f"{label} run is missing its world seed")
            seed = int(run["seed"])
            if seed in scores:
                raise ValueError(f"{label} has duplicate world seed {seed}")
            scores[seed] = float(run["mass_integral"])
        return scores

    seed_order: List[int] | None = None
    deltas_by_seed: Dict[int, List[float]] = {}
    for mix in mixes:
        if mix not in candidate_runs_by_mix or mix not in baseline_runs_by_mix:
            raise ValueError(f"combined paired analysis is missing mix {mix!r}")
        candidate_scores = scores_by_seed(candidate_runs_by_mix[mix], f"candidate {mix!r}")
        baseline_scores = scores_by_seed(baseline_runs_by_mix[mix], f"baseline {mix!r}")
        if candidate_scores.keys() != baseline_scores.keys():
            raise ValueError(f"candidate/baseline world seed sets differ for mix {mix!r}")
        if seed_order is None:
            seed_order = list(candidate_scores)
            deltas_by_seed = {seed: [] for seed in seed_order}
        elif candidate_scores.keys() != deltas_by_seed.keys():
            raise ValueError(f"world seed sets differ across mixes at {mix!r}")
        for seed in seed_order:
            deltas_by_seed[seed].append(candidate_scores[seed] - baseline_scores[seed])

    seed_mean_deltas = [mean(deltas_by_seed[seed]) for seed in seed_order]
    return paired_stats(seed_mean_deltas, [0.0] * len(seed_mean_deltas))


def promotion_decision(
    per_mix_paired: Dict[str, Dict[str, Any]], mixes: Sequence[str]
) -> Dict[str, Any]:
    """Apply the §5.2 promotion rule to a candidate's per-mix paired stats.

    Promote iff the paired mass-integral delta is > 0 at 95% CI on >= 2 mixes
    AND there is no regression (CI entirely below 0) vs the scripted anchor.
    ``mixes`` is deduplicated first so the counts are over DISTINCT mixes.
    """
    mixes = list(dict.fromkeys(mixes))
    significant_mixes = [m for m in mixes if per_mix_paired[m]["significant"]]
    reasons: List[str] = []
    if len(mixes) < 2:
        reasons.append(f"evaluated on {len(mixes)} mix(es); >= 2 required")
    if len(significant_mixes) < 2:
        reasons.append(
            f"paired delta > 0 at 95% CI on {len(significant_mixes)} mix(es)"
            f" ({', '.join(significant_mixes) or 'none'}); >= 2 required"
        )
    if SCRIPTED_ANCHOR_MIX not in mixes:
        reasons.append("scripted anchor mix not evaluated; anchor regression unverifiable")
    elif per_mix_paired[SCRIPTED_ANCHOR_MIX]["regression"]:
        reasons.append("regression vs scripted anchor (95% CI entirely below 0)")
    return {
        "promote": not reasons,
        "significant_mixes": significant_mixes,
        "reasons": reasons,
    }


def print_markdown_report(
    baseline_label: str,
    mixes: Sequence[str],
    candidate_results: Sequence[Dict[str, Any]],
) -> None:
    """Print the per-mix paired results as a markdown table plus verdicts."""
    print("\n## Paired promotion-gate results (metric: mass integral, dead frames = 0)\n")
    print(f"Baseline: `{baseline_label}`\n")
    header = (
        "| candidate | mix | mass_int | Δ vs baseline (95% CI) | wins | surv | deaths |"
        " kills | boost% | verdict |"
    )
    print(header)
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for result in candidate_results:
        name = Path(result["candidate"]).name
        if result.get("error"):
            print(f"| {name} | - | - | - | - | - | - | - | - | ERROR: {result['error']} |")
            continue
        for mix in mixes:
            s = result["per_mix"][mix]["summary"]
            p = result["per_mix"][mix]["paired"]
            verdict = "WIN" if p["significant"] else ("LOSS" if p["regression"] else "ns")
            print(
                f"| {name} | {mix} | {s['mass_integral']:.2f} "
                f"| {p['mean_delta']:+.2f} ± {p['ci95']:.2f} "
                f"| {p['wins']}/{p['n']} | {s['survival_fraction']:.2f} "
                f"| {s['deaths']:.2f} | {s['kills']:.2f} "
                f"| {100 * s['probes']['boost_frame_fraction']:.0f} | {verdict} |"
            )
        c = result["combined_paired"]
        print(
            f"| {name} | **mean across mixes / world** | - | "
            f"{c['mean_delta']:+.2f} ± {c['ci95']:.2f} "
            f"| {c['wins']}/{c['n']} | - | - | - | - | "
            f"{'WIN' if c['significant'] else 'ns'} |"
        )

    print(
        "\nPromotion rule (blueprint §5.2): promote iff paired mass-integral delta > 0 at"
        " 95% CI on >= 2 opponent mixes AND no regression vs the scripted anchor."
    )
    for result in candidate_results:
        name = Path(result["candidate"]).name
        if result.get("error"):
            print(f"  {name}: REJECT (error: {result['error']})")
            continue
        decision = result["decision"]
        if decision["promote"]:
            print(f"  {name}: PROMOTE (significant on {', '.join(decision['significant_mixes'])})")
        else:
            print(f"  {name}: REJECT ({'; '.join(decision['reasons'])})")


def run_pilot(
    baseline_spec: AgentSpec,
    mixes: Sequence[str],
    mix_specs: Dict[str, List[AgentSpec]],
    frames: int,
    seeds: Sequence[int],
    mde_fraction: float,
    engine: str = "live",
    profile: EvaluationProfile | None = None,
) -> Dict[str, Any]:
    """Pilot mode: baseline-only variance estimate and seed-count recommendation.

    The per-seed std of the baseline's mass integral is used as the estimate of
    the paired-delta std (pairing removes common world variance; residual delta
    noise is assumed comparable to the baseline's own per-seed spread).
    """
    print(f"Pilot: baseline {agent_label(baseline_spec)} only, {len(seeds)} seeds\n")
    per_mix: Dict[str, Any] = {}
    recommendations: List[int] = []
    for mix in mixes:
        runs = run_mix(baseline_spec, mix_specs[mix], frames, seeds, engine, profile, mix)
        vals = [r["mass_integral"] for r in runs]
        mu = mean(vals)
        sd = sample_std(vals)
        mde = mde_fraction * mu
        rec = recommended_seed_count(sd, mde) if mde > 0 else None
        per_mix[mix] = {
            "mass_integral_mean": mu,
            "mass_integral_std": sd,
            "mde": mde,
            "recommended_seeds": rec,
            "runs": runs,
        }
        rec_text = str(rec) if rec is not None else "n/a (zero baseline mass)"
        print(
            f"  {mix:9s} mass_int {mu:8.2f} ± sd {sd:6.2f} | "
            f"MDE ({100 * mde_fraction:.0f}%) = {mde:.2f} | recommended seeds: {rec_text}"
        )
        if rec is not None:
            recommendations.append(rec)
    overall = max(recommendations) if recommendations else None
    print(f"\nRecommended paired seed count (max over mixes): {overall}")
    print("(assumes paired-delta std ≈ baseline per-seed std; alpha=0.05, power=0.8)")
    return {"per_mix": per_mix, "recommended_seeds": overall}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "candidates",
        nargs="*",
        help="Candidate checkpoints (or scripted:<kind> stand-ins) to evaluate as the hero",
    )
    p.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Baseline hero for paired deltas (default: incumbent champion)",
    )
    p.add_argument(
        "--opponents",
        default=",".join(DEFAULT_OPPONENTS),
        help="Comma-separated frozen opponent pool (checkpoints or scripted:<kind>)",
    )
    p.add_argument(
        "--opponent",
        default=None,
        help="Legacy alias: single frozen checkpoint used as the whole --opponents pool",
    )
    p.add_argument(
        "--mixes",
        type=parse_mix_list,
        default=list(MIX_NAMES),
        help=f"Comma-separated opponent mixes from {MIX_NAMES} (>= 2 required to gate)",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument(
        "--engine",
        choices=("live", "simd"),
        default="live",
        help=(
            "Rollout engine. 'live' (default): step the Python GameState per "
            "seed (works for the 61-D vector champions; P0/P1 gating unchanged). "
            "'simd': run all seeds as parallel envs in the vectorized BatchSim "
            "(~100x cheaper, blueprint §5.4) — SCRIPTED agents only for now; the "
            "raster network for checkpoints is unbuilt (P3), so checkpoint "
            "agents raise under 'simd'."
        ),
    )
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument(
        "--evaluation-profile",
        choices=("legacy-diagnostic", PROMOTION_V2_WATCH_RECT),
        default="legacy-diagnostic",
        help="Explicit evaluation task; promotion-v2-watch-rect fixes a 5000-frame horizon.",
    )
    p.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list("0-9"),
        help="Seeds: comma-separated (0,1,2) and/or inclusive ranges (0-15)",
    )
    p.add_argument("--json-output", default=None)
    p.add_argument(
        "--snapshot-dir",
        default=None,
        help=(
            "Directory for immutable content-addressed checkpoint inputs and their receipt "
            "(default: runs/evaluation_artifacts)"
        ),
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="Exit 0 iff the (single) candidate passes the §5.2 promotion rule",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Run the baseline only and recommend a seed count (power analysis)",
    )
    p.add_argument(
        "--mde-fraction",
        type=float,
        default=0.03,
        help="Pilot minimum detectable effect as a fraction of baseline mass integral",
    )
    args = p.parse_args(argv)

    if not args.pilot and not args.candidates:
        p.error("at least one candidate is required (or use --pilot)")
    if args.gate and len(args.candidates) != 1:
        p.error("--gate requires exactly one candidate")
    if args.gate and args.pilot:
        p.error("--gate and --pilot are mutually exclusive")

    config_path = Path(args.config)
    if not config_path.exists():
        p.error(f"config not found: {args.config} (default is {DEFAULT_CONFIG})")

    try:
        baseline_spec = parse_agent_spec(args.baseline)
        opponent_pool = [
            parse_agent_spec(s.strip())
            for s in (args.opponent or args.opponents).split(",")
            if s.strip()
        ]
        candidate_specs = [parse_agent_spec(c) for c in args.candidates]
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))

    requested_baseline_spec = baseline_spec
    requested_opponent_pool = list(opponent_pool)
    # Capture every trusted checkpoint before inspecting its header or creating
    # an arena. All later policy loads use these immutable paths, so a rolling
    # "latest" file cannot yield mixed candidate/baseline bytes across seeds.
    # Scripted references retain their compact, public names.
    source_specs = [requested_baseline_spec, *requested_opponent_pool, *candidate_specs]
    snapshot_root = args.snapshot_dir or os.environ.get(
        "SNAKE_DQN_EVAL_SNAPSHOT_DIR", "runs/evaluation_artifacts"
    )
    artifacts = EvaluationArtifacts(snapshot_root)
    try:
        config_snapshot = artifacts.snapshot_config(args.config)
        effective_config = load_and_initialize_config(config_snapshot.snapshot_path)
        profile = evaluation_profile_for_name(args.evaluation_profile, args.frames)
        baseline_spec = artifacts.snapshot_agent_specs([baseline_spec])[0]
        opponent_pool = artifacts.snapshot_agent_specs(opponent_pool)
        candidate_specs = artifacts.snapshot_agent_specs(candidate_specs)
        receipt_path = artifacts.write_receipt(
            config_snapshot=config_snapshot,
            effective_config=effective_config,
            evaluator_path=__file__,
            source_specs=source_specs,
            evaluation_profile=profile,
        )
    except (SnapshotError, ValueError) as exc:
        p.error(str(exc))
    active_profile = None if profile.legacy_diagnostic else profile

    def run_selected_mix(
        hero_spec: AgentSpec, specs: Sequence[AgentSpec], mix_id: str
    ) -> List[Dict[str, Any]]:
        """Keep legacy monkeypatch/caller signatures unchanged until opt-in."""
        if active_profile is None:
            return run_mix(hero_spec, specs, args.frames, args.seeds, args.engine)
        return run_mix(
            hero_spec, specs, args.frames, args.seeds, args.engine, active_profile, mix_id
        )

    # Fail fast: --engine simd can evaluate raster ('raster31v2') checkpoints
    # (the batch sim featurizes them) but NOT 61-D 'vector61' champions. Catch a
    # vector61 baseline or opponent up front (a vector61 CANDIDATE is instead
    # recorded as a per-candidate error by the eval loop) rather than aborting
    # mid-rollout with a traceback.
    if args.engine == "simd":
        from src.model.obs_spec import RASTER31V2, RASTER31V3

        offenders = []
        for spec in [baseline_spec, *opponent_pool]:
            if spec[0] != "checkpoint":
                continue
            obs_spec = _checkpoint_obs_spec_or_unknown(spec[1])
            if obs_spec == RASTER31V2 or (obs_spec == RASTER31V3 and active_profile is not None):
                continue
            offenders.append(agent_label(spec))
        if offenders:
            p.error(
                "--engine simd evaluates raster ('raster31v2'/'raster31v3') checkpoints and "
                "scripted:<kind> agents; the 61-D 'vector61' champions are not "
                "featurized by the batch sim. Use --engine live for them (or pass "
                f"raster/scripted opponents). Offending baseline/opponents: {offenders}"
            )

    num_opponents = int(GameConfig.NUM_SNAKES) - 1
    if num_opponents < 1:
        p.error(f"config num_snakes={GameConfig.NUM_SNAKES}; need >= 2 for opponents")

    requested_mix_specs = {
        mix: build_mix_specs(mix, num_opponents, requested_opponent_pool) for mix in args.mixes
    }
    mix_specs = {mix: build_mix_specs(mix, num_opponents, opponent_pool) for mix in args.mixes}

    # RasterServingPolicy chooses a Q row from the full pre-move roster by
    # dispatch order. It is therefore safe for the slot-0 hero only. Check the
    # expanded, requested mixes rather than the raw pool: a raster checkpoint
    # passed in --opponents is harmless when every requested mix is scripted
    # and thus never places it in the arena. This still runs before any game
    # construction or paired rollout; the SIMD engine has an identity-safe
    # batched policy implementation and supports raster opponents.
    if args.engine == "live":
        from src.model.obs_spec import RASTER31V2

        raster_opponents = []
        for specs in mix_specs.values():
            for spec in specs:
                if spec[0] != "checkpoint":
                    continue
                try:
                    if checkpoint_obs_spec(spec[1]) == RASTER31V2:
                        raster_opponents.append(agent_label(spec))
                except Exception:
                    # Preserve the normal policy-loading error for corrupt or
                    # absent vector checkpoints rather than changing it here.
                    pass
        raster_opponents = list(dict.fromkeys(raster_opponents))
        if raster_opponents:
            p.error(
                "--engine live supports raster31v2 checkpoints only as the "
                "candidate/baseline hero (slot 0); v2 opponents are unsafe because "
                "their serving policy dispatches Q rows in pre-move roster order. "
                "raster31v3 opponents use the identity-addressed serving path. "
                f"Offending opponents: {raster_opponents}"
            )

    print(
        f"Arena: {args.config} | engine={args.engine} | frames={args.frames} | seeds={args.seeds}"
    )
    print(f"Evaluation inputs: {receipt_path}")
    print(f"Baseline: {agent_label(requested_baseline_spec)}")
    for mix in args.mixes:
        print(f"Mix {mix:9s}: {[agent_label(s) for s in requested_mix_specs[mix]]}")
    print()

    if args.pilot:
        pilot = run_pilot(
            baseline_spec,
            args.mixes,
            mix_specs,
            args.frames,
            args.seeds,
            args.mde_fraction,
            args.engine,
            active_profile,
        )
        try:
            artifacts.verify_integrity()
        except SnapshotError as exc:
            print(f"evaluation input integrity failure: {exc}", file=sys.stderr)
            return 2
        if args.json_output:
            out = Path(args.json_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "mode": "pilot",
                        "config": args.config,
                        "engine": args.engine,
                        "frames": args.frames,
                        "seeds": args.seeds,
                        "baseline": agent_label(requested_baseline_spec),
                        "mixes": args.mixes,
                        "pilot": pilot,
                        "evaluation_inputs": {
                            "receipt": str(receipt_path),
                            "checkpoints": artifacts.checkpoint_receipts(),
                        },
                    },
                    indent=2,
                )
            )
            print(f"\nWrote {out}")
        return 0

    if len(args.mixes) < 2:
        print(
            "WARNING: only one mix requested; the §5.2 promotion rule needs >= 2 mixes.",
            file=sys.stderr,
        )

    # Baseline runs once per mix; every candidate pairs against these seeds.
    baseline_runs: Dict[str, List[Dict[str, Any]]] = {}
    baseline_summaries: Dict[str, Dict[str, Any]] = {}
    for mix in args.mixes:
        baseline_runs[mix] = run_selected_mix(baseline_spec, mix_specs[mix], mix)
        baseline_summaries[mix] = summarize_runs(baseline_runs[mix])
        print(
            f"baseline [{mix:9s}] mass_int={baseline_summaries[mix]['mass_integral']:.2f} "
            f"surv={baseline_summaries[mix]['survival_fraction']:.2f}"
        )

    candidate_results: List[Dict[str, Any]] = []
    for cand_spec, cand_raw in zip(candidate_specs, args.candidates):
        try:
            per_mix: Dict[str, Any] = {}
            for mix in args.mixes:
                runs = run_selected_mix(cand_spec, mix_specs[mix], mix)
                cand_vals = [r["mass_integral"] for r in runs]
                base_vals = [r["mass_integral"] for r in baseline_runs[mix]]
                per_mix[mix] = {
                    "runs": runs,
                    "summary": summarize_runs(runs),
                    "paired": paired_stats(cand_vals, base_vals),
                }
            paired_by_mix = {mix: per_mix[mix]["paired"] for mix in args.mixes}
            candidate_results.append(
                {
                    "candidate": cand_raw,
                    "per_mix": per_mix,
                    "combined_paired": combined_paired_stats(
                        {mix: per_mix[mix]["runs"] for mix in args.mixes},
                        baseline_runs,
                        args.mixes,
                    ),
                    "decision": promotion_decision(paired_by_mix, args.mixes),
                }
            )
        except Exception as exc:
            # One candidate failing (e.g. a contract/shape mismatch) must NOT
            # abort the run: record the error and keep evaluating the rest.
            print(f"  ERROR: {Path(cand_raw).name}: {exc}", file=sys.stderr)
            candidate_results.append(
                {
                    "candidate": cand_raw,
                    "error": str(exc),
                    "decision": {"promote": False, "significant_mixes": [], "reasons": [str(exc)]},
                }
            )

    try:
        artifacts.verify_integrity()
    except SnapshotError as exc:
        print(f"evaluation input integrity failure: {exc}", file=sys.stderr)
        return 2

    print_markdown_report(agent_label(requested_baseline_spec), args.mixes, candidate_results)

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "mode": "gate" if args.gate else "eval",
                    "config": args.config,
                    "engine": args.engine,
                    "frames": args.frames,
                    "seeds": args.seeds,
                    "baseline": agent_label(requested_baseline_spec),
                    "mixes": args.mixes,
                    "opponent_pool": [agent_label(s) for s in requested_opponent_pool],
                    "baseline_summaries": baseline_summaries,
                    "baseline_runs": baseline_runs,
                    "candidates": candidate_results,
                    "evaluation_inputs": {
                        "receipt": str(receipt_path),
                        "checkpoints": artifacts.checkpoint_receipts(),
                    },
                },
                indent=2,
            )
        )
        print(f"\nWrote {out}")

    if args.gate:
        return 0 if candidate_results[0]["decision"]["promote"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
