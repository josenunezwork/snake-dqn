"""PPO self-play trainer, throughput ruler, and Colab packet builder for Game V3.1."""

from __future__ import annotations

from research.v3_ppo.buffer import Uint8RolloutBuffer, compute_gae
from research.v3_ppo.config import (
    BYTES_PER_LEARNER_STEP,
    GPU_VRAM_BUDGET_BYTES,
    OBS_BATCH_BYTES_PER_ROW,
    OBS_RASTER_BYTES_PER_STEP,
    V3_PPO_NAMESPACE,
    V3_PPO_SCHEMA,
    NetSize,
    OpponentPolicy,
    V3PPOConfig,
    estimate_rollout_buffer_bytes,
    estimate_sim_memory_bytes,
    estimate_training_memory,
    validate_memory_budget,
)
from research.v3_ppo.eval import (
    compute_step_reward,
    evaluate_policy,
    evaluate_scripted_baseline,
    scripted_opponent_actions,
)
from research.v3_ppo.net import (
    EgoMultiScaleNet,
    EgoMultiScaleNetConfig,
    build_ego_multiscale_net,
    symlog_two_hot_loss,
)
from research.v3_ppo.ppo import (
    V3PPOTrainer,
    pack_sim_state_dict,
    run_training_job,
    unpack_sim_state_dict,
    verify_staged_checkpoint,
)

__all__ = [
    "BYTES_PER_LEARNER_STEP",
    "EgoMultiScaleNet",
    "EgoMultiScaleNetConfig",
    "GPU_VRAM_BUDGET_BYTES",
    "NetSize",
    "OBS_BATCH_BYTES_PER_ROW",
    "OBS_RASTER_BYTES_PER_STEP",
    "OpponentPolicy",
    "Uint8RolloutBuffer",
    "V3PPOConfig",
    "V3PPOTrainer",
    "V3_PPO_NAMESPACE",
    "V3_PPO_SCHEMA",
    "build_ego_multiscale_net",
    "compute_gae",
    "compute_step_reward",
    "estimate_rollout_buffer_bytes",
    "estimate_sim_memory_bytes",
    "estimate_training_memory",
    "evaluate_policy",
    "evaluate_scripted_baseline",
    "pack_sim_state_dict",
    "run_training_job",
    "scripted_opponent_actions",
    "symlog_two_hot_loss",
    "unpack_sim_state_dict",
    "validate_memory_budget",
    "verify_staged_checkpoint",
]
