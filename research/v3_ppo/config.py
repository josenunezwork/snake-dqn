"""Configuration, VRAM sizing, and checkpoint export caps for Game V3.1 PPO (`research/v3_ppo`)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.obs import NUM_RASTER_CHANNELS, NUM_SCALAR_FEATURES

FINE_RASTER_SIZE: int = 64
MID_RASTER_SIZE: int = 32
WIDE_RASTER_SIZE: int = 32

V3_PPO_SCHEMA = "v3-ppo-config/v1"
V3_PPO_NAMESPACE = "v3-ppo-20261111"

NetSize = Literal["S", "M"]
VALID_NET_SIZES: tuple[NetSize, ...] = ("S", "M")

OpponentPolicy = Literal["food_seeking", "random_safe", "cardinal6_safe"]
VALID_OPPONENT_POLICIES: tuple[OpponentPolicy, ...] = (
    "food_seeking",
    "random_safe",
    "cardinal6_safe",
)

# Exact per-learner-step byte layout in `Uint8RolloutBuffer`:
# 3 uint8 rasters: 13*64*64 (53,248) + 13*32*32 (13,312) + 13*32*32 (13,312) = 79,872 bytes.
RASTER_S1_BYTES: int = NUM_RASTER_CHANNELS * FINE_RASTER_SIZE * FINE_RASTER_SIZE  # 53,248
RASTER_S4_BYTES: int = NUM_RASTER_CHANNELS * MID_RASTER_SIZE * MID_RASTER_SIZE  # 13,312
RASTER_S16_BYTES: int = NUM_RASTER_CHANNELS * WIDE_RASTER_SIZE * WIDE_RASTER_SIZE  # 13,312
OBS_RASTER_BYTES_PER_STEP: int = RASTER_S1_BYTES + RASTER_S4_BYTES + RASTER_S16_BYTES  # 79,872
OBS_METADATA_BYTES_PER_STEP: int = NUM_SCALAR_FEATURES * 4 + NUM_ACTIONS * 1  # 32 + 10 = 42
OBS_BATCH_BYTES_PER_ROW: int = OBS_RASTER_BYTES_PER_STEP + OBS_METADATA_BYTES_PER_STEP  # 79,914

# Rollout targets/flags per learner-step:
# actions (u8: 1) + log_probs, values, rewards, advantages, returns (5 * f32: 20)
# + died, truncated, acted (3 * bool: 3) = 24 bytes.
ROLLOUT_SCALAR_BYTES_PER_STEP: int = 1 + 5 * 4 + 3 * 1  # 24
BYTES_PER_LEARNER_STEP: int = OBS_BATCH_BYTES_PER_ROW + ROLLOUT_SCALAR_BYTES_PER_STEP  # 79,938

# Target GPU VRAM budgets (bytes) and default training world counts enforced at packet preparation.
GPU_VRAM_BUDGET_BYTES: dict[str, int] = {
    "L4": 22 * 1_000_000_000,  # 22.0 GB usable budget on 22.03 GiB (23.62 GB) L4
    "A100": 40 * 1_000_000_000,  # 40.0 GB budget on A100-SXM4-40GB
    "H100": 80 * 1_000_000_000,  # 80.0 GB budget on H100-80GB
}
GPU_DEFAULT_TRAIN_ENVS: dict[str, int] = {
    "L4": 128,
    "A100": 256,
    "H100": 512,
}

# Checkpoint sizing calibrated from real `pack_sim_state_dict` (`uint8` zlib level-1 chunks,
# `tensor_min_bytes = 4096`) at default `(R = 600, N = 24, max_length = 2048)`:
#   - raw persistent world grids + rings = 28,779,936 B (27.45 MiB/world);
#   - measured packed size at 3,000 ticks = 1,493,096 B (~1.424 MiB/world; 1,107,618 B/4 = 0.264 MiB
#     at tick 200, 2,838,114 B/4 = 0.677 MiB at tick 1,000, 8,063,394 B/4 = 1.922 MiB at tick
#     5,000);
#   - measured packed size at 15,000 ticks = 3,512,730 B (~3.35 MiB/world) as lazy-timestamp
#     occupancy cells accumulate across visited arena cells;
#   - synthetic 100%-visited arena circle + full 2,048-slot rings = ~15.7 MiB/world.
# We apply a stated 4.8x safety factor over the 15,000-tick measurement (11.2x over 3,000 ticks),
# budgeting 16.0 MiB/world for FULL checkpoint sets (covering even 100% arena saturation) and
# requiring `checkpoint_shard_bytes` to hold at least 1 raw uncompressed world * 1.05 (~28.8 MiB).
DEFAULT_RAW_WORLD_BYTES: int = 28_779_936
MEASURED_PACKED_WORLD_BYTES_3K: int = 1_493_096  # 1.424 MiB/world at tick 3,000 (E=4 shard)
MEASURED_PACKED_WORLD_BYTES_15K: int = 3_512_730  # 3.350 MiB/world at tick 15,000
CHECKPOINT_PACKED_SAFETY_FACTOR_OVER_15K: float = 4.8
BUDGETED_PACKED_WORLD_BYTES_DEFAULT: int = 16 * 1024 * 1024  # 16.0 MiB/world (covers late game)
DEFAULT_CHECKPOINT_SHARD_BYTES: int = 256 * 1024 * 1024  # 256 MiB max per FULL sim shard
MAX_SUPERVISOR_EXPORT_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GiB per-file supervisor ceiling
TELEMETRY_EXPORT_RESERVE_BYTES: int = 64 * 1024 * 1024  # 64 MiB JSONL/meta/manifest reserve

# Whole-session total export volume caps (bytes) across all published LIGHT + FULL exports.
# Sized for up to 24 h of unthrottled 30-min LIGHT exports (<=49 files) plus bounded FULL sets
# (`max_periodic_full_exports + 1` stop FULL = 4 FULL sets by default) + 64 MiB telemetry.
GPU_MAX_TOTAL_EXPORT_BYTES: dict[str, int] = {
    "L4": 12 * 1024 * 1024 * 1024,  # 12.0 GiB session cap (E=128, 24h M <= 11.32 GiB)
    "A100": 24 * 1024 * 1024 * 1024,  # 24.0 GiB session cap (E=256, 24h M <= 19.32 GiB)
    "H100": 40 * 1024 * 1024 * 1024,  # 40.0 GiB session cap (E=512, 24h M <= 35.33 GiB)
}

# Calibrated against measured A100 GameV3Sim footprint at (R=600, N=24, max_length=2048):
# exact persistent world grids+rings = 28,779,936 bytes (27.45 MiB/world; measured 27.4 MiB/world);
# measured peak = ~31.0 MiB/world.
_SIM_PERSISTENT_OVERHEAD_FACTOR = 1.0
_SIM_PEAK_OVER_PERSISTENT_FACTOR = 31.0 / 27.4467
_CUDA_RUNTIME_RESERVE_BYTES = 1_200 * 1024 * 1024  # ~1.20 GiB CUDA context + allocator reserve
_OBS_BUILD_TEMP_BYTES = 256 * 1024 * 1024  # chunked `build_obs` scratch headroom


def default_minibatch_size(net_size: NetSize) -> int:
    """Return default PPO learner minibatch row count for ``net_size``."""
    if net_size not in VALID_NET_SIZES:
        raise ValueError(f"net_size must be one of {VALID_NET_SIZES}, got {net_size!r}")
    return 2048 if net_size == "S" else 1024


def default_actor_chunk_learners(net_size: NetSize) -> int:
    """Return default per-tick forward chunk size (in learner rows) for ``net_size``."""
    if net_size not in VALID_NET_SIZES:
        raise ValueError(f"net_size must be one of {VALID_NET_SIZES}, got {net_size!r}")
    return 6144 if net_size == "S" else 3072


def default_eval_learner_slots(num_snakes: int) -> int:
    """Choose default M-of-N learner seat count for fixed-seed evaluation."""
    if num_snakes <= 1:
        return 1
    return min(8, max(1, num_snakes // 3))


def default_train_envs_for_gpu(gpu: str) -> int:
    """Return calibrated default training world count (`num_envs`) for ``gpu``."""
    gpu_norm = str(gpu).strip().upper()
    if gpu_norm not in GPU_DEFAULT_TRAIN_ENVS:
        raise ValueError(f"gpu must be one of {tuple(GPU_DEFAULT_TRAIN_ENVS)}, got {gpu!r}")
    return int(GPU_DEFAULT_TRAIN_ENVS[gpu_norm])


@dataclass(frozen=True)
class V3PPOConfig:
    """Immutable configuration for Game V3.1 PPO self-play training and evaluation."""

    game_config: GameV3Config = field(default_factory=GameV3Config)
    num_envs: int = 128
    learner_slots: int | None = None
    opponent_policy: OpponentPolicy = "food_seeking"

    # Network & precision settings (`channels_last` and `bf16` autocast ON by default on CUDA)
    net_size: NetSize = "S"
    amp: bool = True
    use_channels_last: bool = True
    compile_model: bool = False
    fast_learner_ops: bool = True

    # Rollout & PPO optimization hyperparameters
    rollout_len: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3.0e-4
    adam_eps: float = 1.0e-5
    weight_decay: float = 1.0e-4
    clip_ratio: float = 0.2
    ppo_epochs: int = 2
    minibatch_size: int | None = None
    actor_chunk_learners: int | None = None
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    # Reward coefficients (mirroring `arena_ppo` survival mass integral + StepInfo deltas/kills
    # + dying-snake-only death penalty; no positive kill bonus on simultaneous death)
    reward_alive_length_coef: float = 0.001
    reward_alive_const: float = 0.0
    reward_length_delta_coef: float = 0.1
    reward_kill_coef: float = 0.5
    reward_death_penalty: float = 1.0

    # Schedule, two-tier atomic checkpointing (every LIGHT exported at
    # `light_checkpoint_interval_sec` default 30 min; periodic FULL exports every
    # `full_checkpoint_interval_sec` default 2 h bounded by `max_periodic_full_exports` plus 1 at
    # cooperative stop / wall guard; `keep_*_checkpoints` prune `work/` only — completed exports in
    # `output/` are never rewritten or deleted), and in-session recovery.
    total_updates: int = 100
    light_checkpoint_interval_sec: float = 1800.0
    full_checkpoint_interval_sec: float = 7200.0
    max_periodic_full_exports: int = 3
    light_resume_margin_minutes: float = 45.0
    checkpoint_shard_bytes: int = DEFAULT_CHECKPOINT_SHARD_BYTES
    checkpoint_every_updates: int = 0
    full_checkpoint_every_updates: int = 0
    keep_full_checkpoints: int = 2
    keep_light_checkpoints: int = 4
    max_in_session_restarts: int = 2
    max_update_seconds: float | None = None

    # Deterministic single-pass M-of-N evaluation: wall-time cadence in minutes
    # (`eval_every_minutes`, default 60 min) and/or update cadence (`eval_every_updates`),
    # running both matchups (`food_seeking` + `random_safe`) in ONE simulator pass across
    # `eval_envs` worlds (`96` worlds/matchup at default `192`), matured via `eval_burnin_ticks`
    # (`250`) before `eval_ticks` (`2000`, yielding `>= 300` completed `random_safe` deaths).
    eval_every_minutes: float = 60.0
    eval_every_updates: int = 0
    eval_envs: int = 192
    eval_burnin_ticks: int = 250
    eval_ticks: int = 2000
    eval_estimated_seconds: float = 95.0
    eval_learner_slots: int | None = None
    eval_seed: int = 20261112
    eval_greedy: bool = False

    seed: int = 20261111
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.game_config, GameV3Config):
            raise TypeError(f"game_config must be a GameV3Config, got {type(self.game_config)!r}")
        if int(self.game_config.max_life_ticks) != 0:
            raise ValueError(
                "V3PPOConfig requires game_config.max_life_ticks == 0 so surviving snakes stay "
                "in the world across rollout boundaries (rollout truncation handles bootstrap)."
            )
        if int(self.num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        n_snakes = int(self.game_config.num_snakes)
        if self.learner_slots is not None:
            ls = int(self.learner_slots)
            if not (1 <= ls <= n_snakes):
                raise ValueError(
                    f"learner_slots must be in [1, {n_snakes}] or None, got {self.learner_slots}"
                )
        if self.opponent_policy not in VALID_OPPONENT_POLICIES:
            raise ValueError(
                f"opponent_policy must be one of {VALID_OPPONENT_POLICIES}, "
                f"got {self.opponent_policy!r}"
            )
        if self.net_size not in VALID_NET_SIZES:
            raise ValueError(f"net_size must be one of {VALID_NET_SIZES}, got {self.net_size!r}")
        if int(self.rollout_len) < 1:
            raise ValueError(f"rollout_len must be >= 1, got {self.rollout_len}")
        if not (0.0 <= float(self.gamma) <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not (0.0 <= float(self.gae_lambda) <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if not math.isfinite(self.lr) or float(self.lr) <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")
        if not math.isfinite(self.adam_eps) or float(self.adam_eps) <= 0.0:
            raise ValueError(f"adam_eps must be finite and > 0, got {self.adam_eps}")
        if not math.isfinite(self.weight_decay) or float(self.weight_decay) < 0.0:
            raise ValueError(f"weight_decay must be finite and >= 0, got {self.weight_decay}")
        if not math.isfinite(self.clip_ratio) or float(self.clip_ratio) <= 0.0:
            raise ValueError(f"clip_ratio must be finite and > 0, got {self.clip_ratio}")
        if int(self.ppo_epochs) < 1:
            raise ValueError(f"ppo_epochs must be >= 1, got {self.ppo_epochs}")
        if self.minibatch_size is not None and int(self.minibatch_size) < 1:
            raise ValueError(f"minibatch_size must be >= 1 or None, got {self.minibatch_size}")
        if self.actor_chunk_learners is not None and int(self.actor_chunk_learners) < 1:
            raise ValueError(
                f"actor_chunk_learners must be >= 1 or None, got {self.actor_chunk_learners}"
            )
        for coef_name in (
            "entropy_coef",
            "value_coef",
            "max_grad_norm",
            "reward_alive_length_coef",
            "reward_alive_const",
            "reward_length_delta_coef",
            "reward_kill_coef",
            "reward_death_penalty",
        ):
            val = float(getattr(self, coef_name))
            if not math.isfinite(val):
                raise ValueError(f"{coef_name} must be finite, got {val}")
        if float(self.reward_death_penalty) < 0.0:
            raise ValueError(
                "reward_death_penalty must be >= 0 (subtracted as -reward_death_penalty on death); "
                f"got {self.reward_death_penalty}"
            )
        if int(self.total_updates) < 0:
            raise ValueError(f"total_updates must be >= 0, got {self.total_updates}")
        if (
            not math.isfinite(self.light_checkpoint_interval_sec)
            or self.light_checkpoint_interval_sec <= 0.0
        ):
            raise ValueError(
                "light_checkpoint_interval_sec must be finite and > 0, "
                f"got {self.light_checkpoint_interval_sec}"
            )
        if (
            not math.isfinite(self.full_checkpoint_interval_sec)
            or self.full_checkpoint_interval_sec <= 0.0
        ):
            raise ValueError(
                "full_checkpoint_interval_sec must be finite and > 0, "
                f"got {self.full_checkpoint_interval_sec}"
            )
        if int(self.checkpoint_shard_bytes) < 1024:
            raise ValueError(
                f"checkpoint_shard_bytes must be >= 1024, got {self.checkpoint_shard_bytes}"
            )
        if int(self.max_periodic_full_exports) < 0:
            raise ValueError(
                f"max_periodic_full_exports must be >= 0, got {self.max_periodic_full_exports}"
            )
        if (
            not math.isfinite(self.light_resume_margin_minutes)
            or float(self.light_resume_margin_minutes) < 0.0
        ):
            raise ValueError(
                "light_resume_margin_minutes must be finite and >= 0, "
                f"got {self.light_resume_margin_minutes}"
            )
        if int(self.checkpoint_every_updates) < 0:
            raise ValueError(
                f"checkpoint_every_updates must be >= 0, got {self.checkpoint_every_updates}"
            )
        if int(self.full_checkpoint_every_updates) < 0:
            raise ValueError(
                "full_checkpoint_every_updates must be >= 0, "
                f"got {self.full_checkpoint_every_updates}"
            )
        if int(self.keep_full_checkpoints) < 1:
            raise ValueError(
                f"keep_full_checkpoints must be >= 1, got {self.keep_full_checkpoints}"
            )
        if int(self.keep_light_checkpoints) < 1:
            raise ValueError(
                f"keep_light_checkpoints must be >= 1, got {self.keep_light_checkpoints}"
            )
        if int(self.max_in_session_restarts) < 0:
            raise ValueError(
                f"max_in_session_restarts must be >= 0, got {self.max_in_session_restarts}"
            )
        if self.max_update_seconds is not None and (
            not math.isfinite(self.max_update_seconds) or float(self.max_update_seconds) <= 0.0
        ):
            raise ValueError(
                f"max_update_seconds must be finite and > 0 or None, got {self.max_update_seconds}"
            )
        if not math.isfinite(self.eval_every_minutes) or float(self.eval_every_minutes) < 0.0:
            raise ValueError(
                f"eval_every_minutes must be finite and >= 0, got {self.eval_every_minutes}"
            )
        if int(self.eval_every_updates) < 0:
            raise ValueError(f"eval_every_updates must be >= 0, got {self.eval_every_updates}")
        if int(self.eval_envs) < 1:
            raise ValueError(f"eval_envs must be >= 1, got {self.eval_envs}")
        if int(self.eval_burnin_ticks) < 0:
            raise ValueError(f"eval_burnin_ticks must be >= 0, got {self.eval_burnin_ticks}")
        if int(self.eval_ticks) < 1:
            raise ValueError(f"eval_ticks must be >= 1, got {self.eval_ticks}")
        if (
            not math.isfinite(self.eval_estimated_seconds)
            or float(self.eval_estimated_seconds) <= 0.0
        ):
            raise ValueError(
                "eval_estimated_seconds must be finite and > 0, "
                f"got {self.eval_estimated_seconds}"
            )
        if self.eval_learner_slots is not None:
            els = int(self.eval_learner_slots)
            if not (1 <= els <= n_snakes):
                raise ValueError(
                    f"eval_learner_slots must be in [1, {n_snakes}] or None, "
                    f"got {self.eval_learner_slots}"
                )

    @property
    def num_snakes(self) -> int:
        """Total snakes per world ``N``."""
        return int(self.game_config.num_snakes)

    @property
    def resolved_learner_slots(self) -> int:
        """Active learner snakes per world ``K`` (`N` by default)."""
        return (
            int(self.game_config.num_snakes)
            if self.learner_slots is None
            else int(self.learner_slots)
        )

    @property
    def total_learners(self) -> int:
        """Total learner seats across all worlds ``M = E * K``."""
        return int(self.num_envs) * self.resolved_learner_slots

    @property
    def resolved_minibatch_size(self) -> int:
        """Effective learner minibatch size."""
        if self.minibatch_size is not None:
            return int(self.minibatch_size)
        return default_minibatch_size(self.net_size)

    @property
    def resolved_actor_chunk_learners(self) -> int:
        """Effective forward chunk size in learner rows."""
        if self.actor_chunk_learners is not None:
            return int(self.actor_chunk_learners)
        return default_actor_chunk_learners(self.net_size)

    @property
    def resolved_eval_learner_slots(self) -> int:
        """Effective M-of-N learner seat count used during fixed-seed evaluation."""
        if self.eval_learner_slots is not None:
            return int(self.eval_learner_slots)
        return default_eval_learner_slots(self.num_snakes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config (including nested ``GameV3Config``) to plain JSON dict."""
        out = asdict(self)
        out["schema"] = V3_PPO_SCHEMA
        out["game_config"] = self.game_config.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V3PPOConfig:
        """Deserialize config produced by :meth:`to_dict`."""
        raw = dict(data)
        raw.pop("schema", None)
        gc_raw = raw.pop("game_config", None)
        game_cfg = (
            gc_raw
            if isinstance(gc_raw, GameV3Config)
            else (GameV3Config.from_dict(gc_raw) if isinstance(gc_raw, dict) else GameV3Config())
        )
        return cls(game_config=game_cfg, **raw)


def estimate_sim_memory_bytes(game_cfg: GameV3Config, num_envs: int) -> dict[str, int]:
    """Estimate persistent and peak ``GameV3Sim`` memory in bytes for ``num_envs`` worlds.

    Matches measured A100 footprint at `(R=600, N=24, max_length=2048)`:
    `27.45 MiB/world` persistent and `~31.0 MiB/world` peak.
    """
    e = int(num_envs)
    g = int(game_cfg.grid_size)
    g4 = g // 4
    g16 = g // 16
    n = int(game_cfg.num_snakes)
    max_len = int(game_cfg.max_length)

    # Persistent per-world grids + ring tensors in GameV3Sim:
    # occ_hi(i32) + occ_lo(i32) + ambient(u8) + corpse(i16) = 11 * G^2
    # occ_subtick_4x(i32) = 4 * G4^2 * N; occ_subtick_16x(i32) = 4 * G16^2 * N
    # ambient_4x/corpse_4x(i32) = 8 * G4^2; ambient_16x/corpse_16x(i32) = 8 * G16^2
    # ring_pos(2*i32) + ring_cell(2*i16) + ring_r_sq(i16) + ring_subtick(i32) = 18 * N * max_len
    raw_world_bytes = (
        11 * g * g
        + 4 * g4 * g4 * n
        + 4 * g16 * g16 * n
        + 8 * (g4 * g4 + g16 * g16)
        + 18 * n * max_len
        + 256 * n
    )
    persistent_per_world = int(math.ceil(raw_world_bytes * _SIM_PERSISTENT_OVERHEAD_FACTOR))
    peak_per_world = int(math.ceil(persistent_per_world * _SIM_PEAK_OVER_PERSISTENT_FACTOR))
    return {
        "raw_world_bytes": raw_world_bytes,
        "persistent_per_world_bytes": persistent_per_world,
        "peak_per_world_bytes": peak_per_world,
        "sim_persistent_bytes": e * persistent_per_world,
        "sim_peak_bytes": e * peak_per_world,
    }


def estimate_rollout_buffer_bytes(
    *,
    num_envs: int,
    learner_slots: int,
    rollout_len: int,
) -> dict[str, int]:
    """Compute exact byte counts for ``Uint8RolloutBuffer`` and the 1-step bootstrap cache."""
    m = int(num_envs) * int(learner_slots)
    total_steps = m * int(rollout_len)
    raster_bytes = total_steps * OBS_RASTER_BYTES_PER_STEP
    rollout_buffer_bytes = total_steps * BYTES_PER_LEARNER_STEP
    bootstrap_obs_bytes = m * OBS_BATCH_BYTES_PER_ROW
    return {
        "total_learners": m,
        "learner_steps_per_rollout": total_steps,
        "bytes_per_learner_step_rasters": OBS_RASTER_BYTES_PER_STEP,
        "bytes_per_learner_step_total": BYTES_PER_LEARNER_STEP,
        "rollout_raster_bytes": raster_bytes,
        "rollout_buffer_bytes": rollout_buffer_bytes,
        "bootstrap_obs_bytes": bootstrap_obs_bytes,
        "total_rollout_storage_bytes": rollout_buffer_bytes + bootstrap_obs_bytes,
    }


def estimate_learner_overhead_bytes(
    *,
    net_size: NetSize,
    minibatch_size: int,
    amp: bool = True,
) -> int:
    """Estimate model parameters, Adam moments, and minibatch activation peak on CUDA."""
    param_count = 1_050_000 if net_size == "S" else 5_200_000
    # fp32 weights + fp32 grads + Adam m/v = 16 bytes/param
    model_and_opt_bytes = param_count * 16
    # Conv stem + fusion activations + float raster copy per minibatch row
    # (~280 KiB S / ~560 KiB M in bf16; 1.8x fp32)
    act_per_row = 280 * 1024 if net_size == "S" else 560 * 1024
    if not amp:
        act_per_row = int(act_per_row * 1.8)
    activation_bytes = int(minibatch_size) * act_per_row
    return int(model_and_opt_bytes + activation_bytes)


def estimate_actor_chunk_temp_bytes(
    *,
    actor_chunk_rows: int,
    net_size: NetSize,
    amp: bool = True,
) -> int:
    """Estimate transient bf16/fp32 raster conversion + stem activations for 1 forward chunk."""
    rows = max(1, int(actor_chunk_rows))
    elem_bytes = 2 if amp else 4
    raster_float_copy_bytes = rows * OBS_RASTER_BYTES_PER_STEP * elem_bytes
    fwd_act_per_row = (64 if net_size == "S" else 128) * 1024
    if not amp:
        fwd_act_per_row *= 2
    return int(raster_float_copy_bytes + rows * fwd_act_per_row)


def estimate_eval_overhead_bytes(cfg: V3PPOConfig) -> int:
    """Estimate transient VRAM allocated when periodic evaluation runs inside training."""
    eval_enabled = (int(cfg.eval_every_updates) > 0) or (float(cfg.eval_every_minutes) > 0.0)
    if not eval_enabled:
        return 0
    eval_sim = estimate_sim_memory_bytes(cfg.game_config, cfg.eval_envs)
    eval_learners = int(cfg.eval_envs) * int(cfg.resolved_eval_learner_slots)
    eval_obs_bytes = eval_learners * OBS_BATCH_BYTES_PER_ROW
    return int(eval_sim["sim_peak_bytes"] + eval_obs_bytes)


def estimate_training_memory(cfg: V3PPOConfig, *, gpu: str | None = None) -> dict[str, Any]:
    """Return complete memory breakdown for ``cfg`` (including actor chunk & eval) and GPU fit."""
    sim_mem = estimate_sim_memory_bytes(cfg.game_config, cfg.num_envs)
    buf_mem = estimate_rollout_buffer_bytes(
        num_envs=cfg.num_envs,
        learner_slots=cfg.resolved_learner_slots,
        rollout_len=cfg.rollout_len,
    )
    eff_mb = cfg.resolved_minibatch_size
    param_count = 1_050_000 if cfg.net_size == "S" else 5_200_000
    model_and_opt_bytes = param_count * 16
    learner_bytes = estimate_learner_overhead_bytes(
        net_size=cfg.net_size,
        minibatch_size=eff_mb,
        amp=bool(cfg.amp),
    )
    learner_act_bytes = learner_bytes - model_and_opt_bytes

    eff_actor_chunk = min(
        max(1, buf_mem["total_learners"]),
        cfg.resolved_actor_chunk_learners,
    )
    actor_chunk_temp_bytes = estimate_actor_chunk_temp_bytes(
        actor_chunk_rows=eff_actor_chunk,
        net_size=cfg.net_size,
        amp=bool(cfg.amp),
    )
    eval_overhead_bytes = estimate_eval_overhead_bytes(cfg)

    # Peak VRAM is the maximum across two mutually exclusive runtime phases:
    #   1) Rollout + PPO learner (`sim_peak_bytes` + buffer + model/opt + max(learner, actor_chunk))
    #   2) Periodic evaluation (`sim_persistent_bytes` + buffer + model/opt + eval_sim + eval_fwd)
    train_phase_peak_bytes = (
        sim_mem["sim_peak_bytes"]
        + buf_mem["total_rollout_storage_bytes"]
        + model_and_opt_bytes
        + max(learner_act_bytes, actor_chunk_temp_bytes)
        + _OBS_BUILD_TEMP_BYTES
        + _CUDA_RUNTIME_RESERVE_BYTES
    )
    if eval_overhead_bytes > 0:
        eval_learners = int(cfg.eval_envs) * int(cfg.resolved_eval_learner_slots)
        eval_fwd_chunk = min(max(1, eval_learners), cfg.resolved_actor_chunk_learners)
        eval_fwd_temp_bytes = estimate_actor_chunk_temp_bytes(
            actor_chunk_rows=eval_fwd_chunk,
            net_size=cfg.net_size,
            amp=bool(cfg.amp),
        )
        eval_phase_peak_bytes = (
            sim_mem["sim_persistent_bytes"]
            + buf_mem["total_rollout_storage_bytes"]
            + model_and_opt_bytes
            + eval_overhead_bytes
            + eval_fwd_temp_bytes
            + _OBS_BUILD_TEMP_BYTES
            + _CUDA_RUNTIME_RESERVE_BYTES
        )
    else:
        eval_phase_peak_bytes = 0
    estimated_peak_bytes = max(train_phase_peak_bytes, eval_phase_peak_bytes)
    result: dict[str, Any] = {
        "num_envs": int(cfg.num_envs),
        "num_snakes": int(cfg.num_snakes),
        "learner_slots": int(cfg.resolved_learner_slots),
        "rollout_len": int(cfg.rollout_len),
        "net_size": cfg.net_size,
        "amp": bool(cfg.amp),
        "minibatch_size": int(eff_mb),
        "actor_chunk_learners": int(eff_actor_chunk),
        **sim_mem,
        **buf_mem,
        "model_and_opt_bytes": int(model_and_opt_bytes),
        "learner_overhead_bytes": int(learner_bytes),
        "actor_chunk_temp_bytes": int(actor_chunk_temp_bytes),
        "eval_overhead_bytes": int(eval_overhead_bytes),
        "obs_build_temp_bytes": _OBS_BUILD_TEMP_BYTES,
        "cuda_runtime_reserve_bytes": _CUDA_RUNTIME_RESERVE_BYTES,
        "train_phase_peak_bytes": int(train_phase_peak_bytes),
        "eval_phase_peak_bytes": int(eval_phase_peak_bytes),
        "estimated_peak_bytes": int(estimated_peak_bytes),
        "estimated_peak_gb": round(estimated_peak_bytes / 1.0e9, 3),
        "estimated_peak_gib": round(estimated_peak_bytes / float(1024**3), 3),
    }
    if gpu is not None:
        gpu_norm = gpu.strip().upper()
        if gpu_norm not in GPU_VRAM_BUDGET_BYTES:
            raise ValueError(
                f"Unsupported GPU {gpu!r}; expected one of {tuple(GPU_VRAM_BUDGET_BYTES)}"
            )
        budget = GPU_VRAM_BUDGET_BYTES[gpu_norm]
        result["gpu"] = gpu_norm
        result["gpu_budget_bytes"] = budget
        result["gpu_budget_gb"] = round(budget / 1.0e9, 3)
        result["fits_gpu_budget"] = bool(estimated_peak_bytes <= budget)
        result["headroom_bytes"] = int(budget - estimated_peak_bytes)
    return result


def validate_memory_budget(cfg: V3PPOConfig, gpu: str) -> dict[str, Any]:
    """Validate that ``cfg`` fits within ``gpu``'s VRAM budget; raise ``ValueError`` if not."""
    est = estimate_training_memory(cfg, gpu=gpu)
    if not est["fits_gpu_budget"]:
        learner_rt_gb = (
            est["model_and_opt_bytes"]
            + max(
                est["learner_overhead_bytes"] - est["model_and_opt_bytes"],
                est["actor_chunk_temp_bytes"],
            )
            + est["eval_overhead_bytes"]
            + est["obs_build_temp_bytes"]
            + est["cuda_runtime_reserve_bytes"]
        ) / 1e9
        raise ValueError(
            f"Refusing configuration that exceeds {est['gpu']} memory budget "
            f"({est['estimated_peak_gb']:.2f} GB estimated peak > "
            f"{est['gpu_budget_gb']:.2f} GB budget): "
            f"worlds={cfg.num_envs}, learners={cfg.resolved_learner_slots}, "
            f"rollout_len={cfg.rollout_len}, net_size={cfg.net_size} "
            f"(sim_peak={est['sim_peak_bytes'] / 1e9:.2f} GB, "
            f"rollout_storage={est['total_rollout_storage_bytes'] / 1e9:.2f} GB, "
            f"learner+actor+eval+runtime={learner_rt_gb:.2f} GB). "
            "Reduce num_envs, learner_slots, or rollout_len, or choose a larger GPU."
        )
    return est


def estimate_checkpoint_export_caps(
    cfg: V3PPOConfig,
    *,
    gpu: str = "L4",
    max_wall_seconds: float = 8.0 * 3600.0,
    expected_updates: int | None = None,
    measured_packed_bytes_per_world: int | None = None,
) -> dict[str, Any]:
    """Size single-file and total-session checkpoint export caps from wall horizon & bytes/world.

    Session export arithmetic:
      - Every LIGHT checkpoint is exported at its real ``light_checkpoint_interval_sec`` (`30` min
        default) with no count cap and no interval stretching:
        ``n_light_time = ceil(max_wall_seconds / light_checkpoint_interval_sec) + 1``
        (`17` max on an `8 h` session, `49` max on a `24 h` session), plus any update-cadence
        exports when ``checkpoint_every_updates > 0``.
      - Periodic FULL checkpoints are bounded by ``max_periodic_full_exports`` (`3` default, one
        every ``full_checkpoint_interval_sec = 7200``s), plus ``1`` at cooperative stop / wall
        guard (`max_full_exports = max_periodic_full_exports + 1 = 4` default).
      - ``keep_full_checkpoints`` and ``keep_light_checkpoints`` govern ``work/`` retention only;
        completed exports in ``output/`` are immutable and never deleted.
    """
    gpu_norm = gpu.strip().upper()
    if gpu_norm not in GPU_MAX_TOTAL_EXPORT_BYTES:
        raise ValueError(
            f"Unsupported GPU {gpu!r}; expected one of {tuple(GPU_MAX_TOTAL_EXPORT_BYTES)}"
        )
    if not math.isfinite(max_wall_seconds) or float(max_wall_seconds) < 0.0:
        raise ValueError(f"max_wall_seconds must be finite and >= 0, got {max_wall_seconds}")

    sim_mem = estimate_sim_memory_bytes(cfg.game_config, 1)
    raw_world_bytes = int(sim_mem["raw_world_bytes"])
    scale = float(raw_world_bytes) / float(DEFAULT_RAW_WORLD_BYTES)

    if measured_packed_bytes_per_world is not None:
        if int(measured_packed_bytes_per_world) < 1:
            raise ValueError(
                "measured_packed_bytes_per_world must be >= 1, "
                f"got {measured_packed_bytes_per_world}"
            )
        budgeted_world_bytes = int(
            math.ceil(
                int(measured_packed_bytes_per_world) * CHECKPOINT_PACKED_SAFETY_FACTOR_OVER_15K
            )
        )
    else:
        budgeted_world_bytes = int(math.ceil(BUDGETED_PACKED_WORLD_BYTES_DEFAULT * scale))

    # Physical uncompressed ceiling for 1 whole world (raw tensors + 5% container framing):
    # a single shard must always be able to hold at least 1 whole world.
    max_single_world_bytes = max(budgeted_world_bytes, int(math.ceil(raw_world_bytes * 1.05)))
    shard_cap_bytes = int(cfg.checkpoint_shard_bytes)

    # LIGHT checkpoint = fp32 weights + Adam m/v (12 bytes/param) + RNG/counters/pickle framing.
    est_light_bytes = (16 if cfg.net_size == "S" else 68) * 1024 * 1024

    est_full_sim_bytes = int(cfg.num_envs) * budgeted_world_bytes
    est_shards_per_full = max(1, int(math.ceil(float(est_full_sim_bytes) / float(shard_cap_bytes))))
    # Sim-shard + index portion of one FULL checkpoint set (companion `light_u*.pt` counted in
    # `max_light_exports` when `full_interval >= light_interval`, or via `extra_full_lights`).
    est_full_sim_only_bytes = est_full_sim_bytes + est_shards_per_full * 64 * 1024

    n_light_time = (
        int(math.ceil(float(max_wall_seconds) / float(cfg.light_checkpoint_interval_sec))) + 1
    )
    n_light_updates = 0
    if int(cfg.checkpoint_every_updates) > 0:
        if expected_updates is not None:
            u_bound = max(0, int(expected_updates))
        elif 0 < int(cfg.total_updates) < 1_000_000_000:
            u_bound = int(cfg.total_updates)
        else:
            # Conservative physical upper bound (1 update/sec) when `checkpoint_every_updates > 0`
            # is requested without `expected_updates` or a finite `total_updates` cap.
            u_bound = int(math.ceil(max(1.0, float(max_wall_seconds))))
        n_light_updates = u_bound // int(cfg.checkpoint_every_updates)

    max_periodic_full = int(cfg.max_periodic_full_exports)
    max_full_exports = max_periodic_full + 1
    extra_full_lights = (
        max_periodic_full
        if (
            float(cfg.full_checkpoint_interval_sec) < float(cfg.light_checkpoint_interval_sec)
            or int(cfg.full_checkpoint_every_updates) > 0
        )
        else 0
    )
    max_light_exports = n_light_time + n_light_updates + extra_full_lights

    total_light_export_bytes = max_light_exports * est_light_bytes
    total_full_sim_export_bytes = max_full_exports * est_full_sim_only_bytes
    worst_case_session_export_bytes = (
        total_light_export_bytes + total_full_sim_export_bytes + TELEMETRY_EXPORT_RESERVE_BYTES
    )

    largest_single_file_bytes = max(shard_cap_bytes, est_light_bytes)
    per_file_max_export_bytes = max(
        256 * 1024 * 1024,
        int(math.ceil(largest_single_file_bytes * 1.25)),
    )
    session_total_cap_bytes = GPU_MAX_TOTAL_EXPORT_BYTES[gpu_norm]

    return {
        "gpu": gpu_norm,
        "max_wall_seconds": float(max_wall_seconds),
        "raw_world_bytes": raw_world_bytes,
        "measured_world_bytes_3k": int(math.ceil(MEASURED_PACKED_WORLD_BYTES_3K * scale)),
        "measured_world_bytes_15k": int(math.ceil(MEASURED_PACKED_WORLD_BYTES_15K * scale)),
        "safety_factor_over_15k": CHECKPOINT_PACKED_SAFETY_FACTOR_OVER_15K,
        "budgeted_packed_world_bytes": budgeted_world_bytes,
        "max_single_world_bytes": max_single_world_bytes,
        "checkpoint_shard_bytes": shard_cap_bytes,
        "est_light_checkpoint_bytes": est_light_bytes,
        "est_full_sim_bytes": est_full_sim_only_bytes,
        "est_shards_per_full": est_shards_per_full,
        "max_periodic_full_exports": max_periodic_full,
        "max_full_exports": max_full_exports,
        "max_light_exports": max_light_exports,
        "total_light_export_bytes": total_light_export_bytes,
        "total_full_sim_export_bytes": total_full_sim_export_bytes,
        "keep_full_checkpoints": int(cfg.keep_full_checkpoints),
        "keep_light_checkpoints": int(cfg.keep_light_checkpoints),
        "per_file_max_export_bytes": per_file_max_export_bytes,
        "supervisor_per_file_limit_bytes": MAX_SUPERVISOR_EXPORT_BYTES,
        "worst_case_session_export_bytes": worst_case_session_export_bytes,
        "session_total_export_cap_bytes": session_total_cap_bytes,
        "single_world_fits_shard": bool(max_single_world_bytes <= shard_cap_bytes),
        "single_file_fits_supervisor": bool(
            per_file_max_export_bytes <= MAX_SUPERVISOR_EXPORT_BYTES
        ),
        "total_session_fits_cap": bool(worst_case_session_export_bytes <= session_total_cap_bytes),
    }


def validate_checkpoint_export_caps(
    cfg: V3PPOConfig,
    gpu: str,
    *,
    max_wall_seconds: float = 8.0 * 3600.0,
    expected_updates: int | None = None,
    measured_packed_bytes_per_world: int | None = None,
) -> dict[str, Any]:
    """Validate shard and whole-session checkpoint export bounds; raise ``ValueError`` if unsafe."""
    caps = estimate_checkpoint_export_caps(
        cfg,
        gpu=gpu,
        max_wall_seconds=max_wall_seconds,
        expected_updates=expected_updates,
        measured_packed_bytes_per_world=measured_packed_bytes_per_world,
    )
    if not caps["single_world_fits_shard"]:
        raise ValueError(
            f"Refusing configuration whose single-world worst-case state "
            f"({caps['max_single_world_bytes'] / 2**20:.2f} MiB) exceeds "
            f"checkpoint_shard_bytes ({caps['checkpoint_shard_bytes'] / 2**20:.2f} MiB)."
        )
    if not caps["single_file_fits_supervisor"]:
        raise ValueError(
            f"Refusing configuration whose per-file export cap "
            f"({caps['per_file_max_export_bytes'] / 2**30:.2f} GiB, from shard/light max "
            f"{max(caps['checkpoint_shard_bytes'], caps['est_light_checkpoint_bytes']) / 2**20:.1f}"
            f" MiB * 1.25) exceeds the supervisor 2.00 GiB per-file limit."
        )
    if not caps["total_session_fits_cap"]:
        raise ValueError(
            f"Refusing configuration whose worst-case session export volume "
            f"({caps['worst_case_session_export_bytes'] / 2**30:.2f} GiB across "
            f"{caps['max_full_exports']} FULL + {caps['max_light_exports']} LIGHT exports "
            f"for {cfg.num_envs} worlds) exceeds {caps['gpu']} session export cap "
            f"({caps['session_total_export_cap_bytes'] / 2**30:.2f} GiB)."
        )
    return caps
