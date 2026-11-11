"""Configuration and VRAM sizing contracts for Game V3.1 PPO self-play (`research/v3_ppo`)."""

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

# Target GPU VRAM budgets (bytes) enforced at packet-preparation time.
GPU_VRAM_BUDGET_BYTES: dict[str, int] = {
    "L4": 22 * 1_000_000_000,  # 22.0 GB usable budget on 22.03 GiB (23.62 GB) L4
    "A100": 40 * 1_000_000_000,  # 40.0 GB budget on A100-SXM4-40GB
    "H100": 80 * 1_000_000_000,  # 80.0 GB budget on H100-80GB
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
    # + dying-snake-only death penalty)
    reward_alive_length_coef: float = 0.001
    reward_alive_const: float = 0.0
    reward_length_delta_coef: float = 0.1
    reward_kill_coef: float = 0.5
    reward_death_penalty: float = 1.0

    # Schedule, atomic checkpointing (default 30 min = 1800s), and evaluation
    total_updates: int = 100
    checkpoint_interval_sec: float = 1800.0
    output_publish_interval_sec: float = 1800.0
    checkpoint_every_updates: int = 0
    keep_periodic_checkpoints: int = 2
    max_update_seconds: float | None = None

    eval_every_updates: int = 25
    eval_envs: int = 16
    eval_ticks: int = 300
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
        if not math.isfinite(self.checkpoint_interval_sec) or self.checkpoint_interval_sec <= 0.0:
            raise ValueError(
                "checkpoint_interval_sec must be finite and > 0, "
                f"got {self.checkpoint_interval_sec}"
            )
        if (
            not math.isfinite(self.output_publish_interval_sec)
            or self.output_publish_interval_sec <= 0.0
        ):
            raise ValueError(
                "output_publish_interval_sec must be finite and > 0, "
                f"got {self.output_publish_interval_sec}"
            )
        if int(self.checkpoint_every_updates) < 0:
            raise ValueError(
                f"checkpoint_every_updates must be >= 0, got {self.checkpoint_every_updates}"
            )
        if int(self.keep_periodic_checkpoints) < 1:
            raise ValueError(
                f"keep_periodic_checkpoints must be >= 1, got {self.keep_periodic_checkpoints}"
            )
        if self.max_update_seconds is not None and (
            not math.isfinite(self.max_update_seconds) or float(self.max_update_seconds) <= 0.0
        ):
            raise ValueError(
                f"max_update_seconds must be finite and > 0 or None, got {self.max_update_seconds}"
            )
        if int(self.eval_every_updates) < 0:
            raise ValueError(f"eval_every_updates must be >= 0, got {self.eval_every_updates}")
        if int(self.eval_envs) < 1:
            raise ValueError(f"eval_envs must be >= 1, got {self.eval_envs}")
        if int(self.eval_ticks) < 1:
            raise ValueError(f"eval_ticks must be >= 1, got {self.eval_ticks}")
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
    # Conv stem + fusion activations per minibatch row (~280 KiB S / ~560 KiB M in bf16; 1.8x fp32)
    act_per_row = 280 * 1024 if net_size == "S" else 560 * 1024
    if not amp:
        act_per_row = int(act_per_row * 1.8)
    activation_bytes = int(minibatch_size) * act_per_row
    return int(model_and_opt_bytes + activation_bytes)


def estimate_training_memory(cfg: V3PPOConfig, *, gpu: str | None = None) -> dict[str, Any]:
    """Return complete memory breakdown for ``cfg`` and optional GPU fit status."""
    sim_mem = estimate_sim_memory_bytes(cfg.game_config, cfg.num_envs)
    buf_mem = estimate_rollout_buffer_bytes(
        num_envs=cfg.num_envs,
        learner_slots=cfg.resolved_learner_slots,
        rollout_len=cfg.rollout_len,
    )
    eff_mb = min(
        cfg.resolved_minibatch_size,
        max(1, buf_mem["learner_steps_per_rollout"]),
    )
    learner_bytes = estimate_learner_overhead_bytes(
        net_size=cfg.net_size,
        minibatch_size=eff_mb,
        amp=bool(cfg.amp),
    )
    estimated_peak_bytes = (
        sim_mem["sim_peak_bytes"]
        + buf_mem["total_rollout_storage_bytes"]
        + learner_bytes
        + _OBS_BUILD_TEMP_BYTES
        + _CUDA_RUNTIME_RESERVE_BYTES
    )
    result: dict[str, Any] = {
        "num_envs": int(cfg.num_envs),
        "num_snakes": int(cfg.num_snakes),
        "learner_slots": int(cfg.resolved_learner_slots),
        "rollout_len": int(cfg.rollout_len),
        "net_size": cfg.net_size,
        "amp": bool(cfg.amp),
        "minibatch_size": int(eff_mb),
        **sim_mem,
        **buf_mem,
        "learner_overhead_bytes": learner_bytes,
        "obs_build_temp_bytes": _OBS_BUILD_TEMP_BYTES,
        "cuda_runtime_reserve_bytes": _CUDA_RUNTIME_RESERVE_BYTES,
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
            est["learner_overhead_bytes"]
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
            f"learner+runtime={learner_rt_gb:.2f} GB). "
            "Reduce num_envs, learner_slots, or rollout_len, or choose a larger GPU."
        )
    return est
