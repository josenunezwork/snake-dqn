"""Configuration, VRAM sizing, and checkpoint export caps for Game V3.1 PPO (`research/v3_ppo`)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.obs import NUM_RASTER_CHANNELS, NUM_SCALAR_FEATURES

_DEFAULT_GAME_CONFIG: GameV3Config = GameV3Config()

V3_PPO_SCHEMA = "v3-ppo-config/v1"
V3_PPO_NAMESPACE = "v3-ppo-20261111"

NetSize = Literal["S", "M"]


@dataclass(frozen=True)
class NetPreset:
    """Single-sourced architecture, memory-model, and batching parameters for a network size."""

    name: NetSize
    ruler_order: int
    # Architecture hyperparameters consumed by `EgoMultiScaleNetConfig.from_size`
    s1_c1: int
    s1_c2: int
    s32_c1: int
    stage8_ch: int
    stem_out_ch: int
    scalar_embed_dim: int
    trunk_dim: int
    trunk_blocks: int
    head_dim: int
    # Exact parameter count and rounded parameter estimate used by the memory model
    exact_param_count: int
    memory_model_param_count: int
    # Measured bf16 activation bytes per row on CUDA
    learner_act_bytes_per_row_bf16: int
    actor_fwd_act_bytes_per_row_bf16: int
    # Default batching and LIGHT checkpoint export byte budget
    default_minibatch_size: int
    default_actor_chunk_learners: int
    est_light_checkpoint_bytes: int


NET_PRESETS: dict[str, NetPreset] = {
    "S": NetPreset(
        name="S",
        ruler_order=0,
        s1_c1=32,
        s1_c2=48,
        s32_c1=32,
        stage8_ch=64,
        stem_out_ch=48,
        scalar_embed_dim=64,
        trunk_dim=168,
        trunk_blocks=1,
        head_dim=128,
        exact_param_count=1_008_657,
        memory_model_param_count=1_050_000,
        learner_act_bytes_per_row_bf16=280 * 1024,
        actor_fwd_act_bytes_per_row_bf16=64 * 1024,
        default_minibatch_size=2048,
        default_actor_chunk_learners=6144,
        est_light_checkpoint_bytes=16 * 1024 * 1024,
    ),
    "M": NetPreset(
        name="M",
        ruler_order=1,
        s1_c1=64,
        s1_c2=96,
        s32_c1=64,
        stage8_ch=144,
        stem_out_ch=96,
        scalar_embed_dim=128,
        trunk_dim=392,
        trunk_blocks=2,
        head_dim=256,
        exact_param_count=4_927_841,
        memory_model_param_count=5_200_000,
        learner_act_bytes_per_row_bf16=560 * 1024,
        actor_fwd_act_bytes_per_row_bf16=128 * 1024,
        default_minibatch_size=1024,
        default_actor_chunk_learners=3072,
        est_light_checkpoint_bytes=68 * 1024 * 1024,
    ),
}

VALID_NET_SIZES: tuple[NetSize, ...] = tuple(p.name for p in NET_PRESETS.values())


def get_net_preset(size: str) -> NetPreset:
    """Return the registered :class:`NetPreset` for ``size`` or raise ``ValueError``."""
    preset = NET_PRESETS.get(size) if isinstance(size, str) else None
    if preset is None:
        raise ValueError(f"Unknown net_size {size!r}; expected one of {VALID_NET_SIZES}")
    return preset


OpponentPolicy = Literal["food_seeking", "random_safe", "cardinal6_safe"]
VALID_OPPONENT_POLICIES: tuple[OpponentPolicy, ...] = (
    "food_seeking",
    "random_safe",
    "cardinal6_safe",
)

# Rollout targets/flags per learner-step:
# actions (u8: 1) + log_probs, values, rewards, advantages, returns (5 * f32: 20)
# + died, truncated, acted (3 * bool: 3) = 24 bytes.
ROLLOUT_SCALAR_BYTES_PER_STEP: int = 1 + 5 * 4 + 3 * 1  # 24


@dataclass(frozen=True)
class ObsLayout:
    """Single-sourced observation tensor layout derived from ``GameV3Config`` & ``game_v3.obs``."""

    channels: int = NUM_RASTER_CHANNELS
    fine_size: int = _DEFAULT_GAME_CONFIG.obs_fine_size
    mid_size: int = _DEFAULT_GAME_CONFIG.obs_coarse_size
    wide_size: int = _DEFAULT_GAME_CONFIG.obs_coarse_size
    scales: tuple[int, int, int] = _DEFAULT_GAME_CONFIG.obs_scales
    scalar_dim: int = NUM_SCALAR_FEATURES
    num_actions: int = NUM_ACTIONS

    @property
    def raster_sizes(self) -> tuple[int, int, int]:
        """Return ``(fine_size, mid_size, wide_size)`` spatial dimensions."""
        return (int(self.fine_size), int(self.mid_size), int(self.wide_size))

    @property
    def raster_s1_bytes(self) -> int:
        """Byte count of one ``raster_s1`` ``uint8`` slice."""
        return int(self.channels) * int(self.fine_size) * int(self.fine_size)

    @property
    def raster_s4_bytes(self) -> int:
        """Byte count of one ``raster_s4`` ``uint8`` slice."""
        return int(self.channels) * int(self.mid_size) * int(self.mid_size)

    @property
    def raster_s16_bytes(self) -> int:
        """Byte count of one ``raster_s16`` ``uint8`` slice."""
        return int(self.channels) * int(self.wide_size) * int(self.wide_size)

    @property
    def raster_bytes_per_step(self) -> int:
        """Total ``uint8`` raster bytes per learner step across all 3 scales."""
        return self.raster_s1_bytes + self.raster_s4_bytes + self.raster_s16_bytes

    @property
    def metadata_bytes_per_step(self) -> int:
        """Byte count of float32 scalars plus bool action mask per learner step."""
        return int(self.scalar_dim) * 4 + int(self.num_actions) * 1

    @property
    def obs_batch_bytes_per_row(self) -> int:
        """Byte count of one row in an ``ObservationBatch``."""
        return self.raster_bytes_per_step + self.metadata_bytes_per_step

    @property
    def bytes_per_learner_step(self) -> int:
        """Total bytes per learner step stored in ``Uint8RolloutBuffer``."""
        return self.obs_batch_bytes_per_row + ROLLOUT_SCALAR_BYTES_PER_STEP

    def to_dict(self) -> dict[str, Any]:
        """Serialize layout to a JSON-compatible dictionary."""
        return {
            "channels": int(self.channels),
            "fine_size": int(self.fine_size),
            "mid_size": int(self.mid_size),
            "wide_size": int(self.wide_size),
            "scales": [int(s) for s in self.scales],
            "scalar_dim": int(self.scalar_dim),
            "num_actions": int(self.num_actions),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ObsLayout:
        """Deserialize layout from a dictionary (e.g. checkpoint metadata)."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected obs_layout dict, got {type(data)!r}")
        required = ("channels", "fine_size", "mid_size", "wide_size", "scalar_dim", "num_actions")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"obs_layout missing required keys: {missing}")
        raw_scales = data.get("scales", _DEFAULT_GAME_CONFIG.obs_scales)
        if not isinstance(raw_scales, (list, tuple)) or len(raw_scales) != 3:
            raise ValueError(f"Invalid obs_layout scales: {raw_scales!r}")
        return cls(
            channels=int(data["channels"]),
            fine_size=int(data["fine_size"]),
            mid_size=int(data["mid_size"]),
            wide_size=int(data["wide_size"]),
            scales=(int(raw_scales[0]), int(raw_scales[1]), int(raw_scales[2])),
            scalar_dim=int(data["scalar_dim"]),
            num_actions=int(data["num_actions"]),
        )


def obs_layout_from_game_config(game_cfg: GameV3Config = _DEFAULT_GAME_CONFIG) -> ObsLayout:
    """Derive the observation layout produced by ``game_v3.obs`` for ``game_cfg``."""
    if len(game_cfg.obs_scales) != 3:
        raise ValueError(f"Expected 3 obs_scales in GameV3Config, got {game_cfg.obs_scales!r}")
    return ObsLayout(
        channels=NUM_RASTER_CHANNELS,
        fine_size=int(game_cfg.obs_fine_size),
        mid_size=int(game_cfg.obs_coarse_size),
        wide_size=int(game_cfg.obs_coarse_size),
        scales=(
            int(game_cfg.obs_scales[0]),
            int(game_cfg.obs_scales[1]),
            int(game_cfg.obs_scales[2]),
        ),
        scalar_dim=NUM_SCALAR_FEATURES,
        num_actions=int(game_cfg.num_actions),
    )


DEFAULT_OBS_LAYOUT: ObsLayout = obs_layout_from_game_config(_DEFAULT_GAME_CONFIG)
FINE_RASTER_SIZE: int = DEFAULT_OBS_LAYOUT.fine_size
MID_RASTER_SIZE: int = DEFAULT_OBS_LAYOUT.mid_size
WIDE_RASTER_SIZE: int = DEFAULT_OBS_LAYOUT.wide_size
RASTER_S1_BYTES: int = DEFAULT_OBS_LAYOUT.raster_s1_bytes
RASTER_S4_BYTES: int = DEFAULT_OBS_LAYOUT.raster_s4_bytes
RASTER_S16_BYTES: int = DEFAULT_OBS_LAYOUT.raster_s16_bytes
OBS_RASTER_BYTES_PER_STEP: int = DEFAULT_OBS_LAYOUT.raster_bytes_per_step
OBS_METADATA_BYTES_PER_STEP: int = DEFAULT_OBS_LAYOUT.metadata_bytes_per_step
OBS_BATCH_BYTES_PER_ROW: int = DEFAULT_OBS_LAYOUT.obs_batch_bytes_per_row
BYTES_PER_LEARNER_STEP: int = DEFAULT_OBS_LAYOUT.bytes_per_learner_step

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

# Train-mode wall-guard headroom constants (`prepare --mode train` only; canary & ruler untouched):
#   headroom = setup_allowance + final_eval_and_save_allowance + final_export_bytes / download_rate
TRAIN_SETUP_ALLOWANCE_SECONDS: float = 120.0
TRAIN_FINAL_EVAL_AND_SAVE_ALLOWANCE_SECONDS: float = 300.0
TRAIN_FINAL_SAVE_AFTER_EVAL_ALLOWANCE_SECONDS: float = 150.0
ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS: float = 1.0  # 1.0 MB/s (1,000,000 B/s)
ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_BYTES_PER_SEC: float = (
    ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS * 1_000_000.0
)
MAX_TRAIN_HEADROOM_SESSION_FRACTION: float = 0.40

# Calibrated against measured A100 GameV3Sim footprint at (R=600, N=24, max_length=2048):
# exact persistent world grids+rings = 28,779,936 bytes (27.45 MiB/world; measured 27.4 MiB/world);
# measured peak = ~31.0 MiB/world.
_SIM_PERSISTENT_OVERHEAD_FACTOR = 1.0
_SIM_PEAK_OVER_PERSISTENT_FACTOR = 31.0 / 27.4467
_CUDA_RUNTIME_RESERVE_BYTES = 1_200 * 1024 * 1024  # ~1.20 GiB CUDA context + allocator reserve
_OBS_BUILD_TEMP_BYTES = 256 * 1024 * 1024  # chunked `build_obs` scratch headroom


def default_minibatch_size(net_size: NetSize) -> int:
    """Return default PPO learner minibatch row count for ``net_size``."""
    return get_net_preset(net_size).default_minibatch_size


def default_actor_chunk_learners(net_size: NetSize) -> int:
    """Return default per-tick forward chunk size (in learner rows) for ``net_size``."""
    return get_net_preset(net_size).default_actor_chunk_learners


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
        get_net_preset(self.net_size)
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


RESUME_MUTABLE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "total_updates",
        "light_checkpoint_interval_sec",
        "full_checkpoint_interval_sec",
        "max_periodic_full_exports",
        "light_resume_margin_minutes",
        "checkpoint_shard_bytes",
        "checkpoint_every_updates",
        "full_checkpoint_every_updates",
        "keep_full_checkpoints",
        "keep_light_checkpoints",
        "max_in_session_restarts",
        "max_update_seconds",
        "eval_every_minutes",
        "eval_every_updates",
        "eval_envs",
        "eval_burnin_ticks",
        "eval_ticks",
        "eval_estimated_seconds",
        "eval_learner_slots",
        "eval_seed",
        "eval_greedy",
        "actor_chunk_learners",
        "compile_model",
        "fast_learner_ops",
    }
)


def immutable_config_diff(
    reference: V3PPOConfig,
    candidate: V3PPOConfig,
) -> dict[str, tuple[Any, Any]]:
    """Return ``{key: (reference_val, candidate_val)}`` for mismatched resume-immutable keys."""
    ref_dict = reference.to_dict()
    cand_dict = candidate.to_dict()
    diffs: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(ref_dict) | set(cand_dict)):
        if key in RESUME_MUTABLE_CONFIG_KEYS or key == "schema":
            continue
        if ref_dict.get(key) != cand_dict.get(key):
            diffs[key] = (ref_dict.get(key), cand_dict.get(key))
    return diffs


def assert_resume_config_compatible(
    reference: V3PPOConfig,
    candidate: V3PPOConfig,
    *,
    context: str = "Checkpoint config mismatch",
    reference_label: str = "ckpt",
    candidate_label: str = "current",
) -> None:
    """Raise ``ValueError`` if ``candidate`` differs from ``reference`` on any immutable key."""
    diffs = immutable_config_diff(reference, candidate)
    if diffs:
        key, (ref_val, cand_val) = next(iter(diffs.items()))
        raise ValueError(
            f"{context} on {key!r}: {reference_label}={ref_val!r} vs "
            f"{candidate_label}={cand_val!r}"
        )


def parse_env_bool(env_name: str, raw: str | None) -> bool | None:
    """Parse an optional boolean environment variable (`0/1/false/true`) or return ``None``."""
    if raw is None or raw.strip() == "":
        return None
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean environment variable {env_name}={raw!r}; expected '0' or '1'"
    )


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
    game_cfg: GameV3Config = _DEFAULT_GAME_CONFIG,
) -> dict[str, int]:
    """Compute exact byte counts for ``Uint8RolloutBuffer`` and the 1-step bootstrap cache."""
    layout = obs_layout_from_game_config(game_cfg)
    m = int(num_envs) * int(learner_slots)
    total_steps = m * int(rollout_len)
    raster_bytes = total_steps * layout.raster_bytes_per_step
    rollout_buffer_bytes = total_steps * layout.bytes_per_learner_step
    bootstrap_obs_bytes = m * layout.obs_batch_bytes_per_row
    return {
        "total_learners": m,
        "learner_steps_per_rollout": total_steps,
        "bytes_per_learner_step_rasters": layout.raster_bytes_per_step,
        "bytes_per_learner_step_total": layout.bytes_per_learner_step,
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
    preset = get_net_preset(net_size)
    # fp32 weights + fp32 grads + Adam m/v = 16 bytes/param
    model_and_opt_bytes = preset.memory_model_param_count * 16
    # Conv stem + fusion activations + float raster copy per minibatch row (1.8x in fp32)
    act_per_row = preset.learner_act_bytes_per_row_bf16
    if not amp:
        act_per_row = int(act_per_row * 1.8)
    activation_bytes = int(minibatch_size) * act_per_row
    return int(model_and_opt_bytes + activation_bytes)


def estimate_actor_chunk_temp_bytes(
    *,
    actor_chunk_rows: int,
    net_size: NetSize,
    amp: bool = True,
    game_cfg: GameV3Config = _DEFAULT_GAME_CONFIG,
) -> int:
    """Estimate transient bf16/fp32 raster conversion + stem activations for 1 forward chunk."""
    layout = obs_layout_from_game_config(game_cfg)
    preset = get_net_preset(net_size)
    rows = max(1, int(actor_chunk_rows))
    elem_bytes = 2 if amp else 4
    raster_float_copy_bytes = rows * layout.raster_bytes_per_step * elem_bytes
    fwd_act_per_row = preset.actor_fwd_act_bytes_per_row_bf16
    if not amp:
        fwd_act_per_row *= 2
    return int(raster_float_copy_bytes + rows * fwd_act_per_row)


def estimate_eval_overhead_bytes(cfg: V3PPOConfig) -> int:
    """Estimate transient VRAM allocated when periodic evaluation runs inside training."""
    eval_enabled = (int(cfg.eval_every_updates) > 0) or (float(cfg.eval_every_minutes) > 0.0)
    if not eval_enabled:
        return 0
    layout = obs_layout_from_game_config(cfg.game_config)
    eval_sim = estimate_sim_memory_bytes(cfg.game_config, cfg.eval_envs)
    eval_learners = int(cfg.eval_envs) * int(cfg.resolved_eval_learner_slots)
    eval_obs_bytes = eval_learners * layout.obs_batch_bytes_per_row
    return int(eval_sim["sim_peak_bytes"] + eval_obs_bytes)


def estimate_training_memory(cfg: V3PPOConfig, *, gpu: str | None = None) -> dict[str, Any]:
    """Return complete memory breakdown for ``cfg`` (including actor chunk & eval) and GPU fit."""
    sim_mem = estimate_sim_memory_bytes(cfg.game_config, cfg.num_envs)
    buf_mem = estimate_rollout_buffer_bytes(
        num_envs=cfg.num_envs,
        learner_slots=cfg.resolved_learner_slots,
        rollout_len=cfg.rollout_len,
        game_cfg=cfg.game_config,
    )
    eff_mb = cfg.resolved_minibatch_size
    preset = get_net_preset(cfg.net_size)
    model_and_opt_bytes = preset.memory_model_param_count * 16
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
        game_cfg=cfg.game_config,
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
            game_cfg=cfg.game_config,
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
    est_light_bytes = get_net_preset(cfg.net_size).est_light_checkpoint_bytes

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


def estimate_train_wall_headroom(
    cfg: V3PPOConfig,
    *,
    gpu: str = "L4",
    session_seconds: float = 4.0 * 3600.0,
    canary_eval_wall_seconds: float | None = None,
    measured_packed_bytes_per_world: int | None = None,
    assumed_download_rate_mbps: float = ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS,
) -> dict[str, Any]:
    """Size ``prepare --mode train`` headroom between ``V3_PPO_MAX_WALL_SECONDS`` and timeout.

    Sizes headroom as the sum of three terms:
      1. ``setup_allowance_seconds`` (`120.0` s),
      2. ``final_eval_and_save_allowance_seconds`` (`300.0` s default, or
         ``canary_eval_wall_seconds + 150.0`` s when provided by the bound canary receipt),
      3. ``final_export_download_seconds``: ``(final_full_bytes + final_light_bytes)`` from
         :func:`estimate_checkpoint_export_caps`'s per-world size model divided by the assumed
         download rate (`1.0 MB/s` = `1,000,000 B/s` default).
    """
    if not math.isfinite(session_seconds) or float(session_seconds) <= 0.0:
        raise ValueError(f"session_seconds must be finite and > 0, got {session_seconds}")
    if not math.isfinite(assumed_download_rate_mbps) or float(assumed_download_rate_mbps) <= 0.0:
        raise ValueError(
            f"assumed_download_rate_mbps must be finite and > 0, got {assumed_download_rate_mbps}"
        )
    if canary_eval_wall_seconds is not None and (
        not math.isfinite(canary_eval_wall_seconds) or float(canary_eval_wall_seconds) < 0.0
    ):
        raise ValueError(
            f"canary_eval_wall_seconds must be finite and >= 0, got {canary_eval_wall_seconds}"
        )

    caps = estimate_checkpoint_export_caps(
        cfg,
        gpu=gpu,
        max_wall_seconds=float(session_seconds),
        measured_packed_bytes_per_world=measured_packed_bytes_per_world,
    )
    setup_sec = float(TRAIN_SETUP_ALLOWANCE_SECONDS)
    if canary_eval_wall_seconds is not None:
        eval_save_sec = float(canary_eval_wall_seconds) + float(
            TRAIN_FINAL_SAVE_AFTER_EVAL_ALLOWANCE_SECONDS
        )
        eval_save_source = "canary_receipt_eval_plus_150s"
    else:
        eval_save_sec = float(TRAIN_FINAL_EVAL_AND_SAVE_ALLOWANCE_SECONDS)
        eval_save_source = "default_300s"

    final_full_bytes = int(caps["est_full_sim_bytes"])
    final_light_bytes = int(caps["est_light_checkpoint_bytes"])
    final_export_bytes = final_full_bytes + final_light_bytes

    download_rate_bps = float(assumed_download_rate_mbps) * 1_000_000.0
    download_sec = float(final_export_bytes) / download_rate_bps
    headroom_sec = setup_sec + eval_save_sec + download_sec

    sess_sec = float(session_seconds)
    max_allowed_sec = float(MAX_TRAIN_HEADROOM_SESSION_FRACTION) * sess_sec
    headroom_fraction = headroom_sec / sess_sec
    fits_fraction = bool(headroom_sec <= max_allowed_sec)
    max_wall_seconds = int(math.floor(sess_sec - headroom_sec))

    return {
        "gpu": str(caps["gpu"]),
        "num_envs": int(cfg.num_envs),
        "net_size": cfg.net_size,
        "session_seconds": sess_sec,
        "setup_allowance_seconds": setup_sec,
        "final_eval_and_save_allowance_seconds": eval_save_sec,
        "final_eval_and_save_source": eval_save_source,
        "canary_eval_wall_seconds": (
            float(canary_eval_wall_seconds) if canary_eval_wall_seconds is not None else None
        ),
        "final_full_bytes": final_full_bytes,
        "final_light_bytes": final_light_bytes,
        "final_export_bytes": final_export_bytes,
        "budgeted_packed_world_bytes": int(caps["budgeted_packed_world_bytes"]),
        "assumed_download_rate_mbps": float(assumed_download_rate_mbps),
        "assumed_download_rate_bytes_per_sec": download_rate_bps,
        "final_export_download_seconds": download_sec,
        "headroom_seconds": headroom_sec,
        "headroom_minutes": round(headroom_sec / 60.0, 2),
        "headroom_fraction": round(headroom_fraction, 6),
        "max_headroom_fraction": float(MAX_TRAIN_HEADROOM_SESSION_FRACTION),
        "max_allowed_headroom_seconds": max_allowed_sec,
        "fits_max_headroom_fraction": fits_fraction,
        "max_wall_seconds": max_wall_seconds,
    }


def validate_train_wall_headroom(
    cfg: V3PPOConfig,
    gpu: str,
    *,
    session_seconds: float,
    canary_eval_wall_seconds: float | None = None,
    measured_packed_bytes_per_world: int | None = None,
    assumed_download_rate_mbps: float = ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS,
) -> dict[str, Any]:
    """Validate ``train`` wall-guard headroom fits ``<= 40%`` of ``session_seconds``."""
    est = estimate_train_wall_headroom(
        cfg,
        gpu=gpu,
        session_seconds=session_seconds,
        canary_eval_wall_seconds=canary_eval_wall_seconds,
        measured_packed_bytes_per_world=measured_packed_bytes_per_world,
        assumed_download_rate_mbps=assumed_download_rate_mbps,
    )
    if not est["fits_max_headroom_fraction"]:
        sess_hours = float(session_seconds) / 3600.0
        pct = 100.0 * float(est["headroom_seconds"]) / float(session_seconds)
        max_pct = 100.0 * float(est["max_headroom_fraction"])
        raise ValueError(
            f"Refusing train packet whose wall-guard headroom "
            f"({est['headroom_seconds']:.1f} s = {est['headroom_seconds'] / 60.0:.1f} min, "
            f"{pct:.1f}% of session: "
            f"setup={est['setup_allowance_seconds']:.1f} s + "
            f"final_eval_and_save={est['final_eval_and_save_allowance_seconds']:.1f} s + "
            f"final_export_download={est['final_export_download_seconds']:.1f} s for "
            f"{est['final_export_bytes'] / 1e6:.1f} MB at "
            f"{est['assumed_download_rate_mbps']:.2f} MB/s) "
            f"exceeds {max_pct:.0f}% of the {sess_hours:g} h session "
            f"({est['max_allowed_headroom_seconds']:.1f} s max on {float(session_seconds):.0f} s "
            f"timeout; worlds={cfg.num_envs}, net_size={cfg.net_size}, gpu={est['gpu']}). "
            "Increase --hours, reduce --num-envs, or supply --measured-packed-bytes-per-world "
            "from a verified run."
        )
    return est
