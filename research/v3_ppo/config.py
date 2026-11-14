"""Configuration, VRAM sizing, and checkpoint export caps for Game V3.1 PPO (`research/v3_ppo`)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
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


OpponentPolicy = Literal[
    "food_seeking",
    "random_safe",
    "cardinal6_safe",
    "head_evader",
    "cut_off",
]
VALID_OPPONENT_POLICIES: tuple[OpponentPolicy, ...] = (
    "food_seeking",
    "random_safe",
    "cardinal6_safe",
    "head_evader",
    "cut_off",
)
DEFAULT_TRAIN_EVAL_OPPONENTS: tuple[OpponentPolicy, ...] = ("food_seeking", "random_safe")


def parse_opponent_policies(
    value: str | Sequence[str],
    *,
    field_name: str = "eval_opponents",
    allow_empty: bool = False,
) -> tuple[OpponentPolicy, ...]:
    """Parse and validate a CSV string or sequence of :data:`VALID_OPPONENT_POLICIES`."""
    if isinstance(value, str):
        stripped = value.strip()
        if allow_empty and stripped.lower() in {"", "none"}:
            return ()
        raw_items: list[Any] = [tok.strip() for tok in stripped.split(",") if tok.strip()]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raise ConfigTypeError(
            f"{field_name} must be a comma-separated string or list/tuple of strings, "
            f"got {type(value)!r}"
        )

    if not raw_items and not allow_empty:
        raise ValueError(f"{field_name} must contain at least 1 opponent policy")

    out: list[OpponentPolicy] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_items):
        if not isinstance(item, str):
            raise ConfigTypeError(f"{field_name}[{idx}] must be a str, got {type(item)!r}")
        tok = item.strip()
        if tok not in VALID_OPPONENT_POLICIES:
            raise ValueError(
                f"{field_name}[{idx}] must be one of {VALID_OPPONENT_POLICIES}, got {item!r}"
            )
        if tok in seen:
            raise ValueError(f"Duplicate opponent policy {tok!r} in {field_name}: {raw_items}")
        seen.add(tok)
        out.append(tok)  # type: ignore[arg-type]
    return tuple(out)


FullExportMode = Literal["none", "final", "periodic"]
VALID_FULL_EXPORT_MODES: tuple[FullExportMode, ...] = ("none", "final", "periodic")

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
# In `--full-exports none` (default), simulator FULL shards are not exported, so the session export
# cap shrinks to a 4.0 GiB LIGHT-only ceiling (49 * 68 MiB M LIGHTs + 64 MiB telemetry = 3.32 GiB).
NONE_MODE_MAX_TOTAL_EXPORT_BYTES: int = 4 * 1024 * 1024 * 1024  # 4.0 GiB LIGHT-only session cap
GPU_MAX_TOTAL_EXPORT_BYTES: dict[str, int] = {
    "L4": 12 * 1024 * 1024 * 1024,  # 12.0 GiB session cap (E=128, 24h M <= 11.32 GiB)
    "A100": 24 * 1024 * 1024 * 1024,  # 24.0 GiB session cap (E=256, 24h M <= 19.32 GiB)
    "H100": 40 * 1024 * 1024 * 1024,  # 40.0 GiB session cap (E=512, 24h M <= 35.33 GiB)
}

# Train-mode wall-guard headroom constants (`prepare --mode train` only; canary & ruler untouched):
#   headroom = setup_allowance (330 s, measured 240-325 s session-start-to-wall_t0)
#            + staged_input_bytes / upload_rate (0 s on fresh runs; prior LIGHT/FULL upload on continuations)
#            + final_eval_and_save_allowance
#            + final_export_bytes / download_rate
# Inside `V3_PPO_MAX_WALL_SECONDS = floor(timeout_s - headroom)`, `run_training_job` additionally
# holds `checkpoint_reserve_sec = min(TRAINER_CHECKPOINT_RESERVE_SECONDS, max(5 s, 0.08 * max_wall))`
# (`90 s` on sessions >= 18.75 min) so the active rollout/PPO update finishes before wrap-up starts.
TRAIN_SETUP_ALLOWANCE_SECONDS: float = 330.0
TRAINER_CHECKPOINT_RESERVE_SECONDS: float = 90.0
TRAIN_FINAL_EVAL_AND_SAVE_ALLOWANCE_SECONDS: float = 300.0
TRAIN_FINAL_SAVE_AFTER_EVAL_ALLOWANCE_SECONDS: float = 150.0
TRAIN_FINAL_TELEMETRY_EXPORT_BYTES: int = 4 * 1024 * 1024  # 4 MiB final metrics/eval/result JSONLs
ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS: float = 1.0  # 1.0 MB/s (1,000,000 B/s)
ASSUMED_STAGED_INPUT_UPLOAD_RATE_BYTES_PER_SEC: float = (
    ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS * 1_000_000.0
)
# Canary upload budget: Colab CLI transport caps canary uploads at 150 s; refusing > 100.0 s at
# ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS (1.0 MB/s) leaves 50 s margin for a hung upload call.
CANARY_MAX_ESTIMATED_UPLOAD_SECONDS: float = 100.0

# Eval & eval-canary upload-time model (`prepare --mode eval` and `prepare --mode canary` with eval
# policies). Mirrors the transport caps in `research/colab_cli/{policy,transfer_policy,canary,transfers}.py`
# (`prepare.py` passes the live constants read from `research/colab_cli/` directly):
#   - `DEFAULT_TRANSFER_SECONDS = MAX_UPLOAD_SECONDS = 150.0` s (`transfer_policy.py`, `canary.py`)
#   - `UPLOAD_CHUNK_BYTES = 2 MiB` (`transfers.py`; files <= 2 MiB use 1 `PUT` + 1 `VERIFY_FILE`
#     `exec`, files > 2 MiB use 2-worker 2 MiB chunk `PUT`s + 2 `exec`s: `CREATE_DIR` & `UPLOAD_ASSEMBLE`)
#   - `STAGING_CONTROL_SECONDS = 30.0` s, `BACKOFF_SECONDS = (2.0, 5.0)`,
#     `STAGING_RECOVERY_SECONDS = 120.0` s (`policy.py`)
#   - `supervisor.py` counts 2 fixed runtime files (`bootstrap.sh`, `source.py` -> 2 `VERIFY_FILE`
#     `exec`s, ~6.0 s clean) plus every staged input inside `transfer_seconds`.
#   - Measured on 2026-11-12 Colab CLI logs/receipts: ~8.5% (`p_hang = 0.085`) of `colab exec`
#     staging control calls hang (`30.0 s` timeout + `2.0 s` backoff + `~18.0 s` idle-helper probe
#     & retry = `50.0 s` per hung call, bounded by `STAGING_RECOVERY_SECONDS = 120.0 s`), while
#     2-worker `ContentsClient` chunk `PUT`s sustain `1.30 MB/s` (`59.33 MB` in `45.34 s`, `83.87 MB`
#     in `57.84 s`). Packet must survive 1 base hung control call (`50.0 s`) plus expected hang cost
#     across all `C = 2 + F_small + 2 * F_chunked` control calls (`0.085 * 50.0 * C = 4.25 * C` s).
EVAL_UPLOAD_CAP_SECONDS: float = 150.0
EVAL_UPLOAD_CHUNK_BYTES: int = 2 * 1024 * 1024
EVAL_UPLOAD_STAGING_CONTROL_SECONDS: float = 30.0
EVAL_UPLOAD_STAGING_RECOVERY_SECONDS: float = 120.0
EVAL_UPLOAD_CONTROL_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 5.0)
EVAL_UPLOAD_CONTROL_IDLE_RETRY_SECONDS: float = 18.0
EVAL_UPLOAD_CONTROL_HANG_PROB: float = 0.085
EVAL_UPLOAD_RUNTIME_FIXED_CONTROL_CALLS: int = 2
EVAL_UPLOAD_RUNTIME_FIXED_SECONDS: float = 6.0
EVAL_UPLOAD_SMALL_FILE_CONTROL_SECONDS: float = 3.0
EVAL_UPLOAD_CHUNKED_FILE_CONTROL_SECONDS: float = 5.0
EVAL_UPLOAD_CHUNK_RATE_MBPS: float = 1.3
EVAL_UPLOAD_CHUNK_RATE_BYTES_PER_SEC: float = EVAL_UPLOAD_CHUNK_RATE_MBPS * 1_000_000.0


def estimate_eval_upload_time(
    file_sizes_bytes: Sequence[int],
    *,
    transfer_cap_seconds: float = EVAL_UPLOAD_CAP_SECONDS,
    upload_chunk_bytes: int = EVAL_UPLOAD_CHUNK_BYTES,
    staging_control_seconds: float = EVAL_UPLOAD_STAGING_CONTROL_SECONDS,
    staging_recovery_seconds: float = EVAL_UPLOAD_STAGING_RECOVERY_SECONDS,
    backoff_first_seconds: float = EVAL_UPLOAD_CONTROL_BACKOFF_SECONDS[0],
    control_hang_prob: float = EVAL_UPLOAD_CONTROL_HANG_PROB,
    chunk_rate_bytes_per_sec: float = EVAL_UPLOAD_CHUNK_RATE_BYTES_PER_SEC,
) -> dict[str, Any]:
    """Estimate Colab CLI staging upload seconds from staged file sizes AND file count."""
    sizes = [int(sz) for sz in file_sizes_bytes]
    if any(sz < 0 for sz in sizes):
        raise ValueError(f"file_sizes_bytes must be non-negative, got {sizes}")
    if upload_chunk_bytes < 1:
        raise ValueError(f"upload_chunk_bytes must be >= 1, got {upload_chunk_bytes}")
    if chunk_rate_bytes_per_sec <= 0.0 or not math.isfinite(chunk_rate_bytes_per_sec):
        raise ValueError(
            f"chunk_rate_bytes_per_sec must be finite and > 0, got {chunk_rate_bytes_per_sec}"
        )
    if transfer_cap_seconds <= 0.0 or not math.isfinite(transfer_cap_seconds):
        raise ValueError(f"transfer_cap_seconds must be finite and > 0, got {transfer_cap_seconds}")

    total_bytes = int(sum(sizes))
    file_count = len(sizes)
    small_file_count = sum(1 for sz in sizes if sz <= upload_chunk_bytes)
    chunked_file_count = file_count - small_file_count
    staging_control_calls = int(
        EVAL_UPLOAD_RUNTIME_FIXED_CONTROL_CALLS + small_file_count + 2 * chunked_file_count
    )
    clean_transfer_seconds = (
        float(EVAL_UPLOAD_RUNTIME_FIXED_SECONDS)
        + float(EVAL_UPLOAD_SMALL_FILE_CONTROL_SECONDS) * float(small_file_count)
        + float(EVAL_UPLOAD_CHUNKED_FILE_CONTROL_SECONDS) * float(chunked_file_count)
        + float(total_bytes) / float(chunk_rate_bytes_per_sec)
    )
    single_hang_seconds = min(
        float(staging_recovery_seconds),
        float(staging_control_seconds)
        + float(backoff_first_seconds)
        + float(EVAL_UPLOAD_CONTROL_IDLE_RETRY_SECONDS),
    )
    expected_hang_seconds = float(control_hang_prob) * single_hang_seconds * staging_control_calls
    hang_allowance_seconds = single_hang_seconds + expected_hang_seconds
    estimated_upload_seconds = clean_transfer_seconds + hang_allowance_seconds
    cap_sec = float(transfer_cap_seconds)
    return {
        "total_bytes": total_bytes,
        "file_count": file_count,
        "small_file_count": small_file_count,
        "chunked_file_count": chunked_file_count,
        "staging_control_calls": staging_control_calls,
        "clean_transfer_seconds": clean_transfer_seconds,
        "single_hang_seconds": single_hang_seconds,
        "expected_hang_seconds": expected_hang_seconds,
        "hang_allowance_seconds": hang_allowance_seconds,
        "estimated_upload_seconds": estimated_upload_seconds,
        "transfer_cap_seconds": cap_sec,
        "headroom_seconds": cap_sec - estimated_upload_seconds,
        "fits_transfer_cap": bool(estimated_upload_seconds <= cap_sec),
        "chunk_rate_mbps": float(chunk_rate_bytes_per_sec) / 1_000_000.0,
        "control_hang_prob": float(control_hang_prob),
    }


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


class ConfigTypeError(TypeError, ValueError):
    """Raised when a ``V3PPOConfig`` field receives a value of the wrong type."""


RECIPE_OVERRIDE_KEYS: tuple[str, ...] = (
    "reward_alive_length_coef",
    "reward_alive_const",
    "reward_length_delta_coef",
    "reward_kill_coef",
    "reward_death_penalty",
    "gamma",
    "gae_lambda",
    "entropy_coef",
    "lr",
    "ppo_epochs",
    "minibatch_size",
    "rollout_len",
    "clip_ratio",
)

_V3_PPO_INT_FIELDS: tuple[str, ...] = (
    "num_envs",
    "rollout_len",
    "ppo_epochs",
    "total_updates",
    "max_periodic_full_exports",
    "checkpoint_shard_bytes",
    "checkpoint_every_updates",
    "full_checkpoint_every_updates",
    "keep_full_checkpoints",
    "keep_light_checkpoints",
    "max_in_session_restarts",
    "eval_every_updates",
    "eval_envs",
    "eval_burnin_ticks",
    "eval_ticks",
    "eval_seed",
    "seed",
)

_V3_PPO_OPTIONAL_INT_FIELDS: tuple[str, ...] = (
    "learner_slots",
    "minibatch_size",
    "actor_chunk_learners",
    "eval_learner_slots",
)

_V3_PPO_FLOAT_FIELDS: tuple[str, ...] = (
    "gamma",
    "gae_lambda",
    "lr",
    "adam_eps",
    "weight_decay",
    "clip_ratio",
    "entropy_coef",
    "value_coef",
    "max_grad_norm",
    "reward_alive_length_coef",
    "reward_alive_const",
    "reward_length_delta_coef",
    "reward_kill_coef",
    "reward_death_penalty",
    "light_checkpoint_interval_sec",
    "full_checkpoint_interval_sec",
    "light_resume_margin_minutes",
    "eval_every_minutes",
    "eval_estimated_seconds",
)

_V3_PPO_BOOL_FIELDS: tuple[str, ...] = (
    "amp",
    "use_channels_last",
    "compile_model",
    "fast_learner_ops",
    "normalize_advantages",
    "eval_greedy",
    "deterministic",
)


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
    # `light_checkpoint_interval_sec` default 30 min; `full_exports` defaults to `"none"` so no
    # simulator FULL shards are exported to `output/` while `work/` FULL saves still run for
    # in-session restart; `"final"` exports only the final boundary FULL; `"periodic"` exports
    # periodic FULLs bounded by `max_periodic_full_exports` plus 1 at cooperative stop / wall
    # guard; `keep_*_checkpoints` prune `work/` only), and in-session recovery.
    total_updates: int = 100
    light_checkpoint_interval_sec: float = 1800.0
    full_checkpoint_interval_sec: float = 7200.0
    full_exports: FullExportMode = "none"
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
    eval_opponents: tuple[OpponentPolicy, ...] = DEFAULT_TRAIN_EVAL_OPPONENTS

    seed: int = 20261111
    deterministic: bool = False
    recipe_change_history: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.game_config, GameV3Config):
            raise ConfigTypeError(
                f"game_config must be a GameV3Config, got {type(self.game_config)!r}"
            )
        for int_name in _V3_PPO_INT_FIELDS:
            iv = getattr(self, int_name)
            if isinstance(iv, bool) or not isinstance(iv, int):
                raise ConfigTypeError(f"{int_name} must be an int, got {iv!r}")
        for opt_int_name in _V3_PPO_OPTIONAL_INT_FIELDS:
            oiv = getattr(self, opt_int_name)
            if oiv is not None and (isinstance(oiv, bool) or not isinstance(oiv, int)):
                raise ConfigTypeError(f"{opt_int_name} must be an int or None, got {oiv!r}")
        for float_name in _V3_PPO_FLOAT_FIELDS:
            fv = getattr(self, float_name)
            if isinstance(fv, bool) or not isinstance(fv, (int, float)):
                raise ConfigTypeError(f"{float_name} must be a float, got {fv!r}")
            if isinstance(fv, int):
                object.__setattr__(self, float_name, float(fv))
        if self.max_update_seconds is not None:
            if isinstance(self.max_update_seconds, bool) or not isinstance(
                self.max_update_seconds, (int, float)
            ):
                raise ConfigTypeError(
                    f"max_update_seconds must be a float or None, got {self.max_update_seconds!r}"
                )
            if isinstance(self.max_update_seconds, int):
                object.__setattr__(self, "max_update_seconds", float(self.max_update_seconds))
        for bool_name in _V3_PPO_BOOL_FIELDS:
            bv = getattr(self, bool_name)
            if not isinstance(bv, bool):
                raise ConfigTypeError(f"{bool_name} must be a bool, got {bv!r}")
        for str_name in ("opponent_policy", "net_size", "full_exports"):
            sv = getattr(self, str_name)
            if not isinstance(sv, str):
                raise ConfigTypeError(f"{str_name} must be a str, got {sv!r}")
        norm_eval_opponents = parse_opponent_policies(
            self.eval_opponents, field_name="eval_opponents", allow_empty=False
        )
        object.__setattr__(self, "eval_opponents", norm_eval_opponents)
        if not isinstance(self.recipe_change_history, (list, tuple)):
            raise ConfigTypeError(
                "recipe_change_history must be a list or tuple, "
                f"got {type(self.recipe_change_history)!r}"
            )
        norm_history: list[dict[str, Any]] = []
        allowed_recipe_set = set(RECIPE_OVERRIDE_KEYS)
        for idx_h, entry in enumerate(self.recipe_change_history):
            if not isinstance(entry, Mapping):
                raise ConfigTypeError(
                    f"recipe_change_history[{idx_h}] must be a mapping, got {type(entry)!r}"
                )
            extra_entry_keys = sorted(
                str(k) for k in entry.keys() if k not in {"update", "changed_fields"}
            )
            if extra_entry_keys:
                raise ValueError(
                    f"Unsupported keys in recipe_change_history[{idx_h}]: {extra_entry_keys}"
                )
            upd = entry.get("update")
            if isinstance(upd, bool) or not isinstance(upd, int) or upd < 0:
                raise ValueError(
                    f"recipe_change_history[{idx_h}].update must be an int >= 0, got {upd!r}"
                )
            ch_fields = entry.get("changed_fields")
            if not isinstance(ch_fields, Mapping) or not ch_fields:
                raise ValueError(
                    f"recipe_change_history[{idx_h}].changed_fields must be a non-empty mapping"
                )
            norm_ch: dict[str, dict[str, Any]] = {}
            for ch_k, ch_v in sorted(ch_fields.items()):
                if ch_k not in allowed_recipe_set:
                    raise ValueError(
                        f"Unsupported key {ch_k!r} in recipe_change_history[{idx_h}].changed_fields"
                    )
                if not isinstance(ch_v, Mapping) or set(ch_v.keys()) != {"old", "new"}:
                    raise ValueError(
                        f"recipe_change_history[{idx_h}].changed_fields[{ch_k!r}] must map to "
                        "exactly {'old': ..., 'new': ...}"
                    )
                old_val = ch_v["old"]
                new_val = ch_v["new"]
                _validate_recipe_field_value(
                    str(ch_k),
                    old_val,
                    context=f"recipe_change_history[{idx_h}].changed_fields[{ch_k!r}].old",
                )
                _validate_recipe_field_value(
                    str(ch_k),
                    new_val,
                    context=f"recipe_change_history[{idx_h}].changed_fields[{ch_k!r}].new",
                )
                if old_val == new_val and type(old_val) is type(new_val):
                    raise ValueError(
                        f"recipe_change_history[{idx_h}].changed_fields[{ch_k!r}] has identical "
                        f"old and new values ({old_val!r})"
                    )
                norm_ch[str(ch_k)] = {"old": old_val, "new": new_val}
            norm_history.append({"update": int(upd), "changed_fields": norm_ch})
        object.__setattr__(self, "recipe_change_history", tuple(norm_history))

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
        if float(self.entropy_coef) < 0.0:
            raise ValueError(f"entropy_coef must be >= 0, got {self.entropy_coef}")
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
        if self.full_exports not in VALID_FULL_EXPORT_MODES:
            raise ValueError(
                f"full_exports must be one of {VALID_FULL_EXPORT_MODES}, got {self.full_exports!r}"
            )
        if int(self.max_periodic_full_exports) < 0:
            raise ValueError(
                f"max_periodic_full_exports must be >= 0, got {self.max_periodic_full_exports}"
            )
        if self.full_exports == "periodic" and int(self.max_periodic_full_exports) < 1:
            raise ValueError(
                "full_exports='periodic' requires max_periodic_full_exports >= 1 "
                "(use full_exports='final' for 0 periodic FULL exports)"
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

    @property
    def exports_final_full(self) -> bool:
        """Whether final boundary FULL simulator checkpoints are exported to ``output/``."""
        return self.full_exports in ("final", "periodic")

    @property
    def effective_max_periodic_full_exports(self) -> int:
        """Effective periodic FULL export cap in ``output/`` under ``self.full_exports``."""
        if self.full_exports == "periodic":
            return int(self.max_periodic_full_exports)
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize config (including nested ``GameV3Config``) to plain JSON dict."""
        out = asdict(self)
        out["schema"] = V3_PPO_SCHEMA
        out["game_config"] = self.game_config.to_dict()
        if self.recipe_change_history:
            out["recipe_change_history"] = [dict(entry) for entry in self.recipe_change_history]
        else:
            out.pop("recipe_change_history", None)
        if tuple(self.eval_opponents) != DEFAULT_TRAIN_EVAL_OPPONENTS:
            out["eval_opponents"] = list(self.eval_opponents)
        else:
            out.pop("eval_opponents", None)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V3PPOConfig:
        """Deserialize config produced by :meth:`to_dict`."""
        if not isinstance(data, Mapping):
            raise ConfigTypeError(f"V3PPOConfig.from_dict expects a mapping, got {type(data)!r}")
        raw = dict(data)
        raw.pop("schema", None)
        gc_raw = raw.pop("game_config", None)
        if gc_raw is None:
            game_cfg = GameV3Config()
        elif isinstance(gc_raw, GameV3Config):
            game_cfg = gc_raw
        elif isinstance(gc_raw, dict):
            game_cfg = GameV3Config.from_dict(gc_raw)
        else:
            raise ConfigTypeError(
                f"game_config must be a mapping or GameV3Config, got {type(gc_raw)!r}"
            )
        valid_fields = {f.name for f in fields(cls)} - {"game_config"}
        unknown = sorted(str(k) for k in raw.keys() if k not in valid_fields)
        if unknown:
            raise ConfigTypeError(f"Unknown V3PPOConfig keys: {unknown}")
        return cls(game_config=game_cfg, **raw)


def parse_recipe_overrides(
    source: str | Path | Mapping[str, Any] | None,
    *,
    source_label: str = "--recipe-json",
) -> dict[str, Any]:
    """Parse and allow-list validate recipe overrides from JSON text, a file path, or a mapping."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        parsed = dict(source)
    elif isinstance(source, Path):
        if not source.is_file():
            raise ValueError(f"{source_label} file does not exist: {source}")
        parsed = json.loads(source.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("{"):
            parsed = json.loads(source)
        else:
            cand_path = Path(source)
            if cand_path.is_file():
                parsed = json.loads(cand_path.read_text(encoding="utf-8"))
            else:
                try:
                    parsed = json.loads(source)
                except Exception as exc:
                    raise ValueError(
                        f"{source_label} must be a JSON object string or an existing JSON file "
                        f"path, got {source!r}"
                    ) from exc
    else:
        raise ConfigTypeError(
            f"{source_label} must be a JSON string, Path, or mapping, got {type(source)!r}"
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{source_label} must decode to a JSON object, got {type(parsed).__name__}"
        )

    allowed = set(RECIPE_OVERRIDE_KEYS)
    unknown = sorted(str(k) for k in parsed.keys() if k not in allowed)
    if unknown:
        raise ValueError(
            f"Unsupported recipe override keys in {source_label}: {unknown}; "
            f"allowed keys are {list(RECIPE_OVERRIDE_KEYS)}"
        )

    return {key: parsed[key] for key in RECIPE_OVERRIDE_KEYS if key in parsed}


def _validate_recipe_field_value(field_name: str, value: Any, *, context: str) -> None:
    """Validate a single allow-listed recipe field value against ``V3PPOConfig`` invariants."""
    try:
        V3PPOConfig(**{field_name: value})
    except ConfigTypeError as exc:
        raise ConfigTypeError(f"Invalid {context}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc


EVAL_POLICY_CONFIG_KEYS: tuple[str, ...] = (
    # Compatibility, network construction, and inference execution fields
    "net_size",
    "amp",
    "use_channels_last",
    "fast_learner_ops",
    "actor_chunk_learners",
    # Full training setup and recipe fields for reporting
    "num_envs",
    "learner_slots",
    "opponent_policy",
    *RECIPE_OVERRIDE_KEYS,
    "adam_eps",
    "weight_decay",
    "value_coef",
    "max_grad_norm",
    "normalize_advantages",
    "seed",
    "deterministic",
)


def extract_eval_policy_config_dict(cfg: V3PPOConfig) -> dict[str, Any]:
    """Return the evaluator-minimal ``V3PPOConfig`` subset for policy-only checkpoints and reports.

    Includes only the fields required by ``arena_eval`` to rebuild :class:`EgoMultiScaleNet` and
    verify compatibility (``game_config``, ``net_size``, ``amp``, ``use_channels_last``,
    ``fast_learner_ops``, ``actor_chunk_learners``) plus the full training recipe and lineage
    fields used in evaluation reports, omitting operational checkpoint/schedule/periodic-eval keys.
    """
    out: dict[str, Any] = {
        "schema": V3_PPO_SCHEMA,
        "game_config": cfg.game_config.to_dict(),
    }
    for key in EVAL_POLICY_CONFIG_KEYS:
        out[key] = getattr(cfg, key)
    if cfg.recipe_change_history:
        out["recipe_change_history"] = [dict(entry) for entry in cfg.recipe_change_history]
    return out


def extract_recipe_dict(cfg: V3PPOConfig) -> dict[str, Any]:
    """Return all allow-listed recipe fields of ``cfg`` in canonical ``RECIPE_OVERRIDE_KEYS`` order."""
    return {key: getattr(cfg, key) for key in RECIPE_OVERRIDE_KEYS}


def recipe_overrides_from_default(
    cfg: V3PPOConfig,
    *,
    reference: V3PPOConfig | None = None,
) -> dict[str, Any]:
    """Return allow-listed recipe fields whose effective value on ``cfg`` differs from defaults."""
    ref = reference if reference is not None else V3PPOConfig(net_size=cfg.net_size)
    out: dict[str, Any] = {}
    for key in RECIPE_OVERRIDE_KEYS:
        if key == "minibatch_size":
            if int(cfg.resolved_minibatch_size) != int(ref.resolved_minibatch_size):
                out[key] = int(cfg.resolved_minibatch_size)
        else:
            if getattr(cfg, key) != getattr(ref, key):
                out[key] = getattr(cfg, key)
    return out


def format_recipe_overrides(overrides: Mapping[str, Any] | None) -> str:
    """Format recipe overrides for summary/log output (`'none'` when empty)."""
    if not overrides:
        return "none"
    return ", ".join(f"{key}={json.dumps(overrides[key])}" for key in sorted(overrides.keys()))


MAX_DROPPED_MINIBATCH_ROW_FRACTION: float = 0.25


def check_minibatch_divisibility(cfg: V3PPOConfig) -> dict[str, Any] | None:
    """Check ``rows_per_update`` divisibility by ``resolved_minibatch_size``.

    Returns ``None`` when ``rows_per_update`` is an exact multiple of ``minibatch_size``.
    Otherwise returns a summary dictionary describing the dropped (or zero-loss-padded) rows per
    PPO epoch, and raises ``ValueError`` when strictly more than ``25%`` of ``rows_per_update``
    would be dropped per epoch.
    """
    rows_per_update = int(cfg.rollout_len) * int(cfg.total_learners)
    mb_size = int(cfg.resolved_minibatch_size)
    if rows_per_update % mb_size == 0:
        return None
    if rows_per_update < mb_size:
        dropped_rows = 0
        padded_rows = mb_size - rows_per_update
    else:
        dropped_rows = rows_per_update % mb_size
        padded_rows = 0

    dropped_fraction = float(dropped_rows) / float(rows_per_update)
    dropped_pct = 100.0 * dropped_fraction
    if dropped_rows * 4 > rows_per_update:
        raise ValueError(
            f"Refusing train packet: rows_per_update={rows_per_update} "
            f"(rollout_len={cfg.rollout_len} * total_learners={cfg.total_learners}) is not "
            f"divisible by minibatch_size={mb_size} and would drop {dropped_rows}/{rows_per_update} "
            f"rows ({dropped_pct:.2f}%) per PPO epoch, exceeding the 25% cap."
        )
    return {
        "rows_per_update": rows_per_update,
        "minibatch_size": mb_size,
        "dropped_rows_per_epoch": dropped_rows,
        "dropped_fraction": round(dropped_fraction, 6),
        "dropped_percent": round(dropped_pct, 2),
        "padded_rows_per_epoch": padded_rows,
    }


def format_minibatch_remainder_warning(info: Mapping[str, Any]) -> str:
    """Format a single-line warning message for non-divisible ``rows_per_update``."""
    rows = int(info["rows_per_update"])
    mb = int(info["minibatch_size"])
    dropped = int(info["dropped_rows_per_epoch"])
    pct = float(info["dropped_percent"])
    padded = int(info.get("padded_rows_per_epoch", 0))
    if padded > 0:
        return (
            f"WARNING [V3_PPO_MINIBATCH_REMAINDER]: rows_per_update={rows} is not divisible by "
            f"minibatch_size={mb}; {dropped}/{rows} rows ({pct:.2f}%) dropped per PPO epoch "
            f"(batch padded with {padded} zero-masked rows to {mb})."
        )
    return (
        f"WARNING [V3_PPO_MINIBATCH_REMAINDER]: rows_per_update={rows} is not divisible by "
        f"minibatch_size={mb}; dropping {dropped}/{rows} rows ({pct:.2f}%) per PPO epoch."
    )


def resolve_full_export_settings(
    full_exports: str | None,
    max_periodic_full_exports: int | None,
    *,
    default_full_exports: FullExportMode = "none",
    default_max_periodic: int = 3,
) -> tuple[FullExportMode, int]:
    """Resolve ``(full_exports, max_periodic_full_exports)`` and refuse conflicting explicit flags.

    Rules:
      - ``--full-exports none`` (default): zero periodic FULL exports and zero final FULL exports.
      - ``--full-exports final``: zero periodic FULL exports + 1 final FULL export (equivalent to
        ``--max-periodic-full-exports 0``).
      - ``--full-exports periodic``: periodic FULL exports (bounded by
        ``max_periodic_full_exports >= 1``) + 1 final FULL export.
      - When only ``--max-periodic-full-exports`` is explicitly provided, ``0`` maps to
        ``("final", 0)`` and ``k >= 1`` maps to ``("periodic", k)``.
      - When both are explicitly provided and disagree, raises ``ValueError``.
    """
    if full_exports is not None and full_exports not in VALID_FULL_EXPORT_MODES:
        raise ValueError(
            f"full_exports must be one of {VALID_FULL_EXPORT_MODES}, got {full_exports!r}"
        )
    if max_periodic_full_exports is not None and int(max_periodic_full_exports) < 0:
        raise ValueError(f"max_periodic_full_exports must be >= 0, got {max_periodic_full_exports}")

    if full_exports is not None and max_periodic_full_exports is not None:
        mp = int(max_periodic_full_exports)
        if full_exports == "none":
            implied = "final" if mp == 0 else "periodic"
            raise ValueError(
                "Conflicting FULL export options: --full-exports none disables all FULL "
                f"simulator exports, whereas --max-periodic-full-exports {mp} selects "
                f"--full-exports {implied}. Omit --max-periodic-full-exports or pass "
                f"--full-exports {implied}."
            )
        if full_exports == "final" and mp != 0:
            raise ValueError(
                "Conflicting FULL export options: --full-exports final disables periodic FULL "
                f"exports (equivalent to --max-periodic-full-exports 0), but "
                f"--max-periodic-full-exports {mp} was given."
            )
        if full_exports == "periodic" and mp == 0:
            raise ValueError(
                "Conflicting FULL export options: --full-exports periodic requires "
                "--max-periodic-full-exports >= 1, but --max-periodic-full-exports 0 "
                "(equivalent to --full-exports final) was given."
            )
        mode_both: FullExportMode = "final" if full_exports == "final" else "periodic"
        return mode_both, mp

    if max_periodic_full_exports is not None:
        mp_only = int(max_periodic_full_exports)
        if mp_only == 0:
            return "final", 0
        return "periodic", mp_only

    eff_mode: FullExportMode = (
        default_full_exports
        if full_exports is None
        else (
            "none"
            if full_exports == "none"
            else ("final" if full_exports == "final" else "periodic")
        )
    )
    if eff_mode == "none":
        return "none", max(0, int(default_max_periodic))
    if eff_mode == "final":
        return "final", 0
    return "periodic", max(1, int(default_max_periodic))


RESUME_MUTABLE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "total_updates",
        "light_checkpoint_interval_sec",
        "full_checkpoint_interval_sec",
        "full_exports",
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
        "eval_opponents",
        "actor_chunk_learners",
        "compile_model",
        "fast_learner_ops",
        "recipe_change_history",
    }
)


def recipe_config_diff(
    reference: V3PPOConfig,
    candidate: V3PPOConfig,
) -> dict[str, tuple[Any, Any]]:
    """Return ``{key: (reference_val, candidate_val)}`` for changed allow-listed recipe keys."""
    diffs: dict[str, tuple[Any, Any]] = {}
    for key in RECIPE_OVERRIDE_KEYS:
        ref_val = getattr(reference, key)
        cand_val = getattr(candidate, key)
        if ref_val != cand_val:
            diffs[key] = (ref_val, cand_val)
    return diffs


def build_recipe_change_record(
    *,
    update: int,
    reference: V3PPOConfig,
    candidate: V3PPOConfig,
) -> dict[str, Any] | None:
    """Build ``{'update': update, 'changed_fields': {k: {'old': o, 'new': n}}}`` or ``None``."""
    diffs = recipe_config_diff(reference, candidate)
    if not diffs:
        return None
    return {
        "update": int(update),
        "changed_fields": {
            key: {"old": old_val, "new": new_val}
            for key, (old_val, new_val) in sorted(diffs.items())
        },
    }


def merge_and_append_recipe_history(
    *,
    prior_history: tuple[dict[str, Any], ...],
    current_cfg: V3PPOConfig,
    new_record: Mapping[str, Any] | None,
) -> V3PPOConfig:
    """Preserve ``prior_history`` prefix on ``current_cfg`` and append ``new_record`` once."""
    merged: list[dict[str, Any]] = []
    for item in (*prior_history, *current_cfg.recipe_change_history):
        norm_item = {
            "update": int(item["update"]),
            "changed_fields": {
                k: {"old": v["old"], "new": v["new"]}
                for k, v in sorted(item["changed_fields"].items())
            },
        }
        if norm_item not in merged:
            merged.append(norm_item)
    if new_record is not None:
        norm_new = {
            "update": int(new_record["update"]),
            "changed_fields": {
                k: {"old": v["old"], "new": v["new"]}
                for k, v in sorted(new_record["changed_fields"].items())
            },
        }
        if not merged or merged[-1] != norm_new:
            merged.append(norm_new)
    if tuple(merged) == current_cfg.recipe_change_history:
        return current_cfg
    return replace(current_cfg, recipe_change_history=tuple(merged))


def immutable_config_diff(
    reference: V3PPOConfig,
    candidate: V3PPOConfig,
    *,
    allow_recipe_change: bool = False,
) -> dict[str, tuple[Any, Any]]:
    """Return ``{key: (reference_val, candidate_val)}`` for mismatched resume-immutable keys."""
    ref_dict = reference.to_dict()
    cand_dict = candidate.to_dict()
    allowed_recipe = set(RECIPE_OVERRIDE_KEYS) if allow_recipe_change else set()
    diffs: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(ref_dict) | set(cand_dict)):
        if key in RESUME_MUTABLE_CONFIG_KEYS or key == "schema" or key in allowed_recipe:
            continue
        if ref_dict.get(key) != cand_dict.get(key):
            diffs[key] = (ref_dict.get(key), cand_dict.get(key))
    return diffs


def assert_resume_config_compatible(
    reference: V3PPOConfig,
    candidate: V3PPOConfig,
    *,
    allow_recipe_change: bool = False,
    context: str = "Checkpoint config mismatch",
    reference_label: str = "ckpt",
    candidate_label: str = "current",
) -> None:
    """Raise ``ValueError`` if ``candidate`` differs from ``reference`` on any immutable key."""
    diffs = immutable_config_diff(
        reference,
        candidate,
        allow_recipe_change=allow_recipe_change,
    )
    if diffs:
        key, (ref_val, cand_val) = next(iter(diffs.items()))
        hint = (
            " (pass --allow-recipe-change to change allow-listed recipe fields on a continuation)"
            if (not allow_recipe_change and key in RECIPE_OVERRIDE_KEYS)
            else ""
        )
        raise ValueError(
            f"{context} on {key!r}: {reference_label}={ref_val!r} vs "
            f"{candidate_label}={cand_val!r}{hint}"
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

    if cfg.full_exports == "none":
        max_periodic_full = 0
        max_full_exports = 0
        extra_full_lights = 0
    elif cfg.full_exports == "final":
        max_periodic_full = 0
        max_full_exports = 1
        extra_full_lights = 0
    else:
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

    if max_full_exports == 0:
        largest_single_file_bytes = max(est_light_bytes, TELEMETRY_EXPORT_RESERVE_BYTES)
        per_file_max_export_bytes = int(math.ceil(largest_single_file_bytes * 1.25))
        session_total_cap_bytes = NONE_MODE_MAX_TOTAL_EXPORT_BYTES
    else:
        largest_single_file_bytes = max(shard_cap_bytes, est_light_bytes)
        per_file_max_export_bytes = max(
            256 * 1024 * 1024,
            int(math.ceil(largest_single_file_bytes * 1.25)),
        )
        session_total_cap_bytes = GPU_MAX_TOTAL_EXPORT_BYTES[gpu_norm]

    return {
        "gpu": gpu_norm,
        "full_exports": cfg.full_exports,
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
    if int(caps["max_full_exports"]) > 0 and not caps["single_world_fits_shard"]:
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
    staged_input_bytes: int = 0,
    assumed_upload_rate_mbps: float = ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS,
    assumed_download_rate_mbps: float = ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS,
) -> dict[str, Any]:
    """Size ``prepare --mode train`` headroom between ``V3_PPO_MAX_WALL_SECONDS`` and timeout.

    Sizes headroom as the sum of four terms:
      1. ``setup_allowance_seconds`` (`330.0` s, measured `240–325` s session-start-to-``wall_t0``),
      2. ``staged_input_upload_seconds`` (`0.0` s for fresh runs; ``staged_input_bytes`` divided by
         ``assumed_upload_rate_mbps`` [`1.0 MB/s` = `1,000,000 B/s`] on continuations),
      3. ``final_eval_and_save_allowance_seconds`` (`300.0` s default, or
         ``canary_eval_wall_seconds + 150.0`` s when provided by the bound canary receipt),
      4. ``final_export_download_seconds``:
         - for ``full_exports == "none"`` (default): ``(final_light_bytes + final_telemetry_bytes)``
           divided by ``assumed_download_rate_mbps`` (`1.0 MB/s` = `1,000,000 B/s`);
         - for ``full_exports in {"final", "periodic"}``: ``(final_full_bytes + final_light_bytes)``
           from :func:`estimate_checkpoint_export_caps`'s per-world size model divided by
           ``assumed_download_rate_mbps`` (`1.0 MB/s` = `1,000,000 B/s`).

    Inside ``max_wall_seconds = floor(session_seconds - headroom)``, ``run_training_job`` holds
    ``trainer_checkpoint_reserve_seconds = min(90.0, max(5.0, 0.08 * max_wall_seconds))`` so an
    in-progress rollout/PPO update completes before final evaluation and checkpoint export begin.
    """
    if not math.isfinite(session_seconds) or float(session_seconds) <= 0.0:
        raise ValueError(f"session_seconds must be finite and > 0, got {session_seconds}")
    if isinstance(staged_input_bytes, bool) or int(staged_input_bytes) < 0:
        raise ValueError(f"staged_input_bytes must be an int >= 0, got {staged_input_bytes!r}")
    if not math.isfinite(assumed_upload_rate_mbps) or float(assumed_upload_rate_mbps) <= 0.0:
        raise ValueError(
            f"assumed_upload_rate_mbps must be finite and > 0, got {assumed_upload_rate_mbps}"
        )
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
    staged_in_bytes = int(staged_input_bytes)
    upload_rate_bps = float(assumed_upload_rate_mbps) * 1_000_000.0
    upload_sec = float(staged_in_bytes) / upload_rate_bps

    if canary_eval_wall_seconds is not None:
        eval_save_sec = float(canary_eval_wall_seconds) + float(
            TRAIN_FINAL_SAVE_AFTER_EVAL_ALLOWANCE_SECONDS
        )
        eval_save_source = "canary_receipt_eval_plus_150s"
    else:
        eval_save_sec = float(TRAIN_FINAL_EVAL_AND_SAVE_ALLOWANCE_SECONDS)
        eval_save_source = "default_300s"

    final_light_bytes = int(caps["est_light_checkpoint_bytes"])
    if cfg.full_exports == "none":
        final_full_bytes = 0
        final_telemetry_bytes = int(TRAIN_FINAL_TELEMETRY_EXPORT_BYTES)
    else:
        final_full_bytes = int(caps["est_full_sim_bytes"])
        final_telemetry_bytes = 0
    final_export_bytes = final_full_bytes + final_light_bytes + final_telemetry_bytes

    download_rate_bps = float(assumed_download_rate_mbps) * 1_000_000.0
    download_sec = float(final_export_bytes) / download_rate_bps
    headroom_sec = setup_sec + upload_sec + eval_save_sec + download_sec

    sess_sec = float(session_seconds)
    max_allowed_sec = float(MAX_TRAIN_HEADROOM_SESSION_FRACTION) * sess_sec
    headroom_fraction = headroom_sec / sess_sec
    fits_fraction = bool(headroom_sec <= max_allowed_sec)
    max_wall_seconds = int(math.floor(sess_sec - headroom_sec))
    eff_max_wall_pos = max(1, max_wall_seconds)
    trainer_reserve_sec = min(
        float(TRAINER_CHECKPOINT_RESERVE_SECONDS),
        max(5.0, float(eff_max_wall_pos) * 0.08),
    )

    return {
        "gpu": str(caps["gpu"]),
        "full_exports": cfg.full_exports,
        "num_envs": int(cfg.num_envs),
        "net_size": cfg.net_size,
        "session_seconds": sess_sec,
        "setup_allowance_seconds": setup_sec,
        "staged_input_bytes": staged_in_bytes,
        "assumed_upload_rate_mbps": float(assumed_upload_rate_mbps),
        "assumed_upload_rate_bytes_per_sec": upload_rate_bps,
        "staged_input_upload_seconds": upload_sec,
        "final_eval_and_save_allowance_seconds": eval_save_sec,
        "final_eval_and_save_source": eval_save_source,
        "canary_eval_wall_seconds": (
            float(canary_eval_wall_seconds) if canary_eval_wall_seconds is not None else None
        ),
        "final_full_bytes": final_full_bytes,
        "final_light_bytes": final_light_bytes,
        "final_telemetry_bytes": final_telemetry_bytes,
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
        "trainer_checkpoint_reserve_seconds": round(trainer_reserve_sec, 4),
    }


def estimate_train_wall_headroom_by_mode(
    cfg: V3PPOConfig,
    *,
    gpu: str = "L4",
    session_seconds: float = 4.0 * 3600.0,
    canary_eval_wall_seconds: float | None = None,
    measured_packed_bytes_per_world: int | None = None,
    staged_input_bytes: int = 0,
    assumed_upload_rate_mbps: float = ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS,
    assumed_download_rate_mbps: float = ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS,
) -> dict[FullExportMode, dict[str, Any]]:
    """Compute train wall-guard headroom side by side across ``none``, ``final``, & ``periodic``."""
    base_dict = cfg.to_dict()
    periodic_cap = max(1, int(cfg.max_periodic_full_exports))
    out: dict[FullExportMode, dict[str, Any]] = {}
    for mode in VALID_FULL_EXPORT_MODES:
        mode_dict = dict(base_dict)
        mode_dict["full_exports"] = mode
        mode_dict["max_periodic_full_exports"] = periodic_cap if mode == "periodic" else 0
        mode_cfg = V3PPOConfig.from_dict(mode_dict)
        out[mode] = estimate_train_wall_headroom(
            mode_cfg,
            gpu=gpu,
            session_seconds=session_seconds,
            canary_eval_wall_seconds=canary_eval_wall_seconds,
            measured_packed_bytes_per_world=measured_packed_bytes_per_world,
            staged_input_bytes=staged_input_bytes,
            assumed_upload_rate_mbps=assumed_upload_rate_mbps,
            assumed_download_rate_mbps=assumed_download_rate_mbps,
        )
    return out


def validate_train_wall_headroom(
    cfg: V3PPOConfig,
    gpu: str,
    *,
    session_seconds: float,
    canary_eval_wall_seconds: float | None = None,
    measured_packed_bytes_per_world: int | None = None,
    staged_input_bytes: int = 0,
    assumed_upload_rate_mbps: float = ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS,
    assumed_download_rate_mbps: float = ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS,
) -> dict[str, Any]:
    """Validate ``train`` wall-guard headroom fits ``<= 40%`` of ``session_seconds``."""
    est = estimate_train_wall_headroom(
        cfg,
        gpu=gpu,
        session_seconds=session_seconds,
        canary_eval_wall_seconds=canary_eval_wall_seconds,
        measured_packed_bytes_per_world=measured_packed_bytes_per_world,
        staged_input_bytes=staged_input_bytes,
        assumed_upload_rate_mbps=assumed_upload_rate_mbps,
        assumed_download_rate_mbps=assumed_download_rate_mbps,
    )
    if not est["fits_max_headroom_fraction"]:
        sess_hours = float(session_seconds) / 3600.0
        pct = 100.0 * float(est["headroom_seconds"]) / float(session_seconds)
        max_pct = 100.0 * float(est["max_headroom_fraction"])
        upload_clause = (
            f"staged_input_upload={est['staged_input_upload_seconds']:.1f} s + "
            if int(est["staged_input_bytes"]) > 0
            else ""
        )
        raise ValueError(
            f"Refusing train packet whose wall-guard headroom "
            f"({est['headroom_seconds']:.1f} s = {est['headroom_seconds'] / 60.0:.1f} min, "
            f"{pct:.1f}% of session in full_exports={cfg.full_exports!r}: "
            f"setup={est['setup_allowance_seconds']:.1f} s + "
            f"{upload_clause}"
            f"final_eval_and_save={est['final_eval_and_save_allowance_seconds']:.1f} s + "
            f"final_export_download={est['final_export_download_seconds']:.1f} s for "
            f"{est['final_export_bytes'] / 1e6:.1f} MB at "
            f"{est['assumed_download_rate_mbps']:.2f} MB/s) "
            f"exceeds {max_pct:.0f}% of the {sess_hours:g} h session "
            f"({est['max_allowed_headroom_seconds']:.1f} s max on {float(session_seconds):.0f} s "
            f"timeout; worlds={cfg.num_envs}, net_size={cfg.net_size}, gpu={est['gpu']}). "
            "Use --full-exports none (LIGHT-only exports), increase --hours, reduce --num-envs, "
            "or supply --measured-packed-bytes-per-world from a verified run."
        )
    return est
