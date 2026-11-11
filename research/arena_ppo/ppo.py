"""GPU-resident clipped PPO self-play learner with compact rollout tape and verified resume.

Determinism note:
- On CPU with ``deterministic=True``, checkpoint save + ``load_checkpoint`` produces
  bit-identical continuation trajectories (`resumed == uninterrupted` across model
  parameters, AdamW moments, simulator tensors, and RNG streams).
- On CUDA with default ``deterministic=False`` (`cudnn.benchmark=True`, `bf16` autocast,
  `scatter_reduce_` / atomic kernels), resumed runs restore exact state and RNG streams
  for training continuity, while minor floating-point order non-determinism is expected.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from research.arena_ppo.net import (
    AUX_DEATH_HORIZONS,
    VALID_NET_ARCHS,
    ArenaNet,
    ArenaPolicyModel,
    NetArch,
    NetSize,
    build_arena_model,
    default_actor_chunk_envs,
    default_minibatch_size,
    symlog_two_hot_loss,
)
from research.arena_ppo.obs import (
    SharedWorldObs,
    build_arena_ego_obs_from_tensors,
    build_arena_ego_obs_gpu,
    build_shared_world_obs_from_tensors,
    build_shared_world_obs_gpu,
)
from research.arena_ppo.runtime import (
    RotatingCreateOnlyJSONLWriter,
    SignalStopFlag,
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    preload_native_sim_extensions,
    publish_hardlinked_artifacts,
    remove_completed_artifact,
    sha256_file,
    synchronize_device,
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.arena_ppo.sim import (
    _SIM_STATE_TENSORS,
    DEFAULT_GRID_H,
    DEFAULT_GRID_W,
    DEFAULT_NUM_SNAKES,
    OPPONENT_KIND_SCRIPTED,
    OPPONENT_KIND_SNAPSHOT,
    LateStateReservoir,
    TrainSim,
    arena_v2_config,
)
from src.core.mechanics_constants import corpse_food_cap

ARENA_PPO_SCHEMA = "arena-ppo-stage1/v1"
ARENA_PPO_NAMESPACE = "arena_ppo_stage1_scratch_v2_20261110"
SUPERVISOR_GRACE_SAVE_LIMIT_SEC = 10.0
UNLIMITED_HOURS_TOTAL_UPDATES: int = 1_000_000_000
MIN_MAX_UPDATE_SECONDS: float = 1200.0
HARD_KILL_MAX_LOSS_SEC: float = 2700.0
PUBLISH_GAP_SAFETY_FACTOR: float = 1.25
_RESUME_MUTABLE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "total_updates",
        "checkpoint_interval_sec",
        "checkpoint_every_updates",
        "keep_periodic_checkpoints",
        "max_update_seconds",
        "actor_chunk_envs",
        "ring_chunk",
        "compile_model",
        "amp",
        "use_bf16_autocast",
        "use_channels_last",
        "output_publish_interval_sec",
        "eval_every_updates",
        "log_rotate_every_updates",
        "log_rotate_every_seconds",
    }
)


class StopRequestedError(RuntimeError):
    """Raised when cooperative signal stop (`SIGUSR1`/`SIGTERM`) interrupts an update."""


@dataclass(frozen=True)
class ArenaPPOConfig:
    """Complete JSON-serializable hyperparameter configuration for Stage 1 Arena PPO."""

    net_arch: NetArch = "shared_trunk"
    net_size: NetSize = "S"
    num_envs: int = 1024
    num_snakes: int = DEFAULT_NUM_SNAKES
    grid_h: int = DEFAULT_GRID_H
    grid_w: int = DEFAULT_GRID_W
    rollout_len: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.97
    lr: float = 3.0e-4
    adam_eps: float = 1.0e-5
    weight_decay: float = 0.0
    clip_ratio: float = 0.2
    ppo_epochs: int = 2
    minibatch_size: int | None = None
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    aux_coef: float = 0.1
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    actor_chunk_envs: int | None = None
    ring_chunk: int = 512
    max_capacity: int = 1600
    max_world_frames: int = 0
    total_updates: int = 100
    seed: int = 20261110
    deterministic: bool = False
    compile_model: bool = False
    amp: bool = False
    use_channels_last: bool = True
    use_bf16_autocast: bool = False
    checkpoint_interval_sec: float = 1200.0
    checkpoint_every_updates: int = 0
    keep_periodic_checkpoints: int = 3
    max_update_seconds: float = 1200.0
    # 30 min default + predictive <= 45 min guard so hard-kill loss never exceeds 45 min
    output_publish_interval_sec: float = 1800.0
    eval_every_updates: int = 0
    log_rotate_every_updates: int = 64
    log_rotate_every_seconds: float = 900.0
    late_reset_enabled: bool = False
    late_reset_min_length: int = 600
    late_reset_prob: float = 0.5
    late_reset_reservoir_size: int = 64

    def __post_init__(self) -> None:
        if self.net_arch not in VALID_NET_ARCHS:
            raise ValueError(
                f"Unsupported net_arch {self.net_arch!r}; expected one of {VALID_NET_ARCHS}"
            )
        object.__setattr__(self, "amp", bool(self.amp))
        object.__setattr__(self, "use_bf16_autocast", bool(self.amp))
        if self.minibatch_size is None:
            object.__setattr__(
                self,
                "minibatch_size",
                default_minibatch_size(self.net_size, net_arch=self.net_arch),
            )
        if self.actor_chunk_envs is None:
            object.__setattr__(
                self,
                "actor_chunk_envs",
                default_actor_chunk_envs(self.net_size, net_arch=self.net_arch),
            )
        if int(self.keep_periodic_checkpoints) < 1:
            raise ValueError(
                f"keep_periodic_checkpoints must be >= 1, got {self.keep_periodic_checkpoints}"
            )
        if float(self.output_publish_interval_sec) <= 0.0:
            raise ValueError(
                "output_publish_interval_sec must be > 0, "
                f"got {self.output_publish_interval_sec}"
            )
        if int(self.eval_every_updates) < 0:
            raise ValueError(f"eval_every_updates must be >= 0, got {self.eval_every_updates}")
        if int(self.log_rotate_every_updates) < 1:
            raise ValueError(
                f"log_rotate_every_updates must be >= 1, got {self.log_rotate_every_updates}"
            )
        if float(self.log_rotate_every_seconds) <= 0.0:
            raise ValueError(
                f"log_rotate_every_seconds must be > 0, got {self.log_rotate_every_seconds}"
            )
        if int(self.late_reset_min_length) < 1:
            raise ValueError(
                f"late_reset_min_length must be >= 1, got {self.late_reset_min_length}"
            )
        if not (0.0 <= float(self.late_reset_prob) <= 1.0):
            raise ValueError(f"late_reset_prob must be in [0, 1], got {self.late_reset_prob}")
        if int(self.late_reset_reservoir_size) < 1:
            raise ValueError(
                f"late_reset_reservoir_size must be >= 1, got {self.late_reset_reservoir_size}"
            )
        if self.late_reset_enabled and int(self.max_world_frames) <= 0:
            raise ValueError(
                "late_reset_enabled=True requires max_world_frames > 0, "
                f"got max_world_frames={self.max_world_frames}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ArenaPPOConfig":
        """Construct ``ArenaPPOConfig`` from dictionary, rejecting unknown fields."""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown ArenaPPOConfig fields: {unknown}")
        merged = dict(raw)
        if "amp" not in merged and "use_bf16_autocast" in merged:
            merged["amp"] = bool(merged["use_bf16_autocast"])
        return cls(**merged)

    @classmethod
    def from_json(cls, source: Path | str) -> "ArenaPPOConfig":
        """Load ``ArenaPPOConfig`` from a JSON file path or raw JSON string."""
        if isinstance(source, Path):
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            candidate = Path(source)
            if source.lstrip().startswith("{"):
                payload = json.loads(source)
            elif candidate.is_file():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(f"ArenaPPOConfig JSON file not found: {source}")
        if not isinstance(payload, dict):
            raise ValueError("ArenaPPOConfig JSON root must be an object")
        return cls.from_dict(payload)


class CompactRolloutBuffer:
    """Lossless GPU-resident rollout buffer storing append-only body tape + PPO tensors.

    Avoids materializing `(T, E*6, 13, 83, 145)` rasters (which would require ~4 TB at
    `E=4096, T=256`) by recording an unwrapped initial body tape `(E, 6, cap + 2*T + 2, 2)`
    plus up to 2 pushed cells per snake per tick and compact `uint8` food arrays. Any
    `(t, e)` world-tick reconstructs bit-identical `(arena, ego, scalars)` tensors on GPU.
    """

    def __init__(
        self,
        *,
        rollout_len: int,
        num_envs: int,
        num_snakes: int,
        cap: int,
        food_capacity: int,
        min_boost_length: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> None:
        self.T = int(rollout_len)
        self.E = int(num_envs)
        self.S = int(num_snakes)
        self.cap = int(cap)
        self.max_seg_used: int = int(cap)
        self.food_capacity = int(food_capacity)
        self.min_boost_length = int(min_boost_length)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.device = device

        tape_len = self.cap + 2 * self.T + 4
        self.body_tape = torch.zeros(
            (self.E, self.S, tape_len, 2), dtype=torch.int16, device=device
        )
        self.tape_cursor = torch.zeros((self.E, self.S), dtype=torch.int32, device=device)
        self.tape_head_idx = torch.zeros((self.T, self.E, self.S), dtype=torch.int32, device=device)

        tes = (self.T, self.E, self.S)
        self.seg_count = torch.zeros(tes, dtype=torch.int16, device=device)
        self.length = torch.zeros(tes, dtype=torch.int16, device=device)
        self.alive = torch.zeros(tes, dtype=torch.bool, device=device)
        self.prev_alive = torch.zeros(tes, dtype=torch.bool, device=device)
        self.boosted = torch.zeros(tes, dtype=torch.bool, device=device)
        self.direction = torch.zeros(tes, dtype=torch.uint8, device=device)
        self.prev_heads = torch.zeros((self.T, self.E, self.S, 2), dtype=torch.int16, device=device)

        food_coord_dtype = torch.uint8 if max(self.grid_w, self.grid_h) <= 256 else torch.int16
        self.food_xy = torch.zeros(
            (self.T, self.E, self.food_capacity, 2), dtype=food_coord_dtype, device=device
        )
        # 0 = absent, 1 = ambient, 2 = corpse
        self.food_kind = torch.zeros(
            (self.T, self.E, self.food_capacity), dtype=torch.uint8, device=device
        )

        self.actions = torch.zeros(tes, dtype=torch.uint8, device=device)
        self.log_probs = torch.zeros(tes, dtype=torch.float32, device=device)
        self.values = torch.zeros(tes, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(tes, dtype=torch.float32, device=device)
        self.died = torch.zeros(tes, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(tes, dtype=torch.bool, device=device)
        self.trunc_bootstrap = torch.zeros(tes, dtype=torch.float32, device=device)
        self.acted = torch.zeros(tes, dtype=torch.bool, device=device)
        self.learner_mask = torch.zeros(tes, dtype=torch.bool, device=device)

        self.advantages = torch.zeros(tes, dtype=torch.float32, device=device)
        self.returns = torch.zeros(tes, dtype=torch.float32, device=device)
        self.aux_targets = torch.zeros(
            (self.T, self.E, self.S, len(AUX_DEATH_HORIZONS)), dtype=torch.float32, device=device
        )
        self.aux_valid = torch.zeros(
            (self.T, self.E, self.S, len(AUX_DEATH_HORIZONS)), dtype=torch.bool, device=device
        )

        self._env_2d = torch.arange(self.E, device=device)[:, None]
        self._snake_2d = torch.arange(self.S, device=device)[None, :]
        self._one_i32 = torch.ones((self.E, self.S), dtype=torch.int32, device=device)
        self._two_i32 = torch.full((self.E, self.S), 2, dtype=torch.int32, device=device)
        self._one_u8 = torch.ones((self.E, self.food_capacity), dtype=torch.uint8, device=device)
        self._two_u8 = torch.full((self.E, self.food_capacity), 2, dtype=torch.uint8, device=device)
        self._zero_u8 = torch.zeros((self.E, self.food_capacity), dtype=torch.uint8, device=device)

    def nbytes(self) -> int:
        """Return exact resident device storage of all buffer tensors in bytes."""
        total = 0
        for val in self.__dict__.values():
            if isinstance(val, torch.Tensor):
                total += val.numel() * val.element_size()
        return int(total)

    def record_pre_step(self, t: int, train_sim: TrainSim) -> None:
        """Snapshot pre-step decision state at rollout tick `t` with zero host syncs."""
        sim = train_sim.sim
        if t == 0:
            env_3d = torch.arange(self.E, device=self.device)[:, None, None]
            snake_3d = torch.arange(self.S, device=self.device)[None, :, None]
            # Place k = 0 (head) at tape index `cap - 1` and k = cap - 1 at index `0`
            rev_k = torch.arange(self.cap - 1, -1, -1, dtype=torch.long, device=self.device)[
                None, None, :
            ]
            rings = (sim.head_ptr[..., None] - rev_k) % self.cap
            self.body_tape[:, :, : self.cap, :] = sim.bodies[env_3d, snake_3d, rings].to(
                torch.int16
            )
            self.tape_cursor.fill_(self.cap)
            self.tape_head_idx[0].fill_(self.cap - 1)
            self.max_seg_used = max(1, min(self.cap, int(train_sim.max_seg_bound)))
        else:
            hp = sim.head_ptr
            hp_m1 = (hp - 1) % self.cap
            cell_k0 = sim.bodies[self._env_2d, self._snake_2d, hp].to(torch.int16)
            cell_k1 = sim.bodies[self._env_2d, self._snake_2d, hp_m1].to(torch.int16)
            two_push = train_sim.prev_alive & sim._boosted_this_step
            first_cell = torch.where(two_push[..., None], cell_k1, cell_k0)
            cur_long = self.tape_cursor.to(torch.long)
            self.body_tape[self._env_2d, self._snake_2d, cur_long] = first_cell
            self.body_tape[self._env_2d, self._snake_2d, cur_long + 1] = cell_k0
            inc = torch.where(two_push, self._two_i32, self._one_i32)
            self.tape_cursor = self.tape_cursor + inc
            self.tape_head_idx[t] = self.tape_cursor - 1
            self.max_seg_used = min(self.cap, max(self.max_seg_used, int(train_sim.max_seg_bound)))

        self.seg_count[t] = sim.seg_count.to(torch.int16)
        self.length[t] = sim.length.to(torch.int16)
        self.alive[t] = sim.alive
        self.prev_alive[t] = train_sim.prev_alive
        self.boosted[t] = sim._boosted_this_step
        self.direction[t] = sim.direction.clamp(0, 3).to(torch.uint8)
        self.prev_heads[t] = train_sim.prev_heads.to(torch.int16)

        fx = sim.food_cells[..., 0]
        fy = sim.food_cells[..., 1]
        live_f = sim.food_present & (fx >= 0) & (fx < self.grid_w) & (fy >= 0) & (fy < self.grid_h)
        self.food_xy[t, ..., 0] = fx.clamp(0, max(0, self.grid_w - 1)).to(self.food_xy.dtype)
        self.food_xy[t, ..., 1] = fy.clamp(0, max(0, self.grid_h - 1)).to(self.food_xy.dtype)
        self.food_kind[t] = torch.where(
            live_f & sim.corpse_food,
            self._two_u8,
            torch.where(live_f, self._one_u8, self._zero_u8),
        )

    def reconstruct_world_ticks(
        self,
        t_idx: torch.Tensor,
        e_idx: torch.Tensor,
        *,
        ring_chunk: int = 512,
        max_seg: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct bit-identical `(arena, ego, scalars, legal_mask)` for `(t_idx, e_idx)` pairs.

        Args:
            t_idx: 1D long tensor `(M,)` of rollout step indices in `0 .. T - 1`.
            e_idx: 1D long tensor `(M,)` of world indices in `0 .. E - 1`.
            ring_chunk: Segment chunk width.
            max_seg: Optional upper bound `1 <= max_seg <= cap` on active segments in rollout.

        Returns:
            Tuple `(arena, ego, scalars, legal_mask)` over all `M * S` seats.
        """
        m_worlds = int(t_idx.shape[0])
        gather_k = self.cap if max_seg is None else max(1, min(self.cap, int(max_seg)))
        head_pos = self.tape_head_idx[t_idx, e_idx].to(torch.long)
        k_offsets = torch.arange(gather_k, dtype=torch.long, device=self.device)[None, None, :]
        gather_pos = (head_pos[..., None] - k_offsets).clamp_min(0)
        e_3d = e_idx[:, None, None]
        s_3d = torch.arange(self.S, device=self.device)[None, :, None]
        ordered_bodies = self.body_tape[e_3d, s_3d, gather_pos]

        seg_sub = self.seg_count[t_idx, e_idx].to(torch.long)
        len_sub = self.length[t_idx, e_idx].to(torch.long)
        alive_sub = self.alive[t_idx, e_idx]
        prev_alive_sub = self.prev_alive[t_idx, e_idx]
        boosted_sub = self.boosted[t_idx, e_idx]
        dir_sub = self.direction[t_idx, e_idx].to(torch.long)
        prev_heads_sub = self.prev_heads[t_idx, e_idx].to(torch.long)

        f_kind = self.food_kind[t_idx, e_idx]
        f_xy = self.food_xy[t_idx, e_idx].to(torch.long)
        f_pres = f_kind > 0
        f_corpse = f_kind == 2

        obs = build_arena_ego_obs_from_tensors(
            ordered_bodies=ordered_bodies,
            seg_count=seg_sub,
            length=len_sub,
            alive=alive_sub,
            direction=dir_sub,
            boosted=boosted_sub,
            prev_heads=prev_heads_sub,
            prev_alive=prev_alive_sub,
            food_cells=f_xy,
            food_present=f_pres,
            corpse_food=f_corpse,
            min_boost_length=self.min_boost_length,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            ring_chunk=ring_chunk,
            max_seg=gather_k,
        )
        normal = torch.ones((m_worlds, self.S, 3), dtype=torch.bool, device=self.device)
        can_boost = alive_sub & (len_sub >= self.min_boost_length)
        legal_mask = torch.cat((normal, can_boost[..., None].expand(-1, -1, 3)), dim=2).reshape(
            m_worlds * self.S, 6
        )
        return obs.arena, obs.ego, obs.scalars, legal_mask

    def reconstruct_shared_world_ticks(
        self,
        t_idx: torch.Tensor,
        e_idx: torch.Tensor,
        *,
        ring_chunk: int = 512,
        max_seg: int | None = None,
    ) -> tuple[SharedWorldObs, torch.Tensor]:
        """Reconstruct bit-identical ``(SharedWorldObs, legal_mask)`` for ``(t_idx, e_idx)`` pairs.

        Args:
            t_idx: 1D long tensor ``(M,)`` of rollout step indices in ``0 .. T - 1``.
            e_idx: 1D long tensor ``(M,)`` of world indices in ``0 .. E - 1``.
            ring_chunk: Segment chunk width.
            max_seg: Optional upper bound ``1 <= max_seg <= cap`` on active segments in rollout.

        Returns:
            Tuple ``(shared_obs, legal_mask)`` over ``M`` world-ticks (`M * S` seats).
        """
        m_worlds = int(t_idx.shape[0])
        gather_k = self.cap if max_seg is None else max(1, min(self.cap, int(max_seg)))
        head_pos = self.tape_head_idx[t_idx, e_idx].to(torch.long)
        k_offsets = torch.arange(gather_k, dtype=torch.long, device=self.device)[None, None, :]
        gather_pos = (head_pos[..., None] - k_offsets).clamp_min(0)
        e_3d = e_idx[:, None, None]
        s_3d = torch.arange(self.S, device=self.device)[None, :, None]
        ordered_bodies = self.body_tape[e_3d, s_3d, gather_pos]

        seg_sub = self.seg_count[t_idx, e_idx].to(torch.long)
        len_sub = self.length[t_idx, e_idx].to(torch.long)
        alive_sub = self.alive[t_idx, e_idx]
        prev_alive_sub = self.prev_alive[t_idx, e_idx]
        boosted_sub = self.boosted[t_idx, e_idx]
        dir_sub = self.direction[t_idx, e_idx].to(torch.long)
        prev_heads_sub = self.prev_heads[t_idx, e_idx].to(torch.long)

        f_kind = self.food_kind[t_idx, e_idx]
        f_xy = self.food_xy[t_idx, e_idx].to(torch.long)
        f_pres = f_kind > 0
        f_corpse = f_kind == 2

        obs = build_shared_world_obs_from_tensors(
            ordered_bodies=ordered_bodies,
            seg_count=seg_sub,
            length=len_sub,
            alive=alive_sub,
            direction=dir_sub,
            boosted=boosted_sub,
            prev_heads=prev_heads_sub,
            prev_alive=prev_alive_sub,
            food_cells=f_xy,
            food_present=f_pres,
            corpse_food=f_corpse,
            min_boost_length=self.min_boost_length,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            ring_chunk=ring_chunk,
            max_seg=gather_k,
        )
        normal = torch.ones((m_worlds, self.S, 3), dtype=torch.bool, device=self.device)
        can_boost = alive_sub & (len_sub >= self.min_boost_length)
        legal_mask = torch.cat((normal, can_boost[..., None].expand(-1, -1, 3)), dim=2).reshape(
            m_worlds * self.S, 6
        )
        return obs, legal_mask

    def compute_gae_and_aux(
        self,
        *,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages, returns, and censoring-free aux death targets on device."""
        gae = torch.zeros((self.E, self.S), dtype=torch.float32, device=self.device)
        zero_f = torch.zeros((self.E, self.S), dtype=torch.float32, device=self.device)

        inf_steps = 1_000_000
        inf_t = torch.full((self.E, self.S), inf_steps, dtype=torch.int32, device=self.device)
        zero_i = torch.zeros((self.E, self.S), dtype=torch.int32, device=self.device)
        one_i = torch.ones((self.E, self.S), dtype=torch.int32, device=self.device)
        steps_to_death = inf_t.clone()
        steps_to_censor = zero_i.clone()

        for t in range(self.T - 1, -1, -1):
            next_vals_cont = last_values if t == self.T - 1 else self.values[t + 1]
            died_t = self.died[t]
            trunc_t = self.truncated[t] & (~died_t)
            acted_t = self.acted[t]

            next_val = torch.where(
                died_t,
                zero_f,
                torch.where(trunc_t, self.trunc_bootstrap[t], next_vals_cont),
            )
            gae_cont = (~died_t) & (~trunc_t) & acted_t
            delta = self.rewards[t] + float(gamma) * next_val - self.values[t]
            gae = torch.where(
                acted_t,
                delta + float(gamma * gae_lambda) * torch.where(gae_cont, gae, zero_f),
                zero_f,
            )
            self.advantages[t] = gae
            self.returns[t] = torch.where(acted_t, gae + self.values[t], zero_f)

            # Backward recurrence for death-within-{16, 64, 256} without censoring bias
            d_dist = (steps_to_death + 1).clamp_max(inf_steps)
            c_dist = (steps_to_censor + 1).clamp_max(inf_steps)
            d_dist = torch.where(died_t, zero_i, torch.where(trunc_t | (~acted_t), inf_t, d_dist))
            c_dist = torch.where(
                died_t,
                inf_t,
                torch.where(trunc_t, one_i, torch.where(~acted_t, zero_i, c_dist)),
            )
            steps_to_death = d_dist
            steps_to_censor = c_dist

            for h_idx, horizon in enumerate(AUX_DEATH_HORIZONS):
                died_within = d_dist < int(horizon)
                survived_horizon = (d_dist >= int(horizon)) & (c_dist >= int(horizon))
                self.aux_valid[t, :, :, h_idx] = acted_t & (died_within | survived_horizon)
                self.aux_targets[t, :, :, h_idx] = died_within.to(torch.float32)


@dataclass
class PPOUpdateMetrics:
    """Telemetry summary returned by one integrated rollout + PPO update."""

    update_idx: int
    agent_steps_collected: int
    world_steps_collected: int
    learner_rows: int
    mean_reward: float
    deaths_count: int
    policy_loss: float
    value_loss: float
    aux_loss: float
    entropy: float
    approx_kl: float
    epoch0_approx_kl: float
    clip_fraction: float
    grad_norm: float
    collect_sec: float
    learn_sec: float
    acted_steps: int = 0
    steps_per_sec: float = 0.0
    mean_length: float = 0.0
    p95_length: float = 0.0
    deaths_per_1k_agent_steps: float = 0.0
    mean_episode_return: float = 0.0
    value_explained_variance: float = 0.0
    aux_loss_h16: float = 0.0
    aux_loss_h64: float = 0.0
    aux_loss_h256: float = 0.0


def estimate_rollout_buffer_bytes(
    *,
    rollout_len: int,
    num_envs: int,
    num_snakes: int = DEFAULT_NUM_SNAKES,
    max_capacity: int = 1600,
    grid_h: int = DEFAULT_GRID_H,
    grid_w: int = DEFAULT_GRID_W,
    food_capacity: int | None = None,
) -> int:
    """Return exact resident device byte footprint of ``CompactRolloutBuffer``."""
    t_steps = int(rollout_len)
    e_worlds = int(num_envs)
    s_snakes = int(num_snakes)
    cap = int(max_capacity)
    gh, gw = int(grid_h), int(grid_w)

    if food_capacity is None:
        sim_cfg = arena_v2_config(
            e_worlds,
            grid_h=gh,
            grid_w=gw,
            num_snakes=s_snakes,
            max_capacity=cap,
        )
        f_cap = int(sim_cfg.max_food + corpse_food_cap(sim_cfg.max_food))
    else:
        f_cap = int(food_capacity)

    tape_len = cap + 2 * t_steps + 4
    tes = t_steps * e_worlds * s_snakes
    food_coord_bytes = 1 if max(gw, gh) <= 256 else 2
    n_aux = len(AUX_DEATH_HORIZONS)

    body_bytes = e_worlds * s_snakes * tape_len * 2 * 2
    cursor_bytes = e_worlds * s_snakes * 4 + tes * 4
    per_seat_tick_bytes = tes * (
        2  # seg_count
        + 2  # length
        + 1  # alive
        + 1  # prev_alive
        + 1  # boosted
        + 1  # direction
        + 4  # prev_heads (2 * int16)
        + 1  # actions
        + 4  # log_probs
        + 4  # values
        + 4  # rewards
        + 1  # died
        + 1  # truncated
        + 4  # trunc_bootstrap
        + 1  # acted
        + 1  # learner_mask
        + 4  # advantages
        + 4  # returns
        + (4 * n_aux)  # aux_targets
        + (1 * n_aux)  # aux_valid
    )
    food_bytes = t_steps * e_worlds * f_cap * (2 * food_coord_bytes + 1)
    helper_bytes = (
        e_worlds * 8 + s_snakes * 8 + 2 * e_worlds * s_snakes * 4 + 3 * e_worlds * f_cap * 1
    )
    return int(body_bytes + cursor_bytes + per_seat_tick_bytes + food_bytes + helper_bytes)


def compute_max_update_seconds(cfg: ArenaPPOConfig, expected_steps_per_sec: float) -> float:
    """Scale `max_update_seconds` to `3x` expected update duration with a `1200 s` floor."""
    if not math.isfinite(expected_steps_per_sec) or float(expected_steps_per_sec) <= 0.0:
        raise ValueError(f"expected_steps_per_sec must be positive, got {expected_steps_per_sec}")
    steps_per_update = int(cfg.rollout_len) * int(cfg.num_envs) * int(cfg.num_snakes)
    expected_sec_per_update = float(steps_per_update) / float(expected_steps_per_sec)
    return max(MIN_MAX_UPDATE_SECONDS, round(3.0 * expected_sec_per_update, 2))


def build_training_plan(
    cfg: ArenaPPOConfig,
    *,
    hours: float,
    expected_steps_per_sec: float,
    max_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute expected PPO updates and agent-steps for a `--hours` run given throughput."""
    if not math.isfinite(hours) or float(hours) <= 0.0:
        raise ValueError(f"hours must be positive, got {hours}")
    if not math.isfinite(expected_steps_per_sec) or float(expected_steps_per_sec) <= 0.0:
        raise ValueError(f"expected_steps_per_sec must be positive, got {expected_steps_per_sec}")

    steps_per_update = int(cfg.rollout_len) * int(cfg.num_envs) * int(cfg.num_snakes)
    nominal_seconds = float(hours) * 3600.0
    planned_seconds = (
        min(nominal_seconds, float(max_wall_seconds))
        if max_wall_seconds is not None and float(max_wall_seconds) > 0.0
        else nominal_seconds
    )
    expected_agent_steps = int(round(planned_seconds * float(expected_steps_per_sec)))
    expected_world_steps = expected_agent_steps // max(1, int(cfg.num_snakes))
    expected_updates = max(1, int(math.floor(expected_agent_steps / float(steps_per_update))))
    nominal_agent_steps = int(round(nominal_seconds * float(expected_steps_per_sec)))
    nominal_updates = max(1, int(math.floor(nominal_agent_steps / float(steps_per_update))))
    expected_sec_per_update = float(steps_per_update) / float(expected_steps_per_sec)
    recommended_max_update_sec = compute_max_update_seconds(cfg, float(expected_steps_per_sec))
    return {
        "hours": float(hours),
        "nominal_seconds": round(nominal_seconds, 2),
        "planned_seconds": round(planned_seconds, 2),
        "expected_steps_per_sec": float(expected_steps_per_sec),
        "agent_steps_per_update": steps_per_update,
        "expected_seconds_per_update": round(expected_sec_per_update, 4),
        "recommended_max_update_seconds": recommended_max_update_sec,
        "expected_updates": expected_updates,
        "expected_agent_steps": expected_agent_steps,
        "expected_world_steps": expected_world_steps,
        "nominal_expected_updates": nominal_updates,
        "nominal_expected_agent_steps": nominal_agent_steps,
    }


def estimate_checkpoint_bytes(cfg: ArenaPPOConfig) -> int:
    """Estimate worst-case single-file byte size of one ``ckpt_*.pt`` checkpoint.

    Sizes ``bodies_live_i16`` from full ``cfg.max_capacity`` (because ``TrainSim.state_dict``
    saves up to global ``max(seg_count)`` across all ``E * S`` snakes), plus ``ArenaNet``
    parameters + AdamW first/second moments, food/corpse arrays, optional late-state
    reservoir, and Torch zip container framing.

    Args:
        cfg: ``ArenaPPOConfig`` describing model and world dimensions.

    Returns:
        Worst-case single-checkpoint size in bytes.
    """
    if cfg.net_arch == "ego_only":
        approx_params = 300_000 if cfg.net_size == "S" else 1_400_000
    else:
        approx_params = 1_100_000 if cfg.net_size == "S" else 5_200_000
    model_and_opt_bytes = approx_params * 16
    sim_cfg = arena_v2_config(
        int(cfg.num_envs),
        grid_h=int(cfg.grid_h),
        grid_w=int(cfg.grid_w),
        num_snakes=int(cfg.num_snakes),
        max_capacity=int(cfg.max_capacity),
    )
    worst_live_seg = int(cfg.max_capacity)
    bodies_i16_bytes = int(cfg.num_envs) * int(cfg.num_snakes) * worst_live_seg * 2 * 2
    food_cap = int(sim_cfg.max_food + corpse_food_cap(sim_cfg.max_food))
    misc_sim_bytes = int(cfg.num_envs) * (food_cap * 6 + int(cfg.num_snakes) * 180 + 64)
    reservoir_bytes = 0
    if cfg.late_reset_enabled:
        reservoir_bytes = int(cfg.late_reset_reservoir_size) * (
            int(cfg.num_snakes) * worst_live_seg * 2 * 2
            + food_cap * 6
            + int(cfg.num_snakes) * 180
            + 64
        )
    overhead_bytes = 8 * 1024 * 1024
    return int(
        model_and_opt_bytes + bodies_i16_bytes + misc_sim_bytes + reservoir_bytes + overhead_bytes
    )


def estimate_max_export_bytes(
    cfg: ArenaPPOConfig,
    *,
    measured_checkpoint_bytes: int | None = None,
) -> int:
    """Compute per-file ``max_export_bytes`` from worst-case single-checkpoint size x 1.25.

    ``research.colab_cli.runtime._exports`` enforces ``max_export_bytes`` per exported file
    (`candidate.stat().st_size > max_export_bytes`), so the limit must cover a single
    worst-case ``max_capacity`` checkpoint with 25% headroom (`<= 2 GiB`).

    Args:
        cfg: Training configuration.
        measured_checkpoint_bytes: Optional empirically measured single-checkpoint size.

    Returns:
        Positive per-file byte cap ``<= 2 GiB`` (`2,147,483,648` supervisor ceiling).
    """
    est_worst = estimate_checkpoint_bytes(cfg)
    single_file_bytes = (
        max(est_worst, int(measured_checkpoint_bytes))
        if measured_checkpoint_bytes is not None and int(measured_checkpoint_bytes) > 0
        else est_worst
    )
    with_margin = max(64 * 1024 * 1024, int(math.ceil(single_file_bytes * 1.25)))
    supervisor_ceiling = 2 * 1024 * 1024 * 1024
    if with_margin > supervisor_ceiling:
        if single_file_bytes > supervisor_ceiling:
            raise ValueError(
                f"Single checkpoint worst-case size ({single_file_bytes} B) exceeds 2 GiB "
                "supervisor per-file ceiling; reduce num_envs or max_capacity."
            )
        return supervisor_ceiling
    return with_margin


def find_newest_verified_checkpoint(ckpt_dir: Path | str) -> Path | None:
    """Return the newest verified non-probe checkpoint in `ckpt_dir`, or None.

    Verifies `.complete` + `.sha256` digests, skips corrupt or partially written
    candidates without aborting the scan, and refuses `canary_export_probe` stubs.

    Args:
        ckpt_dir: Directory containing `ckpt_*.pt` checkpoints.

    Returns:
        Path to newest verified checkpoint `.pt` file, or ``None`` if none exist.
    """
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        return None

    candidates: list[tuple[int, float, Path]] = []
    for pt_path in directory.glob("ckpt_*.pt"):
        if not (Path(f"{pt_path}.complete").is_file() and Path(f"{pt_path}.sha256").is_file()):
            continue
        try:
            verify_completed_artifact(pt_path)
            meta_path = pt_path.with_suffix(".meta.json")
            updates = -1
            if (
                meta_path.is_file()
                and Path(f"{meta_path}.complete").is_file()
                and Path(f"{meta_path}.sha256").is_file()
            ):
                verify_completed_artifact(meta_path)
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict) or bool(meta.get("canary_export_probe")):
                    continue
                updates = int(meta.get("updates_completed", -1))
            else:
                stem_digits = "".join(ch for ch in pt_path.stem if ch.isdigit())
                updates = int(stem_digits) if stem_digits else -1
            candidates.append((updates, float(pt_path.stat().st_mtime), pt_path))
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError, KeyError, TypeError):
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def prune_old_periodic_checkpoints(ckpt_dir: Path | str, *, keep: int = 3) -> list[str]:
    """Keep only the newest ``keep`` unshared periodic ``ckpt_u*.pt`` files in a work dir.

    Must ONLY be called on a scratch/work directory outside ``$COLAB_CLI_ROOT/output``.
    Skips any file with ``st_nlink > 1`` (i.e. already hard-linked into ``output/``) so
    published exports never have their ``st_ctime_ns`` mutated, and never deletes
    ``ckpt_sig_*.pt`` or ``final.pt``.

    Args:
        ckpt_dir: Work checkpoint directory outside ``output/``.
        keep: Number of newest unshared periodic checkpoints to retain (`>= 1`).

    Returns:
        List of pruned checkpoint filenames.
    """
    keep_n = max(1, int(keep))
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        return []

    periodic: list[tuple[int, float, Path]] = []
    for pt_path in directory.glob("ckpt_u*.pt"):
        if pt_path.name.startswith("ckpt_sig_"):
            continue
        if not (Path(f"{pt_path}.complete").is_file() and Path(f"{pt_path}.sha256").is_file()):
            continue
        try:
            if pt_path.stat().st_nlink > 1:
                continue
            verify_completed_artifact(pt_path)
            stem_digits = "".join(ch for ch in pt_path.stem if ch.isdigit())
            updates = int(stem_digits) if stem_digits else -1
            periodic.append((updates, float(pt_path.stat().st_mtime), pt_path))
        except (OSError, ValueError):
            continue

    periodic.sort(key=lambda item: (item[0], item[1]), reverse=True)
    pruned: list[str] = []
    for _, _, old_pt in periodic[keep_n:]:
        if old_pt.exists() and old_pt.stat().st_nlink > 1:
            continue
        old_meta = old_pt.with_suffix(".meta.json")
        remove_completed_artifact(old_meta)
        remove_completed_artifact(old_pt)
        pruned.append(old_pt.name)
    return pruned


class ArenaPPOTrainer:
    """Self-contained GPU-resident PPO trainer with atomic checkpointing and resume."""

    def __init__(
        self,
        cfg: ArenaPPOConfig,
        *,
        device: str | torch.device = "cpu",
        train_sim: TrainSim | None = None,
        snapshot_model: ArenaPolicyModel | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = normalize_device(device)
        ensure_cuda_initialized(self.device)
        configure_determinism(cfg.deterministic)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed & 0xFFFF_FFFF)
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

        self.perm_gen = torch.Generator(device="cpu")
        self.perm_gen.manual_seed((cfg.seed ^ 0x5050_4F5F_5045_524D) & 0x7FFF_FFFF_FFFF_FFFF)

        self.model: ArenaPolicyModel = build_arena_model(
            cfg.net_arch,
            cfg.net_size,
            grid_h=cfg.grid_h,
            grid_w=cfg.grid_w,
            use_channels_last=cfg.use_channels_last,
            use_bf16_autocast=cfg.amp,
            compile_model=cfg.compile_model,
        ).to(self.device)
        self.snapshot_model: ArenaPolicyModel | None = (
            snapshot_model.to(self.device).eval() if snapshot_model is not None else None
        )
        use_fused_adamw = bool(self.device.type == "cuda" and torch.cuda.is_available())
        adamw_kwargs: dict[str, Any] = {
            "lr": cfg.lr,
            "eps": cfg.adam_eps,
            "weight_decay": cfg.weight_decay,
        }
        if use_fused_adamw:
            adamw_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **adamw_kwargs)
        use_fp16_scaler = bool(
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and self.model.uses_fp16_grad_scaler(self.device)
        )
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

        if train_sim is not None:
            if train_sim.device != self.device:
                raise ValueError(
                    f"train_sim.device={train_sim.device} != trainer.device={self.device}"
                )
            self.sim = train_sim
        else:
            self.sim = TrainSim.create(
                cfg.num_envs,
                device=self.device,
                seed=cfg.seed,
                grid_h=cfg.grid_h,
                grid_w=cfg.grid_w,
                num_snakes=cfg.num_snakes,
                max_capacity=cfg.max_capacity,
            )

        self.buffer = CompactRolloutBuffer(
            rollout_len=cfg.rollout_len,
            num_envs=self.sim.E,
            num_snakes=self.sim.S,
            cap=self.sim.cap,
            food_capacity=self.sim.sim.food_capacity,
            min_boost_length=self.sim.min_boost_length,
            grid_h=self.sim.grid_h,
            grid_w=self.sim.grid_w,
            device=self.device,
        )

        self._has_snapshot_opp = bool(
            (self.sim.world_opponent_kind == OPPONENT_KIND_SNAPSHOT).any().detach().cpu().item()
        )
        self._has_scripted_opp = bool(
            (self.sim.world_opponent_kind == OPPONENT_KIND_SCRIPTED).any().detach().cpu().item()
        )

        self.updates_completed: int = 0
        self.agent_steps_collected: int = 0
        self.world_steps_collected: int = 0
        self.time_limit_resets: int = 0
        self.checkpoints_written: int = 0
        self._last_checkpoint_wall: float = time.monotonic()

        self.episode_return_acc = torch.zeros(
            (self.sim.E, self.sim.S), dtype=torch.float32, device=self.device
        )
        self._last_rollout_mean_ep_return: float = 0.0
        self._pending_late_restore = torch.zeros(
            (self.sim.E,), dtype=torch.bool, device=self.device
        )
        self._has_pending_late_restore: bool = False
        self._rollout_max_frame_bound: int | None = None
        self.late_reservoir: LateStateReservoir | None = (
            LateStateReservoir(
                max_size=cfg.late_reset_reservoir_size,
                min_length=cfg.late_reset_min_length,
                num_snakes=self.sim.S,
                max_capacity=self.sim.cap,
                food_capacity=self.sim.sim.food_capacity,
                device=self.device,
                seed=cfg.seed,
            )
            if cfg.late_reset_enabled
            else None
        )

    def refresh_opponent_flags(self) -> None:
        """Recompute cached opponent-presence flags outside the hot tick loop."""
        self._has_snapshot_opp = bool(
            (self.sim.world_opponent_kind == OPPONENT_KIND_SNAPSHOT).any().detach().cpu().item()
        )
        self._has_scripted_opp = bool(
            (self.sim.world_opponent_kind == OPPONENT_KIND_SCRIPTED).any().detach().cpu().item()
        )

    @torch.no_grad()
    def _act_all_seats_chunked(
        self,
        policy: ArenaPolicyModel,
        legal_mask: torch.Tensor,
        *,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run observation build + policy `act` across `E` worlds in `actor_chunk_envs` slices."""
        e_total, s_total = self.sim.E, self.sim.S
        chunk = max(1, min(e_total, int(self.cfg.actor_chunk_envs)))
        is_per_seat = isinstance(policy, ArenaNet)
        if chunk == e_total:
            if is_per_seat:
                obs = build_arena_ego_obs_gpu(self.sim, ring_chunk=self.cfg.ring_chunk)
                act_out = policy.act(
                    obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=greedy
                )
            else:
                shared_obs = build_shared_world_obs_gpu(self.sim, ring_chunk=self.cfg.ring_chunk)
                act_out = policy.act(shared_obs, legal_mask=legal_mask, greedy=greedy)
            return (
                act_out.actions.reshape(e_total, s_total),
                act_out.log_probs.reshape(e_total, s_total),
                act_out.values.reshape(e_total, s_total),
            )

        actions = torch.empty((e_total, s_total), dtype=torch.long, device=self.device)
        log_probs = torch.empty((e_total, s_total), dtype=torch.float32, device=self.device)
        values = torch.empty((e_total, s_total), dtype=torch.float32, device=self.device)
        for start in range(0, e_total, chunk):
            end = min(e_total, start + chunk)
            sl = slice(start, end)
            if is_per_seat:
                obs = build_arena_ego_obs_gpu(
                    self.sim, env_slice=sl, ring_chunk=self.cfg.ring_chunk
                )
                act_out = policy.act(
                    obs.arena,
                    obs.ego,
                    obs.scalars,
                    legal_mask=legal_mask[sl],
                    greedy=greedy,
                )
            else:
                shared_obs = build_shared_world_obs_gpu(
                    self.sim, env_slice=sl, ring_chunk=self.cfg.ring_chunk
                )
                act_out = policy.act(
                    shared_obs,
                    legal_mask=legal_mask[sl],
                    greedy=greedy,
                )
            actions[sl] = act_out.actions.reshape(end - start, s_total)
            log_probs[sl] = act_out.log_probs.reshape(end - start, s_total)
            values[sl] = act_out.values.reshape(end - start, s_total)
        return actions, log_probs, values

    @torch.no_grad()
    def _bootstrap_values_chunked(self) -> torch.Tensor:
        """Compute scalar value predictions `(E, S)` on current prepared `TrainSim` state."""
        e_total, s_total = self.sim.E, self.sim.S
        legal_mask = self.sim.legal_action_mask()
        chunk = max(1, min(e_total, int(self.cfg.actor_chunk_envs)))
        is_per_seat = isinstance(self.model, ArenaNet)
        if chunk == e_total:
            if is_per_seat:
                obs = build_arena_ego_obs_gpu(self.sim, ring_chunk=self.cfg.ring_chunk)
                out = self.model.forward(obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask)
            else:
                shared_obs = build_shared_world_obs_gpu(self.sim, ring_chunk=self.cfg.ring_chunk)
                out = self.model.forward(shared_obs, legal_mask=legal_mask)
            return out.value.reshape(e_total, s_total)

        values = torch.empty((e_total, s_total), dtype=torch.float32, device=self.device)
        for start in range(0, e_total, chunk):
            end = min(e_total, start + chunk)
            sl = slice(start, end)
            if is_per_seat:
                obs = build_arena_ego_obs_gpu(
                    self.sim, env_slice=sl, ring_chunk=self.cfg.ring_chunk
                )
                out = self.model.forward(obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask[sl])
            else:
                shared_obs = build_shared_world_obs_gpu(
                    self.sim, env_slice=sl, ring_chunk=self.cfg.ring_chunk
                )
                out = self.model.forward(shared_obs, legal_mask=legal_mask[sl])
            values[sl] = out.value.reshape(end - start, s_total)
        return values

    def _maybe_apply_world_time_limits(self, t: int, survived: torch.Tensor) -> None:
        """Handle optional `max_world_frames` world resets with prepared `s_{t+1}` bootstrap."""
        max_frames = int(self.cfg.max_world_frames)
        if max_frames <= 0:
            self.buffer.truncated[t].zero_()
            self.buffer.trunc_bootstrap[t].zero_()
            return

        if self._rollout_max_frame_bound is not None and self._rollout_max_frame_bound < max_frames:
            self.buffer.truncated[t].zero_()
            self.buffer.trunc_bootstrap[t].zero_()
            return

        hit = self.sim.sim.frame >= max_frames
        if not bool(hit.any().detach().cpu().item()):
            self.buffer.truncated[t].zero_()
            self.buffer.trunc_bootstrap[t].zero_()
            return

        # Prepare s_{t+1} across all worlds in exact v2 order first; subsequent prepare()
        # at t + 1 (or rollout cut) is an idempotent no-op, keeping ~hit worlds byte-identical.
        self.sim.prepare()
        if self.late_reservoir is not None and self.cfg.late_reset_enabled:
            self.late_reservoir.harvest(self.sim)
        with torch.no_grad():
            pre_reset_vals = self._bootstrap_values_chunked()
        trunc_mask = hit[:, None] & survived
        self.buffer.truncated[t] = trunc_mask
        self.buffer.trunc_bootstrap[t] = torch.where(
            trunc_mask, pre_reset_vals, torch.zeros_like(pre_reset_vals)
        )

        # Mid-rollout time-limit hits ALWAYS receive a fresh 1-segment reset so append-only
        # `CompactRolloutBuffer` body_tape remains 100% lossless; queue reservoir restore for
        # the next rollout start (`t=0`, before full body_tape snapshot).
        self.sim.reset_worlds(hit)
        self.time_limit_resets += int(hit.sum().detach().cpu().item())
        if self.late_reservoir is not None and self.cfg.late_reset_enabled:
            self._pending_late_restore |= hit
            self._has_pending_late_restore = True
        if self._rollout_max_frame_bound is not None:
            self._rollout_max_frame_bound = int(self.sim.sim.frame.max().detach().cpu().item())

    @torch.no_grad()
    def collect_rollout(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> float:
        """Collect `T` ticks across `E * S` seats into `self.buffer`.

        Args:
            should_stop: Optional cooperative stop predicate checked every tick.

        Returns:
            Elapsed wall seconds spent in rollout collection + GAE computation.
        """
        self.model.eval()
        t0 = time.perf_counter()
        ep_ret_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        ep_ret_count = torch.zeros((), dtype=torch.float32, device=self.device)
        zero_es = torch.zeros((self.sim.E, self.sim.S), dtype=torch.float32, device=self.device)

        # Prepare t=0 pre-decision state and perform any queued LateStateReservoir restores
        # strictly BEFORE `record_pre_step(0, ...)` snapshots the full ring buffer.
        self.sim.prepare()
        if (
            self.late_reservoir is not None
            and self.cfg.late_reset_enabled
            and self._has_pending_late_restore
        ):
            restored_mask = self.late_reservoir.restore_from_reservoir(
                self.sim,
                self._pending_late_restore,
                restore_prob=self.cfg.late_reset_prob,
            )
            self._pending_late_restore.zero_()
            self._has_pending_late_restore = False
            self.episode_return_acc[restored_mask] = 0.0

        self._rollout_max_frame_bound = (
            int(self.sim.sim.frame.max().detach().cpu().item())
            if int(self.cfg.max_world_frames) > 0
            else None
        )
        try:
            for t in range(self.cfg.rollout_len):
                if should_stop is not None and should_stop():
                    raise StopRequestedError("Signal stop requested during rollout collection")
                was_unprepared = not bool(self.sim.sim._prepared)
                self.sim.prepare()
                if was_unprepared and self._rollout_max_frame_bound is not None:
                    self._rollout_max_frame_bound += 1
                self.buffer.record_pre_step(t, self.sim)
                legal_mask = self.sim.legal_action_mask()

                self_actions, log_probs, values = self._act_all_seats_chunked(
                    self.model, legal_mask, greedy=False
                )
                snap_actions = None
                if self._has_snapshot_opp and self.snapshot_model is not None:
                    snap_actions, _, _ = self._act_all_seats_chunked(
                        self.snapshot_model, legal_mask, greedy=False
                    )
                scripted_actions = None
                if self._has_scripted_opp:
                    scripted_actions = self.sim.scripted_greedy_food_actions()

                world_actions = self.sim.compose_world_actions(
                    self_actions,
                    snapshot_actions=snap_actions,
                    scripted_actions=scripted_actions,
                )
                info = self.sim.step(world_actions)

                self.buffer.actions[t] = self_actions.to(torch.uint8)
                self.buffer.log_probs[t] = log_probs
                self.buffer.values[t] = values
                self.buffer.rewards[t] = info.reward
                self.buffer.died[t] = info.died
                self.buffer.acted[t] = info.acted
                self.buffer.learner_mask[t] = info.learner_mask
                self._maybe_apply_world_time_limits(t, info.survived)

                self.episode_return_acc = torch.where(
                    info.acted, self.episode_return_acc + info.reward, zero_es
                )
                ep_done = info.acted & (info.died | self.buffer.truncated[t])
                ep_done_learner = ep_done & info.learner_mask
                ep_ret_sum = (
                    ep_ret_sum
                    + torch.where(ep_done_learner, self.episode_return_acc, zero_es).sum()
                )
                ep_ret_count = ep_ret_count + ep_done_learner.to(torch.float32).sum()
                self.episode_return_acc = torch.where(
                    ep_done | (~info.survived), zero_es, self.episode_return_acc
                )
        finally:
            self._rollout_max_frame_bound = None

        self.sim.assert_no_capacity_overflow()

        # Rollout-cut bootstrap on state `T`
        self.sim.prepare()
        if self.late_reservoir is not None and self.cfg.late_reset_enabled:
            self.late_reservoir.harvest(self.sim)
        last_values = self._bootstrap_values_chunked()
        self.buffer.compute_gae_and_aux(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        active_learner = self.sim.sim.alive & self.sim.learner_seat_mask()
        active_cnt = active_learner.to(torch.float32).sum().clamp_min(1.0)
        fallback_ep_ret = torch.where(active_learner, self.episode_return_acc, zero_es).sum() / (
            active_cnt
        )
        mean_ep_ret_t = torch.where(
            ep_ret_count > 0.0,
            ep_ret_sum / ep_ret_count.clamp_min(1.0),
            fallback_ep_ret,
        )

        synchronize_device(self.device)
        elapsed = time.perf_counter() - t0
        self._last_rollout_mean_ep_return = float(mean_ep_ret_t.detach().cpu().item())
        self.world_steps_collected += self.cfg.rollout_len * self.sim.E
        self.agent_steps_collected += self.cfg.rollout_len * self.sim.E * self.sim.S
        return elapsed

    def ppo_update(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, float], float]:
        """Run `ppo_epochs` of minibatch clipped PPO + symlog two-hot value + aux BCE updates."""
        assert self.cfg.minibatch_size is not None
        self.model.train()
        t0 = time.perf_counter()

        valid_3d = self.buffer.acted & self.buffer.learner_mask
        valid_3d_f = valid_3d.to(torch.float32)
        valid_3d_denom = valid_3d_f.sum().clamp_min(1.0)
        zero_3d = torch.zeros_like(self.buffer.returns)
        ret_clean = torch.where(valid_3d, self.buffer.returns, zero_3d)
        val_clean = torch.where(valid_3d, self.buffer.values, zero_3d)
        with torch.no_grad():
            ret_mean = (ret_clean * valid_3d_f).sum() / valid_3d_denom
            ret_var = (((ret_clean - ret_mean).square()) * valid_3d_f).sum() / valid_3d_denom
            val_err = ret_clean - val_clean
            err_mean = (val_err * valid_3d_f).sum() / valid_3d_denom
            err_var = (((val_err - err_mean).square()) * valid_3d_f).sum() / valid_3d_denom
            explained_var_t = torch.where(
                ret_var > 1.0e-8,
                1.0 - (err_var / ret_var.clamp_min(1.0e-8)),
                torch.zeros_like(ret_var),
            )

        adv_all = torch.where(valid_3d, self.buffer.advantages, zero_3d)
        if self.cfg.normalize_advantages:
            mean_adv = (adv_all * valid_3d_f).sum() / valid_3d_denom
            var_adv = (
                ((adv_all - mean_adv) * valid_3d_f).square().sum() / valid_3d_denom
            ).clamp_min(1e-8)
            norm_adv = torch.where(valid_3d, (adv_all - mean_adv) * torch.rsqrt(var_adv), zero_3d)
        else:
            norm_adv = adv_all

        total_world_ticks = self.cfg.rollout_len * self.sim.E
        mb_worlds = max(1, min(total_world_ticks, int(self.cfg.minibatch_size) // self.sim.S))
        vmin = float(self.model.cfg.value_symlog_min)
        vmax = float(self.model.cfg.value_symlog_max)
        is_per_seat = isinstance(self.model, ArenaNet)

        # Single host sync per update to bound body-tape segment gather & tighten live sim bound.
        if self.buffer.seg_count.numel() > 0 and self.sim.sim.seg_count.numel() > 0:
            seg_max_pair = (
                torch.stack(
                    (
                        self.buffer.seg_count.max().to(torch.int64),
                        self.sim.sim.seg_count.max().to(torch.int64),
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            rollout_max_seg = max(1, min(self.buffer.cap, int(seg_max_pair[0])))
            self.sim._max_seg_bound = max(1, min(self.sim.cap, int(seg_max_pair[1])))
        else:
            rollout_max_seg = 1
            self.sim._max_seg_bound = 1

        # Pre-flatten & cast static rollout targets once per update (avoid repeating per epoch).
        n_aux = len(AUX_DEATH_HORIZONS)
        s_seats = self.sim.S
        valid_wt = valid_3d.reshape(total_world_ticks, s_seats)
        valid_f_wt = valid_3d_f.reshape(total_world_ticks, s_seats)
        raw_actions_wt = self.buffer.actions.reshape(total_world_ticks, s_seats).to(torch.long)
        actions_wt = torch.where(valid_wt, raw_actions_wt, torch.zeros_like(raw_actions_wt))
        old_logp_wt = torch.where(valid_3d, self.buffer.log_probs, zero_3d).reshape(
            total_world_ticks, s_seats
        )
        adv_wt = norm_adv.reshape(total_world_ticks, s_seats)
        returns_wt = ret_clean.reshape(total_world_ticks, s_seats)
        aux_mask_wt = (
            self.buffer.aux_valid.reshape(total_world_ticks, s_seats, n_aux) & valid_wt[:, :, None]
        )
        raw_aux_wt = self.buffer.aux_targets.reshape(total_world_ticks, s_seats, n_aux)
        aux_targets_wt = torch.where(aux_mask_wt, raw_aux_wt, torch.zeros_like(raw_aux_wt))
        aux_mask_f_wt = aux_mask_wt.to(torch.float32)

        policy_loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        value_loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        aux_loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        aux_per_h_acc = torch.zeros((n_aux,), dtype=torch.float32, device=self.device)
        entropy_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        kl_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        epoch0_kl_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        clip_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        grad_norm_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        num_batches = 0
        epoch0_batches = 0

        for epoch_idx in range(self.cfg.ppo_epochs):
            perm_cpu = torch.randperm(total_world_ticks, generator=self.perm_gen)
            perm = perm_cpu.to(device=self.device, dtype=torch.long)
            for start in range(0, total_world_ticks, mb_worlds):
                if should_stop is not None and should_stop():
                    raise StopRequestedError("Signal stop requested during learner minibatch")
                flat_wt = perm[start : start + mb_worlds]
                t_idx = torch.div(flat_wt, self.sim.E, rounding_mode="floor")
                e_idx = flat_wt.remainder(self.sim.E)

                valid_f = valid_f_wt.index_select(0, flat_wt).reshape(-1)
                mb_actions = actions_wt.index_select(0, flat_wt).reshape(-1)
                mb_old_logp = old_logp_wt.index_select(0, flat_wt).reshape(-1)
                mb_adv = adv_wt.index_select(0, flat_wt).reshape(-1)
                mb_returns = returns_wt.index_select(0, flat_wt).reshape(-1)
                mb_aux_targets = aux_targets_wt.index_select(0, flat_wt).reshape(-1, n_aux)
                aux_mask_f = aux_mask_f_wt.index_select(0, flat_wt).reshape(-1, n_aux)

                if is_per_seat:
                    arena, ego, scalars, legal_mask = self.buffer.reconstruct_world_ticks(
                        t_idx,
                        e_idx,
                        ring_chunk=self.cfg.ring_chunk,
                        max_seg=rollout_max_seg,
                    )
                    new_logp, entropy, val_logits, aux_logits = self.model.evaluate_actions(
                        arena, ego, scalars, legal_mask, mb_actions
                    )
                else:
                    shared_obs, legal_mask = self.buffer.reconstruct_shared_world_ticks(
                        t_idx,
                        e_idx,
                        ring_chunk=self.cfg.ring_chunk,
                        max_seg=rollout_max_seg,
                    )
                    new_logp, entropy, val_logits, aux_logits = self.model.evaluate_actions(
                        shared_obs, legal_mask, mb_actions
                    )

                valid_count = valid_f.sum().clamp_min(1.0)

                log_ratio = (new_logp - mb_old_logp).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                    * mb_adv
                )
                pg_loss = -(torch.minimum(surr1, surr2) * valid_f).sum() / valid_count

                v_loss_rows = symlog_two_hot_loss(
                    val_logits,
                    mb_returns,
                    self.model.value_support,
                    vmin=vmin,
                    vmax=vmax,
                )
                v_loss = (v_loss_rows * valid_f).sum() / valid_count

                ent_mean = (entropy * valid_f).sum() / valid_count

                aux_denom = aux_mask_f.sum().clamp_min(1.0)
                aux_bce = F.binary_cross_entropy_with_logits(
                    aux_logits, mb_aux_targets, reduction="none"
                )
                aux_loss = (aux_bce * aux_mask_f).sum() / aux_denom

                loss = (
                    pg_loss
                    + float(self.cfg.value_coef) * v_loss
                    + float(self.cfg.aux_coef) * aux_loss
                    - float(self.cfg.entropy_coef) * ent_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.cfg.max_grad_norm
                    )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.cfg.max_grad_norm
                    )
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (((ratio - 1.0) - log_ratio) * valid_f).sum() / valid_count
                    clipped = (
                        ((ratio - 1.0).abs() > self.cfg.clip_ratio).to(torch.float32) * valid_f
                    ).sum() / valid_count
                    aux_denom_h = aux_mask_f.sum(dim=0).clamp_min(1.0)
                    aux_per_h = (aux_bce.detach() * aux_mask_f).sum(dim=0) / aux_denom_h
                    policy_loss_acc += pg_loss.detach()
                    value_loss_acc += v_loss.detach()
                    aux_loss_acc += aux_loss.detach()
                    aux_per_h_acc += aux_per_h
                    entropy_acc += ent_mean.detach()
                    kl_acc += approx_kl.detach()
                    if epoch_idx == 0:
                        epoch0_kl_acc += approx_kl.detach()
                        epoch0_batches += 1
                    clip_acc += clipped.detach()
                    grad_norm_acc += torch.as_tensor(
                        grad_norm, dtype=torch.float32, device=self.device
                    )
                    num_batches += 1

        synchronize_device(self.device)
        elapsed = time.perf_counter() - t0
        self.updates_completed += 1
        inv_b = 1.0 / float(max(1, num_batches))
        inv_e0 = 1.0 / float(max(1, epoch0_batches))
        stats_gpu = (
            torch.cat(
                (
                    torch.stack(
                        (
                            policy_loss_acc * inv_b,
                            value_loss_acc * inv_b,
                            aux_loss_acc * inv_b,
                            entropy_acc * inv_b,
                            kl_acc * inv_b,
                            epoch0_kl_acc * inv_e0,
                            clip_acc * inv_b,
                            grad_norm_acc * inv_b,
                            explained_var_t,
                        )
                    ),
                    aux_per_h_acc * inv_b,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        return (
            {
                "policy_loss": float(stats_gpu[0]),
                "value_loss": float(stats_gpu[1]),
                "aux_loss": float(stats_gpu[2]),
                "entropy": float(stats_gpu[3]),
                "approx_kl": float(stats_gpu[4]),
                "epoch0_approx_kl": float(stats_gpu[5]),
                "clip_fraction": float(stats_gpu[6]),
                "grad_norm": float(stats_gpu[7]),
                "value_explained_variance": float(stats_gpu[8]),
                "aux_loss_h16": float(stats_gpu[9]),
                "aux_loss_h64": float(stats_gpu[10]),
                "aux_loss_h256": float(stats_gpu[11]),
            },
            elapsed,
        )

    def step_update(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> PPOUpdateMetrics:
        """Execute one full `collect_rollout()` + `ppo_update()` iteration."""
        collect_sec = self.collect_rollout(should_stop=should_stop)
        stats, learn_sec = self.ppo_update(should_stop=should_stop)

        with torch.no_grad():
            valid_mask = self.buffer.acted & self.buffer.learner_mask
            valid_f = valid_mask.to(torch.float32)
            valid_sum_t = valid_f.sum()
            acted_sum_t = self.buffer.acted.to(torch.int64).sum()
            deaths_sum_t = self.buffer.died.to(torch.int64).sum()
            mean_rew_t = torch.where(
                valid_sum_t > 0.0,
                (self.buffer.rewards * valid_f).sum() / valid_sum_t.clamp_min(1.0),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )

            alive_f = self.buffer.alive.to(torch.float32)
            alive_denom = alive_f.sum()
            len_f = self.buffer.length.to(torch.float32)
            mean_len_t = torch.where(
                alive_denom > 0.0,
                (len_f * alive_f).sum() / alive_denom.clamp_min(1.0),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
            stride = max(1, self.buffer.T // 16)
            sub_len = len_f[::stride]
            sub_alive = self.buffer.alive[::stride]
            nan_fill = torch.full_like(sub_len, float("nan"))
            masked_len = torch.where(sub_alive, sub_len, nan_fill).reshape(-1)
            p95_t = torch.nanquantile(masked_len, 0.95)

            # Single host sync for all integer & float rollout summary scalars
            int_host = (
                torch.stack((valid_mask.to(torch.int64).sum(), acted_sum_t, deaths_sum_t))
                .detach()
                .cpu()
                .tolist()
            )
            flt_host = torch.stack((mean_rew_t, mean_len_t, p95_t)).detach().cpu().tolist()
            learner_rows = int(int_host[0])
            acted_steps = int(int_host[1])
            deaths_count = int(int_host[2])
            mean_reward = float(flt_host[0]) if learner_rows > 0 else 0.0
            mean_length = 0.0 if math.isnan(flt_host[1]) else float(flt_host[1])
            p95_length = 0.0 if math.isnan(flt_host[2]) else float(flt_host[2])

        total_sec = max(1.0e-9, float(collect_sec + learn_sec))
        rollout_agent_steps = self.cfg.rollout_len * self.sim.E * self.sim.S
        steps_per_sec = float(rollout_agent_steps) / total_sec
        deaths_per_1k = 1000.0 * float(deaths_count) / float(max(1, acted_steps))

        return PPOUpdateMetrics(
            update_idx=self.updates_completed,
            agent_steps_collected=self.agent_steps_collected,
            world_steps_collected=self.world_steps_collected,
            learner_rows=learner_rows,
            mean_reward=mean_reward,
            deaths_count=deaths_count,
            policy_loss=stats["policy_loss"],
            value_loss=stats["value_loss"],
            aux_loss=stats["aux_loss"],
            entropy=stats["entropy"],
            approx_kl=stats["approx_kl"],
            epoch0_approx_kl=stats["epoch0_approx_kl"],
            clip_fraction=stats["clip_fraction"],
            grad_norm=stats["grad_norm"],
            collect_sec=collect_sec,
            learn_sec=learn_sec,
            acted_steps=acted_steps,
            steps_per_sec=steps_per_sec,
            mean_length=mean_length,
            p95_length=p95_length,
            deaths_per_1k_agent_steps=deaths_per_1k,
            mean_episode_return=self._last_rollout_mean_ep_return,
            value_explained_variance=stats["value_explained_variance"],
            aux_loss_h16=stats["aux_loss_h16"],
            aux_loss_h64=stats["aux_loss_h64"],
            aux_loss_h256=stats["aux_loss_h256"],
        )

    def checkpoint_dict(self) -> dict[str, Any]:
        """Build full deterministic state dict (`model`, `optimizer`, `sim`, `RNG`, counters)."""
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        return {
            "schema": ARENA_PPO_SCHEMA,
            "namespace": ARENA_PPO_NAMESPACE,
            "config": self.cfg.to_dict(),
            "net_config": self.model.cfg.to_dict(),
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            },
            "snapshot_state_dict": (
                {k: v.detach().cpu().clone() for k, v in self.snapshot_model.state_dict().items()}
                if self.snapshot_model is not None
                else None
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "grad_scaler_state_dict": (
                self.grad_scaler.state_dict() if self.grad_scaler.is_enabled() else None
            ),
            "sim_state_dict": self.sim.state_dict(),
            "episode_return_acc": self.episode_return_acc.detach().cpu().clone(),
            "pending_late_restore": self._pending_late_restore.detach().cpu().clone(),
            "late_reservoir": (
                self.late_reservoir.state_dict() if self.late_reservoir is not None else None
            ),
            "rng": {
                "torch_cpu": torch.get_rng_state().clone(),
                "torch_cuda": cuda_rng,
                "perm_gen": self.perm_gen.get_state().clone(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            "counters": {
                "updates_completed": int(self.updates_completed),
                "agent_steps_collected": int(self.agent_steps_collected),
                "world_steps_collected": int(self.world_steps_collected),
                "time_limit_resets": int(self.time_limit_resets),
                "checkpoints_written": int(self.checkpoints_written),
            },
        }

    def save_checkpoint(self, path: Path | str, *, reason: str = "periodic") -> tuple[str, str]:
        """Atomically save checkpoint `.pt` + `.meta.json` with `.sha256` + `.complete` markers."""
        target = Path(path)
        payload = self.checkpoint_dict()
        payload["reason"] = reason
        pt_sha = write_create_only_artifact(target, lambda tmp: torch.save(payload, tmp))
        meta = {
            "schema": ARENA_PPO_SCHEMA,
            "namespace": ARENA_PPO_NAMESPACE,
            "reason": reason,
            "checkpoint_file": target.name,
            "checkpoint_sha256": pt_sha,
            "updates_completed": int(self.updates_completed),
            "agent_steps_collected": int(self.agent_steps_collected),
            "world_steps_collected": int(self.world_steps_collected),
            "canary_export_probe": False,
        }
        meta_path = target.with_suffix(".meta.json")
        meta_sha = write_create_only_json(meta_path, meta)
        self.checkpoints_written += 1
        self._last_checkpoint_wall = time.monotonic()
        return pt_sha, meta_sha

    def save_eval_weights(self, path: Path | str) -> str:
        """Export a compact eval-ready weights file (`arena_ppo_weights.pth`)."""
        obs_spec = "arena_ego_v1" if isinstance(self.model, ArenaNet) else "shared_world_ego_v1"
        payload = {
            "schema": "arena-ppo-eval-weights/v1",
            "obs_spec": obs_spec,
            "net_arch": self.cfg.net_arch,
            "net_config": self.model.cfg.to_dict(),
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "updates_completed": int(self.updates_completed),
            "agent_steps_collected": int(self.agent_steps_collected),
            "canary_export_probe": False,
        }
        return write_create_only_artifact(Path(path), lambda tmp: torch.save(payload, tmp))

    def load_checkpoint(self, path: Path | str, *, verify_markers: bool = True) -> dict[str, Any]:
        """Verify and restore full trainer state (`model`, `optimizer`, `sim`, `RNG`, counters)."""
        ckpt_path = Path(path)
        if verify_markers:
            verify_completed_artifact(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or bool(ckpt.get("canary_export_probe")):
            raise ValueError(f"Refusing invalid or canary_export_probe checkpoint: {ckpt_path}")
        if ckpt.get("schema") != ARENA_PPO_SCHEMA:
            raise ValueError(f"Foreign checkpoint schema {ckpt.get('schema')!r} at {ckpt_path}")
        if ckpt.get("namespace") != ARENA_PPO_NAMESPACE:
            raise ValueError(
                f"Foreign checkpoint namespace {ckpt.get('namespace')!r} at {ckpt_path}"
            )
        ckpt_cfg_raw = ckpt.get("config")
        if not isinstance(ckpt_cfg_raw, dict):
            raise ValueError(f"Checkpoint missing config dictionary at {ckpt_path}")
        ckpt_cfg_input = dict(ckpt_cfg_raw)
        if "net_arch" not in ckpt_cfg_input:
            ckpt_net_cfg = ckpt.get("net_config")
            ckpt_cfg_input["net_arch"] = (
                str(ckpt_net_cfg.get("net_arch", "per_seat"))
                if isinstance(ckpt_net_cfg, dict)
                else "per_seat"
            )
        ckpt_cfg = ArenaPPOConfig.from_dict(ckpt_cfg_input).to_dict()
        curr_cfg = self.cfg.to_dict()
        priority_keys = ("net_arch", "net_size", "num_envs")
        ordered_keys = list(priority_keys) + sorted(
            (set(ckpt_cfg) | set(curr_cfg)) - set(priority_keys)
        )
        for key in ordered_keys:
            if key in _RESUME_MUTABLE_CONFIG_KEYS:
                continue
            if ckpt_cfg.get(key) != curr_cfg.get(key):
                raise ValueError(
                    f"Checkpoint config mismatch on {key!r}: "
                    f"ckpt={ckpt_cfg.get(key)!r} vs current={curr_cfg.get(key)!r}"
                )

        self.model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("snapshot_state_dict") is not None and self.snapshot_model is not None:
            self.snapshot_model.load_state_dict(ckpt["snapshot_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        use_fused_now = bool(self.device.type == "cuda" and torch.cuda.is_available())
        for group in self.optimizer.param_groups:
            if "fused" in group or use_fused_now:
                group["fused"] = use_fused_now
        if self.grad_scaler.is_enabled() and isinstance(ckpt.get("grad_scaler_state_dict"), dict):
            self.grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
        self.sim.load_state_dict(ckpt["sim_state_dict"])
        if "episode_return_acc" in ckpt and ckpt["episode_return_acc"] is not None:
            self.episode_return_acc.copy_(
                ckpt["episode_return_acc"].to(device=self.device, dtype=torch.float32)
            )
        else:
            self.episode_return_acc.zero_()
        if "pending_late_restore" in ckpt and ckpt["pending_late_restore"] is not None:
            pending_lr = ckpt["pending_late_restore"]
            if (
                not isinstance(pending_lr, torch.Tensor)
                or pending_lr.dtype != torch.bool
                or tuple(pending_lr.shape) != (self.sim.E,)
            ):
                raise ValueError(
                    f"Invalid pending_late_restore in {ckpt_path}: expected bool tensor of shape "
                    f"{(self.sim.E,)}, got {type(pending_lr).__name__} "
                    f"dtype={getattr(pending_lr, 'dtype', None)} "
                    f"shape={getattr(pending_lr, 'shape', None)}"
                )
            self._pending_late_restore.copy_(pending_lr.to(device=self.device, dtype=torch.bool))
            self._has_pending_late_restore = bool(
                self._pending_late_restore.any().detach().cpu().item()
            )
        else:
            self._pending_late_restore.zero_()
            self._has_pending_late_restore = False
        if self.late_reservoir is not None and ckpt.get("late_reservoir") is not None:
            self.late_reservoir.load_state_dict(ckpt["late_reservoir"])
        self.refresh_opponent_flags()

        counters = ckpt["counters"]
        self.updates_completed = int(counters["updates_completed"])
        self.agent_steps_collected = int(counters["agent_steps_collected"])
        self.world_steps_collected = int(counters["world_steps_collected"])
        self.time_limit_resets = int(counters.get("time_limit_resets", 0))
        self.checkpoints_written = int(counters.get("checkpoints_written", 0))

        # Restore RNG states LAST so tensor loads/allocations above do not advance RNG streams.
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch_cpu"])
        if self.device.type == "cuda" and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        self.perm_gen.set_state(rng["perm_gen"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        self._last_checkpoint_wall = time.monotonic()
        return ckpt


def verify_staged_checkpoint(
    path: Path | str,
    *,
    device: str | torch.device = "cpu",
    verify_markers: bool = False,
) -> dict[str, Any]:
    """Verify a staged continuation checkpoint without allocating rollout or simulator buffers."""
    ckpt_path = Path(path)
    if verify_markers:
        verify_completed_artifact(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or bool(ckpt.get("canary_export_probe")):
        raise ValueError(f"Refusing invalid or canary_export_probe checkpoint: {ckpt_path}")
    if ckpt.get("schema") != ARENA_PPO_SCHEMA:
        raise ValueError(f"Foreign checkpoint schema {ckpt.get('schema')!r} at {ckpt_path}")
    if ckpt.get("namespace") != ARENA_PPO_NAMESPACE:
        raise ValueError(f"Foreign checkpoint namespace {ckpt.get('namespace')!r} at {ckpt_path}")

    ckpt_cfg_raw = ckpt.get("config")
    if not isinstance(ckpt_cfg_raw, dict):
        raise ValueError(f"Checkpoint missing config dictionary at {ckpt_path}")
    ckpt_cfg_input = dict(ckpt_cfg_raw)
    if "net_arch" not in ckpt_cfg_input:
        ckpt_net_cfg = ckpt.get("net_config")
        ckpt_cfg_input["net_arch"] = (
            str(ckpt_net_cfg.get("net_arch", "per_seat"))
            if isinstance(ckpt_net_cfg, dict)
            else "per_seat"
        )
    cfg = ArenaPPOConfig.from_dict(ckpt_cfg_input)

    resolved_dev = normalize_device(device)
    model = build_arena_model(
        cfg.net_arch,
        cfg.net_size,
        grid_h=cfg.grid_h,
        grid_w=cfg.grid_w,
        use_channels_last=cfg.use_channels_last,
        use_bf16_autocast=cfg.use_bf16_autocast,
        compile_model=False,
    ).to(resolved_dev)
    model.load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("snapshot_state_dict") is not None:
        snap_model = build_arena_model(
            cfg.net_arch,
            cfg.net_size,
            grid_h=cfg.grid_h,
            grid_w=cfg.grid_w,
            use_channels_last=cfg.use_channels_last,
            use_bf16_autocast=cfg.use_bf16_autocast,
            compile_model=False,
        ).to(resolved_dev)
        snap_model.load_state_dict(ckpt["snapshot_state_dict"])
        del snap_model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    del optimizer, model

    sim_state = ckpt.get("sim_state_dict")
    if not isinstance(sim_state, dict):
        raise ValueError(f"Checkpoint missing sim_state_dict at {ckpt_path}")
    ckpt_shape = (
        int(sim_state["E"]),
        int(sim_state["S"]),
        int(sim_state["cap"]),
        int(sim_state.get("grid_h", cfg.grid_h)),
        int(sim_state.get("grid_w", cfg.grid_w)),
    )
    exp_shape = (cfg.num_envs, cfg.num_snakes, cfg.max_capacity, cfg.grid_h, cfg.grid_w)
    if ckpt_shape != exp_shape:
        raise ValueError(
            f"Checkpoint TrainSim shape mismatch at {ckpt_path}: "
            f"ckpt={ckpt_shape} vs config={exp_shape}"
        )
    sim_tensors = sim_state.get("sim_tensors")
    if not isinstance(sim_tensors, dict):
        raise ValueError(f"Checkpoint missing sim_tensors at {ckpt_path}")
    for name in _SIM_STATE_TENSORS:
        if name == "bodies" and "bodies_live_i16" in sim_state:
            continue
        if name == "capacity_overflow" and name not in sim_tensors:
            continue
        tensor = sim_tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.shape[:1] != (cfg.num_envs,):
            raise ValueError(f"Invalid sim_tensors[{name!r}] in {ckpt_path}")
    if "bodies_live_i16" in sim_state:
        live = sim_state["bodies_live_i16"]
        if (
            not isinstance(live, torch.Tensor)
            or live.ndim != 4
            or live.shape[:2] != (cfg.num_envs, cfg.num_snakes)
            or live.shape[3] != 2
        ):
            raise ValueError(f"Invalid bodies_live_i16 shape in {ckpt_path}")
    for req_sim_key in ("prev_heads", "prev_alive", "world_opponent_kind"):
        if req_sim_key not in sim_state:
            raise ValueError(f"Checkpoint sim_state_dict missing {req_sim_key!r} at {ckpt_path}")

    if "pending_late_restore" in ckpt and ckpt["pending_late_restore"] is not None:
        pending_lr = ckpt["pending_late_restore"]
        if (
            not isinstance(pending_lr, torch.Tensor)
            or pending_lr.dtype != torch.bool
            or tuple(pending_lr.shape) != (cfg.num_envs,)
        ):
            raise ValueError(
                f"Invalid pending_late_restore in {ckpt_path}: expected bool tensor of shape "
                f"{(cfg.num_envs,)}, got {type(pending_lr).__name__} "
                f"dtype={getattr(pending_lr, 'dtype', None)} "
                f"shape={getattr(pending_lr, 'shape', None)}"
            )
    elif cfg.late_reset_enabled or ckpt.get("late_reservoir") is not None:
        raise ValueError(f"Checkpoint missing pending_late_restore tensor at {ckpt_path}")

    if ckpt.get("late_reservoir") is not None:
        lr_state = ckpt["late_reservoir"]
        if (
            not isinstance(lr_state, dict)
            or int(lr_state["S"]) != cfg.num_snakes
            or int(lr_state["cap"]) != cfg.max_capacity
            or int(lr_state["size"]) > int(cfg.late_reset_reservoir_size)
        ):
            raise ValueError(f"Invalid late_reservoir state in {ckpt_path}")

    counters = ckpt.get("counters")
    rng = ckpt.get("rng")
    if not isinstance(counters, dict) or not isinstance(rng, dict):
        raise ValueError(f"Checkpoint missing counters/rng dictionaries at {ckpt_path}")
    for req_rng_key in ("torch_cpu", "perm_gen", "python", "numpy"):
        if req_rng_key not in rng:
            raise ValueError(f"Checkpoint rng missing {req_rng_key!r} at {ckpt_path}")

    return {
        "schema": ARENA_PPO_SCHEMA,
        "namespace": ARENA_PPO_NAMESPACE,
        "config": cfg.to_dict(),
        "updates_completed": int(counters["updates_completed"]),
        "agent_steps_collected": int(counters["agent_steps_collected"]),
        "world_steps_collected": int(counters["world_steps_collected"]),
    }


def _resolve_work_dir(output_dir: Path, work_dir: Path | None) -> Path:
    """Resolve a scratch checkpoint directory strictly outside `output_dir`."""
    out_resolved = output_dir.resolve()
    if work_dir is not None:
        resolved = work_dir.resolve()
    else:
        cli_root = os.environ.get("COLAB_CLI_ROOT")
        if cli_root and Path(cli_root).resolve() in out_resolved.parents:
            resolved = Path(cli_root).resolve() / "work" / "arena_ppo"
        else:
            resolved = out_resolved.parent / f".{out_resolved.name}_work"
    if resolved == out_resolved or out_resolved in resolved.parents:
        raise ValueError(
            f"work_dir {resolved} must reside outside output_dir {out_resolved} so periodic "
            "checkpoint pruning never mutates immutable supervisor exports."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _publish_work_checkpoint_to_output(
    work_pt: Path,
    ckpt_out_dir: Path,
    dst_names: Sequence[str],
    *,
    reason: str,
    staging_dir: Path,
    fallback_counters: dict[str, int],
) -> tuple[str | None, list[str]]:
    """Hard-link `work_pt` into create-only `ckpt_out_dir` filenames and write `.meta.json`."""
    unique_names = list(dict.fromkeys(dst_names))
    targets = [ckpt_out_dir / name for name in unique_names if not (ckpt_out_dir / name).exists()]
    if not targets:
        first_existing = ckpt_out_dir / unique_names[0] if unique_names else None
        if first_existing is not None and first_existing.is_file():
            return sha256_file(first_existing), []
        return None, []

    work_meta_path = work_pt.with_suffix(".meta.json")
    counters = dict(fallback_counters)
    if work_meta_path.is_file():
        try:
            loaded_meta = json.loads(work_meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded_meta, dict):
                for k in ("updates_completed", "agent_steps_collected", "world_steps_collected"):
                    if k in loaded_meta:
                        counters[k] = int(loaded_meta[k])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    pt_sha = publish_hardlinked_artifacts(work_pt, targets, staging_dir=staging_dir)
    published_names: list[str] = []
    for dst in targets:
        meta_path = dst.with_suffix(".meta.json")
        if not meta_path.exists():
            meta = {
                "schema": ARENA_PPO_SCHEMA,
                "namespace": ARENA_PPO_NAMESPACE,
                "reason": reason,
                "checkpoint_file": dst.name,
                "checkpoint_sha256": pt_sha,
                "updates_completed": int(counters["updates_completed"]),
                "agent_steps_collected": int(counters["agent_steps_collected"]),
                "world_steps_collected": int(counters["world_steps_collected"]),
                "canary_export_probe": False,
            }
            write_create_only_json(meta_path, meta)
        published_names.append(dst.name)
    return pt_sha, published_names


def _format_update_log_record(metrics: PPOUpdateMetrics) -> dict[str, Any]:
    """Build lightweight per-update JSONL record for create-only training log."""
    return {
        "update_idx": int(metrics.update_idx),
        "agent_steps": int(metrics.agent_steps_collected),
        "steps_per_sec": round(float(metrics.steps_per_sec), 2),
        "mean_length": round(float(metrics.mean_length), 4),
        "p95_length": round(float(metrics.p95_length), 4),
        "deaths_per_1k_agent_steps": round(float(metrics.deaths_per_1k_agent_steps), 4),
        "mean_episode_return": round(float(metrics.mean_episode_return), 6),
        "entropy": round(float(metrics.entropy), 6),
        "approx_kl": round(float(metrics.approx_kl), 6),
        "epoch0_approx_kl": round(float(metrics.epoch0_approx_kl), 6),
        "value_explained_variance": round(float(metrics.value_explained_variance), 6),
        "clip_fraction": round(float(metrics.clip_fraction), 6),
        "aux_losses": {
            "total": round(float(metrics.aux_loss), 6),
            "h16": round(float(metrics.aux_loss_h16), 6),
            "h64": round(float(metrics.aux_loss_h64), 6),
            "h256": round(float(metrics.aux_loss_h256), 6),
        },
        "policy_loss": round(float(metrics.policy_loss), 6),
        "value_loss": round(float(metrics.value_loss), 6),
        "deaths_count": int(metrics.deaths_count),
        "collect_sec": round(float(metrics.collect_sec), 4),
        "learn_sec": round(float(metrics.learn_sec), 4),
    }


def run_training_job(
    *,
    cfg: ArenaPPOConfig,
    output_dir: Path,
    device: str | torch.device = "cuda",
    input_dir: Path | None = None,
    ledger_path: Path | None = None,
    max_wall_seconds: float | None = None,
    work_dir: Path | None = None,
    hours: float | None = None,
    expected_steps_per_sec: float | None = None,
    eval_hook: Callable[[ArenaPPOTrainer, int], dict[str, Any]] | None = None,
    step_hook: Callable[[ArenaPPOTrainer, int], None] | None = None,
) -> dict[str, Any]:
    """Run Stage 1 PPO training with signal-safe checkpointing, resume, and verified exports."""
    effective_hours = (
        float(hours)
        if hours is not None
        else (float(max_wall_seconds) / 3600.0 if max_wall_seconds is not None else None)
    )
    training_plan: dict[str, Any] | None = None
    if effective_hours is not None and expected_steps_per_sec is not None:
        training_plan = build_training_plan(
            cfg,
            hours=effective_hours,
            expected_steps_per_sec=float(expected_steps_per_sec),
            max_wall_seconds=max_wall_seconds,
        )
        print(f"ARENA_PPO_PLAN {json.dumps(training_plan, sort_keys=True)}", flush=True)
        phase_stamp("train_plan", "INFO", **training_plan)

    resolved_dev = normalize_device(device)
    phase_stamp(
        "train_setup",
        "START",
        device=str(resolved_dev),
        net_arch=cfg.net_arch,
        net_size=cfg.net_size,
    )
    ext_info = preload_native_sim_extensions(resolved_dev)
    trainer = ArenaPPOTrainer(cfg, device=resolved_dev)

    resolved_work_dir = _resolve_work_dir(output_dir, work_dir)
    ckpt_work_dir = resolved_work_dir / "checkpoints"
    ckpt_work_dir.mkdir(parents=True, exist_ok=True)

    ckpt_out_dir = output_dir / "checkpoints"
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)

    # Check for verified resume checkpoint in output_dir/work_dir or staged input_dir
    resumed_from: str | None = None
    local_candidates = [
        p
        for p in (
            find_newest_verified_checkpoint(ckpt_out_dir),
            find_newest_verified_checkpoint(ckpt_work_dir),
        )
        if p is not None
    ]
    newest_local: Path | None = None
    if local_candidates:
        newest_local = max(
            local_candidates,
            key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or "-1"),
        )
    if newest_local is not None:
        trainer.load_checkpoint(newest_local, verify_markers=True)
        resumed_from = str(newest_local)
    elif ledger_path is not None and ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        resume_entry = ledger.get("resume_checkpoint")
        if isinstance(resume_entry, dict) and input_dir is not None:
            staged_ckpt = Path(input_dir) / str(resume_entry["relpath"])
            if staged_ckpt.is_file():
                actual_sha = hashlib_sha(staged_ckpt)
                if actual_sha != resume_entry.get("sha256"):
                    raise ValueError(f"Staged resume checkpoint SHA mismatch at {staged_ckpt}")
                trainer.load_checkpoint(staged_ckpt, verify_markers=False)
                resumed_from = str(staged_ckpt)

    phase_stamp(
        "train_setup",
        "END",
        resumed_from=resumed_from,
        updates_completed=trainer.updates_completed,
        work_dir=str(resolved_work_dir),
        native_extensions=ext_info,
    )

    wall_start = time.monotonic()
    last_output_publish_wall = wall_start
    last_metrics: dict[str, Any] = {}
    last_update_sec: float = 0.0
    session_updates_completed: int = 0
    # Conservative initial estimate (~100 MiB/s) until first real checkpoint save is timed.
    est_ckpt_bytes = estimate_checkpoint_bytes(cfg)
    last_ckpt_save_sec: float = max(1.0, float(est_ckpt_bytes) / float(100 * 1024 * 1024))
    last_output_publish_sec: float = max(0.5, 0.25 * last_ckpt_save_sec)
    aborted_mid_update = False
    stopped_reason = "completed_target_updates"
    pruned_checkpoints: list[str] = []
    published_checkpoints: list[str] = []
    eval_records_written: list[str] = []
    newest_verified_work_ckpt: Path | None = find_newest_verified_checkpoint(ckpt_work_dir)

    log_writer = RotatingCreateOnlyJSONLWriter(
        output_dir / "train_logs",
        prefix="train_log",
        rotate_every_records=cfg.log_rotate_every_updates,
        rotate_every_seconds=cfg.log_rotate_every_seconds,
    )

    def _current_counters() -> dict[str, int]:
        return {
            "updates_completed": int(trainer.updates_completed),
            "agent_steps_collected": int(trainer.agent_steps_collected),
            "world_steps_collected": int(trainer.world_steps_collected),
        }

    try:
        with SignalStopFlag() as stop_flag:
            phase_stamp("train_loop", "START", target_updates=cfg.total_updates)
            while trainer.updates_completed < cfg.total_updates:
                if stop_flag.stopped:
                    stopped_reason = f"signal:{stop_flag.signal_name}"
                    break

                elapsed_wall = time.monotonic() - wall_start
                if max_wall_seconds is not None:
                    budget = float(max_wall_seconds)
                    export_reserve_sec = max(5.0, 2.5 * last_ckpt_save_sec)
                    if elapsed_wall >= budget:
                        stopped_reason = "wall_budget_guard"
                        break
                    if (
                        last_update_sec > 0.0
                        and (elapsed_wall + 1.5 * last_update_sec + export_reserve_sec) > budget
                    ):
                        stopped_reason = "wall_budget_projected_guard"
                        break

                try:
                    metrics = trainer.step_update(should_stop=lambda: stop_flag.stopped)
                except StopRequestedError:
                    aborted_mid_update = True
                    stopped_reason = f"signal:{stop_flag.signal_name or 'INTERRUPTED'}"
                    break

                session_updates_completed += 1
                last_metrics = asdict(metrics)
                last_update_sec = float(metrics.collect_sec + metrics.learn_sec)
                log_writer.append(_format_update_log_record(metrics))

                now_wall = time.monotonic()
                due_wall = (now_wall - trainer._last_checkpoint_wall) >= float(
                    cfg.checkpoint_interval_sec
                )
                due_updates = (
                    cfg.checkpoint_every_updates > 0
                    and trainer.updates_completed % cfg.checkpoint_every_updates == 0
                )
                # Exempt the first update of a session from `over_max_update` because it
                # includes one-time torch.compile / kernel autotuning / allocator warmup.
                over_max_update = (
                    session_updates_completed > 1
                    and cfg.max_update_seconds > 0.0
                    and last_update_sec > float(cfg.max_update_seconds)
                )
                elapsed_since_publish = now_wall - last_output_publish_wall
                projected_next_publish_gap = elapsed_since_publish + PUBLISH_GAP_SAFETY_FACTOR * (
                    max(0.0, last_update_sec)
                    + max(0.0, last_ckpt_save_sec)
                    + max(0.0, last_output_publish_sec)
                )
                due_output_publish = (
                    elapsed_since_publish >= float(cfg.output_publish_interval_sec)
                    or projected_next_publish_gap >= HARD_KILL_MAX_LOSS_SEC
                )
                if due_wall or due_updates or over_max_update or due_output_publish:
                    periodic_path = ckpt_work_dir / f"ckpt_u{trainer.updates_completed:07d}.pt"
                    if not periodic_path.exists():
                        phase_stamp("checkpoint", "START", update=trainer.updates_completed)
                        t_save = time.monotonic()
                        trainer.save_checkpoint(periodic_path, reason="periodic")
                        last_ckpt_save_sec = time.monotonic() - t_save
                        verify_completed_artifact(periodic_path)
                        newest_verified_work_ckpt = periodic_path
                        pruned_checkpoints.extend(
                            prune_old_periodic_checkpoints(
                                ckpt_work_dir, keep=cfg.keep_periodic_checkpoints
                            )
                        )
                        phase_stamp(
                            "checkpoint",
                            "END",
                            update=trainer.updates_completed,
                            save_sec=round(last_ckpt_save_sec, 4),
                        )
                    else:
                        newest_verified_work_ckpt = periodic_path

                    if due_output_publish and newest_verified_work_ckpt is not None:
                        t_pub = time.monotonic()
                        _, pub_names = _publish_work_checkpoint_to_output(
                            newest_verified_work_ckpt,
                            ckpt_out_dir,
                            [newest_verified_work_ckpt.name],
                            reason="periodic_output_snapshot",
                            staging_dir=ckpt_work_dir,
                            fallback_counters=_current_counters(),
                        )
                        last_output_publish_wall = time.monotonic()
                        last_output_publish_sec = last_output_publish_wall - t_pub
                        published_checkpoints.extend(pub_names)

                if (
                    cfg.eval_every_updates > 0
                    and eval_hook is not None
                    and trainer.updates_completed % cfg.eval_every_updates == 0
                ):
                    eval_payload = eval_hook(trainer, trainer.updates_completed)
                    if isinstance(eval_payload, dict) and eval_payload:
                        eval_path = (
                            output_dir / "eval" / f"eval_u{trainer.updates_completed:07d}.json"
                        )
                        if not eval_path.exists():
                            write_create_only_json(eval_path, eval_payload)
                            eval_records_written.append(eval_path.name)

                if over_max_update:
                    stopped_reason = "max_update_seconds_guard"
                    phase_stamp(
                        "max_update_seconds_guard",
                        "TRIGGERED",
                        update=trainer.updates_completed,
                        last_update_sec=round(last_update_sec, 4),
                        max_update_seconds=float(cfg.max_update_seconds),
                    )
                    break

                if step_hook is not None:
                    step_hook(trainer, trainer.updates_completed)

            phase_stamp(
                "train_loop",
                "END",
                updates_completed=trainer.updates_completed,
                stopped_reason=stopped_reason,
                aborted_mid_update=aborted_mid_update,
            )

        phase_stamp("train_export", "START", aborted_mid_update=aborted_mid_update)
        final_ckpt_sha: str | None = None
        weights_sha: str | None = None
        is_signal_stop = stopped_reason.startswith("signal:")
        is_guard_stop = stopped_reason in {
            "max_update_seconds_guard",
            "wall_budget_guard",
            "wall_budget_projected_guard",
        }
        write_final_exports = (not aborted_mid_update) and (
            (not is_signal_stop) or (last_ckpt_save_sec <= SUPERVISOR_GRACE_SAVE_LIMIT_SEC)
        )

        if write_final_exports:
            existing_u = ckpt_work_dir / f"ckpt_u{trainer.updates_completed:07d}.pt"
            if existing_u.is_file() and Path(f"{existing_u}.complete").is_file():
                verify_completed_artifact(existing_u)
                work_final_src = existing_u
            else:
                work_final_src = ckpt_work_dir / f"ckpt_final_u{trainer.updates_completed:07d}.pt"
                if not work_final_src.exists():
                    t_save = time.monotonic()
                    trainer.save_checkpoint(work_final_src, reason=stopped_reason)
                    last_ckpt_save_sec = time.monotonic() - t_save
                    verify_completed_artifact(work_final_src)

            dst_names: list[str] = []
            if is_signal_stop:
                dst_names.append(f"ckpt_sig_u{trainer.updates_completed:07d}.pt")
            if (
                is_guard_stop
                and newest_verified_work_ckpt is not None
                and newest_verified_work_ckpt.resolve() == work_final_src.resolve()
            ):
                dst_names.append(newest_verified_work_ckpt.name)
            dst_names.append("final.pt")

            final_ckpt_sha, pub_names = _publish_work_checkpoint_to_output(
                work_final_src,
                ckpt_out_dir,
                dst_names,
                reason=stopped_reason,
                staging_dir=ckpt_work_dir,
                fallback_counters=_current_counters(),
            )
            published_checkpoints.extend(pub_names)

            if (
                is_guard_stop
                and newest_verified_work_ckpt is not None
                and newest_verified_work_ckpt.resolve() != work_final_src.resolve()
            ):
                _, extra_pub = _publish_work_checkpoint_to_output(
                    newest_verified_work_ckpt,
                    ckpt_out_dir,
                    [newest_verified_work_ckpt.name],
                    reason=stopped_reason,
                    staging_dir=ckpt_work_dir,
                    fallback_counters=_current_counters(),
                )
                published_checkpoints.extend(extra_pub)

            weights_path = output_dir / "arena_ppo_weights.pth"
            if not weights_path.exists():
                weights_sha = trainer.save_eval_weights(weights_path)
        elif newest_verified_work_ckpt is not None:
            # Mid-update signal abort or oversized grace save: publish newest verified periodic ckpt
            _, abort_pub = _publish_work_checkpoint_to_output(
                newest_verified_work_ckpt,
                ckpt_out_dir,
                [newest_verified_work_ckpt.name],
                reason=stopped_reason,
                staging_dir=ckpt_work_dir,
                fallback_counters=_current_counters(),
            )
            published_checkpoints.extend(abort_pub)
    finally:
        sealed_logs = [p.name for p in log_writer.close()]

    result = {
        "schema": "arena-ppo-train-result/v1",
        "namespace": ARENA_PPO_NAMESPACE,
        "canary_export_probe": False,
        "device": str(resolved_dev),
        "config": cfg.to_dict(),
        "training_plan": training_plan,
        "parameter_count": trainer.model.parameter_count(),
        "resumed_from": resumed_from,
        "stopped_reason": stopped_reason,
        "aborted_mid_update": bool(aborted_mid_update),
        "updates_completed": trainer.updates_completed,
        "agent_steps_collected": trainer.agent_steps_collected,
        "world_steps_collected": trainer.world_steps_collected,
        "total_time_limit_resets": trainer.time_limit_resets,
        "wall_seconds": round(time.monotonic() - wall_start, 4),
        "last_update_seconds": round(last_update_sec, 4),
        "last_checkpoint_save_seconds": round(last_ckpt_save_sec, 4),
        "last_output_publish_seconds": round(last_output_publish_sec, 4),
        "work_dir": str(resolved_work_dir),
        "pruned_periodic_checkpoints": pruned_checkpoints,
        "published_checkpoints": published_checkpoints,
        "train_log_segments": sealed_logs,
        "eval_records": eval_records_written,
        "late_reset_summary": (
            {
                "enabled": True,
                "reservoir_size": trainer.late_reservoir.size,
                "total_harvested": trainer.late_reservoir.total_harvested,
                "total_restored": trainer.late_reservoir.total_restored,
                "total_time_limit_resets": trainer.time_limit_resets,
            }
            if trainer.late_reservoir is not None
            else {
                "enabled": False,
                "total_time_limit_resets": trainer.time_limit_resets,
            }
        ),
        "last_update_metrics": last_metrics,
        "final_checkpoint_sha256": final_ckpt_sha,
        "eval_weights_sha256": weights_sha,
    }
    write_create_only_json(output_dir / "train_result.json", result)
    phase_stamp("train_export", "END", updates_completed=trainer.updates_completed)
    return result


def hashlib_sha(path: Path) -> str:
    """Return SHA-256 hex digest of `path`."""
    return sha256_file(path)


def main() -> int:
    """CLI entrypoint for `python -m research.arena_ppo.ppo`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=os.environ.get("SNAKE_DQN_DEVICE", "cuda"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument(
        "--net-arch",
        type=str,
        choices=list(VALID_NET_ARCHS),
        default=os.environ.get("ARENA_PPO_NET_ARCH") or None,
        help="Policy network architecture (`shared_trunk`, `ego_only`, or `per_seat`).",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=(
            float(os.environ["ARENA_PPO_HOURS"]) if os.environ.get("ARENA_PPO_HOURS") else None
        ),
    )
    parser.add_argument(
        "--expected-steps-per-sec",
        type=float,
        default=(
            float(os.environ["ARENA_PPO_EXPECTED_STEPS_PER_SEC"])
            if os.environ.get("ARENA_PPO_EXPECTED_STEPS_PER_SEC")
            else None
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=(
            float(os.environ["ARENA_PPO_MAX_WALL_SECONDS"])
            if os.environ.get("ARENA_PPO_MAX_WALL_SECONDS")
            else None
        ),
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help=(
            "Enable opt-in CUDA mixed precision "
            "(bf16 on Ampere/Ada/Hopper, fp16+GradScaler else)."
        ),
    )
    amp_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable CUDA mixed precision (fp32 everywhere).",
    )
    args = parser.parse_args()

    raw_cfg_dict: dict[str, Any] = {}
    if args.config_json is not None:
        source = args.config_json
        if source.lstrip().startswith("{"):
            raw_cfg_dict = json.loads(source)
        else:
            raw_cfg_dict = json.loads(Path(source).read_text(encoding="utf-8"))
    env_amp = os.environ.get("ARENA_PPO_AMP")
    if args.amp is not None:
        raw_cfg_dict["amp"] = bool(args.amp)
    elif env_amp in {"0", "1", "false", "true", "False", "True"} and "amp" not in raw_cfg_dict:
        raw_cfg_dict["amp"] = env_amp.lower() in {"1", "true"}
    if args.net_arch is not None:
        prev_arch = raw_cfg_dict.get("net_arch")
        raw_cfg_dict["net_arch"] = args.net_arch
        if prev_arch is not None and prev_arch != args.net_arch:
            # Reset architecture-coupled batch defaults unless explicitly matching
            ns: NetSize = raw_cfg_dict.get("net_size", "S")
            if raw_cfg_dict.get("minibatch_size") == default_minibatch_size(ns, net_arch=prev_arch):
                raw_cfg_dict.pop("minibatch_size", None)
            if raw_cfg_dict.get("actor_chunk_envs") == default_actor_chunk_envs(
                ns, net_arch=prev_arch
            ):
                raw_cfg_dict.pop("actor_chunk_envs", None)
    cfg = ArenaPPOConfig.from_dict(raw_cfg_dict) if raw_cfg_dict else ArenaPPOConfig()

    max_wall_sec = args.max_wall_seconds
    if max_wall_sec is None and args.hours is not None:
        max_wall_sec = float(args.hours) * 3600.0

    # When `--hours` drives the run and `total_updates` was not explicitly pinned, make the
    # update cap effectively unlimited so `max_wall_seconds` governs multi-session continuations
    # and fast GPUs cleanly without premature `completed_target_updates` stops.
    if args.hours is not None and "total_updates" not in raw_cfg_dict:
        cfg = replace(
            cfg,
            total_updates=max(int(cfg.total_updates), UNLIMITED_HOURS_TOTAL_UPDATES),
        )
    if args.expected_steps_per_sec is not None and "max_update_seconds" not in raw_cfg_dict:
        cfg = replace(
            cfg,
            max_update_seconds=compute_max_update_seconds(cfg, float(args.expected_steps_per_sec)),
        )

    run_training_job(
        cfg=cfg,
        output_dir=args.output_dir.resolve(),
        work_dir=args.work_dir.resolve() if args.work_dir is not None else None,
        device=args.device,
        input_dir=args.input_dir.resolve() if args.input_dir is not None else None,
        ledger_path=args.ledger.resolve() if args.ledger is not None else None,
        max_wall_seconds=max_wall_sec,
        hours=args.hours,
        expected_steps_per_sec=args.expected_steps_per_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
