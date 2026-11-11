"""GPU-resident clipped PPO self-play learner with compact rollout tape and verified resume."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.arena_ppo.net import (
    AUX_DEATH_HORIZONS,
    ArenaNet,
    ArenaNetConfig,
    NetSize,
    symlog_two_hot_loss,
)
from research.arena_ppo.obs import build_arena_ego_obs_from_tensors, build_arena_ego_obs_gpu
from research.arena_ppo.runtime import (
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    preload_native_sim_extensions,
    synchronize_device,
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.arena_ppo.sim import (
    OPPONENT_KIND_SCRIPTED,
    OPPONENT_KIND_SNAPSHOT,
    TrainSim,
)

ARENA_PPO_SCHEMA = "arena-ppo-stage1/v1"
ARENA_PPO_NAMESPACE = "arena_ppo_stage1_scratch_v2_20261110"


@dataclass(frozen=True)
class ArenaPPOConfig:
    """Complete JSON-serializable hyperparameter configuration for Stage 1 Arena PPO."""

    net_size: NetSize = "S"
    num_envs: int = 1024
    rollout_len: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.97
    lr: float = 3.0e-4
    adam_eps: float = 1.0e-5
    weight_decay: float = 0.0
    clip_ratio: float = 0.2
    ppo_epochs: int = 2
    minibatch_size: int = 1536
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    aux_coef: float = 0.1
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    actor_chunk_envs: int = 1024
    ring_chunk: int = 512
    max_capacity: int = 1600
    max_world_frames: int = 0
    total_updates: int = 100
    seed: int = 20261110
    deterministic: bool = False
    compile_model: bool = False
    use_channels_last: bool = True
    use_bf16_autocast: bool = True
    checkpoint_interval_sec: float = 1200.0
    checkpoint_every_updates: int = 0

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
        return cls(**raw)

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

        self.food_xy = torch.zeros(
            (self.T, self.E, self.food_capacity, 2), dtype=torch.uint8, device=device
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
        self.food_xy[t, ..., 0] = fx.clamp(0, 255).to(torch.uint8)
        self.food_xy[t, ..., 1] = fy.clamp(0, 255).to(torch.uint8)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct bit-identical `(arena, ego, scalars, legal_mask)` for `(t_idx, e_idx)` pairs.

        Args:
            t_idx: 1D long tensor `(M,)` of rollout step indices in `0 .. T - 1`.
            e_idx: 1D long tensor `(M,)` of world indices in `0 .. E - 1`.
            ring_chunk: Segment chunk width.

        Returns:
            Tuple `(arena, ego, scalars, legal_mask)` over all `M * S` seats.
        """
        m_worlds = int(t_idx.shape[0])
        head_pos = self.tape_head_idx[t_idx, e_idx].to(torch.long)
        k_offsets = torch.arange(self.cap, dtype=torch.long, device=self.device)[None, None, :]
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
        )
        normal = torch.ones((m_worlds, self.S, 3), dtype=torch.bool, device=self.device)
        can_boost = alive_sub & (len_sub >= self.min_boost_length)
        legal_mask = torch.cat((normal, can_boost[..., None].expand(-1, -1, 3)), dim=2).reshape(
            m_worlds * self.S, 6
        )
        return obs.arena, obs.ego, obs.scalars, legal_mask

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
    clip_fraction: float
    grad_norm: float
    collect_sec: float
    learn_sec: float


def find_newest_verified_checkpoint(ckpt_dir: Path | str) -> Path | None:
    """Return the newest verified non-probe checkpoint in `ckpt_dir`, or None.

    Verifies `.complete` + `.sha256` digests and refuses `canary_export_probe` stubs.

    Args:
        ckpt_dir: Directory containing `ckpt_u*.pt` checkpoints.

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
            if bool(meta.get("canary_export_probe")):
                continue
            updates = int(meta.get("updates_completed", -1))
        else:
            stem_digits = "".join(ch for ch in pt_path.stem if ch.isdigit())
            updates = int(stem_digits) if stem_digits else -1
        candidates.append((updates, float(pt_path.stat().st_mtime), pt_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


class ArenaPPOTrainer:
    """Self-contained GPU-resident PPO trainer with atomic checkpointing and resume."""

    def __init__(
        self,
        cfg: ArenaPPOConfig,
        *,
        device: str | torch.device = "cpu",
        train_sim: TrainSim | None = None,
        snapshot_model: ArenaNet | None = None,
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

        net_cfg = ArenaNetConfig.from_size(
            cfg.net_size,
            use_channels_last=cfg.use_channels_last,
            use_bf16_autocast=cfg.use_bf16_autocast,
        )
        self.model = ArenaNet(net_cfg, compile_model=cfg.compile_model).to(self.device)
        self.snapshot_model: ArenaNet | None = (
            snapshot_model.to(self.device).eval() if snapshot_model is not None else None
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )

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
        self.checkpoints_written: int = 0
        self._last_checkpoint_wall: float = time.monotonic()

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
        policy: ArenaNet,
        legal_mask: torch.Tensor,
        *,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run observation build + policy `act` across `E` worlds in `actor_chunk_envs` slices."""
        e_total, s_total = self.sim.E, self.sim.S
        chunk = max(1, min(e_total, int(self.cfg.actor_chunk_envs)))
        if chunk == e_total:
            obs = build_arena_ego_obs_gpu(self.sim, ring_chunk=self.cfg.ring_chunk)
            act_out = policy.act(
                obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=greedy
            )
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
            obs = build_arena_ego_obs_gpu(self.sim, env_slice=sl, ring_chunk=self.cfg.ring_chunk)
            act_out = policy.act(
                obs.arena,
                obs.ego,
                obs.scalars,
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
        if chunk == e_total:
            obs = build_arena_ego_obs_gpu(self.sim, ring_chunk=self.cfg.ring_chunk)
            out = self.model.forward(obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask)
            return out.value.reshape(e_total, s_total)

        values = torch.empty((e_total, s_total), dtype=torch.float32, device=self.device)
        for start in range(0, e_total, chunk):
            end = min(e_total, start + chunk)
            sl = slice(start, end)
            obs = build_arena_ego_obs_gpu(self.sim, env_slice=sl, ring_chunk=self.cfg.ring_chunk)
            out = self.model.forward(obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask[sl])
            values[sl] = out.value.reshape(end - start, s_total)
        return values

    def _maybe_apply_world_time_limits(self, t: int, survived: torch.Tensor) -> None:
        """Handle optional `max_world_frames` world resets with pre-reset value bootstrap."""
        max_frames = int(self.cfg.max_world_frames)
        if max_frames <= 0:
            self.buffer.truncated[t].zero_()
            self.buffer.trunc_bootstrap[t].zero_()
            return

        hit = self.sim.sim.frame >= max_frames
        if not bool(hit.any().detach().cpu().item()):
            self.buffer.truncated[t].zero_()
            self.buffer.trunc_bootstrap[t].zero_()
            return

        # Compute pre-reset bootstrap values before resetting hit worlds
        with torch.no_grad():
            pre_reset_vals = self._bootstrap_values_chunked()
        trunc_mask = hit[:, None] & survived
        self.buffer.truncated[t] = trunc_mask
        self.buffer.trunc_bootstrap[t] = torch.where(
            trunc_mask, pre_reset_vals, torch.zeros_like(pre_reset_vals)
        )

        # Reset hit worlds on device
        sim = self.sim.sim
        sim.alive[hit] = False
        sim.respawn_timer[hit] = 0
        sim.seg_count[hit] = 0
        sim.length[hit] = 1
        sim.boost_frames[hit] = 0
        sim.frames_since_food[hit] = 0
        sim._reward_prev_length[hit] = 1
        sim._boosted_this_step[hit] = False
        sim.food_present[hit] = False
        sim.corpse_food[hit] = False
        sim.food_count[hit] = 0
        sim.frame[hit] = 0
        self.sim.prev_alive[hit] = False
        self.sim.prev_heads[hit] = 0
        sim._respawn_exact()
        sim._refill_food_rows(hit)

    @torch.no_grad()
    def collect_rollout(self) -> float:
        """Collect `T` ticks across `E * 6` seats into `self.buffer`.

        Returns:
            Elapsed wall seconds spent in rollout collection + GAE computation.
        """
        self.model.eval()
        t0 = time.perf_counter()
        for t in range(self.cfg.rollout_len):
            self.sim.prepare()
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

        # Rollout-cut bootstrap on state `T`
        self.sim.prepare()
        last_values = self._bootstrap_values_chunked()
        self.buffer.compute_gae_and_aux(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        synchronize_device(self.device)
        elapsed = time.perf_counter() - t0
        self.world_steps_collected += self.cfg.rollout_len * self.sim.E
        self.agent_steps_collected += self.cfg.rollout_len * self.sim.E * self.sim.S
        return elapsed

    def ppo_update(self) -> tuple[dict[str, float], float]:
        """Run `ppo_epochs` of minibatch clipped PPO + symlog two-hot value + aux BCE updates."""
        self.model.train()
        t0 = time.perf_counter()

        valid_3d = self.buffer.acted & self.buffer.learner_mask
        adv_all = self.buffer.advantages
        if self.cfg.normalize_advantages:
            valid_f = valid_3d.to(torch.float32)
            denom = valid_f.sum().clamp_min(1.0)
            mean_adv = (adv_all * valid_f).sum() / denom
            var_adv = (((adv_all - mean_adv) * valid_f).square().sum() / denom).clamp_min(1e-8)
            norm_adv = (adv_all - mean_adv) * torch.rsqrt(var_adv)
        else:
            norm_adv = adv_all

        total_world_ticks = self.cfg.rollout_len * self.sim.E
        mb_worlds = max(1, min(total_world_ticks, int(self.cfg.minibatch_size) // self.sim.S))

        policy_loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        value_loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        aux_loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        entropy_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        kl_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        clip_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        grad_norm_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        num_batches = 0

        for _ in range(self.cfg.ppo_epochs):
            perm_cpu = torch.randperm(total_world_ticks, generator=self.perm_gen)
            perm = perm_cpu.to(device=self.device, dtype=torch.long)
            for start in range(0, total_world_ticks, mb_worlds):
                flat_wt = perm[start : start + mb_worlds]
                t_idx = torch.div(flat_wt, self.sim.E, rounding_mode="floor")
                e_idx = flat_wt.remainder(self.sim.E)

                arena, ego, scalars, legal_mask = self.buffer.reconstruct_world_ticks(
                    t_idx, e_idx, ring_chunk=self.cfg.ring_chunk
                )
                mb_actions = self.buffer.actions[t_idx, e_idx].reshape(-1).to(torch.long)
                mb_old_logp = self.buffer.log_probs[t_idx, e_idx].reshape(-1)
                mb_adv = norm_adv[t_idx, e_idx].reshape(-1)
                mb_returns = self.buffer.returns[t_idx, e_idx].reshape(-1)
                mb_valid = valid_3d[t_idx, e_idx].reshape(-1)
                mb_aux_targets = self.buffer.aux_targets[t_idx, e_idx].reshape(
                    -1, len(AUX_DEATH_HORIZONS)
                )
                mb_aux_valid = self.buffer.aux_valid[t_idx, e_idx].reshape(
                    -1, len(AUX_DEATH_HORIZONS)
                )

                new_logp, entropy, val_logits, aux_logits = self.model.evaluate_actions(
                    arena, ego, scalars, legal_mask, mb_actions
                )

                valid_f = mb_valid.to(torch.float32)
                valid_count = valid_f.sum().clamp_min(1.0)

                log_ratio = (new_logp - mb_old_logp).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                    * mb_adv
                )
                pg_loss = -(torch.minimum(surr1, surr2) * valid_f).sum() / valid_count

                v_loss_rows = symlog_two_hot_loss(val_logits, mb_returns, self.model.value_support)
                v_loss = (v_loss_rows * valid_f).sum() / valid_count

                ent_mean = (entropy * valid_f).sum() / valid_count

                aux_mask_f = (mb_aux_valid & mb_valid[:, None]).to(torch.float32)
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
                    policy_loss_acc += pg_loss.detach()
                    value_loss_acc += v_loss.detach()
                    aux_loss_acc += aux_loss.detach()
                    entropy_acc += ent_mean.detach()
                    kl_acc += approx_kl.detach()
                    clip_acc += clipped.detach()
                    grad_norm_acc += torch.as_tensor(
                        grad_norm, dtype=torch.float32, device=self.device
                    )
                    num_batches += 1

        synchronize_device(self.device)
        elapsed = time.perf_counter() - t0
        self.updates_completed += 1
        inv_b = 1.0 / float(max(1, num_batches))
        stats_gpu = (
            torch.stack(
                (
                    policy_loss_acc * inv_b,
                    value_loss_acc * inv_b,
                    aux_loss_acc * inv_b,
                    entropy_acc * inv_b,
                    kl_acc * inv_b,
                    clip_acc * inv_b,
                    grad_norm_acc * inv_b,
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
                "clip_fraction": float(stats_gpu[5]),
                "grad_norm": float(stats_gpu[6]),
            },
            elapsed,
        )

    def step_update(self) -> PPOUpdateMetrics:
        """Execute one full `collect_rollout()` + `ppo_update()` iteration."""
        collect_sec = self.collect_rollout()
        stats, learn_sec = self.ppo_update()

        with torch.no_grad():
            valid_mask = self.buffer.acted & self.buffer.learner_mask
            learner_rows = int(valid_mask.sum().detach().cpu().item())
            mean_reward = float(
                self.buffer.rewards[valid_mask].mean().detach().cpu().item()
                if learner_rows > 0
                else 0.0
            )
            deaths_count = int(self.buffer.died.sum().detach().cpu().item())

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
            clip_fraction=stats["clip_fraction"],
            grad_norm=stats["grad_norm"],
            collect_sec=collect_sec,
            learn_sec=learn_sec,
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
            "sim_state_dict": self.sim.state_dict(),
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
        payload = {
            "schema": "arena-ppo-eval-weights/v1",
            "obs_spec": "arena_ego_v1",
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

        self.model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("snapshot_state_dict") is not None and self.snapshot_model is not None:
            self.snapshot_model.load_state_dict(ckpt["snapshot_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.sim.load_state_dict(ckpt["sim_state_dict"])
        self.refresh_opponent_flags()

        counters = ckpt["counters"]
        self.updates_completed = int(counters["updates_completed"])
        self.agent_steps_collected = int(counters["agent_steps_collected"])
        self.world_steps_collected = int(counters["world_steps_collected"])
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


def run_training_job(
    *,
    cfg: ArenaPPOConfig,
    output_dir: Path,
    device: str | torch.device = "cuda",
    input_dir: Path | None = None,
    ledger_path: Path | None = None,
    max_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run Stage 1 PPO training with signal-safe checkpointing, resume, and verified exports."""
    resolved_dev = normalize_device(device)
    phase_stamp("train_setup", "START", device=str(resolved_dev), net_size=cfg.net_size)
    ext_info = preload_native_sim_extensions(resolved_dev)
    trainer = ArenaPPOTrainer(cfg, device=resolved_dev)

    ckpt_out_dir = output_dir / "checkpoints"
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)

    # Check for verified resume checkpoint in output_dir/checkpoints or staged input_dir
    resumed_from: str | None = None
    newest_local = find_newest_verified_checkpoint(ckpt_out_dir)
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
        native_extensions=ext_info,
    )

    stop_signal: list[str] = []
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigusr1 = getattr(signal, "SIGUSR1", None)
    old_usr1 = signal.getsignal(prev_sigusr1) if prev_sigusr1 is not None else None

    def _on_signal(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        stop_signal.append(sig_name)

    signal.signal(signal.SIGTERM, _on_signal)
    if prev_sigusr1 is not None:
        signal.signal(prev_sigusr1, _on_signal)

    wall_start = time.monotonic()
    last_metrics: dict[str, Any] = {}
    stopped_reason = "completed_target_updates"
    try:
        phase_stamp("train_loop", "START", target_updates=cfg.total_updates)
        while trainer.updates_completed < cfg.total_updates:
            if stop_signal:
                stopped_reason = f"signal:{stop_signal[-1]}"
                sig_ckpt = ckpt_out_dir / f"ckpt_sig_u{trainer.updates_completed:07d}.pt"
                if not sig_ckpt.exists():
                    trainer.save_checkpoint(sig_ckpt, reason=stopped_reason)
                break

            elapsed_wall = time.monotonic() - wall_start
            if max_wall_seconds is not None and elapsed_wall >= float(max_wall_seconds):
                stopped_reason = "wall_budget_guard"
                break

            metrics = trainer.step_update()
            last_metrics = asdict(metrics)

            now_wall = time.monotonic()
            due_wall = (now_wall - trainer._last_checkpoint_wall) >= float(
                cfg.checkpoint_interval_sec
            )
            due_updates = (
                cfg.checkpoint_every_updates > 0
                and trainer.updates_completed % cfg.checkpoint_every_updates == 0
            )
            if due_wall or due_updates:
                periodic_path = ckpt_out_dir / f"ckpt_u{trainer.updates_completed:07d}.pt"
                if not periodic_path.exists():
                    phase_stamp("checkpoint", "START", update=trainer.updates_completed)
                    trainer.save_checkpoint(periodic_path, reason="periodic")
                    phase_stamp("checkpoint", "END", update=trainer.updates_completed)

        phase_stamp(
            "train_loop",
            "END",
            updates_completed=trainer.updates_completed,
            stopped_reason=stopped_reason,
        )
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        if prev_sigusr1 is not None and old_usr1 is not None:
            signal.signal(prev_sigusr1, old_usr1)

    phase_stamp("train_export", "START")
    final_ckpt_path = ckpt_out_dir / "final.pt"
    final_ckpt_sha, _ = trainer.save_checkpoint(final_ckpt_path, reason=stopped_reason)
    weights_path = output_dir / "arena_ppo_weights.pth"
    weights_sha = trainer.save_eval_weights(weights_path)

    result = {
        "schema": "arena-ppo-train-result/v1",
        "namespace": ARENA_PPO_NAMESPACE,
        "canary_export_probe": False,
        "device": str(resolved_dev),
        "config": cfg.to_dict(),
        "parameter_count": trainer.model.parameter_count(),
        "resumed_from": resumed_from,
        "stopped_reason": stopped_reason,
        "updates_completed": trainer.updates_completed,
        "agent_steps_collected": trainer.agent_steps_collected,
        "world_steps_collected": trainer.world_steps_collected,
        "wall_seconds": round(time.monotonic() - wall_start, 4),
        "last_update_metrics": last_metrics,
        "final_checkpoint_sha256": final_ckpt_sha,
        "eval_weights_sha256": weights_sha,
    }
    write_create_only_json(output_dir / "train_result.json", result)
    phase_stamp("train_export", "END", updates_completed=trainer.updates_completed)
    return result


def hashlib_sha(path: Path) -> str:
    """Return SHA-256 hex digest of `path`."""
    from research.arena_ppo.runtime import sha256_file

    return sha256_file(path)


def main() -> int:
    """CLI entrypoint for `python -m research.arena_ppo.ppo`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=os.environ.get("SNAKE_DQN_DEVICE", "cuda"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=(
            float(os.environ["ARENA_PPO_MAX_WALL_SECONDS"])
            if os.environ.get("ARENA_PPO_MAX_WALL_SECONDS")
            else None
        ),
    )
    args = parser.parse_args()

    cfg = (
        ArenaPPOConfig.from_json(args.config_json)
        if args.config_json is not None
        else ArenaPPOConfig()
    )
    run_training_job(
        cfg=cfg,
        output_dir=args.output_dir.resolve(),
        device=args.device,
        input_dir=args.input_dir.resolve() if args.input_dir is not None else None,
        ledger_path=args.ledger.resolve() if args.ledger is not None else None,
        max_wall_seconds=args.max_wall_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
