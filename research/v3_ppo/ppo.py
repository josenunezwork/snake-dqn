"""Game V3.1 PPO self-play trainer, atomic checkpoint/resume, and training job runner."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from research.arena_ppo.runtime import (
    RotatingCreateOnlyJSONLWriter,
    SignalStopFlag,
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    publish_hardlinked_artifacts,
    remove_completed_artifact,
    sha256_file,
    synchronize_device,
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.game_v3.config import GameV3Config
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs
from research.game_v3.sim import GameV3Sim
from research.v3_ppo.buffer import Uint8RolloutBuffer
from research.v3_ppo.config import (
    V3_PPO_NAMESPACE,
    V3PPOConfig,
    estimate_training_memory,
)
from research.v3_ppo.eval import (
    compute_step_reward,
    derive_world_seeds,
    evaluate_policy,
    scripted_opponent_actions,
)
from research.v3_ppo.net import (
    EgoMultiScaleNet,
    EgoMultiScaleNetConfig,
    symlog_two_hot_loss,
)

V3_PPO_CHECKPOINT_SCHEMA = "v3-ppo-checkpoint/v1"
V3_PPO_TRAIN_RESULT_SCHEMA = "v3-ppo-train-result/v1"
UNLIMITED_HOURS_TOTAL_UPDATES = 1_000_000_000

_RESUME_MUTABLE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "total_updates",
        "checkpoint_interval_sec",
        "output_publish_interval_sec",
        "checkpoint_every_updates",
        "keep_periodic_checkpoints",
        "max_update_seconds",
        "eval_every_updates",
        "eval_envs",
        "eval_ticks",
        "eval_learner_slots",
        "eval_seed",
        "eval_greedy",
        "actor_chunk_learners",
        "compile_model",
    }
)


class StopRequestedError(RuntimeError):
    """Raised internally to unwind mid-rollout or mid-epoch cleanly on cooperative stop."""


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and Torch CUDA RNG streams deterministically."""
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_cpu_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_tree(v) for v in obj)
    return obj


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)


_SIM_ZLIB_TOTAL_THRESHOLD_BYTES = 64 * 1024 * 1024
_SIM_ZLIB_TENSOR_MIN_BYTES = 4 * 1024 * 1024
_SIM_ZLIB_CHUNK_BYTES = 128 * 1024 * 1024


def pack_sim_state_dict(
    sim_sd: dict[str, Any],
    *,
    total_threshold_bytes: int = _SIM_ZLIB_TOTAL_THRESHOLD_BYTES,
    tensor_min_bytes: int = _SIM_ZLIB_TENSOR_MIN_BYTES,
) -> dict[str, Any]:
    """Losslessly compress large simulator grid tensors when total sim state exceeds threshold.

    Small simulator states remain untouched raw tensors; large production states
    (`E=128..512, R=600`) compress sparse occupancy/food grids via ``zlib`` level 1 in 128 MiB
    slices so checkpoint artifacts stay well below the 2 GiB export cap with exact bit recovery.
    """
    tensors = sim_sd.get("tensors")
    if not isinstance(tensors, dict):
        return sim_sd
    total_bytes = sum(
        t.numel() * t.element_size() for t in tensors.values() if isinstance(t, torch.Tensor)
    )
    if total_bytes <= int(total_threshold_bytes):
        return sim_sd

    raw_tensors: dict[str, torch.Tensor] = {}
    zlib_tensors: dict[str, dict[str, Any]] = {}
    for name, val in tensors.items():
        if not isinstance(val, torch.Tensor):
            continue
        nbytes = val.numel() * val.element_size()
        if nbytes == 0 or nbytes < int(tensor_min_bytes):
            raw_tensors[name] = val
            continue
        cpu_contig = val.detach().cpu().contiguous()
        byte_tensor = cpu_contig.view(torch.uint8).reshape(-1)
        mv = memoryview(byte_tensor.numpy())
        chunks: list[bytes] = []
        for off in range(0, len(mv), _SIM_ZLIB_CHUNK_BYTES):
            chunks.append(zlib.compress(mv[off : off + _SIM_ZLIB_CHUNK_BYTES], level=1))
        zlib_tensors[name] = {
            "shape": list(cpu_contig.shape),
            "dtype": str(cpu_contig.dtype).removeprefix("torch."),
            "nbytes": int(nbytes),
            "chunks": chunks,
        }
    if not zlib_tensors:
        return sim_sd
    out = dict(sim_sd)
    out["tensors"] = raw_tensors
    out["_zlib_tensors"] = zlib_tensors
    return out


def unpack_sim_state_dict(sim_sd: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct exact ``GameV3Sim.state_dict()`` tensors from :func:`pack_sim_state_dict`."""
    zlib_tensors = sim_sd.get("_zlib_tensors")
    if not isinstance(zlib_tensors, dict) or not zlib_tensors:
        return sim_sd
    merged_tensors = dict(sim_sd.get("tensors") or {})
    for name, meta in zlib_tensors.items():
        dtype = getattr(torch, str(meta["dtype"]))
        shape = tuple(int(d) for d in meta["shape"])
        nbytes = int(meta["nbytes"])
        dst_u8 = torch.empty((nbytes,), dtype=torch.uint8)
        dst_mv = memoryview(dst_u8.numpy())
        offset = 0
        for comp_chunk in meta["chunks"]:
            raw_chunk = zlib.decompress(comp_chunk)
            end = offset + len(raw_chunk)
            dst_mv[offset:end] = raw_chunk
            offset = end
        if offset != nbytes:
            raise ValueError(
                f"Corrupt compressed sim tensor {name!r}: expected {nbytes} bytes, got {offset}"
            )
        merged_tensors[name] = dst_u8.view(dtype).reshape(shape)
    out = dict(sim_sd)
    out.pop("_zlib_tensors", None)
    out["tensors"] = merged_tensors
    return out


def prune_old_periodic_checkpoints(checkpoints_dir: Path, *, keep: int) -> list[str]:
    """Delete older unshared ``ckpt_u*.pt`` snapshots in ``work`` beyond ``keep`` newest.

    Never deletes files with ``st_nlink > 1`` (already hard-linked into ``output/``) so
    published export ``st_ctime_ns`` metadata remains immutable.
    """
    if not checkpoints_dir.is_dir():
        return []
    unshared: list[Path] = []
    for pt_path in sorted(checkpoints_dir.glob("ckpt_u*.pt")):
        if pt_path.name.endswith("_stop.pt"):
            continue
        try:
            if pt_path.stat().st_nlink > 1:
                continue
        except OSError:
            continue
        unshared.append(pt_path)
    if len(unshared) <= int(keep):
        return []
    removed: list[str] = []
    for old_path in unshared[: len(unshared) - int(keep)]:
        if old_path.exists() and old_path.stat().st_nlink > 1:
            continue
        remove_completed_artifact(old_path.with_suffix(".meta.json"))
        remove_completed_artifact(old_path)
        removed.append(old_path.name)
    return removed


def find_newest_verified_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Locate newest verified ``final.pt`` or ``ckpt_u*.pt`` inside ``checkpoints_dir``."""
    if not checkpoints_dir.is_dir():
        return None
    final_pt = checkpoints_dir / "final.pt"
    if (
        final_pt.is_file()
        and Path(f"{final_pt}.sha256").is_file()
        and Path(f"{final_pt}.complete").is_file()
    ):
        verify_completed_artifact(final_pt)
        return final_pt
    for cand in reversed(sorted(checkpoints_dir.glob("ckpt_u*.pt"))):
        if Path(f"{cand}.sha256").is_file() and Path(f"{cand}.complete").is_file():
            verify_completed_artifact(cand)
            return cand
    return None


class V3PPOTrainer:
    """Self-play and M-of-N PPO trainer over ``GameV3Sim`` with 1 ``build_obs`` per tick."""

    def __init__(
        self,
        cfg: V3PPOConfig,
        *,
        device: str | torch.device = "cpu",
        learner_mask: torch.Tensor | None = None,
        mature_sim_state: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = normalize_device(device)
        ensure_cuda_initialized(self.device)
        configure_determinism(cfg.deterministic)
        seed_everything(cfg.seed)

        self.perm_gen = torch.Generator(device="cpu")
        self.perm_gen.manual_seed(int(cfg.seed) ^ 0x5BF0_3635)

        world_seeds = derive_world_seeds(cfg.seed, cfg.num_envs, stream_id=0)
        self.sim = GameV3Sim(
            cfg.game_config,
            num_envs=cfg.num_envs,
            device=self.device,
            seeds=world_seeds,
        )
        if mature_sim_state is not None:
            self.sim.load_state_dict(mature_sim_state)

        if learner_mask is not None:
            if learner_mask.shape != (self.sim.E, self.sim.N) or learner_mask.dtype != torch.bool:
                raise ValueError(
                    f"learner_mask must be bool of shape {(self.sim.E, self.sim.N)}, "
                    f"got {learner_mask.dtype} {tuple(learner_mask.shape)}"
                )
            mask_dev = learner_mask.to(device=self.device)
            self._is_prefix_all_learners = bool(mask_dev.all().item())
        else:
            k_slots = cfg.resolved_learner_slots
            mask_dev = torch.zeros((self.sim.E, self.sim.N), dtype=torch.bool, device=self.device)
            mask_dev[:, :k_slots] = True
            self._is_prefix_all_learners = k_slots == self.sim.N

        self.learner_index = LearnerIndex(self.sim, mask_dev)
        self.M = self.learner_index.count

        net_cfg = EgoMultiScaleNetConfig.from_size(
            cfg.net_size,
            use_channels_last=cfg.use_channels_last,
            use_bf16_autocast=cfg.amp,
        )
        self.raw_model = EgoMultiScaleNet(net_cfg).to(self.device)
        if cfg.compile_model and hasattr(torch, "compile"):
            self.model: EgoMultiScaleNet = torch.compile(self.raw_model)  # type: ignore[assignment]
        else:
            self.model = self.raw_model

        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=cfg.lr,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )

        use_fp16_scaler = (
            self.device.type == "cuda" and bool(cfg.amp) and not torch.cuda.is_bf16_supported()
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

        self.buffer = Uint8RolloutBuffer(
            rollout_len=cfg.rollout_len,
            learner_index=self.learner_index,
            sim_shape=(self.sim.E, self.sim.N),
            device=self.device,
        )
        self._bootstrap_obs = allocate_obs_batch(self.sim, self.learner_index)
        self._bootstrap_obs_valid = False

        # Persistent counters and episode return accumulators (serialized in checkpoints)
        self.updates_completed: int = 0
        self.world_steps_collected: int = 0
        self.agent_steps_collected: int = 0
        self.learner_steps_collected: int = 0
        self.episodes_completed: int = 0
        self.obs_builds_executed: int = 0
        self.episode_return_acc = torch.zeros((self.M,), dtype=torch.float32, device=self.device)
        self.last_rollout_mean_ep_return: float = 0.0
        self.last_eval_metrics: dict[str, Any] | None = None

    @torch.no_grad()
    def _act_learners_chunked(
        self,
        obs_s1: torch.Tensor,
        obs_s4: torch.Tensor,
        obs_s16: torch.Tensor,
        obs_scalars: torch.Tensor,
        obs_mask: torch.Tensor,
        *,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m_total = int(obs_s1.shape[0])
        chunk = max(1, min(m_total, self.cfg.resolved_actor_chunk_learners))
        if chunk == m_total:
            out = self.model.act(
                obs_s1,
                obs_s4,
                obs_s16,
                obs_scalars,
                action_mask=obs_mask,
                greedy=greedy,
            )
            return out.actions, out.log_probs, out.values

        actions = torch.empty((m_total,), dtype=torch.long, device=self.device)
        log_probs = torch.empty((m_total,), dtype=torch.float32, device=self.device)
        values = torch.empty((m_total,), dtype=torch.float32, device=self.device)
        for start in range(0, m_total, chunk):
            end = min(m_total, start + chunk)
            out = self.model.act(
                obs_s1[start:end],
                obs_s4[start:end],
                obs_s16[start:end],
                obs_scalars[start:end],
                action_mask=obs_mask[start:end],
                greedy=greedy,
            )
            actions[start:end] = out.actions
            log_probs[start:end] = out.log_probs
            values[start:end] = out.values
        return actions, log_probs, values

    @torch.no_grad()
    def _bootstrap_values_chunked(self) -> torch.Tensor:
        obs = self._bootstrap_obs
        m_total = int(obs.raster_s1.shape[0])
        chunk = max(1, min(m_total, self.cfg.resolved_actor_chunk_learners))
        if chunk == m_total:
            return self.model(
                obs.raster_s1,
                obs.raster_s4,
                obs.raster_s16,
                obs.scalars,
                action_mask=obs.action_mask,
            ).value

        values = torch.empty((m_total,), dtype=torch.float32, device=self.device)
        for start in range(0, m_total, chunk):
            end = min(m_total, start + chunk)
            values[start:end] = self.model(
                obs.raster_s1[start:end],
                obs.raster_s4[start:end],
                obs.raster_s16[start:end],
                obs.scalars[start:end],
                action_mask=obs.action_mask[start:end],
            ).value
        return values

    @torch.no_grad()
    def collect_rollout(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
        stage_timing: bool = False,
    ) -> dict[str, float]:
        """Collect ``T`` ticks into ``self.buffer`` with strictly one ``build_obs`` per tick."""
        self.model.eval()
        t_start = time.perf_counter()

        sim_sec = 0.0
        obs_sec = 0.0
        forward_sec = 0.0

        ep_ret_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        ep_ret_count = torch.zeros((), dtype=torch.float32, device=self.device)
        zeros_m = torch.zeros((self.M,), dtype=torch.float32, device=self.device)

        env_idx = self.learner_index.env_idx
        slot_idx = self.learner_index.slot_idx

        for t in range(self.cfg.rollout_len):
            if should_stop is not None and should_stop():
                raise StopRequestedError("Cooperative stop requested during rollout collection")

            if stage_timing:
                synchronize_device(self.device)
                t_obs0 = time.perf_counter()
            if t == 0 and self._bootstrap_obs_valid:
                obs_t = self.buffer.copy_obs_into_step(0, self._bootstrap_obs)
            else:
                obs_t = build_obs(
                    self.sim,
                    learner_index=self.learner_index,
                    out=self.buffer.obs_view(t),
                )
                self.obs_builds_executed += 1
            if stage_timing:
                synchronize_device(self.device)
                obs_sec += time.perf_counter() - t_obs0

            acted_es = self.sim.alive & (~self.sim.truncated_pending)
            acted_m = acted_es[env_idx, slot_idx]

            if stage_timing:
                t_fwd0 = time.perf_counter()
            learner_actions, log_probs, values = self._act_learners_chunked(
                obs_t.raster_s1,
                obs_t.raster_s4,
                obs_t.raster_s16,
                obs_t.scalars,
                obs_t.action_mask,
                greedy=False,
            )
            if self._is_prefix_all_learners:
                world_actions = learner_actions.view(self.sim.E, self.sim.N)
            else:
                world_actions = scripted_opponent_actions(self.sim, self.cfg.opponent_policy)
                world_actions[env_idx, slot_idx] = learner_actions
            if stage_timing:
                synchronize_device(self.device)
                forward_sec += time.perf_counter() - t_fwd0

            if stage_timing:
                t_sim0 = time.perf_counter()
            is_rollout_end = t == (self.cfg.rollout_len - 1)
            info = self.sim.step(world_actions, truncate=is_rollout_end)
            step_rew_es = compute_step_reward(info, acted_es, self.cfg)
            if stage_timing:
                synchronize_device(self.device)
                sim_sec += time.perf_counter() - t_sim0

            rew_m = step_rew_es[env_idx, slot_idx]
            died_m = acted_m & info.died[env_idx, slot_idx]
            survived_m = acted_m & (~info.died[env_idx, slot_idx])
            trunc_m = survived_m & info.truncated[env_idx, slot_idx]

            self.buffer.actions[t] = learner_actions.to(torch.uint8)
            self.buffer.log_probs[t] = log_probs
            self.buffer.values[t] = values
            self.buffer.rewards[t] = rew_m
            self.buffer.died[t] = died_m
            self.buffer.truncated[t] = trunc_m
            self.buffer.acted[t] = acted_m

            self.episode_return_acc = torch.where(
                acted_m,
                self.episode_return_acc + rew_m,
                self.episode_return_acc,
            )
            # A life ends on death (rollout-end truncation keeps big snakes alive in the world).
            life_done = died_m | self.sim.truncated_pending[env_idx, slot_idx]
            ep_ret_sum = ep_ret_sum + torch.where(life_done, self.episode_return_acc, zeros_m).sum()
            ep_ret_count = ep_ret_count + life_done.to(torch.float32).sum()
            self.episode_return_acc = torch.where(
                life_done | (~survived_m),
                zeros_m,
                self.episode_return_acc,
            )

        # Build post-rollout `s_T` obs ONCE into `_bootstrap_obs` (reused at `t=0` next update)
        if stage_timing:
            synchronize_device(self.device)
            t_obs_boot = time.perf_counter()
        build_obs(
            self.sim,
            learner_index=self.learner_index,
            out=self._bootstrap_obs,
        )
        self.obs_builds_executed += 1
        self._bootstrap_obs_valid = True
        if stage_timing:
            synchronize_device(self.device)
            # Only the very first rollout executes T+1 `build_obs`; steady-state rollouts execute T.
            obs_sec += time.perf_counter() - t_obs_boot
            t_fwd_boot = time.perf_counter()

        last_values = self._bootstrap_values_chunked()
        if stage_timing:
            synchronize_device(self.device)
            forward_sec += time.perf_counter() - t_fwd_boot

        self.buffer.compute_gae(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        alive_now = self.sim.alive[env_idx, slot_idx]
        alive_cnt = alive_now.to(torch.float32).sum().clamp_min(1.0)
        fallback_ret = torch.where(alive_now, self.episode_return_acc, zeros_m).sum() / alive_cnt
        done_cnt_int = int(ep_ret_count.detach().cpu().item())
        mean_ep_ret = float(
            torch.where(ep_ret_count > 0.0, ep_ret_sum / ep_ret_count.clamp_min(1.0), fallback_ret)
            .detach()
            .cpu()
            .item()
        )

        synchronize_device(self.device)
        collect_sec = time.perf_counter() - t_start

        self.episodes_completed += done_cnt_int
        self.last_rollout_mean_ep_return = mean_ep_ret
        self.world_steps_collected += self.cfg.rollout_len * self.sim.E
        self.agent_steps_collected += self.cfg.rollout_len * self.sim.E * self.sim.N
        self.learner_steps_collected += self.cfg.rollout_len * self.M

        out_timing: dict[str, float] = {"collect_seconds": collect_sec}
        if stage_timing:
            out_timing["sim_seconds"] = sim_sec
            out_timing["obs_seconds"] = obs_sec
            out_timing["forward_seconds"] = forward_sec
        return out_timing

    def ppo_update(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, float], float]:
        """Run ``ppo_epochs`` of minibatch clipped PPO + symlog two-hot value updates."""
        self.model.train()
        t0 = time.perf_counter()

        valid_idx = self.buffer.valid_flat_indices()
        n_valid = int(valid_idx.numel())
        if n_valid == 0:
            synchronize_device(self.device)
            return (
                {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "approx_kl": 0.0,
                    "epoch0_approx_kl": 0.0,
                    "clip_frac": 0.0,
                    "grad_norm": 0.0,
                    "valid_learner_rows": 0.0,
                    "minibatches": 0.0,
                },
                time.perf_counter() - t0,
            )

        tm = self.buffer.T * self.buffer.M
        adv_flat = self.buffer.advantages.reshape(tm)
        if self.cfg.normalize_advantages and n_valid > 1:
            valid_adv = adv_flat.index_select(0, valid_idx)
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std(unbiased=False).clamp_min(1.0e-8)
            norm_adv_flat = (adv_flat - adv_mean) / adv_std
        else:
            norm_adv_flat = adv_flat

        mb_size = max(1, min(n_valid, self.cfg.resolved_minibatch_size))
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        epoch0_kl_sum = 0.0
        epoch0_batches = 0
        clip_frac_sum = 0.0
        grad_norm_sum = 0.0
        num_minibatches = 0

        for epoch in range(int(self.cfg.ppo_epochs)):
            if should_stop is not None and should_stop():
                raise StopRequestedError("Cooperative stop requested during PPO learner epochs")

            perm_cpu = torch.randperm(n_valid, generator=self.perm_gen)
            shuffled_valid = valid_idx.index_select(0, perm_cpu.to(device=self.device))

            for start in range(0, n_valid, mb_size):
                if should_stop is not None and should_stop():
                    raise StopRequestedError("Cooperative stop requested between PPO minibatches")
                batch_idx = shuffled_valid[start : min(n_valid, start + mb_size)]
                mb = self.buffer.gather_minibatch(
                    batch_idx,
                    normalized_advantages_flat=norm_adv_flat,
                )

                self.optimizer.zero_grad(set_to_none=True)
                eval_out = self.model.evaluate_actions(
                    mb.raster_s1,
                    mb.raster_s4,
                    mb.raster_s16,
                    mb.scalars,
                    mb.actions,
                    action_mask=mb.action_mask,
                )

                log_ratio = (eval_out.log_probs - mb.old_log_probs).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb.advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                    * mb.advantages
                )
                policy_loss = -torch.minimum(surr1, surr2).mean()
                value_loss = symlog_two_hot_loss(
                    eval_out.value_logits,
                    mb.returns,
                    self.raw_model.bin_centers,
                )
                entropy = eval_out.entropy.mean()
                total_loss = (
                    policy_loss
                    + float(self.cfg.value_coef) * value_loss
                    - float(self.cfg.entropy_coef) * entropy
                )

                if not bool(torch.isfinite(total_loss).all().detach().cpu().item()):
                    raise RuntimeError(
                        f"Non-finite PPO loss encountered at update {self.updates_completed + 1}: "
                        f"policy={float(policy_loss.item())}, value={float(value_loss.item())}, "
                        f"entropy={float(entropy.item())}"
                    )

                if self.scaler.is_enabled():
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(),
                        self.cfg.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(),
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().detach().cpu().item())
                    clip_frac = float(
                        ((ratio - 1.0).abs() > self.cfg.clip_ratio)
                        .to(torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )

                policy_loss_sum += float(policy_loss.detach().cpu().item())
                value_loss_sum += float(value_loss.detach().cpu().item())
                entropy_sum += float(entropy.detach().cpu().item())
                approx_kl_sum += approx_kl
                if epoch == 0:
                    epoch0_kl_sum += approx_kl
                    epoch0_batches += 1
                clip_frac_sum += clip_frac
                grad_norm_sum += float(grad_norm.detach().cpu().item())
                num_minibatches += 1

        self.updates_completed += 1
        synchronize_device(self.device)
        learn_sec = time.perf_counter() - t0
        inv = 1.0 / float(max(1, num_minibatches))
        stats = {
            "policy_loss": policy_loss_sum * inv,
            "value_loss": value_loss_sum * inv,
            "entropy": entropy_sum * inv,
            "approx_kl": approx_kl_sum * inv,
            "epoch0_approx_kl": epoch0_kl_sum / float(max(1, epoch0_batches)),
            "clip_frac": clip_frac_sum * inv,
            "grad_norm": grad_norm_sum * inv,
            "valid_learner_rows": float(n_valid),
            "minibatches": float(num_minibatches),
        }
        return stats, learn_sec

    def train_step(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
        stage_timing: bool = False,
    ) -> dict[str, Any]:
        """Run one complete rollout + PPO update and return metrics + timing."""
        coll = self.collect_rollout(should_stop=should_stop, stage_timing=stage_timing)
        ppo_stats, learn_sec = self.ppo_update(should_stop=should_stop)
        collect_sec = float(coll["collect_seconds"])
        total_sec = max(1.0e-9, collect_sec + learn_sec)
        agent_steps = self.cfg.rollout_len * self.sim.E * self.sim.N
        learner_steps = self.cfg.rollout_len * self.M
        out: dict[str, Any] = {
            "update": self.updates_completed,
            "agent_steps": agent_steps,
            "learner_steps": learner_steps,
            "total_agent_steps": self.agent_steps_collected,
            "total_learner_steps": self.learner_steps_collected,
            "total_world_steps": self.world_steps_collected,
            "episodes_completed": self.episodes_completed,
            "mean_ep_return": round(self.last_rollout_mean_ep_return, 6),
            "collect_seconds": round(collect_sec, 6),
            "learn_seconds": round(learn_sec, 6),
            "update_seconds": round(total_sec, 6),
            "agent_steps_per_sec": round(float(agent_steps) / total_sec, 2),
            "learner_steps_per_sec": round(float(learner_steps) / total_sec, 2),
            **{k: round(v, 6) for k, v in ppo_stats.items()},
        }
        if stage_timing:
            out["sim_seconds"] = round(float(coll["sim_seconds"]), 6)
            out["obs_seconds"] = round(float(coll["obs_seconds"]), 6)
            out["forward_seconds"] = round(float(coll["forward_seconds"]), 6)
        return out

    def run_evaluation(self) -> dict[str, Any]:
        """Run built-in deterministic fixed-seed M-of-N evaluation against scripted opponents."""
        metrics = evaluate_policy(self.raw_model, self.cfg, device=self.device)
        self.last_eval_metrics = metrics
        return metrics

    def checkpoint_dict(self, *, compress_large_grids: bool = True) -> dict[str, Any]:
        """Build full deterministic CPU-resident checkpoint dictionary."""
        raw_sim_sd = self.sim.state_dict()
        sim_sd = pack_sim_state_dict(raw_sim_sd) if compress_large_grids else raw_sim_sd
        return {
            "schema": V3_PPO_CHECKPOINT_SCHEMA,
            "namespace": V3_PPO_NAMESPACE,
            "config": self.cfg.to_dict(),
            "model_state_dict": _to_cpu_tree(self.raw_model.state_dict()),
            "optimizer_state_dict": _to_cpu_tree(self.optimizer.state_dict()),
            "grad_scaler_state_dict": (
                _to_cpu_tree(self.scaler.state_dict()) if self.scaler.is_enabled() else None
            ),
            "sim_state_dict": sim_sd,
            "sim_clean_step_streak": int(self.sim._clean_step_streak),
            "learner_mask": self.learner_index.mask.detach().cpu().clone(),
            "counters": {
                "updates_completed": int(self.updates_completed),
                "world_steps_collected": int(self.world_steps_collected),
                "agent_steps_collected": int(self.agent_steps_collected),
                "learner_steps_collected": int(self.learner_steps_collected),
                "episodes_completed": int(self.episodes_completed),
                "obs_builds_executed": int(self.obs_builds_executed),
                "last_rollout_mean_ep_return": float(self.last_rollout_mean_ep_return),
            },
            "episode_return_acc": self.episode_return_acc.detach().cpu().clone(),
            "last_eval_metrics": self.last_eval_metrics,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().clone(),
                "torch_cuda": (
                    [s.detach().cpu().clone() for s in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available()
                    else None
                ),
                "perm_gen": self.perm_gen.get_state().clone(),
            },
        }

    def save_checkpoint(self, path: Path) -> str:
        """Atomically write checkpoint with ``.sha256`` and ``.complete`` sidecars."""
        payload = self.checkpoint_dict()
        return write_create_only_artifact(path, lambda tmp: torch.save(payload, tmp))

    def load_checkpoint(
        self,
        path: Path,
        *,
        verify_markers: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore trainer, optimizer, simulator, counters, and RNG states in-place."""
        ckpt_path = Path(path)
        if verify_markers:
            verify_completed_artifact(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or ckpt.get("schema") != V3_PPO_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"Invalid checkpoint schema at {ckpt_path}: "
                f"{ckpt.get('schema') if isinstance(ckpt, dict) else type(ckpt)!r}"
            )
        if ckpt.get("namespace") != V3_PPO_NAMESPACE:
            raise ValueError(
                f"Foreign checkpoint namespace {ckpt.get('namespace')!r} != {V3_PPO_NAMESPACE!r}"
            )

        ckpt_cfg = V3PPOConfig.from_dict(ckpt["config"])
        curr_dict = self.cfg.to_dict()
        ckpt_dict = ckpt_cfg.to_dict()
        for key in set(curr_dict) | set(ckpt_dict):
            if key in _RESUME_MUTABLE_CONFIG_KEYS:
                continue
            if curr_dict.get(key) != ckpt_dict.get(key):
                raise ValueError(
                    f"Checkpoint config mismatch on {key!r}: "
                    f"ckpt={ckpt_dict.get(key)!r} vs current={curr_dict.get(key)!r}"
                )

        saved_mask = ckpt.get("learner_mask")
        if isinstance(saved_mask, torch.Tensor):
            if not torch.equal(
                saved_mask.to(device=self.device, dtype=torch.bool), self.learner_index.mask
            ):
                raise ValueError(
                    "Checkpoint learner_mask does not match current trainer learner_mask"
                )

        self.raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        _move_optimizer_state_to_device(self.optimizer, self.device)

        scaler_state = ckpt.get("grad_scaler_state_dict")
        if self.scaler.is_enabled() and isinstance(scaler_state, dict):
            self.scaler.load_state_dict(scaler_state)

        self.sim.load_state_dict(unpack_sim_state_dict(ckpt["sim_state_dict"]))
        self.sim._clean_step_streak = int(ckpt.get("sim_clean_step_streak", 0))
        self._bootstrap_obs_valid = False

        counters = ckpt["counters"]
        self.updates_completed = int(counters["updates_completed"])
        self.world_steps_collected = int(counters["world_steps_collected"])
        self.agent_steps_collected = int(counters["agent_steps_collected"])
        self.learner_steps_collected = int(counters["learner_steps_collected"])
        self.episodes_completed = int(counters.get("episodes_completed", 0))
        self.obs_builds_executed = int(counters.get("obs_builds_executed", 0))
        self.last_rollout_mean_ep_return = float(counters.get("last_rollout_mean_ep_return", 0.0))

        self.episode_return_acc.copy_(
            ckpt["episode_return_acc"].to(device=self.device, dtype=torch.float32)
        )
        self.last_eval_metrics = ckpt.get("last_eval_metrics")

        if restore_rng and isinstance(ckpt.get("rng_state"), dict):
            rng = ckpt["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])
            if (
                torch.cuda.is_available()
                and isinstance(rng.get("torch_cuda"), list)
                and len(rng["torch_cuda"]) == torch.cuda.device_count()
            ):
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            if isinstance(rng.get("perm_gen"), torch.Tensor):
                self.perm_gen.set_state(rng["perm_gen"])

        return ckpt


def verify_staged_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    verify_markers: bool = True,
) -> dict[str, Any]:
    """Validate checkpoint integrity by instantiating a temporary trainer and loading state."""
    ckpt_path = Path(path)
    if verify_markers:
        verify_completed_artifact(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("schema") != V3_PPO_CHECKPOINT_SCHEMA:
        raise ValueError(f"Invalid v3_ppo checkpoint at {ckpt_path}")
    cfg = V3PPOConfig.from_dict(raw["config"])
    trainer = V3PPOTrainer(cfg, device=device)
    trainer.load_checkpoint(ckpt_path, verify_markers=verify_markers, restore_rng=False)
    return {
        "schema": raw["schema"],
        "namespace": raw["namespace"],
        "sha256": sha256_file(ckpt_path),
        "updates_completed": int(trainer.updates_completed),
        "agent_steps_collected": int(trainer.agent_steps_collected),
    }


def _publish_checkpoint_manifest(
    work_ckpt_path: Path,
    out_checkpoints_dir: Path,
    *,
    export_names: list[str],
    reason: str,
    counters: dict[str, Any],
    write_latest_manifest: bool = False,
) -> list[str]:
    out_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    verify_completed_artifact(work_ckpt_path)
    unique_names = list(dict.fromkeys(export_names))
    targets = [
        out_checkpoints_dir / name
        for name in unique_names
        if not (out_checkpoints_dir / name).exists()
    ]
    ckpt_sha = (
        publish_hardlinked_artifacts(work_ckpt_path, targets, staging_dir=work_ckpt_path.parent)
        if targets
        else sha256_file(work_ckpt_path)
    )
    published_files: list[str] = []
    for dst in targets:
        meta_path = dst.with_suffix(".meta.json")
        if not meta_path.exists():
            write_create_only_json(
                meta_path,
                {
                    "schema": "v3-ppo-checkpoint-meta/v1",
                    "namespace": V3_PPO_NAMESPACE,
                    "reason": reason,
                    "checkpoint_file": dst.name,
                    "sha256": ckpt_sha,
                    "size_bytes": int(dst.stat().st_size),
                    **counters,
                },
            )
        published_files.append(dst.name)

    if write_latest_manifest:
        manifest_path = out_checkpoints_dir / "latest_manifest.json"
        if not manifest_path.exists():
            all_exported = sorted(p.name for p in out_checkpoints_dir.glob("*.pt"))
            primary = (
                unique_names[-1] if unique_names else (all_exported[-1] if all_exported else "")
            )
            write_create_only_json(
                manifest_path,
                {
                    "schema": "v3-ppo-latest-checkpoint/v1",
                    "namespace": V3_PPO_NAMESPACE,
                    "reason": reason,
                    "primary_checkpoint": primary,
                    "sha256": ckpt_sha,
                    "size_bytes": int(work_ckpt_path.stat().st_size),
                    "published_files": all_exported,
                    **counters,
                },
            )
    return published_files


def _format_progress_line(
    step_metrics: dict[str, Any],
    *,
    total_updates: int,
    wall_elapsed: float,
    eval_metrics: dict[str, Any] | None = None,
) -> str:
    line = (
        f"V3_PPO_PROGRESS "
        f"update={step_metrics['update']}/{total_updates} "
        f"agent_steps={step_metrics['total_agent_steps']} "
        f"sps={step_metrics['agent_steps_per_sec']:.1f} "
        f"ep_ret={step_metrics['mean_ep_return']:.4f} "
        f"p_loss={step_metrics['policy_loss']:.4f} "
        f"v_loss={step_metrics['value_loss']:.4f} "
        f"ent={step_metrics['entropy']:.4f} "
        f"kl={step_metrics['approx_kl']:.5f} "
        f"wall_s={wall_elapsed:.2f}"
    )
    if eval_metrics is not None:
        comb = eval_metrics["combined"]
        line += (
            f" eval_len_mean={comb['mean_final_length']:.2f}"
            f" eval_len_p95={comb['p95_final_length']:.2f}"
            f" eval_deaths_1k={comb['deaths_per_1k_agent_steps']:.2f}"
            f" eval_kills={comb['kills']}"
            f" eval_ret={comb['mean_episode_return']:.4f}"
        )
    return line


def run_training_job(
    *,
    cfg: V3PPOConfig,
    output_dir: Path,
    work_dir: Path | None = None,
    device: str | torch.device = "cpu",
    input_dir: Path | None = None,
    ledger_path: Path | None = None,
    max_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Execute training loop with periodic evaluation, 30-min atomic checkpoints, and wall guard."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wrk = Path(work_dir).resolve() if work_dir is not None else (out.parent / f".{out.name}_work")
    wrk.mkdir(parents=True, exist_ok=True)
    work_ckpts = wrk / "checkpoints"
    work_ckpts.mkdir(parents=True, exist_ok=True)
    out_ckpts = out / "checkpoints"
    out_ckpts.mkdir(parents=True, exist_ok=True)

    wall_t0 = time.monotonic()
    trainer = V3PPOTrainer(cfg, device=device)

    resumed_from: dict[str, Any] | None = None
    if ledger_path is not None and Path(ledger_path).is_file():
        ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
        if ledger.get("namespace") not in (None, V3_PPO_NAMESPACE):
            raise ValueError(
                f"Foreign ledger namespace {ledger.get('namespace')!r} != {V3_PPO_NAMESPACE!r}"
            )
        resume_entry = ledger.get("resume_checkpoint")
        if isinstance(resume_entry, dict):
            if input_dir is None:
                raise ValueError("ledger specifies resume_checkpoint, but input_dir is None")
            staged_ckpt = Path(input_dir) / str(resume_entry["relpath"])
            if not staged_ckpt.is_file():
                raise FileNotFoundError(f"Missing staged checkpoint at {staged_ckpt}")
            actual_sha = sha256_file(staged_ckpt)
            if actual_sha != resume_entry.get("sha256"):
                raise ValueError(
                    "Staged checkpoint SHA-256 mismatch: "
                    f"{actual_sha} != {resume_entry.get('sha256')}"
                )
            trainer.load_checkpoint(staged_ckpt, verify_markers=False, restore_rng=True)
            resumed_from = {
                "relpath": str(resume_entry["relpath"]),
                "sha256": actual_sha,
                "updates_completed_at_resume": int(trainer.updates_completed),
            }
            phase_stamp(
                "train_resume",
                "OK",
                relpath=resumed_from["relpath"],
                updates_completed=trainer.updates_completed,
            )

    write_create_only_json(out / "config.json", cfg.to_dict())

    metrics_writer = RotatingCreateOnlyJSONLWriter(out / "metrics", prefix="updates")
    eval_writer = RotatingCreateOnlyJSONLWriter(out / "eval", prefix="eval")

    updates_this_session = 0
    stopped_reason = "completed"
    error_message: str | None = None
    last_ckpt_mono = time.monotonic()
    last_pub_mono = time.monotonic()
    checkpoint_reserve_sec = (
        min(90.0, max(5.0, float(max_wall_seconds) * 0.08)) if max_wall_seconds is not None else 0.0
    )

    with SignalStopFlag() as stop_flag:

        def _wall_expired() -> bool:
            if max_wall_seconds is None:
                return False
            return (time.monotonic() - wall_t0) >= (
                float(max_wall_seconds) - checkpoint_reserve_sec
            )

        def _should_stop() -> bool:
            return bool(stop_flag.stopped)

        try:
            phase_stamp(
                "train_loop",
                "START",
                device=str(trainer.device),
                num_envs=cfg.num_envs,
                learner_slots=cfg.resolved_learner_slots,
                net_size=cfg.net_size,
                start_update=trainer.updates_completed,
                target_updates=cfg.total_updates,
            )
            while trainer.updates_completed < int(cfg.total_updates):
                if stop_flag.stopped:
                    stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
                    break
                if _wall_expired():
                    stopped_reason = "wall_clock_guard"
                    break

                u_t0 = time.monotonic()
                step_metrics = trainer.train_step(should_stop=_should_stop)
                u_elapsed = time.monotonic() - u_t0
                updates_this_session += 1

                if cfg.max_update_seconds is not None and u_elapsed > float(cfg.max_update_seconds):
                    raise RuntimeError(
                        f"Update {trainer.updates_completed} took {u_elapsed:.2f}s > "
                        f"max_update_seconds={cfg.max_update_seconds:.2f}s"
                    )

                eval_res: dict[str, Any] | None = None
                is_target_end = trainer.updates_completed >= int(cfg.total_updates)
                if int(cfg.eval_every_updates) > 0 and (
                    (trainer.updates_completed % int(cfg.eval_every_updates) == 0) or is_target_end
                ):
                    eval_res = trainer.run_evaluation()
                    eval_record = {
                        "update": trainer.updates_completed,
                        "total_agent_steps": trainer.agent_steps_collected,
                        **eval_res,
                    }
                    eval_writer.append(eval_record)

                metrics_writer.append(step_metrics)
                now_mono = time.monotonic()
                print(
                    _format_progress_line(
                        step_metrics,
                        total_updates=int(cfg.total_updates),
                        wall_elapsed=now_mono - wall_t0,
                        eval_metrics=eval_res,
                    ),
                    flush=True,
                )

                due_by_updates = int(cfg.checkpoint_every_updates) > 0 and (
                    trainer.updates_completed % int(cfg.checkpoint_every_updates) == 0
                )
                due_by_time = (now_mono - last_ckpt_mono) >= float(cfg.checkpoint_interval_sec)
                if due_by_updates or due_by_time:
                    snap_name = f"ckpt_u{trainer.updates_completed:07d}.pt"
                    snap_work = work_ckpts / snap_name
                    if not snap_work.exists():
                        trainer.save_checkpoint(snap_work)
                    prune_old_periodic_checkpoints(
                        work_ckpts, keep=int(cfg.keep_periodic_checkpoints)
                    )
                    last_ckpt_mono = now_mono

                    if due_by_updates or (
                        (now_mono - last_pub_mono) >= float(cfg.output_publish_interval_sec)
                    ):
                        _publish_checkpoint_manifest(
                            snap_work,
                            out_ckpts,
                            export_names=[snap_name],
                            reason="periodic",
                            counters={
                                "updates_completed": trainer.updates_completed,
                                "agent_steps_collected": trainer.agent_steps_collected,
                                "world_steps_collected": trainer.world_steps_collected,
                            },
                            write_latest_manifest=False,
                        )
                        metrics_writer.seal_current_segment()
                        eval_writer.seal_current_segment()
                        last_pub_mono = now_mono

        except StopRequestedError:
            stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
        except Exception as exc:
            stopped_reason = f"exception:{type(exc).__name__}"
            error_message = str(exc)
            raise
        finally:
            metrics_writer.close()
            eval_writer.close()

            # Run final evaluation if none ran yet and no fatal exception occurred
            if (
                error_message is None
                and int(cfg.eval_every_updates) > 0
                and trainer.last_eval_metrics is None
                and trainer.updates_completed > 0
            ):
                trainer.run_evaluation()

            # Always publish atomic checkpoint on completion or cooperative stop
            final_work = work_ckpts / f"ckpt_u{trainer.updates_completed:07d}_stop.pt"
            if not final_work.exists():
                trainer.save_checkpoint(final_work)
            export_names = ["final.pt"]
            periodic_name = f"ckpt_u{trainer.updates_completed:07d}.pt"
            if not (out_ckpts / periodic_name).exists():
                export_names.insert(0, periodic_name)
            _publish_checkpoint_manifest(
                final_work,
                out_ckpts,
                export_names=export_names,
                reason=stopped_reason,
                counters={
                    "updates_completed": trainer.updates_completed,
                    "agent_steps_collected": trainer.agent_steps_collected,
                    "world_steps_collected": trainer.world_steps_collected,
                },
                write_latest_manifest=True,
            )

            passed = (
                error_message is None
                and stopped_reason in {"completed", "wall_clock_guard"}
                and (int(cfg.total_updates) == 0 or trainer.updates_completed >= 1)
                and (
                    stopped_reason == "completed"
                    or (max_wall_seconds is not None and updates_this_session >= 1)
                )
            )
            status = "PASS" if passed else "FAIL"

            result_path = out / "train_result.json"
            if not result_path.exists():
                result_payload = {
                    "schema": V3_PPO_TRAIN_RESULT_SCHEMA,
                    "namespace": V3_PPO_NAMESPACE,
                    "status": status,
                    "stopped_reason": stopped_reason,
                    "error": error_message,
                    "device": str(trainer.device),
                    "parameter_count": trainer.raw_model.parameter_count(),
                    "updates_completed": trainer.updates_completed,
                    "updates_this_session": updates_this_session,
                    "world_steps_collected": trainer.world_steps_collected,
                    "agent_steps_collected": trainer.agent_steps_collected,
                    "learner_steps_collected": trainer.learner_steps_collected,
                    "episodes_completed": trainer.episodes_completed,
                    "last_rollout_mean_ep_return": round(trainer.last_rollout_mean_ep_return, 6),
                    "last_eval": trainer.last_eval_metrics,
                    "resumed_from": resumed_from,
                    "memory_estimate": estimate_training_memory(cfg),
                    "config": cfg.to_dict(),
                    "wall_seconds": round(time.monotonic() - wall_t0, 4),
                }
                write_create_only_json(result_path, result_payload)
            phase_stamp(
                "train_loop",
                "END",
                status=status,
                stopped_reason=stopped_reason,
                updates_completed=trainer.updates_completed,
            )

    return json.loads((out / "train_result.json").read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m research.v3_ppo.ppo``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=os.environ.get("SNAKE_DQN_DEVICE", "cpu"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--config-json", type=str, default=os.environ.get("V3_PPO_CONFIG_JSON"))

    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-snakes", type=int, default=24)
    parser.add_argument("--arena-radius", type=float, default=600.0)
    parser.add_argument("--learner-slots", type=int, default=None)
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="food_seeking",
        choices=("food_seeking", "random_safe", "cardinal6_safe"),
    )
    parser.add_argument("--net-size", type=str, default="S", choices=("S", "M"))
    parser.add_argument("--rollout-len", type=int, default=32)
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--checkpoint-interval-sec", type=float, default=1800.0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=0)
    parser.add_argument("--eval-every-updates", type=int, default=25)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--eval-ticks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--compile", action="store_true")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=True)
    amp_group.add_argument("--no-amp", "--fp32", dest="amp", action="store_false")

    cl_group = parser.add_mutually_exclusive_group()
    cl_group.add_argument(
        "--channels-last", dest="use_channels_last", action="store_true", default=True
    )
    cl_group.add_argument("--no-channels-last", dest="use_channels_last", action="store_false")

    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=(
            float(os.environ["V3_PPO_MAX_WALL_SECONDS"])
            if os.environ.get("V3_PPO_MAX_WALL_SECONDS")
            else None
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; exits non-zero whenever terminal ``status != 'PASS'``."""
    args = build_arg_parser().parse_args(argv)
    if args.config_json:
        cfg = V3PPOConfig.from_dict(json.loads(args.config_json))
    else:
        game_cfg = GameV3Config(
            arena_radius=float(args.arena_radius),
            num_snakes=int(args.num_snakes),
        )
        cfg = V3PPOConfig(
            game_config=game_cfg,
            num_envs=int(args.num_envs),
            learner_slots=args.learner_slots,
            opponent_policy=args.opponent_policy,
            net_size=args.net_size,
            amp=bool(args.amp),
            use_channels_last=bool(args.use_channels_last),
            compile_model=bool(args.compile),
            rollout_len=int(args.rollout_len),
            ppo_epochs=int(args.ppo_epochs),
            minibatch_size=args.minibatch_size,
            lr=float(args.lr),
            total_updates=int(args.total_updates),
            checkpoint_interval_sec=float(args.checkpoint_interval_sec),
            output_publish_interval_sec=float(args.checkpoint_interval_sec),
            checkpoint_every_updates=int(args.checkpoint_every_updates),
            eval_every_updates=int(args.eval_every_updates),
            eval_envs=int(args.eval_envs),
            eval_ticks=int(args.eval_ticks),
            seed=int(args.seed),
        )

    res = run_training_job(
        cfg=cfg,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        device=args.device,
        input_dir=args.input_dir,
        ledger_path=args.ledger,
        max_wall_seconds=args.max_wall_seconds,
    )
    if res.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
