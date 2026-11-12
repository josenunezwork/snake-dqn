"""Game V3.1 self-play and M-of-N PPO rollout collector, learner update, and trainer state."""

from __future__ import annotations

import inspect
import io
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from research.arena_ppo.runtime import (
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    synchronize_device,
    verify_completed_artifact,
)
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs
from research.game_v3.sim import GameV3Sim
from research.v3_ppo.buffer import Uint8RolloutBuffer, assert_obs_batch_matches_layout
from research.v3_ppo.checkpoint_io import (
    _SIM_ZLIB_TENSOR_MIN_BYTES,
    V3_PPO_FULL_INDEX_SCHEMA,
    V3_PPO_LIGHT_CHECKPOINT_SCHEMA,
    V3_PPO_SIM_SHARD_SCHEMA,
    concat_sim_state_shards,
    pack_sim_state_dict,
    slice_sim_state_dict,
    verify_full_checkpoint_set,
    verify_light_checkpoint,
)
from research.v3_ppo.config import (
    V3_PPO_NAMESPACE,
    ObsLayout,
    V3PPOConfig,
    assert_resume_config_compatible,
    build_recipe_change_record,
    merge_and_append_recipe_history,
    obs_layout_from_game_config,
)
from research.v3_ppo.eval import (
    compute_step_reward,
    derive_world_seeds,
    evaluate_policy,
    scripted_opponent_actions,
)
from research.v3_ppo.exports import write_create_only_artifact, write_create_only_json
from research.v3_ppo.net import (
    EgoMultiScaleNet,
    EgoMultiScaleNetConfig,
    symlog_two_hot_loss,
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


def _build_adamw_optimizer(
    params: Any,
    *,
    lr: float,
    eps: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.AdamW:
    """Construct ``AdamW`` with ``fused=True`` enabled automatically on CUDA."""
    kwargs: dict[str, Any] = {
        "lr": float(lr),
        "eps": float(eps),
        "weight_decay": float(weight_decay),
    }
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        try:
            return torch.optim.AdamW(params, **kwargs, fused=True)
        except RuntimeError:
            pass
    return torch.optim.AdamW(params, **kwargs)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    is_fused = bool(optimizer.defaults.get("fused", False))
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if k == "step":
                if isinstance(v, torch.Tensor):
                    target_dev = device if is_fused else torch.device("cpu")
                    state[k] = v.to(device=target_dev, dtype=torch.float32)
                elif is_fused:
                    state[k] = torch.tensor(float(v), dtype=torch.float32, device=device)
            elif isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)


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
        self.obs_layout = obs_layout_from_game_config(cfg.game_config)
        if self.obs_layout.raster_sizes != EgoMultiScaleNet.STEM_RASTER_SIZES:
            raise ValueError(
                f"Observation raster sizes {self.obs_layout.raster_sizes} from GameV3Config "
                f"do not match EgoMultiScaleNet stem input sizes "
                f"{EgoMultiScaleNet.STEM_RASTER_SIZES}"
            )
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
            fast_preprocess=bool(cfg.fast_learner_ops),
        )
        self.raw_model = EgoMultiScaleNet(net_cfg).to(self.device)
        self.raw_model.validate_obs_layout(self.obs_layout)
        if cfg.compile_model and hasattr(torch, "compile"):
            self.raw_model.compile_execution_paths()
        self.model: EgoMultiScaleNet = self.raw_model

        self._trainable_params: tuple[nn.Parameter, ...] = tuple(
            p for p in self.raw_model.parameters() if p.requires_grad
        )
        self._clip_grad_kwargs: dict[str, Any] = (
            {"foreach": True}
            if (
                self.device.type == "cuda"
                and "foreach" in inspect.signature(nn.utils.clip_grad_norm_).parameters
            )
            else {}
        )

        self.optimizer = _build_adamw_optimizer(
            self._trainable_params,
            lr=cfg.lr,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
            device=self.device,
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
            obs_layout=self.obs_layout,
        )
        self._bootstrap_obs = allocate_obs_batch(self.sim, self.learner_index)
        assert_obs_batch_matches_layout(
            self._bootstrap_obs,
            self.obs_layout,
            batch_rows=self.M,
            context="allocate_obs_batch",
        )
        assert_obs_batch_matches_layout(
            self.buffer.obs_view(0),
            self.obs_layout,
            batch_rows=self.M,
            context="buffer.obs_view(0)",
        )
        self._bootstrap_obs_valid = False

        # Persistent counters and episode return accumulators (serialized in checkpoints)
        self.updates_completed: int = 0
        self.world_steps_collected: int = 0
        self.agent_steps_collected: int = 0
        self.learner_steps_collected: int = 0
        self.episodes_completed: int = 0
        self.obs_builds_executed: int = 0
        self.restart_generation: int = 0
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
            life_done = died_m | self.sim.truncated_pending[env_idx, slot_idx]
            ep_ret_sum = ep_ret_sum + torch.where(life_done, self.episode_return_acc, zeros_m).sum()
            ep_ret_count = ep_ret_count + life_done.to(torch.float32).sum()
            self.episode_return_acc = torch.where(
                life_done | (~survived_m),
                zeros_m,
                self.episode_return_acc,
            )

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
        mean_ret_t = torch.where(
            ep_ret_count > 0.0, ep_ret_sum / ep_ret_count.clamp_min(1.0), fallback_ret
        )
        rollout_pair_cpu = torch.stack((ep_ret_count, mean_ret_t)).detach().cpu()
        done_cnt_int = int(rollout_pair_cpu[0].item())
        mean_ep_ret = float(rollout_pair_cpu[1].item())

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
        """Run ``ppo_epochs`` of uniform-minibatch PPO updates with 1 host sync per update.

        Minibatch sizing policy:
          - Every minibatch passed to ``evaluate_actions`` has exact static size
            ``mb_size = self.cfg.resolved_minibatch_size`` so cuDNN never re-autotunes on
            varying tail shapes across updates.
          - When ``n_valid >= mb_size``, the tail remainder ``n_valid % mb_size`` is dropped
            (freshly re-shuffled each epoch via ``self.perm_gen``).
          - When ``0 < n_valid < mb_size`` (tiny debug/test rollouts), indices are padded to
            ``mb_size`` with a ``0.0`` loss mask on padded rows.
        """
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

        mb_size = max(1, int(self.cfg.resolved_minibatch_size))
        pad_small_batch = n_valid < mb_size
        n_usable = mb_size if pad_small_batch else (n_valid // mb_size) * mb_size
        row_mask: torch.Tensor | None = None
        denom_rows_f = float(mb_size)
        if pad_small_batch:
            row_mask = torch.zeros((mb_size,), dtype=torch.float32, device=self.device)
            row_mask[:n_valid] = 1.0
            denom_rows_f = float(n_valid)

        policy_loss_sum_t = torch.zeros((), dtype=torch.float32, device=self.device)
        value_loss_sum_t = torch.zeros((), dtype=torch.float32, device=self.device)
        entropy_sum_t = torch.zeros((), dtype=torch.float32, device=self.device)
        approx_kl_sum_t = torch.zeros((), dtype=torch.float32, device=self.device)
        epoch0_kl_sum_t = torch.zeros((), dtype=torch.float32, device=self.device)
        clip_frac_sum_t = torch.zeros((), dtype=torch.float32, device=self.device)
        grad_norm_sum_t = torch.zeros((), dtype=torch.float32, device=self.device)
        all_finite_t = torch.ones((), dtype=torch.bool, device=self.device)

        epoch0_batches = 0
        num_minibatches = 0
        fast_ops = bool(self.cfg.fast_learner_ops)
        clip_kwargs = self._clip_grad_kwargs if fast_ops else {}

        for epoch in range(int(self.cfg.ppo_epochs)):
            if should_stop is not None and should_stop():
                raise StopRequestedError("Cooperative stop requested during PPO learner epochs")

            perm_cpu = torch.randperm(n_valid, generator=self.perm_gen)
            perm_dev = (
                perm_cpu.pin_memory().to(device=self.device, non_blocking=True)
                if (fast_ops and self.device.type == "cuda")
                else perm_cpu.to(device=self.device)
            )
            shuffled_valid = valid_idx.index_select(0, perm_dev)

            for start in range(0, n_usable, mb_size):
                if should_stop is not None and should_stop():
                    raise StopRequestedError("Cooperative stop requested between PPO minibatches")

                if pad_small_batch:
                    pad_fill = shuffled_valid[:1].expand(mb_size - n_valid)
                    batch_idx = torch.cat((shuffled_valid, pad_fill), dim=0)
                else:
                    batch_idx = shuffled_valid[start : start + mb_size]

                mb = self.buffer.gather_minibatch(
                    batch_idx,
                    normalized_advantages_flat=norm_adv_flat,
                    include_old_values=not fast_ops,
                )

                self.optimizer.zero_grad(set_to_none=True)
                eval_out = self.model.evaluate_actions(
                    mb.raster_s1,
                    mb.raster_s4,
                    mb.raster_s16,
                    mb.scalars,
                    mb.actions,
                    action_mask=mb.action_mask,
                    compute_values=not fast_ops,
                )

                log_ratio = (eval_out.log_probs - mb.old_log_probs).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb.advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                    * mb.advantages
                )
                per_row_p_loss = -torch.minimum(surr1, surr2)
                if row_mask is None:
                    policy_loss = per_row_p_loss.mean()
                    entropy = eval_out.entropy.mean()
                else:
                    policy_loss = (per_row_p_loss * row_mask).sum() / denom_rows_f
                    entropy = (eval_out.entropy * row_mask).sum() / denom_rows_f

                value_loss = symlog_two_hot_loss(
                    eval_out.value_logits,
                    mb.returns,
                    self.raw_model.bin_centers,
                    mask=row_mask,
                    fast=fast_ops,
                )
                total_loss = (
                    policy_loss
                    + float(self.cfg.value_coef) * value_loss
                    - float(self.cfg.entropy_coef) * entropy
                )
                all_finite_t = all_finite_t & torch.isfinite(total_loss).all()

                if self.scaler.is_enabled():
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self._trainable_params,
                        self.cfg.max_grad_norm,
                        **clip_kwargs,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self._trainable_params,
                        self.cfg.max_grad_norm,
                        **clip_kwargs,
                    )
                    self.optimizer.step()

                with torch.no_grad():
                    kl_rows = (ratio - 1.0) - log_ratio
                    clip_rows = ((ratio - 1.0).abs() > self.cfg.clip_ratio).to(torch.float32)
                    if row_mask is None:
                        approx_kl_t = kl_rows.mean()
                        clip_frac_t = clip_rows.mean()
                    else:
                        approx_kl_t = (kl_rows * row_mask).sum() / denom_rows_f
                        clip_frac_t = (clip_rows * row_mask).sum() / denom_rows_f

                    policy_loss_sum_t = policy_loss_sum_t + policy_loss.detach()
                    value_loss_sum_t = value_loss_sum_t + value_loss.detach()
                    entropy_sum_t = entropy_sum_t + entropy.detach()
                    approx_kl_sum_t = approx_kl_sum_t + approx_kl_t
                    if epoch == 0:
                        epoch0_kl_sum_t = epoch0_kl_sum_t + approx_kl_t
                        epoch0_batches += 1
                    clip_frac_sum_t = clip_frac_sum_t + clip_frac_t
                    grad_norm_t = (
                        grad_norm.detach().to(device=self.device, dtype=torch.float32)
                        if isinstance(grad_norm, torch.Tensor)
                        else torch.tensor(float(grad_norm), dtype=torch.float32, device=self.device)
                    )
                    grad_norm_sum_t = grad_norm_sum_t + grad_norm_t
                    num_minibatches += 1

        synced_vec = (
            torch.stack(
                (
                    policy_loss_sum_t,
                    value_loss_sum_t,
                    entropy_sum_t,
                    approx_kl_sum_t,
                    epoch0_kl_sum_t,
                    clip_frac_sum_t,
                    grad_norm_sum_t,
                    all_finite_t.to(torch.float32),
                )
            )
            .detach()
            .cpu()
        )
        vals = synced_vec.tolist()
        if vals[7] < 0.5:
            raise RuntimeError(
                f"Non-finite PPO loss encountered at update {self.updates_completed + 1}: "
                f"policy_sum={vals[0]}, value_sum={vals[1]}, entropy_sum={vals[2]}"
            )

        self.updates_completed += 1
        synchronize_device(self.device)
        learn_sec = time.perf_counter() - t0
        inv = 1.0 / float(max(1, num_minibatches))
        stats = {
            "policy_loss": float(vals[0]) * inv,
            "value_loss": float(vals[1]) * inv,
            "entropy": float(vals[2]) * inv,
            "approx_kl": float(vals[3]) * inv,
            "epoch0_approx_kl": float(vals[4]) / float(max(1, epoch0_batches)),
            "clip_frac": float(vals[5]) * inv,
            "grad_norm": float(vals[6]) * inv,
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

    def run_evaluation(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Run built-in deterministic fixed-seed M-of-N evaluation against scripted opponents."""
        metrics = evaluate_policy(
            self.raw_model,
            self.cfg,
            device=self.device,
            should_stop=should_stop,
        )
        if not bool(metrics.get("aborted", False)):
            self.last_eval_metrics = metrics
        return metrics

    def light_checkpoint_dict(self, *, restart_generation: int | None = None) -> dict[str, Any]:
        """Build LIGHT CPU-resident checkpoint dictionary (network, opt, counters, RNG; no sim)."""
        eff_gen = (
            int(self.restart_generation) if restart_generation is None else int(restart_generation)
        )
        out: dict[str, Any] = {
            "schema": V3_PPO_LIGHT_CHECKPOINT_SCHEMA,
            "namespace": V3_PPO_NAMESPACE,
            "checkpoint_kind": "light",
            "created_unix_ts": float(time.time()),
            "restart_generation": eff_gen,
            "config": self.cfg.to_dict(),
            "obs_layout": self.obs_layout.to_dict(),
            "model_state_dict": _to_cpu_tree(self.raw_model.state_dict()),
            "optimizer_state_dict": _to_cpu_tree(self.optimizer.state_dict()),
            "grad_scaler_state_dict": (
                _to_cpu_tree(self.scaler.state_dict()) if self.scaler.is_enabled() else None
            ),
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
        if self.cfg.recipe_change_history:
            out["recipe_change_history"] = [dict(r) for r in self.cfg.recipe_change_history]
        return out

    def save_light_checkpoint(self, path: Path, *, restart_generation: int | None = None) -> str:
        """Atomically write a LIGHT checkpoint (`no sim state`) with `.sha256` & `.complete`."""
        payload = self.light_checkpoint_dict(restart_generation=restart_generation)
        return write_create_only_artifact(Path(path), lambda tmp: torch.save(payload, tmp))

    def save_full_checkpoint(
        self,
        checkpoints_dir: Path,
        *,
        tag: str | None = None,
        max_shard_bytes: int | None = None,
        light_path: Path | None = None,
        restart_generation: int | None = None,
    ) -> Path:
        """Atomically write a sharded FULL checkpoint set (`light` + whole-world shards + index).

        Every simulator shard file stays strictly ``<= max_shard_bytes`` (`256 MiB` default),
        each with its own `.sha256` and `.complete` sidecar. The index file
        `full_<tag>.index.json` is written strictly LAST and defines completeness of the set.
        """
        eff_gen = (
            int(self.restart_generation) if restart_generation is None else int(restart_generation)
        )
        ckpt_dir = Path(checkpoints_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        prefix = tag if tag is not None else f"u{self.updates_completed:07d}"
        shard_cap = int(
            self.cfg.checkpoint_shard_bytes if max_shard_bytes is None else max_shard_bytes
        )
        if shard_cap < 1024:
            raise ValueError(f"max_shard_bytes must be >= 1024, got {shard_cap}")

        eff_light = (
            Path(light_path) if light_path is not None else (ckpt_dir / f"light_{prefix}.pt")
        )
        if not eff_light.exists():
            light_sha = self.save_light_checkpoint(
                eff_light,
                restart_generation=eff_gen,
            )
        else:
            light_meta = verify_light_checkpoint(eff_light, require_markers=True)
            if (
                int(light_meta["updates_completed"]) != int(self.updates_completed)
                or int(light_meta["world_steps_collected"]) != int(self.world_steps_collected)
                or int(light_meta.get("restart_generation", 0)) != eff_gen
            ):
                raise ValueError(
                    f"Refusing to build FULL checkpoint {prefix!r} from mismatched LIGHT "
                    f"{eff_light.name}: light=(u={light_meta['updates_completed']}, "
                    f"w={light_meta['world_steps_collected']}, "
                    f"r={light_meta.get('restart_generation', 0)}) vs "
                    f"trainer=(u={self.updates_completed}, w={self.world_steps_collected}, "
                    f"r={eff_gen})"
                )
            light_sha = str(light_meta["sha256"])

        raw_sim_sd = self.sim.state_dict()
        clean_streak = int(self.sim._clean_step_streak)
        total_worlds = int(self.sim.E)

        serialized_shards: list[tuple[int, int, bytes]] = []
        w0 = 0
        est_worlds = max(1, total_worlds)
        while w0 < total_worlds:
            w1 = min(total_worlds, w0 + est_worlds)
            while True:
                sub_sd = slice_sim_state_dict(raw_sim_sd, w0, w1)
                packed_sub = pack_sim_state_dict(
                    sub_sd,
                    total_threshold_bytes=0,
                    tensor_min_bytes=_SIM_ZLIB_TENSOR_MIN_BYTES,
                )
                shard_obj = {
                    "schema": V3_PPO_SIM_SHARD_SCHEMA,
                    "namespace": V3_PPO_NAMESPACE,
                    "world_start": w0,
                    "world_end": w1,
                    "total_worlds": total_worlds,
                    "sim_clean_step_streak": clean_streak,
                    "sim_state_dict": packed_sub,
                }
                buf = io.BytesIO()
                torch.save(shard_obj, buf)
                payload_bytes = buf.getvalue()
                if len(payload_bytes) <= shard_cap:
                    break
                if w1 - w0 <= 1:
                    raise ValueError(
                        f"Single world {w0} packed shard ({len(payload_bytes)} B) exceeds "
                        f"max_shard_bytes ({shard_cap} B)"
                    )
                ratio = (float(shard_cap) * 0.90) / float(len(payload_bytes))
                next_span = max(1, min(w1 - w0 - 1, int((w1 - w0) * ratio)))
                w1 = w0 + next_span

            est_worlds = max(1, w1 - w0)
            serialized_shards.append((w0, w1, payload_bytes))
            w0 = w1

        num_shards = len(serialized_shards)
        shard_records: list[dict[str, Any]] = []
        for idx_s, (sw0, sw1, s_bytes) in enumerate(serialized_shards):
            s_name = f"full_{prefix}.sim_shard_{idx_s:04d}_of_{num_shards:04d}.pt"
            s_path = ckpt_dir / s_name
            s_sha = write_create_only_artifact(s_path, s_bytes)
            shard_records.append(
                {
                    "shard_index": idx_s,
                    "filename": s_name,
                    "sha256": s_sha,
                    "size_bytes": len(s_bytes),
                    "world_start": sw0,
                    "world_end": sw1,
                }
            )

        index_path = ckpt_dir / f"full_{prefix}.index.json"
        index_payload = {
            "schema": V3_PPO_FULL_INDEX_SCHEMA,
            "namespace": V3_PPO_NAMESPACE,
            "checkpoint_kind": "full",
            "tag": prefix,
            "created_unix_ts": float(time.time()),
            "restart_generation": eff_gen,
            "updates_completed": int(self.updates_completed),
            "world_steps_collected": int(self.world_steps_collected),
            "agent_steps_collected": int(self.agent_steps_collected),
            "num_envs": total_worlds,
            "obs_layout": self.obs_layout.to_dict(),
            "max_shard_bytes": shard_cap,
            "sim_clean_step_streak": clean_streak,
            "light_checkpoint": {
                "filename": eff_light.name,
                "sha256": light_sha,
                "size_bytes": int(eff_light.stat().st_size),
            },
            "shards": shard_records,
            "total_sim_shard_bytes": sum(int(r["size_bytes"]) for r in shard_records),
        }
        write_create_only_json(index_path, index_payload)
        return index_path

    def _validate_and_apply_light_dict(
        self,
        ckpt: dict[str, Any],
        *,
        restore_rng: bool,
        allow_recipe_change: bool = False,
    ) -> dict[str, Any] | None:
        ckpt_cfg = V3PPOConfig.from_dict(ckpt["config"])
        assert_resume_config_compatible(
            ckpt_cfg,
            self.cfg,
            allow_recipe_change=allow_recipe_change,
            context="Checkpoint config mismatch",
            reference_label="ckpt",
            candidate_label="current",
        )

        if ckpt.get("obs_layout") is not None:
            saved_layout = ObsLayout.from_dict(ckpt["obs_layout"])
            if saved_layout != self.obs_layout:
                raise ValueError(
                    f"Checkpoint obs_layout {saved_layout.to_dict()} does not match "
                    f"trainer obs_layout {self.obs_layout.to_dict()}"
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
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.cfg.lr)
            group["eps"] = float(self.cfg.adam_eps)
            group["weight_decay"] = float(self.cfg.weight_decay)

        scaler_state = ckpt.get("grad_scaler_state_dict")
        if self.scaler.is_enabled() and isinstance(scaler_state, dict):
            self.scaler.load_state_dict(scaler_state)

        counters = ckpt["counters"]
        self.updates_completed = int(counters["updates_completed"])
        self.world_steps_collected = int(counters["world_steps_collected"])
        self.agent_steps_collected = int(counters["agent_steps_collected"])
        self.learner_steps_collected = int(counters["learner_steps_collected"])
        self.episodes_completed = int(counters.get("episodes_completed", 0))
        self.obs_builds_executed = int(counters.get("obs_builds_executed", 0))
        self.last_rollout_mean_ep_return = float(counters.get("last_rollout_mean_ep_return", 0.0))
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

        recipe_change = build_recipe_change_record(
            update=int(self.updates_completed),
            reference=ckpt_cfg,
            candidate=self.cfg,
        )
        if ckpt_cfg.recipe_change_history or recipe_change is not None:
            self.cfg = merge_and_append_recipe_history(
                prior_history=ckpt_cfg.recipe_change_history,
                current_cfg=self.cfg,
                new_record=recipe_change,
            )
        return recipe_change

    def load_checkpoint(
        self,
        path: Path,
        *,
        verify_markers: bool = True,
        restore_rng: bool = True,
        allow_recipe_change: bool = False,
    ) -> dict[str, Any]:
        """Restore trainer from a FULL `.index.json` or LIGHT `.pt` checkpoint."""
        ckpt_path = Path(path)

        if ckpt_path.name.endswith(".index.json"):
            full_meta = verify_full_checkpoint_set(ckpt_path, require_markers=verify_markers)
            light_ckpt = torch.load(full_meta["light_path"], map_location="cpu", weights_only=False)
            recipe_change = self._validate_and_apply_light_dict(
                light_ckpt,
                restore_rng=restore_rng,
                allow_recipe_change=allow_recipe_change,
            )

            shard_objs = [
                torch.load(sp, map_location="cpu", weights_only=False)
                for sp in full_meta["shard_paths"]
            ]
            merged_sim_sd, clean_streak = concat_sim_state_shards(
                shard_objs,
                expected_num_envs=self.sim.E,
            )
            self.sim.load_state_dict(merged_sim_sd)
            self.sim._clean_step_streak = int(clean_streak)
            self.episode_return_acc.copy_(
                light_ckpt["episode_return_acc"].to(device=self.device, dtype=torch.float32)
            )
            self._bootstrap_obs_valid = False
            info_full: dict[str, Any] = {
                "checkpoint_kind": "full",
                "num_shards": int(full_meta["num_shards"]),
                "worlds_reinitialized": False,
                "updates_completed": int(self.updates_completed),
            }
            if recipe_change is not None:
                info_full["recipe_change"] = recipe_change
            return info_full

        if verify_markers:
            verify_completed_artifact(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid checkpoint object at {ckpt_path}: {type(ckpt)!r}")
        if ckpt.get("namespace") != V3_PPO_NAMESPACE:
            raise ValueError(
                f"Foreign checkpoint namespace {ckpt.get('namespace')!r} != {V3_PPO_NAMESPACE!r}"
            )

        schema = ckpt.get("schema")
        if schema == V3_PPO_LIGHT_CHECKPOINT_SCHEMA:
            recipe_change = self._validate_and_apply_light_dict(
                ckpt,
                restore_rng=restore_rng,
                allow_recipe_change=allow_recipe_change,
            )
            fresh_seeds = derive_world_seeds(
                self.cfg.seed,
                self.cfg.num_envs,
                stream_id=2000 + int(self.updates_completed),
            )
            with torch.no_grad():
                self.sim = GameV3Sim(
                    self.cfg.game_config,
                    num_envs=self.cfg.num_envs,
                    device=self.device,
                    seeds=fresh_seeds,
                )
            self.episode_return_acc.zero_()
            self._bootstrap_obs_valid = False
            print(
                f"V3_PPO_LIGHT_RESUME checkpoint={ckpt_path.name} "
                f"updates_completed={self.updates_completed} worlds_reinitialized=True "
                "message='Resumed from LIGHT checkpoint (worlds reinitialised; "
                "episodic return/length statistics warm up from fresh worlds).'",
                flush=True,
            )
            phase_stamp(
                "light_checkpoint_resume",
                "WORLDS_REINITIALIZED",
                checkpoint=ckpt_path.name,
                updates_completed=self.updates_completed,
            )
            info_light: dict[str, Any] = {
                "checkpoint_kind": "light",
                "num_shards": 0,
                "worlds_reinitialized": True,
                "updates_completed": int(self.updates_completed),
            }
            if recipe_change is not None:
                info_light["recipe_change"] = recipe_change
            return info_light

        raise ValueError(f"Invalid checkpoint schema at {ckpt_path}: {schema!r}")
