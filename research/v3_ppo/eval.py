"""Reward function and deterministic fixed-seed M-of-N evaluation for Game V3.1 PPO."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from research.arena_ppo.runtime import normalize_device, synchronize_device
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs
from research.game_v3.rng import rng_u63_np
from research.game_v3.scripted import (
    cardinal6_safe_actions,
    food_seeking_actions,
    random_safe_actions,
)
from research.game_v3.sim import GameV3Sim, StepInfo
from research.v3_ppo.config import OpponentPolicy, V3PPOConfig
from research.v3_ppo.net import EgoMultiScaleNet

DEFAULT_EVAL_OPPONENTS: tuple[OpponentPolicy, ...] = ("food_seeking", "random_safe")


def derive_world_seeds(base_seed: int, num_envs: int, *, stream_id: int = 0) -> list[int]:
    """Derive ``num_envs`` positive signed-64-bit SplitMix64 world seeds deterministically."""
    slots = np.arange(int(num_envs), dtype=np.int64)
    draws = rng_u63_np(int(base_seed), int(stream_id), 99, slots)
    return [int(v) if int(v) != 0 else 1 for v in draws.tolist()]


def scripted_opponent_actions(sim: GameV3Sim, policy_kind: OpponentPolicy) -> torch.Tensor:
    """Compute ``(E, N)`` int64 actions on ``sim.device`` for ``policy_kind``."""
    if policy_kind == "food_seeking":
        return food_seeking_actions(sim)
    if policy_kind == "random_safe":
        return random_safe_actions(sim)
    if policy_kind == "cardinal6_safe":
        return cardinal6_safe_actions(sim)
    raise ValueError(f"Unsupported opponent_policy {policy_kind!r}")


def compute_step_reward(
    info: StepInfo,
    acted: torch.Tensor,
    cfg: V3PPOConfig,
) -> torch.Tensor:
    """Compute per-seat step reward ``(E, N)`` in ``float32`` from ``StepInfo`` and ``cfg``.

    Mirrors ``arena_ppo``'s survival length-integral reward (`reward_alive_length_coef * L_t`)
    and adds v3 ``StepInfo`` dense growth/boost-shed delta (`reward_length_delta_coef *
    (length - prev_length)` on surviving steps), kill credit (`reward_kill_coef * kill_credit`),
    and a dying-snake-only death penalty (`-reward_death_penalty` where ``acted & info.died``).
    Non-acting slots (`~acted`, e.g. dead/respawn-delay ticks) receive exact ``0.0``.
    """
    died = acted & info.died
    survived = acted & (~info.died)

    post_len_f = info.length.to(torch.float32)
    len_delta_f = (info.length - info.prev_length).to(torch.float32)
    kills_f = info.kill_credit.to(torch.float32)

    survive_reward = (
        float(cfg.reward_alive_const)
        + float(cfg.reward_alive_length_coef) * post_len_f
        + float(cfg.reward_length_delta_coef) * len_delta_f
        + float(cfg.reward_kill_coef) * kills_f
    )
    zeros = torch.zeros_like(post_len_f)
    death_reward = torch.full_like(post_len_f, -float(cfg.reward_death_penalty))
    return torch.where(survived, survive_reward, torch.where(died, death_reward, zeros))


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Save and restore Python, NumPy, Torch CPU, and Torch CUDA RNG states."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def _act_learners_chunked(
    model: EgoMultiScaleNet,
    obs_s1: torch.Tensor,
    obs_s4: torch.Tensor,
    obs_s16: torch.Tensor,
    obs_scalars: torch.Tensor,
    obs_mask: torch.Tensor,
    *,
    chunk_size: int,
    greedy: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    m_total = int(obs_s1.shape[0])
    step = max(1, min(m_total, int(chunk_size)))
    if step == m_total:
        return model.act(
            obs_s1,
            obs_s4,
            obs_s16,
            obs_scalars,
            action_mask=obs_mask,
            greedy=greedy,
            generator=generator,
        ).actions

    out = torch.empty((m_total,), dtype=torch.long, device=obs_s1.device)
    for start in range(0, m_total, step):
        end = min(m_total, start + step)
        out[start:end] = model.act(
            obs_s1[start:end],
            obs_s4[start:end],
            obs_s16[start:end],
            obs_scalars[start:end],
            action_mask=obs_mask[start:end],
            greedy=greedy,
            generator=generator,
        ).actions
    return out


def _summarize_episodes(
    *,
    final_lengths: list[float],
    episode_returns: list[float],
    total_acted_steps: int,
    total_deaths: int,
    total_kills: int,
    alive_len_sum: float,
    alive_ticks: int,
) -> dict[str, Any]:
    lengths_arr = np.asarray(final_lengths, dtype=np.float64)
    returns_arr = np.asarray(episode_returns, dtype=np.float64)
    mean_final_len = float(lengths_arr.mean()) if lengths_arr.size > 0 else 0.0
    p95_final_len = float(np.percentile(lengths_arr, 95)) if lengths_arr.size > 0 else 0.0
    mean_ep_ret = float(returns_arr.mean()) if returns_arr.size > 0 else 0.0
    denom_1k = max(1.0, float(total_acted_steps)) / 1000.0
    return {
        "episodes": int(lengths_arr.size),
        "learner_agent_steps": int(total_acted_steps),
        "mean_final_length": round(mean_final_len, 4),
        "p95_final_length": round(p95_final_len, 4),
        "mean_alive_length": round(alive_len_sum / float(max(1, alive_ticks)), 4),
        "mean_episode_return": round(mean_ep_ret, 6),
        "total_deaths": int(total_deaths),
        "deaths_per_1k_agent_steps": round(float(total_deaths) / denom_1k, 4),
        "kills": int(total_kills),
        "kills_per_1k_agent_steps": round(float(total_kills) / denom_1k, 4),
    }


@torch.no_grad()
def _run_eval_matchup(
    *,
    cfg: V3PPOConfig,
    device: torch.device,
    opponent: OpponentPolicy,
    suite_index: int,
    eval_envs: int,
    eval_ticks: int,
    learner_slots: int,
    eval_seed: int,
    model: EgoMultiScaleNet | None = None,
    baseline_learner_policy: OpponentPolicy | None = None,
    greedy: bool = False,
) -> tuple[dict[str, Any], list[float], list[float]]:
    if (model is None) == (baseline_learner_policy is None):
        raise ValueError("Provide exactly one of `model` or `baseline_learner_policy`")

    world_seeds = derive_world_seeds(eval_seed, eval_envs, stream_id=100 + suite_index)
    sim = GameV3Sim(
        cfg.game_config,
        num_envs=eval_envs,
        device=device,
        seeds=world_seeds,
    )

    n_snakes = sim.N
    k_learners = min(max(1, int(learner_slots)), n_snakes)
    learner_mask = torch.zeros((sim.E, n_snakes), dtype=torch.bool, device=device)
    learner_mask[:, :k_learners] = True
    learner_idx = LearnerIndex(sim, learner_mask)
    obs_buf = allocate_obs_batch(sim, learner_idx) if model is not None else None

    sample_gen = torch.Generator(device="cpu")
    sample_gen.manual_seed(int(rng_u63_np(int(eval_seed), int(suite_index), 98, 0)))

    ep_ret_acc = torch.zeros((sim.E, k_learners), dtype=torch.float32, device=device)
    zeros_ek = torch.zeros((sim.E, k_learners), dtype=torch.float32, device=device)

    final_lengths: list[float] = []
    episode_returns: list[float] = []
    total_acted_steps = 0
    total_deaths = 0
    total_kills = 0
    alive_len_sum = 0.0
    alive_ticks = 0

    chunk_learners = cfg.resolved_actor_chunk_learners

    for t in range(int(eval_ticks)):
        acted_es = sim.alive & (~sim.truncated_pending)
        if model is not None:
            assert obs_buf is not None
            build_obs(sim, learner_index=learner_idx, out=obs_buf)
            learner_act_1d = _act_learners_chunked(
                model,
                obs_buf.raster_s1,
                obs_buf.raster_s4,
                obs_buf.raster_s16,
                obs_buf.scalars,
                obs_buf.action_mask,
                chunk_size=chunk_learners,
                greedy=greedy,
                generator=sample_gen,
            )
            learner_act_ek = learner_act_1d.view(sim.E, k_learners)
        else:
            assert baseline_learner_policy is not None
            learner_act_ek = scripted_opponent_actions(sim, baseline_learner_policy)[:, :k_learners]

        if k_learners < n_snakes:
            world_actions = scripted_opponent_actions(sim, opponent)
            world_actions[:, :k_learners] = learner_act_ek
        else:
            world_actions = learner_act_ek

        is_last_tick = t == (int(eval_ticks) - 1)
        info = sim.step(world_actions, truncate=is_last_tick)
        step_rew = compute_step_reward(info, acted_es, cfg)

        l_acted = acted_es[:, :k_learners]
        l_died = l_acted & info.died[:, :k_learners]
        l_survived = l_acted & (~info.died[:, :k_learners])
        l_trunc = l_survived & info.truncated[:, :k_learners]
        l_done = l_died | l_trunc

        ep_ret_acc = torch.where(l_acted, ep_ret_acc + step_rew[:, :k_learners], ep_ret_acc)

        if bool(l_done.any()):
            ep_final_len_tensor = torch.where(
                l_died,
                info.prev_length[:, :k_learners],
                info.length[:, :k_learners],
            )
            done_lens = ep_final_len_tensor[l_done].detach().cpu().tolist()
            done_rets = ep_ret_acc[l_done].detach().cpu().tolist()
            final_lengths.extend(float(x) for x in done_lens)
            episode_returns.extend(float(x) for x in done_rets)

        ep_ret_acc = torch.where(l_done | (~l_survived), zeros_ek, ep_ret_acc)

        total_acted_steps += int(l_acted.sum().item())
        total_deaths += int(l_died.sum().item())
        total_kills += int(
            torch.where(
                l_survived,
                info.kill_credit[:, :k_learners],
                torch.zeros_like(info.kill_credit[:, :k_learners]),
            )
            .sum()
            .item()
        )
        alive_len_sum += float(
            torch.where(l_survived, info.length[:, :k_learners].float(), zeros_ek).sum().item()
        )
        alive_ticks += int(l_survived.sum().item())

    summary = _summarize_episodes(
        final_lengths=final_lengths,
        episode_returns=episode_returns,
        total_acted_steps=total_acted_steps,
        total_deaths=total_deaths,
        total_kills=total_kills,
        alive_len_sum=alive_len_sum,
        alive_ticks=alive_ticks,
    )
    summary["opponent"] = opponent
    return summary, final_lengths, episode_returns


@torch.no_grad()
def evaluate_policy(
    model: EgoMultiScaleNet,
    cfg: V3PPOConfig,
    *,
    device: str | torch.device = "cpu",
    opponents: Sequence[OpponentPolicy] = DEFAULT_EVAL_OPPONENTS,
    eval_envs: int | None = None,
    eval_ticks: int | None = None,
    eval_learner_slots: int | None = None,
    eval_seed: int | None = None,
    greedy: bool | None = None,
) -> dict[str, Any]:
    """Run deterministic fixed-seed M-of-N evaluation against scripted opponent suites.

    Saves and restores global RNG states so periodic evaluation during training has zero
    effect on rollout/update trajectories.
    """
    resolved_dev = normalize_device(device)
    eff_envs = int(cfg.eval_envs if eval_envs is None else eval_envs)
    eff_ticks = int(cfg.eval_ticks if eval_ticks is None else eval_ticks)
    eff_slots = int(
        cfg.resolved_eval_learner_slots if eval_learner_slots is None else eval_learner_slots
    )
    eff_seed = int(cfg.eval_seed if eval_seed is None else eval_seed)
    eff_greedy = bool(cfg.eval_greedy if greedy is None else greedy)

    was_training = model.training
    model.eval()
    t0 = time.perf_counter()

    all_lengths: list[float] = []
    all_returns: list[float] = []
    tot_steps = 0
    tot_deaths = 0
    tot_kills = 0
    tot_alive_len_sum = 0.0
    tot_alive_ticks = 0
    by_opponent: dict[str, dict[str, Any]] = {}

    try:
        with preserve_rng_state():
            for idx_opp, opp in enumerate(opponents):
                suite_res, lens, rets = _run_eval_matchup(
                    cfg=cfg,
                    device=resolved_dev,
                    opponent=opp,
                    suite_index=idx_opp,
                    eval_envs=eff_envs,
                    eval_ticks=eff_ticks,
                    learner_slots=eff_slots,
                    eval_seed=eff_seed,
                    model=model,
                    greedy=eff_greedy,
                )
                by_opponent[f"vs_{opp}"] = suite_res
                all_lengths.extend(lens)
                all_returns.extend(rets)
                tot_steps += int(suite_res["learner_agent_steps"])
                tot_deaths += int(suite_res["total_deaths"])
                tot_kills += int(suite_res["kills"])
                # Reconstruct exact alive length sum from exact counts
                # (kept exact below via suite_res mean * ticks for display; recomputed cleanly)
                tot_alive_len_sum += float(suite_res["mean_alive_length"]) * max(
                    1, int(suite_res["learner_agent_steps"]) - int(suite_res["total_deaths"])
                )
                tot_alive_ticks += max(
                    0, int(suite_res["learner_agent_steps"]) - int(suite_res["total_deaths"])
                )
    finally:
        if was_training:
            model.train()

    synchronize_device(resolved_dev)
    elapsed = time.perf_counter() - t0
    combined = _summarize_episodes(
        final_lengths=all_lengths,
        episode_returns=all_returns,
        total_acted_steps=tot_steps,
        total_deaths=tot_deaths,
        total_kills=tot_kills,
        alive_len_sum=tot_alive_len_sum,
        alive_ticks=max(1, tot_alive_ticks),
    )
    return {
        "eval_seed": eff_seed,
        "eval_envs": eff_envs,
        "eval_ticks": eff_ticks,
        "eval_learner_slots": min(max(1, eff_slots), cfg.num_snakes),
        "num_snakes": cfg.num_snakes,
        "greedy": eff_greedy,
        "wall_seconds": round(elapsed, 4),
        "combined": combined,
        "matchups": by_opponent,
    }


@torch.no_grad()
def evaluate_scripted_baseline(
    baseline_policy: OpponentPolicy,
    cfg: V3PPOConfig,
    *,
    device: str | torch.device = "cpu",
    opponents: Sequence[OpponentPolicy] = DEFAULT_EVAL_OPPONENTS,
    eval_envs: int | None = None,
    eval_ticks: int | None = None,
    eval_learner_slots: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a scripted baseline (`random_safe`, `food_seeking`, etc.) on identical worlds."""
    resolved_dev = normalize_device(device)
    eff_envs = int(cfg.eval_envs if eval_envs is None else eval_envs)
    eff_ticks = int(cfg.eval_ticks if eval_ticks is None else eval_ticks)
    eff_slots = int(
        cfg.resolved_eval_learner_slots if eval_learner_slots is None else eval_learner_slots
    )
    eff_seed = int(cfg.eval_seed if eval_seed is None else eval_seed)

    t0 = time.perf_counter()
    all_lengths: list[float] = []
    all_returns: list[float] = []
    tot_steps = 0
    tot_deaths = 0
    tot_kills = 0
    tot_alive_len_sum = 0.0
    tot_alive_ticks = 0
    by_opponent: dict[str, dict[str, Any]] = {}

    with preserve_rng_state():
        for idx_opp, opp in enumerate(opponents):
            suite_res, lens, rets = _run_eval_matchup(
                cfg=cfg,
                device=resolved_dev,
                opponent=opp,
                suite_index=idx_opp,
                eval_envs=eff_envs,
                eval_ticks=eff_ticks,
                learner_slots=eff_slots,
                eval_seed=eff_seed,
                baseline_learner_policy=baseline_policy,
            )
            by_opponent[f"vs_{opp}"] = suite_res
            all_lengths.extend(lens)
            all_returns.extend(rets)
            tot_steps += int(suite_res["learner_agent_steps"])
            tot_deaths += int(suite_res["total_deaths"])
            tot_kills += int(suite_res["kills"])
            tot_alive_len_sum += float(suite_res["mean_alive_length"]) * max(
                1, int(suite_res["learner_agent_steps"]) - int(suite_res["total_deaths"])
            )
            tot_alive_ticks += max(
                0, int(suite_res["learner_agent_steps"]) - int(suite_res["total_deaths"])
            )

    synchronize_device(resolved_dev)
    elapsed = time.perf_counter() - t0
    combined = _summarize_episodes(
        final_lengths=all_lengths,
        episode_returns=all_returns,
        total_acted_steps=tot_steps,
        total_deaths=tot_deaths,
        total_kills=tot_kills,
        alive_len_sum=tot_alive_len_sum,
        alive_ticks=max(1, tot_alive_ticks),
    )
    return {
        "baseline_policy": baseline_policy,
        "eval_seed": eff_seed,
        "eval_envs": eff_envs,
        "eval_ticks": eff_ticks,
        "eval_learner_slots": min(max(1, eff_slots), cfg.num_snakes),
        "num_snakes": cfg.num_snakes,
        "wall_seconds": round(elapsed, 4),
        "combined": combined,
        "matchups": by_opponent,
    }
