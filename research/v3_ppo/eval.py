"""Reward function and deterministic burn-in + window M-of-N evaluation for Game V3.1 PPO."""

from __future__ import annotations

import gc
import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
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
_EVAL_EVENT_FLUSH_TICKS: int = 256


@dataclass
class _EvalMatchupRaw:
    """Exact un-rounded accumulators for one evaluation matchup suite."""

    summary: dict[str, Any]
    death_lengths: list[float]
    lifetime_ticks: list[float]
    completed_returns: list[float]
    censored_lengths: list[float]
    acted_steps: int
    mass_gained: int
    net_len_delta: int
    deaths: int
    kills: int
    alive_len_sum: float
    alive_ticks: int


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
    (length - prev_length)` on surviving steps), kill credit (`reward_kill_coef * kill_credit`
    on surviving steps), and a dying-snake-only death penalty (`-reward_death_penalty` where
    ``acted & info.died``).

    Note on simultaneous kills + death: ``GameV3Sim`` awards ``StepInfo.kill_credit`` even when
    the killer dies on the same tick (game rule). In ``compute_step_reward``, dying ticks pay
    strictly ``-reward_death_penalty`` with no positive kill bonus so mutual head-on suicide
    collisions are never subsidized; evaluation telemetry (`eval.py`) counts all ``acted``
    kills including simultaneous deaths. Non-acting slots (`~acted`) receive exact ``0.0``.
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
    death_lengths: list[float],
    lifetime_ticks: list[float],
    completed_returns: list[float],
    censored_lengths: list[float],
    total_acted_steps: int,
    total_mass_gained: int,
    total_net_len_delta: int,
    total_deaths: int,
    total_kills: int,
    alive_len_sum: float,
    alive_ticks: int,
) -> dict[str, Any]:
    """Summarize completed lives, window-censored lives, and survivorship-free rate metrics."""
    d_len_arr = np.asarray(death_lengths, dtype=np.float64)
    life_arr = np.asarray(lifetime_ticks, dtype=np.float64)
    ret_arr = np.asarray(completed_returns, dtype=np.float64)
    cens_arr = np.asarray(censored_lengths, dtype=np.float64)

    mean_death_len = float(d_len_arr.mean()) if d_len_arr.size > 0 else 0.0
    median_death_len = float(np.median(d_len_arr)) if d_len_arr.size > 0 else 0.0
    p95_death_len = float(np.percentile(d_len_arr, 95)) if d_len_arr.size > 0 else 0.0

    mean_life = float(life_arr.mean()) if life_arr.size > 0 else 0.0
    median_life = float(np.median(life_arr)) if life_arr.size > 0 else 0.0
    p95_life = float(np.percentile(life_arr, 95)) if life_arr.size > 0 else 0.0

    mean_ep_ret = float(ret_arr.mean()) if ret_arr.size > 0 else 0.0
    mean_cens_len = float(cens_arr.mean()) if cens_arr.size > 0 else 0.0

    denom_1k = max(1.0, float(total_acted_steps)) / 1000.0
    time_avg_len = round(float(alive_len_sum) / float(max(1, alive_ticks)), 4)

    return {
        "completed_lives": int(d_len_arr.size),
        "censored_lives": int(cens_arr.size),
        "mean_censored_length": round(mean_cens_len, 4),
        "mean_death_length": round(mean_death_len, 4),
        "median_death_length": round(median_death_len, 4),
        "p95_death_length": round(p95_death_len, 4),
        "mean_lifetime_ticks": round(mean_life, 4),
        "median_lifetime_ticks": round(median_life, 4),
        "p95_lifetime_ticks": round(p95_life, 4),
        "mean_completed_episode_return": round(mean_ep_ret, 6),
        "learner_agent_steps": int(total_acted_steps),
        "mass_gained": int(total_mass_gained),
        "mass_gained_per_1k_agent_steps": round(float(total_mass_gained) / denom_1k, 4),
        "net_length_delta_per_1k_agent_steps": round(float(total_net_len_delta) / denom_1k, 4),
        "total_deaths": int(total_deaths),
        "deaths_per_1k_agent_steps": round(float(total_deaths) / denom_1k, 4),
        "kills": int(total_kills),
        "kills_per_1k_agent_steps": round(float(total_kills) / denom_1k, 4),
        "time_averaged_length": time_avg_len,
        "mean_alive_length": time_avg_len,
    }


@torch.no_grad()
def _run_eval_matchup(
    *,
    cfg: V3PPOConfig,
    device: torch.device,
    opponent: OpponentPolicy,
    suite_index: int,
    eval_envs: int,
    eval_burnin_ticks: int,
    eval_ticks: int,
    learner_slots: int,
    eval_seed: int,
    model: EgoMultiScaleNet | None = None,
    baseline_learner_policy: OpponentPolicy | None = None,
    greedy: bool = False,
) -> _EvalMatchupRaw:
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
    life_ticks_acc = torch.zeros((sim.E, k_learners), dtype=torch.int32, device=device)
    zeros_f32 = torch.zeros((sim.E, k_learners), dtype=torch.float32, device=device)
    zeros_i32 = torch.zeros((sim.E, k_learners), dtype=torch.int32, device=device)

    chunk_learners = cfg.resolved_actor_chunk_learners
    burnin = max(0, int(eval_burnin_ticks))
    window = max(1, int(eval_ticks))

    def _step_matchup() -> tuple[torch.Tensor, StepInfo, torch.Tensor]:
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

        info_step = sim.step(world_actions, truncate=False)
        step_rew_es = compute_step_reward(info_step, acted_es, cfg)
        return acted_es, info_step, step_rew_es

    # 1. Burn-in phase: mature food/corpses/lengths; track ongoing life age & return from birth
    for _ in range(burnin):
        acted_es, info, step_rew = _step_matchup()
        l_acted = acted_es[:, :k_learners]
        l_died = l_acted & info.died[:, :k_learners]
        l_survived = l_acted & (~info.died[:, :k_learners])

        ep_ret_acc = torch.where(l_acted, ep_ret_acc + step_rew[:, :k_learners], ep_ret_acc)
        life_ticks_acc = torch.where(l_acted, life_ticks_acc + 1, life_ticks_acc)

        ep_ret_acc = torch.where(l_died | (~l_survived), zeros_f32, ep_ret_acc)
        life_ticks_acc = torch.where(l_died | (~l_survived), zeros_i32, life_ticks_acc)

    # 2. Measurement window: zero per-tick host syncs via on-device accumulators + 256-tick ring
    acted_steps_dev = torch.zeros((), dtype=torch.int64, device=device)
    mass_gained_dev = torch.zeros((), dtype=torch.int64, device=device)
    net_len_delta_dev = torch.zeros((), dtype=torch.int64, device=device)
    deaths_dev = torch.zeros((), dtype=torch.int64, device=device)
    kills_dev = torch.zeros((), dtype=torch.int64, device=device)
    alive_len_sum_dev = torch.zeros((), dtype=torch.int64, device=device)
    alive_ticks_dev = torch.zeros((), dtype=torch.int64, device=device)

    buf_cap = min(window, _EVAL_EVENT_FLUSH_TICKS)
    buf_died = torch.zeros((buf_cap, sim.E, k_learners), dtype=torch.bool, device=device)
    buf_death_len = torch.zeros((buf_cap, sim.E, k_learners), dtype=torch.int32, device=device)
    buf_life_ticks = torch.zeros((buf_cap, sim.E, k_learners), dtype=torch.int32, device=device)
    buf_ep_ret = torch.zeros((buf_cap, sim.E, k_learners), dtype=torch.float32, device=device)

    death_lengths: list[float] = []
    lifetime_ticks: list[float] = []
    completed_returns: list[float] = []

    def _flush_event_ring(valid_rows: int) -> None:
        if valid_rows <= 0:
            return
        d_mask = buf_died[:valid_rows]
        d_lens_cpu = buf_death_len[:valid_rows][d_mask].detach().cpu().tolist()
        if not d_lens_cpu:
            return
        d_life_cpu = buf_life_ticks[:valid_rows][d_mask].detach().cpu().tolist()
        d_ret_cpu = buf_ep_ret[:valid_rows][d_mask].detach().cpu().tolist()
        death_lengths.extend(float(x) for x in d_lens_cpu)
        lifetime_ticks.extend(float(x) for x in d_life_cpu)
        completed_returns.extend(float(x) for x in d_ret_cpu)

    slot_in_buf = 0
    for _ in range(window):
        acted_es, info, step_rew = _step_matchup()
        l_acted = acted_es[:, :k_learners]
        l_died = l_acted & info.died[:, :k_learners]
        l_survived = l_acted & (~info.died[:, :k_learners])

        ep_ret_post = torch.where(l_acted, ep_ret_acc + step_rew[:, :k_learners], ep_ret_acc)
        life_ticks_post = torch.where(l_acted, life_ticks_acc + 1, life_ticks_acc)

        buf_died[slot_in_buf].copy_(l_died)
        buf_death_len[slot_in_buf].copy_(info.prev_length[:, :k_learners])
        buf_life_ticks[slot_in_buf].copy_(life_ticks_post)
        buf_ep_ret[slot_in_buf].copy_(ep_ret_post)
        slot_in_buf += 1
        if slot_in_buf == buf_cap:
            _flush_event_ring(slot_in_buf)
            slot_in_buf = 0

        ep_ret_acc = torch.where(l_died | (~l_survived), zeros_f32, ep_ret_post)
        life_ticks_acc = torch.where(l_died | (~l_survived), zeros_i32, life_ticks_post)

        acted_steps_dev += l_acted.to(torch.int64).sum()
        deaths_dev += l_died.to(torch.int64).sum()
        # Count all `l_acted` kills (including simultaneous kill + death on the same tick).
        kills_dev += torch.where(l_acted, info.kill_credit[:, :k_learners], zeros_i32).sum(
            dtype=torch.int64
        )
        mass_gained_dev += torch.where(l_survived, info.mass_eaten[:, :k_learners], zeros_i32).sum(
            dtype=torch.int64
        )
        len_delta_ek = info.length[:, :k_learners] - info.prev_length[:, :k_learners]
        net_len_delta_dev += torch.where(l_survived, len_delta_ek, zeros_i32).sum(dtype=torch.int64)
        alive_len_sum_dev += torch.where(l_survived, info.length[:, :k_learners], zeros_i32).sum(
            dtype=torch.int64
        )
        alive_ticks_dev += l_survived.to(torch.int64).sum()

    if slot_in_buf > 0:
        _flush_event_ring(slot_in_buf)

    # Censored lives = learner snakes still alive at window end (reported separately, never mixed!).
    alive_end = sim.alive[:, :k_learners]
    censored_lengths = [
        float(x) for x in sim.length[:, :k_learners][alive_end].detach().cpu().tolist()
    ]

    scalars_cpu = torch.stack(
        (
            acted_steps_dev,
            mass_gained_dev,
            net_len_delta_dev,
            deaths_dev,
            kills_dev,
            alive_ticks_dev,
            alive_len_sum_dev,
        )
    ).cpu()
    total_acted_steps = int(scalars_cpu[0].item())
    total_mass_gained = int(scalars_cpu[1].item())
    total_net_len_delta = int(scalars_cpu[2].item())
    total_deaths = int(scalars_cpu[3].item())
    total_kills = int(scalars_cpu[4].item())
    alive_ticks = int(scalars_cpu[5].item())
    alive_len_sum = float(scalars_cpu[6].item())

    summary = _summarize_episodes(
        death_lengths=death_lengths,
        lifetime_ticks=lifetime_ticks,
        completed_returns=completed_returns,
        censored_lengths=censored_lengths,
        total_acted_steps=total_acted_steps,
        total_mass_gained=total_mass_gained,
        total_net_len_delta=total_net_len_delta,
        total_deaths=total_deaths,
        total_kills=total_kills,
        alive_len_sum=alive_len_sum,
        alive_ticks=alive_ticks,
    )
    summary["opponent"] = opponent
    return _EvalMatchupRaw(
        summary=summary,
        death_lengths=death_lengths,
        lifetime_ticks=lifetime_ticks,
        completed_returns=completed_returns,
        censored_lengths=censored_lengths,
        acted_steps=total_acted_steps,
        mass_gained=total_mass_gained,
        net_len_delta=total_net_len_delta,
        deaths=total_deaths,
        kills=total_kills,
        alive_len_sum=alive_len_sum,
        alive_ticks=alive_ticks,
    )


def _combine_matchups(raw_suites: Sequence[_EvalMatchupRaw]) -> dict[str, Any]:
    all_death_lengths: list[float] = []
    all_lifetimes: list[float] = []
    all_returns: list[float] = []
    all_censored: list[float] = []
    tot_steps = 0
    tot_mass = 0
    tot_net_delta = 0
    tot_deaths = 0
    tot_kills = 0
    tot_alive_len_sum = 0.0
    tot_alive_ticks = 0

    for item in raw_suites:
        all_death_lengths.extend(item.death_lengths)
        all_lifetimes.extend(item.lifetime_ticks)
        all_returns.extend(item.completed_returns)
        all_censored.extend(item.censored_lengths)
        tot_steps += item.acted_steps
        tot_mass += item.mass_gained
        tot_net_delta += item.net_len_delta
        tot_deaths += item.deaths
        tot_kills += item.kills
        tot_alive_len_sum += item.alive_len_sum
        tot_alive_ticks += item.alive_ticks

    return _summarize_episodes(
        death_lengths=all_death_lengths,
        lifetime_ticks=all_lifetimes,
        completed_returns=all_returns,
        censored_lengths=all_censored,
        total_acted_steps=tot_steps,
        total_mass_gained=tot_mass,
        total_net_len_delta=tot_net_delta,
        total_deaths=tot_deaths,
        total_kills=tot_kills,
        alive_len_sum=tot_alive_len_sum,
        alive_ticks=tot_alive_ticks,
    )


@torch.no_grad()
def evaluate_policy(
    model: EgoMultiScaleNet,
    cfg: V3PPOConfig,
    *,
    device: str | torch.device = "cpu",
    opponents: Sequence[OpponentPolicy] = DEFAULT_EVAL_OPPONENTS,
    eval_envs: int | None = None,
    eval_burnin_ticks: int | None = None,
    eval_ticks: int | None = None,
    eval_learner_slots: int | None = None,
    eval_seed: int | None = None,
    greedy: bool | None = None,
) -> dict[str, Any]:
    """Run deterministic burn-in + window M-of-N evaluation against scripted opponent suites.

    Saves and restores global RNG states and uses dedicated evaluation ``GameV3Sim`` instances
    so periodic evaluation during training has zero effect on rollout/update trajectories.
    """
    resolved_dev = normalize_device(device)
    eff_envs = int(cfg.eval_envs if eval_envs is None else eval_envs)
    eff_burnin = int(cfg.eval_burnin_ticks if eval_burnin_ticks is None else eval_burnin_ticks)
    eff_ticks = int(cfg.eval_ticks if eval_ticks is None else eval_ticks)
    eff_slots = int(
        cfg.resolved_eval_learner_slots if eval_learner_slots is None else eval_learner_slots
    )
    eff_seed = int(cfg.eval_seed if eval_seed is None else eval_seed)
    eff_greedy = bool(cfg.eval_greedy if greedy is None else greedy)

    was_training = model.training
    model.eval()
    t0 = time.perf_counter()

    raw_suites: list[_EvalMatchupRaw] = []
    by_opponent: dict[str, dict[str, Any]] = {}

    try:
        with preserve_rng_state():
            for idx_opp, opp in enumerate(opponents):
                raw_res = _run_eval_matchup(
                    cfg=cfg,
                    device=resolved_dev,
                    opponent=opp,
                    suite_index=idx_opp,
                    eval_envs=eff_envs,
                    eval_burnin_ticks=eff_burnin,
                    eval_ticks=eff_ticks,
                    learner_slots=eff_slots,
                    eval_seed=eff_seed,
                    model=model,
                    greedy=eff_greedy,
                )
                by_opponent[f"vs_{opp}"] = raw_res.summary
                raw_suites.append(raw_res)
                gc.collect()
    finally:
        if was_training:
            model.train()

    synchronize_device(resolved_dev)
    elapsed = time.perf_counter() - t0
    combined = _combine_matchups(raw_suites)
    return {
        "eval_seed": eff_seed,
        "eval_envs": eff_envs,
        "eval_burnin_ticks": eff_burnin,
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
    eval_burnin_ticks: int | None = None,
    eval_ticks: int | None = None,
    eval_learner_slots: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a scripted baseline (`random_safe`, `food_seeking`, etc.) on identical worlds."""
    resolved_dev = normalize_device(device)
    eff_envs = int(cfg.eval_envs if eval_envs is None else eval_envs)
    eff_burnin = int(cfg.eval_burnin_ticks if eval_burnin_ticks is None else eval_burnin_ticks)
    eff_ticks = int(cfg.eval_ticks if eval_ticks is None else eval_ticks)
    eff_slots = int(
        cfg.resolved_eval_learner_slots if eval_learner_slots is None else eval_learner_slots
    )
    eff_seed = int(cfg.eval_seed if eval_seed is None else eval_seed)

    t0 = time.perf_counter()
    raw_suites: list[_EvalMatchupRaw] = []
    by_opponent: dict[str, dict[str, Any]] = {}

    with preserve_rng_state():
        for idx_opp, opp in enumerate(opponents):
            raw_res = _run_eval_matchup(
                cfg=cfg,
                device=resolved_dev,
                opponent=opp,
                suite_index=idx_opp,
                eval_envs=eff_envs,
                eval_burnin_ticks=eff_burnin,
                eval_ticks=eff_ticks,
                learner_slots=eff_slots,
                eval_seed=eff_seed,
                baseline_learner_policy=baseline_policy,
            )
            by_opponent[f"vs_{opp}"] = raw_res.summary
            raw_suites.append(raw_res)
            gc.collect()

    synchronize_device(resolved_dev)
    elapsed = time.perf_counter() - t0
    combined = _combine_matchups(raw_suites)
    return {
        "baseline_policy": baseline_policy,
        "eval_seed": eff_seed,
        "eval_envs": eff_envs,
        "eval_burnin_ticks": eff_burnin,
        "eval_ticks": eff_ticks,
        "eval_learner_slots": min(max(1, eff_slots), cfg.num_snakes),
        "num_snakes": cfg.num_snakes,
        "wall_seconds": round(elapsed, 4),
        "combined": combined,
        "matchups": by_opponent,
    }
