"""PQN Q(lambda) trainer for the dual-scale raster network (blueprint P3 §3).

PQN (Parallelized Q-Network) is synchronous Q(lambda) on a batch of vectorized
environments — **no replay buffer, no target network, no PER**. Each update:

1. Roll out ``T`` steps on :class:`~src.simd_env.batch_sim.BatchSim` (E envs x S
   snakes) with pool self-play (:mod:`src.training.pqn_selfplay`), collecting
   per-slot ``(obs, action, reward, mask, done, trapped)`` and the hero Q-values.
2. After the rollout, boot the network once more on the final observation to get
   the bootstrap Q(s') for the truncation case.
3. Compute the per-agent Q(lambda) return **backward** over each hero slot's
   rollout with blueprint termination handling:
     - **DEATH** (``done``): return is the reward alone (no bootstrap).
     - **TRAPPED** non-terminal (no valid next action, but not dead this step):
       bootstrap to the DEATH VALUE (a config constant), not 0.
     - **TRUNCATION** (rollout edge, or a mid-rollout reset): bootstrap from the
       masked-max Q(s') over valid next actions.
     - Interior alive steps: standard Q(lambda) mixing
       ``G_t = r_t + gamma * ((1-lambda) * max_a' Q(s',a') + lambda * G_{t+1})``.
4. Take several minibatch SGD steps (Huber loss on ``Q(s,a) - G``), grad-norm
   clip 10, **no TD-target clipping**. Optional horizontal-flip augmentation.

Only HERO-slot transitions train (frozen-opponent slots are ignored). The
network sees ε-greedy behavior with the blueprint ladder (1.0 -> 0.02 over the
first ~50M agent-steps, scaled down for smoke).

Telemetry (:class:`PQNTelemetry`) and tripwires (NaN/inf, max|Q| alarm,
action-collapse) are emitted per update so an outer loop can halt-and-flag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.reward_events import DEATH_REWARD
from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V2, RASTER31V2_SHAPES
from src.model.raster_network import (
    SCALARS_DIM,
    STRATEGIC_SHAPE,
    TACTICAL_SHAPE,
    RasterDuelingNetwork,
    raster_tensors_from_obs,
)
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.featurizer import build_observations, obs_inputs_from_batch_sim
from src.simd_env.gpu_featurizer import build_observations_gpu, obs_inputs_to_torch
from src.training.pqn_selfplay import (
    HERO_POLICY_ID,
    OpponentPool,
    assign_policy_ids,
    batched_act,
)

__all__ = ["PQNConfig", "PQNTrainer", "PQNTelemetry", "TripwireError", "flip_augment"]


# Lateral scalar indices in the 26-D scalar vector that flip SIGN under a
# horizontal (left<->right) mirror of the world. Derived from featurizer §_build_scalars:
#   9  wall dist RIGHT  <-> 11 wall dist LEFT   (swap, handled separately)
#   12 nearest-food ego dx (lateral)            -> negate
#   15 enemy1 ego dx (lateral)                  -> negate
#   19 enemy2 ego dx (lateral)                  -> negate
# The world-x scalar (23) also mirrors, but it is an absolute arena coordinate
# with no canonical flip center under an ego mirror, so it is left untouched (the
# raster/ego lateral signals carry the mirrored geometry). Ahead/behind and
# distances are flip-invariant.
_LATERAL_NEGATE_IDX = (12, 15, 19)
# Wall-distance left/right pair to SWAP under the mirror.
_WALL_LR_SWAP = (9, 11)

# Action layout: 0=turn-left, 1=straight, 2=turn-right (normal),
#                3=turn-left, 4=straight, 5=turn-right (boost).
# A horizontal mirror swaps left<->right: 0<->2 and 3<->5.
_FLIP_ACTION = np.array([2, 1, 0, 5, 4, 3], dtype=np.int64)


@dataclass
class PQNConfig:
    """Configuration for :class:`PQNTrainer` (blueprint P3 §3 defaults).

    Attributes:
        num_envs: E parallel environments.
        num_snakes: S snakes per env.
        rollout_len: T steps collected per update.
        gamma: Discount (blueprint: 0.997).
        lambda_: Q(lambda) trace decay (blueprint: 0.65).
        lr: Adam learning rate (blueprint: 5e-4).
        adam_eps: Adam epsilon (blueprint: 1.5e-4).
        grad_clip: Grad-norm clip (blueprint: 10.0).
        minibatches: Number of minibatch SGD steps per update.
        minibatch_size: Transitions per SGD minibatch (hero transitions are
            sampled without replacement across minibatches, then reshuffled).
        eps_start / eps_end: ε-greedy endpoints (blueprint: 1.0 -> 0.02).
        eps_decay_steps: Agent-steps over which ε decays (blueprint ~50M; scale
            down for smoke).
        hero_frac: Probability a slot is the hero (blueprint: 0.8).
        pool_capacity: Max frozen opponents resident (blueprint: <=10).
        pool_add_interval: Add a hero snapshot to the pool every N updates.
        death_value: Bootstrap value for TRAPPED non-terminal states (blueprint:
            "TRAPPED -> the DEATH VALUE, not 0"). Defaults to the sim's terminal
            death reward (:data:`~src.core.reward_events.DEATH_REWARD` = -3.0) so
            a trapped-but-still-alive state is trained toward the forced-death
            outcome on the following step, not toward 0.
        flip_augment: Enable horizontal-flip augmentation.
        max_frames: Episode-length cap for the episode-progress scalar.
        max_abs_q_alarm: Tripwire threshold on max|Q|.
        seed: Base RNG seed.
        arena_type: Arena type for the sim config.
        mechanics_version: Sim mechanics version (blueprint target: 2).
        reward_version: Reward version recorded in the checkpoint metadata.
    """

    num_envs: int = 8
    num_snakes: int = 6
    rollout_len: int = 32
    gamma: float = 0.997
    lambda_: float = 0.65
    lr: float = 5e-4
    adam_eps: float = 1.5e-4
    grad_clip: float = 10.0
    minibatches: int = 4
    minibatch_size: int = 256
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_steps: int = 50_000_000
    hero_frac: float = 0.8
    pool_capacity: int = 10
    pool_add_interval: int = 50
    death_value: float = DEATH_REWARD
    flip_augment: bool = True
    max_frames: int = 5000
    max_abs_q_alarm: float = 1e3
    seed: int = 0
    arena_type: str = "rectangular"
    mechanics_version: int = 2
    reward_version: int = 2
    profile: bool = False  # print a CUDA-synced per-phase time breakdown each update


@dataclass
class PQNTelemetry:
    """Per-update telemetry snapshot.

    Attributes:
        update: Update index (0-based).
        agent_steps: Cumulative hero agent-steps seen.
        loss: Mean Huber loss over this update's minibatches.
        grad_norm: Mean pre-clip grad norm over minibatches.
        mean_abs_q: Mean |Q| over hero transitions.
        max_abs_q: Max |Q| over hero transitions.
        epsilon: ε used during the rollout.
        mean_reward: Mean hero per-step reward over the rollout.
        action_entropy: Entropy (nats) of the hero action histogram.
        kills_per_ep: Total hero kills over the rollout (per E*T proxy).
        boost_fraction: Fraction of hero steps that engaged boost.
        pool_size: Current opponent-pool size.
    """

    update: int
    agent_steps: int
    loss: float
    grad_norm: float
    mean_abs_q: float
    max_abs_q: float
    epsilon: float
    mean_reward: float
    action_entropy: float
    kills_per_ep: float
    boost_fraction: float
    pool_size: int


class TripwireError(RuntimeError):
    """Raised when a training tripwire fires (NaN/inf, max|Q|, action-collapse)."""


def flip_augment(
    tactical: torch.Tensor,
    strategic: torch.Tensor,
    scalars: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Horizontal-flip augmentation for a batch of raster transitions.

    Mirrors the ego frame left<->right: flip the raster COLUMN axis (last dim),
    swap the lateral wall-distance scalars, negate the lateral ego scalars, and
    remap actions (0<->2, 3<->5). Ahead/behind and distance signals are
    flip-invariant and untouched.

    Args:
        tactical: ``(N, 9, 31, 31)`` float tensor.
        strategic: ``(N, 3, 25, 25)`` float tensor.
        scalars: ``(N, 26)`` float tensor.
        actions: ``(N,)`` long action indices.

    Returns:
        Flipped ``(tactical, strategic, scalars, actions)``.
    """
    tac = torch.flip(tactical, dims=[-1])
    strat = torch.flip(strategic, dims=[-1])
    scal = scalars.clone()
    # Swap left/right wall-distance scalars.
    i, j = _WALL_LR_SWAP
    scal[:, [i, j]] = scal[:, [j, i]]
    # Negate lateral ego scalars.
    for k in _LATERAL_NEGATE_IDX:
        scal[:, k] = -scal[:, k]
    flip_action = torch.as_tensor(_FLIP_ACTION, device=actions.device)
    acts = flip_action[actions]
    return tac, strat, scal, acts


class PQNTrainer:
    """Synchronous Q(lambda) trainer with pool self-play (blueprint P3).

    Owns the network, optimizer, sim, and opponent pool. Call :meth:`update`
    repeatedly (or :meth:`train` for a loop). Each :meth:`update` performs one
    rollout + several minibatch SGD steps and returns a :class:`PQNTelemetry`.

    Args:
        config: Trainer configuration.
        network: Optional pre-built network (default: fresh
            :class:`RasterDuelingNetwork`).
        device: Compute device (default: CPU — the raster net is small enough to
            train on CPU for smoke; use CUDA for full runs).
    """

    def __init__(
        self,
        config: PQNConfig,
        network: Optional[RasterDuelingNetwork] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = config
        self.device = device or torch.device("cpu")
        self.network = (network or RasterDuelingNetwork()).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.lr, eps=config.adam_eps
        )
        self.pool = OpponentPool(capacity=config.pool_capacity, device=self.device)
        self.rng = np.random.default_rng(config.seed)

        sim_cfg = BatchSimConfig(
            num_envs=config.num_envs,
            num_snakes=config.num_snakes,
            mechanics_version=config.mechanics_version,
            gamma=config.gamma,
            arena_type=config.arena_type,
        )
        self.sim = BatchSim(
            sim_cfg,
            seeds=[config.seed + e for e in range(config.num_envs)],
            train_mode=True,
        )

        self.update_idx = 0
        self.agent_steps = 0

    # -- ε schedule ---------------------------------------------------------
    def epsilon(self) -> float:
        """Current ε from the linear ladder (``eps_start`` -> ``eps_end``)."""
        frac = min(1.0, self.agent_steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # -- observation --------------------------------------------------------
    def _sync(self) -> float:
        """Return a timestamp after flushing pending CUDA work (accurate timing)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def _current_obs(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Featurize the sim's current state into device float tensors + mask.

        Returns:
            ``(obs, mask)``: obs dict with ``tactical`` ``(E, S, 9, 31, 31)``,
            ``strategic`` ``(E, S, 3, 25, 25)``, ``scalars`` ``(E, S, 26)``; mask
            ``(E, S, 6)`` bool tensor on ``device``.
        """
        E, S = self.cfg.num_envs, self.cfg.num_snakes
        mask_np = self.sim.get_action_mask()  # (E, S, 6)
        if self.device.type == "cuda":
            # GPU featurizer: transfer only the compact sim state, build the
            # rasters on the (idle) GPU. Byte-identical tactical/strategic planes
            # and <=1e-4 scalars vs the NumPy path (parity-tested), so a
            # GPU-trained policy sees the same inputs the web app serves via the
            # NumPy featurizer. Returns (E, S, ...) tensors directly.
            state = obs_inputs_to_torch(self.sim, self.device, max_frames=self.cfg.max_frames)
            obs_es = build_observations_gpu(state)
            obs_es = {
                "tactical": obs_es["tactical"],
                "strategic": obs_es["strategic"],
                "scalars": obs_es["scalars"],
            }
        else:
            inp = obs_inputs_from_batch_sim(self.sim, max_frames=self.cfg.max_frames)
            obs = build_observations(inp, mask=mask_np)
            tensors = raster_tensors_from_obs(obs, device=self.device)
            # Reshape flat (E*S, ...) back to (E, S, ...).
            obs_es = {
                "tactical": tensors["tactical"].reshape(E, S, *TACTICAL_SHAPE),
                "strategic": tensors["strategic"].reshape(E, S, *STRATEGIC_SHAPE),
                "scalars": tensors["scalars"].reshape(E, S, SCALARS_DIM),
            }
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
        return obs_es, mask

    # -- rollout ------------------------------------------------------------
    def _rollout(self) -> Dict[str, object]:
        """Collect a ``T``-step self-play rollout of hero + frozen transitions.

        Returns:
            Dict of stacked rollout tensors/arrays (see below). Spatial obs are
            kept per-slot; termination bookkeeping is per (t, e, s).
        """
        cfg = self.cfg
        E, S, T = cfg.num_envs, cfg.num_snakes, cfg.rollout_len
        eps = self.epsilon()

        policy_ids = assign_policy_ids(
            E, S, self.pool.policy_ids(), cfg.hero_frac, self.rng, hero_slot0=True
        )

        tac_buf = torch.zeros((T, E, S, *TACTICAL_SHAPE), dtype=torch.float32, device=self.device)
        strat_buf = torch.zeros(
            (T, E, S, *STRATEGIC_SHAPE), dtype=torch.float32, device=self.device
        )
        scal_buf = torch.zeros((T, E, S, SCALARS_DIM), dtype=torch.float32, device=self.device)
        act_buf = np.zeros((T, E, S), dtype=np.int64)
        rew_buf = np.zeros((T, E, S), dtype=np.float64)
        done_buf = np.zeros((T, E, S), dtype=bool)
        trapped_buf = np.zeros((T, E, S), dtype=bool)
        # valid_step[t,e,s]: slot was ALIVE at the start of step t. True on the
        # death step itself (a real terminal transition), False for every
        # post-death "zombie" step (train_mode never respawns). Zombie steps are
        # dropped from the loss and from the backward-return carry.
        valid_buf = np.zeros((T, E, S), dtype=bool)
        # next-mask per step for masked-max bootstrap (mask of s_{t+1}).
        next_mask_buf = torch.zeros((T, E, S, 6), dtype=torch.bool, device=self.device)
        boost_buf = np.zeros((T, E, S), dtype=bool)
        kills_total = 0

        prof = self.cfg.profile
        feat_t = fwd_t = sim_t = 0.0

        for t in range(T):
            if prof:
                t0 = self._sync()
            obs, mask = self._current_obs()
            if prof:
                t1 = self._sync()
                feat_t += t1 - t0
            actions, _ = batched_act(
                self.network, self.pool, policy_ids, obs, mask, eps, self.rng, self.device
            )
            if prof:
                t2 = self._sync()
                fwd_t += t2 - t1

            tac_buf[t] = obs["tactical"]
            strat_buf[t] = obs["strategic"]
            scal_buf[t] = obs["scalars"]
            act_buf[t] = actions
            # A slot is TRAPPED if it currently has no valid action (mask all
            # False) but is alive (so it is not yet a death transition).
            alive = self.sim.get_alive()
            valid_buf[t] = alive
            no_valid = ~mask.any(dim=2).cpu().numpy()
            trapped_buf[t] = no_valid & alive

            if prof:
                t3 = self._sync()
            self.sim.step(actions)
            if prof:
                sim_t += self._sync() - t3

            rew_buf[t] = self.sim.get_reward()
            done_buf[t] = self.sim.get_done()
            boost_buf[t] = self.sim.get_boosted_this_step()
            kills_total += int(self.sim.get_kill_credit().sum())

            # Mask of the NEXT state s_{t+1} for the bootstrap of non-terminal
            # transitions (already updated by step()).
            next_mask_buf[t] = torch.as_tensor(
                self.sim.get_action_mask(), dtype=torch.bool, device=self.device
            )

        # Final observation (s_T) for truncation bootstrap of the last step.
        final_obs, final_mask = self._current_obs()

        if prof:
            self._rollout_prof = {"featurize": feat_t, "forward": fwd_t, "sim": sim_t}

        return {
            "tactical": tac_buf,
            "strategic": strat_buf,
            "scalars": scal_buf,
            "actions": act_buf,
            "rewards": rew_buf,
            "dones": done_buf,
            "trapped": trapped_buf,
            "valid": valid_buf,
            "next_mask": next_mask_buf,
            "boost": boost_buf,
            "policy_ids": policy_ids,
            "final_obs": final_obs,
            "final_mask": final_mask,
            "kills_total": kills_total,
            "epsilon": eps,
        }

    # -- Q(lambda) targets --------------------------------------------------
    def _compute_targets(self, roll: Dict[str, object]) -> torch.Tensor:
        """Compute per-slot Q(lambda) targets ``(T, E, S)`` (blueprint §3).

        Requires one forward over every stored ``(T, E, S)`` obs (for the masked
        next-max) plus the final obs. Backward recursion per agent stream:

            done_t:      G_t = r_t                       (death: no bootstrap)
            trapped_t:   G_t = r_t + gamma * death_value (trapped non-terminal)
            else:        boot = max_{valid} Q(s_{t+1})
                         next_G = G_{t+1} if t<T-1 and step t+1 is a REAL (valid)
                                  transition — including when step t+1 is the
                                  death step, whose G_{t+1}=r_{t+1} then flows
                                  back through the lambda channel;
                                  else boot   (truncation bootstrap at the
                                  rollout edge or across a post-death zombie step)
                         G_t = r_t + gamma * ((1-lambda)*boot + lambda*next_G)

        Post-death "zombie" steps (a slot that was already dead at step entry;
        ``valid[t]`` False) are excluded from training (:meth:`_sgd`) and their
        return is NOT carried into the backward scan: ``next_g`` for a step whose
        successor is a zombie falls back to the truncation bootstrap.

        Args:
            roll: Rollout dict from :meth:`_rollout`.

        Returns:
            ``(T, E, S)`` float tensor of Q(lambda) targets on ``device``.
        """
        cfg = self.cfg
        E, S, T = cfg.num_envs, cfg.num_snakes, cfg.rollout_len
        gamma, lam = cfg.gamma, cfg.lambda_
        neg_inf = torch.finfo(torch.float32).min

        rewards = torch.as_tensor(roll["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(roll["dones"], dtype=torch.bool, device=self.device)
        trapped = torch.as_tensor(roll["trapped"], dtype=torch.bool, device=self.device)
        valid = torch.as_tensor(roll["valid"], dtype=torch.bool, device=self.device)
        next_mask = roll["next_mask"]  # (T, E, S, 6) bool

        # Masked-max Q(s_{t+1}) for every stored step: forward the network on the
        # obs of s_{t+1}. s_{t+1} obs == the stored obs of step t+1 (for t<T-1)
        # and == final_obs for t==T-1.
        with torch.no_grad():
            boot = torch.zeros((T, E, S), dtype=torch.float32, device=self.device)
            for t in range(T):
                if t < T - 1:
                    tac = roll["tactical"][t + 1]
                    strat = roll["strategic"][t + 1]
                    scal = roll["scalars"][t + 1]
                else:
                    tac = roll["final_obs"]["tactical"]
                    strat = roll["final_obs"]["strategic"]
                    scal = roll["final_obs"]["scalars"]
                q_next = self.network(
                    tac.reshape(E * S, *TACTICAL_SHAPE),
                    strat.reshape(E * S, *STRATEGIC_SHAPE),
                    scal.reshape(E * S, SCALARS_DIM),
                ).reshape(E, S, 6)
                m = next_mask[t]
                q_masked = torch.where(m, q_next, torch.full_like(q_next, neg_inf))
                mm = q_masked.max(dim=2).values
                # If no valid next action, masked-max is -inf; fall back to death
                # value (handled by the trapped branch below, but guard here too).
                mm = torch.where(m.any(dim=2), mm, torch.full_like(mm, cfg.death_value))
                boot[t] = mm

        targets = torch.zeros((T, E, S), dtype=torch.float32, device=self.device)
        death_value = float(cfg.death_value)
        # Backward recursion. ``carry[e,s]`` holds ``G_{t+1}`` for the current
        # agent stream and ``succ_valid[e,s]`` records whether that successor step
        # (t+1) was a REAL (alive-at-entry) transition. ``next_g`` for step t uses
        # ``carry`` iff the successor is real — this INCLUDES a death successor,
        # whose ``G_{t+1}=r_{t+1}`` (reward alone) then propagates the terminal
        # return one step back through the lambda channel. When the successor is a
        # post-death "zombie" step (``succ_valid`` False), its return is discarded
        # and step t bootstraps (truncation-like). Death (``done_t``) and trapped
        # branches ignore ``next_g`` entirely.
        carry = torch.zeros((E, S), dtype=torch.float32, device=self.device)
        succ_valid = torch.zeros((E, S), dtype=torch.bool, device=self.device)
        for t in reversed(range(T)):
            r = rewards[t]
            done_t = dones[t]
            trapped_t = trapped[t] & ~done_t  # trapped only if not also dead

            # boot_t is the masked-max Q(s_{t+1}) computed above.
            boot_t = boot[t]

            # next_G: G_{t+1} when the successor is a real (valid) in-rollout
            # transition, else the truncation bootstrap.
            if t < T - 1:
                next_g = torch.where(succ_valid, carry, boot_t)
            else:
                # Rollout edge: truncation -> bootstrap.
                next_g = boot_t

            g_interior = r + gamma * ((1.0 - lam) * boot_t + lam * next_g)
            g_trapped = r + gamma * death_value
            g_death = r

            g = torch.where(
                done_t,
                g_death,
                torch.where(trapped_t, g_trapped, g_interior),
            )
            targets[t] = g
            # Carry g down to step t-1: step t is t-1's successor. A zombie step
            # (not valid) does not seed the carry, so its garbage return never
            # chains into an earlier real transition.
            carry = g
            succ_valid = valid[t]
        return targets

    # -- SGD ----------------------------------------------------------------
    def _sgd(
        self, roll: Dict[str, object], targets: torch.Tensor
    ) -> Tuple[float, float, float, float, float]:
        """Minibatch SGD over HERO transitions (Huber, grad clip, flip aug).

        Args:
            roll: Rollout dict.
            targets: ``(T, E, S)`` Q(lambda) targets.

        Returns:
            ``(mean_loss, mean_grad_norm, mean_abs_q, max_abs_q, entropy)``.
        """
        cfg = self.cfg
        E, S, T = cfg.num_envs, cfg.num_snakes, cfg.rollout_len
        policy_ids = roll["policy_ids"]  # (E, S)

        # Flatten hero transitions across (T, E, S). A slot is a hero transition
        # for a step iff its policy_id == HERO (fixed per rollout) AND it was
        # ALIVE at step entry (``valid``) — post-death zombie steps of a hero
        # slot carry bogus rewards/targets and must not enter the loss.
        hero_es = policy_ids == HERO_POLICY_ID  # (E, S)
        hero_mask_tes = (np.broadcast_to(hero_es[None], (T, E, S)) & roll["valid"]).reshape(-1)
        hero_idx = np.nonzero(hero_mask_tes)[0]
        if hero_idx.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        tac = roll["tactical"].reshape(T * E * S, *TACTICAL_SHAPE)
        strat = roll["strategic"].reshape(T * E * S, *STRATEGIC_SHAPE)
        scal = roll["scalars"].reshape(T * E * S, SCALARS_DIM)
        acts = torch.as_tensor(roll["actions"].reshape(-1), dtype=torch.long, device=self.device)
        tgt = targets.reshape(-1)

        hero_idx_t = torch.as_tensor(hero_idx, device=self.device)
        tac_h = tac[hero_idx_t]
        strat_h = strat[hero_idx_t]
        scal_h = scal[hero_idx_t]
        acts_h = acts[hero_idx_t]
        tgt_h = tgt[hero_idx_t]

        n = hero_idx.size
        losses: List[float] = []
        grad_norms: List[float] = []
        abs_q_vals: List[torch.Tensor] = []
        action_hist = torch.zeros(6, device=self.device)

        for _ in range(cfg.minibatches):
            perm = torch.as_tensor(
                self.rng.permutation(n)[: cfg.minibatch_size], device=self.device
            )
            mtac = tac_h[perm]
            mstrat = strat_h[perm]
            mscal = scal_h[perm]
            macts = acts_h[perm]
            mtgt = tgt_h[perm]

            if cfg.flip_augment and self.rng.random() < 0.5:
                mtac, mstrat, mscal, macts = flip_augment(mtac, mstrat, mscal, macts)

            q = self.network(mtac, mstrat, mscal)  # (m, 6)
            q_taken = q.gather(1, macts.view(-1, 1)).squeeze(1)
            loss = F.smooth_l1_loss(q_taken, mtgt.detach())

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_clip)
            self.optimizer.step()

            losses.append(float(loss.detach()))
            grad_norms.append(float(gnorm))
            abs_q_vals.append(q.detach().abs().reshape(-1))
            action_hist += torch.bincount(macts, minlength=6).float()

        abs_q = torch.cat(abs_q_vals)
        probs = action_hist / action_hist.sum().clamp_min(1.0)
        entropy = float(-(probs * (probs + 1e-12).log()).sum())
        return (
            float(np.mean(losses)),
            float(np.mean(grad_norms)),
            float(abs_q.mean()),
            float(abs_q.max()),
            entropy,
        )

    # -- tripwires ----------------------------------------------------------
    def _check_tripwires(self, tel: PQNTelemetry) -> None:
        """Raise :class:`TripwireError` on NaN/inf, max|Q| alarm, action-collapse.

        Args:
            tel: The telemetry snapshot for the just-finished update.

        Raises:
            TripwireError: When any tripwire condition is met.
        """
        if not np.isfinite(tel.loss) or not np.isfinite(tel.grad_norm):
            raise TripwireError(f"non-finite loss/grad at update {tel.update}")
        if not np.isfinite(tel.max_abs_q) or tel.max_abs_q > self.cfg.max_abs_q_alarm:
            raise TripwireError(
                f"max|Q|={tel.max_abs_q:.3g} exceeded alarm "
                f"{self.cfg.max_abs_q_alarm:.3g} at update {tel.update}"
            )
        # Action-collapse: entropy near zero once past pure-exploration ε.
        if tel.epsilon < 0.5 and tel.action_entropy < 1e-3:
            raise TripwireError(
                f"action collapse (entropy={tel.action_entropy:.3g}) at update {tel.update}"
            )

    # -- public API ---------------------------------------------------------
    def update(self) -> PQNTelemetry:
        """Run one rollout + SGD update; return telemetry.

        Returns:
            The :class:`PQNTelemetry` for this update.

        Raises:
            TripwireError: If a tripwire fires (caller may halt-and-flag).
        """
        cfg = self.cfg
        if cfg.profile:
            p0 = self._sync()
            roll = self._rollout()
            p1 = self._sync()
            targets = self._compute_targets(roll)
            p2 = self._sync()
            loss, gnorm, mean_abs_q, max_abs_q, entropy = self._sgd(roll, targets)
            p3 = self._sync()
            rp = self._rollout_prof
            total = p3 - p0
            steps = cfg.num_envs * cfg.num_snakes * cfg.rollout_len
            print(
                f"[profile] total {total*1e3:6.0f}ms ({steps/total:7.0f} step/s) | "
                f"rollout {(p1-p0)*1e3:5.0f}ms [featurize {rp['featurize']*1e3:5.0f} "
                f"forward {rp['forward']*1e3:5.0f} sim {rp['sim']*1e3:5.0f}] | "
                f"targets {(p2-p1)*1e3:5.0f}ms | sgd {(p3-p2)*1e3:5.0f}ms",
                flush=True,
            )
        else:
            roll = self._rollout()
            targets = self._compute_targets(roll)
            loss, gnorm, mean_abs_q, max_abs_q, entropy = self._sgd(roll, targets)

        E, S, T = cfg.num_envs, cfg.num_snakes, cfg.rollout_len
        hero_es = roll["policy_ids"] == HERO_POLICY_ID
        # Only alive-at-entry hero steps are real transitions (zombie steps of a
        # dead hero slot are excluded, matching the loss/target selection).
        hero_mask_tes = np.broadcast_to(hero_es[None], (T, E, S)) & roll["valid"]
        hero_steps = int(hero_mask_tes.sum())
        self.agent_steps += hero_steps

        # Hero-only reward / boost means.
        hero_rewards = roll["rewards"][hero_mask_tes]
        hero_boost = roll["boost"][hero_mask_tes]
        mean_reward = float(hero_rewards.mean()) if hero_rewards.size else 0.0
        boost_fraction = float(hero_boost.mean()) if hero_boost.size else 0.0

        tel = PQNTelemetry(
            update=self.update_idx,
            agent_steps=self.agent_steps,
            loss=loss,
            grad_norm=gnorm,
            mean_abs_q=mean_abs_q,
            max_abs_q=max_abs_q,
            epsilon=roll["epsilon"],
            mean_reward=mean_reward,
            action_entropy=entropy,
            kills_per_ep=float(roll["kills_total"]),
            boost_fraction=boost_fraction,
            pool_size=len(self.pool),
        )
        self._check_tripwires(tel)

        # Snapshot the hero into the pool at the configured cadence.
        if (
            cfg.pool_capacity > 0
            and self.update_idx > 0
            and self.update_idx % cfg.pool_add_interval == 0
        ):
            self.pool.add_snapshot(self.network)

        self.update_idx += 1
        return tel

    def train(self, num_updates: int) -> List[PQNTelemetry]:
        """Run ``num_updates`` updates, returning the telemetry list.

        Args:
            num_updates: Number of :meth:`update` calls.

        Returns:
            List of per-update telemetry.
        """
        out: List[PQNTelemetry] = []
        for _ in range(num_updates):
            out.append(self.update())
        return out

    # -- checkpoint ---------------------------------------------------------
    def checkpoint_state(self) -> Dict[str, object]:
        """Build a raster checkpoint dict (obs_spec ``raster31v2``, algo ``pqn``).

        The metadata coordinates with the checkpoint-contract module
        (:mod:`src.model.obs_spec`): it records :data:`OBS_SPEC_KEY` =
        ``raster31v2`` and the canonical flat shape keys from
        :meth:`RasterObsShapes.to_metadata` (``tactical_channels`` /
        ``tactical_size`` / ``strategic_channels`` / ``strategic_size`` /
        ``scalars``), so :meth:`InferenceAgent.from_checkpoint` reloads a PQN
        checkpoint and rebuilds a matching :class:`RasterDuelingNetwork` without
        a config. It also records the algorithm/training knobs
        (``gamma``/``lambda``/``algo``/``mechanics_version``/``reward_version``).

        Returns:
            A ``torch.save``-able dict.
        """
        state: Dict[str, object] = {
            "dqn_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            OBS_SPEC_KEY: RASTER31V2,
            "output_size": self.network.output_size,
            "gamma": self.cfg.gamma,
            "lambda": self.cfg.lambda_,
            "algo": "pqn",
            "mechanics_version": int(self.cfg.mechanics_version),
            "reward_version": int(self.cfg.reward_version),
            "update_counter": self.update_idx,
            "agent_steps": self.agent_steps,
        }
        state.update(RASTER31V2_SHAPES.to_metadata())
        return state

    def save_checkpoint(self, path: str) -> None:
        """Write a raster checkpoint to ``path``.

        Args:
            path: Destination file path.
        """
        torch.save(self.checkpoint_state(), path)
