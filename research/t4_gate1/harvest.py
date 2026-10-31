"""Fresh Watch6-v2 exact GPU DeviceSim trajectory harvester & CRN H300 counterfactual brancher.

Uses ``research/gpu_sim_spike`` ``DeviceSim(..., exact_mechanics=True, hero_respawn=True)``
and ``DevicePolicy`` (frozen ``B8 s3`` hero vs ``5x frp3-s12+v8`` opponents) to harvest:
  1. World-disjoint causal trajectory datasets with exact per-life standing-mass returns ``G_t``,
     survival durations ``(tau_t, died_t)``, and alive short/long foraging mass-flux rates
     (hard-cut at hero death, zero cross-respawn bootstrap);
  2. Mature (``L >= min_mature_length``) prepared death-lookback and ordinary root snapshots,
     branched across all canonical legal actions with Common Random Numbers (CRN) selection
     and independent scoring suffixes for ``H_branch`` steps (``H = 300`` in main).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from research.devicesim_v2_parity.harness import (
    SimpleWorldView,
    build_product_world_bundle,
    watch6_v2_config,
)
from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import _ENV_TENSOR_FIELDS, DeviceSim
from research.sandbox_a_resident.ct_branching import (
    compute_q_from_survival_and_mass,
    select_a_star_and_forbidden,
)
from research.t4_gate1.featurizer import (
    CompactBoardDataset,
    CompactSampleBuilder,
)
from research.t4_gate1.model import FORAGING_GAMMA
from src.simd_env.grid_sim import GridBatchSim

_VECTOR_RUNTIME_ENV_TENSORS: tuple[str, ...] = (
    "controlled",
    "prev_dist",
    "prev_id",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
    "sel_state",
    "sel_mask",
    "sel_ready",
    "sel_fresh",
)

DEFAULT_DEATH_LOOKBACKS: tuple[int, ...] = (16, 48, 112, 200)


def derive_crn_rng_key(namespace: str, root_uid: str, suffix_global: int) -> int:
    """Derive a positive 63-bit SplitMix key omitting action ID so actions share CRN per suffix."""
    tag = f"{namespace}:crn-branch:{root_uid}:suffix-{int(suffix_global)}".encode("utf-8")
    val = int.from_bytes(hashlib.sha256(tag).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
    return val if val != 0 else 1


def compute_per_life_targets(
    *,
    acted: np.ndarray,
    died: np.ndarray,
    pre_length: np.ndarray,
    post_length: np.ndarray,
    gamma_f: float = FORAGING_GAMMA,
) -> dict[str, np.ndarray]:
    """Exact O(T) backward recurrence for per-life return, survival, and alive foraging rates.

    Args:
        acted: ``(T, E)`` bool — hero was alive at decision seam ``t``.
        died: ``(T, E)`` bool — hero died during step ``t`` (``sim._last_done[:, 0]``).
        pre_length: ``(T, E)`` int32/float64 — hero pre-step length ``L_t`` at seam ``t``.
        post_length: ``(T, E)`` int32/float64 — hero post-step length ``L^{post}_t``.
        gamma_f: Short-horizon foraging discount factor (default ``0.985``).

    Returns:
        Dict of ``(T, E)`` arrays aligned with seam ``t`` (valid wherever ``acted[t, e]``):
          - ``tau_alive``: int32 post-step ticks survived within ``[t, T)`` by this hero life.
          - ``died_in_session``: bool whether this hero life died before tick ``T``.
          - ``return_g``: float32 exact per-life standing-mass return ``sum (L_post / 1000) I``.
          - ``forage_short``: float32 short-gamma alive per-tick mass-flux rate.
          - ``forage_short_weight``: float32 effective alive weight ``1 - gamma_f ** tau``.
          - ``forage_long``: float32 far-horizon alive drift rate over offsets ``d in [64, tau)``.
          - ``forage_long_valid``: bool ``tau_alive >= 96``.
    """
    t_steps, n_envs = acted.shape
    tau_alive = np.zeros((t_steps, n_envs), dtype=np.int32)
    died_in_session = np.zeros((t_steps, n_envs), dtype=bool)
    return_g = np.zeros((t_steps, n_envs), dtype=np.float64)
    forage_short = np.zeros((t_steps, n_envs), dtype=np.float64)
    forage_short_weight = np.zeros((t_steps, n_envs), dtype=np.float64)
    forage_long = np.zeros((t_steps, n_envs), dtype=np.float64)
    forage_long_valid = np.zeros((t_steps, n_envs), dtype=bool)

    tau_next = np.zeros(n_envs, dtype=np.int32)
    died_next = np.zeros(n_envs, dtype=bool)
    g_next = np.zeros(n_envs, dtype=np.float64)
    f_num_next = np.zeros(n_envs, dtype=np.float64)
    f_den_next = np.zeros(n_envs, dtype=np.float64)
    life_terminal_len = np.zeros(n_envs, dtype=np.float64)

    pre_f = pre_length.astype(np.float64)
    post_f = post_length.astype(np.float64)

    for t in range(t_steps - 1, -1, -1):
        act_t = acted[t]
        die_t = died[t] & act_t
        surv_t = act_t & (~die_t)

        # If hero was not alive pre-step, reset accumulators across the dead/respawn gap.
        tau_cur = np.zeros(n_envs, dtype=np.int32)
        died_cur = np.zeros(n_envs, dtype=bool)
        g_cur = np.zeros(n_envs, dtype=np.float64)
        f_num_cur = np.zeros(n_envs, dtype=np.float64)
        f_den_cur = np.zeros(n_envs, dtype=np.float64)
        term_len_cur = np.zeros(n_envs, dtype=np.float64)

        # Hero died ON step t -> tau = 0, died = True, G = 0.
        died_cur = np.where(die_t, True, died_cur)

        # Hero survived step t -> extend current life suffix from t+1 (if t+1 was same life).
        if np.any(surv_t):
            new_tau = 1 + tau_next
            r_step = post_f[t] / 1000.0
            delta_step = (post_f[t] - pre_f[t]) / 1000.0
            new_g = r_step + g_next
            new_f_num = delta_step + gamma_f * f_num_next
            new_f_den = 1.0 + gamma_f * f_den_next
            new_term_len = np.where(tau_next == 0, post_f[t], life_terminal_len)

            tau_cur = np.where(surv_t, new_tau, tau_cur)
            died_cur = np.where(surv_t, died_next, died_cur)
            g_cur = np.where(surv_t, new_g, g_cur)
            f_num_cur = np.where(surv_t, new_f_num, f_num_cur)
            f_den_cur = np.where(surv_t, new_f_den, f_den_cur)
            term_len_cur = np.where(surv_t, new_term_len, term_len_cur)

        tau_alive[t] = tau_cur
        died_in_session[t] = died_cur
        return_g[t] = g_cur
        safe_den = np.where(f_den_cur > 0.0, f_den_cur, 1.0)
        forage_short[t] = np.where(f_den_cur > 0.0, f_num_cur / safe_den, 0.0)
        forage_short_weight[t] = np.where(tau_cur > 0, 1.0 - np.power(gamma_f, tau_cur), 0.0)

        # Exact telescoping far-horizon drift over d in [64, tau_cur - 1] when tau_cur >= 96.
        long_ok = surv_t & (tau_cur >= 96) & ((t + 63) < t_steps)
        if np.any(long_ok):
            len_at_63 = post_f[min(t + 63, t_steps - 1)]
            span = np.maximum(tau_cur - 64, 1).astype(np.float64)
            drift_long = ((term_len_cur - len_at_63) / 1000.0) / span
            forage_long[t] = np.where(long_ok, drift_long, 0.0)
            forage_long_valid[t] = long_ok

        # Propagate backward to t - 1 ONLY when step t was alive; dead frames never bridge lives.
        tau_next = np.where(act_t, tau_cur, 0)
        died_next = np.where(act_t, died_cur, False)
        g_next = np.where(act_t, g_cur, 0.0)
        f_num_next = np.where(act_t, f_num_cur, 0.0)
        f_den_next = np.where(act_t, f_den_cur, 0.0)
        life_terminal_len = np.where(act_t, term_len_cur, 0.0)

    return {
        "tau_alive": tau_alive,
        "died_in_session": died_in_session,
        "return_g": return_g.astype(np.float32),
        "forage_short": forage_short.astype(np.float32),
        "forage_short_weight": forage_short_weight.astype(np.float32),
        "forage_long": forage_long.astype(np.float32),
        "forage_long_valid": forage_long_valid,
    }


@dataclass
class PreparedRootSnapshot:
    """Lossless single-world prepared DeviceSim + DeviceVectorRuntime snapshot (~6 KB)."""

    root_uid: str
    kind: str  # "death_lookback" or "ordinary"
    lookback: int | None
    world_id: int
    seed: int
    life_id: int
    tick: int
    h_rem: int
    hero_length: int
    b8_action: int
    legal_mask: np.ndarray  # (6,) bool

    # Normalized compact DeviceSim state (prepared == True)
    body_segments: tuple[np.ndarray, ...]  # 6 arrays of shape (seg_count[s], 2) int16, head->tail
    seg_count: np.ndarray  # (6,) int64
    length: np.ndarray  # (6,) int64
    alive: np.ndarray  # (6,) bool
    direction: np.ndarray  # (6,) int64
    boost_frames: np.ndarray  # (6,) int64
    frames_since_food: np.ndarray  # (6,) int64
    respawn_timer: np.ndarray  # (6,) int64
    reward_prev_length: np.ndarray  # (6,) int64
    frame: int
    boosted_this_step: np.ndarray  # (6,) bool
    last_reward: np.ndarray  # (6,) float32
    last_done: np.ndarray  # (6,) bool
    last_death_cause: np.ndarray  # (6,) int64
    last_kills: np.ndarray  # (6,) int64
    last_food_ate: np.ndarray  # (6,) bool
    last_ambient_food_ate: np.ndarray  # (6,) bool
    last_corpse_food_ate: np.ndarray  # (6,) bool
    last_transition_valid: np.ndarray  # (6,) bool
    food_cells: np.ndarray  # (F, 2) int16
    corpse_food: np.ndarray  # (F,) bool

    # Prepared DeviceVectorRuntime state (vector_prepared == True)
    vec_tensors: dict[str, np.ndarray]


@torch.inference_mode()
def capture_prepared_root(
    sim: DeviceSim,
    policy: DevicePolicy,
    *,
    env: int,
    root_uid: str,
    kind: str,
    lookback: int | None,
    world_id: int,
    seed: int,
    life_id: int,
    tick: int,
    h_rem: int,
    b8_action: int,
    legal_mask: np.ndarray,
) -> PreparedRootSnapshot:
    """Capture one world's prepared decision-seam state without retaining RNG fields."""
    if not sim._prepared or not policy.vector_prepared:
        raise ValueError("capture_prepared_root requires prepared DeviceSim and DevicePolicy")

    sc = sim.seg_count[env].detach().cpu().numpy().astype(np.int64)
    hp = sim.head_ptr[env].detach().cpu().numpy().astype(np.int64)
    bodies_env = sim.bodies[env].detach().cpu().numpy()  # (6, cap, 2)
    segs: list[np.ndarray] = []
    for s in range(6):
        n = int(sc[s])
        if n > 0:
            rings = (int(hp[s]) - np.arange(n, dtype=np.int64)) % sim.cap
            segs.append(bodies_env[s, rings].astype(np.int16).copy())
        else:
            segs.append(np.zeros((0, 2), dtype=np.int16))

    fc = int(sim.food_count[env].item())
    f_cells = sim.food_cells[env, :fc].detach().cpu().numpy().astype(np.int16).copy()
    f_corpse = sim.corpse_food[env, :fc].detach().cpu().numpy().astype(bool).copy()

    leg = np.asarray(legal_mask, dtype=bool).reshape(6).copy()
    if not leg.any():
        leg[:3] = True

    vr = policy.vector_runtime
    vec_dict: dict[str, np.ndarray] = {}
    for attr in _VECTOR_RUNTIME_ENV_TENSORS:
        t_val = getattr(vr, attr)
        vec_dict[attr] = t_val[env].detach().cpu().numpy().copy()

    ln = sim.length[env].detach().cpu().numpy().astype(np.int64)
    return PreparedRootSnapshot(
        root_uid=root_uid,
        kind=kind,
        lookback=lookback,
        world_id=int(world_id),
        seed=int(seed),
        life_id=int(life_id),
        tick=int(tick),
        h_rem=int(h_rem),
        hero_length=int(ln[0]),
        b8_action=int(b8_action),
        legal_mask=leg,
        body_segments=tuple(segs),
        seg_count=sc,
        length=ln,
        alive=sim.alive[env].detach().cpu().numpy().astype(bool).copy(),
        direction=sim.direction[env].detach().cpu().numpy().astype(np.int64).copy(),
        boost_frames=sim.boost_frames[env].detach().cpu().numpy().astype(np.int64).copy(),
        frames_since_food=sim.frames_since_food[env].detach().cpu().numpy().astype(np.int64).copy(),
        respawn_timer=sim.respawn_timer[env].detach().cpu().numpy().astype(np.int64).copy(),
        reward_prev_length=(
            sim._reward_prev_length[env].detach().cpu().numpy().astype(np.int64).copy()
        ),
        frame=int(sim.frame[env].item()),
        boosted_this_step=sim._boosted_this_step[env].detach().cpu().numpy().astype(bool).copy(),
        last_reward=sim._last_reward[env].detach().cpu().numpy().astype(np.float32).copy(),
        last_done=sim._last_done[env].detach().cpu().numpy().astype(bool).copy(),
        last_death_cause=sim._last_death_cause[env].detach().cpu().numpy().astype(np.int64).copy(),
        last_kills=sim._last_kills[env].detach().cpu().numpy().astype(np.int64).copy(),
        last_food_ate=sim._last_food_ate[env].detach().cpu().numpy().astype(bool).copy(),
        last_ambient_food_ate=(
            sim._last_ambient_food_ate[env].detach().cpu().numpy().astype(bool).copy()
        ),
        last_corpse_food_ate=(
            sim._last_corpse_food_ate[env].detach().cpu().numpy().astype(bool).copy()
        ),
        last_transition_valid=(
            sim._last_transition_valid[env].detach().cpu().numpy().astype(bool).copy()
        ),
        food_cells=f_cells,
        corpse_food=f_corpse,
        vec_tensors=vec_dict,
    )


@torch.inference_mode()
def restore_roots_into_slab(
    snapshots: Sequence[PreparedRootSnapshot],
    *,
    template_policy: DevicePolicy,
    device: torch.device,
    max_capacity: int,
    rng_keys: Sequence[int],
) -> tuple[DeviceSim, DevicePolicy]:
    """Restore ``B`` prepared root snapshots into a batched ``DeviceSim`` + ``DevicePolicy``."""
    b = len(snapshots)
    if b == 0 or len(rng_keys) != b:
        raise ValueError("snapshots must be nonempty and match rng_keys length")

    cfg = watch6_v2_config(b, max_capacity=max_capacity)
    sim = DeviceSim(
        cfg,
        device=device,
        train_mode=False,
        allow_respawn=True,
        cap=max_capacity,
        move_backend="indexed" if device.type == "cuda" else "reference",
        hero_respawn=False,
        exact_mechanics=True,
    )
    r2_cache.enable(sim)

    bodies_np = np.zeros((b, 6, max_capacity, 2), dtype=np.int64)
    head_ptr_np = np.zeros((b, 6), dtype=np.int64)
    food_cells_np = np.zeros((b, sim.food_capacity, 2), dtype=np.int64)
    food_pres_np = np.zeros((b, sim.food_capacity), dtype=bool)
    corpse_np = np.zeros((b, sim.food_capacity), dtype=bool)
    food_cnt_np = np.zeros(b, dtype=np.int64)

    for row_i, snap in enumerate(snapshots):
        for s in range(6):
            n = int(snap.seg_count[s])
            if n > max_capacity:
                raise ValueError(f"snapshot segment count {n} exceeds max_capacity {max_capacity}")
            if n > 0:
                # Place head (k=0) at ring slot n-1 and tail (k=n-1) at ring slot 0.
                bodies_np[row_i, s, :n] = snap.body_segments[s][::-1]
                head_ptr_np[row_i, s] = n - 1
        fc = int(snap.food_cells.shape[0])
        if fc > 0:
            food_cells_np[row_i, :fc] = snap.food_cells
            food_pres_np[row_i, :fc] = True
            corpse_np[row_i, :fc] = snap.corpse_food
            food_cnt_np[row_i] = fc

    sim.bodies.copy_(torch.as_tensor(bodies_np, dtype=torch.long, device=device))
    sim.head_ptr.copy_(torch.as_tensor(head_ptr_np, dtype=torch.long, device=device))
    sim.seg_count.copy_(
        torch.as_tensor(np.stack([s.seg_count for s in snapshots]), dtype=torch.long, device=device)
    )
    sim.length.copy_(
        torch.as_tensor(np.stack([s.length for s in snapshots]), dtype=torch.long, device=device)
    )
    sim.alive.copy_(
        torch.as_tensor(np.stack([s.alive for s in snapshots]), dtype=torch.bool, device=device)
    )
    sim.direction.copy_(
        torch.as_tensor(np.stack([s.direction for s in snapshots]), dtype=torch.long, device=device)
    )
    sim.boost_frames.copy_(
        torch.as_tensor(
            np.stack([s.boost_frames for s in snapshots]), dtype=torch.long, device=device
        )
    )
    sim.frames_since_food.copy_(
        torch.as_tensor(
            np.stack([s.frames_since_food for s in snapshots]), dtype=torch.long, device=device
        )
    )
    sim.respawn_timer.copy_(
        torch.as_tensor(
            np.stack([s.respawn_timer for s in snapshots]), dtype=torch.long, device=device
        )
    )
    sim._reward_prev_length.copy_(
        torch.as_tensor(
            np.stack([s.reward_prev_length for s in snapshots]), dtype=torch.long, device=device
        )
    )
    sim.frame.copy_(torch.as_tensor([s.frame for s in snapshots], dtype=torch.long, device=device))
    sim._boosted_this_step.copy_(
        torch.as_tensor(
            np.stack([s.boosted_this_step for s in snapshots]), dtype=torch.bool, device=device
        )
    )
    sim._last_reward.copy_(
        torch.as_tensor(
            np.stack([s.last_reward for s in snapshots]), dtype=torch.float32, device=device
        )
    )
    sim._last_done.copy_(
        torch.as_tensor(np.stack([s.last_done for s in snapshots]), dtype=torch.bool, device=device)
    )
    sim._last_death_cause.copy_(
        torch.as_tensor(
            np.stack([s.last_death_cause for s in snapshots]), dtype=torch.long, device=device
        )
    )
    sim._last_kills.copy_(
        torch.as_tensor(
            np.stack([s.last_kills for s in snapshots]), dtype=torch.long, device=device
        )
    )
    sim._last_food_ate.copy_(
        torch.as_tensor(
            np.stack([s.last_food_ate for s in snapshots]), dtype=torch.bool, device=device
        )
    )
    sim._last_ambient_food_ate.copy_(
        torch.as_tensor(
            np.stack([s.last_ambient_food_ate for s in snapshots]), dtype=torch.bool, device=device
        )
    )
    sim._last_corpse_food_ate.copy_(
        torch.as_tensor(
            np.stack([s.last_corpse_food_ate for s in snapshots]), dtype=torch.bool, device=device
        )
    )
    sim._last_transition_valid.copy_(
        torch.as_tensor(
            np.stack([s.last_transition_valid for s in snapshots]), dtype=torch.bool, device=device
        )
    )
    sim.food_cells.copy_(torch.as_tensor(food_cells_np, dtype=torch.long, device=device))
    sim.food_present.copy_(torch.as_tensor(food_pres_np, dtype=torch.bool, device=device))
    sim.corpse_food.copy_(torch.as_tensor(corpse_np, dtype=torch.bool, device=device))
    sim.food_count.copy_(torch.as_tensor(food_cnt_np, dtype=torch.long, device=device))

    sim.terminal_slots[:, 0] = True
    sim.rng_key.copy_(torch.as_tensor(list(rng_keys), dtype=torch.long, device=device))
    sim.rng_counter.zero_()
    sim._prepared = True
    r2_cache.bump(sim)

    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

    vr = DeviceVectorRuntime(template_policy.vector_runtime.params)
    controlled = torch.zeros((b, 6), dtype=torch.bool, device=device)
    controlled[:, 1:] = True
    vr.bind(sim, controlled)
    for attr in _VECTOR_RUNTIME_ENV_TENSORS:
        stacked = np.stack([s.vec_tensors[attr] for s in snapshots], axis=0)
        dst = getattr(vr, attr)
        dst.copy_(torch.as_tensor(stacked, dtype=dst.dtype, device=device))

    slab_policy = replace(
        template_policy,
        device=device,
        vector_runtime=vr,
        vector_prepared=True,
    )
    return sim, slab_policy


def _compact_branch_slab(sim: DeviceSim, policy: DevicePolicy, keep: torch.Tensor) -> None:
    keep_idx = keep.to(device=sim.device, dtype=torch.long).flatten()
    new_e = int(keep_idx.numel())
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            setattr(sim, name, val.index_select(0, keep_idx))
    sim.E = new_e
    sim.cfg = replace(sim.cfg, num_envs=new_e)
    r2_cache.bump(sim)

    vr = policy.vector_runtime
    for attr in _VECTOR_RUNTIME_ENV_TENSORS:
        val = getattr(vr, attr)
        setattr(vr, attr, val.index_select(0, keep_idx))


def build_watch6_v2_sim_and_policy(
    seeds: Sequence[int],
    *,
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    device: str | torch.device = "cpu",
    max_capacity: int = 3200,
) -> tuple[DeviceSim, DevicePolicy, SimpleWorldView]:
    """Instantiate exact Watch6 mechanics-v2 ``DeviceSim`` + ``DevicePolicy`` for ``seeds``."""
    target = torch.device("cuda:0" if str(device) == "cuda" else device)
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")

    cfg = watch6_v2_config(len(seeds), max_capacity=max_capacity)
    cpu_sim = GridBatchSim(cfg, seeds=[int(s) for s in seeds], train_mode=False, allow_respawn=True)
    gpu_sim = DeviceSim.from_reference(
        cpu_sim,
        device=target,
        hero_respawn=True,
        exact_mechanics=True,
    )
    gpu_sim.move_backend = "indexed" if target.type == "cuda" else "reference"
    r2_cache.enable(gpu_sim)

    world = build_product_world_bundle(cpu_sim, b8_checkpoint, frp3_checkpoint)
    gpu_policy = DevicePolicy.from_world(world, device=target, sim=gpu_sim)
    return gpu_sim, gpu_policy, world


@dataclass
class HarvestedWorldPartition:
    """Complete trajectory dataset + selected prepared roots for one world-disjoint partition."""

    partition: str
    world_ids: list[int]
    seeds: list[int]
    dataset: CompactBoardDataset
    roots: list[PreparedRootSnapshot]
    roots_dataset: CompactBoardDataset
    telemetry: dict[str, Any]


def _select_balanced_roots(
    death_candidates: Sequence[PreparedRootSnapshot],
    ordinary_candidates: Sequence[PreparedRootSnapshot],
    *,
    max_roots: int,
    death_fraction: float = 0.625,
) -> list[PreparedRootSnapshot]:
    """Deterministically select balanced death-lookback and ordinary mature roots."""
    if max_roots <= 0:
        return []
    target_death = int(round(max_roots * death_fraction))

    # Sort deterministically by (lookback, world_id, tick) before stratified striding.
    death_sorted = sorted(
        death_candidates,
        key=lambda r: (r.lookback or 0, r.world_id, r.tick, r.root_uid),
    )
    ord_sorted = sorted(
        ordinary_candidates,
        key=lambda r: (
            0 if r.hero_length < 800 else (1 if r.hero_length < 1100 else 2),
            r.world_id,
            r.tick,
            r.root_uid,
        ),
    )

    def _even_subsample(
        items: Sequence[PreparedRootSnapshot], k: int
    ) -> list[PreparedRootSnapshot]:
        if k <= 0 or not items:
            return []
        if len(items) <= k:
            return list(items)
        idxs = np.linspace(0, len(items) - 1, num=k, dtype=np.int64)
        return [items[int(i)] for i in idxs]

    chosen_death = _even_subsample(death_sorted, min(target_death, len(death_sorted)))
    rem_for_ord = max_roots - len(chosen_death)
    chosen_ord = _even_subsample(ord_sorted, min(rem_for_ord, len(ord_sorted)))

    if len(chosen_death) + len(chosen_ord) < max_roots and len(death_sorted) > len(chosen_death):
        used_uids = {r.root_uid for r in chosen_death}
        extra = [r for r in death_sorted if r.root_uid not in used_uids]
        need = max_roots - (len(chosen_death) + len(chosen_ord))
        chosen_death.extend(_even_subsample(extra, need))

    return sorted(chosen_death + chosen_ord, key=lambda r: (r.world_id, r.tick, r.root_uid))


@torch.inference_mode()
def harvest_world_partition(
    *,
    partition: str,
    world_rows: Sequence[Mapping[str, Any]],
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    session_ticks: int,
    max_slab_envs: int = 64,
    max_capacity: int = 3200,
    stride_mature: int = 4,
    stride_early: int = 16,
    min_mature_length: int = 600,
    max_roots: int = 128,
    branch_horizon: int = 300,
    death_lookbacks: Sequence[int] = DEFAULT_DEATH_LOOKBACKS,
    ordinary_root_stride: int = 96,
    device: str | torch.device = "cpu",
    log_progress: Callable[[dict[str, Any]], None] | None = None,
) -> HarvestedWorldPartition:
    """Roll out Watch6-v2 worlds and extract causal per-life returns + mature root snapshots."""
    t0 = time.perf_counter()
    target = torch.device("cuda:0" if str(device) == "cuda" else device)
    rows = list(world_rows)
    if not rows:
        raise ValueError(f"world_rows for partition {partition!r} must be nonempty")

    builder = CompactSampleBuilder()
    target_tau_chunks: list[np.ndarray] = []
    target_died_chunks: list[np.ndarray] = []
    target_g_chunks: list[np.ndarray] = []
    target_fs_chunks: list[np.ndarray] = []
    target_fw_chunks: list[np.ndarray] = []
    target_fl_chunks: list[np.ndarray] = []
    target_flv_chunks: list[np.ndarray] = []

    death_candidates: list[PreparedRootSnapshot] = []
    ordinary_candidates: list[PreparedRootSnapshot] = []

    lookbacks = tuple(sorted(int(lb) for lb in death_lookbacks if int(lb) > 0))
    ring_span = (max(lookbacks) + 1) if (lookbacks and max_roots > 0) else 0

    total_hero_deaths = 0
    total_mature_ticks = 0
    total_alive_ticks = 0
    template_policy: DevicePolicy | None = None

    for slab_start in range(0, len(rows), max_slab_envs):
        slab_rows = rows[slab_start : slab_start + max_slab_envs]
        e_count = len(slab_rows)
        slab_seeds = [int(r["seed"]) for r in slab_rows]
        slab_wids = np.asarray([int(r["global_index"]) for r in slab_rows], dtype=np.int32)
        slab_seeds_np = np.asarray(slab_seeds, dtype=np.int64)

        sim, policy, _ = build_watch6_v2_sim_and_policy(
            slab_seeds,
            b8_checkpoint=b8_checkpoint,
            frp3_checkpoint=frp3_checkpoint,
            device=target,
            max_capacity=max_capacity,
        )
        if template_policy is None:
            template_policy = policy

        acted_seq = np.zeros((session_ticks, e_count), dtype=bool)
        died_seq = np.zeros((session_ticks, e_count), dtype=bool)
        pre_len_seq = np.zeros((session_ticks, e_count), dtype=np.int32)
        post_len_seq = np.zeros((session_ticks, e_count), dtype=np.int32)

        # Record (tick, env_offset) coordinates for every sampled row in this slab.
        sampled_t: list[np.ndarray] = []
        sampled_e: list[np.ndarray] = []

        life_ids = np.zeros(e_count, dtype=np.int32)
        prev_alive = np.ones(e_count, dtype=bool)
        ring_buf: list[list[PreparedRootSnapshot | None]] = [
            [None] * e_count for _ in range(ring_span)
        ]

        last_log = time.perf_counter()
        for tick in range(int(session_ticks)):
            h_rem = int(session_ticks - tick)
            sim.prepare()

            alive_np = sim.alive[:, 0].detach().cpu().numpy().astype(bool)
            respawned = (~prev_alive) & alive_np
            if np.any(respawned):
                life_ids[respawned] += 1
            prev_alive = alive_np.copy()

            _, legal_t, hero_act_t = policy.evaluate(sim)
            full_actions_t = policy.actions(sim, hero_act_t)

            len_np = sim.length[:, 0].detach().cpu().numpy().astype(np.int32)
            b8_act_np = hero_act_t.detach().cpu().numpy().astype(np.int64)
            legal_np = legal_t.detach().cpu().numpy().astype(bool)

            acted_seq[tick] = alive_np
            pre_len_seq[tick] = len_np
            total_alive_ticks += int(alive_np.sum())
            mature_mask = alive_np & (len_np >= min_mature_length)
            total_mature_ticks += int(mature_mask.sum())

            # Strictly causal, outcome-blind trajectory sampling grid (offset by world_id % stride).
            take_mature = mature_mask & (((tick + slab_wids) % stride_mature) == 0)
            take_early = alive_np & (~mature_mask) & (((tick + slab_wids) % stride_early) == 0)
            take_mask = take_mature | take_early
            if np.any(take_mask):
                env_offs = np.flatnonzero(take_mask)
                env_t = torch.as_tensor(env_offs, dtype=torch.long, device=target)
                builder.append_from_sim(
                    sim,
                    env_indices=env_t,
                    world_ids=slab_wids[env_offs],
                    seeds=slab_seeds_np[env_offs],
                    life_ids=life_ids[env_offs],
                    ticks=np.full(env_offs.size, tick, dtype=np.int32),
                    h_rems=np.full(env_offs.size, h_rem, dtype=np.int32),
                    b8_actions=b8_act_np[env_offs],
                    legal_masks=legal_np[env_offs],
                )
                sampled_t.append(np.full(env_offs.size, tick, dtype=np.int32))
                sampled_e.append(env_offs.astype(np.int32))

            # Maintain mature prepared root ring buffer for death lookbacks + ordinary snapshots.
            if ring_span > 0:
                slot_ring = ring_buf[tick % ring_span]
                for env_i in range(e_count):
                    if mature_mask[env_i]:
                        uid = (
                            f"{partition}:w{int(slab_wids[env_i])}:l{int(life_ids[env_i])}:t{tick}"
                        )
                        snap = capture_prepared_root(
                            sim,
                            policy,
                            env=env_i,
                            root_uid=uid,
                            kind="ordinary",
                            lookback=None,
                            world_id=int(slab_wids[env_i]),
                            seed=int(slab_seeds_np[env_i]),
                            life_id=int(life_ids[env_i]),
                            tick=tick,
                            h_rem=h_rem,
                            b8_action=int(b8_act_np[env_i]),
                            legal_mask=legal_np[env_i],
                        )
                        slot_ring[env_i] = snap
                        if (
                            ordinary_root_stride > 0
                            and ((tick + int(slab_wids[env_i]) * 7) % ordinary_root_stride) == 0
                            and h_rem >= int(branch_horizon)
                        ):
                            ordinary_candidates.append(snap)
                    else:
                        slot_ring[env_i] = None

            sim.step(full_actions_t)
            policy.after_step(sim)

            died_now = sim._last_done[:, 0].detach().cpu().numpy().astype(bool)
            post_len_now = sim.length[:, 0].detach().cpu().numpy().astype(np.int32)
            died_seq[tick] = died_now
            post_len_seq[tick] = post_len_now
            total_hero_deaths += int(died_now.sum())

            if ring_span > 0 and np.any(died_now):
                for env_i in np.flatnonzero(died_now).tolist():
                    cur_life = int(life_ids[env_i])
                    for lb in lookbacks:
                        src_tick = tick - lb
                        if src_tick < 0:
                            continue
                        cand = ring_buf[src_tick % ring_span][env_i]
                        if (
                            cand is not None
                            and cand.tick == src_tick
                            and cand.life_id == cur_life
                            and cand.hero_length >= min_mature_length
                        ):
                            death_candidates.append(
                                replace(
                                    cand,
                                    root_uid=f"{cand.root_uid}:death_lb{lb}",
                                    kind="death_lookback",
                                    lookback=int(lb),
                                )
                            )

            now = time.perf_counter()
            if log_progress is not None and (now - last_log >= 90.0):
                last_log = now
                log_progress(
                    {
                        "phase": f"harvest_{partition}",
                        "slab_start": slab_start,
                        "tick": tick + 1,
                        "session_ticks": session_ticks,
                        "sampled_rows": builder.num_rows,
                        "death_root_candidates": len(death_candidates),
                        "ordinary_root_candidates": len(ordinary_candidates),
                        "elapsed_sec": round(now - t0, 2),
                    }
                )

        if bool(sim.capacity_overflow.any().item()):
            raise RuntimeError(
                f"DeviceSim body ring capacity overflow during {partition} harvest "
                f"(max_capacity={max_capacity})"
            )

        targets = compute_per_life_targets(
            acted=acted_seq,
            died=died_seq,
            pre_length=pre_len_seq,
            post_length=post_len_seq,
            gamma_f=FORAGING_GAMMA,
        )
        if sampled_t:
            st = np.concatenate(sampled_t, axis=0)
            se = np.concatenate(sampled_e, axis=0)
            target_tau_chunks.append(targets["tau_alive"][st, se])
            target_died_chunks.append(targets["died_in_session"][st, se])
            target_g_chunks.append(targets["return_g"][st, se])
            target_fs_chunks.append(targets["forage_short"][st, se])
            target_fw_chunks.append(targets["forage_short_weight"][st, se])
            target_fl_chunks.append(targets["forage_long"][st, se])
            target_flv_chunks.append(targets["forage_long_valid"][st, se])

    if builder.num_rows > 0:
        dataset = builder.build(
            tau_alive=np.concatenate(target_tau_chunks, axis=0),
            died_in_session=np.concatenate(target_died_chunks, axis=0),
            return_g=np.concatenate(target_g_chunks, axis=0),
            forage_short=np.concatenate(target_fs_chunks, axis=0),
            forage_short_weight=np.concatenate(target_fw_chunks, axis=0),
            forage_long=np.concatenate(target_fl_chunks, axis=0),
            forage_long_valid=np.concatenate(target_flv_chunks, axis=0),
        )
    else:
        dataset = builder.build()

    selected_roots = _select_balanced_roots(
        death_candidates,
        ordinary_candidates,
        max_roots=max_roots,
    )

    # Materialize a CompactBoardDataset aligned 1:1 with selected_roots for fast critic evaluation.
    root_builder = CompactSampleBuilder()
    if selected_roots and template_policy is not None:
        dummy_keys = [1] * len(selected_roots)
        r_sim, _ = restore_roots_into_slab(
            selected_roots,
            template_policy=template_policy,
            device=torch.device("cpu"),
            max_capacity=max_capacity,
            rng_keys=dummy_keys,
        )
        root_builder.append_from_sim(
            r_sim,
            env_indices=torch.arange(len(selected_roots), dtype=torch.long),
            world_ids=np.asarray([r.world_id for r in selected_roots], dtype=np.int32),
            seeds=np.asarray([r.seed for r in selected_roots], dtype=np.int64),
            life_ids=np.asarray([r.life_id for r in selected_roots], dtype=np.int32),
            ticks=np.asarray([r.tick for r in selected_roots], dtype=np.int32),
            h_rems=np.asarray([r.h_rem for r in selected_roots], dtype=np.int32),
            b8_actions=np.asarray([r.b8_action for r in selected_roots], dtype=np.int64),
            legal_masks=np.stack([r.legal_mask for r in selected_roots], axis=0),
        )
    roots_dataset = root_builder.build()

    wall_s = time.perf_counter() - t0
    world_ticks = len(rows) * int(session_ticks)
    telemetry = {
        "partition": partition,
        "num_worlds": len(rows),
        "session_ticks": int(session_ticks),
        "world_ticks": world_ticks,
        "wall_sec": round(wall_s, 3),
        "world_ticks_per_sec": round(world_ticks / max(wall_s, 1e-6), 2),
        "sampled_rows": int(dataset.num_rows),
        "sampled_mature_rows": int(np.sum(dataset.snake_length[:, 0] >= min_mature_length)),
        "total_alive_ticks": int(total_alive_ticks),
        "total_mature_ticks": int(total_mature_ticks),
        "total_hero_deaths": int(total_hero_deaths),
        "death_root_candidates": len(death_candidates),
        "ordinary_root_candidates": len(ordinary_candidates),
        "selected_roots": len(selected_roots),
        "selected_death_roots": sum(1 for r in selected_roots if r.kind == "death_lookback"),
        "selected_ordinary_roots": sum(1 for r in selected_roots if r.kind == "ordinary"),
    }
    return HarvestedWorldPartition(
        partition=partition,
        world_ids=[int(r["global_index"]) for r in rows],
        seeds=[int(r["seed"]) for r in rows],
        dataset=dataset,
        roots=selected_roots,
        roots_dataset=roots_dataset,
        telemetry=telemetry,
    )


@dataclass
class BranchedRootsTable:
    """Counterfactual H-step branch outcomes across CRN selection + independent scoring suffixes."""

    num_roots: int
    horizon: int
    legal_mask: np.ndarray  # (R, 6) bool
    b8_action: np.ndarray  # (R,) int64
    s_sel: np.ndarray  # (R, 6) float64 (NaN off legal)
    m_sel: np.ndarray  # (R, 6) float64 (NaN off legal)
    q_sel: np.ndarray  # (R, 6) float64 (NaN off legal)
    oracle_a_star: np.ndarray  # (R,) int64 selected strictly on selection suffixes
    s_score: np.ndarray  # (R, 6) float64 (NaN off legal)
    m_score: np.ndarray  # (R, 6) float64 (NaN off legal)
    q_score: np.ndarray  # (R, 6) float64 (NaN off legal)
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class _RootBranchJob:
    root_idx: int
    action: int
    suffix_offset: int
    rng_key: int


@torch.inference_mode()
def branch_prepared_roots(
    roots: Sequence[PreparedRootSnapshot],
    *,
    namespace: str,
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    horizon: int = 300,
    selection_suffixes: Sequence[int] = (0, 1),
    scoring_suffixes: Sequence[int] = (4, 5),
    max_slab_envs: int = 128,
    max_capacity: int = 3200,
    compaction_interval: int = 10,
    device: str | torch.device = "cpu",
    log_progress: Callable[[dict[str, Any]], None] | None = None,
) -> BranchedRootsTable:
    """Branch all legal actions on ``roots`` across CRN selection and scoring suffixes."""
    t0 = time.perf_counter()
    target = torch.device("cuda:0" if str(device) == "cuda" else device)
    n_roots = len(roots)
    if n_roots == 0:
        empty_b = np.zeros((0, 6), dtype=bool)
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros((0, 6), dtype=np.float64)
        return BranchedRootsTable(
            num_roots=0,
            horizon=int(horizon),
            legal_mask=empty_b,
            b8_action=empty_i,
            s_sel=empty_f,
            m_sel=empty_f,
            q_sel=empty_f,
            oracle_a_star=empty_i,
            s_score=empty_f,
            m_score=empty_f,
            q_score=empty_f,
            telemetry={"num_roots": 0, "wall_sec": 0.0},
        )

    if set(selection_suffixes) & set(scoring_suffixes):
        raise ValueError("selection_suffixes and scoring_suffixes must be strictly disjoint")

    # Build a 1-env template DevicePolicy on target device to share frozen B8 + FRP3 weights.
    _, template_policy, _ = build_watch6_v2_sim_and_policy(
        [roots[0].seed],
        b8_checkpoint=b8_checkpoint,
        frp3_checkpoint=frp3_checkpoint,
        device=target,
        max_capacity=max_capacity,
    )

    legal_arr = np.stack([r.legal_mask for r in roots], axis=0).astype(bool)
    b8_arr = np.asarray([r.b8_action for r in roots], dtype=np.int64)

    logical_steps = 0
    executed_steps = 0
    last_log = time.perf_counter()

    def _run_suffix_group(
        phase_name: str, suffixes: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal logical_steps, executed_steps, last_log
        k_count = len(suffixes)
        surv_raw = np.full((n_roots, 6, k_count), np.nan, dtype=np.float64)
        mass_raw = np.full((n_roots, 6, k_count), np.nan, dtype=np.float64)
        jobs: list[_RootBranchJob] = []
        for r_idx, snap in enumerate(roots):
            leg_actions = np.flatnonzero(snap.legal_mask).tolist()
            for s_off, s_glob in enumerate(suffixes):
                crn_key = derive_crn_rng_key(namespace, snap.root_uid, int(s_glob))
                for act in leg_actions:
                    jobs.append(
                        _RootBranchJob(
                            root_idx=r_idx,
                            action=int(act),
                            suffix_offset=s_off,
                            rng_key=crn_key,
                        )
                    )

        for start in range(0, len(jobs), max_slab_envs):
            slab_jobs = jobs[start : start + max_slab_envs]
            b = len(slab_jobs)
            slab_snaps = [roots[j.root_idx] for j in slab_jobs]
            slab_keys = [j.rng_key for j in slab_jobs]
            slab_sim, slab_policy = restore_roots_into_slab(
                slab_snaps,
                template_policy=template_policy,
                device=target,
                max_capacity=max_capacity,
                rng_keys=slab_keys,
            )

            active_slots = torch.arange(b, dtype=torch.long, device=target)
            surv_slab = torch.zeros(b, dtype=torch.float64, device=target)
            mass_sum_slab = torch.zeros(b, dtype=torch.float64, device=target)
            forced0 = torch.tensor([j.action for j in slab_jobs], dtype=torch.long, device=target)

            for h in range(int(horizon)):
                if h == 0:
                    hero_act = forced0
                else:
                    if (
                        compaction_interval > 0
                        and (h % compaction_interval == 0)
                        and not bool(slab_sim.alive[:, 0].all().item())
                    ):
                        keep = slab_sim.alive[:, 0].nonzero(as_tuple=False).flatten()
                        if keep.numel() == 0:
                            break
                        _compact_branch_slab(slab_sim, slab_policy, keep)
                        active_slots = active_slots.index_select(0, keep)
                    slab_sim.prepare()
                    _, _, hero_act = slab_policy.evaluate(slab_sim)

                full_act = slab_policy.actions(slab_sim, hero_act)
                slab_sim.step(full_act)
                slab_policy.after_step(slab_sim)
                executed_steps += int(slab_sim.E)

                alive_now = slab_sim.alive[:, 0]
                step_mass = torch.where(
                    alive_now,
                    slab_sim.length[:, 0].to(dtype=torch.float64),
                    torch.zeros(slab_sim.E, dtype=torch.float64, device=target),
                )
                mass_sum_slab.index_add_(0, active_slots, step_mass)
                if h == int(horizon) - 1:
                    surv_slab.index_copy_(0, active_slots, alive_now.to(dtype=torch.float64))

            if bool(slab_sim.capacity_overflow.any().item()):
                raise RuntimeError("DeviceSim capacity overflow during H300 root branching")

            logical_steps += b * int(horizon)
            s_np = surv_slab.detach().cpu().numpy()
            m_np = (mass_sum_slab / float(horizon)).detach().cpu().numpy()
            for pos_i, job in enumerate(slab_jobs):
                surv_raw[job.root_idx, job.action, job.suffix_offset] = s_np[pos_i]
                mass_raw[job.root_idx, job.action, job.suffix_offset] = m_np[pos_i]

            now = time.perf_counter()
            if log_progress is not None and (now - last_log >= 90.0):
                last_log = now
                log_progress(
                    {
                        "phase": f"branch_{phase_name}",
                        "jobs_done": min(start + max_slab_envs, len(jobs)),
                        "jobs_total": len(jobs),
                        "logical_steps": logical_steps,
                        "executed_steps": executed_steps,
                        "elapsed_sec": round(now - t0, 2),
                    }
                )

        s_mean = np.where(legal_arr, np.nanmean(surv_raw, axis=2), np.nan)
        m_mean = np.where(legal_arr, np.nanmean(mass_raw, axis=2), np.nan)
        return s_mean, m_mean

    s_sel, m_sel = _run_suffix_group("selection", selection_suffixes)
    q_sel, oracle_a_star, _ = select_a_star_and_forbidden(
        s_sel=s_sel,
        m_sel=m_sel,
        legal_mask=legal_arr,
        b8_actions=b8_arr,
    )

    if scoring_suffixes:
        s_score, m_score = _run_suffix_group("scoring", scoring_suffixes)
        q_score = compute_q_from_survival_and_mass(s_score, m_score, legal_arr)
    else:
        # For train-only auxiliary branch supervision, scoring_suffixes may be empty.
        s_score, m_score, q_score = s_sel.copy(), m_sel.copy(), q_sel.copy()

    wall_s = time.perf_counter() - t0
    return BranchedRootsTable(
        num_roots=n_roots,
        horizon=int(horizon),
        legal_mask=legal_arr,
        b8_action=b8_arr,
        s_sel=s_sel,
        m_sel=m_sel,
        q_sel=q_sel,
        oracle_a_star=oracle_a_star,
        s_score=s_score,
        m_score=m_score,
        q_score=q_score,
        telemetry={
            "num_roots": n_roots,
            "horizon": int(horizon),
            "selection_suffixes": [int(x) for x in selection_suffixes],
            "scoring_suffixes": [int(x) for x in scoring_suffixes],
            "logical_branch_steps": logical_steps,
            "executed_sim_steps": executed_steps,
            "compaction_savings_ratio": round(1.0 - (executed_steps / max(logical_steps, 1)), 4),
            "wall_sec": round(wall_s, 3),
        },
    )
