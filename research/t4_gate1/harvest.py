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


@dataclass(frozen=True)
class WorldPartitionSpec:
    """Specification for one world-disjoint partition inside a batched harvest pass."""

    partition: str
    world_rows: Sequence[Mapping[str, Any]]
    max_roots: int = 128


class _PreparedSeamRingBuffer:
    """Device-resident circular buffer of prepared decision-seam states (zero per-tick D2H syncs).

    Records compact device tensor copies for root-enabled worlds at every tick in O(1) GPU memory
    bandwidth and materializes ``PreparedRootSnapshot`` objects to CPU strictly on actual
    ordinary-stride or death-lookback hits (<0.5% of mature env-ticks).
    """

    def __init__(
        self,
        *,
        ring_span: int,
        env_indices: torch.Tensor,
        sim: DeviceSim,
        policy: DevicePolicy,
    ) -> None:
        self.ring_span = int(ring_span)
        self.env_idx = env_indices.to(device=sim.device, dtype=torch.long).reshape(-1)
        self.e_root = int(self.env_idx.numel())
        self.cap = int(sim.cap)
        self.food_cap = int(sim.food_capacity)
        dev = sim.device
        rs = self.ring_span
        er = self.e_root

        self.valid = torch.zeros((rs, er), dtype=torch.bool, device=dev)
        self.tick_buf = torch.full((rs, er), -1, dtype=torch.int32, device=dev)
        self.h_rem_buf = torch.zeros((rs, er), dtype=torch.int32, device=dev)
        self.life_id_buf = torch.zeros((rs, er), dtype=torch.int32, device=dev)
        self.b8_act_buf = torch.zeros((rs, er), dtype=torch.long, device=dev)
        self.legal_buf = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)

        self.bodies = torch.zeros((rs, er, 6, self.cap, 2), dtype=torch.int16, device=dev)
        self.head_ptr = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.seg_count = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.length = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.alive = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)
        self.direction = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.boost_frames = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.frames_since_food = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.respawn_timer = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.reward_prev_length = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.frame = torch.zeros((rs, er), dtype=torch.long, device=dev)
        self.boosted_this_step = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)
        self.last_reward = torch.zeros((rs, er, 6), dtype=torch.float32, device=dev)
        self.last_done = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)
        self.last_death_cause = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.last_kills = torch.zeros((rs, er, 6), dtype=torch.long, device=dev)
        self.last_food_ate = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)
        self.last_ambient_food_ate = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)
        self.last_corpse_food_ate = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)
        self.last_transition_valid = torch.zeros((rs, er, 6), dtype=torch.bool, device=dev)
        self.food_count = torch.zeros((rs, er), dtype=torch.long, device=dev)
        self.food_cells = torch.zeros((rs, er, self.food_cap, 2), dtype=torch.int16, device=dev)
        self.corpse_food = torch.zeros((rs, er, self.food_cap), dtype=torch.bool, device=dev)

        vr = policy.vector_runtime
        self.vec_tensors: dict[str, torch.Tensor] = {}
        for attr in _VECTOR_RUNTIME_ENV_TENSORS:
            t_val = getattr(vr, attr)
            self.vec_tensors[attr] = torch.zeros(
                (rs, er, *t_val.shape[1:]), dtype=t_val.dtype, device=dev
            )

    @torch.inference_mode()
    def record_tick(
        self,
        slot: int,
        sim: DeviceSim,
        policy: DevicePolicy,
        *,
        mature_r: torch.Tensor,
        tick: int,
        h_rem: int,
        life_ids_r: torch.Tensor,
        b8_act_r: torch.Tensor,
        legal_r: torch.Tensor,
    ) -> None:
        if not sim._prepared or not policy.vector_prepared:
            raise ValueError("_PreparedSeamRingBuffer requires prepared DeviceSim and DevicePolicy")
        idx = self.env_idx
        self.valid[slot].copy_(mature_r)
        self.tick_buf[slot].fill_(int(tick))
        self.h_rem_buf[slot].fill_(int(h_rem))
        self.life_id_buf[slot].copy_(life_ids_r)
        self.b8_act_buf[slot].copy_(b8_act_r)
        self.legal_buf[slot].copy_(legal_r)

        self.bodies[slot].copy_(sim.bodies.index_select(0, idx).to(torch.int16))
        self.head_ptr[slot].copy_(sim.head_ptr.index_select(0, idx))
        self.seg_count[slot].copy_(sim.seg_count.index_select(0, idx))
        self.length[slot].copy_(sim.length.index_select(0, idx))
        self.alive[slot].copy_(sim.alive.index_select(0, idx))
        self.direction[slot].copy_(sim.direction.index_select(0, idx))
        self.boost_frames[slot].copy_(sim.boost_frames.index_select(0, idx))
        self.frames_since_food[slot].copy_(sim.frames_since_food.index_select(0, idx))
        self.respawn_timer[slot].copy_(sim.respawn_timer.index_select(0, idx))
        self.reward_prev_length[slot].copy_(sim._reward_prev_length.index_select(0, idx))
        self.frame[slot].copy_(sim.frame.index_select(0, idx))
        self.boosted_this_step[slot].copy_(sim._boosted_this_step.index_select(0, idx))
        self.last_reward[slot].copy_(sim._last_reward.index_select(0, idx))
        self.last_done[slot].copy_(sim._last_done.index_select(0, idx))
        self.last_death_cause[slot].copy_(sim._last_death_cause.index_select(0, idx))
        self.last_kills[slot].copy_(sim._last_kills.index_select(0, idx))
        self.last_food_ate[slot].copy_(sim._last_food_ate.index_select(0, idx))
        self.last_ambient_food_ate[slot].copy_(sim._last_ambient_food_ate.index_select(0, idx))
        self.last_corpse_food_ate[slot].copy_(sim._last_corpse_food_ate.index_select(0, idx))
        self.last_transition_valid[slot].copy_(sim._last_transition_valid.index_select(0, idx))
        self.food_count[slot].copy_(sim.food_count.index_select(0, idx))
        self.food_cells[slot].copy_(sim.food_cells.index_select(0, idx).to(torch.int16))
        self.corpse_food[slot].copy_(sim.corpse_food.index_select(0, idx))

        vr = policy.vector_runtime
        for attr in _VECTOR_RUNTIME_ENV_TENSORS:
            self.vec_tensors[attr][slot].copy_(getattr(vr, attr).index_select(0, idx))

    @torch.inference_mode()
    def extract_snapshot(
        self,
        slot: int,
        r_pos: int,
        *,
        root_uid: str,
        kind: str,
        lookback: int | None,
        world_id: int,
        seed: int,
    ) -> PreparedRootSnapshot:
        sc = self.seg_count[slot, r_pos].detach().cpu().numpy().astype(np.int64)
        hp = self.head_ptr[slot, r_pos].detach().cpu().numpy().astype(np.int64)
        bodies_env = self.bodies[slot, r_pos].detach().cpu().numpy()  # (6, cap, 2) int16
        segs: list[np.ndarray] = []
        for s in range(6):
            n = int(sc[s])
            if n > 0:
                rings = (int(hp[s]) - np.arange(n, dtype=np.int64)) % self.cap
                segs.append(bodies_env[s, rings].astype(np.int16).copy())
            else:
                segs.append(np.zeros((0, 2), dtype=np.int16))

        fc = int(self.food_count[slot, r_pos].item())
        f_cells = self.food_cells[slot, r_pos, :fc].detach().cpu().numpy().astype(np.int16).copy()
        f_corpse = self.corpse_food[slot, r_pos, :fc].detach().cpu().numpy().astype(bool).copy()

        leg = self.legal_buf[slot, r_pos].detach().cpu().numpy().astype(bool).reshape(6).copy()
        if not leg.any():
            leg[:3] = True

        vec_dict: dict[str, np.ndarray] = {
            attr: self.vec_tensors[attr][slot, r_pos].detach().cpu().numpy().copy()
            for attr in _VECTOR_RUNTIME_ENV_TENSORS
        }
        ln = self.length[slot, r_pos].detach().cpu().numpy().astype(np.int64)
        return PreparedRootSnapshot(
            root_uid=root_uid,
            kind=kind,
            lookback=lookback,
            world_id=int(world_id),
            seed=int(seed),
            life_id=int(self.life_id_buf[slot, r_pos].item()),
            tick=int(self.tick_buf[slot, r_pos].item()),
            h_rem=int(self.h_rem_buf[slot, r_pos].item()),
            hero_length=int(ln[0]),
            b8_action=int(self.b8_act_buf[slot, r_pos].item()),
            legal_mask=leg,
            body_segments=tuple(segs),
            seg_count=sc,
            length=ln,
            alive=self.alive[slot, r_pos].detach().cpu().numpy().astype(bool).copy(),
            direction=self.direction[slot, r_pos].detach().cpu().numpy().astype(np.int64).copy(),
            boost_frames=(
                self.boost_frames[slot, r_pos].detach().cpu().numpy().astype(np.int64).copy()
            ),
            frames_since_food=(
                self.frames_since_food[slot, r_pos].detach().cpu().numpy().astype(np.int64).copy()
            ),
            respawn_timer=(
                self.respawn_timer[slot, r_pos].detach().cpu().numpy().astype(np.int64).copy()
            ),
            reward_prev_length=(
                self.reward_prev_length[slot, r_pos].detach().cpu().numpy().astype(np.int64).copy()
            ),
            frame=int(self.frame[slot, r_pos].item()),
            boosted_this_step=(
                self.boosted_this_step[slot, r_pos].detach().cpu().numpy().astype(bool).copy()
            ),
            last_reward=(
                self.last_reward[slot, r_pos].detach().cpu().numpy().astype(np.float32).copy()
            ),
            last_done=self.last_done[slot, r_pos].detach().cpu().numpy().astype(bool).copy(),
            last_death_cause=(
                self.last_death_cause[slot, r_pos].detach().cpu().numpy().astype(np.int64).copy()
            ),
            last_kills=self.last_kills[slot, r_pos].detach().cpu().numpy().astype(np.int64).copy(),
            last_food_ate=(
                self.last_food_ate[slot, r_pos].detach().cpu().numpy().astype(bool).copy()
            ),
            last_ambient_food_ate=(
                self.last_ambient_food_ate[slot, r_pos].detach().cpu().numpy().astype(bool).copy()
            ),
            last_corpse_food_ate=(
                self.last_corpse_food_ate[slot, r_pos].detach().cpu().numpy().astype(bool).copy()
            ),
            last_transition_valid=(
                self.last_transition_valid[slot, r_pos].detach().cpu().numpy().astype(bool).copy()
            ),
            food_cells=f_cells,
            corpse_food=f_corpse,
            vec_tensors=vec_dict,
        )


@torch.inference_mode()
def harvest_world_partitions_batched(
    *,
    specs: Sequence[WorldPartitionSpec],
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    session_ticks: int,
    max_slab_envs: int = 128,
    max_capacity: int = 3200,
    stride_mature: int = 4,
    stride_early: int = 16,
    min_mature_length: int = 600,
    branch_horizon: int = 300,
    death_lookbacks: Sequence[int] = DEFAULT_DEATH_LOOKBACKS,
    ordinary_root_stride: int = 96,
    device: str | torch.device = "cpu",
    log_progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[HarvestedWorldPartition]:
    """Co-simulate one or more world-disjoint partitions in large GPU slabs while keeping datasets
    and root pools strictly isolated per partition (e.g. ``fit`` 102 + ``val`` 26 = 128 in 1 slab).
    """
    t0 = time.perf_counter()
    target = torch.device("cuda:0" if str(device) == "cuda" else device)
    spec_list = list(specs)
    if not spec_list:
        raise ValueError("specs must be nonempty")

    n_parts = len(spec_list)
    part_rows: list[list[Mapping[str, Any]]] = []
    seen_seeds: set[int] = set()
    seen_wids: set[int] = set()
    combined_items: list[tuple[int, Mapping[str, Any]]] = []

    for p_idx, sp in enumerate(spec_list):
        rows_p = list(sp.world_rows)
        if not rows_p:
            raise ValueError(f"world_rows for partition {sp.partition!r} must be nonempty")
        for r in rows_p:
            w_int = int(r["global_index"])
            s_int = int(r["seed"])
            if s_int in seen_seeds or w_int in seen_wids:
                raise ValueError(
                    f"Overlapping world seed/global_index in batched harvest ({sp.partition!r})"
                )
            seen_seeds.add(s_int)
            seen_wids.add(w_int)
            combined_items.append((p_idx, r))
        part_rows.append(rows_p)

    builders = [CompactSampleBuilder() for _ in range(n_parts)]
    target_tau_chunks: list[list[np.ndarray]] = [[] for _ in range(n_parts)]
    target_died_chunks: list[list[np.ndarray]] = [[] for _ in range(n_parts)]
    target_g_chunks: list[list[np.ndarray]] = [[] for _ in range(n_parts)]
    target_fs_chunks: list[list[np.ndarray]] = [[] for _ in range(n_parts)]
    target_fw_chunks: list[list[np.ndarray]] = [[] for _ in range(n_parts)]
    target_fl_chunks: list[list[np.ndarray]] = [[] for _ in range(n_parts)]
    target_flv_chunks: list[list[np.ndarray]] = [[] for _ in range(n_parts)]

    death_candidates: list[list[PreparedRootSnapshot]] = [[] for _ in range(n_parts)]
    ordinary_candidates: list[list[PreparedRootSnapshot]] = [[] for _ in range(n_parts)]
    total_hero_deaths = [0] * n_parts
    total_mature_ticks = [0] * n_parts
    total_alive_ticks = [0] * n_parts

    lookbacks = tuple(sorted(int(lb) for lb in death_lookbacks if int(lb) > 0))
    phase_label = (
        f"harvest_{spec_list[0].partition}"
        if n_parts == 1
        else f"harvest_{'+'.join(sp.partition for sp in spec_list)}"
    )
    template_policy: DevicePolicy | None = None

    for slab_start in range(0, len(combined_items), max_slab_envs):
        slab_items = combined_items[slab_start : slab_start + max_slab_envs]
        e_count = len(slab_items)
        slab_pidxs = np.asarray([p_i for p_i, _ in slab_items], dtype=np.int32)
        slab_seeds = [int(r["seed"]) for _, r in slab_items]
        slab_wids = np.asarray([int(r["global_index"]) for _, r in slab_items], dtype=np.int32)
        slab_seeds_np = np.asarray(slab_seeds, dtype=np.int64)
        slab_part_names = [spec_list[int(p_i)].partition for p_i in slab_pidxs]

        sim, policy, _ = build_watch6_v2_sim_and_policy(
            slab_seeds,
            b8_checkpoint=b8_checkpoint,
            frp3_checkpoint=frp3_checkpoint,
            device=target,
            max_capacity=max_capacity,
        )
        if template_policy is None:
            template_policy = policy

        acted_seq_t = torch.zeros((session_ticks, e_count), dtype=torch.bool, device=target)
        died_seq_t = torch.zeros((session_ticks, e_count), dtype=torch.bool, device=target)
        pre_len_seq_t = torch.zeros((session_ticks, e_count), dtype=torch.int32, device=target)
        post_len_seq_t = torch.zeros((session_ticks, e_count), dtype=torch.int32, device=target)

        sampled_t_by_p: list[list[np.ndarray]] = [[] for _ in range(n_parts)]
        sampled_e_by_p: list[list[np.ndarray]] = [[] for _ in range(n_parts)]

        life_ids_t = torch.zeros(e_count, dtype=torch.int32, device=target)
        prev_alive_t = torch.ones(e_count, dtype=torch.bool, device=target)
        slab_wids_t = torch.as_tensor(slab_wids, dtype=torch.long, device=target)

        root_env_offs = np.asarray(
            [e_i for e_i, p_i in enumerate(slab_pidxs) if spec_list[int(p_i)].max_roots > 0],
            dtype=np.int64,
        )
        ring_span = (max(lookbacks) + 1) if (lookbacks and root_env_offs.size > 0) else 0
        if ring_span > 0:
            root_env_t = torch.as_tensor(root_env_offs, dtype=torch.long, device=target)
            root_wids_t = slab_wids_t.index_select(0, root_env_t)
            ring_buf: _PreparedSeamRingBuffer | None = _PreparedSeamRingBuffer(
                ring_span=ring_span,
                env_indices=root_env_t,
                sim=sim,
                policy=policy,
            )
        else:
            root_env_t = torch.zeros(0, dtype=torch.long, device=target)
            root_wids_t = torch.zeros(0, dtype=torch.long, device=target)
            ring_buf = None

        last_log = time.perf_counter()
        for tick in range(int(session_ticks)):
            h_rem = int(session_ticks - tick)
            sim.prepare()

            alive_t = sim.alive[:, 0]
            respawned_t = (~prev_alive_t) & alive_t
            life_ids_t = life_ids_t + respawned_t.to(torch.int32)
            prev_alive_t = alive_t.clone()

            _, legal_t, hero_act_t = policy.evaluate(sim)
            full_actions_t = policy.actions(sim, hero_act_t)

            len_t = sim.length[:, 0].to(torch.int32)
            acted_seq_t[tick] = alive_t
            pre_len_seq_t[tick] = len_t
            mature_t = alive_t & (len_t >= int(min_mature_length))

            # Strictly causal, outcome-blind trajectory sampling grid (offset by world_id % stride).
            take_mature_t = mature_t & (((tick + slab_wids_t) % int(stride_mature)) == 0)
            take_early_t = alive_t & (~mature_t) & (((tick + slab_wids_t) % int(stride_early)) == 0)
            take_t = take_mature_t | take_early_t

            ord_full_t = torch.zeros(e_count, dtype=torch.bool, device=target)
            if ring_buf is not None:
                slot_idx = tick % ring_span
                mature_r_t = mature_t.index_select(0, root_env_t)
                ring_buf.record_tick(
                    slot_idx,
                    sim,
                    policy,
                    mature_r=mature_r_t,
                    tick=tick,
                    h_rem=h_rem,
                    life_ids_r=life_ids_t.index_select(0, root_env_t),
                    b8_act_r=hero_act_t.index_select(0, root_env_t).to(torch.long),
                    legal_r=legal_t.index_select(0, root_env_t),
                )
                if ordinary_root_stride > 0 and h_rem >= int(branch_horizon):
                    ord_r_t = mature_r_t & (
                        ((tick + root_wids_t * 7) % int(ordinary_root_stride)) == 0
                    )
                    ord_full_t.index_copy_(0, root_env_t, ord_r_t)

            # Single tiny (2, E) boolean mask D2H check for pre-step sampling & ordinary roots.
            pre_events_np = torch.stack((take_t, ord_full_t), dim=0).detach().cpu().numpy()
            take_np = pre_events_np[0]
            ord_np = pre_events_np[1]

            if bool(take_np.any()):
                life_np_cur = life_ids_t.detach().cpu().numpy().astype(np.int32)
                b8_act_np_cur = hero_act_t.detach().cpu().numpy().astype(np.int64)
                legal_np_cur = legal_t.detach().cpu().numpy().astype(bool)
                for p_i in range(n_parts):
                    p_take = take_np & (slab_pidxs == p_i)
                    if not bool(p_take.any()):
                        continue
                    env_offs = np.flatnonzero(p_take)
                    env_t = torch.as_tensor(env_offs, dtype=torch.long, device=target)
                    builders[p_i].append_from_sim(
                        sim,
                        env_indices=env_t,
                        world_ids=slab_wids[env_offs],
                        seeds=slab_seeds_np[env_offs],
                        life_ids=life_np_cur[env_offs],
                        ticks=np.full(env_offs.size, tick, dtype=np.int32),
                        h_rems=np.full(env_offs.size, h_rem, dtype=np.int32),
                        b8_actions=b8_act_np_cur[env_offs],
                        legal_masks=legal_np_cur[env_offs],
                    )
                    sampled_t_by_p[p_i].append(np.full(env_offs.size, tick, dtype=np.int32))
                    sampled_e_by_p[p_i].append(env_offs.astype(np.int32))

            if ring_buf is not None and bool(ord_np.any()):
                slot_idx = tick % ring_span
                for r_pos, env_i in enumerate(root_env_offs.tolist()):
                    if not ord_np[env_i]:
                        continue
                    p_i = int(slab_pidxs[env_i])
                    life_val = int(ring_buf.life_id_buf[slot_idx, r_pos].item())
                    uid = f"{slab_part_names[env_i]}:w{int(slab_wids[env_i])}:l{life_val}:t{tick}"
                    snap = ring_buf.extract_snapshot(
                        slot_idx,
                        r_pos,
                        root_uid=uid,
                        kind="ordinary",
                        lookback=None,
                        world_id=int(slab_wids[env_i]),
                        seed=int(slab_seeds_np[env_i]),
                    )
                    ordinary_candidates[p_i].append(snap)

            sim.step(full_actions_t)
            policy.after_step(sim)

            died_now_t = sim._last_done[:, 0]
            post_len_seq_t[tick] = sim.length[:, 0].to(torch.int32)
            died_seq_t[tick] = died_now_t

            if ring_buf is not None:
                died_r_np = died_now_t.index_select(0, root_env_t).detach().cpu().numpy()
                if bool(died_r_np.any()):
                    cur_lives_r = (
                        life_ids_t.index_select(0, root_env_t)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )
                    for r_pos in np.flatnonzero(died_r_np).tolist():
                        env_i = int(root_env_offs[r_pos])
                        p_i = int(slab_pidxs[env_i])
                        cur_life = int(cur_lives_r[r_pos])
                        for lb in lookbacks:
                            src_tick = tick - lb
                            if src_tick < 0:
                                continue
                            src_slot = src_tick % ring_span
                            if (
                                bool(ring_buf.valid[src_slot, r_pos].item())
                                and int(ring_buf.tick_buf[src_slot, r_pos].item()) == src_tick
                                and int(ring_buf.life_id_buf[src_slot, r_pos].item()) == cur_life
                                and int(ring_buf.length[src_slot, r_pos, 0].item())
                                >= int(min_mature_length)
                            ):
                                base_uid = (
                                    f"{slab_part_names[env_i]}:w{int(slab_wids[env_i])}:"
                                    f"l{cur_life}:t{src_tick}"
                                )
                                snap = ring_buf.extract_snapshot(
                                    src_slot,
                                    r_pos,
                                    root_uid=f"{base_uid}:death_lb{lb}",
                                    kind="death_lookback",
                                    lookback=int(lb),
                                    world_id=int(slab_wids[env_i]),
                                    seed=int(slab_seeds_np[env_i]),
                                )
                                death_candidates[p_i].append(snap)

            now = time.perf_counter()
            if log_progress is not None and (now - last_log >= 90.0):
                last_log = now
                log_progress(
                    {
                        "phase": phase_label,
                        "slab_start": slab_start,
                        "slab_envs": e_count,
                        "tick": tick + 1,
                        "session_ticks": int(session_ticks),
                        "sampled_rows": sum(b.num_rows for b in builders),
                        "death_root_candidates": sum(len(dc) for dc in death_candidates),
                        "ordinary_root_candidates": sum(len(oc) for oc in ordinary_candidates),
                        "elapsed_sec": round(now - t0, 2),
                    }
                )

        if bool(sim.capacity_overflow.any().item()):
            raise RuntimeError(
                f"DeviceSim body ring capacity overflow during {phase_label} "
                f"(max_capacity={max_capacity})"
            )

        acted_seq = acted_seq_t.detach().cpu().numpy().astype(bool)
        died_seq = died_seq_t.detach().cpu().numpy().astype(bool)
        pre_len_seq = pre_len_seq_t.detach().cpu().numpy().astype(np.int32)
        post_len_seq = post_len_seq_t.detach().cpu().numpy().astype(np.int32)
        mature_seq = acted_seq & (pre_len_seq >= int(min_mature_length))

        targets = compute_per_life_targets(
            acted=acted_seq,
            died=died_seq,
            pre_length=pre_len_seq,
            post_length=post_len_seq,
            gamma_f=FORAGING_GAMMA,
        )
        for p_i in range(n_parts):
            p_env_mask = slab_pidxs == p_i
            if bool(p_env_mask.any()):
                total_alive_ticks[p_i] += int(acted_seq[:, p_env_mask].sum())
                total_mature_ticks[p_i] += int(mature_seq[:, p_env_mask].sum())
                total_hero_deaths[p_i] += int(died_seq[:, p_env_mask].sum())
            if sampled_t_by_p[p_i]:
                st = np.concatenate(sampled_t_by_p[p_i], axis=0)
                se = np.concatenate(sampled_e_by_p[p_i], axis=0)
                target_tau_chunks[p_i].append(targets["tau_alive"][st, se])
                target_died_chunks[p_i].append(targets["died_in_session"][st, se])
                target_g_chunks[p_i].append(targets["return_g"][st, se])
                target_fs_chunks[p_i].append(targets["forage_short"][st, se])
                target_fw_chunks[p_i].append(targets["forage_short_weight"][st, se])
                target_fl_chunks[p_i].append(targets["forage_long"][st, se])
                target_flv_chunks[p_i].append(targets["forage_long_valid"][st, se])

    wall_s = time.perf_counter() - t0
    total_worlds_all = max(sum(len(rows_p) for rows_p in part_rows), 1)
    results: list[HarvestedWorldPartition] = []

    for p_i, sp in enumerate(spec_list):
        rows_p = part_rows[p_i]
        builder_p = builders[p_i]
        if builder_p.num_rows > 0:
            dataset_p = builder_p.build(
                tau_alive=np.concatenate(target_tau_chunks[p_i], axis=0),
                died_in_session=np.concatenate(target_died_chunks[p_i], axis=0),
                return_g=np.concatenate(target_g_chunks[p_i], axis=0),
                forage_short=np.concatenate(target_fs_chunks[p_i], axis=0),
                forage_short_weight=np.concatenate(target_fw_chunks[p_i], axis=0),
                forage_long=np.concatenate(target_fl_chunks[p_i], axis=0),
                forage_long_valid=np.concatenate(target_flv_chunks[p_i], axis=0),
            )
        else:
            dataset_p = builder_p.build()

        selected_roots = _select_balanced_roots(
            death_candidates[p_i],
            ordinary_candidates[p_i],
            max_roots=int(sp.max_roots),
        )

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

        part_wall = wall_s * (len(rows_p) / float(total_worlds_all))
        world_ticks = len(rows_p) * int(session_ticks)
        telemetry = {
            "partition": sp.partition,
            "num_worlds": len(rows_p),
            "session_ticks": int(session_ticks),
            "world_ticks": world_ticks,
            "wall_sec": round(part_wall, 3),
            "batch_wall_sec": round(wall_s, 3),
            "batched_partitions": [s.partition for s in spec_list],
            "world_ticks_per_sec": round(world_ticks / max(part_wall, 1e-6), 2),
            "sampled_rows": int(dataset_p.num_rows),
            "sampled_mature_rows": int(np.sum(dataset_p.snake_length[:, 0] >= min_mature_length)),
            "total_alive_ticks": int(total_alive_ticks[p_i]),
            "total_mature_ticks": int(total_mature_ticks[p_i]),
            "total_hero_deaths": int(total_hero_deaths[p_i]),
            "death_root_candidates": len(death_candidates[p_i]),
            "ordinary_root_candidates": len(ordinary_candidates[p_i]),
            "selected_roots": len(selected_roots),
            "selected_death_roots": sum(1 for r in selected_roots if r.kind == "death_lookback"),
            "selected_ordinary_roots": sum(1 for r in selected_roots if r.kind == "ordinary"),
        }
        results.append(
            HarvestedWorldPartition(
                partition=sp.partition,
                world_ids=[int(r["global_index"]) for r in rows_p],
                seeds=[int(r["seed"]) for r in rows_p],
                dataset=dataset_p,
                roots=selected_roots,
                roots_dataset=roots_dataset,
                telemetry=telemetry,
            )
        )
    return results


@torch.inference_mode()
def harvest_world_partition(
    *,
    partition: str,
    world_rows: Sequence[Mapping[str, Any]],
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    session_ticks: int,
    max_slab_envs: int = 128,
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
    return harvest_world_partitions_batched(
        specs=[
            WorldPartitionSpec(
                partition=partition,
                world_rows=world_rows,
                max_roots=max_roots,
            )
        ],
        b8_checkpoint=b8_checkpoint,
        frp3_checkpoint=frp3_checkpoint,
        session_ticks=session_ticks,
        max_slab_envs=max_slab_envs,
        max_capacity=max_capacity,
        stride_mature=stride_mature,
        stride_early=stride_early,
        min_mature_length=min_mature_length,
        branch_horizon=branch_horizon,
        death_lookbacks=death_lookbacks,
        ordinary_root_stride=ordinary_root_stride,
        device=device,
        log_progress=log_progress,
    )[0]


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
    group_id: int  # 0 = selection, 1 = scoring
    root_idx: int
    action: int
    suffix_offset: int
    rng_key: int


def _reduce_legal_branch_outcomes(
    surv_raw: np.ndarray,
    mass_raw: np.ndarray,
    legal_arr: np.ndarray,
    *,
    phase_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Average branch outcomes over legal actions without ``np.nanmean`` empty-slice warnings."""
    leg = np.asarray(legal_arr, dtype=bool)
    if leg.ndim != 2 or leg.shape[1] != 6:
        raise ValueError(f"Expected (R, 6) legal_arr in {phase_name}, got {leg.shape}")
    n_roots = leg.shape[0]
    if n_roots == 0:
        empty = np.zeros((0, 6), dtype=np.float64)
        return empty, empty.copy()
    if surv_raw.ndim != 3 or surv_raw.shape[:2] != (n_roots, 6) or surv_raw.shape[2] <= 0:
        raise ValueError(f"Invalid surv_raw shape {surv_raw.shape} in {phase_name}")
    if mass_raw.shape != surv_raw.shape:
        raise ValueError(f"mass_raw shape {mass_raw.shape} != surv_raw shape {surv_raw.shape}")
    if not bool(np.all(np.any(leg, axis=1))):
        raise ValueError(f"Root with zero legal actions encountered in {phase_name}")

    surv_leg = surv_raw[leg]
    mass_leg = mass_raw[leg]
    if not (bool(np.isfinite(surv_leg).all()) and bool(np.isfinite(mass_leg).all())):
        raise ValueError(
            f"Non-finite survival or mass outcome on legal action during {phase_name} branching"
        )

    s_mean = np.full((n_roots, 6), np.nan, dtype=np.float64)
    m_mean = np.full((n_roots, 6), np.nan, dtype=np.float64)
    s_mean[leg] = np.mean(surv_leg, axis=1)
    m_mean[leg] = np.mean(mass_leg, axis=1)
    return s_mean, m_mean


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
    max_slab_envs: int = 512,
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

    k_sel = len(selection_suffixes)
    if k_sel <= 0:
        raise ValueError("suffixes for selection must be nonempty")
    k_score = len(scoring_suffixes)
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
    if not bool(np.all(np.any(legal_arr, axis=1))):
        raise ValueError("Every branched root must have at least one legal action")
    b8_arr = np.asarray([r.b8_action for r in roots], dtype=np.int64)

    surv_sel_raw = np.full((n_roots, 6, k_sel), np.nan, dtype=np.float64)
    mass_sel_raw = np.full((n_roots, 6, k_sel), np.nan, dtype=np.float64)
    surv_score_raw = (
        np.full((n_roots, 6, k_score), np.nan, dtype=np.float64) if k_score > 0 else None
    )
    mass_score_raw = (
        np.full((n_roots, 6, k_score), np.nan, dtype=np.float64) if k_score > 0 else None
    )

    # Batch selection (group 0) and scoring (group 1) jobs together into large independent slabs.
    jobs: list[_RootBranchJob] = []
    for r_idx, snap in enumerate(roots):
        leg_actions = np.flatnonzero(snap.legal_mask).tolist()
        for s_off, s_glob in enumerate(selection_suffixes):
            crn_key = derive_crn_rng_key(namespace, snap.root_uid, int(s_glob))
            for act in leg_actions:
                jobs.append(
                    _RootBranchJob(
                        group_id=0,
                        root_idx=r_idx,
                        action=int(act),
                        suffix_offset=s_off,
                        rng_key=crn_key,
                    )
                )
        for s_off, s_glob in enumerate(scoring_suffixes):
            crn_key = derive_crn_rng_key(namespace, snap.root_uid, int(s_glob))
            for act in leg_actions:
                jobs.append(
                    _RootBranchJob(
                        group_id=1,
                        root_idx=r_idx,
                        action=int(act),
                        suffix_offset=s_off,
                        rng_key=crn_key,
                    )
                )

    logical_steps = 0
    executed_steps = 0
    last_log = time.perf_counter()
    phase_name = "selection_and_scoring" if k_score > 0 else "selection"

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
            if job.group_id == 0:
                surv_sel_raw[job.root_idx, job.action, job.suffix_offset] = s_np[pos_i]
                mass_sel_raw[job.root_idx, job.action, job.suffix_offset] = m_np[pos_i]
            else:
                assert surv_score_raw is not None and mass_score_raw is not None
                surv_score_raw[job.root_idx, job.action, job.suffix_offset] = s_np[pos_i]
                mass_score_raw[job.root_idx, job.action, job.suffix_offset] = m_np[pos_i]

        now = time.perf_counter()
        if log_progress is not None and (now - last_log >= 90.0):
            last_log = now
            log_progress(
                {
                    "phase": f"branch_{phase_name}",
                    "jobs_done": min(start + max_slab_envs, len(jobs)),
                    "jobs_total": len(jobs),
                    "slab_envs": max_slab_envs,
                    "logical_steps": logical_steps,
                    "executed_steps": executed_steps,
                    "elapsed_sec": round(now - t0, 2),
                }
            )

    s_sel, m_sel = _reduce_legal_branch_outcomes(
        surv_sel_raw,
        mass_sel_raw,
        legal_arr,
        phase_name="selection",
    )
    q_sel, oracle_a_star, _ = select_a_star_and_forbidden(
        s_sel=s_sel,
        m_sel=m_sel,
        legal_mask=legal_arr,
        b8_actions=b8_arr,
    )

    if k_score > 0:
        assert surv_score_raw is not None and mass_score_raw is not None
        s_score, m_score = _reduce_legal_branch_outcomes(
            surv_score_raw,
            mass_score_raw,
            legal_arr,
            phase_name="scoring",
        )
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
