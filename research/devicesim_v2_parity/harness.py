"""Exact mechanics-v2 parity harness for DeviceSim and CTDeviceSim vs CPU GridBatchSim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import (
    DEATH_BODY,
    DEATH_HEAD,
    DEATH_SELF,
    DEATH_WALL,
    DeviceSim,
)
from research.gpu_sim_spike.ego_device import build_observation
from research.gpu_sim_spike.v8_device import v8_decide as gpu_v8_decide
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from src.core.mechanics_constants import corpse_food_cap
from src.simd_env import v8_nb
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego2s_policy import Ego2sSimdPolicy, ego_config_for, forward_q
from src.simd_env.ego_raster import build_ego_raster
from src.simd_env.grid_sim import FOOD_AMBIENT, FOOD_CORPSE, GridBatchSim
from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

FLOAT_TIE_MAX_Q_DIFF = 5e-4
FLOAT_TIE_MIN_BAND = 1e-4
REWARD_ATOL = 1e-5
EGO_SCALAR_ATOL = 1e-4


def splitmix_positive_scalar(key: int, counter: int) -> int:
    """Exact scalar counterpart of DeviceSim._splitmix_positive (63-bit positive int)."""
    u64 = 0xFFFF_FFFF_FFFF_FFFF
    z = (int(key) & u64) ^ (((int(counter) & u64) * 0x9E3779B97F4A7C15) & u64)
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & u64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & u64
    return int((z ^ (z >> 31)) & 0x7FFF_FFFF_FFFF_FFFF)


class SplitMixEnvRngProxy:
    """Per-world EnvRng proxy feeding DeviceSim's exact margined SplitMix64 stream to CPU."""

    def __init__(
        self,
        key: int,
        counter: int,
        *,
        game_width: int,
        game_height: int,
        segment_size: int,
        wall_thickness: int,
    ) -> None:
        self.key = int(key)
        self.counter = int(counter)
        self.game_width = int(game_width)
        self.game_height = int(game_height)
        self.segment_size = int(segment_size)
        self.wall_thickness = int(wall_thickness)
        self.grid_w = self.game_width // self.segment_size
        self.grid_h = self.game_height // self.segment_size
        self.margin = max(0, self.wall_thickness // self.segment_size)
        self.span_w = max(1, self.grid_w - 2 * self.margin)
        self.span_h = max(1, self.grid_h - 2 * self.margin)
        self.draws = 0

    def _draw_find_empty_candidate(self) -> tuple[int, int]:
        self.counter += 1
        self.draws += 1
        z = splitmix_positive_scalar(self.key, self.counter)
        cx = self.margin + (z % self.span_w)
        cy = self.margin + ((z >> 32) % self.span_h)
        return (int(cx) * self.segment_size, int(cy) * self.segment_size)


def install_splitmix_rng_proxies(
    cpu_sim: GridBatchSim, gpu_sim: DeviceSim
) -> list[SplitMixEnvRngProxy]:
    """Bind each CPU world's _rngs[e] to gpu_sim's exact (rng_key[e], rng_counter[e])."""
    keys = gpu_sim.rng_key.detach().cpu().tolist()
    counters = gpu_sim.rng_counter.detach().cpu().tolist()
    proxies: list[SplitMixEnvRngProxy] = []
    for env in range(cpu_sim.E):
        proxy = SplitMixEnvRngProxy(
            int(keys[env]),
            int(counters[env]),
            game_width=int(cpu_sim.cfg.game_width),
            game_height=int(cpu_sim.cfg.game_height),
            segment_size=int(cpu_sim.cfg.segment_size),
            wall_thickness=int(cpu_sim.cfg.wall_thickness),
        )
        cpu_sim._rngs[env] = proxy  # type: ignore[assignment]
        proxies.append(proxy)
    return proxies


def watch6_v2_config(num_envs: int, *, max_capacity: int = 1600) -> BatchSimConfig:
    """Product Watch6 mechanics-v2 arena configuration."""
    return BatchSimConfig(
        num_envs=int(num_envs),
        num_snakes=6,
        game_width=1450,
        game_height=830,
        segment_size=10,
        wall_thickness=10,
        initial_food=250,
        max_food=300,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        gamma=1.0,
        arena_type="rectangular",
        max_capacity=int(max_capacity),
        frame_rate=1,
        kill_scale=0.3,
        death_value=-3.0,
    )


def derive_world_seeds(shard_index: int, num_envs: int) -> list[int]:
    """Deterministic disjoint 63-bit seeds per (shard_index, env_offset)."""
    base_key = 0x5632_5041_5249_5459 ^ ((int(shard_index) + 1) * 0x9E37_79B9)
    return [splitmix_positive_scalar(base_key, env_idx + 1) for env_idx in range(int(num_envs))]


def _populate_mature_v2_world(sim: GridBatchSim, env: int, seed: int) -> None:
    """Populate one world at mature v2 steady state (300 ambient + 150 corpse, contest lengths)."""
    rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)
    gw, gh = int(sim.grid_w), int(sim.grid_h)
    margin = max(1, int(sim.cfg.wall_thickness) // int(sim.cfg.segment_size))
    used: set[tuple[int, int]] = set()

    # Snake lengths chosen to exercise boost burn (>=5), 1.15x size-ratio wins (23 vs 20),
    # mutual head-on ties (22 vs 20), and large corpse FIFO overflows (>=60 segments).
    if env % 2 == 0:
        lengths = (23, 20, 48, 64, 12, 8)
    else:
        lengths = (22, 20, 55, 72, 15, 6)

    y_rows = (10, 12, 26, 40, 56, 70)
    for slot in range(sim.S):
        seg_n = lengths[slot]
        row_y = min(gh - margin - 2, max(margin + 1, y_rows[slot] + int(rng.integers(-1, 2))))
        if slot == 0:
            # Slot 0 heading right toward slot 1 (heading left on nearby row) for early contests.
            hx = 48
            direction = 1
            body = [(hx - k, row_y) for k in range(seg_n)]
        elif slot == 1:
            hx = 55
            direction = 3
            same_row = y_rows[0] if env % 4 == 1 else row_y
            body = [(hx + k, same_row) for k in range(seg_n)]
        else:
            hx = 75 + (slot % 2) * 5
            direction = 3 if slot % 2 == 1 else 1
            step_dx = 1 if direction == 3 else -1
            body = [(hx + k * step_dx, row_y) for k in range(seg_n)]

        for idx, cell in enumerate(body):
            ring = (seg_n - 1 - idx) % sim.cap
            sim.bodies[env, slot, ring] = cell
            used.add(cell)
        sim.head_ptr[env, slot] = seg_n - 1
        sim.seg_count[env, slot] = seg_n
        sim.length[env, slot] = seg_n
        sim._reward_prev_length[env, slot] = seg_n
        sim.direction[env, slot] = direction
        sim.alive[env, slot] = True
        sim.boost_frames[env, slot] = int(sim.cfg.boost_length_cost_frames) - 1
        sim.frames_since_food[env, slot] = int(rng.integers(0, 15))
        sim.respawn_timer[env, slot] = 0
        sim._boosted_this_step[env, slot] = bool(slot % 2 == 0)

    # Fill exact 300 ambient + 150 corpse = 450 pellets on unoccupied interior cells.
    max_food = int(sim.cfg.max_food)
    corpse_cap = corpse_food_cap(max_food)
    all_interior = [
        (x, y)
        for y in range(margin, gh - margin)
        for x in range(margin, gw - margin)
        if (x, y) not in used
    ]
    perm = rng.permutation(len(all_interior))
    chosen_ambient = [all_interior[int(i)] for i in perm[:max_food]]
    chosen_corpse = [all_interior[int(i)] for i in perm[max_food : max_food + corpse_cap]]
    ordered_cells = chosen_ambient + chosen_corpse
    sim.food_cells[env] = list(ordered_cells)
    sim.food_set[env] = set(ordered_cells)
    sim.corpse_cells[env] = set(chosen_corpse)
    sim._rebuild_env_grids(env, food=True)
    sim._food_array[env, : len(ordered_cells)] = np.asarray(ordered_cells, dtype=np.int64)
    sim._food_n[env] = len(ordered_cells)
    sim._amb_count[env] = len(chosen_ambient)
    for cx, cy in chosen_ambient:
        sim._food[env, cy, cx] = FOOD_AMBIENT
    for cx, cy in chosen_corpse:
        sim._food[env, cy, cx] = FOOD_CORPSE


def prepare_cpu_policy_phase(cpu_sim: GridBatchSim) -> None:
    """Run GridBatchSim's pre-action lifecycle matching BatchSim.step_with_policy."""
    if getattr(cpu_sim, "_policy_phase_prepared", False):
        return
    active = cpu_sim._normalize_active_env_mask(None)
    cpu_sim._active_env_mask = active
    try:
        cpu_sim.frame[active] += 1
        cpu_sim._maintain_food()
        if cpu_sim.allow_respawn:
            cpu_sim._respawn_dead()
        cpu_sim._refresh_action_masks(active)
        cpu_sim._policy_phase_prepared = True
    finally:
        del cpu_sim._active_env_mask


def build_watch6_v2_cpu_sim(
    seeds: list[int],
    *,
    max_capacity: int = 1600,
    mature_fraction: float = 0.5,
) -> GridBatchSim:
    """Create a Watch6 mechanics-v2 CPU GridBatchSim with fresh and mature-450 worlds."""
    cfg = watch6_v2_config(len(seeds), max_capacity=max_capacity)
    sim = GridBatchSim(cfg, seeds=seeds, train_mode=False, allow_respawn=True)
    mature_cutoff = int(round(len(seeds) * float(mature_fraction)))
    for env in range(mature_cutoff):
        _populate_mature_v2_world(sim, env, seeds[env])
    if mature_cutoff > 0:
        active = np.ones(sim.E, dtype=bool)
        sim._rebuild_traversed_from_heads(active)
        sim._refresh_action_masks(active)
    return sim


@dataclass
class SimpleWorldView:
    """Minimal CapturedWorld-compatible bundle consumed by DevicePolicy.from_world."""

    sim: GridBatchSim
    runtime: Vector61Runtime
    hero: Ego2sSimdPolicy
    opponents: list[Vector61SimdPolicy]
    seed: int


def build_product_world_bundle(
    sim: GridBatchSim,
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
) -> SimpleWorldView:
    """Attach Watch6 product B8 hero + 5x FRP3-s12+v8 opponents to a batched GridBatchSim."""
    from dataclasses import replace

    from src.simd_env.vector61_policy import Vector61Params

    v61_params = replace(Vector61Params.from_game_config(sim.cfg), use_free_space=True)
    runtime = Vector61Runtime(params=v61_params, backend="numba")
    controlled = np.zeros((sim.E, sim.S), dtype=bool)
    controlled[:, 1:] = True
    runtime.bind(sim, controlled)
    hero = Ego2sSimdPolicy(
        str(b8_checkpoint),
        device="cpu",
        forward="rowwise",
        mask_mode="advisory",
    )
    first_opp = Vector61SimdPolicy(
        str(frp3_checkpoint),
        runtime,
        veto_slots=(1,),
        forward="rowwise",
        veto_variant="v8",
        veto_lambda=8.0,
    )
    opponents = [first_opp]
    for slot in range(2, sim.S):
        clone_opp = Vector61SimdPolicy.__new__(Vector61SimdPolicy)
        clone_opp.__dict__ = dict(first_opp.__dict__)
        clone_opp.veto_slots = frozenset((slot,))
        clone_opp.live_vetoes = {}
        clone_opp.veto_counters = {}
        clone_opp.boost_counters = {}
        opponents.append(clone_opp)
    seed0 = int(sim._seeds[0]) if sim._seeds else 0
    return SimpleWorldView(
        sim=sim,
        runtime=runtime,
        hero=hero,
        opponents=opponents,
        seed=seed0,
    )


def compare_discrete_sim_state(
    cpu_sim: GridBatchSim,
    gpu_sim: DeviceSim,
    *,
    proxies: list[SplitMixEnvRngProxy] | None = None,
    check_step_outputs: bool = True,
) -> list[str]:
    """Return all discrete or reward/mask discrepancies between CPU GridBatchSim and gpu_sim."""
    diffs: list[str] = []
    for field in (
        "alive",
        "length",
        "seg_count",
        "direction",
        "boost_frames",
        "frames_since_food",
        "respawn_timer",
        "frame",
        "_boosted_this_step",
    ):
        cpu_arr = np.asarray(getattr(cpu_sim, field))
        gpu_arr = getattr(gpu_sim, field).detach().cpu().numpy()
        if not np.array_equal(cpu_arr, gpu_arr):
            diffs.append(f"field_mismatch:{field}")

    if check_step_outputs:
        for field in (
            "_last_done",
            "_last_death_cause",
            "_last_kills",
            "_last_food_ate",
            "_last_ambient_food_ate",
            "_last_corpse_food_ate",
            "_last_transition_valid",
        ):
            cpu_arr = np.asarray(getattr(cpu_sim, field))
            gpu_arr = getattr(gpu_sim, field).detach().cpu().numpy()
            if not np.array_equal(cpu_arr, gpu_arr):
                diffs.append(f"step_output_mismatch:{field}")
        cpu_rew = np.asarray(cpu_sim._last_reward, dtype=np.float32)
        gpu_rew = gpu_sim._last_reward.detach().cpu().float().numpy()
        if not np.allclose(cpu_rew, gpu_rew, atol=REWARD_ATOL, rtol=1e-5):
            max_err = float(np.max(np.abs(cpu_rew - gpu_rew)))
            diffs.append(f"reward_mismatch:max_abs={max_err:.6g}")

    # Live ordered body segments 0..seg_count-1 for every alive snake.
    gpu_bodies = gpu_sim.bodies.detach().cpu().numpy()
    gpu_hp = gpu_sim.head_ptr.detach().cpu().numpy()
    gpu_cap = int(gpu_sim.cap)
    cpu_cap = int(cpu_sim.cap)
    for env in range(cpu_sim.E):
        for slot in range(cpu_sim.S):
            if not cpu_sim.alive[env, slot]:
                continue
            n_seg = int(cpu_sim.seg_count[env, slot])
            k = np.arange(n_seg, dtype=np.int64)
            cpu_segs = cpu_sim.bodies[env, slot, (int(cpu_sim.head_ptr[env, slot]) - k) % cpu_cap]
            gpu_segs = gpu_bodies[env, slot, (int(gpu_hp[env, slot]) - k) % gpu_cap]
            if not np.array_equal(cpu_segs, gpu_segs):
                diffs.append(f"body_segments_mismatch:env={env}:slot={slot}")
                break

    # Ordered food list and corpse flags.
    gpu_fc = gpu_sim.food_count.detach().cpu().numpy()
    gpu_fcells = gpu_sim.food_cells.detach().cpu().numpy()
    gpu_fpres = gpu_sim.food_present.detach().cpu().numpy()
    gpu_fcorpse = gpu_sim.corpse_food.detach().cpu().numpy()
    for env in range(cpu_sim.E):
        n_food = len(cpu_sim.food_cells[env])
        if int(gpu_fc[env]) != n_food:
            diffs.append(f"food_count_mismatch:env={env}:cpu={n_food}:gpu={int(gpu_fc[env])}")
            continue
        if int(gpu_fpres[env].sum()) != n_food or not bool(gpu_fpres[env, :n_food].all()):
            diffs.append(f"food_present_packing_mismatch:env={env}")
            continue
        if n_food > 0:
            cpu_cells = np.asarray(cpu_sim.food_cells[env], dtype=np.int64)
            if not np.array_equal(cpu_cells, gpu_fcells[env, :n_food]):
                diffs.append(f"ordered_food_cells_mismatch:env={env}")
                continue
            corpse_set = cpu_sim.corpse_cells[env]
            cpu_corpse = np.fromiter(
                (cell in corpse_set for cell in cpu_sim.food_cells[env]),
                dtype=bool,
                count=n_food,
            )
            if not np.array_equal(cpu_corpse, gpu_fcorpse[env, :n_food]):
                diffs.append(f"ordered_corpse_flags_mismatch:env={env}")

    # Advisory action masks across all (E, S, 6).
    cpu_adv = np.asarray(cpu_sim.get_advisory_action_mask(), dtype=bool)
    gpu_adv = gpu_sim.advisory_mask().detach().cpu().numpy().astype(bool)
    if not np.array_equal(cpu_adv, gpu_adv):
        diffs.append("advisory_mask_mismatch")

    # Per-world RNG draw counter lockstep when SplitMix proxies are installed.
    if proxies is not None:
        gpu_counters = gpu_sim.rng_counter.detach().cpu().tolist()
        for env, proxy in enumerate(proxies):
            if int(proxy.counter) != int(gpu_counters[env]):
                diffs.append(
                    f"rng_counter_mismatch:env={env}:cpu={proxy.counter}:gpu={gpu_counters[env]}"
                )
                break
    return diffs


def compare_ego_bv8_observations(
    cpu_sim: GridBatchSim,
    gpu_sim: DeviceSim,
    hero_net: Any | None = None,
) -> list[str]:
    """Compare ego2s-bv8 hero observations (local, global, scalars) across all live hero rows."""
    live_envs = np.flatnonzero(cpu_sim.alive[:, 0])
    if len(live_envs) == 0:
        return []
    rows_np = np.stack([live_envs, np.zeros_like(live_envs)], axis=1).astype(np.int64)
    cfg = ego_config_for(hero_net) if hero_net is not None else build_bv8_config()
    cpu_obs = build_ego_raster(cpu_sim, cfg, backend="numba", rows=rows_np)

    rows_t = torch.as_tensor(rows_np, dtype=torch.long, device=gpu_sim.device)
    legal_t = gpu_sim.advisory_mask()[rows_t[:, 0], 0]
    v8_feats = (
        gpu_v8_decide(
            gpu_sim,
            rows_t,
            torch.zeros((len(live_envs), 6), dtype=torch.float32, device=gpu_sim.device),
            legal_t,
            8.0,
            features=True,
        )
        .feature_planes()
        .reshape(len(live_envs), 30)
    )
    local_t, glob_t, scalar_t = build_observation(gpu_sim, rows_t, v8_feats)
    diffs: list[str] = []
    if not np.array_equal(cpu_obs["local"], local_t.detach().cpu().numpy()):
        diffs.append("ego_obs_local_uint8_mismatch")
    if not np.array_equal(cpu_obs["global"], glob_t.detach().cpu().numpy()):
        diffs.append("ego_obs_global_uint8_mismatch")
    c_sc = np.asarray(cpu_obs["scalars"], dtype=np.float32)
    g_sc = scalar_t.detach().cpu().float().numpy()
    if not np.allclose(c_sc, g_sc, atol=EGO_SCALAR_ATOL, rtol=1e-5):
        max_err = float(np.max(np.abs(c_sc - g_sc)))
        diffs.append(f"ego_obs_scalars_mismatch:max_abs={max_err:.6g}")
    return diffs


def build_bv8_config() -> Any:
    from src.simd_env.ego_raster import EgoRasterConfig

    return EgoRasterConfig(version="bv8")


def _cpu_product_decisions(
    world: SimpleWorldView,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """Compute canonical Watch6 CPU actions plus raw Qs for float32 tie auditing."""
    from src.training.action_mask import INVALID_Q_VALUE

    sim = world.sim
    actions = np.ones((sim.E, sim.S), dtype=np.int64)
    hero_q_by_env: dict[int, np.ndarray] = {}
    opp_q_mask_by_row: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    live_heroes = np.flatnonzero(sim.alive[:, 0])
    if len(live_heroes) > 0:
        hero_rows = np.stack([live_heroes, np.zeros_like(live_heroes)], axis=1).astype(np.int64)
        obs = build_ego_raster(sim, ego_config_for(world.hero.net), backend="numba", rows=hero_rows)
        q_hero = forward_q(world.hero.net, obs, world.hero.forward).numpy().astype(np.float32)
        adv = sim.get_advisory_action_mask()[live_heroes, 0].astype(bool).copy()
        no_safe = ~adv.any(axis=1)
        adv[no_safe, :3] = True
        masked = np.where(adv, q_hero.astype(np.float64), -np.inf)
        chosen = np.argmax(masked, axis=1).astype(np.int64)
        actions[live_heroes, 0] = chosen
        for idx, env in enumerate(live_heroes.tolist()):
            hero_q_by_env[int(env)] = q_hero[idx]

    opp_rows_list = [
        (env, slot) for slot in range(1, sim.S) for env in range(sim.E) if sim.alive[env, slot]
    ]
    if opp_rows_list:
        opp_rows = np.asarray(opp_rows_list, dtype=np.int64)
        states, exact = world.runtime.selection(opp_rows)
        q_opp = world.opponents[0].q_values(states).detach().cpu().numpy().astype(np.float32)
        v8_res = v8_nb.v8_decide(sim, opp_rows, q_opp, exact, lam=8.0, features=False)
        actions[opp_rows[:, 0], opp_rows[:, 1]] = v8_res.action.astype(np.int64)
        for idx, (env, slot) in enumerate(opp_rows_list):
            mask_fb = exact[idx].astype(bool).copy()
            if not mask_fb.any():
                mask_fb[:3] = True
            masked_q = np.where(mask_fb, q_opp[idx], np.float32(INVALID_Q_VALUE))
            opp_q_mask_by_row[(int(env), int(slot))] = (masked_q, exact[idx].astype(bool).copy())

    return actions, hero_q_by_env, opp_q_mask_by_row


def _classify_policy_mismatch(
    *,
    env: int,
    slot: int,
    cpu_action: int,
    gpu_action: int,
    world: SimpleWorldView,
    gpu_sim: DeviceSim,
    gpu_policy: DevicePolicy,
    hero_q_cpu: dict[int, np.ndarray],
    hero_q_gpu_all: np.ndarray,
    opp_q_mask_cpu: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[bool, dict[str, Any]]:
    """Verify whether a live action difference is an affirmative float32 Q tie."""
    if slot == 0:
        q_cpu = hero_q_cpu[env]
        q_gpu = hero_q_gpu_all[env]
        adv = world.sim.get_advisory_action_mask()[env, 0].astype(bool).copy()
        if not adv.any():
            adv[:3] = True
        both_legal = bool(adv[cpu_action] and adv[gpu_action])
        max_q_diff = float(np.max(np.abs(q_cpu - q_gpu)))
        margin_cpu = float(abs(float(q_cpu[cpu_action]) - float(q_cpu[gpu_action])))
        band = max(FLOAT_TIE_MIN_BAND, 2.5 * max_q_diff)
        is_tie = both_legal and max_q_diff <= FLOAT_TIE_MAX_Q_DIFF and margin_cpu <= band
        return is_tie, {
            "env": env,
            "slot": 0,
            "kind": "hero_b8",
            "cpu_action": cpu_action,
            "gpu_action": gpu_action,
            "both_legal": both_legal,
            "max_q_diff": max_q_diff,
            "cpu_q_margin": margin_cpu,
            "classification": "tie_explained" if is_tie else "unexplained",
        }

    # Opponent FRP3-s12 + v8(lambda=8.0): verify discrete v8 flood planes match 100%
    # and substituting q_gpu into CPU v8_nb.v8_decide reproduces gpu_action.
    assert gpu_policy.vector_runtime is not None
    q_cpu_masked, exact_cpu = opp_q_mask_cpu[(env, slot)]
    row_np = np.array([[env, slot]], dtype=np.int64)
    row_t = torch.tensor([[env, slot]], dtype=torch.long, device=gpu_sim.device)
    vec_gpu, mask_gpu = gpu_policy.vector_runtime.selection(row_t)
    net_gpu = gpu_policy.opponent_networks[slot - 1]
    assert net_gpu is not None
    with torch.inference_mode():
        q_gpu_raw = net_gpu(vec_gpu).detach().cpu().float().numpy()[0]

    cpu_v8 = v8_nb.v8_decide(
        world.sim, row_np, q_cpu_masked[None, :], exact_cpu[None, :], lam=8.0, features=True
    )
    gpu_v8 = gpu_v8_decide(
        gpu_sim,
        row_t,
        torch.as_tensor(q_gpu_raw[None, :], dtype=torch.float32, device=gpu_sim.device),
        mask_gpu,
        8.0,
        features=True,
    )
    discrete_v8_match = (
        np.array_equal(cpu_v8.counts, gpu_v8.counts.detach().cpu().numpy())
        and np.array_equal(cpu_v8.need, gpu_v8.need.detach().cpu().numpy())
        and np.array_equal(cpu_v8.area, gpu_v8.area.detach().cpu().numpy())
        and np.array_equal(cpu_v8.landing, gpu_v8.landing.detach().cpu().numpy())
        and np.array_equal(cpu_v8.head_risky, gpu_v8.head_risky.detach().cpu().numpy())
        and np.array_equal(cpu_v8.head_waived, gpu_v8.head_waived.detach().cpu().numpy())
        and np.array_equal(cpu_v8.v5_eligible, gpu_v8.v5_eligible.detach().cpu().numpy())
    )
    mask_fb = exact_cpu.copy()
    if not mask_fb.any():
        mask_fb[:3] = True
    q_gpu_masked = np.where(mask_fb, q_gpu_raw, np.float32(-1.0e9))
    max_q_diff = float(np.max(np.abs(q_cpu_masked[mask_fb] - q_gpu_masked[mask_fb])))
    replayed_cpu_v8 = v8_nb.v8_decide(
        world.sim, row_np, q_gpu_raw[None, :], exact_cpu[None, :], lam=8.0, features=False
    )
    q_substitution_recovers_gpu = int(replayed_cpu_v8.action[0]) == int(gpu_action)
    both_legal = bool(mask_fb[cpu_action] and mask_fb[gpu_action])
    is_tie = (
        both_legal
        and discrete_v8_match
        and max_q_diff <= FLOAT_TIE_MAX_Q_DIFF
        and q_substitution_recovers_gpu
    )
    return is_tie, {
        "env": env,
        "slot": slot,
        "kind": "opponent_frp3_v8",
        "cpu_action": cpu_action,
        "gpu_action": gpu_action,
        "both_legal": both_legal,
        "discrete_v8_match": discrete_v8_match,
        "max_q_diff": max_q_diff,
        "q_substitution_recovers_gpu": q_substitution_recovers_gpu,
        "classification": "tie_explained" if is_tie else "unexplained",
    }


def run_v2_parity_rollout(
    *,
    seeds: list[int],
    ticks: int,
    device: str | torch.device = "cpu",
    b8_checkpoint: Path | None = None,
    frp3_checkpoint: Path | None = None,
    check_ct_sim: bool = True,
    ego_check_stride: int = 25,
    mature_fraction: float = 0.5,
    max_capacity: int = 1600,
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Run matched-draw Watch6 mechanics-v2 parity across ticks for DeviceSim (+ CTDeviceSim)."""
    target = torch.device(device)
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cpu_sim = build_watch6_v2_cpu_sim(
        seeds,
        max_capacity=max_capacity,
        mature_fraction=mature_fraction,
    )
    move_backend = "indexed"
    gpu_sim = DeviceSim.from_reference(
        cpu_sim,
        device=target,
        hero_respawn=True,
    )
    gpu_sim.move_backend = move_backend
    ct_sim: CTDeviceSim | None = None
    if check_ct_sim:
        ct_sim = CTDeviceSim.from_reference(
            cpu_sim,
            device=target,
            hero_respawn=True,
        )
        ct_sim.move_backend = move_backend

    proxies = install_splitmix_rng_proxies(cpu_sim, gpu_sim)

    world: SimpleWorldView | None = None
    gpu_policy: DevicePolicy | None = None
    if b8_checkpoint is not None and frp3_checkpoint is not None:
        world = build_product_world_bundle(cpu_sim, b8_checkpoint, frp3_checkpoint)
        gpu_policy = DevicePolicy.from_world(
            world,
            device=target,
            sim=gpu_sim,
        )

    action_rng = np.random.default_rng(int(seeds[0]) ^ 0xA5A5_5A5A)
    discrete_mismatches = 0
    ct_discrete_mismatches = 0
    ego_obs_mismatches = 0
    tie_explained_decisions = 0
    unexplained_mismatches = 0
    total_live_decisions = 0
    mismatch_samples: list[dict[str, Any]] = []
    tie_samples: list[dict[str, Any]] = []

    events = {
        "ambient_eats_and_instep_refills": 0,
        "corpse_eats": 0,
        "boost_steps": 0,
        "boost_trail_burns": 0,
        "boost_burns_at_full_450_cap": 0,
        "deaths_total": 0,
        "deaths_hero": 0,
        "deaths_opponent": 0,
        "deaths_wall": 0,
        "deaths_self": 0,
        "deaths_head": 0,
        "deaths_body": 0,
        "headon_size_wins": 0,
        "respawns_total": 0,
        "respawns_hero": 0,
        "respawns_opponent": 0,
        "mid_boost_deaths_followed_by_respawn": 0,
        "rng_draws_matched": 0,
    }
    corpse_cap = corpse_food_cap(int(cpu_sim.cfg.max_food))
    full_board_cap = int(cpu_sim.cfg.max_food) + corpse_cap
    prev_mid_boost_dead = np.zeros((cpu_sim.E, cpu_sim.S), dtype=bool)

    for tick in range(int(ticks)):
        dead_before = ~cpu_sim.alive.copy()

        # 1. Prepare phase (ambient food maintenance + continuous respawn + masks + vector61).
        prepare_cpu_policy_phase(cpu_sim)
        gpu_sim.prepare()
        if ct_sim is not None:
            ct_sim.prepare()

        respawned_now = dead_before & cpu_sim.alive
        if respawned_now.any():
            events["respawns_total"] += int(respawned_now.sum())
            events["respawns_hero"] += int(respawned_now[:, 0].sum())
            events["respawns_opponent"] += int(respawned_now[:, 1:].sum())
            mid_boost_respawn = prev_mid_boost_dead & respawned_now
            if mid_boost_respawn.any():
                events["mid_boost_deaths_followed_by_respawn"] += int(mid_boost_respawn.sum())

        prep_diffs = compare_discrete_sim_state(
            cpu_sim, gpu_sim, proxies=proxies, check_step_outputs=(tick > 0)
        )
        if prep_diffs:
            discrete_mismatches += len(prep_diffs)
            if len(mismatch_samples) < 16:
                mismatch_samples.append({"tick": tick, "phase": "prepare", "diffs": prep_diffs})
        if ct_sim is not None:
            ct_prep_diffs = compare_discrete_sim_state(
                cpu_sim, ct_sim, proxies=proxies, check_step_outputs=(tick > 0)
            )
            if ct_prep_diffs:
                ct_discrete_mismatches += len(ct_prep_diffs)
                if len(mismatch_samples) < 16:
                    mismatch_samples.append(
                        {"tick": tick, "phase": "ct_prepare", "diffs": ct_prep_diffs}
                    )

        # Always check ego2s-bv8 observations on mid-boost hero respawns and periodic ticks.
        need_ego_check = (
            tick == 0
            or (ego_check_stride > 0 and (tick + 1) % ego_check_stride == 0)
            or bool((prev_mid_boost_dead[:, 0] & respawned_now[:, 0]).any())
        )
        if need_ego_check:
            ego_diffs = compare_ego_bv8_observations(
                cpu_sim, gpu_sim, world.hero.net if world is not None else None
            )
            if ego_diffs:
                ego_obs_mismatches += len(ego_diffs)
                if len(mismatch_samples) < 16:
                    mismatch_samples.append({"tick": tick, "phase": "ego_bv8", "diffs": ego_diffs})

        # 2. Action selection & B8 + 5x FRP3-s12+v8 policy decision parity.
        if world is not None and gpu_policy is not None:
            world.runtime.prepare(cpu_sim)
            cpu_actions, hero_q_cpu, opp_q_mask_cpu = _cpu_product_decisions(world)
            q_hero_gpu_t, _legal_gpu_t, hero_act_gpu_t = gpu_policy.evaluate(gpu_sim)
            gpu_actions_t = gpu_policy.actions(gpu_sim, hero_act_gpu_t)
            gpu_actions = gpu_actions_t.detach().cpu().numpy().astype(np.int64)
            q_hero_gpu_np = q_hero_gpu_t.detach().cpu().float().numpy()

            live_mask = cpu_sim.alive
            total_live_decisions += int(live_mask.sum())
            diff_coords = np.argwhere(live_mask & (cpu_actions != gpu_actions))
            for env_i, slot_i in diff_coords.tolist():
                is_tie, detail = _classify_policy_mismatch(
                    env=int(env_i),
                    slot=int(slot_i),
                    cpu_action=int(cpu_actions[env_i, slot_i]),
                    gpu_action=int(gpu_actions[env_i, slot_i]),
                    world=world,
                    gpu_sim=gpu_sim,
                    gpu_policy=gpu_policy,
                    hero_q_cpu=hero_q_cpu,
                    hero_q_gpu_all=q_hero_gpu_np,
                    opp_q_mask_cpu=opp_q_mask_cpu,
                )
                detail["tick"] = tick
                if is_tie:
                    tie_explained_decisions += 1
                    if len(tie_samples) < 16:
                        tie_samples.append(detail)
                else:
                    unexplained_mismatches += 1
                    if len(mismatch_samples) < 16:
                        mismatch_samples.append(detail)
        else:
            # Checkpoint-free action mix favouring boost and turning to exercise all v2 branches.
            adv = cpu_sim.get_advisory_action_mask().astype(bool)
            cpu_actions = np.zeros((cpu_sim.E, cpu_sim.S), dtype=np.int64)
            raw = action_rng.integers(0, 6, size=(cpu_sim.E, cpu_sim.S))
            for env_i in range(cpu_sim.E):
                for slot_i in range(cpu_sim.S):
                    if adv[env_i, slot_i, raw[env_i, slot_i]]:
                        cpu_actions[env_i, slot_i] = int(raw[env_i, slot_i])
                    elif adv[env_i, slot_i].any():
                        safe_choices = np.flatnonzero(adv[env_i, slot_i])
                        # Prefer boost actions (3..5) 50% of the time when safe.
                        cpu_actions[env_i, slot_i] = int(safe_choices[-1])
                    else:
                        cpu_actions[env_i, slot_i] = int(raw[env_i, slot_i] % 3)

        # Track pre-step boost burn conditions at full 450 food cap.
        pre_full_450 = np.array(
            [len(cpu_sim.food_cells[e]) >= full_board_cap for e in range(cpu_sim.E)],
            dtype=bool,
        )
        pre_will_boost = (
            cpu_sim.alive
            & (cpu_actions >= 3)
            & (cpu_sim.length >= int(cpu_sim.cfg.min_boost_length))
        )
        pre_will_burn = (
            pre_will_boost
            & ((cpu_sim.boost_frames + 1) >= int(cpu_sim.cfg.boost_length_cost_frames))
            & (cpu_sim.length > 1)
        )
        events["boost_steps"] += int(pre_will_boost.sum())
        events["boost_trail_burns"] += int(pre_will_burn.sum())
        events["boost_burns_at_full_450_cap"] += int((pre_will_burn & pre_full_450[:, None]).sum())

        # 3. Step all simulators with the common action vector + observe carry.
        actions_t = torch.as_tensor(cpu_actions, dtype=torch.long, device=target)
        cpu_sim.step(cpu_actions)
        gpu_sim.step(actions_t)
        if ct_sim is not None:
            ct_sim.step(actions_t)

        if world is not None and gpu_policy is not None:
            world.runtime.observe_step(cpu_sim)
            gpu_policy.after_step(gpu_sim)

        # Record mechanics-v2 event counters.
        died_now = cpu_sim._last_done
        prev_mid_boost_dead = died_now & cpu_sim._boosted_this_step
        events["ambient_eats_and_instep_refills"] += int(cpu_sim._last_ambient_food_ate.sum())
        events["corpse_eats"] += int(cpu_sim._last_corpse_food_ate.sum())
        events["deaths_total"] += int(died_now.sum())
        events["deaths_hero"] += int(died_now[:, 0].sum())
        events["deaths_opponent"] += int(died_now[:, 1:].sum())
        causes = cpu_sim._last_death_cause
        events["deaths_wall"] += int((died_now & (causes == DEATH_WALL)).sum())
        events["deaths_self"] += int((died_now & (causes == DEATH_SELF)).sum())
        events["deaths_head"] += int((died_now & (causes == DEATH_HEAD)).sum())
        events["deaths_body"] += int((died_now & (causes == DEATH_BODY)).sum())
        head_deaths_in_env = (died_now & (causes == DEATH_HEAD)).any(axis=1)
        survivor_kills = (~died_now) & (cpu_sim._last_kills > 0) & head_deaths_in_env[:, None]
        events["headon_size_wins"] += int(survivor_kills.sum())

        step_diffs = compare_discrete_sim_state(
            cpu_sim, gpu_sim, proxies=proxies, check_step_outputs=True
        )
        if step_diffs:
            discrete_mismatches += len(step_diffs)
            if len(mismatch_samples) < 16:
                mismatch_samples.append({"tick": tick, "phase": "step", "diffs": step_diffs})
        if ct_sim is not None:
            ct_step_diffs = compare_discrete_sim_state(
                cpu_sim, ct_sim, proxies=proxies, check_step_outputs=True
            )
            if ct_step_diffs:
                ct_discrete_mismatches += len(ct_step_diffs)
                if len(mismatch_samples) < 16:
                    mismatch_samples.append(
                        {"tick": tick, "phase": "ct_step", "diffs": ct_step_diffs}
                    )

        if progress_cb is not None:
            progress_cb(tick + 1, int(ticks))

    events["rng_draws_matched"] = int(sum(proxy.draws for proxy in proxies))
    total_unexplained = (
        discrete_mismatches + ct_discrete_mismatches + ego_obs_mismatches + unexplained_mismatches
    )
    return {
        "device": str(target),
        "move_backend": move_backend,
        "num_envs": int(cpu_sim.E),
        "ticks": int(ticks),
        "world_ticks": int(cpu_sim.E) * int(ticks),
        "snake_ticks": int(cpu_sim.E) * int(cpu_sim.S) * int(ticks),
        "policy_mode": "product_b8_frp3_v8" if world is not None else "scripted_v2_stress",
        "check_ct_sim": bool(check_ct_sim),
        "discrete_mismatches": int(discrete_mismatches),
        "ct_discrete_mismatches": int(ct_discrete_mismatches),
        "ego_obs_mismatches": int(ego_obs_mismatches),
        "total_live_decisions": int(total_live_decisions),
        "tie_explained_decisions": int(tie_explained_decisions),
        "unexplained_mismatches": int(total_unexplained),
        "policy_unexplained_mismatches": int(unexplained_mismatches),
        "events": events,
        "tie_samples": tie_samples,
        "mismatch_samples": mismatch_samples,
        "verdict": "PASS" if total_unexplained == 0 else "FAIL",
    }
