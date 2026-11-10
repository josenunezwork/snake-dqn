"""Step B1a (M0-GPU-SITL-Preflight) parity runner, throughput benchmark, and gradient check.

Runs on Mac CPU (`torch.device("cpu")`, single-threaded):
1. Seed disjointness audit for namespace ``gpu-sitl-preflight-20261109-v1``.
2. Natural Watch6 v2 parity: ``E = 4`` worlds × ``32`` hero ticks (`128` roots, `768` branch
   certificates) comparing batched ``CTDeviceSim`` ``open-stop-h32`` against CPU
   ``research/watch_replay/open_stop.py:open_stop_batch``.
3. Long-hero (`L >= 800`) Watch6 v2 parity: ``E = 4`` worlds × ``8`` hero ticks (`32` roots,
   `192` branch certificates) constructed via ``src/simd_env/grid_scenarios.py:inject_serpentines``
   with hero lengths ``800, 840, 900, 960``.
4. Unfrozen ``Ego2sNet`` (`b8-s3` init) margin-loss backward pass gradient check.
5. Throughput measurement (certificates/sec on Mac CPU).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from research.bl_served_search import ceiling
from research.devicesim_v2_parity.harness import (
    build_product_world_bundle,
    prepare_cpu_policy_phase,
    watch6_v2_config,
)
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sitl.open_stop_gpu import (
    MARGIN_DELTA_DEFAULT,
    batched_open_stop_h32,
    margin_loss_gradient_check,
)
from research.gpu_sitl.seed_check import (
    PREFLIGHT_NAMESPACE,
    audit_preflight_seeds,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from research.t3_rollout_labels_20261015.bank_engine import (
    CapturedWorld,
    prepared_q,
    step_prepared,
)
from research.watch_replay.open_stop import (
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
    OPEN_STOP_POCKET_MIN_LENGTH,
    open_stop_batch,
)
from src.core.game_config import get_config, initialize_config
from src.simd_env.ego2s_policy import Ego2sSimdPolicy
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_scenarios import inject_serpentines
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.vector61_featurizer import Vector61Params
from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy
from web.backend.served_checkpoint import (
    B8_S3_FILENAME,
    B8_S3_SHA256,
    FRP3_S12_FILENAME,
    FRP3_S12_SHA256,
    REPO_ROOT,
)

PASS_BAR_AGREEMENT = 0.990
DEFAULT_NATURAL_WORLDS = 4
DEFAULT_NATURAL_TICKS = 32
DEFAULT_LONG_HERO_WORLDS = 4
DEFAULT_LONG_HERO_TICKS = 8
DEFAULT_MAX_CAPACITY = 1600

LONG_HERO_LENGTH_PROFILES: tuple[list[int], ...] = (
    [800, 64, 48, 32, 24, 16],
    [840, 72, 56, 40, 28, 20],
    [900, 80, 64, 48, 32, 24],
    [960, 96, 72, 56, 40, 28],
)
LONG_HERO_BOX_HEIGHT = 8

DEFAULT_ARTIFACT_OUT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "argon-20261026/gpu-sitl-preflight-1109"
)

_SIM_ARRAY_FIELDS: tuple[str, ...] = (
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "frame",
    "_boosted_this_step",
    "_last_reward",
    "_last_done",
    "_last_death_cause",
    "_last_kills",
    "_last_food_ate",
    "_last_ambient_food_ate",
    "_last_corpse_food_ate",
    "_last_transition_valid",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_product_checkpoints(saved_dir: Path | None = None) -> tuple[Path, Path]:
    """Return verified ``(b8_s3_path, frp3_s12_path)`` under ``saved_snakes``."""
    base = Path(saved_dir) if saved_dir is not None else Path(REPO_ROOT) / "saved_snakes"
    b8_path = base / B8_S3_FILENAME
    frp3_path = base / FRP3_S12_FILENAME
    if not b8_path.is_file():
        raise FileNotFoundError(f"Missing B8-s3 checkpoint: {b8_path}")
    if not frp3_path.is_file():
        raise FileNotFoundError(f"Missing FRP3-s12 checkpoint: {frp3_path}")
    if _sha256_file(b8_path) != B8_S3_SHA256:
        raise ValueError(f"B8-s3 sha256 mismatch at {b8_path}")
    if _sha256_file(frp3_path) != FRP3_S12_SHA256:
        raise ValueError(f"FRP3-s12 sha256 mismatch at {frp3_path}")
    return b8_path, frp3_path


def _dedup_food_off_bodies(sim: GridBatchSim) -> None:
    """Relocate any food cell overlapping an injected snake body without touching RNG."""
    gw, gh = int(sim.cfg.grid_width), int(sim.cfg.grid_height)
    for env in range(int(sim.E)):
        blocked: set[tuple[int, int]] = set()
        for slot in range(int(sim.S)):
            if not bool(sim.alive[env, slot]):
                continue
            sc = int(sim.seg_count[env, slot])
            hp = int(sim.head_ptr[env, slot])
            for k in range(sc):
                idx = (hp - k) % sim.cap
                cx, cy = int(sim.bodies[env, slot, idx, 0]), int(sim.bodies[env, slot, idx, 1])
                blocked.add((cx, cy))
        taken = set(blocked)
        new_food: list[tuple[int, int]] = []
        scan_x, scan_y = 2, 2
        for cell in list(sim.food_cells[env]):
            pt = (int(cell[0]), int(cell[1]))
            if pt not in taken:
                taken.add(pt)
                new_food.append(pt)
            else:
                while (scan_x, scan_y) in taken:
                    scan_x += 1
                    if scan_x >= gw - 2:
                        scan_x = 2
                        scan_y += 1
                        if scan_y >= gh - 2:
                            scan_y = 2
                replacement = (scan_x, scan_y)
                taken.add(replacement)
                new_food.append(replacement)
        sim.food_cells[env] = new_food
        sim.food_set[env] = set(new_food)
        sim.corpse_cells[env] = {c for c in sim.corpse_cells[env] if c not in blocked}
        sim._rebuild_env_grids(env, food=True)
    all_envs = np.ones(int(sim.E), dtype=bool)
    sim._rebuild_traversed_from_heads(all_envs)
    sim._refresh_action_masks(all_envs)


def ensure_watch6_v2_game_config() -> None:
    """Align the global GameConfig singleton with shipped Watch6 mechanics v2."""
    cfg = get_config()
    if (
        cfg.game.mechanics_version != 2
        or cfg.game.num_snakes != 6
        or not cfg.network.use_free_space
        or cfg.network.input_size != 61
    ):
        updated = replace(
            cfg,
            game=replace(
                cfg.game,
                mechanics_version=2,
                num_snakes=6,
                min_boost_length=5,
                boost_length_cost_frames=3,
                arena_type="rectangular",
            ),
            network=replace(cfg.network, input_size=61, use_free_space=True),
        )
        initialize_config(updated)


def build_watch6_captured_worlds(
    seeds: Sequence[int],
    b8_path: Path,
    frp3_path: Path,
    *,
    max_capacity: int = DEFAULT_MAX_CAPACITY,
    long_hero_profiles: Sequence[Sequence[int]] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[list[CapturedWorld], DevicePolicy]:
    """Construct ``E`` CPU :class:`CapturedWorld` instances and a matching :class:`DevicePolicy`."""
    ensure_watch6_v2_game_config()
    cfg_1 = watch6_v2_config(1, max_capacity=max_capacity)
    params = replace(Vector61Params.from_game_config(cfg_1), use_free_space=True)

    shared_hero = Ego2sSimdPolicy(
        str(b8_path), device="cpu", forward="rowwise", mask_mode="advisory"
    )
    # Build one template Vector61SimdPolicy so we load FRP3 weights from disk only once.
    dummy_sim = GridBatchSim(cfg_1, seeds=[int(seeds[0])], train_mode=False, allow_respawn=True)
    dummy_rt = Vector61Runtime(params=params, backend="numba")
    dummy_rt.bind(dummy_sim, np.array([[False, True, True, True, True, True]]))
    template_opp = Vector61SimdPolicy(
        str(frp3_path),
        dummy_rt,
        veto_slots=(1, 2, 3, 4, 5),
        forward="rowwise",
        veto_variant="v8",
        veto_lambda=8.0,
    )

    worlds: list[CapturedWorld] = []
    for idx, seed in enumerate(seeds):
        sim = GridBatchSim(cfg_1, seeds=[int(seed)], train_mode=False, allow_respawn=True)
        if long_hero_profiles is not None:
            lengths = [int(x) for x in long_hero_profiles[idx]]
            inject_serpentines([sim], lengths, box_height=LONG_HERO_BOX_HEIGHT)
            sim._rebuild_env_grids(0, food=True)

        rt = Vector61Runtime(params=params, backend="numba")
        rt.bind(sim, np.array([[False, True, True, True, True, True]]))

        opp = object.__new__(Vector61SimdPolicy)
        opp.__dict__ = dict(template_opp.__dict__)
        opp.runtime = rt
        opp.veto_counters = {}
        opp.boost_counters = {}
        opp.live_vetoes = {}
        opp.last_veto = None

        world = CapturedWorld(
            sim=sim,
            runtime=rt,
            hero=shared_hero,
            opponents=[opp] * 5,
            seed=int(seed),
            mix="mixed",
            kinds=["checkpoint"] * 5,
        )
        worlds.append(world)

    cfg_batch = watch6_v2_config(len(seeds), max_capacity=max_capacity)
    batch_ref_sim = GridBatchSim(
        cfg_batch, seeds=[int(s) for s in seeds], train_mode=False, allow_respawn=True
    )
    bundle = build_product_world_bundle(batch_ref_sim, b8_path, frp3_path)
    device_policy = DevicePolicy.from_world(bundle, device=device, prepared=False)
    return worlds, device_policy


def advance_to_next_live_hero_seam(world: CapturedWorld) -> int:
    """Prepare ``world`` at the policy seam, stepping across any dead-hero respawn frames."""
    respawn_steps = 0
    while True:
        prepare_cpu_policy_phase(world.sim)
        if bool(world.sim.alive[0, 0]):
            return respawn_steps
        # Hero is waiting to respawn; step the world under opponent policies.
        world.runtime.prepare(world.sim)
        world.sim._policy_phase_prepared = True
        step_prepared(world, 1)
        respawn_steps += 1
        if respawn_steps > 200:
            raise RuntimeError(f"Hero failed to respawn within 200 frames in seed={world.seed}")


def snapshot_worlds_to_device(
    worlds: Sequence[CapturedWorld],
    device_policy_template: DevicePolicy,
    *,
    sim_cls: type[DeviceSim] = CTDeviceSim,
    device: str | torch.device = "cpu",
    vector_prepared: bool = False,
) -> tuple[DeviceSim, DevicePolicy]:
    """Pack ``E`` 1-env :class:`CapturedWorld` states at the policy seam into ``sim_cls``."""
    e_count = len(worlds)
    cfg = watch6_v2_config(e_count, max_capacity=int(worlds[0].sim.cfg.max_capacity))
    batch_sim = GridBatchSim(
        cfg,
        seeds=[int(w.seed) for w in worlds],
        train_mode=False,
        allow_respawn=True,
    )
    for e_idx, w in enumerate(worlds):
        for name in _SIM_ARRAY_FIELDS:
            getattr(batch_sim, name)[e_idx] = getattr(w.sim, name)[0]
        batch_sim.food_cells[e_idx] = list(w.sim.food_cells[0])
        batch_sim.food_set[e_idx] = set(w.sim.food_set[0])
        batch_sim.corpse_cells[e_idx] = set(w.sim.corpse_cells[0])
        batch_sim._rebuild_env_grids(e_idx, food=True)

    all_envs = np.ones(e_count, dtype=bool)
    batch_sim._rebuild_traversed_from_heads(all_envs)
    batch_sim._refresh_action_masks(all_envs)
    batch_sim._policy_phase_prepared = True

    dev_sim = sim_cls.from_reference(
        batch_sim,
        device=device,
        hero_respawn=False,
        prepared=True,
    )
    dev_policy = device_policy_template.repeat_interleave(1)
    rt = dev_policy.vector_runtime
    dev = dev_policy.device
    if rt.controlled.shape[0] != e_count:
        keep_rows = torch.arange(e_count, device=dev, dtype=torch.long)
        for attr in (
            "controlled",
            "_carry_state",
            "_carry_mask",
            "_carry_valid",
            "_alive_post",
            "sel_state",
            "sel_mask",
            "sel_ready",
            "sel_fresh",
            "prev_dist",
            "prev_id",
        ):
            val = getattr(rt, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(rt, attr, val.index_select(0, keep_rows).clone())

    for e_idx, w in enumerate(worlds):
        cpu_rt = w.runtime
        rt._carry_state[e_idx].copy_(
            torch.as_tensor(cpu_rt._carry_state[0], dtype=torch.float32, device=dev)
        )
        rt._carry_mask[e_idx].copy_(
            torch.as_tensor(cpu_rt._carry_mask[0], dtype=torch.bool, device=dev)
        )
        rt._carry_valid[e_idx].copy_(
            torch.as_tensor(cpu_rt._carry_valid[0], dtype=torch.bool, device=dev)
        )
        rt._alive_post[e_idx].copy_(
            torch.as_tensor(cpu_rt._alive_post[0], dtype=torch.bool, device=dev)
        )
        rt.sel_state[e_idx].copy_(
            torch.as_tensor(cpu_rt.sel_state[0], dtype=torch.float32, device=dev)
        )
        rt.sel_mask[e_idx].copy_(torch.as_tensor(cpu_rt.sel_mask[0], dtype=torch.bool, device=dev))
        rt.sel_ready[e_idx].copy_(
            torch.as_tensor(cpu_rt.sel_ready[0], dtype=torch.bool, device=dev)
        )
        rt.sel_fresh[e_idx].copy_(
            torch.as_tensor(cpu_rt.sel_fresh[0], dtype=torch.bool, device=dev)
        )
        rt.prev_dist[e_idx].copy_(
            torch.as_tensor(cpu_rt.memory.prev_dist[0], dtype=torch.float32, device=dev)
        )
        rt.prev_id[e_idx].copy_(
            torch.as_tensor(cpu_rt.memory.prev_id[0], dtype=torch.long, device=dev)
        )

    dev_policy.vector_prepared = bool(vector_prepared)
    return dev_sim, dev_policy


def _enrich_open_info(open_info: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``v8_open`` and ``pocket_open`` booleans are present on ``terminal_openness``."""
    out = dict(open_info)
    adv_any = bool(out.get("advisory_any", False))
    length = int(out.get("length", 0))
    v8_need = int(out.get("v8_need", 0))
    v8_area = int(out.get("v8_best_safe_area", 0))
    pocket_r = int(out.get("max_pocket_r", 0))
    out["v8_open"] = bool(adv_any and v8_area >= v8_need)
    out["pocket_open"] = bool(
        adv_any and (pocket_r >= length if length >= int(OPEN_STOP_POCKET_MIN_LENGTH) else True)
    )
    return out


def evaluate_cpu_open_stop_all_actions(
    world: CapturedWorld,
    decision: int,
    q: np.ndarray,
    legal: np.ndarray,
    *,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    namespace: str = OPEN_STOP_CRN_NAMESPACE,
) -> dict[str, Any]:
    """Evaluate CPU :func:`open_stop_batch` both with early-stopping and for all 6 actions."""
    early = open_stop_batch(
        world,
        int(decision),
        q,
        legal,
        branch_horizon=branch_horizon,
        namespace=namespace,
    )
    all_outcomes: dict[int, dict[str, Any]] = {}
    for a in range(6):
        existing = early["outcomes"].get(a)
        if existing is not None and "certified" in existing:
            out_a = dict(existing)
            out_a["action"] = int(a)
            out_a["legal"] = bool(legal[a])
            out_a["status"] = "EVALUATED"
            out_a["terminal_openness"] = _enrich_open_info(out_a["terminal_openness"])
            all_outcomes[a] = out_a
        else:
            one_hot = np.zeros(6, dtype=bool)
            one_hot[a] = True
            single = open_stop_batch(
                world,
                int(decision),
                q,
                one_hot,
                branch_horizon=branch_horizon,
                namespace=namespace,
            )
            out_a = dict(single["outcomes"][a])
            out_a["action"] = int(a)
            out_a["legal"] = bool(legal[a])
            out_a["status"] = "EVALUATED"
            out_a["terminal_openness"] = _enrich_open_info(out_a["terminal_openness"])
            all_outcomes[a] = out_a
    return {
        **early,
        "all_outcomes": all_outcomes,
    }


def _diff_subconditions(cpu_out: dict[str, Any], gpu_out: dict[str, Any]) -> list[str]:
    """Identify which certificate boolean sub-conditions differed between CPU and GPU outcomes."""
    diffs: list[str] = []
    if bool(cpu_out["certified"]) != bool(gpu_out["certified"]):
        diffs.append(f"certified(cpu={cpu_out['certified']},gpu={gpu_out['certified']})")
    if bool(cpu_out["raw_survives"]) != bool(gpu_out["raw_survives"]):
        diffs.append(
            f"raw_survives(cpu={cpu_out['raw_survives']},gpu={gpu_out['raw_survives']},"
            f"frames_survived_cpu={cpu_out['frames_survived']},gpu={gpu_out['frames_survived']})"
        )
    cpu_open = cpu_out.get("terminal_openness")
    gpu_open = gpu_out.get("terminal_openness")
    if (
        bool(cpu_out["raw_survives"])
        and bool(gpu_out["raw_survives"])
        and cpu_open is not None
        and gpu_open is not None
    ):
        if bool(cpu_open.get("advisory_any")) != bool(gpu_open.get("advisory_any")):
            diffs.append(
                f"terminal_openness.advisory_any("
                f"cpu={cpu_open.get('advisory_any')},gpu={gpu_open.get('advisory_any')})"
            )
        if bool(cpu_open.get("v8_open")) != bool(gpu_open.get("v8_open")):
            c_area = cpu_open.get("v8_best_safe_area")
            c_need = cpu_open.get("v8_need")
            g_area = gpu_open.get("v8_best_safe_area")
            g_need = gpu_open.get("v8_need")
            diffs.append(
                f"terminal_openness.v8_open("
                f"cpu={cpu_open.get('v8_open')}[area={c_area}/need={c_need}],"
                f"gpu={gpu_open.get('v8_open')}[area={g_area}/need={g_need}])"
            )
        if bool(cpu_open.get("pocket_open")) != bool(gpu_open.get("pocket_open")):
            c_r, c_l = cpu_open.get("max_pocket_r"), cpu_open.get("length")
            g_r, g_l = gpu_open.get("max_pocket_r"), gpu_open.get("length")
            diffs.append(
                f"terminal_openness.pocket_open("
                f"cpu={cpu_open.get('pocket_open')}[R={c_r}/L={c_l}],"
                f"gpu={gpu_open.get('pocket_open')}[R={g_r}/L={g_l}])"
            )
        if bool(cpu_open.get("un_enclosed")) != bool(gpu_open.get("un_enclosed")):
            diffs.append(
                f"terminal_openness.un_enclosed("
                f"cpu={cpu_open.get('un_enclosed')},gpu={gpu_open.get('un_enclosed')})"
            )
    return diffs


def run_cohort_parity(
    cohort_name: str,
    seeds: Sequence[int],
    num_ticks: int,
    b8_path: Path,
    frp3_path: Path,
    *,
    long_hero_profiles: Sequence[Sequence[int]] | None = None,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run ``len(seeds)`` worlds × ``num_ticks`` hero ticks and compare GPU vs CPU certificates."""
    worlds, dev_policy_template = build_watch6_captured_worlds(
        seeds,
        b8_path,
        frp3_path,
        max_capacity=DEFAULT_MAX_CAPACITY,
        long_hero_profiles=long_hero_profiles,
        device=device,
    )
    e_count = len(worlds)
    ego_cfg = EgoRasterConfig(version="bv8")

    certificate_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []

    gpu_wall_seconds = 0.0
    cpu_wall_seconds = 0.0
    gpu_certificates_evaluated = 0
    cpu_certificates_evaluated = 0

    for tick in range(int(num_ticks)):
        for w in worlds:
            advance_to_next_live_hero_seam(w)

        # Snapshot pre-prepare state to CTDeviceSim so DeviceVectorRuntime.prepare runs natively.
        dev_sim, dev_policy = snapshot_worlds_to_device(
            worlds,
            dev_policy_template,
            sim_cls=CTDeviceSim,
            device=device,
            vector_prepared=False,
        )
        decisions = [int(w.sim.frame[0]) for w in worlds]
        world_seeds = [int(w.seed) for w in worlds]

        t0_gpu = time.perf_counter()
        gpu_batch = batched_open_stop_h32(
            dev_sim,
            dev_policy,
            world_seeds=world_seeds,
            decisions=decisions,
            suffix_id=0,
            branch_horizon=branch_horizon,
            namespace=OPEN_STOP_CRN_NAMESPACE,
            compact_dead=True,
            enable_r2=True,
        )
        gpu_wall_seconds += time.perf_counter() - t0_gpu
        gpu_certificates_evaluated += e_count * 6

        # Now prepare CPU worlds and run CPU open_stop_batch reference.
        t0_cpu = time.perf_counter()
        cpu_roots: list[dict[str, Any]] = []
        cpu_bare_actions: list[int] = []
        for e_idx, w in enumerate(worlds):
            w.runtime.prepare(w.sim)
            w.sim._policy_phase_prepared = True
            obs = build_ego_raster(
                w.sim,
                ego_cfg,
                backend="numba",
                rows=np.array([[0, 0]], dtype=np.int64),
            )
            q = np.asarray(prepared_q(w, obs), dtype=np.float32)
            advisory = np.asarray(w.sim.get_advisory_action_mask()[0, 0], dtype=bool)
            legal, bare = ceiling._legal(q, advisory)
            cpu_bare_actions.append(int(bare))
            cpu_res = evaluate_cpu_open_stop_all_actions(
                w,
                decisions[e_idx],
                q,
                legal,
                branch_horizon=branch_horizon,
                namespace=OPEN_STOP_CRN_NAMESPACE,
            )
            cpu_roots.append(cpu_res)
            cpu_certificates_evaluated += 6
        cpu_wall_seconds += time.perf_counter() - t0_cpu

        for e_idx, w in enumerate(worlds):
            root_id = f"{cohort_name}:w{e_idx}:seed{w.seed}:tick{tick}:frame{decisions[e_idx]}"
            hero_len = int(w.sim.length[0, 0])
            cpu_r = cpu_roots[e_idx]
            gpu_r = gpu_batch["roots"][e_idx]

            root_rows.append(
                {
                    "cohort": cohort_name,
                    "root_id": root_id,
                    "world_index": int(e_idx),
                    "world_seed": int(w.seed),
                    "tick": int(tick),
                    "decision_frame": int(decisions[e_idx]),
                    "hero_length": hero_len,
                    "cpu_bare": int(cpu_r["bare"]),
                    "gpu_bare": int(gpu_r["bare"]),
                    "cpu_chosen": int(cpu_r["chosen"]),
                    "gpu_chosen": int(gpu_r["chosen"]),
                    "chosen_agree": bool(int(cpu_r["chosen"]) == int(gpu_r["chosen"])),
                    "cpu_evaluated_actions": list(cpu_r["evaluation_order"]),
                    "gpu_evaluated_actions": list(gpu_r["evaluation_order"]),
                }
            )

            for action in range(6):
                c_out = cpu_r["all_outcomes"][action]
                g_out = gpu_r["all_outcomes"][action]
                in_cpu_early = action in cpu_r["evaluation_order"]
                in_gpu_early = action in gpu_r["evaluation_order"]

                cert_agree = bool(c_out["certified"]) == bool(g_out["certified"])
                raw_agree = bool(c_out["raw_survives"]) == bool(g_out["raw_survives"])
                c_open = c_out["terminal_openness"]
                g_open = g_out["terminal_openness"]
                v8_agree = bool(c_open["v8_open"]) == bool(g_open["v8_open"])
                pocket_agree = bool(c_open["pocket_open"]) == bool(g_open["pocket_open"])
                sub_diffs = _diff_subconditions(c_out, g_out)

                row_record = {
                    "cohort": cohort_name,
                    "root_id": root_id,
                    "world_index": int(e_idx),
                    "world_seed": int(w.seed),
                    "tick": int(tick),
                    "decision_frame": int(decisions[e_idx]),
                    "hero_length": hero_len,
                    "action": int(action),
                    "legal_at_root": bool(c_out["legal"]),
                    "in_early_stop_evaluated": bool(in_cpu_early),
                    "in_gpu_early_stop_evaluated": bool(in_gpu_early),
                    "cpu_certified": bool(c_out["certified"]),
                    "gpu_certified": bool(g_out["certified"]),
                    "certified_agree": bool(cert_agree),
                    "cpu_raw_survives": bool(c_out["raw_survives"]),
                    "gpu_raw_survives": bool(g_out["raw_survives"]),
                    "raw_survives_agree": bool(raw_agree),
                    "v8_open_agree": bool(v8_agree),
                    "pocket_open_agree": bool(pocket_agree),
                    "cpu_frames_executed": int(c_out["frames_executed"]),
                    "gpu_frames_executed": int(g_out["frames_executed"]),
                    "cpu_frames_survived": int(c_out["frames_survived"]),
                    "gpu_frames_survived": int(g_out["frames_survived"]),
                    "cpu_terminal_openness": c_open,
                    "gpu_terminal_openness": g_open,
                    "subcondition_diffs": sub_diffs,
                }
                certificate_rows.append(row_record)

                if not cert_agree or sub_diffs:
                    disagreements.append(row_record)

        # Advance each outer world by 1 frame along the product Watch6 v2 trajectory.
        for e_idx, w in enumerate(worlds):
            step_prepared(w, cpu_bare_actions[e_idx])

    return {
        "cohort": cohort_name,
        "worlds": e_count,
        "ticks": int(num_ticks),
        "roots_total": len(root_rows),
        "certificates_total": len(certificate_rows),
        "gpu_wall_seconds": float(gpu_wall_seconds),
        "cpu_wall_seconds": float(cpu_wall_seconds),
        "gpu_certificates_per_sec": (
            float(gpu_certificates_evaluated / gpu_wall_seconds) if gpu_wall_seconds > 0 else 0.0
        ),
        "cpu_certificates_per_sec": (
            float(cpu_certificates_evaluated / cpu_wall_seconds) if cpu_wall_seconds > 0 else 0.0
        ),
        "root_rows": root_rows,
        "certificate_rows": certificate_rows,
        "disagreements": disagreements,
    }


def summarize_agreement(
    certificate_rows: Sequence[dict[str, Any]],
    root_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute overall and certified-safe / not-safe split agreement statistics."""
    total = len(certificate_rows)
    agree = sum(1 for r in certificate_rows if r["certified_agree"])
    cpu_safe_rows = [r for r in certificate_rows if r["cpu_certified"]]
    cpu_unsafe_rows = [r for r in certificate_rows if not r["cpu_certified"]]
    safe_agree = sum(1 for r in cpu_safe_rows if r["gpu_certified"])
    unsafe_agree = sum(1 for r in cpu_unsafe_rows if not r["gpu_certified"])

    legal_rows = [r for r in certificate_rows if r["legal_at_root"]]
    legal_agree = sum(1 for r in legal_rows if r["certified_agree"])

    early_rows = [r for r in certificate_rows if r["in_early_stop_evaluated"]]
    early_agree = sum(1 for r in early_rows if r["certified_agree"])

    raw_surv_agree = sum(1 for r in certificate_rows if r["raw_survives_agree"])
    v8_open_agree = sum(1 for r in certificate_rows if r["v8_open_agree"])
    pocket_open_agree = sum(1 for r in certificate_rows if r["pocket_open_agree"])

    roots_total = len(root_rows)
    roots_chosen_agree = sum(1 for r in root_rows if r["chosen_agree"])
    roots_bare_agree = sum(1 for r in root_rows if r["cpu_bare"] == r["gpu_bare"])

    cert_disagreements = [r for r in certificate_rows if not r["certified_agree"]]
    sub_disagreements = [r for r in certificate_rows if r["subcondition_diffs"]]

    return {
        "total_certificates": total,
        "agreed_certificates": agree,
        "agreement_rate": float(agree / total) if total else 1.0,
        "cpu_certified_safe_total": len(cpu_safe_rows),
        "cpu_certified_safe_agreed": safe_agree,
        "cpu_certified_safe_agreement_rate": (
            float(safe_agree / len(cpu_safe_rows)) if cpu_safe_rows else 1.0
        ),
        "cpu_not_safe_total": len(cpu_unsafe_rows),
        "cpu_not_safe_agreed": unsafe_agree,
        "cpu_not_safe_agreement_rate": (
            float(unsafe_agree / len(cpu_unsafe_rows)) if cpu_unsafe_rows else 1.0
        ),
        "legal_at_root_total": len(legal_rows),
        "legal_at_root_agreed": legal_agree,
        "legal_at_root_agreement_rate": (
            float(legal_agree / len(legal_rows)) if legal_rows else 1.0
        ),
        "early_stop_evaluated_total": len(early_rows),
        "early_stop_evaluated_agreed": early_agree,
        "early_stop_evaluated_agreement_rate": (
            float(early_agree / len(early_rows)) if early_rows else 1.0
        ),
        "raw_survives_agreed": raw_surv_agree,
        "raw_survives_agreement_rate": float(raw_surv_agree / total) if total else 1.0,
        "v8_open_agreed": v8_open_agree,
        "v8_open_agreement_rate": float(v8_open_agree / total) if total else 1.0,
        "pocket_open_agreed": pocket_open_agree,
        "pocket_open_agreement_rate": float(pocket_open_agree / total) if total else 1.0,
        "roots_total": roots_total,
        "roots_chosen_agreed": roots_chosen_agree,
        "roots_chosen_agreement_rate": (
            float(roots_chosen_agree / roots_total) if roots_total else 1.0
        ),
        "roots_bare_agreed": roots_bare_agree,
        "roots_bare_agreement_rate": (
            float(roots_bare_agree / roots_total) if roots_total else 1.0
        ),
        "certificate_disagreement_count": len(cert_disagreements),
        "any_subcondition_disagreement_count": len(sub_disagreements),
    }


def run_preflight(
    *,
    out_dir: Path = DEFAULT_ARTIFACT_OUT_DIR,
    natural_ticks: int = DEFAULT_NATURAL_TICKS,
    long_hero_ticks: int = DEFAULT_LONG_HERO_TICKS,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    delta: float = MARGIN_DELTA_DEFAULT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute the complete Step B1a Mac preflight and write compressed raw artifacts."""
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_audit = audit_preflight_seeds()
    if not seed_audit["ok"]:
        raise RuntimeError(f"Seed audit failed for {PREFLIGHT_NAMESPACE}: {seed_audit}")

    b8_path, frp3_path = resolve_product_checkpoints()
    natural_seeds = seed_audit["natural_world_seeds"]
    long_hero_seeds = seed_audit["long_hero_world_seeds"]

    natural_res = run_cohort_parity(
        "natural_watch6_v2",
        natural_seeds,
        natural_ticks,
        b8_path,
        frp3_path,
        long_hero_profiles=None,
        branch_horizon=branch_horizon,
        device=device,
    )
    long_hero_res = run_cohort_parity(
        "long_hero_ge800_watch6_v2",
        long_hero_seeds,
        long_hero_ticks,
        b8_path,
        frp3_path,
        long_hero_profiles=LONG_HERO_LENGTH_PROFILES,
        branch_horizon=branch_horizon,
        device=device,
    )

    all_cert_rows = natural_res["certificate_rows"] + long_hero_res["certificate_rows"]
    all_root_rows = natural_res["root_rows"] + long_hero_res["root_rows"]
    all_disagreements = natural_res["disagreements"] + long_hero_res["disagreements"]

    summary_natural = summarize_agreement(natural_res["certificate_rows"], natural_res["root_rows"])
    summary_long_hero = summarize_agreement(
        long_hero_res["certificate_rows"], long_hero_res["root_rows"]
    )
    summary_combined = summarize_agreement(all_cert_rows, all_root_rows)

    # Gradient check on real Watch6 v2 root states (4 natural + 4 long-hero worlds).
    grad_worlds, grad_policy_tpl = build_watch6_captured_worlds(
        natural_seeds + long_hero_seeds,
        b8_path,
        frp3_path,
        max_capacity=DEFAULT_MAX_CAPACITY,
        long_hero_profiles=[
            [3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3],
            *LONG_HERO_LENGTH_PROFILES,
        ],
    )
    for w in grad_worlds:
        advance_to_next_live_hero_seam(w)
    grad_sim, grad_pol = snapshot_worlds_to_device(
        grad_worlds,
        grad_policy_tpl,
        sim_cls=CTDeviceSim,
        device=device,
        vector_prepared=False,
    )
    gradient_report = margin_loss_gradient_check(
        str(b8_path),
        sim=grad_sim,
        delta=delta,
        device=device,
    )

    # Also measure 1 batch (4 natural roots = 24 certificates) with pure-Torch 2D wavefront floods
    # (cpu_fast_flood=False) so both CPU-queue and pure-Torch wavefront rates are reported.
    nat4_sim, nat4_pol = snapshot_worlds_to_device(
        grad_worlds[:4],
        grad_policy_tpl,
        sim_cls=CTDeviceSim,
        device=device,
        vector_prepared=False,
    )
    t0_wavefront = time.perf_counter()
    batched_open_stop_h32(
        nat4_sim,
        nat4_pol,
        world_seeds=natural_seeds,
        decisions=[int(w.sim.frame[0]) for w in grad_worlds[:4]],
        suffix_id=0,
        branch_horizon=branch_horizon,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=False,
    )
    wavefront_wall_sec = time.perf_counter() - t0_wavefront
    wavefront_certs = len(natural_seeds) * 6

    total_gpu_certs = natural_res["certificates_total"] + long_hero_res["certificates_total"]
    total_gpu_sec = natural_res["gpu_wall_seconds"] + long_hero_res["gpu_wall_seconds"]
    total_cpu_sec = natural_res["cpu_wall_seconds"] + long_hero_res["cpu_wall_seconds"]

    throughput_report = {
        "device": str(device),
        "natural_watch6_v2": {
            "certificates": int(natural_res["certificates_total"]),
            "ct_devicesim_wall_seconds": float(natural_res["gpu_wall_seconds"]),
            "ct_devicesim_certs_per_sec": float(natural_res["gpu_certificates_per_sec"]),
            "cpu_open_stop_wall_seconds": float(natural_res["cpu_wall_seconds"]),
            "cpu_open_stop_certs_per_sec": float(natural_res["cpu_certificates_per_sec"]),
        },
        "long_hero_ge800_watch6_v2": {
            "certificates": int(long_hero_res["certificates_total"]),
            "ct_devicesim_wall_seconds": float(long_hero_res["gpu_wall_seconds"]),
            "ct_devicesim_certs_per_sec": float(long_hero_res["gpu_certificates_per_sec"]),
            "cpu_open_stop_wall_seconds": float(long_hero_res["cpu_wall_seconds"]),
            "cpu_open_stop_certs_per_sec": float(long_hero_res["cpu_certificates_per_sec"]),
        },
        "combined": {
            "certificates": int(total_gpu_certs),
            "ct_devicesim_wall_seconds": float(total_gpu_sec),
            "ct_devicesim_certs_per_sec": (
                float(total_gpu_certs / total_gpu_sec) if total_gpu_sec > 0 else 0.0
            ),
            "cpu_open_stop_wall_seconds": float(total_cpu_sec),
            "cpu_open_stop_certs_per_sec": (
                float(total_gpu_certs / total_cpu_sec) if total_cpu_sec > 0 else 0.0
            ),
        },
        "pure_torch_wavefront_4roots_24certs_sample": {
            "certificates": int(wavefront_certs),
            "wall_seconds": float(wavefront_wall_sec),
            "certs_per_sec": (
                float(wavefront_certs / wavefront_wall_sec) if wavefront_wall_sec > 0 else 0.0
            ),
        },
    }

    natural_pass = bool(summary_natural["agreement_rate"] >= PASS_BAR_AGREEMENT)
    long_hero_pass = bool(summary_long_hero["agreement_rate"] >= PASS_BAR_AGREEMENT)
    combined_pass = bool(summary_combined["agreement_rate"] >= PASS_BAR_AGREEMENT)
    grad_finite_pass = bool(
        gradient_report["all_grads_present"] and gradient_report["all_grads_finite"]
    )
    overall_pass = bool(natural_pass and combined_pass and grad_finite_pass)

    total_wall_seconds = time.perf_counter() - t_start

    raw_payload = {
        "schema": "gpu-sitl-preflight-raw/v1",
        "namespace": PREFLIGHT_NAMESPACE,
        "verdict": "PASS" if overall_pass else "FAIL",
        "pass_bar": {
            "branch_certificate_agreement_min": PASS_BAR_AGREEMENT,
            "all_ego2s_gradients_finite": True,
        },
        "checkpoints": {
            "b8_s3": {"path": str(b8_path), "sha256": B8_S3_SHA256},
            "frp3_s12": {"path": str(frp3_path), "sha256": FRP3_S12_SHA256},
        },
        "seed_audit": seed_audit,
        "summary_natural": summary_natural,
        "summary_long_hero_ge800": summary_long_hero,
        "summary_combined": summary_combined,
        "throughput": throughput_report,
        "gradient_check": gradient_report,
        "disagreements": all_disagreements,
        "roots": all_root_rows,
        "certificates": all_cert_rows,
        "total_wall_seconds": float(total_wall_seconds),
    }

    raw_gz_path = out_dir / "gpu_sitl_preflight_certificates_20261109.json.gz"
    with gzip.open(raw_gz_path, "wt", encoding="utf-8") as gz_file:
        json.dump(raw_payload, gz_file, indent=2)

    raw_gz_sha256 = _sha256_file(raw_gz_path)
    summary_payload = {
        "schema": "gpu-sitl-preflight-summary/v1",
        "namespace": PREFLIGHT_NAMESPACE,
        "verdict": "PASS" if overall_pass else "FAIL",
        "pass_bar": {
            "branch_certificate_agreement_min": PASS_BAR_AGREEMENT,
            "all_ego2s_gradients_finite": True,
        },
        "bar_checks": {
            "natural_agreement_ge_99pct": natural_pass,
            "long_hero_agreement_ge_99pct": long_hero_pass,
            "combined_agreement_ge_99pct": combined_pass,
            "all_gradients_finite": grad_finite_pass,
        },
        "raw_certificates_gz_path": str(raw_gz_path),
        "raw_certificates_gz_sha256": raw_gz_sha256,
        "raw_certificates_gz_bytes": int(raw_gz_path.stat().st_size),
        "seed_audit": seed_audit,
        "summary_natural": summary_natural,
        "summary_long_hero_ge800": summary_long_hero,
        "summary_combined": summary_combined,
        "throughput": throughput_report,
        "gradient_check": gradient_report,
        "disagreements": all_disagreements,
        "total_wall_seconds": float(total_wall_seconds),
    }
    summary_path = out_dir / "gpu_sitl_preflight_summary_20261109.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step B1a M0-GPU-SITL-Preflight on Mac CPU.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_OUT_DIR,
        help="Artifact directory for compressed per-certificate output.",
    )
    parser.add_argument("--natural-ticks", type=int, default=DEFAULT_NATURAL_TICKS)
    parser.add_argument("--long-hero-ticks", type=int, default=DEFAULT_LONG_HERO_TICKS)
    parser.add_argument("--branch-horizon", type=int, default=OPEN_STOP_BRANCH_HORIZON)
    parser.add_argument("--delta", type=float, default=MARGIN_DELTA_DEFAULT)
    args = parser.parse_args()

    torch.set_num_threads(1)
    summary = run_preflight(
        out_dir=args.out_dir,
        natural_ticks=args.natural_ticks,
        long_hero_ticks=args.long_hero_ticks,
        branch_horizon=args.branch_horizon,
        delta=args.delta,
        device="cpu",
    )
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "bar_checks": summary["bar_checks"],
                "natural_agreement_rate": summary["summary_natural"]["agreement_rate"],
                "long_hero_agreement_rate": summary["summary_long_hero_ge800"]["agreement_rate"],
                "combined_agreement_rate": summary["summary_combined"]["agreement_rate"],
                "disagreements_count": len(summary["disagreements"]),
                "all_grads_finite": summary["gradient_check"]["all_grads_finite"],
                "exact_zero_param_tensors": summary["gradient_check"]["exact_zero_param_tensors"],
                "exact_zero_scalar_params": summary["gradient_check"]["exact_zero_scalar_params"],
                "throughput_combined_ct_certs_per_sec": summary["throughput"]["combined"][
                    "ct_devicesim_certs_per_sec"
                ],
                "raw_certificates_gz_path": summary["raw_certificates_gz_path"],
                "total_wall_seconds": summary["total_wall_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
