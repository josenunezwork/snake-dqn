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
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from research.bl_served_search import ceiling
from research.devicesim_v2_parity.harness import (
    SimpleWorldView,
    SplitMixEnvRngProxy,
    build_product_world_bundle,
    prepare_cpu_policy_phase,
    watch6_v2_config,
)
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sitl.open_stop_gpu import (
    MARGIN_DELTA_DEFAULT,
    RESCUE_MARGIN_DELTA_DEFAULT,
    FastDevicePolicy,
    FastDeviceVectorRuntime,
    assert_sim_and_policy_on_device,
    batched_open_stop_h32,
    margin_loss_gradient_check,
    move_device_policy_to_device,
    normalize_device,
)
from research.gpu_sitl.seed_check import (
    EXTENSION_MATURE_WORLD_INDICES,
    PREFLIGHT_NAMESPACE,
    audit_preflight_seeds,
    preflight_seeds,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from research.t3_rollout_labels_20261015.bank_engine import (
    CapturedWorld,
    prepared_q,
    step_prepared,
)
from research.watch_replay import open_stop as open_stop_mod
from research.watch_replay.open_stop import (
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
    OPEN_STOP_POCKET_MIN_LENGTH,
    open_stop_batch,
)
from src.core.game_config import get_config, initialize_config
from src.simd_env.ego2s_policy import Ego2sSimdPolicy
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_scenarios import _heading, inject_serpentines, serpentine_path
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
EXT_MATCHED_RNG_CERT_AGREE_MIN = 191  # >= 191 / 192 (99.48%)
EXT_MATCHED_RNG_INT_AGREE_RATE_MIN = 0.980  # >= 98.0%
EXT_CROSS_RNG_CERT_AGREE_MIN = 186  # >= 186 / 192 (96.875%)
EXT_CROSS_RNG_CHOSEN_AGREE_MIN = 30  # >= 30 / 32 (93.75%)

EXT_MIN_SURVIVING_ENCLOSED = 12
EXT_MIN_NEAR_THRESHOLD_SURVIVING = 8
EXT_MIN_RESCUE_ROOTS = 8

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
DEFAULT_EXTENSION_OUT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "argon-20261026/gpu-sitl-preflight-fix-1109"
)
FROZEN_COHORT_FILENAME = "mature_cohort_32_roots_cpu_frozen_20261109.json"

MATURE_EXTENSION_SPECS: tuple[dict[str, Any], ...] = (
    # Group A (8 roots, indices 8..15): dynamic doorway pocket_open=False, v8_open=True,
    # near-threshold (0.85*L <= max_pocket_r < L), rescue-override (chosen != bare).
    {
        "world_index": 8,
        "kind": "doorway_pocket",
        "hero_length": 420,
        "pocket_w": 40,
        "pocket_h": 10,
    },
    {
        "world_index": 9,
        "kind": "doorway_pocket",
        "hero_length": 500,
        "pocket_w": 47,
        "pocket_h": 10,
    },
    {
        "world_index": 10,
        "kind": "doorway_pocket",
        "hero_length": 580,
        "pocket_w": 48,
        "pocket_h": 12,
    },
    {
        "world_index": 11,
        "kind": "doorway_pocket",
        "hero_length": 660,
        "pocket_w": 54,
        "pocket_h": 12,
    },
    {
        "world_index": 12,
        "kind": "doorway_pocket",
        "hero_length": 740,
        "pocket_w": 52,
        "pocket_h": 14,
    },
    {
        "world_index": 13,
        "kind": "doorway_pocket",
        "hero_length": 820,
        "pocket_w": 57,
        "pocket_h": 14,
    },
    {
        "world_index": 14,
        "kind": "doorway_pocket",
        "hero_length": 920,
        "pocket_w": 56,
        "pocket_h": 16,
    },
    {
        "world_index": 15,
        "kind": "doorway_pocket",
        "hero_length": 1040,
        "pocket_w": 57,
        "pocket_h": 18,
    },
    # Group B (6 roots, indices 16..21): dynamic doorway v8_open=False (area < 160),
    # near-threshold (136 <= v8_best_safe_area < 160), rescue-override (chosen != bare).
    {"world_index": 16, "kind": "doorway_v8", "hero_length": 320, "pocket_w": 22, "pocket_h": 8},
    {"world_index": 17, "kind": "doorway_v8", "hero_length": 360, "pocket_w": 22, "pocket_h": 8},
    {"world_index": 18, "kind": "doorway_v8", "hero_length": 440, "pocket_w": 23, "pocket_h": 8},
    {"world_index": 19, "kind": "doorway_v8", "hero_length": 540, "pocket_w": 22, "pocket_h": 8},
    {"world_index": 20, "kind": "doorway_v8", "hero_length": 680, "pocket_w": 23, "pocket_h": 8},
    {"world_index": 21, "kind": "doorway_v8", "hero_length": 860, "pocket_w": 22, "pocket_h": 8},
    # Group C (4 roots, indices 22..25): near-threshold safe-side (L <= max_pocket_r <= 1.15*L).
    {
        "world_index": 22,
        "kind": "doorway_safe_near_threshold",
        "hero_length": 450,
        "pocket_w": 49,
        "pocket_h": 10,
    },
    {
        "world_index": 23,
        "kind": "doorway_safe_near_threshold",
        "hero_length": 550,
        "pocket_w": 51,
        "pocket_h": 12,
    },
    {
        "world_index": 24,
        "kind": "doorway_safe_near_threshold",
        "hero_length": 650,
        "pocket_w": 59,
        "pocket_h": 12,
    },
    {
        "world_index": 25,
        "kind": "doorway_safe_near_threshold",
        "hero_length": 750,
        "pocket_w": 59,
        "pocket_h": 14,
    },
    # Group D (14 roots, indices 26..39): multi-tick rolled bare b8-s3 Watch6-v2 mature
    # inject_serpentines worlds spanning L in [310, 1185].
    {"world_index": 26, "kind": "rolled_serpentine", "hero_length": 310, "rollout_ticks": 2},
    {"world_index": 27, "kind": "rolled_serpentine", "hero_length": 380, "rollout_ticks": 3},
    {"world_index": 28, "kind": "rolled_serpentine", "hero_length": 460, "rollout_ticks": 4},
    {"world_index": 29, "kind": "rolled_serpentine", "hero_length": 520, "rollout_ticks": 2},
    {"world_index": 30, "kind": "rolled_serpentine", "hero_length": 600, "rollout_ticks": 3},
    {"world_index": 31, "kind": "rolled_serpentine", "hero_length": 680, "rollout_ticks": 4},
    {"world_index": 32, "kind": "rolled_serpentine", "hero_length": 760, "rollout_ticks": 5},
    {"world_index": 33, "kind": "rolled_serpentine", "hero_length": 840, "rollout_ticks": 2},
    {"world_index": 34, "kind": "rolled_serpentine", "hero_length": 900, "rollout_ticks": 3},
    {"world_index": 35, "kind": "rolled_serpentine", "hero_length": 960, "rollout_ticks": 4},
    {"world_index": 36, "kind": "rolled_serpentine", "hero_length": 1020, "rollout_ticks": 5},
    {"world_index": 37, "kind": "rolled_serpentine", "hero_length": 1080, "rollout_ticks": 6},
    {"world_index": 38, "kind": "rolled_serpentine", "hero_length": 1140, "rollout_ticks": 3},
    {"world_index": 39, "kind": "rolled_serpentine", "hero_length": 1180, "rollout_ticks": 4},
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


def _assert_not_exists_create_only(path: Path) -> None:
    sha_p = Path(f"{path}.sha256")
    comp_p = Path(f"{path}.complete")
    if path.exists() or sha_p.exists() or comp_p.exists():
        raise FileExistsError(f"Refusing to overwrite create-only evidence file: {path}")


def _write_atomic_bytes_or_fn(path: Path, write_fn: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_p = Path(tmp_name)
    try:
        write_fn(tmp_p)
        os.replace(tmp_p, path)
    except Exception:
        if tmp_p.exists():
            tmp_p.unlink()
        raise


def _write_create_only_evidence(path: Path, write_fn: Callable[[Path], None]) -> str:
    """Write ``path`` atomically with ``.sha256`` and ``.complete`` sidecars, refusing overwrites."""
    _assert_not_exists_create_only(path)
    _write_atomic_bytes_or_fn(path, write_fn)
    digest = _sha256_file(path)
    sidecar_line = f"{digest}  {path.name}\n"
    _write_atomic_bytes_or_fn(
        Path(f"{path}.sha256"),
        lambda tmp: tmp.write_text(sidecar_line, encoding="utf-8"),
    )
    _write_atomic_bytes_or_fn(
        Path(f"{path}.complete"),
        lambda tmp: tmp.write_text(sidecar_line, encoding="utf-8"),
    )
    return digest


def _write_deterministic_gzip_json(tmp_path: Path, payload: Any) -> None:
    raw_bytes = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with tmp_path.open("wb") as raw_f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, mtime=0) as gz_f:
            gz_f.write(raw_bytes)


def _verify_frozen_cohort_file(cohort_path: Path) -> str:
    sha_path = Path(f"{cohort_path}.sha256")
    comp_path = Path(f"{cohort_path}.complete")
    if not cohort_path.is_file() or not sha_path.is_file() or not comp_path.is_file():
        raise RuntimeError(
            f"Frozen CPU cohort or sidecars (.sha256/.complete) missing at {cohort_path}; "
            "run --mode freeze-cohort first."
        )
    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    comp_sha = comp_path.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = _sha256_file(cohort_path)
    if actual_sha != expected_sha or actual_sha != comp_sha:
        raise RuntimeError(
            f"Frozen CPU cohort SHA mismatch: actual={actual_sha}, "
            f"sha256={expected_sha}, complete={comp_sha}"
        )
    return actual_sha


def resolve_product_checkpoints(saved_dir: Path | None = None) -> tuple[Path, Path]:
    """Return verified ``(b8_s3_path, frp3_s12_path)`` under ``saved_snakes``."""
    candidates: list[Path] = []
    if saved_dir is not None:
        candidates.append(Path(saved_dir))
    else:
        env_dir = os.environ.get("SNAKE_CKPT_DIR", "").strip()
        if env_dir:
            candidates.append(Path(env_dir))
        candidates.append(Path(REPO_ROOT) / "saved_snakes")
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                common_saved = Path(proc.stdout.strip()).resolve().parent / "saved_snakes"
                if common_saved not in candidates:
                    candidates.append(common_saved)
        except OSError:
            pass
    base = candidates[0]
    for cand in candidates:
        if (cand / B8_S3_FILENAME).is_file() and (cand / FRP3_S12_FILENAME).is_file():
            base = cand
            break
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
        or cfg.game.min_boost_length != 5
        or cfg.game.boost_length_cost_frames != 3
        or cfg.game.arena_type != "rectangular"
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
    cpu_featurizer_backend: str = "numba",
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
    dummy_rt = Vector61Runtime(params=params, backend=cpu_featurizer_backend)
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

        rt = Vector61Runtime(params=params, backend=cpu_featurizer_backend)
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
    batch_rt = Vector61Runtime(params=params, backend=cpu_featurizer_backend)
    batch_controlled = np.zeros((len(seeds), 6), dtype=bool)
    batch_controlled[:, 1:] = True
    batch_rt.bind(batch_ref_sim, batch_controlled)
    batch_opp = object.__new__(Vector61SimdPolicy)
    batch_opp.__dict__ = dict(template_opp.__dict__)
    batch_opp.runtime = batch_rt
    bundle = SimpleWorldView(
        sim=batch_ref_sim,
        runtime=batch_rt,
        hero=shared_hero,
        opponents=[batch_opp] * 5,
        seed=int(seeds[0]),
    )
    target_dev = normalize_device(device)
    init_dev = torch.device("cpu") if target_dev.type == "mps" else target_dev
    device_policy = FastDevicePolicy.from_world(
        bundle,
        device=init_dev,
        prepared=False,
    )
    move_device_policy_to_device(device_policy, target_dev, e_count=len(seeds))
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

    target_dev = normalize_device(device)
    dev_sim = sim_cls.from_reference(
        batch_sim,
        device=target_dev,
        hero_respawn=False,
        prepared=True,
    )
    dev_policy = device_policy_template.repeat_interleave(1)
    move_device_policy_to_device(dev_policy, dev_sim.device, e_count=e_count)
    rt = dev_policy.vector_runtime
    dev = dev_policy.device

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
    assert_sim_and_policy_on_device(dev_sim, dev_policy, dev_sim.device)
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


@contextmanager
def _maybe_matched_branch_rng(enabled: bool) -> Iterator[None]:
    """Align CPU branch clones with ``CTDeviceSim``'s ``SplitMix64`` per-branch RNG stream."""
    if not enabled:
        yield
        return

    orig_reseed = open_stop_mod.reseed_branch_suffix
    orig_clone = open_stop_mod._clone_root

    def _patched_reseed(branch: Any, decision: int, suffix_id: int, *, namespace: str) -> int:
        seed64 = orig_reseed(branch, decision, suffix_id, namespace=namespace)
        branch._matched_branch_seed64 = int(seed64)
        return int(seed64)

    def _patched_clone(root: Any) -> Any:
        matched_seed64 = getattr(root, "_matched_branch_seed64", None)
        if hasattr(root, "_matched_branch_seed64"):
            delattr(root, "_matched_branch_seed64")
        branch = orig_clone(root)
        if matched_seed64 is not None:
            pos_seed = int(matched_seed64) & 0x7FFF_FFFF_FFFF_FFFF
            if pos_seed == 0:
                pos_seed = 1
            sim = branch.sim
            for env_idx in range(int(sim.E)):
                proxy = SplitMixEnvRngProxy(
                    pos_seed ^ env_idx,
                    0,
                    game_width=int(sim.cfg.game_width),
                    game_height=int(sim.cfg.game_height),
                    segment_size=int(sim.cfg.segment_size),
                    wall_thickness=int(sim.cfg.wall_thickness),
                )
                sim._rngs[env_idx]._draw_find_empty_candidate = (  # type: ignore[method-assign]
                    proxy._draw_find_empty_candidate
                )
        return branch

    open_stop_mod.reseed_branch_suffix = _patched_reseed
    open_stop_mod._clone_root = _patched_clone
    try:
        yield
    finally:
        open_stop_mod.reseed_branch_suffix = orig_reseed
        open_stop_mod._clone_root = orig_clone


def evaluate_cpu_open_stop_all_actions(
    world: CapturedWorld,
    decision: int,
    q: np.ndarray,
    legal: np.ndarray,
    *,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    namespace: str = OPEN_STOP_CRN_NAMESPACE,
    matched_rng: bool = False,
) -> dict[str, Any]:
    """Evaluate CPU :func:`open_stop_batch` both with early-stopping and for all 6 actions."""
    with _maybe_matched_branch_rng(matched_rng):
        early = open_stop_batch(
            world,
            int(decision),
            q,
            legal,
            branch_horizon=branch_horizon,
            namespace=namespace,
        )
        suffix = int(open_stop_mod.OPEN_STOP_SELECTION_SUFFIXES[0])
        root_clone = open_stop_mod._clone_root(world)
        all_outcomes: dict[int, dict[str, Any]] = {}
        with open_stop_mod.configure("kernel", open_stop_mod.RowStats()):
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
                    single_out = open_stop_mod._eval_branch_suffix(
                        root_clone,
                        int(a),
                        int(decision),
                        suffix,
                        branch_horizon=branch_horizon,
                        namespace=namespace,
                    )
                    out_a = dict(single_out)
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
        "min_hero_length": min(int(r["hero_length"]) for r in root_rows) if root_rows else 0,
        "max_hero_length": max(int(r["hero_length"]) for r in root_rows) if root_rows else 0,
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
    """Compute overall, certified-safe / not-safe split, and exact integer agreement statistics."""
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
    all_bool_agree = sum(
        1 for r in certificate_rows if r["certified_agree"] and not r["subcondition_diffs"]
    )

    surviving_rows = [
        r for r in certificate_rows if r["cpu_raw_survives"] and r["gpu_raw_survives"]
    ]
    surv_total = len(surviving_rows)

    def _int_eq(row: dict[str, Any], key: str) -> bool:
        return int(row["cpu_terminal_openness"][key]) == int(row["gpu_terminal_openness"][key])

    len_agree_all = sum(1 for r in certificate_rows if _int_eq(r, "length"))
    v8_area_agree_all = sum(1 for r in certificate_rows if _int_eq(r, "v8_best_safe_area"))
    pocket_r_agree_all = sum(1 for r in certificate_rows if _int_eq(r, "max_pocket_r"))
    all3_int_agree_all = sum(
        1
        for r in certificate_rows
        if _int_eq(r, "length") and _int_eq(r, "v8_best_safe_area") and _int_eq(r, "max_pocket_r")
    )

    len_agree_surv = sum(1 for r in surviving_rows if _int_eq(r, "length"))
    v8_area_agree_surv = sum(1 for r in surviving_rows if _int_eq(r, "v8_best_safe_area"))
    pocket_r_agree_surv = sum(1 for r in surviving_rows if _int_eq(r, "max_pocket_r"))
    all3_int_agree_surv = sum(
        1
        for r in surviving_rows
        if _int_eq(r, "length") and _int_eq(r, "v8_best_safe_area") and _int_eq(r, "max_pocket_r")
    )

    return {
        "total_certificates": total,
        "agreed_certificates": agree,
        "agreement_rate": float(agree / total) if total else 0.0,
        "all_boolean_subconditions_agreed": all_bool_agree,
        "all_boolean_subconditions_agreement_rate": (
            float(all_bool_agree / total) if total else 0.0
        ),
        "cpu_certified_safe_total": len(cpu_safe_rows),
        "cpu_certified_safe_agreed": safe_agree,
        "cpu_certified_safe_agreement_rate": (
            float(safe_agree / len(cpu_safe_rows)) if cpu_safe_rows else 0.0
        ),
        "cpu_not_safe_total": len(cpu_unsafe_rows),
        "cpu_not_safe_agreed": unsafe_agree,
        "cpu_not_safe_agreement_rate": (
            float(unsafe_agree / len(cpu_unsafe_rows)) if cpu_unsafe_rows else 0.0
        ),
        "legal_at_root_total": len(legal_rows),
        "legal_at_root_agreed": legal_agree,
        "legal_at_root_agreement_rate": (
            float(legal_agree / len(legal_rows)) if legal_rows else 0.0
        ),
        "early_stop_evaluated_total": len(early_rows),
        "early_stop_evaluated_agreed": early_agree,
        "early_stop_evaluated_agreement_rate": (
            float(early_agree / len(early_rows)) if early_rows else 0.0
        ),
        "raw_survives_agreed": raw_surv_agree,
        "raw_survives_agreement_rate": float(raw_surv_agree / total) if total else 0.0,
        "v8_open_agreed": v8_open_agree,
        "v8_open_agreement_rate": float(v8_open_agree / total) if total else 0.0,
        "pocket_open_agreed": pocket_open_agree,
        "pocket_open_agreement_rate": float(pocket_open_agree / total) if total else 0.0,
        "exact_int_all_certificates": {
            "total": total,
            "length_agreed": len_agree_all,
            "length_rate": float(len_agree_all / total) if total else 0.0,
            "v8_best_safe_area_agreed": v8_area_agree_all,
            "v8_best_safe_area_rate": float(v8_area_agree_all / total) if total else 0.0,
            "max_pocket_r_agreed": pocket_r_agree_all,
            "max_pocket_r_rate": float(pocket_r_agree_all / total) if total else 0.0,
            "all_three_agreed": all3_int_agree_all,
            "all_three_rate": float(all3_int_agree_all / total) if total else 0.0,
        },
        "exact_int_surviving_branches": {
            "total": surv_total,
            "length_agreed": len_agree_surv,
            "length_rate": float(len_agree_surv / surv_total) if surv_total else 0.0,
            "v8_best_safe_area_agreed": v8_area_agree_surv,
            "v8_best_safe_area_rate": (
                float(v8_area_agree_surv / surv_total) if surv_total else 0.0
            ),
            "max_pocket_r_agreed": pocket_r_agree_surv,
            "max_pocket_r_rate": float(pocket_r_agree_surv / surv_total) if surv_total else 0.0,
            "all_three_agreed": all3_int_agree_surv,
            "all_three_rate": float(all3_int_agree_surv / surv_total) if surv_total else 0.0,
        },
        "roots_total": roots_total,
        "roots_chosen_agreed": roots_chosen_agree,
        "roots_chosen_agreement_rate": (
            float(roots_chosen_agree / roots_total) if roots_total else 0.0
        ),
        "roots_bare_agreed": roots_bare_agree,
        "roots_bare_agreement_rate": (
            float(roots_bare_agree / roots_total) if roots_total else 0.0
        ),
        "certificate_disagreement_count": len(cert_disagreements),
        "any_subcondition_disagreement_count": len(sub_disagreements),
    }


def _set_snake_cells(
    sim: GridBatchSim,
    env: int,
    slot: int,
    cells_head_first: Sequence[tuple[int, int]],
) -> None:
    """Write ``cells_head_first`` into ``sim`` for ``(env, slot)`` and validate adjacency."""
    sim.bodies[env, slot] = 0
    n = len(cells_head_first)
    for i in range(n - 1):
        d = abs(cells_head_first[i][0] - cells_head_first[i + 1][0]) + abs(
            cells_head_first[i][1] - cells_head_first[i + 1][1]
        )
        if d != 1:
            raise ValueError(f"slot {slot} non-adjacent cells at index {i}")
    if len(set(cells_head_first)) != n:
        raise ValueError(f"slot {slot} has duplicate body cells")
    sim.bodies[env, slot, :n] = np.array(cells_head_first[::-1], dtype=np.int64)
    sim.head_ptr[env, slot] = n - 1
    sim.seg_count[env, slot] = n
    sim.length[env, slot] = n
    sim.alive[env, slot] = True
    sim.direction[env, slot] = _heading(list(cells_head_first))
    sim.boost_frames[env, slot] = 0
    sim.frames_since_food[env, slot] = 0
    sim._reward_prev_length[env, slot] = n


def _setup_doorway_world(
    w: CapturedWorld,
    hero_length: int,
    pocket_w: int,
    pocket_h: int,
) -> None:
    """Configure a dynamic doorway Watch6-v2 world and step 1 organic warmup frame."""
    sim = w.sim
    stem_head_to_top = [(pocket_w + 2, r) for r in range(pocket_h - 1, -1, -1)]
    rem_l = int(hero_length) - len(stem_head_to_top)
    body_tail: list[tuple[int, int]] = []
    for r_off in range(25):
        r = r_off
        cols = range(pocket_w + 3, 144) if r_off % 2 == 0 else range(143, pocket_w + 2, -1)
        for c in cols:
            body_tail.append((c, r))
            if len(body_tail) == rem_l:
                break
        if len(body_tail) == rem_l:
            break
    hero_cells = stem_head_to_top + body_tail
    if len(hero_cells) != int(hero_length):
        raise ValueError(f"Expected hero_length={hero_length}, got {len(hero_cells)}")
    _set_snake_cells(sim, 0, 0, hero_cells)

    nozzle3 = [
        (3, pocket_h),
        (4, pocket_h),
        (4, pocket_h - 1),
        (3, pocket_h - 1),
        (2, pocket_h - 1),
        (1, pocket_h - 1),
        (1, pocket_h - 2),
        (2, pocket_h - 2),
        (3, pocket_h - 2),
        (4, pocket_h - 2),
        (5, pocket_h - 2),
        (5, pocket_h - 1),
        (5, pocket_h),
        (5, pocket_h + 1),
        (4, pocket_h + 1),
        (3, pocket_h + 1),
        (2, pocket_h + 1),
        (1, pocket_h + 1),
        (1, pocket_h + 2),
        (2, pocket_h + 2),
        (3, pocket_h + 2),
        (4, pocket_h + 2),
        (5, pocket_h + 2),
        (6, pocket_h + 2),
        (6, pocket_h + 1),
        (6, pocket_h),
    ]
    east_wall = [(c, pocket_h) for c in range(7, pocket_w + 1)]
    opp1_down = [(pocket_w, r) for r in range(pocket_h + 1, pocket_h + 6)]
    opp1_serp: list[tuple[int, int]] = []
    for r_off in range(14):
        r = pocket_h + 6 + r_off
        cols = range(pocket_w, 6, -1) if r_off % 2 == 0 else range(7, pocket_w + 1)
        for c in cols:
            opp1_serp.append((c, r))
    opp1_cells = (nozzle3 + east_wall + opp1_down + opp1_serp)[:260]
    _set_snake_cells(sim, 0, 1, opp1_cells)

    for sidx, (r0, slen) in enumerate([(34, 300), (45, 260), (56, 220), (67, 180)], start=2):
        scells = serpentine_path(65, r0, 75, 8, slen)
        _set_snake_cells(sim, 0, sidx, scells)

    occupied: set[tuple[int, int]] = set()
    for sidx in range(6):
        n = int(sim.seg_count[0, sidx])
        for k in range(n):
            c = (int(sim.bodies[0, sidx, k, 0]), int(sim.bodies[0, sidx, k, 1]))
            occupied.add(c)
    keep = [c for c in sim.food_cells[0] if c not in occupied]
    for bait in [
        (pocket_w + 1, pocket_h - 1),
        (pocket_w, pocket_h - 1),
        (pocket_w - 1, pocket_h - 1),
        (0, pocket_h + 1),
        (0, pocket_h + 2),
    ]:
        if bait not in occupied:
            if bait in keep:
                keep.remove(bait)
            keep.insert(0, bait)
    keep = keep[: int(sim.cfg.max_food)]
    sim.food_cells[0] = keep
    sim.food_set[0] = set(keep)
    sim.corpse_cells[0].intersection_update(keep)
    sim._rebuild_env_grids(0, food=True)
    all_envs = np.ones(1, dtype=bool)
    sim._rebuild_traversed_from_heads(all_envs)
    sim._refresh_action_masks(all_envs)

    advance_to_next_live_hero_seam(w)
    w.runtime.prepare(sim)
    sim._policy_phase_prepared = True
    step_prepared(w, 2)
    advance_to_next_live_hero_seam(w)


def _setup_rolled_serpentine_world(
    w: CapturedWorld,
    hero_length: int,
    rollout_ticks: int,
    ego_cfg: EgoRasterConfig,
    *,
    ego_raster_backend: str = "numba",
) -> None:
    """Inject mature serpentine snakes and roll bare ``b8-s3`` for ``rollout_ticks`` frames."""
    lengths = [int(hero_length), 96, 72, 56, 40, 28]
    inject_serpentines([w.sim], lengths, box_height=10)
    w.sim._rebuild_env_grids(0, food=True)

    for _ in range(int(rollout_ticks)):
        advance_to_next_live_hero_seam(w)
        w.runtime.prepare(w.sim)
        w.sim._policy_phase_prepared = True
        obs = build_ego_raster(
            w.sim,
            ego_cfg,
            backend=ego_raster_backend,
            rows=np.array([[0, 0]], dtype=np.int64),
        )
        q = np.asarray(prepared_q(w, obs), dtype=np.float32)
        adv = np.asarray(w.sim.get_advisory_action_mask()[0, 0], dtype=bool)
        _, bare = ceiling._legal(q, adv)
        step_prepared(w, int(bare))
    advance_to_next_live_hero_seam(w)


def build_extension_worlds(
    b8_path: Path,
    frp3_path: Path,
    *,
    specs: Sequence[dict[str, Any]] | None = None,
    device: str = "cpu",
    cpu_featurizer_backend: str = "numba",
    ego_raster_backend: str = "numba",
) -> tuple[list[CapturedWorld], DevicePolicy, list[dict[str, Any]]]:
    """Build the 32 mature extension ``CapturedWorld`` roots at their pre-prepare hero seam."""
    if specs is None:
        specs = MATURE_EXTENSION_SPECS
    indices = [int(s["world_index"]) for s in specs]
    seeds = preflight_seeds(indices)
    worlds, dev_policy_template = build_watch6_captured_worlds(
        seeds,
        b8_path,
        frp3_path,
        max_capacity=DEFAULT_MAX_CAPACITY,
        long_hero_profiles=None,
        device=device,
        cpu_featurizer_backend=cpu_featurizer_backend,
    )
    ego_cfg = EgoRasterConfig(version="bv8")
    meta_rows: list[dict[str, Any]] = []
    for idx, (w, spec) in enumerate(zip(worlds, specs)):
        kind = str(spec["kind"])
        if kind.startswith("doorway"):
            _setup_doorway_world(
                w,
                int(spec["hero_length"]),
                int(spec["pocket_w"]),
                int(spec["pocket_h"]),
            )
        elif kind == "rolled_serpentine":
            _setup_rolled_serpentine_world(
                w,
                int(spec["hero_length"]),
                int(spec["rollout_ticks"]),
                ego_cfg,
                ego_raster_backend=ego_raster_backend,
            )
        else:
            raise ValueError(f"Unknown extension world kind: {kind!r}")
        meta_rows.append(
            {
                "cohort_root_index": int(idx),
                "world_index": int(spec["world_index"]),
                "world_seed": int(w.seed),
                "kind": kind,
                "decision_frame": int(w.sim.frame[0]),
                "hero_length_root": int(w.sim.length[0, 0]),
                "spec": dict(spec),
            }
        )
    return worlds, dev_policy_template, meta_rows


def evaluate_cpu_cohort_reference(
    worlds: Sequence[CapturedWorld],
    meta_rows: Sequence[dict[str, Any]],
    *,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    matched_rng: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Run CPU ``open_stop_batch`` on cloned roots and compute CPU-only cohort composition."""
    ego_cfg = EgoRasterConfig(version="bv8")
    cpu_roots: list[dict[str, Any]] = []

    surviving_total = 0
    surviving_enclosed = 0
    surviving_v8_false = 0
    surviving_pocket_false = 0
    surviving_v8_true_pocket_false = 0
    near_threshold_surviving = 0
    rescue_roots = 0
    hero_lengths: list[int] = []

    t0 = time.perf_counter()
    for w, meta in zip(worlds, meta_rows):
        w_cpu = open_stop_mod._clone_root(w)
        w_cpu.runtime.prepare(w_cpu.sim)
        w_cpu.sim._policy_phase_prepared = True
        obs = build_ego_raster(
            w_cpu.sim,
            ego_cfg,
            backend="numba",
            rows=np.array([[0, 0]], dtype=np.int64),
        )
        q = np.asarray(prepared_q(w_cpu, obs), dtype=np.float32)
        adv = np.asarray(w_cpu.sim.get_advisory_action_mask()[0, 0], dtype=bool)
        legal, _ = ceiling._legal(q, adv)
        decision = int(w_cpu.sim.frame[0])
        res = evaluate_cpu_open_stop_all_actions(
            w_cpu,
            decision,
            q,
            legal,
            branch_horizon=branch_horizon,
            namespace=OPEN_STOP_CRN_NAMESPACE,
            matched_rng=matched_rng,
        )
        res["root_meta"] = dict(meta)
        res["root_q"] = [float(x) for x in q.tolist()]
        res["root_legal"] = [bool(x) for x in legal.tolist()]
        cpu_roots.append(res)

        hero_lengths.append(int(w_cpu.sim.length[0, 0]))
        if int(res["chosen"]) != int(res["bare"]):
            rescue_roots += 1
        for a in range(6):
            out_a = res["all_outcomes"][a]
            if bool(out_a["raw_survives"]):
                surviving_total += 1
                op = out_a["terminal_openness"]
                length = int(op["length"])
                pocket_r = int(op["max_pocket_r"])
                v8_area = int(op["v8_best_safe_area"])
                if not bool(op["un_enclosed"]):
                    surviving_enclosed += 1
                    if not bool(op["v8_open"]):
                        surviving_v8_false += 1
                    if not bool(op["pocket_open"]):
                        surviving_pocket_false += 1
                    if bool(op["v8_open"]) and not bool(op["pocket_open"]):
                        surviving_v8_true_pocket_false += 1
                if (0.85 * length <= pocket_r <= 1.15 * length) or (136 <= v8_area <= 184):
                    near_threshold_surviving += 1
    cpu_wall = time.perf_counter() - t0

    composition = {
        "roots_total": len(worlds),
        "certificates_total": len(worlds) * 6,
        "hero_length_min": min(hero_lengths) if hero_lengths else 0,
        "hero_length_max": max(hero_lengths) if hero_lengths else 0,
        "hero_length_in_300_1200": bool(
            hero_lengths and 300 <= min(hero_lengths) and max(hero_lengths) <= 1200
        ),
        "surviving_branches_total": surviving_total,
        "surviving_but_enclosed_branches": surviving_enclosed,
        "surviving_v8_open_false": surviving_v8_false,
        "surviving_pocket_open_false": surviving_pocket_false,
        "surviving_v8_true_pocket_false": surviving_v8_true_pocket_false,
        "near_threshold_surviving_branches": near_threshold_surviving,
        "rescue_override_roots": rescue_roots,
        "composition_bars_met": bool(
            len(worlds) == len(EXTENSION_MATURE_WORLD_INDICES)
            and hero_lengths
            and 300 <= min(hero_lengths)
            and max(hero_lengths) <= 1200
            and surviving_enclosed >= EXT_MIN_SURVIVING_ENCLOSED
            and surviving_v8_false >= 1
            and surviving_pocket_false >= 1
            and near_threshold_surviving >= EXT_MIN_NEAR_THRESHOLD_SURVIVING
            and rescue_roots >= EXT_MIN_RESCUE_ROOTS
        ),
    }
    return cpu_roots, composition, float(cpu_wall)


def freeze_mature_cohort_cpu(
    *,
    out_dir: Path = DEFAULT_EXTENSION_OUT_DIR,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
) -> dict[str, Any]:
    """Audit seeds, verify 32-root CPU composition, and freeze root list before device run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / FROZEN_COHORT_FILENAME
    _assert_not_exists_create_only(cohort_path)

    seed_audit = audit_preflight_seeds()
    if not seed_audit["ok"]:
        raise RuntimeError(f"Seed audit failed for {PREFLIGHT_NAMESPACE}: {seed_audit}")

    b8_path, frp3_path = resolve_product_checkpoints()
    worlds, _, meta_rows = build_extension_worlds(b8_path, frp3_path, device="cpu")
    cpu_roots_cross, composition_cross, cpu_wall_cross = evaluate_cpu_cohort_reference(
        worlds,
        meta_rows,
        branch_horizon=branch_horizon,
        matched_rng=False,
    )
    if not composition_cross["composition_bars_met"]:
        raise RuntimeError(f"CPU cohort composition failed bars: {composition_cross}")

    frozen_roots_manifest: list[dict[str, Any]] = []
    for meta, cpu_r in zip(meta_rows, cpu_roots_cross):
        surv_enclosed_actions = [
            int(a)
            for a in range(6)
            if bool(cpu_r["all_outcomes"][a]["raw_survives"])
            and not bool(cpu_r["all_outcomes"][a]["terminal_openness"]["un_enclosed"])
        ]
        near_th_actions = [
            int(a)
            for a in range(6)
            if bool(cpu_r["all_outcomes"][a]["raw_survives"])
            and (
                (
                    0.85 * int(cpu_r["all_outcomes"][a]["terminal_openness"]["length"])
                    <= int(cpu_r["all_outcomes"][a]["terminal_openness"]["max_pocket_r"])
                    <= 1.15 * int(cpu_r["all_outcomes"][a]["terminal_openness"]["length"])
                )
                or (
                    136
                    <= int(cpu_r["all_outcomes"][a]["terminal_openness"]["v8_best_safe_area"])
                    <= 184
                )
            )
        ]
        frozen_roots_manifest.append(
            {
                **meta,
                "cpu_bare_action": int(cpu_r["bare"]),
                "cpu_chosen_action": int(cpu_r["chosen"]),
                "cpu_is_rescue_override": bool(int(cpu_r["chosen"]) != int(cpu_r["bare"])),
                "cpu_surviving_enclosed_actions": surv_enclosed_actions,
                "cpu_near_threshold_actions": near_th_actions,
            }
        )

    payload = {
        "schema": "gpu-sitl-preflight-ext-frozen-cohort/v1",
        "namespace": PREFLIGHT_NAMESPACE,
        "world_indices": list(EXTENSION_MATURE_WORLD_INDICES),
        "world_seeds": [int(m["world_seed"]) for m in meta_rows],
        "seed_audit_ok": bool(seed_audit["ok"]),
        "cpu_reference_composition": composition_cross,
        "cpu_reference_wall_seconds": float(cpu_wall_cross),
        "roots": frozen_roots_manifest,
    }
    cohort_sha = _write_create_only_evidence(
        cohort_path,
        lambda tmp: tmp.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
    )
    sha_path = Path(f"{cohort_path}.sha256")
    cache_path = out_dir / "_cpu_cohort_cache.json"
    _write_atomic_bytes_or_fn(
        cache_path,
        lambda tmp: tmp.write_text(
            json.dumps(
                {
                    "cohort_sha256": cohort_sha,
                    "branch_horizon": int(branch_horizon),
                    "cpu_cross_roots": cpu_roots_cross,
                    "cpu_cross_wall_seconds": float(cpu_wall_cross),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        ),
    )
    return {
        "cohort_path": str(cohort_path),
        "cohort_sha256": cohort_sha,
        "sha_sidecar_path": str(sha_path),
        "complete_sidecar_path": str(Path(f"{cohort_path}.complete")),
        "cpu_reference_composition": composition_cross,
        "cpu_reference_wall_seconds": float(cpu_wall_cross),
    }


def _outcome_for_action(root_res: dict[str, Any], action: int) -> dict[str, Any]:
    outcomes = root_res["all_outcomes"]
    if action in outcomes:
        return outcomes[action]
    return outcomes[str(action)]


def _build_comparison_rows(
    cohort_name: str,
    meta_rows: Sequence[dict[str, Any]],
    cpu_roots: Sequence[dict[str, Any]],
    gpu_roots: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-root, per-certificate, and disagreement rows between CPU and GPU results."""
    root_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []

    for e_idx, (meta, cpu_r, gpu_r) in enumerate(zip(meta_rows, cpu_roots, gpu_roots)):
        w_idx = int(meta["world_index"])
        w_seed = int(meta["world_seed"])
        decision = int(meta["decision_frame"])
        hero_len = int(meta["hero_length_root"])
        root_id = f"{cohort_name}:r{e_idx}:w{w_idx}:seed{w_seed}:frame{decision}"

        root_rows.append(
            {
                "cohort": cohort_name,
                "root_id": root_id,
                "cohort_root_index": int(e_idx),
                "world_index": w_idx,
                "world_seed": w_seed,
                "kind": str(meta["kind"]),
                "decision_frame": decision,
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
            c_out = _outcome_for_action(cpu_r, action)
            g_out = _outcome_for_action(gpu_r, action)
            in_cpu_early = action in cpu_r["evaluation_order"]
            in_gpu_early = action in gpu_r["evaluation_order"]

            cert_agree = bool(c_out["certified"]) == bool(g_out["certified"])
            raw_agree = bool(c_out["raw_survives"]) == bool(g_out["raw_survives"])
            c_open = c_out["terminal_openness"]
            g_open = g_out["terminal_openness"]
            v8_agree = bool(c_open["v8_open"]) == bool(g_open["v8_open"])
            pocket_agree = bool(c_open["pocket_open"]) == bool(g_open["pocket_open"])
            sub_diffs = _diff_subconditions(c_out, g_out)

            int_diffs: list[str] = []
            for ikey in ("length", "v8_best_safe_area", "max_pocket_r"):
                if int(c_open[ikey]) != int(g_open[ikey]):
                    int_diffs.append(f"{ikey}(cpu={c_open[ikey]},gpu={g_open[ikey]})")

            row_record = {
                "cohort": cohort_name,
                "root_id": root_id,
                "cohort_root_index": int(e_idx),
                "world_index": w_idx,
                "world_seed": w_seed,
                "kind": str(meta["kind"]),
                "decision_frame": decision,
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
                "integer_diffs": int_diffs,
            }
            certificate_rows.append(row_record)
            if not cert_agree or sub_diffs or int_diffs:
                disagreements.append(row_record)

    return root_rows, certificate_rows, disagreements


def run_extension_shard(
    *,
    out_dir: Path = DEFAULT_EXTENSION_OUT_DIR,
    shard_start: int = 0,
    shard_end: int = 32,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run ``MATURE_EXTENSION_SPECS[shard_start:shard_end]`` after verifying frozen cohort."""
    s_start = int(shard_start)
    s_end = int(shard_end)
    n_specs = len(MATURE_EXTENSION_SPECS)
    if s_start < 0 or s_end > n_specs or s_start >= s_end:
        raise ValueError(
            f"Invalid shard bounds [{s_start}, {s_end}) for {n_specs} MATURE_EXTENSION_SPECS"
        )
    out_dir = Path(out_dir)
    cohort_path = out_dir / FROZEN_COHORT_FILENAME
    actual_sha = _verify_frozen_cohort_file(cohort_path)

    specs_slice = MATURE_EXTENSION_SPECS[s_start:s_end]
    b8_path, frp3_path = resolve_product_checkpoints()
    worlds, dev_policy_tpl, meta_rows = build_extension_worlds(
        b8_path,
        frp3_path,
        specs=specs_slice,
        device=device,
    )
    for local_i, m in enumerate(meta_rows):
        m["cohort_root_index"] = s_start + local_i

    cpu_matched_roots, _, cpu_matched_wall = evaluate_cpu_cohort_reference(
        worlds,
        meta_rows,
        branch_horizon=branch_horizon,
        matched_rng=True,
    )
    cache_path = out_dir / "_cpu_cohort_cache.json"
    cached_ok = False
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cohort_sha256") == actual_sha and int(
            cached.get("branch_horizon", -1)
        ) == int(branch_horizon):
            cpu_cross_roots = cached["cpu_cross_roots"][s_start:s_end]
            cpu_cross_wall = float(cached["cpu_cross_wall_seconds"]) * (
                len(specs_slice) / float(n_specs)
            )
            cached_ok = True
    if not cached_ok:
        cpu_cross_roots, _, cpu_cross_wall = evaluate_cpu_cohort_reference(
            worlds,
            meta_rows,
            branch_horizon=branch_horizon,
            matched_rng=False,
        )

    dev_sim, dev_pol = snapshot_worlds_to_device(
        worlds,
        dev_policy_tpl,
        sim_cls=CTDeviceSim,
        device=device,
        vector_prepared=False,
    )
    decisions = [int(m["decision_frame"]) for m in meta_rows]
    world_seeds = [int(m["world_seed"]) for m in meta_rows]

    t0_gpu = time.perf_counter()
    gpu_batch_wavefront = batched_open_stop_h32(
        dev_sim,
        dev_pol,
        world_seeds=world_seeds,
        decisions=decisions,
        suffix_id=0,
        branch_horizon=branch_horizon,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=False,
        return_host_records=True,
    )
    gpu_wavefront_wall = time.perf_counter() - t0_gpu

    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{s_start:02d}_{s_end:02d}.json"
    shard_payload = {
        "cohort_sha256": actual_sha,
        "branch_horizon": int(branch_horizon),
        "shard_start": s_start,
        "shard_end": s_end,
        "meta_rows": meta_rows,
        "cpu_matched_roots": cpu_matched_roots,
        "cpu_cross_roots": cpu_cross_roots,
        "gpu_wavefront_roots": gpu_batch_wavefront["roots"],
        "cpu_matched_wall_seconds": float(cpu_matched_wall),
        "cpu_cross_wall_seconds": float(cpu_cross_wall),
        "gpu_wavefront_wall_seconds": float(gpu_wavefront_wall),
    }
    _write_atomic_bytes_or_fn(
        shard_path,
        lambda tmp: tmp.write_text(
            json.dumps(shard_payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
    )
    return {
        "shard_path": str(shard_path),
        "roots": len(meta_rows),
        "gpu_wavefront_wall_seconds": float(gpu_wavefront_wall),
        "cpu_matched_wall_seconds": float(cpu_matched_wall),
        "cpu_cross_wall_seconds": float(cpu_cross_wall),
    }


def finalize_extension_preflight(
    *,
    out_dir: Path = DEFAULT_EXTENSION_OUT_DIR,
    delta: float = RESCUE_MARGIN_DELTA_DEFAULT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Aggregate shard results for all 32 mature roots, run grad check, and write report."""
    out_dir = Path(out_dir)
    raw_gz_path = out_dir / "gpu_sitl_preflight_ext_certificates_20261109.json.gz"
    summary_path = out_dir / "gpu_sitl_preflight_ext_summary_20261109.json"
    _assert_not_exists_create_only(raw_gz_path)
    _assert_not_exists_create_only(summary_path)

    cohort_path = out_dir / FROZEN_COHORT_FILENAME
    actual_sha = _verify_frozen_cohort_file(cohort_path)
    frozen_cohort = json.loads(cohort_path.read_text(encoding="utf-8"))

    shard_dir = out_dir / "_shards"
    shard_files = sorted(shard_dir.glob("shard_*.json"))
    if not shard_files:
        raise RuntimeError(f"No shard_*.json files found in {shard_dir}")

    parsed_shards: list[tuple[int, int, Path, dict[str, Any]]] = []
    for sf in shard_files:
        sp = json.loads(sf.read_text(encoding="utf-8"))
        sh_sha = sp.get("cohort_sha256")
        if sh_sha is not None and sh_sha != actual_sha:
            raise RuntimeError(
                f"Shard {sf.name} cohort_sha256={sh_sha} does not match frozen cohort {actual_sha}"
            )
        s_start = int(sp["shard_start"])
        s_end = int(sp["shard_end"])
        if s_start < 0 or s_end <= s_start or len(sp["meta_rows"]) != (s_end - s_start):
            raise RuntimeError(f"Malformed shard bounds or meta_rows count in {sf.name}")
        parsed_shards.append((s_start, s_end, sf, sp))
    parsed_shards.sort(key=lambda item: (item[0], item[1], item[2].name))

    n_expected = len(MATURE_EXTENSION_SPECS)
    expected_cursor = 0
    meta_rows: list[dict[str, Any]] = []
    cpu_matched_roots: list[dict[str, Any]] = []
    cpu_cross_roots: list[dict[str, Any]] = []
    gpu_wavefront_roots: list[dict[str, Any]] = []
    cpu_matched_wall = 0.0
    cpu_cross_wall = 0.0
    gpu_wavefront_wall = 0.0

    for s_start, s_end, sf, sp in parsed_shards:
        if s_start != expected_cursor:
            raise RuntimeError(
                f"Non-contiguous or overlapping shard {sf.name}: expected shard_start="
                f"{expected_cursor}, got [{s_start}, {s_end})"
            )
        expected_cursor = s_end
        meta_rows.extend(sp["meta_rows"])
        cpu_matched_roots.extend(sp["cpu_matched_roots"])
        cpu_cross_roots.extend(sp["cpu_cross_roots"])
        gpu_wavefront_roots.extend(sp["gpu_wavefront_roots"])
        cpu_matched_wall += float(sp["cpu_matched_wall_seconds"])
        cpu_cross_wall += float(sp["cpu_cross_wall_seconds"])
        gpu_wavefront_wall += float(sp["gpu_wavefront_wall_seconds"])

    if expected_cursor != n_expected or len(meta_rows) != n_expected:
        raise RuntimeError(
            f"Expected contiguous shards covering 0..{n_expected}, got cursor={expected_cursor}, "
            f"roots={len(meta_rows)}"
        )

    frozen_roots = frozen_cohort["roots"]
    if len(frozen_roots) != n_expected:
        raise RuntimeError(f"Frozen cohort root count {len(frozen_roots)} != {n_expected}")
    for idx, (m_row, f_row) in enumerate(zip(meta_rows, frozen_roots)):
        for key in (
            "cohort_root_index",
            "world_index",
            "world_seed",
            "decision_frame",
            "hero_length_root",
        ):
            if int(m_row[key]) != int(f_row[key]):
                raise RuntimeError(
                    f"Shard root {idx} mismatch against frozen cohort on {key}: "
                    f"{m_row[key]} != {f_row[key]}"
                )

    matched_root_rows, matched_cert_rows, matched_disagreements = _build_comparison_rows(
        "mature_32_matched_rng_wavefront",
        meta_rows,
        cpu_matched_roots,
        gpu_wavefront_roots,
    )
    cross_root_rows, cross_cert_rows, cross_disagreements = _build_comparison_rows(
        "mature_32_cross_rng_wavefront",
        meta_rows,
        cpu_cross_roots,
        gpu_wavefront_roots,
    )

    summary_matched = summarize_agreement(matched_cert_rows, matched_root_rows)
    summary_cross = summarize_agreement(cross_cert_rows, cross_root_rows)

    # Registered rescue-hinge gradient check on rescue roots with both safe and unsafe legal actions
    eligible_grad_indices: list[int] = []
    for r_idx, r in enumerate(gpu_wavefront_roots):
        leg = [bool(_outcome_for_action(r, a)["legal"]) for a in range(6)]
        cert = [bool(_outcome_for_action(r, a)["certified"]) for a in range(6)]
        if any(l and c for l, c in zip(leg, cert)) and any(l and not c for l, c in zip(leg, cert)):
            eligible_grad_indices.append(r_idx)
    grad_indices = (
        eligible_grad_indices[:8]
        if eligible_grad_indices
        else list(range(min(8, len(MATURE_EXTENSION_SPECS))))
    )
    b8_path, frp3_path = resolve_product_checkpoints()
    grad_specs = tuple(MATURE_EXTENSION_SPECS[i] for i in grad_indices)
    grad_worlds, grad_pol_tpl, _ = build_extension_worlds(
        b8_path,
        frp3_path,
        specs=grad_specs,
        device=device,
    )
    grad_sim, _ = snapshot_worlds_to_device(
        grad_worlds,
        grad_pol_tpl,
        sim_cls=CTDeviceSim,
        device=device,
        vector_prepared=False,
    )
    legal_list = []
    safe_list = []
    unsafe_list = []
    astar_list = []
    for r_idx in grad_indices:
        r = gpu_wavefront_roots[r_idx]
        leg = [bool(_outcome_for_action(r, a)["legal"]) for a in range(6)]
        cert = [bool(_outcome_for_action(r, a)["certified"]) for a in range(6)]
        saf = [is_leg and is_cert for is_leg, is_cert in zip(leg, cert)]
        uns = [is_leg and (not is_cert) for is_leg, is_cert in zip(leg, cert)]
        legal_list.append(leg)
        safe_list.append(saf)
        unsafe_list.append(uns)
        astar_list.append(int(r["chosen"]))

    target_dev = normalize_device(device)
    gradient_report = margin_loss_gradient_check(
        str(b8_path),
        sim=grad_sim,
        delta=delta,
        device=target_dev,
        freeze_value_head=True,
        legal_mask=torch.tensor(legal_list, dtype=torch.bool, device=target_dev),
        safe_mask=torch.tensor(safe_list, dtype=torch.bool, device=target_dev),
        unsafe_mask=torch.tensor(unsafe_list, dtype=torch.bool, device=target_dev),
        a_star=torch.tensor(astar_list, dtype=torch.int64, device=target_dev),
    )

    seed_audit = audit_preflight_seeds()
    if not seed_audit["ok"]:
        raise RuntimeError(f"Seed audit failed for {PREFLIGHT_NAMESPACE}: {seed_audit}")
    comp = frozen_cohort["cpu_reference_composition"]

    part1_bool_pass = bool(summary_matched["agreed_certificates"] >= EXT_MATCHED_RNG_CERT_AGREE_MIN)
    part1_int_surv = summary_matched["exact_int_surviving_branches"]
    part1_int_all = summary_matched["exact_int_all_certificates"]
    part1_int_pass = bool(
        part1_int_surv["length_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
        and part1_int_surv["v8_best_safe_area_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
        and part1_int_surv["max_pocket_r_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
        and part1_int_all["length_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
        and part1_int_all["v8_best_safe_area_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
        and part1_int_all["max_pocket_r_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
    )
    part1_surviving_joint_all_three_pass = bool(
        part1_int_surv["all_three_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
    )
    part1_all_certs_joint_all_three_pass = bool(
        part1_int_all["all_three_rate"] >= EXT_MATCHED_RNG_INT_AGREE_RATE_MIN
    )
    part2_bool_pass = bool(summary_cross["agreed_certificates"] >= EXT_CROSS_RNG_CERT_AGREE_MIN)
    part2_chosen_pass = bool(summary_cross["roots_chosen_agreed"] >= EXT_CROSS_RNG_CHOSEN_AGREE_MIN)
    grad_pass = bool(
        gradient_report["all_grads_present"]
        and gradient_report["all_grads_finite"]
        and gradient_report["all_non_value_grads_finite_and_nonzero"]
    )
    comp_pass = bool(comp["composition_bars_met"])
    overall_pass = bool(
        comp_pass
        and part1_bool_pass
        and part1_int_pass
        and part2_bool_pass
        and part2_chosen_pass
        and grad_pass
    )
    integrity_checks = {
        "seed_audit_ok": bool(seed_audit["ok"]),
        "part1_matched_rng_surviving_all_three_ge_98pct": part1_surviving_joint_all_three_pass,
        "part1_matched_rng_all_certs_all_three_ge_98pct": part1_all_certs_joint_all_three_pass,
    }
    integrity_pass = bool(all(integrity_checks.values()))

    raw_payload = {
        "schema": "gpu-sitl-preflight-ext-raw/v1",
        "namespace": PREFLIGHT_NAMESPACE,
        "verdict": "PASS" if overall_pass else "FAIL",
        "integrity_pass": integrity_pass,
        "integrity_checks": integrity_checks,
        "frozen_cohort_path": str(cohort_path),
        "frozen_cohort_sha256": actual_sha,
        "cpu_reference_composition": comp,
        "seed_audit": seed_audit,
        "summary_matched_rng_wavefront": summary_matched,
        "summary_cross_rng_wavefront": summary_cross,
        "matched_rng_disagreements": matched_disagreements,
        "cross_rng_disagreements": cross_disagreements,
        "gradient_check": gradient_report,
        "matched_roots": matched_root_rows,
        "matched_certificates": matched_cert_rows,
        "cross_roots": cross_root_rows,
        "cross_certificates": cross_cert_rows,
    }
    raw_gz_sha256 = _write_create_only_evidence(
        raw_gz_path,
        lambda tmp: _write_deterministic_gzip_json(tmp, raw_payload),
    )

    summary_payload = {
        "schema": "gpu-sitl-preflight-ext-summary/v1",
        "namespace": PREFLIGHT_NAMESPACE,
        "verdict": "PASS" if overall_pass else "FAIL",
        "integrity_pass": integrity_pass,
        "integrity_checks": integrity_checks,
        "frozen_cohort_path": str(cohort_path),
        "frozen_cohort_sha256": actual_sha,
        "raw_certificates_gz_path": str(raw_gz_path),
        "raw_certificates_gz_sha256": raw_gz_sha256,
        "raw_certificates_gz_bytes": int(raw_gz_path.stat().st_size),
        "bar_checks": {
            "cpu_composition_bars_met": comp_pass,
            "part1_matched_rng_bool_ge_191_of_192": part1_bool_pass,
            "part1_matched_rng_exact_int_ge_98pct": part1_int_pass,
            "part2_cross_rng_bool_ge_186_of_192": part2_bool_pass,
            "part2_cross_rng_chosen_ge_30_of_32": part2_chosen_pass,
            "rescue_hinge_all_trainable_grads_finite_nonzero": grad_pass,
        },
        "cpu_reference_composition": comp,
        "summary_matched_rng_wavefront": summary_matched,
        "summary_cross_rng_wavefront": summary_cross,
        "matched_rng_disagreements": matched_disagreements,
        "cross_rng_disagreements": cross_disagreements,
        "gradient_check": gradient_report,
        "wall_seconds": {
            "gpu_wavefront_total": float(gpu_wavefront_wall),
            "cpu_matched_rng_total": float(cpu_matched_wall),
            "cpu_cross_rng_total": float(cpu_cross_wall),
        },
    }
    _write_create_only_evidence(
        summary_path,
        lambda tmp: tmp.write_text(
            json.dumps(summary_payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
    )
    return summary_payload


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
    raw_gz_path = out_dir / "gpu_sitl_preflight_certificates_20261109.json.gz"
    summary_path = out_dir / "gpu_sitl_preflight_summary_20261109.json"
    _assert_not_exists_create_only(raw_gz_path)
    _assert_not_exists_create_only(summary_path)

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
    base8_worlds, grad_policy_tpl = build_watch6_captured_worlds(
        natural_seeds + long_hero_seeds,
        b8_path,
        frp3_path,
        max_capacity=DEFAULT_MAX_CAPACITY,
        long_hero_profiles=None,
        device=device,
    )
    nat4_worlds, nat4_policy_tpl = build_watch6_captured_worlds(
        natural_seeds,
        b8_path,
        frp3_path,
        max_capacity=DEFAULT_MAX_CAPACITY,
        long_hero_profiles=None,
        device=device,
    )
    long4_worlds, _ = build_watch6_captured_worlds(
        long_hero_seeds,
        b8_path,
        frp3_path,
        max_capacity=DEFAULT_MAX_CAPACITY,
        long_hero_profiles=LONG_HERO_LENGTH_PROFILES,
        device=device,
    )
    grad_worlds = base8_worlds[:4] + long4_worlds
    for w in grad_worlds:
        advance_to_next_live_hero_seam(w)
    for w in nat4_worlds:
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
    # (cpu_fast_flood=False) and verify parity against CPU reference.
    nat4_sim, nat4_pol = snapshot_worlds_to_device(
        nat4_worlds,
        nat4_policy_tpl,
        sim_cls=CTDeviceSim,
        device=device,
        vector_prepared=False,
    )
    nat4_decisions = [int(w.sim.frame[0]) for w in nat4_worlds]
    t0_wavefront = time.perf_counter()
    wavefront_batch = batched_open_stop_h32(
        nat4_sim,
        nat4_pol,
        world_seeds=natural_seeds,
        decisions=nat4_decisions,
        suffix_id=0,
        branch_horizon=branch_horizon,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=False,
    )
    wavefront_wall_sec = time.perf_counter() - t0_wavefront
    wavefront_certs = len(natural_seeds) * 6
    ego_cfg = EgoRasterConfig(version="bv8")
    wavefront_agreed = 0
    for e_idx, w in enumerate(nat4_worlds):
        w_cpu = open_stop_mod._clone_root(w)
        w_cpu.runtime.prepare(w_cpu.sim)
        w_cpu.sim._policy_phase_prepared = True
        obs = build_ego_raster(
            w_cpu.sim,
            ego_cfg,
            backend="numba",
            rows=np.array([[0, 0]], dtype=np.int64),
        )
        q = np.asarray(prepared_q(w_cpu, obs), dtype=np.float32)
        adv = np.asarray(w_cpu.sim.get_advisory_action_mask()[0, 0], dtype=bool)
        legal, _ = ceiling._legal(q, adv)
        cpu_r = evaluate_cpu_open_stop_all_actions(
            w_cpu,
            nat4_decisions[e_idx],
            q,
            legal,
            branch_horizon=branch_horizon,
            namespace=OPEN_STOP_CRN_NAMESPACE,
            matched_rng=True,
        )
        gpu_r = wavefront_batch["roots"][e_idx]
        for a in range(6):
            if bool(cpu_r["all_outcomes"][a]["certified"]) == bool(
                gpu_r["all_outcomes"][a]["certified"]
            ):
                wavefront_agreed += 1

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
            "agreed_certificates": int(wavefront_agreed),
            "wall_seconds": float(wavefront_wall_sec),
            "certs_per_sec": (
                float(wavefront_certs / wavefront_wall_sec) if wavefront_wall_sec > 0 else 0.0
            ),
        },
    }

    natural_pass = bool(summary_natural["agreement_rate"] >= PASS_BAR_AGREEMENT)
    long_hero_pass = bool(summary_long_hero["agreement_rate"] >= PASS_BAR_AGREEMENT)
    long_hero_min_length_pass = bool(int(long_hero_res["min_hero_length"]) >= 800)
    combined_pass = bool(summary_combined["agreement_rate"] >= PASS_BAR_AGREEMENT)
    wavefront_pass = bool(wavefront_agreed == wavefront_certs)
    grad_finite_pass = bool(
        gradient_report["all_grads_present"]
        and gradient_report["all_grads_finite"]
        and gradient_report["all_non_value_grads_finite_and_nonzero"]
    )
    overall_pass = bool(natural_pass and combined_pass and grad_finite_pass)
    integrity_checks = {
        "seed_audit_ok": bool(seed_audit["ok"]),
        "long_hero_agreement_ge_99pct": long_hero_pass,
        "long_hero_min_length_ge_800": long_hero_min_length_pass,
        "wavefront_24certs_exact_agree": wavefront_pass,
    }
    integrity_pass = bool(all(integrity_checks.values()))

    total_wall_seconds = time.perf_counter() - t_start

    raw_payload = {
        "schema": "gpu-sitl-preflight-raw/v1",
        "namespace": PREFLIGHT_NAMESPACE,
        "verdict": "PASS" if overall_pass else "FAIL",
        "integrity_pass": integrity_pass,
        "integrity_checks": integrity_checks,
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

    raw_gz_sha256 = _write_create_only_evidence(
        raw_gz_path,
        lambda tmp: _write_deterministic_gzip_json(tmp, raw_payload),
    )
    summary_payload = {
        "schema": "gpu-sitl-preflight-summary/v1",
        "namespace": PREFLIGHT_NAMESPACE,
        "verdict": "PASS" if overall_pass else "FAIL",
        "integrity_pass": integrity_pass,
        "integrity_checks": integrity_checks,
        "pass_bar": {
            "branch_certificate_agreement_min": PASS_BAR_AGREEMENT,
            "all_ego2s_gradients_finite": True,
        },
        "bar_checks": {
            "natural_agreement_ge_99pct": natural_pass,
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
    _write_create_only_evidence(
        summary_path,
        lambda tmp: tmp.write_text(
            json.dumps(summary_payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
    )
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step B1a / B1a-ext Preflight on Mac CPU.")
    parser.add_argument(
        "--mode",
        choices=("b1a", "freeze-cohort", "ext-shard", "ext-finalize"),
        default="b1a",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact directory for output.",
    )
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=32)
    parser.add_argument("--natural-ticks", type=int, default=DEFAULT_NATURAL_TICKS)
    parser.add_argument("--long-hero-ticks", type=int, default=DEFAULT_LONG_HERO_TICKS)
    parser.add_argument("--branch-horizon", type=int, default=OPEN_STOP_BRANCH_HORIZON)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.mode == "freeze-cohort":
        out_dir = args.out_dir or DEFAULT_EXTENSION_OUT_DIR
        res = freeze_mature_cohort_cpu(out_dir=out_dir, branch_horizon=args.branch_horizon)
        print(json.dumps(res, indent=2))
        return 0
    if args.mode == "ext-shard":
        out_dir = args.out_dir or DEFAULT_EXTENSION_OUT_DIR
        res = run_extension_shard(
            out_dir=out_dir,
            shard_start=args.shard_start,
            shard_end=args.shard_end,
            branch_horizon=args.branch_horizon,
            device="cpu",
        )
        print(json.dumps(res, indent=2))
        return 0
    if args.mode == "ext-finalize":
        out_dir = args.out_dir or DEFAULT_EXTENSION_OUT_DIR
        res = finalize_extension_preflight(
            out_dir=out_dir,
            delta=(args.delta if args.delta is not None else RESCUE_MARGIN_DELTA_DEFAULT),
            device="cpu",
        )
        print(json.dumps(res["bar_checks"], indent=2))
        return 0 if res.get("verdict") == "PASS" else 1

    out_dir = args.out_dir or DEFAULT_ARTIFACT_OUT_DIR
    summary = run_preflight(
        out_dir=out_dir,
        natural_ticks=args.natural_ticks,
        long_hero_ticks=args.long_hero_ticks,
        branch_horizon=args.branch_horizon,
        delta=(args.delta if args.delta is not None else MARGIN_DELTA_DEFAULT),
        device="cpu",
    )
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "integrity_pass": summary["integrity_pass"],
                "integrity_checks": summary["integrity_checks"],
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
    return 0 if summary.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
