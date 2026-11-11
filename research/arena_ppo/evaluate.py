"""Fixed-opponent evaluation tool for trained Stage 1 `research.arena_ppo` policies.

Evaluates Arena PPO checkpoints on the exact mechanics-v2 `TrainSim` (`DeviceSim`,
`83x145` rectangular grid, `6` snakes, continuous respawn) they were trained on,
against fixed opponents:
  - `random_safe`: vectorized safe-normal-move anchor (`advisory_mask[:, :, :3]`)
  - `greedy_food`: vectorized nearest-pellet safe-normal-move anchor
  - `b8` (or `b8-s3`, `b8:<path>`): released bare DQN champion `b8-s3` (`ego2s-bv8`)
    using exact device `ego2s-b` + `v8` feature extraction and advisory Q-masking
  - `untrained`: freshly initialized network of the same `(net_arch, net_size)`
  - `ckpt:<path>`: another Arena PPO checkpoint for head-to-head evaluation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from research.arena_ppo.net import (
    VALID_NET_ARCHS,
    ArenaNet,
    ArenaPolicyModel,
    NetArch,
    NetSize,
    build_arena_model,
    default_actor_chunk_envs,
)
from research.arena_ppo.obs import (
    ArenaEgoObs,
    SharedWorldObs,
    build_arena_ego_obs_gpu,
    build_shared_world_obs_gpu,
)
from research.arena_ppo.ppo import (
    ARENA_PPO_SCHEMA,
    ArenaPPOConfig,
    StopRequestedError,
)
from research.arena_ppo.runtime import (
    SignalStopFlag,
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    preload_native_sim_extensions,
    sha256_file,
    synchronize_device,
    write_create_only_json,
)
from research.arena_ppo.sim import (
    DEFAULT_GRID_H,
    DEFAULT_GRID_W,
    DEFAULT_NUM_SNAKES,
    StepInfo,
    TrainSim,
)
from research.gpu_sim_spike import ego_device, r2_cache, v8_device, v8_indexed
from research.gpu_sim_spike.device_sim import (
    _CARDINAL,
    _ENV_TENSOR_FIELDS,
    DeviceSim,
    _splitmix_positive,
)
from src.core.mechanics_constants import HEADON_SIZE_RATIO
from src.model.ego2s_network import (
    OBS_SPEC_BV8,
    Ego2sNet,
    load_ego2s_checkpoint,
    spec_for,
)
from src.simd_env.batch_sim import DEATH_BODY, DEATH_HEAD, DEATH_SELF, DEATH_WALL

EVAL_SCHEMA: str = "arena-ppo-eval/v1"
ARENA_PPO_EVAL_WEIGHTS_SCHEMA: str = "arena-ppo-eval-weights/v1"
DEFAULT_MAX_CAPACITY: int = 1600
DEFAULT_RING_CHUNK: int = 512

# Sized so even a pure `random_safe` board (~0.33 deaths / 1k seat-ticks on 83x145 with 6 snakes)
# yields ~380 >= 300 completed learner lives at K = 1 (`256 * 4500 = 1.152M` learner seat-ticks).
DEFAULT_EVAL_WORLDS: int = 256
DEFAULT_EVAL_TICKS: int = 4500
DEFAULT_EVAL_BURNIN: int = 250
DEFAULT_EVAL_SEED: int = 20261111
DEFAULT_B8_ENV_CHUNK: int = 32
DEFAULT_V8_LAMBDA: float = 8.0

B8_S3_SHA256: str = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
B8_REPO_RELPATH: Path = Path("saved_snakes/b8_s3_t7_20261016.pth")
B8_DEFAULT_ARTIFACT_PATH: Path = (
    REPO_ROOT.parent / "snake-dqn-artifacts" / "redesign-m4b-mac-20261013" / "B8_s3" / "final.pth"
)

SCRIPTED_OPPONENT_NAMES: tuple[str, ...] = ("random_safe", "greedy_food")
SEAT_ASSIGNMENT_MODES: tuple[str, ...] = ("rotate", "prefix")


@dataclass
class LoadedArenaPolicy:
    """Loaded or freshly initialized Arena PPO policy plus checkpoint provenance."""

    model: ArenaPolicyModel
    net_arch: NetArch
    net_size: NetSize
    grid_h: int
    grid_w: int
    num_snakes: int
    max_capacity: int
    ring_chunk: int
    actor_chunk_envs: int
    amp: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StepCollisionAttribution:
    """Exact per-tick death and killer->victim breakdown across all `(E, S)` seats."""

    wall_deaths: torch.Tensor  # (E, S) bool
    self_deaths: torch.Tensor  # (E, S) bool
    pair_kills: torch.Tensor  # (E, S, S) int64: [env, killer, victim] in {0, 1}
    pair_mutual: torch.Tensor  # (E, S, S) bool: symmetric [env, seat_a, seat_b]


def _clean_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    return {
        (k[len(prefix) :] if isinstance(k, str) and k.startswith(prefix) else k): v
        for k, v in raw_state.items()
    }


def _resolve_net_arch(raw_arch: Any) -> NetArch:
    arch = str(raw_arch or "per_seat").strip().lower()
    if arch not in VALID_NET_ARCHS:
        raise ValueError(f"Unsupported Arena PPO net_arch={raw_arch!r}; expected {VALID_NET_ARCHS}")
    if arch == "shared_trunk":
        return "shared_trunk"
    if arch == "ego_only":
        return "ego_only"
    return "per_seat"


def _resolve_net_size(raw_size: Any) -> NetSize:
    sz = str(raw_size or "S").strip().upper()
    if sz not in ("S", "M"):
        raise ValueError(f"Unsupported Arena PPO net_size={raw_size!r}; expected 'S' or 'M'")
    return "S" if sz == "S" else "M"


def load_arena_ppo_policy(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    actor_chunk_override: int | None = None,
) -> LoadedArenaPolicy:
    """Load an Arena PPO checkpoint (`arena-ppo-stage1/v1` or `arena-ppo-eval-weights/v1`).

    Args:
        checkpoint_path: Path to `.pt` training checkpoint or `.pth` eval weights.
        device: Target execution device (`cpu`, `cuda`, or `mps`).
        actor_chunk_override: Optional world-chunk override for forward batching.

    Returns:
        `LoadedArenaPolicy` with `.model` in `.eval()` mode on `device`.
    """
    resolved_dev = normalize_device(device)
    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Arena PPO checkpoint not found: {ckpt_file}")

    digest = sha256_file(ckpt_file)
    blob = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"Checkpoint {ckpt_file} did not deserialize to a dict")

    schema = str(blob.get("schema") or "")
    net_cfg = blob.get("net_config") if isinstance(blob.get("net_config"), dict) else {}

    if schema == ARENA_PPO_SCHEMA:
        raw_cfg = dict(blob.get("config") or {})
        raw_cfg.setdefault("net_arch", net_cfg.get("net_arch", "per_seat"))
        cfg = ArenaPPOConfig.from_dict(raw_cfg)
        net_arch = cfg.net_arch
        net_size = cfg.net_size
        grid_h = int(cfg.grid_h)
        grid_w = int(cfg.grid_w)
        num_snakes = int(cfg.num_snakes)
        max_capacity = int(cfg.max_capacity)
        ring_chunk = int(cfg.ring_chunk)
        actor_chunk = int(actor_chunk_override or cfg.actor_chunk_envs or 256)
        amp = bool(net_cfg.get("use_bf16_autocast", cfg.amp))
        state_dict = blob.get("model_state_dict")
        counters = blob.get("counters") if isinstance(blob.get("counters"), dict) else {}
        updates_completed = int(counters.get("updates_completed", 0))
        agent_steps_collected = int(counters.get("agent_steps_collected", 0))
        world_steps_collected = int(counters.get("world_steps_collected", 0))
    elif schema == ARENA_PPO_EVAL_WEIGHTS_SCHEMA:
        net_arch = _resolve_net_arch(blob.get("net_arch") or net_cfg.get("net_arch"))
        net_size = _resolve_net_size(net_cfg.get("size", "S"))
        grid_h = DEFAULT_GRID_H
        grid_w = DEFAULT_GRID_W
        num_snakes = DEFAULT_NUM_SNAKES
        max_capacity = DEFAULT_MAX_CAPACITY
        ring_chunk = DEFAULT_RING_CHUNK
        amp = bool(net_cfg.get("use_bf16_autocast", True))
        actor_chunk = int(
            actor_chunk_override or default_actor_chunk_envs(net_size, net_arch=net_arch)
        )
        state_dict = blob.get("state_dict")
        updates_completed = int(blob.get("updates_completed", 0))
        agent_steps_collected = int(blob.get("agent_steps_collected", 0))
        world_steps_collected = agent_steps_collected // max(1, num_snakes)
    else:
        raise ValueError(
            f"Unsupported Arena PPO checkpoint schema {schema!r} at {ckpt_file}; "
            f"expected {ARENA_PPO_SCHEMA!r} or {ARENA_PPO_EVAL_WEIGHTS_SCHEMA!r}"
        )

    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {ckpt_file} is missing state_dict/model_state_dict")

    model = build_arena_model(
        net_arch,
        net_size,
        grid_h=grid_h,
        grid_w=grid_w,
        compile_model=False,
        amp=amp,
    )
    model.load_state_dict(_clean_state_dict(state_dict))
    model = model.to(resolved_dev).eval()

    meta = {
        "path": str(ckpt_file),
        "sha256": digest,
        "schema": schema,
        "net_arch": net_arch,
        "net_size": net_size,
        "updates_completed": updates_completed,
        "agent_steps_collected": agent_steps_collected,
        "world_steps_collected": world_steps_collected,
    }
    return LoadedArenaPolicy(
        model=model,
        net_arch=net_arch,
        net_size=net_size,
        grid_h=grid_h,
        grid_w=grid_w,
        num_snakes=num_snakes,
        max_capacity=max_capacity,
        ring_chunk=ring_chunk,
        actor_chunk_envs=actor_chunk,
        amp=amp,
        metadata=meta,
    )


def build_untrained_arena_policy(
    *,
    net_arch: NetArch,
    net_size: NetSize,
    device: str | torch.device,
    seed: int,
    grid_h: int = DEFAULT_GRID_H,
    grid_w: int = DEFAULT_GRID_W,
    num_snakes: int = DEFAULT_NUM_SNAKES,
    max_capacity: int = DEFAULT_MAX_CAPACITY,
    ring_chunk: int = DEFAULT_RING_CHUNK,
    actor_chunk_envs: int | None = None,
    amp: bool = True,
) -> LoadedArenaPolicy:
    """Construct a deterministically initialized untrained network of the same design."""
    resolved_dev = normalize_device(device)
    init_key = f"arena-ppo-eval/untrained-init|{net_arch}|{net_size}|{grid_h}x{grid_w}|{int(seed)}"
    init_seed = int.from_bytes(hashlib.sha256(init_key.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = build_arena_model(
            net_arch,
            net_size,
            grid_h=grid_h,
            grid_w=grid_w,
            compile_model=False,
            amp=amp,
        )
    model = model.to(resolved_dev).eval()
    chunk = int(actor_chunk_envs or default_actor_chunk_envs(net_size, net_arch=net_arch))
    return LoadedArenaPolicy(
        model=model,
        net_arch=net_arch,
        net_size=net_size,
        grid_h=grid_h,
        grid_w=grid_w,
        num_snakes=num_snakes,
        max_capacity=max_capacity,
        ring_chunk=ring_chunk,
        actor_chunk_envs=chunk,
        amp=amp,
        metadata={
            "path": None,
            "sha256": None,
            "schema": "untrained",
            "init_seed": init_seed,
            "net_arch": net_arch,
            "net_size": net_size,
            "updates_completed": 0,
            "agent_steps_collected": 0,
            "world_steps_collected": 0,
        },
    )


@torch.no_grad()
def build_trainer_observations_and_mask(
    train_sim: TrainSim,
    net_arch: NetArch,
    *,
    env_slice: slice | None = None,
    ring_chunk: int = DEFAULT_RING_CHUNK,
) -> tuple[ArenaEgoObs | SharedWorldObs, torch.Tensor]:
    """Build exact trainer observations and legal action mask on a prepared `TrainSim`.

    Calls the identical `obs.py` and `sim.legal_action_mask()` entrypoints used by
    `ArenaPPOTrainer._act_all_seats_chunked`.
    """
    if not bool(train_sim.sim._prepared):
        raise RuntimeError(
            "TrainSim.prepare() must be called before building evaluation observations"
        )
    full_legal = train_sim.legal_action_mask()
    legal = full_legal if env_slice is None else full_legal[env_slice]
    if net_arch == "per_seat":
        obs: ArenaEgoObs | SharedWorldObs = build_arena_ego_obs_gpu(
            train_sim, env_slice=env_slice, ring_chunk=ring_chunk
        )
    else:
        obs = build_shared_world_obs_gpu(train_sim, env_slice=env_slice, ring_chunk=ring_chunk)
    return obs, legal


@torch.no_grad()
def act_arena_policy_chunked(
    train_sim: TrainSim,
    policy: LoadedArenaPolicy,
    legal_mask: torch.Tensor,
    *,
    greedy: bool = False,
    actor_chunk_envs: int | None = None,
) -> torch.Tensor:
    """Run an Arena PPO policy across all `(E, S)` seats using trainer observation chunks."""
    e_total, s_total = train_sim.E, train_sim.S
    chunk = max(1, min(e_total, int(actor_chunk_envs or policy.actor_chunk_envs)))
    is_per_seat = isinstance(policy.model, ArenaNet)

    if chunk == e_total:
        if is_per_seat:
            obs = build_arena_ego_obs_gpu(train_sim, ring_chunk=policy.ring_chunk)
            out = policy.model.act(
                obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=greedy
            )
        else:
            shared_obs = build_shared_world_obs_gpu(train_sim, ring_chunk=policy.ring_chunk)
            out = policy.model.act(shared_obs, legal_mask=legal_mask, greedy=greedy)
        return out.actions.reshape(e_total, s_total)

    actions = torch.empty((e_total, s_total), dtype=torch.long, device=train_sim.device)
    for start in range(0, e_total, chunk):
        end = min(e_total, start + chunk)
        sl = slice(start, end)
        if is_per_seat:
            obs = build_arena_ego_obs_gpu(train_sim, env_slice=sl, ring_chunk=policy.ring_chunk)
            out = policy.model.act(
                obs.arena,
                obs.ego,
                obs.scalars,
                legal_mask=legal_mask[sl],
                greedy=greedy,
            )
        else:
            shared_obs = build_shared_world_obs_gpu(
                train_sim, env_slice=sl, ring_chunk=policy.ring_chunk
            )
            out = policy.model.act(shared_obs, legal_mask=legal_mask[sl], greedy=greedy)
        actions[sl] = out.actions.reshape(end - start, s_total)
    return actions


@torch.no_grad()
def scripted_random_safe_actions(train_sim: TrainSim) -> torch.Tensor:
    """Vectorized `random_safe` heuristic on `TrainSim` tensors (`(E, S)` int64 in `0..2`).

    Uses `DeviceSim.advisory_mask()[:, :, :3]` (1-step non-fatal normal moves) with
    fallback to action `1` (straight) when all 3 normal actions collide, matching
    `src.evaluation.anchors.random_safe_action`. Choices among safe normal actions are
    drawn uniformly via a stateless 63-bit SplitMix mixer keyed by
    `(world_seed=rng_key[e], slot=s, transition_frame=frame[e]-1)`.
    """
    if not bool(train_sim.sim._prepared):
        raise RuntimeError(
            "TrainSim.prepare() must be called before scripted_random_safe_actions()"
        )
    sim = train_sim.sim
    normal_safe = sim.advisory_mask()[:, :, :3]
    safe_count = normal_safe.sum(dim=-1)

    slot_idx = torch.arange(sim.S, dtype=torch.long, device=sim.device)[None, :]
    transition_frame = (sim.frame[:, None] - 1).clamp_min(0)
    # Distinct odd golden-ratio constants mix (rng_key, slot) and transition_frame cleanly
    slot_salt = slot_idx * 5871781006564002453 + 1442695040888963407
    mixed_key = _splitmix_positive(sim.rng_key[:, None] ^ slot_salt, transition_frame + 1)
    rank = mixed_key.remainder(safe_count.clamp_min(1))

    picked = ((normal_safe.long().cumsum(dim=-1) - 1 == rank[..., None]) & normal_safe).long()
    chosen = picked.argmax(dim=-1)
    actions = torch.where(safe_count > 0, chosen, torch.ones_like(chosen))
    return torch.where(sim.alive, actions, torch.ones_like(actions))


def resolve_b8_checkpoint_path(explicit_path: str | Path | None = None) -> tuple[Path, str]:
    """Resolve bare `b8-s3` checkpoint path and SHA-256 digest.

    When `explicit_path` is None, resolves the canonical released `b8-s3` checkpoint
    (`saved_snakes/b8_s3_t7_20261016.pth` or the local artifact store copy) and strictly
    verifies `B8_S3_SHA256`. When `explicit_path` is supplied (e.g. `b8:<path>` or
    `--b8-checkpoint`), verifies file existence and returns its actual SHA-256.
    """
    if explicit_path is not None:
        cand = Path(explicit_path).expanduser().resolve()
        if not cand.is_file():
            raise FileNotFoundError(f"Explicit B8 checkpoint not found: {cand}")
        return cand, sha256_file(cand)

    candidates: list[Path] = [
        (REPO_ROOT / B8_REPO_RELPATH).resolve(),
        B8_DEFAULT_ARTIFACT_PATH.resolve(),
    ]
    env_override = os.environ.get("ARENA_PPO_B8_CHECKPOINT", "").strip()
    if env_override:
        candidates.insert(0, Path(env_override).expanduser().resolve())

    try:
        from web.backend.served_checkpoint import resolve_served_checkpoint

        served = resolve_served_checkpoint("b8-s3")
        if served.path:
            candidates.insert(0, Path(served.path).resolve())
    except Exception:
        pass

    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen or not cand.is_file():
            seen.add(cand)
            continue
        seen.add(cand)
        digest = sha256_file(cand)
        if digest == B8_S3_SHA256:
            return cand, digest

    raise FileNotFoundError(
        "Could not locate released bare b8-s3 checkpoint with SHA-256 "
        f"{B8_S3_SHA256}; checked {[str(c) for c in candidates]}"
    )


def _slice_device_sim_view(sim: DeviceSim, start: int, end: int) -> DeviceSim:
    """Create a read-only world-sliced `DeviceSim` view over environments `[start:end]`."""
    if start == 0 and end == sim.E:
        return sim
    sub = object.__new__(DeviceSim)
    sub.__dict__ = sim.__dict__.copy()
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            setattr(sub, name, val[start:end])
    sub.E = int(end - start)
    sub.exact_mechanics = sim.exact_mechanics
    r2_cache.disable(sub)
    return sub


@torch.no_grad()
def compute_b8_observations_and_actions(
    sim: DeviceSim,
    net: Ego2sNet,
    rows: torch.Tensor,
    *,
    b8_env_chunk: int = DEFAULT_B8_ENV_CHUNK,
    v8_lambda: float = DEFAULT_V8_LAMBDA,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute exact `ego2s-bv8` (or `ego2s-b`) features, Q-values, and advisory actions.

    Matches the product serving / SIMD path (`Ego2sServingPolicy` + `AISnake` advisory
    mask with normal-only `0..2` fallback when all 6 advisory actions are blocked).

    Args:
        sim: Prepared `DeviceSim` instance.
        net: Loaded `Ego2sNet` in eval mode on `sim.device`.
        rows: Tensor of shape `(N, 2)` holding `(env_idx, slot_idx)` pairs.
        b8_env_chunk: Maximum worlds processed per feature-extraction chunk.
        v8_lambda: v8 tail-area reranking parameter used by the frozen v8 feature extractor.

    Returns:
        Tuple `((local, glob, scalars), q_values, chosen_actions)`.
    """
    dev = sim.device
    rows = rows.to(device=dev, dtype=torch.long).reshape(-1, 2)
    n_rows = int(rows.shape[0])
    obs_spec = spec_for(net)
    needs_v8 = obs_spec == OBS_SPEC_BV8
    n_scalars = int(net.config["n_scalars"])
    local_ch = int(net.config["local_channels"])

    if n_rows == 0:
        empty_loc = torch.zeros((0, local_ch, 31, 31), dtype=torch.uint8, device=dev)
        empty_glob = torch.zeros((0, 4, 37, 37), dtype=torch.uint8, device=dev)
        empty_sca = torch.zeros((0, n_scalars), dtype=torch.float32, device=dev)
        empty_q = torch.zeros((0, 6), dtype=torch.float32, device=dev)
        empty_act = torch.zeros((0,), dtype=torch.long, device=dev)
        return (empty_loc, empty_glob, empty_sca), empty_q, empty_act

    chunk_e = max(1, int(b8_env_chunk))
    advisory_all = sim.advisory_mask()

    if sim.E <= chunk_e:
        row_adv = advisory_all[rows[:, 0], rows[:, 1]]
        v8_feat: torch.Tensor | None = None
        if needs_v8:
            zero_q = torch.zeros((n_rows, 6), dtype=torch.float32, device=dev)
            v8_feat = (
                v8_device.v8_decide(sim, rows, zero_q, row_adv, v8_lambda, features=True)
                .feature_planes()
                .reshape(n_rows, 30)
            )
        loc, glob, sca = ego_device.build_observation(sim, rows, v8_feat)
        q_vals = net(loc, glob, sca).float()
        fallback = torch.arange(6, device=dev)[None, :] < 3
        resolved = torch.where(row_adv.any(dim=1, keepdim=True), row_adv, fallback)
        chosen = q_vals.masked_fill(~resolved, float("-inf")).argmax(dim=1)
        return (loc, glob, sca), q_vals, chosen

    out_loc = torch.empty((n_rows, local_ch, 31, 31), dtype=torch.uint8, device=dev)
    out_glob = torch.empty((n_rows, 4, 37, 37), dtype=torch.uint8, device=dev)
    out_sca = torch.empty((n_rows, n_scalars), dtype=torch.float32, device=dev)
    out_q = torch.empty((n_rows, 6), dtype=torch.float32, device=dev)
    out_act = torch.empty((n_rows,), dtype=torch.long, device=dev)
    fallback = torch.arange(6, device=dev)[None, :] < 3

    env_col = rows[:, 0]
    for start in range(0, sim.E, chunk_e):
        end = min(sim.E, start + chunk_e)
        sel = (env_col >= start) & (env_col < end)
        idx = sel.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        sub_sim = _slice_device_sim_view(sim, start, end)
        sub_rows = rows.index_select(0, idx).clone()
        sub_rows[:, 0] -= start
        sub_adv = advisory_all[rows.index_select(0, idx)[:, 0], sub_rows[:, 1]]
        sub_n = int(idx.numel())
        sub_v8: torch.Tensor | None = None
        if needs_v8:
            zero_q = torch.zeros((sub_n, 6), dtype=torch.float32, device=dev)
            sub_v8 = (
                v8_device.v8_decide(sub_sim, sub_rows, zero_q, sub_adv, v8_lambda, features=True)
                .feature_planes()
                .reshape(sub_n, 30)
            )
        c_loc, c_glob, c_sca = ego_device.build_observation(sub_sim, sub_rows, sub_v8)
        c_q = net(c_loc, c_glob, c_sca).float()
        c_res = torch.where(sub_adv.any(dim=1, keepdim=True), sub_adv, fallback)
        c_act = c_q.masked_fill(~c_res, float("-inf")).argmax(dim=1)

        out_loc.index_copy_(0, idx, c_loc)
        out_glob.index_copy_(0, idx, c_glob)
        out_sca.index_copy_(0, idx, c_sca)
        out_q.index_copy_(0, idx, c_q)
        out_act.index_copy_(0, idx, c_act)

    return (out_loc, out_glob, out_sca), out_q, out_act


class B8OpponentController:
    """Batched bare B8 (`ego2s-bv8`) opponent controller operating directly on `TrainSim`."""

    def __init__(
        self,
        device: torch.device,
        *,
        checkpoint_path: str | Path | None = None,
        b8_env_chunk: int = DEFAULT_B8_ENV_CHUNK,
    ) -> None:
        resolved_path, digest = resolve_b8_checkpoint_path(checkpoint_path)
        self.device = device
        self.checkpoint_path = resolved_path
        self.sha256 = digest
        self.net = load_ego2s_checkpoint(str(resolved_path), device=device)
        self.obs_spec = spec_for(self.net)
        self.b8_env_chunk = max(1, int(b8_env_chunk))
        if device.type == "cuda":
            from research.gpu_sim_spike import flood_cuda

            flood_cuda.available()
            if v8_indexed.enabled():
                v8_indexed._load()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "b8",
            "name": "b8-s3" if self.sha256 == B8_S3_SHA256 else self.checkpoint_path.name,
            "path": str(self.checkpoint_path),
            "sha256": self.sha256,
            "obs_spec": self.obs_spec,
            "mask_mode": "advisory_with_normal_fallback",
            "veto": None,
        }

    @torch.no_grad()
    def act_seats(self, train_sim: TrainSim, seat_mask: torch.Tensor) -> torch.Tensor:
        """Select actions `(E, S)` for active opponent seats (`seat_mask & alive`)."""
        sim = train_sim.sim
        active = seat_mask & sim.alive
        out_actions = torch.ones((sim.E, sim.S), dtype=torch.long, device=sim.device)
        rows = active.nonzero(as_tuple=False)
        if rows.shape[0] == 0:
            return out_actions
        _, _, chosen = compute_b8_observations_and_actions(
            sim,
            self.net,
            rows,
            b8_env_chunk=self.b8_env_chunk,
        )
        out_actions[rows[:, 0], rows[:, 1]] = chosen
        return out_actions


def build_learner_seat_mask(
    worlds: int,
    num_snakes: int,
    learner_seats: int,
    *,
    seat_assignment: str = "rotate",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build `(W, S)` boolean mask with exactly `learner_seats` True entries per world."""
    w = int(worlds)
    s = int(num_snakes)
    k = int(learner_seats)
    if w < 1:
        raise ValueError(f"worlds must be >= 1, got {worlds}")
    if s < 1:
        raise ValueError(f"num_snakes must be >= 1, got {num_snakes}")
    if k < 0 or k > s:
        raise ValueError(f"learner_seats must be in [0, {s}], got {learner_seats}")
    if seat_assignment not in SEAT_ASSIGNMENT_MODES:
        raise ValueError(
            f"seat_assignment must be one of {SEAT_ASSIGNMENT_MODES}, got {seat_assignment!r}"
        )

    dev = torch.device(device)
    if k == 0:
        return torch.zeros((w, s), dtype=torch.bool, device=dev)
    if k == s:
        return torch.ones((w, s), dtype=torch.bool, device=dev)

    slots = torch.arange(s, device=dev)[None, :]
    if seat_assignment == "prefix":
        return (slots < k).expand(w, s).clone()

    env_idx = torch.arange(w, device=dev)[:, None]
    rel_slot = (slots - env_idx) % s
    return rel_slot < k


@torch.no_grad()
def attribute_step_deaths_and_kills(
    sim: DeviceSim,
    pre_head0: torch.Tensor,
    info: StepInfo,
) -> StepCollisionAttribution:
    """Reconstruct exact killer->victim and mutual-tie attribution on worlds with deaths.

    Runs only on the subset of worlds where `info.died.any(dim=1)` is True, using the
    exact mechanics-v2 collision resolver order of `DeviceSim._resolve_collisions_reference`,
    and verifies bit-exact equality against `info.death_cause` and `info.kills`.
    """
    e_total, s_total = sim.E, sim.S
    dev = sim.device
    wall_full = torch.zeros((e_total, s_total), dtype=torch.bool, device=dev)
    self_full = torch.zeros((e_total, s_total), dtype=torch.bool, device=dev)
    pair_kills_full = torch.zeros((e_total, s_total, s_total), dtype=torch.long, device=dev)
    pair_mutual_full = torch.zeros((e_total, s_total, s_total), dtype=torch.bool, device=dev)

    death_envs = info.died.any(dim=1).nonzero(as_tuple=False).flatten()
    m = int(death_envs.numel())
    if m == 0:
        return StepCollisionAttribution(
            wall_deaths=wall_full,
            self_deaths=self_full,
            pair_kills=pair_kills_full,
            pair_mutual=pair_mutual_full,
        )

    pre_alive = info.acted.index_select(0, death_envs)
    head0 = pre_head0.index_select(0, death_envs)
    post_dir = sim.direction.index_select(0, death_envs)
    boost = sim._boosted_this_step.index_select(0, death_envs)
    head_ptr = sim.head_ptr.index_select(0, death_envs)
    seg_count = sim.seg_count.index_select(0, death_envs)
    bodies = sim.bodies.index_select(0, death_envs)
    lengths = sim.length.index_select(0, death_envs)

    cardinal = torch.tensor(_CARDINAL, dtype=torch.long, device=dev)
    delta = cardinal[post_dir]
    head1 = head0 + delta
    head2 = head1 + delta
    traversed = torch.stack((head1, head2), dim=2)
    traversed_valid = torch.stack((pre_alive, boost), dim=2)

    in_arena = ((~traversed_valid) | sim._in_arena(traversed)).all(dim=2)

    k = torch.arange(sim.cap, device=dev)[None, None, :]
    rings = (head_ptr[..., None] - k) % sim.cap
    body = bodies.gather(2, rings[..., None].expand(-1, -1, -1, 2))
    valid_body = pre_alive[..., None] & (k < seg_count[..., None])

    same = (traversed[:, :, :, None, None, :] == body[:, None, None, :, :, :]).all(dim=-1)
    same &= traversed_valid[:, :, :, None, None]
    own_offsets = k >= 3
    eye = torch.eye(s_total, dtype=torch.bool, device=dev)[None]
    own = eye[:, :, None, :, None]
    self_hit = (
        same & own & valid_body[:, None, None, :, :] & own_offsets[:, None, None, :, :]
    ).any(dim=(2, 3, 4))

    head_equal = (traversed[:, :, :, None, None, :] == traversed[:, None, None, :, :, :]).all(
        dim=-1
    )
    head_equal &= traversed_valid[:, :, :, None, None] & traversed_valid[:, None, None, :, :]
    head_equal = head_equal.any(dim=(2, 4))
    swap_a = (head0[:, :, None, None, None, :] == traversed[:, None, None, :, :, :]).all(dim=-1)
    swap_b = (traversed[:, :, :, None, None, :] == head0[:, None, None, :, None, :]).all(dim=-1)
    head_equal |= (swap_a & swap_b).any(dim=(2, 4))
    head_pair = head_equal & ~eye

    body_other = same & valid_body[:, None, None, :, :] & ~own
    body_pair = body_other.any(dim=(2, 4))

    death = torch.zeros((m, s_total), dtype=torch.long, device=dev)
    dead = torch.zeros((m, s_total), dtype=torch.bool, device=dev)
    kills = torch.zeros((m, s_total), dtype=torch.long, device=dev)
    wall_sub = torch.zeros((m, s_total), dtype=torch.bool, device=dev)
    self_sub = torch.zeros((m, s_total), dtype=torch.bool, device=dev)
    pair_kills_sub = torch.zeros((m, s_total, s_total), dtype=torch.long, device=dev)
    pair_mutual_sub = torch.zeros((m, s_total, s_total), dtype=torch.bool, device=dev)

    for i in range(s_total):
        wall_i = pre_alive[:, i] & ~in_arena[:, i] & ~dead[:, i]
        wall_sub[:, i] = wall_i
        death[:, i] = torch.where(wall_i, torch.full_like(death[:, i], DEATH_WALL), death[:, i])
        dead[:, i] |= wall_i

        self_i = pre_alive[:, i] & self_hit[:, i] & ~dead[:, i]
        self_sub[:, i] = self_i
        death[:, i] = torch.where(self_i, torch.full_like(death[:, i], DEATH_SELF), death[:, i])
        dead[:, i] |= self_i

        for j in range(s_total):
            if i == j:
                continue
            pair_i = pre_alive[:, i] & ~dead[:, i]
            head = pair_i & pre_alive[:, j] & ~dead[:, j] & head_pair[:, i, j]
            body_target_live = (
                torch.ones_like(pair_i) if sim.v2 else (pre_alive[:, j] & ~dead[:, j])
            )
            bodyhit = pair_i & body_target_live & ~head_pair[:, i, j] & body_pair[:, i, j]
            if sim.v2:
                iwins = head & (
                    lengths[:, i].float() >= float(HEADON_SIZE_RATIO) * lengths[:, j].float()
                )
                jwins = head & (
                    lengths[:, j].float() >= float(HEADON_SIZE_RATIO) * lengths[:, i].float()
                )
            else:
                iwins = torch.zeros_like(head)
                jwins = torch.zeros_like(head)
            mutual = head & ~iwins & ~jwins
            die_i = bodyhit | mutual | jwins
            die_j = mutual | iwins
            cause_i = torch.where(
                bodyhit,
                torch.full_like(death[:, i], DEATH_BODY),
                torch.full_like(death[:, i], DEATH_HEAD),
            )
            death[:, i] = torch.where(die_i, cause_i, death[:, i])
            death[:, j] = torch.where(die_j, torch.full_like(death[:, j], DEATH_HEAD), death[:, j])
            kills[:, i] += iwins.long()
            kills[:, j] += (jwins | bodyhit).long()
            pair_kills_sub[:, i, j] += iwins.long()
            pair_kills_sub[:, j, i] += (jwins | bodyhit).long()
            pair_mutual_sub[:, i, j] |= mutual
            pair_mutual_sub[:, j, i] |= mutual
            dead[:, i] |= die_i
            dead[:, j] |= die_j

    expected_cause = sim._last_death_cause.index_select(0, death_envs)
    expected_kills = sim._last_kills.index_select(0, death_envs)
    if not bool(torch.equal(death, expected_cause)) or not bool(torch.equal(kills, expected_kills)):
        raise RuntimeError(
            "Step collision attribution diverged from DeviceSim._last_death_cause / _last_kills"
        )

    wall_full.index_copy_(0, death_envs, wall_sub)
    self_full.index_copy_(0, death_envs, self_sub)
    pair_kills_full.index_copy_(0, death_envs, pair_kills_sub)
    pair_mutual_full.index_copy_(0, death_envs, pair_mutual_sub)
    return StepCollisionAttribution(
        wall_deaths=wall_full,
        self_deaths=self_full,
        pair_kills=pair_kills_full,
        pair_mutual=pair_mutual_full,
    )


class _SideMetricsAccumulator:
    """Device-accumulated post-burn-in evaluation metrics for one side (`learner` or `opponent`)."""

    def __init__(self, seats_per_world: int, worlds: int, device: torch.device) -> None:
        self.seats_per_world = int(seats_per_world)
        self.worlds = int(worlds)
        self.device = device
        self.ticks_recorded = 0

        self.acted_steps = torch.zeros((), dtype=torch.int64, device=device)
        self.boost_steps = torch.zeros((), dtype=torch.int64, device=device)
        self.sum_alive_length = torch.zeros((), dtype=torch.int64, device=device)
        self.lives_carried_in = torch.zeros((), dtype=torch.int64, device=device)
        self.lives_spawned = torch.zeros((), dtype=torch.int64, device=device)
        self.completed_lives = torch.zeros((), dtype=torch.int64, device=device)
        self.completed_born_in_window = torch.zeros((), dtype=torch.int64, device=device)
        self.food_eaten = torch.zeros((), dtype=torch.int64, device=device)
        self.ambient_food_eaten = torch.zeros((), dtype=torch.int64, device=device)
        self.corpse_food_eaten = torch.zeros((), dtype=torch.int64, device=device)

        self.wall_deaths = torch.zeros((), dtype=torch.int64, device=device)
        self.self_deaths = torch.zeros((), dtype=torch.int64, device=device)
        self.body_deaths = torch.zeros((), dtype=torch.int64, device=device)
        self.head_deaths = torch.zeros((), dtype=torch.int64, device=device)
        self.killed_by_other = torch.zeros((), dtype=torch.int64, device=device)
        self.killed_by_same = torch.zeros((), dtype=torch.int64, device=device)
        self.mutual_other = torch.zeros((), dtype=torch.int64, device=device)
        self.mutual_same = torch.zeros((), dtype=torch.int64, device=device)

        self.kills_vs_other = torch.zeros((), dtype=torch.int64, device=device)
        self.kills_same_side = torch.zeros((), dtype=torch.int64, device=device)

        self._death_lengths: list[torch.Tensor] = []

    @torch.no_grad()
    def record_window_open(self, side_mask: torch.Tensor, alive: torch.Tensor) -> None:
        self.lives_carried_in += (side_mask & alive).to(torch.int64).sum()

    @torch.no_grad()
    def record_respawns(self, side_mask: torch.Tensor, newly_spawned: torch.Tensor) -> None:
        self.lives_spawned += (side_mask & newly_spawned).to(torch.int64).sum()

    @torch.no_grad()
    def record_step(
        self,
        *,
        sim: DeviceSim,
        side_mask: torch.Tensor,
        other_mask: torch.Tensor,
        pre_length: torch.Tensor,
        post_length: torch.Tensor,
        boosted: torch.Tensor,
        born_in_window: torch.Tensor,
        info: StepInfo,
        attrib: StepCollisionAttribution,
    ) -> None:
        self.ticks_recorded += 1
        if self.seats_per_world == 0:
            return

        acted_side = side_mask & info.acted
        died_side = side_mask & info.died

        self.acted_steps += acted_side.to(torch.int64).sum()
        self.boost_steps += (acted_side & boosted).to(torch.int64).sum()
        self.sum_alive_length += torch.where(
            acted_side, pre_length.to(torch.int64), torch.zeros_like(pre_length, dtype=torch.int64)
        ).sum()

        self.completed_lives += died_side.to(torch.int64).sum()
        self.completed_born_in_window += (died_side & born_in_window).to(torch.int64).sum()
        self.food_eaten += (acted_side & sim._last_food_ate).to(torch.int64).sum()
        self.ambient_food_eaten += (acted_side & sim._last_ambient_food_ate).to(torch.int64).sum()
        self.corpse_food_eaten += (acted_side & sim._last_corpse_food_ate).to(torch.int64).sum()

        if bool(died_side.any()):
            self._death_lengths.append(post_length[died_side].detach().cpu())

        self.wall_deaths += (died_side & attrib.wall_deaths).to(torch.int64).sum()
        self.self_deaths += (died_side & attrib.self_deaths).to(torch.int64).sum()
        self.body_deaths += (
            (died_side & (sim._last_death_cause == DEATH_BODY)).to(torch.int64).sum()
        )
        self.head_deaths += (
            (died_side & (sim._last_death_cause == DEATH_HEAD)).to(torch.int64).sum()
        )

        # pair_kills[e, killer, victim]:
        killer_is_side = side_mask[:, :, None]
        killer_is_other = other_mask[:, :, None]
        victim_is_side = side_mask[:, None, :]
        victim_is_other = other_mask[:, None, :]

        self.kills_vs_other += (
            attrib.pair_kills * (killer_is_side & victim_is_other).to(torch.int64)
        ).sum()
        self.kills_same_side += (
            attrib.pair_kills * (killer_is_side & victim_is_side).to(torch.int64)
        ).sum()
        self.killed_by_other += (
            attrib.pair_kills * (killer_is_other & victim_is_side).to(torch.int64)
        ).sum()
        self.killed_by_same += (
            attrib.pair_kills * (killer_is_side & victim_is_side).to(torch.int64)
        ).sum()

        # pair_mutual[e, victim, partner] is True on at most one partner per victim per tick:
        self.mutual_other += (
            (attrib.pair_mutual & victim_is_side & killer_is_other).any(dim=1).to(torch.int64).sum()
        )
        self.mutual_same += (
            (attrib.pair_mutual & victim_is_side & killer_is_side).any(dim=1).to(torch.int64).sum()
        )

    @torch.no_grad()
    def finalize(
        self,
        *,
        side_mask: torch.Tensor,
        final_alive: torch.Tensor,
        final_length: torch.Tensor,
    ) -> dict[str, Any]:
        seat_ticks = int(self.worlds * self.seats_per_world * self.ticks_recorded)
        if self.seats_per_world == 0:
            return {
                "seats_per_world": 0,
                "completed_lives": 0,
                "completed_lives_born_in_window": 0,
                "censored_lives": 0,
                "lives_carried_into_window": 0,
                "lives_spawned_in_window": 0,
                "mean_length_at_death": None,
                "median_length_at_death": None,
                "p95_length_at_death": None,
                "mean_censored_length": None,
                "time_averaged_length": None,
                "seat_tick_averaged_length": None,
                "deaths_per_1k_agent_steps": None,
                "kills_per_1k_agent_steps": None,
                "kills_vs_other_side_per_1k_agent_steps": None,
                "kills_same_side_per_1k_agent_steps": None,
                "self_or_wall_deaths_per_1k_agent_steps": None,
                "food_per_1k_steps": None,
                "raw_counts": {
                    "seat_ticks": 0,
                    "acted_agent_steps": 0,
                    "boost_steps": 0,
                    "sum_alive_length": 0,
                    "completed_lives": 0,
                    "completed_lives_born_in_window": 0,
                    "censored_lives": 0,
                    "lives_carried_into_window": 0,
                    "lives_spawned_in_window": 0,
                    "deaths": 0,
                    "deaths_by_cause": {
                        "wall": 0,
                        "self": 0,
                        "self_or_wall": 0,
                        "killed_by_other_side": 0,
                        "killed_by_same_side": 0,
                        "mutual_head_on_other_side": 0,
                        "mutual_head_on_same_side": 0,
                        "mutual_head_on_total": 0,
                        "body_collision_total": 0,
                        "head_collision_total": 0,
                    },
                    "kills": 0,
                    "kills_by_target": {
                        "vs_other_side": 0,
                        "same_side": 0,
                    },
                    "food_eaten": 0,
                    "ambient_food_eaten": 0,
                    "corpse_food_eaten": 0,
                },
            }

        censored_mask = side_mask & final_alive
        censored_lives = int(censored_mask.to(torch.int64).sum().detach().cpu().item())
        censored_mean_len: float | None = None
        if censored_lives > 0:
            censored_vals = final_length[censored_mask].detach().cpu().numpy().astype(np.float64)
            censored_mean_len = round(float(np.mean(censored_vals)), 4)

        acted = int(self.acted_steps.detach().cpu().item())
        boost_n = int(self.boost_steps.detach().cpu().item())
        sum_len = int(self.sum_alive_length.detach().cpu().item())
        carried_in = int(self.lives_carried_in.detach().cpu().item())
        spawned = int(self.lives_spawned.detach().cpu().item())
        completed = int(self.completed_lives.detach().cpu().item())
        completed_born = int(self.completed_born_in_window.detach().cpu().item())
        food_total = int(self.food_eaten.detach().cpu().item())
        food_amb = int(self.ambient_food_eaten.detach().cpu().item())
        food_corpse = int(self.corpse_food_eaten.detach().cpu().item())

        wall_d = int(self.wall_deaths.detach().cpu().item())
        self_d = int(self.self_deaths.detach().cpu().item())
        body_d = int(self.body_deaths.detach().cpu().item())
        head_d = int(self.head_deaths.detach().cpu().item())
        killed_other = int(self.killed_by_other.detach().cpu().item())
        killed_same = int(self.killed_by_same.detach().cpu().item())
        mut_other = int(self.mutual_other.detach().cpu().item())
        mut_same = int(self.mutual_same.detach().cpu().item())
        mut_total = mut_other + mut_same
        self_or_wall = wall_d + self_d

        kills_other = int(self.kills_vs_other.detach().cpu().item())
        kills_same = int(self.kills_same_side.detach().cpu().item())
        kills_total = kills_other + kills_same

        if carried_in + spawned != completed + censored_lives:
            raise RuntimeError(
                f"Life conservation invariant failed: carried_in({carried_in}) + "
                f"spawned({spawned}) != completed({completed}) + censored({censored_lives})"
            )
        if self_or_wall + killed_other + killed_same + mut_total != completed:
            raise RuntimeError(
                f"Death attribution sum mismatch: self_or_wall({self_or_wall}) + "
                f"killed_other({killed_other}) + killed_same({killed_same}) + "
                f"mutual({mut_total}) != completed({completed})"
            )

        if self._death_lengths:
            death_arr = torch.cat(self._death_lengths, dim=0).numpy().astype(np.float64)
            mean_death_len: float | None = round(float(np.mean(death_arr)), 4)
            median_death_len: float | None = round(float(np.percentile(death_arr, 50)), 4)
            p95_death_len: float | None = round(float(np.percentile(death_arr, 95)), 4)
        else:
            mean_death_len = None
            median_death_len = None
            p95_death_len = None

        if acted > 0:
            denom_1k = float(acted) / 1000.0
            time_avg_len: float | None = round(float(sum_len) / float(acted), 4)
            deaths_1k: float | None = round(float(completed) / denom_1k, 4)
            kills_1k: float | None = round(float(kills_total) / denom_1k, 4)
            kills_other_1k: float | None = round(float(kills_other) / denom_1k, 4)
            kills_same_1k: float | None = round(float(kills_same) / denom_1k, 4)
            self_wall_1k: float | None = round(float(self_or_wall) / denom_1k, 4)
            food_1k: float | None = round(float(food_total) / denom_1k, 4)
        else:
            time_avg_len = None
            deaths_1k = None
            kills_1k = None
            kills_other_1k = None
            kills_same_1k = None
            self_wall_1k = None
            food_1k = None

        seat_tick_avg_len = round(float(sum_len) / float(seat_ticks), 4) if seat_ticks > 0 else None

        return {
            "seats_per_world": self.seats_per_world,
            "completed_lives": completed,
            "completed_lives_born_in_window": completed_born,
            "censored_lives": censored_lives,
            "lives_carried_into_window": carried_in,
            "lives_spawned_in_window": spawned,
            "mean_length_at_death": mean_death_len,
            "median_length_at_death": median_death_len,
            "p95_length_at_death": p95_death_len,
            "mean_censored_length": censored_mean_len,
            "time_averaged_length": time_avg_len,
            "seat_tick_averaged_length": seat_tick_avg_len,
            "deaths_per_1k_agent_steps": deaths_1k,
            "kills_per_1k_agent_steps": kills_1k,
            "kills_vs_other_side_per_1k_agent_steps": kills_other_1k,
            "kills_same_side_per_1k_agent_steps": kills_same_1k,
            "self_or_wall_deaths_per_1k_agent_steps": self_wall_1k,
            "food_per_1k_steps": food_1k,
            "raw_counts": {
                "seat_ticks": seat_ticks,
                "acted_agent_steps": acted,
                "boost_steps": boost_n,
                "sum_alive_length": sum_len,
                "completed_lives": completed,
                "completed_lives_born_in_window": completed_born,
                "censored_lives": censored_lives,
                "lives_carried_into_window": carried_in,
                "lives_spawned_in_window": spawned,
                "deaths": completed,
                "deaths_by_cause": {
                    "wall": wall_d,
                    "self": self_d,
                    "self_or_wall": self_or_wall,
                    "killed_by_other_side": killed_other,
                    "killed_by_same_side": killed_same,
                    "mutual_head_on_other_side": mut_other,
                    "mutual_head_on_same_side": mut_same,
                    "mutual_head_on_total": mut_total,
                    "body_collision_total": body_d,
                    "head_collision_total": head_d,
                },
                "kills": kills_total,
                "kills_by_target": {
                    "vs_other_side": kills_other,
                    "same_side": kills_same,
                },
                "food_eaten": food_total,
                "ambient_food_eaten": food_amb,
                "corpse_food_eaten": food_corpse,
            },
        }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_str = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _parse_opponent_spec(
    opponent: str,
    *,
    b8_checkpoint: str | Path | None = None,
) -> tuple[str, str | Path | None]:
    raw = str(opponent).strip()
    if raw in SCRIPTED_OPPONENT_NAMES:
        return raw, None
    if raw in ("b8", "b8-s3"):
        return "b8", b8_checkpoint
    if raw.startswith("b8:"):
        b8_sub = raw[len("b8:") :].strip()
        if not b8_sub:
            raise ValueError("Empty path in --opponent b8:<PATH>")
        return "b8", b8_sub
    if raw == "untrained":
        return "untrained", None
    if raw.startswith("ckpt:"):
        ckpt_sub = raw[len("ckpt:") :].strip()
        if not ckpt_sub:
            raise ValueError("Empty path in --opponent ckpt:<PATH>")
        return "ckpt", ckpt_sub
    raise ValueError(
        f"Unsupported --opponent {opponent!r}; expected one of "
        "random_safe, greedy_food, b8, b8-s3, b8:<PATH>, untrained, ckpt:<PATH>"
    )


@torch.no_grad()
def evaluate_policy(
    *,
    checkpoint: str | Path | None,
    opponent: str,
    learner_seats: int,
    worlds: int = DEFAULT_EVAL_WORLDS,
    ticks: int = DEFAULT_EVAL_TICKS,
    burnin: int = DEFAULT_EVAL_BURNIN,
    seed: int = DEFAULT_EVAL_SEED,
    device: str | torch.device = "cpu",
    out: str | Path | None = None,
    greedy: bool = False,
    opponent_greedy: bool = False,
    seat_assignment: str = "rotate",
    actor_chunk_envs: int | None = None,
    b8_env_chunk: int = DEFAULT_B8_ENV_CHUNK,
    b8_checkpoint: str | Path | None = None,
    max_wall_seconds: float | None = None,
    create_only_sidecars: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate a trained Arena PPO policy against a fixed opponent on `TrainSim`.

    Args:
        checkpoint: Path to learner Arena PPO checkpoint (required when `learner_seats > 0`
            or when `opponent == 'untrained'`).
        opponent: Opponent spec (`random_safe`, `greedy_food`, `b8`, `b8-s3`, `b8:PATH`,
            `untrained`, or `ckpt:PATH`).
        learner_seats: Number of learner seats `K` per world (`0 <= K <= S`). `K = S` runs
            pure self-play; `K = 0` runs opponent-only baseline.
        worlds: Parallel world count `W` (`>= 1`).
        ticks: Measured post-burn-in ticks `T` (`>= 1`).
        burnin: Uncounted burn-in ticks `B` (`>= 0`) executed with the same seated policies.
        seed: Deterministic seed governing world RNG and policy sampling.
        device: Execution device (`cpu`, `cuda`, or `mps`).
        out: Optional output JSON file path.
        greedy: Select argmax actions for the learner checkpoint instead of sampling.
        opponent_greedy: Select argmax actions for `untrained` / `ckpt:PATH` opponents.
        seat_assignment: `"rotate"` (balanced across slots `0..S-1`) or `"prefix"` (`0..K-1`).
        actor_chunk_envs: Optional world chunk size for policy forward passes.
        b8_env_chunk: World chunk size for B8 `ego2s-bv8` observation/Q evaluation.
        b8_checkpoint: Optional explicit path for `b8` checkpoint staging.
        max_wall_seconds: Optional hard wall-clock runaway guard.
        create_only_sidecars: Write `out` with `.sha256` and `.complete` sidecars.
        should_stop: Optional cooperative signal-stop predicate.

    Returns:
        JSON-serializable evaluation result dictionary.
    """
    if int(worlds) < 1:
        raise ValueError(f"worlds must be >= 1, got {worlds}")
    if int(ticks) < 1:
        raise ValueError(f"ticks must be >= 1, got {ticks}")
    if int(burnin) < 0:
        raise ValueError(f"burnin must be >= 0, got {burnin}")

    wall_t0 = time.perf_counter()
    mono_t0 = time.monotonic()
    resolved_dev = normalize_device(device)
    if resolved_dev.type == "cuda":
        ensure_cuda_initialized(resolved_dev)
        preload_native_sim_extensions(resolved_dev)
    elif resolved_dev.type == "cpu":
        configure_determinism(True)

    opp_kind, opp_arg = _parse_opponent_spec(opponent, b8_checkpoint=b8_checkpoint)

    learner_policy: LoadedArenaPolicy | None = None
    if checkpoint is not None:
        learner_policy = load_arena_ppo_policy(
            checkpoint, resolved_dev, actor_chunk_override=actor_chunk_envs
        )
    elif int(learner_seats) > 0:
        raise ValueError("--checkpoint is required when --learner-seats > 0")
    elif opp_kind == "untrained":
        raise ValueError("--checkpoint is required to match architecture for --opponent untrained")

    grid_h = learner_policy.grid_h if learner_policy is not None else DEFAULT_GRID_H
    grid_w = learner_policy.grid_w if learner_policy is not None else DEFAULT_GRID_W
    num_snakes = learner_policy.num_snakes if learner_policy is not None else DEFAULT_NUM_SNAKES
    max_capacity = (
        learner_policy.max_capacity if learner_policy is not None else DEFAULT_MAX_CAPACITY
    )

    k_seats = int(learner_seats)
    if k_seats < 0 or k_seats > num_snakes:
        raise ValueError(f"--learner-seats must be in [0, {num_snakes}], got {learner_seats}")
    opp_seats = num_snakes - k_seats

    # Construct opponent controller (only materialized if opp_seats > 0, though metadata is always
    # populated so K = S self-play rows still record opponent identity cleanly).
    opp_arena_policy: LoadedArenaPolicy | None = None
    opp_b8: B8OpponentController | None = None
    if opp_kind in SCRIPTED_OPPONENT_NAMES:
        opponent_info: dict[str, Any] = {
            "spec": opponent,
            "kind": opp_kind,
            "version": "scripted-anchor/v1",
        }
    elif opp_kind == "untrained":
        assert learner_policy is not None
        opp_arena_policy = build_untrained_arena_policy(
            net_arch=learner_policy.net_arch,
            net_size=learner_policy.net_size,
            device=resolved_dev,
            seed=int(seed),
            grid_h=grid_h,
            grid_w=grid_w,
            num_snakes=num_snakes,
            max_capacity=max_capacity,
            ring_chunk=learner_policy.ring_chunk,
            actor_chunk_envs=actor_chunk_envs or learner_policy.actor_chunk_envs,
            amp=learner_policy.amp,
        )
        opponent_info = {"spec": opponent, "kind": "untrained", **opp_arena_policy.metadata}
    elif opp_kind == "ckpt":
        assert opp_arg is not None
        opp_arena_policy = load_arena_ppo_policy(
            opp_arg, resolved_dev, actor_chunk_override=actor_chunk_envs
        )
        if (opp_arena_policy.grid_h, opp_arena_policy.grid_w) != (grid_h, grid_w) or (
            opp_arena_policy.num_snakes != num_snakes
        ):
            opp_shape = (
                opp_arena_policy.grid_h,
                opp_arena_policy.grid_w,
                opp_arena_policy.num_snakes,
            )
            raise ValueError(
                f"Opponent checkpoint world shape {opp_shape} differs from "
                f"evaluation world {(grid_h, grid_w, num_snakes)}"
            )
        opponent_info = {"spec": opponent, "kind": "ckpt", **opp_arena_policy.metadata}
    elif opp_kind == "b8":
        if opp_seats > 0 or opp_arg is not None:
            opp_b8 = B8OpponentController(
                resolved_dev,
                checkpoint_path=opp_arg,
                b8_env_chunk=b8_env_chunk,
            )
            opponent_info = {"spec": opponent, **opp_b8.metadata}
        else:
            b8_path, b8_sha = resolve_b8_checkpoint_path(opp_arg)
            opponent_info = {
                "spec": opponent,
                "kind": "b8",
                "name": "b8-s3" if b8_sha == B8_S3_SHA256 else b8_path.name,
                "path": str(b8_path),
                "sha256": b8_sha,
                "obs_spec": OBS_SPEC_BV8,
                "mask_mode": "advisory_with_normal_fallback",
                "veto": None,
            }
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled opponent kind {opp_kind!r}")

    # Seed RNG streams deterministically AFTER model construction so weight init never shifts
    # policy action sampling for `(checkpoint, opponent, seed)`.
    int_seed = int(seed)
    random.seed(int_seed)
    np.random.seed(int_seed & 0xFFFF_FFFF)
    torch.manual_seed(int_seed)
    if resolved_dev.type == "cuda":
        torch.cuda.manual_seed_all(int_seed)

    train_sim = TrainSim.create(
        int(worlds),
        device=resolved_dev,
        seed=int_seed,
        grid_h=grid_h,
        grid_w=grid_w,
        num_snakes=num_snakes,
        max_capacity=max_capacity,
    )
    learner_mask = build_learner_seat_mask(
        int(worlds),
        num_snakes,
        k_seats,
        seat_assignment=seat_assignment,
        device=resolved_dev,
    )
    opponent_mask = ~learner_mask

    can_share_world_obs = (
        k_seats > 0
        and opp_seats > 0
        and learner_policy is not None
        and opp_arena_policy is not None
        and not isinstance(learner_policy.model, ArenaNet)
        and not isinstance(opp_arena_policy.model, ArenaNet)
        and learner_policy.ring_chunk == opp_arena_policy.ring_chunk
    )

    def _select_world_actions() -> torch.Tensor:
        legal_mask = train_sim.legal_action_mask()
        if can_share_world_obs:
            assert learner_policy is not None and opp_arena_policy is not None
            chunk = max(
                1,
                min(
                    train_sim.E,
                    int(
                        actor_chunk_envs
                        or min(learner_policy.actor_chunk_envs, opp_arena_policy.actor_chunk_envs)
                    ),
                ),
            )
            l_act = torch.empty((train_sim.E, train_sim.S), dtype=torch.long, device=resolved_dev)
            o_act = torch.empty((train_sim.E, train_sim.S), dtype=torch.long, device=resolved_dev)
            for start in range(0, train_sim.E, chunk):
                end = min(train_sim.E, start + chunk)
                sl = slice(start, end)
                shared_obs = build_shared_world_obs_gpu(
                    train_sim, env_slice=sl, ring_chunk=learner_policy.ring_chunk
                )
                l_out = learner_policy.model.act(
                    shared_obs, legal_mask=legal_mask[sl], greedy=greedy
                )
                o_out = opp_arena_policy.model.act(
                    shared_obs, legal_mask=legal_mask[sl], greedy=opponent_greedy
                )
                l_act[sl] = l_out.actions.reshape(end - start, train_sim.S)
                o_act[sl] = o_out.actions.reshape(end - start, train_sim.S)
            return torch.where(learner_mask, l_act, o_act)

        if k_seats > 0:
            assert learner_policy is not None
            learner_actions = act_arena_policy_chunked(
                train_sim,
                learner_policy,
                legal_mask,
                greedy=greedy,
                actor_chunk_envs=actor_chunk_envs,
            )
        else:
            learner_actions = torch.ones(
                (train_sim.E, train_sim.S), dtype=torch.long, device=resolved_dev
            )

        if opp_seats == 0:
            return learner_actions

        if opp_kind == "greedy_food":
            opp_actions = train_sim.scripted_greedy_food_actions()
        elif opp_kind == "random_safe":
            opp_actions = scripted_random_safe_actions(train_sim)
        elif opp_kind in ("untrained", "ckpt"):
            assert opp_arena_policy is not None
            opp_actions = act_arena_policy_chunked(
                train_sim,
                opp_arena_policy,
                legal_mask,
                greedy=opponent_greedy,
                actor_chunk_envs=actor_chunk_envs,
            )
        elif opp_kind == "b8":
            assert opp_b8 is not None
            opp_actions = opp_b8.act_seats(train_sim, opponent_mask)
        else:  # pragma: no cover
            raise AssertionError(f"Unhandled opponent kind {opp_kind!r}")

        if k_seats == 0:
            return opp_actions
        return torch.where(learner_mask, learner_actions, opp_actions)

    # 1. Burn-in phase (`0 .. B-1`), excluded from all metrics
    synchronize_device(resolved_dev)
    burnin_t0 = time.perf_counter()
    print(
        f"ARENA_PPO_EVAL_PHASE START phase=burnin ticks={int(burnin)} worlds={int(worlds)} "
        f"ts={time.time():.3f}",
        file=sys.stderr,
        flush=True,
    )
    for b_tick in range(int(burnin)):
        if should_stop is not None and should_stop():
            raise StopRequestedError(f"Stop requested during evaluation burn-in at tick {b_tick}")
        if max_wall_seconds is not None and (time.monotonic() - mono_t0) >= float(max_wall_seconds):
            raise TimeoutError(
                f"Evaluation exceeded max_wall_seconds={max_wall_seconds} during burn-in"
            )
        train_sim.prepare()
        actions = _select_world_actions()
        train_sim.step(actions)
    train_sim.assert_no_capacity_overflow()
    synchronize_device(resolved_dev)
    burnin_sec = time.perf_counter() - burnin_t0
    print(
        f"ARENA_PPO_EVAL_PHASE END phase=burnin elapsed_sec={burnin_sec:.3f} ts={time.time():.3f}",
        file=sys.stderr,
        flush=True,
    )

    # 2. Measured evaluation window (`B .. B+T-1`)
    learner_acc = _SideMetricsAccumulator(k_seats, int(worlds), resolved_dev)
    opponent_acc = _SideMetricsAccumulator(opp_seats, int(worlds), resolved_dev)
    born_in_window = torch.zeros((int(worlds), num_snakes), dtype=torch.bool, device=resolved_dev)

    eval_t0 = time.perf_counter()
    print(
        f"ARENA_PPO_EVAL_PHASE START phase=eval ticks={int(ticks)} worlds={int(worlds)} "
        f"ts={time.time():.3f}",
        file=sys.stderr,
        flush=True,
    )
    prev_post_alive = train_sim.sim.alive.clone()
    for m_tick in range(int(ticks)):
        if should_stop is not None and should_stop():
            raise StopRequestedError(
                f"Stop requested during evaluation window at tick {m_tick}/{ticks}"
            )
        if max_wall_seconds is not None and (time.monotonic() - mono_t0) >= float(max_wall_seconds):
            raise TimeoutError(
                f"Evaluation exceeded max_wall_seconds={max_wall_seconds} at tick {m_tick}/{ticks}"
            )

        train_sim.prepare()
        curr_alive = train_sim.sim.alive
        if m_tick == 0:
            learner_acc.record_window_open(learner_mask, curr_alive)
            opponent_acc.record_window_open(opponent_mask, curr_alive)
            if int(burnin) == 0:
                born_in_window.copy_(curr_alive)
            else:
                born_in_window.zero_()
        else:
            newly_spawned = (~prev_post_alive) & curr_alive
            born_in_window |= newly_spawned
            learner_acc.record_respawns(learner_mask, newly_spawned)
            opponent_acc.record_respawns(opponent_mask, newly_spawned)

        pre_length = train_sim.sim.length.clone()
        pre_head0 = train_sim.sim.heads().clone()

        actions = _select_world_actions()
        info = train_sim.step(actions)
        attrib = attribute_step_deaths_and_kills(train_sim.sim, pre_head0, info)

        post_length = train_sim.sim.length
        boosted = train_sim.sim._boosted_this_step

        learner_acc.record_step(
            sim=train_sim.sim,
            side_mask=learner_mask,
            other_mask=opponent_mask,
            pre_length=pre_length,
            post_length=post_length,
            boosted=boosted,
            born_in_window=born_in_window,
            info=info,
            attrib=attrib,
        )
        opponent_acc.record_step(
            sim=train_sim.sim,
            side_mask=opponent_mask,
            other_mask=learner_mask,
            pre_length=pre_length,
            post_length=post_length,
            boosted=boosted,
            born_in_window=born_in_window,
            info=info,
            attrib=attrib,
        )
        born_in_window &= info.survived
        prev_post_alive = info.survived.clone()

    train_sim.assert_no_capacity_overflow()
    synchronize_device(resolved_dev)
    eval_sec = time.perf_counter() - eval_t0
    total_wall_sec = time.perf_counter() - wall_t0
    print(
        f"ARENA_PPO_EVAL_PHASE END phase=eval elapsed_sec={eval_sec:.3f} "
        f"total_sec={total_wall_sec:.3f} ts={time.time():.3f}",
        file=sys.stderr,
        flush=True,
    )

    learner_summary = learner_acc.finalize(
        side_mask=learner_mask,
        final_alive=train_sim.sim.alive,
        final_length=train_sim.sim.length,
    )
    opponent_summary = opponent_acc.finalize(
        side_mask=opponent_mask,
        final_alive=train_sim.sim.alive,
        final_length=train_sim.sim.length,
    )

    l_raw = learner_summary["raw_counts"]
    o_raw = opponent_summary["raw_counts"]
    l_to_o = int(l_raw["kills_by_target"]["vs_other_side"])
    o_to_l = int(o_raw["kills_by_target"]["vs_other_side"])
    l_to_l = int(l_raw["kills_by_target"]["same_side"])
    o_to_o = int(o_raw["kills_by_target"]["same_side"])
    l_sw = int(l_raw["deaths_by_cause"]["self_or_wall"])
    o_sw = int(o_raw["deaths_by_cause"]["self_or_wall"])
    l_mut = int(l_raw["deaths_by_cause"]["mutual_head_on_total"])
    o_mut = int(o_raw["deaths_by_cause"]["mutual_head_on_total"])

    if l_to_o != int(o_raw["deaths_by_cause"]["killed_by_other_side"]):
        raise RuntimeError(
            "Cross-side kill mismatch: learner->opponent != opponent killed_by_other"
        )
    if o_to_l != int(l_raw["deaths_by_cause"]["killed_by_other_side"]):
        raise RuntimeError("Cross-side kill mismatch: opponent->learner != learner killed_by_other")

    eval_world_steps = int(worlds) * int(ticks)
    eval_agent_steps = eval_world_steps * num_snakes
    total_world_steps = int(worlds) * (int(burnin) + int(ticks))
    total_agent_steps = total_world_steps * num_snakes

    result: dict[str, Any] = {
        "schema": EVAL_SCHEMA,
        "checkpoint_info": learner_policy.metadata if learner_policy is not None else None,
        "opponent_info": opponent_info,
        "seating": {
            "num_snakes": num_snakes,
            "learner_seats": k_seats,
            "opponent_seats": opp_seats,
            "seat_assignment": seat_assignment,
            "learner_greedy": bool(greedy),
            "opponent_greedy": bool(opponent_greedy),
        },
        "config": {
            "worlds": int(worlds),
            "ticks": int(ticks),
            "burnin": int(burnin),
            "seed": int_seed,
            "device": str(resolved_dev),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_capacity": max_capacity,
            "actor_chunk_envs": (
                int(actor_chunk_envs)
                if actor_chunk_envs is not None
                else (learner_policy.actor_chunk_envs if learner_policy is not None else None)
            ),
            "b8_env_chunk": int(b8_env_chunk) if opp_kind == "b8" else None,
        },
        "timing": {
            "wall_seconds": round(float(total_wall_sec), 4),
            "burnin_seconds": round(float(burnin_sec), 4),
            "eval_seconds": round(float(eval_sec), 4),
            "eval_world_steps": eval_world_steps,
            "eval_agent_steps": eval_agent_steps,
            "eval_world_steps_per_sec": round(float(eval_world_steps) / max(1e-9, eval_sec), 2),
            "eval_agent_steps_per_sec": round(float(eval_agent_steps) / max(1e-9, eval_sec), 2),
            "total_world_steps": total_world_steps,
            "total_agent_steps": total_agent_steps,
            "total_agent_steps_per_sec": round(
                float(total_agent_steps) / max(1e-9, total_wall_sec), 2
            ),
        },
        "kill_matrix": {
            "kills_credited_total": l_to_o + o_to_l + l_to_l + o_to_o,
            "learner_to_opponent": l_to_o,
            "opponent_to_learner": o_to_l,
            "learner_to_learner": l_to_l,
            "opponent_to_opponent": o_to_o,
            "same_side_total": l_to_l + o_to_o,
            "self_or_wall_total": l_sw + o_sw,
            "learner_self_or_wall": l_sw,
            "opponent_self_or_wall": o_sw,
            "mutual_head_on_deaths_total": l_mut + o_mut,
            "learner_mutual_head_on_deaths": l_mut,
            "opponent_mutual_head_on_deaths": o_mut,
            "deaths_total": int(l_raw["deaths"]) + int(o_raw["deaths"]),
        },
        "learner": learner_summary,
        "opponent": opponent_summary,
    }

    if out is not None:
        out_path = Path(out).resolve()
        if create_only_sidecars:
            write_create_only_json(out_path, result)
        else:
            _atomic_write_json(out_path, result)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `python -m research.arena_ppo.evaluate`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained Arena PPO checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        help=(
            "Opponent specification: random_safe, greedy_food, b8 (or b8-s3 / b8:PATH), "
            "untrained, or ckpt:PATH."
        ),
    )
    parser.add_argument(
        "--learner-seats",
        type=int,
        default=1,
        help="Number of learner seats K per world in [0, S] (K=S self-play, K=0 opponent-only).",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=DEFAULT_EVAL_WORLDS,
        help=f"Number of parallel worlds W (default: {DEFAULT_EVAL_WORLDS}).",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_EVAL_TICKS,
        help=f"Post-burn-in evaluation ticks T (default: {DEFAULT_EVAL_TICKS}).",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=DEFAULT_EVAL_BURNIN,
        help=f"Burn-in ticks B excluded from metrics (default: {DEFAULT_EVAL_BURNIN}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=f"Deterministic seed S (default: {DEFAULT_EVAL_SEED}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device: cpu or cuda (also accepts cuda:N / mps).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for result JSON.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy argmax actions for the learner checkpoint (default: sample from policy).",
    )
    parser.add_argument(
        "--opponent-greedy",
        action="store_true",
        help="Use greedy argmax actions for untrained / ckpt:PATH opponents.",
    )
    parser.add_argument(
        "--seat-assignment",
        choices=SEAT_ASSIGNMENT_MODES,
        default="rotate",
        help="Learner seat placement across worlds: rotate (balanced slots) or prefix (0..K-1).",
    )
    parser.add_argument(
        "--actor-chunk-envs",
        type=int,
        default=None,
        help="Optional world batch chunk size for Arena PPO forward passes.",
    )
    parser.add_argument(
        "--b8-env-chunk",
        type=int,
        default=DEFAULT_B8_ENV_CHUNK,
        help=f"World chunk size for B8 feature extraction (default: {DEFAULT_B8_ENV_CHUNK}).",
    )
    parser.add_argument(
        "--b8-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit B8 checkpoint file path (for staged remote inputs).",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=(
            float(os.environ["ARENA_PPO_MAX_WALL_SECONDS"])
            if os.environ.get("ARENA_PPO_MAX_WALL_SECONDS")
            else None
        ),
        help="Optional wall-clock timeout guard in seconds.",
    )
    parser.add_argument(
        "--sidecars",
        action="store_true",
        help="Write create-only .sha256 and .complete sidecars alongside --out.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m research.arena_ppo.evaluate`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    with SignalStopFlag() as stop_flag:
        res = evaluate_policy(
            checkpoint=args.checkpoint,
            opponent=args.opponent,
            learner_seats=args.learner_seats,
            worlds=args.worlds,
            ticks=args.ticks,
            burnin=args.burnin,
            seed=args.seed,
            device=args.device,
            out=args.out,
            greedy=bool(args.greedy),
            opponent_greedy=bool(args.opponent_greedy),
            seat_assignment=args.seat_assignment,
            actor_chunk_envs=args.actor_chunk_envs,
            b8_env_chunk=args.b8_env_chunk,
            b8_checkpoint=args.b8_checkpoint,
            max_wall_seconds=args.max_wall_seconds,
            create_only_sidecars=bool(args.sidecars),
            should_stop=lambda: stop_flag.stopped,
        )
    if args.out is None:
        print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
