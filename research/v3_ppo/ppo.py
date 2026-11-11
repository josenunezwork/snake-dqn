"""Game V3.1 PPO self-play trainer, LIGHT + sharded FULL checkpoint/resume, and job runner."""

from __future__ import annotations

import argparse
import gc
import inspect
import io
import json
import os
import random
import time
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from research.arena_ppo.runtime import (
    RotatingCreateOnlyJSONLWriter,
    SignalStopFlag,
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    publish_hardlinked_artifacts,
    remove_completed_artifact,
    sha256_file,
    synchronize_device,
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.game_v3.config import GameV3Config
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs
from research.game_v3.sim import GameV3Sim
from research.v3_ppo.buffer import Uint8RolloutBuffer
from research.v3_ppo.config import (
    DEFAULT_CHECKPOINT_SHARD_BYTES,
    V3_PPO_NAMESPACE,
    V3PPOConfig,
    estimate_training_memory,
)
from research.v3_ppo.eval import (
    compute_step_reward,
    derive_world_seeds,
    evaluate_policy,
    scripted_opponent_actions,
)
from research.v3_ppo.net import (
    EgoMultiScaleNet,
    EgoMultiScaleNetConfig,
    symlog_two_hot_loss,
)

V3_PPO_CHECKPOINT_SCHEMA = "v3-ppo-checkpoint/v1"
V3_PPO_LIGHT_CHECKPOINT_SCHEMA = "v3-ppo-light-checkpoint/v2"
V3_PPO_FULL_INDEX_SCHEMA = "v3-ppo-full-checkpoint-index/v2"
V3_PPO_SIM_SHARD_SCHEMA = "v3-ppo-sim-shard/v2"
V3_PPO_TRAIN_RESULT_SCHEMA = "v3-ppo-train-result/v1"
UNLIMITED_HOURS_TOTAL_UPDATES = 1_000_000_000

_INIT_ANCHOR_LIGHT_NAME = "light_u0000000_init.pt"

_RESUME_MUTABLE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "total_updates",
        "light_checkpoint_interval_sec",
        "full_checkpoint_interval_sec",
        "checkpoint_shard_bytes",
        "checkpoint_every_updates",
        "full_checkpoint_every_updates",
        "keep_full_checkpoints",
        "keep_light_checkpoints",
        "max_in_session_restarts",
        "max_update_seconds",
        "eval_every_minutes",
        "eval_every_updates",
        "eval_envs",
        "eval_burnin_ticks",
        "eval_ticks",
        "eval_learner_slots",
        "eval_seed",
        "eval_greedy",
        "actor_chunk_learners",
        "compile_model",
        # Legacy r1 keys accepted via V3PPOConfig.from_dict
        "checkpoint_interval_sec",
        "output_publish_interval_sec",
        "keep_periodic_checkpoints",
    }
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


_SIM_ZLIB_TOTAL_THRESHOLD_BYTES = 0
_SIM_ZLIB_TENSOR_MIN_BYTES = 4 * 1024
_SIM_ZLIB_CHUNK_BYTES = 128 * 1024 * 1024


def pack_sim_state_dict(
    sim_sd: dict[str, Any],
    *,
    total_threshold_bytes: int = _SIM_ZLIB_TOTAL_THRESHOLD_BYTES,
    tensor_min_bytes: int = _SIM_ZLIB_TENSOR_MIN_BYTES,
) -> dict[str, Any]:
    """Losslessly compress simulator tensors (`>= 4 KiB`) into binary ``uint8`` zlib chunks.

    Stores compressed chunks as ``torch.uint8`` 1D tensors rather than Python ``bytes`` so
    ``torch.save`` (Pickle Protocol 2) writes raw binary storage without UTF-8 expansion.
    """
    tensors = sim_sd.get("tensors")
    if not isinstance(tensors, dict):
        return sim_sd
    total_bytes = sum(
        t.numel() * t.element_size() for t in tensors.values() if isinstance(t, torch.Tensor)
    )
    if total_bytes <= int(total_threshold_bytes):
        return sim_sd

    raw_tensors: dict[str, torch.Tensor] = {}
    zlib_tensors: dict[str, dict[str, Any]] = {}
    for name, val in tensors.items():
        if not isinstance(val, torch.Tensor):
            continue
        nbytes = val.numel() * val.element_size()
        if nbytes == 0 or nbytes < int(tensor_min_bytes):
            raw_tensors[name] = val.detach().cpu().contiguous()
            continue
        cpu_contig = val.detach().cpu().contiguous()
        byte_tensor = cpu_contig.view(torch.uint8).reshape(-1)
        mv = memoryview(byte_tensor.numpy())
        chunks: list[torch.Tensor] = []
        for off in range(0, len(mv), _SIM_ZLIB_CHUNK_BYTES):
            comp_bytes = zlib.compress(mv[off : off + _SIM_ZLIB_CHUNK_BYTES], level=1)
            chunks.append(torch.frombuffer(bytearray(comp_bytes), dtype=torch.uint8))
        zlib_tensors[name] = {
            "shape": list(cpu_contig.shape),
            "dtype": str(cpu_contig.dtype).removeprefix("torch."),
            "nbytes": int(nbytes),
            "chunks": chunks,
        }
    if not zlib_tensors:
        return sim_sd
    out = dict(sim_sd)
    out["tensors"] = raw_tensors
    out["_zlib_tensors"] = zlib_tensors
    return out


def unpack_sim_state_dict(sim_sd: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct exact ``GameV3Sim.state_dict()`` tensors from :func:`pack_sim_state_dict`."""
    zlib_tensors = sim_sd.get("_zlib_tensors")
    if not isinstance(zlib_tensors, dict) or not zlib_tensors:
        return sim_sd
    merged_tensors = dict(sim_sd.get("tensors") or {})
    for name, meta in zlib_tensors.items():
        dtype = getattr(torch, str(meta["dtype"]))
        shape = tuple(int(d) for d in meta["shape"])
        nbytes = int(meta["nbytes"])
        dst_u8 = torch.empty((nbytes,), dtype=torch.uint8)
        dst_mv = memoryview(dst_u8.numpy())
        offset = 0
        for comp_chunk in meta["chunks"]:
            if isinstance(comp_chunk, torch.Tensor):
                comp_buf: bytes | memoryview = memoryview(comp_chunk.detach().cpu().numpy())
            else:
                comp_buf = comp_chunk
            raw_chunk = zlib.decompress(comp_buf)
            end = offset + len(raw_chunk)
            dst_mv[offset:end] = raw_chunk
            offset = end
        if offset != nbytes:
            raise ValueError(
                f"Corrupt compressed sim tensor {name!r}: expected {nbytes} bytes, got {offset}"
            )
        merged_tensors[name] = dst_u8.view(dtype).reshape(shape)
    out = dict(sim_sd)
    out.pop("_zlib_tensors", None)
    out["tensors"] = merged_tensors
    return out


def slice_sim_state_dict(
    sim_sd: dict[str, Any],
    world_start: int,
    world_end: int,
) -> dict[str, Any]:
    """Extract whole-world slice ``[world_start:world_end]`` from raw ``GameV3Sim.state_dict()``."""
    w0 = int(world_start)
    w1 = int(world_end)
    total_e = int(sim_sd["num_envs"])
    if not (0 <= w0 < w1 <= total_e):
        raise ValueError(f"Invalid world slice [{w0}, {w1}) for num_envs={total_e}")
    tensors = sim_sd["tensors"]
    sliced_tensors = {name: val[w0:w1].detach().cpu().contiguous() for name, val in tensors.items()}
    return {
        "config": sim_sd["config"],
        "num_envs": w1 - w0,
        "tick": int(sim_sd["tick"]),
        "subtick_offset": int(sim_sd.get("subtick_offset", 0)),
        "tensors": sliced_tensors,
    }


def concat_sim_state_shards(
    shard_payloads: Sequence[dict[str, Any]],
    *,
    expected_num_envs: int,
) -> tuple[dict[str, Any], int]:
    """Verify contiguous world coverage across ``shard_payloads`` and concatenate unpacked state."""
    if not shard_payloads:
        raise ValueError("Empty shard_payloads passed to concat_sim_state_shards")

    ordered = sorted(shard_payloads, key=lambda s: int(s["world_start"]))
    cursor = 0
    unpacked_list: list[dict[str, Any]] = []
    ref_cfg: dict[str, Any] | None = None
    ref_tick: int | None = None
    ref_subtick: int | None = None
    ref_streak: int | None = None

    for item in ordered:
        if item.get("schema") != V3_PPO_SIM_SHARD_SCHEMA:
            raise ValueError(f"Invalid sim shard schema: {item.get('schema')!r}")
        if item.get("namespace") != V3_PPO_NAMESPACE:
            raise ValueError(f"Foreign sim shard namespace: {item.get('namespace')!r}")
        w0 = int(item["world_start"])
        w1 = int(item["world_end"])
        if w0 != cursor or w1 <= w0:
            raise ValueError(
                f"Non-contiguous sim shards: expected world_start={cursor}, got [{w0}, {w1})"
            )
        if int(item.get("total_worlds", -1)) != int(expected_num_envs):
            raise ValueError(
                f"Sim shard total_worlds {item.get('total_worlds')} != {expected_num_envs}"
            )
        unpacked = unpack_sim_state_dict(item["sim_state_dict"])
        if int(unpacked["num_envs"]) != (w1 - w0):
            raise ValueError(
                f"Shard [{w0}, {w1}) internal num_envs={unpacked['num_envs']} != {w1 - w0}"
            )
        if ref_cfg is None:
            ref_cfg = unpacked["config"]
            ref_tick = int(unpacked["tick"])
            ref_subtick = int(unpacked.get("subtick_offset", 0))
            ref_streak = int(item.get("sim_clean_step_streak", 0))
        else:
            if (
                unpacked["config"] != ref_cfg
                or int(unpacked["tick"]) != ref_tick
                or int(unpacked.get("subtick_offset", 0)) != ref_subtick
            ):
                raise ValueError("Inconsistent simulator metadata across FULL checkpoint shards")
        unpacked_list.append(unpacked)
        cursor = w1

    if cursor != int(expected_num_envs):
        raise ValueError(
            f"Incomplete FULL checkpoint shards: covered {cursor}/{expected_num_envs} worlds"
        )

    assert ref_cfg is not None and ref_tick is not None and ref_subtick is not None
    tensor_names = tuple(unpacked_list[0]["tensors"].keys())
    if len(unpacked_list) == 1:
        merged_tensors = dict(unpacked_list[0]["tensors"])
    else:
        merged_tensors = {
            name: torch.cat([u["tensors"][name] for u in unpacked_list], dim=0)
            for name in tensor_names
        }
    merged_sd = {
        "config": ref_cfg,
        "num_envs": int(expected_num_envs),
        "tick": ref_tick,
        "subtick_offset": ref_subtick,
        "tensors": merged_tensors,
    }
    return merged_sd, int(ref_streak or 0)


def verify_light_checkpoint(
    light_path: Path,
    *,
    require_markers: bool = True,
) -> dict[str, Any]:
    """Validate a LIGHT checkpoint file and return summary metadata."""
    p = Path(light_path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing LIGHT checkpoint file: {p}")
    digest = verify_completed_artifact(p) if require_markers else sha256_file(p)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("schema") != V3_PPO_LIGHT_CHECKPOINT_SCHEMA:
        raise ValueError(f"Invalid LIGHT checkpoint schema at {p}")
    if ckpt.get("namespace") != V3_PPO_NAMESPACE:
        raise ValueError(f"Foreign LIGHT checkpoint namespace at {p}: {ckpt.get('namespace')!r}")
    counters = ckpt.get("counters")
    if not isinstance(counters, dict) or "updates_completed" not in counters:
        raise ValueError(f"Corrupt counters in LIGHT checkpoint at {p}")
    return {
        "kind": "light",
        "path": p,
        "files": [p],
        "sha256": digest,
        "size_bytes": int(p.stat().st_size),
        "updates_completed": int(counters["updates_completed"]),
        "agent_steps_collected": int(counters.get("agent_steps_collected", 0)),
        "config": ckpt.get("config"),
    }


def verify_full_checkpoint_set(
    index_path: Path,
    *,
    require_markers: bool = True,
) -> dict[str, Any]:
    """Validate a FULL checkpoint index, its LIGHT file, and every simulator shard."""
    idx_p = Path(index_path)
    if not idx_p.is_file():
        raise FileNotFoundError(f"Missing FULL checkpoint index: {idx_p}")
    idx_sha = verify_completed_artifact(idx_p) if require_markers else sha256_file(idx_p)
    idx_obj = json.loads(idx_p.read_text(encoding="utf-8"))
    if not isinstance(idx_obj, dict) or idx_obj.get("schema") != V3_PPO_FULL_INDEX_SCHEMA:
        raise ValueError(f"Invalid FULL checkpoint index schema at {idx_p}")
    if idx_obj.get("namespace") != V3_PPO_NAMESPACE:
        raise ValueError(f"Foreign FULL checkpoint index namespace at {idx_p}")

    ckpt_dir = idx_p.parent
    light_rec = idx_obj.get("light_checkpoint")
    if not isinstance(light_rec, dict) or not light_rec.get("filename"):
        raise ValueError(f"Missing light_checkpoint entry in {idx_p}")
    light_p = ckpt_dir / str(light_rec["filename"])
    if not light_p.is_file():
        raise FileNotFoundError(f"Missing FULL checkpoint light file: {light_p}")
    light_sha = verify_completed_artifact(light_p) if require_markers else sha256_file(light_p)
    if light_sha != light_rec.get("sha256"):
        raise ValueError(
            f"FULL checkpoint light SHA mismatch on {light_p.name}: "
            f"{light_sha} != {light_rec.get('sha256')}"
        )
    if int(light_p.stat().st_size) != int(light_rec.get("size_bytes", -1)):
        raise ValueError(f"FULL checkpoint light size mismatch on {light_p.name}")
    light_meta = verify_light_checkpoint(light_p, require_markers=require_markers)

    shards_rec = idx_obj.get("shards")
    if not isinstance(shards_rec, list) or not shards_rec:
        raise ValueError(f"FULL checkpoint index {idx_p} lists zero shards")

    expected_envs = int(idx_obj["num_envs"])
    cursor = 0
    shard_paths: list[Path] = []
    total_shard_bytes = 0
    for s_rec in shards_rec:
        if not isinstance(s_rec, dict) or not s_rec.get("filename"):
            raise ValueError(f"Malformed shard entry in {idx_p}")
        w0 = int(s_rec["world_start"])
        w1 = int(s_rec["world_end"])
        if w0 != cursor or w1 <= w0:
            raise ValueError(
                f"Non-contiguous shard range [{w0}, {w1}) at cursor={cursor} in {idx_p}"
            )
        sp = ckpt_dir / str(s_rec["filename"])
        if not sp.is_file():
            raise FileNotFoundError(f"Missing FULL checkpoint shard: {sp}")
        s_sha = verify_completed_artifact(sp) if require_markers else sha256_file(sp)
        if s_sha != s_rec.get("sha256"):
            raise ValueError(
                f"FULL checkpoint shard SHA mismatch on {sp.name}: "
                f"{s_sha} != {s_rec.get('sha256')}"
            )
        s_size = int(sp.stat().st_size)
        if s_size != int(s_rec.get("size_bytes", -1)):
            raise ValueError(
                f"FULL checkpoint shard size mismatch on {sp.name}: "
                f"{s_size} != {s_rec.get('size_bytes')}"
            )
        shard_paths.append(sp)
        total_shard_bytes += s_size
        cursor = w1

    if cursor != expected_envs:
        raise ValueError(f"FULL checkpoint index {idx_p} covers {cursor}/{expected_envs} worlds")

    updates_completed = int(idx_obj["updates_completed"])
    if int(light_meta["updates_completed"]) != updates_completed:
        raise ValueError(
            f"FULL index updates_completed ({updates_completed}) != "
            f"LIGHT updates_completed ({light_meta['updates_completed']})"
        )

    all_files = [idx_p, light_p, *shard_paths]
    return {
        "kind": "full",
        "path": idx_p,
        "light_path": light_p,
        "shard_paths": shard_paths,
        "files": all_files,
        "sha256": idx_sha,
        "size_bytes": int(idx_p.stat().st_size) + int(light_meta["size_bytes"]) + total_shard_bytes,
        "num_shards": len(shard_paths),
        "updates_completed": updates_completed,
        "agent_steps_collected": int(idx_obj.get("agent_steps_collected", 0)),
        "config": light_meta["config"],
    }


def resolve_resume_checkpoint(
    checkpoints_dir: Path,
    *,
    require_markers: bool = True,
    include_init_anchor: bool = False,
) -> dict[str, Any] | None:
    """Resolve resume checkpoint in coordinator order: newest complete FULL, else newest LIGHT."""
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.is_dir():
        return None

    # 1. Check all `full_*.index.json` candidates; pick newest complete FULL set.
    valid_full: list[tuple[int, str, dict[str, Any]]] = []
    for idx_cand in sorted(ckpt_dir.glob("full_*.index.json")):
        if require_markers and not (
            Path(f"{idx_cand}.complete").is_file() and Path(f"{idx_cand}.sha256").is_file()
        ):
            print(
                f"V3_PPO_SKIP_INCOMPLETE_FULL index={idx_cand.name} reason=missing_index_markers",
                flush=True,
            )
            continue
        try:
            meta = verify_full_checkpoint_set(idx_cand, require_markers=require_markers)
            valid_full.append((int(meta["updates_completed"]), idx_cand.name, meta))
        except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            print(
                f"V3_PPO_SKIP_INCOMPLETE_FULL index={idx_cand.name} "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

    if valid_full:
        valid_full.sort(key=lambda t: (t[0], t[1]))
        return valid_full[-1][2]

    # 2. Else fall back to newest verified LIGHT checkpoint (`light_*.pt`).
    valid_light: list[tuple[int, int, str, dict[str, Any]]] = []
    for light_cand in sorted(ckpt_dir.glob("light_*.pt")):
        if not include_init_anchor and light_cand.name == _INIT_ANCHOR_LIGHT_NAME:
            continue
        if require_markers and not (
            Path(f"{light_cand}.complete").is_file() and Path(f"{light_cand}.sha256").is_file()
        ):
            continue
        try:
            meta_l = verify_light_checkpoint(light_cand, require_markers=require_markers)
            is_non_init = 0 if light_cand.name == _INIT_ANCHOR_LIGHT_NAME else 1
            valid_light.append(
                (int(meta_l["updates_completed"]), is_non_init, light_cand.name, meta_l)
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            print(
                f"V3_PPO_SKIP_INVALID_LIGHT file={light_cand.name} "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

    if valid_light:
        valid_light.sort(key=lambda t: (t[0], t[1], t[2]))
        return valid_light[-1][3]

    # 3. Legacy r1 single-file fallback (`final.pt` / `ckpt_u*.pt`) if present.
    legacy_cands = [ckpt_dir / "final.pt", *reversed(sorted(ckpt_dir.glob("ckpt_u*.pt")))]
    for leg in legacy_cands:
        if not leg.is_file():
            continue
        if require_markers and not (
            Path(f"{leg}.complete").is_file() and Path(f"{leg}.sha256").is_file()
        ):
            continue
        try:
            sha = verify_completed_artifact(leg) if require_markers else sha256_file(leg)
            obj = torch.load(leg, map_location="cpu", weights_only=False)
            if (
                isinstance(obj, dict)
                and obj.get("schema") == V3_PPO_CHECKPOINT_SCHEMA
                and obj.get("namespace") == V3_PPO_NAMESPACE
            ):
                cnt = obj.get("counters") or {}
                return {
                    "kind": "legacy_full",
                    "path": leg,
                    "files": [leg],
                    "sha256": sha,
                    "size_bytes": int(leg.stat().st_size),
                    "updates_completed": int(cnt.get("updates_completed", 0)),
                    "agent_steps_collected": int(cnt.get("agent_steps_collected", 0)),
                    "config": obj.get("config"),
                }
        except Exception:
            continue

    return None


def find_newest_verified_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return path to newest verified FULL `.index.json` or fallback LIGHT `.pt` checkpoint."""
    resolved = resolve_resume_checkpoint(checkpoints_dir, require_markers=True)
    return Path(resolved["path"]) if resolved is not None else None


def _is_unshared_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_nlink <= 1
    except OSError:
        return False


def prune_old_checkpoints(
    checkpoints_dir: Path,
    *,
    keep_full: int = 2,
    keep_light: int = 4,
) -> list[str]:
    """Retain newest ``keep_full`` FULL sets and ``keep_light`` LIGHT files (unshared only).

    Never unlinks files with ``st_nlink > 1`` (already hard-linked into ``output/``) and never
    unlinks a ``light_*.pt`` still referenced by a retained ``full_*.index.json``.
    """
    if not checkpoints_dir.is_dir():
        return []

    removed: list[str] = []

    # 1. Inspect all FULL index files in `checkpoints_dir`.
    full_entries: list[tuple[int, str, Path, list[Path], str | None, bool]] = []
    for idx_path in sorted(checkpoints_dir.glob("full_*.index.json")):
        try:
            idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
            u_num = int(idx_obj.get("updates_completed", -1))
            light_name = (
                str(idx_obj["light_checkpoint"]["filename"])
                if isinstance(idx_obj.get("light_checkpoint"), dict)
                and idx_obj["light_checkpoint"].get("filename")
                else None
            )
            s_files = [
                checkpoints_dir / str(s["filename"])
                for s in (idx_obj.get("shards") or [])
                if isinstance(s, dict) and s.get("filename")
            ]
        except Exception:
            continue
        all_set_files = [idx_path, *s_files]
        can_prune = all((not p.exists()) or _is_unshared_file(p) for p in all_set_files)
        full_entries.append((u_num, idx_path.name, idx_path, s_files, light_name, can_prune))

    full_entries.sort(key=lambda t: (t[0], t[1]))
    prunable_full = [entry for entry in full_entries if entry[5]]
    to_remove_full_names: set[str] = set()
    if len(prunable_full) > int(keep_full):
        for _, idx_name, idx_path, s_files, _, _ in prunable_full[
            : len(prunable_full) - int(keep_full)
        ]:
            to_remove_full_names.add(idx_name)
            for sp in s_files:
                if _is_unshared_file(sp):
                    remove_completed_artifact(sp)
                    removed.append(sp.name)
            if _is_unshared_file(idx_path):
                remove_completed_artifact(idx_path.with_suffix(".meta.json"))
                remove_completed_artifact(idx_path)
                removed.append(idx_path.name)

    protected_lights: set[str] = {
        entry[4]
        for entry in full_entries
        if entry[1] not in to_remove_full_names and entry[4] is not None
    }

    # 2. Prune unshared standalone LIGHT files beyond `keep_light` newest.
    unshared_lights: list[Path] = []
    for lp in sorted(checkpoints_dir.glob("light_*.pt")):
        if lp.name == _INIT_ANCHOR_LIGHT_NAME or lp.name in protected_lights:
            continue
        if _is_unshared_file(lp):
            unshared_lights.append(lp)

    if len(unshared_lights) > int(keep_light):
        for old_lp in unshared_lights[: len(unshared_lights) - int(keep_light)]:
            if _is_unshared_file(old_lp):
                remove_completed_artifact(old_lp.with_suffix(".meta.json"))
                remove_completed_artifact(old_lp)
                removed.append(old_lp.name)

    return removed


def prune_old_periodic_checkpoints(checkpoints_dir: Path, *, keep: int) -> list[str]:
    """Backward-compatible wrapper around :func:`prune_old_checkpoints`."""
    return prune_old_checkpoints(checkpoints_dir, keep_full=int(keep), keep_light=max(4, int(keep)))


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
        )
        self.raw_model = EgoMultiScaleNet(net_cfg).to(self.device)
        if cfg.compile_model and hasattr(torch, "compile"):
            self.model: EgoMultiScaleNet = torch.compile(self.raw_model)  # type: ignore[assignment]
        else:
            self.model = self.raw_model

        self.optimizer = _build_adamw_optimizer(
            self.raw_model.parameters(),
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
        )
        self._bootstrap_obs = allocate_obs_batch(self.sim, self.learner_index)
        self._bootstrap_obs_valid = False

        # Persistent counters and episode return accumulators (serialized in checkpoints)
        self.updates_completed: int = 0
        self.world_steps_collected: int = 0
        self.agent_steps_collected: int = 0
        self.learner_steps_collected: int = 0
        self.episodes_completed: int = 0
        self.obs_builds_executed: int = 0
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

        for epoch in range(int(self.cfg.ppo_epochs)):
            if should_stop is not None and should_stop():
                raise StopRequestedError("Cooperative stop requested during PPO learner epochs")

            perm_cpu = torch.randperm(n_valid, generator=self.perm_gen)
            shuffled_valid = valid_idx.index_select(0, perm_cpu.to(device=self.device))

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
                )

                self.optimizer.zero_grad(set_to_none=True)
                eval_out = self.model.evaluate_actions(
                    mb.raster_s1,
                    mb.raster_s4,
                    mb.raster_s16,
                    mb.scalars,
                    mb.actions,
                    action_mask=mb.action_mask,
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
                        self.raw_model.parameters(),
                        self.cfg.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(),
                        self.cfg.max_grad_norm,
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

        # Single device-to-host synchronization for all scalar statistics of this update.
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

    def run_evaluation(self) -> dict[str, Any]:
        """Run built-in deterministic fixed-seed M-of-N evaluation against scripted opponents."""
        metrics = evaluate_policy(self.raw_model, self.cfg, device=self.device)
        self.last_eval_metrics = metrics
        return metrics

    def light_checkpoint_dict(self) -> dict[str, Any]:
        """Build LIGHT CPU-resident checkpoint dictionary (network, opt, counters, RNG; no sim)."""
        return {
            "schema": V3_PPO_LIGHT_CHECKPOINT_SCHEMA,
            "namespace": V3_PPO_NAMESPACE,
            "checkpoint_kind": "light",
            "config": self.cfg.to_dict(),
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

    def save_light_checkpoint(self, path: Path) -> str:
        """Atomically write a LIGHT checkpoint (`no sim state`) with `.sha256` & `.complete`."""
        payload = self.light_checkpoint_dict()
        return write_create_only_artifact(Path(path), lambda tmp: torch.save(payload, tmp))

    def save_full_checkpoint(
        self,
        checkpoints_dir: Path,
        *,
        tag: str | None = None,
        max_shard_bytes: int | None = None,
        light_path: Path | None = None,
    ) -> Path:
        """Atomically write a sharded FULL checkpoint set (`light` + whole-world shards + index).

        Every simulator shard file stays strictly ``<= max_shard_bytes`` (`256 MiB` default),
        each with its own `.sha256` and `.complete` sidecar. The index file
        `full_<tag>.index.json` is written strictly LAST and defines completeness of the set.
        """
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
            light_sha = self.save_light_checkpoint(eff_light)
        else:
            light_sha = verify_completed_artifact(eff_light)

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
            "updates_completed": int(self.updates_completed),
            "world_steps_collected": int(self.world_steps_collected),
            "agent_steps_collected": int(self.agent_steps_collected),
            "num_envs": total_worlds,
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

    def checkpoint_dict(self, *, compress_large_grids: bool = True) -> dict[str, Any]:
        """Build single-dict CPU snapshot (retained for direct unit-test inspection)."""
        raw_sim_sd = self.sim.state_dict()
        sim_sd = pack_sim_state_dict(raw_sim_sd) if compress_large_grids else raw_sim_sd
        base = self.light_checkpoint_dict()
        base["schema"] = V3_PPO_CHECKPOINT_SCHEMA
        base["checkpoint_kind"] = "legacy_full"
        base["sim_state_dict"] = sim_sd
        base["sim_clean_step_streak"] = int(self.sim._clean_step_streak)
        return base

    def save_checkpoint(self, path: Path) -> str:
        """Write a checkpoint: FULL `.index.json` if `path` ends with `.json`, else `.pt`."""
        target = Path(path)
        if target.name.endswith(".index.json"):
            tag = target.name.removeprefix("full_").removesuffix(".index.json")
            idx_p = self.save_full_checkpoint(target.parent, tag=tag)
            return sha256_file(idx_p)
        if target.name.startswith("light_"):
            return self.save_light_checkpoint(target)
        payload = self.checkpoint_dict()
        return write_create_only_artifact(target, lambda tmp: torch.save(payload, tmp))

    def _validate_and_apply_light_dict(
        self,
        ckpt: dict[str, Any],
        *,
        restore_rng: bool,
    ) -> None:
        ckpt_cfg = V3PPOConfig.from_dict(ckpt["config"])
        curr_dict = self.cfg.to_dict()
        ckpt_dict = ckpt_cfg.to_dict()
        for key in set(curr_dict) | set(ckpt_dict):
            if key in _RESUME_MUTABLE_CONFIG_KEYS or key == "schema":
                continue
            if curr_dict.get(key) != ckpt_dict.get(key):
                raise ValueError(
                    f"Checkpoint config mismatch on {key!r}: "
                    f"ckpt={ckpt_dict.get(key)!r} vs current={curr_dict.get(key)!r}"
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

    def load_checkpoint(
        self,
        path: Path,
        *,
        verify_markers: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore trainer from a FULL `.index.json`, LIGHT `.pt`, or legacy r1 `.pt`."""
        ckpt_path = Path(path)

        # Case 1: Sharded FULL checkpoint set (`full_*.index.json`)
        if ckpt_path.name.endswith(".index.json"):
            full_meta = verify_full_checkpoint_set(ckpt_path, require_markers=verify_markers)
            light_ckpt = torch.load(full_meta["light_path"], map_location="cpu", weights_only=False)
            self._validate_and_apply_light_dict(light_ckpt, restore_rng=restore_rng)

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
            return {
                "checkpoint_kind": "full",
                "num_shards": int(full_meta["num_shards"]),
                "worlds_reinitialized": False,
                "updates_completed": int(self.updates_completed),
            }

        # Case 2: LIGHT checkpoint or legacy r1 `.pt`
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
            self._validate_and_apply_light_dict(ckpt, restore_rng=restore_rng)
            # Re-initialise fresh deterministic worlds derived from `(seed, updates_completed)`
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
                f"V3_PPO_LIGHT_RESUME_WARNING checkpoint={ckpt_path.name} "
                f"updates_completed={self.updates_completed} worlds_reinitialized=True "
                "message='Resumed from LIGHT checkpoint (no sim state); worlds were freshly "
                "initialised and episodic return/length statistics need a warm-up.'",
                flush=True,
            )
            phase_stamp(
                "light_checkpoint_resume",
                "WARN_WORLDS_REINITIALIZED",
                checkpoint=ckpt_path.name,
                updates_completed=self.updates_completed,
            )
            return {
                "checkpoint_kind": "light",
                "num_shards": 0,
                "worlds_reinitialized": True,
                "updates_completed": int(self.updates_completed),
            }

        if schema == V3_PPO_CHECKPOINT_SCHEMA:
            self._validate_and_apply_light_dict(ckpt, restore_rng=restore_rng)
            self.sim.load_state_dict(unpack_sim_state_dict(ckpt["sim_state_dict"]))
            self.sim._clean_step_streak = int(ckpt.get("sim_clean_step_streak", 0))
            self.episode_return_acc.copy_(
                ckpt["episode_return_acc"].to(device=self.device, dtype=torch.float32)
            )
            self._bootstrap_obs_valid = False
            return {
                "checkpoint_kind": "legacy_full",
                "num_shards": 1,
                "worlds_reinitialized": False,
                "updates_completed": int(self.updates_completed),
            }

        raise ValueError(f"Invalid checkpoint schema at {ckpt_path}: {schema!r}")


def verify_staged_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    verify_markers: bool = True,
) -> dict[str, Any]:
    """Validate FULL `.index.json`, LIGHT `.pt`, or legacy `.pt` by restoring a trainer."""
    ckpt_path = Path(path)
    if ckpt_path.name.endswith(".index.json"):
        full_meta = verify_full_checkpoint_set(ckpt_path, require_markers=verify_markers)
        cfg = V3PPOConfig.from_dict(full_meta["config"])
        trainer = V3PPOTrainer(cfg, device=device)
        trainer.load_checkpoint(ckpt_path, verify_markers=verify_markers, restore_rng=False)
        return {
            "schema": V3_PPO_FULL_INDEX_SCHEMA,
            "namespace": V3_PPO_NAMESPACE,
            "checkpoint_kind": "full",
            "num_shards": int(full_meta["num_shards"]),
            "sha256": str(full_meta["sha256"]),
            "updates_completed": int(trainer.updates_completed),
            "agent_steps_collected": int(trainer.agent_steps_collected),
        }

    if verify_markers:
        verify_completed_artifact(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("schema") not in {
        V3_PPO_LIGHT_CHECKPOINT_SCHEMA,
        V3_PPO_CHECKPOINT_SCHEMA,
    }:
        raise ValueError(f"Invalid v3_ppo checkpoint at {ckpt_path}")
    cfg = V3PPOConfig.from_dict(raw["config"])
    trainer = V3PPOTrainer(cfg, device=device)
    load_info = trainer.load_checkpoint(ckpt_path, verify_markers=verify_markers, restore_rng=False)
    return {
        "schema": str(raw["schema"]),
        "namespace": str(raw["namespace"]),
        "checkpoint_kind": str(load_info["checkpoint_kind"]),
        "num_shards": int(load_info["num_shards"]),
        "sha256": sha256_file(ckpt_path),
        "updates_completed": int(trainer.updates_completed),
        "agent_steps_collected": int(trainer.agent_steps_collected),
    }


def publish_checkpoint_to_output(
    work_ckpt_path: Path,
    out_checkpoints_dir: Path,
    *,
    reason: str,
    counters: dict[str, Any],
) -> list[str]:
    """Hard-link a verified LIGHT `.pt` or FULL `.index.json` set into ``out_checkpoints_dir``.

    For FULL sets, links `light_*.pt` and all `sim_shard_*.pt` BEFORE linking `full_*.index.json`
    so the index becomes `.complete` strictly last.
    """
    out_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    src = Path(work_ckpt_path)
    published: list[str] = []

    if src.name.endswith(".index.json"):
        full_meta = verify_full_checkpoint_set(src, require_markers=True)
        # Publish light + all shards first, then index strictly last!
        ordered_sources = [full_meta["light_path"], *full_meta["shard_paths"], src]
        for f_src in ordered_sources:
            dst = out_checkpoints_dir / f_src.name
            if not dst.exists():
                publish_hardlinked_artifacts(f_src, [dst], staging_dir=f_src.parent)
                published.append(dst.name)
        meta_path = (out_checkpoints_dir / src.name).with_suffix(".meta.json")
        if not meta_path.exists():
            write_create_only_json(
                meta_path,
                {
                    "schema": "v3-ppo-checkpoint-meta/v2",
                    "namespace": V3_PPO_NAMESPACE,
                    "checkpoint_kind": "full",
                    "reason": reason,
                    "checkpoint_file": src.name,
                    "sha256": str(full_meta["sha256"]),
                    "size_bytes": int(full_meta["size_bytes"]),
                    "num_shards": int(full_meta["num_shards"]),
                    **counters,
                },
            )
        return published

    verify_completed_artifact(src)
    dst = out_checkpoints_dir / src.name
    ckpt_sha = (
        publish_hardlinked_artifacts(src, [dst], staging_dir=src.parent)
        if not dst.exists()
        else sha256_file(dst)
    )
    if dst.name not in published:
        published.append(dst.name)
    meta_path = dst.with_suffix(".meta.json")
    if not meta_path.exists():
        write_create_only_json(
            meta_path,
            {
                "schema": "v3-ppo-checkpoint-meta/v2",
                "namespace": V3_PPO_NAMESPACE,
                "checkpoint_kind": "light",
                "reason": reason,
                "checkpoint_file": dst.name,
                "sha256": ckpt_sha,
                "size_bytes": int(dst.stat().st_size),
                **counters,
            },
        )
    return published


def write_latest_checkpoint_manifest(
    out_checkpoints_dir: Path,
    *,
    reason: str,
    counters: dict[str, Any],
) -> Path:
    """Write create-only ``checkpoints/latest_manifest.json`` summarizing exported checkpoints."""
    out_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_checkpoints_dir / "latest_manifest.json"
    if manifest_path.exists():
        verify_completed_artifact(manifest_path)
        return manifest_path

    best = resolve_resume_checkpoint(out_checkpoints_dir, require_markers=True)
    all_files = sorted(
        p.name
        for p in out_checkpoints_dir.iterdir()
        if p.is_file()
        and not p.name.endswith((".sha256", ".complete", ".meta.json"))
        and p.name != "latest_manifest.json"
    )
    payload: dict[str, Any] = {
        "schema": "v3-ppo-latest-checkpoint/v2",
        "namespace": V3_PPO_NAMESPACE,
        "reason": reason,
        "primary_checkpoint": best["path"].name if best is not None else "",
        "primary_kind": best["kind"] if best is not None else "none",
        "sha256": best["sha256"] if best is not None else "",
        "size_bytes": int(best["size_bytes"]) if best is not None else 0,
        "published_files": all_files,
        **counters,
    }
    write_create_only_json(manifest_path, payload)
    return manifest_path


def _publish_checkpoint_manifest(
    work_ckpt_path: Path,
    out_checkpoints_dir: Path,
    *,
    export_names: list[str] | None = None,
    reason: str,
    counters: dict[str, Any],
    write_latest_manifest: bool = False,
) -> list[str]:
    """Compatibility helper publishing ``work_ckpt_path`` (and optional aliases) to ``out``."""
    published = publish_checkpoint_to_output(
        work_ckpt_path,
        out_checkpoints_dir,
        reason=reason,
        counters=counters,
    )
    if export_names and not work_ckpt_path.name.endswith(".index.json"):
        extra_targets = [
            out_checkpoints_dir / name
            for name in dict.fromkeys(export_names)
            if name != work_ckpt_path.name and not (out_checkpoints_dir / name).exists()
        ]
        if extra_targets:
            publish_hardlinked_artifacts(
                work_ckpt_path, extra_targets, staging_dir=work_ckpt_path.parent
            )
            published.extend(t.name for t in extra_targets)
    if write_latest_manifest:
        write_latest_checkpoint_manifest(
            out_checkpoints_dir,
            reason=reason,
            counters=counters,
        )
    return published


def _validate_staged_ledger_resume(
    *,
    input_dir: Path | None,
    ledger_path: Path | None,
) -> dict[str, Any] | None:
    """Validate optional staged continuation checkpoint (FULL set or LIGHT) in ``input_dir``."""
    if ledger_path is None or not Path(ledger_path).is_file():
        return None
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    if ledger.get("namespace") not in (None, V3_PPO_NAMESPACE):
        raise ValueError(
            f"Foreign ledger namespace {ledger.get('namespace')!r} != {V3_PPO_NAMESPACE!r}"
        )
    resume_entry = ledger.get("resume_checkpoint")
    if not isinstance(resume_entry, dict):
        return None
    if input_dir is None:
        raise ValueError("ledger specifies resume_checkpoint, but input_dir is None")

    in_dir = Path(input_dir)
    primary_rel = str(resume_entry["relpath"])
    staged_primary = in_dir / primary_rel
    if not staged_primary.is_file():
        raise FileNotFoundError(f"Missing staged checkpoint at {staged_primary}")
    actual_sha = sha256_file(staged_primary)
    if actual_sha != resume_entry.get("sha256"):
        raise ValueError(
            f"Staged checkpoint SHA-256 mismatch: {actual_sha} != {resume_entry.get('sha256')}"
        )

    # Verify companion files if `companion_files` are listed (for sharded FULL sets)
    for comp in resume_entry.get("companion_files") or []:
        comp_path = in_dir / str(comp["relpath"])
        if not comp_path.is_file():
            raise FileNotFoundError(f"Missing staged FULL companion file: {comp_path}")
        c_sha = sha256_file(comp_path)
        if c_sha != comp.get("sha256"):
            raise ValueError(
                f"Staged companion SHA-256 mismatch on {comp_path.name}: "
                f"{c_sha} != {comp.get('sha256')}"
            )

    if staged_primary.name.endswith(".index.json"):
        meta = verify_full_checkpoint_set(staged_primary, require_markers=False)
    elif (
        torch.load(staged_primary, map_location="cpu", weights_only=False).get("schema")
        == V3_PPO_LIGHT_CHECKPOINT_SCHEMA
    ):
        meta = verify_light_checkpoint(staged_primary, require_markers=False)
    else:
        obj = torch.load(staged_primary, map_location="cpu", weights_only=False)
        cnt = obj.get("counters") or {}
        meta = {
            "kind": "legacy_full",
            "path": staged_primary,
            "files": [staged_primary],
            "sha256": actual_sha,
            "size_bytes": int(staged_primary.stat().st_size),
            "updates_completed": int(cnt.get("updates_completed", 0)),
        }
    meta["source"] = "staged_ledger"
    meta["require_markers"] = False
    meta["relpath"] = primary_rel
    return meta


def _select_best_resume_candidate(
    *,
    work_ckpts: Path,
    out_ckpts: Path,
    staged_meta: dict[str, Any] | None,
    include_init_anchor: bool = False,
) -> dict[str, Any] | None:
    """Pick best verified checkpoint across `work_ckpts`, `out_ckpts`, and `staged_meta`.

    Applies the coordinator resume rule globally across available directories:
    newest complete FULL set (`kind in {'full', 'legacy_full'}`), else newest LIGHT.
    """
    candidates: list[dict[str, Any]] = []
    for src_name, d_path, inc_init in (
        ("work_checkpoints", work_ckpts, include_init_anchor),
        ("output_checkpoints", out_ckpts, False),
    ):
        found = resolve_resume_checkpoint(
            d_path,
            require_markers=True,
            include_init_anchor=inc_init,
        )
        if found is not None:
            cand = dict(found)
            cand["source"] = src_name
            cand["require_markers"] = True
            cand["relpath"] = Path(found["path"]).name
            candidates.append(cand)

    if staged_meta is not None:
        candidates.append(staged_meta)

    if not candidates:
        return None

    full_cands = [c for c in candidates if c["kind"] in {"full", "legacy_full"}]
    if full_cands:
        full_cands.sort(key=lambda c: (int(c["updates_completed"]), str(c["relpath"])))
        return full_cands[-1]

    light_cands = [
        c
        for c in candidates
        if c["kind"] == "light" and str(c["relpath"]) != _INIT_ANCHOR_LIGHT_NAME
    ]
    if light_cands:
        light_cands.sort(key=lambda c: (int(c["updates_completed"]), str(c["relpath"])))
        return light_cands[-1]

    # Only initial u0000000 anchor exists (e.g. crash/stop during update 1)
    candidates.sort(key=lambda c: int(c["updates_completed"]))
    return candidates[-1]


def _format_progress_line(
    step_metrics: dict[str, Any],
    *,
    total_updates: int,
    wall_elapsed: float,
    eval_metrics: dict[str, Any] | None = None,
) -> str:
    line = (
        f"V3_PPO_PROGRESS "
        f"update={step_metrics['update']}/{total_updates} "
        f"agent_steps={step_metrics['total_agent_steps']} "
        f"sps={step_metrics['agent_steps_per_sec']:.1f} "
        f"ep_ret={step_metrics['mean_ep_return']:.4f} "
        f"p_loss={step_metrics['policy_loss']:.4f} "
        f"v_loss={step_metrics['value_loss']:.4f} "
        f"ent={step_metrics['entropy']:.4f} "
        f"kl={step_metrics['approx_kl']:.5f} "
        f"wall_s={wall_elapsed:.2f}"
    )
    if eval_metrics is not None:
        comb = eval_metrics["combined"]
        line += (
            f" eval_tavg_len={comb['time_averaged_length']:.2f}"
            f" eval_dlen_mean={comb['mean_death_length']:.2f}"
            f" eval_dlen_p95={comb['p95_death_length']:.2f}"
            f" eval_deaths={comb['total_deaths']}"
            f" eval_deaths_1k={comb['deaths_per_1k_agent_steps']:.2f}"
            f" eval_mass_1k={comb['mass_gained_per_1k_agent_steps']:.2f}"
            f" eval_kills_1k={comb['kills_per_1k_agent_steps']:.2f}"
        )
    return line


def run_training_job(
    *,
    cfg: V3PPOConfig,
    output_dir: Path,
    work_dir: Path | None = None,
    device: str | torch.device = "cpu",
    input_dir: Path | None = None,
    ledger_path: Path | None = None,
    max_wall_seconds: float | None = None,
    step_hook: Callable[[V3PPOTrainer, int], None] | None = None,
) -> dict[str, Any]:
    """Execute training loop with wall-clock eval, LIGHT/FULL checkpoints, and crash recovery.

    Guarantees:
      - LIGHT checkpoints saved every ``light_checkpoint_interval_sec`` (`30` min default) and
        on stop; sharded FULL sets saved every ``full_checkpoint_interval_sec`` (`2` h default)
        and on a clean stop (`completed` or `wall_clock_guard`).
      - Cooperative stop mid-update discards the half-updated trainer state rather than saving
        corrupted/mid-rollout weights, publishing the newest clean update-boundary checkpoint.
      - Transient runtime errors inside a session recover from the newest verified checkpoint
        in ``work/``, ``output/``, or ``ledger`` up to ``cfg.max_in_session_restarts`` times.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wrk = Path(work_dir).resolve() if work_dir is not None else (out.parent / f".{out.name}_work")
    wrk.mkdir(parents=True, exist_ok=True)
    work_ckpts = wrk / "checkpoints"
    work_ckpts.mkdir(parents=True, exist_ok=True)
    out_ckpts = out / "checkpoints"
    out_ckpts.mkdir(parents=True, exist_ok=True)

    # Idempotent create-only `config.json` (safe on in-session or directory re-entry)
    cfg_json_path = out / "config.json"
    if cfg_json_path.exists():
        verify_completed_artifact(cfg_json_path)
        existing_cfg = V3PPOConfig.from_dict(json.loads(cfg_json_path.read_text(encoding="utf-8")))
        ex_dict = existing_cfg.to_dict()
        cur_dict = cfg.to_dict()
        for k in set(ex_dict) | set(cur_dict):
            if k in _RESUME_MUTABLE_CONFIG_KEYS or k == "schema":
                continue
            if ex_dict.get(k) != cur_dict.get(k):
                raise ValueError(
                    f"Existing config.json mismatch on immutable key {k!r}: "
                    f"{ex_dict.get(k)!r} != {cur_dict.get(k)!r}"
                )
    else:
        write_create_only_json(cfg_json_path, cfg.to_dict())

    wall_t0 = time.monotonic()
    staged_meta = _validate_staged_ledger_resume(input_dir=input_dir, ledger_path=ledger_path)

    trainer = V3PPOTrainer(cfg, device=device)
    resumed_from: dict[str, Any] | None = None

    initial_cand = _select_best_resume_candidate(
        work_ckpts=work_ckpts,
        out_ckpts=out_ckpts,
        staged_meta=staged_meta,
        include_init_anchor=False,
    )
    if initial_cand is not None:
        load_info = trainer.load_checkpoint(
            Path(initial_cand["path"]),
            verify_markers=bool(initial_cand["require_markers"]),
            restore_rng=True,
        )
        resumed_from = {
            "relpath": str(initial_cand["relpath"]),
            "source": str(initial_cand["source"]),
            "checkpoint_kind": str(load_info["checkpoint_kind"]),
            "worlds_reinitialized": bool(load_info["worlds_reinitialized"]),
            "sha256": str(initial_cand["sha256"]),
            "updates_completed_at_resume": int(trainer.updates_completed),
        }
        phase_stamp(
            "train_resume",
            "OK",
            relpath=resumed_from["relpath"],
            kind=resumed_from["checkpoint_kind"],
            source=resumed_from["source"],
            updates_completed=trainer.updates_completed,
        )
    else:
        # Save a lightweight u0000000 anchor in `work_ckpts` so a stop/error during update 1
        # can roll back cleanly without ever saving half-updated weights.
        init_anchor = work_ckpts / _INIT_ANCHOR_LIGHT_NAME
        if not init_anchor.exists():
            trainer.save_light_checkpoint(init_anchor)

    metrics_writer = RotatingCreateOnlyJSONLWriter(out / "metrics", prefix="updates")
    eval_writer = RotatingCreateOnlyJSONLWriter(out / "eval", prefix="eval")

    updates_this_session = 0
    in_session_restarts = 0
    stopped_reason = "completed"
    error_message: str | None = None
    trainer_state_clean = True
    aborted_mid_update = False

    now_init = time.monotonic()
    last_light_work_mono = now_init
    last_full_work_mono = now_init
    last_light_pub_mono = now_init
    last_full_pub_mono = now_init
    last_eval_mono = now_init

    # Bound published periodic checkpoints per session so `out_ckpts` never exceeds
    # `keep_full_checkpoints` FULL sets and `keep_light_checkpoints` LIGHT files
    # (remember `colab_cli` forbids unlinking `.complete` files inside `output/`).
    max_periodic_light_pubs = max(0, int(cfg.keep_light_checkpoints) - 1)
    max_periodic_full_pubs = max(0, int(cfg.keep_full_checkpoints) - 1)
    published_periodic_lights = 0
    published_periodic_fulls = 0

    eff_light_pub_interval = float(cfg.light_checkpoint_interval_sec)
    eff_full_pub_interval = float(cfg.full_checkpoint_interval_sec)
    if max_wall_seconds is not None and max_periodic_light_pubs > 0:
        eff_light_pub_interval = max(
            eff_light_pub_interval,
            float(max_wall_seconds) / float(max_periodic_light_pubs + 0.25),
        )
    if max_wall_seconds is not None and max_periodic_full_pubs > 0:
        eff_full_pub_interval = max(
            eff_full_pub_interval,
            float(max_wall_seconds) / float(max_periodic_full_pubs + 0.25),
        )

    checkpoint_reserve_sec = (
        min(90.0, max(5.0, float(max_wall_seconds) * 0.08)) if max_wall_seconds is not None else 0.0
    )

    def _counters_dict(t_ref: V3PPOTrainer) -> dict[str, Any]:
        return {
            "updates_completed": int(t_ref.updates_completed),
            "agent_steps_collected": int(t_ref.agent_steps_collected),
            "world_steps_collected": int(t_ref.world_steps_collected),
        }

    with SignalStopFlag() as stop_flag:

        def _wall_expired() -> bool:
            if max_wall_seconds is None:
                return False
            return (time.monotonic() - wall_t0) >= (
                float(max_wall_seconds) - checkpoint_reserve_sec
            )

        def _should_stop() -> bool:
            return bool(stop_flag.stopped)

        try:
            phase_stamp(
                "train_loop",
                "START",
                device=str(trainer.device),
                num_envs=cfg.num_envs,
                learner_slots=cfg.resolved_learner_slots,
                net_size=cfg.net_size,
                start_update=trainer.updates_completed,
                target_updates=cfg.total_updates,
            )
            while trainer.updates_completed < int(cfg.total_updates):
                if stop_flag.stopped:
                    stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
                    break
                if _wall_expired():
                    stopped_reason = "wall_clock_guard"
                    break

                try:
                    trainer_state_clean = False
                    u_t0 = time.monotonic()
                    if step_hook is not None:
                        step_hook(trainer, int(trainer.updates_completed) + 1)
                    step_metrics = trainer.train_step(should_stop=_should_stop)
                    u_elapsed = time.monotonic() - u_t0
                    trainer_state_clean = True
                    updates_this_session += 1

                    if cfg.max_update_seconds is not None and u_elapsed > float(
                        cfg.max_update_seconds
                    ):
                        raise RuntimeError(
                            f"Update {trainer.updates_completed} took {u_elapsed:.2f}s > "
                            f"max_update_seconds={cfg.max_update_seconds:.2f}s"
                        )
                except StopRequestedError:
                    aborted_mid_update = True
                    stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
                    break
                except RuntimeError as exc:
                    if (
                        in_session_restarts >= int(cfg.max_in_session_restarts)
                        or stop_flag.stopped
                        or _wall_expired()
                    ):
                        raise
                    in_session_restarts += 1
                    print(
                        f"V3_PPO_IN_SESSION_RESTART attempt={in_session_restarts}/"
                        f"{cfg.max_in_session_restarts} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    phase_stamp(
                        "in_session_restart",
                        "RECOVER",
                        attempt=in_session_restarts,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    rec_dev = trainer.device
                    del trainer
                    gc.collect()
                    if rec_dev.type == "cuda":
                        torch.cuda.empty_cache()

                    trainer = V3PPOTrainer(cfg, device=rec_dev)
                    rec_cand = _select_best_resume_candidate(
                        work_ckpts=work_ckpts,
                        out_ckpts=out_ckpts,
                        staged_meta=staged_meta,
                        include_init_anchor=True,
                    )
                    if rec_cand is not None:
                        trainer.load_checkpoint(
                            Path(rec_cand["path"]),
                            verify_markers=bool(rec_cand["require_markers"]),
                            restore_rng=True,
                        )
                    # Advance minibatch permutation stream so a transient batch spike isn't replayed
                    _ = torch.randperm(8, generator=trainer.perm_gen)
                    trainer_state_clean = True
                    continue

                now_mono = time.monotonic()
                eval_res: dict[str, Any] | None = None
                is_target_end = trainer.updates_completed >= int(cfg.total_updates)
                due_eval_updates = int(cfg.eval_every_updates) > 0 and (
                    (trainer.updates_completed % int(cfg.eval_every_updates) == 0) or is_target_end
                )
                due_eval_minutes = float(cfg.eval_every_minutes) > 0.0 and (
                    (now_mono - last_eval_mono) >= float(cfg.eval_every_minutes) * 60.0
                )
                if due_eval_updates or due_eval_minutes:
                    eval_res = trainer.run_evaluation()
                    eval_record = {
                        "update": trainer.updates_completed,
                        "total_agent_steps": trainer.agent_steps_collected,
                        **eval_res,
                    }
                    eval_writer.append(eval_record)
                    last_eval_mono = time.monotonic()

                metrics_writer.append(step_metrics)
                now_mono = time.monotonic()
                print(
                    _format_progress_line(
                        step_metrics,
                        total_updates=int(cfg.total_updates),
                        wall_elapsed=now_mono - wall_t0,
                        eval_metrics=eval_res,
                    ),
                    flush=True,
                )

                tag = f"u{trainer.updates_completed:07d}"
                due_light_u = int(cfg.checkpoint_every_updates) > 0 and (
                    trainer.updates_completed % int(cfg.checkpoint_every_updates) == 0
                )
                due_full_u = int(cfg.full_checkpoint_every_updates) > 0 and (
                    trainer.updates_completed % int(cfg.full_checkpoint_every_updates) == 0
                )
                due_light_t = (now_mono - last_light_work_mono) >= float(
                    cfg.light_checkpoint_interval_sec
                )
                due_full_t = (now_mono - last_full_work_mono) >= float(
                    cfg.full_checkpoint_interval_sec
                )

                light_work_path = work_ckpts / f"light_{tag}.pt"
                if due_light_u or due_light_t or due_full_u or due_full_t:
                    if not light_work_path.exists():
                        trainer.save_light_checkpoint(light_work_path)
                    last_light_work_mono = now_mono

                full_work_index: Path | None = None
                if due_full_u or due_full_t:
                    cand_idx = work_ckpts / f"full_{tag}.index.json"
                    if not cand_idx.exists():
                        full_work_index = trainer.save_full_checkpoint(
                            work_ckpts,
                            tag=tag,
                            light_path=light_work_path,
                        )
                    else:
                        full_work_index = cand_idx
                    last_full_work_mono = now_mono

                if due_light_u or due_light_t or due_full_u or due_full_t:
                    prune_old_checkpoints(
                        work_ckpts,
                        keep_full=int(cfg.keep_full_checkpoints),
                        keep_light=int(cfg.keep_light_checkpoints),
                    )

                # Publish periodic FULL if due and within session periodic FULL budget
                if full_work_index is not None and (
                    due_full_u
                    or (
                        published_periodic_fulls < max_periodic_full_pubs
                        and (now_mono - last_full_pub_mono) >= eff_full_pub_interval
                    )
                ):
                    if published_periodic_fulls < max_periodic_full_pubs or due_full_u:
                        publish_checkpoint_to_output(
                            full_work_index,
                            out_ckpts,
                            reason="periodic_full",
                            counters=_counters_dict(trainer),
                        )
                        published_periodic_fulls += 1
                        last_full_pub_mono = now_mono
                        last_light_pub_mono = now_mono
                        metrics_writer.seal_current_segment()
                        eval_writer.seal_current_segment()
                elif light_work_path.exists() and (
                    due_light_u
                    or (
                        published_periodic_lights < max_periodic_light_pubs
                        and (now_mono - last_light_pub_mono) >= eff_light_pub_interval
                    )
                ):
                    if published_periodic_lights < max_periodic_light_pubs or due_light_u:
                        publish_checkpoint_to_output(
                            light_work_path,
                            out_ckpts,
                            reason="periodic_light",
                            counters=_counters_dict(trainer),
                        )
                        published_periodic_lights += 1
                        last_light_pub_mono = now_mono
                        metrics_writer.seal_current_segment()
                        eval_writer.seal_current_segment()

        except Exception as exc:
            stopped_reason = f"exception:{type(exc).__name__}"
            error_message = str(exc)
            raise
        finally:
            metrics_writer.close()
            eval_writer.close()

            is_clean_stop = (
                trainer_state_clean
                and not aborted_mid_update
                and error_message is None
                and stopped_reason in {"completed", "wall_clock_guard"}
            )

            eval_configured = (int(cfg.eval_every_updates) > 0) or (
                float(cfg.eval_every_minutes) > 0.0
            )
            if (
                is_clean_stop
                and eval_configured
                and trainer.last_eval_metrics is None
                and trainer.updates_completed > 0
                and not stop_flag.stopped
            ):
                ev_final = trainer.run_evaluation()
                eval_writer_final = RotatingCreateOnlyJSONLWriter(out / "eval", prefix="eval_final")
                eval_writer_final.append(
                    {
                        "update": trainer.updates_completed,
                        "total_agent_steps": trainer.agent_steps_collected,
                        **ev_final,
                    }
                )
                eval_writer_final.close()

            if trainer_state_clean and not aborted_mid_update and error_message is None:
                # Trainer is at an exact update boundary: publish LIGHT on any clean boundary stop,
                # plus sharded FULL on a clean stop (`completed` or `wall_clock_guard`).
                stop_tag = f"u{trainer.updates_completed:07d}"
                stop_light = work_ckpts / f"light_{stop_tag}.pt"
                if not stop_light.exists():
                    trainer.save_light_checkpoint(stop_light)

                if is_clean_stop:
                    stop_full = work_ckpts / f"full_{stop_tag}.index.json"
                    if not stop_full.exists():
                        stop_full = trainer.save_full_checkpoint(
                            work_ckpts,
                            tag=stop_tag,
                            light_path=stop_light,
                        )
                    publish_checkpoint_to_output(
                        stop_full,
                        out_ckpts,
                        reason=stopped_reason,
                        counters=_counters_dict(trainer),
                    )
                else:
                    publish_checkpoint_to_output(
                        stop_light,
                        out_ckpts,
                        reason=stopped_reason,
                        counters=_counters_dict(trainer),
                    )
                prune_old_checkpoints(
                    work_ckpts,
                    keep_full=int(cfg.keep_full_checkpoints),
                    keep_light=int(cfg.keep_light_checkpoints),
                )
                final_counters = _counters_dict(trainer)
            else:
                # Mid-update stop or fatal exception: NEVER save or publish half-updated state!
                # Publish any unexported verified boundary checkpoint from `work_ckpts` first, then
                # restore trainer counters from the best verified boundary candidate.
                work_clean = resolve_resume_checkpoint(
                    work_ckpts,
                    require_markers=True,
                    include_init_anchor=(staged_meta is None),
                )
                if work_clean is not None:
                    wb_path = Path(work_clean["path"])
                    if wb_path.name != _INIT_ANCHOR_LIGHT_NAME:
                        publish_checkpoint_to_output(
                            wb_path,
                            out_ckpts,
                            reason=f"rollback_{stopped_reason}",
                            counters={
                                "updates_completed": int(work_clean["updates_completed"]),
                                "agent_steps_collected": int(
                                    work_clean.get("agent_steps_collected", 0)
                                ),
                                "world_steps_collected": 0,
                            },
                        )

                clean_fallback = _select_best_resume_candidate(
                    work_ckpts=work_ckpts,
                    out_ckpts=out_ckpts,
                    staged_meta=staged_meta,
                    include_init_anchor=True,
                )
                if clean_fallback is not None:
                    fb_path = Path(clean_fallback["path"])
                    if (
                        bool(clean_fallback.get("require_markers", False))
                        and fb_path.parent != out_ckpts
                        and fb_path.name != _INIT_ANCHOR_LIGHT_NAME
                    ):
                        publish_checkpoint_to_output(
                            fb_path,
                            out_ckpts,
                            reason=f"rollback_{stopped_reason}",
                            counters={
                                "updates_completed": int(clean_fallback["updates_completed"]),
                                "agent_steps_collected": int(
                                    clean_fallback.get("agent_steps_collected", 0)
                                ),
                                "world_steps_collected": 0,
                            },
                        )
                    # Restore trainer weights & counters to clean boundary for honest reporting
                    trainer.load_checkpoint(
                        fb_path,
                        verify_markers=bool(clean_fallback.get("require_markers", False)),
                        restore_rng=False,
                    )
                final_counters = _counters_dict(trainer)

            write_latest_checkpoint_manifest(
                out_ckpts,
                reason=stopped_reason,
                counters=final_counters,
            )

            passed = (
                error_message is None
                and not aborted_mid_update
                and stopped_reason in {"completed", "wall_clock_guard"}
                and (int(cfg.total_updates) == 0 or trainer.updates_completed >= 1)
                and (
                    stopped_reason == "completed"
                    or (max_wall_seconds is not None and updates_this_session >= 1)
                )
            )
            status = "PASS" if passed else "FAIL"

            result_path = out / "train_result.json"
            if not result_path.exists():
                result_payload = {
                    "schema": V3_PPO_TRAIN_RESULT_SCHEMA,
                    "namespace": V3_PPO_NAMESPACE,
                    "status": status,
                    "stopped_reason": stopped_reason,
                    "aborted_mid_update": bool(aborted_mid_update),
                    "in_session_restarts": int(in_session_restarts),
                    "error": error_message,
                    "device": str(trainer.device),
                    "parameter_count": trainer.raw_model.parameter_count(),
                    "updates_completed": trainer.updates_completed,
                    "updates_this_session": updates_this_session,
                    "world_steps_collected": trainer.world_steps_collected,
                    "agent_steps_collected": trainer.agent_steps_collected,
                    "learner_steps_collected": trainer.learner_steps_collected,
                    "episodes_completed": trainer.episodes_completed,
                    "last_rollout_mean_ep_return": round(trainer.last_rollout_mean_ep_return, 6),
                    "last_eval": trainer.last_eval_metrics,
                    "resumed_from": resumed_from,
                    "memory_estimate": estimate_training_memory(cfg),
                    "config": cfg.to_dict(),
                    "wall_seconds": round(time.monotonic() - wall_t0, 4),
                }
                write_create_only_json(result_path, result_payload)
            phase_stamp(
                "train_loop",
                "END",
                status=status,
                stopped_reason=stopped_reason,
                updates_completed=trainer.updates_completed,
                in_session_restarts=in_session_restarts,
            )

    return json.loads((out / "train_result.json").read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m research.v3_ppo.ppo``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=os.environ.get("SNAKE_DQN_DEVICE", "cpu"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--config-json", type=str, default=os.environ.get("V3_PPO_CONFIG_JSON"))

    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-snakes", type=int, default=24)
    parser.add_argument("--arena-radius", type=float, default=600.0)
    parser.add_argument("--learner-slots", type=int, default=None)
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="food_seeking",
        choices=("food_seeking", "random_safe", "cardinal6_safe"),
    )
    parser.add_argument("--net-size", type=str, default="S", choices=("S", "M"))
    parser.add_argument("--rollout-len", type=int, default=32)
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument(
        "--light-checkpoint-interval-sec",
        "--checkpoint-interval-sec",
        dest="light_checkpoint_interval_sec",
        type=float,
        default=1800.0,
    )
    parser.add_argument("--full-checkpoint-interval-sec", type=float, default=7200.0)
    parser.add_argument(
        "--checkpoint-shard-bytes",
        type=int,
        default=DEFAULT_CHECKPOINT_SHARD_BYTES,
    )
    parser.add_argument("--checkpoint-every-updates", type=int, default=0)
    parser.add_argument("--full-checkpoint-every-updates", type=int, default=0)
    parser.add_argument("--eval-every-minutes", type=float, default=30.0)
    parser.add_argument("--eval-every-updates", type=int, default=0)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--eval-burnin-ticks", type=int, default=500)
    parser.add_argument("--eval-ticks", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--compile", action="store_true")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=True)
    amp_group.add_argument("--no-amp", "--fp32", dest="amp", action="store_false")

    cl_group = parser.add_mutually_exclusive_group()
    cl_group.add_argument(
        "--channels-last", dest="use_channels_last", action="store_true", default=True
    )
    cl_group.add_argument("--no-channels-last", dest="use_channels_last", action="store_false")

    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=(
            float(os.environ["V3_PPO_MAX_WALL_SECONDS"])
            if os.environ.get("V3_PPO_MAX_WALL_SECONDS")
            else None
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; exits non-zero whenever terminal ``status != 'PASS'``."""
    args = build_arg_parser().parse_args(argv)
    if args.config_json:
        cfg = V3PPOConfig.from_dict(json.loads(args.config_json))
    else:
        game_cfg = GameV3Config(
            arena_radius=float(args.arena_radius),
            num_snakes=int(args.num_snakes),
        )
        cfg = V3PPOConfig(
            game_config=game_cfg,
            num_envs=int(args.num_envs),
            learner_slots=args.learner_slots,
            opponent_policy=args.opponent_policy,
            net_size=args.net_size,
            amp=bool(args.amp),
            use_channels_last=bool(args.use_channels_last),
            compile_model=bool(args.compile),
            rollout_len=int(args.rollout_len),
            ppo_epochs=int(args.ppo_epochs),
            minibatch_size=args.minibatch_size,
            lr=float(args.lr),
            total_updates=int(args.total_updates),
            light_checkpoint_interval_sec=float(args.light_checkpoint_interval_sec),
            full_checkpoint_interval_sec=float(args.full_checkpoint_interval_sec),
            checkpoint_shard_bytes=int(args.checkpoint_shard_bytes),
            checkpoint_every_updates=int(args.checkpoint_every_updates),
            full_checkpoint_every_updates=int(args.full_checkpoint_every_updates),
            eval_every_minutes=float(args.eval_every_minutes),
            eval_every_updates=int(args.eval_every_updates),
            eval_envs=int(args.eval_envs),
            eval_burnin_ticks=int(args.eval_burnin_ticks),
            eval_ticks=int(args.eval_ticks),
            seed=int(args.seed),
        )

    res = run_training_job(
        cfg=cfg,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        device=args.device,
        input_dir=args.input_dir,
        ledger_path=args.ledger,
        max_wall_seconds=args.max_wall_seconds,
    )
    if res.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
