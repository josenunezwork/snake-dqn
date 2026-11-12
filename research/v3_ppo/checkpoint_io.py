"""LIGHT and sharded FULL checkpoint I/O, compression, verification, retention, and publishing."""

from __future__ import annotations

import json
import shutil
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from research.arena_ppo.runtime import (
    remove_completed_artifact,
    sha256_file,
    verify_completed_artifact,
)
from research.v3_ppo.config import (
    V3_PPO_NAMESPACE,
    ObsLayout,
    V3PPOConfig,
    obs_layout_from_game_config,
)
from research.v3_ppo.exports import (
    artifact_or_sidecar_exists,
    publish_hardlinked_artifacts,
    write_create_only_artifact,
    write_create_only_json,
)

V3_PPO_LIGHT_CHECKPOINT_SCHEMA = "v3-ppo-light-checkpoint/v2"
V3_PPO_FULL_INDEX_SCHEMA = "v3-ppo-full-checkpoint-index/v2"
V3_PPO_SIM_SHARD_SCHEMA = "v3-ppo-sim-shard/v2"

INIT_ANCHOR_LIGHT_NAME = "light_u0000000_init.pt"
_INIT_ANCHOR_LIGHT_NAME = INIT_ANCHOR_LIGHT_NAME

_SIM_ZLIB_TOTAL_THRESHOLD_BYTES = 0
_SIM_ZLIB_TENSOR_MIN_BYTES = 4 * 1024
_SIM_ZLIB_CHUNK_BYTES = 128 * 1024 * 1024


def ckpt_tag(updates_completed: int, restart_generation: int) -> str:
    """Format canonical checkpoint filename tag from update count and restart generation."""
    if int(restart_generation) <= 0:
        return f"u{int(updates_completed):07d}"
    return f"u{int(updates_completed):07d}_r{int(restart_generation):02d}"


_ckpt_tag = ckpt_tag


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
    out: dict[str, Any] = {
        "config": sim_sd["config"],
        "num_envs": w1 - w0,
        "tick": int(sim_sd["tick"]),
        "subtick_offset": int(sim_sd.get("subtick_offset", 0)),
        "tensors": sliced_tensors,
    }
    if "clean_step_streak" in sim_sd:
        out["clean_step_streak"] = int(sim_sd["clean_step_streak"])
    return out


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
        shard_streak = int(
            unpacked["clean_step_streak"]
            if "clean_step_streak" in unpacked
            else item.get("sim_clean_step_streak", 0)
        )
        if ref_cfg is None:
            ref_cfg = unpacked["config"]
            ref_tick = int(unpacked["tick"])
            ref_subtick = int(unpacked.get("subtick_offset", 0))
            ref_streak = shard_streak
        else:
            if (
                unpacked["config"] != ref_cfg
                or int(unpacked["tick"]) != ref_tick
                or int(unpacked.get("subtick_offset", 0)) != ref_subtick
                or shard_streak != ref_streak
            ):
                raise ValueError("Inconsistent simulator metadata across FULL checkpoint shards")
        unpacked_list.append(unpacked)
        cursor = w1

    if cursor != int(expected_num_envs):
        raise ValueError(
            f"Incomplete FULL checkpoint shards: covered {cursor}/{expected_num_envs} worlds"
        )

    assert ref_cfg is not None and ref_tick is not None and ref_subtick is not None
    streak_val = int(ref_streak or 0)
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
        "clean_step_streak": streak_val,
        "tensors": merged_tensors,
    }
    return merged_sd, streak_val


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
    ckpt_obs_layout: dict[str, Any] | None = None
    if ckpt.get("obs_layout") is not None:
        parsed_layout = ObsLayout.from_dict(ckpt["obs_layout"])
        if isinstance(ckpt.get("config"), dict):
            cfg_layout = obs_layout_from_game_config(
                V3PPOConfig.from_dict(ckpt["config"]).game_config
            )
            if parsed_layout != cfg_layout:
                raise ValueError(
                    f"LIGHT checkpoint obs_layout {parsed_layout.to_dict()} != "
                    f"config obs_layout {cfg_layout.to_dict()} at {p}"
                )
        ckpt_obs_layout = parsed_layout.to_dict()
    st = p.stat()
    raw_ts = ckpt.get("created_unix_ts")
    created_ts = float(raw_ts) if isinstance(raw_ts, (int, float)) else float(st.st_mtime)
    return {
        "kind": "light",
        "path": p,
        "files": [p],
        "sha256": digest,
        "size_bytes": int(st.st_size),
        "updates_completed": int(counters["updates_completed"]),
        "world_steps_collected": int(counters.get("world_steps_collected", 0)),
        "agent_steps_collected": int(counters.get("agent_steps_collected", 0)),
        "created_unix_ts": created_ts,
        "restart_generation": int(ckpt.get("restart_generation", 0)),
        "config": ckpt.get("config"),
        "obs_layout": ckpt_obs_layout,
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
    idx_rgen = int(idx_obj.get("restart_generation", light_meta.get("restart_generation", 0)))
    if int(light_meta.get("restart_generation", 0)) != idx_rgen:
        raise ValueError(
            f"FULL index restart_generation ({idx_rgen}) != "
            f"LIGHT restart_generation ({light_meta.get('restart_generation', 0)})"
        )

    idx_obs_layout: dict[str, Any] | None = light_meta.get("obs_layout")
    if idx_obj.get("obs_layout") is not None:
        parsed_idx_layout = ObsLayout.from_dict(idx_obj["obs_layout"])
        if light_meta.get("obs_layout") is not None:
            parsed_light_layout = ObsLayout.from_dict(light_meta["obs_layout"])
            if parsed_idx_layout != parsed_light_layout:
                raise ValueError(
                    f"FULL index obs_layout {parsed_idx_layout.to_dict()} != "
                    f"LIGHT obs_layout {parsed_light_layout.to_dict()} at {idx_p}"
                )
        if isinstance(light_meta.get("config"), dict):
            cfg_layout = obs_layout_from_game_config(
                V3PPOConfig.from_dict(light_meta["config"]).game_config
            )
            if parsed_idx_layout != cfg_layout:
                raise ValueError(
                    f"FULL index obs_layout {parsed_idx_layout.to_dict()} != "
                    f"config obs_layout {cfg_layout.to_dict()} at {idx_p}"
                )
        idx_obs_layout = parsed_idx_layout.to_dict()

    idx_st = idx_p.stat()
    raw_ts = idx_obj.get("created_unix_ts")
    created_ts = float(raw_ts) if isinstance(raw_ts, (int, float)) else float(idx_st.st_mtime)
    all_files = [idx_p, light_p, *shard_paths]
    return {
        "kind": "full",
        "path": idx_p,
        "light_path": light_p,
        "shard_paths": shard_paths,
        "files": all_files,
        "sha256": idx_sha,
        "size_bytes": int(idx_st.st_size) + int(light_meta["size_bytes"]) + total_shard_bytes,
        "num_shards": len(shard_paths),
        "updates_completed": updates_completed,
        "world_steps_collected": int(
            idx_obj.get("world_steps_collected", light_meta.get("world_steps_collected", 0))
        ),
        "agent_steps_collected": int(idx_obj.get("agent_steps_collected", 0)),
        "created_unix_ts": created_ts,
        "restart_generation": idx_rgen,
        "config": light_meta["config"],
        "obs_layout": idx_obs_layout,
    }


def _is_unshared_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_nlink <= 1
    except OSError:
        return False


def prune_work_checkpoints_after_rewind(
    work_ckpts: Path,
    *,
    rewind_updates: int,
) -> list[str]:
    """Remove unshared ``work_ckpts`` artifacts strictly newer than ``rewind_updates``."""
    if not work_ckpts.is_dir():
        return []
    removed: list[str] = []
    for idx_path in sorted(work_ckpts.glob("full_*.index.json")):
        try:
            idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
            u_num = int(idx_obj.get("updates_completed", -1))
            s_files = [
                work_ckpts / str(s["filename"])
                for s in (idx_obj.get("shards") or [])
                if isinstance(s, dict) and s.get("filename")
            ]
        except Exception:
            continue
        if u_num <= int(rewind_updates):
            continue
        for sp in s_files:
            if _is_unshared_file(sp):
                remove_completed_artifact(sp)
                removed.append(sp.name)
        if _is_unshared_file(idx_path):
            remove_completed_artifact(idx_path.with_suffix(".meta.json"))
            remove_completed_artifact(idx_path)
            removed.append(idx_path.name)

    for lp in sorted(work_ckpts.glob("light_*.pt")):
        if lp.name == INIT_ANCHOR_LIGHT_NAME or not _is_unshared_file(lp):
            continue
        try:
            meta_l = verify_light_checkpoint(lp, require_markers=False)
            if int(meta_l["updates_completed"]) <= int(rewind_updates):
                continue
        except Exception:
            pass
        remove_completed_artifact(lp.with_suffix(".meta.json"))
        remove_completed_artifact(lp)
        removed.append(lp.name)
    return removed


_prune_work_checkpoints_after_rewind = prune_work_checkpoints_after_rewind


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
    keep_f = max(1, int(keep_full))
    retained_full_names: set[str] = {e[1] for e in full_entries[-keep_f:]}
    to_remove_full_names: set[str] = set()
    for _, idx_name, idx_path, s_files, _, can_prune in full_entries:
        if idx_name in retained_full_names or not can_prune:
            continue
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

    standalone_lights: list[Path] = []
    for lp in sorted(checkpoints_dir.glob("light_*.pt")):
        if lp.name == INIT_ANCHOR_LIGHT_NAME or lp.name in protected_lights:
            continue
        standalone_lights.append(lp)

    keep_l = max(1, int(keep_light))
    retained_light_names: set[str] = {p.name for p in standalone_lights[-keep_l:]}
    for old_lp in standalone_lights:
        if old_lp.name not in retained_light_names and _is_unshared_file(old_lp):
            remove_completed_artifact(old_lp.with_suffix(".meta.json"))
            remove_completed_artifact(old_lp)
            removed.append(old_lp.name)

    return removed


def publish_checkpoint_to_output(
    work_ckpt_path: Path,
    out_checkpoints_dir: Path,
    *,
    reason: str,
    counters: dict[str, Any],
    copy_only: bool = False,
) -> list[str]:
    """Publish a verified LIGHT `.pt` or FULL `.index.json` set into ``out_checkpoints_dir``.

    For FULL sets, publishes `light_*.pt` and all `sim_shard_*.pt` BEFORE `full_*.index.json`
    so the index becomes `.complete` strictly last. Uses hard links from persistent ``work/`` by
    default (`copy_only=False`) or fresh single-inode copies when `copy_only=True`.
    """
    out_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    src = Path(work_ckpt_path)
    published: list[str] = []

    if src.name.endswith(".index.json"):
        full_meta = verify_full_checkpoint_set(src, require_markers=True)
        ordered_sources = [full_meta["light_path"], *full_meta["shard_paths"], src]
        for f_src in ordered_sources:
            dst = out_checkpoints_dir / f_src.name
            if not artifact_or_sidecar_exists(dst):
                if copy_only:
                    write_create_only_artifact(dst, lambda tmp, s=f_src: shutil.copyfile(s, tmp))
                else:
                    publish_hardlinked_artifacts(f_src, [dst], staging_dir=f_src.parent)
                published.append(dst.name)
        meta_path = (out_checkpoints_dir / src.name).with_suffix(".meta.json")
        if not artifact_or_sidecar_exists(meta_path):
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
    if not artifact_or_sidecar_exists(dst):
        ckpt_sha = (
            write_create_only_artifact(dst, lambda tmp, s=src: shutil.copyfile(s, tmp))
            if copy_only
            else publish_hardlinked_artifacts(src, [dst], staging_dir=src.parent)
        )
    else:
        ckpt_sha = sha256_file(dst)
    if dst.name not in published:
        published.append(dst.name)
    meta_path = dst.with_suffix(".meta.json")
    if not artifact_or_sidecar_exists(meta_path):
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
