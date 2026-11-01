"""Atomic immutable sub-replicate stage & critic-epoch checkpointing for Track-4 Gate 1.

Persists finished harvest partitions, counterfactual root-branch tables, per-epoch critic
training states, and frozen pre-test critic/baseline weights under ``output/t4_gate1/checkpoints``
with ``.sha256`` and ``.complete`` sidecars published last (never mutating or removing any
published file, preserving ``research/colab_cli/runtime.py`` immutable export invariants).

On resume, verifies ``.complete`` + ``.sha256`` sidecars and inner ``(study, namespace,
replicate_index, stage, config_fingerprint)`` metadata before skipping verified stages or
continuing critic training from the latest verified epoch ``k + 1``.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.t4_gate1.featurizer import CompactBoardDataset
from research.t4_gate1.harvest import (
    BranchedRootsTable,
    HarvestedWorldPartition,
    PreparedRootSnapshot,
)
from research.t4_gate1.model import LengthHorizonBaseline, PrivilegedFactoredCritic
from tools.ops.package_spec import publish_immutable, verify_immutable

CHECKPOINT_SCHEMA = "t34-t4-gate1-stage-checkpoint/v1"
_CONTAINER_MAGIC = b"T4CKPT01"

_DATASET_ARRAY_FIELDS: tuple[str, ...] = (
    "world_id",
    "seed",
    "life_id",
    "tick",
    "h_rem",
    "b8_action",
    "legal_mask",
    "snake_alive",
    "snake_length",
    "snake_seg_count",
    "snake_dir",
    "snake_boost_frames",
    "snake_hunger",
    "snake_respawn",
    "snake_boosted",
    "head_xy",
    "tail_xy",
    "seg_row_offsets",
    "seg_row_counts",
    "seg_slot",
    "seg_x",
    "seg_y",
    "seg_ttl",
    "food_row_offsets",
    "food_row_counts",
    "food_x",
    "food_y",
    "food_is_corpse",
    "tau_alive",
    "died_in_session",
    "return_g",
    "forage_short",
    "forage_short_weight",
    "forage_long",
    "forage_long_valid",
)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def harvest_config_fingerprint(
    config: Mapping[str, Any],
    *,
    study: str,
    namespace: str,
    partition: str,
    world_rows: Sequence[Mapping[str, Any]],
    max_roots: int,
) -> str:
    """Deterministic digest binding all parameters that affect a partition harvest."""
    return _json_sha256(
        {
            "study": str(study),
            "namespace": str(namespace),
            "partition": str(partition),
            "b8_sha256": str(config.get("b8_sha256", "")),
            "frp3_sha256": str(config.get("frp3_sha256", "")),
            "session_ticks": int(config["session_ticks"]),
            "max_capacity": int(config.get("max_capacity", 3200)),
            "stride_mature": int(config.get("stride_mature", 4)),
            "stride_early": int(config.get("stride_early", 16)),
            "min_mature_length": int(config.get("min_mature_length", 600)),
            "branch_horizon": int(config.get("branch_horizon", 300)),
            "max_roots": int(max_roots),
            "death_lookbacks": sorted(
                int(x) for x in config.get("death_lookbacks", [16, 48, 112, 200]) if int(x) > 0
            ),
            "ordinary_root_stride": int(config.get("ordinary_root_stride", 96)),
            "worlds": [
                {"global_index": int(r["global_index"]), "seed": int(r["seed"])} for r in world_rows
            ],
        }
    )


def branch_config_fingerprint(
    config: Mapping[str, Any],
    *,
    study: str,
    namespace: str,
    roots: Sequence[PreparedRootSnapshot],
    horizon: int,
    selection_suffixes: Sequence[int],
    scoring_suffixes: Sequence[int],
) -> str:
    """Deterministic digest binding all parameters that affect CRN root branching."""
    return _json_sha256(
        {
            "study": str(study),
            "namespace": str(namespace),
            "b8_sha256": str(config.get("b8_sha256", "")),
            "frp3_sha256": str(config.get("frp3_sha256", "")),
            "max_capacity": int(config.get("max_capacity", 3200)),
            "horizon": int(horizon),
            "selection_suffixes": [int(x) for x in selection_suffixes],
            "scoring_suffixes": [int(x) for x in scoring_suffixes],
            "root_uids": [str(r.root_uid) for r in roots],
        }
    )


def critic_config_fingerprint(
    config: Mapping[str, Any],
    *,
    study: str,
    namespace: str,
    replicate_index: int,
    critic_seed: int,
    fit_fingerprint: str,
    val_fingerprint: str,
    train_branch_suffixes: Sequence[int],
) -> str:
    """Deterministic digest binding upstream fit/val data + critic training hyperparameters."""
    return _json_sha256(
        {
            "study": str(study),
            "namespace": str(namespace),
            "replicate_index": int(replicate_index),
            "critic_seed": int(critic_seed),
            "fit_fingerprint": str(fit_fingerprint),
            "val_fingerprint": str(val_fingerprint),
            "train_branch_suffixes": [int(x) for x in train_branch_suffixes],
            "max_capacity": int(config.get("max_capacity", 3200)),
            "min_mature_length": int(config.get("min_mature_length", 600)),
            "branch_horizon": int(config.get("branch_horizon", 300)),
            "critic_epochs": int(config.get("critic_epochs", 12)),
            "critic_batch_size": int(config.get("critic_batch_size", 512)),
            "critic_lr": float(config.get("critic_lr", 6e-4)),
            "critic_weight_decay": float(config.get("critic_weight_decay", 1e-3)),
            "critic_early_stop_patience": config.get("critic_early_stop_patience"),
            "critic_min_epochs": int(config.get("critic_min_epochs", 1)),
        }
    )


def _pack_container(
    meta: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    compress_level: int = 1,
) -> bytes:
    """Pack JSON metadata + named contiguous NumPy arrays into a fast zlib-1 binary container."""
    array_index: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(arrays.keys()):
        arr = np.ascontiguousarray(arrays[name])
        raw = arr.tobytes()
        nbytes = len(raw)
        array_index.append(
            {
                "name": name,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        chunks.append(raw)
        offset += nbytes

    full_header = {
        "meta": dict(meta),
        "arrays": array_index,
        "uncompressed_bytes": offset,
    }
    header_bytes = json.dumps(full_header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    compressed_blob = zlib.compress(b"".join(chunks), level=compress_level)
    return _CONTAINER_MAGIC + struct.pack("<Q", len(header_bytes)) + header_bytes + compressed_blob


def _unpack_container(raw: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Unpack a ``T4CKPT01`` binary container into ``(meta, arrays_dict)``."""
    if len(raw) < 16 or raw[:8] != _CONTAINER_MAGIC:
        raise ValueError("Invalid Gate-1 stage checkpoint magic header")
    (header_len,) = struct.unpack("<Q", raw[8:16])
    if header_len <= 0 or 16 + header_len > len(raw):
        raise ValueError("Corrupt Gate-1 stage checkpoint header length")
    full_header = json.loads(raw[16 : 16 + header_len].decode("utf-8"))
    blob = zlib.decompress(raw[16 + header_len :])
    if len(blob) != int(full_header["uncompressed_bytes"]):
        raise ValueError("Decompressed checkpoint array payload size mismatch")

    arrays: dict[str, np.ndarray] = {}
    mv = memoryview(blob)
    for spec in full_header["arrays"]:
        off = int(spec["offset"])
        nbytes = int(spec["nbytes"])
        dt = np.dtype(str(spec["dtype"]))
        shape = tuple(int(x) for x in spec["shape"])
        arr = np.frombuffer(mv[off : off + nbytes], dtype=dt).reshape(shape).copy()
        arrays[str(spec["name"])] = arr
    return dict(full_header["meta"]), arrays


def _encode_dataset(prefix: str, ds: CompactBoardDataset, out: dict[str, np.ndarray]) -> int:
    for field_name in _DATASET_ARRAY_FIELDS:
        out[f"{prefix}.{field_name}"] = np.asarray(getattr(ds, field_name))
    return int(ds.num_rows)


def _decode_dataset(
    prefix: str, num_rows: int, arrays: Mapping[str, np.ndarray]
) -> CompactBoardDataset:
    kwargs: dict[str, Any] = {"num_rows": int(num_rows)}
    for field_name in _DATASET_ARRAY_FIELDS:
        kwargs[field_name] = arrays[f"{prefix}.{field_name}"]
    return CompactBoardDataset(**kwargs)


def _encode_roots(
    roots: Sequence[PreparedRootSnapshot], out: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    meta_rows: list[dict[str, Any]] = []
    for i, r in enumerate(roots):
        p = f"root.{i}"
        meta_rows.append(
            {
                "root_uid": str(r.root_uid),
                "kind": str(r.kind),
                "lookback": int(r.lookback) if r.lookback is not None else None,
                "world_id": int(r.world_id),
                "seed": int(r.seed),
                "life_id": int(r.life_id),
                "tick": int(r.tick),
                "h_rem": int(r.h_rem),
                "hero_length": int(r.hero_length),
                "b8_action": int(r.b8_action),
                "frame": int(r.frame),
                "vec_keys": sorted(r.vec_tensors.keys()),
            }
        )
        out[f"{p}.legal_mask"] = np.asarray(r.legal_mask, dtype=bool)
        for s in range(6):
            out[f"{p}.body.{s}"] = np.asarray(r.body_segments[s], dtype=np.int16)
        out[f"{p}.seg_count"] = np.asarray(r.seg_count, dtype=np.int64)
        out[f"{p}.length"] = np.asarray(r.length, dtype=np.int64)
        out[f"{p}.alive"] = np.asarray(r.alive, dtype=bool)
        out[f"{p}.direction"] = np.asarray(r.direction, dtype=np.int64)
        out[f"{p}.boost_frames"] = np.asarray(r.boost_frames, dtype=np.int64)
        out[f"{p}.frames_since_food"] = np.asarray(r.frames_since_food, dtype=np.int64)
        out[f"{p}.respawn_timer"] = np.asarray(r.respawn_timer, dtype=np.int64)
        out[f"{p}.reward_prev_length"] = np.asarray(r.reward_prev_length, dtype=np.int64)
        out[f"{p}.boosted_this_step"] = np.asarray(r.boosted_this_step, dtype=bool)
        out[f"{p}.last_reward"] = np.asarray(r.last_reward, dtype=np.float32)
        out[f"{p}.last_done"] = np.asarray(r.last_done, dtype=bool)
        out[f"{p}.last_death_cause"] = np.asarray(r.last_death_cause, dtype=np.int64)
        out[f"{p}.last_kills"] = np.asarray(r.last_kills, dtype=np.int64)
        out[f"{p}.last_food_ate"] = np.asarray(r.last_food_ate, dtype=bool)
        out[f"{p}.last_ambient_food_ate"] = np.asarray(r.last_ambient_food_ate, dtype=bool)
        out[f"{p}.last_corpse_food_ate"] = np.asarray(r.last_corpse_food_ate, dtype=bool)
        out[f"{p}.last_transition_valid"] = np.asarray(r.last_transition_valid, dtype=bool)
        out[f"{p}.food_cells"] = np.asarray(r.food_cells, dtype=np.int16)
        out[f"{p}.corpse_food"] = np.asarray(r.corpse_food, dtype=bool)
        for vk in sorted(r.vec_tensors.keys()):
            out[f"{p}.vec.{vk}"] = np.asarray(r.vec_tensors[vk])
    return meta_rows


def _decode_roots(
    meta_rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]
) -> list[PreparedRootSnapshot]:
    roots: list[PreparedRootSnapshot] = []
    for i, m in enumerate(meta_rows):
        p = f"root.{i}"
        vec_dict = {str(vk): arrays[f"{p}.vec.{vk}"] for vk in m["vec_keys"]}
        roots.append(
            PreparedRootSnapshot(
                root_uid=str(m["root_uid"]),
                kind=str(m["kind"]),
                lookback=int(m["lookback"]) if m["lookback"] is not None else None,
                world_id=int(m["world_id"]),
                seed=int(m["seed"]),
                life_id=int(m["life_id"]),
                tick=int(m["tick"]),
                h_rem=int(m["h_rem"]),
                hero_length=int(m["hero_length"]),
                b8_action=int(m["b8_action"]),
                legal_mask=arrays[f"{p}.legal_mask"],
                body_segments=tuple(arrays[f"{p}.body.{s}"] for s in range(6)),
                seg_count=arrays[f"{p}.seg_count"],
                length=arrays[f"{p}.length"],
                alive=arrays[f"{p}.alive"],
                direction=arrays[f"{p}.direction"],
                boost_frames=arrays[f"{p}.boost_frames"],
                frames_since_food=arrays[f"{p}.frames_since_food"],
                respawn_timer=arrays[f"{p}.respawn_timer"],
                reward_prev_length=arrays[f"{p}.reward_prev_length"],
                frame=int(m["frame"]),
                boosted_this_step=arrays[f"{p}.boosted_this_step"],
                last_reward=arrays[f"{p}.last_reward"],
                last_done=arrays[f"{p}.last_done"],
                last_death_cause=arrays[f"{p}.last_death_cause"],
                last_kills=arrays[f"{p}.last_kills"],
                last_food_ate=arrays[f"{p}.last_food_ate"],
                last_ambient_food_ate=arrays[f"{p}.last_ambient_food_ate"],
                last_corpse_food_ate=arrays[f"{p}.last_corpse_food_ate"],
                last_transition_valid=arrays[f"{p}.last_transition_valid"],
                food_cells=arrays[f"{p}.food_cells"],
                corpse_food=arrays[f"{p}.corpse_food"],
                vec_tensors=vec_dict,
            )
        )
    return roots


def _validate_envelope(
    meta: Mapping[str, Any],
    *,
    study: str,
    namespace: str,
    replicate_index: int,
    stage: str,
    fingerprint: str,
    path: Path,
) -> None:
    if meta.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Foreign checkpoint schema in {path}: {meta.get('schema')!r}")
    if (
        meta.get("study") != study
        or meta.get("namespace") != namespace
        or int(meta.get("replicate_index", -1)) != int(replicate_index)
        or meta.get("stage") != stage
    ):
        raise ValueError(
            f"Refusing foreign checkpoint {path}: got "
            f"(study={meta.get('study')!r}, namespace={meta.get('namespace')!r}, "
            f"rep={meta.get('replicate_index')!r}, stage={meta.get('stage')!r}), expected "
            f"({study!r}, {namespace!r}, {replicate_index}, {stage!r})"
        )
    if meta.get("config_fingerprint") != fingerprint:
        raise ValueError(
            f"Refusing mismatched config_fingerprint in checkpoint {path}: "
            f"got {meta.get('config_fingerprint')!r}, expected {fingerprint!r}"
        )


class ReplicateCheckpointManager:
    """Atomic immutable stage/epoch checkpoint writer & verified resume loader for one replicate."""

    def __init__(
        self,
        *,
        study: str,
        namespace: str,
        replicate_index: int,
        export_root: Path | None,
        resume_roots: Sequence[Path] = (),
        max_export_bytes: int = 128 * 1024**2,
    ) -> None:
        self.study = str(study)
        self.namespace = str(namespace)
        self.replicate_index = int(replicate_index)
        self.rep_subdir = f"rep{self.replicate_index:04d}"
        self.export_dir = (export_root / self.rep_subdir) if export_root is not None else None
        search: list[Path] = []
        for root in resume_roots:
            cand = root / self.rep_subdir
            if cand not in search:
                search.append(cand)
        if self.export_dir is not None and self.export_dir not in search:
            search.append(self.export_dir)
        self.search_dirs = tuple(search)
        self.max_export_bytes = int(max_export_bytes)
        self.exported_files: list[dict[str, Any]] = []
        self.resumed_stages: list[str] = []

    def _find_verified_file(self, filename: str) -> tuple[Path, dict[str, str | int]] | None:
        for d in self.search_dirs:
            candidate = d / filename
            marker = Path(f"{candidate}.complete")
            if not marker.is_file():
                continue
            verified = verify_immutable(candidate)
            return candidate, verified
        return None

    def _publish_bytes(self, filename: str, stage: str, raw: bytes) -> dict[str, Any]:
        if self.export_dir is None:
            return {"stage": stage, "file": filename, "exported": False, "size_bytes": len(raw)}
        if len(raw) > self.max_export_bytes:
            raise ValueError(
                f"Stage checkpoint {filename} exceeds max_export_bytes: "
                f"{len(raw)} > {self.max_export_bytes}"
            )
        target = self.export_dir / filename
        if Path(f"{target}.complete").is_file():
            verified = verify_immutable(target)
            rec = {
                "stage": stage,
                "file": filename,
                "path": str(target),
                "sha256": str(verified["sha256"]),
                "size_bytes": int(verified["size"]),
                "already_present": True,
            }
            return rec
        # Clean up any incomplete orphan partial file (without .complete) from an interrupted run.
        if not Path(f"{target}.complete").exists():
            Path(f"{target}.sha256").unlink(missing_ok=True)
            target.unlink(missing_ok=True)
        pub = publish_immutable(target, raw)
        rec = {
            "stage": stage,
            "file": filename,
            "path": str(target),
            "sha256": str(pub["sha256"]),
            "size_bytes": int(pub["size"]),
        }
        self.exported_files.append(rec)
        return rec

    def save_harvest(
        self,
        stage: str,
        part: HarvestedWorldPartition,
        *,
        fingerprint: str,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        filename = f"{stage}.ckpt"
        arrays: dict[str, np.ndarray] = {}
        ds_rows = _encode_dataset("ds", part.dataset, arrays)
        rds_rows = _encode_dataset("rds", part.roots_dataset, arrays)
        roots_meta = _encode_roots(part.roots, arrays)
        meta = {
            "schema": CHECKPOINT_SCHEMA,
            "study": self.study,
            "namespace": self.namespace,
            "replicate_index": self.replicate_index,
            "stage": stage,
            "config_fingerprint": fingerprint,
            "partition": part.partition,
            "world_ids": [int(x) for x in part.world_ids],
            "seeds": [int(x) for x in part.seeds],
            "dataset_num_rows": ds_rows,
            "roots_dataset_num_rows": rds_rows,
            "roots_meta": roots_meta,
            "telemetry": part.telemetry,
            "extra_meta": dict(extra_meta or {}),
        }
        return self._publish_bytes(filename, stage, _pack_container(meta, arrays))

    def try_load_harvest(
        self,
        stage: str,
        *,
        fingerprint: str,
        required_extra: Mapping[str, Any] | None = None,
    ) -> HarvestedWorldPartition | None:
        found = self._find_verified_file(f"{stage}.ckpt")
        if found is None:
            return None
        path, _ = found
        meta, arrays = _unpack_container(path.read_bytes())
        _validate_envelope(
            meta,
            study=self.study,
            namespace=self.namespace,
            replicate_index=self.replicate_index,
            stage=stage,
            fingerprint=fingerprint,
            path=path,
        )
        if required_extra:
            extra = meta.get("extra_meta", {})
            for k, v in required_extra.items():
                if extra.get(k) != v:
                    raise ValueError(
                        f"Checkpoint {path} failed anti-leakage metadata check on {k!r}: "
                        f"got {extra.get(k)!r}, expected {v!r}"
                    )
        part = HarvestedWorldPartition(
            partition=str(meta["partition"]),
            world_ids=[int(x) for x in meta["world_ids"]],
            seeds=[int(x) for x in meta["seeds"]],
            dataset=_decode_dataset("ds", int(meta["dataset_num_rows"]), arrays),
            roots=_decode_roots(meta["roots_meta"], arrays),
            roots_dataset=_decode_dataset("rds", int(meta["roots_dataset_num_rows"]), arrays),
            telemetry=dict(meta["telemetry"]),
        )
        self.resumed_stages.append(stage)
        return part

    def save_branch(
        self,
        stage: str,
        branched: BranchedRootsTable,
        *,
        fingerprint: str,
    ) -> dict[str, Any]:
        filename = f"{stage}.ckpt"
        arrays = {
            "legal_mask": np.asarray(branched.legal_mask, dtype=bool),
            "b8_action": np.asarray(branched.b8_action, dtype=np.int64),
            "s_sel": np.asarray(branched.s_sel, dtype=np.float64),
            "m_sel": np.asarray(branched.m_sel, dtype=np.float64),
            "q_sel": np.asarray(branched.q_sel, dtype=np.float64),
            "oracle_a_star": np.asarray(branched.oracle_a_star, dtype=np.int64),
            "s_score": np.asarray(branched.s_score, dtype=np.float64),
            "m_score": np.asarray(branched.m_score, dtype=np.float64),
            "q_score": np.asarray(branched.q_score, dtype=np.float64),
        }
        meta = {
            "schema": CHECKPOINT_SCHEMA,
            "study": self.study,
            "namespace": self.namespace,
            "replicate_index": self.replicate_index,
            "stage": stage,
            "config_fingerprint": fingerprint,
            "num_roots": int(branched.num_roots),
            "horizon": int(branched.horizon),
            "telemetry": branched.telemetry,
        }
        return self._publish_bytes(filename, stage, _pack_container(meta, arrays))

    def try_load_branch(self, stage: str, *, fingerprint: str) -> BranchedRootsTable | None:
        found = self._find_verified_file(f"{stage}.ckpt")
        if found is None:
            return None
        path, _ = found
        meta, arrays = _unpack_container(path.read_bytes())
        _validate_envelope(
            meta,
            study=self.study,
            namespace=self.namespace,
            replicate_index=self.replicate_index,
            stage=stage,
            fingerprint=fingerprint,
            path=path,
        )
        table = BranchedRootsTable(
            num_roots=int(meta["num_roots"]),
            horizon=int(meta["horizon"]),
            legal_mask=arrays["legal_mask"],
            b8_action=arrays["b8_action"],
            s_sel=arrays["s_sel"],
            m_sel=arrays["m_sel"],
            q_sel=arrays["q_sel"],
            oracle_a_star=arrays["oracle_a_star"],
            s_score=arrays["s_score"],
            m_score=arrays["m_score"],
            q_score=arrays["q_score"],
            telemetry=dict(meta["telemetry"]),
        )
        self.resumed_stages.append(stage)
        return table

    def save_critic_epoch(
        self,
        *,
        epoch: int,
        fingerprint: str,
        critic: PrivilegedFactoredCritic,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        np_rng_state: Mapping[str, Any],
        torch_cpu_rng_state: torch.Tensor,
        best_epoch: int,
        best_score: float,
        best_state: Mapping[str, torch.Tensor],
        epoch_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        stage = f"critic-epoch-{int(epoch):02d}"
        filename = f"{stage}.ckpt"
        best_equals_model = int(best_epoch) == int(epoch)
        model_cpu = {k: v.detach().cpu().float() for k, v in critic.state_dict().items()}
        pt_payload: dict[str, Any] = {
            "model_state": model_cpu,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "torch_cpu_rng_state": torch_cpu_rng_state.detach().cpu(),
        }
        if not best_equals_model:
            pt_payload["best_state"] = {k: v.detach().cpu().float() for k, v in best_state.items()}
        buf = io.BytesIO()
        torch.save(pt_payload, buf)
        torch_blob = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        meta = {
            "schema": CHECKPOINT_SCHEMA,
            "study": self.study,
            "namespace": self.namespace,
            "replicate_index": self.replicate_index,
            "stage": stage,
            "config_fingerprint": fingerprint,
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "best_equals_model": best_equals_model,
            "epoch_history": [dict(x) for x in epoch_history],
            "np_rng_state": json.loads(json.dumps(dict(np_rng_state))),
        }
        return self._publish_bytes(
            filename, stage, _pack_container(meta, {"torch_blob": torch_blob})
        )

    def try_load_latest_critic_epoch(
        self,
        *,
        max_epochs: int,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        for epoch in range(int(max_epochs), 0, -1):
            stage = f"critic-epoch-{epoch:02d}"
            found = self._find_verified_file(f"{stage}.ckpt")
            if found is None:
                continue
            path, _ = found
            meta, arrays = _unpack_container(path.read_bytes())
            _validate_envelope(
                meta,
                study=self.study,
                namespace=self.namespace,
                replicate_index=self.replicate_index,
                stage=stage,
                fingerprint=fingerprint,
                path=path,
            )
            raw_pt = arrays["torch_blob"].tobytes()
            pt_state = torch.load(io.BytesIO(raw_pt), map_location="cpu", weights_only=True)
            model_state = pt_state["model_state"]
            best_state = (
                {k: v.clone() for k, v in model_state.items()}
                if bool(meta.get("best_equals_model", False))
                else pt_state["best_state"]
            )
            self.resumed_stages.append(stage)
            return {
                "epoch": int(meta["epoch"]),
                "best_epoch": int(meta["best_epoch"]),
                "best_score": float(meta["best_score"]),
                "epoch_history": list(meta["epoch_history"]),
                "np_rng_state": dict(meta["np_rng_state"]),
                "model_state": model_state,
                "best_state": best_state,
                "optimizer_state": pt_state["optimizer_state"],
                "scheduler_state": pt_state["scheduler_state"],
                "torch_cpu_rng_state": pt_state["torch_cpu_rng_state"],
            }
        return None

    def save_frozen_critic(
        self,
        *,
        fingerprint: str,
        critic: PrivilegedFactoredCritic,
        baseline: LengthHorizonBaseline,
        train_summary: Mapping[str, Any],
        frozen_critic_sha256: str,
        fit_telemetry: Mapping[str, Any],
        val_telemetry: Mapping[str, Any],
        fit_branch_telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        stage = "critic-frozen"
        filename = f"{stage}.ckpt"
        arrays: dict[str, np.ndarray] = {
            "baseline.bin_hazard_logits": np.asarray(baseline.bin_hazard_logits, dtype=np.float64),
            "baseline.bin_forage_raw": np.asarray(baseline.bin_forage_raw, dtype=np.float64),
            "baseline.ridge_weights": np.asarray(baseline.ridge_weights, dtype=np.float64),
        }
        sd = critic.state_dict()
        state_keys = sorted(sd.keys())
        for k in state_keys:
            arrays[f"critic.{k}"] = sd[k].detach().cpu().float().contiguous().numpy()
        meta = {
            "schema": CHECKPOINT_SCHEMA,
            "study": self.study,
            "namespace": self.namespace,
            "replicate_index": self.replicate_index,
            "stage": stage,
            "config_fingerprint": fingerprint,
            "state_keys": state_keys,
            "baseline_ridge_alpha": float(baseline.ridge_alpha),
            "frozen_critic_sha256": str(frozen_critic_sha256),
            "train_summary": dict(train_summary),
            "fit_telemetry": dict(fit_telemetry),
            "val_telemetry": dict(val_telemetry),
            "fit_branch_telemetry": dict(fit_branch_telemetry),
        }
        return self._publish_bytes(filename, stage, _pack_container(meta, arrays))

    def try_load_frozen_critic(
        self,
        *,
        fingerprint: str,
        device: torch.device,
    ) -> dict[str, Any] | None:
        stage = "critic-frozen"
        found = self._find_verified_file(f"{stage}.ckpt")
        if found is None:
            return None
        path, _ = found
        meta, arrays = _unpack_container(path.read_bytes())
        _validate_envelope(
            meta,
            study=self.study,
            namespace=self.namespace,
            replicate_index=self.replicate_index,
            stage=stage,
            fingerprint=fingerprint,
            path=path,
        )
        critic = PrivilegedFactoredCritic()
        sd = {str(k): torch.from_numpy(arrays[f"critic.{k}"]) for k in meta["state_keys"]}
        critic.load_state_dict(sd)
        critic = critic.to(device)
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
        observed_sha = critic.state_dict_sha256()
        expected_sha = str(meta["frozen_critic_sha256"])
        if observed_sha != expected_sha:
            raise ValueError(
                f"Frozen critic SHA256 mismatch in {path}: "
                f"got {observed_sha}, expected {expected_sha}"
            )
        baseline = LengthHorizonBaseline(
            bin_hazard_logits=arrays["baseline.bin_hazard_logits"],
            bin_forage_raw=arrays["baseline.bin_forage_raw"],
            ridge_weights=arrays["baseline.ridge_weights"],
            ridge_alpha=float(meta["baseline_ridge_alpha"]),
        )
        self.resumed_stages.append(stage)
        return {
            "critic": critic,
            "baseline": baseline,
            "frozen_critic_sha256": observed_sha,
            "train_summary": dict(meta["train_summary"]),
            "fit_telemetry": dict(meta["fit_telemetry"]),
            "val_telemetry": dict(meta["val_telemetry"]),
            "fit_branch_telemetry": dict(meta["fit_branch_telemetry"]),
        }
