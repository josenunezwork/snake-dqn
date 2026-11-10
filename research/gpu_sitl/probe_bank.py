"""Pre-training probe-bank harvester and every-200-update tripwire evaluator for Step B1b.

Harvests ``t34-gpu-sitl-probe32-20261109-v1`` (`N = 32` Watch6-v2 worlds, ``H = 5,000``)
once before training from frozen ``bare b8-s3`` trajectories using the Stage 1B
5-suffix certification rule (``suffix 0`` + independent scoring ``suffixes 4..7``,
``n_diff == 0``):
- Decisive rescue roots: ``bare`` fails ``H = 32`` openness under ``suffix 0`` AND
  all ``suffixes 4..7`` while ``open-stop-h32-nb`` (`K = 1`) chosen action ``a*(s)``
  succeeds under ``suffix 0`` AND all ``suffixes 4..7`` (``n_diff == 0``).
- Stratified ordinary roots: ``210`` roots/world (`6,720` total across ``32`` worlds,
  stratified across length deciles ``D1..D10``) where ``bare`` is certified safe
  under ``suffix 0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from research.devicesim_v2_parity.harness import (
    build_product_world_bundle,
    watch6_v2_config,
)
from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.ego_device import build_observation
from research.gpu_sim_spike.v8_device import v8_decide as gpu_v8_decide
from research.gpu_sitl.open_stop_gpu import (
    _ENV_TENSOR_FIELDS,
    _VECTOR_RUNTIME_ENV_FIELDS,
    FastDeviceVectorRuntime,
    _maybe_cpu_fast_flood,
    assert_sim_and_policy_on_device,
    batched_open_stop_h32,
    concat_sims_and_policies,
    ensure_cuda_initialized,
    move_device_policy_to_device,
    normalize_device,
    verify_cuda_native_extensions,
)
from research.gpu_sitl.preflight import (
    MATURE_EXTENSION_SPECS,
    build_extension_worlds,
    resolve_product_checkpoints,
    snapshot_worlds_to_device,
)
from research.gpu_sitl.seed_check import (
    PROBE32_NAMESPACE,
    PROBE32_WORLD_COUNT,
    SHAKEOUT_PROBE_NAMESPACE,
    audit_b1b_seeds,
    probe32_seeds,
    shakeout_probe_seeds,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from research.watch_replay.open_stop import (
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
    OPEN_STOP_V8_LAMBDA,
)
from src.model.ego2s_network import Ego2sNet
from src.simd_env.grid_sim import GridBatchSim
from web.backend.served_checkpoint import B8_S3_SHA256, FRP3_S12_SHA256

PROBE32_HORIZON = 5000
PROBE32_ORDINARY_PER_WORLD = 210
PROBE32_DECILE_COUNT = 10
PROBE32_SCORING_SUFFIXES: tuple[int, ...] = (4, 5, 6, 7)

TRIPWIRE_MAX_DECILE_ORDINARY_DISAGREEMENT = 0.050
TRIPWIRE_MAX_BOOST_SHARE_RATIO = 1.50
TRIPWIRE_MIN_DECISIVE_RESCUE_AGREEMENT = 0.700


def sha256_file(path: Path | str) -> str:
    """Compute hex SHA256 digest of ``path``."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_atomic_with_sha256(
    path: Path,
    write_fn: Any,
    *,
    write_complete_marker: bool = True,
) -> str:
    """Write ``path`` via temporary file rename, then write ``.sha256`` and ``.complete``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    write_fn(tmp_path)
    tmp_path.replace(path)
    digest = sha256_file(path)
    sha_path = Path(f"{path}.sha256")
    sha_tmp = sha_path.with_name(f".{sha_path.name}.tmp")
    sha_tmp.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    sha_tmp.replace(sha_path)
    if write_complete_marker:
        comp_path = Path(f"{path}.complete")
        comp_tmp = comp_path.with_name(f".{comp_path.name}.tmp")
        comp_tmp.write_text(f"{digest}\n", encoding="utf-8")
        comp_tmp.replace(comp_path)
    return digest


def verify_file_sha256(path: Path | str, *, require_complete: bool = True) -> str:
    """Verify ``path`` against its ``<path>.sha256`` (and ``<path>.complete``) sidecar."""
    p = Path(path)
    sha_p = Path(f"{p}.sha256")
    comp_p = Path(f"{p}.complete")
    if not p.is_file():
        raise FileNotFoundError(f"Missing file: {p}")
    if not sha_p.is_file():
        raise FileNotFoundError(f"Missing SHA256 sidecar: {sha_p}")
    if require_complete and not comp_p.is_file():
        raise FileNotFoundError(f"Missing completion marker: {comp_p}")
    expected = sha_p.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(p)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {p}: expected {expected}, got {actual}")
    return actual


def build_watch6_ct_device_bundle(
    seeds: Sequence[int],
    b8_path: Path | str,
    frp3_path: Path | str,
    *,
    device: str | torch.device = "cpu",
    hero_respawn: bool = True,
    max_capacity: int = 1600,
) -> tuple[CTDeviceSim, DevicePolicy]:
    """Initialize a batch of Watch6-v2 ``CTDeviceSim`` worlds and ``DevicePolicy``."""
    dev = normalize_device(device)
    if dev.type == "cuda":
        verify_cuda_native_extensions(dev)
    cfg_batch = watch6_v2_config(len(seeds), max_capacity=max_capacity)
    batch_ref_sim = GridBatchSim(
        cfg_batch,
        seeds=[int(s) for s in seeds],
        train_mode=False,
        allow_respawn=True,
    )
    bundle = build_product_world_bundle(batch_ref_sim, Path(b8_path), Path(frp3_path))
    dev_policy = DevicePolicy.from_world(bundle, device=dev, prepared=False)
    if dev_policy.vector_runtime is not None:
        dev_policy.vector_runtime.__class__ = FastDeviceVectorRuntime
    dev_sim = CTDeviceSim.from_reference(
        batch_ref_sim,
        device=dev,
        hero_respawn=hero_respawn,
        prepared=False,
    )
    r2_cache.enable(dev_sim)
    assert_sim_and_policy_on_device(dev_sim, dev_policy, dev)
    return dev_sim, dev_policy


@torch.inference_mode()
def _extract_hero_root_observations_impl(
    sim: DeviceSim,
    *,
    cpu_fast_flood: bool = True,
) -> dict[str, torch.Tensor]:
    e_count = int(sim.E)
    dev = sim.device
    if dev.type == "cuda":
        verify_cuda_native_extensions(dev)
    hero_rows = torch.stack(
        (
            torch.arange(e_count, device=dev, dtype=torch.long),
            torch.zeros(e_count, device=dev, dtype=torch.long),
        ),
        dim=1,
    )
    with _maybe_cpu_fast_flood(bool(cpu_fast_flood and dev.type == "cpu")):
        advisory = sim.advisory_mask()[:, 0]
        fallback = ~advisory.any(dim=1, keepdim=True)
        fallback_mask = torch.tensor(
            [[True, True, True, False, False, False]],
            dtype=torch.bool,
            device=dev,
        )
        legal_mask = torch.where(fallback, fallback_mask.expand_as(advisory), advisory)
        v8_feats = (
            gpu_v8_decide(
                sim,
                hero_rows,
                torch.zeros((e_count, 6), device=dev),
                advisory,
                float(OPEN_STOP_V8_LAMBDA),
                features=True,
            )
            .feature_planes()
            .reshape(e_count, 30)
        )
        local_u8, global_u8, scalars_f32 = build_observation(sim, hero_rows, v8_feats)
    return {
        "local_u8": local_u8.to(dtype=torch.uint8),
        "global_u8": global_u8.to(dtype=torch.uint8),
        "scalars_f32": scalars_f32.to(dtype=torch.float32),
        "advisory_mask": advisory.to(dtype=torch.bool),
        "legal_mask": legal_mask.to(dtype=torch.bool),
        "hero_alive": sim.alive[:, 0].clone(),
        "hero_length": sim.length[:, 0].to(dtype=torch.long).clone(),
        "decision_frame": sim.frame.to(dtype=torch.long).clone(),
    }


def extract_hero_root_observations(
    sim: DeviceSim,
    *,
    cpu_fast_flood: bool = True,
) -> dict[str, torch.Tensor]:
    """Extract ``ego2s-bv8`` hero observations and legal masks for all worlds in ``sim``."""
    out = _extract_hero_root_observations_impl(sim, cpu_fast_flood=cpu_fast_flood)
    with torch.inference_mode(False):
        return {k: v.clone() for k, v in out.items()}


@torch.inference_mode()
def slice_single_row_cpu_state(
    batch_sim: DeviceSim,
    batch_pol: DevicePolicy,
    row_idx: int,
) -> dict[str, Any]:
    """Slice a single row ``[row_idx : row_idx + 1]`` directly to CPU without full-batch cloning."""
    idx = int(row_idx)
    sim_tensors = {
        name: getattr(batch_sim, name)[idx : idx + 1].detach().cpu().clone()
        for name in _ENV_TENSOR_FIELDS
        if isinstance(getattr(batch_sim, name, None), torch.Tensor)
    }
    rt = batch_pol.vector_runtime
    rt_tensors = {
        name: getattr(rt, name)[idx : idx + 1].detach().cpu().clone()
        for name in _VECTOR_RUNTIME_ENV_FIELDS
        if isinstance(getattr(rt, name, None), torch.Tensor)
    }
    return {
        "sim_tensors": sim_tensors,
        "rt_tensors": rt_tensors,
        "vector_prepared": bool(batch_pol.vector_prepared),
        "sim_prepared": bool(getattr(batch_sim, "_prepared", False)),
    }


@torch.inference_mode()
def reconstruct_batch_from_cpu_row_states(
    row_states: Sequence[dict[str, Any]],
    template_sim: DeviceSim,
    template_pol: DevicePolicy,
    target_device: torch.device | str,
) -> tuple[DeviceSim, DevicePolicy]:
    """Build a ``(batch_sim, batch_pol)`` on ``target_device`` from 1-row CPU state dicts."""
    if not row_states:
        raise ValueError("row_states must be non-empty")
    target_dev = normalize_device(target_device)
    total_e = len(row_states)
    out_sim = template_sim.clone()
    out_sim.device = target_dev
    out_pol = template_pol.repeat_interleave(1)
    move_device_policy_to_device(out_pol, target_dev)

    for name in _ENV_TENSOR_FIELDS:
        if all(name in rs["sim_tensors"] for rs in row_states):
            setattr(
                out_sim,
                name,
                torch.cat([rs["sim_tensors"][name] for rs in row_states], dim=0).to(target_dev),
            )
    out_sim.E = total_e
    out_sim._prepared = bool(row_states[0].get("sim_prepared", False))
    r2_cache.bump(out_sim)

    out_rt = out_pol.vector_runtime
    for name in _VECTOR_RUNTIME_ENV_FIELDS:
        if all(name in rs["rt_tensors"] for rs in row_states):
            setattr(
                out_rt,
                name,
                torch.cat([rs["rt_tensors"][name] for rs in row_states], dim=0).to(target_dev),
            )
    out_rt.E = total_e
    out_pol.vector_prepared = bool(row_states[0].get("vector_prepared", True))
    assert_sim_and_policy_on_device(out_sim, out_pol, target_dev)
    return out_sim, out_pol


def select_ordinary_buffer_slot(
    decision_frame: int,
    current_buf_len: int,
    *,
    horizon: int,
    max_buffer_per_world: int,
    existing_frames: Sequence[int] | None = None,
) -> int | None:
    """Return buffer slot index to write for an ordinary frame at ``decision_frame``,
    or ``None`` if the frame's horizon bucket is already represented in a full buffer.

    Always appends while ``current_buf_len < max_buffer_per_world`` so the candidate
    buffer fills whenever a world has at least ``max_buffer_per_world`` (or ``>= 210``)
    suffix-0-safe ordinary frames. Once full, replaces duplicate entries from the most
    over-represented horizon bucket so retained candidates span ``[0, horizon)`` by
    actual ``decision_frame`` deterministically.
    """
    cap = max(1, int(max_buffer_per_world))
    denom = max(cap, int(horizon))
    t = int(decision_frame)
    if t < 0:
        return None
    if int(current_buf_len) < cap:
        return int(current_buf_len)
    target_bucket = min(cap - 1, (t * cap) // denom)
    if existing_frames is not None and len(existing_frames) >= cap:
        slot_buckets = [min(cap - 1, (max(0, int(f)) * cap) // denom) for f in existing_frames]
        if target_bucket in slot_buckets:
            return None
        counts: dict[int, int] = {}
        for b in slot_buckets:
            counts[b] = counts.get(b, 0) + 1
        dup_buckets = [b for b, c in counts.items() if c > 1]
        if not dup_buckets:
            return int(target_bucket)
        best_dup_bucket = max(dup_buckets, key=lambda b: (counts[b], -b))
        for idx in range(len(slot_buckets) - 1, -1, -1):
            if slot_buckets[idx] == best_dup_bucket:
                return int(idx)
    if t == 0 or (t * cap) // denom > ((t - 1) * cap) // denom:
        b_max_init = min(cap - 1, ((cap - 1) * cap) // denom)
        reserved = {min(cap - 1, (b * denom + cap - 1) // cap) for b in range(b_max_init + 1)}
        if target_bucket <= b_max_init:
            return min(cap - 1, (target_bucket * denom + cap - 1) // cap)
        free_slots = [s for s in range(cap) if s not in reserved]
        if free_slots:
            free_idx = min(len(free_slots) - 1, target_bucket - (b_max_init + 1))
            return int(free_slots[free_idx])
        return int(target_bucket)
    return None


@torch.inference_mode()
def _evaluate_scoring_suffixes_on_snapshots(
    snapshots: Sequence[dict[str, Any]],
    *,
    template_sim: DeviceSim | None = None,
    template_pol: DevicePolicy | None = None,
    scoring_suffixes: Sequence[int] = PROBE32_SCORING_SUFFIXES,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    namespace: str = OPEN_STOP_CRN_NAMESPACE,
    batch_roots: int = 256,
    cpu_fast_flood: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate independent scoring ``suffixes 4..7`` on a list of root snapshots."""
    if not snapshots:
        return []
    out_records: list[dict[str, Any]] = []
    for start in range(0, len(snapshots), int(batch_roots)):
        chunk = snapshots[start : start + int(batch_roots)]
        if all("row_state" in c for c in chunk):
            if template_sim is None or template_pol is None:
                raise ValueError(
                    "template_sim and template_pol are required when snapshots use CPU row_state"
                )
            batch_sim, batch_pol = reconstruct_batch_from_cpu_row_states(
                [c["row_state"] for c in chunk],
                template_sim,
                template_pol,
                template_sim.device,
            )
        else:
            sims = [c["sim_1"] for c in chunk]
            pols = [c["pol_1"] for c in chunk]
            batch_sim, batch_pol = concat_sims_and_policies(sims, pols)
        w_seeds = [int(c["world_seed"]) for c in chunk]
        decs = [int(c["decision_frame"]) for c in chunk]

        safe_by_suffix: dict[int, torch.Tensor] = {}
        for suf in scoring_suffixes:
            res_suf = batched_open_stop_h32(
                batch_sim,
                batch_pol,
                world_seeds=w_seeds,
                decisions=decs,
                suffix_id=int(suf),
                branch_horizon=int(branch_horizon),
                namespace=namespace,
                compact_dead=True,
                enable_r2=True,
                cpu_fast_flood=cpu_fast_flood,
                return_host_records=False,
            )
            safe_by_suffix[int(suf)] = res_suf["safe_mask"].detach().cpu()

        for i, item in enumerate(chunk):
            bare_a = int(item["bare_action"])
            a_star = int(item["a_star_action"])
            safe_s0 = item["safe_mask_s0"].clone().cpu()
            consensus_safe = safe_s0.clone()
            n_diff_rescue = 0
            n_fail_ordinary = 0
            per_suffix_safe: dict[int, list[bool]] = {}
            for suf in scoring_suffixes:
                s_mask = safe_by_suffix[int(suf)][i]
                per_suffix_safe[int(suf)] = [bool(x) for x in s_mask.tolist()]
                consensus_safe &= s_mask
                bare_ok = bool(s_mask[bare_a].item())
                star_ok = bool(s_mask[a_star].item()) if 0 <= a_star < 6 else False
                if bare_ok or not star_ok:
                    n_diff_rescue += 1
                if not bare_ok:
                    n_fail_ordinary += 1

            rec = dict(item)
            rec.pop("sim_1", None)
            rec.pop("pol_1", None)
            rec.pop("row_state", None)
            rec["safe_by_scoring_suffix"] = per_suffix_safe
            rec["consensus_safe_mask"] = consensus_safe
            rec["n_diff_rescue"] = int(n_diff_rescue)
            rec["n_fail_ordinary"] = int(n_fail_ordinary)
            out_records.append(rec)
    return out_records


@torch.inference_mode()
def harvest_probe32_bank(
    output_path: Path | str,
    *,
    b8_checkpoint: Path | str | None = None,
    opponent_checkpoint: Path | str | None = None,
    device: str | torch.device = "cpu",
    num_worlds: int = PROBE32_WORLD_COUNT,
    horizon: int = PROBE32_HORIZON,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    ordinary_per_world: int = PROBE32_ORDINARY_PER_WORLD,
    scoring_suffixes: Sequence[int] = PROBE32_SCORING_SUFFIXES,
    harvest_batch_roots: int = 256,
    checkpoint_every_seconds: float = 900.0,
    resume_ckpt_path: Path | str | None = None,
    cpu_fast_flood: bool = True,
    dry_run: bool = False,
    shakeout: bool = False,
    namespace_override: str | None = None,
    seed_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Harvest ``t34-gpu-sitl-probe32-20261109-v1`` before training and write ``.sha256``."""
    out_path = Path(output_path)
    sha_path = Path(f"{out_path}.sha256")
    comp_path = Path(f"{out_path}.complete")
    if out_path.exists() or sha_path.exists() or comp_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing probe32 bank or sidecar at {out_path}"
        )

    t0 = time.perf_counter()
    dev = ensure_cuda_initialized(device)
    if dev.type == "cuda":
        verify_cuda_native_extensions(dev)

    is_shakeout = bool(shakeout or namespace_override is not None or seed_override is not None)
    eff_dry_run = bool(dry_run or is_shakeout)
    active_namespace = str(
        namespace_override
        if namespace_override is not None
        else (SHAKEOUT_PROBE_NAMESPACE if is_shakeout else PROBE32_NAMESPACE)
    )
    if is_shakeout and active_namespace == PROBE32_NAMESPACE:
        raise ValueError(
            f"Shake-out probe harvest must never use registered namespace {PROBE32_NAMESPACE!r}"
        )

    if not eff_dry_run:
        if (
            int(num_worlds) != PROBE32_WORLD_COUNT
            or int(horizon) != PROBE32_HORIZON
            or int(branch_horizon) != OPEN_STOP_BRANCH_HORIZON
            or int(ordinary_per_world) != PROBE32_ORDINARY_PER_WORLD
            or tuple(int(x) for x in scoring_suffixes) != PROBE32_SCORING_SUFFIXES
        ):
            raise ValueError(
                "Non-dry-run probe32 harvest must use exact registered parameters: "
                f"N={PROBE32_WORLD_COUNT}, H={PROBE32_HORIZON}, "
                f"branch_horizon={OPEN_STOP_BRANCH_HORIZON}, "
                f"ordinary_per_world={PROBE32_ORDINARY_PER_WORLD}, "
                f"scoring_suffixes={PROBE32_SCORING_SUFFIXES}"
            )

    seed_audit = audit_b1b_seeds(scan_code_roots=[])
    if not bool(seed_audit["ok"]):
        raise RuntimeError(f"Step B1b seed audit failed: {seed_audit}")

    if b8_checkpoint is None or opponent_checkpoint is None:
        default_b8, default_opp = resolve_product_checkpoints()
        b8_checkpoint = b8_checkpoint or default_b8
        opponent_checkpoint = opponent_checkpoint or default_opp
    b8_path = Path(b8_checkpoint)
    opp_path = Path(opponent_checkpoint)

    if seed_override is not None:
        seeds = [int(s) for s in seed_override]
    elif is_shakeout:
        seeds = shakeout_probe_seeds(int(num_worlds))
    else:
        seeds = probe32_seeds(int(num_worlds))
    sim, policy = build_watch6_ct_device_bundle(
        seeds,
        b8_path,
        opp_path,
        device=dev,
        hero_respawn=True,
    )

    # First pass: step all N worlds for H ticks under frozen bare b8-s3, batching alive roots
    # up to harvest_batch_roots for suffix-0 open-stop-h32 evaluation.
    pending_rows: list[dict[str, Any]] = []
    candidate_rescue_snapshots: list[dict[str, Any]] = []
    s0_ordinary_by_world: dict[int, list[dict[str, Any]]] = {int(s): [] for s in seeds}
    s0_ordinary_seen_by_world: dict[int, int] = {int(s): 0 for s in seeds}
    oversample_per_world = int(ordinary_per_world) + max(10, int(ordinary_per_world) // 5)
    max_s0_buffer_per_world = int(oversample_per_world)
    total_alive_hero_decisions = 0
    start_tick = 0

    default_ckpt_path = Path(f"{out_path}.ckpt.pt")
    active_resume_ckpt = (
        Path(resume_ckpt_path) if resume_ckpt_path is not None else default_ckpt_path
    )
    if active_resume_ckpt.is_file() and Path(f"{active_resume_ckpt}.sha256").is_file():
        verify_file_sha256(
            active_resume_ckpt,
            require_complete=Path(f"{active_resume_ckpt}.complete").is_file(),
        )
        ckpt_payload = torch.load(active_resume_ckpt, map_location="cpu", weights_only=False)
        if ckpt_payload.get("namespace") != active_namespace:
            raise ValueError(
                f"Probe harvest checkpoint namespace mismatch: {ckpt_payload.get('namespace')!r}"
            )
        if [int(s) for s in ckpt_payload.get("world_seeds", [])] != [int(s) for s in seeds]:
            raise ValueError("Probe harvest checkpoint world_seeds mismatch")
        start_tick = int(ckpt_payload["completed_tick"])
        total_alive_hero_decisions = int(ckpt_payload["total_alive_hero_decisions"])
        candidate_rescue_snapshots = list(ckpt_payload["candidate_rescue_snapshots"])
        s0_ordinary_by_world = {
            int(k): list(v) for k, v in ckpt_payload["s0_ordinary_by_world"].items()
        }
        s0_ordinary_seen_by_world = {
            int(k): int(v) for k, v in ckpt_payload["s0_ordinary_seen_by_world"].items()
        }
        sim_state = ckpt_payload["sim_state"]
        for name in _ENV_TENSOR_FIELDS:
            if name in sim_state["sim"] and isinstance(getattr(sim, name, None), torch.Tensor):
                setattr(sim, name, sim_state["sim"][name].to(dev).clone())
        sim.E = int(sim_state["sim"]["E"])
        sim._prepared = bool(sim_state["sim"].get("_prepared", False))
        r2_cache.bump(sim)
        rt = policy.vector_runtime
        for name in _VECTOR_RUNTIME_ENV_FIELDS:
            if name in sim_state["rt"] and isinstance(getattr(rt, name, None), torch.Tensor):
                setattr(rt, name, sim_state["rt"][name].to(dev).clone())
        policy.vector_prepared = bool(sim_state["rt"]["vector_prepared"])

    def _save_harvest_checkpoint(completed_tick: int) -> str:
        sim_state = {
            "sim": {
                "E": int(sim.E),
                "_prepared": bool(getattr(sim, "_prepared", False)),
                **{
                    name: getattr(sim, name).detach().cpu().clone()
                    for name in _ENV_TENSOR_FIELDS
                    if isinstance(getattr(sim, name, None), torch.Tensor)
                },
            },
            "rt": {
                "E": int(getattr(policy.vector_runtime, "E", sim.E)),
                "vector_prepared": bool(policy.vector_prepared),
                **{
                    name: getattr(policy.vector_runtime, name).detach().cpu().clone()
                    for name in _VECTOR_RUNTIME_ENV_FIELDS
                    if isinstance(getattr(policy.vector_runtime, name, None), torch.Tensor)
                },
            },
        }
        ckpt_obj = {
            "schema": "t34-gpu-sitl-probe32-harvest-ckpt/v1",
            "namespace": active_namespace,
            "world_seeds": [int(s) for s in seeds],
            "completed_tick": int(completed_tick),
            "horizon": int(horizon),
            "total_alive_hero_decisions": int(total_alive_hero_decisions),
            "candidate_rescue_snapshots": candidate_rescue_snapshots,
            "s0_ordinary_by_world": s0_ordinary_by_world,
            "s0_ordinary_seen_by_world": s0_ordinary_seen_by_world,
            "sim_state": sim_state,
        }
        tick_ckpt_path = (
            out_path.parent / "checkpoints" / f"{out_path.name}.t{int(completed_tick):04d}.ckpt.pt"
        )
        if not tick_ckpt_path.exists() and not Path(f"{tick_ckpt_path}.complete").exists():
            write_atomic_with_sha256(
                tick_ckpt_path,
                lambda tmp: torch.save(ckpt_obj, tmp),
                write_complete_marker=True,
            )
        return write_atomic_with_sha256(
            default_ckpt_path,
            lambda tmp: torch.save(ckpt_obj, tmp),
            write_complete_marker=bool(
                int(completed_tick) >= int(horizon)
                and not Path(f"{default_ckpt_path}.complete").exists()
            ),
        )

    def _flush_pending_suffix0() -> None:
        nonlocal pending_rows
        if not pending_rows:
            return
        batch_sim, batch_pol = reconstruct_batch_from_cpu_row_states(
            [m["row_state"] for m in pending_rows],
            sim,
            policy,
            dev,
        )
        w_seeds = [int(m["world_seed"]) for m in pending_rows]
        decs = [int(m["decision_frame"]) for m in pending_rows]
        res0 = batched_open_stop_h32(
            batch_sim,
            batch_pol,
            world_seeds=w_seeds,
            decisions=decs,
            suffix_id=0,
            branch_horizon=int(branch_horizon),
            namespace=OPEN_STOP_CRN_NAMESPACE,
            compact_dead=True,
            enable_r2=True,
            cpu_fast_flood=cpu_fast_flood,
            return_host_records=False,
        )
        safe_s0 = res0["safe_mask"].detach().cpu()
        rescue_s0 = res0["rescue_mask"].detach().cpu()
        ordinary_s0 = res0["ordinary_mask"].detach().cpu()
        bare_s0 = res0["bare_actions"].detach().cpu()
        astar_s0 = res0["a_star_actions"].detach().cpu()
        q_s0 = res0["root_q"].detach().cpu()

        for idx, item in enumerate(pending_rows):
            if bool(rescue_s0[idx].item()):
                item["safe_mask_s0"] = safe_s0[idx]
                item["bare_action"] = int(bare_s0[idx].item())
                item["a_star_action"] = int(astar_s0[idx].item())
                item["root_q_b8"] = q_s0[idx]
                candidate_rescue_snapshots.append(item)
            elif bool(ordinary_s0[idx].item()):
                ws = int(item["world_seed"])
                seen_k = s0_ordinary_seen_by_world[ws]
                s0_ordinary_seen_by_world[ws] = seen_k + 1
                world_buf = s0_ordinary_by_world[ws]
                slot = select_ordinary_buffer_slot(
                    int(item["decision_frame"]),
                    len(world_buf),
                    horizon=int(horizon),
                    max_buffer_per_world=max_s0_buffer_per_world,
                    existing_frames=[int(x["decision_frame"]) for x in world_buf],
                )
                if slot is not None:
                    item["safe_mask_s0"] = safe_s0[idx]
                    item["bare_action"] = int(bare_s0[idx].item())
                    item["a_star_action"] = int(astar_s0[idx].item())
                    item["root_q_b8"] = q_s0[idx]
                    if slot == len(world_buf):
                        world_buf.append(item)
                    else:
                        world_buf[slot] = item
                else:
                    item.pop("row_state", None)
            else:
                item.pop("row_state", None)
        pending_rows = []

    last_ckpt_wall = time.perf_counter()
    for _tick in range(int(start_tick), int(horizon)):
        sim.prepare()
        policy.vector_runtime.prepare(sim)
        policy.vector_prepared = True

        obs = extract_hero_root_observations(sim, cpu_fast_flood=cpu_fast_flood)
        _, _, hero_bare_act = policy.evaluate(sim)

        alive_mask = obs["hero_alive"]
        alive_rows = alive_mask.nonzero(as_tuple=False).squeeze(1)
        n_alive = int(alive_rows.numel())
        total_alive_hero_decisions += n_alive

        if n_alive > 0:
            for r_idx in alive_rows.detach().cpu().tolist():
                r = int(r_idx)
                pending_rows.append(
                    {
                        "world_seed": int(seeds[r]),
                        "world_index": int(r),
                        "decision_frame": int(obs["decision_frame"][r].item()),
                        "hero_length": int(obs["hero_length"][r].item()),
                        "local_u8": obs["local_u8"][r].detach().cpu(),
                        "global_u8": obs["global_u8"][r].detach().cpu(),
                        "scalars_f32": obs["scalars_f32"][r].detach().cpu(),
                        "legal_mask": obs["legal_mask"][r].detach().cpu(),
                        "row_state": slice_single_row_cpu_state(sim, policy, r),
                    }
                )
            if len(pending_rows) >= int(harvest_batch_roots):
                _flush_pending_suffix0()

        # Advance frozen bare b8-s3 trajectory by 1 frame.
        step_actions = policy.actions(sim, hero_bare_act)
        sim.step(step_actions)
        policy.after_step(sim)

        now_wall = time.perf_counter()
        if (now_wall - last_ckpt_wall) >= float(checkpoint_every_seconds):
            _flush_pending_suffix0()
            _save_harvest_checkpoint(_tick + 1)
            last_ckpt_wall = time.perf_counter()

    _flush_pending_suffix0()

    for ws in seeds:
        n_s0_ord = len(s0_ordinary_by_world[int(ws)])
        if not eff_dry_run and n_s0_ord < int(ordinary_per_world):
            raise RuntimeError(
                f"World {ws} yielded only {n_s0_ord} < {ordinary_per_world} "
                "suffix-0-safe ordinary roots"
            )

    # In dry_run / shakeout mode with tiny horizon, if 0 natural candidate rescue roots were
    # encountered, include a mature rescue spec from MATURE_EXTENSION_SPECS (bound to seeds[0]
    # in active_namespace) so the run exercises rescue certification and tripwire paths.
    if eff_dry_run and not candidate_rescue_snapshots:
        ext_worlds, ext_pol_tpl, _ = build_extension_worlds(
            b8_path,
            opp_path,
            specs=[MATURE_EXTENSION_SPECS[0]],
        )
        ext_sim, ext_pol = snapshot_worlds_to_device(
            ext_worlds,
            ext_pol_tpl,
            sim_cls=CTDeviceSim,
            device=dev,
            vector_prepared=False,
        )
        ext_obs = extract_hero_root_observations(ext_sim, cpu_fast_flood=cpu_fast_flood)
        probe_rescue_seed = int(seeds[0])
        ext_res0 = batched_open_stop_h32(
            ext_sim,
            ext_pol,
            world_seeds=[probe_rescue_seed],
            decisions=[int(ext_worlds[0].sim.frame[0])],
            suffix_id=0,
            branch_horizon=int(branch_horizon),
            namespace=OPEN_STOP_CRN_NAMESPACE,
            compact_dead=True,
            enable_r2=True,
            cpu_fast_flood=cpu_fast_flood,
            return_host_records=False,
        )
        candidate_rescue_snapshots.append(
            {
                "world_seed": probe_rescue_seed,
                "world_index": 0,
                "decision_frame": int(ext_worlds[0].sim.frame[0]),
                "hero_length": int(ext_obs["hero_length"][0].item()),
                "local_u8": ext_obs["local_u8"][0].detach().cpu(),
                "global_u8": ext_obs["global_u8"][0].detach().cpu(),
                "scalars_f32": ext_obs["scalars_f32"][0].detach().cpu(),
                "legal_mask": ext_obs["legal_mask"][0].detach().cpu(),
                "safe_mask_s0": ext_res0["safe_mask"][0].detach().cpu(),
                "bare_action": int(ext_res0["bare_actions"][0].item()),
                "a_star_action": int(ext_res0["a_star_actions"][0].item()),
                "root_q_b8": ext_res0["root_q"][0].detach().cpu(),
                "row_state": slice_single_row_cpu_state(ext_sim, ext_pol, 0),
            }
        )

    # Evaluate suffixes 4..7 on all candidate rescue roots and keep n_diff == 0.
    scored_rescue = _evaluate_scoring_suffixes_on_snapshots(
        candidate_rescue_snapshots,
        template_sim=sim,
        template_pol=policy,
        scoring_suffixes=scoring_suffixes,
        branch_horizon=branch_horizon,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        batch_roots=harvest_batch_roots,
        cpu_fast_flood=cpu_fast_flood,
    )
    decisive_rescue_roots = [r for r in scored_rescue if int(r["n_diff_rescue"]) == 0]
    if eff_dry_run and not decisive_rescue_roots and scored_rescue:
        # In dry-run/shakeout with short branch_horizon, ensure at least 1 decisive root
        fallback_rec = dict(scored_rescue[0])
        fallback_rec["consensus_safe_mask"] = fallback_rec["safe_mask_s0"].clone()
        decisive_rescue_roots = [fallback_rec]
    if not eff_dry_run and not decisive_rescue_roots:
        raise RuntimeError(
            "Zero Stage 1B certified decisive rescue roots harvested on probe32 bank"
        )

    # Score buffered ordinary candidates on independent scoring suffixes 4..7 per registration
    # §3.3 item 4 (Stage 1B certification rule). Prefer fully certified roots
    # (n_fail_ordinary == 0) and backfill by ascending n_fail_ordinary when a world has fewer
    # than ordinary_per_world fully certified.
    all_ordinary_candidates: list[dict[str, Any]] = []
    for ws in seeds:
        all_ordinary_candidates.extend(s0_ordinary_by_world[int(ws)])
    scored_ordinary_all = _evaluate_scoring_suffixes_on_snapshots(
        all_ordinary_candidates,
        template_sim=sim,
        template_pol=policy,
        scoring_suffixes=scoring_suffixes,
        branch_horizon=branch_horizon,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        batch_roots=harvest_batch_roots,
        cpu_fast_flood=cpu_fast_flood,
    )
    scored_ordinary_by_world: dict[int, list[dict[str, Any]]] = {int(s): [] for s in seeds}
    for rec in scored_ordinary_all:
        scored_ordinary_by_world[int(rec["world_seed"])].append(rec)

    def _stratified_pick(
        candidates: Sequence[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        if k <= 0 or not candidates:
            return []
        if k >= len(candidates):
            return [dict(c) for c in candidates]
        idx_pick = np.linspace(0, len(candidates) - 1, k, dtype=int)
        picked_indices: list[int] = []
        seen_idx: set[int] = set()
        for raw_i in idx_pick.tolist():
            cand_i = int(raw_i)
            while cand_i in seen_idx and cand_i + 1 < len(candidates):
                cand_i += 1
            while cand_i in seen_idx and cand_i - 1 >= 0:
                cand_i -= 1
            if cand_i not in seen_idx:
                seen_idx.add(cand_i)
                picked_indices.append(cand_i)
        return [dict(candidates[idx_p]) for idx_p in sorted(picked_indices)]

    harvested_ordinary_roots: list[dict[str, Any]] = []
    selected_ordinary_by_world: dict[int, list[dict[str, Any]]] = {}
    for ws in seeds:
        world_cands = scored_ordinary_by_world[int(ws)]
        target_k = min(int(ordinary_per_world), len(world_cands))
        certified_cands = sorted(
            [r for r in world_cands if int(r["n_fail_ordinary"]) == 0],
            key=lambda x: (int(x["hero_length"]), int(x["decision_frame"])),
        )
        backfill_cands = sorted(
            [r for r in world_cands if int(r["n_fail_ordinary"]) > 0],
            key=lambda x: (
                int(x["n_fail_ordinary"]),
                int(x["hero_length"]),
                int(x["decision_frame"]),
            ),
        )
        if len(certified_cands) >= target_k:
            selected_world = _stratified_pick(certified_cands, target_k)
        else:
            selected_world = [dict(r) for r in certified_cands]
            needed = target_k - len(selected_world)
            for n_fail_val in range(1, len(scoring_suffixes) + 1):
                if needed <= 0:
                    break
                tier = [r for r in backfill_cands if int(r["n_fail_ordinary"]) == n_fail_val]
                if not tier:
                    continue
                if len(tier) <= needed:
                    selected_world.extend(dict(r) for r in tier)
                    needed -= len(tier)
                else:
                    selected_world.extend(_stratified_pick(tier, needed))
                    needed = 0
            selected_world.sort(key=lambda x: (int(x["hero_length"]), int(x["decision_frame"])))
        selected_ordinary_by_world[int(ws)] = selected_world
        harvested_ordinary_roots.extend(selected_world)

    if not harvested_ordinary_roots:
        raise RuntimeError("Zero ordinary roots harvested for probe bank")

    # Assign length deciles D1..D10 (0..9) across harvested ordinary roots sorted by length.
    harvested_ordinary_roots.sort(
        key=lambda x: (
            int(x["hero_length"]),
            int(x["world_seed"]),
            int(x["decision_frame"]),
        )
    )
    n_ord = len(harvested_ordinary_roots)
    decile_Meta: dict[str, dict[str, int]] = {}
    for pos, rec in enumerate(harvested_ordinary_roots):
        dec_idx = min(PROBE32_DECILE_COUNT - 1, (pos * PROBE32_DECILE_COUNT) // max(1, n_ord))
        rec["length_decile"] = int(dec_idx)

    for d in range(PROBE32_DECILE_COUNT):
        d_rows = [r for r in harvested_ordinary_roots if int(r["length_decile"]) == d]
        dec_key = f"D{d + 1}"
        if d_rows:
            lengths_d = [int(r["hero_length"]) for r in d_rows]
            cert_d = sum(1 for r in d_rows if int(r["n_fail_ordinary"]) == 0)
            decile_Meta[dec_key] = {
                "count": len(d_rows),
                "fully_certified_count": int(cert_d),
                "backfilled_count": int(len(d_rows) - cert_d),
                "min_length": min(lengths_d),
                "max_length": max(lengths_d),
            }
        else:
            decile_Meta[dec_key] = {
                "count": 0,
                "fully_certified_count": 0,
                "backfilled_count": 0,
                "min_length": 0,
                "max_length": 0,
            }

    per_world_ordinary_certification: dict[str, dict[str, Any]] = {}
    for ws in seeds:
        ws_int = int(ws)
        world_cands = scored_ordinary_by_world[ws_int]
        world_selected = selected_ordinary_by_world[ws_int]
        cert_cand_cnt = sum(1 for r in world_cands if int(r["n_fail_ordinary"]) == 0)
        sel_cert_cnt = sum(1 for r in world_selected if int(r["n_fail_ordinary"]) == 0)
        per_world_ordinary_certification[str(ws_int)] = {
            "world_seed": ws_int,
            "s0_ordinary_seen": int(s0_ordinary_seen_by_world[ws_int]),
            "candidates_buffered": len(world_cands),
            "candidates_fully_certified_count": int(cert_cand_cnt),
            "selected_count": len(world_selected),
            "selected_fully_certified_count": int(sel_cert_cnt),
            "selected_backfilled_count": int(len(world_selected) - sel_cert_cnt),
            "selected_n_fail_histogram": {
                str(nf): sum(1 for r in world_selected if int(r["n_fail_ordinary"]) == nf)
                for nf in range(len(scoring_suffixes) + 1)
            },
            "decile_counts": {
                f"D{d + 1}": sum(1 for r in world_selected if int(r.get("length_decile", -1)) == d)
                for d in range(PROBE32_DECILE_COUNT)
            },
            "decile_fully_certified_counts": {
                f"D{d + 1}": sum(
                    1
                    for r in world_selected
                    if int(r.get("length_decile", -1)) == d and int(r["n_fail_ordinary"]) == 0
                )
                for d in range(PROBE32_DECILE_COUNT)
            },
        }

    def _pack_root_tensors(rows: Sequence[dict[str, Any]], *, is_ordinary: bool) -> dict[str, Any]:
        count = len(rows)
        if count == 0:
            return {
                "count": 0,
                "local_u8": torch.zeros((0, 7, 31, 31), dtype=torch.uint8),
                "global_u8": torch.zeros((0, 4, 37, 37), dtype=torch.uint8),
                "scalars_f32": torch.zeros((0, 60), dtype=torch.float32),
                "legal_mask": torch.zeros((0, 6), dtype=torch.bool),
                "safe_mask_s0": torch.zeros((0, 6), dtype=torch.bool),
                "consensus_safe_mask": torch.zeros((0, 6), dtype=torch.bool),
                "bare_action": torch.zeros((0,), dtype=torch.long),
                "a_star_action": torch.zeros((0,), dtype=torch.long),
                "hero_length": torch.zeros((0,), dtype=torch.long),
                "world_seed": torch.zeros((0,), dtype=torch.long),
                "decision_frame": torch.zeros((0,), dtype=torch.long),
                "length_decile": torch.zeros((0,), dtype=torch.long),
                "n_fail_ordinary": torch.zeros((0,), dtype=torch.long),
                "n_diff_rescue": torch.zeros((0,), dtype=torch.long),
            }
        return {
            "count": int(count),
            "local_u8": torch.stack([r["local_u8"] for r in rows], dim=0).to(torch.uint8),
            "global_u8": torch.stack([r["global_u8"] for r in rows], dim=0).to(torch.uint8),
            "scalars_f32": torch.stack([r["scalars_f32"] for r in rows], dim=0).to(torch.float32),
            "legal_mask": torch.stack([r["legal_mask"] for r in rows], dim=0).to(torch.bool),
            "safe_mask_s0": torch.stack([r["safe_mask_s0"] for r in rows], dim=0).to(torch.bool),
            "consensus_safe_mask": torch.stack([r["consensus_safe_mask"] for r in rows], dim=0).to(
                torch.bool
            ),
            "bare_action": torch.tensor([int(r["bare_action"]) for r in rows], dtype=torch.long),
            "a_star_action": torch.tensor(
                [int(r["a_star_action"]) for r in rows], dtype=torch.long
            ),
            "hero_length": torch.tensor([int(r["hero_length"]) for r in rows], dtype=torch.long),
            "world_seed": torch.tensor([int(r["world_seed"]) for r in rows], dtype=torch.long),
            "decision_frame": torch.tensor(
                [int(r["decision_frame"]) for r in rows], dtype=torch.long
            ),
            "length_decile": torch.tensor(
                [int(r.get("length_decile", -1)) if is_ordinary else -1 for r in rows],
                dtype=torch.long,
            ),
            "n_fail_ordinary": torch.tensor(
                [int(r.get("n_fail_ordinary", 0)) for r in rows],
                dtype=torch.long,
            ),
            "n_diff_rescue": torch.tensor(
                [int(r.get("n_diff_rescue", 0)) for r in rows],
                dtype=torch.long,
            ),
        }

    rescue_pack = _pack_root_tensors(decisive_rescue_roots, is_ordinary=False)
    ordinary_pack = _pack_root_tensors(harvested_ordinary_roots, is_ordinary=True)
    elapsed_s = float(time.perf_counter() - t0)

    b8_ord_boost_share = (
        float((ordinary_pack["bare_action"] >= 3).float().mean().item())
        if ordinary_pack["count"] > 0
        else 0.0
    )
    rescue_worlds_covered = len({int(r["world_seed"]) for r in decisive_rescue_roots})

    ordinary_fully_certified_total = sum(
        1 for r in harvested_ordinary_roots if int(r["n_fail_ordinary"]) == 0
    )
    ordinary_backfilled_total = int(ordinary_pack["count"]) - int(ordinary_fully_certified_total)

    meta_summary: dict[str, Any] = {
        "schema": (
            "t34-gpu-sitl-shakeout-probe-bank/v1" if is_shakeout else "t34-gpu-sitl-probe32-bank/v1"
        ),
        "namespace": active_namespace,
        "dry_run": bool(eff_dry_run),
        "shakeout": bool(is_shakeout),
        "not_evidence": bool(eff_dry_run or is_shakeout),
        "evidence_class": (
            "SHAKEOUT_NOT_EVIDENCE"
            if is_shakeout
            else ("DRY_RUN_NON_EVIDENCE" if dry_run else "REGISTERED_PROBE32_BANK")
        ),
        "num_worlds": int(num_worlds),
        "world_seeds": [int(s) for s in seeds],
        "horizon": int(horizon),
        "branch_horizon": int(branch_horizon),
        "selection_suffix": 0,
        "scoring_suffixes": [int(x) for x in scoring_suffixes],
        "ordinary_per_world_target": int(ordinary_per_world),
        "total_alive_hero_decisions": int(total_alive_hero_decisions),
        "candidate_rescue_roots_s0": len(candidate_rescue_snapshots),
        "decisive_rescue_roots_count": int(rescue_pack["count"]),
        "decisive_rescue_worlds_covered": int(rescue_worlds_covered),
        "ordinary_roots_count": int(ordinary_pack["count"]),
        "ordinary_fully_certified_count": int(ordinary_fully_certified_total),
        "ordinary_backfilled_count": int(ordinary_backfilled_total),
        "per_world_ordinary_certification": per_world_ordinary_certification,
        "b8_ordinary_boost_share": float(b8_ord_boost_share),
        "length_deciles": decile_Meta,
        "b8_checkpoint_sha256": sha256_file(b8_path),
        "b8_expected_sha256": B8_S3_SHA256,
        "opponent_checkpoint_sha256": sha256_file(opp_path),
        "opponent_expected_sha256": FRP3_S12_SHA256,
        "device": str(dev),
    }

    payload = {
        "meta": dict(meta_summary),
        "rescue": rescue_pack,
        "ordinary": ordinary_pack,
    }

    if out_path.exists() or sha_path.exists() or comp_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing probe32 bank or sidecar at {out_path}"
        )
    pt_sha256 = write_atomic_with_sha256(
        out_path,
        lambda tmp: torch.save(payload, tmp),
        write_complete_marker=True,
    )

    meta_summary["elapsed_seconds"] = elapsed_s
    meta_summary["probe_bank_path"] = str(out_path)
    meta_summary["probe_bank_sha256"] = pt_sha256

    meta_path = Path(f"{out_path}.meta.json")
    meta_sha256 = write_atomic_with_sha256(
        meta_path,
        lambda tmp: tmp.write_text(
            json.dumps(meta_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        ),
        write_complete_marker=True,
    )
    meta_summary["probe_bank_meta_sha256"] = meta_sha256
    return meta_summary


def load_probe32_bank(
    path: Path | str,
    *,
    expected_namespace: str = PROBE32_NAMESPACE,
) -> dict[str, Any]:
    """Verify ``<path>.sha256`` and load the harvested probe bank."""
    digest = verify_file_sha256(path)
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "meta" not in payload:
        raise ValueError(f"Invalid probe32 bank payload in {path}")
    actual_ns = payload["meta"].get("namespace")
    if actual_ns != expected_namespace:
        raise ValueError(
            f"Probe bank namespace mismatch in {path}: expected {expected_namespace!r}, "
            f"got {actual_ns!r}"
        )
    if expected_namespace == PROBE32_NAMESPACE and payload["meta"].get("shakeout"):
        raise ValueError(
            f"Refusing to load shake-out (NOT EVIDENCE) probe bank {path} as registered probe32"
        )
    payload["verified_sha256"] = digest
    return payload


@torch.no_grad()
def evaluate_probe_bank_tripwires(
    student_net: Ego2sNet,
    probe_bank: dict[str, Any],
    *,
    update_step: int,
    check_decisive_rescue_bar: bool = False,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> dict[str, Any]:
    """Evaluate the every-200-update fit-drift & selectivity tripwires on ``probe_bank``.

    Checks (all greedy actions taken strictly over ``a in legal(s)``):
    (i) Ordinary disagreement vs ``b8-s3`` ``<= 5.0%`` in every length decile ``D1..D10``.
    (ii) Greedy boost share ``<= 1.5x`` ``b8-s3``'s probe boost share.
    (iii) When ``check_decisive_rescue_bar=True`` (at ``update == 1000``), held-out decisive
          rescue agreement ``>= 70.0%`` on the probe bank's certified decisive roots.
    """
    dev = ensure_cuda_initialized(device)
    was_training = student_net.training
    student_net.eval()
    try:
        ord_pack = probe_bank["ordinary"]
        res_pack = probe_bank["rescue"]
        n_ord = int(ord_pack["count"])
        n_res = int(res_pack["count"])

        def _predict_legal_greedy(pack: dict[str, Any]) -> tuple[torch.Tensor, bool]:
            total = int(pack["count"])
            if total == 0:
                return torch.zeros(0, dtype=torch.long), True
            acts_out: list[torch.Tensor] = []
            all_finite = True
            for start in range(0, total, int(batch_size)):
                end = min(total, start + int(batch_size))
                loc = pack["local_u8"][start:end].to(dev)
                glb = pack["global_u8"][start:end].to(dev)
                sca = pack["scalars_f32"][start:end].to(dev)
                leg = pack["legal_mask"][start:end].to(dev)
                q = student_net(loc, glb, sca)
                if not bool(torch.isfinite(q).all().item()):
                    all_finite = False
                q_masked = q.masked_fill(~leg, -torch.inf)
                acts_out.append(q_masked.argmax(dim=1).detach().cpu())
            return torch.cat(acts_out, dim=0), all_finite

        ord_student_act, ord_q_finite = _predict_legal_greedy(ord_pack)
        res_student_act, res_q_finite = _predict_legal_greedy(res_pack)

        ord_bare_act = ord_pack["bare_action"].cpu()
        ord_deciles = ord_pack["length_decile"].cpu()
        ord_disagree = ord_student_act != ord_bare_act
        overall_ord_disagree = float(ord_disagree.float().mean().item()) if n_ord > 0 else 0.0

        decile_disagreement: dict[str, float] = {}
        decile_counts: dict[str, int] = {}
        max_decile_disagreement = 0.0
        for d in range(PROBE32_DECILE_COUNT):
            d_key = f"D{d + 1}"
            mask_d = ord_deciles == d
            cnt_d = int(mask_d.sum().item())
            decile_counts[d_key] = cnt_d
            if cnt_d > 0:
                rate_d = float(ord_disagree[mask_d].float().mean().item())
            else:
                rate_d = 0.0
            decile_disagreement[d_key] = rate_d
            if rate_d > max_decile_disagreement:
                max_decile_disagreement = rate_d

        decile_tripwire_pass = bool(
            max_decile_disagreement <= TRIPWIRE_MAX_DECILE_ORDINARY_DISAGREEMENT
        )

        student_boost_share = (
            float((ord_student_act >= 3).float().mean().item()) if n_ord > 0 else 0.0
        )
        b8_boost_share = float((ord_bare_act >= 3).float().mean().item()) if n_ord > 0 else 0.0
        if b8_boost_share > 0.0:
            boost_share_ratio = student_boost_share / b8_boost_share
            boost_tripwire_pass = bool(
                student_boost_share <= TRIPWIRE_MAX_BOOST_SHARE_RATIO * b8_boost_share + 1e-12
            )
        else:
            boost_share_ratio = 0.0 if student_boost_share == 0.0 else float("inf")
            boost_tripwire_pass = bool(student_boost_share == 0.0)

        if n_res > 0:
            res_idx = torch.arange(n_res, dtype=torch.long)
            res_act_clamped = res_student_act.clamp(0, 5)
            consensus_safe = res_pack["consensus_safe_mask"].cpu()
            s0_safe = res_pack["safe_mask_s0"].cpu()
            a_star = res_pack["a_star_action"].cpu()
            in_consensus_safe = consensus_safe[res_idx, res_act_clamped]
            in_s0_safe = s0_safe[res_idx, res_act_clamped]
            eq_a_star = res_student_act == a_star
            decisive_rescue_agreement_rate = float(in_consensus_safe.float().mean().item())
            decisive_rescue_s0_safe_rate = float(in_s0_safe.float().mean().item())
            decisive_rescue_exact_a_star_rate = float(eq_a_star.float().mean().item())
        else:
            decisive_rescue_agreement_rate = 0.0
            decisive_rescue_s0_safe_rate = 0.0
            decisive_rescue_exact_a_star_rate = 0.0

        decisive_rescue_bar_pass = (
            bool(decisive_rescue_agreement_rate >= TRIPWIRE_MIN_DECISIVE_RESCUE_AGREEMENT)
            if check_decisive_rescue_bar
            else True
        )

        # Check legal-only invariant on all predicted actions
        ord_illegal = (
            int((~ord_pack["legal_mask"].cpu()[torch.arange(n_ord), ord_student_act]).sum().item())
            if n_ord > 0
            else 0
        )
        res_illegal = (
            int((~res_pack["legal_mask"].cpu()[torch.arange(n_res), res_student_act]).sum().item())
            if n_res > 0
            else 0
        )
        all_q_finite = bool(ord_q_finite and res_q_finite)
        all_pass = bool(
            decile_tripwire_pass
            and boost_tripwire_pass
            and decisive_rescue_bar_pass
            and all_q_finite
            and ord_illegal == 0
            and res_illegal == 0
        )
        return {
            "update_step": int(update_step),
            "check_decisive_rescue_bar": bool(check_decisive_rescue_bar),
            "all_pass": all_pass,
            "ordinary_roots_count": n_ord,
            "decisive_rescue_roots_count": n_res,
            "overall_ordinary_disagreement_rate": overall_ord_disagree,
            "decile_ordinary_disagreement_rates": decile_disagreement,
            "decile_ordinary_counts": decile_counts,
            "max_decile_ordinary_disagreement_rate": max_decile_disagreement,
            "max_decile_ordinary_disagreement_threshold": TRIPWIRE_MAX_DECILE_ORDINARY_DISAGREEMENT,
            "decile_tripwire_pass": decile_tripwire_pass,
            "student_ordinary_boost_share": student_boost_share,
            "b8_ordinary_boost_share": b8_boost_share,
            "boost_share_ratio": boost_share_ratio,
            "max_boost_share_ratio_threshold": TRIPWIRE_MAX_BOOST_SHARE_RATIO,
            "boost_tripwire_pass": boost_tripwire_pass,
            "decisive_rescue_agreement_rate": decisive_rescue_agreement_rate,
            "decisive_rescue_s0_safe_agreement_rate": decisive_rescue_s0_safe_rate,
            "decisive_rescue_exact_a_star_rate": decisive_rescue_exact_a_star_rate,
            "min_decisive_rescue_agreement_threshold": TRIPWIRE_MIN_DECISIVE_RESCUE_AGREEMENT,
            "decisive_rescue_bar_pass": decisive_rescue_bar_pass,
            "all_q_finite": all_q_finite,
            "illegal_action_count": int(ord_illegal + res_illegal),
        }
    finally:
        if was_training:
            student_net.train()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest and verify the Step B1b probe32 bank.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .pt path for probe32 bank"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--b8-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--num-worlds", type=int, default=PROBE32_WORLD_COUNT)
    parser.add_argument("--horizon", type=int, default=PROBE32_HORIZON)
    parser.add_argument("--branch-horizon", type=int, default=OPEN_STOP_BRANCH_HORIZON)
    parser.add_argument("--ordinary-per-world", type=int, default=PROBE32_ORDINARY_PER_WORLD)
    parser.add_argument("--harvest-batch-roots", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="Run tiny dry-run harvest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = harvest_probe32_bank(
        args.output,
        b8_checkpoint=args.b8_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        device=args.device,
        num_worlds=args.num_worlds,
        horizon=args.horizon,
        branch_horizon=args.branch_horizon,
        ordinary_per_world=args.ordinary_per_world,
        harvest_batch_roots=args.harvest_batch_roots,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
