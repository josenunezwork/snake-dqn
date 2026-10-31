"""Pruned depth-2 K=15 macro-option fallback search on C/T TRAIN 1-step-all-die roots.

Implements OPUS_META §3 row 5 and dir40 review-7:
1. Strict train-only split hygiene: refuses any root whose ``partition != "train"``
   or ``is_assessment`` is truthy (``assessment_roots_used == 0``).
2. Strict ``exact_stop`` gating: macro-option search runs *only* on roots where all
   canonical legal 1-step H300 B8 continuations die on selection suffix ``0``
   (``all_die_sel0 == True``). Roots with >=1 surviving 1-step continuation emit
   ``exact_stop`` untouched (0 newly caused ordinary flips by construction).
3. Frozen ``{O_0..O_5}`` option library (``K=15`` ticks per ply in main, exact
   pending-growth ``seg_count < length`` and boost-burn legality, NO ``!spacious``
   interrupt).
4. Pruned depth-2 search on CRN selection suffix ``0``:
   - Ply 1: evaluate all 6 options for ``K`` ticks from the prepared root snapshot.
   - Retain up to ``top_m_ply1`` (3) surviving Ply-1 options ranked causally by
     ``(any_certified, max_v2_surv, max_v_surv, hero_length, -o_1)`` at ``t = K``.
   - Ply 2 + continuation: from each retained post-Ply-1 prepared snapshot, evaluate
     all 6 Ply-2 options for ``K`` ticks followed by ``H - 2*K`` ticks of bare B8
     greedy continuation (with exact dead-hero compaction).
   - Freeze ``(o_1*, o_2*)`` and ``macro_triggered = (survive_sel0 == 1.0)`` with a
     SHA-256 selection witness *before* touching independent scoring suffixes ``4..7``.
5. Independent-suffix evaluation: evaluate frozen ``(o_1*, o_2*)`` across CRN scoring
   suffixes ``4..7`` and publish atomic per-batch JSON units with ``.sha256`` and
   ``.complete`` sidecars under resume-by-default ledger verification.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.r7_macro_fallback import ALLOCATION_NAME, NAMESPACE, OPTION_NAMES
from research.r7_macro_fallback.options import (
    evaluate_all_options,
    extract_states_from_ct_sim,
    select_batch_option_actions,
)
from research.sandbox_a_resident.ct_branching import _compact_sim_in_place
from research.sandbox_a_resident.ct_mechanics import (
    CTDeviceSim,
    CTOpponentRuntime,
    derive_branch_keys,
)
from research.sandbox_a_resident.ct_snapshots import (
    CompactWorldSnapshot,
    capture_world_snapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    TrainingRoster,
    training_rosters,
)
from src.model.ego2s_network import load_ego2s_checkpoint
from src.model.inference_agent import InferenceAgent
from src.simd_env.batch_sim import BatchSimConfig
from tools.ops.package_spec import (
    LEDGER_SCHEMA,
    empty_ledger,
    publish_json,
    read_ledger,
    unit_id,
)

NUM_OPTIONS = len(OPTION_NAMES)
STUDY = "t34-r7-macro-fallback-1031"
UNIT_KIND = "ct-train-all-die-batch"
ALLOCATION_INDEX = 0
B8_S3_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def batch_unit_stem(batch_index: int) -> str:
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError(f"batch_index must be a non-negative int, got {batch_index!r}")
    return f"batch-{batch_index:04d}"


def batch_unit_id(batch_index: int) -> str:
    return unit_id(NAMESPACE, ALLOCATION_NAME, batch_index)


def publish_verified_ledger(
    ledger_path: Path, verified_entries: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Publish a sidecar-sealed ``r33-verified-ledger/v1`` file for ``NAMESPACE``."""
    base = empty_ledger(NAMESPACE, ALLOCATION_NAME)
    units_list: list[dict[str, Any]] = []
    for idx in sorted(verified_entries):
        meta = verified_entries[idx]
        units_list.append(
            {
                "unit_id": batch_unit_id(int(idx)),
                "index": int(idx),
                "sha256": str(meta["sha256"]),
                "size": int(meta["size"]),
            }
        )
    base["units"] = units_list
    assert base["schema"] == LEDGER_SCHEMA
    return publish_json(ledger_path, base)


def _sha256_file(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _load_vector61(path: Path, device: torch.device) -> nn.Module:
    agent = InferenceAgent.from_checkpoint(str(path), device=device)
    if getattr(agent, "obs_spec", None) != "vector61":
        raise ValueError(f"{path} is not a vector61 checkpoint")
    return agent.network.eval()


@torch.inference_mode()
def select_b8_greedy_actions(
    sim: CTDeviceSim,
    b8_network: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Observe prepared hero rows and select canonical masked-greedy B8 actions on device."""
    _local, _glob, _scalars, legal, q_b8, _raw = observe_hero_bv8(sim, b8_network)
    fallback = torch.zeros_like(legal)
    fallback[:, :3] = True
    allowed = torch.where(legal.any(dim=1, keepdim=True), legal, fallback)
    masked_q = q_b8.masked_fill(~allowed, -torch.inf)
    actions = torch.argmax(masked_q, dim=1)
    return actions, legal, q_b8


def assert_train_only_roots(records: Sequence[Mapping[str, Any]]) -> None:
    """Refuse any root from the held-out assessment split or non-all-die selection gate."""
    for idx, rec in enumerate(records):
        part = rec.get("partition")
        is_assess = bool(rec.get("is_assessment", False))
        if part != "train" or is_assess:
            raise ValueError(
                f"Refusing non-train root at index {idx}: partition={part!r} "
                f"is_assessment={is_assess!r} (assessment_roots_used must equal 0)"
            )
        if not bool(rec.get("all_die_sel0", False)):
            raise ValueError(
                f"Refusing root {rec.get('root_uid')!r} at index {idx}: all_die_sel0 is False "
                "(strict exact_stop gating forbids macro search when any 1-step branch survives)"
            )


def _write_progress(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "progress.json.partial"
    tmp.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output_dir / "progress.json")
    print(f"R7_PROGRESS {json.dumps(dict(payload), sort_keys=True)}", flush=True)


def _build_slab(
    snapshots: Sequence[CompactWorldSnapshot],
    rosters: Sequence[TrainingRoster],
    *,
    base_cfg: BatchSimConfig,
    device: torch.device,
    b8_network: nn.Module,
    checkpoint_networks: Mapping[str, nn.Module],
    rekey_sim_keys: Sequence[int] | None = None,
    rekey_opp_keys: Sequence[int] | None = None,
) -> tuple[CTDeviceSim, CTOpponentRuntime]:
    """Instantiate a ``CTDeviceSim`` + ``CTOpponentRuntime`` slab and restore snapshots."""
    b_count = len(snapshots)
    if b_count < 1 or len(rosters) != b_count:
        raise ValueError("snapshots and rosters must have equal positive length")
    max_cap = max(
        int(base_cfg.max_capacity),
        max(int(s.body_cap) for s in snapshots),
    )
    slab_cfg = replace(
        base_cfg,
        num_envs=b_count,
        max_capacity=max_cap,
        body_storage_capacity=max(int(base_cfg.body_storage_capacity), max_cap),
    )
    slab_sim = CTDeviceSim(
        slab_cfg,
        device=device,
        train_mode=False,
        allow_respawn=True,
        move_backend="indexed" if device.type == "cuda" else "reference",
    )
    r2_cache.enable(slab_sim)
    slab_opp = CTOpponentRuntime(
        slab_sim,
        list(rosters),
        b8_network=b8_network,
        checkpoint_networks=checkpoint_networks,
    )
    restore_world_snapshots_into(snapshots, slab_sim, slab_opp)
    slab_sim.terminal_slots[:, 0] = True
    if rekey_sim_keys is not None or rekey_opp_keys is not None:
        if rekey_sim_keys is None or rekey_opp_keys is None:
            raise ValueError("rekey_sim_keys and rekey_opp_keys must be provided together")
        sim_keys_t = torch.tensor(list(rekey_sim_keys), dtype=torch.long, device=device)
        opp_keys_t = torch.tensor(list(rekey_opp_keys), dtype=torch.long, device=device)
        slab_sim.rng_key.copy_(sim_keys_t)
        slab_sim.rng_counter.zero_()
        slab_opp.rekey_branches(opp_keys_t)
    return slab_sim, slab_opp


@torch.inference_mode()
def run_two_ply_macro_rollouts(
    snapshots: Sequence[CompactWorldSnapshot],
    rosters: Sequence[TrainingRoster],
    o1_ids: Sequence[int],
    o2_ids: Sequence[int],
    *,
    sim_rng_keys: Sequence[int],
    opp_rng_keys: Sequence[int],
    horizon: int,
    macro_ticks: int,
    compaction_interval: int,
    base_cfg: BatchSimConfig,
    device: torch.device,
    b8_network: nn.Module,
    checkpoint_networks: Mapping[str, nn.Module],
) -> dict[str, Any]:
    """Roll out frozen ``(o_1, o_2)`` pairs for ``horizon`` ticks from prepared root snapshots.

    Steps ``0 .. K-1`` execute closed-loop option ``o_1``; steps ``K .. 2K-1`` execute
    closed-loop option ``o_2``; steps ``2K .. H-1`` execute greedy bare B8 continuation.
    Never interrupts on ``!spacious``.
    """
    b_total = len(snapshots)
    if b_total == 0:
        return {
            "survive": np.zeros(0, dtype=np.float64),
            "mean_mass": np.zeros(0, dtype=np.float64),
            "ply1_nonfatal_ticks": np.zeros(0, dtype=np.int64),
            "two_ply_nonfatal_ticks": np.zeros(0, dtype=np.int64),
            "executed_sim_steps": 0,
        }
    slab_sim, slab_opp = _build_slab(
        snapshots,
        rosters,
        base_cfg=base_cfg,
        device=device,
        b8_network=b8_network,
        checkpoint_networks=checkpoint_networks,
        rekey_sim_keys=sim_rng_keys,
        rekey_opp_keys=opp_rng_keys,
    )
    # Record a device B8 observation at batch start (verifies prepared root & device path).
    select_b8_greedy_actions(slab_sim, b8_network)

    active_slots = np.arange(b_total, dtype=np.int64)
    active_slots_t = torch.arange(b_total, dtype=torch.long, device=device)
    o1_arr = np.asarray(o1_ids, dtype=np.int64)
    o2_arr = np.asarray(o2_ids, dtype=np.int64)

    survive_out = torch.zeros(b_total, dtype=torch.float64, device=device)
    mass_sum = torch.zeros(b_total, dtype=torch.float64, device=device)
    ply1_ticks = torch.zeros(b_total, dtype=torch.long, device=device)
    two_ply_ticks = torch.zeros(b_total, dtype=torch.long, device=device)
    executed_sim_steps = 0

    for h in range(horizon):
        if h > 0:
            if (
                h >= 2 * macro_ticks
                and compaction_interval > 0
                and ((h - 2 * macro_ticks) % compaction_interval == 0)
                and not bool(slab_sim.alive[:, 0].all().item())
            ):
                keep = slab_sim.alive[:, 0].nonzero(as_tuple=False).flatten()
                if int(keep.numel()) == 0:
                    break
                _compact_sim_in_place(slab_sim, keep)
                slab_opp.compact_rows(keep)
                active_slots_t = active_slots_t.index_select(0, keep)
                active_slots = active_slots_t.detach().cpu().numpy()
            slab_sim.prepare()

        if h < macro_ticks:
            hero_act, _ = select_batch_option_actions(slab_sim, o1_arr[active_slots])
        elif h < 2 * macro_ticks:
            hero_act, _ = select_batch_option_actions(slab_sim, o2_arr[active_slots])
        else:
            hero_act, _, _ = select_b8_greedy_actions(slab_sim, b8_network)

        full_actions = slab_opp.actions(slab_sim, hero_act)
        slab_sim.step(full_actions)
        slab_opp.after_step(slab_sim)
        executed_sim_steps += int(slab_sim.E)

        alive_now = slab_sim.alive[:, 0]
        if h < macro_ticks:
            ply1_ticks.index_add_(0, active_slots_t, alive_now.to(dtype=torch.long))
        if h < 2 * macro_ticks:
            two_ply_ticks.index_add_(0, active_slots_t, alive_now.to(dtype=torch.long))

        step_mass = torch.where(
            alive_now,
            slab_sim.length[:, 0].to(dtype=torch.float64),
            torch.zeros(slab_sim.E, dtype=torch.float64, device=device),
        )
        mass_sum.index_add_(0, active_slots_t, step_mass)
        if h == horizon - 1:
            survive_out.index_copy_(0, active_slots_t, alive_now.to(dtype=torch.float64))

    return {
        "survive": survive_out.detach().cpu().numpy(),
        "mean_mass": (mass_sum / float(horizon)).detach().cpu().numpy(),
        "ply1_nonfatal_ticks": ply1_ticks.detach().cpu().numpy(),
        "two_ply_nonfatal_ticks": two_ply_ticks.detach().cpu().numpy(),
        "executed_sim_steps": int(executed_sim_steps),
    }


@torch.inference_mode()
def evaluate_root_batch(
    root_records: Sequence[Mapping[str, Any]],
    *,
    rosters_by_world: Sequence[TrainingRoster],
    horizon: int,
    macro_ticks: int,
    top_m_ply1: int,
    selection_suffix: int,
    scoring_suffixes: Sequence[int],
    compaction_interval: int,
    max_slab_envs: int,
    canary_only: bool,
    base_cfg: BatchSimConfig,
    device: torch.device,
    b8_network: nn.Module,
    checkpoint_networks: Mapping[str, nn.Module],
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Run pruned depth-2 macro search on selection suffix then score independent suffixes."""
    assert_train_only_roots(root_records)
    if macro_ticks < 1 or horizon <= 2 * macro_ticks:
        raise ValueError(
            f"Require macro_ticks >= 1 and horizon > 2*macro_ticks; "
            f"got K={macro_ticks}, H={horizon}"
        )
    if top_m_ply1 < 1 or top_m_ply1 > NUM_OPTIONS:
        raise ValueError(f"top_m_ply1 must be in 1..{NUM_OPTIONS}, got {top_m_ply1}")
    if selection_suffix != 0:
        raise ValueError("R7 freezes selection_suffix == 0 (CRN with 1-step exact_stop suffix 0)")
    if any(int(s) < 4 or int(s) > 7 for s in scoring_suffixes):
        raise ValueError(
            f"scoring_suffixes must lie in independent suffixes 4..7, got {scoring_suffixes}"
        )

    n_roots = len(root_records)
    if n_roots == 0:
        raise ValueError("root_records must be nonempty")

    t0 = time.perf_counter()
    executed_sim_steps = 0

    # --- PHASE 1A: Ply-1 evaluation (6 options x n_roots for K ticks on selection suffix 0) ---
    ply1_jobs: list[tuple[int, int]] = [
        (r_idx, opt1) for r_idx in range(n_roots) for opt1 in range(NUM_OPTIONS)
    ]
    ply1_alive_end = np.zeros((n_roots, NUM_OPTIONS), dtype=bool)
    ply1_nonfatal_ticks = np.zeros((n_roots, NUM_OPTIONS), dtype=np.int64)
    ply1_mass_sum = np.zeros((n_roots, NUM_OPTIONS), dtype=np.float64)
    ply1_end_length = np.zeros((n_roots, NUM_OPTIONS), dtype=np.int64)
    ply1_diag = np.zeros((n_roots, NUM_OPTIONS, 4), dtype=np.float64)
    ply1_snapshots: dict[tuple[int, int], CompactWorldSnapshot] = {}

    for start in range(0, len(ply1_jobs), max_slab_envs):
        chunk = ply1_jobs[start : start + max_slab_envs]
        snaps = [root_records[r_idx]["snapshot"] for r_idx, _ in chunk]
        rosters = [rosters_by_world[int(root_records[r_idx]["world_id"])] for r_idx, _ in chunk]
        bkeys = [
            derive_branch_keys(str(root_records[r_idx]["root_uid"]), selection_suffix)
            for r_idx, _ in chunk
        ]
        opt1_ids = np.array([opt1 for _, opt1 in chunk], dtype=np.int64)

        slab_sim, slab_opp = _build_slab(
            snaps,
            rosters,
            base_cfg=base_cfg,
            device=device,
            b8_network=b8_network,
            checkpoint_networks=checkpoint_networks,
            rekey_sim_keys=[bk.sim_rng_key for bk in bkeys],
            rekey_opp_keys=[bk.opp_rng_key for bk in bkeys],
        )
        # Exercise device B8 forward + argmax on prepared root slab.
        select_b8_greedy_actions(slab_sim, b8_network)

        b_chunk = len(chunk)
        ticks_alive = torch.zeros(b_chunk, dtype=torch.long, device=device)
        mass_acc = torch.zeros(b_chunk, dtype=torch.float64, device=device)

        for h in range(macro_ticks):
            if h > 0:
                slab_sim.prepare()
            hero_act, _ = select_batch_option_actions(slab_sim, opt1_ids)
            full_actions = slab_opp.actions(slab_sim, hero_act)
            slab_sim.step(full_actions)
            slab_opp.after_step(slab_sim)
            executed_sim_steps += int(slab_sim.E)

            alive_now = slab_sim.alive[:, 0]
            ticks_alive += alive_now.to(dtype=torch.long)
            mass_acc += torch.where(
                alive_now,
                slab_sim.length[:, 0].to(dtype=torch.float64),
                torch.zeros(b_chunk, dtype=torch.float64, device=device),
            )

        # Prepare post-Ply-1 boundary once so surviving Ply-1 states are captured prepared=True
        # with exact post-prepare RNG counters matching continuous step h == K.
        alive_end_np = slab_sim.alive[:, 0].detach().cpu().numpy().astype(bool)
        if bool(alive_end_np.any()):
            slab_sim.prepare()
            end_states = extract_states_from_ct_sim(slab_sim)
        else:
            end_states = []

        ticks_np = ticks_alive.detach().cpu().numpy()
        mass_np = mass_acc.detach().cpu().numpy()
        lengths_np = slab_sim.length[:, 0].detach().cpu().numpy()

        for row_pos, (r_idx, opt1) in enumerate(chunk):
            ply1_nonfatal_ticks[r_idx, opt1] = int(ticks_np[row_pos])
            ply1_mass_sum[r_idx, opt1] = float(mass_np[row_pos])
            if alive_end_np[row_pos]:
                ply1_alive_end[r_idx, opt1] = True
                ply1_end_length[r_idx, opt1] = int(lengths_np[row_pos])
                _, _, dg = evaluate_all_options(end_states[row_pos])
                ply1_diag[r_idx, opt1] = dg
                rec = root_records[r_idx]
                ply1_snapshots[(r_idx, opt1)] = capture_world_snapshot(
                    slab_sim,
                    slab_opp,
                    row_pos,
                    world_id=int(rec["world_id"]),
                    life_id=int(rec["life_id"]),
                    tick=int(rec["tick"]) + macro_ticks,
                )

    # Select up to top_m_ply1 surviving Ply-1 options per root using causal t=K diagnostics.
    retained_ply1: list[list[int]] = []
    ply2_jobs: list[tuple[int, int, int]] = []
    for r_idx in range(n_roots):
        surv_opts = [opt1 for opt1 in range(NUM_OPTIONS) if ply1_alive_end[r_idx, opt1]]
        surv_opts.sort(
            key=lambda opt1: (
                float(ply1_diag[r_idx, opt1, 0]),  # any_certified
                float(ply1_diag[r_idx, opt1, 1]),  # max_v2_surv
                float(ply1_diag[r_idx, opt1, 2]),  # max_v_surv
                int(ply1_end_length[r_idx, opt1]),
                -int(opt1),
            ),
            reverse=True,
        )
        kept = surv_opts[:top_m_ply1]
        retained_ply1.append(kept)
        for opt1 in kept:
            for opt2 in range(NUM_OPTIONS):
                ply2_jobs.append((r_idx, opt1, opt2))

    if progress_cb is not None:
        progress_cb(
            {
                "stage": "ply1_complete",
                "roots": n_roots,
                "ply1_surviving_roots": int(ply1_alive_end.any(axis=1).sum()),
                "ply2_branches_queued": len(ply2_jobs),
                "executed_sim_steps": executed_sim_steps,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
        )

    # --- PHASE 1B: Ply-2 (K ticks) + B8 continuation (H - 2K ticks) on selection suffix 0 ---
    pair_survive_s0 = np.zeros((n_roots, NUM_OPTIONS, NUM_OPTIONS), dtype=np.float64)
    pair_mean_mass_s0 = np.zeros((n_roots, NUM_OPTIONS, NUM_OPTIONS), dtype=np.float64)
    pair_two_ply_ticks_s0 = np.zeros((n_roots, NUM_OPTIONS, NUM_OPTIONS), dtype=np.int64)
    rem_steps = horizon - macro_ticks

    for start in range(0, len(ply2_jobs), max_slab_envs):
        chunk = ply2_jobs[start : start + max_slab_envs]
        snaps = [ply1_snapshots[(r_idx, opt1)] for r_idx, opt1, _ in chunk]
        rosters = [rosters_by_world[int(root_records[r_idx]["world_id"])] for r_idx, _, _ in chunk]
        opt2_arr = np.array([opt2 for _, _, opt2 in chunk], dtype=np.int64)

        # Do NOT reset rng_counter/rng_step: ply1_snapshots already hold exact post-Ply-1 RNG state.
        slab_sim, slab_opp = _build_slab(
            snaps,
            rosters,
            base_cfg=base_cfg,
            device=device,
            b8_network=b8_network,
            checkpoint_networks=checkpoint_networks,
            rekey_sim_keys=None,
            rekey_opp_keys=None,
        )

        b_chunk = len(chunk)
        active_slots = np.arange(b_chunk, dtype=np.int64)
        active_slots_t = torch.arange(b_chunk, dtype=torch.long, device=device)
        surv_slab = torch.zeros(b_chunk, dtype=torch.float64, device=device)
        mass_rem = torch.zeros(b_chunk, dtype=torch.float64, device=device)
        ply2_ticks = torch.zeros(b_chunk, dtype=torch.long, device=device)

        for t in range(rem_steps):
            if t > 0:
                if (
                    t >= macro_ticks
                    and compaction_interval > 0
                    and ((t - macro_ticks) % compaction_interval == 0)
                    and not bool(slab_sim.alive[:, 0].all().item())
                ):
                    keep = slab_sim.alive[:, 0].nonzero(as_tuple=False).flatten()
                    if int(keep.numel()) == 0:
                        break
                    _compact_sim_in_place(slab_sim, keep)
                    slab_opp.compact_rows(keep)
                    active_slots_t = active_slots_t.index_select(0, keep)
                    active_slots = active_slots_t.detach().cpu().numpy()
                slab_sim.prepare()

            if t < macro_ticks:
                hero_act, _ = select_batch_option_actions(slab_sim, opt2_arr[active_slots])
            else:
                hero_act, _, _ = select_b8_greedy_actions(slab_sim, b8_network)

            full_actions = slab_opp.actions(slab_sim, hero_act)
            slab_sim.step(full_actions)
            slab_opp.after_step(slab_sim)
            executed_sim_steps += int(slab_sim.E)

            alive_now = slab_sim.alive[:, 0]
            if t < macro_ticks:
                ply2_ticks.index_add_(0, active_slots_t, alive_now.to(dtype=torch.long))
            step_mass = torch.where(
                alive_now,
                slab_sim.length[:, 0].to(dtype=torch.float64),
                torch.zeros(slab_sim.E, dtype=torch.float64, device=device),
            )
            mass_rem.index_add_(0, active_slots_t, step_mass)
            if t == rem_steps - 1:
                surv_slab.index_copy_(0, active_slots_t, alive_now.to(dtype=torch.float64))

        surv_np = surv_slab.detach().cpu().numpy()
        mass_rem_np = mass_rem.detach().cpu().numpy()
        ply2_ticks_np = ply2_ticks.detach().cpu().numpy()

        for row_pos, (r_idx, opt1, opt2) in enumerate(chunk):
            pair_survive_s0[r_idx, opt1, opt2] = float(surv_np[row_pos])
            total_mass = ply1_mass_sum[r_idx, opt1] + float(mass_rem_np[row_pos])
            pair_mean_mass_s0[r_idx, opt1, opt2] = total_mass / float(horizon)
            pair_two_ply_ticks_s0[r_idx, opt1, opt2] = int(
                ply1_nonfatal_ticks[r_idx, opt1] + int(ply2_ticks_np[row_pos])
            )

    # --- FREEZE (o_1*, o_2*) AND macro_triggered ON SUFFIX 0 BEFORE SCORING SUFFIXES 4..7 ---
    frozen_decisions: list[dict[str, Any]] = []
    for r_idx, rec in enumerate(root_records):
        kept = retained_ply1[r_idx]
        best_o1 = 0
        best_o2 = 0
        best_q = -1.0
        best_surv = 0.0
        best_mass = 0.0
        best_ply1_ticks = int(ply1_nonfatal_ticks[r_idx, 0])
        best_two_ply_ticks = int(ply1_nonfatal_ticks[r_idx, 0])

        if kept:
            for opt1 in kept:
                for opt2 in range(NUM_OPTIONS):
                    sv = float(pair_survive_s0[r_idx, opt1, opt2])
                    ms = float(pair_mean_mass_s0[r_idx, opt1, opt2])
                    q_val = sv + 0.25 * (ms / 1000.0)
                    if q_val > best_q:
                        best_q = q_val
                        best_o1 = int(opt1)
                        best_o2 = int(opt2)
                        best_surv = sv
                        best_mass = ms
                        best_ply1_ticks = int(ply1_nonfatal_ticks[r_idx, opt1])
                        best_two_ply_ticks = int(pair_two_ply_ticks_s0[r_idx, opt1, opt2])
        else:
            # No Ply-1 option survived K ticks; pick longest-lasting Ply-1 diagnostic tie-break
            # and do NOT trigger macro fallback (remains exact_stop).
            best_o1 = int(np.argmax(ply1_nonfatal_ticks[r_idx]))
            best_o2 = 0
            best_surv = 0.0
            best_mass = float(ply1_mass_sum[r_idx, best_o1]) / float(horizon)
            best_q = 0.25 * (best_mass / 1000.0)
            best_ply1_ticks = int(ply1_nonfatal_ticks[r_idx, best_o1])
            best_two_ply_ticks = best_ply1_ticks

        frozen_decisions.append(
            {
                "root_uid": str(rec["root_uid"]),
                "macro_triggered": bool(best_surv >= 0.5),
                "o1_star": int(best_o1),
                "o2_star": int(best_o2),
                "o1_name": OPTION_NAMES[best_o1],
                "o2_name": OPTION_NAMES[best_o2],
                "sel0_survive": float(best_surv),
                "sel0_mean_mass": round(float(best_mass), 4),
                "sel0_q": round(float(best_q), 6),
                "sel0_ply1_nonfatal_ticks": int(best_ply1_ticks),
                "sel0_two_ply_nonfatal_ticks": int(best_two_ply_ticks),
                "retained_ply1_options": [int(x) for x in kept],
            }
        )

    selection_freeze_sha256 = hashlib.sha256(
        json.dumps(frozen_decisions, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if progress_cb is not None:
        progress_cb(
            {
                "stage": "selection_frozen",
                "roots": n_roots,
                "macro_triggered_count": sum(1 for d in frozen_decisions if d["macro_triggered"]),
                "selection_freeze_sha256": selection_freeze_sha256,
                "executed_sim_steps": executed_sim_steps,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
        )

    # --- PHASE 2: Independent scoring suffixes 4..7 using frozen (o_1*, o_2*) ---
    # Evaluate all macro_triggered roots across independent suffixes 4..7.
    # (If canary_only and 0 roots triggered at tiny H, still score root 0 to prove Phase 2 path.)
    score_jobs: list[tuple[int, int, int]] = []  # (r_idx, s_pos, suffix_k)
    for r_idx, dec in enumerate(frozen_decisions):
        if dec["macro_triggered"] or (canary_only and r_idx == 0):
            for s_pos, suffix_k in enumerate(scoring_suffixes):
                score_jobs.append((r_idx, s_pos, int(suffix_k)))

    n_score_suf = len(scoring_suffixes)
    score_surv = np.zeros((n_roots, n_score_suf), dtype=np.float64)
    score_mass = np.zeros((n_roots, n_score_suf), dtype=np.float64)
    score_ply1_ticks = np.zeros((n_roots, n_score_suf), dtype=np.int64)
    score_two_ply_ticks = np.zeros((n_roots, n_score_suf), dtype=np.int64)
    scored_mask = np.zeros(n_roots, dtype=bool)

    for start in range(0, len(score_jobs), max_slab_envs):
        chunk = score_jobs[start : start + max_slab_envs]
        snaps = [root_records[r_idx]["snapshot"] for r_idx, _, _ in chunk]
        rosters = [rosters_by_world[int(root_records[r_idx]["world_id"])] for r_idx, _, _ in chunk]
        o1_list = [int(frozen_decisions[r_idx]["o1_star"]) for r_idx, _, _ in chunk]
        o2_list = [int(frozen_decisions[r_idx]["o2_star"]) for r_idx, _, _ in chunk]
        bkeys = [
            derive_branch_keys(str(root_records[r_idx]["root_uid"]), suffix_k)
            for r_idx, _, suffix_k in chunk
        ]

        res = run_two_ply_macro_rollouts(
            snaps,
            rosters,
            o1_list,
            o2_list,
            sim_rng_keys=[bk.sim_rng_key for bk in bkeys],
            opp_rng_keys=[bk.opp_rng_key for bk in bkeys],
            horizon=horizon,
            macro_ticks=macro_ticks,
            compaction_interval=compaction_interval,
            base_cfg=base_cfg,
            device=device,
            b8_network=b8_network,
            checkpoint_networks=checkpoint_networks,
        )
        executed_sim_steps += int(res["executed_sim_steps"])
        for row_pos, (r_idx, s_pos, _suffix_k) in enumerate(chunk):
            scored_mask[r_idx] = True
            score_surv[r_idx, s_pos] = float(res["survive"][row_pos])
            score_mass[r_idx, s_pos] = float(res["mean_mass"][row_pos])
            score_ply1_ticks[r_idx, s_pos] = int(res["ply1_nonfatal_ticks"][row_pos])
            score_two_ply_ticks[r_idx, s_pos] = int(res["two_ply_nonfatal_ticks"][row_pos])

    root_outcomes: list[dict[str, Any]] = []
    for r_idx, rec in enumerate(root_records):
        dec = frozen_decisions[r_idx]
        triggered = bool(dec["macro_triggered"])
        exact_stop_score_s = float(rec["exact_stop_s_score"])
        exact_stop_score_m = float(rec["exact_stop_m_score"])
        if triggered:
            indep_surv_vec = [round(float(x), 6) for x in score_surv[r_idx].tolist()]
            indep_mass_vec = [round(float(x), 4) for x in score_mass[r_idx].tolist()]
            indep_ply1_vec = [int(x) for x in score_ply1_ticks[r_idx].tolist()]
            indep_two_ply_vec = [int(x) for x in score_two_ply_ticks[r_idx].tolist()]
            policy_score_s = float(np.mean(score_surv[r_idx]))
            policy_score_m = float(np.mean(score_mass[r_idx]))
            mean_ply1_ticks = float(np.mean(score_ply1_ticks[r_idx]))
            mean_two_ply_ticks = float(np.mean(score_two_ply_ticks[r_idx]))
        else:
            # Strict exact_stop gating: non-triggered roots emit exact_stop untouched.
            indep_surv_vec = (
                [round(float(x), 6) for x in score_surv[r_idx].tolist()]
                if scored_mask[r_idx]
                else []
            )
            indep_mass_vec = (
                [round(float(x), 4) for x in score_mass[r_idx].tolist()]
                if scored_mask[r_idx]
                else []
            )
            indep_ply1_vec = (
                [int(x) for x in score_ply1_ticks[r_idx].tolist()] if scored_mask[r_idx] else []
            )
            indep_two_ply_vec = (
                [int(x) for x in score_two_ply_ticks[r_idx].tolist()] if scored_mask[r_idx] else []
            )
            policy_score_s = exact_stop_score_s
            policy_score_m = exact_stop_score_m
            mean_ply1_ticks = 0.0
            mean_two_ply_ticks = 0.0

        root_outcomes.append(
            {
                "root_index": int(rec["root_index"]),
                "root_uid": str(rec["root_uid"]),
                "partition": str(rec["partition"]),
                "kind": str(rec["kind"]),
                "world_id": int(rec["world_id"]),
                "mix": str(rec["mix"]),
                "life_id": int(rec["life_id"]),
                "tick": int(rec["tick"]),
                "lookback": None if rec.get("lookback") is None else int(rec["lookback"]),
                "hero_length": int(rec["hero_length"]),
                "length_bin": str(rec["length_bin"]),
                "all_die_sel0": True,
                "exact_stop_action_sel0": int(rec["exact_stop_action_sel0"]),
                "exact_stop_s_sel0": float(rec["exact_stop_s_sel0"]),
                "exact_stop_s_score": round(exact_stop_score_s, 6),
                "exact_stop_m_score": round(exact_stop_score_m, 4),
                **dec,
                "indep_scoring_suffixes": [int(s) for s in scoring_suffixes],
                "indep_survive_by_suffix": indep_surv_vec,
                "indep_mean_mass_by_suffix": indep_mass_vec,
                "indep_ply1_nonfatal_ticks_by_suffix": indep_ply1_vec,
                "indep_two_ply_nonfatal_ticks_by_suffix": indep_two_ply_vec,
                "policy_s_score": round(policy_score_s, 6),
                "policy_m_score": round(policy_score_m, 4),
                "rescue_delta_s_score": round(policy_score_s - exact_stop_score_s, 6),
                "indep_mean_ply1_nonfatal_ticks": round(mean_ply1_ticks, 4),
                "indep_mean_two_ply_nonfatal_ticks": round(mean_two_ply_ticks, 4),
            }
        )

    return {
        "num_roots": n_roots,
        "selection_freeze_sha256": selection_freeze_sha256,
        "macro_triggered_count": sum(1 for r in root_outcomes if r["macro_triggered"]),
        "executed_sim_steps": int(executed_sim_steps),
        "elapsed_seconds": round(time.perf_counter() - t0, 4),
        "roots": root_outcomes,
    }


def run_search_from_files(
    *,
    input_dir: Path,
    config_path: Path,
    ledger_path: Path,
    output_dir: Path,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """Execute resumable batch units from staged inputs and publish sidecar-sealed exports."""
    t_start = time.perf_counter()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "t34-r7-search-config/v1":
        raise ValueError(f"Unsupported R7 config schema: {cfg.get('schema')!r}")

    ledger_rows = read_ledger(ledger_path, NAMESPACE, ALLOCATION_NAME)
    verified_units = {str(row["unit_id"]) for row in ledger_rows}

    b8_path = input_dir / str(cfg.get("b8_checkpoint", "b8-s3.pth"))
    frp3_path = input_dir / str(cfg.get("frp3_checkpoint", "frp3-s12.pth"))
    champ_path = input_dir / str(cfg.get("champion_checkpoint", "champion.pth"))
    roots_path = input_dir / str(cfg.get("roots_file", "train-all-die-roots.pt"))

    if bool(cfg.get("verify_checkpoint_hashes", True)):
        if _sha256_file(b8_path) != B8_S3_SHA256:
            raise ValueError(f"b8 checkpoint SHA mismatch: {b8_path}")
        if _sha256_file(frp3_path) != FRP3_S12_SHA256:
            raise ValueError(f"frp3 checkpoint SHA mismatch: {frp3_path}")
        if _sha256_file(champ_path) != CHAMPION_SHA256:
            raise ValueError(f"champion checkpoint SHA mismatch: {champ_path}")

    if roots_path.suffix == ".xz":
        with lzma.open(roots_path, "rb") as stream:
            roots_bundle = torch.load(
                io.BytesIO(stream.read()), map_location="cpu", weights_only=False
            )
    else:
        roots_bundle = torch.load(roots_path, map_location="cpu", weights_only=False)
    if roots_bundle.get("schema") != "t34-r7-train-all-die-roots/v1":
        raise ValueError(f"Foreign roots bundle schema: {roots_bundle.get('schema')!r}")
    if int(roots_bundle.get("assessment_roots_used", -1)) != 0:
        raise ValueError("Refusing roots bundle with nonzero assessment_roots_used")

    raw_batches: dict[int, list[dict[str, Any]]] = {}
    for entry in roots_bundle["roots"]:
        rec = dict(entry)
        rec["snapshot"] = CompactWorldSnapshot.from_state_dict(entry["snapshot"])
        b_idx = int(rec["batch_index"])
        raw_batches.setdefault(b_idx, []).append(rec)

    all_records = [rec for b_list in raw_batches.values() for rec in b_list]
    assert_train_only_roots(all_records)

    dev = torch.device(device_str)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested for R7 search but torch.cuda.is_available() is False"
            )
        if dev.index is None:
            dev = torch.device("cuda:0")

    b8_net = load_ego2s_checkpoint(str(b8_path), str(dev)).eval()
    checkpoint_networks = {
        FRP3_S12_SHA256: _load_vector61(frp3_path, dev),
        CHAMPION_SHA256: _load_vector61(champ_path, dev),
    }

    num_worlds = int(roots_bundle.get("total_worlds", 1024))
    rosters_by_world = training_rosters(num_worlds)
    base_cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        max_capacity=int(roots_bundle.get("max_body_capacity", 10_400)),
        body_storage_capacity=int(roots_bundle.get("max_body_capacity", 10_400)),
        mechanics_version=1,
    )

    batch_indices = [int(x) for x in cfg["batch_indices"]]
    max_roots_per_batch = cfg.get("max_roots_per_batch")
    canary_only = bool(cfg.get("canary_only", False))
    horizon = int(cfg["horizon"])
    macro_ticks = int(cfg["macro_ticks"])
    top_m_ply1 = int(cfg.get("top_m_ply1", 3))
    selection_suffix = int(cfg.get("selection_suffix", 0))
    scoring_suffixes = [int(x) for x in cfg.get("scoring_suffixes", [4, 5, 6, 7])]
    compaction_interval = int(cfg.get("compaction_interval", 10))
    max_slab_envs = int(cfg.get("max_slab_envs", 512))

    skipped_unit_ids: list[str] = []
    computed_unit_ids: list[str] = []
    batch_summaries: list[dict[str, Any]] = []
    total_sim_steps = 0

    _write_progress(
        output_dir,
        {
            "phase": "startup",
            "device": str(dev),
            "canary_only": canary_only,
            "batches_configured": len(batch_indices),
            "verified_in_ledger": sorted(verified_units),
        },
    )

    for b_idx in batch_indices:
        uid = batch_unit_id(b_idx)
        stem = batch_unit_stem(b_idx)
        if uid in verified_units:
            skipped_unit_ids.append(uid)
            continue
        batch_recs = list(raw_batches.get(b_idx, []))
        if max_roots_per_batch is not None:
            batch_recs = batch_recs[: int(max_roots_per_batch)]
        if not batch_recs:
            raise ValueError(f"Batch {uid} resolved to 0 roots")

        batch_eval = evaluate_root_batch(
            batch_recs,
            rosters_by_world=rosters_by_world,
            horizon=horizon,
            macro_ticks=macro_ticks,
            top_m_ply1=top_m_ply1,
            selection_suffix=selection_suffix,
            scoring_suffixes=scoring_suffixes,
            compaction_interval=compaction_interval,
            max_slab_envs=max_slab_envs,
            canary_only=canary_only,
            base_cfg=base_cfg,
            device=dev,
            b8_network=b8_net,
            checkpoint_networks=checkpoint_networks,
            progress_cb=lambda p, uid=uid: _write_progress(output_dir, {"unit_id": uid, **p}),
        )
        total_sim_steps += int(batch_eval["executed_sim_steps"])

        unit_payload = {
            "schema": "t34-r7-batch-unit/v1",
            "study": STUDY,
            "namespace": NAMESPACE,
            "allocation": ALLOCATION_NAME,
            "unit_kind": UNIT_KIND,
            "allocation_index": ALLOCATION_INDEX,
            "unit_id": uid,
            "unit_stem": stem,
            "batch_index": b_idx,
            "canary_only": canary_only,
            "assessment_roots_used": 0,
            "spacious_interrupt_enabled": False,
            "horizon": horizon,
            "macro_ticks": macro_ticks,
            "top_m_ply1": top_m_ply1,
            "selection_suffix": selection_suffix,
            "scoring_suffixes": scoring_suffixes,
            "device": str(dev),
            **batch_eval,
        }
        publish_json(output_dir / "units" / f"{stem}.json", unit_payload)
        computed_unit_ids.append(uid)
        batch_summaries.append(
            {
                "unit_id": uid,
                "num_roots": batch_eval["num_roots"],
                "macro_triggered_count": batch_eval["macro_triggered_count"],
                "selection_freeze_sha256": batch_eval["selection_freeze_sha256"],
                "executed_sim_steps": batch_eval["executed_sim_steps"],
                "elapsed_seconds": batch_eval["elapsed_seconds"],
            }
        )
        _write_progress(
            output_dir,
            {
                "phase": "unit_published",
                "unit_id": uid,
                "computed_units": len(computed_unit_ids),
                "skipped_units": len(skipped_unit_ids),
                "total_sim_steps": total_sim_steps,
            },
        )

    skip_proof = {
        "schema": "t34-r7-skip-proof/v1",
        "study": STUDY,
        "namespace": NAMESPACE,
        "unit_kind": UNIT_KIND,
        "allocation_index": ALLOCATION_INDEX,
        "ledger_sha256": _sha256_file(ledger_path),
        "verified_in_ledger": sorted(verified_units),
        "skipped_unit_ids": skipped_unit_ids,
        "computed_unit_ids": computed_unit_ids,
        "recomputed_verified_overlap": sorted(set(computed_unit_ids) & set(verified_units)),
    }
    if skip_proof["recomputed_verified_overlap"]:
        raise RuntimeError("Recomputed an already verified unit")
    publish_json(output_dir / "skip-proof.json", skip_proof)

    run_summary = {
        "schema": "t34-r7-attempt-diagnostic/v1",
        "study": STUDY,
        "namespace": NAMESPACE,
        "unit_kind": UNIT_KIND,
        "allocation_index": ALLOCATION_INDEX,
        "canary_only": canary_only,
        "authoritative_readout": False,
        "assessment_roots_used": 0,
        "spacious_interrupt_enabled": False,
        "device": str(dev),
        "cuda_device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "horizon": horizon,
        "macro_ticks": macro_ticks,
        "top_m_ply1": top_m_ply1,
        "computed_unit_ids": computed_unit_ids,
        "skipped_unit_ids": skipped_unit_ids,
        "batch_summaries": batch_summaries,
        "total_executed_sim_steps": total_sim_steps,
        "wall_seconds": round(time.perf_counter() - t_start, 4),
    }
    publish_json(output_dir / "run-summary.json", run_summary)
    _write_progress(output_dir, {"phase": "complete", "wall_seconds": run_summary["wall_seconds"]})
    return run_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_search_from_files(
        input_dir=args.input_dir,
        config_path=args.config,
        ledger_path=args.ledger,
        output_dir=args.output,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
