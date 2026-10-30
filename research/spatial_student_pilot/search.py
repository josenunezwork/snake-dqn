"""Finite exact-simulator first-action search and terminal-hero paired gameplay.

Selection and assessment randomness have separate immutable suffix identities.
No rollout observation buffer, learned leaf value, or tree is used here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.core import allowed_mask
from research.sandbox_a_resident.ct_branching import _compact_sim_in_place
from research.sandbox_a_resident.ct_mechanics import (
    CTDeviceSim,
    CTOpponentRuntime,
    derive_branch_keys,
)
from research.sandbox_a_resident.ct_snapshots import (
    CompactWorldSnapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.training_rosters import training_rosters


@dataclass(frozen=True)
class _Job:
    root: int
    action: int
    suffix: int
    closed_loop: bool = False


def select_mass_actions(
    mass: np.ndarray, legal: np.ndarray, b8_actions: Sequence[int]
) -> np.ndarray:
    """Select maximum mean H300 mass; exact ties prefer B8, then lowest ID."""
    values = np.asarray(mass, dtype=np.float64)
    masks = np.asarray(legal, dtype=bool)
    if values.ndim == 3:
        values = values.mean(axis=2)
    if values.shape != masks.shape or values.shape[1:] != (6,):
        raise ValueError("expected mass and mask with shape (roots,6)")
    chosen = []
    for row, mask, baseline in zip(values, masks, b8_actions):
        ids = np.flatnonzero(mask)
        if ids.size == 0 or not np.isfinite(row[mask]).all():
            raise ValueError("empty mask or nonfinite legal return")
        ties = ids[row[ids] == row[ids].max()]
        chosen.append(int(baseline) if int(baseline) in ties else int(ties[0]))
    return np.asarray(chosen, dtype=np.int64)


@torch.inference_mode()
def student_actions(sim: Any, student: nn.Module) -> torch.Tensor:
    """Run the bare final spatial actor with the canonical legal-action fallback."""
    from research.spatial_student_pilot.observation import build_observation, compact_from_sim

    obs = build_observation(compact_from_sim(sim), device=sim.device)
    logits = student(*obs)
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.shape != (sim.E, 6) or not bool(torch.isfinite(logits).all()):
        raise ValueError("student produced nonfinite or malformed policy logits")
    mask = allowed_mask(sim.advisory_mask()[:, 0])
    return logits.masked_fill(~mask, -torch.inf).argmax(1)


def _new_sim(cfg: Any, count: int, device: Any) -> CTDeviceSim:
    dev = torch.device(device)
    sim = CTDeviceSim(
        cfg.sim_config(count),
        device=dev,
        train_mode=False,
        allow_respawn=True,
        move_backend="indexed" if dev.type == "cuda" else "reference",
    )
    r2_cache.enable(sim)
    return sim


def _check_roots(roots: Sequence[Mapping[str, Any]]) -> None:
    uids = [str(r["uid"]) for r in roots]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate root UID")
    for root in roots:
        mask = np.asarray(root["mask"], dtype=bool)
        if mask.shape != (6,) or not mask.any() or not mask[int(root["b8_action"])]:
            raise ValueError("invalid root mask or B8 action")
        if not root["snapshot"]["prepared"]:
            raise ValueError("search requires the prepared root seam")


@torch.inference_mode()
def _rollouts(
    roots: Sequence[Mapping[str, Any]],
    jobs: Sequence[_Job],
    *,
    cfg: Any,
    b8: nn.Module,
    networks: Mapping[str, nn.Module],
    device: Any,
    student: nn.Module | None = None,
    stop: Callable[[], bool] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Run bounded slabs, restoring all world/opponent state before CRN rekeying."""
    for job in jobs:
        mask = np.asarray(roots[job.root]["mask"], dtype=bool)
        if job.action < 0 or job.action >= 6 or not mask[job.action]:
            raise ValueError("illegal forced first action")
    rosters = training_rosters(cfg.worlds)
    horizon = int(cfg.branch_horizon)
    mass_out = np.zeros(len(jobs), dtype=np.float64)
    alive_out = np.zeros(len(jobs), dtype=np.float64)
    executed = 0
    action_counts = torch.zeros(6, dtype=torch.long, device=device)
    cap = int(cfg.branch_batch_size)
    if cap < 1 or horizon < 1:
        raise ValueError("positive slab size and horizon required")
    last_progress = time.monotonic()
    for start in range(0, len(jobs), cap):
        if stop is not None and stop():
            raise InterruptedError("incomplete search slab; caller must not publish unit")
        slab = jobs[start : start + cap]
        sim = _new_sim(cfg, len(slab), device)
        opponents = CTOpponentRuntime(
            sim,
            [rosters[int(roots[j.root]["roster_index"])] for j in slab],
            b8_network=b8,
            checkpoint_networks=networks,
        )
        restore_world_snapshots_into(
            [CompactWorldSnapshot.from_state_dict(roots[j.root]["snapshot"]) for j in slab],
            sim,
            opponents,
        )
        # Restoring a root must preserve the captured decision and legal mask.
        _, _, _, mask, _, actual = observe_hero_bv8(sim, b8)
        expected_mask = torch.as_tensor(
            np.asarray([roots[j.root]["mask"] for j in slab]), device=sim.device
        ).bool()
        expected_b8 = torch.tensor(
            [int(roots[j.root]["b8_action"]) for j in slab], device=sim.device
        )
        if not torch.equal(mask, expected_mask) or not torch.equal(actual, expected_b8):
            raise ValueError("restored root observation/mask/B8 action identity failure")
        keys = [derive_branch_keys(str(roots[j.root]["uid"]), j.suffix) for j in slab]
        sim.rng_key.copy_(torch.tensor([k.sim_rng_key for k in keys], device=sim.device))
        sim.rng_counter.zero_()
        opponents.rekey_branches(torch.tensor([k.opp_rng_key for k in keys], device=sim.device))
        sim.terminal_slots[:, 0] = True
        slots = torch.arange(len(slab), device=sim.device)
        mass_sum = torch.zeros(len(slab), dtype=torch.float64, device=sim.device)
        survivors = torch.zeros_like(mass_sum)
        closed = torch.tensor([j.closed_loop for j in slab], device=sim.device)
        for tick in range(horizon):
            now = time.monotonic()
            if progress is not None and now - last_progress >= 45.0:
                progress(
                    {
                        "phase": "branch",
                        "tick": tick,
                        "completed_jobs": start,
                        "total_jobs": len(jobs),
                        "executed_steps": executed,
                    }
                )
                last_progress = now
            if stop is not None and tick % 10 == 0 and stop():
                raise InterruptedError("incomplete rollout; caller must not publish unit")
            if tick == 0:
                actions = torch.tensor([j.action for j in slab], device=sim.device)
            else:
                if tick % 10 == 0 and not bool(sim.alive[:, 0].all()):
                    keep = sim.alive[:, 0].nonzero(as_tuple=False).flatten()
                    if keep.numel() == 0:
                        break
                    _compact_sim_in_place(sim, keep)
                    opponents.compact_rows(keep)
                    slots = slots.index_select(0, keep)
                    closed = closed.index_select(0, keep)
                sim.prepare()
                actions = observe_hero_bv8(sim, b8)[-1]
                if bool(closed.any()):
                    if student is None:
                        raise ValueError("closed-loop rollout requires final student")
                    spatial_actions = student_actions(sim, student)
                    actions = torch.where(closed, spatial_actions, actions)
            action_counts += torch.bincount(actions[closed & sim.alive[:, 0]], minlength=6)
            sim.step(opponents.actions(sim, actions))
            opponents.after_step(sim)
            if bool(sim.capacity_overflow.any()):
                raise ValueError("body capacity overflow invalidates rollout")
            executed += sim.E
            alive = sim.alive[:, 0]
            mass_sum.index_add_(0, slots, torch.where(alive, sim.length[:, 0], 0).double())
            if tick == horizon - 1:
                survivors.index_copy_(0, slots, alive.double())
        mass_out[start : start + len(slab)] = (mass_sum / horizon).cpu().numpy()
        alive_out[start : start + len(slab)] = survivors.cpu().numpy()
        if progress is not None:
            progress(
                {
                    "phase": "branch",
                    "completed": start + len(slab),
                    "total": len(jobs),
                    "executed_steps": executed,
                }
            )
    return mass_out, alive_out, executed, action_counts.cpu().numpy()


def _freeze_hash(roots: Sequence[Mapping[str, Any]], actions: Sequence[int]) -> str:
    payload = [(str(r["uid"]), int(a)) for r, a in zip(roots, actions)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def label_roots(
    roots: Sequence[Mapping[str, Any]],
    *,
    cfg: Any,
    b8: nn.Module,
    networks: Mapping[str, nn.Module],
    device: Any,
    stop: Any = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Train labels consume selection suffixes 0,1 only; score suffixes never enter fit."""
    _check_roots(roots)
    prefix = f"{cfg.namespace}/a{cfg.allocation_index}/"
    if any(not str(root["uid"]).startswith(prefix) for root in roots):
        raise ValueError("foreign root namespace/allocation prefix")
    if tuple(cfg.selection_suffixes) != (0, 1):
        raise ValueError("frozen selection suffix identities are 0,1")
    jobs = [
        _Job(i, a, k)
        for i, r in enumerate(roots)
        for a in np.flatnonzero(np.asarray(r["mask"], bool))
        for k in (0, 1)
    ]
    mass, alive, _, _ = _rollouts(
        roots,
        jobs,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        stop=stop,
        progress=progress,
    )
    masses = np.full((len(roots), 6, 2), np.nan)
    survival = masses.copy()
    for j, m, s in zip(jobs, mass, alive):
        masses[j.root, j.action, j.suffix] = m
        survival[j.root, j.action, j.suffix] = s
    chosen = select_mass_actions(
        masses, np.asarray([r["mask"] for r in roots]), [r["b8_action"] for r in roots]
    )
    freeze = _freeze_hash(roots, chosen)
    return [
        dict(
            r,
            target_action=int(a),
            selection_mass=m.tolist(),
            selection_survival=s.tolist(),
            selection_freeze_sha256=freeze,
        )
        for r, a, m, s in zip(roots, chosen, masses, survival)
    ]


@torch.inference_mode()
def assess_roots(
    roots: Sequence[Mapping[str, Any]],
    student: nn.Module,
    *,
    cfg: Any,
    b8: nn.Module,
    networks: Mapping[str, nn.Module],
    device: Any,
    stop: Any = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Freeze teacher on 0,1 then score frozen first actions independently on 4,5."""
    if any(r["partition"] != "assessment" for r in roots):
        raise ValueError("assessment scorer refuses training roots")
    if tuple(cfg.scoring_suffixes) != (4, 5):
        raise ValueError("frozen scoring suffix identities are 4,5")
    labels = label_roots(
        roots, cfg=cfg, b8=b8, networks=networks, device=device, stop=stop, progress=progress
    )
    rosters = training_rosters(cfg.worlds)
    spatial = []
    for start in range(0, len(roots), int(cfg.branch_batch_size)):
        chunk = roots[start : start + int(cfg.branch_batch_size)]
        sim = _new_sim(cfg, len(chunk), device)
        opponents = CTOpponentRuntime(
            sim,
            [rosters[int(r["roster_index"])] for r in chunk],
            b8_network=b8,
            checkpoint_networks=networks,
        )
        restore_world_snapshots_into(
            [CompactWorldSnapshot.from_state_dict(r["snapshot"]) for r in chunk],
            sim,
            opponents,
        )
        spatial.extend(student_actions(sim, student).cpu().tolist())
    jobs, descriptors = [], []
    for i, (root, label, action) in enumerate(zip(roots, labels, spatial)):
        for arm, first, closed in (
            ("b8", root["b8_action"], False),
            ("teacher", label["target_action"], False),
            ("student", action, False),
            ("closed_loop", action, True),
        ):
            for k in (4, 5):
                jobs.append(_Job(i, int(first), k, closed))
                descriptors.append((i, arm))
    mass, alive, _, action_counts = _rollouts(
        roots,
        jobs,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        student=student,
        stop=stop,
        progress=progress,
    )
    result = [
        dict(
            uid=r["uid"],
            partition="assessment",
            world_id=int(r["world_id"]),
            kind=r["kind"],
            mix=r["mix"],
            b8_action=int(r["b8_action"]),
            teacher_action=int(l["target_action"]),
            student_action=int(a),
            selection_freeze_sha256=l["selection_freeze_sha256"],
            scoring_suffixes=[4, 5],
        )
        for r, l, a in zip(roots, labels, spatial)
    ]
    for (i, arm), m, s in zip(descriptors, mass, alive):
        result[i].setdefault(arm + "_mass", []).append(float(m))
        result[i].setdefault(arm + "_survival", []).append(float(s))
    if result:
        result[0]["closed_loop_action_counts"] = action_counts.tolist()
    return result


@torch.inference_mode()
def paired_gameplay(
    worlds: Sequence[Mapping[str, Any]],
    student: nn.Module,
    *,
    cfg: Any,
    b8: nn.Module,
    networks: Mapping[str, nn.Module],
    device: Any,
    stop: Any = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Fresh paired H5000 worlds: terminal hero; mass and alive use the full horizon."""
    if len({int(w["world_id"]) for w in worlds}) != len(worlds):
        raise ValueError("duplicate paired gameplay world")
    roster_table = training_rosters(cfg.gameplay_worlds)
    output = [dict(world_id=int(w["world_id"]), mix=w["mix"]) for w in worlds]
    last_progress = time.monotonic()
    for start in range(0, len(worlds), int(cfg.world_batch_size)):
        chunk = worlds[start : start + int(cfg.world_batch_size)]
        for arm in ("b8", "student"):
            arm_started = time.monotonic()
            executed_world_steps = 0
            sim = _new_sim(cfg, len(chunk), device)
            sim.rng_key.copy_(
                torch.tensor(
                    [
                        int(w.get("seed", cfg.world_seed(int(w["world_id"]), "gameplay")))
                        for w in chunk
                    ],
                    device=sim.device,
                )
            )
            sim.rng_counter.zero_()
            sim.alive.fill_(False)
            sim.length.zero_()
            sim.seg_count.zero_()
            sim.head_ptr.zero_()
            sim.boost_frames.zero_()
            sim.frames_since_food.zero_()
            sim.terminal_slots.fill_(False)
            sim.reset_fresh_worlds()
            sim.terminal_slots[:, 0] = True
            opponents = CTOpponentRuntime(
                sim,
                [roster_table[int(w["roster_index"])] for w in chunk],
                b8_network=b8,
                checkpoint_networks=networks,
                opp_rng_keys=[
                    int(w.get("opp_seed", cfg.opp_seed(int(w["world_id"]), "gameplay")))
                    for w in chunk
                ],
            )
            for i in range(len(chunk)):
                digest = hashlib.sha256()
                for field in (
                    "bodies",
                    "head_ptr",
                    "seg_count",
                    "length",
                    "alive",
                    "direction",
                    "boost_frames",
                    "frames_since_food",
                    "respawn_timer",
                    "frame",
                    "rng_key",
                    "rng_counter",
                    "food_cells",
                    "food_present",
                    "corpse_food",
                    "terminal_slots",
                ):
                    digest.update(field.encode())
                    digest.update(
                        getattr(sim, field)[i].detach().cpu().contiguous().numpy().tobytes()
                    )
                digest.update(str(opponents.rosters[i].identity).encode())
                digest.update(opponents.rng_key[i].detach().cpu().numpy().tobytes())
                output[start + i][arm + "_initial_state_sha256"] = digest.hexdigest()
            total_mass = torch.zeros(len(chunk), dtype=torch.float64, device=sim.device)
            total_alive = torch.zeros_like(total_mass)
            counts = torch.zeros(6, dtype=torch.long, device=sim.device)
            slots = torch.arange(len(chunk), device=sim.device)
            for tick in range(int(cfg.horizon)):
                now = time.monotonic()
                if progress is not None and now - last_progress >= 45.0:
                    progress(
                        {
                            "phase": "gameplay",
                            "arm": arm,
                            "tick": tick,
                            "completed_worlds": start,
                            "total_worlds": len(worlds),
                        }
                    )
                    last_progress = now
                if stop is not None and tick % 10 == 0 and stop():
                    raise InterruptedError("incomplete gameplay; no unit may be published")
                if tick % 10 == 0 and not bool(sim.alive[:, 0].all()):
                    keep = sim.alive[:, 0].nonzero(as_tuple=False).flatten()
                    if keep.numel() == 0:
                        break
                    _compact_sim_in_place(sim, keep)
                    opponents.compact_rows(keep)
                    slots = slots.index_select(0, keep)
                sim.prepare()
                actions = (
                    observe_hero_bv8(sim, b8)[-1] if arm == "b8" else student_actions(sim, student)
                )
                active_actions = actions[sim.alive[:, 0]]
                counts += torch.bincount(active_actions, minlength=6)
                sim.step(opponents.actions(sim, actions))
                opponents.after_step(sim)
                executed_world_steps += sim.E
                if bool(sim.capacity_overflow.any()):
                    raise ValueError("body capacity overflow invalidates gameplay")
                alive = sim.alive[:, 0]
                total_mass.index_add_(0, slots, torch.where(alive, sim.length[:, 0], 0).double())
                total_alive.index_add_(0, slots, alive.double())
            for i, (m, a) in enumerate(zip(total_mass.cpu(), total_alive.cpu())):
                output[start + i][arm + "_mass"] = float(m / cfg.horizon)
                output[start + i][arm + "_alive"] = float(a / cfg.horizon)
            # Batch counts are emitted once and summed exactly by readout.
            output[start][arm + "_action_counts"] = counts.cpu().tolist()
            output[start][arm + "_executed_world_steps"] = executed_world_steps
            output[start][arm + "_wall_seconds"] = time.monotonic() - arm_started
            if progress is not None:
                progress(
                    {
                        "phase": "gameplay",
                        "arm": arm,
                        "completed": start + len(chunk),
                        "total": len(worlds),
                    }
                )
    return output
