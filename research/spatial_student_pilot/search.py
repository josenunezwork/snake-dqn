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
    derive_branch_keys,
)
from research.sandbox_a_resident.ct_snapshots import (
    CompactWorldSnapshot,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.batching import BatchedOpponents, observe_hero_bv8, stable_q


@dataclass(frozen=True)
class _Job:
    root: int
    action: int
    suffix: int
    closed_loop: bool = False


# Prospective restore-verification tolerance; never changes the forced tick-0 action.
RESTORE_Q_EPSILON = 1e-5


def _sync(device: Any) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def verify_restored_action(
    mask: torch.Tensor,
    expected_mask: torch.Tensor,
    q: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> list[dict[str, Any]]:
    """Permit only a top-two near tie involving the captured legal action."""
    if not torch.equal(mask, expected_mask) or not bool(torch.isfinite(q).all()):
        raise ValueError("restored root observation/mask/B8 action identity failure")
    events = []
    for row in (actual != expected).nonzero(as_tuple=False).flatten().tolist():
        values = q[row].masked_fill(~mask[row], -torch.inf)
        top = values.topk(2).values
        gap = float(top[0] - values[int(expected[row])])
        margin = float(top[0] - top[1])
        if not bool(mask[row, expected[row]]) or max(gap, margin) > RESTORE_Q_EPSILON:
            raise ValueError("restored root observation/mask/B8 action identity failure")
        events.append(
            dict(
                row=row,
                expected=int(expected[row]),
                actual=int(actual[row]),
                top2_margin=margin,
                expected_gap=gap,
                epsilon=RESTORE_Q_EPSILON,
            )
        )
    return events


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
def student_actions(
    sim: Any,
    student: nn.Module,
    env_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Memory-bounded bare actor inference; original fp32 model and legal fallback."""
    from research.spatial_student_pilot.observation import build_observation, compact_from_sim

    ids = torch.arange(sim.E, device=sim.device) if env_indices is None else env_indices
    results = []
    for start in range(0, len(ids), 128):
        selected = ids[start : start + 128]
        obs = build_observation(compact_from_sim(sim, selected), device=sim.device)
        logits = student(*obs)
        if isinstance(logits, tuple):
            logits = logits[0]
        mask = allowed_mask(sim.advisory_mask()[selected, 0])
        logits = stable_q(student, obs, logits, mask)
        if logits.shape != (len(selected), 6) or not bool(torch.isfinite(logits).all()):
            raise ValueError("student produced nonfinite or malformed policy logits")
        results.append(logits.masked_fill(~mask, -torch.inf).argmax(1))
    return torch.cat(results) if results else torch.empty(0, device=sim.device, dtype=torch.long)


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
    order = sorted(
        range(len(jobs)),
        key=lambda i: (rosters[int(roots[jobs[i].root]["roster_index"])].identity, i),
    )
    ordered_jobs = [jobs[i] for i in order]
    for start in range(0, len(jobs), cap):
        if stop is not None and stop():
            raise InterruptedError("incomplete search slab; caller must not publish unit")
        slab = ordered_jobs[start : start + cap]
        sim = _new_sim(cfg, len(slab), device)
        opponents = BatchedOpponents(
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
        _, _, _, mask, q, actual = observe_hero_bv8(sim, b8)
        expected_mask = torch.as_tensor(
            np.asarray([roots[j.root]["mask"] for j in slab]), device=sim.device
        ).bool()
        expected_b8 = torch.tensor(
            [int(roots[j.root]["b8_action"]) for j in slab], device=sim.device
        )
        ties = verify_restored_action(mask, expected_mask, q, actual, expected_b8)
        if ties:
            event = {"phase": "restore_near_tie", "slab_start": start, "mismatches": ties}
            if progress is not None:
                progress(event)
            # Always retain diagnostics in stdout, even when progress is throttled.
            print(json.dumps(event, sort_keys=True), flush=True)
        keys = [derive_branch_keys(str(roots[j.root]["uid"]), j.suffix) for j in slab]
        sim.rng_key.copy_(torch.tensor([k.sim_rng_key for k in keys], device=sim.device))
        sim.rng_counter.zero_()
        opponents.rekey_branches(torch.tensor([k.opp_rng_key for k in keys], device=sim.device))
        sim.terminal_slots[:, 0] = True
        slots = torch.arange(len(slab), device=sim.device)
        mass_sum = torch.zeros(len(slab), dtype=torch.float64, device=sim.device)
        survivors = torch.zeros_like(mass_sum)
        closed = torch.tensor([j.closed_loop for j in slab], device=sim.device)
        _sync(sim.device)
        loop_started = time.monotonic()
        completed_ticks = 0
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
                if cfg.mode != "ruler" and tick % 10 == 0 and not bool(sim.alive[:, 0].all()):
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
                    selected = closed.nonzero(as_tuple=False).flatten()
                    spatial_actions = actions.clone()
                    spatial_actions[selected] = student_actions(sim, student, selected)
                    actions = torch.where(closed, spatial_actions, actions)
            action_counts += torch.bincount(actions[closed & sim.alive[:, 0]], minlength=6)
            sim.step(opponents.actions(sim, actions))
            opponents.after_step(sim)
            if bool(sim.capacity_overflow.any()):
                raise ValueError("body capacity overflow invalidates rollout")
            executed += sim.E
            completed_ticks += 1
            alive = sim.alive[:, 0]
            mass_sum.index_add_(0, slots, torch.where(alive, sim.length[:, 0], 0).double())
            if tick == horizon - 1:
                survivors.index_copy_(0, slots, alive.double())
        _sync(sim.device)
        loop_seconds = time.monotonic() - loop_started
        if progress is not None:
            progress(
                {
                    "phase": "branch_timing",
                    "slab_size": len(slab),
                    "ticks": completed_ticks,
                    "loop_seconds": loop_seconds,
                    "closed_loop": any(j.closed_loop for j in slab),
                }
            )
        mass_out[np.asarray(order[start : start + len(slab)])] = (mass_sum / horizon).cpu().numpy()
        alive_out[np.asarray(order[start : start + len(slab)])] = survivors.cpu().numpy()
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


def prepare_assessment(
    roots: Sequence[Mapping[str, Any]],
    *,
    cfg: Any,
    b8: nn.Module,
    networks: Mapping[str, nn.Module],
    device: Any,
    stop: Any = None,
    progress: Any = None,
    labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Student-independent selection and independent B8/teacher scoring, sealed."""
    if any(r["partition"] != "assessment" for r in roots):
        raise ValueError("assessment preparation refuses training roots")
    labels = (
        labels
        if labels is not None
        else label_roots(
            roots, cfg=cfg, b8=b8, networks=networks, device=device, stop=stop, progress=progress
        )
    )
    jobs = list(
        dict.fromkeys(
            _Job(i, int(first), suffix)
            for i, (r, label) in enumerate(zip(roots, labels))
            for first in (r["b8_action"], label["target_action"])
            for suffix in (4, 5)
        )
    )
    mass, alive, _, _ = _rollouts(
        roots, jobs, cfg=cfg, b8=b8, networks=networks, device=device, stop=stop, progress=progress
    )
    return dict(
        labels=labels,
        scores=[
            ((j.root, j.action, j.suffix, j.closed_loop), (float(m), float(a)))
            for j, m, a in zip(jobs, mass, alive)
        ],
    )


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
    prepared: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Freeze teacher on 0,1 then score frozen first actions independently on 4,5."""
    if any(r["partition"] != "assessment" for r in roots):
        raise ValueError("assessment scorer refuses training roots")
    if tuple(cfg.scoring_suffixes) != (4, 5):
        raise ValueError("frozen scoring suffix identities are 4,5")
    prepared = prepared or prepare_assessment(
        roots, cfg=cfg, b8=b8, networks=networks, device=device, stop=stop, progress=progress
    )
    labels = prepared["labels"]
    if [r["uid"] for r in labels] != [r["uid"] for r in roots]:
        raise ValueError("foreign prepared assessment roots")
    rosters = training_rosters(cfg.worlds)
    spatial = []
    for start in range(0, len(roots), int(cfg.branch_batch_size)):
        chunk = roots[start : start + int(cfg.branch_batch_size)]
        sim = _new_sim(cfg, len(chunk), device)
        opponents = BatchedOpponents(
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
    unique = list(dict.fromkeys(jobs))
    known = {tuple(key): value for key, value in prepared["scores"]}
    expected_keys = {
        (i, int(a), suffix, False)
        for i, (root, label) in enumerate(zip(roots, labels))
        for a in (root["b8_action"], label["target_action"])
        for suffix in (4, 5)
    }
    if set(known) != expected_keys or len(known) != len(prepared["scores"]):
        raise ValueError("foreign or duplicate independent scoring keys")
    missing = [j for j in unique if (j.root, j.action, j.suffix, j.closed_loop) not in known]
    mass, alive, _, action_counts = _rollouts(
        roots,
        missing,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        student=student,
        stop=stop,
        progress=progress,
    )
    for job, m, a in zip(missing, mass, alive):
        known[(job.root, job.action, job.suffix, job.closed_loop)] = (float(m), float(a))
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
    for job, (i, arm) in zip(jobs, descriptors):
        m, s = known[(job.root, job.action, job.suffix, job.closed_loop)]
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
    ruler_ticks: int | None = None,
    ruler_arm_seconds: float = 45.0,
) -> list[dict[str, Any]]:
    """Fresh paired H5000 worlds: terminal hero; mass and alive use the full horizon."""
    if ruler_ticks is not None and (cfg.mode != "ruler" or not 1 <= ruler_ticks <= 600):
        raise ValueError("bounded gameplay tick probe is ruler-only")
    if len({int(w["world_id"]) for w in worlds}) != len(worlds):
        raise ValueError("duplicate paired gameplay world")
    roster_table = training_rosters(cfg.gameplay_worlds)
    output = [dict(world_id=int(w["world_id"]), mix=w["mix"]) for w in worlds]
    last_progress = time.monotonic()
    for start in range(0, len(worlds), int(cfg.world_batch_size)):
        chunk = worlds[start : start + int(cfg.world_batch_size)]
        # Both arms share one sim allocation and tick loop, with duplicate keys
        # preserving exact paired initial state and counter-RNG streams.
        paired = [*chunk, *chunk]
        n = len(chunk)
        sim = _new_sim(cfg, 2 * n, device)
        sim.rng_key.copy_(
            torch.tensor(
                [
                    int(w.get("seed", cfg.world_seed(int(w["world_id"]), "gameplay")))
                    for w in paired
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
        opponents = BatchedOpponents(
            sim,
            [roster_table[int(w["roster_index"])] for w in paired],
            b8_network=b8,
            checkpoint_networks=networks,
            opp_rng_keys=[
                int(w.get("opp_seed", cfg.opp_seed(int(w["world_id"]), "gameplay"))) for w in paired
            ],
        )
        for i in range(2 * n):
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
                digest.update(getattr(sim, field)[i].detach().cpu().contiguous().numpy().tobytes())
            digest.update(str(opponents.rosters[i].identity).encode())
            digest.update(opponents.rng_key[i].detach().cpu().numpy().tobytes())
            arm = "b8" if i < n else "student"
            output[start + i % n][arm + "_initial_state_sha256"] = digest.hexdigest()
        total_mass = torch.zeros(2 * n, dtype=torch.float64, device=sim.device)
        total_alive = torch.zeros_like(total_mass)
        counts = torch.zeros((2, 6), dtype=torch.long, device=sim.device)
        executed = torch.zeros(2, dtype=torch.long, device=sim.device)
        slots = torch.arange(2 * n, device=sim.device)
        closed = slots >= n
        _sync(sim.device)
        loop_started = time.monotonic()
        completed_ticks = 0
        for tick in range(int(cfg.horizon) if ruler_ticks is None else ruler_ticks):
            now = time.monotonic()
            if ruler_ticks is not None and tick > 0 and now - loop_started >= ruler_arm_seconds:
                break
            if progress is not None and now - last_progress >= 45.0:
                progress(
                    dict(
                        phase="gameplay",
                        arm="paired",
                        tick=tick,
                        completed_worlds=start,
                        total_worlds=len(worlds),
                    )
                )
                last_progress = now
            if stop is not None and tick % 10 == 0 and stop():
                raise InterruptedError("incomplete gameplay; no unit may be published")
            if ruler_ticks is None and tick % 10 == 0 and not bool(sim.alive[:, 0].all()):
                keep = sim.alive[:, 0].nonzero(as_tuple=False).flatten()
                if keep.numel() == 0:
                    break
                _compact_sim_in_place(sim, keep)
                opponents.compact_rows(keep)
                slots = slots.index_select(0, keep)
                closed = closed.index_select(0, keep)
            sim.prepare()
            actions = observe_hero_bv8(sim, b8)[-1]
            selected = closed.nonzero(as_tuple=False).flatten()
            spatial = actions.clone()
            if selected.numel():
                spatial[selected] = student_actions(sim, student, selected)
            actions = torch.where(closed, spatial, actions)
            for arm_index, selected_arm in enumerate((~closed, closed)):
                counts[arm_index] += torch.bincount(
                    actions[selected_arm & sim.alive[:, 0]], minlength=6
                )
                executed[arm_index] += selected_arm.sum()
            sim.step(opponents.actions(sim, actions))
            opponents.after_step(sim)
            completed_ticks += 1
            if bool(sim.capacity_overflow.any()):
                raise ValueError("body capacity overflow invalidates gameplay")
            alive = sim.alive[:, 0]
            total_mass.index_add_(0, slots, torch.where(alive, sim.length[:, 0], 0).double())
            total_alive.index_add_(0, slots, alive.double())
        _sync(sim.device)
        seconds = time.monotonic() - loop_started
        for arm_index, arm in enumerate(("b8", "student")):
            begin = arm_index * n
            for i in range(n):
                output[start + i][arm + "_mass"] = float(total_mass[begin + i] / cfg.horizon)
                output[start + i][arm + "_alive"] = float(total_alive[begin + i] / cfg.horizon)
            output[start][arm + "_action_counts"] = counts[arm_index].cpu().tolist()
            output[start][arm + "_executed_world_steps"] = int(executed[arm_index])
            output[start][arm + "_ticks"] = completed_ticks
            # Same loop price for both; consumers must charge it only once.
            output[start][arm + "_loop_seconds"] = seconds
            output[start][arm + "_slab_size"] = n
            output[start][arm + "_wall_seconds"] = time.monotonic() - loop_started
        output[start]["paired_loop_seconds"] = seconds
        output[start]["paired_slab_size"] = 2 * n
    return output
