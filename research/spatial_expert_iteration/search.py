"""Student-continuation search, versioned labels, restore-once device fan-out."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_branching import _compact_sim_in_place
from research.sandbox_a_resident.ct_mechanics import CTOpponentRuntime, derive_branch_keys
from research.sandbox_a_resident.ct_snapshots import (
    CompactWorldSnapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.search import (
    _check_roots,
    _new_sim,
    select_mass_actions,
    student_actions,
    verify_restored_action,
)

from .config import RoundConfig, identity


def tensor_identity(value: Any) -> str:
    """Stable bytes identity for model, snapshot and opponent carry trees."""
    h = hashlib.sha256()

    def visit(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            h.update(str((obj.dtype.str, obj.shape)).encode())
            h.update(obj.tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj):
                h.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(str(len(obj)).encode())
            for item in obj:
                visit(item)
        else:
            h.update(repr(obj).encode())
        h.update(b"\x00")

    visit(value)
    return h.hexdigest()


def label_key(
    root: dict[str, Any], cfg: RoundConfig, continuation_sha: str, opponents_sha: str
) -> str:
    return identity(
        dict(
            root_uid=root["uid"],
            state=tensor_identity(root["snapshot"]),
            continuation_sha256=continuation_sha,
            opponents_sha256=opponents_sha,
            horizon=cfg.branch_horizon,
            suffix_namespace=cfg.namespace,
            suffixes=cfg.selection_suffixes,
            tie="current-student-then-lowest",
            config_sha256=cfg.sha256,
        )
    )


@dataclass(frozen=True)
class Job:
    root: int
    action: int
    suffix: int


@torch.inference_mode()
def rollouts(
    roots: Sequence[dict[str, Any]],
    jobs: Sequence[Job],
    *,
    cfg: RoundConfig,
    student: Any,
    b8: Any,
    networks: dict[str, Any],
    device: str,
    stop: Callable[[], bool] = lambda: False,
    progress: Callable[[dict[str, Any]], None] = lambda _: None,
    fanout: bool = True,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Restore/verify unique roots once, then index-select full device states.

    The reference `fanout=False` restores every duplicate branch independently.
    Opponent carry is copied into a new runtime per slab (never compact_rows with
    repeated indices: that helper assumes unique survivors). No RNG prefix reuse,
    action pruning, precision conversion or terminal-lane refill is performed.
    """
    _check_roots(roots)
    table = training_rosters(max(cfg.worlds, cfg.assessment_worlds))

    def restored(selected: Sequence[dict[str, Any]]) -> tuple[Any, Any]:
        sim = _new_sim(cfg, len(selected), device)
        opponents = CTOpponentRuntime(
            sim,
            [table[int(r["roster_index"])] for r in selected],
            b8_network=b8,
            checkpoint_networks=networks,
        )
        restore_world_snapshots_into(
            [CompactWorldSnapshot.from_state_dict(r["snapshot"]) for r in selected], sim, opponents
        )
        _, _, _, masks, q, actual = observe_hero_bv8(sim, b8)
        verify_restored_action(
            masks,
            torch.stack([r["mask"] for r in selected]).to(device),
            q,
            actual,
            torch.tensor([r["b8_action"] for r in selected], device=device),
        )
        return sim, opponents

    if not roots:
        return np.empty(0), np.empty(0), []
    base, base_opponents = restored(roots)
    first = student_actions(base, student).cpu().tolist()
    # Only small opponent carry is decoded on host once per canonical root.
    carries = [base_opponents.snapshot_row(i) for i in range(len(roots))]
    mass = np.zeros(len(jobs), dtype=np.float64)
    survival = np.zeros(len(jobs), dtype=np.float64)
    for begin in range(0, len(jobs), cfg.branch_batch_size):
        if stop():
            raise InterruptedError("partial search slab")
        slab = jobs[begin : begin + cfg.branch_batch_size]
        if any(
            j.root < 0
            or j.root >= len(roots)
            or not 0 <= j.action < 6
            or not bool(roots[j.root]["mask"][j.action])
            for j in slab
        ):
            raise ValueError("illegal branch job")
        selected = [roots[j.root] for j in slab]
        if fanout:
            # Copy the base object, then index-select every environment tensor.
            # _compact_sim_in_place allocates independent tensors including bodies,
            # RNG counters, food, growth, terminal flags and all clocks.
            import copy

            sim = copy.copy(base)
            r2_cache.inherit(sim, base)
            _compact_sim_in_place(sim, torch.tensor([j.root for j in slab], device=device))
            opponents = CTOpponentRuntime(
                sim,
                [table[int(r["roster_index"])] for r in selected],
                b8_network=b8,
                checkpoint_networks=networks,
            )
            opponents.load_row_snapshots([carries[j.root] for j in slab])
        else:
            sim, opponents = restored(selected)
        keys = [
            derive_branch_keys(f"{cfg.namespace}/{r['uid']}", j.suffix)
            for r, j in zip(selected, slab)
        ]
        sim.rng_key.copy_(torch.tensor([k.sim_rng_key for k in keys], device=device))
        sim.rng_counter.zero_()
        opponents.rekey_branches(torch.tensor([k.opp_rng_key for k in keys], device=device))
        sim.terminal_slots[:, 0] = True
        total = torch.zeros(len(slab), dtype=torch.float64, device=device)
        for tick in range(cfg.branch_horizon):
            if stop():
                raise InterruptedError("partial student-continuation search")
            if tick:
                sim.prepare()
                actions = student_actions(sim, student)
            else:
                actions = torch.tensor([j.action for j in slab], device=device)
            sim.step(opponents.actions(sim, actions))
            opponents.after_step(sim)
            if bool(sim.capacity_overflow.any()):
                raise ValueError("branch body overflow")
            total += sim.length[:, 0] * sim.alive[:, 0]
            progress(dict(stage="search", slab=begin, tick=tick + 1, jobs=len(jobs)))
        if trace is not None:
            for i, job in enumerate(slab):
                trace.append(
                    dict(
                        job=job,
                        state=tensor_identity(
                            {name: getattr(sim, name)[i] for name in _ENV_TENSOR_FIELDS}
                        ),
                        opponent=tensor_identity(opponents.snapshot_row(i)),
                    )
                )
        mass[begin : begin + len(slab)] = (total / cfg.branch_horizon).cpu().numpy()
        survival[begin : begin + len(slab)] = sim.alive[:, 0].cpu().numpy()
    return mass, survival, first


def label_roots(
    roots: Sequence[dict[str, Any]],
    *,
    cfg: RoundConfig,
    student: Any,
    continuation_sha: str,
    opponents_sha: str,
    b8: Any,
    networks: dict[str, Any],
    device: str,
    stop: Callable[[], bool] = lambda: False,
    progress: Callable[[dict[str, Any]], None] = lambda _: None,
) -> list[dict[str, Any]]:
    jobs = [
        Job(i, int(a), k)
        for i, r in enumerate(roots)
        for a in np.flatnonzero(np.asarray(r["mask"], bool))
        for k in cfg.selection_suffixes
    ]
    if not roots:
        return []
    mass, survival, first = rollouts(
        roots,
        jobs,
        cfg=cfg,
        student=student,
        b8=b8,
        networks=networks,
        device=device,
        stop=stop,
        progress=progress,
    )
    masses = np.full((len(roots), 6, 2), np.nan)
    alive = masses.copy()
    for j, m, s in zip(jobs, mass, survival):
        masses[j.root, j.action, cfg.selection_suffixes.index(j.suffix)] = m
        alive[j.root, j.action, cfg.selection_suffixes.index(j.suffix)] = s
    actions = select_mass_actions(masses, np.asarray([r["mask"] for r in roots]), first)
    return [
        dict(
            r,
            target_action=int(a),
            teacher_q=m.mean(1),
            selection_mass=m,
            selection_survival=s,
            continuation_sha256=continuation_sha,
            label_key=label_key(r, cfg, continuation_sha, opponents_sha),
        )
        for r, a, m, s in zip(roots, actions, masses, alive)
    ]


def assess(
    roots: Sequence[dict[str, Any]],
    *,
    cfg: RoundConfig,
    current: Any,
    candidate: Any,
    continuation_sha: str,
    opponents_sha: str,
    b8: Any,
    networks: dict[str, Any],
    device: str,
    stop: Callable[[], bool] = lambda: False,
    progress: Callable[[dict[str, Any]], None] = lambda _: None,
) -> list[dict[str, Any]]:
    """Independent suffixes score frozen teacher/current/candidate first actions.

    All first-action arms follow the frozen pre-round student. Candidate closed
    loop efficacy is measured separately by fresh paired_gameplay against B8.
    Identical actions share one rollout; descriptors retain paired parent IDs.
    """
    if any(r["partition"] != "assessment" for r in roots):
        raise ValueError("assessment requires fresh held-out roots")
    labels = label_roots(
        roots,
        cfg=cfg,
        student=current,
        continuation_sha=continuation_sha,
        opponents_sha=opponents_sha,
        b8=b8,
        networks=networks,
        device=device,
        stop=stop,
        progress=progress,
    )
    # Obtain candidate actions through the same restoration verification path.
    _, _, actions = rollouts(
        roots,
        [],
        cfg=cfg,
        student=candidate,
        b8=b8,
        networks=networks,
        device=device,
        stop=stop,
        progress=progress,
    )
    unique: dict[Job, int] = {}
    descriptors = []
    for i, (r, label, action) in enumerate(zip(roots, labels, actions)):
        for arm, first in (
            ("b8", r["b8_action"]),
            ("current", r["student_action"]),
            ("teacher", label["target_action"]),
            ("candidate", action),
        ):
            for suffix in cfg.scoring_suffixes:
                job = Job(i, int(first), suffix)
                if job not in unique:
                    unique[job] = len(unique)
                descriptors.append((i, arm, unique[job]))
    mass, survival, _ = rollouts(
        roots,
        list(unique),
        cfg=cfg,
        student=current,
        b8=b8,
        networks=networks,
        device=device,
        stop=stop,
        progress=progress,
    )
    result = [
        dict(
            uid=r["uid"],
            parent_id=r["parent_id"],
            partition="assessment",
            label_key=l["label_key"],
            scoring_suffixes=cfg.scoring_suffixes,
        )
        for r, l in zip(roots, labels)
    ]
    for i, arm, j in descriptors:
        result[i].setdefault(arm + "_mass", []).append(float(mass[j]))
        result[i].setdefault(arm + "_survival", []).append(float(survival[j]))
    return result
