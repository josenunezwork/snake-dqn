"""Natural terminal B8 collection with bounded outcome-independent sparse sampling."""

from __future__ import annotations

import hashlib
import heapq
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_snapshots import (
    CompactWorldSnapshot,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.batching import BatchedOpponents, capture_rows, observe_hero_bv8
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.mechanics import PilotCTDeviceSim as CTDeviceSim


def _priority(cfg: PilotConfig, world: int, tick: int, kind: str) -> int:
    raw = f"{cfg.namespace}/{cfg.allocation_index}/{world}/{tick}/{kind}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _cpu_compact(compact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: val.detach().cpu().clone() if isinstance(val, torch.Tensor) else val
        for key, val in compact.items()
    }


@torch.inference_mode()
def collect_batch(
    cfg: PilotConfig,
    world_ids: Sequence[int],
    *,
    b8: nn.Module,
    networks: Mapping[str, nn.Module],
    device: str,
    stop: Callable[[], bool],
    progress: Callable[[dict[str, Any]], None],
    ruler_seconds: float | None = None,
    ruler_ticks: int | None = None,
    ruler_fixtures: Sequence[dict[str, Any]] | None = None,
    ruler_roster_indices: Sequence[int] | None = None,
    ruler_census: Any = None,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """A unit is one complete fixed parent-world batch; no replacement episodes."""
    started = time.monotonic()
    dev = torch.device(device)
    ids = list(world_ids)
    if (
        any(
            v is not None
            for v in (
                ruler_seconds,
                ruler_ticks,
                ruler_fixtures,
                ruler_roster_indices,
                ruler_census,
            )
        )
        and cfg.mode != "ruler"
    ):
        raise ValueError("historical fixtures/tick probes are ruler-only")
    if ruler_fixtures is not None and len(ruler_fixtures) != len(ids):
        raise ValueError("fixture count must equal ruler slab size")
    sim = CTDeviceSim(
        cfg.sim_config(len(ids)),
        device=dev,
        train_mode=False,
        allow_respawn=True,
        move_backend="indexed" if dev.type == "cuda" else "reference",
    )
    r2_cache.enable(sim)
    if ruler_census is not None:
        from research.spatial_student_pilot.ruler_cells import DiagnosticTiming

        sim._timing = DiagnosticTiming(dev, ruler_census)
    sim.rng_key.copy_(torch.tensor([cfg.world_seed(i) for i in ids], device=dev))
    sim.rng_counter.zero_()
    sim.alive.fill_(False)
    sim.length.fill_(1)
    sim._reward_prev_length.fill_(1)
    sim.terminal_slots.fill_(False)
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    roster_table = training_rosters(cfg.worlds)
    roster_indices = [r["roster_index"] for r in ruler_fixtures] if ruler_fixtures else ids
    if ruler_roster_indices is not None:
        if len(ruler_roster_indices) != len(ids):
            raise ValueError("ruler roster count mismatch")
        roster_indices = list(ruler_roster_indices)
    rosters = [roster_table[i] for i in roster_indices]
    opponents = BatchedOpponents(
        sim,
        rosters,
        b8_network=b8,
        checkpoint_networks=networks,
        opp_rng_keys=[cfg.opp_seed(i) for i in ids],
    )
    if ruler_fixtures is not None:
        restore_world_snapshots_into(
            [CompactWorldSnapshot.from_state_dict(r["snapshot"]) for r in ruler_fixtures],
            sim,
            opponents,
        )
        sim.terminal_slots[:, 0] = True
    partitions = cfg.partitions()
    # Equal quota per fixed collection batch; final global priority selection
    # cannot overrepresent a batch due to differing natural survival duration.
    batch_count = (cfg.worlds + cfg.collection_unit_worlds - 1) // cfg.collection_unit_worlds
    demo_cap = (cfg.demos + batch_count - 1) // batch_count
    demos: dict[int, list[tuple[int, int, dict[str, Any]]]] = {}
    roots: dict[tuple[int, str], list[tuple[int, int, dict[str, Any]]]] = {}
    decisions = 0
    decisions_by_world = {world: 0 for world in ids}
    completed_ticks = 0
    last_progress = started
    tick_samples = []
    reached_l600: set[int] = set()
    start_tick = 0
    if resume_state is not None:
        if resume_state["world_ids"] != ids or resume_state["config_sha256"] != cfg.sha256:
            raise ValueError("foreign collection resume snapshot")
        start_tick = resume_state["next_tick"]
        if not 0 < start_tick < cfg.horizon or start_tick % cfg.collection_checkpoint_every:
            raise ValueError("ineligible collection resume boundary")
        restore_world_snapshots_into(
            [CompactWorldSnapshot.from_state_dict(r) for r in resume_state["snapshots"]],
            sim,
            opponents,
        )
        demos, roots = resume_state["demos"], resume_state["roots"]
        decisions = resume_state["decisions"]
        decisions_by_world = resume_state["decisions_by_world"]
        tick_samples = resume_state["tick_samples"]
        reached_l600 = set(resume_state["reached_l600_world_ids"])
        completed_ticks = start_tick
    decision_counts = torch.tensor([decisions_by_world[w] for w in ids], device=dev)
    reached = torch.tensor([w in reached_l600 for w in ids], device=dev)
    pending_samples = []

    def flush_metrics() -> None:
        nonlocal decisions
        counts = decision_counts.cpu().tolist()
        decisions = sum(counts)
        decisions_by_world.update(zip(ids, counts))
        reached_l600.update(w for w, yes in zip(ids, reached.cpu().tolist()) if yes)
        if pending_samples:
            metrics = torch.stack([item[2] for item in pending_samples]).cpu().tolist()
            for (tick_id, seconds, _), values in zip(pending_samples, metrics):
                tick_samples.append(
                    dict(
                        tick=tick_id,
                        seconds=seconds,
                        mature_worlds=int(values[0]),
                        living_worlds=int(values[1]),
                        mean_hero_length=values[2] / len(ids),
                        max_hero_length=int(values[3]),
                    )
                )
            pending_samples.clear()

    for tick in range(start_tick, cfg.horizon if ruler_ticks is None else ruler_ticks):
        if ruler_seconds is not None and time.monotonic() - started >= ruler_seconds:
            break
        if stop():
            raise InterruptedError("partial collection batch; no result unit published")
        if dev.type == "cuda" and cfg.mode == "ruler":
            torch.cuda.synchronize(dev)
        tick_started = time.monotonic()
        with sim.stage("prepare"):
            sim.prepare()
        with sim.stage("hero"):
            _, _, _, masks, q_b8, actions = observe_hero_bv8(sim, b8)
        living, size = sim.alive[:, 0], sim.length[:, 0]
        decision_counts += living.long()
        reached |= living & (size >= 600)
        metrics = torch.stack(
            ((living & (size >= 600)).sum(), living.sum(), size.sum(), size.max())
        )
        admission_tick = tick % 16 == 0 or cfg.tiny
        if admission_tick:
            alive, lengths = torch.stack((living.long(), size), dim=1).cpu().t().tolist()
        else:
            alive, lengths = [], []
        # Admission is CPU hash/heap work. Resolve every acceptance before any
        # state extraction, then gather each accepted row exactly once.
        admissions = []
        for row, world in (enumerate(ids) if admission_tick else []):
            if not alive[row]:
                continue
            part = partitions[world]
            pool = demos.setdefault(world // cfg.collection_unit_worlds, [])
            if part == "train" and (tick % 16 == 0 or cfg.tiny):
                priority = _priority(cfg, world, tick, "demo")
                if len(pool) < demo_cap or priority < -pool[0][0]:
                    admissions.append((row, world, "demo", priority, pool))
            kinds = []
            if tick % 128 == 0 or cfg.tiny:
                kinds.append("ordinary")
            if (tick % 32 == 0 and lengths[row] >= 600) or cfg.tiny:
                kinds.append("late")
            if len(kinds) == 2:
                kinds = ["ordinary" if tick % 2 == 0 else "late"] if cfg.tiny else ["ordinary"]
            for kind in kinds:
                pool = roots.setdefault((world, kind), [])
                priority = _priority(cfg, world, tick, kind)
                if len(pool) < 2 or priority < -pool[0][0]:
                    admissions.append((row, world, kind, priority, pool))
        # Multiple candidates in one logical demo heap must be compared in
        # their original row order, including intervening heap replacements.
        accepted = []
        for row, world, kind, priority, pool in admissions:
            cap = demo_cap if kind == "demo" else 2
            if len(pool) == cap and priority >= -pool[0][0]:
                continue
            record = dict(world_id=world, partition=partitions[world], tick=tick, priority=priority)
            token = world * cfg.horizon + tick if kind == "demo" else tick
            item = (-priority, token, record)
            if len(pool) == cap:
                heapq.heapreplace(pool, item)
            else:
                heapq.heappush(pool, item)
            accepted.append((row, world, kind, record))
        # Discard candidates displaced later in this same tick, without copying.
        retained = {id(item[2]) for pool in [*demos.values(), *roots.values()] for item in pool}
        accepted = [a for a in accepted if id(a[3]) in retained]
        rows = sorted({a[0] for a in accepted})
        if rows:
            compacts, snapshots = capture_rows(
                sim,
                opponents,
                rows,
                world_ids=[ids[r] for r in rows],
                tick=tick,
                max_body_capacity=cfg.body_capacity,
            )
            by_row = {row: (c, snap) for row, c, snap in zip(rows, compacts, snapshots)}
            action_cpu, mask_cpu = actions[rows].cpu().tolist(), masks[rows].cpu()
            q_cpu = q_b8[rows].cpu()
            decisions_by_row = dict(zip(rows, zip(action_cpu, mask_cpu, q_cpu)))
            for row, world, kind, record in accepted:
                compact, snapshot = by_row[row]
                action, mask, q = decisions_by_row[row]
                record.update(
                    b8_action=int(action), mask=mask.clone(), compact=compact, q_b8=q.clone()
                )
                if kind != "demo":
                    record.update(
                        uid=f"{cfg.namespace}/a{cfg.allocation_index}/w{world}/t{tick}",
                        kind=kind,
                        mix=rosters[row].mix,
                        roster_index=roster_indices[row],
                        hero_length=int(lengths[row]),
                        snapshot=snapshot.to_state_dict(),
                    )
        with sim.stage("opponents"):
            all_actions = opponents.actions(sim, actions)
        sim.step(all_actions)
        with sim.stage("vector_observe"):
            opponents.after_step(sim)
        completed_ticks += 1
        if dev.type == "cuda" and cfg.mode == "ruler":
            torch.cuda.synchronize(dev)
        pending_samples.append((tick, time.monotonic() - tick_started, metrics))
        torch._assert_async(~sim.capacity_overflow.any(), "body overflow invalidates collection")
        if (
            checkpoint is not None
            and completed_ticks < cfg.horizon
            and (
                completed_ticks % cfg.collection_checkpoint_every == 0
                or (ruler_ticks is not None and completed_ticks == ruler_ticks)
            )
        ):
            flush_metrics()
            checkpoint_started = time.monotonic()
            _, snapshots = capture_rows(
                sim,
                opponents,
                list(range(sim.E)),
                world_ids=ids,
                tick=completed_ticks,
                max_body_capacity=cfg.body_capacity,
            )
            checkpoint(
                dict(
                    capture_seconds=time.monotonic() - checkpoint_started,
                    config_sha256=cfg.sha256,
                    world_ids=ids,
                    next_tick=completed_ticks,
                    snapshots=[s.to_state_dict() for s in snapshots],
                    demos=demos,
                    roots=roots,
                    decisions=decisions,
                    decisions_by_world=decisions_by_world,
                    tick_samples=tick_samples,
                    reached_l600_world_ids=sorted(reached_l600),
                )
            )
        if tick % 10 == 0 or time.monotonic() - last_progress >= 45:
            progress({"stage": "collection", "ticks": tick + 1, "worlds": len(ids)})
            last_progress = time.monotonic()
    flush_metrics()
    return {
        **({"stage_diagnostics": sim._timing.consume()} if ruler_census is not None else {}),
        "schema": "spatial-collection/v1",
        "world_ids": ids,
        "decisions": decisions,
        "decisions_by_world": decisions_by_world,
        "roots": [entry[2] for key in sorted(roots) for entry in sorted(roots[key])],
        "demos": [entry[2] for key in sorted(demos) for entry in sorted(demos[key])],
        "elapsed_seconds": time.monotonic() - started,
        "world_ticks": len(ids) * completed_ticks,
        "ruler_partial": ruler_seconds is not None or ruler_ticks is not None,
        "historical_fixture_probe": ruler_fixtures is not None,
        "tick_samples": tick_samples,
        "reached_l600_world_ids": sorted(reached_l600),
    }


def select_dataset(cfg: PilotConfig, batches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Frozen priority selection after complete collection; no label/outcome access."""
    seen = [i for batch in batches for i in batch["world_ids"]]
    if sorted(seen) != list(range(cfg.worlds)) or len(seen) != len(set(seen)):
        raise ValueError("incomplete/duplicate collection worlds")
    candidates = [row for batch in batches for row in batch["roots"]]
    selected = []
    for part, kind, count in (
        ("train", "late", cfg.train_late),
        ("train", "ordinary", cfg.train_ordinary),
        ("assessment", "late", cfg.assessment_late),
        ("assessment", "ordinary", cfg.assessment_ordinary),
    ):
        pool = sorted(
            (r for r in candidates if r["partition"] == part and r["kind"] == kind),
            key=lambda r: (r["priority"], r["uid"]),
        )
        # The tiny fixture tests producer plumbing, never scientific support.
        if len(pool) < count:
            raise RuntimeError(f"INCOMPLETE_SUPPORT {part}/{kind}: {len(pool)} < {count}")
        selected.extend(pool[:count])
    demos = sorted(
        [r for b in batches for r in b["demos"]],
        key=lambda r: (r["priority"], r["world_id"], r["tick"]),
    )[: cfg.demos]
    if len(demos) != cfg.demos:
        raise RuntimeError("INCOMPLETE_SUPPORT ordinary demonstrations")
    decisions = sum(b["decisions"] for b in batches)
    if decisions < cfg.min_decisions:
        raise RuntimeError("INCOMPLETE_SUPPORT actual B8 decisions")
    train_worlds = {r["world_id"] for r in selected if r["partition"] == "train"}
    assessment = {r["world_id"] for r in selected if r["partition"] == "assessment"}
    if train_worlds & assessment or any(r["world_id"] in assessment for r in demos):
        raise ValueError("whole-world training/assessment contamination")
    if not cfg.tiny:
        late = [r for r in selected if r["partition"] == "assessment" and r["kind"] == "late"]
        if len({r["world_id"] for r in late}) < 64:
            raise RuntimeError("INCOMPLETE_SUPPORT independent late parents")
    return {"roots": selected, "demos": demos, "decisions": decisions}
