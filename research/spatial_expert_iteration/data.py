"""Natural student occupancy, immutable parent ancestry, and outcome-blind queries."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Sequence

import torch

from research.sandbox_a_resident.ct_mechanics import CTOpponentRuntime
from research.sandbox_a_resident.ct_snapshots import (
    CompactWorldSnapshot,
    capture_world_snapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.collect import _cpu_compact
from research.spatial_student_pilot.observation import compact_from_sim, compact_take
from research.spatial_student_pilot.search import _new_sim, student_actions

from .config import RoundConfig, identity


def rank(*parts: Any) -> int:
    return int(identity(parts)[:16], 16)


def validate_rows(rows: Sequence[dict[str, Any]], partition: str) -> None:
    """Descendants inherit original parent partition, including all later rounds."""
    seen = set()
    for row in rows:
        if row["uid"] in seen:
            raise ValueError("duplicate root identity")
        seen.add(row["uid"])
        if row["partition"] != partition or row["parent_partition"] != partition:
            raise ValueError("held-out ancestry contamination")
        if not row["parent_id"] or not row["collector_sha256"]:
            raise ValueError("missing provenance")


def query_gate(rows: Sequence[dict[str, Any]], cfg: RoundConfig) -> list[dict[str, Any]]:
    """Fixed disagreement priority + independent hash-Bernoulli random floor.

    Horvitz-Thompson weights target the entire frozen candidate population;
    neither selection nor weights depend on search returns. Never pseudo-label.
    """
    validate_rows(rows, "train")
    ordered = sorted(
        rows,
        key=lambda r: (
            -(r.get("gate_student_action", r["student_action"]) != r["b8_action"]),
            rank(cfg.namespace, "gate-rank", r["uid"]),
        ),
    )
    top = {r["uid"] for r in ordered[: math.ceil(len(rows) * cfg.query_top_fraction)]}
    audit = []
    for row in rows:
        p = 1.0 if row["uid"] in top else cfg.query_floor
        draw = rank(cfg.namespace, cfg.seed, "query-floor", row["uid"]) / 2**64
        audit.append(dict(uid=row["uid"], inclusion_probability=p, queried=draw < p))
    return audit


@torch.inference_mode()
def collect(
    cfg: RoundConfig,
    ids: Sequence[int],
    *,
    student: Any,
    student_sha: str,
    b8: Any,
    networks: dict[str, Any],
    device: str,
    partition: str = "train",
    anchors: Sequence[dict[str, Any]] = (),
    stop: Callable[[], bool] = lambda: False,
    progress: Callable[[dict[str, Any]], None] = lambda _: None,
) -> dict[str, Any]:
    """Collect fixed natural worlds; search never intervenes in student actions.

    Logical reservoir quotas depend on original 128-world groups, independently
    of the larger physics batch. Anchors are training-only full prepared states;
    fresh assessment never imports an anchor or uses the query gate.
    """
    if partition not in {"train", "assessment"} or not ids:
        raise ValueError("invalid collection population")
    validate_rows(anchors, "train")
    if anchors and (partition != "train" or len(anchors) != len(ids)):
        raise ValueError("anchor population must be wholly training")
    sim = _new_sim(cfg, len(ids), device)
    phase = "assessment" if partition == "assessment" else "collection"
    sim.rng_key.copy_(torch.tensor([cfg.world_seed(i, phase) for i in ids], device=device))
    sim.rng_counter.zero_()
    sim.alive.fill_(False)
    sim.length.fill_(1)
    sim._reward_prev_length.fill_(1)
    sim.terminal_slots.fill_(False)
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    table = training_rosters(max(cfg.worlds, cfg.assessment_worlds))
    roster_ids = [int(a["roster_index"]) for a in anchors] if anchors else list(ids)
    opponents = CTOpponentRuntime(
        sim,
        [table[i] for i in roster_ids],
        b8_network=b8,
        checkpoint_networks=networks,
        opp_rng_keys=[cfg.opp_seed(i, phase) for i in ids],
    )
    if anchors:
        restore_world_snapshots_into(
            [CompactWorldSnapshot.from_state_dict(a["snapshot"]) for a in anchors], sim, opponents
        )
        sim.terminal_slots[:, 0] = True
    roots: dict[int, list[dict[str, Any]]] = defaultdict(list)
    demos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    horizon = cfg.anchor_horizon if anchors else cfg.horizon
    decisions = 0
    for tick in range(horizon):
        if stop():
            raise InterruptedError("partial collection is not a completed unit")
        sim.prepare()
        _, _, _, masks, q, b8_actions = observe_hero_bv8(sim, b8)
        actions = student_actions(sim, student)
        alive = sim.alive[:, 0].cpu().tolist()
        lengths = sim.length[:, 0].cpu().tolist()
        decisions += sum(alive)
        admissions = []
        for row, world in enumerate(ids):
            if not alive[row]:
                continue
            priority = rank(cfg.namespace, phase, world, tick)
            root_ok = tick % cfg.root_every == 0
            pool = roots[world]
            if root_ok and len(pool) >= cfg.roots_per_parent:
                root_ok = priority < max(r["priority"] for r in pool)
            group = world // cfg.logical_reservoir_size
            demo_cap = math.ceil(cfg.demos / math.ceil(cfg.worlds / cfg.logical_reservoir_size))
            demo_ok = partition == "train" and tick % cfg.demo_every == 0
            if demo_ok and len(demos[group]) >= demo_cap:
                demo_ok = priority < max(r["priority"] for r in demos[group])
            if root_ok or demo_ok:
                admissions.append((row, world, priority, root_ok, demo_ok, group))
        # A single sparse gather for accepted rows; dense planes remain ephemeral.
        compacts = (
            _cpu_compact(compact_from_sim(sim, [a[0] for a in admissions])) if admissions else None
        )
        for local, (row, world, priority, root_ok, demo_ok, group) in enumerate(admissions):
            parent = anchors[row]["parent_id"] if anchors else f"{cfg.namespace}/{phase}/w{world}"
            record = dict(
                uid=f"{cfg.namespace}/{phase}/w{world}/t{tick}",
                world_id=world,
                parent_id=parent,
                parent_partition=partition,
                parent_seed=(
                    anchors[row]["parent_seed"] if anchors else cfg.world_seed(world, phase)
                ),
                parent_opp_seed=(
                    anchors[row]["parent_opp_seed"] if anchors else cfg.opp_seed(world, phase)
                ),
                partition=partition,
                tick=tick,
                kind="late" if lengths[row] >= cfg.mature_length else "ordinary",
                hero_length=int(lengths[row]),
                mix=table[roster_ids[row]].mix,
                roster_index=roster_ids[row],
                priority=priority,
                collector_sha256=student_sha,
                start_distribution="anchor" if anchors else "fresh",
                ancestor_uid=anchors[row]["uid"] if anchors else None,
                b8_action=int(b8_actions[row]),
                b8_q=q[row].cpu().clone(),
                student_action=int(actions[row]),
                mask=masks[row].cpu().clone(),
                compact=compact_take(compacts, [local]),
            )
            if demo_ok:
                demos[group].append(record)
                demos[group] = sorted(demos[group], key=lambda r: r["priority"])[:demo_cap]
            if root_ok:
                snapshot = capture_world_snapshot(
                    sim,
                    opponents,
                    row,
                    world_id=world,
                    life_id=0,
                    tick=tick,
                    max_body_capacity=cfg.body_capacity,
                )
                roots[world].append(dict(record, snapshot=snapshot.to_state_dict()))
                roots[world] = sorted(roots[world], key=lambda r: r["priority"])[
                    : cfg.roots_per_parent
                ]
        sim.step(opponents.actions(sim, actions))
        opponents.after_step(sim)
        if bool(sim.capacity_overflow.any()):
            raise ValueError("natural collection body overflow")
        progress(dict(stage="collect", partition=partition, tick=tick + 1, worlds=len(ids)))
    return dict(
        world_ids=list(ids),
        partition=partition,
        decisions=decisions,
        roots=[r for key in sorted(roots) for r in roots[key]],
        demos=[r for key in sorted(demos) for r in demos[key]],
    )


def dataset(
    cfg: RoundConfig, batches: Sequence[dict[str, Any]], reservoir: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Choose a fixed candidate population without inspecting labels or deaths."""
    validate_rows(reservoir, "train")
    for part, count in (("train", cfg.worlds), ("assessment", cfg.assessment_worlds)):
        ids = [i for batch in batches if batch["partition"] == part for i in batch["world_ids"]]
        if sorted(ids) != list(range(count)):
            raise ValueError("incomplete/duplicate parent-world population")
    train = [r for b in batches if b["partition"] == "train" for r in b["roots"]]
    held = [r for b in batches if b["partition"] == "assessment" for r in b["roots"]]
    demos = [r for b in batches if b["partition"] == "train" for r in b["demos"]]
    # Historical states remain explicitly versioned. Re-query under this round's
    # continuation; no historical target is silently treated as current search.
    reused = sorted(reservoir, key=lambda r: rank(cfg.namespace, "reuse", r["uid"]))[
        : cfg.reservoir_roots
    ]
    train = sorted(train, key=lambda r: (r["priority"], r["uid"]))
    train = train[: cfg.candidate_roots - len(reused)] + reused
    held = sorted(held, key=lambda r: (r["priority"], r["uid"]))[: cfg.assessment_roots]
    demos = sorted(demos, key=lambda r: (r["priority"], r["uid"]))[: cfg.demos]
    if (
        len(train) != cfg.candidate_roots
        or len(held) != cfg.assessment_roots
        or len(demos) != cfg.demos
    ):
        raise RuntimeError("INCOMPLETE_SUPPORT: never replace failed natural worlds")
    validate_rows(train, "train")
    validate_rows(held, "assessment")
    validate_rows(demos, "train")
    if {r["parent_id"] for r in train + demos} & {r["parent_id"] for r in held}:
        raise ValueError("held-out parent collision")
    mature = [r for r in train if r["hero_length"] >= cfg.mature_length]
    if (
        len(mature) < cfg.min_mature_roots
        or len({r["parent_id"] for r in mature}) < cfg.min_mature_parents
        or len({r["parent_id"] for r in held if r["hero_length"] >= cfg.mature_length})
        < cfg.min_assessment_mature_parents
    ):
        raise RuntimeError("INCOMPLETE_SUPPORT: natural mature parents")
    return dict(train=train, assessment=held, demos=demos, reservoir=mature)
