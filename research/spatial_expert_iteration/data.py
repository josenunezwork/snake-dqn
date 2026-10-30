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


def select_anchors(cfg: RoundConfig, reservoir: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash-sample distinct root UIDs without replacement, independent of outcomes."""
    validate_rows(reservoir, "train")
    if max(cfg.anchor_worlds, cfg.reservoir_roots) > len(reservoir):
        raise ValueError("anchor/reuse allocation exceeds distinct reservoir roots")
    return sorted(
        reservoir, key=lambda r: (rank(cfg.namespace, cfg.seed, "anchor", r["uid"]), r["uid"])
    )[: cfg.anchor_worlds]


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
    ranks = {r["uid"]: i for i, r in enumerate(ordered)}
    audit = []
    for row in rows:
        p = 1.0 if row["uid"] in top else cfg.query_floor
        draw = rank(cfg.namespace, cfg.seed, "query-floor", row["uid"]) / 2**64
        audit.append(
            dict(
                uid=row["uid"],
                parent_id=row["parent_id"],
                inclusion_probability=p,
                queried=draw < p,
                ranking=cfg.query_ranking,
                disagreement_rank=ranks[row["uid"]],
                population=len(rows),
                disagreement_top_quartile=ranks[row["uid"]] < math.ceil(len(rows) / 4),
                gate_student_action=row.get("gate_student_action", row["student_action"]),
                b8_action=row["b8_action"],
                student_logits=row.get("gate_student_logits"),
                student_top2_margin=row.get("gate_top2_margin"),
            )
        )
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
    roots: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
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
            ordinary = tick % cfg.ordinary_root_every == 0
            late = tick % cfg.root_every == 0 and lengths[row] >= cfg.mature_length
            # Disjoint stratum schedules, as in the pilot. Tiny alternates to
            # exercise both strata without manufacturing maturity.
            kind = (
                ("late" if tick % 2 else "ordinary")
                if cfg.tiny
                else ("ordinary" if ordinary else "late")
            )
            root_ok = (ordinary or late) and (kind != "late" or late)
            pool = roots[(world, kind)]
            if root_ok and len(pool) >= cfg.roots_per_parent:
                root_ok = priority < max(r["priority"] for r in pool)
            group = world // cfg.logical_reservoir_size
            demo_cap = math.ceil(cfg.demos / math.ceil(cfg.worlds / cfg.logical_reservoir_size))
            demo_ok = partition == "train" and tick % cfg.demo_every == 0
            if demo_ok and len(demos[group]) >= demo_cap:
                demo_ok = priority < max(r["priority"] for r in demos[group])
            if root_ok or demo_ok:
                admissions.append((row, world, priority, root_ok, demo_ok, group, kind))
        # A single sparse gather for accepted rows; dense planes remain ephemeral.
        compacts = (
            _cpu_compact(compact_from_sim(sim, [a[0] for a in admissions])) if admissions else None
        )
        for local, (row, world, priority, root_ok, demo_ok, group, kind) in enumerate(admissions):
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
                kind=kind,
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
                roots[(world, kind)].append(dict(record, snapshot=snapshot.to_state_dict()))
                roots[(world, kind)] = sorted(roots[(world, kind)], key=lambda r: r["priority"])[
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


def check_support_possible(cfg: RoundConfig, batches: Sequence[dict[str, Any]]) -> None:
    """Stop only when even perfect remaining worlds cannot supply frozen counts.

    Generous upper bounds ignore later priority truncation; they can delay a
    refusal but cannot replace, refill, or stop a still-feasible allocation.
    """
    for partition, worlds, quota, parents in (
        ("train", cfg.worlds, cfg.candidate_roots - cfg.reservoir_roots, cfg.min_mature_parents),
        (
            "assessment",
            cfg.assessment_worlds,
            cfg.assessment_roots,
            cfg.min_assessment_mature_parents,
        ),
    ):
        selected = [b for b in batches if b["partition"] == partition]
        ids = [i for b in selected for i in b["world_ids"]]
        seen = set(ids)
        remaining = worlds - len(ids)
        roots = [r for b in selected for r in b["roots"]]
        mature = [
            r
            for r in roots
            if r["start_distribution"] == "fresh" and r["hero_length"] >= cfg.mature_length
        ]
        fresh_left = sum(
            i not in seen and (partition != "train" or i >= cfg.anchor_worlds)
            for i in range(worlds)
        )
        required_mature = cfg.min_mature_roots if partition == "train" else parents
        if (
            len(roots) + remaining * 2 * cfg.roots_per_parent < quota
            or len(mature) + fresh_left * cfg.roots_per_parent < required_mature
            or len({r["parent_id"] for r in mature}) + fresh_left < parents
        ):
            raise RuntimeError(f"INCOMPLETE_SUPPORT: {partition} impossible after collection batch")
        if partition == "train":
            demos = sum(len(b["demos"]) for b in selected)
            # At most one demo per world/tick, with a generous longest horizon.
            per_world = math.ceil(max(cfg.horizon, cfg.anchor_horizon) / cfg.demo_every)
            if demos + remaining * per_world < cfg.demos:
                raise RuntimeError("INCOMPLETE_SUPPORT: demos impossible after collection batch")


def stratified_roots(
    rows: Sequence[dict[str, Any]],
    count: int,
    cfg: RoundConfig,
    min_parents: int,
    min_fresh: int,
) -> list[dict[str, Any]]:
    """Reserve late slots and fresh parent support before ordinary hash fill.

    Selection uses lengths/ancestry only, never deaths or search outcomes.
    Per-parent caps apply separately in each stratum, as in the pilot.
    Insufficient natural support refuses.
    """
    late_count = max(math.ceil(count * cfg.late_fraction), min_fresh)
    if late_count > count:
        raise RuntimeError("INCOMPLETE_SUPPORT: late quota exceeds fresh population")
    ordered = sorted(rows, key=lambda r: (r["priority"], r["uid"]))
    selected, used, parents = [], set(), defaultdict(int)
    stratum_counts = defaultdict(int)

    def take(row: dict[str, Any]) -> None:
        selected.append(row)
        used.add(row["uid"])
        parents[row["parent_id"]] += 1
        stratum_counts[(row["parent_id"], row["kind"])] += 1

    late = [r for r in ordered if r["kind"] == "late"]
    fresh = [
        r
        for r in late
        if r["start_distribution"] == "fresh" and r["hero_length"] >= cfg.mature_length
    ]
    for row in fresh:
        if sum(n > 0 for n in parents.values()) >= min_parents:
            break
        if not parents[row["parent_id"]]:
            take(row)
    if len([n for n in parents.values() if n]) < min_parents:
        raise RuntimeError("INCOMPLETE_SUPPORT: natural mature parents")
    for pool, quota in ((fresh, min_fresh), (late, late_count), (ordered, count)):
        for row in pool:
            if len(selected) >= quota:
                break
            if (
                row["uid"] in used
                or stratum_counts[(row["parent_id"], row["kind"])] >= cfg.roots_per_parent
            ):
                continue
            if pool is ordered and row["kind"] != "ordinary":
                continue
            take(row)
        if len(selected) < quota:
            raise RuntimeError("INCOMPLETE_SUPPORT: late/ordinary stratum quota")
    return selected


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
    train = stratified_roots(
        train, cfg.candidate_roots - len(reused), cfg, cfg.min_mature_parents, cfg.min_mature_roots
    )
    train = [dict(r) for r in train + reused]
    held = stratified_roots(
        held,
        cfg.assessment_roots,
        cfg,
        cfg.min_assessment_mature_parents,
        cfg.min_assessment_mature_parents,
    )
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
    mature = [dict(r) for r in train if r["hero_length"] >= cfg.mature_length]
    reused_ids = {r["uid"] for r in reused}
    fresh_mature = [
        r for r in mature if r["uid"] not in reused_ids and r["start_distribution"] == "fresh"
    ]
    if (
        len(fresh_mature) < cfg.min_mature_roots
        or len({r["parent_id"] for r in fresh_mature}) < cfg.min_mature_parents
        or len({r["parent_id"] for r in held if r["hero_length"] >= cfg.mature_length})
        < cfg.min_assessment_mature_parents
    ):
        raise RuntimeError("INCOMPLETE_SUPPORT: natural mature parents")
    return dict(train=train, assessment=held, demos=demos, reservoir=mature)
