"""Natural terminal B8 collection with bounded outcome-independent sparse sampling."""

from __future__ import annotations

import hashlib
import heapq
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_snapshots import capture_world_snapshot, observe_hero_bv8
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.observation import compact_from_sim


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
) -> dict[str, Any]:
    """A unit is one complete fixed parent-world batch; no replacement episodes."""
    started = time.monotonic()
    dev = torch.device(device)
    ids = list(world_ids)
    sim = CTDeviceSim(
        cfg.sim_config(len(ids)),
        device=dev,
        train_mode=False,
        allow_respawn=True,
        move_backend="indexed" if dev.type == "cuda" else "reference",
    )
    r2_cache.enable(sim)
    sim.rng_key.copy_(torch.tensor([cfg.world_seed(i) for i in ids], device=dev))
    sim.rng_counter.zero_()
    sim.alive.fill_(False)
    sim.length.fill_(1)
    sim._reward_prev_length.fill_(1)
    sim.terminal_slots.fill_(False)
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    roster_table = training_rosters(cfg.worlds)
    rosters = [roster_table[i] for i in ids]
    opponents = CTOpponentRuntime(
        sim,
        rosters,
        b8_network=b8,
        checkpoint_networks=networks,
        opp_rng_keys=[cfg.opp_seed(i) for i in ids],
    )
    partitions = cfg.partitions()
    # Equal quota per fixed collection batch; final global priority selection
    # cannot overrepresent a batch due to differing natural survival duration.
    batch_count = (cfg.worlds + cfg.world_batch_size - 1) // cfg.world_batch_size
    demo_cap = (cfg.demos + batch_count - 1) // batch_count
    demos: list[tuple[int, int, dict[str, Any]]] = []
    roots: dict[tuple[int, str], list[tuple[int, int, dict[str, Any]]]] = {}
    decisions = 0
    last_progress = started
    for tick in range(cfg.horizon):
        if ruler_seconds is not None and time.monotonic() - started >= ruler_seconds:
            break
        if stop():
            raise InterruptedError("partial collection batch; no result unit published")
        sim.prepare()
        _, _, _, masks, _, actions = observe_hero_bv8(sim, b8)
        alive = sim.alive[:, 0].cpu().tolist()
        lengths = sim.length[:, 0].cpu().tolist()
        decisions += sum(alive)
        for row, world in enumerate(ids):
            if not alive[row]:
                continue
            part = partitions[world]
            if part == "train" and (tick % 16 == 0 or cfg.tiny):
                priority = _priority(cfg, world, tick, "demo")
                if len(demos) < demo_cap or priority < -demos[0][0]:
                    record = {
                        "world_id": world,
                        "partition": part,
                        "tick": tick,
                        "priority": priority,
                        "b8_action": int(actions[row]),
                        "mask": masks[row].cpu().clone(),
                        "compact": _cpu_compact(compact_from_sim(sim, [row])),
                    }
                    item = (-priority, world * cfg.horizon + tick, record)
                    if len(demos) == demo_cap:
                        heapq.heapreplace(demos, item)
                    else:
                        heapq.heappush(demos, item)
            kinds = []
            if tick % 128 == 0 or cfg.tiny:
                kinds.append("ordinary")
            if (tick % 32 == 0 and lengths[row] >= 600) or cfg.tiny:
                kinds.append("late")
            # Ordinary takes precedence on shared schedule; never duplicate roots.
            if len(kinds) == 2:
                kinds = ["ordinary" if tick % 2 == 0 else "late"] if cfg.tiny else ["ordinary"]
            for kind in kinds:
                pool = roots.setdefault((world, kind), [])
                priority = _priority(cfg, world, tick, kind)
                if len(pool) == 2 and priority >= -pool[0][0]:
                    continue
                snapshot = capture_world_snapshot(
                    sim,
                    opponents,
                    row,
                    world_id=world,
                    life_id=0,
                    tick=tick,
                    max_body_capacity=cfg.body_capacity,
                )
                record = {
                    "uid": f"{cfg.namespace}/a{cfg.allocation_index}/w{world}/t{tick}",
                    "world_id": world,
                    "partition": part,
                    "tick": tick,
                    "kind": kind,
                    "mix": rosters[row].mix,
                    "roster_index": world,
                    "hero_length": int(lengths[row]),
                    "priority": priority,
                    "snapshot": snapshot.to_state_dict(),
                    "b8_action": int(actions[row]),
                    "mask": masks[row].cpu().clone(),
                    "compact": _cpu_compact(compact_from_sim(sim, [row])),
                }
                item = (-priority, tick, record)
                if len(pool) == 2:
                    heapq.heapreplace(pool, item)
                else:
                    heapq.heappush(pool, item)
        sim.step(opponents.actions(sim, actions))
        opponents.after_step(sim)
        if bool(sim.capacity_overflow.any()):
            raise ValueError("body overflow invalidates collection")
        if tick % 10 == 0 or time.monotonic() - last_progress >= 45:
            progress({"stage": "collection", "ticks": tick + 1, "worlds": len(ids)})
            last_progress = time.monotonic()
    return {
        "schema": "spatial-collection/v1",
        "world_ids": ids,
        "decisions": decisions,
        "roots": [entry[2] for key in sorted(roots) for entry in sorted(roots[key])],
        "demos": [entry[2] for entry in sorted(demos)],
        "elapsed_seconds": time.monotonic() - started,
        "world_ticks": len(ids) * (tick + 1),
        "ruler_partial": ruler_seconds is not None,
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
