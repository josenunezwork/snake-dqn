"""Whole E1024 trajectory checkpoints, published as bounded immutable row shards."""

from __future__ import annotations

import math
from typing import Any

from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.resume import UnitStore


def publish_collection(store: UnitStore, cfg: PilotConfig, state: dict[str, Any]) -> None:
    """Publish all fixed shards before the complete-sim boundary marker, last."""
    groups = math.ceil(cfg.worlds / cfg.collection_unit_worlds)
    tick = state["next_tick"]
    names = []
    records = getattr(store, "_collection_records", {})
    store._collection_records = records
    for group in range(groups):
        ids = [i for i in state["world_ids"] if i // cfg.collection_unit_worlds == group]
        indexes = [state["world_ids"].index(i) for i in ids]
        name = f"cstate-{tick * groups + group:07d}"
        names.append(name)
        if store.has(name):
            continue
        fresh = {}

        def references(pool: list[Any], kind: str) -> list[Any]:
            out = []
            for priority, token, record in pool:
                identity = f"w{record['world_id']}/t{record['tick']}/{kind}"
                if identity not in records:
                    records[identity] = record
                    fresh[identity] = record
                out.append((priority, token, identity))
            return out

        demos = {group: references(state["demos"].get(group, []), "demo")}
        roots = {
            key: references(pool, key[1]) for key, pool in state["roots"].items() if key[0] in ids
        }
        store.publish(
            name,
            dict(
                config_sha256=cfg.sha256,
                next_tick=tick,
                world_ids=ids,
                snapshots=[state["snapshots"][i] for i in indexes],
                demos=demos,
                roots=roots,
                records=fresh,
                decisions=sum(state["decisions_by_world"][i] for i in ids),
                decisions_by_world={i: state["decisions_by_world"][i] for i in ids},
                tick_samples=state["tick_samples"],
                reached_l600_world_ids=[i for i in state["reached_l600_world_ids"] if i in ids],
            ),
        )
    store.publish(f"cseal-{tick:07d}", dict(next_tick=tick, shards=names))


def restore_collection(store: UnitStore, cfg: PilotConfig) -> dict[str, Any] | None:
    """Only the latest verified eligible full-sim seal may resume collection."""
    # Read only verified immutable trajectory payloads. Records are keyed by
    # frozen world/tick/kind, not by outcomes; reused IDs never get regenerated
    # into an export. The latest seal alone supplies live simulator/heaps.
    records = {}
    for name in store.list_ids("cstate-"):
        state = store.get(name)
        if set(records) & set(state["records"]):
            raise ValueError("duplicate collection record identity")
        records.update(state["records"])
    store._collection_records = records
    seals = store.list_ids("cseal-")
    if not seals:
        return None
    seal = store.get(seals[-1])
    groups = math.ceil(cfg.worlds / cfg.collection_unit_worlds)
    expected = [f"cstate-{seal['next_tick'] * groups + g:07d}" for g in range(groups)]
    if seal["shards"] != expected:
        raise ValueError("foreign collection checkpoint shards")
    shards = [store.get(name) for name in expected]
    if any(
        s is None or s["next_tick"] != seal["next_tick"] or s["config_sha256"] != cfg.sha256
        for s in shards
    ):
        raise ValueError("incomplete collection trajectory checkpoint")

    def hydrate(pools: dict[Any, list[Any]]) -> dict[Any, list[Any]]:
        try:
            return {
                key: [(p, t, records[ref]) for p, t, ref in pool] for key, pool in pools.items()
            }
        except KeyError as error:
            raise ValueError("missing collection record dependency") from error

    state = dict(
        config_sha256=cfg.sha256,
        next_tick=seal["next_tick"],
        world_ids=[i for s in shards for i in s["world_ids"]],
        snapshots=[i for s in shards for i in s["snapshots"]],
        demos=hydrate({k: v for s in shards for k, v in s["demos"].items()}),
        roots=hydrate({k: v for s in shards for k, v in s["roots"].items()}),
        decisions=sum(s["decisions"] for s in shards),
        decisions_by_world={k: v for s in shards for k, v in s["decisions_by_world"].items()},
        tick_samples=shards[0]["tick_samples"],
        reached_l600_world_ids=[i for s in shards for i in s["reached_l600_world_ids"]],
    )
    if state["world_ids"] != list(range(cfg.worlds)):
        raise ValueError("foreign collection checkpoint world allocation")
    return state


def split_collection(cfg: PilotConfig, batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the frozen 128-world demo reservoirs/export boundaries byte-addressable."""
    groups = math.ceil(cfg.worlds / cfg.collection_unit_worlds)
    out = []
    for group in range(groups):
        ids = [i for i in batch["world_ids"] if i // cfg.collection_unit_worlds == group]
        if not ids:
            continue
        out.append(
            dict(
                batch,
                world_ids=ids,
                decisions=sum(batch["decisions_by_world"][i] for i in ids),
                decisions_by_world={i: batch["decisions_by_world"][i] for i in ids},
                roots=[r for r in batch["roots"] if r["world_id"] in ids],
                demos=[r for r in batch["demos"] if r["world_id"] in ids],
                reached_l600_world_ids=[i for i in batch["reached_l600_world_ids"] if i in ids],
                world_ticks=len(ids) * (batch["world_ticks"] // len(batch["world_ids"])),
            )
        )
    return out
