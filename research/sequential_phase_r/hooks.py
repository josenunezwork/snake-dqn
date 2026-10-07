"""Integration hooks: how a future FRP-family study's evaluate / merge opts into a sequential
Phase R (look schedule, records -> look deltas, look analysis -> create-only receipt).

A study keeps its own unit runner, shards, namespaces and merge. It adds:

1. **Before any Phase R shard:** ``plan = make_plan(N, RULE)`` (the rule spec pinned in its
   pre-registration) and ``receipts.write_plan(seq_root, plan.as_dict(), banks, study)``.
2. **Per look k:** ``binding = receipts.look_gate(seq_root, k)`` before a look-k shard starts
   (record ``binding`` in the shard's start marker); run the units of :func:`look_units` (look
   k's world index range only); merge the look's records.
3. **After look k's records are merged:** ``analyse_look(seq_root, plan, entries, k, flags)``
   writes ``looks/look-<k>.json``; continue only on ``action == "CONTINUE"``.

Records are the FRP screen-record format: ``hero``, ``seed``, ``mix``, ``world_seed``, the
nested episode's ``prefix_h5000`` block and full ``record`` (``mass_integral``,
``survival_fraction``), ``control`` (direct-H5000 controls are never decision data).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from research.sequential_phase_r import receipts
from src.evaluation.sequential_phase_r import (
    SequentialPhaseRPlan,
    invalidate,
    required_inputs,
    sequential_phase_r_decision,
)

Key = Tuple[str, int, str, int]


class LookDataError(ValueError):
    """The look's record set is incomplete, duplicated or not the planned one."""


# ----------------------------------------------------------------------------- schedule
def look_ranges(look_sizes: Sequence[int]) -> List[Tuple[int, int]]:
    """World index range ``[lo, hi)`` of every look in each bank."""
    out, lo = [], 0
    for hi in look_sizes:
        out.append((lo, int(hi)))
        lo = int(hi)
    return out


def _batches(lo: int, hi: int, max_batch: int) -> List[Tuple[int, int]]:
    """Balanced batches of at most ``max_batch`` indexes covering ``[lo, hi)``."""
    count = -(-(hi - lo) // max_batch)
    base, extra = divmod(hi - lo, count)
    out, start = [], lo
    for i in range(count):
        size = base + (1 if i < extra else 0)
        out.append((start, start + size))
        start += size
    return out


def look_units(
    plan: SequentialPhaseRPlan,
    look: int,
    banks: Mapping[int, Sequence[int]],
    heroes: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]] = (),
    max_batch: int = 8,
) -> List[Dict[str, Any]]:
    """The units of look ``look`` (look-major schedule).

    ``heroes``: ``{"hero": label, "seeds": [...], "tier": int}`` -- every hero that plays a
    seed's worlds (incumbent, candidate, control arm, champion reference, descriptive cells).
    ``controls``: direct-H5000 prefix controls ``{"hero", "seeds"}``; they play world index 0
    and therefore belong to look 0. A unit is one hero x seed x mix x batch of <= ``max_batch``
    worlds inside the look's index range; ``group`` (seed, mix, batch) keeps every hero of a
    world in one shard (the per-world platform rule).
    """
    lo, hi = look_ranges(plan.look_sizes)[look]
    units: List[Dict[str, Any]] = []
    for row in heroes:
        for seed in row["seeds"]:
            bank = [int(w) for w in banks[int(seed)]]
            if len(bank) != plan.n_worlds:
                raise LookDataError(
                    f"bank of seed {seed} has {len(bank)} worlds, not {plan.n_worlds}"
                )
            for mix in plan.rule["mixes"]:
                for a, b in _batches(lo, hi, max_batch):
                    units.append(
                        {
                            "unit_id": f"L{look}|{row['hero']}|s{seed}|{mix}|{a}",
                            "look": look,
                            "hero": row["hero"],
                            "seed": int(seed),
                            "mix": mix,
                            "worlds": bank[a:b],
                            "world_index_range": [a, b],
                            "tier": int(row.get("tier", 1)),
                            "control": False,
                            "group": f"L{look}|s{seed}|{mix}|{a}",
                        }
                    )
    if look == 0:
        for row in controls:
            for seed in row["seeds"]:
                bank = [int(w) for w in banks[int(seed)]]
                for mix in plan.rule["mixes"]:
                    units.append(
                        {
                            "unit_id": f"L0|control|{row['hero']}|s{seed}|{mix}",
                            "look": 0,
                            "hero": row["hero"],
                            "seed": int(seed),
                            "mix": mix,
                            "worlds": [bank[0]],
                            "world_index_range": [0, 1],
                            "tier": int(row.get("tier", 1)),
                            "control": True,
                            "group": f"L0|s{seed}|{mix}|0",  # world 0's batch
                        }
                    )
    return units


# ----------------------------------------------------------------------------- records
def load_entries(dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    """Every ``records/*.json`` of the given shard directories, with path and sha256."""
    out = []
    for d in dirs:
        for path in sorted(Path(d, "records").glob("*.json")):
            entry = json.loads(path.read_text())
            entry["_path"] = str(path)
            entry["_sha256"] = receipts.sha256_file(path)
            out.append(entry)
    return out


def metric_value(entry: Mapping[str, Any], metric: str) -> float:
    """``mi5`` / ``surv5`` from the nested record's exact H5000 prefix block; ``mi10`` /
    ``surv10`` from the full nested record. A record without a prefix block is not decision
    data (it would be read at the wrong horizon)."""
    if entry.get("prefix_h5000") is None:
        raise LookDataError("decision records must be nested (prefix_h5000 present)")
    source = entry["prefix_h5000"] if metric in ("mi5", "surv5") else entry["record"]
    key = "mass_integral" if metric.startswith("mi") else "survival_fraction"
    return float(source[key])


def _index(entries: Iterable[Mapping[str, Any]]) -> Dict[Key, Mapping[str, Any]]:
    out: Dict[Key, Mapping[str, Any]] = {}
    for entry in entries:
        if entry.get("control"):
            continue
        key = (str(entry["hero"]), int(entry["seed"]), str(entry["mix"]), int(entry["world_seed"]))
        if key in out:
            raise LookDataError(f"duplicate record {key}")
        out[key] = entry
    return out


def look_deltas(
    plan: SequentialPhaseRPlan,
    banks: Mapping[int, Sequence[int]],
    entries: Iterable[Mapping[str, Any]],
    look: int,
) -> Tuple[Dict[str, Any], List[Mapping[str, Any]], List[Key]]:
    """``(deltas, used records, beyond-look keys)`` for look ``look``.

    ``deltas[cell][metric][seed][mix]`` = ``hero - baseline`` over the bank's first
    ``look_sizes[look]`` worlds. Missing records raise; records of a decision hero on a world
    beyond the look are returned as ``beyond`` (a prefix-integrity violation).
    """
    rule = plan.rule
    size = plan.look_sizes[look]
    index = _index(entries)
    deltas: Dict[str, Any] = {}
    used: Dict[Key, Mapping[str, Any]] = {}
    for cell, metrics in required_inputs(rule).items():
        hero, baseline = rule["cells"][cell]["hero"], rule["cells"][cell]["baseline"]
        deltas[cell] = {m: {} for m in metrics}
        for seed in rule["cells"][cell]["seeds"]:
            worlds = [int(w) for w in banks[int(seed)]][:size]
            for metric in metrics:
                deltas[cell][metric][seed] = {}
            for mix in rule["mixes"]:
                rows = []
                for world in worlds:
                    pair = []
                    for who in (hero, baseline):
                        key = (who, int(seed), mix, world)
                        if key not in index:
                            raise LookDataError(f"look {look}: missing record {key}")
                        used[key] = index[key]
                        pair.append(index[key])
                    rows.append(pair)
                for metric in metrics:
                    deltas[cell][metric][seed][mix] = [
                        metric_value(c, metric) - metric_value(b, metric) for c, b in rows
                    ]
    decision_heroes = {rule["cells"][c]["hero"] for c in rule["cells"]} | {
        rule["cells"][c]["baseline"] for c in rule["cells"]
    }
    position = {int(seed): {int(w): i for i, w in enumerate(bank)} for seed, bank in banks.items()}
    beyond = sorted(
        key
        for key in index
        if key[0] in decision_heroes
        and key[1] in position
        and position[key[1]].get(key[3], -1) >= size
    )
    return deltas, list(used.values()), beyond


def deltas_sha256(deltas: Mapping[str, Any]) -> str:
    plain = {
        c: {m: {str(s): v for s, v in by_seed.items()} for m, by_seed in by_m.items()}
        for c, by_m in deltas.items()
    }
    return receipts.canonical_sha(plain)


# ----------------------------------------------------------------------------- analysis
def analyse_look(
    root: Path,
    plan: SequentialPhaseRPlan,
    entries: Sequence[Mapping[str, Any]],
    look: int,
    flags: Mapping[str, bool],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide look ``look`` and write its create-only receipt; returns the receipt payload.

    Refuses (raises) when the plan on disk is not ``plan`` or the look gate does not allow the
    look. A missing record raises ``LookDataError`` (the look is incomplete: no receipt; the
    study stays INCOMPLETE unless the records arrive). Records beyond the look make the
    decision invalid (prefix integrity) and the receipt's action HALT.
    """
    payload, _ = receipts.read_plan(root)
    if payload["plan_sha256"] != plan.sha256() or payload["plan"] != json.loads(
        json.dumps(plan.as_dict())
    ):
        raise receipts.ReceiptError("the plan on disk is not this plan")
    if look > 0:
        receipts.look_gate(root, look)  # previous receipt exists and says CONTINUE
    elif receipts.receipt_path(root, 0).exists():
        raise receipts.ReceiptError("look 0 is already analysed")
    banks = {int(s): [int(w) for w in v] for s, v in payload["banks"].items()}
    deltas, used, beyond = look_deltas(plan, banks, entries, look)
    decision = sequential_phase_r_decision(plan, look, deltas, flags)
    if beyond:
        decision = invalidate(
            decision,
            f"{len(beyond)} decision record(s) beyond look {look} present at analysis "
            f"(prefix integrity), e.g. {list(beyond[0])}",
        )
    rows = [
        {
            "key": [e["hero"], int(e["seed"]), e["mix"], int(e["world_seed"])],
            "path": e.get("_path", ""),
            "sha256": e.get("_sha256", ""),
        }
        for e in used
    ]
    receipts.write_look_receipt(
        root, look, decision, rows, deltas_sha256(deltas), flags, extra=extra
    )
    return receipts.load(receipts.receipt_path(root, look))
