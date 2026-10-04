"""Per-WORLD platform rule shared by the runner, merge and study intakes (stdlib only).

A *unit* is every episode compared or paired within one world: all arms (including replay /
determinism controls) of one ``(wrapper, mix, world_seed)``. The runner dispatches a unit
as a whole to ONE worker (pod or serverless job), so each paired comparison runs on one CPU
model; different worlds may run on different models. A unit re-dispatched after a lost
worker restarts in full on one worker.

Study packages that used to require "one platform id per run" should call
:func:`check_per_world` instead (it requires one platform signature per unit)::

    from research.runpod_fanout.platform_rule import check_per_world
    problems = check_per_world(records)        # {episode key or any id: record entry}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ISA features that change which BLAS/oneDNN kernels run (recorded in the platform stamp).
ISA_FLAGS = (
    "sse4_2",
    "avx",
    "avx2",
    "fma",
    "f16c",
    "avx512f",
    "avx512dq",
    "avx512bw",
    "avx512vl",
    "avx512_vnni",
    "avx512_bf16",
    "avx512_fp16",
    "amx_tile",
    "amx_bf16",
    "amx_int8",
)


def isa_flags_from_cpuinfo(text: str) -> List[str]:
    for line in text.splitlines():
        if line.startswith("flags"):
            have = set(line.split(":", 1)[1].split())
            return [f for f in ISA_FLAGS if f in have]
    return []


def local_isa_flags() -> List[str]:
    try:
        return isa_flags_from_cpuinfo(Path("/proc/cpuinfo").read_text())
    except OSError:
        return []  # macOS arm64: no x86 ISA flags


def unit_of_key(key: str) -> str:
    """``<wrapper>__h<H>__<engine>/<arm>-<mix>-<seed>.json`` -> ``<wrapper>|<mix>|<seed>``."""
    group, name = str(key).split("/", 1)
    wrapper = group.split("__h", 1)[0]
    stem = name[:-5] if name.endswith(".json") else name
    _arm, mix, seed = stem.rsplit("-", 2)
    return f"{wrapper}|{mix}|{int(seed)}"


def unit_of_episode(ep: Mapping[str, Any]) -> str:
    return f"{ep['wrapper']}|{ep['mix']}|{int(ep['world_seed'])}"


def group_units(keys: Iterable[str]) -> Dict[str, List[str]]:
    """unit -> its episode keys (input order kept)."""
    out: Dict[str, List[str]] = {}
    for k in keys:
        out.setdefault(unit_of_key(k), []).append(k)
    return out


def platform_signature(entry: Mapping[str, Any]) -> str:
    """What must be equal within a unit: platform id (OS/py/torch/numpy/isa cap/CPU model),
    the worker's ISA flags, MKL_CBWR and torch threads."""
    p = entry.get("platform") if isinstance(entry.get("platform"), Mapping) else {}
    return json.dumps(
        [
            p.get("platform_id"),
            list(p.get("isa_flags") or []),
            p.get("mkl_cbwr"),
            p.get("torch_threads"),
        ],
        separators=(",", ":"),
    )


def unit_signatures(records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """unit -> {signature: count}."""
    out: Dict[str, Dict[str, int]] = {}
    for key, entry in records.items():
        sig = platform_signature(entry)
        bucket = out.setdefault(unit_of_key(key), {})
        bucket[sig] = bucket.get(sig, 0) + 1
    return out


def check_per_world(records: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Problems: records without a platform id, units whose records span >1 platform."""
    problems = []
    for key, entry in sorted(records.items()):
        p = entry.get("platform") if isinstance(entry.get("platform"), Mapping) else {}
        if not p.get("platform_id"):
            problems.append(f"{key}: no platform stamp")
    for unit, sigs in sorted(unit_signatures(records).items()):
        if len(sigs) > 1:
            problems.append(f"unit {unit} spans {len(sigs)} platforms: {sorted(sigs)}")
    return problems


def stamp_worker(entry: Dict[str, Any], worker: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Add the worker's ISA flags / id / provider kind to ``entry['platform']`` (in place)."""
    if worker is None:
        return entry
    p = entry.setdefault("platform", {})
    p["isa_flags"] = list(worker.get("isa_flags") or [])
    if worker.get("worker_id"):
        p["worker_id"] = str(worker["worker_id"])
    if worker.get("backend"):
        p["backend"] = str(worker["backend"])
    return entry


def choose_unit_sources(
    runs: Sequence[Tuple[str, Mapping[str, Mapping[str, Any]]]],
) -> Dict[str, str]:
    """For merge: unit -> the run label whose copy of the unit is used (most records for the
    unit; ties go to the later run)."""
    best: Dict[str, Tuple[int, int, str]] = {}
    for order, (label, records) in enumerate(runs):
        counts: Dict[str, int] = {}
        for key in records:
            u = unit_of_key(key)
            counts[u] = counts.get(u, 0) + 1
        for u, n in counts.items():
            if u not in best or (n, order) >= best[u][:2]:
                best[u] = (n, order, label)
    return {u: v[2] for u, v in best.items()}
