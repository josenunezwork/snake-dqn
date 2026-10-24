"""R6 A100 E-scaling ruler for the R2 engine (I' arm): E1024 vs E2048 vs E4096.

Authority: ``JUDGMENT_POST_I0S_DIRECTION_20261024.md`` step 5 and round-36 R6.
Pre-registered in ``R6_PROTOCOL_20261024.md``; reduced by ``r6_analyze.py``.

Per roster and E in the ladder (base, 2x, 4x), each cell:

1. builds the admitted I0 engine with R2 registered (``r2_bench.build_case``)
   and pins every policy network forward to microbatches of the ladder base
   (``Microbatched``: same per-row kernels/shapes at every E);
2. mapped-output proof: from reset, ``PROOF_TICKS`` progressing ticks; after
   each tick a SHA-256 per world over every world-indexed simulator, vector
   runtime and collector field plus that world's action-grid row and hero Q
   row.  World ``j`` must equal base-E world ``j mod base`` (all worlds are
   ``repeat_interleave`` copies of one fixture world);
3. ``WARMUPS`` progressing ticks, then ``REPEATS`` equal-work blocks: reset
   once, ``TICKS_AT_MAX * (E_max / E)`` progressing ticks, so every block
   completes ``E_max * TICKS_AT_MAX`` world-steps; complete world-steps/s =
   work / synchronized wall; block final state digests must agree;
4. memory: peak reserved/allocated with peaks reset before the build, plus the
   context/driver overhead measured before the build; physical-memory
   headroom = 1 - (overhead + peak reserved) / device total.

An out-of-memory cell is published as ``OOM`` (INCOMPLETE, never a smaller-E
pass).  Engineering evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import device_policy
from research.gpu_sim_spike import integration_bench as i0
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import integration_native, r2_bench, r2_cache
from research.gpu_sim_spike import r6_units as units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes

SCHEMA = "gpu-sim-spike-r6-bench/v1"
PROOF_TICKS = {"smoke": 2, "canary": 3, "main": 3}
WARMUPS = {"smoke": 1, "canary": 1, "main": 3}
REPEATS = {"smoke": 1, "canary": 1, "main": 3}
TICKS_AT_MAX = {"smoke": 2, "canary": 8, "main": 8}
DEADLINE_LIMIT = {"smoke": 600, "canary": 360, "main": 1260}


class R6Invalid(ValueError):
    """Nondeterminism or native failure: the R6 packet is INVALID."""


class Microbatched(nn.Module):
    """Run a policy network in fixed row microbatches (R6: pinned microbatch shapes)."""

    def __init__(self, inner: nn.Module, size: int) -> None:
        super().__init__()
        self.inner, self.size = inner, int(size)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        rows = args[0].shape[0]
        if rows <= self.size:
            return self.inner(*args)
        parts = [
            self.inner(*(arg[start : start + self.size] for arg in args))
            for start in range(0, rows, self.size)
        ]
        return torch.cat(parts, dim=0)


def pin_microbatches(policy: Any, size: int) -> None:
    wrapped: dict[int, nn.Module] = {}

    def wrap(net: Any) -> Any:
        if net is None:
            return None
        if id(net) not in wrapped:
            wrapped[id(net)] = Microbatched(net, size)
        return wrapped[id(net)]

    policy.b8_network = wrap(policy.b8_network)
    policy.opponent_networks = tuple(wrap(net) for net in policy.opponent_networks)


def _rows(value: torch.Tensor, worlds: int, axis: int) -> np.ndarray:
    tensor = value.detach()
    if axis:
        tensor = tensor.transpose(0, axis)
    array = tensor.cpu().contiguous().numpy()
    return array.reshape(worlds, -1).view(np.uint8)


def world_digests(case: tuple, extra: dict[str, torch.Tensor]) -> list[str]:
    """SHA-256 per world over every world-indexed persistent field and output row."""
    sim, policy, _template, _ptemplate, collection, _arm = case
    worlds = int(sim.E)
    hashers = [hashlib.sha256() for _ in range(worlds)]
    fields: list[tuple[str, torch.Tensor, int]] = []
    for owner, values in (("sim", vars(sim)), ("runtime", vars(policy.vector_runtime))):
        for key in sorted(values):
            value = values[key]
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == worlds:
                fields.append((f"{owner}.{key}", value, 0))
    for key in sorted(collection):
        fields.append((f"collection.{key}", collection[key], 1))
    for key in sorted(extra):
        fields.append((f"output.{key}", extra[key], 0))
    for name, value, axis in fields:
        rows = _rows(value, worlds, axis)
        label = f"{name}|{value.dtype}|{list(value.shape[1:] if not axis else value.shape[:1])}"
        for world in range(worlds):
            hashers[world].update(label.encode())
            hashers[world].update(rows[world].tobytes())
    return [h.hexdigest() for h in hashers]


def mapped_proof(case: tuple, ticks: int) -> list[list[str]]:
    """Per-tick per-world digests over ``ticks`` progressing ticks from reset."""
    sim, policy, template, policy_template, collection, _arm = case
    captured: dict[str, torch.Tensor] = {}
    evaluate, actions = device_policy.DevicePolicy.evaluate, device_policy.DevicePolicy.actions

    def seen_evaluate(self: Any, target: Any) -> Any:
        q, legal, hero = evaluate(self, target)
        captured.update(hero_q=q, hero_legal=legal)
        return q, legal, hero

    def seen_actions(self: Any, target: Any, hero: Any = None) -> Any:
        grid = actions(self, target, hero)
        captured["action_grid"] = grid
        return grid

    rows = []
    with (
        patch.object(device_policy.DevicePolicy, "evaluate", seen_evaluate),
        patch.object(device_policy.DevicePolicy, "actions", seen_actions),
        engine._arm("I"),
        torch.inference_mode(),
    ):
        ruler._reset(sim, policy, template, policy_template)
        for _ in range(ticks):
            captured.clear()
            ruler._full_tick(sim, policy, collection, 0)
            rows.append(world_digests(case, dict(captured)))
    return rows


def _memory_baseline(device: str) -> dict[str, Any]:
    if device != "cuda":
        return {"device": device}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    return {
        "total_bytes": int(total),
        "free_bytes_before": int(free),
        "reserved_bytes_before": int(reserved),
        "context_overhead_bytes": int(total - free - reserved),
    }


def _memory_result(device: str, base: dict[str, Any]) -> dict[str, Any]:
    if device != "cuda":
        return {"device": device, "headroom_fraction": None}
    torch.cuda.synchronize()
    peak_reserved = int(torch.cuda.max_memory_reserved())
    physical = base["context_overhead_bytes"] + peak_reserved
    return {
        **base,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": peak_reserved,
        "physical_peak_bytes": physical,
        "headroom_fraction": 1.0 - physical / base["total_bytes"],
        "free_bytes_after": int(torch.cuda.mem_get_info()[0]),
    }


def _blocks(case: tuple, device: str, ticks: int, repeats: int, deadline: float) -> list[dict]:
    sim, policy, template, policy_template, _collection, arm = case
    collection = ruler._collection(ticks, sim.E, sim.device)
    out = []
    for repeat in range(repeats):
        if time.monotonic() >= deadline:
            raise TimeoutError("R6 deadline before an equal-work block")
        tick_seconds = []
        with engine._arm("I"), torch.inference_mode():
            i0._sync(device)
            started = time.perf_counter()
            ruler._reset(sim, policy, template, policy_template)
            for index in range(ticks):
                began = time.perf_counter()
                ruler._full_tick(sim, policy, collection, index)
                i0._sync(device)
                tick_seconds.append(time.perf_counter() - began)
            i0._sync(device)
            elapsed = time.perf_counter() - started
        out.append(
            {
                "repeat": repeat,
                "ticks": ticks,
                "completed_world_steps": sim.E * ticks,
                "wall_seconds": elapsed,
                "tick_seconds": tick_seconds,
                "complete_world_steps_per_second": sim.E * ticks / elapsed,
                "final_digest": engine.digest((*case[:4], collection, arm)),
            }
        )
    if len({row["final_digest"] for row in out}) != 1:
        raise R6Invalid("nondeterministic equal-work block")
    return out


def measure(
    checkpoint: Path,
    unit: str,
    device: str,
    deadline: float,
    lane: str,
    base_digests: dict[str, list[list[str]]],
) -> dict[str, Any]:
    mix, worlds = units.parse(unit)
    base, double, top = units.LADDERS[lane]
    ticks = TICKS_AT_MAX[lane] * (top // worlds)
    ruler._release_cell(device)
    memory = _memory_baseline(device)
    case = None
    try:
        case = r2_bench.build_case(checkpoint, "cell", mix, worlds, device, "I_prime")
        pin_microbatches(case[1], base)
        proof = mapped_proof(case, PROOF_TICKS[lane])
        if worlds == base:
            base_digests[mix] = proof
        reference = base_digests.get(mix)
        mismatches = None
        if reference is not None:
            mismatches = sum(
                row[world] != reference[tick][world % base]
                for tick, row in enumerate(proof)
                for world in range(worlds)
            )
        with engine._arm("I"), torch.inference_mode():
            ruler._reset(case[0], case[1], case[2], case[3])
            for index in range(WARMUPS[lane]):
                ruler._full_tick(case[0], case[1], case[4], 0)
        blocks = _blocks(case, device, ticks, REPEATS[lane], deadline)
        rates = [row["complete_world_steps_per_second"] for row in blocks]
        return {
            "status": "MEASURED",
            "unit": unit,
            "mix": mix,
            "worlds": worlds,
            "ladder": [base, double, top],
            "microbatch_rows": base,
            "equal_work_world_steps": top * TICKS_AT_MAX[lane],
            "mapped_proof": {
                "ticks": PROOF_TICKS[lane],
                "reference_worlds": base,
                "mapping": "world j -> base world j mod base",
                "compared_world_ticks": None if mismatches is None else worlds * len(proof),
                "mismatched_world_ticks": mismatches,
                "exact": None if mismatches is None else mismatches == 0,
                "digest_list_sha256": hashlib.sha256(json.dumps(proof).encode()).hexdigest(),
                # Base-E digests are kept so a resumed attempt can still map larger E.
                "world_digests": proof if worlds == base else None,
            },
            "blocks": blocks,
            "median_complete_world_steps_per_second": float(np.median(rates)),
            "memory": _memory_result(device, memory),
            "r2_window": r2_cache.window(case[0]),
            "r2_stats": r2_cache.stats(case[0]),
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "status": "OOM",
            "unit": unit,
            "mix": mix,
            "worlds": worlds,
            "error": str(exc)[:2000],
            "memory": _memory_result(device, memory),
        }
    finally:
        case = None
        ruler._release_cell(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    lane = units.mode(smoke=args.smoke, canary=args.canary)
    if args.device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("R6 requires A100")
    if args.device == "cpu" and lane != "smoke":
        raise ValueError("CPU is tiny smoke only")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= DEADLINE_LIMIT[lane]:
        raise ValueError("R6 requires a finite positive bounded producer deadline")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    deadline = started + args.seconds
    commit = source_commit()
    cells = units.allocation(smoke=args.smoke, canary=args.canary)
    ns = units.namespace(commit, smoke=args.smoke, canary=args.canary)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(110):
            print(json.dumps({"event": "r6_heartbeat", "elapsed": time.monotonic() - started}))

    threading.Thread(target=heartbeat, daemon=True).start()
    base = {"schema": SCHEMA, "commit": commit, "source_sha256": source_hashes(), "lane": lane}

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        units.publish_json(output / "r6-bench.json", result)
        stop.set()
        return result

    native = integration_native.run(output / "native-parity.json", device=args.device)
    if native["status"] not in ("PASS", "PREFLIGHT"):
        return finish(
            {
                **base,
                "status": "INVALID" if native["status"] == "INVALID" else "INCOMPLETE",
                "reason": "NATIVE_PARITY_REQUIRED",
                "completed_units": [],
            }
        )
    prior = units.load_ledger(args.ledger, ns, cells)
    completed, skipped = [], []
    base_digests: dict[str, list[list[str]]] = {}
    try:
        for unit in cells:
            if unit in prior:
                mix, worlds = units.parse(unit)
                saved = prior[unit].get("mapped_proof", {}).get("world_digests")
                if worlds == units.LADDERS[lane][0] and saved is not None:
                    base_digests[mix] = saved
                skipped.append(unit)
                completed.append(unit)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("R6 deadline before a cell")
            value = measure(args.checkpoint, unit, args.device, deadline, lane, base_digests)
            units.publish_unit(output, ns, cells, unit, value)
            completed.append(unit)
            print(json.dumps({"event": "unit_done", "unit": unit, "status": value["status"]}))
    except R6Invalid as exc:
        units.build_ledger(
            output,
            output / "verified-ledger.json",
            commit=commit,
            smoke=args.smoke,
            canary=args.canary,
        )
        failure = {**base, "status": "INVALID", "reason": str(exc), "completed_units": completed}
        units.publish_json(output / "r6-failure.json", failure)
        return finish(failure)
    except TimeoutError as exc:
        units.build_ledger(
            output,
            output / "verified-ledger.json",
            commit=commit,
            smoke=args.smoke,
            canary=args.canary,
        )
        return finish(
            {**base, "status": "INCOMPLETE", "reason": str(exc), "completed_units": completed}
        )
    ledger = units.build_ledger(
        output, output / "verified-ledger.json", commit=commit, smoke=args.smoke, canary=args.canary
    )
    return finish(
        {
            **base,
            "status": "MEASURED_PENDING_MAC_REVIEW",
            "device": args.device,
            "elapsed_seconds": time.monotonic() - started,
            "namespace": ledger["namespace"],
            "completed_units": completed,
            "skipped_units": skipped,
            "native_parity": "native-parity.json",
            "ledger": "verified-ledger.json",
            "arm": "I_prime (R2) only",
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    result = run(a)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "MEASURED_PENDING_MAC_REVIEW":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
