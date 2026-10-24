"""Matched reference/indexed movement packet; exactness precedes throughput."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import move_units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes


class MoveTerminal(RuntimeError):
    """A registered mismatch or throughput kill."""


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _state(sim: Any) -> dict[str, np.ndarray]:
    """Full public DeviceSim state excluding diagnostic/backend bookkeeping."""
    skip = {"move_backend", "_prepared", "stage_timing", "device", "cfg"}
    out: dict[str, np.ndarray] = {}
    for name, value in sorted(vars(sim).items()):
        if name not in skip and isinstance(value, torch.Tensor):
            out[name] = value.detach().cpu().numpy()
    return out


def _same(left: Any, right: Any) -> dict[str, Any]:
    a, b = _state(left), _state(right)
    if left._prepared != right._prepared:
        return {"match": False, "reason": "prepared_phase"}
    if a.keys() != b.keys():
        return {"match": False, "reason": "state_keys", "left": sorted(a), "right": sorted(b)}
    for key in a:
        if (
            a[key].shape != b[key].shape
            or a[key].dtype != b[key].dtype
            or a[key].tobytes() != b[key].tobytes()
        ):
            return {
                "match": False,
                "reason": "state",
                "field": key,
                "left_sha256": hashlib.sha256(a[key].tobytes()).hexdigest(),
                "right_sha256": hashlib.sha256(b[key].tobytes()).hexdigest(),
            }
    return {
        "match": True,
        "fields": len(a),
        "state_sha256": hashlib.sha256(b"".join(a[key].tobytes() for key in sorted(a))).hexdigest(),
    }


def _record(root: Path, unit_id: str, sim: Any, actions: torch.Tensor) -> dict[str, Any]:
    path = root / "recorded-states" / f"{unit_id}.npz"
    if path.exists():
        raise FileExistsError(f"immutable fixture already exists: {path}")
    payload = {f"state_{name}": value for name, value in _state(sim).items()}
    payload["actions"] = actions.detach().cpu().numpy()
    payload["config_json"] = np.asarray(json.dumps(dataclasses.asdict(sim.cfg), sort_keys=True))
    payload["lifecycle_json"] = np.asarray(
        json.dumps(
            {
                "prepared": sim._prepared,
                "train_mode": sim.train_mode,
                "allow_respawn": sim.allow_respawn,
                "cap": sim.cap,
            }
        )
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    meta = move_units.publish_bytes(path, stream.getvalue())
    return {"path": str(path.relative_to(root)), **meta}


def _actions(sim: Any, repeat: int) -> torch.Tensor:
    # Deterministic action tape exercises boost and normal movement without a policy cost.
    rows = torch.arange(sim.E * sim.S, device=sim.device).reshape(sim.E, sim.S)
    return ((rows + repeat) % 6).long()


def _stage_once(sim: Any, actions: torch.Tensor, device: str) -> dict[str, float]:
    """Attribute instance calls only; plain full-step timing remains separate."""
    values = {"respawn": 0.0, "collision": 0.0}
    old_respawn, old_flags, old_resolve = (
        sim._respawn,
        sim._collision_flags_reference,
        sim._resolve_collisions_reference,
    )

    def wrap(name: str, fn: Any):
        def call(*args: Any, **kwargs: Any):
            _sync(device)
            begin = time.perf_counter()
            result = fn(*args, **kwargs)
            _sync(device)
            values[name] += time.perf_counter() - begin
            return result

        return call

    sim._respawn = wrap("respawn", old_respawn)
    if sim.move_backend == "reference":
        sim._collision_flags_reference = wrap("collision", old_flags)
        sim._resolve_collisions_reference = wrap("collision", old_resolve)
    else:
        # Candidate calls this module-level function lazily; wrapping the import target
        # is scoped to this diagnostic pass and never changes the plain measurement.
        import research.gpu_sim_spike.move_cuda as move_cuda

        old_collision = move_cuda.collisions
        move_cuda.collisions = wrap("collision", old_collision)
    try:
        _sync(device)
        begin = time.perf_counter()
        sim.step(actions)
        _sync(device)
        values["step"] = time.perf_counter() - begin
    finally:
        sim._respawn, sim._collision_flags_reference, sim._resolve_collisions_reference = (
            old_respawn,
            old_flags,
            old_resolve,
        )
        if sim.move_backend == "indexed":
            move_cuda.collisions = old_collision
    return values


FROZEN_DEVICE_SIM = Path(__file__).with_name("fixtures") / "move" / "device_sim_pre_move.py"
FROZEN_DEVICE_SIM_SHA256 = "816ebd5fbd4016bb01eda5fe31ee7739f8cb7bf021b2885684f769ea63dc58ae"


def _frozen_device_sim() -> type[Any]:
    """Load the vendored cd26df50 oracle without relying on a shallow git history."""
    if hashlib.sha256(FROZEN_DEVICE_SIM.read_bytes()).hexdigest() != FROZEN_DEVICE_SIM_SHA256:
        raise MoveTerminal("frozen DeviceSim fixture digest differs")
    spec = importlib.util.spec_from_file_location("move_device_sim_pre_move", FROZEN_DEVICE_SIM)
    if spec is None or spec.loader is None:
        raise MoveTerminal("cannot load frozen DeviceSim fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceSim


def _build(
    checkpoint: Path, mix: str, worlds: int, device: str, backend: str
) -> tuple[Any, Any, Any, Any]:
    # The current reference exposes stage seams. The frozen full-step oracle is
    # invoked separately, so its monolithic legacy implementation stays intact.
    sim, policy, template, policy_template = ruler._build_case(
        checkpoint, "long-" + mix, worlds, device
    )
    sim.move_backend = backend
    template.move_backend = backend
    if device == "cuda" and os.environ.get("SNAKE_COLAB_CANARY") == "1":
        # Real same-package CUDA policy action outside the timed stage; required
        # by the qualified canary observer. This supplies no CPU-policy proof.
        _, _, hero = policy.evaluate(sim)
        policy.actions(sim, hero)
        ruler._reset(sim, policy, template, policy_template)
    return sim, policy, template, policy_template


def _frozen_step_oracle(
    checkpoint: Path, mix: str, worlds: int, device: str, actions: torch.Tensor, candidate: Any
) -> dict[str, Any]:
    """Independent pre-MOVE full-step oracle from vendored cd26df50 bytes."""
    from research.gpu_sim_spike import realistic_benchmark as realistic

    world = realistic._long_world(checkpoint, "long-" + mix, (800, 900, 1000, 1100, 1300, 1500))
    frozen = _frozen_device_sim().from_reference(world.sim, device=device).repeat_interleave(worlds)
    frozen._prepared = False
    frozen.step(actions)
    result = _same(frozen, candidate)
    result.update(
        {"oracle": "vendored-cd26df50-device_sim", "fixture_sha256": FROZEN_DEVICE_SIM_SHA256}
    )
    return result


def _cpu_grid_seam(checkpoint: Path, mix: str, root: Path | None = None) -> dict[str, Any]:
    """Compare logical CPU outputs bytewise; retain representation differences.

    Ordered food lists and fixed device food buffers use different storage, so
    compare their live prefixes and all corpse flags, plus device padding
    invariants. Numeric outputs are compared with their ORIGINAL dtype/bytes.
    """
    from research.gpu_sim_spike import realistic_benchmark as realistic
    from research.gpu_sim_spike.device_sim import DeviceSim

    grid = realistic._long_world(checkpoint, "long-" + mix, (800, 900, 1000, 1100, 1300, 1500)).sim
    grid.train_mode = False
    grid.allow_respawn = False
    # Already prepared by _long_world. Initial lifecycle is skipped; eating
    # may still trigger CPU food-replacement RNG, which is recorded explicitly.
    sim = DeviceSim.from_reference(grid, device="cpu")
    sim.train_mode = False
    sim.allow_respawn = False
    sim.move_backend = "indexed"
    actions = torch.ones((1, sim.S), dtype=torch.long)
    fixture = _record(root, "cpu-" + mix, sim, actions) if root else None
    baseline = _frozen_device_sim().from_reference(grid, device="cpu")
    baseline.train_mode = False
    baseline.allow_respawn = False
    cpu_rng_before = grid._rngs[0]._rng.getstate()
    grid.step(actions.numpy())
    sim.step(actions)
    baseline.step(actions)
    optimization = _same(baseline, sim)
    fields = (
        "bodies",
        "head_ptr",
        "seg_count",
        "length",
        "alive",
        "direction",
        "boost_frames",
        "frames_since_food",
        "respawn_timer",
        "_reward_prev_length",
        "frame",
        "_boosted_this_step",
        "_last_reward",
        "_last_done",
        "_last_death_cause",
        "_last_kills",
        "_last_food_ate",
        "_last_ambient_food_ate",
        "_last_corpse_food_ate",
        "_last_transition_valid",
    )
    results = {}
    for name in fields:
        left = np.asarray(getattr(grid, name))
        right = getattr(sim, name).detach().numpy()
        match = (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.tobytes() == right.tobytes()
        )
        results[name] = {
            "match": match,
            "cpu_dtype": str(left.dtype),
            "device_dtype": str(right.dtype),
            "cpu_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
            "device_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
        }
        if not match and left.size <= 12 and right.size <= 12:
            results[name].update(cpu_values=left.tolist(), device_values=right.tolist())
    count = int(sim.food_count[0])
    left_food = np.asarray(grid.food_cells[0], dtype=np.int64).reshape(-1, 2)
    right_food = sim.food_cells[0, :count].numpy()
    corpse = np.asarray([tuple(cell) in grid.corpse_cells[0] for cell in left_food], dtype=np.bool_)
    results["ordered_food"] = {
        "match": left_food.shape == right_food.shape and left_food.tobytes() == right_food.tobytes()
    }
    results["corpse_flags"] = {
        "match": corpse.tobytes() == sim.corpse_food[0, :count].numpy().tobytes()
    }
    results["food_padding"] = {
        "match": bool(sim.food_present[0, :count].all())
        and not bool(sim.food_present[0, count:].any())
        and not bool(sim.corpse_food[0, count:].any())
    }
    bad = [name for name, value in results.items() if not value["match"]]
    return {
        "status": "PASS" if not bad else "MISMATCH",
        "mix": mix,
        "scope": "GridBatchSim recorded prepared seam; respawn disabled",
        "fields": results,
        "mismatches": bad,
        "recorded_fixture": fixture,
        "rng_scope": "Counter RNG is not CPU MT; no logical-draw compatibility claimed",
        "cpu_food_rng_advanced": grid._rngs[0]._rng.getstate() != cpu_rng_before,
        "cpu_policy_identity": "NOT_TESTED: supplied straight-action tape",
        "frozen_optimization_parity": optimization,
    }


def _measure_pair(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    repeats: int,
    warmups: int,
    root: Path | None,
    unit_id: str,
    deadline: float = float("inf"),
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {"reference": [], "indexed": []}
    parity: list[dict[str, Any]] = []
    fixture = None
    for rep in range(repeats):
        if time.monotonic() >= deadline:
            raise TimeoutError("MOVE wall cap reached")
        pair = {}
        for backend in (("reference", "indexed") if rep % 2 == 0 else ("indexed", "reference")):
            sim, policy, template, policy_template = _build(
                checkpoint, mix, worlds, device, backend
            )
            try:
                for warm in range(warmups):
                    ruler._reset(sim, policy, template, policy_template)
                    sim.move_backend = backend
                    sim.step(_actions(sim, warm))
                ruler._reset(sim, policy, template, policy_template)
                sim.move_backend = backend
                actions = _actions(sim, rep)
                if fixture is None and root is not None:
                    fixture = _record(root, unit_id, sim, actions)
                diagnostic = _stage_once(sim, actions, device)
                ruler._reset(sim, policy, template, policy_template)
                sim.move_backend = backend
                _sync(device)
                started = time.perf_counter()
                sim.step(actions)
                _sync(device)
                plain = time.perf_counter() - started
                pair[backend] = sim
                rows[backend].append(
                    {"repeat": rep, "plain_step_seconds": plain, "stage_seconds": diagnostic}
                )
            finally:
                del policy, template, policy_template
        proof = _same(pair["reference"], pair["indexed"])
        proof["frozen_oracle"] = _frozen_step_oracle(
            checkpoint, mix, worlds, device, actions, pair["indexed"]
        )
        proof["repeat"] = rep
        parity.append(proof)
        del pair
    if not all(row["match"] and row["frozen_oracle"]["match"] for row in parity):
        raise MoveTerminal("MOVE_MISMATCH")
    summary = {
        backend: {
            "median_step_seconds": statistics.median(x["plain_step_seconds"] for x in values),
            "median_stage_seconds": {
                stage: statistics.median(x["stage_seconds"][stage] for x in values)
                for stage in ("respawn", "collision", "step")
            },
            "samples": values,
        }
        for backend, values in rows.items()
    }
    candidate, baseline = summary["indexed"], summary["reference"]
    # Plain synchronized full-step medians decide the bar; wrapped stages only
    # attribute time and deliberately retain their instrumentation overhead.
    speed = baseline["median_step_seconds"] / candidate["median_step_seconds"]
    if worlds == 1024 and speed < 2.0:
        raise MoveTerminal(f"MOVE_SLOW_E1024 speedup={speed:.6f}")
    return {
        "mix": mix,
        "worlds": worlds,
        "backends": summary,
        "stage_speedup": speed,
        "parity": parity,
        "recorded_fixture": fixture,
        "cpu_verification": "see cpu-parity.json",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    commit, cells = source_commit(), move_units.allocation(smoke=args.smoke, canary=args.canary)
    existing = move_units.load_ledger(
        args.ledger, move_units.namespace(commit, smoke=args.smoke, canary=args.canary), cells
    )
    deadline, output = time.monotonic() + args.seconds, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    # Carry verified immutable replay payloads into this fresh attempt without
    # recomputing completed timing units. CPU preflight fixtures are diagnostic.
    import shutil

    prior = json.loads(args.ledger.read_text())
    for fixture in prior.get("fixtures", []):
        if not Path(fixture["path"]).name.startswith("cell-"):
            continue
        for suffix in ("", ".sha256", ".complete"):
            src = Path(str(args.ledger.parent / fixture["path"]) + suffix)
            dst = Path(str(output / fixture["path"]) + suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise FileExistsError(dst)
            shutil.copyfile(src, dst)
    cpu_rows = [
        _cpu_grid_seam(args.checkpoint, mix, output) for mix in ("scripted", "frozen", "mixed")
    ]
    cpu_proof = {
        "schema": "gpu-sim-spike-move-cpu/v1",
        "commit": commit,
        "status": "PASS" if all(row["status"] == "PASS" for row in cpu_rows) else "INVALID",
        "rosters": cpu_rows,
    }
    move_units.publish_json(output / "cpu-parity.json", cpu_proof)
    if cpu_proof["status"] != "PASS":
        result = {
            "schema": "gpu-sim-spike-move-bench/v1",
            "status": "INVALID",
            "commit": commit,
            "reason": "MOVE_CPU_GRID_MISMATCH",
            "completed_units": [],
            "source_sha256": source_hashes(),
            "cpu_parity": "cpu-parity.json",
            "performance_measurement": "NOT_RUN: CPU byte gate failed",
        }
        move_units.publish_json(output / "move-bench.json", result)
        return result
    for unit_id in cells:
        if unit_id in existing:
            move_units.publish_unit(
                output,
                move_units.namespace(commit, smoke=args.smoke, canary=args.canary),
                cells,
                unit_id,
                existing[unit_id],
            )
            continue
        mix, worlds = unit_id.split("-")[1], int(unit_id.rsplit("e", 1)[1])
        if time.monotonic() >= deadline:
            raise TimeoutError("MOVE wall cap reached")
        cell = _measure_pair(
            args.checkpoint,
            mix,
            worlds,
            args.device,
            args.repeats,
            args.warmups,
            output,
            unit_id,
            deadline,
        )
        move_units.publish_unit(
            output,
            move_units.namespace(commit, smoke=args.smoke, canary=args.canary),
            cells,
            unit_id,
            cell,
        )
        print(json.dumps({"event": "move_cell_complete", "unit_id": unit_id}), flush=True)
    ledger = move_units.build_ledger(
        output,
        args.output_dir / "verified-ledger.json",
        commit=commit,
        smoke=args.smoke,
        canary=args.canary,
    )
    result = {
        "schema": "gpu-sim-spike-move-bench/v1",
        "status": "COMPLETE",
        "commit": commit,
        "source_sha256": source_hashes(),
        "namespace": ledger["namespace"],
        "completed_units": [row["unit_id"] for row in ledger["units"]],
        "cpu_parity_scope": (
            "recorded GridBatchSim seam only; counter-RNG divergence excludes universal CPU parity"
        ),
        "kill_rule": "mismatch or E1024 stage speedup below 2x is terminal",
        "ledger": "verified-ledger.json",
    }
    move_units.publish_json(args.output_dir / "move-bench.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
