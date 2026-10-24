"""Fresh MOVE2 optimization benchmark: frozen-device identity before timing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import move2_parity as parity
from research.gpu_sim_spike import move2_units
from research.gpu_sim_spike.benchmark import source_commit


class Move2Terminal(RuntimeError):
    """A benchmark identity or route condition stopped the packet."""


def _source_hashes() -> dict[str, str]:
    """Hash only the frozen transport's MOVE2 source closure, never all package files."""
    package = Path(__file__).parent
    names = (
        "move2_bench.py",
        "move2_bench.sh",
        "move2_parity.py",
        "move2_units.py",
        "move_cuda.py",
        "move_cuda.cu",
        "move_cuda.cpp",
        "device_sim.py",
        "a100_ruler.py",
        "realistic_benchmark.py",
        "move2_events.py",
        "move2_measure.py",
        "move_bench.py",
    )
    result = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in names
        if (package / name).is_file()
    }
    result["fixtures/move/device_sim_pre_move.py"] = hashlib.sha256(
        parity.FROZEN_DEVICE_SIM.read_bytes()
    ).hexdigest()
    repo = package.parents[1]
    for relative in (
        "src/simd_env/grid_sim.py",
        "src/simd_env/grid_sim_nb.py",
        "src/simd_env/batch_sim.py",
        "src/simd_env/rng.py",
        "src/simd_env/grid_scenarios.py",
        "src/core/reward_events.py",
        "src/core/mechanics_constants.py",
    ):
        path = repo / relative
        if not path.is_file():
            raise Move2Terminal(f"required CPU source absent from MOVE2 closure: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _frozen_class() -> type[Any]:
    if (
        hashlib.sha256(parity.FROZEN_DEVICE_SIM.read_bytes()).hexdigest()
        != parity.FROZEN_DEVICE_SIM_SHA256
    ):
        raise Move2Terminal("frozen DeviceSim digest differs")
    spec = importlib.util.spec_from_file_location("move2_frozen_device", parity.FROZEN_DEVICE_SIM)
    if spec is None or spec.loader is None:
        raise Move2Terminal("cannot load frozen DeviceSim")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceSim


def _actions(sim: Any, repeat: int) -> torch.Tensor:
    rows = torch.arange(sim.E * sim.S, device=sim.device).reshape(sim.E, sim.S)
    return ((rows + repeat) % 6).long()


def _build(
    checkpoint: Path, mix: str, worlds: int, device: str, backend: str
) -> tuple[Any, Any, Any, Any]:
    sim, policy, template, policy_template = ruler._build_case(
        checkpoint, "long-" + mix, worlds, device
    )
    sim.move_backend = backend
    template.move_backend = backend
    return sim, policy, template, policy_template


def _cpu_discrepancies(checkpoint: Path, device: str = "cpu") -> dict[str, Any]:
    """Reproduce inherited deviations on frozen baseline; never admits CPU correctness."""
    from research.gpu_sim_spike import realistic_benchmark as realistic
    from research.gpu_sim_spike.device_sim import DeviceSim

    rows = []
    expected_inherited = {
        "scripted": ["_last_reward"],
        "frozen": ["_last_reward", "ordered_food", "corpse_flags"],
        "mixed": ["_last_reward", "ordered_food", "corpse_flags"],
    }
    for mix in ("scripted", "frozen", "mixed"):
        grid = realistic._long_world(
            checkpoint, "long-" + mix, (800, 900, 1000, 1100, 1300, 1500)
        ).sim
        grid.train_mode = False
        grid.allow_respawn = False
        candidate, baseline = DeviceSim.from_reference(
            grid, device=device
        ), _frozen_class().from_reference(grid, device=device)
        for sim in (candidate, baseline):
            sim.train_mode = False
            sim.allow_respawn = False
            sim.move_backend = "indexed" if sim is candidate else "reference"
        initial_baseline = parity.state_descriptor(baseline)
        initial_candidate = parity.state_descriptor(candidate)
        initial_identity = parity.compare(baseline, candidate)
        actions = torch.ones((1, candidate.S), dtype=torch.long)
        rng_before = grid._rngs[0]._rng.getstate()
        baseline_rng_before = {
            name: parity.digest_bytes(getattr(baseline, name).detach().cpu().numpy().tobytes())
            for name in ("rng_key", "rng_counter")
        }
        actions = actions.to(device)
        grid.step(actions.cpu().numpy())
        candidate.step(actions)
        baseline.step(actions)
        frozen = parity.compare(baseline, candidate)
        # The CPU mapping remains lossless: compare every device tensor that
        # has an independent GridBatchSim field, with its original dtype/bytes.
        fields = {
            name: parity.field_record(
                getattr(grid, name), getattr(candidate, name).detach().cpu().numpy()
            )
            for name in parity.tensor_inventory(candidate)
            if hasattr(grid, name) and isinstance(getattr(grid, name), np.ndarray)
        }
        count = int(candidate.food_count[0])
        food_cpu = np.asarray(grid.food_cells[0], dtype=np.int64).reshape(-1, 2)
        fields["ordered_food"] = parity.field_record(
            food_cpu, candidate.food_cells[0, :count].detach().cpu().numpy()
        )
        corpse_cpu = np.asarray(
            [tuple(cell) in grid.corpse_cells[0] for cell in food_cpu], dtype=np.bool_
        )
        fields["corpse_flags"] = parity.field_record(
            corpse_cpu, candidate.corpse_food[0, :count].detach().cpu().numpy()
        )
        fields["food_corpse_alignment"] = {
            "cpu_sha256": parity.digest_bytes(food_cpu.tobytes() + corpse_cpu.tobytes()),
            "device_sha256": parity.digest_bytes(
                candidate.food_cells[0, :count].detach().cpu().numpy().tobytes()
                + candidate.corpse_food[0, :count].detach().cpu().numpy().tobytes()
            ),
            "match": food_cpu.tobytes() + corpse_cpu.tobytes()
            == candidate.food_cells[0, :count].detach().cpu().numpy().tobytes()
            + candidate.corpse_food[0, :count].detach().cpu().numpy().tobytes(),
            "interpretation": (
                "aligned food/corpse diagnostic only; not an independent corpse claim"
            ),
        }
        fields["food_padding"] = {
            "match": bool(candidate.food_present[0, :count].all())
            and not bool(candidate.food_present[0, count:].any())
            and not bool(candidate.corpse_food[0, count:].any())
        }
        bad = [
            name
            for name, row in fields.items()
            if name != "food_corpse_alignment" and not row["match"]
        ]
        # Padding is a device-only invariant rather than a CPU representation
        # difference; an invalid padding field is a new discrepancy.
        if not fields["food_padding"]["match"]:
            bad.append("food_padding")
        reward = fields["_last_reward"]
        reward["cpu_bytes_hex"] = getattr(grid, "_last_reward").tobytes().hex()
        reward["device_bytes_hex"] = candidate._last_reward.detach().cpu().numpy().tobytes().hex()
        baseline_rng_after = {
            name: parity.digest_bytes(getattr(baseline, name).detach().cpu().numpy().tobytes())
            for name in ("rng_key", "rng_counter")
        }
        cpu_rng_after = grid._rngs[0]._rng.getstate()
        valid = initial_identity["match"] and frozen["match"] and bad == expected_inherited[mix]
        rows.append(
            {
                "mix": mix,
                "input": {
                    "actions_sha256": parity.digest_bytes(actions.detach().cpu().numpy().tobytes()),
                    "initial_frozen_state": initial_baseline,
                    "initial_candidate_state": initial_candidate,
                    "initial_identity": initial_identity,
                },
                "frozen_device_identity": frozen,
                "fields": fields,
                "inherited_fields": bad,
                "expected_inherited_fields": expected_inherited[mix],
                "valid_inherited_scope": valid,
                "first_divergent_stage": "post-GridBatchSim.step vs post-frozen-device.step",
                "first_divergent_field": bad[0] if bad else None,
                "cpu_rng": {
                    "before_sha256": parity.digest_bytes(pickle.dumps(rng_before, protocol=4)),
                    "after_sha256": parity.digest_bytes(pickle.dumps(cpu_rng_after, protocol=4)),
                    "advanced": cpu_rng_after != rng_before,
                },
                "baseline_counter_rng": {
                    "before": baseline_rng_before,
                    "after": baseline_rng_after,
                },
            }
        )
    return {
        "schema": "gpu-sim-spike-move2-cpu-discrepancies/v1",
        "status": "DIAGNOSTIC" if all(row["valid_inherited_scope"] for row in rows) else "INVALID",
        "classification": "CPU_EXACTNESS_NOT_CLAIMED",
        "rows": rows,
    }


def _native_coverage(device: str) -> dict[str, Any]:
    """Run the explicit event suite; static labels and skipped tests never qualify CUDA."""
    from research.gpu_sim_spike import move2_events

    return move2_events.run(device)


def _measure(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    repeats: int,
    warmups: int,
    deadline: float,
) -> dict[str, Any]:
    # Kept as the runtime seam so the compact event/timing implementation can
    # reuse the builders above without a top-level circular import.
    from research.gpu_sim_spike import move2_measure

    return move2_measure.measure(
        checkpoint=checkpoint,
        mix=mix,
        worlds=worlds,
        device=device,
        repeats=repeats,
        warmups=warmups,
        deadline=deadline,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    commit = source_commit()
    cells = move2_units.allocation(smoke=args.smoke, canary=args.canary)
    output, deadline = args.output_dir, time.monotonic() + args.seconds
    output.mkdir(parents=True, exist_ok=True)
    existing = move2_units.load_ledger(
        args.ledger, move2_units.namespace(commit, smoke=args.smoke, canary=args.canary), cells
    )
    discrepancies = _cpu_discrepancies(args.checkpoint, args.device)
    discrepancies.update(
        {
            "commit": commit,
            "source_sha256": _source_hashes(),
            "device": args.device,
        }
    )
    if args.expected_cpu_discrepancies:
        expected = json.loads(args.expected_cpu_discrepancies.read_text())
        discrepancies["expected_input_sha256"] = hashlib.sha256(
            args.expected_cpu_discrepancies.read_bytes()
        ).hexdigest()
        discrepancies["matches_expected"] = expected.get("rows") == discrepancies.get("rows")
        if not discrepancies["matches_expected"]:
            discrepancies["status"] = "INVALID"
    move2_units.publish_json(output / "cpu-discrepancies.json", discrepancies)
    # The same compact, immutable inventory is also named as the ruling's
    # discrepancy register so downstream pack/reducer code never has to infer
    # an unbound correspondence between diagnostics and admission.
    register = {
        **discrepancies,
        "schema": "gpu-sim-spike-move2-discrepancy-register/v1",
        "ruling_sha256": "51b637243a57db2a64baa0d80174386994e3e9f71d545589c31628ad58b4948b",
    }
    move2_units.publish_json(output / "discrepancy-register.json", register)
    native = _native_coverage(args.device)
    native.update({"commit": commit, "source_sha256": _source_hashes()})
    move2_units.publish_json(output / "native-parity.json", native)
    if discrepancies["status"] == "INVALID":
        result = {
            "schema": "gpu-sim-spike-move2-bench/v1",
            "status": "INVALID",
            "reason": "CPU_DISCREPANCY_FINGERPRINT_CHANGED",
            "commit": commit,
            "source_sha256": _source_hashes(),
            "cpu_discrepancies": "cpu-discrepancies.json",
            "discrepancy_register": "discrepancy-register.json",
            "native_parity": "native-parity.json",
            "completed_units": [],
        }
        move2_units.publish_json(output / "move2-bench.json", result)
        return result
    if args.device == "cuda" and native["status"] != "PASS":
        result = {
            "schema": "gpu-sim-spike-move2-bench/v1",
            "status": "INCOMPLETE" if native["status"] == "INCOMPLETE" else "INVALID",
            "reason": "NATIVE_PARITY_REQUIRED",
            "commit": commit,
            "source_sha256": _source_hashes(),
            "cpu_discrepancies": "cpu-discrepancies.json",
            "discrepancy_register": "discrepancy-register.json",
            "native_parity": "native-parity.json",
            "completed_units": [],
        }
        move2_units.publish_json(output / "move2-bench.json", result)
        return result
    completed, measured = [], {}
    try:
        for unit_id in cells:
            if unit_id in existing:
                move2_units.publish_unit(
                    output,
                    move2_units.namespace(commit, smoke=args.smoke, canary=args.canary),
                    cells,
                    unit_id,
                    existing[unit_id],
                )
                completed.append(unit_id)
                continue
            mix, worlds = unit_id.split("-")[1], int(unit_id.rsplit("e", 1)[1])
            row = _measure(
                args.checkpoint, mix, worlds, args.device, args.repeats, args.warmups, deadline
            )
            measured[unit_id] = row
            move2_units.publish_unit(
                output,
                move2_units.namespace(commit, smoke=args.smoke, canary=args.canary),
                cells,
                unit_id,
                row,
            )
            completed.append(unit_id)
    except Move2Terminal as exc:
        status = "STOP_ROUTE" if "LT_2X" in str(exc) else "INVALID"
        result = {
            "schema": "gpu-sim-spike-move2-bench/v1",
            "status": status,
            "reason": str(exc),
            "commit": commit,
            "source_sha256": _source_hashes(),
            "cpu_discrepancies": "cpu-discrepancies.json",
            "discrepancy_register": "discrepancy-register.json",
            "native_parity": "native-parity.json",
            "completed_units": completed,
        }
        move2_units.publish_json(output / "move2-bench.json", result)
        return result
    except TimeoutError as exc:
        result = {
            "schema": "gpu-sim-spike-move2-bench/v1",
            "status": "INCOMPLETE",
            "reason": str(exc),
            "commit": commit,
            "source_sha256": _source_hashes(),
            "cpu_discrepancies": "cpu-discrepancies.json",
            "native_parity": "native-parity.json",
            "completed_units": completed,
        }
        move2_units.publish_json(output / "move2-bench.json", result)
        return result
    ledger = move2_units.build_ledger(
        output, output / "verified-ledger.json", commit=commit, smoke=args.smoke, canary=args.canary
    )
    all_rows = {**existing, **measured}
    stop = any(
        row.get("speedup", float("inf")) < 2.0
        for unit, row in all_rows.items()
        if unit.endswith("e1024")
    )
    result = {
        "schema": "gpu-sim-spike-move2-bench/v1",
        "status": "STOP_ROUTE" if stop else "PASS",
        "reason": "STOP_ROUTE_E1024_LT_2X" if stop else None,
        "commit": commit,
        "source_sha256": _source_hashes(),
        "namespace": ledger["namespace"],
        "completed_units": completed,
        "cpu_discrepancies": "cpu-discrepancies.json",
        "discrepancy_register": "discrepancy-register.json",
        "native_parity": "native-parity.json",
        "ledger": "verified-ledger.json",
        "rule": "INVALID precedes INCOMPLETE precedes STOP_ROUTE; E1024 requires >=2x",
    }
    move2_units.publish_json(output / "move2-bench.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-cpu-discrepancies", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
