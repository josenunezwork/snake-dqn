"""Hash-bound inherited discrepancy classification; no CPU dynamics waiver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.gpu_sim_spike import realistic_benchmark as realistic
from research.gpu_sim_spike import v8_split_bench as split
from research.gpu_sim_spike.device_sim import DeviceSim


def _frozen_type() -> type:
    root = Path(__file__).parent
    path = root / "fixtures/v8_split/device_sim_85180521.py"
    bound = json.loads((root / "V8_SPLIT_FROZEN_COMPONENTS.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != bound["files"]["device_sim.py"]:
        raise split.ParityMismatch("frozen discrepancy oracle digest differs")
    spec = importlib.util.spec_from_file_location("v8_split_discrepancy_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceSim


def _record(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    match = (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )
    result = {
        "match": match,
        "cpu_dtype": str(left.dtype),
        "device_dtype": str(right.dtype),
        "cpu_shape": list(left.shape),
        "device_shape": list(right.shape),
        "cpu_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "device_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
    }
    if not match:
        result.update(cpu_bytes_hex=left.tobytes().hex(), device_bytes_hex=right.tobytes().hex())
    return result


def probe(checkpoint: Path, device: str) -> dict[str, Any]:
    rows = []
    expected = {
        "scripted": ["_last_reward"],
        "frozen": ["_last_reward", "ordered_food", "corpse_flags"],
        "mixed": ["_last_reward", "ordered_food", "corpse_flags"],
    }
    for mix in ("scripted", "frozen", "mixed"):
        cpu = realistic._long_world(
            checkpoint, "long-" + mix, (800, 900, 1000, 1100, 1300, 1500)
        ).sim
        cpu.train_mode = False
        cpu.allow_respawn = False
        baseline = _frozen_type().from_reference(cpu, device=device)
        candidate = DeviceSim.from_reference(cpu, device=device)
        for sim in (baseline, candidate):
            sim.train_mode = False
            sim.allow_respawn = False
        if split.inventory(baseline) != split.inventory(candidate):
            raise split.ParityMismatch("discrepancy initial frozen device identity failed")
        initial = split.inventory(baseline)
        initial.pop("device")
        initial_rng = {n: split._value(getattr(baseline, n)) for n in ("rng_key", "rng_counter")}
        rng_before = cpu._rngs[0]._rng.getstate()
        actions = torch.ones((1, candidate.S), dtype=torch.long, device=device)
        cpu.step(actions.cpu().numpy())
        baseline.step(actions)
        candidate.step(actions)
        if split.inventory(baseline) != split.inventory(candidate):
            raise split.ParityMismatch("discrepancy frozen-device output identity failed")
        fields = {
            key: _record(getattr(cpu, key), value.detach().cpu().numpy())
            for key, value in vars(candidate).items()
            if isinstance(value, torch.Tensor) and isinstance(getattr(cpu, key, None), np.ndarray)
        }
        count = int(candidate.food_count[0])
        food = np.asarray(cpu.food_cells[0], dtype=np.int64).reshape(-1, 2)
        corpse = np.asarray([tuple(cell) in cpu.corpse_cells[0] for cell in food], dtype=np.bool_)
        fields["ordered_food"] = _record(food, candidate.food_cells[0, :count].cpu().numpy())
        fields["corpse_flags"] = _record(corpse, candidate.corpse_food[0, :count].cpu().numpy())
        if (
            not bool(candidate.food_present[0, :count].all())
            or bool(candidate.food_present[0, count:].any())
            or bool(candidate.corpse_food[0, count:].any())
        ):
            raise split.ParityMismatch("unexpected food padding discrepancy")
        mismatches = [k for k, v in fields.items() if not v["match"]]
        if mismatches != expected[mix]:
            raise split.ParityMismatch("new CPU discrepancy field: " + str(mismatches))
        rows.append(
            {
                "mix": mix,
                "initial_state": initial,
                "fields": fields,
                "inherited_fields": mismatches,
                "frozen_device_identity": True,
                "actions": split._value(actions),
                "initial_rng": initial_rng,
                "final_rng": {n: split._value(getattr(baseline, n)) for n in initial_rng},
                "cpu_rng_before_sha256": hashlib.sha256(
                    pickle.dumps(rng_before, protocol=4)
                ).hexdigest(),
                "cpu_rng_after_sha256": hashlib.sha256(
                    pickle.dumps(cpu._rngs[0]._rng.getstate(), protocol=4)
                ).hexdigest(),
                "first_divergent_stage": "prepared matched step",
                "first_divergent_field": mismatches[0],
            }
        )
    return {
        "schema": "v8-split-discrepancies/v1",
        "classification": "CPU_EXACTNESS_NOT_CLAIMED",
        "rows": rows,
        "counter_rng_is_cpu_mt": False,
    }
