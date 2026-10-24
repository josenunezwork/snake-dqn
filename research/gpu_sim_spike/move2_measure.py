"""Matched recorded inputs and profiler-free timing against the frozen device."""

from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike import move2_parity as parity


def restore(sim: Any, template: Any) -> None:
    """Restore exact fields and dtypes, removing state first created by warmup.

    The older resident reset copies into existing storage and does not restore a
    float64 pre-step reward after step replaced it with float32, or remove the
    first-step capacity_overflow field. A measured repeat starts at the complete
    original recorded state, including that inventory and dtype lifecycle.
    """
    for name in set(vars(sim)) - set(vars(template)):
        delattr(sim, name)
    for name, value in vars(template).items():
        current = getattr(sim, name, None)
        if isinstance(value, torch.Tensor):
            if (
                isinstance(current, torch.Tensor)
                and current.shape == value.shape
                and current.dtype == value.dtype
                and current.device == value.device
            ):
                current.copy_(value)
            else:
                setattr(sim, name, value.clone())
        else:
            setattr(sim, name, copy.copy(value))


def measure(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    repeats: int,
    warmups: int,
    deadline: float,
) -> dict[str, Any]:
    """Run the frozen ruler; hashes/descriptors replace full-state trace exports."""
    from research.gpu_sim_spike import move2_bench as bench
    from research.gpu_sim_spike import realistic_benchmark as realistic
    from research.gpu_sim_spike.move_bench import _stage_once

    if repeats != 3 or warmups != 3:
        raise ValueError("MOVE2 ruler requires exactly three repeats and warmups")
    samples: dict[str, list[float]] = {"reference": [], "indexed": []}
    stages: dict[str, list[dict[str, float]]] = {"reference": [], "indexed": []}
    proofs, inputs = [], []
    for repeat in range(repeats):
        pair, recorded = {}, None
        for backend in (("reference", "indexed") if repeat % 2 == 0 else ("indexed", "reference")):
            if time.monotonic() >= deadline:
                raise TimeoutError("MOVE2 wall cap before next timing arm")
            current, policy, current_template, policy_template = bench._build(
                checkpoint, mix, worlds, device, backend
            )
            try:
                if device == "cuda":
                    # Required by the family canary; outside isolated step timing.
                    _q, _legal, hero = policy.evaluate(current)
                    policy.actions(current, hero)
                restore(current, current_template)
                if backend == "reference":
                    world = realistic._long_world(
                        checkpoint, "long-" + mix, (800, 900, 1000, 1100, 1300, 1500)
                    )
                    sim = bench._frozen_class().from_reference(world.sim, device=device)
                    sim = sim.repeat_interleave(worlds)
                    sim._prepared = False
                    template = sim.clone()
                    initial = parity.compare(sim, current)
                    if not initial["match"]:
                        raise bench.Move2Terminal(f"FROZEN_INITIAL_STATE:{initial}")
                else:
                    sim, template = current, current_template
                if sim.cap != 10400:
                    raise bench.Move2Terminal("MOVE2_FULL_CAPACITY_REQUIRED")
                for warmup in range(warmups):
                    restore(sim, template)
                    sim.step(bench._actions(sim, warmup))
                restore(sim, template)
                descriptor = parity.state_descriptor(sim)
                if descriptor["non_tensors"].get("stage_timing", {}).get("enabled"):
                    raise bench.Move2Terminal("PROFILER_ENABLED_IN_PLAIN_MEASUREMENT")
                actions = bench._actions(sim, repeat)
                action_bytes = actions.detach().cpu().numpy().tobytes()
                arm_input = {
                    "state": descriptor,
                    "actions": {
                        "shape": list(actions.shape),
                        "dtype": "int64",
                        "sha256": parity.digest_bytes(action_bytes),
                        "generator": "(arange(E*S).reshape(E,S)+repeat)%6",
                        "repeat": repeat,
                    },
                }
                if recorded is None:
                    recorded = arm_input
                elif json.dumps(recorded, sort_keys=True) != json.dumps(arm_input, sort_keys=True):
                    raise bench.Move2Terminal("REFERENCE_INDEXED_RECORDED_INPUTS_DIFFER")
                bench._sync(device)
                started = time.perf_counter()
                sim.step(actions)
                bench._sync(device)
                samples[backend].append(time.perf_counter() - started)
                pair[backend] = sim
                # Attribution is a separate pass on the uninstrumented current
                # seam twin, never the authoritative frozen/reference stopwatch.
                diagnostic = current_template.clone()
                stages[backend].append(_stage_once(diagnostic, actions, device))
                del diagnostic
            finally:
                del policy, current_template, policy_template
        same = parity.compare(pair["reference"], pair["indexed"])
        proofs.append(
            {
                "repeat": repeat,
                "candidate_vs_reference": same,
                "candidate_vs_frozen": same,
                "comparator": "independent vendored cd26df50 full step",
            }
        )
        if not same["match"]:
            raise bench.Move2Terminal(f"FROZEN_DEVICE_BYTE_IDENTITY_FAILED:{same}")
        assert recorded is not None
        inputs.append(
            {
                "repeat": repeat,
                "input_sha256": parity.digest_bytes(json.dumps(recorded, sort_keys=True).encode()),
                **recorded,
            }
        )
        del pair
        if time.monotonic() >= deadline:
            raise TimeoutError("MOVE2 wall cap after timing repeat")
    speed = statistics.median(samples["reference"]) / statistics.median(samples["indexed"])
    return {
        "mix": mix,
        "worlds": worlds,
        "warmed_interleaved_repeats": repeats,
        "warmups": warmups,
        "capacity": 10400,
        "lengths": [800, 900, 1000, 1100, 1300, 1500],
        "reference_seconds": samples["reference"],
        "indexed_seconds": samples["indexed"],
        "speedup": speed,
        "frozen_device_identity": proofs,
        "recorded_inputs": inputs,
        "stage_seconds_diagnostic_only": stages,
        "authoritative_measurement": (
            "synchronized profiler-free full simulator step; policy excluded"
        ),
        "reference_source_sha256": parity.FROZEN_DEVICE_SIM_SHA256,
    }
