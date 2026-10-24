"""Independent pre-timing parity proof for the I0 combined engine.

It intentionally runs before warmups and never contributes clocks to the
benchmark endpoint.  MOVE is checked against the vendored pre-MOVE device
implementation, while geometry/carry is checked against the canonical CPU
vector61 featurizer on every actual feature invocation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import move2_parity
from research.gpu_sim_spike import v8_bench as legacy

FROZEN_MOVE_SHA256 = move2_parity.FROZEN_DEVICE_SIM_SHA256


class IntegrationParityError(ValueError):
    """A required, actual I0 proof observation did not match."""


def _frozen_move_type() -> type[Any]:
    path = move2_parity.FROZEN_DEVICE_SIM
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_MOVE_SHA256:
        raise IntegrationParityError("frozen MOVE source hash differs")
    spec = importlib.util.spec_from_file_location("_i0_frozen_move", path)
    if spec is None or spec.loader is None:
        raise IntegrationParityError("cannot load frozen MOVE oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceSim


def frozen_move_clone(sim: Any) -> Any:
    """Clone a live arm state into the independently vendored MOVE class."""
    frozen = object.__new__(_frozen_move_type())
    for key, value in vars(sim).items():
        setattr(frozen, key, value.clone() if isinstance(value, torch.Tensor) else copy.copy(value))
    return frozen


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _state_match(expected: Any, actual: Any, label: str) -> dict[str, Any]:
    comparison = move2_parity.compare(expected, actual)
    if not comparison["match"]:
        raise IntegrationParityError(label + " differs from frozen MOVE: " + str(comparison))
    return comparison


def _cpu_feature_check(
    *,
    name: str,
    sim: Any,
    runtime: Any,
    original: Any,
    baseline: list[np.ndarray],
    repair_columns: set[int],
    call_records: list[dict[str, Any]],
) -> Any:
    """Wrap a real feature call with canonical CPU bytes and carry checks."""

    def checked(*args: Any, **kwargs: Any) -> torch.Tensor:
        expected, next_dist, next_id = legacy._cpu_feature_expected(sim, runtime)
        previous_dist = runtime.prev_dist.detach().cpu().numpy().copy()
        previous_id = runtime.prev_id.detach().cpu().numpy().copy()
        result = original(*args, **kwargs)
        actual = legacy._bytes(result)
        expected_all = np.tile(expected, (int(sim.E), 1))
        if result.shape != expected_all.shape:
            raise IntegrationParityError("feature call did not preserve complete row lifecycle")
        if name == "R":
            baseline.append(result.detach().cpu().numpy().copy())
            cpu_match = actual == expected_all.tobytes()
            changed: set[int] = set()
        else:
            if actual != expected_all.tobytes():
                raise IntegrationParityError("I geometry feature bytes differ from canonical CPU")
            store = sim.alive.detach().cpu().numpy().copy()
            mask = kwargs.get("memory_mask")
            if mask is not None:
                store &= mask.detach().cpu().numpy().reshape(store.shape)
            if not kwargs.get("update_memory", True):
                store[:] = False
            want_dist = np.where(store, np.tile(next_dist, (sim.E, 1)), previous_dist)
            want_id = np.where(store, np.tile(next_id, (sim.E, 1)), previous_id)
            if (
                legacy._bytes(runtime.prev_dist) != want_dist.tobytes()
                or legacy._bytes(runtime.prev_id) != want_id.tobytes()
            ):
                raise IntegrationParityError("I geometry carry differs from canonical CPU")
            candidate_index = sum(record["arm"] == "I" for record in call_records)
            if len(baseline) <= candidate_index:
                raise IntegrationParityError("I feature call has no matched R feature call")
            old = baseline[candidate_index]
            candidate = result.detach().cpu().numpy()
            changed = set(
                map(int, np.argwhere(old.view(np.uint32) != candidate.view(np.uint32))[:, 1])
            )
            if not changed <= set(legacy.REPAIR_COLUMNS):
                raise IntegrationParityError("I changed an undeclared vector61 column")
            repair_columns.update(changed)
            cpu_match = True
        call_records.append(
            {
                "arm": name,
                "cpu_match": cpu_match,
                "columns": int(result.shape[1]),
                "changed_from_R": sorted(changed),
            }
        )
        return result

    return checked


def cpu_discrepancy_register(checkpoint: Path, device: str = "cpu") -> dict[str, Any]:
    """Reproduce the inherited CPU/device deviations through the frozen MOVE lane.

    This delegates to the established independent GridBatchSim-versus-vendored
    MOVE procedure.  The result retains source states, action dtype/bytes,
    frozen baseline reproduction, ordered food/corpse alignment, and the
    first divergent field for every roster; it is diagnostic evidence, never
    a CPU-exactness pass.
    """
    from research.gpu_sim_spike import move2_bench

    result = move2_bench._cpu_discrepancies(checkpoint, device)
    if result.get("status") != "DIAGNOSTIC":
        raise IntegrationParityError("inherited CPU discrepancy reproduction is invalid")
    return result


def prove(cases: dict[str, tuple], device: str) -> dict[str, Any]:
    """Run one same-action, full-state R/I proof outside the timing endpoint.

    ``cases`` must be fresh ``integration_engine._case`` six-tuples.  The
    proof mutates them, so callers must reset/rebuild before benchmark warmups.
    A mismatch raises; success returns only observations obtained by this run.
    """
    if set(cases) != set(engine.ARMS):
        raise ValueError("I0 proof requires exactly R and I cases")
    left, right = cases["R"], cases["I"]
    engine.assert_oracle_separation(left, right)
    for name, case in cases.items():
        if len(case) != 6 or case[-1] != name:
            raise ValueError("I0 proof received a malformed arm case")

    # Reset first, then create independent frozen MOVE states before either
    # policy observes a feature.  The candidate's actual step is compared to
    # this frozen clone; it never borrows candidate movement helpers.
    for case in cases.values():
        ruler._reset(case[0], case[1], case[2], case[3])
    frozen = {name: frozen_move_clone(case[0]) for name, case in cases.items()}
    frozen_initial = {name: move2_parity.state_descriptor(value) for name, value in frozen.items()}
    tape: list[torch.Tensor] = []
    baseline_features: list[np.ndarray] = []
    repair_columns: set[int] = set()
    feature_records: list[dict[str, Any]] = []
    mechanics: dict[str, dict[str, Any]] = {}

    for name in engine.ARMS:
        sim, policy, _template, _policy_template, collection, _ = cases[name]
        runtime = policy.vector_runtime
        original_features, original_step = runtime.features, sim.step
        before_count = len(feature_records)

        def step(
            actions: torch.Tensor,
            *,
            _name: str = name,
            _original_step: Any = original_step,
        ) -> Any:
            if _name == "R":
                tape.append(actions.detach().clone())
            if len(tape) != 1:
                raise IntegrationParityError("I did not consume R's one-tick supplied action tape")
            frozen[_name].step(tape[0])
            return _original_step(tape[0])

        wrapped = _cpu_feature_check(
            name=name,
            sim=sim,
            runtime=runtime,
            original=original_features,
            baseline=baseline_features,
            repair_columns=repair_columns,
            call_records=feature_records,
        )
        with (
            engine._arm(name),
            patch.object(runtime, "features", wrapped),
            patch.object(sim, "step", step),
            torch.inference_mode(),
        ):
            ruler._full_tick(sim, policy, collection, 0)
            _sync(device)
        if len(feature_records) == before_count:
            raise IntegrationParityError(name + " did not execute a vector61 feature call")
        mechanics[name] = _state_match(frozen[name], sim, name)

    r_count = sum(record["arm"] == "R" for record in feature_records)
    i_count = sum(record["arm"] == "I" for record in feature_records)
    if r_count != i_count:
        raise IntegrationParityError("R/I actual vector61 feature call counts differ")
    # Full product state is also checked after stripping only declared routing
    # differences: geometry's direct carry caches and MOVE's route selector.
    r_state, i_state = engine.snapshot(left), engine.snapshot(right)
    r_sim, i_sim = dict(r_state["sim"]), dict(i_state["sim"])
    r_sim.pop("move_backend", None)
    i_sim.pop("move_backend", None)
    if r_sim != i_sim or r_state["collection"] != i_state["collection"]:
        raise IntegrationParityError("R/I mechanics or collection inventory differs")
    if r_state["policy"] != i_state["policy"]:
        raise IntegrationParityError("R/I unrelated policy inventory differs")
    allowed_runtime = {"prev_dist", "prev_id", "_carry_state", "sel_state"}
    r_runtime, i_runtime = r_state["runtime"], i_state["runtime"]
    if set(r_runtime) != set(i_runtime) or any(
        r_runtime[key] != i_runtime[key] for key in r_runtime if key not in allowed_runtime
    ):
        raise IntegrationParityError("R/I unrelated vector runtime inventory differs")

    return {
        "frozen_device_identity": True,
        "cpu_geometry_carry": True,
        "action_tape": {
            "ticks": len(tape),
            "shape": list(tape[0].shape),
            "sha256": hashlib.sha256(legacy._bytes(tape[0])).hexdigest(),
        },
        "frozen_move": mechanics,
        "matched_input": {
            "action_dtype": str(tape[0].dtype),
            "action_bytes_sha256": hashlib.sha256(legacy._bytes(tape[0])).hexdigest(),
            "pre_step_frozen_state": frozen_initial,
        },
        "feature_calls": feature_records,
        "declared_geometry_repair_columns": list(legacy.REPAIR_COLUMNS),
        "observed_geometry_repair_columns": sorted(repair_columns),
        "state_inventory": {"R": r_state, "I": i_state},
        "v8_route": {
            "indexed_enabled": False,
            "scope": "ordinary shared route; independent V8 packet remains binding",
        },
    }
