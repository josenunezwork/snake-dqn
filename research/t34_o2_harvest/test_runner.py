"""Pure schema checks for the O2 triple producer."""

from __future__ import annotations

import pickle
from concurrent.futures import Future

import pytest

from research.t34_o2_harvest.runner import (
    O2RunError,
    ScientificFault,
    _run_parallel,
    canonical_world_row,
    forecast_admissible,
    forecast_row_steps,
)


def _arm(digest: str) -> dict:
    return {"initial_digest": digest, "mass_integral": 0.0, "controller": {}}


def test_canonical_row_requires_paired_source_identity() -> None:
    digest = "a" * 64
    row = canonical_world_row(
        row={"world_id": "x", "world_seed": 1, "mix": "scripted"},
        arms={"bare": _arm(digest), "d": _arm(digest), "do": _arm(digest)},
        checkpoint_sha256="b" * 64,
        cfg={"horizon": 5000, "mechanics_version": 2},
        elapsed_seconds=1.0,
    )
    assert row["paired"]["source_digest"] == digest
    assert row["operational"]["threads"] == 1


def test_canonical_row_refuses_unpaired_source_state() -> None:
    with pytest.raises(O2RunError, match="paired arms"):
        canonical_world_row(
            row={"world_id": "x"},
            arms={"bare": _arm("a" * 64), "d": _arm("b" * 64), "do": _arm("a" * 64)},
            checkpoint_sha256="c" * 64,
            cfg={"horizon": 5000},
            elapsed_seconds=1.0,
        )


def test_forecast_is_the_three_arm_two_shared_tree_upper_bound() -> None:
    assert forecast_row_steps(worlds=1, roots=8, actions=6) == 15000 + 48000


def test_forecast_refusal_is_a_pre_study_gate() -> None:
    assert not forecast_admissible(forecast_seconds=11521, cap_seconds=11520, deadline=float("inf"))
    assert not forecast_admissible(
        forecast_seconds=float("nan"), cap_seconds=11520, deadline=float("inf")
    )


def test_scientific_fault_round_trips_through_a_spawn_worker_pickle() -> None:
    fault = ScientificFault.from_exception(
        row={"world_id": "w", "world_seed": 1},
        arm="do",
        decision=7,
        phase="study",
        exc=ValueError("nonfinite"),
    )
    restored = pickle.loads(pickle.dumps(fault))
    assert restored.payload == fault.payload


def test_bounded_scheduler_does_not_admit_a_second_wave_after_a_fault() -> None:
    class ImmediateExecutor:
        def __init__(self) -> None:
            self.submitted: list[int] = []

        def submit(self, _worker, task):
            self.submitted.append(task)
            future = Future()
            if task == 0:
                future.set_exception(RuntimeError("fault"))
            else:
                future.set_result(task)
            return future

        def shutdown(self, **_kwargs) -> None:
            return None

    executor = ImmediateExecutor()
    with pytest.raises(RuntimeError, match="fault"):
        _run_parallel(
            [0, 1, 2, 3],
            lambda value: value,
            2,
            phase="study",
            executor_factory=lambda: executor,
        )
    assert executor.submitted == [0, 1]
