"""Pure schema checks for the dynamically chunked O2 V2 producer."""

from __future__ import annotations

import copy
import pickle
from concurrent.futures import Future

import pytest
from scipy.stats import t

from research.t34_o2_harvest import analyze_v2, bank_v2, plan_v2, runner_v2
from research.t34_o2_harvest.runner_v2 import (
    O2RunError,
    ScientificFault,
    _run_parallel,
    canonical_world_row,
    forecast_admissible,
    forecast_row_steps,
)
from research.t34_round0.common import B8_SHA


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


def test_forecast_requires_the_registered_chunk_world_count() -> None:
    with pytest.raises(TypeError):
        forecast_row_steps()  # type: ignore[call-arg]


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


def test_progress_telemetry_excludes_unsubmitted_work(monkeypatch, capsys) -> None:
    """A full first wave reports no completion before its futures are consumed."""

    class ImmediateExecutor:
        def submit(self, _worker, task):
            future = Future()
            future.set_result(task)
            return future

        def shutdown(self, **_kwargs) -> None:
            return None

    monkeypatch.setattr(runner_v2, "wait", lambda pending, **_kwargs: (set(pending), set()))
    times = iter((0.0, 121.0))
    monkeypatch.setattr(runner_v2.time, "monotonic", lambda: next(times))
    assert runner_v2._run_parallel(
        [0, 1], lambda value: value, 2, phase="study", executor_factory=ImmediateExecutor
    ) == [0, 1]
    assert '"done": 0' in capsys.readouterr().out


def test_holm_v2_rejects_a_result_that_only_passed_the_old_point_one_family() -> None:
    summaries = {
        name: {"mean": 1.8, "se": 1.0, "df": 100.0} for name in ("do_minus_bare", "do_minus_d")
    }
    result = analyze_v2.holm(summaries)
    assert {entry["holm_tail_alpha"] for entry in result.values()} == {0.025, 0.05}
    assert not any(entry["holm_pass"] for entry in result.values())


def _v2_arm(mass: int, *, dead: bool) -> dict:
    decisions = 99 if dead else 100
    actions = [3] + [i % 3 for i in range(decisions - 1)]
    return {
        "mass_integral": mass * decisions / 5000,
        "mass_trace": [mass] * decisions,
        "survival_fraction": (decisions - int(dead)) / 5000,
        "no_death": int(not dead),
        "deaths": int(dead),
        "decisions": decisions,
        "boost_decisions": 1,
        "action_counts": [actions.count(i) for i in range(6)],
        "action_trace": actions,
        "unconditional_faults": [],
        "initial_digest": "a" * 64,
        "final_digest": "b" * 64,
        "controller": {},
    }


def _v2_records(bank: dict) -> list[dict]:
    records = []
    for index, world in enumerate(bank["rows"]):
        dead = index % 2 == 0
        records.append(
            {
                "schema": "t34-o2-harvest-world/v2",
                "world": world,
                "horizon": 5000,
                "checkpoint_sha256": B8_SHA,
                "cfg": {"mechanics_version": 2},
                "operational": {"device": "cpu", "threads": 1},
                "arms": {
                    "bare": _v2_arm(80, dead=dead),
                    "d": _v2_arm(90, dead=dead),
                    "do": _v2_arm(130, dead=False),
                },
            }
        )
    return records


def test_v2_full_bank_validation_and_decision_use_dynamic_1260_world_denominators(
    monkeypatch,
) -> None:
    bank = bank_v2.materialize()
    assert len(bank["rows"]) == 1260
    assert {row["mix"] for row in bank["rows"]} == {"scripted", "frozen", "mixed"}
    assert all(sum(row["mix"] == mix for row in bank["rows"]) == 420 for mix in plan_v2.MIXES)
    records = _v2_records(bank)
    monkeypatch.setattr(analyze_v2, "_validate_arm", lambda _arm: None)
    monkeypatch.setattr(analyze_v2, "_validate_controller", lambda _arm, _name: None)
    original_stratified = analyze_v2._stratified
    confidences: list[float] = []

    def tracked_stratified(groups, confidence):
        confidences.append(confidence)
        return original_stratified(groups, confidence)

    monkeypatch.setattr(analyze_v2, "_stratified", tracked_stratified)
    analyze_v2._records(records, bank["rows"])
    result = analyze_v2.decision(records, bank["rows"])
    assert result["schema"] == "t34-o2-decision/v2"
    assert result["per_mix"]["scripted"]["absolute_alive"] == pytest.approx(0.02)
    assert confidences.count(0.10) == 2  # mass clear-stop remains the registered upper90 guard
    assert confidences.count(0.05) == 1  # survival guard remains its original two-sided 95% CI
    assert -1.5 + float(t.ppf(0.90, 100)) < 0 < -1.5 + float(t.ppf(0.95, 100))
    with pytest.raises(EOFError, match="1260"):
        analyze_v2._records(records[:-1], bank["rows"])
    tampered = copy.deepcopy(records)
    tampered[0], tampered[1] = tampered[1], tampered[0]
    with pytest.raises(analyze_v2.InvalidEvidence, match="reordered"):
        analyze_v2._records(tampered, bank["rows"])


def test_v2_registered_chunks_bind_distinct_130_and_110_world_caps() -> None:
    plan = plan_v2.optimal_capacity()
    registration = {"bank": bank_v2.materialize(), "chunks": plan["chunks"], "caps": plan_v2.caps()}
    for chunk in (plan["chunks"][0], plan["chunks"][-1]):
        manifest = {
            "env": {"T34_O2_START": str(chunk["start"]), "T34_O2_COUNT": str(chunk["count"])},
            "o2": {key: chunk[key] for key in ("start", "count", "mix")},
        }
        selected, caps = analyze_v2._registered_chunk(manifest, registration)
        assert selected == chunk
        assert caps["units_per_job"] == chunk["caps"]["units_per_job"]
        assert caps["ordinary_job_seconds"] == chunk["caps"]["ordinary_job_seconds"]
    bad = {
        "env": {"T34_O2_START": "0", "T34_O2_COUNT": "130"},
        "o2": {"start": 0, "count": 130, "mix": "scripted"},
    }
    with pytest.raises(analyze_v2.InvalidEvidence, match="metadata"):
        analyze_v2._registered_chunk(bad, registration)
