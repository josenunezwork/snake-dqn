"""Focused Round 0 producer contracts; no gameplay or campaign arm is run."""

from __future__ import annotations

import json
import pickle
import platform
from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_round0 import runner


@pytest.mark.parametrize("opt_in", [None, False, True])
def test_teacher_cuda_vector_recipe_can_preserve_paired_native_dispatch(tmp_path, opt_in) -> None:
    from research.t34_exit_b8.teacher_v2 import run_worlds

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "sentinel").write_text("fixture")
    opponents = SimpleNamespace(batch_device_vectors=False)
    teacher = SimpleNamespace(device="cuda", _opponents=opponents)
    config = {
        "checkpoint": checkpoint,
        "output_dir": output,
        "rows": [{"world_id": "fixture"}],
        "teacher": teacher,
    }
    if opt_in is not None:
        config["batch_device_vectors"] = opt_in
    # Stop before constructing any world or executing any policy, while testing
    # the actual CUDA recipe selection independently of CUDA availability.
    with pytest.raises(ValueError, match="output directory must be fresh"):
        run_worlds(config)
    assert opponents.batch_device_vectors is (True if opt_in is None else opt_in)


@pytest.mark.parametrize(
    "message",
    [
        "illegal action",
        "illegal hero action",
        "hero action must be in [0, 5]",
    ],
)
def test_real_illegal_action_faults_are_classified(message: str) -> None:
    assert runner._fault_marker(ValueError(message)) == "illegal_action"


def test_nonfinite_q_is_fault_but_io_error_is_incomplete() -> None:
    assert runner._fault_marker(ValueError("policy returned non-finite Q values")) == "nonfinite_q"
    assert runner._fault_marker(OSError("missing input file")) is None


def _bank(*, preflight: bool, rows: int) -> dict:
    horizon = 500 if preflight else 5000
    return {
        "schema": "t34-round0-bank/v1",
        "preflight": preflight,
        "rows": [
            {
                "world_id": f"round/{index}",
                "world_seed": index + 1,
                "mix": "scripted" if index == 0 else "mixed",
                "opponents": [["scripted", "greedy_food"]],
                "opponent_veto": "v8",
                "horizon": horizon,
            }
            for index in range(rows)
        ],
    }


def test_preflight_bank_requires_exactly_two_h500_rows() -> None:
    assert len(runner._validate_bank(_bank(preflight=True, rows=2), preflight=True)) == 2
    with pytest.raises(runner.Round0Error, match="requires 2 rows"):
        runner._validate_bank(_bank(preflight=True, rows=1), preflight=True)


def test_production_bank_requires_whole_240_world_allocation() -> None:
    with pytest.raises(runner.Round0Error, match="requires 240 rows"):
        runner._validate_bank(_bank(preflight=False, rows=2), preflight=False)


def test_trace_stripping_removes_every_label_but_keeps_actions_and_hashes() -> None:
    raw = {
        "schema": "t34-exit-world/v1",
        "actions": [[1, 1, 1, 1, 1, 1]],
        "hashes": {"0": "before", "1": "after"},
        "labels": [{"decision": 0, "decisive": True}],
        "teacher_v2": {"label_rule": "bisection-radius3-prepared-cache"},
    }
    stripped = runner._strip_labels(raw)
    assert stripped["labels"] == []
    assert "teacher_v2" not in stripped
    assert stripped["actions"] == raw["actions"]
    assert stripped["hashes"] == raw["hashes"]


def test_create_only_json_receipt_has_runtime_export_markers(tmp_path) -> None:
    path = tmp_path / "chunk.json"
    runner._write_new(path, {"schema": "fixture"})
    assert path.with_name("chunk.json.sha256").read_text().startswith(runner.sha256(path))
    assert path.with_name("chunk.json.complete").is_file()
    with pytest.raises(FileExistsError):
        runner._write_new(path, {"schema": "replacement"})


def test_execution_recipe_rejects_wrong_mode_or_thread_count(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    runner._validate_execution(preflight=True, device="cpu", threads=1, seconds=1)
    with pytest.raises(runner.Round0Error, match="preflight requires"):
        runner._validate_execution(preflight=True, device="cuda", threads=1, seconds=1)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    runner._validate_execution(preflight=False, device="cpu", threads=1, seconds=1)
    with pytest.raises(runner.Round0Error, match="production requires"):
        runner._validate_execution(preflight=False, device="cuda", threads=1, seconds=1)
    with pytest.raises(runner.Round0Error, match="production requires"):
        runner._validate_execution(preflight=False, device="cpu", threads=2, seconds=1)


def test_production_source_fence_requires_clean_ratified_bytes(monkeypatch, tmp_path) -> None:
    ratification = tmp_path / "ratification.json"
    ratification.write_text("{}")
    hashes = {"research/t34_round0/runner.py": "fixture"}
    monkeypatch.setattr(runner, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_tracked_source_is_clean", lambda: True)
    monkeypatch.setattr(runner, "_source_is_detached", lambda: True)
    monkeypatch.setattr("research.t34_round0.common.source_hashes", lambda _root: hashes)
    monkeypatch.setattr(
        runner,
        "_load",
        lambda _path: {"commit": "a" * 40, "source_hashes": hashes},
    )
    snapshot = runner._source_fence(preflight=False, ratification=ratification)
    assert snapshot == {"commit": "a" * 40, "source_hashes": hashes}
    runner._check_source_fence(snapshot, preflight=False)
    monkeypatch.setattr(runner, "_tracked_source_is_clean", lambda: False)
    with pytest.raises(runner.Round0Error, match="tracked source changed"):
        runner._check_source_fence(snapshot, preflight=False)


def test_pair_worker_failure_preserves_arm_and_fault_marker() -> None:
    failure = runner.Round0PairError(
        "teacher", "world/7", ValueError("policy returned non-finite Q values")
    )
    assert failure.arm == "teacher"
    assert failure.world_id == "world/7"
    assert failure.marker == "nonfinite_q"


def test_pair_worker_failure_is_pickle_safe_for_spawn_pool_transport() -> None:
    failure = runner.Round0PairError("bare", "world/8", ValueError("illegal hero action"))
    copied = pickle.loads(pickle.dumps(failure))
    assert isinstance(copied, runner.Round0PairError)
    assert (copied.arm, copied.world_id, copied.marker, str(copied)) == (
        "bare",
        "world/8",
        "illegal_action",
        "illegal hero action",
    )


def test_worker_metric_receipt_requires_counted_legal_actions() -> None:
    metrics = {
        "mass_integral": 1.0,
        "survival_fraction": 1.0,
        "boost_share": 0.0,
        "one_action_share": 1.0,
        "decisions": 2,
        "deaths": 0,
        "boost_decisions": 0,
        "no_death": 1,
        "action_counts": [2, 0, 0, 0, 0, 0],
    }
    assert runner._validated_metrics(metrics)["decisions"] == 2
    metrics["action_counts"] = [1, 0, 0, 0, 0, 0]
    with pytest.raises(runner.Round0Error, match="invalid action counts"):
        runner._validated_metrics(metrics)


def test_out_of_order_ready_results_do_not_reduce_worker_submission_capacity() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.submitted: list[tuple[object, dict]] = []

        def submit(self, function, request):
            future = object()
            self.submitted.append((function, request))
            return future

    pool = FakePool()
    early_world_is_still_running = object()
    pending = {early_world_is_still_running: 0}
    # Pair 1 completed before pair 0 and is deliberately retained for canonical
    # publication.  It must not make the scheduler leave two workers idle.
    ready = {1: {"index": 1}}
    next_index = runner._submit_available(
        pool,
        pending,
        next_index=2,
        stop_index=6,
        workers=3,
        request_for_index=lambda index: {"index": index},
    )
    assert ready[1]["index"] == 1
    assert [request["index"] for _function, request in pool.submitted] == [2, 3]
    assert len(pending) == 3
    assert next_index == 4


def test_dispatch_admission_requires_canonical_registration_and_fixed_attempt(tmp_path) -> None:
    registration = tmp_path / "ratification.json"
    registration.write_text(json.dumps({"commit": "a" * 40, "status": "RATIFIED"}))
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {
                "schema": "t34-round0-admission/v1",
                "phase_path": "/evidence/t34/chunks/chunk-007/attempt-01",
                "registration_sha256": runner._canonical_sha(runner._load(registration)),
                "claim_commit": "b" * 40,
            }
        )
    )
    assert runner._validate_dispatch_admission(admission, registration)["claim_commit"] == "b" * 40
    admission.write_text(
        json.dumps(
            {
                "schema": "t34-round0-admission/v1",
                "phase_path": "/evidence/t34/chunks/chunk-007/attempt-03",
                "registration_sha256": runner._canonical_sha(runner._load(registration)),
                "claim_commit": "b" * 40,
            }
        )
    )
    with pytest.raises(runner.Round0Error, match="fixed registered chunk attempt"):
        runner._validate_dispatch_admission(admission, registration)


def test_production_recipe_declares_a5_k12_single_thread_layout() -> None:
    assert runner.PRODUCTION_WORKERS == 12


def test_action_window_checks_exact_rolling_100k_without_counter_rebuild() -> None:
    window = runner._ActionWindow()
    for _ in range(100000):
        collapsed = window.add(1)
    assert collapsed
    assert window.counts == [0, 100000, 0, 0, 0, 0]
    assert window.add(2)
    assert window.counts == [0, 99999, 1, 0, 0, 0]
    for _ in range(9998):
        assert window.add(2)
    assert not window.add(2)
    assert window.counts == [0, 90000, 10000, 0, 0, 0]


def test_bare_prepares_before_evaluation_without_environment_drift(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "b8.pth"
    checkpoint.write_bytes(b"fixture")
    order: list[str] = []
    before = {
        name: __import__("os").environ.get(name)
        for name in ("NUMBA_NUM_THREADS", "SNAKE_DQN_DEVICE")
    }

    class FakeTeacher:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def evaluate(self, _world):
            order.append("evaluate")
            assert order[0] == "prepare"
            return SimpleNamespace(q=np.zeros(6), greedy_action=lambda: 1)

    class Sim:
        E = S = 1

        def __init__(self) -> None:
            self.alive = np.asarray([[True]])

    class World:
        sim = Sim()

    def step(_world, _teacher, _action):
        order.append("step")
        return [1, 0, 0, 0, 0, 0]

    monkeypatch.setattr(runner, "ExitTeacher", FakeTeacher)
    monkeypatch.setattr(runner, "B8Policy", lambda _device: object())
    monkeypatch.setattr(runner, "build_world", lambda _row, _checkpoint: World())
    monkeypatch.setattr(runner, "prepare_world", lambda world: order.append("prepare") or world)
    monkeypatch.setattr(runner, "_step", step)
    monkeypatch.setattr(runner, "state_hash", lambda _sim: "state")
    trace = runner._bare_trace(
        {"world_id": "w", "world_seed": 1, "mix": "scripted"}, checkpoint, frames=1, device="cpu"
    )
    assert trace["actions"] == [[1, 0, 0, 0, 0, 0]]
    assert order == ["prepare", "evaluate", "step"]
    assert before == {name: __import__("os").environ.get(name) for name in before}


def test_trace_metrics_keep_dead_frames_in_full_horizon_denominator(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "b8.pth"
    checkpoint.write_bytes(b"fixture")

    class Sim:
        E = S = 1

        def __init__(self) -> None:
            self.alive = np.asarray([[True]])
            self.length = np.asarray([[10]])
            self.frame = 0

        def step(self, _actions) -> None:
            self.frame += 1
            self.alive[0][0] = False

    class Runtime:
        def observe_step(self, _sim) -> None:
            pass

    class World:
        def __init__(self) -> None:
            self.sim, self.runtime = Sim(), Runtime()

    monkeypatch.setattr(runner, "build_world", lambda _row, _checkpoint: World())
    monkeypatch.setattr(runner, "prepare_world", lambda world: world)
    monkeypatch.setattr(runner, "state_hash", lambda sim: f"h{sim.frame}")
    trace = {
        "schema": "t34-exit-world/v1",
        "complete": True,
        "checkpoint_sha256": runner.sha256(checkpoint),
        "row": {"world_id": "w"},
        "frames": 5,
        "actions": [[1, 0, 0, 0, 0, 0]],
        "hashes": {"0": "h0", "1": "h1"},
    }
    metrics = runner._metrics_from_trace(trace, checkpoint)
    assert metrics["mass_integral"] == 0
    assert metrics["survival_fraction"] == 0
    assert metrics["deaths"] == 1
