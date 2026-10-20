"""Numerical and resumability checks; never execute a retention world."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.t34_pathb import retention_oc_v2 as oc
from research.t34_pathb.analyze_v2 import _stratified


def _pool() -> dict:
    rows = [[float(i * 5), (i - 16) / 200, 0.9] for i in range(32)]
    return {
        "t3_pools": {"b8_s3": rows},
        "watch_baselines": [{"decision_mean_length": float(i + 5)} for i in range(25)],
        "source_files": {"original-development.json": "1" * 64},
    }


def test_independent_pooled_intervals_match_retention_analyzer() -> None:
    samples = np.random.default_rng(22).normal(size=(4, 3, 32))
    means, lower, _ = oc.intervals(samples, 0.05)
    for index, sample in enumerate(samples):
        expected = _stratified({mix: sample[i].tolist() for i, mix in enumerate(oc.MIXES)}, 0.05)
        assert means[index] == pytest.approx(expected["mean"], abs=1e-12)
        assert lower[index] == pytest.approx(expected["lower"], abs=1e-9)


def test_zero_spread_and_strict_survival_boundary() -> None:
    samples = np.zeros((1, 3, 32))
    mean, lower, mixlower = oc.intervals(samples, 0.05)
    assert mean.tolist() == [0.0]
    assert lower.tolist() == [0.0]
    assert (lower - 0.05 > -0.05).tolist() == [False]
    assert (mixlower - 0.075 > -0.075).tolist() == [[False, False, False]]


def test_scenarios_are_exact_amendment_margins() -> None:
    scenarios = oc.scenarios()
    assert scenarios["pooled_minus_0.05"] == [-0.05] * 3
    assert scenarios["no_change"] == [0.0] * 3
    for i, mix in enumerate(oc.MIXES):
        expected = [0.0] * 3
        expected[i] = -0.075
        assert scenarios[f"{mix}_minus_0.075"] == expected
    assert oc.THETAS == (20, 26.8, 40, 60, 80)


def test_batch_reproducible_and_joint_is_subset_of_mass_power() -> None:
    pool = _pool()
    first = oc.simulate_batch(pool, 15, 0, 250)
    assert first == oc.simulate_batch(pool, 15, 0, 250)
    assert first != oc.simulate_batch(pool, 15, 250, 250)
    assert all(0 <= value <= 250 for value in first.values())
    for key, value in first.items():
        if "|no_change|" in key and key.endswith("joint_mass_and_band"):
            mass_key = key.replace("joint_mass_and_band", "mass_point_and_lb")
            sign_key = key.replace("joint_mass_and_band", "mass_lb_positive")
            assert value <= first[mass_key] <= first[sign_key]


def test_numerical_thread_pools_are_capped() -> None:
    assert all(
        oc.os.environ[name] == "1"
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    )


def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.t34_pathb import retention_v2

    monkeypatch.setattr(oc, "development_pool", _pool)
    monkeypatch.setattr(retention_v2, "source_inventory", lambda repo: {"simulator.py": "abc"})


def test_resumed_work_matches_uninterrupted_and_receipt_is_create_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate(monkeypatch)
    out, state = tmp_path / "out.json", tmp_path / "state.json"
    clock = iter([0.0, 0.0, 2.0])
    original_clock = oc.time.monotonic
    monkeypatch.setattr(oc.time, "monotonic", lambda: next(clock))
    paused = oc.run(out, state, max_seconds=1)
    assert paused["status"] == "INCOMPLETE"
    assert paused["completed"] == 250
    assert not out.exists()
    stale = state.with_suffix(state.suffix + ".tmp")
    stale.write_text("interrupted legacy checkpoint write")
    orphan = tmp_path / f".{state.name}.abandoned.tmp"
    orphan.write_text("interrupted unique checkpoint write")
    monkeypatch.setattr(oc.time, "monotonic", original_clock)
    oc.run(out, state, resume=True)
    assert stale.read_text() == "interrupted legacy checkpoint write"
    assert orphan.read_text() == "interrupted unique checkpoint write"
    assert list(tmp_path.glob(f".{state.name}.*.tmp")) == [orphan]
    direct = tmp_path / "direct.json"
    oc.run(direct, tmp_path / "direct-state.json")
    assert json.loads(out.read_text()) == json.loads(direct.read_text())
    receipt = json.loads(out.read_text())
    assert receipt["completed"] == 20000
    assert receipt["development_only"] is True
    assert receipt["efficacy_rows_produced"] is False
    assert receipt["active_arms"] == ["paced", "unpaced"]
    assert receipt["world_pool_sha256"] == oc._digest(receipt["development_pool"])
    with pytest.raises(FileExistsError):
        oc.run(out, tmp_path / "other-state.json")


def test_resume_refuses_source_or_development_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate(monkeypatch)
    out, state = tmp_path / "out.json", tmp_path / "state.json"
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(oc.time, "monotonic", lambda: next(clock))
    assert oc.run(out, state, max_seconds=1)["status"] == "INCOMPLETE"
    data = json.loads(state.read_text())
    data["identity"]["world_pool_sha256"] = "changed"
    state.write_text(json.dumps(data))
    monkeypatch.setattr(oc.time, "monotonic", lambda: 0.0)
    with pytest.raises(ValueError, match="identity changed"):
        oc.run(out, state, resume=True)


def test_counts_are_not_a_scientific_pass_when_a_regression_cell_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate(monkeypatch)
    original_batch = oc.simulate_batch

    def biased(pool: dict, seed: int, offset: int, count: int) -> dict[str, int]:
        values = original_batch(pool, seed, offset, count)
        key = (
            "t3_dense_b8_s3_scripted_shape_all_mixes|pooled_minus_0.05|"
            "theta=80|alpha=0.1|joint_mass_and_band"
        )
        values[key] = count
        return values

    monkeypatch.setattr(oc, "simulate_batch", biased)
    out = tmp_path / "out.json"
    result = oc.run(out, tmp_path / "state.json")
    assert result["status"] == "FAIL"
    receipt = json.loads(out.read_text())
    assert receipt["acceptance"]["failed_cells"]


@pytest.mark.parametrize("reps", [0, 2000, 19999, 20001, 100001])
def test_registered_simulation_size_and_time_bounds(reps: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reps"):
        oc.run(tmp_path / "out.json", tmp_path / "state.json", reps=reps)
    with pytest.raises(ValueError, match="max-seconds"):
        oc.run(tmp_path / "out.json", tmp_path / "state.json", max_seconds=301)


def test_checkpoint_write_failure_preserves_state_and_cleans_own_temp(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    original = '{"completed": 250}\n'
    state.write_text(original)
    with pytest.raises(ValueError):
        oc._atomic_state(state, {"invalid": float("nan")})
    assert state.read_text() == original
    assert not list(tmp_path.glob(f".{state.name}.*.tmp"))
