"""Focused pure checks for the T3 replay evidence contract."""

from __future__ import annotations

import numpy as np
import pytest

from research.t3_diagnostics_20261016.replay import (
    DisagreementObserver,
    SeriesObserver,
    _death_frames,
    _mass_summary,
)


def test_advisory_greedy_uses_normal_only_fallback_and_float64_argmax() -> None:
    q = np.array(
        [[1.0, 2.0, 3.0, 100.0, 101.0, 102.0], [1.0, 4.0, 3.0, 9.0, 8.0, 7.0]], dtype=np.float32
    )
    advisory = np.array([[False] * 6, [False, True, False, False, False, False]])
    np.testing.assert_array_equal(
        DisagreementObserver._greedy(q, advisory), np.array([2, 1], dtype=np.int8)
    )


def test_series_observer_uses_post_mass_and_one_based_death_frame() -> None:
    observer = SeriesObserver([11], 2)
    observer(
        {
            "frame": 0,
            "pre": {"alive": np.array([True]), "logical_mass": np.array([7])},
            "post": {"alive": np.array([True]), "logical_mass": np.array([8])},
        }
    )
    observer(
        {
            "frame": 1,
            "pre": {"alive": np.array([True]), "logical_mass": np.array([8])},
            "post": {"alive": np.array([False]), "logical_mass": np.array([8])},
        }
    )
    np.testing.assert_array_equal(observer.length, [[8, 0]])
    np.testing.assert_array_equal(observer.death_frame, [2])
    np.testing.assert_array_equal(observer.death_length, [8])


def test_terminal_status_rejects_missing_or_failed_final(tmp_path, monkeypatch) -> None:
    import research.t3_diagnostics_20261016.replay as replay

    monkeypatch.setattr(replay, "TRAIN", tmp_path)
    (tmp_path / "status.txt").write_text("LABELS_s10 exit=1\n")
    with pytest.raises(ValueError, match="all 15"):
        replay._terminal_t3_status()


def test_death_frames_censors_survivors() -> None:
    series = {"alive": np.array([[True, False, False], [True, True, True]], dtype=bool)}
    assert _death_frames(series) == [2]


def test_mass_summary_uses_h5000_common_survivor_cohort_and_exact_identity(tmp_path) -> None:
    series = tmp_path / "series"
    series.mkdir()
    for seed in range(10, 15):
        ctrl_length = np.full((1, 5000), 10, dtype=np.int32)
        arm_length = np.full((1, 5000), 12, dtype=np.int32)
        ctrl_alive = np.ones((1, 5000), dtype=bool)
        arm_alive = np.ones((1, 5000), dtype=bool)
        arm_alive[0, 3000:] = False
        arm_length[0, 3000:] = 0
        np.savez_compressed(
            series / f"CTRL_s{seed}.npz",
            seeds=np.array([seed]),
            length=ctrl_length,
            alive=ctrl_alive,
        )
        np.savez_compressed(
            series / f"LABELS_s{seed}.npz",
            seeds=np.array([seed]),
            length=arm_length,
            alive=arm_alive,
        )
    summary = _mass_summary("LABELS", tmp_path)
    assert summary["common_survivor_length_delta"]["2000"]["worlds"] == 0
    assert summary["per_seed"][0]["common_survivor_length_delta"]["2000"]["worlds"] == 0
    assert summary["per_seed"][0]["identity_residual_mass_frames"] == 0
