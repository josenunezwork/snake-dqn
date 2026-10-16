"""Focused pure contracts for the declared T4a post-hoc readouts."""

from __future__ import annotations

import numpy as np
import pytest

from research.t4a_posthoc_20261016 import capture, spillover


def _q(actions: list[int]) -> np.ndarray:
    values = np.zeros((len(actions), 6), dtype=np.float32)
    values[np.arange(len(actions)), actions] = 1.0
    return values


def test_spillover_uses_raw_argmax_exact_exclusion_and_three_way_support() -> None:
    bank = {
        "side": np.asarray(["decisive", "decisive", "counter", "counter"]),
        "length": np.asarray([100, 800, 100, 800]),
        "fingerprint": np.asarray(["bank-a", "bank-b", "bank-c", "bank-d"]),
        "ctrl_q": _q([0, 0, 0, 0]),
        "candidate_q": _q([3, 0, 0, 1]),
    }
    trajectory = {
        "length": np.asarray([100, 100, 800, 800, 1200]),
        "fingerprint": np.asarray(["bank-a", "outside-1", "outside-2", "outside-3", "outside-4"]),
        "ctrl_q": _q([0, 0, 0, 0, 0]),
        "candidate_q": _q([0, 3, 0, 1, 0]),
    }
    summary = spillover.spillover_summary(bank, trajectory)
    assert summary["c_exclusion"]["excluded_bank_fingerprint_rows"] == 1
    assert summary["c_exclusion"]["outside_bank_rows"] == 4
    # c is conditioned by its own length strata; tied values fall in D10 under
    # the established right-sided decile convention.
    c = summary["c_ctrl_trajectory_outside_bank_matched_length_and_decile"]
    assert c["by_length_and_decile"]["lt400"]["10"]["decisions"] == 1
    assert c["by_length_and_decile"]["lt400"]["10"]["cross_boost_changes"] == 1
    matched = summary["matched_common_support"]
    assert matched["a_decisive_bank_rows"]["retained_rows"] == 2
    assert matched["b_counter_bank_rows"]["retained_rows"] == 2
    # The 1200-length trajectory state has no a/b counterpart and is excluded.
    assert matched["c_ctrl_trajectory_outside_bank_rows"]["retained_rows"] == 3


def test_spillover_requires_both_bank_sides_and_outside_trajectory_row() -> None:
    base = {
        "side": np.asarray(["decisive"]),
        "length": np.asarray([100]),
        "fingerprint": np.asarray(["same"]),
        "ctrl_q": _q([0]),
        "candidate_q": _q([0]),
    }
    trajectory = {
        "length": np.asarray([100]),
        "fingerprint": np.asarray(["same"]),
        "ctrl_q": _q([0]),
        "candidate_q": _q([0]),
    }
    with pytest.raises(ValueError, match="both decisive and counter"):
        spillover.spillover_summary(base, trajectory)


def test_growth_uses_fixed_h5000_and_frame_specific_shared_survivors(tmp_path) -> None:
    seeds = np.asarray([10, 11])
    ctrl_length = np.full((2, 5000), 10, dtype=np.int32)
    candidate_length = np.full((2, 5000), 12, dtype=np.int32)
    ctrl_alive = np.ones((2, 5000), dtype=bool)
    candidate_alive = np.ones((2, 5000), dtype=bool)
    candidate_alive[1, 2999:] = False
    candidate_length[1, 2999:] = 0
    ctrl_path, candidate_path = tmp_path / "ctrl.npz", tmp_path / "candidate.npz"
    np.savez_compressed(ctrl_path, seeds=seeds, length=ctrl_length, alive=ctrl_alive)
    np.savez_compressed(candidate_path, seeds=seeds, length=candidate_length, alive=candidate_alive)
    summary = spillover.growth_summary(candidate_path, ctrl_path)
    assert summary["frames"]["1000"]["shared_survivors"] == 2
    assert summary["frames"]["5000"]["shared_survivors"] == 1
    assert summary["frames"]["5000"]["candidate_minus_ctrl_mean_alive_length"] == 2.0


def test_fingerprint_covers_layout_dtype_shape_and_bytes() -> None:
    local = np.zeros((1, 1, 2), dtype=np.uint8)
    global_ = np.zeros((1, 1, 2), dtype=np.uint8)
    scalars = np.zeros((1, 2), dtype=np.float32)
    first = spillover.observation_fingerprints(local, global_, scalars)
    local[0, 0, 0] = 1
    second = spillover.observation_fingerprints(local, global_, scalars)
    assert first.shape == (1,)
    assert first[0] != second[0]


def test_ctrl_trajectory_writer_preserves_seed_slot_and_frame(tmp_path) -> None:
    # Avoid loading a model: writing is deliberately separable from collection.
    observer = object.__new__(capture.CtrlTrajectoryCapture)
    observer.rows = {
        "TWO_SIDED": [
            {
                "seed_slot": 7,
                "frame": 123,
                "length": 456,
                "fingerprint": "a" * 64,
                "ctrl_q": _q([0])[0],
                "candidate_q": _q([3])[0],
            }
        ]
    }
    path = observer.write(tmp_path, 12)["TWO_SIDED"]
    with np.load(path, allow_pickle=False) as saved:
        assert saved["seed_slot"].tolist() == [7]
        assert saved["frame"].tolist() == [123]
        assert saved["length"].tolist() == [456]
