"""Synthetic only tests for the sealed F2 C0 fallback."""

from __future__ import annotations

import numpy as np

from research.a100_a2.run import fit_unit
from research.bl_residual.f2_c0 import _fit_c0
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint


def _data(n: int = 20) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3403)
    world = np.asarray([f"synthetic-{index // 2}" for index in range(n)])
    fold = np.asarray([(index // 2) % 5 for index in range(n)], dtype=np.int64)
    died = rng.integers(0, 2, (n, 6), dtype=np.uint8)
    alive = rng.integers(0, 300, (n, 6), dtype=np.int16)
    alive[~died.astype(bool)] = 300
    return {
        "local": rng.integers(0, 256, (n, 7, 31, 31), dtype=np.uint8),
        "global": rng.integers(0, 256, (n, 4, 37, 37), dtype=np.uint8),
        "scalars": rng.normal(size=(n, 60)).astype(np.float32),
        "world": world,
        "fold": fold,
        "legal": np.ones((n, 6), dtype=bool),
        "died": died,
        "alive_steps": alive,
        "observed": np.ones((n, 6), dtype=bool),
        "decisive": np.zeros(n, dtype=bool),
        "a_star": np.zeros(n, dtype=np.int64),
        "q": np.zeros((n, 6), dtype=np.float32),
    }


def test_cached_c0_refit_matches_a2_units_on_synthetic_data(tmp_path):
    checkpoint = tmp_path / "b8.pth"
    save_ego2s_checkpoint(checkpoint, Ego2sNet(7, 60))
    data = _data()
    actual = _fit_c0(
        data,
        checkpoint,
        device="cpu",
        epochs=1,
        batch_size=2,
        doses=(25,),
        seeds=(20261022,),
    )[25]
    expected = np.empty_like(actual)
    for fold in range(5):
        unit = fit_unit(data, str(checkpoint), "c0", "d300", 25, 20261022, fold, "cpu", 1, 2)
        expected[unit["row_index"]] = unit["p_d300"]
    # Frozen embeddings are cached in a different batch grouping from A2's
    # repeated CNN forwards.  CPU convolution reduction order can therefore
    # differ by a few float32 ulps while the fitted recipe is identical.
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-6)


def test_refuses_noncanonical_folds_before_any_fit(tmp_path):
    checkpoint = tmp_path / "b8.pth"
    save_ego2s_checkpoint(checkpoint, Ego2sNet(7, 60))
    data = _data()
    data["fold"] = np.zeros(len(data["fold"]), dtype=np.int64)
    try:
        _fit_c0(
            data,
            checkpoint,
            device="cpu",
            epochs=1,
            batch_size=2,
            doses=(25,),
            seeds=(20261022,),
        )
    except ValueError as error:
        assert "five frozen A2 folds" in str(error)
    else:
        raise AssertionError("noncanonical folds must refuse")


def test_refuses_a_nonregistered_dose(tmp_path):
    checkpoint = tmp_path / "b8.pth"
    save_ego2s_checkpoint(checkpoint, Ego2sNet(7, 60))
    try:
        _fit_c0(
            _data(),
            checkpoint,
            device="cpu",
            epochs=1,
            batch_size=2,
            doses=(50,),
            seeds=(20261022,),
        )
    except ValueError as error:
        assert "25% and 100%" in str(error)
    else:
        raise AssertionError("50% F2 C0 fit must refuse")
