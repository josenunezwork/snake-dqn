"""Leakage, branch-label and clustered-bootstrap checks for T5 fits."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

from research.t5_separability_20261016.fit import (  # noqa: E402
    bootstrap_auc,
    design,
    weighted_auc,
    world_folds,
)


def test_folds_keep_entire_world_and_are_label_independent():
    world = np.repeat([f"world_{i}" for i in range(21)], np.arange(1, 22))
    fold = world_folds(world)
    assert set(fold) == set(range(5))
    for key in np.unique(world):
        assert len(np.unique(fold[world == key])) == 1
    np.testing.assert_array_equal(world_folds(world[::-1]), fold[::-1])


def data():
    legal = np.ones((3, 6), dtype=bool)
    legal[0, 1] = False
    died = np.array([[1, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1]], bool)
    return {
        "scalars": np.zeros((3, 60)),
        "features": np.zeros((3, 24)),
        "legal": legal,
        "died": died,
        "horizon": np.full((3, 6), 300),
    }


def test_death_uses_actual_legal_branches_and_equal_action_identity():
    d = data()
    for key, width in (("60", 66), ("60+24", 90), ("24", 30)):
        x, y, state, action = design(d, "death_d300", key)
        assert x.shape == (17, width)
        assert not np.any((state == 0) & (action == 1))
        np.testing.assert_array_equal(x[:, -6:], np.eye(6)[action])
        np.testing.assert_array_equal(y, d["died"][state, action])


def test_boost_only_quantifies_over_legal_normals_and_requires_one():
    d = data()
    _, y, state, _ = design(d, "boost_only_escape", "24")
    np.testing.assert_array_equal(y, [1, 0, 0])
    d["legal"][2, :3] = False
    _, y, state, _ = design(d, "boost_only_escape", "24")
    np.testing.assert_array_equal(state, [0, 1])


def test_incomplete_survivor_is_rejected():
    d = data()
    d["horizon"][0, 3] = 299
    with pytest.raises(ValueError, match="complete D300"):
        design(d, "death_d300", "60")


def test_weighted_auc_and_bootstrap_handle_ties_and_cluster_multiplicity():
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
    score = np.array([0.1, 0.5, 0.5, 0.8, 0.9, 0.9])
    weights = np.array([2, 2, 0, 0, 1, 1])
    assert weighted_auc(y, score, weights) == pytest.approx(
        roc_auc_score(y, score, sample_weight=weights)
    )
    world = np.repeat(["a", "b", "c"], 2)
    got = bootstrap_auc(y, {"60": score, "60+24": score, "24": score}, world, 2000)
    assert got["worlds"] == 3
    assert got["metrics"]["60"]["valid_bootstraps"] == 2000
    assert got["metrics"]["delta_60+24_minus_60"]["ci95"] == [0.0, 0.0]
    rng = np.random.default_rng(20261017)
    reference = []
    for _ in range(2000):
        counts = np.bincount(rng.integers(3, size=3), minlength=3)
        reference.append(roc_auc_score(y, score, sample_weight=np.repeat(counts, 2)))
    np.testing.assert_allclose(got["metrics"]["60"]["ci95"], np.quantile(reference, [0.025, 0.975]))
