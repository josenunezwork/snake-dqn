"""An independent weighted-midrank bootstrap must agree with sklearn's AUC."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

from research.t8_probes_20261017.audit import independent_bootstrap  # noqa: E402


def test_independent_bootstrap_matches_sklearn_weighted_ties_and_paired_deltas() -> None:
    y = np.array([0, 1, 0, 1, 0, 1], np.uint8)
    world = np.array(["w0", "w0", "w1", "w1", "w2", "w2"])
    scores = {
        "baseline": np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.4]),
        "arrival": np.array([0.3, 0.5, 0.3, 0.8, 0.4, 0.8]),
    }
    actual = independent_bootstrap(y, scores, world)
    draws = {key: [] for key in scores}
    rng = np.random.default_rng(20261017)
    for _ in range(2000):
        counts = np.bincount(rng.integers(3, size=3), minlength=3)
        weights = np.repeat(counts, 2)
        for key, values in scores.items():
            draws[key].append(roc_auc_score(y, values, sample_weight=weights))
    for key in scores:
        np.testing.assert_allclose(actual[key]["ci95"], np.quantile(draws[key], [0.025, 0.975]))
    delta = np.array(draws["arrival"]) - draws["baseline"]
    np.testing.assert_allclose(actual["delta"]["ci95"], np.quantile(delta, [0.025, 0.975]))
