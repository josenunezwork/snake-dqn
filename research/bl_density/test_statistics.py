"""Synthetic tests for development-only density realism summaries."""

import numpy as np

from research.bl_density.statistics import compare, ks_compare, length_summary


def _rows(prefix, centers, repeats=2):
    return [
        {"episode_id": f"{prefix}-{episode}", "features": [float(center), float(center * 2 + row)]}
        for episode, center in enumerate(centers)
        for row in range(repeats)
    ]


def test_feature_ks_is_exact_and_has_sample_counts():
    result = compare(
        [{"episode_id": "r-0", "features": [0]}, {"episode_id": "r-1", "features": [1]}],
        [{"episode_id": "c-0", "features": [0]}, {"episode_id": "c-1", "features": [2]}],
        ["v8_0"],
    )
    assert result["feature_ks"]["v8_0"] == {"statistic": 0.5, "n_reference": 2, "n_candidate": 2}
    assert result["status"] == "INSUFFICIENT"


def test_shifted_episode_groups_have_high_heldout_auc_and_never_overlap():
    reference = _rows("reference", range(8))
    candidate = _rows("candidate", range(100, 108))
    result = compare(reference, candidate, ["v8_0", "q_gap"])
    classifier = result["classifier"]
    assert result["status"] == "OK"
    assert classifier["auc"] == 1.0
    split = classifier["split"]
    assert not set(split["reference_test_episode_ids"]) & set(split["reference_train_episode_ids"])
    assert not set(split["candidate_test_episode_ids"]) & set(split["candidate_train_episode_ids"])
    assert classifier["n_reference_rows"] == 16
    assert classifier["n_candidate_episodes"] == 8


def test_training_constant_columns_are_dropped_but_all_constant_fails_closed():
    constant = [{"episode_id": f"r-{index}", "features": [1.0, 2.0]} for index in range(4)]
    shifted = [{"episode_id": f"c-{index}", "features": [1.0, 2.0]} for index in range(4)]
    constant_result = compare(constant, shifted, ["v8_0", "q_gap"])
    assert constant_result["status"] == "INSUFFICIENT"
    assert constant_result["classifier"]["reason"] == "all_training_features_constant"

    reference = [{"episode_id": f"r-{index}", "features": [index, 2.0]} for index in range(4)]
    candidate = [{"episode_id": f"c-{index}", "features": [index + 10, 2.0]} for index in range(4)]
    dropped_result = compare(reference, candidate, ["v8_0", "q_gap"])
    assert dropped_result["status"] == "OK"
    assert dropped_result["classifier"]["dropped_constant_feature_names"] == ["q_gap"]
    assert dropped_result["classifier"]["used_feature_names"] == ["v8_0"]


def test_nonfinite_rows_and_shared_episode_id_fail_closed():
    constant = [{"episode_id": f"r-{index}", "features": [1.0, 2.0]} for index in range(4)]
    nonfinite = compare(constant, [{"episode_id": "c", "features": [np.nan, 3.0]}], ["a", "b"])
    assert nonfinite["status"] == "INSUFFICIENT"
    assert nonfinite["reason"] == "nonfinite_feature"
    reference = _rows("shared", range(4))
    candidate = _rows("shared", range(10, 14))
    overlap = compare(reference, candidate, ["v8_0", "q_gap"])
    assert overlap["status"] == "INSUFFICIENT"
    assert overlap["classifier"]["reason"] == "episode_identity_in_both_classes"


def test_insufficient_requires_four_independent_episodes_per_class():
    result = compare(_rows("r", range(3)), _rows("c", range(3, 6)), ["v8_0", "q_gap"])
    assert result["status"] == "INSUFFICIENT"
    assert result["classifier"]["reason"] == "fewer_than_4_episodes_per_class"


def test_length_distribution_has_requested_bins_and_quantiles():
    result = length_summary([799, 800, 999, 1000, 1199, 1200, 1499, 1500])
    assert result["status"] == "OK"
    assert result["histogram"] == {
        "lt_800": 1,
        "800_999": 2,
        "1000_1199": 2,
        "1200_1499": 2,
        "ge_1500": 1,
    }
    assert result["quantiles"]["p50"] == 1099.5
    assert result["quantiles"]["p10"] == 799.7
    assert result["quantiles"]["p99"] == 1499.93


def test_geometry_ks_is_separate_from_the_frozen_classifier_features():
    result = ks_compare(
        [{"hero_length": 800, "wall_distance": 3}, {"hero_length": 900, "wall_distance": 4}],
        [{"hero_length": 1200, "wall_distance": 8}, {"hero_length": 1300, "wall_distance": 9}],
        ["hero_length", "wall_distance"],
    )
    assert result["status"] == "OK"
    assert result["fields"]["hero_length"]["statistic"] == 1.0
