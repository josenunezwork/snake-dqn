"""Focused contracts for controlled observation-probe statistics."""

from __future__ import annotations

import copy

import pytest

from src.evaluation.observation_probe_stats import aggregate_world_metrics, pair_metrics


def test_same_observation_can_prefer_different_actions() -> None:
    result = pair_metrics([1.0, 0.0, None, 0.0, None, 0.0], [0.0, 1.0, None, 0.0, None, 0.0])
    assert result["left_preferred_actions"] == [0]
    assert result["right_preferred_actions"] == [1]
    assert result["disjoint_best_sets"] is True
    assert result["best_common_action"] == 0
    assert result["best_common_action_regret"] == pytest.approx(0.5)
    assert result["number_of_choices"] == 4


def test_overlapping_near_best_sets_are_not_disjoint() -> None:
    result = pair_metrics([1.0, 0.995, None, 0.0, None, 0.0], [0.995, 1.0, None, 0.0, None, 0.0])
    assert result["left_preferred_actions"] == [0, 1]
    assert result["right_preferred_actions"] == [0, 1]
    assert result["disjoint_best_sets"] is False


def test_single_action_has_zero_regret() -> None:
    result = pair_metrics([None, None, 2.5, None, None, None], [None, None, -3.0, None, None, None])
    assert result["best_common_action"] == 2
    assert result["best_common_action_regret"] == 0.0
    assert result["directional_order_disagreements"] == 0


def test_pair_metrics_is_action_permutation_invariant() -> None:
    left = [0.2, 1.0, None, -0.5, None, 0.8]
    right = [1.0, 0.3, None, -0.5, None, 0.8]
    permutation = [5, 3, 2, 1, 4, 0]
    original = pair_metrics(left, right)
    permuted = pair_metrics([left[i] for i in permutation], [right[i] for i in permutation])
    assert permuted["disjoint_best_sets"] == original["disjoint_best_sets"]
    assert permuted["best_common_action_regret"] == pytest.approx(
        original["best_common_action_regret"]
    )
    assert (
        permuted["directional_order_disagreements"] == original["directional_order_disagreements"]
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([0.0] * 5, [0.0] * 5),
        ([0.0, None, 0.0, 0.0, 0.0, 0.0], [0.0] * 6),
        ([None] * 6, [None] * 6),
        ([float("nan")] * 6, [0.0] * 6),
        ([True] + [0.0] * 5, [0.0] * 6),
    ],
)
def test_pair_metrics_rejects_malformed_inputs(
    left: list[float | None], right: list[float | None]
) -> None:
    with pytest.raises(ValueError):
        pair_metrics(left, right)
    with pytest.raises(ValueError):
        pair_metrics([0.0] * 6, [0.0] * 6, tie_margin=-0.01)


def _records(world_count: int, *, duplicate_pairs: int = 1) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric = pair_metrics([1.0, 0.0, None, 0.0, None, 0.0], [0.0, 1.0, None, 0.0, None, 0.0])
    for world in range(world_count):
        for copy_index in range(duplicate_pairs):
            rows.append({"world_id": f"w{world}", "pair_id": f"p{copy_index}", "metrics": metric})
    return rows


def test_aggregate_uses_world_not_pair_as_bootstrap_unit() -> None:
    expected = [f"w{index}" for index in range(8)]
    single = aggregate_world_metrics(_records(8), expected_world_ids=expected, bootstrap_seed=7)
    duplicated = aggregate_world_metrics(
        _records(8, duplicate_pairs=3), expected_world_ids=expected, bootstrap_seed=7
    )
    assert single["eligible_world_count"] == duplicated["eligible_world_count"] == 8
    assert single["best_common_action_regret_l95"] == pytest.approx(
        duplicated["best_common_action_regret_l95"]
    )
    assert duplicated["accepted_pair_count"] == 24
    assert duplicated["decision"] == "ADVANCE_INFORMATION_DIAGNOSTIC"


def test_aggregate_keeps_missing_worlds_uncertain_and_input_immutable() -> None:
    rows = _records(1)
    original = copy.deepcopy(rows)
    result = aggregate_world_metrics(rows, expected_world_ids=["w0", "w1"], bootstrap_seed=2)
    assert rows == original
    assert result["eligible_world_count"] == 1
    assert result["uncertain_world_count"] == 1
    assert result["world_rows"][1]["mean_best_common_action_regret"] is None
    assert result["decision"] == "INCONCLUSIVE_NOT_ADVANCED"


def test_aggregate_rejects_duplicate_unknown_and_malformed_records() -> None:
    metric = pair_metrics([1.0] + [0.0] * 5, [0.0, 1.0] + [0.0] * 4)
    row = {"world_id": "w0", "pair_id": "p0", "metrics": metric}
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_world_metrics([row, row], expected_world_ids=["w0"], bootstrap_seed=1)
    with pytest.raises(ValueError, match="unknown"):
        aggregate_world_metrics(
            [{**row, "world_id": "other"}], expected_world_ids=["w0"], bootstrap_seed=1
        )
    with pytest.raises(ValueError, match="missing"):
        aggregate_world_metrics(
            [{"world_id": "w0", "pair_id": "p0"}], expected_world_ids=["w0"], bootstrap_seed=1
        )
