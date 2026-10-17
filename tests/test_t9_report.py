"""T9 paired denominator, sign and spatial-onset checks."""

from research.t9_divergence_20261017.report import paired_summary, spatial_divergence


def test_pairs_drop_deaths_and_unreachable_distances_without_zero_imputation() -> None:
    rows = [
        {
            "fatal": {"trace": [{"offset": 10, "tail_bfs": 4}]},
            "surviving": {"trace": [{"offset": 10, "tail_bfs": 9}]},
        },
        {
            "fatal": {"trace": [{"offset": 10, "tail_bfs": None}]},
            "surviving": {"trace": [{"offset": 10, "tail_bfs": 9}]},
        },
        {"fatal": {"trace": []}, "surviving": {"trace": [{"offset": 10, "tail_bfs": 9}]}},
    ]
    result = paired_summary(rows, 10, "tail_bfs")
    assert result["alive_pairs"] == 2
    assert result["n"] == 1
    assert result["quartiles"] == [-5, -5, -5]
    assert result["negative"] == 1


def test_spatial_onset_requires_smaller_fatal_pocket_and_uses_square_crop() -> None:
    row = {
        "fatal": {
            "trace": [
                {"offset": 0},
                {"offset": 1, "head_in_largest": True, "head_region": 10},
                {
                    "offset": 2,
                    "head_in_largest": False,
                    "head_region": 5,
                    "boundary_max_chebyshev": 16,
                    "boundary_outside15_fraction": 0.25,
                },
            ]
        },
        "surviving": {
            "trace": [
                {"offset": 0},
                {"offset": 1, "head_region": 20},
                {"offset": 2, "head_region": 20, "head_in_largest": True},
            ]
        },
    }
    result = spatial_divergence(row)
    assert result["offset"] == 2
    assert result["outside15"] is True
    row["fatal"]["trace"] = row["fatal"]["trace"][:2]
    assert spatial_divergence(row) == {"offset": None, "outside15": None}
