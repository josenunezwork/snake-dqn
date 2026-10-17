from research.t12_b8_serving_20261017.seed_audit import (
    NAMESPACE,
    PURPOSE_COUNTS,
    generate_banks,
    seed_for,
)


def test_t12_banks_are_fixed_and_disjoint():
    result = generate_banks()

    assert result["namespace"] == NAMESPACE
    assert result["count"] == 73
    assert {name: row["count"] for name, row in result["banks"].items()} == PURPOSE_COUNTS
    assert len({seed for row in result["banks"].values() for seed in row["seeds"]}) == 73


def test_t12_seed_recipe_is_purpose_labelled_and_bounded():
    assert seed_for("watch", 0) == seed_for("watch", 0)
    assert seed_for("watch", 0) != seed_for("play", 0)

    try:
        seed_for("watch", PURPOSE_COUNTS["watch"])
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-range index was accepted")
