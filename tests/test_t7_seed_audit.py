from research.redesign_t7_phase_r_20261016.seed_audit import BANK_NS, generate_banks


def test_t7_banks_are_fixed_and_disjoint():
    result = generate_banks()
    assert result["namespace"] == BANK_NS
    assert result["count"] == 960
    assert len(result["banks"]) == 5
    assert all(row["count"] == 192 for row in result["banks"].values())
    assert len({seed for row in result["banks"].values() for seed in row["seeds"]}) == 960
