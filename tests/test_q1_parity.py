from research.q0q1_20261017 import q1_parity


def test_q1_parity_rejects_wrong_namespace_before_comparison() -> None:
    value = {"schema": "q1-parity/v1", "horizon": 5000, "worlds": [], "bank_namespace": "wrong"}
    result = q1_parity.compare_artifacts(value, value)
    assert result["pass"] is False
    assert any("namespace" in error for error in result["errors"])
