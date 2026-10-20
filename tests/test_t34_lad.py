"""Pure contracts for T34 LAD measurement aggregation."""

from research.t34_shadow.bank_measure import summarize, timing_summary
from research.t34_shadow.lad import search_lad_teacher, search_lad_truth


def _expand_with_survivors(survivors: set[int]):
    def expand(decision: int) -> dict:
        return {
            "greedy": 0,
            "q": [1.0, 0.0],
            "outcomes": {
                "0": {"survives": False},
                "1": {"survives": decision in survivors},
            },
        }

    return expand


def test_teacher_stride_rule_discloses_a_negative_to_negative_island() -> None:
    expand = _expand_with_survivors({5})
    teacher = search_lad_teacher(0, 20, expand)
    truth = search_lad_truth(0, 20, expand)
    assert teacher["t_star"] is None
    assert truth["t_star"] == 5


def test_timing_summary_keeps_empty_samples_explicit() -> None:
    assert timing_summary([]) == {
        "count": 0,
        "p50_seconds": None,
        "p95_seconds": None,
        "p99_seconds": None,
    }


def test_summary_separates_t3_survival_from_interventions(monkeypatch) -> None:
    monkeypatch.setattr("research.t34_shadow.bank_measure.world_names", lambda: ["b8_s0_0"])
    monkeypatch.setattr(
        "research.t34_shadow.bank_measure.t1_lad_reachability",
        lambda values: {"samples": len(values)},
    )
    result = summarize(
        [
            {
                "world": "b8_s0_0",
                "historical_action_changes": 2,
                "recovered_action_changes": 1,
                "search_row_steps": 123,
                "operational_row_steps": 456,
                "cases": [{"search_seconds": 0.2}],
            }
        ]
    )
    assert result["recovery_fraction"] == 0.5
    assert result["t3_dense_survival"] == {
        "survivors": 54,
        "worlds": 64,
        "meaning": "survival, not interventions",
    }
    assert result["historical_action_changes"] == 2
