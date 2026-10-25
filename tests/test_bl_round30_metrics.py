"""Focused contract tests for the isolated Round-30 metrics reducer."""

from research.bl_round30.analyze import resolved_conversion
from research.bl_round30.metrics import (
    archive_harm_metrics,
    complete_h300,
    conversion_build_trigger,
    decision_label,
    heldout_reducer,
    r30_2_gate_reasons,
    served_selection,
    summarize_training_rows,
    validate_input_rows,
)


def _row(
    parent=1,
    life=1,
    *,
    masses=(2, 9, 5, 0, 0, 0),
    died=(True, False, False, True, True, True),
    status=("EXACT",) * 6,
    mask=(True, True, True, False, False, False),
    q=(1, 2, 2, 0, 0, 0),
    base=0,
    control=2,
    successor=1,
    probability=0.5,
    ordinary=False,
    declared_nonrescue=True,
):
    return {
        "parent_world": parent,
        "life_id": life,
        "horizon": 300,
        "mass": masses,
        "died": died,
        "status": status,
        "legal": (True,) * 6,
        "action_mask": mask,
        "q": q,
        "base_action": base,
        "control_action": control,
        "successor_action": successor,
        "sampling_prob": probability,
        "ordinary": ordinary,
        "declared_nonrescue": declared_nonrescue,
    }


def test_served_selection_uses_normal_fallback_and_first_tie():
    row = _row(mask=(False,) * 6, q=(4, 4, 3, 99, 99, 99))
    assert served_selection(row) == 0


def test_complete_h300_refuses_unknown_legal_selectable_branch():
    status = ("EXACT", "UNKNOWN", "EXACT", "NOT_EVALUATED", "NOT_EVALUATED", "NOT_EVALUATED")
    assert not complete_h300(_row(status=status))
    # Unknown unselectable boost branches do not turn a normal-only row incomplete.
    assert complete_h300(_row(status=("EXACT", "EXACT", "EXACT", "UNKNOWN", "UNKNOWN", "UNKNOWN")))


def test_conversion_trigger_requires_32_parents_and_parent_life_ratio():
    rows = [_row(parent=index, life=1) for index in range(31)]
    assert conversion_build_trigger(rows)["label"] == "INSUFFICIENT"
    result = conversion_build_trigger(rows + [_row(parent=31, life=1)])
    assert result["label"] == "GO"
    assert result["ratio"] == 4 / 7


def test_conversion_trigger_excludes_ordinary_and_b8_surviving_rows():
    rows = [_row(parent=index) for index in range(31)]
    rows.append(_row(parent=31, ordinary=True))
    assert conversion_build_trigger(rows)["label"] == "INSUFFICIENT"
    rows[-1] = _row(parent=31, died=(False, False, False, True, True, True))
    assert conversion_build_trigger(rows)["label"] == "INSUFFICIENT"


def test_conversion_nonretention_is_reported_separately():
    result = conversion_build_trigger([_row(parent=index, control=0) for index in range(32)])
    assert result["control_nonretention"] == 1


def test_conversion_uses_resolved_adam_or_lbfgs_control_and_holds_unresolved():
    rows = [
        {
            **_row(parent=index, control=2),
            "adam_action": 2,
            "lbfgs_action": 1,
        }
        for index in range(32)
    ]
    hashes = {"adam": "a" * 64, "lbfgs": "b" * 64}
    adequate = resolved_conversion(rows, "OPTIMIZER_ADEQUATE", hashes)
    material = resolved_conversion(rows, "OPTIMIZER_MATERIAL", hashes)
    unresolved = resolved_conversion(rows, "INCOMPLETE", hashes)
    assert adequate["control"] == {
        "name": "adam",
        "procedure": "adam",
        "endpoint_sha256": "a" * 64,
    }
    assert adequate["label"] == "GO" and material["label"] == "HOLD"
    assert material["control"]["name"] == "lbfgs"
    assert unresolved == {"label": "HOLD", "reason": "optimizer_unresolved", "control": None}


def test_conversion_below_ten_percent_threshold_closes_ranking_route():
    rows = [_row(parent=index, masses=(2, 9, 8.5, 0, 0, 0)) for index in range(32)]
    result = conversion_build_trigger(rows)
    assert result["ratio"] == 0.5 / 7
    assert result["label"] == "HOLD"
    assert result["disposition"] == "CLOSE_RANKING"
    assert result["successor_allowed"] is False


def test_r30_2_gate_reasons_report_each_unmet_admission_bar():
    reasons = r30_2_gate_reasons("HOLD", {"label": "HOLD"}, "HOLD_R30_2")
    assert reasons["successor_allowed"] is False
    assert reasons["r30_2_go_reasons"] == []
    assert reasons["r30_2_no_go_reasons"] == [
        "optimizer_unresolved",
        "conversion_trigger_not_met",
        "archive_absent",
    ]


def test_heldout_reducer_retains_failed_selection_and_requires_archive():
    rows = [_row(parent=index, life=1) for index in range(32)]
    no_archive = heldout_reducer(rows)
    assert no_archive["label"] == "HOLD"
    assert no_archive["failed_selection"]["control"] == 0
    # A control that selects the poorer survivor has positive regret; successor selects best.
    result = heldout_reducer(rows, natural_rows=[_row(parent=100, base=0, successor=1)])
    assert result["control_regret"] == 4
    assert result["successor_regret"] == 0
    assert result["regret_reduction"] == 1
    assert result["label"] == "GO_OFFLINE"


def test_archive_unknown_override_is_harm_and_precedence_is_fixed():
    unknown = _row(
        base=1, successor=0, status=("UNKNOWN", "EXACT", "EXACT", "EXACT", "EXACT", "EXACT")
    )
    archive = archive_harm_metrics([unknown], "successor_action")
    assert archive["harm_count"] == 1
    assert (
        decision_label(
            invalid=True, incomplete=True, insufficient=True, hold=True, numeric_pass=True
        )
        == "INVALID"
    )
    assert (
        decision_label(incomplete=True, insufficient=True, hold=True, numeric_pass=True)
        == "INCOMPLETE"
    )
    assert decision_label(insufficient=True, hold=True, numeric_pass=True) == "INSUFFICIENT"
    assert decision_label(hold=True, numeric_pass=True) == "HOLD"
    assert decision_label(numeric_pass=False) == "KILL_SUCCESSOR"


def test_heldout_unknown_selection_is_incomplete_not_dropped():
    rows = [_row(parent=index, successor=3) for index in range(32)]
    result = heldout_reducer(rows, natural_rows=[_row(parent=100)])
    assert result["unknown_selection"]["successor"] == 32
    assert result["label"] == "INCOMPLETE"


def test_summary_margin_includes_unknown_selectable_competitor_and_validation_flags_invalid():
    row = _row(
        mask=(True, True, True, True, False, False),
        q=(1, 2, 2, 10, 0, 0),
        status=("EXACT", "EXACT", "EXACT", "UNKNOWN", "EXACT", "EXACT"),
    )
    summary = summarize_training_rows([row])["by_parent_life"][0]["rows"][0]
    assert summary["margin"] == -8
    assert summary["unknown"] == [3]
    duplicate = _row()
    duplicate["row_id"] = "same"
    other = _row(parent=2)
    other["row_id"] = "same"
    other["lineage_valid"] = False
    assert validate_input_rows([duplicate, other], expected_namespace="r30")


def test_heldout_invalid_lineage_precedes_archive_hold_and_archive_needs_declared_coverage():
    rows = [_row(parent=index) for index in range(32)]
    rows[0]["lineage_valid"] = False
    archive = _row(parent=100, declared_nonrescue=False)
    result = heldout_reducer(rows, natural_rows=[archive])
    assert result["label"] == "INVALID"
    assert result["archive"]["coverage"] is False


def test_heldout_dying_selection_keeps_its_finite_survivor_reference_regret():
    rows = [_row(parent=index, control=0) for index in range(32)]
    result = heldout_reducer(rows, natural_rows=[_row(parent=100)])
    assert result["control_regret"] == 7
    assert result["failed_selection"]["control"] == 32


def test_existing_canonical_look_never_deserializes_diagnostics(tmp_path, monkeypatch):
    import pytest

    from research.bl_round30 import analyze

    look, binding, unit = (tmp_path / x for x in ("look.json", "binding.json", "unit.pth"))
    look.write_text("already claimed")
    binding.write_text("metadata")
    unit.write_bytes(b"sealed diagnostic fixture")

    def forbidden(*args, **kwargs):
        raise AssertionError("sealed payload deserialized on a second look")

    monkeypatch.setattr(analyze.torch, "load", forbidden)
    monkeypatch.setattr(analyze.np, "load", forbidden)
    with pytest.raises(FileExistsError):
        analyze._claim_and_load_unit(look, binding, unit, "r30-1")
