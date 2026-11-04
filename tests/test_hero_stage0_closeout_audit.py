"""Unit tests for the pre-registered T34 closeout audit (Part A & Part B)."""

from __future__ import annotations

import json

import pytest

from research.hero_stage0.closeout_audit import (
    DEFAULT_B8_HEAD_CLOSEDLOOP_DIR,
    DEFAULT_CLOSEOUT_OUT_DIR,
    PART_A_CANDIDATE_ARMS,
    render_closeout_report,
    score_part_a_b8_head_closedloop,
    summarize_part_b_regime,
)


@pytest.mark.skipif(
    not (DEFAULT_B8_HEAD_CLOSEDLOOP_DIR / "main-01" / "output" / "receipt.json").is_file(),
    reason="b8-head-closedloop receipt artifact not present on this machine",
)
def test_score_part_a_b8_head_closedloop_real_receipt() -> None:
    part_a = score_part_a_b8_head_closedloop(DEFAULT_B8_HEAD_CLOSEDLOOP_DIR)
    assert part_a["receipt_status"] == "PASS"
    assert part_a["stop_verified"] is True
    assert part_a["cleanup_error"] is None
    assert part_a["manifest_sha256_matches_receipt"] is True
    assert part_a["closedloop_readout_commit_check_passes"] is True
    assert part_a["total_rows"] == 960
    assert len(part_a["verified_units"]) == 3
    assert part_a["part_a_verdict"] == "INCONCLUSIVE"
    assert part_a["n_arms_passing_measured_bars"] == 0
    assert part_a["stop_rule_a_zero_of_four_pass_measured"] is True
    for arm in PART_A_CANDIDATE_ARMS:
        ev = part_a["arm_evaluations"][arm]
        assert ev["passes_measured_bars"] is False
        assert ev["pass_pooled_mass_gain_ge_40"] is False
        assert ev["pass_pooled_mass_lb90_gt_0"] is False
        assert ev["pass_scripted_no_death_gain_ge_003"] is False
        assert ev["pass_survival_band_v2"] is False
        assert ev["verdict_on_full_contract_sec2"] == "INCONCLUSIVE"


def test_summarize_part_b_regime_stop_rule_b_logic() -> None:
    ov_records = []
    for i in range(310):
        w = i % 48
        ov_records.append(
            {
                "world_index": w,
                "seed": 1000 + w,
                "frame": 100 + i,
                "decision": 99 + i,
                "life": 1,
                "length": 400 if i < 111 else 800,
                "bare_action": 0,
                "teacher_action": 1,
                "is_certified_rescue_root": True,
                "is_raw_rescue_root": True,
                "bare_failed_by_enclosure_only": False,
                "actions_chosen": {
                    "fail_regime": 1 if i < 60 else 0,
                    "pass_regime": 1 if i < 220 else 0,
                },
                "action_h32_eval": {
                    0: {
                        "action": 0,
                        "is_override": False,
                        "exact_teacher_match": False,
                        "suf0_certified_h32": False,
                        "suf0_raw_survives_h32": False,
                        "suf0_frames_survived_h32": 10,
                        "suf4_7_n_diff_certified": 4,
                        "suf4_7_n_diff_raw": 4,
                        "suf4_7_n_certified": 0,
                        "suf4_7_n_raw": 0,
                        "rescued_suf0_certified": False,
                        "rescued_suf0_and_suf4_7_ndiff0": False,
                        "rescued_suf4_7_ndiff0": False,
                        "rescued_suf0_raw": False,
                    },
                    1: {
                        "action": 1,
                        "is_override": True,
                        "exact_teacher_match": True,
                        "suf0_certified_h32": True,
                        "suf0_raw_survives_h32": True,
                        "suf0_frames_survived_h32": 32,
                        "suf4_7_n_diff_certified": 0,
                        "suf4_7_n_diff_raw": 0,
                        "suf4_7_n_certified": 4,
                        "suf4_7_n_raw": 4,
                        "rescued_suf0_certified": True,
                        "rescued_suf0_and_suf4_7_ndiff0": True,
                        "rescued_suf4_7_ndiff0": True,
                        "rescued_suf0_raw": True,
                    },
                },
            }
        )

    ord_records = []
    for j in range(10080):
        w = j % 48
        ord_records.append(
            {
                "world_index": w,
                "seed": 1000 + w,
                "frame": 200 + j,
                "decision": 199 + j,
                "life": 1,
                "length": 350,
                "bare_action": 0,
                "r1_bare_certified_h12": True,
                "r1_branches_evaluated": 1,
                "r2_same_life": True,
                "r2_t_minus_3_triggered": False,
                "r2_t_minus_3_overrode": False,
                "actions_chosen": {
                    "fail_regime": 0,
                    "pass_regime": 1 if j < 10 else 0,
                },
                "ord_cand_eval": (
                    {
                        1: {
                            "action": 1,
                            "suf0_certified_h32": True,
                            "suf0_raw_survives_h32": True,
                            "suf4_7_n_diff_certified": 0,
                            "suf4_7_n_diff_raw": 0,
                            "lethal_certified": False,
                            "lethal_raw": False,
                        }
                    }
                    if j < 10
                    else {}
                ),
            }
        )

    fail_sum = summarize_part_b_regime(ov_records, ord_records, "fail_regime")
    assert fail_sum["bars"]["passes_stop_rule_b"] is False
    assert fail_sum["rescue_retention_h32_suf0_certified"]["overall"]["n_rescued"] == 60

    pass_sum = summarize_part_b_regime(ov_records, ord_records, "pass_regime")
    assert pass_sum["bars"]["passes_stop_rule_b"] is True
    assert pass_sum["rescue_retention_h32_suf0_certified"]["overall"]["n_rescued"] == 220
    assert pass_sum["ordinary_false_overrides"]["n_false_overrides"] == 10


@pytest.mark.skipif(
    not (DEFAULT_CLOSEOUT_OUT_DIR / "CLOSEOUT_AUDIT_READOUT.json").is_file(),
    reason="Closeout audit readout artifact not present on this machine",
)
def test_closeout_audit_readout_and_report_rendering() -> None:
    readout = json.loads(
        (DEFAULT_CLOSEOUT_OUT_DIR / "CLOSEOUT_AUDIT_READOUT.json").read_text(encoding="utf-8")
    )
    assert readout["part_a_verdict"] == "INCONCLUSIVE"
    assert readout["part_b_verdict"] == "CLOSE T34"
    assert readout["single_verdict"] == "CLOSE T34"
    pb = readout["part_b"]
    assert pb["n_certified_rescue_roots"] == 310
    assert pb["n_ordinary_roots"] == 10080
    assert pb["sealed_gate_namespace_touched"] is False
    assert pb["any_regime_passes_stop_rule_b"] is False

    report_md = render_closeout_report(readout)
    assert "Single Overall Pre-Registered Verdict:** **`CLOSE T34`**" in report_md
    assert "Part A Verdict (`b8-head-closedloop` receipt):** **`INCONCLUSIVE`**" in report_md
