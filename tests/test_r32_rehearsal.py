"""Independent Codex-V acceptance checks for the R32 synthetic rehearsal."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from research.bl_ppo.exports import digest, export_json
from research.bl_residual import analyze
from research.r32_rehearsal import decisions, job, lifecycle, package, statistics
from research.r32_rehearsal.matrix import run_matrix

ORACLE = json.loads(
    (Path(__file__).resolve().parents[1] / "research/r32_rehearsal/ORACLE.json").read_text()
)


def test_independent_oracle_covers_all_cases_and_r9_labels() -> None:
    assert decisions.case_names() == list(ORACLE["cases"])
    assert set(ORACLE["r9_labels"].values()) == set(analyze.LABELS)
    assert len(ORACLE["r9_labels"]) == 10

    for name in decisions.case_names():
        expected = ORACLE["cases"][name]
        case = decisions.run_case(name)
        actual = case["decision"]
        assert case["pass"] is True, name
        assert actual["label"] == expected["expected_label"], name
        if "expected_flag" in expected:
            assert actual.get("flag") == expected["expected_flag"], name
        if "expected_watch6" in expected:
            assert actual.get("watch6") == expected["expected_watch6"], name
        if "expected_reason" in expected:
            assert expected["expected_reason"] in actual.get("reasons", []), name


def test_lifecycle_matrix_has_three_positive_paths_and_twelve_faults(tmp_path: Path) -> None:
    result = run_matrix(tmp_path / "matrix")
    assert result["counts"] == {"positive": 3, "faults": 12}
    assert result["positive"]["clean"]["status"] == "COMPLETE"
    assert result["positive"]["interrupted_resumed"]["status"] == "COMPLETE"
    assert result["positive"]["interruption_evidence"] == {
        "before_marker_ignored": True,
        "after_marker_preserved": True,
        "mid_verified": [0, 1],
    }
    assert result["positive"]["scientific_stop"] == {
        "status": "COMPLETE",
        "verified": [0, 1],
        "missing": [],
        "not_run_by_rule": [2, 3],
        "replacement_allowed": False,
        "scientific_result": False,
    }
    assert {name: fault["classification"] for name, fault in result["faults"].items()} == ORACLE[
        "lifecycle_faults"
    ]
    assert result["faults"]["timeout_after_successful_export"]["status"] == "STOP_UNVERIFIED"
    assert result["faults"]["timeout_after_successful_export"]["verified"] == [0, 1]


def test_not_run_rule_is_distinct_from_missing_required_watch() -> None:
    not_run = decisions.run_case("not_run_by_rule")["decision"]
    missing = decisions.run_case("missing_required_watch")["decision"]
    assert not_run["label"] == "INCONCLUSIVE"
    assert not_run["watch6"] == "NOT_RUN_BY_RULE"
    assert missing["label"] == "INCOMPLETE"


def test_delivery_truth_table_keeps_worker_and_supervisor_receipts_distinct() -> None:
    rows = {row["artifact"]: row for row in ORACLE["delivery_truth_table"]}
    worker = rows["worker_receipt"]
    supervisor = rows["supervisor_receipt"]
    assert worker["path"].endswith("main-output/exports/receipt.json")
    assert supervisor["path"].endswith("main-output/receipt.json")
    assert worker["authority"] == "portable_worker"
    assert worker["stop_verified"] == "absent_or_false"
    assert worker["admission"] == "diagnostic_only"
    assert supervisor["authority"] == "qualified_supervisor"
    assert supervisor["stop_verified"] == "true"
    assert supervisor["admission"] == "qualified_receipt_only"
    assert {
        "saved_manifest_sha256",
        "source_sha256",
        "spec_sha256",
        "exported_units",
        "positive_accounting",
        "stop_verified",
    }.issubset(supervisor["required_fields"])


def test_portable_resume_skips_exact_verified_units(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    package.prepare_inputs(inputs, tiny=True)
    spec_path = inputs / "spec.json"
    spec = json.loads(spec_path.read_text())
    first_output = tmp_path / "first-output"
    first = job.run_portable(inputs, first_output, mode="tiny")
    assert first["done"] == first["expected"] == 3
    assert first["completed"] == [0, 1, 2]

    lifecycle.build_ledger([first_output], spec, tmp_path / "verified-ledger.json")
    resumed_inputs = tmp_path / "resumed-inputs"
    shutil.copytree(inputs, resumed_inputs)
    source_ledger = tmp_path / "verified-ledger.json"
    target_ledger = resumed_inputs / "verified-ledger.json"
    for suffix in ("", ".sha256", ".complete"):
        shutil.copyfile(str(source_ledger) + suffix, str(target_ledger) + suffix)
    resumed_spec = json.loads((resumed_inputs / "spec.json").read_text())
    resumed_spec["ledger_sha256"] = digest(target_ledger)
    for suffix in ("", ".sha256", ".complete"):
        (resumed_inputs / ("spec.json" + suffix)).unlink()
    export_json(resumed_inputs / "spec.json", resumed_spec)

    original_run_case = decisions.run_case

    def should_not_recompute(name: str) -> dict:
        raise AssertionError(f"verified decision unit recomputed: {name}")

    decisions.run_case = should_not_recompute
    try:
        resumed = job.run_portable(resumed_inputs, tmp_path / "resumed-output", mode="tiny")
    finally:
        decisions.run_case = original_run_case
    assert resumed["done"] == resumed["expected"] == 3
    assert resumed["completed"] == []
    assert resumed["skipped"] == [0, 1, 2]
    assert not list((tmp_path / "resumed-output").glob("unit-*.json.complete"))


def test_frozen_statistics_sanity_is_sensitivity_only() -> None:
    protocol = statistics.load_protocol()
    units = statistics.frozen_units(tiny=True)
    expected_scenarios = (
        protocol["scenarios"]["critical"] + protocol["scenarios"]["sensitivity_only"]
    )
    assert [unit["scenario"] for unit in units] == expected_scenarios
    assert {unit["trials"] for unit in units} == {protocol["compute"]["tiny_preflight_trials"]}
    assert statistics.deterministic_scenario_units(protocol) == [
        {key: value for key, value in unit.items() if key != "trials"} for unit in units
    ]

    sds = {mix: float(index) for index, mix in enumerate(statistics.MIXES, start=1)}
    allocation = {mix: 128 for mix in statistics.MIXES}
    assert statistics.equal_weight_variance(sds, allocation) == (1.0 + 4.0 + 9.0) / (9.0 * 128)
    assert statistics.welch_satterthwaite_df(sds, allocation) > 0.0

    futility = statistics.futility_conditional_probability(
        np.zeros((3, 64)), {mix: 120.0 for mix in statistics.MIXES}, trials=200, seed=3403
    )
    assert futility["analytic_model"] == "gaussian"
    assert futility["trials"] == 200
    assert 0.0 <= futility["conditional_probability"] <= 1.0
    assert statistics.conservative_decision(protocol, {})["verdict"] == "RETAIN_FIXED_DESIGN"


def test_tiny_statistics_preserve_paired_subset_and_equality_cp_invariance() -> None:
    result = statistics.run_frozen_unit("effect_40_homogeneous", trials=200)
    fixed = result["complete_contract_power"]["fixed"]["rate"]
    futility = result["complete_contract_power"]["futility_64_128"]["rate"]
    loss = result["complete_contract_power"]["paired_power_loss_fixed_minus_futility"]["rate"]
    assert 0.0 <= futility <= fixed <= 1.0
    assert np.isclose(loss, fixed - futility)

    observed = np.zeros((3, 64))
    sds = {mix: 120.0 for mix in statistics.MIXES}
    first = statistics.futility_conditional_probability(observed, sds, trials=200, seed=3403)
    second = statistics.futility_conditional_probability(observed, sds, trials=7, seed=9917)
    assert first["conditional_probability"] == second["conditional_probability"]
    assert first["threshold"] == second["threshold"]
    assert first["conditional_mean"] == second["conditional_mean"]
    assert first["conditional_se"] == second["conditional_se"]
