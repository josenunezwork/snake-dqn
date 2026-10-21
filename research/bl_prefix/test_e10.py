from __future__ import annotations

import pytest

from research.bl_prefix.analysis import IncompleteResult
from research.bl_prefix.e10 import e10_summary


def _ledger() -> list[dict]:
    rows = []
    for mix in ("scripted", "frozen", "mixed"):
        for index in range(2):
            rows.append(
                {
                    "source_world": f"{mix}-{index}",
                    "mix": mix,
                    "death_frame": 3000,
                    "last_living_length": 1000,
                    "survived_h5000": False,
                    "eligible": True,
                    "selected": True,
                    "sealer": "hero",
                }
            )
    return rows


def _training() -> list[dict]:
    return [
        {
            "source_world": row["source_world"],
            "ancestor": ancestor,
            "training_index": index,
            "paired_delta": 45.0,
            "candidate_survival": True,
            "b8_survival": False,
            "realized_non_greedy": 0.05,
            "both_alive_delta": 0.0,
            "candidate_only_mass": 0.0,
            "b8_only_mass": 0.0,
        }
        for row in _ledger()
        for ancestor in (100, 150, 200)
        for index in range(8)
    ]


def _confirmations() -> list[dict]:
    return [
        {
            "source_world": row["source_world"],
            "ancestor": ancestor,
            "stream": stream,
            "fallback_count": 0,
            "candidate_score": 30.0,
            "b8_score": 0.0,
            "candidate_survival": True,
            "b8_survival": False,
            "b8_death_frame": 3000,
            "b8_last_living_length": 1000.0,
            "candidate_tail_mean_length": 950.0,
        }
        for row in _ledger()
        for ancestor in (100, 150, 200)
        for stream in (0, 1)
    ]


def test_e10_complete_envelope_bridge_and_productive_transfer() -> None:
    result = e10_summary(
        _ledger(),
        {"R": {mix: 10.0 for mix in ("scripted", "frozen", "mixed")}},
        _training(),
        _confirmations(),
    )
    assert (
        result["envelope"]["decision"] == "INCONCLUSIVE"
    )  # UB90 == 45, so strict less-than is false.
    assert result["bridge"]["R"]["equal_mix_mean"] == pytest.approx(10.0)
    assert result["productive_confirmations"]["scripted"]["productive_sources"] == 2
    assert result["coverage"]["scripted"]["eligible_missing"] == 0


def test_e10_never_imputes_missing_eligible_training_outcomes_as_zero() -> None:
    result = e10_summary(
        _ledger(),
        {"R": {mix: 1.0 for mix in ("scripted", "frozen", "mixed")}},
        _training()[:-1],
        _confirmations(),
    )
    assert result["envelope"]["status"] == "INCOMPLETE"
    assert result["envelope"]["decision"] == "INCOMPLETE"


def test_e10_rejects_duplicate_collection_identity() -> None:
    with pytest.raises(IncompleteResult, match="collection ledger"):
        e10_summary(
            _ledger() + [_ledger()[0]],
            {"R": {mix: 1.0 for mix in ("scripted", "frozen", "mixed")}},
            _training(),
            _confirmations(),
        )


def test_two_phase_population_weighting_and_conditional_denominators() -> None:
    ledger = _ledger()
    for mix in ("scripted", "frozen", "mixed"):
        ledger.extend(
            [
                {
                    "source_world": f"{mix}-unsampled",
                    "mix": mix,
                    "death_frame": 3000,
                    "last_living_length": 1000,
                    "survived_h5000": False,
                    "eligible": True,
                    "selected": False,
                    "sealer": "unresolved",
                },
                {
                    "source_world": f"{mix}-survivor",
                    "mix": mix,
                    "death_frame": None,
                    "last_living_length": None,
                    "survived_h5000": True,
                    "eligible": False,
                    "selected": False,
                    "sealer": "none",
                },
            ]
        )
    result = e10_summary(
        ledger,
        {"R": {mix: 10.0 for mix in ("scripted", "frozen", "mixed")}},
        _training(),
        _confirmations(),
        collateral=10.0,
    )
    assert result["envelope"]["statistic"]["mean"] == pytest.approx(0.75 * 45)
    assert result["envelope"]["statistic"]["se"] == pytest.approx(45 * 0.25 / (3**0.5))
    assert result["envelope"]["confirmed_only"]["mean"] == pytest.approx(0.75 * 30)
    assert result["bridge"]["R"]["equal_mix_mean"] == pytest.approx(7.5)
    cell = result["productive_confirmations"]["scripted"]
    assert cell["productive_wilson"]["total"] == 2
    assert cell["natural_productive_fraction"] == pytest.approx(0.75)
    assert cell["gross_supply"] == pytest.approx(0.75 * 950 * 2001 / 5000)
    assert result["coverage"]["scripted"]["eligible_unmeasured_by_design"] == 1
    assert result["coverage"]["scripted"]["eligible_missing"] == 0
    assert (
        result["envelope"]["sensitivity"]["joint_stopping_aware_ub90"]
        >= result["envelope"]["statistic"]["mean"]
    )
    assert result["productive_by_ancestor"]["scripted"]["100"]["wilson"]["total"] == 2


@pytest.mark.parametrize(
    "delta,expected", [(40.0, "INCONCLUSIVE"), (39.999, "FINITE_MENU_INSUFFICIENT")]
)
def test_insufficiency_threshold_is_strict(delta: float, expected: str) -> None:
    training = _training()
    for row in training:
        row["paired_delta"] = delta
    result = e10_summary(_ledger(), {}, training, _confirmations())
    assert result["envelope"]["decision"] == expected


def test_missing_confirmations_do_not_become_measured_zero_supply() -> None:
    result = e10_summary(_ledger(), {}, _training(), [])
    assert result["envelope"]["status"] == "INCOMPLETE"
    assert not result["archive_transfer_admissible"]


@pytest.mark.parametrize("b8_alive,candidate_alive", [(True, True), (False, False)])
def test_nonproductive_valid_confirmations_cannot_admit_archive(
    b8_alive: bool, candidate_alive: bool
) -> None:
    confirmations = _confirmations()
    for row in confirmations:
        row["b8_survival"] = b8_alive
        row["candidate_survival"] = candidate_alive
        row["candidate_tail_mean_length"] = None if b8_alive else 0.0
        row["b8_death_frame"] = None if b8_alive else 3000
        row["b8_last_living_length"] = None if b8_alive else 1000
    result = e10_summary(_ledger(), {}, _training(), confirmations, collateral=0.0)
    assert result["historical_e7"]["decision"] == "TRANSFER_ADMISSIBLE"
    assert result["productive_equal_mix_gross_supply"] == 0
    assert result["tau_required_equal_mix"]["0.0"] is None
    assert not result["archive_transfer_admissible"]


def test_cli_imports_and_reports_incomplete_inputs_with_sidecars(tmp_path) -> None:
    import subprocess
    import sys

    from research.bl_ppo.exports import export_json, verify_export
    from research.bl_prefix.e10 import run

    result = subprocess.run(
        [sys.executable, "-m", "research.bl_prefix.e10", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--bank" in result.stdout
    bank = tmp_path / "bank.json"
    export_json(bank, {"schema": "bl-prefix-bank/v1", "preflight": True, "scientific": False})
    report = run(bank, tmp_path / "missing-e6.json", tmp_path / "missing-e7.json", tmp_path / "out")
    assert report["status"] == "INCOMPLETE"
    assert verify_export(tmp_path / "out/e10-results.json")


def test_cli_rejects_a_different_bank_even_with_valid_sidecars(tmp_path) -> None:
    from research.bl_ppo.exports import export_json
    from research.bl_prefix.e10 import run

    bank = tmp_path / "bank.json"
    e6, e7 = tmp_path / "e6.json", tmp_path / "e7.json"
    export_json(
        bank,
        {
            "schema": "bl-prefix-bank/v1",
            "scientific": True,
            "preflight": False,
            "audit_sha256": "audit",
            "b8": {"sha256": "b8"},
        },
    )
    for path, schema in ((e6, "bl-prefix-e6-results/v1"), (e7, "bl-prefix-e7-results/v1")):
        export_json(
            path,
            {
                "schema": schema,
                "status": "COMPLETE",
                "scientific": True,
                "preflight": False,
                "bank_sha256": "different-bank",
                "audit_sha256": "audit",
                "b8_sha256": "b8",
            },
        )
    report = run(bank, e6, e7, tmp_path / "out")
    assert report["status"] == "INCOMPLETE"
    assert "provenance" in report["reason"]


def test_complete_reporting_cli_reconciles_raw_e6_and_shared_bank(tmp_path) -> None:
    """Exercise the complete file-to-file reporting path without gameplay."""
    from research.bl_ppo.exports import export_json, verify_export
    from research.bl_prefix.e10 import run

    ledger, roots, e6_rows, training, confirmations = [], [], [], [], []
    for mix, quota in (("scripted", 44), ("frozen", 42), ("mixed", 42)):
        for index in range(quota):
            early, ordinary = f"{mix}-death-{index}", f"{mix}-ordinary-{index}"
            ledger.extend(
                [
                    {
                        "source_world": early,
                        "mix": mix,
                        "death_frame": 3000,
                        "last_living_length": 1000,
                        "survived_h5000": False,
                        "eligible": True,
                        "selected": True,
                        "sealer": "hero",
                    },
                    {
                        "source_world": ordinary,
                        "mix": mix,
                        "death_frame": None,
                        "last_living_length": None,
                        "survived_h5000": True,
                        "eligible": False,
                        "selected": False,
                        "sealer": "none",
                    },
                ]
            )
            roots.extend(
                [
                    {"source_world": early, "stratum": "early"},
                    {"source_world": ordinary, "stratum": "ordinary", "matched_death_world": early},
                ]
            )
            for stratum, source in (("early", early), ("ordinary", ordinary)):
                for stream in (0, 1):
                    for phase in ("primary", "secondary"):
                        for arm in ("B8", "R", "F"):
                            e6_rows.append(
                                {
                                    "source_world": source,
                                    "mix": mix,
                                    "stratum": stratum,
                                    "stream": stream,
                                    "phase": phase,
                                    "controller": arm,
                                    "score": 100.0 if arm == "B8" else 110.0,
                                    "ledger": {
                                        "both_alive_delta": 10.0,
                                        "candidate_only_mass": 0.0,
                                        "b8_only_mass": 0.0,
                                        "common_alive_prefix_growth": 3.0,
                                    },
                                }
                            )
            for ancestor in (100, 150, 200):
                training.extend(
                    {
                        "source_world": early,
                        "ancestor": ancestor,
                        "training_index": sample,
                        "paired_delta": 45.0,
                    }
                    for sample in range(8)
                )
                for stream in (0, 1):
                    row = dict(
                        _confirmations()[0], source_world=early, ancestor=ancestor, stream=stream
                    )
                    confirmations.append(row)
    bank = tmp_path / "bank.json"
    bank_info = export_json(
        bank,
        {
            "schema": "bl-prefix-bank/v1",
            "scientific": True,
            "preflight": False,
            "audit_sha256": "audit",
            "b8": {"sha256": "b8"},
            "roots": roots,
            "collection_ledger": ledger,
        },
    )
    provenance = {
        "bank_sha256": bank_info["sha256"],
        "audit_sha256": "audit",
        "b8_sha256": "b8",
        "scientific": True,
        "preflight": False,
        "status": "COMPLETE",
    }
    e6, e7 = tmp_path / "e6.json", tmp_path / "e7.json"
    export_json(e6, dict(provenance, schema="bl-prefix-e6-results/v1", rows=e6_rows))
    export_json(
        e7,
        dict(
            provenance,
            schema="bl-prefix-e7-results/v1",
            training_outcomes=training,
            confirmations=confirmations,
        ),
    )
    result = run(bank, e6, e7, tmp_path / "out")
    assert result["status"] == "COMPLETE"
    assert result["bridge"]["R"]["equal_mix_mean"] == pytest.approx(5.0)
    assert result["envelope"]["statistic"]["mean"] == pytest.approx(22.5)
    assert result["e6_caution_ledger_by_death_stratum"][0]["common_alive_prefix_growth"] == 3.0
    assert verify_export(tmp_path / "out/e10-results.json")
