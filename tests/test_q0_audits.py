"""Focused, non-gameplay regression tests for the Q0 CPU receipt tools."""

from __future__ import annotations

import json
from pathlib import Path

from research.q0q1_20261017 import audit_b8_init, audit_t23_mass


def test_mass_report_is_incomplete_and_does_not_reconstruct_missing_snapshots(
    tmp_path: Path,
) -> None:
    report = audit_t23_mass.build_report(tmp_path / "no-artifacts", None)
    assert report["status"] == "INCOMPLETE"
    assert report["development_packet"]["missing_snapshots_stay_missing"] is True
    assert all(
        not snapshot["available"]
        for seed in report["saved_artifact_availability"].values()
        for snapshot in seed.values()
    )


def test_mass_summary_keeps_unavailable_diagnostics_explicit() -> None:
    output = audit_t23_mass.summarize_rows(
        [
            {
                "record": {
                    "mass_integral": 4.0,
                    "survival_fraction": 0.5,
                    "probes": {"peak_length": 650, "food_eaten": 10, "death_cause": "wall"},
                    "denominators": {"alive_frames": 100},
                }
            }
        ]
    )
    row = output["strata"]["short_lt700"]
    assert row["full_horizon_mass"] == 4.0
    assert row["food_mass_per_alive_decision"] == 0.1
    assert "td_vs_anchor_loss" in row["unavailable"]


def test_b8_inventory_never_calls_file_name_an_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "seed-7101" / "final_8000k.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not a real checkpoint")
    report = audit_b8_init.build_report(tmp_path / "missing-b8.pth", tmp_path)
    assert report["status"] == "INCOMPLETE"
    entry = report["t23_final_inventory"]["7101"]["finals"][0]
    assert entry["path"].endswith("final_8000k.pth")
    assert len(entry["sha256"]) == 64
    assert report["fresh_observation_packet"]["observation_cap"] == 4096


def test_observation_assessment_rejects_mask_or_wrapper_divergence() -> None:
    row = {
        "mix": "scripted",
        "short_lt700": True,
        "long_ge800": True,
        "crowded": True,
        "open": True,
        "exact_q_tie": True,
        "boost_eligible": True,
        "action_a": 0,
        "action_b": 0,
        "fallback_used": False,
        "mask_disagreement": False,
        "wrapper_disagreement": False,
    }
    rows = [dict(row, mix=("scripted", "frozen", "mixed")[index % 3]) for index in range(4096)]
    assert audit_b8_init.assess_observations(rows)["status"] == "COMPLETE"
    rows[2]["wrapper_disagreement"] = True
    try:
        audit_b8_init.assess_observations(rows)
    except ValueError as exc:
        assert "fail-closed" in str(exc)
    else:
        raise AssertionError("wrapper mismatch must fail closed")


def test_cli_writes_parseable_incomplete_receipt(tmp_path: Path) -> None:
    target = tmp_path / "audit.json"
    assert audit_t23_mass.main(["--artifact-root", str(tmp_path), "--out", str(target)]) == 0
    value = json.loads(target.read_text())
    assert value["schema"] == audit_t23_mass.SCHEMA
    assert value["status"] == "INCOMPLETE"
