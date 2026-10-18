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
    entry = report["t23_final_inventory"]["7101"]["invalid"][0]
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


def test_inventory_binds_real_embedded_metadata_manifest_and_final_pin(tmp_path, monkeypatch):
    import torch

    from research.q0q1_20261017 import audit_common

    folder = tmp_path / "seed-7101"
    folder.mkdir()
    path = folder / "arbitrary_name.pth"
    meta = {
        "seed": 7101,
        "transitions": 8_000_000,
        "init_sha256": audit_common.B8_SHA256,
        "m3c": {"namespace": "t23-r0-screen/v1"},
    }
    torch.save({"meta": meta, "state_dict": {}}, path)
    digest = audit_common.sha256_file(path)
    (folder / "manifest.jsonl").write_text(
        json.dumps({"file": path.name, "sha256": digest, "transitions": 8_000_000}) + "\n"
    )
    monkeypatch.setattr(
        audit_common, "load_json", lambda _: {"frozen_checkpoint_sha256s": {"7101:8000000": digest}}
    )
    inventory = audit_common.checkpoint_inventory(tmp_path)
    assert inventory["7101"]["final"][0]["valid"]
    assert audit_b8_init.initializer_binding(tmp_path)["7101"]["matches_released_b8"]
    monkeypatch.setattr(
        audit_common,
        "load_json",
        lambda _: {"frozen_checkpoint_sha256s": {"7101:8000000": "0" * 64}},
    )
    assert not audit_common.checkpoint_inventory(tmp_path)["7101"]["final"][0]["valid"]
    meta["seed"] = 7102
    torch.save({"meta": meta}, path)
    assert not audit_common.checkpoint_inventory(tmp_path)["7101"]["final"][0]["valid"]


def test_reservoir_retains_late_long_open_state_and_tie_executes_selectors():
    sampler = audit_b8_init.CoverageReservoir(capacity=20)
    base = {"mix": "scripted", "short_lt700": True, "crowded": True, "action_a": 1, "action_b": 1}
    for _ in range(100):
        sampler.add(dict(base))
    sampler.add(
        {
            **base,
            "short_lt700": False,
            "long_ge800": True,
            "crowded": False,
            "open": True,
            "mix": "mixed",
        }
    )
    for _ in range(100):
        sampler.add(dict(base))
    assert len(sampler.rows) == 20
    assert any(row.get("long_ge800") and row.get("open") for row in sampler.rows)
    tie = audit_b8_init.exact_tie_fixture()
    assert tie["action_a"] == tie["action_b"] == 1  # action zero is masked
    assert tie["exact_q_tie"]


def test_trainer_telemetry_schema_and_identity_mix_groups(tmp_path):
    folder = tmp_path / "seed-7101"
    folder.mkdir()
    row = {
        "transitions": 2_000_000,
        "eps": 0.034,
        "gamma": 0.99,
        "lambda": 1.0,
        "td": 0.25,
        "anchor": 0.75,
        "qmax": 90.0,
        "greedy_boost_rate": 0.001,
    }
    (folder / "log.jsonl").write_text(json.dumps(row) + "\n")
    samples = audit_t23_mass.discover(tmp_path)["telemetry"]["7101"]["samples"]
    assert samples[0]["td"] == 0.25 and samples[0]["lambda"] == 1.0
    common = {
        "training_seed": 7101,
        "mix": "scripted",
        "world_seed": 12,
        "alive_frame_fraction": 1.0,
        "no_death_episode_survival": True,
    }
    rows = [
        {**common, "arm": "b8", "checkpoint_label": "b8", "mass_integral": 10.0},
        {
            **common,
            "arm": "t23",
            "checkpoint_label": "2m",
            "mass_integral": 6.0,
            "q0_metrics": {
                "decision_frames": 10,
                "food_mass_collected": 3,
                "policy_diagnostics": {"probe_decisions": 10, "action_disagreements_b8": 2},
            },
        },
    ]
    summary = audit_t23_mass.summarize_rows(rows)
    assert len(summary["by_identity_seed_mix"]) == 2
    assert summary["paired_mass_decomposition"][0]["mass_integral_delta"] == -4.0
    assert summary["exact_event_metrics"]["action_disagreement_with_b8"] == 0.2
