"""Focused unit contracts for L2's frozen dense rule and analysis math."""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import pytest

from research.bl_served_search import analyze, ceiling, chunks
from research.t3_rollout_labels_20261015.trigger import DenseTrigger


def test_progress_creates_fresh_output_parent(tmp_path) -> None:
    path = tmp_path / "new" / "nested" / "progress.json"
    ceiling._write_progress(path, completed=0, total=3, started=time.perf_counter())
    value = json.loads(path.read_text())
    assert (value["completed"], value["total"], value["eta_seconds"]) == (0, 3, None)
    assert not path.with_suffix(".json.tmp").exists()


def test_t3_score_excludes_fatal_transition_and_prefers_bare() -> None:
    # The mixed/frozen serial adapter counts its fatal transition.  It must not
    # tie a branch surviving the full H300 continuation.
    dead_at_depth = {"survives": False, "frames_executed": 300}
    survivor = {"survives": True, "frames_executed": 300}
    assert ceiling._outcome_score(dead_at_depth) == 299
    assert ceiling._choose({0: dead_at_depth, 1: survivor}, 0, np.zeros(6)) == 1
    both_survive = {0: survivor, 1: survivor}
    assert ceiling._choose(both_survive, 0, np.array([0, 9, 0, 0, 0, 0])) == 0


def test_dense_clock_anchors_before_empty_advisory_skip() -> None:
    trigger = DenseTrigger(min_length=600, interval=5)
    # The first eligible frame has no advisory action.  T3 still anchors here.
    assert trigger.fires(100, 600, True)
    assert not trigger.fires(101, 600, True)
    assert trigger.fires(105, 600, True)


def test_equal_mix_pooled_bound_and_frozen_kill() -> None:
    per_mix = {
        name: analyze.one_sided_90([0.0, 0.0, 0.0, 0.0] if name != "scripted" else [120.0] * 4)
        for name in ceiling.MIX_COUNTS
    }
    pooled = analyze.pooled_equivalent(per_mix)
    assert pooled["estimate"] == 40.0
    assert pooled["lower90"] == pooled["upper90"] == 40.0


def test_transfer_status_is_a_separate_budget_clause() -> None:
    ready = analyze.transfer_status(
        {
            "frozen": {"estimate": 0.1, "upper90": 0.2},
            "mixed": {"estimate": 0.1, "upper90": 0.2},
            "scripted": {"estimate": 0.0, "upper90": 0.0},
        }
    )
    assert ready["label"] == "TRANSFER_READY"
    zero_frozen = analyze.transfer_status(
        {
            "frozen": {"estimate": 0.0, "upper90": 0.2},
            "mixed": {"estimate": 0.1, "upper90": 0.2},
            "scripted": {"estimate": 0.1, "upper90": 0.2},
        }
    )
    assert zero_frozen["label"] == "TRANSFER_UNRESOLVED"
    negative_scripted_upper = analyze.transfer_status(
        {
            "frozen": {"estimate": 0.1, "upper90": 0.2},
            "mixed": {"estimate": 0.1, "upper90": 0.2},
            "scripted": {"estimate": 3.0, "upper90": -0.01},
        }
    )
    assert negative_scripted_upper["label"] == "TRANSFER_UNRESOLVED"


def test_prospective_transfer_does_not_replace_scientific_label() -> None:
    def row(mix: str) -> dict:
        return {
            "mix": mix,
            "paired_delta": {
                "mass_integral": 0.0,
                "alive_frame_fraction": 0.0,
                "no_death_episode": 0.0,
            },
            "baseline": {"mass_integral": 0.0},
            "dense": {
                "mass_integral": 0.0,
                "decisions": 1,
                "action_changes": 0,
                "horizon_surviving_changes": 0,
                "delay_only_changes": 0,
                "residual_changes": 0,
            },
            "mass_ledger": {
                "common_alive_productivity_delta": 0.0,
                "dense_only_survival_mass": 0.0,
                "baseline_only_survival_mass": 0.0,
                "total": 0.0,
            },
            "world_ledger": {
                "productive_rescue": False,
                "d300_only_escape": False,
                "re_death": False,
                "delay_only_changes": 0,
                "newly_caused_death": False,
                "common_alive_growth_loss": 0.0,
                "natural_world_denominator": 1,
                "baseline_died": False,
                "baseline_sealer": "none",
                "interventions": {"proposed": 0, "completed": 0},
            },
        }

    result = analyze.analyze([row(mix) for mix in ceiling.MIX_COUNTS for _ in range(2)])
    assert result["decision"]["label"] == "PACKAGE_KILL"
    assert result["prospective_l4_transfer"]["label"] == "TRANSFER_UNRESOLVED"
    assert result["prospective_l4_transfer"]["authoritative_for_l2"] is False


def test_bool_survival_endpoint_is_accepted_and_delta_checked() -> None:
    expected = {"world_id": "x", "world_seed": 1, "mix": "scripted", "horizon": 5000, "plan": {}}
    row = {
        "schema": ceiling.ROW_SCHEMA,
        **expected,
        "baseline": {
            "mass_integral": 2.0,
            "alive_frame_fraction": 0.5,
            "no_death_episode": False,
            "decisions": 2501,
        },
        "dense": {
            "mass_integral": 3.0,
            "alive_frame_fraction": 1.0,
            "no_death_episode": True,
            "decisions": 10,
            "delay_only_changes": 0,
            "proposed_interventions": 1,
            "completed_interventions": 1,
        },
        "paired_delta": {
            "mass_integral": 1.0,
            "alive_frame_fraction": 0.5,
            "no_death_episode": 1.0,
        },
        "mass_ledger": {
            "common_alive_productivity_delta": -0.5,
            "dense_only_survival_mass": 1.5,
            "baseline_only_survival_mass": 0.0,
            "total": 1.0,
        },
        "trajectories": {
            "baseline_mass": [4] * 2500 + [0] * 2500,
            "dense_mass": [3] * 5000,
        },
        "world_ledger": {
            "productive_rescue": False,
            "d300_only_escape": False,
            "re_death": False,
            "delay_only_changes": 0,
            "newly_caused_death": False,
            "common_alive_growth_loss": 0.5,
            "natural_world_denominator": 1,
            "baseline_died": True,
            "b8_death_frame": 2500,
            "b8_death_length": 4,
            "b8_death_cause": 1,
            "baseline_sealer": "unresolved",
            "dense_sealer": "none",
            "interventions": {
                "proposed": 1,
                "completed": 1,
                "consumed_before_deadline": None,
                "consumed_before_deadline_availability": "UNAVAILABLE_UNPACED_L2",
            },
        },
    }
    assert analyze._row_error(row, expected) is None
    redistributed = copy.deepcopy(row)
    redistributed["mass_ledger"]["common_alive_productivity_delta"] = 0.5
    redistributed["mass_ledger"]["dense_only_survival_mass"] = 0.5
    assert "components" in str(analyze._row_error(redistributed, expected))
    false_rescue = copy.deepcopy(row)
    false_rescue["world_ledger"]["productive_rescue"] = True
    assert "productive rescue" in str(analyze._row_error(false_rescue, expected))
    false_alive_fraction = copy.deepcopy(row)
    false_alive_fraction["baseline"]["alive_frame_fraction"] = 1.0
    false_alive_fraction["paired_delta"]["alive_frame_fraction"] = 0.0
    assert "alive fraction" in str(analyze._row_error(false_alive_fraction, expected))
    false_no_death = copy.deepcopy(row)
    false_no_death["baseline"]["no_death_episode"] = True
    false_no_death["paired_delta"]["no_death_episode"] = 0.0
    assert "no-death endpoint" in str(analyze._row_error(false_no_death, expected))
    shifted_death_length = copy.deepcopy(row)
    shifted_death_length["world_ledger"]["b8_death_length"] = 5
    assert "death length" in str(analyze._row_error(shifted_death_length, expected))
    shifted = copy.deepcopy(row)
    shifted["trajectories"]["baseline_mass"] = [
        value + 1 for value in shifted["trajectories"]["baseline_mass"]
    ]
    shifted["trajectories"]["dense_mass"] = [
        value + 1 for value in shifted["trajectories"]["dense_mass"]
    ]
    assert "baseline trajectory" in str(analyze._row_error(shifted, expected))
    row["paired_delta"]["no_death_episode"] = 0.0
    assert "paired delta mismatch" in str(analyze._row_error(row, expected))


def test_run_world_mass_ledger_reconciles(monkeypatch) -> None:
    results = iter(
        (
            {
                "mass_integral": 1.0,
                "alive_frame_fraction": 0.5,
                "no_death_episode": False,
                "decisions": 2,
                "_mass_trace": [1, 1],
                "_alive_trace": [True, True],
                "death_frame": None,
                "death_length": None,
                "death_cause": None,
                "d300_only_escapes": 0,
                "re_deaths": 0,
                "delay_only_changes": 0,
                "proposed_interventions": 0,
                "completed_interventions": 0,
                "consumed_before_deadline_interventions": None,
                "consumed_before_deadline_availability": "UNAVAILABLE_UNPACED_L2",
            },
            {
                "mass_integral": 2.0,
                "alive_frame_fraction": 1.0,
                "no_death_episode": True,
                "decisions": 2,
                "_mass_trace": [1, 3],
                "_alive_trace": [True, True],
                "death_frame": None,
                "death_length": None,
                "death_cause": None,
                "d300_only_escapes": 0,
                "re_deaths": 0,
                "delay_only_changes": 0,
                "proposed_interventions": 0,
                "completed_interventions": 0,
                "consumed_before_deadline_interventions": None,
                "consumed_before_deadline_availability": "UNAVAILABLE_UNPACED_L2",
            },
        )
    )
    monkeypatch.setattr(ceiling, "run_arm", lambda *args, **kwargs: next(results))
    row = ceiling.run_world(
        {
            "world_id": "fixture",
            "world_seed": 1,
            "mix": "scripted",
            "horizon": 2,
            "plan": {"seed": 1, "mix": "scripted"},
        },
        b8_path=__import__("pathlib").Path("ignored"),
    )
    assert row["mass_ledger"] == {
        "common_alive_productivity_delta": 1.0,
        "dense_only_survival_mass": 0.0,
        "baseline_only_survival_mass": 0.0,
        "total": 1.0,
    }


def test_chunks_are_seed_ordered_interleaved_and_balanced() -> None:
    rows = []
    for mix, count in ceiling.MIX_COUNTS.items():
        for index in range(count):
            rows.append({"world_id": f"{mix}/{index}", "world_seed": count - index, "mix": mix})
    chunk = chunks.chunk_rows(rows, 0)
    assert len(chunk) == 40
    assert [row["mix"] for row in chunk[:6]] == ["frozen", "mixed", "scripted"] * 2
    assert {mix: sum(row["mix"] == mix for row in chunk) for mix in ceiling.MIX_COUNTS} == {
        "frozen": 16,
        "mixed": 16,
        "scripted": 8,
    }
    assert [row["world_seed"] for row in chunk if row["mix"] == "frozen"] == list(range(1, 17))


def test_runtime_parity_rejects_copied_bank_or_missing_runtime_identity() -> None:
    source = {"source.py": "0" * 64}
    parity = {
        "status": "VERIFIED",
        "bank_sha256": "1" * 64,
        "b8_sha256": ceiling.B8_SHA256,
        "source_manifest": source,
        "mac_reference_sha256": "2" * 64,
        "runtime_verification_sha256": "3" * 64,
        "runtime_identity": {"node": "n", "machine": "m", "colab_cli_root": "/r"},
    }
    assert (
        ceiling.validate_runtime_parity(
            parity, bank_sha256="1" * 64, source=source, require_local_runtime=False
        )
        == parity
    )
    with pytest.raises(ValueError, match="different bank"):
        ceiling.validate_runtime_parity(
            parity, bank_sha256="4" * 64, source=source, require_local_runtime=False
        )
    parity["runtime_identity"] = {}
    with pytest.raises(ValueError, match="runtime identity"):
        ceiling.validate_runtime_parity(
            parity, bank_sha256="1" * 64, source=source, require_local_runtime=False
        )
