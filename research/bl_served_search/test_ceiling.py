"""Focused unit contracts for L2's frozen dense rule and analysis math."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from research.bl_served_search import analyze, ceiling
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


def test_bool_survival_endpoint_is_accepted_and_delta_checked() -> None:
    expected = {"world_id": "x", "world_seed": 1, "mix": "scripted", "horizon": 5000, "plan": {}}
    row = {
        "schema": ceiling.ROW_SCHEMA,
        **expected,
        "baseline": {
            "mass_integral": 2.0,
            "alive_frame_fraction": 0.2,
            "no_death_episode": False,
            "decisions": 10,
        },
        "dense": {
            "mass_integral": 3.0,
            "alive_frame_fraction": 0.3,
            "no_death_episode": True,
            "decisions": 10,
        },
        "paired_delta": {
            "mass_integral": 1.0,
            "alive_frame_fraction": 0.1,
            "no_death_episode": 1.0,
        },
        "mass_ledger": {
            "common_alive_productivity_delta": 0.1,
            "dense_only_survival_mass": 0.9,
            "baseline_only_survival_mass": 0.0,
            "total": 1.0,
        },
    }
    assert analyze._row_error(row, expected) is None
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
            },
            {
                "mass_integral": 2.0,
                "alive_frame_fraction": 1.0,
                "no_death_episode": True,
                "decisions": 2,
                "_mass_trace": [1, 3],
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
