"""Adversarial execution evidence checks for the Mac amendment."""

import copy

import pytest

from research.bl_served_search import analyze, ceiling


def execution_row(mix="frozen", mode="exact_stop"):
    checkpoint_count = {"frozen": 5, "mixed": 3, "scripted": 0}[mix]
    return {
        "plan": {
            "specs": [["checkpoint", "unused"]] * checkpoint_count
            + [["scripted", "greedy_food"]] * (5 - checkpoint_count),
            "member_sha256s": ["a" * 64] * 5,
        },
        "opponent_member_sha256s": ["a" * 64] * 5,
        "opponent_veto_slots": [[] for _ in range(checkpoint_count)],
        "mix": mix,
        "search_mode": mode,
        "native_mode": "stock",
        "worker_threads": {
            "environment": dict.fromkeys(ceiling.THREAD_ENV, "1"),
            "torch": 1,
            "numba": 1,
        },
        "native_v8": {
            "mode": "stock",
            "kernel_rows": 0,
            "calls": 0,
            "stock_rows": 0,
            "fallback_rows": 0,
            "noncandidate_rows": 0,
            "python_calls": 0,
            "root_stock_rows": 0,
        },
        "dense": {
            "batches": 1,
            "branches_evaluated": 1,
            "branches_skipped": 1,
            "batch_diagnostics": [
                {
                    "decision": 4999,
                    "bare": 1,
                    "chosen": 1,
                    "evaluation_order": [1],
                    "branches_evaluated": 1,
                    "branches_skipped": 1,
                    "outcomes": {
                        "1": {"survives": True, "frames_executed": 300},
                        "2": {"status": "NOT_EVALUATED"},
                    },
                }
            ],
        },
    }


def test_observed_execution_passes_even_at_last_scoring_decision():
    assert analyze.execution_error(execution_row()) is None
    assert analyze.execution_error(execution_row("scripted")) is None


@pytest.mark.parametrize(
    "key",
    [
        "calls",
        "kernel_rows",
        "stock_rows",
        "fallback_rows",
        "noncandidate_rows",
        "python_calls",
        "root_stock_rows",
    ],
)
def test_forbidden_native_rows_refused(key):
    row = execution_row()
    row["native_v8"][key] = 1
    assert analyze.execution_error(row)


@pytest.mark.parametrize(
    "key",
    [
        "kernel_rows",
        "stock_rows",
        "fallback_rows",
        "noncandidate_rows",
        "calls",
        "python_calls",
        "root_stock_rows",
    ],
)
def test_missing_counters_refused(key):
    row = execution_row()
    del row["native_v8"][key]
    assert analyze.execution_error(row)


def test_wrong_mode_veto_activity_and_thread_raise_refused():
    row = execution_row()
    for changed in (
        {**row, "search_mode": "all"},
        {**row, "native_mode": "kernel"},
        {**row, "native_v8": {**row["native_v8"], "kernel_rows": 1}},
        {**row, "opponent_veto_slots": [[1]] + [[]] * 4},
        {**row, "worker_threads": {**row["worker_threads"], "torch": 2}},
    ):
        assert analyze.execution_error(changed)


def test_short_survivor_and_forged_skipped_outcome_and_totals_refused():
    for mutation in ("short", "skipped", "totals", "order"):
        row = copy.deepcopy(execution_row())
        batch = row["dense"]["batch_diagnostics"][0]
        if mutation == "short":
            batch["outcomes"]["1"]["frames_executed"] = 299
        elif mutation == "skipped":
            batch["outcomes"]["2"] = {"survives": True, "frames_executed": 300}
        elif mutation == "totals":
            row["dense"]["branches_skipped"] = 0
        else:
            batch["evaluation_order"] = [2]
        assert analyze.execution_error(row)


def test_all_mode_requires_explicit_dev_override_and_all_branches():
    row = execution_row(mode="all")
    assert analyze.execution_error(row)
    assert analyze.execution_error(row, search_mode="all")
    row["dense"]["branches_evaluated"] = 2
    row["dense"]["branches_skipped"] = 0
    batch = row["dense"]["batch_diagnostics"][0]
    batch.update(evaluation_order=[1, 2], branches_evaluated=2, branches_skipped=0)
    batch["outcomes"]["2"] = {"survives": False, "frames_executed": 2}
    assert analyze.execution_error(row, search_mode="all") is None


def test_run_arm_forwards_search_mode_and_full_depth(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from research.t3_rollout_labels_20261015 import bank_engine

    monkeypatch.setattr(ceiling, "opponent_attestation", lambda w: {})
    world = SimpleNamespace(sim=SimpleNamespace(alive=np.array([[True]]), length=np.array([[2]])))
    monkeypatch.setattr(bank_engine, "build_world", lambda *a, **kw: world)
    seen = []

    def step(*args, **kwargs):
        seen.append(kwargs)
        return {"fired": False}

    monkeypatch.setattr(ceiling, "_step", step)
    ceiling.run_arm(
        {}, dense=True, b8_path=Path("unused"), horizon=1, search_mode="exact_stop", full_depth=True
    )
    assert seen == [{"dense": True, "horizon": 1, "search_mode": "exact_stop", "full_depth": True}]


def test_spawned_worker_enters_native_configuration(monkeypatch):
    from research.bl_k1 import native

    def run_world(row, **kwargs):
        active = native._SETTINGS.get()
        assert active.mode == "kernel"
        assert active.stats is kwargs["native_stats"]
        assert kwargs["search_mode"] == "exact_stop"
        return row

    monkeypatch.setattr(ceiling, "run_world", run_world)
    assert ceiling._worker(({"world_id": "dev"}, "unused", "exact_stop", "kernel")) == {
        "world_id": "dev"
    }


@pytest.mark.parametrize("mix", ["frozen", "mixed", "scripted"])
def test_all_zero_vetoes_and_explicit_empty_checkpoint_slots_accepted(mix):
    row = execution_row(mix)
    assert analyze.execution_error(row) is None
    row["native_mode"] = row["native_v8"]["mode"] = "kernel"
    assert analyze.execution_error(row, native_mode="kernel") is None


@pytest.mark.parametrize("mutation", ["absent", "nonempty", "missing", "roster", "plan"])
def test_constructed_opponent_attestation_refusals(mutation):
    row = execution_row()
    if mutation == "absent":
        del row["opponent_veto_slots"]
    elif mutation == "nonempty":
        row["opponent_veto_slots"][0] = [1]
    elif mutation == "missing":
        row["opponent_veto_slots"].pop()
    elif mutation == "roster":
        row["opponent_member_sha256s"][0] = "b" * 64
    else:
        row["plan"]["specs"] = [None] * 5
    assert analyze.execution_error(row)


def test_opponent_attestation_reads_actual_policies_not_enable_flag(tmp_path):
    from types import SimpleNamespace

    from src.evaluation.strict_promotion import scripted_agent
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    path = tmp_path / "model.pth"
    path.write_bytes(b"model bytes")
    checkpoint = object.__new__(Vector61SimdPolicy)
    checkpoint.checkpoint_path, checkpoint.veto_slots = str(path), frozenset()
    anchor = object.__new__(FastProfileAnchorSimdPolicy)
    anchor.kind = "random_safe"
    world = SimpleNamespace(opponents=[checkpoint, anchor, checkpoint])
    observed = ceiling.opponent_attestation(world)
    assert observed == {
        "opponent_member_sha256s": [
            ceiling.digest_file(path),
            scripted_agent("random_safe")["sha256"],
            ceiling.digest_file(path),
        ],
        "opponent_veto_slots": [[], []],
    }
    checkpoint.veto_slots = frozenset([1, 3])
    assert ceiling.opponent_attestation(world)["opponent_veto_slots"] == [[1, 3], [1, 3]]
    path.write_bytes(b"drift")
    assert (
        ceiling.opponent_attestation(world)["opponent_member_sha256s"]
        != observed["opponent_member_sha256s"]
    )
