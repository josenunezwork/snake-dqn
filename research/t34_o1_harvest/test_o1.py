"""Targeted mechanical tests; all rows are synthetic, no study worlds played."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from research.t34_o1_harvest.analyze import decision, qualifying_root, validate_ledger
from research.t34_o1_harvest.bank import materialize
from research.t34_o1_harvest.runner import fixture_forecast, rank_branches, reservoir_priority


def ledger(intake: int = 0, source: str = "death_drop", steps: int = 500) -> dict[str, Any]:
    totals = dict.fromkeys(("ambient", "death_drop", "boost_pellet"), 0)
    totals[source] = intake
    events = []
    for tick in range(steps):
        gain = {s: (v if tick == 0 else 0) for s, v in totals.items()}
        events.append(
            {
                "env": 0,
                "frame": tick,
                "intake": gain,
                "cumulative_intake": totals.copy(),
                "boost_cost": 0,
                "cumulative_boost_cost": 0,
            }
        )
    env = {
        "env": 0,
        "initial_stored_length": 10,
        "sources": totals.copy(),
        "cumulative_intake": totals.copy(),
        "weighted_cumulative_intake": {s: v * steps for s, v in totals.items()},
        "boost_cost": 0,
        "weighted_cumulative_boost_cost": 0,
        "final_stored_length": 10 + intake,
    }
    return {"per_env": [env], "events": events}


def root(gain: int = 5, source: str = "death_drop") -> dict[str, Any]:
    branches = {}
    for action, delta in ((1, 0), (2, gain)):
        branches[str(action)] = {
            "action": action,
            "mass_sum": (10 + delta) * 500,
            "alive_steps": 500,
            "row_steps": 500,
            "survives": True,
            "ledger": ledger(delta, source),
            "delta_mass_sum": delta * 500,
            "delta_A500": float(delta),
        }
    return {
        "q": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "horizon": 500,
        "source_preserved": True,
        "bare_action": 1,
        "advisory_legal": [False, True, True, False, False, False],
        "illegal_actions": [0, 3, 4, 5],
        "branches": branches,
        "ranked_actions": [2, 1] if gain > 0 else [1, 2],
        "decision": 1000,
        "time_root": True,
        "event_root": False,
    }


def worlds(positive: int = 8, fm: int = 4) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = materialize()["rows"]
    values = []
    for index, row in enumerate(rows):
        success = index < positive - fm or 16 <= index < 16 + fm
        values.append(
            {
                "row": row,
                "complete": True,
                "accounting": True,
                "cfg": {"mechanics_version": 2},
                "frames": 5000,
                "horizon": 500,
                "checkpoint_sha256": (
                    "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
                ),
                "base_survives": False,
                "base_death_cause": 1,
                "base_ledger": ledger(steps=1001),
                "base_steps": 1001,
                "events": [],
                "roots": [root(5 if success else 0)],
            }
        )
    return values, rows


def test_exact_kill_go_boundaries() -> None:
    for n, fm, label in ((8, 4, "GO"), (7, 4, "KILL"), (8, 3, "KILL"), (0, 0, "KILL")):
        values, rows = worlds(n, fm)
        result = decision(values, rows)
        assert result["status"] == label, result


@pytest.mark.parametrize("source", ["ambient", "boost_pellet"])
def test_gross_non_death_gain_never_qualifies(source: str) -> None:
    assert qualifying_root(root(source=source))[0] is False


def test_death_threshold_and_bare_tie() -> None:
    assert qualifying_root(root())[0] is True
    assert qualifying_root(root(4))[0] is False
    assert rank_branches({0: {"mass_sum": 1}, 1: {"mass_sum": 1}}, 1) == [1, 0]


def test_bad_ledger_missing_branch_and_duplicate_world_are_invalid() -> None:
    values, rows = worlds()
    bad = copy.deepcopy(values)
    bad[0]["roots"][0]["branches"]["2"]["ledger"]["per_env"][0]["final_stored_length"] += 1
    assert decision(bad, rows)["status"] == "INVALID"
    bad = copy.deepcopy(values)
    del bad[0]["roots"][0]["branches"]["2"]
    assert decision(bad, rows)["status"] == "INVALID"
    assert decision(values[:-1], rows)["status"] == "INCOMPLETE"
    bad = list(values)
    bad[-1] = values[0]
    assert decision(bad, rows)["status"] == "INVALID"


def test_cumulative_sources_recomputed() -> None:
    value = ledger(5)
    value["events"][20]["cumulative_intake"]["death_drop"] = 4
    with pytest.raises(ValueError, match="cumulative"):
        validate_ledger(value, 500)


def test_survival_is_required_for_both_branches() -> None:
    value = root()
    b = value["branches"]["1"]
    b.update(alive_steps=499, row_steps=500, survives=False)
    assert qualifying_root(value)[0] is False


def test_all_banks_and_fixture_namespaces_disjoint() -> None:
    bank = materialize()
    rows = sum(
        (
            bank[k]
            for k in (
                "rows",
                "excluded_timing_rows",
                "excluded_parity_rows",
                "excluded_preflight_rows",
            )
        ),
        [],
    )
    assert len({r["world_seed"] for r in rows}) == 55
    assert len({r["world_id"] for r in rows}) == 55
    assert bank["mechanics_version"] == 2
    assert {
        m: sum(r["mix"] == m for r in bank["rows"]) for m in ("scripted", "frozen", "mixed")
    } == dict.fromkeys(("scripted", "frozen", "mixed"), 16)


def test_event_reservoir_is_deterministic_and_outcome_independent() -> None:
    forward = sorted(range(100), key=lambda d: reservoir_priority(7, d))[:4]
    reverse = sorted(reversed(range(100)), key=lambda d: reservoir_priority(7, d))[:4]
    assert forward == reverse
    assert forward != sorted(range(100), key=lambda d: reservoir_priority(8, d))[:4]


def test_forecast_uses_worst_rates_and_rounds_worker_waves_up() -> None:
    w = {
        "base_steps": 40,
        "base_seconds": 1.0,
        "elapsed_seconds": 2.0,
        "roots": [{"branches": {"1": {"elapsed_seconds": 1.0, "row_steps": 8}}}],
    }
    forecast = fixture_forecast([w], 11)
    assert forecast["remaining_work_seconds"] == 2 * (5 * (5000 / 40 + 24000 / 8) + 10000 / 40)
    assert forecast["l4_forecast_seconds_including_setup_export"] > 1200


def test_judgment_kill_rule_is_verbatim() -> None:
    repo = Path(__file__).resolve().parents[2]
    judgment = (repo / "docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md").read_text()
    text = judgment.split("- **Kill rule (frozen before data).** ", 1)[1].split(
        "\n- **What it must beat", 1
    )[0]
    # Markdown indentation in the source bullet is not semantic text.
    expected = " ".join(text.split())
    prereg = (Path(__file__).parent / "PREREGISTRATION_O1.md").read_text()
    assert expected in " ".join(prereg.split())


def _synthetic_collect(row: dict[str, Any], checkpoint: str, **_: Any) -> dict[str, Any]:
    """Pure seeded leaf fixture; exercises spawn transport without gameplay."""
    import random

    rng = random.Random(row["world_seed"])
    return {"row": row, "checkpoint": checkpoint, "draws": [rng.randrange(10000) for _ in range(8)]}


def _spawn_fixture_initializer() -> None:
    from research.t34_o1_harvest import runner

    runner.collect = _synthetic_collect


def test_real_spawn_merge_preserves_serial_scientific_bytes(monkeypatch: Any) -> None:
    import functools
    import json
    from concurrent.futures import ProcessPoolExecutor

    from research.t34_o1_harvest import runner

    rows = [{"world_seed": seed, "world_id": str(i)} for i, seed in enumerate((9, 3, 7))]
    monkeypatch.setattr(runner, "collect", _synthetic_collect)
    serial = runner.collect_many(rows, "fixture-only", 1)
    monkeypatch.setattr(
        runner,
        "ProcessPoolExecutor",
        functools.partial(ProcessPoolExecutor, initializer=_spawn_fixture_initializer),
    )
    worker_serial = runner.collect_many(rows, "fixture-only", 1)
    parallel = runner.collect_many(rows, "fixture-only", 2)
    assert json.dumps(serial, sort_keys=True) == json.dumps(worker_serial, sort_keys=True)
    assert json.dumps(serial, sort_keys=True) == json.dumps(parallel, sort_keys=True)


def test_slow_bootstrap_cannot_spend_reserve(tmp_path: Path) -> None:
    import json

    from research.t34_o1_harvest.runner import ordinary_remaining

    (tmp_path / "state.json").write_text(json.dumps({"started_at": 1000.0}))
    (tmp_path / "spec.json").write_text(json.dumps({"timeout_seconds": 14000.0}))
    assert ordinary_remaining(tmp_path, now=4000.0) == 14000 - 3000 - 2880 - 60
    assert ordinary_remaining(tmp_path, now=20000.0) < 0
