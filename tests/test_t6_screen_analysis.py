"""Focused contracts for the registered T6 RESTART analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.t6_screen_20261017.analyze import (
    ARM,
    EXTENSION_ROUND,
    _decisive_gaps,
    _phase_a_decisive_spools,
    analyze,
    extension_requirements,
    heldout_decisive_q_gaps,
    read_training_telemetry,
)


def test_qgap_uses_stored_b8_fatal_action_and_best_recorded_survivor() -> None:
    import numpy as np

    q = np.array([[9.0, 3.0, 7.0, 8.0, 0.0, 0.0]])
    gaps = _decisive_gaps(
        q,
        np.array([1, 2, 3]),
        np.array([True, False, False]),
        {(7, 11): [0, 1, 2]},
        {b"state": 1},
        [b"state", b"state", b"state"],
    )
    assert gaps.tolist() == [pytest.approx(-5.0)]


@pytest.mark.parametrize(
    "actions,died,b8,match",
    [([1, 2], [True, True], 1, "survivor"), ([1, 2], [False, False], 1, "fatal")],
)
def test_qgap_rejects_missing_registered_fatal_or_survivor(actions, died, b8, match) -> None:
    import numpy as np

    with pytest.raises(ValueError, match=match):
        _decisive_gaps(
            np.zeros((1, 6)),
            np.asarray(actions),
            np.asarray(died),
            {(7, 11): list(range(len(actions)))},
            {b"state": b8},
            [b"state"] * len(actions),
        )


def test_qgap_rejects_wrong_immutable_manifest_before_checkpoint_load(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        heldout_decisive_q_gaps({}, tmp_path)


def test_phase_a_spool_hash_drift_is_rejected_before_spool_use(tmp_path: Path) -> None:
    store = tmp_path / "store"
    phase = store / "staging/phase-a"
    decisive = store / "staging/decisive"
    holdout = store / "banks/holdout"
    phase.mkdir(parents=True)
    decisive.mkdir(parents=True)
    (holdout / "worlds").mkdir(parents=True)
    spool = decisive / "world-1.npz"
    spool.write_bytes(b"changed")
    (phase / "world-1.json").write_text(
        json.dumps(
            {
                "schema": "t4e-phase-a-world/v1",
                "decisive_spool": "decisive/world-1.npz",
                "decisive_spool_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="Phase-A decisive spool hash mismatch"):
        _phase_a_decisive_spools(store, holdout, {})


def _row(
    name: str, mix: str, deaths: list[int], mass: float = 100.0, round_index: int = 72
) -> dict:
    count = 96 if mix == "scripted" else 32
    worlds = [str(index + 1000 * round_index) for index in range(count)]
    values = [deaths[index % len(deaths)] for index in range(count)]
    return {
        "name": name,
        "round_index": round_index,
        "mix": mix,
        "frames": 5000,
        "mass": {world: mass for world in worlds},
        "deaths": dict(zip(worlds, values)),
        "alive_frames": {
            world: 5000 if death == 0 else 2500 for world, death in zip(worlds, values)
        },
        "length_at_5000": {
            world: 1000 if death == 0 else 0 for world, death in zip(worlds, values)
        },
    }


def _dataset(overrides: dict | None = None) -> tuple[list[dict], dict[str, int]]:
    overrides = overrides or {}
    rows = []
    for arm in (ARM, "CTRL"):
        for seed in range(10, 15):
            for mix in ("scripted", "frozen", "mixed"):
                deaths, mass = overrides.get((arm, mix), ([1], 100.0))
                rows.append(_row(f"{arm}_s{seed}", mix, deaths, mass))
    for mix in ("scripted", "frozen", "mixed"):
        rows.append(_row("baseline", mix, [0], 75.0))
    return rows, {f"{ARM}_s{seed}": 0 for seed in range(10, 15)}


def test_go_uses_seed_paired_primary_and_all_mixes() -> None:
    rows, statuses = _dataset({(ARM, "scripted"): ([0], 80.0)})
    data = analyze(rows, statuses)["arms"][ARM]
    assert data["decision"] == "GO"
    assert data["vs_ctrl"]["scripted"]["episode_survival"] == {
        "mean": 1.0,
        "exact_mean_numerator": 1,
        "exact_mean_denominator": 1,
        "seed_se": 0.0,
        "pooled_world_se": 0.0,
        "seeds": [10, 11, 12, 13, 14],
        "seed_n": 5,
        "world_n": 480,
        "complete": True,
    }
    assert data["mass_vs_incumbent"]["mixed"]["seed_n"] == 5


@pytest.mark.parametrize(
    "count_deltas,expected", [([-8, 0, 3, 0, 5], "KILL"), ([-8, 0, 3, 0, 6], "EXTEND")]
)
def test_survival_decision_uses_exact_counts_at_zero(count_deltas, expected) -> None:
    rows, statuses = _dataset()
    for seed, delta in zip(range(10, 15), count_deltas):
        for arm, survivors in (("CTRL", 48), (ARM, 48 + delta)):
            index = next(
                i
                for i, row in enumerate(rows)
                if row["name"] == f"{arm}_s{seed}" and row["mix"] == "scripted"
            )
            rows[index] = _row(
                f"{arm}_s{seed}", "scripted", [0] * survivors + [1] * (96 - survivors)
            )
    result = analyze(rows, statuses)
    data = result["arms"][ARM]
    assert data["decision"] == expected
    primary = data["vs_ctrl"]["scripted"]["episode_survival"]
    assert primary["mean"] == sum(count_deltas) / 480
    if expected == "KILL":
        assert primary["exact_mean_numerator"] == 0
        assert extension_requirements(result, rows) == []


def test_exact_rule_and_single_extension() -> None:
    rows, statuses = _dataset({(ARM, "scripted"): ([0] + [1] * 19, 100.0)})
    provisional = analyze(rows, statuses)
    assert provisional["arms"][ARM]["decision"] == "EXTEND"
    jobs = extension_requirements(provisional, rows)
    assert len(jobs) == 10
    assert {job["round_index"] for job in jobs} == {EXTENSION_ROUND}
    for job in jobs:
        rows.append(_row(job["name"], "scripted", [0] + [1] * 19, round_index=EXTENSION_ROUND))
    final = analyze(rows, statuses)["arms"][ARM]
    assert final["initial_decision"] == "EXTEND"
    assert final["decision"] == "INCONCLUSIVE"
    assert extension_requirements({"arms": {ARM: final}}, rows) == []


def test_nonpositive_and_halted_are_kill_without_fake_probe_data() -> None:
    halted = analyze([], {f"{ARM}_s10": 7})
    assert halted["arms"][ARM]["decision"] == "KILL"
    assert json.dumps(halted, allow_nan=False)
    rows, statuses = _dataset()
    killed = analyze(rows, statuses)
    assert killed["arms"][ARM]["decision"] == "KILL"
    assert (
        killed["arms"][ARM]["shared_survivor_length_h5000"]["by_mix"]["mixed"]["per_seed"][0][
            "mean_delta"
        ]
        is None
    )
    assert json.dumps(killed, allow_nan=False)


def test_shared_survivor_is_per_mix_and_conditions_on_both_alive() -> None:
    rows, statuses = _dataset()
    candidate = next(row for row in rows if row["name"] == f"{ARM}_s10" and row["mix"] == "frozen")
    control = next(row for row in rows if row["name"] == "CTRL_s10" and row["mix"] == "frozen")
    candidate["deaths"]["72000"] = control["deaths"]["72000"] = 0
    candidate["length_at_5000"]["72000"] = 1400
    control["length_at_5000"]["72000"] = 1000
    candidate["deaths"]["72001"] = control["deaths"]["72001"] = 0
    candidate["length_at_5000"]["72001"] = 1300
    control["length_at_5000"]["72001"] = 1000
    frozen = analyze(rows, statuses)["arms"][ARM]["shared_survivor_length_h5000"]["by_mix"][
        "frozen"
    ]
    assert frozen["per_seed"][0]["shared_survivors"] == 2
    assert frozen["per_seed"][0]["mean_delta"] == pytest.approx(350.0)


def test_training_readout_preserves_counts_null_zero_denominator_and_boost_curve(
    tmp_path: Path,
) -> None:
    roots = {}
    for seed in range(10, 15):
        root = tmp_path / f"{ARM}_s{seed}"
        root.mkdir()
        totals = {
            "restart_decisions_ge800": 10,
            "restart_deaths_ge800": 2,
            "fresh_decisions_ge800": 0 if seed == 10 else 20,
            "fresh_deaths_ge800": 0 if seed == 10 else 1,
            "restart_decisions": 15,
            "restart_deaths": 3,
            "fresh_decisions": 30,
            "fresh_deaths": 2,
            "restart_length_mean": 900.0,
            "fresh_length_mean": 800.0,
            "restart_death_length_mean": 950.0,
            "fresh_death_length_mean": 850.0,
        }
        (root / "summary.json").write_text(json.dumps({"restart_totals": totals}))
        (root / "log.jsonl").write_text(
            json.dumps(
                {
                    "transitions": 100000,
                    "greedy_boost_rate": 0.001,
                    "greedy_boost_rate_lt_700": 0.0001,
                }
            )
            + "\n"
        )
        roots[root.name] = root
    output = read_training_telemetry(roots)
    assert output["per_seed"]["10"]["sides"]["fresh"]["deaths_per_1k_ge800"] is None
    assert output["pooled"]["restart"]["decisions_ge800"] == 50
    assert output["pooled"]["restart"]["deaths_ge800"] == 10
    assert output["pooled"]["restart"]["deaths_per_1k_ge800"] == 200.0
    assert output["pooled"]["fresh"]["decisions_ge800"] == 80
    assert len(output["boost_curve"]) == 5
