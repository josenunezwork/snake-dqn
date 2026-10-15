"""Synthetic contracts for the T2b Huber-delta screen."""

from __future__ import annotations

import math

import pytest

from research.t2_screen_20261015.analyze_t2b import analyze, extension_requirements


def _row(name, mix, deaths, mass=100.0, round_index=60, offset=0):
    worlds = [str(value) for value in range(offset, offset + len(deaths))]
    return {
        "name": name,
        "mix": mix,
        "round_index": round_index,
        "frames": 5000,
        "mass": {world: mass for world in worlds},
        "deaths": {world: death for world, death in zip(worlds, deaths)},
        "alive_frames": {
            world: 5000 if death == 0 else 2500 for world, death in zip(worlds, deaths)
        },
    }


def _statuses(**failed):
    values = {f"{arm}_s{seed}": 0 for arm in ("HUBER10", "HUBER30") for seed in range(10, 15)}
    values.update(failed)
    return values


def _rows(overrides=None):
    overrides = overrides or {}
    rows = []
    for arm in ("CTRL", "HUBER10", "HUBER30", "HUBER", "HUBER_ANCHOR0"):
        for seed in range(10, 15):
            for mix in ("scripted", "frozen", "mixed"):
                deaths, mass = overrides.get((arm, mix), ([1, 1], 100.0))
                rows.append(_row(f"{arm}_s{seed}", mix, deaths, mass))
    for mix in ("scripted", "frozen", "mixed"):
        rows.append(_row("baseline", mix, [0, 0], 50.0))
    return rows


def test_go_requires_all_mix_mass_and_reports_dose_response():
    rows = _rows({("HUBER10", "scripted"): ([0, 0], 80.0)})
    result = analyze(rows, _statuses())
    arm = result["arms"]["HUBER10"]
    assert arm["decision"] == "GO"
    assert arm["vs_ctrl"]["scripted"]["episode_survival"]["mean"] == 1.0
    assert arm["vs_ctrl"]["scripted"]["mass"]["seed_se"] == 0.0
    assert set(result["dose_response"]) == {"HUBER10", "HUBER30", "HUBER", "HUBER_ANCHOR0"}
    assert result["dose_response"]["HUBER_ANCHOR0"]["anchor_label"] == "0"


def test_mass_failure_on_any_mix_extends_not_go():
    rows = _rows(
        {
            ("HUBER10", "scripted"): ([0, 0], 80.0),
            ("HUBER10", "frozen"): ([1, 1], 60.0),
        }
    )
    assert analyze(rows, _statuses())["arms"]["HUBER10"]["decision"] == "EXTEND"


def test_failed_run_blocks_go_but_nonpositive_available_primary_kills():
    rows = _rows({("HUBER10", "scripted"): ([0, 0], 80.0)})
    rows = [row for row in rows if not row["name"] == "HUBER10_s12"]
    assert analyze(rows, _statuses(HUBER10_s12=1))["arms"]["HUBER10"]["decision"] == "INCOMPLETE"
    rows = _rows()
    rows = [row for row in rows if not row["name"] == "HUBER10_s12"]
    assert analyze(rows, _statuses(HUBER10_s12=1))["arms"]["HUBER10"]["decision"] == "KILL"


def test_one_extension_reuses_existing_ctrl_and_then_exhausts():
    rows = _rows(
        {
            ("HUBER10", "scripted"): ([0, 1], 100.0),
            ("HUBER10", "frozen"): ([1, 1], 60.0),
        }
    )
    initial = analyze(rows, _statuses())
    assert initial["arms"]["HUBER10"]["decision"] == "EXTEND"
    jobs = extension_requirements(initial, rows)
    assert len(jobs) == 10
    rows.extend(_row(f"CTRL_s{seed}", "scripted", [1, 1], 100.0, 61, 2) for seed in range(10, 15))
    partial = analyze(rows, _statuses())["arms"]["HUBER10"]
    assert partial["extension_exhausted"] is False
    jobs = extension_requirements({"arms": {"HUBER10": partial}}, rows)
    assert {job["name"] for job in jobs} == {f"HUBER10_s{seed}" for seed in range(10, 15)}
    rows.extend(
        _row(f"HUBER10_s{seed}", "scripted", [0, 1], 100.0, 61, 2) for seed in range(10, 15)
    )
    final = analyze(rows, _statuses())["arms"]["HUBER10"]
    assert final["decision"] == "EXTEND"
    assert final["extension_exhausted"] is True
    assert extension_requirements({"arms": {"HUBER10": final}}, rows) == []


def test_partial_extension_cannot_transiently_promote_a_first_seed():
    initial_deaths = [0] * 5 + [1] * 91
    rows = _rows({("HUBER10", "scripted"): (initial_deaths, 100.0)})
    rows = [row for row in rows if row["mix"] != "scripted"]
    rows.extend(_row(f"CTRL_s{seed}", "scripted", [1] * 96) for seed in range(10, 15))
    for arm in ("HUBER30", "HUBER", "HUBER_ANCHOR0"):
        rows.extend(_row(f"{arm}_s{seed}", "scripted", [1] * 96) for seed in range(10, 15))
    rows.extend(_row(f"HUBER10_s{seed}", "scripted", initial_deaths) for seed in range(10, 15))
    rows.append(_row("baseline", "scripted", [1] * 96))
    initial = analyze(rows, _statuses())
    assert initial["arms"]["HUBER10"]["decision"] == "EXTEND"
    rows.extend(
        [
            _row("HUBER10_s10", "scripted", [0] * 96, 100.0, 61, 96),
            _row("CTRL_s10", "scripted", [1] * 96, 100.0, 61, 96),
        ]
    )
    partial = analyze(rows, _statuses())["arms"]["HUBER10"]
    assert partial["decision"] == "EXTEND"
    assert partial["extension_complete"] is False
    jobs = extension_requirements({"arms": {"HUBER10": partial}}, rows)
    assert {job["name"] for job in jobs} == {
        f"{arm}_s{seed}" for arm in ("HUBER10", "CTRL") for seed in range(11, 15)
    }


def test_pairing_and_se_are_seed_level_and_baseline_is_not_extended():
    rows = _rows()
    for seed, survivors in zip(range(10, 15), (2, 1, 0, 0, 0)):
        rows = [
            row
            for row in rows
            if not (row["name"] == f"HUBER10_s{seed}" and row["mix"] == "scripted")
        ]
        rows.append(_row(f"HUBER10_s{seed}", "scripted", [0] * survivors + [1] * (2 - survivors)))
    rows.extend(_row(f"CTRL_s{seed}", "scripted", [1, 1], 100.0, 61, 2) for seed in range(10, 15))
    result = analyze(rows, _statuses())
    metric = result["arms"]["HUBER10"]["vs_ctrl"]["scripted"]["episode_survival"]
    assert metric["mean"] == pytest.approx(0.3)
    assert metric["seed_se"] == pytest.approx(math.sqrt(0.2 / 5))
    assert result["arms"]["HUBER10"]["vs_incumbent"]["scripted"]["mass"]["world_n"] == 10


def test_requires_all_ten_terminal_statuses():
    with pytest.raises(ValueError, match="all 10 runs"):
        analyze(_rows(), {})
