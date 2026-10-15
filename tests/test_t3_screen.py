import json
from pathlib import Path

import pytest

from research.t3_screen_20261015.analyze_t3 import (
    ARMS,
    EXTENSION_ROUND,
    analyze,
    extension_requirements,
    read_telemetry,
    write_results,
)


def _row(name, mix, deaths, mass=100.0, round_index=62, offset=0):
    worlds = [str(index) for index in range(offset, offset + len(deaths))]
    return {
        "name": name,
        "round_index": round_index,
        "mix": mix,
        "frames": 5000,
        "mass": {world: mass for world in worlds},
        "deaths": {world: death for world, death in zip(worlds, deaths)},
        "alive_frames": {
            world: 5000 if death == 0 else 2500 for world, death in zip(worlds, deaths)
        },
        "death_lengths": {world: 20.0 for world in worlds},
    }


def _dataset(overrides=None):
    overrides = overrides or {}
    rows = []
    for arm in ("CTRL",) + ARMS:
        for seed in range(10, 15):
            for mix in ("scripted", "frozen", "mixed"):
                deaths, mass = overrides.get((arm, mix), ([1, 1], 100.0))
                rows.append(_row(f"{arm}_s{seed}", mix, deaths, mass))
    for mix in ("scripted", "frozen", "mixed"):
        rows.append(_row("baseline", mix, [0, 0], 75.0))
    statuses = {f"{arm}_s{seed}": 0 for arm in ARMS for seed in range(10, 15)}
    return rows, statuses


def test_go_uses_seed_paired_se_and_reports_secondaries(tmp_path: Path):
    rows, statuses = _dataset({("LABELS", "scripted"): ([0, 0], 80.0)})
    telemetry = {
        f"LABELS_s{seed}": {
            "greedy_boost_rate": 0.1,
            "bank_agree_2m": 0.9,
            "bank_agree_8m": 0.95,
        }
        for seed in range(10, 15)
    }
    result = analyze(rows, statuses, telemetry)
    primary = result["arms"]["LABELS"]["vs_ctrl"]["scripted"]["episode_survival"]
    assert result["arms"]["LABELS"]["decision"] == "GO"
    assert primary["mean"] == 1.0
    assert primary["seed_se"] == 0.0
    assert result["arms"]["LABELS"]["vs_ctrl"]["scripted"]["mass"]["mean"] == -20.0
    assert result["arms"]["LABELS"]["telemetry"]["bank_agree_8m_mean"] == 0.95
    write_results(result, tmp_path / "result.json", tmp_path / "result.md")
    assert "NaN" not in (tmp_path / "result.json").read_text()
    assert "bare network" in (tmp_path / "result.md").read_text()


def test_halted_run_is_kill_and_no_candidate_probe_is_required():
    rows, statuses = _dataset()
    statuses["LABELS_s10"] = 9
    rows = [row for row in rows if row["name"] != "LABELS_s10"]
    result = analyze(rows, statuses)
    assert result["arms"]["LABELS"]["decision"] == "KILL"
    assert result["arms"]["LABELS"]["failed_runs"] == [{"name": "LABELS_s10", "exit": 9}]


def test_extend_once_uses_round_63_and_deduplicates_ctrl():
    rows, statuses = _dataset(
        {
            ("CTRL", "scripted"): ([1] * 20, 100.0),
            ("LABELS", "scripted"): ([0] + [1] * 19, 100.0),
            ("LABELS_KEEPV8", "scripted"): ([1] * 20, 100.0),
            ("LABELS_KEEPV8_LO", "scripted"): ([1] * 20, 100.0),
        }
    )
    rows = [row for row in rows if not (row["name"] == "baseline" and row["mix"] == "scripted")]
    rows.append(_row("baseline", "scripted", [0] * 20, 75.0))
    initial = analyze(rows, statuses)
    assert initial["arms"]["LABELS"]["decision"] == "EXTEND"
    jobs = extension_requirements(initial, rows)
    assert len(jobs) == 10
    assert {job["round_index"] for job in jobs} == {EXTENSION_ROUND}
    rows.extend(
        _row(
            name,
            "scripted",
            ([0] + [1] * 19) if name.startswith("LABELS") else ([1] * 20),
            100.0,
            EXTENSION_ROUND,
            20,
        )
        for name in [f"LABELS_s{seed}" for seed in range(10, 15)]
        + [f"CTRL_s{seed}" for seed in range(10, 15)]
    )
    final = analyze(rows, statuses)
    assert final["arms"]["LABELS"]["extension_exhausted"] is True
    assert extension_requirements(final, rows) == []


def test_primary_nonpositive_kills_and_mismatched_pairs_fail():
    rows, statuses = _dataset({("LABELS", "scripted"): ([1, 1], 100.0)})
    assert analyze(rows, statuses)["arms"]["LABELS"]["decision"] == "KILL"
    rows, statuses = _dataset()
    rows[0]["mass"].pop("0")
    with pytest.raises(ValueError, match="missing or inconsistent"):
        analyze(rows, statuses)


def test_read_telemetry_uses_nearest_2m_and_8m_and_keeps_curve(tmp_path: Path):
    run = tmp_path / "LABELS_s10"
    run.mkdir()
    (run / "log.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"transitions": 1_900_000, "bank_agree": 0.8, "greedy_boost_rate": 0.1},
                {"transitions": 2_100_000, "bank_agree": 0.9, "greedy_boost_rate": 0.2},
                {"transitions": 8_000_128, "bank_agree": 0.95, "greedy_boost_rate": 0.3},
            )
        )
        + "\n"
    )
    observed = read_telemetry([tmp_path])["LABELS_s10"]
    assert observed["bank_agree_2m"] == {
        "transitions": 1_900_000,
        "value": 0.8,
        "target": 2_000_000,
        "bound": 100_000,
    }
    assert observed["bank_agree_8m"] == {
        "transitions": 8_000_128,
        "value": 0.95,
        "target": 8_000_000,
        "bound": 100_000,
    }
    assert observed["greedy_boost_rate"] == 0.3
    assert len(observed["greedy_boost_rate_curve"]) == 3


def test_death_lengths_are_descriptive_when_deaths_differ_by_policy():
    rows, statuses = _dataset()
    candidate = next(
        row for row in rows if row["name"] == "LABELS_s10" and row["mix"] == "scripted"
    )
    control = next(row for row in rows if row["name"] == "CTRL_s10" and row["mix"] == "scripted")
    candidate["deaths"] = {"0": 1, "1": 0}
    candidate["alive_frames"] = {"0": 2500, "1": 5000}
    candidate["death_lengths"] = {"0": 30.0}
    control["deaths"] = {"0": 0, "1": 1}
    control["alive_frames"] = {"0": 5000, "1": 2500}
    control["death_lengths"] = {"1": 10.0}
    result = analyze(rows, statuses)
    lengths = result["arms"]["LABELS"]["death_lengths"]["scripted"]
    assert lengths["candidate"][0]["mean"] == 30.0
    assert lengths["ctrl"][0]["mean"] == 10.0


def test_attribution_applies_registered_both_go_and_low_anchor_cases():
    rows, statuses = _dataset({(arm, "scripted"): ([0, 0], 100.0) for arm in ARMS})
    result = analyze(rows, statuses)
    assert "labels carry the effect" in result["attribution"]
    assert "LABELS_KEEPV8_LO also GO" in result["attribution"]


def test_attribution_complete_after_exhausted_extension():
    rows, statuses = _dataset()
    result = analyze(rows, statuses)
    assert "pending" not in result["attribution"]
    assert all(arm in result["attribution"] for arm in ARMS)
