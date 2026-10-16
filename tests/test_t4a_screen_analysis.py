import json
from pathlib import Path

import pytest

from research.t4a_screen_20261016.analyze_t4a import (
    ARM,
    EXTENSION_ROUND,
    analyze,
    extension_requirements,
    read_telemetry,
)


def _row(name, mix, deaths, mass=100.0, round_index=65, offset=0):
    count = 96 if mix == "scripted" else 32
    worlds = [str(index) for index in range(offset, offset + count)]
    return {
        "name": name,
        "round_index": round_index,
        "mix": mix,
        "frames": 5000,
        "mass": {world: mass for world in worlds},
        "deaths": {world: death for world, death in zip(worlds, deaths * count)},
        "alive_frames": {
            world: 5000 if death == 0 else 2500 for world, death in zip(worlds, deaths * count)
        },
        "death_lengths": {world: 20.0 for world in worlds},
    }


def _dataset(overrides=None):
    overrides = overrides or {}
    rows = []
    for arm in ("CTRL", ARM):
        for seed in range(10, 15):
            for mix in ("scripted", "frozen", "mixed"):
                deaths, mass = overrides.get((arm, mix), ([1], 100.0))
                rows.append(_row(f"{arm}_s{seed}", mix, deaths, mass))
    for mix in ("scripted", "frozen", "mixed"):
        rows.append(_row("baseline", mix, [0], 75.0))
    return rows, {f"{ARM}_s{seed}": 0 for seed in range(10, 15)}


def test_go_uses_five_seed_means_and_reports_required_secondaries():
    rows, statuses = _dataset({(ARM, "scripted"): ([0], 80.0)})
    result = analyze(rows, statuses)
    data = result["arms"][ARM]
    primary = data["vs_ctrl"]["scripted"]["episode_survival"]
    assert data["decision"] == "GO"
    assert primary["mean"] == 1.0
    assert primary["seed_n"] == 5
    assert primary["seed_se"] == 0.0
    assert data["mass_vs_incumbent"]["mixed"]["seed_n"] == 5
    assert data["death_lengths"]["frozen"]["candidate"][0]["death_count"] == 32


def test_one_extension_is_registered_and_intermediate_final_is_inconclusive():
    rows, statuses = _dataset({(ARM, "scripted"): ([0] + [1] * 19, 100.0)})
    initial = analyze(rows, statuses)
    assert initial["arms"][ARM]["decision"] == "EXTEND"
    jobs = extension_requirements(initial, rows)
    assert len(jobs) == 10
    assert {job["round_index"] for job in jobs} == {EXTENSION_ROUND}
    rows.extend(
        _row(name, "scripted", [0] + [1] * 19, 100.0, EXTENSION_ROUND, 1000)
        for name in [f"{ARM}_s{seed}" for seed in range(10, 15)]
        + [f"CTRL_s{seed}" for seed in range(10, 15)]
    )
    final = analyze(rows, statuses)
    data = final["arms"][ARM]
    assert data["initial_decision"] == "EXTEND"
    assert data["decision"] == "INCONCLUSIVE"
    assert data["extension_exhausted"] is True
    assert extension_requirements(final, rows) == []


def test_partial_or_unregistered_extension_cannot_change_initial_decision():
    rows, statuses = _dataset({(ARM, "scripted"): ([0] + [1] * 19, 100.0)})
    baseline = analyze(rows, statuses)["arms"][ARM]["vs_ctrl"]["scripted"]["episode_survival"]
    rows += [
        _row(f"{ARM}_s10", "scripted", [0], 100.0, EXTENSION_ROUND, 1000),
        _row("CTRL_s10", "scripted", [1], 100.0, EXTENSION_ROUND, 1000),
    ]
    partial = analyze(rows, statuses)["arms"][ARM]
    assert partial["decision"] == "EXTEND"
    assert partial["vs_ctrl"]["scripted"]["episode_survival"] == baseline
    rows, statuses = _dataset({(ARM, "scripted"): ([0], 100.0)})
    rows += [
        _row(f"{ARM}_s{seed}", "scripted", [0], 100.0, EXTENSION_ROUND, 1000)
        for seed in range(10, 15)
    ]
    rows += [
        _row(f"CTRL_s{seed}", "scripted", [1], 100.0, EXTENSION_ROUND, 1000)
        for seed in range(10, 15)
    ]
    with pytest.raises(ValueError, match="initial T4a EXTEND"):
        analyze(rows, statuses)


def test_telemetry_uses_final_post_8m_window_and_side_verdict(tmp_path: Path):
    roots = {}
    for seed in range(10, 15):
        root = tmp_path / f"{ARM}_s{seed}"
        root.mkdir()
        (root / "log.jsonl").write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "transitions": 7_999_999,
                        "bank_agree_decisive": 0.2,
                        "bank_agree_counter": 0.2,
                    },
                    {
                        "transitions": 8_000_128,
                        "bank_agree": 0.95,
                        "bank_agree_decisive": 0.91,
                        "bank_agree_counter": 0.92,
                        "greedy_boost_rate": 0.03,
                    },
                )
            )
            + "\n"
        )
        roots[root.name] = root
    telemetry = read_telemetry(roots)
    rows, statuses = _dataset()
    data = analyze(rows, statuses, telemetry)["arms"][ARM]["telemetry"]
    assert data["bank_agree_decisive_8m_mean"] == pytest.approx(0.91)
    assert data["bank_agree_counter_8m_mean"] == pytest.approx(0.92)
    assert "both >0.9" in data["bank_agree_readout_verdict"]
    assert (
        telemetry[f"{ARM}_s10"]["bank_agree_8m"]["bank_agree_decisive"]["transitions"] == 8_000_128
    )


def test_missing_side_is_not_zero_or_a_verdict_of_input_limit(tmp_path: Path):
    roots = {}
    for seed in range(10, 15):
        root = tmp_path / f"{ARM}_s{seed}"
        root.mkdir()
        (root / "log.jsonl").write_text(
            json.dumps({"transitions": 8_000_128, "bank_agree_decisive": 0.1}) + "\n"
        )
        roots[root.name] = root
    rows, statuses = _dataset()
    telemetry = analyze(rows, statuses, read_telemetry(roots))["arms"][ARM]["telemetry"]
    assert telemetry["bank_agree_counter_8m_n"] == 0
    assert "missing" in telemetry["bank_agree_readout_verdict"]


def test_halted_lane_can_report_kill_without_fabricating_unstarted_statuses():
    result = analyze([], {f"{ARM}_s10": 9})
    data = result["arms"][ARM]
    assert data["decision"] == "KILL"
    assert data["failed_runs"] == [{"name": f"{ARM}_s10", "exit": 9}]
    assert data["unstarted_runs"] == [f"{ARM}_s{seed}" for seed in range(11, 15)]
