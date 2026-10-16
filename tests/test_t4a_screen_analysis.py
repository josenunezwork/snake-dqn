import json
from pathlib import Path

import pytest

from research.t4a_screen_20261016.analyze_t4a import (
    ARM,
    EXTENSION_ROUND,
    _decision,
    _telemetry_for,
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
                        "qmax": 17.0,
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
    run = telemetry[f"{ARM}_s10"]
    assert run["source_log"]["path"].endswith("log.jsonl")
    assert len(run["source_log"]["sha256"]) == 64
    assert run["telemetry_records"][-1]["qmax"] == 17.0
    assert run["final_telemetry"]["qmax"] == 17.0


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


def test_primary_uses_nonzero_seed_paired_se_not_pooled_world_se():
    rows, statuses = _dataset()
    per_seed_survivors = [0, 12, 24, 36, 48]
    for seed, survivors in zip(range(10, 15), per_seed_survivors):
        row = next(
            item for item in rows if item["name"] == f"{ARM}_s{seed}" and item["mix"] == "scripted"
        )
        deaths = {str(index): 0 if index < survivors else 1 for index in range(96)}
        row["deaths"] = deaths
        row["alive_frames"] = {
            world: 5000 if death == 0 else 2500 for world, death in deaths.items()
        }
    primary = analyze(rows, statuses)["arms"][ARM]["vs_ctrl"]["scripted"]["episode_survival"]
    values = [survivors / 96 for survivors in per_seed_survivors]
    expected = (sum((value - sum(values) / 5) ** 2 for value in values) / 4) ** 0.5 / 5**0.5
    assert primary["seed_se"] == pytest.approx(expected)
    assert primary["pooled_world_se"] != pytest.approx(primary["seed_se"])


def _complete_decision_data(primary=0.06, mass=-30.0, frozen=-0.05, mixed=-0.05):
    def metric(mean):
        return {"mean": mean, "complete": True}

    return {
        "vs_ctrl": {
            "scripted": {
                "episode_survival": metric(primary),
                "alive_survival": metric(0.0),
                "mass": metric(mass),
            },
            "frozen": {
                "episode_survival": metric(frozen),
                "alive_survival": metric(0.0),
                "mass": metric(mass),
            },
            "mixed": {
                "episode_survival": metric(mixed),
                "alive_survival": metric(0.0),
                "mass": metric(mass),
            },
        }
    }


def test_decision_exact_go_guards_and_nonpositive_kill():
    statuses = {f"{ARM}_s{seed}": 0 for seed in range(10, 15)}
    assert _decision(_complete_decision_data(), statuses, False) == "GO"
    assert _decision(_complete_decision_data(primary=0.059999), statuses, False) == "EXTEND"
    assert _decision(_complete_decision_data(mass=-30.000001), statuses, False) == "EXTEND"
    assert _decision(_complete_decision_data(frozen=-0.050001), statuses, False) == "EXTEND"
    assert _decision(_complete_decision_data(mixed=-0.050001), statuses, False) == "EXTEND"
    assert _decision(_complete_decision_data(primary=0.0), statuses, False) == "KILL"
    assert _decision(_complete_decision_data(primary=-0.000001), statuses, False) == "KILL"


def test_bank_agreement_diagnostic_thresholds_are_strict():
    def telemetry(decisive, counter):
        return {
            f"{ARM}_s{seed}": {
                "bank_agree_2m": {},
                "bank_agree_8m": {
                    "bank_agree_decisive": {"value": decisive},
                    "bank_agree_counter": {"value": counter},
                },
            }
            for seed in range(10, 15)
        }

    assert "inconclusive" in _telemetry_for(telemetry(0.9, 0.91))["bank_agree_readout_verdict"]
    assert (
        "both >0.9" in _telemetry_for(telemetry(0.900001, 0.900001))["bank_agree_readout_verdict"]
    )
    assert "inconclusive" in _telemetry_for(telemetry(0.8, 0.91))["bank_agree_readout_verdict"]
    assert "input limit" in _telemetry_for(telemetry(0.799999, 0.91))["bank_agree_readout_verdict"]


def test_retained_telemetry_has_raw_records_nominal_bins_and_report_tables(tmp_path: Path):
    from research.t4a_screen_20261016.analyze_t4a import markdown

    roots = {}
    for seed in range(10, 15):
        root = tmp_path / f"{ARM}_s{seed}"
        root.mkdir()
        (root / "log.jsonl").write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "transitions": 100_128,
                        "bank_agree_decisive": 0.2 + seed / 1000,
                        "bank_agree_counter": 0.8,
                    },
                    {
                        "transitions": 8_000_128,
                        "bank_agree_decisive": 0.91,
                        "bank_agree_counter": 0.92,
                        "td": 0.12,
                        "anchor": 0.03,
                        "qmax": 12.0,
                        "bank_margin_loss": 0.04,
                        "bank_rows_mean": 128.0,
                        "tight_frac": None,
                    },
                )
            )
            + "\n"
        )
        roots[root.name] = root
    telemetry = read_telemetry(roots)
    rows, statuses = _dataset()
    rerolls = {
        f"{ARM}_s{seed}": {
            "target_action_survival_fraction": 0.7,
            "raw_argmax_matches_t4a_target_by_side": {"decisive": 0.6, "counter": 0.8},
            "advisory_argmax_matches_t4a_target_by_side": {"decisive": 0.65, "counter": 0.85},
            "states_by_side": {"decisive": 120, "counter": 80},
        }
        for seed in range(10, 15)
    }
    result = analyze(rows, statuses, telemetry, rerolls)
    aggregate = result["arms"][ARM]["telemetry"]["bank_agreement_per_100k_means"]
    assert aggregate["bank_agree_decisive"][0] == {
        "transitions": 100_000,
        "mean": pytest.approx(0.212),
        "n": 5,
    }
    assert telemetry[f"{ARM}_s10"]["final_telemetry"]["bank_rows_mean"] == 128.0
    report = markdown(result)
    assert "Final retained learner telemetry" in report
    assert "Five-seed side means per nominal 100k" in report
    assert "Per-final re-roll" in report
    assert f"| {ARM}_s10 | 0.7000 | 0.6000 / 0.8000" in report


def test_t3_context_preserves_exhausted_extension_and_source_binding():
    from research.t4a_screen_20261016.analyze_t4a import markdown, t3_context_from_result

    context = t3_context_from_result(
        {
            "arms": {
                "LABELS": {
                    "decision": "EXTEND",
                    "extension_exhausted": True,
                    "vs_ctrl": {
                        "scripted": {"episode_survival": {"mean": 0.01}, "mass": {"mean": -80.0}}
                    },
                }
            }
        }
    )
    assert context["LABELS"]["decision"] == "EXTEND exhausted / stop"
    rows, statuses = _dataset()
    result = analyze(rows, statuses, t3_context=context)
    result["t3_context_binding"] = {"source_commit": "abc123", "sha256": "deadbeef"}
    report = markdown(result)
    assert "| TWO_SIDED |" in report
    assert "| LABELS |" in report
    assert "EXTEND exhausted / stop" in report
    assert "abc123" in report and "deadbeef" in report
