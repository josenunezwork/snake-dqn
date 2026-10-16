import pytest
import json

from research.t3_screen_20261015.analyze_t4c import (
    ARM,
    EXTENSION_ROUND,
    analyze,
    extension_requirements,
    markdown,
)


def _row(name, mix, death, mass=100.0, round_index=66, offset=0):
    count = 96 if mix == "scripted" else 32
    worlds = [str(offset + index) for index in range(count)]
    return {
        "name": name,
        "round_index": round_index,
        "mix": mix,
        "frames": 5000,
        "mass": {world: mass for world in worlds},
        "deaths": {world: death for world in worlds},
        "alive_frames": {world: 5000 if not death else 2500 for world in worlds},
        "length_at_5000": {world: 20.0 if not death else 0.0 for world in worlds},
    }


def _data(candidate_death=0):
    rows = []
    for seed in range(10, 15):
        for mix in ("scripted", "frozen", "mixed"):
            rows.extend(
                (_row(f"CTRL_s{seed}", mix, 1), _row(f"{ARM}_s{seed}", mix, candidate_death))
            )
    rows += [_row("baseline", mix, 1, 75.0) for mix in ("scripted", "frozen", "mixed")]
    return rows, {f"{ARM}_s{seed}": 0 for seed in range(10, 15)}


def test_go_uses_registered_five_seed_primary_and_mass_guards():
    rows, statuses = _data()
    result = analyze(rows, statuses)
    data = result["arms"][ARM]
    assert data["decision"] == "GO"
    assert data["vs_ctrl"]["scripted"]["episode_survival"]["mean"] == 1.0
    assert data["vs_ctrl"]["mixed"]["mass"]["seed_n"] == 5


def test_nonpositive_primary_kills_and_halted_run_kills_without_fake_rows():
    rows, statuses = _data(1)
    assert analyze(rows, statuses)["arms"][ARM]["decision"] == "KILL"
    halted = analyze([], {f"{ARM}_s10": 9})
    assert halted["arms"][ARM]["decision"] == "KILL"
    # The watcher writes strict JSON: absent readouts must be null, not NaN.
    json.dumps(halted, allow_nan=False)


def test_exactly_one_extension_uses_scripted_rows_only_and_exhausts_to_inconclusive():
    rows, statuses = _data(1)
    first = analyze(rows, statuses)
    assert first["arms"][ARM]["decision"] == "KILL"
    # 1/96 surviving gives positive but below GO, therefore EXTEND.
    for row in rows:
        if row["name"].startswith(ARM) and row["mix"] == "scripted":
            key = next(iter(row["deaths"]))
            row["deaths"][key] = 0
            row["alive_frames"][key] = 5000
            row["length_at_5000"][key] = 20
    first = analyze(rows, statuses)
    assert first["arms"][ARM]["decision"] == "EXTEND"
    assert len(extension_requirements(first, rows)) == 10
    rows += [
        _row(name, "scripted", 1, round_index=EXTENSION_ROUND, offset=1000)
        for seed in range(10, 15)
        for name in (f"{ARM}_s{seed}", f"CTRL_s{seed}")
    ]
    final = analyze(rows, statuses)["arms"][ARM]
    assert final["decision"] == "INCONCLUSIVE"
    assert extension_requirements({"arms": {ARM: final}}, rows) == []


def test_shared_survivor_length_is_paired_and_requires_complete_length_maps():
    rows, statuses = _data()
    data = analyze(rows, statuses)["arms"][ARM]["shared_survivor_length_h5000"]
    assert data["shared_survivors"] == 0  # CTRL dies in this fixture
    for row in rows:
        if row["name"].startswith("CTRL") and row["mix"] == "scripted":
            for key in row["deaths"]:
                row["deaths"][key] = 0
                row["alive_frames"][key] = 5000
                row["length_at_5000"][key] = 10
    assert (
        analyze(rows, statuses)["arms"][ARM]["shared_survivor_length_h5000"][
            "candidate_minus_ctrl_mean_length"
        ]
        == 10
    )
    rows[0]["length_at_5000"].pop(next(iter(rows[0]["length_at_5000"])))
    with pytest.raises(ValueError, match="length_at_5000"):
        analyze(rows, statuses)


def test_markdown_exposes_all_registered_readouts_and_denominators():
    rows, statuses = _data()
    result = analyze(rows, statuses)
    result["arms"][ARM]["initial_decision"] = "EXTEND"
    result["arms"][ARM]["extension_started"] = True
    result["arms"][ARM]["extension_complete"] = True
    agreement = {
        key: {"rows": 10, "agreement": 0.9}
        for key in (
            "retained_decisive_all",
            "retained_decisive_gap_lt100",
            "retained_decisive_gap_100_300",
            "in_window_counter",
            "outside_counter",
        )
    }
    result["arms"][ARM]["diagnostics"] = {
        "bank_agreement_five_seed_means": {"retained_decisive_all": {"mean": 0.9, "n": 5}},
        "bank_agreement_raw_six_action": {f"{ARM}_s10": agreement},
        "telemetry_final_100k": {
            "five_seed_means": {"bank_agree_decisive": {"mean": 0.8, "n": 5}},
            "per_run": {
                f"{ARM}_s10": {
                    "bank_agree_decisive": 0.8,
                    "bank_agree_counter": 0.9,
                    "bank_agree_outside": 0.95,
                }
            },
        },
        "drift_candidate_readout": {
            "per_seed": {"10": {"raw_change_rate": 0.1, "consequential_change_rate": 0.02}},
            "baseline_per_seed": {
                "10": {"raw_change_rate": 0.08, "consequential_change_rate": 0.01}
            },
            "candidate_raw_change_rate": 0.1,
            "candidate_consequential_change_rate": 0.02,
            "baseline_five_seed_mean": {"raw_change_rate": 0.08, "consequential_change_rate": 0.01},
        },
        "drift": {"within_ten_percentage_points": True},
    }
    report = markdown(result)
    for phrase in (
        "Initial decision: EXTEND",
        "mass vs incumbent",
        "100-300 gap",
        "Final 100k sampled telemetry",
        "B8 raw",
        "Within 10 percentage points: True",
        "Five-seed mean:",
    ):
        assert phrase in report
