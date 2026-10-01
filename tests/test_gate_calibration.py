"""Tests for the read-only gate-calibration analysis (research/gate_calibration_20260926)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "research" / "gate_calibration_20260926" / "analyze.py"
REAL_SOURCE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/"
    "task-aligned-strict-challenge-20260926/pilot-v1/output"
)


def _load():
    spec = importlib.util.spec_from_file_location("gate_calibration_analyze", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ga = _load()


def _record(mix, seed, survival, alive_mass, cause="self"):
    return {
        "deaths": 1,
        "kills": 0,
        "mass_integral": survival * alive_mass,
        "max_mass": alive_mass * 1.5,
        "mean_mass_alive": alive_mass,
        "probes": {"death_cause": cause, "food_eaten": 10, "boost_frame_fraction": 0.1},
        "seed": seed,
        "survival_fraction": survival,
        "world_identity": {"mix_id": mix, "seed": seed},
    }


def _write(root, stage, arm, rec):
    folder = root / stage / "producer" / "records"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{arm}-{rec['world_identity']['mix_id']}-{rec['seed']}.json"
    path.write_text(json.dumps(rec), encoding="utf-8")


def _synthetic_tree(root, candidate_scale=1.0, n=6):
    """Development Apex bank plus a paired pilot on disjoint seeds."""
    causes = ["self", "self", "head_on", "self", "wall", "self"]
    for m, mix in enumerate(ga.MIXES):
        for i in range(n):
            s = 0.1 + 0.05 * i + 0.01 * m
            a = 50.0 + 20.0 * ((i * 7 + m) % 5)
            _write(root, "development", "incumbent", _record(mix, 100 + i, s, a, causes[i % 6]))
            s2 = 0.12 + 0.04 * ((i * 3) % n)
            a2 = 60.0 + 15.0 * ((i * 5 + m) % 4)
            _write(root, "pilot", "incumbent", _record(mix, 500 + i, s2, a2))
            _write(
                root,
                "pilot",
                "candidate",
                _record(mix, 500 + i, s2, a2 * candidate_scale, "enemy_body"),
            )
    return root


def test_pilot_sizing_rule_reproduced():
    # Values copied from the saved pilot sizing.json (paired SD, MDE, recommended_n).
    assert ga.required_count(56.32086260491015, 5.260730000000001) == 1072
    assert ga.required_count(118.26351385132836, 3.8815350000000004) == 8676
    assert ga.required_count(55.80648244442426, 7.2648375000000005) == 552
    assert ga.required_count(None, 1.0) is None
    assert ga.PLANNING_ALPHA == pytest.approx(0.05 / 3)


def test_rank_and_sign_helpers():
    assert ga.average_ranks([3.0, 1.0, 3.0, 2.0]).tolist() == [3.5, 1.0, 3.5, 2.0]
    # 5 wins of 16 is the pilot's per-mix record: two-sided p ~= 0.21.
    assert ga.sign_test_two_sided(5, 11) == pytest.approx(0.2101135, abs=1e-6)
    assert ga.sign_test_two_sided(0, 0) is None
    assert ga.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ga.pearson([1, 1, 1], [1, 2, 3]) is None
    stats = ga.describe([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0 and stats["sd"] == 1.0 and stats["cv"] == 0.5


def test_load_stage_rejects_mismatched_filename(tmp_path):
    folder = tmp_path / "pilot" / "producer" / "records"
    folder.mkdir(parents=True)
    (folder / "incumbent-frozen-1.json").write_text(
        json.dumps(_record("frozen", 2, 0.2, 50.0)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        ga.load_stage(str(tmp_path / "pilot"), "pilot")


def test_pairing_requires_identical_worlds():
    inc = [ga.normalize_record(_record("frozen", 1, 0.2, 50.0), "incumbent", "pilot")]
    cand = [ga.normalize_record(_record("frozen", 2, 0.2, 50.0), "candidate", "pilot")]
    with pytest.raises(ValueError, match="worlds differ"):
        ga.pair_by_world(cand, inc)


def test_normalize_marks_alive_at_horizon():
    raw = _record("mixed", 7, 1.0, 40.0)
    raw["deaths"] = 0
    raw["probes"]["death_cause"] = None
    assert ga.normalize_record(raw, "incumbent", "pilot")["death_cause"] == "alive_at_horizon"
    with pytest.raises(ValueError, match="unknown mix"):
        ga.normalize_record(_record("solo", 7, 0.5, 40.0), "incumbent", "pilot")


def test_aa_identical_arms_have_zero_delta(tmp_path):
    result = ga.analyze(str(_synthetic_tree(tmp_path, candidate_scale=1.0)), 10.0)
    for mix in ga.MIXES:
        block = result["paired"]["per_mix"][mix]["mass_integral"]
        assert block["delta"]["mean"] == 0.0
        assert block["delta"]["sd"] == 0.0
        assert block["ties"] == block["n"] == 6
        assert block["delta_t"] is None
    assert result["inputs"]["development_pilot_world_overlap"] == 0
    timing = result["death_timing"]["development/incumbent/all"]
    assert timing["max_identity_error"] == pytest.approx(0.0, abs=1e-9)
    shares = (
        timing["share_var_log_survival"]
        + timing["share_var_log_mean_mass_alive"]
        + timing["share_2cov"]
    )
    assert shares == pytest.approx(1.0)


def test_clear_loser_triggers_futility_and_counts(tmp_path):
    result = ga.analyze(str(_synthetic_tree(tmp_path, candidate_scale=0.3)), 10.0)
    frozen = result["paired"]["per_mix"]["frozen"]
    assert frozen["mass_integral"]["losses"] == 6
    assert frozen["survival_fraction"]["ties"] == 6
    last_look = result["futility"]["looks"]["6"]
    assert last_look["mass_integral"]["stop"] is True
    causes = result["death_causes"]
    assert causes["pilot/candidate/all"] == {"enemy_body": 18}
    assert causes["development/incumbent/all"] == {"head_on": 3, "self": 12, "wall": 3}
    budget = result["throughput"]["per_estimand"]["mass_integral"]
    assert budget["candidate_episodes"] == 3 * budget["gate_worlds_per_mix"]
    assert budget["live_cpu_hours_per_candidate"] == pytest.approx(
        budget["candidate_episodes"] * 10.0 / 3600.0
    )
    ladder = result["log_mde_ladder"]["0.10"]
    assert ladder["delta_log_mde"] == pytest.approx(math.log(1.1))


def test_main_is_deterministic_and_guards_output(tmp_path):
    source = _synthetic_tree(tmp_path / "src")
    out_a = tmp_path / "out" / "a.json"
    out_b = tmp_path / "out" / "b.json"
    assert ga.main(["--source", str(source), "--out", str(out_a)]) == 0
    assert ga.main(["--source", str(source), "--out", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()
    payload = json.loads(out_a.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ga.SCHEMA_VERSION
    assert "UNAUDITED" in payload["disclaimer"]
    with pytest.raises(SystemExit):
        ga.main(["--source", str(source), "--out", str(source / "x.json")])
    with pytest.raises(SystemExit):
        ga.main(["--source", str(source), "--out", os.path.join(ga.PROTECTED_ROOT, "x.json")])


@pytest.mark.skipif(not REAL_SOURCE.is_dir(), reason="saved pilot-v1 records not present")
def test_real_records_reproduce_saved_pilot_sizing():
    sizing = json.loads((REAL_SOURCE / "pilot" / "producer" / "sizing.json").read_text())
    result = ga.analyze(str(REAL_SOURCE))
    assert result["inputs"]["record_count"] == 144
    for mix, saved in sizing["sizing"]["per_mix"].items():
        block = result["paired"]["per_mix"][mix]["mass_integral"]
        assert block["delta"]["sd"] == pytest.approx(saved["paired_delta_std"], rel=1e-12)
        assert block["mde"] == pytest.approx(saved["mde"], rel=1e-12)
        assert block["required_worlds_paired"] == saved["recommended_n"]
    assert result["death_causes"]["all_incumbent"] == {
        "enemy_body": 1,
        "head_on": 9,
        "self": 85,
        "wall": 1,
    }
