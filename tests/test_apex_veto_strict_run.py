"""Tests for the Tier-2 Apex veto strict runner (research/apex_veto_strict_20260927).

Record-shape and audit rules are exercised on REAL Tier-1 screen records copied to
``tests/fixtures/apex_veto_strict_runner`` (positive) and on mutated copies (negative),
per governance "Audit scope" rule 4.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_strict_20260927 import strict_run as sr
from src.evaluation.strict_promotion import (
    StrictPromotionArtifactError,
    _validate_e1_record,
    validate_strict_world_record,
)

FIXTURES = Path(__file__).parent / "fixtures" / "apex_veto_strict_runner"
SCRIPTED_SEED = 1022974241
FROZEN_SEED = 1397599937
SCREEN_ROWS = {
    (row["mix"], row["world_seed"]): row
    for row in dev_screen._design_rows(dev_screen.screen_seeds(40))
}
HAVE_PILOT = dev_screen.DEFAULT_PILOT_OUTPUT.is_dir()
HAVE_SCREEN = (sr.SCREEN_RUN / "records").is_dir()
HAVE_CHECKPOINTS = all(
    (dev_screen.DEFAULT_CHECKPOINT_DIR / n).is_file() for n, _ in dev_screen.POOL
)


def fixture(arm: str, mix: str, seed: int) -> dict:
    return json.loads((FIXTURES / f"{arm}-{mix}-{seed}.json").read_text())


@pytest.fixture
def profile(setup_config):
    """The pinned deployment profile, resolved from the pinned config (not from a record)."""
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    load_and_initialize_config(str(dev_screen.DEFAULT_CONFIG))
    resolved = evaluation_profile_for_name(sr.PROFILE_NAME, sr.HORIZON)
    assert resolved.digest == dev_screen.PROFILE_DIGEST
    return {"descriptor": resolved.descriptor(), "digest": resolved.digest}


@pytest.fixture
def wrapper() -> dict:
    return sr.wrapper_identity()


# ---------------------------------------------------------------- namespaces and rosters


class TestNamespaces:
    def test_counts_unique_and_mutually_disjoint(self):
        seeds = sr.namespace_seeds()
        assert {k: len(v) for k, v in seeds.items()} == {"dev": 16, "final": 300, "serving": 50}
        everything = [s for values in seeds.values() for s in values]
        assert len(set(everything)) == len(everything)
        assert seeds == sr.namespace_seeds()  # deterministic recipe

    @pytest.mark.skipif(not HAVE_PILOT, reason="strict pilot records not available")
    def test_fresh_against_every_checked_namespace(self):
        report = sr.namespace_report(sr.namespace_seeds(), dev_screen.DEFAULT_PILOT_OUTPUT)
        assert report["disjoint"] is True
        for row in report["namespaces"].values():
            assert row["earlier_namespaces"]["recipe_reproduces_pilot"] is True
            assert row["tier1_screen_or_smoke_overlap"] == []

    def test_fails_closed_without_pilot_records(self):
        assert sr.namespace_report(sr.namespace_seeds(), None)["disjoint"] is False

    @pytest.mark.skipif(not HAVE_PILOT, reason="strict pilot records not available")
    def test_detects_screen_and_cross_namespace_overlap(self):
        seeds = sr.namespace_seeds()
        screen = copy.deepcopy(seeds)
        screen["final"][5] = dev_screen.screen_seeds(40)[3]
        assert sr.namespace_report(screen, dev_screen.DEFAULT_PILOT_OUTPUT)["disjoint"] is False
        cross = copy.deepcopy(seeds)
        cross["serving"][0] = cross["dev"][0]
        report = sr.namespace_report(cross, dev_screen.DEFAULT_PILOT_OUTPUT)
        assert report["disjoint"] is False
        assert report["cross_namespace_overlap"]["dev|serving"] == [cross["dev"][0]]


class TestRostersAndShards:
    def test_prefix_rows_equal_full_bank_prefix(self):
        bank = sr.namespace_seeds()["final"]
        full = sr.build_rosters([], bank, [])["final"]
        prefix = sr.build_rosters([], bank[:235], [])["final"]
        for mix in sr.MIXES:
            assert [r for r in prefix if r["mix"] == mix] == [r for r in full if r["mix"] == mix][
                :235
            ]

    def test_stage_plans_and_deterministic_balanced_shards(self):
        seeds = sr.namespace_seeds()
        rosters = sr.build_rosters(seeds["dev"], seeds["final"][:7], seeds["serving"])
        calibration = sr.stage_plan("calibration", rosters["calibration"])
        final = sr.stage_plan("final", rosters["final"])
        serving = sr.stage_plan("serving", rosters["serving"])
        assert sum(len(u) for u in calibration) == 48
        assert all([e["arm"] for e in u] == ["incumbent", "candidate"] for u in final)
        assert sum(len(u) for u in final) == 42 and len(serving) == 50
        assert {e["arm"] for u in serving for e in u} == {"candidate"}
        shards = [sr.shard_units(final, k) for k in range(sr.WORKERS)]
        ids = [e["episode_id"] for shard in shards for unit in shard for e in unit]
        assert sorted(ids) == sorted(e["episode_id"] for u in final for e in u)
        assert abs(len(shards[0]) - len(shards[1])) <= 1
        assert shards == [sr.shard_units(final, k) for k in range(sr.WORKERS)]

    def test_serving_rows_are_selfplay_incumbent(self):
        rows = sr.serving_rows([7, 8])
        assert [r["mix"] for r in rows] == [sr.SERVING_MIX] * 2
        assert {s["member_sha256"] for r in rows for s in r["slots"]} == {sr.CHAMPION[1]}

    def test_smoke_rosters_are_one_mix_and_empty_safe(self):
        rosters = sr.build_rosters([], [123], [], mixes=["scripted"])
        assert rosters["calibration"] == [] and rosters["serving"] == []
        assert [r["mix"] for r in rosters["final"]] == ["scripted"]


# ---------------------------------------------------------------- sizing


class TestSizing:
    @pytest.mark.skipif(not HAVE_SCREEN, reason="Tier-1 screen records not available")
    def test_real_screen_pilot_freezes_n_235(self):
        pilot = sr.screen_pilot_deltas(sr.SCREEN_RUN)
        assert pilot["summary_deltas_match"] is True
        assert {m: len(v) for m, v in pilot["deltas_by_mix"].items()} == {
            "frozen": 40,
            "scripted": 40,
            "mixed": 40,
        }
        sizing = sr.frozen_final_n(pilot["deltas_by_mix"])
        assert sizing["final_worlds_per_mix"] == 235 and sizing["feasible"] is True
        assert sizing["pilot_sizing"]["per_mix"]["scripted"]["recommended_n"] == 235

    def test_floor_and_kill_criterion(self):
        tight = {m: [20.0 + (i % 2) for i in range(10)] for m in sr.MIXES}
        assert sr.frozen_final_n(tight)["final_worlds_per_mix"] == sr.N_FLOOR
        wide = {m: [(-1) ** i * 150.0 for i in range(10)] for m in sr.MIXES}
        sizing = sr.frozen_final_n(wide)
        assert sizing["final_worlds_per_mix"] > sr.N_MAX and sizing["feasible"] is False

    def test_pilot_deltas_from_real_records_layout(self, tmp_path):
        (tmp_path / "records").mkdir()
        for name in FIXTURES.glob("*.json"):
            (tmp_path / "records" / name.name).write_bytes(name.read_bytes())
        pilot = sr.screen_pilot_deltas(tmp_path)
        a, b = fixture("A", "scripted", SCRIPTED_SEED), fixture("B", "scripted", SCRIPTED_SEED)
        assert pilot["deltas_by_mix"]["scripted"] == [
            b["record"]["mass_integral"] - a["record"]["mass_integral"]
        ]
        assert pilot["summary_deltas_match"] is None  # no summary in this layout
        walls = pilot["episode_wall_seconds"]
        assert walls["A-scripted"] == [a["wall_seconds"]] and walls["B-frozen"] == [
            fixture("B", "frozen", FROZEN_SEED)["wall_seconds"]
        ]

    @pytest.mark.skipif(not HAVE_SCREEN, reason="Tier-1 screen records not available")
    def test_real_screen_runtime_projection_fits_the_final_cap(self):
        walls = sr.screen_pilot_deltas(sr.SCREEN_RUN)["episode_wall_seconds"]
        assert {k: len(v) for k, v in walls.items()} == {
            f"{a}-{m}": 40 for a in ("A", "B") for m in sr.MIXES
        }
        projection = sr.runtime_projection(walls, 235)
        per_mix = projection["per_mix_episode_seconds"]
        assert per_mix["frozen"]["candidate_mean_seconds"] == pytest.approx(52.4, abs=0.1)
        assert per_mix["frozen"]["max_seconds"] == pytest.approx(271.3, abs=0.1)
        assert projection["pair_seconds_per_world_triplet"] == pytest.approx(189.8, abs=0.1)
        assert projection["rollout_seconds_per_worker"] == pytest.approx(22302, abs=5)
        assert projection["within_limit"] is True
        assert projection["fraction_of_worker_budget"] == pytest.approx(0.570, abs=0.002)
        old_cap = sr.runtime_projection(walls, 235, cap=30000)
        assert old_cap["within_limit"] is False  # the former 30000 s cap would be refused

    def test_projection_refuses_missing_timings(self):
        walls = {f"{a}-{m}": [10.0] for a in ("A", "B") for m in sr.MIXES}
        assert sr.runtime_projection(walls, 40)["within_limit"] is True
        walls["B-mixed"] = []
        with pytest.raises(sr.StrictRunError, match="no measured screen wall times"):
            sr.runtime_projection(walls, 40)


# ---------------------------------------------------------------- record shape (real records)


class TestRecordShape:
    def test_real_candidate_and_incumbent_pass(self, profile, wrapper):
        for mix, seed in (("scripted", SCRIPTED_SEED), ("frozen", FROZEN_SEED)):
            row = SCREEN_ROWS[(mix, seed)]
            validate_strict_world_record(fixture("A", mix, seed)["record"], profile, row)
            validate_strict_world_record(
                fixture("B", mix, seed)["record"],
                profile,
                row,
                candidate_wrapper=wrapper["descriptor"],
            )

    def test_default_path_unchanged(self, profile):
        row = SCREEN_ROWS[("scripted", SCRIPTED_SEED)]
        record = fixture("A", "scripted", SCRIPTED_SEED)["record"]
        assert validate_strict_world_record(record, profile, row) == _validate_e1_record(
            record, profile, row
        )
        with pytest.raises(StrictPromotionArtifactError, match="extra"):
            _validate_e1_record(fixture("B", "scripted", SCRIPTED_SEED)["record"], profile, row)

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda r: r["probes"]["safety_veto"].update(method="free-space-veto/v1"), "identity"),
            (lambda r: r["probes"]["safety_veto"]["counters"].update(kept_base=0), "inconsistent"),
            (lambda r: r["probes"]["safety_veto"]["counters"].update(extra=1), "extra"),
            (lambda r: r["probes"]["safety_veto"]["counters"].update(decisions=-1), "non-neg"),
            (lambda r: r["probes"].pop("safety_veto"), "lacks its wrapper"),
            (lambda r: r.update(survival_fraction=0.5), "survival_fraction"),
            (lambda r: r.update(evaluation_profile_digest="0" * 64), "identity"),
        ],
    )
    def test_mutated_candidate_rejected(self, profile, wrapper, mutate, message):
        record = copy.deepcopy(fixture("B", "scripted", SCRIPTED_SEED)["record"])
        mutate(record)
        with pytest.raises(StrictPromotionArtifactError, match=message):
            validate_strict_world_record(
                record,
                profile,
                SCREEN_ROWS[("scripted", SCRIPTED_SEED)],
                candidate_wrapper=wrapper["descriptor"],
            )

    def test_incumbent_with_wrapper_and_wrong_roster_rejected(self, profile, wrapper):
        record = fixture("A", "scripted", SCRIPTED_SEED)["record"]
        with pytest.raises(StrictPromotionArtifactError, match="lacks its wrapper"):
            validate_strict_world_record(
                record,
                profile,
                SCREEN_ROWS[("scripted", SCRIPTED_SEED)],
                candidate_wrapper=wrapper["descriptor"],
            )
        with pytest.raises(StrictPromotionArtifactError, match="identity"):
            validate_strict_world_record(record, profile, SCREEN_ROWS[("frozen", FROZEN_SEED)])


# ---------------------------------------------------------------- self-check audit rules


def real_entries(stage: str, wrapper: dict, rows=None) -> tuple:
    """Envelopes built by the producer's own ``episode_entry`` around real records."""
    rows = rows or [SCREEN_ROWS[("scripted", SCRIPTED_SEED)], SCREEN_ROWS[("frozen", FROZEN_SEED)]]
    arms = {"final": ("A", "B"), "calibration": ("A",), "serving": ("B",)}[stage]
    entries = {}
    for unit in sr.stage_plan(stage, rows):
        for episode, arm in zip(unit, arms):
            source = fixture(arm, episode["mix"], episode["world_seed"])
            row = SCREEN_ROWS[(episode["mix"], episode["world_seed"])]
            entries[episode["episode_id"]] = sr.episode_entry(
                episode, row, source["record"], wrapper, 0, 1.0, source["world_index"]
            )
    index = {SCRIPTED_SEED: 19, FROZEN_SEED: 9}
    return entries, rows, index


class TestSelfCheckRules:
    def test_real_final_entries_pass_and_report_veto_consistency(self, profile, wrapper):
        entries, rows, index = real_entries("final", wrapper)
        assert {e["world_index"] for e in entries.values()} == {19, 9}
        result = sr.audit_stage_entries("final", entries, rows, profile, wrapper, index)
        assert result["failures"] == []
        assert result["reported_not_gated"] == {
            "candidate_records_with_veto": 2,
            "veto_decisions_equal_decision_frames": 2,
        }

    def test_veto_counter_mismatch_is_reported_not_gated(self, profile, wrapper):
        entries, rows, index = real_entries("final", wrapper)
        key = f"final-candidate-scripted-{SCRIPTED_SEED}"
        counters = entries[key]["record"]["probes"]["safety_veto"]["counters"]
        counters["decisions"] += 1
        counters["kept_base"] += 1  # keeps the code-guaranteed identity
        result = sr.audit_stage_entries("final", entries, rows, profile, wrapper, index)
        assert result["failures"] == []
        assert result["reported_not_gated"]["veto_decisions_equal_decision_frames"] == 1

    @pytest.mark.parametrize(
        "mutate, fragment",
        [
            (lambda e: e.update(wrapper=None), "envelope"),
            (lambda e: e.update(safety_veto=False), "envelope"),
            (lambda e: e.update(world_index=3), "envelope"),
            (lambda e: e.update(roster_member_sha256s=["0" * 64] * 5), "envelope"),
            (lambda e: e.update(wrapper_source_sha256="f" * 64), "envelope"),
            (lambda e: e["record"]["probes"].pop("safety_veto"), "record shape"),
            (lambda e: e["record"].update(mass_integral=-1.0), "record shape"),
        ],
    )
    def test_mutated_candidate_envelope_fails(self, profile, wrapper, mutate, fragment):
        entries, rows, index = real_entries("final", wrapper)
        mutate(entries[f"final-candidate-scripted-{SCRIPTED_SEED}"])
        failures = sr.audit_stage_entries("final", entries, rows, profile, wrapper, index)[
            "failures"
        ]
        assert failures and fragment in failures[0]

    def test_swapped_arms_and_unplanned_entries_fail(self, profile, wrapper):
        entries, rows, index = real_entries("final", wrapper)
        inc = f"final-incumbent-scripted-{SCRIPTED_SEED}"
        cand = f"final-candidate-scripted-{SCRIPTED_SEED}"
        entries[inc]["record"], entries[cand]["record"] = (
            entries[cand]["record"],
            entries[inc]["record"],
        )
        entries["final-candidate-mixed-1"] = copy.deepcopy(entries[cand])
        failures = sr.audit_stage_entries("final", entries, rows, profile, wrapper, index)[
            "failures"
        ]
        assert any(inc in f and "record shape" in f for f in failures)
        assert any(cand in f and "record shape" in f for f in failures)
        assert any("not in the stage plan" in f for f in failures)

    def test_smoke_mode_checks_wrapper_presence_only(self, wrapper):
        entries, rows, index = real_entries("final", wrapper)
        assert (
            sr.audit_stage_entries("final", entries, rows, None, wrapper, index)["failures"] == []
        )
        entries[f"final-incumbent-frozen-{FROZEN_SEED}"]["record"]["probes"]["safety_veto"] = {}
        failures = sr.audit_stage_entries("final", entries, rows, None, wrapper, index)["failures"]
        assert failures and "presence" in failures[0]


# ---------------------------------------------------------------- stage evidence integrity


def write_stage(stage_dir: Path, stage: str, entries: dict, rows: list, drop=()) -> None:
    """Lay out real envelopes the way two workers write them."""
    (stage_dir / "records").mkdir(parents=True)
    units = sr.stage_plan(stage, rows)
    for shard in range(sr.WORKERS):
        planned = [e["episode_id"] for u in sr.shard_units(units, shard) for e in u]
        done = {}
        for episode_id in planned:
            if episode_id in drop:
                continue
            done[episode_id] = sr.write_durable(
                stage_dir / "records" / f"{episode_id}.json", entries[episode_id]
            )
        (stage_dir / f"shard-{shard}").mkdir()
        sr.write_durable(
            stage_dir / f"shard-{shard}" / "report.json",
            {
                "planned_episode_ids": planned,
                "records_sha256": done,
                "complete": len(done) == len(planned),
                "stopped_reason": "deadline: 10s left < 45s budget" if drop else None,
            },
        )


class TestCollectStage:
    def test_complete_real_stage(self, tmp_path, wrapper):
        entries, rows, _ = real_entries("final", wrapper)
        write_stage(tmp_path, "final", entries, rows)
        collected = sr.collect_stage(tmp_path, "final", rows)
        assert collected["complete"] and collected["missing"] == []
        assert collected["entries"] == json.loads(json.dumps(entries))

    def test_tampered_extra_and_missing_records(self, tmp_path, wrapper):
        entries, rows, _ = real_entries("final", wrapper)
        write_stage(tmp_path / "a", "final", entries, rows)
        path = tmp_path / "a" / "records" / f"final-candidate-frozen-{FROZEN_SEED}.json"
        path.write_text(path.read_text().replace('"mass_integral": ', '"mass_integral": 1'))
        with pytest.raises(sr.StrictRunError, match="bytes changed"):
            sr.collect_stage(tmp_path / "a", "final", rows)

        write_stage(tmp_path / "b", "final", entries, rows)
        (tmp_path / "b" / "records" / "final-candidate-mixed-1.json").write_text("{}")
        with pytest.raises(sr.StrictRunError, match="differ from shard reports"):
            sr.collect_stage(tmp_path / "b", "final", rows)

        dropped = f"final-candidate-scripted-{SCRIPTED_SEED}"
        write_stage(tmp_path / "c", "final", entries, rows, drop={dropped})
        collected = sr.collect_stage(tmp_path / "c", "final", rows)
        assert collected["complete"] is False and collected["missing"] == [dropped]
        assert collected["stopped"][0].startswith("deadline")


# ---------------------------------------------------------------- calibration and decision


def synthetic_final(wrapper: dict, shift: float, n: int = 12):
    """Final-stage entries over n worlds per mix built from real record shapes."""
    base_a = fixture("A", "scripted", SCRIPTED_SEED)
    base_b = fixture("B", "scripted", SCRIPTED_SEED)
    seeds = list(range(1000, 1000 + n))
    rows = [
        {"mix": mix, "world_seed": seed, "slots": [{"slot": 1, "member_sha256": sr.CHAMPION[1]}]}
        for mix in sr.MIXES
        for seed in seeds
    ]
    entries = {}
    for row in rows:
        for arm, source, extra in (("incumbent", base_a, 0.0), ("candidate", base_b, shift)):
            record = copy.deepcopy(source["record"])
            wobble = ((row["world_seed"] * 7919) % 13) - 6.0
            record["mass_integral"] = 40.0 + wobble * (1.0 if arm == "candidate" else 0.5) + extra
            record["survival_fraction"] = 0.6  # inside every calibration band
            unit = {
                "episode_id": f"final-{arm}-{row['mix']}-{row['world_seed']}",
                "stage": "final",
                "arm": arm,
                "mix": row["mix"],
                "world_seed": row["world_seed"],
            }
            entries[unit["episode_id"]] = sr.episode_entry(unit, row, record, wrapper, 0, 1.0, 0)
    return entries, rows


def calibration_entries() -> dict:
    """Two incumbent worlds per mix from the two real incumbent records."""
    entries = {}
    for mix in sr.MIXES:
        for k, (m, seed) in enumerate((("scripted", SCRIPTED_SEED), ("frozen", FROZEN_SEED))):
            entries[f"calibration-incumbent-{mix}-{k}"] = {
                "mix": mix,
                "arm": "incumbent",
                "record": fixture("A", m, seed)["record"],
            }
    return entries


class TestCalibrationAndDecision:
    def test_calibration_margin_and_bands_from_real_records(self):
        entries = calibration_entries()
        calibration = sr.calibration_reference(entries)
        a = fixture("A", "scripted", SCRIPTED_SEED)["record"]
        b = fixture("A", "frozen", FROZEN_SEED)["record"]
        expected = 0.03 * (a["mass_integral"] + b["mass_integral"]) / 2
        assert calibration["absolute_delta_ni"] == pytest.approx(expected, abs=1e-12)
        band = calibration["survival_bands"]["mixed"]
        mean = (a["survival_fraction"] + b["survival_fraction"]) / 2
        assert band["metric"] == "survival_fraction" and band["mix_scope"] == ["mixed"]
        assert band["lower"] == pytest.approx(mean - 0.02) and band["upper"] == pytest.approx(
            mean + 1
        )
        assert sr.audit_calibration(entries, calibration) == []
        tampered = copy.deepcopy(calibration)
        tampered["absolute_delta_ni"] *= 1.01
        tampered["survival_bands"]["frozen"]["lower"] -= 0.01
        problems = sr.audit_calibration(entries, tampered)
        assert "absolute_delta_ni differs" in problems
        assert any("frozen" in p for p in problems)

    def test_audit_t_matches_frozen_student_t(self):
        from src.scripts.eval_stats import student_t_isf, student_t_sf

        for t, df in ((0.3, 5), (1.7, 39), (2.4, 234), (-1.1, 12), (5.5, 100)):
            assert sr.audit_t_sf(t, df) == pytest.approx(student_t_sf(t, df), abs=1e-8)
        for p, df in ((0.05, 234), (0.05 / 3, 39), (0.2, 10)):
            assert sr.audit_t_isf(p, df) == pytest.approx(student_t_isf(p, df), abs=1e-6)

    @pytest.mark.parametrize("shift, passes", [(25.0, True), (0.0, False)])
    def test_producer_and_audit_agree(self, wrapper, shift, passes):
        entries, rows = synthetic_final(wrapper, shift)
        calibration = sr.calibration_reference(calibration_entries())
        decision = sr.final_decision(sr.paired_final(entries, rows), calibration)
        assert decision["passes"] is passes and decision["valid"] is True
        assert decision["strict_promotion_decision"]["decision_method"] == "strict-promotion-v1"
        audited = sr.audit_final(entries, rows, calibration, decision)
        assert audited["problems"] == [] and audited["passes"] is passes
        assert audited["cross_check"]["agree"] is True

    def test_audit_final_catches_producer_disagreement_and_unpaired(self, wrapper):
        entries, rows = synthetic_final(wrapper, 25.0)
        calibration = sr.calibration_reference(calibration_entries())
        decision = sr.final_decision(sr.paired_final(entries, rows), calibration)
        lying = dict(decision, passes=False)
        assert (
            "audit pass/fail differs from the producer decision"
            in sr.audit_final(entries, rows, calibration, lying)["problems"]
        )
        entries.pop(f"final-candidate-mixed-{rows[-1]['world_seed']}")
        assert sr.audit_final(entries, rows, calibration, decision)["problems"][0].startswith(
            "unpaired"
        )

    def test_band_out_of_range_fails_strict(self, wrapper):
        entries, rows = synthetic_final(wrapper, 25.0)
        calibration = sr.calibration_reference(calibration_entries())
        calibration["survival_bands"]["frozen"]["lower"] = 0.99
        decision = sr.final_decision(sr.paired_final(entries, rows), calibration)
        assert decision["strict_promotion_decision"]["passes"] is True
        assert decision["bands_pass"] is False and decision["passes"] is False

    def test_outcome_table(self):
        ok = {"valid": True, "passes": True}
        kwargs = dict(
            feasible=True,
            deadline_stop=False,
            failure=None,
            audit_passed=True,
            decision=ok,
            decisions_agree=True,
        )
        assert sr.classify_outcome(**kwargs) == "STRICT_PASS"
        assert sr.classify_outcome(**{**kwargs, "decision": {"valid": True}}) == "STRICT_FAIL"
        assert sr.classify_outcome(**{**kwargs, "feasible": False}) == "STOP_INFEASIBLE"
        assert sr.classify_outcome(**{**kwargs, "deadline_stop": True}) == "INCOMPLETE"
        for bad in ({"failure": "x"}, {"audit_passed": False}, {"decisions_agree": False}):
            assert sr.classify_outcome(**{**kwargs, **bad}) == "INVALID_STOP"
        assert sr.classify_outcome(**{**kwargs, "decision": {"valid": False}}) == "INVALID_STOP"
        assert sr.producer_claim({"decision": ok}) == {"outcome": "STRICT_PASS"}
        assert sr.producer_claim({})["outcome"] == "INVALID_STOP"


# ---------------------------------------------------------------- supervision and slots


def py(code: str) -> list:
    return [sys.executable, "-c", code]


class TestSupervision:
    def run(self, tmp_path, commands, **kwargs):
        kwargs.setdefault("poll_seconds", 0.05)
        kwargs.setdefault("grace_seconds", 2.0)
        return sr.supervise_children(
            commands,
            cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            logs=[tmp_path / f"c{i}.log" for i in range(len(commands))],
            heartbeats=kwargs.pop("heartbeats", [None] * len(commands)),
            **kwargs,
        )

    def test_two_clean_children(self, tmp_path):
        result = self.run(
            tmp_path, [py("pass"), py("import time; time.sleep(0.3)")], wall_seconds=30
        )
        assert result["cause"] is None and sr.clean_supervision(result)

    def test_failed_child_stops_sibling(self, tmp_path):
        began = time.monotonic()
        result = self.run(
            tmp_path,
            [py("raise SystemExit(3)"), py("import time; time.sleep(60)")],
            wall_seconds=60,
        )
        assert result["cause"] == "child_failed" and time.monotonic() - began < 20
        assert result["children"][1]["termination"] in ("terminated", "killed")
        assert not sr.clean_supervision(result)

    def test_wall_rss_and_heartbeat_watchdogs(self, tmp_path):
        sleeper = py("import time; time.sleep(60)")
        for name in ("a", "b", "c"):
            (tmp_path / name).mkdir()
        assert self.run(tmp_path / "a", [sleeper], wall_seconds=0.5)["cause"] == "wall_timeout"
        assert (
            self.run(tmp_path / "b", [sleeper], wall_seconds=30, rss_limit_bytes=1)["cause"]
            == "rss_limit"
        )
        stale = self.run(
            tmp_path / "c",
            [sleeper],
            wall_seconds=30,
            heartbeats=[tmp_path / "c" / "never.json"],
            heartbeat_stale_seconds=0.5,
        )
        assert stale["cause"] == "heartbeat_stale"

    def test_run_slots_are_all_or_nothing_and_released(self, tmp_path):
        busy = (tmp_path / "cpu-slot-2.lock").open("a+")
        fcntl.flock(busy.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(sr.StrictRunError, match="CPU slots unavailable"):
                sr.acquire_run_slots(tmp_path, timeout=0.3)
            probe = (tmp_path / "cpu-slot-1.lock").open("a+")  # slot 1 was not kept
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            probe.close()
        finally:
            busy.close()
        held = sr.acquire_run_slots(tmp_path, timeout=0.3)
        with pytest.raises(sr.StrictRunError, match="CPU slots unavailable"):
            sr.acquire_run_slots(tmp_path, timeout=0.3)
        sr.release_run_slots(held)
        sr.release_run_slots(sr.acquire_run_slots(tmp_path, timeout=0.3))

    def test_workers_share_the_parents_slot_locks(self, tmp_path):
        """Held by the parent across children; inherited fds hold it, fresh opens cannot."""
        slots = sr.acquire_run_slots(tmp_path, timeout=0.3)
        try:
            fds = [handle.fileno() for handle in slots]
            for k, fd in enumerate(fds):
                sr.assert_inherited_slot(fd, tmp_path, k + 1)
            with pytest.raises(sr.StrictRunError, match="is not cpu-slot-2.lock"):
                sr.assert_inherited_slot(fds[0], tmp_path, 2)
            flock = "import fcntl,sys; fcntl.flock({fd}, fcntl.LOCK_EX | fcntl.LOCK_NB)"
            fresh = (
                "import fcntl; h = open({path!r}, 'a+')\n"
                "try:\n fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "except BlockingIOError:\n raise SystemExit(0)\nraise SystemExit(3)"
            )
            for stage in ("calibration", "final"):  # locks persist between stages
                (tmp_path / stage).mkdir()
                result = self.run(
                    tmp_path / stage,
                    [py(flock.format(fd=fd)) for fd in fds],
                    wall_seconds=30,
                    pass_fds=[(fd,) for fd in fds],
                )
                assert sr.clean_supervision(result), result
            (tmp_path / "outsider").mkdir()
            outsider = self.run(
                tmp_path / "outsider",
                [py(fresh.format(path=str(tmp_path / f"cpu-slot-{k}.lock"))) for k in (1, 2)],
                wall_seconds=30,
            )
            assert sr.clean_supervision(outsider), outsider  # another job cannot take a slot
            (tmp_path / "no-fd").mkdir()
            missing = self.run(tmp_path / "no-fd", [py(flock.format(fd=fds[0]))], wall_seconds=30)
            assert missing["children"][0]["returncode"] != 0  # fds are not leaked by default
        finally:
            sr.release_run_slots(slots)


# ---------------------------------------------------------------- intent


@pytest.fixture(autouse=True)
def tier2_root(tmp_path, monkeypatch):
    """The pre-registered Tier-2 root is replaced by a tmp root in every test."""
    monkeypatch.setattr(sr, "TIER2_OUT_ROOT", tmp_path / "run-v1")
    return tmp_path / "run-v1"


def real_intent(tmp_path: Path, **overrides) -> dict:
    kwargs = dict(
        out_root=tmp_path / "run-v1",
        deadline=datetime.now(timezone.utc) + timedelta(hours=15),
        authorization_quote="i agree, continue",
        slot_lock_root=tmp_path / "locks",
        allow_dirty_source=True,
    )
    kwargs.update(overrides)
    return sr.build_intent(**kwargs)


NEEDS_REAL = pytest.mark.skipif(
    not (HAVE_PILOT and HAVE_SCREEN and HAVE_CHECKPOINTS),
    reason="strict pilot, Tier-1 screen or checkpoints not available",
)


class TestIntent:
    @NEEDS_REAL
    def test_real_intent_binds_design_and_frozen_n(self, tmp_path):
        intent = real_intent(tmp_path)
        assert intent["sizing"]["final_worlds_per_mix"] == 235
        assert intent["sizing"]["pilot_sizing"]["required_final_worlds"] == 235
        assert intent["final_seeds"] == sr.namespace_seeds()["final"][:235]
        assert intent["namespace_report"]["disjoint"] is True
        assert intent["roster_parity"]["compared"] == 48
        candidate = intent["candidate"]
        assert candidate["checkpoint_sha256"] == sr.CHAMPION[1]
        assert candidate["wrapper"] == "free-space-veto/v2-speed-preserving"
        assert candidate["wrapper_source_sha256"] == dev_screen.sha256_file(sr.WRAPPER_SOURCE)
        assert intent["incumbent"]["wrapper"] is None
        assert intent["caps"]["stage_seconds"] == {
            "calibration": 1800,
            "final": 45000,
            "serving": 3600,
            "audit": 900,
        }
        projection = intent["caps"]["final_runtime_projection"]
        assert projection["within_limit"] is True and projection["concurrency_measured"] is False
        assert "episode_wall_seconds" not in intent["pilot"]
        assert intent["serving_stage_exercises_web_path"] is False
        assert intent["serving_stage_kind"] == sr.SERVING_STAGE_KIND
        assert any("single wrapped hero" in claim for claim in intent["non_claims"])
        assert str(sr.INDEPENDENT_AUDIT) in intent["audit"]["independent_command"]
        assert str(Path(__file__).resolve().parents[1] / "src/evaluation/safety_veto.py") in (
            intent["source_closure"]["files"]
        )
        sr.validate_intent(intent)
        path = sr.prepare(intent)
        assert path == tmp_path / "run-v1" / "intent.json"
        with pytest.raises(sr.StrictRunError, match="already exists"):
            sr.prepare(intent)

    @NEEDS_REAL
    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda i: i["caps"]["stage_seconds"].update(final=40000), "stage caps"),
            (lambda i: i["caps"].update(retry_authorized=True), "retry"),
            (lambda i: i["candidate"]["wrapper_identity"].update(method="x"), "wrapper"),
            (lambda i: i["protocol"].update(sha256="0" * 64), "protocol"),
            (lambda i: i["source_closure"]["files"].update({"/nonexistent.py": "0" * 64}), "drift"),
            (lambda i: i.update(output_root="/tmp/elsewhere/run-v1"), "pre-registered root"),
            (lambda i: i.update(smoke_frames=40), "smoke dry-run may not use"),
        ],
    )
    def test_validate_intent_rejects(self, tmp_path, mutate, message):
        intent = real_intent(tmp_path)
        mutate(intent)
        with pytest.raises(sr.StrictRunError, match=message):
            sr.validate_intent(intent)

    def test_construction_failures(self, tmp_path, monkeypatch):
        with pytest.raises(sr.StrictRunError, match="deadline"):
            real_intent(tmp_path, deadline=datetime.now(timezone.utc) + timedelta(hours=2))
        with pytest.raises(sr.StrictRunError, match="forbidden"):
            real_intent(tmp_path, out_root=tmp_path / "ongoing-research-20260913" / "x")
        with pytest.raises(sr.StrictRunError, match="authorization"):
            real_intent(tmp_path, authorization_quote="  ")
        with pytest.raises(sr.StrictRunError, match="pre-registered root"):
            real_intent(tmp_path, out_root=tmp_path / "elsewhere")
        for smoke_root in (tmp_path / "run-v1", tmp_path / "other" / "run-v2"):
            with pytest.raises(sr.StrictRunError, match="smoke dry-run may not use"):
                real_intent(tmp_path, out_root=smoke_root, smoke_frames=40)
        monkeypatch.setattr(
            sr, "source_closure", lambda extra=(): {"dirty": [" M src/x.py"], "files": {}}
        )
        with pytest.raises(sr.StrictRunError, match="dirty source"):
            real_intent(tmp_path, allow_dirty_source=False)


def test_output_root_policy_and_cli_defaults(monkeypatch):
    """Smoke defaults to smoke-v1 and never run-v1; a real intent only uses run-v1."""
    monkeypatch.setattr(sr, "TIER2_OUT_ROOT", sr.DEFAULT_OUT_ROOT)
    assert sr.DEFAULT_SMOKE_ROOT.parent == sr.DEFAULT_OUT_ROOT.parent == sr.STUDY_ARTIFACT_ROOT
    sr.check_output_root(sr.DEFAULT_OUT_ROOT, smoke=False)
    sr.check_output_root(sr.DEFAULT_SMOKE_ROOT, smoke=True)
    with pytest.raises(sr.StrictRunError, match="smoke dry-run may not use"):
        sr.check_output_root(sr.DEFAULT_OUT_ROOT, smoke=True)
    for other in (sr.DEFAULT_SMOKE_ROOT, sr.STUDY_ARTIFACT_ROOT / "run-v2"):
        with pytest.raises(sr.StrictRunError, match="pre-registered root"):
            sr.check_output_root(other, smoke=False)
    seen = []

    def capture(**kwargs):
        seen.append(kwargs)
        raise sr.StrictRunError("captured")

    monkeypatch.setattr(sr, "build_intent", capture)
    base = ["prepare", "--deadline-utc", "2026-09-28T00:00:00+00:00"]
    base += ["--authorization-quote", "q"]
    for extra, expected in (
        (["--smoke-frames", "40"], sr.DEFAULT_SMOKE_ROOT),
        ([], sr.DEFAULT_OUT_ROOT),
    ):
        with pytest.raises(sr.StrictRunError, match="captured"):
            sr.main(base + extra)
        assert seen[-1]["out_root"] == expected


# ---------------------------------------------------------------- end-to-end dry run


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="champion checkpoints not available")
def test_smoke_dry_run_end_to_end(tmp_path):
    """prepare -> run (2 workers, slot locks, 2 x 40-frame episodes) -> self-check -> closeout."""
    intent = sr.build_intent(
        out_root=tmp_path / "smoke",
        deadline=datetime.now(timezone.utc) + timedelta(hours=1),
        authorization_quote="test smoke",
        slot_lock_root=tmp_path / "locks",
        python=Path(sys.executable),
        smoke_frames=40,
    )
    (tmp_path / "locks").mkdir()
    path = sr.prepare(intent)
    environment = dict(os.environ)
    try:
        closeout = sr.run(path)
    finally:  # run_stages exports the supervisor's thread limits into os.environ
        os.environ.clear()
        os.environ.update(environment)
    assert closeout["outcome"] == "SMOKE_NO_DECISION", closeout
    audit = json.loads((tmp_path / "smoke" / "output" / "audit" / "audit.json").read_text())
    assert audit["status"] == "PASS" and audit["mode"] == "smoke", audit.get("failures")
    assert closeout["audit_passed"] is True and closeout["failure"] is None
    output = tmp_path / "smoke" / "output"
    records = sorted(p.name for p in (output / "final" / "records").glob("*.json"))
    assert len(records) == 2 and all(n.startswith("final-") for n in records)
    assert json.loads((output / "self-check" / "report.json").read_text())["passed"] is True
    assert (output / "closeout.json").is_file() and not (output / "receipt.json").exists()
    with pytest.raises(FileExistsError):
        sr.run(path)  # create-only: never resumes or retries
