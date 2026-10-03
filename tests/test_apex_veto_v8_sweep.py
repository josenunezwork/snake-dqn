"""Tests for the v8 DEV lambda sweep + 3-slot calibration harness (never plays a game)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_v7_screen_20261002 import screen as v7screen
from research.apex_veto_v8_lambda_sweep_20261002 import sweep

COMMIT = "d" * 40
SLOTS = ("cpu-slot-1.lock", "cpu-slot-2.lock", "cpu-slot-3.lock")


@pytest.fixture(autouse=True)
def no_episodes(monkeypatch):
    """Every episode runner is fatal in every test of this file."""
    from src.scripts import tournament_eval

    def fatal(*args, **kwargs):
        raise AssertionError("a sweep test must never play an episode")

    monkeypatch.setattr(tournament_eval, "rollout", fatal)
    monkeypatch.setattr(dev_screen, "run_episode", fatal)


def _soon(hours=1.0):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _entry(arm, mix, seed, mass, changes=0, decisions=1000):
    entry = {
        "arm": arm,
        "mix": mix,
        "world_seed": seed,
        "wall_seconds": 30.0,
        "record": {"mass_integral": mass, "probes": {"death_cause": "self"}},
    }
    if arm != "A":
        entry["veto_diagnostics"] = {
            "action_differs_from_reference": changes,
            "decisions": decisions,
            "head_risky_vetoes": changes,
            "v7": {"rerank_changes": 1, "decisions": decisions},
        }
    return entry


# ---------------------------------------------------------------- namespaces, arms, plan


class TestNamespacesAndArms:
    def test_worlds_are_fresh_against_every_earlier_bank(self):
        earlier = set(sweep.EARLIER_DOMAINS)
        assert set(v7screen.EARLIER_DOMAINS) <= earlier
        assert set(sweep.V7_STRICT_DOMAINS) | set(sweep.SCREEN_DOMAINS) <= earlier
        assert "apex-veto-v7-dev-v1" in earlier and "apex-veto-v7-screen-v1" in earlier
        assert "worlds" in sweep.EARLIER_DOMAINS["apex-veto-web-serving-v1"]
        assert sweep.DOMAIN == "apex-veto-v8-dev-v1"
        for spec, own in ((sweep.SPEC, sweep.DOMAIN), (sweep.SMOKE_SPEC, sweep.SMOKE_DOMAIN)):
            assert not any(key.startswith(f"{own}/") for key in spec.extra_namespaces)
            seeds = set(dev_screen.screen_seeds(8, own, sweep.NAMESPACE))
            for name, banned in spec.extra_namespaces.items():
                assert not seeds & set(banned), name
        assert spec.require_slot_locks and spec.require_ac_power
        assert spec.max_wall_seconds == 3 * 3600

    def test_arms_installers_and_identities(self):
        from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
        from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto

        assert sweep.INSTALLERS["R"] is sweep.INSTALLERS["H800"]
        assert sweep.SPEC.arm_vetoes["R"] is sweep.SPEC.arm_vetoes["H800"]
        assert sweep.arm_method("A") == "free-space-veto/v7-space-preference(lambda=4.0)"
        assert sweep.arm_method("H1600") == "free-space-veto/v8-space-and-head(lambda=16.0)"
        assert sweep.arm_method("R") == sweep.arm_method("H800")
        assert sweep.expected_descriptor("A") == SpacePreferenceVeto(4.0).descriptor()
        assert sweep.expected_descriptor("H400") == SpaceAndHeadVeto(4.0).descriptor()
        assert sweep.INSTALLERS["H800"].keywords == {"lam": 8.0}

    def test_installed_v8_carries_the_arm_a_reference(self, setup_config):
        from src.game.ai_snake import AISnake

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=SimpleNamespace())
        veto = sweep.INSTALLERS["H1600"](snake)
        assert snake.safety_veto is veto and veto.lam == 16.0 and veto.reference_lambda == 4.0
        assert veto.head_avoidance is True

    def test_shard_plan_covers_every_episode_once_and_replays_cross_shard(self):
        seeds = list(range(100, 108))
        rows = dev_screen._design_rows(seeds)
        plans = {k: sweep.shard_plan(rows, seeds, dev_screen.MIXES, k, 3, False) for k in range(3)}
        keys = [tuple(key) for k in range(3) for key in sweep.plan_keys(plans[k])]
        assert len(keys) == len(set(keys)) == 99 and all(len(plans[k]) == 33 for k in range(3))
        owner = {key: k for k in range(3) for key in map(tuple, sweep.plan_keys(plans[k]))}
        for row in rows:
            homes = {owner[(arm, row["mix"], row["world_seed"])] for arm in sweep.SWEEP_ARMS}
            assert len(homes) == 1  # every arm of a world-mix pair in one shard
        for mix in dev_screen.MIXES:
            replay = [key for key in owner if key[0] == "R" and key[1] == mix]
            assert len(replay) == 1 and replay[0][2] == seeds[0]
            assert owner[replay[0]] != owner[("H800", mix, seeds[0])]
        for k in range(3):
            assert [arm for arm, _ in plans[k]][-1] == "R"
            assert [arm for arm, _ in plans[k][:4]] == list(sweep.SWEEP_ARMS)
        smoke = sweep.shard_plan(rows[:1], seeds[:1], ("frozen",), 0, 1, True)
        assert [arm for arm, _ in smoke] == ["A", "H400"]


# ---------------------------------------------------------------- selection rule


def _per_seed(value, seeds):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * len(seeds)


def _flat(frozen, scripted, mixed):
    return {"frozen": frozen, "scripted": scripted, "mixed": mixed}


def _sweep_entries(deltas, seeds=(1, 2), changes=None, decisions=1000):
    changes = changes or {}
    entries = []
    for mix in dev_screen.MIXES:
        for index, seed in enumerate(seeds):
            entries.append(_entry("A", mix, seed, 100.0))
            for arm in sweep.V8_ARMS:
                delta = _per_seed(deltas[arm][mix], seeds)[index]
                count = _per_seed(changes.get(arm, 3), seeds)[index]
                entries.append(_entry(arm, mix, seed, 100.0 + delta, count, decisions))
    for mix in dev_screen.MIXES:
        twin = next(e for e in entries if e["arm"] == "H800" and e["mix"] == mix)
        entries.append(dict(twin, arm="R"))
    return entries


class TestSelection:
    def _summary(self, entries, monkeypatch, smoke=False, checks_pass=True, planned=None):
        monkeypatch.setattr(
            sweep, "self_check", lambda *a, **k: {"passes": checks_pass, "failures": []}
        )
        return sweep.summarize(
            entries,
            [1, 2],
            dev_screen.MIXES,
            smoke=smoke,
            planned_episodes=len(entries) if planned is None else planned,
            replay_planned=3,
            source={"commit": COMMIT, "dirty_paths": ""},
        )

    def test_highest_pooled_qualifying_lambda_is_selected(self, monkeypatch):
        deltas = {
            "H400": _flat(5, 5, 5),
            "H800": _flat(40, -11, 40),  # best pooled but a clear loser in scripted (sd 0)
            "H1600": _flat(20, (-30, 10), 12),  # scripted mean -10, noisy: not a loser
        }
        summary = self._summary(_sweep_entries(deltas), monkeypatch)
        selection = summary["selection"]
        assert selection["status"] == "SELECTED" and selection["passes"] is True
        assert selection["selected_lambda"] == 16.0
        assert selection["qualifying_lambdas"] == [4.0, 16.0]
        row = summary["lambdas"]["H800"]
        assert row["active"] and row["clear_loser_in_some_mix"] and not row["qualifies"]
        assert summary["lambdas"]["H400"]["changes_vs_A"] == 18

    def test_one_active_episode_per_mix_suffices(self, monkeypatch):
        deltas = {arm: _flat((5, 0), (5, 0), (5, 0)) for arm in sweep.V8_ARMS}
        entries = _sweep_entries(deltas, changes={arm: (2, 0) for arm in sweep.V8_ARMS})
        selection = self._summary(entries, monkeypatch)["selection"]
        assert selection["status"] == "SELECTED"
        # Ties go to the SMALLER lambda (closest to the strict-gated arm A).
        assert selection["selected_lambda"] == 4.0

    @pytest.mark.parametrize(
        "changes, deltas",
        [({}, 0), ({arm: 0 for arm in sweep.V8_ARMS}, 5)],
    )
    def test_no_active_arm_stops(self, monkeypatch, changes, deltas):
        entries = _sweep_entries(
            {a: _flat(deltas, deltas, deltas) for a in sweep.V8_ARMS}, changes=changes
        )
        selection = self._summary(entries, monkeypatch)["selection"]
        assert selection["status"] == "NONE_ACTIVE" and selection["selected_lambda"] is None

    def test_rate_floor_and_none_qualifies(self, monkeypatch):
        deltas = {arm: _flat(5, 5, 5) for arm in sweep.V8_ARMS}
        rare = _sweep_entries(deltas, changes={a: 1 for a in sweep.V8_ARMS}, decisions=10**5)
        assert self._summary(rare, monkeypatch)["selection"]["status"] == "NONE_ACTIVE"
        losers = {arm: _flat(50, -20, 50) for arm in sweep.V8_ARMS}
        selection = self._summary(_sweep_entries(losers), monkeypatch)["selection"]
        assert selection["status"] == "NONE_QUALIFIES" and selection["passes"] is False

    def test_earlier_statuses_take_precedence(self, monkeypatch):
        entries = _sweep_entries({arm: _flat(5, 5, 5) for arm in sweep.V8_ARMS})
        smoke = self._summary(entries, monkeypatch, smoke=True)
        assert smoke["selection"]["status"] == "SMOKE_NO_SELECTION"
        bad = self._summary(entries, monkeypatch, checks_pass=False)
        assert bad["selection"]["status"] == "INVALID_SELF_CHECK_FAILED"
        short = self._summary(entries[1:], monkeypatch, planned=len(entries))
        assert short["selection"]["status"] == "INCOMPLETE"
        drift = [dict(e) for e in entries]
        drift[-1] = dict(drift[-1], record={"mass_integral": 1.0, "probes": {}})
        assert self._summary(drift, monkeypatch)["selection"]["status"] == (
            "INVALID_NONDETERMINISTIC"
        )


# ---------------------------------------------------------------- self-check on real probes


def _probe_entry(arm, veto, seed=5):
    """A sweep entry built from a real veto's record (no rollout: smoke-mode fields only)."""
    diagnostics = veto.diagnostics_record() if hasattr(veto, "diagnostics_record") else None
    return {
        "schema_version": sweep.SCHEMA,
        "authority": sweep.AUTHORITY,
        "screen": sweep.SWEEP_ID,
        "arm": arm,
        "mix": "scripted",
        "world_seed": seed,
        "hero_sha256": dev_screen.CHAMPION[1],
        "safety_veto": True,
        "safety_veto_method": veto.record()["method"],
        "veto_diagnostics": diagnostics,
        "record": {"seed": seed, "mass_integral": 12.5, "probes": {"safety_veto": veto.record()}},
    }


class TestCheckEntry:
    def test_real_v7_and_v8_probes_pass_and_mismatches_fail(self, setup_config):
        from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
        from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto
        from tests.test_safety_veto_v5 import random_states

        states = random_states(3, 20)
        arm_a = SpacePreferenceVeto(4.0)
        h800 = SpaceAndHeadVeto(8.0, reference_lambda=4.0)
        for me, roster, q, mask, base in states:
            arm_a.apply(me, roster, q, mask, base)
            h800.apply(me, roster, q, mask, base)
        ok = {"failures": [], "warnings": []}
        assert sweep.check_entry(_probe_entry("A", arm_a), smoke=True) == ok
        assert sweep.check_entry(_probe_entry("H800", h800), smoke=True) == ok
        assert sweep.check_entry(_probe_entry("R", h800), smoke=True) == ok
        assert sweep.check_entry(_probe_entry("H1600", h800), smoke=True)["failures"]
        assert sweep.check_entry(_probe_entry("A", h800), smoke=True)["failures"]
        no_reference = SpaceAndHeadVeto(8.0)
        for me, roster, q, mask, base in states[:3]:
            no_reference.apply(me, roster, q, mask, base)
        failures = sweep.check_entry(_probe_entry("H800", no_reference), smoke=True)["failures"]
        assert any("reference diagnostic" in message for message in failures)
        full = sweep.check_entry(_probe_entry("H800", h800), smoke=False)["failures"]
        assert any("H5000" in message for message in full)
        assert sweep.v8_identities_hold(h800.diagnostics_record(), h800.counters.to_dict())

    def test_unknown_arm_and_duplicates_fail_the_self_check(self):
        result = sweep.self_check([_entry("Z", "frozen", 1, 1.0)], [1], smoke=True)
        assert not result["passes"] and "unknown arm" in result["failures"][0]
        twice = sweep.self_check([_entry("Z", "frozen", 1, 1.0)] * 2, [1], smoke=True)
        assert any("duplicate" in message for message in twice["failures"])


# ---------------------------------------------------------------- calibration


def _check(ok=True, reasons=(), ratio=None):
    keys = {"shard": {"ratio": ratio}} if ratio is not None else {}
    return {
        "event": "thermal_guard_check",
        "ok": ok,
        "reasons": list(reasons),
        "readings": {"slowdown": {"keys": keys}},
    }


def _shard_row(shard, guard_events=(), start="2026-10-03T00:00:00+00:00", end=None):
    return {
        "shard": shard,
        "slot_held": SLOTS[shard],
        "started_utc": start,
        "finished_utc": end or "2026-10-03T00:30:00+00:00",
        "guard": sweep.guard_stats(list(guard_events)),
        "start_barrier_passed": True,
        "arm_a_finished_utc": ["2026-10-03T00:10:00+00:00"],
    }


def _mix_entries(seconds_by_mix):
    return [
        dict(_entry("A", mix, i, 1.0), wall_seconds=seconds)
        for mix, seconds in seconds_by_mix.items()
        for i in range(8)
    ]


def _a_entries(seconds):
    return [
        dict(_entry("A", mix, i, 1.0), wall_seconds=seconds)
        for mix in dev_screen.MIXES
        for i in range(8)
    ]


BASELINE = {"ok": True, "times_by_mix": {mix: [30.0] * 8 for mix in dev_screen.MIXES}}


class TestCalibration:
    def test_guard_stats_classify_reasons(self):
        events = [
            _check(),
            _check(False, ["slowdown:shard"], ratio=1.7),
            {"event": "thermal_guard_backoff", "reasons": ["slowdown:shard"]},
            _check(False, ["CPU_Speed_Limit=80<100"]),
            {"event": "thermal_guard_backoff", "reasons": ["CPU_Speed_Limit=80<100"]},
            {"event": "thermal_guard_stop", "reason": "thermal: still not ok"},
            {"arm": "A", "wall_seconds": 3.0},
        ]
        stats = sweep.guard_stats(events)
        assert stats["checks"] == 3 and stats["not_ok_checks"] == 2
        assert stats["thermal_not_ok_checks"] == 1 and stats["slowdown_pauses"] == 1
        assert stats["thermal_pauses"] == 1 and stats["stops"] == 1
        assert stats["max_slowdown_ratio"] == 1.7
        assert stats["reason_counts"] == {"CPU_Speed_Limit": 1, "slowdown": 1}

    def test_pass(self):
        shards = [_shard_row(k, [_check(ratio=1.1)]) for k in range(3)]
        report = sweep.calibration_report(shards, _a_entries(36.0), BASELINE, complete=True)
        assert report["status"] == "CALIBRATION_PASS", report["problems"]
        assert report["wall_ratio_A_vs_baseline"] == pytest.approx(1.2)
        assert report["max_in_run_slowdown_ratio"] == 1.1
        assert report["shard_overlap_seconds"] == 1800
        assert report["arm_a_episodes_outside_concurrent_window"] == 0
        assert report["criteria"]["wall_ratio_below_max_in_every_mix"]

    def test_slowdown_pauses_are_reported_only(self):
        pauses = [{"event": "thermal_guard_backoff", "reasons": ["slowdown:shard"]}] * 3
        shards = [_shard_row(0, pauses), _shard_row(1), _shard_row(2)]
        report = sweep.calibration_report(shards, _a_entries(30.0), BASELINE, complete=True)
        assert report["status"] == "CALIBRATION_PASS", report["problems"]
        assert report["reported_only"]["slowdown_pauses"] == 3
        assert "slowdown_pauses_at_most_max" not in report["criteria"]

    def test_one_slow_mix_fails_although_the_pooled_ratio_passes(self):
        entries = _mix_entries({"frozen": 31.5, "mixed": 36.0, "scripted": 43.5})
        shards = [_shard_row(k) for k in range(3)]
        report = sweep.calibration_report(shards, entries, BASELINE, complete=True)
        assert report["wall_ratio_A_vs_baseline"] == pytest.approx(37.0 / 30.0)
        assert report["criteria"]["wall_ratio_below_max"]
        assert not report["criteria"]["wall_ratio_below_max_in_every_mix"]
        assert report["wall_per_mix"]["scripted"]["ratio"] == pytest.approx(1.45)
        assert report["status"] == "CALIBRATION_FAIL"

    def test_the_wall_bound_is_strict(self):
        shards = [_shard_row(k) for k in range(3)]
        report = sweep.calibration_report(shards, _a_entries(39.0), BASELINE, complete=True)
        assert report["wall_ratio_A_vs_baseline"] == pytest.approx(1.30)
        assert report["status"] == "CALIBRATION_FAIL"
        missing_mix = _mix_entries({"frozen": 30.0, "mixed": 30.0})
        report = sweep.calibration_report(shards, missing_mix, BASELINE, complete=True)
        assert report["status"] == "CALIBRATION_FAIL"  # scripted has no A ratio

    @pytest.mark.parametrize(
        "events, seconds",
        [
            ([_check(False, ["thermal_warning_level=1"])], 30.0),
            ([_check(False, ["on_battery"])], 30.0),
            ([{"event": "thermal_guard_check", "reasons": ["pmset_unavailable"]}], 30.0),
            ([], 40.0),  # 1.33 x the baseline
        ],
    )
    def test_fail(self, events, seconds):
        shards = [_shard_row(0, events), _shard_row(1), _shard_row(2)]
        report = sweep.calibration_report(shards, _a_entries(seconds), BASELINE, complete=True)
        assert report["status"] == "CALIBRATION_FAIL"

    def test_a_thermal_stop_fails_even_when_incomplete(self):
        stop = {"event": "thermal_guard_stop", "reason": "thermal: persistent slowdown"}
        shards = [_shard_row(0, [stop]), _shard_row(1), _shard_row(2)]
        report = sweep.calibration_report(shards, _a_entries(30.0), BASELINE, complete=False)
        assert report["status"] == "CALIBRATION_FAIL"

    def test_invalid(self):
        good = [_shard_row(k) for k in range(3)]
        entries = _a_entries(30.0)
        cases = [
            (good[:2], BASELINE, True),
            ([*good[:2], dict(good[2], slot_held=SLOTS[0])], BASELINE, True),
            (
                [
                    *good[:2],
                    _shard_row(
                        2, start="2026-10-03T01:00:00+00:00", end="2026-10-03T01:30:00+00:00"
                    ),
                ],
                BASELINE,
                True,
            ),
            (
                [*good[:2], _shard_row(2, start="2026-10-03T00:29:59+00:00")],  # 1 s overlap
                BASELINE,
                True,
            ),
            (
                [  # 25 min overlap < 0.9 x the 30 min shortest shard
                    *good[:2],
                    _shard_row(
                        2, start="2026-10-03T00:05:00+00:00", end="2026-10-03T00:40:00+00:00"
                    ),
                ],
                BASELINE,
                True,
            ),
            ([*good[:2], dict(good[2], start_barrier_passed=None)], BASELINE, True),
            (good, {"ok": False, "reason": "sha", "times_by_mix": {}}, True),
            (good, BASELINE, False),  # a deadline stop
        ]
        for shards, baseline, complete in cases:
            report = sweep.calibration_report(shards, entries, baseline, complete)
            assert report["status"] == "CALIBRATION_INVALID", report

    def test_baseline_is_pinned(self, tmp_path):
        fake = tmp_path / "events.jsonl"
        fake.write_text(json.dumps({"arm": "L400", "mix": "frozen", "wall_seconds": 3.0}) + "\n")
        loaded = sweep.load_baseline(fake)
        assert not loaded["ok"] and loaded["times_by_mix"] == {"frozen": [3.0]}
        assert not sweep.load_baseline(tmp_path / "missing.jsonl")["ok"]
        if sweep.BASELINE_EVENTS.is_file():  # the real pinned file on the research host
            real = sweep.load_baseline(sweep.BASELINE_EVENTS)
            assert real["ok"] and sorted(map(len, real["times_by_mix"].values())) == [8, 8, 8]


# ---------------------------------------------------------------- run guards (never play)

FLAGS = ("--use-slot-locks", "--slot-pool", "3", "--thermal-guard", "--require-ac-power")


class TestRunGuards:
    def _argv(self, tmp_path, *extra, hours=1.0, flags=FLAGS, shard="0"):
        return [
            "run",
            "--shard",
            shard,
            "--out",
            str(tmp_path / "o"),
            "--deadline-utc",
            _soon(hours),
            "--slot-lock-root",
            str(tmp_path / "locks"),
            *flags,
            *extra,
        ]

    @pytest.fixture(autouse=True)
    def no_locks(self, monkeypatch):
        def fatal(*args, **kwargs):
            raise AssertionError("a refused run must not reach the slot lock")

        monkeypatch.setattr(dev_screen, "acquire_cpu_slots", fatal)
        monkeypatch.setattr(dev_screen, "_git_state", lambda: {"commit": COMMIT, "dirty_paths": ""})

    @pytest.mark.parametrize(
        "flags",
        [
            (),
            ("--use-slot-locks", "--thermal-guard"),  # pool 2
            ("--use-slot-locks", "--slot-pool", "3"),  # no guard
            ("--slot-pool", "3", "--thermal-guard"),  # no locks
        ],
    )
    def test_slot_pool_3_and_the_guard_are_required(self, tmp_path, flags):
        assert sweep.main(self._argv(tmp_path, flags=flags)) == 2
        assert not (tmp_path / "o").exists()

    @pytest.mark.parametrize(
        "extra, shard",
        [
            (("--worlds-per-mix", "4"), "0"),
            (("--shards", "2"), "0"),
            ((), "3"),
            (("--smoke-frames", "500", "--worlds-per-mix", "1"), "0"),  # smoke needs 1 shard
        ],
    )
    def test_size_and_shard_guards(self, tmp_path, extra, shard):
        assert sweep.main(self._argv(tmp_path, *extra, shard=shard)) == 2
        assert not (tmp_path / "o").exists()

    def test_wall_cap_deadline_existing_out_and_dirty_tree(self, tmp_path, monkeypatch):
        assert sweep.main(self._argv(tmp_path, hours=3.1)) == 2
        assert sweep.main(self._argv(tmp_path, hours=-0.1)) == 2
        (tmp_path / "o").mkdir()
        assert sweep.main(self._argv(tmp_path)) == 2
        assert not any((tmp_path / "o").iterdir())
        (tmp_path / "o").rmdir()
        dirty = {"commit": COMMIT, "dirty_paths": " M research/x.py\n?? notes.txt"}
        monkeypatch.setattr(dev_screen, "_git_state", lambda: dirty)
        assert sweep.main(self._argv(tmp_path)) == 2
        assert not (tmp_path / "o").exists()

    def test_smoke_size_guard(self, tmp_path):
        smoke = ("--shards", "1")
        for extra in (
            ("--smoke-frames", "501", "--worlds-per-mix", "1"),
            ("--smoke-frames", "500", "--worlds-per-mix", "2"),
            ("--smoke-frames", "500", "--worlds-per-mix", "1", "--smoke-mixes", "frozen,mixed"),
        ):
            with pytest.raises(SystemExit):
                sweep.main(self._argv(tmp_path, *smoke, *extra))
        assert not (tmp_path / "o").exists()

    def test_missing_third_slot_refuses_before_any_output(
        self, setup_config, tmp_path, monkeypatch, capsys
    ):
        # Every refusal passes (smoke), preflight runs; the tmp lock root has no slot files.
        monkeypatch.undo()
        from src.scripts import tournament_eval

        def fatal(*args, **kwargs):
            raise AssertionError("a sweep test must never play an episode")

        monkeypatch.setattr(tournament_eval, "rollout", fatal)
        monkeypatch.setattr(dev_screen, "run_episode", fatal)
        monkeypatch.setattr(dev_screen, "_configure_torch", lambda: None)
        monkeypatch.setattr(dev_screen, "on_ac_power", lambda: True)
        (tmp_path / "locks").mkdir()
        argv = self._argv(
            tmp_path, "--shards", "1", "--smoke-frames", "500", "--worlds-per-mix", "1"
        )
        assert sweep.main(argv) == 2
        assert "cpu-slot-3.lock" in capsys.readouterr().err  # reached the pool-3 lock step
        assert not (tmp_path / "o").exists()
        assert not any((tmp_path / "locks").iterdir())  # never created


# ---------------------------------------------------------------- episode loop


def _fake_runner(calls):
    def run(arm, row, world_index, lookup, profile, records_dir, smoke_frames=None, spec=None):
        calls.append((arm, row["mix"], row["world_seed"]))
        mass = 100.0 + len(calls)
        return {"arm": arm, "wall_seconds": 2.0, "record": {"mass_integral": mass}}

    return run


class TestEpisodeLoop:
    def _args(self, hours=1.0):
        return SimpleNamespace(
            deadline_utc=datetime.now(timezone.utc) + timedelta(hours=hours),
            min_episode_budget_seconds=1.0,
            thermal_backoff_seconds=1.0,
            thermal_max_backoffs=1,
            smoke_frames=None,
            shard=1,
        )

    def _plan(self):
        return [(arm, {"mix": "frozen", "world_seed": 7}) for arm in sweep.SWEEP_ARMS]

    def test_guard_admits_and_keys_the_whole_shard(self, tmp_path, monkeypatch):
        from research.compute.thermal_guard import ThermalGuard, admit_next_episode

        monkeypatch.setattr(dev_screen, "_thermal_sleep", lambda s: None)
        guard = ThermalGuard(pmset_reader=lambda: {"gated": False}, ac_reader=lambda: True)
        calls = []
        entries, stopped, events = sweep._play(
            self._args(),
            sweep.SPEC,
            self._plan(),
            [7],
            {},
            None,
            tmp_path,
            guard,
            tmp_path,
            _fake_runner(calls),
            admit_next_episode,
        )
        assert (
            stopped is None
            and len(entries) == 4
            and [c[0] for c in calls] == ["A", "H400", "H800", "H1600"]
        )
        assert guard.times == {"shard": [2.0] * 4}
        lines = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert sum(1 for x in lines if x.get("event") == "thermal_guard_check") == 4
        assert [x["arm"] for x in lines if "arm" in x] == list(sweep.SWEEP_ARMS)
        assert all(x["shard"] == 1 for x in lines if "arm" in x)
        assert sweep.guard_stats(events)["checks"] == 4

    def test_thermal_stop_plays_nothing(self, tmp_path, monkeypatch):
        from research.compute.thermal_guard import ThermalGuard, admit_next_episode

        sleeps = []
        monkeypatch.setattr(dev_screen, "_thermal_sleep", sleeps.append)
        hot = "Thermal Warning Level = 2\nNote: No CPU power status has been recorded\n"
        hot += "Note: No performance warning level has been recorded\n"
        guard = ThermalGuard(
            pmset_reader=lambda: {"gated": True, "available": True, "text": hot},
            ac_reader=lambda: True,
        )
        calls = []
        entries, stopped, events = sweep._play(
            self._args(),
            sweep.SPEC,
            self._plan(),
            [7],
            {},
            None,
            tmp_path,
            guard,
            tmp_path,
            _fake_runner(calls),
            admit_next_episode,
        )
        assert entries == [] and calls == [] and stopped.startswith("thermal: ")
        stats = sweep.guard_stats(events)
        assert stats["stops"] == 1 and stats["thermal_not_ok_checks"] == 2 and sleeps == [1.0]

    def test_deadline_budget_stops_before_the_guard(self, tmp_path):
        def never(*args, **kwargs):
            raise AssertionError("the deadline check comes first")

        entries, stopped, _ = sweep._play(
            self._args(hours=0.0001),
            sweep.SPEC,
            self._plan(),
            [7],
            {},
            None,
            tmp_path,
            None,
            tmp_path,
            _fake_runner([]),
            never,
        )
        assert entries == [] and stopped.startswith("deadline: ")


# ---------------------------------------------------------------- merge (fake shard dirs)

MASS = {"A": 100.0, "H400": 104.0, "H800": 110.0, "H1600": 90.0}


def _write_shards(root, commit=COMMIT, tamper=None):
    seeds = dev_screen.screen_seeds(8, sweep.DOMAIN, sweep.NAMESPACE)
    rows = dev_screen._design_rows(seeds)
    h800 = {}
    dirs = []
    for shard in range(3):
        plan = sweep.shard_plan(rows, seeds, dev_screen.MIXES, shard, 3, False)
        directory = root / f"shard-{shard}"
        (directory / "records").mkdir(parents=True)
        intent = {
            "schema_version": sweep.SCHEMA,
            "sweep_id": sweep.SWEEP_ID,
            "smoke_frames": None,
            "shard": shard,
            "shards": 3,
            "protocol_sha256": "p" * 64,
            "worlds": {"domain": sweep.DOMAIN, "namespace": "worlds", "seeds": seeds},
            "mixes": list(dev_screen.MIXES),
            "git": {"commit": commit if shard else COMMIT, "dirty_paths": ""},
            "plan": sweep.plan_keys(plan),
            "start_barrier": {"passed": True},
        }
        if tamper is not None and shard == 2:
            intent["plan"] = intent["plan"][:-1]
        (directory / "intent.json").write_text(json.dumps(intent))
        (directory / "shard_summary.json").write_text(
            json.dumps(
                {
                    "shard": shard,
                    "slot_held": SLOTS[shard],
                    "started_utc": "2026-10-03T00:00:00+00:00",
                    "finished_utc": "2026-10-03T00:30:00+00:00",
                    "episodes_run": len(plan),
                    "stopped_reason": None,
                }
            )
        )
        (directory / "events.jsonl").write_text(json.dumps(_check(ratio=1.05)) + "\n")
        for arm, row in plan:
            if arm == "R":
                continue
            entry = _entry(arm, row["mix"], row["world_seed"], MASS[arm], changes=2)
            if arm == "H800":
                h800[(row["mix"], row["world_seed"])] = entry
            name = f"{arm}-{row['mix']}-{row['world_seed']}.json"
            (directory / "records" / name).write_text(json.dumps(entry))
        dirs.append(directory)
    for shard, directory in enumerate(dirs):
        for key in json.loads((directory / "intent.json").read_text())["plan"]:
            if key[0] == "R":
                twin = dict(h800[(key[1], key[2])], arm="R")
                (directory / "records" / f"R-{key[1]}-{key[2]}.json").write_text(json.dumps(twin))
    return dirs


class TestMerge:
    @pytest.fixture
    def baseline(self, tmp_path, monkeypatch):
        path = tmp_path / "baseline.jsonl"
        lines = [
            {"arm": "L400", "mix": mix, "wall_seconds": 28.0}
            for mix in dev_screen.MIXES
            for _ in range(8)
        ]
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        monkeypatch.setattr(sweep, "BASELINE_EVENTS_SHA256", dev_screen.sha256_file(path))
        monkeypatch.setattr(
            sweep, "check_entry", lambda entry, smoke: {"failures": [], "warnings": []}
        )
        return path

    def _merge(self, tmp_path, dirs, baseline):
        argv = ["merge", "--shard-dirs", *map(str, dirs), "--out", str(tmp_path / "merged")]
        return sweep.main([*argv, "--baseline-events", str(baseline)])

    def test_merge_selects_and_calibrates(self, tmp_path, baseline):
        dirs = _write_shards(tmp_path)
        assert self._merge(tmp_path, dirs, baseline) == 0
        summary = json.loads((tmp_path / "merged" / "summary.json").read_text())
        assert summary["complete"] and summary["episodes_run"] == 99
        assert summary["replay_control"]["passes"] and summary["replay_control"]["compared"] == 3
        selection = summary["selection"]
        assert selection["status"] == "SELECTED" and selection["selected_lambda"] == 8.0
        assert summary["lambdas"]["H1600"]["clear_loser_in_some_mix"]
        calibration = summary["calibration"]
        assert calibration["status"] == "CALIBRATION_PASS", calibration["problems"]
        assert calibration["wall_ratio_A_vs_baseline"] == pytest.approx(30.0 / 28.0)
        assert calibration["slots_held"] == sorted(SLOTS)
        assert [row["shard"] for row in summary["shards"]] == [0, 1, 2]

    def test_merge_refusals(self, tmp_path, baseline):
        dirs = _write_shards(tmp_path / "a")
        assert self._merge(tmp_path, dirs[:2], baseline) == 2
        assert self._merge(tmp_path, [dirs[0], dirs[0], dirs[1]], baseline) == 2
        other = _write_shards(tmp_path / "b", commit="e" * 40)
        assert self._merge(tmp_path, other, baseline) == 2
        tampered = _write_shards(tmp_path / "c", tamper=True)
        assert self._merge(tmp_path, tampered, baseline) == 2
        assert self._merge(tmp_path, [tmp_path / "nope"] * 3, baseline) == 2
        assert not (tmp_path / "merged").exists()
        (tmp_path / "merged").mkdir()
        assert self._merge(tmp_path, dirs, baseline) == 2


# ---------------------------------------------------------------- start barrier, slots-free

IDENTITY = {"commit": COMMIT, "protocol_sha256": "p" * 64, "shards": 3}


def _marker(directory, shard, slot, *, pid=None, age_seconds=0.0, **identity):
    written = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    marker = {**IDENTITY, **identity, "shard": shard, "slot_held": slot}
    marker.update(pid=os.getpid() if pid is None else pid, written_utc=written.isoformat())
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"shard-{shard}.json").write_text(json.dumps(marker))


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestStartBarrier:
    def _wait(self, directory, shard=0, slot=SLOTS[2], **kwargs):
        clock = _Clock()
        return sweep.start_barrier(
            directory, shard, IDENTITY, slot, 5.0, sleep=clock.sleep, clock=clock, **kwargs
        )

    def test_passes_when_all_three_shards_hold_distinct_slots(self, tmp_path):
        _marker(tmp_path, 1, SLOTS[0])
        _marker(tmp_path, 2, SLOTS[1])
        report = self._wait(tmp_path)
        assert report["passed"] and report["problems"] == []
        own = json.loads((tmp_path / "shard-0.json").read_text())
        assert own["slot_held"] == SLOTS[2] and own["pid"] == os.getpid()
        assert own["commit"] == COMMIT and own["shards"] == 3

    @pytest.mark.parametrize(
        "sibling, expected",
        [
            ({}, "shard 2 has no readable marker"),  # a partial launch: shard 2 never started
            ({"slot": SLOTS[0]}, "not the 3 distinct pool-3 slots"),
            ({"age_seconds": 60.0}, "marker is stale"),
            ({"pid": 999999999}, "is not running"),
            ({"commit": "e" * 40}, "another commit"),
            ({"protocol_sha256": "q" * 64}, "another commit, protocol"),
        ],
    )
    def test_a_partial_or_mismatched_launch_never_passes(self, tmp_path, sibling, expected):
        _marker(tmp_path, 1, SLOTS[0])
        if sibling:
            _marker(tmp_path, 2, sibling.pop("slot", SLOTS[1]), **sibling)
        report = self._wait(tmp_path)
        assert not report["passed"] and report["waited_seconds"] >= 5.0
        assert any(expected in problem for problem in report["problems"]), report["problems"]


class TestSlotsFree:
    def test_all_free_busy_and_missing(self, tmp_path):
        import fcntl

        assert sweep.main(["slots-free", "--slot-lock-root", str(tmp_path)]) == 2  # missing
        assert not any(tmp_path.iterdir())  # never created
        for name in SLOTS:
            (tmp_path / name).write_text("")
        assert sweep.main(["slots-free", "--slot-lock-root", str(tmp_path)]) == 0
        with (tmp_path / SLOTS[0]).open("r") as held:  # e.g. the strict run on slot 1
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert sweep.main(["slots-free", "--slot-lock-root", str(tmp_path)]) == 2
        assert sweep.main(["slots-free", "--slot-lock-root", str(tmp_path)]) == 0  # released


class TestRunBarrier:
    @pytest.fixture
    def ready(self, setup_config, tmp_path, monkeypatch):
        monkeypatch.setattr(dev_screen, "_configure_torch", lambda: None)
        monkeypatch.setattr(dev_screen, "on_ac_power", lambda: True)
        monkeypatch.setattr(dev_screen, "_git_state", lambda: {"commit": COMMIT, "dirty_paths": ""})
        monkeypatch.setattr(dev_screen, "preflight_failures", lambda *a: [])
        locks = tmp_path / "locks"
        locks.mkdir()
        for name in SLOTS:
            (locks / name).write_text("")
        return [
            "run",
            "--shard",
            "0",
            "--out",
            str(tmp_path / "run" / "shard-0"),
            "--deadline-utc",
            _soon(1.0),
            "--slot-lock-root",
            str(locks),
            "--barrier-timeout-seconds",
            "0.2",
            *FLAGS,
        ]

    def test_a_lone_shard_exits_before_any_output_or_episode(
        self, ready, tmp_path, monkeypatch, capsys
    ):
        def body(*args, **kwargs):
            raise AssertionError("a refused shard must not reach the shard body")

        monkeypatch.setattr(sweep, "_run_shard", body)
        assert sweep.main(ready) == 2
        assert "start_barrier_refusal" in capsys.readouterr().err
        assert not (tmp_path / "run" / "shard-0").exists()
        marker = json.loads((tmp_path / "run" / ".barrier" / "shard-0.json").read_text())
        assert marker["slot_held"] == "cpu-slot-3.lock"  # took slot 3 first, then refused
        assert sweep.main(["slots-free", "--slot-lock-root", str(tmp_path / "locks")]) == 0

    def test_a_full_rendezvous_reaches_the_shard_body(self, ready, tmp_path, monkeypatch):
        sha = dev_screen.sha256_file(sweep.SPEC.protocol)
        for shard, slot in ((1, SLOTS[0]), (2, SLOTS[1])):
            _marker(tmp_path / "run" / ".barrier", shard, slot, protocol_sha256=sha)
        seen = {}

        def body(*args, barrier=None, **kwargs):
            seen.update(slot=args[-1], barrier=barrier)
            return 0

        monkeypatch.setattr(sweep, "_run_shard", body)
        assert sweep.main(ready) == 0
        assert seen["slot"] == "cpu-slot-3.lock" and seen["barrier"]["passed"]
