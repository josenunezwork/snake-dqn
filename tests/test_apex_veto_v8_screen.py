"""Tests for the v8 (lambda=8) vs v7 (lambda=4) Tier-1 screen harness (never plays a game)."""

from __future__ import annotations

import fcntl
import json
import os
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_v8_lambda_sweep_20261002 import sweep
from research.apex_veto_v8_screen_20261002 import screen

COMMIT = "f" * 40
SLOTS = ("cpu-slot-1.lock", "cpu-slot-2.lock", "cpu-slot-3.lock")
FLAGS = ["--use-slot-locks", "--slot-pool", "3", "--thermal-guard", "--require-ac-power"]


@pytest.fixture(autouse=True)
def no_episodes(monkeypatch):
    """Every episode runner is fatal in every test of this file."""
    from src.scripts import tournament_eval

    def fatal(*args, **kwargs):
        raise AssertionError("a screen test must never play an episode")

    monkeypatch.setattr(tournament_eval, "rollout", fatal)
    monkeypatch.setattr(dev_screen, "run_episode", fatal)


def _soon(hours=1.0):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _entry(arm, mix, seed, mass, survival=1.0, cause=None, record=None):
    return {
        "arm": arm,
        "mix": mix,
        "world_seed": seed,
        "wall_seconds": 30.0,
        "veto_diagnostics": (
            {"action_differs_from_reference": 1, "head_risky_vetoes": 1}
            if arm in ("B", "D")
            else None
        ),
        "record": record
        or {
            "mass_integral": mass,
            "survival_fraction": survival,
            "probes": {"death_cause": cause},
        },
    }


def _ok_check(entries, seeds, smoke):
    return {"entries": len(entries), "failures": [], "warnings": [], "passes": True}


def _entries(deltas_by_mix, seeds, controls=True):
    out = []
    for mix, deltas in deltas_by_mix.items():
        for seed, delta in zip(seeds, deltas):
            out.append(_entry("A", mix, seed, 100.0))
            out.append(_entry("B", mix, seed, 100.0 + delta))
        if controls:
            out.append(_entry("C", mix, seeds[0], 100.0))
            out.append(_entry("D", mix, seeds[0], 100.0 + deltas[0]))
    return out


def _summarize(entries, seeds, mixes=dev_screen.MIXES, controls=3):
    return screen.summarize(
        entries,
        seeds,
        mixes,
        smoke=False,
        planned_episodes=2 * len(seeds) * len(mixes) + 2 * controls // 3 * 3,
        control_planned=2 * controls // 3 * 3,
        source={"commit": COMMIT, "dirty_paths": ""},
        checker=_ok_check,
    )


# ---------------------------------------------------------------- worlds and arms


class TestWorldsAndArms:
    def test_domain_is_the_reserved_one_and_excluded_banks_are_all_there(self):
        assert screen.DOMAIN == "apex-veto-v8-screen-v1" in sweep.SCREEN_DOMAINS
        assert screen.SMOKE_DOMAIN == "apex-veto-v8-screen-smoke-v1" in sweep.SCREEN_DOMAINS
        earlier = set(screen.EARLIER_DOMAINS)
        assert set(sweep.EARLIER_DOMAINS) <= earlier
        for name in (
            "apex-veto-v8-dev-v1",
            "apex-veto-v8-dev-smoke-v1",
            *sweep.V7_STRICT_DOMAINS,
            "apex-veto-v7-screen-v1",
            "apex-veto-v7-dev-v1",
            "apex-veto-v7-web-serving-v1",
            "apex-veto-v7-web-serving-smoke-v1",
        ):
            assert name in earlier, name
        assert set(screen.EARLIER_DOMAINS["apex-veto-v7-web-serving-v1"]) >= {
            "watch",
            "play",
            "parity",
        }

    def test_screen_seeds_avoid_every_earlier_namespace(self):
        for own in (screen.DOMAIN, screen.SMOKE_DOMAIN):
            banned = screen.earlier_namespaces(own)
            assert not any(key.startswith(f"{own}/") for key in banned)
            seeds = set(dev_screen.screen_seeds(screen.WORLDS_PER_MIX, own, screen.NAMESPACE))
            assert len(seeds) == screen.WORLDS_PER_MIX
            for name, values in banned.items():
                assert not seeds & set(values), name
        real = set(dev_screen.screen_seeds(60, screen.DOMAIN, "worlds"))
        smoke = set(dev_screen.screen_seeds(60, screen.SMOKE_DOMAIN, "worlds"))
        assert not real & smoke

    def test_observed_seed_scan_reads_nested_keys_and_fails_closed(self, tmp_path):
        rosters = tmp_path / "rosters.json"
        rosters.write_text(
            json.dumps(
                {
                    "final": [{"world_seed": 11, "slots": [{"slot": 1}]}],
                    "serving": [{"world_seed": 12}],
                    "x": {"seeds": {"watch": [13, 14]}, "paired_seeds": [15], "other": [99]},
                }
            )
        )
        observed = screen.observed_seed_namespaces([rosters])
        assert observed == {f"observed:{rosters}": [11, 12, 13, 14, 15]}
        with pytest.raises(FileNotFoundError):
            screen.observed_seed_namespaces([tmp_path / "missing.json"])
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"no": [1, 2]}))
        with pytest.raises(ValueError):
            screen.observed_seed_namespaces([empty])

    def test_arms_installers_and_descriptors(self):
        from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
        from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto

        assert screen.INSTALLERS["A"] is screen.INSTALLERS["C"] is sweep.INSTALLERS["A"]
        assert screen.INSTALLERS["B"] is screen.INSTALLERS["D"] is sweep.INSTALLERS["H800"]
        assert sweep.INSTALLERS["H800"].keywords == {"lam": 8.0}
        assert screen.arm_method("A") == "free-space-veto/v7-space-preference(lambda=4.0)"
        assert screen.arm_method("D") == "free-space-veto/v8-space-and-head(lambda=8.0)"
        assert screen.expected_descriptor("C") == SpacePreferenceVeto(4.0).descriptor()
        assert screen.expected_descriptor("B") == SpaceAndHeadVeto(8.0).descriptor()
        spec = screen.make_spec(screen.DOMAIN, {})
        assert spec.arm_vetoes["C"] is spec.arm_vetoes["A"]
        assert spec.require_slot_locks and spec.require_ac_power
        assert spec.max_wall_seconds == 4 * 3600

    def test_installed_b_is_v8_at_8_with_the_v7_4_reference(self, setup_config):
        from src.game.ai_snake import AISnake

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=SimpleNamespace())
        veto = screen.INSTALLERS["B"](snake)
        assert snake.safety_veto is veto and veto.lam == 8.0 and veto.reference_lambda == 4.0
        assert veto.head_avoidance is True


class TestSweepBinding:
    def test_pinned_summary_and_its_checks(self, tmp_path):
        good = {
            "schema_version": sweep.SCHEMA,
            "sweep_id": sweep.SWEEP_ID,
            "smoke": False,
            "selection": {"status": "SELECTED", "passes": True, "selected_lambda": 8.0},
            "source": {"commit": "a" * 40},
        }
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(good))
        sha = dev_screen.sha256_file(path)
        assert screen.check_sweep_binding(path, sha)["selected_lambda"] == 8.0
        with pytest.raises(screen.SweepBindingError, match="sha256"):
            screen.check_sweep_binding(path, "0" * 64)
        for change in (
            {"smoke": True},
            {"selection": {"status": "SELECTED", "passes": True, "selected_lambda": 16.0}},
            {"selection": {"status": "NONE_QUALIFIES", "passes": False}},
        ):
            bad = tmp_path / f"bad{len(list(tmp_path.iterdir()))}.json"
            bad.write_text(json.dumps({**good, **change}))
            with pytest.raises(screen.SweepBindingError):
                screen.check_sweep_binding(bad, dev_screen.sha256_file(bad))
        with pytest.raises(screen.SweepBindingError, match="does not exist"):
            screen.check_sweep_binding(tmp_path / "nope.json", sha)


# ---------------------------------------------------------------- plan


class TestPlan:
    def test_every_episode_once_and_controls_cross_shard(self):
        seeds = list(range(1000, 1060))
        rows = dev_screen._design_rows(seeds)
        plans = {k: screen.shard_plan(rows, seeds, dev_screen.MIXES, k, 3, False) for k in range(3)}
        assert [len(plans[k]) for k in range(3)] == [120, 126, 126]
        keys = [tuple(key) for k in range(3) for key in screen.plan_keys(plans[k])]
        assert len(keys) == len(set(keys)) == 372
        owner = {key: k for k in range(3) for key in map(tuple, screen.plan_keys(plans[k]))}
        for mix in dev_screen.MIXES:
            for seed in seeds:
                assert ("A", mix, seed) in owner and ("B", mix, seed) in owner
                assert owner[("A", mix, seed)] == owner[("B", mix, seed)]
            for seed in seeds[: screen.CONTROL_WORLDS]:
                assert owner[("C", mix, seed)] != owner[("A", mix, seed)]
                assert owner[("D", mix, seed)] != owner[("B", mix, seed)]
        controls = [key for key in keys if key[0] in ("C", "D")]
        assert len(controls) == 2 * 3 * screen.CONTROL_WORLDS
        # within a shard: A/B pairs first in (world index, mix) order, controls last
        for k in range(3):
            arms = [arm for arm, _ in plans[k]]
            first_control = next((i for i, a in enumerate(arms) if a in "CD"), len(arms))
            assert all(a in "AB" for a in arms[:first_control])
            assert all(a in "CD" for a in arms[first_control:])

    def test_smoke_plays_a_and_b_only(self):
        rows = dev_screen._design_rows([7])
        rows = [row for row in rows if row["mix"] == "scripted"]
        plan = screen.shard_plan(rows, [7], ("scripted",), 0, 1, True)
        assert [arm for arm, _ in plan] == ["A", "B"]


# ---------------------------------------------------------------- statistics and decision


class TestStatistics:
    def test_mix_bounds_match_the_t_formula(self):
        deltas = [1.0, -2.0, 4.0, 0.0, 3.0]
        row = screen.mix_bounds(deltas)
        mean = sum(deltas) / 5
        sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / 4)
        assert row["mean"] == pytest.approx(mean)
        assert row["upper_90"] == pytest.approx(mean + 1.5332 * sd / math.sqrt(5), abs=1e-3)
        assert row["lower_90"] == pytest.approx(mean - 1.5332 * sd / math.sqrt(5), abs=1e-3)
        assert screen.mix_bounds([3.0])["upper_90"] is None

    def test_pooled_bound_equals_the_stratified_80pct_two_sided_lower_end(self):
        from src.evaluation import screen_stats

        deltas = {
            "frozen": [0.0, -3.0, 1.0, 0.0, -10.0, 2.0],
            "scripted": [50.0, 0.0, 120.0, -5.0, 0.0, 30.0, 8.0],
            "mixed": [0.0, 4.0, -1.0, 9.0, 0.0],
        }
        mine = screen.pooled_bound(deltas)
        ref = screen_stats.stratified_mean_of_means(deltas, confidence=0.80)
        assert mine["mean"] == pytest.approx(ref["mean_of_means"])
        assert mine["se"] == pytest.approx(ref["se"])
        assert mine["df"] == pytest.approx(ref["df"])
        assert mine["lower_90"] == pytest.approx(ref["ci"][0], rel=1e-9)

    def test_all_ties_give_the_mean_as_bound(self):
        pooled = screen.pooled_bound({"a": [0.0, 0.0], "b": [0.0, 0.0]})
        assert pooled["lower_90"] == 0.0 and pooled["se"] == 0.0

    def test_decide(self):
        ok_mix = {"upper_90": 5.0}
        loss_mix = {"upper_90": -0.1}
        assert screen.decide({"f": ok_mix}, {"mean": 3.0, "lower_90": 0.1})["status"] == "ADVANCE"
        out = screen.decide({"f": loss_mix, "s": ok_mix}, {"mean": 30.0, "lower_90": 10.0})
        assert out["status"] == "NOT_ADVANCED" and out["loss_mixes"] == ["f"]
        assert screen.decide({"f": ok_mix}, {"mean": 3.0, "lower_90": 0.0})["status"] == (
            "NOT_ADVANCED"
        )
        assert screen.decide({"f": ok_mix}, {"mean": None, "lower_90": None})["status"] == (
            "NOT_ADVANCED"
        )


class TestSummarize:
    SEEDS = list(range(1, 11))

    def _deltas(self, frozen, scripted, mixed):
        return {"frozen": frozen, "scripted": scripted, "mixed": mixed}

    def test_advance_on_a_clear_gain_without_losses(self):
        d = self._deltas(
            [0, 1, 0, 2, 0, 0, 1, 0, 0, 3],
            [50, 40, 0, 60, 70, 0, 30, 20, 50, 40],
            [0, 5, 0, 10, 0, 0, 0, 8, 0, 2],
        )
        summary = _summarize(_entries(d, self.SEEDS), self.SEEDS)
        assert summary["decision"] == "ADVANCE"
        assert summary["pooled"]["pairs"] == 30 and summary["control"]["passes"]
        assert summary["per_mix"]["scripted"]["mass_integral"]["wins_B"] == 8

    def test_a_significant_frozen_loss_blocks_advance(self):
        d = self._deltas(
            [-10, -12, -9, -11, -10, -13, -8, -10, -12, -9],
            [200, 150, 180, 220, 160, 190, 170, 210, 200, 180],
            [0, 5, 0, 10, 0, 0, 0, 8, 0, 2],
        )
        summary = _summarize(_entries(d, self.SEEDS), self.SEEDS)
        assert summary["pooled"]["lower_90"] > 0
        assert summary["decision"] == "NOT_ADVANCED"
        assert summary["rule_outcome"]["loss_mixes"] == ["frozen"]

    def test_noisy_gain_without_a_positive_bound_is_not_advanced(self):
        d = self._deltas([0] * 9 + [5], [0] * 9 + [300], [0] * 10)
        summary = _summarize(_entries(d, self.SEEDS), self.SEEDS)
        assert summary["pooled"]["mean"] > 0 and summary["pooled"]["lower_90"] <= 0
        assert summary["decision"] == "NOT_ADVANCED"

    def test_incomplete_and_nondeterministic_and_self_check_precede(self):
        d = self._deltas([1] * 10, [50] * 9 + [51], [3] * 9 + [4])
        entries = _entries(d, self.SEEDS)
        assert _summarize(entries[:-3], self.SEEDS)["decision"] == "INCOMPLETE"
        broken = [dict(e) for e in entries]
        for e in broken:
            if e["arm"] == "D" and e["mix"] == "mixed":
                e["record"] = {**e["record"], "mass_integral": -1.0}
        out = _summarize(broken, self.SEEDS)
        assert out["decision"] == "INVALID_NONDETERMINISTIC"
        assert out["control"]["mismatches"] == [
            {"arm": "D", "mix": "mixed", "world_seed": self.SEEDS[0]}
        ]
        failing = screen.summarize(
            entries,
            self.SEEDS,
            dev_screen.MIXES,
            smoke=False,
            planned_episodes=len(entries),
            control_planned=6,
            source={},
        )  # the real checker rejects these synthetic entries
        assert failing["decision"] == "INVALID_SELF_CHECK_FAILED"

    def test_death_causes_head_on_and_survival_are_reported(self):
        seeds = [1, 2]
        entries = [
            _entry("A", "scripted", 1, 10.0, survival=0.5, cause="head_on"),
            _entry("B", "scripted", 1, 20.0, survival=1.0),
            _entry("A", "scripted", 2, 10.0, survival=0.4, cause="self"),
            _entry("B", "scripted", 2, 5.0, survival=0.2, cause="head_on"),
        ]
        out = screen.summarize(
            entries,
            seeds,
            ("scripted",),
            smoke=True,
            planned_episodes=4,
            control_planned=0,
            source={},
            checker=_ok_check,
        )
        row = out["per_mix"]["scripted"]
        assert out["decision"] == "SMOKE_NO_DECISION"
        assert row["head_on_deaths"] == {"A": 1, "B": 1}
        assert row["death_causes"]["A"] == {"head_on": 1, "self": 1}
        assert row["survival_fraction"]["survived_B"] == 1
        assert row["survival_fraction"]["mean_delta"] == pytest.approx(0.15)
        assert row["v8_counters_B"]["head_risky_vetoes"] == 2.0
        assert out["head_on_deaths_total"] == {"A": 1, "B": 1}


# ---------------------------------------------------------------- self-check on real probes


def _probe_entry(arm, veto, seed=5):
    diagnostics = veto.diagnostics_record() if hasattr(veto, "diagnostics_record") else None
    return {
        "schema_version": screen.SCHEMA,
        "authority": screen.AUTHORITY,
        "screen": screen.SCREEN_ID,
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
    def test_real_probes_pass_and_mismatches_fail(self, setup_config):
        from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
        from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto
        from tests.test_safety_veto_v5 import random_states

        states = random_states(3, 20)
        arm_a = SpacePreferenceVeto(4.0)
        arm_b = SpaceAndHeadVeto(8.0, reference_lambda=4.0)
        for me, roster, q, mask, base in states:
            arm_a.apply(me, roster, q, mask, base)
            arm_b.apply(me, roster, q, mask, base)
        ok = {"failures": [], "warnings": []}
        for arm, veto in (("A", arm_a), ("C", arm_a), ("B", arm_b), ("D", arm_b)):
            assert screen.check_entry(_probe_entry(arm, veto), smoke=True) == ok, arm
        assert screen.check_entry(_probe_entry("B", arm_a), smoke=True)["failures"]
        assert screen.check_entry(_probe_entry("A", arm_b), smoke=True)["failures"]
        wrong_lambda = SpaceAndHeadVeto(16.0, reference_lambda=4.0)
        assert screen.check_entry(_probe_entry("B", wrong_lambda), smoke=True)["failures"]
        no_reference = SpaceAndHeadVeto(8.0)
        for me, roster, q, mask, base in states[:3]:
            no_reference.apply(me, roster, q, mask, base)
        failures = screen.check_entry(_probe_entry("B", no_reference), smoke=True)["failures"]
        assert any("reference diagnostic" in message for message in failures)
        full = screen.check_entry(_probe_entry("B", arm_b), smoke=False)["failures"]
        assert any("H5000" in message for message in full)
        sweep_entry = {**_probe_entry("A", arm_a), "schema_version": sweep.SCHEMA}
        assert screen.check_entry(sweep_entry, smoke=True)["failures"]

    def test_unknown_unplanned_and_duplicates_fail(self):
        result = screen.self_check([_entry("Z", "frozen", 1, 1.0)], [1], smoke=True)
        assert not result["passes"] and "unknown arm" in result["failures"][0]
        assert not screen.self_check([], [1], smoke=True)["passes"]


# ---------------------------------------------------------------- run refusals and merge


@pytest.fixture
def locks(tmp_path):
    root = tmp_path / "locks"
    root.mkdir()
    for name in SLOTS:
        (root / name).write_text("")
    return root


def _run_argv(tmp_path, locks, *extra, flags=FLAGS, hours=2.0, out="shard-0", shard="0"):
    return [
        "run",
        "--shard",
        shard,
        "--out",
        str(tmp_path / "run" / out),
        "--deadline-utc",
        _soon(hours),
        "--slot-lock-root",
        str(locks),
        "--slot-timeout-seconds",
        "0.2",
        "--barrier-timeout-seconds",
        "0.2",
        *extra,
        *flags,
    ]


@pytest.fixture
def real_run_env(tmp_path, locks, monkeypatch):
    """Point the real-run path/lock checks at temp dirs; the tree is clean, binding ok."""
    monkeypatch.setattr(screen, "SLOT_LOCK_ROOT", locks)
    monkeypatch.setattr(screen, "ALLOWED_RUN_ROOTS", (tmp_path / "run",))
    monkeypatch.setattr(screen, "tree_refusal", lambda: None)
    monkeypatch.setattr(dev_screen, "_git_state", lambda: {"commit": COMMIT, "dirty_paths": ""})
    monkeypatch.setattr(screen, "check_sweep_binding", lambda *a, **k: {"selected_lambda": 8.0})
    return tmp_path


class TestRunRefusals:
    @pytest.mark.parametrize(
        "extra, flags, hours, expected",
        [
            ([], ["--use-slot-locks", "--thermal-guard"], 2.0, "--slot-pool 3"),
            ([], ["--use-slot-locks", "--slot-pool", "3"], 2.0, "--thermal-guard"),
            (["--worlds-per-mix", "40"], FLAGS, 2.0, "pre-registered size is 60"),
            (["--shards", "2"], FLAGS, 2.0, "3 shards"),
            ([], FLAGS, 4.5, "wall-time cap"),
            ([], FLAGS, 1.0, "less than 5733 s"),
            (["--thermal-backoff-seconds", "5"], FLAGS, 2.0, "non-default --thermal-backoff"),
            (["--thermal-max-backoffs", "9"], FLAGS, 2.0, "non-default --thermal-max"),
            (["--min-episode-budget-seconds", "1"], FLAGS, 2.0, "non-default --min-episode"),
        ],
    )
    def test_refused_before_any_output(
        self, real_run_env, locks, capsys, extra, flags, hours, expected
    ):
        tmp_path = real_run_env
        assert screen.main(_run_argv(tmp_path, locks, *extra, flags=flags, hours=hours)) == 2
        assert expected in capsys.readouterr().err
        assert not (tmp_path / "run").exists()

    def test_out_and_lock_root_are_pinned_outside_a_smoke(
        self, real_run_env, locks, capsys, monkeypatch
    ):
        tmp_path = real_run_env
        assert screen.main(_run_argv(tmp_path, locks, out="elsewhere")) == 2
        assert "refusing --out" in capsys.readouterr().err
        assert screen.main(_run_argv(tmp_path, locks, out="shard-1")) == 2  # shard 0 -> shard-1
        assert "refusing --out" in capsys.readouterr().err
        monkeypatch.setattr(screen, "SLOT_LOCK_ROOT", tmp_path / "other-locks")
        assert screen.main(_run_argv(tmp_path, locks)) == 2
        assert "non-default --slot-lock-root" in capsys.readouterr().err
        assert not (tmp_path / "run").exists()

    def test_real_out_paths_are_the_run_root_or_recovery(self):
        dirs = screen.expected_shard_dirs(2)
        assert dirs == [
            screen.RUN_ROOT.resolve() / "shard-2",
            screen.RUN_ROOT.resolve() / "recovery-1" / "shard-2",
        ]
        assert str(screen.RUN_ROOT).endswith(
            "snake-dqn-artifacts/apex-veto-v8-screen-20261002/run-v1"
        )

    def test_tree_refusal_rejects_untracked_and_untracked_package(self, monkeypatch):
        calls = []

        def git(*args):
            calls.append(args)
            if args[0] == "status":
                return 0, state["porcelain"]
            return state["ls"], ""

        monkeypatch.setattr(screen, "_git", git)
        state = {"porcelain": "?? notes.txt\n", "ls": 0}
        assert "dirty tree" in screen.tree_refusal()
        assert "--untracked-files=all" in calls[0]
        state = {"porcelain": " M research/x.py\n", "ls": 0}
        assert "dirty tree" in screen.tree_refusal()
        state = {"porcelain": "", "ls": 1}
        assert "is not tracked" in screen.tree_refusal()
        state = {"porcelain": "", "ls": 0}
        assert screen.tree_refusal() is None

    def test_dirty_tree_and_bad_binding_are_refused(self, real_run_env, locks, capsys, monkeypatch):
        tmp_path = real_run_env
        monkeypatch.setattr(screen, "tree_refusal", lambda: "refusing: dirty tree (test)")
        assert screen.main(_run_argv(tmp_path, locks)) == 2
        assert "dirty tree (test)" in capsys.readouterr().err
        monkeypatch.setattr(screen, "tree_refusal", lambda: None)

        def bad(*a, **k):
            raise screen.SweepBindingError("nope")

        monkeypatch.setattr(screen, "check_sweep_binding", bad)
        assert screen.main(_run_argv(tmp_path, locks)) == 2
        assert "sweep binding: nope" in capsys.readouterr().err
        assert not (tmp_path / "run").exists()

    def test_smoke_shape_and_location(self, tmp_path, locks, capsys):
        smoke = ["--smoke-frames", "500", "--worlds-per-mix", "1"]
        argv = _run_argv(tmp_path, locks, *smoke)  # --shards defaults to 3
        assert screen.main(argv) == 2
        assert "--shard 0 --shards 1" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            screen.main(_run_argv(tmp_path, locks, "--smoke-frames", "900", "--shards", "1"))

    @pytest.fixture
    def preflight_ok(self, setup_config, monkeypatch):
        monkeypatch.setattr(dev_screen, "_configure_torch", lambda: None)
        monkeypatch.setattr(dev_screen, "on_ac_power", lambda: True)
        monkeypatch.setattr(dev_screen, "preflight_failures", lambda *a: [])
        monkeypatch.setattr(screen, "observed_seed_namespaces", lambda *a: {})

    def test_held_slots_are_never_bypassed(
        self, real_run_env, preflight_ok, locks, capsys, monkeypatch
    ):
        tmp_path = real_run_env

        def body(*args, **kwargs):
            raise AssertionError("no slot -> no shard body")

        monkeypatch.setattr(screen, "_run_shard", body)
        handles = [(locks / name).open("r") for name in SLOTS]
        try:
            for handle in handles:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert screen.main(_run_argv(tmp_path, locks)) == 2
            assert "slot lock not acquired" in capsys.readouterr().err
        finally:
            for handle in handles:
                handle.close()
        assert not (tmp_path / "run" / "shard-0").exists()

    def test_a_lone_shard_fails_the_start_barrier_before_any_output(
        self, real_run_env, preflight_ok, locks, capsys, monkeypatch
    ):
        tmp_path = real_run_env

        def body(*args, **kwargs):
            raise AssertionError("a refused shard must not reach the shard body")

        monkeypatch.setattr(screen, "_run_shard", body)
        assert screen.main(_run_argv(tmp_path, locks)) == 2
        assert "start_barrier_refusal" in capsys.readouterr().err
        assert not (tmp_path / "run" / "shard-0").exists()
        marker = json.loads((tmp_path / "run" / ".barrier" / "shard-0.json").read_text())
        assert marker["slot_held"] == "cpu-slot-3.lock" and "screen_sha256" in marker

    def test_a_full_rendezvous_reaches_the_shard_body(
        self, real_run_env, preflight_ok, locks, monkeypatch
    ):
        tmp_path = real_run_env
        identity = {
            "commit": COMMIT,
            "protocol_sha256": dev_screen.sha256_file(screen.HERE / "protocol.md"),
            "screen_sha256": screen.package_sha256()["screen.py"],
            "shards": 3,
        }
        barrier_dir = tmp_path / "run" / ".barrier"
        barrier_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc).isoformat()
        for shard, slot in ((1, SLOTS[0]), (2, SLOTS[1])):
            marker = {**identity, "shard": shard, "slot_held": slot, "pid": os.getpid()}
            marker["written_utc"] = now
            (barrier_dir / f"shard-{shard}.json").write_text(json.dumps(marker))
        seen = {}

        def body(*args, barrier=None, **kwargs):
            seen["slot"], seen["barrier"] = args[9], barrier
            return 0

        monkeypatch.setattr(screen, "_run_shard", body)
        assert screen.main(_run_argv(tmp_path, locks)) == 0
        assert seen["slot"] == "cpu-slot-3.lock" and seen["barrier"]["passed"]

    def test_missing_observed_source_refuses(self, tmp_path, locks, capsys, monkeypatch):
        def missing(*a):
            raise FileNotFoundError("observed-seed source missing")

        monkeypatch.setattr(screen, "observed_seed_namespaces", missing)
        smoke = ["--smoke-frames", "500", "--worlds-per-mix", "1", "--shards", "1"]
        assert screen.main(_run_argv(tmp_path, locks, *smoke, out="smoke")) == 2
        assert "observed-seed source missing" in capsys.readouterr().err


class TestShardBodyAndMerge:
    """The shard body and merge with a fake runner (no rollout)."""

    def _args(self, shard, out):
        return SimpleNamespace(
            deadline_utc=datetime.now(timezone.utc) + timedelta(hours=1),
            min_episode_budget_seconds=45.0,
            thermal_backoff_seconds=60.0,
            thermal_max_backoffs=5,
            smoke_frames=None,
            shard=shard,
            shards=3,
            mixes=dev_screen.MIXES,
            checkpoint_dir=out,
            config=dev_screen.DEFAULT_CONFIG,
            slot_lock_root=Path("/nonexistent"),
        )

    @pytest.fixture
    def fake_body(self, setup_config, monkeypatch):
        monkeypatch.setattr(dev_screen, "_git_state", lambda: {"commit": COMMIT, "dirty_paths": ""})
        monkeypatch.setattr(dev_screen, "snapshot_checkpoints", lambda *a: {})
        monkeypatch.setattr(dev_screen, "agent_lookup", lambda *a: {})
        monkeypatch.setattr(dev_screen, "on_ac_power", lambda: True)

        class Guard:
            def config(self):
                return {}

            def record_episode(self, key, seconds):
                self.key = key

        monkeypatch.setattr(dev_screen, "make_thermal_guard", lambda ac: Guard())
        import research.compute.thermal_guard as tg

        monkeypatch.setattr(tg, "admit_next_episode", lambda guard, **kw: (True, None, {}))

    def _fake_runner(self, played, crash_after=None):
        def runner(arm, row, index, lookup, profile, records_dir, smoke_frames=None, spec=None):
            if crash_after is not None and len(played) >= crash_after:
                raise RuntimeError("boom")
            played.append((arm, row["mix"], row["world_seed"]))
            entry = {
                "arm": arm,
                "mix": row["mix"],
                "world_seed": row["world_seed"],
                "wall_seconds": 0.01,
                "record": {"mass_integral": 1.0, "survival_fraction": 1.0, "probes": {}},
            }
            dev_screen.write_new_json(
                records_dir / f"{arm}-{row['mix']}-{row['world_seed']}.json", entry
            )
            return entry

        return runner

    def _body(self, tmp_path, runner, shard=1):
        seeds = [1001, 1002, 1003]
        out = tmp_path / f"shard-{shard}"
        code = screen._run_shard(
            self._args(shard, tmp_path),
            [],
            screen.make_spec(screen.DOMAIN, {}),
            out,
            seeds,
            {"disjoint": True},
            {"checked": True},
            None,
            False,
            "cpu-slot-3.lock",
            runner=runner,
            barrier={"passed": True},
        )
        return code, out, seeds

    def test_shard_body_writes_intent_records_and_summary(self, fake_body, tmp_path):
        played = []
        code, out, seeds = self._body(tmp_path, self._fake_runner(played))
        assert code == 3  # fake records fail the gating self-check, which is reported
        intent = json.loads((out / "intent.json").read_text())
        rows = dev_screen._design_rows(seeds)
        expected = screen.plan_keys(screen.shard_plan(rows, seeds, dev_screen.MIXES, 1, 3, False))
        assert intent["plan"] == expected and [list(p) for p in played] == expected
        assert intent["decision_rule"] == screen.DECISION_RULE
        assert intent["package_sha256"] == screen.package_sha256()
        assert intent["run_params"] == screen.FIXED_RUN_PARAMS
        assert intent["start_barrier"] == {"passed": True}
        summary = json.loads((out / "shard_summary.json").read_text())
        assert summary["episodes_run"] == len(expected) and not summary["self_check"]["passes"]

    def test_a_crash_still_writes_the_shard_summary(self, fake_body, tmp_path):
        played = []
        code, out, _ = self._body(tmp_path, self._fake_runner(played, crash_after=4))
        assert code == 4
        summary = json.loads((out / "shard_summary.json").read_text())
        assert summary["stopped_reason"].startswith("error: RuntimeError: boom")
        assert summary["episodes_run"] == 4 and not summary["snapshot_changed"]

    def _merge_shards(self):
        sha = dev_screen.sha256_file(screen.HERE / "protocol.md")
        seeds = dev_screen.screen_seeds(60, screen.DOMAIN, "worlds")
        rows = dev_screen._design_rows(seeds)

        def shard(k):
            plan = screen.plan_keys(screen.shard_plan(rows, seeds, dev_screen.MIXES, k, 3, False))
            return {
                "dir": str(screen.expected_shard_dirs(k)[0]),
                "entries": [],
                "intent": {
                    "schema_version": screen.SCHEMA,
                    "screen_id": screen.SCREEN_ID,
                    "smoke_frames": None,
                    "shard": k,
                    "shards": 3,
                    "protocol_sha256": sha,
                    "worlds": {"domain": screen.DOMAIN, "seeds": seeds},
                    "mixes": list(dev_screen.MIXES),
                    "git": {"commit": COMMIT},
                    "package_sha256": screen.package_sha256(),
                    "slot_locks": {"root": str(screen.SLOT_LOCK_ROOT)},
                    "run_params": dict(screen.FIXED_RUN_PARAMS),
                    "start_barrier": {"passed": True},
                    "plan": plan,
                },
            }

        return [shard(k) for k in range(3)]

    def test_merge_accepts_a_well_formed_set(self):
        assert screen.merge_refusal(self._merge_shards()) is None

    @pytest.mark.parametrize(
        "mutate, expected",
        [
            (lambda s: s.pop(), "exactly 3"),
            (lambda s: s[2]["intent"].update(screen_id="x"), "not from this screen"),
            (lambda s: s[2]["intent"].update(smoke_frames=500), "smoke"),
            (lambda s: s[2]["intent"].update(shard=1), "0..2"),
            (lambda s: [x["intent"].update(protocol_sha256="p") for x in s], "protocol bytes"),
            (
                lambda s: [x["intent"]["worlds"].update(seeds=[1, 2]) for x in s],
                "seeds are not",
            ),
            (lambda s: s[1]["intent"].update(package_sha256={"screen.py": "z"}), "screen.py"),
            (lambda s: s[1]["intent"].update(slot_locks={"root": "/tmp/x"}), "slot lock root"),
            (
                lambda s: s[1]["intent"].update(run_params={"thermal_backoff_seconds": 1}),
                "operating parameters",
            ),
            (lambda s: s[1]["intent"].update(start_barrier={"passed": False}), "start barrier"),
            (lambda s: s[1].update(dir="/tmp/shard-1"), "directory"),
            (lambda s: s[2]["intent"].update(git={"commit": "e" * 40}), "different commits"),
            (lambda s: s[2]["intent"]["plan"].reverse(), "plan differs"),
            (
                lambda s: s[0].update(entries=[{"arm": "A", "mix": "frozen", "world_seed": 5}]),
                "prefix of its plan",
            ),
        ],
    )
    def test_merge_refuses_wrong_structure(self, mutate, expected):
        shards = self._merge_shards()
        mutate(shards)
        refusal = screen.merge_refusal(shards)
        assert refusal and expected in refusal, refusal

    def test_pairing_and_role_label(self):
        a = _entry("A", "frozen", 1, 1.0)
        b = _entry("B", "frozen", 1, 2.0)
        a["roster_member_sha256s"], b["roster_member_sha256s"] = ["x"], ["y"]
        assert screen.pairing_failures([a, b])
        b["roster_member_sha256s"] = ["x"]
        assert not screen.pairing_failures([a, b])
        b["record"] = {**b["record"], "world_identity": {"seed": 9}}
        assert screen.pairing_failures([a, b])
        d = {m: [0.0, 1.0, 2.0] for m in dev_screen.MIXES}
        seeds = [1, 2, 3]
        out = _summarize(_entries(d, seeds), seeds)
        assert "no Holm" in out["pooled_informational"]["role"]
