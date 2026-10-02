"""Tests for the v7 DEV lambda sweep and the v7 Tier-1 screen harnesses (never play)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_v6_screen_20261002 import screen as v6screen
from research.apex_veto_v7_lambda_sweep_20261002 import sweep
from research.apex_veto_v7_screen_20261002 import screen
from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto
from src.evaluation.safety_veto_v7 import SpacePreferenceVeto

REPO = Path(__file__).resolve().parents[1]
COMMIT = "c" * 40


@pytest.fixture
def no_episodes(monkeypatch):
    """Make every episode runner fatal (v3 revision 4: a guard test ran a real screen)."""
    from src.scripts import tournament_eval

    def fatal(*args, **kwargs):
        raise AssertionError("a guard test must never play an episode")

    monkeypatch.setattr(tournament_eval, "rollout", fatal)
    monkeypatch.setattr(dev_screen, "run_episode", fatal)


def _soon(hours=1.0):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _entry(arm, mix, seed, mass):
    return {
        "arm": arm,
        "mix": mix,
        "world_seed": seed,
        "wall_seconds": 1.0,
        "record": {"mass_integral": mass, "probes": {"death_cause": "self"}},
    }


# ---------------------------------------------------------------- namespaces and specs


class TestNamespaces:
    def test_sweep_worlds_are_fresh_against_every_earlier_bank(self):
        earlier = set(sweep.EARLIER_DOMAINS)
        assert set(v6screen.EARLIER_DOMAINS) <= earlier
        assert {v6screen.DOMAIN, v6screen.SMOKE_DOMAIN, *sweep.SCREEN_DOMAINS} <= earlier
        assert {screen.DOMAIN, screen.SMOKE_DOMAIN} == set(sweep.SCREEN_DOMAINS)
        assert sweep.DOMAIN == "apex-veto-v7-dev-v1"
        for spec, own in ((sweep.SPEC, sweep.DOMAIN), (sweep.SMOKE_SPEC, sweep.SMOKE_DOMAIN)):
            assert not any(k.startswith(f"{own}/") for k in spec.extra_namespaces)
            seeds = dev_screen.screen_seeds(8, own, sweep.NAMESPACE)
            report = dev_screen.disjointness_report(
                seeds, None, screen_domain=own, extra_namespaces=spec.extra_namespaces
            )
            assert report["disjoint"] is True
            assert len(report["extra_namespaces"]) == len(spec.extra_namespaces) >= 42

    def test_screen_worlds_are_fresh_including_the_sweep(self):
        earlier = set(screen.EARLIER_DOMAINS)
        assert set(v6screen.EARLIER_DOMAINS) <= earlier
        assert {sweep.DOMAIN, sweep.SMOKE_DOMAIN, v6screen.DOMAIN} <= earlier
        spec = screen.build_spec(screen.DOMAIN, 2.0, {})
        assert not any(k.startswith(f"{screen.DOMAIN}/") for k in spec.extra_namespaces)
        assert any(k.startswith(f"{sweep.DOMAIN}/") for k in spec.extra_namespaces)
        seeds = dev_screen.screen_seeds(40, screen.DOMAIN, screen.NAMESPACE)
        report = dev_screen.disjointness_report(
            seeds, None, screen_domain=screen.DOMAIN, extra_namespaces=spec.extra_namespaces
        )
        assert report["disjoint"] is True

    def test_sweep_spec_and_installers(self):
        spec = sweep.SPEC
        assert spec.arm_vetoes["R"] is spec.arm_vetoes["L200"]
        assert spec.arm_vetoes["A"] is sweep.install_v5
        assert set(spec.arm_vetoes) == {"A", "L100", "L200", "L400", "R"}
        assert spec.require_slot_locks and spec.require_ac_power
        assert spec.max_wall_seconds == sweep.MAX_WALL_SECONDS == 3 * 3600
        assert sweep.expected_descriptor("A") == BoostAwareFreeSpaceVeto().descriptor()
        assert sweep.expected_descriptor("R") == SpacePreferenceVeto(2.0).descriptor()
        assert sweep.arm_method("L100") == "free-space-veto/v7-space-preference(lambda=1.0)"

    def test_screen_spec(self):
        binding = {"summary_sha256": "s" * 64, "source_commit": COMMIT}
        spec = screen.build_spec(screen.DOMAIN, 4.0, binding)
        assert spec.arm_vetoes["A"] is spec.arm_vetoes["C"] is screen.install_v5
        assert spec.arm_vetoes["B"] is spec.arm_vetoes["D"]
        assert spec.arm_vetoes["B"].keywords == {"lam": 4.0}
        assert spec.replay_worlds == 4 and spec.max_wall_seconds == 4 * 3600
        assert "s" * 64 in spec.arm_descriptions["B"] and "lambda=4.0" in (
            spec.arm_descriptions["B"]
        )
        fields = dev_screen.spec_intent_fields(
            spec, 276, datetime(2026, 10, 3, tzinfo=timezone.utc)
        )
        protocol = REPO / "research/apex_veto_v7_screen_20261002/protocol.md"
        assert fields["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
        assert "B/D replay control on 4 worlds" in fields["decision_rule"]
        assert screen.expected_descriptor("D", 4.0) == SpacePreferenceVeto(4.0).descriptor()

    def test_sweep_plan_runs_every_arm_per_world_then_replays(self):
        rows = [{"mix": m, "world_seed": s} for m in dev_screen.MIXES for s in (11, 12)]
        plan = sweep.plan_episodes(rows, dev_screen.MIXES, smoke=False)
        assert len(plan) == 2 * 3 * 4 + 3
        assert [arm for arm, _ in plan[:4]] == list(sweep.SWEEP_ARMS)
        assert [(a, r["world_seed"]) for a, r in plan[-3:]] == [("R", 11)] * 3
        smoke = sweep.plan_episodes(rows[:1], ("frozen",), smoke=True)
        assert [arm for arm, _ in smoke] == ["A", "L400"]


# ---------------------------------------------------------------- selection rule


def _per_seed(value, seeds):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * len(seeds)


def _sweep_entries(deltas, seeds=(1, 2), changes=None, decisions=1000):
    """A at 100 per world; arm deltas per mix from ``deltas[arm][mix]`` (a value or one per
    seed); every v7 episode has ``changes[arm]`` re-rank changes (default 3)."""
    changes = changes or {}
    entries = []
    for mix in dev_screen.MIXES:
        for index, seed in enumerate(seeds):
            entries.append(_entry("A", mix, seed, 100.0))
            for arm in sweep.V7_ARMS:
                delta = _per_seed(deltas[arm][mix], seeds)[index]
                entry = _entry(arm, mix, seed, 100.0 + delta)
                count = _per_seed(changes.get(arm, 3), seeds)[index]
                entry["veto_diagnostics"] = {"rerank_changes": count, "decisions": decisions}
                entries.append(entry)
    for mix in dev_screen.MIXES:
        twin = next(e for e in entries if e["arm"] == "L200" and e["mix"] == mix)
        entries.append(dict(twin, arm="R"))
    return entries


def _flat(frozen, scripted, mixed):
    return {"frozen": frozen, "scripted": scripted, "mixed": mixed}


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
            "L100": _flat(5, 5, 5),
            "L200": _flat(40, -11, 40),  # best pooled but a clear loser in scripted (sd 0)
            "L400": _flat(20, (-30, 10), 12),  # scripted mean -10, but noisy: not a loser
        }
        summary = self._summary(_sweep_entries(deltas), monkeypatch)
        selection = summary["selection"]
        assert selection["status"] == "SELECTED" and selection["passes"] is True
        assert selection["selected_lambda"] == 4.0
        assert selection["qualifying_lambdas"] == [1.0, 4.0]
        row = summary["lambdas"]["L200"]
        assert row["pooled_mean_delta"] == pytest.approx(23.0)
        assert row["active"] and row["clear_loser_in_some_mix"] and not row["qualifies"]
        scripted = summary["lambdas"]["L400"]["per_mix"]["scripted"]
        assert scripted["upper_bound_90_one_sided"] == pytest.approx(-10 + 3.0777 * 20)
        assert summary["source"]["commit"] == COMMIT

    def test_an_inert_arm_can_never_be_selected(self, monkeypatch):
        # L100 changes nothing (all deltas 0); the active arms pool negative. The first
        # draft's rule (floor -10, ties to the smaller lambda) would have picked L100.
        deltas = {
            "L100": _flat(0, 0, 0),
            "L200": _flat((-30, 20), (-25, 15), (-20, 10)),
            "L400": _flat((-40, 20), (-35, 15), (-30, 10)),
        }
        summary = self._summary(_sweep_entries(deltas, changes={"L100": 0}), monkeypatch)
        assert summary["lambdas"]["L100"]["active"] is False
        assert summary["selection"]["status"] == "SELECTED"
        assert summary["selection"]["selected_lambda"] == 2.0
        assert summary["selection"]["qualifying_lambdas"] == [2.0, 4.0]

    @pytest.mark.parametrize(
        "changes, deltas",
        [
            ({}, 0),  # changes but never a nonzero delta
            ({arm: (1, 0) for arm in sweep.V7_ARMS}, 5),  # 1 episode per mix with changes
            ({arm: 0 for arm in sweep.V7_ARMS}, 5),  # no changes at all
        ],
    )
    def test_no_active_arm_stops(self, monkeypatch, changes, deltas):
        entries = _sweep_entries(
            {a: _flat(deltas, deltas, deltas) for a in sweep.V7_ARMS}, changes=changes
        )
        selection = self._summary(entries, monkeypatch)["selection"]
        assert selection["status"] == "NONE_ACTIVE" and selection["passes"] is False
        assert selection["selected_lambda"] is None

    def test_rate_floor(self, monkeypatch):
        deltas = {arm: _flat(5, 5, 5) for arm in sweep.V7_ARMS}
        rare = _sweep_entries(deltas, changes={a: 1 for a in sweep.V7_ARMS}, decisions=10**5)
        assert self._summary(rare, monkeypatch)["selection"]["status"] == "NONE_ACTIVE"
        ok = _sweep_entries(deltas, changes={a: 1 for a in sweep.V7_ARMS}, decisions=10**4)
        assert self._summary(ok, monkeypatch)["selection"]["status"] == "SELECTED"

    def test_ties_go_to_the_larger_lambda(self, monkeypatch):
        deltas = {arm: _flat((-3, 1), (-3, 1), (-3, 1)) for arm in sweep.V7_ARMS}
        selection = self._summary(_sweep_entries(deltas), monkeypatch)["selection"]
        assert selection["status"] == "SELECTED" and selection["selected_lambda"] == 4.0

    def test_none_qualifies_stops(self, monkeypatch):
        deltas = {arm: _flat(50, -20, 50) for arm in sweep.V7_ARMS}
        selection = self._summary(_sweep_entries(deltas), monkeypatch)["selection"]
        assert selection["status"] == "NONE_QUALIFIES" and selection["passes"] is False
        assert selection["selected_lambda"] is None

    def test_earlier_statuses_take_precedence(self, monkeypatch):
        deltas = {arm: _flat(5, 5, 5) for arm in sweep.V7_ARMS}
        entries = _sweep_entries(deltas)
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

    def test_unknown_arm_and_bad_entry_fail_the_self_check(self):
        result = sweep.self_check([_entry("Z", "frozen", 1, 1.0)], [1], smoke=True)
        assert not result["passes"] and "unknown arm" in result["failures"][0]
        result = sweep.self_check([_entry("L200", "frozen", 1, float("nan"))], [1], smoke=True)
        assert not result["passes"]
        assert any("finite" in failure for failure in result["failures"])
        # Missing v7 diagnostics now gate (the activity gate reads them).
        assert any("v7 counters missing" in failure for failure in result["failures"])
        assert not sweep.self_check([], [1], smoke=True)["passes"]


# ---------------------------------------------------------------- live record self-check


class TestLiveEntrySelfCheck:
    def test_sweep_and_screen_checks_accept_real_v7_and_v5_records(
        self, setup_config, tmp_path, monkeypatch
    ):
        import torch

        from src.core.config_loader import load_and_initialize_config
        from src.model.apex_network import ApexNetwork
        from src.scripts.tournament_eval import rollout

        def fatal(*args, **kwargs):
            raise AssertionError("this test must not run sweep or screen episodes")

        monkeypatch.setattr(dev_screen, "run_episode", fatal)
        cfg = tmp_path / "tiny.yaml"
        cfg.write_text("game:\n  width: 300\n  height: 200\n  num_snakes: 3\n")
        load_and_initialize_config(str(cfg))
        torch.manual_seed(0)
        net = ApexNetwork(input_size=58, hidden_size=16, output_size=6)
        path = tmp_path / "vector.pth"
        torch.save({"dqn_state_dict": net.state_dict(), "input_size": 58, "hidden_size": 16}, path)
        opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
        for arm in ("A", "L200"):
            with dev_screen.hero_veto_installer(sweep.INSTALLERS[arm]) as installed:
                record = rollout(("checkpoint", str(path)), opponents, 40, 5, hero_safety_veto=True)
            entry = {
                "schema_version": sweep.SCHEMA,
                "authority": sweep.AUTHORITY,
                "screen": sweep.SWEEP_ID,
                "arm": arm,
                "mix": "scripted",
                "world_seed": 5,
                "hero_sha256": dev_screen.CHAMPION[1],
                "safety_veto": True,
                "safety_veto_method": record["probes"]["safety_veto"]["method"],
                "veto_diagnostics": installed[-1].diagnostics_record(),
                "record": record,
            }
            assert sweep.check_entry(entry, smoke=True) == {"failures": [], "warnings": []}
            assert sweep.check_entry(dict(entry, arm="L100"), smoke=True)["failures"]
            screen_arm = "A" if arm == "A" else "B"
            screen_entry = dict(
                entry, arm=screen_arm, schema_version=screen.SCHEMA, screen=screen.SCREEN_ID
            )
            screen_entry["authority"] = screen.AUTHORITY
            result = screen.check_entry(screen_entry, 2.0, smoke=True)
            assert result == {"failures": [], "warnings": []}
            assert screen.check_entry(screen_entry, 1.0, smoke=True)["failures"] or arm == "A"


# ---------------------------------------------------------------- sweep binding


def _write_sweep(tmp_path, **changes):
    """A synthetic sweep run directory (intent + summary) that passes unless changed."""
    run = tmp_path / "sweep-run"
    run.mkdir(exist_ok=True)
    intent = {"git": {"commit": changes.pop("intent_commit", COMMIT), "dirty_paths": ""}}
    (run / "intent.json").write_text(json.dumps(intent))
    summary = {
        "schema_version": sweep.SCHEMA,
        "sweep_id": sweep.SWEEP_ID,
        "smoke": False,
        "intent_sha256": dev_screen.sha256_file(run / "intent.json"),
        "selection": {"status": "SELECTED", "passes": True, "selected_lambda": 2.0},
        "source": {"commit": COMMIT, "dirty_paths": "?? notes.txt"},
    }
    for key, value in changes.items():
        if key in ("status", "passes", "selected_lambda"):
            summary["selection"][key] = value
        elif key in ("commit", "dirty_paths"):
            summary["source"][key] = value
        else:
            summary[key] = value
    (run / "summary.json").write_text(json.dumps(summary))
    return run / "summary.json"


CLEAN = {"commit": COMMIT, "dirty_paths": "?? research/untracked.txt"}


class TestSweepBinding:
    def test_a_selected_sweep_from_this_commit_binds(self, tmp_path):
        lam, binding = screen.load_sweep_selection(_write_sweep(tmp_path), CLEAN)
        assert lam == 2.0 and binding["source_commit"] == COMMIT
        assert binding["summary_sha256"] == dev_screen.sha256_file(
            tmp_path / "sweep-run/summary.json"
        )

    @pytest.mark.parametrize(
        "changes, git",
        [
            ({"status": "NONE_QUALIFIES", "passes": False, "selected_lambda": None}, CLEAN),
            ({"passes": False}, CLEAN),
            ({"selected_lambda": 0.75}, CLEAN),
            ({"selected_lambda": 0.5}, CLEAN),  # the first draft's grid
            ({"selected_lambda": True}, CLEAN),
            ({"smoke": True}, CLEAN),
            ({"schema_version": "other"}, CLEAN),
            ({"intent_sha256": "0" * 64}, CLEAN),
            ({"commit": "d" * 40}, CLEAN),
            ({"intent_commit": "d" * 40}, CLEAN),
            ({"dirty_paths": " M src/evaluation/safety_veto_v7.py"}, CLEAN),
            ({}, {"commit": "d" * 40, "dirty_paths": ""}),
            ({}, {"commit": COMMIT, "dirty_paths": " M research/x.py"}),
            ({}, {"commit": "", "dirty_paths": ""}),
        ],
    )
    def test_anything_else_is_refused(self, tmp_path, changes, git):
        with pytest.raises(screen.SweepBindingError):
            screen.load_sweep_selection(_write_sweep(tmp_path, **changes), git)

    def test_missing_files_are_refused(self, tmp_path):
        with pytest.raises(screen.SweepBindingError):
            screen.load_sweep_selection(tmp_path / "nope.json", CLEAN)
        path = _write_sweep(tmp_path)
        (path.parent / "intent.json").unlink()
        with pytest.raises(screen.SweepBindingError):
            screen.load_sweep_selection(path, CLEAN)


# ---------------------------------------------------------------- main guards (never play)


class TestSweepGuards:
    def _argv(self, tmp_path, *extra, hours=1.0):
        return ["--out", str(tmp_path / "o"), "--deadline-utc", _soon(hours), *extra]

    @pytest.mark.parametrize("worlds", ["4", "9", "40"])
    def test_non_preregistered_size_is_refused(self, no_episodes, tmp_path, worlds):
        assert sweep.main(self._argv(tmp_path, "--worlds-per-mix", worlds)) == 2
        assert not (tmp_path / "o").exists()

    def test_wall_cap_past_deadline_and_existing_out(self, no_episodes, tmp_path):
        assert sweep.main(self._argv(tmp_path, hours=3.1)) == 2
        assert sweep.main(self._argv(tmp_path, hours=-0.1)) == 2
        (tmp_path / "o").mkdir()
        assert sweep.main(self._argv(tmp_path)) == 2
        assert not any((tmp_path / "o").iterdir())

    def test_tracked_modifications_are_refused_before_any_lock_or_output(
        self, no_episodes, tmp_path, monkeypatch
    ):
        def fatal(*args, **kwargs):
            raise AssertionError("the dirty-tree guard must run before the slot lock")

        monkeypatch.setattr(dev_screen, "acquire_cpu_slots", fatal)
        dirty = {"commit": COMMIT, "dirty_paths": " M research/x.py\n?? notes.txt"}
        monkeypatch.setattr(dev_screen, "_git_state", lambda: dirty)
        assert sweep.main(self._argv(tmp_path)) == 2
        assert not (tmp_path / "o").exists()
        assert sweep.tracked_modifications(dirty["dirty_paths"]) == [" M research/x.py"]
        assert sweep.tracked_modifications("?? a\n?? b\n") == []
        assert screen.tracked_modifications is sweep.tracked_modifications

    def test_smoke_size_guard(self, no_episodes, tmp_path):
        for extra in (
            ("--smoke-frames", "501", "--worlds-per-mix", "1"),
            ("--smoke-frames", "500", "--worlds-per-mix", "2"),
            ("--smoke-frames", "500", "--worlds-per-mix", "1", "--smoke-mixes", "frozen,mixed"),
            ("--smoke-frames", "500", "--worlds-per-mix", "1", "--smoke-mixes", "bogus"),
        ):
            with pytest.raises(SystemExit):
                sweep.main(self._argv(tmp_path, *extra))
        assert not (tmp_path / "o").exists()


class TestScreenGuards:
    def _argv(self, tmp_path, *extra, hours=1.0):
        return ["--out", str(tmp_path / "o"), "--deadline-utc", _soon(hours), *extra]

    def test_a_sweep_summary_is_required(self, no_episodes, tmp_path):
        assert screen.main(self._argv(tmp_path)) == 2
        assert screen.main(self._argv(tmp_path, "--smoke-lambda", "2.0")) == 2  # not a smoke
        bad = _write_sweep(tmp_path, status="NONE_QUALIFIES", passes=False)
        assert screen.main(self._argv(tmp_path, "--sweep-summary", str(bad))) == 2
        assert not (tmp_path / "o").exists()

    def test_sweep_from_another_commit_is_refused(self, no_episodes, tmp_path):
        path = _write_sweep(tmp_path)  # COMMIT is never this checkout's HEAD
        assert screen.main(self._argv(tmp_path, "--sweep-summary", str(path))) == 2
        assert not (tmp_path / "o").exists()

    @pytest.mark.parametrize(
        "extra",
        [
            ("--worlds-per-mix", "5", "--determinism-worlds", "2"),
            ("--determinism-worlds", "0"),
            ("--worlds-per-mix", "41", "--determinism-worlds", "8"),
        ],
    )
    def test_non_preregistered_sizes_are_refused(self, no_episodes, tmp_path, extra):
        path = _write_sweep(tmp_path)
        argv = self._argv(tmp_path, "--sweep-summary", str(path), *extra)
        assert screen.main(argv) == 2
        assert not (tmp_path / "o").exists()

    def test_bound_spec_still_hits_the_wall_cap(self, no_episodes, tmp_path, monkeypatch):
        head = {"commit": COMMIT, "dirty_paths": ""}
        monkeypatch.setattr(dev_screen, "_git_state", lambda: head)
        path = _write_sweep(tmp_path)
        argv = self._argv(tmp_path, "--sweep-summary", str(path), hours=4.1)
        assert screen.main(argv) == 2
        assert not (tmp_path / "o").exists()

    def test_smoke_lambda_rules(self, no_episodes, tmp_path):
        smoke = ("--smoke-frames", "500", "--worlds-per-mix", "1", "--determinism-worlds", "0")
        assert screen.main(self._argv(tmp_path, *smoke, "--smoke-lambda", "0.3")) == 2
        path = _write_sweep(tmp_path)
        argv = self._argv(tmp_path, *smoke, "--smoke-lambda", "2.0", "--sweep-summary", str(path))
        assert screen.main(argv) == 2
        with pytest.raises(SystemExit):  # 4 episodes > 2
            screen.main(self._argv(tmp_path, "--smoke-frames", "500", "--smoke-lambda", "2.0"))
        assert not (tmp_path / "o").exists()

    def test_mismatched_control_arms_are_refused(self, no_episodes, tmp_path):
        spec = screen.build_spec(screen.DOMAIN, 2.0, {})
        argv = self._argv(tmp_path)
        for arm, other in (("D", screen.install_v5), ("C", spec.arm_vetoes["B"])):
            bad = dataclasses.replace(spec, arm_vetoes=dict(spec.arm_vetoes, **{arm: other}))
            assert dev_screen.main(argv, spec=bad) == 2
        assert not (tmp_path / "o").exists()


class TestReports:
    def test_v7_reports_sum_counters_and_rates(self):
        diag = {key: 2 for key in sweep.V7_SUMMED}
        diag.update(decisions=10, rerank_changes=1, area_cap_max=400, apply_seconds_total=0.01)
        entries = [dict(_entry("B", "frozen", s, 1.0), veto_diagnostics=diag) for s in (1, 2)]
        entries.append(_entry("A", "frozen", 1, 1.0))
        report = screen.v7_report(entries, "B")
        assert report["total"]["decisions"] == 20 and report["total"]["rerank_changes"] == 2
        assert report["total"]["rerank_change_rate_per_decision"] == pytest.approx(0.1)
        assert report["total"]["mean_apply_seconds_per_decision"] == pytest.approx(0.001)
        assert report["per_mix"]["frozen"]["area_cap_max"] == 400
        sweep_entries = [dict(e, arm="L200") if e["arm"] == "B" else e for e in entries]
        by_arm = sweep.v7_counter_report(sweep_entries)["by_arm"]
        assert by_arm["L200"]["rerank_changes"] == 2 and "rerank_changes" not in by_arm["A"]
        causes = sweep.death_causes(entries)["by_arm_and_mix"]
        assert causes["B"]["frozen"] == {"self": 2}
