"""Unit tests for the opt-in thermal guard, the third slot and dev_screen's opt-in flags.

Never plays: every test that reaches a harness keeps ``tournament_eval.rollout`` fatal, and
tests only use tmp slot lock roots (the shared slots 1/2 are never touched).
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.apex_safety_20260926 import dev_screen
from research.compute import slot_setup
from research.compute import thermal_guard as tg

NOMINAL = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)
THROTTLED = (
    "Thermal Warning Level = 0\n"
    "Note: No performance warning level has been recorded\n"
    "2026-10-02 10:00:00 +0000 CPU Power notify\n"
    "\tCPU_Scheduler_Limit \t= 100\n"
    "\tCPU_Available_CPUs \t= 18\n"
    "\tCPU_Speed_Limit \t= 70\n"
)


# The pre-review detector setting; mechanics tests use it explicitly (defaults are wider).
FAST = {"baseline_k": 6, "slowdown": 1.30, "window": 3}


def reader(text):
    return lambda: {"platform": "darwin", "gated": True, "available": True, "text": text}


@pytest.fixture
def no_episodes(monkeypatch):
    from src.scripts import tournament_eval

    def fatal(*args, **kwargs):
        raise AssertionError("a guard test must never play an episode")

    monkeypatch.setattr(tournament_eval, "rollout", fatal)
    monkeypatch.setattr(dev_screen, "run_episode", fatal)


# ---------------------------------------------------------------- pmset parser


class TestParser:
    def test_nominal_notes_are_level_zero_and_ok(self):
        parsed = tg.parse_pmset_therm(NOMINAL)
        assert parsed["thermal_warning_level"] == 0
        assert parsed["performance_warning_level"] == 0
        assert parsed["cpu_power_status_note"] is True
        assert parsed["CPU_Speed_Limit"] is None and parsed["unparsed"] == []
        assert tg.thermal_reasons(parsed) == []

    def test_cpu_speed_limit_below_100_is_not_ok(self):
        parsed = tg.parse_pmset_therm(THROTTLED)
        assert parsed["CPU_Speed_Limit"] == 70
        assert parsed["CPU_Scheduler_Limit"] == 100 and parsed["CPU_Available_CPUs"] == 18
        assert parsed["thermal_warning_level"] == 0
        assert tg.thermal_reasons(parsed) == ["CPU_Speed_Limit=70<100"]

    def test_full_speed_limit_block_is_ok(self):
        parsed = tg.parse_pmset_therm(THROTTLED.replace("= 70", "= 100"))
        assert tg.thermal_reasons(parsed) == []

    @pytest.mark.parametrize(
        "line, key, level",
        [
            ("Thermal Warning Level = 2", "thermal_warning_level", 2),
            ("Thermal Warning Level = 5", "thermal_warning_level", 5),
            ("Performance Warning Level = 3", "performance_warning_level", 3),
            ("Performance Warning Level = 5", "performance_warning_level", 5),
        ],
    )
    def test_warning_levels(self, line, key, level):
        text = NOMINAL + line + "\n"
        parsed = tg.parse_pmset_therm(text)
        assert parsed[key] == level
        assert f"{key}={level}" in tg.thermal_reasons(parsed)

    @pytest.mark.parametrize(
        "line",
        [
            "Error:Failed to get thermal warning level with error code 0xe00002c0",
            "Error: Failed to get performance warning level with error code 0xe00002c0",
            "Error: No CPU power status with error code 0xe00002c0",
        ],
    )
    def test_pmset_error_lines_fail_closed(self, line):
        # pmset's own format strings (``strings /usr/bin/pmset``); "0x..." is never a level.
        for text in (line + "\n" + NOMINAL, NOMINAL + line + "\n"):
            parsed = tg.parse_pmset_therm(text)
            assert parsed["unparsed"] == [line]
            assert "pmset_unparsed_lines=1" in tg.thermal_reasons(parsed)
        alone = tg.parse_pmset_therm(line)
        assert alone["thermal_warning_level"] is None  # was 0 (nominal) from "0x..."
        assert alone["performance_warning_level"] is None
        reasons = tg.thermal_reasons(alone)
        assert "pmset_unparsed_lines=1" in reasons
        if "CPU power" in line:
            assert "cpu_power_status_unknown" in reasons

    @pytest.mark.parametrize(
        "line",
        [
            "Thermal Warning Level = -1",
            "thermal warning level: 1",
            "2026-10-02 Performance Warning Level Set To 3",
            "Thermal Warning Level = 2 (elevated)",
        ],
    )
    def test_non_pmset_level_formats_fail_closed(self, line):
        parsed = tg.parse_pmset_therm(NOMINAL + line + "\n")
        assert parsed["unparsed"] == [line]
        assert "pmset_unparsed_lines=1" in tg.thermal_reasons(parsed)

    def test_negative_level_is_unknown(self):
        parsed = {**tg.parse_pmset_therm(NOMINAL), "thermal_warning_level": -1}
        assert tg.thermal_reasons(parsed) == ["thermal_warning_level_unknown"]

    def test_last_level_line_wins(self):
        text = "Thermal Warning Level = 2\nThermal Warning Level = 0\n" + NOMINAL
        assert tg.thermal_reasons(tg.parse_pmset_therm(text)) == []

    def test_missing_lines_fail_closed(self):
        parsed = tg.parse_pmset_therm("")
        assert tg.thermal_reasons(parsed) == [
            "thermal_warning_level_unknown",
            "performance_warning_level_unknown",
            "cpu_power_status_unknown",
        ]
        only_thermal = tg.parse_pmset_therm(NOMINAL.splitlines()[0])
        assert "performance_warning_level_unknown" in tg.thermal_reasons(only_thermal)
        assert "thermal_warning_level_unknown" not in tg.thermal_reasons(only_thermal)

    def test_unparsed_relevant_lines_fail_closed(self):
        parsed = tg.parse_pmset_therm(NOMINAL + "Thermal warning level is elevated\n")
        assert parsed["unparsed"] == ["Thermal warning level is elevated"]
        assert "pmset_unparsed_lines=1" in tg.thermal_reasons(parsed)
        odd = tg.parse_pmset_therm(NOMINAL + "Some new limit flag\n")
        assert odd["unparsed"] == ["Some new limit flag"]

    def test_irrelevant_lines_are_ignored(self):
        parsed = tg.parse_pmset_therm(NOMINAL + "2026-10-02 CPU Power notify\nhello\n")
        assert tg.thermal_reasons(parsed) == []


class TestReader:
    def test_off_darwin_is_not_gated(self):
        out = tg.read_pmset_therm(runner=None, platform="linux")
        assert out == {"platform": "linux", "gated": False, "available": False, "text": None}

    def test_runner_failure_is_recorded_not_raised(self):
        def boom(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="pmset", timeout=10)

        out = tg.read_pmset_therm(runner=boom, platform="darwin")
        assert out["gated"] and not out["available"] and "TimeoutExpired" in out["error"]

    def test_runner_output(self):
        def ok(cmd, **kwargs):
            assert cmd == ["pmset", "-g", "therm"] and kwargs["timeout"] > 0
            return subprocess.CompletedProcess(cmd, 0, stdout=NOMINAL, stderr="")

        assert tg.read_pmset_therm(runner=ok, platform="darwin")["text"] == NOMINAL


# ---------------------------------------------------------------- slowdown and guard


class TestSlowdown:
    def test_needs_full_baseline_and_window(self):
        report = tg.slowdown_report({"A/frozen": [10.0] * 6 + [20.0, 20.0]}, **FAST)
        row = report["keys"]["A/frozen"]
        assert row["baseline_median"] == 10.0 and "ratio" not in row
        assert report["flagged"] == []

    def test_flags_more_than_30_percent(self):
        times = {"A/frozen": [10, 11, 9, 10, 10, 10, 13.2, 13.4, 13.5], "B/frozen": [10] * 9}
        report = tg.slowdown_report(times, **FAST)
        assert report["flagged"] == ["A/frozen"]
        assert report["keys"]["A/frozen"]["ratio"] == pytest.approx(1.34)
        assert report["keys"]["B/frozen"]["ratio"] == 1.0

    def test_exactly_30_percent_is_not_flagged(self):
        assert tg.slowdown_report({"k": [10] * 6 + [13, 13, 13]}, **FAST)["flagged"] == []

    def test_median_ignores_one_slow_outlier(self):
        assert tg.slowdown_report({"k": [10] * 6 + [10, 99, 10]}, **FAST)["flagged"] == []

    def test_acknowledged_evidence_is_ignored(self):
        times = {"k": [10] * 6 + [20, 20, 20]}
        assert tg.slowdown_report(times, acknowledged={"k": 3}, **FAST)["flagged"] == []
        times["k"].extend([20, 20, 20])
        assert tg.slowdown_report(times, acknowledged={"k": 3}, **FAST)["flagged"] == ["k"]

    def test_wide_defaults_ignore_content_variance(self):
        assert (tg.DEFAULT_BASELINE_K, tg.DEFAULT_SLOWDOWN, tg.DEFAULT_WINDOW) == (12, 1.60, 8)
        # Content-driven 10-45 s spread (CV ~0.35): the old 6/1.30/3 setting pauses here.
        times = {"k": [12, 20, 15, 30, 18, 22, 14, 25, 19, 16, 21, 17] + [26, 30, 24, 28] * 2}
        assert tg.slowdown_report(times, **FAST)["flagged"] == ["k"]
        report = tg.slowdown_report(times)
        assert report["flagged"] == [] and report["keys"]["k"]["ratio"] < 1.6
        assert tg.slowdown_report({"k": [20] * 12 + [33] * 8})["flagged"] == ["k"]
        assert tg.slowdown_report({"k": [20] * 12 + [33] * 7})["flagged"] == []

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            tg.ThermalGuard(slowdown=1.0)
        with pytest.raises(ValueError):
            tg.slowdown_report({}, baseline_k=0)
        with pytest.raises(ValueError):
            tg.ThermalGuard(max_slowdown_backoffs=-1)


class TestGuard:
    def test_ok_shape(self):
        guard = tg.ThermalGuard(baseline_k=6, slowdown=1.30, pmset_reader=reader(NOMINAL))
        result = guard.check()
        assert set(result) == {"ok", "reasons", "readings"}
        assert result["ok"] is True and result["reasons"] == []
        assert result["readings"]["pmset"]["parsed"]["thermal_warning_level"] == 0
        assert "text" not in result["readings"]["pmset"]
        json.dumps(result, allow_nan=False)

    def test_throttle_slowdown_battery_and_reader_errors_combine(self):
        def broken_ac():
            raise OSError("no pmset")

        guard = tg.ThermalGuard(**FAST, pmset_reader=reader(THROTTLED), ac_reader=broken_ac)
        for seconds in [10] * 6 + [20] * 3:
            guard.record_episode("B/mixed", seconds)
        result = guard.check()
        assert result["ok"] is False
        assert result["reasons"] == ["CPU_Speed_Limit=70<100", "slowdown:B/mixed", "on_battery"]
        assert result["readings"]["ac_power"] is False

    def test_unavailable_pmset_fails_closed_and_off_darwin_is_skipped(self):
        down = tg.ThermalGuard(pmset_reader=lambda: {"gated": True, "available": False})
        assert down.check()["reasons"] == ["pmset_unavailable"]

        def raises():
            raise RuntimeError("bug")

        assert tg.ThermalGuard(pmset_reader=raises).check()["reasons"] == ["pmset_unavailable"]
        off = tg.ThermalGuard(pmset_reader=lambda: tg.read_pmset_therm(platform="linux"))
        result = off.check()
        assert result["ok"] and result["readings"]["pmset"]["skipped"] == "platform_not_gated"

    def test_acknowledge_clears_slowdown_until_new_evidence(self):
        guard = tg.ThermalGuard(**FAST, pmset_reader=reader(NOMINAL))
        for seconds in [10] * 6 + [20] * 3:
            guard.record_episode("A/frozen", seconds)
        assert guard.check()["reasons"] == ["slowdown:A/frozen"]
        guard.acknowledge_slowdown()
        assert guard.check()["ok"]


class TestAdmission:
    def _run(self, guard, left=10_000.0, budget=45.0, backoff=60.0, max_backoffs=2):
        log, sleeps = [], []
        result = tg.admit_next_episode(
            guard,
            backoff_seconds=backoff,
            max_backoffs=max_backoffs,
            seconds_left=lambda: left,
            budget_seconds=budget,
            log=log.append,
            sleep=sleeps.append,
        )
        return result, log, sleeps

    def test_ok_admits_without_sleeping(self):
        (admit, reason, counts), log, sleeps = self._run(
            tg.ThermalGuard(pmset_reader=reader(NOMINAL))
        )
        assert admit and reason is None and sleeps == []
        assert counts == {"checks": 1, "not_ok_checks": 0, "backoffs": 0, "backoff_seconds": 0}
        assert [e["event"] for e in log] == ["thermal_guard_check"]

    def test_recovers_after_one_backoff(self):
        texts = iter([THROTTLED, NOMINAL])
        guard = tg.ThermalGuard(pmset_reader=lambda: reader(next(texts))())
        (admit, reason, counts), log, sleeps = self._run(guard)
        assert admit and reason is None and sleeps == [60.0]
        assert counts["backoffs"] == 1 and counts["not_ok_checks"] == 1
        events = [e["event"] for e in log]
        assert events == ["thermal_guard_check", "thermal_guard_backoff", "thermal_guard_check"]

    def test_slowdown_is_answered_by_one_pause(self):
        guard = tg.ThermalGuard(**FAST, pmset_reader=reader(NOMINAL))
        for seconds in [10] * 6 + [20] * 3:
            guard.record_episode("A/frozen", seconds)
        (admit, _, counts), _, sleeps = self._run(guard)
        assert admit and sleeps == [60.0] and counts["backoffs"] == 1

    def test_persistent_slowdown_stops_without_another_pause(self):
        guard = tg.ThermalGuard(**FAST, max_slowdown_backoffs=2, pmset_reader=reader(NOMINAL))
        for seconds in [10] * 6:
            guard.record_episode("A/frozen", seconds)
        stops = []
        for streak in range(3):
            for seconds in [20] * 3:
                guard.record_episode("A/frozen", seconds)
            (admit, reason, counts), log, sleeps = self._run(guard, max_backoffs=5)
            stops.append((admit, len(sleeps), counts["backoffs"]))
        assert stops == [(True, 1, 1), (True, 1, 1), (False, 0, 0)]
        assert reason.startswith("thermal: persistent slowdown (A/frozen) still flagged after 2")
        assert log[0]["reasons"] == ["slowdown_persistent:A/frozen"]
        assert log[0]["readings"]["slowdown"]["consecutive_backoffs"] == {"A/frozen": 2}
        assert log[-1] == {"event": "thermal_guard_stop", "reason": reason}

    def test_unflagged_fresh_window_resets_the_streak(self):
        guard = tg.ThermalGuard(**FAST, max_slowdown_backoffs=2, pmset_reader=reader(NOMINAL))
        for seconds in [10] * 6 + [20] * 3:
            guard.record_episode("A/frozen", seconds)
        assert self._run(guard)[0][0] and guard.slowdown_backoffs == {"A/frozen": 1}
        for seconds in [10] * 3:
            guard.record_episode("A/frozen", seconds)
        assert self._run(guard)[2] == [] and guard.slowdown_backoffs == {"A/frozen": 0}
        for _ in range(2):
            for seconds in [20] * 3:
                guard.record_episode("A/frozen", seconds)
            (admit, _, _), _, sleeps = self._run(guard)
            assert admit and sleeps == [60.0]
        assert guard.slowdown_backoffs == {"A/frozen": 2}

    def test_pmset_backoffs_do_not_count_as_slowdown_backoffs(self):
        texts = iter([THROTTLED, NOMINAL] * 3)
        guard = tg.ThermalGuard(**FAST, pmset_reader=lambda: reader(next(texts))())
        for _ in range(3):
            assert self._run(guard)[0][0]
        assert guard.slowdown_backoffs == {}

    def test_budget_exhausted_stops_thermal(self):
        guard = tg.ThermalGuard(pmset_reader=reader(THROTTLED))
        (admit, reason, counts), log, sleeps = self._run(guard, max_backoffs=3)
        assert not admit and sleeps == [60.0] * 3
        assert reason.startswith("thermal: still not ok (CPU_Speed_Limit=70<100) after 3")
        assert counts == {"checks": 4, "not_ok_checks": 4, "backoffs": 3, "backoff_seconds": 180}
        assert log[-1] == {"event": "thermal_guard_stop", "reason": reason}

    def test_backoff_never_crosses_the_deadline_budget(self):
        guard = tg.ThermalGuard(pmset_reader=reader(THROTTLED))
        (admit, reason, _), _, sleeps = self._run(guard, left=100.0, budget=45.0)
        assert not admit and sleeps == [] and reason.startswith("thermal: ")
        assert "would cross the episode budget" in reason


# ---------------------------------------------------------------- third slot (tmp roots only)


def _touch(root, *numbers):
    for k in numbers:
        (root / f"cpu-slot-{k}.lock").touch()


def _held_names(handles):
    return [Path(h.name).name for h in handles]


class TestThirdSlot:
    def test_default_pool_is_unchanged_and_ignores_slot_3(self, tmp_path):
        assert dev_screen.SLOT_LOCK_FILES == ("cpu-slot-1.lock", "cpu-slot-2.lock")
        _touch(tmp_path, 1, 2, 3)
        handles = dev_screen.acquire_cpu_slots(tmp_path, 2, timeout=0)
        try:
            assert _held_names(handles) == ["cpu-slot-1.lock", "cpu-slot-2.lock"]
        finally:
            dev_screen.release_cpu_slots(handles)
        with pytest.raises(ValueError, match=r"slot count must be 1\.\.2"):
            dev_screen.acquire_cpu_slots(tmp_path, 3, timeout=0)

    def test_pool_3_prefers_the_dev_slot_then_falls_back(self, tmp_path):
        _touch(tmp_path, 1, 2, 3)
        first = dev_screen.acquire_cpu_slots(tmp_path, 1, timeout=0, pool=3)
        try:
            assert _held_names(first) == ["cpu-slot-3.lock"]
            second = dev_screen.acquire_cpu_slots(tmp_path, 1, timeout=0, pool=3)
            assert _held_names(second) == ["cpu-slot-1.lock"]
            dev_screen.release_cpu_slots(second)
        finally:
            dev_screen.release_cpu_slots(first)

    def test_pool_3_times_out_when_all_three_are_held(self, tmp_path):
        _touch(tmp_path, 1, 2, 3)
        busy = [(tmp_path / f"cpu-slot-{k}.lock").open("r") for k in (1, 2, 3)]
        try:
            for handle in busy:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(TimeoutError):
                dev_screen.acquire_cpu_slots(tmp_path, 1, timeout=0, pool=3)
        finally:
            dev_screen.release_cpu_slots(busy)

    def test_pool_3_never_creates_slot_3(self, tmp_path):
        _touch(tmp_path, 1, 2)
        with pytest.raises(FileNotFoundError, match="cpu-slot-3.lock"):
            dev_screen.acquire_cpu_slots(tmp_path, 1, timeout=0, pool=3)
        assert not (tmp_path / "cpu-slot-3.lock").exists()
        with pytest.raises(ValueError, match="slot pool"):
            dev_screen.acquire_cpu_slots(tmp_path, 1, timeout=0, pool=4)

    def test_setup_creates_slot_3_only_explicitly(self, tmp_path, capsys):
        with pytest.raises(FileNotFoundError, match="strict slot files missing"):
            slot_setup.create_third_slot(tmp_path)
        _touch(tmp_path, 1, 2)
        assert slot_setup.main(["--root", str(tmp_path)]) == 0  # dry-run status only
        assert not (tmp_path / "cpu-slot-3.lock").exists()
        assert slot_setup.create_third_slot(tmp_path)["created"] is True
        (tmp_path / "cpu-slot-3.lock").write_text("keep")
        again = slot_setup.create_third_slot(tmp_path)
        assert again["created"] is False and (tmp_path / "cpu-slot-3.lock").read_text() == "keep"
        assert slot_setup.main(["--root", str(tmp_path / "nope"), "--create"]) == 2
        capsys.readouterr()


# ---------------------------------------------------------------- dev_screen opt-in flags

# intent.json keys of a default dev_screen run before this change (DEFAULT_SPEC).
ORIGINAL_INTENT_KEYS = {
    "schema_version", "authority", "promotion_authorized", "protocol", "argv", "started_utc",
    "deadline_utc", "git", "config", "profile", "smoke_frames", "mixes", "hero",
    "checkpoint_pool", "checkpoint_snapshots", "arms", "worlds", "disjointness",
    "roster_parity", "preregistered_design", "planned_episodes", "threads",
}  # fmt: skip
ORIGINAL_SUMMARY_TAIL = {"finished_utc", "episodes_run", "planned_episodes", "stopped_reason"}


def _soon(hours=2.0):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _fake_entry(arm, row, world_index, lookup, profile, records_dir, smoke_frames=None, spec=None):
    """A recorded entry without playing (``tournament_eval.rollout`` stays fatal)."""
    mass = 100.0 + world_index + (7.0 + world_index if arm in ("B", "D") else 0.0)
    counters = {"decisions": 10, "vetoes_applied": 1}
    return {
        "arm": arm,
        "mix": row["mix"],
        "world_seed": row["world_seed"],
        "wall_seconds": 1.0,
        "record": {
            "mass_integral": mass,
            "survival_fraction": 1.0,
            "probes": {"death_cause": None, "safety_veto": {"counters": counters}},
            "denominators": {},
        },
    }


@pytest.fixture
def harness(monkeypatch, no_episodes):
    """Stub everything around the loop; ``run_episode`` stays fatal unless a test fakes it."""
    monkeypatch.setattr(dev_screen, "snapshot_checkpoints", lambda directory, out: {})
    monkeypatch.setattr(dev_screen, "agent_lookup", lambda snapshots: {})
    monkeypatch.setattr(
        dev_screen,
        "_design_rows",
        lambda seeds: [
            {"mix": mix, "world_seed": seed, "slots": []}
            for mix in dev_screen.MIXES
            for seed in seeds
        ],
    )
    monkeypatch.setattr(dev_screen, "_git_state", lambda: {"commit": "c" * 40, "dirty_paths": ""})
    sleeps = []
    monkeypatch.setattr(dev_screen, "_thermal_sleep", sleeps.append)

    def run(tmp_path, extra=(), design=(2, 1), spec=dev_screen.DEFAULT_SPEC, use_locks=False):
        out = tmp_path / "out"
        argv = ["--out", str(out), "--deadline-utc", _soon(), "--worlds-per-mix", str(design[0])]
        argv += ["--determinism-worlds", str(design[1]), *extra]
        args = dev_screen.parse_args(argv)
        seeds = list(range(1000, 1000 + design[0]))
        code = dev_screen._run_screen(
            args, argv, spec, out, seeds, {"disjoint": True}, {}, None, use_locks, False
        )
        assert code == 0
        events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
        return (
            json.loads((out / "intent.json").read_text()),
            json.loads((out / "summary.json").read_text()),
            events,
        )

    run.sleeps = sleeps
    return run


def _guard_with(monkeypatch, texts):
    """Make ``--thermal-guard`` read ``texts`` in order (the last one repeats)."""
    texts = list(texts)

    def make(ac_reader):
        def read():
            return reader(texts.pop(0) if len(texts) > 1 else texts[0])()

        return tg.ThermalGuard(pmset_reader=read, ac_reader=ac_reader)

    monkeypatch.setattr(dev_screen, "make_thermal_guard", make)


class TestDevScreenDefaults:
    def test_new_flags_default_off(self):
        args = dev_screen.parse_args(["--out", "x", "--deadline-utc", _soon()])
        assert (args.slot_pool, args.thermal_guard) == (2, False)
        assert (args.thermal_backoff_seconds, args.thermal_max_backoffs) == (60.0, 5)
        with pytest.raises(SystemExit):
            dev_screen.parse_args(["--out", "x", "--deadline-utc", _soon(), "--slot-pool", "4"])
        with pytest.raises(SystemExit):
            dev_screen.parse_args(
                ["--out", "x", "--deadline-utc", _soon(), "--thermal-backoff-seconds", "0"]
            )

    def test_default_run_records_are_unchanged(self, harness, tmp_path, monkeypatch):
        monkeypatch.setattr(dev_screen, "run_episode", _fake_entry)

        def no_guard(ac_reader):
            raise AssertionError("the guard must not be built without --thermal-guard")

        monkeypatch.setattr(dev_screen, "make_thermal_guard", no_guard)
        intent, summary, events = harness(tmp_path, use_locks=True)
        assert set(intent) == ORIGINAL_INTENT_KEYS | {"slot_locks"}
        assert set(intent["slot_locks"]) == {"root", "held", "held_from"}
        assert "thermal_guard" not in summary and ORIGINAL_SUMMARY_TAIL <= set(summary)
        assert summary["decision_rule"] == dev_screen.decision_rule_text()
        assert len(events) == intent["planned_episodes"] == 15
        assert all("event" not in line for line in events)
        assert harness.sleeps == []

    @pytest.mark.parametrize("module", ["v5_screen", "v6_screen", "v7_screen"])
    def test_existing_screen_specs_are_unchanged(self, harness, tmp_path, monkeypatch, module):
        from research.apex_veto_v5_screen_20261001 import screen as v5_screen
        from research.apex_veto_v6_screen_20261002 import screen as v6_screen
        from research.apex_veto_v7_screen_20261002 import screen as v7_screen

        binding = {"summary_sha256": "s" * 64, "source_commit": "c" * 40}
        spec = {
            "v5_screen": lambda: v5_screen.SPEC,
            "v6_screen": lambda: v6_screen.SPEC,
            "v7_screen": lambda: v7_screen.build_spec(v7_screen.DOMAIN, 4.0, binding),
        }[module]()
        monkeypatch.setattr(dev_screen, "run_episode", _fake_entry)
        intent, summary, events = harness(tmp_path, spec=spec, use_locks=True)
        assert "thermal_guard" not in intent and "thermal_guard" not in summary
        assert set(intent["slot_locks"]) == {"root", "held", "held_from"}
        assert intent["decision_rule"] == dev_screen.decision_rule_text(spec.replay_worlds)
        assert all("event" not in line for line in events)

    def test_main_refuses_slot_pool_3_without_locks(self, no_episodes, tmp_path, capsys):
        argv = ["--out", str(tmp_path / "out"), "--deadline-utc", _soon(), "--slot-pool", "3"]
        assert dev_screen.main(argv) == 2
        assert "--slot-pool needs slot locks" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()

    def test_main_refuses_slot_pool_3_without_thermal_guard(
        self, no_episodes, tmp_path, capsys, monkeypatch
    ):
        def no_slots(*args, **kwargs):
            raise AssertionError("no slot may be acquired")

        monkeypatch.setattr(dev_screen, "acquire_cpu_slots", no_slots)
        base = ["--deadline-utc", _soon(), "--use-slot-locks", "--slot-pool", "3"]
        base += ["--slot-lock-root", str(tmp_path / "locks")]
        assert dev_screen.main(["--out", str(tmp_path / "out"), *base]) == 2
        assert "--slot-pool 3 needs --thermal-guard" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()
        # With the guard the pool check passes (stopped next by the existing --out).
        (tmp_path / "exists").mkdir()
        assert dev_screen.main(["--out", str(tmp_path / "exists"), *base, "--thermal-guard"]) == 2
        err = capsys.readouterr().err
        assert "already exists" in err and "--thermal-guard" not in err

    def test_slot_pool_3_is_recorded_in_intent(self, harness, tmp_path, monkeypatch):
        monkeypatch.setattr(dev_screen, "run_episode", _fake_entry)
        intent, _, _ = harness(tmp_path, extra=["--slot-pool", "3"], use_locks=True)
        assert intent["slot_locks"]["pool"] == list(dev_screen.SLOT_POOL_3_FILES)


class TestDevScreenThermalGuard:
    def test_backoff_then_resume_is_recorded(self, harness, tmp_path, monkeypatch):
        monkeypatch.setattr(dev_screen, "run_episode", _fake_entry)
        _guard_with(monkeypatch, [THROTTLED, NOMINAL])
        intent, summary, events = harness(tmp_path, extra=["--thermal-guard"])
        assert intent["thermal_guard"]["backoff_seconds"] == 60.0
        assert intent["thermal_guard"]["max_backoffs"] == 5
        assert harness.sleeps == [60.0]
        guard = summary["thermal_guard"]
        assert guard["backoffs"] == 1 and guard["stopped"] is False
        assert guard["checks"] == summary["planned_episodes"] + 1
        assert summary["episodes_run"] == summary["planned_episodes"]
        assert summary["stopped_reason"] is None
        kinds = [line.get("event") for line in events]
        assert kinds[:3] == ["thermal_guard_check", "thermal_guard_backoff", "thermal_guard_check"]
        assert kinds.count(None) == summary["planned_episodes"]  # episode lines unchanged
        first = events[0]
        assert first["ok"] is False and first["reasons"] == ["CPU_Speed_Limit=70<100"]
        assert first["readings"]["pmset"]["parsed"]["CPU_Speed_Limit"] == 70

    def test_guard_stop_is_incomplete_thermal_without_playing(self, harness, tmp_path, monkeypatch):
        # run_episode stays fatal: the guard must stop before any episode is admitted.
        _guard_with(monkeypatch, [THROTTLED])
        intent, summary, events = harness(
            tmp_path,
            extra=["--thermal-guard", "--thermal-max-backoffs", "2"],
            design=dev_screen.PREREGISTERED_DESIGN,
        )
        assert harness.sleeps == [60.0, 60.0]
        assert summary["decision"] == "INCOMPLETE" and summary["episodes_run"] == 0
        assert summary["stopped_reason"].startswith("thermal: still not ok (CPU_Speed_Limit")
        guard = summary["thermal_guard"]
        assert guard["stopped"] is True and guard["stop_reason"] == summary["stopped_reason"]
        assert guard["backoffs"] == 2 and guard["not_ok_checks"] == 3
        assert guard["last_check"]["ok"] is False
        assert events[-1] == {
            "done": 0,
            "event": "thermal_guard_stop",
            "reason": summary["stopped_reason"],
            "utc": events[-1]["utc"],
        }

    def test_mid_run_stop_keeps_played_episodes(self, harness, tmp_path, monkeypatch):
        monkeypatch.setattr(dev_screen, "run_episode", _fake_entry)
        _guard_with(monkeypatch, [NOMINAL, NOMINAL, NOMINAL, THROTTLED])
        _, summary, _ = harness(
            tmp_path,
            extra=["--thermal-guard", "--thermal-max-backoffs", "1"],
            design=dev_screen.PREREGISTERED_DESIGN,
        )
        assert summary["episodes_run"] == 3 and summary["decision"] == "INCOMPLETE"
        assert summary["stopped_reason"].startswith("thermal: ")
        assert harness.sleeps == [60.0]

    def test_ac_reader_is_wired_only_with_require_ac_power(self, harness, tmp_path, monkeypatch):
        seen = []

        def make(ac_reader):
            seen.append(ac_reader)
            return tg.ThermalGuard(pmset_reader=reader(NOMINAL), ac_reader=lambda: False)

        monkeypatch.setattr(dev_screen, "make_thermal_guard", make)
        _, summary, _ = harness(
            tmp_path, extra=["--thermal-guard", "--require-ac-power", "--thermal-max-backoffs", "0"]
        )
        assert seen == [dev_screen.on_ac_power]
        assert summary["stopped_reason"].startswith("thermal: still not ok (on_battery)")
        assert harness.sleeps == []
