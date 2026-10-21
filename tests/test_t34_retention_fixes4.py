"""FIX-FIRST #4: synthetic timing, ledger, and host-baseline contracts only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.t34_pathb import analyze_v2 as analysis
from research.t34_pathb import host_guard_v2, registration_retention_v2, retention_v2, timing_v2
from tests.test_t34_pathb_analysis_v2 import _host_summary, _records, _timing
from tests.test_t34_pathb_registration_retention_v2 import _draft_inputs

POLICY = json.loads(Path("research/t34_pathb/RETENTION_PROFILE_V2.json").read_text())["load_policy"]


@pytest.mark.parametrize("phase,cap", [("timing", 3600), ("precheck", 28800), ("study", 144000)])
def test_ledgers_accept_only_their_own_phase_cap(phase: str, cap: int) -> None:
    assert analysis.PHASE_CAP_SECONDS[phase] == cap
    ledger = _host_summary(phase=phase, contended_seconds=5)
    assert analysis._registered_contention_fraction(ledger, phase=phase) == 5 / cap
    for other in analysis.PHASE_CAP_SECONDS:
        if other != phase:
            with pytest.raises(analysis.InvalidRetention, match="phase cap"):
                analysis._registered_contention_fraction(_host_summary(phase=other), phase=phase)
    ledger["elapsed_seconds"] = cap
    with pytest.raises(analysis.InvalidRetention, match="elapsed cap"):
        analysis._registered_contention_fraction(ledger, phase=phase)


@pytest.mark.parametrize("bad_cap", [None, 0, 40, 2])
def test_draft_validator_rejects_missing_or_changed_timing_cap(monkeypatch, tmp_path, bad_cap):
    draft, _ = _draft_inputs(tmp_path, monkeypatch)
    assert draft["timing_max_elapsed_hours"] == retention_v2.TIMING_MAX_ELAPSED_HOURS == 1.0
    draft["timing_max_elapsed_hours"] = bad_cap
    with pytest.raises(retention_v2.RegistrationError, match="timing_max_elapsed_hours"):
        registration_retention_v2.validate_registration(draft, root=tmp_path)


@pytest.mark.parametrize("bad_cap", [None, 40])
def test_runtime_validator_rejects_missing_or_changed_timing_cap(monkeypatch, bad_cap):
    from research.t34_pathb import source_binding_v2

    monkeypatch.setattr(source_binding_v2, "verify_frozen_worktree", lambda *_: None)
    registration = {
        "schema": "t34-pathb-v2-registration/v1",
        "ratified": True,
        "source_commit": "a" * 40,
        "arms": list(retention_v2.ARMS),
        "mac_arms": list(retention_v2.MAC_ARMS),
        "unpaced_colab": retention_v2.unpaced_colab_policy(),
        "paced_fps": 24,
        "bare_fps": 0,
        "terminal_per_mix": 48,
        "product_sessions": 32,
        "horizon": 5000,
        "workers": 1,
        "load_profile": POLICY,
        "max_elapsed_hours": 40,
        "max_cpu_hours": 60,
        "precheck_max_elapsed_hours": 8,
        "timing_max_elapsed_hours": bad_cap,
    }
    with pytest.raises(retention_v2.RegistrationError, match="cap differs"):
        retention_v2.require_registration(registration)


def test_profile_timing_cap_must_match_registration(monkeypatch, tmp_path):
    draft, _ = _draft_inputs(tmp_path, monkeypatch)
    path = tmp_path / "profile.json"
    profile = json.loads(path.read_text())
    profile["timing_max_elapsed_hours"] = 40
    path.write_text(json.dumps(profile))
    draft["profile"]["sha256"] = registration_retention_v2._sha256(path)
    with pytest.raises(retention_v2.RegistrationError, match="profile timing_max_elapsed_hours"):
        registration_retention_v2.validate_registration(draft, root=tmp_path)


@pytest.mark.parametrize(
    "cpu_count,load,baseline", [(8, 8.0, 2.0), (4, 0.5, 0.5), (None, 8.0, 0.25)]
)
def test_idle_baseline_has_absolute_cpu_count_ceiling(monkeypatch, cpu_count, load, baseline):
    now = [0.0]
    monkeypatch.setattr(host_guard_v2.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(host_guard_v2.os, "getloadavg", lambda: (load, 0, 0))
    monkeypatch.setattr(host_guard_v2, "_power_thermal", lambda: (True, True, True))
    monkeypatch.setattr(host_guard_v2, "_own_pids", lambda: set())
    monkeypatch.setattr(host_guard_v2, "_processes", lambda: [])
    guard = host_guard_v2.HostGuard(POLICY, clock=lambda: now[0])
    first = guard._measure()
    assert first["idle_load_1m"] == baseline
    assert first["load_1m_threshold"] == baseline + 3
    now[0] = 65
    later = guard._measure()
    assert later["idle_load_1m"] == baseline
    if load > baseline + 3:
        assert first["quiet"] is False
        assert later["contended"] is True
        with pytest.raises(host_guard_v2.HostAdmissionError):
            guard.admit()


@pytest.mark.parametrize("arm", ["bare", "paced"])
@pytest.mark.parametrize("overrun", [0, 1])
def test_one_scored_sample_requires_retry_even_below_72_seconds(arm, overrun):
    timing = _timing()
    timing["host_summary"] = _host_summary(contended_seconds=5)
    timing["probes"][f"{arm}-24"]["host_contended_ticks"] = [False] * 9 + [True]
    timing["probes"]["paced-24"]["overruns"] = overrun
    passed, reason = retention_v2._timing_passes(timing)
    assert passed is False
    assert reason.startswith("INCONCLUSIVE-retry")
    # Timing invalidity must be discovered before consuming any efficacy rows.
    assert analysis.decide([], timing, complete=False)["verdict"] == "INCONCLUSIVE-retry"
    assert analysis.decide(_records(), timing)["verdict"] == "INCONCLUSIVE-retry"
    with pytest.raises(retention_v2.RegistrationError, match="INCONCLUSIVE-retry"):
        retention_v2.require_timing_gate(timing)


def test_warmup_contention_is_separate_from_scored_probe_ticks(monkeypatch):
    monkeypatch.setattr(timing_v2.time, "sleep", lambda _: None)
    samples = iter([True, False, False])
    session = SimpleNamespace(step=lambda: None, last_update_ms=1)
    result = timing_v2.probe(
        session,
        fps=24,
        ticks=2,
        warmup=1,
        check_host=lambda: {"contended": next(samples)},
    )
    assert result["warmup_host_contended_ticks"] == [True]
    assert result["host_contended_ticks"] == [False, False]
    assert result["contended"] is False


def _fake_gate(monkeypatch, tmp_path, *, refuse=False, host_records=None):
    now = [100.0]
    caps = []
    monkeypatch.setattr(retention_v2.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(retention_v2.time, "process_time", lambda: 1.0)
    for name in ("require_registration", "_require_committed_pushed_registration"):
        monkeypatch.setattr(retention_v2, name, lambda *_: None)
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    monkeypatch.setattr(retention_v2, "_final_source_check", lambda: None)

    samples = [] if host_records is None else host_records

    class Guard:
        records = samples
        admission_records = [{"quiet": not refuse}]

        def __init__(self, _policy, *, phase_cap_seconds):
            caps.append(phase_cap_seconds)

        def admit(self):
            if refuse:
                raise host_guard_v2.HostAdmissionError("busy")

        def start(self):
            self.check_budget()

        def close(self):
            pass

        def between_rows(self, **_kwargs):
            pass

        def check(self):
            return {"contended": False}

        def summary(self, **_kwargs):
            return _host_summary()

    monkeypatch.setattr(retention_v2, "HostGuard", Guard)
    registration = {
        "study_root": str(tmp_path),
        "history_ledger": {
            "repo": str(tmp_path),
            "path": f"research/t34_pathb/ledger/{tmp_path.name}.jsonl",
        },
        "timing_attempt_parent": str(tmp_path / "timing"),
        "timing_max_attempts": 3,
        "ratified": False,
        "source_commit": "a" * 40,
        "source_hashes": {},
        "load_profile": POLICY,
        "timing_row": {},
        "timing_ticks": 10,
        "timing_warmup": 0,
    }
    return registration, now, caps


@pytest.mark.parametrize("refuse", [False, True])
def test_each_timing_attempt_is_recorded_and_cannot_be_overwritten(monkeypatch, tmp_path, refuse):
    registration, _, caps = _fake_gate(monkeypatch, tmp_path, refuse=refuse)
    timing = _timing()
    timing["probes"]["paced-24"]["contended"] = True
    monkeypatch.setattr(retention_v2.timing_v2, "run_probes", lambda *_a, **_k: timing)
    for name in ("attempt-01", "attempt-02"):
        out = tmp_path / "timing" / name
        if refuse:
            with pytest.raises(host_guard_v2.HostAdmissionError):
                retention_v2.run_timing_gate(out, registration, registration_path=Path(__file__))
            assert not (out / "decision.json").exists()
        else:
            result = retention_v2.run_timing_gate(
                out, registration, registration_path=Path(__file__)
            )
            assert result["status"] == "INCONCLUSIVE-retry"
        record = json.loads((out / "timing_attempt.json").read_text())
        assert record["attempt_id"] == str(out.resolve())
        assert record["status"] == ("ADMISSION_REFUSED" if refuse else "INCONCLUSIVE-retry")
        assert record["efficacy_rows_produced"] is record["automatic_retry"] is False
        assert not (out / "records").exists()
        before = (out / "timing_attempt.json").read_bytes()
        with pytest.raises(retention_v2.RegistrationError, match="fresh"):
            retention_v2.run_timing_gate(out, registration, registration_path=Path(__file__))
        assert (out / "timing_attempt.json").read_bytes() == before
    assert caps == [3600, 3600]


def test_timing_one_hour_cap_is_checked_inside_probe_and_records_failure(monkeypatch, tmp_path):
    registration, now, caps = _fake_gate(monkeypatch, tmp_path)

    def probes(*_args, check_cap, **_kwargs):
        now[0] += 3599
        check_cap()
        now[0] += 1
        check_cap()
        pytest.fail("timing continued past one hour")

    monkeypatch.setattr(retention_v2.timing_v2, "run_probes", probes)
    out = tmp_path / "timing" / "attempt-01"
    with pytest.raises(TimeoutError, match="1-hour timing elapsed cap"):
        retention_v2.run_timing_gate(out, registration, registration_path=Path(__file__))
    assert caps == [3600]
    assert json.loads((out / "decision.json").read_text())["verdict"] == "KILL"
    assert json.loads((out / "timing_attempt.json").read_text())["status"] == "KILL"
    assert not (out / "records").exists()


@pytest.mark.parametrize("boundary_failure", [False, True])
def test_final_scored_sample_is_preserved_through_probe_or_boundary_exit(
    monkeypatch, tmp_path, boundary_failure
):
    records = []
    registration, now, _ = _fake_gate(monkeypatch, tmp_path, host_records=records)
    steps = [0]

    def step():
        steps[0] += 1
        if steps[0] == 40:
            # Published inside the last scored tick, after its cached check.
            records.append(
                {"timestamp_monotonic": now[0], "contended": True, "reasons": ["thermal_not_ok"]}
            )

    def boundary(_guard, _session=None):
        if steps[0] == 40 and boundary_failure:
            error = host_guard_v2.HostContentionError("power persisted at row boundary")
            error.measured_records = records[-1:]
            raise error

    monkeypatch.setattr(retention_v2, "_between_rows", boundary)
    monkeypatch.setattr(
        timing_v2.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    adapter = SimpleNamespace(
        open_watch=lambda *_: SimpleNamespace(
            step=step,
            last_update_ms=1,
            close=lambda **_: None,
        )
    )
    out = tmp_path / "timing" / "attempt-01"
    result = retention_v2.run_timing_gate(
        out, registration, adapter=adapter, registration_path=Path(__file__)
    )
    assert result["status"] == "INCONCLUSIVE-retry"
    attempt = json.loads((out / "timing_attempt.json").read_text())
    assert attempt["status"] == "INCONCLUSIVE-retry"
    assert attempt["scored_host_contended"] is True
    assert not (out / "records").exists()
    if not boundary_failure:
        timing = json.loads((out / "timing_gate.json").read_text())
        # All beginning-of-tick reads were quiet; the window catches the tail.
        assert all(not probe["contended"] for probe in timing["probes"].values())
        assert timing["scored_host_contended"] is True
        assert analysis.decide([], timing, complete=False)["verdict"] == "INCONCLUSIVE-retry"


def test_timing_guard_budget_is_exactly_72_seconds():
    guard = host_guard_v2.HostGuard(POLICY, phase_cap_seconds=analysis.PHASE_CAP_SECONDS["timing"])
    assert guard.summary()["contention_budget_seconds"] == 72


@pytest.fixture(autouse=True)
def _isolate_published_history(monkeypatch):
    """These focused local tests isolate the FIX6 Git transport boundary.

    Actual publication, receipt deletion and tampering are exercised against
    a real temporary Git remote in test_t34_retention_fixes6.py.
    """
    from research.t34_pathb import retention_v2 as runner

    monkeypatch.setattr(runner, "require_published_history", lambda *_: None)
    monkeypatch.setattr(runner, "finalize_phase", lambda *_: None)
    monkeypatch.setattr(runner, "claim_phase", lambda *_: None)
