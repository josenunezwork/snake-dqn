"""FIX-FIRST #5: synthetic, fixed-parent timing-history admission regressions."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from research.t34_pathb import registration_retention_v2 as packaging
from research.t34_pathb import retention_v2 as runner
from tests.test_t34_pathb_analysis_v2 import _host_summary, _timing
from tests.test_t34_pathb_registration_retention_v2 import _draft_inputs
from tests.test_t34_retention_fixes4 import _fake_gate


def _registration(tmp_path):
    return {
        "study_root": str(tmp_path),
        "history_ledger": {
            "repo": str(tmp_path),
            "path": f"research/t34_pathb/ledger/{tmp_path.name}.jsonl",
        },
        "timing_attempt_parent": str(tmp_path / "timing"),
        "timing_max_attempts": 3,
        "source_commit": "a" * 40,
        "source_hashes": {"source.py": "b" * 64},
        "timing_row": {"seed": 1, "mix": "mixed"},
        "timing_ticks": 10,
        "timing_warmup": 0,
    }


def _attempt(registration, number, status):
    path = Path(registration["timing_attempt_parent"]) / f"attempt-{number:02d}"
    path.mkdir(parents=True)
    evidence = {}
    timing = None
    if status == "PASS":
        timing = {
            **_timing(),
            **{
                key: registration[key]
                for key in (
                    "source_commit",
                    "source_hashes",
                    "timing_row",
                    "timing_ticks",
                    "timing_warmup",
                )
            },
            "status": "PASS",
            "host_summary": _host_summary(),
            "elapsed_seconds": 10.0,
            "cpu_seconds": 1.0,
            "source_validation_path": str(path / "source_validated.json"),
            "registration_sha256": runner._canonical_sha(registration),
        }
        for probe in timing["probes"].values():
            probe["warmup"] = 0
        runner._write_new(path / "timing_gate.json", timing)
        runner._write_new(
            path / "source_validated.json",
            {
                "schema": "t34-pathb-v2-source-validation/v1",
                "status": "PASS",
                "source_commit": registration["source_commit"],
                "receipts": {"timing_gate.json": runner._canonical_sha(timing)},
            },
        )
        evidence["timing_gate.json"] = runner._sha(path / "timing_gate.json")
    elif status != "ADMISSION_REFUSED":
        runner._write_new(path / "decision.json", {"verdict": status})
        evidence["decision.json"] = runner._sha(path / "decision.json")
    runner._write_new(
        path / "timing_attempt.json",
        {
            "schema": "t34-pathb-v2-timing-attempt/v1",
            "attempt_id": str(path),
            "source_commit": registration["source_commit"],
            "registration_sha256": runner._canonical_sha(registration),
            "status": status,
            "evidence_sha256": evidence,
        },
    )
    runner._write_new(
        path / "timing_finalized.json",
        {
            "schema": "t34-pathb-v2-timing-finalized/v1",
            "attempt_sha256": runner._sha(path / "timing_attempt.json"),
            "source_failed": False,
        },
    )
    return path, timing


@pytest.mark.parametrize(
    "prior",
    [
        [],
        ["INCONCLUSIVE-retry"],
        ["ADMISSION_REFUSED"],
        ["ADMISSION_REFUSED", "INCONCLUSIVE-retry"],
    ],
)
def test_latest_pass_after_only_authorized_predecessors_admits(tmp_path, prior):
    registration = _registration(tmp_path)
    for number, status in enumerate(prior, 1):
        _attempt(registration, number, status)
    _, timing = _attempt(registration, len(prior) + 1, "PASS")
    runner.require_timing_gate(timing, registration)
    with pytest.raises(runner.RegistrationError, match="already passed|cap"):
        runner._reserve_timing_attempt(
            tmp_path / "timing" / f"attempt-{len(prior) + 2:02d}", registration
        )


@pytest.mark.parametrize("terminal", ["KILL-timing", "INCONCLUSIVE", "KILL", "INCOMPLETE"])
@pytest.mark.parametrize("phase", ["timing", "precheck", "efficacy"])
def test_terminal_history_anywhere_blocks_all_admission(tmp_path, terminal, phase):
    registration = _registration(tmp_path)
    _attempt(registration, 1, terminal)
    _, timing = _attempt(registration, 2, "PASS")
    with pytest.raises(runner.RegistrationError, match="terminal timing history"):
        if phase == "timing":
            runner._reserve_timing_attempt(tmp_path / "timing" / "attempt-03", registration)
        elif phase == "precheck":
            runner.run_precheck(
                [],
                timing_gate=timing,
                registration=registration,
                registration_path=Path(__file__),
                out=tmp_path / "precheck",
            )
        else:
            runner.run_registered(
                tmp_path / "efficacy",
                registration,
                {},
                timing_gate=timing,
                registration_path=Path(__file__),
            )
    assert not (tmp_path / "efficacy").exists()
    assert not (tmp_path / "precheck").exists()
    assert not (tmp_path / "timing" / "attempt-03").exists()


@pytest.mark.parametrize("later", ["INCONCLUSIVE-retry", "ADMISSION_REFUSED", "PASS"])
def test_older_pass_cannot_be_supplied_after_later_attempt(tmp_path, later):
    registration = _registration(tmp_path)
    _, old = _attempt(registration, 1, "PASS")
    _attempt(registration, 2, later)
    with pytest.raises(runner.RegistrationError, match="latest|LATEST|earlier timing"):
        runner.require_timing_gate(old, registration)


@pytest.mark.parametrize("status", ["INCONCLUSIVE-retry", "ADMISSION_REFUSED"])
def test_three_attempt_cap_refuses_fourth_before_host_or_world(monkeypatch, tmp_path, status):
    registration, _, _ = _fake_gate(monkeypatch, tmp_path)
    for number in range(1, 4):
        _attempt(registration, number, status)
    monkeypatch.setattr(runner, "HostGuard", lambda *_a, **_k: pytest.fail("host admitted"))
    with pytest.raises(runner.RegistrationError, match="cap is 3"):
        runner.run_timing_gate(
            tmp_path / "timing" / "attempt-04", registration, registration_path=Path(__file__)
        )
    assert not (tmp_path / "timing" / "attempt-04").exists()


@pytest.mark.parametrize(
    "relative",
    ["elsewhere/attempt-01", "timing/attempt-02", "timing/attempt-1", "timing/attempt-00"],
)
def test_attempt_must_use_fixed_parent_and_next_number(tmp_path, relative):
    registration = _registration(tmp_path)
    with pytest.raises(runner.RegistrationError, match="next registered directory"):
        runner._reserve_timing_attempt(tmp_path / relative, registration)
    assert not (tmp_path / relative).exists()


def test_concurrent_reservation_has_one_winner_and_preserves_existing_bytes(tmp_path):
    registration = _registration(tmp_path)
    out = tmp_path / "timing" / "attempt-01"

    def reserve():
        try:
            runner._reserve_timing_attempt(out, registration)
            return "reserved"
        except runner.RegistrationError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: reserve(), range(2))) == ["refused", "reserved"]
    marker = out / "timing_attempt.json"
    marker.write_text("preserved incomplete reservation")
    assert reserve() == "refused"
    assert marker.read_text() == "preserved incomplete reservation"


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "malformed",
        "gap",
        "symlink",
        "unknown",
        "registration",
        "source",
        "verdict",
        "evidence",
        "source_failure",
        "four_entries",
    ],
)
def test_corrupt_or_unfinished_history_fails_closed(tmp_path, corruption):
    registration = _registration(tmp_path)
    path, timing = _attempt(registration, 1, "PASS")
    marker = path / "timing_attempt.json"
    record = json.loads(marker.read_text())
    if corruption == "missing":
        marker.unlink()
    elif corruption == "malformed":
        marker.write_text("{bad json")
    elif corruption == "gap":
        path.rename(path.with_name("attempt-03"))
    elif corruption == "symlink":
        moved = tmp_path / "moved"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=True)
    elif corruption == "unknown":
        (path.parent / "other").mkdir()
    elif corruption in ("registration", "source"):
        record["registration_sha256" if corruption == "registration" else "source_commit"] = (
            "c" * 64
        )
        marker.write_text(json.dumps(record))
    elif corruption == "verdict":
        record["status"] = "INCONCLUSIVE-retry"
        marker.write_text(json.dumps(record))
        finalized_path = path / "timing_finalized.json"
        finalized = runner._load(finalized_path)
        finalized["attempt_sha256"] = runner._sha(marker)
        finalized_path.write_text(json.dumps(finalized))
    elif corruption == "evidence":
        (path / "timing_gate.json").write_text("{}")
    elif corruption == "source_failure":
        (path / "source_failure.json").write_text("{}")
    else:
        for number in range(2, 5):
            _attempt(registration, number, "ADMISSION_REFUSED")
    with pytest.raises(runner.RegistrationError):
        runner.require_timing_gate(timing, registration)
    with pytest.raises(runner.RegistrationError):
        runner._reserve_timing_attempt(path.parent / "attempt-02", registration)


@pytest.mark.parametrize(
    "field,value",
    [
        ("study_root", None),
        ("study_root", "relative"),
        ("timing_attempt_parent", None),
        ("timing_attempt_parent", "/tmp/elsewhere"),
        ("timing_max_attempts", None),
        ("timing_max_attempts", 4),
        ("timing_max_attempts", True),
    ],
)
def test_registration_rejects_missing_or_changed_history_policy(
    monkeypatch, tmp_path, field, value
):
    draft, _ = _draft_inputs(tmp_path, monkeypatch)
    draft[field] = value
    with pytest.raises(runner.RegistrationError, match="timing|study_root"):
        packaging.validate_registration(draft, root=tmp_path)


def test_pass_copied_elsewhere_cannot_replace_registered_receipt(tmp_path):
    registration = _registration(tmp_path)
    _, timing = _attempt(registration, 1, "PASS")
    timing["source_validation_path"] = str(tmp_path / "elsewhere" / "source_validated.json")
    with pytest.raises(runner.RegistrationError, match="LATEST"):
        runner.require_timing_gate(timing, registration)


@pytest.mark.parametrize("prior", ["INCONCLUSIVE-retry", "ADMISSION_REFUSED"])
def test_real_producer_records_latest_pass_after_retryable_history(monkeypatch, tmp_path, prior):
    registration, _, _ = _fake_gate(monkeypatch, tmp_path)
    _attempt(registration, 1, prior)
    timing = _timing()
    for probe in timing["probes"].values():
        probe["warmup"] = 0
    monkeypatch.setattr(runner.timing_v2, "run_probes", lambda *_a, **_k: timing)
    out = tmp_path / "timing" / "attempt-02"
    assert (
        runner.run_timing_gate(out, registration, registration_path=Path(__file__))["status"]
        == "PASS"
    )
    receipt = runner._load(out / "timing_gate.json")
    # _fake_gate intentionally bypasses the frozen execution decorator; emulate
    # only its final certificate, leaving real history and admission checks live.
    runner._write_new(
        out / "source_validated.json",
        {
            "schema": "t34-pathb-v2-source-validation/v1",
            "status": "PASS",
            "source_commit": registration["source_commit"],
            "receipts": {"timing_gate.json": runner._canonical_sha(receipt)},
        },
    )
    runner.require_timing_gate(receipt, registration)


def test_real_timing_failure_is_terminal_before_another_host_admission(monkeypatch, tmp_path):
    registration, _, _ = _fake_gate(monkeypatch, tmp_path)
    timing = _timing()
    timing["probes"]["paced-24"]["overruns"] = 1
    monkeypatch.setattr(runner.timing_v2, "run_probes", lambda *_a, **_k: timing)
    out = tmp_path / "timing" / "attempt-01"
    assert (
        runner.run_timing_gate(out, registration, registration_path=Path(__file__))["status"]
        == "KILL-timing"
    )
    assert runner._load(out / "timing_gate.json")["status"] == "KILL-timing"
    assert runner._load(out / "timing_attempt.json")["status"] == "KILL-timing"
    monkeypatch.setattr(runner, "HostGuard", lambda *_a, **_k: pytest.fail("host admitted"))
    with pytest.raises(runner.RegistrationError, match="terminal timing history"):
        runner.run_timing_gate(
            out.with_name("attempt-02"), registration, registration_path=Path(__file__)
        )


def test_hidden_terminal_decision_cannot_be_relabelled_retry(tmp_path):
    registration = _registration(tmp_path)
    path, _ = _attempt(registration, 1, "KILL-timing")
    marker = path / "timing_attempt.json"
    record = runner._load(marker)
    record["status"] = "INCONCLUSIVE-retry"
    marker.write_text(json.dumps(record))
    finalized_path = path / "timing_finalized.json"
    finalized = runner._load(finalized_path)
    finalized["attempt_sha256"] = runner._sha(marker)
    finalized_path.write_text(json.dumps(finalized))
    _, timing = _attempt(registration, 2, "PASS")
    with pytest.raises(runner.RegistrationError, match="inconsistent timing history verdict"):
        runner.require_timing_gate(timing, registration)


def test_runtime_validator_checks_fixed_parent_policy_before_source_admission(monkeypatch):
    from research.t34_pathb import source_binding_v2
    from tests.test_t34_retention_fixes4 import POLICY

    monkeypatch.setattr(source_binding_v2, "verify_frozen_worktree", lambda *_: None)
    registration = {
        "schema": "t34-pathb-v2-registration/v1",
        "ratified": True,
        "source_commit": "a" * 40,
        "arms": list(runner.ARMS),
        "mac_arms": list(runner.MAC_ARMS),
        "unpaced_colab": runner.unpaced_colab_policy(),
        "paced_fps": 24,
        "terminal_per_mix": 48,
        "product_sessions": 32,
        "horizon": 5000,
        "workers": 1,
        "load_profile": POLICY,
        "max_elapsed_hours": 40,
        "max_cpu_hours": 60,
        "precheck_max_elapsed_hours": 8,
        "timing_max_elapsed_hours": 1,
    }
    with pytest.raises(runner.RegistrationError, match="fixed study_root"):
        runner.require_registration(registration)


def test_unfinalized_retry_blocks_next_attempt_until_source_fence_closes(tmp_path):
    registration = _registration(tmp_path)
    path, _ = _attempt(registration, 1, "INCONCLUSIVE-retry")
    finalized = path / "timing_finalized.json"
    saved = finalized.read_bytes()
    finalized.unlink()
    with pytest.raises(runner.RegistrationError, match="incomplete timing history"):
        runner._reserve_timing_attempt(path.with_name("attempt-02"), registration)
    finalized.write_bytes(saved)
    runner._reserve_timing_attempt(path.with_name("attempt-02"), registration)


def test_duplicate_invocation_cannot_finalize_another_pending_attempt(monkeypatch, tmp_path):
    registration, _, _ = _fake_gate(monkeypatch, tmp_path)
    path, _ = _attempt(registration, 1, "INCONCLUSIVE-retry")
    (path / "timing_finalized.json").unlink()
    with pytest.raises(runner.RegistrationError, match="fresh"):
        runner.run_timing_gate(path, registration, registration_path=Path(__file__))
    assert not (path / "timing_finalized.json").exists()


@pytest.mark.parametrize("source_fails", [False, True])
def test_retry_remains_pending_until_enclosing_source_fence_finishes(
    monkeypatch, tmp_path, source_fails
):
    from research.t34_pathb import source_binding_v2

    registration, _, _ = _fake_gate(monkeypatch, tmp_path)
    registration["ratified"] = True
    events = []

    class Fence:
        def start(self):
            events.append("start")

        def check(self):
            pass

        def final_check(self):
            if source_fails:
                raise source_binding_v2.SourceBindingError("source changed during timing")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(source_binding_v2, "FrozenWorktreeGuard", lambda *_: Fence())
    timing = _timing()
    timing["probes"]["paced-24"]["contended"] = True
    monkeypatch.setattr(runner.timing_v2, "run_probes", lambda *_a, **_k: timing)
    out = tmp_path / "timing" / "attempt-01"
    write = runner._write_new

    def writing(path, value):
        if path.name == "timing_finalized.json":
            assert events[-1] == "closed"
        write(path, value)
        if path.name == "timing_attempt.json":
            assert "closed" not in events
            with pytest.raises(runner.RegistrationError, match="incomplete timing history"):
                runner._reserve_timing_attempt(out.with_name("attempt-02"), registration)

    monkeypatch.setattr(runner, "_write_new", writing)
    if source_fails:
        with pytest.raises(source_binding_v2.SourceBindingError):
            runner.run_timing_gate(out, registration, registration_path=Path(__file__))
        assert runner._load(out / "timing_finalized.json")["source_failed"] is True
        with pytest.raises(runner.RegistrationError, match="finalized source outcome"):
            runner._reserve_timing_attempt(out.with_name("attempt-02"), registration)
    else:
        assert (
            runner.run_timing_gate(out, registration, registration_path=Path(__file__))["status"]
            == "INCONCLUSIVE-retry"
        )
        assert runner._load(out / "timing_finalized.json")["source_failed"] is False
        runner._reserve_timing_attempt(out.with_name("attempt-02"), registration)


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
