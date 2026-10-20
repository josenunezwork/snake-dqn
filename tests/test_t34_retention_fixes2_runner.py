"""Regression contracts for the second T34 retention runner fixes.

These tests use injected sessions and guards only.  They do not start a live
service, a worker, a timing probe, or a gameplay row.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.t34_pathb import retention_v2


def test_precheck_has_an_independent_eight_hour_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The development pre-check cap is separate from the 40-hour study cap."""
    monkeypatch.setattr(retention_v2.time, "monotonic", lambda: 8 * 3600 + 1)
    monkeypatch.setattr(retention_v2.time, "process_time", lambda: 0.0)

    with pytest.raises(TimeoutError, match="8-hour pre-check"):
        retention_v2._precheck_cap(0.0, 0.0)

    # At the same clock reading, a study clock that started eight hours ago is
    # still below the registered 40-hour elapsed cap.
    retention_v2._cap(8 * 3600, 0.0)


def test_registered_phase_does_not_debit_precheck_from_forty_hour_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 8-hour pre-check receipt is provenance, not a 40-hour study debit."""
    captured_starts: list[tuple[float, float, bool]] = []
    analysis_seen: list[bool] = []
    monkeypatch.setattr(retention_v2, "require_timing_gate", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_require_committed_pushed_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    monkeypatch.setattr(retention_v2, "_final_source_check", lambda: None)
    monkeypatch.setattr(retention_v2, "MIXES", ("mixed",))
    monkeypatch.setattr(retention_v2, "TERMINAL_PER_MIX", 1)
    monkeypatch.setattr(retention_v2, "PRODUCT_SESSIONS", 1)

    def capture_cap(started: float, cpu_started: float) -> None:
        captured_starts.append((started, cpu_started, bool(analysis_seen)))

    monkeypatch.setattr(retention_v2, "_cap", capture_cap)

    class Guard:
        records: list[dict] = []

        def check_budget(self) -> None:
            pass

        def admit(self) -> None:
            pass

        def start(self) -> None:
            pass

        def check(self, **_kwargs: object) -> dict:
            return {"contended": False, "reasons": []}

        def between_rows(self, *, on_pause=None) -> None:
            del on_pause

        def summary(self) -> dict:
            return {}

        def close(self) -> None:
            pass

    monkeypatch.setattr(retention_v2, "HostGuard", lambda _policy, **_kwargs: Guard())
    monkeypatch.setattr(
        retention_v2,
        "_collect",
        lambda *_args, **_kwargs: {
            "identity_status": "PASS",
            "parity_status": "PASS",
            "controller": {},
        },
    )
    monkeypatch.setattr(
        retention_v2,
        "analyze_path",
        lambda _out: (analysis_seen.append(True) or {"verdict": "INCONCLUSIVE"}),
    )

    registration = {
        "ratified": False,
        "load_profile": {},
        "source_commit": "a" * 40,
        "terminal_rows": [{"seed": 1, "mix": "mixed"}],
        "product_rows": [{"seed": 2, "mix": "mixed"}],
    }
    timing_gate = {"elapsed_seconds": 1.0, "cpu_seconds": 2.0, "status": "PASS"}
    precheck = {
        "elapsed_seconds": 8 * 3600.0 - 1.0,
        "cpu_seconds": 7 * 3600.0,
        "timing_gate_sha256": retention_v2._canonical_sha(timing_gate),
        "registration_sha256": retention_v2._canonical_sha(registration),
    }
    session = SimpleNamespace(
        controller_state=lambda: {},
        close=lambda **_kwargs: None,
    )
    opened: list[tuple[int, str]] = []

    def open_session(_row: dict, fps: int, arm: str) -> SimpleNamespace:
        opened.append((fps, arm))
        return session

    adapter = SimpleNamespace(open_terminal=open_session, open_watch=open_session)

    result = retention_v2.run_registered(
        tmp_path / "run",
        registration,
        precheck,
        adapter=adapter,
        timing_gate=timing_gate,
        registration_path=Path(__file__),
    )

    assert result == {"verdict": "INCONCLUSIVE"}
    assert opened == [(24, "bare"), (24, "paced")] * 2
    record = json.loads((tmp_path / "run/records/terminal-000.json").read_text())
    assert set(record["arms"]) == {"bare", "paced"}
    assert captured_starts
    # run_registered subtracts the timing phase only.  If the pre-check were
    # charged against the study clock, this would be roughly eight hours old.
    assert captured_starts[0][0] > time.monotonic() - 60
    assert any(seen for _started, _cpu_started, seen in captured_starts)


def test_collect_uses_nonblocking_cached_host_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host sampling in the tick loop cannot sleep or pause a gameplay row."""

    class Session:
        hero_length, hero_alive, hero_death = 10, True, False
        decisions_delta, boost_decisions_delta = 1, 0

        def step(self) -> None:
            pass

        def controller_state(self) -> dict:
            return {}

    class Guard:
        calls = 0

        def check(self, **_kwargs: object) -> dict:
            self.calls += 1
            return {"contended": False, "reasons": []}

    guard = Guard()
    monkeypatch.setattr(
        retention_v2.time, "sleep", lambda _seconds: pytest.fail("tick collection slept")
    )
    result = retention_v2._collect(
        Session(),
        horizon=3,
        fps=0,
        started=time.monotonic(),
        cpu_started=time.process_time(),
        host_guard=guard,
    )

    assert guard.calls == 3
    assert result["host_contended_ticks"] == [False, False, False]


def test_frozen_source_close_failure_retains_an_inconclusive_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.t34_pathb import source_binding_v2

    class Fence:
        def start(self):
            return self

        def final_check(self) -> None:
            pass

        def close(self) -> None:
            raise source_binding_v2.SourceBindingError("HEAD moved during analysis")

    monkeypatch.setattr(source_binding_v2, "FrozenWorktreeGuard", lambda *_args, **_kwargs: Fence())

    @retention_v2._frozen_execution
    def phase(out: Path, registration: dict) -> str:
        out.mkdir()
        return "completed"

    with pytest.raises(source_binding_v2.SourceBindingError, match="HEAD moved"):
        phase(
            tmp_path / "run",
            {"ratified": True, "source_commit": "a" * 40},
        )
    marker = tmp_path / "run" / "source_failure.json"
    assert marker.is_file()
    assert marker.read_text().find('"invalidates_decision": true') >= 0


def test_restored_pass_receipt_cannot_bypass_source_failure_marker(tmp_path: Path) -> None:
    """A late fence failure invalidates a restored PASS receipt beside its certificate."""
    out = tmp_path / "run"
    out.mkdir()
    certificate_path = out / "source_validated.json"
    receipt = {
        "status": "PASS",
        "source_commit": "a" * 40,
        "source_validation_path": str(certificate_path.resolve()),
    }
    certificate = {
        "schema": "t34-pathb-v2-source-validation/v1",
        "status": "PASS",
        "source_commit": receipt["source_commit"],
        "receipts": {"timing_gate.json": retention_v2._canonical_sha(receipt)},
    }
    certificate_path.write_text(json.dumps(certificate))
    # A caller restored the original PASS JSON, but the fence's durable marker remains.
    (out / "source_failure.json").write_text(
        json.dumps({"status": "INCONCLUSIVE", "invalidates_decision": True})
    )

    with pytest.raises(retention_v2.RegistrationError, match="invalidated"):
        retention_v2._require_phase_source(receipt, "timing_gate.json")


@pytest.mark.parametrize("receipt_name", ["timing_gate.json", "precheck.json"])
def test_phase_source_requires_certificate_for_each_admission_receipt(
    tmp_path: Path, receipt_name: str
) -> None:
    """Both pre-data receipts require a final source certificate and exact hash binding."""
    receipt = {"status": "PASS", "source_commit": "b" * 40}
    with pytest.raises(retention_v2.RegistrationError, match="lacks"):
        retention_v2._require_phase_source(receipt, receipt_name)

    certificate_path = tmp_path / f"{receipt_name}.source_validated.json"
    receipt["source_validation_path"] = str(certificate_path.resolve())
    certificate = {
        "schema": "t34-pathb-v2-source-validation/v1",
        "status": "PASS",
        "source_commit": receipt["source_commit"],
        "receipts": {receipt_name: retention_v2._canonical_sha(receipt)},
    }
    certificate_path.write_text(json.dumps(certificate))
    retention_v2._require_phase_source(receipt, receipt_name)

    certificate["receipts"][receipt_name] = "0" * 64
    certificate_path.write_text(json.dumps(certificate))
    with pytest.raises(retention_v2.RegistrationError, match="does not bind"):
        retention_v2._require_phase_source(receipt, receipt_name)
