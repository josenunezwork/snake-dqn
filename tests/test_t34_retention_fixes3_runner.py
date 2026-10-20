"""FIX-FIRST #3 admission refusal contracts for every retention phase."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from research.t34_pathb import host_guard_v2, retention_v2

TERMINAL_FILES = {
    "decision.json",
    "incomplete.json",
    "fault.json",
    "complete.json",
    "timing_gate.json",
    "precheck.json",
    "source_validated.json",
}


class _AdmissionRefusingGuard:
    records: list[dict] = []

    def admit(self) -> None:
        raise host_guard_v2.HostAdmissionError("host quiet admission refused: load_1m_excess")

    def start(self) -> None:
        pytest.fail("admission refusal started a phase")

    def close(self) -> None:
        pass


def _bypass_registration_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "require_timing_gate", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_require_committed_pushed_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)


@pytest.mark.parametrize(
    ("phase", "expected_cap_seconds"),
    [
        ("timing", 3600),
        ("precheck", 8 * 3600),
        ("study", 40 * 3600),
    ],
)
def test_admission_refusal_creates_no_terminal_artifacts_or_worlds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    expected_cap_seconds: int,
) -> None:
    """A busy host refuses before the phase can construct a world or result."""
    _bypass_registration_checks(monkeypatch)
    captured: list[dict] = []

    def make_guard(_policy: dict, **kwargs: object) -> _AdmissionRefusingGuard:
        captured.append(kwargs)
        return _AdmissionRefusingGuard()

    monkeypatch.setattr(retention_v2, "HostGuard", make_guard)
    out = tmp_path / phase
    registration = {"ratified": False, "load_profile": {}, "source_commit": "a" * 40}
    timing_gate = {"elapsed_seconds": 1.0, "cpu_seconds": 1.0, "status": "PASS"}

    if phase == "timing":
        registration.update({"timing_row": {}, "timing_ticks": 1, "timing_warmup": 0})
        adapter = SimpleNamespace(open_watch=lambda *_args: pytest.fail("world constructed"))

        def invoke():
            return retention_v2.run_timing_gate(
                out, registration, adapter=adapter, registration_path=Path(__file__)
            )

    elif phase == "precheck":

        def invoke():
            return retention_v2.run_precheck(
                [],
                open_terminal=lambda *_args: pytest.fail("world constructed"),
                timing_gate=timing_gate,
                registration=registration,
                registration_path=Path(__file__),
                out=out,
            )

    else:
        monkeypatch.setattr(retention_v2, "MIXES", ("mixed",))
        monkeypatch.setattr(retention_v2, "TERMINAL_PER_MIX", 1)
        monkeypatch.setattr(retention_v2, "PRODUCT_SESSIONS", 1)
        registration.update(
            {
                "terminal_rows": [{"seed": 1, "mix": "mixed"}],
                "product_rows": [{"seed": 2, "mix": "mixed"}],
            }
        )
        precheck = {
            "elapsed_seconds": 1.0,
            "cpu_seconds": 1.0,
            "timing_gate_sha256": retention_v2._canonical_sha(timing_gate),
            "registration_sha256": retention_v2._canonical_sha(registration),
        }
        adapter = SimpleNamespace(
            open_terminal=lambda *_args: pytest.fail("world constructed"),
            open_watch=lambda *_args: pytest.fail("world constructed"),
        )

        def invoke():
            return retention_v2.run_registered(
                out,
                registration,
                precheck,
                adapter=adapter,
                timing_gate=timing_gate,
                registration_path=Path(__file__),
            )

    with pytest.raises(host_guard_v2.HostAdmissionError, match="host quiet admission refused"):
        invoke()

    assert captured == [{"phase_cap_seconds": expected_cap_seconds}]
    assert not out.exists() or not ({path.name for path in out.iterdir()} & TERMINAL_FILES)
